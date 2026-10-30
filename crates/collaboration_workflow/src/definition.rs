use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use yaml_rust2::scanner::{Scanner, TokenType};

pub const CURRENT_DEFINITION_VERSION: u32 = 1;
pub const MAX_DEFINITION_BYTES: usize = 64 * 1024;
pub const MAX_YAML_DEPTH: usize = 16;
pub const MAX_YAML_NODES: usize = 2_048;
pub const MAX_WORKFLOW_STEPS: usize = 64;
pub const MAX_CONDITION_BYTES: usize = 4_096;
pub const MAX_STEP_TIMEOUT_SECS: u64 = 600;
pub const MAX_DELAY_SECS: u64 = 270;
pub const MAX_RETRY_ATTEMPTS: u16 = 8;
pub const MAX_RETRY_ELAPSED_SECS: u64 = 3_600;
pub const MAX_RETRY_BACKOFF_SECS: u64 = 300;

const MAX_NAME_BYTES: usize = 256;
const MAX_DESCRIPTION_BYTES: usize = 4_096;
const MAX_STEP_ID_BYTES: usize = 64;
const MAX_SECRET_REFERENCES: usize = 32;
const MAX_SECRET_NAME_BYTES: usize = 64;
const MAX_SECRET_REFERENCE_BYTES: usize = 256;
const MAX_TEMPLATE_BYTES: usize = 16 * 1024;
const MAX_TEMPLATE_EXPRESSIONS: usize = 64;
const MAX_HEADER_COUNT: usize = 32;
const MAX_HEADER_NAME_BYTES: usize = 128;
const MAX_HEADER_VALUE_BYTES: usize = 8 * 1024;
const MAX_TOTAL_HEADER_BYTES: usize = 16 * 1024;
const MAX_URL_BYTES: usize = 2_048;
const MAX_APPROVAL_WAIT_SECS: u64 = 30 * 24 * 60 * 60;
const MAX_SCHEDULE_INTERVAL_SECS: u64 = 366 * 24 * 60 * 60;
const MAX_ERROR_DETAIL_BYTES: usize = 512;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DefinitionError {
    #[error("workflow definition exceeds the encoded byte limit")]
    DefinitionTooLarge,
    #[error("workflow definition exceeds the YAML node limit")]
    TooManyYamlNodes,
    #[error("workflow definition exceeds the YAML depth limit")]
    YamlTooDeep,
    #[error("YAML anchors and aliases are not supported in workflow definitions")]
    YamlAliasNotSupported,
    #[error("YAML tags and directives are not supported in workflow definitions")]
    YamlTagNotSupported,
    #[error("workflow definitions must contain exactly one YAML document")]
    MultipleYamlDocuments,
    #[error("invalid workflow YAML: {detail}")]
    InvalidYaml { detail: String },
    #[error("workflow definition version {version} is not supported")]
    UnsupportedVersion { version: u32 },
    #[error("workflow trigger is not supported")]
    UnsupportedTrigger,
    #[error("workflow action is not supported")]
    UnsupportedAction,
    #[error("workflow retry mode is not supported")]
    UnsupportedRetryMode,
    #[error("workflow retry failure class is not supported")]
    UnsupportedRetryClass,
    #[error("literal secrets are prohibited at {path}")]
    SecretLiteral { path: &'static str },
    #[error("invalid workflow definition at {path}: {rule}")]
    InvalidField {
        path: &'static str,
        rule: &'static str,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WorkflowDefinition {
    version: u32,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    enabled: bool,
    trigger: WorkflowTrigger,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    secrets: BTreeMap<String, SecretReference>,
    retry: RetryPolicy,
    steps: Vec<WorkflowStep>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalWorkflowDefinition {
    version: u32,
    name: String,
    #[serde(default)]
    description: Option<String>,
    enabled: bool,
    trigger: WorkflowTrigger,
    #[serde(default)]
    secrets: BTreeMap<String, SecretReference>,
    retry: RetryPolicy,
    steps: Vec<WorkflowStep>,
}

impl<'de> Deserialize<'de> for WorkflowDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let canonical = CanonicalWorkflowDefinition::deserialize(deserializer)?;
        let definition = Self {
            version: canonical.version,
            name: canonical.name,
            description: canonical.description,
            enabled: canonical.enabled,
            trigger: canonical.trigger,
            secrets: canonical.secrets,
            retry: canonical.retry,
            steps: canonical.steps,
        };
        validate_canonical_definition(&definition).map_err(serde::de::Error::custom)?;
        Ok(definition)
    }
}

impl WorkflowDefinition {
    pub fn parse_yaml(yaml: &str) -> Result<Self, DefinitionError> {
        validate_yaml_shape(yaml)?;
        let raw: RawWorkflowDefinition =
            serde_yaml_ng::from_str(yaml).map_err(|error| DefinitionError::InvalidYaml {
                detail: bounded_error_detail(&error.to_string()),
            })?;
        Self::try_from(raw)
    }

    pub fn parse_canonical_json(json: &str) -> Result<Self, DefinitionError> {
        validate_yaml_shape(json)?;
        let definition: Self =
            serde_yaml_ng::from_str(json).map_err(|error| DefinitionError::InvalidYaml {
                detail: bounded_error_detail(&error.to_string()),
            })?;
        Ok(definition)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn trigger(&self) -> &WorkflowTrigger {
        &self.trigger
    }

    pub fn secrets(&self) -> &BTreeMap<String, SecretReference> {
        &self.secrets
    }

    pub fn retry(&self) -> &RetryPolicy {
        &self.retry
    }

    pub fn steps(&self) -> &[WorkflowStep] {
        &self.steps
    }
}

pub fn parse_yaml(yaml: &str) -> Result<WorkflowDefinition, DefinitionError> {
    WorkflowDefinition::parse_yaml(yaml)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "on", rename_all = "snake_case")]
pub enum WorkflowTrigger {
    MessagePosted {
        #[serde(rename = "if", skip_serializing_if = "Option::is_none")]
        condition: Option<ConditionExpression>,
    },
    ReactionAdded {
        #[serde(skip_serializing_if = "Option::is_none")]
        emoji: Option<String>,
    },
    DiffPosted {
        #[serde(rename = "if", skip_serializing_if = "Option::is_none")]
        condition: Option<ConditionExpression>,
    },
    Schedule {
        schedule: Schedule,
    },
    Webhook,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Schedule {
    Cron(String),
    IntervalSeconds(u64),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ConditionExpression(String);

impl ConditionExpression {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SecretReference {
    credential: String,
}

impl SecretReference {
    pub fn credential(&self) -> &str {
        &self.credential
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RetryPolicy {
    Never,
    Exponential {
        max_attempts: u16,
        max_elapsed_secs: u64,
        initial_backoff_secs: u64,
        max_backoff_secs: u64,
        jitter: RetryJitter,
        retry_on: BTreeSet<RetryFailureClass>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryJitter {
    Full,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryFailureClass {
    RateLimited,
    TemporaryUnavailable,
    Timeout,
    Transport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkflowStep {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(rename = "if", skip_serializing_if = "Option::is_none")]
    condition: Option<ConditionExpression>,
    timeout_secs: u64,
    #[serde(flatten)]
    action: StepAction,
}

impl WorkflowStep {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn condition(&self) -> Option<&ConditionExpression> {
        self.condition.as_ref()
    }

    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    pub fn action(&self) -> &StepAction {
        &self.action
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum StepAction {
    SendMessage {
        text: TemplateString,
        #[serde(skip_serializing_if = "Option::is_none")]
        channel: Option<TemplateString>,
    },
    SendDm {
        to: TemplateString,
        text: TemplateString,
    },
    SetChannelTopic {
        topic: TemplateString,
    },
    AddReaction {
        emoji: TemplateString,
    },
    CallWebhook {
        url: String,
        method: WebhookMethod,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, ActionValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        body: Option<TemplateString>,
    },
    RequestApproval {
        from: String,
        message: TemplateString,
        #[serde(rename = "approval_timeout_secs")]
        timeout_secs: u64,
    },
    Delay {
        duration_secs: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct TemplateString(String);

impl TemplateString {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ActionValue {
    Literal(TemplateString),
    Secret { secret_ref: String },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum WebhookMethod {
    Post,
    Put,
    Patch,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkflowDefinition {
    version: u32,
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_true")]
    enabled: bool,
    trigger: RawTrigger,
    #[serde(default)]
    secrets: BTreeMap<String, RawSecretReference>,
    #[serde(default)]
    retry: Option<RawRetryPolicy>,
    steps: Vec<RawStep>,
    #[serde(default, rename = "_webhook_secret")]
    legacy_webhook_secret: Option<serde_yaml_ng::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTrigger {
    on: String,
    #[serde(default, rename = "if", alias = "filter")]
    condition: Option<String>,
    #[serde(default)]
    emoji: Option<String>,
    #[serde(default)]
    cron: Option<String>,
    #[serde(default)]
    interval: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawSecretReference {
    Reference(RawCredentialReference),
    Literal(serde_yaml_ng::Value),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCredentialReference {
    credential: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRetryPolicy {
    mode: String,
    #[serde(default)]
    max_attempts: Option<u16>,
    #[serde(default)]
    max_elapsed_secs: Option<u64>,
    #[serde(default)]
    initial_backoff_secs: Option<u64>,
    #[serde(default)]
    max_backoff_secs: Option<u64>,
    #[serde(default)]
    jitter: Option<String>,
    #[serde(default)]
    retry_on: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStep {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "if")]
    condition: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    action: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    to: Option<String>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    emoji: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    headers: Option<BTreeMap<String, RawActionValue>>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    timeout: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawActionValue {
    Literal(String),
    Secret(RawActionSecretReference),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawActionSecretReference {
    secret_ref: String,
}

fn default_true() -> bool {
    true
}

impl TryFrom<RawWorkflowDefinition> for WorkflowDefinition {
    type Error = DefinitionError;

    fn try_from(raw: RawWorkflowDefinition) -> Result<Self, Self::Error> {
        if raw.version != CURRENT_DEFINITION_VERSION {
            return Err(DefinitionError::UnsupportedVersion {
                version: raw.version,
            });
        }
        if raw.legacy_webhook_secret.is_some() {
            return Err(DefinitionError::SecretLiteral {
                path: "_webhook_secret",
            });
        }
        validate_required_string(&raw.name, MAX_NAME_BYTES, "name")?;
        if let Some(description) = raw.description.as_deref() {
            validate_optional_string(description, MAX_DESCRIPTION_BYTES, "description")?;
        }
        if raw.steps.is_empty() || raw.steps.len() > MAX_WORKFLOW_STEPS {
            return Err(invalid("steps", "must contain between 1 and 64 entries"));
        }

        let secrets = validate_secrets(raw.secrets)?;
        let trigger = validate_trigger(raw.trigger)?;
        let retry = validate_retry(raw.retry)?;
        let mut identifiers = BTreeSet::new();
        let mut steps = Vec::with_capacity(raw.steps.len());
        for step in raw.steps {
            if !valid_identifier(&step.id, MAX_STEP_ID_BYTES) {
                return Err(invalid(
                    "steps[].id",
                    "must be a unique ASCII identifier of at most 64 bytes",
                ));
            }
            if !identifiers.insert(step.id.clone()) {
                return Err(invalid("steps[].id", "must be unique"));
            }
            steps.push(validate_step(step, &secrets)?);
        }

        Ok(Self {
            version: raw.version,
            name: raw.name.trim().to_owned(),
            description: raw.description.map(|value| value.trim().to_owned()),
            enabled: raw.enabled,
            trigger,
            secrets,
            retry,
            steps,
        })
    }
}

fn validate_canonical_definition(definition: &WorkflowDefinition) -> Result<(), DefinitionError> {
    if definition.version != CURRENT_DEFINITION_VERSION {
        return Err(DefinitionError::UnsupportedVersion {
            version: definition.version,
        });
    }
    validate_required_string(&definition.name, MAX_NAME_BYTES, "name")?;
    if let Some(description) = definition.description.as_deref() {
        validate_optional_string(description, MAX_DESCRIPTION_BYTES, "description")?;
    }
    if definition.secrets.len() > MAX_SECRET_REFERENCES {
        return Err(invalid("secrets", "contains too many references"));
    }
    for (name, reference) in &definition.secrets {
        if !valid_identifier(name, MAX_SECRET_NAME_BYTES) {
            return Err(invalid(
                "secrets.<name>",
                "must be an ASCII identifier of at most 64 bytes",
            ));
        }
        validate_required_string(
            &reference.credential,
            MAX_SECRET_REFERENCE_BYTES,
            "secrets.<name>.credential",
        )?;
        if reference.credential.chars().any(char::is_control) {
            return Err(invalid(
                "secrets.<name>.credential",
                "must not contain control characters",
            ));
        }
    }
    match &definition.trigger {
        WorkflowTrigger::MessagePosted { condition }
        | WorkflowTrigger::DiffPosted { condition } => {
            if let Some(condition) = condition {
                validate_condition(condition.as_str())?;
            }
        }
        WorkflowTrigger::ReactionAdded { emoji } => {
            if let Some(emoji) = emoji {
                validate_required_string(emoji, 128, "trigger.emoji")?;
            }
        }
        WorkflowTrigger::Schedule { schedule } => match schedule {
            Schedule::Cron(cron) => validate_cron(cron)?,
            Schedule::IntervalSeconds(seconds)
                if (60..=MAX_SCHEDULE_INTERVAL_SECS).contains(seconds) => {}
            Schedule::IntervalSeconds(_) => {
                return Err(invalid(
                    "trigger.interval",
                    "must be between 60 seconds and 366 days",
                ));
            }
        },
        WorkflowTrigger::Webhook => {}
    }
    match &definition.retry {
        RetryPolicy::Never => {}
        RetryPolicy::Exponential {
            max_attempts,
            max_elapsed_secs,
            initial_backoff_secs,
            max_backoff_secs,
            jitter: RetryJitter::Full,
            retry_on,
        } => {
            if !(2..=MAX_RETRY_ATTEMPTS).contains(max_attempts)
                || *max_elapsed_secs == 0
                || *max_elapsed_secs > MAX_RETRY_ELAPSED_SECS
                || *initial_backoff_secs == 0
                || initial_backoff_secs > max_backoff_secs
                || *max_backoff_secs > MAX_RETRY_BACKOFF_SECS
                || max_backoff_secs > max_elapsed_secs
                || retry_on.is_empty()
            {
                return Err(invalid("retry", "canonical retry policy is invalid"));
            }
        }
    }
    if definition.steps.is_empty() || definition.steps.len() > MAX_WORKFLOW_STEPS {
        return Err(invalid("steps", "must contain between 1 and 64 entries"));
    }
    let mut identifiers = BTreeSet::new();
    for step in &definition.steps {
        if !valid_identifier(&step.id, MAX_STEP_ID_BYTES) || !identifiers.insert(step.id.as_str()) {
            return Err(invalid(
                "steps[].id",
                "must be a unique ASCII identifier of at most 64 bytes",
            ));
        }
        if let Some(name) = step.name.as_deref() {
            validate_optional_string(name, MAX_NAME_BYTES, "steps[].name")?;
        }
        if let Some(condition) = &step.condition {
            validate_condition(condition.as_str())?;
        }
        if step.timeout_secs == 0 || step.timeout_secs > MAX_STEP_TIMEOUT_SECS {
            return Err(invalid("steps[].timeout_secs", "must be between 1 and 600"));
        }
        validate_canonical_action(&step.action, &definition.secrets)?;
    }
    Ok(())
}

fn validate_canonical_action(
    action: &StepAction,
    secrets: &BTreeMap<String, SecretReference>,
) -> Result<(), DefinitionError> {
    match action {
        StepAction::SendMessage { text, channel } => {
            validate_template(text.as_str(), secrets, true, "steps[].text")?;
            if let Some(channel) = channel {
                validate_template(channel.as_str(), secrets, false, "steps[].channel")?;
            }
        }
        StepAction::SendDm { to, text } => {
            validate_template(to.as_str(), secrets, false, "steps[].to")?;
            validate_template(text.as_str(), secrets, true, "steps[].text")?;
        }
        StepAction::SetChannelTopic { topic } => {
            validate_template(topic.as_str(), secrets, false, "steps[].topic")?;
        }
        StepAction::AddReaction { emoji } => {
            validate_template(emoji.as_str(), secrets, false, "steps[].emoji")?;
        }
        StepAction::CallWebhook {
            url,
            method: _,
            headers,
            body,
        } => {
            validate_required_string(url, MAX_URL_BYTES, "steps[].url")?;
            if url.contains("{{") || contains_secret_word(url) {
                return Err(invalid(
                    "steps[].url",
                    "must be a fixed non-secret destination",
                ));
            }
            if headers.len() > MAX_HEADER_COUNT {
                return Err(invalid("steps[].headers", "contains too many entries"));
            }
            let mut total_bytes = 0_usize;
            for (name, value) in headers {
                if name.is_empty()
                    || name.len() > MAX_HEADER_NAME_BYTES
                    || !name.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
                    })
                    || forbidden_header(name)
                {
                    return Err(invalid(
                        "steps[].headers.<name>",
                        "is not an allowed outbound header name",
                    ));
                }
                total_bytes = total_bytes.saturating_add(name.len());
                match value {
                    ActionValue::Literal(value) => {
                        if sensitive_header(name) {
                            return Err(DefinitionError::SecretLiteral {
                                path: "steps[].headers.<value>",
                            });
                        }
                        total_bytes = total_bytes.saturating_add(value.as_str().len());
                        if value.as_str().len() > MAX_HEADER_VALUE_BYTES {
                            return Err(invalid(
                                "steps[].headers.<value>",
                                "exceeds the header value byte limit",
                            ));
                        }
                        validate_template(
                            value.as_str(),
                            secrets,
                            true,
                            "steps[].headers.<value>",
                        )?;
                    }
                    ActionValue::Secret { secret_ref } => {
                        if !secrets.contains_key(secret_ref) {
                            return Err(invalid(
                                "steps[].headers.<value>.secret_ref",
                                "must name a declared secret reference",
                            ));
                        }
                        total_bytes = total_bytes.saturating_add(secret_ref.len());
                    }
                }
                if total_bytes > MAX_TOTAL_HEADER_BYTES {
                    return Err(invalid(
                        "steps[].headers",
                        "exceeds the aggregate header byte limit",
                    ));
                }
            }
            if let Some(body) = body {
                reject_literal_secret_assignments(body.as_str(), "steps[].body")?;
                validate_template(body.as_str(), secrets, true, "steps[].body")?;
            }
        }
        StepAction::RequestApproval {
            from,
            message,
            timeout_secs,
        } => {
            validate_required_string(from, 256, "steps[].from")?;
            validate_template(message.as_str(), secrets, false, "steps[].message")?;
            if *timeout_secs == 0 || *timeout_secs > MAX_APPROVAL_WAIT_SECS {
                return Err(invalid(
                    "steps[].timeout",
                    "must be between 1 second and 30 days",
                ));
            }
        }
        StepAction::Delay { duration_secs } if (1..=MAX_DELAY_SECS).contains(duration_secs) => {}
        StepAction::Delay { .. } => {
            return Err(invalid(
                "steps[].duration",
                "must be between 1 and 270 seconds",
            ));
        }
    }
    Ok(())
}

fn validate_yaml_shape(yaml: &str) -> Result<(), DefinitionError> {
    if yaml.len() > MAX_DEFINITION_BYTES {
        return Err(DefinitionError::DefinitionTooLarge);
    }
    let mut depth = 0_usize;
    let mut nodes = 0_usize;
    let mut explicit_documents = 0_usize;
    let mut scanner = Scanner::new(yaml.chars());
    for token in scanner.by_ref() {
        match token.1 {
            TokenType::Alias(_) | TokenType::Anchor(_) => {
                return Err(DefinitionError::YamlAliasNotSupported);
            }
            TokenType::Tag(_, _)
            | TokenType::TagDirective(_, _)
            | TokenType::VersionDirective(_, _) => {
                return Err(DefinitionError::YamlTagNotSupported);
            }
            TokenType::DocumentStart => {
                explicit_documents += 1;
                if explicit_documents > 1 {
                    return Err(DefinitionError::MultipleYamlDocuments);
                }
            }
            TokenType::BlockSequenceStart
            | TokenType::BlockMappingStart
            | TokenType::FlowSequenceStart
            | TokenType::FlowMappingStart => {
                nodes += 1;
                depth += 1;
                if nodes > MAX_YAML_NODES {
                    return Err(DefinitionError::TooManyYamlNodes);
                }
                if depth > MAX_YAML_DEPTH {
                    return Err(DefinitionError::YamlTooDeep);
                }
            }
            TokenType::BlockEnd | TokenType::FlowSequenceEnd | TokenType::FlowMappingEnd => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| DefinitionError::InvalidYaml {
                        detail: "unbalanced YAML collection".to_owned(),
                    })?;
            }
            TokenType::Scalar(_, _) => {
                nodes += 1;
                if nodes > MAX_YAML_NODES {
                    return Err(DefinitionError::TooManyYamlNodes);
                }
            }
            _ => {}
        }
    }
    if let Some(error) = scanner.get_error() {
        return Err(DefinitionError::InvalidYaml {
            detail: bounded_error_detail(&error.to_string()),
        });
    }
    Ok(())
}

fn validate_secrets(
    raw: BTreeMap<String, RawSecretReference>,
) -> Result<BTreeMap<String, SecretReference>, DefinitionError> {
    if raw.len() > MAX_SECRET_REFERENCES {
        return Err(invalid("secrets", "contains too many references"));
    }
    raw.into_iter()
        .map(|(name, reference)| {
            if !valid_identifier(&name, MAX_SECRET_NAME_BYTES) {
                return Err(invalid(
                    "secrets.<name>",
                    "must be an ASCII identifier of at most 64 bytes",
                ));
            }
            let reference = match reference {
                RawSecretReference::Reference(reference) => reference,
                RawSecretReference::Literal(literal) => {
                    drop(literal);
                    return Err(DefinitionError::SecretLiteral {
                        path: "secrets.<name>",
                    });
                }
            };
            validate_required_string(
                &reference.credential,
                MAX_SECRET_REFERENCE_BYTES,
                "secrets.<name>.credential",
            )?;
            if reference.credential.chars().any(char::is_control) {
                return Err(invalid(
                    "secrets.<name>.credential",
                    "must not contain control characters",
                ));
            }
            Ok((
                name,
                SecretReference {
                    credential: reference.credential,
                },
            ))
        })
        .collect()
}

fn validate_trigger(raw: RawTrigger) -> Result<WorkflowTrigger, DefinitionError> {
    match raw.on.as_str() {
        "message_posted" | "diff_posted" => {
            reject_present(&[
                ("trigger.emoji", raw.emoji.is_some()),
                ("trigger.cron", raw.cron.is_some()),
                ("trigger.interval", raw.interval.is_some()),
            ])?;
            let condition = raw
                .condition
                .as_deref()
                .map(validate_condition)
                .transpose()?;
            if raw.on == "message_posted" {
                Ok(WorkflowTrigger::MessagePosted { condition })
            } else {
                Ok(WorkflowTrigger::DiffPosted { condition })
            }
        }
        "reaction_added" => {
            reject_present(&[
                ("trigger.if", raw.condition.is_some()),
                ("trigger.cron", raw.cron.is_some()),
                ("trigger.interval", raw.interval.is_some()),
            ])?;
            if let Some(emoji) = raw.emoji.as_deref() {
                validate_required_string(emoji, 128, "trigger.emoji")?;
            }
            Ok(WorkflowTrigger::ReactionAdded { emoji: raw.emoji })
        }
        "schedule" => {
            reject_present(&[
                ("trigger.if", raw.condition.is_some()),
                ("trigger.emoji", raw.emoji.is_some()),
            ])?;
            let schedule = match (raw.cron, raw.interval) {
                (Some(cron), None) => {
                    validate_cron(&cron)?;
                    Schedule::Cron(cron)
                }
                (None, Some(interval)) => {
                    let seconds = parse_duration_secs(&interval, "trigger.interval")?;
                    if !(60..=MAX_SCHEDULE_INTERVAL_SECS).contains(&seconds) {
                        return Err(invalid(
                            "trigger.interval",
                            "must be between 60 seconds and 366 days",
                        ));
                    }
                    Schedule::IntervalSeconds(seconds)
                }
                _ => {
                    return Err(invalid(
                        "trigger",
                        "schedule requires exactly one of cron or interval",
                    ));
                }
            };
            Ok(WorkflowTrigger::Schedule { schedule })
        }
        "webhook" => {
            reject_present(&[
                ("trigger.if", raw.condition.is_some()),
                ("trigger.emoji", raw.emoji.is_some()),
                ("trigger.cron", raw.cron.is_some()),
                ("trigger.interval", raw.interval.is_some()),
            ])?;
            Ok(WorkflowTrigger::Webhook)
        }
        _ => Err(DefinitionError::UnsupportedTrigger),
    }
}

fn validate_retry(raw: Option<RawRetryPolicy>) -> Result<RetryPolicy, DefinitionError> {
    let Some(raw) = raw else {
        return Ok(RetryPolicy::Never);
    };
    match raw.mode.as_str() {
        "never" => {
            if raw.max_attempts.is_some()
                || raw.max_elapsed_secs.is_some()
                || raw.initial_backoff_secs.is_some()
                || raw.max_backoff_secs.is_some()
                || raw.jitter.is_some()
                || raw.retry_on.is_some()
            {
                return Err(invalid(
                    "retry",
                    "never mode cannot declare retry parameters",
                ));
            }
            Ok(RetryPolicy::Never)
        }
        "exponential" => {
            let max_attempts = required(raw.max_attempts, "retry.max_attempts")?;
            let max_elapsed_secs = required(raw.max_elapsed_secs, "retry.max_elapsed_secs")?;
            let initial_backoff_secs =
                required(raw.initial_backoff_secs, "retry.initial_backoff_secs")?;
            let max_backoff_secs = required(raw.max_backoff_secs, "retry.max_backoff_secs")?;
            if !(2..=MAX_RETRY_ATTEMPTS).contains(&max_attempts) {
                return Err(invalid("retry.max_attempts", "must be between 2 and 8"));
            }
            if max_elapsed_secs == 0 || max_elapsed_secs > MAX_RETRY_ELAPSED_SECS {
                return Err(invalid(
                    "retry.max_elapsed_secs",
                    "must be between 1 and 3600",
                ));
            }
            if initial_backoff_secs == 0
                || initial_backoff_secs > max_backoff_secs
                || max_backoff_secs > MAX_RETRY_BACKOFF_SECS
                || max_backoff_secs > max_elapsed_secs
            {
                return Err(invalid(
                    "retry",
                    "backoff must be positive, ordered and within elapsed bounds",
                ));
            }
            let jitter = match raw.jitter.as_deref() {
                Some("full") => RetryJitter::Full,
                _ => {
                    return Err(invalid(
                        "retry.jitter",
                        "exponential retry requires full jitter",
                    ));
                }
            };
            let retry_on = raw
                .retry_on
                .ok_or_else(|| invalid("retry.retry_on", "must not be empty"))?
                .into_iter()
                .map(|class| match class.as_str() {
                    "rate_limited" => Ok(RetryFailureClass::RateLimited),
                    "temporary_unavailable" => Ok(RetryFailureClass::TemporaryUnavailable),
                    "timeout" => Ok(RetryFailureClass::Timeout),
                    "transport" => Ok(RetryFailureClass::Transport),
                    _ => Err(DefinitionError::UnsupportedRetryClass),
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            if retry_on.is_empty() {
                return Err(invalid("retry.retry_on", "must not be empty"));
            }
            Ok(RetryPolicy::Exponential {
                max_attempts,
                max_elapsed_secs,
                initial_backoff_secs,
                max_backoff_secs,
                jitter,
                retry_on,
            })
        }
        _ => Err(DefinitionError::UnsupportedRetryMode),
    }
}

fn validate_step(
    raw: RawStep,
    secrets: &BTreeMap<String, SecretReference>,
) -> Result<WorkflowStep, DefinitionError> {
    if let Some(name) = raw.name.as_deref() {
        validate_optional_string(name, MAX_NAME_BYTES, "steps[].name")?;
    }
    let condition = raw
        .condition
        .as_deref()
        .map(validate_condition)
        .transpose()?;
    let timeout_secs = raw.timeout_secs.unwrap_or(300);
    if timeout_secs == 0 || timeout_secs > MAX_STEP_TIMEOUT_SECS {
        return Err(invalid("steps[].timeout_secs", "must be between 1 and 600"));
    }
    let action = validate_action(&raw, secrets)?;
    Ok(WorkflowStep {
        id: raw.id,
        name: raw.name,
        condition,
        timeout_secs,
        action,
    })
}

fn validate_action(
    raw: &RawStep,
    secrets: &BTreeMap<String, SecretReference>,
) -> Result<StepAction, DefinitionError> {
    match raw.action.as_str() {
        "send_message" => {
            reject_action_fields(raw, &["text", "channel"])?;
            Ok(StepAction::SendMessage {
                text: validate_template(
                    required_ref(raw.text.as_ref(), "steps[].text")?,
                    secrets,
                    true,
                    "steps[].text",
                )?,
                channel: raw
                    .channel
                    .as_ref()
                    .map(|value| validate_template(value, secrets, false, "steps[].channel"))
                    .transpose()?,
            })
        }
        "send_dm" => {
            reject_action_fields(raw, &["to", "text"])?;
            Ok(StepAction::SendDm {
                to: validate_template(
                    required_ref(raw.to.as_ref(), "steps[].to")?,
                    secrets,
                    false,
                    "steps[].to",
                )?,
                text: validate_template(
                    required_ref(raw.text.as_ref(), "steps[].text")?,
                    secrets,
                    true,
                    "steps[].text",
                )?,
            })
        }
        "set_channel_topic" => {
            reject_action_fields(raw, &["topic"])?;
            Ok(StepAction::SetChannelTopic {
                topic: validate_template(
                    required_ref(raw.topic.as_ref(), "steps[].topic")?,
                    secrets,
                    false,
                    "steps[].topic",
                )?,
            })
        }
        "add_reaction" => {
            reject_action_fields(raw, &["emoji"])?;
            Ok(StepAction::AddReaction {
                emoji: validate_template(
                    required_ref(raw.emoji.as_ref(), "steps[].emoji")?,
                    secrets,
                    false,
                    "steps[].emoji",
                )?,
            })
        }
        "call_webhook" => {
            reject_action_fields(raw, &["url", "method", "headers", "body"])?;
            let url = required_ref(raw.url.as_ref(), "steps[].url")?;
            validate_required_string(url, MAX_URL_BYTES, "steps[].url")?;
            if url.contains("{{") || contains_secret_word(url) {
                return Err(invalid(
                    "steps[].url",
                    "must be a fixed non-secret destination",
                ));
            }
            let method = match raw.method.as_deref().unwrap_or("POST") {
                "POST" => WebhookMethod::Post,
                "PUT" => WebhookMethod::Put,
                "PATCH" => WebhookMethod::Patch,
                _ => return Err(invalid("steps[].method", "is not an allowed method")),
            };
            let headers = validate_headers(raw.headers.as_ref(), secrets)?;
            let body = raw
                .body
                .as_ref()
                .map(|body| {
                    reject_literal_secret_assignments(body, "steps[].body")?;
                    validate_template(body, secrets, true, "steps[].body")
                })
                .transpose()?;
            Ok(StepAction::CallWebhook {
                url: url.clone(),
                method,
                headers,
                body,
            })
        }
        "request_approval" => {
            reject_action_fields(raw, &["from", "message", "timeout"])?;
            let from = required_ref(raw.from.as_ref(), "steps[].from")?;
            validate_required_string(from, 256, "steps[].from")?;
            let timeout_secs = match raw.timeout.as_deref() {
                Some(timeout) => parse_duration_secs(timeout, "steps[].timeout")?,
                None => 24 * 60 * 60,
            };
            if timeout_secs == 0 || timeout_secs > MAX_APPROVAL_WAIT_SECS {
                return Err(invalid(
                    "steps[].timeout",
                    "must be between 1 second and 30 days",
                ));
            }
            Ok(StepAction::RequestApproval {
                from: from.clone(),
                message: validate_template(
                    required_ref(raw.message.as_ref(), "steps[].message")?,
                    secrets,
                    false,
                    "steps[].message",
                )?,
                timeout_secs,
            })
        }
        "delay" => {
            reject_action_fields(raw, &["duration"])?;
            let duration_secs = parse_duration_secs(
                required_ref(raw.duration.as_ref(), "steps[].duration")?,
                "steps[].duration",
            )?;
            if duration_secs == 0 || duration_secs > MAX_DELAY_SECS {
                return Err(invalid(
                    "steps[].duration",
                    "must be between 1 and 270 seconds",
                ));
            }
            Ok(StepAction::Delay { duration_secs })
        }
        _ => Err(DefinitionError::UnsupportedAction),
    }
}

fn validate_headers(
    headers: Option<&BTreeMap<String, RawActionValue>>,
    secrets: &BTreeMap<String, SecretReference>,
) -> Result<BTreeMap<String, ActionValue>, DefinitionError> {
    let Some(headers) = headers else {
        return Ok(BTreeMap::new());
    };
    if headers.len() > MAX_HEADER_COUNT {
        return Err(invalid("steps[].headers", "contains too many entries"));
    }
    let mut total_bytes = 0_usize;
    headers
        .iter()
        .map(|(name, value)| {
            if name.is_empty()
                || name.len() > MAX_HEADER_NAME_BYTES
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
                || forbidden_header(name)
            {
                return Err(invalid(
                    "steps[].headers.<name>",
                    "is not an allowed outbound header name",
                ));
            }
            total_bytes = total_bytes.saturating_add(name.len());
            let value = match value {
                RawActionValue::Literal(value) => {
                    if sensitive_header(name) {
                        return Err(DefinitionError::SecretLiteral {
                            path: "steps[].headers.<value>",
                        });
                    }
                    total_bytes = total_bytes.saturating_add(value.len());
                    if value.len() > MAX_HEADER_VALUE_BYTES {
                        return Err(invalid(
                            "steps[].headers.<value>",
                            "exceeds the header value byte limit",
                        ));
                    }
                    ActionValue::Literal(validate_template(
                        value,
                        secrets,
                        true,
                        "steps[].headers.<value>",
                    )?)
                }
                RawActionValue::Secret(reference) => {
                    if !secrets.contains_key(&reference.secret_ref) {
                        return Err(invalid(
                            "steps[].headers.<value>.secret_ref",
                            "must name a declared secret reference",
                        ));
                    }
                    total_bytes = total_bytes.saturating_add(reference.secret_ref.len());
                    ActionValue::Secret {
                        secret_ref: reference.secret_ref.clone(),
                    }
                }
            };
            if total_bytes > MAX_TOTAL_HEADER_BYTES {
                return Err(invalid(
                    "steps[].headers",
                    "exceeds the aggregate header byte limit",
                ));
            }
            Ok((name.clone(), value))
        })
        .collect()
}

fn validate_condition(value: &str) -> Result<ConditionExpression, DefinitionError> {
    validate_required_string(value, MAX_CONDITION_BYTES, "condition")?;
    if value.chars().any(char::is_control) || contains_secret_word(&value) {
        return Err(invalid(
            "condition",
            "must be a non-secret expression without control characters",
        ));
    }
    let mut depth = 0_usize;
    let mut quoted = None;
    let mut escaped = false;
    for character in value.chars() {
        if let Some(quote) = quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == quote {
                quoted = None;
            }
            continue;
        }
        match character {
            '\'' | '"' => quoted = Some(character),
            '(' => {
                depth += 1;
                if depth > MAX_YAML_DEPTH {
                    return Err(invalid("condition", "exceeds nesting limit"));
                }
            }
            ')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| invalid("condition", "has unbalanced parentheses"))?;
            }
            _ => {}
        }
    }
    if quoted.is_some() || depth != 0 {
        return Err(invalid(
            "condition",
            "has an unterminated string or unbalanced parentheses",
        ));
    }
    Ok(ConditionExpression(value.to_owned()))
}

fn validate_template(
    value: &str,
    secrets: &BTreeMap<String, SecretReference>,
    allow_secrets: bool,
    path: &'static str,
) -> Result<TemplateString, DefinitionError> {
    validate_required_string(value, MAX_TEMPLATE_BYTES, path)?;
    if value.chars().any(|character| character == '\0') {
        return Err(invalid(path, "must not contain NUL"));
    }
    let mut rest = value;
    let mut expression_count = 0_usize;
    while let Some(start) = rest.find("{{") {
        if rest[..start].contains("}}") {
            return Err(invalid(path, "contains an unmatched template terminator"));
        }
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find("}}") else {
            return Err(invalid(
                path,
                "contains an unterminated template expression",
            ));
        };
        let expression = after_start[..end].trim();
        if expression.is_empty() || expression.contains("{{") {
            return Err(invalid(path, "contains an invalid template expression"));
        }
        expression_count += 1;
        if expression_count > MAX_TEMPLATE_EXPRESSIONS {
            return Err(invalid(path, "contains too many template expressions"));
        }
        if let Some(secret_name) = expression.strip_prefix("secrets.") {
            if !allow_secrets || !secrets.contains_key(secret_name) {
                return Err(invalid(path, "contains an unavailable secret reference"));
            }
        } else if contains_secret_word(expression) {
            return Err(invalid(path, "contains an invalid secret expression"));
        }
        rest = &after_start[end + 2..];
    }
    if rest.contains("}}") {
        return Err(invalid(path, "contains an unmatched template terminator"));
    }
    Ok(TemplateString(value.to_owned()))
}

fn reject_literal_secret_assignments(
    value: &str,
    path: &'static str,
) -> Result<(), DefinitionError> {
    let lower = value.to_ascii_lowercase();
    for name in [
        "secret",
        "password",
        "token",
        "access_token",
        "api_key",
        "credential",
        "authorization",
    ] {
        for separator in [":", "="] {
            for prefix in [
                format!("{name}{separator}"),
                format!("\"{name}\"{separator}"),
            ] {
                let mut remainder = lower.as_str();
                while let Some(index) = remainder.find(&prefix) {
                    let before = &remainder[..index];
                    let word_boundary = before
                        .chars()
                        .next_back()
                        .is_none_or(|character| !character.is_ascii_alphanumeric());
                    let assigned =
                        remainder[index + prefix.len()..].trim_start_matches(|character: char| {
                            character.is_ascii_whitespace() || character == '\'' || character == '"'
                        });
                    if word_boundary && !assigned.starts_with("{{ secrets.") {
                        return Err(DefinitionError::SecretLiteral { path });
                    }
                    remainder = &remainder[index + prefix.len()..];
                }
            }
        }
    }
    Ok(())
}

fn reject_action_fields(raw: &RawStep, allowed: &[&str]) -> Result<(), DefinitionError> {
    let fields = [
        ("text", raw.text.is_some()),
        ("channel", raw.channel.is_some()),
        ("to", raw.to.is_some()),
        ("topic", raw.topic.is_some()),
        ("emoji", raw.emoji.is_some()),
        ("url", raw.url.is_some()),
        ("method", raw.method.is_some()),
        ("headers", raw.headers.is_some()),
        ("body", raw.body.is_some()),
        ("from", raw.from.is_some()),
        ("message", raw.message.is_some()),
        ("timeout", raw.timeout.is_some()),
        ("duration", raw.duration.is_some()),
    ];
    if fields
        .iter()
        .any(|(field, present)| *present && !allowed.contains(field))
    {
        return Err(invalid(
            "steps[]",
            "contains a field that does not belong to its action",
        ));
    }
    Ok(())
}

fn reject_present(fields: &[(&'static str, bool)]) -> Result<(), DefinitionError> {
    if let Some((path, _)) = fields.iter().find(|(_, present)| *present) {
        return Err(invalid(path, "does not belong to this trigger"));
    }
    Ok(())
}

fn validate_cron(value: &str) -> Result<(), DefinitionError> {
    validate_required_string(value, 256, "trigger.cron")?;
    let fields: Vec<_> = value.split_whitespace().collect();
    if !(5..=7).contains(&fields.len())
        || fields.iter().any(|field| {
            field.is_empty()
                || !field
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"*?,-/#".contains(&byte))
        })
    {
        return Err(invalid(
            "trigger.cron",
            "must contain 5 to 7 bounded cron fields",
        ));
    }
    let normalized = match fields.len() {
        5 => format!("0 {value} *"),
        6 => format!("{value} *"),
        _ => value.to_owned(),
    };
    normalized
        .parse::<cron::Schedule>()
        .map_err(|_| invalid("trigger.cron", "is not a valid cron expression"))?;
    Ok(())
}

fn parse_duration_secs(value: &str, path: &'static str) -> Result<u64, DefinitionError> {
    let (digits, multiplier) = match value.as_bytes().last().copied() {
        Some(b's') => (&value[..value.len() - 1], 1_u64),
        Some(b'm') => (&value[..value.len() - 1], 60),
        Some(b'h') => (&value[..value.len() - 1], 60 * 60),
        Some(b'd') => (&value[..value.len() - 1], 24 * 60 * 60),
        _ => {
            return Err(invalid(
                path,
                "must be an integer duration ending in s, m, h or d",
            ));
        }
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(
            path,
            "must be an integer duration ending in s, m, h or d",
        ));
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|number| number.checked_mul(multiplier))
        .ok_or_else(|| invalid(path, "duration is out of range"))
}

fn validate_required_string(
    value: &str,
    max_bytes: usize,
    path: &'static str,
) -> Result<(), DefinitionError> {
    if value.trim().is_empty() || value.len() > max_bytes {
        return Err(invalid(path, "must be non-empty and within its byte limit"));
    }
    Ok(())
}

fn validate_optional_string(
    value: &str,
    max_bytes: usize,
    path: &'static str,
) -> Result<(), DefinitionError> {
    if value.len() > max_bytes {
        return Err(invalid(path, "exceeds its byte limit"));
    }
    Ok(())
}

fn valid_identifier(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value
            .bytes()
            .enumerate()
            .all(|(index, byte)| byte.is_ascii_alphanumeric() || byte == b'_' && index > 0)
}

fn contains_secret_word(value: &str) -> bool {
    value
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|word| {
            matches!(
                word.to_ascii_lowercase().as_str(),
                "secret" | "secrets" | "password" | "token" | "credential" | "api_key"
            )
        })
}

fn sensitive_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == "authorization"
        || lower == "proxy-authorization"
        || lower == "cookie"
        || lower == "set-cookie"
        || contains_secret_word(&lower)
        || lower.ends_with("-key")
}

fn forbidden_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host"
            | "content-length"
            | "connection"
            | "keep-alive"
            | "transfer-encoding"
            | "upgrade"
            | "te"
            | "trailer"
            | "forwarded"
            | "via"
            | "x-forwarded-for"
            | "x-forwarded-host"
            | "x-forwarded-proto"
    )
}

fn required<T>(value: Option<T>, path: &'static str) -> Result<T, DefinitionError> {
    value.ok_or_else(|| invalid(path, "is required"))
}

fn required_ref<'a, T>(value: Option<&'a T>, path: &'static str) -> Result<&'a T, DefinitionError> {
    value.ok_or_else(|| invalid(path, "is required"))
}

fn invalid(path: &'static str, rule: &'static str) -> DefinitionError {
    DefinitionError::InvalidField { path, rule }
}

fn bounded_error_detail(detail: &str) -> String {
    if detail.len() <= MAX_ERROR_DETAIL_BYTES {
        return detail.to_owned();
    }
    let mut end = MAX_ERROR_DETAIL_BYTES;
    while !detail.is_char_boundary(end) {
        end -= 1;
    }
    detail[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_V1: &str = r#"
version: 1
name: Incident response
description: Escalate matching messages
trigger:
  on: message_posted
  if: 'str_contains(trigger_text, "P1")'
secrets:
  incident_api:
    credential: workflow/incident-api
retry:
  mode: exponential
  max_attempts: 4
  max_elapsed_secs: 300
  initial_backoff_secs: 1
  max_backoff_secs: 30
  jitter: full
  retry_on: [rate_limited, temporary_unavailable, timeout, transport]
steps:
  - id: notify
    if: 'trigger_author != ""'
    timeout_secs: 10
    action: call_webhook
    url: https://hooks.example.com/incidents
    headers:
      Authorization:
        secret_ref: incident_api
      X-Workflow: '{{ trigger.event_id }}'
    body: '{"summary":"{{ trigger.text }}","key":"{{ secrets.incident_api }}"}'
  - id: approve
    action: request_approval
    from: '@incident-commander'
    message: Approve escalation?
    timeout: 4h
"#;

    #[test]
    fn parses_supported_v1_definition_with_conditions_steps_secrets_and_retry() {
        let definition = parse_yaml(VALID_V1).expect("valid v1 definition");
        assert_eq!(definition.version(), CURRENT_DEFINITION_VERSION);
        assert_eq!(definition.name(), "Incident response");
        assert_eq!(definition.steps().len(), 2);
        assert!(matches!(
            definition.retry(),
            RetryPolicy::Exponential {
                max_attempts: 4,
                ..
            }
        ));
        assert_eq!(
            definition
                .secrets()
                .get("incident_api")
                .map(SecretReference::credential),
            Some("workflow/incident-api")
        );
    }

    #[test]
    fn rejects_unknown_version_and_action() {
        let unknown_version = VALID_V1.replacen("version: 1", "version: 2", 1);
        assert_eq!(
            parse_yaml(&unknown_version),
            Err(DefinitionError::UnsupportedVersion { version: 2 })
        );

        let unknown_action = VALID_V1.replacen("action: call_webhook", "action: run_shell", 1);
        assert_eq!(
            parse_yaml(&unknown_action),
            Err(DefinitionError::UnsupportedAction)
        );
    }

    #[test]
    fn rejects_secret_literals_but_accepts_typed_references() {
        let literal_declaration = VALID_V1.replacen(
            "incident_api:\n    credential: workflow/incident-api",
            "incident_api: plaintext-token",
            1,
        );
        assert_eq!(
            parse_yaml(&literal_declaration),
            Err(DefinitionError::SecretLiteral {
                path: "secrets.<name>"
            })
        );

        let literal_header = VALID_V1.replacen(
            "Authorization:\n        secret_ref: incident_api",
            "Authorization: Bearer plaintext-token",
            1,
        );
        assert_eq!(
            parse_yaml(&literal_header),
            Err(DefinitionError::SecretLiteral {
                path: "steps[].headers.<value>"
            })
        );

        let literal_body = VALID_V1.replacen(
            "{\"summary\":\"{{ trigger.text }}\",\"key\":\"{{ secrets.incident_api }}\"}",
            "{\"token\":\"plaintext-token\"}",
            1,
        );
        assert_eq!(
            parse_yaml(&literal_body),
            Err(DefinitionError::SecretLiteral {
                path: "steps[].body"
            })
        );
    }

    #[test]
    fn rejects_unbounded_or_zero_delay_retry_policy() {
        let missing_attempt_limit = VALID_V1.replacen("  max_attempts: 4\n", "", 1);
        assert_eq!(
            parse_yaml(&missing_attempt_limit),
            Err(invalid("retry.max_attempts", "is required"))
        );

        let unbounded = VALID_V1.replacen("max_attempts: 4", "max_attempts: 0", 1);
        assert_eq!(
            parse_yaml(&unbounded),
            Err(invalid("retry.max_attempts", "must be between 2 and 8"))
        );

        let zero_delay = VALID_V1.replacen("initial_backoff_secs: 1", "initial_backoff_secs: 0", 1);
        assert_eq!(
            parse_yaml(&zero_delay),
            Err(invalid(
                "retry",
                "backoff must be positive, ordered and within elapsed bounds"
            ))
        );
    }

    #[test]
    fn rejects_aliases_unknown_fields_and_oversized_conditions() {
        let alias = VALID_V1.replacen(
            "name: Incident response",
            "name: &name Incident response",
            1,
        ) + "\ndisplay_name: *name\n";
        assert_eq!(
            parse_yaml(&alias),
            Err(DefinitionError::YamlAliasNotSupported)
        );

        let unknown_field = VALID_V1.replacen(
            "name: Incident response",
            "unexpected: true\nname: Incident response",
            1,
        );
        assert!(matches!(
            parse_yaml(&unknown_field),
            Err(DefinitionError::InvalidYaml { .. })
        ));

        let condition = "x".repeat(MAX_CONDITION_BYTES + 1);
        let oversized = VALID_V1.replacen(
            "if: 'trigger_author != \"\"'",
            &format!("if: '{condition}'"),
            1,
        );
        assert_eq!(
            parse_yaml(&oversized),
            Err(invalid(
                "condition",
                "must be non-empty and within its byte limit"
            ))
        );
    }

    #[test]
    fn rejects_parser_resource_bombs_and_invalid_cron() {
        let oversized = "x".repeat(MAX_DEFINITION_BYTES + 1);
        assert_eq!(
            parse_yaml(&oversized),
            Err(DefinitionError::DefinitionTooLarge)
        );

        let deeply_nested = format!(
            "{}0{}",
            "[".repeat(MAX_YAML_DEPTH + 1),
            "]".repeat(MAX_YAML_DEPTH + 1)
        );
        assert_eq!(
            parse_yaml(&deeply_nested),
            Err(DefinitionError::YamlTooDeep)
        );

        let invalid_cron = VALID_V1.replace(
            "on: message_posted\n  if: 'str_contains(trigger_text, \"P1\")'",
            "on: schedule\n  cron: '99 99 99 99 99'",
        );
        assert_eq!(
            parse_yaml(&invalid_cron),
            Err(invalid("trigger.cron", "is not a valid cron expression"))
        );
    }

    #[test]
    fn absent_retry_policy_is_never_and_buzz_actions_remain_supported() {
        let yaml = r#"
version: 1
name: Buzz action compatibility
trigger:
  on: schedule
  cron: '0 9 * * 1-5'
steps:
  - id: message
    action: send_message
    text: hello
  - id: dm
    action: send_dm
    to: '{{ trigger.author }}'
    text: hello
  - id: topic
    action: set_channel_topic
    topic: active
  - id: react
    action: add_reaction
    emoji: eyes
  - id: wait
    action: delay
    duration: 5s
"#;
        let definition = parse_yaml(yaml).expect("supported Buzz action shapes");
        assert_eq!(definition.retry(), &RetryPolicy::Never);
        assert_eq!(definition.steps().len(), 5);
    }

    #[test]
    fn canonical_definition_round_trips_through_the_validated_storage_shape() {
        let definition = parse_yaml(VALID_V1).expect("valid definition");
        let canonical = serde_json::to_string(&definition).expect("canonical storage shape");
        assert_eq!(
            WorkflowDefinition::parse_canonical_json(&canonical),
            Ok(definition)
        );
    }
}
