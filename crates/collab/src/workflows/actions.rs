use std::{collections::BTreeMap, fmt, time::Duration};

use async_trait::async_trait;
use collaboration_domain::{AuthenticatedPrincipal, OperationId, PrincipalId, TenantContext};
use collaboration_workflow::definition::{
    ActionValue, RetryFailureClass, StepAction, WebhookMethod,
};
use reqwest::header::{HeaderName, HeaderValue};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use tokio::time::timeout;
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{
    repository::{
        StoredWorkflowDefinition, StoredWorkflowRun, StoredWorkflowStep, WorkflowLifecycle,
        WorkflowRunLeaseFence, WorkflowRunState, WorkflowStepState,
    },
    webhook::{
        SystemWebhookDnsResolver, WebhookDnsResolver, WebhookNetworkPolicy,
        WebhookTransportPolicyError,
    },
};

pub const MAX_RENDERED_ACTION_FIELD_BYTES: usize = 64 * 1024;
pub const MAX_ACTION_OUTPUT_BYTES: usize = 64 * 1024;
pub const MAX_OUTBOUND_WEBHOOK_BODY_BYTES: usize = 1024 * 1024;
pub const MAX_OUTBOUND_WEBHOOK_RESPONSE_BYTES: usize = 1024 * 1024;
pub const OUTBOUND_WEBHOOK_TIMEOUT: Duration = Duration::from_secs(10);

const MAX_SECRET_BYTES: usize = 64 * 1024;
const MAX_SECRET_VERSION_BYTES: usize = 128;
const MAX_ERROR_MESSAGE_BYTES: usize = 512;
const MAX_RESPONSE_CONTENT_TYPE_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowActionKind {
    SendMessage,
    SendDm,
    SetChannelTopic,
    AddReaction,
    CallWebhook,
    RequestAgentJob,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkflowActionTarget {
    Channel(String),
    Principal(String),
    Message {
        channel: String,
        message: String,
    },
    Webhook(String),
    AgentJob {
        target_executor_principal_id: PrincipalId,
    },
}

#[derive(Clone, Debug)]
pub struct WorkflowActionAuthorization<'a> {
    pub tenant: &'a TenantContext,
    pub definition: &'a StoredWorkflowDefinition,
    pub run: &'a StoredWorkflowRun,
    pub step: &'a StoredWorkflowStep,
    pub lease: &'a WorkflowRunLeaseFence,
    pub action_kind: WorkflowActionKind,
    pub target: WorkflowActionTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowCommandDisposition {
    Applied,
    Duplicate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowCommandReceipt {
    pub operation_id: OperationId,
    pub disposition: WorkflowCommandDisposition,
    pub output: JsonValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalWorkflowCommand {
    SendMessage {
        channel: String,
        text: String,
    },
    SendDm {
        recipient: String,
        text: String,
    },
    SetChannelTopic {
        channel: String,
        topic: String,
    },
    AddReaction {
        channel: String,
        message: String,
        emoji: String,
    },
    RequestAgentJob {
        target_executor_principal_id: PrincipalId,
        prompt: String,
    },
}

#[async_trait]
pub trait WorkflowActionAuthority: Send + Sync {
    async fn authorize(
        &self,
        request: &WorkflowActionAuthorization<'_>,
    ) -> Result<AuthenticatedPrincipal, WorkflowActionError>;
}

#[async_trait]
pub trait CanonicalWorkflowCommandSink: Send + Sync {
    async fn submit(
        &self,
        tenant: &TenantContext,
        actor: &AuthenticatedPrincipal,
        operation_id: OperationId,
        command: CanonicalWorkflowCommand,
    ) -> Result<WorkflowCommandReceipt, WorkflowActionError>;
}

pub struct ResolvedActionSecret {
    version: String,
    value: Vec<u8>,
}

impl ResolvedActionSecret {
    pub fn new(version: impl Into<String>, value: Vec<u8>) -> Result<Self, WorkflowActionError> {
        let version = version.into();
        if version.is_empty()
            || version.len() > MAX_SECRET_VERSION_BYTES
            || version.trim() != version
            || version.chars().any(char::is_control)
            || value.is_empty()
            || value.len() > MAX_SECRET_BYTES
        {
            return Err(WorkflowActionError::SecretUnavailable);
        }
        Ok(Self { version, value })
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

impl fmt::Debug for ResolvedActionSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedActionSecret")
            .field("version", &self.version)
            .field("value", &"[REDACTED]")
            .finish()
    }
}

impl Drop for ResolvedActionSecret {
    fn drop(&mut self) {
        self.value.fill(0);
    }
}

#[async_trait]
pub trait WorkflowActionSecretResolver: Send + Sync {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        workflow_id: Uuid,
        secret_name: &str,
        credential_reference: &str,
    ) -> Result<ResolvedActionSecret, WorkflowActionError>;
}

#[derive(Clone, Debug)]
pub struct WorkflowActionAttempt<'a> {
    pub tenant: &'a TenantContext,
    pub definition: &'a StoredWorkflowDefinition,
    pub run: &'a StoredWorkflowRun,
    pub step: &'a StoredWorkflowStep,
    pub lease: &'a WorkflowRunLeaseFence,
    pub default_channel: Option<&'a str>,
    pub previous_step_outputs: &'a BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkflowActionOutcome {
    Completed {
        operation_id: OperationId,
        disposition: WorkflowCommandDisposition,
        output: JsonValue,
    },
    ApprovalRequired {
        from: String,
        message: String,
        timeout_secs: u64,
    },
}

pub struct WorkflowActionExecutor<A, C, S, D = SystemWebhookDnsResolver> {
    authority: A,
    commands: C,
    secrets: S,
    network_policy: WebhookNetworkPolicy<D>,
}

impl<A, C, S> WorkflowActionExecutor<A, C, S, SystemWebhookDnsResolver>
where
    A: WorkflowActionAuthority,
    C: CanonicalWorkflowCommandSink,
    S: WorkflowActionSecretResolver,
{
    pub fn system(authority: A, commands: C, secrets: S) -> Self {
        Self::new(authority, commands, secrets, WebhookNetworkPolicy::system())
    }
}

impl<A, C, S, D> WorkflowActionExecutor<A, C, S, D>
where
    A: WorkflowActionAuthority,
    C: CanonicalWorkflowCommandSink,
    S: WorkflowActionSecretResolver,
    D: WebhookDnsResolver,
{
    pub fn new(
        authority: A,
        commands: C,
        secrets: S,
        network_policy: WebhookNetworkPolicy<D>,
    ) -> Self {
        Self {
            authority,
            commands,
            secrets,
            network_policy,
        }
    }

    pub async fn execute(
        &self,
        attempt: WorkflowActionAttempt<'_>,
    ) -> Result<WorkflowActionOutcome, WorkflowActionError> {
        let action = validate_attempt(&attempt)?;
        let operation_id = OperationId::from_uuid(attempt.step.operation_id);
        match action {
            StepAction::SendMessage { text, channel } => {
                let channel = match channel {
                    Some(channel) => render_template(channel.as_str(), &attempt)?,
                    None => attempt
                        .default_channel
                        .map(str::to_owned)
                        .ok_or(WorkflowActionError::MissingTarget)?,
                };
                validate_channel_target(&channel)?;
                let text = render_template(text.as_str(), &attempt)?;
                self.execute_command(
                    &attempt,
                    WorkflowActionKind::SendMessage,
                    WorkflowActionTarget::Channel(channel.clone()),
                    CanonicalWorkflowCommand::SendMessage { channel, text },
                )
                .await
            }
            StepAction::SendDm { to, text } => {
                let recipient = render_template(to.as_str(), &attempt)?;
                validate_principal_target(&recipient)?;
                let text = render_template(text.as_str(), &attempt)?;
                self.execute_command(
                    &attempt,
                    WorkflowActionKind::SendDm,
                    WorkflowActionTarget::Principal(recipient.clone()),
                    CanonicalWorkflowCommand::SendDm { recipient, text },
                )
                .await
            }
            StepAction::SetChannelTopic { topic } => {
                let channel = attempt
                    .default_channel
                    .map(str::to_owned)
                    .ok_or(WorkflowActionError::MissingTarget)?;
                validate_channel_target(&channel)?;
                let topic = render_template(topic.as_str(), &attempt)?;
                self.execute_command(
                    &attempt,
                    WorkflowActionKind::SetChannelTopic,
                    WorkflowActionTarget::Channel(channel.clone()),
                    CanonicalWorkflowCommand::SetChannelTopic { channel, topic },
                )
                .await
            }
            StepAction::AddReaction { emoji } => {
                let channel = attempt
                    .default_channel
                    .map(str::to_owned)
                    .ok_or(WorkflowActionError::MissingTarget)?;
                let message = trigger_scalar(&attempt, "message_id")?;
                validate_channel_target(&channel)?;
                validate_message_target(&message)?;
                let emoji = render_template(emoji.as_str(), &attempt)?;
                self.execute_command(
                    &attempt,
                    WorkflowActionKind::AddReaction,
                    WorkflowActionTarget::Message {
                        channel: channel.clone(),
                        message: message.clone(),
                    },
                    CanonicalWorkflowCommand::AddReaction {
                        channel,
                        message,
                        emoji,
                    },
                )
                .await
            }
            StepAction::CallWebhook {
                url,
                method,
                headers,
                body,
            } => {
                self.execute_webhook(
                    &attempt,
                    operation_id,
                    url,
                    *method,
                    headers,
                    body.as_ref().map(|body| body.as_str()),
                )
                .await
            }
            StepAction::RequestApproval {
                from,
                message,
                timeout_secs,
            } => Ok(WorkflowActionOutcome::ApprovalRequired {
                from: from.clone(),
                message: render_template(message.as_str(), &attempt)?,
                timeout_secs: *timeout_secs,
            }),
            StepAction::Delay { duration_secs } => {
                timeout(
                    Duration::from_secs(
                        attempt.definition.definition.steps()[usize::from(attempt.step.index)]
                            .timeout_secs(),
                    ),
                    tokio::time::sleep(Duration::from_secs(*duration_secs)),
                )
                .await
                .map_err(|_| WorkflowActionError::Timeout)?;
                Ok(WorkflowActionOutcome::Completed {
                    operation_id,
                    disposition: WorkflowCommandDisposition::Applied,
                    output: json!({ "delayed_seconds": duration_secs }),
                })
            }
        }
    }

    async fn execute_command(
        &self,
        attempt: &WorkflowActionAttempt<'_>,
        action_kind: WorkflowActionKind,
        target: WorkflowActionTarget,
        command: CanonicalWorkflowCommand,
    ) -> Result<WorkflowActionOutcome, WorkflowActionError> {
        let actor = self.authorize(attempt, action_kind, target).await?;
        let operation_id = OperationId::from_uuid(attempt.step.operation_id);
        let receipt = self
            .commands
            .submit(attempt.tenant, &actor, operation_id, command)
            .await?;
        if receipt.operation_id != operation_id {
            return Err(WorkflowActionError::ConflictingReceipt);
        }
        validate_output(&receipt.output)?;
        Ok(WorkflowActionOutcome::Completed {
            operation_id,
            disposition: receipt.disposition,
            output: receipt.output,
        })
    }

    async fn authorize(
        &self,
        attempt: &WorkflowActionAttempt<'_>,
        action_kind: WorkflowActionKind,
        target: WorkflowActionTarget,
    ) -> Result<AuthenticatedPrincipal, WorkflowActionError> {
        let actor = self
            .authority
            .authorize(&WorkflowActionAuthorization {
                tenant: attempt.tenant,
                definition: attempt.definition,
                run: attempt.run,
                step: attempt.step,
                lease: attempt.lease,
                action_kind,
                target,
            })
            .await?;
        if actor.community_id() != attempt.tenant.community_id()
            || actor.principal_id() != attempt.definition.creator_principal_id
        {
            return Err(WorkflowActionError::PermissionDenied);
        }
        Ok(actor)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_webhook(
        &self,
        attempt: &WorkflowActionAttempt<'_>,
        operation_id: OperationId,
        url: &str,
        method: WebhookMethod,
        headers: &BTreeMap<String, ActionValue>,
        body: Option<&str>,
    ) -> Result<WorkflowActionOutcome, WorkflowActionError> {
        self.authorize(
            attempt,
            WorkflowActionKind::CallWebhook,
            WorkflowActionTarget::Webhook(url.to_owned()),
        )
        .await?;
        let destination = self.network_policy.pin(url).await?;
        let client = destination.build_client()?;
        let mut request = client.request(
            match method {
                WebhookMethod::Post => reqwest::Method::POST,
                WebhookMethod::Put => reqwest::Method::PUT,
                WebhookMethod::Patch => reqwest::Method::PATCH,
            },
            destination.url().clone(),
        );
        request = request.header("Idempotency-Key", operation_id.as_uuid().to_string());
        let mut secret_names = BTreeMap::<String, String>::new();
        for value in headers.values() {
            match value {
                ActionValue::Literal(value) => {
                    collect_secret_references(value.as_str(), attempt, &mut secret_names)?;
                }
                ActionValue::Secret { secret_ref } => {
                    insert_secret_reference(secret_ref, attempt, &mut secret_names)?;
                }
            }
        }
        if let Some(body) = body {
            collect_secret_references(body, attempt, &mut secret_names)?;
        }
        let mut resolved_secrets = BTreeMap::new();
        for (secret_name, credential_reference) in secret_names {
            let secret = self
                .secrets
                .resolve(
                    attempt.tenant,
                    attempt.definition.identity.workflow_id(),
                    &secret_name,
                    &credential_reference,
                )
                .await?;
            resolved_secrets.insert(secret_name, secret);
        }
        let mut rendered_header_bytes = 0_usize;
        for (name, value) in headers {
            if name.eq_ignore_ascii_case("idempotency-key") {
                return Err(WorkflowActionError::InvalidRenderedInput);
            }
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| WorkflowActionError::InvalidRenderedInput)?;
            let value = match value {
                ActionValue::Literal(value) => {
                    render_sensitive_template(value.as_str(), attempt, &resolved_secrets)?
                }
                ActionValue::Secret { secret_ref } => {
                    let secret = resolved_secrets
                        .get(secret_ref)
                        .ok_or(WorkflowActionError::SecretUnavailable)?;
                    Zeroizing::new(
                        std::str::from_utf8(&secret.value)
                            .map_err(|_| WorkflowActionError::SecretUnavailable)?
                            .to_owned(),
                    )
                }
            };
            rendered_header_bytes = rendered_header_bytes
                .saturating_add(name.as_str().len())
                .saturating_add(value.len());
            if rendered_header_bytes > MAX_RENDERED_ACTION_FIELD_BYTES {
                return Err(WorkflowActionError::RenderedInputTooLarge);
            }
            let value = HeaderValue::from_str(value.as_str())
                .map_err(|_| WorkflowActionError::InvalidRenderedInput)?;
            request = request.header(name, value);
        }
        if let Some(body) = body {
            let body = render_sensitive_template(body, attempt, &resolved_secrets)?;
            if body.len() > MAX_OUTBOUND_WEBHOOK_BODY_BYTES {
                return Err(WorkflowActionError::RenderedInputTooLarge);
            }
            request = request.body(body.to_string());
        }
        let mut response = timeout(OUTBOUND_WEBHOOK_TIMEOUT, request.send())
            .await
            .map_err(|_| WorkflowActionError::AmbiguousDelivery)?
            .map_err(|_| WorkflowActionError::AmbiguousDelivery)?;
        let status = response.status();
        if status.is_redirection() {
            return Err(WorkflowActionError::RedirectRejected);
        }
        if !status.is_success() {
            return Err(WorkflowActionError::Rejected);
        }
        if response
            .headers()
            .get(reqwest::header::CONTENT_ENCODING)
            .is_some_and(|value| value.as_bytes() != b"identity")
        {
            return Err(WorkflowActionError::UnsupportedResponseEncoding);
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if content_type
            .as_ref()
            .is_some_and(|value| value.len() > MAX_RESPONSE_CONTENT_TYPE_BYTES)
        {
            return Err(WorkflowActionError::InvalidOutput);
        }
        let mut response_bytes = Vec::new();
        while let Some(chunk) = timeout(OUTBOUND_WEBHOOK_TIMEOUT, response.chunk())
            .await
            .map_err(|_| WorkflowActionError::AmbiguousDelivery)?
            .map_err(|_| WorkflowActionError::AmbiguousDelivery)?
        {
            if response_bytes.len().saturating_add(chunk.len())
                > MAX_OUTBOUND_WEBHOOK_RESPONSE_BYTES
            {
                return Err(WorkflowActionError::ResponseTooLarge);
            }
            response_bytes.extend_from_slice(&chunk);
        }
        let output = json!({
            "status": status.as_u16(),
            "body_bytes": response_bytes.len(),
            "body_sha256": hex::encode(Sha256::digest(&response_bytes)),
            "content_type": content_type,
            "credential_versions": resolved_secrets
                .iter()
                .map(|(name, secret)| json!({
                    "secret_ref": name,
                    "version": secret.version(),
                }))
                .collect::<Vec<_>>(),
        });
        validate_output(&output)?;
        Ok(WorkflowActionOutcome::Completed {
            operation_id,
            disposition: WorkflowCommandDisposition::Applied,
            output,
        })
    }
}

fn validate_attempt<'a>(
    attempt: &'a WorkflowActionAttempt<'_>,
) -> Result<&'a StepAction, WorkflowActionError> {
    if attempt.definition.identity.community_id() != attempt.tenant.community_id()
        || attempt.run.identity.community_id() != attempt.tenant.community_id()
        || attempt.run.workflow != attempt.definition.identity
        || attempt.run.definition_version != attempt.definition.definition_version
        || attempt.definition.current_definition_version != attempt.definition.definition_version
        || attempt.definition.lifecycle != WorkflowLifecycle::Active
        || !attempt.definition.definition.enabled()
        || attempt.run.state != WorkflowRunState::Running
        || attempt.step.state != WorkflowStepState::Running
        || usize::from(attempt.step.index) >= attempt.definition.definition.steps().len()
    {
        return Err(WorkflowActionError::StaleAttempt);
    }
    let definition_step = &attempt.definition.definition.steps()[usize::from(attempt.step.index)];
    if definition_step.id() != attempt.step.step_id || attempt.step.operation_id.is_nil() {
        return Err(WorkflowActionError::StaleAttempt);
    }
    Ok(definition_step.action())
}

fn render_template(
    template: &str,
    attempt: &WorkflowActionAttempt<'_>,
) -> Result<String, WorkflowActionError> {
    render_template_with_secrets(template, attempt, None)
}

fn render_sensitive_template(
    template: &str,
    attempt: &WorkflowActionAttempt<'_>,
    secrets: &BTreeMap<String, ResolvedActionSecret>,
) -> Result<Zeroizing<String>, WorkflowActionError> {
    render_template_with_secrets(template, attempt, Some(secrets)).map(Zeroizing::new)
}

fn render_template_with_secrets(
    template: &str,
    attempt: &WorkflowActionAttempt<'_>,
    secrets: Option<&BTreeMap<String, ResolvedActionSecret>>,
) -> Result<String, WorkflowActionError> {
    let mut output = String::with_capacity(template.len());
    let mut remainder = template;
    while let Some(start) = remainder.find("{{") {
        output.push_str(&remainder[..start]);
        let expression = &remainder[start + 2..];
        let end = expression
            .find("}}")
            .ok_or(WorkflowActionError::InvalidRenderedInput)?;
        let path = expression[..end].trim();
        if path.is_empty() || path.contains("{{") {
            return Err(WorkflowActionError::InvalidRenderedInput);
        }
        let resolved = Zeroizing::new(resolve_template_path(path, attempt, secrets)?);
        output.push_str(resolved.as_str());
        if output.len() > MAX_RENDERED_ACTION_FIELD_BYTES {
            return Err(WorkflowActionError::RenderedInputTooLarge);
        }
        remainder = &expression[end + 2..];
    }
    if remainder.contains("}}") {
        return Err(WorkflowActionError::InvalidRenderedInput);
    }
    output.push_str(remainder);
    if output.len() > MAX_RENDERED_ACTION_FIELD_BYTES {
        return Err(WorkflowActionError::RenderedInputTooLarge);
    }
    Ok(output)
}

fn resolve_template_path(
    path: &str,
    attempt: &WorkflowActionAttempt<'_>,
    secrets: Option<&BTreeMap<String, ResolvedActionSecret>>,
) -> Result<String, WorkflowActionError> {
    let mut segments = path.split('.');
    let root = segments
        .next()
        .ok_or(WorkflowActionError::InvalidRenderedInput)?;
    if root == "secrets" {
        let name = segments
            .next()
            .ok_or(WorkflowActionError::SecretUnavailable)?;
        if name.is_empty() || segments.next().is_some() {
            return Err(WorkflowActionError::SecretUnavailable);
        }
        let secret = secrets
            .and_then(|secrets| secrets.get(name))
            .ok_or(WorkflowActionError::SecretUnavailable)?;
        return std::str::from_utf8(&secret.value)
            .map(str::to_owned)
            .map_err(|_| WorkflowActionError::SecretUnavailable);
    }
    let mut value = match root {
        "trigger" => &attempt.run.trigger_context,
        "steps" => {
            let step_id = segments
                .next()
                .ok_or(WorkflowActionError::InvalidRenderedInput)?;
            attempt
                .previous_step_outputs
                .get(step_id)
                .ok_or(WorkflowActionError::InvalidRenderedInput)?
        }
        _ => return Err(WorkflowActionError::InvalidRenderedInput),
    };
    for segment in segments {
        if segment.is_empty() {
            return Err(WorkflowActionError::InvalidRenderedInput);
        }
        value = value
            .as_object()
            .and_then(|object| object.get(segment))
            .ok_or(WorkflowActionError::InvalidRenderedInput)?;
    }
    match value {
        JsonValue::Null => Ok(String::new()),
        JsonValue::Bool(value) => Ok(value.to_string()),
        JsonValue::Number(value) => Ok(value.to_string()),
        JsonValue::String(value) if !value.contains("{{") && !value.contains("}}") => {
            Ok(value.clone())
        }
        _ => Err(WorkflowActionError::InvalidRenderedInput),
    }
}

fn trigger_scalar(
    attempt: &WorkflowActionAttempt<'_>,
    field: &str,
) -> Result<String, WorkflowActionError> {
    resolve_template_path(&format!("trigger.{field}"), attempt, None)
}

fn collect_secret_references(
    template: &str,
    attempt: &WorkflowActionAttempt<'_>,
    references: &mut BTreeMap<String, String>,
) -> Result<(), WorkflowActionError> {
    let mut remainder = template;
    while let Some(start) = remainder.find("{{") {
        let expression = &remainder[start + 2..];
        let end = expression
            .find("}}")
            .ok_or(WorkflowActionError::InvalidRenderedInput)?;
        let path = expression[..end].trim();
        if let Some(secret_name) = path.strip_prefix("secrets.") {
            insert_secret_reference(secret_name, attempt, references)?;
        }
        remainder = &expression[end + 2..];
    }
    Ok(())
}

fn insert_secret_reference(
    secret_name: &str,
    attempt: &WorkflowActionAttempt<'_>,
    references: &mut BTreeMap<String, String>,
) -> Result<(), WorkflowActionError> {
    let reference = attempt
        .definition
        .definition
        .secrets()
        .get(secret_name)
        .ok_or(WorkflowActionError::SecretUnavailable)?;
    references.insert(secret_name.to_owned(), reference.credential().to_owned());
    Ok(())
}

fn validate_output(output: &JsonValue) -> Result<(), WorkflowActionError> {
    let bytes = serde_json::to_vec(output).map_err(|_| WorkflowActionError::InvalidOutput)?;
    if bytes.len() > MAX_ACTION_OUTPUT_BYTES {
        return Err(WorkflowActionError::OutputTooLarge);
    }
    Ok(())
}

fn validate_channel_target(value: &str) -> Result<(), WorkflowActionError> {
    let parsed = Uuid::parse_str(value).map_err(|_| WorkflowActionError::InvalidRenderedInput)?;
    if parsed.is_nil() || parsed.to_string() != value {
        return Err(WorkflowActionError::InvalidRenderedInput);
    }
    Ok(())
}

fn validate_principal_target(value: &str) -> Result<(), WorkflowActionError> {
    if Uuid::parse_str(value).is_ok_and(|parsed| !parsed.is_nil() && parsed.to_string() == value) {
        return Ok(());
    }
    if value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
        && value.bytes().any(|byte| byte != b'0')
    {
        return Ok(());
    }
    Err(WorkflowActionError::InvalidRenderedInput)
}

fn validate_message_target(value: &str) -> Result<(), WorkflowActionError> {
    if value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
        && value.bytes().any(|byte| byte != b'0')
    {
        return Ok(());
    }
    Err(WorkflowActionError::InvalidRenderedInput)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WorkflowActionError {
    #[error("workflow action attempt is stale or invalid")]
    StaleAttempt,
    #[error("workflow action is not currently authorized")]
    PermissionDenied,
    #[error("workflow action target is unavailable")]
    MissingTarget,
    #[error("workflow action rendered an invalid input")]
    InvalidRenderedInput,
    #[error("workflow action rendered input exceeds its byte limit")]
    RenderedInputTooLarge,
    #[error("workflow action credential is unavailable")]
    SecretUnavailable,
    #[error("canonical command service is unavailable")]
    CommandUnavailable,
    #[error("canonical command rejected the action")]
    Rejected,
    #[error("canonical command returned a conflicting receipt")]
    ConflictingReceipt,
    #[error("workflow action output is invalid")]
    InvalidOutput,
    #[error("workflow action output exceeds its byte limit")]
    OutputTooLarge,
    #[error("workflow action timed out")]
    Timeout,
    #[error("workflow action transport failed")]
    Transport,
    #[error("workflow webhook delivery has an ambiguous outcome")]
    AmbiguousDelivery,
    #[error("workflow webhook redirect was rejected")]
    RedirectRejected,
    #[error("workflow webhook response exceeds its byte limit")]
    ResponseTooLarge,
    #[error("workflow webhook response encoding is unsupported")]
    UnsupportedResponseEncoding,
    #[error("workflow action was rate limited")]
    RateLimited,
    #[error("workflow action dependency is temporarily unavailable")]
    TemporaryUnavailable,
    #[error("workflow webhook transport policy rejected the target")]
    WebhookPolicy,
}

impl From<WebhookTransportPolicyError> for WorkflowActionError {
    fn from(_: WebhookTransportPolicyError) -> Self {
        Self::WebhookPolicy
    }
}

impl WorkflowActionError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::StaleAttempt => "stale_attempt",
            Self::PermissionDenied => "permission_denied",
            Self::MissingTarget => "missing_target",
            Self::InvalidRenderedInput => "invalid_rendered_input",
            Self::RenderedInputTooLarge => "rendered_input_too_large",
            Self::SecretUnavailable => "secret_unavailable",
            Self::CommandUnavailable => "command_unavailable",
            Self::Rejected => "rejected",
            Self::ConflictingReceipt => "conflicting_receipt",
            Self::InvalidOutput => "invalid_output",
            Self::OutputTooLarge => "output_too_large",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::AmbiguousDelivery => "ambiguous_delivery",
            Self::RedirectRejected => "redirect_rejected",
            Self::ResponseTooLarge => "response_too_large",
            Self::UnsupportedResponseEncoding => "unsupported_response_encoding",
            Self::RateLimited => "rate_limited",
            Self::TemporaryUnavailable => "temporary_unavailable",
            Self::WebhookPolicy => "webhook_policy",
        }
    }

    pub const fn retry_failure_class(self) -> Option<RetryFailureClass> {
        match self {
            Self::RateLimited => Some(RetryFailureClass::RateLimited),
            Self::TemporaryUnavailable | Self::CommandUnavailable => {
                Some(RetryFailureClass::TemporaryUnavailable)
            }
            Self::Timeout => Some(RetryFailureClass::Timeout),
            Self::Transport => Some(RetryFailureClass::Transport),
            _ => None,
        }
    }

    pub const fn requires_repair(self) -> bool {
        matches!(
            self,
            Self::AmbiguousDelivery
                | Self::RedirectRejected
                | Self::ResponseTooLarge
                | Self::UnsupportedResponseEncoding
        )
    }
}

pub fn bounded_action_error(error: &WorkflowActionError) -> String {
    let message = error.to_string();
    message.chars().take(MAX_ERROR_MESSAGE_BYTES).collect()
}
