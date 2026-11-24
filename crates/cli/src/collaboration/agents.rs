use std::{collections::BTreeMap, fmt};

use collaboration_domain::{
    AggregateVersion, CommunityId, Job, JobCommand, JobIdentity, JobState, OperationId,
    PrivateAgentCatalogProjectionSource, PublicAgentCatalogProjection,
};
use nostr_compat::{
    EventId, PublicKey,
    buzz_nips::{
        agent_activity::validate_engram_slug,
        identity::{ArchiveAction, ArchivedIdentitySnapshot, IdentityArchiveRequest},
    },
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

const MAX_PRIVATE_INPUT_BYTES: usize = 1_048_576;
const MAX_RESOURCE_ID_BYTES: usize = 2_048;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySlug(String);

impl MemorySlug {
    pub fn new(value: impl Into<String>) -> Result<Self, AgentsCliError> {
        let value = value.into();
        validate_engram_slug(&value).map_err(|_| AgentsCliError::InvalidRequest)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct PrivateText(String);

impl PrivateText {
    pub fn new(value: impl Into<String>, allow_empty: bool) -> Result<Self, AgentsCliError> {
        let value = value.into();
        if (!allow_empty && value.is_empty()) || value.len() > MAX_PRIVATE_INPUT_BYTES {
            return Err(AgentsCliError::InvalidRequest);
        }
        Ok(Self(value))
    }

    pub fn expose_to_executor(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PrivateText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateText(<redacted>)")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentHash(String);

impl ContentHash {
    pub fn parse(value: impl Into<String>) -> Result<Self, AgentsCliError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AgentsCliError::InvalidRequest);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersonaPackInput {
    pub manifest_json: PrivateText,
    pub files: BTreeMap<String, PrivateText>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersonaSummary {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub model: Option<String>,
    pub runtime: Option<String>,
    pub skills: Vec<String>,
    pub mcp_server_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersonaPackSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub personas: Vec<PersonaSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryListing {
    pub slug: MemorySlug,
    pub event_id: EventId,
    pub created_at: u64,
}

#[derive(Clone, Eq, PartialEq)]
pub struct MemoryValue {
    pub slug: MemorySlug,
    pub event_id: EventId,
    pub value: PrivateText,
}

impl fmt::Debug for MemoryValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryValue")
            .field("slug", &self.slug)
            .field("event_id", &self.event_id)
            .field("value", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SnapshotSummary {
    pub owner_public_key: PublicKey,
    pub agent_public_key: PublicKey,
    pub snapshot_id: String,
    pub predecessor_snapshot_id: Option<String>,
    pub source_generation: u64,
    pub source_event_id: String,
    pub created_at: u64,
    pub persona_sha256: ContentHash,
    pub team_sha256: Option<ContentHash>,
    pub runtime_sha256: ContentHash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: String,
    pub version: Option<AggregateVersion>,
}

impl AgentWriteReceipt {
    pub fn new(
        operation_id: OperationId,
        resource_id: impl Into<String>,
        version: Option<AggregateVersion>,
    ) -> Result<Self, AgentsCliError> {
        let resource_id = resource_id.into();
        if resource_id.trim().is_empty()
            || resource_id.len() > MAX_RESOURCE_ID_BYTES
            || resource_id.chars().any(char::is_control)
        {
            return Err(AgentsCliError::InvalidRequest);
        }
        Ok(Self {
            operation_id,
            resource_id,
            version,
        })
    }
}

#[derive(Clone)]
pub enum AgentsCliCommand {
    CreateAgentDraft {
        community_id: CommunityId,
        private_catalog: PrivateAgentCatalogProjectionSource,
        operation_id: OperationId,
    },
    UpdateAgentDraft {
        community_id: CommunityId,
        private_catalog: PrivateAgentCatalogProjectionSource,
        expected_generation: u64,
        operation_id: OperationId,
    },
    GetAgentCatalog {
        community_id: CommunityId,
        owner_public_key: PublicKey,
    },
    ArchiveIdentity {
        community_id: CommunityId,
        request: IdentityArchiveRequest,
        operation_id: OperationId,
    },
    UnarchiveIdentity {
        community_id: CommunityId,
        request: IdentityArchiveRequest,
        operation_id: OperationId,
    },
    ArchivedIdentities {
        community_id: CommunityId,
    },
    ValidatePersonaPack(PersonaPackInput),
    InspectPersonaPack(PersonaPackInput),
    ListMemories {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
    },
    GetMemory {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        slug: MemorySlug,
    },
    HashMemory {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        slug: MemorySlug,
    },
    SetMemory {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        slug: MemorySlug,
        value: PrivateText,
        operation_id: OperationId,
    },
    PatchMemory {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        slug: MemorySlug,
        patch: PrivateText,
        base_hash: Option<ContentHash>,
        dry_run: bool,
        operation_id: OperationId,
    },
    RemoveMemory {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        slug: MemorySlug,
        operation_id: OperationId,
    },
    CreateSnapshot {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        operation_id: OperationId,
    },
    ListSnapshots {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
    },
    RestoreSnapshot {
        owner_public_key: PublicKey,
        agent_public_key: PublicKey,
        snapshot_id: String,
        operation_id: OperationId,
    },
    GetJob {
        identity: JobIdentity,
    },
    SubmitJob {
        command: JobCommand,
    },
    CancelJob {
        command: JobCommand,
    },
    DelegateJob {
        command: JobCommand,
        ancestry: Vec<JobIdentity>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AgentsCliVerb {
    DraftCreate,
    DraftUpdate,
    CatalogGet,
    Archive,
    Unarchive,
    Archived,
    PackValidate,
    PackInspect,
    MemoryList,
    MemoryGet,
    MemoryHash,
    MemorySet,
    MemoryPatch,
    MemoryRemove,
    SnapshotCreate,
    SnapshotList,
    SnapshotRestore,
    JobGet,
    JobSubmit,
    JobCancel,
    JobDelegate,
}

impl AgentsCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::DraftCreate => "agents.draft-create",
            Self::DraftUpdate => "agents.draft-update",
            Self::CatalogGet => "agents.catalog",
            Self::Archive => "agents.archive",
            Self::Unarchive => "agents.unarchive",
            Self::Archived => "agents.archived",
            Self::PackValidate => "pack.validate",
            Self::PackInspect => "pack.inspect",
            Self::MemoryList => "mem.ls",
            Self::MemoryGet => "mem.get",
            Self::MemoryHash => "mem.hash",
            Self::MemorySet => "mem.set",
            Self::MemoryPatch => "mem.patch",
            Self::MemoryRemove => "mem.rm",
            Self::SnapshotCreate => "agents.snapshot.create",
            Self::SnapshotList => "agents.snapshot.list",
            Self::SnapshotRestore => "agents.snapshot.restore",
            Self::JobGet => "agents.job.get",
            Self::JobSubmit => "agents.job.submit",
            Self::JobCancel => "agents.job.cancel",
            Self::JobDelegate => "agents.job.delegate",
        }
    }
}

impl AgentsCliCommand {
    const fn verb(&self) -> AgentsCliVerb {
        match self {
            Self::CreateAgentDraft { .. } => AgentsCliVerb::DraftCreate,
            Self::UpdateAgentDraft { .. } => AgentsCliVerb::DraftUpdate,
            Self::GetAgentCatalog { .. } => AgentsCliVerb::CatalogGet,
            Self::ArchiveIdentity { .. } => AgentsCliVerb::Archive,
            Self::UnarchiveIdentity { .. } => AgentsCliVerb::Unarchive,
            Self::ArchivedIdentities { .. } => AgentsCliVerb::Archived,
            Self::ValidatePersonaPack(_) => AgentsCliVerb::PackValidate,
            Self::InspectPersonaPack(_) => AgentsCliVerb::PackInspect,
            Self::ListMemories { .. } => AgentsCliVerb::MemoryList,
            Self::GetMemory { .. } => AgentsCliVerb::MemoryGet,
            Self::HashMemory { .. } => AgentsCliVerb::MemoryHash,
            Self::SetMemory { .. } => AgentsCliVerb::MemorySet,
            Self::PatchMemory { .. } => AgentsCliVerb::MemoryPatch,
            Self::RemoveMemory { .. } => AgentsCliVerb::MemoryRemove,
            Self::CreateSnapshot { .. } => AgentsCliVerb::SnapshotCreate,
            Self::ListSnapshots { .. } => AgentsCliVerb::SnapshotList,
            Self::RestoreSnapshot { .. } => AgentsCliVerb::SnapshotRestore,
            Self::GetJob { .. } => AgentsCliVerb::JobGet,
            Self::SubmitJob { .. } => AgentsCliVerb::JobSubmit,
            Self::CancelJob { .. } => AgentsCliVerb::JobCancel,
            Self::DelegateJob { .. } => AgentsCliVerb::JobDelegate,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::ArchiveIdentity { request, .. } => request.action == ArchiveAction::Archive,
            Self::UnarchiveIdentity { request, .. } => {
                request.action == ArchiveAction::Unarchive && request.replaced_by.is_none()
            }
            Self::RestoreSnapshot { snapshot_id, .. } => valid_resource_id(snapshot_id),
            Self::SubmitJob { command } => {
                command.kind().command_type() == collaboration_domain::JobCommandType::Request
            }
            Self::CancelJob { command } => {
                command.kind().command_type() == collaboration_domain::JobCommandType::Cancel
            }
            Self::DelegateJob { command, ancestry } => {
                command.kind().command_type() == collaboration_domain::JobCommandType::Request
                    && !ancestry.is_empty()
            }
            _ => true,
        }
    }
}

impl fmt::Debug for AgentsCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentsCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentsCliOutcome {
    Applied(AgentWriteReceipt),
    Catalog(PublicAgentCatalogProjection),
    Archived(ArchivedIdentitySnapshot),
    PersonaPack(PersonaPackSummary),
    PersonaPackValid,
    Memories(Vec<MemoryListing>),
    Memory(MemoryValue),
    MemoryHash(ContentHash),
    Snapshot(SnapshotSummary),
    Snapshots(Vec<SnapshotSummary>),
    Job(Job),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentsCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    PermissionDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl AgentsCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "agents_cli_invalid_request",
            Self::NotFound => "agents_cli_not_found",
            Self::Unavailable => "agents_cli_unavailable",
            Self::PermissionDenied => "agents_cli_permission_denied",
            Self::PartialFailure => "agents_cli_completion_unknown",
            Self::Unexpected => "agents_cli_unexpected_response",
            Self::Conflict => "agents_cli_stale_version",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::PermissionDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait AgentsCliExecutor {
    fn execute(&self, command: AgentsCliCommand) -> Result<AgentsCliOutcome, AgentsCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentsCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_agents_command(
    executor: &impl AgentsCliExecutor,
    command: AgentsCliCommand,
) -> AgentsCliExecution {
    let verb = command.verb();
    if !command.is_valid() {
        return error_output(verb, AgentsCliError::InvalidRequest);
    }
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Some(output) => AgentsCliExecution {
                stdout: format!("{output}\n"),
                stderr: String::new(),
                exit_code: 0,
            },
            None => error_output(verb, AgentsCliError::Unexpected),
        },
        Err(error) => error_output(verb, error),
    }
}

fn error_output(verb: AgentsCliVerb, error: AgentsCliError) -> AgentsCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    AgentsCliExecution {
        stdout: String::new(),
        stderr: format!(
            "{}\n",
            json!({
                "command": verb.as_str(),
                "error": contract.category,
                "error_code": diagnostic,
                "message": diagnostic,
                "ok": false,
                "retryable": contract.retryable,
            })
        ),
        exit_code: contract.exit_class as i32,
    }
}

fn success_output(verb: AgentsCliVerb, outcome: AgentsCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (
            AgentsCliVerb::DraftCreate
            | AgentsCliVerb::DraftUpdate
            | AgentsCliVerb::Archive
            | AgentsCliVerb::Unarchive
            | AgentsCliVerb::MemorySet
            | AgentsCliVerb::MemoryRemove
            | AgentsCliVerb::SnapshotCreate
            | AgentsCliVerb::SnapshotRestore
            | AgentsCliVerb::JobSubmit
            | AgentsCliVerb::JobCancel
            | AgentsCliVerb::JobDelegate,
            AgentsCliOutcome::Applied(receipt),
        ) => Some(receipt_output(verb, receipt)),
        (AgentsCliVerb::MemoryPatch, AgentsCliOutcome::Applied(receipt)) => {
            Some(receipt_output(verb, receipt))
        }
        (AgentsCliVerb::CatalogGet, AgentsCliOutcome::Catalog(catalog)) => Some(json!({
            "catalog": catalog,
            "command": verb.as_str(),
            "ok": true,
        })),
        (AgentsCliVerb::Archived, AgentsCliOutcome::Archived(snapshot)) => Some(json!({
            "command": verb.as_str(),
            "identities": snapshot.identities.iter().map(|identity| identity.to_hex()).collect::<Vec<_>>(),
            "ok": true,
        })),
        (AgentsCliVerb::PackValidate, AgentsCliOutcome::PersonaPackValid) => Some(json!({
            "command": verb.as_str(), "ok": true, "valid": true,
        })),
        (AgentsCliVerb::PackInspect, AgentsCliOutcome::PersonaPack(pack)) => {
            Some(persona_pack_output(verb, pack))
        }
        (AgentsCliVerb::MemoryList, AgentsCliOutcome::Memories(memories)) => Some(json!({
            "command": verb.as_str(),
            "memories": memories.iter().map(memory_listing_output).collect::<Vec<_>>(),
            "ok": true,
        })),
        (AgentsCliVerb::MemoryGet, AgentsCliOutcome::Memory(memory)) => Some(json!({
            "command": verb.as_str(),
            "event_id": memory.event_id,
            "ok": true,
            "slug": memory.slug.as_str(),
            "value": memory.value.expose_to_executor(),
        })),
        (AgentsCliVerb::MemoryHash, AgentsCliOutcome::MemoryHash(hash)) => Some(json!({
            "command": verb.as_str(), "ok": true, "sha256": hash.as_str(),
        })),
        (AgentsCliVerb::SnapshotList, AgentsCliOutcome::Snapshots(snapshots)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "snapshots": snapshots.iter().map(snapshot_output).collect::<Vec<_>>(),
        })),
        (AgentsCliVerb::JobGet, AgentsCliOutcome::Job(job)) => Some(job_output(verb, &job)),
        _ => None,
    }
}

fn persona_pack_output(verb: AgentsCliVerb, pack: PersonaPackSummary) -> Value {
    json!({
        "command": verb.as_str(),
        "description": pack.description,
        "id": pack.id,
        "name": pack.name,
        "ok": true,
        "personas": pack.personas.iter().map(|persona| json!({
            "description": persona.description,
            "display_name": persona.display_name,
            "mcp_server_count": persona.mcp_server_count,
            "model": persona.model,
            "name": persona.name,
            "runtime": persona.runtime,
            "skills": persona.skills,
        })).collect::<Vec<_>>(),
        "version": pack.version,
    })
}

fn memory_listing_output(memory: &MemoryListing) -> Value {
    json!({
        "created_at": memory.created_at,
        "event_id": memory.event_id,
        "slug": memory.slug.as_str(),
    })
}

fn snapshot_output(snapshot: &SnapshotSummary) -> Value {
    json!({
        "agent_public_key": snapshot.agent_public_key.to_hex(),
        "created_at": snapshot.created_at,
        "owner_public_key": snapshot.owner_public_key.to_hex(),
        "persona_sha256": snapshot.persona_sha256.as_str(),
        "predecessor_snapshot_id": snapshot.predecessor_snapshot_id,
        "runtime_sha256": snapshot.runtime_sha256.as_str(),
        "snapshot_id": snapshot.snapshot_id,
        "source_event_id": snapshot.source_event_id,
        "source_generation": snapshot.source_generation,
        "team_sha256": snapshot.team_sha256.as_ref().map(ContentHash::as_str),
    })
}

fn job_output(verb: AgentsCliVerb, job: &Job) -> Value {
    json!({
        "command": verb.as_str(),
        "community_id": job.identity().community_id(),
        "job_id": job.identity().job_id(),
        "ok": true,
        "requested_at_millis": job.requested_at_millis(),
        "requester_principal_id": job.requester_principal_id(),
        "state": job_state_output(job.state()),
        "target_executor_principal_id": job.target_executor_principal_id(),
        "updated_at_millis": job.updated_at_millis(),
        "version": job.version(),
    })
}

fn job_state_output(state: JobState) -> Value {
    match state {
        JobState::Requested => json!({ "kind": "requested" }),
        JobState::Accepted {
            executor_principal_id,
        } => {
            json!({ "executor_principal_id": executor_principal_id, "kind": "accepted" })
        }
        JobState::InProgress {
            executor_principal_id,
        } => {
            json!({ "executor_principal_id": executor_principal_id, "kind": "in_progress" })
        }
        JobState::Completed {
            executor_principal_id,
        } => {
            json!({ "executor_principal_id": executor_principal_id, "kind": "completed" })
        }
        JobState::Cancelled {
            executor_principal_id,
            cancelled_by_principal_id,
        } => json!({
            "cancelled_by_principal_id": cancelled_by_principal_id,
            "executor_principal_id": executor_principal_id,
            "kind": "cancelled",
        }),
        JobState::Failed {
            executor_principal_id,
            reported_by_principal_id,
        } => json!({
            "executor_principal_id": executor_principal_id,
            "kind": "failed",
            "reported_by_principal_id": reported_by_principal_id,
        }),
    }
}

fn receipt_output(verb: AgentsCliVerb, receipt: AgentWriteReceipt) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": receipt.resource_id,
        "version": receipt.version,
    })
}

fn valid_resource_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_RESOURCE_ID_BYTES
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use collaboration_domain::{
        AggregateId, JobCommandKind, NostrEventId, NostrPublicKey, PrincipalId,
        PrivateAgentProjectionState, PrivateAgentReference,
    };
    use nostr_compat::buzz_nips::identity::IdentityArchiveRequest;
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<AgentsCliCommand>>,
        result: RefCell<Option<Result<AgentsCliOutcome, AgentsCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<AgentsCliOutcome, AgentsCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl AgentsCliExecutor for TestExecutor {
        fn execute(&self, command: AgentsCliCommand) -> Result<AgentsCliOutcome, AgentsCliError> {
            self.command.replace(Some(command));
            self.result.borrow_mut().take().expect("called once")
        }
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(2))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn compat_public_key(byte: u8) -> PublicKey {
        PublicKey::from_bytes([byte; 32])
    }

    fn domain_public_key(byte: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([byte; 32])
    }

    fn receipt(resource_id: &str, version: Option<AggregateVersion>) -> AgentWriteReceipt {
        AgentWriteReceipt::new(operation_id(), resource_id, version).expect("receipt")
    }

    fn private_catalog(secret: &str) -> PrivateAgentCatalogProjectionSource {
        PrivateAgentCatalogProjectionSource {
            owner_public_key: domain_public_key(3),
            personas: Vec::new(),
            teams: Vec::new(),
            managed_agents: vec![PrivateAgentProjectionState {
                owner_public_key: domain_public_key(3),
                agent_public_key: domain_public_key(4),
                generation: 1,
                current_event_id: NostrEventId::from_bytes([5; 32]),
                environment_references: vec![PrivateAgentReference::new(secret).expect("env")],
                credential_references: vec![
                    PrivateAgentReference::new("credential-secret").expect("credential"),
                ],
                local_source_path: None,
                backend_reference: None,
                respond_to_allowlist: Vec::new(),
            }],
        }
    }

    fn request_job() -> JobCommand {
        JobCommand::new(
            JobIdentity::new(community_id(), AggregateId::from_uuid(Uuid::from_u128(6)))
                .expect("identity"),
            operation_id(),
            AggregateVersion::FIRST,
            10,
            JobCommandKind::Request {
                requester_principal_id: principal_id(7),
                target_executor_principal_id: principal_id(8),
            },
        )
        .expect("job request")
    }

    #[test]
    fn lifecycle_requests_forward_private_state_without_diagnostic_or_output_leaks() {
        let secret = "PRIVATE-ENVIRONMENT-REFERENCE";
        let command = AgentsCliCommand::CreateAgentDraft {
            community_id: community_id(),
            private_catalog: private_catalog(secret),
            operation_id: operation_id(),
        };
        assert!(!format!("{command:?}").contains(secret));
        let executor = TestExecutor::returning(Ok(AgentsCliOutcome::Applied(receipt(
            "agent-draft",
            Some(AggregateVersion::FIRST),
        ))));
        let output = execute_agents_command(&executor, command);
        assert_eq!(output.exit_code, 0);
        assert!(!output.stdout.contains(secret));
        assert!(!output.stdout.contains("credential-secret"));
        assert!(matches!(
            executor.command.take(),
            Some(AgentsCliCommand::CreateAgentDraft { .. })
        ));
    }

    #[test]
    fn persona_inspection_is_metadata_only() {
        let prompt = "PRIVATE-PERSONA-PROMPT";
        let environment = "PRIVATE-MCP-ENVIRONMENT";
        let input = PersonaPackInput {
            manifest_json: PrivateText::new("{}", false).expect("manifest"),
            files: BTreeMap::from([
                (
                    "persona.md".into(),
                    PrivateText::new(prompt, false).expect("prompt"),
                ),
                (
                    "mcp.json".into(),
                    PrivateText::new(environment, false).expect("mcp"),
                ),
            ]),
        };
        let executor =
            TestExecutor::returning(Ok(AgentsCliOutcome::PersonaPack(PersonaPackSummary {
                id: "pack".into(),
                name: "Pack".into(),
                version: "1".into(),
                description: None,
                personas: vec![PersonaSummary {
                    name: "helper".into(),
                    display_name: "Helper".into(),
                    description: "Helps".into(),
                    model: Some("model".into()),
                    runtime: Some("acp".into()),
                    skills: vec!["edit".into()],
                    mcp_server_count: 1,
                }],
            })));
        let output = execute_agents_command(&executor, AgentsCliCommand::InspectPersonaPack(input));
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("Helper"));
        assert!(!output.stdout.contains(prompt));
        assert!(!output.stdout.contains(environment));
    }

    #[test]
    fn memory_values_are_exposed_only_by_explicit_get() {
        let secret = "PRIVATE-MEMORY-VALUE";
        let slug = MemorySlug::new("mem/notes").expect("slug");
        let event_id = EventId::from_bytes([9; 32]);
        let listing = execute_agents_command(
            &TestExecutor::returning(Ok(AgentsCliOutcome::Memories(vec![MemoryListing {
                slug: slug.clone(),
                event_id,
                created_at: 10,
            }]))),
            AgentsCliCommand::ListMemories {
                owner_public_key: compat_public_key(3),
                agent_public_key: compat_public_key(4),
            },
        );
        assert!(!listing.stdout.contains(secret));

        let get = execute_agents_command(
            &TestExecutor::returning(Ok(AgentsCliOutcome::Memory(MemoryValue {
                slug: slug.clone(),
                event_id,
                value: PrivateText::new(secret, false).expect("value"),
            }))),
            AgentsCliCommand::GetMemory {
                owner_public_key: compat_public_key(3),
                agent_public_key: compat_public_key(4),
                slug: slug.clone(),
            },
        );
        assert!(get.stdout.contains(secret));

        let set = AgentsCliCommand::SetMemory {
            owner_public_key: compat_public_key(3),
            agent_public_key: compat_public_key(4),
            slug,
            value: PrivateText::new(secret, false).expect("value"),
            operation_id: operation_id(),
        };
        assert!(!format!("{set:?}").contains(secret));
    }

    #[test]
    fn snapshot_output_contains_integrity_metadata_but_no_private_documents() {
        let hash = ContentHash::parse("a".repeat(64)).expect("hash");
        let output = execute_agents_command(
            &TestExecutor::returning(Ok(AgentsCliOutcome::Snapshots(vec![SnapshotSummary {
                owner_public_key: compat_public_key(3),
                agent_public_key: compat_public_key(4),
                snapshot_id: "snapshot-1".into(),
                predecessor_snapshot_id: None,
                source_generation: 2,
                source_event_id: "event-2".into(),
                created_at: 10,
                persona_sha256: hash.clone(),
                team_sha256: None,
                runtime_sha256: hash,
            }]))),
            AgentsCliCommand::ListSnapshots {
                owner_public_key: compat_public_key(3),
                agent_public_key: compat_public_key(4),
            },
        );
        assert!(output.stdout.contains("persona_sha256"));
        assert!(output.stdout.contains("runtime_sha256"));
        for private_field in ["persona_json", "team_json", "runtime_json", "credential"] {
            assert!(!output.stdout.contains(private_field));
        }
    }

    #[test]
    fn delegation_and_cancellation_preserve_canonical_job_commands() {
        let request = request_job();
        let parent = JobIdentity::new(community_id(), AggregateId::from_uuid(Uuid::from_u128(10)))
            .expect("parent");
        let command = AgentsCliCommand::DelegateJob {
            command: request.clone(),
            ancestry: vec![parent],
        };
        let executor = TestExecutor::returning(Ok(AgentsCliOutcome::Applied(receipt(
            "job-child",
            Some(AggregateVersion::FIRST),
        ))));
        let output = execute_agents_command(&executor, command);
        assert_eq!(output.exit_code, 0);
        assert!(matches!(
            executor.command.take(),
            Some(AgentsCliCommand::DelegateJob { command, ancestry })
                if command == request && ancestry == vec![parent]
        ));

        let cancel = JobCommand::new(
            request.identity(),
            OperationId::from_uuid(Uuid::from_u128(11)),
            AggregateVersion::new(2).expect("version"),
            11,
            JobCommandKind::Cancel {
                actor_principal_id: principal_id(7),
            },
        )
        .expect("cancel");
        let output = execute_agents_command(
            &TestExecutor::returning(Ok(AgentsCliOutcome::Applied(receipt(
                "job-child",
                Some(AggregateVersion::new(2).expect("version")),
            )))),
            AgentsCliCommand::CancelJob { command: cancel },
        );
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("agents.job.cancel"));
    }

    #[test]
    fn archive_preflight_and_stable_error_matrix_fail_closed() {
        let invalid_archive = IdentityArchiveRequest {
            action: ArchiveAction::Unarchive,
            target: compat_public_key(3),
            reason: None,
            replaced_by: None,
            attestation: None,
            content: String::new(),
        };
        let executor =
            TestExecutor::returning(Ok(AgentsCliOutcome::Archived(ArchivedIdentitySnapshot {
                identities: Vec::new(),
            })));
        let output = execute_agents_command(
            &executor,
            AgentsCliCommand::ArchiveIdentity {
                community_id: community_id(),
                request: invalid_archive,
                operation_id: operation_id(),
            },
        );
        assert_eq!(output.exit_code, 1);
        assert!(executor.command.borrow().is_none());

        let cases = [
            (AgentsCliError::InvalidRequest, "user_error", 1, false),
            (AgentsCliError::NotFound, "not_found", 1, false),
            (AgentsCliError::Unavailable, "network_error", 2, true),
            (AgentsCliError::PartialFailure, "delivery_unknown", 2, false),
            (AgentsCliError::PermissionDenied, "auth_error", 3, false),
            (AgentsCliError::Unexpected, "error", 4, false),
            (AgentsCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_agents_command(
                &TestExecutor::returning(Err(error)),
                AgentsCliCommand::ArchivedIdentities {
                    community_id: community_id(),
                },
            );
            assert_eq!(output.exit_code, exit_code);
            let envelope: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(envelope["error"], category);
            assert_eq!(envelope["retryable"], retryable);
        }

        let mismatch = execute_agents_command(
            &TestExecutor::returning(Ok(AgentsCliOutcome::PersonaPackValid)),
            AgentsCliCommand::ArchivedIdentities {
                community_id: community_id(),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
    }
}
