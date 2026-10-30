use std::fmt;

use collab::git::{
    object_store::GitRefManifest,
    repository_registry::{
        HostedAuthority, HostedRepository, HostedRepositoryLifecycle, RepositoryCoordinate,
        RepositoryPermission,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, NostrEventId, OperationId, ProjectChannelReference,
    ProjectDescription, ProjectDisplayName, ProjectGroup, ProjectGroupRecordFields, ProjectSlug,
    ProjectVisibility, RepositoryCoordinate as ProjectRepositoryCoordinate,
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectMutation {
    pub community_id: CommunityId,
    pub slug: ProjectSlug,
    pub name: Option<ProjectDisplayName>,
    pub description: Option<ProjectDescription>,
    pub repositories: Vec<ProjectRepositoryCoordinate>,
    pub channel_reference: Option<ProjectChannelReference>,
    pub visibility: ProjectVisibility,
    pub expected_source_event_id: Option<NostrEventId>,
    pub operation_id: OperationId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CloneCoordinates {
    pub https_url: Option<String>,
    pub ssh_url: Option<String>,
    pub nostr_coordinate: String,
}

impl CloneCoordinates {
    pub fn new(
        https_url: Option<String>,
        ssh_url: Option<String>,
        nostr_coordinate: String,
    ) -> Result<Self, GitCliError> {
        let coordinates = Self {
            https_url,
            ssh_url,
            nostr_coordinate,
        };
        for value in coordinates
            .https_url
            .iter()
            .chain(coordinates.ssh_url.iter())
            .chain(std::iter::once(&coordinates.nostr_coordinate))
        {
            if value.is_empty() || value.len() > 2_048 || value.chars().any(char::is_control) {
                return Err(GitCliError::InvalidRequest);
            }
        }
        Ok(coordinates)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum GitCliCommand {
    GetProject {
        community_id: CommunityId,
        slug: ProjectSlug,
    },
    ListProjects {
        community_id: CommunityId,
    },
    CreateProject(ProjectMutation),
    UpdateProject(ProjectMutation),
    DeleteProject {
        community_id: CommunityId,
        slug: ProjectSlug,
        expected_source_event_id: NostrEventId,
        operation_id: OperationId,
    },
    GetRepository {
        community_id: CommunityId,
        repository_id: AggregateId,
    },
    ListRepositories {
        community_id: CommunityId,
    },
    HostRepository {
        community_id: CommunityId,
        coordinate: RepositoryCoordinate,
        authority: HostedAuthority,
        operation_id: OperationId,
    },
    RenameRepository {
        community_id: CommunityId,
        repository_id: AggregateId,
        discriminator: String,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    ArchiveRepository {
        community_id: CommunityId,
        repository_id: AggregateId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    GrantPermission {
        community_id: CommunityId,
        repository_id: AggregateId,
        principal_id: collaboration_domain::PrincipalId,
        permission: RepositoryPermission,
        operation_id: OperationId,
    },
    RevokePermission {
        community_id: CommunityId,
        repository_id: AggregateId,
        principal_id: collaboration_domain::PrincipalId,
        permission: RepositoryPermission,
        operation_id: OperationId,
    },
    Refs {
        community_id: CommunityId,
        repository_id: AggregateId,
    },
    Clone {
        community_id: CommunityId,
        repository_id: AggregateId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GitCliVerb {
    GetProject,
    ListProjects,
    CreateProject,
    UpdateProject,
    DeleteProject,
    GetRepository,
    ListRepositories,
    HostRepository,
    RenameRepository,
    ArchiveRepository,
    GrantPermission,
    RevokePermission,
    Refs,
    Clone,
}

impl GitCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::GetProject => "project.get",
            Self::ListProjects => "project.list",
            Self::CreateProject => "project.create",
            Self::UpdateProject => "project.update",
            Self::DeleteProject => "project.delete",
            Self::GetRepository => "repository.get",
            Self::ListRepositories => "repository.list",
            Self::HostRepository => "repository.host",
            Self::RenameRepository => "repository.rename",
            Self::ArchiveRepository => "repository.archive",
            Self::GrantPermission => "repository.grant",
            Self::RevokePermission => "repository.revoke",
            Self::Refs => "repository.refs",
            Self::Clone => "repository.clone",
        }
    }
}

impl GitCliCommand {
    const fn verb(&self) -> GitCliVerb {
        match self {
            Self::GetProject { .. } => GitCliVerb::GetProject,
            Self::ListProjects { .. } => GitCliVerb::ListProjects,
            Self::CreateProject(_) => GitCliVerb::CreateProject,
            Self::UpdateProject(_) => GitCliVerb::UpdateProject,
            Self::DeleteProject { .. } => GitCliVerb::DeleteProject,
            Self::GetRepository { .. } => GitCliVerb::GetRepository,
            Self::ListRepositories { .. } => GitCliVerb::ListRepositories,
            Self::HostRepository { .. } => GitCliVerb::HostRepository,
            Self::RenameRepository { .. } => GitCliVerb::RenameRepository,
            Self::ArchiveRepository { .. } => GitCliVerb::ArchiveRepository,
            Self::GrantPermission { .. } => GitCliVerb::GrantPermission,
            Self::RevokePermission { .. } => GitCliVerb::RevokePermission,
            Self::Refs { .. } => GitCliVerb::Refs,
            Self::Clone { .. } => GitCliVerb::Clone,
        }
    }
}

impl fmt::Debug for GitCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GitWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: AggregateId,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GitCliOutcome {
    Project(ProjectGroupRecordFields),
    Projects(Vec<ProjectGroupRecordFields>),
    Repository(HostedRepository),
    Repositories(Vec<HostedRepository>),
    Refs(GitRefManifest),
    Clone(CloneCoordinates),
    Applied(GitWriteReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    PermissionDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl GitCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "git_cli_invalid_request",
            Self::NotFound => "git_cli_not_found",
            Self::Unavailable => "git_cli_unavailable",
            Self::PermissionDenied => "git_cli_permission_denied",
            Self::PartialFailure => "git_cli_completion_unknown",
            Self::Unexpected => "git_cli_unexpected_response",
            Self::Conflict => "git_cli_stale_version",
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

pub trait GitCliExecutor {
    fn execute(&self, command: GitCliCommand) -> Result<GitCliOutcome, GitCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_git_command(
    executor: &impl GitCliExecutor,
    command: GitCliCommand,
) -> GitCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Some(output) => GitCliExecution {
                stdout: format!("{output}\n"),
                stderr: String::new(),
                exit_code: 0,
            },
            None => error_output(verb, GitCliError::Unexpected),
        },
        Err(error) => error_output(verb, error),
    }
}

fn error_output(verb: GitCliVerb, error: GitCliError) -> GitCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    GitCliExecution {
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

fn success_output(verb: GitCliVerb, outcome: GitCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (GitCliVerb::GetProject, GitCliOutcome::Project(project)) => project_output(&project, verb),
        (GitCliVerb::ListProjects, GitCliOutcome::Projects(projects)) => {
            let projects = projects
                .iter()
                .map(|project| project_output(project, GitCliVerb::GetProject))
                .collect::<Option<Vec<_>>>()?;
            Some(json!({ "command": verb.as_str(), "ok": true, "projects": projects }))
        }
        (GitCliVerb::GetRepository, GitCliOutcome::Repository(repository)) => {
            Some(repository_output(&repository, verb))
        }
        (GitCliVerb::ListRepositories, GitCliOutcome::Repositories(repositories)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "repositories": repositories.iter().map(|repository| repository_output(repository, GitCliVerb::GetRepository)).collect::<Vec<_>>(),
        })),
        (GitCliVerb::Refs, GitCliOutcome::Refs(manifest)) => Some(refs_output(verb, &manifest)),
        (GitCliVerb::Clone, GitCliOutcome::Clone(coordinates)) => Some(json!({
            "command": verb.as_str(),
            "https_url": coordinates.https_url,
            "nostr_coordinate": coordinates.nostr_coordinate,
            "ok": true,
            "ssh_url": coordinates.ssh_url,
        })),
        (
            GitCliVerb::CreateProject
            | GitCliVerb::UpdateProject
            | GitCliVerb::DeleteProject
            | GitCliVerb::HostRepository
            | GitCliVerb::RenameRepository
            | GitCliVerb::ArchiveRepository
            | GitCliVerb::GrantPermission
            | GitCliVerb::RevokePermission,
            GitCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn project_output(fields: &ProjectGroupRecordFields, verb: GitCliVerb) -> Option<Value> {
    let project = ProjectGroup::from_record(fields.clone()).ok()?;
    let fields = project.fields();
    Some(json!({
        "channel_reference": fields.channel_reference.as_ref().map(ProjectChannelReference::as_str),
        "command": verb.as_str(),
        "description": fields.description.as_ref().map(ProjectDescription::as_str),
        "display_name": project.display_name(),
        "name": fields.name.as_ref().map(ProjectDisplayName::as_str),
        "ok": true,
        "repositories": fields.repositories.iter().map(|repository| json!({
            "coordinate": repository.coordinate(),
            "relay_hint": repository.relay_hint(),
        })).collect::<Vec<_>>(),
        "signer_public_key": hex_bytes(fields.signer_public_key.as_bytes()),
        "slug": fields.slug.as_str(),
        "source_event_id": hex_bytes(fields.source.event_id.as_bytes()),
        "visibility": project_visibility(fields.visibility),
    }))
}

const fn project_visibility(visibility: ProjectVisibility) -> &'static str {
    match visibility {
        ProjectVisibility::Listed => "listed",
        ProjectVisibility::Unlisted => "unlisted",
    }
}

fn repository_output(repository: &HostedRepository, verb: GitCliVerb) -> Value {
    json!({
        "authority": authority_output(&repository.authority),
        "command": verb.as_str(),
        "community_id": repository.community_id,
        "coordinate": {
            "discriminator": repository.coordinate.discriminator,
            "owner_public_key": hex_bytes(repository.coordinate.owner_public_key.as_bytes()),
        },
        "lifecycle": repository_lifecycle(repository.lifecycle),
        "ok": true,
        "repository_id": repository.repository_id,
        "version": repository.authority_version,
    })
}

fn authority_output(authority: &HostedAuthority) -> Value {
    match authority {
        HostedAuthority::SimHostedNip34 { storage_handle_id } => json!({
            "kind": "zed_hosted_nip34",
            "storage_handle_id": storage_handle_id,
        }),
        HostedAuthority::ExternalProvider(provider) => json!({
            "instance": provider.provider_instance,
            "kind": "external_provider",
            "owner": provider.owner,
            "provider": provider.provider_kind,
            "repository": provider.repository,
        }),
    }
}

const fn repository_lifecycle(lifecycle: HostedRepositoryLifecycle) -> &'static str {
    match lifecycle {
        HostedRepositoryLifecycle::Active => "active",
        HostedRepositoryLifecycle::Archived => "archived",
    }
}

fn refs_output(verb: GitCliVerb, manifest: &GitRefManifest) -> Value {
    json!({
        "command": verb.as_str(),
        "head": manifest.head().map(|head| head.as_str()),
        "ok": true,
        "refs": manifest.refs().iter().map(|(name, object_id)| json!({
            "name": name.as_str(),
            "object_id": object_id.as_str(),
        })).collect::<Vec<_>>(),
    })
}

fn write_output(verb: GitCliVerb, receipt: GitWriteReceipt) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": receipt.resource_id,
        "version": receipt.version,
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::{BTreeMap, BTreeSet},
    };

    use collab::git::{
        object_store::{GitObjectId, GitRefName},
        repository_registry::ExternalProviderCoordinate,
    };
    use collaboration_domain::{
        MessageSource, NostrPublicKey, PrincipalId, Provenance, SourceRecordId, SourceSystem,
    };
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<GitCliCommand>>,
        result: RefCell<Option<Result<GitCliOutcome, GitCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<GitCliOutcome, GitCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl GitCliExecutor for TestExecutor {
        fn execute(&self, command: GitCliCommand) -> Result<GitCliOutcome, GitCliError> {
            self.command.replace(Some(command));
            self.result.borrow_mut().take().expect("called once")
        }
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }
    fn repository_id() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(2))
    }
    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(3))
    }
    fn public_key() -> NostrPublicKey {
        NostrPublicKey::from_bytes([7; 32])
    }

    fn project() -> ProjectGroupRecordFields {
        ProjectGroupRecordFields {
            signer_public_key: public_key(),
            source: MessageSource {
                event_id: NostrEventId::from_bytes([8; 32]),
                event_created_at: 10,
            },
            slug: ProjectSlug::new("zed").expect("slug"),
            name: Some(ProjectDisplayName::new("Zed").expect("name")),
            description: Some(ProjectDescription::new("Editor").expect("description")),
            repositories: vec![
                ProjectRepositoryCoordinate::parse(
                    &format!("30617:{}:zed", "07".repeat(32)),
                    Some("wss://relay.example.com".into()),
                )
                .expect("coordinate"),
            ],
            channel_reference: Some(ProjectChannelReference::new("dev").expect("channel")),
            visibility: ProjectVisibility::Listed,
        }
    }

    fn hosted(authority: HostedAuthority) -> HostedRepository {
        HostedRepository {
            community_id: community_id(),
            repository_id: repository_id(),
            coordinate: RepositoryCoordinate::new(public_key(), "zed").expect("coordinate"),
            authority,
            authority_version: AggregateVersion::FIRST,
            lifecycle: HostedRepositoryLifecycle::Active,
            provenance: Provenance::new(
                SourceSystem::Zed,
                SourceRecordId::new("repository").expect("source"),
                10,
            ),
            archived_at_millis: None,
            created_at_millis: 10,
            updated_at_millis: 10,
        }
    }

    #[test]
    fn project_grouping_output_preserves_members_and_identity() {
        let output = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Project(project()))),
            GitCliCommand::GetProject {
                community_id: community_id(),
                slug: ProjectSlug::new("zed").expect("slug"),
            },
        );
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("\"slug\":\"zed\""));
        assert!(output.stdout.contains("30617:"));
        assert!(output.stdout.contains("wss://relay.example.com"));
    }

    #[test]
    fn clone_coordinates_and_refs_have_golden_shapes() {
        let coordinates = CloneCoordinates::new(
            Some("https://git.example.com/zed.git".into()),
            Some("git@git.example.com:zed.git".into()),
            format!("30617:{}:zed", "07".repeat(32)),
        )
        .expect("clone coordinates");
        let clone = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Clone(coordinates))),
            GitCliCommand::Clone {
                community_id: community_id(),
                repository_id: repository_id(),
            },
        );
        assert!(clone.stdout.contains("https://git.example.com/zed.git"));
        assert!(clone.stdout.contains("git@git.example.com:zed.git"));

        let ref_name = GitRefName::parse("refs/heads/main").expect("ref");
        let object_id = GitObjectId::parse("a".repeat(40)).expect("object id");
        let manifest = GitRefManifest::new(
            Some(ref_name.clone()),
            BTreeMap::from([(ref_name, object_id)]),
            BTreeSet::new(),
            None,
        );
        let refs = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Refs(manifest))),
            GitCliCommand::Refs {
                community_id: community_id(),
                repository_id: repository_id(),
            },
        );
        assert!(refs.stdout.contains("refs/heads/main"));
        assert!(refs.stdout.contains(&"a".repeat(40)));
    }

    #[test]
    fn hosting_output_distinguishes_zed_and_external_authority() {
        let internal = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Repository(hosted(
                HostedAuthority::SimHostedNip34 {
                    storage_handle_id: Uuid::from_u128(4),
                },
            )))),
            GitCliCommand::GetRepository {
                community_id: community_id(),
                repository_id: repository_id(),
            },
        );
        assert!(internal.stdout.contains("zed_hosted_nip34"));

        let external = hosted(HostedAuthority::ExternalProvider(
            ExternalProviderCoordinate::new("github", "github.com", "zed-industries", "zed")
                .expect("provider"),
        ));
        let output = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Repository(external))),
            GitCliCommand::GetRepository {
                community_id: community_id(),
                repository_id: repository_id(),
            },
        );
        assert!(output.stdout.contains("external_provider"));
        assert!(output.stdout.contains("github.com"));
    }

    #[test]
    fn permission_mutations_forward_exact_principal_permission_and_operation() {
        let command = GitCliCommand::GrantPermission {
            community_id: community_id(),
            repository_id: repository_id(),
            principal_id: PrincipalId::from_uuid(Uuid::from_u128(5)),
            permission: RepositoryPermission::Write,
            operation_id: operation_id(),
        };
        let executor = TestExecutor::returning(Ok(GitCliOutcome::Applied(GitWriteReceipt {
            operation_id: operation_id(),
            resource_id: repository_id(),
            version: AggregateVersion::FIRST,
        })));
        let output = execute_git_command(&executor, command.clone());
        assert_eq!(executor.command.take(), Some(command));
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("repository.grant"));
    }

    #[test]
    fn permission_errors_and_invalid_outcomes_use_stable_exit_contracts() {
        let cases = [
            (GitCliError::InvalidRequest, "user_error", 1, false),
            (GitCliError::NotFound, "not_found", 1, false),
            (GitCliError::Unavailable, "network_error", 2, true),
            (GitCliError::PartialFailure, "delivery_unknown", 2, false),
            (GitCliError::PermissionDenied, "auth_error", 3, false),
            (GitCliError::Unexpected, "error", 4, false),
            (GitCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_git_command(
                &TestExecutor::returning(Err(error)),
                GitCliCommand::GetRepository {
                    community_id: community_id(),
                    repository_id: repository_id(),
                },
            );
            assert_eq!(output.exit_code, exit_code);
            let envelope: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(envelope["error"], category);
            assert_eq!(envelope["retryable"], retryable);
        }
        let mismatch = execute_git_command(
            &TestExecutor::returning(Ok(GitCliOutcome::Projects(Vec::new()))),
            GitCliCommand::Clone {
                community_id: community_id(),
                repository_id: repository_id(),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
        assert!(mismatch.stderr.contains("git_cli_unexpected_response"));
    }
}
