use std::{fmt, num::NonZeroU32};

use collaboration_domain::{
    AggregateId, AggregateVersion, ChannelCreateFields, ChannelDescription, ChannelInviteTarget,
    ChannelLifecycleState, ChannelMetadataRecordFields, ChannelMetadataText, ChannelName,
    ChannelRecordFields, ChannelTemplate, ChannelType, ChannelVisibility, CommunityId, InviteId,
    MembershipCreateFields, MembershipRecordFields, MembershipRole, MembershipScope,
    MembershipStatus, OperationId, PrincipalId,
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelUpdateRequest {
    name: Option<ChannelName>,
    description: Option<Option<ChannelDescription>>,
    visibility: Option<ChannelVisibility>,
    ttl_seconds: Option<Option<NonZeroU32>>,
}

impl ChannelUpdateRequest {
    pub fn new(
        name: Option<ChannelName>,
        description: Option<Option<ChannelDescription>>,
        visibility: Option<ChannelVisibility>,
        ttl_seconds: Option<Option<NonZeroU32>>,
    ) -> Result<Self, ChannelCliError> {
        if name.is_none() && description.is_none() && visibility.is_none() && ttl_seconds.is_none()
        {
            return Err(ChannelCliError::InvalidRequest);
        }
        Ok(Self {
            name,
            description,
            visibility,
            ttl_seconds,
        })
    }

    pub fn name(&self) -> Option<&ChannelName> {
        self.name.as_ref()
    }

    pub fn description(&self) -> Option<Option<&ChannelDescription>> {
        self.description.as_ref().map(Option::as_ref)
    }

    pub const fn visibility(&self) -> Option<ChannelVisibility> {
        self.visibility
    }

    pub const fn ttl_seconds(&self) -> Option<Option<NonZeroU32>> {
        self.ttl_seconds
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelInviteCreateRequest {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub role: MembershipRole,
    pub max_uses: Option<NonZeroU32>,
    pub expires_at_millis: u64,
    pub operation_id: OperationId,
}

impl ChannelInviteCreateRequest {
    pub const fn target(&self) -> ChannelInviteTarget {
        ChannelInviteTarget::Channel(self.channel_id)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum ChannelCliCommand {
    Get {
        community_id: CommunityId,
        channel_id: AggregateId,
    },
    List {
        community_id: CommunityId,
        visibility: Option<ChannelVisibility>,
        membership_only: bool,
        limit: NonZeroU32,
    },
    Create {
        fields: ChannelCreateFields,
        operation_id: OperationId,
    },
    Update {
        community_id: CommunityId,
        channel_id: AggregateId,
        expected_version: AggregateVersion,
        update: ChannelUpdateRequest,
        operation_id: OperationId,
    },
    Archive {
        community_id: CommunityId,
        channel_id: AggregateId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    Restore {
        community_id: CommunityId,
        channel_id: AggregateId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    ListMembers {
        community_id: CommunityId,
        channel_id: AggregateId,
    },
    AddMember {
        fields: MembershipCreateFields,
        operation_id: OperationId,
    },
    SetMemberRole {
        community_id: CommunityId,
        channel_id: AggregateId,
        principal_id: PrincipalId,
        expected_version: AggregateVersion,
        role: MembershipRole,
        operation_id: OperationId,
    },
    RemoveMember {
        community_id: CommunityId,
        channel_id: AggregateId,
        principal_id: PrincipalId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    CreateInvite(ChannelInviteCreateRequest),
    RedeemInvite {
        community_id: CommunityId,
        bearer_token: String,
        operation_id: OperationId,
    },
    RevokeInvite {
        community_id: CommunityId,
        channel_id: AggregateId,
        invite_id: InviteId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    CreateFromTemplate {
        fields: ChannelCreateFields,
        template: ChannelTemplate,
        operation_id: OperationId,
    },
    SetTopic {
        community_id: CommunityId,
        channel_id: AggregateId,
        expected_version: AggregateVersion,
        topic: Option<ChannelMetadataText>,
        updated_at_millis: u64,
        operation_id: OperationId,
    },
    GetCanvas {
        community_id: CommunityId,
        channel_id: AggregateId,
    },
    SetCanvas {
        community_id: CommunityId,
        channel_id: AggregateId,
        expected_version: AggregateVersion,
        canvas: Option<ChannelMetadataText>,
        updated_at_millis: u64,
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChannelCliVerb {
    Get,
    List,
    Create,
    Update,
    Archive,
    Restore,
    ListMembers,
    AddMember,
    SetMemberRole,
    RemoveMember,
    CreateInvite,
    RedeemInvite,
    RevokeInvite,
    CreateFromTemplate,
    SetTopic,
    GetCanvas,
    SetCanvas,
}

impl ChannelCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "channel.get",
            Self::List => "channel.list",
            Self::Create => "channel.create",
            Self::Update => "channel.update",
            Self::Archive => "channel.archive",
            Self::Restore => "channel.restore",
            Self::ListMembers => "channel.member.list",
            Self::AddMember => "channel.member.add",
            Self::SetMemberRole => "channel.member.set_role",
            Self::RemoveMember => "channel.member.remove",
            Self::CreateInvite => "channel.invite.create",
            Self::RedeemInvite => "channel.invite.redeem",
            Self::RevokeInvite => "channel.invite.revoke",
            Self::CreateFromTemplate => "channel.template.create",
            Self::SetTopic => "channel.topic.set",
            Self::GetCanvas => "channel.canvas.get",
            Self::SetCanvas => "channel.canvas.set",
        }
    }
}

impl ChannelCliCommand {
    const fn verb(&self) -> ChannelCliVerb {
        match self {
            Self::Get { .. } => ChannelCliVerb::Get,
            Self::List { .. } => ChannelCliVerb::List,
            Self::Create { .. } => ChannelCliVerb::Create,
            Self::Update { .. } => ChannelCliVerb::Update,
            Self::Archive { .. } => ChannelCliVerb::Archive,
            Self::Restore { .. } => ChannelCliVerb::Restore,
            Self::ListMembers { .. } => ChannelCliVerb::ListMembers,
            Self::AddMember { .. } => ChannelCliVerb::AddMember,
            Self::SetMemberRole { .. } => ChannelCliVerb::SetMemberRole,
            Self::RemoveMember { .. } => ChannelCliVerb::RemoveMember,
            Self::CreateInvite(_) => ChannelCliVerb::CreateInvite,
            Self::RedeemInvite { .. } => ChannelCliVerb::RedeemInvite,
            Self::RevokeInvite { .. } => ChannelCliVerb::RevokeInvite,
            Self::CreateFromTemplate { .. } => ChannelCliVerb::CreateFromTemplate,
            Self::SetTopic { .. } => ChannelCliVerb::SetTopic,
            Self::GetCanvas { .. } => ChannelCliVerb::GetCanvas,
            Self::SetCanvas { .. } => ChannelCliVerb::SetCanvas,
        }
    }
}

impl fmt::Debug for ChannelCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChannelCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelResourceId {
    Channel(AggregateId),
    Membership(PrincipalId),
    Invite(InviteId),
    Metadata(AggregateId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: ChannelResourceId,
    pub version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct ChannelInviteReceipt {
    pub write: ChannelWriteReceipt,
    pub bearer_token: String,
}

impl fmt::Debug for ChannelInviteReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChannelInviteReceipt")
            .field("write", &self.write)
            .field("bearer_token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelTemplateReceipt {
    pub operation_id: OperationId,
    pub channel: ChannelRecordFields,
    pub template_id: AggregateId,
    pub metadata_version: Option<AggregateVersion>,
    pub members_added: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChannelCliOutcome {
    Channel(ChannelRecordFields),
    Channels(Vec<ChannelRecordFields>),
    Members(Vec<MembershipRecordFields>),
    InviteCreated(ChannelInviteReceipt),
    TemplateApplied(ChannelTemplateReceipt),
    Canvas(ChannelMetadataRecordFields),
    Applied(ChannelWriteReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    AuthorizationDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl ChannelCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "channel_cli_invalid_request",
            Self::NotFound => "channel_cli_not_found",
            Self::Unavailable => "channel_cli_unavailable",
            Self::AuthorizationDenied => "channel_cli_authorization_denied",
            Self::PartialFailure => "channel_cli_completion_unknown",
            Self::Unexpected => "channel_cli_unexpected_response",
            Self::Conflict => "channel_cli_stale_version",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::AuthorizationDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait ChannelCliExecutor {
    fn execute(&self, command: ChannelCliCommand) -> Result<ChannelCliOutcome, ChannelCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_channel_command(
    executor: &impl ChannelCliExecutor,
    command: ChannelCliCommand,
) -> ChannelCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => success_output(verb, outcome)
            .map(ChannelCliExecution::success)
            .unwrap_or_else(|| error_output(verb, ChannelCliError::Unexpected)),
        Err(error) => error_output(verb, error),
    }
}

impl ChannelCliExecution {
    fn success(value: Value) -> Self {
        Self {
            stdout: format!("{value}\n"),
            stderr: String::new(),
            exit_code: 0,
        }
    }

    fn failure(value: Value, exit_code: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("{value}\n"),
            exit_code,
        }
    }
}

fn success_output(verb: ChannelCliVerb, outcome: ChannelCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (ChannelCliVerb::Get, ChannelCliOutcome::Channel(channel)) => {
            Some(channel_output(verb, &channel))
        }
        (ChannelCliVerb::List, ChannelCliOutcome::Channels(channels)) => Some(json!({
            "channels": channels.iter().map(|channel| channel_output(verb, channel)).collect::<Vec<_>>(),
            "command": verb.as_str(),
            "ok": true,
        })),
        (ChannelCliVerb::ListMembers, ChannelCliOutcome::Members(members)) => Some(json!({
            "command": verb.as_str(),
            "members": members.iter().map(member_output).collect::<Vec<_>>(),
            "ok": true,
        })),
        (ChannelCliVerb::CreateInvite, ChannelCliOutcome::InviteCreated(receipt)) => {
            Some(invite_output(verb, receipt))
        }
        (ChannelCliVerb::CreateFromTemplate, ChannelCliOutcome::TemplateApplied(receipt)) => {
            Some(template_output(verb, receipt))
        }
        (ChannelCliVerb::GetCanvas, ChannelCliOutcome::Canvas(metadata)) => Some(json!({
            "canvas": metadata.canvas.as_ref().map(ChannelMetadataText::as_str),
            "channel_id": metadata.channel_id,
            "command": verb.as_str(),
            "community_id": metadata.community_id,
            "ok": true,
            "updated_at_millis": metadata.updated_at_millis,
            "updated_by_principal_id": metadata.updated_by_principal_id,
            "version": metadata.version,
        })),
        (
            ChannelCliVerb::Create
            | ChannelCliVerb::Update
            | ChannelCliVerb::Archive
            | ChannelCliVerb::Restore
            | ChannelCliVerb::AddMember
            | ChannelCliVerb::SetMemberRole
            | ChannelCliVerb::RemoveMember
            | ChannelCliVerb::RedeemInvite
            | ChannelCliVerb::RevokeInvite
            | ChannelCliVerb::SetTopic
            | ChannelCliVerb::SetCanvas,
            ChannelCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn error_output(verb: ChannelCliVerb, error: ChannelCliError) -> ChannelCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    ChannelCliExecution::failure(
        json!({
            "command": verb.as_str(),
            "error": contract.category,
            "error_code": diagnostic,
            "message": diagnostic,
            "ok": false,
            "retryable": contract.retryable,
        }),
        contract.exit_class as i32,
    )
}

fn channel_output(verb: ChannelCliVerb, channel: &ChannelRecordFields) -> Value {
    json!({
        "channel_id": channel.channel_id,
        "command": verb.as_str(),
        "community_id": channel.community_id,
        "creator_principal_id": channel.creator_principal_id,
        "description": channel.description.as_ref().map(ChannelDescription::as_str),
        "expiration": channel.expiration.map(|expiration| json!({
            "expires_at_millis": expiration.expires_at_millis,
            "ttl_seconds": expiration.ttl_seconds.get(),
        })),
        "lifecycle": lifecycle(channel.lifecycle_state),
        "name": channel.name.as_str(),
        "ok": true,
        "type": channel_type(channel.channel_type),
        "version": channel.version,
        "visibility": visibility(channel.visibility),
    })
}

fn member_output(member: &MembershipRecordFields) -> Value {
    let scope = match member.scope {
        MembershipScope::Community => json!({ "kind": "community" }),
        MembershipScope::Channel(channel_id) => {
            json!({ "channel_id": channel_id, "kind": "channel" })
        }
    };
    json!({
        "added_by_principal_id": member.added_by_principal_id,
        "community_id": member.community_id,
        "principal_id": member.principal_id,
        "role": role(member.role),
        "scope": scope,
        "status": membership_status(member.status),
        "version": member.version,
    })
}

fn invite_output(verb: ChannelCliVerb, receipt: ChannelInviteReceipt) -> Value {
    let (resource_kind, resource_id) = resource_output(receipt.write.resource_id);
    json!({
        "bearer_token": receipt.bearer_token,
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.write.operation_id,
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "version": receipt.write.version,
    })
}

fn template_output(verb: ChannelCliVerb, receipt: ChannelTemplateReceipt) -> Value {
    json!({
        "channel": channel_output(verb, &receipt.channel),
        "command": verb.as_str(),
        "members_added": receipt.members_added,
        "metadata_version": receipt.metadata_version,
        "ok": true,
        "operation_id": receipt.operation_id,
        "template_id": receipt.template_id,
    })
}

fn write_output(verb: ChannelCliVerb, receipt: ChannelWriteReceipt) -> Value {
    let (resource_kind, resource_id) = resource_output(receipt.resource_id);
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "version": receipt.version,
    })
}

fn resource_output(resource: ChannelResourceId) -> (&'static str, String) {
    match resource {
        ChannelResourceId::Channel(channel_id) => ("channel", channel_id.to_string()),
        ChannelResourceId::Membership(principal_id) => ("membership", principal_id.to_string()),
        ChannelResourceId::Invite(invite_id) => ("invite", invite_id.as_uuid().to_string()),
        ChannelResourceId::Metadata(channel_id) => ("metadata", channel_id.to_string()),
    }
}

const fn channel_type(value: ChannelType) -> &'static str {
    match value {
        ChannelType::Stream => "stream",
        ChannelType::Forum => "forum",
        ChannelType::DirectMessage => "direct_message",
        ChannelType::Workflow => "workflow",
        ChannelType::Ephemeral => "ephemeral",
        ChannelType::Huddle => "huddle",
    }
}

const fn visibility(value: ChannelVisibility) -> &'static str {
    match value {
        ChannelVisibility::Open => "open",
        ChannelVisibility::Private => "private",
    }
}

const fn lifecycle(value: ChannelLifecycleState) -> &'static str {
    match value {
        ChannelLifecycleState::Active => "active",
        ChannelLifecycleState::Archived => "archived",
        ChannelLifecycleState::Deleted => "deleted",
        ChannelLifecycleState::Expired => "expired",
    }
}

const fn role(value: MembershipRole) -> &'static str {
    match value {
        MembershipRole::Owner => "owner",
        MembershipRole::Admin => "admin",
        MembershipRole::Member => "member",
        MembershipRole::Guest => "guest",
        MembershipRole::Bot => "bot",
    }
}

const fn membership_status(value: MembershipStatus) -> &'static str {
    match value {
        MembershipStatus::Active => "active",
        MembershipStatus::Revoked => "revoked",
        MembershipStatus::Archived => "archived",
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use collaboration_domain::{
        ChannelTemplateBackend, ChannelTemplateReference, ChannelTemplateReferenceKind,
    };
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<ChannelCliCommand>>,
        result: RefCell<Option<Result<ChannelCliOutcome, ChannelCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<ChannelCliOutcome, ChannelCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl ChannelCliExecutor for TestExecutor {
        fn execute(
            &self,
            command: ChannelCliCommand,
        ) -> Result<ChannelCliOutcome, ChannelCliError> {
            self.command.replace(Some(command));
            self.result
                .borrow_mut()
                .take()
                .expect("the test executor is called once")
        }
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(20))
    }

    fn channel(channel_type: ChannelType, value: u128) -> ChannelRecordFields {
        ChannelRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(value),
            name: ChannelName::new(format!("channel-{value}")).expect("name"),
            channel_type,
            visibility: if matches!(
                channel_type,
                ChannelType::DirectMessage | ChannelType::Workflow | ChannelType::Huddle
            ) {
                ChannelVisibility::Private
            } else {
                ChannelVisibility::Open
            },
            lifecycle_state: ChannelLifecycleState::Active,
            description: Some(ChannelDescription::new("A channel").expect("description")),
            creator_principal_id: principal_id(2),
            expiration: None,
            version: AggregateVersion::FIRST,
        }
    }

    fn command_for_error() -> ChannelCliCommand {
        ChannelCliCommand::Archive {
            community_id: community_id(),
            channel_id: aggregate_id(3),
            expected_version: AggregateVersion::FIRST,
            operation_id: operation_id(),
        }
    }

    fn metadata(canvas: Option<&str>) -> ChannelMetadataRecordFields {
        ChannelMetadataRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(3),
            topic: Some(ChannelMetadataText::new("Ship it").expect("topic")),
            canvas: canvas
                .map(ChannelMetadataText::new)
                .transpose()
                .expect("canvas"),
            version: AggregateVersion::FIRST,
            updated_by_principal_id: principal_id(2),
            updated_at_millis: 10,
        }
    }

    #[test]
    fn every_channel_type_has_stable_list_output() {
        let types = [
            ChannelType::Stream,
            ChannelType::Forum,
            ChannelType::DirectMessage,
            ChannelType::Workflow,
            ChannelType::Ephemeral,
            ChannelType::Huddle,
        ];
        let channels = types
            .into_iter()
            .enumerate()
            .map(|(index, channel_type)| channel(channel_type, index as u128 + 3))
            .collect();
        let output = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::Channels(channels))),
            ChannelCliCommand::List {
                community_id: community_id(),
                visibility: None,
                membership_only: false,
                limit: NonZeroU32::new(100).expect("limit"),
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(
            value["channels"]
                .as_array()
                .expect("channel array")
                .iter()
                .map(|channel| channel["type"].as_str().expect("type"))
                .collect::<Vec<_>>(),
            [
                "stream",
                "forum",
                "direct_message",
                "workflow",
                "ephemeral",
                "huddle"
            ]
        );
        assert_eq!(output.exit_code, 0);
    }

    #[test]
    fn membership_roles_and_invite_credentials_are_stable_and_redacted() {
        let roles = [
            MembershipRole::Owner,
            MembershipRole::Admin,
            MembershipRole::Member,
            MembershipRole::Guest,
            MembershipRole::Bot,
        ];
        let members = roles
            .into_iter()
            .enumerate()
            .map(|(index, role)| MembershipRecordFields {
                community_id: community_id(),
                scope: MembershipScope::Channel(aggregate_id(3)),
                principal_id: principal_id(index as u128 + 10),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
                added_by_principal_id: Some(principal_id(2)),
            })
            .collect();
        let members_output = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::Members(members))),
            ChannelCliCommand::ListMembers {
                community_id: community_id(),
                channel_id: aggregate_id(3),
            },
        );
        let value: Value = serde_json::from_str(&members_output.stdout).expect("JSON");
        assert_eq!(
            value["members"]
                .as_array()
                .expect("members")
                .iter()
                .map(|member| member["role"].as_str().expect("role"))
                .collect::<Vec<_>>(),
            ["owner", "admin", "member", "guest", "bot"]
        );

        let token = "secret-invite-token";
        let receipt = ChannelInviteReceipt {
            write: ChannelWriteReceipt {
                operation_id: operation_id(),
                resource_id: ChannelResourceId::Invite(InviteId::from_uuid(Uuid::from_u128(30))),
                version: AggregateVersion::FIRST,
            },
            bearer_token: token.into(),
        };
        assert!(!format!("{receipt:?}").contains(token));
        let command = ChannelCliCommand::RedeemInvite {
            community_id: community_id(),
            bearer_token: token.into(),
            operation_id: operation_id(),
        };
        assert!(!format!("{command:?}").contains(token));
        let invite_executor =
            TestExecutor::returning(Ok(ChannelCliOutcome::InviteCreated(receipt)));
        let invite_output = execute_channel_command(
            &invite_executor,
            ChannelCliCommand::CreateInvite(ChannelInviteCreateRequest {
                community_id: community_id(),
                channel_id: aggregate_id(3),
                role: MembershipRole::Guest,
                max_uses: NonZeroU32::new(2),
                expires_at_millis: 60_000,
                operation_id: operation_id(),
            }),
        );
        assert!(invite_output.stdout.contains(token));
        assert!(matches!(
            invite_executor.command.borrow().as_ref(),
            Some(ChannelCliCommand::CreateInvite(request))
                if request.target() == ChannelInviteTarget::Channel(aggregate_id(3))
                    && request.role == MembershipRole::Guest
        ));
    }

    #[test]
    fn template_topic_and_canvas_use_canonical_values() {
        let template = ChannelTemplate::new(
            aggregate_id(40),
            ChannelName::new("Review").expect("name"),
            None,
            ChannelType::Forum,
            ChannelVisibility::Private,
            Some(ChannelMetadataText::new("# {channel.name}").expect("canvas")),
            vec![ChannelTemplateReference {
                kind: ChannelTemplateReferenceKind::Persona,
                id: "builtin:reviewer".into(),
                runtime: Some("acp".into()),
                model: None,
                role: Some("bot".into()),
                backend: Some(ChannelTemplateBackend::Local),
            }],
            false,
            AggregateVersion::FIRST,
        )
        .expect("template");
        let fields = ChannelCreateFields {
            community_id: community_id(),
            channel_id: aggregate_id(3),
            name: ChannelName::new("launch").expect("name"),
            channel_type: ChannelType::Forum,
            visibility: ChannelVisibility::Private,
            description: None,
            creator_principal_id: principal_id(2),
            ttl_seconds: None,
            now_millis: 10,
        };
        let output = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::TemplateApplied(
                ChannelTemplateReceipt {
                    operation_id: operation_id(),
                    channel: channel(ChannelType::Forum, 3),
                    template_id: aggregate_id(40),
                    metadata_version: Some(AggregateVersion::FIRST),
                    members_added: 1,
                },
            ))),
            ChannelCliCommand::CreateFromTemplate {
                fields,
                template,
                operation_id: operation_id(),
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["members_added"], 1);
        assert_eq!(value["channel"]["type"], "forum");

        let canvas_output = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::Canvas(metadata(Some("# launch"))))),
            ChannelCliCommand::GetCanvas {
                community_id: community_id(),
                channel_id: aggregate_id(3),
            },
        );
        assert!(canvas_output.stdout.contains("# launch"));

        let topic = ChannelMetadataText::new("Release readiness").expect("topic");
        let topic_executor =
            TestExecutor::returning(Ok(ChannelCliOutcome::Applied(ChannelWriteReceipt {
                operation_id: operation_id(),
                resource_id: ChannelResourceId::Metadata(aggregate_id(3)),
                version: AggregateVersion::FIRST,
            })));
        let topic_output = execute_channel_command(
            &topic_executor,
            ChannelCliCommand::SetTopic {
                community_id: community_id(),
                channel_id: aggregate_id(3),
                expected_version: AggregateVersion::FIRST,
                topic: Some(topic),
                updated_at_millis: 10,
                operation_id: operation_id(),
            },
        );
        assert_eq!(topic_output.exit_code, 0);
        assert!(matches!(
            topic_executor.command.borrow().as_ref(),
            Some(ChannelCliCommand::SetTopic { topic: Some(topic), .. })
                if topic.as_str() == "Release readiness"
        ));
    }

    #[test]
    fn archive_and_complete_error_matrix_have_stable_exit_contracts() {
        let receipt = ChannelWriteReceipt {
            operation_id: operation_id(),
            resource_id: ChannelResourceId::Channel(aggregate_id(3)),
            version: AggregateVersion::FIRST,
        };
        let archived = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::Applied(receipt))),
            command_for_error(),
        );
        assert!(archived.stdout.contains("channel.archive"));
        assert_eq!(archived.exit_code, 0);

        let cases = [
            (ChannelCliError::InvalidRequest, "user_error", 1, false),
            (ChannelCliError::NotFound, "not_found", 1, false),
            (ChannelCliError::Unavailable, "network_error", 2, true),
            (
                ChannelCliError::PartialFailure,
                "delivery_unknown",
                2,
                false,
            ),
            (ChannelCliError::AuthorizationDenied, "auth_error", 3, false),
            (ChannelCliError::Unexpected, "error", 4, false),
            (ChannelCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output =
                execute_channel_command(&TestExecutor::returning(Err(error)), command_for_error());
            let value: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(value["error"], category);
            assert_eq!(value["retryable"], retryable);
            assert_eq!(output.exit_code, exit_code);
        }
    }

    #[test]
    fn empty_updates_and_mismatched_outcomes_fail_closed() {
        assert_eq!(
            ChannelUpdateRequest::new(None, None, None, None),
            Err(ChannelCliError::InvalidRequest)
        );
        let output = execute_channel_command(
            &TestExecutor::returning(Ok(ChannelCliOutcome::Canvas(metadata(None)))),
            command_for_error(),
        );
        assert!(output.stdout.is_empty());
        assert_eq!(output.exit_code, 4);
        assert!(output.stderr.contains("channel_cli_unexpected_response"));
    }
}
