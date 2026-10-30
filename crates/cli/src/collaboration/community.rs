use std::{fmt, num::NonZeroU32};

use collaboration_domain::{
    AggregateVersion, AuthoredValue, ChannelInviteTarget, CommunityCreateFields, CommunityId,
    CommunityJoinPolicy, CommunityLifecycleState, CommunityRecordFields, CommunityUpdate,
    IdentityProfile, InviteId, MembershipCreateFields, MembershipRecordFields, MembershipRole,
    MembershipScope, MembershipStatus, OperationId, PrincipalId, ProfileId, ProfileKind,
    ProfileMetadata, SocialList, SocialListKind, SocialReference,
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityOutputFormat {
    Json,
    Compact,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityInviteCreateRequest {
    pub community_id: CommunityId,
    pub target: ChannelInviteTarget,
    pub role: MembershipRole,
    pub max_uses: Option<NonZeroU32>,
    pub expires_at_millis: u64,
    pub operation_id: OperationId,
}

#[derive(Clone, Eq, PartialEq)]
pub enum CommunityCliCommand {
    GetProfile {
        community_id: CommunityId,
        profile_id: ProfileId,
    },
    SetProfile {
        community_id: CommunityId,
        profile_id: ProfileId,
        expected_version: Option<AggregateVersion>,
        metadata: ProfileMetadata,
        operation_id: OperationId,
    },
    GetSocialList {
        community_id: CommunityId,
        profile_id: ProfileId,
        kind: SocialListKind,
    },
    SetSocialList {
        community_id: CommunityId,
        profile_id: ProfileId,
        expected_version: AggregateVersion,
        list: SocialList,
        operation_id: OperationId,
    },
    GetCommunity {
        community_id: CommunityId,
    },
    ListCommunities,
    CreateCommunity {
        fields: CommunityCreateFields,
        operation_id: OperationId,
    },
    UpdateCommunity {
        community_id: CommunityId,
        expected_version: AggregateVersion,
        update: CommunityUpdate,
        operation_id: OperationId,
    },
    ListMembers {
        community_id: CommunityId,
    },
    AddMember {
        fields: MembershipCreateFields,
        operation_id: OperationId,
    },
    SetMemberRole {
        community_id: CommunityId,
        principal_id: PrincipalId,
        expected_version: AggregateVersion,
        role: MembershipRole,
        operation_id: OperationId,
    },
    RemoveMember {
        community_id: CommunityId,
        principal_id: PrincipalId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    CreateInvite(CommunityInviteCreateRequest),
    RedeemInvite {
        community_id: CommunityId,
        bearer_token: String,
        operation_id: OperationId,
    },
    RevokeInvite {
        community_id: CommunityId,
        invite_id: InviteId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommunityCliVerb {
    GetProfile,
    SetProfile,
    GetSocialList,
    SetSocialList,
    GetCommunity,
    ListCommunities,
    CreateCommunity,
    UpdateCommunity,
    ListMembers,
    AddMember,
    SetMemberRole,
    RemoveMember,
    CreateInvite,
    RedeemInvite,
    RevokeInvite,
}

impl CommunityCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::GetProfile => "profile.get",
            Self::SetProfile => "profile.set",
            Self::GetSocialList => "social.get",
            Self::SetSocialList => "social.set",
            Self::GetCommunity => "community.get",
            Self::ListCommunities => "community.list",
            Self::CreateCommunity => "community.create",
            Self::UpdateCommunity => "community.update",
            Self::ListMembers => "member.list",
            Self::AddMember => "member.add",
            Self::SetMemberRole => "member.set_role",
            Self::RemoveMember => "member.remove",
            Self::CreateInvite => "invite.create",
            Self::RedeemInvite => "invite.redeem",
            Self::RevokeInvite => "invite.revoke",
        }
    }
}

impl CommunityCliCommand {
    const fn verb(&self) -> CommunityCliVerb {
        match self {
            Self::GetProfile { .. } => CommunityCliVerb::GetProfile,
            Self::SetProfile { .. } => CommunityCliVerb::SetProfile,
            Self::GetSocialList { .. } => CommunityCliVerb::GetSocialList,
            Self::SetSocialList { .. } => CommunityCliVerb::SetSocialList,
            Self::GetCommunity { .. } => CommunityCliVerb::GetCommunity,
            Self::ListCommunities => CommunityCliVerb::ListCommunities,
            Self::CreateCommunity { .. } => CommunityCliVerb::CreateCommunity,
            Self::UpdateCommunity { .. } => CommunityCliVerb::UpdateCommunity,
            Self::ListMembers { .. } => CommunityCliVerb::ListMembers,
            Self::AddMember { .. } => CommunityCliVerb::AddMember,
            Self::SetMemberRole { .. } => CommunityCliVerb::SetMemberRole,
            Self::RemoveMember { .. } => CommunityCliVerb::RemoveMember,
            Self::CreateInvite(_) => CommunityCliVerb::CreateInvite,
            Self::RedeemInvite { .. } => CommunityCliVerb::RedeemInvite,
            Self::RevokeInvite { .. } => CommunityCliVerb::RevokeInvite,
        }
    }
}

impl fmt::Debug for CommunityCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommunityCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityResourceId {
    Profile(ProfileId),
    Community(CommunityId),
    Membership(PrincipalId),
    Invite(InviteId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: CommunityResourceId,
    pub version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct CommunityInviteReceipt {
    pub write: CommunityWriteReceipt,
    pub bearer_token: String,
}

impl fmt::Debug for CommunityInviteReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommunityInviteReceipt")
            .field("write", &self.write)
            .field("bearer_token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CommunityCliOutcome {
    Profile(IdentityProfile),
    SocialList(AuthoredValue<SocialList>),
    Community(CommunityRecordFields),
    Communities(Vec<CommunityRecordFields>),
    Members(Vec<MembershipRecordFields>),
    InviteCreated(CommunityInviteReceipt),
    Applied(CommunityWriteReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    AuthorizationDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl CommunityCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "community_cli_invalid_request",
            Self::NotFound => "community_cli_not_found",
            Self::Unavailable => "community_cli_unavailable",
            Self::AuthorizationDenied => "community_cli_authorization_denied",
            Self::PartialFailure => "community_cli_completion_unknown",
            Self::Unexpected => "community_cli_unexpected_response",
            Self::Conflict => "community_cli_stale_version",
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

pub trait CommunityCliExecutor {
    fn execute(
        &self,
        command: CommunityCliCommand,
    ) -> Result<CommunityCliOutcome, CommunityCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl CommunityCliExecution {
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

pub fn execute_community_command(
    executor: &impl CommunityCliExecutor,
    command: CommunityCliCommand,
    format: CommunityOutputFormat,
) -> CommunityCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome, format) {
            Some(output) => CommunityCliExecution::success(output),
            None => error_output(verb, CommunityCliError::Unexpected),
        },
        Err(error) => error_output(verb, error),
    }
}

fn error_output(verb: CommunityCliVerb, error: CommunityCliError) -> CommunityCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    CommunityCliExecution::failure(
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

fn success_output(
    verb: CommunityCliVerb,
    outcome: CommunityCliOutcome,
    format: CommunityOutputFormat,
) -> Option<Value> {
    match (verb, outcome) {
        (CommunityCliVerb::GetProfile, CommunityCliOutcome::Profile(profile)) => {
            Some(profile_output(&profile, format))
        }
        (CommunityCliVerb::GetSocialList, CommunityCliOutcome::SocialList(list)) => {
            Some(social_list_output(&list))
        }
        (CommunityCliVerb::GetCommunity, CommunityCliOutcome::Community(community)) => {
            Some(community_output(&community))
        }
        (CommunityCliVerb::ListCommunities, CommunityCliOutcome::Communities(communities)) => {
            Some(json!({
                "command": verb.as_str(),
                "communities": communities.iter().map(community_output).collect::<Vec<_>>(),
                "ok": true,
            }))
        }
        (CommunityCliVerb::ListMembers, CommunityCliOutcome::Members(members)) => Some(json!({
            "command": verb.as_str(),
            "members": members.iter().map(member_output).collect::<Vec<_>>(),
            "ok": true,
        })),
        (CommunityCliVerb::CreateInvite, CommunityCliOutcome::InviteCreated(receipt)) => {
            Some(invite_output(verb, receipt))
        }
        (
            CommunityCliVerb::SetProfile
            | CommunityCliVerb::SetSocialList
            | CommunityCliVerb::CreateCommunity
            | CommunityCliVerb::UpdateCommunity
            | CommunityCliVerb::AddMember
            | CommunityCliVerb::SetMemberRole
            | CommunityCliVerb::RemoveMember
            | CommunityCliVerb::RedeemInvite
            | CommunityCliVerb::RevokeInvite,
            CommunityCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn profile_output(profile: &IdentityProfile, format: CommunityOutputFormat) -> Value {
    let fields = profile.fields();
    let display_name = fields
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.value.display_name.as_deref());
    let base = json!({
        "author_public_key": hex_bytes(fields.author_public_key.as_bytes()),
        "command": CommunityCliVerb::GetProfile.as_str(),
        "community_id": fields.community_id,
        "display_name": display_name,
        "ok": true,
        "profile_id": fields.profile_id,
        "version": fields.version,
    });
    if format == CommunityOutputFormat::Compact {
        return base;
    }

    let metadata = fields.metadata.as_ref().map(|metadata| &metadata.value);
    let (profile_kind, claimed_owner) = match &fields.kind {
        ProfileKind::Human => ("human", None),
        ProfileKind::Agent(agent) => (
            "agent",
            agent
                .claimed_owner
                .as_ref()
                .map(|owner| hex_bytes(owner.as_bytes())),
        ),
    };
    json!({
        "about": metadata.and_then(|metadata| metadata.about.as_deref()),
        "author_public_key": hex_bytes(fields.author_public_key.as_bytes()),
        "avatar_url": metadata.and_then(|metadata| metadata.avatar_url.as_deref()),
        "claimed_owner_public_key": claimed_owner,
        "command": CommunityCliVerb::GetProfile.as_str(),
        "community_id": fields.community_id,
        "display_name": display_name,
        "name": metadata.and_then(|metadata| metadata.name.as_deref()),
        "nip05_handle": metadata.and_then(|metadata| metadata.nip05_handle.as_deref()),
        "ok": true,
        "profile_id": fields.profile_id,
        "profile_kind": profile_kind,
        "version": fields.version,
    })
}

fn social_list_output(list: &AuthoredValue<SocialList>) -> Value {
    json!({
        "command": CommunityCliVerb::GetSocialList.as_str(),
        "entries": list.value.entries.iter().map(social_reference_output).collect::<Vec<_>>(),
        "kind": social_list_kind(&list.value.kind),
        "ok": true,
        "source_created_at": list.source_created_at,
        "source_event_id": hex_bytes(list.source_event_id.as_bytes()),
    })
}

fn social_list_kind(kind: &SocialListKind) -> Value {
    match kind {
        SocialListKind::Contacts => json!({ "kind": "contacts" }),
        SocialListKind::Mutes => json!({ "kind": "mutes" }),
        SocialListKind::Pins => json!({ "kind": "pins" }),
        SocialListKind::Bookmarks => json!({ "kind": "bookmarks" }),
        SocialListKind::Emojis => json!({ "kind": "emojis" }),
        SocialListKind::NamedFollow(name) => json!({ "kind": "named_follow", "name": name }),
    }
}

fn social_reference_output(reference: &SocialReference) -> Value {
    match reference {
        SocialReference::PublicKey(public_key) => {
            json!({ "kind": "public_key", "value": hex_bytes(public_key.as_bytes()) })
        }
        SocialReference::Event(event_id) => {
            json!({ "kind": "event", "value": hex_bytes(event_id.as_bytes()) })
        }
        SocialReference::Coordinate(value) => json!({ "kind": "coordinate", "value": value }),
        SocialReference::Hashtag(value) => json!({ "kind": "hashtag", "value": value }),
        SocialReference::Url(value) => json!({ "kind": "url", "value": value }),
        SocialReference::Word(value) => json!({ "kind": "word", "value": value }),
        SocialReference::Emoji(value) => json!({ "kind": "emoji", "value": value }),
    }
}

fn community_output(community: &CommunityRecordFields) -> Value {
    let join_policy = match &community.join_policy {
        CommunityJoinPolicy::Open => json!({ "kind": "open" }),
        CommunityJoinPolicy::AcceptanceRequired(version) => json!({
            "kind": "acceptance_required",
            "version": version.as_str(),
        }),
    };
    json!({
        "command": CommunityCliVerb::GetCommunity.as_str(),
        "community_id": community.community_id,
        "host": community.host.as_str(),
        "icon": community.icon.as_ref().map(|icon| icon.as_str()),
        "join_policy": join_policy,
        "lifecycle": lifecycle(community.lifecycle_state),
        "ok": true,
        "version": community.version,
    })
}

const fn lifecycle(state: CommunityLifecycleState) -> &'static str {
    match state {
        CommunityLifecycleState::Active => "active",
        CommunityLifecycleState::Archived => "archived",
        CommunityLifecycleState::Quiescing => "quiescing",
        CommunityLifecycleState::Fenced => "fenced",
        CommunityLifecycleState::Tombstone => "tombstone",
    }
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

const fn role(role: MembershipRole) -> &'static str {
    match role {
        MembershipRole::Owner => "owner",
        MembershipRole::Admin => "admin",
        MembershipRole::Member => "member",
        MembershipRole::Guest => "guest",
        MembershipRole::Bot => "bot",
    }
}

const fn membership_status(status: MembershipStatus) -> &'static str {
    match status {
        MembershipStatus::Active => "active",
        MembershipStatus::Revoked => "revoked",
        MembershipStatus::Archived => "archived",
    }
}

fn invite_output(verb: CommunityCliVerb, receipt: CommunityInviteReceipt) -> Value {
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

fn write_output(verb: CommunityCliVerb, receipt: CommunityWriteReceipt) -> Value {
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

fn resource_output(resource: CommunityResourceId) -> (&'static str, String) {
    match resource {
        CommunityResourceId::Profile(profile_id) => ("profile", profile_id.to_string()),
        CommunityResourceId::Community(community_id) => ("community", community_id.to_string()),
        CommunityResourceId::Membership(principal_id) => ("membership", principal_id.to_string()),
        CommunityResourceId::Invite(invite_id) => ("invite", invite_id.as_uuid().to_string()),
    }
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
    use std::{cell::RefCell, collections::BTreeSet};

    use collaboration_domain::{
        AggregateId, AuthoredValue, CommunityHost, CommunityIconUpdate, NostrEventId,
        NostrPublicKey, ProfileRecordFields,
    };
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<CommunityCliCommand>>,
        result: RefCell<Option<Result<CommunityCliOutcome, CommunityCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<CommunityCliOutcome, CommunityCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl CommunityCliExecutor for TestExecutor {
        fn execute(
            &self,
            command: CommunityCliCommand,
        ) -> Result<CommunityCliOutcome, CommunityCliError> {
            self.command.replace(Some(command));
            self.result
                .borrow_mut()
                .take()
                .expect("the test executor is called once")
        }
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn profile_id() -> ProfileId {
        ProfileId::from_uuid(Uuid::from_u128(2))
    }

    fn principal_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(3))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(4))
    }

    fn profile() -> IdentityProfile {
        let author = NostrPublicKey::from_bytes([7; 32]);
        IdentityProfile::new(ProfileRecordFields {
            profile_id: profile_id(),
            community_id: community_id(),
            author_public_key: author,
            kind: ProfileKind::Human,
            metadata: Some(AuthoredValue {
                source_event_id: NostrEventId::from_bytes([8; 32]),
                source_author: author,
                source_created_at: 10,
                value: ProfileMetadata {
                    display_name: Some("Ada".into()),
                    name: Some("ada".into()),
                    avatar_url: Some("https://example.com/ada.png".into()),
                    about: Some("Builds reliable systems".into()),
                    nip05_handle: Some("ada@example.com".into()),
                },
            }),
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("valid profile fixture")
    }

    #[test]
    fn profile_json_and_compact_output_are_golden() {
        let command = CommunityCliCommand::GetProfile {
            community_id: community_id(),
            profile_id: profile_id(),
        };
        let full = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::Profile(profile()))),
            command.clone(),
            CommunityOutputFormat::Json,
        );
        assert_eq!(
            full.stdout,
            format!(
                "{}\n",
                json!({
                    "about": "Builds reliable systems",
                    "author_public_key": "07".repeat(32),
                    "avatar_url": "https://example.com/ada.png",
                    "claimed_owner_public_key": null,
                    "command": "profile.get",
                    "community_id": community_id(),
                    "display_name": "Ada",
                    "name": "ada",
                    "nip05_handle": "ada@example.com",
                    "ok": true,
                    "profile_id": profile_id(),
                    "profile_kind": "human",
                    "version": 1,
                })
            )
        );
        assert!(full.stderr.is_empty());
        assert_eq!(full.exit_code, 0);

        let compact = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::Profile(profile()))),
            command,
            CommunityOutputFormat::Compact,
        );
        assert_eq!(
            compact.stdout,
            format!(
                "{}\n",
                json!({
                    "author_public_key": "07".repeat(32),
                    "command": "profile.get",
                    "community_id": community_id(),
                    "display_name": "Ada",
                    "ok": true,
                    "profile_id": profile_id(),
                    "version": 1,
                })
            )
        );
    }

    #[test]
    fn social_and_membership_reads_use_canonical_records() {
        let social = AuthoredValue {
            source_event_id: NostrEventId::from_bytes([9; 32]),
            source_author: NostrPublicKey::from_bytes([7; 32]),
            source_created_at: 11,
            value: SocialList {
                kind: SocialListKind::Contacts,
                entries: BTreeSet::from([
                    SocialReference::PublicKey(NostrPublicKey::from_bytes([6; 32])),
                    SocialReference::Hashtag("rust".into()),
                ]),
            },
        };
        let social_output = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::SocialList(social))),
            CommunityCliCommand::GetSocialList {
                community_id: community_id(),
                profile_id: profile_id(),
                kind: SocialListKind::Contacts,
            },
            CommunityOutputFormat::Json,
        );
        assert!(social_output.stdout.contains("\"kind\":\"contacts\""));
        assert!(social_output.stdout.contains(&"06".repeat(32)));

        let member = MembershipRecordFields {
            community_id: community_id(),
            scope: MembershipScope::Channel(AggregateId::from_uuid(Uuid::from_u128(5))),
            principal_id: principal_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
            added_by_principal_id: Some(PrincipalId::from_uuid(Uuid::from_u128(6))),
        };
        let membership_output = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::Members(vec![member]))),
            CommunityCliCommand::ListMembers {
                community_id: community_id(),
            },
            CommunityOutputFormat::Compact,
        );
        assert_eq!(membership_output.exit_code, 0);
        assert!(
            membership_output
                .stdout
                .contains("\"command\":\"member.list\"")
        );
        assert!(membership_output.stdout.contains("\"role\":\"member\""));
        assert!(membership_output.stdout.contains("\"kind\":\"channel\""));
    }

    #[test]
    fn invite_token_is_explicit_once_and_redacted_elsewhere() {
        let invite_id = InviteId::from_uuid(Uuid::from_u128(7));
        let request = CommunityInviteCreateRequest {
            community_id: community_id(),
            target: ChannelInviteTarget::Community,
            role: MembershipRole::Guest,
            max_uses: NonZeroU32::new(2),
            expires_at_millis: 60_000,
            operation_id: operation_id(),
        };
        let command = CommunityCliCommand::CreateInvite(request);
        assert!(!format!("{command:?}").contains("bearer"));

        let receipt = CommunityInviteReceipt {
            write: CommunityWriteReceipt {
                operation_id: operation_id(),
                resource_id: CommunityResourceId::Invite(invite_id),
                version: AggregateVersion::FIRST,
            },
            bearer_token: "invite-secret".into(),
        };
        assert!(!format!("{receipt:?}").contains("invite-secret"));
        let output = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::InviteCreated(receipt))),
            command,
            CommunityOutputFormat::Json,
        );
        assert_eq!(output.exit_code, 0);
        assert!(output.stderr.is_empty());
        assert_eq!(
            output.stdout,
            format!(
                "{}\n",
                json!({
                    "bearer_token": "invite-secret",
                    "command": "invite.create",
                    "ok": true,
                    "operation_id": operation_id(),
                    "resource_id": invite_id.as_uuid().to_string(),
                    "resource_kind": "invite",
                    "version": 1,
                })
            )
        );

        let redeem = CommunityCliCommand::RedeemInvite {
            community_id: community_id(),
            bearer_token: "invite-secret".into(),
            operation_id: operation_id(),
        };
        assert!(!format!("{redeem:?}").contains("invite-secret"));
    }

    #[test]
    fn permission_and_exit_classes_use_the_common_contract() {
        let cases = [
            (CommunityCliError::InvalidRequest, "user_error", 1, false),
            (CommunityCliError::NotFound, "not_found", 1, false),
            (CommunityCliError::Unavailable, "network_error", 2, true),
            (
                CommunityCliError::PartialFailure,
                "delivery_unknown",
                2,
                false,
            ),
            (
                CommunityCliError::AuthorizationDenied,
                "auth_error",
                3,
                false,
            ),
            (CommunityCliError::Unexpected, "error", 4, false),
            (CommunityCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_community_command(
                &TestExecutor::returning(Err(error)),
                CommunityCliCommand::GetCommunity {
                    community_id: community_id(),
                },
                CommunityOutputFormat::Json,
            );
            assert!(output.stdout.is_empty());
            assert_eq!(output.exit_code, exit_code);
            let envelope: Value =
                serde_json::from_str(&output.stderr).expect("failure output is one JSON object");
            assert_eq!(envelope["error"], category);
            assert_eq!(envelope["retryable"], retryable);
            assert_eq!(envelope["command"], "community.get");
            assert!(!output.stderr.contains("invite-secret"));
        }
    }

    #[test]
    fn mismatched_executor_outcome_fails_closed() {
        let output = execute_community_command(
            &TestExecutor::returning(Ok(CommunityCliOutcome::Members(Vec::new()))),
            CommunityCliCommand::GetProfile {
                community_id: community_id(),
                profile_id: profile_id(),
            },
            CommunityOutputFormat::Json,
        );
        assert!(output.stdout.is_empty());
        assert_eq!(output.exit_code, 4);
        assert!(output.stderr.contains("community_cli_unexpected_response"));
    }

    #[test]
    fn every_mutation_forwards_canonical_operation_and_version_fields() {
        let update = CommunityCliCommand::UpdateCommunity {
            community_id: community_id(),
            expected_version: AggregateVersion::FIRST,
            update: CommunityUpdate {
                host: None,
                icon: CommunityIconUpdate::Unchanged,
            },
            operation_id: operation_id(),
        };
        let executor =
            TestExecutor::returning(Ok(CommunityCliOutcome::Applied(CommunityWriteReceipt {
                operation_id: operation_id(),
                resource_id: CommunityResourceId::Community(community_id()),
                version: AggregateVersion::new(2).expect("version two"),
            })));
        let output =
            execute_community_command(&executor, update.clone(), CommunityOutputFormat::Json);
        assert_eq!(executor.command.take(), Some(update));
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("\"command\":\"community.update\""));
        assert!(output.stdout.contains("\"version\":2"));

        let fields = CommunityCreateFields {
            community_id: community_id(),
            host: CommunityHost::new("community.example.com").expect("valid host"),
            icon: None,
            join_policy: CommunityJoinPolicy::Open,
        };
        let command = CommunityCliCommand::CreateCommunity {
            fields,
            operation_id: operation_id(),
        };
        assert_eq!(command.verb().as_str(), "community.create");
    }
}
