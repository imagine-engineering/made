use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, PrincipalId, authorize,
};

pub const MAX_DM_PARTICIPANTS: usize = 9;
pub const MIN_DM_PARTICIPANTS: usize = 2;
const MAX_DM_MUTATIONS: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmLifecycleState {
    Open,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmParticipantState {
    Active,
    Left,
    Removed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmMutationKind {
    ParticipantAdded { participant_id: PrincipalId },
    ParticipantRemoved { participant_id: PrincipalId },
    Left,
    Reopened,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DmMutation {
    pub actor_principal_id: PrincipalId,
    pub kind: DmMutationKind,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmRecordFields {
    pub community_id: CommunityId,
    pub dm_id: AggregateId,
    pub initial_participants: BTreeSet<PrincipalId>,
    pub participant_states: BTreeMap<PrincipalId, DmParticipantState>,
    pub lifecycle_state: DmLifecycleState,
    pub mutations: Vec<DmMutation>,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmOpenFields {
    pub community_id: CommunityId,
    pub dm_id: AggregateId,
    pub participants: Vec<PrincipalId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMessage {
    fields: DmRecordFields,
}

impl DirectMessage {
    pub fn open(
        fields: DmOpenFields,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, DmError> {
        validate_dm_id(fields.dm_id)?;
        let participants = validate_initial_participants(fields.participants)?;
        let actor_principal_id = authorization_subject(authorization);
        authorize_open(authorization, fields.community_id, actor_principal_id)?;
        if !participants.contains(&actor_principal_id) {
            return Err(DmError::ActorNotParticipant);
        }
        let participant_states = participants
            .iter()
            .copied()
            .map(|participant_id| (participant_id, DmParticipantState::Active))
            .collect();
        Ok(Self {
            fields: DmRecordFields {
                community_id: fields.community_id,
                dm_id: fields.dm_id,
                initial_participants: participants,
                participant_states,
                lifecycle_state: DmLifecycleState::Open,
                mutations: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn from_record(fields: DmRecordFields) -> Result<Self, DmError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &DmRecordFields {
        &self.fields
    }

    pub fn is_active_participant(&self, principal_id: PrincipalId) -> bool {
        self.fields.participant_states.get(&principal_id) == Some(&DmParticipantState::Active)
    }

    pub fn add_participant(
        &mut self,
        expected_version: AggregateVersion,
        participant_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<DmCommandOutcome, DmError> {
        let actor_principal_id = self.authorize_active_participant(authorization)?;
        self.apply_command(
            expected_version,
            actor_principal_id,
            DmMutationKind::ParticipantAdded { participant_id },
        )
    }

    pub fn remove_participant(
        &mut self,
        expected_version: AggregateVersion,
        participant_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<DmCommandOutcome, DmError> {
        let actor_principal_id = self.authorize_active_participant(authorization)?;
        self.apply_command(
            expected_version,
            actor_principal_id,
            DmMutationKind::ParticipantRemoved { participant_id },
        )
    }

    pub fn leave(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<DmCommandOutcome, DmError> {
        let actor_principal_id = self.authorize_active_participant(authorization)?;
        self.apply_command(expected_version, actor_principal_id, DmMutationKind::Left)
    }

    pub fn reopen(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<DmCommandOutcome, DmError> {
        let actor_principal_id = authorization_subject(authorization);
        authorize_reopen(
            authorization,
            self.fields.community_id,
            self.fields.dm_id,
            actor_principal_id,
        )?;
        if self.fields.participant_states.get(&actor_principal_id)
            != Some(&DmParticipantState::Left)
        {
            return Err(DmError::CannotReopen);
        }
        self.apply_command(
            expected_version,
            actor_principal_id,
            DmMutationKind::Reopened,
        )
    }

    fn authorize_active_participant(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<PrincipalId, DmError> {
        authorize_active_dm(authorization, self.fields.community_id, self.fields.dm_id)?;
        let actor_principal_id = authorization_subject(authorization);
        if !self.is_active_participant(actor_principal_id) {
            return Err(DmError::ActorNotParticipant);
        }
        Ok(actor_principal_id)
    }

    fn apply_command(
        &mut self,
        expected_version: AggregateVersion,
        actor_principal_id: PrincipalId,
        kind: DmMutationKind,
    ) -> Result<DmCommandOutcome, DmError> {
        self.require_version(expected_version)?;
        let mut participant_states = self.fields.participant_states.clone();
        if !apply_mutation(&mut participant_states, actor_principal_id, kind)? {
            return Ok(DmCommandOutcome::Unchanged);
        }
        if self.fields.mutations.len() >= MAX_DM_MUTATIONS {
            return Err(DmError::TooManyMutations);
        }
        let resulting_version = self
            .fields
            .version
            .next()
            .ok_or(DmError::VersionExhausted)?;
        self.fields.participant_states = participant_states;
        self.fields.lifecycle_state = lifecycle_state(&self.fields.participant_states);
        self.fields.mutations.push(DmMutation {
            actor_principal_id,
            kind,
            resulting_version,
        });
        self.fields.version = resulting_version;
        Ok(DmCommandOutcome::Applied)
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), DmError> {
        if self.fields.version != expected {
            return Err(DmError::StaleVersion {
                expected,
                actual: self.fields.version,
            });
        }
        Ok(())
    }
}

fn validate_dm_id(dm_id: AggregateId) -> Result<(), DmError> {
    if dm_id.as_uuid().is_nil() {
        return Err(DmError::InvalidDmId);
    }
    Ok(())
}

fn validate_initial_participants(
    participants: Vec<PrincipalId>,
) -> Result<BTreeSet<PrincipalId>, DmError> {
    if participants
        .iter()
        .any(|participant_id| participant_id.as_uuid().is_nil())
    {
        return Err(DmError::InvalidParticipant);
    }
    let participant_count = participants.len();
    let participants = participants.into_iter().collect::<BTreeSet<_>>();
    if participant_count != participants.len() {
        return Err(DmError::DuplicateParticipant);
    }
    if participants.len() < MIN_DM_PARTICIPANTS {
        return Err(DmError::TooFewParticipants);
    }
    if participants.len() > MAX_DM_PARTICIPANTS {
        return Err(DmError::TooManyParticipants);
    }
    Ok(participants)
}

fn apply_mutation(
    participant_states: &mut BTreeMap<PrincipalId, DmParticipantState>,
    actor_principal_id: PrincipalId,
    kind: DmMutationKind,
) -> Result<bool, DmError> {
    if actor_principal_id.as_uuid().is_nil() {
        return Err(DmError::InvalidParticipant);
    }
    match kind {
        DmMutationKind::ParticipantAdded { participant_id } => {
            require_active_actor(participant_states, actor_principal_id)?;
            require_other_participant(actor_principal_id, participant_id)?;
            match participant_states.get(&participant_id) {
                Some(DmParticipantState::Active) => return Ok(false),
                Some(DmParticipantState::Left) => return Err(DmError::ParticipantMustReopen),
                Some(DmParticipantState::Removed) | None => {}
            }
            require_capacity(participant_states)?;
            participant_states.insert(participant_id, DmParticipantState::Active);
        }
        DmMutationKind::ParticipantRemoved { participant_id } => {
            require_active_actor(participant_states, actor_principal_id)?;
            require_other_participant(actor_principal_id, participant_id)?;
            if participant_states.get(&participant_id) != Some(&DmParticipantState::Active) {
                return Err(DmError::TargetNotActive);
            }
            participant_states.insert(participant_id, DmParticipantState::Removed);
        }
        DmMutationKind::Left => {
            require_active_actor(participant_states, actor_principal_id)?;
            participant_states.insert(actor_principal_id, DmParticipantState::Left);
        }
        DmMutationKind::Reopened => {
            if participant_states.get(&actor_principal_id) != Some(&DmParticipantState::Left) {
                return Err(DmError::CannotReopen);
            }
            require_capacity(participant_states)?;
            participant_states.insert(actor_principal_id, DmParticipantState::Active);
        }
    }
    Ok(true)
}

fn require_active_actor(
    participant_states: &BTreeMap<PrincipalId, DmParticipantState>,
    actor_principal_id: PrincipalId,
) -> Result<(), DmError> {
    if participant_states.get(&actor_principal_id) != Some(&DmParticipantState::Active) {
        return Err(DmError::ActorNotParticipant);
    }
    Ok(())
}

fn require_other_participant(
    actor_principal_id: PrincipalId,
    participant_id: PrincipalId,
) -> Result<(), DmError> {
    if participant_id.as_uuid().is_nil() {
        return Err(DmError::InvalidParticipant);
    }
    if actor_principal_id == participant_id {
        return Err(DmError::SelfMembershipMutation);
    }
    Ok(())
}

fn require_capacity(
    participant_states: &BTreeMap<PrincipalId, DmParticipantState>,
) -> Result<(), DmError> {
    let active_count = participant_states
        .values()
        .filter(|state| **state == DmParticipantState::Active)
        .count();
    if active_count >= MAX_DM_PARTICIPANTS {
        return Err(DmError::TooManyParticipants);
    }
    Ok(())
}

fn lifecycle_state(
    participant_states: &BTreeMap<PrincipalId, DmParticipantState>,
) -> DmLifecycleState {
    if participant_states
        .values()
        .filter(|state| **state == DmParticipantState::Active)
        .take(MIN_DM_PARTICIPANTS)
        .count()
        == MIN_DM_PARTICIPANTS
    {
        DmLifecycleState::Open
    } else {
        DmLifecycleState::Closed
    }
}

fn authorize_open(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    actor_principal_id: PrincipalId,
) -> Result<(), DmError> {
    if request.action != AuthorizationAction::Write
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Community
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.owner_principal_id.is_some()
        || request.resource.channel_id.is_some()
        || authorization_subject(request) != actor_principal_id
    {
        return Err(DmError::AuthorizationShape);
    }
    authorize_dm(request)
}

fn authorize_active_dm(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    dm_id: AggregateId,
) -> Result<(), DmError> {
    if request.action != AuthorizationAction::Write
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Channel
        || request.resource.resource_id != dm_id
        || request.resource.owner_principal_id.is_some()
        || request.resource.channel_id != Some(dm_id)
    {
        return Err(DmError::AuthorizationShape);
    }
    authorize_dm(request)
}

fn authorize_reopen(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    dm_id: AggregateId,
    actor_principal_id: PrincipalId,
) -> Result<(), DmError> {
    if request.action != AuthorizationAction::Write
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Conversation
        || request.resource.resource_id != dm_id
        || request.resource.owner_principal_id.is_some()
        || request.resource.channel_id.is_some()
        || authorization_subject(request) != actor_principal_id
    {
        return Err(DmError::AuthorizationShape);
    }
    authorize_dm(request)
}

fn authorize_dm(request: &AuthorizationRequest<'_>) -> Result<(), DmError> {
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(DmError::Unauthorized(denial)),
    }
}

fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

fn validate_record(fields: &DmRecordFields) -> Result<(), DmError> {
    validate_dm_id(fields.dm_id)?;
    validate_initial_participants(fields.initial_participants.iter().copied().collect())?;
    if fields.mutations.len() > MAX_DM_MUTATIONS {
        return Err(DmError::TooManyMutations);
    }
    let mut participant_states = fields
        .initial_participants
        .iter()
        .copied()
        .map(|participant_id| (participant_id, DmParticipantState::Active))
        .collect::<BTreeMap<_, _>>();
    let mut version = AggregateVersion::FIRST;
    for mutation in &fields.mutations {
        let expected_version = version.next().ok_or(DmError::InvalidHistory)?;
        if mutation.resulting_version != expected_version
            || !apply_mutation(
                &mut participant_states,
                mutation.actor_principal_id,
                mutation.kind,
            )
            .map_err(|_| DmError::InvalidHistory)?
        {
            return Err(DmError::InvalidHistory);
        }
        version = expected_version;
    }
    if fields.participant_states != participant_states
        || fields.lifecycle_state != lifecycle_state(&participant_states)
        || fields.version != version
    {
        return Err(DmError::InvalidHistory);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmError {
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    InvalidDmId,
    InvalidParticipant,
    DuplicateParticipant,
    TooFewParticipants,
    TooManyParticipants,
    ActorNotParticipant,
    SelfMembershipMutation,
    ParticipantMustReopen,
    TargetNotActive,
    CannotReopen,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    TooManyMutations,
    VersionExhausted,
    InvalidHistory,
}

impl fmt::Display for DmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorizationShape | Self::Unauthorized(_) => {
                formatter.write_str("direct-message command is not authorized")
            }
            Self::InvalidDmId => formatter.write_str("direct-message id is invalid"),
            Self::InvalidParticipant | Self::DuplicateParticipant => {
                formatter.write_str("direct-message participant set is invalid")
            }
            Self::TooFewParticipants => {
                formatter.write_str("direct message requires at least two participants")
            }
            Self::TooManyParticipants => {
                formatter.write_str("direct message supports at most nine active participants")
            }
            Self::ActorNotParticipant => {
                formatter.write_str("direct-message actor is not an active participant")
            }
            Self::SelfMembershipMutation => {
                formatter.write_str("direct-message participants must leave themselves")
            }
            Self::ParticipantMustReopen => {
                formatter.write_str("departed direct-message participant must reopen themselves")
            }
            Self::TargetNotActive => {
                formatter.write_str("direct-message target is not an active participant")
            }
            Self::CannotReopen => {
                formatter.write_str("direct-message participant cannot reopen this conversation")
            }
            Self::StaleVersion { .. } => formatter.write_str("direct-message version is stale"),
            Self::TooManyMutations => {
                formatter.write_str("direct-message mutation history is full")
            }
            Self::VersionExhausted => formatter.write_str("direct-message version is exhausted"),
            Self::InvalidHistory => formatter.write_str("direct-message history is invalid"),
        }
    }
}

impl Error for DmError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, ChannelMembership,
        CommunityMembership, MembershipRole, MembershipStatus, PrincipalScopes, ServiceAccountId,
        TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("messages:write").expect("scope")
    }

    fn scopes() -> PrincipalScopes {
        PrincipalScopes::new([scope()]).expect("scopes")
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community_id, "dm-test").expect("route")),
            &[],
        )
        .expect("tenant")
    }

    fn authenticated(
        community_id: CommunityId,
        actor_principal_id: PrincipalId,
    ) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            actor_principal_id,
            community_id,
            ServiceAccountId::new(1),
            scopes(),
        )
    }

    fn community_membership(
        community_id: CommunityId,
        actor_principal_id: PrincipalId,
    ) -> CommunityMembership {
        CommunityMembership {
            community_id,
            principal_id: actor_principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }
    }

    fn open_request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        let community_id = tenant.community_id();
        AuthorizationRequest {
            tenant,
            principal: authenticated,
            required_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id.as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(community_membership(
                community_id,
                authenticated.principal_id(),
            )),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn active_request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        dm_id: AggregateId,
    ) -> AuthorizationRequest<'a> {
        let community_id = tenant.community_id();
        AuthorizationRequest {
            tenant,
            principal: authenticated,
            required_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Channel,
                resource_id: dm_id,
                owner_principal_id: None,
                channel_id: Some(dm_id),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(community_membership(
                community_id,
                authenticated.principal_id(),
            )),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id,
                channel_id: dm_id,
                principal_id: authenticated.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }

    fn reopen_request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        dm_id: AggregateId,
    ) -> AuthorizationRequest<'a> {
        let community_id = tenant.community_id();
        AuthorizationRequest {
            tenant,
            principal: authenticated,
            required_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Conversation,
                resource_id: dm_id,
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(community_membership(
                community_id,
                authenticated.principal_id(),
            )),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn dm() -> (
        DirectMessage,
        TenantContext,
        AuthorizationScope,
        AggregateId,
    ) {
        let community_id = community(1);
        let dm_id = aggregate(10);
        let tenant = tenant(community_id);
        let required_scope = scope();
        let actor = authenticated(community_id, principal(1));
        let authorization = open_request(&tenant, &actor, &required_scope);
        let dm = DirectMessage::open(
            DmOpenFields {
                community_id,
                dm_id,
                participants: vec![principal(1), principal(2)],
            },
            &authorization,
        )
        .expect("open direct message");
        (dm, tenant, required_scope, dm_id)
    }

    #[test]
    fn participants_add_remove_leave_and_reopen_with_contiguous_versions() {
        let (mut dm, tenant, required_scope, dm_id) = dm();
        let actor_one = authenticated(tenant.community_id(), principal(1));
        let actor_three = authenticated(tenant.community_id(), principal(3));

        assert_eq!(
            dm.add_participant(
                AggregateVersion::FIRST,
                principal(3),
                &active_request(&tenant, &actor_one, &required_scope, dm_id),
            ),
            Ok(DmCommandOutcome::Applied)
        );
        let version_two = AggregateVersion::new(2).expect("version");
        assert_eq!(
            dm.remove_participant(
                version_two,
                principal(2),
                &active_request(&tenant, &actor_one, &required_scope, dm_id),
            ),
            Ok(DmCommandOutcome::Applied)
        );
        let version_three = AggregateVersion::new(3).expect("version");
        assert_eq!(
            dm.leave(
                version_three,
                &active_request(&tenant, &actor_three, &required_scope, dm_id),
            ),
            Ok(DmCommandOutcome::Applied)
        );
        assert_eq!(dm.fields().lifecycle_state, DmLifecycleState::Closed);
        let version_four = AggregateVersion::new(4).expect("version");
        assert_eq!(
            dm.reopen(
                version_four,
                &reopen_request(&tenant, &actor_three, &required_scope, dm_id),
            ),
            Ok(DmCommandOutcome::Applied)
        );
        assert_eq!(dm.fields().lifecycle_state, DmLifecycleState::Open);
        assert_eq!(
            dm.fields().version,
            AggregateVersion::new(5).expect("version")
        );
        assert_eq!(dm.fields().mutations.len(), 4);
        assert_eq!(
            DirectMessage::from_record(dm.fields().clone()),
            Ok(dm.clone())
        );
    }

    #[test]
    fn stale_commands_are_atomic() {
        let (mut dm, tenant, required_scope, dm_id) = dm();
        let actor = authenticated(tenant.community_id(), principal(1));
        let authorization = active_request(&tenant, &actor, &required_scope, dm_id);
        dm.add_participant(AggregateVersion::FIRST, principal(3), &authorization)
            .expect("add participant");
        let before = dm.clone();

        assert_eq!(
            dm.remove_participant(AggregateVersion::FIRST, principal(2), &authorization),
            Err(DmError::StaleVersion {
                expected: AggregateVersion::FIRST,
                actual: AggregateVersion::new(2).expect("version"),
            })
        );
        assert_eq!(dm, before);
    }

    #[test]
    fn outsiders_and_removed_participants_cannot_mutate_or_self_reopen() {
        let (mut dm, tenant, required_scope, dm_id) = dm();
        let participant = authenticated(tenant.community_id(), principal(1));
        let outsider = authenticated(tenant.community_id(), principal(3));
        let before = dm.clone();

        assert_eq!(
            dm.add_participant(
                AggregateVersion::FIRST,
                principal(4),
                &active_request(&tenant, &outsider, &required_scope, dm_id),
            ),
            Err(DmError::ActorNotParticipant)
        );
        assert_eq!(dm, before);

        dm.remove_participant(
            AggregateVersion::FIRST,
            principal(2),
            &active_request(&tenant, &participant, &required_scope, dm_id),
        )
        .expect("remove participant");
        let removed = authenticated(tenant.community_id(), principal(2));
        assert_eq!(
            dm.reopen(
                AggregateVersion::new(2).expect("version"),
                &reopen_request(&tenant, &removed, &required_scope, dm_id),
            ),
            Err(DmError::CannotReopen)
        );
    }

    #[test]
    fn participant_bounds_and_transition_rules_fail_closed() {
        let community_id = community(1);
        let dm_id = aggregate(10);
        let tenant = tenant(community_id);
        let required_scope = scope();
        let actor = authenticated(community_id, principal(1));
        let authorization = open_request(&tenant, &actor, &required_scope);
        assert_eq!(
            DirectMessage::open(
                DmOpenFields {
                    community_id,
                    dm_id,
                    participants: vec![principal(1), principal(1)],
                },
                &authorization,
            ),
            Err(DmError::DuplicateParticipant)
        );
        assert_eq!(
            DirectMessage::open(
                DmOpenFields {
                    community_id,
                    dm_id,
                    participants: (1..=10).map(principal).collect(),
                },
                &authorization,
            ),
            Err(DmError::TooManyParticipants)
        );

        let (mut dm, tenant, required_scope, dm_id) = dm();
        let actor = authenticated(tenant.community_id(), principal(1));
        let authorization = active_request(&tenant, &actor, &required_scope, dm_id);
        assert_eq!(
            dm.remove_participant(AggregateVersion::FIRST, principal(1), &authorization),
            Err(DmError::SelfMembershipMutation)
        );
        assert_eq!(
            dm.add_participant(AggregateVersion::FIRST, principal(2), &authorization),
            Ok(DmCommandOutcome::Unchanged)
        );
        assert_eq!(dm.fields().version, AggregateVersion::FIRST);
    }

    #[test]
    fn hydration_rejects_state_or_version_that_history_does_not_produce() {
        let (dm, _, _, _) = dm();
        let mut invalid_state = dm.fields().clone();
        invalid_state
            .participant_states
            .insert(principal(2), DmParticipantState::Removed);
        assert_eq!(
            DirectMessage::from_record(invalid_state),
            Err(DmError::InvalidHistory)
        );

        let mut invalid_version = dm.fields().clone();
        invalid_version.version = AggregateVersion::new(2).expect("version");
        assert_eq!(
            DirectMessage::from_record(invalid_version),
            Err(DmError::InvalidHistory)
        );
    }
}
