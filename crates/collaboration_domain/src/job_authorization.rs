use std::collections::BTreeSet;

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthenticatedPrincipalKind, CommunityId,
    CommunityMembership, Job, JobCommand, JobCommandKind, JobCommandType, JobIdentity,
    MembershipRole, MembershipStatus, PrincipalId, TenantContext,
};

pub const MAX_JOB_DELEGATION_DEPTH: usize = 8;
pub const MAX_ACTIVE_CHILD_JOBS: usize = 16;
pub const MAX_ACTIVE_COMMUNITY_JOBS: usize = 256;
pub const MAX_JOB_RESOURCE_IDS: usize = 64;

const JOBS_WRITE_SCOPE: &str = "jobs:write";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobTransitionSet(u8);

impl JobTransitionSet {
    const REQUEST: u8 = 1 << 0;
    const ACCEPT: u8 = 1 << 1;
    const PROGRESS: u8 = 1 << 2;
    const RESULT: u8 = 1 << 3;
    const CANCEL: u8 = 1 << 4;
    const ERROR: u8 = 1 << 5;

    pub fn new(commands: impl IntoIterator<Item = JobCommandType>) -> Self {
        let mut bits = 0;
        for command in commands {
            bits |= match command {
                JobCommandType::Request => Self::REQUEST,
                JobCommandType::Accept => Self::ACCEPT,
                JobCommandType::Progress => Self::PROGRESS,
                JobCommandType::Result => Self::RESULT,
                JobCommandType::Cancel => Self::CANCEL,
                JobCommandType::Error => Self::ERROR,
            };
        }
        Self(bits)
    }

    pub const fn all() -> Self {
        Self(
            Self::REQUEST
                | Self::ACCEPT
                | Self::PROGRESS
                | Self::RESULT
                | Self::CANCEL
                | Self::ERROR,
        )
    }

    pub const fn contains(self, command: JobCommandType) -> bool {
        let bit = match command {
            JobCommandType::Request => Self::REQUEST,
            JobCommandType::Accept => Self::ACCEPT,
            JobCommandType::Progress => Self::PROGRESS,
            JobCommandType::Result => Self::RESULT,
            JobCommandType::Cancel => Self::CANCEL,
            JobCommandType::Error => Self::ERROR,
        };
        self.0 & bit != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobTeamRole {
    Owner,
    Coordinator,
    Executor,
    Observer,
}

impl JobTeamRole {
    const fn permits(self, command: JobCommandType) -> bool {
        match self {
            Self::Owner => true,
            Self::Coordinator => matches!(
                command,
                JobCommandType::Request | JobCommandType::Cancel | JobCommandType::Error
            ),
            Self::Executor => matches!(
                command,
                JobCommandType::Accept
                    | JobCommandType::Progress
                    | JobCommandType::Result
                    | JobCommandType::Error
            ),
            Self::Observer => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobTeamMembership {
    pub community_id: CommunityId,
    pub team_id: AggregateId,
    pub principal_id: PrincipalId,
    pub role: JobTeamRole,
    pub status: MembershipStatus,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobServiceGrant {
    community_id: CommunityId,
    job_id: AggregateId,
    job_version: AggregateVersion,
    service_principal_id: PrincipalId,
    permitted_transitions: JobTransitionSet,
    expires_at_millis: u64,
    revoked: bool,
}

impl JobServiceGrant {
    pub fn new(
        community_id: CommunityId,
        job_id: AggregateId,
        job_version: AggregateVersion,
        service_principal_id: PrincipalId,
        permitted_transitions: JobTransitionSet,
        expires_at_millis: u64,
    ) -> Result<Self, JobAuthorizationDenial> {
        if community_id.as_uuid().is_nil()
            || job_id.as_uuid().is_nil()
            || service_principal_id.as_uuid().is_nil()
            || permitted_transitions.is_empty()
        {
            return Err(JobAuthorizationDenial::InvalidServiceGrant);
        }
        Ok(Self {
            community_id,
            job_id,
            job_version,
            service_principal_id,
            permitted_transitions,
            expires_at_millis,
            revoked: false,
        })
    }

    pub fn revoke(&mut self) {
        self.revoked = true;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobDelegationGrant {
    community_id: CommunityId,
    parent_job_id: AggregateId,
    child_job_id: AggregateId,
    delegator_principal_id: PrincipalId,
    delegate_principal_id: PrincipalId,
    membership_version: AggregateVersion,
    permitted_transitions: JobTransitionSet,
    resource_ids: BTreeSet<AggregateId>,
    expires_at_millis: u64,
    revoked: bool,
}

impl JobDelegationGrant {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        community_id: CommunityId,
        parent_job_id: AggregateId,
        child_job_id: AggregateId,
        delegator_principal_id: PrincipalId,
        delegate_principal_id: PrincipalId,
        membership_version: AggregateVersion,
        permitted_transitions: JobTransitionSet,
        resource_ids: impl IntoIterator<Item = AggregateId>,
        expires_at_millis: u64,
    ) -> Result<Self, JobAuthorizationDenial> {
        if community_id.as_uuid().is_nil()
            || parent_job_id.as_uuid().is_nil()
            || child_job_id.as_uuid().is_nil()
            || parent_job_id == child_job_id
            || delegator_principal_id.as_uuid().is_nil()
            || delegate_principal_id.as_uuid().is_nil()
            || permitted_transitions.is_empty()
        {
            return Err(JobAuthorizationDenial::InvalidDelegation);
        }
        let resource_ids =
            collect_resource_ids(resource_ids).ok_or(JobAuthorizationDenial::InvalidDelegation)?;
        Ok(Self {
            community_id,
            parent_job_id,
            child_job_id,
            delegator_principal_id,
            delegate_principal_id,
            membership_version,
            permitted_transitions,
            resource_ids,
            expires_at_millis,
            revoked: false,
        })
    }

    pub fn revoke(&mut self) {
        self.revoked = true;
    }
}

pub struct JobAuthorizationRequest<'a> {
    pub tenant: &'a TenantContext,
    pub principal: &'a AuthenticatedPrincipal,
    pub job: &'a Job,
    pub command: &'a JobCommand,
    pub current_membership_version: AggregateVersion,
    pub community_membership: Option<CommunityMembership>,
    pub team_id: Option<AggregateId>,
    pub current_team_version: Option<AggregateVersion>,
    pub team_membership: Option<JobTeamMembership>,
    pub service_grant: Option<&'a JobServiceGrant>,
    pub delegation: Option<&'a JobDelegationGrant>,
    pub ancestry: &'a [JobIdentity],
    pub authorized_resource_ids: &'a [AggregateId],
    pub job_resource_ids: &'a [AggregateId],
    pub active_child_jobs: usize,
    pub active_community_jobs: usize,
    pub now_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobAuthorizationDecision {
    Allowed,
    Denied(JobAuthorizationDenial),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobAuthorizationDenial {
    TenantMismatch,
    InvalidTransition,
    MissingScope,
    ActorMismatch,
    InvalidResourceScope,
    ResourceScopeExceeded,
    MissingMembership,
    InactiveMembership,
    StaleMembership,
    InsufficientRole,
    InvalidTeamMembership,
    InactiveTeamMembership,
    StaleTeamMembership,
    InvalidServiceGrant,
    RevokedServiceGrant,
    ExpiredServiceGrant,
    ServiceCommandNotGranted,
    InvalidDelegation,
    RevokedDelegation,
    ExpiredDelegation,
    DelegationCycle,
    DelegationDepthExceeded,
    DelegationCommandNotGranted,
    DelegationResourceScopeExceeded,
    ActiveChildLimitExceeded,
    CommunityJobLimitExceeded,
}

pub fn authorize_job_transition(request: &JobAuthorizationRequest<'_>) -> JobAuthorizationDecision {
    let identity = request.job.identity();
    if request.tenant.community_id() != request.principal.community_id()
        || identity.community_id() != request.tenant.community_id()
        || request.command.identity() != identity
    {
        return denied(JobAuthorizationDenial::TenantMismatch);
    }
    let mut candidate = request.job.clone();
    if candidate.apply(request.command.clone()).is_err() {
        return denied(JobAuthorizationDenial::InvalidTransition);
    }
    if !request
        .principal
        .scopes()
        .iter()
        .any(|scope| scope.as_str() == JOBS_WRITE_SCOPE)
    {
        return denied(JobAuthorizationDenial::MissingScope);
    }

    let command_type = request.command.kind().command_type();
    let subject_principal_id = authorization_subject(request.principal);
    if command_actor(request.command.kind()) != subject_principal_id {
        return denied(JobAuthorizationDenial::ActorMismatch);
    }
    let authorized_resources =
        match collect_resource_ids(request.authorized_resource_ids.iter().copied()) {
            Some(resources) => resources,
            None => return denied(JobAuthorizationDenial::InvalidResourceScope),
        };
    let job_resources = match collect_resource_ids(request.job_resource_ids.iter().copied()) {
        Some(resources) => resources,
        None => return denied(JobAuthorizationDenial::InvalidResourceScope),
    };
    if !job_resources.is_subset(&authorized_resources) {
        return denied(JobAuthorizationDenial::ResourceScopeExceeded);
    }
    if command_type == JobCommandType::Request
        && request.active_community_jobs >= MAX_ACTIVE_COMMUNITY_JOBS
    {
        return denied(JobAuthorizationDenial::CommunityJobLimitExceeded);
    }

    if let Some(delegation) = request.delegation {
        return authorize_delegation(
            request,
            delegation,
            subject_principal_id,
            command_type,
            &job_resources,
        );
    }
    if !request.ancestry.is_empty() {
        return denied(JobAuthorizationDenial::InvalidDelegation);
    }

    if matches!(
        request.principal.kind(),
        AuthenticatedPrincipalKind::Service { .. }
    ) {
        return authorize_service(request, subject_principal_id, command_type);
    }
    if request.service_grant.is_some() {
        return denied(JobAuthorizationDenial::InvalidServiceGrant);
    }

    let community_role = match request.community_membership {
        Some(membership) => {
            match validate_community_membership(request, membership, subject_principal_id) {
                Ok(role) => Some(role),
                Err(denial) => return denied(denial),
            }
        }
        None => None,
    };
    let team_role = match request.team_membership {
        Some(membership) => {
            match validate_team_membership(request, membership, subject_principal_id) {
                Ok(role) => Some(role),
                Err(denial) => return denied(denial),
            }
        }
        None => None,
    };

    if matches!(
        request.principal.kind(),
        AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
    ) && team_role.is_none()
    {
        return denied(JobAuthorizationDenial::MissingMembership);
    }
    if community_role.is_none() && team_role.is_none() {
        return denied(JobAuthorizationDenial::MissingMembership);
    }
    if community_role.is_some_and(|role| community_role_permits(role, request, command_type))
        || team_role.is_some_and(|role| role.permits(command_type))
    {
        JobAuthorizationDecision::Allowed
    } else {
        denied(JobAuthorizationDenial::InsufficientRole)
    }
}

fn authorize_service(
    request: &JobAuthorizationRequest<'_>,
    subject_principal_id: PrincipalId,
    command_type: JobCommandType,
) -> JobAuthorizationDecision {
    let Some(grant) = request.service_grant else {
        return denied(JobAuthorizationDenial::InvalidServiceGrant);
    };
    if grant.community_id != request.tenant.community_id()
        || grant.job_id != request.job.identity().job_id()
        || grant.job_version != request.job.version()
        || grant.service_principal_id != subject_principal_id
    {
        return denied(JobAuthorizationDenial::InvalidServiceGrant);
    }
    if grant.revoked {
        return denied(JobAuthorizationDenial::RevokedServiceGrant);
    }
    if grant.expires_at_millis < request.now_millis {
        return denied(JobAuthorizationDenial::ExpiredServiceGrant);
    }
    if !grant.permitted_transitions.contains(command_type) {
        return denied(JobAuthorizationDenial::ServiceCommandNotGranted);
    }
    JobAuthorizationDecision::Allowed
}

fn authorize_delegation(
    request: &JobAuthorizationRequest<'_>,
    delegation: &JobDelegationGrant,
    subject_principal_id: PrincipalId,
    command_type: JobCommandType,
    job_resources: &BTreeSet<AggregateId>,
) -> JobAuthorizationDecision {
    let identity = request.job.identity();
    if delegation.community_id != identity.community_id()
        || delegation.child_job_id != identity.job_id()
        || delegation.delegate_principal_id != subject_principal_id
        || delegation.delegator_principal_id.as_uuid().is_nil()
        || delegation.membership_version != request.current_membership_version
    {
        return denied(JobAuthorizationDenial::InvalidDelegation);
    }
    if delegation.revoked {
        return denied(JobAuthorizationDenial::RevokedDelegation);
    }
    if delegation.expires_at_millis < request.now_millis {
        return denied(JobAuthorizationDenial::ExpiredDelegation);
    }
    if !delegation.permitted_transitions.contains(command_type) {
        return denied(JobAuthorizationDenial::DelegationCommandNotGranted);
    }
    if !job_resources.is_subset(&delegation.resource_ids) {
        return denied(JobAuthorizationDenial::DelegationResourceScopeExceeded);
    }
    if request.ancestry.len() > MAX_JOB_DELEGATION_DEPTH {
        return denied(JobAuthorizationDenial::DelegationDepthExceeded);
    }
    let mut ancestor_ids = BTreeSet::new();
    for ancestor in request.ancestry {
        if ancestor.community_id() != identity.community_id() {
            return denied(JobAuthorizationDenial::TenantMismatch);
        }
        if ancestor.job_id() == identity.job_id() || !ancestor_ids.insert(ancestor.job_id()) {
            return denied(JobAuthorizationDenial::DelegationCycle);
        }
    }
    if request.ancestry.is_empty()
        || request
            .ancestry
            .last()
            .is_none_or(|parent| parent.job_id() != delegation.parent_job_id)
    {
        return denied(JobAuthorizationDenial::InvalidDelegation);
    }
    if command_type == JobCommandType::Request {
        if request.active_child_jobs >= MAX_ACTIVE_CHILD_JOBS {
            return denied(JobAuthorizationDenial::ActiveChildLimitExceeded);
        }
    }
    JobAuthorizationDecision::Allowed
}

fn validate_community_membership(
    request: &JobAuthorizationRequest<'_>,
    membership: CommunityMembership,
    subject_principal_id: PrincipalId,
) -> Result<MembershipRole, JobAuthorizationDenial> {
    if membership.community_id != request.tenant.community_id()
        || membership.principal_id != subject_principal_id
    {
        return Err(JobAuthorizationDenial::TenantMismatch);
    }
    if membership.version != request.current_membership_version {
        return Err(JobAuthorizationDenial::StaleMembership);
    }
    if membership.status != MembershipStatus::Active {
        return Err(JobAuthorizationDenial::InactiveMembership);
    }
    Ok(membership.role)
}

fn validate_team_membership(
    request: &JobAuthorizationRequest<'_>,
    membership: JobTeamMembership,
    subject_principal_id: PrincipalId,
) -> Result<JobTeamRole, JobAuthorizationDenial> {
    if membership.community_id != request.tenant.community_id()
        || membership.principal_id != subject_principal_id
        || Some(membership.team_id) != request.team_id
    {
        return Err(JobAuthorizationDenial::InvalidTeamMembership);
    }
    if Some(membership.version) != request.current_team_version {
        return Err(JobAuthorizationDenial::StaleTeamMembership);
    }
    if membership.status != MembershipStatus::Active {
        return Err(JobAuthorizationDenial::InactiveTeamMembership);
    }
    Ok(membership.role)
}

fn community_role_permits(
    role: MembershipRole,
    request: &JobAuthorizationRequest<'_>,
    command_type: JobCommandType,
) -> bool {
    match role {
        MembershipRole::Owner | MembershipRole::Admin => true,
        MembershipRole::Member => match command_type {
            JobCommandType::Request => {
                authorization_subject(request.principal) == request.job.requester_principal_id()
            }
            JobCommandType::Accept
            | JobCommandType::Progress
            | JobCommandType::Result
            | JobCommandType::Error => {
                authorization_subject(request.principal)
                    == request.job.target_executor_principal_id()
            }
            JobCommandType::Cancel => {
                authorization_subject(request.principal) == request.job.requester_principal_id()
            }
        },
        MembershipRole::Guest | MembershipRole::Bot => false,
    }
}

fn authorization_subject(principal: &AuthenticatedPrincipal) -> PrincipalId {
    match principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => principal.principal_id(),
    }
}

fn command_actor(kind: JobCommandKind) -> PrincipalId {
    match kind {
        JobCommandKind::Request {
            requester_principal_id,
            ..
        } => requester_principal_id,
        JobCommandKind::Accept {
            executor_principal_id,
        }
        | JobCommandKind::Progress {
            executor_principal_id,
        }
        | JobCommandKind::Result {
            executor_principal_id,
        } => executor_principal_id,
        JobCommandKind::Cancel { actor_principal_id }
        | JobCommandKind::Error { actor_principal_id } => actor_principal_id,
    }
}

fn collect_resource_ids(
    resource_ids: impl IntoIterator<Item = AggregateId>,
) -> Option<BTreeSet<AggregateId>> {
    let mut resources = BTreeSet::new();
    for (index, resource_id) in resource_ids.into_iter().enumerate() {
        if index >= MAX_JOB_RESOURCE_IDS
            || resource_id.as_uuid().is_nil()
            || !resources.insert(resource_id)
        {
            return None;
        }
    }
    Some(resources)
}

const fn denied(denial: JobAuthorizationDenial) -> JobAuthorizationDecision {
    JobAuthorizationDecision::Denied(denial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentProfile, AuthorizationScope, IdentityProfile, JobCommandOutcome,
        NostrAuthenticationMethod, NostrEventId, NostrPublicKey, OwnerAttestationEvidence,
        PrincipalScopes, ProfileId, ProfileKind, ProfileRecordFields, ServiceAccountId,
        TrustedTenantRoute,
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

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community_id, "job-policy").expect("route")),
            &[],
        )
        .expect("tenant")
    }

    fn scopes(values: &[&str]) -> PrincipalScopes {
        PrincipalScopes::new(
            values
                .iter()
                .map(|value| AuthorizationScope::new(*value).expect("scope")),
        )
        .expect("scope set")
    }

    fn identity() -> JobIdentity {
        JobIdentity::new(community(1), aggregate(10)).expect("job identity")
    }

    fn command(version: u64, kind: JobCommandKind) -> JobCommand {
        JobCommand::new(
            identity(),
            crate::OperationId::from_uuid(Uuid::from_u128(100 + u128::from(version))),
            AggregateVersion::new(version).expect("version"),
            1_000 + version,
            kind,
        )
        .expect("job command")
    }

    fn requested_job() -> Job {
        Job::request(command(
            1,
            JobCommandKind::Request {
                requester_principal_id: principal(2),
                target_executor_principal_id: principal(3),
            },
        ))
        .expect("requested job")
    }

    fn membership(
        principal_id: PrincipalId,
        role: MembershipRole,
        status: MembershipStatus,
    ) -> CommunityMembership {
        CommunityMembership {
            community_id: community(1),
            principal_id,
            role,
            status,
            version: AggregateVersion::FIRST,
        }
    }

    fn account(principal_id: PrincipalId, granted_scopes: &[&str]) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id,
            community(1),
            ServiceAccountId::new(1),
            scopes(granted_scopes),
        )
    }

    fn agent_profile() -> IdentityProfile {
        let agent_public_key = NostrPublicKey::from_bytes([3; 32]);
        let owner_public_key = NostrPublicKey::from_bytes([2; 32]);
        IdentityProfile::new(ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(30)),
            community_id: community(1),
            author_public_key: agent_public_key,
            kind: ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(owner_public_key),
                owner_attestation: Some(OwnerAttestationEvidence {
                    owner_public_key,
                    agent_public_key,
                    proof_event_id: NostrEventId::from_bytes([4; 32]),
                    exact_conditions: "kind=43001".to_owned(),
                    verified_at: 1,
                }),
            }),
            metadata: None,
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("agent profile")
    }

    fn agent() -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::owner_attested_agent(
            principal(3),
            community(1),
            &agent_profile(),
            NostrAuthenticationMethod::Nip42,
            scopes(&[JOBS_WRITE_SCOPE]),
        )
        .expect("agent")
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a AuthenticatedPrincipal,
        job: &'a Job,
        transition: &'a JobCommand,
    ) -> JobAuthorizationRequest<'a> {
        JobAuthorizationRequest {
            tenant,
            principal: authenticated,
            job,
            command: transition,
            current_membership_version: AggregateVersion::FIRST,
            community_membership: None,
            team_id: None,
            current_team_version: None,
            team_membership: None,
            service_grant: None,
            delegation: None,
            ancestry: &[],
            authorized_resource_ids: &[],
            job_resource_ids: &[],
            active_child_jobs: 0,
            active_community_jobs: 1,
            now_millis: 100,
        }
    }

    #[test]
    fn owner_team_and_service_each_receive_only_their_transition_authority() {
        let tenant = tenant(community(1));
        let job = requested_job();

        let owner = account(principal(2), &[JOBS_WRITE_SCOPE]);
        let cancel = command(
            2,
            JobCommandKind::Cancel {
                actor_principal_id: principal(2),
            },
        );
        let mut owner_request = request(&tenant, &owner, &job, &cancel);
        owner_request.community_membership = Some(membership(
            principal(2),
            MembershipRole::Owner,
            MembershipStatus::Active,
        ));
        assert_eq!(
            authorize_job_transition(&owner_request),
            JobAuthorizationDecision::Allowed
        );

        let agent = agent();
        let accept = command(
            2,
            JobCommandKind::Accept {
                executor_principal_id: principal(3),
            },
        );
        let mut team_request = request(&tenant, &agent, &job, &accept);
        team_request.team_id = Some(aggregate(20));
        team_request.current_team_version = Some(AggregateVersion::FIRST);
        team_request.team_membership = Some(JobTeamMembership {
            community_id: community(1),
            team_id: aggregate(20),
            principal_id: principal(3),
            role: JobTeamRole::Executor,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        });
        assert_eq!(
            authorize_job_transition(&team_request),
            JobAuthorizationDecision::Allowed
        );

        let service = AuthenticatedPrincipal::service(
            principal(4),
            community(1),
            "job-recovery",
            scopes(&[JOBS_WRITE_SCOPE]),
        )
        .expect("service");
        let failure = command(
            2,
            JobCommandKind::Error {
                actor_principal_id: principal(4),
            },
        );
        let service_grant = JobServiceGrant::new(
            community(1),
            aggregate(10),
            AggregateVersion::FIRST,
            principal(4),
            JobTransitionSet::new([JobCommandType::Error]),
            200,
        )
        .expect("service grant");
        let mut service_request = request(&tenant, &service, &job, &failure);
        service_request.service_grant = Some(&service_grant);
        assert_eq!(
            authorize_job_transition(&service_request),
            JobAuthorizationDecision::Allowed
        );
    }

    #[test]
    fn revoked_community_and_team_members_cannot_transition_jobs() {
        let tenant = tenant(community(1));
        let job = requested_job();
        let owner = account(principal(2), &[JOBS_WRITE_SCOPE]);
        let cancel = command(
            2,
            JobCommandKind::Cancel {
                actor_principal_id: principal(2),
            },
        );
        let mut community_request = request(&tenant, &owner, &job, &cancel);
        community_request.community_membership = Some(membership(
            principal(2),
            MembershipRole::Owner,
            MembershipStatus::Revoked,
        ));
        assert_eq!(
            authorize_job_transition(&community_request),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::InactiveMembership)
        );

        let agent = agent();
        let accept = command(
            2,
            JobCommandKind::Accept {
                executor_principal_id: principal(3),
            },
        );
        let mut team_request = request(&tenant, &agent, &job, &accept);
        team_request.team_id = Some(aggregate(20));
        team_request.current_team_version = Some(AggregateVersion::FIRST);
        team_request.team_membership = Some(JobTeamMembership {
            community_id: community(1),
            team_id: aggregate(20),
            principal_id: principal(3),
            role: JobTeamRole::Executor,
            status: MembershipStatus::Revoked,
            version: AggregateVersion::FIRST,
        });
        assert_eq!(
            authorize_job_transition(&team_request),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::InactiveTeamMembership)
        );
    }

    #[test]
    fn delegation_rejects_cycles_and_excessive_depth() {
        let tenant = tenant(community(1));
        let agent = agent();
        let job = requested_job();
        let accept = command(
            2,
            JobCommandKind::Accept {
                executor_principal_id: principal(3),
            },
        );
        let grant = JobDelegationGrant::new(
            community(1),
            aggregate(30),
            aggregate(10),
            principal(2),
            principal(3),
            AggregateVersion::FIRST,
            JobTransitionSet::all(),
            [],
            200,
        )
        .expect("delegation grant");
        let cycle = [identity()];
        let mut cycle_request = request(&tenant, &agent, &job, &accept);
        cycle_request.delegation = Some(&grant);
        cycle_request.ancestry = &cycle;
        assert_eq!(
            authorize_job_transition(&cycle_request),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::DelegationCycle)
        );

        let ancestry = (0..=MAX_JOB_DELEGATION_DEPTH)
            .map(|index| {
                JobIdentity::new(community(1), aggregate(30 + index as u128))
                    .expect("ancestor identity")
            })
            .collect::<Vec<_>>();
        let mut depth_request = request(&tenant, &agent, &job, &accept);
        depth_request.delegation = Some(&grant);
        depth_request.ancestry = &ancestry;
        assert_eq!(
            authorize_job_transition(&depth_request),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::DelegationDepthExceeded)
        );
    }

    #[test]
    fn job_and_delegation_resource_scopes_can_only_narrow() {
        let tenant = tenant(community(1));
        let agent = agent();
        let job = requested_job();
        let accept = command(
            2,
            JobCommandKind::Accept {
                executor_principal_id: principal(3),
            },
        );
        let authorized = [aggregate(40), aggregate(41)];
        let requested = [aggregate(40), aggregate(42)];
        let mut direct = request(&tenant, &agent, &job, &accept);
        direct.authorized_resource_ids = &authorized;
        direct.job_resource_ids = &requested;
        assert_eq!(
            authorize_job_transition(&direct),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::ResourceScopeExceeded)
        );

        let grant = JobDelegationGrant::new(
            community(1),
            aggregate(30),
            aggregate(10),
            principal(2),
            principal(3),
            AggregateVersion::FIRST,
            JobTransitionSet::all(),
            [aggregate(40)],
            200,
        )
        .expect("delegation grant");
        let ancestry = [JobIdentity::new(community(1), aggregate(30)).expect("parent identity")];
        let delegated_resources = [aggregate(40), aggregate(41)];
        let mut delegated = request(&tenant, &agent, &job, &accept);
        delegated.delegation = Some(&grant);
        delegated.ancestry = &ancestry;
        delegated.authorized_resource_ids = &authorized;
        delegated.job_resource_ids = &delegated_resources;
        assert_eq!(
            authorize_job_transition(&delegated),
            JobAuthorizationDecision::Denied(
                JobAuthorizationDenial::DelegationResourceScopeExceeded
            )
        );

        let oversized_resources = (0..=MAX_JOB_RESOURCE_IDS)
            .map(|index| aggregate(100 + index as u128))
            .collect::<Vec<_>>();
        let mut oversized = request(&tenant, &agent, &job, &accept);
        oversized.authorized_resource_ids = &oversized_resources;
        oversized.job_resource_ids = &oversized_resources;
        assert_eq!(
            authorize_job_transition(&oversized),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::InvalidResourceScope)
        );
    }

    #[test]
    fn missing_scope_and_wrong_actor_fail_before_role_authority() {
        let tenant = tenant(community(1));
        let job = requested_job();
        let no_scope = account(principal(2), &[]);
        let cancel = command(
            2,
            JobCommandKind::Cancel {
                actor_principal_id: principal(2),
            },
        );
        let mut missing_scope = request(&tenant, &no_scope, &job, &cancel);
        missing_scope.community_membership = Some(membership(
            principal(2),
            MembershipRole::Owner,
            MembershipStatus::Active,
        ));
        assert_eq!(
            authorize_job_transition(&missing_scope),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::MissingScope)
        );

        let owner = account(principal(2), &[JOBS_WRITE_SCOPE]);
        let forged = command(
            2,
            JobCommandKind::Cancel {
                actor_principal_id: principal(5),
            },
        );
        let mut wrong_actor = request(&tenant, &owner, &job, &forged);
        wrong_actor.community_membership = Some(membership(
            principal(2),
            MembershipRole::Owner,
            MembershipStatus::Active,
        ));
        assert_eq!(
            authorize_job_transition(&wrong_actor),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::ActorMismatch)
        );
    }

    #[test]
    fn delegation_admission_limits_do_not_block_terminal_cleanup() {
        let tenant = tenant(community(1));
        let requested = requested_job();
        let initial_request = requested
            .history()
            .first()
            .expect("request command")
            .clone();
        let requester = account(principal(2), &[JOBS_WRITE_SCOPE]);
        let mut community_limit = request(&tenant, &requester, &requested, &initial_request);
        community_limit.community_membership = Some(membership(
            principal(2),
            MembershipRole::Member,
            MembershipStatus::Active,
        ));
        community_limit.active_community_jobs = MAX_ACTIVE_COMMUNITY_JOBS;
        assert_eq!(
            authorize_job_transition(&community_limit),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::CommunityJobLimitExceeded)
        );

        let requester_grant = JobDelegationGrant::new(
            community(1),
            aggregate(30),
            aggregate(10),
            principal(5),
            principal(2),
            AggregateVersion::FIRST,
            JobTransitionSet::all(),
            [],
            200,
        )
        .expect("requester delegation grant");
        let ancestry = [JobIdentity::new(community(1), aggregate(30)).expect("parent identity")];
        let mut child_limit = request(&tenant, &requester, &requested, &initial_request);
        child_limit.delegation = Some(&requester_grant);
        child_limit.ancestry = &ancestry;
        child_limit.active_child_jobs = MAX_ACTIVE_CHILD_JOBS;
        assert_eq!(
            authorize_job_transition(&child_limit),
            JobAuthorizationDecision::Denied(JobAuthorizationDenial::ActiveChildLimitExceeded)
        );

        let agent = agent();
        let mut job = requested_job();
        assert_eq!(
            job.apply(command(
                2,
                JobCommandKind::Accept {
                    executor_principal_id: principal(3),
                },
            )),
            Ok(JobCommandOutcome::Applied)
        );
        let failure = command(
            3,
            JobCommandKind::Error {
                actor_principal_id: principal(3),
            },
        );
        let grant = JobDelegationGrant::new(
            community(1),
            aggregate(30),
            aggregate(10),
            principal(2),
            principal(3),
            AggregateVersion::FIRST,
            JobTransitionSet::all(),
            [],
            200,
        )
        .expect("delegation grant");
        let ancestry = [JobIdentity::new(community(1), aggregate(30)).expect("parent identity")];
        let mut cleanup = request(&tenant, &agent, &job, &failure);
        cleanup.delegation = Some(&grant);
        cleanup.ancestry = &ancestry;
        cleanup.active_child_jobs = MAX_ACTIVE_CHILD_JOBS;
        cleanup.active_community_jobs = MAX_ACTIVE_COMMUNITY_JOBS;
        assert_eq!(
            authorize_job_transition(&cleanup),
            JobAuthorizationDecision::Allowed
        );
    }
}
