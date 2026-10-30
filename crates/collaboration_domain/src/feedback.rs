use std::{collections::BTreeSet, error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, MessageSource, NostrEventId, OperationId, PrincipalId, authorize,
};

const FEEDBACK_SUBMIT_SCOPE: &str = "feedback:submit";
const FEEDBACK_MANAGE_SCOPE: &str = "feedback:manage";
const MAX_FEEDBACK_BODY_BYTES: usize = 32 * 1_024;
const MAX_FEEDBACK_STATUS_MUTATIONS: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackCategory {
    Bug,
    Praise,
    NeedsWork,
}

#[derive(Clone, Eq, PartialEq)]
pub struct FeedbackBody(String);

impl FeedbackBody {
    pub fn new(value: impl Into<String>) -> Result<Self, FeedbackError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(FeedbackError::EmptyBody);
        }
        if trimmed.len() > MAX_FEEDBACK_BODY_BYTES {
            return Err(FeedbackError::BodyTooLarge);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for FeedbackBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FeedbackBody([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackStatus {
    Submitted,
    Reviewing,
    Resolved,
    Declined,
}

impl FeedbackStatus {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Resolved | Self::Declined)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackStatusReason {
    Acknowledged,
    Addressed,
    Duplicate,
    OutOfScope,
    UnableToReproduce,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeedbackStatusSource {
    pub operation_id: OperationId,
    pub occurred_at: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeedbackStatusMutation {
    pub source: FeedbackStatusSource,
    pub operator_principal_id: PrincipalId,
    pub status: FeedbackStatus,
    pub reason: FeedbackStatusReason,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeedbackCreateFields {
    pub community_id: CommunityId,
    pub source: MessageSource,
    pub category: Option<FeedbackCategory>,
    pub body: FeedbackBody,
}

#[derive(Clone, Eq, PartialEq)]
pub struct FeedbackRecordFields {
    pub community_id: CommunityId,
    pub source: MessageSource,
    pub submitter_principal_id: PrincipalId,
    pub category: Option<FeedbackCategory>,
    pub body: FeedbackBody,
    pub status: FeedbackStatus,
    pub status_mutations: Vec<FeedbackStatusMutation>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeedbackStatusView {
    pub community_id: CommunityId,
    pub feedback_event_id: NostrEventId,
    pub category: Option<FeedbackCategory>,
    pub status: FeedbackStatus,
    pub reason: Option<FeedbackStatusReason>,
    pub version: AggregateVersion,
    pub updated_at: u64,
}

#[derive(Clone, Eq, PartialEq)]
pub struct Feedback {
    fields: FeedbackRecordFields,
}

impl Feedback {
    pub fn submit(
        fields: FeedbackCreateFields,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, FeedbackError> {
        validate_submit_authorization(fields.community_id, authorization)?;
        fields
            .source
            .validate()
            .map_err(|_| FeedbackError::InvalidSource)?;
        let submitter_principal_id = authorization_subject(authorization);
        Ok(Self {
            fields: FeedbackRecordFields {
                community_id: fields.community_id,
                source: fields.source,
                submitter_principal_id,
                category: fields.category,
                body: fields.body,
                status: FeedbackStatus::Submitted,
                status_mutations: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn from_record(fields: FeedbackRecordFields) -> Result<Self, FeedbackError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &FeedbackRecordFields {
        &self.fields
    }

    pub fn update_status(
        &mut self,
        expected_version: AggregateVersion,
        status: FeedbackStatus,
        reason: FeedbackStatusReason,
        source: FeedbackStatusSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<FeedbackCommandOutcome, FeedbackError> {
        validate_operator_authorization(
            self.fields.community_id,
            AuthorizationAction::Manage,
            authorization,
        )?;
        validate_status_source(source)?;
        validate_status_reason(status, reason)?;
        let operator_principal_id = authorization_subject(authorization);
        if let Some(existing) = self
            .fields
            .status_mutations
            .iter()
            .find(|mutation| mutation.source.operation_id == source.operation_id)
        {
            if existing.source == source
                && existing.operator_principal_id == operator_principal_id
                && existing.status == status
                && existing.reason == reason
            {
                return Ok(FeedbackCommandOutcome::Unchanged);
            }
            return Err(FeedbackError::ConflictingOperation);
        }
        if self.fields.version != expected_version {
            return Err(FeedbackError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        if self.fields.status_mutations.len() >= MAX_FEEDBACK_STATUS_MUTATIONS {
            return Err(FeedbackError::TooManyStatusMutations);
        }
        validate_transition(self.fields.status, status)?;
        let latest_timestamp = self
            .fields
            .status_mutations
            .last()
            .map_or(self.fields.source.event_created_at, |mutation| {
                mutation.source.occurred_at
            });
        if source.occurred_at < latest_timestamp {
            return Err(FeedbackError::InvalidTimestamp);
        }
        let resulting_version = self
            .fields
            .version
            .next()
            .ok_or(FeedbackError::VersionExhausted)?;
        self.fields.status_mutations.push(FeedbackStatusMutation {
            source,
            operator_principal_id,
            status,
            reason,
            resulting_version,
        });
        self.fields.status = status;
        self.fields.version = resulting_version;
        Ok(FeedbackCommandOutcome::Applied)
    }

    pub fn status_view(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<FeedbackStatusView, FeedbackError> {
        validate_operator_authorization(
            self.fields.community_id,
            AuthorizationAction::Read,
            authorization,
        )?;
        let latest = self.fields.status_mutations.last();
        Ok(FeedbackStatusView {
            community_id: self.fields.community_id,
            feedback_event_id: self.fields.source.event_id,
            category: self.fields.category,
            status: self.fields.status,
            reason: latest.map(|mutation| mutation.reason),
            version: self.fields.version,
            updated_at: latest.map_or(self.fields.source.event_created_at, |mutation| {
                mutation.source.occurred_at
            }),
        })
    }
}

impl fmt::Debug for Feedback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Feedback")
            .field("community_id", &self.fields.community_id)
            .field("status", &self.fields.status)
            .field("status_mutation_count", &self.fields.status_mutations.len())
            .field("version", &self.fields.version)
            .field("private_submission", &"[REDACTED]")
            .finish()
    }
}

impl fmt::Debug for FeedbackRecordFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FeedbackRecordFields")
            .field("community_id", &self.community_id)
            .field("status", &self.status)
            .field("status_mutation_count", &self.status_mutations.len())
            .field("version", &self.version)
            .field("private_submission", &"[REDACTED]")
            .finish()
    }
}

fn validate_record(fields: &FeedbackRecordFields) -> Result<(), FeedbackError> {
    if fields.community_id.as_uuid().is_nil() || fields.submitter_principal_id.as_uuid().is_nil() {
        return Err(FeedbackError::InvalidIdentity);
    }
    fields
        .source
        .validate()
        .map_err(|_| FeedbackError::InvalidSource)?;
    if fields.status_mutations.len() > MAX_FEEDBACK_STATUS_MUTATIONS {
        return Err(FeedbackError::TooManyStatusMutations);
    }
    let mut status = FeedbackStatus::Submitted;
    let mut version = AggregateVersion::FIRST;
    let mut timestamp = fields.source.event_created_at;
    let mut operations = BTreeSet::new();
    for mutation in &fields.status_mutations {
        validate_status_source(mutation.source)?;
        validate_status_reason(mutation.status, mutation.reason)?;
        if mutation.operator_principal_id.as_uuid().is_nil()
            || mutation.source.occurred_at < timestamp
            || !operations.insert(mutation.source.operation_id)
            || !mutation.resulting_version.follows(version)
        {
            return Err(FeedbackError::InvalidRecord);
        }
        validate_transition(status, mutation.status)?;
        status = mutation.status;
        version = mutation.resulting_version;
        timestamp = mutation.source.occurred_at;
    }
    if fields.status != status || fields.version != version {
        return Err(FeedbackError::InvalidRecord);
    }
    Ok(())
}

fn validate_submit_authorization(
    community_id: CommunityId,
    request: &AuthorizationRequest<'_>,
) -> Result<(), FeedbackError> {
    if request.required_scope.as_str() != FEEDBACK_SUBMIT_SCOPE
        || request.action != AuthorizationAction::Write
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Community
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.channel_id.is_some()
        || request.resource.owner_principal_id.is_some()
    {
        return Err(FeedbackError::AuthorizationShape);
    }
    authorize_request(request)
}

fn validate_operator_authorization(
    community_id: CommunityId,
    action: AuthorizationAction,
    request: &AuthorizationRequest<'_>,
) -> Result<(), FeedbackError> {
    if request.required_scope.as_str() != FEEDBACK_MANAGE_SCOPE
        || request.action != action
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Administration
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.channel_id.is_some()
        || request.resource.owner_principal_id.is_some()
    {
        return Err(FeedbackError::AuthorizationShape);
    }
    authorize_request(request)?;
    let membership = request
        .community_membership
        .ok_or(FeedbackError::Unauthorized(
            AuthorizationDenial::MissingMembership,
        ))?;
    if !matches!(
        membership.role,
        crate::MembershipRole::Owner | crate::MembershipRole::Admin
    ) {
        return Err(FeedbackError::Unauthorized(
            AuthorizationDenial::InsufficientRole,
        ));
    }
    Ok(())
}

fn authorize_request(request: &AuthorizationRequest<'_>) -> Result<(), FeedbackError> {
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(FeedbackError::Unauthorized(denial)),
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

fn validate_status_source(source: FeedbackStatusSource) -> Result<(), FeedbackError> {
    if source.operation_id.as_uuid().is_nil() || source.occurred_at == 0 {
        return Err(FeedbackError::InvalidSource);
    }
    Ok(())
}

fn validate_status_reason(
    status: FeedbackStatus,
    reason: FeedbackStatusReason,
) -> Result<(), FeedbackError> {
    let valid = match status {
        FeedbackStatus::Submitted => false,
        FeedbackStatus::Reviewing => reason == FeedbackStatusReason::Acknowledged,
        FeedbackStatus::Resolved => matches!(
            reason,
            FeedbackStatusReason::Addressed | FeedbackStatusReason::UnableToReproduce
        ),
        FeedbackStatus::Declined => matches!(
            reason,
            FeedbackStatusReason::Duplicate | FeedbackStatusReason::OutOfScope
        ),
    };
    if valid {
        Ok(())
    } else {
        Err(FeedbackError::InvalidStatusReason)
    }
}

fn validate_transition(current: FeedbackStatus, next: FeedbackStatus) -> Result<(), FeedbackError> {
    if current.is_terminal() || current == next || next == FeedbackStatus::Submitted {
        return Err(FeedbackError::InvalidStatusTransition);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackError {
    EmptyBody,
    BodyTooLarge,
    InvalidIdentity,
    InvalidSource,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    InvalidStatusReason,
    InvalidStatusTransition,
    InvalidTimestamp,
    ConflictingOperation,
    TooManyStatusMutations,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    VersionExhausted,
    InvalidRecord,
}

impl fmt::Display for FeedbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBody => formatter.write_str("feedback body must not be empty"),
            Self::BodyTooLarge => formatter.write_str("feedback body is too large"),
            Self::InvalidIdentity => formatter.write_str("feedback identity is invalid"),
            Self::InvalidSource => formatter.write_str("feedback source is invalid"),
            Self::AuthorizationShape | Self::Unauthorized(_) => {
                formatter.write_str("feedback operation is not authorized")
            }
            Self::InvalidStatusReason => formatter.write_str("feedback status reason is invalid"),
            Self::InvalidStatusTransition => {
                formatter.write_str("feedback status transition is invalid")
            }
            Self::InvalidTimestamp => formatter.write_str("feedback timestamp is invalid"),
            Self::ConflictingOperation => formatter.write_str("feedback operation conflicts"),
            Self::TooManyStatusMutations => formatter.write_str("feedback status history is full"),
            Self::StaleVersion { .. } => formatter.write_str("feedback version is stale"),
            Self::VersionExhausted => formatter.write_str("feedback version is exhausted"),
            Self::InvalidRecord => formatter.write_str("feedback record is invalid"),
        }
    }
}

impl Error for FeedbackError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, CommunityMembership,
        MembershipRole, MembershipStatus, PrincipalScopes, ServiceAccountId, TenantContext,
        TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community_id(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "feedback-test")
                    .expect("trusted tenant route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn scope(value: &str) -> AuthorizationScope {
        AuthorizationScope::new(value).expect("authorization scope")
    }

    fn principal(
        community_id: CommunityId,
        principal_id: PrincipalId,
        scopes: impl IntoIterator<Item = AuthorizationScope>,
    ) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id,
            community_id,
            ServiceAccountId::new(principal_id.as_uuid().as_u128() as u64),
            PrincipalScopes::new(scopes).expect("principal scopes"),
        )
    }

    fn authorization_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        action: AuthorizationAction,
        resource_community_id: CommunityId,
        resource_kind: AuthorizationResourceKind,
        membership_community_id: CommunityId,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope,
            action,
            resource: AuthorizationResource {
                community_id: resource_community_id,
                kind: resource_kind,
                resource_id: AggregateId::from_uuid(resource_community_id.as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: membership_community_id,
                principal_id: principal.principal_id(),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn source(value: u8, event_created_at: u64) -> MessageSource {
        MessageSource {
            event_id: NostrEventId::from_bytes([value; 32]),
            event_created_at,
        }
    }

    fn status_source(value: u128, occurred_at: u64) -> FeedbackStatusSource {
        FeedbackStatusSource {
            operation_id: OperationId::from_uuid(Uuid::from_u128(value)),
            occurred_at,
        }
    }

    fn submitted_feedback() -> Feedback {
        let community_id = community_id(1);
        let tenant = tenant(community_id);
        let submit_scope = scope(FEEDBACK_SUBMIT_SCOPE);
        let submitter = principal(community_id, principal_id(10), [submit_scope.clone()]);
        let request = authorization_request(
            &tenant,
            &submitter,
            &submit_scope,
            AuthorizationAction::Write,
            community_id,
            AuthorizationResourceKind::Community,
            community_id,
            MembershipRole::Member,
        );
        Feedback::submit(
            FeedbackCreateFields {
                community_id,
                source: source(1, 10),
                category: Some(FeedbackCategory::Bug),
                body: FeedbackBody::new("private feedback").expect("feedback body"),
            },
            &request,
        )
        .expect("authorized feedback")
    }

    #[test]
    fn authorized_member_submits_private_feedback() {
        let feedback = submitted_feedback();

        assert_eq!(feedback.fields().community_id, community_id(1));
        assert_eq!(feedback.fields().submitter_principal_id, principal_id(10));
        assert_eq!(feedback.fields().category, Some(FeedbackCategory::Bug));
        assert_eq!(feedback.fields().body.as_str(), "private feedback");
        assert_eq!(feedback.fields().status, FeedbackStatus::Submitted);
        assert_eq!(feedback.fields().version, AggregateVersion::FIRST);
        assert!(feedback.fields().status_mutations.is_empty());
    }

    #[test]
    fn feedback_debug_and_operator_projection_redact_private_context() {
        let feedback = submitted_feedback();
        let record_debug = format!("{:?}", feedback.fields());
        let feedback_debug = format!("{feedback:?}");
        for rendered in [&record_debug, &feedback_debug] {
            assert!(!rendered.contains("private feedback"));
            assert!(!rendered.contains(&principal_id(10).as_uuid().to_string()));
            assert!(rendered.contains("[REDACTED]"));
        }

        let community_id = community_id(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(FEEDBACK_MANAGE_SCOPE);
        let operator = principal(community_id, principal_id(20), [manage_scope.clone()]);
        let read_request = authorization_request(
            &tenant,
            &operator,
            &manage_scope,
            AuthorizationAction::Read,
            community_id,
            AuthorizationResourceKind::Administration,
            community_id,
            MembershipRole::Admin,
        );
        let view = feedback
            .status_view(&read_request)
            .expect("authorized status projection");
        assert_eq!(view.feedback_event_id, source(1, 10).event_id);
        assert_eq!(view.status, FeedbackStatus::Submitted);
        assert_eq!(view.reason, None);
    }

    #[test]
    fn admin_updates_status_with_versioning_and_idempotency() {
        let mut feedback = submitted_feedback();
        let community_id = community_id(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(FEEDBACK_MANAGE_SCOPE);
        let operator = principal(community_id, principal_id(20), [manage_scope.clone()]);
        let manage_request = authorization_request(
            &tenant,
            &operator,
            &manage_scope,
            AuthorizationAction::Manage,
            community_id,
            AuthorizationResourceKind::Administration,
            community_id,
            MembershipRole::Admin,
        );
        let first_source = status_source(100, 20);
        assert_eq!(
            feedback.update_status(
                AggregateVersion::FIRST,
                FeedbackStatus::Reviewing,
                FeedbackStatusReason::Acknowledged,
                first_source,
                &manage_request,
            ),
            Ok(FeedbackCommandOutcome::Applied)
        );
        assert_eq!(
            feedback.update_status(
                AggregateVersion::FIRST,
                FeedbackStatus::Reviewing,
                FeedbackStatusReason::Acknowledged,
                first_source,
                &manage_request,
            ),
            Ok(FeedbackCommandOutcome::Unchanged)
        );

        let second_version = AggregateVersion::new(2).expect("second version");
        assert_eq!(
            feedback.update_status(
                second_version,
                FeedbackStatus::Resolved,
                FeedbackStatusReason::Addressed,
                status_source(101, 30),
                &manage_request,
            ),
            Ok(FeedbackCommandOutcome::Applied)
        );
        assert_eq!(feedback.fields().status, FeedbackStatus::Resolved);
        assert_eq!(feedback.fields().version.get(), 3);
        assert_eq!(feedback.fields().status_mutations.len(), 2);
        assert_eq!(
            Feedback::from_record(feedback.fields().clone()),
            Ok(feedback)
        );
    }

    #[test]
    fn member_cannot_read_or_update_operator_status() {
        let mut feedback = submitted_feedback();
        let community_id = community_id(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(FEEDBACK_MANAGE_SCOPE);
        let member = principal(community_id, principal_id(30), [manage_scope.clone()]);
        let read_request = authorization_request(
            &tenant,
            &member,
            &manage_scope,
            AuthorizationAction::Read,
            community_id,
            AuthorizationResourceKind::Administration,
            community_id,
            MembershipRole::Member,
        );
        assert_eq!(
            feedback.status_view(&read_request),
            Err(FeedbackError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );

        let manage_request = authorization_request(
            &tenant,
            &member,
            &manage_scope,
            AuthorizationAction::Manage,
            community_id,
            AuthorizationResourceKind::Administration,
            community_id,
            MembershipRole::Member,
        );
        assert_eq!(
            feedback.update_status(
                AggregateVersion::FIRST,
                FeedbackStatus::Reviewing,
                FeedbackStatusReason::Acknowledged,
                status_source(200, 20),
                &manage_request,
            ),
            Err(FeedbackError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );
        assert!(feedback.fields().status_mutations.is_empty());
    }

    #[test]
    fn foreign_tenant_cannot_submit_or_read_feedback() {
        let target_community_id = community_id(1);
        let foreign_community_id = community_id(2);
        let foreign_tenant = tenant(foreign_community_id);
        let submit_scope = scope(FEEDBACK_SUBMIT_SCOPE);
        let manage_scope = scope(FEEDBACK_MANAGE_SCOPE);
        let foreign_principal = principal(
            foreign_community_id,
            principal_id(40),
            [submit_scope.clone(), manage_scope.clone()],
        );
        let submit_request = authorization_request(
            &foreign_tenant,
            &foreign_principal,
            &submit_scope,
            AuthorizationAction::Write,
            target_community_id,
            AuthorizationResourceKind::Community,
            foreign_community_id,
            MembershipRole::Member,
        );
        assert_eq!(
            Feedback::submit(
                FeedbackCreateFields {
                    community_id: target_community_id,
                    source: source(2, 10),
                    category: None,
                    body: FeedbackBody::new("foreign feedback").expect("feedback body"),
                },
                &submit_request,
            ),
            Err(FeedbackError::Unauthorized(
                AuthorizationDenial::TenantMismatch
            ))
        );

        let feedback = submitted_feedback();
        let read_request = authorization_request(
            &foreign_tenant,
            &foreign_principal,
            &manage_scope,
            AuthorizationAction::Read,
            target_community_id,
            AuthorizationResourceKind::Administration,
            foreign_community_id,
            MembershipRole::Admin,
        );
        assert_eq!(
            feedback.status_view(&read_request),
            Err(FeedbackError::Unauthorized(
                AuthorizationDenial::TenantMismatch
            ))
        );
    }
}
