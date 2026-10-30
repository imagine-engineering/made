use std::fmt;

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, AuditOutcome, AuthorizationAction, AuthorizationResourceKind,
    AuthorizationScope, CommunityId, CommunityMembership, MembershipRole,
    ModerationAuthorizationDecision, ModerationAuthorizationDenial, ModerationAuthorizationRequest,
    ModerationCommandSource, ModerationError, ModerationReport, ModerationReportContext,
    ModerationReportReason, ModerationReportTarget, ModerationResolution, ModerationRestriction,
    NostrPublicKey, OperationId, PrincipalId, authorize_with_moderation,
};

use crate::audit::events::{AuditEventContext, ModerationAuditOperation, SecurityAuditEvent};

const MODERATION_REPORT_SCOPE: &str = "moderation:report";
const MODERATION_MANAGE_SCOPE: &str = "moderation:manage";
const COMMUNITY_MANAGE_SCOPE: &str = "communities:manage";
pub const MAX_OPERATOR_REPORTS: u16 = 200;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveVersionFence {
    expected: Option<AggregateVersion>,
    current: Option<AggregateVersion>,
}

impl ArchiveVersionFence {
    pub const fn new(
        expected: Option<AggregateVersion>,
        current: Option<AggregateVersion>,
    ) -> Self {
        Self { expected, current }
    }

    fn next_version(self) -> Result<AggregateVersion, ModerationOperatorError> {
        match (self.expected, self.current) {
            (None, None) => Ok(AggregateVersion::FIRST),
            (Some(expected), Some(current)) if expected == current => current
                .next()
                .ok_or(ModerationOperatorError::InvalidRequest),
            (None, Some(_)) | (Some(_), None) | (Some(_), Some(_)) => {
                Err(ModerationOperatorError::StaleAction)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModerationOperatorCommand {
    ListReports {
        limit: u16,
        source: ModerationCommandSource,
    },
    FileReport {
        report_id: AggregateId,
        target: ModerationReportTarget,
        reason: ModerationReportReason,
        private_context: Option<ModerationReportContext>,
        source: ModerationCommandSource,
    },
    ResolveReport {
        report: ModerationReport,
        expected_version: AggregateVersion,
        resolution: ModerationResolution,
        source: ModerationCommandSource,
    },
    ApplyBan {
        restriction: ModerationRestriction,
        expected_version: AggregateVersion,
        expires_at_millis: Option<u64>,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        source: ModerationCommandSource,
    },
    ApplyTimeout {
        restriction: ModerationRestriction,
        expected_version: AggregateVersion,
        expires_at_millis: u64,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        source: ModerationCommandSource,
    },
    LiftBan {
        restriction: ModerationRestriction,
        expected_version: AggregateVersion,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        source: ModerationCommandSource,
    },
    LiftTimeout {
        restriction: ModerationRestriction,
        expected_version: AggregateVersion,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        source: ModerationCommandSource,
    },
    ArchiveIdentity {
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        identity_public_key: NostrPublicKey,
        version_fence: ArchiveVersionFence,
        source: ModerationCommandSource,
    },
    ArchiveCommunity {
        version_fence: ArchiveVersionFence,
        source: ModerationCommandSource,
    },
}

impl ModerationOperatorCommand {
    const fn source(&self) -> ModerationCommandSource {
        match self {
            Self::ListReports { source, .. }
            | Self::FileReport { source, .. }
            | Self::ResolveReport { source, .. }
            | Self::ApplyBan { source, .. }
            | Self::ApplyTimeout { source, .. }
            | Self::LiftBan { source, .. }
            | Self::LiftTimeout { source, .. }
            | Self::ArchiveIdentity { source, .. }
            | Self::ArchiveCommunity { source, .. } => *source,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityArchiveCommand {
    pub community_id: CommunityId,
    pub target_principal_id: PrincipalId,
    pub identity_public_key: NostrPublicKey,
    pub archive_version: AggregateVersion,
    pub actor_principal_id: PrincipalId,
    pub source: ModerationCommandSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityArchiveCommand {
    pub community_id: CommunityId,
    pub archive_version: AggregateVersion,
    pub actor_principal_id: PrincipalId,
    pub source: ModerationCommandSource,
}

#[derive(Clone, Eq, PartialEq)]
pub enum ModerationOperatorWrite {
    Report(ModerationReport),
    Restriction(ModerationRestriction),
    IdentityArchive(IdentityArchiveCommand),
    CommunityArchive(CommunityArchiveCommand),
}

impl fmt::Debug for ModerationOperatorWrite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Report(report) => formatter.debug_tuple("Report").field(report).finish(),
            Self::Restriction(restriction) => formatter
                .debug_tuple("Restriction")
                .field(restriction)
                .finish(),
            Self::IdentityArchive(command) => formatter
                .debug_struct("IdentityArchive")
                .field("community_id", &command.community_id)
                .field("archive_version", &command.archive_version)
                .field("identity", &"[REDACTED]")
                .finish(),
            Self::CommunityArchive(command) => formatter
                .debug_tuple("CommunityArchive")
                .field(command)
                .finish(),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum ModerationBackendCommand {
    ListReports { limit: u16 },
    Write(ModerationOperatorWrite),
}

impl fmt::Debug for ModerationBackendCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ListReports { limit } => formatter
                .debug_struct("ListReports")
                .field("limit", limit)
                .finish(),
            Self::Write(write) => formatter.debug_tuple("Write").field(write).finish(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModerationWriteReceipt {
    pub operation_id: OperationId,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModerationBackendResponse {
    Reports(Vec<ModerationReport>),
    Written(ModerationWriteReceipt),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedModerationOperation {
    command: ModerationBackendCommand,
    audit_event: SecurityAuditEvent,
    expected_receipt: Option<ModerationWriteReceipt>,
}

impl AuthorizedModerationOperation {
    pub const fn command(&self) -> &ModerationBackendCommand {
        &self.command
    }

    pub const fn audit_event(&self) -> &SecurityAuditEvent {
        &self.audit_event
    }

    pub const fn expected_receipt(&self) -> Option<ModerationWriteReceipt> {
        self.expected_receipt
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationBackendError {
    Conflict,
    Unavailable,
    PartialFailure,
    InvalidData,
}

#[async_trait]
pub trait ModerationOperatorBackend: Send + Sync {
    async fn execute(
        &self,
        community_id: CommunityId,
        operation: &AuthorizedModerationOperation,
    ) -> Result<ModerationBackendResponse, ModerationBackendError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModerationOperatorOutcome {
    Reports(Vec<ModerationReport>),
    Applied(ModerationWriteReceipt),
}

pub struct ModerationOperatorApi<B> {
    backend: B,
}

impl<B> ModerationOperatorApi<B>
where
    B: ModerationOperatorBackend,
{
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> B {
        self.backend
    }

    pub async fn execute(
        &self,
        authorization: &ModerationAuthorizationRequest<'_, '_>,
        command: ModerationOperatorCommand,
    ) -> Result<ModerationOperatorOutcome, ModerationOperatorError> {
        let community_id = authorization.authorization.tenant.community_id();
        let limit = match &command {
            ModerationOperatorCommand::ListReports { limit, .. } => Some(*limit),
            _ => None,
        };
        let operation = prepare_operation(authorization, command)?;
        let response = self
            .backend
            .execute(community_id, &operation)
            .await
            .map_err(ModerationOperatorError::from_backend)?;
        validate_response(community_id, limit, operation.expected_receipt, response)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationOperatorError {
    AuthorizationDenied,
    TenantMismatch,
    InvalidRequest,
    StaleAction,
    Unavailable,
    PartialFailure,
    InvalidBackendResponse,
}

impl ModerationOperatorError {
    pub const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::AuthorizationDenied => "moderation_operator_denied",
            Self::TenantMismatch => "moderation_operator_tenant_mismatch",
            Self::InvalidRequest => "moderation_operator_invalid_request",
            Self::StaleAction => "moderation_operator_stale_action",
            Self::Unavailable => "moderation_operator_unavailable",
            Self::PartialFailure => "moderation_operator_partial_failure",
            Self::InvalidBackendResponse => "moderation_operator_invalid_backend_response",
        }
    }

    const fn from_backend(error: ModerationBackendError) -> Self {
        match error {
            ModerationBackendError::Conflict => Self::StaleAction,
            ModerationBackendError::Unavailable => Self::Unavailable,
            ModerationBackendError::PartialFailure => Self::PartialFailure,
            ModerationBackendError::InvalidData => Self::InvalidBackendResponse,
        }
    }
}

impl fmt::Display for ModerationOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "moderation operator request failed ({})",
            self.diagnostic_code()
        )
    }
}

impl std::error::Error for ModerationOperatorError {}

fn prepare_operation(
    authorization: &ModerationAuthorizationRequest<'_, '_>,
    command: ModerationOperatorCommand,
) -> Result<AuthorizedModerationOperation, ModerationOperatorError> {
    validate_authorization_shape(authorization.authorization, &command)?;
    let control = if matches!(
        &command,
        ModerationOperatorCommand::ApplyBan { .. }
            | ModerationOperatorCommand::ApplyTimeout { .. }
            | ModerationOperatorCommand::LiftBan { .. }
            | ModerationOperatorCommand::LiftTimeout { .. }
            | ModerationOperatorCommand::ResolveReport { .. }
    ) {
        Some(
            collaboration_domain::moderation_policy::admit_moderation_control_plane(
                authorization,
                command.source(),
            )
            .map_err(map_policy_denial)?,
        )
    } else {
        None
    };
    let control_authorization = ModerationAuthorizationRequest {
        authorization: authorization.authorization,
        restriction: authorization.restriction,
        principal_archive: authorization.principal_archive,
        community_archive: authorization.community_archive,
        resource_context: control
            .map(collaboration_domain::ModerationResourceContext::RestrictionManagement)
            .unwrap_or(authorization.resource_context),
    };
    match authorize_with_moderation(&control_authorization) {
        ModerationAuthorizationDecision::Allowed(_) => {}
        ModerationAuthorizationDecision::Denied(denial) => {
            return Err(map_policy_denial(denial));
        }
    }

    let request = authorization.authorization;
    let community_id = request.tenant.community_id();
    let actor_principal_id = effective_principal_id(request);
    let source = command.source();
    let context = AuditEventContext::new(
        request.tenant,
        source.operation_id,
        Some(request.principal),
        AuditOutcome::Succeeded,
        None,
        source.occurred_at_millis,
    )
    .map_err(|_| ModerationOperatorError::InvalidRequest)?;

    let (backend_command, audit_operation, subject_principal_id, record_id, expected_receipt) =
        match command {
            ModerationOperatorCommand::ListReports { limit, .. } => {
                require_operator(request)?;
                if !(1..=MAX_OPERATOR_REPORTS).contains(&limit) {
                    return Err(ModerationOperatorError::InvalidRequest);
                }
                (
                    ModerationBackendCommand::ListReports { limit },
                    ModerationAuditOperation::ViewQueue,
                    None,
                    community_id.as_uuid(),
                    None,
                )
            }
            ModerationOperatorCommand::FileReport {
                report_id,
                target,
                reason,
                private_context,
                source,
            } => {
                let subject_principal_id = match target {
                    ModerationReportTarget::Principal(principal_id) => Some(principal_id),
                    ModerationReportTarget::Event(_) | ModerationReportTarget::BlobSha256(_) => {
                        None
                    }
                };
                let report = ModerationReport::file(
                    report_id,
                    community_id,
                    target,
                    reason,
                    private_context,
                    source,
                    request,
                )
                .map_err(map_domain_error)?;
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: report.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Report(report)),
                    ModerationAuditOperation::Report,
                    subject_principal_id,
                    report_id.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::ResolveReport {
                mut report,
                expected_version,
                resolution,
                source,
            } => {
                require_operator(request)?;
                report
                    .resolve(expected_version, resolution, source, request)
                    .map_err(map_domain_error)?;
                let report_id = report.fields().report_id;
                let subject_principal_id = match report.fields().target {
                    ModerationReportTarget::Principal(principal_id) => Some(principal_id),
                    ModerationReportTarget::Event(_) | ModerationReportTarget::BlobSha256(_) => {
                        None
                    }
                };
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: report.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Report(report)),
                    ModerationAuditOperation::ResolveReport,
                    subject_principal_id,
                    report_id.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::ApplyBan {
                mut restriction,
                expected_version,
                expires_at_millis,
                target_membership,
                current_target_membership_version,
                source,
            } => {
                require_operator(request)?;
                restriction
                    .apply_ban(
                        expected_version,
                        expires_at_millis,
                        source,
                        target_membership,
                        current_target_membership_version,
                        request,
                    )
                    .map_err(map_domain_error)?;
                let target_principal_id = restriction.fields().target_principal_id;
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: restriction.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Restriction(
                        restriction,
                    )),
                    ModerationAuditOperation::ApplyRestriction,
                    Some(target_principal_id),
                    target_principal_id.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::ApplyTimeout {
                mut restriction,
                expected_version,
                expires_at_millis,
                target_membership,
                current_target_membership_version,
                source,
            } => {
                require_operator(request)?;
                restriction
                    .apply_timeout(
                        expected_version,
                        expires_at_millis,
                        source,
                        target_membership,
                        current_target_membership_version,
                        request,
                    )
                    .map_err(map_domain_error)?;
                let target_principal_id = restriction.fields().target_principal_id;
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: restriction.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Restriction(
                        restriction,
                    )),
                    ModerationAuditOperation::ApplyRestriction,
                    Some(target_principal_id),
                    target_principal_id.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::LiftBan {
                mut restriction,
                expected_version,
                target_membership,
                current_target_membership_version,
                source,
            } => {
                require_operator(request)?;
                let outcome = restriction
                    .lift_ban(
                        expected_version,
                        source,
                        target_membership,
                        current_target_membership_version,
                        request,
                    )
                    .map_err(map_domain_error)?;
                if outcome != collaboration_domain::ModerationCommandOutcome::Applied {
                    return Err(ModerationOperatorError::InvalidRequest);
                }
                let target = restriction.fields().target_principal_id;
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: restriction.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Restriction(
                        restriction,
                    )),
                    ModerationAuditOperation::LiftRestriction,
                    Some(target),
                    target.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::LiftTimeout {
                mut restriction,
                expected_version,
                target_membership,
                current_target_membership_version,
                source,
            } => {
                require_operator(request)?;
                let outcome = restriction
                    .lift_timeout(
                        expected_version,
                        source,
                        target_membership,
                        current_target_membership_version,
                        request,
                    )
                    .map_err(map_domain_error)?;
                if outcome != collaboration_domain::ModerationCommandOutcome::Applied {
                    return Err(ModerationOperatorError::InvalidRequest);
                }
                let target = restriction.fields().target_principal_id;
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: restriction.fields().version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::Restriction(
                        restriction,
                    )),
                    ModerationAuditOperation::LiftRestriction,
                    Some(target),
                    target.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::ArchiveIdentity {
                target_membership,
                current_target_membership_version,
                identity_public_key,
                version_fence,
                source,
            } => {
                require_operator_target(
                    request,
                    target_membership,
                    current_target_membership_version,
                )?;
                let archive_version = version_fence.next_version()?;
                let command = IdentityArchiveCommand {
                    community_id,
                    target_principal_id: target_membership.principal_id,
                    identity_public_key,
                    archive_version,
                    actor_principal_id,
                    source,
                };
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: archive_version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::IdentityArchive(
                        command,
                    )),
                    ModerationAuditOperation::ArchiveIdentity,
                    Some(target_membership.principal_id),
                    target_membership.principal_id.as_uuid(),
                    Some(receipt),
                )
            }
            ModerationOperatorCommand::ArchiveCommunity {
                version_fence,
                source,
            } => {
                require_owner(request)?;
                let archive_version = version_fence.next_version()?;
                let command = CommunityArchiveCommand {
                    community_id,
                    archive_version,
                    actor_principal_id,
                    source,
                };
                let receipt = ModerationWriteReceipt {
                    operation_id: source.operation_id,
                    version: archive_version,
                };
                (
                    ModerationBackendCommand::Write(ModerationOperatorWrite::CommunityArchive(
                        command,
                    )),
                    ModerationAuditOperation::ArchiveCommunity,
                    None,
                    community_id.as_uuid(),
                    Some(receipt),
                )
            }
        };

    let audit_event = SecurityAuditEvent::Moderation {
        context,
        operation: audit_operation,
        subject_principal_id,
        record_id,
    };
    audit_event
        .clone()
        .into_record()
        .map_err(|_| ModerationOperatorError::InvalidRequest)?;
    Ok(AuthorizedModerationOperation {
        command: backend_command,
        audit_event,
        expected_receipt,
    })
}

fn validate_authorization_shape(
    request: &collaboration_domain::AuthorizationRequest<'_>,
    command: &ModerationOperatorCommand,
) -> Result<(), ModerationOperatorError> {
    let (scope, action, resource_kind) = match command {
        ModerationOperatorCommand::ListReports { .. } => (
            MODERATION_MANAGE_SCOPE,
            AuthorizationAction::Read,
            AuthorizationResourceKind::Administration,
        ),
        ModerationOperatorCommand::FileReport { .. } => (
            MODERATION_REPORT_SCOPE,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
        ),
        ModerationOperatorCommand::ResolveReport { .. }
        | ModerationOperatorCommand::ApplyBan { .. }
        | ModerationOperatorCommand::ApplyTimeout { .. }
        | ModerationOperatorCommand::LiftBan { .. }
        | ModerationOperatorCommand::LiftTimeout { .. }
        | ModerationOperatorCommand::ArchiveIdentity { .. } => (
            MODERATION_MANAGE_SCOPE,
            AuthorizationAction::Manage,
            AuthorizationResourceKind::Administration,
        ),
        ModerationOperatorCommand::ArchiveCommunity { .. } => (
            COMMUNITY_MANAGE_SCOPE,
            AuthorizationAction::Delete,
            AuthorizationResourceKind::Community,
        ),
    };
    let expected_scope =
        AuthorizationScope::new(scope).map_err(|_| ModerationOperatorError::InvalidRequest)?;
    if request.required_scope != &expected_scope
        || request.action != action
        || request.resource.kind != resource_kind
        || request.resource.resource_id.as_uuid() != request.tenant.community_id().as_uuid()
    {
        return Err(ModerationOperatorError::AuthorizationDenied);
    }
    Ok(())
}

fn require_operator(
    request: &collaboration_domain::AuthorizationRequest<'_>,
) -> Result<MembershipRole, ModerationOperatorError> {
    let role = request
        .community_membership
        .map(|membership| membership.role)
        .ok_or(ModerationOperatorError::AuthorizationDenied)?;
    match role {
        MembershipRole::Owner | MembershipRole::Admin => Ok(role),
        MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot => {
            Err(ModerationOperatorError::AuthorizationDenied)
        }
    }
}

fn require_owner(
    request: &collaboration_domain::AuthorizationRequest<'_>,
) -> Result<(), ModerationOperatorError> {
    match require_operator(request)? {
        MembershipRole::Owner => Ok(()),
        MembershipRole::Admin => Err(ModerationOperatorError::AuthorizationDenied),
        MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot => {
            Err(ModerationOperatorError::AuthorizationDenied)
        }
    }
}

fn require_operator_target(
    request: &collaboration_domain::AuthorizationRequest<'_>,
    target_membership: CommunityMembership,
    current_target_membership_version: AggregateVersion,
) -> Result<(), ModerationOperatorError> {
    let actor_role = require_operator(request)?;
    let actor_principal_id = effective_principal_id(request);
    if target_membership.community_id != request.tenant.community_id() {
        return Err(ModerationOperatorError::TenantMismatch);
    }
    if target_membership.version != current_target_membership_version {
        return Err(ModerationOperatorError::StaleAction);
    }
    if target_membership.principal_id == actor_principal_id
        || target_membership.role == MembershipRole::Owner
        || (actor_role == MembershipRole::Admin && target_membership.role == MembershipRole::Admin)
    {
        return Err(ModerationOperatorError::AuthorizationDenied);
    }
    Ok(())
}

fn effective_principal_id(request: &collaboration_domain::AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        collaboration_domain::AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

fn map_policy_denial(denial: ModerationAuthorizationDenial) -> ModerationOperatorError {
    match denial {
        ModerationAuthorizationDenial::TenantMismatch => ModerationOperatorError::TenantMismatch,
        ModerationAuthorizationDenial::PolicyUnavailable
        | ModerationAuthorizationDenial::AmbiguousPolicyState => {
            ModerationOperatorError::Unavailable
        }
        ModerationAuthorizationDenial::Authorization(_)
        | ModerationAuthorizationDenial::InvalidPolicyInput
        | ModerationAuthorizationDenial::Banned
        | ModerationAuthorizationDenial::TimedOut
        | ModerationAuthorizationDenial::IdentityArchived
        | ModerationAuthorizationDenial::CommunityArchived => {
            ModerationOperatorError::AuthorizationDenied
        }
    }
}

fn map_domain_error(error: ModerationError) -> ModerationOperatorError {
    match error {
        ModerationError::TenantMismatch => ModerationOperatorError::TenantMismatch,
        ModerationError::StaleTarget
        | ModerationError::StaleVersion { .. }
        | ModerationError::ConflictingOperation => ModerationOperatorError::StaleAction,
        ModerationError::AuthorizationShape
        | ModerationError::Unauthorized(_)
        | ModerationError::ProtectedTarget
        | ModerationError::SelfRestriction
        | ModerationError::PersonalMuteOwnerMismatch => {
            ModerationOperatorError::AuthorizationDenied
        }
        ModerationError::InvalidIdentity
        | ModerationError::InvalidOperationId
        | ModerationError::InvalidReportContext
        | ModerationError::InvalidTimestamp
        | ModerationError::InvalidExpiry
        | ModerationError::InvalidTransition
        | ModerationError::SelfMute
        | ModerationError::TooManyTransitions
        | ModerationError::VersionExhausted
        | ModerationError::InvalidRecord => ModerationOperatorError::InvalidRequest,
    }
}

fn validate_response(
    community_id: CommunityId,
    limit: Option<u16>,
    expected_receipt: Option<ModerationWriteReceipt>,
    response: ModerationBackendResponse,
) -> Result<ModerationOperatorOutcome, ModerationOperatorError> {
    match (limit, expected_receipt, response) {
        (Some(limit), None, ModerationBackendResponse::Reports(reports))
            if reports.len() <= usize::from(limit)
                && reports
                    .iter()
                    .all(|report| report.fields().community_id == community_id) =>
        {
            Ok(ModerationOperatorOutcome::Reports(reports))
        }
        (None, Some(expected), ModerationBackendResponse::Written(actual))
            if actual == expected =>
        {
            Ok(ModerationOperatorOutcome::Applied(actual))
        }
        _ => Err(ModerationOperatorError::InvalidBackendResponse),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModerationRestrictionReadError {
    #[error("moderation restriction transaction is unsupported")]
    UnsupportedDatabase,
    #[error("moderation restriction transaction crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("moderation restriction history is invalid")]
    InvalidHistory,
    #[error("moderation restriction storage is unavailable")]
    Unavailable,
}

pub async fn current_restriction_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &collaboration_domain::TenantContext,
    target_principal_id: PrincipalId,
) -> Result<Option<ModerationRestriction>, ModerationRestrictionReadError> {
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement};
    if transaction.get_database_backend() != DatabaseBackend::Postgres {
        return Err(ModerationRestrictionReadError::UnsupportedDatabase);
    }
    if target_principal_id.as_uuid().is_nil() {
        return Err(ModerationRestrictionReadError::InvalidHistory);
    }
    let setting = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?
        .ok_or(ModerationRestrictionReadError::Unavailable)?;
    let configured: Option<String> = setting
        .try_get("", "community_id")
        .map_err(|_| ModerationRestrictionReadError::InvalidHistory)?;
    if configured.is_some_and(|configured| configured != tenant.community_id().to_string()) {
        return Err(ModerationRestrictionReadError::TenantBoundaryViolation);
    }
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    let rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT restriction.community_id, restriction.target_subject_principal_id AS target_principal_id,
       restriction.restriction_version::text AS restriction_version,
       restriction.is_current, restriction.ban_state, restriction.timeout_state,
       (extract(epoch FROM restriction.ban_expires_at) * 1000)::text AS ban_expires_at,
       (extract(epoch FROM restriction.timeout_expires_at) * 1000)::text AS timeout_expires_at,
       restriction.transition_kind, restriction.actor_principal_id, restriction.operation_id,
       (extract(epoch FROM restriction.occurred_at) * 1000)::text AS occurred_at,
       (action.actor_principal_id = restriction.actor_principal_id
        AND action.action_kind = restriction.transition_kind
        AND action.record_kind = 'restriction'
        AND action.record_id = restriction.target_subject_principal_id
        AND action.target_principal_id = restriction.target_subject_principal_id
        AND action.occurred_at = restriction.occurred_at
        AND audit.actor_principal_id = restriction.actor_principal_id
        AND audit.outcome = 'succeeded'
        AND audit.action = CASE
            WHEN restriction.transition_kind IN ('lift_ban', 'lift_timeout') THEN $4
            ELSE $3 END
        AND audit.occurred_at_millis = (extract(epoch FROM restriction.occurred_at) * 1000)) AS action_matches
FROM public.collaboration_moderation_restriction_versions AS restriction
LEFT JOIN public.collaboration_moderation_actions AS action
  ON action.community_id = restriction.community_id
 AND action.operation_id = restriction.operation_id
LEFT JOIN public.collaboration_audit_entries AS audit
  ON audit.community_id = restriction.community_id
 AND audit.operation_id = restriction.operation_id
WHERE restriction.community_id = $1 AND restriction.target_subject_principal_id = $2
ORDER BY restriction.restriction_version
LIMIT 1001
"#,
            [
                tenant.community_id().as_uuid().into(),
                target_principal_id.as_uuid().into(),
                ModerationAuditOperation::ApplyRestriction.action().into(),
                ModerationAuditOperation::LiftRestriction.action().into(),
            ],
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    if rows.is_empty() {
        return Ok(None);
    }
    let restriction = hydrate_restriction_rows(rows, tenant.community_id(), target_principal_id)?;
    for transition in &restriction.fields().transitions {
        use crate::audit::repository::{AuditRepository, AuditRepositoryError};
        use collaboration_domain::RestrictionTransitionKind;
        let entry = AuditRepository::load_operation_in_transaction(
            transaction,
            tenant,
            transition.source.operation_id,
        )
        .await
        .map_err(|error| match error {
            AuditRepositoryError::UnsupportedDatabase => {
                ModerationRestrictionReadError::UnsupportedDatabase
            }
            AuditRepositoryError::TenantBoundaryViolation => {
                ModerationRestrictionReadError::TenantBoundaryViolation
            }
            AuditRepositoryError::Unavailable(_) => ModerationRestrictionReadError::Unavailable,
            _ => ModerationRestrictionReadError::InvalidHistory,
        })?
        .ok_or(ModerationRestrictionReadError::InvalidHistory)?;
        let expected_action = match transition.kind {
            RestrictionTransitionKind::ApplyBan { .. }
            | RestrictionTransitionKind::ApplyTimeout { .. } => {
                ModerationAuditOperation::ApplyRestriction.action()
            }
            RestrictionTransitionKind::LiftBan | RestrictionTransitionKind::LiftTimeout => {
                ModerationAuditOperation::LiftRestriction.action()
            }
        };
        let record = entry.record();
        if record.operation_id() != transition.source.operation_id
            || record.action().as_str() != expected_action
            || record.actor_principal_id() != Some(transition.actor_principal_id)
            || record.outcome() != AuditOutcome::Succeeded
            || record.occurred_at_millis() != transition.source.occurred_at_millis
        {
            return Err(ModerationRestrictionReadError::InvalidHistory);
        }
    }
    Ok(Some(restriction))
}

#[derive(Clone)]
pub(crate) struct AdmittedModerationTargetObservation {
    authorization: collaboration_domain::moderation_policy::ModerationControlPlaneAuthorization,
    signed_event: nostr_compat::SignedEvent,
    target_public_key: NostrPublicKey,
}

impl AdmittedModerationTargetObservation {
    pub(crate) const fn authorization(
        &self,
    ) -> collaboration_domain::moderation_policy::ModerationControlPlaneAuthorization {
        self.authorization
    }
    pub(crate) const fn signed_event(&self) -> &nostr_compat::SignedEvent {
        &self.signed_event
    }
    pub(crate) const fn target_public_key(&self) -> NostrPublicKey {
        self.target_public_key
    }
}

pub(crate) async fn lock_moderation_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &collaboration_domain::TenantContext,
) -> Result<(), ModerationRestrictionReadError> {
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement};
    if transaction.get_database_backend() != DatabaseBackend::Postgres {
        return Err(ModerationRestrictionReadError::UnsupportedDatabase);
    }
    let setting = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?
        .ok_or(ModerationRestrictionReadError::Unavailable)?;
    let configured: Option<String> = setting
        .try_get("", "community_id")
        .map_err(|_| ModerationRestrictionReadError::InvalidHistory)?;
    if configured.is_some_and(|value| value != tenant.community_id().to_string()) {
        return Err(ModerationRestrictionReadError::TenantBoundaryViolation);
    }
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            [format!("collaboration-moderation:{}", tenant.community_id()).into()],
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    Ok(())
}

pub struct NostrRestrictionCommandSink {
    connection: sea_orm::DatabaseConnection,
    profiles: crate::identity::profile_repository::IdentityProfileRepository,
}

impl NostrRestrictionCommandSink {
    pub fn new(
        connection: sea_orm::DatabaseConnection,
    ) -> Result<Self, crate::identity::profile_repository::IdentityProfileRepositoryError> {
        if !matches!(
            &connection,
            sea_orm::DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(crate::identity::profile_repository::IdentityProfileRepositoryError::UnsupportedDatabase);
        }
        let profiles = crate::identity::profile_repository::IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(connection.get_postgres_connection_pool().clone()),
        )?;
        Ok(Self {
            connection,
            profiles,
        })
    }

    async fn submit_in_transaction(
        &self,
        transaction: &sea_orm::DatabaseTransaction,
        command: &crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
        now_millis: u64,
    ) -> Result<
        crate::collaboration_command::DomainCommandReceipt,
        crate::collaboration_command::DomainCommandSubmissionError,
    > {
        use crate::{
            audit::repository::{AuditRepository, ExpectedAuditHead},
            collaboration_command::{DomainCommandReceipt, DomainCommandSubmissionError},
            messages::channel_admission::current_subject_memberships_in_transaction,
        };
        use collaboration_domain::{
            AuthorizationRequest, AuthorizationResource, ModerationCommandOutcome,
            ModerationResourceContext, ModerationSnapshot, RestrictionTransitionKind,
        };
        use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement};
        let rejected = || DomainCommandSubmissionError::Rejected;
        let unavailable = || DomainCommandSubmissionError::Unavailable;
        let tenant = command.tenant();
        let signed = command.payload().signed_event();
        let kind = u32::from(signed.event.kind);
        if !matches!(
            kind,
            nostr_compat::generated_kinds::KIND_MODERATION_BAN
                | nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
                | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
        ) || command.contract_version()
            != crate::collaboration_command::CURRENT_COMMAND_CONTRACT_VERSION
            || command.principal().community_id() != tenant.community_id()
            || command.expected_version().is_some()
            || command.predecessor().is_some()
            || command.operation_id()
                != crate::nostr::event_ingest::operation_id(
                    tenant.community_id().as_uuid(),
                    signed.claimed_id,
                )
            || !crate::nostr::event_ingest::principal_matches_author(
                command.principal().kind(),
                signed.event.public_key,
            )
        {
            return Err(rejected());
        }
        nostr_compat::verify_signed_event(
            signed,
            nostr_compat::TimestampPolicy::Bounded {
                now: now_millis / 1000,
                max_past_seconds: 120,
                max_future_seconds: 120,
            },
        )
        .map_err(|_| rejected())?;
        let target_key = signed_restriction_target(&signed.event).ok_or_else(rejected)?;
        let public_reason = signed
            .event
            .tags
            .iter()
            .find(|tag| tag.first().is_some_and(|value| value == "reason"))
            .and_then(|tag| tag.get(1))
            .cloned();
        let expires_at_millis = if matches!(
            kind,
            nostr_compat::generated_kinds::KIND_MODERATION_BAN
                | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
        ) {
            signed
                .event
                .tags
                .iter()
                .find(|tag| tag.first().is_some_and(|value| value == "expiration"))
                .and_then(|tag| tag.get(1))
                .map(|value| {
                    value
                        .parse::<u64>()
                        .map_err(|_| rejected())?
                        .checked_mul(1000)
                        .ok_or_else(rejected)
                })
                .transpose()?
        } else {
            None
        };
        if matches!(
            kind,
            nostr_compat::generated_kinds::KIND_MODERATION_BAN
                | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
        ) && expires_at_millis.is_some_and(|expiry| expiry <= now_millis)
            || kind == nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
                && expires_at_millis.is_none()
        {
            return Err(rejected());
        }
        lock_moderation_in_transaction(transaction, tenant)
            .await
            .map_err(|_| unavailable())?;
        let serving =
            crate::messages::channel_admission::current_community_serving_state_in_transaction(
                transaction,
                tenant,
            )
            .await
            .map_err(|_| unavailable())?
            .ok_or_else(rejected)?;
        if serving.lifecycle != collaboration_domain::CommunityLifecycleState::Active {
            return Err(rejected());
        }
        let actor = collaboration_domain::authorization::authorization_subject(command.principal());
        transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT principal_id FROM public.collaboration_community_memberships WHERE community_id = $1 AND principal_id = $2 FOR SHARE",
            [tenant.community_id().as_uuid().into(), actor.as_uuid().into()])).await.map_err(|_| unavailable())?;
        let memberships =
            current_subject_memberships_in_transaction(transaction, tenant, actor, None)
                .await
                .map_err(|_| unavailable())?;
        let membership = memberships.community.ok_or_else(rejected)?;
        let actor_restriction = current_restriction_in_transaction(transaction, tenant, actor)
            .await
            .map_err(|_| unavailable())?;
        let (principal_archive, community_archive) =
            current_actor_archives_in_transaction(transaction, tenant, actor)
                .await
                .map_err(|_| unavailable())?;
        let scope = AuthorizationScope::new("messages:write").map_err(|_| rejected())?;
        let authorization = AuthorizationRequest {
            tenant,
            principal: command.principal(),
            required_scope: &scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id: tenant.community_id(),
                kind: AuthorizationResourceKind::Administration,
                resource_id: AggregateId::from_uuid(tenant.community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: membership.version,
            community_membership: Some(membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis,
        };
        let source = ModerationCommandSource {
            operation_id: command.operation_id(),
            occurred_at_millis: now_millis,
        };
        let control = collaboration_domain::moderation_policy::admit_moderation_control_plane(
            &ModerationAuthorizationRequest {
                authorization: &authorization,
                restriction: actor_restriction
                    .as_ref()
                    .map_or(ModerationSnapshot::Absent, ModerationSnapshot::Current),
                principal_archive,
                community_archive,
                resource_context: ModerationResourceContext::Current,
            },
            source,
        )
        .map_err(|_| rejected())?;
        let admitted = AdmittedModerationTargetObservation {
            authorization: control,
            signed_event: signed.clone(),
            target_public_key: target_key,
        };
        let prepared = self
            .profiles
            .prepare_admitted_moderation_target_in_transaction(transaction, tenant, &admitted)
            .await
            .map_err(|_| unavailable())?;
        let target_id = prepared.principal_id();
        transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT principal_id FROM public.collaboration_community_memberships WHERE community_id = $1 AND principal_id = $2 FOR SHARE",
            [tenant.community_id().as_uuid().into(), target_id.as_uuid().into()])).await.map_err(|_| unavailable())?;
        let target_member =
            current_subject_memberships_in_transaction(transaction, tenant, target_id, None)
                .await
                .map_err(|_| unavailable())?
                .community;
        let target = collaboration_domain::moderation::ModerationTargetSnapshot::new(
            tenant.community_id(),
            target_id,
            target_member,
            target_member.map(|member| member.version),
        )
        .map_err(|_| rejected())?;
        let mut restriction = current_restriction_in_transaction(transaction, tenant, target_id)
            .await
            .map_err(|_| unavailable())?
            .unwrap_or(
                ModerationRestriction::new(tenant.community_id(), target_id)
                    .map_err(|_| rejected())?,
            );
        let audit_operation = if matches!(
            kind,
            nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
        ) {
            ModerationAuditOperation::LiftRestriction
        } else {
            ModerationAuditOperation::ApplyRestriction
        };
        if let Some(existing) = AuditRepository::load_operation_in_transaction(
            transaction,
            tenant,
            command.operation_id(),
        )
        .await
        .map_err(|_| unavailable())?
        {
            let transition = restriction
                .fields()
                .transitions
                .iter()
                .find(|transition| transition.source.operation_id == command.operation_id())
                .ok_or_else(rejected)?;
            let expected_kind = match kind {
                nostr_compat::generated_kinds::KIND_MODERATION_BAN => {
                    RestrictionTransitionKind::ApplyBan { expires_at_millis }
                }
                nostr_compat::generated_kinds::KIND_MODERATION_UNBAN => {
                    RestrictionTransitionKind::LiftBan
                }
                nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT => {
                    RestrictionTransitionKind::ApplyTimeout {
                        expires_at_millis: expires_at_millis.ok_or_else(rejected)?,
                    }
                }
                nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT => {
                    RestrictionTransitionKind::LiftTimeout
                }
                _ => return Err(rejected()),
            };
            let expected_record = SecurityAuditEvent::Moderation {
                context: AuditEventContext::new(
                    tenant,
                    command.operation_id(),
                    Some(command.principal()),
                    AuditOutcome::Succeeded,
                    None,
                    transition.source.occurred_at_millis,
                )
                .map_err(|_| rejected())?,
                operation: audit_operation,
                subject_principal_id: Some(target_id),
                record_id: target_id.as_uuid(),
            }
            .into_record()
            .map_err(|_| rejected())?;
            let original_action = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
                "SELECT source_record_id FROM public.collaboration_moderation_actions WHERE community_id = $1 AND operation_id = $2",
                [tenant.community_id().as_uuid().into(), command.operation_id().as_uuid().into()])).await.map_err(|_| unavailable())?.ok_or_else(rejected)?;
            let source_id: String = original_action
                .try_get("", "source_record_id")
                .map_err(|_| unavailable())?;
            if existing.record() != &expected_record
                || transition.actor_principal_id != actor
                || transition.kind != expected_kind
                || source_id != signed.claimed_id.to_hex()
            {
                return Err(rejected());
            }
            return Ok(DomainCommandReceipt::duplicate(
                command.operation_id(),
                transition.resulting_version,
            ));
        }
        let expected_version = restriction.fields().version;
        let outcome = match kind {
            nostr_compat::generated_kinds::KIND_MODERATION_BAN => restriction.apply_ban_to_subject(
                expected_version,
                expires_at_millis,
                source,
                target,
                &authorization,
            ),
            nostr_compat::generated_kinds::KIND_MODERATION_UNBAN => {
                restriction.lift_ban_to_subject(expected_version, source, target, &authorization)
            }
            nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT => restriction
                .apply_timeout_to_subject(
                    expected_version,
                    expires_at_millis.ok_or_else(rejected)?,
                    source,
                    target,
                    &authorization,
                ),
            nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT => restriction
                .lift_timeout_to_subject(expected_version, source, target, &authorization),
            _ => return Err(rejected()),
        }
        .map_err(|_| rejected())?;
        if outcome != ModerationCommandOutcome::Applied {
            return Err(rejected());
        }
        let audit_record = SecurityAuditEvent::Moderation {
            context: AuditEventContext::new(
                tenant,
                command.operation_id(),
                Some(command.principal()),
                AuditOutcome::Succeeded,
                None,
                now_millis,
            )
            .map_err(|_| rejected())?,
            operation: audit_operation,
            subject_principal_id: Some(target_id),
            record_id: target_id.as_uuid(),
        }
        .into_record()
        .map_err(|_| rejected())?;
        let head = AuditRepository::load_head_in_transaction(transaction, tenant)
            .await
            .map_err(|_| unavailable())?;
        let expected_head = head.map_or(ExpectedAuditHead::Empty, ExpectedAuditHead::Entry);
        AuditRepository::append_in_transaction(transaction, tenant, expected_head, audit_record)
            .await
            .map_err(|_| unavailable())?;
        let action_kind = match kind {
            nostr_compat::generated_kinds::KIND_MODERATION_BAN => "apply_ban",
            nostr_compat::generated_kinds::KIND_MODERATION_UNBAN => "lift_ban",
            nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT => "apply_timeout",
            nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT => "lift_timeout",
            _ => return Err(rejected()),
        };
        transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
INSERT INTO public.collaboration_moderation_actions (community_id, action_id, operation_id, actor_principal_id,
 action_kind, record_kind, record_id, target_principal_id, public_reason, occurred_at, source_system,
 source_record_id, source_observed_at, integrity_algorithm, integrity_value)
VALUES ($1,$2,$3,$4,$5,'restriction',$6,$6,$7,to_timestamp(CAST($8 AS numeric) / 1000),'nostr',$9,
 to_timestamp(CAST($8 AS numeric) / 1000),'nostr_event_id',$9)
"#, [tenant.community_id().as_uuid().into(), uuid::Uuid::new_v4().into(), command.operation_id().as_uuid().into(), actor.as_uuid().into(),
            action_kind.into(), target_id.as_uuid().into(), public_reason.into(), now_millis.to_string().into(), signed.claimed_id.to_hex().into()])).await.map_err(|_| unavailable())?;
        self.profiles
            .commit_prepared_moderation_subject_in_transaction(transaction, tenant, prepared)
            .await
            .map_err(|_| unavailable())?;
        write_restriction_in_transaction(
            transaction,
            tenant,
            &restriction,
            target_member.is_some(),
            signed.claimed_id,
        )
        .await
        .map_err(|_| unavailable())?;
        Ok(DomainCommandReceipt::new(
            command.operation_id(),
            restriction.fields().version,
        ))
    }
}

pub(crate) fn signed_restriction_target(
    event: &nostr_compat::CanonicalEvent,
) -> Option<NostrPublicKey> {
    event.tags.iter().find_map(|tag| {
        if tag.first().is_some_and(|name| name == "p") {
            tag.get(1)
                .filter(|value| {
                    value.len() == 64
                        && value.chars().all(|character| character.is_ascii_hexdigit())
                })
                .and_then(|value| nostr_compat::PublicKey::from_hex(value).ok())
                .map(|key| NostrPublicKey::from_bytes(*key.as_bytes()))
        } else {
            None
        }
    })
}

#[async_trait]
impl crate::collaboration_command::DomainCommandSink<crate::nostr::event_ingest::NostrEventCommand>
    for NostrRestrictionCommandSink
{
    async fn submit(
        &self,
        command: crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> Result<
        crate::collaboration_command::DomainCommandReceipt,
        crate::collaboration_command::DomainCommandSubmissionError,
    > {
        use crate::collaboration_command::DomainCommandSubmissionError;
        use sea_orm::{IsolationLevel, TransactionTrait as _};
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let now_millis = u64::try_from(now.as_millis())
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let transaction = self
            .connection
            .begin_with_config(Some(IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let result = async {
            use crate::db::collaboration::{
                event_repository::{
                    EventRepository, EventStoreOutcome, EventVerificationState, VerifiedEventRecord,
                },
                persistence_policy::{EventPersistencePolicy, PrivacyAdmission},
            };
            let receipt = self
                .submit_in_transaction(&transaction, &command, now_millis)
                .await?;
            let record = VerifiedEventRecord::new(
                command.tenant().community_id(),
                command.payload().signed_event().clone(),
                EventVerificationState::Live,
                now_millis,
                nostr_compat::TimestampPolicy::Bounded {
                    now: now_millis / 1000,
                    max_past_seconds: 120,
                    max_future_seconds: 120,
                },
            )
            .map_err(|_| DomainCommandSubmissionError::Rejected)?;
            let persistence = EventPersistencePolicy::evaluate(
                record.signed_event().event.kind,
                PrivacyAdmission::community(),
            )
            .map_err(|_| DomainCommandSubmissionError::Rejected)?;
            let events = EventRepository::new(sea_orm::DatabaseConnection::from(
                self.connection.get_postgres_connection_pool().clone(),
            ))
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
            // The original signed event and moderation transition share one commit,
            // including exact retries that must not append another action or audit row.
            match events
                .store_in_transaction(&transaction, command.tenant(), &record, persistence)
                .await
                .map_err(|_| DomainCommandSubmissionError::Unavailable)?
            {
                EventStoreOutcome::Inserted | EventStoreOutcome::Duplicate => Ok(receipt),
                EventStoreOutcome::Stale | EventStoreOutcome::EphemeralNotPersisted => {
                    Err(DomainCommandSubmissionError::Rejected)
                }
            }
        }
        .await;
        match result {
            Ok(receipt) => {
                transaction
                    .commit()
                    .await
                    .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
                Ok(receipt)
            }
            Err(error) => {
                transaction
                    .rollback()
                    .await
                    .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
                Err(error)
            }
        }
    }
}

pub(crate) async fn current_actor_archives_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &collaboration_domain::TenantContext,
    actor: PrincipalId,
) -> Result<
    (
        collaboration_domain::ModerationSnapshot<collaboration_domain::IdentityArchiveSnapshot>,
        collaboration_domain::ModerationSnapshot<collaboration_domain::CommunityArchiveSnapshot>,
    ),
    ModerationRestrictionReadError,
> {
    use collaboration_domain::{
        CommunityArchivePolicyState, CommunityArchiveSnapshot, IdentityArchivePolicyState,
        IdentityArchiveSnapshot, ModerationSnapshot,
    };
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement};
    let invalid = || ModerationRestrictionReadError::InvalidHistory;
    let identity = transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
SELECT archive_version::text AS archive_version, archive_state, actor_principal_id, operation_id,
 (extract(epoch FROM occurred_at) * 1000)::text AS occurred_at
FROM public.collaboration_identity_archive_versions WHERE community_id = $1 AND target_principal_id = $2 AND is_current LIMIT 2
"#, [tenant.community_id().as_uuid().into(), actor.as_uuid().into()])).await.map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    if identity.len() > 1 {
        return Err(invalid());
    }
    let identity = match identity.first() {
        None => ModerationSnapshot::Absent,
        Some(row) => {
            validate_archive_audit_in_transaction(
                transaction,
                tenant,
                row,
                ModerationAuditOperation::ArchiveIdentity,
                actor.as_uuid(),
                Some(actor),
            )
            .await?;
            let version = AggregateVersion::new(
                row.try_get::<String>("", "archive_version")
                    .map_err(|_| invalid())?
                    .parse()
                    .map_err(|_| invalid())?,
            )
            .ok_or_else(invalid)?;
            let state = match row
                .try_get::<String>("", "archive_state")
                .map_err(|_| invalid())?
                .as_str()
            {
                "archived" => IdentityArchivePolicyState::Archived,
                "visible" => IdentityArchivePolicyState::Visible,
                _ => return Err(invalid()),
            };
            ModerationSnapshot::Current(IdentityArchiveSnapshot {
                community_id: tenant.community_id(),
                principal_id: actor,
                state,
                version,
            })
        }
    };
    let community = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT archive_version::text AS archive_version, archive_state, actor_principal_id, operation_id,
 (extract(epoch FROM occurred_at) * 1000)::text AS occurred_at
FROM public.collaboration_community_archive_versions WHERE community_id = $1 AND is_current LIMIT 2
"#,
            [tenant.community_id().as_uuid().into()],
        ))
        .await
        .map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    if community.len() > 1 {
        return Err(invalid());
    }
    let community = match community.first() {
        None => ModerationSnapshot::Absent,
        Some(row) => {
            validate_archive_audit_in_transaction(
                transaction,
                tenant,
                row,
                ModerationAuditOperation::ArchiveCommunity,
                tenant.community_id().as_uuid(),
                None,
            )
            .await?;
            let version = AggregateVersion::new(
                row.try_get::<String>("", "archive_version")
                    .map_err(|_| invalid())?
                    .parse()
                    .map_err(|_| invalid())?,
            )
            .ok_or_else(invalid)?;
            let state = match row
                .try_get::<String>("", "archive_state")
                .map_err(|_| invalid())?
                .as_str()
            {
                "archived" => CommunityArchivePolicyState::Archived,
                "active" => CommunityArchivePolicyState::Active,
                _ => return Err(invalid()),
            };
            ModerationSnapshot::Current(CommunityArchiveSnapshot {
                community_id: tenant.community_id(),
                state,
                version,
            })
        }
    };
    Ok((identity, community))
}

async fn validate_archive_audit_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &collaboration_domain::TenantContext,
    row: &sea_orm::QueryResult,
    operation: ModerationAuditOperation,
    record_id: uuid::Uuid,
    subject_id: Option<PrincipalId>,
) -> Result<(), ModerationRestrictionReadError> {
    let invalid = || ModerationRestrictionReadError::InvalidHistory;
    let operation_id =
        OperationId::from_uuid(row.try_get("", "operation_id").map_err(|_| invalid())?);
    let actor = PrincipalId::from_uuid(
        row.try_get("", "actor_principal_id")
            .map_err(|_| invalid())?,
    );
    let occurred_at = exact_milliseconds(
        &row.try_get::<String>("", "occurred_at")
            .map_err(|_| invalid())?,
    )?;
    let entry = crate::audit::repository::AuditRepository::load_operation_in_transaction(
        transaction,
        tenant,
        operation_id,
    )
    .await
    .map_err(|error| match error {
        crate::audit::repository::AuditRepositoryError::Unavailable(_) => {
            ModerationRestrictionReadError::Unavailable
        }
        _ => invalid(),
    })?
    .ok_or_else(invalid)?;
    let record = entry.record();
    if record.action().as_str() != operation.action()
        || record.actor_principal_id() != Some(actor)
        || record.occurred_at_millis() != occurred_at
        || record.outcome() != AuditOutcome::Succeeded
    {
        return Err(invalid());
    }
    for (name, id) in [
        ("moderation_record_id", Some(record_id)),
        ("subject_principal_id", subject_id.map(|id| id.as_uuid())),
    ] {
        let field = record
            .fields()
            .as_slice()
            .iter()
            .find(|field| field.name().as_str() == name);
        match (field, id) {
            (Some(field), Some(id)) if matches!(field.value(), collaboration_domain::AuditValue::Identifier(value) if value.as_str() == id.to_string()) =>
                {}
            (None, None) => {}
            _ => return Err(invalid()),
        }
    }
    Ok(())
}

async fn write_restriction_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &collaboration_domain::TenantContext,
    restriction: &ModerationRestriction,
    member_target: bool,
    event_id: nostr_compat::EventId,
) -> Result<(), ModerationRestrictionReadError> {
    use collaboration_domain::{BanState, RestrictionTransitionKind, TimeoutState};
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement};
    let fields = restriction.fields();
    if fields.community_id != tenant.community_id() {
        return Err(ModerationRestrictionReadError::TenantBoundaryViolation);
    }
    let transition = fields
        .transitions
        .last()
        .ok_or(ModerationRestrictionReadError::InvalidHistory)?;
    let (ban_state, ban_expires) = match fields.ban {
        BanState::None => ("none", None),
        BanState::Active {
            expires_at_millis, ..
        } => ("active", expires_at_millis),
    };
    let (timeout_state, timeout_expires) = match fields.timeout {
        TimeoutState::None => ("none", None),
        TimeoutState::Active {
            expires_at_millis, ..
        } => ("active", Some(expires_at_millis)),
    };
    let kind = match transition.kind {
        RestrictionTransitionKind::ApplyBan { .. } => "apply_ban",
        RestrictionTransitionKind::LiftBan => "lift_ban",
        RestrictionTransitionKind::ApplyTimeout { .. } => "apply_timeout",
        RestrictionTransitionKind::LiftTimeout => "lift_timeout",
    };
    let result = transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "UPDATE public.collaboration_moderation_restriction_versions SET is_current = false WHERE community_id = $1 AND target_subject_principal_id = $2 AND is_current",
        [tenant.community_id().as_uuid().into(), fields.target_principal_id.as_uuid().into()])).await.map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    if result.rows_affected() != u64::from(fields.transitions.len() > 1) {
        return Err(ModerationRestrictionReadError::InvalidHistory);
    }
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
INSERT INTO public.collaboration_moderation_restriction_versions (
 community_id, target_principal_id, target_identity_principal_id, restriction_version, is_current,
 ban_state, ban_expires_at, timeout_state, timeout_expires_at, transition_kind, actor_principal_id,
 operation_id, occurred_at, source_system, source_record_id, source_observed_at, integrity_algorithm, integrity_value)
VALUES ($1,$2,$3,CAST($4 AS numeric),true,$5,to_timestamp(CAST($6 AS numeric) / 1000),$7,
 to_timestamp(CAST($8 AS numeric) / 1000),$9,$10,$11,to_timestamp(CAST($12 AS numeric) / 1000),'nostr',$13,
 to_timestamp(CAST($12 AS numeric) / 1000),'nostr_event_id',$13)
"#, [tenant.community_id().as_uuid().into(), member_target.then_some(fields.target_principal_id.as_uuid()).into(),
        (!member_target).then_some(fields.target_principal_id.as_uuid()).into(), fields.version.get().to_string().into(),
        ban_state.into(), ban_expires.map(|millis| millis.to_string()).into(), timeout_state.into(), timeout_expires.map(|millis| millis.to_string()).into(),
        kind.into(), transition.actor_principal_id.as_uuid().into(), transition.source.operation_id.as_uuid().into(),
        transition.source.occurred_at_millis.to_string().into(), event_id.to_hex().into()])).await.map_err(|_| ModerationRestrictionReadError::Unavailable)?;
    Ok(())
}

fn hydrate_restriction_rows(
    rows: Vec<sea_orm::QueryResult>,
    community_id: CommunityId,
    target_principal_id: PrincipalId,
) -> Result<ModerationRestriction, ModerationRestrictionReadError> {
    use collaboration_domain::{
        BanState, RestrictionTransition, RestrictionTransitionKind, TimeoutState,
    };
    let invalid = || ModerationRestrictionReadError::InvalidHistory;
    let mut transitions = Vec::with_capacity(rows.len());
    let mut current_count = 0;
    for row in &rows {
        macro_rules! field {
            ($name:literal, $type:ty) => {
                row.try_get::<$type>("", $name).map_err(|_| invalid())?
            };
        }
        if CommunityId::from_uuid(field!("community_id", uuid::Uuid)) != community_id
            || PrincipalId::from_uuid(field!("target_principal_id", uuid::Uuid))
                != target_principal_id
            || field!("action_matches", Option<bool>) != Some(true)
        {
            return Err(invalid());
        }
        let version = AggregateVersion::new(
            field!("restriction_version", String)
                .parse()
                .map_err(|_| invalid())?,
        )
        .ok_or_else(invalid)?;
        let expiry = |name| -> Result<Option<u64>, ModerationRestrictionReadError> {
            row.try_get::<Option<String>>("", name)
                .map_err(|_| invalid())?
                .map(|value| exact_milliseconds(&value))
                .transpose()
        };
        let ban_expiry = expiry("ban_expires_at")?;
        let timeout_expiry = expiry("timeout_expires_at")?;
        let kind = match field!("transition_kind", String).as_str() {
            "apply_ban" => RestrictionTransitionKind::ApplyBan {
                expires_at_millis: ban_expiry,
            },
            "lift_ban" => RestrictionTransitionKind::LiftBan,
            "apply_timeout" => RestrictionTransitionKind::ApplyTimeout {
                expires_at_millis: timeout_expiry.ok_or_else(invalid)?,
            },
            "lift_timeout" => RestrictionTransitionKind::LiftTimeout,
            _ => return Err(invalid()),
        };
        transitions.push(RestrictionTransition {
            kind,
            actor_principal_id: PrincipalId::from_uuid(field!("actor_principal_id", uuid::Uuid)),
            source: ModerationCommandSource {
                operation_id: OperationId::from_uuid(field!("operation_id", uuid::Uuid)),
                occurred_at_millis: exact_milliseconds(&field!("occurred_at", String))?,
            },
            resulting_version: version,
        });
    }
    let mut snapshots = rows.iter().enumerate();
    let restriction = ModerationRestriction::from_history_with_observer(
        community_id,
        target_principal_id,
        &transitions,
        |fields, _| {
            let invalid = || collaboration_domain::ModerationError::InvalidRecord;
            let (index, row) = snapshots.next().ok_or_else(invalid)?;
            let expiry = |name| -> Result<Option<u64>, collaboration_domain::ModerationError> {
                row.try_get::<Option<String>>("", name)
                    .map_err(|_| invalid())?
                    .map(|value| exact_milliseconds(&value).map_err(|_| invalid()))
                    .transpose()
            };
            let ban_expiry = expiry("ban_expires_at")?;
            let timeout_expiry = expiry("timeout_expires_at")?;
            let ban_matches = match (
                row.try_get::<String>("", "ban_state")
                    .map_err(|_| invalid())?
                    .as_str(),
                fields.ban,
            ) {
                ("none", BanState::None) => ban_expiry.is_none(),
                (
                    "active",
                    BanState::Active {
                        expires_at_millis, ..
                    },
                ) => expires_at_millis == ban_expiry,
                _ => false,
            };
            let timeout_matches = match (
                row.try_get::<String>("", "timeout_state")
                    .map_err(|_| invalid())?
                    .as_str(),
                fields.timeout,
            ) {
                ("none", TimeoutState::None) => timeout_expiry.is_none(),
                (
                    "active",
                    TimeoutState::Active {
                        expires_at_millis, ..
                    },
                ) => Some(expires_at_millis) == timeout_expiry,
                _ => false,
            };
            if !ban_matches || !timeout_matches {
                return Err(invalid());
            }
            if row
                .try_get::<bool>("", "is_current")
                .map_err(|_| invalid())?
            {
                current_count += 1;
                if index + 1 != rows.len() {
                    return Err(invalid());
                }
            }
            Ok(())
        },
    )
    .map_err(|_| ModerationRestrictionReadError::InvalidHistory)?;
    if current_count != 1 {
        return Err(invalid());
    }
    Ok(restriction)
}

pub(crate) fn exact_milliseconds(value: &str) -> Result<u64, ModerationRestrictionReadError> {
    let (integer, fraction) = value.split_once('.').unwrap_or((value, ""));
    if !fraction.bytes().all(|byte| byte == b'0') {
        return Err(ModerationRestrictionReadError::InvalidHistory);
    }
    integer
        .parse()
        .map_err(|_| ModerationRestrictionReadError::InvalidHistory)
}

pub(crate) async fn admit_signed_restriction_transport(
    connection: &sea_orm::DatabaseConnection,
    profiles: &crate::identity::profile_repository::IdentityProfileRepository,
    tenant: &collaboration_domain::TenantContext,
    principal: &collaboration_domain::AuthenticatedPrincipal,
    now_seconds: u64,
) -> Result<
    crate::tenant_admission::AuthorizedRpcRequest,
    crate::nostr::auth::NostrPrincipalResolutionError,
> {
    use crate::nostr::auth::NostrPrincipalResolutionError;
    use collaboration_domain::{
        AggregateId, AuthorizationAction, AuthorizationRequest, AuthorizationResource,
        AuthorizationResourceKind, AuthorizationScope,
    };
    use sea_orm::{IsolationLevel, TransactionTrait as _};
    let transaction = connection
        .begin_with_config(Some(IsolationLevel::ReadCommitted), None)
        .await
        .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
    let result = async {
        lock_moderation_in_transaction(&transaction, tenant)
            .await
            .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
        let membership = profiles
            .current_closed_nip42_membership_in_transaction(
                &transaction,
                tenant,
                principal,
                now_seconds,
            )
            .await?;
        let scope = AuthorizationScope::new("messages:write")
            .map_err(|_| NostrPrincipalResolutionError::Denied)?;
        crate::tenant_admission::AuthorizedRpcRequest::authorize(&AuthorizationRequest {
            tenant,
            principal,
            required_scope: &scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: tenant.community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(tenant.community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: membership.version,
            community_membership: Some(membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: now_seconds
                .checked_mul(1000)
                .ok_or(NostrPrincipalResolutionError::Unavailable)?,
        })
        .map_err(|_| NostrPrincipalResolutionError::Denied)
    }
    .await;
    match result {
        Ok(admission) => {
            transaction
                .commit()
                .await
                .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
            Ok(admission)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
            Err(error)
        }
    }
}
