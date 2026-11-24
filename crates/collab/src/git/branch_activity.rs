use std::sync::Arc;

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AuthenticatedPrincipalKind, AuthorizationAction, AuthorizationRequest,
    AuthorizationResourceKind, BranchCollaboration, BranchCollaborationIdentity,
    BranchLifecycleState, BranchUpdateKind, GitCommitId, OperationId, PrincipalId,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    branch_channel::{BranchChannelError, BranchChannelService},
    object_store::GitContentDigest,
    smart_http_write::GitPushReceipt,
};

const BRANCH_ACTIVITY_NAMESPACE: Uuid = Uuid::from_u128(0xb37acf55_9199_5fc9_b3dc_6641d962ee71);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum BranchActivityKind {
    Created {
        commit: GitCommitId,
    },
    Updated {
        previous_commit: GitCommitId,
        current_commit: GitCommitId,
        update_kind: BranchUpdateKind,
    },
}

impl BranchActivityKind {
    pub const fn current_commit(&self) -> &GitCommitId {
        match self {
            Self::Created { commit } => commit,
            Self::Updated { current_commit, .. } => current_commit,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BranchActivityEvent {
    event_id: AggregateId,
    operation_id: OperationId,
    actor_principal_id: PrincipalId,
    branch: BranchCollaborationIdentity,
    channel_id: AggregateId,
    branch_version: collaboration_domain::AggregateVersion,
    kind: BranchActivityKind,
    parent_manifest: Option<GitContentDigest>,
    published_manifest: GitContentDigest,
    observed_at_millis: u64,
}

impl BranchActivityEvent {
    pub const fn event_id(&self) -> AggregateId {
        self.event_id
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn actor_principal_id(&self) -> PrincipalId {
        self.actor_principal_id
    }

    pub const fn branch(&self) -> &BranchCollaborationIdentity {
        &self.branch
    }

    pub const fn channel_id(&self) -> AggregateId {
        self.channel_id
    }

    pub const fn branch_version(&self) -> collaboration_domain::AggregateVersion {
        self.branch_version
    }

    pub const fn kind(&self) -> &BranchActivityKind {
        &self.kind
    }

    pub const fn parent_manifest(&self) -> Option<&GitContentDigest> {
        self.parent_manifest.as_ref()
    }

    pub const fn published_manifest(&self) -> &GitContentDigest {
        &self.published_manifest
    }

    pub const fn observed_at_millis(&self) -> u64 {
        self.observed_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchActivityAppendOutcome {
    Inserted,
    Existing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchActivitySinkError {
    Conflict,
    Unavailable,
}

#[async_trait]
pub trait BranchActivitySink: Send + Sync {
    async fn append(
        &self,
        event: &BranchActivityEvent,
    ) -> Result<BranchActivityAppendOutcome, BranchActivitySinkError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchChannelResolutionError {
    Rejected,
    Conflict,
    Unavailable,
}

#[async_trait]
pub trait BranchChannelResolver: Send + Sync {
    async fn resolve_or_create(
        &self,
        branch: &BranchCollaboration,
        creator_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<AggregateId, BranchChannelResolutionError>;
}

#[async_trait]
impl BranchChannelResolver for BranchChannelService {
    async fn resolve_or_create(
        &self,
        branch: &BranchCollaboration,
        creator_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<AggregateId, BranchChannelResolutionError> {
        self.resolve_for_activity(branch, creator_principal_id, authorization)
            .await
            .map(|binding| binding.channel_id())
            .map_err(|error| match error {
                BranchChannelError::BindingConflict => BranchChannelResolutionError::Conflict,
                BranchChannelError::Unavailable(_) => BranchChannelResolutionError::Unavailable,
                BranchChannelError::UnsupportedBackend
                | BranchChannelError::BranchUnavailable
                | BranchChannelError::InvalidRecord
                | BranchChannelError::Domain(_) => BranchChannelResolutionError::Rejected,
            })
    }
}

pub struct BranchActivityProjector {
    channel_resolver: Arc<dyn BranchChannelResolver>,
    sink: Arc<dyn BranchActivitySink>,
}

impl BranchActivityProjector {
    pub fn new(
        channel_resolver: Arc<dyn BranchChannelResolver>,
        sink: Arc<dyn BranchActivitySink>,
    ) -> Self {
        Self {
            channel_resolver,
            sink,
        }
    }

    pub async fn project_accepted_ref(
        &self,
        receipt: &GitPushReceipt,
        previous: Option<&BranchCollaboration>,
        current: &BranchCollaboration,
        push_authorization: &AuthorizationRequest<'_>,
        channel_authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchActivityEvent, BranchActivityProjectionError> {
        let published_manifest = validate_receipt(receipt)?;
        validate_push_shape(push_authorization, current)?;
        let kind = classify_transition(previous, current)?;
        let creator_principal_id = authorization_subject(channel_authorization);
        let channel_id = self
            .channel_resolver
            .resolve_or_create(current, creator_principal_id, channel_authorization)
            .await
            .map_err(BranchActivityProjectionError::Channel)?;
        let event = BranchActivityEvent {
            event_id: activity_event_id(receipt.operation_id, &current.fields().identity, &kind),
            operation_id: receipt.operation_id,
            actor_principal_id: authorization_subject(push_authorization),
            branch: current.fields().identity.clone(),
            channel_id,
            branch_version: current.fields().version,
            kind,
            parent_manifest: receipt.parent_manifest.clone(),
            published_manifest,
            observed_at_millis: push_authorization.now_millis,
        };
        self.sink
            .append(&event)
            .await
            .map_err(BranchActivityProjectionError::Sink)?;
        Ok(event)
    }
}

fn validate_receipt(
    receipt: &GitPushReceipt,
) -> Result<GitContentDigest, BranchActivityProjectionError> {
    if receipt.operation_id.as_uuid().is_nil() {
        return Err(BranchActivityProjectionError::InvalidReceipt);
    }
    if !receipt.applied {
        return Err(BranchActivityProjectionError::PushNotApplied);
    }
    receipt
        .published_manifest
        .clone()
        .ok_or(BranchActivityProjectionError::InvalidReceipt)
}

fn validate_push_shape(
    authorization: &AuthorizationRequest<'_>,
    branch: &BranchCollaboration,
) -> Result<(), BranchActivityProjectionError> {
    let identity = &branch.fields().identity;
    if authorization.tenant.community_id() != identity.community_id()
        || authorization.resource.community_id != identity.community_id()
        || authorization.resource.kind != AuthorizationResourceKind::Repository
        || authorization.resource.resource_id != identity.repository_id()
        || authorization.resource.channel_id.is_some()
        || authorization.action != AuthorizationAction::Write
    {
        return Err(BranchActivityProjectionError::AuthorizationMismatch);
    }
    Ok(())
}

fn classify_transition(
    previous: Option<&BranchCollaboration>,
    current: &BranchCollaboration,
) -> Result<BranchActivityKind, BranchActivityProjectionError> {
    let fields = current.fields();
    if fields.lifecycle_state != BranchLifecycleState::Active || fields.merge.is_some() {
        return Err(BranchActivityProjectionError::InvalidTransition);
    }
    let Some(previous) = previous else {
        if fields.version != collaboration_domain::AggregateVersion::FIRST
            || fields.last_head_update.is_some()
        {
            return Err(BranchActivityProjectionError::InvalidTransition);
        }
        return Ok(BranchActivityKind::Created {
            commit: fields.head_commit.clone(),
        });
    };
    let previous_fields = previous.fields();
    let Some(update) = fields.last_head_update.as_ref() else {
        return Err(BranchActivityProjectionError::InvalidTransition);
    };
    if previous_fields.identity != fields.identity
        || previous_fields.lifecycle_state != BranchLifecycleState::Active
        || !fields.version.follows(previous_fields.version)
        || update.previous_commit() != &previous_fields.head_commit
        || update.current_commit() != &fields.head_commit
        || update.previous_commit() == update.current_commit()
    {
        return Err(BranchActivityProjectionError::InvalidTransition);
    }
    Ok(BranchActivityKind::Updated {
        previous_commit: update.previous_commit().clone(),
        current_commit: update.current_commit().clone(),
        update_kind: update.kind(),
    })
}

fn activity_event_id(
    operation_id: OperationId,
    branch: &BranchCollaborationIdentity,
    kind: &BranchActivityKind,
) -> AggregateId {
    let mut source = Vec::with_capacity(16 + 16 + 16 + 8 + 1 + 128);
    source.extend_from_slice(operation_id.as_uuid().as_bytes());
    source.extend_from_slice(branch.community_id().as_uuid().as_bytes());
    source.extend_from_slice(branch.repository_id().as_uuid().as_bytes());
    source.extend_from_slice(&branch.generation().get().to_be_bytes());
    source.extend_from_slice(branch.branch_ref().as_str().as_bytes());
    match kind {
        BranchActivityKind::Created { commit } => {
            source.push(0);
            source.extend_from_slice(commit.as_str().as_bytes());
        }
        BranchActivityKind::Updated {
            previous_commit,
            current_commit,
            update_kind,
        } => {
            source.push(1);
            source.extend_from_slice(previous_commit.as_str().as_bytes());
            source.push(match update_kind {
                BranchUpdateKind::FastForward => 0,
                BranchUpdateKind::Force => 1,
            });
            source.extend_from_slice(current_commit.as_str().as_bytes());
        }
    }
    AggregateId::from_uuid(Uuid::new_v5(&BRANCH_ACTIVITY_NAMESPACE, &source))
}

fn authorization_subject(authorization: &AuthorizationRequest<'_>) -> PrincipalId {
    match authorization.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => authorization.principal.principal_id(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum BranchActivityProjectionError {
    #[error("Git push was not applied")]
    PushNotApplied,
    #[error("Git push receipt is invalid")]
    InvalidReceipt,
    #[error("Git push authorization does not match the branch")]
    AuthorizationMismatch,
    #[error("branch activity transition is invalid")]
    InvalidTransition,
    #[error("branch channel resolution failed")]
    Channel(BranchChannelResolutionError),
    #[error("branch activity append failed")]
    Sink(BranchActivitySinkError),
}
