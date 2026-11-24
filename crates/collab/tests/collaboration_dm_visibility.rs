use std::collections::BTreeMap;

use collab::messages::dm_visibility::{
    DmVisibilityAccess, DmVisibilityError, DmVisibilityMutation, DmVisibilityRepository,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn dm_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn viewer_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(3))
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id(), "dm-visibility-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn scope() -> AuthorizationScope {
    AuthorizationScope::new("collaboration:dms:visibility").expect("scope")
}

fn principal(scope: &AuthorizationScope) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        viewer_id(),
        community_id(),
        ServiceAccountId::new(4),
        PrincipalScopes::new([scope.clone()]).expect("scopes"),
    )
}

fn community_authorization<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    scope: &'a AuthorizationScope,
    include_membership: bool,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope: scope,
        action: AuthorizationAction::Read,
        resource: AuthorizationResource {
            community_id: community_id(),
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(community_id().as_uuid()),
            owner_principal_id: Some(viewer_id()),
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: include_membership.then_some(CommunityMembership {
            community_id: community_id(),
            principal_id: viewer_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: 1_900_000_000_000,
    }
}

fn channel_authorization<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    scope: &'a AuthorizationScope,
    channel_id: AggregateId,
    include_channel_membership: bool,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope: scope,
        action: AuthorizationAction::Write,
        resource: AuthorizationResource {
            community_id: community_id(),
            kind: AuthorizationResourceKind::Channel,
            resource_id: channel_id,
            owner_principal_id: None,
            channel_id: Some(channel_id),
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(CommunityMembership {
            community_id: community_id(),
            principal_id: viewer_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        current_channel_membership_version: include_channel_membership
            .then_some(AggregateVersion::FIRST),
        channel_membership: include_channel_membership.then_some(ChannelMembership {
            community_id: community_id(),
            channel_id,
            principal_id: viewer_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        delegation: None,
        now_millis: 1_900_000_000_000,
    }
}

fn mutation_row(channel_id: AggregateId) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("channel_id".to_owned(), channel_id.as_uuid().into()),
        ("principal_id".to_owned(), viewer_id().as_uuid().into()),
    ])
}

fn snapshot_row(channel_id: AggregateId) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([("channel_id".to_owned(), channel_id.as_uuid().into())])
}

fn tenant_exec_result() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

#[tokio::test]
async fn hide_persists_only_the_authenticated_viewers_dm_presentation_state() {
    let channel_id = dm_id(20);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([tenant_exec_result()])
        .append_query_results([vec![mutation_row(channel_id)]])
        .into_connection();
    let repository = DmVisibilityRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = channel_authorization(&tenant, &principal, &scope, channel_id, true);

    let result = repository
        .hide(
            DmVisibilityAccess {
                authorization: &authorization,
            },
            channel_id,
        )
        .await
        .expect("hide DM");

    assert_eq!(result, DmVisibilityMutation::Hidden);
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("SET hidden_at = clock_timestamp()"));
    assert!(log.contains("membership.principal_id = $3"));
    assert!(log.contains("membership.status = 'active'"));
    assert!(log.contains("membership.membership_version = CAST($4 AS numeric)"));
    assert!(log.contains("channel.channel_type = 'dm'"));
    assert!(!log.contains("collaboration_messages"));
}

#[tokio::test]
async fn reopen_clears_hide_state_without_deleting_messages_or_membership() {
    let channel_id = dm_id(21);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([tenant_exec_result()])
        .append_query_results([vec![mutation_row(channel_id)]])
        .into_connection();
    let repository = DmVisibilityRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = channel_authorization(&tenant, &principal, &scope, channel_id, true);

    let result = repository
        .reopen(
            DmVisibilityAccess {
                authorization: &authorization,
            },
            channel_id,
        )
        .await
        .expect("reopen DM");

    assert_eq!(result, DmVisibilityMutation::Reopened);
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("SET hidden_at = NULL"));
    assert!(log.contains("membership.hidden_at IS NOT NULL"));
    assert!(!log.contains("DELETE FROM"));
    assert!(!log.contains("collaboration_messages"));
}

#[tokio::test]
async fn removed_participant_cannot_change_or_appear_in_visibility_state() {
    let channel_id = dm_id(22);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([tenant_exec_result()])
        .append_query_results([Vec::<BTreeMap<String, SeaValue>>::new()])
        .into_connection();
    let repository = DmVisibilityRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = channel_authorization(&tenant, &principal, &scope, channel_id, true);

    let result = repository
        .hide(
            DmVisibilityAccess {
                authorization: &authorization,
            },
            channel_id,
        )
        .await;

    assert!(matches!(result, Err(DmVisibilityError::NotAvailable)));
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("membership.status = 'active'"));
    assert!(log.contains("channel.lifecycle_state = 'active'"));
    assert!(!log.contains("count("));
}

#[tokio::test]
async fn snapshot_returns_only_the_authorized_viewers_active_hidden_dm_set() {
    let first = dm_id(23);
    let second = dm_id(24);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([tenant_exec_result()])
        .append_query_results([vec![snapshot_row(first), snapshot_row(second)]])
        .into_connection();
    let repository = DmVisibilityRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = community_authorization(&tenant, &principal, &scope, true);

    let snapshot = repository
        .snapshot(DmVisibilityAccess {
            authorization: &authorization,
        })
        .await
        .expect("visibility snapshot");

    assert_eq!(snapshot.hidden_dm_ids(), &[first, second]);
    assert_eq!(snapshot.hidden_count(), 2);
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("membership.principal_id = $2"));
    assert!(log.contains("membership.hidden_at IS NOT NULL"));
    assert!(log.contains("membership.status = 'active'"));
    assert!(log.contains("channel.channel_type = 'dm'"));
    assert!(!log.contains("count("));
}

#[tokio::test]
async fn authorization_denial_happens_before_ids_or_counts_can_be_queried() {
    let channel_id = dm_id(25);
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let snapshot_authorization = community_authorization(&tenant, &principal, &scope, false);
    let snapshot_repository =
        DmVisibilityRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("snapshot repository");
    let snapshot_result = snapshot_repository
        .snapshot(DmVisibilityAccess {
            authorization: &snapshot_authorization,
        })
        .await;
    assert!(matches!(
        snapshot_result,
        Err(DmVisibilityError::Unauthorized(_))
    ));
    assert!(
        snapshot_repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );

    let mutation_authorization =
        channel_authorization(&tenant, &principal, &scope, channel_id, false);
    let mutation_repository =
        DmVisibilityRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("mutation repository");
    let mutation_result = mutation_repository
        .hide(
            DmVisibilityAccess {
                authorization: &mutation_authorization,
            },
            channel_id,
        )
        .await;
    assert!(matches!(
        mutation_result,
        Err(DmVisibilityError::Unauthorized(_))
    ));
    assert!(
        mutation_repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}
