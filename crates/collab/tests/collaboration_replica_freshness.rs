use std::time::{Duration, Instant};

use collab::{
    freshness::{
        FreshnessIssue, FreshnessLimits, HeartbeatObservation, ReplicaFreshnessState,
        ReplicaFreshnessTracker, ReplicaObservation,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use uuid::Uuid;

fn tenant_context(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "replica-freshness")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn observation(
    epoch: Uuid,
    token: u64,
    observed_at: Instant,
    authoritative_cursor: u64,
    projection_cursor: u64,
    pubsub_available: bool,
) -> ReplicaObservation {
    ReplicaObservation {
        heartbeat: Some(HeartbeatObservation {
            epoch,
            token,
            observed_at,
        }),
        authoritative_cursor,
        projection_cursor,
        pubsub_available,
    }
}

#[test]
fn replica_freshness_distinguishes_healthy_lagging_disconnected_and_recovering() {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    let tenant = tenant_context(community_id);
    let limits = FreshnessLimits::new(Duration::from_secs(30), 5).expect("limits");
    let mut tracker = ReplicaFreshnessTracker::new(community_id, limits);
    let start = Instant::now();
    let epoch = Uuid::from_u128(7);

    let healthy = tracker
        .observe(&tenant, observation(epoch, 1, start, 10, 10, true), start)
        .expect("healthy observation");
    assert_eq!(healthy.state, ReplicaFreshnessState::Healthy);
    assert_eq!(healthy.last_trustworthy_cursor, Some(10));

    let lagging = tracker
        .observe(
            &tenant,
            observation(epoch, 2, start + Duration::from_secs(1), 25, 15, true),
            start + Duration::from_secs(1),
        )
        .expect("lagging observation");
    assert_eq!(lagging.state, ReplicaFreshnessState::Lagging);
    assert_eq!(lagging.issue, Some(FreshnessIssue::ProjectionLag));
    assert_eq!(lagging.projection_lag, 10);
    assert_eq!(lagging.last_trustworthy_cursor, Some(10));

    let disconnected = tracker
        .observe(
            &tenant,
            observation(epoch, 3, start + Duration::from_secs(2), 25, 25, false),
            start + Duration::from_secs(2),
        )
        .expect("disconnected observation");
    assert_eq!(disconnected.state, ReplicaFreshnessState::Disconnected);
    assert_eq!(disconnected.issue, Some(FreshnessIssue::PubSubUnavailable));
    assert_eq!(disconnected.last_trustworthy_cursor, Some(10));

    let recovering = tracker
        .observe(
            &tenant,
            observation(epoch, 4, start + Duration::from_secs(3), 26, 26, true),
            start + Duration::from_secs(3),
        )
        .expect("recovering observation");
    assert_eq!(recovering.state, ReplicaFreshnessState::Recovering);
    assert_eq!(recovering.last_trustworthy_cursor, Some(26));

    let recovered = tracker
        .observe(
            &tenant,
            observation(epoch, 5, start + Duration::from_secs(4), 27, 27, true),
            start + Duration::from_secs(4),
        )
        .expect("recovered observation");
    assert_eq!(recovered.state, ReplicaFreshnessState::Healthy);
    assert_eq!(recovered.last_trustworthy_cursor, Some(27));
}

#[test]
fn replica_freshness_fails_closed_on_stale_or_regressed_heartbeats() {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    let tenant = tenant_context(community_id);
    let foreign_tenant = tenant_context(CommunityId::from_uuid(Uuid::from_u128(2)));
    let limits = FreshnessLimits::new(Duration::from_secs(30), 5).expect("limits");
    let mut tracker = ReplicaFreshnessTracker::new(community_id, limits);
    let start = Instant::now();
    let epoch = Uuid::from_u128(7);

    tracker
        .observe(&tenant, observation(epoch, 10, start, 10, 10, true), start)
        .expect("initial observation");
    let stale = tracker
        .observe(
            &tenant,
            observation(epoch, 10, start + Duration::from_secs(20), 11, 11, true),
            start + Duration::from_secs(31),
        )
        .expect("stale observation");
    assert_eq!(stale.state, ReplicaFreshnessState::Disconnected);
    assert_eq!(stale.issue, Some(FreshnessIssue::StaleHeartbeat));
    assert_eq!(stale.heartbeat_age, Some(Duration::from_secs(31)));
    assert_eq!(stale.last_trustworthy_cursor, Some(10));

    let regressed = tracker
        .observe(
            &tenant,
            observation(epoch, 9, start + Duration::from_secs(32), 12, 12, true),
            start + Duration::from_secs(32),
        )
        .expect("regressed observation");
    assert_eq!(regressed.state, ReplicaFreshnessState::Disconnected);
    assert_eq!(
        regressed.issue,
        Some(FreshnessIssue::HeartbeatTokenRegressed)
    );
    assert_eq!(regressed.heartbeat_token, Some(10));

    let epoch_changed = tracker
        .observe(
            &tenant,
            observation(
                Uuid::from_u128(8),
                1,
                start + Duration::from_secs(33),
                12,
                12,
                true,
            ),
            start + Duration::from_secs(33),
        )
        .expect("new epoch observation");
    assert_eq!(epoch_changed.state, ReplicaFreshnessState::Disconnected);
    assert_eq!(
        epoch_changed.issue,
        Some(FreshnessIssue::HeartbeatEpochChanged)
    );

    let foreign = tracker.observe(
        &foreign_tenant,
        observation(
            Uuid::from_u128(8),
            2,
            start + Duration::from_secs(34),
            12,
            12,
            true,
        ),
        start + Duration::from_secs(34),
    );
    assert!(foreign.is_err());
}
