use std::collections::BTreeMap;

use collab::{
    push::outbox::{
        EncryptedPushAuthority, PushEndpointAuthorityState, PushLeaseEventReference,
        PushLeasePersistenceOutcome, PushLeasePersistenceRecord, PushOutboxError,
        PushOutboxRepository, PushWakeClaim, PushWakeEnqueueOutcome, PushWakeJobRequest,
        PushWakeTerminalOutcome,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    CommunityId, PrincipalId, PushCapabilityReference, PushEndpointGeneration, PushInstallationId,
    PushLease, PushLeaseActivation, PushLeaseAddress, PushLeaseGeneration, PushLeaseState,
    PushWakeRequest, TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(value: u128) -> TenantContext {
    let community_id = community(value);
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "collaboration-push-outbox")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn generation(value: u64) -> PushLeaseGeneration {
    PushLeaseGeneration::new(value).expect("positive generation")
}

fn endpoint_generation(value: u64) -> PushEndpointGeneration {
    PushEndpointGeneration::new(value).expect("positive endpoint generation")
}

fn capability(value: u8) -> PushCapabilityReference {
    PushCapabilityReference::from_digest([value; 32]).expect("nonzero capability")
}

fn address(community_id: CommunityId) -> PushLeaseAddress {
    PushLeaseAddress {
        community_id,
        owner_principal_id: PrincipalId::from_uuid(Uuid::from_u128(2)),
        installation_id: PushInstallationId::new("installation-one").expect("valid installation"),
    }
}

fn activation(value: u64, expires_at_millis: u64) -> PushLeaseActivation {
    PushLeaseActivation {
        generation: generation(value),
        expires_at_millis,
        capability_reference: capability(value as u8),
        endpoint_generation: endpoint_generation(value),
    }
}

fn active_record(
    lease: PushLease,
    event_value: u8,
    accepted_at_millis: u64,
) -> PushLeasePersistenceRecord {
    PushLeasePersistenceRecord::new(
        lease,
        PushLeaseEventReference::new([event_value; 32], u64::from(event_value)),
        Some(
            EncryptedPushAuthority::new(
                vec![event_value; 64],
                vec![event_value + 1; 96],
                format!("push-key-{event_value}"),
            )
            .expect("encrypted authority"),
        ),
        PushEndpointAuthorityState::Enabled,
        accepted_at_millis,
        accepted_at_millis,
    )
    .expect("active persistence record")
}

fn success() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

fn no_rows() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 0,
    }
}

fn active_lease_row(
    community_id: CommunityId,
    generation: u64,
    event_value: u8,
    expires_at_millis: i64,
) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("community_id".into(), community_id.as_uuid().into()),
        ("owner_principal_id".into(), Uuid::from_u128(2).into()),
        (
            "installation_id".into(),
            "installation-one".to_owned().into(),
        ),
        ("source_event_id".into(), vec![event_value; 32].into()),
        (
            "source_created_at_text".into(),
            u64::from(event_value).to_string().into(),
        ),
        ("generation_text".into(), generation.to_string().into()),
        ("active".into(), true.into()),
        ("expires_at_millis".into(), expires_at_millis.into()),
        (
            "last_active_expires_at_millis".into(),
            expires_at_millis.into(),
        ),
        ("revoked_at_millis".into(), Option::<i64>::None.into()),
        (
            "endpoint_generation_text".into(),
            generation.to_string().into(),
        ),
        (
            "capability_reference".into(),
            vec![generation as u8; 32].into(),
        ),
        ("capability_ciphertext".into(), vec![event_value; 64].into()),
        (
            "subscription_policy_ciphertext".into(),
            vec![event_value + 1; 96].into(),
        ),
        (
            "custody_key_id".into(),
            format!("push-key-{event_value}").into(),
        ),
        ("endpoint_enabled".into(), true.into()),
        (
            "endpoint_disabled_at_millis".into(),
            Option::<i64>::None.into(),
        ),
        ("accepted_at_millis".into(), 1_100_i64.into()),
        ("updated_at_millis".into(), 1_100_i64.into()),
    ])
}

fn revoked_lease_row(
    community_id: CommunityId,
    generation: u64,
    last_active_expires_at_millis: i64,
) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("community_id".into(), community_id.as_uuid().into()),
        ("owner_principal_id".into(), Uuid::from_u128(2).into()),
        (
            "installation_id".into(),
            "installation-one".to_owned().into(),
        ),
        ("source_event_id".into(), vec![9_u8; 32].into()),
        ("source_created_at_text".into(), "9".to_owned().into()),
        ("generation_text".into(), generation.to_string().into()),
        ("active".into(), false.into()),
        ("expires_at_millis".into(), 7_000_i64.into()),
        (
            "last_active_expires_at_millis".into(),
            last_active_expires_at_millis.into(),
        ),
        ("revoked_at_millis".into(), 2_000_i64.into()),
        (
            "endpoint_generation_text".into(),
            Option::<String>::None.into(),
        ),
        (
            "capability_reference".into(),
            Option::<Vec<u8>>::None.into(),
        ),
        (
            "capability_ciphertext".into(),
            Option::<Vec<u8>>::None.into(),
        ),
        (
            "subscription_policy_ciphertext".into(),
            Option::<Vec<u8>>::None.into(),
        ),
        ("custody_key_id".into(), Option::<String>::None.into()),
        ("endpoint_enabled".into(), false.into()),
        ("endpoint_disabled_at_millis".into(), 2_000_i64.into()),
        ("accepted_at_millis".into(), 2_000_i64.into()),
        ("updated_at_millis".into(), 2_000_i64.into()),
    ])
}

fn authorized_wake(community_id: CommunityId) -> collaboration_domain::PushWake {
    let lease = PushLease::activate(address(community_id), activation(1, 5_000), 1_000)
        .expect("active lease");
    lease
        .authorize_wake(
            PushWakeRequest {
                lease_generation: generation(1),
                endpoint_generation: endpoint_generation(1),
                capability_reference: capability(1),
            },
            1_100,
        )
        .expect("authorized wake")
}

fn wake_request(community_id: CommunityId) -> PushWakeJobRequest {
    PushWakeJobRequest::new(
        Uuid::from_u128(10),
        Uuid::from_u128(11),
        authorized_wake(community_id),
        [7_u8; 32],
        1_100,
        1_000,
    )
    .expect("wake job request")
}

fn wake_identity_row() -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("wake_id".into(), Uuid::from_u128(10).into()),
        ("request_id".into(), Uuid::from_u128(11).into()),
        ("owner_principal_id".into(), Uuid::from_u128(2).into()),
        (
            "installation_id".into(),
            "installation-one".to_owned().into(),
        ),
        ("lease_generation_text".into(), "1".to_owned().into()),
        ("endpoint_generation_text".into(), "1".to_owned().into()),
        ("capability_reference".into(), vec![1_u8; 32].into()),
        ("source_event_id".into(), vec![7_u8; 32].into()),
        ("expires_at_millis".into(), 5_000_i64.into()),
        ("available_at_millis".into(), 1_100_i64.into()),
        ("created_at_millis".into(), 1_000_i64.into()),
    ])
}

fn claimed_wake_row(
    community_id: CommunityId,
    attempt_count: i32,
    claim_id: Uuid,
    claim_expires_at_millis: i64,
) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("community_id".into(), community_id.as_uuid().into()),
        ("wake_id".into(), Uuid::from_u128(10).into()),
        ("request_id".into(), Uuid::from_u128(11).into()),
        ("owner_principal_id".into(), Uuid::from_u128(2).into()),
        (
            "installation_id".into(),
            "installation-one".to_owned().into(),
        ),
        ("lease_generation_text".into(), "1".to_owned().into()),
        ("endpoint_generation_text".into(), "1".to_owned().into()),
        ("capability_reference".into(), vec![1_u8; 32].into()),
        ("source_event_id".into(), vec![7_u8; 32].into()),
        ("expires_at_millis".into(), 5_000_i64.into()),
        ("attempt_count".into(), attempt_count.into()),
        ("claim_id".into(), claim_id.into()),
        (
            "claim_expires_at_millis".into(),
            claim_expires_at_millis.into(),
        ),
    ])
}

#[tokio::test]
async fn push_lease_repository_replaces_only_with_a_newer_generation_and_reads_encryption() {
    let tenant = tenant(1);
    let mut lease =
        PushLease::activate(address(tenant.community_id()), activation(1, 5_000), 1_000)
            .expect("first lease");
    let first = active_record(lease.clone(), 4, 1_100);
    lease
        .replace(activation(2, 6_000), 1_200)
        .expect("newer lease");
    let replacement = active_record(lease, 5, 1_300);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([
            success(),
            success(),
            success(),
            success(),
            success(),
            success(),
            no_rows(),
        ])
        .append_query_results([vec![active_lease_row(tenant.community_id(), 2, 5, 6_000)]])
        .into_connection();
    let repository = PushOutboxRepository::new(database).expect("Postgres repository");

    assert_eq!(
        repository
            .upsert_lease(&tenant, &first)
            .await
            .expect("insert"),
        PushLeasePersistenceOutcome::Applied
    );
    assert_eq!(
        repository
            .upsert_lease(&tenant, &replacement)
            .await
            .expect("replace"),
        PushLeasePersistenceOutcome::Applied
    );
    let loaded = repository
        .load_lease(&tenant, &replacement.lease().fields().address)
        .await
        .expect("load replacement")
        .expect("stored replacement");
    assert_eq!(loaded.lease().fields().generation, generation(2));
    assert_eq!(
        loaded
            .encrypted_authority()
            .expect("encrypted authority")
            .custody_key_id(),
        "push-key-5"
    );
    let diagnostics = format!("{loaded:?}");
    assert!(!diagnostics.contains("push-key-5"));
    assert!(!diagnostics.contains(&format!("{:?}", vec![5_u8; 64])));
    assert_eq!(
        repository
            .upsert_lease(&tenant, &replacement)
            .await
            .expect("stale write"),
        PushLeasePersistenceOutcome::Stale
    );

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(
        log.contains("EXCLUDED.generation > public.collaboration_push_leases.generation"),
        "{log}"
    );
    assert!(log.contains("capability_ciphertext"), "{log}");
    assert!(log.contains("set_config('app.community_id'"), "{log}");
}

#[tokio::test]
async fn push_lease_repository_persists_a_revocation_without_device_secrets() {
    let tenant = tenant(1);
    let mut lease =
        PushLease::activate(address(tenant.community_id()), activation(1, 5_000), 1_000)
            .expect("active lease");
    lease
        .revoke(generation(2), 7_000, 2_000)
        .expect("newer revocation");
    let revoked = PushLeasePersistenceRecord::new(
        lease,
        PushLeaseEventReference::new([9_u8; 32], 9),
        None,
        PushEndpointAuthorityState::Disabled {
            disabled_at_millis: 2_000,
        },
        2_000,
        2_000,
    )
    .expect("revoked persistence record");
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([success(), success(), success()])
        .append_query_results([vec![revoked_lease_row(tenant.community_id(), 2, 5_000)]])
        .into_connection();
    let repository = PushOutboxRepository::new(database).expect("Postgres repository");

    assert_eq!(
        repository
            .upsert_lease(&tenant, &revoked)
            .await
            .expect("store revocation"),
        PushLeasePersistenceOutcome::Applied
    );
    let loaded = repository
        .load_lease(&tenant, &revoked.lease().fields().address)
        .await
        .expect("load revocation")
        .expect("stored revocation");
    assert!(matches!(
        loaded.lease().fields().state,
        PushLeaseState::Revoked {
            revoked_at_millis: 2_000
        }
    ));
    assert_eq!(loaded.lease().fields().last_active_expires_at_millis, 5_000);
    assert_eq!(loaded.encrypted_authority(), None);
    assert!(matches!(
        loaded.endpoint_authority(),
        PushEndpointAuthorityState::Disabled {
            disabled_at_millis: 2_000
        }
    ));
}

#[tokio::test]
async fn push_wake_claim_recovers_after_a_crash_and_completes_under_the_new_claim() {
    let tenant = tenant(1);
    let first_claim_id = Uuid::from_u128(20);
    let recovered_claim_id = Uuid::from_u128(21);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([success(), success(), success(), success()])
        .append_query_results([
            vec![claimed_wake_row(
                tenant.community_id(),
                1,
                first_claim_id,
                1_500,
            )],
            vec![claimed_wake_row(
                tenant.community_id(),
                2,
                recovered_claim_id,
                2_000,
            )],
        ])
        .into_connection();
    let repository = PushOutboxRepository::new(database).expect("Postgres repository");

    let first = repository
        .claim_wakes(
            &tenant,
            PushWakeClaim::new(first_claim_id, 1_000, 1_500, 1).expect("first claim"),
        )
        .await
        .expect("claim wake");
    let first = first.first().expect("one first claim");
    assert_eq!(first.attempt_count(), 1);
    let recovered = repository
        .claim_wakes(
            &tenant,
            PushWakeClaim::new(recovered_claim_id, 1_500, 2_000, 1).expect("recovery claim"),
        )
        .await
        .expect("recover wake");
    let recovered = recovered.first().expect("one recovered claim");
    assert_eq!(recovered.attempt_count(), 2);
    assert_eq!(recovered.claim_id(), recovered_claim_id);
    assert_eq!(
        recovered.payload(),
        collaboration_domain::PushWakePayload::Reconnect
    );
    repository
        .complete_wake(&tenant, recovered, PushWakeTerminalOutcome::Accepted, 1_600)
        .await
        .expect("complete recovered wake");

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("job.claim_expires_at <="), "{log}");
    assert!(log.contains("FOR UPDATE OF job SKIP LOCKED"), "{log}");
    assert!(
        log.contains("lease.generation = job.lease_generation"),
        "{log}"
    );
    assert!(log.contains("AND claim_id ="), "{log}");
}

#[tokio::test]
async fn push_wake_enqueue_returns_the_existing_exact_job_for_a_duplicate() {
    let tenant = tenant(1);
    let request = wake_request(tenant.community_id());
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([success(), success(), success(), no_rows()])
        .append_query_results([vec![wake_identity_row()]])
        .into_connection();
    let repository = PushOutboxRepository::new(database).expect("Postgres repository");

    assert_eq!(
        repository
            .enqueue_wake(&tenant, &request, 1_200)
            .await
            .expect("first wake"),
        PushWakeEnqueueOutcome::Enqueued
    );
    assert_eq!(
        repository
            .enqueue_wake(&tenant, &request, 1_200)
            .await
            .expect("duplicate wake"),
        PushWakeEnqueueOutcome::Duplicate
    );

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("ON CONFLICT DO NOTHING"), "{log}");
    assert!(log.contains("lease.endpoint_enabled"), "{log}");
    assert!(log.contains("lease.capability_reference ="), "{log}");
}

#[tokio::test]
async fn push_executor_claim_operations_revalidate_retry_and_disable_exact_authority() {
    let tenant = tenant(1);
    let wake_id = Uuid::from_u128(10);
    let claim_id = Uuid::from_u128(20);
    let authorized_row: BTreeMap<String, SeaValue> =
        BTreeMap::from([("authorized".into(), true.into())]);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([success(), success(), success(), success(), success()])
        .append_query_results([vec![authorized_row]])
        .into_connection();
    let repository = PushOutboxRepository::new(database).expect("Postgres repository");

    assert!(
        repository
            .revalidate_claim(&tenant, wake_id, claim_id, 1_000)
            .await
            .expect("revalidate claim")
    );
    repository
        .retry_claim(&tenant, wake_id, claim_id, 2_000, 1_000)
        .await
        .expect("release retry");
    assert!(
        repository
            .disable_claimed_endpoint(&tenant, wake_id, claim_id, 900, 1_000)
            .await
            .expect("disable exact endpoint")
    );

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(
        log.contains("lease.generation = job.lease_generation"),
        "{log}"
    );
    assert!(log.contains("claim_expires_at >="), "{log}");
    assert!(log.contains("SET state = 'pending'"), "{log}");
    assert!(log.contains("SET endpoint_enabled = false"), "{log}");
    assert!(
        log.contains("lease.endpoint_generation = job.endpoint_generation"),
        "{log}"
    );
}

#[tokio::test]
async fn push_outbox_rejects_foreign_tenants_before_writes_and_on_defensive_reads() {
    let local_tenant = tenant(1);
    let foreign_lease = PushLease::activate(address(community(2)), activation(1, 5_000), 1_000)
        .expect("foreign lease");
    let foreign_record = active_record(foreign_lease, 4, 1_100);
    let repository =
        PushOutboxRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("Postgres repository");

    assert!(matches!(
        repository
            .upsert_lease(&local_tenant, &foreign_record)
            .await,
        Err(PushOutboxError::TenantBoundaryViolation)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );

    let claim_id = Uuid::from_u128(30);
    let repository = PushOutboxRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_exec_results([success()])
            .append_query_results([vec![claimed_wake_row(community(2), 1, claim_id, 1_500)]])
            .into_connection(),
    )
    .expect("Postgres repository");
    assert!(matches!(
        repository
            .claim_wakes(
                &local_tenant,
                PushWakeClaim::new(claim_id, 1_000, 1_500, 1).expect("claim"),
            )
            .await,
        Err(PushOutboxError::TenantBoundaryViolation)
    ));
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("ROLLBACK"), "{log}");
    assert!(log.contains("WHERE job.community_id ="), "{log}");
}
