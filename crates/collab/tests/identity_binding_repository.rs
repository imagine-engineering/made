use std::collections::BTreeMap;

use collab::identity::binding_repository::{
    IdentityBindingRepository, IdentityBindingRepositoryError,
};
use collaboration_domain::{
    AccountBinding, AccountBindingFields, AggregateVersion, BindingId, BindingStatus,
    BindingVerification, BindingVerificationMethod, CommunityId, EvidenceReference, NostrPublicKey,
    OperationId, OrganizationPolicyVersion, PrincipalId, ProfileId, ServiceAccountId,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value};
use uuid::Uuid;

fn binding(
    community_id: CommunityId,
    status: BindingStatus,
    version: AggregateVersion,
) -> AccountBinding {
    AccountBinding::new(AccountBindingFields {
        binding_id: BindingId::from_uuid(Uuid::from_u128(11)),
        community_id,
        service_account_id: ServiceAccountId::new(7),
        profile_id: ProfileId::from_uuid(Uuid::from_u128(12)),
        public_key: NostrPublicKey::from_bytes([3; 32]),
        status,
        verification: Some(BindingVerification {
            method: BindingVerificationMethod::ExistingKeyChallenge,
            evidence_reference: EvidenceReference::new("evidence:repository-test")
                .expect("evidence reference"),
            verified_at_millis: 20,
        }),
        predecessor: None,
        successor: None,
        created_at_millis: 10,
        activated_at_millis: Some(30),
        terminal_at_millis: status.is_historical().then_some(40),
        organization_policy_version: OrganizationPolicyVersion::FIRST,
        actor_principal_id: PrincipalId::from_uuid(Uuid::from_u128(13)),
        version,
        audit_reference: OperationId::from_uuid(Uuid::from_u128(14)),
    })
    .expect("valid binding fixture")
}

fn binding_row(binding: &AccountBinding) -> BTreeMap<String, Value> {
    let fields = binding.fields();
    let verification = fields.verification.as_ref();
    BTreeMap::from([
        ("community_id".into(), fields.community_id.as_uuid().into()),
        ("binding_id".into(), fields.binding_id.as_uuid().into()),
        ("version".into(), (fields.version.get() as i64).into()),
        (
            "service_account_id".into(),
            (fields.service_account_id.get() as i64).into(),
        ),
        ("profile_id".into(), fields.profile_id.as_uuid().into()),
        (
            "nostr_public_key".into(),
            fields.public_key.as_bytes().to_vec().into(),
        ),
        (
            "status".into(),
            match fields.status {
                BindingStatus::Pending => "pending",
                BindingStatus::Verified => "verified",
                BindingStatus::Active => "active",
                BindingStatus::Rotated => "rotated",
                BindingStatus::Revoked => "revoked",
                BindingStatus::Archived => "archived",
            }
            .to_owned()
            .into(),
        ),
        (
            "verification_method".into(),
            verification
                .map(|value| match value.method {
                    BindingVerificationMethod::GeneratedKeyChallenge => "generated_key_challenge",
                    BindingVerificationMethod::ExistingKeyChallenge => "existing_key_challenge",
                    BindingVerificationMethod::ImportedKeyChallenge => "imported_key_challenge",
                    BindingVerificationMethod::PairedKeyChallenge => "paired_key_challenge",
                    BindingVerificationMethod::RestoredKeyChallenge => "restored_key_challenge",
                    BindingVerificationMethod::MigratedEvidence => "migrated_evidence",
                })
                .map(str::to_owned)
                .into(),
        ),
        (
            "evidence_reference".into(),
            verification
                .map(|value| value.evidence_reference.as_str().to_owned())
                .into(),
        ),
        (
            "verified_at_millis".into(),
            verification
                .map(|value| value.verified_at_millis as i64)
                .into(),
        ),
        (
            "predecessor_binding_id".into(),
            fields
                .predecessor
                .map(|value| value.binding_id.as_uuid())
                .into(),
        ),
        (
            "predecessor_version".into(),
            fields
                .predecessor
                .map(|value| value.version.get() as i64)
                .into(),
        ),
        (
            "successor_binding_id".into(),
            fields
                .successor
                .map(|value| value.binding_id.as_uuid())
                .into(),
        ),
        (
            "successor_version".into(),
            fields
                .successor
                .map(|value| value.version.get() as i64)
                .into(),
        ),
        (
            "created_at_millis".into(),
            (fields.created_at_millis as i64).into(),
        ),
        (
            "activated_at_millis".into(),
            fields.activated_at_millis.map(|value| value as i64).into(),
        ),
        (
            "terminal_at_millis".into(),
            fields.terminal_at_millis.map(|value| value as i64).into(),
        ),
        (
            "organization_policy_version".into(),
            (fields.organization_policy_version.get() as i64).into(),
        ),
        (
            "actor_principal_id".into(),
            fields.actor_principal_id.as_uuid().into(),
        ),
        (
            "audit_reference".into(),
            fields.audit_reference.as_uuid().into(),
        ),
    ])
}

fn repository_with_current(
    current: &AccountBinding,
    execution_results: usize,
) -> IdentityBindingRepository {
    let connection = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![binding_row(current)]])
        .append_exec_results((0..execution_results).map(|_| MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }))
        .into_connection();
    IdentityBindingRepository::new(connection).expect("Postgres repository")
}

#[tokio::test]
async fn identity_binding_repository_rejects_cross_tenant_rows() {
    let stored_community = CommunityId::from_uuid(Uuid::from_u128(1));
    let requested_community = CommunityId::from_uuid(Uuid::from_u128(2));
    let stored = binding(
        stored_community,
        BindingStatus::Active,
        AggregateVersion::FIRST,
    );
    let repository = repository_with_current(&stored, 1);

    let result = repository
        .current(requested_community, stored.binding_id())
        .await;

    assert!(matches!(
        result,
        Err(IdentityBindingRepositoryError::TenantBoundaryViolation)
    ));
}

#[tokio::test]
async fn identity_binding_repository_appends_a_revocation_as_the_current_version() {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    let active = binding(community_id, BindingStatus::Active, AggregateVersion::FIRST);
    let revoked = binding(
        community_id,
        BindingStatus::Revoked,
        AggregateVersion::new(2).expect("version two"),
    );
    let repository = repository_with_current(&active, 3);

    let stored = repository
        .append_version(community_id, Some(active.version()), &revoked)
        .await
        .expect("append revocation");

    assert_eq!(stored, revoked);
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("set_config('app.community_id'"), "{log}");
    assert!(log.contains("SET is_current = false"), "{log}");
    assert!(log.contains("collaboration_identity_bindings"), "{log}");
    assert!(log.contains("revoked"), "{log}");
}

#[tokio::test]
async fn identity_binding_repository_rolls_back_a_version_conflict() {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    let active = binding(community_id, BindingStatus::Active, AggregateVersion::FIRST);
    let revoked = binding(
        community_id,
        BindingStatus::Revoked,
        AggregateVersion::new(2).expect("version two"),
    );
    let repository = repository_with_current(&active, 1);

    let result = repository
        .append_version(
            community_id,
            Some(AggregateVersion::new(2).expect("stale expected version")),
            &revoked,
        )
        .await;

    assert!(matches!(
        result,
        Err(IdentityBindingRepositoryError::VersionConflict)
    ));
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("ROLLBACK"), "{log}");
    assert!(!log.contains("SET is_current = false"), "{log}");
    assert!(!log.contains("INSERT INTO"), "{log}");
}
