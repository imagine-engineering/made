use std::collections::BTreeMap;

use collab::audit::repository::{
    AuditExportCursor, AuditHead, AuditRepository, AuditRepositoryError,
};
use collaboration_domain::{
    AuditAction, AuditChainBridge, AuditChainPosition, AuditChainSource, AuditEntry, AuditField,
    AuditFieldName, AuditFields, AuditHash, AuditOutcome, AuditRecord, AuditRedaction, AuditValue,
    CommunityId, OperationId, PrincipalId, TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(value: u128) -> TenantContext {
    let community_id = community(value);
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "audit-chain-integration")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn record(operation: u128, action: &str, outcome: AuditOutcome) -> AuditRecord {
    AuditRecord::new(
        OperationId::from_uuid(Uuid::from_u128(operation)),
        AuditAction::new(action).expect("action"),
        Some(PrincipalId::from_uuid(Uuid::from_u128(90))),
        outcome,
        1_900_000_000_000 + u64::try_from(operation).expect("fixture operation"),
        AuditFields::new(vec![AuditField::new(
            AuditFieldName::new("private_payload").expect("field name"),
            AuditValue::Redacted(AuditRedaction::PrivateContent),
        )])
        .expect("fields"),
    )
    .expect("record")
}

fn native_chain(community_id: CommunityId) -> [AuditEntry; 3] {
    let first = AuditEntry::append(
        AuditChainPosition::genesis(community_id).expect("genesis"),
        record(10, "membership.add", AuditOutcome::Succeeded),
    )
    .expect("first entry");
    let second = AuditEntry::append(
        AuditChainPosition::after(&first).expect("second position"),
        record(11, "workflow.execute_step", AuditOutcome::Failed),
    )
    .expect("second entry");
    let third = AuditEntry::append(
        AuditChainPosition::after(&second).expect("third position"),
        record(12, "moderation.resolve_report", AuditOutcome::Succeeded),
    )
    .expect("third entry");
    [first, second, third]
}

fn imported_entry(community_id: CommunityId) -> AuditEntry {
    let bridge =
        AuditChainBridge::new(AuditChainSource::BuzzV1, 42, AuditHash::from_bytes([7; 32]))
            .expect("bridge");
    AuditEntry::append(
        AuditChainPosition::from_imported(community_id, bridge).expect("import position"),
        record(20, "migration.complete", AuditOutcome::Succeeded),
    )
    .expect("imported entry")
}

fn repository(
    query_results: Vec<Vec<BTreeMap<String, SeaValue>>>,
    affected_rows: usize,
) -> AuditRepository {
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results(query_results)
        .append_exec_results((0..affected_rows).map(|_| MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }))
        .into_connection();
    AuditRepository::new(database).expect("repository")
}

fn entry_row(entry: &AuditEntry) -> BTreeMap<String, SeaValue> {
    let bridge = entry.chain_bridge();
    BTreeMap::from([
        ("community_id".into(), entry.community_id().as_uuid().into()),
        ("sequence_text".into(), entry.sequence().to_string().into()),
        ("entry_hash".into(), entry.hash().as_bytes().to_vec().into()),
        (
            "previous_hash".into(),
            entry
                .previous_hash()
                .map(|hash| hash.as_bytes().to_vec())
                .into(),
        ),
        (
            "operation_id".into(),
            entry.record().operation_id().as_uuid().into(),
        ),
        (
            "action".into(),
            entry.record().action().as_str().to_owned().into(),
        ),
        (
            "actor_principal_id".into(),
            entry
                .record()
                .actor_principal_id()
                .map(PrincipalId::as_uuid)
                .into(),
        ),
        (
            "outcome".into(),
            outcome_name(entry.record().outcome()).to_owned().into(),
        ),
        (
            "occurred_at_millis_text".into(),
            entry.record().occurred_at_millis().to_string().into(),
        ),
        (
            "fields_json".into(),
            serde_json::to_string(entry.record().fields())
                .expect("fields JSON")
                .into(),
        ),
        (
            "bridge_source".into(),
            bridge.map(|_| "buzz_v1".to_owned()).into(),
        ),
        (
            "bridge_source_sequence_text".into(),
            bridge
                .map(|bridge| bridge.source_sequence().to_string())
                .into(),
        ),
        (
            "bridge_source_head".into(),
            bridge
                .map(|bridge| bridge.source_head().as_bytes().to_vec())
                .into(),
        ),
    ])
}

fn cursor_row(exporter_id: &str, head: AuditHead) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("exporter_id".into(), exporter_id.to_owned().into()),
        ("cursor_version_text".into(), "1".to_owned().into()),
        (
            "exported_through_sequence_text".into(),
            head.sequence().to_string().into(),
        ),
        (
            "exported_through_hash".into(),
            head.hash().as_bytes().to_vec().into(),
        ),
    ])
}

fn outcome_name(outcome: AuditOutcome) -> &'static str {
    match outcome {
        AuditOutcome::Succeeded => "succeeded",
        AuditOutcome::Failed => "failed",
        AuditOutcome::Denied => "denied",
        AuditOutcome::Cancelled => "cancelled",
    }
}

async fn read_error(
    tenant: &TenantContext,
    rows: Vec<BTreeMap<String, SeaValue>>,
    from_sequence: u64,
) -> AuditRepositoryError {
    repository(vec![rows], 1)
        .read_segment(tenant, from_sequence, 10)
        .await
        .expect_err("corrupt chain must fail")
}

#[tokio::test]
async fn audit_chain_verifies_full_partial_and_cursor_export_segments() {
    let tenant = tenant(1);
    let [first, second, third] = native_chain(tenant.community_id());

    let full = repository(
        vec![vec![
            entry_row(&first),
            entry_row(&second),
            entry_row(&third),
        ]],
        1,
    )
    .read_segment(&tenant, 1, 10)
    .await
    .expect("full chain");
    assert_eq!(
        full.entries(),
        &[first.clone(), second.clone(), third.clone()]
    );
    assert_eq!(full.end_head(), Some(AuditHead::from_entry(&third)));

    let partial = repository(vec![vec![entry_row(&second), entry_row(&third)]], 1)
        .read_segment(&tenant, 2, 10)
        .await
        .expect("partial chain");
    assert_eq!(partial.entries(), &[second.clone(), third.clone()]);

    let exporter_id = "operator-audit-v1";
    let export_repository = repository(
        vec![
            vec![cursor_row(exporter_id, AuditHead::from_entry(&first))],
            vec![entry_row(&second), entry_row(&third)],
        ],
        2,
    );
    let cursor: AuditExportCursor = export_repository
        .load_export_cursor(&tenant, exporter_id)
        .await
        .expect("cursor read")
        .expect("cursor");
    let exported = export_repository
        .export_segment(&tenant, Some(&cursor), 10)
        .await
        .expect("export segment");
    assert_eq!(exported.entries(), &[second.clone(), third.clone()]);
    assert_eq!(exported.end_head(), Some(AuditHead::from_entry(&third)));
}

#[tokio::test]
async fn audit_chain_detects_deletion_reorder_and_mutation() {
    let tenant = tenant(1);
    let [first, second, third] = native_chain(tenant.community_id());

    let deletion = read_error(&tenant, vec![entry_row(&first), entry_row(&third)], 1).await;
    assert!(matches!(
        deletion,
        AuditRepositoryError::InvalidRecord | AuditRepositoryError::Domain(_)
    ));

    let reorder = read_error(
        &tenant,
        vec![entry_row(&first), entry_row(&third), entry_row(&second)],
        1,
    )
    .await;
    assert!(matches!(
        reorder,
        AuditRepositoryError::InvalidRecord | AuditRepositoryError::Domain(_)
    ));

    let mut mutated_second = entry_row(&second);
    mutated_second.insert("action".into(), "workflow.cancel_run".to_owned().into());
    let mutation = read_error(&tenant, vec![entry_row(&first), mutated_second], 1).await;
    assert!(matches!(mutation, AuditRepositoryError::Domain(_)));
}

#[tokio::test]
async fn audit_chain_detects_wrong_imported_head_with_operator_safe_diagnostics() {
    let tenant = tenant(0xfeed);
    let imported = imported_entry(tenant.community_id());
    let verified = repository(vec![vec![entry_row(&imported)]], 1)
        .read_segment(&tenant, imported.sequence(), 10)
        .await
        .expect("valid imported bridge");
    assert_eq!(verified.entries(), std::slice::from_ref(&imported));

    let mut wrong_head = entry_row(&imported);
    wrong_head.insert("bridge_source_head".into(), vec![8_u8; 32].into());
    let error = read_error(&tenant, vec![wrong_head], imported.sequence()).await;
    assert!(matches!(error, AuditRepositoryError::Domain(_)));

    let diagnostic = error.to_string();
    for private_value in [
        tenant.community_id().to_string(),
        imported.record().operation_id().to_string(),
        hex::encode(imported.hash().as_bytes()),
        "private_payload".to_owned(),
    ] {
        assert!(!diagnostic.contains(&private_value));
    }
}

fn head_row(head: AuditHead) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("sequence_text".into(), head.sequence().to_string().into()),
        ("entry_hash".into(), head.hash().as_bytes().to_vec().into()),
    ])
}

fn tenant_setting_row(tenant: &TenantContext) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([(
        "community_id".into(),
        tenant.community_id().to_string().into(),
    )])
}

#[tokio::test]
async fn usage_accounting_canonical_operation_retry_and_conflict() -> anyhow::Result<()> {
    use collab::audit::repository::ExpectedAuditHead;
    use collab::audit::usage::{
        UsageAccounting, UsageAccountingError, UsageRecord, UsageSourceReference, UsageWriteOutcome,
    };
    let tenant = tenant(1);
    let record = |duration| {
        UsageRecord::agent_job(
            &tenant,
            OperationId::from_uuid(Uuid::from_u128(10)),
            UsageSourceReference::agent_job(&tenant, Uuid::from_u128(100))?,
            1,
            duration,
            AuditOutcome::Succeeded,
            1_900_000_000_010,
        )
    };
    let original = record(50)?;
    let first = UsageAccounting::new(repository(
        vec![vec![tenant_setting_row(&tenant)], vec![], vec![]],
        5,
    ));
    let expected_entry = match first
        .record(&tenant, ExpectedAuditHead::Empty, original.clone())
        .await?
    {
        UsageWriteOutcome::Applied(entry) => entry,
        UsageWriteOutcome::AlreadyRecorded(_) => anyhow::bail!("first write was not applied"),
    };
    expected_entry.verify()?;
    assert_eq!(
        expected_entry.record().operation_id(),
        original.operation_id()
    );
    assert_eq!(expected_entry.record().action().as_str(), "usage.agent_job");
    assert_eq!(expected_entry.record().actor_principal_id(), None);
    let retry = UsageAccounting::new(repository(
        vec![
            vec![tenant_setting_row(&tenant)],
            vec![entry_row(&expected_entry)],
        ],
        1,
    ));
    assert!(
        matches!(retry.record(&tenant, ExpectedAuditHead::Empty, original.clone()).await?, UsageWriteOutcome::AlreadyRecorded(entry) if entry == expected_entry)
    );
    let retry_log = format!(
        "{:#?}",
        retry
            .into_repository()
            .into_connection()
            .into_transaction_log()
    );
    assert!(retry_log.contains("current_setting('app.community_id'"));
    assert!(retry_log.contains("operation_id = $2"));
    assert!(!retry_log.contains("INSERT INTO public.collaboration_audit_entries"));
    let racing = UsageAccounting::new(repository(
        vec![
            vec![tenant_setting_row(&tenant)],
            vec![],
            vec![head_row(AuditHead::from_entry(&expected_entry))],
            vec![tenant_setting_row(&tenant)],
            vec![entry_row(&expected_entry)],
        ],
        4,
    ));
    assert!(
        matches!(racing.record(&tenant, ExpectedAuditHead::Empty, original.clone()).await?, UsageWriteOutcome::AlreadyRecorded(entry) if entry == expected_entry)
    );
    let racing_log = format!(
        "{:#?}",
        racing
            .into_repository()
            .into_connection()
            .into_transaction_log()
    );
    assert!(!racing_log.contains("INSERT INTO public.collaboration_audit_entries"));
    let conflict = UsageAccounting::new(repository(
        vec![
            vec![tenant_setting_row(&tenant)],
            vec![entry_row(&expected_entry)],
        ],
        1,
    ));
    assert!(matches!(
        conflict
            .record(&tenant, ExpectedAuditHead::Empty, record(51)?)
            .await,
        Err(UsageAccountingError::DuplicateConflict)
    ));
    Ok(())
}

#[tokio::test]
async fn audit_operation_supplied_transaction_fences_tenant_and_stored_integrity()
-> anyhow::Result<()> {
    use sea_orm::TransactionTrait as _;
    let tenant = tenant(1);
    let [first, second, _] = native_chain(tenant.community_id());
    for entry in [&first, &second] {
        let connection = repository(
            vec![vec![tenant_setting_row(&tenant)], vec![entry_row(entry)]],
            1,
        )
        .into_connection();
        let transaction = connection.begin().await?;
        let actual = AuditRepository::load_operation_in_transaction(
            &transaction,
            &tenant,
            entry.record().operation_id(),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("operation missing"))?;
        assert_eq!(&actual, entry);
        transaction.rollback().await?;
    }
    let foreign = self::tenant(2);
    let connection = repository(vec![vec![tenant_setting_row(&foreign)]], 0).into_connection();
    let transaction = connection.begin().await?;
    assert!(matches!(
        AuditRepository::load_operation_in_transaction(
            &transaction,
            &tenant,
            first.record().operation_id()
        )
        .await,
        Err(AuditRepositoryError::TenantBoundaryViolation)
    ));
    transaction.rollback().await?;
    let log = format!("{:#?}", connection.into_transaction_log());
    assert!(!log.contains("operation_id = $2"));
    assert!(!log.contains("set_config"));
    let mut wrong_hash = entry_row(&first);
    wrong_hash.insert("entry_hash".into(), vec![255_u8; 32].into());
    let mut wrong_fields = entry_row(&first);
    wrong_fields.insert(
        "fields_json".into(),
        serde_json::to_string(&AuditFields::default())?.into(),
    );
    for row in [wrong_hash, wrong_fields] {
        let connection =
            repository(vec![vec![tenant_setting_row(&tenant)], vec![row]], 1).into_connection();
        let transaction = connection.begin().await?;
        assert!(matches!(
            AuditRepository::load_operation_in_transaction(
                &transaction,
                &tenant,
                first.record().operation_id()
            )
            .await,
            Err(AuditRepositoryError::Domain(
                collaboration_domain::AuditError::HashMismatch
            ))
        ));
        transaction.rollback().await?;
    }
    let connection =
        repository(vec![vec![tenant_setting_row(&tenant)], vec![]], 1).into_connection();
    let transaction = connection.begin().await?;
    assert!(
        AuditRepository::load_operation_in_transaction(
            &transaction,
            &tenant,
            first.record().operation_id()
        )
        .await?
        .is_none()
    );
    transaction.rollback().await?;
    Ok(())
}
