use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use collab::{
    collaboration_command::{
        CommandAdapter, DomainCommand, DomainCommandDisposition, DomainCommandSink,
        DomainCommandSubmissionError,
    },
    db::collaboration::outbox::{
        AppliedCommand, CommandFingerprint, OutboxOperation, TransactionalCommandMutation,
        TransactionalCommandOutbox,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateVersion, AuthenticatedPrincipal, CommunityId, OperationId, PrincipalId,
    PrincipalScopes, Provenance, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseTransaction, DbErr, MockDatabase, MockExecResult,
    Statement, Value as SeaValue,
};
use sha2::{Digest, Sha256, Sha384};
use sqlx::migrate::{MigrationSource, MigrationType};
use uuid::Uuid;

const VERSION: i64 = 20_260_820_000_400;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000400_collaboration_outbox.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000400_collaboration_outbox.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(value: u128) -> TenantContext {
    let community_id = community(value);
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "collaboration-outbox")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn command(payload: &[u8]) -> DomainCommand<Vec<u8>> {
    let tenant = tenant(1);
    let principal = AuthenticatedPrincipal::service(
        PrincipalId::from_uuid(Uuid::from_u128(10)),
        tenant.community_id(),
        "collaboration-command-test",
        PrincipalScopes::default(),
    )
    .expect("service principal");
    DomainCommand::new(
        OperationId::from_uuid(Uuid::from_u128(20)),
        tenant,
        principal,
        Some(AggregateVersion::FIRST),
        None,
        CommandAdapter::NostrInProcess,
        payload.to_vec(),
    )
}

fn fingerprint(payload: &[u8]) -> [u8; 32] {
    Sha256::digest(payload).into()
}

fn receipt_row(payload: &[u8]) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("contract_version".into(), 1_i32.into()),
        ("principal_id".into(), Uuid::from_u128(10).into()),
        (
            "originating_adapter".into(),
            "nostr_in_process".to_owned().into(),
        ),
        ("command_kind".into(), "test.mutation".to_owned().into()),
        (
            "command_fingerprint".into(),
            fingerprint(payload).to_vec().into(),
        ),
        ("authoritative_version_text".into(), "2".to_owned().into()),
        ("has_outbox".into(), true.into()),
    ])
}

fn successful_exec() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

struct TestMutation {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl TransactionalCommandMutation<Vec<u8>> for TestMutation {
    fn fingerprint(
        &self,
        command: &DomainCommand<Vec<u8>>,
    ) -> Result<CommandFingerprint, DomainCommandSubmissionError> {
        CommandFingerprint::new("test.mutation", command.payload())
    }

    async fn apply(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<Vec<u8>>,
    ) -> Result<AppliedCommand, DomainCommandSubmissionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                "UPDATE public.authoritative_test_records SET version = 2".to_owned(),
            ))
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let outbox_operation = OutboxOperation::new(
            "test.mutated",
            Provenance::new(
                SourceSystem::Zed,
                SourceRecordId::new(command.operation_id().to_string()).expect("source ID"),
                1_900_000_000_000,
            )
            .with_source_version("2"),
            command.payload().clone(),
        )?;
        Ok(AppliedCommand::new(
            AggregateVersion::new(2).expect("version two"),
            outbox_operation,
        ))
    }
}

#[tokio::test]
async fn collaboration_outbox_retry_returns_one_receipt_without_reapplying() {
    let calls = Arc::new(AtomicUsize::new(0));
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([
            successful_exec(),
            successful_exec(),
            successful_exec(),
            successful_exec(),
            successful_exec(),
            successful_exec(),
            successful_exec(),
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 0,
            },
        ])
        .append_query_results([vec![receipt_row(b"payload")]])
        .into_connection();
    let store = TransactionalCommandOutbox::new(
        database,
        TestMutation {
            calls: calls.clone(),
        },
    )
    .expect("Postgres outbox");

    let first = store.submit(command(b"payload")).await.expect("accepted");
    let retry = store.submit(command(b"payload")).await.expect("duplicate");

    assert_eq!(first.disposition(), DomainCommandDisposition::Applied);
    assert_eq!(retry.disposition(), DomainCommandDisposition::Duplicate);
    assert_eq!(first.authoritative_version(), retry.authoritative_version());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let log = format!("{:#?}", store.into_connection().into_transaction_log());
    let lock_position = log
        .find("pg_advisory_xact_lock")
        .expect("topic writer lock");
    let insert_position = log
        .find("INSERT INTO public.collaboration_outbox")
        .expect("outbox insertion");
    assert!(
        lock_position < insert_position,
        "serialize before allocating the cursor"
    );
    assert_eq!(
        log.matches("pg_advisory_xact_lock").count(),
        1,
        "duplicates do not acquire another writer lock"
    );
    assert_eq!(
        log.matches("INSERT INTO public.collaboration_outbox")
            .count(),
        1
    );
    assert_eq!(
        log.matches("UPDATE public.authoritative_test_records")
            .count(),
        1
    );
}

#[tokio::test]
async fn collaboration_outbox_rolls_back_the_authoritative_mutation_when_enqueue_fails() {
    let calls = Arc::new(AtomicUsize::new(0));
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([
            successful_exec(),
            successful_exec(),
            successful_exec(),
            successful_exec(),
        ])
        .append_exec_errors([DbErr::Custom("outbox interrupted".to_owned())])
        .into_connection();
    let store = TransactionalCommandOutbox::new(
        database,
        TestMutation {
            calls: calls.clone(),
        },
    )
    .expect("Postgres outbox");

    assert_eq!(
        store.submit(command(b"payload")).await,
        Err(DomainCommandSubmissionError::Unavailable)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let log = format!("{:#?}", store.into_connection().into_transaction_log());
    assert!(log.contains("UPDATE public.authoritative_test_records"));
    assert!(log.contains("INSERT INTO public.collaboration_outbox"));
    assert!(log.contains("ROLLBACK"));
    assert!(!log.contains("COMMIT"));
}

#[tokio::test]
async fn collaboration_outbox_lock_failure_rolls_back_without_allocating_a_cursor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([successful_exec(), successful_exec(), successful_exec()])
        .append_exec_errors([DbErr::Custom("topic lock unavailable".to_owned())])
        .into_connection();
    let store = TransactionalCommandOutbox::new(
        database,
        TestMutation {
            calls: calls.clone(),
        },
    )
    .expect("Postgres outbox");
    assert_eq!(
        store.submit(command(b"payload")).await,
        Err(DomainCommandSubmissionError::Unavailable)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let log = format!("{:#?}", store.into_connection().into_transaction_log());
    assert!(log.contains("pg_advisory_xact_lock"));
    assert!(!log.contains("INSERT INTO public.collaboration_outbox"));
    assert!(log.contains("ROLLBACK"));
    assert!(!log.contains("COMMIT"));
}

#[tokio::test]
async fn collaboration_outbox_rejects_operation_id_reuse_with_different_content() {
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([
            successful_exec(),
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 0,
            },
        ])
        .append_query_results([vec![receipt_row(b"original")]])
        .into_connection();
    let store = TransactionalCommandOutbox::new(
        database,
        TestMutation {
            calls: Arc::new(AtomicUsize::new(0)),
        },
    )
    .expect("Postgres outbox");

    assert_eq!(
        store.submit(command(b"different")).await,
        Err(DomainCommandSubmissionError::Rejected)
    );
    let log = format!("{:#?}", store.into_connection().into_transaction_log());
    assert!(log.contains("ROLLBACK"));
    assert!(!log.contains("authoritative_test_records"));
}

#[tokio::test]
async fn collaboration_outbox_rejects_cross_tenant_principals_before_database_io() {
    let tenant = tenant(1);
    let principal = AuthenticatedPrincipal::service(
        PrincipalId::from_uuid(Uuid::from_u128(10)),
        community(2),
        "cross-tenant-test",
        PrincipalScopes::default(),
    )
    .expect("service principal");
    let command = DomainCommand::new(
        OperationId::from_uuid(Uuid::from_u128(20)),
        tenant,
        principal,
        None,
        None,
        CommandAdapter::NostrInProcess,
        b"payload".to_vec(),
    );
    let store = TransactionalCommandOutbox::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
        TestMutation {
            calls: Arc::new(AtomicUsize::new(0)),
        },
    )
    .expect("Postgres outbox");

    assert_eq!(
        store.submit(command).await,
        Err(DomainCommandSubmissionError::Rejected)
    );
    assert!(store.into_connection().into_transaction_log().is_empty());
}

#[test]
fn collaboration_outbox_schema_is_ordered_tenant_fenced_and_reversible() {
    for required in [
        "PRIMARY KEY (community_id, operation_id)",
        "UNIQUE (community_id, operation_id)",
        "outbox_sequence bigint GENERATED ALWAYS AS IDENTITY",
        "PRIMARY KEY (community_id, outbox_sequence)",
        "REFERENCES public.collaboration_command_receipts (community_id, operation_id)",
        "octet_length(command_fingerprint) = 32",
        "octet_length(payload) <= 1048576",
        "collaboration_outbox_delivery",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(UP.contains(required), "missing outbox invariant {required}");
    }
    assert!(DOWN.starts_with("DROP TABLE public.collaboration_outbox;"));
    assert!(!DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn collaboration_outbox_migration_has_stable_up_and_down_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let outbox_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(outbox_migrations.len(), 2);
    let up = outbox_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("outbox up migration");
    let down = outbox_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("outbox down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}
