use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::{
    db::collaboration::rebuild::{
        ProjectionDriftState, ProjectionMaterialization, ProjectionRebuildAdapter,
        ProjectionRebuildError, ProjectionRebuilder, ProjectionRow, ProjectionRows,
        ProjectionSource,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    CommunityId, Provenance, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseTransaction, MockDatabase, MockExecResult, Statement,
    Value as SeaValue,
};
use uuid::Uuid;

fn tenant(value: u128) -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(value));
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "projection-rebuild")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn source() -> ProjectionSource {
    ProjectionSource::new(
        "conversation_activity",
        Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new("conversation:42").expect("source ID"),
            1_900_000_000_000,
        )
        .with_source_version("7"),
    )
    .expect("projection source")
}

fn authoritative_rows() -> ProjectionRows {
    ProjectionRows::new(vec![
        ProjectionRow::new("activity:2", b"second".to_vec()).expect("row"),
        ProjectionRow::new("activity:1", b"first".to_vec()).expect("row"),
    ])
    .expect("rows")
}

fn successful_exec() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

#[derive(Clone)]
struct TestProjectionAdapter {
    authority: ProjectionRows,
    projection: Arc<Mutex<Option<ProjectionMaterialization>>>,
    seed_drift: bool,
    fail_replace: bool,
}

impl TestProjectionAdapter {
    fn lock_projection(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<ProjectionMaterialization>>, ProjectionRebuildError>
    {
        self.projection
            .lock()
            .map_err(|_| ProjectionRebuildError::AdapterRejected)
    }
}

#[async_trait]
impl ProjectionRebuildAdapter for TestProjectionAdapter {
    async fn load_authority(
        &self,
        transaction: &DatabaseTransaction,
        _tenant: &TenantContext,
        _source: &ProjectionSource,
    ) -> Result<ProjectionRows, ProjectionRebuildError> {
        transaction
            .query_all(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT source_version, record_key, payload FROM public.authoritative_test_records"
                    .to_owned(),
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        Ok(self.authority.clone())
    }

    async fn replace_projection(
        &self,
        transaction: &DatabaseTransaction,
        _tenant: &TenantContext,
        source: &ProjectionSource,
        rows: &ProjectionRows,
    ) -> Result<(), ProjectionRebuildError> {
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                "DELETE FROM public.derived_test_projection WHERE aggregate_id = 'conversation:42'"
                    .to_owned(),
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        if self.fail_replace {
            return Err(ProjectionRebuildError::AdapterRejected);
        }
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                "INSERT INTO public.derived_test_projection SELECT * FROM rebuild_input".to_owned(),
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        *self.lock_projection()? = Some(ProjectionMaterialization::new(
            source.provenance().source_version.clone(),
            rows.clone(),
        )?);
        Ok(())
    }

    async fn load_projection(
        &self,
        transaction: &DatabaseTransaction,
        _tenant: &TenantContext,
        _source: &ProjectionSource,
    ) -> Result<ProjectionMaterialization, ProjectionRebuildError> {
        transaction
            .query_all(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT source_version, record_key, payload FROM public.derived_test_projection"
                    .to_owned(),
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        let projection = self
            .lock_projection()?
            .clone()
            .ok_or(ProjectionRebuildError::AdapterRejected)?;
        if !self.seed_drift {
            return Ok(projection);
        }
        let mut rows = projection.rows().as_slice().to_vec();
        rows.push(ProjectionRow::new("activity:drift", b"drift".to_vec())?);
        ProjectionMaterialization::new(Some("seeded-drift".to_owned()), ProjectionRows::new(rows)?)
    }
}

fn database(rebuild_count: usize) -> sea_orm::DatabaseConnection {
    let query_results = (0..rebuild_count * 2)
        .map(|_| Vec::<BTreeMap<String, SeaValue>>::new())
        .collect::<Vec<_>>();
    let exec_results = (0..rebuild_count * 4)
        .map(|_| successful_exec())
        .collect::<Vec<_>>();
    MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results(query_results)
        .append_exec_results(exec_results)
        .into_connection()
}

#[tokio::test]
async fn collaboration_projection_rebuild_is_deterministic_and_does_not_mutate_authority() {
    let projected = Arc::new(Mutex::new(None));
    let rebuilder = ProjectionRebuilder::new(
        database(2),
        TestProjectionAdapter {
            authority: authoritative_rows(),
            projection: projected.clone(),
            seed_drift: false,
            fail_replace: false,
        },
    )
    .expect("Postgres rebuilder");
    let tenant = tenant(1);
    let source = source();

    let first = rebuilder
        .rebuild(&tenant, &source)
        .await
        .expect("first rebuild");
    let first_projection = projected.lock().expect("projection lock").clone();
    let second = rebuilder
        .rebuild(&tenant, &source)
        .await
        .expect("second rebuild");
    let second_projection = projected.lock().expect("projection lock").clone();

    assert_eq!(first.state(), ProjectionDriftState::Clean);
    assert_eq!(second.state(), ProjectionDriftState::Clean);
    assert_eq!(first.authoritative_hash(), second.authoritative_hash());
    assert_eq!(first.projection_hash(), second.projection_hash());
    assert_eq!(first_projection, second_projection);
    let log = format!("{:#?}", rebuilder.into_connection().into_transaction_log());
    assert_eq!(log.matches("SELECT source_version").count(), 4);
    assert_eq!(
        log.matches("DELETE FROM public.derived_test_projection")
            .count(),
        2
    );
    assert_eq!(
        log.matches("INSERT INTO public.collaboration_projection_checkpoints")
            .count(),
        2
    );
    assert!(!log.contains("UPDATE public.authoritative_test_records"));
    assert!(!log.contains("DELETE FROM public.authoritative_test_records"));
}

#[tokio::test]
async fn collaboration_projection_rebuild_reports_seeded_drift_with_tenant_scope() {
    let rebuilder = ProjectionRebuilder::new(
        database(1),
        TestProjectionAdapter {
            authority: authoritative_rows(),
            projection: Arc::new(Mutex::new(None)),
            seed_drift: true,
            fail_replace: false,
        },
    )
    .expect("Postgres rebuilder");
    let tenant = tenant(7);

    let diagnostic = rebuilder
        .rebuild(&tenant, &source())
        .await
        .expect("drift is an observable rebuild result");

    assert_eq!(diagnostic.community_id(), tenant.community_id());
    assert_eq!(diagnostic.projection_name(), "conversation_activity");
    assert_eq!(diagnostic.source_record_id(), "conversation:42");
    assert_eq!(diagnostic.state(), ProjectionDriftState::Diverged);
    assert_eq!(diagnostic.authoritative_source_version(), Some("7"));
    assert_eq!(diagnostic.projection_source_version(), Some("seeded-drift"));
    assert_eq!(diagnostic.authoritative_count(), 2);
    assert_eq!(diagnostic.projection_count(), 3);
    assert_ne!(
        diagnostic.authoritative_hash(),
        diagnostic.projection_hash()
    );
    let log = format!("{:#?}", rebuilder.into_connection().into_transaction_log());
    assert!(log.contains("diverged"));
    assert!(log.contains(tenant.community_id().as_uuid().to_string().as_str()));
}

#[tokio::test]
async fn collaboration_projection_rebuild_rolls_back_a_partial_projection_replace() {
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([Vec::<BTreeMap<String, SeaValue>>::new()])
        .append_exec_results([successful_exec(), successful_exec()])
        .into_connection();
    let rebuilder = ProjectionRebuilder::new(
        database,
        TestProjectionAdapter {
            authority: authoritative_rows(),
            projection: Arc::new(Mutex::new(None)),
            seed_drift: false,
            fail_replace: true,
        },
    )
    .expect("Postgres rebuilder");

    assert!(matches!(
        rebuilder.rebuild(&tenant(1), &source()).await,
        Err(ProjectionRebuildError::AdapterRejected)
    ));
    let log = format!("{:#?}", rebuilder.into_connection().into_transaction_log());
    assert!(log.contains("DELETE FROM public.derived_test_projection"));
    assert!(log.contains("ROLLBACK"));
    assert!(!log.contains("collaboration_projection_checkpoints"));
}
