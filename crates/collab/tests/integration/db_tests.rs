mod buffer_tests;
mod channel_tests;
mod db_tests;
mod extension_tests;
mod migrations;

use std::sync::Arc;

use collections::HashSet;
use gpui::BackgroundExecutor;
use parking_lot::Mutex;
use rand::prelude::*;
use sea_orm::ConnectionTrait;
use sqlx::migrate::MigrateDatabase;

pub use self::migrations::run_database_migrations;

use collab::db::*;

pub struct TestDb {
    pub db: Option<Arc<Database>>,
    pub connection: Option<sqlx::AnyConnection>,
}

impl TestDb {
    pub fn sqlite(executor: BackgroundExecutor) -> Self {
        let url = "sqlite::memory:";
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        let mut db = runtime.block_on(async {
            let mut options = ConnectOptions::new(url);
            options.max_connections(5);
            let mut db = Database::new(options).await.unwrap();
            let sql = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/migrations.sqlite/20221109000000_test_schema.sql"
            ));
            db.pool
                .execute(sea_orm::Statement::from_string(
                    db.pool.get_database_backend(),
                    sql,
                ))
                .await
                .unwrap();
            db.initialize_notification_kinds().await.unwrap();
            db
        });

        db.test_options = Some(DatabaseTestOptions {
            executor,
            runtime,
            query_failure_probability: parking_lot::Mutex::new(0.0),
        });

        Self {
            db: Some(Arc::new(db)),
            connection: None,
        }
    }

    pub fn postgres(executor: BackgroundExecutor) -> Self {
        static LOCK: Mutex<()> = Mutex::new(());

        let _guard = LOCK.lock();
        let mut rng = StdRng::from_os_rng();
        let database_name = format!("zed-test-{}", rng.random::<u128>());
        let url = std::env::var("COLLAB_TEST_DATABASE_URL")
            .ok()
            .and_then(|base| {
                let mut base = url::Url::parse(&base).ok()?;
                base.set_path(&database_name);
                Some(base.to_string())
            })
            .unwrap_or_else(|| format!("postgres://postgres@localhost/{database_name}"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        let mut db = runtime.block_on(async {
            sqlx::Postgres::create_database(&url)
                .await
                .expect("failed to create test db");
            let mut options = ConnectOptions::new(url);
            // A zero idle timeout reaps every returned connection and loses prepared statements.
            options.max_connections(5);
            let mut db = Database::new(options).await.unwrap();
            let migrations_path = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
            run_database_migrations(db.options(), migrations_path)
                .await
                .unwrap();
            db.initialize_notification_kinds().await.unwrap();
            db
        });

        db.test_options = Some(DatabaseTestOptions {
            executor,
            runtime,
            query_failure_probability: parking_lot::Mutex::new(0.0),
        });

        Self {
            db: Some(Arc::new(db)),
            connection: None,
        }
    }

    pub fn db(&self) -> &Arc<Database> {
        self.db.as_ref().unwrap()
    }

    pub fn set_query_failure_probability(&self, probability: f64) {
        let database = self.db.as_ref().unwrap();
        let test_options = database.test_options.as_ref().unwrap();
        *test_options.query_failure_probability.lock() = probability;
    }
}

#[macro_export]
macro_rules! test_both_dbs {
    ($test_name:ident, $postgres_test_name:ident, $sqlite_test_name:ident) => {
        #[gpui::test]
        async fn $postgres_test_name(cx: &mut gpui::TestAppContext) {
            // In CI, only run postgres tests on Linux (where we have the postgres service).
            // Locally, always run them (assuming postgres is available).
            if std::env::var("CI").is_ok() && !cfg!(target_os = "linux") {
                return;
            }
            let test_db = $crate::db_tests::TestDb::postgres(cx.executor().clone());
            $test_name(test_db.db()).await;
        }

        #[gpui::test]
        async fn $sqlite_test_name(cx: &mut gpui::TestAppContext) {
            let test_db = $crate::db_tests::TestDb::sqlite(cx.executor().clone());
            $test_name(test_db.db()).await;
        }
    };
}

#[gpui::test]
async fn postgres_fixture_reuses_connections_and_closes_owned_database(
    cx: &mut gpui::TestAppContext,
) {
    let fixture = TestDb::postgres(cx.executor());
    let database = fixture.db();
    let pool = database.pool.get_postgres_connection_pool().clone();
    let url = database.options.get_url().to_owned();
    let backend_ids = database
        .test_options
        .as_ref()
        .expect("owned database runtime")
        .runtime
        .block_on(async {
            async fn settle_idle_connections(pool: &sqlx::PgPool) {
                while pool.num_idle() != pool.size() as usize {
                    tokio::task::yield_now().await;
                }
            }
            settle_idle_connections(&pool).await;
            let initial_size = pool.size();
            assert!(initial_size > 0 && initial_size <= 5);
            let mut connections = Vec::new();
            for _ in 0..initial_size {
                connections.push(pool.acquire().await?);
            }
            let mut initial_backend_ids = HashSet::default();
            for connection in &mut connections {
                let backend_id: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
                    .fetch_one(&mut **connection)
                    .await?;
                initial_backend_ids.insert(backend_id);
            }
            assert_eq!(initial_backend_ids.len(), initial_size as usize);
            drop(connections);
            settle_idle_connections(&pool).await;
            let mut backend_ids = HashSet::default();
            for _ in 0..16 {
                tokio::task::yield_now().await;
                let backend_id: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
                    .fetch_one(&pool)
                    .await?;
                assert!(
                    initial_backend_ids.contains(&backend_id),
                    "sequential fixture queries must reuse their exact existing backend: {backend_id}, initial={initial_backend_ids:?}"
                );
                backend_ids.insert(backend_id);
                settle_idle_connections(&pool).await;
            }
            assert_eq!(pool.size(), initial_size);
            anyhow::Ok(backend_ids)
        })
        .expect("query owned PostgreSQL connections");
    assert!(!backend_ids.is_empty());
    drop(fixture);
    assert!(
        pool.is_closed(),
        "fixture disposal must close its owned pool"
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .expect("database disposal verification runtime");
    assert!(
        !runtime
            .block_on(sqlx::Postgres::database_exists(&url))
            .expect("inspect disposed database"),
        "fixture disposal must remove only its owned database"
    );
}

impl Drop for TestDb {
    fn drop(&mut self) {
        let db = self.db.take().unwrap();
        if let sea_orm::DatabaseBackend::Postgres = db.pool.get_database_backend() {
            db.test_options.as_ref().unwrap().runtime.block_on(async {
                use util::ResultExt;
                db.pool.get_postgres_connection_pool().close().await;
                sqlx::Postgres::drop_database(db.options.get_url())
                    .await
                    .log_err();
            })
        }
    }
}

#[track_caller]
fn assert_channel_tree_matches(actual: Vec<Channel>, expected: Vec<Channel>) {
    let expected_channels = expected.into_iter().collect::<HashSet<_>>();
    let actual_channels = actual.into_iter().collect::<HashSet<_>>();
    pretty_assertions::assert_eq!(expected_channels, actual_channels);
}

fn channel_tree(channels: &[(ChannelId, &[ChannelId], &'static str)]) -> Vec<Channel> {
    use std::collections::HashMap;

    let mut result = Vec::new();
    let mut order_by_parent: HashMap<Vec<ChannelId>, i32> = HashMap::new();

    for (id, parent_path, name) in channels {
        let parent_key = parent_path.to_vec();
        let order = if parent_key.is_empty() {
            1
        } else {
            *order_by_parent
                .entry(parent_key.clone())
                .and_modify(|e| *e += 1)
                .or_insert(1)
        };

        result.push(Channel {
            id: *id,
            name: (*name).to_owned(),
            visibility: ChannelVisibility::Members,
            parent_path: parent_key,
            channel_order: order,
        });
    }

    result
}

async fn new_test_user(db: &Arc<Database>) -> UserId {
    db.create_user(false).await.unwrap().user_id
}
