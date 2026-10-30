use sqlx::migrate::{MigrationSource, MigrationType};
use std::path::Path;

const VERSION: i64 = 20_260_815_000_100;

#[tokio::test]
async fn identity_binding_migration_has_safe_forward_and_down_paths() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let binding_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(binding_migrations.len(), 2);

    let up = binding_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("reversible up migration");
    let down = binding_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("reversible down migration");
    assert!(
        up.sql
            .contains("CREATE TABLE public.collaboration_identity_bindings")
    );
    assert!(
        down.sql
            .contains("DROP TABLE public.collaboration_identity_bindings")
    );
}

#[test]
fn identity_binding_migration_is_tenant_fenced_and_secret_free() {
    let up = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/20260815000100_collaboration_identity_bindings.up.sql"
    ));
    for required in [
        "PRIMARY KEY (community_id, binding_id, version)",
        "(community_id, binding_id)",
        "(community_id, service_account_id, profile_id)",
        "(community_id, nostr_public_key)",
        "FOREIGN KEY (community_id, predecessor_binding_id, predecessor_version)",
        "FOREIGN KEY (community_id, successor_binding_id, successor_version)",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(up.contains(required), "missing tenant fence {required}");
    }

    let normalized = up.to_ascii_lowercase();
    for forbidden in [
        "private_key",
        "secret_key",
        "nsec",
        "mnemonic",
        "seed_phrase",
        "raw_challenge",
        "challenge_payload",
    ] {
        assert!(
            !normalized.contains(forbidden),
            "migration contains forbidden secret column {forbidden}"
        );
    }
}
