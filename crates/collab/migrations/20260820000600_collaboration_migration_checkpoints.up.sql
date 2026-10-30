CREATE TABLE public.collaboration_migration_runs (
    run_id uuid PRIMARY KEY,
    community_id uuid NOT NULL,
    source_revision text NOT NULL CHECK (
        octet_length(source_revision) BETWEEN 1 AND 256
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (community_id, run_id)
);

ALTER TABLE public.collaboration_migration_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_migration_runs FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_migration_runs_admission
    ON public.collaboration_migration_runs
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_migration_runs_community
    ON public.collaboration_migration_runs
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );

CREATE TABLE public.collaboration_migration_checkpoints (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    stream_name text NOT NULL CHECK (
        stream_name IN (
            'signed_events', 'community_state', 'object_git_metadata',
            'desktop_state', 'agent_state', 'workflow_state',
            'moderation_state', 'media_state'
        )
    ),
    shard_id text NOT NULL CHECK (octet_length(shard_id) BETWEEN 1 AND 256),
    checkpoint_version numeric(20, 0) NOT NULL CHECK (
        checkpoint_version BETWEEN 1 AND 18446744073709551615
    ),
    status text NOT NULL CHECK (
        status IN (
            'pending', 'running', 'interrupted', 'verifying',
            'verified', 'failed', 'rolled_back'
        )
    ),
    source_cursor_sequence numeric(20, 0) NOT NULL CHECK (
        source_cursor_sequence BETWEEN 0 AND 18446744073709551615
    ),
    source_cursor_token bytea CHECK (
        source_cursor_token IS NULL OR octet_length(source_cursor_token) <= 65536
    ),
    target_cursor_sequence numeric(20, 0) NOT NULL CHECK (
        target_cursor_sequence BETWEEN 0 AND 18446744073709551615
    ),
    target_cursor_token bytea CHECK (
        target_cursor_token IS NULL OR octet_length(target_cursor_token) <= 65536
    ),
    scanned_count numeric(20, 0) NOT NULL CHECK (
        scanned_count BETWEEN 0 AND 18446744073709551615
    ),
    imported_count numeric(20, 0) NOT NULL CHECK (
        imported_count BETWEEN 0 AND 18446744073709551615
    ),
    skipped_count numeric(20, 0) NOT NULL CHECK (
        skipped_count BETWEEN 0 AND 18446744073709551615
    ),
    failed_count numeric(20, 0) NOT NULL CHECK (
        failed_count BETWEEN 0 AND 18446744073709551615
    ),
    source_hash bytea CHECK (source_hash IS NULL OR octet_length(source_hash) = 32),
    target_hash bytea CHECK (target_hash IS NULL OR octet_length(target_hash) = 32),
    rollback_label text NOT NULL CHECK (octet_length(rollback_label) BETWEEN 1 AND 256),
    rollback_irreversible boolean NOT NULL DEFAULT false,
    irreversible_at timestamptz,
    last_error text CHECK (last_error IS NULL OR octet_length(last_error) <= 2048),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, run_id, stream_name, shard_id),
    FOREIGN KEY (community_id, run_id)
        REFERENCES public.collaboration_migration_runs (community_id, run_id)
        ON DELETE RESTRICT,
    CHECK (imported_count + skipped_count + failed_count <= scanned_count),
    CHECK (rollback_irreversible = (irreversible_at IS NOT NULL)),
    CHECK (status <> 'rolled_back' OR NOT rollback_irreversible),
    CHECK (status <> 'failed' OR last_error IS NOT NULL),
    CHECK (status NOT IN ('pending', 'running', 'verifying', 'verified', 'rolled_back')
           OR last_error IS NULL)
);

CREATE FUNCTION public.guard_collaboration_migration_checkpoint_update() RETURNS trigger AS $$
BEGIN
    IF NEW.community_id IS DISTINCT FROM OLD.community_id
       OR NEW.run_id IS DISTINCT FROM OLD.run_id
       OR NEW.stream_name IS DISTINCT FROM OLD.stream_name
       OR NEW.shard_id IS DISTINCT FROM OLD.shard_id THEN
        RAISE EXCEPTION 'migration checkpoint identity is immutable'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.checkpoint_version <> OLD.checkpoint_version + 1 THEN
        RAISE EXCEPTION 'migration checkpoint version conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;
    IF NEW.source_cursor_sequence < OLD.source_cursor_sequence
       OR NEW.target_cursor_sequence < OLD.target_cursor_sequence
       OR NEW.scanned_count < OLD.scanned_count
       OR NEW.imported_count < OLD.imported_count
       OR NEW.skipped_count < OLD.skipped_count
       OR NEW.failed_count < OLD.failed_count THEN
        RAISE EXCEPTION 'migration checkpoint progress cannot regress'
            USING ERRCODE = 'check_violation';
    END IF;
    IF (NEW.source_cursor_sequence = OLD.source_cursor_sequence
            AND NEW.source_cursor_token IS DISTINCT FROM OLD.source_cursor_token)
       OR (NEW.target_cursor_sequence = OLD.target_cursor_sequence
            AND NEW.target_cursor_token IS DISTINCT FROM OLD.target_cursor_token) THEN
        RAISE EXCEPTION 'migration checkpoint cursor token changed without progress'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.scanned_count = OLD.scanned_count
       AND NEW.imported_count = OLD.imported_count
       AND NEW.skipped_count = OLD.skipped_count
       AND NEW.failed_count = OLD.failed_count
       AND (NEW.source_hash IS DISTINCT FROM OLD.source_hash
            OR NEW.target_hash IS DISTINCT FROM OLD.target_hash) THEN
        RAISE EXCEPTION 'migration checkpoint integrity changed without progress'
            USING ERRCODE = 'check_violation';
    END IF;
    IF OLD.rollback_irreversible AND NOT NEW.rollback_irreversible THEN
        RAISE EXCEPTION 'migration rollback boundary cannot become reversible'
            USING ERRCODE = 'check_violation';
    END IF;
    IF OLD.rollback_irreversible
       AND (NEW.rollback_label IS DISTINCT FROM OLD.rollback_label
            OR NEW.irreversible_at IS DISTINCT FROM OLD.irreversible_at) THEN
        RAISE EXCEPTION 'migration irreversible boundary is immutable'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NOT (
        (OLD.status = 'pending' AND NEW.status IN ('running', 'failed'))
        OR (OLD.status = 'running' AND NEW.status IN ('running', 'interrupted', 'verifying', 'failed'))
        OR (OLD.status = 'interrupted' AND NEW.status IN ('running', 'failed', 'rolled_back'))
        OR (OLD.status = 'verifying' AND NEW.status IN ('verified', 'failed'))
        OR (OLD.status = 'failed' AND NEW.status IN ('running', 'rolled_back'))
        OR (OLD.status = 'verified' AND NEW.status = 'rolled_back')
    ) THEN
        RAISE EXCEPTION 'migration checkpoint status transition is invalid'
            USING ERRCODE = 'check_violation';
    END IF;
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_migration_checkpoints_update_guard
    BEFORE UPDATE ON public.collaboration_migration_checkpoints
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_migration_checkpoint_update();

CREATE INDEX collaboration_migration_checkpoints_status
    ON public.collaboration_migration_checkpoints (
        community_id, status, updated_at, run_id, stream_name, shard_id
    );

ALTER TABLE public.collaboration_migration_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_migration_checkpoints FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_migration_checkpoints_admission
    ON public.collaboration_migration_checkpoints
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_migration_checkpoints_community
    ON public.collaboration_migration_checkpoints
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
