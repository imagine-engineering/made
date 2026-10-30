CREATE TABLE public.collaboration_projection_checkpoints (
    community_id uuid NOT NULL,
    projection_name text NOT NULL CHECK (
        octet_length(projection_name) BETWEEN 1 AND 128
    ),
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 1024
    ),
    source_observed_at timestamptz NOT NULL,
    source_integrity_algorithm text CHECK (
        source_integrity_algorithm IS NULL OR source_integrity_algorithm IN (
            'sha256', 'nostr_event_id', 'git_object_id'
        )
    ),
    source_integrity_value text CHECK (
        source_integrity_value IS NULL
        OR octet_length(source_integrity_value) BETWEEN 1 AND 1024
    ),
    projection_version numeric(20, 0) NOT NULL CHECK (
        projection_version BETWEEN 1 AND 18446744073709551615
    ),
    reset_generation numeric(20, 0) NOT NULL CHECK (
        reset_generation BETWEEN 1 AND 18446744073709551615
    ),
    cursor bytea CHECK (cursor IS NULL OR octet_length(cursor) <= 65536),
    drift_state text NOT NULL CHECK (
        drift_state IN ('clean', 'suspect', 'diverged', 'rebuilding', 'reset_pending')
    ),
    authoritative_hash bytea CHECK (
        authoritative_hash IS NULL OR octet_length(authoritative_hash) = 32
    ),
    projection_hash bytea CHECK (
        projection_hash IS NULL OR octet_length(projection_hash) = 32
    ),
    projected_at timestamptz NOT NULL,
    reset_at timestamptz,
    last_error text CHECK (last_error IS NULL OR octet_length(last_error) <= 2048),
    PRIMARY KEY (community_id, projection_name, source_system, source_record_id),
    CHECK (
        (source_integrity_algorithm IS NULL) = (source_integrity_value IS NULL)
    ),
    CHECK (
        drift_state <> 'diverged'
        OR (
            authoritative_hash IS NOT NULL
            AND projection_hash IS NOT NULL
            AND authoritative_hash <> projection_hash
        )
    ),
    CHECK (drift_state <> 'clean' OR last_error IS NULL),
    CHECK (drift_state <> 'reset_pending' OR cursor IS NULL),
    CHECK ((reset_at IS NULL) = (reset_generation = 1))
);

CREATE INDEX collaboration_projection_checkpoints_scan
    ON public.collaboration_projection_checkpoints (
        community_id,
        projection_name,
        drift_state,
        projected_at
    );

CREATE INDEX collaboration_projection_checkpoints_source
    ON public.collaboration_projection_checkpoints (
        community_id,
        source_system,
        source_record_id,
        source_version
    );

CREATE FUNCTION public.guard_collaboration_projection_checkpoint_update() RETURNS trigger AS $$
BEGIN
    IF NEW.community_id IS DISTINCT FROM OLD.community_id
       OR NEW.projection_name IS DISTINCT FROM OLD.projection_name
       OR NEW.source_system IS DISTINCT FROM OLD.source_system
       OR NEW.source_record_id IS DISTINCT FROM OLD.source_record_id THEN
        RAISE EXCEPTION 'projection checkpoint identity is immutable'
            USING ERRCODE = 'check_violation';
    END IF;

    IF NEW.projection_version <> OLD.projection_version + 1 THEN
        RAISE EXCEPTION 'projection checkpoint version conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;

    IF NEW.reset_generation <> OLD.reset_generation
       AND NEW.reset_generation <> OLD.reset_generation + 1 THEN
        RAISE EXCEPTION 'projection reset generation conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;

    IF NEW.reset_generation = OLD.reset_generation + 1
       AND (
            NEW.drift_state <> 'reset_pending'
            OR NEW.cursor IS NOT NULL
            OR NEW.reset_at IS NULL
       ) THEN
        RAISE EXCEPTION 'projection reset must fence stale work and clear its cursor'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_projection_checkpoints_update_guard
    BEFORE UPDATE ON public.collaboration_projection_checkpoints
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_projection_checkpoint_update();

ALTER TABLE public.collaboration_projection_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_projection_checkpoints FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_projection_checkpoints_admission
    ON public.collaboration_projection_checkpoints
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_projection_checkpoints_community
    ON public.collaboration_projection_checkpoints
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
