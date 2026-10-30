CREATE TABLE public.collaboration_audit_entries (
    community_id uuid NOT NULL,
    sequence numeric(20, 0) NOT NULL CHECK (
        sequence BETWEEN 1 AND 18446744073709551615
    ),
    entry_hash bytea NOT NULL CHECK (octet_length(entry_hash) = 32),
    previous_hash bytea CHECK (
        previous_hash IS NULL OR octet_length(previous_hash) = 32
    ),
    previous_sequence numeric(20, 0) GENERATED ALWAYS AS (
        CASE WHEN previous_hash IS NULL THEN NULL ELSE sequence - 1 END
    ) STORED,
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    action text NOT NULL CHECK (
        octet_length(action) BETWEEN 1 AND 128
        AND action ~ '^[a-z0-9_.-]+$'
    ),
    actor_principal_id uuid CHECK (
        actor_principal_id IS NULL
        OR actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    outcome text NOT NULL CHECK (
        outcome IN ('succeeded', 'failed', 'denied', 'cancelled')
    ),
    occurred_at_millis numeric(20, 0) NOT NULL CHECK (
        occurred_at_millis BETWEEN 1 AND 18446744073709551615
    ),
    fields jsonb NOT NULL CHECK (
        jsonb_typeof(fields) = 'array'
        AND jsonb_array_length(fields) <= 32
        AND octet_length(fields::text) <= 16384
    ),
    bridge_source text CHECK (bridge_source IN ('buzz_v1')),
    bridge_source_sequence numeric(20, 0) CHECK (
        bridge_source_sequence BETWEEN 1 AND 18446744073709551614
    ),
    bridge_source_head bytea CHECK (
        bridge_source_head IS NULL OR octet_length(bridge_source_head) = 32
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, sequence),
    UNIQUE (community_id, entry_hash),
    UNIQUE (community_id, sequence, entry_hash),
    UNIQUE (community_id, operation_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, previous_sequence, previous_hash)
        REFERENCES public.collaboration_audit_entries (
            community_id, sequence, entry_hash
        ) ON DELETE RESTRICT,
    CHECK (
        (sequence = 1
            AND previous_hash IS NULL
            AND bridge_source IS NULL
            AND bridge_source_sequence IS NULL
            AND bridge_source_head IS NULL)
        OR (sequence > 1
            AND previous_hash IS NOT NULL
            AND bridge_source IS NULL
            AND bridge_source_sequence IS NULL
            AND bridge_source_head IS NULL)
        OR (previous_hash IS NULL
            AND bridge_source = 'buzz_v1'
            AND bridge_source_sequence IS NOT NULL
            AND bridge_source_head IS NOT NULL
            AND sequence = bridge_source_sequence + 1)
    )
);

CREATE FUNCTION public.reject_collaboration_audit_entry_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'collaboration audit entries are immutable'
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_audit_entries_immutable
    BEFORE UPDATE OR DELETE ON public.collaboration_audit_entries
    FOR EACH ROW EXECUTE FUNCTION public.reject_collaboration_audit_entry_mutation();

CREATE TABLE public.collaboration_audit_heads (
    community_id uuid PRIMARY KEY,
    sequence numeric(20, 0) NOT NULL CHECK (
        sequence BETWEEN 1 AND 18446744073709551615
    ),
    entry_hash bytea NOT NULL CHECK (octet_length(entry_hash) = 32),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (community_id, sequence, entry_hash)
        REFERENCES public.collaboration_audit_entries (
            community_id, sequence, entry_hash
        ) ON DELETE RESTRICT
);

CREATE FUNCTION public.guard_collaboration_audit_head_mutation() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'collaboration audit heads cannot be deleted'
            USING ERRCODE = 'check_violation';
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF EXISTS (
            SELECT 1
            FROM public.collaboration_audit_entries AS entry
            WHERE entry.community_id = NEW.community_id
              AND entry.sequence > NEW.sequence
        ) THEN
            RAISE EXCEPTION 'collaboration audit head is stale'
                USING ERRCODE = 'serialization_failure';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.community_id IS DISTINCT FROM OLD.community_id THEN
        RAISE EXCEPTION 'collaboration audit head identity is immutable'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.sequence <> OLD.sequence + 1 THEN
        RAISE EXCEPTION 'collaboration audit head sequence conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.collaboration_audit_entries AS entry
        WHERE entry.community_id = NEW.community_id
          AND entry.sequence = NEW.sequence
          AND entry.entry_hash = NEW.entry_hash
          AND entry.previous_hash = OLD.entry_hash
          AND entry.bridge_source IS NULL
    ) THEN
        RAISE EXCEPTION 'collaboration audit head predecessor conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;

    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_audit_heads_guard
    BEFORE INSERT OR UPDATE OR DELETE ON public.collaboration_audit_heads
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_audit_head_mutation();

CREATE TABLE public.collaboration_audit_export_cursors (
    community_id uuid NOT NULL,
    exporter_id text NOT NULL CHECK (
        octet_length(exporter_id) BETWEEN 1 AND 128
        AND exporter_id ~ '^[a-z0-9_.-]+$'
    ),
    cursor_version numeric(20, 0) NOT NULL CHECK (
        cursor_version BETWEEN 1 AND 18446744073709551615
    ),
    exported_through_sequence numeric(20, 0) NOT NULL CHECK (
        exported_through_sequence BETWEEN 1 AND 18446744073709551615
    ),
    exported_through_hash bytea NOT NULL CHECK (
        octet_length(exported_through_hash) = 32
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, exporter_id),
    FOREIGN KEY (community_id, exported_through_sequence, exported_through_hash)
        REFERENCES public.collaboration_audit_entries (
            community_id, sequence, entry_hash
        ) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at)
);

CREATE FUNCTION public.guard_collaboration_audit_export_cursor_mutation() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'collaboration audit export cursors cannot be deleted'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.community_id IS DISTINCT FROM OLD.community_id
       OR NEW.exporter_id IS DISTINCT FROM OLD.exporter_id THEN
        RAISE EXCEPTION 'collaboration audit export cursor identity is immutable'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.cursor_version <> OLD.cursor_version + 1 THEN
        RAISE EXCEPTION 'collaboration audit export cursor version conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;
    IF NEW.exported_through_sequence <= OLD.exported_through_sequence THEN
        RAISE EXCEPTION 'collaboration audit export cursor cannot regress'
            USING ERRCODE = 'check_violation';
    END IF;

    NEW.created_at = OLD.created_at;
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_audit_export_cursors_guard
    BEFORE UPDATE OR DELETE ON public.collaboration_audit_export_cursors
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_audit_export_cursor_mutation();

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_audit_entries',
        'collaboration_audit_heads',
        'collaboration_audit_export_cursors'
    ] LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format(
            'CREATE POLICY %I ON public.%I AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true)',
            table_name || '_admission',
            table_name
        );
        EXECUTE format(
            'CREATE POLICY %I ON public.%I AS RESTRICTIVE FOR ALL USING (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid) WITH CHECK (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid)',
            table_name || '_community',
            table_name
        );
    END LOOP;
END;
$$;
