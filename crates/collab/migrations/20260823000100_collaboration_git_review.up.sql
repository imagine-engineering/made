CREATE TABLE public.collaboration_git_review_projections (
    community_id uuid NOT NULL,
    review_id uuid NOT NULL CHECK (
        review_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    repository_id uuid NOT NULL,
    current_revision numeric(20, 0) NOT NULL CHECK (current_revision >= 1),
    current_head_commit text NOT NULL CHECK (
        octet_length(current_head_commit) IN (40, 64)
        AND current_head_commit ~ '^[0-9a-f]+$'
    ),
    aggregate_version numeric(20, 0) NOT NULL CHECK (aggregate_version >= 1),
    projection_generation numeric(20, 0) NOT NULL CHECK (projection_generation >= 1),
    ci_suite_count integer NOT NULL CHECK (ci_suite_count BETWEEN 0 AND 1000),
    review_payload jsonb NOT NULL CHECK (
        jsonb_typeof(review_payload) = 'object'
        AND pg_column_size(review_payload) <= 33554432
    ),
    review_hash bytea NOT NULL CHECK (octet_length(review_hash) = 32),
    projection_hash bytea NOT NULL CHECK (octet_length(projection_hash) = 32),
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, review_id),
    FOREIGN KEY (community_id, repository_id)
        REFERENCES public.collaboration_hosted_repositories (community_id, repository_id)
        ON DELETE RESTRICT,
    CHECK (
        (integrity_algorithm IS NULL AND integrity_value IS NULL)
        OR (integrity_algorithm IS NOT NULL AND integrity_value IS NOT NULL)
    ),
    CHECK (updated_at >= created_at)
);

CREATE INDEX collaboration_git_review_projections_repository
    ON public.collaboration_git_review_projections (
        community_id, repository_id, current_head_commit, review_id
    );

CREATE TABLE public.collaboration_git_ci_projections (
    community_id uuid NOT NULL,
    review_id uuid NOT NULL,
    suite_id uuid NOT NULL CHECK (
        suite_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    repository_id uuid NOT NULL,
    revision numeric(20, 0) NOT NULL CHECK (revision >= 1),
    head_commit text NOT NULL CHECK (
        octet_length(head_commit) IN (40, 64)
        AND head_commit ~ '^[0-9a-f]+$'
    ),
    suite_status text NOT NULL CHECK (
        suite_status IN ('pending', 'running', 'success', 'failure', 'cancelled')
    ),
    aggregate_version numeric(20, 0) NOT NULL CHECK (aggregate_version >= 1),
    projection_generation numeric(20, 0) NOT NULL CHECK (projection_generation >= 1),
    suite_payload jsonb NOT NULL CHECK (
        jsonb_typeof(suite_payload) = 'object'
        AND pg_column_size(suite_payload) <= 33554432
    ),
    suite_hash bytea NOT NULL CHECK (octet_length(suite_hash) = 32),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, review_id, suite_id),
    FOREIGN KEY (community_id, review_id)
        REFERENCES public.collaboration_git_review_projections (community_id, review_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, repository_id)
        REFERENCES public.collaboration_hosted_repositories (community_id, repository_id)
        ON DELETE RESTRICT,
    CHECK (updated_at >= created_at)
);

CREATE INDEX collaboration_git_ci_projections_commit
    ON public.collaboration_git_ci_projections (
        community_id, repository_id, head_commit, suite_status, suite_id
    );

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_git_review_projections',
        'collaboration_git_ci_projections'
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
