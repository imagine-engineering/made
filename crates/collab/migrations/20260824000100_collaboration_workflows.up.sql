CREATE TABLE public.collaboration_workflow_definitions (
    community_id uuid NOT NULL,
    workflow_id uuid NOT NULL CHECK (
        workflow_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    creator_principal_id uuid NOT NULL CHECK (
        creator_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    scope_kind text NOT NULL CHECK (scope_kind IN ('community', 'project')),
    project_signer_public_key bytea,
    project_slug text,
    project_record_version numeric(20, 0),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, workflow_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, creator_principal_id)
        REFERENCES public.collaboration_community_memberships (
            community_id, principal_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        project_record_version
    ) REFERENCES public.collaboration_project_groups (
        community_id,
        project_signer_public_key,
        project_slug,
        record_version
    ) ON DELETE RESTRICT,
    CHECK (
        (scope_kind = 'community'
            AND project_signer_public_key IS NULL
            AND project_slug IS NULL
            AND project_record_version IS NULL)
        OR (scope_kind = 'project'
            AND octet_length(project_signer_public_key) = 32
            AND octet_length(project_slug) BETWEEN 1 AND 1024
            AND project_record_version >= 1)
    )
);

CREATE INDEX collaboration_workflow_definitions_project
    ON public.collaboration_workflow_definitions (
        community_id,
        project_signer_public_key,
        project_slug,
        workflow_id
    )
    WHERE scope_kind = 'project';

CREATE TABLE public.collaboration_workflow_definition_versions (
    community_id uuid NOT NULL,
    workflow_id uuid NOT NULL,
    definition_version numeric(20, 0) NOT NULL CHECK (
        definition_version BETWEEN 1 AND 18446744073709551615
    ),
    definition_schema_version integer NOT NULL CHECK (
        definition_schema_version = 1
    ),
    name text NOT NULL CHECK (octet_length(name) BETWEEN 1 AND 256),
    definition jsonb NOT NULL CHECK (
        jsonb_typeof(definition) = 'object'
        AND octet_length(definition::text) <= 65536
    ),
    definition_sha256 bytea NOT NULL CHECK (
        octet_length(definition_sha256) = 32
    ),
    author_principal_id uuid NOT NULL CHECK (
        author_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    source_system text NOT NULL CHECK (
        octet_length(source_system) BETWEEN 1 AND 64
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 512
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 128
    ),
    source_observed_at timestamptz NOT NULL,
    source_integrity_sha256 bytea CHECK (
        source_integrity_sha256 IS NULL
        OR octet_length(source_integrity_sha256) = 32
    ),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, workflow_id, definition_version),
    UNIQUE (community_id, workflow_id, definition_sha256),
    UNIQUE (
        community_id, source_system, source_record_id, source_version
    ),
    FOREIGN KEY (community_id, workflow_id)
        REFERENCES public.collaboration_workflow_definitions (
            community_id, workflow_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, author_principal_id)
        REFERENCES public.collaboration_community_memberships (
            community_id, principal_id
        ) ON DELETE RESTRICT,
    CHECK (created_at >= source_observed_at)
);

CREATE INDEX collaboration_workflow_definition_versions_source
    ON public.collaboration_workflow_definition_versions (
        community_id, source_system, source_record_id, definition_version
    );

CREATE TABLE public.collaboration_workflow_definition_heads (
    community_id uuid NOT NULL,
    workflow_id uuid NOT NULL,
    current_definition_version numeric(20, 0) NOT NULL CHECK (
        current_definition_version BETWEEN 1 AND 18446744073709551615
    ),
    head_revision numeric(20, 0) NOT NULL CHECK (
        head_revision BETWEEN 1 AND 18446744073709551615
    ),
    lifecycle_state text NOT NULL CHECK (
        lifecycle_state IN ('draft', 'active', 'disabled', 'archived')
    ),
    source_system text NOT NULL CHECK (
        octet_length(source_system) BETWEEN 1 AND 64
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 512
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 128
    ),
    source_observed_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, workflow_id),
    FOREIGN KEY (community_id, workflow_id, current_definition_version)
        REFERENCES public.collaboration_workflow_definition_versions (
            community_id, workflow_id, definition_version
        ) ON DELETE RESTRICT,
    CHECK (updated_at >= source_observed_at)
);

CREATE INDEX collaboration_workflow_definition_heads_active
    ON public.collaboration_workflow_definition_heads (
        community_id, lifecycle_state, updated_at, workflow_id
    )
    WHERE lifecycle_state = 'active';

CREATE TABLE public.collaboration_workflow_runs (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL CHECK (
        run_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    workflow_id uuid NOT NULL,
    definition_version numeric(20, 0) NOT NULL CHECK (
        definition_version BETWEEN 1 AND 18446744073709551615
    ),
    trigger_operation_id uuid NOT NULL CHECK (
        trigger_operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    trigger_kind text NOT NULL CHECK (
        trigger_kind IN ('event', 'schedule', 'webhook', 'manual')
    ),
    trigger_source_id text NOT NULL CHECK (
        octet_length(trigger_source_id) BETWEEN 1 AND 512
    ),
    trigger_context jsonb NOT NULL CHECK (
        jsonb_typeof(trigger_context) = 'object'
        AND octet_length(trigger_context::text) <= 1048576
    ),
    run_version numeric(20, 0) NOT NULL CHECK (
        run_version BETWEEN 1 AND 18446744073709551615
    ),
    status text NOT NULL CHECK (
        status IN (
            'claimed', 'queued', 'running', 'waiting_approval',
            'retry_scheduled', 'repair_required', 'completed',
            'failed', 'cancelled'
        )
    ),
    current_step_index smallint NOT NULL CHECK (
        current_step_index BETWEEN 0 AND 64
    ),
    error_code text CHECK (
        error_code IS NULL OR octet_length(error_code) BETWEEN 1 AND 64
    ),
    error_message text CHECK (
        error_message IS NULL OR octet_length(error_message) <= 4096
    ),
    source_system text NOT NULL CHECK (
        octet_length(source_system) BETWEEN 1 AND 64
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 512
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 128
    ),
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, run_id),
    UNIQUE (community_id, trigger_operation_id),
    UNIQUE (community_id, run_id, workflow_id, definition_version),
    FOREIGN KEY (community_id, workflow_id, definition_version)
        REFERENCES public.collaboration_workflow_definition_versions (
            community_id, workflow_id, definition_version
        ) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at),
    CHECK (started_at IS NULL OR started_at >= created_at),
    CHECK (completed_at IS NULL OR completed_at >= created_at),
    CHECK (
        (status IN ('completed', 'failed', 'cancelled')
            AND completed_at IS NOT NULL)
        OR (status NOT IN ('completed', 'failed', 'cancelled')
            AND completed_at IS NULL)
    ),
    CHECK (
        (status IN ('failed', 'repair_required')
            AND error_code IS NOT NULL)
        OR status NOT IN ('failed', 'repair_required')
    )
);

CREATE INDEX collaboration_workflow_runs_ready
    ON public.collaboration_workflow_runs (
        community_id, status, updated_at, run_id
    )
    WHERE status IN ('queued', 'running', 'retry_scheduled');
CREATE INDEX collaboration_workflow_runs_definition
    ON public.collaboration_workflow_runs (
        community_id, workflow_id, definition_version, created_at DESC, run_id
    );

CREATE TABLE public.collaboration_workflow_steps (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    workflow_id uuid NOT NULL,
    definition_version numeric(20, 0) NOT NULL,
    step_index smallint NOT NULL CHECK (step_index BETWEEN 0 AND 63),
    step_id text NOT NULL CHECK (
        octet_length(step_id) BETWEEN 1 AND 64
        AND step_id ~ '^[A-Za-z0-9][A-Za-z0-9_]{0,63}$'
    ),
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    state text NOT NULL CHECK (
        state IN (
            'pending', 'running', 'waiting_approval', 'retry_scheduled',
            'repair_required', 'completed', 'skipped', 'failed', 'cancelled'
        )
    ),
    attempt_count smallint NOT NULL CHECK (attempt_count BETWEEN 0 AND 8),
    output jsonb CHECK (
        output IS NULL OR octet_length(output::text) <= 65536
    ),
    error_code text CHECK (
        error_code IS NULL OR octet_length(error_code) BETWEEN 1 AND 64
    ),
    error_message text CHECK (
        error_message IS NULL OR octet_length(error_message) <= 4096
    ),
    source_system text NOT NULL CHECK (
        octet_length(source_system) BETWEEN 1 AND 64
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 512
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 128
    ),
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, run_id, step_index),
    UNIQUE (community_id, run_id, step_id),
    UNIQUE (community_id, operation_id),
    FOREIGN KEY (
        community_id, run_id, workflow_id, definition_version
    ) REFERENCES public.collaboration_workflow_runs (
        community_id, run_id, workflow_id, definition_version
    ) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at),
    CHECK (started_at IS NULL OR started_at >= created_at),
    CHECK (completed_at IS NULL OR completed_at >= created_at),
    CHECK (
        (state IN ('completed', 'skipped', 'failed', 'cancelled')
            AND completed_at IS NOT NULL)
        OR (state NOT IN ('completed', 'skipped', 'failed', 'cancelled')
            AND completed_at IS NULL)
    ),
    CHECK (
        (state IN ('failed', 'repair_required') AND error_code IS NOT NULL)
        OR state NOT IN ('failed', 'repair_required')
    )
);

CREATE INDEX collaboration_workflow_steps_state
    ON public.collaboration_workflow_steps (
        community_id, state, updated_at, run_id, step_index
    )
    WHERE state IN ('running', 'waiting_approval', 'retry_scheduled', 'repair_required');

CREATE TABLE public.collaboration_workflow_retries (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_index smallint NOT NULL,
    attempt_number smallint NOT NULL CHECK (attempt_number BETWEEN 2 AND 8),
    retry_operation_id uuid NOT NULL CHECK (
        retry_operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    failure_class text NOT NULL CHECK (
        failure_class IN (
            'rate_limited', 'temporary_unavailable', 'timeout', 'transport'
        )
    ),
    state text NOT NULL CHECK (
        state IN ('scheduled', 'claimed', 'completed', 'exhausted', 'cancelled')
    ),
    scheduled_at timestamptz NOT NULL,
    due_at timestamptz NOT NULL,
    claimed_at timestamptz,
    completed_at timestamptz,
    source_system text NOT NULL CHECK (
        octet_length(source_system) BETWEEN 1 AND 64
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 512
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 128
    ),
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, run_id, step_index, attempt_number),
    UNIQUE (community_id, retry_operation_id),
    FOREIGN KEY (community_id, run_id, step_index)
        REFERENCES public.collaboration_workflow_steps (
            community_id, run_id, step_index
        ) ON DELETE RESTRICT,
    CHECK (due_at > scheduled_at),
    CHECK (created_at >= source_observed_at),
    CHECK (claimed_at IS NULL OR claimed_at >= scheduled_at),
    CHECK (completed_at IS NULL OR completed_at >= scheduled_at),
    CHECK (
        (state = 'scheduled' AND claimed_at IS NULL AND completed_at IS NULL)
        OR (state = 'claimed' AND claimed_at IS NOT NULL AND completed_at IS NULL)
        OR (state IN ('completed', 'exhausted', 'cancelled')
            AND completed_at IS NOT NULL)
    )
);

CREATE INDEX collaboration_workflow_retries_due
    ON public.collaboration_workflow_retries (
        community_id, due_at, run_id, step_index, attempt_number
    )
    WHERE state = 'scheduled';

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_workflow_definitions',
        'collaboration_workflow_definition_versions',
        'collaboration_workflow_definition_heads',
        'collaboration_workflow_runs',
        'collaboration_workflow_steps',
        'collaboration_workflow_retries'
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
