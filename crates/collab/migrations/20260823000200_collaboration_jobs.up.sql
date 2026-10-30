CREATE TABLE public.collaboration_jobs (
    community_id uuid NOT NULL,
    job_id uuid NOT NULL CHECK (
        job_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    requester_principal_id uuid NOT NULL CHECK (
        requester_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    target_executor_principal_id uuid NOT NULL CHECK (
        target_executor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    current_executor_principal_id uuid CHECK (
        current_executor_principal_id IS NULL
        OR current_executor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    current_version numeric(20, 0) NOT NULL CHECK (
        current_version BETWEEN 1 AND 18446744073709551615
    ),
    current_state text NOT NULL CHECK (
        current_state IN (
            'requested', 'accepted', 'in_progress',
            'completed', 'cancelled', 'failed'
        )
    ),
    requested_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, job_id),
    UNIQUE (community_id, job_id, current_version),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    CHECK (
        (current_state = 'requested' AND current_executor_principal_id IS NULL)
        OR (current_state IN ('accepted', 'in_progress', 'completed')
            AND current_executor_principal_id = target_executor_principal_id)
        OR current_state IN ('cancelled', 'failed')
    ),
    CHECK (updated_at >= requested_at)
);

CREATE INDEX collaboration_jobs_requester
    ON public.collaboration_jobs (
        community_id, requester_principal_id, updated_at DESC, job_id
    );
CREATE INDEX collaboration_jobs_executor
    ON public.collaboration_jobs (
        community_id, target_executor_principal_id, current_state, updated_at, job_id
    )
    WHERE current_state IN ('requested', 'accepted', 'in_progress');

CREATE TABLE public.collaboration_job_versions (
    community_id uuid NOT NULL,
    job_id uuid NOT NULL,
    version numeric(20, 0) NOT NULL CHECK (
        version BETWEEN 1 AND 18446744073709551615
    ),
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    command_type text NOT NULL CHECK (
        command_type IN ('request', 'accept', 'progress', 'result', 'cancel', 'error')
    ),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    executor_principal_id uuid CHECK (
        executor_principal_id IS NULL
        OR executor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    occurred_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, job_id, version),
    UNIQUE (community_id, operation_id),
    FOREIGN KEY (community_id, job_id)
        REFERENCES public.collaboration_jobs (community_id, job_id) ON DELETE RESTRICT,
    CHECK (
        (command_type IN ('accept', 'progress', 'result')
            AND executor_principal_id IS NOT NULL)
        OR (command_type IN ('request', 'cancel', 'error')
            AND executor_principal_id IS NULL)
    ),
    CHECK (recorded_at >= occurred_at)
);

CREATE INDEX collaboration_job_versions_operation
    ON public.collaboration_job_versions (community_id, operation_id, job_id, version);

CREATE TABLE public.collaboration_job_ancestry (
    community_id uuid NOT NULL,
    ancestor_job_id uuid NOT NULL,
    descendant_job_id uuid NOT NULL,
    depth smallint NOT NULL CHECK (depth BETWEEN 1 AND 8),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, ancestor_job_id, descendant_job_id),
    UNIQUE (community_id, descendant_job_id, depth),
    FOREIGN KEY (community_id, ancestor_job_id)
        REFERENCES public.collaboration_jobs (community_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, descendant_job_id)
        REFERENCES public.collaboration_jobs (community_id, job_id) ON DELETE RESTRICT,
    CHECK (ancestor_job_id <> descendant_job_id)
);

CREATE INDEX collaboration_job_ancestry_descendants
    ON public.collaboration_job_ancestry (
        community_id, descendant_job_id, depth DESC, ancestor_job_id
    );
CREATE INDEX collaboration_job_ancestry_direct_children
    ON public.collaboration_job_ancestry (
        community_id, ancestor_job_id, descendant_job_id
    )
    WHERE depth = 1;

CREATE TABLE public.collaboration_job_executor_leases (
    community_id uuid NOT NULL,
    job_id uuid NOT NULL,
    job_version numeric(20, 0) NOT NULL CHECK (
        job_version BETWEEN 1 AND 18446744073709551615
    ),
    lease_generation numeric(20, 0) NOT NULL CHECK (
        lease_generation BETWEEN 1 AND 18446744073709551615
    ),
    lease_id uuid NOT NULL CHECK (
        lease_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    executor_principal_id uuid NOT NULL CHECK (
        executor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    state text NOT NULL CHECK (state IN ('active', 'released')),
    acquired_at timestamptz NOT NULL,
    last_heartbeat_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    recovery_after timestamptz NOT NULL,
    released_at timestamptz,
    release_reason text CHECK (
        release_reason IS NULL
        OR release_reason IN ('completed', 'cancelled', 'failed', 'expired', 'replaced')
    ),
    PRIMARY KEY (community_id, job_id, lease_generation),
    UNIQUE (community_id, lease_id),
    FOREIGN KEY (community_id, job_id, job_version)
        REFERENCES public.collaboration_job_versions (community_id, job_id, version)
        ON DELETE RESTRICT,
    CHECK (
        (state = 'active' AND released_at IS NULL AND release_reason IS NULL)
        OR (state = 'released' AND released_at IS NOT NULL AND release_reason IS NOT NULL)
    ),
    CHECK (last_heartbeat_at >= acquired_at),
    CHECK (expires_at >= last_heartbeat_at),
    CHECK (recovery_after >= expires_at),
    CHECK (released_at IS NULL OR released_at >= acquired_at)
);

CREATE UNIQUE INDEX collaboration_job_executor_leases_one_active
    ON public.collaboration_job_executor_leases (community_id, job_id)
    WHERE state = 'active';
CREATE INDEX collaboration_job_executor_leases_recovery
    ON public.collaboration_job_executor_leases (
        community_id, recovery_after, job_id, lease_generation
    )
    WHERE state = 'active';
CREATE INDEX collaboration_job_executor_leases_executor
    ON public.collaboration_job_executor_leases (
        community_id, executor_principal_id, expires_at, job_id
    )
    WHERE state = 'active';

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_jobs',
        'collaboration_job_versions',
        'collaboration_job_ancestry',
        'collaboration_job_executor_leases'
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
