CREATE TABLE public.collaboration_workflow_run_leases (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    run_version numeric(20, 0) NOT NULL CHECK (
        run_version BETWEEN 1 AND 18446744073709551615
    ),
    lease_generation numeric(20, 0) NOT NULL CHECK (
        lease_generation BETWEEN 1 AND 18446744073709551615
    ),
    lease_id uuid NOT NULL CHECK (
        lease_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    worker_id text NOT NULL CHECK (
        octet_length(worker_id) BETWEEN 1 AND 128
    ),
    state text NOT NULL CHECK (state IN ('active', 'released')),
    acquired_at timestamptz NOT NULL,
    last_heartbeat_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    recovery_after timestamptz NOT NULL,
    released_at timestamptz,
    release_reason text CHECK (
        release_reason IS NULL OR release_reason IN (
            'completed', 'cancelled', 'failed', 'expired', 'replaced'
        )
    ),
    PRIMARY KEY (community_id, run_id, lease_generation),
    UNIQUE (community_id, lease_id),
    FOREIGN KEY (community_id, run_id)
        REFERENCES public.collaboration_workflow_runs (
            community_id, run_id
        ) ON DELETE RESTRICT,
    CHECK (last_heartbeat_at >= acquired_at),
    CHECK (expires_at >= last_heartbeat_at),
    CHECK (recovery_after >= expires_at),
    CHECK (
        (state = 'active' AND released_at IS NULL AND release_reason IS NULL)
        OR (state = 'released'
            AND released_at IS NOT NULL
            AND release_reason IS NOT NULL
            AND released_at >= acquired_at)
    )
);

CREATE UNIQUE INDEX collaboration_workflow_run_leases_one_active
    ON public.collaboration_workflow_run_leases (community_id, run_id)
    WHERE state = 'active';
CREATE INDEX collaboration_workflow_run_leases_recovery
    ON public.collaboration_workflow_run_leases (
        community_id, recovery_after, run_id, lease_generation
    )
    WHERE state = 'active';

ALTER TABLE public.collaboration_workflow_run_leases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_run_leases FORCE ROW LEVEL SECURITY;
CREATE POLICY collaboration_workflow_run_leases_admission
    ON public.collaboration_workflow_run_leases
    AS PERMISSIVE FOR ALL
    USING (true)
    WITH CHECK (true);
CREATE POLICY collaboration_workflow_run_leases_community
    ON public.collaboration_workflow_run_leases
    AS RESTRICTIVE FOR ALL
    USING (
        community_id = NULLIF(
            current_setting('app.community_id', true), ''
        )::uuid
    )
    WITH CHECK (
        community_id = NULLIF(
            current_setting('app.community_id', true), ''
        )::uuid
    );
