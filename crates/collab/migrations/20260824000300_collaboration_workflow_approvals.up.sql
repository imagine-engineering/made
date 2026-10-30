ALTER TABLE public.collaboration_workflow_steps
    ADD CONSTRAINT collaboration_workflow_steps_operation_identity
    UNIQUE (community_id, run_id, step_index, operation_id);

CREATE TABLE public.collaboration_workflow_approvals (
    community_id uuid NOT NULL,
    approval_id uuid NOT NULL CHECK (
        approval_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    run_id uuid NOT NULL,
    workflow_id uuid NOT NULL,
    definition_version numeric(20, 0) NOT NULL CHECK (
        definition_version BETWEEN 1 AND 18446744073709551615
    ),
    workflow_creator_principal_id uuid NOT NULL,
    step_index smallint NOT NULL CHECK (step_index BETWEEN 0 AND 63),
    step_operation_id uuid NOT NULL CHECK (
        step_operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    capability_sha256 bytea NOT NULL CHECK (octet_length(capability_sha256) = 32),
    eligibility_kind text NOT NULL CHECK (
        eligibility_kind IN ('any_member', 'owner', 'administrator', 'principal')
    ),
    eligible_principal_id uuid,
    request_message text NOT NULL CHECK (
        octet_length(request_message) BETWEEN 1 AND 16384
    ),
    state text NOT NULL CHECK (
        state IN ('pending', 'granted', 'denied', 'expired', 'cancelled')
    ),
    decision_operation_id uuid,
    decided_by_principal_id uuid,
    decision_note text CHECK (
        decision_note IS NULL OR octet_length(decision_note) <= 4096
    ),
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    decided_at timestamptz,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, approval_id),
    UNIQUE (community_id, capability_sha256),
    UNIQUE (community_id, run_id, step_index),
    UNIQUE (community_id, decision_operation_id),
    FOREIGN KEY (
        community_id, run_id, workflow_id, definition_version
    ) REFERENCES public.collaboration_workflow_runs (
        community_id, run_id, workflow_id, definition_version
    ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, run_id, step_index, step_operation_id)
        REFERENCES public.collaboration_workflow_steps (
            community_id, run_id, step_index, operation_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, eligible_principal_id)
        REFERENCES public.collaboration_community_memberships (
            community_id, principal_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, workflow_creator_principal_id)
        REFERENCES public.collaboration_community_memberships (
            community_id, principal_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, decided_by_principal_id)
        REFERENCES public.collaboration_community_memberships (
            community_id, principal_id
        ) ON DELETE RESTRICT,
    CHECK (
        (eligibility_kind = 'principal' AND eligible_principal_id IS NOT NULL)
        OR (eligibility_kind <> 'principal' AND eligible_principal_id IS NULL)
    ),
    CHECK (expires_at > created_at),
    CHECK (updated_at >= created_at),
    CHECK (
        (state = 'pending'
            AND decision_operation_id IS NULL
            AND decided_by_principal_id IS NULL
            AND decided_at IS NULL)
        OR (state <> 'pending'
            AND decision_operation_id IS NOT NULL
            AND decided_at IS NOT NULL)
    )
);

CREATE INDEX collaboration_workflow_approvals_pending
    ON public.collaboration_workflow_approvals (
        community_id, expires_at, run_id, step_index
    )
    WHERE state = 'pending';

CREATE TABLE public.collaboration_workflow_approval_outbox (
    community_id uuid NOT NULL,
    outbox_id uuid NOT NULL CHECK (
        outbox_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    approval_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_index smallint NOT NULL CHECK (step_index BETWEEN 0 AND 63),
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    intent_kind text NOT NULL CHECK (
        intent_kind IN ('notify', 'resume', 'cancel')
    ),
    state text NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'claimed', 'completed', 'failed')
    ),
    attempt_count smallint NOT NULL DEFAULT 0 CHECK (
        attempt_count BETWEEN 0 AND 32
    ),
    available_at timestamptz NOT NULL,
    claimed_at timestamptz,
    completed_at timestamptz,
    last_error text CHECK (
        last_error IS NULL OR octet_length(last_error) <= 2048
    ),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, outbox_id),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, approval_id, intent_kind),
    FOREIGN KEY (community_id, approval_id)
        REFERENCES public.collaboration_workflow_approvals (
            community_id, approval_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, run_id, step_index)
        REFERENCES public.collaboration_workflow_steps (
            community_id, run_id, step_index
        ) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at),
    CHECK (claimed_at IS NULL OR claimed_at >= created_at),
    CHECK (completed_at IS NULL OR completed_at >= created_at),
    CHECK (
        (state = 'pending' AND claimed_at IS NULL AND completed_at IS NULL)
        OR (state = 'claimed' AND claimed_at IS NOT NULL AND completed_at IS NULL)
        OR (state IN ('completed', 'failed') AND completed_at IS NOT NULL)
    )
);

CREATE INDEX collaboration_workflow_approval_outbox_ready
    ON public.collaboration_workflow_approval_outbox (
        community_id, state, available_at, created_at, outbox_id
    )
    WHERE state = 'pending';

ALTER TABLE public.collaboration_workflow_approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_approvals FORCE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_approval_outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_approval_outbox FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_workflow_approvals_admission
    ON public.collaboration_workflow_approvals
    AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY collaboration_workflow_approvals_community
    ON public.collaboration_workflow_approvals
    AS RESTRICTIVE FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );

CREATE POLICY collaboration_workflow_approval_outbox_admission
    ON public.collaboration_workflow_approval_outbox
    AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY collaboration_workflow_approval_outbox_community
    ON public.collaboration_workflow_approval_outbox
    AS RESTRICTIVE FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
