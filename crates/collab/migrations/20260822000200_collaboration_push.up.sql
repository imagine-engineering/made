CREATE TABLE public.collaboration_push_leases (
    community_id uuid NOT NULL,
    owner_principal_id uuid NOT NULL,
    installation_id text NOT NULL CHECK (
        octet_length(installation_id) BETWEEN 1 AND 64
        AND installation_id = btrim(installation_id)
    ),
    source_event_id bytea NOT NULL CHECK (octet_length(source_event_id) = 32),
    source_created_at numeric(20, 0) NOT NULL CHECK (
        source_created_at BETWEEN 0 AND 18446744073709551615
    ),
    generation numeric(20, 0) NOT NULL CHECK (
        generation BETWEEN 1 AND 9007199254740991
    ),
    active boolean NOT NULL,
    expires_at timestamptz NOT NULL,
    last_active_expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    endpoint_generation numeric(20, 0) CHECK (
        endpoint_generation IS NULL
        OR endpoint_generation BETWEEN 1 AND 9007199254740991
    ),
    capability_reference bytea CHECK (
        capability_reference IS NULL OR octet_length(capability_reference) = 32
    ),
    capability_ciphertext bytea CHECK (
        capability_ciphertext IS NULL
        OR octet_length(capability_ciphertext) BETWEEN 1 AND 16384
    ),
    subscription_policy_ciphertext bytea CHECK (
        subscription_policy_ciphertext IS NULL
        OR octet_length(subscription_policy_ciphertext) BETWEEN 1 AND 1048576
    ),
    custody_key_id text CHECK (
        custody_key_id IS NULL OR octet_length(custody_key_id) BETWEEN 1 AND 128
    ),
    endpoint_enabled boolean NOT NULL,
    endpoint_disabled_at timestamptz,
    accepted_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, owner_principal_id, installation_id),
    UNIQUE (community_id, owner_principal_id, installation_id, generation),
    UNIQUE (community_id, source_event_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, owner_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    CHECK (
        (active
            AND revoked_at IS NULL
            AND endpoint_generation IS NOT NULL
            AND capability_reference IS NOT NULL
            AND capability_ciphertext IS NOT NULL
            AND subscription_policy_ciphertext IS NOT NULL
            AND custody_key_id IS NOT NULL
            AND last_active_expires_at = expires_at)
        OR (NOT active
            AND revoked_at IS NOT NULL
            AND NOT endpoint_enabled
            AND endpoint_generation IS NULL
            AND capability_reference IS NULL
            AND capability_ciphertext IS NULL
            AND subscription_policy_ciphertext IS NULL
            AND custody_key_id IS NULL)
    ),
    CHECK (
        (endpoint_enabled AND endpoint_disabled_at IS NULL)
        OR (NOT endpoint_enabled AND endpoint_disabled_at IS NOT NULL)
    ),
    CHECK (updated_at >= accepted_at)
);

CREATE UNIQUE INDEX collaboration_push_leases_active_capability
    ON public.collaboration_push_leases (
        community_id, owner_principal_id, capability_reference
    )
    WHERE active AND endpoint_enabled;
CREATE INDEX collaboration_push_leases_expiry
    ON public.collaboration_push_leases (
        community_id, active, endpoint_enabled, expires_at
    );

CREATE TABLE public.collaboration_push_wake_jobs (
    community_id uuid NOT NULL,
    wake_id uuid NOT NULL,
    request_id uuid NOT NULL,
    owner_principal_id uuid NOT NULL,
    installation_id text NOT NULL CHECK (
        octet_length(installation_id) BETWEEN 1 AND 64
    ),
    lease_generation numeric(20, 0) NOT NULL CHECK (
        lease_generation BETWEEN 1 AND 9007199254740991
    ),
    endpoint_generation numeric(20, 0) NOT NULL CHECK (
        endpoint_generation BETWEEN 1 AND 9007199254740991
    ),
    capability_reference bytea NOT NULL CHECK (
        octet_length(capability_reference) = 32
    ),
    source_event_id bytea NOT NULL CHECK (octet_length(source_event_id) = 32),
    expires_at timestamptz NOT NULL,
    state text NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'leased', 'delivered', 'failed', 'suppressed')
    ),
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    claim_id uuid,
    claim_expires_at timestamptz,
    terminal_outcome text CHECK (
        terminal_outcome IS NULL
        OR terminal_outcome IN (
            'accepted', 'invalid_endpoint', 'retry_exhausted',
            'lease_unavailable', 'authorization_lost', 'expired'
        )
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at timestamptz,
    PRIMARY KEY (community_id, wake_id),
    UNIQUE (community_id, request_id),
    UNIQUE (community_id, capability_reference, source_event_id),
    FOREIGN KEY (community_id, owner_principal_id, installation_id)
        REFERENCES public.collaboration_push_leases (
            community_id, owner_principal_id, installation_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    CHECK (
        (state = 'leased') = (claim_id IS NOT NULL AND claim_expires_at IS NOT NULL)
    ),
    CHECK (
        state = 'leased' OR (claim_id IS NULL AND claim_expires_at IS NULL)
    ),
    CHECK (
        (state IN ('delivered', 'failed', 'suppressed'))
        = (completed_at IS NOT NULL AND terminal_outcome IS NOT NULL)
    ),
    CHECK (expires_at >= created_at)
);

CREATE INDEX collaboration_push_wake_jobs_due
    ON public.collaboration_push_wake_jobs (
        community_id, available_at, created_at, wake_id
    )
    WHERE state = 'pending';
CREATE INDEX collaboration_push_wake_jobs_recovery
    ON public.collaboration_push_wake_jobs (
        community_id, claim_expires_at, wake_id
    )
    WHERE state = 'leased';

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_push_leases',
        'collaboration_push_wake_jobs'
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
