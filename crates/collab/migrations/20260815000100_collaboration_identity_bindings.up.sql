CREATE TABLE public.collaboration_identity_bindings (
    community_id uuid NOT NULL,
    binding_id uuid NOT NULL,
    version bigint NOT NULL CHECK (version > 0),
    is_current boolean NOT NULL,
    service_account_id bigint NOT NULL CHECK (service_account_id >= 0),
    profile_id uuid NOT NULL,
    nostr_public_key bytea NOT NULL CHECK (octet_length(nostr_public_key) = 32),
    status text NOT NULL CHECK (
        status IN ('pending', 'verified', 'active', 'rotated', 'revoked', 'archived')
    ),
    verification_method text CHECK (
        verification_method IS NULL OR verification_method IN (
            'generated_key_challenge',
            'existing_key_challenge',
            'imported_key_challenge',
            'paired_key_challenge',
            'restored_key_challenge',
            'migrated_evidence'
        )
    ),
    evidence_reference text CHECK (
        evidence_reference IS NULL OR octet_length(evidence_reference) BETWEEN 1 AND 1024
    ),
    verified_at timestamptz,
    predecessor_binding_id uuid,
    predecessor_version bigint CHECK (predecessor_version > 0),
    successor_binding_id uuid,
    successor_version bigint CHECK (successor_version > 0),
    created_at timestamptz NOT NULL,
    activated_at timestamptz,
    terminal_at timestamptz,
    organization_policy_version bigint NOT NULL CHECK (organization_policy_version > 0),
    actor_principal_id uuid NOT NULL,
    audit_reference uuid NOT NULL,
    PRIMARY KEY (community_id, binding_id, version),
    CHECK (
        (predecessor_binding_id IS NULL) = (predecessor_version IS NULL)
        AND (successor_binding_id IS NULL) = (successor_version IS NULL)
    ),
    CHECK (predecessor_binding_id IS DISTINCT FROM binding_id),
    CHECK (successor_binding_id IS DISTINCT FROM binding_id),
    CHECK (
        (status = 'pending' AND verification_method IS NULL AND evidence_reference IS NULL
            AND verified_at IS NULL AND activated_at IS NULL AND terminal_at IS NULL)
        OR
        (status <> 'pending' AND verification_method IS NOT NULL
            AND evidence_reference IS NOT NULL AND verified_at IS NOT NULL)
    ),
    CHECK (
        verification_method IS DISTINCT FROM 'migrated_evidence'
        OR status IN ('rotated', 'revoked', 'archived')
    ),
    CHECK (verified_at IS NULL OR verified_at >= created_at),
    CHECK (activated_at IS NULL OR activated_at >= verified_at),
    CHECK (
        (status = 'verified' AND activated_at IS NULL AND terminal_at IS NULL)
        OR (status = 'active' AND activated_at IS NOT NULL AND terminal_at IS NULL)
        OR (status IN ('rotated', 'archived') AND activated_at IS NOT NULL
            AND terminal_at IS NOT NULL AND terminal_at >= activated_at)
        OR (status = 'revoked' AND terminal_at IS NOT NULL
            AND terminal_at >= COALESCE(activated_at, verified_at))
        OR status = 'pending'
    ),
    CHECK (status <> 'rotated' OR successor_binding_id IS NOT NULL),
    FOREIGN KEY (community_id, predecessor_binding_id, predecessor_version)
        REFERENCES public.collaboration_identity_bindings (community_id, binding_id, version)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (community_id, successor_binding_id, successor_version)
        REFERENCES public.collaboration_identity_bindings (community_id, binding_id, version)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX collaboration_identity_bindings_current
    ON public.collaboration_identity_bindings (community_id, binding_id)
    WHERE is_current;

CREATE UNIQUE INDEX collaboration_identity_bindings_active_profile
    ON public.collaboration_identity_bindings (community_id, service_account_id, profile_id)
    WHERE is_current AND status = 'active';

CREATE INDEX collaboration_identity_bindings_public_key
    ON public.collaboration_identity_bindings (community_id, nostr_public_key)
    WHERE is_current;

CREATE INDEX collaboration_identity_bindings_account
    ON public.collaboration_identity_bindings (community_id, service_account_id, status)
    WHERE is_current;

ALTER TABLE public.collaboration_identity_bindings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_identity_bindings FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_identity_bindings_admission
    ON public.collaboration_identity_bindings
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_identity_bindings_community
    ON public.collaboration_identity_bindings
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
