CREATE TABLE public.collaboration_identity_profiles (
    community_id uuid NOT NULL,
    profile_id uuid NOT NULL CHECK (profile_id <> '00000000-0000-0000-0000-000000000000'),
    principal_id uuid NOT NULL CHECK (principal_id <> '00000000-0000-0000-0000-000000000000'),
    author_public_key bytea NOT NULL CHECK (octet_length(author_public_key) = 32),
    profile_kind text NOT NULL CHECK (profile_kind IN ('human', 'agent')),
    owner_public_key bytea CHECK (owner_public_key IS NULL OR octet_length(owner_public_key) = 32),
    first_owner_proof_event_id bytea CHECK (
        first_owner_proof_event_id IS NULL OR octet_length(first_owner_proof_event_id) = 32
    ),
    source_event_id bytea NOT NULL CHECK (octet_length(source_event_id) = 32),
    aggregate_version numeric(20, 0) NOT NULL CHECK (
        aggregate_version BETWEEN 1 AND 18446744073709551615
    ),
    profile_payload jsonb NOT NULL CHECK (jsonb_typeof(profile_payload) = 'object'),
    profile_hash bytea NOT NULL CHECK (octet_length(profile_hash) = 32),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, profile_id),
    UNIQUE (community_id, principal_id),
    UNIQUE (community_id, author_public_key),
    CHECK (author_public_key IS DISTINCT FROM owner_public_key),
    CHECK (
        (profile_kind = 'human' AND owner_public_key IS NULL AND first_owner_proof_event_id IS NULL)
        OR (profile_kind = 'agent' AND owner_public_key IS NOT NULL AND first_owner_proof_event_id IS NOT NULL)
    ),
    FOREIGN KEY (community_id, source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, first_owner_proof_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at)
);

CREATE FUNCTION public.guard_collaboration_identity_profile_update() RETURNS trigger AS $$
BEGIN
    IF NEW.community_id IS DISTINCT FROM OLD.community_id
       OR NEW.profile_id IS DISTINCT FROM OLD.profile_id
       OR NEW.principal_id IS DISTINCT FROM OLD.principal_id
       OR NEW.author_public_key IS DISTINCT FROM OLD.author_public_key
       OR NEW.profile_kind IS DISTINCT FROM OLD.profile_kind
       OR NEW.owner_public_key IS DISTINCT FROM OLD.owner_public_key
       OR NEW.first_owner_proof_event_id IS DISTINCT FROM OLD.first_owner_proof_event_id THEN
        RAISE EXCEPTION 'identity profile association and first owner are immutable'
            USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.aggregate_version <> OLD.aggregate_version + 1 THEN
        RAISE EXCEPTION 'identity profile aggregate version conflict'
            USING ERRCODE = 'serialization_failure';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_identity_profile_update_guard
    BEFORE UPDATE ON public.collaboration_identity_profiles
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_identity_profile_update();

ALTER TABLE public.collaboration_identity_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_identity_profiles FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_identity_profiles_admission
    ON public.collaboration_identity_profiles AS PERMISSIVE FOR ALL
    USING (true) WITH CHECK (true);

CREATE POLICY collaboration_identity_profiles_community
    ON public.collaboration_identity_profiles AS RESTRICTIVE FOR ALL
    USING (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid)
    WITH CHECK (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid);
