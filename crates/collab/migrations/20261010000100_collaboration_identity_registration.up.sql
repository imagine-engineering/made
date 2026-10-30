ALTER TABLE public.collaboration_identity_profiles
    ALTER COLUMN profile_kind DROP NOT NULL,
    ALTER COLUMN profile_payload DROP NOT NULL,
    ALTER COLUMN source_event_id DROP NOT NULL,
    ADD COLUMN first_owner_auth_proof jsonb CHECK (first_owner_auth_proof IS NULL OR jsonb_typeof(first_owner_auth_proof) = 'object'),
    ADD COLUMN latest_owner_auth_proof jsonb CHECK (latest_owner_auth_proof IS NULL OR jsonb_typeof(latest_owner_auth_proof) = 'object'),
    ADD COLUMN registration_source_event_id bytea CHECK (registration_source_event_id IS NULL OR octet_length(registration_source_event_id) = 32),
    ADD COLUMN registration_verified_at_seconds numeric(20, 0) CHECK (registration_verified_at_seconds BETWEEN 0 AND 18446744073709551615),
    ADD COLUMN projection_hash_version smallint NOT NULL DEFAULT 1 CHECK (projection_hash_version IN (1, 2)),
    ADD COLUMN legacy_profile_hash bytea CHECK (legacy_profile_hash IS NULL OR octet_length(legacy_profile_hash) = 32);

DO $$
DECLARE constraint_name text;
BEGIN
    FOR constraint_name IN SELECT conname FROM pg_constraint
        WHERE conrelid = 'public.collaboration_identity_profiles'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%profile_kind%first_owner_proof_event_id%'
    LOOP
        EXECUTE format('ALTER TABLE public.collaboration_identity_profiles DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END;
$$;

ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_registration_shape CHECK (COALESCE((
    (registration_source_event_id IS NULL) = (registration_verified_at_seconds IS NULL)
    AND (
        (profile_kind IS NULL AND profile_payload IS NULL AND owner_public_key IS NULL
            AND first_owner_proof_event_id IS NULL AND first_owner_auth_proof IS NULL
            AND latest_owner_auth_proof IS NULL AND source_event_id IS NULL
            AND registration_source_event_id IS NOT NULL)
        OR (profile_kind = 'human' AND profile_payload IS NOT NULL AND source_event_id IS NOT NULL AND owner_public_key IS NULL
            AND first_owner_proof_event_id IS NULL AND first_owner_auth_proof IS NULL
            AND latest_owner_auth_proof IS NULL)
        OR (profile_kind = 'agent' AND profile_payload IS NOT NULL AND owner_public_key IS NOT NULL
            AND (source_event_id IS NOT NULL OR latest_owner_auth_proof IS NOT NULL)
            AND ((first_owner_proof_event_id IS NOT NULL) <> (first_owner_auth_proof IS NOT NULL)))
    )
), false));

CREATE OR REPLACE FUNCTION public.guard_collaboration_identity_profile_update() RETURNS trigger AS $$
BEGIN
    IF NEW.community_id IS DISTINCT FROM OLD.community_id
       OR NEW.profile_id IS DISTINCT FROM OLD.profile_id
       OR NEW.principal_id IS DISTINCT FROM OLD.principal_id
       OR NEW.author_public_key IS DISTINCT FROM OLD.author_public_key
       OR (OLD.profile_kind IS NOT NULL AND NEW.profile_kind IS DISTINCT FROM OLD.profile_kind)
       OR (OLD.owner_public_key IS NOT NULL AND NEW.owner_public_key IS DISTINCT FROM OLD.owner_public_key)
       OR (OLD.first_owner_proof_event_id IS NOT NULL AND NEW.first_owner_proof_event_id IS DISTINCT FROM OLD.first_owner_proof_event_id)
       OR (OLD.first_owner_auth_proof IS NOT NULL AND NEW.first_owner_auth_proof IS DISTINCT FROM OLD.first_owner_auth_proof)
       OR (OLD.registration_source_event_id IS NOT NULL AND NEW.registration_source_event_id IS DISTINCT FROM OLD.registration_source_event_id)
       OR (OLD.registration_verified_at_seconds IS NOT NULL AND NEW.registration_verified_at_seconds IS DISTINCT FROM OLD.registration_verified_at_seconds)
       OR NEW.projection_hash_version < OLD.projection_hash_version THEN
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
