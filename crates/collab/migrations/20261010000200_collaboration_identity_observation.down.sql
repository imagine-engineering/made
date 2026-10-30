DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'identity observation rollback requires every-tenant visibility' USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;
LOCK TABLE public.collaboration_identity_profiles IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_identity_profiles
        WHERE observation_proof IS NOT NULL OR observation_source_event_id IS NOT NULL
           OR observation_verified_at_millis IS NOT NULL OR projection_hash_version = 3) THEN
        RAISE EXCEPTION 'identity observation rollback would discard observed identity evidence' USING ERRCODE = 'check_violation';
    END IF;
END;
$$;
ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_registration_shape,
    DROP CONSTRAINT collaboration_identity_observation_provenance,
    DROP CONSTRAINT collaboration_identity_observation_event_fk,
    DROP CONSTRAINT collaboration_identity_projection_hash_version,
    DROP COLUMN observation_proof,
    DROP COLUMN observation_source_event_id,
    DROP COLUMN observation_verified_at_millis,
    ADD CONSTRAINT collaboration_identity_profiles_projection_hash_version_check CHECK (projection_hash_version IN (1, 2));
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
