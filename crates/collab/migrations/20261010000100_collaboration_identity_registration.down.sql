DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
        WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'identity registration rollback requires a migration role that can inspect every tenant'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;

LOCK TABLE public.collaboration_identity_profiles IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_identity_profiles
        WHERE profile_kind IS NULL OR profile_payload IS NULL OR source_event_id IS NULL
           OR first_owner_auth_proof IS NOT NULL OR latest_owner_auth_proof IS NOT NULL
           OR registration_source_event_id IS NOT NULL OR registration_verified_at_seconds IS NOT NULL
           OR (projection_hash_version = 2 AND legacy_profile_hash IS NULL)) THEN
        RAISE EXCEPTION 'identity registration rollback would discard retained identity or owner evidence'
            USING ERRCODE = 'check_violation';
    END IF;
END;
$$;

DROP TRIGGER collaboration_identity_profile_update_guard ON public.collaboration_identity_profiles;
UPDATE public.collaboration_identity_profiles SET profile_hash = legacy_profile_hash WHERE projection_hash_version = 2;
ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_registration_shape,
    ALTER COLUMN profile_kind SET NOT NULL,
    ALTER COLUMN profile_payload SET NOT NULL,
    ALTER COLUMN source_event_id SET NOT NULL,
    DROP COLUMN first_owner_auth_proof,
    DROP COLUMN latest_owner_auth_proof,
    DROP COLUMN registration_source_event_id,
    DROP COLUMN registration_verified_at_seconds,
    DROP COLUMN projection_hash_version,
    DROP COLUMN legacy_profile_hash,
    ADD CHECK (
        (profile_kind = 'human' AND owner_public_key IS NULL AND first_owner_proof_event_id IS NULL)
        OR (profile_kind = 'agent' AND owner_public_key IS NOT NULL AND first_owner_proof_event_id IS NOT NULL)
    );

CREATE OR REPLACE FUNCTION public.guard_collaboration_identity_profile_update() RETURNS trigger AS $$
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
