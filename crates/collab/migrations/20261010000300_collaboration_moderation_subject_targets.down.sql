DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'moderation subject rollback requires every-tenant visibility' USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;
LOCK TABLE public.collaboration_moderation_restriction_versions, public.collaboration_identity_profiles IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_moderation_restriction_versions WHERE target_identity_principal_id IS NOT NULL)
       OR EXISTS (SELECT 1 FROM public.collaboration_identity_profiles WHERE observation_moderation_operation_id IS NOT NULL
           OR observation_proof->>'source_kind' = 'AdmittedModerationTarget') THEN
        RAISE EXCEPTION 'moderation subject rollback would discard retained identity or restriction evidence' USING ERRCODE = 'check_violation';
    END IF;
END;
$$;
ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_observation_moderation_fk,
    DROP CONSTRAINT collaboration_identity_observation_provenance,
    DROP COLUMN observation_moderation_operation_id;
ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_observation_provenance CHECK (COALESCE((
    (observation_proof IS NULL AND observation_source_event_id IS NULL AND observation_verified_at_millis IS NULL)
    OR (observation_proof IS NOT NULL AND observation_verified_at_millis IS NOT NULL
        AND observation_proof->>'source_kind' IN ('RetainedEventAuthor', 'VerifiedNip42Key', 'VerifiedNip42PresentedOwner')
        AND ((observation_proof->>'source_kind' = 'RetainedEventAuthor') = (observation_source_event_id IS NOT NULL))
        AND projection_hash_version = 3)
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
       OR (OLD.observation_proof IS NOT NULL AND NEW.observation_proof IS DISTINCT FROM OLD.observation_proof)
       OR (OLD.observation_source_event_id IS NOT NULL AND NEW.observation_source_event_id IS DISTINCT FROM OLD.observation_source_event_id)
       OR (OLD.observation_verified_at_millis IS NOT NULL AND NEW.observation_verified_at_millis IS DISTINCT FROM OLD.observation_verified_at_millis)
       OR NEW.projection_hash_version < OLD.projection_hash_version THEN
        RAISE EXCEPTION 'identity profile association and first owner are immutable' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.aggregate_version <> OLD.aggregate_version + 1 THEN
        RAISE EXCEPTION 'identity profile aggregate version conflict' USING ERRCODE = 'serialization_failure';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP INDEX public.collaboration_moderation_subject_restrictions_active;
DROP INDEX public.collaboration_moderation_subject_restrictions_current;
ALTER TABLE public.collaboration_moderation_restriction_versions
    DROP CONSTRAINT collaboration_moderation_restriction_versions_pkey,
    DROP CONSTRAINT collaboration_moderation_identity_target_fk,
    DROP CONSTRAINT collaboration_moderation_subject_target_shape,
    DROP COLUMN target_subject_principal_id,
    DROP COLUMN target_identity_principal_id,
    ALTER COLUMN target_principal_id SET NOT NULL,
    ADD PRIMARY KEY (community_id, target_principal_id, restriction_version);
