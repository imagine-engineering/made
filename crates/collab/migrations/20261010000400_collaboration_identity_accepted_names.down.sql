DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'accepted name rollback requires every-tenant visibility' USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;
LOCK TABLE public.collaboration_identity_profiles IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_identity_profiles WHERE nip05_processed_event_id IS NOT NULL OR projection_hash_version = 4) THEN
        RAISE EXCEPTION 'accepted name rollback would discard retained projection provenance' USING ERRCODE = 'check_violation';
    END IF;
END;
$$;
DROP INDEX public.collaboration_identity_accepted_names_unique;
ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_accepted_name_event_fk,
    DROP CONSTRAINT collaboration_identity_processed_name_event_fk,
    DROP CONSTRAINT collaboration_identity_accepted_name_shape,
    DROP CONSTRAINT collaboration_identity_projection_hash_version,
    DROP CONSTRAINT collaboration_identity_observation_provenance,
    DROP COLUMN accepted_nip05_name,
    DROP COLUMN accepted_nip05_source_event_id,
    DROP COLUMN nip05_processed_event_id,
    DROP COLUMN nip05_projection_host;
ALTER TABLE public.collaboration_identity_profiles
    ADD CONSTRAINT collaboration_identity_projection_hash_version CHECK (projection_hash_version IN (1, 2, 3));
ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_observation_provenance CHECK (COALESCE((
    (observation_proof IS NULL AND observation_source_event_id IS NULL AND observation_verified_at_millis IS NULL AND observation_moderation_operation_id IS NULL)
    OR (observation_proof IS NOT NULL AND observation_verified_at_millis IS NOT NULL
        AND observation_proof->>'source_kind' IN ('RetainedEventAuthor', 'VerifiedNip42Key', 'VerifiedNip42PresentedOwner', 'AdmittedModerationTarget')
        AND ((observation_proof->>'source_kind' = 'RetainedEventAuthor') = (observation_source_event_id IS NOT NULL))
        AND ((observation_proof->>'source_kind' = 'AdmittedModerationTarget') = (observation_moderation_operation_id IS NOT NULL))
        AND projection_hash_version = 3)
), false));
