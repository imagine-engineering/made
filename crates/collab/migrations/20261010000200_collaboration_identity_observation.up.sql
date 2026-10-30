ALTER TABLE public.collaboration_identity_profiles
    ADD COLUMN observation_proof jsonb CHECK (observation_proof IS NULL OR jsonb_typeof(observation_proof) = 'object'),
    ADD COLUMN observation_source_event_id bytea CHECK (observation_source_event_id IS NULL OR octet_length(observation_source_event_id) = 32),
    ADD COLUMN observation_verified_at_millis numeric(20, 0) CHECK (observation_verified_at_millis BETWEEN 0 AND 18446744073709551615),
    ADD CONSTRAINT collaboration_identity_observation_event_fk
        FOREIGN KEY (community_id, observation_source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT;

ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_observation_provenance CHECK (COALESCE((
    (observation_proof IS NULL AND observation_source_event_id IS NULL AND observation_verified_at_millis IS NULL)
    OR (observation_proof IS NOT NULL AND observation_verified_at_millis IS NOT NULL
        AND observation_proof->>'source_kind' IN ('RetainedEventAuthor', 'VerifiedNip42Key', 'VerifiedNip42PresentedOwner')
        AND ((observation_proof->>'source_kind' = 'RetainedEventAuthor') = (observation_source_event_id IS NOT NULL))
        AND projection_hash_version = 3)
), false));

ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_profiles_projection_hash_version_check;
ALTER TABLE public.collaboration_identity_profiles
    ADD CONSTRAINT collaboration_identity_projection_hash_version CHECK (projection_hash_version IN (1, 2, 3)),
    DROP CONSTRAINT collaboration_identity_registration_shape,
    ADD CONSTRAINT collaboration_identity_registration_shape CHECK (COALESCE((
        (registration_source_event_id IS NULL) = (registration_verified_at_seconds IS NULL)
        AND (
            (profile_kind IS NULL AND profile_payload IS NULL AND owner_public_key IS NULL
                AND first_owner_proof_event_id IS NULL AND first_owner_auth_proof IS NULL
                AND latest_owner_auth_proof IS NULL AND source_event_id IS NULL
                AND (registration_source_event_id IS NOT NULL OR observation_proof IS NOT NULL))
            OR (profile_kind = 'human' AND profile_payload IS NOT NULL AND source_event_id IS NOT NULL AND owner_public_key IS NULL
                AND first_owner_proof_event_id IS NULL AND first_owner_auth_proof IS NULL AND latest_owner_auth_proof IS NULL)
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
