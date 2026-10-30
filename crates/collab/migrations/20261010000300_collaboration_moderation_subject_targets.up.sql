ALTER TABLE public.collaboration_moderation_restriction_versions
    ADD COLUMN target_identity_principal_id uuid,
    ADD COLUMN target_subject_principal_id uuid GENERATED ALWAYS AS
        (COALESCE(target_principal_id, target_identity_principal_id)) STORED,
    ADD CONSTRAINT collaboration_moderation_subject_target_shape CHECK (
        (target_principal_id IS NOT NULL) <> (target_identity_principal_id IS NOT NULL)
    ),
    ADD CONSTRAINT collaboration_moderation_identity_target_fk
        FOREIGN KEY (community_id, target_identity_principal_id)
        REFERENCES public.collaboration_identity_profiles (community_id, principal_id)
        ON DELETE RESTRICT;
ALTER TABLE public.collaboration_moderation_restriction_versions
    DROP CONSTRAINT collaboration_moderation_restriction_versions_pkey,
    ALTER COLUMN target_principal_id DROP NOT NULL,
    ADD PRIMARY KEY (community_id, target_subject_principal_id, restriction_version);
CREATE UNIQUE INDEX collaboration_moderation_subject_restrictions_current
    ON public.collaboration_moderation_restriction_versions
        (community_id, target_subject_principal_id) WHERE is_current;
CREATE INDEX collaboration_moderation_subject_restrictions_active
    ON public.collaboration_moderation_restriction_versions
        (community_id, target_subject_principal_id, ban_state, timeout_state)
    WHERE is_current AND (ban_state = 'active' OR timeout_state = 'active');
ALTER TABLE public.collaboration_identity_profiles
    ADD COLUMN observation_moderation_operation_id uuid,
    ADD CONSTRAINT collaboration_identity_observation_moderation_fk
        FOREIGN KEY (community_id, observation_moderation_operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    DROP CONSTRAINT collaboration_identity_observation_provenance;
ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_observation_provenance CHECK (COALESCE((
    (observation_proof IS NULL AND observation_source_event_id IS NULL AND observation_verified_at_millis IS NULL AND observation_moderation_operation_id IS NULL)
    OR (observation_proof IS NOT NULL AND observation_verified_at_millis IS NOT NULL
        AND observation_proof->>'source_kind' IN ('RetainedEventAuthor', 'VerifiedNip42Key', 'VerifiedNip42PresentedOwner', 'AdmittedModerationTarget')
        AND ((observation_proof->>'source_kind' = 'RetainedEventAuthor') = (observation_source_event_id IS NOT NULL))
        AND ((observation_proof->>'source_kind' = 'AdmittedModerationTarget') = (observation_moderation_operation_id IS NOT NULL))
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
       OR (OLD.observation_moderation_operation_id IS NOT NULL AND NEW.observation_moderation_operation_id IS DISTINCT FROM OLD.observation_moderation_operation_id)
       OR NEW.projection_hash_version < OLD.projection_hash_version THEN
        RAISE EXCEPTION 'identity profile association and first owner are immutable' USING ERRCODE = 'check_violation';
    END IF;
    IF NEW.aggregate_version <> OLD.aggregate_version + 1 THEN
        RAISE EXCEPTION 'identity profile aggregate version conflict' USING ERRCODE = 'serialization_failure';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
