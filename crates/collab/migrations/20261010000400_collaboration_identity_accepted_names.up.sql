ALTER TABLE public.collaboration_identity_profiles
    ADD COLUMN accepted_nip05_name text CHECK (accepted_nip05_name IS NULL OR char_length(accepted_nip05_name) BETWEEN 3 AND 255),
    ADD COLUMN accepted_nip05_source_event_id bytea CHECK (accepted_nip05_source_event_id IS NULL OR octet_length(accepted_nip05_source_event_id) = 32),
    ADD COLUMN nip05_processed_event_id bytea CHECK (nip05_processed_event_id IS NULL OR octet_length(nip05_processed_event_id) = 32),
    ADD COLUMN nip05_projection_host text CHECK (nip05_projection_host IS NULL OR char_length(nip05_projection_host) BETWEEN 1 AND 253),
    ADD CONSTRAINT collaboration_identity_accepted_name_event_fk
        FOREIGN KEY (community_id, accepted_nip05_source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    ADD CONSTRAINT collaboration_identity_processed_name_event_fk
        FOREIGN KEY (community_id, nip05_processed_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    ADD CONSTRAINT collaboration_identity_accepted_name_shape CHECK (COALESCE((
        (nip05_processed_event_id IS NULL AND nip05_projection_host IS NULL
            AND accepted_nip05_name IS NULL AND accepted_nip05_source_event_id IS NULL
            AND projection_hash_version IN (1, 2, 3))
        OR (nip05_processed_event_id IS NOT NULL AND nip05_projection_host IS NOT NULL
            AND projection_hash_version = 4
            AND ((accepted_nip05_name IS NULL) = (accepted_nip05_source_event_id IS NULL))
            AND (accepted_nip05_name IS NULL OR (
                position('@' IN accepted_nip05_name) > 1
                AND split_part(accepted_nip05_name, '@', 2) <> '')))
    ), false)),
    DROP CONSTRAINT collaboration_identity_projection_hash_version;
ALTER TABLE public.collaboration_identity_profiles
    ADD CONSTRAINT collaboration_identity_projection_hash_version CHECK (projection_hash_version IN (1, 2, 3, 4)),
    DROP CONSTRAINT collaboration_identity_observation_provenance;
ALTER TABLE public.collaboration_identity_profiles ADD CONSTRAINT collaboration_identity_observation_provenance CHECK (COALESCE((
    (observation_proof IS NULL AND observation_source_event_id IS NULL AND observation_verified_at_millis IS NULL AND observation_moderation_operation_id IS NULL)
    OR (observation_proof IS NOT NULL AND observation_verified_at_millis IS NOT NULL
        AND observation_proof->>'source_kind' IN ('RetainedEventAuthor', 'VerifiedNip42Key', 'VerifiedNip42PresentedOwner', 'AdmittedModerationTarget')
        AND ((observation_proof->>'source_kind' = 'RetainedEventAuthor') = (observation_source_event_id IS NOT NULL))
        AND ((observation_proof->>'source_kind' = 'AdmittedModerationTarget') = (observation_moderation_operation_id IS NOT NULL))
        AND projection_hash_version IN (3, 4))
), false));
CREATE UNIQUE INDEX collaboration_identity_accepted_names_unique
    ON public.collaboration_identity_profiles (community_id, lower(accepted_nip05_name))
    WHERE accepted_nip05_name IS NOT NULL;
