ALTER TABLE public.collaboration_identity_profiles
    ADD CONSTRAINT collaboration_identity_profile_principal_reference
    UNIQUE (community_id, profile_id, principal_id);

ALTER TABLE public.collaboration_channel_memberships
    ADD COLUMN agent_profile_id uuid,
    ADD COLUMN persistent_member_principal_id uuid GENERATED ALWAYS AS
        (CASE WHEN agent_profile_id IS NULL THEN principal_id ELSE NULL END) STORED,
    ADD CONSTRAINT collaboration_channel_memberships_persistent_member_fk
        FOREIGN KEY (community_id, persistent_member_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT collaboration_channel_memberships_agent_profile_fk
        FOREIGN KEY (community_id, agent_profile_id, principal_id)
        REFERENCES public.collaboration_identity_profiles (community_id, profile_id, principal_id)
        ON DELETE RESTRICT;

DO $$
DECLARE existing_constraint text; constraint_count integer;
BEGIN
    SELECT count(*), min(constraint_name.conname::text) INTO constraint_count, existing_constraint
    FROM pg_constraint AS constraint_name
    WHERE constraint_name.conrelid = 'public.collaboration_channel_memberships'::regclass
      AND constraint_name.confrelid = 'public.collaboration_community_memberships'::regclass
      AND constraint_name.contype = 'f'
      AND constraint_name.conkey = ARRAY[
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_channel_memberships'::regclass AND attname = 'community_id'),
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_channel_memberships'::regclass AND attname = 'principal_id')
      ]::smallint[];
    IF constraint_count <> 1 THEN
        RAISE EXCEPTION 'unexpected original membership reference' USING ERRCODE = 'check_violation';
    END IF;
    EXECUTE format('ALTER TABLE public.collaboration_channel_memberships DROP CONSTRAINT %I', existing_constraint);
END;
$$;

ALTER TABLE public.collaboration_messages
    ADD COLUMN agent_author_profile_id uuid,
    ADD COLUMN persistent_member_principal_id uuid GENERATED ALWAYS AS
        (CASE WHEN agent_author_profile_id IS NULL THEN author_principal_id ELSE NULL END) STORED,
    ADD CONSTRAINT collaboration_messages_persistent_member_fk
        FOREIGN KEY (community_id, persistent_member_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT collaboration_messages_agent_profile_fk
        FOREIGN KEY (community_id, agent_author_profile_id, author_principal_id)
        REFERENCES public.collaboration_identity_profiles (community_id, profile_id, principal_id)
        ON DELETE RESTRICT;

DO $$
DECLARE existing_constraint text; constraint_count integer;
BEGIN
    SELECT count(*), min(constraint_name.conname::text) INTO constraint_count, existing_constraint
    FROM pg_constraint AS constraint_name
    WHERE constraint_name.conrelid = 'public.collaboration_messages'::regclass
      AND constraint_name.confrelid = 'public.collaboration_community_memberships'::regclass
      AND constraint_name.contype = 'f'
      AND constraint_name.conkey = ARRAY[
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_messages'::regclass AND attname = 'community_id'),
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_messages'::regclass AND attname = 'author_principal_id')
      ]::smallint[];
    IF constraint_count <> 1 THEN
        RAISE EXCEPTION 'unexpected original membership reference' USING ERRCODE = 'check_violation';
    END IF;
    EXECUTE format('ALTER TABLE public.collaboration_messages DROP CONSTRAINT %I', existing_constraint);
END;
$$;

ALTER TABLE public.collaboration_message_auxiliary_events
    ADD COLUMN agent_actor_profile_id uuid,
    ADD COLUMN persistent_member_principal_id uuid GENERATED ALWAYS AS
        (CASE WHEN agent_actor_profile_id IS NULL THEN actor_principal_id ELSE NULL END) STORED,
    ADD CONSTRAINT collaboration_message_auxiliary_events_persistent_member_fk
        FOREIGN KEY (community_id, persistent_member_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT collaboration_message_auxiliary_events_agent_profile_fk
        FOREIGN KEY (community_id, agent_actor_profile_id, actor_principal_id)
        REFERENCES public.collaboration_identity_profiles (community_id, profile_id, principal_id)
        ON DELETE RESTRICT;

DO $$
DECLARE existing_constraint text; constraint_count integer;
BEGIN
    SELECT count(*), min(constraint_name.conname::text) INTO constraint_count, existing_constraint
    FROM pg_constraint AS constraint_name
    WHERE constraint_name.conrelid = 'public.collaboration_message_auxiliary_events'::regclass
      AND constraint_name.confrelid = 'public.collaboration_community_memberships'::regclass
      AND constraint_name.contype = 'f'
      AND constraint_name.conkey = ARRAY[
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_message_auxiliary_events'::regclass AND attname = 'community_id'),
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_message_auxiliary_events'::regclass AND attname = 'actor_principal_id')
      ]::smallint[];
    IF constraint_count <> 1 THEN
        RAISE EXCEPTION 'unexpected original membership reference' USING ERRCODE = 'check_violation';
    END IF;
    EXECUTE format('ALTER TABLE public.collaboration_message_auxiliary_events DROP CONSTRAINT %I', existing_constraint);
END;
$$;

ALTER TABLE public.collaboration_message_read_states
    ADD COLUMN agent_profile_id uuid,
    ADD COLUMN persistent_member_principal_id uuid GENERATED ALWAYS AS
        (CASE WHEN agent_profile_id IS NULL THEN principal_id ELSE NULL END) STORED,
    ADD CONSTRAINT collaboration_message_read_states_persistent_member_fk
        FOREIGN KEY (community_id, persistent_member_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT collaboration_message_read_states_agent_profile_fk
        FOREIGN KEY (community_id, agent_profile_id, principal_id)
        REFERENCES public.collaboration_identity_profiles (community_id, profile_id, principal_id)
        ON DELETE CASCADE;

DO $$
DECLARE existing_constraint text; constraint_count integer;
BEGIN
    SELECT count(*), min(constraint_name.conname::text) INTO constraint_count, existing_constraint
    FROM pg_constraint AS constraint_name
    WHERE constraint_name.conrelid = 'public.collaboration_message_read_states'::regclass
      AND constraint_name.confrelid = 'public.collaboration_community_memberships'::regclass
      AND constraint_name.contype = 'f'
      AND constraint_name.conkey = ARRAY[
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_message_read_states'::regclass AND attname = 'community_id'),
          (SELECT attnum FROM pg_attribute WHERE attrelid = 'public.collaboration_message_read_states'::regclass AND attname = 'principal_id')
      ]::smallint[];
    IF constraint_count <> 1 THEN
        RAISE EXCEPTION 'unexpected original membership reference' USING ERRCODE = 'check_violation';
    END IF;
    EXECUTE format('ALTER TABLE public.collaboration_message_read_states DROP CONSTRAINT %I', existing_constraint);
END;
$$;

ALTER TABLE public.collaboration_messages
    ADD COLUMN agent_author_owner_principal_id uuid,
    ADD COLUMN agent_author_proof_event_id bytea,
    ADD COLUMN agent_author_attribution jsonb,
    ADD CONSTRAINT collaboration_messages_agent_author_shape CHECK (COALESCE(
        (agent_author_profile_id IS NULL AND agent_author_owner_principal_id IS NULL
            AND agent_author_proof_event_id IS NULL AND agent_author_attribution IS NULL)
        OR (agent_author_profile_id IS NOT NULL AND agent_author_owner_principal_id IS NOT NULL
            AND agent_author_owner_principal_id <> author_principal_id
            AND octet_length(agent_author_proof_event_id) = 32
            AND jsonb_typeof(agent_author_attribution) = 'object'), false)),
    ADD CONSTRAINT collaboration_messages_agent_owner_fk
        FOREIGN KEY (community_id, agent_author_owner_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT;

CREATE FUNCTION public.guard_collaboration_virtual_agent_reference() RETURNS trigger AS $$
DECLARE profile_reference uuid; subject uuid;
BEGIN
    IF TG_TABLE_NAME IN ('collaboration_channel_memberships', 'collaboration_message_read_states') THEN
        profile_reference := NEW.agent_profile_id; subject := NEW.principal_id;
    ELSIF TG_TABLE_NAME = 'collaboration_messages' THEN
        profile_reference := NEW.agent_author_profile_id; subject := NEW.author_principal_id;
        IF TG_OP = 'UPDATE' AND (
            NEW.author_principal_id IS DISTINCT FROM OLD.author_principal_id
            OR NEW.agent_author_profile_id IS DISTINCT FROM OLD.agent_author_profile_id
            OR NEW.agent_author_owner_principal_id IS DISTINCT FROM OLD.agent_author_owner_principal_id
            OR NEW.agent_author_proof_event_id IS DISTINCT FROM OLD.agent_author_proof_event_id
            OR NEW.agent_author_attribution IS DISTINCT FROM OLD.agent_author_attribution
        ) THEN
            RAISE EXCEPTION 'message author attribution is immutable' USING ERRCODE = 'check_violation';
        END IF;
    ELSE
        profile_reference := NEW.agent_actor_profile_id; subject := NEW.actor_principal_id;
    END IF;
    IF profile_reference IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM public.collaboration_identity_profiles
        WHERE community_id = NEW.community_id AND profile_id = profile_reference
            AND principal_id = subject AND profile_kind = 'agent'
            AND owner_public_key IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'canonical verified agent profile required' USING ERRCODE = 'check_violation';
    END IF;
    IF TG_TABLE_NAME = 'collaboration_messages' AND profile_reference IS NOT NULL THEN
      IF NOT EXISTS (
        SELECT 1 FROM public.collaboration_identity_profiles AS agent
        JOIN public.collaboration_identity_profiles AS owner
          ON owner.community_id = agent.community_id AND owner.author_public_key = agent.owner_public_key
        WHERE agent.community_id = NEW.community_id AND agent.profile_id = profile_reference
          AND agent.principal_id = subject AND owner.principal_id = NEW.agent_author_owner_principal_id
    ) THEN
        RAISE EXCEPTION 'message owner does not match canonical identity' USING ERRCODE = 'check_violation';
      END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_channel_memberships_virtual_agent_reference
    BEFORE INSERT OR UPDATE ON public.collaboration_channel_memberships
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_virtual_agent_reference();

CREATE TRIGGER collaboration_messages_virtual_agent_reference
    BEFORE INSERT OR UPDATE ON public.collaboration_messages
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_virtual_agent_reference();

CREATE TRIGGER collaboration_message_auxiliary_events_virtual_agent_reference
    BEFORE INSERT OR UPDATE ON public.collaboration_message_auxiliary_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_virtual_agent_reference();

CREATE TRIGGER collaboration_message_read_states_virtual_agent_reference
    BEFORE INSERT OR UPDATE ON public.collaboration_message_read_states
    FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_virtual_agent_reference();
