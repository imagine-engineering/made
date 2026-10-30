DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'virtual-author rollback requires complete tenant visibility' USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;
LOCK TABLE public.collaboration_channel_memberships, public.collaboration_messages,
    public.collaboration_message_auxiliary_events, public.collaboration_message_read_states IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_channel_memberships WHERE agent_profile_id IS NOT NULL)
       OR EXISTS (SELECT 1 FROM public.collaboration_messages WHERE agent_author_profile_id IS NOT NULL)
       OR EXISTS (SELECT 1 FROM public.collaboration_message_auxiliary_events WHERE agent_actor_profile_id IS NOT NULL)
       OR EXISTS (SELECT 1 FROM public.collaboration_message_read_states WHERE agent_profile_id IS NOT NULL) THEN
        RAISE EXCEPTION 'virtual agent attribution prevents lossless rollback' USING ERRCODE = 'check_violation';
    END IF;
END;
$$;
DROP TRIGGER collaboration_channel_memberships_virtual_agent_reference ON public.collaboration_channel_memberships;
ALTER TABLE public.collaboration_channel_memberships
    DROP CONSTRAINT collaboration_channel_memberships_agent_profile_fk,
    DROP CONSTRAINT collaboration_channel_memberships_persistent_member_fk,
    DROP COLUMN persistent_member_principal_id,
    DROP COLUMN agent_profile_id,
    ADD FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT;
DROP TRIGGER collaboration_messages_virtual_agent_reference ON public.collaboration_messages;
ALTER TABLE public.collaboration_messages
    DROP CONSTRAINT collaboration_messages_agent_author_shape,
    DROP CONSTRAINT collaboration_messages_agent_profile_fk,
    DROP CONSTRAINT collaboration_messages_persistent_member_fk,
    DROP COLUMN persistent_member_principal_id,
    DROP COLUMN agent_author_profile_id,
    ADD FOREIGN KEY (community_id, author_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT;
DROP TRIGGER collaboration_message_auxiliary_events_virtual_agent_reference ON public.collaboration_message_auxiliary_events;
ALTER TABLE public.collaboration_message_auxiliary_events
    DROP CONSTRAINT collaboration_message_auxiliary_events_agent_profile_fk,
    DROP CONSTRAINT collaboration_message_auxiliary_events_persistent_member_fk,
    DROP COLUMN persistent_member_principal_id,
    DROP COLUMN agent_actor_profile_id,
    ADD FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT;
DROP TRIGGER collaboration_message_read_states_virtual_agent_reference ON public.collaboration_message_read_states;
ALTER TABLE public.collaboration_message_read_states
    DROP CONSTRAINT collaboration_message_read_states_agent_profile_fk,
    DROP CONSTRAINT collaboration_message_read_states_persistent_member_fk,
    DROP COLUMN persistent_member_principal_id,
    DROP COLUMN agent_profile_id,
    ADD FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE CASCADE;
ALTER TABLE public.collaboration_messages
    DROP CONSTRAINT collaboration_messages_agent_owner_fk,
    DROP COLUMN agent_author_owner_principal_id,
    DROP COLUMN agent_author_proof_event_id,
    DROP COLUMN agent_author_attribution;
DROP FUNCTION public.guard_collaboration_virtual_agent_reference();
ALTER TABLE public.collaboration_identity_profiles
    DROP CONSTRAINT collaboration_identity_profile_principal_reference;
