DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_user AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'signed reason rollback requires every-tenant visibility' USING ERRCODE = 'insufficient_privilege';
    END IF;
END;
$$;
LOCK TABLE public.collaboration_moderation_actions IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.collaboration_moderation_actions WHERE octet_length(public_reason) > 1024) THEN
        RAISE EXCEPTION 'signed reason rollback would discard retained moderation evidence' USING ERRCODE = 'check_violation';
    END IF;
END;
$$;
ALTER TABLE public.collaboration_moderation_actions
    DROP CONSTRAINT collaboration_moderation_actions_public_reason_check,
    ADD CONSTRAINT collaboration_moderation_actions_public_reason_check CHECK (
        public_reason IS NULL OR octet_length(public_reason) <= 1024
    );
