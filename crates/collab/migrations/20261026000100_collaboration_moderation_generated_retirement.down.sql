CREATE OR REPLACE FUNCTION public.guard_collaboration_moderation_version_retirement() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'collaboration moderation versions cannot be deleted'
            USING ERRCODE = 'check_violation';
    END IF;
    IF OLD.is_current
       AND NOT NEW.is_current
       AND (to_jsonb(NEW) - 'is_current') = (to_jsonb(OLD) - 'is_current') THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'collaboration moderation history is immutable'
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;
