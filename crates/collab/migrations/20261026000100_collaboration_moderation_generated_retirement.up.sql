CREATE OR REPLACE FUNCTION public.guard_collaboration_moderation_version_retirement() RETURNS trigger AS $$
DECLARE
    generated_columns text[];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'collaboration moderation versions cannot be deleted'
            USING ERRCODE = 'check_violation';
    END IF;
    -- Stored generated values are not computed in NEW until after BEFORE triggers.
    -- Their ordinary source columns remain part of the immutable comparison.
    SELECT COALESCE(array_agg(attribute.attname::text), ARRAY[]::text[])
      INTO generated_columns
      FROM pg_catalog.pg_attribute AS attribute
     WHERE attribute.attrelid = TG_RELID
       AND attribute.attnum > 0
       AND attribute.attgenerated <> ''
       AND NOT attribute.attisdropped;
    IF OLD.is_current
       AND NOT NEW.is_current
       AND (to_jsonb(NEW) - 'is_current' - generated_columns)
           = (to_jsonb(OLD) - 'is_current' - generated_columns) THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'collaboration moderation history is immutable'
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;
