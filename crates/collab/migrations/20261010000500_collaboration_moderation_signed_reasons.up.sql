ALTER TABLE public.collaboration_moderation_actions
    DROP CONSTRAINT collaboration_moderation_actions_public_reason_check,
    ADD CONSTRAINT collaboration_moderation_actions_public_reason_check CHECK (
        public_reason IS NULL OR octet_length(public_reason) <= 1024
        OR COALESCE((
            source_system = 'nostr' AND record_kind = 'restriction'
            AND action_kind IN ('apply_ban', 'lift_ban', 'apply_timeout', 'lift_timeout')
            AND integrity_algorithm = 'nostr_event_id'
            AND source_record_id ~ '^[0-9a-f]{64}$'
            AND integrity_value = source_record_id
            AND octet_length(public_reason) <= 524288
        ), false)
    );
