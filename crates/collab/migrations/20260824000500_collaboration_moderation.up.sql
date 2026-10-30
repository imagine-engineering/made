CREATE TABLE public.collaboration_moderation_actions (
    community_id uuid NOT NULL,
    action_id uuid NOT NULL CHECK (
        action_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    action_kind text NOT NULL CHECK (action_kind IN (
        'file_report',
        'resolve_dismissed',
        'resolve_content_removed',
        'resolve_member_removed',
        'resolve_timed_out',
        'resolve_banned',
        'resolve_escalated',
        'apply_ban',
        'lift_ban',
        'apply_timeout',
        'lift_timeout',
        'archive_identity',
        'restore_identity',
        'archive_community',
        'restore_community'
    )),
    record_kind text NOT NULL CHECK (record_kind IN (
        'report', 'restriction', 'identity_archive', 'community_archive', 'content'
    )),
    record_id uuid NOT NULL CHECK (
        record_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    target_principal_id uuid CHECK (
        target_principal_id IS NULL
        OR target_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    target_event_id bytea CHECK (
        target_event_id IS NULL OR octet_length(target_event_id) = 32
    ),
    reason_code text CHECK (
        reason_code IS NULL OR octet_length(reason_code) BETWEEN 1 AND 128
    ),
    public_reason text CHECK (
        public_reason IS NULL OR octet_length(public_reason) <= 1024
    ),
    private_reason text CHECK (
        private_reason IS NULL OR octet_length(private_reason) <= 4096
    ),
    occurred_at timestamptz NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, action_id),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, source_system, source_record_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_audit_entries (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))
);

CREATE INDEX collaboration_moderation_actions_record
    ON public.collaboration_moderation_actions (
        community_id, record_kind, record_id, occurred_at DESC, action_id
    );

CREATE TABLE public.collaboration_moderation_reports (
    community_id uuid NOT NULL,
    report_id uuid NOT NULL CHECK (
        report_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    reporter_principal_id uuid NOT NULL CHECK (
        reporter_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    target_kind text NOT NULL CHECK (target_kind IN ('event', 'principal', 'blob')),
    target_event_id bytea CHECK (
        target_event_id IS NULL OR octet_length(target_event_id) = 32
    ),
    target_principal_id uuid CHECK (
        target_principal_id IS NULL
        OR target_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    target_blob_sha256 bytea CHECK (
        target_blob_sha256 IS NULL OR octet_length(target_blob_sha256) = 32
    ),
    reason_kind text NOT NULL CHECK (reason_kind IN (
        'spam', 'profanity', 'illegal_content', 'nudity', 'malware', 'impersonation', 'other'
    )),
    private_context text CHECK (
        private_context IS NULL OR octet_length(private_context) BETWEEN 1 AND 4096
    ),
    filed_operation_id uuid NOT NULL,
    filed_at timestamptz NOT NULL,
    aggregate_version numeric(20, 0) NOT NULL CHECK (aggregate_version = 1),
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, report_id),
    UNIQUE (community_id, filed_operation_id),
    UNIQUE (community_id, source_system, source_record_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, reporter_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, target_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, filed_operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (
        (target_kind = 'event'
            AND target_event_id IS NOT NULL
            AND target_principal_id IS NULL
            AND target_blob_sha256 IS NULL)
        OR (target_kind = 'principal'
            AND target_event_id IS NULL
            AND target_principal_id IS NOT NULL
            AND target_blob_sha256 IS NULL)
        OR (target_kind = 'blob'
            AND target_event_id IS NULL
            AND target_principal_id IS NULL
            AND target_blob_sha256 IS NOT NULL)
    )
);

CREATE INDEX collaboration_moderation_reports_queue
    ON public.collaboration_moderation_reports (community_id, filed_at DESC, report_id);
CREATE INDEX collaboration_moderation_reports_event_target
    ON public.collaboration_moderation_reports (community_id, target_event_id)
    WHERE target_event_id IS NOT NULL;
CREATE INDEX collaboration_moderation_reports_principal_target
    ON public.collaboration_moderation_reports (community_id, target_principal_id)
    WHERE target_principal_id IS NOT NULL;

CREATE TABLE public.collaboration_moderation_report_resolutions (
    community_id uuid NOT NULL,
    report_id uuid NOT NULL,
    resolution_kind text NOT NULL CHECK (resolution_kind IN (
        'dismissed', 'content_removed', 'member_removed', 'timed_out', 'banned', 'escalated'
    )),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    operation_id uuid NOT NULL,
    resolved_at timestamptz NOT NULL,
    resulting_version numeric(20, 0) NOT NULL CHECK (resulting_version = 2),
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, report_id),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, source_system, source_record_id),
    FOREIGN KEY (community_id, report_id)
        REFERENCES public.collaboration_moderation_reports (community_id, report_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))
);

CREATE TABLE public.collaboration_moderation_restriction_versions (
    community_id uuid NOT NULL,
    target_principal_id uuid NOT NULL CHECK (
        target_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    restriction_version numeric(20, 0) NOT NULL CHECK (restriction_version >= 1),
    is_current boolean NOT NULL,
    ban_state text NOT NULL CHECK (ban_state IN ('none', 'active')),
    ban_expires_at timestamptz,
    timeout_state text NOT NULL CHECK (timeout_state IN ('none', 'active')),
    timeout_expires_at timestamptz,
    transition_kind text NOT NULL CHECK (transition_kind IN (
        'apply_ban', 'lift_ban', 'apply_timeout', 'lift_timeout'
    )),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    operation_id uuid NOT NULL,
    occurred_at timestamptz NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, target_principal_id, restriction_version),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, source_system, source_record_id, restriction_version),
    FOREIGN KEY (community_id, target_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (ban_state = 'active' OR ban_expires_at IS NULL),
    CHECK ((timeout_state = 'active') = (timeout_expires_at IS NOT NULL)),
    CHECK (ban_expires_at IS NULL OR ban_expires_at > occurred_at),
    CHECK (timeout_expires_at IS NULL OR timeout_expires_at > occurred_at),
    CHECK (transition_kind <> 'apply_ban' OR ban_state = 'active'),
    CHECK (transition_kind <> 'lift_ban' OR ban_state = 'none'),
    CHECK (transition_kind <> 'apply_timeout' OR timeout_state = 'active'),
    CHECK (transition_kind <> 'lift_timeout' OR timeout_state = 'none')
);

CREATE UNIQUE INDEX collaboration_moderation_restrictions_current
    ON public.collaboration_moderation_restriction_versions (
        community_id, target_principal_id
    ) WHERE is_current;
CREATE INDEX collaboration_moderation_restrictions_active
    ON public.collaboration_moderation_restriction_versions (
        community_id, target_principal_id, ban_state, timeout_state
    ) WHERE is_current AND (ban_state = 'active' OR timeout_state = 'active');

CREATE TABLE public.collaboration_personal_mute_versions (
    community_id uuid NOT NULL,
    owner_principal_id uuid NOT NULL CHECK (
        owner_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    muted_principal_id uuid NOT NULL CHECK (
        muted_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    mute_version numeric(20, 0) NOT NULL CHECK (mute_version >= 1),
    is_current boolean NOT NULL,
    mute_state text NOT NULL CHECK (mute_state IN ('muted', 'unmuted')),
    operation_id uuid NOT NULL CHECK (
        operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    occurred_at timestamptz NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, owner_principal_id, muted_principal_id, mute_version),
    UNIQUE (community_id, owner_principal_id, operation_id),
    UNIQUE (
        community_id, owner_principal_id, source_system, source_record_id, mute_version
    ),
    FOREIGN KEY (community_id, owner_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, muted_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK (owner_principal_id <> muted_principal_id),
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))
);

CREATE UNIQUE INDEX collaboration_personal_mutes_current
    ON public.collaboration_personal_mute_versions (
        community_id, owner_principal_id, muted_principal_id
    ) WHERE is_current;
CREATE UNIQUE INDEX collaboration_personal_mutes_active
    ON public.collaboration_personal_mute_versions (
        community_id, owner_principal_id, muted_principal_id
    ) WHERE is_current AND mute_state = 'muted';

CREATE TABLE public.collaboration_identity_archive_versions (
    community_id uuid NOT NULL,
    target_principal_id uuid NOT NULL CHECK (
        target_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    identity_public_key bytea NOT NULL CHECK (octet_length(identity_public_key) = 32),
    archive_version numeric(20, 0) NOT NULL CHECK (archive_version >= 1),
    is_current boolean NOT NULL,
    archive_state text NOT NULL CHECK (archive_state IN ('archived', 'visible')),
    consent_path text NOT NULL CHECK (consent_path IN ('self', 'owner', 'admin')),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    replacement_principal_id uuid CHECK (
        replacement_principal_id IS NULL
        OR replacement_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    operation_id uuid NOT NULL,
    occurred_at timestamptz NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, identity_public_key, archive_version),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, source_system, source_record_id, archive_version),
    FOREIGN KEY (community_id, target_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, replacement_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (archive_state = 'archived' OR replacement_principal_id IS NULL)
);

CREATE UNIQUE INDEX collaboration_identity_archives_current
    ON public.collaboration_identity_archive_versions (
        community_id, identity_public_key
    ) WHERE is_current;
CREATE UNIQUE INDEX collaboration_identity_archives_active
    ON public.collaboration_identity_archive_versions (
        community_id, identity_public_key
    ) WHERE is_current AND archive_state = 'archived';

CREATE TABLE public.collaboration_community_archive_versions (
    community_id uuid NOT NULL,
    archive_version numeric(20, 0) NOT NULL CHECK (archive_version >= 1),
    is_current boolean NOT NULL,
    archive_state text NOT NULL CHECK (archive_state IN ('archived', 'active')),
    actor_principal_id uuid NOT NULL CHECK (
        actor_principal_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    operation_id uuid NOT NULL,
    occurred_at timestamptz NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 256
    ),
    source_observed_at timestamptz NOT NULL,
    integrity_algorithm text CHECK (
        integrity_algorithm IS NULL
        OR integrity_algorithm IN ('sha256', 'nostr_event_id', 'git_object_id')
    ),
    integrity_value text CHECK (
        integrity_value IS NULL OR octet_length(integrity_value) BETWEEN 1 AND 1024
    ),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, archive_version),
    UNIQUE (community_id, operation_id),
    UNIQUE (community_id, source_system, source_record_id, archive_version),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_moderation_actions (community_id, operation_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))
);

CREATE UNIQUE INDEX collaboration_community_archives_current
    ON public.collaboration_community_archive_versions (community_id)
    WHERE is_current;
CREATE UNIQUE INDEX collaboration_community_archives_active
    ON public.collaboration_community_archive_versions (community_id)
    WHERE is_current AND archive_state = 'archived';

CREATE FUNCTION public.reject_collaboration_moderation_history_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'collaboration moderation history is immutable'
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;

CREATE FUNCTION public.guard_collaboration_moderation_version_retirement() RETURNS trigger AS $$
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

CREATE TRIGGER collaboration_moderation_actions_immutable
    BEFORE UPDATE OR DELETE ON public.collaboration_moderation_actions
    FOR EACH ROW EXECUTE FUNCTION public.reject_collaboration_moderation_history_mutation();
CREATE TRIGGER collaboration_moderation_reports_immutable
    BEFORE UPDATE OR DELETE ON public.collaboration_moderation_reports
    FOR EACH ROW EXECUTE FUNCTION public.reject_collaboration_moderation_history_mutation();
CREATE TRIGGER collaboration_moderation_report_resolutions_immutable
    BEFORE UPDATE OR DELETE ON public.collaboration_moderation_report_resolutions
    FOR EACH ROW EXECUTE FUNCTION public.reject_collaboration_moderation_history_mutation();

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_moderation_restriction_versions',
        'collaboration_personal_mute_versions',
        'collaboration_identity_archive_versions',
        'collaboration_community_archive_versions'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_moderation_version_retirement()',
            table_name || '_retirement_guard',
            table_name
        );
    END LOOP;
END;
$$;

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_moderation_actions',
        'collaboration_moderation_reports',
        'collaboration_moderation_report_resolutions',
        'collaboration_moderation_restriction_versions',
        'collaboration_personal_mute_versions',
        'collaboration_identity_archive_versions',
        'collaboration_community_archive_versions'
    ] LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format(
            'CREATE POLICY %I ON public.%I AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true)',
            table_name || '_admission',
            table_name
        );
        EXECUTE format(
            'CREATE POLICY %I ON public.%I AS RESTRICTIVE FOR ALL USING (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid) WITH CHECK (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid)',
            table_name || '_community',
            table_name
        );
    END LOOP;
END;
$$;
