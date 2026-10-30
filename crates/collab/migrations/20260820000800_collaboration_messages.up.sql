CREATE TABLE public.collaboration_messages (
    community_id uuid NOT NULL,
    message_id uuid NOT NULL CHECK (
        message_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    channel_id uuid NOT NULL,
    source_event_id bytea NOT NULL CHECK (octet_length(source_event_id) = 32),
    current_event_id bytea NOT NULL CHECK (octet_length(current_event_id) = 32),
    deleted_by_event_id bytea CHECK (
        deleted_by_event_id IS NULL OR octet_length(deleted_by_event_id) = 32
    ),
    author_principal_id uuid NOT NULL,
    message_created_at numeric(20, 0) NOT NULL CHECK (
        message_created_at BETWEEN 0 AND 18446744073709551615
    ),
    lifecycle_state text NOT NULL CHECK (
        lifecycle_state IN ('active', 'edited', 'deleted')
    ),
    message_version numeric(20, 0) NOT NULL CHECK (message_version >= 1),
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
    projected_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, message_id),
    UNIQUE (community_id, source_event_id),
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels (community_id, channel_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, author_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, source_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, current_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, deleted_by_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (
        (lifecycle_state = 'active'
            AND current_event_id = source_event_id
            AND deleted_by_event_id IS NULL)
        OR (lifecycle_state = 'edited'
            AND current_event_id <> source_event_id
            AND deleted_by_event_id IS NULL)
        OR (lifecycle_state = 'deleted' AND deleted_by_event_id IS NOT NULL)
    )
);

CREATE INDEX collaboration_messages_channel_window
    ON public.collaboration_messages (
        community_id,
        channel_id,
        message_created_at DESC,
        source_event_id ASC
    );
CREATE INDEX collaboration_messages_author_window
    ON public.collaboration_messages (
        community_id,
        author_principal_id,
        message_created_at DESC,
        source_event_id ASC
    );
CREATE INDEX collaboration_messages_tombstones
    ON public.collaboration_messages (
        community_id,
        channel_id,
        message_created_at DESC,
        source_event_id ASC
    ) WHERE lifecycle_state = 'deleted';
CREATE INDEX collaboration_messages_provenance
    ON public.collaboration_messages (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_message_auxiliary_events (
    community_id uuid NOT NULL,
    auxiliary_event_id bytea NOT NULL CHECK (octet_length(auxiliary_event_id) = 32),
    channel_id uuid NOT NULL,
    target_message_event_id bytea CHECK (
        target_message_event_id IS NULL OR octet_length(target_message_event_id) = 32
    ),
    actor_principal_id uuid NOT NULL,
    auxiliary_kind text NOT NULL CHECK (
        auxiliary_kind IN (
            'edit',
            'delete',
            'reaction_add',
            'reaction_remove',
            'pin',
            'unpin',
            'bookmark',
            'unbookmark',
            'schedule',
            'schedule_cancel',
            'schedule_publish'
        )
    ),
    related_event_id bytea CHECK (
        related_event_id IS NULL OR octet_length(related_event_id) = 32
    ),
    emoji text CHECK (emoji IS NULL OR octet_length(emoji) BETWEEN 1 AND 4096),
    schedule_id uuid,
    scheduled_for numeric(20, 0) CHECK (
        scheduled_for IS NULL
        OR scheduled_for BETWEEN 0 AND 18446744073709551615
    ),
    event_created_at numeric(20, 0) NOT NULL CHECK (
        event_created_at BETWEEN 0 AND 18446744073709551615
    ),
    is_tombstone boolean NOT NULL,
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
    projected_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, auxiliary_event_id),
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels (community_id, channel_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, actor_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, auxiliary_event_id)
        REFERENCES public.collaboration_events (community_id, event_id) ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (
        (auxiliary_kind IN (
            'edit', 'delete', 'reaction_add', 'reaction_remove',
            'pin', 'unpin', 'bookmark', 'unbookmark'
        ) AND target_message_event_id IS NOT NULL)
        OR (auxiliary_kind IN ('schedule', 'schedule_cancel', 'schedule_publish'))
    ),
    CHECK (
        (auxiliary_kind IN ('reaction_add', 'reaction_remove') AND emoji IS NOT NULL)
        OR (auxiliary_kind NOT IN ('reaction_add', 'reaction_remove') AND emoji IS NULL)
    ),
    CHECK (
        (auxiliary_kind = 'schedule'
            AND schedule_id IS NOT NULL
            AND scheduled_for IS NOT NULL)
        OR (auxiliary_kind IN ('schedule_cancel', 'schedule_publish')
            AND schedule_id IS NOT NULL
            AND scheduled_for IS NULL)
        OR (auxiliary_kind NOT IN ('schedule', 'schedule_cancel', 'schedule_publish')
            AND schedule_id IS NULL
            AND scheduled_for IS NULL)
    ),
    CHECK (
        (auxiliary_kind IN (
            'reaction_remove', 'unpin', 'unbookmark',
            'schedule_cancel', 'schedule_publish'
        ) AND related_event_id IS NOT NULL)
        OR (auxiliary_kind NOT IN (
            'reaction_remove', 'unpin', 'unbookmark',
            'schedule_cancel', 'schedule_publish'
        ))
    ),
    CHECK (
        is_tombstone = (auxiliary_kind IN (
            'delete', 'reaction_remove', 'unpin', 'unbookmark', 'schedule_cancel'
        ))
    )
);

CREATE INDEX collaboration_message_auxiliary_target_window
    ON public.collaboration_message_auxiliary_events (
        community_id,
        channel_id,
        target_message_event_id,
        event_created_at ASC,
        auxiliary_event_id ASC
    ) WHERE target_message_event_id IS NOT NULL;
CREATE INDEX collaboration_message_auxiliary_kind_window
    ON public.collaboration_message_auxiliary_events (
        community_id,
        channel_id,
        auxiliary_kind,
        event_created_at DESC,
        auxiliary_event_id ASC
    );
CREATE INDEX collaboration_message_auxiliary_tombstones
    ON public.collaboration_message_auxiliary_events (
        community_id,
        channel_id,
        event_created_at ASC,
        auxiliary_event_id ASC
    ) WHERE is_tombstone;
CREATE INDEX collaboration_message_auxiliary_schedules
    ON public.collaboration_message_auxiliary_events (
        community_id,
        scheduled_for,
        schedule_id,
        auxiliary_event_id
    ) WHERE auxiliary_kind = 'schedule';
CREATE INDEX collaboration_message_auxiliary_events_provenance
    ON public.collaboration_message_auxiliary_events (
        community_id, source_system, source_record_id, source_version
    );

ALTER TABLE public.collaboration_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_messages FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_messages_admission
    ON public.collaboration_messages
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_messages_community
    ON public.collaboration_messages
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );

ALTER TABLE public.collaboration_message_auxiliary_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_message_auxiliary_events FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_message_auxiliary_events_admission
    ON public.collaboration_message_auxiliary_events
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_message_auxiliary_events_community
    ON public.collaboration_message_auxiliary_events
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
