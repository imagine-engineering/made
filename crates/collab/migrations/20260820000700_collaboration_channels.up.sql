CREATE TABLE public.collaboration_communities (
    community_id uuid PRIMARY KEY,
    host text NOT NULL CHECK (
        octet_length(host) BETWEEN 1 AND 255
        AND host = lower(host)
        AND host = btrim(host)
    ),
    icon text CHECK (icon IS NULL OR octet_length(icon) <= 262144),
    lifecycle_state text NOT NULL CHECK (
        lifecycle_state IN ('active', 'archived', 'quiescing', 'fenced', 'tombstone')
    ),
    join_policy_version text CHECK (
        join_policy_version IS NULL OR octet_length(join_policy_version) <= 256
    ),
    aggregate_version numeric(20, 0) NOT NULL CHECK (aggregate_version >= 1),
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
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (updated_at >= created_at)
);

CREATE UNIQUE INDEX collaboration_communities_host
    ON public.collaboration_communities (host);
CREATE INDEX collaboration_communities_provenance
    ON public.collaboration_communities (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_community_memberships (
    community_id uuid NOT NULL,
    principal_id uuid NOT NULL,
    role text NOT NULL CHECK (role IN ('owner', 'admin', 'member', 'guest', 'bot')),
    status text NOT NULL CHECK (status IN ('active', 'revoked', 'archived')),
    membership_version numeric(20, 0) NOT NULL CHECK (membership_version >= 1),
    added_by_principal_id uuid,
    joined_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
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
    PRIMARY KEY (community_id, principal_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (updated_at >= joined_at)
);

CREATE INDEX collaboration_community_memberships_status
    ON public.collaboration_community_memberships (community_id, status, role, principal_id);
CREATE INDEX collaboration_community_memberships_provenance
    ON public.collaboration_community_memberships (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_join_policy_acceptances (
    community_id uuid NOT NULL,
    principal_id uuid NOT NULL,
    policy_version text NOT NULL CHECK (octet_length(policy_version) BETWEEN 1 AND 256),
    accepted_at timestamptz NOT NULL,
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
    PRIMARY KEY (community_id, principal_id, policy_version),
    FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))
);

CREATE INDEX collaboration_join_policy_acceptances_provenance
    ON public.collaboration_join_policy_acceptances (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_channels (
    community_id uuid NOT NULL,
    channel_id uuid NOT NULL CHECK (
        channel_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    name text NOT NULL CHECK (octet_length(name) BETWEEN 1 AND 255),
    channel_type text NOT NULL CHECK (
        channel_type IN ('stream', 'forum', 'dm', 'workflow', 'ephemeral', 'huddle')
    ),
    visibility text NOT NULL CHECK (visibility IN ('open', 'private')),
    lifecycle_state text NOT NULL CHECK (
        lifecycle_state IN ('active', 'archived', 'deleted', 'expired')
    ),
    description text CHECK (description IS NULL OR octet_length(description) <= 65536),
    creator_principal_id uuid NOT NULL,
    nip29_group_id text CHECK (
        nip29_group_id IS NULL OR octet_length(nip29_group_id) BETWEEN 1 AND 255
    ),
    topic_required boolean NOT NULL DEFAULT false,
    max_members integer CHECK (max_members IS NULL OR max_members > 0),
    ttl_seconds integer CHECK (ttl_seconds IS NULL OR ttl_seconds > 0),
    expires_at timestamptz,
    channel_version numeric(20, 0) NOT NULL CHECK (channel_version >= 1),
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
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, channel_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, creator_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK ((ttl_seconds IS NULL) = (expires_at IS NULL)),
    CHECK (updated_at >= created_at)
);

CREATE UNIQUE INDEX collaboration_channels_nip29_group
    ON public.collaboration_channels (community_id, nip29_group_id)
    WHERE nip29_group_id IS NOT NULL;
CREATE INDEX collaboration_channels_listing
    ON public.collaboration_channels (
        community_id, lifecycle_state, channel_type, visibility, updated_at DESC, channel_id
    );
CREATE INDEX collaboration_channels_provenance
    ON public.collaboration_channels (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_channel_invites (
    community_id uuid NOT NULL,
    invite_id uuid NOT NULL,
    channel_id uuid,
    token_hash bytea NOT NULL CHECK (octet_length(token_hash) = 32),
    role text NOT NULL CHECK (role IN ('member', 'guest')),
    status text NOT NULL CHECK (status IN ('active', 'revoked', 'exhausted', 'expired')),
    max_uses integer CHECK (max_uses BETWEEN 1 AND 10000),
    use_count integer NOT NULL CHECK (use_count >= 0),
    expires_at timestamptz NOT NULL,
    created_by_principal_id uuid NOT NULL,
    invite_version numeric(20, 0) NOT NULL CHECK (invite_version >= 1),
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
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, invite_id),
    UNIQUE (community_id, token_hash),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels (community_id, channel_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, created_by_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (max_uses IS NULL OR use_count <= max_uses),
    CHECK (updated_at >= created_at)
);

CREATE INDEX collaboration_channel_invites_expiry
    ON public.collaboration_channel_invites (community_id, status, expires_at, invite_id);
CREATE INDEX collaboration_channel_invites_provenance
    ON public.collaboration_channel_invites (
        community_id, source_system, source_record_id, source_version
    );

CREATE TABLE public.collaboration_channel_memberships (
    community_id uuid NOT NULL,
    channel_id uuid NOT NULL,
    principal_id uuid NOT NULL,
    role text NOT NULL CHECK (role IN ('owner', 'admin', 'member', 'guest', 'bot')),
    status text NOT NULL CHECK (status IN ('active', 'revoked', 'archived')),
    membership_version numeric(20, 0) NOT NULL CHECK (membership_version >= 1),
    invited_by_principal_id uuid,
    joined_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    hidden_at timestamptz,
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
    PRIMARY KEY (community_id, channel_id, principal_id),
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels (community_id, channel_id) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL)),
    CHECK (updated_at >= joined_at)
);

CREATE INDEX collaboration_channel_memberships_principal
    ON public.collaboration_channel_memberships (
        community_id, principal_id, status, channel_id
    );
CREATE INDEX collaboration_channel_memberships_channel
    ON public.collaboration_channel_memberships (
        community_id, channel_id, status, role, principal_id
    );
CREATE INDEX collaboration_channel_memberships_provenance
    ON public.collaboration_channel_memberships (
        community_id, source_system, source_record_id, source_version
    );

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_communities',
        'collaboration_community_memberships',
        'collaboration_join_policy_acceptances',
        'collaboration_channels',
        'collaboration_channel_invites',
        'collaboration_channel_memberships'
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
