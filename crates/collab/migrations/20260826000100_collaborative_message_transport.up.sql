ALTER TABLE public.collaboration_command_receipts
    DROP CONSTRAINT collaboration_command_receipts_originating_adapter_check;
ALTER TABLE public.collaboration_command_receipts
    ADD CONSTRAINT collaboration_command_receipts_originating_adapter_check CHECK (
        originating_adapter IN (
            'nostr_in_process',
            'nostr_temporary_sidecar',
            'zed_rpc'
        )
    );

CREATE TABLE public.collaboration_zed_community_bindings (
    legacy_root_channel_id bigint PRIMARY KEY REFERENCES public.channels(id) ON DELETE CASCADE,
    community_id uuid NOT NULL UNIQUE REFERENCES public.collaboration_communities(community_id)
        ON DELETE CASCADE
);

CREATE TABLE public.collaboration_zed_channel_bindings (
    legacy_channel_id bigint PRIMARY KEY REFERENCES public.channels(id) ON DELETE CASCADE,
    community_id uuid NOT NULL,
    channel_id uuid NOT NULL,
    UNIQUE (community_id, channel_id),
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels(community_id, channel_id) ON DELETE CASCADE
);

CREATE TABLE public.collaboration_zed_principal_bindings (
    legacy_user_id bigint NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    community_id uuid NOT NULL,
    principal_id uuid NOT NULL,
    signing_public_key bytea CHECK (
        signing_public_key IS NULL OR octet_length(signing_public_key) = 32
    ),
    display_name text NOT NULL CHECK (octet_length(display_name) BETWEEN 1 AND 255),
    avatar_url text NOT NULL DEFAULT '' CHECK (octet_length(avatar_url) <= 4096),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (legacy_user_id, community_id),
    UNIQUE (community_id, principal_id),
    FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships(community_id, principal_id)
        ON DELETE CASCADE
);

CREATE TABLE public.collaboration_message_read_states (
    community_id uuid NOT NULL,
    channel_id uuid NOT NULL,
    principal_id uuid NOT NULL,
    last_outbox_sequence bigint NOT NULL CHECK (last_outbox_sequence >= 0),
    operation_id uuid NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, channel_id, principal_id),
    UNIQUE (community_id, operation_id),
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels(community_id, channel_id) ON DELETE CASCADE,
    FOREIGN KEY (community_id, principal_id)
        REFERENCES public.collaboration_community_memberships(community_id, principal_id)
        ON DELETE CASCADE
);

ALTER TABLE public.collaboration_message_read_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_message_read_states FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_message_read_states_admission
    ON public.collaboration_message_read_states
    AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY collaboration_message_read_states_community
    ON public.collaboration_message_read_states
    AS RESTRICTIVE FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );

CREATE INDEX collaboration_message_read_states_frontier
    ON public.collaboration_message_read_states (
        community_id, channel_id, last_outbox_sequence, principal_id
    );
