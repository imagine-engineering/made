CREATE TABLE public.collaboration_project_groups (
    community_id uuid NOT NULL,
    project_signer_public_key bytea NOT NULL CHECK (
        octet_length(project_signer_public_key) = 32
    ),
    project_slug text NOT NULL CHECK (
        octet_length(project_slug) BETWEEN 1 AND 1024
    ),
    record_version numeric(20, 0) NOT NULL CHECK (record_version >= 1),
    is_current boolean NOT NULL,
    source_event_id bytea NOT NULL CHECK (octet_length(source_event_id) = 32),
    source_created_at numeric(20, 0) NOT NULL CHECK (
        source_created_at BETWEEN 0 AND 18446744073709551615
    ),
    name text CHECK (name IS NULL OR octet_length(name) <= 256),
    description text CHECK (
        description IS NULL OR octet_length(description) <= 2048
    ),
    visibility text NOT NULL CHECK (visibility IN ('listed', 'unlisted')),
    channel_reference text CHECK (
        channel_reference IS NULL OR octet_length(channel_reference) <= 256
    ),
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        record_version
    ),
    UNIQUE (community_id, source_event_id),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at)
);

CREATE UNIQUE INDEX collaboration_project_groups_current
    ON public.collaboration_project_groups (
        community_id, project_signer_public_key, project_slug
    )
    WHERE is_current;
CREATE INDEX collaboration_project_groups_listing
    ON public.collaboration_project_groups (
        community_id, visibility, updated_at DESC, project_signer_public_key, project_slug
    )
    WHERE is_current;

CREATE TABLE public.collaboration_project_repository_bindings (
    community_id uuid NOT NULL,
    project_signer_public_key bytea NOT NULL CHECK (
        octet_length(project_signer_public_key) = 32
    ),
    project_slug text NOT NULL CHECK (
        octet_length(project_slug) BETWEEN 1 AND 1024
    ),
    repository_kind integer NOT NULL DEFAULT 30617 CHECK (repository_kind = 30617),
    repository_owner_public_key bytea NOT NULL CHECK (
        octet_length(repository_owner_public_key) = 32
    ),
    repository_discriminator text NOT NULL CHECK (
        octet_length(repository_discriminator) BETWEEN 1 AND 1024
    ),
    binding_version numeric(20, 0) NOT NULL CHECK (binding_version >= 1),
    project_record_version numeric(20, 0) NOT NULL CHECK (project_record_version >= 1),
    is_current boolean NOT NULL,
    binding_state text NOT NULL CHECK (binding_state IN ('active', 'deleted')),
    relay_hint text CHECK (relay_hint IS NULL OR octet_length(relay_hint) <= 524288),
    deleted_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        repository_owner_public_key,
        repository_discriminator,
        binding_version
    ),
    FOREIGN KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        project_record_version
    ) REFERENCES public.collaboration_project_groups (
        community_id,
        project_signer_public_key,
        project_slug,
        record_version
    ) ON DELETE RESTRICT,
    CHECK (
        (binding_state = 'active' AND deleted_at IS NULL)
        OR (binding_state = 'deleted' AND deleted_at IS NOT NULL)
    ),
    CHECK (updated_at >= created_at),
    CHECK (deleted_at IS NULL OR deleted_at >= created_at)
);

CREATE UNIQUE INDEX collaboration_project_repository_bindings_current
    ON public.collaboration_project_repository_bindings (
        community_id,
        project_signer_public_key,
        project_slug,
        repository_owner_public_key,
        repository_discriminator
    )
    WHERE is_current;
CREATE INDEX collaboration_project_repository_bindings_project
    ON public.collaboration_project_repository_bindings (
        community_id,
        project_signer_public_key,
        project_slug,
        binding_state,
        repository_owner_public_key,
        repository_discriminator
    )
    WHERE is_current;

CREATE TABLE public.collaboration_project_channel_bindings (
    community_id uuid NOT NULL,
    project_signer_public_key bytea NOT NULL CHECK (
        octet_length(project_signer_public_key) = 32
    ),
    project_slug text NOT NULL CHECK (
        octet_length(project_slug) BETWEEN 1 AND 1024
    ),
    binding_version numeric(20, 0) NOT NULL CHECK (binding_version >= 1),
    project_record_version numeric(20, 0) NOT NULL CHECK (project_record_version >= 1),
    channel_id uuid NOT NULL CHECK (
        channel_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    is_current boolean NOT NULL,
    binding_state text NOT NULL CHECK (binding_state IN ('active', 'deleted')),
    deleted_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        binding_version
    ),
    FOREIGN KEY (
        community_id,
        project_signer_public_key,
        project_slug,
        project_record_version
    ) REFERENCES public.collaboration_project_groups (
        community_id,
        project_signer_public_key,
        project_slug,
        record_version
    ) ON DELETE RESTRICT,
    FOREIGN KEY (community_id, channel_id)
        REFERENCES public.collaboration_channels (community_id, channel_id) ON DELETE RESTRICT,
    CHECK (
        (binding_state = 'active' AND deleted_at IS NULL)
        OR (binding_state = 'deleted' AND deleted_at IS NOT NULL)
    ),
    CHECK (updated_at >= created_at),
    CHECK (deleted_at IS NULL OR deleted_at >= created_at)
);

CREATE UNIQUE INDEX collaboration_project_channel_bindings_current
    ON public.collaboration_project_channel_bindings (
        community_id, project_signer_public_key, project_slug
    )
    WHERE is_current;
CREATE INDEX collaboration_project_channel_bindings_channel
    ON public.collaboration_project_channel_bindings (
        community_id, channel_id, binding_state, project_signer_public_key, project_slug
    )
    WHERE is_current;

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_project_groups',
        'collaboration_project_repository_bindings',
        'collaboration_project_channel_bindings'
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
