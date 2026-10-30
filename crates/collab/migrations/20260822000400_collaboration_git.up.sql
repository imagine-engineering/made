CREATE TABLE public.collaboration_hosted_repositories (
    community_id uuid NOT NULL,
    repository_id uuid NOT NULL CHECK (
        repository_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    repository_kind integer NOT NULL DEFAULT 30617 CHECK (repository_kind = 30617),
    repository_owner_public_key bytea NOT NULL CHECK (
        octet_length(repository_owner_public_key) = 32
    ),
    repository_discriminator text NOT NULL CHECK (
        octet_length(repository_discriminator) BETWEEN 1 AND 64
        AND repository_discriminator = btrim(repository_discriminator)
        AND repository_discriminator !~ '[[:space:]/\\]'
        AND repository_discriminator NOT IN ('.', '..')
        AND repository_discriminator !~ '[[:cntrl:]]'
    ),
    authority_kind text NOT NULL CHECK (
        authority_kind IN ('sim_hosted_nip34', 'external_provider')
    ),
    authority_version numeric(20, 0) NOT NULL CHECK (authority_version >= 1),
    lifecycle_state text NOT NULL CHECK (lifecycle_state IN ('active', 'archived')),
    provider_kind text CHECK (
        provider_kind IS NULL OR octet_length(provider_kind) BETWEEN 1 AND 64
    ),
    provider_instance text CHECK (
        provider_instance IS NULL OR octet_length(provider_instance) BETWEEN 1 AND 512
    ),
    provider_owner text CHECK (
        provider_owner IS NULL OR octet_length(provider_owner) BETWEEN 1 AND 512
    ),
    provider_repository text CHECK (
        provider_repository IS NULL OR octet_length(provider_repository) BETWEEN 1 AND 512
    ),
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
    archived_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, repository_id),
    UNIQUE (
        community_id,
        repository_kind,
        repository_owner_public_key,
        repository_discriminator
    ),
    UNIQUE (community_id, repository_id, authority_kind),
    FOREIGN KEY (community_id)
        REFERENCES public.collaboration_communities (community_id) ON DELETE RESTRICT,
    CHECK (
        (authority_kind = 'sim_hosted_nip34'
            AND provider_kind IS NULL
            AND provider_instance IS NULL
            AND provider_owner IS NULL
            AND provider_repository IS NULL)
        OR (authority_kind = 'external_provider'
            AND provider_kind IS NOT NULL
            AND provider_instance IS NOT NULL
            AND provider_owner IS NOT NULL
            AND provider_repository IS NOT NULL)
    ),
    CHECK (
        (lifecycle_state = 'active' AND archived_at IS NULL)
        OR (lifecycle_state = 'archived' AND archived_at IS NOT NULL)
    ),
    CHECK (updated_at >= created_at),
    CHECK (archived_at IS NULL OR archived_at >= created_at)
);

CREATE INDEX collaboration_hosted_repositories_authority
    ON public.collaboration_hosted_repositories (
        community_id, lifecycle_state, authority_kind, repository_id
    );
CREATE INDEX collaboration_hosted_repositories_provider
    ON public.collaboration_hosted_repositories (
        community_id, provider_kind, provider_instance, provider_owner, provider_repository
    )
    WHERE authority_kind = 'external_provider';

CREATE TABLE public.collaboration_git_storage_handles (
    community_id uuid NOT NULL,
    storage_handle_id uuid NOT NULL CHECK (
        storage_handle_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    repository_id uuid NOT NULL,
    authority_kind text NOT NULL DEFAULT 'sim_hosted_nip34' CHECK (
        authority_kind = 'sim_hosted_nip34'
    ),
    handle_version numeric(20, 0) NOT NULL CHECK (handle_version >= 1),
    lifecycle_state text NOT NULL CHECK (lifecycle_state IN ('active', 'archived')),
    archived_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, storage_handle_id),
    UNIQUE (community_id, repository_id),
    FOREIGN KEY (community_id, repository_id, authority_kind)
        REFERENCES public.collaboration_hosted_repositories (
            community_id, repository_id, authority_kind
        ) ON DELETE RESTRICT,
    CHECK (
        (lifecycle_state = 'active' AND archived_at IS NULL)
        OR (lifecycle_state = 'archived' AND archived_at IS NOT NULL)
    ),
    CHECK (updated_at >= created_at),
    CHECK (archived_at IS NULL OR archived_at >= created_at)
);

CREATE INDEX collaboration_git_storage_handles_repository
    ON public.collaboration_git_storage_handles (
        community_id, repository_id, lifecycle_state, storage_handle_id
    );

CREATE TABLE public.collaboration_git_repository_grants (
    community_id uuid NOT NULL,
    repository_id uuid NOT NULL,
    grantee_principal_id uuid NOT NULL,
    permission text NOT NULL CHECK (permission IN ('read', 'write', 'admin')),
    grant_version numeric(20, 0) NOT NULL CHECK (grant_version >= 1),
    grant_state text NOT NULL CHECK (grant_state IN ('active', 'revoked')),
    granted_by_principal_id uuid NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (
        community_id, repository_id, grantee_principal_id, permission
    ),
    FOREIGN KEY (community_id, repository_id)
        REFERENCES public.collaboration_hosted_repositories (community_id, repository_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, grantee_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (community_id, granted_by_principal_id)
        REFERENCES public.collaboration_community_memberships (community_id, principal_id)
        ON DELETE RESTRICT,
    CHECK (
        (grant_state = 'active' AND revoked_at IS NULL)
        OR (grant_state = 'revoked' AND revoked_at IS NOT NULL)
    ),
    CHECK (updated_at >= created_at),
    CHECK (revoked_at IS NULL OR revoked_at >= created_at)
);

CREATE INDEX collaboration_git_repository_grants_grantee
    ON public.collaboration_git_repository_grants (
        community_id, grantee_principal_id, grant_state, permission, repository_id
    );

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'collaboration_hosted_repositories',
        'collaboration_git_storage_handles',
        'collaboration_git_repository_grants'
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
