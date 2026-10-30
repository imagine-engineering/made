ALTER TABLE public.collaboration_events
    ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (
        CASE
            WHEN kind IN (0, 9, 40002, 45001, 45003)
            THEN to_tsvector('simple'::regconfig, content)
            ELSE NULL::tsvector
        END
    ) STORED;

CREATE INDEX collaboration_events_search_fts
    ON public.collaboration_events
    USING GIN (search_tsv)
    WHERE search_tsv IS NOT NULL;

CREATE TABLE public.collaboration_search_documents (
    community_id uuid NOT NULL,
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text NOT NULL CHECK (
        octet_length(source_version) BETWEEN 1 AND 1024
    ),
    source_observed_at timestamptz NOT NULL,
    projection_version numeric(20, 0) NOT NULL CHECK (
        projection_version BETWEEN 1 AND 18446744073709551615
    ),
    document_type text NOT NULL CHECK (
        document_type IN (
            'profile', 'community', 'project', 'repository', 'task',
            'agent', 'workflow', 'media'
        )
    ),
    visibility_scope text NOT NULL CHECK (
        visibility_scope IN ('community', 'authorized_restricted', 'excluded')
    ),
    title text NOT NULL DEFAULT '' CHECK (octet_length(title) <= 32768),
    body text NOT NULL CHECK (octet_length(body) <= 262144),
    projected_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    search_tsv tsvector GENERATED ALWAYS AS (
        CASE
            WHEN visibility_scope = 'community'
            THEN
                setweight(to_tsvector('simple'::regconfig, title), 'A')
                || setweight(to_tsvector('simple'::regconfig, body), 'B')
            ELSE NULL::tsvector
        END
    ) STORED,
    PRIMARY KEY (community_id, source_system, source_record_id)
);

CREATE INDEX collaboration_search_documents_tenant_time
    ON public.collaboration_search_documents (
        community_id,
        projected_at DESC,
        source_system,
        source_record_id
    );

CREATE INDEX collaboration_search_documents_fts
    ON public.collaboration_search_documents
    USING GIN (search_tsv)
    WHERE search_tsv IS NOT NULL;

ALTER TABLE public.collaboration_search_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_search_documents FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_search_documents_admission
    ON public.collaboration_search_documents
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_search_documents_community
    ON public.collaboration_search_documents
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
