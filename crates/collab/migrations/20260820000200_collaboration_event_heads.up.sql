CREATE TABLE public.collaboration_event_heads (
    community_id uuid NOT NULL,
    kind integer NOT NULL CHECK (kind BETWEEN 0 AND 65535),
    author_public_key bytea NOT NULL CHECK (octet_length(author_public_key) = 32),
    discriminator text NOT NULL CHECK (octet_length(discriminator) <= 1024),
    head_event_created_at numeric(20, 0) NOT NULL CHECK (
        head_event_created_at BETWEEN 0 AND 18446744073709551615
    ),
    head_event_id bytea NOT NULL CHECK (octet_length(head_event_id) = 32),
    live_event_id bytea CHECK (live_event_id IS NULL OR octet_length(live_event_id) = 32),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, kind, author_public_key, discriminator)
);

CREATE INDEX collaboration_event_heads_live_event
    ON public.collaboration_event_heads (community_id, live_event_id)
    WHERE live_event_id IS NOT NULL;

ALTER TABLE public.collaboration_event_heads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_event_heads FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_event_heads_admission
    ON public.collaboration_event_heads
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_event_heads_community
    ON public.collaboration_event_heads
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
