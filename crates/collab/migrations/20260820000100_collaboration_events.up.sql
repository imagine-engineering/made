CREATE TABLE public.collaboration_events (
    community_id uuid NOT NULL,
    event_id bytea NOT NULL CHECK (octet_length(event_id) = 32),
    author_public_key bytea NOT NULL CHECK (octet_length(author_public_key) = 32),
    event_created_at numeric(20, 0) NOT NULL CHECK (
        event_created_at BETWEEN 0 AND 18446744073709551615
    ),
    kind integer NOT NULL CHECK (kind BETWEEN 0 AND 65535),
    tags jsonb NOT NULL CHECK (jsonb_typeof(tags) = 'array'),
    content text NOT NULL CHECK (octet_length(content) <= 262144),
    canonical_event_bytes bytea NOT NULL CHECK (
        octet_length(canonical_event_bytes) BETWEEN 1 AND 524288
    ),
    signature bytea NOT NULL CHECK (octet_length(signature) = 64),
    signature_state text NOT NULL CHECK (
        signature_state IN ('verified_live', 'verified_historical')
    ),
    verified_at timestamptz NOT NULL,
    persistence_class text NOT NULL CHECK (
        persistence_class IN ('regular', 'replaceable', 'parameterized_replaceable')
    ),
    discriminator text CHECK (
        discriminator IS NULL OR octet_length(discriminator) <= 1024
    ),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (community_id, event_id),
    CHECK (
        (persistence_class = 'regular' AND discriminator IS NULL)
        OR (persistence_class = 'replaceable' AND discriminator = '')
        OR (persistence_class = 'parameterized_replaceable' AND discriminator IS NOT NULL)
    )
) PARTITION BY HASH (community_id);

DO $$
DECLARE
    partition_index integer;
    partition_name text;
BEGIN
    FOR partition_index IN 0..15 LOOP
        partition_name := format('collaboration_events_p%s', partition_index);
        EXECUTE format(
            'CREATE TABLE public.%I PARTITION OF public.collaboration_events FOR VALUES WITH (MODULUS 16, REMAINDER %s)',
            partition_name,
            partition_index
        );
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', partition_name);
        EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', partition_name);
        EXECUTE format(
            'CREATE POLICY collaboration_events_admission ON public.%I AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true)',
            partition_name
        );
        EXECUTE format(
            'CREATE POLICY collaboration_events_community ON public.%I AS RESTRICTIVE FOR ALL USING (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid) WITH CHECK (community_id = NULLIF(current_setting(''app.community_id'', true), '''')::uuid)',
            partition_name
        );
    END LOOP;
END;
$$;

CREATE INDEX collaboration_events_chronological
    ON public.collaboration_events (
        community_id,
        event_created_at DESC,
        event_id ASC
    );

CREATE INDEX collaboration_events_kind_chronological
    ON public.collaboration_events (
        community_id,
        kind,
        event_created_at DESC,
        event_id ASC
    );

CREATE INDEX collaboration_events_author_kind_chronological
    ON public.collaboration_events (
        community_id,
        author_public_key,
        kind,
        event_created_at DESC,
        event_id ASC
    );

CREATE INDEX collaboration_events_addressable_head
    ON public.collaboration_events (
        community_id,
        kind,
        author_public_key,
        discriminator,
        event_created_at DESC,
        event_id ASC
    )
    WHERE persistence_class IN ('replaceable', 'parameterized_replaceable');

CREATE FUNCTION public.reject_collaboration_event_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'collaboration event records are immutable'
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER collaboration_events_immutable
    BEFORE UPDATE ON public.collaboration_events
    FOR EACH ROW EXECUTE FUNCTION public.reject_collaboration_event_update();

ALTER TABLE public.collaboration_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_events FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_events_admission
    ON public.collaboration_events
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_events_community
    ON public.collaboration_events
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
