CREATE TABLE public.collaboration_command_receipts (
    community_id uuid NOT NULL,
    operation_id uuid NOT NULL,
    contract_version integer NOT NULL CHECK (
        contract_version BETWEEN 1 AND 65535
    ),
    principal_id uuid NOT NULL,
    originating_adapter text NOT NULL CHECK (
        originating_adapter IN ('nostr_in_process', 'nostr_temporary_sidecar')
    ),
    command_kind text NOT NULL CHECK (
        octet_length(command_kind) BETWEEN 1 AND 128
    ),
    command_fingerprint bytea NOT NULL CHECK (
        octet_length(command_fingerprint) = 32
    ),
    expected_version numeric(20, 0) CHECK (
        expected_version IS NULL
        OR expected_version BETWEEN 1 AND 18446744073709551615
    ),
    predecessor_version numeric(20, 0) CHECK (
        predecessor_version IS NULL
        OR predecessor_version BETWEEN 1 AND 18446744073709551615
    ),
    authoritative_version numeric(20, 0) CHECK (
        authoritative_version IS NULL
        OR authoritative_version BETWEEN 1 AND 18446744073709551615
    ),
    accepted_at timestamptz,
    PRIMARY KEY (community_id, operation_id),
    CHECK ((authoritative_version IS NULL) = (accepted_at IS NULL))
);

CREATE TABLE public.collaboration_outbox (
    community_id uuid NOT NULL,
    outbox_sequence bigint GENERATED ALWAYS AS IDENTITY,
    operation_id uuid NOT NULL,
    authoritative_version numeric(20, 0) NOT NULL CHECK (
        authoritative_version BETWEEN 1 AND 18446744073709551615
    ),
    topic text NOT NULL CHECK (octet_length(topic) BETWEEN 1 AND 128),
    source_system text NOT NULL CHECK (
        source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')
    ),
    source_record_id text NOT NULL CHECK (
        octet_length(source_record_id) BETWEEN 1 AND 1024
    ),
    source_version text CHECK (
        source_version IS NULL OR octet_length(source_version) BETWEEN 1 AND 1024
    ),
    source_observed_at timestamptz NOT NULL,
    source_integrity_algorithm text CHECK (
        source_integrity_algorithm IS NULL OR source_integrity_algorithm IN (
            'sha256', 'nostr_event_id', 'git_object_id'
        )
    ),
    source_integrity_value text CHECK (
        source_integrity_value IS NULL
        OR octet_length(source_integrity_value) BETWEEN 1 AND 1024
    ),
    payload bytea NOT NULL CHECK (octet_length(payload) <= 1048576),
    delivery_state text NOT NULL DEFAULT 'pending' CHECK (
        delivery_state IN ('pending', 'leased', 'delivered', 'failed')
    ),
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    delivered_at timestamptz,
    last_error text CHECK (last_error IS NULL OR octet_length(last_error) <= 2048),
    PRIMARY KEY (community_id, outbox_sequence),
    UNIQUE (community_id, operation_id),
    FOREIGN KEY (community_id, operation_id)
        REFERENCES public.collaboration_command_receipts (community_id, operation_id),
    CHECK (
        (source_integrity_algorithm IS NULL) = (source_integrity_value IS NULL)
    ),
    CHECK (
        (delivery_state = 'delivered') = (delivered_at IS NOT NULL)
    )
);

CREATE INDEX collaboration_command_receipts_principal
    ON public.collaboration_command_receipts (community_id, principal_id, accepted_at);

CREATE INDEX collaboration_outbox_delivery
    ON public.collaboration_outbox (
        community_id,
        delivery_state,
        available_at,
        outbox_sequence
    );

ALTER TABLE public.collaboration_command_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_command_receipts FORCE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_outbox FORCE ROW LEVEL SECURITY;

CREATE POLICY collaboration_command_receipts_admission
    ON public.collaboration_command_receipts
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_command_receipts_community
    ON public.collaboration_command_receipts
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );

CREATE POLICY collaboration_outbox_admission
    ON public.collaboration_outbox
    AS PERMISSIVE
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY collaboration_outbox_community
    ON public.collaboration_outbox
    AS RESTRICTIVE
    FOR ALL
    USING (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    )
    WITH CHECK (
        community_id = NULLIF(current_setting('app.community_id', true), '')::uuid
    );
