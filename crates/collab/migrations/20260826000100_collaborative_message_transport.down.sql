DROP TABLE public.collaboration_message_read_states;
DROP TABLE public.collaboration_zed_principal_bindings;
DROP TABLE public.collaboration_zed_channel_bindings;
DROP TABLE public.collaboration_zed_community_bindings;

ALTER TABLE public.collaboration_command_receipts
    DROP CONSTRAINT collaboration_command_receipts_originating_adapter_check;
ALTER TABLE public.collaboration_command_receipts
    ADD CONSTRAINT collaboration_command_receipts_originating_adapter_check CHECK (
        originating_adapter IN ('nostr_in_process', 'nostr_temporary_sidecar')
    );
