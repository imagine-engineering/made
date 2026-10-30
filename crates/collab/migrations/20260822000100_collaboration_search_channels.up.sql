ALTER TABLE public.collaboration_search_documents
    DROP CONSTRAINT collaboration_search_documents_document_type_check;

ALTER TABLE public.collaboration_search_documents
    ADD CONSTRAINT collaboration_search_documents_document_type_check CHECK (
        document_type IN (
            'profile', 'community', 'channel', 'project', 'repository', 'task',
            'agent', 'workflow', 'media'
        )
    );
