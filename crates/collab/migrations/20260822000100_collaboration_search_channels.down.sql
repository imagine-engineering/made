ALTER TABLE public.collaboration_search_documents NO FORCE ROW LEVEL SECURITY;

DELETE FROM public.collaboration_search_documents
WHERE document_type = 'channel';

ALTER TABLE public.collaboration_search_documents FORCE ROW LEVEL SECURITY;

ALTER TABLE public.collaboration_search_documents
    DROP CONSTRAINT collaboration_search_documents_document_type_check;

ALTER TABLE public.collaboration_search_documents
    ADD CONSTRAINT collaboration_search_documents_document_type_check CHECK (
        document_type IN (
            'profile', 'community', 'project', 'repository', 'task',
            'agent', 'workflow', 'media'
        )
    );
