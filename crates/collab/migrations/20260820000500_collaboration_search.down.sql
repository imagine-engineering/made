DROP TABLE public.collaboration_search_documents;
DROP INDEX public.collaboration_events_search_fts;
ALTER TABLE public.collaboration_events DROP COLUMN search_tsv;
