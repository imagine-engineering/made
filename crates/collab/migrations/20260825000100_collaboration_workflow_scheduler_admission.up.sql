CREATE TABLE public.collaboration_workflow_ready_queue_index (
    entry_key bytea PRIMARY KEY CHECK (octet_length(entry_key) = 32),
    queued_at timestamptz NOT NULL
);

REVOKE ALL ON public.collaboration_workflow_ready_queue_index FROM PUBLIC;

ALTER TABLE public.collaboration_workflow_runs NO FORCE ROW LEVEL SECURITY;
INSERT INTO public.collaboration_workflow_ready_queue_index (entry_key, queued_at)
SELECT
    sha256(uuid_send(community_id) || uuid_send(run_id)),
    updated_at
FROM public.collaboration_workflow_runs
WHERE status = 'queued';
ALTER TABLE public.collaboration_workflow_runs FORCE ROW LEVEL SECURITY;

CREATE FUNCTION public.collaboration_workflow_admit_ready_queue()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    community_queue_depth bigint;
    deployment_queue_depth bigint;
BEGIN
    IF NEW.status <> 'queued'
        OR (TG_OP = 'UPDATE' AND OLD.status = 'queued') THEN
        RETURN NEW;
    END IF;

    PERFORM pg_advisory_xact_lock(7449358843737115665);

    SELECT count(*)
    INTO community_queue_depth
    FROM public.collaboration_workflow_runs
    WHERE community_id = NEW.community_id
      AND status = 'queued';

    IF community_queue_depth >= 1000 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'workflow_scheduler_capacity_unavailable:community_queue';
    END IF;

    SELECT count(*)
    INTO deployment_queue_depth
    FROM public.collaboration_workflow_ready_queue_index;

    IF deployment_queue_depth >= 10000 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'workflow_scheduler_capacity_unavailable:deployment_queue';
    END IF;

    INSERT INTO public.collaboration_workflow_ready_queue_index (
        entry_key,
        queued_at
    ) VALUES (
        sha256(uuid_send(NEW.community_id) || uuid_send(NEW.run_id)),
        NEW.updated_at
    );

    RETURN NEW;
END;
$$;

CREATE FUNCTION public.collaboration_workflow_release_ready_queue()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    removed_entries bigint;
BEGIN
    IF OLD.status <> 'queued'
        OR (TG_OP = 'UPDATE' AND NEW.status = 'queued') THEN
        RETURN COALESCE(NEW, OLD);
    END IF;

    PERFORM pg_advisory_xact_lock(7449358843737115665);

    DELETE FROM public.collaboration_workflow_ready_queue_index
    WHERE entry_key = sha256(
        uuid_send(OLD.community_id) || uuid_send(OLD.run_id)
    );
    GET DIAGNOSTICS removed_entries = ROW_COUNT;

    IF removed_entries <> 1 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'workflow_scheduler_queue_index_divergence';
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$;

CREATE FUNCTION public.collaboration_workflow_observe_ready_queue(
    requested_community_id uuid
)
RETURNS TABLE (
    community_queue_depth bigint,
    community_oldest_at_millis bigint,
    deployment_queue_depth bigint,
    deployment_oldest_at_millis bigint
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
BEGIN
    IF NULLIF(current_setting('app.community_id', true), '')::uuid
        IS DISTINCT FROM requested_community_id THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'workflow_scheduler_tenant_context_mismatch';
    END IF;

    RETURN QUERY
    SELECT
        (
            SELECT count(*)::bigint
            FROM public.collaboration_workflow_runs
            WHERE community_id = requested_community_id
              AND status = 'queued'
        ),
        (
            SELECT floor(extract(epoch FROM min(updated_at)) * 1000)::bigint
            FROM public.collaboration_workflow_runs
            WHERE community_id = requested_community_id
              AND status = 'queued'
        ),
        (
            SELECT count(*)::bigint
            FROM public.collaboration_workflow_ready_queue_index
        ),
        (
            SELECT floor(extract(epoch FROM min(queued_at)) * 1000)::bigint
            FROM public.collaboration_workflow_ready_queue_index
        );
END;
$$;

CREATE FUNCTION public.collaboration_workflow_admit_execution()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    community_execution_count bigint;
    definition_execution_count bigint;
BEGIN
    IF NEW.state <> 'active' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'collaboration-workflow-execution:' || NEW.community_id::text,
            0
        )
    );

    SELECT
        count(*),
        count(*) FILTER (WHERE run.workflow_id = candidate.workflow_id)
    INTO community_execution_count, definition_execution_count
    FROM public.collaboration_workflow_run_leases AS lease
    JOIN public.collaboration_workflow_runs AS run
      ON run.community_id = lease.community_id
     AND run.run_id = lease.run_id
    CROSS JOIN (
        SELECT workflow_id
        FROM public.collaboration_workflow_runs
        WHERE community_id = NEW.community_id
          AND run_id = NEW.run_id
    ) AS candidate
    WHERE lease.community_id = NEW.community_id
      AND lease.state = 'active'
      AND lease.recovery_after > statement_timestamp();

    IF community_execution_count >= 16 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'workflow_scheduler_capacity_unavailable:community_execution';
    END IF;

    IF definition_execution_count >= 4 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'workflow_scheduler_capacity_unavailable:definition_execution';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER collaboration_workflow_runs_ready_queue_admission
BEFORE INSERT OR UPDATE OF status
ON public.collaboration_workflow_runs
FOR EACH ROW
EXECUTE FUNCTION public.collaboration_workflow_admit_ready_queue();

CREATE TRIGGER collaboration_workflow_runs_ready_queue_release
AFTER UPDATE OF status OR DELETE
ON public.collaboration_workflow_runs
FOR EACH ROW
EXECUTE FUNCTION public.collaboration_workflow_release_ready_queue();

CREATE TRIGGER collaboration_workflow_run_leases_execution_admission
BEFORE INSERT
ON public.collaboration_workflow_run_leases
FOR EACH ROW
EXECUTE FUNCTION public.collaboration_workflow_admit_execution();
