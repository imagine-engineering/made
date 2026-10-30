DROP TRIGGER collaboration_workflow_run_leases_execution_admission
    ON public.collaboration_workflow_run_leases;
DROP TRIGGER collaboration_workflow_runs_ready_queue_release
    ON public.collaboration_workflow_runs;
DROP TRIGGER collaboration_workflow_runs_ready_queue_admission
    ON public.collaboration_workflow_runs;
DROP FUNCTION public.collaboration_workflow_admit_execution();
DROP FUNCTION public.collaboration_workflow_observe_ready_queue(uuid);
DROP FUNCTION public.collaboration_workflow_release_ready_queue();
DROP FUNCTION public.collaboration_workflow_admit_ready_queue();
DROP TABLE public.collaboration_workflow_ready_queue_index;
