use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use agent_client_protocol::schema::v1 as acp;
use anyhow::{Context as _, Result, ensure};
use chrono::Utc;
use gpui::{App, AsyncApp, SharedString, Task, WeakEntity};
use project::trusted_worktrees::TrustedWorktrees;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    AgentTool, Thread, ToolCallEventStream, ToolInput,
    spec_review::{ReviewPass, review},
    spec_workflow::{
        self, ReviewRecord, SpecBinding, SpecProject, SpecificationSession, TaskSession,
        TerminalEvidence, Verification,
    },
};

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Operation {
    Inspect,
    Draft,
    Status,
    Revise,
    Review,
    Approve,
    Prepare,
    BeginTask,
    RecordCheck,
    CompleteTask,
    Report,
    Drift,
    Leave,
}

/// Guide substantial requests from clarification through a project specification, independent
/// review, explicit user approval, implementation tasks, and observed acceptance evidence.
/// Use Inspect to begin a substantial feature. Use the ordinary tools for small edits.
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub(crate) struct SpecWorkflowInput {
    operation: Operation,
    /// Project-relative Markdown path. Omit to use the active or uniquely discovered feature.
    path: Option<String>,
    /// Lowercase feature name with hyphens. Required for Draft.
    slug: Option<String>,
    /// Complete Markdown spec for Draft or Revise.
    specification: Option<String>,
    /// Current revision from Status or Review. Required for Revise and Approve.
    revision: Option<String>,
    /// Project-relative files explicitly relevant to Review or Drift.
    references: Option<Vec<String>>,
    /// Major finding IDs and the reasons agreed with the user. Approve pauses for the user's decision.
    accepted_trade_offs: Option<BTreeMap<String, String>>,
    /// New Markdown plan. Omit for an existing plan.
    plan: Option<String>,
    /// New Markdown task list. Omit for an existing task list.
    tasks: Option<String>,
    /// Optional planning files named research.md, data-model.md, quickstart.md or contracts/<file>.
    supporting_documents: Option<BTreeMap<String, String>>,
    /// A T001-style task ID for BeginTask, RecordCheck, or CompleteTask.
    task: Option<String>,
    /// Acceptance criteria verified by the observed terminal check.
    criteria: Option<Vec<String>>,
    /// Correctness properties explicitly asserted by the observed terminal check.
    properties: Option<Vec<String>>,
    /// The local receipt identifier returned by an actual terminal tool call.
    receipt: Option<String>,
}

#[derive(Clone, Default)]
pub(crate) struct SpecSession {
    pub binding: Option<SpecBinding>,
    pub task: Option<String>,
    pub task_run: Option<Uuid>,
    pub evidence: Vec<TerminalEvidence>,
}

pub(crate) struct SpecWorkflowTool {
    thread: WeakEntity<Thread>,
}

impl SpecWorkflowTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }

    fn bind(
        &self,
        binding: Option<SpecBinding>,
        task: Option<String>,
        cx: &AsyncApp,
    ) -> Result<()> {
        cx.update(|cx| {
            self.thread.update(cx, |thread, cx| {
                let same = thread.spec_session.as_ref().is_some_and(|existing| {
                    (match (&existing.binding, &binding) {
                        (Some(existing), Some(binding)) => {
                            existing.path == binding.path
                                && existing.project.root() == binding.project.root()
                        }
                        _ => false,
                    }) && existing.task == task
                });
                if !same {
                    let path = binding.as_ref().map(|binding| binding.path.clone());
                    if thread.specification_path != path {
                        thread.specification_overview = None;
                    }
                    thread.specification_path = path;
                    thread.spec_session = Some(SpecSession {
                        binding,
                        task_run: task.as_ref().map(|_| Uuid::new_v4()),
                        task,
                        evidence: Vec::new(),
                    });
                    cx.notify();
                }
            })
        })
    }

    async fn run_operation(
        &self,
        mut input: SpecWorkflowInput,
        events: ToolCallEventStream,
        cx: &AsyncApp,
    ) -> Result<String> {
        let (root, previous, model, session_id, editor_task, owning_project) = self.thread.read_with(cx, |thread, cx| {
            ensure!(
                thread.can_use_spec_workflow(cx),
                "Specifications require an interactive built-in agent in a trusted local project"
            );
            Ok::<_, anyhow::Error>((
                spec_root(thread, cx)?,
                thread.spec_session.clone(),
                thread.model().cloned(),
                thread.id().0.to_string(),
                thread.editor_specification_task.clone(),
                thread.project().clone(),
            ))
        })??;
        let session_link = SpecificationSession {
            session_id,
            tool_call_id: events.tool_call_id().0.to_string(),
        };
        if editor_task.is_some() {
            ensure!(
                matches!(
                    input.operation,
                    Operation::Status
                        | Operation::BeginTask
                        | Operation::RecordCheck
                        | Operation::CompleteTask
                        | Operation::Report
                ),
                "Finish or stop the current editor task before changing the workflow"
            );
        }
        if matches!(input.operation, Operation::Leave) {
            ensure!(confirm(&events, "Leave specification workflow", "Return this thread to ordinary chat. The specification and its progress stay in the project.", "Leave workflow", cx).await?,
                "Specification workflow remains active");
            cx.update(|cx| {
                self.thread.update(cx, |thread, cx| {
                    thread.spec_session = None;
                    thread.specification_path = None;
                    thread.specification_overview = None;
                    cx.notify();
                })
            })?;
            return Ok("Returned to ordinary chat.".into());
        }

        let project = cx
            .background_executor()
            .spawn(async move { SpecProject::open(root) })
            .await?;
        if matches!(input.operation, Operation::Inspect) {
            self.bind(None, None, cx)?;
            let principles = cx
                .background_executor()
                .spawn({
                    let project = project.clone();
                    async move { project.principles() }
                })
                .await?;
            return Ok(format!(
                "Specification design is active. Clarify consequential decisions before drafting.\n\n{}",
                principles
                    .into_iter()
                    .map(|document| format!("### {}\n\n{}", document.path, document.text))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            ));
        }
        let path = if matches!(input.operation, Operation::Draft) {
            let slug = input.slug.context("Provide the feature name")?;
            let specification = input
                .specification
                .take()
                .context("Provide the specification Markdown")?;
            self.bind(None, None, cx)?;
            cx.background_executor()
                .spawn({
                    let project = project.clone();
                    async move { project.draft(&slug, &specification) }
                })
                .await?
        } else {
            let path = input.path.or_else(|| {
                previous
                    .as_ref()
                    .and_then(|session| session.binding.as_ref())
                    .filter(|binding| binding.project.root() == project.root())
                    .map(|binding| binding.path.clone())
            });
            cx.background_executor()
                .spawn({
                    let project = project.clone();
                    async move { project.locate(path.as_deref()) }
                })
                .await?
        };
        let binding = SpecBinding {
            project: project.clone(),
            path: path.clone(),
        };
        let active_task = previous
            .as_ref()
            .filter(|session| {
                session
                    .binding
                    .as_ref()
                    .is_some_and(|old| old.path == path && old.project.root() == project.root())
            })
            .and_then(|session| session.task.clone());
        let next_task = match input.operation {
            Operation::BeginTask
            | Operation::Revise
            | Operation::Review
            | Operation::Drift
            | Operation::Draft
            | Operation::Prepare => None,
            _ => active_task.clone(),
        };
        self.bind(Some(binding.clone()), next_task, cx)?;
        cx.update(|cx| ensure_saved(&owning_project, &binding, &[], cx))?;

        let mut snapshot = cx
            .background_executor()
            .spawn({
                let project = project.clone();
                let path = path.clone();
                async move { project.snapshot(&path) }
            })
            .await?;
        if let Some((requested, revision)) = editor_task {
            ensure!(
                snapshot.revision == revision,
                "The approved specification revision changed"
            );
            if matches!(
                input.operation,
                Operation::BeginTask | Operation::RecordCheck | Operation::CompleteTask
            ) {
                ensure!(
                    input.task.as_deref() == Some(requested.as_str()),
                    "The editor selected task {requested}"
                );
            }
        }
        let output = match input.operation {
            Operation::Draft | Operation::Status => {
                cx.background_executor()
                    .spawn(async move { project.report(&snapshot) })
                    .await?
            }
            Operation::Revise => {
                ensure!(
                    input.revision.as_deref() == Some(snapshot.revision.as_str()),
                    "Reload the current revision before editing"
                );
                let text = input
                    .specification
                    .context("Provide the revised specification Markdown")?;
                cx.background_executor()
                    .spawn(async move {
                        project.revise(&snapshot, &text)?;
                        project.report(&project.snapshot(&path)?)
                    })
                    .await?
            }
            Operation::Review | Operation::Drift => {
                let model = model.context(
                    "Select an OpenAI-compatible Chat Completions model for independent review",
                )?;
                ensure!(
                    cx.update(|cx| model.supports_isolated_review(cx)),
                    "This model has no isolated review transport. Select an OpenAI-compatible Chat Completions model without custom headers."
                );
                let drift = matches!(input.operation, Operation::Drift);
                let references = input.references.unwrap_or_default();
                ensure!(
                    !drift || !references.is_empty(),
                    "Select current code or tests for the drift check"
                );
                cx.update(|cx| ensure_saved(&owning_project, &binding, &references, cx))?;
                let references = cx
                    .background_executor()
                    .spawn({
                        let project = project.clone();
                        let path = path.clone();
                        async move { project.references(&references, &path) }
                    })
                    .await?;
                let reference_names = references
                    .iter()
                    .map(|document| document.path.as_str())
                    .collect::<Vec<_>>();
                let materials = if reference_names.is_empty() {
                    "Specification only".to_owned()
                } else {
                    format!("Specification and {}", reference_names.join(", "))
                };
                let provider = model.provider_id().0.to_string();
                let model_id = model.id().0.to_string();
                if !references.is_empty() {
                    ensure!(
                        confirm(
                            &events,
                            "Review reference files",
                            &format!("{materials}\n\nReviewer model: {provider} / {model_id}"),
                            "Send for review",
                            cx
                        )
                        .await?,
                        "Review cancelled"
                    );
                }
                cx.update(|cx| ensure_saved(&owning_project, &binding, &[], cx))?;
                let review_passes: &[ReviewPass] = if drift {
                    &[ReviewPass::Drift]
                } else {
                    &ReviewPass::DESIGN
                };
                let mut passes = Vec::new();
                for pass in review_passes {
                    events.update_fields(
                        acp::ToolCallUpdateFields::new()
                            .title(format!(
                                "{} · {}",
                                if drift {
                                    "Check specification drift"
                                } else {
                                    "Review specification"
                                },
                                pass.title()
                            ))
                            .content(vec![acp::ToolCallContent::from(materials.clone())]),
                    );
                    passes.push((
                        *pass,
                        review(
                            model.clone(),
                            *pass,
                            &snapshot.document,
                            &references,
                            &events,
                            cx,
                        )
                        .await?,
                    ));
                }
                cx.update(|cx| ensure_saved(&owning_project, &binding, &[], cx))?;
                let record = ReviewRecord {
                    id: Uuid::new_v4(),
                    revision: snapshot.revision.clone(),
                    reviewed_at: Utc::now(),
                    provider,
                    model: model_id,
                    references: spec_workflow::reference_digests(&references),
                    passes,
                    session: Some(session_link),
                };
                let report = record.markdown();
                snapshot.state.reviews.push(record);
                cx.background_executor()
                    .spawn(async move {
                        project.save(&snapshot)?;
                        Ok::<_, anyhow::Error>(format!(
                            "{materials}\n\nRevision `{}`{report}",
                            snapshot.revision
                        ))
                    })
                    .await?
            }
            Operation::Approve => {
                ensure!(
                    input.revision.as_deref() == Some(snapshot.revision.as_str()),
                    "Specify the current reviewed revision"
                );
                let record = snapshot
                    .current_review()
                    .context("Review this revision before approval")?;
                let mut approval =
                    record.approval(input.accepted_trade_offs.unwrap_or_default())?;
                let trade_offs = approval
                    .accepted_trade_offs
                    .iter()
                    .map(|(id, reason)| format!("- {id}: {reason}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let message = format!(
                    "{}\n\nRevision `{}`\n\n{trade_offs}{}",
                    path,
                    snapshot.revision,
                    record.markdown()
                );
                ensure!(
                    confirm(
                        &events,
                        "Approve reviewed specification",
                        &message,
                        "Approve revision",
                        cx
                    )
                    .await?,
                    "Approval withheld. Discuss changes in chat and revise the specification."
                );
                approval.approved_at = Utc::now();
                cx.update(|cx| ensure_saved(&owning_project, &binding, &[], cx))?;
                snapshot.state.approvals.push(approval);
                cx.background_executor()
                    .spawn(async move {
                        project.save(&snapshot)?;
                        project.report(&snapshot)
                    })
                    .await?
            }
            Operation::Prepare => {
                cx.background_executor()
                    .spawn(async move {
                        if let Some(documents) = input.supporting_documents {
                            project.write_supporting_documents(&snapshot, documents)?;
                        }
                        project.prepare(
                            &mut snapshot,
                            input.plan.as_deref(),
                            input.tasks.as_deref(),
                        )?;
                        Ok::<_, anyhow::Error>(
                            "Plan and tasks registered against the approved revision.".into(),
                        )
                    })
                    .await?
            }
            Operation::BeginTask => {
                let task = input.task.context("Select a task ID")?;
                let description = cx
                    .background_executor()
                    .spawn({
                        let project = project.clone();
                        let task = task.clone();
                        async move {
                            let description = project.registered_task(&snapshot, &task)?;
                            snapshot.state.task_sessions.insert(
                                task,
                                TaskSession {
                                    revision: snapshot.revision.clone(),
                                    session: session_link,
                                },
                            );
                            project.save(&snapshot)?;
                            Ok::<_, anyhow::Error>(description)
                        }
                    })
                    .await?;
                self.bind(Some(binding), Some(task.clone()), cx)?;
                format!("Task {task} is ready. {description}")
            }
            Operation::RecordCheck => {
                let task = input.task.context("Select the active task ID")?;
                ensure!(
                    active_task.as_deref() == Some(task.as_str()),
                    "Start this task before recording evidence"
                );
                let receipt = input.receipt.context("Provide the terminal receipt ID")?;
                let evidence = previous
                    .as_ref()
                    .and_then(|session| session.evidence.iter().find(|item| item.id == receipt))
                    .cloned()
                    .context("That terminal receipt does not belong to the active task")?;
                let criteria = input
                    .criteria
                    .context("Specify the acceptance criteria this check covers")?;
                let properties = input.properties.unwrap_or_default();
                cx.background_executor()
                    .spawn(async move {
                        let check = Verification {
                            revision: snapshot.revision.clone(),
                            task,
                            criteria,
                            properties,
                            evidence,
                        };
                        project.record_verification(&mut snapshot, check)?;
                        project.write_verification(&project.snapshot(&path)?)
                    })
                    .await?
            }
            Operation::CompleteTask => {
                let task = input.task.context("Select the active task ID")?;
                ensure!(
                    active_task.as_deref() == Some(task.as_str()),
                    "Start this task before completing it"
                );
                let output = cx
                    .background_executor()
                    .spawn(async move {
                        project.complete_task(&snapshot, &task)?;
                        project.report(&snapshot)
                    })
                    .await?;
                self.bind(Some(binding), None, cx)?;
                output
            }
            Operation::Report => {
                cx.background_executor()
                    .spawn(async move { project.write_verification(&snapshot) })
                    .await?
            }
            Operation::Inspect | Operation::Leave => {
                unreachable!("Operations handled before loading a specification")
            }
        };
        let display = self.thread.read_with(cx, |thread, cx| {
            let binding = thread.spec_session.as_ref()?.binding.as_ref()?;
            thread.can_use_spec_workflow(cx).then(|| binding.clone())
        })?;
        if let Some(binding) = display {
            let overview = cx
                .background_executor()
                .spawn({
                    let binding = binding.clone();
                    async move {
                        let snapshot = binding.project.snapshot(&binding.path)?;
                        binding.project.overview(&snapshot)
                    }
                })
                .await?;
            cx.update(|cx| {
                self.thread.update(cx, |thread, cx| {
                    if thread.specification_path.as_deref() == Some(binding.path.as_str()) {
                        thread.specification_overview = Some(overview);
                        cx.notify();
                    }
                })
            })?;
        }
        Ok(output)
    }
}

pub(crate) fn ensure_saved(
    owning_project: &gpui::Entity<project::Project>,
    binding: &SpecBinding,
    references: &[String],
    cx: &App,
) -> Result<()> {
    let project = owning_project.read(cx);
    ensure!(
        project.is_local()
            && !TrustedWorktrees::has_restricted_worktrees(&project.worktree_store(), cx),
        "Workspace trust or project availability changed"
    );
    let directory = std::path::Path::new(&binding.path)
        .parent()
        .context("Invalid project file")?;
    let mut paths = vec![
        PathBuf::from(&binding.path),
        PathBuf::from(".specify/memory/constitution.md"),
        PathBuf::from("AGENTS.md"),
        PathBuf::from(".rules"),
    ];
    paths.extend(
        [
            "plan.md",
            "tasks.md",
            "made-workflow.json",
            "verification.md",
        ]
        .map(|file| directory.join(file)),
    );
    paths.extend(references.iter().map(PathBuf::from));
    for path in paths {
        if let Some(buffer) = project
            .find_project_path(binding.project.root().join(&path), cx)
            .and_then(|path| project.buffer_store().read(cx).get_by_path(&path))
        {
            ensure!(
                !buffer.read(cx).is_dirty(),
                "Save {} before continuing this workflow",
                path.display()
            );
        }
    }
    Ok(())
}

impl AgentTool for SpecWorkflowTool {
    type Input = SpecWorkflowInput;
    type Output = String;
    const NAME: &'static str = "specification_workflow";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn allow_in_restricted_mode() -> bool {
        false
    }

    fn initial_title(
        &self,
        _: Result<Self::Input, serde_json::Value>,
        _: &mut App,
    ) -> SharedString {
        "Specification workflow".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        events: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let result = async {
                let input = input.recv().await?;
                ensure!(
                    !events.was_cancelled_by_user(),
                    "Specification operation cancelled"
                );
                self.run_operation(input, events, cx).await
            }
            .await;
            result.map_err(|error| format!("{error:#}"))
        })
    }
}

pub(crate) fn spec_root(thread: &Thread, cx: &App) -> Result<PathBuf> {
    let project = thread.project().read(cx);
    let mut worktrees = project.visible_worktrees(cx).filter_map(|worktree| {
        let worktree = worktree.read(cx);
        worktree
            .as_local()
            .filter(|local| local.root_entry().is_some_and(|entry| entry.is_dir()))
            .map(|_| worktree.abs_path().to_path_buf())
    });
    let root = worktrees.next().context("Open a local project folder")?;
    ensure!(
        worktrees.next().is_none(),
        "Open a single project folder to manage its specifications"
    );
    ensure!(
        !TrustedWorktrees::has_restricted_worktrees(&project.worktree_store(), cx),
        "Trust this project before using the specification workflow"
    );
    Ok(root)
}

async fn confirm(
    events: &ToolCallEventStream,
    title: &str,
    message: &str,
    accept: &str,
    cx: &AsyncApp,
) -> Result<bool> {
    let accepted = acp::PermissionOptionId::new("specification-confirm");
    let options = vec![
        acp::PermissionOption::new(
            accepted.clone(),
            accept.to_owned(),
            acp::PermissionOptionKind::AllowOnce,
        ),
        acp::PermissionOption::new(
            acp::PermissionOptionId::new("specification-cancel"),
            "Cancel",
            acp::PermissionOptionKind::RejectOnce,
        ),
    ];
    let result = cx
        .update(|cx| {
            events.prompt_for_decision(
                Some(title.to_owned()),
                Some(message.to_owned()),
                options,
                cx,
            )
        })
        .await?;
    Ok(result == accepted && !events.was_cancelled_by_user())
}
