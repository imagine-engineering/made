use crate::recipe_schedule::{RecipeRunStatus, RecipeSchedule, RecipeScheduleState};
use acp_thread::AgentConnection as _;
use acp_thread::ClientUserMessageId;
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Context as _, Result, bail, ensure};
use chrono::{DateTime, Utc};
use collections::{HashMap, HashSet};
use fs::Fs;
use gpui::{App, AppContext as _, Context, Entity, Global, Task, TaskExt as _, WeakEntity};
use project::{Project, ProjectPath, trusted_worktrees::TrustedWorktrees};
use serde::{Deserialize, Serialize};
use std::rc::Rc;
use std::sync::Arc;
use util::ResultExt as _;
use util::rel_path::RelPath;
use uuid::Uuid;

struct GlobalRecipeStore(Entity<RecipeStore>);
impl Global for GlobalRecipeStore {}

pub enum RecipeStoreEvent {
    Review {
        project: Entity<Project>,
        proposal: Rc<RecipeProposal>,
    },
    Manage {
        project: Entity<Project>,
        schedule: Option<Uuid>,
    },
    OpenSession {
        project: Entity<Project>,
        session_id: acp::SessionId,
    },
    RunFinished {
        schedule: RecipeSchedule,
        status: RecipeRunStatus,
    },
}
impl gpui::EventEmitter<RecipeStoreEvent> for RecipeStore {}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct RecipeWorkflowTimeout(String);

#[derive(Debug, thiserror::Error)]
#[error("The scheduled project, trust or provider changed before execution")]
struct ScheduleNeedsAttention;

pub(crate) fn needs_attention(error: &anyhow::Error) -> bool {
    use language_model::LanguageModelCompletionError;
    error.is::<ScheduleNeedsAttention>()
        || error.is::<crate::NoModelConfiguredError>()
        || matches!(
            error.downcast_ref::<LanguageModelCompletionError>(),
            Some(
                LanguageModelCompletionError::NoApiKey { .. }
                    | LanguageModelCompletionError::AuthenticationError { .. }
                    | LanguageModelCompletionError::PermissionError { .. }
                    | LanguageModelCompletionError::DataRetentionConsentRequired { .. }
                    | LanguageModelCompletionError::ApiEndpointNotFound { .. }
            )
        )
        || error
            .downcast_ref::<acp::Error>()
            .is_some_and(|error| error.code == acp::ErrorCode::AuthRequired)
}

/// Draft a Made recipe and optionally a recurring schedule for desktop review.
/// Ask the user about materially ambiguous instructions, required parameters,
/// worktree, recurrence or time zone before calling. For example, weekday 9 AM
/// is `0 9 * * 1-5`, but the IANA time zone must be known, not guessed.
/// Supply either native version-1 recipe YAML or an existing ID from /recipes.
/// YAML supports title, description, prompt, instructions, parameters and command;
/// parameters use literal {{name}} placeholders. Never include secrets.
/// This tool does not save, run or enable anything. The user must save and explicitly
/// enable unattended execution in the management UI. Report only the returned status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecipeProposalInput {
    /// Stable name for this proposal in this conversation; reuse on repeated calls.
    pub proposal_id: String,
    pub recipe_yaml: Option<String>,
    /// Existing discovery ID, qualified recipe command, or an unambiguous recipe title.
    pub recipe_id: Option<String>,
    /// Filename for a new local recipe, directly inside .agents/recipes.
    pub filename: Option<String>,
    #[serde(default)]
    pub parameters: HashMap<String, String>,
    /// Absolute visible worktree path. May be omitted only for a single-root project.
    pub worktree: Option<String>,
    /// Provider and model must both be supplied, or both omitted for the current model.
    pub provider: Option<String>,
    pub model: Option<String>,
    /// Omit for recipe-only creation.
    pub schedule: Option<RecipeRecurrenceProposal>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecipeRecurrenceProposal {
    pub title: String,
    pub cron: String,
    pub timezone: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeReviewStatus {
    Pending,
    Opened,
    Dismissed,
    Unavailable,
    Cancelled,
}

#[derive(Clone)]
pub struct RecipeProposal {
    pub input: RecipeProposalInput,
    pub recipe: Recipe,
    pub parameters: HashMap<String, String>,
    pub worktree: project::WorktreeId,
    pub model: language_model::ConfiguredModel,
    pub schedule_id: Uuid,
    pub status: Rc<std::cell::Cell<RecipeReviewStatus>>,
}

impl RecipeProposal {
    pub fn message(&self) -> String {
        match self.status.get() {
            RecipeReviewStatus::Pending => "Draft is waiting for desktop review.",
            RecipeReviewStatus::Opened => "Draft opened in Made Recipes and Schedules for review. This tool has not saved, run or enabled anything. The user must save and explicitly enable unattended execution in that UI.",
            RecipeReviewStatus::Dismissed => "Draft review was closed. This repeated call did not reopen it or create anything. Inspect management for anything the user previously saved.",
            RecipeReviewStatus::Unavailable => "Draft review could not open. Close any active dialog and retry the same proposal. Nothing was saved or enabled by this call.",
            RecipeReviewStatus::Cancelled => "Draft request was cancelled. Nothing was saved or enabled by this call.",
        }.into()
    }
}

pub struct RecipeStore {
    pub scan: Option<crate::recipe_scan::RecipeScanProgress>,
    pub(crate) scan_task: Option<Task<()>>,
    pub scan_cleanup_active: bool,
    pub(crate) scan_cancellation: Option<futures::channel::oneshot::Sender<()>>,
    proposals: Vec<(gpui::EntityId, gpui::EntityId, Rc<RecipeProposal>)>,
    pub state: RecipeScheduleState,
    pub error: Option<String>,
    pub busy: bool,
    revision: i64,
    database: Option<Arc<crate::ThreadsDatabase>>,
    clock: Arc<dyn scheduler::Clock>,
    connections: Vec<(WeakEntity<Project>, WeakEntity<crate::NativeAgent>)>,
    active_session: Option<Entity<acp_thread::AcpThread>>,
    active_schedule: Option<RecipeSchedule>,
    cancellation: Option<RecipeRunStatus>,
    task: Option<Task<()>>,
    timer: Option<Task<()>>,
}

impl RecipeStore {
    pub fn propose(
        &mut self,
        conversation: gpui::EntityId,
        project: Entity<Project>,
        input: RecipeProposalInput,
        current_model: Option<language_model::ConfiguredModel>,
        request_status: Rc<std::cell::Cell<RecipeReviewStatus>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Rc<RecipeProposal>>> {
        let discovery = discover_recipes(project.clone(), project.read(cx).fs().clone(), cx);
        cx.spawn(async move |this, cx| {
            let entries = discovery.await;
            this.update(cx, |this, cx| {
                ensure!(!input.proposal_id.trim().is_empty() && input.proposal_id.len() <= 128, "Use a proposal ID of 1–128 characters");
                for (owner, project_id, proposal) in &mut this.proposals {
                    if *owner != conversation || *project_id != project.entity_id() { continue; }
                    if proposal.input.proposal_id == input.proposal_id {
                        ensure!(proposal.input == input, "Proposal ID already used with different fields; use a new ID for a revised proposal");
                        if proposal.status.get() == RecipeReviewStatus::Unavailable {
                            *proposal = Rc::new(RecipeProposal { status: request_status, ..proposal.as_ref().clone() });
                            cx.emit(RecipeStoreEvent::Review { project, proposal: proposal.clone() });
                        }
                        return Ok(proposal.clone());
                    }
                    let mut previous = proposal.input.clone();
                    previous.proposal_id = input.proposal_id.clone();
                    if previous == input { return Ok(proposal.clone()); }
                }
                ensure!(this.proposals.len() < 128, "Draft review limit reached for this Made session; use recipe management directly");
                ensure!(project.read(cx).is_local(), "Recipe proposals need a local project");
                let worktrees = project.read(cx).visible_worktrees(cx).collect::<Vec<_>>();
                let worktree = if let Some(path) = &input.worktree {
                    worktrees.iter().find(|worktree| worktree.read(cx).abs_path().as_ref() == std::path::Path::new(path)).context("Proposed worktree is not visible in this project")?
                } else {
                    ensure!(worktrees.len() == 1, "Ask the user which worktree to use");
                    worktrees.first().context("Open a project worktree first")?
                };
                let worktree_id = worktree.read(cx).id();
                let worktree_store = project.read(cx).worktree_store();
                ensure!(TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| trusted.update(cx, |trusted, cx| trusted.can_trust(&worktree_store, worktree_id, cx))), "Recipe proposals need a trusted worktree");
                let recipe = match (&input.recipe_yaml, &input.recipe_id) {
                    (Some(yaml), None) => {
                        let filename = input.filename.as_deref().context("Supply a filename for the new recipe")?;
                        ensure!(!filename.contains('/') && !filename.contains('\\'), "Use a filename without directories");
                        let path = ProjectPath { worktree_id, path: RelPath::from_unix_str(&format!(".agents/recipes/{filename}"))?.into() };
                        validate_local_path(&project, &path, cx)?;
                        Recipe::from_yaml(yaml)?
                    }
                    (None, Some(id)) => {
                        ensure!(input.filename.is_none(), "Do not supply a filename for an existing recipe");
                        let matches = entries.iter().filter(|entry| {
                            &entry.id == id || entry.qualified_command() == id.trim_start_matches('/') || entry.recipe.as_ref().is_ok_and(|recipe| &recipe.title == id)
                        }).collect::<Vec<_>>();
                        ensure!(matches.len() == 1, "Recipe is unavailable or ambiguous; ask the user to choose a discovery ID: {}", entries.iter().take(32).map(|entry| entry.id.as_str()).collect::<Vec<_>>().join(", "));
                        matches.first().context("Choose an existing recipe")?.recipe.clone().map_err(anyhow::Error::msg)?
                    }
                    _ => bail!("Supply exactly one of recipe_yaml or recipe_id"),
                };
                let parameters = recipe.resolve_parameters(&input.parameters)?;
                recipe.render(&parameters)?;
                let model = match (&input.provider, &input.model) {
                    (None, None) => current_model.context("Choose an available provider and model")?,
                    (Some(provider), Some(model)) => language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| registry.select_model(&language_model::SelectedModel { provider: provider.clone().into(), model: language_model::LanguageModelId(model.clone().into()) }, cx)).context("Proposed provider/model is unavailable")?,
                    _ => bail!("Supply both provider and model, or neither"),
                };
                if let Some(schedule) = &input.schedule {
                    ensure!(!schedule.title.trim().is_empty(), "Supply a schedule title");
                    crate::recipe_schedule::next_run(&schedule.cron, &schedule.timezone, this.clock.utc_now())?;
                }
                ensure!(request_status.get() == RecipeReviewStatus::Pending, "Recipe draft request cancelled");
                let proposal = Rc::new(RecipeProposal { input, recipe, parameters, worktree: worktree_id, model, schedule_id: Uuid::new_v4(), status: request_status });
                this.proposals.push((conversation, project.entity_id(), proposal.clone()));
                cx.emit(RecipeStoreEvent::Review { project, proposal: proposal.clone() });
                Ok(proposal)
            })?
        })
    }

    pub fn request_management(
        &mut self,
        project: Entity<Project>,
        schedule: Option<Uuid>,
        cx: &mut Context<Self>,
    ) {
        cx.emit(RecipeStoreEvent::Manage { project, schedule });
    }

    pub fn command(
        &mut self,
        project: Entity<Project>,
        arguments: &str,
        cx: &mut Context<Self>,
    ) -> Task<Result<String>> {
        let parsed = (|| {
            let words = crate::split_shell_words(arguments)?;
            let operation = words.first().map(String::as_str).unwrap_or("list");
            if matches!(operation, "create" | "manage") {
                ensure!(words.len() == 1, "Usage: /schedules create");
                self.request_management(project, None, cx);
                return Ok(Task::ready(Ok(
                    "Opened recipe and schedule management.".to_owned()
                )));
            }
            if operation == "list" {
                ensure!(words.len() <= 1, "Usage: /schedules list");
                let output = self
                    .state
                    .schedules
                    .iter()
                    .filter(|schedule| !schedule.deleted)
                    .map(|schedule| {
                        format!(
                            "{} — {}\n{} · {} · next {}\n{}",
                            schedule.id,
                            schedule.title,
                            schedule.cron,
                            schedule.timezone,
                            schedule.next_run,
                            if schedule.paused { "Paused" } else { "Enabled" }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                return Ok(Task::ready(Ok(if output.is_empty() {
                    "No schedules. Use /schedules create to open management.".into()
                } else {
                    output
                })));
            }
            let id = words
                .get(1)
                .context("Supply a schedule ID from /schedules list")?
                .parse::<Uuid>()
                .context("Invalid schedule ID")?;
            ensure!(
                words.len() == 2
                    || (operation == "resume"
                        && words.get(2).is_some_and(|word| word == "--consent")
                        && words.len() == 3),
                "Unexpected schedule arguments"
            );
            let task = match operation {
                "edit" => {
                    self.request_management(project, Some(id), cx);
                    Task::ready(Ok(()))
                }
                "pause" => self.set_enabled(id, false, false, cx),
                "resume" => self.set_enabled(id, true, words.len() == 3, cx),
                "run" => self.run_schedule(id, cx),
                "delete" => self.delete_schedule(id, cx),
                "cancel" => {
                    ensure!(
                        self.state
                            .active_run()
                            .is_some_and(|run| run.schedule_id == id),
                        "This schedule has no active run"
                    );
                    self.cancel_schedule(id, cx)?;
                    Task::ready(Ok(()))
                }
                "history" => {
                    return Ok(Task::ready(Ok(self
                        .state
                        .runs
                        .iter()
                        .filter(|run| run.schedule_id == id)
                        .map(|run| {
                            format!(
                                "{} · {} · {}\nSession: {}",
                                run.id,
                                run.scheduled_for,
                                run.status.label(),
                                run.session_id.as_deref().unwrap_or("not created")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n"))));
                }
                "open" => {
                    let run = self
                        .state
                        .runs
                        .iter()
                        .rev()
                        .find(|run| run.schedule_id == id && run.session_id.is_some())
                        .context("No resulting session is available")?;
                    let session_id = acp::SessionId::new(
                        run.session_id.clone().context("Session is unavailable")?,
                    );
                    cx.emit(RecipeStoreEvent::OpenSession {
                        project,
                        session_id,
                    });
                    Task::ready(Ok(()))
                }
                _ => bail!(
                    "Use /schedules list, create, edit, pause, resume <id> --consent, run, cancel, delete, history or open"
                ),
            };
            Ok(cx.spawn(async move |_, _| {
                task.await?;
                Ok("Schedule updated.".to_owned())
            }))
        })();
        match parsed {
            Ok(task) => task,
            Err(error) => Task::ready(Err(error)),
        }
    }
    pub fn init_global(cx: &mut App) {
        if Self::try_global(cx).is_some() {
            return;
        }
        let store = cx.new(|cx| Self {
            scan: None,
            scan_task: None,
            scan_cleanup_active: false,
            scan_cancellation: None,
            proposals: Vec::new(),
            state: RecipeScheduleState::default(),
            error: None,
            busy: true,
            revision: 0,
            database: None,
            clock: cx
                .background_executor()
                .scheduler_executor()
                .scheduler()
                .clock(),
            connections: Vec::new(),
            active_session: None,
            active_schedule: None,
            cancellation: None,
            task: None,
            timer: None,
        });
        cx.set_global(GlobalRecipeStore(store.clone()));
        store.update(cx, |store, cx| {
            cx.on_app_quit(|store, cx| {
                store.timer.take();
                store.cancel_run(cx);
                store.cancel_scan(cx);
                let schedule = store.task.take().unwrap_or_else(|| Task::ready(()));
                let scan = store.scan_task.take().unwrap_or_else(|| Task::ready(()));
                cx.spawn(async move |_, _| {
                    futures::join!(schedule, scan);
                })
            })
            .detach();
            let database = crate::ThreadsDatabase::connect(cx);
            store.timer = Some(cx.spawn(async move |this, cx| {
                let initialized = async {
                    let database = database.await.map_err(|error| anyhow::anyhow!(error))?;
                    let (revision, mut state) = database.load_recipe_state().await?;
                    let now = this.read_with(cx, |this, _| this.clock.utc_now())?;
                    state.recover(now)?;
                    let revision = database.save_recipe_state(revision, state.clone()).await?;
                    this.update(cx, |this, cx| {
                        this.database = Some(database);
                        this.state = state;
                        this.revision = revision;
                        this.busy = false;
                        this.recover_scans(cx);
                        cx.notify();
                    })?;
                    anyhow::Ok(())
                }
                .await;
                if let Err(error) = initialized {
                    this.update(cx, |this, cx| {
                        this.error = Some(format!("Schedule storage is unavailable: {error}"));
                        this.busy = false;
                        cx.notify();
                    })
                    .log_err();
                    return;
                }
                let mut previous = this.read_with(cx, |this, _| this.clock.utc_now()).ok();
                loop {
                    cx.background_executor()
                        .timer(std::time::Duration::from_secs(1))
                        .await;
                    let Ok(now) = this.read_with(cx, |this, _| this.clock.utc_now()) else {
                        break;
                    };
                    let missed = previous
                        .is_none_or(|last| now.signed_duration_since(last).num_seconds() >= 60);
                    previous = Some(now);
                    let Ok(poll) = this.update(cx, |this, cx| {
                        if this.active_session.is_some() && this.cancellation.is_none() {
                            let valid = this.active_schedule.as_ref().is_some_and(|schedule| {
                                this.connections
                                    .iter()
                                    .filter_map(|(project, _)| project.upgrade())
                                    .any(|project| {
                                        validate_scheduled_project(&project, schedule, cx).is_ok()
                                    })
                            });
                            if !valid {
                                this.cancellation = Some(RecipeRunStatus::NeedsAttention);
                                if let Some(session) = &this.active_session {
                                    session
                                        .update(cx, |session, cx| session.cancel(cx))
                                        .detach();
                                }
                            }
                        }
                        if this.busy || this.error.is_some() {
                            return None;
                        }
                        let due = this.state.schedules.iter().any(|schedule| {
                            !schedule.deleted
                                && !schedule.paused
                                && schedule.consented
                                && schedule.next_run <= now
                        });
                        due.then(|| this.transact(move |state, _| state.tick(now, missed), cx))
                    }) else {
                        break;
                    };
                    if let Some(poll) = poll {
                        match poll.await {
                            Ok(Some(id)) => {
                                this.update(cx, |this, cx| this.start_run(id, cx)).log_err();
                            }
                            Ok(None) => {}
                            Err(error) => log::error!("Schedule admission failed: {error}"),
                        }
                    }
                }
            }));
        });
    }

    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalRecipeStore>()
            .map(|store| store.0.clone())
    }

    pub fn has_active_work(&self, cx: &App) -> bool {
        self.busy
            || self.state.active_run().is_some()
            || self.scan_task.is_some()
            || self.scan_cleanup_active
            || self
                .connections
                .iter()
                .filter_map(|(_, agent)| agent.upgrade())
                .any(|agent| agent.read(cx).has_active_work(cx))
    }

    pub fn register(&mut self, project: &Entity<Project>, agent: &Entity<crate::NativeAgent>) {
        self.connections.retain(|(existing, connection)| {
            existing.upgrade().is_some()
                && connection.upgrade().is_some()
                && existing.entity_id() != project.entity_id()
        });
        self.connections
            .push((project.downgrade(), agent.downgrade()));
    }

    pub(crate) fn transact<T: 'static>(
        &mut self,
        update: impl FnOnce(&mut RecipeScheduleState, DateTime<Utc>) -> Result<T>,
        cx: &mut Context<Self>,
    ) -> Task<Result<T>> {
        if self.busy {
            return Task::ready(Err(anyhow::anyhow!("Schedule storage is busy; try again")));
        }
        let Some(database) = self.database.clone() else {
            return Task::ready(Err(anyhow::anyhow!("Schedule storage is unavailable")));
        };
        if let Some(error) = &self.error {
            return Task::ready(Err(anyhow::anyhow!(error.clone())));
        }
        let mut state = self.state.clone();
        let result = match update(&mut state, self.clock.utc_now()) {
            Ok(result) => result,
            Err(error) => return Task::ready(Err(error)),
        };
        self.busy = true;
        cx.notify();
        let revision = self.revision;
        let (completion, result_receiver) = futures::channel::oneshot::channel();
        cx.spawn(async move |this, cx| {
            let result = async {
                let saved = database.save_recipe_state(revision, state.clone()).await;
                this.update(cx, |this, cx| {
                    this.busy = false;
                    match saved {
                        Ok(revision) => {
                            this.revision = revision;
                            this.state = state;
                        }
                        Err(ref error) => {
                            this.error = Some(format!("Could not save schedules: {error}"));
                        }
                    }
                    cx.notify();
                })?;
                saved?;
                Ok(result)
            }
            .await;
            if completion.send(result).is_err() {
                log::debug!("Recipe state transaction completed after its caller stopped waiting");
            }
        })
        .detach();
        cx.spawn(async move |_, _| {
            result_receiver
                .await
                .context("Recipe state transaction was interrupted")?
        })
    }

    pub fn save_schedule(
        &mut self,
        schedule: RecipeSchedule,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.transact(move |state, now| state.save_schedule(schedule, now), cx)
    }

    pub fn set_enabled(
        &mut self,
        id: Uuid,
        enabled: bool,
        consent: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.transact(
            move |state, now| state.set_enabled(id, enabled, consent, now),
            cx,
        )
    }

    pub fn run_schedule(&mut self, id: Uuid, cx: &mut Context<Self>) -> Task<Result<()>> {
        let admitted = self.transact(move |state, now| state.admit(id, now), cx);
        cx.spawn(async move |this, cx| {
            let id = admitted.await?;
            this.update(cx, |this, cx| this.start_run(id, cx))?;
            Ok(())
        })
    }

    pub fn cancel_schedule(&mut self, id: Uuid, cx: &mut Context<Self>) -> Result<()> {
        ensure!(
            self.state
                .active_run()
                .is_some_and(|run| run.schedule_id == id),
            "This schedule has no active run"
        );
        self.cancel_run(cx);
        Ok(())
    }

    fn cancel_run(&mut self, cx: &mut Context<Self>) {
        self.cancellation = Some(RecipeRunStatus::Cancelled);
        if let Some(session) = &self.active_session {
            session
                .update(cx, |session, cx| session.cancel(cx))
                .detach();
        }
    }

    pub fn delete_schedule(&mut self, id: Uuid, cx: &mut Context<Self>) -> Task<Result<()>> {
        let deleted = self.transact(
            move |state, _| {
                let schedule = state
                    .schedules
                    .iter_mut()
                    .find(|schedule| schedule.id == id && !schedule.deleted)
                    .context("Schedule is unavailable")?;
                schedule.deleted = true;
                schedule.paused = true;
                Ok(())
            },
            cx,
        );
        cx.spawn(async move |this, cx| {
            deleted.await?;
            this.update(cx, |this, cx| {
                if this
                    .state
                    .active_run()
                    .is_some_and(|run| run.schedule_id == id)
                {
                    this.cancel_run(cx);
                }
            })?;
            Ok(())
        })
    }

    fn start_run(&mut self, id: Uuid, cx: &mut Context<Self>) {
        self.task = Some(cx.spawn(async move |this, cx| {
            let mut started = false;
            let mut cleanup = None;
            let mut final_thread = None;
            let result = async {
                let schedule = this.read_with(cx, |this, _| {
                    let run = this.state.runs.iter().find(|run| run.id == id).context("Run is unavailable")?;
                    this.state.schedules.iter().find(|schedule| schedule.id == run.schedule_id).cloned().context("Schedule is unavailable")
                })??;
                let (project, agent, model) = this.update(cx, |this, cx| {
                    ensure!(this.cancellation.is_none() && !schedule.deleted, "Run was cancelled");
                    let paths = util::path_list::PathList::deserialize(&schedule.project_paths);
                    let (project, agent) = this.connections.iter().find_map(|(project, agent)| {
                        let project = project.upgrade()?;
                        let agent = agent.upgrade()?;
                        let roots = project.read(cx).visible_worktrees(cx).map(|worktree| worktree.read(cx).abs_path().to_path_buf()).collect::<Vec<_>>();
                        (util::path_list::PathList::new(&roots) == paths).then_some((project, agent))
                    }).context("The scheduled project is not open in Made")?;
                    ensure!(project.read(cx).is_local(), "Scheduled recipes require a local project");
                    let worktree = project.read(cx).visible_worktrees(cx).find(|worktree| worktree.read(cx).abs_path().as_ref() == schedule.worktree.as_path()).context("The scheduled worktree is unavailable")?;
                    let worktree_store = project.read(cx).worktree_store();
                    let trusted = TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| trusted.update(cx, |trusted, cx| trusted.can_trust(&worktree_store, worktree.read(cx).id(), cx)));
                    ensure!(trusted, "The scheduled worktree needs trust approval");
                    let configured = language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| registry.select_model(&language_model::SelectedModel {
                        provider: language_model::LanguageModelProviderId(schedule.provider.clone().into()),
                        model: language_model::LanguageModelId(schedule.model.clone().into()),
                    }, cx)).context("The scheduled provider or model is unavailable")?;
                    ensure!(configured.provider.is_authenticated(cx), "The scheduled provider needs credentials");
                    anyhow::Ok((project, agent, configured.model))
                })??;
                let fs = agent.read_with(cx, |agent, _| agent.fs.clone());
                ensure!(fs.metadata(&schedule.worktree).await?.is_some_and(|metadata| metadata.is_dir), "The scheduled worktree no longer exists");
                let connection = Rc::new(crate::NativeAgentConnection(agent.clone()));
                let session = agent.update(cx, |agent, cx| agent.new_session(project.clone(), cx));
                let session_id = session.read_with(cx, |session, _| session.session_id().clone());
                cleanup = Some((connection.clone(), session_id.clone()));
                let thread = agent.read_with(cx, |agent, _| agent.sessions.get(&session_id).map(|session| session.thread.clone())).context("Native session is unavailable")?;
                final_thread = Some((thread.clone(), util::path_list::PathList::deserialize(&schedule.project_paths)));
                thread.update(cx, |thread, cx| {
                    thread.set_model(model, cx);
                    thread.set_automatic_retries(false);
                });
                let payload = thread.read_with(cx, |thread, cx| thread.to_db(cx)).await;
                cx.update(|cx| crate::ThreadStore::global(cx).update(cx, |store, cx| store.save_thread(session_id.clone(), payload, util::path_list::PathList::deserialize(&schedule.project_paths), cx))).await?;
                let session_id_for_history = session_id.to_string();
                while this.read_with(cx, |this, _| this.busy)? {
                    cx.background_executor().timer(std::time::Duration::from_millis(10)).await;
                }
                this.update(cx, |this, cx| this.transact(move |state, _| {
                    let run = state.runs.iter_mut().find(|run| run.id == id).context("Run is unavailable")?;
                    run.session_id = Some(session_id_for_history);
                    run.status = RecipeRunStatus::Running;
                    Ok(())
                }, cx))?.await?;
                let subscription = this.update(cx, |this, cx| {
                    this.active_session = Some(session.clone());
                    this.active_schedule = Some(schedule.clone());
                    cx.subscribe(&session, |this, session, event, cx| {
                        let status = match event {
                            acp_thread::AcpThreadEvent::ToolAuthorizationRequested(_) | acp_thread::AcpThreadEvent::ElicitationRequested(_) => Some(RecipeRunStatus::NeedsAttention),
                            acp_thread::AcpThreadEvent::Retry(_) => Some(RecipeRunStatus::Failed),
                            _ => None,
                        };
                        if let Some(status) = status {
                            this.cancellation = Some(status);
                            cx.spawn(async move |_, cx| {
                                session.update(cx, |session, cx| session.cancel(cx)).await;
                            }).detach();
                        }
                    })
                })?;
                ensure!(this.read_with(cx, |this, _| this.cancellation.is_none())?, "Run was cancelled before execution");
                started = true;
                let run = session.update(cx, |session, cx| session.run_turn(cx, async move |_, cx| {
                    cx.update(|cx| {
                        validate_scheduled_project(&project, &schedule, cx).map_err(|_| anyhow::anyhow!(ScheduleNeedsAttention))?;
                        anyhow::Ok(execute_recipe(&connection, session_id, ClientUserMessageId::new(), &schedule.recipe, &schedule.parameters, Vec::new(), cx))
                    })?.await
                }));
                let response = run.await?;
                drop(subscription);
                Ok(response)
            }.await;
            if let Some((thread, paths)) = final_thread {
                thread.update(cx, |thread, _| thread.set_automatic_retries(true));
                let (session_id, payload) = thread.read_with(cx, |thread, cx| (thread.id().clone(), thread.to_db(cx)));
                let payload = payload.await;
                if let Err(error) = cx.update(|cx| crate::ThreadStore::global(cx).update(cx, |store, cx| store.save_thread(session_id, payload, paths, cx))).await {
                    this.update(cx, |this, cx| {
                        this.error = Some(format!("Could not save scheduled session history: {error}. Restart Made after resolving the storage problem."));
                        cx.notify();
                    }).log_err();
                }
            }
            if let Some((connection, session_id)) = cleanup {
                if let Err(error) = cx.update(|cx| connection.close_session(&session_id, cx)).await {
                    log::error!("Could not close scheduled session: {error}");
                }
            }
            let status = this.read_with(cx, |this, _| this.cancellation).ok().flatten().unwrap_or_else(|| {
                match &result {
                    Ok(Some(response)) if response.stop_reason == acp::StopReason::EndTurn => RecipeRunStatus::Succeeded,
                    Ok(None) => RecipeRunStatus::Cancelled,
                    Ok(Some(_)) => RecipeRunStatus::Failed,
                    Err(_) if !started => RecipeRunStatus::NeedsAttention,
                    Err(error) if needs_attention(error) => RecipeRunStatus::NeedsAttention,
                    Err(_) => RecipeRunStatus::Failed,
                }
            });
            let detail = match status {
                RecipeRunStatus::NeedsAttention => Some("The run needs attention. Review project availability, trust, provider credentials and requested permissions before resuming.".to_owned()),
                RecipeRunStatus::Failed => Some("The agent run failed; inspect its session for details. No retry was started.".to_owned()),
                _ => None,
            };
            if let Err(error) = &result { log::error!("Scheduled recipe failed: {error}"); }
            loop {
                let Ok(busy) = this.read_with(cx, |this, _| this.busy) else { return };
                if !busy { break; }
                cx.background_executor().timer(std::time::Duration::from_millis(10)).await;
            }
            if let Ok(saved) = this.update(cx, |this, cx| this.transact(move |state, now| state.finish(id, status, detail, now), cx)) {
                if let Err(error) = saved.await { log::error!("Could not finalize scheduled run: {error}"); }
            }
            this.update(cx, |this, cx| {
                this.active_session = None;
                this.active_schedule = None;
                this.cancellation = None;
                if let Some(schedule) = this.state.runs.iter().find(|run| run.id == id && !run.status.is_active()).and_then(|run| this.state.schedules.iter().find(|schedule| schedule.id == run.schedule_id)).cloned() {
                    cx.emit(RecipeStoreEvent::RunFinished { schedule, status });
                }
                cx.notify();
            }).log_err();
        }));
    }
}

pub const MAX_RECIPE_BYTES: usize = 256 * 1024;
pub const MAX_RENDERED_RECIPE_BYTES: usize = 1024 * 1024;

fn validate_scheduled_project(
    project: &Entity<Project>,
    schedule: &RecipeSchedule,
    cx: &mut App,
) -> Result<()> {
    ensure!(
        project.read(cx).is_local(),
        "Scheduled recipes require a local project"
    );
    let roots = project
        .read(cx)
        .visible_worktrees(cx)
        .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    ensure!(
        util::path_list::PathList::new(&roots)
            == util::path_list::PathList::deserialize(&schedule.project_paths),
        "The scheduled project paths changed"
    );
    let worktree = project
        .read(cx)
        .visible_worktrees(cx)
        .find(|worktree| worktree.read(cx).abs_path().as_ref() == schedule.worktree.as_path())
        .context("The scheduled worktree is unavailable")?;
    ensure!(
        worktree
            .read(cx)
            .root_entry()
            .is_some_and(|entry| entry.is_dir()),
        "The scheduled worktree no longer exists"
    );
    let store = project.read(cx).worktree_store();
    let trusted = TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
        trusted.update(cx, |trusted, cx| {
            trusted.can_trust(&store, worktree.read(cx).id(), cx)
        })
    });
    ensure!(trusted, "The scheduled worktree needs trust approval");
    let registry = language_model::LanguageModelRegistry::read_global(cx);
    let provider = registry
        .provider(&schedule.provider.clone().into())
        .context("The scheduled provider is unavailable")?;
    ensure!(
        provider.is_authenticated(cx),
        "The scheduled provider needs credentials"
    );
    ensure!(
        provider
            .provided_models(cx)
            .iter()
            .any(|model| model.id().0.as_ref() == schedule.model),
        "The scheduled model is unavailable"
    );
    Ok(())
}

fn validate_local_path(
    project: &Entity<Project>,
    path: &ProjectPath,
    cx: &mut App,
) -> Result<std::path::PathBuf> {
    let relative = path.path.as_unix_str();
    let name = relative
        .strip_prefix(".agents/recipes/")
        .context("Choose a file in .agents/recipes")?;
    ensure!(
        !name.contains('/') && (name.ends_with(".yaml") || name.ends_with(".yml")),
        "Choose a YAML file directly in .agents/recipes"
    );
    let worktree = project
        .read(cx)
        .visible_worktrees(cx)
        .find(|worktree| worktree.read(cx).id() == path.worktree_id)
        .context("Recipe worktree is unavailable")?;
    let store = project.read(cx).worktree_store();
    let trusted = TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
        trusted.update(cx, |trusted, cx| {
            trusted.can_trust(&store, path.worktree_id, cx)
        })
    });
    ensure!(
        trusted && project.read(cx).is_local(),
        "Recipe management needs a trusted local worktree"
    );
    let absolute = worktree.read(cx).absolutize(&path.path);
    let roots = project
        .read(cx)
        .visible_worktrees(cx)
        .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    ensure!(
        matches!(crate::tools::resolve_project_path(project.read(cx), &absolute, &roots, cx)?, crate::tools::ResolvedProjectPath::Safe(ref resolved) if resolved == path),
        "Recipe path escapes its worktree"
    );
    Ok(absolute)
}

pub fn save_local_recipe(
    project: Entity<Project>,
    path: ProjectPath,
    yaml: String,
    expected: Option<String>,
    cx: &mut App,
) -> Task<Result<()>> {
    if let Err(error) =
        Recipe::from_yaml(&yaml).and_then(|_| validate_local_path(&project, &path, cx))
    {
        return Task::ready(Err(error));
    }
    cx.spawn(async move |cx| {
        let exists = project.read_with(cx, |project, cx| {
            project
                .worktree_for_id(path.worktree_id, cx)
                .is_some_and(|worktree| worktree.read(cx).entry_for_path(&path.path).is_some())
        });
        let buffer = if exists {
            ensure!(expected.is_some(), "A recipe already exists at this path");
            project
                .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
                .await?
        } else {
            ensure!(
                expected.is_none(),
                "The recipe was removed; refresh before saving"
            );
            project.update(cx, |project, cx| {
                project.create_local_buffer("", None, false, cx)
            })
        };
        cx.update(|cx| {
            validate_local_path(&project, &path, cx)?;
            if let Some(expected) = expected {
                ensure!(
                    buffer.read(cx).as_text_snapshot().as_rope().to_string() == expected,
                    "The recipe changed while editing; refresh before saving"
                );
            }
            buffer.update(cx, |buffer, cx| {
                let length = buffer.len();
                buffer.edit([(0..length, yaml)], None, cx);
            });
            anyhow::Ok(())
        })?;
        project
            .update(cx, |project, cx| project.save_buffer_as(buffer, path, cx))
            .await
    })
}

pub fn delete_local_recipe(
    project: Entity<Project>,
    path: ProjectPath,
    cx: &mut App,
) -> Task<Result<()>> {
    if let Err(error) = validate_local_path(&project, &path, cx) {
        return Task::ready(Err(error));
    }
    let entry = project
        .read(cx)
        .worktree_for_id(path.worktree_id, cx)
        .and_then(|worktree| {
            worktree
                .read(cx)
                .entry_for_path(&path.path)
                .map(|entry| entry.id)
        });
    match entry.and_then(|entry| project.update(cx, |project, cx| project.delete_entry(entry, cx)))
    {
        Some(task) => task,
        None => Task::ready(Err(anyhow::anyhow!("Recipe is unavailable"))),
    }
}

pub fn start_recipe(
    connection: Rc<crate::NativeAgentConnection>,
    project: Entity<Project>,
    entry: RecipeEntry,
    parameters: HashMap<String, String>,
    cx: &mut App,
) -> Result<Entity<acp_thread::AcpThread>> {
    let recipe = entry.recipe.map_err(anyhow::Error::msg)?;
    recipe.render(&parameters)?;
    if let Some(path) = &entry.path {
        validate_local_path(&project, path, cx)?;
    }
    let session = connection
        .0
        .update(cx, |agent, cx| agent.new_session(project, cx));
    let session_id = session.read(cx).session_id().clone();
    let cleanup = (connection.clone(), session_id.clone());
    let run = session.update(cx, |session, cx| {
        session.run_turn(cx, async move |_, cx| {
            cx.update(|cx| {
                execute_recipe(
                    &connection,
                    session_id,
                    ClientUserMessageId::new(),
                    &recipe,
                    &parameters,
                    Vec::new(),
                    cx,
                )
            })
            .await
        })
    });
    cx.spawn(async move |cx| {
        let result = run.await;
        cx.update(|cx| cleanup.0.close_session(&cleanup.1, cx))
            .await?;
        result.map(|_| ())
    })
    .detach_and_log_err(cx);
    Ok(session)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeEntry {
    pub id: String,
    pub origin: String,
    pub path: Option<ProjectPath>,
    pub recipe: std::result::Result<Recipe, String>,
}

impl RecipeEntry {
    pub fn qualified_command(&self) -> String {
        let id = self
            .id
            .split('/')
            .map(|part| url::form_urlencoded::byte_serialize(part.as_bytes()).collect::<String>())
            .collect::<Vec<_>>()
            .join("/");
        format!("recipe/{id}")
    }
}

pub(crate) fn commands_for_project<'a>(
    state: &'a crate::ProjectState,
    cx: &App,
) -> Vec<(&'a RecipeEntry, acp::AvailableCommand)> {
    let mut used = crate::NATIVE_COMMANDS
        .iter()
        .map(|command| command.name.to_owned())
        .chain(state.skills.iter().map(|skill| skill.name.clone()))
        .chain(
            state
                .context_server_registry
                .read(cx)
                .prompts()
                .map(|prompt| prompt.prompt.name.clone()),
        )
        .collect::<HashSet<_>>();
    let mut commands = Vec::new();
    for entry in &state.recipes {
        let Ok(recipe) = &entry.recipe else { continue };
        let hint = recipe
            .parameters
            .iter()
            .map(|parameter| {
                if parameter.required && parameter.default.is_none() {
                    format!("<{}=value>", parameter.name)
                } else {
                    format!("[{}=value]", parameter.name)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let mut names = vec![entry.qualified_command()];
        if let Some(alias) = &recipe.command
            && used.insert(alias.clone())
        {
            names.push(alias.clone());
        }
        for name in names {
            commands.push((
                entry,
                acp::AvailableCommand::new(
                    name,
                    format!("{} — {}", recipe.description, entry.origin),
                )
                .meta(acp_thread::meta_with_command_category(
                    acp_thread::CommandCategory::Native,
                ))
                .input(acp::AvailableCommandInput::Unstructured(
                    acp::UnstructuredCommandInput::new(if hint.is_empty() {
                        "No parameters".to_owned()
                    } else {
                        hint.clone()
                    }),
                )),
            ));
        }
    }
    commands
}

pub(crate) fn run_command(
    connection: &crate::NativeAgentConnection,
    session_id: acp::SessionId,
    client_user_message_id: ClientUserMessageId,
    entry: RecipeEntry,
    arguments: &str,
    attachments: Vec<acp::ContentBlock>,
    cx: &mut App,
) -> Task<Result<acp::PromptResponse>> {
    let validate = || -> Result<_> {
        let recipe = entry
            .recipe
            .as_ref()
            .map_err(|error| anyhow::anyhow!(error.clone()))?;
        let parameters = recipe.parse_arguments(arguments)?;
        Ok((recipe.clone(), parameters))
    };
    let (recipe, parameters) = match validate() {
        Ok(validated) => validated,
        Err(error) => return Task::ready(Err(error)),
    };
    if let Some(path) = entry.path {
        let project = connection
            .0
            .read(cx)
            .session_project_state(&session_id)
            .map(|state| state.project.clone());
        let Some(project) = project else {
            return Task::ready(Err(anyhow::anyhow!("Recipe project is unavailable")));
        };
        let store = project.read(cx).worktree_store();
        let trusted = TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
            trusted.update(cx, |trusted, cx| {
                trusted.can_trust(&store, path.worktree_id, cx)
            })
        });
        let available = project.read(cx).visible_worktrees(cx).any(|worktree| {
            let worktree = worktree.read(cx);
            worktree.id() == path.worktree_id
                && worktree
                    .entry_for_path(&path.path)
                    .is_some_and(|entry| entry.is_file() && !entry.is_external)
        });
        if !trusted || !available {
            return Task::ready(Err(anyhow::anyhow!(
                "Recipe worktree is unavailable or not trusted; refresh recipes and review workspace trust"
            )));
        }
    }
    execute_recipe(
        connection,
        session_id,
        client_user_message_id,
        &recipe,
        &parameters,
        attachments,
        cx,
    )
}

pub fn builtin_recipes() -> Vec<RecipeEntry> {
    vec![RecipeEntry {
        id: "builtin/release-risk".to_owned(),
        origin: "Built-in".to_owned(),
        path: None,
        recipe: Recipe::from_yaml(include_str!("templates/recipes/release-risk.yaml"))
            .map_err(|error| error.to_string()),
    }]
}

pub(crate) fn execute_recipe(
    connection: &crate::NativeAgentConnection,
    session_id: acp::SessionId,
    client_user_message_id: ClientUserMessageId,
    recipe: &Recipe,
    parameters: &HashMap<String, String>,
    attachments: Vec<acp::ContentBlock>,
    cx: &mut App,
) -> Task<Result<acp::PromptResponse>> {
    if recipe.version == 2 {
        return execute_workflow(
            connection,
            session_id,
            client_user_message_id,
            recipe.clone(),
            parameters.clone(),
            attachments,
            cx,
        );
    }
    let rendered = match recipe.render(parameters) {
        Ok(rendered) => rendered,
        Err(error) => return Task::ready(Err(error)),
    };
    let Some(state) = connection.0.read(cx).session_project_state(&session_id) else {
        return Task::ready(Err(anyhow::anyhow!("Recipe session is unavailable")));
    };
    let path_style = state.project.read(cx).path_style(cx);
    connection.run_turn(session_id, cx, move |thread, cx| {
        if thread.read(cx).model().is_none() {
            return Err(anyhow::anyhow!(crate::NoModelConfiguredError));
        }
        let content: Vec<_> = std::iter::once(acp::ContentBlock::from(rendered))
            .chain(attachments)
            .map(|block| crate::UserMessageContent::from_content_block(block, path_style))
            .collect::<Vec<_>>();
        thread.update(cx, |thread, cx| {
            thread.send(client_user_message_id, content, cx)
        })
    })
}

fn execute_workflow(
    connection: &crate::NativeAgentConnection,
    session_id: acp::SessionId,
    first_message_id: ClientUserMessageId,
    recipe: Recipe,
    parameters: HashMap<String, String>,
    attachments: Vec<acp::ContentBlock>,
    cx: &mut App,
) -> Task<Result<acp::PromptResponse>> {
    use crate::recipe_workflow::{FailurePolicy, StepOutcome, StepStatus};
    use futures::FutureExt as _;
    let validated = recipe
        .validate()
        .and_then(|()| recipe.resolve_parameters(&parameters));
    let parameters = match validated {
        Ok(parameters) => parameters,
        Err(error) => return Task::ready(Err(error)),
    };
    let Some((thread, session, project)) =
        connection
            .0
            .read(cx)
            .sessions
            .get(&session_id)
            .map(|session| {
                (
                    session.thread.clone(),
                    session.acp_thread.clone(),
                    session.thread.read(cx).project().clone(),
                )
            })
    else {
        return Task::ready(Err(anyhow::anyhow!("Recipe session is unavailable")));
    };
    if thread.read(cx).recipe_permission_denied.is_some() {
        return Task::ready(Err(anyhow::anyhow!(
            "A recipe workflow is already running in this session"
        )));
    }
    let denied = Rc::new(std::cell::Cell::new(false));
    let (cancellation_sender, cancellation) = futures::channel::oneshot::channel();
    let previous_retries = thread.update(cx, |thread, _| {
        let previous = thread.automatic_retries_enabled();
        thread.set_automatic_retries(false);
        thread.recipe_permission_denied = Some(denied.clone());
        thread.recipe_cancellation = Some(cancellation_sender);
        previous
    });
    let restore_policy = util::defer({
        let thread = thread.clone();
        let mut cx = cx.to_async();
        move || {
            thread.update(&mut cx, |thread, cx| {
                thread.cancel(cx).detach();
                thread.set_automatic_retries(previous_retries);
                thread.recipe_permission_denied = None;
                thread.recipe_cancellation = None;
            })
        }
    });
    let overall_deadline = cx
        .background_executor()
        .timer(std::time::Duration::from_secs(15 * 60))
        .fuse();
    let connection = connection.clone();
    cx.spawn(async move |cx| {
        let _restore_policy = restore_policy;
        let _authorization = cx.update(|cx| {
            let denied = denied.clone();
            let thread = thread.clone();
            cx.subscribe(&session, move |session, event, cx| {
                if let acp_thread::AcpThreadEvent::EntryUpdated(index) = event
                    && let Some(acp_thread::AgentThreadEntry::ToolCall(call)) =
                        session.read(cx).entries().get(*index)
                    && matches!(call.status, acp_thread::ToolCallStatus::Rejected)
                {
                    denied.set(true);
                    thread.update(cx, |thread, cx| thread.cancel(cx)).detach();
                }
            })
        });
        futures::pin_mut!(overall_deadline);
        let (result, interruption) = {
            let operation = async {
                let mut outcomes = HashMap::<String, StepOutcome>::default();
                let mut index = 0;
                let mut output_bytes = 0usize;
                let mut first_message_id = Some(first_message_id);
                let mut attachments = Some(attachments);
                while let Some(step) = recipe.steps.get(index) {
                    ensure!(
                        !denied.get(),
                        "Recipe stopped because tool permission was denied"
                    );
                    cx.update(|cx| {
                        ensure!(
                            !project.read(cx).is_disconnected(cx),
                            "Recipe project disconnected"
                        );
                        ensure!(
                            !TrustedWorktrees::has_restricted_worktrees(
                                &project.read(cx).worktree_store(),
                                cx
                            ),
                            "Recipe project is no longer trusted"
                        );
                        Ok::<_, anyhow::Error>(())
                    })?;
                    let enabled = step
                        .when
                        .as_ref()
                        .map(|condition| condition.evaluate(&parameters, &outcomes))
                        .transpose()?
                        .unwrap_or(true);
                    if !enabled {
                        outcomes.insert(
                            step.id.clone(),
                            StepOutcome {
                                status: StepStatus::Skipped,
                                output: None,
                            },
                        );
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: skipped (condition was false)", step.id),
                            cx,
                        );
                        index += 1;
                        continue;
                    }
                    let prompt = crate::recipe_workflow::render_step(step, &parameters, &outcomes)?;
                    let mut single = recipe.clone();
                    single.version = 1;
                    single.steps.clear();
                    single.prompt = prompt;
                    single.instructions = recipe
                        .instructions
                        .as_ref()
                        .map(|instructions| {
                            render_template(instructions, |name| {
                                parameters
                                    .get(name)
                                    .cloned()
                                    .context("Missing recipe parameter")
                            })
                        })
                        .transpose()?;
                    single.parameters.clear();
                    // Already-rendered output is literal data, so it must never pass through the template renderer again.
                    let rendered = single
                        .instructions
                        .iter()
                        .map(String::as_str)
                        .chain(std::iter::once(single.prompt.as_str()))
                        .filter(|text| !text.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    ensure!(
                        rendered.len() <= MAX_RENDERED_RECIPE_BYTES,
                        "Rendered step exceeds 1 MiB"
                    );
                    let step_deadline = cx
                        .background_executor()
                        .timer(std::time::Duration::from_secs(5 * 60))
                        .fuse();
                    futures::pin_mut!(step_deadline);
                    workflow_progress(
                        &thread,
                        &session,
                        format!("Step {}: running", step.id),
                        cx,
                    );
                    {
                        let persistence = persist_workflow_session(&connection, &thread, &project, cx).fuse();
                        futures::pin_mut!(persistence);
                        futures::select_biased! {
                            _ = step_deadline => return Err(RecipeWorkflowTimeout(format!("Step {} timed out while saving its session", step.id)).into()),
                            result = persistence => result?,
                        }
                    }
                    let first_message =
                        thread.read_with(cx, |thread, _| thread.recipe_message_count());
                    let message_id = first_message_id
                        .take()
                        .unwrap_or_else(ClientUserMessageId::new);
                    let attachments = attachments.take().unwrap_or_default();
                    let path_style = project.read_with(cx, |project, cx| project.path_style(cx));
                    let operation = cx
                        .update(|cx| {
                            connection.run_turn(session_id.clone(), cx, move |thread, cx| {
                                ensure!(
                                    thread.read(cx).model().is_some(),
                                    crate::NoModelConfiguredError
                                );
                                let content: Vec<_> =
                                    std::iter::once(acp::ContentBlock::from(rendered))
                                        .chain(attachments)
                                        .map(|block| {
                                            crate::UserMessageContent::from_content_block(
                                                block, path_style,
                                            )
                                        })
                                        .collect();
                                thread.update(cx, |thread, cx| thread.send(message_id, content, cx))
                            })
                        })
                        .fuse();
                    futures::pin_mut!(operation);
                    let result = futures::select_biased! {
                        _ = step_deadline => None,
                        result = operation => Some(result),
                    };
                    let Some(result) = result else {
                        thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: timed out; workflow stopped", step.id),
                            cx,
                        );
                        return Err(RecipeWorkflowTimeout("Recipe workflow timed out".into()).into());
                    };
                    if let Some(reason) = thread.read_with(cx, |thread, _| {
                        thread
                            .recipe_scan_policy
                            .as_ref()
                            .and_then(|policy| policy.blocked_dependency())
                    }) {
                        thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: test incomplete; {reason}", step.id),
                            cx,
                        );
                        bail!("Recipe test is incomplete: {reason}");
                    }
                    thread.read_with(cx, |thread, _| {
                        if let Some(policy) = &thread.recipe_scan_policy {
                            policy.check()?;
                        }
                        anyhow::Ok(())
                    })?;
                    if denied.get() {
                        thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: permission denied; workflow stopped", step.id),
                            cx,
                        );
                        bail!("Recipe workflow needs attention: tool permission denied");
                    }
                    if let Ok(response) = &result
                        && response.stop_reason == acp::StopReason::Cancelled
                    {
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: cancelled; workflow stopped", step.id),
                            cx,
                        );
                        return result;
                    }
                    if result.as_ref().is_err_and(|error| needs_attention(error)) {
                        thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: needs attention; workflow stopped", step.id),
                            cx,
                        );
                        return result;
                    }
                    let (output, tool_failed) = thread
                        .read_with(cx, |thread, _| thread.recipe_step_output(first_message))?;
                    output_bytes = output_bytes.saturating_add(output.len());
                    ensure!(
                        output_bytes <= crate::recipe_workflow::MAX_WORKFLOW_OUTPUT,
                        "Workflow output exceeds 1 MiB"
                    );
                    let succeeded = !tool_failed
                        && result
                            .as_ref()
                            .is_ok_and(|response| response.stop_reason == acp::StopReason::EndTurn);
                    if !succeeded {
                        thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                    }
                    let status = if succeeded {
                        StepStatus::Succeeded
                    } else {
                        StepStatus::Failed
                    };
                    workflow_progress(
                        &thread,
                        &session,
                        format!("Step {}: {}", step.id, status.as_str()),
                        cx,
                    );
                    outcomes.insert(
                        step.id.clone(),
                        StepOutcome {
                            status,
                            output: succeeded.then_some(output),
                        },
                    );
                    {
                        let persistence = persist_workflow_session(&connection, &thread, &project, cx).fuse();
                        futures::pin_mut!(persistence);
                        futures::select_biased! {
                            _ = step_deadline => return Err(RecipeWorkflowTimeout(format!("Step {} timed out while saving its session", step.id)).into()),
                            result = persistence => result?,
                        }
                    }
                    let target = if succeeded {
                        step.next.as_ref()
                    } else {
                        match &step.on_failure {
                            FailurePolicy::Stop => {
                                result?;
                                bail!("Step {} did not complete successfully", step.id);
                            }
                            FailurePolicy::Continue => None,
                            FailurePolicy::GoTo(target) => Some(target),
                        }
                    };
                    let next = match target {
                        Some(target) => recipe
                            .steps
                            .iter()
                            .position(|step| &step.id == target)
                            .context("Workflow target is unavailable")?,
                        None => index + 1,
                    };
                    for unvisited in recipe.steps.iter().take(next).skip(index + 1) {
                        outcomes.insert(
                            unvisited.id.clone(),
                            StepOutcome {
                                status: StepStatus::Skipped,
                                output: None,
                            },
                        );
                        workflow_progress(
                            &thread,
                            &session,
                            format!("Step {}: unvisited (branch selected)", unvisited.id),
                            cx,
                        );
                    }
                    index = next;
                }
                workflow_progress(
                    &thread,
                    &session,
                    "Workflow finished; inspect individual step outcomes and unvisited branches."
                        .into(),
                    cx,
                );
                Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
            }
            .fuse();
            let cancellation = cancellation.fuse();
            futures::pin_mut!(operation, cancellation);
            futures::select_biased! {
                _ = cancellation => (
                    Ok(acp::PromptResponse::new(acp::StopReason::Cancelled)),
                    Some("Workflow cancelled; later steps were not executed"),
                ),
                _ = overall_deadline => (
                    Err(RecipeWorkflowTimeout("Recipe workflow timed out".into()).into()),
                    Some("Workflow timed out; later steps were not executed"),
                ),
                result = operation => (result, None),
            }
        };
        if let Some(message) = interruption {
            thread.update(cx, |thread, cx| thread.cancel(cx)).await;
            workflow_progress(&thread, &session, message.into(), cx);
        }
        result
    })
}

fn workflow_progress(
    thread: &Entity<crate::Thread>,
    session: &Entity<acp_thread::AcpThread>,
    message: String,
    cx: &mut gpui::AsyncApp,
) {
    thread.update(cx, |thread, cx| {
        if let Some(policy) = &thread.recipe_scan_policy {
            let now = cx
                .background_executor()
                .scheduler_executor()
                .scheduler()
                .clock()
                .utc_now();
            policy
                .observations
                .borrow_mut()
                .push(format!("{now} · {message}"));
        }
        thread.push_acp_agent_block(message.clone().into(), cx)
    });
    session.update(cx, |session, cx| {
        session.push_assistant_content_block(message.into(), false, cx)
    });
}

pub(crate) async fn persist_workflow_session(
    connection: &crate::NativeAgentConnection,
    thread: &Entity<crate::Thread>,
    project: &Entity<Project>,
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    let scan_policy = thread.read_with(cx, |thread, _| thread.recipe_scan_policy.clone());
    let paths = project.read_with(cx, |project, cx| {
        util::path_list::PathList::new(
            &project
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        )
    });
    let paths = scan_policy
        .as_ref()
        .and_then(|policy| policy.project_paths.as_ref())
        .map(util::path_list::PathList::deserialize)
        .unwrap_or(paths);
    let (id, payload) = thread.read_with(cx, |thread, cx| (thread.id().clone(), thread.to_db(cx)));
    let payload = payload.await;
    let store = connection
        .0
        .read_with(cx, |agent, _| agent.thread_store.clone());
    store
        .update(cx, |store, cx| {
            store.save_thread(id.clone(), payload, paths, cx)
        })
        .await?;
    if let Some(policy) = scan_policy
        && let Some(run_id) = policy.run_id
    {
        let owner = cx
            .update(|cx| RecipeStore::try_global(cx))
            .context("Recipe history owner is unavailable")?;
        let observations = policy.observations.borrow().clone();
        ensure!(
            observations.len() <= 256 && observations.iter().all(|entry| entry.len() <= 4096),
            "Test report exceeds its bounds"
        );
        owner
            .update(cx, |owner, cx| {
                owner.transact(
                    move |state, _| {
                        let record = state
                            .scans
                            .iter_mut()
                            .find(|record| record.identity.run_id == run_id)
                            .context("Saved test is unavailable")?;
                        record.session_id = Some(id.to_string());
                        record.observations = observations;
                        Ok(())
                    },
                    cx,
                )
            })
            .await?;
    }
    Ok(())
}

pub fn discover_recipes(
    project: Entity<Project>,
    fs: Arc<dyn Fs>,
    cx: &mut App,
) -> Task<Vec<RecipeEntry>> {
    let worktrees = project.read(cx).visible_worktrees(cx).collect::<Vec<_>>();
    cx.spawn(async move |cx| {
        let mut recipes = Vec::new();
        let canonical_roots = crate::tools::canonicalize_worktree_roots(&project, &fs, cx).await;
        for worktree in worktrees {
            let first_recipe = recipes.len();
            let (worktree_id, root_name, scan_complete) = worktree.read_with(cx, |worktree, _| {
                (
                    worktree.id(),
                    worktree.root_name_str().to_owned(),
                    worktree.as_local().map(|local| local.scan_complete()),
                )
            });
            let trusted = cx.update(|cx| {
                let store = project.read(cx).worktree_store();
                TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
                    trusted.update(cx, |trusted, cx| trusted.can_trust(&store, worktree_id, cx))
                })
            });
            if !trusted {
                continue;
            }
            if let Some(scan_complete) = scan_complete {
                scan_complete.await;
            }
            let prefix = match RelPath::from_unix_str(".agents/recipes") {
                Ok(prefix) => prefix,
                Err(error) => {
                    log::error!("Invalid recipe directory: {error}");
                    return recipes;
                }
            };
            for directory in [prefix.parent(), Some(prefix)].into_iter().flatten() {
                if let Err(error) = crate::expand_worktree_directory(&worktree, directory, cx).await
                {
                    recipes.push(RecipeEntry {
                        id: format!("{worktree_id}/discovery-error"),
                        origin: root_name.clone(),
                        path: None,
                        recipe: Err(format!("Could not scan recipe directory: {error}")),
                    });
                }
            }
            let files = worktree.read_with(cx, |worktree, _| {
                worktree
                    .child_entries(prefix)
                    .filter(|entry| {
                        entry.is_file()
                            && (entry.path.as_unix_str().ends_with(".yaml")
                                || entry.path.as_unix_str().ends_with(".yml"))
                    })
                    .map(|entry| {
                        (
                            entry.path.clone(),
                            entry.size,
                            entry.is_external,
                            worktree.absolutize(&entry.path),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for (path, size, external, absolute_path) in files {
                let project_path = ProjectPath {
                    worktree_id,
                    path: path.clone(),
                };
                let result = async {
                    ensure!(size <= MAX_RECIPE_BYTES as u64, "Recipe exceeds 256 KiB");
                    ensure!(!external, "Recipe path escapes its worktree");
                    cx.update(|cx| {
                        match crate::tools::resolve_project_path(
                            project.read(cx),
                            &absolute_path,
                            &canonical_roots,
                            cx,
                        )? {
                            crate::tools::ResolvedProjectPath::Safe(resolved)
                                if resolved == project_path =>
                            {
                                Ok(())
                            }
                            _ => anyhow::bail!("Recipe path escapes its worktree"),
                        }
                    })?;
                    let buffer = project
                        .update(cx, |project, cx| {
                            project.open_buffer(project_path.clone(), cx)
                        })
                        .await?;
                    let content = buffer.read_with(cx, |buffer, _| {
                        buffer.as_text_snapshot().as_rope().to_string()
                    });
                    Recipe::from_yaml(&content)
                }
                .await;
                recipes.push(RecipeEntry {
                    id: format!("{worktree_id}/{}", path.as_unix_str()),
                    origin: format!("{root_name}/{}", path.as_unix_str()),
                    path: Some(project_path),
                    recipe: result.map_err(|error| error.to_string()),
                });
            }
            let still_trusted = cx.update(|cx| {
                let store = project.read(cx).worktree_store();
                project
                    .read(cx)
                    .visible_worktrees(cx)
                    .any(|visible| visible == worktree)
                    && TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
                        trusted.update(cx, |trusted, cx| trusted.can_trust(&store, worktree_id, cx))
                    })
            });
            if !still_trusted {
                recipes.truncate(first_recipe);
            }
        }
        recipes.extend(builtin_recipes());
        recipes
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub version: u32,
    pub title: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_parameters"
    )]
    pub parameters: Vec<RecipeParameter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::recipe_workflow::deserialize_steps"
    )]
    pub steps: Vec<crate::recipe_workflow::RecipeStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<crate::recipe_import::RecipeProvenance>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeParameter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

fn deserialize_parameters<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<RecipeParameter>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Parameters;

    impl<'de> serde::de::Visitor<'de> for Parameters {
        type Value = Vec<RecipeParameter>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("at most 64 recipe parameters")
        }

        fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut parameters = Vec::new();
            while let Some(parameter) = sequence.next_element()? {
                if parameters.len() == 64 {
                    return Err(serde::de::Error::custom("Recipe exceeds 64 parameters"));
                }
                parameters.push(parameter);
            }
            Ok(parameters)
        }
    }

    // Bound expansion while deserializing: a small YAML document can repeat a large anchor.
    deserializer.deserialize_seq(Parameters)
}

pub(crate) fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == b'-'
        })
}

impl Recipe {
    pub fn from_yaml(content: &str) -> Result<Self> {
        ensure!(content.len() <= MAX_RECIPE_BYTES, "Recipe exceeds 256 KiB");
        let recipe: Self = serde_yaml::from_str(content).map_err(|error| {
            // Deserializer diagnostics can quote input values, which may contain secrets.
            match error.location() {
                Some(location) => anyhow::anyhow!(
                    "Invalid recipe YAML or field type at line {}, column {}",
                    location.line(),
                    location.column()
                ),
                None => anyhow::anyhow!("Invalid recipe YAML or field type"),
            }
        })?;
        recipe.validate()?;
        Ok(recipe)
    }

    pub fn to_yaml(&self) -> Result<String> {
        self.validate()?;
        let yaml = serde_yaml::to_string(self).context("Could not serialize recipe")?;
        ensure!(yaml.len() <= MAX_RECIPE_BYTES, "Recipe exceeds 256 KiB");
        Ok(yaml)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.version, 1 | 2),
            "Unsupported recipe version; expected 1 or 2"
        );
        ensure!(!self.title.trim().is_empty(), "Recipe title is required");
        ensure!(self.title.len() <= 256, "Recipe title exceeds 256 bytes");
        ensure!(
            !self.description.trim().is_empty(),
            "Recipe description is required"
        );
        ensure!(
            self.version == 2
                || !self.prompt.trim().is_empty()
                || self
                    .instructions
                    .as_deref()
                    .is_some_and(|instructions| !instructions.trim().is_empty()),
            "Recipe prompt or instructions are required"
        );
        if let Some(command) = &self.command {
            ensure!(
                valid_name(command),
                "Recipe command must contain 1–64 lowercase letters, digits or hyphens"
            );
        }
        ensure!(self.parameters.len() <= 64, "Recipe exceeds 64 parameters");
        let mut names = HashSet::default();
        for parameter in &self.parameters {
            ensure!(
                valid_name(&parameter.name),
                "Parameter names must contain 1–64 lowercase letters, digits or hyphens"
            );
            ensure!(
                names.insert(parameter.name.as_str()),
                "Duplicate recipe parameter: {}",
                parameter.name
            );
        }
        for template in std::iter::once(self.prompt.as_str()).chain(self.instructions.as_deref()) {
            render_template(template, |name| {
                ensure!(names.contains(name), "Undeclared recipe parameter: {name}");
                Ok(String::new())
            })?;
        }
        crate::recipe_workflow::validate(self)?;
        if let Some(provenance) = &self.imported_from {
            provenance.validate()?;
        }
        Ok(())
    }

    pub fn resolve_parameters(
        &self,
        supplied: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>> {
        self.validate()?;
        for name in supplied.keys() {
            ensure!(
                self.parameters
                    .iter()
                    .any(|parameter| &parameter.name == name),
                "Unknown recipe parameter: {name}"
            );
        }
        self.parameters
            .iter()
            .map(|parameter| {
                let value = supplied.get(&parameter.name).or(parameter.default.as_ref());
                ensure!(
                    value.is_some_and(|value| !value.trim().is_empty()) || !parameter.required,
                    "Missing required recipe parameter: {}",
                    parameter.name
                );
                Ok((parameter.name.clone(), value.cloned().unwrap_or_default()))
            })
            .collect()
    }

    pub fn render(&self, supplied: &HashMap<String, String>) -> Result<String> {
        let values = self.resolve_parameters(supplied)?;
        let mut rendered = String::new();
        for template in self
            .instructions
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(self.prompt.as_str()))
            .filter(|template| !template.trim().is_empty())
        {
            if !rendered.is_empty() {
                rendered.push_str("\n\n");
            }
            rendered.push_str(&render_template(template, |name| {
                values
                    .get(name)
                    .cloned()
                    .with_context(|| format!("Missing recipe parameter: {name}"))
            })?);
            ensure!(
                rendered.len() <= MAX_RENDERED_RECIPE_BYTES,
                "Rendered recipe exceeds 1 MiB"
            );
        }
        for step in &self.steps {
            let prompt = render_template(&step.prompt, |name| {
                if name.starts_with("steps.") {
                    Ok(format!(
                        "{{{{{name}}}}} [resolved from observed output at run time]"
                    ))
                } else {
                    values
                        .get(name)
                        .cloned()
                        .with_context(|| format!("Missing recipe parameter: {name}"))
                }
            })?;
            rendered.push_str(&format!("\n\nStep {}\n{}\n", step.id, prompt));
            if let Some(condition) = &step.when {
                rendered.push_str(&format!(
                    "Run only when:\n{}",
                    serde_yaml::to_string(condition)?
                ));
            }
            rendered.push_str(&format!(
                "On success: {}\nOn failure: {}",
                step.next.as_deref().unwrap_or("next step (or finish)"),
                match &step.on_failure {
                    crate::recipe_workflow::FailurePolicy::Stop => "stop workflow".to_owned(),
                    crate::recipe_workflow::FailurePolicy::Continue =>
                        "continue to next step".to_owned(),
                    crate::recipe_workflow::FailurePolicy::GoTo(target) =>
                        format!("go to {target}"),
                }
            ));
            ensure!(
                rendered.len() <= MAX_RENDERED_RECIPE_BYTES,
                "Rendered recipe exceeds 1 MiB"
            );
        }
        Ok(rendered)
    }

    pub fn parse_arguments(&self, input: &str) -> Result<HashMap<String, String>> {
        let mut supplied = HashMap::default();
        for word in crate::split_shell_words(input)
            .context("Invalid recipe arguments; check quoting and escapes")?
        {
            let (name, value) = word
                .split_once('=')
                .context("Use recipe arguments as name=value; quote values containing spaces")?;
            ensure!(
                supplied.insert(name.to_owned(), value.to_owned()).is_none(),
                "Duplicate recipe parameter: {name}"
            );
        }
        self.resolve_parameters(&supplied)
    }
}

pub(crate) fn render_template(
    template: &str,
    mut value: impl FnMut(&str) -> Result<String>,
) -> Result<String> {
    let mut remainder = template;
    let mut rendered = String::new();
    while let Some((prefix, placeholder)) = remainder.split_once("{{") {
        ensure!(
            !prefix.contains("}}"),
            "Unmatched recipe parameter delimiter"
        );
        let (name, suffix) = placeholder
            .split_once("}}")
            .context("Unclosed recipe parameter placeholder")?;
        let name = name.trim();
        ensure!(
            valid_name(name)
                || name
                    .strip_prefix("steps.")
                    .and_then(|name| name.strip_suffix(".output"))
                    .is_some_and(valid_name),
            "Invalid recipe placeholder; use {{name}} or {{steps.id.output}}"
        );
        let replacement = value(name)?;
        if rendered
            .len()
            .saturating_add(prefix.len())
            .saturating_add(replacement.len())
            > MAX_RENDERED_RECIPE_BYTES
        {
            bail!("Rendered recipe exceeds 1 MiB");
        }
        rendered.push_str(prefix);
        rendered.push_str(&replacement);
        remainder = suffix;
    }
    ensure!(
        !remainder.contains("}}"),
        "Unmatched recipe parameter delimiter"
    );
    ensure!(
        rendered.len().saturating_add(remainder.len()) <= MAX_RENDERED_RECIPE_BYTES,
        "Rendered recipe exceeds 1 MiB"
    );
    rendered.push_str(remainder);
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECIPE: &str = "version: 1\ntitle: Review\ndescription: Review a change\nprompt: 'Review {{ change }}'\nparameters:\n  - name: change\n    description: Change to review\n    required: true\n";

    #[test]
    fn recipe_builtin_is_valid_and_renders_without_network() -> Result<()> {
        let entries = builtin_recipes();
        let entry = entries.first().context("Missing built-in release review")?;
        let recipe = entry
            .recipe
            .as_ref()
            .map_err(|error| anyhow::anyhow!(error.clone()))?;
        assert_eq!(recipe.command.as_deref(), Some("release-risk"));
        assert!(
            recipe
                .render(&HashMap::default())?
                .contains("the current changes")
        );
        Ok(())
    }

    #[test]
    fn recipe_yaml_round_trip_preserves_values() -> Result<()> {
        let recipe = Recipe::from_yaml(RECIPE)?;
        assert_eq!(Recipe::from_yaml(&recipe.to_yaml()?)?, recipe);
        Ok(())
    }

    #[test]
    fn recipe_rejects_invalid_schema_without_disclosing_values() {
        for content in [
            RECIPE.replace("version: 1", "version: 2"),
            RECIPE.replace("title: Review", "title: ''"),
            RECIPE.replace("version: 1", "version: PRIVATE_VALUE"),
            format!("{RECIPE}tools: PRIVATE_VALUE\n"),
            format!("{RECIPE}title: PRIVATE_VALUE\n"),
            RECIPE.replace("{{ change }}", "{{ undeclared }}"),
            RECIPE.replace("{{ change }}", "{{ change | shell }}"),
        ] {
            let error = Recipe::from_yaml(&content).expect_err("invalid recipe accepted");
            assert!(!error.to_string().contains("PRIVATE_VALUE"));
        }
    }

    #[test]
    fn recipe_parameter_values_are_literal_and_bounded() -> Result<()> {
        let recipe = Recipe::from_yaml(RECIPE)?;
        let values = recipe.parse_arguments("change='{{not-expanded}}\nversion: 999'")?;
        assert_eq!(
            recipe.render(&values)?,
            "Review {{not-expanded}}\nversion: 999"
        );
        let oversized =
            HashMap::from_iter([("change".to_string(), "x".repeat(MAX_RENDERED_RECIPE_BYTES))]);
        assert!(recipe.render(&oversized).is_err());
        Ok(())
    }

    #[test]
    fn recipe_parameter_errors_precede_execution() -> Result<()> {
        let recipe = Recipe::from_yaml(RECIPE)?;
        for input in [
            "",
            "other=value",
            "change=a change=b",
            "change='unfinished",
            "change",
            "change=''",
        ] {
            assert!(recipe.parse_arguments(input).is_err(), "accepted {input}");
        }
        let duplicate = format!("{RECIPE}  - name: change\n    description: Duplicate\n");
        assert!(Recipe::from_yaml(&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn recipe_parameter_defaults_and_optional_values() -> Result<()> {
        let recipe = Recipe::from_yaml(&RECIPE.replace(
            "required: true",
            "required: false\n    default: current change",
        ))?;
        assert_eq!(recipe.render(&HashMap::default())?, "Review current change");
        let recipe = Recipe::from_yaml(&RECIPE.replace("required: true", "required: false"))?;
        assert_eq!(recipe.render(&HashMap::default())?, "Review ");
        assert!(Recipe::from_yaml(&" ".repeat(MAX_RECIPE_BYTES + 1)).is_err());
        Ok(())
    }

    #[test]
    fn recipe_accepts_instructions_only_and_bounds_alias_expansion() -> Result<()> {
        let recipe = Recipe::from_yaml(
            "version: 1\ntitle: Review\ndescription: Review changes\ninstructions: Review the current changes\n",
        )?;
        assert_eq!(
            recipe.render(&HashMap::default())?,
            "Review the current changes"
        );
        let repeated = format!(
            "version: 1\ntitle: Review\ndescription: Review\nprompt: Review\nparameters:\n  - &parameter {{name: change, description: Change}}\n{}",
            "  - *parameter\n".repeat(65)
        );
        assert!(Recipe::from_yaml(&repeated).is_err());
        Ok(())
    }
}
