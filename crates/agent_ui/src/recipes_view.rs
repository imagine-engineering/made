use std::{cell::RefCell, rc::Rc};

use agent::recipe_schedule::{RecipeSchedule, local_timezone, next_run};
use agent::recipes::{Recipe, RecipeEntry, RecipeProposal, RecipeReviewStatus, RecipeStore};
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Context as _, Result};
use collections::HashMap;
use editor::Editor;
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Render, Subscription,
    Task, WeakEntity, Window,
};
use language_model::{ConfiguredModel, LanguageModelRegistry, SelectedModel};
use project::{Project, ProjectPath};
use ui::{Checkbox, ContextMenu, PopoverMenu, prelude::*};
use util::{ResultExt as _, rel_path::RelPath};
use uuid::Uuid;
use workspace::ModalView;

use crate::{
    AgentPanel,
    language_model_selector::{LanguageModelSelector, language_model_selector},
};

struct RecipeImportPreview {
    yaml: String,
    previous_yaml: String,
    updating: bool,
    selected_id: Option<String>,
}

pub struct RecipesView {
    proposal: Option<Rc<RecipeProposal>>,
    panel: WeakEntity<AgentPanel>,
    project: Entity<Project>,
    connection: Rc<agent::NativeAgentConnection>,
    store: Entity<RecipeStore>,
    focus: FocusHandle,
    entries: Vec<RecipeEntry>,
    selected: Option<RecipeEntry>,
    selected_schedule: Option<Uuid>,
    source: Option<String>,
    search: Entity<Editor>,
    filename: Entity<Editor>,
    yaml: Entity<Editor>,
    import_url: Entity<Editor>,
    import_revision: Entity<Editor>,
    import_path: Entity<Editor>,
    import_use_token: bool,
    import_updating: bool,
    import_task: Option<Task<()>>,
    import_preview: Option<RecipeImportPreview>,
    link_notice: Option<String>,
    scan_image: Entity<Editor>,
    scan_endpoint: Entity<Editor>,
    scan_server: Entity<Editor>,
    scan_files: Entity<Editor>,
    scan_preparation: Option<Task<()>>,
    scan_preview: Option<(String, agent::recipe_scan::PreparedRecipeScan)>,
    parameters: Vec<(String, Entity<Editor>)>,
    title: Entity<Editor>,
    cron: Entity<Editor>,
    timezone: Entity<Editor>,
    worktree: Option<project::WorktreeId>,
    model: Rc<RefCell<Option<ConfiguredModel>>>,
    model_selector: Entity<LanguageModelSelector>,
    error: Option<String>,
    loading: bool,
    operation: Option<Task<()>>,
    _observation: Subscription,
}

fn input(value: &str, placeholder: &str, window: &mut Window, cx: &mut App) -> Entity<Editor> {
    cx.new(|cx| {
        let mut editor = Editor::single_line(window, cx);
        editor.set_placeholder_text(placeholder, window, cx);
        editor.set_text(value.to_owned(), window, cx);
        editor
    })
}

impl RecipesView {
    pub fn new(
        panel: WeakEntity<AgentPanel>,
        project: Entity<Project>,
        connection: Rc<agent::NativeAgentConnection>,
        store: Entity<RecipeStore>,
        schedule: Option<Uuid>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus = cx.focus_handle();
        let model = Rc::new(RefCell::new(
            LanguageModelRegistry::read_global(cx).default_model(),
        ));
        let active_model = model.clone();
        let fs = project.read(cx).fs().clone();
        let selected = cx.entity().downgrade();
        let model_selector = cx.new(|cx| {
            language_model_selector(
                move |_| active_model.borrow().clone(),
                move |model, cx| {
                    let configured =
                        LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                            registry.select_model(
                                &SelectedModel {
                                    provider: model.provider_id(),
                                    model: model.id(),
                                },
                                cx,
                            )
                        });
                    selected
                        .update(cx, |view, cx| {
                            *view.model.borrow_mut() = configured;
                            cx.notify();
                        })
                        .log_err();
                },
                move |model, favorite, cx| {
                    crate::favorite_models::toggle_in_settings(model, favorite, fs.clone(), cx)
                },
                true,
                focus.clone(),
                window,
                cx,
            )
        });
        let worktree = project
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .map(|worktree| worktree.read(cx).id());
        let observation = cx.observe(&store, |_, _, cx| cx.notify());
        let mut view = Self {
            proposal: None,
            panel,
            project,
            connection,
            store,
            focus,
            entries: Vec::new(),
            selected: None,
            selected_schedule: None,
            source: None,
            search: input("", "Search recipes", window, cx),
            filename: input("review.yaml", "Recipe filename.yaml", window, cx),
            yaml: cx.new(|cx| Editor::auto_height(4, 12, window, cx)),
            import_url: input("", "HTTPS recipe or GitHub repository/blob URL", window, cx),
            import_revision: input("", "Revision (repository URLs only)", window, cx),
            import_path: input("", "Recipe path (repository URLs only)", window, cx),
            import_use_token: false,
            import_updating: false,
            import_task: None,
            import_preview: None,
            link_notice: None,
            scan_image: input("", "Already-local image@sha256:digest", window, cx),
            scan_endpoint: input(
                "",
                "Local Docker endpoint (unix:///…/docker.sock)",
                window,
                cx,
            ),
            scan_server: input(
                "",
                "Absolute path to matching Linux Made server",
                window,
                cx,
            ),
            scan_files: cx.new(|cx| Editor::auto_height(3, 8, window, cx)),
            scan_preparation: None,
            scan_preview: None,
            parameters: Vec::new(),
            title: input("", "Schedule title", window, cx),
            cron: input("0 9 * * 1-5", "Five-field cron", window, cx),
            timezone: input(
                &local_timezone().unwrap_or_default(),
                "IANA time zone",
                window,
                cx,
            ),
            worktree,
            model,
            model_selector,
            error: None,
            loading: false,
            operation: None,
            _observation: observation,
        };
        cx.observe(&view.yaml, |_, _, cx| cx.notify()).detach();
        for editor in [
            &view.scan_image,
            &view.scan_endpoint,
            &view.scan_server,
            &view.scan_files,
        ] {
            cx.observe(editor, |_, _, cx| cx.notify()).detach();
        }
        cx.observe(&view.search, |_, _, cx| cx.notify()).detach();
        cx.observe(&view.cron, |_, _, cx| cx.notify()).detach();
        cx.observe(&view.timezone, |_, _, cx| cx.notify()).detach();
        cx.observe(&agent::ThreadStore::global(cx), |_, _, cx| cx.notify())
            .detach();
        cx.on_release(|view, _| {
            view.dismiss_proposal();
        })
        .detach();
        view.new_recipe(window, cx);
        if let Some(id) = schedule {
            view.select_schedule(id, window, cx);
        }
        view.refresh(cx);
        if view.worktree.is_some_and(|worktree_id| {
            view.project
                .read(cx)
                .git_store()
                .read(cx)
                .repository_and_path_for_project_path(
                    &ProjectPath {
                        worktree_id,
                        path: RelPath::empty().into(),
                    },
                    cx,
                )
                .is_some()
        }) {
            view.select_tracked_scan_files(window, cx);
        }
        view
    }

    fn dismiss_proposal(&mut self) {
        if let Some(proposal) = self.proposal.take() {
            if proposal.status.get() == RecipeReviewStatus::Opened {
                proposal.status.set(RecipeReviewStatus::Dismissed);
            }
        }
    }

    pub fn review_proposal(
        &mut self,
        proposal: Rc<RecipeProposal>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.new_recipe(window, cx);
        self.worktree = Some(proposal.worktree);
        *self.model.borrow_mut() = Some(proposal.model.clone());
        self.filename.update(cx, |editor, cx| {
            editor.set_text(
                proposal.input.filename.clone().unwrap_or_default(),
                window,
                cx,
            )
        });
        match proposal.recipe.to_yaml() {
            Ok(yaml) => self
                .yaml
                .update(cx, |editor, cx| editor.set_text(yaml, window, cx)),
            Err(error) => {
                self.error = Some(error.to_string());
                proposal.status.set(RecipeReviewStatus::Unavailable);
                return;
            }
        }
        self.refresh_parameters(window, cx);
        for (name, editor) in &self.parameters {
            if let Some(value) = proposal.parameters.get(name) {
                editor.update(cx, |editor, cx| editor.set_text(value.clone(), window, cx));
            }
        }
        if let Some(schedule) = &proposal.input.schedule {
            self.title.update(cx, |editor, cx| {
                editor.set_text(schedule.title.clone(), window, cx)
            });
            self.cron.update(cx, |editor, cx| {
                editor.set_text(schedule.cron.clone(), window, cx)
            });
            self.timezone.update(cx, |editor, cx| {
                editor.set_text(schedule.timezone.clone(), window, cx)
            });
        }
        proposal.status.set(RecipeReviewStatus::Opened);
        self.proposal = Some(proposal);
        cx.notify();
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        let fs = self.project.read(cx).fs().clone();
        let task = agent::recipes::discover_recipes(self.project.clone(), fs, cx);
        self.loading = true;
        cx.spawn(async move |this, cx| {
            let entries = task.await;
            this.update(cx, |this, cx| {
                this.entries = entries;
                this.loading = false;
                cx.notify();
            })
            .log_err();
        })
        .detach();
    }

    fn new_recipe(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.link_notice = None;
        self.import_task = None;
        self.import_preview = None;
        self.import_updating = false;
        self.import_use_token = false;
        self.dismiss_proposal();
        self.selected = None;
        self.selected_schedule = None;
        self.source = None;
        self.error = None;
        self.yaml.update(cx, |editor, cx| editor.set_text("version: 1\ntitle: Review changes\ndescription: Review the current changes\nprompt: Review {{focus}} and report actionable findings.\nparameters:\n  - name: focus\n    description: What to review\n    default: the current changes\ncommand: review-changes\n", window, cx));
        self.refresh_parameters(window, cx);
        cx.notify();
    }

    fn new_workflow(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.new_recipe(window, cx);
        self.yaml.update(cx, |editor, cx| editor.set_text(
            "version: 2\ntitle: Review and report\ndescription: Inspect changes and summarize observed findings\nparameters:\n  - name: focus\n    description: What to review\n    default: the current changes\nsteps:\n  - id: review\n    prompt: Review {{focus}} and report actionable findings.\n    on_failure: stop\n  - id: report\n    prompt: 'Summarize these findings: {{steps.review.output}}'\n    when:\n      operator: equals\n      left: {kind: status, step: review}\n      right: {kind: literal, value: succeeded}\n    on_failure: stop\n", window, cx));
        self.refresh_parameters(window, cx);
    }

    fn append_workflow_step(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| -> anyhow::Result<String> {
            let mut recipe = Recipe::from_yaml(&self.yaml.read(cx).text(cx))?;
            anyhow::ensure!(recipe.version == 2, "Choose New workflow to author steps");
            anyhow::ensure!(
                recipe.steps.len() < agent::recipe_workflow::MAX_STEPS,
                "A workflow supports at most 32 steps"
            );
            let name = (1..=agent::recipe_workflow::MAX_STEPS + 1)
                .map(|index| format!("step-{index}"))
                .find(|name| recipe.steps.iter().all(|step| &step.id != name))
                .context("Could not choose a unique step name")?;
            recipe.steps.push(agent::recipe_workflow::RecipeStep {
                id: name,
                prompt: "Describe the work for this step.".into(),
                when: None,
                next: None,
                on_failure: agent::recipe_workflow::FailurePolicy::Stop,
            });
            recipe.to_yaml()
        })();
        match result {
            Ok(yaml) => {
                self.yaml
                    .update(cx, |editor, cx| editor.set_text(yaml, window, cx));
                self.refresh_parameters(window, cx);
            }
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
            }
        }
    }

    pub fn review_link(
        &mut self,
        link: agent::recipe_import::RecipeLink,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Err(error) = link.encode() {
            self.error = Some(error.to_string());
            cx.notify();
            return;
        }
        self.new_recipe(window, cx);
        self.filename.update(cx, |editor, cx| {
            editor.set_text("linked-recipe.yaml", window, cx)
        });
        match link {
            agent::recipe_import::RecipeLink::Embedded { yaml } => {
                self.yaml
                    .update(cx, |editor, cx| editor.set_text(yaml, window, cx));
                self.refresh_parameters(window, cx);
                self.link_notice = Some("Untrusted recipe link. Review the complete instructions, parameters and steps below. Opening this link did not save or run the recipe or grant any permissions.".into());
            }
            agent::recipe_import::RecipeLink::Source(request) => {
                self.yaml
                    .update(cx, |editor, cx| editor.set_text("", window, cx));
                self.parameters.clear();
                self.import_url
                    .update(cx, |editor, cx| editor.set_text(request.url, window, cx));
                self.import_revision.update(cx, |editor, cx| {
                    editor.set_text(request.github_revision.unwrap_or_default(), window, cx)
                });
                self.import_path.update(cx, |editor, cx| {
                    editor.set_text(request.github_path.unwrap_or_default(), window, cx)
                });
                self.link_notice = Some("Untrusted recipe source link. Review the destination below before choosing Fetch for review. Opening this link did not contact the source, run a recipe or enable a schedule.".into());
            }
        }
        cx.notify();
    }

    fn copy_recipe_link(&mut self, source: bool, cx: &mut Context<Self>) {
        let result = (|| -> Result<String> {
            let yaml = self.yaml.read(cx).text(cx);
            let recipe = Recipe::from_yaml(&yaml)?;
            let link = if source {
                agent::recipe_import::RecipeLink::Source(
                    recipe
                        .imported_from
                        .context("This recipe has no import source")?
                        .request,
                )
            } else {
                agent::recipe_import::RecipeLink::Embedded { yaml }
            };
            link.encode()
        })();
        match result {
            Ok(link) => {
                cx.write_to_clipboard(gpui::ClipboardItem::new_string(link));
                self.error = None;
                self.link_notice = Some("Recipe link copied. Embedded links contain instructions and parameter defaults; review them before sharing.".into());
            }
            Err(error) => self.error = Some(error.to_string()),
        }
        cx.notify();
    }

    fn prepare_import_update(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| -> Result<agent::recipe_import::RecipeImportRequest> {
            anyhow::ensure!(
                self.selected
                    .as_ref()
                    .is_some_and(|entry| entry.path.is_some()),
                "Select a saved imported recipe to update"
            );
            Recipe::from_yaml(&self.yaml.read(cx).text(cx))?
                .imported_from
                .map(|source| source.request)
                .context("This recipe has no import source")
        })();
        match result {
            Ok(request) => {
                self.import_task = None;
                self.import_preview = None;
                self.import_url
                    .update(cx, |editor, cx| editor.set_text(request.url, window, cx));
                self.import_revision.update(cx, |editor, cx| {
                    editor.set_text(request.github_revision.unwrap_or_default(), window, cx)
                });
                self.import_path.update(cx, |editor, cx| {
                    editor.set_text(request.github_path.unwrap_or_default(), window, cx)
                });
                self.import_updating = true;
                self.import_use_token = false;
                self.error = None;
            }
            Err(error) => self.error = Some(error.to_string()),
        }
        cx.notify();
    }

    fn fetch_import(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.import_task.is_some() || self.operation.is_some() {
            return;
        }
        let result = (|| -> Result<_> {
            let revision = self.import_revision.read(cx).text(cx);
            let path = self.import_path.read(cx).text(cx);
            let request = agent::recipe_import::RecipeImportRequest {
                url: self.import_url.read(cx).text(cx),
                github_revision: (!revision.trim().is_empty()).then_some(revision),
                github_path: (!path.trim().is_empty()).then_some(path),
            };
            agent::recipe_import::validate_request(&request)?;
            let token = if self.import_use_token {
                anyhow::ensure!(
                    url::Url::parse(&request.url)?.host_str() == Some("github.com"),
                    "GitHub credentials cannot be used for another source"
                );
                Some(std::env::var("GITHUB_TOKEN").ok().filter(|token| !token.trim().is_empty()).context("GITHUB_TOKEN is unavailable; configure repository access or import without it")?)
            } else {
                None
            };
            Ok((request, token))
        })();
        let (request, token) = match result {
            Ok(input) => input,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        let http = self.project.read(cx).client().http_client();
        let previous_yaml = self.yaml.read(cx).text(cx);
        let updating = self.import_updating;
        let selected_id = self.selected.as_ref().map(|entry| entry.id.clone());
        self.import_preview = None;
        self.error = None;
        let fetch = cx.background_spawn(async move {
            agent::recipe_import::fetch_recipe(http, request, token).await
        });
        self.import_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = fetch.await.and_then(|recipe| recipe.to_yaml());
            this.update_in(cx, |this, _, cx| {
                this.complete_import_fetch(result, previous_yaml, updating, selected_id, cx);
            })
            .log_err();
        }));
        cx.notify();
    }

    fn complete_import_fetch(
        &mut self,
        result: Result<String>,
        previous_yaml: String,
        updating: bool,
        selected_id: Option<String>,
        cx: &mut Context<Self>,
    ) {
        self.import_task = None;
        match result {
            Ok(yaml) => {
                self.import_preview = Some(RecipeImportPreview {
                    yaml,
                    previous_yaml,
                    updating,
                    selected_id,
                })
            }
            Err(error) => self.error = Some(error.to_string()),
        }
        cx.notify();
    }

    fn apply_import_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.operation.is_some() {
            return;
        }
        let Some(preview) = self.import_preview.as_ref() else {
            return;
        };
        if preview.previous_yaml != self.yaml.read(cx).text(cx)
            || preview.selected_id != self.selected.as_ref().map(|entry| entry.id.clone())
        {
            self.error = Some("The current draft changed while the import was fetched. Fetch again to review against the current draft.".into());
            cx.notify();
            return;
        }
        let Some(preview) = self.import_preview.take() else {
            return;
        };
        self.dismiss_proposal();
        if !preview.updating {
            self.selected = None;
            self.source = None;
            self.filename.update(cx, |editor, cx| {
                editor.set_text("imported-recipe.yaml", window, cx)
            });
        }
        self.selected_schedule = None;
        self.yaml
            .update(cx, |editor, cx| editor.set_text(preview.yaml, window, cx));
        self.refresh_parameters(window, cx);
    }

    fn cancel_import(&mut self, cx: &mut Context<Self>) {
        self.import_task = None;
        self.import_preview = None;
        self.error = Some("Import cancelled. The current draft was preserved.".into());
        cx.notify();
    }

    fn refresh_parameters(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match Recipe::from_yaml(&self.yaml.read(cx).text(cx)) {
            Ok(recipe) => {
                let previous = self.parameter_values(cx);
                self.parameters = recipe
                    .parameters
                    .iter()
                    .map(|parameter| {
                        let value = previous
                            .get(&parameter.name)
                            .cloned()
                            .or_else(|| parameter.default.clone())
                            .unwrap_or_default();
                        (
                            parameter.name.clone(),
                            input(&value, &parameter.description, window, cx),
                        )
                    })
                    .collect();
                for (_, editor) in &self.parameters {
                    cx.observe(editor, |_, _, cx| cx.notify()).detach();
                }
                self.error = None;
            }
            Err(error) => {
                self.error = Some(error.to_string());
            }
        }
        cx.notify();
    }

    fn parameter_values(&self, cx: &App) -> HashMap<String, String> {
        self.parameters
            .iter()
            .map(|(name, editor)| (name.clone(), editor.read(cx).text(cx)))
            .collect()
    }

    fn select_recipe(&mut self, entry: RecipeEntry, window: &mut Window, cx: &mut Context<Self>) {
        self.dismiss_proposal();
        self.selected_schedule = None;
        self.selected = Some(entry.clone());
        self.source = None;
        self.error = None;
        if let Some(path) = &entry.path {
            self.worktree = Some(path.worktree_id);
            self.filename.update(cx, |editor, cx| {
                editor.set_text(
                    path.path.file_name().unwrap_or_default().to_owned(),
                    window,
                    cx,
                )
            });
            let path = path.clone();
            let task = self
                .project
                .update(cx, |project, cx| project.open_buffer(path, cx));
            self.loading = true;
            cx.spawn_in(window, async move |this, cx| {
                let result = task.await;
                this.update_in(cx, |this, window, cx| {
                    this.loading = false;
                    if this
                        .selected
                        .as_ref()
                        .is_none_or(|selected| selected.id != entry.id)
                    {
                        return;
                    }
                    match result {
                        Ok(buffer) => {
                            let text = buffer.read(cx).as_text_snapshot().as_rope().to_string();
                            this.source = Some(text.clone());
                            this.yaml
                                .update(cx, |editor, cx| editor.set_text(text, window, cx));
                            this.refresh_parameters(window, cx);
                        }
                        Err(error) => this.error = Some(error.to_string()),
                    }
                    cx.notify();
                })
                .log_err();
            })
            .detach();
        } else if let Ok(recipe) = entry.recipe {
            match recipe.to_yaml() {
                Ok(yaml) => {
                    self.yaml
                        .update(cx, |editor, cx| editor.set_text(yaml, window, cx));
                    self.refresh_parameters(window, cx);
                }
                Err(error) => self.error = Some(error.to_string()),
            }
        }
        cx.notify();
    }

    fn select_schedule(&mut self, id: Uuid, window: &mut Window, cx: &mut Context<Self>) {
        self.dismiss_proposal();
        let Some(schedule) = self
            .store
            .read(cx)
            .state
            .schedules
            .iter()
            .find(|schedule| schedule.id == id)
            .cloned()
        else {
            self.error = Some("Schedule is unavailable".into());
            return;
        };
        self.selected_schedule = Some(id);
        self.selected = None;
        self.source = None;
        self.title
            .update(cx, |editor, cx| editor.set_text(schedule.title, window, cx));
        self.cron
            .update(cx, |editor, cx| editor.set_text(schedule.cron, window, cx));
        self.timezone.update(cx, |editor, cx| {
            editor.set_text(schedule.timezone, window, cx)
        });
        self.worktree = self
            .project
            .read(cx)
            .visible_worktrees(cx)
            .find(|worktree| worktree.read(cx).abs_path().as_ref() == schedule.worktree.as_path())
            .map(|worktree| worktree.read(cx).id());
        *self.model.borrow_mut() = LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
            registry.select_model(
                &SelectedModel {
                    provider: schedule.provider.into(),
                    model: language_model::LanguageModelId(schedule.model.into()),
                },
                cx,
            )
        });
        match schedule.recipe.to_yaml() {
            Ok(yaml) => self
                .yaml
                .update(cx, |editor, cx| editor.set_text(yaml, window, cx)),
            Err(error) => self.error = Some(error.to_string()),
        }
        self.refresh_parameters(window, cx);
        for (name, editor) in &self.parameters {
            if let Some(value) = schedule.parameters.get(name) {
                editor.update(cx, |editor, cx| editor.set_text(value.clone(), window, cx));
            }
        }
        cx.notify();
    }

    fn report(&mut self, task: Task<Result<()>>, cx: &mut Context<Self>) {
        self.error = None;
        self.operation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                if let Err(error) = result {
                    this.error = Some(error.to_string());
                }
                this.operation = None;
                this.refresh(cx);
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn save_recipe(&mut self, cx: &mut Context<Self>) {
        if self
            .proposal
            .as_ref()
            .is_some_and(|proposal| proposal.status.get() == RecipeReviewStatus::Cancelled)
        {
            self.error = Some(
                "Draft request was cancelled. Close this review or start a new recipe.".into(),
            );
            cx.notify();
            return;
        }
        let result = (|| {
            let worktree_id = self.worktree.context("Choose a worktree")?;
            let filename = self.filename.read(cx).text(cx);
            anyhow::ensure!(
                !filename.contains('/') && !filename.contains('\\'),
                "Use a filename without directories"
            );
            let path = ProjectPath {
                worktree_id,
                path: RelPath::from_unix_str(&format!(".agents/recipes/{filename}"))?.into(),
            };
            let expected = self
                .selected
                .as_ref()
                .and_then(|entry| entry.path.as_ref())
                .filter(|original| **original == path)
                .and(self.source.clone());
            let yaml = self.yaml.read(cx).text(cx);
            let task = agent::recipes::save_local_recipe(
                self.project.clone(),
                path.clone(),
                yaml.clone(),
                expected,
                cx,
            );
            Ok((path, yaml, task))
        })();
        match result {
            Ok((path, yaml, task)) => {
                let task = cx.spawn(async move |this, cx| {
                    task.await?;
                    this.update(cx, |this, cx| {
                        this.source = Some(yaml.clone());
                        this.selected = Some(RecipeEntry {
                            id: format!("{}/{}", path.worktree_id, path.path.as_unix_str()),
                            origin: path.path.as_unix_str().to_owned(),
                            path: Some(path),
                            recipe: Recipe::from_yaml(&yaml).map_err(|error| error.to_string()),
                        });
                        cx.notify();
                    })?;
                    anyhow::Ok(())
                });
                self.report(task, cx);
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
        cx.notify();
    }

    fn save_schedule(&mut self, cx: &mut Context<Self>) {
        if self
            .proposal
            .as_ref()
            .is_some_and(|proposal| proposal.status.get() == RecipeReviewStatus::Cancelled)
        {
            self.error = Some(
                "Draft request was cancelled. Close this review or start a new recipe.".into(),
            );
            cx.notify();
            return;
        }
        let result = (|| -> Result<_> {
            let worktree_id = self.worktree.context("Choose a worktree")?;
            let worktree = self
                .project
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .context("Worktree is unavailable")?;
            let model = self
                .model
                .borrow()
                .clone()
                .context("Choose a provider and model")?;
            let now = cx
                .background_executor()
                .scheduler_executor()
                .scheduler()
                .clock()
                .utc_now();
            let cron = self.cron.read(cx).text(cx);
            let timezone = self.timezone.read(cx).text(cx);
            let id = self
                .selected_schedule
                .or_else(|| self.proposal.as_ref().map(|proposal| proposal.schedule_id))
                .unwrap_or_else(Uuid::new_v4);
            let schedule = RecipeSchedule {
                id,
                title: self.title.read(cx).text(cx),
                project_paths: self
                    .project
                    .read(cx)
                    .worktree_paths(cx)
                    .folder_path_list()
                    .serialize(),
                worktree: worktree.read(cx).abs_path().to_path_buf(),
                next_run: next_run(&cron, &timezone, now)?,
                cron,
                timezone,
                recipe: Recipe::from_yaml(&self.yaml.read(cx).text(cx))?,
                parameters: self.parameter_values(cx),
                provider: model.provider.id().to_string(),
                model: model.model.id().0.to_string(),
                paused: true,
                consented: false,
                deleted: false,
                last_run: None,
            };
            Ok((
                id,
                self.store
                    .update(cx, |store, cx| store.save_schedule(schedule, cx)),
            ))
        })();
        match result {
            Ok((id, task)) => {
                self.selected_schedule = Some(id);
                self.report(task, cx);
            }
            Err(error) => self.error = Some(error.to_string()),
        }
        cx.notify();
    }

    fn run_recipe(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .proposal
            .as_ref()
            .is_some_and(|proposal| proposal.status.get() == RecipeReviewStatus::Cancelled)
        {
            self.error = Some(
                "Draft request was cancelled. Close this review or start a new recipe.".into(),
            );
            cx.notify();
            return;
        }
        let result = (|| -> Result<_> {
            let recipe = Recipe::from_yaml(&self.yaml.read(cx).text(cx))?;
            let entry = RecipeEntry {
                id: "preview".into(),
                origin: "Recipe editor".into(),
                path: self.selected.as_ref().and_then(|entry| entry.path.clone()),
                recipe: Ok(recipe),
            };
            agent::recipes::start_recipe(
                self.connection.clone(),
                self.project.clone(),
                entry,
                self.parameter_values(cx),
                cx,
            )
        })();
        match result {
            Ok(session) => {
                let id = session.read(cx).session_id().clone();
                self.open_session(id, window, cx);
            }
            Err(error) => self.error = Some(error.to_string()),
        }
        cx.notify();
    }

    fn scan_review_key(&self, cx: &App) -> String {
        let mut parameters = self.parameter_values(cx).into_iter().collect::<Vec<_>>();
        parameters.sort();
        format!(
            "{:?}",
            (
                self.yaml.read(cx).text(cx),
                parameters,
                self.worktree,
                self.model
                    .borrow()
                    .as_ref()
                    .map(|model| (model.provider.id(), model.model.id())),
                self.scan_image.read(cx).text(cx),
                self.scan_endpoint.read(cx).text(cx),
                self.scan_server.read(cx).text(cx),
                self.scan_files.read(cx).text(cx),
            )
        )
    }

    fn select_tracked_scan_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| -> Result<_> {
            let worktree = self.worktree.context("Choose a local worktree")?;
            let path = ProjectPath {
                worktree_id: worktree,
                path: RelPath::empty().into(),
            };
            let (repository, _) = self
                .project
                .read(cx)
                .git_store()
                .read(cx)
                .repository_and_path_for_project_path(&path, cx)
                .context("No Git repository for this worktree; enter relative files explicitly")?;
            let task = repository.update(cx, |repository, _| repository.tracked_paths());
            Ok((worktree, repository, task))
        })();
        let (worktree, repository, task) = match result {
            Ok(value) => value,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self.scan_preview = None;
        self.scan_preparation = Some(cx.spawn_in(window, async move |this, cx| {
            let result = task
                .await
                .context("Git file selection was cancelled")
                .and_then(|result| result);
            this.update_in(cx, |this, window, cx| {
                this.scan_preparation = None;
                let result = result.and_then(|paths| {
                    anyhow::ensure!(
                        this.worktree == Some(worktree),
                        "Worktree changed; select files again"
                    );
                    let mut selected = Vec::new();
                    for path in paths {
                        let Some(path) = repository.read(cx).repo_path_to_project_path(&path, cx)
                        else {
                            continue;
                        };
                        if path.worktree_id != worktree {
                            continue;
                        }
                        anyhow::ensure!(
                            !path.path.as_unix_str().contains(['\n', '\r']),
                            "Select files with newline-free names explicitly"
                        );
                        selected.push(path.path.as_unix_str().to_owned());
                    }
                    selected.sort();
                    selected.dedup();
                    this.scan_files.update(cx, |editor, cx| {
                        editor.set_text(selected.join("\n"), window, cx)
                    });
                    Ok(())
                });
                this.error = result.err().map(|error| error.to_string());
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn prepare_scan(&mut self, cx: &mut Context<Self>) {
        self.scan_preview = None;
        let result = (|| -> Result<_> {
            let worktree = self
                .worktree
                .and_then(|id| self.project.read(cx).worktree_for_id(id, cx))
                .context("Choose an available local worktree")?;
            let model = self
                .model
                .borrow()
                .as_ref()
                .context("Choose a provider and model")?
                .model
                .clone();
            Ok(agent::recipe_scan::RecipeScanRequest {
                project: self.project.clone(),
                agent: self.connection.0.clone(),
                worktree: worktree.read(cx).abs_path().to_path_buf(),
                recipe: Recipe::from_yaml(&self.yaml.read(cx).text(cx))?,
                parameters: self.parameter_values(cx),
                selected_files: self
                    .scan_files
                    .read(cx)
                    .text(cx)
                    .lines()
                    .filter(|line| !line.is_empty())
                    .map(std::path::PathBuf::from)
                    .collect(),
                model,
                image: self.scan_image.read(cx).text(cx),
                endpoint: self.scan_endpoint.read(cx).text(cx),
                server_binary: self.scan_server.read(cx).text(cx).into(),
            })
        })();
        let request = match result {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        let key = self.scan_review_key(cx);
        let task = self
            .store
            .update(cx, |store, cx| store.prepare_scan(request, cx));
        self.error = None;
        self.scan_preparation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                this.scan_preparation = None;
                match result {
                    Ok(prepared) if this.scan_review_key(cx) == key => {
                        this.scan_preview = Some((key, prepared))
                    }
                    Ok(_) => {
                        this.error = Some(
                            "Test inputs changed during preparation. Prepare a new review.".into(),
                        )
                    }
                    Err(error) => {
                        this.error = Some(format!("Could not prepare Docker test: {error:#}"))
                    }
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn launch_scan(&mut self, cx: &mut Context<Self>) {
        let result = (|| -> Result<()> {
            let (key, prepared) = self
                .scan_preview
                .take()
                .context("Prepare and review the test before launch")?;
            anyhow::ensure!(
                key == self.scan_review_key(cx),
                "Test inputs changed. Prepare and review the new copy before launch."
            );
            self.store
                .update(cx, |store, cx| store.run_scan(prepared, true, cx))?;
            Ok(())
        })();
        self.error = result.err().map(|error| error.to_string());
        cx.notify();
    }

    fn render_scan(&self, disabled: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let active = self.store.read(cx).scan.as_ref();
        let running = active.is_some_and(|scan| scan.outcome.is_none());
        let phase = active.map(|scan| format!("{} · {}", scan.title, scan.phase));
        let observations = active
            .map(|scan| scan.observations.borrow().clone())
            .unwrap_or_default();
        let session = active
            .and_then(|scan| scan.session.as_ref())
            .map(|session| session.read(cx).session_id().clone());
        let history = self.store.read(cx).state.scans.clone();
        let preview_current = self
            .scan_preview
            .as_ref()
            .is_some_and(|(key, _)| *key == self.scan_review_key(cx));
        v_flex().w_full().min_w_0().gap_2()
            .child(Label::new("Test recipe in Docker"))
            .child(Label::new("Actual workflow execution in a disposable project copy. Original files are never mounted. Tools are offline; provider requests run through Made and disclose the copied context to the selected provider. Credentials stay with Made. Network or credential-dependent tools make the test incomplete.").size(LabelSize::Small))
            .child(Label::new("Select a local immutable image and a matching Linux Made server. Nothing is downloaded automatically. Copy only necessary files; exclude secrets. Relative file paths, one per line (empty means an empty project copy).").size(LabelSize::Small))
            .child(self.scan_image.clone()).child(self.scan_endpoint.clone()).child(self.scan_server.clone())
            .child(Button::new("scan-tracked", "Select tracked working-tree files").disabled(disabled || self.scan_preparation.is_some()).on_click(cx.listener(|this, _, window, cx| this.select_tracked_scan_files(window, cx))))
            .child(self.scan_files.clone())
            .child(Button::new("scan-prepare", "Prepare test for review").disabled(disabled || running || self.scan_preparation.is_some()).on_click(cx.listener(|this, _, _, cx| this.prepare_scan(cx))))
            .when(self.scan_preparation.is_some(), |view| view.child(Label::new("Preparing copy and checking local runtime…")).child(Button::new("scan-cancel-preparation", "Cancel preparation").on_click(cx.listener(|this, _, _, cx| { this.scan_preparation = None; this.scan_preview = None; cx.notify(); }))))
            .when_some(self.scan_preview.as_ref(), |view, (_, prepared)| {
                view.child(Label::new("Review this exact test"))
                    .child(Label::new(format!("Worktree: {}\nImage: {}\nRuntime SHA-256: {}\nProvider destination: {}", prepared.worktree().display(), prepared.image(), prepared.server_digest(), prepared.provider_endpoint())))
                    .child(Label::new(prepared.recipe().render(prepared.parameters()).unwrap_or_else(|error| error.to_string())))
                    .children(prepared.files().iter().map(|file| Label::new(format!("{} · {} bytes · SHA-256 {}", file.path.display(), file.bytes, file.sha256)).size(LabelSize::Small)))
                    .children(prepared.exclusions().iter().map(|file| Label::new(format!("Excluded {}: {}", file.path.display(), file.reason)).color(Color::Warning)))
                    .child(Label::new("Nonroot UID 1000; no Docker socket, host mounts, tool credentials or network. File/search and terminal tools only, with one-time approval. 2 CPUs, 2 GiB RAM, 256 processes; 15-minute workflow, 5-minute step and setup limits. This does not enable a schedule or unattended execution.").size(LabelSize::Small))
                    .when(!preview_current, |view| view.child(Label::new("Inputs changed. Prepare a new review.").color(Color::Warning)))
                    .child(div().debug_selector(|| "scan-launch".to_owned()).child(Button::new("scan-launch", "Authorize and launch this isolated test").full_width().disabled(disabled || running || !preview_current).on_click(cx.listener(|this, _, _, cx| this.launch_scan(cx)))))
                    .child(Button::new("scan-dismiss", "Discard test review").on_click(cx.listener(|this, _, _, cx| { this.scan_preview = None; cx.notify(); })))
            })
            .when_some(phase, |view, phase| view.child(Label::new(phase)))
            .children(observations.into_iter().map(Label::new))
            .when(running, |view| view.child(Button::new("scan-cancel", "Cancel test and clean up").on_click(cx.listener(|this, _, _, cx| this.store.update(cx, |store, cx| store.cancel_scan(cx))))))
            .when_some(session, |view, session| view.child(Button::new("scan-live-session", "Open test session and tool approvals").on_click(cx.listener(move |this, _, window, cx| this.open_session(session.clone(), window, cx)))))
            .child(Label::new("Docker test history"))
            .children(history.into_iter().rev().enumerate().map(|(index, scan)| {
                let id = scan.identity.run_id;
                let session = scan.session_id.map(acp::SessionId::new);
                let available = session.as_ref().is_some_and(|id| agent::ThreadStore::global(cx).read(cx).thread_from_session_id(id).is_some());
                v_flex().gap_1().child(Label::new(format!("{} · {} · {}", scan.created_at, scan.recipe.title, scan.outcome.map(agent::recipe_scan::RecipeScanOutcome::label).unwrap_or("Running"))))
                    .child(Label::new(format!("Worktree: {}\nProvider destination: {}\nRuntime SHA-256: {}", scan.worktree.display(), scan.provider_endpoint, scan.runtime_digest)).size(LabelSize::Small))
                    .child(Label::new(scan.recipe.render(&scan.parameters).unwrap_or_else(|error| error.to_string())).size(LabelSize::Small))
                    .children(scan.copied_files.iter().map(|file| Label::new(format!("Copied {} · {} bytes · {}", file.path.display(), file.bytes, file.sha256)).size(LabelSize::Small)))
                    .children(scan.exclusions.iter().map(|file| Label::new(format!("Excluded {}: {}", file.path.display(), file.reason)).size(LabelSize::Small)))
                    .children(scan.recipe.steps.iter().filter(|step| !scan.observations.iter().any(|observation| observation.contains(&format!("Step {}:", step.id)))).map(|step| Label::new(format!("Step {}: not reached", step.id))))
                    .when_some(scan.detail, |view, detail| view.child(Label::new(detail)))
                    .children(scan.observations.into_iter().map(Label::new))
                    .child(Label::new(format!("Test {} · {} · {} / {}", id, scan.identity.image, scan.provider, scan.model)).size(LabelSize::Small))
                    .when_some(scan.cleanup_error, |view, error| view.child(Label::new(format!("Cleanup required: {error}")).color(Color::Error)))
                    .when(scan.cleanup_pending, |view| view.child(Button::new(("scan-cleanup", index), "Retry cleanup").disabled(disabled || running || self.store.read(cx).scan_cleanup_active).on_click(cx.listener(move |this, _, _, cx| { let result = this.store.update(cx, |store, cx| store.retry_scan_cleanup(id, cx)); this.error = result.err().map(|error| error.to_string()); cx.notify(); }))))
                    .child(Button::new(("scan-history-session", index), if available { "Open test transcript" } else { "Session unavailable" }).disabled(!available).on_click(cx.listener(move |this, _, window, cx| { if let Some(session) = session.clone() { this.open_session(session, window, cx); } })))
            }))
            .child(Label::new("Success covers only the observed branches with these inputs, image and model. It does not prove unvisited branches, determinism, security, production compatibility or future scheduled success. Saved test transcripts are read-only.").size(LabelSize::Small))
    }

    fn open_session(&mut self, id: acp::SessionId, window: &mut Window, cx: &mut Context<Self>) {
        self.panel
            .update(cx, |panel, cx| {
                panel.open_thread(id, None, None, window, cx)
            })
            .log_err();
        cx.emit(DismissEvent);
    }
}

impl EventEmitter<DismissEvent> for RecipesView {}
impl ModalView for RecipesView {
    fn on_before_dismiss(
        &mut self,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> workspace::DismissDecision {
        self.dismiss_proposal();
        workspace::DismissDecision::Dismiss(true)
    }
}
impl Focusable for RecipesView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for RecipesView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.search.read(cx).text(cx).to_lowercase();
        let entries = self
            .entries
            .iter()
            .filter(|entry| {
                format!("{} {:?}", entry.origin, entry.recipe)
                    .to_lowercase()
                    .contains(&query)
            })
            .cloned()
            .collect::<Vec<_>>();
        let schedules = self
            .store
            .read(cx)
            .state
            .schedules
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let disabled = self.loading
            || self.operation.is_some()
            || self.store.read(cx).busy
            || self
                .proposal
                .as_ref()
                .is_some_and(|proposal| proposal.status.get() == RecipeReviewStatus::Cancelled);
        let rendered = Recipe::from_yaml(&self.yaml.read(cx).text(cx))
            .and_then(|recipe| recipe.render(&self.parameter_values(cx)));
        let selected_schedule = self
            .selected_schedule
            .and_then(|id| schedules.iter().find(|schedule| schedule.id == id))
            .cloned();
        let matches_saved = selected_schedule.as_ref().is_some_and(|schedule| {
            !schedule.deleted
                && schedule.title == self.title.read(cx).text(cx)
                && schedule.cron == self.cron.read(cx).text(cx)
                && schedule.timezone == self.timezone.read(cx).text(cx)
                && Recipe::from_yaml(&self.yaml.read(cx).text(cx))
                    .is_ok_and(|recipe| recipe == schedule.recipe)
                && schedule.parameters == self.parameter_values(cx)
                && self.model.borrow().as_ref().is_some_and(|model| {
                    model.provider.id().to_string() == schedule.provider
                        && model.model.id().0.as_ref() == schedule.model
                })
                && self
                    .worktree
                    .and_then(|id| self.project.read(cx).worktree_for_id(id, cx))
                    .is_some_and(|worktree| {
                        worktree.read(cx).abs_path().as_ref() == schedule.worktree.as_path()
                    })
        });
        let roots = self
            .project
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| {
                (
                    worktree.read(cx).id(),
                    worktree.read(cx).abs_path().display().to_string(),
                )
            })
            .collect::<Vec<_>>();
        let root_label = roots
            .iter()
            .find(|(id, _)| Some(*id) == self.worktree)
            .map(|(_, path)| path.clone())
            .unwrap_or_else(|| "Choose worktree".into());
        let weak = cx.entity().downgrade();
        let model_selector = self.model_selector.clone();
        let model_label = self
            .model
            .borrow()
            .as_ref()
            .map(|model| model.model.name().0.to_string())
            .unwrap_or_else(|| "Choose model".into());
        let error = self
            .error
            .clone()
            .or_else(|| self.store.read(cx).error.clone());
        let now = cx
            .background_executor()
            .scheduler_executor()
            .scheduler()
            .clock()
            .utc_now();
        let zone = self.timezone.read(cx).text(cx);
        let preview = next_run(&self.cron.read(cx).text(cx), &zone, now).map(|next| {
            zone.parse::<chrono_tz::Tz>()
                .map(|zone| next.with_timezone(&zone).to_string())
                .unwrap_or_else(|_| next.to_string())
        });
        v_flex().id("made-recipes").debug_selector(|| "made-recipes".to_owned()).key_context("RecipesView").track_focus(&self.focus)
            .on_action(cx.listener(|_, _: &menu::Cancel, _, cx| cx.emit(DismissEvent)))
            .w(rems(64.)).max_h(rems(48.)).overflow_y_scroll().p_4().gap_3().elevation_3(cx)
            .child(h_flex().justify_between().child(Label::new("Made Recipes and Schedules")).child(Button::new("close", "Close").on_click(cx.listener(|_, _, _, cx| cx.emit(DismissEvent)))))
            .when(self.proposal.is_some(), |view| view.child(Label::new("Agent draft for review. Nothing is enabled. Review the instructions and save; enabling unattended execution requires your separate consent.")))
            .when_some(error, |view, error| view.child(Label::new(error).color(Color::Error)))
            .when_some(self.link_notice.clone(), |view, notice| view.child(Label::new(notice)))
            .when(self.loading, |view| view.child(Label::new("Loading recipes…")))
            .child(h_flex().gap_2().child(self.search.clone()).child(Button::new("refresh", "Refresh").on_click(cx.listener(|this, _, _, cx| this.refresh(cx)))).child(Button::new("new", "New recipe").on_click(cx.listener(|this, _, window, cx| this.new_recipe(window, cx)))))
            .child(h_flex().w_full().items_start().gap_4()
                .child(v_flex().w(rems(20.)).flex_shrink_0().gap_2()
                    .child(Label::new("Recipes"))
                    .when(entries.is_empty(), |view| view.child(Label::new("No matching recipes. Create one in a trusted worktree.")))
                    .children(entries.into_iter().enumerate().map(|(index, entry)| {
                        let label = entry.recipe.as_ref().map(|recipe| recipe.title.clone()).unwrap_or_else(|_| "Invalid recipe".into());
                        v_flex().child(Label::new(entry.origin.clone()).size(LabelSize::Small)).child(Button::new(("recipe", index), label).on_click(cx.listener(move |this, _, window, cx| this.select_recipe(entry.clone(), window, cx))))
                    }))
                    .child(Label::new("Schedules"))
                    .when(schedules.is_empty(), |view| view.child(Label::new("No schedules yet. Choose a recipe and save a schedule.")))
                    .children(schedules.into_iter().enumerate().map(|(index, schedule)| {
                        let id = schedule.id;
                        Button::new(("schedule", index), format!("{} · {}", schedule.title, if schedule.deleted { "Deleted" } else if schedule.paused { "Paused" } else { "Enabled" })).on_click(cx.listener(move |this, _, window, cx| this.select_schedule(id, window, cx)))
                    })))
                .child(v_flex().flex_1().min_w_0().gap_2()
                    .child(PopoverMenu::new("worktree").trigger(Button::new("worktree-button", root_label)).menu(move |window, cx| Some(ContextMenu::build(window, cx, |mut menu, _, _| {
                        for (id, path) in &roots { let id = *id; let weak = weak.clone(); menu = menu.entry(path.clone(), None, move |_, cx| { weak.update(cx, |this, cx| { this.worktree = Some(id); cx.notify(); }).log_err(); }); }
                        menu
                    }))))
                    .child(h_flex().gap_2()
                        .child(Button::new("new-workflow", "New workflow").on_click(cx.listener(|this, _, window, cx| this.new_workflow(window, cx))))
                        .child(Button::new("add-workflow-step", "Add step").on_click(cx.listener(|this, _, window, cx| this.append_workflow_step(window, cx)))))
                    .child(h_flex().gap_2()
                        .child(Button::new("copy-recipe-link", "Copy recipe link").on_click(cx.listener(|this, _, _, cx| this.copy_recipe_link(false, cx))))
                        .child(Button::new("copy-recipe-source-link", "Copy source link").on_click(cx.listener(|this, _, _, cx| this.copy_recipe_link(true, cx)))))
                    .child(Label::new("Import a recipe for review"))
                    .child(self.import_url.clone()).child(self.import_revision.clone()).child(self.import_path.clone())
                    .child(Label::new("HTTPS only. No credential-bearing URLs, redirects, symlinks or dependency fetching. Repository URLs require an explicit revision and file path. Importing never runs a recipe or enables a schedule.").size(LabelSize::Small))
                    .child(Checkbox::new("import-github-token", if self.import_use_token { ToggleState::Selected } else { ToggleState::Unselected })
                        .label("Use existing GITHUB_TOKEN for this GitHub import")
                        .on_click(cx.listener(|this, state: &ToggleState, _, cx| { this.import_use_token = *state == ToggleState::Selected; cx.notify(); })))
                    .child(h_flex().gap_2()
                        .child(Button::new("fetch-recipe-import", if self.import_updating { "Fetch update for review" } else { "Fetch for review" }).disabled(self.import_task.is_some() || disabled).on_click(cx.listener(|this, _, window, cx| this.fetch_import(window, cx))))
                        .child(Button::new("prepare-recipe-update", "Review saved import source").on_click(cx.listener(|this, _, window, cx| this.prepare_import_update(window, cx))))
                        .child(Button::new("cancel-recipe-import", "Cancel import").disabled(self.import_task.is_none() && self.import_preview.is_none()).on_click(cx.listener(|this, _, _, cx| this.cancel_import(cx)))))
                    .when(self.import_task.is_some(), |view| view.child(Label::new("Fetching a bounded draft…")))
                    .when_some(self.import_preview.as_ref(), |view, preview| view
                        .child(Label::new("Current draft (preserved until you apply the review)"))
                        .child(Label::new(preview.previous_yaml.clone()))
                        .child(Label::new("Proposed import, including source identity"))
                        .child(Label::new(preview.yaml.clone()))
                        .child(Label::new("Review every instruction, step, parameter and command. Applying changes only the editor draft; save separately. Existing schedule snapshots stay unchanged.").size(LabelSize::Small))
                        .child(Button::new("apply-recipe-import", "Apply reviewed import to draft").disabled(disabled).on_click(cx.listener(|this, _, window, cx| this.apply_import_preview(window, cx)))))
                    .child(Label::new("Recipe YAML · parameters use {{name}}"))
                    .child(Label::new("Workflow steps run in order. Edit id, prompt, when, next and on_failure below. Branch targets must be later steps; output references must be earlier steps. Invalid edits are retained for correction.").size(LabelSize::Small))
                    .child(self.filename.clone()).child(self.yaml.clone())
                    .child(h_flex().gap_2()
                        .child(Button::new("validate", "Validate / refresh parameters").on_click(cx.listener(|this, _, window, cx| this.refresh_parameters(window, cx))))
                        .child(Button::new("save-recipe", "Save recipe").disabled(disabled || self.proposal.as_ref().is_some_and(|proposal| proposal.input.recipe_id.is_some())).on_click(cx.listener(|this, _, _, cx| this.save_recipe(cx))))
                        .child(Button::new("delete-recipe", "Delete recipe").disabled(disabled || self.selected.as_ref().is_none_or(|entry| entry.path.is_none())).on_click(cx.listener(|this, _, _, cx| {
                            if let Some(path) = this.selected.as_ref().and_then(|entry| entry.path.clone()) { let task = agent::recipes::delete_local_recipe(this.project.clone(), path, cx); this.report(task, cx); }
                        }))))
                    .children(self.parameters.iter().map(|(name, editor)| v_flex().child(Label::new(name.clone())).child(editor.clone())))
                    .child(Label::new("Instructions and workflow plan with parameters"))
                    .child(Label::new(rendered.unwrap_or_else(|error| error.to_string())))
                    .child(Button::new("run-recipe", "Run recipe in a new session").disabled(disabled).on_click(cx.listener(|this, _, window, cx| this.run_recipe(window, cx))))
                    .child(PopoverMenu::new("recipe-model").trigger(Button::new("model", model_label)).menu(move |_, _| Some(model_selector.clone())))
                    .child(self.render_scan(disabled, cx))
                    .child(Label::new("Recurring schedule"))
                    .child(Label::new("Parameters are saved as ordinary text. Keep secrets in provider credential settings.").size(LabelSize::Small))
                    .child(self.title.clone()).child(self.cron.clone()).child(self.timezone.clone())
                    .child(Label::new(match preview { Ok(next) => format!("Preview: {next}"), Err(error) => error.to_string() }).size(LabelSize::Small))
                    .child(Label::new("Runs only while Made is open. Missed runs and overlaps are skipped. Saved recipes are used; no retries or crash replay.").size(LabelSize::Small))
                    .child(Button::new("save-schedule", "Save schedule paused").disabled(disabled).on_click(cx.listener(|this, _, _, cx| this.save_schedule(cx))))
                    .when_some(selected_schedule, |view, schedule| {
                        let id = schedule.id;
                        let zone = schedule.timezone.parse::<chrono_tz::Tz>().ok();
                        let next = zone.map(|zone| schedule.next_run.with_timezone(&zone).to_string()).unwrap_or_else(|| schedule.next_run.to_string());
                        view.child(Label::new(format!("Next: {next}")))
                            .child(Label::new(format!("Saved worktree: {}", schedule.worktree.display())).size(LabelSize::Small))
                            .child(Label::new(format!("Project roots: {}", util::path_list::PathList::deserialize(&schedule.project_paths).paths().iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", "))).size(LabelSize::Small))
                            .when(!matches_saved && !schedule.deleted, |view| view.child(Label::new("Save your edits before enabling or running this schedule.").color(Color::Warning)))
                            .child(Label::new("Enabling consents to unattended execution of this saved recipe in this project with this model. Existing trust, sandbox and tool permissions still apply; requests for approval stop the run.").size(LabelSize::Small))
                            .child(h_flex().gap_2()
                                .child(div().debug_selector(|| "recipe-consent".to_owned()).child(Button::new("enable-schedule", if schedule.paused { "Enable with these permissions" } else { "Pause" }).disabled(disabled || schedule.deleted || (schedule.paused && !matches_saved)).on_click(cx.listener(move |this, _, _, cx| { if this.proposal.as_ref().is_some_and(|proposal| proposal.status.get() == RecipeReviewStatus::Cancelled) { this.error = Some("Draft request was cancelled".into()); cx.notify(); return; } let task = this.store.update(cx, |store, cx| store.set_enabled(id, schedule.paused, schedule.paused, cx)); this.report(task, cx); }))))
                                .child(Button::new("run-schedule", "Run saved schedule").disabled(disabled || !matches_saved).on_click(cx.listener(move |this, _, _, cx| { let task = this.store.update(cx, |store, cx| store.run_schedule(id, cx)); this.report(task, cx); })))
                                .child(Button::new("cancel-run", "Cancel run").disabled(disabled || self.store.read(cx).state.active_run().is_none_or(|run| run.schedule_id != id)).on_click(cx.listener(move |this, _, _, cx| { let result = this.store.update(cx, |store, cx| store.cancel_schedule(id, cx)); this.report(Task::ready(result), cx); })))
                                .child(Button::new("delete-schedule", "Delete schedule").disabled(disabled).on_click(cx.listener(move |this, _, _, cx| { let task = this.store.update(cx, |store, cx| store.delete_schedule(id, cx)); this.report(task, cx); }))))
                            .child(Label::new("Run history"))
                            .children(self.store.read(cx).state.runs.iter().rev().filter(|run| run.schedule_id == id).enumerate().map(|(index, run)| {
                                let session = run.session_id.clone().map(acp::SessionId::new);
                                let available = session.as_ref().is_some_and(|id| agent::ThreadStore::global(cx).read(cx).thread_from_session_id(id).is_some());
                                v_flex().child(Label::new(format!("{} · {}", run.scheduled_for, run.status.label())))
                                    .when_some(run.detail.clone(), |view, detail| view.child(Label::new(detail).size(LabelSize::Small)))
                                    .child(Button::new(("open-run", index), if available { "Open session" } else { "Session unavailable" }).disabled(!available).on_click(cx.listener(move |this, _, window, cx| { if let Some(id) = session.clone() { this.open_session(id, window, cx); } })))
                            }))
                    })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::{TestAppContext, VisualTestContext};
    use language_model::{LanguageModelRegistry, fake_provider::FakeLanguageModelProvider};
    use std::{path::Path, sync::Arc};
    use workspace::MultiWorkspace;

    #[gpui::test]
    async fn recipe_proposal_desktop_consent_persistence_and_session_flow(cx: &mut TestAppContext) {
        crate::conversation_view::tests::init_test(cx);
        let model = Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let provider = FakeLanguageModelProvider::default().with_models(vec![model.clone()]);
        cx.update(|cx| {
            LanguageModelRegistry::test(cx);
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(Arc::new(provider), cx)
            });
            agent::ThreadStore::init_global(cx);
        });
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project", serde_json::json!({})).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        cx.update(RecipeStore::init_global);
        cx.run_until_parked();
        let store = cx.update(|cx| RecipeStore::try_global(cx).expect("store"));
        let native = cx.update(|cx| {
            agent::NativeAgent::new(
                agent::ThreadStore::global(cx),
                agent::Templates::new(),
                fs,
                cx,
            )
        });
        store.update(cx, |store, _| store.register(&project, &native));
        let connection = Rc::new(agent::NativeAgentConnection(native));
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |workspace, _| workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        cx.simulate_resize(gpui::size(gpui::px(1200.), gpui::px(1000.)));
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        use acp_thread::AgentConnection as _;
        use agent::{AgentTool as _, ProposeRecipeTool, ToolCallEventStream, ToolInput};
        let session = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    project
                        .read(cx)
                        .worktree_paths(cx)
                        .folder_path_list()
                        .clone(),
                    cx,
                )
            })
            .await
            .expect("session");
        let thread = cx.update(|_, cx| {
            connection
                .thread(session.read(cx).session_id(), cx)
                .expect("native thread")
        });
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let proposal = agent::recipes::RecipeProposalInput {
            proposal_id: "weekday-risk".into(),
            recipe_yaml: Some("version: 1\ntitle: Review risks\ndescription: Review project risks\nprompt: Review {{scope}} for risky changes.\nparameters:\n  - name: scope\n    description: Review scope\n    required: true\n".into()),
            recipe_id: None, filename: Some("weekday-risk.yaml".into()),
            parameters: HashMap::from_iter([("scope".into(), "this project".into())]),
            worktree: None, provider: None, model: None,
            schedule: Some(agent::recipes::RecipeRecurrenceProposal { title: "Weekday risk review".into(), cron: "0 9 * * 1-5".into(), timezone: "Europe/London".into() }),
        };
        let tool = Arc::new(ProposeRecipeTool::new(thread.downgrade()));
        let turn = session.update(cx, |session, cx| {
            session.send_raw(
                "Every weekday at 9 AM Europe/London, review this project for risky changes",
                cx,
            )
        });
        let turn = cx.spawn(async move |_| turn.await);
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("model request");
        assert!(
            request
                .tools
                .iter()
                .any(|tool| tool.name == "propose_recipe")
        );
        let arguments = serde_json::to_value(&proposal).expect("tool input");
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::ToolUse(
                language_model::LanguageModelToolUse {
                    id: "proposal-call".into(),
                    name: "propose_recipe".into(),
                    raw_input: arguments.to_string(),
                    input: language_model::LanguageModelToolUseInput::Json(arguments),
                    is_input_complete: true,
                    thought_signature: None,
                },
            ),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        cx.background_executor
            .advance_clock(std::time::Duration::from_millis(50));
        cx.run_until_parked();
        let response = model.pending_completions().pop().expect("tool response");
        assert!(
            response
                .messages
                .iter()
                .any(|message| message.string_contents().contains("not saved"))
        );
        model.send_last_completion_stream_text_chunk(
            "The draft is open for review; save and explicitly enable it there.",
        );
        model.end_last_completion_stream();
        cx.run_until_parked();
        turn.await.expect("conversation completed");
        let view = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_modal::<RecipesView>(cx)
            })
            .expect("real management modal");
        let mut blocked = proposal.clone();
        blocked.proposal_id = "blocked-review".into();
        blocked.filename = Some("blocked.yaml".into());
        assert!(
            cx.update(|_, cx| tool.clone().run(
                ToolInput::resolved(&blocked),
                ToolCallEventStream::test().0,
                cx
            ))
            .await
            .expect_err("active modal preserved")
            .contains("could not open")
        );
        view.read_with(cx, |view, cx| {
            assert_eq!(view.title.read(cx).text(cx), "Weekday risk review");
            assert_eq!(view.cron.read(cx).text(cx), "0 9 * * 1-5");
            assert_eq!(view.timezone.read(cx).text(cx), "Europe/London");
            assert_eq!(
                Recipe::from_yaml(&view.yaml.read(cx).text(cx))
                    .expect("recipe")
                    .render(&view.parameter_values(cx))
                    .expect("render"),
                "Review this project for risky changes."
            );
        });
        assert!(store.read_with(cx, |store, _| store.state.schedules.is_empty()));
        view.update_in(cx, |view, window, cx| {
            view.title.update(cx, |editor, cx| {
                editor.set_text("User edited review", window, cx)
            })
        });
        cx.update(|_, cx| {
            tool.clone().run(
                ToolInput::resolved(&proposal),
                ToolCallEventStream::test().0,
                cx,
            )
        })
        .await
        .expect("repeat");
        assert_eq!(
            view.read_with(cx, |view, cx| view.title.read(cx).text(cx)),
            "User edited review"
        );
        view.update(cx, |view, cx| view.save_recipe(cx));
        cx.run_until_parked();
        store.update(cx, |store, _| {
            store.error = Some("Storage unavailable".into())
        });
        view.update(cx, |view, cx| view.save_schedule(cx));
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.error.is_some()));
        assert!(store.read_with(cx, |store, _| store.state.schedules.is_empty()));
        store.update(cx, |store, _| store.error = None);
        view.update(cx, |view, cx| view.save_schedule(cx));
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.error.is_none(), "{:?}", view.error)
        });
        let id = view.read_with(cx, |view, _| view.selected_schedule.expect("saved"));
        view.update(cx, |view, cx| view.save_schedule(cx));
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(store.state.schedules.len(), 1);
            let schedule = store.state.schedules.first().expect("schedule");
            assert!(schedule.paused && !schedule.consented);
            assert!(store.state.runs.is_empty());
        });
        // Drive the actual consent button rather than invoking the owner's consent API.
        let modal = cx.debug_bounds("made-recipes").expect("modal bounds");
        cx.simulate_event(gpui::ScrollWheelEvent {
            position: modal.center(),
            delta: gpui::ScrollDelta::Pixels(gpui::point(gpui::px(0.), gpui::px(-2000.))),
            modifiers: gpui::Modifiers::default(),
            touch_phase: gpui::TouchPhase::Moved,
        });
        cx.run_until_parked();
        let bounds = cx.debug_bounds("recipe-consent").expect("consent control");
        cx.simulate_click(bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| {
            store.state.schedules.first().expect("schedule").consented
        }));
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("manual run");
        cx.run_until_parked();
        model.send_last_completion_stream_text_chunk("Risk review complete");
        model.end_last_completion_stream();
        cx.run_until_parked();
        let session_id = store.read_with(cx, |store, _| {
            let run = store.state.runs.last().expect("history");
            assert_eq!(
                run.status,
                agent::recipe_schedule::RecipeRunStatus::Succeeded
            );
            acp::SessionId::new(run.session_id.clone().expect("session link"))
        });
        view.update_in(cx, |view, window, cx| {
            view.open_session(session_id, window, cx)
        });
        cx.run_until_parked();
        assert!(panel.read_with(cx, |panel, cx| panel.active_agent_thread(cx).is_some()));
        let repeated = cx
            .update(|_, cx| {
                tool.clone().run(
                    ToolInput::resolved(&proposal),
                    ToolCallEventStream::test().0,
                    cx,
                )
            })
            .await
            .expect("dismissed");
        assert!(repeated.contains("closed"));
        assert!(workspace.read_with(cx, |workspace, cx| {
            workspace.active_modal::<RecipesView>(cx).is_none()
        }));
        cx.update(|_, cx| {
            tool.clone().run(
                ToolInput::resolved(&blocked),
                ToolCallEventStream::test().0,
                cx,
            )
        })
        .await
        .expect("retry failed handoff");
        let retried = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_modal::<RecipesView>(cx)
            })
            .expect("retried review");
        retried.update_in(cx, |_, _, cx| cx.emit(DismissEvent));
        cx.run_until_parked();
        let mut existing = proposal.clone();
        existing.proposal_id = "existing-recipe".into();
        existing.recipe_yaml = None;
        existing.recipe_id = Some("builtin/release-risk".into());
        existing.filename = None;
        existing.parameters.clear();
        cx.update(|_, cx| {
            tool.clone().run(
                ToolInput::resolved(&existing),
                ToolCallEventStream::test().0,
                cx,
            )
        })
        .await
        .expect("existing recipe review");
        let existing_view = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_modal::<RecipesView>(cx)
            })
            .expect("existing recipe modal");
        existing_view.update_in(cx, |_, _, cx| cx.emit(DismissEvent));
        cx.run_until_parked();
        let mut recipe_only = proposal.clone();
        recipe_only.proposal_id = "recipe-only".into();
        recipe_only.filename = Some("only.yaml".into());
        recipe_only.schedule = None;
        cx.update(|_, cx| {
            tool.run(
                ToolInput::resolved(&recipe_only),
                ToolCallEventStream::test().0,
                cx,
            )
        })
        .await
        .expect("recipe-only review");
        let only_view = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_modal::<RecipesView>(cx)
            })
            .expect("recipe-only modal");
        only_view.update(cx, |view, cx| view.save_recipe(cx));
        cx.run_until_parked();
        only_view.read_with(cx, |view, _| {
            assert!(view.error.is_none(), "{:?}", view.error)
        });
        assert_eq!(
            store.read_with(cx, |store, _| store.state.schedules.len()),
            1
        );
    }

    #[gpui::test]
    async fn recipe_management_schedule_flow(cx: &mut TestAppContext) {
        recipe_management_flow(cx, RecipeFlow::Prompt).await;
    }

    #[gpui::test]
    async fn recipe_workflow_authoring_schedule_and_session_flow(cx: &mut TestAppContext) {
        recipe_management_flow(cx, RecipeFlow::Workflow).await;
    }

    #[gpui::test]
    async fn recipe_import_review_cancellation_update_and_session_flow(cx: &mut TestAppContext) {
        recipe_management_flow(cx, RecipeFlow::Imported).await;
    }

    #[gpui::test]
    async fn recipe_link_review_preserves_drafts_and_requires_explicit_actions(
        cx: &mut TestAppContext,
    ) {
        recipe_management_flow(cx, RecipeFlow::Linked).await;
    }

    #[gpui::test]
    async fn recipe_scan_review_rejects_unprepared_launch_and_cancel_preserves_consent(
        cx: &mut TestAppContext,
    ) {
        recipe_management_flow(cx, RecipeFlow::ScanReview).await;
    }

    #[gpui::test]
    #[ignore = "requires local Docker, an immutable image and matching Linux Made server"]
    async fn recipe_scan_real_desktop_review_execution_and_history(cx: &mut TestAppContext) {
        cx.background_executor.allow_parking();
        recipe_management_flow(cx, RecipeFlow::Docker).await;
    }

    enum RecipeFlow {
        Docker,
        ScanReview,
        Prompt,
        Workflow,
        Imported,
        Linked,
    }

    async fn recipe_management_flow(cx: &mut TestAppContext, flow: RecipeFlow) {
        let workflow = matches!(flow, RecipeFlow::Workflow);
        let imported = matches!(flow, RecipeFlow::Imported);
        crate::conversation_view::tests::init_test(cx);
        let model = Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let provider = FakeLanguageModelProvider::default().with_models(vec![model.clone()]);
        cx.update(|cx| {
            LanguageModelRegistry::test(cx);
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                let provider = Arc::new(provider);
                registry.register_provider(provider.clone(), cx);
                registry.set_default_model(
                    Some(ConfiguredModel {
                        provider,
                        model: model.clone(),
                    }),
                    cx,
                );
            });
            agent::ThreadStore::init_global(cx);
        });
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        let root = matches!(flow, RecipeFlow::Docker)
            .then(|| tempfile::tempdir().expect("disposable source"));
        let path = root
            .as_ref()
            .map(|root| root.path())
            .unwrap_or(Path::new("/project"));
        if let Some(root) = &root {
            std::fs::write(root.path().join("sentinel"), "original").expect("source sentinel");
            cx.update(|cx| {
                release_channel::init(remote_server::VERSION.parse().expect("runtime version"), cx);
                release_channel::AppCommitSha::set_global(
                    release_channel::AppCommitSha::new(
                        std::env::var("MADE_RECIPE_TEST_COMMIT").expect("runtime commit"),
                    ),
                    cx,
                );
            });
        }
        fs.insert_tree(path, serde_json::json!({"sentinel": "original"}))
            .await;
        let project = Project::test(fs.clone(), [path], cx).await;
        cx.update(RecipeStore::init_global);
        cx.run_until_parked();
        let store = cx.update(|cx| RecipeStore::try_global(cx).expect("store"));
        let native = cx.update(|cx| {
            agent::NativeAgent::new(
                agent::ThreadStore::global(cx),
                agent::Templates::new(),
                fs,
                cx,
            )
        });
        store.update(cx, |store, _| store.register(&project, &native));
        let connection = Rc::new(agent::NativeAgentConnection(native));
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |workspace, _| workspace.workspace().clone())
            .expect("workspace");
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        cx.simulate_resize(gpui::size(gpui::px(1200.), gpui::px(1000.)));
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let mut view = None;
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.toggle_modal(window, cx, |window, cx| {
                view = Some(cx.entity());
                RecipesView::new(
                    panel.downgrade(),
                    project,
                    connection,
                    store.clone(),
                    None,
                    window,
                    cx,
                )
            })
        });
        let mut view = view.expect("management view");
        cx.run_until_parked();
        if matches!(flow, RecipeFlow::Docker) {
            view.update_in(cx, |view, window, cx| {
                view.scan_image.update(cx, |editor, cx| {
                    editor.set_text(
                        std::env::var("MADE_RECIPE_TEST_IMAGE").expect("image"),
                        window,
                        cx,
                    )
                });
                view.scan_endpoint.update(cx, |editor, cx| {
                    editor.set_text(
                        std::env::var("MADE_RECIPE_TEST_DOCKER_HOST").expect("endpoint"),
                        window,
                        cx,
                    )
                });
                view.scan_server.update(cx, |editor, cx| {
                    editor.set_text(
                        std::env::var("MADE_RECIPE_TEST_SERVER").expect("runtime"),
                        window,
                        cx,
                    )
                });
                view.scan_files
                    .update(cx, |editor, cx| editor.set_text("sentinel", window, cx));
                view.new_workflow(window, cx);
                view.prepare_scan(cx);
            });
            let preparation = view.update(cx, |view, _| {
                view.scan_preparation.take().expect("preparation task")
            });
            preparation.await;
            view.read_with(cx, |view, _| {
                assert!(view.scan_preview.is_some(), "{:?}", view.error)
            });
            assert!(store.read_with(cx, |store, _| store.state.scans.is_empty()));
            assert!(model.pending_completions().is_empty());
            // Changing a reviewed field cannot authorize the earlier snapshot.
            view.update_in(cx, |view, window, cx| {
                view.scan_files
                    .update(cx, |editor, cx| editor.set_text("", window, cx));
                view.launch_scan(cx);
                assert!(
                    view.error
                        .as_ref()
                        .expect("stale review")
                        .contains("changed")
                );
                view.scan_files
                    .update(cx, |editor, cx| editor.set_text("sentinel", window, cx));
                view.prepare_scan(cx);
            });
            view.update(cx, |view, _| {
                view.scan_preparation.take().expect("new preparation")
            })
            .await;
            cx.run_until_parked();
            let bounds = cx
                .debug_bounds("scan-launch")
                .expect("review launch button");
            let modal = cx.debug_bounds("made-recipes").expect("management modal");
            cx.simulate_event(gpui::ScrollWheelEvent {
                position: modal.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(
                    gpui::px(0.),
                    modal.center().y - bounds.center().y,
                )),
                modifiers: gpui::Modifiers::default(),
                touch_phase: gpui::TouchPhase::Moved,
            });
            cx.run_until_parked();
            let launch = cx
                .debug_bounds("scan-launch")
                .expect("visible launch button");
            // Pumping to parked here would wait on workflow timers before the fake model can answer.
            cx.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                        position: launch.center(),
                        modifiers: gpui::Modifiers::default(),
                        button: gpui::MouseButton::Left,
                        click_count: 1,
                        first_mouse: false,
                    }),
                    cx,
                );
                window.dispatch_event(
                    gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                        position: launch.center(),
                        modifiers: gpui::Modifiers::default(),
                        button: gpui::MouseButton::Left,
                        click_count: 1,
                    }),
                    cx,
                );
            });
            assert!(
                store.read_with(cx, |store, _| store.scan.is_some()),
                "launch bounds {launch:?}; modal {modal:?}; error {:?}; review {}",
                view.read_with(cx, |view, _| view.error.clone()),
                view.read_with(cx, |view, cx| view
                    .scan_preview
                    .as_ref()
                    .is_some_and(|(key, _)| *key == view.scan_review_key(cx)))
            );
            let run_id = store.read_with(cx, |store, _| store.scan.as_ref().expect("launched").id);
            view.update(cx, |view, cx| view.launch_scan(cx));
            assert_eq!(
                store.read_with(cx, |store, _| store.scan.as_ref().expect("same run").id),
                run_id
            );
            let mut completed = false;
            for _ in 0..6000 {
                for request in model.pending_completions() {
                    model.send_completion_stream_text_chunk(&request, "Observed review");
                    model.send_completion_stream_event(
                        &request,
                        language_model::LanguageModelCompletionEvent::Stop(
                            language_model::StopReason::EndTurn,
                        ),
                    );
                    model.end_completion_stream(&request);
                }
                completed =
                    store.read_with(cx, |store, _| {
                        store.state.scans.iter().any(|scan| {
                            scan.identity.run_id == run_id && scan.finished_at.is_some()
                        })
                    });
                if completed {
                    break;
                }
                cx.background_executor
                    .timer(std::time::Duration::from_millis(20))
                    .await;
            }
            assert!(completed, "test and cleanup must finish");
            let session = store.read_with(cx, |store, _| {
                let record = store.state.scans.last().expect("history");
                assert_eq!(
                    record.outcome,
                    Some(agent::recipe_scan::RecipeScanOutcome::Succeeded),
                    "{:?}",
                    record.detail
                );
                assert!(!record.cleanup_pending, "{:?}", record.cleanup_error);
                assert!(!record.observations.is_empty());
                assert!(store.state.schedules.is_empty());
                acp::SessionId::new(record.session_id.clone().expect("session link"))
            });
            assert_eq!(
                std::fs::read_to_string(root.as_ref().expect("source").path().join("sentinel"))
                    .expect("sentinel"),
                "original"
            );
            view.update_in(cx, |view, window, cx| {
                view.open_session(session, window, cx)
            });
            cx.run_until_parked();
            let session = panel.read_with(cx, |panel, cx| {
                panel.active_agent_thread(cx).expect("opened transcript")
            });
            session.read_with(cx, |session, cx| {
                let connection = session
                    .connection()
                    .clone()
                    .downcast::<agent::NativeAgentConnection>()
                    .expect("native owner");
                assert!(
                    connection
                        .thread(session.session_id(), cx)
                        .expect("native thread")
                        .read(cx)
                        .is_recipe_scan_history()
                );
            });
            return;
        }
        if matches!(flow, RecipeFlow::ScanReview) {
            view.update_in(cx, |view, window, cx| {
                let before = view.scan_review_key(cx);
                view.scan_files
                    .update(cx, |editor, cx| editor.set_text("README.md", window, cx));
                assert_ne!(before, view.scan_review_key(cx));
                for _ in 0..2 {
                    view.launch_scan(cx);
                    assert!(
                        view.error
                            .as_ref()
                            .expect("review required")
                            .contains("review")
                    );
                }
                let original = view.yaml.read(cx).text(cx);
                view.worktree = None;
                view.prepare_scan(cx);
                assert!(
                    view.error
                        .as_ref()
                        .expect("worktree required")
                        .contains("worktree")
                );
                assert!(view.scan_preparation.is_none());
                view.scan_preparation = Some(cx.spawn(async |_, cx| {
                    cx.background_executor()
                        .timer(std::time::Duration::from_secs(300))
                        .await;
                    panic!("Cancelled preparation must not complete");
                }));
                view.scan_preparation = None;
                view.scan_preview = None;
                assert_eq!(view.yaml.read(cx).text(cx), original);
            });
            cx.run_until_parked();
            assert!(store.read_with(cx, |store, _| store.scan.is_none()
                && store.state.scans.is_empty()
                && store.state.schedules.is_empty()));
            assert!(model.pending_completions().is_empty());
            view.update_in(cx, |_, _, cx| cx.emit(DismissEvent));
            return;
        }
        if matches!(flow, RecipeFlow::Linked) {
            let yaml = "version: 1\ntitle: Linked review\ndescription: Review linked instructions\nprompt: Review project changes\n".to_owned();
            let link = agent::recipe_import::RecipeLink::Embedded { yaml: yaml.clone() };
            let original = view.read_with(cx, |view, cx| view.yaml.read(cx).text(cx));
            panel.update_in(cx, |panel, window, cx| {
                panel.review_recipe_link(link.clone(), window, cx)
            });
            cx.run_until_parked();
            let active = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<RecipesView>(cx)
                })
                .expect("original dialog");
            assert_eq!(active.entity_id(), view.entity_id());
            assert_eq!(
                view.read_with(cx, |view, cx| view.yaml.read(cx).text(cx)),
                original
            );
            view.update_in(cx, |_, _, cx| cx.emit(DismissEvent));
            cx.run_until_parked();
            let source = agent::recipe_import::RecipeLink::Source(
                agent::recipe_import::RecipeImportRequest {
                    url: "https://example.com/review.yaml".into(),
                    ..Default::default()
                },
            );
            panel.update_in(cx, |panel, window, cx| {
                panel.review_recipe_link(source, window, cx)
            });
            cx.run_until_parked();
            let source_view = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<RecipesView>(cx)
                })
                .expect("source review");
            source_view.read_with(cx, |view, cx| {
                assert_eq!(
                    view.import_url.read(cx).text(cx),
                    "https://example.com/review.yaml"
                );
                assert!(view.yaml.read(cx).text(cx).is_empty());
                assert!(view.import_task.is_none());
                assert!(!view.import_use_token);
            });
            assert!(model.pending_completions().is_empty());
            assert!(store.read_with(cx, |store, _| store.state.schedules.is_empty()));
            source_view.update_in(cx, |_, _, cx| cx.emit(DismissEvent));
            cx.run_until_parked();
            panel.update_in(cx, |panel, window, cx| {
                panel.review_recipe_link(link.clone(), window, cx)
            });
            cx.run_until_parked();
            view = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<RecipesView>(cx)
                })
                .expect("embedded review");
            view.update(cx, |view, cx| {
                assert_eq!(view.yaml.read(cx).text(cx), yaml);
                assert!(view.selected.is_none());
                assert!(view.selected_schedule.is_none());
                view.copy_recipe_link(false, cx);
                let copied = cx
                    .read_from_clipboard()
                    .and_then(|item| item.text())
                    .expect("copied link");
                assert_eq!(
                    agent::recipe_import::RecipeLink::decode(&copied).expect("valid copied link"),
                    link
                );
            });
            assert!(model.pending_completions().is_empty());
            assert!(store.read_with(cx, |store, _| store.state.schedules.is_empty()));
        }
        if imported {
            view.update_in(cx, |view, window, cx| {
                let original = view.yaml.read(cx).text(cx);
                view.import_url.update(cx, |editor, cx| editor.set_text("http://localhost/recipe.yaml", window, cx));
                view.fetch_import(window, cx);
                assert!(view.error.is_some());
                assert!(view.import_task.is_none());
                assert_eq!(view.yaml.read(cx).text(cx), original);
                let mut recipe = Recipe::from_yaml("version: 1\ntitle: Imported review\ndescription: Reviewed remote instructions\nprompt: Review this project\n").expect("recipe");
                recipe.imported_from = Some(agent::recipe_import::RecipeProvenance {
                    request: agent::recipe_import::RecipeImportRequest { url: "https://example.com/review.yaml".into(), ..Default::default() },
                    commit: None, sha256: "a".repeat(64),
                });
                let yaml = recipe.to_yaml().expect("import draft");
                view.complete_import_fetch(Ok(yaml.clone()), original.clone(), false, None, cx);
                assert_eq!(view.yaml.read(cx).text(cx), original);
                view.cancel_import(cx);
                assert!(view.import_preview.is_none());
                assert_eq!(view.yaml.read(cx).text(cx), original);
                view.complete_import_fetch(Ok(yaml.clone()), original.clone(), false, None, cx);
                view.yaml.update(cx, |editor, cx| editor.set_text("changed draft", window, cx));
                view.apply_import_preview(window, cx);
                assert!(view.error.as_ref().expect("stale review error").contains("changed"));
                assert_eq!(view.yaml.read(cx).text(cx), "changed draft");
                view.yaml.update(cx, |editor, cx| editor.set_text(original, window, cx));
                view.apply_import_preview(window, cx);
                assert_eq!(view.yaml.read(cx).text(cx), yaml);
                assert!(view.selected.is_none());
            });
            assert!(model.pending_completions().is_empty());
            assert!(store.read_with(cx, |store, _| store.state.schedules.is_empty()));
        }
        if workflow {
            view.update_in(cx, |view, window, cx| {
                view.new_workflow(window, cx);
                view.append_workflow_step(window, cx);
                let valid = view.yaml.read(cx).text(cx);
                let preview = Recipe::from_yaml(&valid)
                    .expect("valid workflow")
                    .render(&view.parameter_values(cx))
                    .expect("preview");
                assert!(preview.contains("Step review"));
                assert!(preview.contains("resolved from observed output at run time"));
                assert!(preview.contains("On failure: stop workflow"));
                view.yaml.update(cx, |editor, cx| {
                    editor.set_text("version: [invalid", window, cx)
                });
                view.append_workflow_step(window, cx);
                assert!(view.error.is_some());
                assert_eq!(view.yaml.read(cx).text(cx), "version: [invalid");
                view.yaml
                    .update(cx, |editor, cx| editor.set_text(valid, window, cx));
                view.refresh_parameters(window, cx);
            });
        }
        view.update(cx, |view, cx| view.save_recipe(cx));
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.error.is_none(), "{:?}", view.error)
        });
        view.update_in(cx, |view, window, cx| {
            view.title
                .update(cx, |title, cx| title.set_text("Daily review", window, cx));
            view.save_schedule(cx);
        });
        cx.run_until_parked();
        let id = view.read_with(cx, |view, _| {
            assert!(view.error.is_none(), "{:?}", view.error);
            view.selected_schedule.expect("saved schedule")
        });
        if imported {
            let snapshot = store.read_with(cx, |store, _| {
                store
                    .state
                    .schedules
                    .first()
                    .expect("schedule")
                    .recipe
                    .clone()
            });
            view.update_in(cx, |view, window, cx| {
                view.prepare_import_update(window, cx);
                assert_eq!(
                    view.import_url.read(cx).text(cx),
                    "https://example.com/review.yaml"
                );
                assert!(view.import_task.is_none());
                assert!(!view.import_use_token);
                let previous = view.yaml.read(cx).text(cx);
                let mut updated = Recipe::from_yaml(&previous).expect("saved import");
                updated.prompt = "Updated instructions requiring a new review".into();
                updated.imported_from.as_mut().expect("source").sha256 = "b".repeat(64);
                let selected = view.selected.as_ref().map(|entry| entry.id.clone());
                view.complete_import_fetch(
                    Ok(updated.to_yaml().expect("updated draft")),
                    previous,
                    true,
                    selected.clone(),
                    cx,
                );
                view.apply_import_preview(window, cx);
                assert_eq!(
                    view.selected.as_ref().map(|entry| entry.id.clone()),
                    selected
                );
                view.save_recipe(cx);
            });
            cx.run_until_parked();
            view.read_with(cx, |view, _| {
                assert!(view.error.is_none(), "{:?}", view.error)
            });
            store.read_with(cx, |store, _| {
                let schedule = store.state.schedules.first().expect("schedule");
                assert_eq!(schedule.recipe, snapshot);
                assert!(schedule.paused && !schedule.consented);
            });
        }
        let enable = store.update(cx, |store, cx| store.set_enabled(id, true, true, cx));
        enable.await.expect("enable");
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("run");
        cx.run_until_parked();
        assert!(!model.pending_completions().is_empty());
        for _ in 0..if workflow { 3 } else { 1 } {
            model.send_last_completion_stream_text_chunk("Review complete");
            model.end_last_completion_stream();
            cx.run_until_parked();
        }
        let session_id = store.read_with(cx, |store, _| {
            let run = store.state.runs.last().expect("history");
            assert_eq!(
                run.status,
                agent::recipe_schedule::RecipeRunStatus::Succeeded
            );
            acp::SessionId::new(run.session_id.clone().expect("session link"))
        });
        assert!(cx.debug_bounds("made-recipes").is_some());
        view.update_in(cx, |view, window, cx| {
            view.open_session(session_id, window, cx)
        });
        cx.run_until_parked();
        assert!(panel.read_with(cx, |panel, cx| panel.active_agent_thread(cx).is_some()));
    }
}
