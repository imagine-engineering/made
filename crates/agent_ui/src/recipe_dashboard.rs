use std::collections::BTreeMap;

use agent::recipe_schedule::{
    RecipeActivity, RecipeActivityStep, RecipeRunStatus, RecipeSchedule, RecipeStepStatus,
};
use agent::recipe_workflow::FailurePolicy;
use agent::recipes::{Recipe, RecipeEntry, RecipeStore};
use collections::{HashMap, HashSet};
use editor::Editor;
use gpui::{
    AnyElement, App, Context, Entity, EventEmitter, FocusHandle, Focusable, Render, SharedString,
    WeakEntity, Window, canvas,
};
use project::Project;
use ui::{ButtonLike, Divider, Headline, HeadlineSize, TintColor, Tooltip, prelude::*};
use util::{ResultExt as _, path_list::PathList};
use uuid::Uuid;
use workspace::{Item, Workspace, item::ItemEvent};

use crate::{AgentPanel, recipes_view::schedule_time_label};

#[derive(Clone)]
pub(crate) enum RecipePage {
    Library,
    Definition(RecipeEntry),
    Run(Uuid),
}

pub(crate) struct RecipeDashboard {
    project: Entity<Project>,
    panel: WeakEntity<AgentPanel>,
    store: Entity<RecipeStore>,
    page: RecipePage,
    focus: FocusHandle,
    entries: Vec<RecipeEntry>,
    inputs: Vec<(String, Entity<Editor>)>,
    search: Entity<Editor>,
    collapsed: HashSet<SharedString>,
    history_expanded: Option<bool>,
    narrow: bool,
    browsing: bool,
    selected_step: Option<usize>,
    loading: bool,
    error: Option<String>,
}

impl RecipeDashboard {
    pub(crate) fn open(
        workspace: &mut Workspace,
        page: RecipePage,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let Some(store) = RecipeStore::try_global(cx) else {
            workspace.show_error("Recipe storage is unavailable", cx);
            return;
        };
        let Some(panel) = workspace.panel::<AgentPanel>(cx) else {
            workspace.show_error("The built-in agent is unavailable", cx);
            return;
        };
        let existing = workspace.items_of_type::<Self>(cx).next();
        if let Some(existing) = existing {
            if !matches!(page, RecipePage::Library) {
                existing.update(cx, |view, cx| view.open_page(page, window, cx));
            } else {
                existing.update(cx, |view, cx| {
                    view.browsing = true;
                    cx.notify();
                });
            }
            workspace.activate_item(&existing, true, true, window, cx);
            return;
        }
        let project = workspace.project().clone();
        let item = cx.new(|cx| {
            let search = cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_placeholder_text("Search recipes, runs and schedules", window, cx);
                editor
            });
            cx.observe(&search, |_, _, cx| cx.notify()).detach();
            cx.observe(&store, |_, _, cx| cx.notify()).detach();
            let mut dashboard = Self {
                project,
                panel: panel.downgrade(),
                store,
                page: RecipePage::Library,
                focus: cx.focus_handle(),
                entries: Vec::new(),
                inputs: Vec::new(),
                search,
                collapsed: HashSet::default(),
                history_expanded: None,
                narrow: false,
                browsing: true,
                selected_step: None,
                loading: false,
                error: None,
            };
            dashboard.open_page(page, window, cx);
            dashboard.refresh(cx);
            dashboard
        });
        workspace.add_item_to_active_pane(Box::new(item), None, true, window, cx);
    }

    fn open_page(&mut self, page: RecipePage, window: &mut Window, cx: &mut Context<Self>) {
        let same_page = match (&self.page, &page) {
            (RecipePage::Library, RecipePage::Library) => true,
            (RecipePage::Definition(current), RecipePage::Definition(next)) => {
                current.id == next.id
            }
            (RecipePage::Run(current), RecipePage::Run(next)) => current == next,
            _ => false,
        };
        if !same_page {
            self.inputs.clear();
            self.selected_step = None;
            if let RecipePage::Definition(entry) = &page
                && let Ok(recipe) = &entry.recipe
            {
                for parameter in &recipe.parameters {
                    let editor = cx.new(|cx| {
                        let mut editor = Editor::single_line(window, cx);
                        editor.set_placeholder_text(
                            if parameter.required {
                                "Required"
                            } else {
                                "Optional"
                            },
                            window,
                            cx,
                        );
                        if let Some(default) = &parameter.default {
                            editor.set_text(default.clone(), window, cx);
                        }
                        editor
                    });
                    cx.observe(&editor, |_, _, cx| cx.notify()).detach();
                    self.inputs.push((parameter.name.clone(), editor));
                }
            }
        }
        self.browsing = matches!(page, RecipePage::Library);
        self.page = page;
        cx.notify();
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.loading {
            return;
        }
        self.loading = true;
        let discovered = agent::recipes::discover_recipes(
            self.project.clone(),
            self.project.read(cx).fs().clone(),
            cx,
        );
        cx.spawn(async move |this, cx| {
            let entries = discovered.await;
            this.update(cx, |this, cx| {
                this.entries = entries;
                this.loading = false;
                cx.notify();
            })
            .log_err();
        })
        .detach();
        cx.notify();
    }

    fn parameters(&self, cx: &App) -> HashMap<String, String> {
        self.inputs
            .iter()
            .map(|(name, input)| (name.clone(), input.read(cx).text(cx)))
            .collect()
    }

    fn render_navigator(&self, cx: &mut Context<Self>) -> AnyElement {
        let paths = self.project.read(cx).worktree_paths(cx);
        let folders = paths.folder_path_list();
        let store = self.store.read(cx);
        let query = self.search.read(cx).text(cx).trim().to_lowercase();
        let matches = |text: &str| query.is_empty() || text.to_lowercase().contains(&query);
        let (mut active, mut history): (Vec<_>, Vec<_>) = store
            .activity
            .values()
            .filter(|run| &PathList::deserialize(&run.project_paths) == folders)
            .filter(|run| matches(&run.title) || matches(run_status(run).0))
            .cloned()
            .partition(|run| run.status.is_active());
        active.sort_by_key(|run| (run.decision.is_none(), std::cmp::Reverse(run.started_at)));
        history.sort_by_key(|run| std::cmp::Reverse(run.started_at));
        let mut schedules = store
            .state
            .schedules
            .iter()
            .filter(|schedule| {
                !schedule.deleted && &PathList::deserialize(&schedule.project_paths) == folders
            })
            .filter(|schedule| {
                matches(&schedule.title)
                    || matches(&schedule.recipe.title)
                    || matches(&schedule.timezone)
            })
            .cloned()
            .collect::<Vec<_>>();
        schedules.sort_by_key(|schedule| (schedule.paused, schedule.next_run));
        let mut scopes = BTreeMap::<String, Vec<RecipeEntry>>::new();
        for entry in &self.entries {
            let scope = entry
                .path
                .as_ref()
                .and_then(|path| self.project.read(cx).worktree_for_id(path.worktree_id, cx))
                .map(|worktree| worktree.read(cx).root_name_str().to_owned())
                .unwrap_or_else(|| {
                    if entry.id.starts_with("builtin/") {
                        "Built-in".into()
                    } else {
                        "Project".into()
                    }
                });
            if matches(&scope)
                || entry.recipe.as_ref().map_or_else(
                    |_| matches(&entry.origin),
                    |recipe| matches(&recipe.title) || matches(&recipe.description),
                )
            {
                scopes.entry(scope).or_default().push(entry.clone());
            }
        }
        let scope_counts = scopes.values().map(Vec::len).sum::<usize>();
        let border = cx.theme().colors().border_variant;
        let mut navigator = v_flex()
            .gap_1()
            .min_w_0()
            .child(
                h_flex()
                    .justify_between()
                    .gap_1()
                    .child(Label::new("Recipes").weight(gpui::FontWeight::SEMIBOLD))
                    .child(
                        h_flex()
                            .id("recipe-navigator-toolbar")
                            .role(gpui::Role::Toolbar)
                            .aria_label("Recipe library actions")
                            .h_8()
                            .gap_1()
                            .child(
                                IconButton::new("refresh-recipes", IconName::RefreshTitle)
                                    .aria_label("Refresh recipes")
                                    .tab_index(0_isize)
                                    .tooltip(Tooltip::text("Refresh recipes"))
                                    .disabled(self.loading)
                                    .on_click(cx.listener(|view, _, _, cx| view.refresh(cx))),
                            )
                            .child(
                                IconButton::new("manage-recipe-files", IconName::FolderOpen)
                                    .aria_label("Manage recipe files")
                                    .tab_index(0_isize)
                                    .tooltip(Tooltip::text("Manage recipe files"))
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        view.panel
                                            .update(cx, |panel, cx| {
                                                panel.open_recipe_files(window, cx)
                                            })
                                            .log_err();
                                    })),
                            ),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .min_w_0()
                    .px_1p5()
                    .border_1()
                    .border_color(border)
                    .rounded_sm()
                    .child(small_icon(IconName::MagnifyingGlass, Color::Muted))
                    .child(div().flex_1().min_w_0().child(self.search.clone())),
            )
            .child(self.section("runs", "Runs", active.len() + history.len(), cx));
        if !self.collapsed.contains(&SharedString::from("runs")) {
            if active.is_empty() {
                navigator = navigator.child(div().pl_4().child(muted(if query.is_empty() {
                    "No active runs"
                } else {
                    "No matching active runs"
                })));
            }
            for run in active {
                navigator = navigator.child(self.render_run_row(run, cx));
            }
            if !history.is_empty() {
                let expanded = self.history_expanded.unwrap_or(!query.is_empty());
                navigator = navigator.child(
                    div().pl_3().child(
                        disclosure(
                            "recipe-history",
                            format!("History · {}", history.len()),
                            expanded,
                        )
                        .on_click(cx.listener(move |view, _, _, cx| {
                            view.history_expanded = Some(!expanded);
                            cx.notify();
                        })),
                    ),
                );
                if expanded {
                    for run in history {
                        navigator = navigator.child(self.render_run_row(run, cx));
                    }
                }
            }
        }
        navigator = navigator.child(Divider::horizontal()).child(self.section(
            "schedules",
            "Schedules",
            schedules.len(),
            cx,
        ));
        if !self.collapsed.contains(&SharedString::from("schedules")) {
            if schedules.is_empty() {
                navigator = navigator.child(div().pl_4().child(muted("No schedules")));
            }
            for schedule in schedules {
                navigator = navigator.child(self.render_schedule_row(schedule, cx));
            }
        }
        navigator = navigator.child(Divider::horizontal()).child(self.section(
            "library",
            "Recipe Library",
            scope_counts,
            cx,
        ));
        if !self.collapsed.contains(&SharedString::from("library")) {
            if scopes.is_empty() {
                navigator = navigator.child(div().pl_4().child(muted(if self.loading {
                    "Loading recipes…"
                } else {
                    "No matching recipes"
                })));
            }
            for (scope, entries) in scopes {
                let key = SharedString::from(format!("scope-{scope}"));
                let expanded = !self.collapsed.contains(&key);
                navigator = navigator.child(div().pl_3().child(self.section(
                    key,
                    &scope,
                    entries.len(),
                    cx,
                )));
                if expanded {
                    for entry in entries {
                        let (title, purpose) = entry.recipe.as_ref().map_or_else(
                            |error| (entry.origin.clone(), error.clone()),
                            |recipe| {
                                (
                                    recipe.title.clone(),
                                    recipe
                                        .description
                                        .lines()
                                        .next()
                                        .unwrap_or_default()
                                        .to_owned(),
                                )
                            },
                        );
                        let selected = matches!(&self.page, RecipePage::Definition(current) if current.id == entry.id);
                        navigator = navigator.child(
                            div().pl_6().child(
                                navigator_row(
                                    SharedString::from(entry.id.clone()),
                                    title,
                                    selected,
                                    IconName::ListTree,
                                    Color::Muted,
                                )
                                .aria_description(purpose.clone())
                                .child(muted(purpose).truncate().flex_1())
                                .on_click(cx.listener(
                                    move |view, _, window, cx| {
                                        view.open_page(
                                            RecipePage::Definition(entry.clone()),
                                            window,
                                            cx,
                                        );
                                    },
                                )),
                            ),
                        );
                    }
                }
            }
        }
        navigator.into_any_element()
    }

    fn section(
        &self,
        key: impl Into<SharedString>,
        label: &str,
        count: usize,
        cx: &mut Context<Self>,
    ) -> ButtonLike {
        let key = key.into();
        let expanded = !self.collapsed.contains(&key);
        disclosure(key.clone(), format!("{label} · {count}"), expanded).on_click(cx.listener(
            move |view, _, _, cx| {
                if expanded {
                    view.collapsed.insert(key.clone());
                } else {
                    view.collapsed.remove(&key);
                }
                cx.notify();
            },
        ))
    }

    fn render_run_row(&self, run: RecipeActivity, cx: &mut Context<Self>) -> AnyElement {
        let id = run.id;
        let (status, icon, color) = run_status(&run);
        let selected = matches!(&self.page, RecipePage::Run(current) if *current == id);
        div()
            .pl_4()
            .child(
                navigator_row(
                    SharedString::from(format!("run-{id}")),
                    run.title,
                    selected,
                    icon,
                    color,
                )
                .aria_description(status)
                .child(muted(status).truncate().flex_1())
                .on_click(cx.listener(move |view, _, window, cx| {
                    view.open_page(RecipePage::Run(id), window, cx)
                })),
            )
            .into_any_element()
    }

    fn render_schedule_row(&self, schedule: RecipeSchedule, cx: &mut Context<Self>) -> AnyElement {
        let id = schedule.id;
        let now = cx
            .background_executor()
            .scheduler_executor()
            .scheduler()
            .clock()
            .utc_now();
        let store = self.store.read(cx);
        let busy = store.busy;
        let latest = store
            .state
            .runs
            .iter()
            .rev()
            .find(|run| run.schedule_id == id)
            .cloned();
        let next = if schedule.paused {
            format!("Paused · {} · No upcoming run", schedule.timezone)
        } else {
            format!(
                "Enabled · {}",
                schedule_time_label(schedule.next_run, &schedule.timezone, now)
            )
        };
        let latest_label = latest.as_ref().map(|run| {
            let status = if store
                .activity
                .get(&run.id)
                .is_some_and(|run| run.decision.is_some())
            {
                "Waiting for you"
            } else {
                run.status.label()
            };
            format!("Last run · {status}")
        });
        v_flex()
            .pl_4()
            .gap_1()
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new(if schedule.title == schedule.recipe.title {
                        schedule.title
                    } else {
                        format!("{} · {}", schedule.title, schedule.recipe.title)
                    }))
                    .child(
                        h_flex()
                            .gap_1()
                            .child(small_icon(IconName::Clock, Color::Muted))
                            .child(muted(next)),
                    )
                    .when(latest.is_none(), |details| {
                        details.child(muted("No runs yet"))
                    }),
            )
            .child(
                h_flex()
                    .flex_wrap()
                    .gap_1()
                    .when_some(latest.zip(latest_label), |actions, (run, label)| {
                        actions.child(
                            control(SharedString::from(format!("last-run-{id}")), label).on_click(
                                cx.listener(move |view, _, window, cx| {
                                    view.open_page(RecipePage::Run(run.id), window, cx)
                                }),
                            ),
                        )
                    })
                    .child(
                        control(SharedString::from(format!("edit-schedule-{id}")), "Edit")
                            .on_click(cx.listener(move |view, _, window, cx| {
                                view.schedule(Some(id), None, window, cx)
                            })),
                    )
                    .child(
                        control(
                            SharedString::from(format!("toggle-schedule-{id}")),
                            if schedule.paused { "Resume" } else { "Pause" },
                        )
                        .disabled(busy)
                        .on_click(cx.listener(
                            move |view, _, window, cx| {
                                if schedule.paused {
                                    view.schedule(Some(id), None, window, cx);
                                } else {
                                    let update = view.store.update(cx, |store, cx| {
                                        store.set_enabled(id, false, false, cx)
                                    });
                                    cx.spawn(async move |view, cx| {
                                        if let Err(error) = update.await {
                                            view.update(cx, |view, cx| {
                                                view.error = Some(error.to_string());
                                                cx.notify();
                                            })
                                            .log_err();
                                        }
                                    })
                                    .detach();
                                }
                            },
                        )),
                    ),
            )
            .into_any_element()
    }

    fn schedule(
        &self,
        id: Option<Uuid>,
        entry: Option<RecipeEntry>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let recipe = entry.map(|entry| (entry, self.parameters(cx)));
        self.panel
            .update(cx, |panel, cx| {
                panel.open_recipe_schedule(id, recipe, window, cx)
            })
            .log_err();
    }

    fn render_definition(
        &self,
        entry: RecipeEntry,
        recipe: &Recipe,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let can_run = recipe.resolve_parameters(&self.parameters(cx)).is_ok();
        let launch_entry = entry.clone();
        let schedule_entry = entry.clone();
        let mut content = v_flex().gap_4()
            .child(h_flex().justify_between().items_start().gap_2().flex_wrap()
                .child(Headline::new(recipe.title.clone()).size(HeadlineSize::Medium))
                .child(h_flex().gap_1()
                    .child(control("run-recipe-now", "Run now").style(ButtonStyle::Filled)
                        .start_icon(small_icon(IconName::PlayOutlined, Color::Default)).disabled(!can_run)
                        .on_click(cx.listener(move |view, _, window, cx| {
                            let parameters = view.parameters(cx);
                            view.panel.update(cx, |panel, cx| {
                                panel.start_recipe_from_dashboard(launch_entry.clone(), parameters, window, cx)
                            }).log_err();
                        })))
                    .child(control("schedule-recipe", "Schedule")
                        .start_icon(small_icon(IconName::Clock, Color::Muted))
                        .on_click(cx.listener(move |view, _, window, cx| {
                            view.schedule(None, Some(schedule_entry.clone()), window, cx);
                        })))))
            .child(section_title("Overview"))
            .child(Label::new(recipe.description.clone()))
            .when(recipe.specification, |view| view.child(muted("Uses the project specification, independent review, revision approval, and verification.")))
            .when(!self.inputs.is_empty(), |view| view.child(section_title("Inputs")))
            .children(self.inputs.iter().zip(&recipe.parameters).map(|((name, editor), parameter)| {
                v_flex().gap_1().child(Label::new(name.clone())).child(muted(parameter.description.clone()))
                    .child(div().p_2().rounded_sm().border_1().border_color(cx.theme().colors().border).child(editor.clone()))
            }))
            .child(section_title("Steps"));
        if recipe.steps.is_empty() {
            content = content
                .child(step_row(
                    0,
                    "Run recipe".into(),
                    "Native agent turn".into(),
                    Color::Muted,
                ))
                .child(Label::new(recipe.prompt.clone()).size(LabelSize::Small));
        }
        for (index, step) in recipe.steps.iter().enumerate() {
            let mut conditions = Vec::new();
            if let Some(condition) = &step.when {
                conditions.push(format!("When {}", condition.description()));
            }
            if let Some(next) = &step.next {
                conditions.push(format!("On success → {}", step_title(next)));
            }
            match &step.on_failure {
                FailurePolicy::Stop => {}
                FailurePolicy::Continue => conditions.push("On failure → continue".into()),
                FailurePolicy::GoTo(target) => {
                    conditions.push(format!("On failure → {}", step_title(target)))
                }
            }
            if let Some(attempts) = step.repeat.filter(|attempts| *attempts > 1) {
                conditions.push(format!("Repeat up to {attempts} times"));
            }
            content = content
                .child(step_row(
                    index,
                    step_title(&step.id),
                    conditions.join(" · "),
                    Color::Muted,
                ))
                .child(
                    div()
                        .pl_5()
                        .child(Label::new(step.prompt.clone()).size(LabelSize::Small)),
                );
        }
        content.into_any_element()
    }

    fn render_run(&self, id: Uuid, cx: &mut Context<Self>) -> AnyElement {
        let store = self.store.read(cx);
        let Some(run) = store.activity.get(&id).cloned() else {
            let Some(run) = store.state.runs.iter().find(|run| run.id == id).cloned() else {
                return muted("Run history is unavailable.").into_any_element();
            };
            return v_flex()
                .gap_2()
                .child(Headline::new("Scheduled run").size(HeadlineSize::Medium))
                .child(Label::new(run.status.label()))
                .when_some(run.detail, |view, detail| view.child(muted(detail)))
                .when_some(run.session_id, |view, session| {
                    view.child(
                        control("open-legacy-recipe-result", "Open conversation").on_click(
                            cx.listener(move |dashboard, _, window, cx| {
                                dashboard
                                    .panel
                                    .update(cx, |panel, cx| {
                                        panel.open_recipe_conversation(
                                            session.clone(),
                                            None,
                                            None,
                                            window,
                                            cx,
                                        )
                                    })
                                    .log_err();
                            }),
                        ),
                    )
                })
                .into_any_element();
        };
        let (status, icon, color) = run_status(&run);
        let active = run.current_step;
        let selected = self.selected_step.or(active).or_else(|| {
            run.steps
                .iter()
                .rposition(|step| {
                    step.turn.is_some()
                        || matches!(
                            step.status,
                            RecipeStepStatus::Completed | RecipeStepStatus::Failed
                        )
                })
                .or((!run.steps.is_empty()).then_some(0))
        });
        let selected_entry = selected.and_then(|index| run.steps.get(index));
        let needs_decision = run.decision.is_some();
        let originating_recipe = self.entries.iter().filter(|entry| {
            entry
                .recipe
                .as_ref()
                .is_ok_and(|recipe| recipe.title == run.title)
        });
        let mut origins = originating_recipe.cloned();
        let origin = origins.next().filter(|_| origins.next().is_none());
        let origin_unavailable = origin.is_none();
        v_flex()
            .gap_3()
            .min_w_0()
            .child(Headline::new(run.title.clone()).size(HeadlineSize::Medium))
            .child(
                h_flex()
                    .gap_2()
                    .flex_wrap()
                    .justify_between()
                    .child(
                        h_flex()
                            .gap_1()
                            .child(small_icon(icon, color))
                            .child(Label::new(status).color(color))
                            .child(muted(if run.schedule_id.is_some() {
                                "Scheduled run"
                            } else {
                                "Manual run"
                            })),
                    )
                    .child(
                        control(
                            "open-run-conversation",
                            if needs_decision {
                                "Open decision"
                            } else if run.status.is_active() {
                                "Open conversation"
                            } else {
                                "Open result"
                            },
                        )
                        .style(ButtonStyle::Filled)
                        .start_icon(small_icon(IconName::Chat, Color::Default))
                        .on_click(cx.listener(
                            move |view, _, window, cx| {
                                view.open_conversation(id, needs_decision, None, window, cx)
                            },
                        )),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .flex_wrap()
                    .child(muted("Recipe"))
                    .when_some(origin, |row, entry| {
                        row.child(
                            control("open-originating-recipe", run.title.clone()).on_click(
                                cx.listener(move |view, _, window, cx| {
                                    view.open_page(
                                        RecipePage::Definition(entry.clone()),
                                        window,
                                        cx,
                                    )
                                }),
                            ),
                        )
                    })
                    .when(origin_unavailable, |row| {
                        row.child(Label::new(run.title.clone()))
                    }),
            )
            .child(Divider::horizontal())
            .child(
                h_flex()
                    .gap_1()
                    .justify_between()
                    .flex_wrap()
                    .child(section_title("Steps"))
                    .when(active.is_some() && active != selected, |row| {
                        row.child(control("go-to-active-step", "Go to active step").on_click(
                            cx.listener(|view, _, _, cx| {
                                view.selected_step = None;
                                cx.notify();
                            }),
                        ))
                    }),
            )
            .child(
                v_flex()
                    .gap_0p5()
                    .children(run.steps.iter().enumerate().map(|(index, step)| {
                        let (label, color, icon) = step_status(step, &run, index);
                        navigator_row(
                            SharedString::from(format!("recipe-step-{index}")),
                            format!("{}. {}", index + 1, step_title(&step.id)),
                            selected == Some(index),
                            icon,
                            color,
                        )
                        .aria_description(label.clone())
                        .child(
                            Label::new(label)
                                .size(LabelSize::Small)
                                .color(color)
                                .truncate()
                                .flex_1(),
                        )
                        .on_click(cx.listener(move |view, _, _, cx| {
                            view.selected_step = Some(index);
                            cx.notify();
                        }))
                    })),
            )
            .when_some(selected.zip(selected_entry), |panel, (index, step)| {
                let (label, color, _) = step_status(step, &run, index);
                let turn = step.turn.clone();
                let outcome = step.outcome.clone();
                panel
                    .child(Divider::horizontal())
                    .child(
                        h_flex()
                            .gap_2()
                            .justify_between()
                            .flex_wrap()
                            .child(
                                v_flex()
                                    .gap_1()
                                    .child(
                                        Label::new(format!(
                                            "Step {} · {}",
                                            index + 1,
                                            step_title(&step.id)
                                        ))
                                        .weight(gpui::FontWeight::SEMIBOLD),
                                    )
                                    .child(Label::new(label).size(LabelSize::Small).color(color)),
                            )
                            .when(turn.is_some(), |header| {
                                header.child(
                                    control("open-recipe-step", "Open step in chat")
                                        .start_icon(small_icon(IconName::Chat, Color::Muted))
                                        .on_click(cx.listener(move |view, _, window, cx| {
                                            view.open_conversation(
                                                id,
                                                false,
                                                turn.clone(),
                                                window,
                                                cx,
                                            )
                                        })),
                                )
                            }),
                    )
                    .when_some(outcome, |panel, text| {
                        panel.child(
                            div()
                                .whitespace_normal()
                                .child(Label::new(text).size(LabelSize::Small)),
                        )
                    })
            })
            .into_any_element()
    }

    pub(crate) fn open_pending_decision(&self, window: &mut Window, cx: &mut Context<Self>) {
        if let RecipePage::Run(id) = self.page
            && self
                .store
                .read(cx)
                .activity
                .get(&id)
                .is_some_and(|run| run.decision.is_some())
        {
            self.open_conversation(id, true, None, window, cx);
        }
    }

    fn open_conversation(
        &self,
        id: Uuid,
        decision: bool,
        turn: Option<acp_thread::ClientUserMessageId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(run) = self.store.read(cx).activity.get(&id).cloned() else {
            return;
        };
        let target = if decision { run.decision } else { None };
        self.panel
            .update(cx, |panel, cx| {
                panel.open_recipe_conversation(run.session_id, target, turn, window, cx);
            })
            .log_err();
    }
}

pub(super) fn disclosure(
    id: impl Into<gpui::ElementId>,
    label: String,
    expanded: bool,
) -> ButtonLike {
    ButtonLike::new(id)
        .tab_index(0_isize)
        .width(gpui::relative(1.))
        .aria_label(label.clone())
        .aria_expanded(expanded)
        .child(
            h_flex()
                .w_full()
                .gap_1()
                .min_w_0()
                .child(small_icon(
                    if expanded {
                        IconName::ChevronDown
                    } else {
                        IconName::ChevronRight
                    },
                    Color::Muted,
                ))
                .child(
                    Label::new(label)
                        .size(LabelSize::Small)
                        .weight(gpui::FontWeight::SEMIBOLD)
                        .truncate(),
                ),
        )
}

pub(super) fn navigator_row(
    id: impl Into<gpui::ElementId>,
    title: String,
    selected: bool,
    icon: IconName,
    color: Color,
) -> ButtonLike {
    ButtonLike::new(id)
        .tab_index(0_isize)
        .aria_label(title.clone())
        .width(gpui::relative(1.))
        .height(rems(1.75).into())
        .toggle_state(selected)
        .child(small_icon(icon, color))
        .child(
            Label::new(title)
                .size(LabelSize::Small)
                .weight(gpui::FontWeight::MEDIUM)
                .truncate(),
        )
}

fn run_status(run: &RecipeActivity) -> (&'static str, IconName, Color) {
    if run.decision.is_some() {
        return ("Waiting for you", IconName::Warning, Color::Warning);
    }
    let (icon, color) = match run.status {
        RecipeRunStatus::Succeeded => (IconName::Check, Color::Success),
        RecipeRunStatus::Admitted | RecipeRunStatus::Running => {
            (IconName::PlayOutlined, Color::Accent)
        }
        RecipeRunStatus::Failed | RecipeRunStatus::NeedsAttention => {
            (IconName::Warning, Color::Error)
        }
        _ => (IconName::Clock, Color::Muted),
    };
    (run.status.label(), icon, color)
}

fn step_status(
    step: &RecipeActivityStep,
    run: &RecipeActivity,
    index: usize,
) -> (String, Color, IconName) {
    let (label, color, icon) = match step.status {
        RecipeStepStatus::Pending if !run.status.is_active() => {
            ("Not run", Color::Muted, IconName::Clock)
        }
        RecipeStepStatus::Pending => ("Pending", Color::Muted, IconName::Clock),
        RecipeStepStatus::Running if run.decision.is_some() && run.current_step == Some(index) => {
            ("Waiting for you", Color::Warning, IconName::Warning)
        }
        RecipeStepStatus::Running => ("In progress", Color::Accent, IconName::PlayOutlined),
        RecipeStepStatus::Completed => ("Completed", Color::Success, IconName::Check),
        RecipeStepStatus::Failed => ("Stopped", Color::Error, IconName::Warning),
        RecipeStepStatus::Skipped => ("Skipped", Color::Muted, IconName::Clock),
    };
    let detail = if step.repetitions > 1 && step.iteration > 0 {
        format!("{label} · {} of {}", step.iteration, step.repetitions)
    } else {
        label.into()
    };
    (detail, color, icon)
}

fn step_row(index: usize, title: String, detail: String, color: Color) -> impl IntoElement {
    h_flex()
        .gap_2()
        .child(Label::new(format!("{}.", index + 1)).color(Color::Muted))
        .child(Label::new(title).weight(gpui::FontWeight::MEDIUM))
        .when(!detail.is_empty(), |row| {
            row.child(Label::new(detail).size(LabelSize::Small).color(color))
        })
}

fn step_title(id: &str) -> String {
    id.replace('-', " ")
}

fn small_icon(name: IconName, color: Color) -> Icon {
    Icon::new(name).size(IconSize::Small).color(color)
}

fn muted(text: impl Into<SharedString>) -> Label {
    Label::new(text).size(LabelSize::Small).color(Color::Muted)
}

fn section_title(text: &'static str) -> Label {
    Label::new(text).weight(gpui::FontWeight::SEMIBOLD)
}

fn control(id: impl Into<gpui::ElementId>, name: impl Into<SharedString>) -> Button {
    Button::new(id, name)
        .size(ButtonSize::Compact)
        .tab_index(0_isize)
}

pub(super) fn move_recipe_focus(event: &gpui::KeyDownEvent, window: &mut Window, cx: &mut App) {
    let modifiers = event.keystroke.modifiers;
    if event.keystroke.key != "tab"
        || modifiers.alt
        || modifiers.control
        || modifiers.platform
        || modifiers.function
    {
        return;
    }
    if modifiers.shift {
        window.focus_prev(cx);
    } else {
        window.focus_next(cx);
    }
    window.prevent_default();
    cx.stop_propagation();
}

impl Render for RecipeDashboard {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let has_detail = !matches!(self.page, RecipePage::Library);
        let show_navigator = !self.narrow || self.browsing || !has_detail;
        let show_detail = has_detail && (!self.narrow || !self.browsing);
        let border = cx.theme().colors().border_variant;
        let this = cx.weak_entity();
        let narrow = self.narrow;
        div()
            .id("recipes-editor")
            .track_focus(&self.focus)
            .tab_group()
            .role(gpui::Role::Group)
            .aria_label("Recipes")
            .on_key_down(move_recipe_focus)
            .relative()
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .child(
                canvas(
                    move |bounds, window, cx| {
                        let measured = bounds.size.width < rems(41.).to_pixels(window.rem_size());
                        if measured != narrow {
                            window.defer(cx, move |_, cx| {
                                this.update(cx, |view, cx| {
                                    if view.narrow != measured {
                                        view.narrow = measured;
                                        cx.notify();
                                    }
                                })
                                .log_err();
                            });
                        }
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .size_full(),
            )
            .child(
                v_flex()
                    .size_full()
                    .min_h_0()
                    .when(self.narrow && has_detail, |panel| {
                        let destination = if self.browsing {
                            "Show details"
                        } else {
                            "Browse recipes"
                        };
                        panel.child(
                            h_flex()
                                .id("recipe-narrow-toolbar")
                                .role(gpui::Role::Toolbar)
                                .aria_label("Recipe view")
                                .px_2()
                                .h_8()
                                .gap_1()
                                .border_b_1()
                                .border_color(border)
                                .child(
                                    IconButton::new("toggle-recipe-navigator", IconName::ListTree)
                                        .aria_label("Browse recipes")
                                        .aria_description(destination)
                                        .tooltip(Tooltip::text(destination))
                                        .toggle_state(self.browsing)
                                        .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                                        .tab_index(0_isize)
                                        .on_click(cx.listener(|view, _, _, cx| {
                                            view.browsing = !view.browsing;
                                            cx.notify();
                                        })),
                                ),
                        )
                    })
                    .when_some(self.error.clone(), |panel, error| {
                        panel.child(Label::new(error).color(Color::Error))
                    })
                    .child(
                        h_flex()
                            .flex_1()
                            .min_h_0()
                            .min_w_0()
                            .items_start()
                            .when(show_navigator, |body| {
                                body.child(
                                    div()
                                        .id("recipe-navigator")
                                        .h_full()
                                        .min_w_0()
                                        .overflow_y_scroll()
                                        .p_2()
                                        .when(show_detail, |nav| {
                                            nav.w(rems(18.))
                                                .flex_shrink_0()
                                                .border_r_1()
                                                .border_color(border)
                                        })
                                        .when(!show_detail, |nav| nav.flex_1())
                                        .child(self.render_navigator(cx)),
                                )
                            })
                            .when(show_detail, |body| {
                                let detail = match self.page.clone() {
                                    RecipePage::Library => div().into_any_element(),
                                    RecipePage::Definition(entry) => match &entry.recipe {
                                        Ok(recipe) => {
                                            self.render_definition(entry.clone(), recipe, cx)
                                        }
                                        Err(error) => Label::new(error.clone())
                                            .color(Color::Error)
                                            .into_any_element(),
                                    },
                                    RecipePage::Run(id) => self.render_run(id, cx),
                                };
                                body.child(
                                    div()
                                        .id("recipe-details")
                                        .h_full()
                                        .flex_1()
                                        .min_w_0()
                                        .overflow_y_scroll()
                                        .p_3()
                                        .child(div().max_w(rems(60.)).child(detail)),
                                )
                            }),
                    ),
            )
    }
}

impl EventEmitter<ItemEvent> for RecipeDashboard {}

impl Focusable for RecipeDashboard {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Item for RecipeDashboard {
    type Event = ItemEvent;

    fn tab_content_text(&self, _: usize, _: &App) -> SharedString {
        "Recipes".into()
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn to_item_events(event: &ItemEvent, emit: &mut dyn FnMut(ItemEvent)) {
        emit(*event);
    }
}
