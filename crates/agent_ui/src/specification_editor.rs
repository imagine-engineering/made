use std::{
    collections::BTreeSet,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use ::ui::{
    Button, ButtonSize, ButtonStyle, CommonAnimationExt as _, Disclosure, Label, LabelSize,
    ListItem, TintColor, Tooltip, prelude::*,
};
use agent::{SpecificationOverview, SpecificationPhase, SpecificationSession, SpecificationTask};
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Context as _, Result, ensure};
use editor::{
    Anchor, Editor, EditorBlockGuide, EditorTextAction, HighlightKey, MultiBufferOffset,
    SelectionEffects, ToOffset,
    display_map::{BlockPlacement, BlockProperties, BlockStyle, Crease, CreaseId, CustomBlockId},
    scroll::Autoscroll,
};
use gpui::{
    AnyElement, App, Context, Entity, EventEmitter, FocusHandle, HighlightStyle, IntoElement,
    Render, SharedString, Subscription, Task, UnderlineStyle, WeakEntity, Window,
};
use language::{Buffer, LocalFile};
use picker::{Picker, PickerDelegate};
use project::{Project, WorktreeId, trusted_worktrees::TrustedWorktrees};
use workspace::{
    ToolbarItemEvent, ToolbarItemLocation, ToolbarItemView, Workspace, item::ItemHandle,
};

use crate::{AgentPanel, ConversationView};

pub(crate) fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace
            .register_action(
                |workspace, _: &crate::FocusSpecificationControls, window, cx| {
                    if let Some(toolbar) = specification_toolbar(workspace, cx)
                        && toolbar.read(cx).overview.is_some()
                    {
                        let focus = toolbar.read(cx).focus.clone();
                        focus.focus(window, cx);
                        window.focus_next(cx);
                    }
                },
            )
            .register_action(
                |workspace, _: &crate::OpenSpecificationTaskActions, window, cx| {
                    if let Some(toolbar) = specification_toolbar(workspace, cx)
                        && let Some(picker) = toolbar.read(cx).task_actions_at_cursor(&toolbar, cx)
                    {
                        workspace.toggle_modal(window, cx, |window, cx| {
                            Picker::uniform_list(picker, window, cx)
                        });
                    }
                },
            );
    })
    .detach();
}

fn specification_toolbar(workspace: &Workspace, cx: &App) -> Option<Entity<SpecificationEditor>> {
    workspace
        .active_pane()
        .read(cx)
        .toolbar()
        .read(cx)
        .item_of_type()
}

pub struct SpecificationEditor {
    workspace: WeakEntity<Workspace>,
    focus: FocusHandle,
    source: Option<Source>,
    overview: Option<SpecificationOverview>,
    design: bool,
    refresh: Option<Task<()>>,
    busy: bool,
    running_task: Option<(PathBuf, String)>,
    feedback: Option<SharedString>,
}

struct Source {
    project: Entity<Project>,
    worktree: WorktreeId,
    root: PathBuf,
    path: PathBuf,
    editor: Entity<Editor>,
    buffer: Entity<Buffer>,
    blocks: Vec<CustomBlockId>,
    creases: Vec<CreaseId>,
    _subscriptions: Vec<Subscription>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Requirements,
    Design,
    Tasks,
}

#[derive(Clone)]
enum TaskAction {
    Run(String),
    Session(SpecificationSession),
    Evidence { task: String, observed_at: String },
    Criterion(String),
    Dependency(String),
    ToggleCheckbox(String, bool),
}

impl SpecificationEditor {
    pub fn new(workspace: WeakEntity<Workspace>, cx: &mut Context<Self>) -> Self {
        Self {
            workspace,
            focus: cx.focus_handle(),
            source: None,
            overview: None,
            design: false,
            refresh: None,
            busy: false,
            running_task: None,
            feedback: None,
        }
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(source) = &self.source else {
            return;
        };
        let (root, path) = (source.root.clone(), source.path.clone());
        let load = cx.background_spawn(async move { agent::specification_for_editor(root, &path) });
        self.refresh = Some(cx.spawn(async move |this, cx| {
            let result = load.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(overview) => this.overview = overview,
                    Err(error) => {
                        this.overview = None;
                        this.feedback = Some(format!("{error:#}").into());
                    }
                }
                let location = this.location();
                cx.emit(ToolbarItemEvent::ChangeLocation(location));
                this.update_blocks(false, cx);
                cx.notify();
            })
            .ok();
        }));
    }

    fn location(&self) -> ToolbarItemLocation {
        if self.overview.is_some() {
            ToolbarItemLocation::Secondary
        } else {
            ToolbarItemLocation::Hidden
        }
    }

    fn update_blocks(&mut self, replace: bool, cx: &mut Context<Self>) {
        let toolbar = cx.weak_entity();
        let Some(source) = &mut self.source else {
            return;
        };
        let is_tasks = self
            .overview
            .as_ref()
            .and_then(|overview| overview.tasks_path.as_ref())
            == Some(&source.path);
        if !replace && is_tasks && !source.blocks.is_empty() {
            source.editor.update(cx, |_, cx| cx.notify());
            return;
        }
        let text = source.buffer.read(cx).text();
        source.editor.update(cx, |editor, cx| {
            editor.remove_blocks(source.blocks.drain(..).collect(), None, cx);
            editor.remove_creases(source.creases.drain(..), cx);
            if !is_tasks {
                editor.set_text_actions(Vec::new(), cx);
                editor.set_block_guides(Vec::new(), cx);
                editor.clear_highlights(HighlightKey::SpecificationTaskReference, cx);
                editor.clear_highlights(HighlightKey::SpecificationTaskTitle, cx);
                return;
            }
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let rows = agent::specification_task_rows(&text);
            let known_tasks = Arc::new(rows.iter().map(|row| row.id.clone()).collect());
            let mut blocks = Vec::with_capacity(rows.len());
            let mut guides = Vec::with_capacity(rows.len());
            let mut creases = Vec::new();
            let mut text_actions = Vec::new();
            let mut references = Vec::new();
            let mut titles = Vec::new();

            for row in rows {
                let checkbox = row.offset + row.indent + 2;
                let anchor = |offset| snapshot.anchor_before(MultiBufferOffset(offset));
                let title = checkbox + 4;
                if title < row.title_end {
                    titles.push(anchor(title)..anchor(row.title_end));
                }
                let guide = anchor(row.offset + row.indent)..anchor(row.end);
                guides.push(guide);

                if row.title_end < row.end {
                    creases.push(Crease::Inline {
                        range: anchor(row.title_end)..anchor(row.end),
                        placeholder: editor.default_fold_placeholder(cx),
                        render_toggle: Some(Arc::new(|row, folded, toggle, _, _| {
                            Disclosure::new(("specification-task-fold", row.0), !folded)
                                .on_click(move |_, window, cx| toggle(!folded, window, cx))
                                .into_any_element()
                        })),
                        render_trailer: None,
                        metadata: None,
                    });
                }

                text_actions.push(EditorTextAction::new(
                    anchor(checkbox)..anchor(checkbox + 3),
                    false,
                    toggle_task_checkbox,
                ));
                for (range, id) in row.references {
                    let range = anchor(range.start)..anchor(range.end);
                    references.push(range.clone());
                    let toolbar = toolbar.clone();
                    text_actions.push(EditorTextAction::new(
                        range,
                        true,
                        move |_, _, window, cx| {
                            let toolbar = toolbar.clone();
                            let action = TaskAction::Criterion(id.clone());
                            window.defer(cx, move |window, cx| {
                                dispatch_task_action(&toolbar, action, window, cx);
                            });
                        },
                    ));
                }

                let toolbar = toolbar.clone();
                let known_tasks = Arc::clone(&known_tasks);
                blocks.push(BlockProperties {
                    placement: BlockPlacement::Above(anchor(row.offset)),
                    height: Some(1),
                    style: BlockStyle::Spacer,
                    priority: 0,
                    render: Arc::new(move |block| {
                        let Some(view) = toolbar.upgrade() else {
                            return div().into_any_element();
                        };
                        view.read(block.app).task_block(
                            &row.id,
                            row.indent,
                            row.complete,
                            &known_tasks,
                            &toolbar,
                            block,
                        )
                    }),
                });
            }
            source.blocks = editor.insert_blocks(blocks, None, cx);
            source.creases = editor.insert_creases(creases, cx);
            editor.set_block_guides(
                source
                    .blocks
                    .iter()
                    .copied()
                    .zip(guides)
                    .map(|(block, range)| EditorBlockGuide { block, range })
                    .collect(),
                cx,
            );
            editor.set_text_actions(text_actions, cx);
            editor.highlight_text(
                HighlightKey::SpecificationTaskTitle,
                titles,
                HighlightStyle {
                    font_weight: Some(gpui::FontWeight::SEMIBOLD),
                    ..Default::default()
                },
                cx,
            );
            editor.highlight_text(
                HighlightKey::SpecificationTaskReference,
                references,
                HighlightStyle {
                    color: Some(cx.theme().colors().text_accent),
                    underline: Some(UnderlineStyle {
                        thickness: px(1.),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                cx,
            );
        });
    }

    fn task_block(
        &self,
        id: &str,
        indent: usize,
        checked: bool,
        known_tasks: &BTreeSet<String>,
        toolbar: &WeakEntity<Self>,
        block: &editor::display_map::BlockContext,
    ) -> AnyElement {
        let Some(specification) = &self.overview else {
            return div().into_any_element();
        };
        let task = specification.tasks.iter().find(|task| task.id == id);
        let running = self.task_running(specification, id, block.app);
        let mut row = h_flex()
            .id(SharedString::from(format!("spec-task-{id}")))
            .role(gpui::Role::Toolbar)
            .aria_label(format!("Task {id} controls"))
            .tab_group()
            .on_key_down(move_specification_focus)
            .gap_1()
            .h_full()
            .pl(block.em_width * (indent + 2) as f32 + block.indent_guide_padding)
            .text_ui(block.app);
        if running {
            row = row.child(
                h_flex()
                    .gap_1()
                    .child(
                        task_icon(IconName::LoadCircle, Color::Accent).with_keyed_rotate_animation(
                            SharedString::from(format!("specification-task-progress-{id}")),
                            2,
                        ),
                    )
                    .child(
                        Label::new("In progress")
                            .size(LabelSize::Small)
                            .color(Color::Accent),
                    ),
            );
        } else if checked {
            let verified =
                !self.has_unsaved_files(block.app) && task.is_some_and(|task| task.verified);
            row = row.child(if verified {
                h_flex()
                    .gap_1()
                    .child(task_icon(IconName::Check, Color::Success))
                    .child(
                        Label::new("Verified")
                            .size(LabelSize::Small)
                            .color(Color::Success),
                    )
            } else {
                h_flex()
                    .gap_1()
                    .child(task_icon(IconName::Info, Color::Muted))
                    .child(
                        Label::new("Completed · Unverified")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    )
            });
        } else if let Some(task) = task.filter(|task| !task.blocked_by.is_empty()) {
            let reason = h_flex()
                .gap_1()
                .child(task_icon(IconName::Lock, Color::Warning))
                .child(
                    Label::new("Blocked by")
                        .size(LabelSize::Small)
                        .color(Color::Warning),
                )
                .children(task.blocked_by.iter().map(|dependency| {
                    if dependency != id && known_tasks.contains(dependency) {
                        task_button(TaskAction::Dependency(dependency.clone()), toolbar)
                            .color(Color::Warning)
                            .aria_label(format!("Go to prerequisite {dependency}"))
                            .into_any_element()
                    } else {
                        Label::new(dependency.clone())
                            .size(LabelSize::Small)
                            .color(Color::Warning)
                            .into_any_element()
                    }
                }));
            row = row.child(reason);
        } else if !task.is_some_and(|task| task.complete) {
            row = row.child(
                task_button(TaskAction::Run(id.to_owned()), toolbar)
                    .disabled(!self.can_run_task(task, block.app)),
            );
        }
        let links = task_links(id, task);
        row.when(links.clone().next().is_some(), |row| {
            row.child(
                div()
                    .mx_0p5()
                    .w_px()
                    .h_3()
                    .bg(block.app.theme().colors().border_variant),
            )
            .children(links.map(|action| task_button(action, toolbar)))
        })
        .into_any_element()
    }

    fn task_running(&self, specification: &SpecificationOverview, id: &str, cx: &App) -> bool {
        self.running_task
            .as_ref()
            .is_some_and(|(path, task)| path == &specification.path && task == id)
            || self
                .workspace
                .upgrade()
                .and_then(|workspace| workspace.read(cx).panel::<AgentPanel>(cx))
                .and_then(|panel| panel.read(cx).active_native_agent_thread(cx))
                .is_some_and(|thread| {
                    let thread = thread.read(cx);
                    thread
                        .specification_overview()
                        .is_some_and(|overview| overview.path == specification.path)
                        && thread.active_specification_task() == Some(id)
                })
    }

    fn can_run_task(&self, task: Option<&SpecificationTask>, cx: &App) -> bool {
        self.overview
            .as_ref()
            .is_some_and(|specification| specification.implementation_ready)
            && !self.busy
            && task.is_some_and(|task| !task.complete && task.blocked_by.is_empty())
            && self.source.as_ref().is_some_and(|source| {
                !TrustedWorktrees::has_restricted_worktrees(
                    &source.project.read(cx).worktree_store(),
                    cx,
                )
            })
            && !self.has_unsaved_files(cx)
    }

    fn task_actions_at_cursor(
        &self,
        toolbar: &Entity<Self>,
        cx: &App,
    ) -> Option<TaskActionsPicker> {
        let source = self.source.as_ref()?;
        let overview = self.overview.as_ref()?;
        if overview.tasks_path.as_ref() != Some(&source.path) {
            return None;
        }
        let editor = source.editor.read(cx);
        let snapshot = editor.buffer().read(cx).snapshot(cx);
        let cursor = editor
            .selections
            .newest_anchor()
            .head()
            .to_offset(&snapshot)
            .0;
        let rows = agent::specification_task_rows(&source.buffer.read(cx).text());
        let index = rows
            .partition_point(|row| row.offset <= cursor)
            .saturating_sub(1);
        let row = rows.get(index)?;
        let task = overview.tasks.iter().find(|task| task.id == row.id);
        let start = (!row.complete
            && self.can_run_task(task, cx)
            && !self.task_running(overview, &row.id, cx))
        .then(|| TaskAction::Run(row.id.clone()));
        let actions = start
            .into_iter()
            .chain(task_links(&row.id, task))
            .chain(
                row.criteria
                    .iter()
                    .chain(&row.properties)
                    .cloned()
                    .map(TaskAction::Criterion),
            )
            .chain(
                row.dependencies
                    .iter()
                    .filter(|dependency| {
                        *dependency != &row.id && rows.iter().any(|task| task.id == **dependency)
                    })
                    .cloned()
                    .map(TaskAction::Dependency),
            )
            .chain(
                (!editor.read_only(cx))
                    .then(|| TaskAction::ToggleCheckbox(row.id.clone(), row.complete)),
            )
            .collect();
        Some(TaskActionsPicker {
            toolbar: toolbar.downgrade(),
            task_id: row.id.clone(),
            actions,
            matches: Vec::new(),
            selected: 0,
        })
    }

    fn has_unsaved_files(&self, cx: &App) -> bool {
        self.source
            .as_ref()
            .zip(self.overview.as_ref())
            .is_some_and(|(source, overview)| {
                feature_buffers(&source.project, &source.root, &overview.path, cx)
                    .iter()
                    .any(|(_, buffer)| buffer.read(cx).is_dirty())
            })
    }

    fn open_view(&mut self, view: View, window: &mut Window, cx: &mut Context<Self>) {
        let Some(overview) = &self.overview else {
            return;
        };
        self.design = view == View::Design;
        match view {
            View::Requirements => self.open_file(
                overview.path.clone(),
                Some(Target::Requirements),
                window,
                cx,
            ),
            View::Design => self.open_file(overview.path.clone(), Some(Target::Design), window, cx),
            View::Tasks => {
                if let Some(path) = &overview.tasks_path {
                    self.open_file(path.clone(), None, window, cx);
                }
            }
        }
        cx.notify();
    }

    fn open_file(
        &mut self,
        path: PathBuf,
        target: Option<Target>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let workspace = self.workspace.clone();
        let loading = workspace.update(cx, |workspace, cx| {
            workspace.open_abs_path(path, Default::default(), window, cx)
        });
        cx.spawn_in(window, async move |this, cx| {
            let result: Result<()> = async {
                let item = loading?.await?;
                if let Some(target) = target {
                    let editor = item
                        .downcast::<Editor>()
                        .context("Open the Markdown source in the editor")?;
                    cx.update(|window, cx| {
                        editor.update(cx, |editor, cx| {
                            let Some(buffer) = editor.buffer().read(cx).as_singleton() else {
                                return;
                            };
                            let text = buffer.read(cx).text();
                            if let Some((offset, length)) = target.range(&text) {
                                let range =
                                    MultiBufferOffset(offset)..MultiBufferOffset(offset + length);
                                editor.unfold_ranges(&[range.clone()], false, false, cx);
                                editor.change_selections(
                                    SelectionEffects::scroll(Autoscroll::center()),
                                    window,
                                    cx,
                                    |selections| {
                                        selections.select_ranges([range]);
                                    },
                                );
                            }
                        });
                    })?;
                }
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| this.finish_action(result, cx))
                .ok();
        })
        .detach();
    }

    fn finish_action(&mut self, result: Result<()>, cx: &mut Context<Self>) {
        if let Err(error) = result {
            self.feedback = Some(format!("{error:#}").into());
        }
        cx.notify();
    }

    fn handle_task(&mut self, action: TaskAction, window: &mut Window, cx: &mut Context<Self>) {
        let Some(overview) = &self.overview else {
            return;
        };
        match action {
            TaskAction::Run(id) => self.run(Some(id), window, cx),
            TaskAction::Session(session) => self.open_conversation(Some(session), true, window, cx),
            TaskAction::Evidence { task, observed_at } => {
                if let Some(path) = &overview.verification_path {
                    self.open_file(
                        path.clone(),
                        Some(Target::Evidence { task, observed_at }),
                        window,
                        cx,
                    );
                }
            }
            TaskAction::Criterion(id) => self.open_file(
                overview.path.clone(),
                Some(Target::Criterion(id)),
                window,
                cx,
            ),
            TaskAction::Dependency(id) => {
                if let Some(path) = &overview.tasks_path {
                    self.open_file(path.clone(), Some(Target::TaskRow(id)), window, cx);
                }
            }
            TaskAction::ToggleCheckbox(id, _) => {
                if let Some(source) = &self.source {
                    let text = source.buffer.read(cx).text();
                    if let Some(row) = agent::specification_task_rows(&text)
                        .into_iter()
                        .find(|row| row.id == id)
                    {
                        let start = row.offset + row.indent + 2;
                        source.editor.update(cx, |editor, cx| {
                            let snapshot = editor.buffer().read(cx).snapshot(cx);
                            let range = snapshot.anchor_before(MultiBufferOffset(start))
                                ..snapshot.anchor_before(MultiBufferOffset(start + 3));
                            toggle_task_checkbox(editor, &range, window, cx);
                        });
                    }
                }
            }
        }
    }

    fn select_conversation(
        &self,
        specification: &SpecificationOverview,
        session: Option<&SpecificationSession>,
        require_session: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<Entity<ConversationView>> {
        self.workspace.update(cx, |workspace, cx| {
            let panel = workspace
                .panel::<AgentPanel>(cx)
                .context("The built-in agent is unavailable")?;
            AgentPanel::reveal_conversation(workspace, true, window, cx);
            panel.update(cx, |panel, cx| {
                panel.editor_specification_conversation(
                    specification,
                    session,
                    require_session,
                    window,
                    cx,
                )
            })
        })?
    }

    fn open_conversation(
        &mut self,
        session: Option<SpecificationSession>,
        require_session: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(overview) = self.overview.clone() else {
            return;
        };
        let conversation =
            self.select_conversation(&overview, session.as_ref(), require_session, window, cx);
        let Ok(conversation) = conversation else {
            self.finish_action(conversation.map(|_| ()), cx);
            return;
        };
        let loaded = conversation.update(cx, |conversation, cx| {
            conversation.wait_for_transcript(window, cx)
        });
        cx.spawn(async move |this, cx| {
            let result: Result<()> = async {
                let view = loaded.await?;
                let native = conversation
                    .read_with(cx, |conversation, cx| conversation.as_native_thread(cx))
                    .context("Open the built-in agent for this specification")?;
                native
                    .update(cx, |native, cx| {
                        native.resume_specification(Some(overview.path), cx)
                    })
                    .await?;
                if let Some(session) = session {
                    let id = acp::ToolCallId::new(session.tool_call_id);
                    let found = view.update(cx, |view, cx| view.reveal_tool_call(&id, cx));
                    ensure!(
                        found || !require_session,
                        "The original task turn is unavailable in this conversation"
                    );
                }
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| this.finish_action(result, cx))
                .ok();
        })
        .detach();
    }

    fn run(&mut self, task: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let Some((source, overview)) = self.source.as_ref().zip(self.overview.clone()) else {
            return;
        };
        if self.busy || !overview.implementation_ready || self.has_unsaved_files(cx) {
            return;
        }
        let root = source.root.clone();
        let project = source.project.clone();
        let task_is_single = task.is_some();
        let conversation = self.select_conversation(
            &overview,
            overview.review_session.as_ref(),
            false,
            window,
            cx,
        );
        let Ok(conversation) = conversation else {
            self.finish_action(conversation.map(|_| ()), cx);
            return;
        };
        let loaded = conversation.update(cx, |conversation, cx| {
            conversation.wait_for_transcript(window, cx)
        });
        self.feedback = None;
        self.busy = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result: Result<()> = async {
                loaded.await?;
                let (native, author) = conversation
                    .read_with(cx, |conversation, cx| {
                        Some((
                            conversation.as_native_thread(cx)?,
                            conversation.root_thread(cx)?,
                        ))
                    })
                    .context("Task execution requires the built-in agent")?;
                let mut requested = task;
                loop {
                    let current = load_specification(&root, &overview.path, cx).await?;
                    ensure!(
                        current.revision == overview.revision,
                        "Paused. Revision {} needs review.",
                        short_revision(&current.revision)
                    );
                    ensure!(
                        current.implementation_ready,
                        "Paused. Review and register the current plan and task files."
                    );
                    let requested_task = if let Some(id) = requested.take() {
                        Some(
                            current
                                .tasks
                                .iter()
                                .find(|task| task.id == id)
                                .context("The selected task is unavailable")?,
                        )
                    } else {
                        current
                            .tasks
                            .iter()
                            .find(|task| !task.complete && task.blocked_by.is_empty())
                    };
                    let Some(next) = requested_task else {
                        ensure!(
                            current.tasks.iter().all(|task| task.complete),
                            "Paused. Remaining tasks have unresolved dependencies."
                        );
                        break;
                    };
                    ensure!(
                        !next.complete && next.blocked_by.is_empty(),
                        "Task {} is no longer runnable",
                        next.id
                    );
                    let id = next.id.clone();
                    let dirty = project.read_with(cx, |_, cx| {
                        feature_buffers(&project, &root, &overview.path, cx)
                            .iter()
                            .any(|(_, buffer)| buffer.read(cx).is_dirty())
                    });
                    ensure!(
                        !dirty,
                        "Paused. Save or resolve changes to the feature files."
                    );
                    ensure!(
                        author.read_with(cx, |author, _| author.status()
                            == acp_thread::ThreadStatus::Idle),
                        "Wait for the current agent turn to finish"
                    );
                    native
                        .update(cx, |native, cx| {
                            native.resume_specification(Some(overview.path.clone()), cx)
                        })
                        .await?;
                    native.update(cx, |native, _| {
                        native.request_editor_specification_task(
                            id.clone(),
                            overview.revision.clone(),
                        )
                    })?;
                    this.update(cx, |this, cx| {
                        this.running_task = Some((overview.path.clone(), id.clone()));
                        this.update_blocks(false, cx);
                        cx.notify();
                    })
                    .ok();
                    let response = author
                        .update(cx, |author, cx| {
                            author.send(
                                vec![
                                    format!("Start {id} and record its acceptance checks.").into(),
                                ],
                                cx,
                            )
                        })
                        .await;
                    native.update(cx, |native, cx| {
                        native.finish_editor_specification_task();
                        cx.notify();
                    });
                    response?;
                    let after = load_specification(&root, &overview.path, cx).await?;
                    ensure!(
                        after.revision == overview.revision,
                        "Paused. Revision {} needs review.",
                        short_revision(&after.revision)
                    );
                    ensure!(
                        after
                            .tasks
                            .iter()
                            .any(|task| task.id == id && task.complete),
                        "Paused before completing {id}. See the conversation for details."
                    );
                    this.update(cx, |this, cx| {
                        if this
                            .overview
                            .as_ref()
                            .is_some_and(|view| view.path == overview.path)
                        {
                            this.overview = Some(after);
                            this.update_blocks(false, cx);
                        }
                    })
                    .ok();
                    if task_is_single {
                        break;
                    }
                }
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| {
                this.busy = false;
                this.running_task = None;
                this.finish_action(result, cx);
                this.refresh(cx);
            })
            .ok();
        })
        .detach();
    }

    fn sync(&mut self, cx: &mut Context<Self>) {
        let Some((source, overview)) = self.source.as_ref().zip(self.overview.as_ref()) else {
            return;
        };
        let fs = source.project.read(cx).fs().clone();
        let buffers = feature_buffers(&source.project, &source.root, &overview.path, cx);
        self.feedback = None;
        cx.spawn(async move |this, cx| {
            let mut messages = Vec::new();
            for (path, buffer) in buffers {
                let result: Result<()> = async {
                    let metadata = fs.metadata(&path).await?;
                    let reload = buffer.update(cx, |buffer, cx| {
                        if buffer.is_dirty() {
                            let changed = buffer.has_conflict()
                                || metadata.as_ref().map(|file| file.mtime) != buffer.saved_mtime();
                            if changed {
                                buffer.set_conflict();
                                cx.notify();
                                messages.push(format!(
                                    "Conflict in {}. Unsaved edits kept.",
                                    file_name(&path)
                                ));
                            } else {
                                messages
                                    .push(format!("Unsaved edits kept in {}.", file_name(&path)));
                            }
                            None
                        } else if metadata.is_some() {
                            Some(buffer.reload_if_clean(cx))
                        } else {
                            messages.push(format!("{} was removed on disk.", file_name(&path)));
                            None
                        }
                    });
                    if let Some(reload) = reload {
                        reload
                            .await
                            .context("Could not reload the file from disk")?;
                        if buffer.read_with(cx, |buffer, _| buffer.is_dirty()) {
                            messages.push(format!("Unsaved edits kept in {}.", file_name(&path)));
                        }
                    }
                    Ok(())
                }
                .await;
                if let Err(error) = result {
                    messages.push(format!("Could not sync {}. {error:#}", file_name(&path)));
                }
            }
            this.update(cx, |this, cx| {
                this.feedback = (!messages.is_empty()).then(|| messages.join(" ").into());
                this.refresh(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }
}

impl EventEmitter<ToolbarItemEvent> for SpecificationEditor {}

impl ToolbarItemView for SpecificationEditor {
    fn set_active_pane_item(
        &mut self,
        item: Option<&dyn ItemHandle>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> ToolbarItemLocation {
        let editor = item.and_then(|item| item.downcast::<Editor>());
        if self
            .source
            .as_ref()
            .is_some_and(|source| Some(&source.editor) == editor.as_ref())
        {
            return self.location();
        }
        if let Some(source) = self.source.take() {
            cx.defer(move |cx| {
                source.editor.update(cx, |editor, cx| {
                    editor.remove_blocks(source.blocks.into_iter().collect(), None, cx);
                    editor.remove_creases(source.creases, cx);
                    editor.set_block_guides(Vec::new(), cx);
                    editor.set_text_actions(Vec::new(), cx);
                    editor.clear_highlights(HighlightKey::SpecificationTaskReference, cx);
                    editor.clear_highlights(HighlightKey::SpecificationTaskTitle, cx);
                });
            });
        }
        self.refresh.take();
        self.overview = None;
        self.feedback = None;
        self.source = (|| {
            let editor = editor?;
            let buffer = editor.read(cx).buffer().read(cx).as_singleton()?;
            let file = project::File::from_dyn(buffer.read(cx).file())?;
            let path = file.abs_path(cx);
            if !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("spec.md" | "plan.md" | "tasks.md" | "verification.md")
            ) {
                return None;
            }
            let workspace = self.workspace.upgrade()?;
            let project = workspace.read(cx).project().clone();
            if !project.read(cx).is_local() {
                return None;
            }
            let (worktree, root) = {
                let mut roots = project.read(cx).visible_worktrees(cx);
                let root = roots.next()?;
                if roots.next().is_some() {
                    return None;
                }
                (root.read(cx).id(), root.read(cx).abs_path().to_path_buf())
            };
            let events = cx.subscribe(&project, |this, _, event, cx| {
                let Some(source) = &this.source else { return };
                if let project::Event::WorktreeUpdatedEntries(worktree, entries) = event
                    && source.worktree == *worktree
                    && entries.iter().any(|(path, _, _)| {
                        is_feature_file(
                            &source.root,
                            &source.path,
                            &source.root.join(path.as_unix_str()),
                        )
                    })
                {
                    this.refresh(cx);
                }
            });
            let edited = cx.subscribe(&buffer, |_, _, event, cx| {
                if matches!(
                    event,
                    language::BufferEvent::Edited { .. } | language::BufferEvent::Reloaded
                ) {
                    let weak = cx.weak_entity();
                    cx.defer(move |cx| {
                        weak.update(cx, |this, cx| this.update_blocks(true, cx))
                            .ok();
                    });
                }
                cx.notify();
            });
            Some(Source {
                editor,
                buffer,
                project,
                worktree,
                root,
                path,
                blocks: Vec::new(),
                creases: Vec::new(),
                _subscriptions: vec![events, edited],
            })
        })();
        self.refresh(cx);
        self.location()
    }
}

impl Render for SpecificationEditor {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some((source, specification)) = self.source.as_ref().zip(self.overview.as_ref()) else {
            return div().into_any_element();
        };
        let active = if specification.tasks_path.as_ref() == Some(&source.path)
            || specification.verification_path.as_ref() == Some(&source.path)
        {
            View::Tasks
        } else if self.design || specification.plan_path.as_ref() == Some(&source.path) {
            View::Design
        } else {
            View::Requirements
        };
        let dirty = self.has_unsaved_files(cx);
        let restricted = TrustedWorktrees::has_restricted_worktrees(
            &source.project.read(cx).worktree_store(),
            cx,
        );
        let pending = matches!(
            specification.phase,
            SpecificationPhase::Draft
                | SpecificationPhase::NeedsReview
                | SpecificationPhase::ResolveFindings
                | SpecificationPhase::AwaitingApproval
        );
        let revision = short_revision(&specification.revision);
        let status = match specification.phase {
            SpecificationPhase::NeedsReview => Some(format!("Revision {revision} needs review")),
            SpecificationPhase::AwaitingApproval => {
                Some(format!("Revision {revision} awaiting approval"))
            }
            SpecificationPhase::ResolveFindings => {
                Some(format!("Revision {revision} has blocking findings"))
            }
            SpecificationPhase::Draft => Some(format!("Revision {revision} is a draft")),
            _ if self.busy => self
                .running_task
                .as_ref()
                .map(|(_, task)| format!("{task} in progress")),
            _ => None,
        };
        let findings = (pending && specification.findings > 0).then(|| {
            let label = if specification.findings == 1 {
                "finding"
            } else {
                "findings"
            };
            format!("{} unresolved {label}", specification.findings)
        });
        let status = status
            .into_iter()
            .chain(findings)
            .chain(dirty.then(|| "Unsaved feature files".to_owned()))
            .collect::<Vec<_>>()
            .join(" · ");
        let can_run = !restricted
            && !dirty
            && !self.busy
            && specification.implementation_ready
            && specification
                .tasks
                .iter()
                .any(|task| !task.complete && task.blocked_by.is_empty());
        let session = specification.review_session.clone();
        let stage_route = h_flex()
            .id("specification-stages")
            .role(gpui::Role::Group)
            .aria_label("Specification stage")
            .h_8()
            .gap_1()
            .flex_shrink_0()
            .children(
                [
                    (
                        View::Requirements,
                        "Requirements",
                        IconName::FileTextOutlined,
                    ),
                    (View::Design, "Design", IconName::Pencil),
                    (View::Tasks, "Tasks", IconName::ListTodo),
                ]
                .into_iter()
                .map(|(view, label, icon)| {
                    h_flex()
                        .gap_1()
                        .when(view != View::Requirements, |group| {
                            group.child(task_icon(IconName::ChevronRight, Color::Muted))
                        })
                        .child(
                            IconButton::new(label, icon)
                                .aria_label(label)
                                .tooltip(Tooltip::text(label))
                                .tab_index(0isize)
                                .toggle_state(active == view)
                                .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                                .disabled(view == View::Tasks && specification.tasks_path.is_none())
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.open_view(view, window, cx)
                                })),
                        )
                }),
            );
        let stage = h_flex()
            .w_full()
            .gap_1p5()
            .flex_wrap()
            .child(
                Label::new(specification.title.clone())
                    .size(LabelSize::Small)
                    .weight(gpui::FontWeight::SEMIBOLD)
                    .truncate(),
            )
            .child(stage_route)
            .when(active == View::Design, |row| {
                row.when_some(specification.plan_path.clone(), |row, path| {
                    row.child(
                        IconButton::new("specification-plan", IconName::FileDoc)
                            .aria_label("Open plan")
                            .tooltip(Tooltip::text("Open plan"))
                            .tab_index(0isize)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.open_file(path.clone(), None, window, cx)
                            })),
                    )
                })
            });
        let revisions = h_flex()
            .gap_2()
            .flex_wrap()
            .child(revision_label(
                "Reviewed",
                specification.reviewed.as_deref(),
                specification.reviewed.as_deref() == Some(&specification.revision),
            ))
            .child(revision_label(
                "Approved",
                specification.approved.as_deref(),
                specification.approved.as_deref() == Some(&specification.revision),
            ));
        let actions = h_flex()
            .w_full()
            .gap_1()
            .flex_wrap()
            .when(!status.is_empty(), |actions| {
                actions.child(
                    Label::new(status)
                        .size(LabelSize::Small)
                        .color(Color::Warning),
                )
            })
            .when(pending, |actions| {
                actions.child(
                    Button::new("specification-open-review", "Open review")
                        .size(ButtonSize::Compact)
                        .style(ButtonStyle::Filled)
                        .start_icon(task_icon(IconName::Chat, Color::Default))
                        .tab_index(0isize)
                        .disabled(restricted)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.open_conversation(session.clone(), false, window, cx)
                        })),
                )
            })
            .child(div().flex_grow_1())
            .child(
                IconButton::new("specification-sync", IconName::RefreshTitle)
                    .aria_label("Sync files")
                    .tooltip(Tooltip::text("Sync files"))
                    .tab_index(0isize)
                    .on_click(cx.listener(|this, _, _, cx| this.sync(cx))),
            )
            .child(
                Button::new("specification-run-remaining", "Run remaining tasks")
                    .size(ButtonSize::Compact)
                    .start_icon(task_icon(IconName::PlayOutlined, Color::Default))
                    .tab_index(0isize)
                    .style(if pending {
                        ButtonStyle::Subtle
                    } else {
                        ButtonStyle::Filled
                    })
                    .disabled(!can_run)
                    .on_click(cx.listener(|this, _, window, cx| this.run(None, window, cx))),
            );
        v_flex()
            .id("specification-editor-bar")
            .track_focus(&self.focus)
            .role(gpui::Role::Toolbar)
            .aria_label("Specification")
            .tab_group()
            .on_key_down(move_specification_focus)
            .w_full()
            .gap_1()
            .px_2()
            .py_1()
            .child(stage)
            .child(revisions)
            .child(actions)
            .when_some(self.feedback.clone(), |bar, feedback| {
                bar.child(
                    Label::new(feedback)
                        .size(LabelSize::Small)
                        .color(Color::Warning),
                )
            })
            .into_any_element()
    }
}

fn revision_label(label: &str, revision: Option<&str>, current: bool) -> Label {
    Label::new(format!(
        "{label} · {}",
        revision.map(short_revision).unwrap_or("Pending")
    ))
    .size(LabelSize::Small)
    .color(if current {
        Color::Success
    } else {
        Color::Muted
    })
}

fn move_specification_focus(event: &gpui::KeyDownEvent, window: &mut Window, cx: &mut App) {
    let modifiers = event.keystroke.modifiers;
    if modifiers.control || modifiers.alt || modifiers.platform || modifiers.function {
        return;
    }
    match (event.keystroke.key.as_str(), modifiers.shift) {
        ("tab" | "right", false) => window.focus_next(cx),
        ("tab", true) | ("left", false) => window.focus_prev(cx),
        _ => return,
    }
    window.prevent_default();
    cx.stop_propagation();
}

impl TaskAction {
    fn label(&self) -> SharedString {
        match self {
            Self::Run(_) => "Start task".into(),
            Self::Session(_) => "View session".into(),
            Self::Evidence { .. } => "View evidence".into(),
            Self::Criterion(id) | Self::Dependency(id) => id.clone().into(),
            Self::ToggleCheckbox(_, checked) => if *checked {
                "Uncheck task"
            } else {
                "Check task"
            }
            .into(),
        }
    }
}

fn task_links(
    id: &str,
    task: Option<&SpecificationTask>,
) -> impl Iterator<Item = TaskAction> + Clone {
    let session = task
        .and_then(|task| task.session.clone())
        .map(TaskAction::Session);
    let evidence = task
        .and_then(|task| task.latest_check.clone())
        .map(|observed_at| TaskAction::Evidence {
            task: id.to_owned(),
            observed_at,
        });
    session.into_iter().chain(evidence)
}

struct TaskActionsPicker {
    toolbar: WeakEntity<SpecificationEditor>,
    task_id: String,
    actions: Vec<TaskAction>,
    matches: Vec<usize>,
    selected: usize,
}

impl PickerDelegate for TaskActionsPicker {
    type ListItem = ListItem;

    fn name() -> &'static str {
        "specification-task-actions"
    }

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected
    }

    fn set_selected_index(
        &mut self,
        index: usize,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) {
        self.selected = index;
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        format!("Task {} actions", self.task_id).into()
    }

    fn no_matches_text(&self, _window: &mut Window, _cx: &mut App) -> Option<SharedString> {
        Some("No task actions available".into())
    }

    fn update_matches(
        &mut self,
        query: String,
        _window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let query = query.trim().to_lowercase();
        self.matches = self
            .actions
            .iter()
            .enumerate()
            .filter(|(_, action)| action.label().to_lowercase().contains(&query))
            .map(|(index, _)| index)
            .collect();
        self.selected = 0;
        cx.notify();
        Task::ready(())
    }

    fn confirm(&mut self, _secondary: bool, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        let Some(action) = self
            .matches
            .get(self.selected)
            .and_then(|index| self.actions.get(*index))
            .cloned()
        else {
            return;
        };
        let toolbar = self.toolbar.clone();
        cx.emit(gpui::DismissEvent);
        window.defer(cx, move |window, cx| {
            dispatch_task_action(&toolbar, action, window, cx);
        });
    }

    fn dismissed(&mut self, _window: &mut Window, _cx: &mut Context<Picker<Self>>) {}

    fn render_match(
        &self,
        index: usize,
        selected: bool,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let label = self.actions.get(*self.matches.get(index)?)?.label();
        Some(
            ListItem::new(index)
                .inset(true)
                .toggle_state(selected)
                .aria_role(gpui::Role::MenuItem)
                .aria_label(label.clone())
                .when(selected, ListItem::aria_active_descendant)
                .child(Label::new(label)),
        )
    }
}

fn task_icon(icon: IconName, color: Color) -> Icon {
    Icon::new(icon).size(IconSize::Small).color(color)
}

fn task_button(action: TaskAction, toolbar: &WeakEntity<SpecificationEditor>) -> Button {
    let toolbar = toolbar.clone();
    let label = action.label();
    let (icon, color) = match &action {
        TaskAction::Run(_) => (Some(IconName::BoltOutlined), Color::Accent),
        TaskAction::Session(_) => (Some(IconName::Chat), Color::Muted),
        TaskAction::Evidence { .. } => (Some(IconName::ListTodo), Color::Muted),
        _ => (None, Color::Muted),
    };
    Button::new(label.clone(), label)
        .size(ButtonSize::None)
        .label_size(LabelSize::Small)
        .start_icon(icon.map(|icon| task_icon(icon, color)))
        .color(color)
        .tab_index(0isize)
        .on_click(move |_, window, cx| dispatch_task_action(&toolbar, action.clone(), window, cx))
}

fn dispatch_task_action(
    toolbar: &WeakEntity<SpecificationEditor>,
    action: TaskAction,
    window: &mut Window,
    cx: &mut App,
) {
    toolbar
        .update(cx, |toolbar, cx| toolbar.handle_task(action, window, cx))
        .ok();
}

fn toggle_task_checkbox(
    editor: &mut Editor,
    range: &Range<Anchor>,
    window: &mut Window,
    cx: &mut Context<Editor>,
) {
    if editor.read_only(cx) {
        return;
    }
    let snapshot = editor.buffer().read(cx).snapshot(cx);
    let start = range.start.to_offset(&snapshot);
    let end = range.end.to_offset(&snapshot);
    let marker = snapshot.text_for_range(start..end).collect::<String>();
    let replacement = match marker.as_str() {
        "[ ]" => "x",
        "[x]" | "[X]" => " ",
        _ => return,
    };
    editor.finalize_last_transaction(cx);
    editor.transact(window, cx, |editor, _, cx| {
        editor.edit([(start + 1_usize..start + 2_usize, replacement)], cx);
    });
    editor.finalize_last_transaction(cx);
}

fn is_feature_file(root: &Path, specification: &Path, path: &Path) -> bool {
    path == specification
        || path.strip_prefix(root).is_ok_and(|path| {
            matches!(
                path.to_str(),
                Some("AGENTS.md" | ".rules" | ".specify/memory/constitution.md")
            )
        })
        || (path.parent() == specification.parent()
            && matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("spec.md" | "tasks.md" | "plan.md" | "made-workflow.json" | "verification.md")
            ))
}

fn feature_buffers(
    project: &Entity<Project>,
    root: &Path,
    specification: &Path,
    cx: &App,
) -> Vec<(PathBuf, Entity<Buffer>)> {
    project
        .read(cx)
        .buffer_store()
        .read(cx)
        .buffers()
        .filter_map(|buffer| {
            let path = project::File::from_dyn(buffer.read(cx).file())?.abs_path(cx);
            is_feature_file(root, specification, &path).then_some((path, buffer))
        })
        .collect()
}

async fn load_specification(
    root: &Path,
    path: &Path,
    cx: &gpui::AsyncApp,
) -> Result<SpecificationOverview> {
    let (root, path) = (root.to_owned(), path.to_owned());
    cx.background_executor()
        .spawn(async move {
            agent::specification_for_editor(root, &path)?
                .context("The specification is no longer available")
        })
        .await
}

fn short_revision(revision: &str) -> &str {
    revision.get(..10).unwrap_or(revision)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

enum Target {
    Requirements,
    Design,
    Evidence { task: String, observed_at: String },
    TaskRow(String),
    Criterion(String),
}

impl Target {
    fn range(&self, text: &str) -> Option<(usize, usize)> {
        if let Self::TaskRow(id) = self {
            return agent::specification_task_rows(text)
                .into_iter()
                .find(|row| &row.id == id)
                .map(|row| (row.offset, row.title_end - row.offset));
        }
        let mut offset = 0;
        let mut fallback = None;
        for line in text.split_inclusive('\n') {
            let heading = line
                .trim_start()
                .trim_start_matches('#')
                .trim()
                .to_ascii_lowercase();
            let range = (offset, line.trim_end().len());
            let found = match self {
                Self::Requirements => {
                    line.trim_start().starts_with("##")
                        && (heading.starts_with("requirements")
                            || heading.starts_with("user scenarios"))
                }
                Self::Design => {
                    line.trim_start().starts_with("##")
                        && (heading.starts_with("design")
                            || heading.starts_with("technical design")
                            || heading.starts_with("architecture"))
                }
                Self::Evidence { task, observed_at } => {
                    if let Some(title) = line.strip_prefix("## Task ") {
                        if fallback.is_some() {
                            break;
                        }
                        if title.trim() == task {
                            fallback = Some(range);
                        }
                    }
                    fallback.is_some()
                        && line
                            .split(" · ")
                            .nth(1)
                            .is_some_and(|date| date == observed_at)
                }
                Self::TaskRow(_) => false,
                Self::Criterion(id) => {
                    let mut words = line
                        .trim_start_matches([' ', '\t', '-', '*', '+', '#', '`'])
                        .split(|character: char| {
                            !character.is_ascii_alphanumeric() && character != '-'
                        });
                    if words.next() == Some(id) {
                        return Some(range);
                    }
                    if fallback.is_none() && words.any(|word| word == id) {
                        fallback = Some(range);
                    }
                    false
                }
            };
            if found {
                return Some(range);
            }
            offset += line.len();
        }
        fallback
    }
}
