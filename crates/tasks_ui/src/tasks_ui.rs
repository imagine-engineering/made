use std::{
    path::Path,
    sync::{Arc, LazyLock},
};

use anyhow::Context as _;
use collections::HashMap;
use editor::{Editor, MultiBufferOffset, ToPoint as _};
use gpui::{App, AppContext as _, Context, Entity, Task, TaskExt, Window};
use project::{Location, TaskContexts, TaskSourceKind, Worktree};
use task::{RevealTarget, TaskContext, TaskId, TaskTemplate, TaskVariables, VariableName};
use tree_sitter::{Query, StreamingIterator as _};
use workspace::Workspace;

mod modal;
#[cfg(feature = "test-explorer")]
mod test_explorer;
#[cfg(feature = "multiplayer-tools")]
mod workflows;

pub use modal::{Rerun, ShowAttachModal, Spawn, TaskOverrides, TasksModal};
#[cfg(feature = "test-explorer")]
pub use test_explorer::*;
#[cfg(feature = "multiplayer-tools")]
pub use workflows::*;

/// Inserts `new_task` (pretty-printed JSON object text) at the end of the top-level JSON
/// array in the editor's buffer, creating the array if the buffer has none, and moves the
/// cursor to the inserted task. The edit is left unsaved so callers decide whether to persist it.
pub fn insert_task_json_into_editor(
    editor: &mut Editor,
    new_task: String,
    window: &mut Window,
    cx: &mut Context<Editor>,
) -> anyhow::Result<()> {
    static LAST_ITEM_QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(
            &tree_sitter_json::LANGUAGE.into(),
            "(document (array (object) @object))", // TODO: use "." anchor to only match last object
        )
        .expect("Failed to create LAST_ITEM_QUERY")
    });
    static EMPTY_ARRAY_QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(
            &tree_sitter_json::LANGUAGE.into(),
            "(document (array) @array)",
        )
        .expect("Failed to create EMPTY_ARRAY_QUERY")
    });

    let content = editor.text(cx);
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_json::LANGUAGE.into())?;
    let mut cursor = tree_sitter::QueryCursor::new();
    let syntax_tree = parser
        .parse(&content, None)
        .context("could not parse tasks file")?;
    let mut matches = cursor.matches(
        &LAST_ITEM_QUERY,
        syntax_tree.root_node(),
        content.as_bytes(),
    );

    let mut last_offset = None;
    while let Some(mat) = matches.next() {
        if let Some(pos) = mat.captures.first().map(|m| m.node.byte_range().end) {
            last_offset = Some(MultiBufferOffset(pos))
        }
    }
    let mut edits = Vec::new();
    let mut cursor_position = MultiBufferOffset(0);

    if let Some(pos) = last_offset {
        edits.push((pos..pos, format!(",\n{new_task}")));
        cursor_position = pos + ",\n  ".len();
    } else {
        let mut matches = cursor.matches(
            &EMPTY_ARRAY_QUERY,
            syntax_tree.root_node(),
            content.as_bytes(),
        );

        if let Some(mat) = matches.next() {
            if let Some(pos) = mat.captures.first().map(|m| m.node.byte_range().end - 1) {
                edits.push((
                    MultiBufferOffset(pos)..MultiBufferOffset(pos),
                    format!("\n{new_task}\n"),
                ));
                cursor_position = MultiBufferOffset(pos) + "\n  ".len();
            }
        } else {
            edits.push((
                MultiBufferOffset(0)..MultiBufferOffset(0),
                format!("[\n{}\n]", new_task),
            ));
            cursor_position = MultiBufferOffset("[\n  ".len());
        }
    }
    editor.transact(window, cx, |editor, window, cx| {
        editor.edit(edits, cx);
        let snapshot = editor.buffer().read(cx).read(cx);
        let point = cursor_position.to_point(&snapshot);
        drop(snapshot);
        editor.go_to_singleton_buffer_point(point, window, cx);
    });
    Ok(())
}

pub fn init(cx: &mut App) {
    #[cfg(feature = "test-explorer")]
    test_explorer::init(cx);

    cx.observe_new(
        |workspace: &mut Workspace, _: Option<&mut Window>, _: &mut Context<Workspace>| {
            workspace
                .register_action(spawn_task_or_modal)
                .register_action(move |workspace, action: &modal::Rerun, window, cx| {
                    if let Some((task_source_kind, mut last_scheduled_task)) = workspace
                        .project()
                        .read(cx)
                        .task_store()
                        .read(cx)
                        .task_inventory()
                        .and_then(|inventory| {
                            inventory.read(cx).last_scheduled_task(
                                action
                                    .task_id
                                    .as_ref()
                                    .map(|id| TaskId(id.clone()))
                                    .as_ref(),
                            )
                        })
                    {
                        if action.reevaluate_context {
                            let mut original_task = last_scheduled_task.original_task().clone();
                            if let Some(allow_concurrent_runs) = action.allow_concurrent_runs {
                                original_task.allow_concurrent_runs = allow_concurrent_runs;
                            }
                            if let Some(use_new_terminal) = action.use_new_terminal {
                                original_task.use_new_terminal = use_new_terminal;
                            }
                            let task_contexts = task_contexts(workspace, window, cx);
                            cx.spawn_in(window, async move |workspace, cx| {
                                let task_contexts = task_contexts.await;
                                let default_context = TaskContext::default();
                                workspace
                                    .update_in(cx, |workspace, window, cx| {
                                        workspace.schedule_task(
                                            task_source_kind,
                                            &original_task,
                                            task_contexts
                                                .active_context()
                                                .unwrap_or(&default_context),
                                            false,
                                            window,
                                            cx,
                                        )
                                    })
                                    .ok()
                            })
                            .detach()
                        } else {
                            let resolved = &mut last_scheduled_task.resolved;

                            if let Some(allow_concurrent_runs) = action.allow_concurrent_runs {
                                resolved.allow_concurrent_runs = allow_concurrent_runs;
                            }
                            if let Some(use_new_terminal) = action.use_new_terminal {
                                resolved.use_new_terminal = use_new_terminal;
                            }

                            workspace.schedule_resolved_task(
                                task_source_kind,
                                last_scheduled_task,
                                false,
                                window,
                                cx,
                            );
                        }
                    } else {
                        spawn_task_or_modal(
                            workspace,
                            &Spawn::ViaModal {
                                reveal_target: None,
                            },
                            window,
                            cx,
                        );
                    };
                });
        },
    )
    .detach();
}

fn spawn_task_or_modal(
    workspace: &mut Workspace,
    action: &Spawn,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if let Some(provider) = workspace.debugger_provider() {
        provider.spawn_task_or_modal(workspace, action, window, cx);
        return;
    }

    match action {
        Spawn::ByName {
            task_name,
            reveal_target,
        } => {
            let overrides = reveal_target.map(|reveal_target| TaskOverrides {
                reveal_target: Some(reveal_target),
            });
            let name = task_name.clone();
            spawn_tasks_filtered(move |(_, task)| task.label.eq(&name), overrides, window, cx)
                .detach_and_log_err(cx)
        }
        Spawn::ByTag {
            task_tag,
            reveal_target,
        } => {
            let overrides = reveal_target.map(|reveal_target| TaskOverrides {
                reveal_target: Some(reveal_target),
            });
            let tag = task_tag.clone();
            spawn_tasks_filtered(
                move |(_, task)| task.tags.contains(&tag),
                overrides,
                window,
                cx,
            )
            .detach_and_log_err(cx)
        }
        Spawn::ViaModal { reveal_target } => {
            toggle_modal(workspace, *reveal_target, window, cx).detach()
        }
    }
}

pub fn toggle_modal(
    workspace: &mut Workspace,
    reveal_target: Option<RevealTarget>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> Task<()> {
    let task_store = workspace.project().read(cx).task_store().clone();
    let workspace_handle = workspace.weak_handle();
    let can_open_modal = workspace
        .project()
        .read_with(cx, |project, _| !project.is_via_collab());
    if can_open_modal {
        let task_contexts = task_contexts(workspace, window, cx);
        cx.spawn_in(window, async move |workspace, cx| {
            let task_contexts = Arc::new(task_contexts.await);
            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.toggle_modal(window, cx, |window, cx| {
                        TasksModal::new(
                            task_store.clone(),
                            task_contexts,
                            reveal_target.map(|target| TaskOverrides {
                                reveal_target: Some(target),
                            }),
                            true,
                            workspace_handle,
                            window,
                            cx,
                        )
                    })
                })
                .ok();
        })
    } else {
        Task::ready(())
    }
}

pub fn spawn_tasks_filtered<F>(
    mut predicate: F,
    overrides: Option<TaskOverrides>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> Task<anyhow::Result<()>>
where
    F: FnMut((&TaskSourceKind, &TaskTemplate)) -> bool + 'static,
{
    cx.spawn_in(window, async move |workspace, cx| {
        let task_contexts = workspace.update_in(cx, |workspace, window, cx| {
            task_contexts(workspace, window, cx)
        })?;
        let task_contexts = task_contexts.await;
        let mut tasks = workspace
            .update(cx, |workspace, cx| {
                let Some(task_inventory) = workspace
                    .project()
                    .read(cx)
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .cloned()
                else {
                    return Task::ready(Vec::new());
                };
                let (language, buffer) = task_contexts
                    .location()
                    .map(|location| {
                        let buffer = location.buffer.clone();
                        (
                            buffer.read(cx).language_at(location.range.start),
                            Some(buffer),
                        )
                    })
                    .unwrap_or_default();
                task_inventory
                    .read(cx)
                    .list_tasks(buffer, language, task_contexts.worktree(), cx)
            })?
            .await;

        let did_spawn = workspace
            .update_in(cx, |workspace, window, cx| {
                let default_context = TaskContext::default();
                let active_context = task_contexts.active_context().unwrap_or(&default_context);

                tasks.retain_mut(|(task_source_kind, target_task)| {
                    if predicate((task_source_kind, target_task)) {
                        if let Some(overrides) = &overrides
                            && let Some(target_override) = overrides.reveal_target
                        {
                            target_task.reveal_target = target_override;
                        }
                        workspace.schedule_task(
                            task_source_kind.clone(),
                            target_task,
                            active_context,
                            false,
                            window,
                            cx,
                        );
                        true
                    } else {
                        false
                    }
                });

                if tasks.is_empty() { None } else { Some(()) }
            })?
            .is_some();
        if !did_spawn {
            workspace
                .update_in(cx, |workspace, window, cx| {
                    spawn_task_or_modal(
                        workspace,
                        &Spawn::ViaModal {
                            reveal_target: overrides.and_then(|overrides| overrides.reveal_target),
                        },
                        window,
                        cx,
                    );
                })
                .ok();
        }

        Ok(())
    })
}

pub fn task_contexts(
    workspace: &Workspace,
    window: &mut Window,
    cx: &mut App,
) -> Task<TaskContexts> {
    let active_item = workspace.active_item(cx);
    let active_worktree = active_item
        .as_ref()
        .and_then(|item| item.project_path(cx))
        .map(|project_path| project_path.worktree_id)
        .filter(|worktree_id| {
            workspace
                .project()
                .read(cx)
                .worktree_for_id(*worktree_id, cx)
                .is_some_and(|worktree| is_visible_directory(&worktree, cx))
        })
        .or_else(|| {
            workspace
                .visible_worktrees(cx)
                .next()
                .map(|tree| tree.read(cx).id())
        });

    let active_editor = active_item.and_then(|item| item.act_as::<Editor>(cx));

    let editor_context_task = active_editor.as_ref().map(|active_editor| {
        active_editor.update(cx, |editor, cx| editor.task_context(window, cx))
    });

    let location = active_editor.as_ref().and_then(|editor| {
        editor.update(cx, |editor, cx| {
            let selection = editor.selections.newest_anchor();
            let multi_buffer = editor.buffer().clone();
            let multi_buffer_snapshot = multi_buffer.read(cx).snapshot(cx);
            let (buffer_snapshot, buffer_offset) =
                multi_buffer_snapshot.point_to_buffer_offset(selection.head())?;
            let buffer_anchor = buffer_snapshot.anchor_before(buffer_offset);
            let buffer = multi_buffer.read(cx).buffer(buffer_snapshot.remote_id())?;
            Some(Location {
                buffer,
                range: buffer_anchor..buffer_anchor,
            })
        })
    });

    let lsp_task_sources = active_editor
        .as_ref()
        .map(|active_editor| {
            active_editor.update(cx, |editor, cx| editor.lsp_task_sources(false, false, cx))
        })
        .unwrap_or_default();

    let latest_selection = active_editor.as_ref().and_then(|active_editor| {
        let snapshot = active_editor.read(cx).buffer().read(cx).snapshot(cx);
        snapshot
            .anchor_to_buffer_anchor(active_editor.read(cx).selections.newest_anchor().head())
            .map(|(anchor, _)| anchor)
    });

    let mut worktree_abs_paths = workspace
        .worktrees(cx)
        .filter(|worktree| is_visible_directory(worktree, cx))
        .map(|worktree| {
            let worktree = worktree.read(cx);
            (worktree.id(), worktree.abs_path())
        })
        .collect::<HashMap<_, _>>();

    cx.background_spawn(async move {
        let mut task_contexts = TaskContexts::default();

        task_contexts.lsp_task_sources = lsp_task_sources;
        task_contexts.latest_selection = latest_selection;

        if let Some(editor_context_task) = editor_context_task
            && let Some(editor_context) = editor_context_task.await
        {
            task_contexts.active_item_context = Some((active_worktree, location, editor_context));
        }

        if let Some(active_worktree) = active_worktree {
            if let Some(active_worktree_abs_path) = worktree_abs_paths.remove(&active_worktree) {
                task_contexts.active_worktree_context =
                    Some((active_worktree, worktree_context(&active_worktree_abs_path)));
            }
        } else if worktree_abs_paths.len() == 1 {
            task_contexts.active_worktree_context = worktree_abs_paths
                .drain()
                .next()
                .map(|(id, abs_path)| (id, worktree_context(&abs_path)));
        }

        task_contexts.other_worktree_contexts.extend(
            worktree_abs_paths
                .into_iter()
                .map(|(id, abs_path)| (id, worktree_context(&abs_path))),
        );
        task_contexts
    })
}

fn is_visible_directory(worktree: &Entity<Worktree>, cx: &App) -> bool {
    let worktree = worktree.read(cx);
    worktree.is_visible() && worktree.root_entry().is_some_and(|entry| entry.is_dir())
}

fn worktree_context(worktree_abs_path: &Path) -> TaskContext {
    let mut task_variables = TaskVariables::default();
    task_variables.insert(
        VariableName::WorktreeRoot,
        worktree_abs_path.to_string_lossy().into_owned(),
    );
    TaskContext {
        cwd: Some(worktree_abs_path.to_path_buf()),
        task_variables,
        project_env: HashMap::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use editor::{Editor, MultiBufferOffset, SelectionEffects};
    use gpui::TestAppContext;
    use language::{Language, LanguageConfig};
    use project::{BasicContextProvider, FakeFs, Project, task_store::TaskStore};
    use serde_json::json;
    use task::{TaskContext, TaskVariables, VariableName};
    use ui::VisualContext;
    use util::{path, rel_path::rel_path};
    use workspace::{AppState, MultiWorkspace};

    use crate::task_contexts;

    #[gpui::test]
    async fn test_default_language_context(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({
                ".zed": {
                    "tasks.json": r#"[
                            {
                                "label": "example task",
                                "command": "echo",
                                "args": ["4"]
                            },
                            {
                                "label": "another one",
                                "command": "echo",
                                "args": ["55"]
                            },
                        ]"#,
                },
                "a.ts": "function this_is_a_test() { }",
                "rust": {
                                    "b.rs": "use std; fn this_is_a_rust_file() { }",
                }

            }),
        )
        .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let (worktree_store, git_store) = project.read_with(cx, |project, _| {
            (project.worktree_store(), project.git_store().clone())
        });
        let rust_language = Arc::new(
            Language::new(
                LanguageConfig {
                    name: "Rust".into(),
                    ..Default::default()
                },
                Some(tree_sitter_rust::LANGUAGE.into()),
            )
            .with_outline_query(
                r#"(function_item
            "fn" @context
            name: (_) @name) @item"#,
            )
            .unwrap()
            .with_context_provider(Some(Arc::new(BasicContextProvider::new(
                worktree_store.clone(),
                git_store.clone(),
            )))),
        );

        let typescript_language = Arc::new(
            Language::new(
                LanguageConfig {
                    name: "TypeScript".into(),
                    ..Default::default()
                },
                Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            )
            .with_outline_query(
                r#"(function_declaration
                    "async"? @context
                    "function" @context
                    name: (_) @name
                    parameters: (formal_parameters
                        "(" @context
                        ")" @context)) @item"#,
            )
            .unwrap()
            .with_context_provider(Some(Arc::new(BasicContextProvider::new(
                worktree_store.clone(),
                git_store.clone(),
            )))),
        );

        let worktree_id = project.update(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let buffer1 = workspace
            .update(cx, |this, cx| {
                this.project().update(cx, |this, cx| {
                    this.open_buffer((worktree_id, rel_path("a.ts")), cx)
                })
            })
            .await
            .unwrap();
        buffer1.update(cx, |this, cx| {
            this.set_language(Some(typescript_language), cx)
        });
        let editor1 = cx.new_window_entity(|window, cx| {
            Editor::for_buffer(buffer1, Some(project.clone()), window, cx)
        });
        let buffer2 = workspace
            .update(cx, |this, cx| {
                this.project().update(cx, |this, cx| {
                    this.open_buffer((worktree_id, rel_path("rust/b.rs")), cx)
                })
            })
            .await
            .unwrap();
        buffer2.update(cx, |this, cx| this.set_language(Some(rust_language), cx));
        let editor2 = cx
            .new_window_entity(|window, cx| Editor::for_buffer(buffer2, Some(project), window, cx));

        let first_context = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.add_item_to_center(Box::new(editor1.clone()), window, cx);
                workspace.add_item_to_center(Box::new(editor2.clone()), window, cx);
                assert_eq!(
                    workspace.active_item(cx).unwrap().item_id(),
                    editor2.entity_id()
                );
                task_contexts(workspace, window, cx)
            })
            .await;

        assert_eq!(
            first_context
                .active_context()
                .expect("Should have an active context"),
            &TaskContext {
                cwd: Some(path!("/dir").into()),
                task_variables: TaskVariables::from_iter([
                    (VariableName::File, path!("/dir/rust/b.rs").into()),
                    (VariableName::Filename, "b.rs".into()),
                    (VariableName::RelativeFile, path!("rust/b.rs").into()),
                    (VariableName::RelativeDir, "rust".into()),
                    (VariableName::Dirname, path!("/dir/rust").into()),
                    (VariableName::Stem, "b".into()),
                    (VariableName::WorktreeRoot, path!("/dir").into()),
                    (VariableName::Row, "1".into()),
                    (VariableName::Column, "1".into()),
                    (VariableName::Language, "Rust".into()),
                ]),
                project_env: HashMap::default(),
            }
        );

        // And now, let's select an identifier.
        editor2.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(14)..MultiBufferOffset(18)])
            })
        });

        assert_eq!(
            workspace
                .update_in(cx, |workspace, window, cx| {
                    task_contexts(workspace, window, cx)
                })
                .await
                .active_context()
                .expect("Should have an active context"),
            &TaskContext {
                cwd: Some(path!("/dir").into()),
                task_variables: TaskVariables::from_iter([
                    (VariableName::File, path!("/dir/rust/b.rs").into()),
                    (VariableName::Filename, "b.rs".into()),
                    (VariableName::RelativeFile, path!("rust/b.rs").into()),
                    (VariableName::RelativeDir, "rust".into()),
                    (VariableName::Dirname, path!("/dir/rust").into()),
                    (VariableName::Stem, "b".into()),
                    (VariableName::WorktreeRoot, path!("/dir").into()),
                    (VariableName::Row, "1".into()),
                    (VariableName::Column, "15".into()),
                    (VariableName::SelectedText, "is_i".into()),
                    (VariableName::Symbol, "this_is_a_rust_file".into()),
                    (VariableName::Language, "Rust".into()),
                ]),
                project_env: HashMap::default(),
            }
        );

        assert_eq!(
            workspace
                .update_in(cx, |workspace, window, cx| {
                    // Now, let's switch the active item to .ts file.
                    workspace.activate_item(&editor1, true, true, window, cx);
                    task_contexts(workspace, window, cx)
                })
                .await
                .active_context()
                .expect("Should have an active context"),
            &TaskContext {
                cwd: Some(path!("/dir").into()),
                task_variables: TaskVariables::from_iter([
                    (VariableName::File, path!("/dir/a.ts").into()),
                    (VariableName::Filename, "a.ts".into()),
                    (VariableName::RelativeFile, "a.ts".into()),
                    (VariableName::RelativeDir, ".".into()),
                    (VariableName::Dirname, path!("/dir").into()),
                    (VariableName::Stem, "a".into()),
                    (VariableName::WorktreeRoot, path!("/dir").into()),
                    (VariableName::Row, "1".into()),
                    (VariableName::Column, "1".into()),
                    (VariableName::Symbol, "this_is_a_test".into()),
                    (VariableName::Language, "TypeScript".into()),
                ]),
                project_env: HashMap::default(),
            }
        );
    }

    #[gpui::test]
    async fn test_rerun_artifact_binding_preserves_task_identity_and_rebinds_output(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::RefCell, collections::HashSet, process::ExitStatus, rc::Rc};

        use gpui::{App, Focusable as _, Task, Window};
        use project::TaskSourceKind;
        use task::{SpawnInTerminal, TaskArtifact, TaskArtifactArgumentBinding, TaskArtifactKind};
        use workspace::TerminalProvider;

        struct RecordingTerminal(Rc<RefCell<Vec<SpawnInTerminal>>>);

        impl TerminalProvider for RecordingTerminal {
            fn spawn(
                &self,
                task: SpawnInTerminal,
                _window: &mut Window,
                _cx: &mut App,
            ) -> Task<Option<anyhow::Result<ExitStatus>>> {
                self.0.borrow_mut().push(task);
                // Completion must not open an artifact and change the context being reevaluated.
                Task::ready(None)
            }
        }

        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/fixture"), json!({"target": {}}))
            .await;
        let project = Project::test(fs, [path!("/fixture").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let launches = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(RecordingTerminal(launches.clone()));
        });
        let unbound_path = "target/coverage.json";
        let template = task::TaskTemplate {
            label: "Invocation-bound collector".into(),
            command: "collector".into(),
            args: vec![
                "collect".into(),
                "--output-path".into(),
                unbound_path.into(),
                "--literal-path".into(),
                unbound_path.into(),
            ],
            allow_concurrent_runs: true,
            save: task::SaveStrategy::None,
            artifact: Some(TaskArtifact {
                path: unbound_path.into(),
                kind: TaskArtifactKind::Data,
                max_bytes: 1024,
            }),
            artifact_argument_binding: Some(TaskArtifactArgumentBinding { argument_index: 2 }),
            ..Default::default()
        };
        let contexts = workspace
            .update_in(cx, |workspace, window, cx| {
                task_contexts(workspace, window, cx)
            })
            .await;
        let initial_context = contexts.active_context().expect("fixture worktree context");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_task(
                TaskSourceKind::UserInput,
                &template,
                initial_context,
                false,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let mut task_id = None;
        let mut output_paths = HashSet::new();
        for (invocation, reevaluate_context) in
            [None, Some(false), Some(true), Some(false), Some(true)]
                .into_iter()
                .enumerate()
        {
            if let Some(reevaluate_context) = reevaluate_context {
                workspace.update_in(cx, |workspace, window, cx| {
                    window.focus(&workspace.focus_handle(cx), cx);
                });
                cx.dispatch_action(crate::Rerun {
                    reevaluate_context,
                    ..Default::default()
                });
                cx.run_until_parked();
            }
            let (source, scheduled) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("fixture task inventory")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("registered Rerun schedules a task")
            });
            assert_eq!(source, TaskSourceKind::UserInput);
            assert_eq!(scheduled.original_task(), &template);
            let expected_id = task_id.get_or_insert(scheduled.id.clone());
            assert_eq!(&scheduled.id, &*expected_id);
            assert_eq!(scheduled.resolved.id, scheduled.id);
            assert_eq!(launches.borrow().len(), invocation + 1);
            assert_eq!(launches.borrow().last(), Some(&scheduled.resolved));
            let artifact = scheduled
                .resolved_artifact()
                .expect("collector artifact declaration");
            assert_ne!(artifact.path, unbound_path);
            assert!(output_paths.insert(artifact.path.clone()));
            assert_eq!(artifact.kind, TaskArtifactKind::Data);
            assert_eq!(artifact.max_bytes, 1024);
            assert!(scheduled.resolved_artifact_group().is_none());
            let mut expected_arguments = template.args.clone();
            *expected_arguments
                .get_mut(2)
                .expect("bound output argument") = artifact.path.clone();
            assert_eq!(scheduled.resolved.args, expected_arguments);
            assert_eq!(scheduled.resolved.cwd, initial_context.cwd);
            assert!(scheduled.resolved.command_label.contains(&artifact.path));
        }
        assert_eq!(output_paths.len(), 5);
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn test_rerun_artifact_group_binding_preserves_task_identity_and_rebinds_pair(
        cx: &mut TestAppContext,
    ) {
        check_rerun_artifact_group_entrypoints(cx, false).await;
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_profile_capture_entrypoint_registered_rerun_holds_before_launch(
        cx: &mut TestAppContext,
    ) {
        check_rerun_artifact_group_entrypoints(cx, true).await;
    }

    #[cfg(unix)]
    async fn check_rerun_artifact_group_entrypoints(
        cx: &mut TestAppContext,
        native_disclosure: bool,
    ) {
        use std::{cell::RefCell, collections::HashSet, process::ExitStatus, rc::Rc};

        use gpui::{App, Focusable as _, Task, Window};
        use project::TaskSourceKind;
        use task::{
            SpawnInTerminal, TaskArtifact, TaskArtifactGroupArgumentBinding, TaskArtifactKind,
            TaskArtifactRelatedArgumentBinding,
        };
        use workspace::TerminalProvider;

        type ReceiverResult = anyhow::Result<Vec<(String, std::process::Output)>>;
        type Received = (Vec<Option<ReceiverResult>>, Option<std::task::Waker>);
        struct ReceivingTerminal {
            launches: Rc<RefCell<Vec<SpawnInTerminal>>>,
            received: Rc<RefCell<Received>>,
        }

        impl TerminalProvider for ReceivingTerminal {
            fn spawn_prepared(
                &self,
                task: SpawnInTerminal,
                _handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<anyhow::Result<ExitStatus>>> {
                let terminal = Self {
                    launches: self.launches.clone(),
                    received: self.received.clone(),
                };
                window.to_async(cx).spawn(async move |cx| {
                    let prepared = match prepare(task, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    let result =
                        match cx.update(|window, cx| terminal.spawn(prepared.launch, window, cx)) {
                            Ok(result) => result,
                            Err(error) => return Some(Err(error)),
                        };
                    match prepared.completion {
                        Some(complete) => complete(result, cx).await,
                        None => result.await,
                    }
                })
            }

            fn spawn(
                &self,
                task: SpawnInTerminal,
                _window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<anyhow::Result<ExitStatus>>> {
                self.launches.borrow_mut().push(task.clone());
                let received = self.received.clone();
                cx.spawn(async move |_| {
                    let result: anyhow::Result<_> = async {
                        let mut outputs = Vec::new();
                        for shell in [
                            "/bin/sh",
                            "/bin/bash",
                            #[cfg(target_os = "macos")]
                            "/bin/zsh",
                        ] {
                            let prepared = terminal_view::terminal_panel::prepare_task_for_spawn(
                                &task,
                                &task::Shell::Program(shell.into()),
                                false,
                            );
                            let mut command =
                                std::process::Command::new(prepared.command.ok_or_else(|| {
                                    anyhow::anyhow!("prepared native argument receiver shell")
                                })?);
                            command
                                .args(prepared.args)
                                .envs(&task.env)
                                .env("RUST_NATIVE_LITERAL_VALUE", "expanded");
                            let output = util::process::Child::spawn(
                                command,
                                std::process::Stdio::null(),
                                std::process::Stdio::piped(),
                                std::process::Stdio::piped(),
                            )?
                            .output()
                            .await?;
                            outputs.push((shell.to_owned(), output));
                        }
                        Ok(outputs)
                    }
                    .await;
                    let waker = {
                        let mut received = received.borrow_mut();
                        received.0.push(Some(result));
                        received.1.take()
                    };
                    if let Some(waker) = waker {
                        waker.wake();
                    }
                    // This receiver does not produce profile artifacts or certify capture completion.
                    None
                })
            }
        }

        cx.executor().allow_parking();
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/fixture"), json!({"target": {}}))
            .await;
        let project = Project::test(fs, [path!("/fixture").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let launches = Rc::new(RefCell::new(Vec::new()));
        let received = Rc::new(RefCell::new((Vec::new(), None)));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(ReceivingTerminal {
                launches: launches.clone(),
                received: received.clone(),
            });
        });
        if native_disclosure {
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-native-profiling-experimental",
                    true,
                    cx,
                );
            });
        }
        let unbound_path = "target/native captures/capture.json";
        let template = task::TaskTemplate {
            label: "Invocation-bound collector".into(),
            command: "/usr/bin/printf".into(),
            args: vec![
                "%s\\0".into(),
                "--output-path".into(),
                unbound_path.into(),
                "--literal-path".into(),
                unbound_path.into(),
                "$ZED_UNAVAILABLE_NATIVE_ARG".into(),
                "' ; $HOME $(printf injected) 日本語 \"quotes\"".into(),
                "$RUST_NATIVE_LITERAL_VALUE".into(),
                "".into(),
            ],
            literal_arguments: true,
            allow_concurrent_runs: true,
            save: task::SaveStrategy::None,
            artifact_group: Some(task::TaskArtifactGroup {
                format: if native_disclosure {
                    "samply-0.13.1"
                } else {
                    "paired-data-v1"
                }
                .into(),
                files: collections::BTreeMap::from_iter([
                    (
                        "profile".into(),
                        TaskArtifact {
                            path: unbound_path.into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    ),
                    (
                        "symbols".into(),
                        TaskArtifact {
                            path: "target/native captures/capture.syms.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 2048,
                        },
                    ),
                ]),
            }),
            artifact_group_argument_binding: Some(TaskArtifactGroupArgumentBinding {
                argument_index: 2,
                role: "profile".into(),
                related: vec![TaskArtifactRelatedArgumentBinding {
                    role: "symbols".into(),
                    extension: "syms.json".into(),
                    argument_index: None,
                }],
            }),
            ..Default::default()
        };
        let persisted = serde_json::to_string(&task::TaskTemplates(vec![template]))
            .expect("persist template mode");
        let task::TaskTemplates(mut loaded) =
            serde_json::from_str(&persisted).expect("load persisted template mode");
        let template = loaded.pop().expect("one persisted template");
        let contexts = workspace
            .update_in(cx, |workspace, window, cx| {
                task_contexts(workspace, window, cx)
            })
            .await;
        let initial_context = contexts.active_context().expect("fixture worktree context");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_task(
                TaskSourceKind::UserInput,
                &template,
                initial_context,
                false,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let mut task_id = None;
        let mut output_paths = HashSet::new();
        let mut companion_paths = HashSet::new();
        for (invocation, reevaluate_context) in
            [None, Some(false), Some(true), Some(false), Some(true)]
                .into_iter()
                .enumerate()
        {
            if let Some(reevaluate_context) = reevaluate_context {
                workspace.update_in(cx, |workspace, window, cx| {
                    window.focus(&workspace.focus_handle(cx), cx);
                });
                cx.dispatch_action(crate::Rerun {
                    reevaluate_context,
                    ..Default::default()
                });
                cx.run_until_parked();
            }
            let (source, scheduled) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("fixture task inventory")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("registered Rerun schedules a task")
            });
            assert_eq!(source, TaskSourceKind::UserInput);
            assert_eq!(scheduled.original_task(), &template);
            let expected_id = task_id.get_or_insert(scheduled.id.clone());
            assert_eq!(&scheduled.id, &*expected_id);
            assert_eq!(scheduled.resolved.id, scheduled.id);
            if native_disclosure {
                assert_eq!(
                    launches.borrow().len(),
                    invocation,
                    "No receiver runs while consent is held"
                );
                let (message, detail) = cx
                    .pending_prompt()
                    .expect("Every native invocation requires consent");
                assert_eq!(message, "Capture a native profile on the project host?");
                let cwd = scheduled.resolved.cwd.as_ref().expect("Final host cwd");
                for artifact in scheduled
                    .resolved_artifact_group()
                    .expect("Final pair")
                    .files
                    .values()
                {
                    assert!(
                        detail.contains(&cwd.join(&artifact.path).to_string_lossy().to_string())
                    );
                }
                for term in [
                    "not uploaded",
                    "telemetry",
                    "raw files remain",
                    "Delete them",
                    "session-only",
                    "only to this capture",
                ] {
                    assert!(
                        detail.contains(term),
                        "Missing privacy or retention term: {term}"
                    );
                }
                assert!(!detail.contains("RUST_NATIVE_LITERAL_VALUE"));
                assert!(!detail.contains("ZED_UNAVAILABLE_NATIVE_ARG"));
                assert!(!detail.contains("$(printf injected)"));
                cx.simulate_prompt_answer("Capture");
                cx.run_until_parked();
            }
            assert_eq!(launches.borrow().len(), invocation + 1);
            assert_eq!(launches.borrow().last(), Some(&scheduled.resolved));
            assert!(scheduled.resolved_artifact().is_none());
            let group = scheduled.resolved_artifact_group().expect("collector pair");
            let artifact = group.files.get("profile").expect("profile");
            let symbols = group.files.get("symbols").expect("symbols");
            assert_eq!(
                std::path::Path::new(&artifact.path).with_extension("syms.json"),
                std::path::Path::new(&symbols.path)
            );
            assert_ne!(symbols.path, "target/native captures/capture.syms.json");
            assert!(companion_paths.insert(symbols.path.clone()));
            assert_eq!(symbols.max_bytes, 2048);
            let filename = std::path::Path::new(&artifact.path)
                .file_name()
                .and_then(|name| name.to_str())
                .expect("bound output filename");
            let uuid = filename
                .strip_prefix("capture-")
                .and_then(|filename| filename.strip_suffix(".json"))
                .expect("one invocation suffix");
            assert_eq!(
                uuid.len(),
                32,
                "cached reruns must not accumulate invocation suffixes"
            );
            assert!(uuid.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_ne!(artifact.path, unbound_path);
            assert!(output_paths.insert(artifact.path.clone()));
            assert_eq!(artifact.kind, TaskArtifactKind::Data);
            assert_eq!(artifact.max_bytes, 1024);
            assert_eq!(group.files.len(), 2);
            let mut expected_arguments = template.args.clone();
            *expected_arguments
                .get_mut(2)
                .expect("bound output argument") = artifact.path.clone();
            assert_eq!(scheduled.resolved.args, expected_arguments);
            assert_eq!(scheduled.resolved.literal_prefix_args, template.args.len());
            assert!(scheduled.original_task().literal_arguments);
            assert_eq!(scheduled.resolved.cwd, initial_context.cwd);
            assert!(scheduled.resolved.command_label.contains(&artifact.path));
            let outputs = std::future::poll_fn(|cx| {
                let mut received = received.borrow_mut();
                if let Some(result) = received.0.get_mut(invocation).and_then(Option::take) {
                    std::task::Poll::Ready(result)
                } else {
                    received.1 = Some(cx.waker().clone());
                    std::task::Poll::Pending
                }
            })
            .await
            .expect("actual receiver processes");
            let expected = expected_arguments
                .iter()
                .skip(1)
                .flat_map(|argument| {
                    argument
                        .as_bytes()
                        .iter()
                        .copied()
                        .chain(std::iter::once(0))
                })
                .collect::<Vec<_>>();
            assert_eq!(outputs.len(), if cfg!(target_os = "macos") { 3 } else { 2 });
            for (shell, output) in outputs {
                assert!(
                    output.status.success(),
                    "{shell} invocation {invocation}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(
                    output.stdout, expected,
                    "{shell} registered invocation {invocation}"
                );
                assert!(
                    output.stderr.is_empty(),
                    "receiver stderr must remain empty"
                );
                eprintln!(
                    "native-pair-registered-rerun shell={shell} invocation={invocation} reevaluate={reevaluate_context:?} task_id={} profile={} symbols={} argv_bytes={}",
                    scheduled.id.0,
                    artifact.path,
                    symbols.path,
                    output.stdout.len()
                );
            }
        }
        assert_eq!(output_paths.len(), 5);
        assert_eq!(companion_paths.len(), 5);
        if native_disclosure {
            cx.dispatch_action(crate::Rerun {
                reevaluate_context: true,
                ..Default::default()
            });
            cx.run_until_parked();
            assert!(cx.has_pending_prompt());
            assert_eq!(launches.borrow().len(), 5);
            cx.simulate_prompt_answer("Cancel");
            cx.run_until_parked();
            assert!(!cx.has_pending_prompt());
            assert_eq!(
                launches.borrow().len(),
                5,
                "Cancelled native Rerun cannot reach the receiver"
            );
        }
    }

    pub(crate) fn init_test(cx: &mut TestAppContext) -> Arc<AppState> {
        cx.update(|cx| {
            let state = AppState::test(cx);
            crate::init(cx);
            editor::init(cx);
            TaskStore::init(None);
            state
        })
    }
}
