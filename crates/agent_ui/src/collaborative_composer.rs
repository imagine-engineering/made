use std::{error::Error, fmt};

use gpui::{App, Context, Entity, EntityId, Focusable, Render, Subscription, Window};
use ui::prelude::*;
use workspace::{
    Workspace,
    collaborative_composer::{
        CollaborativeComposerActionError, CollaborativeComposerProvider,
        CollaborativeComposerRegistration, CollaborativeComposerRegistrationError,
    },
};

use crate::{AgentPanel, conversation_view::ThreadView, message_editor::MessageEditorEvent};

pub struct CollaborativeComposerAdapter {
    thread_view_id: EntityId,
    provider: CollaborativeComposerProvider,
}

impl CollaborativeComposerAdapter {
    pub fn from_agent_panel(
        agent_panel: &Entity<AgentPanel>,
        workspace: &Entity<Workspace>,
        cx: &mut App,
    ) -> Result<Self, CollaborativeComposerAdapterError> {
        let thread_view = agent_panel
            .read(cx)
            .active_conversation_view()
            .and_then(|conversation| conversation.read(cx).collaborative_active_thread().cloned())
            .ok_or(CollaborativeComposerAdapterError::ThreadUnavailable)?;
        let (project, message_editor) = thread_view.read_with(cx, |thread_view, _| {
            (
                thread_view.project.upgrade(),
                thread_view.message_editor.clone(),
            )
        });
        let project = project.ok_or(CollaborativeComposerAdapterError::ProjectUnavailable)?;
        if project.entity_id() != workspace.read(cx).project().entity_id() {
            return Err(CollaborativeComposerAdapterError::ProjectMismatch);
        }
        let focus_handle = message_editor.read(cx).focus_handle(cx);

        let submit_editor = message_editor.downgrade();
        let cancel_editor = message_editor.downgrade();
        let composer = cx.new(|cx| CollaborativeAcpComposer::new(thread_view.clone(), cx));
        let footer = cx.new(|cx| {
            let observation = cx.observe(&thread_view, |_, _, cx| cx.notify());
            CollaborativeAcpComposerFooter {
                thread_view: thread_view.clone(),
                _thread_view_observation: observation,
            }
        });
        let provider = CollaborativeComposerProvider::new(
            project,
            composer.into(),
            move |cx| {
                submit_editor
                    .update(cx, |message_editor, cx| {
                        if message_editor.is_empty(cx) {
                            return Err(CollaborativeComposerActionError::EmptyInput);
                        }
                        message_editor.send(cx);
                        Ok(())
                    })
                    .map_err(|_| CollaborativeComposerActionError::ThreadUnavailable)?
            },
            move |cx| {
                cancel_editor
                    .update(cx, |_, cx| cx.emit(MessageEditorEvent::Cancel))
                    .map_err(|_| CollaborativeComposerActionError::ThreadUnavailable)
            },
        )
        .with_focus_handle(focus_handle)
        .with_input_in_timeline()
        .with_footer(footer.into());
        Ok(Self {
            thread_view_id: thread_view.entity_id(),
            provider,
        })
    }

    pub fn thread_view_id(&self) -> EntityId {
        self.thread_view_id
    }

    pub fn register_in_workspace(
        self,
        workspace: &mut Workspace,
        cx: &mut gpui::Context<Workspace>,
    ) -> Result<CollaborativeComposerRegistration, CollaborativeComposerAdapterError> {
        workspace
            .register_collaborative_composer_provider(self.provider, cx)
            .map_err(CollaborativeComposerAdapterError::Registration)
    }
}

struct CollaborativeAcpComposer {
    thread_view: Entity<ThreadView>,
    _thread_view_observation: Subscription,
}

struct CollaborativeAcpComposerFooter {
    thread_view: Entity<ThreadView>,
    _thread_view_observation: Subscription,
}

impl Render for CollaborativeAcpComposerFooter {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.thread_view.update(cx, |thread_view, cx| {
            thread_view.render_collaborative_composer_footer(window, cx)
        })
    }
}

impl CollaborativeAcpComposer {
    fn new(thread_view: Entity<ThreadView>, cx: &mut Context<Self>) -> Self {
        let observation = cx.observe(&thread_view, |_, _, cx| cx.notify());
        Self {
            thread_view,
            _thread_view_observation: observation,
        }
    }
}

impl Render for CollaborativeAcpComposer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("collaborative-native-acp-composer")
            .debug_selector(|| "COLLABORATIVE-NATIVE-ACP-COMPOSER".to_owned())
            .w_full()
            .max_h(vh(0.5, window))
            .overflow_y_scroll()
            .child(self.thread_view.update(cx, |thread_view, cx| {
                thread_view.render_collaborative_composer(window, cx)
            }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeComposerAdapterError {
    ThreadUnavailable,
    ProjectUnavailable,
    ProjectMismatch,
    Registration(CollaborativeComposerRegistrationError),
}

impl fmt::Display for CollaborativeComposerAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadUnavailable => formatter.write_str("active agent thread is unavailable"),
            Self::ProjectUnavailable => formatter.write_str("active agent project is unavailable"),
            Self::ProjectMismatch => {
                formatter.write_str("active agent thread belongs to a different project")
            }
            Self::Registration(error) => write!(formatter, "composer registration failed: {error}"),
        }
    }
}

impl Error for CollaborativeComposerAdapterError {}

#[cfg(test)]
mod tests {
    use acp_thread::{PermissionOptions, StubAgentConnection, ThreadStatus, ToolCallStatus};
    use agent_client_protocol::schema::v1 as acp;
    use collections::HashMap;
    use fs::FakeFs;
    use gpui::{TestAppContext, VisualTestContext, px, size};
    use project::Project;
    use workspace::MultiWorkspace;

    use crate::thread_metadata_store::ThreadMetadataStore;

    use super::*;

    #[gpui::test]
    async fn collaborative_thread_activation_focuses_input_without_requesting_agent_dock(
        cx: &mut TestAppContext,
    ) {
        crate::test_support::init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            ThreadMetadataStore::init_global(cx);
        });
        let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("test workspace");
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        cx.simulate_resize(size(px(1200.), px(700.)));
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        crate::test_support::open_thread_with_connection(&panel, StubAgentConnection::new(), cx);
        let adapter = cx
            .update(|_, cx| CollaborativeComposerAdapter::from_agent_panel(&panel, &workspace, cx))
            .expect("canonical active thread composer");
        workspace
            .update(cx, |workspace, cx| {
                adapter.register_in_workspace(workspace, cx)
            })
            .expect("canonical composer registration");
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.close_all_docks(window, cx);
            AgentPanel::reveal_conversation(workspace, true, window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| {
            window.simulate_next_frame(cx);
            window.simulate_next_frame(cx);
        });
        workspace.update_in(cx, |workspace, window, cx| {
            let focus = workspace
                .collaborative_composer()
                .focus_handle()
                .expect("registered native message editor");
            assert!(focus.is_focused(window));
            for position in [
                workspace::dock::DockPosition::Left,
                workspace::dock::DockPosition::Bottom,
                workspace::dock::DockPosition::Right,
            ] {
                assert!(!workspace.is_dock_at_position_open(position, cx));
            }
        });
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER").is_some());
    }

    #[gpui::test]
    async fn collaborative_native_no_model_keyboard_submit_preserves_closed_docks(
        cx: &mut TestAppContext,
    ) {
        crate::test_support::init_test(cx);
        cx.update(|cx| {
            prompt_store::init(cx);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.set_default_model(None, cx);
            });
            ThreadMetadataStore::init_global(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", serde_json::json!({ "main.rs": "fn main() {}" }))
            .await;
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        let project = Project::test(fs, [std::path::Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("test workspace");
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        cx.simulate_resize(size(px(1200.), px(800.)));
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel.update(cx, |panel, cx| {
                panel.activate_new_thread(false, crate::AgentThreadSource::AgentPanel, window, cx);
            });
            panel
        });
        cx.run_until_parked();
        let thread_view = panel.read_with(cx, |panel, cx| {
            panel.active_thread_view(cx).expect("native draft thread")
        });
        let message_editor = thread_view.read_with(cx, |view, _| view.message_editor.clone());
        thread_view.read_with(cx, |view, cx| {
            assert!(
                view.thread
                    .read(cx)
                    .connection()
                    .clone()
                    .downcast::<agent::NativeAgentConnection>()
                    .is_some()
            );
        });
        thread_view.read_with(cx, |view, cx| {
            let connection = view
                .thread
                .read(cx)
                .connection()
                .clone()
                .downcast::<agent::NativeAgentConnection>()
                .expect("real native connection");
            let thread = connection
                .thread(view.thread.read(cx).session_id(), cx)
                .expect("real native session");
            assert!(thread.read(cx).model().is_none(), "no configured model");
        });
        cx.update(|_, cx| {
            cx.bind_keys([
                gpui::KeyBinding::new(
                    "cmd-enter",
                    crate::ChatWithFollow,
                    Some("AcpThread > Editor"),
                ),
                gpui::KeyBinding::new(
                    "ctrl-enter",
                    zed_actions::agent::Chat,
                    Some("AcpThread > Editor"),
                ),
            ]);
        });
        let composer = cx.update(|_, cx| {
            CollaborativeComposerAdapter::from_agent_panel(&panel, &workspace, cx)
                .expect("canonical native composer")
        });
        let timeline = cx.update(|_, cx| {
            crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                &panel, &workspace, cx,
            )
            .expect("canonical native timeline")
        });
        workspace.update(cx, |workspace, cx| {
            composer
                .register_in_workspace(workspace, cx)
                .expect("composer registration");
            timeline
                .register_in_workspace(workspace, cx)
                .expect("timeline registration");
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        cx.update(|window, cx| {
            window.simulate_next_frame(cx);
            window.simulate_next_frame(cx);
        });
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER").is_none());
        let input = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
            .expect("empty inline input");
        let conversation = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
            .expect("empty conversation");
        assert!(conversation.contains(&input.origin));
        assert!(input.bottom() < conversation.bottom() - px(100.));
        assert!(
            cx.debug_bounds("COLLABORATIVE-INPUT-AUTHOR-IMAGE")
                .is_none()
        );
        workspace.update_in(cx, |workspace, window, cx| {
            AgentPanel::reveal_conversation(workspace, true, window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| {
            window.simulate_next_frame(cx);
            window.simulate_next_frame(cx);
            assert!(message_editor.read(cx).focus_handle(cx).is_focused(window));
        });
        let user_store = thread_view.read_with(cx, |view, cx| {
            view.thread.read(cx).project().read(cx).user_store()
        });
        user_store.update(cx, |store, cx| {
            store.set_current_user_for_tests(
                std::sync::Arc::new(client::User {
                    legacy_id: 1,
                    username: "disposable-local-author".into(),
                    avatar_uri: "https://author-fixture.invalid/avatar.png".into(),
                    name: Some("Disposable Local Author".to_owned()),
                }),
                cx,
            );
        });
        cx.run_until_parked();
        cx.update(|window, cx| window.simulate_next_frame(cx));
        assert!(
            cx.debug_bounds("COLLABORATIVE-INPUT-AUTHOR-IMAGE")
                .is_some()
        );
        for width in [700., 1200.] {
            cx.simulate_resize(size(px(width), px(800.)));
            message_editor.update_in(cx, |editor, window, cx| {
                editor.set_text("First draft line\nSecond draft line", window, cx);
            });
            cx.run_until_parked();
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.simulate_next_frame(cx);
            });
            let author = cx
                .debug_bounds("COLLABORATIVE-INPUT-AUTHOR")
                .expect("inline draft author");
            let editor = cx
                .debug_bounds("ACP-MESSAGE-EDITOR-CONTENT")
                .expect("canonical editor beside author");
            assert!(author.right() < editor.left());
            assert!(author.top() < editor.top() + px(12.));
            assert!(editor.size.width > px(100.));
            thread_view.read_with(cx, |view, cx| {
                assert_eq!(view.message_editor.entity_id(), message_editor.entity_id());
                assert_eq!(
                    view.message_editor.read(cx).text(cx),
                    "First draft line\nSecond draft line"
                );
            });
            cx.update(|window, cx| {
                assert!(message_editor.read(cx).focus_handle(cx).is_focused(window));
            });
        }
        for (key, entry_count) in [("cmd-enter", 1), ("ctrl-enter", 2)] {
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.close_all_docks(window, cx);
                AgentPanel::reveal_conversation(workspace, true, window, cx);
            });
            cx.run_until_parked();
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.simulate_next_frame(cx);
            });
            message_editor.update_in(cx, |editor, window, cx| {
                editor.set_text("Disposable no-model keyboard submission", window, cx);
            });
            cx.simulate_keystrokes(key);
            cx.run_until_parked();
            let thread = thread_view.read_with(cx, |view, _| view.thread.clone());
            cx.condition(&thread, |thread, _| thread.status() == ThreadStatus::Idle)
                .await;
            thread_view.read_with(cx, |view, cx| {
                assert_eq!(view.thread.read(cx).status(), ThreadStatus::Idle);
                assert_eq!(
                    view.thread.read(cx).execution_outcome(),
                    acp_thread::ThreadExecutionOutcome::Failed
                );
                assert_eq!(view.thread.read(cx).entries().len(), entry_count);
                assert_eq!(view.message_editor.entity_id(), message_editor.entity_id());
                assert_eq!(view.list_state.item_count(), entry_count + 1);
                assert!(!view.generating_indicator_in_list);
                let input = view
                    .list_state
                    .bounds_for_item(entry_count)
                    .expect("input follows actual messages");
                let previous = view
                    .list_state
                    .bounds_for_item(entry_count - 1)
                    .expect("actual last message");
                assert!(input.top() >= previous.bottom());
            });
            workspace.update(cx, |workspace, cx| {
                for position in [
                    workspace::dock::DockPosition::Left,
                    workspace::dock::DockPosition::Bottom,
                    workspace::dock::DockPosition::Right,
                ] {
                    assert!(!workspace.is_dock_at_position_open(position, cx));
                }
            });
            assert!(
                cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
                    .is_some()
            );
        }
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("Retained inline input across Editor switch", window, cx);
        });
        cx.dispatch_action(workspace::SwitchToEditorWorkspace);
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| window.simulate_next_frame(cx));
        thread_view.read_with(cx, |view, cx| {
            assert!(!view.collaborative_input_in_list);
            assert_eq!(
                view.list_state.item_count(),
                view.thread.read(cx).entries().len()
            );
            assert_eq!(view.message_editor.entity_id(), message_editor.entity_id());
        });
        assert!(
            cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
                .is_none()
        );
        assert!(cx.debug_bounds("COLLABORATIVE-INPUT-AUTHOR").is_none());
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.close_all_docks(window, cx)
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        cx.update(|window, cx| window.simulate_next_frame(cx));
        thread_view.read_with(cx, |view, cx| {
            assert!(view.collaborative_input_in_list);
            assert_eq!(
                view.list_state.item_count(),
                view.thread.read(cx).entries().len() + 1
            );
            assert_eq!(view.message_editor.entity_id(), message_editor.entity_id());
            assert_eq!(
                view.message_editor.read(cx).text(cx),
                "Retained inline input across Editor switch"
            );
        });
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER").is_none());
        assert!(
            cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
                .is_some()
        );
    }

    #[gpui::test]
    async fn collaborative_composer_routes_submit_and_cancel_to_the_active_acp_thread(
        cx: &mut TestAppContext,
    ) {
        crate::test_support::init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            ThreadMetadataStore::init_global(cx);
        });

        let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .expect("test workspace should exist");
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        cx.simulate_resize(size(px(900.), px(700.)));

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<AgentPanel>(window, cx);
            panel
        });
        let tool_call_id = acp::ToolCallId::new("collaborative-permission");
        let denied_tool_call_id = acp::ToolCallId::new("collaborative-denial");
        let connection = StubAgentConnection::new().with_permission_requests(HashMap::from_iter([
            (
                tool_call_id.clone(),
                PermissionOptions::Flat(vec![acp::PermissionOption::new(
                    "allow",
                    "Allow audit action",
                    acp::PermissionOptionKind::AllowOnce,
                )]),
            ),
            (
                denied_tool_call_id.clone(),
                PermissionOptions::Flat(vec![acp::PermissionOption::new(
                    "deny",
                    "Deny audit action",
                    acp::PermissionOptionKind::RejectOnce,
                )]),
            ),
        ]));
        crate::test_support::open_thread_with_connection(&panel, connection.clone(), cx);

        let adapter = cx
            .update(|_, cx| CollaborativeComposerAdapter::from_agent_panel(&panel, &workspace, cx));
        let adapter = adapter.expect("active ACP composer should adapt");
        workspace
            .update(cx, |workspace, cx| {
                adapter.register_in_workspace(workspace, cx)
            })
            .expect("active ACP composer should register");
        let timeline = cx
            .update(|_, cx| {
                crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                    &panel, &workspace, cx,
                )
            })
            .expect("active ACP timeline should adapt");
        workspace
            .update(cx, |workspace, cx| {
                timeline.register_in_workspace(workspace, cx)
            })
            .expect("active ACP timeline should register");
        assert_eq!(
            workspace.update(cx, |workspace, cx| {
                workspace.submit_collaborative_composer(cx)
            }),
            Err(CollaborativeComposerActionError::EmptyInput)
        );

        let thread_view = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("active thread view should exist")
        });
        let message_editor =
            thread_view.read_with(cx, |thread_view, _| thread_view.message_editor.clone());
        let message_editor_focus =
            message_editor.read_with(cx, |message_editor, cx| message_editor.focus_handle(cx));
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        cx.dispatch_action(workspace::collaborative_focus::RestoreCollaborativeFocus);
        cx.dispatch_action(workspace::collaborative_focus::FocusNextCollaborativeRegion);
        cx.run_until_parked();
        assert!(cx.update(|window, _| message_editor_focus.is_focused(window)));
        cx.simulate_keystrokes("tab");
        assert!(
            !cx.update(|window, _| message_editor_focus.is_focused(window)),
            "Tab must leave the collaborative prompt for the next workspace region"
        );
        cx.simulate_keystrokes("shift-tab");
        assert!(cx.update(|window, _| message_editor_focus.is_focused(window)));
        cx.update(|_, cx| {
            cx.bind_keys([
                gpui::KeyBinding::new("f6", workspace::FocusNextPart, Some("Workspace")),
                gpui::KeyBinding::new("shift-f6", workspace::FocusPreviousPart, Some("Workspace")),
            ]);
        });
        cx.simulate_keystrokes("f6");
        assert!(!cx.update(|window, _| message_editor_focus.is_focused(window)));
        let hidden_editor_focus = workspace.read_with(cx, |workspace, cx| {
            workspace.active_pane().read(cx).focus_handle(cx)
        });
        assert!(
            !cx.update(|window, cx| hidden_editor_focus.contains_focused(window, cx)),
            "region navigation must not focus the hidden Editor workspace pane"
        );
        cx.simulate_keystrokes("shift-f6");
        assert!(
            cx.update(|window, _| message_editor_focus.is_focused(window)),
            "native region shortcuts must return to the visible collaborative composer"
        );
        message_editor.read_with(cx, |message_editor, cx| {
            assert!(matches!(
                message_editor.editor().read(cx).mode(),
                editor::EditorMode::AutoHeight { min_lines: 1, .. }
            ));
        });
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER").is_none());
        let composer_bounds = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
            .expect("same native input is mounted in the conversation");
        let conversation_bounds = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
            .expect("canonical conversation");
        assert!(conversation_bounds.contains(&composer_bounds.origin));
        assert!(composer_bounds.right() <= conversation_bounds.right());
        assert!(composer_bounds.bottom() <= conversation_bounds.bottom());
        assert!(composer_bounds.bottom() < conversation_bounds.bottom() - px(100.));
        thread_view.read_with(cx, |view, _| {
            assert_eq!(view.list_state.item_count(), 1);
            let row = view
                .list_state
                .bounds_for_item(0)
                .expect("focusable input row");
            assert!(row.contains(&composer_bounds.origin));
            assert!(composer_bounds.bottom() <= row.bottom());
        });
        let send_bounds = cx
            .debug_bounds("ICON-Send")
            .expect("native send control must be mounted, even for an empty draft");
        let footer = cx
            .debug_bounds("WORKSPACE-STATUS-BAR")
            .expect("native composer controls share the status footer");
        assert!(footer.top() >= composer_bounds.bottom());
        assert!(footer.contains(&send_bounds.origin));
        assert!(send_bounds.right() <= footer.right() && send_bounds.bottom() <= footer.bottom());
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.status_bar.get_or_insert_default().show = Some(false);
                });
            });
        });
        cx.run_until_parked();
        assert!(cx.debug_bounds("WORKSPACE-STATUS-BAR").is_none());
        let footer = cx
            .debug_bounds("COLLABORATIVE-COMPOSER-FOOTER")
            .expect("hiding workspace status must retain native send and model controls");
        let send = cx
            .debug_bounds("ICON-Send")
            .expect("send with hidden status");
        assert!(footer.contains(&send.center()));

        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("Review the project", window, cx);
        });
        cx.run_until_parked();
        let send_bounds = cx
            .debug_bounds("ICON-Send")
            .expect("send control should render");
        cx.simulate_click(send_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.status_bar.get_or_insert_default().show = Some(true);
                });
            });
        });
        cx.run_until_parked();

        let thread = panel.read_with(cx, |panel, cx| {
            panel
                .active_agent_thread(cx)
                .expect("active ACP thread should exist")
        });
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Generating
        );
        assert_eq!(thread.read_with(cx, |thread, _| thread.entries().len()), 1);

        let stop_bounds = cx
            .debug_bounds("ICON-Stop")
            .expect("generating thread must expose its native cancel control");
        cx.simulate_click(stop_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );
        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("Continue reviewing", window, cx);
            message_editor.send(cx);
            window.focus(&message_editor.focus_handle(cx), cx);
        });
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Generating
        );
        cx.dispatch_action(menu::Cancel);
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle,
            "keyboard cancellation must reach the same native ThreadView actions"
        );

        let multiline_prompt = "Review the first file\nThen review the second file";
        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text(multiline_prompt, window, cx);
        });
        let send = cx
            .debug_bounds("ICON-Send")
            .expect("send after cancellation");
        cx.simulate_click(send.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        let session_id = thread.read_with(cx, |thread, cx| {
            assert!(thread.to_markdown(cx).contains(multiline_prompt));
            assert_eq!(thread.status(), ThreadStatus::Generating);
            thread.session_id().clone()
        });
        for text in [
            "First streamed paragraph.\n\n",
            "Second streamed paragraph.",
        ] {
            cx.update(|_, cx| {
                connection.send_update(
                    session_id.clone(),
                    acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(text.into())),
                    cx,
                );
            });
            cx.run_until_parked();
            for _ in 0..32 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(16));
                cx.run_until_parked();
            }
            assert!(thread.read_with(cx, |thread, cx| {
                thread.to_markdown(cx).contains(text.trim())
            }));
            assert!(
                cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
                    .is_some()
            );
        }
        connection.end_turn(session_id, acp::StopReason::EndTurn);
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );

        connection.set_next_prompt_updates(vec![acp::SessionUpdate::ToolCall(
            acp::ToolCall::new(tool_call_id.clone(), "Audit action").kind(acp::ToolKind::Read),
        )]);
        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("Request the audit action", window, cx);
        });
        cx.run_until_parked();
        let send = cx.debug_bounds("ICON-Send").expect("send after streaming");
        cx.simulate_click(send.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert!(thread.read_with(cx, |thread, _| thread.entries().iter().any(|entry| {
            matches!(entry, acp_thread::AgentThreadEntry::ToolCall(call)
                if call.id == tool_call_id && matches!(call.status, ToolCallStatus::WaitingForConfirmation { .. }))
        })));
        let allow = cx
            .debug_bounds("ACP-PERMISSION-OPTIONS-collaborative-permission")
            .unwrap_or_else(|| {
                panic!(
                    "permission action must render in Collaborative presentation: {:?}",
                    cx.update(|window, _| window.debug_a11y_tree_json())
                )
            });
        cx.simulate_click(
            gpui::point(allow.left() + px(16.), allow.center().y),
            gpui::Modifiers::default(),
        );
        cx.run_until_parked();
        assert!(thread.read_with(cx, |thread, _| thread.entries().iter().all(|entry| {
            !matches!(entry, acp_thread::AgentThreadEntry::ToolCall(call)
                if call.id == tool_call_id && matches!(call.status, ToolCallStatus::WaitingForConfirmation { .. }))
        })));
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );
        assert!(matches!(
            connection.permission_outcome(&tool_call_id),
            Some(acp_thread::RequestPermissionOutcome::Selected(outcome))
                if outcome.option_kind == acp::PermissionOptionKind::AllowOnce
                    && outcome.option_id.0.as_ref() == "allow"
        ));

        connection.set_next_prompt_updates(vec![acp::SessionUpdate::ToolCall(
            acp::ToolCall::new(denied_tool_call_id.clone(), "Denied audit action")
                .kind(acp::ToolKind::Read),
        )]);
        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("Request the denied action", window, cx);
        });
        let send = cx
            .debug_bounds("ICON-Send")
            .expect("send after permission acceptance");
        cx.simulate_click(send.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        let deny = cx
            .debug_bounds("ACP-PERMISSION-OPTIONS-collaborative-denial")
            .expect("native denial option must render");
        cx.simulate_click(
            gpui::point(deny.left() + px(16.), deny.center().y),
            gpui::Modifiers::default(),
        );
        cx.run_until_parked();
        assert!(matches!(
            connection.permission_outcome(&denied_tool_call_id),
            Some(acp_thread::RequestPermissionOutcome::Selected(outcome))
                if outcome.option_kind == acp::PermissionOptionKind::RejectOnce
                    && outcome.option_id.0.as_ref() == "deny"
        ));
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );

        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("A turn whose transport will fail", window, cx);
        });
        let send = cx
            .debug_bounds("ICON-Send")
            .expect("send after permission denial");
        cx.simulate_click(send.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        let session_id = thread.read_with(cx, |thread, _| thread.session_id().clone());
        connection.fail_turn(&session_id, anyhow::anyhow!("Audit transport disconnected"));
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );
        let error = cx
            .debug_bounds("ACP-THREAD-ERROR")
            .expect("native error must render");
        let composer = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
            .expect("same inline input and error callout after failure");
        assert!(composer.contains(&error.origin));
        assert!(error.bottom() <= composer.bottom());
        assert!(thread_view.read_with(cx, |view, cx| {
            view.thread_error_markdown
                .as_ref()
                .expect("rendered native error message")
                .read(cx)
                .source()
                .contains("Audit transport disconnected")
        }));

        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.set_text("Recover after the transport failure", window, cx);
        });
        let send = cx
            .debug_bounds("ICON-Send")
            .expect("send after transport failure");
        cx.simulate_click(send.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Generating
        );
        assert!(cx.debug_bounds("ACP-THREAD-ERROR").is_none());
        connection.end_turn(session_id, acp::StopReason::EndTurn);
        cx.run_until_parked();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.status()),
            ThreadStatus::Idle
        );
        assert!(thread.read_with(cx, |thread, cx| {
            thread
                .to_markdown(cx)
                .contains("Recover after the transport failure")
        }));
    }
}
