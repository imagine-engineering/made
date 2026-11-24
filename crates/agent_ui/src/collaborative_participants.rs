use std::{error::Error, fmt, rc::Rc};

use acp_thread::ThreadStatus;
use gpui::{App, Entity, EntityId, SharedString, SharedUri, WeakEntity};
use project::Project;
use workspace::{
    Workspace,
    collaborative_participants::{
        CollaborativeConnectionState, CollaborativeExecutionLocation, CollaborativeExecutionPhase,
        CollaborativeExecutionStatus, CollaborativeParticipant, CollaborativeParticipantProvider,
        CollaborativeParticipantProviderState, CollaborativeParticipantViewData,
    },
};

use crate::{
    Agent, AgentPanel, conversation_view::ThreadView, thread_metadata_store::ThreadMetadataStore,
};

type RoomStateReader =
    Rc<dyn Fn(&App) -> (Vec<CollaborativeParticipant>, CollaborativeConnectionState)>;

pub struct CollaborativeParticipantAdapter {
    project: Entity<Project>,
    thread_view: WeakEntity<ThreadView>,
    thread_view_id: EntityId,
    room_state_reader: Option<RoomStateReader>,
}

impl CollaborativeParticipantAdapter {
    pub fn from_agent_panel(
        agent_panel: &Entity<AgentPanel>,
        workspace: &Entity<Workspace>,
        cx: &App,
    ) -> Result<Self, CollaborativeParticipantAdapterError> {
        let thread_view = agent_panel
            .read(cx)
            .active_thread_view(cx)
            .ok_or(CollaborativeParticipantAdapterError::ThreadUnavailable)?;
        let project = thread_view
            .read(cx)
            .project
            .upgrade()
            .ok_or(CollaborativeParticipantAdapterError::ProjectUnavailable)?;
        if project.entity_id() != workspace.read(cx).project().entity_id() {
            return Err(CollaborativeParticipantAdapterError::ProjectMismatch);
        }

        Ok(Self {
            project,
            thread_view: thread_view.downgrade(),
            thread_view_id: thread_view.entity_id(),
            room_state_reader: None,
        })
    }

    pub fn with_room_state_reader(
        mut self,
        reader: impl Fn(&App) -> (Vec<CollaborativeParticipant>, CollaborativeConnectionState) + 'static,
    ) -> Self {
        self.room_state_reader = Some(Rc::new(reader));
        self
    }

    pub fn thread_view_id(&self) -> EntityId {
        self.thread_view_id
    }

    pub fn state(&self, cx: &App) -> CollaborativeParticipantProviderState {
        match self.view_data(cx) {
            Ok(view_data) => CollaborativeParticipantProviderState::Ready(view_data),
            Err(CollaborativeParticipantAdapterError::ThreadUnavailable) => {
                CollaborativeParticipantProviderState::Unavailable
            }
            Err(error) => CollaborativeParticipantProviderState::failed(error.to_string()),
        }
    }

    pub fn into_provider(self) -> CollaborativeParticipantProvider {
        let project = self.project.clone();
        let source_id = self.thread_view_id;
        CollaborativeParticipantProvider::from_reader(project, source_id, move |cx| self.state(cx))
    }

    fn view_data(
        &self,
        cx: &App,
    ) -> Result<CollaborativeParticipantViewData, CollaborativeParticipantAdapterError> {
        let thread_view = self
            .thread_view
            .upgrade()
            .ok_or(CollaborativeParticipantAdapterError::ThreadUnavailable)?;
        let project = thread_view
            .read(cx)
            .project
            .upgrade()
            .ok_or(CollaborativeParticipantAdapterError::ProjectUnavailable)?;
        if project.entity_id() != self.project.entity_id() {
            return Err(CollaborativeParticipantAdapterError::ProjectMismatch);
        }

        let active_thread_id = Some(thread_view.read(cx).root_thread_id);
        let location = active_thread_id
            .and_then(|thread_id| {
                ThreadMetadataStore::try_global(cx).and_then(|store| {
                    store
                        .read(cx)
                        .entry(thread_id)
                        .map(|metadata| metadata.remote_connection.clone())
                })
            })
            .map(|remote_connection| match remote_connection {
                Some(remote_connection) => CollaborativeExecutionLocation::Remote(Some(
                    remote_connection.display_name().into(),
                )),
                None => CollaborativeExecutionLocation::Local,
            })
            .unwrap_or(CollaborativeExecutionLocation::Unknown);
        let task_title = active_thread_id
            .and_then(|thread_id| {
                ThreadMetadataStore::try_global(cx).and_then(|store| {
                    store
                        .read(cx)
                        .entry(thread_id)
                        .map(|metadata| metadata.display_title())
                })
            })
            .or_else(|| thread_view.read(cx).thread.read(cx).title())
            .or_else(|| Some(crate::DEFAULT_THREAD_TITLE.into()));

        let mut view_data = thread_view.read_with(cx, |thread_view, cx| {
            let agent_id = thread_view.agent_id.clone();
            let agent_server_store = project.read(cx).agent_server_store().clone();
            let display_name = agent_server_store
                .read(cx)
                .agent_display_name(&agent_id)
                .unwrap_or_else(|| Agent::from(agent_id.clone()).label());
            let avatar_uri = thread_view
                .agent_icon_from_external_svg
                .as_ref()
                .filter(|icon| icon.starts_with("https://") || icon.starts_with("http://"))
                .cloned()
                .map(SharedUri::from);
            let model = thread_view.current_model_id(cx).map(SharedString::from);
            let runtime = if thread_view.as_native_connection(cx).is_some() {
                agent::native_agent_name()
            } else {
                "ACP".into()
            };
            let thread = thread_view.thread.read(cx);
            let phase = if thread.is_waiting_for_confirmation() {
                CollaborativeExecutionPhase::WaitingForUser
            } else if thread.status() == ThreadStatus::Generating {
                CollaborativeExecutionPhase::Running
            } else if thread.had_error() {
                CollaborativeExecutionPhase::Failed
            } else {
                CollaborativeExecutionPhase::Idle
            };
            CollaborativeParticipantViewData {
                participants: vec![CollaborativeParticipant::agent(
                    agent_id.0,
                    display_name,
                    avatar_uri,
                    workspace::collaborative_participants::CollaborativeParticipantPresence::Online,
                )],
                execution: Some(CollaborativeExecutionStatus {
                    phase,
                    model,
                    runtime: Some(runtime),
                    location,
                }),
                task_title,
                connection: CollaborativeConnectionState::Disconnected,
            }
        });

        if let Some(room_state_reader) = &self.room_state_reader {
            let (mut participants, connection) = room_state_reader(cx);
            participants.append(&mut view_data.participants);
            view_data.participants = participants;
            view_data.connection = connection;
        }
        Ok(view_data)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeParticipantAdapterError {
    ThreadUnavailable,
    ProjectUnavailable,
    ProjectMismatch,
}

impl fmt::Display for CollaborativeParticipantAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadUnavailable => formatter.write_str("active agent thread is unavailable"),
            Self::ProjectUnavailable => formatter.write_str("active agent project is unavailable"),
            Self::ProjectMismatch => {
                formatter.write_str("active agent thread belongs to a different project")
            }
        }
    }
}

impl Error for CollaborativeParticipantAdapterError {}

#[cfg(test)]
mod tests {
    use acp_thread::StubAgentConnection;
    use fs::FakeFs;
    use gpui::{AppContext as _, TestAppContext, VisualTestContext, px, size};
    use project::Project;
    use workspace::MultiWorkspace;

    use crate::thread_metadata_store::ThreadMetadataStore;

    use super::*;

    #[gpui::test]
    async fn collaborative_participants_read_the_active_native_thread(cx: &mut TestAppContext) {
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
            panel
        });
        crate::test_support::open_thread_with_connection(&panel, StubAgentConnection::new(), cx);
        let thread_view = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("active thread view should exist")
        });

        let adapter = cx
            .update(|_, cx| {
                CollaborativeParticipantAdapter::from_agent_panel(&panel, &workspace, cx)
            })
            .expect("active native thread should adapt");
        assert_eq!(adapter.thread_view_id(), thread_view.entity_id());
        let registration = workspace
            .update(cx, |workspace, cx| {
                workspace.register_collaborative_participant_provider(adapter.into_provider(), cx)
            })
            .expect("native participant reader should register");

        workspace.read_with(cx, |workspace, cx| {
            let CollaborativeParticipantProviderState::Ready(view_data) =
                workspace.collaborative_participants().state(cx)
            else {
                panic!("active native thread should provide participant state")
            };
            assert_eq!(
                view_data
                    .execution
                    .as_ref()
                    .map(|execution| execution.phase),
                Some(CollaborativeExecutionPhase::Idle)
            );
            assert_eq!(
                view_data.task_title.as_deref(),
                Some(crate::DEFAULT_THREAD_TITLE)
            );
        });

        crate::test_support::send_message(&panel, cx);
        workspace.read_with(cx, |workspace, cx| {
            let CollaborativeParticipantProviderState::Ready(view_data) =
                workspace.collaborative_participants().state(cx)
            else {
                panic!("active native thread should remain registered")
            };
            assert_eq!(
                view_data
                    .execution
                    .as_ref()
                    .map(|execution| execution.phase),
                Some(CollaborativeExecutionPhase::Running)
            );
        });

        workspace.update(cx, |workspace, cx| {
            assert!(workspace.unregister_collaborative_participant_provider(registration, cx));
        });
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| {
                workspace.collaborative_participants().state(cx)
            }),
            CollaborativeParticipantProviderState::Unavailable
        );
    }
}
