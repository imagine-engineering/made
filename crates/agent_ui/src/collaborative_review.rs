use std::{error::Error, fmt};

use acp_thread::AcpThread;
use gpui::{App, AppContext as _, Entity, EntityId, Window};
use project::Project;
use workspace::{
    Workspace,
    collaborative_review::{
        CollaborativeReviewRegistration, CollaborativeReviewRegistrationError,
        CollaborativeReviewSlot,
    },
};

use crate::AgentDiffPane;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborativeAgentReviewError {
    ThreadUnavailable,
    StaleProject,
    StaleActionLog,
    Registration(CollaborativeReviewRegistrationError),
}

impl fmt::Display for CollaborativeAgentReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadUnavailable => formatter.write_str("no active agent thread is available"),
            Self::StaleProject => {
                formatter.write_str("the agent review belongs to another project")
            }
            Self::StaleActionLog => {
                formatter.write_str("the agent review action log is no longer current")
            }
            Self::Registration(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl Error for CollaborativeAgentReviewError {}

pub struct CollaborativeAgentReviewAdapter {
    project: Entity<Project>,
    thread: Entity<AcpThread>,
    action_log_id: EntityId,
    pane: Entity<AgentDiffPane>,
}

impl CollaborativeAgentReviewAdapter {
    pub fn new(
        thread: Option<Entity<AcpThread>>,
        workspace: &Entity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<Self, CollaborativeAgentReviewError> {
        let thread = thread.ok_or(CollaborativeAgentReviewError::ThreadUnavailable)?;
        let project = thread.read(cx).project().clone();
        if project.entity_id() != workspace.read(cx).project().entity_id() {
            return Err(CollaborativeAgentReviewError::StaleProject);
        }

        let action_log_id = thread.read(cx).action_log().entity_id();
        let pane =
            cx.new(|cx| AgentDiffPane::new(thread.clone(), workspace.downgrade(), window, cx));
        Ok(Self {
            project,
            thread,
            action_log_id,
            pane,
        })
    }

    pub fn pane(&self) -> &Entity<AgentDiffPane> {
        &self.pane
    }

    pub fn register_in_workspace(
        &self,
        workspace: &mut Workspace,
        cx: &mut gpui::Context<Workspace>,
    ) -> Result<CollaborativeReviewRegistration, CollaborativeAgentReviewError> {
        if self.project.entity_id() != workspace.project().entity_id()
            || self.thread.read(cx).project().entity_id() != self.project.entity_id()
        {
            return Err(CollaborativeAgentReviewError::StaleProject);
        }
        if self.thread.read(cx).action_log().entity_id() != self.action_log_id {
            return Err(CollaborativeAgentReviewError::StaleActionLog);
        }

        workspace
            .register_collaborative_review_provider(
                CollaborativeReviewSlot::AgentChanges,
                &self.project,
                self.pane.clone().into(),
                cx,
            )
            .map_err(CollaborativeAgentReviewError::Registration)
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, rc::Rc};

    use acp_thread::AgentConnection as _;
    use agent_client_protocol::schema::v1 as acp;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use project::Project;
    use workspace::{MultiWorkspace, PathList};

    use super::*;

    #[gpui::test]
    async fn collaborative_agent_review_adapter(cx: &mut TestAppContext) {
        crate::test_support::init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let other_project = Project::test(fs, [], cx).await;
        let connection = Rc::new(acp_thread::StubAgentConnection::new());
        let thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new::<&Path>(&[]), cx)
            })
            .await
            .expect("stub thread should start");

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());

        let unavailable = cx.update(|window, cx| {
            CollaborativeAgentReviewAdapter::new(None, &workspace, window, cx)
                .err()
                .expect("a missing active thread should stay unavailable")
        });
        assert_eq!(
            unavailable,
            CollaborativeAgentReviewError::ThreadUnavailable
        );

        let adapter = cx
            .update(|window, cx| {
                CollaborativeAgentReviewAdapter::new(Some(thread.clone()), &workspace, window, cx)
            })
            .expect("the canonical thread should adapt");
        let pane_id = adapter.pane().entity_id();
        workspace
            .update(cx, |workspace, cx| {
                adapter.register_in_workspace(workspace, cx)
            })
            .expect("the adapter should register in its canonical workspace");
        workspace.read_with(cx, |workspace, _| {
            assert_eq!(
                workspace
                    .collaborative_review()
                    .selected_view()
                    .expect("agent review should be selected")
                    .entity_id(),
                pane_id
            );
        });

        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_center(Box::new(adapter.pane().clone()), window, cx);
        });
        thread.update(cx, |thread, cx| {
            thread
                .handle_session_update(
                    acp::SessionUpdate::ToolCall(
                        acp::ToolCall::new("failed-review", "Review failure")
                            .status(acp::ToolCallStatus::Failed),
                    ),
                    cx,
                )
                .expect("controlled failed tool call");
        });
        for width in [400., 600.] {
            cx.simulate_resize(gpui::size(gpui::px(width), gpui::px(700.)));
            cx.run_until_parked();
            let message = cx
                .debug_bounds("AGENT-REVIEW-EMPTY-MESSAGE")
                .expect("failed review message");
            assert!(message.left() >= gpui::px(0.));
            assert!(message.right() <= gpui::px(width));
            if width == 400. {
                assert!(
                    message.size.height > gpui::px(20.),
                    "the error message should wrap in the narrow pane: {message:?}"
                );
            }
        }

        let (other_multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(other_project, window, cx));
        let other_workspace = other_multi_workspace
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        let stale = other_workspace
            .update(cx, |workspace, cx| {
                adapter.register_in_workspace(workspace, cx)
            })
            .expect_err("cross-project reuse should fail closed");
        assert_eq!(stale, CollaborativeAgentReviewError::StaleProject);
    }
}
