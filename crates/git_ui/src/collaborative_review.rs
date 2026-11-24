use std::{error::Error, fmt};

use gpui::{App, AppContext as _, Entity, EntityId, Window};
use project::Project;
use workspace::{
    Workspace,
    collaborative_review::{
        CollaborativeReviewRegistration, CollaborativeReviewRegistrationError,
        CollaborativeReviewSlot,
    },
};

use crate::project_diff::ProjectDiff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborativeProjectReviewError {
    ProjectDiffUnavailable,
    StaleProject,
    StaleGitStore,
    Registration(CollaborativeReviewRegistrationError),
}

impl fmt::Display for CollaborativeProjectReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectDiffUnavailable => formatter.write_str("the project diff is unavailable"),
            Self::StaleProject => formatter.write_str("the review belongs to another project"),
            Self::StaleGitStore => formatter.write_str("the review Git store is no longer current"),
            Self::Registration(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl Error for CollaborativeProjectReviewError {}

pub struct CollaborativeProjectReviewAdapter {
    project: Entity<Project>,
    git_store_id: EntityId,
    project_diff: Entity<ProjectDiff>,
}

impl CollaborativeProjectReviewAdapter {
    pub fn from_workspace(
        workspace: &Entity<Workspace>,
        cx: &App,
    ) -> Result<Self, CollaborativeProjectReviewError> {
        let workspace = workspace.read(cx);
        let project = workspace.project().clone();
        let git_store_id = project.read(cx).git_store().entity_id();
        let project_diff = workspace
            .item_of_type::<ProjectDiff>(cx)
            .ok_or(CollaborativeProjectReviewError::ProjectDiffUnavailable)?;
        Ok(Self {
            project,
            git_store_id,
            project_diff,
        })
    }

    pub fn from_workspace_or_create(
        workspace: &Entity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<Self, CollaborativeProjectReviewError> {
        if let Ok(adapter) = Self::from_workspace(workspace, cx) {
            return Ok(adapter);
        }

        let project = workspace.read(cx).project().clone();
        let git_store_id = project.read(cx).git_store().entity_id();
        let project_diff =
            cx.new(|cx| ProjectDiff::new(project.clone(), workspace.clone(), window, cx));
        Ok(Self {
            project,
            git_store_id,
            project_diff,
        })
    }

    pub fn project_diff(&self) -> &Entity<ProjectDiff> {
        &self.project_diff
    }

    pub fn register_in_workspace(
        &self,
        workspace: &mut Workspace,
        cx: &mut gpui::Context<Workspace>,
    ) -> Result<CollaborativeReviewRegistration, CollaborativeProjectReviewError> {
        if self.project.entity_id() != workspace.project().entity_id() {
            return Err(CollaborativeProjectReviewError::StaleProject);
        }
        if self.project.read(cx).git_store().entity_id() != self.git_store_id {
            return Err(CollaborativeProjectReviewError::StaleGitStore);
        }

        let base = workspace.collaborative_project_review_base(cx);
        self.project_diff.update(cx, |project_diff, cx| {
            project_diff.use_collaborative_presentation(base, cx);
        });
        workspace
            .register_collaborative_review_provider(
                CollaborativeReviewSlot::ProjectChanges,
                &self.project,
                self.project_diff.clone().into(),
                cx,
            )
            .map_err(CollaborativeProjectReviewError::Registration)
    }
}

#[cfg(test)]
mod tests {
    use fs::FakeFs;
    use gpui::TestAppContext;
    use settings::SettingsStore;
    use workspace::MultiWorkspace;

    use super::*;

    #[gpui::test]
    async fn collaborative_project_review_reuses_the_native_project_diff(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.set_global(db::AppDatabase::test_new());
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());

        let adapter = cx
            .update(|window, cx| {
                CollaborativeProjectReviewAdapter::from_workspace_or_create(&workspace, window, cx)
            })
            .expect("native project diff should adapt");
        let project_diff_id = adapter.project_diff().entity_id();

        workspace.update(cx, |workspace, cx| {
            workspace.set_collaborative_project_review_base(
                project::git_store::diff_buffer_list::DiffBase::Staged,
                cx,
            );
        });
        workspace
            .update(cx, |workspace, cx| {
                adapter.register_in_workspace(workspace, cx)
            })
            .expect("native project diff should register");
        cx.run_until_parked();
        adapter.project_diff().read_with(cx, |diff, cx| {
            assert_eq!(
                diff.diff_base(cx),
                &project::git_store::diff_buffer_list::DiffBase::Staged
            );
        });

        workspace.read_with(cx, |workspace, _| {
            assert_eq!(
                workspace
                    .collaborative_review()
                    .selected_view()
                    .expect("project review should be selected")
                    .entity_id(),
                project_diff_id
            );
        });
    }
}
