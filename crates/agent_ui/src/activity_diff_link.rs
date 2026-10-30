use std::{collections::HashMap, error::Error, fmt, ops::Range};

use action_log::ActionLog;
use gpui::{Entity, EntityId};
use language::Point;
use project::ProjectPath;

use crate::activity_projection::ActivityLink;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivityDiffSourceIdentity {
    action_log_id: EntityId,
    project_diff_id: EntityId,
}

impl ActivityDiffSourceIdentity {
    pub fn new<DiffOwner: 'static>(
        action_log: &Entity<ActionLog>,
        diff_owner: &Entity<DiffOwner>,
    ) -> Self {
        Self::from_entity_ids(action_log.entity_id(), diff_owner.entity_id())
    }

    fn from_entity_ids(action_log_id: EntityId, project_diff_id: EntityId) -> Self {
        Self {
            action_log_id,
            project_diff_id,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ActivityDiffTargetId {
    repository_id: String,
    change_id: String,
    file_id: String,
    hunk_id: String,
}

impl ActivityDiffTargetId {
    pub fn new(
        repository_id: impl Into<String>,
        change_id: impl Into<String>,
        file_id: impl Into<String>,
        hunk_id: impl Into<String>,
    ) -> Result<Self, ActivityDiffLinkError> {
        let target = Self {
            repository_id: repository_id.into(),
            change_id: change_id.into(),
            file_id: file_id.into(),
            hunk_id: hunk_id.into(),
        };
        if target.repository_id.trim().is_empty()
            || target.change_id.trim().is_empty()
            || target.file_id.trim().is_empty()
            || target.hunk_id.trim().is_empty()
        {
            return Err(ActivityDiffLinkError::EmptyStableId);
        }
        Ok(target)
    }

    pub fn repository_id(&self) -> &str {
        &self.repository_id
    }

    pub fn change_id(&self) -> &str {
        &self.change_id
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn hunk_id(&self) -> &str {
        &self.hunk_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivityDiffTarget {
    id: ActivityDiffTargetId,
    project_path: ProjectPath,
    hunk_range: Range<Point>,
}

impl ActivityDiffTarget {
    pub fn new(
        id: ActivityDiffTargetId,
        project_path: ProjectPath,
        hunk_range: Range<Point>,
    ) -> Result<Self, ActivityDiffLinkError> {
        if hunk_range.start > hunk_range.end {
            return Err(ActivityDiffLinkError::InvalidHunkRange);
        }
        Ok(Self {
            id,
            project_path,
            hunk_range,
        })
    }

    pub fn id(&self) -> &ActivityDiffTargetId {
        &self.id
    }

    pub fn project_path(&self) -> &ProjectPath {
        &self.project_path
    }

    pub fn hunk_range(&self) -> &Range<Point> {
        &self.hunk_range
    }
}

pub struct ActivityDiffIndex {
    sources: ActivityDiffSourceIdentity,
    targets: HashMap<ActivityDiffTargetId, ActivityDiffTarget>,
}

impl ActivityDiffIndex {
    pub fn new<DiffOwner: 'static>(
        action_log: &Entity<ActionLog>,
        diff_owner: &Entity<DiffOwner>,
    ) -> Self {
        Self::for_sources(ActivityDiffSourceIdentity::new(action_log, diff_owner))
    }

    fn for_sources(sources: ActivityDiffSourceIdentity) -> Self {
        Self {
            sources,
            targets: HashMap::default(),
        }
    }

    pub fn insert(&mut self, target: ActivityDiffTarget) -> Result<(), ActivityDiffLinkError> {
        if self.targets.contains_key(target.id()) {
            return Err(ActivityDiffLinkError::DuplicateTarget);
        }
        self.targets.insert(target.id.clone(), target);
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum StableActivityLink {
    Action(String),
    GitChange {
        repository_id: String,
        change_id: String,
    },
}

impl StableActivityLink {
    fn from_activity_link(link: &ActivityLink) -> Result<Self, ActivityDiffLinkError> {
        let stable_link = match link {
            ActivityLink::Action { action_id } => Self::Action(action_id.clone()),
            ActivityLink::GitChange {
                repository_id,
                change_id,
            } => Self::GitChange {
                repository_id: repository_id.clone(),
                change_id: change_id.clone(),
            },
            ActivityLink::Entity { .. } => return Err(ActivityDiffLinkError::UnsupportedLink),
        };
        if stable_link.has_empty_id() {
            return Err(ActivityDiffLinkError::EmptyStableId);
        }
        Ok(stable_link)
    }

    fn has_empty_id(&self) -> bool {
        match self {
            Self::Action(action_id) => action_id.trim().is_empty(),
            Self::GitChange {
                repository_id,
                change_id,
            } => repository_id.trim().is_empty() || change_id.trim().is_empty(),
        }
    }
}

#[derive(Clone)]
struct ActivityDiffBinding {
    target_id: ActivityDiffTargetId,
    original_project_path: ProjectPath,
}

pub struct ActivityDiffLinkResolver {
    sources: ActivityDiffSourceIdentity,
    bindings: HashMap<StableActivityLink, ActivityDiffBinding>,
}

impl ActivityDiffLinkResolver {
    pub fn new<DiffOwner: 'static>(
        action_log: &Entity<ActionLog>,
        diff_owner: &Entity<DiffOwner>,
    ) -> Self {
        Self::for_sources(ActivityDiffSourceIdentity::new(action_log, diff_owner))
    }

    fn for_sources(sources: ActivityDiffSourceIdentity) -> Self {
        Self {
            sources,
            bindings: HashMap::default(),
        }
    }

    pub fn bind(
        &mut self,
        link: &ActivityLink,
        target_id: ActivityDiffTargetId,
        original_project_path: ProjectPath,
    ) -> Result<(), ActivityDiffLinkError> {
        let link = StableActivityLink::from_activity_link(link)?;
        if let StableActivityLink::GitChange {
            repository_id,
            change_id,
        } = &link
            && (repository_id != target_id.repository_id() || change_id != target_id.change_id())
        {
            return Err(ActivityDiffLinkError::ChangeIdentityMismatch);
        }
        if self.bindings.contains_key(&link) {
            return Err(ActivityDiffLinkError::DuplicateBinding);
        }
        self.bindings.insert(
            link,
            ActivityDiffBinding {
                target_id,
                original_project_path,
            },
        );
        Ok(())
    }

    pub fn resolve(
        &self,
        link: &ActivityLink,
        index: &ActivityDiffIndex,
    ) -> Result<ActivityDiffResolution, ActivityDiffLinkError> {
        if self.sources != index.sources {
            return Err(ActivityDiffLinkError::StaleSources);
        }
        let link = StableActivityLink::from_activity_link(link)?;
        let binding = self
            .bindings
            .get(&link)
            .ok_or(ActivityDiffLinkError::UnboundLink)?;
        if let Some(target) = index.targets.get(&binding.target_id) {
            if target.project_path() == &binding.original_project_path {
                return Ok(ActivityDiffResolution::Exact(target.clone()));
            }
            return Ok(ActivityDiffResolution::Moved {
                original_project_path: binding.original_project_path.clone(),
                target: target.clone(),
            });
        }

        let change_exists = index.targets.keys().any(|candidate| {
            candidate.repository_id() == binding.target_id.repository_id()
                && candidate.change_id() == binding.target_id.change_id()
        });
        if !change_exists {
            return Err(ActivityDiffLinkError::StaleChange);
        }
        let file_exists = index.targets.keys().any(|candidate| {
            candidate.repository_id() == binding.target_id.repository_id()
                && candidate.change_id() == binding.target_id.change_id()
                && candidate.file_id() == binding.target_id.file_id()
        });
        if !file_exists {
            return Err(ActivityDiffLinkError::StaleFile);
        }
        Err(ActivityDiffLinkError::MissingHunk)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActivityDiffResolution {
    Exact(ActivityDiffTarget),
    Moved {
        original_project_path: ProjectPath,
        target: ActivityDiffTarget,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityDiffLinkError {
    UnsupportedLink,
    EmptyStableId,
    InvalidHunkRange,
    DuplicateTarget,
    DuplicateBinding,
    ChangeIdentityMismatch,
    UnboundLink,
    StaleSources,
    StaleChange,
    StaleFile,
    MissingHunk,
}

impl fmt::Display for ActivityDiffLinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::UnsupportedLink => "activity link does not identify an action or Git change",
            Self::EmptyStableId => "activity diff identifiers must not be empty",
            Self::InvalidHunkRange => "activity diff hunk range is invalid",
            Self::DuplicateTarget => "activity diff target is duplicated",
            Self::DuplicateBinding => "activity diff link is already bound",
            Self::ChangeIdentityMismatch => "Git link and diff target identify different changes",
            Self::UnboundLink => "activity diff link has no canonical target",
            Self::StaleSources => "activity diff sources have been replaced",
            Self::StaleChange => "activity diff change is no longer available",
            Self::StaleFile => "activity diff file is no longer available",
            Self::MissingHunk => "activity diff hunk is no longer available",
        };
        formatter.write_str(message)
    }
}

impl Error for ActivityDiffLinkError {}

#[cfg(test)]
mod tests {
    use gpui::{AppContext as _, TestAppContext};
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn project_path(worktree_id: usize, path: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(worktree_id),
            path: RelPath::from_unix_str(path)
                .expect("test path should be relative")
                .into(),
        }
    }

    fn target_id(hunk_id: &str) -> ActivityDiffTargetId {
        ActivityDiffTargetId::new("repo-1", "change-1", "file-1", hunk_id)
            .expect("test target identifiers should be valid")
    }

    fn target(path: ProjectPath, hunk_id: &str) -> ActivityDiffTarget {
        ActivityDiffTarget::new(
            target_id(hunk_id),
            path,
            Point::new(10, 0)..Point::new(14, 0),
        )
        .expect("test target should be valid")
    }

    #[gpui::test]
    fn activity_diff_link(cx: &mut TestAppContext) {
        let (sources, stale_sources) = cx.update(|cx| {
            let action_log = cx.new(|_| ());
            let project_diff = cx.new(|_| ());
            let replacement_project_diff = cx.new(|_| ());
            (
                ActivityDiffSourceIdentity::from_entity_ids(
                    action_log.entity_id(),
                    project_diff.entity_id(),
                ),
                ActivityDiffSourceIdentity::from_entity_ids(
                    action_log.entity_id(),
                    replacement_project_diff.entity_id(),
                ),
            )
        });
        let original_path = project_path(1, "src/old.rs");
        let action_link = ActivityLink::Action {
            action_id: "session-1/tool/edit-1".into(),
        };
        let git_link = ActivityLink::GitChange {
            repository_id: "repo-1".into(),
            change_id: "change-1".into(),
        };
        let mut resolver = ActivityDiffLinkResolver::for_sources(sources);
        resolver
            .bind(&action_link, target_id("hunk-1"), original_path.clone())
            .expect("action should bind to its canonical hunk");
        resolver
            .bind(&git_link, target_id("hunk-1"), original_path.clone())
            .expect("Git change should bind to its canonical hunk");

        let mut exact_index = ActivityDiffIndex::for_sources(sources);
        exact_index
            .insert(target(original_path.clone(), "hunk-1"))
            .expect("exact target should index");
        assert!(matches!(
            resolver.resolve(&action_link, &exact_index),
            Ok(ActivityDiffResolution::Exact(_))
        ));
        assert!(matches!(
            resolver.resolve(&git_link, &exact_index),
            Ok(ActivityDiffResolution::Exact(_))
        ));

        let moved_path = project_path(1, "src/new.rs");
        let mut moved_index = ActivityDiffIndex::for_sources(sources);
        moved_index
            .insert(target(moved_path.clone(), "hunk-1"))
            .expect("moved target should index");
        let ActivityDiffResolution::Moved {
            original_project_path,
            target: moved_target,
        } = resolver
            .resolve(&action_link, &moved_index)
            .expect("stable file identity should follow a move")
        else {
            panic!("moved path should be reported")
        };
        assert_eq!(original_project_path, original_path);
        assert_eq!(moved_target.project_path(), &moved_path);

        let stale_index = ActivityDiffIndex::for_sources(stale_sources);
        assert_eq!(
            resolver.resolve(&action_link, &stale_index),
            Err(ActivityDiffLinkError::StaleSources)
        );

        let empty_index = ActivityDiffIndex::for_sources(sources);
        assert_eq!(
            resolver.resolve(&action_link, &empty_index),
            Err(ActivityDiffLinkError::StaleChange)
        );

        let mut missing_hunk_index = ActivityDiffIndex::for_sources(sources);
        missing_hunk_index
            .insert(target(original_path, "replacement-hunk"))
            .expect("replacement hunk should index");
        assert_eq!(
            resolver.resolve(&action_link, &missing_hunk_index),
            Err(ActivityDiffLinkError::MissingHunk)
        );
    }
}
