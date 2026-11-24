use std::{error::Error, fmt, path::PathBuf};

use collaboration_domain::{
    AggregateId, ChannelLifecycleState, CommunityId, ProjectChannelReference, ProjectGroup,
    ProjectGroupIdentity, RepositoryCoordinate,
};
use gpui::App;

use crate::{
    Project, ProjectGroupKey, WorktreeId,
    collaboration_repository::{CollaborationRepositoryBinding, CollaborationRepositoryIdentity},
    git_store::RepositoryId,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationProjectNavigationLifecycle {
    Active,
    Archived,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationChannelNavigationBinding {
    source_reference: ProjectChannelReference,
    community_id: CommunityId,
    channel_id: AggregateId,
    lifecycle_state: ChannelLifecycleState,
}

impl CollaborationChannelNavigationBinding {
    pub const fn new(
        source_reference: ProjectChannelReference,
        community_id: CommunityId,
        channel_id: AggregateId,
        lifecycle_state: ChannelLifecycleState,
    ) -> Self {
        Self {
            source_reference,
            community_id,
            channel_id,
            lifecycle_state,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborationChannelNavigationTarget {
    community_id: CommunityId,
    channel_id: AggregateId,
}

impl CollaborationChannelNavigationTarget {
    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationWorktreeNavigationTarget {
    worktree_id: WorktreeId,
    path: PathBuf,
}

impl CollaborationWorktreeNavigationTarget {
    pub const fn worktree_id(&self) -> WorktreeId {
        self.worktree_id
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationLocalRepositoryNavigationTarget {
    repository_id: RepositoryId,
    work_directory: PathBuf,
    worktrees: Vec<CollaborationWorktreeNavigationTarget>,
}

impl CollaborationLocalRepositoryNavigationTarget {
    pub const fn repository_id(&self) -> RepositoryId {
        self.repository_id
    }

    pub fn work_directory(&self) -> &PathBuf {
        &self.work_directory
    }

    pub fn worktrees(&self) -> &[CollaborationWorktreeNavigationTarget] {
        &self.worktrees
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationRepositoryNavigationTarget {
    hosted_coordinate: RepositoryCoordinate,
    local_targets: Vec<CollaborationLocalRepositoryNavigationTarget>,
}

impl CollaborationRepositoryNavigationTarget {
    pub const fn hosted_coordinate(&self) -> &RepositoryCoordinate {
        &self.hosted_coordinate
    }

    pub fn local_targets(&self) -> &[CollaborationLocalRepositoryNavigationTarget] {
        &self.local_targets
    }

    pub fn is_available(&self) -> bool {
        !self.local_targets.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationProjectNavigationTarget {
    project_identity: ProjectGroupIdentity,
    native_project: ProjectGroupKey,
    repositories: Vec<CollaborationRepositoryNavigationTarget>,
    channel: Option<CollaborationChannelNavigationTarget>,
}

impl CollaborationProjectNavigationTarget {
    pub const fn project_identity(&self) -> &ProjectGroupIdentity {
        &self.project_identity
    }

    pub const fn native_project(&self) -> &ProjectGroupKey {
        &self.native_project
    }

    pub fn repositories(&self) -> &[CollaborationRepositoryNavigationTarget] {
        &self.repositories
    }

    pub const fn channel(&self) -> Option<CollaborationChannelNavigationTarget> {
        self.channel
    }
}

impl Project {
    pub fn resolve_collaboration_navigation(
        &self,
        project_group: &ProjectGroup,
        lifecycle: CollaborationProjectNavigationLifecycle,
        repository_bindings: &[CollaborationRepositoryBinding],
        channel_binding: Option<&CollaborationChannelNavigationBinding>,
        cx: &App,
    ) -> Result<CollaborationProjectNavigationTarget, CollaborationNavigationResolutionError> {
        if lifecycle == CollaborationProjectNavigationLifecycle::Archived {
            return Err(CollaborationNavigationResolutionError::ArchivedProjectGroup);
        }

        for (index, binding) in repository_bindings.iter().enumerate() {
            if !project_group
                .fields()
                .repositories
                .iter()
                .any(|coordinate| same_repository(coordinate, binding.hosted_coordinate()))
            {
                return Err(CollaborationNavigationResolutionError::UnexpectedRepositoryBinding);
            }
            if repository_bindings[..index].iter().any(|existing| {
                same_repository(existing.hosted_coordinate(), binding.hosted_coordinate())
            }) {
                return Err(CollaborationNavigationResolutionError::DuplicateRepositoryBinding);
            }
        }

        let mut worktrees = self
            .visible_worktrees(cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                CollaborationWorktreeNavigationTarget {
                    worktree_id: worktree.id(),
                    path: worktree.abs_path().as_ref().to_path_buf(),
                }
            })
            .collect::<Vec<_>>();
        worktrees.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then_with(|| left.worktree_id.cmp(&right.worktree_id))
        });

        let mut repositories = Vec::with_capacity(project_group.fields().repositories.len());
        for coordinate in &project_group.fields().repositories {
            let mut local_targets = Vec::new();
            if let Some(binding) = repository_bindings
                .iter()
                .find(|binding| same_repository(coordinate, binding.hosted_coordinate()))
            {
                for (repository_id, repository) in self.repositories(cx) {
                    let repository = repository.read(cx);
                    if &CollaborationRepositoryIdentity::from_repository(&repository)
                        != binding.repository_identity()
                    {
                        continue;
                    }
                    let work_directory = repository.work_directory_abs_path.as_ref().to_path_buf();
                    local_targets.push(CollaborationLocalRepositoryNavigationTarget {
                        repository_id: *repository_id,
                        worktrees: worktrees
                            .iter()
                            .filter(|worktree| worktree.path == work_directory)
                            .cloned()
                            .collect(),
                        work_directory,
                    });
                }
            }
            local_targets.sort_by(|left, right| {
                left.work_directory
                    .cmp(&right.work_directory)
                    .then_with(|| left.repository_id.cmp(&right.repository_id))
            });
            repositories.push(CollaborationRepositoryNavigationTarget {
                hosted_coordinate: coordinate.clone(),
                local_targets,
            });
        }

        let channel = match (
            project_group.fields().channel_reference.as_ref(),
            channel_binding,
        ) {
            (None, None) | (Some(_), None) => None,
            (None, Some(_)) => {
                return Err(CollaborationNavigationResolutionError::UnexpectedChannelBinding);
            }
            (Some(reference), Some(binding)) => {
                if reference != &binding.source_reference {
                    return Err(CollaborationNavigationResolutionError::ChannelBindingMismatch);
                }
                (binding.lifecycle_state == ChannelLifecycleState::Active).then_some(
                    CollaborationChannelNavigationTarget {
                        community_id: binding.community_id,
                        channel_id: binding.channel_id,
                    },
                )
            }
        };

        Ok(CollaborationProjectNavigationTarget {
            project_identity: project_group.identity(),
            native_project: self.project_group_key(cx),
            repositories,
            channel,
        })
    }
}

fn same_repository(left: &RepositoryCoordinate, right: &RepositoryCoordinate) -> bool {
    left.owner_public_key() == right.owner_public_key()
        && left.discriminator() == right.discriminator()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationNavigationResolutionError {
    ArchivedProjectGroup,
    UnexpectedRepositoryBinding,
    DuplicateRepositoryBinding,
    UnexpectedChannelBinding,
    ChannelBindingMismatch,
}

impl fmt::Display for CollaborationNavigationResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArchivedProjectGroup => formatter.write_str("project group is archived"),
            Self::UnexpectedRepositoryBinding => {
                formatter.write_str("repository binding is not part of the signed project group")
            }
            Self::DuplicateRepositoryBinding => {
                formatter.write_str("repository binding is duplicated")
            }
            Self::UnexpectedChannelBinding => {
                formatter.write_str("project group has no signed channel reference")
            }
            Self::ChannelBindingMismatch => {
                formatter.write_str("channel binding does not match the signed project group")
            }
        }
    }
}

impl Error for CollaborationNavigationResolutionError {}
