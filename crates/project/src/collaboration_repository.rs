use std::{error::Error, fmt, path::Path};

use collaboration_domain::RepositoryCoordinate;
use gpui::App;

use crate::{
    Project,
    git_store::{RepositoryId, RepositorySnapshot, is_submodule_git_dir, repo_identity_path},
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CollaborationRepositoryIdentity(Box<Path>);

impl CollaborationRepositoryIdentity {
    pub fn from_repository(repository: &RepositorySnapshot) -> Self {
        let identity_path = if is_submodule_git_dir(&repository.repository_dir_abs_path) {
            repository.work_directory_abs_path.as_ref()
        } else {
            repo_identity_path(&repository.common_dir_abs_path)
        };
        Self(identity_path.into())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationRepositoryBinding {
    repository_identity: CollaborationRepositoryIdentity,
    hosted_coordinate: RepositoryCoordinate,
}

impl CollaborationRepositoryBinding {
    pub fn new(
        repository_identity: CollaborationRepositoryIdentity,
        hosted_coordinate: RepositoryCoordinate,
    ) -> Self {
        Self {
            repository_identity,
            hosted_coordinate,
        }
    }

    pub fn repository_identity(&self) -> &CollaborationRepositoryIdentity {
        &self.repository_identity
    }

    pub fn hosted_coordinate(&self) -> &RepositoryCoordinate {
        &self.hosted_coordinate
    }
}

impl Project {
    pub fn collaboration_repository_identity(
        &self,
        repository_id: RepositoryId,
        cx: &App,
    ) -> Result<CollaborationRepositoryIdentity, CollaborationRepositoryError> {
        let repository = self.repositories(cx).get(&repository_id).ok_or(
            CollaborationRepositoryError::RepositoryNotFound(repository_id),
        )?;
        Ok(CollaborationRepositoryIdentity::from_repository(
            repository.read(cx),
        ))
    }

    pub fn collaboration_repository_binding(
        &self,
        repository_id: RepositoryId,
        hosted_coordinate: RepositoryCoordinate,
        cx: &App,
    ) -> Result<CollaborationRepositoryBinding, CollaborationRepositoryError> {
        Ok(CollaborationRepositoryBinding::new(
            self.collaboration_repository_identity(repository_id, cx)?,
            hosted_coordinate,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationRepositoryError {
    RepositoryNotFound(RepositoryId),
}

impl fmt::Display for CollaborationRepositoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RepositoryNotFound(repository_id) => {
                write!(
                    formatter,
                    "repository {repository_id:?} is no longer available"
                )
            }
        }
    }
}

impl Error for CollaborationRepositoryError {}
