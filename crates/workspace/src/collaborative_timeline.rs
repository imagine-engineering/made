use std::{error::Error, fmt};

use gpui::{AnyView, Entity, EntityId};
use project::Project;

#[derive(Clone)]
pub struct CollaborativeTimelineProvider {
    project: Entity<Project>,
    view: AnyView,
}

impl CollaborativeTimelineProvider {
    pub fn new(project: Entity<Project>, view: AnyView) -> Self {
        Self { project, view }
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn view(&self) -> &AnyView {
        &self.view
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeTimelineRegistration {
    provider_id: EntityId,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeTimelineRegistrationError {
    ProjectMismatch,
    ProviderOccupied,
    RegistrationExhausted,
}

impl fmt::Display for CollaborativeTimelineRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectMismatch => {
                formatter.write_str("timeline provider belongs to a different project")
            }
            Self::ProviderOccupied => formatter.write_str("timeline provider is already occupied"),
            Self::RegistrationExhausted => {
                formatter.write_str("timeline registration generation is exhausted")
            }
        }
    }
}

impl Error for CollaborativeTimelineRegistrationError {}

pub struct CollaborativeTimelineHost {
    project: Entity<Project>,
    provider: Option<CollaborativeTimelineProvider>,
    provider_generation: Option<u64>,
    next_generation: u64,
}

impl CollaborativeTimelineHost {
    pub fn new(project: Entity<Project>) -> Self {
        Self {
            project,
            provider: None,
            provider_generation: None,
            next_generation: 0,
        }
    }

    pub fn register(
        &mut self,
        provider: CollaborativeTimelineProvider,
    ) -> Result<CollaborativeTimelineRegistration, CollaborativeTimelineRegistrationError> {
        if provider.project.entity_id() != self.project.entity_id() {
            return Err(CollaborativeTimelineRegistrationError::ProjectMismatch);
        }
        if self.provider.is_some() {
            return Err(CollaborativeTimelineRegistrationError::ProviderOccupied);
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CollaborativeTimelineRegistrationError::RegistrationExhausted)?;
        self.next_generation = generation;
        let registration = CollaborativeTimelineRegistration {
            provider_id: provider.view.entity_id(),
            generation,
        };
        self.provider = Some(provider);
        self.provider_generation = Some(generation);
        Ok(registration)
    }

    pub fn unregister(&mut self, registration: CollaborativeTimelineRegistration) -> bool {
        if !self
            .provider
            .as_ref()
            .is_some_and(|provider| provider.view.entity_id() == registration.provider_id)
            || self.provider_generation != Some(registration.generation)
        {
            return false;
        }
        self.provider = None;
        self.provider_generation = None;
        true
    }

    pub fn view(&self) -> Option<AnyView> {
        self.provider.as_ref().map(|provider| provider.view.clone())
    }
}
