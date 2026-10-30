use std::{error::Error, fmt, rc::Rc};

use client::{LegacyUserId, User};
use gpui::{Entity, EntityId, SharedString, SharedUri};
use project::Project;

const UNKNOWN_MODEL_LABEL: &str = "Unknown model";
const UNKNOWN_RUNTIME_LABEL: &str = "Unknown runtime";
const UNKNOWN_LOCATION_LABEL: &str = "Unknown location";
const UNKNOWN_PARTICIPANT_LABEL: &str = "Unknown participant";
const PROVIDER_FAILURE_LABEL: &str = "Participant status unavailable";

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum CollaborativeParticipantIdentity {
    Human(LegacyUserId),
    Agent(SharedString),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeParticipantPresence {
    Online,
    Away,
    Busy,
    Offline,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborativeParticipant {
    pub identity: CollaborativeParticipantIdentity,
    pub display_name: SharedString,
    pub avatar_uri: Option<SharedUri>,
    pub presence: CollaborativeParticipantPresence,
}

impl CollaborativeParticipant {
    pub fn human(user: &User, presence: CollaborativeParticipantPresence) -> Self {
        let display_name = user
            .name
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .map(SharedString::from)
            .or_else(|| (!user.username.trim().is_empty()).then(|| user.username.clone()))
            .unwrap_or_else(|| UNKNOWN_PARTICIPANT_LABEL.into());
        let avatar_uri = (!user.avatar_uri.trim().is_empty()).then(|| user.avatar_uri.clone());
        Self {
            identity: CollaborativeParticipantIdentity::Human(user.legacy_id),
            display_name,
            avatar_uri,
            presence,
        }
    }

    pub fn agent(
        agent_id: impl Into<SharedString>,
        display_name: impl Into<SharedString>,
        avatar_uri: Option<SharedUri>,
        presence: CollaborativeParticipantPresence,
    ) -> Self {
        let agent_id = agent_id.into();
        let display_name = display_name.into();
        Self {
            identity: CollaborativeParticipantIdentity::Agent(agent_id),
            display_name: if display_name.trim().is_empty() {
                UNKNOWN_PARTICIPANT_LABEL.into()
            } else {
                display_name
            },
            avatar_uri,
            presence,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeExecutionPhase {
    Idle,
    Running,
    WaitingForUser,
    Failed,
    Completed,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeExecutionLocation {
    Local,
    Remote(Option<SharedString>),
    Unknown,
}

impl CollaborativeExecutionLocation {
    pub fn label(&self) -> SharedString {
        match self {
            Self::Local => "Local".into(),
            Self::Remote(Some(location)) if !location.trim().is_empty() => {
                format!("Remote · {location}").into()
            }
            Self::Remote(_) => "Remote".into(),
            Self::Unknown => UNKNOWN_LOCATION_LABEL.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborativeExecutionStatus {
    pub phase: CollaborativeExecutionPhase,
    pub model: Option<SharedString>,
    pub runtime: Option<SharedString>,
    pub location: CollaborativeExecutionLocation,
}

impl CollaborativeExecutionStatus {
    pub fn model_label(&self) -> SharedString {
        non_empty_label(self.model.as_ref(), UNKNOWN_MODEL_LABEL)
    }

    pub fn runtime_label(&self) -> SharedString {
        non_empty_label(self.runtime.as_ref(), UNKNOWN_RUNTIME_LABEL)
    }

    pub fn location_label(&self) -> SharedString {
        self.location.label()
    }
}

fn non_empty_label(value: Option<&SharedString>, unknown_label: &'static str) -> SharedString {
    value
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| unknown_label.into())
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CollaborativeParticipantViewData {
    pub participants: Vec<CollaborativeParticipant>,
    pub execution: Option<CollaborativeExecutionStatus>,
    pub task_title: Option<SharedString>,
    pub connection: CollaborativeConnectionState,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CollaborativeConnectionState {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Failed,
}

impl CollaborativeConnectionState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disconnected => "No active collaboration call",
            Self::Connecting => "Collaboration call reconnecting",
            Self::Connected => "Collaboration call connected",
            Self::Failed => "Collaboration call offline",
        }
    }

    pub fn supports_room_actions(self) -> bool {
        self == Self::Connected
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeParticipantProviderState {
    Ready(CollaborativeParticipantViewData),
    Failed(SharedString),
    Unavailable,
}

impl CollaborativeParticipantProviderState {
    pub fn failed(message: impl Into<SharedString>) -> Self {
        let message = message.into();
        Self::Failed(if message.trim().is_empty() {
            PROVIDER_FAILURE_LABEL.into()
        } else {
            message
        })
    }
}

type ParticipantStateReader = Rc<dyn Fn(&gpui::App) -> CollaborativeParticipantProviderState>;

#[derive(Clone)]
pub struct CollaborativeParticipantProvider {
    project: Entity<Project>,
    source_id: EntityId,
    state_reader: ParticipantStateReader,
}

impl CollaborativeParticipantProvider {
    pub fn new(
        project: Entity<Project>,
        source_id: EntityId,
        state: CollaborativeParticipantProviderState,
    ) -> Self {
        Self::from_reader(project, source_id, move |_| state.clone())
    }

    pub fn from_reader(
        project: Entity<Project>,
        source_id: EntityId,
        state_reader: impl Fn(&gpui::App) -> CollaborativeParticipantProviderState + 'static,
    ) -> Self {
        Self {
            project,
            source_id,
            state_reader: Rc::new(state_reader),
        }
    }

    pub fn state(&self, cx: &gpui::App) -> CollaborativeParticipantProviderState {
        (self.state_reader)(cx)
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeParticipantRegistration {
    source_id: EntityId,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeParticipantProviderError {
    ProjectMismatch,
    ProviderOccupied,
    StaleRegistration,
    RegistrationExhausted,
}

impl fmt::Display for CollaborativeParticipantProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectMismatch => {
                formatter.write_str("participant provider belongs to a different project")
            }
            Self::ProviderOccupied => formatter.write_str("participant provider is occupied"),
            Self::StaleRegistration => {
                formatter.write_str("participant provider registration is stale")
            }
            Self::RegistrationExhausted => {
                formatter.write_str("participant provider generation is exhausted")
            }
        }
    }
}

impl Error for CollaborativeParticipantProviderError {}

pub struct CollaborativeParticipantHost {
    project: Entity<Project>,
    provider: Option<CollaborativeParticipantProvider>,
    provider_generation: Option<u64>,
    next_generation: u64,
}

impl CollaborativeParticipantHost {
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
        provider: CollaborativeParticipantProvider,
    ) -> Result<CollaborativeParticipantRegistration, CollaborativeParticipantProviderError> {
        if provider.project.entity_id() != self.project.entity_id() {
            return Err(CollaborativeParticipantProviderError::ProjectMismatch);
        }
        if self.provider.is_some() {
            return Err(CollaborativeParticipantProviderError::ProviderOccupied);
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CollaborativeParticipantProviderError::RegistrationExhausted)?;
        self.next_generation = generation;
        let registration = CollaborativeParticipantRegistration {
            source_id: provider.source_id,
            generation,
        };
        self.provider = Some(provider);
        self.provider_generation = Some(generation);
        Ok(registration)
    }

    pub fn unregister(&mut self, registration: CollaborativeParticipantRegistration) -> bool {
        let is_current = self
            .provider
            .as_ref()
            .is_some_and(|provider| provider.source_id == registration.source_id)
            && self.provider_generation == Some(registration.generation);
        if !is_current {
            return false;
        }
        self.provider = None;
        self.provider_generation = None;
        true
    }

    pub fn state(&self, cx: &gpui::App) -> CollaborativeParticipantProviderState {
        self.provider
            .as_ref()
            .map(|provider| provider.state(cx))
            .unwrap_or(CollaborativeParticipantProviderState::Unavailable)
    }

    pub fn provider(&self) -> Option<CollaborativeParticipantProvider> {
        self.provider.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, path::Path};

    use fs::FakeFs;
    use gpui::{AppContext as _, Empty, TestAppContext};
    use settings::SettingsStore;

    use super::*;

    #[gpui::test]
    async fn collaborative_participant_view_data(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.set_global(db::AppDatabase::test_new());
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let file_system = FakeFs::new(cx.executor());
        let project = Project::test(file_system.clone(), [Path::new("/project")], cx).await;
        let other_project = Project::test(file_system, [Path::new("/other")], cx).await;
        let source = cx.new(|_| Empty);
        let replacement_source = cx.new(|_| Empty);
        let user = User {
            legacy_id: 42,
            username: "human-handle".into(),
            avatar_uri: "https://example.test/human.png".into(),
            name: Some("Human Name".to_owned()),
        };
        let human =
            CollaborativeParticipant::human(&user, CollaborativeParticipantPresence::Online);
        let agent = CollaborativeParticipant::agent(
            "agent:reviewer",
            "Reviewer",
            None,
            CollaborativeParticipantPresence::Busy,
        );
        assert_eq!(human.identity, CollaborativeParticipantIdentity::Human(42));
        assert_eq!(human.display_name.as_ref(), "Human Name");
        assert_eq!(
            agent.identity,
            CollaborativeParticipantIdentity::Agent("agent:reviewer".into())
        );

        let unknown_execution = CollaborativeExecutionStatus {
            phase: CollaborativeExecutionPhase::Unknown,
            model: None,
            runtime: Some("".into()),
            location: CollaborativeExecutionLocation::Unknown,
        };
        assert_eq!(
            unknown_execution.model_label().as_ref(),
            UNKNOWN_MODEL_LABEL
        );
        assert_eq!(
            unknown_execution.runtime_label().as_ref(),
            UNKNOWN_RUNTIME_LABEL
        );
        assert_eq!(
            unknown_execution.location_label().as_ref(),
            UNKNOWN_LOCATION_LABEL
        );

        let execution = CollaborativeExecutionStatus {
            phase: CollaborativeExecutionPhase::Running,
            model: Some("claude-sonnet".into()),
            runtime: Some("ACP".into()),
            location: CollaborativeExecutionLocation::Remote(Some("build-host".into())),
        };
        assert_eq!(execution.model_label().as_ref(), "claude-sonnet");
        assert_eq!(execution.runtime_label().as_ref(), "ACP");
        assert_eq!(execution.location_label().as_ref(), "Remote · build-host");

        let mut host = CollaborativeParticipantHost::new(project.clone());
        assert_eq!(
            cx.update(|cx| host.state(cx)),
            CollaborativeParticipantProviderState::Unavailable
        );
        let mismatch = CollaborativeParticipantProvider::new(
            other_project,
            source.entity_id(),
            CollaborativeParticipantProviderState::Unavailable,
        );
        assert_eq!(
            host.register(mismatch),
            Err(CollaborativeParticipantProviderError::ProjectMismatch)
        );

        let ready =
            CollaborativeParticipantProviderState::Ready(CollaborativeParticipantViewData {
                participants: vec![human, agent],
                execution: Some(execution),
                task_title: Some("Review changes".into()),
                connection: CollaborativeConnectionState::Connected,
            });
        let source_ready = Rc::new(Cell::new(true));
        let registration = host
            .register(CollaborativeParticipantProvider::from_reader(
                project.clone(),
                source.entity_id(),
                {
                    let source_ready = source_ready.clone();
                    let ready = ready.clone();
                    move |_| {
                        if source_ready.get() {
                            ready.clone()
                        } else {
                            CollaborativeParticipantProviderState::failed("")
                        }
                    }
                },
            ))
            .expect("canonical participant provider should register");
        assert_eq!(cx.update(|cx| host.state(cx)), ready);
        assert_eq!(
            host.register(CollaborativeParticipantProvider::new(
                project.clone(),
                replacement_source.entity_id(),
                CollaborativeParticipantProviderState::Unavailable,
            )),
            Err(CollaborativeParticipantProviderError::ProviderOccupied)
        );
        source_ready.set(false);
        assert_eq!(
            cx.update(|cx| host.state(cx)),
            CollaborativeParticipantProviderState::Failed(PROVIDER_FAILURE_LABEL.into())
        );
        assert!(host.unregister(registration));

        let replacement = host
            .register(CollaborativeParticipantProvider::new(
                project,
                replacement_source.entity_id(),
                CollaborativeParticipantProviderState::Unavailable,
            ))
            .expect("replacement provider should register");
        assert!(!host.unregister(registration));
        assert!(host.unregister(replacement));
    }
}
