use std::collections::BTreeMap;

use agent::{
    ManagedAgentCasOutcome, ManagedAgentInsertOutcome, ManagedAgentRepository,
    ManagedAgentRepositoryError, ProjectionWriteOutcome,
};
use agent_settings::{
    managed_agent::{
        EnvironmentReference, EnvironmentVariableName, ManagedAgentConfiguration,
        ManagedAgentState, ManagedAgentVersion, ModelId, PrivateManagedAgentRecord, ProviderId,
        RuntimeId,
    },
    team::{
        AgentIdentityRecord, AgentTeamMember, AgentTeamRecord, NostrEventId as SettingsEventId,
        NostrPublicKey as SettingsPublicKey, PersonaReference, PublicAgentCatalogRecord,
        PublicPersonaShareRecord, PublicTeamMemberShareRecord, PublicTeamShareRecord, TeamRole,
    },
};
use collaboration_domain::{
    NostrEventId as DomainEventId, NostrPublicKey as DomainPublicKey, OwnerAttestationEvidence,
    PrivateAgentCatalogProjectionSource, PrivateAgentProjectionState, PrivateAgentReference,
    PrivatePersonaProjectionSource, PrivateTeamMemberProjectionSource, PrivateTeamProjectionSource,
    project_public_agent_catalog,
};
use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, IntoElement, Render, SharedString, Window,
};
use ui::{Button, ButtonStyle, Color, Label, LabelSize, prelude::*};

#[derive(Clone)]
pub struct PersonaDraft {
    pub slug: String,
    pub display_name: String,
    pub description: String,
    pub system_prompt: Option<String>,
    pub avatar_url: Option<String>,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

#[derive(Clone)]
pub struct TeamMemberDraft {
    pub identity: AgentIdentityRecord,
    pub role: TeamRole,
    pub persona: PublicPersonaShareRecord,
}

#[derive(Clone)]
pub struct TeamDraft {
    pub team_id: String,
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub members: Vec<TeamMemberDraft>,
}

#[derive(Clone)]
pub struct ManagedAgentDraft {
    pub agent_public_key: String,
    pub event_id: String,
    pub runtime: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeAgentSettingsStatus {
    Idle,
    Saving,
    Saved(SharedString),
    Shared,
    Conflict(SharedString),
    RevokedOwner(SharedString),
    ValidationError(SharedString),
    Unavailable(SharedString),
}

impl CollaborativeAgentSettingsStatus {
    fn label(&self) -> Option<SharedString> {
        match self {
            Self::Idle => None,
            Self::Saving => Some("Saving agent settings…".into()),
            Self::Saved(message)
            | Self::Conflict(message)
            | Self::RevokedOwner(message)
            | Self::ValidationError(message)
            | Self::Unavailable(message) => Some(message.clone()),
            Self::Shared => Some("Public catalog projection is ready to publish.".into()),
        }
    }

    fn color(&self) -> Color {
        match self {
            Self::Conflict(_) | Self::RevokedOwner(_) => Color::Warning,
            Self::ValidationError(_) | Self::Unavailable(_) => Color::Error,
            Self::Idle | Self::Saving | Self::Saved(_) | Self::Shared => Color::Muted,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CollaborativeAgentSettingsEvent {
    CreatePersonaRequested,
    EditPersonaRequested(String),
    CreateTeamRequested,
    EditTeamRequested(String),
    CreateManagedAgentRequested,
    EditManagedAgentRequested(SettingsPublicKey),
    ShareRequested(SettingsPublicKey),
}

pub struct CollaborativeAgentSettings {
    repository: ManagedAgentRepository,
    owner_public_key: SettingsPublicKey,
    personas: Vec<PublicPersonaShareRecord>,
    teams: Vec<AgentTeamRecord>,
    public_teams: Vec<PublicTeamShareRecord>,
    managed_agents: BTreeMap<SettingsPublicKey, PrivateManagedAgentRecord>,
    status: CollaborativeAgentSettingsStatus,
    focus_handle: FocusHandle,
}

impl CollaborativeAgentSettings {
    pub fn new(
        repository: ManagedAgentRepository,
        owner_public_key: SettingsPublicKey,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            repository,
            owner_public_key,
            personas: Vec::new(),
            teams: Vec::new(),
            public_teams: Vec::new(),
            managed_agents: BTreeMap::new(),
            status: CollaborativeAgentSettingsStatus::Idle,
            focus_handle: cx.focus_handle(),
        }
    }

    pub fn status(&self) -> &CollaborativeAgentSettingsStatus {
        &self.status
    }

    pub fn persona_count(&self) -> usize {
        self.personas.len()
    }

    pub fn team_count(&self) -> usize {
        self.teams.len()
    }

    pub fn managed_agent_count(&self) -> usize {
        self.managed_agents.len()
    }

    pub fn managed_agent(
        &self,
        public_key: &SettingsPublicKey,
    ) -> Option<&PrivateManagedAgentRecord> {
        self.managed_agents.get(public_key)
    }

    pub fn create_persona(&mut self, draft: PersonaDraft, cx: &mut Context<Self>) {
        let record = PersonaReference::published(self.owner_public_key.clone(), draft.slug)
            .and_then(|source| {
                PublicPersonaShareRecord::new(
                    source,
                    draft.display_name,
                    draft.description,
                    draft.system_prompt,
                    draft.avatar_url,
                    draft.runtime,
                    draft.model,
                    draft.provider,
                )
            });
        match record {
            Ok(record)
                if self
                    .personas
                    .iter()
                    .any(|persona| persona.source == record.source) =>
            {
                self.set_conflict("A persona with this public slug already exists.", cx);
            }
            Ok(record) => {
                self.personas.push(record);
                self.set_saved("Persona created.", cx);
            }
            Err(_) => self.set_validation_error("Persona fields are invalid.", cx),
        }
    }

    pub fn edit_persona(
        &mut self,
        expected_slug: &str,
        draft: PersonaDraft,
        cx: &mut Context<Self>,
    ) {
        let Some(index) = self.personas.iter().position(|persona| {
            matches!(&persona.source, PersonaReference::Published { slug, .. } if slug == expected_slug)
        }) else {
            self.set_conflict("The persona changed before the edit was applied.", cx);
            return;
        };
        let record = PersonaReference::published(self.owner_public_key.clone(), draft.slug)
            .and_then(|source| {
                PublicPersonaShareRecord::new(
                    source,
                    draft.display_name,
                    draft.description,
                    draft.system_prompt,
                    draft.avatar_url,
                    draft.runtime,
                    draft.model,
                    draft.provider,
                )
            });
        match record {
            Ok(record)
                if self
                    .personas
                    .iter()
                    .enumerate()
                    .any(|(other_index, persona)| {
                        other_index != index && persona.source == record.source
                    }) =>
            {
                self.set_conflict("A persona with this public slug already exists.", cx);
            }
            Ok(record) => {
                self.personas[index] = record;
                self.set_saved("Persona updated.", cx);
            }
            Err(_) => self.set_validation_error("Persona fields are invalid.", cx),
        }
    }

    pub fn create_team(&mut self, draft: TeamDraft, cx: &mut Context<Self>) {
        if self.teams.iter().any(|team| team.team_id == draft.team_id) {
            self.set_conflict("A team with this coordinate already exists.", cx);
            return;
        }
        match self.build_team(draft) {
            Ok((team, public_team)) => {
                self.teams.push(team);
                self.public_teams.push(public_team);
                self.set_saved("Team created.", cx);
            }
            Err(TeamDraftError::RevokedIdentity) => {
                self.status = CollaborativeAgentSettingsStatus::RevokedOwner(
                    "A revoked agent identity cannot be added to a team.".into(),
                );
                cx.notify();
            }
            Err(TeamDraftError::Invalid) => {
                self.set_validation_error("Team fields or owner attestations are invalid.", cx);
            }
        }
    }

    pub fn edit_team(&mut self, expected_team_id: &str, draft: TeamDraft, cx: &mut Context<Self>) {
        let Some(index) = self
            .teams
            .iter()
            .position(|team| team.team_id == expected_team_id)
        else {
            self.set_conflict("The team changed before the edit was applied.", cx);
            return;
        };
        if self
            .teams
            .iter()
            .enumerate()
            .any(|(other_index, team)| other_index != index && team.team_id == draft.team_id)
        {
            self.set_conflict("A team with this coordinate already exists.", cx);
            return;
        }
        match self.build_team(draft) {
            Ok((team, public_team)) => {
                self.teams[index] = team;
                self.public_teams[index] = public_team;
                self.set_saved("Team updated.", cx);
            }
            Err(TeamDraftError::RevokedIdentity) => {
                self.status = CollaborativeAgentSettingsStatus::RevokedOwner(
                    "A revoked agent identity cannot be added to a team.".into(),
                );
                cx.notify();
            }
            Err(TeamDraftError::Invalid) => {
                self.set_validation_error("Team fields or owner attestations are invalid.", cx);
            }
        }
    }

    pub fn create_managed_agent(&mut self, draft: ManagedAgentDraft, cx: &mut Context<Self>) {
        let record = match self.build_initial_managed_agent(draft) {
            Ok(record) => record,
            Err(()) => {
                self.set_validation_error("Managed-agent fields are invalid.", cx);
                return;
            }
        };
        self.status = CollaborativeAgentSettingsStatus::Saving;
        cx.notify();
        let repository = self.repository.clone();
        cx.spawn(async move |this, cx| {
            let outcome = repository.insert(&record).await;
            this.update(cx, |this, cx| match outcome {
                Ok(ManagedAgentInsertOutcome::Inserted) => {
                    this.managed_agents
                        .insert(record.agent_public_key().clone(), record);
                    this.set_saved("Managed agent created.", cx);
                }
                Ok(ManagedAgentInsertOutcome::AlreadyExists) => {
                    this.set_conflict("The managed agent already exists.", cx);
                }
                Err(error) => this.set_repository_error(error, cx),
            })
        })
        .detach_and_log_err(cx);
    }

    pub fn edit_managed_agent(
        &mut self,
        agent_public_key: &SettingsPublicKey,
        expected_version: ManagedAgentVersion,
        draft: ManagedAgentDraft,
        cx: &mut Context<Self>,
    ) {
        if draft.agent_public_key != agent_public_key.as_str() {
            self.set_validation_error("Managed-agent identity cannot be changed by an edit.", cx);
            return;
        }
        let Some(current_record) = self.managed_agents.get(agent_public_key).cloned() else {
            self.set_conflict("The managed agent is no longer available.", cx);
            return;
        };
        let event_id = match SettingsEventId::parse(draft.event_id.clone()) {
            Ok(event_id) => event_id,
            Err(_) => {
                self.set_validation_error("Managed-agent event identity is invalid.", cx);
                return;
            }
        };
        let configuration = match managed_agent_configuration(draft) {
            Ok(configuration) => configuration,
            Err(()) => {
                self.set_validation_error("Managed-agent configuration is invalid.", cx);
                return;
            }
        };
        let mut next_record = current_record;
        if next_record
            .replace(&expected_version, event_id, configuration)
            .is_err()
        {
            self.set_conflict("The managed agent changed before the edit was applied.", cx);
            return;
        }
        self.status = CollaborativeAgentSettingsStatus::Saving;
        cx.notify();
        let repository = self.repository.clone();
        let agent_public_key = agent_public_key.clone();
        cx.spawn(async move |this, cx| {
            let outcome = repository
                .compare_and_swap(&expected_version, &next_record)
                .await;
            let current = if matches!(outcome, Ok(ManagedAgentCasOutcome::Stale)) {
                repository
                    .load(
                        next_record.owner_public_key(),
                        next_record.agent_public_key(),
                    )
                    .ok()
                    .flatten()
            } else {
                None
            };
            this.update(cx, |this, cx| match outcome {
                Ok(ManagedAgentCasOutcome::Applied) => {
                    this.managed_agents.insert(agent_public_key, next_record);
                    this.set_saved("Private managed-agent configuration updated.", cx);
                }
                Ok(ManagedAgentCasOutcome::Stale) => {
                    if let Some(current) = current {
                        this.managed_agents
                            .insert(current.agent_public_key().clone(), current);
                    }
                    this.set_conflict("The managed agent changed on another writer.", cx);
                }
                Err(error) => this.set_repository_error(error, cx),
            })
        })
        .detach_and_log_err(cx);
    }

    pub fn share_catalog(
        &mut self,
        source_agent_public_key: &SettingsPublicKey,
        projected_at: u64,
        cx: &mut Context<Self>,
    ) {
        let Some(source_record) = self.managed_agents.get(source_agent_public_key).cloned() else {
            self.set_validation_error(
                "A current managed agent is required to share the catalog.",
                cx,
            );
            return;
        };
        let projection = match self.build_public_projection() {
            Ok(projection) => projection,
            Err(()) => {
                self.set_validation_error(
                    "The catalog cannot be shared until its records are valid.",
                    cx,
                );
                return;
            }
        };
        self.status = CollaborativeAgentSettingsStatus::Saving;
        cx.notify();
        let repository = self.repository.clone();
        cx.spawn(async move |this, cx| {
            let outcome = repository
                .rebuild_public_projection(&source_record, &projection, projected_at)
                .await;
            this.update(cx, |this, cx| match outcome {
                Ok(ProjectionWriteOutcome::Stored) => {
                    this.status = CollaborativeAgentSettingsStatus::Shared;
                    cx.notify();
                }
                Ok(ProjectionWriteOutcome::Stale) => {
                    this.set_conflict("The managed agent changed before sharing completed.", cx);
                }
                Err(error) => this.set_repository_error(error, cx),
            })
        })
        .detach_and_log_err(cx);
    }

    fn build_team(
        &self,
        draft: TeamDraft,
    ) -> Result<(AgentTeamRecord, PublicTeamShareRecord), TeamDraftError> {
        let mut private_members = Vec::with_capacity(draft.members.len());
        let mut public_members = Vec::with_capacity(draft.members.len());
        for member in draft.members {
            if !matches!(
                member.identity.status,
                agent_settings::team::AgentIdentityStatus::Active
            ) {
                return Err(TeamDraftError::RevokedIdentity);
            }
            private_members.push(AgentTeamMember {
                identity: member.identity.clone(),
                persona: member.persona.source.clone(),
                role: member.role.clone(),
            });
            public_members.push(PublicTeamMemberShareRecord {
                agent_public_key: member.identity.public_key,
                owner_attestation: member.identity.owner_attestation,
                role: member.role,
                persona: member.persona,
            });
        }
        let team = AgentTeamRecord::new(
            draft.team_id.clone(),
            self.owner_public_key.clone(),
            draft.name.clone(),
            draft.description.clone(),
            draft.instructions.clone(),
            private_members,
        )
        .map_err(|_| TeamDraftError::Invalid)?;
        let public_team = PublicTeamShareRecord::new(
            draft.team_id,
            self.owner_public_key.clone(),
            draft.name,
            draft.description,
            draft.instructions,
            public_members,
        )
        .map_err(|_| TeamDraftError::Invalid)?;
        Ok((team, public_team))
    }

    fn build_initial_managed_agent(
        &self,
        draft: ManagedAgentDraft,
    ) -> Result<PrivateManagedAgentRecord, ()> {
        let agent_public_key =
            SettingsPublicKey::parse(draft.agent_public_key.clone()).map_err(|_| ())?;
        let event_id = SettingsEventId::parse(draft.event_id.clone()).map_err(|_| ())?;
        let configuration = managed_agent_configuration(draft)?;
        PrivateManagedAgentRecord::new(
            self.owner_public_key.clone(),
            agent_public_key,
            event_id,
            configuration,
        )
        .map_err(|_| ())
    }

    fn build_public_projection(
        &self,
    ) -> Result<collaboration_domain::PublicAgentCatalogProjection, ()> {
        PublicAgentCatalogRecord::new(
            self.owner_public_key.clone(),
            self.personas.clone(),
            self.public_teams.clone(),
        )
        .map_err(|_| ())?;
        let owner_public_key = domain_public_key(&self.owner_public_key)?;
        let personas = self
            .personas
            .iter()
            .map(private_persona_source)
            .collect::<Result<Vec<_>, _>>()?;
        let teams = self
            .public_teams
            .iter()
            .map(private_team_source)
            .collect::<Result<Vec<_>, _>>()?;
        let managed_agents = self
            .managed_agents
            .values()
            .filter_map(|record| match record.state() {
                ManagedAgentState::Active(configuration) => Some((record, configuration)),
                ManagedAgentState::Deleted { .. } => None,
            })
            .map(private_managed_agent_source)
            .collect::<Result<Vec<_>, _>>()?;
        project_public_agent_catalog(&PrivateAgentCatalogProjectionSource {
            owner_public_key,
            personas,
            teams,
            managed_agents,
        })
        .map_err(|_| ())
    }

    fn set_saved(&mut self, message: &'static str, cx: &mut Context<Self>) {
        self.status = CollaborativeAgentSettingsStatus::Saved(message.into());
        cx.notify();
    }

    fn set_conflict(&mut self, message: &'static str, cx: &mut Context<Self>) {
        self.status = CollaborativeAgentSettingsStatus::Conflict(message.into());
        cx.notify();
    }

    fn set_validation_error(&mut self, message: &'static str, cx: &mut Context<Self>) {
        self.status = CollaborativeAgentSettingsStatus::ValidationError(message.into());
        cx.notify();
    }

    fn set_repository_error(&mut self, error: ManagedAgentRepositoryError, cx: &mut Context<Self>) {
        self.status = match error {
            ManagedAgentRepositoryError::InvalidTransition => {
                CollaborativeAgentSettingsStatus::Conflict(
                    "The managed agent changed before the operation completed.".into(),
                )
            }
            ManagedAgentRepositoryError::DeletedRecord => {
                CollaborativeAgentSettingsStatus::Conflict(
                    "The managed agent has been deleted.".into(),
                )
            }
            ManagedAgentRepositoryError::InvalidProjection
            | ManagedAgentRepositoryError::ProjectionOwnerMismatch => {
                CollaborativeAgentSettingsStatus::ValidationError(
                    "The public catalog projection is invalid.".into(),
                )
            }
            ManagedAgentRepositoryError::Unavailable(_)
            | ManagedAgentRepositoryError::CorruptSnapshot
            | ManagedAgentRepositoryError::CorruptProjection => {
                CollaborativeAgentSettingsStatus::Unavailable(
                    "Managed-agent settings are unavailable. Private values were not displayed."
                        .into(),
                )
            }
        };
        cx.notify();
    }

    fn render_section_header(
        title: &'static str,
        count: usize,
        button: Button,
    ) -> impl IntoElement {
        h_flex()
            .w_full()
            .justify_between()
            .gap_3()
            .child(
                h_flex().gap_2().child(Label::new(title)).child(
                    Label::new(count.to_string())
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                ),
            )
            .child(button)
    }

    fn render_empty(label: &'static str) -> impl IntoElement {
        Label::new(label).size(LabelSize::Small).color(Color::Muted)
    }
}

impl EventEmitter<CollaborativeAgentSettingsEvent> for CollaborativeAgentSettings {}

impl Focusable for CollaborativeAgentSettings {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for CollaborativeAgentSettings {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let persona_rows = self.personas.iter().map(|persona| {
            let (slug, owner) = match &persona.source {
                PersonaReference::Published {
                    owner_public_key,
                    slug,
                } => (slug.clone(), owner_public_key.as_str()),
                PersonaReference::Local { persona_id } => (persona_id.clone(), "local"),
            };
            let edit_slug = slug.clone();
            v_flex()
                .p_2()
                .gap_1()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .rounded_md()
                .child(
                    h_flex()
                        .justify_between()
                        .gap_2()
                        .child(Label::new(persona.display_name.clone()))
                        .child(
                            Button::new(
                                SharedString::from(format!("edit-collaborative-persona-{slug}")),
                                "Edit",
                            )
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(
                                move |_, _, _, cx| {
                                    cx.emit(CollaborativeAgentSettingsEvent::EditPersonaRequested(
                                        edit_slug.clone(),
                                    ));
                                },
                            )),
                        ),
                )
                .child(
                    Label::new(format!("{slug} · {owner}"))
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
        });
        let team_rows = self.teams.iter().map(|team| {
            let team_id = team.team_id.clone();
            let edit_team_id = team_id.clone();
            v_flex()
                .p_2()
                .gap_1()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .rounded_md()
                .child(
                    h_flex()
                        .justify_between()
                        .gap_2()
                        .child(Label::new(team.name.clone()))
                        .child(
                            Button::new(
                                SharedString::from(format!("edit-collaborative-team-{team_id}")),
                                "Edit",
                            )
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(
                                move |_, _, _, cx| {
                                    cx.emit(CollaborativeAgentSettingsEvent::EditTeamRequested(
                                        edit_team_id.clone(),
                                    ));
                                },
                            )),
                        ),
                )
                .child(
                    Label::new(format!("{} · {} members", team_id, team.members.len()))
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
        });
        let managed_agent_rows = self.managed_agents.values().map(|record| {
            let agent_public_key = record.agent_public_key().clone();
            let edit_public_key = agent_public_key.clone();
            let share_public_key = agent_public_key.clone();
            let (runtime, provider, model, environment_count) = match record.state() {
                ManagedAgentState::Active(configuration) => (
                    configuration.runtime().as_str(),
                    configuration.provider().map(ProviderId::as_str).unwrap_or("default provider"),
                    configuration.model().map(ModelId::as_str).unwrap_or("default model"),
                    configuration.environment().len(),
                ),
                ManagedAgentState::Deleted { .. } => ("deleted", "unavailable", "unavailable", 0),
            };
            v_flex()
                .p_2()
                .gap_1()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .rounded_md()
                .child(
                    h_flex()
                        .justify_between()
                        .gap_2()
                        .child(Label::new(short_public_key(agent_public_key.as_str())))
                        .child(
                            h_flex()
                                .gap_1()
                                .child(
                                    Button::new(
                                        SharedString::from(format!(
                                            "edit-collaborative-managed-agent-{}",
                                            agent_public_key.as_str()
                                        )),
                                        "Edit private",
                                    )
                                    .style(ButtonStyle::Subtle)
                                    .on_click(cx.listener(move |_, _, _, cx| {
                                        cx.emit(
                                            CollaborativeAgentSettingsEvent::EditManagedAgentRequested(
                                                edit_public_key.clone(),
                                            ),
                                        );
                                    })),
                                )
                                .child(
                                    Button::new(
                                        SharedString::from(format!(
                                            "share-collaborative-managed-agent-{}",
                                            agent_public_key.as_str()
                                        )),
                                        "Share catalog",
                                    )
                                    .style(ButtonStyle::Outlined)
                                    .on_click(cx.listener(move |_, _, _, cx| {
                                        cx.emit(CollaborativeAgentSettingsEvent::ShareRequested(
                                            share_public_key.clone(),
                                        ));
                                    })),
                                ),
                        ),
                )
                .child(
                    Label::new(format!("{runtime} · {provider} · {model}"))
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .child(
                    Label::new(format!(
                        "Private environment bindings: {environment_count} · references hidden"
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
        });

        v_flex()
            .id("collaborative-agent-settings")
            .track_focus(&self.focus_handle)
            .size_full()
            .overflow_y_scroll()
            .bg(cx.theme().colors().editor_background)
            .p_4()
            .gap_4()
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Agent catalog"))
                    .child(
                        Label::new(
                            "Create personas, teams, and managed agents. Public sharing is explicit.",
                        )
                        .color(Color::Muted),
                    ),
            )
            .child(
                v_flex()
                    .p_3()
                    .gap_1()
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .rounded_md()
                    .child(Label::new("Private configuration"))
                    .child(
                        Label::new(
                            "Environment and credential references remain private and are never shown in the shared catalog.",
                        )
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    ),
            )
            .when_some(self.status.label(), |this, label| {
                this.child(
                    div()
                        .id("collaborative-agent-settings-status")
                        .p_2()
                        .rounded_md()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .child(Label::new(label).color(self.status.color())),
                )
            })
            .child(
                v_flex()
                    .gap_2()
                    .child(Self::render_section_header(
                        "Personas",
                        self.personas.len(),
                        Button::new("create-collaborative-persona", "Create persona")
                            .style(ButtonStyle::Outlined)
                            .on_click(cx.listener(|_, _, _, cx| {
                                cx.emit(
                                    CollaborativeAgentSettingsEvent::CreatePersonaRequested,
                                );
                            })),
                    ))
                    .when(self.personas.is_empty(), |this| {
                        this.child(Self::render_empty("No personas configured."))
                    })
                    .children(persona_rows),
            )
            .child(
                v_flex()
                    .gap_2()
                    .child(Self::render_section_header(
                        "Teams",
                        self.teams.len(),
                        Button::new("create-collaborative-team", "Create team")
                            .style(ButtonStyle::Outlined)
                            .on_click(cx.listener(|_, _, _, cx| {
                                cx.emit(CollaborativeAgentSettingsEvent::CreateTeamRequested);
                            })),
                    ))
                    .when(self.teams.is_empty(), |this| {
                        this.child(Self::render_empty("No teams configured."))
                    })
                    .children(team_rows),
            )
            .child(
                v_flex()
                    .gap_2()
                    .child(Self::render_section_header(
                        "Managed agents",
                        self.managed_agents.len(),
                        Button::new("create-collaborative-managed-agent", "Create managed agent")
                            .style(ButtonStyle::Outlined)
                            .on_click(cx.listener(|_, _, _, cx| {
                                cx.emit(
                                    CollaborativeAgentSettingsEvent::CreateManagedAgentRequested,
                                );
                            })),
                    ))
                    .when(self.managed_agents.is_empty(), |this| {
                        this.child(Self::render_empty("No managed agents configured."))
                    })
                    .children(managed_agent_rows),
            )
    }
}

#[derive(Clone, Copy)]
enum TeamDraftError {
    RevokedIdentity,
    Invalid,
}

fn managed_agent_configuration(draft: ManagedAgentDraft) -> Result<ManagedAgentConfiguration, ()> {
    ManagedAgentConfiguration::new(
        RuntimeId::parse(draft.runtime).map_err(|_| ())?,
        draft
            .provider
            .map(ProviderId::parse)
            .transpose()
            .map_err(|_| ())?,
        draft
            .model
            .map(ModelId::parse)
            .transpose()
            .map_err(|_| ())?,
        draft.environment,
    )
    .map_err(|_| ())
}

fn private_persona_source(
    persona: &PublicPersonaShareRecord,
) -> Result<PrivatePersonaProjectionSource, ()> {
    let publication_slug = match &persona.source {
        PersonaReference::Published { slug, .. } => Some(slug.clone()),
        PersonaReference::Local { .. } => None,
    };
    Ok(PrivatePersonaProjectionSource {
        publication_slug,
        display_name: persona.display_name.clone(),
        description: persona.description.clone(),
        system_prompt: persona.system_prompt.clone(),
        avatar_url: persona.avatar_url.clone(),
        runtime: persona.runtime.clone(),
        model: persona.model.clone(),
        provider: persona.provider.clone(),
        environment_references: Vec::new(),
        credential_references: Vec::new(),
        local_source_path: None,
    })
}

fn private_team_source(team: &PublicTeamShareRecord) -> Result<PrivateTeamProjectionSource, ()> {
    Ok(PrivateTeamProjectionSource {
        team_id: team.team_id.clone(),
        owner_public_key: domain_public_key(&team.owner_public_key)?,
        name: team.name.clone(),
        description: team.description.clone(),
        instructions: team.instructions.clone(),
        members: team
            .members
            .iter()
            .map(|member| {
                Ok(PrivateTeamMemberProjectionSource {
                    agent_public_key: domain_public_key(&member.agent_public_key)?,
                    owner_attestation: OwnerAttestationEvidence {
                        owner_public_key: domain_public_key(
                            &member.owner_attestation.owner_public_key,
                        )?,
                        agent_public_key: domain_public_key(
                            &member.owner_attestation.agent_public_key,
                        )?,
                        proof_event_id: domain_event_id(&member.owner_attestation.proof_event_id)?,
                        exact_conditions: member.owner_attestation.exact_conditions.clone(),
                        verified_at: member.owner_attestation.verified_at,
                    },
                    role: member.role.as_str().to_string(),
                    persona: private_persona_source(&member.persona)?,
                    respond_to_allowlist: Vec::new(),
                    environment_references: Vec::new(),
                    credential_references: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, ()>>()?,
        local_source_path: None,
    })
}

fn private_managed_agent_source(
    (record, configuration): (&PrivateManagedAgentRecord, &ManagedAgentConfiguration),
) -> Result<PrivateAgentProjectionState, ()> {
    let mut environment_references = Vec::new();
    let mut credential_references = Vec::new();
    for reference in configuration.environment().values() {
        match reference {
            EnvironmentReference::ProcessEnvironment(variable) => {
                environment_references
                    .push(PrivateAgentReference::new(variable.as_str()).map_err(|_| ())?);
            }
            EnvironmentReference::ProtectedCredential(reference) => {
                credential_references
                    .push(PrivateAgentReference::new(reference.as_str()).map_err(|_| ())?);
            }
        }
    }
    Ok(PrivateAgentProjectionState {
        owner_public_key: domain_public_key(record.owner_public_key())?,
        agent_public_key: domain_public_key(record.agent_public_key())?,
        generation: record.version().generation(),
        current_event_id: domain_event_id(record.version().event_id())?,
        environment_references,
        credential_references,
        local_source_path: None,
        backend_reference: None,
        respond_to_allowlist: Vec::new(),
    })
}

fn domain_public_key(public_key: &SettingsPublicKey) -> Result<DomainPublicKey, ()> {
    decode_hex(public_key.as_str()).map(DomainPublicKey::from_bytes)
}

fn domain_event_id(event_id: &SettingsEventId) -> Result<DomainEventId, ()> {
    decode_hex(event_id.as_str()).map(DomainEventId::from_bytes)
}

fn decode_hex(value: &str) -> Result<[u8; 32], ()> {
    if value.len() != 64 {
        return Err(());
    }
    let mut bytes = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_nibble(chunk[0])?;
        let low = decode_nibble(chunk[1])?;
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

fn decode_nibble(value: u8) -> Result<u8, ()> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(()),
    }
}

fn short_public_key(public_key: &str) -> String {
    let prefix = public_key.get(..8).unwrap_or(public_key);
    let suffix = public_key
        .get(public_key.len().saturating_sub(8)..)
        .unwrap_or(public_key);
    format!("{prefix}…{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_settings::{
        managed_agent::ProtectedCredentialReference,
        team::{AgentIdentityStatus, OwnerAttestationRecord},
    };
    use gpui::TestAppContext;

    fn lower_hex(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn public_key(byte: u8) -> SettingsPublicKey {
        SettingsPublicKey::parse(lower_hex(byte)).expect("fixture public key must be valid")
    }

    fn event_id(byte: u8) -> SettingsEventId {
        SettingsEventId::parse(lower_hex(byte)).expect("fixture event ID must be valid")
    }

    fn persona(slug: &str) -> PersonaDraft {
        PersonaDraft {
            slug: slug.to_string(),
            display_name: "Reviewer".to_string(),
            description: "Reviews changes".to_string(),
            system_prompt: Some("Review carefully".to_string()),
            avatar_url: None,
            runtime: Some("claude-code".to_string()),
            model: Some("claude-opus-4-1".to_string()),
            provider: Some("anthropic".to_string()),
        }
    }

    fn managed_agent(
        agent_byte: u8,
        event_byte: u8,
        credential_reference: &str,
    ) -> ManagedAgentDraft {
        let mut environment = BTreeMap::new();
        environment.insert(
            EnvironmentVariableName::parse("ANTHROPIC_API_KEY")
                .expect("fixture environment variable must be valid"),
            EnvironmentReference::ProtectedCredential(
                ProtectedCredentialReference::parse(credential_reference)
                    .expect("fixture credential reference must be valid"),
            ),
        );
        ManagedAgentDraft {
            agent_public_key: lower_hex(agent_byte),
            event_id: lower_hex(event_byte),
            runtime: "claude-code".to_string(),
            provider: Some("anthropic".to_string()),
            model: Some("claude-opus-4-1".to_string()),
            environment,
        }
    }

    fn team_member(owner: u8, agent: u8, status: AgentIdentityStatus) -> TeamMemberDraft {
        let owner_public_key = public_key(owner);
        let agent_public_key = public_key(agent);
        let attestation = OwnerAttestationRecord::new(
            owner_public_key.clone(),
            agent_public_key.clone(),
            event_id(9),
            "may review changes",
            10,
        )
        .expect("fixture attestation must be valid");
        let identity = AgentIdentityRecord::new(agent_public_key, attestation, status)
            .expect("fixture identity must be structurally valid");
        TeamMemberDraft {
            identity,
            role: TeamRole::parse("reviewer").expect("fixture role must be valid"),
            persona: PublicPersonaShareRecord::new(
                PersonaReference::published(owner_public_key, "reviewer")
                    .expect("fixture persona reference must be valid"),
                "Reviewer",
                "Reviews changes",
                None,
                None,
                None,
                None,
                None,
            )
            .expect("fixture persona must be valid"),
        }
    }

    #[gpui::test]
    async fn create_share_and_private_edit(cx: &mut TestAppContext) {
        let repository =
            ManagedAgentRepository::open_test_database("collaborative_agent_settings_create").await;
        let repository_for_assertion = repository.clone();
        let owner_public_key = public_key(1);
        let view =
            cx.new(|cx| CollaborativeAgentSettings::new(repository, owner_public_key.clone(), cx));

        view.update(cx, |view, cx| {
            view.create_persona(persona("reviewer"), cx);
            view.create_team(
                TeamDraft {
                    team_id: "review-team".to_string(),
                    name: "Review Team".to_string(),
                    description: None,
                    instructions: None,
                    members: vec![team_member(1, 2, AgentIdentityStatus::Active)],
                },
                cx,
            );
            view.create_managed_agent(managed_agent(2, 3, "credentials/anthropic/private-one"), cx);
        });
        cx.run_until_parked();

        let agent_public_key = public_key(2);
        let expected_version = view.read_with(cx, |view, _| {
            assert_eq!(view.persona_count(), 1);
            assert_eq!(view.team_count(), 1);
            assert_eq!(view.managed_agent_count(), 1);
            view.managed_agent(&agent_public_key)
                .expect("managed agent must exist")
                .version()
                .clone()
        });
        view.update(cx, |view, cx| {
            view.share_catalog(&agent_public_key, 11, cx);
        });
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |view, _| view.status().clone()),
            CollaborativeAgentSettingsStatus::Shared
        );
        let projection = repository_for_assertion
            .load_public_projection(&owner_public_key, &agent_public_key)
            .expect("load shared projection")
            .expect("shared projection must exist");
        let serialized = projection.projection.to_string();
        assert!(serialized.contains("review-team"));
        assert!(!serialized.contains("private-one"));
        assert!(!serialized.contains("ANTHROPIC_API_KEY"));

        view.update(cx, |view, cx| {
            view.edit_managed_agent(
                &agent_public_key,
                expected_version,
                managed_agent(2, 4, "credentials/anthropic/private-two"),
                cx,
            );
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            let record = view
                .managed_agent(&agent_public_key)
                .expect("edited managed agent must exist");
            assert_eq!(record.version().generation(), 2);
            assert_eq!(
                view.status(),
                &CollaborativeAgentSettingsStatus::Saved(
                    "Private managed-agent configuration updated.".into()
                )
            );
        });
    }

    #[gpui::test]
    async fn stale_private_edit_surfaces_conflict_and_refreshes(cx: &mut TestAppContext) {
        let repository =
            ManagedAgentRepository::open_test_database("collaborative_agent_settings_conflict")
                .await;
        let owner_public_key = public_key(1);
        let agent_public_key = public_key(2);
        let view =
            cx.new(|cx| CollaborativeAgentSettings::new(repository.clone(), owner_public_key, cx));
        view.update(cx, |view, cx| {
            view.create_managed_agent(managed_agent(2, 3, "credentials/first"), cx);
        });
        cx.run_until_parked();
        let stale_version = view.read_with(cx, |view, _| {
            view.managed_agent(&agent_public_key)
                .expect("managed agent must exist")
                .version()
                .clone()
        });
        let mut external = repository
            .load(&public_key(1), &agent_public_key)
            .expect("load external record")
            .expect("external record must exist");
        external
            .replace(
                &stale_version,
                event_id(4),
                managed_agent_configuration(managed_agent(2, 4, "credentials/external"))
                    .expect("external configuration must be valid"),
            )
            .expect("external update must be valid");
        assert_eq!(
            repository
                .compare_and_swap(&stale_version, &external)
                .await
                .expect("external CAS must run"),
            ManagedAgentCasOutcome::Applied
        );

        view.update(cx, |view, cx| {
            view.edit_managed_agent(
                &agent_public_key,
                stale_version,
                managed_agent(2, 5, "credentials/local"),
                cx,
            );
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(matches!(
                view.status(),
                CollaborativeAgentSettingsStatus::Conflict(_)
            ));
            assert_eq!(
                view.managed_agent(&agent_public_key)
                    .expect("refreshed agent must exist")
                    .version()
                    .generation(),
                2
            );
        });
    }

    #[gpui::test]
    async fn revoked_owner_and_invalid_fields_are_visible(cx: &mut TestAppContext) {
        let repository =
            ManagedAgentRepository::open_test_database("collaborative_agent_settings_validation")
                .await;
        let view = cx.new(|cx| CollaborativeAgentSettings::new(repository, public_key(1), cx));
        view.update(cx, |view, cx| {
            view.create_team(
                TeamDraft {
                    team_id: "review-team".to_string(),
                    name: "Review Team".to_string(),
                    description: None,
                    instructions: None,
                    members: vec![team_member(
                        1,
                        2,
                        AgentIdentityStatus::Revoked { revoked_at: 12 },
                    )],
                },
                cx,
            );
        });
        assert!(matches!(
            view.read_with(cx, |view, _| view.status().clone()),
            CollaborativeAgentSettingsStatus::RevokedOwner(_)
        ));
        view.update(cx, |view, cx| {
            view.create_persona(persona("Invalid Slug"), cx);
        });
        assert!(matches!(
            view.read_with(cx, |view, _| view.status().clone()),
            CollaborativeAgentSettingsStatus::ValidationError(_)
        ));
    }
}
