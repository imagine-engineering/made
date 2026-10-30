use crate::{NostrEventId, NostrPublicKey, OwnerAttestationEvidence};
use serde::Serialize;
use std::{collections::BTreeSet, error::Error, fmt};

const MAX_PERSONA_SLUG_BYTES: usize = 64;
const MAX_TEAM_ID_CHARACTERS: usize = 64;
const MAX_NAME_BYTES: usize = 256;
const MAX_DESCRIPTION_BYTES: usize = 4_096;
const MAX_PROMPT_BYTES: usize = 262_144;
const MAX_PUBLIC_VALUE_BYTES: usize = 16_384;
const MAX_ROLE_BYTES: usize = 32;
const MAX_REFERENCE_BYTES: usize = 512;
const MAX_ATTESTATION_CONDITIONS_BYTES: usize = 1_024;
const MAX_MEMBERS: usize = 256;
const MAX_CATALOG_ENTRIES: usize = 1_024;
const MAX_PRIVATE_REFERENCES: usize = 256;
const MAX_SAFE_GENERATION: u64 = (1_u64 << 53) - 1;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrivateAgentReference(String);

impl PrivateAgentReference {
    pub fn new(value: impl Into<String>) -> Result<Self, AgentProjectionError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.len() > MAX_REFERENCE_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(AgentProjectionError::InvalidPrivateReference);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PrivateAgentReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateAgentReference(<redacted>)")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProjectionField {
    PersonaSlug,
    PersonaDisplayName,
    PersonaDescription,
    PersonaSystemPrompt,
    PersonaAvatarUrl,
    PersonaRuntime,
    PersonaModel,
    PersonaProvider,
    TeamId,
    TeamName,
    TeamDescription,
    TeamInstructions,
    MemberAgentPublicKey,
    MemberOwnerAttestation,
    MemberRole,
    EnvironmentReference,
    CredentialReference,
    LocalSourcePath,
    BackendReference,
    ManagedAgentVersion,
    RespondToAllowlist,
}

impl AgentProjectionField {
    pub const ALL: [Self; 21] = [
        Self::PersonaSlug,
        Self::PersonaDisplayName,
        Self::PersonaDescription,
        Self::PersonaSystemPrompt,
        Self::PersonaAvatarUrl,
        Self::PersonaRuntime,
        Self::PersonaModel,
        Self::PersonaProvider,
        Self::TeamId,
        Self::TeamName,
        Self::TeamDescription,
        Self::TeamInstructions,
        Self::MemberAgentPublicKey,
        Self::MemberOwnerAttestation,
        Self::MemberRole,
        Self::EnvironmentReference,
        Self::CredentialReference,
        Self::LocalSourcePath,
        Self::BackendReference,
        Self::ManagedAgentVersion,
        Self::RespondToAllowlist,
    ];

    pub const fn is_public(self) -> bool {
        match self {
            Self::PersonaSlug
            | Self::PersonaDisplayName
            | Self::PersonaDescription
            | Self::PersonaSystemPrompt
            | Self::PersonaAvatarUrl
            | Self::PersonaRuntime
            | Self::PersonaModel
            | Self::PersonaProvider
            | Self::TeamId
            | Self::TeamName
            | Self::TeamDescription
            | Self::TeamInstructions
            | Self::MemberAgentPublicKey
            | Self::MemberOwnerAttestation
            | Self::MemberRole => true,
            Self::EnvironmentReference
            | Self::CredentialReference
            | Self::LocalSourcePath
            | Self::BackendReference
            | Self::ManagedAgentVersion
            | Self::RespondToAllowlist => false,
        }
    }
}

pub fn validate_public_projection_fields(
    fields: impl IntoIterator<Item = AgentProjectionField>,
) -> Result<(), AgentProjectionError> {
    for field in fields {
        if !field.is_public() {
            return Err(AgentProjectionError::PrivateFieldRequested);
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateAgentProjectionState {
    pub owner_public_key: NostrPublicKey,
    pub agent_public_key: NostrPublicKey,
    pub generation: u64,
    pub current_event_id: NostrEventId,
    pub environment_references: Vec<PrivateAgentReference>,
    pub credential_references: Vec<PrivateAgentReference>,
    pub local_source_path: Option<PrivateAgentReference>,
    pub backend_reference: Option<PrivateAgentReference>,
    pub respond_to_allowlist: Vec<NostrPublicKey>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivatePersonaProjectionSource {
    pub publication_slug: Option<String>,
    pub display_name: String,
    pub description: String,
    pub system_prompt: Option<String>,
    pub avatar_url: Option<String>,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub environment_references: Vec<PrivateAgentReference>,
    pub credential_references: Vec<PrivateAgentReference>,
    pub local_source_path: Option<PrivateAgentReference>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateTeamMemberProjectionSource {
    pub agent_public_key: NostrPublicKey,
    pub owner_attestation: OwnerAttestationEvidence,
    pub role: String,
    pub persona: PrivatePersonaProjectionSource,
    pub respond_to_allowlist: Vec<NostrPublicKey>,
    pub environment_references: Vec<PrivateAgentReference>,
    pub credential_references: Vec<PrivateAgentReference>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateTeamProjectionSource {
    pub team_id: String,
    pub owner_public_key: NostrPublicKey,
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub members: Vec<PrivateTeamMemberProjectionSource>,
    pub local_source_path: Option<PrivateAgentReference>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateAgentCatalogProjectionSource {
    pub owner_public_key: NostrPublicKey,
    pub personas: Vec<PrivatePersonaProjectionSource>,
    pub teams: Vec<PrivateTeamProjectionSource>,
    pub managed_agents: Vec<PrivateAgentProjectionState>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicPersonaProjection {
    pub slug: String,
    pub display_name: String,
    pub description: String,
    pub system_prompt: Option<String>,
    pub avatar_url: Option<String>,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicEmbeddedPersonaProjection {
    pub display_name: String,
    pub description: String,
    pub system_prompt: Option<String>,
    pub avatar_url: Option<String>,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicTeamMemberProjection {
    pub agent_public_key: NostrPublicKey,
    pub owner_attestation: OwnerAttestationEvidence,
    pub role: String,
    pub persona: PublicEmbeddedPersonaProjection,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicTeamProjection {
    pub team_id: String,
    pub owner_public_key: NostrPublicKey,
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub members: Vec<PublicTeamMemberProjection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicAgentCatalogProjection {
    pub owner_public_key: NostrPublicKey,
    pub personas: Vec<PublicPersonaProjection>,
    pub teams: Vec<PublicTeamProjection>,
}

pub fn project_public_agent_catalog(
    source: &PrivateAgentCatalogProjectionSource,
) -> Result<PublicAgentCatalogProjection, AgentProjectionError> {
    if source.personas.len().saturating_add(source.teams.len()) > MAX_CATALOG_ENTRIES {
        return Err(AgentProjectionError::TooManyCatalogEntries);
    }

    for managed_agent in &source.managed_agents {
        validate_private_agent(managed_agent, source.owner_public_key)?;
    }

    let mut persona_slugs = BTreeSet::new();
    let mut personas = Vec::with_capacity(source.personas.len());
    for persona in &source.personas {
        let slug = persona
            .publication_slug
            .as_deref()
            .ok_or(AgentProjectionError::MissingPersonaSlug)?;
        validate_persona(persona)?;
        if !persona_slugs.insert(slug) {
            return Err(AgentProjectionError::DuplicatePersona);
        }
        personas.push(PublicPersonaProjection {
            slug: slug.to_string(),
            display_name: persona.display_name.clone(),
            description: persona.description.clone(),
            system_prompt: persona.system_prompt.clone(),
            avatar_url: persona.avatar_url.clone(),
            runtime: persona.runtime.clone(),
            model: persona.model.clone(),
            provider: persona.provider.clone(),
        });
    }

    let mut team_ids = BTreeSet::new();
    let mut teams = Vec::with_capacity(source.teams.len());
    for team in &source.teams {
        validate_team(team, source.owner_public_key)?;
        if !team_ids.insert(team.team_id.as_str()) {
            return Err(AgentProjectionError::DuplicateTeam);
        }
        let members = team
            .members
            .iter()
            .map(project_team_member)
            .collect::<Result<Vec<_>, _>>()?;
        teams.push(PublicTeamProjection {
            team_id: team.team_id.clone(),
            owner_public_key: team.owner_public_key,
            name: team.name.clone(),
            description: team.description.clone(),
            instructions: team.instructions.clone(),
            members,
        });
    }

    Ok(PublicAgentCatalogProjection {
        owner_public_key: source.owner_public_key,
        personas,
        teams,
    })
}

fn project_team_member(
    member: &PrivateTeamMemberProjectionSource,
) -> Result<PublicTeamMemberProjection, AgentProjectionError> {
    validate_persona(&member.persona)?;
    Ok(PublicTeamMemberProjection {
        agent_public_key: member.agent_public_key,
        owner_attestation: member.owner_attestation.clone(),
        role: member.role.clone(),
        persona: PublicEmbeddedPersonaProjection {
            display_name: member.persona.display_name.clone(),
            description: member.persona.description.clone(),
            system_prompt: member.persona.system_prompt.clone(),
            avatar_url: member.persona.avatar_url.clone(),
            runtime: member.persona.runtime.clone(),
            model: member.persona.model.clone(),
            provider: member.persona.provider.clone(),
        },
    })
}

fn validate_private_agent(
    agent: &PrivateAgentProjectionState,
    catalog_owner: NostrPublicKey,
) -> Result<(), AgentProjectionError> {
    if agent.owner_public_key != catalog_owner || agent.owner_public_key == agent.agent_public_key {
        return Err(AgentProjectionError::OwnerMismatch);
    }
    if agent.generation == 0 || agent.generation > MAX_SAFE_GENERATION {
        return Err(AgentProjectionError::InvalidManagedAgentVersion);
    }
    validate_private_reference_count(
        agent
            .environment_references
            .len()
            .saturating_add(agent.credential_references.len()),
    )
}

fn validate_persona(persona: &PrivatePersonaProjectionSource) -> Result<(), AgentProjectionError> {
    if persona
        .publication_slug
        .as_deref()
        .is_some_and(|slug| !valid_persona_slug(slug))
    {
        return Err(AgentProjectionError::InvalidPersonaSlug);
    }
    validate_required_text(&persona.display_name, MAX_NAME_BYTES)?;
    validate_required_text(&persona.description, MAX_DESCRIPTION_BYTES)?;
    validate_optional_text(persona.system_prompt.as_deref(), MAX_PROMPT_BYTES)?;
    for value in [
        persona.avatar_url.as_deref(),
        persona.runtime.as_deref(),
        persona.model.as_deref(),
        persona.provider.as_deref(),
    ] {
        validate_optional_text(value, MAX_PUBLIC_VALUE_BYTES)?;
    }
    validate_private_reference_count(
        persona
            .environment_references
            .len()
            .saturating_add(persona.credential_references.len()),
    )
}

fn validate_team(
    team: &PrivateTeamProjectionSource,
    catalog_owner: NostrPublicKey,
) -> Result<(), AgentProjectionError> {
    if team.owner_public_key != catalog_owner {
        return Err(AgentProjectionError::OwnerMismatch);
    }
    if !valid_team_id(&team.team_id) {
        return Err(AgentProjectionError::InvalidTeamId);
    }
    validate_required_text(&team.name, MAX_NAME_BYTES)?;
    validate_optional_text(team.description.as_deref(), MAX_DESCRIPTION_BYTES)?;
    validate_optional_text(team.instructions.as_deref(), MAX_PROMPT_BYTES)?;
    if team.members.len() > MAX_MEMBERS {
        return Err(AgentProjectionError::TooManyMembers);
    }
    let mut agents = BTreeSet::new();
    for member in &team.members {
        if !agents.insert(member.agent_public_key) {
            return Err(AgentProjectionError::DuplicateMember);
        }
        if member.owner_attestation.owner_public_key != team.owner_public_key
            || member.owner_attestation.agent_public_key != member.agent_public_key
        {
            return Err(AgentProjectionError::OwnerMismatch);
        }
        if member.owner_attestation.verified_at == 0
            || member.owner_attestation.exact_conditions.len() > MAX_ATTESTATION_CONDITIONS_BYTES
            || member.owner_attestation.exact_conditions.contains('\0')
        {
            return Err(AgentProjectionError::InvalidOwnerAttestation);
        }
        if !valid_role(&member.role) {
            return Err(AgentProjectionError::InvalidRole);
        }
        validate_private_reference_count(
            member
                .environment_references
                .len()
                .saturating_add(member.credential_references.len()),
        )?;
    }
    Ok(())
}

fn validate_private_reference_count(count: usize) -> Result<(), AgentProjectionError> {
    if count > MAX_PRIVATE_REFERENCES {
        return Err(AgentProjectionError::TooManyPrivateReferences);
    }
    Ok(())
}

fn valid_persona_slug(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    value.len() <= MAX_PERSONA_SLUG_BYTES
        && (first.is_ascii_lowercase() || first.is_ascii_digit())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn valid_team_id(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_TEAM_ID_CHARACTERS
        && !value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
}

fn valid_role(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    value.len() <= MAX_ROLE_BYTES
        && first.is_ascii_lowercase()
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn validate_required_text(value: &str, maximum: usize) -> Result<(), AgentProjectionError> {
    if value.trim().is_empty() || value.len() > maximum || value.contains('\0') {
        return Err(AgentProjectionError::InvalidPublicField);
    }
    Ok(())
}

fn validate_optional_text(value: Option<&str>, maximum: usize) -> Result<(), AgentProjectionError> {
    if value.is_some_and(|value| value.len() > maximum || value.contains('\0')) {
        return Err(AgentProjectionError::InvalidPublicField);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProjectionError {
    PrivateFieldRequested,
    InvalidPrivateReference,
    InvalidPublicField,
    InvalidPersonaSlug,
    MissingPersonaSlug,
    InvalidTeamId,
    InvalidRole,
    InvalidOwnerAttestation,
    InvalidManagedAgentVersion,
    OwnerMismatch,
    DuplicatePersona,
    DuplicateTeam,
    DuplicateMember,
    TooManyPrivateReferences,
    TooManyMembers,
    TooManyCatalogEntries,
}

impl fmt::Display for AgentProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::PrivateFieldRequested => "private agent field cannot enter a public projection",
            Self::InvalidPrivateReference => "invalid private agent reference",
            Self::InvalidPublicField => "invalid public agent field",
            Self::InvalidPersonaSlug => "invalid public persona slug",
            Self::MissingPersonaSlug => "standalone public persona is missing its slug",
            Self::InvalidTeamId => "invalid public team identifier",
            Self::InvalidRole => "invalid public team role",
            Self::InvalidOwnerAttestation => "invalid public owner attestation",
            Self::InvalidManagedAgentVersion => "invalid private managed-agent version",
            Self::OwnerMismatch => "agent projection owner does not match",
            Self::DuplicatePersona => "agent projection contains a duplicate persona",
            Self::DuplicateTeam => "agent projection contains a duplicate team",
            Self::DuplicateMember => "agent projection contains a duplicate team member",
            Self::TooManyPrivateReferences => "agent record contains too many private references",
            Self::TooManyMembers => "public team contains too many members",
            Self::TooManyCatalogEntries => "public catalog contains too many entries",
        };
        formatter.write_str(message)
    }
}

impl Error for AgentProjectionError {}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET_ENVIRONMENT: &str = "private/environment/reference";
    const SECRET_CREDENTIAL: &str = "private/credential/reference";
    const SECRET_PATH: &str = "private/local/source/path";
    const SECRET_BACKEND: &str = "private/backend/reference";

    fn owner() -> NostrPublicKey {
        NostrPublicKey::from_bytes([1; 32])
    }

    fn agent() -> NostrPublicKey {
        NostrPublicKey::from_bytes([2; 32])
    }

    fn private_reference(value: &str) -> PrivateAgentReference {
        PrivateAgentReference::new(value).expect("fixture reference must be valid")
    }

    fn persona(slug: Option<&str>) -> PrivatePersonaProjectionSource {
        PrivatePersonaProjectionSource {
            publication_slug: slug.map(str::to_string),
            display_name: "Reviewer".to_string(),
            description: "Reviews changes".to_string(),
            system_prompt: Some("Review carefully".to_string()),
            avatar_url: Some("https://example.test/reviewer.png".to_string()),
            runtime: Some("claude-code".to_string()),
            model: Some("claude-opus-4-1".to_string()),
            provider: Some("anthropic".to_string()),
            environment_references: vec![private_reference(SECRET_ENVIRONMENT)],
            credential_references: vec![private_reference(SECRET_CREDENTIAL)],
            local_source_path: Some(private_reference(SECRET_PATH)),
        }
    }

    fn member() -> PrivateTeamMemberProjectionSource {
        PrivateTeamMemberProjectionSource {
            agent_public_key: agent(),
            owner_attestation: OwnerAttestationEvidence {
                owner_public_key: owner(),
                agent_public_key: agent(),
                proof_event_id: NostrEventId::from_bytes([3; 32]),
                exact_conditions: String::new(),
                verified_at: 1,
            },
            role: "reviewer".to_string(),
            persona: persona(None),
            respond_to_allowlist: vec![NostrPublicKey::from_bytes([4; 32])],
            environment_references: vec![private_reference(SECRET_ENVIRONMENT)],
            credential_references: vec![private_reference(SECRET_CREDENTIAL)],
        }
    }

    fn source() -> PrivateAgentCatalogProjectionSource {
        PrivateAgentCatalogProjectionSource {
            owner_public_key: owner(),
            personas: vec![persona(Some("reviewer"))],
            teams: vec![PrivateTeamProjectionSource {
                team_id: "builtin-team:review".to_string(),
                owner_public_key: owner(),
                name: "Review Team".to_string(),
                description: Some("A shared team".to_string()),
                instructions: Some("Coordinate reviews".to_string()),
                members: vec![member()],
                local_source_path: Some(private_reference(SECRET_PATH)),
            }],
            managed_agents: vec![PrivateAgentProjectionState {
                owner_public_key: owner(),
                agent_public_key: agent(),
                generation: 1,
                current_event_id: NostrEventId::from_bytes([5; 32]),
                environment_references: vec![private_reference(SECRET_ENVIRONMENT)],
                credential_references: vec![private_reference(SECRET_CREDENTIAL)],
                local_source_path: Some(private_reference(SECRET_PATH)),
                backend_reference: Some(private_reference(SECRET_BACKEND)),
                respond_to_allowlist: vec![NostrPublicKey::from_bytes([4; 32])],
            }],
        }
    }

    #[test]
    fn projection_exhaustively_omits_private_fields() {
        let source = source();
        let PrivateAgentCatalogProjectionSource {
            owner_public_key: _,
            personas,
            teams,
            managed_agents,
        } = &source;
        let PrivatePersonaProjectionSource {
            publication_slug: _,
            display_name: _,
            description: _,
            system_prompt: _,
            avatar_url: _,
            runtime: _,
            model: _,
            provider: _,
            environment_references: _,
            credential_references: _,
            local_source_path: _,
        } = &personas[0];
        let PrivateTeamProjectionSource {
            team_id: _,
            owner_public_key: _,
            name: _,
            description: _,
            instructions: _,
            members,
            local_source_path: _,
        } = &teams[0];
        let PrivateTeamMemberProjectionSource {
            agent_public_key: _,
            owner_attestation: _,
            role: _,
            persona: _,
            respond_to_allowlist: _,
            environment_references: _,
            credential_references: _,
        } = &members[0];
        let PrivateAgentProjectionState {
            owner_public_key: _,
            agent_public_key: _,
            generation: _,
            current_event_id: _,
            environment_references: _,
            credential_references: _,
            local_source_path: _,
            backend_reference: _,
            respond_to_allowlist: _,
        } = &managed_agents[0];

        let projection = project_public_agent_catalog(&source)
            .expect("valid private source must produce a public projection");
        let json = serde_json::to_string(&projection).expect("projection must serialize");

        for private_value in [
            SECRET_ENVIRONMENT,
            SECRET_CREDENTIAL,
            SECRET_PATH,
            SECRET_BACKEND,
        ] {
            assert!(!json.contains(private_value));
        }
        assert!(!json.contains("respond_to_allowlist"));
        assert!(!json.contains("managed_agents"));
        assert!(!json.contains("publication_slug"));
        assert!(json.contains("builtin-team:review"));
        assert!(json.contains("Review carefully"));
    }

    #[test]
    fn every_private_field_class_is_rejected() {
        let public_fields = AgentProjectionField::ALL
            .into_iter()
            .filter(|field| field.is_public())
            .collect::<Vec<_>>();
        assert_eq!(validate_public_projection_fields(public_fields), Ok(()));

        for field in AgentProjectionField::ALL
            .into_iter()
            .filter(|field| !field.is_public())
        {
            assert_eq!(
                validate_public_projection_fields([field]),
                Err(AgentProjectionError::PrivateFieldRequested)
            );
        }
    }

    #[test]
    fn embedded_persona_omits_local_slug_and_allowlist() {
        let projection = project_public_agent_catalog(&source())
            .expect("valid private source must produce a public projection");
        let member = &projection.teams[0].members[0];

        assert_eq!(member.persona.display_name, "Reviewer");
        let json = serde_json::to_string(member).expect("member must serialize");
        assert!(!json.contains("slug"));
        assert!(!json.contains("allowlist"));
    }

    #[test]
    fn malformed_owner_and_public_fields_fail_closed() {
        let mut wrong_owner = source();
        wrong_owner.teams[0].owner_public_key = NostrPublicKey::from_bytes([9; 32]);
        assert!(matches!(
            project_public_agent_catalog(&wrong_owner),
            Err(AgentProjectionError::OwnerMismatch)
        ));

        let mut missing_slug = source();
        missing_slug.personas[0].publication_slug = None;
        assert!(matches!(
            project_public_agent_catalog(&missing_slug),
            Err(AgentProjectionError::MissingPersonaSlug)
        ));

        let mut invalid_team = source();
        invalid_team.teams[0].team_id = "has whitespace".to_string();
        assert!(matches!(
            project_public_agent_catalog(&invalid_team),
            Err(AgentProjectionError::InvalidTeamId)
        ));

        let mut unverified_attestation = source();
        unverified_attestation.teams[0].members[0]
            .owner_attestation
            .verified_at = 0;
        assert!(matches!(
            project_public_agent_catalog(&unverified_attestation),
            Err(AgentProjectionError::InvalidOwnerAttestation)
        ));
    }
}
