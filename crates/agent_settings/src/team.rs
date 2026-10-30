use std::{collections::BTreeMap, error::Error, fmt};

const MAX_TEAM_ID_CHARACTERS: usize = 64;
const MAX_ROLE_BYTES: usize = 32;
const MAX_NAME_BYTES: usize = 256;
const MAX_DESCRIPTION_BYTES: usize = 4_096;
const MAX_INSTRUCTIONS_BYTES: usize = 262_144;
const MAX_MEMBERS: usize = 256;
const MAX_CATALOG_ENTRIES: usize = 1_024;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NostrPublicKey(String);

impl NostrPublicKey {
    pub fn parse(value: impl Into<String>) -> Result<Self, TeamRecordError> {
        let value = value.into();
        validate_lower_hex(&value)
            .then_some(Self(value))
            .ok_or(TeamRecordError::InvalidPublicKey)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NostrEventId(String);

impl NostrEventId {
    pub fn parse(value: impl Into<String>) -> Result<Self, TeamRecordError> {
        let value = value.into();
        validate_lower_hex(&value)
            .then_some(Self(value))
            .ok_or(TeamRecordError::InvalidEventId)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnerAttestationRecord {
    pub owner_public_key: NostrPublicKey,
    pub agent_public_key: NostrPublicKey,
    pub proof_event_id: NostrEventId,
    pub exact_conditions: String,
    pub verified_at: u64,
}

impl OwnerAttestationRecord {
    pub fn new(
        owner_public_key: NostrPublicKey,
        agent_public_key: NostrPublicKey,
        proof_event_id: NostrEventId,
        exact_conditions: impl Into<String>,
        verified_at: u64,
    ) -> Result<Self, TeamRecordError> {
        let exact_conditions = exact_conditions.into();
        if owner_public_key == agent_public_key
            || exact_conditions.len() > 4_096
            || exact_conditions.contains('\0')
            || verified_at == 0
        {
            return Err(TeamRecordError::InvalidOwnerAttestation);
        }
        Ok(Self {
            owner_public_key,
            agent_public_key,
            proof_event_id,
            exact_conditions,
            verified_at,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentIdentityStatus {
    Active,
    Revoked { revoked_at: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentIdentityRecord {
    pub public_key: NostrPublicKey,
    pub owner_attestation: OwnerAttestationRecord,
    pub status: AgentIdentityStatus,
}

impl AgentIdentityRecord {
    pub fn new(
        public_key: NostrPublicKey,
        owner_attestation: OwnerAttestationRecord,
        status: AgentIdentityStatus,
    ) -> Result<Self, TeamRecordError> {
        if owner_attestation.agent_public_key != public_key
            || matches!(status, AgentIdentityStatus::Revoked { revoked_at: 0 })
        {
            return Err(TeamRecordError::InvalidOwnerAttestation);
        }
        Ok(Self {
            public_key,
            owner_attestation,
            status,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PersonaReference {
    Local {
        persona_id: String,
    },
    Published {
        owner_public_key: NostrPublicKey,
        slug: String,
    },
}

impl PersonaReference {
    pub fn local(persona_id: impl Into<String>) -> Result<Self, TeamRecordError> {
        let persona_id = persona_id.into();
        validate_team_coordinate(&persona_id)?;
        Ok(Self::Local { persona_id })
    }

    pub fn published(
        owner_public_key: NostrPublicKey,
        slug: impl Into<String>,
    ) -> Result<Self, TeamRecordError> {
        let slug = slug.into();
        if !valid_persona_slug(&slug) {
            return Err(TeamRecordError::InvalidPersonaSlug);
        }
        Ok(Self::Published {
            owner_public_key,
            slug,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TeamRole(String);

impl TeamRole {
    pub fn parse(value: impl Into<String>) -> Result<Self, TeamRecordError> {
        let value = value.into();
        let mut characters = value.chars();
        let Some(first) = characters.next() else {
            return Err(TeamRecordError::InvalidRole);
        };
        if value.len() > MAX_ROLE_BYTES
            || !first.is_ascii_lowercase()
            || characters.any(|character| {
                !character.is_ascii_lowercase()
                    && !character.is_ascii_digit()
                    && character != '_'
                    && character != '-'
            })
        {
            return Err(TeamRecordError::InvalidRole);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentTeamMember {
    pub identity: AgentIdentityRecord,
    pub persona: PersonaReference,
    pub role: TeamRole,
}

#[derive(Clone, PartialEq)]
pub struct AgentTeamRecord {
    pub team_id: String,
    pub owner_public_key: NostrPublicKey,
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub members: Vec<AgentTeamMember>,
}

impl fmt::Debug for AgentTeamRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentTeamRecord")
            .field("team_id", &self.team_id)
            .field("owner_public_key", &self.owner_public_key)
            .field("name", &self.name)
            .field("description", &self.description)
            .field(
                "instructions",
                &self.instructions.as_ref().map(|_| "<redacted>"),
            )
            .field("members", &self.members)
            .finish()
    }
}

impl AgentTeamRecord {
    pub fn new(
        team_id: impl Into<String>,
        owner_public_key: NostrPublicKey,
        name: impl Into<String>,
        description: Option<String>,
        instructions: Option<String>,
        members: Vec<AgentTeamMember>,
    ) -> Result<Self, TeamRecordError> {
        let team = Self {
            team_id: team_id.into(),
            owner_public_key,
            name: name.into(),
            description,
            instructions,
            members,
        };
        team.validate()?;
        Ok(team)
    }

    pub fn change_owner(
        &self,
        new_owner: NostrPublicKey,
        replacement_attestations: Vec<OwnerAttestationRecord>,
    ) -> Result<Self, TeamRecordError> {
        let mut replacements = BTreeMap::new();
        for attestation in replacement_attestations {
            if attestation.owner_public_key != new_owner {
                return Err(TeamRecordError::OwnerMismatch);
            }
            let agent = attestation.agent_public_key.clone();
            if replacements.insert(agent.clone(), attestation).is_some() {
                return Err(TeamRecordError::DuplicateOwnerAttestation(agent));
            }
        }

        let mut members = Vec::with_capacity(self.members.len());
        for member in &self.members {
            let attestation = replacements
                .remove(&member.identity.public_key)
                .ok_or_else(|| {
                    TeamRecordError::MissingOwnerAttestation(member.identity.public_key.clone())
                })?;
            let identity = AgentIdentityRecord::new(
                member.identity.public_key.clone(),
                attestation,
                member.identity.status,
            )?;
            members.push(AgentTeamMember {
                identity,
                persona: member.persona.clone(),
                role: member.role.clone(),
            });
        }
        if let Some(unused) = replacements.into_keys().next() {
            return Err(TeamRecordError::UnexpectedOwnerAttestation(unused));
        }

        Self::new(
            self.team_id.clone(),
            new_owner,
            self.name.clone(),
            self.description.clone(),
            self.instructions.clone(),
            members,
        )
    }

    fn validate(&self) -> Result<(), TeamRecordError> {
        validate_team_coordinate(&self.team_id)?;
        validate_required_text(&self.name, MAX_NAME_BYTES, "team name")?;
        validate_optional_text(
            self.description.as_deref(),
            MAX_DESCRIPTION_BYTES,
            "team description",
        )?;
        validate_optional_text(
            self.instructions.as_deref(),
            MAX_INSTRUCTIONS_BYTES,
            "team instructions",
        )?;
        if self.members.len() > MAX_MEMBERS {
            return Err(TeamRecordError::TooManyMembers);
        }

        let mut identities = BTreeMap::new();
        for member in &self.members {
            if identities
                .insert(member.identity.public_key.clone(), ())
                .is_some()
            {
                return Err(TeamRecordError::DuplicateMember(
                    member.identity.public_key.clone(),
                ));
            }
            if !matches!(member.identity.status, AgentIdentityStatus::Active) {
                return Err(TeamRecordError::RevokedIdentity(
                    member.identity.public_key.clone(),
                ));
            }
            if member.identity.owner_attestation.owner_public_key != self.owner_public_key
                || member.identity.owner_attestation.agent_public_key != member.identity.public_key
            {
                return Err(TeamRecordError::OwnerMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub struct PublicPersonaShareRecord {
    pub source: PersonaReference,
    pub display_name: String,
    pub description: String,
    pub system_prompt: Option<String>,
    pub avatar_url: Option<String>,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

impl fmt::Debug for PublicPersonaShareRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicPersonaShareRecord")
            .field("source", &self.source)
            .field("display_name", &self.display_name)
            .field("description", &self.description)
            .field(
                "system_prompt",
                &self.system_prompt.as_ref().map(|_| "<redacted>"),
            )
            .field("avatar_url", &self.avatar_url)
            .field("runtime", &self.runtime)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .finish()
    }
}

impl PublicPersonaShareRecord {
    pub fn new(
        source: PersonaReference,
        display_name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: Option<String>,
        avatar_url: Option<String>,
        runtime: Option<String>,
        model: Option<String>,
        provider: Option<String>,
    ) -> Result<Self, TeamRecordError> {
        let record = Self {
            source,
            display_name: display_name.into(),
            description: description.into(),
            system_prompt,
            avatar_url,
            runtime,
            model,
            provider,
        };
        validate_required_text(&record.display_name, MAX_NAME_BYTES, "persona display name")?;
        validate_required_text(
            &record.description,
            MAX_DESCRIPTION_BYTES,
            "persona description",
        )?;
        validate_optional_text(
            record.system_prompt.as_deref(),
            MAX_INSTRUCTIONS_BYTES,
            "persona system prompt",
        )?;
        for (value, field) in [
            (record.avatar_url.as_deref(), "persona avatar URL"),
            (record.runtime.as_deref(), "persona runtime"),
            (record.model.as_deref(), "persona model"),
            (record.provider.as_deref(), "persona provider"),
        ] {
            validate_optional_text(value, MAX_DESCRIPTION_BYTES, field)?;
        }
        Ok(record)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicTeamMemberShareRecord {
    pub agent_public_key: NostrPublicKey,
    pub owner_attestation: OwnerAttestationRecord,
    pub role: TeamRole,
    pub persona: PublicPersonaShareRecord,
}

#[derive(Clone, PartialEq)]
pub struct PublicTeamShareRecord {
    pub team_id: String,
    pub owner_public_key: NostrPublicKey,
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub members: Vec<PublicTeamMemberShareRecord>,
}

impl fmt::Debug for PublicTeamShareRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicTeamShareRecord")
            .field("team_id", &self.team_id)
            .field("owner_public_key", &self.owner_public_key)
            .field("name", &self.name)
            .field("description", &self.description)
            .field(
                "instructions",
                &self.instructions.as_ref().map(|_| "<redacted>"),
            )
            .field("members", &self.members)
            .finish()
    }
}

impl PublicTeamShareRecord {
    pub fn new(
        team_id: impl Into<String>,
        owner_public_key: NostrPublicKey,
        name: impl Into<String>,
        description: Option<String>,
        instructions: Option<String>,
        members: Vec<PublicTeamMemberShareRecord>,
    ) -> Result<Self, TeamRecordError> {
        let team_id = team_id.into();
        validate_team_coordinate(&team_id)?;
        let record = Self {
            team_id,
            owner_public_key,
            name: name.into(),
            description,
            instructions,
            members,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<(), TeamRecordError> {
        validate_team_coordinate(&self.team_id)?;
        validate_required_text(&self.name, MAX_NAME_BYTES, "public team name")?;
        validate_optional_text(
            self.description.as_deref(),
            MAX_DESCRIPTION_BYTES,
            "public team description",
        )?;
        validate_optional_text(
            self.instructions.as_deref(),
            MAX_INSTRUCTIONS_BYTES,
            "public team instructions",
        )?;
        if self.members.len() > MAX_MEMBERS {
            return Err(TeamRecordError::TooManyMembers);
        }
        let mut members = BTreeMap::new();
        for member in &self.members {
            if members
                .insert(member.agent_public_key.clone(), ())
                .is_some()
            {
                return Err(TeamRecordError::DuplicateMember(
                    member.agent_public_key.clone(),
                ));
            }
            if member.owner_attestation.owner_public_key != self.owner_public_key
                || member.owner_attestation.agent_public_key != member.agent_public_key
            {
                return Err(TeamRecordError::OwnerMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicAgentCatalogRecord {
    pub owner_public_key: NostrPublicKey,
    pub personas: Vec<PublicPersonaShareRecord>,
    pub teams: Vec<PublicTeamShareRecord>,
}

impl PublicAgentCatalogRecord {
    pub fn new(
        owner_public_key: NostrPublicKey,
        personas: Vec<PublicPersonaShareRecord>,
        teams: Vec<PublicTeamShareRecord>,
    ) -> Result<Self, TeamRecordError> {
        if personas.len().saturating_add(teams.len()) > MAX_CATALOG_ENTRIES {
            return Err(TeamRecordError::TooManyCatalogEntries);
        }
        let mut persona_sources = BTreeMap::new();
        for persona in &personas {
            if persona_sources.insert(persona.source.clone(), ()).is_some() {
                return Err(TeamRecordError::DuplicateCatalogPersona(
                    persona.source.clone(),
                ));
            }
        }
        let mut team_ids = BTreeMap::new();
        for team in &teams {
            if team.owner_public_key != owner_public_key {
                return Err(TeamRecordError::OwnerMismatch);
            }
            team.validate()?;
            if team_ids.insert(team.team_id.clone(), ()).is_some() {
                return Err(TeamRecordError::DuplicateCatalogTeam(team.team_id.clone()));
            }
        }
        Ok(Self {
            owner_public_key,
            personas,
            teams,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TeamRecordError {
    InvalidPublicKey,
    InvalidEventId,
    InvalidOwnerAttestation,
    InvalidTeamCoordinate,
    InvalidPersonaSlug,
    InvalidRole,
    InvalidText(&'static str),
    TooManyMembers,
    TooManyCatalogEntries,
    DuplicateMember(NostrPublicKey),
    RevokedIdentity(NostrPublicKey),
    OwnerMismatch,
    MissingOwnerAttestation(NostrPublicKey),
    DuplicateOwnerAttestation(NostrPublicKey),
    UnexpectedOwnerAttestation(NostrPublicKey),
    DuplicateCatalogPersona(PersonaReference),
    DuplicateCatalogTeam(String),
}

impl fmt::Display for TeamRecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKey => write!(formatter, "invalid Nostr public key"),
            Self::InvalidEventId => write!(formatter, "invalid Nostr event id"),
            Self::InvalidOwnerAttestation => write!(formatter, "invalid owner attestation"),
            Self::InvalidTeamCoordinate => write!(formatter, "invalid team coordinate"),
            Self::InvalidPersonaSlug => write!(formatter, "invalid persona slug"),
            Self::InvalidRole => write!(formatter, "invalid team role"),
            Self::InvalidText(field) => write!(formatter, "invalid {field}"),
            Self::TooManyMembers => write!(formatter, "team has too many members"),
            Self::TooManyCatalogEntries => write!(formatter, "catalog has too many entries"),
            Self::DuplicateMember(_) => write!(formatter, "team contains a duplicate member"),
            Self::RevokedIdentity(_) => write!(formatter, "team contains a revoked identity"),
            Self::OwnerMismatch => write!(formatter, "record owner does not match attestation"),
            Self::MissingOwnerAttestation(_) => {
                write!(formatter, "owner change is missing a member attestation")
            }
            Self::DuplicateOwnerAttestation(_) => {
                write!(formatter, "owner change contains a duplicate attestation")
            }
            Self::UnexpectedOwnerAttestation(_) => {
                write!(formatter, "owner change contains an unexpected attestation")
            }
            Self::DuplicateCatalogPersona(_) => {
                write!(formatter, "catalog contains a duplicate persona")
            }
            Self::DuplicateCatalogTeam(_) => write!(formatter, "catalog contains a duplicate team"),
        }
    }
}

impl Error for TeamRecordError {}

fn validate_lower_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_team_coordinate(value: &str) -> Result<(), TeamRecordError> {
    if value.is_empty()
        || value.chars().count() > MAX_TEAM_ID_CHARACTERS
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(TeamRecordError::InvalidTeamCoordinate);
    }
    Ok(())
}

fn valid_persona_slug(value: &str) -> bool {
    let mut characters = value.chars();
    characters.next().is_some_and(|first| {
        (first.is_ascii_lowercase() || first.is_ascii_digit())
            && value.len() <= 64
            && characters.all(|character| {
                character.is_ascii_lowercase()
                    || character.is_ascii_digit()
                    || character == '_'
                    || character == '-'
            })
    })
}

fn validate_required_text(
    value: &str,
    maximum: usize,
    field: &'static str,
) -> Result<(), TeamRecordError> {
    if value.trim().is_empty() || value.len() > maximum || value.contains('\0') {
        return Err(TeamRecordError::InvalidText(field));
    }
    Ok(())
}

fn validate_optional_text(
    value: Option<&str>,
    maximum: usize,
    field: &'static str,
) -> Result<(), TeamRecordError> {
    if value.is_some_and(|value| value.len() > maximum || value.contains('\0')) {
        return Err(TeamRecordError::InvalidText(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER_ONE: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const OWNER_TWO: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const AGENT_ONE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const AGENT_TWO: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const EVENT_ONE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const EVENT_TWO: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn public_key(value: &str) -> NostrPublicKey {
        NostrPublicKey::parse(value).expect("test public key")
    }

    fn event_id(value: &str) -> NostrEventId {
        NostrEventId::parse(value).expect("test event id")
    }

    fn identity(
        owner: &str,
        agent: &str,
        proof: &str,
        status: AgentIdentityStatus,
    ) -> AgentIdentityRecord {
        let attestation = OwnerAttestationRecord::new(
            public_key(owner),
            public_key(agent),
            event_id(proof),
            "",
            1,
        )
        .expect("test attestation");
        AgentIdentityRecord::new(public_key(agent), attestation, status).expect("test identity")
    }

    fn member(agent: &str, proof: &str) -> AgentTeamMember {
        AgentTeamMember {
            identity: identity(OWNER_ONE, agent, proof, AgentIdentityStatus::Active),
            persona: PersonaReference::local("builtin:fizz").expect("persona reference"),
            role: TeamRole::parse("reviewer").expect("role"),
        }
    }

    #[test]
    fn duplicate_member_is_rejected() {
        let duplicate = member(AGENT_ONE, EVENT_ONE);
        let result = AgentTeamRecord::new(
            "builtin-team:review",
            public_key(OWNER_ONE),
            "Review",
            None,
            None,
            vec![duplicate.clone(), duplicate],
        );

        assert_eq!(
            result,
            Err(TeamRecordError::DuplicateMember(public_key(AGENT_ONE)))
        );
    }

    #[test]
    fn revoked_identity_is_rejected() {
        let revoked = AgentTeamMember {
            identity: identity(
                OWNER_ONE,
                AGENT_ONE,
                EVENT_ONE,
                AgentIdentityStatus::Revoked { revoked_at: 2 },
            ),
            persona: PersonaReference::local("builtin:fizz").expect("persona reference"),
            role: TeamRole::parse("reviewer").expect("role"),
        };

        let result = AgentTeamRecord::new(
            "team-1",
            public_key(OWNER_ONE),
            "Review",
            None,
            None,
            vec![revoked],
        );
        assert_eq!(
            result,
            Err(TeamRecordError::RevokedIdentity(public_key(AGENT_ONE)))
        );
    }

    #[test]
    fn public_catalog_accepts_embedded_team_personas() {
        let persona = PublicPersonaShareRecord::new(
            PersonaReference::published(public_key(OWNER_ONE), "reviewer")
                .expect("published persona"),
            "Reviewer",
            "Reviews changes",
            Some("Review carefully".to_string()),
            None,
            Some("goose".to_string()),
            Some("model".to_string()),
            Some("provider".to_string()),
        )
        .expect("public persona");
        let identity = identity(OWNER_ONE, AGENT_ONE, EVENT_ONE, AgentIdentityStatus::Active);
        let team = PublicTeamShareRecord::new(
            "builtin-team:review",
            public_key(OWNER_ONE),
            "Review Team",
            Some("A shared team".to_string()),
            Some("Coordinate reviews".to_string()),
            vec![PublicTeamMemberShareRecord {
                agent_public_key: identity.public_key,
                owner_attestation: identity.owner_attestation,
                role: TeamRole::parse("reviewer").expect("role"),
                persona: persona.clone(),
            }],
        )
        .expect("public team");
        let catalog =
            PublicAgentCatalogRecord::new(public_key(OWNER_ONE), vec![persona], vec![team])
                .expect("public catalog");

        assert_eq!(catalog.personas.len(), 1);
        assert_eq!(catalog.teams[0].members.len(), 1);
        assert_eq!(catalog.teams[0].team_id, "builtin-team:review");
    }

    #[test]
    fn owner_change_requires_exact_replacement_attestations() {
        let team = AgentTeamRecord::new(
            "team-1",
            public_key(OWNER_ONE),
            "Review",
            None,
            Some("Coordinate".to_string()),
            vec![member(AGENT_ONE, EVENT_ONE), member(AGENT_TWO, EVENT_TWO)],
        )
        .expect("team");
        let replacement_one = OwnerAttestationRecord::new(
            public_key(OWNER_TWO),
            public_key(AGENT_ONE),
            event_id(EVENT_TWO),
            "",
            3,
        )
        .expect("replacement attestation");
        assert_eq!(
            team.change_owner(public_key(OWNER_TWO), vec![replacement_one.clone()]),
            Err(TeamRecordError::MissingOwnerAttestation(public_key(
                AGENT_TWO
            )))
        );

        let replacement_two = OwnerAttestationRecord::new(
            public_key(OWNER_TWO),
            public_key(AGENT_TWO),
            event_id(EVENT_ONE),
            "",
            3,
        )
        .expect("replacement attestation");
        let changed = team
            .change_owner(
                public_key(OWNER_TWO),
                vec![replacement_two, replacement_one],
            )
            .expect("owner change");

        assert_eq!(changed.owner_public_key, public_key(OWNER_TWO));
        assert!(changed.members.iter().all(|member| {
            member.identity.owner_attestation.owner_public_key == public_key(OWNER_TWO)
        }));
    }
}
