use crate::{AggregateVersion, CommunityId, NostrPublicKey, ProfileId};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::{collections::BTreeSet, fmt};

const MAX_DISPLAY_NAME_BYTES: usize = 256;
const MAX_ABOUT_BYTES: usize = 4_096;
const MAX_URL_BYTES: usize = 2_048;
const MAX_NIP05_BYTES: usize = 320;
const MAX_NIP05_SCALARS: usize = 320;
const MAX_NIP05_UTF8_BYTES: usize = 1_280;
const MAX_STATUS_BYTES: usize = 1_024;
const MAX_STATUS_KIND_BYTES: usize = 64;
const MAX_ATTESTATION_CONDITIONS_BYTES: usize = 1_024;
const MAX_SOCIAL_LISTS: usize = 256;
const MAX_SOCIAL_ENTRIES: usize = 10_000;
const MAX_SOCIAL_VALUE_BYTES: usize = 2_048;
const MAX_ARCHIVE_STATES: usize = 256;
const MAX_ARCHIVE_REASON_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct NostrEventId([u8; 32]);

impl NostrEventId {
    pub const fn from_bytes(value: [u8; 32]) -> Self {
        Self(value)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AuthoredValue<T> {
    pub source_event_id: NostrEventId,
    pub source_author: NostrPublicKey,
    pub source_created_at: u64,
    pub value: T,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProfileMetadata {
    pub display_name: Option<String>,
    pub name: Option<String>,
    pub avatar_url: Option<String>,
    pub about: Option<String>,
    pub nip05_handle: Option<String>,
}

impl ProfileMetadata {
    pub fn validate(&self) -> Result<(), ProfileError> {
        validate_metadata(self)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStatusKind {
    General,
    Music,
    Custom(String),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProfileStatus {
    pub kind: ProfileStatusKind,
    pub content: String,
    pub expires_at: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SocialListKind {
    Contacts,
    Mutes,
    Pins,
    Bookmarks,
    Emojis,
    NamedFollow(String),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SocialReference {
    PublicKey(NostrPublicKey),
    Event(NostrEventId),
    Coordinate(String),
    Hashtag(String),
    Url(String),
    Word(String),
    Emoji(String),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SocialList {
    pub kind: SocialListKind,
    pub entries: BTreeSet<SocialReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OwnerAttestationEvidence {
    pub owner_public_key: NostrPublicKey,
    pub agent_public_key: NostrPublicKey,
    pub proof_event_id: NostrEventId,
    pub exact_conditions: String,
    pub verified_at: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentProfile {
    pub claimed_owner: Option<NostrPublicKey>,
    pub owner_attestation: Option<OwnerAttestationEvidence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileKind {
    Human,
    Agent(AgentProfile),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelayArchiveStatus {
    Visible,
    Archived,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveConsent {
    SelfRequested {
        actor: NostrPublicKey,
    },
    OwnerAttested {
        owner: NostrPublicKey,
        attestation: OwnerAttestationEvidence,
    },
    Admin {
        actor: NostrPublicKey,
    },
    RelayPolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RelayArchiveRecord {
    pub relay_public_key: NostrPublicKey,
    pub target_public_key: NostrPublicKey,
    pub status: RelayArchiveStatus,
    pub consent: ArchiveConsent,
    pub reason: Option<String>,
    pub replacement_public_key: Option<NostrPublicKey>,
    pub source_event_id: NostrEventId,
    pub source_author: NostrPublicKey,
    pub source_created_at: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProfileRecordFields {
    pub profile_id: ProfileId,
    pub community_id: CommunityId,
    pub author_public_key: NostrPublicKey,
    pub kind: ProfileKind,
    pub metadata: Option<AuthoredValue<ProfileMetadata>>,
    pub statuses: Vec<AuthoredValue<ProfileStatus>>,
    pub social_lists: Vec<AuthoredValue<SocialList>>,
    pub relay_archive_states: Vec<RelayArchiveRecord>,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct IdentityProfile(ProfileRecordFields);

impl IdentityProfile {
    pub fn new(fields: ProfileRecordFields) -> Result<Self, ProfileError> {
        validate_profile(&fields)?;
        Ok(Self(fields))
    }

    pub const fn fields(&self) -> &ProfileRecordFields {
        &self.0
    }

    pub const fn profile_id(&self) -> ProfileId {
        self.0.profile_id
    }

    pub const fn community_id(&self) -> CommunityId {
        self.0.community_id
    }

    pub const fn author_public_key(&self) -> NostrPublicKey {
        self.0.author_public_key
    }

    pub const fn kind(&self) -> &ProfileKind {
        &self.0.kind
    }

    pub const fn version(&self) -> AggregateVersion {
        self.0.version
    }

    pub fn is_archived_on(&self, relay: NostrPublicKey) -> bool {
        self.0.relay_archive_states.iter().any(|state| {
            state.relay_public_key == relay && state.status == RelayArchiveStatus::Archived
        })
    }
}

impl<'de> Deserialize<'de> for IdentityProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = ProfileRecordFields::deserialize(deserializer)?;
        Self::new(fields).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProfileError {
    InvalidMetadata(&'static str),
    InvalidStatus(&'static str),
    InvalidSocialList(&'static str),
    InvalidOwnerAttestation,
    TooManySocialLists,
    DuplicateSocialList,
    TooManyArchiveStates,
    DuplicateRelayArchiveState,
    InvalidArchiveState(&'static str),
    ForeignAuthoredProjection,
    DifferentProfile,
    DifferentCommunity,
    AuthorChanged,
    ProfileKindChanged,
    VersionDoesNotFollow,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata(reason) => {
                write!(formatter, "invalid profile metadata: {reason}")
            }
            Self::InvalidStatus(reason) => write!(formatter, "invalid profile status: {reason}"),
            Self::InvalidSocialList(reason) => write!(formatter, "invalid social list: {reason}"),
            Self::InvalidOwnerAttestation => {
                formatter.write_str("agent owner claim requires matching verified attestation")
            }
            Self::TooManySocialLists => formatter.write_str("too many social lists"),
            Self::DuplicateSocialList => formatter.write_str("duplicate social list kind"),
            Self::TooManyArchiveStates => formatter.write_str("too many relay archive states"),
            Self::DuplicateRelayArchiveState => {
                formatter.write_str("duplicate relay archive state")
            }
            Self::InvalidArchiveState(reason) => {
                write!(formatter, "invalid relay archive state: {reason}")
            }
            Self::ForeignAuthoredProjection => {
                formatter.write_str("profile projection has a different Nostr author")
            }
            Self::DifferentProfile => formatter.write_str("profile id changed during update"),
            Self::DifferentCommunity => {
                formatter.write_str("profile community changed during update")
            }
            Self::AuthorChanged => {
                formatter.write_str("profile Nostr author changed during update")
            }
            Self::ProfileKindChanged => {
                formatter.write_str("human and agent profile kinds cannot be interchanged")
            }
            Self::VersionDoesNotFollow => {
                formatter.write_str("profile version does not follow its predecessor")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

fn validate_profile(fields: &ProfileRecordFields) -> Result<(), ProfileError> {
    validate_kind(&fields.kind, fields.author_public_key)?;
    if let Some(metadata) = &fields.metadata {
        validate_authored(metadata, fields.author_public_key)?;
        validate_metadata(&metadata.value)?;
    }

    let mut status_kinds = BTreeSet::new();
    for status in &fields.statuses {
        validate_authored(status, fields.author_public_key)?;
        validate_status(&status.value)?;
        if status
            .value
            .expires_at
            .is_some_and(|expires_at| expires_at < status.source_created_at)
        {
            return Err(ProfileError::InvalidStatus(
                "expiration precedes source event",
            ));
        }
        if !status_kinds.insert(status.value.kind.clone()) {
            return Err(ProfileError::InvalidStatus("duplicate status kind"));
        }
    }

    if fields.social_lists.len() > MAX_SOCIAL_LISTS {
        return Err(ProfileError::TooManySocialLists);
    }
    let mut social_kinds = BTreeSet::new();
    for list in &fields.social_lists {
        validate_authored(list, fields.author_public_key)?;
        validate_social_list(&list.value)?;
        if !social_kinds.insert(list.value.kind.clone()) {
            return Err(ProfileError::DuplicateSocialList);
        }
    }

    if fields.relay_archive_states.len() > MAX_ARCHIVE_STATES {
        return Err(ProfileError::TooManyArchiveStates);
    }
    let mut relays = BTreeSet::new();
    for archive in &fields.relay_archive_states {
        validate_archive(archive, fields.author_public_key)?;
        if !relays.insert(archive.relay_public_key) {
            return Err(ProfileError::DuplicateRelayArchiveState);
        }
    }
    Ok(())
}

fn validate_kind(kind: &ProfileKind, author: NostrPublicKey) -> Result<(), ProfileError> {
    let ProfileKind::Agent(agent) = kind else {
        return Ok(());
    };
    match (&agent.claimed_owner, &agent.owner_attestation) {
        (None, None) => Ok(()),
        (Some(owner), Some(attestation))
            if valid_owner_attestation(attestation, *owner, author) =>
        {
            Ok(())
        }
        _ => Err(ProfileError::InvalidOwnerAttestation),
    }
}

fn validate_authored<T>(
    value: &AuthoredValue<T>,
    author: NostrPublicKey,
) -> Result<(), ProfileError> {
    if value.source_author != author {
        return Err(ProfileError::ForeignAuthoredProjection);
    }
    Ok(())
}

fn validate_metadata(metadata: &ProfileMetadata) -> Result<(), ProfileError> {
    validate_optional_text(
        metadata.display_name.as_deref(),
        MAX_DISPLAY_NAME_BYTES,
        "display name",
    )?;
    validate_optional_text(metadata.name.as_deref(), MAX_DISPLAY_NAME_BYTES, "name")?;
    validate_optional_text(metadata.avatar_url.as_deref(), MAX_URL_BYTES, "avatar URL")?;
    validate_optional_text(metadata.about.as_deref(), MAX_ABOUT_BYTES, "about")?;
    match metadata.nip05_handle.as_deref() {
        Some(handle) if !handle.is_ascii() => {
            if handle.len() > MAX_NIP05_UTF8_BYTES || handle.chars().count() > MAX_NIP05_SCALARS {
                return Err(ProfileError::InvalidMetadata("NIP-05 handle"));
            }
            Ok(())
        }
        handle => validate_optional_text(handle, MAX_NIP05_BYTES, "NIP-05 handle"),
    }
}

fn validate_optional_text(
    value: Option<&str>,
    maximum: usize,
    field: &'static str,
) -> Result<(), ProfileError> {
    if value.is_some_and(|value| value.len() > maximum) {
        return Err(ProfileError::InvalidMetadata(field));
    }
    Ok(())
}

fn validate_status(status: &ProfileStatus) -> Result<(), ProfileError> {
    if status.content.len() > MAX_STATUS_BYTES {
        return Err(ProfileError::InvalidStatus("content exceeds limit"));
    }
    if let ProfileStatusKind::Custom(kind) = &status.kind {
        if kind.is_empty() || kind.len() > MAX_STATUS_KIND_BYTES {
            return Err(ProfileError::InvalidStatus("invalid custom kind"));
        }
    }
    Ok(())
}

fn validate_social_list(list: &SocialList) -> Result<(), ProfileError> {
    if list.entries.len() > MAX_SOCIAL_ENTRIES {
        return Err(ProfileError::InvalidSocialList("entry count exceeds limit"));
    }
    if let SocialListKind::NamedFollow(name) = &list.kind {
        if name.is_empty() || name.len() > MAX_STATUS_KIND_BYTES {
            return Err(ProfileError::InvalidSocialList("invalid named follow list"));
        }
    }
    for entry in &list.entries {
        let compatible = match &list.kind {
            SocialListKind::Contacts | SocialListKind::NamedFollow(_) => {
                matches!(entry, SocialReference::PublicKey(_))
            }
            SocialListKind::Mutes => matches!(
                entry,
                SocialReference::PublicKey(_)
                    | SocialReference::Event(_)
                    | SocialReference::Coordinate(_)
                    | SocialReference::Hashtag(_)
                    | SocialReference::Word(_)
            ),
            SocialListKind::Pins => matches!(
                entry,
                SocialReference::Event(_) | SocialReference::Coordinate(_)
            ),
            SocialListKind::Bookmarks => matches!(
                entry,
                SocialReference::Event(_)
                    | SocialReference::Coordinate(_)
                    | SocialReference::Hashtag(_)
                    | SocialReference::Url(_)
            ),
            SocialListKind::Emojis => matches!(
                entry,
                SocialReference::Emoji(_) | SocialReference::Coordinate(_)
            ),
        };
        if !compatible || social_value_too_long(entry) {
            return Err(ProfileError::InvalidSocialList(
                "entry is incompatible or exceeds limit",
            ));
        }
    }
    Ok(())
}

fn social_value_too_long(reference: &SocialReference) -> bool {
    match reference {
        SocialReference::Coordinate(value)
        | SocialReference::Hashtag(value)
        | SocialReference::Url(value)
        | SocialReference::Word(value)
        | SocialReference::Emoji(value) => value.is_empty() || value.len() > MAX_SOCIAL_VALUE_BYTES,
        SocialReference::PublicKey(_) | SocialReference::Event(_) => false,
    }
}

fn validate_archive(
    archive: &RelayArchiveRecord,
    target: NostrPublicKey,
) -> Result<(), ProfileError> {
    if archive.target_public_key != target || archive.source_author != archive.relay_public_key {
        return Err(ProfileError::InvalidArchiveState(
            "target or relay signature does not match profile scope",
        ));
    }
    if archive
        .reason
        .as_ref()
        .is_some_and(|reason| reason.len() > MAX_ARCHIVE_REASON_BYTES)
    {
        return Err(ProfileError::InvalidArchiveState("reason exceeds limit"));
    }
    if archive.status == RelayArchiveStatus::Visible && archive.replacement_public_key.is_some() {
        return Err(ProfileError::InvalidArchiveState(
            "visible state cannot name a replacement",
        ));
    }
    match &archive.consent {
        ArchiveConsent::SelfRequested { actor } if *actor != target => Err(
            ProfileError::InvalidArchiveState("self actor is not target"),
        ),
        ArchiveConsent::OwnerAttested { owner, attestation }
            if !valid_owner_attestation(attestation, *owner, target) =>
        {
            Err(ProfileError::InvalidArchiveState(
                "owner consent has no matching attestation",
            ))
        }
        _ => Ok(()),
    }
}

fn valid_owner_attestation(
    attestation: &OwnerAttestationEvidence,
    owner: NostrPublicKey,
    agent: NostrPublicKey,
) -> bool {
    owner != agent
        && attestation.owner_public_key == owner
        && attestation.agent_public_key == agent
        && attestation.exact_conditions.len() <= MAX_ATTESTATION_CONDITIONS_BYTES
}

pub fn validate_profile_update(
    previous: &IdentityProfile,
    next: &IdentityProfile,
) -> Result<(), ProfileError> {
    if previous.profile_id() != next.profile_id() {
        return Err(ProfileError::DifferentProfile);
    }
    if previous.community_id() != next.community_id() {
        return Err(ProfileError::DifferentCommunity);
    }
    if previous.author_public_key() != next.author_public_key() {
        return Err(ProfileError::AuthorChanged);
    }
    if std::mem::discriminant(previous.kind()) != std::mem::discriminant(next.kind()) {
        return Err(ProfileError::ProfileKindChanged);
    }
    if !next.version().follows(previous.version()) {
        return Err(ProfileError::VersionDoesNotFollow);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn key(byte: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([byte; 32])
    }

    fn event(byte: u8) -> NostrEventId {
        NostrEventId::from_bytes([byte; 32])
    }

    fn fields(kind: ProfileKind) -> ProfileRecordFields {
        let author = key(1);
        ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(1)),
            community_id: CommunityId::from_uuid(Uuid::from_u128(2)),
            author_public_key: author,
            kind,
            metadata: Some(AuthoredValue {
                source_event_id: event(3),
                source_author: author,
                source_created_at: 10,
                value: ProfileMetadata {
                    display_name: Some("Builder".into()),
                    ..ProfileMetadata::default()
                },
            }),
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        }
    }

    #[test]
    fn profile_metadata_validation_retains_existing_byte_budgets() {
        for (limit, metadata) in [
            (
                MAX_DISPLAY_NAME_BYTES,
                ProfileMetadata {
                    display_name: Some("x".repeat(MAX_DISPLAY_NAME_BYTES)),
                    ..ProfileMetadata::default()
                },
            ),
            (
                MAX_DISPLAY_NAME_BYTES,
                ProfileMetadata {
                    name: Some("x".repeat(MAX_DISPLAY_NAME_BYTES)),
                    ..ProfileMetadata::default()
                },
            ),
            (
                MAX_URL_BYTES,
                ProfileMetadata {
                    avatar_url: Some("x".repeat(MAX_URL_BYTES)),
                    ..ProfileMetadata::default()
                },
            ),
            (
                MAX_ABOUT_BYTES,
                ProfileMetadata {
                    about: Some("x".repeat(MAX_ABOUT_BYTES)),
                    ..ProfileMetadata::default()
                },
            ),
            (
                MAX_NIP05_BYTES,
                ProfileMetadata {
                    nip05_handle: Some("x".repeat(MAX_NIP05_BYTES)),
                    ..ProfileMetadata::default()
                },
            ),
        ] {
            assert!(metadata.validate().is_ok());
            let mut oversized = metadata;
            for value in [
                &mut oversized.display_name,
                &mut oversized.name,
                &mut oversized.avatar_url,
                &mut oversized.about,
                &mut oversized.nip05_handle,
            ]
            .into_iter()
            .flatten()
            {
                assert_eq!(value.len(), limit);
                value.push('x');
            }
            assert!(matches!(
                oversized.validate(),
                Err(ProfileError::InvalidMetadata(_))
            ));
        }
        let utf8 = ProfileMetadata {
            name: Some("é".repeat(MAX_DISPLAY_NAME_BYTES / 2)),
            ..ProfileMetadata::default()
        };
        assert!(utf8.validate().is_ok());
        let utf8 = ProfileMetadata {
            name: Some("é".repeat(MAX_DISPLAY_NAME_BYTES / 2 + 1)),
            ..ProfileMetadata::default()
        };
        assert!(matches!(
            utf8.validate(),
            Err(ProfileError::InvalidMetadata("name"))
        ));
    }

    #[test]
    fn profile_metadata_nip05_accepts_bounded_unicode_without_broadening_other_fields() {
        for handle in [
            "é".repeat(MAX_NIP05_SCALARS),
            "🦀".repeat(MAX_NIP05_SCALARS),
        ] {
            assert!(handle.len() > MAX_NIP05_BYTES);
            assert!(handle.len() <= MAX_NIP05_UTF8_BYTES);
            assert_eq!(handle.chars().count(), MAX_NIP05_SCALARS);
            assert!(
                ProfileMetadata {
                    nip05_handle: Some(handle),
                    ..ProfileMetadata::default()
                }
                .validate()
                .is_ok()
            );
        }
        for handle in [
            "é".repeat(MAX_NIP05_SCALARS + 1),
            "🦀".repeat(MAX_NIP05_SCALARS + 1),
        ] {
            assert_eq!(
                ProfileMetadata {
                    nip05_handle: Some(handle),
                    ..ProfileMetadata::default()
                }
                .validate(),
                Err(ProfileError::InvalidMetadata("NIP-05 handle"))
            );
        }
        assert_eq!(
            ProfileMetadata {
                name: Some("é".repeat(MAX_DISPLAY_NAME_BYTES / 2 + 1)),
                nip05_handle: Some("🦀".repeat(MAX_NIP05_SCALARS)),
                ..ProfileMetadata::default()
            }
            .validate(),
            Err(ProfileError::InvalidMetadata("name"))
        );
    }

    fn human() -> IdentityProfile {
        IdentityProfile::new(fields(ProfileKind::Human)).expect("valid human profile")
    }

    fn attestation(owner: NostrPublicKey, agent: NostrPublicKey) -> OwnerAttestationEvidence {
        OwnerAttestationEvidence {
            owner_public_key: owner,
            agent_public_key: agent,
            proof_event_id: event(4),
            exact_conditions: "kind=0".into(),
            verified_at: 20,
        }
    }

    #[test]
    fn profile_human_and_agent_authorship_remain_distinct_from_ownership() {
        let human = human();
        let owner = key(2);
        let agent_author = human.author_public_key();
        let agent = IdentityProfile::new(fields(ProfileKind::Agent(AgentProfile {
            claimed_owner: Some(owner),
            owner_attestation: Some(attestation(owner, agent_author)),
        })))
        .expect("valid owner-attested agent");

        assert_eq!(agent.author_public_key(), agent_author);
        assert_ne!(agent.author_public_key(), owner);
        assert!(matches!(agent.kind(), ProfileKind::Agent(_)));
    }

    #[test]
    fn profile_rejects_unattested_or_mismatched_agent_owner_changes() {
        let owner = key(2);
        let mut missing = fields(ProfileKind::Agent(AgentProfile {
            claimed_owner: Some(owner),
            owner_attestation: None,
        }));
        assert_eq!(
            IdentityProfile::new(missing.clone()),
            Err(ProfileError::InvalidOwnerAttestation)
        );

        missing.kind = ProfileKind::Agent(AgentProfile {
            claimed_owner: Some(owner),
            owner_attestation: Some(attestation(owner, key(9))),
        });
        assert_eq!(
            IdentityProfile::new(missing),
            Err(ProfileError::InvalidOwnerAttestation)
        );
    }

    #[test]
    fn profile_archive_is_relay_scoped_and_never_rewrites_history() {
        let mut archived_fields = fields(ProfileKind::Human);
        let author = archived_fields.author_public_key;
        let relay = key(7);
        archived_fields
            .relay_archive_states
            .push(RelayArchiveRecord {
                relay_public_key: relay,
                target_public_key: author,
                status: RelayArchiveStatus::Archived,
                consent: ArchiveConsent::SelfRequested { actor: author },
                reason: Some("retired".into()),
                replacement_public_key: None,
                source_event_id: event(8),
                source_author: relay,
                source_created_at: 30,
            });
        let archived = IdentityProfile::new(archived_fields).expect("valid archive state");

        assert!(archived.is_archived_on(relay));
        assert!(!archived.is_archived_on(key(8)));
        assert_eq!(archived.author_public_key(), author);

        let mut replacement_fields = archived.fields().clone();
        replacement_fields.author_public_key = key(9);
        replacement_fields.version = archived.version().next().expect("next version");
        replacement_fields.metadata = None;
        replacement_fields.relay_archive_states.clear();
        let replacement =
            IdentityProfile::new(replacement_fields).expect("independently valid profile");
        assert_eq!(
            validate_profile_update(&archived, &replacement),
            Err(ProfileError::AuthorChanged)
        );
    }

    #[test]
    fn profile_status_and_social_lists_retain_signed_author_scope() {
        let mut profile_fields = fields(ProfileKind::Human);
        let author = profile_fields.author_public_key;
        profile_fields.statuses.push(AuthoredValue {
            source_event_id: event(5),
            source_author: author,
            source_created_at: 11,
            value: ProfileStatus {
                kind: ProfileStatusKind::General,
                content: "building".into(),
                expires_at: None,
            },
        });
        profile_fields.social_lists.push(AuthoredValue {
            source_event_id: event(6),
            source_author: author,
            source_created_at: 12,
            value: SocialList {
                kind: SocialListKind::Contacts,
                entries: BTreeSet::from([SocialReference::PublicKey(key(2))]),
            },
        });
        IdentityProfile::new(profile_fields.clone()).expect("valid authored projections");

        profile_fields.social_lists[0].source_author = key(9);
        assert_eq!(
            IdentityProfile::new(profile_fields),
            Err(ProfileError::ForeignAuthoredProjection)
        );
    }
}
