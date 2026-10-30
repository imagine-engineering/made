use std::{collections::BTreeSet, error::Error, fmt};

use crate::{MessageSource, NostrPublicKey};

const REPOSITORY_KIND: &str = "30617";
const MAX_PROJECT_SLUG_BYTES: usize = 1_024;
const MAX_PROJECT_NAME_BYTES: usize = 256;
const MAX_PROJECT_DESCRIPTION_BYTES: usize = 2_048;
const MAX_PROJECT_CHANNEL_BYTES: usize = 256;
const MAX_PROJECT_MEMBERS: usize = 64;
const MAX_REPOSITORY_DISCRIMINATOR_BYTES: usize = 1_024;
const MAX_RELAY_HINT_BYTES: usize = 512 * 1_024;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProjectSlug(String);

impl ProjectSlug {
    pub fn new(value: impl Into<String>) -> Result<Self, ProjectGroupError> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_PROJECT_SLUG_BYTES {
            return Err(ProjectGroupError::InvalidSlug);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectDisplayName(String);

impl ProjectDisplayName {
    pub fn new(value: impl Into<String>) -> Result<Self, ProjectGroupError> {
        let value = value.into();
        if value.len() > MAX_PROJECT_NAME_BYTES {
            return Err(ProjectGroupError::InvalidName);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectDescription(String);

impl ProjectDescription {
    pub fn new(value: impl Into<String>) -> Result<Self, ProjectGroupError> {
        let value = value.into();
        if value.len() > MAX_PROJECT_DESCRIPTION_BYTES {
            return Err(ProjectGroupError::InvalidDescription);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectChannelReference(String);

impl ProjectChannelReference {
    pub fn new(value: impl Into<String>) -> Result<Self, ProjectGroupError> {
        let value = value.into();
        if value.len() > MAX_PROJECT_CHANNEL_BYTES {
            return Err(ProjectGroupError::InvalidChannelReference);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectVisibility {
    Listed,
    Unlisted,
}

impl ProjectVisibility {
    pub fn from_metadata(value: Option<&str>) -> Self {
        if value == Some("unlisted") {
            Self::Unlisted
        } else {
            Self::Listed
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RepositoryCoordinate {
    owner_public_key: NostrPublicKey,
    discriminator: String,
    relay_hint: Option<String>,
}

impl RepositoryCoordinate {
    pub fn parse(coordinate: &str, relay_hint: Option<String>) -> Result<Self, ProjectGroupError> {
        let mut parts = coordinate.splitn(3, ':');
        if parts.next() != Some(REPOSITORY_KIND) {
            return Err(ProjectGroupError::InvalidRepositoryCoordinate);
        }
        let owner = parts
            .next()
            .ok_or(ProjectGroupError::InvalidRepositoryCoordinate)?;
        let discriminator = parts
            .next()
            .ok_or(ProjectGroupError::InvalidRepositoryCoordinate)?;
        if owner.len() != 64
            || !owner
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || discriminator.is_empty()
            || discriminator.len() > MAX_REPOSITORY_DISCRIMINATOR_BYTES
            || relay_hint
                .as_ref()
                .is_some_and(|relay_hint| relay_hint.len() > MAX_RELAY_HINT_BYTES)
        {
            return Err(ProjectGroupError::InvalidRepositoryCoordinate);
        }
        let mut owner_bytes = [0; 32];
        for (index, pair) in owner.as_bytes().chunks_exact(2).enumerate() {
            owner_bytes[index] =
                decode_hex_pair(pair).ok_or(ProjectGroupError::InvalidRepositoryCoordinate)?;
        }
        Ok(Self {
            owner_public_key: NostrPublicKey::from_bytes(owner_bytes),
            discriminator: discriminator.to_owned(),
            relay_hint,
        })
    }

    pub const fn owner_public_key(&self) -> NostrPublicKey {
        self.owner_public_key
    }

    pub fn discriminator(&self) -> &str {
        &self.discriminator
    }

    pub fn relay_hint(&self) -> Option<&str> {
        self.relay_hint.as_deref()
    }

    pub fn coordinate(&self) -> String {
        let mut coordinate = String::with_capacity(7 + 64 + self.discriminator.len());
        coordinate.push_str(REPOSITORY_KIND);
        coordinate.push(':');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in self.owner_public_key.as_bytes() {
            coordinate.push(char::from(HEX[usize::from(byte >> 4)]));
            coordinate.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        coordinate.push(':');
        coordinate.push_str(&self.discriminator);
        coordinate
    }
}

fn decode_hex_pair(pair: &[u8]) -> Option<u8> {
    let [high, low] = pair else {
        return None;
    };
    Some(decode_hex_digit(*high)? * 16 + decode_hex_digit(*low)?)
}

fn decode_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProjectGroupIdentity {
    signer_public_key: NostrPublicKey,
    slug: ProjectSlug,
}

impl ProjectGroupIdentity {
    pub const fn signer_public_key(&self) -> NostrPublicKey {
        self.signer_public_key
    }

    pub const fn slug(&self) -> &ProjectSlug {
        &self.slug
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectGroupRecordFields {
    pub signer_public_key: NostrPublicKey,
    pub source: MessageSource,
    pub slug: ProjectSlug,
    pub name: Option<ProjectDisplayName>,
    pub description: Option<ProjectDescription>,
    pub repositories: Vec<RepositoryCoordinate>,
    pub channel_reference: Option<ProjectChannelReference>,
    pub visibility: ProjectVisibility,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectGroup {
    fields: ProjectGroupRecordFields,
}

impl ProjectGroup {
    pub fn from_signed_metadata(
        signer_public_key: NostrPublicKey,
        source: MessageSource,
        slug: impl Into<String>,
        name: Option<String>,
        description: Option<String>,
        repositories: impl IntoIterator<Item = RepositoryCoordinate>,
        channel_reference: Option<String>,
        visibility_value: Option<&str>,
    ) -> Result<Self, ProjectGroupError> {
        Self::from_record(ProjectGroupRecordFields {
            signer_public_key,
            source,
            slug: ProjectSlug::new(slug)?,
            name: name.map(ProjectDisplayName::new).transpose()?,
            description: description.map(ProjectDescription::new).transpose()?,
            repositories: repositories.into_iter().collect(),
            channel_reference: channel_reference
                .map(ProjectChannelReference::new)
                .transpose()?,
            visibility: ProjectVisibility::from_metadata(visibility_value),
        })
    }

    pub fn from_record(mut fields: ProjectGroupRecordFields) -> Result<Self, ProjectGroupError> {
        if fields
            .signer_public_key
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
        {
            return Err(ProjectGroupError::InvalidSigner);
        }
        fields
            .source
            .validate()
            .map_err(|_| ProjectGroupError::InvalidSource)?;
        if fields.repositories.len() > MAX_PROJECT_MEMBERS {
            return Err(ProjectGroupError::TooManyRepositories);
        }
        let mut identities = BTreeSet::new();
        for repository in &fields.repositories {
            if !identities.insert((
                repository.owner_public_key,
                repository.discriminator.as_str(),
            )) {
                return Err(ProjectGroupError::DuplicateRepository);
            }
        }
        fields.repositories.sort_by(|left, right| {
            left.owner_public_key
                .cmp(&right.owner_public_key)
                .then_with(|| left.discriminator.cmp(&right.discriminator))
        });
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &ProjectGroupRecordFields {
        &self.fields
    }

    pub fn identity(&self) -> ProjectGroupIdentity {
        ProjectGroupIdentity {
            signer_public_key: self.fields.signer_public_key,
            slug: self.fields.slug.clone(),
        }
    }

    pub fn display_name(&self) -> &str {
        self.fields
            .name
            .as_ref()
            .map_or_else(|| self.fields.slug.as_str(), ProjectDisplayName::as_str)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectGroupError {
    InvalidSigner,
    InvalidSource,
    InvalidSlug,
    InvalidName,
    InvalidDescription,
    InvalidChannelReference,
    InvalidRepositoryCoordinate,
    TooManyRepositories,
    DuplicateRepository,
}

impl fmt::Display for ProjectGroupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSigner => formatter.write_str("project signer is invalid"),
            Self::InvalidSource => formatter.write_str("project source is invalid"),
            Self::InvalidSlug => formatter.write_str("project slug is invalid"),
            Self::InvalidName => formatter.write_str("project name is invalid"),
            Self::InvalidDescription => formatter.write_str("project description is invalid"),
            Self::InvalidChannelReference => {
                formatter.write_str("project channel reference is invalid")
            }
            Self::InvalidRepositoryCoordinate => {
                formatter.write_str("project repository coordinate is invalid")
            }
            Self::TooManyRepositories => formatter.write_str("project has too many repositories"),
            Self::DuplicateRepository => formatter.write_str("project repeats a repository"),
        }
    }
}

impl Error for ProjectGroupError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NostrEventId;

    fn public_key(value: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([value; 32])
    }

    fn source(value: u8, created_at: u64) -> MessageSource {
        MessageSource {
            event_id: NostrEventId::from_bytes([value; 32]),
            event_created_at: created_at,
        }
    }

    fn coordinate(owner: u8, discriminator: &str) -> RepositoryCoordinate {
        RepositoryCoordinate::parse(
            &format!(
                "30617:{}:{discriminator}",
                format!("{owner:02x}").repeat(32)
            ),
            None,
        )
        .expect("repository coordinate")
    }

    #[test]
    fn multi_repository_groups_preserve_cross_owner_provenance() {
        let group = ProjectGroup::from_signed_metadata(
            public_key(9),
            source(1, 10),
            "platform",
            Some("Platform".into()),
            Some("Desktop, relay, and mobile".into()),
            [
                coordinate(2, "mobile"),
                coordinate(1, "relay"),
                coordinate(1, "desktop:app"),
            ],
            Some("018f0f90-2db8-7b32-bdce-6d36ae5bc901".into()),
            Some("listed"),
        )
        .expect("project group");

        assert_eq!(group.display_name(), "Platform");
        assert_eq!(group.fields().repositories.len(), 3);
        assert_eq!(
            group
                .fields()
                .repositories
                .iter()
                .map(RepositoryCoordinate::owner_public_key)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([public_key(1), public_key(2)])
        );
        assert_eq!(
            group.fields().repositories[0].discriminator(),
            "desktop:app"
        );
        assert_eq!(group.identity().signer_public_key(), public_key(9));
        assert_ne!(
            group.identity().signer_public_key(),
            group.fields().repositories[0].owner_public_key()
        );
    }

    #[test]
    fn visibility_defaults_to_listed_and_only_exact_unlisted_hides() {
        for value in [None, Some("listed"), Some("private"), Some("")] {
            let group = ProjectGroup::from_signed_metadata(
                public_key(1),
                source(1, 10),
                "project",
                None,
                None,
                [],
                None,
                value,
            )
            .expect("listed project");
            assert_eq!(group.fields().visibility, ProjectVisibility::Listed);
            assert_eq!(group.display_name(), "project");
        }
        let unlisted = ProjectGroup::from_signed_metadata(
            public_key(1),
            source(2, 20),
            "project",
            None,
            None,
            [],
            None,
            Some("unlisted"),
        )
        .expect("unlisted project");
        assert_eq!(unlisted.fields().visibility, ProjectVisibility::Unlisted);
    }

    #[test]
    fn malformed_and_duplicate_repository_coordinates_fail_closed() {
        for coordinate in [
            "30618:0101010101010101010101010101010101010101010101010101010101010101:repo",
            "30617:0101:repo",
            "30617:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:repo",
            "30617:0101010101010101010101010101010101010101010101010101010101010101:",
        ] {
            assert_eq!(
                RepositoryCoordinate::parse(coordinate, None),
                Err(ProjectGroupError::InvalidRepositoryCoordinate)
            );
        }
        let repository = coordinate(1, "repo");
        assert_eq!(
            ProjectGroup::from_signed_metadata(
                public_key(1),
                source(1, 10),
                "project",
                None,
                None,
                [repository.clone(), repository],
                None,
                None,
            ),
            Err(ProjectGroupError::DuplicateRepository)
        );
    }
}
