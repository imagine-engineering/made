use crate::generated_kinds::{
    KIND_GIT_REPO_ANNOUNCEMENT, KIND_GIT_REPO_STATE, KIND_GIT_STATUS_CLOSED, KIND_GIT_STATUS_DRAFT,
    KIND_GIT_STATUS_MERGED, KIND_GIT_STATUS_OPEN,
};
use crate::{CanonicalEvent, EventId, PublicKey};
use std::collections::HashSet;
use std::fmt;

const MAX_REPOSITORY_ID_BYTES: usize = 64;
const MAX_NAME_BYTES: usize = 128;
const MAX_DESCRIPTION_BYTES: usize = 1_024;
const MAX_URL_BYTES: usize = 512;
const MAX_RELAY_URL_BYTES: usize = 256;
const MAX_URLS: usize = 5;
const MAX_RELAYS: usize = 10;
const MAX_MAINTAINERS: usize = 64;
const MAX_REFS: usize = 4_096;
const MAX_REF_NAME_BYTES: usize = 1_024;
const MAX_STATUS_RECIPIENTS: usize = 256;
const MAX_STATUS_PATCHES: usize = 4_096;
const MAX_STATUS_COMMITS: usize = 4_096;
const MAX_STATUS_CONTENT_BYTES: usize = 64 * 1_024;
const MAX_EXTRA_TAGS: usize = 256;
const MAX_TAG_VALUES: usize = 4_096;
const MAX_TAG_VALUE_BYTES: usize = 64 * 1_024;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Nip34RepositoryCodecError {
    #[error("unsupported NIP-34 repository kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid NIP-34 repository event: {0}")]
    InvalidEvent(String),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GitObjectId(String);

impl GitObjectId {
    pub fn from_hex(value: &str) -> Result<Self, Nip34RepositoryCodecError> {
        if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid_event(
                "Git object id must be 40 or 64 hexadecimal characters",
            ));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_hex(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GitObjectId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RepositoryCoordinate {
    pub owner: PublicKey,
    pub identifier: String,
    pub relay_hint: Option<String>,
}

impl RepositoryCoordinate {
    pub fn parse(value: &str) -> Result<Self, Nip34RepositoryCodecError> {
        let mut parts = value.splitn(3, ':');
        if parts.next() != Some("30617") {
            return Err(invalid_event(
                "repository coordinate must reference kind 30617",
            ));
        }
        let owner = PublicKey::from_hex(parts.next().unwrap_or_default())
            .map_err(|error| invalid_event(format!("invalid repository owner: {error}")))?;
        let identifier = parts.next().unwrap_or_default();
        validate_repository_id(identifier)?;
        Ok(Self {
            owner,
            identifier: identifier.to_owned(),
            relay_hint: None,
        })
    }

    pub fn parse_tag(tag: &[String]) -> Result<Self, Nip34RepositoryCodecError> {
        if !(2..=3).contains(&tag.len()) || tag.first().map(String::as_str) != Some("a") {
            return Err(invalid_event(
                "repository a tag must contain a coordinate and optional relay hint",
            ));
        }
        let mut coordinate = Self::parse(&tag[1])?;
        if let Some(relay_hint) = tag.get(2) {
            if !relay_hint.is_empty() {
                validate_relay_url(relay_hint)?;
            }
            coordinate.relay_hint = Some(relay_hint.clone());
        }
        Ok(coordinate)
    }

    pub fn value(&self) -> String {
        format!("30617:{}:{}", self.owner.to_hex(), self.identifier)
    }

    pub fn to_tag(&self) -> Result<Vec<String>, Nip34RepositoryCodecError> {
        validate_repository_id(&self.identifier)?;
        let mut tag = vec!["a".into(), self.value()];
        if let Some(relay_hint) = &self.relay_hint {
            if !relay_hint.is_empty() {
                validate_relay_url(relay_hint)?;
            }
            tag.push(relay_hint.clone());
        }
        Ok(tag)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubordinateRepository {
    pub coordinate: RepositoryCoordinate,
    pub git_url: String,
    pub relay_hint: String,
    pub author: PublicKey,
}

impl SubordinateRepository {
    fn parse_tag(tag: &[String]) -> Result<Self, Nip34RepositoryCodecError> {
        if tag.len() != 4 || tag.first().map(String::as_str) != Some("u") {
            return Err(invalid_event(
                "u tag must contain repository, relay and author values",
            ));
        }
        let (coordinate, git_url) = tag[1]
            .split_once('|')
            .ok_or_else(|| invalid_event("u repository value must contain a git URL"))?;
        validate_clone_url(git_url)?;
        if !tag[2].is_empty() {
            validate_relay_url(&tag[2])?;
        }
        let author = PublicKey::from_hex(&tag[3])
            .map_err(|error| invalid_event(format!("invalid subordinate author: {error}")))?;
        Ok(Self {
            coordinate: RepositoryCoordinate::parse(coordinate)?,
            git_url: git_url.to_owned(),
            relay_hint: tag[2].clone(),
            author,
        })
    }

    fn to_tag(&self) -> Result<Vec<String>, Nip34RepositoryCodecError> {
        validate_repository_id(&self.coordinate.identifier)?;
        validate_clone_url(&self.git_url)?;
        if !self.relay_hint.is_empty() {
            validate_relay_url(&self.relay_hint)?;
        }
        Ok(vec![
            "u".into(),
            format!("{}|{}", self.coordinate.value(), self.git_url),
            self.relay_hint.clone(),
            self.author.to_hex(),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryAnnouncement {
    pub identifier: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub web_urls: Vec<String>,
    pub clone_urls: Vec<String>,
    pub relays: Vec<String>,
    pub earliest_unique_commit: Option<GitObjectId>,
    pub maintainers: Vec<PublicKey>,
    pub subordinate: Option<SubordinateRepository>,
    pub hashtags: Vec<String>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl RepositoryAnnouncement {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34RepositoryCodecError> {
        if u32::from(event.kind) != KIND_GIT_REPO_ANNOUNCEMENT {
            return Err(Nip34RepositoryCodecError::UnsupportedKind(event.kind));
        }
        let mut identifier = None;
        let mut name = None;
        let mut description = None;
        let mut web_urls = None;
        let mut clone_urls = None;
        let mut relays = None;
        let mut earliest_unique_commit = None;
        let mut maintainers = None;
        let mut subordinate = None;
        let mut hashtags = Vec::new();
        let mut extra_tags = Vec::new();

        for tag in &event.tags {
            let Some(tag_name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match tag_name {
                "d" => set_once(
                    &mut identifier,
                    parse_single_value(tag, "d", MAX_REPOSITORY_ID_BYTES)?,
                    "d",
                )?,
                "name" => set_once(
                    &mut name,
                    parse_single_value(tag, "name", MAX_NAME_BYTES)?,
                    "name",
                )?,
                "description" => set_once(
                    &mut description,
                    parse_single_value(tag, "description", MAX_DESCRIPTION_BYTES)?,
                    "description",
                )?,
                "web" => set_once(
                    &mut web_urls,
                    parse_urls(tag, "web", MAX_URLS, validate_web_url)?,
                    "web",
                )?,
                "clone" => set_once(
                    &mut clone_urls,
                    parse_urls(tag, "clone", MAX_URLS, validate_clone_url)?,
                    "clone",
                )?,
                "relays" => set_once(
                    &mut relays,
                    parse_urls(tag, "relays", MAX_RELAYS, validate_relay_url)?,
                    "relays",
                )?,
                "r" if tag.get(2).map(String::as_str) == Some("euc") => {
                    if tag.len() != 3 {
                        return Err(invalid_event("euc r tag must have exactly three values"));
                    }
                    set_once(
                        &mut earliest_unique_commit,
                        GitObjectId::from_hex(&tag[1])?,
                        "euc r",
                    )?;
                }
                "r" => return Err(invalid_event("repository r tag must use the euc marker")),
                "maintainers" => {
                    set_once(&mut maintainers, parse_maintainers(tag)?, "maintainers")?
                }
                "u" => set_once(
                    &mut subordinate,
                    SubordinateRepository::parse_tag(tag)?,
                    "u",
                )?,
                "t" => {
                    let hashtag = parse_single_value(tag, "t", MAX_NAME_BYTES)?;
                    if hashtags.contains(&hashtag) {
                        return Err(invalid_event("duplicate repository hashtag"));
                    }
                    hashtags.push(hashtag);
                }
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }

        let identifier = identifier.ok_or_else(|| invalid_event("missing d tag"))?;
        validate_repository_id(&identifier)?;
        Ok(Self {
            identifier,
            name,
            description,
            web_urls: web_urls.unwrap_or_default(),
            clone_urls: clone_urls.unwrap_or_default(),
            relays: relays.unwrap_or_default(),
            earliest_unique_commit,
            maintainers: maintainers.unwrap_or_default(),
            subordinate,
            hashtags,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        owner: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34RepositoryCodecError> {
        validate_repository_id(&self.identifier)?;
        validate_optional_text(&self.name, "name", MAX_NAME_BYTES)?;
        validate_optional_text(&self.description, "description", MAX_DESCRIPTION_BYTES)?;
        validate_count(&self.web_urls, "web URLs", MAX_URLS)?;
        validate_count(&self.clone_urls, "clone URLs", MAX_URLS)?;
        validate_count(&self.relays, "relays", MAX_RELAYS)?;
        validate_count(&self.maintainers, "maintainers", MAX_MAINTAINERS)?;
        for url in &self.web_urls {
            validate_web_url(url)?;
        }
        for url in &self.clone_urls {
            validate_clone_url(url)?;
        }
        for relay in &self.relays {
            validate_relay_url(relay)?;
        }
        ensure_unique(
            self.maintainers.iter().map(|key| key.to_hex()),
            "maintainer",
        )?;
        ensure_unique(self.hashtags.iter().cloned(), "repository hashtag")?;
        validate_extra_tags(&self.extra_tags, announcement_tag_name)?;

        let mut tags = vec![vec!["d".into(), self.identifier.clone()]];
        push_optional_tag(&mut tags, "name", self.name.as_ref());
        push_optional_tag(&mut tags, "description", self.description.as_ref());
        push_multi_tag(&mut tags, "web", &self.web_urls);
        push_multi_tag(&mut tags, "clone", &self.clone_urls);
        push_multi_tag(&mut tags, "relays", &self.relays);
        if let Some(commit) = &self.earliest_unique_commit {
            tags.push(vec!["r".into(), commit.to_string(), "euc".into()]);
        }
        if !self.maintainers.is_empty() {
            let mut tag = vec!["maintainers".into()];
            tag.extend(self.maintainers.iter().map(|key| key.to_hex()));
            tags.push(tag);
        }
        if let Some(subordinate) = &self.subordinate {
            tags.push(subordinate.to_tag()?);
        }
        tags.extend(
            self.hashtags
                .iter()
                .map(|hashtag| vec!["t".into(), hashtag.clone()]),
        );
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            owner,
            created_at,
            KIND_GIT_REPO_ANNOUNCEMENT as u16,
            tags,
            self.content.clone(),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryRef {
    pub name: String,
    pub target: GitObjectId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryState {
    pub identifier: String,
    pub refs: Vec<RepositoryRef>,
    pub head: Option<String>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl RepositoryState {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34RepositoryCodecError> {
        if u32::from(event.kind) != KIND_GIT_REPO_STATE {
            return Err(Nip34RepositoryCodecError::UnsupportedKind(event.kind));
        }
        let mut identifier = None;
        let mut head = None;
        let mut refs = Vec::new();
        let mut ref_names = HashSet::new();
        let mut extra_tags = Vec::new();
        for tag in &event.tags {
            let Some(tag_name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match tag_name {
                "d" => set_once(
                    &mut identifier,
                    parse_single_value(tag, "d", MAX_REPOSITORY_ID_BYTES)?,
                    "d",
                )?,
                "HEAD" => {
                    let value = parse_single_value(tag, "HEAD", MAX_REF_NAME_BYTES + 5)?;
                    let head_ref = value
                        .strip_prefix("ref: ")
                        .ok_or_else(|| invalid_event("HEAD must use the ref: <branch> form"))?;
                    validate_ref_name(head_ref, true)?;
                    set_once(&mut head, head_ref.to_owned(), "HEAD")?;
                }
                name if name.starts_with("refs/") => {
                    if refs.len() >= MAX_REFS {
                        return Err(invalid_event("repository state exceeds ref limit"));
                    }
                    validate_ref_name(name, false)?;
                    if !ref_names.insert(name.to_owned()) {
                        return Err(invalid_event("duplicate repository ref"));
                    }
                    let target = parse_single_value(tag, name, 64)?;
                    refs.push(RepositoryRef {
                        name: name.to_owned(),
                        target: GitObjectId::from_hex(&target)?,
                    });
                }
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }
        let identifier = identifier.ok_or_else(|| invalid_event("missing d tag"))?;
        validate_repository_id(&identifier)?;
        Ok(Self {
            identifier,
            refs,
            head,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        owner: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34RepositoryCodecError> {
        validate_repository_id(&self.identifier)?;
        validate_count(&self.refs, "repository refs", MAX_REFS)?;
        let mut names = HashSet::new();
        for reference in &self.refs {
            validate_ref_name(&reference.name, false)?;
            if !names.insert(&reference.name) {
                return Err(invalid_event("duplicate repository ref"));
            }
        }
        if let Some(head) = &self.head {
            validate_ref_name(head, true)?;
        }
        validate_extra_tags(&self.extra_tags, state_tag_name)?;

        let mut tags = vec![vec!["d".into(), self.identifier.clone()]];
        tags.extend(
            self.refs
                .iter()
                .map(|reference| vec![reference.name.clone(), reference.target.to_string()]),
        );
        if let Some(head) = &self.head {
            tags.push(vec!["HEAD".into(), format!("ref: {head}")]);
        }
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            owner,
            created_at,
            KIND_GIT_REPO_STATE as u16,
            tags,
            self.content.clone(),
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepositoryStatus {
    Open,
    AppliedOrResolved,
    Closed,
    Draft,
}

impl RepositoryStatus {
    fn from_kind(kind: u16) -> Result<Self, Nip34RepositoryCodecError> {
        match u32::from(kind) {
            KIND_GIT_STATUS_OPEN => Ok(Self::Open),
            KIND_GIT_STATUS_MERGED => Ok(Self::AppliedOrResolved),
            KIND_GIT_STATUS_CLOSED => Ok(Self::Closed),
            KIND_GIT_STATUS_DRAFT => Ok(Self::Draft),
            _ => Err(Nip34RepositoryCodecError::UnsupportedKind(kind)),
        }
    }

    fn kind(self) -> u16 {
        match self {
            Self::Open => KIND_GIT_STATUS_OPEN as u16,
            Self::AppliedOrResolved => KIND_GIT_STATUS_MERGED as u16,
            Self::Closed => KIND_GIT_STATUS_CLOSED as u16,
            Self::Draft => KIND_GIT_STATUS_DRAFT as u16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppliedPatchReference {
    pub event_id: EventId,
    pub relay_hint: Option<String>,
    pub author_hint: Option<PublicKey>,
}

impl AppliedPatchReference {
    fn parse_tag(tag: &[String]) -> Result<Self, Nip34RepositoryCodecError> {
        if !(2..=4).contains(&tag.len()) || tag.first().map(String::as_str) != Some("q") {
            return Err(invalid_event("q tag has an invalid shape"));
        }
        let event_id = EventId::from_hex(&tag[1])
            .map_err(|error| invalid_event(format!("invalid q event id: {error}")))?;
        let relay_hint = tag.get(2).cloned();
        if let Some(relay_hint) = &relay_hint {
            if !relay_hint.is_empty() {
                validate_relay_url(relay_hint)?;
            }
        }
        let author_hint = tag
            .get(3)
            .map(|value| {
                if relay_hint.is_none() {
                    return Err(invalid_event("q author hint requires a relay hint"));
                }
                PublicKey::from_hex(value)
                    .map_err(|error| invalid_event(format!("invalid q author: {error}")))
            })
            .transpose()?;
        Ok(Self {
            event_id,
            relay_hint,
            author_hint,
        })
    }

    fn to_tag(&self) -> Result<Vec<String>, Nip34RepositoryCodecError> {
        let mut tag = vec!["q".into(), self.event_id.to_hex()];
        if let Some(relay_hint) = &self.relay_hint {
            if !relay_hint.is_empty() {
                validate_relay_url(relay_hint)?;
            }
            tag.push(relay_hint.clone());
        }
        if let Some(author_hint) = self.author_hint {
            if self.relay_hint.is_none() {
                return Err(invalid_event("q author hint requires a relay hint"));
            }
            tag.push(author_hint.to_hex());
        }
        Ok(tag)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryStatusEvent {
    pub status: RepositoryStatus,
    pub root_event: EventId,
    pub accepted_revision_root: Option<EventId>,
    pub recipients: Vec<PublicKey>,
    pub repository: Option<RepositoryCoordinate>,
    pub earliest_unique_commit: Option<GitObjectId>,
    pub applied_patches: Vec<AppliedPatchReference>,
    pub merge_commit: Option<GitObjectId>,
    pub applied_as_commits: Vec<GitObjectId>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl RepositoryStatusEvent {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34RepositoryCodecError> {
        let status = RepositoryStatus::from_kind(event.kind)?;
        if event.content.len() > MAX_STATUS_CONTENT_BYTES {
            return Err(invalid_event("status content exceeds 64 KiB"));
        }
        let mut root_event = None;
        let mut accepted_revision_root = None;
        let mut recipients = Vec::new();
        let mut recipient_set = HashSet::new();
        let mut repository = None;
        let mut r_commits = Vec::new();
        let mut applied_patches = Vec::new();
        let mut merge_commit = None;
        let mut applied_as_commits = None;
        let mut extra_tags = Vec::new();

        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "e" => {
                    if tag.len() != 4 || !tag[2].is_empty() {
                        return Err(invalid_event(
                            "status e tags require an empty relay and root/reply marker",
                        ));
                    }
                    let event_id = EventId::from_hex(&tag[1]).map_err(|error| {
                        invalid_event(format!("invalid status event reference: {error}"))
                    })?;
                    match tag[3].as_str() {
                        "root" => set_once(&mut root_event, event_id, "root e")?,
                        "reply" => set_once(&mut accepted_revision_root, event_id, "reply e")?,
                        _ => return Err(invalid_event("status e tag has an invalid marker")),
                    }
                }
                "p" => {
                    if recipients.len() >= MAX_STATUS_RECIPIENTS {
                        return Err(invalid_event("status exceeds recipient limit"));
                    }
                    let value = parse_single_value(tag, "p", 64)?;
                    let recipient = PublicKey::from_hex(&value).map_err(|error| {
                        invalid_event(format!("invalid status recipient: {error}"))
                    })?;
                    if !recipient_set.insert(recipient) {
                        return Err(invalid_event("duplicate status recipient"));
                    }
                    recipients.push(recipient);
                }
                "a" => set_once(&mut repository, RepositoryCoordinate::parse_tag(tag)?, "a")?,
                "r" => {
                    if r_commits.len() > MAX_STATUS_COMMITS {
                        return Err(invalid_event("status exceeds commit reference limit"));
                    }
                    let value = parse_single_value(tag, "r", 64)?;
                    r_commits.push(GitObjectId::from_hex(&value)?);
                }
                "q" => {
                    if applied_patches.len() >= MAX_STATUS_PATCHES {
                        return Err(invalid_event("status exceeds applied patch limit"));
                    }
                    applied_patches.push(AppliedPatchReference::parse_tag(tag)?);
                }
                "merge-commit" => {
                    let value = parse_single_value(tag, "merge-commit", 64)?;
                    set_once(
                        &mut merge_commit,
                        GitObjectId::from_hex(&value)?,
                        "merge-commit",
                    )?;
                }
                "applied-as-commits" => set_once(
                    &mut applied_as_commits,
                    parse_object_ids(tag, "applied-as-commits", MAX_STATUS_COMMITS)?,
                    "applied-as-commits",
                )?,
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }

        let root_event = root_event.ok_or_else(|| invalid_event("missing root e tag"))?;
        let applied_as_commits = applied_as_commits.unwrap_or_default();
        if status != RepositoryStatus::AppliedOrResolved
            && (!applied_patches.is_empty()
                || merge_commit.is_some()
                || !applied_as_commits.is_empty())
        {
            return Err(invalid_event(
                "applied patch and commit metadata requires kind 1631",
            ));
        }
        let output_commits = merge_commit
            .iter()
            .chain(applied_as_commits.iter())
            .collect::<HashSet<_>>();
        for output in &output_commits {
            if !r_commits.contains(output) {
                return Err(invalid_event(
                    "merged/applied commit is missing its matching r tag",
                ));
            }
        }
        let mut euc = None;
        let mut seen_r = HashSet::new();
        for commit in r_commits {
            if !seen_r.insert(commit.clone()) {
                return Err(invalid_event("duplicate status r tag"));
            }
            if !output_commits.contains(&commit) {
                set_once(&mut euc, commit, "earliest unique commit r")?;
            }
        }
        Ok(Self {
            status,
            root_event,
            accepted_revision_root,
            recipients,
            repository,
            earliest_unique_commit: euc,
            applied_patches,
            merge_commit,
            applied_as_commits,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34RepositoryCodecError> {
        if self.content.len() > MAX_STATUS_CONTENT_BYTES {
            return Err(invalid_event("status content exceeds 64 KiB"));
        }
        validate_count(&self.recipients, "status recipients", MAX_STATUS_RECIPIENTS)?;
        validate_count(&self.applied_patches, "applied patches", MAX_STATUS_PATCHES)?;
        validate_count(
            &self.applied_as_commits,
            "applied commits",
            MAX_STATUS_COMMITS,
        )?;
        ensure_unique(self.recipients.iter().copied(), "status recipient")?;
        ensure_unique(self.applied_as_commits.iter().cloned(), "applied commit")?;
        if self.status != RepositoryStatus::AppliedOrResolved
            && (!self.applied_patches.is_empty()
                || self.merge_commit.is_some()
                || !self.applied_as_commits.is_empty())
        {
            return Err(invalid_event(
                "applied patch and commit metadata requires kind 1631",
            ));
        }
        validate_extra_tags(&self.extra_tags, status_tag_name)?;

        let mut tags = vec![vec![
            "e".into(),
            self.root_event.to_hex(),
            String::new(),
            "root".into(),
        ]];
        if let Some(revision) = self.accepted_revision_root {
            tags.push(vec![
                "e".into(),
                revision.to_hex(),
                String::new(),
                "reply".into(),
            ]);
        }
        tags.extend(
            self.recipients
                .iter()
                .map(|recipient| vec!["p".into(), recipient.to_hex()]),
        );
        if let Some(repository) = &self.repository {
            tags.push(repository.to_tag()?);
        }
        if let Some(commit) = &self.earliest_unique_commit {
            tags.push(vec!["r".into(), commit.to_string()]);
        }
        for patch in &self.applied_patches {
            tags.push(patch.to_tag()?);
        }
        if let Some(commit) = &self.merge_commit {
            tags.push(vec!["merge-commit".into(), commit.to_string()]);
            tags.push(vec!["r".into(), commit.to_string()]);
        }
        if !self.applied_as_commits.is_empty() {
            let mut tag = vec!["applied-as-commits".into()];
            tag.extend(self.applied_as_commits.iter().map(ToString::to_string));
            tags.push(tag);
            tags.extend(
                self.applied_as_commits
                    .iter()
                    .map(|commit| vec!["r".into(), commit.to_string()]),
            );
        }
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            self.status.kind(),
            tags,
            self.content.clone(),
        ))
    }
}

fn validate_repository_id(value: &str) -> Result<(), Nip34RepositoryCodecError> {
    if value.is_empty()
        || value.len() > MAX_REPOSITORY_ID_BYTES
        || value.starts_with('.')
        || value.contains("..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(invalid_event(
            "repository identifier must match [A-Za-z0-9._-]{1,64}, without a leading dot or '..'",
        ));
    }
    Ok(())
}

fn validate_ref_name(value: &str, require_head: bool) -> Result<(), Nip34RepositoryCodecError> {
    let valid_namespace = if require_head {
        value.starts_with("refs/heads/")
    } else {
        value.starts_with("refs/heads/") || value.starts_with("refs/tags/")
    };
    if !valid_namespace
        || value.len() > MAX_REF_NAME_BYTES
        || value.ends_with('/')
        || value.ends_with('.')
        || value.ends_with(".lock")
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.bytes().any(|byte| {
            byte <= b' '
                || byte == 0x7f
                || matches!(byte, b'~' | b'^' | b':' | b'?' | b'*' | b'[' | b'\\')
        })
    {
        return Err(invalid_event("invalid Git branch or tag ref name"));
    }
    let short_name = value
        .split_once('/')
        .map(|(_, suffix)| suffix)
        .unwrap_or_default();
    if short_name.is_empty()
        || short_name.split('/').any(|component| {
            component.is_empty()
                || component.starts_with('.')
                || component.ends_with('.')
                || component.ends_with(".lock")
        })
    {
        return Err(invalid_event("invalid Git branch or tag ref component"));
    }
    Ok(())
}

fn validate_web_url(value: &str) -> Result<(), Nip34RepositoryCodecError> {
    validate_url(value, MAX_URL_BYTES, &["http://", "https://"], "web URL")
}

fn validate_clone_url(value: &str) -> Result<(), Nip34RepositoryCodecError> {
    if value.is_empty() || value.len() > MAX_URL_BYTES || value.chars().any(char::is_control) {
        return Err(invalid_event("clone URL must contain 1-512 safe bytes"));
    }
    Ok(())
}

fn validate_relay_url(value: &str) -> Result<(), Nip34RepositoryCodecError> {
    validate_url(
        value,
        MAX_RELAY_URL_BYTES,
        &["ws://", "wss://"],
        "relay URL",
    )
}

fn validate_url(
    value: &str,
    maximum: usize,
    schemes: &[&str],
    label: &str,
) -> Result<(), Nip34RepositoryCodecError> {
    if value.len() > maximum
        || !schemes.iter().any(|scheme| value.starts_with(scheme))
        || value.chars().any(char::is_control)
    {
        return Err(invalid_event(format!("invalid {label}")));
    }
    Ok(())
}

fn parse_single_value(
    tag: &[String],
    name: &str,
    maximum: usize,
) -> Result<String, Nip34RepositoryCodecError> {
    if tag.len() != 2 || tag[1].is_empty() || tag[1].len() > maximum {
        return Err(invalid_event(format!(
            "{name} tag must contain one nonempty value of at most {maximum} bytes"
        )));
    }
    Ok(tag[1].clone())
}

fn parse_urls(
    tag: &[String],
    name: &str,
    maximum: usize,
    validate: fn(&str) -> Result<(), Nip34RepositoryCodecError>,
) -> Result<Vec<String>, Nip34RepositoryCodecError> {
    if tag.len() < 2 || tag.len() - 1 > maximum {
        return Err(invalid_event(format!(
            "{name} tag must contain 1-{maximum} values"
        )));
    }
    let values = tag[1..].to_vec();
    for value in &values {
        validate(value)?;
    }
    ensure_unique(values.iter().cloned(), name)?;
    Ok(values)
}

fn parse_maintainers(tag: &[String]) -> Result<Vec<PublicKey>, Nip34RepositoryCodecError> {
    if tag.len() < 2 || tag.len() - 1 > MAX_MAINTAINERS {
        return Err(invalid_event("maintainers tag has an invalid value count"));
    }
    let maintainers = tag[1..]
        .iter()
        .map(|value| {
            PublicKey::from_hex(value)
                .map_err(|error| invalid_event(format!("invalid maintainer: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ensure_unique(maintainers.iter().copied(), "maintainer")?;
    Ok(maintainers)
}

fn parse_object_ids(
    tag: &[String],
    name: &str,
    maximum: usize,
) -> Result<Vec<GitObjectId>, Nip34RepositoryCodecError> {
    if tag.len() < 2 || tag.len() - 1 > maximum {
        return Err(invalid_event(format!("{name} has an invalid value count")));
    }
    let values = tag[1..]
        .iter()
        .map(|value| GitObjectId::from_hex(value))
        .collect::<Result<Vec<_>, _>>()?;
    ensure_unique(values.iter().cloned(), name)?;
    Ok(values)
}

fn set_once<T>(
    slot: &mut Option<T>,
    value: T,
    name: &str,
) -> Result<(), Nip34RepositoryCodecError> {
    if slot.replace(value).is_some() {
        return Err(invalid_event(format!("duplicate {name} tag")));
    }
    Ok(())
}

fn ensure_unique<T>(
    values: impl IntoIterator<Item = T>,
    name: &str,
) -> Result<(), Nip34RepositoryCodecError>
where
    T: Eq + std::hash::Hash,
{
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(invalid_event(format!("duplicate {name}")));
        }
    }
    Ok(())
}

fn validate_count<T>(
    values: &[T],
    name: &str,
    maximum: usize,
) -> Result<(), Nip34RepositoryCodecError> {
    if values.len() > maximum {
        return Err(invalid_event(format!("{name} exceeds limit {maximum}")));
    }
    Ok(())
}

fn validate_optional_text(
    value: &Option<String>,
    name: &str,
    maximum: usize,
) -> Result<(), Nip34RepositoryCodecError> {
    if let Some(value) = value {
        if value.is_empty() || value.len() > maximum {
            return Err(invalid_event(format!(
                "{name} must contain 1-{maximum} bytes"
            )));
        }
    }
    Ok(())
}

fn push_optional_tag(tags: &mut Vec<Vec<String>>, name: &str, value: Option<&String>) {
    if let Some(value) = value {
        tags.push(vec![name.into(), value.clone()]);
    }
}

fn push_multi_tag(tags: &mut Vec<Vec<String>>, name: &str, values: &[String]) {
    if !values.is_empty() {
        let mut tag = Vec::with_capacity(values.len() + 1);
        tag.push(name.into());
        tag.extend(values.iter().cloned());
        tags.push(tag);
    }
}

fn push_extra_tag(
    tags: &mut Vec<Vec<String>>,
    tag: &[String],
) -> Result<(), Nip34RepositoryCodecError> {
    if tags.len() >= MAX_EXTRA_TAGS {
        return Err(invalid_event("event exceeds extra tag limit"));
    }
    validate_tag_bounds(tag)?;
    tags.push(tag.to_vec());
    Ok(())
}

fn validate_extra_tags(
    tags: &[Vec<String>],
    reserved: fn(&str) -> bool,
) -> Result<(), Nip34RepositoryCodecError> {
    validate_count(tags, "extra tags", MAX_EXTRA_TAGS)?;
    for tag in tags {
        validate_tag_bounds(tag)?;
        if reserved(&tag[0]) {
            return Err(invalid_event("extra tag uses a reserved NIP-34 tag name"));
        }
    }
    Ok(())
}

fn validate_tag_bounds(tag: &[String]) -> Result<(), Nip34RepositoryCodecError> {
    if tag.is_empty()
        || tag.len() > MAX_TAG_VALUES
        || tag.iter().any(|value| value.len() > MAX_TAG_VALUE_BYTES)
    {
        return Err(invalid_event("extra tag exceeds structural limits"));
    }
    Ok(())
}

fn announcement_tag_name(name: &str) -> bool {
    matches!(
        name,
        "d" | "name" | "description" | "web" | "clone" | "relays" | "r" | "maintainers" | "u" | "t"
    )
}

fn state_tag_name(name: &str) -> bool {
    name == "d" || name == "HEAD" || name.starts_with("refs/")
}

fn status_tag_name(name: &str) -> bool {
    matches!(
        name,
        "e" | "p" | "a" | "r" | "q" | "merge-commit" | "applied-as-commits"
    )
}

fn invalid_event(message: impl Into<String>) -> Nip34RepositoryCodecError {
    Nip34RepositoryCodecError::InvalidEvent(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const MAINTAINER: &str = "c6047f9441ed7d6d3045406e95c07cd85a207230f3dc9c0db865c3e0b0f2bdc7";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const SECOND_COMMIT: &str = "89abcdef0123456789abcdef0123456789abcdef";
    const ROOT_EVENT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PATCH_EVENT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn public_key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("valid public key fixture")
    }

    fn canonical_event(kind: u16, tags: Vec<Vec<String>>, content: &str) -> CanonicalEvent {
        CanonicalEvent::new(public_key(OWNER), 1_787_356_800, kind, tags, content.into())
    }

    #[test]
    fn nip34_repository_announcement_golden_round_trips_clone_urls_and_maintainers() {
        let tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["d","zed"],["name","Zed"],["description","A code editor"],["web","https://zed.dev","https://github.com/zed-industries/zed"],["clone","https://github.com/zed-industries/zed.git","git@github.com:zed-industries/zed.git"],["relays","wss://relay.ngit.dev"],["r","{COMMIT}","euc"],["maintainers","{MAINTAINER}"],["t","editor"],["future-metadata","preserved"]]"#
        ))
        .expect("valid golden tags");
        let event = canonical_event(KIND_GIT_REPO_ANNOUNCEMENT as u16, tags, "");

        let announcement = RepositoryAnnouncement::parse_event(&event).expect("valid announcement");
        assert_eq!(announcement.clone_urls.len(), 2);
        assert_eq!(announcement.maintainers, vec![public_key(MAINTAINER)]);
        assert_eq!(
            announcement
                .to_event(event.public_key, event.created_at)
                .expect("encodable announcement"),
            event
        );
    }

    #[test]
    fn nip34_repository_state_golden_round_trips_refs_and_head() {
        let tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["d","zed"],["refs/heads/main","{COMMIT}"],["refs/tags/v1.0.0","{SECOND_COMMIT}"],["HEAD","ref: refs/heads/main"],["future-state","preserved"]]"#
        ))
        .expect("valid golden tags");
        let event = canonical_event(KIND_GIT_REPO_STATE as u16, tags, "");

        let state = RepositoryState::parse_event(&event).expect("valid repository state");
        assert_eq!(state.refs.len(), 2);
        assert_eq!(state.head.as_deref(), Some("refs/heads/main"));
        assert_eq!(
            state
                .to_event(event.public_key, event.created_at)
                .expect("encodable repository state"),
            event
        );
    }

    #[test]
    fn nip34_repository_status_golden_round_trips_applied_refs() {
        let tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["e","{ROOT_EVENT}","","root"],["p","{MAINTAINER}"],["a","30617:{OWNER}:zed","wss://relay.ngit.dev"],["r","{COMMIT}"],["q","{PATCH_EVENT}","wss://relay.ngit.dev","{MAINTAINER}"],["merge-commit","{SECOND_COMMIT}"],["r","{SECOND_COMMIT}"],["future-status","preserved"]]"#
        ))
        .expect("valid golden tags");
        let event = canonical_event(KIND_GIT_STATUS_MERGED as u16, tags, "Applied cleanly");

        let status = RepositoryStatusEvent::parse_event(&event).expect("valid status");
        assert_eq!(status.status, RepositoryStatus::AppliedOrResolved);
        assert_eq!(status.applied_patches.len(), 1);
        assert_eq!(
            status
                .to_event(event.public_key, event.created_at)
                .expect("encodable status"),
            event
        );
    }

    #[test]
    fn nip34_repository_coordinates_reject_malformed_golden_cases() {
        let malformed = [
            "",
            "30618:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:zed",
            "30617:short:zed",
            "30617:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:",
            "30617:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:.hidden",
            "30617:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:zed..fork",
            "30617:79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798:zed",
        ];

        for value in malformed {
            assert!(
                RepositoryCoordinate::parse(value).is_err(),
                "accepted malformed coordinate {value:?}"
            );
        }
        let valid = RepositoryCoordinate::parse(&format!("30617:{OWNER}:zed"))
            .expect("valid repository coordinate");
        assert_eq!(valid.value(), format!("30617:{OWNER}:zed"));

        let hinted_tag = vec!["a".into(), valid.value(), String::new()];
        assert_eq!(
            RepositoryCoordinate::parse_tag(&hinted_tag)
                .expect("empty relay placeholder is valid")
                .to_tag()
                .expect("empty relay placeholder is encodable"),
            hinted_tag
        );

        let uppercase_commit = COMMIT.to_uppercase();
        assert_eq!(
            GitObjectId::from_hex(&uppercase_commit)
                .expect("Buzz accepts uppercase object IDs")
                .as_hex(),
            uppercase_commit
        );
    }

    #[test]
    fn nip34_repository_rejects_invalid_ref_and_status_metadata() {
        let invalid_ref = canonical_event(
            KIND_GIT_REPO_STATE as u16,
            vec![
                vec!["d".into(), "zed".into()],
                vec!["refs/heads/../escape".into(), COMMIT.into()],
            ],
            "",
        );
        assert!(RepositoryState::parse_event(&invalid_ref).is_err());

        let invalid_status = canonical_event(
            KIND_GIT_STATUS_CLOSED as u16,
            vec![
                vec!["e".into(), ROOT_EVENT.into(), String::new(), "root".into()],
                vec!["merge-commit".into(), COMMIT.into()],
                vec!["r".into(), COMMIT.into()],
            ],
            "",
        );
        assert!(RepositoryStatusEvent::parse_event(&invalid_status).is_err());
    }
}
