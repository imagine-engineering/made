use crate::generated_kinds::{
    KIND_GIT_ISSUE, KIND_GIT_PATCH, KIND_GIT_PR_UPDATE, KIND_GIT_PULL_REQUEST,
};
use crate::nip34_repository::{
    GitObjectId, Nip34RepositoryCodecError, RepositoryCoordinate, RepositoryStatusEvent,
};
use crate::{CanonicalEvent, EventId, PublicKey};
use std::collections::HashSet;
use uuid::Uuid;

const KIND_COMMENT: u16 = 1_111;
const MAX_PATCH_BYTES: usize = 60 * 1_024;
const MAX_CONTENT_BYTES: usize = 64 * 1_024;
const MAX_SUBJECT_BYTES: usize = 256;
const MAX_LABEL_BYTES: usize = 256;
const MAX_LABELS: usize = 128;
const MAX_RECIPIENTS: usize = 256;
const MAX_CLONE_URL_BYTES: usize = 512;
const MAX_CLONE_URLS: usize = 5;
const MAX_BRANCH_NAME_BYTES: usize = 1_024;
const MAX_SIGNATURE_BYTES: usize = 64 * 1_024;
const MAX_COMMITTER_FIELD_BYTES: usize = 1_024;
const MAX_RELAY_URL_BYTES: usize = 256;
const MAX_EXTRA_TAGS: usize = 256;
const MAX_TAG_VALUES: usize = 4_096;
const MAX_TAG_VALUE_BYTES: usize = 64 * 1_024;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Nip34CollaborationCodecError {
    #[error("unsupported NIP-34 collaboration kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid NIP-34 collaboration event: {0}")]
    InvalidEvent(String),
}

impl From<Nip34RepositoryCodecError> for Nip34CollaborationCodecError {
    fn from(error: Nip34RepositoryCodecError) -> Self {
        Self::InvalidEvent(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchPosition {
    Continuation,
    Root,
    RootRevision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitCommitter {
    pub name: String,
    pub email: String,
    pub timestamp: String,
    pub timezone_offset_minutes: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitPatch {
    pub repository: RepositoryCoordinate,
    pub earliest_unique_commit: Option<GitObjectId>,
    pub recipients: Vec<PublicKey>,
    pub reply_to: Option<EventId>,
    pub position: PatchPosition,
    pub commit: Option<GitObjectId>,
    pub parent_commit: Option<GitObjectId>,
    pub commit_pgp_signature: Option<String>,
    pub committer: Option<GitCommitter>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl GitPatch {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        require_kind(event, KIND_GIT_PATCH)?;
        if event.content.trim().is_empty() || event.content.len() > MAX_PATCH_BYTES {
            return Err(invalid_event(
                "patch content must contain 1-61440 non-whitespace bytes",
            ));
        }

        let mut repository = None;
        let mut earliest_unique_commit = None;
        let mut recipients = Vec::new();
        let mut reply_to = None;
        let mut root = false;
        let mut root_revision = false;
        let mut commit = None;
        let mut commit_r = None;
        let mut parent_commit = None;
        let mut commit_pgp_signature = None;
        let mut committer = None;
        let mut extra_tags = Vec::new();

        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "a" => set_once(&mut repository, RepositoryCoordinate::parse_tag(tag)?, "a")?,
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
                "r" => set_once(
                    &mut commit_r,
                    GitObjectId::from_hex(&single_value(tag, "r", 64)?)?,
                    "commit r",
                )?,
                "p" => recipients.push(parse_public_key_tag(tag, "p")?),
                "e" => {
                    if tag.len() != 4 || !tag[2].is_empty() || tag[3].as_str() != "reply" {
                        return Err(invalid_event(
                            "patch e tag must use an empty relay and reply marker",
                        ));
                    }
                    set_once(
                        &mut reply_to,
                        parse_event_id(&tag[1], "patch reply")?,
                        "reply e",
                    )?;
                }
                "t" if tag.get(1).map(String::as_str) == Some("root") => {
                    require_exact_len(tag, "root t", 2)?;
                    if root {
                        return Err(invalid_event("duplicate root t tag"));
                    }
                    root = true;
                }
                "t" if tag.get(1).map(String::as_str) == Some("root-revision") => {
                    require_exact_len(tag, "root-revision t", 2)?;
                    if root_revision {
                        return Err(invalid_event("duplicate root-revision t tag"));
                    }
                    root_revision = true;
                }
                "t" => return Err(invalid_event("unsupported patch t tag")),
                "commit" => set_once(
                    &mut commit,
                    GitObjectId::from_hex(&single_value(tag, "commit", 64)?)?,
                    "commit",
                )?,
                "parent-commit" => set_once(
                    &mut parent_commit,
                    GitObjectId::from_hex(&single_value(tag, "parent-commit", 64)?)?,
                    "parent-commit",
                )?,
                "commit-pgp-sig" => {
                    if tag.len() != 2 || tag[1].len() > MAX_SIGNATURE_BYTES {
                        return Err(invalid_event("commit-pgp-sig has an invalid shape"));
                    }
                    set_once(&mut commit_pgp_signature, tag[1].clone(), "commit-pgp-sig")?;
                }
                "committer" => {
                    set_once(&mut committer, GitCommitter::parse_tag(tag)?, "committer")?
                }
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }

        if root && root_revision {
            return Err(invalid_event(
                "patch cannot be both a root and a root revision",
            ));
        }
        if commit != commit_r {
            return Err(invalid_event(
                "commit tag and unmarked r tag must both be present and equal",
            ));
        }
        let repository = repository.ok_or_else(|| invalid_event("missing repository a tag"))?;
        validate_recipients(&recipients, repository.owner)?;
        Ok(Self {
            repository,
            earliest_unique_commit,
            recipients,
            reply_to,
            position: if root {
                PatchPosition::Root
            } else if root_revision {
                PatchPosition::RootRevision
            } else {
                PatchPosition::Continuation
            },
            commit,
            parent_commit,
            commit_pgp_signature,
            committer,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34CollaborationCodecError> {
        if self.content.trim().is_empty() || self.content.len() > MAX_PATCH_BYTES {
            return Err(invalid_event(
                "patch content must contain 1-61440 non-whitespace bytes",
            ));
        }
        validate_recipients(&self.recipients, self.repository.owner)?;
        validate_extra_tags(&self.extra_tags, patch_reserved_tag)?;

        let mut tags = vec![self.repository.to_tag()?];
        if let Some(euc) = &self.earliest_unique_commit {
            tags.push(vec!["r".into(), euc.to_string(), "euc".into()]);
        }
        tags.extend(
            self.recipients
                .iter()
                .map(|recipient| vec!["p".into(), recipient.to_hex()]),
        );
        if let Some(reply_to) = self.reply_to {
            tags.push(vec![
                "e".into(),
                reply_to.to_hex(),
                String::new(),
                "reply".into(),
            ]);
        }
        match self.position {
            PatchPosition::Continuation => {}
            PatchPosition::Root => tags.push(vec!["t".into(), "root".into()]),
            PatchPosition::RootRevision => tags.push(vec!["t".into(), "root-revision".into()]),
        }
        if let Some(commit) = &self.commit {
            tags.push(vec!["commit".into(), commit.to_string()]);
            tags.push(vec!["r".into(), commit.to_string()]);
        }
        if let Some(parent_commit) = &self.parent_commit {
            tags.push(vec!["parent-commit".into(), parent_commit.to_string()]);
        }
        if let Some(signature) = &self.commit_pgp_signature {
            if signature.len() > MAX_SIGNATURE_BYTES {
                return Err(invalid_event("commit-pgp-sig exceeds 64 KiB"));
            }
            tags.push(vec!["commit-pgp-sig".into(), signature.clone()]);
        }
        if let Some(committer) = &self.committer {
            tags.push(committer.to_tag()?);
        }
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            KIND_GIT_PATCH as u16,
            tags,
            self.content.clone(),
        ))
    }
}

impl GitCommitter {
    fn parse_tag(tag: &[String]) -> Result<Self, Nip34CollaborationCodecError> {
        if tag.len() != 5 || tag[1..].iter().any(|value| value.is_empty()) {
            return Err(invalid_event(
                "committer tag must contain name, email, timestamp and timezone",
            ));
        }
        validate_text(&tag[1], "committer name", MAX_COMMITTER_FIELD_BYTES)?;
        validate_text(&tag[2], "committer email", MAX_COMMITTER_FIELD_BYTES)?;
        tag[3]
            .parse::<i64>()
            .map_err(|_| invalid_event("committer timestamp must be a signed integer"))?;
        let timezone = tag[4]
            .parse::<i16>()
            .map_err(|_| invalid_event("committer timezone must be integer minutes"))?;
        if !(-1_439..=1_439).contains(&timezone) {
            return Err(invalid_event("committer timezone is out of range"));
        }
        Ok(Self {
            name: tag[1].clone(),
            email: tag[2].clone(),
            timestamp: tag[3].clone(),
            timezone_offset_minutes: tag[4].clone(),
        })
    }

    fn to_tag(&self) -> Result<Vec<String>, Nip34CollaborationCodecError> {
        Self::parse_tag(&[
            "committer".into(),
            self.name.clone(),
            self.email.clone(),
            self.timestamp.clone(),
            self.timezone_offset_minutes.clone(),
        ])
        .map(|_| {
            vec![
                "committer".into(),
                self.name.clone(),
                self.email.clone(),
                self.timestamp.clone(),
                self.timezone_offset_minutes.clone(),
            ]
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitIssue {
    pub repository: RepositoryCoordinate,
    pub recipients: Vec<PublicKey>,
    pub subject: String,
    pub labels: Vec<String>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl GitIssue {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        require_kind(event, KIND_GIT_ISSUE)?;
        validate_content(&event.content, MAX_CONTENT_BYTES, true)?;
        let mut repository = None;
        let mut recipients = Vec::new();
        let mut subject = None;
        let mut labels = Vec::new();
        let mut extra_tags = Vec::new();
        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "a" => set_once(&mut repository, RepositoryCoordinate::parse_tag(tag)?, "a")?,
                "p" => recipients.push(parse_public_key_tag(tag, "p")?),
                "subject" => set_once(
                    &mut subject,
                    single_value(tag, "subject", MAX_SUBJECT_BYTES)?,
                    "subject",
                )?,
                "t" => labels.push(single_value(tag, "t", MAX_LABEL_BYTES)?),
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }
        let repository = repository.ok_or_else(|| invalid_event("missing repository a tag"))?;
        validate_recipients(&recipients, repository.owner)?;
        validate_labels(&labels)?;
        Ok(Self {
            repository,
            recipients,
            subject: subject.ok_or_else(|| invalid_event("missing subject tag"))?,
            labels,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34CollaborationCodecError> {
        validate_content(&self.content, MAX_CONTENT_BYTES, true)?;
        validate_text(&self.subject, "subject", MAX_SUBJECT_BYTES)?;
        validate_recipients(&self.recipients, self.repository.owner)?;
        validate_labels(&self.labels)?;
        validate_extra_tags(&self.extra_tags, issue_reserved_tag)?;
        let mut tags = vec![self.repository.to_tag()?];
        tags.extend(
            self.recipients
                .iter()
                .map(|recipient| vec!["p".into(), recipient.to_hex()]),
        );
        tags.push(vec!["subject".into(), self.subject.clone()]);
        tags.extend(
            self.labels
                .iter()
                .map(|label| vec!["t".into(), label.clone()]),
        );
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            KIND_GIT_ISSUE as u16,
            tags,
            self.content.clone(),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitPullRequest {
    pub repository: RepositoryCoordinate,
    pub earliest_unique_commit: Option<GitObjectId>,
    pub recipients: Vec<PublicKey>,
    pub subject: String,
    pub labels: Vec<String>,
    pub tip_commit: GitObjectId,
    pub clone_urls: Vec<String>,
    pub channel_id: Option<Uuid>,
    pub branch_name: Option<String>,
    pub merge_base: Option<GitObjectId>,
    pub revision_of: Option<EventId>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl GitPullRequest {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        require_kind(event, KIND_GIT_PULL_REQUEST)?;
        validate_content(&event.content, MAX_CONTENT_BYTES, true)?;
        let mut repository = None;
        let mut earliest_unique_commit = None;
        let mut recipients = Vec::new();
        let mut subject = None;
        let mut labels = Vec::new();
        let mut tip_commit = None;
        let mut clone_urls = None;
        let mut channel_id = None;
        let mut branch_name = None;
        let mut merge_base = None;
        let mut revision_of = None;
        let mut extra_tags = Vec::new();
        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "a" => set_once(&mut repository, RepositoryCoordinate::parse_tag(tag)?, "a")?,
                "r" => set_once(
                    &mut earliest_unique_commit,
                    GitObjectId::from_hex(&single_value(tag, "r", 64)?)?,
                    "r",
                )?,
                "p" => recipients.push(parse_public_key_tag(tag, "p")?),
                "subject" => set_once(
                    &mut subject,
                    single_value(tag, "subject", MAX_SUBJECT_BYTES)?,
                    "subject",
                )?,
                "t" => labels.push(single_value(tag, "t", MAX_LABEL_BYTES)?),
                "c" => set_once(
                    &mut tip_commit,
                    GitObjectId::from_hex(&single_value(tag, "c", 64)?)?,
                    "c",
                )?,
                "clone" => set_once(&mut clone_urls, parse_clone_urls(tag)?, "clone")?,
                "h" => {
                    let value = single_value(tag, "h", 36)?;
                    let parsed = Uuid::parse_str(&value)
                        .map_err(|_| invalid_event("h tag must be a canonical UUID"))?;
                    if parsed.to_string() != value {
                        return Err(invalid_event("h tag must be a canonical UUID"));
                    }
                    set_once(&mut channel_id, parsed, "h")?;
                }
                "branch-name" => {
                    set_once(&mut branch_name, parse_branch_name(tag)?, "branch-name")?
                }
                "merge-base" => set_once(
                    &mut merge_base,
                    GitObjectId::from_hex(&single_value(tag, "merge-base", 64)?)?,
                    "merge-base",
                )?,
                "e" => set_once(
                    &mut revision_of,
                    parse_bare_event_tag(tag, "revision e")?,
                    "revision e",
                )?,
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }
        let repository = repository.ok_or_else(|| invalid_event("missing repository a tag"))?;
        validate_recipients(&recipients, repository.owner)?;
        validate_labels(&labels)?;
        Ok(Self {
            repository,
            earliest_unique_commit,
            recipients,
            subject: subject.ok_or_else(|| invalid_event("missing subject tag"))?,
            labels,
            tip_commit: tip_commit.ok_or_else(|| invalid_event("missing c tag"))?,
            clone_urls: clone_urls.ok_or_else(|| invalid_event("missing clone tag"))?,
            channel_id,
            branch_name,
            merge_base,
            revision_of,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34CollaborationCodecError> {
        validate_content(&self.content, MAX_CONTENT_BYTES, true)?;
        validate_text(&self.subject, "subject", MAX_SUBJECT_BYTES)?;
        validate_recipients(&self.recipients, self.repository.owner)?;
        validate_labels(&self.labels)?;
        validate_clone_urls(&self.clone_urls)?;
        if let Some(branch_name) = &self.branch_name {
            validate_branch_name(branch_name)?;
        }
        validate_extra_tags(&self.extra_tags, pull_request_reserved_tag)?;
        let mut tags = vec![self.repository.to_tag()?];
        if let Some(euc) = &self.earliest_unique_commit {
            tags.push(vec!["r".into(), euc.to_string()]);
        }
        tags.extend(
            self.recipients
                .iter()
                .map(|recipient| vec!["p".into(), recipient.to_hex()]),
        );
        tags.push(vec!["subject".into(), self.subject.clone()]);
        tags.extend(
            self.labels
                .iter()
                .map(|label| vec!["t".into(), label.clone()]),
        );
        tags.push(vec!["c".into(), self.tip_commit.to_string()]);
        if let Some(channel_id) = self.channel_id {
            tags.push(vec!["h".into(), channel_id.to_string()]);
        }
        tags.push(clone_tag(&self.clone_urls));
        if let Some(branch_name) = &self.branch_name {
            tags.push(vec!["branch-name".into(), branch_name.clone()]);
        }
        if let Some(merge_base) = &self.merge_base {
            tags.push(vec!["merge-base".into(), merge_base.to_string()]);
        }
        if let Some(revision_of) = self.revision_of {
            tags.push(vec!["e".into(), revision_of.to_hex()]);
        }
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            KIND_GIT_PULL_REQUEST as u16,
            tags,
            self.content.clone(),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitPullRequestUpdate {
    pub repository: RepositoryCoordinate,
    pub earliest_unique_commit: Option<GitObjectId>,
    pub recipients: Vec<PublicKey>,
    pub pull_request_event: EventId,
    pub pull_request_author: PublicKey,
    pub tip_commit: GitObjectId,
    pub clone_urls: Vec<String>,
    pub merge_base: Option<GitObjectId>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl GitPullRequestUpdate {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        require_kind(event, KIND_GIT_PR_UPDATE)?;
        validate_content(&event.content, MAX_CONTENT_BYTES, true)?;
        let mut repository = None;
        let mut earliest_unique_commit = None;
        let mut recipients = Vec::new();
        let mut pull_request_event = None;
        let mut pull_request_author = None;
        let mut tip_commit = None;
        let mut clone_urls = None;
        let mut merge_base = None;
        let mut extra_tags = Vec::new();
        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "a" => set_once(&mut repository, RepositoryCoordinate::parse_tag(tag)?, "a")?,
                "r" => set_once(
                    &mut earliest_unique_commit,
                    GitObjectId::from_hex(&single_value(tag, "r", 64)?)?,
                    "r",
                )?,
                "p" => recipients.push(parse_public_key_tag(tag, "p")?),
                "E" => set_once(
                    &mut pull_request_event,
                    parse_bare_event_tag(tag, "E")?,
                    "E",
                )?,
                "P" => set_once(
                    &mut pull_request_author,
                    parse_public_key_tag(tag, "P")?,
                    "P",
                )?,
                "c" => set_once(
                    &mut tip_commit,
                    GitObjectId::from_hex(&single_value(tag, "c", 64)?)?,
                    "c",
                )?,
                "clone" => set_once(&mut clone_urls, parse_clone_urls(tag)?, "clone")?,
                "merge-base" => set_once(
                    &mut merge_base,
                    GitObjectId::from_hex(&single_value(tag, "merge-base", 64)?)?,
                    "merge-base",
                )?,
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }
        let repository = repository.ok_or_else(|| invalid_event("missing repository a tag"))?;
        validate_recipients(&recipients, repository.owner)?;
        let pull_request_author = pull_request_author
            .ok_or_else(|| invalid_event("missing pull request author P tag"))?;
        Ok(Self {
            repository,
            earliest_unique_commit,
            recipients,
            pull_request_event: pull_request_event
                .ok_or_else(|| invalid_event("missing pull request E tag"))?,
            pull_request_author,
            tip_commit: tip_commit.ok_or_else(|| invalid_event("missing c tag"))?,
            clone_urls: clone_urls.ok_or_else(|| invalid_event("missing clone tag"))?,
            merge_base,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34CollaborationCodecError> {
        validate_content(&self.content, MAX_CONTENT_BYTES, true)?;
        validate_recipients(&self.recipients, self.repository.owner)?;
        validate_clone_urls(&self.clone_urls)?;
        validate_extra_tags(&self.extra_tags, pull_request_update_reserved_tag)?;
        let mut tags = vec![self.repository.to_tag()?];
        if let Some(euc) = &self.earliest_unique_commit {
            tags.push(vec!["r".into(), euc.to_string()]);
        }
        tags.extend(
            self.recipients
                .iter()
                .map(|recipient| vec!["p".into(), recipient.to_hex()]),
        );
        tags.push(vec!["E".into(), self.pull_request_event.to_hex()]);
        tags.push(vec!["P".into(), self.pull_request_author.to_hex()]);
        tags.push(vec!["c".into(), self.tip_commit.to_string()]);
        tags.push(clone_tag(&self.clone_urls));
        if let Some(merge_base) = &self.merge_base {
            tags.push(vec!["merge-base".into(), merge_base.to_string()]);
        }
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            KIND_GIT_PR_UPDATE as u16,
            tags,
            self.content.clone(),
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitThreadKind {
    Patch,
    PullRequest,
    Issue,
}

impl GitThreadKind {
    pub fn event_kind(self) -> u16 {
        match self {
            Self::Patch => KIND_GIT_PATCH as u16,
            Self::PullRequest => KIND_GIT_PULL_REQUEST as u16,
            Self::Issue => KIND_GIT_ISSUE as u16,
        }
    }

    fn from_event_kind(value: &str) -> Result<Self, Nip34CollaborationCodecError> {
        let kind = value
            .parse::<u16>()
            .map_err(|_| invalid_event("comment root kind is not a NIP-34 thread"))?;
        Self::from_kind(kind)
    }

    fn from_kind(kind: u16) -> Result<Self, Nip34CollaborationCodecError> {
        match kind {
            kind if kind == KIND_GIT_PATCH as u16 => Ok(Self::Patch),
            kind if kind == KIND_GIT_PULL_REQUEST as u16 => Ok(Self::PullRequest),
            kind if kind == KIND_GIT_ISSUE as u16 => Ok(Self::Issue),
            _ => Err(invalid_event("event kind is not a NIP-34 thread")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitThreadTarget {
    pub event_id: EventId,
    pub kind: GitThreadKind,
    pub author: PublicKey,
    pub repository: RepositoryCoordinate,
}

impl GitThreadTarget {
    pub fn from_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        let kind = GitThreadKind::from_kind(event.kind)?;
        let repository = match kind {
            GitThreadKind::Patch => GitPatch::parse_event(event)?.repository,
            GitThreadKind::PullRequest => GitPullRequest::parse_event(event)?.repository,
            GitThreadKind::Issue => GitIssue::parse_event(event)?.repository,
        };
        let event_id = event.event_id().map_err(|error| {
            invalid_event(format!("failed to compute thread event id: {error}"))
        })?;
        Ok(Self {
            event_id,
            kind,
            author: event.public_key,
            repository,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommentEventReference {
    pub event_id: EventId,
    pub relay_hint: String,
    pub author: PublicKey,
    pub author_relay_hint: String,
}

impl CommentEventReference {
    fn to_tag(&self, name: &str) -> Result<Vec<String>, Nip34CollaborationCodecError> {
        validate_optional_relay(&self.relay_hint)?;
        Ok(vec![
            name.into(),
            self.event_id.to_hex(),
            self.relay_hint.clone(),
            self.author.to_hex(),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParsedCommentEventReference {
    event_id: EventId,
    relay_hint: String,
    author_hint: Option<PublicKey>,
}

impl ParsedCommentEventReference {
    fn parse_tag(tag: &[String], name: &str) -> Result<Self, Nip34CollaborationCodecError> {
        if !(2..=4).contains(&tag.len()) || tag.first().map(String::as_str) != Some(name) {
            return Err(invalid_event(format!("{name} tag has an invalid shape")));
        }
        let relay_hint = tag.get(2).cloned().unwrap_or_default();
        validate_optional_relay(&relay_hint)?;
        Ok(Self {
            event_id: parse_event_id(&tag[1], name)?,
            relay_hint,
            author_hint: tag
                .get(3)
                .map(|value| parse_public_key(value, name))
                .transpose()?,
        })
    }

    fn resolve_author(
        self,
        author: PublicKey,
        author_relay_hint: String,
        name: &str,
    ) -> Result<CommentEventReference, Nip34CollaborationCodecError> {
        if self.author_hint.is_some_and(|hint| hint != author) {
            return Err(invalid_event(format!(
                "{name} event and author tags differ"
            )));
        }
        Ok(CommentEventReference {
            event_id: self.event_id,
            relay_hint: self.relay_hint,
            author,
            author_relay_hint,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommentPubkeyReference {
    pub public_key: PublicKey,
    pub relay_hint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommentParentKind {
    Root(GitThreadKind),
    Comment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitComment {
    pub root: CommentEventReference,
    pub root_kind: GitThreadKind,
    pub parent: CommentEventReference,
    pub parent_kind: CommentParentKind,
    pub mentions: Vec<CommentPubkeyReference>,
    pub content: String,
    pub extra_tags: Vec<Vec<String>>,
}

impl GitComment {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, Nip34CollaborationCodecError> {
        if event.kind != KIND_COMMENT {
            return Err(Nip34CollaborationCodecError::UnsupportedKind(event.kind));
        }
        validate_content(&event.content, MAX_CONTENT_BYTES, false)?;
        let mut root = None;
        let mut root_kind = None;
        let mut root_author = None;
        let mut parent = None;
        let mut parent_kind_value = None;
        let mut parent_authors = Vec::new();
        let mut extra_tags = Vec::new();
        for tag in &event.tags {
            let Some(name) = tag.first().map(String::as_str) else {
                return Err(invalid_event("tag must not be empty"));
            };
            match name {
                "E" => set_once(
                    &mut root,
                    ParsedCommentEventReference::parse_tag(tag, "E")?,
                    "E",
                )?,
                "K" => set_once(
                    &mut root_kind,
                    GitThreadKind::from_event_kind(&single_value(tag, "K", 5)?)?,
                    "K",
                )?,
                "P" => set_once(
                    &mut root_author,
                    parse_public_key_tag_with_relay(tag, "P")?,
                    "P",
                )?,
                "e" => set_once(
                    &mut parent,
                    ParsedCommentEventReference::parse_tag(tag, "e")?,
                    "e",
                )?,
                "k" => set_once(&mut parent_kind_value, single_value(tag, "k", 5)?, "k")?,
                "p" => parent_authors.push(parse_public_key_tag_with_relay(tag, "p")?),
                _ => push_extra_tag(&mut extra_tags, tag)?,
            }
        }
        let root = root.ok_or_else(|| invalid_event("missing root E tag"))?;
        let root_kind = root_kind.ok_or_else(|| invalid_event("missing root K tag"))?;
        let (root_author, root_author_relay) =
            root_author.ok_or_else(|| invalid_event("missing root P tag"))?;
        let root = root.resolve_author(root_author, root_author_relay, "root")?;
        let parent = parent.ok_or_else(|| invalid_event("missing parent e tag"))?;
        let parent_author_index = match parent.author_hint {
            Some(author) => parent_authors
                .iter()
                .position(|(candidate, _)| *candidate == author)
                .ok_or_else(|| invalid_event("parent event author is missing from p tags"))?,
            None => {
                if parent_authors.len() != 1 {
                    return Err(invalid_event(
                        "parent e tag without an author requires exactly one p tag",
                    ));
                }
                0
            }
        };
        let (parent_author, parent_author_relay) = parent_authors.remove(parent_author_index);
        let parent = parent.resolve_author(parent_author, parent_author_relay, "parent")?;
        let parent_kind_value = parent_kind_value.ok_or_else(|| invalid_event("missing k tag"))?;
        let parent_kind = if parent_kind_value == KIND_COMMENT.to_string() {
            CommentParentKind::Comment
        } else {
            CommentParentKind::Root(GitThreadKind::from_event_kind(&parent_kind_value)?)
        };
        validate_comment_ancestry(root_kind, &root, &parent_kind, &parent)?;
        let mentions = parent_authors
            .into_iter()
            .map(|(public_key, relay_hint)| CommentPubkeyReference {
                public_key,
                relay_hint,
            })
            .collect::<Vec<_>>();
        ensure_unique(
            mentions.iter().map(|mention| mention.public_key),
            "comment mention",
        )?;
        Ok(Self {
            root,
            root_kind,
            parent,
            parent_kind,
            mentions,
            content: event.content.clone(),
            extra_tags,
        })
    }

    pub fn to_event(
        &self,
        author: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, Nip34CollaborationCodecError> {
        validate_content(&self.content, MAX_CONTENT_BYTES, false)?;
        validate_comment_ancestry(self.root_kind, &self.root, &self.parent_kind, &self.parent)?;
        if self
            .mentions
            .iter()
            .any(|mention| mention.public_key == self.parent.author)
        {
            return Err(invalid_event(
                "parent author must not be duplicated in comment mentions",
            ));
        }
        ensure_unique(
            self.mentions.iter().map(|mention| mention.public_key),
            "comment mention",
        )?;
        validate_extra_tags(&self.extra_tags, comment_reserved_tag)?;
        let mut tags = vec![
            self.root.to_tag("E")?,
            vec!["K".into(), self.root_kind.event_kind().to_string()],
            public_key_with_relay_tag("P", self.root.author, &self.root.author_relay_hint)?,
            self.parent.to_tag("e")?,
            vec![
                "k".into(),
                match self.parent_kind {
                    CommentParentKind::Root(kind) => kind.event_kind(),
                    CommentParentKind::Comment => KIND_COMMENT,
                }
                .to_string(),
            ],
            public_key_with_relay_tag("p", self.parent.author, &self.parent.author_relay_hint)?,
        ];
        tags.extend(
            self.mentions
                .iter()
                .map(|mention| {
                    public_key_with_relay_tag("p", mention.public_key, &mention.relay_hint)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        tags.extend(self.extra_tags.clone());
        Ok(CanonicalEvent::new(
            author,
            created_at,
            KIND_COMMENT,
            tags,
            self.content.clone(),
        ))
    }
}

pub fn validate_status_reference(
    status: &RepositoryStatusEvent,
    target: &GitThreadTarget,
) -> Result<(), Nip34CollaborationCodecError> {
    if status.root_event != target.event_id {
        return Err(invalid_event(
            "status root does not match the thread target",
        ));
    }
    if !status.recipients.contains(&target.author) {
        return Err(invalid_event("status omits the thread author recipient"));
    }
    if let Some(repository) = &status.repository {
        if repository.value() != target.repository.value() {
            return Err(invalid_event(
                "status repository does not match the thread target",
            ));
        }
    }
    Ok(())
}

fn validate_comment_ancestry(
    root_kind: GitThreadKind,
    root: &CommentEventReference,
    parent_kind: &CommentParentKind,
    parent: &CommentEventReference,
) -> Result<(), Nip34CollaborationCodecError> {
    match parent_kind {
        CommentParentKind::Root(kind)
            if *kind == root_kind
                && parent.event_id == root.event_id
                && parent.author == root.author =>
        {
            Ok(())
        }
        CommentParentKind::Comment if parent.event_id != root.event_id => Ok(()),
        CommentParentKind::Root(_) => Err(invalid_event(
            "top-level comment parent must equal its NIP-34 root",
        )),
        CommentParentKind::Comment => Err(invalid_event(
            "nested comment parent must differ from the NIP-34 root",
        )),
    }
}

fn require_kind(event: &CanonicalEvent, kind: u32) -> Result<(), Nip34CollaborationCodecError> {
    if u32::from(event.kind) != kind {
        return Err(Nip34CollaborationCodecError::UnsupportedKind(event.kind));
    }
    Ok(())
}

fn parse_public_key_tag(
    tag: &[String],
    name: &str,
) -> Result<PublicKey, Nip34CollaborationCodecError> {
    if tag.len() != 2 || tag.first().map(String::as_str) != Some(name) {
        return Err(invalid_event(format!(
            "{name} tag must contain exactly one public key"
        )));
    }
    parse_public_key(&tag[1], name)
}

fn parse_public_key_tag_with_relay(
    tag: &[String],
    name: &str,
) -> Result<(PublicKey, String), Nip34CollaborationCodecError> {
    if !(2..=3).contains(&tag.len()) || tag.first().map(String::as_str) != Some(name) {
        return Err(invalid_event(format!("{name} tag has an invalid shape")));
    }
    let relay_hint = tag.get(2).cloned().unwrap_or_default();
    validate_optional_relay(&relay_hint)?;
    Ok((parse_public_key(&tag[1], name)?, relay_hint))
}

fn public_key_with_relay_tag(
    name: &str,
    public_key: PublicKey,
    relay_hint: &str,
) -> Result<Vec<String>, Nip34CollaborationCodecError> {
    validate_optional_relay(relay_hint)?;
    if relay_hint.is_empty() {
        Ok(vec![name.into(), public_key.to_hex()])
    } else {
        Ok(vec![name.into(), public_key.to_hex(), relay_hint.into()])
    }
}

fn parse_public_key(value: &str, name: &str) -> Result<PublicKey, Nip34CollaborationCodecError> {
    PublicKey::from_hex(value)
        .map_err(|error| invalid_event(format!("invalid {name} public key: {error}")))
}

fn parse_event_id(value: &str, name: &str) -> Result<EventId, Nip34CollaborationCodecError> {
    EventId::from_hex(value)
        .map_err(|error| invalid_event(format!("invalid {name} event id: {error}")))
}

fn parse_bare_event_tag(
    tag: &[String],
    name: &str,
) -> Result<EventId, Nip34CollaborationCodecError> {
    if tag.len() != 2 {
        return Err(invalid_event(format!(
            "{name} tag must contain exactly one event id"
        )));
    }
    parse_event_id(&tag[1], name)
}

fn validate_recipients(
    recipients: &[PublicKey],
    repository_owner: PublicKey,
) -> Result<(), Nip34CollaborationCodecError> {
    if recipients.is_empty() || recipients.len() > MAX_RECIPIENTS {
        return Err(invalid_event("recipient count is outside 1-256"));
    }
    ensure_unique(recipients.iter().copied(), "recipient")?;
    if !recipients.contains(&repository_owner) {
        return Err(invalid_event("repository owner is missing from p tags"));
    }
    Ok(())
}

fn validate_labels(labels: &[String]) -> Result<(), Nip34CollaborationCodecError> {
    if labels.len() > MAX_LABELS {
        return Err(invalid_event("label count exceeds 128"));
    }
    for label in labels {
        validate_text(label, "label", MAX_LABEL_BYTES)?;
    }
    ensure_unique(labels.iter().cloned(), "label")
}

fn parse_clone_urls(tag: &[String]) -> Result<Vec<String>, Nip34CollaborationCodecError> {
    if tag.first().map(String::as_str) != Some("clone") {
        return Err(invalid_event("invalid clone tag"));
    }
    let urls = tag[1..].to_vec();
    validate_clone_urls(&urls)?;
    Ok(urls)
}

fn validate_clone_urls(urls: &[String]) -> Result<(), Nip34CollaborationCodecError> {
    if urls.is_empty() || urls.len() > MAX_CLONE_URLS {
        return Err(invalid_event("clone URL count is outside 1-5"));
    }
    for url in urls {
        if url.is_empty() || url.len() > MAX_CLONE_URL_BYTES || url.chars().any(char::is_control) {
            return Err(invalid_event("clone URL must contain 1-512 safe bytes"));
        }
    }
    ensure_unique(urls.iter().cloned(), "clone URL")
}

fn clone_tag(urls: &[String]) -> Vec<String> {
    let mut tag = vec!["clone".into()];
    tag.extend(urls.iter().cloned());
    tag
}

fn parse_branch_name(tag: &[String]) -> Result<String, Nip34CollaborationCodecError> {
    let value = single_value(tag, "branch-name", MAX_BRANCH_NAME_BYTES)?;
    validate_branch_name(&value)?;
    Ok(value)
}

fn validate_branch_name(value: &str) -> Result<(), Nip34CollaborationCodecError> {
    if value.starts_with('-')
        || value.starts_with('.')
        || value.ends_with('/')
        || value.ends_with('.')
        || value.ends_with(".lock")
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.len() > MAX_BRANCH_NAME_BYTES
        || value.bytes().any(|byte| {
            byte <= b' '
                || byte == 0x7f
                || matches!(byte, b'~' | b'^' | b':' | b'?' | b'*' | b'[' | b'\\')
        })
        || value.split('/').any(|component| {
            component.is_empty()
                || component.starts_with('.')
                || component.ends_with('.')
                || component.ends_with(".lock")
        })
    {
        return Err(invalid_event("invalid recommended branch name"));
    }
    Ok(())
}

fn validate_content(
    content: &str,
    maximum: usize,
    allow_empty: bool,
) -> Result<(), Nip34CollaborationCodecError> {
    if content.len() > maximum || (!allow_empty && content.is_empty()) {
        return Err(invalid_event(format!(
            "content must contain {}-{maximum} bytes",
            usize::from(!allow_empty)
        )));
    }
    Ok(())
}

fn validate_text(
    value: &str,
    name: &str,
    maximum: usize,
) -> Result<(), Nip34CollaborationCodecError> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(invalid_event(format!(
            "{name} must contain 1-{maximum} safe bytes"
        )));
    }
    Ok(())
}

fn validate_optional_relay(value: &str) -> Result<(), Nip34CollaborationCodecError> {
    if !value.is_empty()
        && (value.len() > MAX_RELAY_URL_BYTES
            || !(value.starts_with("ws://") || value.starts_with("wss://"))
            || value.chars().any(char::is_control))
    {
        return Err(invalid_event("invalid relay hint"));
    }
    Ok(())
}

fn single_value(
    tag: &[String],
    name: &str,
    maximum: usize,
) -> Result<String, Nip34CollaborationCodecError> {
    if tag.len() != 2 || tag[1].is_empty() || tag[1].len() > maximum {
        return Err(invalid_event(format!(
            "{name} tag must contain one nonempty value of at most {maximum} bytes"
        )));
    }
    Ok(tag[1].clone())
}

fn require_exact_len(
    tag: &[String],
    name: &str,
    length: usize,
) -> Result<(), Nip34CollaborationCodecError> {
    if tag.len() != length {
        return Err(invalid_event(format!(
            "{name} tag must contain {length} values"
        )));
    }
    Ok(())
}

fn set_once<T>(
    slot: &mut Option<T>,
    value: T,
    name: &str,
) -> Result<(), Nip34CollaborationCodecError> {
    if slot.replace(value).is_some() {
        return Err(invalid_event(format!("duplicate {name} tag")));
    }
    Ok(())
}

fn ensure_unique<T>(
    values: impl IntoIterator<Item = T>,
    name: &str,
) -> Result<(), Nip34CollaborationCodecError>
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

fn push_extra_tag(
    tags: &mut Vec<Vec<String>>,
    tag: &[String],
) -> Result<(), Nip34CollaborationCodecError> {
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
) -> Result<(), Nip34CollaborationCodecError> {
    if tags.len() > MAX_EXTRA_TAGS {
        return Err(invalid_event("event exceeds extra tag limit"));
    }
    for tag in tags {
        validate_tag_bounds(tag)?;
        if reserved(&tag[0]) {
            return Err(invalid_event("extra tag uses a reserved tag name"));
        }
    }
    Ok(())
}

fn validate_tag_bounds(tag: &[String]) -> Result<(), Nip34CollaborationCodecError> {
    if tag.is_empty()
        || tag.len() > MAX_TAG_VALUES
        || tag.iter().any(|value| value.len() > MAX_TAG_VALUE_BYTES)
    {
        return Err(invalid_event("extra tag exceeds structural limits"));
    }
    Ok(())
}

fn patch_reserved_tag(name: &str) -> bool {
    matches!(
        name,
        "a" | "r" | "p" | "e" | "t" | "commit" | "parent-commit" | "commit-pgp-sig" | "committer"
    )
}

fn issue_reserved_tag(name: &str) -> bool {
    matches!(name, "a" | "p" | "subject" | "t")
}

fn pull_request_reserved_tag(name: &str) -> bool {
    matches!(
        name,
        "a" | "r"
            | "p"
            | "subject"
            | "t"
            | "c"
            | "clone"
            | "h"
            | "branch-name"
            | "merge-base"
            | "e"
    )
}

fn pull_request_update_reserved_tag(name: &str) -> bool {
    matches!(
        name,
        "a" | "r" | "p" | "E" | "P" | "c" | "clone" | "merge-base"
    )
}

fn comment_reserved_tag(name: &str) -> bool {
    matches!(name, "E" | "K" | "P" | "e" | "k" | "p")
}

fn invalid_event(message: impl Into<String>) -> Nip34CollaborationCodecError {
    Nip34CollaborationCodecError::InvalidEvent(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nip34_repository::{RepositoryStatus, RepositoryStatusEvent};

    const OWNER: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const AUTHOR: &str = "c6047f9441ed7d6d3045406e95c07cd85a207230f3dc9c0db865c3e0b0f2bdc7";
    const REVIEWER: &str = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const PARENT_COMMIT: &str = "89abcdef0123456789abcdef0123456789abcdef";
    const ROOT_PATCH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PARENT_COMMENT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("valid public key fixture")
    }

    fn event(kind: u16, tags: Vec<Vec<String>>, content: &str) -> CanonicalEvent {
        CanonicalEvent::new(key(AUTHOR), 1_787_356_800, kind, tags, content.into())
    }

    fn repository() -> RepositoryCoordinate {
        RepositoryCoordinate::parse(&format!("30617:{OWNER}:zed"))
            .expect("valid repository coordinate")
    }

    #[test]
    fn nip34_collaboration_patch_series_and_revision_golden_round_trip() {
        let root_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["a","30617:{OWNER}:zed"],["r","{PARENT_COMMIT}","euc"],["p","{OWNER}"],["p","{REVIEWER}"],["t","root"],["commit","{COMMIT}"],["r","{COMMIT}"],["parent-commit","{PARENT_COMMIT}"],["committer","Ada","ada@example.com","1787356800","60"],["future-patch","preserved"]]"#
        ))
        .expect("valid patch tags");
        let root = event(KIND_GIT_PATCH as u16, root_tags, "diff --git a/x b/x");
        let parsed = GitPatch::parse_event(&root).expect("valid root patch");
        assert_eq!(parsed.position, PatchPosition::Root);
        assert_eq!(
            parsed
                .to_event(root.public_key, root.created_at)
                .expect("encodable root patch"),
            root
        );

        let revision_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["a","30617:{OWNER}:zed"],["p","{OWNER}"],["e","{ROOT_PATCH}","","reply"],["t","root-revision"]]"#
        ))
        .expect("valid revision tags");
        let revision = event(KIND_GIT_PATCH as u16, revision_tags, "revised patch");
        let parsed = GitPatch::parse_event(&revision).expect("valid patch revision");
        assert_eq!(parsed.position, PatchPosition::RootRevision);
        assert_eq!(
            parsed.reply_to,
            Some(parse_event_id(ROOT_PATCH, "fixture").expect("valid fixture event id"))
        );
        assert_eq!(
            parsed
                .to_event(revision.public_key, revision.created_at)
                .expect("encodable patch revision"),
            revision
        );
    }

    #[test]
    fn nip34_collaboration_pull_request_and_update_golden_round_trip() {
        let pull_request_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["a","30617:{OWNER}:zed"],["r","{PARENT_COMMIT}"],["p","{OWNER}"],["subject","Add collaboration"],["t","feature"],["c","{COMMIT}"],["h","11111111-1111-4111-8111-111111111111"],["clone","https://example.com/zed.git","git@example.com:zed.git"],["branch-name","feature/collaboration"],["merge-base","{PARENT_COMMIT}"],["e","{ROOT_PATCH}"],["future-pr","preserved"]]"#
        ))
        .expect("valid pull request tags");
        let pull_request = event(
            KIND_GIT_PULL_REQUEST as u16,
            pull_request_tags,
            "Pull request body",
        );
        let parsed = GitPullRequest::parse_event(&pull_request).expect("valid pull request");
        assert_eq!(parsed.clone_urls.len(), 2);
        assert_eq!(
            parsed
                .to_event(pull_request.public_key, pull_request.created_at)
                .expect("encodable pull request"),
            pull_request
        );

        let update_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["a","30617:{OWNER}:zed"],["r","{PARENT_COMMIT}"],["p","{OWNER}"],["E","{ROOT_PATCH}"],["P","{AUTHOR}"],["c","{COMMIT}"],["clone","https://example.com/zed.git"],["merge-base","{PARENT_COMMIT}"]]"#
        ))
        .expect("valid pull request update tags");
        let update = event(KIND_GIT_PR_UPDATE as u16, update_tags, "Rebased");
        let parsed = GitPullRequestUpdate::parse_event(&update).expect("valid PR update");
        assert_eq!(parsed.pull_request_author, key(AUTHOR));
        assert_eq!(
            parsed
                .to_event(update.public_key, update.created_at)
                .expect("encodable PR update"),
            update
        );
    }

    #[test]
    fn nip34_collaboration_issue_and_comment_links_golden_round_trip() {
        let issue_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["a","30617:{OWNER}:zed"],["p","{OWNER}"],["subject","Crash on launch"],["t","bug"],["future-issue","preserved"]]"#
        ))
        .expect("valid issue tags");
        let issue = event(KIND_GIT_ISSUE as u16, issue_tags, "Steps to reproduce");
        let parsed_issue = GitIssue::parse_event(&issue).expect("valid issue");
        assert_eq!(
            parsed_issue
                .to_event(issue.public_key, issue.created_at)
                .expect("encodable issue"),
            issue
        );
        let issue_id = issue.event_id().expect("issue event id");
        let target = GitThreadTarget::from_event(&issue).expect("validated issue target");
        assert_eq!(target.event_id, issue_id);
        assert_eq!(target.repository.value(), repository().value());

        let comment_tags: Vec<Vec<String>> = serde_json::from_str(&format!(
            r#"[["E","{}","wss://relay.example","{AUTHOR}"],["K","1621"],["P","{AUTHOR}","wss://relay.example"],["e","{}","wss://relay.example","{AUTHOR}"],["k","1621"],["p","{AUTHOR}","wss://relay.example"],["p","{REVIEWER}"],["future-comment","preserved"]]"#,
            issue_id.to_hex(),
            issue_id.to_hex()
        ))
        .expect("valid comment tags");
        let comment = event(KIND_COMMENT, comment_tags, "I can reproduce this.");
        let parsed = GitComment::parse_event(&comment).expect("valid issue comment");
        assert_eq!(parsed.root_kind, GitThreadKind::Issue);
        assert_eq!(
            parsed.mentions,
            vec![CommentPubkeyReference {
                public_key: key(REVIEWER),
                relay_hint: String::new(),
            }]
        );
        assert_eq!(
            parsed
                .to_event(comment.public_key, comment.created_at)
                .expect("encodable issue comment"),
            comment
        );
    }

    #[test]
    fn nip34_collaboration_nested_comment_and_status_references_validate() {
        let root = CommentEventReference {
            event_id: parse_event_id(ROOT_PATCH, "fixture").expect("root id"),
            relay_hint: "wss://relay.example".into(),
            author: key(AUTHOR),
            author_relay_hint: "wss://relay.example".into(),
        };
        let parent = CommentEventReference {
            event_id: parse_event_id(PARENT_COMMENT, "fixture").expect("parent id"),
            relay_hint: String::new(),
            author: key(REVIEWER),
            author_relay_hint: String::new(),
        };
        let comment = GitComment {
            root: root.clone(),
            root_kind: GitThreadKind::Patch,
            parent,
            parent_kind: CommentParentKind::Comment,
            mentions: vec![],
            content: "Nested reply".into(),
            extra_tags: vec![],
        };
        let encoded = comment
            .to_event(key(AUTHOR), 1_787_356_801)
            .expect("encodable nested comment");
        assert_eq!(
            GitComment::parse_event(&encoded).expect("valid nested comment"),
            comment
        );

        let target = GitThreadTarget {
            event_id: root.event_id,
            kind: GitThreadKind::Patch,
            author: root.author,
            repository: repository(),
        };
        let status = RepositoryStatusEvent {
            status: RepositoryStatus::Open,
            root_event: root.event_id,
            accepted_revision_root: None,
            recipients: vec![root.author],
            repository: Some(repository()),
            earliest_unique_commit: None,
            applied_patches: vec![],
            merge_commit: None,
            applied_as_commits: vec![],
            content: String::new(),
            extra_tags: vec![],
        };
        validate_status_reference(&status, &target).expect("valid status reference");

        let wrong_target = GitThreadTarget {
            event_id: parse_event_id(PARENT_COMMENT, "fixture").expect("other event id"),
            ..target
        };
        assert!(validate_status_reference(&status, &wrong_target).is_err());
    }

    #[test]
    fn nip34_collaboration_rejects_invalid_ancestry_and_required_links() {
        let invalid_patch = event(
            KIND_GIT_PATCH as u16,
            vec![
                repository().to_tag().expect("repo tag"),
                vec!["p".into(), OWNER.into()],
                vec!["t".into(), "root".into()],
                vec!["t".into(), "root-revision".into()],
            ],
            "patch",
        );
        assert!(GitPatch::parse_event(&invalid_patch).is_err());

        let root = CommentEventReference {
            event_id: parse_event_id(ROOT_PATCH, "fixture").expect("root id"),
            relay_hint: String::new(),
            author: key(AUTHOR),
            author_relay_hint: String::new(),
        };
        let invalid_comment = GitComment {
            root,
            root_kind: GitThreadKind::Issue,
            parent: CommentEventReference {
                event_id: parse_event_id(PARENT_COMMENT, "fixture").expect("parent id"),
                relay_hint: String::new(),
                author: key(AUTHOR),
                author_relay_hint: String::new(),
            },
            parent_kind: CommentParentKind::Root(GitThreadKind::Issue),
            mentions: vec![],
            content: "invalid".into(),
            extra_tags: vec![],
        };
        assert!(invalid_comment.to_event(key(REVIEWER), 1).is_err());

        let missing_owner_issue = event(
            KIND_GIT_ISSUE as u16,
            vec![
                repository().to_tag().expect("repo tag"),
                vec!["p".into(), REVIEWER.into()],
                vec!["subject".into(), "Missing owner".into()],
            ],
            "",
        );
        assert!(GitIssue::parse_event(&missing_owner_issue).is_err());
    }
}
