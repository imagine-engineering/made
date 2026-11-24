use std::{error::Error, fmt, num::NonZeroU64};

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{AggregateId, AggregateVersion, CommunityId};

const MAX_BRANCH_REF_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BranchRefName(String);

impl BranchRefName {
    pub fn parse(value: impl Into<String>) -> Result<Self, BranchCollaborationError> {
        let value = value.into();
        if !is_safe_branch_ref(&value) {
            return Err(BranchCollaborationError::InvalidBranchRef);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for BranchRefName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct GitCommitId(String);

impl GitCommitId {
    pub fn parse(value: impl Into<String>) -> Result<Self, BranchCollaborationError> {
        let value = value.into();
        if !is_lower_hex(&value, 40) && !is_lower_hex(&value, 64) {
            return Err(BranchCollaborationError::InvalidCommitId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for GitCommitId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BranchGeneration(NonZeroU64);

impl BranchGeneration {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub fn next(self) -> Option<Self> {
        self.get().checked_add(1).and_then(Self::new)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BranchCollaborationIdentity {
    community_id: CommunityId,
    repository_id: AggregateId,
    branch_ref: BranchRefName,
    generation: BranchGeneration,
}

impl BranchCollaborationIdentity {
    pub fn new(
        community_id: CommunityId,
        repository_id: AggregateId,
        branch_ref: BranchRefName,
        generation: BranchGeneration,
    ) -> Result<Self, BranchCollaborationError> {
        if community_id.as_uuid().is_nil() {
            return Err(BranchCollaborationError::InvalidCommunityId);
        }
        if repository_id.as_uuid().is_nil() {
            return Err(BranchCollaborationError::InvalidRepositoryId);
        }
        Ok(Self {
            community_id,
            repository_id,
            branch_ref,
            generation,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn repository_id(&self) -> AggregateId {
        self.repository_id
    }

    pub const fn branch_ref(&self) -> &BranchRefName {
        &self.branch_ref
    }

    pub const fn generation(&self) -> BranchGeneration {
        self.generation
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BranchCommitIdentity {
    branch: BranchCollaborationIdentity,
    commit: GitCommitId,
}

impl BranchCommitIdentity {
    pub fn new(branch: BranchCollaborationIdentity, commit: GitCommitId) -> Self {
        Self { branch, commit }
    }

    pub const fn branch(&self) -> &BranchCollaborationIdentity {
        &self.branch
    }

    pub const fn commit(&self) -> &GitCommitId {
        &self.commit
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchUpdateKind {
    FastForward,
    Force,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BranchHeadUpdate {
    previous_commit: GitCommitId,
    current_commit: GitCommitId,
    kind: BranchUpdateKind,
}

impl BranchHeadUpdate {
    pub const fn previous_commit(&self) -> &GitCommitId {
        &self.previous_commit
    }

    pub const fn current_commit(&self) -> &GitCommitId {
        &self.current_commit
    }

    pub const fn kind(&self) -> BranchUpdateKind {
        self.kind
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BranchMerge {
    source_commit: GitCommitId,
    target_branch: BranchRefName,
    result_commit: GitCommitId,
}

impl BranchMerge {
    pub const fn source_commit(&self) -> &GitCommitId {
        &self.source_commit
    }

    pub const fn target_branch(&self) -> &BranchRefName {
        &self.target_branch
    }

    pub const fn result_commit(&self) -> &GitCommitId {
        &self.result_commit
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchArchiveReason {
    Deleted,
    Merged,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "reason")]
pub enum BranchLifecycleState {
    Active,
    Merged,
    Archived(BranchArchiveReason),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BranchCollaborationRecordFields {
    pub identity: BranchCollaborationIdentity,
    pub head_commit: GitCommitId,
    pub last_head_update: Option<BranchHeadUpdate>,
    pub merge: Option<BranchMerge>,
    pub lifecycle_state: BranchLifecycleState,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchCollaboration {
    fields: BranchCollaborationRecordFields,
}

impl BranchCollaboration {
    pub fn create(
        identity: BranchCollaborationIdentity,
        head_commit: GitCommitId,
    ) -> Result<Self, BranchCollaborationError> {
        Self::from_record(BranchCollaborationRecordFields {
            identity,
            head_commit,
            last_head_update: None,
            merge: None,
            lifecycle_state: BranchLifecycleState::Active,
            version: AggregateVersion::FIRST,
        })
    }

    pub fn from_record(
        fields: BranchCollaborationRecordFields,
    ) -> Result<Self, BranchCollaborationError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &BranchCollaborationRecordFields {
        &self.fields
    }

    pub fn head_identity(&self) -> BranchCommitIdentity {
        BranchCommitIdentity::new(
            self.fields.identity.clone(),
            self.fields.head_commit.clone(),
        )
    }

    pub fn update_head(
        &mut self,
        expected_version: AggregateVersion,
        expected_commit: &GitCommitId,
        new_commit: GitCommitId,
        kind: BranchUpdateKind,
    ) -> Result<BranchCommandOutcome, BranchCollaborationError> {
        self.require_active()?;
        self.require_version(expected_version)?;
        self.require_head(expected_commit)?;
        if self.fields.head_commit == new_commit {
            return Ok(BranchCommandOutcome::Unchanged);
        }
        let next_version = self.next_version()?;
        self.fields.last_head_update = Some(BranchHeadUpdate {
            previous_commit: self.fields.head_commit.clone(),
            current_commit: new_commit.clone(),
            kind,
        });
        self.fields.head_commit = new_commit;
        self.fields.version = next_version;
        Ok(BranchCommandOutcome::Applied)
    }

    pub fn merge(
        &mut self,
        expected_version: AggregateVersion,
        expected_commit: &GitCommitId,
        target_branch: BranchRefName,
        result_commit: GitCommitId,
    ) -> Result<BranchCommandOutcome, BranchCollaborationError> {
        let merge = BranchMerge {
            source_commit: expected_commit.clone(),
            target_branch,
            result_commit,
        };
        self.require_version(expected_version)?;
        self.require_head(expected_commit)?;
        if self.fields.lifecycle_state == BranchLifecycleState::Merged
            && self.fields.merge.as_ref() == Some(&merge)
        {
            return Ok(BranchCommandOutcome::Unchanged);
        }
        self.require_active()?;
        if merge.target_branch == self.fields.identity.branch_ref {
            return Err(BranchCollaborationError::InvalidMergeTarget);
        }
        self.fields.version = self.next_version()?;
        self.fields.lifecycle_state = BranchLifecycleState::Merged;
        self.fields.merge = Some(merge);
        Ok(BranchCommandOutcome::Applied)
    }

    pub fn archive(
        &mut self,
        expected_version: AggregateVersion,
        expected_commit: &GitCommitId,
        reason: BranchArchiveReason,
    ) -> Result<BranchCommandOutcome, BranchCollaborationError> {
        self.require_version(expected_version)?;
        self.require_head(expected_commit)?;
        if self.fields.lifecycle_state == BranchLifecycleState::Archived(reason) {
            return Ok(BranchCommandOutcome::Unchanged);
        }
        match (self.fields.lifecycle_state, reason) {
            (BranchLifecycleState::Active, BranchArchiveReason::Deleted)
            | (BranchLifecycleState::Merged, BranchArchiveReason::Merged) => {}
            _ => return Err(BranchCollaborationError::InvalidTransition),
        }
        self.fields.version = self.next_version()?;
        self.fields.lifecycle_state = BranchLifecycleState::Archived(reason);
        Ok(BranchCommandOutcome::Applied)
    }

    pub fn recreate(
        &self,
        expected_version: AggregateVersion,
        expected_commit: &GitCommitId,
        head_commit: GitCommitId,
    ) -> Result<Self, BranchCollaborationError> {
        self.require_version(expected_version)?;
        self.require_head(expected_commit)?;
        if !matches!(
            self.fields.lifecycle_state,
            BranchLifecycleState::Archived(_)
        ) {
            return Err(BranchCollaborationError::InvalidTransition);
        }
        let generation = self
            .fields
            .identity
            .generation
            .next()
            .ok_or(BranchCollaborationError::GenerationExhausted)?;
        let identity = BranchCollaborationIdentity::new(
            self.fields.identity.community_id,
            self.fields.identity.repository_id,
            self.fields.identity.branch_ref.clone(),
            generation,
        )?;
        Self::create(identity, head_commit)
    }

    fn require_active(&self) -> Result<(), BranchCollaborationError> {
        if self.fields.lifecycle_state != BranchLifecycleState::Active {
            return Err(BranchCollaborationError::InvalidTransition);
        }
        Ok(())
    }

    fn require_version(
        &self,
        expected_version: AggregateVersion,
    ) -> Result<(), BranchCollaborationError> {
        if self.fields.version != expected_version {
            return Err(BranchCollaborationError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_head(&self, expected_commit: &GitCommitId) -> Result<(), BranchCollaborationError> {
        if &self.fields.head_commit != expected_commit {
            return Err(BranchCollaborationError::StaleCommit {
                expected: expected_commit.clone(),
                actual: self.fields.head_commit.clone(),
            });
        }
        Ok(())
    }

    fn next_version(&self) -> Result<AggregateVersion, BranchCollaborationError> {
        self.fields
            .version
            .next()
            .ok_or(BranchCollaborationError::VersionExhausted)
    }
}

fn validate_record(
    fields: &BranchCollaborationRecordFields,
) -> Result<(), BranchCollaborationError> {
    BranchCollaborationIdentity::new(
        fields.identity.community_id,
        fields.identity.repository_id,
        fields.identity.branch_ref.clone(),
        fields.identity.generation,
    )?;
    if fields.last_head_update.as_ref().is_some_and(|update| {
        update.previous_commit == update.current_commit
            || update.current_commit != fields.head_commit
    }) {
        return Err(BranchCollaborationError::InvalidRecord);
    }
    match (fields.lifecycle_state, &fields.merge) {
        (BranchLifecycleState::Active, None)
        | (BranchLifecycleState::Archived(BranchArchiveReason::Deleted), None) => {}
        (BranchLifecycleState::Merged, Some(merge))
        | (BranchLifecycleState::Archived(BranchArchiveReason::Merged), Some(merge)) => {
            if merge.source_commit != fields.head_commit
                || merge.target_branch == fields.identity.branch_ref
            {
                return Err(BranchCollaborationError::InvalidRecord);
            }
        }
        _ => return Err(BranchCollaborationError::InvalidRecord),
    }
    Ok(())
}

fn is_safe_branch_ref(value: &str) -> bool {
    if !value.starts_with("refs/heads/")
        || value.len() > MAX_BRANCH_REF_BYTES
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
        return false;
    }
    let short_name = value.strip_prefix("refs/heads/").unwrap_or_default();
    !short_name.is_empty()
        && short_name.split('/').all(|component| {
            !component.is_empty()
                && !component.starts_with('.')
                && !component.ends_with('.')
                && !component.ends_with(".lock")
        })
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BranchCollaborationError {
    InvalidCommunityId,
    InvalidRepositoryId,
    InvalidBranchRef,
    InvalidCommitId,
    InvalidMergeTarget,
    InvalidRecord,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    StaleCommit {
        expected: GitCommitId,
        actual: GitCommitId,
    },
    InvalidTransition,
    GenerationExhausted,
    VersionExhausted,
}

impl fmt::Display for BranchCollaborationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommunityId => formatter.write_str("community identifier is invalid"),
            Self::InvalidRepositoryId => formatter.write_str("repository identifier is invalid"),
            Self::InvalidBranchRef => formatter.write_str("branch ref is invalid"),
            Self::InvalidCommitId => formatter.write_str("Git commit identifier is invalid"),
            Self::InvalidMergeTarget => formatter.write_str("branch merge target is invalid"),
            Self::InvalidRecord => formatter.write_str("branch collaboration record is invalid"),
            Self::StaleVersion { .. } => formatter.write_str("branch version is stale"),
            Self::StaleCommit { .. } => formatter.write_str("branch commit is stale"),
            Self::InvalidTransition => formatter.write_str("branch transition is invalid"),
            Self::GenerationExhausted => formatter.write_str("branch generation is exhausted"),
            Self::VersionExhausted => formatter.write_str("branch version is exhausted"),
        }
    }
}

impl Error for BranchCollaborationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn commit(value: u64) -> GitCommitId {
        GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
    }

    fn branch_ref(value: &str) -> BranchRefName {
        BranchRefName::parse(format!("refs/heads/{value}")).expect("valid branch")
    }

    fn identity() -> BranchCollaborationIdentity {
        BranchCollaborationIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            AggregateId::from_uuid(Uuid::from_u128(2)),
            branch_ref("feature/auth"),
            BranchGeneration::FIRST,
        )
        .expect("valid identity")
    }

    fn branch() -> BranchCollaboration {
        BranchCollaboration::create(identity(), commit(10)).expect("valid branch")
    }

    #[test]
    fn branch_recreation_uses_a_new_generation_and_preserves_the_archive() {
        let mut archived = branch();
        assert_eq!(
            archived.archive(
                AggregateVersion::FIRST,
                &commit(10),
                BranchArchiveReason::Deleted,
            ),
            Ok(BranchCommandOutcome::Applied)
        );

        let recreated = archived
            .recreate(archived.fields().version, &commit(10), commit(20))
            .expect("recreate branch");

        assert_eq!(
            archived.fields().lifecycle_state,
            BranchLifecycleState::Archived(BranchArchiveReason::Deleted)
        );
        assert_eq!(archived.fields().identity.generation().get(), 1);
        assert_eq!(recreated.fields().identity.generation().get(), 2);
        assert_eq!(
            recreated.fields().identity.branch_ref(),
            archived.fields().identity.branch_ref()
        );
        assert_ne!(recreated.fields().identity, archived.fields().identity);
        assert_eq!(recreated.fields().head_commit, commit(20));
        assert_eq!(recreated.fields().version, AggregateVersion::FIRST);
        assert_eq!(
            recreated.fields().lifecycle_state,
            BranchLifecycleState::Active
        );
    }

    #[test]
    fn force_update_records_the_exact_head_transition() {
        let mut branch = branch();

        assert_eq!(
            branch.update_head(
                AggregateVersion::FIRST,
                &commit(10),
                commit(30),
                BranchUpdateKind::Force,
            ),
            Ok(BranchCommandOutcome::Applied)
        );

        assert_eq!(branch.fields().head_commit, commit(30));
        assert_eq!(branch.fields().version.get(), 2);
        let update = branch
            .fields()
            .last_head_update
            .as_ref()
            .expect("head update");
        assert_eq!(update.previous_commit(), &commit(10));
        assert_eq!(update.current_commit(), &commit(30));
        assert_eq!(update.kind(), BranchUpdateKind::Force);
    }

    #[test]
    fn merge_then_archive_retains_source_target_and_result_commits() {
        let mut branch = branch();

        assert_eq!(
            branch.merge(
                AggregateVersion::FIRST,
                &commit(10),
                branch_ref("main"),
                commit(40),
            ),
            Ok(BranchCommandOutcome::Applied)
        );
        let merged_version = branch.fields().version;
        assert_eq!(
            branch.archive(merged_version, &commit(10), BranchArchiveReason::Merged,),
            Ok(BranchCommandOutcome::Applied)
        );

        assert_eq!(
            branch.fields().lifecycle_state,
            BranchLifecycleState::Archived(BranchArchiveReason::Merged)
        );
        let merge = branch.fields().merge.as_ref().expect("merge metadata");
        assert_eq!(merge.source_commit(), &commit(10));
        assert_eq!(merge.target_branch(), &branch_ref("main"));
        assert_eq!(merge.result_commit(), &commit(40));
        assert_eq!(branch.head_identity().commit(), &commit(10));
    }

    #[test]
    fn stale_commit_and_version_fail_without_mutating_state() {
        let mut branch = branch();
        let original = branch.clone();

        assert_eq!(
            branch.update_head(
                AggregateVersion::FIRST,
                &commit(9),
                commit(10),
                BranchUpdateKind::FastForward,
            ),
            Err(BranchCollaborationError::StaleCommit {
                expected: commit(9),
                actual: commit(10),
            })
        );
        assert_eq!(branch, original);
        assert_eq!(
            branch.update_head(
                AggregateVersion::FIRST,
                &commit(9),
                commit(20),
                BranchUpdateKind::FastForward,
            ),
            Err(BranchCollaborationError::StaleCommit {
                expected: commit(9),
                actual: commit(10),
            })
        );
        assert_eq!(branch, original);
        assert_eq!(
            branch.merge(
                AggregateVersion::new(2).expect("version"),
                &commit(10),
                branch_ref("main"),
                commit(20),
            ),
            Err(BranchCollaborationError::StaleVersion {
                expected: AggregateVersion::new(2).expect("version"),
                actual: AggregateVersion::FIRST,
            })
        );
        assert_eq!(branch, original);
    }

    #[test]
    fn branch_and_commit_identifiers_reject_unsafe_or_ambiguous_values() {
        for value in [
            "main",
            "refs/tags/v1",
            "refs/heads/../main",
            "refs/heads/.hidden",
            "refs/heads/main.lock",
            "refs/heads/main~1",
        ] {
            assert_eq!(
                BranchRefName::parse(value),
                Err(BranchCollaborationError::InvalidBranchRef)
            );
        }
        assert_eq!(
            GitCommitId::parse("A".repeat(40)),
            Err(BranchCollaborationError::InvalidCommitId)
        );
        assert_eq!(
            GitCommitId::parse("a".repeat(39)),
            Err(BranchCollaborationError::InvalidCommitId)
        );
        assert!(GitCommitId::parse("a".repeat(64)).is_ok());
    }
}
