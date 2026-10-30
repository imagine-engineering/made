use crate::{AssetIdentity, OutputCommitReceipt};
use serde::{Deserialize, Serialize};
use sim_types::{AttemptId, ProfileId, PromptId};
use std::collections::BTreeMap;
use thiserror::Error;
use uuid::Uuid;

pub const RECOVERY_JOURNAL_SCHEMA_VERSION: u16 = 2;
pub const MAX_RECOVERY_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryOutputReceipt {
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    proposal_id: Uuid,
    operation_id: Uuid,
    identity: AssetIdentity,
    sha256: String,
    byte_size: u64,
    collision_counter: u32,
}

impl RecoveryOutputReceipt {
    fn from_commit_receipt(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        receipt: &OutputCommitReceipt,
    ) -> Self {
        let operation = receipt.operation();
        Self {
            profile_id,
            prompt_id,
            attempt_id,
            proposal_id: receipt.proposal_id(),
            operation_id: operation.operation_id,
            identity: operation.identity.clone(),
            sha256: operation.sha256.clone(),
            byte_size: operation.byte_size,
            collision_counter: operation.collision_counter,
        }
    }

    pub const fn profile_id(&self) -> ProfileId {
        self.profile_id
    }

    pub const fn prompt_id(&self) -> PromptId {
        self.prompt_id
    }

    pub const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    pub const fn proposal_id(&self) -> Uuid {
        self.proposal_id
    }

    pub const fn operation_id(&self) -> Uuid {
        self.operation_id
    }

    pub fn identity(&self) -> &AssetIdentity {
        &self.identity
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }

    pub const fn collision_counter(&self) -> u32 {
        self.collision_counter
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryJournal {
    schema_version: u16,
    receipts: Vec<RecoveryOutputReceipt>,
}

impl Default for RecoveryJournal {
    fn default() -> Self {
        Self {
            schema_version: RECOVERY_JOURNAL_SCHEMA_VERSION,
            receipts: Vec::new(),
        }
    }
}

impl RecoveryJournal {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn receipts(&self) -> &[RecoveryOutputReceipt] {
        &self.receipts
    }

    pub fn receipts_for_attempt(
        &self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
    ) -> impl Iterator<Item = &RecoveryOutputReceipt> {
        self.receipts.iter().filter(move |receipt| {
            receipt.profile_id == profile_id
                && receipt.prompt_id == prompt_id
                && receipt.attempt_id == attempt_id
        })
    }

    pub fn record_output_receipt(
        &mut self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        receipt: &OutputCommitReceipt,
    ) -> Result<(), RecoveryError> {
        let operation = receipt.operation();
        let scope = operation
            .execution_scope
            .as_ref()
            .ok_or(RecoveryError::UnscopedOutputReceipt(receipt.proposal_id()))?;
        if scope.profile_id != profile_id
            || scope.prompt_id != prompt_id
            || scope.attempt_id != attempt_id
        {
            return Err(RecoveryError::OutputReceiptScopeMismatch {
                proposal_id: receipt.proposal_id(),
            });
        }
        let receipt =
            RecoveryOutputReceipt::from_commit_receipt(profile_id, prompt_id, attempt_id, receipt);
        if let Some(existing) = self.receipts.iter().find(|existing| {
            existing.proposal_id == receipt.proposal_id
                || existing.operation_id == receipt.operation_id
        }) {
            return if existing == &receipt {
                Ok(())
            } else {
                Err(RecoveryError::ConflictingOutputReceipt {
                    proposal_id: receipt.proposal_id,
                    operation_id: receipt.operation_id,
                })
            };
        }
        self.receipts.push(receipt);
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, RecoveryError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| RecoveryError::Serialization(error.to_string()))?;
        if encoded.len() > MAX_RECOVERY_JOURNAL_BYTES {
            return Err(RecoveryError::Oversized);
        }
        Ok(encoded)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, RecoveryError> {
        if bytes.len() > MAX_RECOVERY_JOURNAL_BYTES {
            return Err(RecoveryError::Oversized);
        }
        let journal: Self = serde_json::from_slice(bytes)
            .map_err(|error| RecoveryError::Serialization(error.to_string()))?;
        journal.validate()?;
        Ok(journal)
    }

    fn validate(&self) -> Result<(), RecoveryError> {
        if self.schema_version != RECOVERY_JOURNAL_SCHEMA_VERSION {
            return Err(RecoveryError::UnsupportedSchema(self.schema_version));
        }
        let mut proposals = BTreeMap::new();
        let mut operations = BTreeMap::new();
        for receipt in &self.receipts {
            if let Some(previous) = proposals.insert(receipt.proposal_id, receipt) {
                return if previous == receipt {
                    Err(RecoveryError::DuplicateOutputReceipt {
                        proposal_id: receipt.proposal_id,
                        operation_id: receipt.operation_id,
                    })
                } else {
                    Err(RecoveryError::ConflictingOutputReceipt {
                        proposal_id: receipt.proposal_id,
                        operation_id: receipt.operation_id,
                    })
                };
            }
            if let Some(previous) = operations.insert(receipt.operation_id, receipt) {
                return if previous == receipt {
                    Err(RecoveryError::DuplicateOutputReceipt {
                        proposal_id: receipt.proposal_id,
                        operation_id: receipt.operation_id,
                    })
                } else {
                    Err(RecoveryError::ConflictingOutputReceipt {
                        proposal_id: receipt.proposal_id,
                        operation_id: receipt.operation_id,
                    })
                };
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RecoveryError {
    #[error("output receipt for proposal {0} has no checked execution scope")]
    UnscopedOutputReceipt(Uuid),
    #[error("output receipt for proposal {proposal_id} belongs to another execution scope")]
    OutputReceiptScopeMismatch { proposal_id: Uuid },
    #[error("output receipt for proposal {proposal_id} and operation {operation_id} is duplicated")]
    DuplicateOutputReceipt {
        proposal_id: Uuid,
        operation_id: Uuid,
    },
    #[error(
        "output receipt reuses proposal {proposal_id} or operation {operation_id} with different immutable facts"
    )]
    ConflictingOutputReceipt {
        proposal_id: Uuid,
        operation_id: Uuid,
    },
    #[error("recovery journal exceeds its byte limit")]
    Oversized,
    #[error("recovery journal schema {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("recovery journal serialization failed: {0}")]
    Serialization(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssetNamespace, AssetRoots, AuthorizedCapabilities, OutputCommitter, OutputExecutionScope,
        OutputProposal, authorize_native_output_committer,
    };
    use sim_types::CancellationToken;
    use std::{error::Error, fs};

    struct ReceiptFixture {
        _directory: tempfile::TempDir,
        profile_id: ProfileId,
        capabilities: AuthorizedCapabilities,
        committer: OutputCommitter,
    }

    impl ReceiptFixture {
        fn new() -> Result<Self, Box<dyn Error>> {
            let directory = tempfile::tempdir()?;
            let profile_id = ProfileId(Uuid::from_u128(1));
            let mut typed_roots = Vec::new();
            for (namespace, name) in [
                (AssetNamespace::Input, "input"),
                (AssetNamespace::Output, "output"),
                (AssetNamespace::Temporary, "temporary"),
                (AssetNamespace::Model, "model"),
                (AssetNamespace::Plugin, "plugin"),
            ] {
                let path = directory.path().join(name);
                fs::create_dir(&path)?;
                typed_roots.push((namespace, path));
            }
            let roots = AssetRoots::new(profile_id.0.to_string(), typed_roots)?;
            let capabilities = authorize_native_output_committer(&roots.profile_id)?;
            let committer = OutputCommitter::open(roots)?;
            Ok(Self {
                _directory: directory,
                profile_id,
                capabilities,
                committer,
            })
        }

        fn commit(
            &mut self,
            prompt_id: PromptId,
            attempt_id: AttemptId,
            proposal_id: Uuid,
            bytes: &[u8],
        ) -> Result<OutputCommitReceipt, Box<dyn Error>> {
            let proposal = OutputProposal::new(
                proposal_id,
                AssetNamespace::Output,
                "recovery/receipt",
                "bin",
                0,
                0,
                0,
                bytes.to_vec(),
            )?;
            let receipts = self.committer.commit_scoped_proposal_batch(
                &OutputExecutionScope {
                    profile_id: self.profile_id,
                    prompt_id,
                    attempt_id,
                },
                &[proposal],
                chrono::Local::now().fixed_offset(),
                &self.capabilities,
                &CancellationToken::default(),
            )?;
            receipts
                .into_iter()
                .next()
                .ok_or_else(|| "output committer returned no receipt".into())
        }
    }

    #[test]
    fn records_only_checked_output_receipt_facts_and_replays_exactly() -> Result<(), Box<dyn Error>>
    {
        let mut fixture = ReceiptFixture::new()?;
        let prompt_id = PromptId(Uuid::from_u128(2));
        let attempt_id = AttemptId(Uuid::from_u128(3));
        let receipt = fixture.commit(prompt_id, attempt_id, Uuid::from_u128(4), b"first")?;
        let mut journal = RecoveryJournal::default();
        journal.record_output_receipt(fixture.profile_id, prompt_id, attempt_id, &receipt)?;
        journal.record_output_receipt(fixture.profile_id, prompt_id, attempt_id, &receipt)?;

        assert_eq!(journal.receipts().len(), 1);
        let recorded = journal
            .receipts_for_attempt(fixture.profile_id, prompt_id, attempt_id)
            .next()
            .ok_or("missing receipt fact")?;
        assert_eq!(recorded.profile_id(), fixture.profile_id);
        assert_eq!(recorded.prompt_id(), prompt_id);
        assert_eq!(recorded.attempt_id(), attempt_id);
        assert_eq!(recorded.proposal_id(), receipt.proposal_id());
        assert_eq!(recorded.operation_id(), receipt.operation().operation_id);
        assert_eq!(recorded.identity(), &receipt.operation().identity);
        assert_eq!(recorded.sha256(), receipt.operation().sha256.as_str());
        assert_eq!(recorded.byte_size(), receipt.operation().byte_size);
        assert_eq!(
            recorded.collision_counter(),
            receipt.operation().collision_counter
        );

        let decoded = RecoveryJournal::decode(&journal.encode()?)?;
        assert_eq!(decoded, journal);
        assert_eq!(decoded.schema_version(), RECOVERY_JOURNAL_SCHEMA_VERSION);
        Ok(())
    }

    #[test]
    fn rejects_cross_attempt_receipt_reuse() -> Result<(), Box<dyn Error>> {
        let mut fixture = ReceiptFixture::new()?;
        let prompt_id = PromptId(Uuid::from_u128(2));
        let first_attempt = AttemptId(Uuid::from_u128(3));
        let second_attempt = AttemptId(Uuid::from_u128(4));
        let proposal_id = Uuid::from_u128(5);
        let first = fixture.commit(prompt_id, first_attempt, proposal_id, b"first")?;
        let mut journal = RecoveryJournal::default();
        journal.record_output_receipt(fixture.profile_id, prompt_id, first_attempt, &first)?;

        assert!(matches!(
            journal.record_output_receipt(fixture.profile_id, prompt_id, second_attempt, &first,),
            Err(RecoveryError::OutputReceiptScopeMismatch { .. })
        ));
        assert_eq!(journal.receipts().len(), 1);
        Ok(())
    }

    #[test]
    fn decode_rejects_duplicate_and_conflicting_receipt_facts() -> Result<(), Box<dyn Error>> {
        let mut fixture = ReceiptFixture::new()?;
        let prompt_id = PromptId(Uuid::from_u128(2));
        let attempt_id = AttemptId(Uuid::from_u128(3));
        let receipt = fixture.commit(prompt_id, attempt_id, Uuid::from_u128(4), b"first")?;
        let mut journal = RecoveryJournal::default();
        journal.record_output_receipt(fixture.profile_id, prompt_id, attempt_id, &receipt)?;

        let mut duplicate = journal.clone();
        duplicate.receipts.push(duplicate.receipts[0].clone());
        assert!(matches!(
            RecoveryJournal::decode(&serde_json::to_vec(&duplicate)?),
            Err(RecoveryError::DuplicateOutputReceipt { .. })
        ));

        let mut conflicting = journal;
        let mut altered = conflicting.receipts[0].clone();
        altered.attempt_id = AttemptId(Uuid::from_u128(9));
        conflicting.receipts.push(altered);
        assert!(matches!(
            RecoveryJournal::decode(&serde_json::to_vec(&conflicting)?),
            Err(RecoveryError::ConflictingOutputReceipt { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_schema_and_oversized_documents() -> Result<(), Box<dyn Error>> {
        let unsupported = RecoveryJournal {
            schema_version: RECOVERY_JOURNAL_SCHEMA_VERSION + 1,
            receipts: Vec::new(),
        };
        assert_eq!(
            RecoveryJournal::decode(&serde_json::to_vec(&unsupported)?),
            Err(RecoveryError::UnsupportedSchema(
                RECOVERY_JOURNAL_SCHEMA_VERSION + 1
            ))
        );
        assert_eq!(
            RecoveryJournal::decode(&vec![b' '; MAX_RECOVERY_JOURNAL_BYTES + 1]),
            Err(RecoveryError::Oversized)
        );
        Ok(())
    }
}
