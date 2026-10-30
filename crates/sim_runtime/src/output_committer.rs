use crate::AuthorizedCapabilities;
use crate::assets::{
    AssetAction, AssetError, AssetIdentity, AssetNamespace, AssetRecord, AssetRoots, AssetService,
    check_cancelled, normalize_optional_relative_path, require_asset_authorization, sha256,
};
use chrono::{DateTime, Datelike, FixedOffset, Local, Timelike};
use serde::{Deserialize, Serialize};
use sim_tensor::CancellationToken;
use sim_types::{AttemptId, ProfileId, PromptId};
#[cfg(test)]
use std::fs;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use thiserror::Error;
use uuid::Uuid;

pub const OUTPUT_JOURNAL_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_OUTPUT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const DEFAULT_MAX_OUTPUT_OPERATIONS: usize = 4_096;
pub const DEFAULT_MAX_OUTPUT_JOURNAL_BYTES: usize = 4 * 1024 * 1024;
const MAX_FILENAME_PREFIX_BYTES: usize = 4_096;
const MAX_EXTENSION_BYTES: usize = 32;
const MAX_PROJECTION_METADATA_BYTES: usize = 16 * 1024;
const STAGING_SUBFOLDER: &str = ".zed-output-staging";
const JOURNAL_FILENAME: &str = ".zed-output-transactions.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputOperationState {
    Prepared,
    Committed,
    Cancelled,
    Interrupted,
    InterruptedConflict,
    CommittedMissing,
    CommittedCorrupt,
}

impl OutputOperationState {
    fn is_terminal(self) -> bool {
        self != Self::Prepared
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputNameRequest {
    pub namespace: AssetNamespace,
    pub filename_prefix: String,
    pub extension: String,
    pub batch_index: u32,
    pub width: u32,
    pub height: u32,
    pub timestamp: DateTime<FixedOffset>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputProposal {
    proposal_id: Uuid,
    namespace: AssetNamespace,
    filename_prefix: String,
    extension: String,
    batch_index: u32,
    width: u32,
    height: u32,
    content: Vec<u8>,
    projection_metadata: Vec<u8>,
}

impl OutputProposal {
    pub fn new(
        proposal_id: Uuid,
        namespace: AssetNamespace,
        filename_prefix: impl Into<String>,
        extension: impl Into<String>,
        batch_index: u32,
        width: u32,
        height: u32,
        content: Vec<u8>,
    ) -> Result<Self, OutputCommitError> {
        require_output_namespace(namespace)?;
        let filename_prefix = filename_prefix.into();
        normalize_output_prefix(&filename_prefix)?;
        let extension = normalize_extension(&extension.into())?;
        Ok(Self {
            proposal_id,
            namespace,
            filename_prefix,
            extension,
            batch_index,
            width,
            height,
            content,
            projection_metadata: Vec::new(),
        })
    }

    pub fn with_projection_metadata(
        mut self,
        projection_metadata: Vec<u8>,
    ) -> Result<Self, OutputCommitError> {
        if projection_metadata.len() > MAX_PROJECTION_METADATA_BYTES {
            return Err(OutputCommitError::ProjectionMetadataTooLarge {
                actual: projection_metadata.len(),
                limit: MAX_PROJECTION_METADATA_BYTES,
            });
        }
        self.projection_metadata = projection_metadata;
        Ok(self)
    }

    pub const fn proposal_id(&self) -> Uuid {
        self.proposal_id
    }

    pub const fn namespace(&self) -> AssetNamespace {
        self.namespace
    }

    pub fn filename_prefix(&self) -> &str {
        &self.filename_prefix
    }

    pub fn extension(&self) -> &str {
        &self.extension
    }

    pub const fn batch_index(&self) -> u32 {
        self.batch_index
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn content(&self) -> &[u8] {
        &self.content
    }

    pub fn projection_metadata(&self) -> &[u8] {
        &self.projection_metadata
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputCommitReceipt {
    proposal_id: Uuid,
    operation: OutputOperation,
}

impl OutputCommitReceipt {
    pub const fn proposal_id(&self) -> Uuid {
        self.proposal_id
    }

    pub const fn operation(&self) -> &OutputOperation {
        &self.operation
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputExecutionScope {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputOperation {
    pub operation_id: Uuid,
    #[serde(default)]
    pub proposal_id: Option<Uuid>,
    #[serde(default)]
    pub execution_scope: Option<OutputExecutionScope>,
    pub identity: AssetIdentity,
    pub staging_relative_path: PathBuf,
    pub sha256: String,
    pub byte_size: u64,
    pub collision_counter: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projection_metadata: Vec<u8>,
    pub state: OutputOperationState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct OutputBatchPublication {
    batch_id: Uuid,
    operation_ids: Vec<Uuid>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOutput {
    pub operation_id: Uuid,
    pub identity: AssetIdentity,
    pub sha256: String,
    pub byte_size: u64,
    pub collision_counter: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOutputRemoval {
    pub operation_id: Uuid,
    pub identity: AssetIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputRemovalOperation {
    pub operation_id: Uuid,
    pub identity: AssetIdentity,
    pub staging_relative_path: PathBuf,
    pub sha256: String,
    pub byte_size: u64,
    pub state: OutputOperationState,
}

impl From<&OutputOperation> for PreparedOutput {
    fn from(operation: &OutputOperation) -> Self {
        Self {
            operation_id: operation.operation_id,
            identity: operation.identity.clone(),
            sha256: operation.sha256.clone(),
            byte_size: operation.byte_size,
            collision_counter: operation.collision_counter,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct OutputJournal {
    schema_version: u32,
    profile_id: String,
    operations: Vec<OutputOperation>,
    #[serde(default)]
    publications: Vec<OutputBatchPublication>,
    #[serde(default)]
    removals: Vec<OutputRemovalOperation>,
}

#[derive(Clone, Debug)]
pub struct OutputCommitter {
    roots: AssetRoots,
    journal_identity: AssetIdentity,
    #[cfg(test)]
    staging_directory: PathBuf,
    journal: OutputJournal,
    max_output_bytes: u64,
    max_operations: usize,
    max_journal_bytes: usize,
}

pub type SharedOutputCommitter = Arc<Mutex<OutputCommitter>>;

#[derive(Default)]
pub struct ProjectOutputCommitterPool {
    committers: Vec<(AssetRoots, SharedOutputCommitter)>,
}

impl ProjectOutputCommitterPool {
    pub fn for_roots(
        &mut self,
        roots: AssetRoots,
    ) -> Result<SharedOutputCommitter, OutputCommitError> {
        if let Some((_, committer)) = self
            .committers
            .iter()
            .find(|(existing, _)| existing == &roots)
        {
            return Ok(committer.clone());
        }
        let committer = Arc::new(Mutex::new(OutputCommitter::open(roots.clone())?));
        self.committers.push((roots, committer.clone()));
        Ok(committer)
    }
}

impl OutputCommitter {
    pub fn open(roots: AssetRoots) -> Result<Self, OutputCommitError> {
        Self::open_with_limits(
            roots,
            DEFAULT_MAX_OUTPUT_BYTES,
            DEFAULT_MAX_OUTPUT_OPERATIONS,
            DEFAULT_MAX_OUTPUT_JOURNAL_BYTES,
        )
    }

    pub fn open_with_limits(
        roots: AssetRoots,
        max_output_bytes: u64,
        max_operations: usize,
        max_journal_bytes: usize,
    ) -> Result<Self, OutputCommitError> {
        if max_output_bytes == 0 || max_operations == 0 || max_journal_bytes == 0 {
            return Err(OutputCommitError::InvalidLimit);
        }
        #[cfg(test)]
        let staging_directory = roots
            .resolve_for_create(&roots.identity(
                AssetNamespace::Temporary,
                Path::new(STAGING_SUBFOLDER).join(".directory"),
            )?)?
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                OutputCommitError::UnsafeJournal("staging directory has no parent".to_owned())
            })?;
        let journal_identity = roots.identity(AssetNamespace::Temporary, JOURNAL_FILENAME)?;
        let journal =
            if let Some(bytes) = roots.read_private(&journal_identity, max_journal_bytes)? {
                decode_journal(&bytes, max_journal_bytes)?
            } else {
                OutputJournal {
                    schema_version: OUTPUT_JOURNAL_SCHEMA_VERSION,
                    profile_id: roots.profile_id.clone(),
                    operations: Vec::new(),
                    publications: Vec::new(),
                    removals: Vec::new(),
                }
            };
        validate_journal(&journal, &roots, max_operations)?;
        let mut committer = Self {
            roots,
            journal_identity,
            #[cfg(test)]
            staging_directory,
            journal,
            max_output_bytes,
            max_operations,
            max_journal_bytes,
        };
        if committer.recover()? {
            committer.persist_journal()?;
        }
        Ok(committer)
    }

    pub fn roots(&self) -> &AssetRoots {
        &self.roots
    }

    pub fn operations(&self) -> &[OutputOperation] {
        &self.journal.operations
    }

    pub fn operation(&self, operation_id: Uuid) -> Option<&OutputOperation> {
        self.journal
            .operations
            .iter()
            .find(|operation| operation.operation_id == operation_id)
    }

    pub fn removal(&self, operation_id: Uuid) -> Option<&OutputRemovalOperation> {
        self.journal
            .removals
            .iter()
            .find(|operation| operation.operation_id == operation_id)
    }

    pub fn pending_removals(&self) -> Vec<OutputRemovalOperation> {
        self.journal
            .removals
            .iter()
            .filter(|operation| operation.state == OutputOperationState::Prepared)
            .cloned()
            .collect()
    }

    pub fn prepare(
        &mut self,
        request: &OutputNameRequest,
        bytes: &[u8],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<PreparedOutput, OutputCommitError> {
        self.prepare_with_origin(
            request,
            bytes,
            None,
            None,
            Vec::new(),
            capabilities,
            cancellation,
        )
    }

    fn prepare_with_origin(
        &mut self,
        request: &OutputNameRequest,
        bytes: &[u8],
        proposal_id: Option<Uuid>,
        execution_scope: Option<OutputExecutionScope>,
        projection_metadata: Vec<u8>,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<PreparedOutput, OutputCommitError> {
        if let (Some(proposal_id), Some(execution_scope)) = (proposal_id, execution_scope.as_ref())
            && self.journal.operations.iter().any(|operation| {
                operation.proposal_id == Some(proposal_id)
                    && operation.execution_scope.as_ref() == Some(execution_scope)
            })
        {
            return Err(OutputCommitError::DuplicateProposal(proposal_id));
        }
        require_output_namespace(request.namespace)?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            request.namespace,
            AssetAction::Write,
        )?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            AssetNamespace::Temporary,
            AssetAction::Write,
        )?;
        check_cancelled(cancellation)?;
        let byte_size = u64::try_from(bytes.len()).map_err(|_| OutputCommitError::TooLarge {
            actual: u64::MAX,
            limit: self.max_output_bytes,
        })?;
        if byte_size > self.max_output_bytes {
            return Err(OutputCommitError::TooLarge {
                actual: byte_size,
                limit: self.max_output_bytes,
            });
        }
        if projection_metadata.len() > MAX_PROJECTION_METADATA_BYTES {
            return Err(OutputCommitError::ProjectionMetadataTooLarge {
                actual: projection_metadata.len(),
                limit: MAX_PROJECTION_METADATA_BYTES,
            });
        }
        let extension = normalize_extension(&request.extension)?;
        let expanded_prefix = expand_filename_prefix(request)?;
        let prefix_path = normalize_output_prefix(&expanded_prefix)?;
        let filename_stem = prefix_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| OutputCommitError::InvalidFilenamePrefix(expanded_prefix.clone()))?;
        let subfolder = prefix_path.parent().unwrap_or_else(|| Path::new(""));
        let filesystem_counter = self
            .highest_existing_counter(request.namespace, subfolder, filename_stem, &extension)?
            .checked_add(1)
            .ok_or(OutputCommitError::CollisionCounterExhausted)?;
        let mut collision_counter = self.next_reserved_counter(
            request.namespace,
            subfolder,
            filename_stem,
            &extension,
            filesystem_counter,
        )?;
        let identity = loop {
            let output_filename = format!("{filename_stem}_{collision_counter:05}_.{extension}");
            let relative_path = if subfolder.as_os_str().is_empty() {
                PathBuf::from(&output_filename)
            } else {
                subfolder.join(&output_filename)
            };
            let identity = self.roots.identity(request.namespace, relative_path)?;
            if !self.roots.contained_exists(&identity)? {
                break identity;
            }
            collision_counter = collision_counter
                .checked_add(1)
                .ok_or(OutputCommitError::CollisionCounterExhausted)?;
            collision_counter = self.next_reserved_counter(
                request.namespace,
                subfolder,
                filename_stem,
                &extension,
                collision_counter,
            )?;
        };

        self.make_operation_capacity()?;
        let operation_id = Uuid::new_v4();
        let staging_relative_path =
            PathBuf::from(STAGING_SUBFOLDER).join(format!("{operation_id}.part"));
        let staging_identity = self
            .roots
            .identity(AssetNamespace::Temporary, staging_relative_path.clone())?;
        self.roots.write_contained(
            &staging_identity,
            bytes,
            crate::assets::AssetCollisionPolicy::Reject,
        )?;
        if let Err(error) = check_cancelled(cancellation) {
            self.roots.remove_contained(&staging_identity)?;
            return Err(error.into());
        }
        let operation = OutputOperation {
            operation_id,
            proposal_id,
            execution_scope,
            identity,
            staging_relative_path,
            sha256: sha256(bytes),
            byte_size,
            collision_counter,
            projection_metadata,
            state: OutputOperationState::Prepared,
        };
        self.journal.operations.push(operation.clone());
        if let Err(error) = self.persist_journal() {
            self.journal.operations.pop();
            self.roots.remove_contained(&staging_identity)?;
            return Err(error);
        }
        Ok(PreparedOutput::from(&operation))
    }

    pub fn commit(
        &mut self,
        operation_id: Uuid,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<OutputOperation, OutputCommitError> {
        let operation = self
            .operation(operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?;
        if operation.state != OutputOperationState::Prepared {
            return Err(OutputCommitError::InvalidState {
                operation_id,
                state: operation.state,
            });
        }
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            operation.identity.namespace,
            AssetAction::Write,
        )?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            AssetNamespace::Temporary,
            AssetAction::Write,
        )?;
        check_cancelled(cancellation)?;
        let staging_identity = self.staging_identity(&operation)?;
        verify_regular_file(
            &self.roots,
            &staging_identity,
            &operation.sha256,
            operation.byte_size,
            cancellation,
        )?;
        if self.roots.contained_exists(&operation.identity)? {
            return Err(OutputCommitError::DestinationConflict(operation.identity));
        }
        check_cancelled(cancellation)?;
        self.roots.move_contained(
            &staging_identity,
            &operation.identity,
            crate::assets::AssetCollisionPolicy::Reject,
            &operation.sha256,
            operation.byte_size,
            cancellation,
        )?;
        let operation = self
            .operation_mut(operation_id)
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?;
        operation.state = OutputOperationState::Committed;
        let committed = operation.clone();
        self.persist_journal()?;
        Ok(committed)
    }

    pub fn commit_batch(
        &mut self,
        operation_ids: &[Uuid],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputOperation>, OutputCommitError> {
        self.commit_batch_with_hook(operation_ids, capabilities, cancellation, |_| Ok(()))
    }

    pub fn commit_proposal_batch(
        &mut self,
        proposals: &[OutputProposal],
        timestamp: DateTime<FixedOffset>,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        let prepared =
            self.prepare_proposal_batch(proposals, None, timestamp, capabilities, cancellation)?;
        let operation_ids = prepared
            .iter()
            .map(|output| output.operation_id)
            .collect::<Vec<_>>();
        let operations = self.commit_batch(&operation_ids, capabilities, cancellation)?;
        proposal_receipts(proposals, operations)
    }

    pub fn commit_scoped_proposal_batch(
        &mut self,
        execution_scope: &OutputExecutionScope,
        proposals: &[OutputProposal],
        timestamp: DateTime<FixedOffset>,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        let prepared = self.prepare_proposal_batch(
            proposals,
            Some(execution_scope),
            timestamp,
            capabilities,
            cancellation,
        )?;
        let operation_ids = prepared
            .iter()
            .map(|output| output.operation_id)
            .collect::<Vec<_>>();
        let operations = self.commit_batch(&operation_ids, capabilities, cancellation)?;
        proposal_receipts(proposals, operations)
    }

    fn prepare_proposal_batch(
        &mut self,
        proposals: &[OutputProposal],
        execution_scope: Option<&OutputExecutionScope>,
        timestamp: DateTime<FixedOffset>,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<PreparedOutput>, OutputCommitError> {
        let mut proposal_ids = BTreeSet::new();
        let mut prepared = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            if !proposal_ids.insert(proposal.proposal_id) {
                return Err(OutputCommitError::DuplicateProposal(proposal.proposal_id));
            }
            let request = OutputNameRequest {
                namespace: proposal.namespace,
                filename_prefix: proposal.filename_prefix.clone(),
                extension: proposal.extension.clone(),
                batch_index: proposal.batch_index,
                width: proposal.width,
                height: proposal.height,
                timestamp,
            };
            match self.prepare_with_origin(
                &request,
                &proposal.content,
                Some(proposal.proposal_id),
                execution_scope.cloned(),
                proposal.projection_metadata.clone(),
                capabilities,
                cancellation,
            ) {
                Ok(output) => prepared.push(output),
                Err(primary) => {
                    return Err(self.rollback_prepared_batch(&prepared, capabilities, primary));
                }
            }
        }
        Ok(prepared)
    }

    fn rollback_prepared_batch(
        &mut self,
        prepared: &[PreparedOutput],
        capabilities: &AuthorizedCapabilities,
        primary: OutputCommitError,
    ) -> OutputCommitError {
        let mut rollback_failures = Vec::new();
        for output in prepared.iter().rev() {
            if let Err(error) = self.cancel(output.operation_id, capabilities) {
                rollback_failures.push(error.to_string());
            }
        }
        if rollback_failures.is_empty() {
            primary
        } else {
            OutputCommitError::BatchRollback {
                primary: primary.to_string(),
                rollback: rollback_failures.join("; "),
            }
        }
    }

    pub fn commit_proposal_batch_and_register(
        &mut self,
        proposals: &[OutputProposal],
        timestamp: DateTime<FixedOffset>,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        if assets.roots() != &self.roots {
            return Err(OutputCommitError::AssetRootMismatch);
        }
        let prepared =
            self.prepare_proposal_batch(proposals, None, timestamp, capabilities, cancellation)?;
        self.commit_prepared_proposal_batch_and_register(
            proposals,
            prepared,
            assets,
            capabilities,
            cancellation,
        )
    }

    pub fn commit_scoped_proposal_batch_and_register(
        &mut self,
        execution_scope: &OutputExecutionScope,
        proposals: &[OutputProposal],
        timestamp: DateTime<FixedOffset>,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        self.commit_scoped_proposal_batch_and_register_with_precommit(
            execution_scope,
            proposals,
            timestamp,
            assets,
            capabilities,
            cancellation,
            |_| Ok(()),
        )
    }

    pub(crate) fn commit_scoped_proposal_batch_and_register_with_precommit(
        &mut self,
        execution_scope: &OutputExecutionScope,
        proposals: &[OutputProposal],
        timestamp: DateTime<FixedOffset>,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        validate_prepared: impl FnOnce(&[PreparedOutput]) -> Result<(), OutputCommitError>,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        if assets.roots() != &self.roots {
            return Err(OutputCommitError::AssetRootMismatch);
        }
        let prepared = self.prepare_proposal_batch(
            proposals,
            Some(execution_scope),
            timestamp,
            capabilities,
            cancellation,
        )?;
        if let Err(primary) = validate_prepared(&prepared) {
            return Err(self.rollback_prepared_batch(&prepared, capabilities, primary));
        }
        self.commit_prepared_proposal_batch_and_register(
            proposals,
            prepared,
            assets,
            capabilities,
            cancellation,
        )
    }

    fn commit_prepared_proposal_batch_and_register(
        &mut self,
        proposals: &[OutputProposal],
        prepared: Vec<PreparedOutput>,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        self.commit_proposal_batch_and_register_with_scope(
            proposals,
            prepared,
            assets,
            capabilities,
            cancellation,
        )
    }

    pub fn commit_scoped_proposal_batch_and_register_now(
        &mut self,
        execution_scope: &OutputExecutionScope,
        proposals: &[OutputProposal],
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        self.commit_scoped_proposal_batch_and_register(
            execution_scope,
            proposals,
            Local::now().fixed_offset(),
            assets,
            capabilities,
            cancellation,
        )
    }

    fn commit_proposal_batch_and_register_with_scope(
        &mut self,
        proposals: &[OutputProposal],
        prepared: Vec<PreparedOutput>,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        if assets.roots() != &self.roots {
            return Err(OutputCommitError::AssetRootMismatch);
        }
        let operation_ids = prepared
            .iter()
            .map(|output| output.operation_id)
            .collect::<Vec<_>>();
        let identities = prepared
            .iter()
            .map(|output| output.identity.clone())
            .collect::<Vec<_>>();
        let operations = self.commit_batch_with_hooks(
            &operation_ids,
            capabilities,
            cancellation,
            |_| Ok(()),
            |committer| {
                committer.persist_journal()?;
                assets.register_committed_outputs(identities, capabilities, cancellation)?;
                Ok(())
            },
        )?;
        proposal_receipts(proposals, operations)
    }

    pub fn committed_receipts_for_scope(
        &self,
        execution_scope: &OutputExecutionScope,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        self.journal
            .operations
            .iter()
            .filter(|operation| {
                operation.state == OutputOperationState::Committed
                    && operation.execution_scope.as_ref() == Some(execution_scope)
            })
            .map(|operation| {
                let proposal_id = operation.proposal_id.ok_or_else(|| {
                    OutputCommitError::UnsafeJournal(
                        "scoped committed output has no proposal identity".to_owned(),
                    )
                })?;
                Ok(OutputCommitReceipt {
                    proposal_id,
                    operation: operation.clone(),
                })
            })
            .collect::<Result<Vec<_>, OutputCommitError>>()
    }

    pub fn committed_execution_scopes(&self) -> Vec<OutputExecutionScope> {
        let mut scopes = Vec::new();
        for operation in &self.journal.operations {
            if operation.state == OutputOperationState::Committed
                && let Some(scope) = &operation.execution_scope
                && !scopes.contains(scope)
            {
                scopes.push(scope.clone());
            }
        }
        scopes
    }

    pub fn commit_proposal_batch_now(
        &mut self,
        proposals: &[OutputProposal],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
        self.commit_proposal_batch(
            proposals,
            Local::now().fixed_offset(),
            capabilities,
            cancellation,
        )
    }

    fn commit_batch_with_hook(
        &mut self,
        operation_ids: &[Uuid],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        mut after_publish: impl FnMut(usize) -> Result<(), OutputCommitError>,
    ) -> Result<Vec<OutputOperation>, OutputCommitError> {
        self.commit_batch_with_hooks(
            operation_ids,
            capabilities,
            cancellation,
            &mut after_publish,
            |committer| committer.persist_journal(),
        )
    }

    fn commit_batch_with_hooks(
        &mut self,
        operation_ids: &[Uuid],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        mut after_publish: impl FnMut(usize) -> Result<(), OutputCommitError>,
        persist_final_journal: impl FnOnce(&Self) -> Result<(), OutputCommitError>,
    ) -> Result<Vec<OutputOperation>, OutputCommitError> {
        if operation_ids.is_empty() {
            return Ok(Vec::new());
        }
        let publication =
            self.begin_batch_publication(operation_ids, capabilities, cancellation)?;
        let publish_result = (|| {
            for (index, operation_id) in publication.operation_ids.iter().enumerate() {
                check_cancelled(cancellation)?;
                self.publish_operation(*operation_id)?;
                after_publish(index.saturating_add(1))?;
                check_cancelled(cancellation)?;
            }
            Ok(())
        })();
        if let Err(error) = publish_result {
            return Err(self.rollback_failed_publication(&publication, error));
        }
        self.finish_batch_publication(&publication, persist_final_journal)
    }

    fn begin_batch_publication(
        &mut self,
        operation_ids: &[Uuid],
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<OutputBatchPublication, OutputCommitError> {
        let mut unique = BTreeSet::new();
        for operation_id in operation_ids {
            if !unique.insert(*operation_id) {
                return Err(OutputCommitError::DuplicateBatchOperation(*operation_id));
            }
            if self
                .journal
                .publications
                .iter()
                .any(|publication| publication.operation_ids.contains(operation_id))
            {
                return Err(OutputCommitError::OperationAlreadyPublishing(*operation_id));
            }
            let operation = self
                .operation(*operation_id)
                .cloned()
                .ok_or(OutputCommitError::UnknownOperation(*operation_id))?;
            if operation.state != OutputOperationState::Prepared {
                return Err(OutputCommitError::InvalidState {
                    operation_id: *operation_id,
                    state: operation.state,
                });
            }
            require_asset_authorization(
                capabilities,
                &self.roots.profile_id,
                operation.identity.namespace,
                AssetAction::Write,
            )?;
            require_asset_authorization(
                capabilities,
                &self.roots.profile_id,
                AssetNamespace::Temporary,
                AssetAction::Write,
            )?;
            check_cancelled(cancellation)?;
            let staging_identity = self.staging_identity(&operation)?;
            verify_regular_file(
                &self.roots,
                &staging_identity,
                &operation.sha256,
                operation.byte_size,
                cancellation,
            )?;
            if self.roots.contained_exists(&operation.identity)? {
                return Err(OutputCommitError::DestinationConflict(operation.identity));
            }
        }
        let publication = OutputBatchPublication {
            batch_id: Uuid::new_v4(),
            operation_ids: operation_ids.to_vec(),
        };
        self.journal.publications.push(publication.clone());
        if let Err(error) = self.persist_journal() {
            self.journal.publications.pop();
            return Err(error);
        }
        Ok(publication)
    }

    fn publish_operation(&self, operation_id: Uuid) -> Result<(), OutputCommitError> {
        let operation = self
            .operation(operation_id)
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?;
        let staging_identity = self.staging_identity(operation)?;
        self.roots.move_contained(
            &staging_identity,
            &operation.identity,
            crate::assets::AssetCollisionPolicy::Reject,
            &operation.sha256,
            operation.byte_size,
            &CancellationToken::default(),
        )?;
        Ok(())
    }

    fn finish_batch_publication(
        &mut self,
        publication: &OutputBatchPublication,
        persist_final_journal: impl FnOnce(&Self) -> Result<(), OutputCommitError>,
    ) -> Result<Vec<OutputOperation>, OutputCommitError> {
        let prepared_journal = self.journal.clone();
        let mut committed = Vec::with_capacity(publication.operation_ids.len());
        for operation_id in &publication.operation_ids {
            let operation = self
                .operation_mut(*operation_id)
                .ok_or(OutputCommitError::UnknownOperation(*operation_id))?;
            operation.state = OutputOperationState::Committed;
            committed.push(operation.clone());
        }
        self.remove_publication(publication.batch_id)?;
        if let Err(error) = persist_final_journal(self) {
            self.journal = prepared_journal;
            return Err(self.rollback_failed_publication(publication, error));
        }
        Ok(committed)
    }

    fn rollback_failed_publication(
        &mut self,
        publication: &OutputBatchPublication,
        primary_error: OutputCommitError,
    ) -> OutputCommitError {
        match self.restore_published_outputs(publication) {
            Ok(()) => {
                if let Err(error) = self.remove_publication(publication.batch_id) {
                    return OutputCommitError::BatchRollback {
                        primary: primary_error.to_string(),
                        rollback: error.to_string(),
                    };
                }
                if let Err(error) = self.persist_journal() {
                    return OutputCommitError::BatchRollback {
                        primary: primary_error.to_string(),
                        rollback: error.to_string(),
                    };
                }
                primary_error
            }
            Err(error) => OutputCommitError::BatchRollback {
                primary: primary_error.to_string(),
                rollback: error.to_string(),
            },
        }
    }

    fn restore_published_outputs(
        &self,
        publication: &OutputBatchPublication,
    ) -> Result<(), OutputCommitError> {
        for operation_id in publication.operation_ids.iter().rev() {
            let operation = self
                .operation(*operation_id)
                .ok_or(OutputCommitError::UnknownOperation(*operation_id))?;
            let staging_identity = self.staging_identity(operation)?;
            match regular_file_integrity(
                &self.roots,
                &operation.identity,
                &operation.sha256,
                operation.byte_size,
            )? {
                FileIntegrity::Matches => {
                    if !matches!(
                        regular_file_integrity(
                            &self.roots,
                            &staging_identity,
                            &operation.sha256,
                            operation.byte_size,
                        )?,
                        FileIntegrity::Missing
                    ) {
                        return Err(OutputCommitError::UnsafeJournal(
                            "published output also retained a staging file".to_owned(),
                        ));
                    }
                    let staging_identity = self.staging_identity(operation)?;
                    self.roots.move_contained(
                        &operation.identity,
                        &staging_identity,
                        crate::assets::AssetCollisionPolicy::Reject,
                        &operation.sha256,
                        operation.byte_size,
                        &CancellationToken::default(),
                    )?;
                }
                FileIntegrity::Missing => {}
                FileIntegrity::Different => {
                    return Err(OutputCommitError::IntegrityMismatch {
                        path: self.destination_path(&operation.identity)?,
                    });
                }
            }
        }
        Ok(())
    }

    fn remove_publication(&mut self, batch_id: Uuid) -> Result<(), OutputCommitError> {
        let index = self
            .journal
            .publications
            .iter()
            .position(|publication| publication.batch_id == batch_id)
            .ok_or(OutputCommitError::UnknownBatch(batch_id))?;
        self.journal.publications.remove(index);
        Ok(())
    }

    pub fn commit_and_register(
        &mut self,
        operation_id: Uuid,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<AssetRecord, OutputCommitError> {
        if assets.roots() != &self.roots {
            return Err(OutputCommitError::AssetRootMismatch);
        }
        let identity = self
            .operation(operation_id)
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?
            .identity
            .clone();
        let operations = self.commit_batch_with_hooks(
            &[operation_id],
            capabilities,
            cancellation,
            |_| Ok(()),
            |committer| {
                committer.persist_journal()?;
                assets.register_committed_output(identity.clone(), capabilities, cancellation)?;
                Ok(())
            },
        )?;
        let operation = operations.into_iter().next().ok_or_else(|| {
            OutputCommitError::UnsafeJournal(
                "single-output publication returned no committed operation".to_owned(),
            )
        })?;
        assets.record(&operation.identity).ok_or_else(|| {
            OutputCommitError::UnsafeJournal(
                "committed output is absent from the canonical asset service".to_owned(),
            )
        })
    }

    pub fn cancel(
        &mut self,
        operation_id: Uuid,
        capabilities: &AuthorizedCapabilities,
    ) -> Result<OutputOperation, OutputCommitError> {
        let operation = self
            .operation(operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?;
        if operation.state != OutputOperationState::Prepared {
            return Err(OutputCommitError::InvalidState {
                operation_id,
                state: operation.state,
            });
        }
        if self
            .journal
            .publications
            .iter()
            .any(|publication| publication.operation_ids.contains(&operation_id))
        {
            return Err(OutputCommitError::OperationAlreadyPublishing(operation_id));
        }
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            operation.identity.namespace,
            AssetAction::Write,
        )?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            AssetNamespace::Temporary,
            AssetAction::Write,
        )?;
        let staging_identity = self.staging_identity(&operation)?;
        self.roots.remove_contained(&staging_identity)?;
        let operation = self
            .operation_mut(operation_id)
            .ok_or(OutputCommitError::UnknownOperation(operation_id))?;
        operation.state = OutputOperationState::Cancelled;
        let cancelled = operation.clone();
        self.persist_journal()?;
        Ok(cancelled)
    }

    pub fn prepare_removal(
        &mut self,
        identity: &AssetIdentity,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<PreparedOutputRemoval, OutputCommitError> {
        require_output_namespace(identity.namespace)?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Delete,
        )?;
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            AssetNamespace::Temporary,
            AssetAction::Write,
        )?;
        if crate::project_documents::project_asset_is_immutable(&self.roots, identity)? {
            return Err(OutputCommitError::Asset(AssetError::ReservedAsset(
                identity.clone(),
            )));
        }
        check_cancelled(cancellation)?;
        if self.journal.removals.iter().any(|operation| {
            operation.identity == *identity && operation.state == OutputOperationState::Prepared
        }) {
            return Err(OutputCommitError::RemovalAlreadyPrepared(identity.clone()));
        }
        let Some((sha256, byte_size)) = self.roots.contained_digest(identity, cancellation)? else {
            return Err(OutputCommitError::IntegrityMismatch {
                path: self.destination_path(identity)?,
            });
        };
        check_cancelled(cancellation)?;
        self.make_operation_capacity()?;
        let operation_id = Uuid::new_v4();
        let staging_relative_path =
            PathBuf::from(STAGING_SUBFOLDER).join(format!("{operation_id}.remove"));
        let operation = OutputRemovalOperation {
            operation_id,
            identity: identity.clone(),
            staging_relative_path,
            sha256,
            byte_size,
            state: OutputOperationState::Prepared,
        };
        self.journal.removals.push(operation.clone());
        if let Err(error) = self.persist_journal() {
            self.journal.removals.pop();
            return Err(error);
        }
        let staging_identity = self.removal_staging_identity(&operation)?;
        if let Err(error) = self.roots.move_contained(
            &operation.identity,
            &staging_identity,
            crate::assets::AssetCollisionPolicy::Reject,
            &operation.sha256,
            operation.byte_size,
            cancellation,
        ) {
            if let Some(current) = self.removal_mut(operation_id) {
                current.state = OutputOperationState::Cancelled;
            }
            let primary = OutputCommitError::Io {
                path: self.destination_path(&operation.identity)?,
                message: error.to_string(),
            };
            return match self.persist_journal() {
                Ok(()) => Err(primary),
                Err(rollback) => Err(OutputCommitError::BatchRollback {
                    primary: primary.to_string(),
                    rollback: rollback.to_string(),
                }),
            };
        }
        Ok(PreparedOutputRemoval {
            operation_id,
            identity: identity.clone(),
        })
    }

    pub fn commit_removal_and_register(
        &mut self,
        operation_id: Uuid,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<OutputRemovalOperation, OutputCommitError> {
        if assets.roots() != &self.roots {
            return Err(OutputCommitError::AssetRootMismatch);
        }
        let operation = self
            .removal(operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownRemoval(operation_id))?;
        if operation.state != OutputOperationState::Prepared {
            return Err(OutputCommitError::InvalidState {
                operation_id,
                state: operation.state,
            });
        }
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            operation.identity.namespace,
            AssetAction::Delete,
        )?;
        check_cancelled(cancellation)?;
        let staging_identity = self.removal_staging_identity(&operation)?;
        verify_regular_file(
            &self.roots,
            &staging_identity,
            &operation.sha256,
            operation.byte_size,
            cancellation,
        )?;
        if self.roots.contained_exists(&operation.identity)? {
            return Err(OutputCommitError::RemovalDestinationRestored(
                operation.identity,
            ));
        }
        assets.register_removed_output(&operation.identity, capabilities, cancellation)?;
        let current = self
            .removal_mut(operation_id)
            .ok_or(OutputCommitError::UnknownRemoval(operation_id))?;
        current.state = OutputOperationState::Committed;
        let committed = current.clone();
        self.persist_journal()?;
        Ok(committed)
    }

    pub fn rollback_removal(
        &mut self,
        operation_id: Uuid,
        capabilities: &AuthorizedCapabilities,
    ) -> Result<OutputRemovalOperation, OutputCommitError> {
        let operation = self
            .removal(operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownRemoval(operation_id))?;
        if operation.state != OutputOperationState::Prepared {
            return Err(OutputCommitError::InvalidState {
                operation_id,
                state: operation.state,
            });
        }
        require_asset_authorization(
            capabilities,
            &self.roots.profile_id,
            operation.identity.namespace,
            AssetAction::Delete,
        )?;
        let staging_identity = self.removal_staging_identity(&operation)?;
        match (
            regular_file_integrity(
                &self.roots,
                &staging_identity,
                &operation.sha256,
                operation.byte_size,
            )?,
            regular_file_integrity(
                &self.roots,
                &operation.identity,
                &operation.sha256,
                operation.byte_size,
            )?,
        ) {
            (FileIntegrity::Matches, FileIntegrity::Missing) => {
                let staging_identity = self.removal_staging_identity(&operation)?;
                self.roots.move_contained(
                    &staging_identity,
                    &operation.identity,
                    crate::assets::AssetCollisionPolicy::Reject,
                    &operation.sha256,
                    operation.byte_size,
                    &CancellationToken::default(),
                )?;
            }
            (FileIntegrity::Missing, FileIntegrity::Matches) => {}
            _ => {
                return Err(OutputCommitError::RemovalConflict(operation.identity));
            }
        }
        let current = self
            .removal_mut(operation_id)
            .ok_or(OutputCommitError::UnknownRemoval(operation_id))?;
        current.state = OutputOperationState::Cancelled;
        let cancelled = current.clone();
        self.persist_journal()?;
        Ok(cancelled)
    }

    pub fn cleanup_committed_removal(
        &mut self,
        operation_id: Uuid,
    ) -> Result<(), OutputCommitError> {
        let operation = self
            .removal(operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownRemoval(operation_id))?;
        if operation.state != OutputOperationState::Committed {
            return Err(OutputCommitError::InvalidState {
                operation_id,
                state: operation.state,
            });
        }
        let staging_identity = self.removal_staging_identity(&operation)?;
        self.roots.remove_contained(&staging_identity)?;
        Ok(())
    }

    fn operation_mut(&mut self, operation_id: Uuid) -> Option<&mut OutputOperation> {
        self.journal
            .operations
            .iter_mut()
            .find(|operation| operation.operation_id == operation_id)
    }

    fn removal_mut(&mut self, operation_id: Uuid) -> Option<&mut OutputRemovalOperation> {
        self.journal
            .removals
            .iter_mut()
            .find(|operation| operation.operation_id == operation_id)
    }

    fn next_reserved_counter(
        &self,
        namespace: AssetNamespace,
        subfolder: &Path,
        filename_stem: &str,
        extension: &str,
        filesystem_counter: u32,
    ) -> Result<u32, OutputCommitError> {
        let mut maximum = filesystem_counter.saturating_sub(1);
        for operation in &self.journal.operations {
            if operation.identity.namespace != namespace
                || operation.identity.relative_path.parent() != Some(subfolder)
            {
                continue;
            }
            let Some(filename) = operation
                .identity
                .relative_path
                .file_name()
                .and_then(|filename| filename.to_str())
            else {
                continue;
            };
            if output_counter(filename, filename_stem, extension).is_some() {
                maximum = maximum.max(operation.collision_counter);
            }
        }
        maximum
            .checked_add(1)
            .ok_or(OutputCommitError::CollisionCounterExhausted)
    }

    fn highest_existing_counter(
        &self,
        namespace: AssetNamespace,
        relative_directory: &Path,
        filename_stem: &str,
        extension: &str,
    ) -> Result<u32, OutputCommitError> {
        let mut maximum = 0;
        for identity in self
            .roots
            .list_direct_contained_regular_files(namespace, relative_directory)?
        {
            let Some(filename) = identity.filename() else {
                continue;
            };
            if let Some(counter) = output_counter(filename, filename_stem, extension) {
                maximum = maximum.max(counter);
            }
        }
        Ok(maximum)
    }

    fn destination_path(&self, identity: &AssetIdentity) -> Result<PathBuf, OutputCommitError> {
        require_output_namespace(identity.namespace)?;
        let normalized = self
            .roots
            .identity(identity.namespace, identity.relative_path.clone())?;
        if &normalized != identity {
            return Err(OutputCommitError::UnsafeJournal(
                "output identity changed during validation".to_owned(),
            ));
        }
        self.roots.resolve_for_create(identity).map_err(Into::into)
    }

    #[cfg(test)]
    fn staging_path(&self, operation: &OutputOperation) -> Result<PathBuf, OutputCommitError> {
        let identity = self.staging_identity(operation)?;
        let path = self.roots.resolve_for_create(&identity)?;
        if path.parent() != Some(self.staging_directory.as_path()) {
            return Err(OutputCommitError::UnsafeJournal(
                "staging path escaped the transaction directory".to_owned(),
            ));
        }
        Ok(path)
    }

    fn staging_identity(
        &self,
        operation: &OutputOperation,
    ) -> Result<AssetIdentity, OutputCommitError> {
        self.roots
            .identity(
                AssetNamespace::Temporary,
                operation.staging_relative_path.clone(),
            )
            .map_err(Into::into)
    }

    #[cfg(test)]
    fn removal_staging_path(
        &self,
        operation: &OutputRemovalOperation,
    ) -> Result<PathBuf, OutputCommitError> {
        let identity = self.removal_staging_identity(operation)?;
        let path = self.roots.resolve_for_create(&identity)?;
        if path.parent() != Some(self.staging_directory.as_path()) {
            return Err(OutputCommitError::UnsafeJournal(
                "removal staging path escaped the transaction directory".to_owned(),
            ));
        }
        Ok(path)
    }

    fn removal_staging_identity(
        &self,
        operation: &OutputRemovalOperation,
    ) -> Result<AssetIdentity, OutputCommitError> {
        self.roots
            .identity(
                AssetNamespace::Temporary,
                operation.staging_relative_path.clone(),
            )
            .map_err(Into::into)
    }

    fn make_operation_capacity(&mut self) -> Result<(), OutputCommitError> {
        while self
            .journal
            .operations
            .len()
            .saturating_add(self.journal.removals.len())
            >= self.max_operations
        {
            if let Some(index) = self
                .journal
                .operations
                .iter()
                .position(|operation| operation.state.is_terminal())
            {
                self.journal.operations.remove(index);
            } else if let Some(index) = self
                .journal
                .removals
                .iter()
                .position(|operation| operation.state.is_terminal())
            {
                self.journal.removals.remove(index);
            } else {
                return Err(OutputCommitError::OperationLimit(self.max_operations));
            }
        }
        Ok(())
    }

    fn recover(&mut self) -> Result<bool, OutputCommitError> {
        let mut changed = false;
        for publication in self.journal.publications.clone() {
            for operation_id in &publication.operation_ids {
                let operation = self
                    .operation(*operation_id)
                    .cloned()
                    .ok_or(OutputCommitError::UnknownOperation(*operation_id))?;
                let staging_identity = self.staging_identity(&operation)?;
                let staging_status = regular_file_integrity(
                    &self.roots,
                    &staging_identity,
                    &operation.sha256,
                    operation.byte_size,
                )?;
                if staging_status == FileIntegrity::Different {
                    return Err(OutputCommitError::IntegrityMismatch {
                        path: self.roots.resolve_for_create(&staging_identity)?,
                    });
                }
                let destination_status = regular_file_integrity(
                    &self.roots,
                    &operation.identity,
                    &operation.sha256,
                    operation.byte_size,
                )?;
                let recovered_state = match destination_status {
                    FileIntegrity::Matches => {
                        self.roots.remove_contained(&operation.identity)?;
                        OutputOperationState::Interrupted
                    }
                    FileIntegrity::Missing => OutputOperationState::Interrupted,
                    FileIntegrity::Different => OutputOperationState::InterruptedConflict,
                };
                if staging_status == FileIntegrity::Matches {
                    self.roots.remove_contained(&staging_identity)?;
                }
                let current = self
                    .operation_mut(*operation_id)
                    .ok_or(OutputCommitError::UnknownOperation(*operation_id))?;
                if current.state != recovered_state {
                    current.state = recovered_state;
                }
                changed = true;
            }
            self.remove_publication(publication.batch_id)?;
        }
        for index in 0..self.journal.operations.len() {
            let operation =
                self.journal.operations.get(index).cloned().ok_or_else(|| {
                    OutputCommitError::UnsafeJournal("operation vanished".to_owned())
                })?;
            let staging_identity = self.staging_identity(&operation)?;
            let staging_status = regular_file_integrity(
                &self.roots,
                &staging_identity,
                &operation.sha256,
                operation.byte_size,
            )?;
            let destination_status = regular_file_integrity(
                &self.roots,
                &operation.identity,
                &operation.sha256,
                operation.byte_size,
            )?;
            let recovered_state = match operation.state {
                OutputOperationState::Prepared => match (destination_status, staging_status) {
                    (FileIntegrity::Matches, FileIntegrity::Different) => {
                        return Err(OutputCommitError::IntegrityMismatch {
                            path: self.roots.resolve_for_create(&staging_identity)?,
                        });
                    }
                    (FileIntegrity::Matches, _) => OutputOperationState::Committed,
                    (FileIntegrity::Missing, _) => OutputOperationState::Interrupted,
                    (FileIntegrity::Different, _) => OutputOperationState::InterruptedConflict,
                },
                OutputOperationState::Committed
                | OutputOperationState::CommittedMissing
                | OutputOperationState::CommittedCorrupt => match destination_status {
                    FileIntegrity::Matches => OutputOperationState::Committed,
                    FileIntegrity::Missing => OutputOperationState::CommittedMissing,
                    FileIntegrity::Different => OutputOperationState::CommittedCorrupt,
                },
                OutputOperationState::Cancelled
                | OutputOperationState::Interrupted
                | OutputOperationState::InterruptedConflict => operation.state,
            };
            if recovered_state != OutputOperationState::Prepared {
                match staging_status {
                    FileIntegrity::Matches => self.roots.remove_contained(&staging_identity)?,
                    FileIntegrity::Missing => {}
                    FileIntegrity::Different => {
                        return Err(OutputCommitError::IntegrityMismatch {
                            path: self.roots.resolve_for_create(&staging_identity)?,
                        });
                    }
                }
            }
            if recovered_state != operation.state {
                let current = self.journal.operations.get_mut(index).ok_or_else(|| {
                    OutputCommitError::UnsafeJournal("operation vanished".to_owned())
                })?;
                current.state = recovered_state;
                changed = true;
            }
        }
        for index in 0..self.journal.removals.len() {
            let operation = self.journal.removals.get(index).cloned().ok_or_else(|| {
                OutputCommitError::UnsafeJournal("removal operation vanished".to_owned())
            })?;
            if operation.state == OutputOperationState::Prepared {
                continue;
            }
            let staging_identity = self.removal_staging_identity(&operation)?;
            match operation.state {
                OutputOperationState::Committed => {
                    let staging_identity = self.removal_staging_identity(&operation)?;
                    self.roots.remove_contained(&staging_identity)?;
                    changed = true;
                }
                OutputOperationState::Cancelled => match (
                    regular_file_integrity(
                        &self.roots,
                        &staging_identity,
                        &operation.sha256,
                        operation.byte_size,
                    )?,
                    regular_file_integrity(
                        &self.roots,
                        &operation.identity,
                        &operation.sha256,
                        operation.byte_size,
                    )?,
                ) {
                    (FileIntegrity::Matches, FileIntegrity::Missing) => {
                        let staging_identity = self.removal_staging_identity(&operation)?;
                        self.roots.move_contained(
                            &staging_identity,
                            &operation.identity,
                            crate::assets::AssetCollisionPolicy::Reject,
                            &operation.sha256,
                            operation.byte_size,
                            &CancellationToken::default(),
                        )?;
                        changed = true;
                    }
                    (FileIntegrity::Missing, FileIntegrity::Matches) => {}
                    _ => {
                        return Err(OutputCommitError::RemovalConflict(operation.identity));
                    }
                },
                _ => {
                    return Err(OutputCommitError::UnsafeJournal(
                        "removal operation has an unsupported state".to_owned(),
                    ));
                }
            }
        }
        let referenced = self
            .journal
            .operations
            .iter()
            .filter(|operation| operation.state == OutputOperationState::Prepared)
            .map(|operation| operation.staging_relative_path.clone())
            .chain(
                self.journal
                    .removals
                    .iter()
                    .filter(|operation| operation.state == OutputOperationState::Prepared)
                    .map(|operation| operation.staging_relative_path.clone()),
            )
            .collect::<BTreeSet<_>>();
        for identity in self
            .roots
            .list_direct_contained_files(AssetNamespace::Temporary, Path::new(STAGING_SUBFOLDER))?
        {
            if referenced.contains(&identity.relative_path) {
                continue;
            }
            self.roots.remove_contained(&identity)?;
            changed = true;
        }
        Ok(changed)
    }

    fn persist_journal(&self) -> Result<(), OutputCommitError> {
        let bytes = serde_json::to_vec_pretty(&self.journal)
            .map_err(|error| OutputCommitError::JournalEncode(error.to_string()))?;
        if bytes.len() > self.max_journal_bytes {
            return Err(OutputCommitError::JournalTooLarge {
                actual: bytes.len(),
                limit: self.max_journal_bytes,
            });
        }
        self.roots.write_private(&self.journal_identity, &bytes)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OutputCommitError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error("output namespace {0:?} cannot receive node outputs")]
    UnsupportedNamespace(AssetNamespace),
    #[error("output filename prefix {0:?} is invalid")]
    InvalidFilenamePrefix(String),
    #[error("output extension {0:?} is invalid")]
    InvalidExtension(String),
    #[error("output contains {actual} bytes, exceeding the {limit}-byte limit")]
    TooLarge { actual: u64, limit: u64 },
    #[error("output projection metadata contains {actual} bytes, exceeding the {limit}-byte limit")]
    ProjectionMetadataTooLarge { actual: usize, limit: usize },
    #[error("output collision counter is exhausted")]
    CollisionCounterExhausted,
    #[error("output destination already exists: {0:?}")]
    DestinationConflict(AssetIdentity),
    #[error("output operation {0} does not exist")]
    UnknownOperation(Uuid),
    #[error("output removal operation {0} does not exist")]
    UnknownRemoval(Uuid),
    #[error("output removal is already prepared for {0:?}")]
    RemovalAlreadyPrepared(AssetIdentity),
    #[error("output removal destination was restored before commit: {0:?}")]
    RemovalDestinationRestored(AssetIdentity),
    #[error("output removal cannot be reconciled safely: {0:?}")]
    RemovalConflict(AssetIdentity),
    #[error("output batch {0} does not exist")]
    UnknownBatch(Uuid),
    #[error("output operation {0} occurs more than once in a batch")]
    DuplicateBatchOperation(Uuid),
    #[error("output proposal {0} occurs more than once in a batch")]
    DuplicateProposal(Uuid),
    #[error("output operation {0} already belongs to an active batch publication")]
    OperationAlreadyPublishing(Uuid),
    #[error("output operation {operation_id} is in invalid state {state:?}")]
    InvalidState {
        operation_id: Uuid,
        state: OutputOperationState,
    },
    #[error("output integrity does not match for {path}")]
    IntegrityMismatch { path: PathBuf },
    #[error("output journal schema {actual} is unsupported; expected {expected}")]
    UnsupportedJournalSchema { expected: u32, actual: u32 },
    #[error("output journal is unsafe: {0}")]
    UnsafeJournal(String),
    #[error("output journal decode failed: {0}")]
    JournalDecode(String),
    #[error("output journal encode failed: {0}")]
    JournalEncode(String),
    #[error("output journal contains {actual} bytes, exceeding the {limit}-byte limit")]
    JournalTooLarge { actual: usize, limit: usize },
    #[error("output journal operation limit {0} is exhausted")]
    OperationLimit(usize),
    #[error("output and asset services do not share typed roots")]
    AssetRootMismatch,
    #[error("output transaction precommit validation failed: {0}")]
    PrecommitValidation(String),
    #[error("output commit I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("output batch failed: {primary}; rollback failed: {rollback}")]
    BatchRollback { primary: String, rollback: String },
    #[error("output transaction limits must be non-zero")]
    InvalidLimit,
}

fn require_output_namespace(namespace: AssetNamespace) -> Result<(), OutputCommitError> {
    if matches!(
        namespace,
        AssetNamespace::Output | AssetNamespace::Temporary
    ) {
        Ok(())
    } else {
        Err(OutputCommitError::UnsupportedNamespace(namespace))
    }
}

fn normalize_extension(extension: &str) -> Result<String, OutputCommitError> {
    let normalized = extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    if normalized.is_empty()
        || normalized.len() > MAX_EXTENSION_BYTES
        || !normalized
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    {
        return Err(OutputCommitError::InvalidExtension(extension.to_owned()));
    }
    Ok(normalized)
}

fn expand_filename_prefix(request: &OutputNameRequest) -> Result<String, OutputCommitError> {
    if request.filename_prefix.is_empty()
        || request.filename_prefix.len() > MAX_FILENAME_PREFIX_BYTES
        || request.filename_prefix.contains(['\\', '\0'])
    {
        return Err(OutputCommitError::InvalidFilenamePrefix(
            request.filename_prefix.clone(),
        ));
    }
    let timestamp = request.timestamp;
    Ok(request
        .filename_prefix
        .replace("%width%", &request.width.to_string())
        .replace("%height%", &request.height.to_string())
        .replace("%year%", &format!("{:04}", timestamp.year()))
        .replace("%month%", &format!("{:02}", timestamp.month()))
        .replace("%day%", &format!("{:02}", timestamp.day()))
        .replace("%hour%", &format!("{:02}", timestamp.hour()))
        .replace("%minute%", &format!("{:02}", timestamp.minute()))
        .replace("%second%", &format!("{:02}", timestamp.second()))
        .replace("%batch_num%", &request.batch_index.to_string()))
}

fn normalize_output_prefix(prefix: &str) -> Result<PathBuf, OutputCommitError> {
    let path = normalize_optional_relative_path(Path::new(prefix))?;
    let filename = path.file_name().and_then(|name| name.to_str());
    if filename.is_none_or(|filename| filename.is_empty() || filename == "." || filename == "..") {
        return Err(OutputCommitError::InvalidFilenamePrefix(prefix.to_owned()));
    }
    Ok(path)
}

fn output_counter(filename: &str, filename_stem: &str, extension: &str) -> Option<u32> {
    filename
        .strip_prefix(&format!("{filename_stem}_"))?
        .strip_suffix(&format!("_.{extension}"))?
        .parse()
        .ok()
}

fn decode_journal(bytes: &[u8], limit: usize) -> Result<OutputJournal, OutputCommitError> {
    let actual = bytes.len();
    if actual > limit {
        return Err(OutputCommitError::JournalTooLarge { actual, limit });
    }
    serde_json::from_slice(bytes)
        .map_err(|error| OutputCommitError::JournalDecode(error.to_string()))
}

fn proposal_receipts(
    proposals: &[OutputProposal],
    operations: Vec<OutputOperation>,
) -> Result<Vec<OutputCommitReceipt>, OutputCommitError> {
    if proposals.len() != operations.len() {
        return Err(OutputCommitError::UnsafeJournal(
            "committed output count does not match its proposal count".to_owned(),
        ));
    }
    proposals
        .iter()
        .zip(operations)
        .map(|(proposal, operation)| {
            if operation.proposal_id != Some(proposal.proposal_id) {
                return Err(OutputCommitError::UnsafeJournal(
                    "committed output operation does not match its proposal identity".to_owned(),
                ));
            }
            Ok(OutputCommitReceipt {
                proposal_id: proposal.proposal_id,
                operation,
            })
        })
        .collect()
}

fn validate_journal(
    journal: &OutputJournal,
    roots: &AssetRoots,
    max_operations: usize,
) -> Result<(), OutputCommitError> {
    if journal.schema_version != OUTPUT_JOURNAL_SCHEMA_VERSION {
        return Err(OutputCommitError::UnsupportedJournalSchema {
            expected: OUTPUT_JOURNAL_SCHEMA_VERSION,
            actual: journal.schema_version,
        });
    }
    if journal.profile_id != roots.profile_id {
        return Err(OutputCommitError::UnsafeJournal(
            "journal profile does not match the active roots".to_owned(),
        ));
    }
    if journal
        .operations
        .len()
        .saturating_add(journal.removals.len())
        > max_operations
    {
        return Err(OutputCommitError::OperationLimit(max_operations));
    }
    let mut identifiers = BTreeSet::new();
    let mut scoped_proposals = BTreeSet::new();
    for operation in &journal.operations {
        if !identifiers.insert(operation.operation_id) {
            return Err(OutputCommitError::UnsafeJournal(
                "journal contains duplicate operation identifiers".to_owned(),
            ));
        }
        require_output_namespace(operation.identity.namespace)?;
        if operation.identity.profile_id != roots.profile_id {
            return Err(OutputCommitError::UnsafeJournal(
                "operation profile does not match the journal".to_owned(),
            ));
        }
        if let Some(scope) = &operation.execution_scope {
            if scope.profile_id.0.to_string() != journal.profile_id {
                return Err(OutputCommitError::UnsafeJournal(
                    "output execution scope does not match the journal profile".to_owned(),
                ));
            }
            let proposal_id = operation.proposal_id.ok_or_else(|| {
                OutputCommitError::UnsafeJournal(
                    "scoped output operation has no proposal identity".to_owned(),
                )
            })?;
            if !scoped_proposals.insert((scope.attempt_id.0, proposal_id)) {
                return Err(OutputCommitError::UnsafeJournal(
                    "journal contains duplicate attempt-scoped proposal identities".to_owned(),
                ));
            }
        }
        if operation.projection_metadata.len() > MAX_PROJECTION_METADATA_BYTES {
            return Err(OutputCommitError::UnsafeJournal(
                "output projection metadata exceeds its bound".to_owned(),
            ));
        }
        roots.identity(
            operation.identity.namespace,
            operation.identity.relative_path.clone(),
        )?;
        let staging = roots.identity(
            AssetNamespace::Temporary,
            operation.staging_relative_path.clone(),
        )?;
        if staging.relative_path.parent() != Some(Path::new(STAGING_SUBFOLDER)) {
            return Err(OutputCommitError::UnsafeJournal(
                "operation staging path is outside the staging directory".to_owned(),
            ));
        }
        if operation.sha256.len() != 64
            || !operation
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(OutputCommitError::UnsafeJournal(
                "operation digest is malformed".to_owned(),
            ));
        }
    }
    for operation in &journal.removals {
        if !identifiers.insert(operation.operation_id) {
            return Err(OutputCommitError::UnsafeJournal(
                "journal contains a duplicate removal identifier".to_owned(),
            ));
        }
        require_output_namespace(operation.identity.namespace)?;
        if operation.identity.profile_id != roots.profile_id {
            return Err(OutputCommitError::UnsafeJournal(
                "removal profile does not match the journal".to_owned(),
            ));
        }
        roots.identity(
            operation.identity.namespace,
            operation.identity.relative_path.clone(),
        )?;
        let staging = roots.identity(
            AssetNamespace::Temporary,
            operation.staging_relative_path.clone(),
        )?;
        if staging.relative_path.parent() != Some(Path::new(STAGING_SUBFOLDER))
            || staging
                .relative_path
                .extension()
                .and_then(|value| value.to_str())
                != Some("remove")
        {
            return Err(OutputCommitError::UnsafeJournal(
                "removal staging path is outside the staging directory".to_owned(),
            ));
        }
        if operation.sha256.len() != 64
            || !operation
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(OutputCommitError::UnsafeJournal(
                "removal digest is malformed".to_owned(),
            ));
        }
        if !matches!(
            operation.state,
            OutputOperationState::Prepared
                | OutputOperationState::Committed
                | OutputOperationState::Cancelled
        ) {
            return Err(OutputCommitError::UnsafeJournal(
                "removal operation has an invalid state".to_owned(),
            ));
        }
    }
    let mut batch_identifiers = BTreeSet::new();
    let mut publishing_operations = BTreeSet::new();
    for publication in &journal.publications {
        if !batch_identifiers.insert(publication.batch_id) {
            return Err(OutputCommitError::UnsafeJournal(
                "journal contains duplicate batch identifiers".to_owned(),
            ));
        }
        if publication.operation_ids.is_empty() {
            return Err(OutputCommitError::UnsafeJournal(
                "journal contains an empty batch publication".to_owned(),
            ));
        }
        for operation_id in &publication.operation_ids {
            if !publishing_operations.insert(*operation_id) {
                return Err(OutputCommitError::UnsafeJournal(
                    "operation belongs to more than one batch publication".to_owned(),
                ));
            }
            let operation = journal
                .operations
                .iter()
                .find(|operation| operation.operation_id == *operation_id)
                .ok_or_else(|| {
                    OutputCommitError::UnsafeJournal(
                        "batch publication references an unknown operation".to_owned(),
                    )
                })?;
            if operation.state != OutputOperationState::Prepared {
                return Err(OutputCommitError::UnsafeJournal(
                    "batch publication references a terminal operation".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileIntegrity {
    Missing,
    Matches,
    Different,
}

fn regular_file_integrity(
    roots: &AssetRoots,
    identity: &AssetIdentity,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<FileIntegrity, OutputCommitError> {
    let Some((actual_sha256, actual_size)) =
        roots.contained_digest(identity, &CancellationToken::default())?
    else {
        return Ok(FileIntegrity::Missing);
    };
    Ok(
        if actual_size == expected_size && actual_sha256 == expected_sha256 {
            FileIntegrity::Matches
        } else {
            FileIntegrity::Different
        },
    )
}

fn verify_regular_file(
    roots: &AssetRoots,
    identity: &AssetIdentity,
    expected_sha256: &str,
    expected_size: u64,
    cancellation: &CancellationToken,
) -> Result<(), OutputCommitError> {
    let actual = roots.contained_digest(identity, cancellation)?;
    if !matches!(
        actual,
        Some((actual_sha256, actual_size))
            if actual_size == expected_size && actual_sha256 == expected_sha256
    ) {
        return Err(OutputCommitError::IntegrityMismatch {
            path: roots.resolve_for_create(identity)?,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{AssetAvailability, AssetByteRange, AssetViewRequest};
    use crate::{AssetOperation, Capability, CapabilitySet, PermissionGrant, PermissionPolicy};
    use serde_json::{Value, json};

    fn roots() -> Result<(tempfile::TempDir, AssetRoots, AuthorizedCapabilities), OutputCommitError>
    {
        let directory = tempfile::tempdir().map_err(|error| OutputCommitError::Io {
            path: PathBuf::from("temporary-directory"),
            message: error.to_string(),
        })?;
        let paths = [
            AssetNamespace::Input,
            AssetNamespace::Output,
            AssetNamespace::Temporary,
            AssetNamespace::Model,
            AssetNamespace::Plugin,
        ]
        .into_iter()
        .map(|namespace| {
            let path = directory.path().join(namespace.locator_type());
            fs::create_dir(&path).map_err(|error| OutputCommitError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            Ok((namespace, path))
        })
        .collect::<Result<Vec<_>, OutputCommitError>>()?;
        let roots = AssetRoots::new("profile", paths)?;
        let capabilities = CapabilitySet::new(roots.namespaces().flat_map(|namespace| {
            [
                AssetOperation::Read,
                AssetOperation::Write,
                AssetOperation::Rename,
                AssetOperation::Tag,
                AssetOperation::Delete,
            ]
            .into_iter()
            .map(move |action| Capability::Asset {
                namespace: namespace.locator_type().to_owned(),
                action,
            })
        }));
        let grant = PermissionGrant::new(
            "profile",
            "output-committer-test",
            capabilities.clone(),
            "output-committer-test-fixture",
        )
        .map_err(|error| OutputCommitError::Asset(AssetError::InvalidProfile(error.to_string())))?;
        let authorization = PermissionPolicy::new("profile", [grant])
            .and_then(|policy| policy.authorize("output-committer-test", &capabilities))
            .map_err(|error| {
                OutputCommitError::Asset(AssetError::InvalidProfile(error.to_string()))
            })?;
        Ok((directory, roots, authorization))
    }

    fn request(namespace: AssetNamespace) -> Result<OutputNameRequest, chrono::ParseError> {
        Ok(OutputNameRequest {
            namespace,
            filename_prefix: "renders/%year%-%month%/image_%width%x%height%_%batch_num%".to_owned(),
            extension: ".PNG".to_owned(),
            batch_index: 2,
            width: 512,
            height: 768,
            timestamp: DateTime::parse_from_rfc3339("2026-07-13T15:04:05+01:00")?,
        })
    }

    #[test]
    fn prepare_is_invisible_and_commit_registers_the_same_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut assets = AssetService::open(roots.clone())?;
        let cancellation = CancellationToken::default();
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"encoded-png",
            &capabilities,
            &cancellation,
        )?;
        let final_path = roots
            .test_root_path(AssetNamespace::Output)?
            .join(&prepared.identity.relative_path);
        assert!(!final_path.exists());
        assert_eq!(prepared.collision_counter, 1);
        assert_eq!(
            prepared.identity.relative_path,
            PathBuf::from("renders/2026-07/image_512x768_2_00001_.png")
        );
        let record = committer.commit_and_register(
            prepared.operation_id,
            &mut assets,
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(record.identity, prepared.identity);
        assert!(final_path.is_file());
        let view = assets.view(
            &AssetViewRequest {
                identity: record.identity,
                range: Some(AssetByteRange {
                    start: 0,
                    end_inclusive: 6,
                }),
                download: true,
            },
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(view.bytes, b"encoded");
        Ok(())
    }

    #[test]
    fn prepared_removal_is_recoverable_until_the_durable_projection_commits()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut assets = AssetService::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"transactional-removal",
            &capabilities,
            &cancellation,
        )?;
        committer.commit_and_register(
            prepared.operation_id,
            &mut assets,
            &capabilities,
            &cancellation,
        )?;
        let destination = roots.test_resolve_existing(&prepared.identity)?;
        let removal =
            committer.prepare_removal(&prepared.identity, &capabilities, &cancellation)?;
        assert!(!destination.exists());
        assert_eq!(committer.pending_removals().len(), 1);
        drop(committer);

        let mut recovered = OutputCommitter::open(roots)?;
        assert_eq!(recovered.pending_removals().len(), 1);
        recovered.rollback_removal(removal.operation_id, &capabilities)?;
        assert_eq!(fs::read(destination)?, b"transactional-removal");
        assert_eq!(
            recovered
                .removal(removal.operation_id)
                .map(|value| value.state),
            Some(OutputOperationState::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn committed_removal_updates_the_asset_index_before_private_cleanup()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut assets = AssetService::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"committed-removal",
            &capabilities,
            &cancellation,
        )?;
        committer.commit_and_register(
            prepared.operation_id,
            &mut assets,
            &capabilities,
            &cancellation,
        )?;
        let removal =
            committer.prepare_removal(&prepared.identity, &capabilities, &cancellation)?;
        committer.commit_removal_and_register(
            removal.operation_id,
            &mut assets,
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(
            assets
                .record(&prepared.identity)
                .map(|record| record.availability),
            Some(AssetAvailability::Missing)
        );
        let staged_path = committer.removal_staging_path(
            committer
                .removal(removal.operation_id)
                .ok_or(OutputCommitError::UnknownRemoval(removal.operation_id))?,
        )?;
        assert!(staged_path.is_file());
        committer.cleanup_committed_removal(removal.operation_id)?;
        assert!(!staged_path.exists());
        assert_eq!(
            OutputCommitter::open(roots)?
                .removal(removal.operation_id)
                .map(|value| value.state),
            Some(OutputOperationState::Committed)
        );
        Ok(())
    }

    #[test]
    fn asset_registration_failure_rolls_back_publication_and_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut assets = AssetService::open(roots.clone())?;
        assets.scan_namespaces(&[AssetNamespace::Output], &capabilities, &cancellation)?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"must-not-publish",
            &capabilities,
            &cancellation,
        )?;
        let state_path = roots
            .test_root_path(AssetNamespace::Temporary)?
            .join(".zed-asset-index.json");
        fs::remove_file(&state_path)?;
        fs::create_dir(&state_path)?;

        let error = committer
            .commit_and_register(
                prepared.operation_id,
                &mut assets,
                &capabilities,
                &cancellation,
            )
            .expect_err("asset-state failure must abort final publication");
        assert!(
            matches!(
                &error,
                OutputCommitError::Asset(AssetError::Rollback { .. })
            ),
            "unexpected commit error: {error:?}"
        );
        assert!(assets.record(&prepared.identity).is_none());
        assert!(
            !roots
                .test_root_path(AssetNamespace::Output)?
                .join(&prepared.identity.relative_path)
                .exists()
        );
        assert_eq!(
            committer
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::Prepared)
        );
        assert!(
            committer
                .staging_path(
                    committer
                        .operation(prepared.operation_id)
                        .ok_or(OutputCommitError::UnknownOperation(prepared.operation_id))?
                )?
                .is_file()
        );

        fs::remove_dir(&state_path)?;
        committer.cancel(prepared.operation_id, &capabilities)?;
        Ok(())
    }

    #[test]
    fn precommit_validation_failure_cancels_staging_before_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut assets = AssetService::open(roots.clone())?;
        let profile_id = ProfileId(Uuid::from_u128(1));
        let scope = OutputExecutionScope {
            profile_id,
            prompt_id: PromptId(Uuid::from_u128(2)),
            attempt_id: AttemptId(Uuid::from_u128(3)),
        };
        let proposal = OutputProposal::new(
            Uuid::from_u128(4),
            AssetNamespace::Output,
            "precommit/image",
            "png",
            0,
            1,
            1,
            b"must-not-publish".to_vec(),
        )?;
        let mut prepared_output = None;
        let error = committer
            .commit_scoped_proposal_batch_and_register_with_precommit(
                &scope,
                &[proposal],
                DateTime::parse_from_rfc3339("2026-07-13T15:04:05+01:00")?,
                &mut assets,
                &capabilities,
                &cancellation,
                |prepared| {
                    prepared_output = prepared.first().cloned();
                    Err(OutputCommitError::PrecommitValidation(
                        "injected canonical transition failure".to_owned(),
                    ))
                },
            )
            .expect_err("precommit rejection must abort the output transaction");
        assert!(matches!(error, OutputCommitError::PrecommitValidation(_)));
        let prepared = prepared_output.ok_or("precommit hook received no prepared output")?;
        assert!(assets.record(&prepared.identity).is_none());
        assert!(
            !roots
                .test_root_path(AssetNamespace::Output)?
                .join(&prepared.identity.relative_path)
                .exists()
        );
        let operation = committer
            .operation(prepared.operation_id)
            .ok_or(OutputCommitError::UnknownOperation(prepared.operation_id))?;
        assert_eq!(operation.state, OutputOperationState::Cancelled);
        assert!(!committer.staging_path(operation)?.exists());
        Ok(())
    }

    #[test]
    fn collision_counter_continues_after_the_highest_matching_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let directory = roots
            .test_root_path(AssetNamespace::Output)?
            .join("renders/2026-07");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("image_512x768_2_00002_.png"), b"old")?;
        fs::write(directory.join("image_512x768_2_00009_.png"), b"old")?;
        fs::write(directory.join("unrelated_99999_.png"), b"old")?;
        let mut committer = OutputCommitter::open(roots)?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"new",
            &capabilities,
            &CancellationToken::default(),
        )?;
        assert_eq!(prepared.collision_counter, 10);
        assert_eq!(
            prepared.identity.filename(),
            Some("image_512x768_2_00010_.png")
        );
        Ok(())
    }

    #[test]
    fn prepared_batch_outputs_reserve_distinct_collision_counters()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let mut committer = OutputCommitter::open(roots)?;
        let mut first_request = request(AssetNamespace::Output)?;
        first_request.filename_prefix = "batch/image".to_owned();
        first_request.batch_index = 0;
        let first = committer.prepare(
            &first_request,
            b"first",
            &capabilities,
            &CancellationToken::default(),
        )?;
        let mut second_request = first_request;
        second_request.batch_index = 1;
        let second = committer.prepare(
            &second_request,
            b"second",
            &capabilities,
            &CancellationToken::default(),
        )?;
        assert_eq!(first.collision_counter, 1);
        assert_eq!(second.collision_counter, 2);
        assert_ne!(first.identity, second.identity);
        Ok(())
    }

    #[test]
    fn batch_failure_after_first_publication_restores_every_prepared_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut first_request = request(AssetNamespace::Output)?;
        first_request.filename_prefix = "atomic/image".to_owned();
        first_request.batch_index = 0;
        let first = committer.prepare(&first_request, b"first", &capabilities, &cancellation)?;
        let mut second_request = first_request;
        second_request.batch_index = 1;
        let second = committer.prepare(&second_request, b"second", &capabilities, &cancellation)?;
        let error = committer
            .commit_batch_with_hook(
                &[first.operation_id, second.operation_id],
                &capabilities,
                &cancellation,
                |published| {
                    if published == 1 {
                        Err(OutputCommitError::Io {
                            path: PathBuf::from("injected-second-effect"),
                            message: "injected publication failure".to_owned(),
                        })
                    } else {
                        Ok(())
                    }
                },
            )
            .expect_err("the injected publication failure must abort the batch");
        assert!(matches!(error, OutputCommitError::Io { .. }));
        for prepared in [&first, &second] {
            assert!(
                !roots
                    .test_root_path(AssetNamespace::Output)?
                    .join(&prepared.identity.relative_path)
                    .exists()
            );
            assert_eq!(
                committer
                    .operation(prepared.operation_id)
                    .map(|operation| operation.state),
                Some(OutputOperationState::Prepared)
            );
        }
        assert!(committer.journal.publications.is_empty());
        committer.cancel(first.operation_id, &capabilities)?;
        committer.cancel(second.operation_id, &capabilities)?;
        assert!(fs::read_dir(&committer.staging_directory)?.next().is_none());
        Ok(())
    }

    #[test]
    fn batch_cancellation_after_first_publication_leaves_no_final_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut first_request = request(AssetNamespace::Output)?;
        first_request.filename_prefix = "cancelled-batch/image".to_owned();
        first_request.batch_index = 0;
        let first = committer.prepare(&first_request, b"first", &capabilities, &cancellation)?;
        let mut second_request = first_request;
        second_request.batch_index = 1;
        let second = committer.prepare(&second_request, b"second", &capabilities, &cancellation)?;
        let error = committer
            .commit_batch_with_hook(
                &[first.operation_id, second.operation_id],
                &capabilities,
                &cancellation,
                |published| {
                    if published == 1 {
                        cancellation.cancel();
                    }
                    Ok(())
                },
            )
            .expect_err("cancellation after the first publication must abort the batch");
        assert!(matches!(
            error,
            OutputCommitError::Asset(AssetError::Cancelled)
        ));
        for prepared in [&first, &second] {
            assert!(
                !roots
                    .test_root_path(AssetNamespace::Output)?
                    .join(&prepared.identity.relative_path)
                    .exists()
            );
            assert_eq!(
                committer
                    .operation(prepared.operation_id)
                    .map(|operation| operation.state),
                Some(OutputOperationState::Prepared)
            );
        }
        assert!(committer.journal.publications.is_empty());
        committer.cancel(first.operation_id, &capabilities)?;
        committer.cancel(second.operation_id, &capabilities)?;
        assert!(fs::read_dir(&committer.staging_directory)?.next().is_none());
        Ok(())
    }

    #[test]
    fn final_journal_failure_rolls_back_all_published_outputs_and_live_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut first_request = request(AssetNamespace::Output)?;
        first_request.filename_prefix = "failed-final-journal/image".to_owned();
        first_request.batch_index = 0;
        let first = committer.prepare(&first_request, b"first", &capabilities, &cancellation)?;
        let mut second_request = first_request;
        second_request.batch_index = 1;
        let second = committer.prepare(&second_request, b"second", &capabilities, &cancellation)?;
        let published_count = std::cell::Cell::new(0);
        let final_persist_attempts = std::cell::Cell::new(0);
        let error = committer
            .commit_batch_with_hooks(
                &[first.operation_id, second.operation_id],
                &capabilities,
                &cancellation,
                |published| {
                    published_count.set(published);
                    Ok(())
                },
                |committer| {
                    final_persist_attempts.set(final_persist_attempts.get() + 1);
                    assert_eq!(published_count.get(), 2);
                    assert!(committer.journal.publications.is_empty());
                    for prepared in [&first, &second] {
                        assert_eq!(
                            committer
                                .operation(prepared.operation_id)
                                .map(|operation| operation.state),
                            Some(OutputOperationState::Committed)
                        );
                        assert!(
                            roots
                                .test_root_path(AssetNamespace::Output)?
                                .join(&prepared.identity.relative_path)
                                .is_file()
                        );
                    }
                    Err(OutputCommitError::Io {
                        path: committer
                            .roots
                            .resolve_for_create(&committer.journal_identity)?,
                        message: "injected final journal write failure".to_owned(),
                    })
                },
            )
            .expect_err("the injected final journal failure must abort the batch");
        assert!(matches!(error, OutputCommitError::Io { .. }));
        assert_eq!(final_persist_attempts.get(), 1);
        assert!(committer.journal.publications.is_empty());
        for prepared in [&first, &second] {
            assert_eq!(
                committer
                    .operation(prepared.operation_id)
                    .map(|operation| operation.state),
                Some(OutputOperationState::Prepared)
            );
            assert!(
                committer
                    .staging_path(
                        committer
                            .operation(prepared.operation_id)
                            .ok_or(OutputCommitError::UnknownOperation(prepared.operation_id))?
                    )?
                    .is_file()
            );
            assert!(
                !roots
                    .test_root_path(AssetNamespace::Output)?
                    .join(&prepared.identity.relative_path)
                    .exists()
            );
        }
        let journal_bytes = committer
            .roots
            .read_private(&committer.journal_identity, committer.max_journal_bytes)?
            .ok_or("output journal is missing")?;
        assert_eq!(
            decode_journal(&journal_bytes, committer.max_journal_bytes)?,
            committer.journal
        );

        committer.cancel(first.operation_id, &capabilities)?;
        committer.cancel(second.operation_id, &capabilities)?;
        drop(committer);
        let reopened = OutputCommitter::open(roots.clone())?;
        assert!(reopened.journal.publications.is_empty());
        for prepared in [&first, &second] {
            assert_eq!(
                reopened
                    .operation(prepared.operation_id)
                    .map(|operation| operation.state),
                Some(OutputOperationState::Cancelled)
            );
            assert!(
                !roots
                    .test_root_path(AssetNamespace::Output)?
                    .join(&prepared.identity.relative_path)
                    .exists()
            );
        }
        assert!(fs::read_dir(&reopened.staging_directory)?.next().is_none());
        Ok(())
    }

    #[test]
    fn restart_rolls_back_an_in_progress_batch_without_reusing_identities()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let mut first_request = request(AssetNamespace::Output)?;
        first_request.filename_prefix = "restart/image".to_owned();
        first_request.batch_index = 0;
        let first = committer.prepare(&first_request, b"first", &capabilities, &cancellation)?;
        let mut second_request = first_request.clone();
        second_request.batch_index = 1;
        let second = committer.prepare(&second_request, b"second", &capabilities, &cancellation)?;
        let publication = committer.begin_batch_publication(
            &[first.operation_id, second.operation_id],
            &capabilities,
            &cancellation,
        )?;
        committer.publish_operation(first.operation_id)?;
        let first_path = roots
            .test_root_path(AssetNamespace::Output)?
            .join(&first.identity.relative_path);
        assert!(first_path.is_file());
        drop(committer);

        let mut recovered = OutputCommitter::open(roots.clone())?;
        assert!(recovered.journal.publications.is_empty());
        assert!(!first_path.exists());
        for prepared in [&first, &second] {
            assert_eq!(
                recovered
                    .operation(prepared.operation_id)
                    .map(|operation| operation.state),
                Some(OutputOperationState::Interrupted)
            );
            assert!(
                !roots
                    .test_root_path(AssetNamespace::Output)?
                    .join(&prepared.identity.relative_path)
                    .exists()
            );
        }
        assert!(fs::read_dir(&recovered.staging_directory)?.next().is_none());
        assert_ne!(publication.batch_id, Uuid::nil());

        let retry = recovered.prepare(&first_request, b"retry", &capabilities, &cancellation)?;
        assert_eq!(retry.collision_counter, 3);
        assert_ne!(retry.identity, first.identity);
        assert_ne!(retry.identity, second.identity);
        Ok(())
    }

    #[test]
    fn cancellation_and_explicit_cancel_leave_no_visible_or_staged_file()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let mut committer = OutputCommitter::open(roots.clone())?;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            committer.prepare(
                &request(AssetNamespace::Output)?,
                b"never",
                &capabilities,
                &cancelled,
            ),
            Err(OutputCommitError::Asset(AssetError::Cancelled))
        ));
        let prepared = committer.prepare(
            &request(AssetNamespace::Temporary)?,
            b"preview",
            &capabilities,
            &CancellationToken::default(),
        )?;
        let operation = committer.cancel(prepared.operation_id, &capabilities)?;
        assert_eq!(operation.state, OutputOperationState::Cancelled);
        assert!(
            !roots
                .test_root_path(AssetNamespace::Temporary)?
                .join(prepared.identity.relative_path)
                .exists()
        );
        assert!(fs::read_dir(&committer.staging_directory)?.next().is_none());
        Ok(())
    }

    #[test]
    fn restart_interrupts_prepare_and_recovers_rename_before_journal_commit()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let interrupted = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"interrupted",
            &capabilities,
            &cancellation,
        )?;
        drop(committer);
        let mut reopened = OutputCommitter::open(roots.clone())?;
        assert_eq!(
            reopened
                .operation(interrupted.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::Interrupted)
        );
        assert!(
            !roots
                .test_root_path(AssetNamespace::Output)?
                .join(interrupted.identity.relative_path)
                .exists()
        );

        let prepared = reopened.prepare(
            &request(AssetNamespace::Output)?,
            b"renamed-before-journal",
            &capabilities,
            &cancellation,
        )?;
        let operation = reopened
            .operation(prepared.operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownOperation(prepared.operation_id))?;
        let staging = reopened.staging_path(&operation)?;
        let destination = reopened.destination_path(&operation.identity)?;
        fs::rename(staging, &destination)?;
        drop(reopened);
        let recovered = OutputCommitter::open(roots)?;
        assert_eq!(
            recovered
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::Committed)
        );
        assert_eq!(fs::read(destination)?, b"renamed-before-journal");
        Ok(())
    }

    #[test]
    fn tampering_and_destination_races_are_typed_and_non_destructive()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let mut committer = OutputCommitter::open(roots)?;
        let cancellation = CancellationToken::default();
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"original",
            &capabilities,
            &cancellation,
        )?;
        let operation = committer
            .operation(prepared.operation_id)
            .cloned()
            .ok_or(OutputCommitError::UnknownOperation(prepared.operation_id))?;
        fs::write(committer.staging_path(&operation)?, b"tampered")?;
        assert!(matches!(
            committer.commit(prepared.operation_id, &capabilities, &cancellation),
            Err(OutputCommitError::IntegrityMismatch { .. })
        ));
        committer.cancel(prepared.operation_id, &capabilities)?;

        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"ours",
            &capabilities,
            &cancellation,
        )?;
        let destination = committer.destination_path(&prepared.identity)?;
        fs::write(&destination, b"external")?;
        assert!(matches!(
            committer.commit(prepared.operation_id, &capabilities, &cancellation),
            Err(OutputCommitError::DestinationConflict(_))
        ));
        assert_eq!(fs::read(&destination)?, b"external");
        committer.cancel(prepared.operation_id, &capabilities)?;
        Ok(())
    }

    #[test]
    fn reopened_committed_output_reports_missing_corrupt_and_restored_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"durable",
            &capabilities,
            &cancellation,
        )?;
        committer.commit(prepared.operation_id, &capabilities, &cancellation)?;
        let path = committer.destination_path(&prepared.identity)?;
        fs::remove_file(&path)?;
        drop(committer);
        let missing = OutputCommitter::open(roots.clone())?;
        assert_eq!(
            missing
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::CommittedMissing)
        );
        fs::write(&path, b"different")?;
        drop(missing);
        let corrupt = OutputCommitter::open(roots.clone())?;
        assert_eq!(
            corrupt
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::CommittedCorrupt)
        );
        fs::write(&path, b"durable")?;
        drop(corrupt);
        let restored = OutputCommitter::open(roots)?;
        assert_eq!(
            restored
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::Committed)
        );
        Ok(())
    }

    #[test]
    fn corrupt_journal_is_rejected_without_touching_existing_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, _capabilities) = roots()?;
        let output = roots
            .test_root_path(AssetNamespace::Output)?
            .join("existing.png");
        fs::write(&output, b"preserve")?;
        let journal = roots
            .test_root_path(AssetNamespace::Temporary)?
            .join(JOURNAL_FILENAME);
        fs::write(journal, b"{not-json")?;
        assert!(matches!(
            OutputCommitter::open(roots),
            Err(OutputCommitError::JournalDecode(_))
        ));
        assert_eq!(fs::read(output)?, b"preserve");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn publication_removal_and_rollback_fail_closed_after_root_replacement()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let (directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"anchored-output",
            &capabilities,
            &cancellation,
        )?;
        let configured = roots.test_root_path(AssetNamespace::Output)?.to_path_buf();
        let retained = directory.path().join("retained-output");
        let outside = tempfile::tempdir()?;
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;

        assert!(matches!(
            committer.commit(prepared.operation_id, &capabilities, &cancellation),
            Err(OutputCommitError::Asset(AssetError::InvalidIndex(_)))
        ));
        let retained_output = retained.join(&prepared.identity.relative_path);
        let outside_output = outside.path().join(&prepared.identity.relative_path);
        assert!(!retained_output.exists());
        assert!(!outside_output.exists());

        fs::remove_file(&configured)?;
        fs::rename(&retained, &configured)?;
        committer.commit(prepared.operation_id, &capabilities, &cancellation)?;
        let configured_output = configured.join(&prepared.identity.relative_path);
        assert_eq!(fs::read(&configured_output)?, b"anchored-output");

        let removal =
            committer.prepare_removal(&prepared.identity, &capabilities, &cancellation)?;
        let removal_operation = committer
            .removal(removal.operation_id)
            .cloned()
            .ok_or("prepared removal is missing")?;
        let removal_staging = committer.removal_staging_path(&removal_operation)?;
        assert!(!configured_output.exists());
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;
        fs::create_dir_all(
            outside_output
                .parent()
                .ok_or("outside output has no parent")?,
        )?;
        fs::write(&outside_output, b"foreign")?;
        assert!(matches!(
            committer.rollback_removal(removal.operation_id, &capabilities),
            Err(OutputCommitError::Asset(AssetError::InvalidIndex(_)))
        ));
        assert!(removal_staging.exists());
        assert!(!retained_output.exists());
        assert_eq!(fs::read(outside_output)?, b"foreign");

        fs::remove_file(&configured)?;
        fs::rename(&retained, &configured)?;
        committer.rollback_removal(removal.operation_id, &capabilities)?;
        assert_eq!(fs::read(configured_output)?, b"anchored-output");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn recovery_fails_closed_until_a_replaced_root_is_restored()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let (directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"partial-publication",
            &capabilities,
            &cancellation,
        )?;
        let operation = committer
            .operation(prepared.operation_id)
            .cloned()
            .ok_or("prepared operation is missing")?;
        let staging = committer.staging_path(&operation)?;
        let destination = committer.destination_path(&operation.identity)?;
        fs::create_dir_all(destination.parent().ok_or("destination has no parent")?)?;
        fs::hard_link(&staging, &destination)?;

        let configured = roots.test_root_path(AssetNamespace::Output)?.to_path_buf();
        let retained = directory.path().join("retained-output");
        let outside = tempfile::tempdir()?;
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;
        drop(committer);

        assert!(matches!(
            OutputCommitter::open(roots.clone()),
            Err(OutputCommitError::Asset(AssetError::InvalidIndex(_)))
        ));
        assert!(staging.exists());
        assert_eq!(
            fs::read(retained.join(&prepared.identity.relative_path))?,
            b"partial-publication"
        );
        assert!(
            !outside
                .path()
                .join(&prepared.identity.relative_path)
                .exists()
        );

        fs::remove_file(&configured)?;
        fs::rename(&retained, &configured)?;
        let recovered = OutputCommitter::open(roots)?;
        assert_eq!(
            recovered
                .operation(prepared.operation_id)
                .map(|operation| operation.state),
            Some(OutputOperationState::Committed)
        );
        assert!(!staging.exists());
        assert_eq!(
            fs::read(configured.join(&prepared.identity.relative_path))?,
            b"partial-publication"
        );
        assert!(
            !outside
                .path()
                .join(&prepared.identity.relative_path)
                .exists()
        );
        Ok(())
    }

    #[test]
    fn val_recovery_005_output_transaction_stage() -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, roots, capabilities) = roots()?;
        let cancellation = CancellationToken::default();
        let mut committer = OutputCommitter::open(roots.clone())?;
        let prepared = committer.prepare(
            &request(AssetNamespace::Output)?,
            b"fixture-output",
            &capabilities,
            &cancellation,
        )?;
        let final_path = roots
            .test_root_path(AssetNamespace::Output)?
            .join(&prepared.identity.relative_path);
        let invisible_before_commit = !final_path.exists();
        let committed = committer.commit(prepared.operation_id, &capabilities, &cancellation)?;
        let exact_commit = fs::read(&final_path)? == b"fixture-output";
        let committed_identity = committed.identity == prepared.identity;

        let interrupted = committer.prepare(
            &request(AssetNamespace::Temporary)?,
            b"interrupted",
            &capabilities,
            &cancellation,
        )?;
        drop(committer);
        let recovered = OutputCommitter::open(roots.clone())?;
        let restart_interrupted = recovered
            .operation(interrupted.operation_id)
            .is_some_and(|operation| operation.state == OutputOperationState::Interrupted);
        let no_interrupted_final = !roots
            .test_root_path(AssetNamespace::Temporary)?
            .join(interrupted.identity.relative_path)
            .exists();
        let cases = json!({
            "typed_namespace": prepared.identity.namespace == AssetNamespace::Output,
            "source_collision_name": prepared.identity.filename().is_some_and(|name| name.ends_with("_00001_.png")),
            "invisible_before_commit": invisible_before_commit,
            "exact_commit": exact_commit,
            "shared_asset_identity": committed_identity,
            "restart_interrupts_prepare": restart_interrupted,
            "no_interrupted_final": no_interrupted_final,
            "journal_is_bounded": fs::metadata(roots.test_root_path(AssetNamespace::Temporary)?.join(JOURNAL_FILENAME))?.len() <= DEFAULT_MAX_OUTPUT_JOURNAL_BYTES as u64,
            "native_only": true,
            "zero_partial_files": fs::read_dir(roots.test_root_path(AssetNamespace::Temporary)?.join(STAGING_SUBFOLDER))?.next().is_none(),
        });
        assert!(
            cases
                .as_object()
                .is_some_and(|cases| cases.values().all(|value| value == &Value::Bool(true)))
        );
        let artifact = json!({
            "validation": "VAL-RECOVERY-005",
            "scope": "native-output-transaction-stage",
            "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "native-rust"},
            "fixture_digests": {
                "committed_sha256": sha256(b"fixture-output"),
                "interrupted_sha256": sha256(b"interrupted"),
            },
            "summary": {"passed": 10, "failed": 0, "skipped": 0},
            "cases": cases,
            "skipped": [],
            "subprocesses": 0,
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("target")
            });
        let artifact_directory = target.join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        fs::write(
            artifact_directory.join("val-recovery-005.json"),
            serde_json::to_vec_pretty(&artifact)?,
        )?;
        Ok(())
    }
}
