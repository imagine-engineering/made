use std::collections::BTreeMap;

use collab::migration::buzz::checkpoint::{
    MigrationCheckpoint, MigrationCheckpointStatus, MigrationCheckpointUpdate, MigrationCounts,
    MigrationCursor, MigrationStream, RollbackBoundary,
};
use collaboration_domain::CommunityId;
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[path = "../../zed/src/migration/buzz/agent_staging.rs"]
mod agent_staging;
#[path = "../../zed/src/migration/buzz/desktop_state.rs"]
mod desktop_state;

use agent_staging::{
    BuzzAgentJsonKind, BuzzAgentJsonSource, BuzzAgentJsonStagingRecord, BuzzAgentStagingBatch,
    BuzzAgentStagingImporter, BuzzAgentStagingRecord,
};
use desktop_state::import_desktop_state_bytes;

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeploymentFixture {
    binary: Vec<u8>,
    configuration: Vec<u8>,
    data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StreamObservation {
    source_count: u64,
    target_count: u64,
    source_hash: [u8; 32],
    target_hash: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerificationSet(BTreeMap<&'static str, StreamObservation>);

impl VerificationSet {
    fn aggregate(&self) -> Result<AggregateObservation, RecoveryError> {
        let mut source_hasher = Sha256::new();
        let mut target_hasher = Sha256::new();
        let mut scanned = 0_u64;
        let mut imported = 0_u64;
        for (stream, observation) in &self.0 {
            scanned = scanned
                .checked_add(observation.source_count.max(observation.target_count))
                .ok_or(RecoveryError::InvalidObservation)?;
            imported = imported
                .checked_add(observation.target_count)
                .ok_or(RecoveryError::InvalidObservation)?;
            hash_part(&mut source_hasher, stream.as_bytes());
            hash_part(&mut source_hasher, &observation.source_count.to_be_bytes());
            hash_part(&mut source_hasher, &observation.source_hash);
            hash_part(&mut target_hasher, stream.as_bytes());
            hash_part(&mut target_hasher, &observation.target_count.to_be_bytes());
            hash_part(&mut target_hasher, &observation.target_hash);
        }
        Ok(AggregateObservation {
            counts: MigrationCounts {
                scanned,
                imported,
                skipped: 0,
                failed: 0,
            },
            source_hash: source_hasher.finalize().into(),
            target_hash: target_hasher.finalize().into(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AggregateObservation {
    counts: MigrationCounts,
    source_hash: [u8; 32],
    target_hash: [u8; 32],
}

#[derive(Debug, Eq, PartialEq)]
enum RecoveryError {
    Divergence,
    Halted,
    InvalidObservation,
    Checkpoint,
}

#[derive(Clone)]
struct RecoveryHarness {
    pre_boundary: DeploymentFixture,
    active: DeploymentFixture,
    checkpoint: MigrationCheckpoint,
    last_verified: Option<AggregateObservation>,
    halted: bool,
}

impl RecoveryHarness {
    fn new(pre_boundary: DeploymentFixture) -> Result<Self, RecoveryError> {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
        let boundary = RollbackBoundary::reversible("pre-buzz-import")
            .map_err(|_| RecoveryError::Checkpoint)?;
        let checkpoint = MigrationCheckpoint::new(
            community_id,
            Uuid::from_u128(2),
            "buzz-schema-30",
            MigrationStream::DesktopState,
            "combined-private-state",
            boundary.clone(),
        )
        .map_err(|_| RecoveryError::Checkpoint)?;
        let checkpoint = checkpoint
            .transition(MigrationCheckpointUpdate {
                status: MigrationCheckpointStatus::Running,
                source_cursor: MigrationCursor::new(0, None)
                    .map_err(|_| RecoveryError::Checkpoint)?,
                target_cursor: MigrationCursor::new(0, None)
                    .map_err(|_| RecoveryError::Checkpoint)?,
                counts: MigrationCounts::default(),
                source_hash: None,
                target_hash: None,
                rollback_boundary: boundary,
                last_error: None,
            })
            .map_err(|_| RecoveryError::Checkpoint)?;
        Ok(Self {
            active: pre_boundary.clone(),
            pre_boundary,
            checkpoint,
            last_verified: None,
            halted: false,
        })
    }

    fn activate_candidate(&mut self, candidate: DeploymentFixture) {
        self.active = candidate;
    }

    fn verify(
        &mut self,
        expected: &VerificationSet,
        observed: &VerificationSet,
    ) -> Result<(), RecoveryError> {
        if self.halted {
            return Err(RecoveryError::Halted);
        }
        if expected != observed {
            self.fail_and_restore()?;
            return Err(RecoveryError::Divergence);
        }
        let aggregate = observed.aggregate()?;
        let boundary = self.checkpoint.rollback_boundary().clone();
        self.checkpoint = self
            .checkpoint
            .transition(MigrationCheckpointUpdate {
                status: MigrationCheckpointStatus::Running,
                source_cursor: MigrationCursor::new(aggregate.counts.scanned, None)
                    .map_err(|_| RecoveryError::Checkpoint)?,
                target_cursor: MigrationCursor::new(aggregate.counts.imported, None)
                    .map_err(|_| RecoveryError::Checkpoint)?,
                counts: aggregate.counts,
                source_hash: Some(aggregate.source_hash),
                target_hash: Some(aggregate.target_hash),
                rollback_boundary: boundary,
                last_error: None,
            })
            .map_err(|_| RecoveryError::Checkpoint)?;
        self.last_verified = Some(aggregate);
        Ok(())
    }

    fn interrupt_and_resume(&mut self) -> Result<(), RecoveryError> {
        let observation = self
            .last_verified
            .ok_or(RecoveryError::InvalidObservation)?;
        self.transition_without_progress(
            MigrationCheckpointStatus::Interrupted,
            Some("simulated interruption".to_owned()),
            observation,
        )?;
        self.transition_without_progress(MigrationCheckpointStatus::Running, None, observation)
    }

    fn fail_and_restore(&mut self) -> Result<(), RecoveryError> {
        let observation = self
            .last_verified
            .ok_or(RecoveryError::InvalidObservation)?;
        self.transition_without_progress(
            MigrationCheckpointStatus::Failed,
            Some("verification divergence".to_owned()),
            observation,
        )?;
        self.active = self.pre_boundary.clone();
        self.transition_without_progress(MigrationCheckpointStatus::RolledBack, None, observation)?;
        self.halted = true;
        Ok(())
    }

    fn transition_without_progress(
        &mut self,
        status: MigrationCheckpointStatus,
        last_error: Option<String>,
        observation: AggregateObservation,
    ) -> Result<(), RecoveryError> {
        let boundary = self.checkpoint.rollback_boundary().clone();
        self.checkpoint = self
            .checkpoint
            .transition(MigrationCheckpointUpdate {
                status,
                source_cursor: self.checkpoint.source_cursor().clone(),
                target_cursor: self.checkpoint.target_cursor().clone(),
                counts: observation.counts,
                source_hash: Some(observation.source_hash),
                target_hash: Some(observation.target_hash),
                rollback_boundary: boundary,
                last_error,
            })
            .map_err(|_| RecoveryError::Checkpoint)?;
        Ok(())
    }
}

fn desktop_observation() -> StreamObservation {
    let source = serde_json::to_vec(&serde_json::json!({
        "snapshot_version": 1,
        "captured_at_millis": 1_700_000_000_000_u64,
        "source_application_id": "xyz.block.buzz.app",
        "general_configuration": {"prevent_sleep": true},
        "local_storage": {},
        "archive": {
            "schema_version": 1,
            "migration_markers": [],
            "events": [],
            "scopes": [],
            "subscriptions": []
        }
    }))
    .expect("desktop fixture");
    let first = import_desktop_state_bytes(&source).expect("desktop import");
    let replay = import_desktop_state_bytes(&source).expect("desktop replay");
    assert_eq!(first, replay);
    StreamObservation {
        source_count: 1,
        target_count: u64::try_from(first.settings.len()).expect("settings count"),
        source_hash: first.source_hash,
        target_hash: first.target_hash,
    }
}

fn agent_observation() -> StreamObservation {
    let source = serde_json::to_vec(&serde_json::json!({
        "id": "reviewer",
        "display_name": "Reviewer",
        "system_prompt": "Review the change"
    }))
    .expect("agent fixture");
    let record = BuzzAgentJsonStagingRecord::from_source(
        BuzzAgentJsonSource {
            owner_profile_id: "profile-1",
            source_sequence: 1,
            source_path: "agents/managed-agents.json#reviewer",
            semantic_id: "reviewer",
            kind: BuzzAgentJsonKind::Persona,
            source_schema_version: 1,
            source_bytes: &source,
        },
        Vec::new(),
    )
    .expect("agent record");
    let batch =
        BuzzAgentStagingBatch::new(1, "profile-1", vec![BuzzAgentStagingRecord::Json(record)])
            .expect("agent batch");
    assert_eq!(batch.records().len(), 1);
    let first = BuzzAgentStagingImporter::stage("profile-1", &batch).expect("agent stage");
    let replay = BuzzAgentStagingImporter::stage("profile-1", &batch).expect("agent replay");
    assert_eq!(first, replay);
    StreamObservation {
        source_count: 1,
        target_count: first.checkpoint.staged,
        source_hash: first.checkpoint.source_hash,
        target_hash: first.checkpoint.staged_hash,
    }
}

fn verification_set() -> VerificationSet {
    VerificationSet(BTreeMap::from([
        ("agent_state", agent_observation()),
        ("desktop_state", desktop_observation()),
    ]))
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_be_bytes());
    hasher.update(bytes);
}

#[test]
fn resumes_idempotently_then_halts_and_restores_on_divergence() {
    let pre_boundary = DeploymentFixture {
        binary: b"zed-before".to_vec(),
        configuration: b"gateway_enabled=false".to_vec(),
        data: b"canonical-before".to_vec(),
    };
    let candidate = DeploymentFixture {
        binary: b"zed-candidate".to_vec(),
        configuration: b"gateway_enabled=false;migration=verify".to_vec(),
        data: b"canonical-candidate".to_vec(),
    };
    let expected = verification_set();
    let mut harness = RecoveryHarness::new(pre_boundary.clone()).expect("recovery harness");
    harness.activate_candidate(candidate.clone());

    harness.verify(&expected, &expected).expect("first window");
    let counts_after_first = harness.checkpoint.counts();
    harness.interrupt_and_resume().expect("resume checkpoint");
    harness
        .verify(&expected, &expected)
        .expect("idempotent replay");
    assert_eq!(harness.checkpoint.counts(), counts_after_first);
    assert_eq!(harness.active, candidate);

    let mut divergent = expected.clone();
    divergent
        .0
        .get_mut("agent_state")
        .expect("agent observation")
        .target_hash[0] ^= 0xff;
    assert_eq!(
        harness.verify(&expected, &divergent),
        Err(RecoveryError::Divergence)
    );
    assert_eq!(
        harness.checkpoint.status(),
        MigrationCheckpointStatus::RolledBack
    );
    assert_eq!(harness.active, pre_boundary);
    assert_eq!(
        harness.verify(&expected, &expected),
        Err(RecoveryError::Halted)
    );
}
