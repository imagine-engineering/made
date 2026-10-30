use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use chrono::{DateTime, Utc};
use db::{
    query,
    sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
    sqlez_macros::sql,
};
use futures::{FutureExt as _, future::BoxFuture};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use sim_media::NativeAudioPayload;
use sim_model::NativeModelResourceIdentity;
use sim_nodes::{NativeModelResourceRequest, execution::NativeModelResourcePreflight};
use sim_plugin_sdk::ValueFamily;
use sim_types::NodeId;

use crate::{
    ATTEMPT_HISTORY_SCHEMA_VERSION, AttemptRecord, AttemptState, CompiledPlan, ExecutionCommandAck,
    ExecutionDataSource, ExecutionQueue, ExecutionSnapshotStatus, LegacyMigrationResult,
    PersistedNativeConditioningControlEvidence, PersistedNativeModelTrainingEvidence,
    PersistedNativeModelTransformEvidence, QueuedPrompt, RequestId,
};

pub const PERSISTED_EXECUTION_ATTEMPT_SCHEMA_VERSION: u16 = 2;
pub const PERSISTED_EXECUTION_PROFILE_SCHEMA_VERSION: u16 = 1;
pub const PERSISTED_NATIVE_MODEL_RESOURCE_EVIDENCE_SCHEMA_VERSION: u16 = 1;
pub const PERSISTED_NATIVE_AUDIO_EVIDENCE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedNativeModelResourceEvidence {
    pub schema_version: u16,
    pub node_id: NodeId,
    pub preflight: NativeModelResourcePreflight,
    pub resource_identity: NativeModelResourceIdentity,
}

impl PersistedNativeModelResourceEvidence {
    pub fn checked(
        node_id: NodeId,
        preflight: NativeModelResourcePreflight,
        resource_identity: NativeModelResourceIdentity,
    ) -> Result<Self> {
        let evidence = Self {
            schema_version: PERSISTED_NATIVE_MODEL_RESOURCE_EVIDENCE_SCHEMA_VERSION,
            node_id,
            preflight,
            resource_identity,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == PERSISTED_NATIVE_MODEL_RESOURCE_EVIDENCE_SCHEMA_VERSION,
            "unsupported persisted native model-resource evidence schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            !self.node_id.0.trim().is_empty(),
            "persisted native model-resource evidence has no node identity"
        );
        let request = NativeModelResourceRequest::checked(
            self.preflight.role(),
            self.preflight.folder_category(),
            self.preflight.source_names().to_vec(),
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        self.preflight
            .validate_for(&request)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        self.resource_identity
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            self.resource_identity.role() == self.preflight.role(),
            "persisted native model-resource role changed"
        );
        Ok(())
    }
}

pub const NATIVE_MODEL_RESOURCE_EVIDENCE_UI_KEY: &str = "native_model_resource_evidence";
pub const NATIVE_CONDITIONING_CONTROL_EVIDENCE_UI_KEY: &str =
    "native_conditioning_control_evidence";
pub const NATIVE_MODEL_TRANSFORM_EVIDENCE_UI_KEY: &str = "native_model_transform_evidence";
pub const NATIVE_MODEL_TRAINING_EVIDENCE_UI_KEY: &str = "native_model_training_evidence";
pub const NATIVE_AUDIO_EVIDENCE_UI_KEY: &str = "native_audio_evidence";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedNativeAudioEvidence {
    pub schema_version: u16,
    pub node_id: NodeId,
    pub semantic_digest_sha256: String,
    pub transport_sha256: String,
    pub sample_rate: u32,
    pub shape: [u64; 3],
    pub transport_bytes: Vec<u8>,
}

impl PersistedNativeAudioEvidence {
    pub fn checked(node_id: NodeId, payload: &NativeAudioPayload) -> Result<Self> {
        payload
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let shape: [u64; 3] = payload
            .waveform()
            .descriptor()
            .shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("persisted native AUDIO shape is not rank three"))?;
        let transport_bytes = crate::native_audio_persistence_transport(payload)
            .and_then(|transport| transport.to_bytes())
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let evidence = Self {
            schema_version: PERSISTED_NATIVE_AUDIO_EVIDENCE_SCHEMA_VERSION,
            node_id,
            semantic_digest_sha256: encode_sha256(payload.semantic_digest_sha256()),
            transport_sha256: format!("{:x}", Sha256::digest(&transport_bytes)),
            sample_rate: payload.sample_rate(),
            shape,
            transport_bytes,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == PERSISTED_NATIVE_AUDIO_EVIDENCE_SCHEMA_VERSION,
            "unsupported persisted native AUDIO evidence schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            !self.node_id.0.trim().is_empty()
                && is_lowercase_sha256(&self.semantic_digest_sha256)
                && is_lowercase_sha256(&self.transport_sha256)
                && self.transport_sha256 == format!("{:x}", Sha256::digest(&self.transport_bytes))
                && self.shape[0] > 0
                && self.shape[1] > 0
                && self.shape[2] == 0
                && (1..=384_000).contains(&self.sample_rate),
            "persisted native AUDIO evidence is invalid"
        );
        let transport = crate::ProviderTransportResponse::from_bytes(&self.transport_bytes)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            transport.class_type() == crate::NATIVE_AUDIO_PERSISTENCE_TRANSPORT_CLASS,
            "persisted native AUDIO transport class changed"
        );
        let [audio, audio_list] = transport.ports() else {
            anyhow::bail!("persisted native AUDIO transport ports changed");
        };
        anyhow::ensure!(
            audio.port_id() == "audio"
                && audio.present()
                && audio.values().len() == 1
                && audio_list.port_id() == "audio_list"
                && audio_list.present()
                && audio_list.values().len() == 1,
            "persisted native AUDIO/list transport contract changed"
        );
        for value in audio.values().iter().chain(audio_list.values()) {
            anyhow::ensure!(
                value.type_id() == "AUDIO"
                    && value.family() == ValueFamily::Tensor
                    && value.encoding() == crate::ProviderTransportEncoding::NativePayload,
                "persisted native AUDIO transport value changed"
            );
        }
        Ok(())
    }

    pub fn matches_audio(&self, payload: &NativeAudioPayload) -> bool {
        Self::checked(self.node_id.clone(), payload).is_ok_and(|actual| actual == *self)
    }
}

pub fn native_audio_reconstruction_for_retry(
    attempts: &[PersistedExecutionAttempt],
    attempt_id: crate::AttemptId,
) -> Result<BTreeMap<NodeId, PersistedNativeAudioEvidence>> {
    let retry_of = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == attempt_id)
        .and_then(|attempt| attempt.record.retry_of);
    let Some(retry_of) = retry_of else {
        return Ok(BTreeMap::new());
    };
    let prior = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == retry_of)
        .ok_or_else(|| anyhow::anyhow!("native AUDIO retry source is unavailable"))?;
    if prior.record.state != AttemptState::Succeeded {
        return Ok(BTreeMap::new());
    }
    let ui_outputs = prior
        .record
        .events
        .iter()
        .rev()
        .find(|event| matches!(event.kind, crate::AttemptEventKind::Succeeded))
        .and_then(|event| event.data.as_ref())
        .and_then(|data| data.get("ui_outputs"))
        .and_then(serde_json::Value::as_object);
    let Some(ui_outputs) = ui_outputs else {
        return Ok(BTreeMap::new());
    };
    native_audio_evidence_from_ui_outputs(ui_outputs)
}

pub fn native_audio_evidence_from_ui_outputs(
    ui_outputs: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeAudioEvidence>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_AUDIO_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: PersistedNativeAudioEvidence = serde_json::from_value(evidence.clone())?;
        let node_id = NodeId(node_id.clone());
        anyhow::ensure!(
            evidence.node_id == node_id && evidence.validate().is_ok(),
            "persisted native AUDIO worker evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id, evidence).is_none(),
            "persisted native AUDIO worker evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

pub fn native_audio_evidence_from_node_outputs(
    ui_outputs: &BTreeMap<NodeId, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeAudioEvidence>> {
    let ui_outputs = ui_outputs
        .iter()
        .map(|(node_id, value)| (node_id.0.clone(), value.clone()))
        .collect();
    native_audio_evidence_from_ui_outputs(&ui_outputs)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_sha256(value: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

pub fn native_model_resource_reconstruction_for_retry(
    attempts: &[PersistedExecutionAttempt],
    attempt_id: crate::AttemptId,
) -> Result<BTreeMap<NodeId, Vec<PersistedNativeModelResourceEvidence>>> {
    let retry_of = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == attempt_id)
        .and_then(|attempt| attempt.record.retry_of);
    let Some(retry_of) = retry_of else {
        return Ok(BTreeMap::new());
    };
    let prior = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == retry_of)
        .ok_or_else(|| anyhow::anyhow!("native model-resource retry source is unavailable"))?;
    if prior.record.state != AttemptState::Succeeded {
        return Ok(BTreeMap::new());
    }
    let ui_outputs = prior
        .record
        .events
        .iter()
        .rev()
        .find(|event| matches!(event.kind, crate::AttemptEventKind::Succeeded))
        .and_then(|event| event.data.as_ref())
        .and_then(|data| data.get("ui_outputs"))
        .and_then(serde_json::Value::as_object);
    let Some(ui_outputs) = ui_outputs else {
        return Ok(BTreeMap::new());
    };
    native_model_resource_evidence_from_ui_outputs(ui_outputs)
}

pub fn native_model_resource_evidence_from_ui_outputs(
    ui_outputs: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<NodeId, Vec<PersistedNativeModelResourceEvidence>>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_MODEL_RESOURCE_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: Vec<PersistedNativeModelResourceEvidence> =
            serde_json::from_value(evidence.clone())?;
        let node_id = NodeId(node_id.clone());
        anyhow::ensure!(
            !evidence.is_empty()
                && evidence
                    .iter()
                    .all(|record| { record.node_id == node_id && record.validate().is_ok() }),
            "persisted native model-resource evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id, evidence).is_none(),
            "persisted native model-resource evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

pub fn native_conditioning_control_reconstruction_for_retry(
    attempts: &[PersistedExecutionAttempt],
    attempt_id: crate::AttemptId,
) -> Result<BTreeMap<NodeId, PersistedNativeConditioningControlEvidence>> {
    let retry_of = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == attempt_id)
        .and_then(|attempt| attempt.record.retry_of);
    let Some(retry_of) = retry_of else {
        return Ok(BTreeMap::new());
    };
    let prior = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == retry_of)
        .ok_or_else(|| {
            anyhow::anyhow!("native conditioning/control retry source is unavailable")
        })?;
    if prior.record.state != AttemptState::Succeeded {
        return Ok(BTreeMap::new());
    }
    let ui_outputs = prior
        .record
        .events
        .iter()
        .rev()
        .find(|event| matches!(event.kind, crate::AttemptEventKind::Succeeded))
        .and_then(|event| event.data.as_ref())
        .and_then(|data| data.get("ui_outputs"))
        .and_then(serde_json::Value::as_object);
    let Some(ui_outputs) = ui_outputs else {
        return Ok(BTreeMap::new());
    };
    native_conditioning_control_evidence_from_ui_outputs(ui_outputs)
}

pub fn native_model_transform_reconstruction_for_retry(
    attempts: &[PersistedExecutionAttempt],
    attempt_id: crate::AttemptId,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTransformEvidence>> {
    let retry_of = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == attempt_id)
        .and_then(|attempt| attempt.record.retry_of);
    let Some(retry_of) = retry_of else {
        return Ok(BTreeMap::new());
    };
    let prior = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == retry_of)
        .ok_or_else(|| anyhow::anyhow!("native model-transform retry source is unavailable"))?;
    if prior.record.state != AttemptState::Succeeded {
        return Ok(BTreeMap::new());
    }
    let ui_outputs = prior
        .record
        .events
        .iter()
        .rev()
        .find(|event| matches!(event.kind, crate::AttemptEventKind::Succeeded))
        .and_then(|event| event.data.as_ref())
        .and_then(|data| data.get("ui_outputs"))
        .and_then(serde_json::Value::as_object);
    let Some(ui_outputs) = ui_outputs else {
        return Ok(BTreeMap::new());
    };
    native_model_transform_evidence_from_ui_outputs(ui_outputs)
}

pub fn native_model_training_reconstruction_for_retry(
    attempts: &[PersistedExecutionAttempt],
    attempt_id: crate::AttemptId,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTrainingEvidence>> {
    let retry_of = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == attempt_id)
        .and_then(|attempt| attempt.record.retry_of);
    let Some(retry_of) = retry_of else {
        return Ok(BTreeMap::new());
    };
    let prior = attempts
        .iter()
        .find(|attempt| attempt.record.attempt_id == retry_of)
        .ok_or_else(|| anyhow::anyhow!("native model-training retry source is unavailable"))?;
    if prior.record.state != AttemptState::Succeeded {
        return Ok(BTreeMap::new());
    }
    let ui_outputs = prior
        .record
        .events
        .iter()
        .rev()
        .find(|event| matches!(event.kind, crate::AttemptEventKind::Succeeded))
        .and_then(|event| event.data.as_ref())
        .and_then(|data| data.get("ui_outputs"))
        .and_then(serde_json::Value::as_object);
    let Some(ui_outputs) = ui_outputs else {
        return Ok(BTreeMap::new());
    };
    native_model_training_evidence_from_ui_outputs(ui_outputs)
}

pub fn native_model_training_evidence_from_ui_outputs(
    ui_outputs: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTrainingEvidence>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_MODEL_TRAINING_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: PersistedNativeModelTrainingEvidence =
            serde_json::from_value(evidence.clone())?;
        let node_id = NodeId(node_id.clone());
        anyhow::ensure!(
            evidence.node_id == node_id && evidence.validate().is_ok(),
            "native model-training worker evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id, evidence).is_none(),
            "native model-training worker evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

pub fn native_model_transform_evidence_from_ui_outputs(
    ui_outputs: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTransformEvidence>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_MODEL_TRANSFORM_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: PersistedNativeModelTransformEvidence =
            serde_json::from_value(evidence.clone())?;
        let node_id = NodeId(node_id.clone());
        anyhow::ensure!(
            evidence.node_id == node_id && evidence.validate().is_ok(),
            "native model-transform worker evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id, evidence).is_none(),
            "native model-transform worker evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

pub fn native_conditioning_control_evidence_from_ui_outputs(
    ui_outputs: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeConditioningControlEvidence>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_CONDITIONING_CONTROL_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: PersistedNativeConditioningControlEvidence =
            serde_json::from_value(evidence.clone())?;
        let node_id = NodeId(node_id.clone());
        anyhow::ensure!(
            evidence.node_id == node_id && evidence.validate().is_ok(),
            "native conditioning/control worker evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id, evidence).is_none(),
            "native conditioning/control worker evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

pub fn native_conditioning_control_evidence_from_node_outputs(
    ui_outputs: &BTreeMap<NodeId, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeConditioningControlEvidence>> {
    let ui_outputs = ui_outputs
        .iter()
        .map(|(node_id, output)| (node_id.0.clone(), output.clone()))
        .collect::<serde_json::Map<_, _>>();
    native_conditioning_control_evidence_from_ui_outputs(&ui_outputs)
}

pub fn native_model_transform_evidence_from_node_outputs(
    ui_outputs: &BTreeMap<NodeId, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTransformEvidence>> {
    let ui_outputs = ui_outputs
        .iter()
        .map(|(node_id, output)| (node_id.0.clone(), output.clone()))
        .collect::<serde_json::Map<_, _>>();
    native_model_transform_evidence_from_ui_outputs(&ui_outputs)
}

pub fn native_model_training_evidence_from_node_outputs(
    ui_outputs: &BTreeMap<NodeId, serde_json::Value>,
) -> Result<BTreeMap<NodeId, PersistedNativeModelTrainingEvidence>> {
    let ui_outputs = ui_outputs
        .iter()
        .map(|(node_id, output)| (node_id.0.clone(), output.clone()))
        .collect::<serde_json::Map<_, _>>();
    native_model_training_evidence_from_ui_outputs(&ui_outputs)
}

pub fn native_model_resource_evidence_from_node_outputs(
    ui_outputs: &BTreeMap<NodeId, serde_json::Value>,
) -> Result<BTreeMap<NodeId, Vec<PersistedNativeModelResourceEvidence>>> {
    let mut reconstructed = BTreeMap::new();
    for (node_id, output) in ui_outputs {
        let Some(evidence) = output.get(NATIVE_MODEL_RESOURCE_EVIDENCE_UI_KEY) else {
            continue;
        };
        let evidence: Vec<PersistedNativeModelResourceEvidence> =
            serde_json::from_value(evidence.clone())?;
        anyhow::ensure!(
            !evidence.is_empty()
                && evidence
                    .iter()
                    .all(|record| { record.node_id == *node_id && record.validate().is_ok() }),
            "native model-resource worker evidence is invalid"
        );
        anyhow::ensure!(
            reconstructed.insert(node_id.clone(), evidence).is_none(),
            "native model-resource worker evidence repeats a node"
        );
    }
    Ok(reconstructed)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedExecutionProfile {
    pub schema_version: u16,
    pub profile_id: crate::ProfileId,
    pub revision: u64,
    pub source_revision: Option<u64>,
    pub source: ExecutionDataSource,
    pub status: ExecutionSnapshotStatus,
    pub next_attempt: Option<String>,
    pub completed_requests: Vec<RequestId>,
    pub recent_command_results: Vec<ExecutionCommandAck>,
}

impl PersistedExecutionProfile {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == PERSISTED_EXECUTION_PROFILE_SCHEMA_VERSION,
            "unsupported persisted execution-profile schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.completed_requests.len() <= crate::COMPLETED_REQUEST_CAPACITY,
            "persisted execution profile exceeds the completed-request bound"
        );
        anyhow::ensure!(
            self.recent_command_results.len() <= crate::RECENT_COMMAND_RESULT_CAPACITY,
            "persisted execution profile exceeds the command-result bound"
        );
        let mut completed = BTreeSet::new();
        for request_id in &self.completed_requests {
            anyhow::ensure!(
                completed.insert(request_id.0),
                "persisted execution profile repeats a completed request"
            );
        }
        for acknowledgement in &self.recent_command_results {
            anyhow::ensure!(
                acknowledgement.profile_id == self.profile_id,
                "persisted command result belongs to another profile"
            );
            anyhow::ensure!(
                completed.contains(&acknowledgement.request_id.0),
                "persisted command result is missing its completed-request identity"
            );
        }
        if let Some(next_attempt) = &self.next_attempt {
            next_attempt
                .parse::<u128>()
                .map_err(|error| anyhow::anyhow!("invalid next attempt cursor: {error}"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PersistedQueueMetadata {
    pub position: usize,
    pub priority: i32,
    pub front: bool,
    pub enqueue_sequence: u64,
    pub queued_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedExecutionAttempt {
    pub schema_version: u16,
    pub record: AttemptRecord,
    pub plan: Option<CompiledPlan>,
    pub source: ExecutionDataSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue: Option<PersistedQueueMetadata>,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, serde_json::Value>,
}

impl PersistedExecutionAttempt {
    pub fn new(
        record: AttemptRecord,
        plan: Option<CompiledPlan>,
        source: ExecutionDataSource,
    ) -> Result<Self> {
        let attempt = Self {
            schema_version: PERSISTED_EXECUTION_ATTEMPT_SCHEMA_VERSION,
            record,
            plan,
            source,
            queue: None,
            unknown_fields: BTreeMap::new(),
        };
        attempt.validate()?;
        Ok(attempt)
    }

    pub fn new_with_queue(
        record: AttemptRecord,
        plan: Option<CompiledPlan>,
        source: ExecutionDataSource,
        queue: Option<PersistedQueueMetadata>,
    ) -> Result<Self> {
        let mut attempt = Self::new(record, plan, source)?;
        attempt.queue = queue;
        attempt.validate()?;
        Ok(attempt)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == PERSISTED_EXECUTION_ATTEMPT_SCHEMA_VERSION,
            "unsupported persisted execution attempt schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.record.schema_version == ATTEMPT_HISTORY_SCHEMA_VERSION,
            "unsupported attempt history schema {}",
            self.record.schema_version
        );
        anyhow::ensure!(
            self.record
                .source_projection
                .as_ref()
                .is_none_or(crate::AttemptSourceProjection::is_valid),
            "persisted execution attempt has an invalid source projection"
        );
        crate::execution_presentation::validate_record_integrity(&self.record)?;
        ensure_unknown_fields_do_not_shadow(
            &self.unknown_fields,
            &["schema_version", "record", "plan", "source", "queue"],
        )?;
        ensure_unknown_fields_do_not_shadow(
            &self.record.persistence_unknown_fields,
            &[
                "schema_version",
                "profile_id",
                "prompt_id",
                "attempt_id",
                "retry_of",
                "retry_source",
                "source_projection",
                "state",
                "last_sequence",
                "canonical_event_count",
                "events",
                "output_availability_overrides",
                "created_at",
                "finished_at",
            ],
        )?;
        if let Some(plan) = &self.plan {
            anyhow::ensure!(
                plan.prompt_id == self.record.prompt_id,
                "persisted execution plan does not match its attempt prompt"
            );
            plan.validate_integrity()
                .map_err(|error| anyhow::anyhow!("persisted compiled plan is invalid: {error}"))?;
            ensure_unknown_fields_do_not_shadow(
                &plan.persistence_unknown_fields,
                &[
                    "prompt_id",
                    "client_id",
                    "prompt_number",
                    "extra_data",
                    "unknown",
                    "nodes",
                    "topological_order",
                    "static_required_nodes",
                    "output_nodes",
                    "provider_execution",
                ],
            )?;
        }
        anyhow::ensure!(
            self.queue.is_none() || self.record.state == AttemptState::Queued,
            "only queued execution attempts may retain queue metadata"
        );
        Ok(())
    }
}

fn validate_profile_projection(
    profile_id: crate::ProfileId,
    attempts: &[PersistedExecutionAttempt],
) -> Result<()> {
    let mut attempt_ids = BTreeSet::new();
    let mut queue_positions = BTreeSet::new();
    let mut queued = Vec::new();
    for attempt in attempts {
        attempt.validate()?;
        anyhow::ensure!(
            attempt.record.profile_id == profile_id,
            "persisted execution attempt belongs to another profile"
        );
        anyhow::ensure!(
            attempt_ids.insert(attempt.record.attempt_id.0),
            "persisted execution projection repeats an attempt identity"
        );
        if let Some(queue) = &attempt.queue {
            anyhow::ensure!(
                queue_positions.insert(queue.position),
                "persisted execution projection repeats a queue position"
            );
            let plan = attempt.plan.clone().ok_or_else(|| {
                anyhow::anyhow!("queued persisted execution attempt has no retained plan")
            })?;
            queued.push((
                queue.position,
                QueuedPrompt {
                    profile_id,
                    prompt_id: attempt.record.prompt_id,
                    attempt_id: attempt.record.attempt_id,
                    plan,
                    priority: queue.priority,
                    front: queue.front,
                    enqueue_sequence: queue.enqueue_sequence,
                    queued_at: queue.queued_at,
                },
            ));
        }
    }
    queued.sort_by_key(|(position, _)| *position);
    anyhow::ensure!(
        queued
            .iter()
            .enumerate()
            .all(|(expected, (position, _))| expected == *position),
        "persisted execution queue positions are not contiguous"
    );
    ExecutionQueue::from_ordered(queued.into_iter().map(|(_, queued)| queued).collect())?;
    Ok(())
}

fn ensure_unknown_fields_do_not_shadow(
    fields: &BTreeMap<String, serde_json::Value>,
    reserved: &[&str],
) -> Result<()> {
    for key in fields.keys() {
        anyhow::ensure!(
            !reserved.contains(&key.as_str()),
            "unknown field `{key}` shadows a persisted attempt field"
        );
    }
    Ok(())
}

pub struct SimRuntimeDb(ThreadSafeConnection);

pub trait ExecutionAttemptPersistence: Send + Sync {
    fn replace_execution_state(
        &self,
        profile: PersistedExecutionProfile,
        attempts: Vec<PersistedExecutionAttempt>,
    ) -> BoxFuture<'static, Result<()>>;

    fn load_execution_state(
        &self,
        profile_id: crate::ProfileId,
    ) -> Result<(
        Option<PersistedExecutionProfile>,
        Vec<PersistedExecutionAttempt>,
    )>;
}

impl Domain for SimRuntimeDb {
    // Existing migrations are keyed by this domain; renaming it would hide saved state.
    const NAME: &str = "ComfyRuntimeDb";
    const MIGRATIONS: &[&str] = &[
        sql!(
            CREATE TABLE sim_runtime_profiles (
                profile_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE sim_runtime_attempts (
                attempt_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                attempt_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE sim_runtime_workspaces (
                workspace_id TEXT NOT NULL,
                window_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, window_id)
            ) STRICT;
            CREATE TABLE sim_runtime_legacy_profiles (
                migration_id TEXT PRIMARY KEY,
                inactive_profile_json TEXT NOT NULL,
                migration_result_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE sim_runtime_mappings (
                profile_id TEXT NOT NULL,
                legacy_identifier TEXT NOT NULL,
                native_identifier TEXT NOT NULL,
                provenance TEXT NOT NULL,
                PRIMARY KEY (profile_id, legacy_identifier)
            ) STRICT;
        ),
        // Publication cannot be proven from this checkout, so the append-only
        // migration preserves any unexpected rows while disabling every superseded
        // store and moving legacy attempt envelopes out of the active table.
        sql!(
            ALTER TABLE sim_runtime_workspaces RENAME TO sim_runtime_workspace_quarantine;
            ALTER TABLE sim_runtime_workspace_quarantine
                ADD COLUMN quarantine_reason TEXT NOT NULL
                DEFAULT "superseded by workspace::SerializableItem and WorkspaceDb";

            ALTER TABLE sim_runtime_profiles RENAME TO sim_runtime_profile_quarantine;
            ALTER TABLE sim_runtime_profile_quarantine
                ADD COLUMN quarantine_reason TEXT NOT NULL
                DEFAULT "superseded by Zed SettingsStore runtime profiles";

            ALTER TABLE sim_runtime_mappings RENAME TO sim_runtime_mapping_quarantine;
            ALTER TABLE sim_runtime_mapping_quarantine
                ADD COLUMN quarantine_reason TEXT NOT NULL
                DEFAULT "superseded by owner-specific legacy mapping adapters";

            CREATE TABLE sim_runtime_attempt_quarantine (
                attempt_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                attempt_json TEXT NOT NULL,
                quarantine_reason TEXT NOT NULL
            ) STRICT;
            INSERT INTO sim_runtime_attempt_quarantine(
                attempt_id,
                profile_id,
                attempt_json,
                quarantine_reason
            )
            SELECT
                attempt_id,
                profile_id,
                attempt_json,
                "superseded unversioned PersistedAttempt envelope"
            FROM sim_runtime_attempts
            WHERE CASE
                WHEN json_valid(attempt_json)
                THEN json_type(attempt_json, "$.schema_version") IS NULL
                ELSE TRUE
            END;
            DELETE FROM sim_runtime_attempts
            WHERE attempt_id IN (
                SELECT attempt_id FROM sim_runtime_attempt_quarantine
            );
        ),
        sql!(
            ALTER TABLE sim_runtime_legacy_profiles
                RENAME TO sim_runtime_legacy_profile_quarantine;
            ALTER TABLE sim_runtime_legacy_profile_quarantine
                ADD COLUMN quarantine_reason TEXT NOT NULL
                DEFAULT "superseded redundant inactive-profile projection";
            CREATE TABLE sim_runtime_legacy_profiles (
                migration_id TEXT PRIMARY KEY,
                migration_result_json TEXT NOT NULL
            ) STRICT;

            INSERT INTO sim_runtime_attempt_quarantine(
                attempt_id,
                profile_id,
                attempt_json,
                quarantine_reason
            )
            SELECT
                attempt_id,
                profile_id,
                attempt_json,
                "predates canonical attempt-integrity validation"
            FROM sim_runtime_attempts;
            DELETE FROM sim_runtime_attempts;

            CREATE TRIGGER sim_runtime_attempt_profile_immutable
            BEFORE UPDATE OF profile_id ON sim_runtime_attempts
            WHEN OLD.profile_id <> NEW.profile_id
            BEGIN
                SELECT RAISE(ABORT, "execution attempt profile identity is immutable");
            END;

            CREATE TRIGGER sim_runtime_legacy_migration_immutable
            BEFORE UPDATE OF migration_result_json ON sim_runtime_legacy_profiles
            WHEN OLD.migration_result_json <> NEW.migration_result_json
            BEGIN
                SELECT RAISE(ABORT, "legacy migration identity is immutable");
            END;
        ),
        sql!(
            CREATE TABLE sim_runtime_execution_profiles (
                profile_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL
            ) STRICT;
            CREATE TRIGGER sim_runtime_execution_profile_identity_immutable
            BEFORE UPDATE OF profile_id ON sim_runtime_execution_profiles
            WHEN OLD.profile_id <> NEW.profile_id
            BEGIN
                SELECT RAISE(ABORT, "execution profile identity is immutable");
            END;
        ),
    ];
}

db::static_connection!(SimRuntimeDb, []);

impl SimRuntimeDb {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
    pub async fn replace_execution_profile(
        &self,
        profile: PersistedExecutionProfile,
        attempts: Vec<PersistedExecutionAttempt>,
    ) -> Result<()> {
        profile.validate()?;
        validate_profile_projection(profile.profile_id, &attempts)?;
        let profile_identity = profile.profile_id.0.to_string();
        let profile_json = serde_json::to_string(&profile)?;
        let mut rows = attempts
            .into_iter()
            .map(|attempt| {
                Ok((
                    attempt.record.attempt_id.0.to_string(),
                    profile_identity.clone(),
                    serde_json::to_string(&attempt)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        self.0
            .write(move |connection| {
                connection.with_savepoint("replace_sim_execution_profile_state", || {
                    let current_profile_json = connection
                        .select_row_bound::<String, String>(
                            "SELECT profile_json FROM sim_runtime_execution_profiles WHERE profile_id = ?",
                        )?(profile_identity.clone())?;
                    if let Some(current_profile_json) = current_profile_json {
                        let current: PersistedExecutionProfile =
                            serde_json::from_str(&current_profile_json)?;
                        current.validate()?;
                        anyhow::ensure!(
                            current.profile_id == profile.profile_id,
                            "persisted execution profile identity does not match its database key"
                        );
                        let existing = connection
                            .select_bound::<String, (String, String)>(
                                "SELECT attempt_id, attempt_json FROM sim_runtime_attempts WHERE profile_id = ? ORDER BY attempt_id ASC",
                            )?(profile_identity.clone())?;
                        let requested = rows
                            .iter()
                            .map(|(attempt_id, _, attempt_json)| {
                                (attempt_id.clone(), attempt_json.clone())
                            })
                            .collect::<Vec<_>>();
                        if current.revision > profile.revision {
                            anyhow::bail!(
                                "stale execution profile revision {} cannot replace revision {}",
                                profile.revision,
                                current.revision
                            );
                        }
                        if current.revision == profile.revision {
                            if current_profile_json == profile_json && existing == requested {
                                return Ok(());
                            }
                            let mut source_revision_projection = profile.clone();
                            source_revision_projection.source_revision = current.source_revision;
                            anyhow::ensure!(
                                profile.source_revision > current.source_revision
                                    && source_revision_projection == current
                                    && existing == requested,
                                "conflicting execution profile projection at revision {}",
                                profile.revision
                            );
                        }
                    }
                    let mut delete = connection.exec_bound(
                        "DELETE FROM sim_runtime_attempts WHERE profile_id = ?",
                    )?;
                    delete(profile_identity.clone())?;
                    let mut insert = connection.exec_bound(
                        "INSERT INTO sim_runtime_attempts(attempt_id, profile_id, attempt_json) VALUES (?, ?, ?)",
                    )?;
                    for row in rows {
                        insert(row)?;
                    }
                    let mut replace_profile = connection.exec_bound(
                        "INSERT INTO sim_runtime_execution_profiles(profile_id, profile_json) VALUES (?, ?) ON CONFLICT(profile_id) DO UPDATE SET profile_json = excluded.profile_json",
                    )?;
                    replace_profile((profile_identity, profile_json))?;
                    Ok(())
                })
            })
            .await
    }

    pub fn load_execution_profile(
        &self,
        profile_id: crate::ProfileId,
    ) -> Result<Option<PersistedExecutionProfile>> {
        self.load_execution_profile_json(&profile_id.0.to_string())?
            .map(|json| {
                let profile: PersistedExecutionProfile = serde_json::from_str(&json)?;
                profile.validate()?;
                anyhow::ensure!(
                    profile.profile_id == profile_id,
                    "persisted execution profile identity does not match its database key"
                );
                Ok(profile)
            })
            .transpose()
    }

    fn load_execution_attempt(
        &self,
        attempt_id: Uuid,
    ) -> Result<Option<PersistedExecutionAttempt>> {
        self.load_attempt_json(&attempt_id.to_string())?
            .map(|(stored_profile_id, json)| {
                let attempt: PersistedExecutionAttempt = serde_json::from_str(&json)?;
                attempt.validate()?;
                anyhow::ensure!(
                    attempt.record.attempt_id.0 == attempt_id,
                    "persisted execution attempt identity does not match its database key"
                );
                anyhow::ensure!(
                    attempt.record.profile_id.0.to_string() == stored_profile_id,
                    "persisted execution attempt profile does not match its database projection"
                );
                Ok(attempt)
            })
            .transpose()
    }

    pub fn load_execution_attempt_for_profile(
        &self,
        profile_id: crate::ProfileId,
        attempt_id: crate::AttemptId,
    ) -> Result<Option<PersistedExecutionAttempt>> {
        let attempt = self.load_execution_attempt(attempt_id.0)?;
        if let Some(attempt) = &attempt {
            anyhow::ensure!(
                attempt.record.profile_id == profile_id,
                "persisted execution attempt belongs to another profile"
            );
        }
        Ok(attempt)
    }

    pub fn load_execution_attempts_for_profile(
        &self,
        profile_id: crate::ProfileId,
    ) -> Result<Vec<PersistedExecutionAttempt>> {
        let mut attempts = self
            .load_execution_attempt_jsons_for_profile(&profile_id.0.to_string())?
            .into_iter()
            .map(|json| {
                (|| {
                    let value: serde_json::Value = serde_json::from_str(&json)?;
                    anyhow::ensure!(
                        value.get("schema_version").is_some(),
                        "active execution-attempt storage contains an unversioned envelope"
                    );
                    let attempt: PersistedExecutionAttempt = serde_json::from_value(value)?;
                    attempt.validate()?;
                    anyhow::ensure!(
                        attempt.record.profile_id == profile_id,
                        "persisted execution attempt belongs to another profile"
                    );
                    Ok(attempt)
                })()
            })
            .collect::<Result<Vec<_>>>()?;
        attempts.sort_by(|left, right| {
            left.record
                .created_at
                .cmp(&right.record.created_at)
                .then(left.record.attempt_id.0.cmp(&right.record.attempt_id.0))
        });
        validate_profile_projection(profile_id, &attempts)?;
        Ok(attempts)
    }

    pub async fn save_legacy_migration(&self, migration: &LegacyMigrationResult) -> Result<()> {
        migration.validate()?;
        self.save_legacy_migration_json(
            migration.migration_id.to_string(),
            serde_json::to_string(migration)?,
        )
        .await
    }

    pub fn load_legacy_migration(
        &self,
        migration_id: Uuid,
    ) -> Result<Option<LegacyMigrationResult>> {
        self.load_legacy_migration_json(&migration_id.to_string())?
            .map(|json| {
                let migration = crate::legacy_connections::decode_legacy_migration(&json)?;
                anyhow::ensure!(
                    migration.migration_id == migration_id,
                    "legacy migration identity does not match its database key"
                );
                Ok(migration)
            })
            .transpose()
    }

    query! {
        fn load_attempt_json(attempt_id: &str) -> Result<Option<(String, String)>> {
            SELECT profile_id, attempt_json
            FROM sim_runtime_attempts
            WHERE attempt_id = (?)
        }
    }

    query! {
        fn load_execution_attempt_jsons_for_profile(profile_id: &str) -> Result<Vec<String>> {
            SELECT attempt_json
            FROM sim_runtime_attempts
            WHERE profile_id = (?)
            ORDER BY attempt_id ASC
        }
    }

    query! {
        fn load_execution_profile_json(profile_id: &str) -> Result<Option<String>> {
            SELECT profile_json
            FROM sim_runtime_execution_profiles
            WHERE profile_id = (?)
        }
    }

    query! {
        async fn save_legacy_migration_json(
            migration_id: String,
            migration_result_json: String
        ) -> Result<()> {
            INSERT INTO sim_runtime_legacy_profiles(
                migration_id,
                migration_result_json
            ) VALUES ((?), (?))
            ON CONFLICT(migration_id) DO UPDATE SET
                migration_result_json = excluded.migration_result_json
        }
    }

    query! {
        fn load_legacy_migration_json(migration_id: &str) -> Result<Option<String>> {
            SELECT migration_result_json
            FROM sim_runtime_legacy_profiles
            WHERE migration_id = (?)
        }
    }
}

impl ExecutionAttemptPersistence for SimRuntimeDb {
    fn replace_execution_state(
        &self,
        profile: PersistedExecutionProfile,
        attempts: Vec<PersistedExecutionAttempt>,
    ) -> BoxFuture<'static, Result<()>> {
        let database = self.clone();
        async move { database.replace_execution_profile(profile, attempts).await }.boxed()
    }

    fn load_execution_state(
        &self,
        profile_id: crate::ProfileId,
    ) -> Result<(
        Option<PersistedExecutionProfile>,
        Vec<PersistedExecutionAttempt>,
    )> {
        Ok((
            self.load_execution_profile(profile_id)?,
            self.load_execution_attempts_for_profile(profile_id)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use db::sqlez::{connection::Connection, domain::Domain};
    use serde_json::json;
    use sha2::Sha256;
    use std::{
        error::Error,
        fs, io,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    use super::*;
    use crate::{AttemptEvent, AttemptEventKind, LegacyComfyProfile, migrate_legacy_profile};

    static PERSISTENCE_VALIDATION_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn native_model_resource_execution_persists_recipe_and_reconstruction_evidence()
    -> Result<(), Box<dyn Error>> {
        let profile_id = crate::ProfileId(Uuid::from_u128(0x402_100));
        let prompt_id = sim_types::PromptId(Uuid::from_u128(0x402_101));
        let prior_attempt_id = crate::AttemptId(Uuid::from_u128(0x402_102));
        let retry_attempt_id = crate::AttemptId(Uuid::from_u128(0x402_103));
        let node_id = NodeId("model-resource".to_owned());
        let request = NativeModelResourceRequest::checked(
            sim_model::NativeModelResourceRole::OpticalFlow,
            "optical_flow",
            vec!["raft.safetensors".to_owned()],
        )?;
        let preflight = NativeModelResourcePreflight::checked(
            &request,
            "a".repeat(64),
            vec!["b".repeat(64)],
            "c".repeat(64),
        )?;
        let resource_identity = NativeModelResourceIdentity::checked(
            sim_model::NativeModelResourceRole::OpticalFlow,
            "raft-large",
            "safetensors",
            "d".repeat(64),
            "e".repeat(64),
        )?;
        let evidence = PersistedNativeModelResourceEvidence::checked(
            node_id.clone(),
            preflight,
            resource_identity,
        )?;
        let at = Utc::now();
        let mut prior = AttemptRecord::queued(profile_id, prompt_id, prior_attempt_id);
        prior.apply(AttemptEvent {
            profile_id,
            prompt_id,
            attempt_id: prior_attempt_id,
            sequence: 0,
            node_id: None,
            at,
            kind: AttemptEventKind::Started,
            data: None,
        })?;
        let node_outputs = BTreeMap::from([(
            node_id.clone(),
            json!({NATIVE_MODEL_RESOURCE_EVIDENCE_UI_KEY: [evidence]}),
        )]);
        prior.apply(AttemptEvent {
            profile_id,
            prompt_id,
            attempt_id: prior_attempt_id,
            sequence: 1,
            node_id: None,
            at,
            kind: AttemptEventKind::Succeeded,
            data: Some(json!({"ui_outputs": node_outputs})),
        })?;
        let prior = PersistedExecutionAttempt::new(prior, None, ExecutionDataSource::Persisted)?;
        let mut retry = AttemptRecord::queued(profile_id, prompt_id, retry_attempt_id);
        retry.retry_of = Some(prior_attempt_id);
        let retry = PersistedExecutionAttempt::new(retry, None, ExecutionDataSource::Persisted)?;

        assert_eq!(
            native_model_resource_reconstruction_for_retry(&[prior, retry], retry_attempt_id,)?,
            BTreeMap::from([(node_id, vec![evidence])])
        );
        Ok(())
    }

    fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn write_persistence_validation_artifact(
        workspace_root: &Path,
        cases: &BTreeMap<&str, bool>,
        validation_id: &str,
        scope: &str,
        artifact_filename: &str,
    ) -> Result<(), Box<dyn Error>> {
        let artifact_directory = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("target"))
            .join("sim-parity");
        let artifact_path = artifact_directory.join(artifact_filename);
        match fs::remove_file(&artifact_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut fixture_digests = BTreeMap::new();
        for relative_path in [
            ".agents/specs/comfy-parity/catalogs/authoritative-ownership.csv",
            ".agents/specs/comfy-parity/ownership-policy.json",
            "crates/sim_runtime/src/persistence.rs",
            "crates/sim_runtime/src/legacy_connections.rs",
            "crates/sim_runtime/src/queue_history.rs",
            "crates/sim_runtime/src/prompt_compiler.rs",
            "crates/sim_ui/src/workflow_item.rs",
            "crates/workspace/src/item.rs",
        ] {
            fixture_digests.insert(
                relative_path,
                format!(
                    "{:x}",
                    Sha256::digest(fs::read(workspace_root.join(relative_path))?)
                ),
            );
        }
        fs::create_dir_all(&artifact_directory)?;
        let artifact = json!({
            "validation_id": validation_id,
            "scope": scope,
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "database": {
                    "migration_fixtures": "sqlite-in-memory",
                    "restart_round_trip": "sqlite-file-backed-close-and-reopen",
                },
                "development_oracle_executed": false,
            },
            "fixture_digests": fixture_digests,
            "summary": {
                "passed": cases.len(),
                "failed": 0,
                "skipped": 0,
            },
            "cases": cases,
            "skipped": [],
            "validation_closure": {
                "claimed": true,
                "scope": "versioned typed attempt persistence, inactive migration records, duplicate-store quarantine, unknown-field preservation, restart, and atomic rejection",
            },
            "release_closure_required": false,
        });
        let temporary_path = artifact_directory.join(format!("{artifact_filename}.tmp"));
        match fs::remove_file(&temporary_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        fs::write(&temporary_path, serde_json::to_vec_pretty(&artifact)?)?;
        fs::rename(temporary_path, artifact_path)?;
        Ok(())
    }

    fn collect_repository_rust_sources(
        directory: &Path,
        sources: &mut Vec<(PathBuf, String)>,
    ) -> Result<(), Box<dyn Error>> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                collect_repository_rust_sources(&path, sources)?;
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("._"))
            {
                let source = fs::read_to_string(&path)?;
                sources.push((path, source));
            }
        }
        Ok(())
    }

    fn repository_declaration_count(
        sources: &[(PathBuf, String)],
        path_prefix: Option<&Path>,
        keyword: &str,
        symbol: &str,
    ) -> usize {
        sources
            .iter()
            .filter(|(path, _)| path_prefix.is_none_or(|prefix| path.starts_with(prefix)))
            .flat_map(|(_, source)| source.lines())
            .filter(|line| {
                let line = line.trim_start();
                if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
                    return false;
                }
                let mut tokens = line.split_whitespace();
                while let Some(token) = tokens.next() {
                    if token == keyword {
                        return tokens.next().is_some_and(|candidate| {
                            candidate.strip_prefix(symbol).is_some_and(|suffix| {
                                suffix.chars().next().is_none_or(|character| {
                                    !character.is_alphanumeric() && character != '_'
                                })
                            })
                        });
                    }
                }
                false
            })
            .count()
    }

    fn normalized_prefix_count(sources: &[(PathBuf, String)], prefix: &str) -> usize {
        sources
            .iter()
            .flat_map(|(_, source)| source.lines())
            .filter(|line| {
                line.split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .starts_with(prefix)
            })
            .count()
    }

    fn table_count(connection: &Connection, table: &str) -> i64 {
        let mut select = connection
            .select_row_bound("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?")
            .expect("prepare schema query");
        select(table).expect("query schema").unwrap_or_default()
    }

    fn row_count(connection: &Connection, table: &str) -> i64 {
        let mut select = connection
            .select_row(&format!("SELECT COUNT(*) FROM {table}"))
            .expect("prepare row-count query");
        select().expect("query row count").unwrap_or_default()
    }

    async fn open_persistent_runtime_db(path: &Path) -> Result<SimRuntimeDb> {
        let uri = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("runtime database path is not UTF-8"))?;
        Ok(SimRuntimeDb(
            ThreadSafeConnection::builder::<SimRuntimeDb>(uri, true)
                .build()
                .await?,
        ))
    }

    fn execution_profile(profile_id: crate::ProfileId, revision: u64) -> PersistedExecutionProfile {
        PersistedExecutionProfile {
            schema_version: PERSISTED_EXECUTION_PROFILE_SCHEMA_VERSION,
            profile_id,
            revision,
            source_revision: None,
            source: ExecutionDataSource::Live,
            status: ExecutionSnapshotStatus::Ready,
            next_attempt: None,
            completed_requests: Vec::new(),
            recent_command_results: Vec::new(),
        }
    }

    #[test]
    fn migrations_quarantine_superseded_nonempty_stores() {
        let connection = Connection::open_memory(Some("sim_runtime_migrations"));
        connection
            .migrate(
                SimRuntimeDb::NAME,
                &SimRuntimeDb::MIGRATIONS[..1],
                &mut |_, _, _| false,
            )
            .expect("apply original migration");
        connection
            .exec(
                "INSERT INTO sim_runtime_profiles VALUES ('profile', '{}');
                 INSERT INTO sim_runtime_workspaces VALUES ('workspace', 'window', 'profile', '{}');
                 INSERT INTO sim_runtime_mappings VALUES ('profile', 'legacy', 'native', 'fixture');
                 INSERT INTO sim_runtime_legacy_profiles VALUES ('migration', '{}', '{}');
                 INSERT INTO sim_runtime_attempts VALUES ('legacy-attempt', 'profile', '{\"state\":\"legacy\"}');",
            )
            .expect("prepare legacy fixture")()
            .expect("insert legacy fixture");
        connection
            .migrate(
                SimRuntimeDb::NAME,
                SimRuntimeDb::MIGRATIONS,
                &mut |_, _, _| false,
            )
            .expect("apply consolidation migration");

        for table in [
            "sim_runtime_profiles",
            "sim_runtime_workspaces",
            "sim_runtime_mappings",
        ] {
            assert_eq!(table_count(&connection, table), 0, "active table {table}");
        }
        for table in [
            "sim_runtime_profile_quarantine",
            "sim_runtime_workspace_quarantine",
            "sim_runtime_mapping_quarantine",
            "sim_runtime_attempt_quarantine",
            "sim_runtime_legacy_profile_quarantine",
        ] {
            assert_eq!(table_count(&connection, table), 1, "missing table {table}");
            assert_eq!(row_count(&connection, table), 1, "lost row in {table}");
        }
        for table in [
            "sim_runtime_profiles",
            "sim_runtime_attempts",
            "sim_runtime_legacy_profiles",
        ] {
            let expected = i64::from(table != "sim_runtime_profiles");
            assert_eq!(table_count(&connection, table), expected, "table {table}");
        }
        assert_eq!(row_count(&connection, "sim_runtime_attempts"), 0);
    }

    #[test]
    fn attempts_and_inactive_legacy_migrations_round_trip() {
        smol::block_on(async {
            let database = SimRuntimeDb::open_test_db("sim_runtime_round_trip").await;
            let attempt_id = crate::AttemptId(Uuid::new_v4());
            let profile_id = crate::ProfileId(Uuid::new_v4());
            let mut record =
                AttemptRecord::queued(profile_id, crate::PromptId(Uuid::new_v4()), attempt_id);
            record.persistence_unknown_fields.insert(
                "future_attempt_state".into(),
                serde_json::json!({"preserved": true}),
            );
            let mut execution_attempt =
                PersistedExecutionAttempt::new(record, None, ExecutionDataSource::Persisted)
                    .expect("valid typed execution attempt");
            execution_attempt
                .unknown_fields
                .insert("future_receipt".into(), serde_json::json!({"sequence": 7}));
            database
                .replace_execution_profile(
                    execution_profile(profile_id, 1),
                    vec![execution_attempt.clone()],
                )
                .await
                .expect("save typed execution profile");
            assert_eq!(
                database
                    .load_execution_attempt(attempt_id.0)
                    .expect("load typed execution attempt"),
                Some(execution_attempt.clone())
            );

            let migration = migrate_legacy_profile(
                LegacyComfyProfile {
                    name: "Imported server".into(),
                    endpoint: Some("https://user:pass@example.invalid/private?token=hidden".into()),
                    credential: Some("must-not-persist".into()),
                    model_roots: vec!["models".into()],
                    api_host_enabled: false,
                    plugin_mappings: vec!["Legacy=Native".into()],
                    workflow_state: BTreeMap::new(),
                    unknown_fields: BTreeMap::from([(
                        "future_evidence".into(),
                        serde_json::json!({"preserved": true}),
                    )]),
                },
                Uuid::from_u128(31),
                Uuid::from_u128(32),
            )
            .expect("migrate legacy profile");
            database
                .save_legacy_migration(&migration)
                .await
                .expect("persist inactive migration");

            let shared_connection = database.clone();
            assert_eq!(
                shared_connection
                    .load_execution_attempt_for_profile(profile_id, attempt_id)
                    .expect("load attempt through shared connection"),
                Some(execution_attempt)
            );
            assert_eq!(
                shared_connection
                    .load_legacy_migration(migration.migration_id)
                    .expect("load migration through shared connection"),
                Some(migration)
            );
        });
    }

    #[test]
    fn full_profile_replacement_and_enumeration_are_profile_scoped() {
        smol::block_on(async {
            let database =
                SimRuntimeDb::open_test_db("sim_runtime_profile_attempt_enumeration").await;
            let first_profile = crate::ProfileId(Uuid::from_u128(1));
            let second_profile = crate::ProfileId(Uuid::from_u128(2));
            let first_attempt_id = crate::AttemptId(Uuid::from_u128(11));
            let second_attempt_id = crate::AttemptId(Uuid::from_u128(12));
            let first = PersistedExecutionAttempt::new(
                AttemptRecord::queued(
                    first_profile,
                    crate::PromptId(Uuid::from_u128(21)),
                    first_attempt_id,
                ),
                None,
                ExecutionDataSource::Recovery,
            )
            .expect("valid first attempt");
            let second = PersistedExecutionAttempt::new(
                AttemptRecord::queued(
                    second_profile,
                    crate::PromptId(Uuid::from_u128(22)),
                    second_attempt_id,
                ),
                None,
                ExecutionDataSource::Recovery,
            )
            .expect("valid second attempt");
            database
                .replace_execution_profile(execution_profile(first_profile, 1), vec![first.clone()])
                .await
                .expect("save first profile");
            database
                .replace_execution_profile(
                    execution_profile(second_profile, 1),
                    vec![second.clone()],
                )
                .await
                .expect("save second profile");
            let enumerated = database
                .load_execution_attempts_for_profile(first_profile)
                .expect("enumerate first profile");
            assert_eq!(enumerated.as_slice(), std::slice::from_ref(&first));
            assert!(
                database
                    .replace_execution_profile(
                        execution_profile(second_profile, 2),
                        vec![first.clone()],
                    )
                    .await
                    .is_err()
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(first_profile)
                    .expect("first attempt remains")
                    .as_slice(),
                std::slice::from_ref(&first)
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(second_profile)
                    .expect("rejected cross-profile replacement preserves second profile"),
                [second]
            );
            database
                .replace_execution_profile(execution_profile(second_profile, 2), Vec::new())
                .await
                .expect("replace second profile with an empty projection");
            assert!(
                database
                    .load_execution_attempts_for_profile(second_profile)
                    .expect("enumerate empty second profile")
                    .is_empty()
            );
        });
    }

    #[test]
    fn profile_replacement_is_atomic_and_preserves_other_profiles() {
        smol::block_on(async {
            let database = SimRuntimeDb::open_test_db("sim_runtime_atomic_profile_replace").await;
            let first_profile = crate::ProfileId(Uuid::from_u128(1));
            let second_profile = crate::ProfileId(Uuid::from_u128(2));
            let attempt = |profile_id, prompt, attempt| {
                PersistedExecutionAttempt::new(
                    AttemptRecord::queued(
                        profile_id,
                        crate::PromptId(Uuid::from_u128(prompt)),
                        crate::AttemptId(Uuid::from_u128(attempt)),
                    ),
                    None,
                    ExecutionDataSource::Recovery,
                )
                .expect("valid replacement fixture")
            };
            let original = vec![
                attempt(first_profile, 21, 11),
                attempt(first_profile, 22, 12),
            ];
            database
                .replace_execution_profile(execution_profile(first_profile, 1), original.clone())
                .await
                .expect("persist original profile projection");
            let other = attempt(second_profile, 23, 13);
            database
                .replace_execution_profile(
                    execution_profile(second_profile, 1),
                    vec![other.clone()],
                )
                .await
                .expect("persist other profile projection");

            let replacement = attempt(first_profile, 24, 14);
            database
                .replace_execution_profile(
                    execution_profile(first_profile, 2),
                    vec![replacement.clone()],
                )
                .await
                .expect("replace profile projection");
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(first_profile)
                    .expect("load replaced profile"),
                vec![replacement.clone()]
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(second_profile)
                    .expect("load untouched profile"),
                vec![other.clone()]
            );

            let colliding = attempt(first_profile, 25, 13);
            assert!(
                database
                    .replace_execution_profile(
                        execution_profile(first_profile, 3),
                        vec![colliding],
                    )
                    .await
                    .is_err()
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(first_profile)
                    .expect("failed replacement preserves prior profile"),
                vec![replacement]
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(second_profile)
                    .expect("failed replacement preserves other profile"),
                vec![other]
            );
        });
    }

    #[test]
    fn execution_profile_revision_rejects_stale_projection_replacement() {
        smol::block_on(async {
            let database = SimRuntimeDb::open_test_db("sim_runtime_execution_profile_cas").await;
            let profile_id = crate::ProfileId(Uuid::from_u128(0x81));
            let mut profile = PersistedExecutionProfile {
                schema_version: PERSISTED_EXECUTION_PROFILE_SCHEMA_VERSION,
                profile_id,
                revision: 2,
                source_revision: None,
                source: ExecutionDataSource::Live,
                status: ExecutionSnapshotStatus::Ready,
                next_attempt: Some(0x90_u128.to_string()),
                completed_requests: Vec::new(),
                recent_command_results: Vec::new(),
            };
            database
                .replace_execution_profile(profile.clone(), Vec::new())
                .await
                .expect("persist initial execution profile");
            let attempt = PersistedExecutionAttempt::new(
                AttemptRecord::queued(
                    profile_id,
                    crate::PromptId(Uuid::from_u128(0x82)),
                    crate::AttemptId(Uuid::from_u128(0x83)),
                ),
                None,
                ExecutionDataSource::Live,
            )
            .expect("valid stale projection fixture");
            assert!(
                database
                    .replace_execution_profile(profile.clone(), vec![attempt.clone()])
                    .await
                    .is_err()
            );
            assert!(
                database
                    .load_execution_attempts_for_profile(profile_id)
                    .expect("load preserved execution projection")
                    .is_empty()
            );

            profile.revision = 3;
            database
                .replace_execution_profile(profile.clone(), vec![attempt.clone()])
                .await
                .expect("newer revision replaces execution projection");
            database
                .replace_execution_profile(profile.clone(), vec![attempt.clone()])
                .await
                .expect("identical revision and projection are idempotent");
            profile.source_revision = Some(7);
            database
                .replace_execution_profile(profile.clone(), vec![attempt.clone()])
                .await
                .expect("monotonic source-only reconciliation is durable");
            let mut conflicting = profile.clone();
            conflicting.status = ExecutionSnapshotStatus::Unavailable {
                failure: crate::ExecutionFailure::new(
                    "conflicting_projection",
                    "same presentation revision cannot change durable state",
                ),
            };
            assert!(
                database
                    .replace_execution_profile(conflicting, vec![attempt.clone()])
                    .await
                    .is_err()
            );
            assert_eq!(
                database
                    .load_execution_profile(profile_id)
                    .expect("load execution profile"),
                Some(profile)
            );
            assert_eq!(
                database
                    .load_execution_attempts_for_profile(profile_id)
                    .expect("load current execution projection"),
                vec![attempt]
            );
        });
    }

    #[test]
    fn persisted_provider_plan_rejects_execution_fact_tampering() -> Result<(), Box<dyn Error>> {
        let descriptor = crate::NativeNodeDescriptor {
            schema_version: sim_nodes::LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "PersistedProvider".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: None,
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: vec![crate::NativeOutputDescriptor {
                name: "value".to_owned(),
                produced_type: crate::NativeValueType::Primitive(
                    crate::NativePrimitiveType::Number,
                ),
                is_list: false,
            }],
            output_node: true,
            effect: crate::NativeEffectClass::Provider,
            cache: crate::NativeCachePolicy::Never,
        };
        let mut registry = crate::NativeNodeRegistry::default();
        registry.register_descriptor(descriptor)?;
        let prompt_id = crate::PromptId(Uuid::from_u128(0x371));
        let submission = sim_types::PromptSubmission {
            prompt: sim_types::ApiPrompt(BTreeMap::from([(
                sim_types::NodeId::from("provider"),
                sim_types::PromptNode {
                    class_type: "PersistedProvider".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )])),
            prompt_id: Some(prompt_id),
            client_id: None,
            number: None,
            extra_data: BTreeMap::from([("publisher_note".to_owned(), json!("comfy:image@1"))]),
            unknown: BTreeMap::from([(
                "prior_provider".to_owned(),
                json!("zed.comfy.provider.openrouter"),
            )]),
        };
        let pin =
            crate::NativeProviderRegistryPin::checked(4, "a".repeat(64), vec!["b".repeat(64)])?;
        let plan = crate::PromptCompiler::new(&registry)
            .with_provider_registry_pin(pin)?
            .compile(submission)?;
        let record = AttemptRecord::queued(
            crate::ProfileId(Uuid::from_u128(0x372)),
            prompt_id,
            crate::AttemptId(Uuid::from_u128(0x373)),
        );
        PersistedExecutionAttempt::new(
            record.clone(),
            Some(plan.clone()),
            ExecutionDataSource::Persisted,
        )?;

        let encoded = serde_json::to_vec(&plan)?;
        let restored: CompiledPlan = serde_json::from_slice(&encoded)?;
        restored.validate_integrity()?;
        assert_eq!(restored, plan);
        assert_eq!(serde_json::to_vec(&restored)?, encoded);
        assert_eq!(
            restored.provider_registry_pin(),
            plan.provider_registry_pin()
        );
        assert_eq!(
            restored.extra_data.get("publisher_note"),
            Some(&json!("comfy:image@1"))
        );
        assert_eq!(
            restored.unknown.get("prior_provider"),
            Some(&json!("zed.comfy.provider.openrouter"))
        );

        let mut tampered = plan;
        tampered.unknown.insert("tampered".to_owned(), json!(true));
        assert!(
            PersistedExecutionAttempt::new(record, Some(tampered), ExecutionDataSource::Persisted,)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn invalid_attempt_update_preserves_the_last_valid_record() {
        smol::block_on(async {
            let database = SimRuntimeDb::open_test_db("sim_runtime_atomic_attempt").await;
            let profile_id = crate::ProfileId(Uuid::from_u128(41));
            let attempt_id = crate::AttemptId(Uuid::from_u128(42));
            let valid = PersistedExecutionAttempt::new(
                AttemptRecord::queued(profile_id, crate::PromptId(Uuid::from_u128(43)), attempt_id),
                None,
                ExecutionDataSource::Live,
            )
            .expect("valid attempt");
            database
                .replace_execution_profile(execution_profile(profile_id, 1), vec![valid.clone()])
                .await
                .expect("persist valid profile");

            let mut invalid = valid.clone();
            invalid.schema_version = u16::MAX;
            assert!(
                database
                    .replace_execution_profile(execution_profile(profile_id, 2), vec![invalid])
                    .await
                    .is_err()
            );
            assert_eq!(
                database
                    .load_execution_attempt(attempt_id.0)
                    .expect("load preserved attempt"),
                Some(valid)
            );
        });
    }

    #[test]
    fn cross_profile_collision_and_malformed_state_preserve_the_valid_record() {
        smol::block_on(async {
            let database = SimRuntimeDb::open_test_db("sim_runtime_reject_invalid_overwrite").await;
            let profile_id = crate::ProfileId(Uuid::from_u128(51));
            let other_profile_id = crate::ProfileId(Uuid::from_u128(52));
            let attempt_id = crate::AttemptId(Uuid::from_u128(53));
            let valid = PersistedExecutionAttempt::new(
                AttemptRecord::queued(profile_id, crate::PromptId(Uuid::from_u128(54)), attempt_id),
                None,
                ExecutionDataSource::Live,
            )
            .expect("valid attempt");
            database
                .replace_execution_profile(execution_profile(profile_id, 1), vec![valid.clone()])
                .await
                .expect("persist valid profile");

            let conflicting = PersistedExecutionAttempt::new(
                AttemptRecord::queued(
                    other_profile_id,
                    crate::PromptId(Uuid::from_u128(55)),
                    attempt_id,
                ),
                None,
                ExecutionDataSource::Live,
            )
            .expect("individually valid conflicting attempt");
            assert!(
                database
                    .replace_execution_profile(
                        execution_profile(other_profile_id, 1),
                        vec![conflicting],
                    )
                    .await
                    .is_err()
            );

            let mut malformed = valid.clone();
            malformed.record.state = crate::AttemptState::Running;
            assert!(
                database
                    .replace_execution_profile(execution_profile(profile_id, 2), vec![malformed],)
                    .await
                    .is_err()
            );
            assert_eq!(
                database
                    .load_execution_attempt(attempt_id.0)
                    .expect("load preserved attempt"),
                Some(valid)
            );
        });
    }

    fn run_persistence_validation(
        validation_id: &str,
        scope: &str,
        artifact_filename: &str,
    ) -> Result<(), Box<dyn Error>> {
        let _validation_guard = PERSISTENCE_VALIDATION_LOCK
            .lock()
            .map_err(|_| io::Error::other("persistence validation lock is poisoned"))?;
        let workspace_root = workspace_root()?;
        let mut cases = BTreeMap::new();

        let migration_connection =
            Connection::open_memory(Some("val_runtime_persistence_001_migrations"));
        migration_connection.migrate(
            SimRuntimeDb::NAME,
            &SimRuntimeDb::MIGRATIONS[..1],
            &mut |_, _, _| false,
        )?;
        migration_connection.exec(
            "INSERT INTO sim_runtime_profiles VALUES ('profile', '{\"future\":true}');
             INSERT INTO sim_runtime_workspaces VALUES ('workspace', 'window', 'profile', '{\"future\":true}');
             INSERT INTO sim_runtime_mappings VALUES ('profile', 'legacy', 'native', 'fixture');
             INSERT INTO sim_runtime_legacy_profiles VALUES ('migration', '{\"future\":true}', '{\"future\":true}');
             INSERT INTO sim_runtime_attempts VALUES ('legacy-attempt', 'profile', '{\"state\":\"legacy\"}');",
        )?()?;
        migration_connection.migrate(
            SimRuntimeDb::NAME,
            SimRuntimeDb::MIGRATIONS,
            &mut |_, _, _| false,
        )?;
        cases.insert(
            "superseded_active_stores_are_absent",
            [
                "sim_runtime_profiles",
                "sim_runtime_workspaces",
                "sim_runtime_mappings",
            ]
            .into_iter()
            .all(|table| table_count(&migration_connection, table) == 0),
        );
        cases.insert(
            "unexpected_legacy_rows_are_quarantined_losslessly",
            [
                "sim_runtime_profile_quarantine",
                "sim_runtime_workspace_quarantine",
                "sim_runtime_mapping_quarantine",
                "sim_runtime_attempt_quarantine",
                "sim_runtime_legacy_profile_quarantine",
            ]
            .into_iter()
            .all(|table| {
                table_count(&migration_connection, table) == 1
                    && row_count(&migration_connection, table) == 1
            }),
        );
        cases.insert(
            "active_attempt_store_excludes_unversioned_envelopes",
            row_count(&migration_connection, "sim_runtime_attempts") == 0,
        );
        cases.insert("migration_prefix_is_recorded_once", {
            let mut count = migration_connection
                .select_row_bound::<_, i64>("SELECT COUNT(*) FROM migrations WHERE domain=?")?;
            count(SimRuntimeDb::NAME)?.unwrap_or_default()
                == i64::try_from(SimRuntimeDb::MIGRATIONS.len())?
        });
        cases.insert("every_migration_prefix_reaches_current_schema", {
            (0..SimRuntimeDb::MIGRATIONS.len()).all(|prefix| {
                let connection = Connection::open_memory(Some(&format!(
                    "val_runtime_persistence_001_prefix_{prefix}"
                )));
                connection
                    .migrate(
                        SimRuntimeDb::NAME,
                        &SimRuntimeDb::MIGRATIONS[..prefix],
                        &mut |_, _, _| false,
                    )
                    .and_then(|()| {
                        connection.migrate(
                            SimRuntimeDb::NAME,
                            SimRuntimeDb::MIGRATIONS,
                            &mut |_, _, _| false,
                        )
                    })
                    .is_ok()
                    && table_count(&connection, "sim_runtime_attempts") == 1
                    && table_count(&connection, "sim_runtime_profiles") == 0
                    && table_count(&connection, "sim_runtime_workspaces") == 0
                    && table_count(&connection, "sim_runtime_mappings") == 0
            })
        });

        let versioned_connection =
            Connection::open_memory(Some("val_runtime_persistence_001_versioned_quarantine"));
        versioned_connection.migrate(
            SimRuntimeDb::NAME,
            &SimRuntimeDb::MIGRATIONS[..2],
            &mut |_, _, _| false,
        )?;
        versioned_connection.exec(
            "INSERT INTO sim_runtime_attempts VALUES ('future-attempt', 'profile', '{\"schema_version\":999}');
             INSERT INTO sim_runtime_attempts VALUES ('malformed-v2-attempt', 'profile', '{\"schema_version\":2}');",
        )?()?;
        versioned_connection.migrate(
            SimRuntimeDb::NAME,
            SimRuntimeDb::MIGRATIONS,
            &mut |_, _, _| false,
        )?;
        cases.insert(
            "pre_validation_versioned_rows_are_quarantined",
            row_count(&versioned_connection, "sim_runtime_attempts") == 0
                && row_count(&versioned_connection, "sim_runtime_attempt_quarantine") == 2,
        );

        smol::block_on(async {
            let temporary_directory = tempfile::tempdir()?;
            let database_path = temporary_directory.path().join("runtime.sqlite");
            let database = open_persistent_runtime_db(&database_path).await?;
            let profile_id = crate::ProfileId(Uuid::from_u128(4_001));
            let other_profile_id = crate::ProfileId(Uuid::from_u128(4_002));
            let prompt_id = crate::PromptId(Uuid::from_u128(4_003));
            let attempt_id = crate::AttemptId(Uuid::from_u128(4_004));
            let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
            record
                .persistence_unknown_fields
                .insert("future_attempt_state".into(), json!({"nested": [1, 2, 3]}));
            let mut plan = CompiledPlan {
                prompt_id,
                client_id: Some("validation-client".into()),
                prompt_number: Some(1.0),
                extra_data: BTreeMap::new(),
                unknown: BTreeMap::new(),
                nodes: BTreeMap::new(),
                topological_order: Vec::new(),
                static_required_nodes: Default::default(),
                output_nodes: Vec::new(),
                provider_execution: None,
                persistence_unknown_fields: BTreeMap::from([(
                    "future_plan_state".into(),
                    json!({"preserved": true}),
                )]),
            };
            plan.unknown
                .insert("source_unknown".into(), json!("preserved"));
            let mut attempt =
                PersistedExecutionAttempt::new(record, Some(plan), ExecutionDataSource::Persisted)?;
            attempt
                .unknown_fields
                .insert("future_envelope_state".into(), json!({"preserved": true}));
            database
                .replace_execution_profile(execution_profile(profile_id, 1), vec![attempt.clone()])
                .await?;
            drop(database);
            let restarted = open_persistent_runtime_db(&database_path).await?;
            let loaded = restarted
                .load_execution_attempt_for_profile(profile_id, attempt_id)?
                .ok_or_else(|| anyhow::anyhow!("typed attempt disappeared after restart"))?;
            cases.insert("typed_attempt_restarts_exactly", loaded == attempt);
            cases.insert(
                "nested_unknown_attempt_and_plan_fields_survive",
                loaded
                    .record
                    .persistence_unknown_fields
                    .contains_key("future_attempt_state")
                    && loaded.plan.as_ref().is_some_and(|plan| {
                        plan.persistence_unknown_fields
                            .contains_key("future_plan_state")
                    })
                    && loaded.unknown_fields.contains_key("future_envelope_state"),
            );
            cases.insert(
                "profile_scope_is_enforced_on_read",
                restarted
                    .load_execution_attempt_for_profile(other_profile_id, attempt_id)
                    .is_err(),
            );
            cases.insert(
                "cross_profile_projection_is_non_destructive",
                restarted
                    .replace_execution_profile(
                        execution_profile(other_profile_id, 1),
                        vec![attempt.clone()],
                    )
                    .await
                    .is_err()
                    && restarted.load_execution_attempt(attempt_id.0)?.as_ref() == Some(&attempt),
            );

            let mut invalid = attempt.clone();
            invalid.schema_version = u16::MAX;
            cases.insert(
                "invalid_update_is_rejected_before_overwrite",
                restarted
                    .replace_execution_profile(execution_profile(profile_id, 2), vec![invalid])
                    .await
                    .is_err()
                    && restarted.load_execution_attempt(attempt_id.0)?.as_ref() == Some(&attempt),
            );

            let conflicting = PersistedExecutionAttempt::new(
                AttemptRecord::queued(
                    other_profile_id,
                    crate::PromptId(Uuid::from_u128(4_007)),
                    attempt_id,
                ),
                None,
                ExecutionDataSource::Live,
            )?;
            cases.insert(
                "cross_profile_attempt_collision_is_rejected",
                restarted
                    .replace_execution_profile(
                        execution_profile(other_profile_id, 1),
                        vec![conflicting],
                    )
                    .await
                    .is_err()
                    && restarted.load_execution_attempt(attempt_id.0)?.as_ref() == Some(&attempt),
            );

            let mut malformed = attempt.clone();
            malformed.record.state = crate::AttemptState::Running;
            cases.insert(
                "malformed_state_is_rejected_before_overwrite",
                restarted
                    .replace_execution_profile(execution_profile(profile_id, 2), vec![malformed])
                    .await
                    .is_err()
                    && restarted.load_execution_attempt(attempt_id.0)?.as_ref() == Some(&attempt),
            );

            let migration = migrate_legacy_profile(
                LegacyComfyProfile {
                    name: "Imported server".into(),
                    endpoint: Some("https://user:pass@example.invalid/private?token=hidden".into()),
                    credential: Some("must-not-persist".into()),
                    model_roots: vec!["models".into()],
                    api_host_enabled: false,
                    plugin_mappings: vec!["Legacy=Native".into()],
                    workflow_state: BTreeMap::new(),
                    unknown_fields: BTreeMap::from([
                        ("future_evidence".into(), json!({"preserved": true})),
                        (
                            "nested".into(),
                            json!({"access_token": "hidden", "kept": {"password": "hidden", "value": 1}}),
                        ),
                    ]),
                },
                Uuid::from_u128(4_005),
                Uuid::from_u128(4_006),
            )?;
            restarted.save_legacy_migration(&migration).await?;
            let reloaded_migration = restarted
                .load_legacy_migration(migration.migration_id)?
                .ok_or_else(|| anyhow::anyhow!("legacy migration disappeared after restart"))?;
            let serialized_migration = serde_json::to_string(&reloaded_migration)?;
            cases.insert(
                "legacy_migration_identity_is_deterministic",
                reloaded_migration.migration_id == Uuid::from_u128(4_005)
                    && reloaded_migration.native_profile.id == Uuid::from_u128(4_006),
            );
            cases.insert(
                "legacy_credentials_never_reconnect_or_persist",
                reloaded_migration.credential_removed
                    && !reloaded_migration.inactive_legacy_profile.active
                    && !serialized_migration.contains("must-not-persist")
                    && !serialized_migration.contains("hidden")
                    && !serialized_migration.contains("access_token")
                    && !serialized_migration.contains("password"),
            );
            cases.insert(
                "legacy_endpoint_is_reduced_to_a_nonsecret_origin",
                reloaded_migration
                    .inactive_legacy_profile
                    .former_endpoint
                    .as_ref()
                    .map(crate::InactiveLegacyOrigin::display)
                    == Some("https://example.invalid")
                    && !serialized_migration.contains("user:pass")
                    && !serialized_migration.contains("token=hidden"),
            );
            cases.insert(
                "legacy_replacement_stays_disabled_pending_owner_review",
                reloaded_migration.native_profile.model_roots.is_empty()
                    && !reloaded_migration.native_profile.api_host.enabled
                    && reloaded_migration
                        .presentation()
                        .conversion_steps
                        .iter()
                        .all(|step| step.requires_explicit_acceptance),
            );
            cases.insert(
                "nonsecret_legacy_unknowns_survive",
                reloaded_migration
                    .inactive_legacy_profile
                    .unknown_fields
                    .get("future_evidence")
                    == Some(&json!({"preserved": true}))
                    && reloaded_migration
                        .inactive_legacy_profile
                        .unknown_fields
                        .get("nested")
                        == Some(&json!({"kept": {"value": 1}})),
            );

            let mut conflicting_migration = migration.clone();
            conflicting_migration.native_profile.id = Uuid::from_u128(4_008);
            cases.insert(
                "legacy_migration_collision_is_rejected_before_overwrite",
                restarted
                    .save_legacy_migration(&conflicting_migration)
                    .await
                    .is_err()
                    && restarted
                        .load_legacy_migration(migration.migration_id)?
                        .as_ref()
                        == Some(&migration),
            );

            let corrupt_attempt_id = crate::AttemptId(Uuid::from_u128(4_009));
            let corrupt_attempt = PersistedExecutionAttempt::new(
                AttemptRecord::queued(profile_id, prompt_id, corrupt_attempt_id),
                None,
                ExecutionDataSource::Persisted,
            )?;
            let corrupt_json = serde_json::to_string(&corrupt_attempt)?;
            let corrupt_profile = other_profile_id.0.to_string();
            restarted
                .write(move |connection| {
                    let statement = format!(
                        "INSERT INTO sim_runtime_attempts(attempt_id, profile_id, attempt_json) VALUES ('{}', '{}', '{}')",
                        corrupt_attempt_id.0,
                        corrupt_profile,
                        corrupt_json.replace('\'', "''")
                    );
                    connection.exec(&statement).and_then(|mut execute| execute())
                })
                .await?;
            cases.insert(
                "database_profile_projection_mismatch_is_rejected",
                restarted
                    .load_execution_attempt(corrupt_attempt_id.0)
                    .is_err(),
            );
            Ok::<(), anyhow::Error>(())
        })?;

        let mut repository_sources = Vec::new();
        collect_repository_rust_sources(&workspace_root.join("crates"), &mut repository_sources)?;
        let runtime_path = workspace_root.join("crates/sim_runtime");
        cases.insert(
            "no_second_runtime_workspace_state_type",
            repository_declaration_count(
                &repository_sources,
                None,
                "struct",
                "PersistedWorkspaceState",
            ) == 0
                && repository_declaration_count(
                    &repository_sources,
                    None,
                    "struct",
                    "ComfyWorkspaceRecord",
                ) == 0,
        );
        cases.insert(
            "runtime_profiles_are_not_written_to_a_second_database",
            repository_declaration_count(
                &repository_sources,
                Some(&runtime_path),
                "fn",
                "save_profile",
            ) == 0
                && repository_declaration_count(
                    &repository_sources,
                    Some(&runtime_path),
                    "fn",
                    "load_profile",
                ) == 0,
        );
        cases.insert(
            "repository_has_one_workspace_and_runtime_database_owner",
            repository_declaration_count(&repository_sources, None, "struct", "WorkspaceDb") == 1
                && repository_declaration_count(
                    &repository_sources,
                    None,
                    "struct",
                    "SimRuntimeDb",
                ) == 1
                && repository_declaration_count(
                    &repository_sources,
                    None,
                    "struct",
                    "SimWorkflowDb",
                ) == 1
                && normalized_prefix_count(
                    &repository_sources,
                    "impl SerializableItem for GraphWorkspaceItem",
                ) == 1
                && normalized_prefix_count(
                    &repository_sources,
                    "db::static_connection!(SimWorkflowDb, [WorkspaceDb]);",
                ) == 1,
        );
        let persistence_source =
            fs::read_to_string(workspace_root.join("crates/sim_runtime/src/persistence.rs"))?;
        let production_persistence_source = persistence_source
            .split_once("#[cfg(test)]")
            .map_or(persistence_source.as_str(), |(production, _)| production);
        cases.insert(
            "active_legacy_migration_store_has_one_canonical_payload",
            production_persistence_source.contains(
                "CREATE TABLE sim_runtime_legacy_profiles (\n                migration_id TEXT PRIMARY KEY,\n                migration_result_json TEXT NOT NULL",
            ) && !production_persistence_source.contains(
                "INSERT INTO sim_runtime_legacy_profiles(\n                migration_id,\n                inactive_profile_json",
            ),
        );

        if cases.values().any(|passed| !passed) {
            return Err(io::Error::other(format!(
                "VAL-RUNTIME-PERSISTENCE-001 cases failed: {cases:?}"
            ))
            .into());
        }
        write_persistence_validation_artifact(
            &workspace_root,
            &cases,
            validation_id,
            scope,
            artifact_filename,
        )
    }

    #[test]
    fn val_domain_001() -> Result<(), Box<dyn Error>> {
        run_persistence_validation(
            "VAL-DOMAIN-001",
            "persistence-schemas-migrations-restart-and-atomic-failure",
            "val-domain-001.json",
        )
    }

    #[test]
    fn val_runtime_persistence_001() -> Result<(), Box<dyn Error>> {
        run_persistence_validation(
            "VAL-RUNTIME-PERSISTENCE-001",
            "native-runtime-operational-persistence-foundation",
            "val-runtime-persistence-001.json",
        )
    }
}
