use crate::http::{
    CapabilityState, HttpBody, HttpCapabilities, MutationIdentity, NativeHttpServices,
    NativeMutationReconciliation, NativeRequestAuthority, NativeServiceError,
    NativeServiceErrorKind, NativeServiceOperation, NativeServiceRequest, NativeServiceResponse,
    http_route_catalog,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sim_runtime::{
    AssetAvailability, AssetNamespace, AssetQuery, AssetRecord, AttemptPresentation, AttemptState,
    AuthorizedCapabilities, CompiledPlan, ExecutionCommandAck, ExecutionCommandOutcome,
    ExecutionCommandReceiptState, ExecutionControlCommand, ExecutionControlCommandKind,
    ExecutionController, ExecutionFailureOrigin, ExecutionSnapshot, ExecutionSnapshotStatus,
    InputBinding, InputMode, NativeExecutionRegistryBundle, NativeInputRequirement,
    NativeInputSchemaMetadata, NativeNodeBindingDisposition, NativeNodeRegistry, NativePrimitive,
    NativeProviderRegistryPin, NativeSchemaProvenance, NativeSchemaValue, NativeUploadKind,
    NativeValue, NodeRegistry, ObjectInfoNode, ObjectInfoRegistry, ProfileId, PromptCompiler,
    PromptId, RECENT_COMMAND_RESULT_CAPACITY, RequestId, RuntimeNodeDescriptor,
    RuntimeNodePresentation, SharedAssetService, SharedExecutionPresentationService,
    generated_native_node_registry_projection, native_image_catalog_bindings,
};
use sim_types::{HttpMethod, PromptSubmission};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::Component,
    sync::{Arc, Mutex, MutexGuard},
};

const MAXIMUM_JOB_PAGE_SIZE: usize = 1_000;
const MAXIMUM_HISTORY_PAGE_SIZE: usize = 4_096;
const MAXIMUM_ASSET_SCAN_RESULTS: usize = 100_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProviderDeploymentIdentity {
    profile_id: String,
    component_inventory_candidate_sha256: String,
    component_generation: u64,
    signed_component_snapshot_sha256: String,
    provider_binding_set_sha256: Option<String>,
    registry_digest_sha256: String,
    provider_registry_identity_sha256: Option<String>,
    execution_bundle_identity_sha256: String,
}

impl NativeProviderDeploymentIdentity {
    pub fn from_registry_bundle(
        bundle: &NativeExecutionRegistryBundle,
        candidate_identity: &extension_host::ComponentInventoryCandidateIdentity,
    ) -> Result<Self, crate::NativeApiHostError> {
        let components = bundle.worker_deployment().begin().components();
        let candidate_descriptors = candidate_identity.descriptors();
        if components.len() != candidate_descriptors.len()
            || components
                .iter()
                .zip(candidate_descriptors)
                .any(|(component, candidate)| {
                    component.extension_id() != candidate.extension_id()
                        || component.extension_version() != candidate.extension_version()
                        || component.manifest_digest_sha256().as_str()
                            != candidate.manifest_sha256()
                        || component.component_digest_sha256().as_str()
                            != candidate.component_sha256()
                })
        {
            return Err(crate::NativeApiHostError::InvalidConfiguration(
                "component inventory candidate does not match the signed registry deployment"
                    .into(),
            ));
        }
        let mut snapshot = Sha256::new();
        snapshot.update(b"zed.native-provider-signed-component-snapshot.v1\0");
        snapshot.update(bundle.profile_id().0.as_bytes());
        snapshot.update(
            bundle
                .worker_deployment()
                .begin()
                .generation()
                .get()
                .to_le_bytes(),
        );
        for component in bundle.worker_deployment().begin().components() {
            for field in [
                component.extension_id().as_bytes(),
                component.extension_version().as_bytes(),
                component.plugin_identifier().as_bytes(),
                component.plugin_version().as_bytes(),
                component.manifest_digest_sha256().as_str().as_bytes(),
                component.component_digest_sha256().as_str().as_bytes(),
            ] {
                snapshot.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
                snapshot.update(field);
            }
        }
        let provider_binding_set_sha256 = bundle.provider_registry().map(|registry| {
            let mut digest = Sha256::new();
            digest.update(b"zed.native-provider-binding-set.v1\0");
            for binding in registry.binding_digests_sha256() {
                digest.update(
                    u64::try_from(binding.len())
                        .unwrap_or(u64::MAX)
                        .to_le_bytes(),
                );
                digest.update(binding.as_bytes());
            }
            format!("{:x}", digest.finalize())
        });
        Ok(Self {
            profile_id: bundle.profile_id().0.to_string(),
            component_inventory_candidate_sha256: candidate_identity.as_sha256().to_owned(),
            component_generation: bundle.worker_deployment().begin().generation().get(),
            signed_component_snapshot_sha256: format!("{:x}", snapshot.finalize()),
            provider_binding_set_sha256,
            registry_digest_sha256: bundle
                .worker_deployment()
                .begin()
                .registry_digest_sha256()
                .as_str()
                .to_owned(),
            provider_registry_identity_sha256: bundle
                .provider_registry()
                .map(NativeProviderRegistryPin::identity_sha256),
            execution_bundle_identity_sha256: bundle.identity_sha256().to_owned(),
        })
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn component_generation(&self) -> u64 {
        self.component_generation
    }

    pub fn component_inventory_candidate_sha256(&self) -> &str {
        &self.component_inventory_candidate_sha256
    }

    pub fn signed_component_snapshot_sha256(&self) -> &str {
        &self.signed_component_snapshot_sha256
    }

    pub fn provider_binding_set_sha256(&self) -> Option<&str> {
        self.provider_binding_set_sha256.as_deref()
    }

    pub fn same_signed_deployment(&self, other: &Self) -> bool {
        self.profile_id == other.profile_id
            && self.component_inventory_candidate_sha256
                == other.component_inventory_candidate_sha256
            && self.component_generation == other.component_generation
            && self.signed_component_snapshot_sha256 == other.signed_component_snapshot_sha256
            && self.provider_binding_set_sha256 == other.provider_binding_set_sha256
    }

    pub fn registry_digest_sha256(&self) -> &str {
        &self.registry_digest_sha256
    }

    pub fn provider_registry_identity_sha256(&self) -> Option<&str> {
        self.provider_registry_identity_sha256.as_deref()
    }

    pub fn execution_bundle_identity_sha256(&self) -> &str {
        &self.execution_bundle_identity_sha256
    }
}

enum CommandSequenceReconciliation {
    Completed,
    NotApplied,
    Unresolved(String),
}

pub struct NativeRuntimeHttpServices {
    profile_id: ProfileId,
    profile_identity: String,
    presentation: SharedExecutionPresentationService,
    controller: Arc<dyn ExecutionController>,
    registry: NativeNodeRegistry,
    provider_registry: Option<NativeProviderRegistryPin>,
    assets: Option<SharedAssetService>,
    asset_reader_authorization: Option<AuthorizedCapabilities>,
}

impl NativeRuntimeHttpServices {
    pub fn native_image(
        profile_id: ProfileId,
        presentation: SharedExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
    ) -> Result<Self, NativeServiceError> {
        Self::new(
            profile_id,
            presentation,
            controller,
            generated_native_node_registry_projection(None).map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Unavailable,
                    "native_image_registry_unavailable",
                    error.to_string(),
                )
            })?,
        )
    }

    #[cfg(test)]
    pub(crate) fn native_image_for_test(
        profile_id: ProfileId,
        controller: Arc<dyn ExecutionController>,
    ) -> Result<Self, NativeServiceError> {
        let mut presentation =
            sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                .map_err(presentation_error)?;
        presentation
            .initialize_profile(
                profile_id,
                sim_runtime::ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(presentation_error)?;
        Self::native_image(
            profile_id,
            sim_runtime::ExecutionPresentationOwner::ephemeral(presentation),
            controller,
        )
    }

    pub fn new(
        profile_id: ProfileId,
        presentation: SharedExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
        registry: NativeNodeRegistry,
    ) -> Result<Self, NativeServiceError> {
        registry
            .validate_comprehensive_bindings()
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Unavailable,
                    "native_execution_registry_incomplete",
                    error.to_string(),
                )
            })?;
        let profile_identity = profile_id.0.to_string();
        let service = Self {
            profile_id,
            profile_identity,
            presentation,
            controller,
            registry,
            provider_registry: None,
            assets: None,
            asset_reader_authorization: None,
        };
        service.snapshot()?;
        Ok(service)
    }

    pub fn from_registry_bundle(
        bundle: &NativeExecutionRegistryBundle,
        presentation: SharedExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
    ) -> Result<Self, NativeServiceError> {
        let mut service = Self::new(
            bundle.profile_id(),
            presentation,
            controller,
            bundle.registry().clone(),
        )?;
        service.provider_registry = bundle.provider_registry().cloned();
        Ok(service)
    }

    pub fn with_assets(
        mut self,
        assets: SharedAssetService,
        authorization: AuthorizedCapabilities,
    ) -> Result<Self, NativeServiceError> {
        let asset_profile = lock(&assets, "native_asset_state_poisoned")?
            .roots()
            .profile_id
            .clone();
        if asset_profile != self.profile_identity {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Forbidden,
                "native_asset_profile_mismatch",
                format!(
                    "asset profile {asset_profile:?} does not match API profile {:?}",
                    self.profile_identity
                ),
            ));
        }
        if authorization.profile_id() != self.profile_identity {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Forbidden,
                "native_asset_authorization_profile_mismatch",
                "native API asset authorization belongs to another runtime profile",
            ));
        }
        self.assets = Some(assets);
        self.asset_reader_authorization = Some(authorization);
        Ok(self)
    }

    pub fn presentation(&self) -> SharedExecutionPresentationService {
        self.presentation.clone()
    }

    fn asset_reader_authorization(&self) -> Result<&AuthorizedCapabilities, NativeServiceError> {
        self.asset_reader_authorization.as_ref().ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "native_asset_authorization_unavailable",
                "native API asset access requires an injected sealed permission grant",
            )
        })
    }

    pub fn http_capabilities(&self) -> Result<HttpCapabilities, NativeServiceError> {
        let catalog = http_route_catalog().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "native_http_catalog_invalid",
                error.to_string(),
            )
        })?;
        let mut capabilities = HttpCapabilities::default();
        for route in catalog {
            let method = route.contract.identity.method;
            let path = route.contract.identity.canonical_path.as_str();
            let state = if supports_native_route(method, path, self.assets.is_some()) {
                CapabilityState::Available
            } else {
                CapabilityState::Unavailable {
                    dependency: capability_owner_path(path).to_owned(),
                    reason: format!(
                        "{} is owned by a later native Rust service",
                        route.feature_id()
                    ),
                }
            };
            capabilities.set(route.feature_id(), state);
        }
        Ok(capabilities)
    }

    fn authorize(
        &self,
        authority: Option<&NativeRequestAuthority>,
    ) -> Result<(), NativeServiceError> {
        let authority = authority.ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Unauthorized,
                "native_request_authority_required",
                "native runtime services require host-authenticated request authority",
            )
        })?;
        if authority.profile_id != self.profile_identity {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Forbidden,
                "native_request_profile_mismatch",
                "request authority belongs to another native runtime profile",
            ));
        }
        if authority.principal.trim().is_empty() {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Unauthorized,
                "native_request_principal_required",
                "request authority has no authenticated principal identity",
            ));
        }
        if authority.plugin_id.is_some() != authority.plugin_digest.is_some() {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Forbidden,
                "native_plugin_authority_incomplete",
                "plugin route authority must bind both plugin identity and artifact digest",
            ));
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<ExecutionSnapshot, NativeServiceError> {
        self.presentation
            .snapshot(self.profile_id)
            .map_err(presentation_error)
    }

    fn dispatch_command(
        &self,
        command: ExecutionControlCommand,
    ) -> Result<ExecutionCommandAck, NativeServiceError> {
        let acknowledgement = smol::block_on(
            self.presentation
                .dispatch_durable(command, self.controller.as_ref()),
        )
        .map_err(presentation_error)?;
        match &acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted { .. } => Ok(acknowledgement),
            ExecutionCommandOutcome::Rejected { failure } => {
                let kind = match failure.origin {
                    ExecutionFailureOrigin::Validation => NativeServiceErrorKind::Invalid,
                    ExecutionFailureOrigin::Permission => NativeServiceErrorKind::Forbidden,
                    ExecutionFailureOrigin::Transport | ExecutionFailureOrigin::Provider => {
                        NativeServiceErrorKind::Unavailable
                    }
                    ExecutionFailureOrigin::Node
                    | ExecutionFailureOrigin::Decoding
                    | ExecutionFailureOrigin::Filesystem
                    | ExecutionFailureOrigin::Unknown => NativeServiceErrorKind::Conflict,
                };
                Err(NativeServiceError::new(
                    kind,
                    failure.code.clone(),
                    failure.message.clone(),
                ))
            }
        }
    }

    fn command_sequence_reconciliation(
        &self,
        request: &NativeServiceRequest,
        operation: &str,
    ) -> Result<CommandSequenceReconciliation, NativeServiceError> {
        let mut completed = false;
        for ordinal in 0..RECENT_COMMAND_RESULT_CAPACITY {
            let request_id = request_id(request, operation, ordinal)?;
            match self
                .presentation
                .command_receipt_state(self.profile_id, request_id)
                .map_err(presentation_error)?
            {
                ExecutionCommandReceiptState::Completed(acknowledgement) => {
                    match acknowledgement.outcome {
                        ExecutionCommandOutcome::Accepted { .. } => completed = true,
                        ExecutionCommandOutcome::Rejected { failure } => {
                            return Ok(CommandSequenceReconciliation::Unresolved(format!(
                                "canonical command {request_id:?} was rejected with {}: {}",
                                failure.code, failure.message
                            )));
                        }
                    }
                }
                ExecutionCommandReceiptState::Pending => {
                    return Ok(CommandSequenceReconciliation::Unresolved(format!(
                        "canonical command {request_id:?} remains pending"
                    )));
                }
                ExecutionCommandReceiptState::ReceiptUnavailable => {
                    return Ok(CommandSequenceReconciliation::Unresolved(format!(
                        "canonical command {request_id:?} completed without a retained receipt"
                    )));
                }
                ExecutionCommandReceiptState::NotApplied => {
                    return Ok(if completed {
                        CommandSequenceReconciliation::Completed
                    } else {
                        CommandSequenceReconciliation::NotApplied
                    });
                }
            }
        }
        Ok(CommandSequenceReconciliation::Unresolved(format!(
            "the mutation consumed the full {RECENT_COMMAND_RESULT_CAPACITY}-receipt reconciliation window"
        )))
    }

    fn prompt_plan(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<CompiledPlan, NativeServiceError> {
        let value = required_json_object(request)?;
        let submission: PromptSubmission = serde_json::from_value(Value::Object(value.clone()))
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Invalid,
                    "invalid_prompt_submission",
                    error.to_string(),
                )
            })?;
        let compiler = PromptCompiler::new(&self.registry);
        let compiler = match self.provider_registry.as_ref() {
            Some(provider_registry) => compiler
                .with_provider_registry_pin(provider_registry.clone())
                .map_err(|error| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Unavailable,
                        "provider_registry_unavailable",
                        error.to_string(),
                    )
                })?,
            None => compiler,
        };
        compiler.compile(submission).map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "prompt_validation_failed",
                error.to_string(),
            )
        })
    }

    fn reconcile_prompt(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        let plan = self.prompt_plan(request)?;
        let command_request_id = request_id(request, "submit_prompt", 0)?;
        match self
            .presentation
            .command_receipt_state(self.profile_id, command_request_id)
            .map_err(presentation_error)?
        {
            ExecutionCommandReceiptState::Completed(acknowledgement) => {
                match acknowledgement.outcome {
                    ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: Some(attempt_id),
                    } => Ok(NativeMutationReconciliation::Committed(
                        prompt_submission_response(&plan, attempt_id),
                    )),
                    ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: None,
                    } => Ok(NativeMutationReconciliation::Unresolved {
                        reason: "the canonical prompt receipt has no assigned attempt identity"
                            .to_owned(),
                    }),
                    ExecutionCommandOutcome::Rejected { failure } => {
                        Ok(NativeMutationReconciliation::Unresolved {
                            reason: format!(
                                "the canonical prompt command was rejected with {}: {}",
                                failure.code, failure.message
                            ),
                        })
                    }
                }
            }
            ExecutionCommandReceiptState::Pending => Ok(NativeMutationReconciliation::Unresolved {
                reason: "the canonical prompt command remains pending".to_owned(),
            }),
            ExecutionCommandReceiptState::NotApplied => {
                Ok(NativeMutationReconciliation::NotApplied)
            }
            ExecutionCommandReceiptState::ReceiptUnavailable => {
                Ok(NativeMutationReconciliation::Unresolved {
                    reason: "the canonical prompt command completed without a retained receipt"
                        .to_owned(),
                })
            }
        }
    }

    fn reconcile_queue_mutation(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        let object = required_json_object(request)?;
        let clear = optional_boolean(object, "clear")?.unwrap_or(false);
        let delete = optional_string_array(object, "delete")?
            .into_iter()
            .map(|prompt_id| parse_prompt_id(&prompt_id))
            .collect::<Result<HashSet<_>, _>>()?;
        match self.command_sequence_reconciliation(request, "queue_mutation")? {
            CommandSequenceReconciliation::NotApplied => {
                Ok(NativeMutationReconciliation::NotApplied)
            }
            CommandSequenceReconciliation::Unresolved(reason) => {
                Ok(NativeMutationReconciliation::Unresolved { reason })
            }
            CommandSequenceReconciliation::Completed => {
                if clear && !delete.is_empty() {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason: "a combined queue clear/delete cannot prove that every derived command completed"
                            .to_owned(),
                    });
                }
                let snapshot = self.snapshot()?;
                if snapshot
                    .queue
                    .iter()
                    .any(|queued| delete.contains(&queued.prompt_id))
                {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason: "a prompt targeted by the queue deletion remains queued".to_owned(),
                    });
                }
                Ok(NativeMutationReconciliation::Committed(empty_response(200)))
            }
        }
    }

    fn reconcile_history_mutation(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        let object = required_json_object(request)?;
        let clear = optional_boolean(object, "clear")?.unwrap_or(false);
        let delete = optional_string_array(object, "delete")?
            .into_iter()
            .map(|prompt_id| parse_prompt_id(&prompt_id))
            .collect::<Result<HashSet<_>, _>>()?;
        match self.command_sequence_reconciliation(request, "history_mutation")? {
            CommandSequenceReconciliation::NotApplied => {
                Ok(NativeMutationReconciliation::NotApplied)
            }
            CommandSequenceReconciliation::Unresolved(reason) => {
                Ok(NativeMutationReconciliation::Unresolved { reason })
            }
            CommandSequenceReconciliation::Completed => {
                if clear && !delete.is_empty() {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason: "a combined history clear/delete cannot prove that every derived command completed"
                            .to_owned(),
                    });
                }
                let snapshot = self.snapshot()?;
                if snapshot.attempts.iter().any(|attempt| {
                    attempt.state.is_terminal() && delete.contains(&attempt.prompt_id)
                }) {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason:
                            "a terminal attempt targeted by the history deletion remains retained"
                                .to_owned(),
                    });
                }
                Ok(NativeMutationReconciliation::Committed(empty_response(200)))
            }
        }
    }

    fn reconcile_interrupt(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        let prompt_id = interrupt_prompt_id(request)?;
        match self.command_sequence_reconciliation(request, "interrupt")? {
            CommandSequenceReconciliation::NotApplied => {
                Ok(NativeMutationReconciliation::NotApplied)
            }
            CommandSequenceReconciliation::Unresolved(reason) => {
                Ok(NativeMutationReconciliation::Unresolved { reason })
            }
            CommandSequenceReconciliation::Completed => {
                let snapshot = self.snapshot()?;
                if snapshot.attempts.iter().any(|attempt| {
                    attempt.state == AttemptState::Running
                        && prompt_id.is_none_or(|prompt_id| attempt.prompt_id == prompt_id)
                }) {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason: "an attempt targeted by the interruption remains running"
                            .to_owned(),
                    });
                }
                Ok(NativeMutationReconciliation::Committed(empty_response(200)))
            }
        }
    }

    fn reconcile_jobs(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        let path = request.route.canonical_path.as_str();
        let job_ids = match (request.route.method, path) {
            (HttpMethod::Post, "/api/jobs/{job_id}/cancel") => {
                vec![required_path(request, "job_id")?.to_owned()]
            }
            (HttpMethod::Post, "/api/jobs/cancel") => {
                required_string_array(required_json_object(request)?, "job_ids")?
            }
            _ => {
                return Ok(NativeMutationReconciliation::Unresolved {
                    reason: "the native job route has no canonical mutation reconciler".to_owned(),
                });
            }
        };
        let prompt_ids = job_ids
            .iter()
            .map(|job_id| parse_prompt_id(job_id))
            .collect::<Result<HashSet<_>, _>>()?;
        match self.command_sequence_reconciliation(request, "cancel_job")? {
            CommandSequenceReconciliation::NotApplied => {
                Ok(NativeMutationReconciliation::NotApplied)
            }
            CommandSequenceReconciliation::Unresolved(reason) => {
                Ok(NativeMutationReconciliation::Unresolved { reason })
            }
            CommandSequenceReconciliation::Completed => {
                let snapshot = self.snapshot()?;
                if snapshot.attempts.iter().any(|attempt| {
                    prompt_ids.contains(&attempt.prompt_id)
                        && matches!(attempt.state, AttemptState::Queued | AttemptState::Running)
                }) {
                    return Ok(NativeMutationReconciliation::Unresolved {
                        reason: "a job targeted by cancellation remains cancellable".to_owned(),
                    });
                }
                Ok(NativeMutationReconciliation::Committed(
                    NativeServiceResponse::json(200, json!({"cancelled": true})),
                ))
            }
        }
    }

    fn features(&self) -> Result<NativeServiceResponse, NativeServiceError> {
        let snapshot = self.snapshot()?;
        Ok(NativeServiceResponse::json(
            200,
            json!({
                "zed_native_api": {
                    "protocol_version": sim_types::NATIVE_PROTOCOL_VERSION,
                    "profile_id": self.profile_identity,
                    "native_execution": true,
                    "native_node_count": self.registry.descriptor_len(),
                    "native_asset_index": self.assets.is_some(),
                    "python_execution": false,
                    "javascript_extension_execution": false,
                    "external_server_forwarding": false,
                    "execution_status": snapshot.status,
                }
            }),
        ))
    }

    fn prompt_status(&self) -> Result<NativeServiceResponse, NativeServiceError> {
        Ok(NativeServiceResponse::json(
            200,
            self.execution_status_projection()?,
        ))
    }

    fn execution_status_projection(&self) -> Result<Value, NativeServiceError> {
        let snapshot = self.snapshot()?;
        let running = snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.state,
                    AttemptState::Running | AttemptState::Cancelling
                )
            })
            .count();
        Ok(json!({
            "exec_info": {
                "queue_remaining": snapshot.queue.len(),
                "running": running,
            }
        }))
    }

    fn submit_prompt(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let value = required_json_object(request)?;
        let front = value.get("front").and_then(Value::as_bool).unwrap_or(false);
        let plan = self.prompt_plan(request)?;
        let command = ExecutionControlCommand {
            request_id: request_id(request, "submit_prompt", 0)?,
            profile_id: self.profile_id,
            expected_revision: None,
            kind: ExecutionControlCommandKind::Queue {
                plan: plan.clone(),
                priority: 0,
                front,
            },
        };
        let acknowledgement = self.dispatch_command(command)?;
        let assigned_attempt_id = match acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id,
            } => assigned_attempt_id,
            ExecutionCommandOutcome::Rejected { .. } => None,
        };
        let assigned_attempt_id = assigned_attempt_id.ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "native_prompt_attempt_identity_missing",
                "an accepted native prompt did not assign an attempt identity",
            )
        })?;
        Ok(prompt_submission_response(&plan, assigned_attempt_id))
    }

    fn queue_snapshot(&self) -> Result<NativeServiceResponse, NativeServiceError> {
        let (snapshot, persisted) = self
            .presentation
            .snapshot_with_persisted_attempts(self.profile_id)
            .map_err(presentation_error)?;
        let queue_pending = snapshot
            .queue
            .iter()
            .map(|queued| queue_tuple(&queued.plan, Some(queued.enqueue_sequence)))
            .collect::<Result<Vec<_>, _>>()?;
        let plans = persisted
            .into_iter()
            .filter_map(|attempt| attempt.plan.map(|plan| (attempt.record.attempt_id, plan)))
            .collect::<HashMap<_, _>>();
        let queue_running = snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.state,
                    AttemptState::Running | AttemptState::Cancelling
                )
            })
            .filter_map(|attempt| plans.get(&attempt.attempt_id))
            .map(|plan| queue_tuple(plan, None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NativeServiceResponse::json(
            200,
            json!({
                "queue_running": queue_running,
                "queue_pending": queue_pending,
            }),
        ))
    }

    fn mutate_queue(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let object = required_json_object(request)?;
        let clear = optional_boolean(object, "clear")?.unwrap_or(false);
        let delete = optional_string_array(object, "delete")?;
        let snapshot = self.snapshot()?;
        let mut commands = Vec::new();
        if clear {
            commands.push(ExecutionControlCommandKind::ClearPending {
                reason: "native HTTP queue clear".to_owned(),
            });
        }
        for prompt_id in delete {
            let prompt_id = parse_prompt_id(&prompt_id)?;
            commands.extend(
                snapshot
                    .queue
                    .iter()
                    .filter(|queued| queued.prompt_id == prompt_id)
                    .map(|queued| ExecutionControlCommandKind::Cancel {
                        attempt_id: queued.attempt_id,
                        reason: "native HTTP queue deletion".to_owned(),
                    }),
            );
        }
        for (index, kind) in commands.into_iter().enumerate() {
            self.dispatch_command(ExecutionControlCommand {
                request_id: request_id(request, "queue_mutation", index)?,
                profile_id: self.profile_id,
                expected_revision: None,
                kind,
            })?;
        }
        Ok(empty_response(200))
    }

    fn history(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let (snapshot, persisted) = self
            .presentation
            .snapshot_with_persisted_attempts(self.profile_id)
            .map_err(presentation_error)?;
        let plans = persisted
            .into_iter()
            .filter_map(|attempt| attempt.plan.map(|plan| (attempt.record.attempt_id, plan)))
            .collect::<HashMap<_, _>>();
        let requested_prompt = request
            .route
            .path_parameters
            .get("prompt_id")
            .map(|value| parse_prompt_id(value))
            .transpose()?;
        let maximum = query_usize(request, "max_items")?.unwrap_or(MAXIMUM_HISTORY_PAGE_SIZE);
        if maximum > MAXIMUM_HISTORY_PAGE_SIZE {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "history_page_too_large",
                format!("max_items exceeds {MAXIMUM_HISTORY_PAGE_SIZE}"),
            ));
        }
        let offset = query_usize(request, "offset")?.unwrap_or(0);
        let mut attempts = snapshot.attempts.iter().collect::<Vec<_>>();
        attempts.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then(right.attempt_id.0.cmp(&left.attempt_id.0))
        });
        let filtered = attempts
            .into_iter()
            .filter(|attempt| {
                requested_prompt.is_none_or(|prompt_id| attempt.prompt_id == prompt_id)
            })
            .skip(offset)
            .take(maximum);
        let mut response = Map::new();
        for attempt in filtered {
            let plan = plans.get(&attempt.attempt_id).ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Unavailable,
                    "native_history_plan_unavailable",
                    "retained native history does not include its compatibility prompt plan",
                )
            })?;
            response.insert(
                attempt.prompt_id.0.to_string(),
                history_record(attempt, plan)?,
            );
        }
        Ok(NativeServiceResponse::json(200, Value::Object(response)))
    }

    fn mutate_history(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let object = required_json_object(request)?;
        let clear = optional_boolean(object, "clear")?.unwrap_or(false);
        let delete = optional_string_array(object, "delete")?;
        let snapshot = self.snapshot()?;
        let mut commands = Vec::new();
        if clear {
            commands.push(ExecutionControlCommandKind::ClearHistory);
        }
        for prompt_id in delete {
            let prompt_id = parse_prompt_id(&prompt_id)?;
            commands.extend(
                snapshot
                    .attempts
                    .iter()
                    .filter(|attempt| attempt.prompt_id == prompt_id && attempt.state.is_terminal())
                    .map(|attempt| ExecutionControlCommandKind::RemoveHistory {
                        attempt_id: attempt.attempt_id,
                    }),
            );
        }
        for (index, kind) in commands.into_iter().enumerate() {
            self.dispatch_command(ExecutionControlCommand {
                request_id: request_id(request, "history_mutation", index)?,
                profile_id: self.profile_id,
                expected_revision: None,
                kind,
            })?;
        }
        Ok(empty_response(200))
    }

    fn interrupt(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let prompt_id = interrupt_prompt_id(request)?;
        let snapshot = self.snapshot()?;
        let attempts = snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.state,
                    AttemptState::Running | AttemptState::Cancelling
                ) && prompt_id.is_none_or(|prompt_id| attempt.prompt_id == prompt_id)
            })
            .map(|attempt| attempt.attempt_id)
            .collect::<Vec<_>>();
        for (index, attempt_id) in attempts.into_iter().enumerate() {
            self.dispatch_command(ExecutionControlCommand {
                request_id: request_id(request, "interrupt", index)?,
                profile_id: self.profile_id,
                expected_revision: None,
                kind: ExecutionControlCommandKind::Interrupt {
                    attempt_id,
                    reason: "native HTTP interrupt".to_owned(),
                },
            })?;
        }
        Ok(empty_response(200))
    }

    fn jobs(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let path = request.route.canonical_path.as_str();
        match (request.route.method, path) {
            (HttpMethod::Get, "/api/jobs") => self.list_jobs(request),
            (HttpMethod::Get, "/api/jobs/{job_id}") => self.get_job(request),
            (HttpMethod::Post, "/api/jobs/{job_id}/cancel") => {
                let job_id = required_path(request, "job_id")?;
                let cancelled = self.cancel_jobs(request, &[job_id.to_owned()])?;
                Ok(NativeServiceResponse::json(
                    200,
                    json!({"cancelled": cancelled}),
                ))
            }
            (HttpMethod::Post, "/api/jobs/cancel") => {
                let body = required_json_object(request)?;
                let jobs = required_string_array(body, "job_ids")?;
                let cancelled = self.cancel_jobs(request, &jobs)?;
                Ok(NativeServiceResponse::json(
                    200,
                    json!({"cancelled": cancelled}),
                ))
            }
            _ => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_job_service",
                "the native job service does not own this route shape",
            ),
        }
    }

    fn list_jobs(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let statuses = query_first(request, "status")
            .map(|statuses| {
                statuses
                    .split(',')
                    .filter(|status| !status.trim().is_empty())
                    .map(|status| status.trim().to_ascii_lowercase())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let valid = BTreeSet::from([
            "pending".to_owned(),
            "in_progress".to_owned(),
            "completed".to_owned(),
            "failed".to_owned(),
        ]);
        if !statuses.is_subset(&valid) {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "invalid_job_status",
                "status must contain pending, in_progress, completed, or failed",
            ));
        }
        let limit = query_usize(request, "limit")?.unwrap_or(100);
        if limit == 0 || limit > MAXIMUM_JOB_PAGE_SIZE {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "invalid_job_limit",
                format!("limit must be between 1 and {MAXIMUM_JOB_PAGE_SIZE}"),
            ));
        }
        let offset = query_usize(request, "offset")?.unwrap_or(0);
        if query_first(request, "workflow_id").is_some() {
            return capability_unavailable(
                &request.route.canonical_feature_id,
                "native_workflow_repository",
                "workflow_id filtering requires the native workflow repository",
            );
        }
        let sort_by = query_first(request, "sort_by").unwrap_or("created_at");
        if !matches!(sort_by, "created_at" | "execution_duration") {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "invalid_job_sort",
                "sort_by must be created_at or execution_duration",
            ));
        }
        let sort_order = match query_first(request, "sort_order") {
            Some("asc") => "asc",
            _ => "desc",
        };
        let snapshot = self.snapshot()?;
        let mut attempts = snapshot
            .attempts
            .iter()
            .filter(|attempt| statuses.is_empty() || statuses.contains(job_status(attempt.state)))
            .collect::<Vec<_>>();
        attempts.sort_by(|left, right| {
            if sort_by == "execution_duration" {
                execution_duration_milliseconds(left)
                    .cmp(&execution_duration_milliseconds(right))
                    .then(left.created_at.cmp(&right.created_at))
            } else {
                left.created_at.cmp(&right.created_at)
            }
        });
        if sort_order == "desc" {
            attempts.reverse();
        }
        let total = attempts.len();
        let jobs = attempts
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(job_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NativeServiceResponse::json(
            200,
            json!({"jobs": jobs, "total": total, "offset": offset, "limit": limit}),
        ))
    }

    fn get_job(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let prompt_id = parse_prompt_id(required_path(request, "job_id")?)?;
        let snapshot = self.snapshot()?;
        let attempt = snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.prompt_id == prompt_id)
            .max_by(|left, right| {
                left.created_at
                    .cmp(&right.created_at)
                    .then(left.attempt_id.0.cmp(&right.attempt_id.0))
            })
            .ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::NotFound,
                    "job_not_found",
                    format!("job {} was not found", prompt_id.0),
                )
            })?;
        Ok(NativeServiceResponse::json(200, job_record(attempt)?))
    }

    fn cancel_jobs(
        &self,
        request: &NativeServiceRequest,
        job_ids: &[String],
    ) -> Result<bool, NativeServiceError> {
        let prompt_ids = job_ids
            .iter()
            .map(|job_id| parse_prompt_id(job_id))
            .collect::<Result<HashSet<_>, _>>()?;
        let snapshot = self.snapshot()?;
        let mut attempts = snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                prompt_ids.contains(&attempt.prompt_id)
                    && matches!(
                        attempt.state,
                        AttemptState::Queued | AttemptState::Running | AttemptState::Cancelling
                    )
            })
            .map(|attempt| attempt.attempt_id)
            .collect::<Vec<_>>();
        attempts.sort_by_key(|attempt_id| attempt_id.0);
        attempts.dedup();
        for (index, attempt_id) in attempts.iter().copied().enumerate() {
            self.dispatch_command(ExecutionControlCommand {
                request_id: request_id(request, "cancel_job", index)?,
                profile_id: self.profile_id,
                expected_revision: None,
                kind: ExecutionControlCommandKind::Cancel {
                    attempt_id,
                    reason: "native HTTP job cancellation".to_owned(),
                },
            })?;
        }
        Ok(!attempts.is_empty())
    }

    fn node_catalog(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let source_registry = NodeRegistry::built_in().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "native_node_source_registry_invalid",
                error.to_string(),
            )
        })?;
        let object_info =
            ObjectInfoRegistry::from_node_registry(&source_registry).map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "native_object_info_registry_invalid",
                    error.to_string(),
                )
            })?;
        let bindings = native_image_catalog_bindings().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Unavailable,
                "native_node_catalog_unavailable",
                error.to_string(),
            )
        })?;
        let requested = request.route.path_parameters.get("node_class");
        let mut result = Map::new();
        for (class_type, runtime) in self.registry.descriptors() {
            if requested.is_some_and(|requested| requested != class_type) {
                continue;
            }
            let disposition = self
                .registry
                .binding_disposition(class_type)
                .ok_or_else(|| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "native_node_binding_missing",
                        format!("{class_type} has metadata but no compiled binding"),
                    )
                })?;
            let unavailable_reason = self.registry.unavailable_reason(class_type);
            let source = object_info.nodes().get(class_type);
            let python_module = if let Some(source) = source {
                source.source_python_module.as_str()
            } else {
                self.registry
                    .implementation_namespace(class_type)
                    .ok_or_else(|| {
                        NativeServiceError::new(
                            NativeServiceErrorKind::Internal,
                            "native_plugin_namespace_missing",
                            format!("{class_type} has no signed implementation namespace"),
                        )
                    })?
            };
            if disposition == NativeNodeBindingDisposition::Unavailable
                && source.is_some_and(|source| source.source_schema.is_none())
            {
                let source = source.ok_or_else(|| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "native_node_source_metadata_missing",
                        format!("{class_type} has no canonical source object-info row"),
                    )
                })?;
                result.insert(class_type.to_owned(), project_inactive_source_node(source));
                continue;
            }
            let presentation = self.registry.presentation(class_type).ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "native_node_presentation_missing",
                    format!("{class_type} has no checked presentation projection"),
                )
            })?;
            let mut input_choice_overrides = HashMap::new();
            let mut input_option_overrides = HashMap::new();
            if let Some(binding) = bindings.get(class_type) {
                if let Some(source) = source
                    && binding.native.python_module != source.source_python_module
                {
                    return Err(NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "native_node_source_module_mismatch",
                        format!("{class_type} disagrees with its canonical source Python module"),
                    ));
                }
                for input in &binding.native.inputs {
                    if input.choices_from_input_assets || !input.choices.is_empty() {
                        let choices = if input.choices_from_input_assets {
                            self.input_asset_choices()?
                        } else {
                            input.choices.iter().cloned().map(Value::String).collect()
                        };
                        input_choice_overrides.insert(input.name.clone(), choices);
                    }
                    input_option_overrides.insert(
                        input.name.clone(),
                        input.options.clone().into_iter().collect::<Map<_, _>>(),
                    );
                }
            }
            let mut projected = project_component_node(
                class_type,
                python_module,
                runtime,
                presentation,
                disposition,
                unavailable_reason,
                source,
                &input_choice_overrides,
                &input_option_overrides,
            )?;
            if let Some(binding) = bindings.get(class_type)
                && let Some(node) = projected.as_object_mut()
            {
                node.insert(
                    "description".to_owned(),
                    Value::String(binding.native.description.clone()),
                );
                node.insert(
                    "search_aliases".to_owned(),
                    json!(binding.native.search_aliases),
                );
                node.insert(
                    "essentials_category".to_owned(),
                    json!(binding.native.essentials_category),
                );
                node.insert(
                    "has_intermediate_output".to_owned(),
                    Value::Bool(binding.native.has_intermediate_output),
                );
            }
            result.insert(class_type.to_owned(), projected);
        }
        for (class_type, source) in object_info.nodes() {
            if result.contains_key(class_type)
                || requested.is_some_and(|requested| requested != class_type)
            {
                continue;
            }
            let projected = if source.source_schema.is_some() {
                project_unbound_source_node(source)
            } else {
                project_inactive_source_node(source)
            };
            result.insert(class_type.clone(), projected);
        }
        if requested.is_some() && result.is_empty() {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::NotFound,
                "native_node_not_found",
                "the requested node is not in the canonical native source catalog",
            ));
        }
        Ok(NativeServiceResponse::json(200, Value::Object(result)))
    }

    fn input_asset_choices(&self) -> Result<Vec<Value>, NativeServiceError> {
        let Some(assets) = &self.assets else {
            return Ok(Vec::new());
        };
        let authorization = self.asset_reader_authorization()?;
        let assets = lock(assets, "native_asset_state_poisoned")?;
        let mut offset = 0;
        let mut choices = Vec::new();
        while choices.len() < MAXIMUM_ASSET_SCAN_RESULTS {
            let page = assets
                .list_authorized(
                    &AssetQuery {
                        namespace: Some(AssetNamespace::Input),
                        availability: Some(AssetAvailability::Present),
                        offset,
                        limit: 1_000,
                        ..AssetQuery::default()
                    },
                    authorization,
                )
                .map_err(asset_error)?;
            let page_len = page.records.len();
            choices.extend(
                page.records
                    .into_iter()
                    .filter(|record| record.content_type.starts_with("image/"))
                    .map(|record| {
                        Value::String(record.identity.relative_path.to_string_lossy().into_owned())
                    }),
            );
            let Some(next_offset) = page.next_offset else {
                break;
            };
            if page_len == 0 || next_offset <= offset {
                return Err(NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "native_asset_pagination_invalid",
                    "the canonical asset index did not advance while projecting input choices",
                ));
            }
            offset = next_offset;
        }
        choices.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        choices.truncate(MAXIMUM_ASSET_SCAN_RESULTS);
        Ok(choices)
    }

    fn models(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        let records = self.model_records()?;
        match request.route.canonical_path.as_str() {
            "/models" => {
                let folders = model_folders(&records);
                Ok(NativeServiceResponse::json(200, json!(folders)))
            }
            "/models/{folder}" => {
                let folder = required_path(request, "folder")?;
                Ok(NativeServiceResponse::json(
                    200,
                    json!(model_paths(&records, folder)),
                ))
            }
            "/experiment/models" => Ok(NativeServiceResponse::json(
                200,
                Value::Array(experimental_model_folders(&records)),
            )),
            _ => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_model_preview_service",
                "model previews require a native preview provider",
            ),
        }
    }

    fn model_records(&self) -> Result<Vec<AssetRecord>, NativeServiceError> {
        let assets = self.assets.as_ref().ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Unavailable,
                "native_asset_service_unavailable",
                "the native API profile has no typed asset service",
            )
        })?;
        let authorization = self.asset_reader_authorization()?;
        let assets = lock(assets, "native_asset_state_poisoned")?;
        let mut records = Vec::new();
        let mut offset = 0;
        loop {
            let page = assets
                .list_authorized(
                    &AssetQuery {
                        namespace: Some(AssetNamespace::Model),
                        availability: Some(AssetAvailability::Present),
                        offset,
                        limit: MAXIMUM_JOB_PAGE_SIZE,
                        ..AssetQuery::default()
                    },
                    authorization,
                )
                .map_err(asset_error)?;
            records.extend(page.records);
            if records.len() > MAXIMUM_ASSET_SCAN_RESULTS {
                return Err(NativeServiceError::new(
                    NativeServiceErrorKind::Oversized,
                    "native_model_catalog_too_large",
                    format!("native model catalog exceeds {MAXIMUM_ASSET_SCAN_RESULTS} records"),
                ));
            }
            let Some(next) = page.next_offset else {
                break;
            };
            offset = next;
        }
        Ok(records)
    }

    fn catalog_route(
        &self,
        request: &NativeServiceRequest,
        feature_id: &str,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        match (request.route.method, request.route.canonical_path.as_str()) {
            (HttpMethod::Get, "/health") => {
                let snapshot = self.snapshot()?;
                if matches!(snapshot.status, ExecutionSnapshotStatus::Unavailable { .. }) {
                    Ok(NativeServiceResponse::bytes(
                        503,
                        "text/plain; charset=utf-8",
                        "Service Unavailable",
                    ))
                } else {
                    Ok(NativeServiceResponse::bytes(
                        200,
                        "text/plain; charset=utf-8",
                        "OK",
                    ))
                }
            }
            (HttpMethod::Get, "/system_stats") => {
                let snapshot = self.snapshot()?;
                Ok(NativeServiceResponse::json(
                    200,
                    json!({
                        "system": {
                            "native": true,
                            "profile_id": self.profile_identity,
                            "queue_pending": snapshot.queue.len(),
                            "attempts_retained": snapshot.attempts.len(),
                            "python_runtime": false,
                            "external_server": false,
                        },
                        "devices": [],
                    }),
                ))
            }
            (HttpMethod::Get, "/api/history_v2") => self.history(request),
            (HttpMethod::Get, "/api/history_v2/{prompt_id}") => self.history(request),
            (HttpMethod::Get, "/api/job/{job_id}/status") => {
                let prompt_id = parse_prompt_id(required_path(request, "job_id")?)?;
                let snapshot = self.snapshot()?;
                let attempt = snapshot
                    .attempts
                    .iter()
                    .filter(|attempt| attempt.prompt_id == prompt_id)
                    .max_by_key(|attempt| (attempt.created_at, attempt.attempt_id.0))
                    .ok_or_else(|| {
                        NativeServiceError::new(
                            NativeServiceErrorKind::NotFound,
                            "job_not_found",
                            "the requested native job was not found",
                        )
                    })?;
                Ok(NativeServiceResponse::json(
                    200,
                    json!({"job_id": prompt_id, "status": job_status(attempt.state)}),
                ))
            }
            (HttpMethod::Get, "/internal/files/{directory_type}") => {
                let namespace = parse_asset_namespace(required_path(request, "directory_type")?)?;
                let records = self.asset_records(namespace)?;
                let paths = records
                    .iter()
                    .map(|record| normalized_relative_path(record))
                    .collect::<Vec<_>>();
                Ok(NativeServiceResponse::json(200, json!(paths)))
            }
            (HttpMethod::Get, "/internal/folder_paths") => {
                let assets = self.assets.as_ref().ok_or_else(|| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Unavailable,
                        "native_asset_service_unavailable",
                        "the native API profile has no typed asset service",
                    )
                })?;
                let authorization = self.asset_reader_authorization()?;
                let assets = lock(assets, "native_asset_state_poisoned")?;
                let paths = assets
                    .authorized_compatibility_folder_paths(authorization)
                    .map_err(asset_error)?
                    .into_iter()
                    .map(|(namespace, path)| {
                        (
                            namespace.locator_type().to_owned(),
                            Value::String(path.to_string_lossy().into_owned()),
                        )
                    })
                    .collect::<Map<_, _>>();
                Ok(NativeServiceResponse::json(200, Value::Object(paths)))
            }
            _ => capability_unavailable(
                feature_id,
                capability_owner(request),
                "this compatibility route is owned by a later native Rust service",
            ),
        }
    }

    fn asset_records(
        &self,
        namespace: AssetNamespace,
    ) -> Result<Vec<AssetRecord>, NativeServiceError> {
        let assets = self.assets.as_ref().ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Unavailable,
                "native_asset_service_unavailable",
                "the native API profile has no typed asset service",
            )
        })?;
        let authorization = self.asset_reader_authorization()?;
        let assets = lock(assets, "native_asset_state_poisoned")?;
        let mut records = Vec::new();
        let mut offset = 0;
        loop {
            let page = assets
                .list_authorized(
                    &AssetQuery {
                        namespace: Some(namespace),
                        availability: Some(AssetAvailability::Present),
                        offset,
                        limit: MAXIMUM_JOB_PAGE_SIZE,
                        ..AssetQuery::default()
                    },
                    authorization,
                )
                .map_err(asset_error)?;
            records.extend(page.records);
            if records.len() > MAXIMUM_ASSET_SCAN_RESULTS {
                return Err(NativeServiceError::new(
                    NativeServiceErrorKind::Oversized,
                    "native_asset_catalog_too_large",
                    format!("native asset catalog exceeds {MAXIMUM_ASSET_SCAN_RESULTS} records"),
                ));
            }
            let Some(next) = page.next_offset else {
                break;
            };
            offset = next;
        }
        Ok(records)
    }
}

impl NativeHttpServices for NativeRuntimeHttpServices {
    fn dispatch(
        &self,
        request: NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError> {
        self.authorize(request.authority.as_ref())?;
        match &request.operation {
            NativeServiceOperation::Root => capability_unavailable(
                &request.route.canonical_feature_id,
                "gpui_frontend",
                "Zed's GPUI frontend is not served as a web static bundle",
            ),
            NativeServiceOperation::Features => self.features(),
            NativeServiceOperation::PromptStatus => self.prompt_status(),
            NativeServiceOperation::SubmitPrompt => self.submit_prompt(&request),
            NativeServiceOperation::QueueSnapshot => self.queue_snapshot(),
            NativeServiceOperation::QueueMutation => self.mutate_queue(&request),
            NativeServiceOperation::HistoryRead => self.history(&request),
            NativeServiceOperation::HistoryMutation => self.mutate_history(&request),
            NativeServiceOperation::Interrupt => self.interrupt(&request),
            NativeServiceOperation::Jobs => self.jobs(&request),
            NativeServiceOperation::Models => self.models(&request),
            NativeServiceOperation::NodeCatalog => self.node_catalog(&request),
            NativeServiceOperation::Assets => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_asset_reference_service",
                "database-style asset reference routes are not owned by the typed filesystem asset index",
            ),
            NativeServiceOperation::Upload => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_multipart_asset_service",
                "multipart compatibility uploads require the native upload transaction service",
            ),
            NativeServiceOperation::UserData => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_profile_document_service",
                "user-data compatibility routes require the native profile document service",
            ),
            NativeServiceOperation::Settings => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_settings_service",
                "settings compatibility routes require the native settings service",
            ),
            NativeServiceOperation::Extensions => capability_unavailable(
                &request.route.canonical_feature_id,
                "rust_wasm_extension_host",
                "extension routes are available only through registered Rust/WASM components",
            ),
            NativeServiceOperation::StaticAsset => capability_unavailable(
                &request.route.canonical_feature_id,
                "gpui_frontend",
                "Zed does not expose GPUI resources through a web static-file root",
            ),
            NativeServiceOperation::WebSocketUpgrade => capability_unavailable(
                &request.route.canonical_feature_id,
                "native_websocket_transport",
                "WebSocket upgrades are owned by the native API transport, not HTTP services",
            ),
            NativeServiceOperation::CatalogRoute { feature_id } => {
                self.catalog_route(&request, feature_id)
            }
        }
    }

    fn reconcile_mutation(
        &self,
        request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        self.authorize(request.authority.as_ref())?;
        match &request.operation {
            NativeServiceOperation::SubmitPrompt => self.reconcile_prompt(request),
            NativeServiceOperation::QueueMutation => self.reconcile_queue_mutation(request),
            NativeServiceOperation::HistoryMutation => self.reconcile_history_mutation(request),
            NativeServiceOperation::Interrupt => self.reconcile_interrupt(request),
            NativeServiceOperation::Jobs => self.reconcile_jobs(request),
            operation => Ok(NativeMutationReconciliation::Unresolved {
                reason: format!(
                    "native operation {operation:?} has no canonical command-receipt reconciliation"
                ),
            }),
        }
    }

    fn status_projection(&self) -> Result<Option<Value>, NativeServiceError> {
        self.execution_status_projection().map(Some)
    }
}

fn project_component_node(
    class_type: &str,
    python_module: &str,
    runtime: &RuntimeNodeDescriptor,
    presentation: &RuntimeNodePresentation,
    disposition: NativeNodeBindingDisposition,
    unavailable_reason: Option<&str>,
    source: Option<&ObjectInfoNode>,
    input_choice_overrides: &HashMap<String, Vec<Value>>,
    input_option_overrides: &HashMap<String, Map<String, Value>>,
) -> Result<Value, NativeServiceError> {
    let mut required = Map::new();
    let mut optional = Map::new();
    let mut hidden = Map::new();
    let mut required_order = Vec::new();
    let mut optional_order = Vec::new();
    let mut hidden_order = Vec::new();
    let catalog_schema = source.and_then(|source| source.source_schema.as_ref());
    if let Some(schema) = catalog_schema {
        for input in &schema.inputs {
            let runtime_input = runtime
                .inputs
                .iter()
                .find(|candidate| candidate.name == input.schema.name)
                .ok_or_else(|| schema_projection_error(class_type, &input.schema.name))?;
            insert_projected_input(
                &input.schema,
                input.requirement,
                runtime_input.lazy,
                input_choice_overrides
                    .get(&input.schema.name)
                    .map(Vec::as_slice),
                input_option_overrides.get(&input.schema.name),
                &mut required,
                &mut optional,
                &mut hidden,
                &mut required_order,
                &mut optional_order,
                &mut hidden_order,
            );
        }
    } else if let Some(schema) = runtime.source_schema.as_ref() {
        for (input, input_schema) in runtime.inputs.iter().zip(&schema.inputs) {
            let requirement = if input.hidden {
                NativeInputRequirement::Hidden
            } else if input.required {
                NativeInputRequirement::Required
            } else {
                NativeInputRequirement::Optional
            };
            insert_projected_input(
                input_schema,
                requirement,
                input.lazy,
                input_choice_overrides.get(&input.name).map(Vec::as_slice),
                input_option_overrides.get(&input.name),
                &mut required,
                &mut optional,
                &mut hidden,
                &mut required_order,
                &mut optional_order,
                &mut hidden_order,
            );
        }
    } else {
        return Err(NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            "native_node_source_schema_missing",
            format!("{class_type} has no exact source or signed plugin schema"),
        ));
    }
    let (outputs, output_names, output_tooltips) = if let Some(schema) = catalog_schema {
        (
            schema
                .outputs
                .iter()
                .map(|output| Value::String(output.source_type_name.clone()))
                .collect::<Vec<_>>(),
            schema
                .outputs
                .iter()
                .map(|output| {
                    Value::String(
                        output
                            .display_name
                            .as_ref()
                            .or(output.source_name.as_ref())
                            .cloned()
                            .unwrap_or_else(|| output.source_type_name.clone()),
                    )
                })
                .collect::<Vec<_>>(),
            schema
                .outputs
                .iter()
                .map(|output| {
                    output
                        .tooltip
                        .clone()
                        .map(Value::String)
                        .unwrap_or(Value::Null)
                })
                .collect::<Vec<_>>(),
        )
    } else {
        let schema = runtime.source_schema.as_ref().ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "native_node_source_schema_missing",
                format!("{class_type} has no exact output schema"),
            )
        })?;
        (
            schema
                .outputs
                .iter()
                .map(|output| Value::String(output.source_type_name.clone()))
                .collect::<Vec<_>>(),
            schema
                .outputs
                .iter()
                .map(|output| {
                    Value::String(
                        output
                            .display_name
                            .clone()
                            .unwrap_or_else(|| output.name.clone()),
                    )
                })
                .collect::<Vec<_>>(),
            schema
                .outputs
                .iter()
                .map(|output| {
                    output
                        .tooltip
                        .clone()
                        .map(Value::String)
                        .unwrap_or(Value::Null)
                })
                .collect::<Vec<_>>(),
        )
    };
    let source_node_schema = catalog_schema
        .map(|schema| &schema.node)
        .or_else(|| runtime.source_schema.as_ref().map(|schema| &schema.node));
    let source_presentation = catalog_schema.map(|schema| &schema.presentation);
    let zed_schema = if let Some(schema) = catalog_schema {
        serde_json::to_value(schema)
    } else {
        serde_json::to_value(runtime.source_schema.as_ref())
    }
    .map_err(|error| {
        NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            "native_node_source_schema_serialization_failed",
            error.to_string(),
        )
    })?;
    let mut value = json!({
        "input": {"required": required, "optional": optional, "hidden": hidden},
        "input_order": {"required": required_order, "optional": optional_order, "hidden": hidden_order},
        "is_input_list": runtime.inputs.iter().any(|input| input.cardinality != InputMode::Scalar),
        "output": outputs,
        "output_is_list": runtime.outputs.iter().map(|output| output.is_list).collect::<Vec<_>>(),
        "output_name": output_names,
        "output_tooltips": output_tooltips,
        "name": class_type,
        "display_name": source_presentation.and_then(|value| value.display_name.as_deref()).unwrap_or(&presentation.display_name),
        "description": source_presentation.and_then(|value| value.description.as_deref()).unwrap_or(""),
        "python_module": python_module,
        "category": presentation.category,
        "output_node": runtime.output_node,
        "has_intermediate_output": source_node_schema.is_some_and(|schema| schema.has_intermediate_output),
        "deprecated": source_presentation.is_some_and(|value| value.is_deprecated) || presentation.is_deprecated,
        "experimental": source_presentation.is_some_and(|value| value.is_experimental) || presentation.is_experimental,
        "dev_only": source_node_schema.is_some_and(|schema| schema.development_only),
        "api_node": source_node_schema.is_some_and(|schema| schema.api_node),
        "price_badge": source_node_schema.and_then(|schema| schema.price_badge.as_ref()).and_then(project_schema_value),
        "search_aliases": presentation.search_aliases,
        "essentials_category": source_node_schema.and_then(|schema| schema.essentials_category.as_deref()),
        "zed_native_binding": native_binding_projection(disposition, unavailable_reason),
        "zed_source": source.map(native_source_projection),
        "zed_schema": zed_schema,
    });
    if source_node_schema
        .is_some_and(|schema| schema.provenance == NativeSchemaProvenance::SourceV1)
        && let Some(node) = value.as_object_mut()
    {
        if !source_presentation.is_some_and(|value| value.is_deprecated)
            && !presentation.is_deprecated
        {
            node.remove("deprecated");
        }
        if !source_presentation.is_some_and(|value| value.is_experimental)
            && !presentation.is_experimental
        {
            node.remove("experimental");
        }
        if !source_node_schema.is_some_and(|schema| schema.development_only) {
            node.remove("dev_only");
        }
        if !source_node_schema.is_some_and(|schema| schema.api_node) {
            node.remove("api_node");
        }
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn insert_projected_input(
    schema: &NativeInputSchemaMetadata,
    requirement: NativeInputRequirement,
    lazy: bool,
    choice_override: Option<&[Value]>,
    option_override: Option<&Map<String, Value>>,
    required: &mut Map<String, Value>,
    optional: &mut Map<String, Value>,
    hidden: &mut Map<String, Value>,
    required_order: &mut Vec<String>,
    optional_order: &mut Vec<String>,
    hidden_order: &mut Vec<String>,
) {
    let type_name = schema.source_type_names.join("|");
    if requirement == NativeInputRequirement::Hidden {
        hidden.insert(schema.name.clone(), Value::String(type_name));
        hidden_order.push(schema.name.clone());
        return;
    }
    let type_specification = if let Some(choices) = choice_override {
        Value::Array(choices.to_vec())
    } else {
        let choices = schema
            .choices
            .iter()
            .map(project_schema_value)
            .collect::<Option<Vec<_>>>();
        choices
            .filter(|choices| !choices.is_empty())
            .map(Value::Array)
            .unwrap_or(Value::String(type_name))
    };
    let value = json!([
        type_specification,
        project_input_options(schema, lazy, option_override)
    ]);
    match requirement {
        NativeInputRequirement::Required => {
            required_order.push(schema.name.clone());
            required.insert(schema.name.clone(), value);
        }
        NativeInputRequirement::Optional | NativeInputRequirement::Preserved => {
            optional_order.push(schema.name.clone());
            optional.insert(schema.name.clone(), value);
        }
        NativeInputRequirement::Hidden => {}
    }
}

fn project_input_options(
    schema: &NativeInputSchemaMetadata,
    lazy: bool,
    option_override: Option<&Map<String, Value>>,
) -> Value {
    let mut options = Map::new();
    insert_schema_option(&mut options, "default", schema.default.as_ref());
    insert_schema_option(&mut options, "min", schema.minimum.as_ref());
    insert_schema_option(&mut options, "max", schema.maximum.as_ref());
    insert_schema_option(&mut options, "step", schema.step.as_ref());
    insert_optional_string(&mut options, "display_name", schema.display_name.as_deref());
    insert_optional_string(&mut options, "tooltip", schema.tooltip.as_deref());
    insert_true(&mut options, "multiline", schema.multiline);
    insert_true(&mut options, "socketless", schema.socketless);
    insert_optional_string(&mut options, "widgetType", schema.widget_type.as_deref());
    insert_true(&mut options, "forceInput", schema.force_input);
    insert_true(&mut options, "rawLink", schema.raw_link);
    insert_true(&mut options, "advanced", schema.advanced);
    insert_true(&mut options, "lazy", lazy);
    if let Some(upload) = schema.upload {
        let name = match upload {
            NativeUploadKind::Image => "image_upload",
            NativeUploadKind::Audio => "audio_upload",
            NativeUploadKind::Video => "video_upload",
            NativeUploadKind::Model => "model_upload",
            NativeUploadKind::Artifact => "artifact_upload",
        };
        options.insert(name.to_owned(), Value::Bool(true));
    }
    let mut preserved = Map::new();
    for field in &schema.extra {
        if let Some(value) = project_schema_value(&field.value) {
            options.insert(field.name.clone(), value);
        } else {
            preserved.insert(field.name.clone(), schema_value_wire(&field.value));
        }
    }
    if !preserved.is_empty() {
        options.insert(
            "zed_source_expressions".to_owned(),
            Value::Object(preserved),
        );
    }
    if let Some(overrides) = option_override {
        for (name, value) in overrides {
            options.entry(name.clone()).or_insert_with(|| value.clone());
        }
    }
    Value::Object(options)
}

fn insert_schema_option(
    options: &mut Map<String, Value>,
    name: &str,
    value: Option<&NativeSchemaValue>,
) {
    if let Some(value) = value.and_then(project_schema_value) {
        options.insert(name.to_owned(), value);
    }
}

fn insert_optional_string(options: &mut Map<String, Value>, name: &str, value: Option<&str>) {
    if let Some(value) = value {
        options.insert(name.to_owned(), Value::String(value.to_owned()));
    }
}

fn insert_true(options: &mut Map<String, Value>, name: &str, value: bool) {
    if value {
        options.insert(name.to_owned(), Value::Bool(true));
    }
}

fn project_schema_value(value: &NativeSchemaValue) -> Option<Value> {
    match value {
        NativeSchemaValue::Null => Some(Value::Null),
        NativeSchemaValue::Boolean { value } => Some(Value::Bool(*value)),
        NativeSchemaValue::SignedInteger { value } => Some(Value::from(*value)),
        NativeSchemaValue::UnsignedInteger { value } => Some(Value::from(*value)),
        NativeSchemaValue::FiniteDecimal { value } => {
            value.parse::<serde_json::Number>().ok().map(Value::Number)
        }
        NativeSchemaValue::String { value } => Some(Value::String(value.clone())),
        NativeSchemaValue::List { values } => values
            .iter()
            .map(project_schema_value)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        NativeSchemaValue::Object { fields } => fields
            .iter()
            .map(|field| Some((field.name.clone(), project_schema_value(&field.value)?)))
            .collect::<Option<Map<_, _>>>()
            .map(Value::Object),
        NativeSchemaValue::PreservedExpression { .. } => None,
    }
}

fn schema_value_wire(value: &NativeSchemaValue) -> Value {
    match value {
        NativeSchemaValue::PreservedExpression { source, sha256 } => {
            json!({"kind": "preserved_expression", "source": source, "sha256": sha256})
        }
        value => project_schema_value(value).unwrap_or_else(|| {
            serde_json::to_value(value).unwrap_or_else(
                |error| json!({"kind": "serialization_error", "message": error.to_string()}),
            )
        }),
    }
}

fn schema_projection_error(class_type: &str, input_name: &str) -> NativeServiceError {
    NativeServiceError::new(
        NativeServiceErrorKind::Internal,
        "native_node_port_mismatch",
        format!("{class_type} source input `{input_name}` is absent from the execution binding"),
    )
}

fn project_inactive_source_node(source: &ObjectInfoNode) -> Value {
    json!({
        "input": {"required": {}, "optional": {}, "hidden": {}},
        "input_order": {"required": [], "optional": [], "hidden": []},
        "is_input_list": false,
        "output": [],
        "output_is_list": [],
        "output_name": [],
        "output_tooltips": [],
        "name": source.node_identifier,
        "display_name": source.display_name,
        "description": source.inactive_reason.as_deref().unwrap_or(""),
        "python_module": source.source_python_module,
        "category": source.category,
        "output_node": source.output.output_node,
        "has_intermediate_output": false,
        "deprecated": source.availability == "deprecated/dead",
        "experimental": source.availability == "experimental",
        "dev_only": false,
        "api_node": source.source_python_module.starts_with("comfy_api_nodes."),
        "search_aliases": [],
        "essentials_category": null,
        "zed_native_binding": native_binding_projection(
            NativeNodeBindingDisposition::Unavailable,
            source.inactive_reason.as_deref(),
        ),
        "zed_source": native_source_projection(source),
        "zed_schema": {
            "schema_source": source.schema_source,
            "input": source.input,
            "output": source.output,
        },
    })
}

fn project_unbound_source_node(source: &ObjectInfoNode) -> Value {
    let Some(schema) = source.source_schema.as_ref() else {
        return project_inactive_source_node(source);
    };
    let mut required = Map::new();
    let mut optional = Map::new();
    let mut hidden = Map::new();
    let mut required_order = Vec::new();
    let mut optional_order = Vec::new();
    let mut hidden_order = Vec::new();
    for input in &schema.inputs {
        insert_projected_input(
            &input.schema,
            input.requirement,
            false,
            None,
            None,
            &mut required,
            &mut optional,
            &mut hidden,
            &mut required_order,
            &mut optional_order,
            &mut hidden_order,
        );
    }
    let disposition = match source.catalog_status {
        sim_runtime::CatalogNodeStatus::ProviderRequired => {
            NativeNodeBindingDisposition::ProviderRequired
        }
        sim_runtime::CatalogNodeStatus::DescriptorOnly
        | sim_runtime::CatalogNodeStatus::Inactive => NativeNodeBindingDisposition::Unavailable,
    };
    let reason = match disposition {
        NativeNodeBindingDisposition::ProviderRequired => {
            Some("requires a verified native provider")
        }
        NativeNodeBindingDisposition::Unavailable => Some("native implementation not integrated"),
        NativeNodeBindingDisposition::Executable => None,
    };
    let outputs = schema
        .outputs
        .iter()
        .map(|output| Value::String(output.source_type_name.clone()))
        .collect::<Vec<_>>();
    let output_names = schema
        .outputs
        .iter()
        .map(|output| {
            Value::String(
                output
                    .display_name
                    .as_ref()
                    .or(output.source_name.as_ref())
                    .cloned()
                    .unwrap_or_else(|| output.source_type_name.clone()),
            )
        })
        .collect::<Vec<_>>();
    let output_tooltips = schema
        .outputs
        .iter()
        .map(|output| {
            output
                .tooltip
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null)
        })
        .collect::<Vec<_>>();
    json!({
        "input": {"required": required, "optional": optional, "hidden": hidden},
        "input_order": {"required": required_order, "optional": optional_order, "hidden": hidden_order},
        "is_input_list": false,
        "output": outputs,
        "output_is_list": vec![false; schema.outputs.len()],
        "output_name": output_names,
        "output_tooltips": output_tooltips,
        "name": source.node_identifier,
        "display_name": schema.presentation.display_name.as_deref().unwrap_or(&source.display_name),
        "description": schema.presentation.description.as_deref().unwrap_or(""),
        "python_module": source.source_python_module,
        "category": source.category,
        "output_node": source.output.output_node,
        "has_intermediate_output": schema.node.has_intermediate_output,
        "deprecated": schema.presentation.is_deprecated,
        "experimental": schema.presentation.is_experimental,
        "dev_only": schema.node.development_only,
        "api_node": schema.node.api_node,
        "price_badge": schema.node.price_badge.as_ref().and_then(project_schema_value),
        "search_aliases": [],
        "essentials_category": schema.node.essentials_category,
        "zed_native_binding": native_binding_projection(disposition, reason),
        "zed_source": native_source_projection(source),
        "zed_schema": schema,
    })
}

fn native_binding_projection(
    disposition: NativeNodeBindingDisposition,
    unavailable_reason: Option<&str>,
) -> Value {
    let disposition = match disposition {
        NativeNodeBindingDisposition::Executable => "executable",
        NativeNodeBindingDisposition::ProviderRequired => "provider_required",
        NativeNodeBindingDisposition::Unavailable => "unavailable",
    };
    json!({
        "disposition": disposition,
        "reason": unavailable_reason,
    })
}

fn native_source_projection(source: &ObjectInfoNode) -> Value {
    json!({
        "feature_id": source.feature_id,
        "availability": source.availability,
        "catalog_status": source.catalog_status,
        "inactive_reason": source.inactive_reason,
    })
}

fn lock<'a, T>(
    mutex: &'a Mutex<T>,
    code: &'static str,
) -> Result<MutexGuard<'a, T>, NativeServiceError> {
    mutex.lock().map_err(|_| {
        NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            code,
            "native service state could not be accessed",
        )
    })
}

fn presentation_error(error: impl std::fmt::Display) -> NativeServiceError {
    NativeServiceError::new(
        NativeServiceErrorKind::Conflict,
        "native_execution_state_conflict",
        error.to_string(),
    )
}

fn asset_error(error: impl std::fmt::Display) -> NativeServiceError {
    NativeServiceError::new(
        NativeServiceErrorKind::Conflict,
        "native_asset_operation_failed",
        error.to_string(),
    )
}

fn capability_unavailable<T>(
    feature_id: &str,
    owner: &str,
    reason: &str,
) -> Result<T, NativeServiceError> {
    Err(NativeServiceError::new(
        NativeServiceErrorKind::Unavailable,
        "native_route_capability_unavailable",
        format!("{feature_id} requires {owner}: {reason}"),
    ))
}

fn empty_response(status: u16) -> NativeServiceResponse {
    NativeServiceResponse {
        status,
        content_type: "application/octet-stream".to_owned(),
        headers: Default::default(),
        body: HttpBody::Empty,
    }
}

fn capability_owner(request: &NativeServiceRequest) -> &'static str {
    capability_owner_path(request.route.canonical_path.as_str())
}

fn capability_owner_path(path: &str) -> &'static str {
    if path.starts_with("/api/workflows") || path.starts_with("/userdata") {
        "native_profile_document_service"
    } else if path.starts_with("/api/assets") || path == "/api/tags" || path == "/view" {
        "native_asset_reference_service"
    } else if path.starts_with("/settings") {
        "native_settings_service"
    } else if path.starts_with("/extensions") {
        "rust_wasm_extension_host"
    } else {
        "native_compatibility_service"
    }
}

fn supports_native_route(method: HttpMethod, path: &str, has_assets: bool) -> bool {
    matches!(
        (method, path),
        (HttpMethod::Get, "/features")
            | (HttpMethod::Get | HttpMethod::Post, "/prompt")
            | (HttpMethod::Get | HttpMethod::Post, "/queue")
            | (HttpMethod::Get | HttpMethod::Post, "/history")
            | (HttpMethod::Get, "/history/{prompt_id}")
            | (HttpMethod::Post, "/interrupt")
            | (HttpMethod::Get, "/api/jobs")
            | (HttpMethod::Post, "/api/jobs/cancel")
            | (HttpMethod::Get, "/api/jobs/{job_id}")
            | (HttpMethod::Post, "/api/jobs/{job_id}/cancel")
            | (HttpMethod::Get, "/object_info")
            | (HttpMethod::Get, "/object_info/{node_class}")
            | (HttpMethod::Get, "/health")
            | (HttpMethod::Get, "/system_stats")
            | (HttpMethod::Get, "/api/history_v2")
            | (HttpMethod::Get, "/api/history_v2/{prompt_id}")
            | (HttpMethod::Get, "/api/job/{job_id}/status")
    ) || has_assets
        && matches!(
            (method, path),
            (HttpMethod::Get, "/models")
                | (HttpMethod::Get, "/models/{folder}")
                | (HttpMethod::Get, "/experiment/models")
                | (HttpMethod::Get, "/internal/files/{directory_type}")
                | (HttpMethod::Get, "/internal/folder_paths")
        )
}

fn required_json_object(
    request: &NativeServiceRequest,
) -> Result<&Map<String, Value>, NativeServiceError> {
    request
        .json_body
        .as_ref()
        .and_then(Value::as_object)
        .ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "json_object_required",
                "request body must be a JSON object",
            )
        })
}

fn optional_boolean(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<bool>, NativeServiceError> {
    object
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Invalid,
                    "invalid_boolean_field",
                    format!("{field} must be a boolean"),
                )
            })
        })
        .transpose()
}

fn optional_string_array(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, NativeServiceError> {
    match object.get(field) {
        Some(_) => required_string_array(object, field),
        None => Ok(Vec::new()),
    }
}

fn required_string_array(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, NativeServiceError> {
    let values = object.get(field).and_then(Value::as_array).ok_or_else(|| {
        NativeServiceError::new(
            NativeServiceErrorKind::Invalid,
            "invalid_string_array_field",
            format!("{field} must be an array of strings"),
        )
    })?;
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Invalid,
                    "invalid_string_array_field",
                    format!("{field} must contain only strings"),
                )
            })
        })
        .collect()
}

fn query_first<'a>(request: &'a NativeServiceRequest, name: &str) -> Option<&'a str> {
    request
        .query
        .get(name)
        .and_then(|values| values.first())
        .map(String::as_str)
}

fn query_usize(
    request: &NativeServiceRequest,
    name: &str,
) -> Result<Option<usize>, NativeServiceError> {
    query_first(request, name)
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Invalid,
                    "invalid_query_integer",
                    format!("{name} must be a non-negative integer"),
                )
            })
        })
        .transpose()
}

fn required_path<'a>(
    request: &'a NativeServiceRequest,
    name: &str,
) -> Result<&'a str, NativeServiceError> {
    request
        .route
        .path_parameters
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "path_parameter_required",
                format!("path parameter {name} is required"),
            )
        })
}

fn parse_prompt_id(value: &str) -> Result<PromptId, NativeServiceError> {
    let prompt_id: PromptId =
        serde_json::from_value(Value::String(value.to_owned())).map_err(|_| {
            NativeServiceError::new(
                NativeServiceErrorKind::Invalid,
                "invalid_prompt_id",
                "prompt or job identity must be a canonical UUID",
            )
        })?;
    if prompt_id.0.to_string() != value {
        return Err(NativeServiceError::new(
            NativeServiceErrorKind::Invalid,
            "invalid_prompt_id",
            "prompt or job identity must use canonical lowercase UUID formatting",
        ));
    }
    Ok(prompt_id)
}

fn interrupt_prompt_id(
    request: &NativeServiceRequest,
) -> Result<Option<PromptId>, NativeServiceError> {
    request
        .json_body
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|body| body.get("prompt_id"))
        .filter(|value| !value.is_null())
        .map(|value| {
            value.as_str().ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Invalid,
                    "invalid_prompt_id",
                    "prompt_id must be a UUID string",
                )
            })
        })
        .transpose()?
        .map(parse_prompt_id)
        .transpose()
}

fn prompt_submission_response(
    plan: &CompiledPlan,
    attempt_id: sim_runtime::AttemptId,
) -> NativeServiceResponse {
    NativeServiceResponse::json(
        200,
        json!({
            "prompt_id": plan.prompt_id,
            "number": plan.prompt_number,
            "node_errors": {},
            "attempt_id": attempt_id,
        }),
    )
}

fn request_id(
    request: &NativeServiceRequest,
    operation: &str,
    ordinal: usize,
) -> Result<RequestId, NativeServiceError> {
    let identity = match &request.mutation_identity {
        MutationIdentity::IdempotencyKey(key) => format!("idempotency:{key}"),
        MutationIdentity::DurableAttempt(attempt) => format!("attempt:{attempt}"),
        MutationIdentity::Untracked => format!(
            "untracked:{}:{}:{}",
            request.route.canonical_feature_id,
            operation,
            Sha256::digest(&request.body)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ),
    };
    let digest = Sha256::digest(
        format!(
            "{}:{identity}:{operation}:{ordinal}",
            request.route.canonical_feature_id
        )
        .as_bytes(),
    );
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let value = format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    );
    serde_json::from_value(Value::String(value)).map_err(|error| {
        NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            "request_identity_generation_failed",
            error.to_string(),
        )
    })
}

fn queue_tuple(
    plan: &CompiledPlan,
    enqueue_sequence: Option<u64>,
) -> Result<Value, NativeServiceError> {
    let mut prompt = Map::new();
    for (node_id, node) in &plan.nodes {
        let inputs = node
            .inputs
            .iter()
            .map(|(name, binding)| {
                let value = match binding {
                    InputBinding::Literal { value } => native_literal_json(value)?,
                    InputBinding::Link {
                        source,
                        output_index,
                        ..
                    } => json!([source.0, output_index]),
                };
                Ok((name.clone(), value))
            })
            .collect::<Result<Map<_, _>, NativeServiceError>>()?;
        prompt.insert(
            node_id.0.clone(),
            json!({"class_type": node.class_type, "inputs": inputs}),
        );
    }
    let number = plan
        .prompt_number
        .or_else(|| enqueue_sequence.map(|sequence| sequence as f64))
        .unwrap_or_default();
    let outputs = plan
        .output_nodes
        .iter()
        .map(|node| node.0.clone())
        .collect::<Vec<_>>();
    Ok(json!([
        number,
        plan.prompt_id,
        Value::Object(prompt),
        plan.extra_data,
        outputs
    ]))
}

fn native_literal_json(value: &NativeValue) -> Result<Value, NativeServiceError> {
    match value {
        NativeValue::Primitive { value } => Ok(match value {
            NativePrimitive::Null => Value::Null,
            NativePrimitive::Boolean(value) => Value::Bool(*value),
            NativePrimitive::Integer(value) => Value::Number((*value).into()),
            NativePrimitive::UnsignedInteger(value) => Value::Number((*value).into()),
            NativePrimitive::Number(value) => serde_json::Number::from_f64(*value)
                .map(Value::Number)
                .ok_or_else(|| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "native_prompt_literal_invalid",
                        "compiled prompt contains a non-finite numeric literal",
                    )
                })?,
            NativePrimitive::String(value) => Value::String(value.clone()),
        }),
        NativeValue::List { values } => values
            .iter()
            .map(native_literal_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        NativeValue::PreservedUnknown { value, .. } => Ok(value.clone()),
        NativeValue::Handle { .. } => Err(NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            "native_prompt_handle_not_serializable",
            "compiled prompt contains a process-local handle literal",
        )),
    }
}

fn history_record(
    attempt: &AttemptPresentation,
    plan: &CompiledPlan,
) -> Result<Value, NativeServiceError> {
    let mut outputs = Map::new();
    for output in &attempt.outputs {
        let node = outputs
            .entry(output.node_id.0.clone())
            .or_insert_with(|| json!({"zed_native_outputs": []}));
        let list = node
            .get_mut("zed_native_outputs")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "history_output_projection_failed",
                    "native output projection is not an array",
                )
            })?;
        list.push(to_value(output)?);
    }
    Ok(json!({
        "prompt": queue_tuple(plan, None)?,
        "outputs": outputs,
        "status": {
            "status_str": attempt_state_name(attempt.state),
            "completed": attempt.state == AttemptState::Succeeded,
            "messages": [],
        },
        "meta": {
            "attempt_id": attempt.attempt_id,
            "retry_of": attempt.retry_of,
            "failure": attempt.failure,
            "finished_at": attempt.finished_at,
        }
    }))
}

fn job_record(attempt: &AttemptPresentation) -> Result<Value, NativeServiceError> {
    Ok(json!({
        "job_id": attempt.prompt_id,
        "attempt_id": attempt.attempt_id,
        "status": job_status(attempt.state),
        "created_at": attempt.created_at,
        "completed_at": attempt.finished_at,
        "outputs": attempt.outputs,
        "error": attempt.failure,
    }))
}

fn to_value(value: &impl Serialize) -> Result<Value, NativeServiceError> {
    serde_json::to_value(value).map_err(|error| {
        NativeServiceError::new(
            NativeServiceErrorKind::Internal,
            "native_response_serialization_failed",
            error.to_string(),
        )
    })
}

fn attempt_state_name(state: AttemptState) -> &'static str {
    match state {
        AttemptState::Queued => "queued",
        AttemptState::Running => "running",
        AttemptState::Cancelling => "cancelling",
        AttemptState::Succeeded => "success",
        AttemptState::Failed => "error",
        AttemptState::Cancelled => "cancelled",
        AttemptState::Interrupted => "interrupted",
    }
}

fn job_status(state: AttemptState) -> &'static str {
    match state {
        AttemptState::Queued => "pending",
        AttemptState::Running | AttemptState::Cancelling => "in_progress",
        AttemptState::Succeeded => "completed",
        AttemptState::Failed | AttemptState::Cancelled | AttemptState::Interrupted => "failed",
    }
}

fn execution_duration_milliseconds(attempt: &AttemptPresentation) -> i64 {
    attempt
        .finished_at
        .map(|finished_at| {
            finished_at
                .signed_duration_since(attempt.created_at)
                .num_milliseconds()
                .max(0)
        })
        .unwrap_or(i64::MAX)
}

fn parse_asset_namespace(value: &str) -> Result<AssetNamespace, NativeServiceError> {
    let canonical = match value {
        "temporary" => "temp",
        "models" => "model",
        "plugins" => "plugin",
        value => value,
    };
    AssetNamespace::from_locator_type(canonical).map_err(|_| {
        NativeServiceError::new(
            NativeServiceErrorKind::NotFound,
            "asset_namespace_not_found",
            "the requested typed asset namespace is not configured",
        )
    })
}

fn normalized_relative_path(record: &AssetRecord) -> String {
    record
        .identity
        .relative_path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn model_folders(records: &[AssetRecord]) -> Vec<String> {
    records
        .iter()
        .filter_map(|record| {
            record.identity.relative_path.components().next().and_then(
                |component| match component {
                    Component::Normal(value) => value.to_str().map(str::to_owned),
                    _ => None,
                },
            )
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn model_paths(records: &[AssetRecord], folder: &str) -> Vec<String> {
    records
        .iter()
        .filter_map(|record| {
            let path = normalized_relative_path(record);
            path.strip_prefix(folder)
                .and_then(|relative| relative.strip_prefix('/'))
                .map(str::to_owned)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn experimental_model_folders(records: &[AssetRecord]) -> Vec<Value> {
    model_folders(records)
        .into_iter()
        .map(|folder| {
            let logical_roots = records
                .iter()
                .filter(|record| {
                    record
                        .identity
                        .relative_path
                        .components()
                        .next()
                        .is_some_and(|component| component.as_os_str() == folder.as_str())
                })
                .map(|record| {
                    record
                        .identity
                        .root_id
                        .as_deref()
                        .unwrap_or(record.identity.namespace.locator_type())
                        .to_owned()
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(Value::String)
                .collect::<Vec<_>>();
            json!({
                "name": folder,
                "folders": logical_roots,
            })
        })
        .collect()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AcceptingExecutionController;

#[cfg(test)]
impl ExecutionController for AcceptingExecutionController {
    fn accept(
        &self,
        _command: &ExecutionControlCommand,
        _assigned_attempt_id: Option<sim_runtime::AttemptId>,
    ) -> Result<(), sim_runtime::ExecutionFailure> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::http::{HttpRequest, match_http_route};
    use bytes::Bytes;
    use sim_runtime::{
        PermissionPolicy, authorize_native_api_asset_reader, open_native_profile_asset_service,
    };
    use sim_types::CancellationToken;
    use std::{
        collections::BTreeMap,
        error::Error,
        fs,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    struct PresentationlessNode;

    impl sim_runtime::NativeNode for PresentationlessNode {
        fn class_type(&self) -> &str {
            "Presentationless"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            _context: sim_runtime::NodeContext,
            _inputs: BTreeMap<String, sim_runtime::NativeValue>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<sim_runtime::NodeOutcome, sim_runtime::NodeFailure>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async {
                Ok(sim_runtime::NodeOutcome::Values {
                    outputs: Vec::new(),
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    #[derive(Clone, Debug)]
    struct CountingExecutionController {
        calls: Arc<AtomicUsize>,
    }

    impl ExecutionController for CountingExecutionController {
        fn accept(
            &self,
            _command: &ExecutionControlCommand,
            _assigned_attempt_id: Option<sim_runtime::AttemptId>,
        ) -> Result<(), sim_runtime::ExecutionFailure> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn unavailable_registry(
        class_type: &str,
        reason: &str,
    ) -> Result<NativeNodeRegistry, NativeServiceError> {
        catalog_status_registry(class_type, reason, None)
    }

    fn provider_required_registry(
        class_type: &str,
        provider: &str,
        reason: &str,
    ) -> Result<NativeNodeRegistry, NativeServiceError> {
        catalog_status_registry(class_type, reason, Some(provider))
    }

    fn catalog_status_registry(
        class_type: &str,
        reason: &str,
        provider: Option<&str>,
    ) -> Result<NativeNodeRegistry, NativeServiceError> {
        let source = NodeRegistry::built_in().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_source_registry_invalid",
                error.to_string(),
            )
        })?;
        let catalog = source.descriptor(class_type).ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_source_descriptor_missing",
                class_type,
            )
        })?;
        let object_info = ObjectInfoRegistry::from_node_registry(&source).map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_source_projection_invalid",
                error.to_string(),
            )
        })?;
        let source_node = object_info.nodes().get(class_type).ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_source_projection_missing",
                class_type,
            )
        })?;
        let source_schema = source_node.source_schema.as_ref();
        let category = if catalog.category == "(empty root category declared by source)" {
            String::new()
        } else {
            catalog.category.clone()
        };
        let inputs = source_schema
            .map(|schema| {
                schema
                    .inputs
                    .iter()
                    .map(|input| {
                        Ok(sim_runtime::NativeInputDescriptor {
                            name: input.schema.name.clone(),
                            accepted_types: sim_runtime::NativeTypeUnion::new([
                                sim_runtime::NativeValueType::Any,
                            ])?,
                            required: input.requirement == NativeInputRequirement::Required,
                            hidden: input.requirement == NativeInputRequirement::Hidden,
                            lazy: false,
                            cardinality: sim_runtime::NativePortCardinality::Scalar,
                            allows_literal: true,
                        })
                    })
                    .collect::<Result<Vec<_>, sim_runtime::NativeNodeContractError>>()
            })
            .transpose()
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_source_inputs_invalid",
                    error.to_string(),
                )
            })?
            .unwrap_or_default();
        let outputs = source_schema
            .map(|schema| {
                schema
                    .outputs
                    .iter()
                    .map(|output| sim_runtime::RuntimeOutputDescriptor {
                        name: output
                            .source_name
                            .clone()
                            .unwrap_or_else(|| output.source_type_name.clone()),
                        produced_type: sim_runtime::NativeValueType::Any,
                        is_list: false,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let descriptor = sim_runtime::NativeNodeDescriptor {
            schema_version: 1,
            class_type: class_type.to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: None,
            inputs,
            dynamic_inputs: Vec::new(),
            outputs,
            output_node: catalog.output_node,
            effect: if provider.is_some() {
                sim_runtime::NativeEffectClass::Provider
            } else {
                sim_runtime::NativeEffectClass::Pure
            },
            cache: sim_runtime::NativeCachePolicy::InputIdentity,
        };
        let presentation = sim_runtime::NativeNodePresentation {
            display_name: catalog.display_name.clone(),
            category,
            description: String::new(),
            output_names: descriptor
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        };
        let binding = if let Some(provider) = provider {
            sim_runtime::NativeNodeBinding::ProviderRequired {
                feature_id: catalog.feature_id.clone(),
                descriptor,
                presentation,
                provider: provider.to_owned(),
                reason: reason.to_owned(),
            }
        } else {
            sim_runtime::NativeNodeBinding::Unavailable {
                feature_id: catalog.feature_id.clone(),
                descriptor,
                presentation,
                reason: reason.to_owned(),
            }
        };
        let mut registry = NativeNodeRegistry::default();
        registry
            .register_native_bindings([binding])
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_native_binding_invalid",
                    error.to_string(),
                )
            })?;
        Ok(registry)
    }

    fn profile(value: &str) -> Result<ProfileId, NativeServiceError> {
        serde_json::from_value(Value::String(value.to_owned())).map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_profile_invalid",
                error.to_string(),
            )
        })
    }

    fn authority(profile_id: ProfileId) -> NativeRequestAuthority {
        NativeRequestAuthority {
            profile_id: profile_id.0.to_string(),
            principal: "native-test".to_owned(),
            scopes: BTreeSet::from(["api:read".to_owned(), "api:write".to_owned()]),
            plugin_id: None,
            plugin_digest: None,
        }
    }

    fn request(
        method: HttpMethod,
        path: &str,
        operation: NativeServiceOperation,
        profile_id: ProfileId,
        body: Option<Value>,
    ) -> Result<NativeServiceRequest, NativeServiceError> {
        let route = match_http_route(method, path)
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_catalog_invalid",
                    error.to_string(),
                )
            })?
            .ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_route_missing",
                    path,
                )
            })?;
        let encoded = body
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_body_invalid",
                    error.to_string(),
                )
            })?
            .unwrap_or_default();
        Ok(NativeServiceRequest {
            route,
            operation,
            query: BTreeMap::new(),
            headers: BTreeMap::new(),
            body: Bytes::from(encoded),
            json_body: body,
            range: None,
            mutation_identity: MutationIdentity::IdempotencyKey("native-test-key".to_owned()),
            authority: Some(authority(profile_id)),
        })
    }

    fn body_json(response: NativeServiceResponse) -> Result<Value, NativeServiceError> {
        match response.body {
            HttpBody::Json(value) => Ok(value),
            _ => Err(NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_response_not_json",
                "expected JSON response",
            )),
        }
    }

    fn with_mutation_key(mut request: NativeServiceRequest, key: &str) -> NativeServiceRequest {
        request.mutation_identity = MutationIdentity::IdempotencyKey(key.to_owned());
        request
    }

    #[test]
    fn native_prompt_submission_mutates_the_shared_execution_service()
    -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let mut presentation =
            sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                .map_err(presentation_error)?;
        presentation
            .initialize_profile(
                profile_id,
                sim_runtime::ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(presentation_error)?;
        let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
        let services = NativeRuntimeHttpServices::native_image(
            profile_id,
            presentation.clone(),
            Arc::new(AcceptingExecutionController),
        )?;
        assert!(Arc::ptr_eq(&services.presentation(), &presentation));
        let prompt = json!({
            "prompt_id": "00000000-0000-0000-0000-000000000101",
            "number": 7,
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
            }
        });
        let response = services.dispatch(request(
            HttpMethod::Post,
            "/prompt",
            NativeServiceOperation::SubmitPrompt,
            profile_id,
            Some(prompt),
        )?)?;
        assert_eq!(response.status, 200);
        let body = body_json(response)?;
        assert_eq!(body["prompt_id"], "00000000-0000-0000-0000-000000000101");
        let snapshot = services.snapshot()?;
        assert_eq!(snapshot.queue.len(), 1);
        assert_eq!(snapshot.queue[0].prompt_id.0.to_string(), body["prompt_id"]);
        let projection = services.status_projection()?.ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_status_projection_missing",
                "native runtime services did not expose their queue status projection",
            )
        })?;
        assert_eq!(projection["exec_info"]["queue_remaining"], 1);
        assert_eq!(projection["exec_info"]["running"], 0);
        Ok(())
    }

    #[test]
    fn canonical_prompt_receipt_reconciles_and_absent_receipt_is_not_applied()
    -> Result<(), Box<dyn Error>> {
        let profile_id = profile("00000000-0000-0000-0000-000000000301")?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let submitted = with_mutation_key(
            request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(json!({
                    "prompt_id": "00000000-0000-0000-0000-000000000302",
                    "number": 17,
                    "prompt": {
                        "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
                    }
                })),
            )?,
            "reconciled-prompt",
        );
        let dispatched = body_json(services.dispatch(submitted.clone())?)?;
        match services.reconcile_mutation(&submitted)? {
            NativeMutationReconciliation::Committed(response) => {
                assert_eq!(body_json(response)?, dispatched);
            }
            reconciliation => {
                return Err(format!(
                    "expected committed prompt reconciliation, found {reconciliation:?}"
                )
                .into());
            }
        }
        assert_eq!(services.snapshot()?.queue.len(), 1);

        let not_applied = with_mutation_key(
            request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(json!({
                    "prompt_id": "00000000-0000-0000-0000-000000000303",
                    "prompt": {
                        "1": {"class_type": "LoadImage", "inputs": {"image": "other.png"}},
                        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
                    }
                })),
            )?,
            "never-dispatched-prompt",
        );
        assert!(matches!(
            services.reconcile_mutation(&not_applied)?,
            NativeMutationReconciliation::NotApplied
        ));
        assert_eq!(services.snapshot()?.queue.len(), 1);
        Ok(())
    }

    #[test]
    fn canonical_delete_cancel_and_clear_receipts_reconstruct_service_responses()
    -> Result<(), Box<dyn Error>> {
        let profile_id = profile("00000000-0000-0000-0000-000000000311")?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let prompt_body = |prompt_id: &str| {
            json!({
                "prompt_id": prompt_id,
                "prompt": {
                    "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                    "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
                }
            })
        };

        let deleted_prompt = "00000000-0000-0000-0000-000000000312";
        services.dispatch(with_mutation_key(
            request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(prompt_body(deleted_prompt)),
            )?,
            "delete-seed",
        ))?;
        let queue_delete = with_mutation_key(
            request(
                HttpMethod::Post,
                "/queue",
                NativeServiceOperation::QueueMutation,
                profile_id,
                Some(json!({"delete": [deleted_prompt]})),
            )?,
            "queue-delete",
        );
        services.dispatch(queue_delete.clone())?;
        assert!(matches!(
            services.reconcile_mutation(&queue_delete)?,
            NativeMutationReconciliation::Committed(NativeServiceResponse {
                status: 200,
                body: HttpBody::Empty,
                ..
            })
        ));

        let history_delete = with_mutation_key(
            request(
                HttpMethod::Post,
                "/history",
                NativeServiceOperation::HistoryMutation,
                profile_id,
                Some(json!({"delete": [deleted_prompt]})),
            )?,
            "history-delete",
        );
        services.dispatch(history_delete.clone())?;
        assert!(matches!(
            services.reconcile_mutation(&history_delete)?,
            NativeMutationReconciliation::Committed(NativeServiceResponse {
                status: 200,
                body: HttpBody::Empty,
                ..
            })
        ));

        let cancelled_prompt = "00000000-0000-0000-0000-000000000313";
        services.dispatch(with_mutation_key(
            request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(prompt_body(cancelled_prompt)),
            )?,
            "cancel-seed",
        ))?;
        let job_cancel = with_mutation_key(
            request(
                HttpMethod::Post,
                &format!("/api/jobs/{cancelled_prompt}/cancel"),
                NativeServiceOperation::Jobs,
                profile_id,
                None,
            )?,
            "job-cancel",
        );
        assert_eq!(
            body_json(services.dispatch(job_cancel.clone())?)?["cancelled"],
            true
        );
        match services.reconcile_mutation(&job_cancel)? {
            NativeMutationReconciliation::Committed(response) => {
                assert_eq!(body_json(response)?["cancelled"], true);
            }
            reconciliation => {
                return Err(format!(
                    "expected committed job cancellation, found {reconciliation:?}"
                )
                .into());
            }
        }

        let history_clear = with_mutation_key(
            request(
                HttpMethod::Post,
                "/history",
                NativeServiceOperation::HistoryMutation,
                profile_id,
                Some(json!({"clear": true})),
            )?,
            "history-clear",
        );
        services.dispatch(history_clear.clone())?;
        assert!(matches!(
            services.reconcile_mutation(&history_clear)?,
            NativeMutationReconciliation::Committed(NativeServiceResponse {
                status: 200,
                body: HttpBody::Empty,
                ..
            })
        ));

        let cleared_prompt = "00000000-0000-0000-0000-000000000314";
        services.dispatch(with_mutation_key(
            request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(prompt_body(cleared_prompt)),
            )?,
            "clear-seed",
        ))?;
        let queue_clear = with_mutation_key(
            request(
                HttpMethod::Post,
                "/queue",
                NativeServiceOperation::QueueMutation,
                profile_id,
                Some(json!({"clear": true})),
            )?,
            "queue-clear",
        );
        services.dispatch(queue_clear.clone())?;
        assert!(matches!(
            services.reconcile_mutation(&queue_clear)?,
            NativeMutationReconciliation::Committed(NativeServiceResponse {
                status: 200,
                body: HttpBody::Empty,
                ..
            })
        ));
        Ok(())
    }

    pub(crate) fn validate_native_object_info_fixture() -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let bound_registry = sim_runtime::native_image_registry_projection().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_native_registry_invalid",
                error.to_string(),
            )
        })?;
        let load_image_descriptor =
            bound_registry
                .descriptor("LoadImage")
                .cloned()
                .ok_or_else(|| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "test_native_descriptor_missing",
                        "LoadImage is absent from the native registry fixture",
                    )
                })?;
        for class_type in [
            "LoadImage",
            "ImageScale",
            "ImageInvert",
            "SaveImage",
            "PreviewImage",
        ] {
            if bound_registry.node(class_type).is_none() {
                return Err(NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_native_implementation_missing",
                    format!("{class_type} has metadata but no executable node"),
                ));
            }
        }
        let mut unbound_registry = NativeNodeRegistry::default();
        unbound_registry
            .register_descriptor(load_image_descriptor)
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_native_descriptor_invalid",
                    error.to_string(),
                )
            })?;
        if NativeRuntimeHttpServices::new(
            profile_id,
            sim_runtime::ExecutionPresentationOwner::ephemeral(
                sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                    .map_err(presentation_error)?,
            ),
            Arc::new(AcceptingExecutionController),
            unbound_registry,
        )
        .is_ok()
        {
            return Err(NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_unbound_registry_accepted",
                "object-info accepted a descriptor without an executable node",
            ));
        }
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let response = services.dispatch(request(
            HttpMethod::Get,
            "/object_info",
            NativeServiceOperation::NodeCatalog,
            profile_id,
            None,
        )?)?;
        let body = body_json(response)?;
        assert_eq!(body.as_object().map(Map::len), Some(801));
        assert_eq!(body["LoadImage"]["python_module"], "nodes");
        assert_eq!(
            body["LoadImage"]["zed_source"]["catalog_status"],
            "descriptor-only"
        );
        assert!(body["LoadImage"]["zed_source"]["feature_id"].is_string());
        assert_eq!(
            body["LoadImage"]["input"]["required"]["image"][0],
            json!([])
        );
        assert_eq!(
            body["LoadImage"]["input"]["required"]["image"][1]["image_upload"],
            true
        );
        assert_eq!(body["SaveImage"]["output_node"], true);
        assert_eq!(
            body["SaveImage"]["description"],
            "Saves the input images to your ComfyUI output directory."
        );
        assert_eq!(
            body["SaveImage"]["input"]["required"]["filename_prefix"][1]["default"],
            "ComfyUI"
        );
        assert_eq!(
            body["SaveImage"]["input_order"]["hidden"],
            json!(["prompt", "extra_pnginfo"])
        );
        assert_eq!(body["ImageScale"]["display_name"], "Upscale Image");
        assert_eq!(body["ImageScale"]["category"], "image/upscaling");
        assert_eq!(
            body["ImageScale"]["input"]["required"]["upscale_method"][0],
            json!(["nearest-exact", "bilinear", "area", "bicubic", "lanczos"])
        );
        assert_eq!(
            body["ImageScale"]["input"]["required"]["width"][1],
            json!({"default": 512, "min": 0, "max": 16384, "step": 1})
        );
        assert_eq!(
            body["ImageScale"]["input"]["required"]["crop"][0],
            json!(["disabled", "center"])
        );
        assert_eq!(
            body["ImageScale"]["search_aliases"],
            json!([
                "resize",
                "resize image",
                "scale image",
                "image resize",
                "zoom",
                "zoom in",
                "change size"
            ])
        );
        assert_eq!(body["ImageScale"]["essentials_category"], "Image Tools");
        assert_eq!(body["ImageInvert"]["display_name"], "Invert Image Colors");
        assert_eq!(body["ImageInvert"]["category"], "image/color");
        assert_eq!(body["ImageInvert"]["description"], "");
        assert_eq!(body["ImageInvert"]["has_intermediate_output"], false);
        assert!(body["ImageScale"].get("experimental").is_none());
        let open_ai = &body["OpenAIGPTImage1"];
        assert_eq!(open_ai["deprecated"], true);
        assert_eq!(open_ai["api_node"], true);
        assert_eq!(open_ai["input"]["required"]["prompt"][1]["default"], "");
        assert_eq!(open_ai["input"]["required"]["prompt"][1]["multiline"], true);
        assert_eq!(
            open_ai["input"]["optional"]["quality"][0],
            json!(["low", "medium", "high"])
        );
        assert_eq!(open_ai["input"]["optional"]["custom_width"][1]["min"], 1024);
        assert_eq!(open_ai["input"]["optional"]["custom_width"][1]["max"], 3840);
        assert_eq!(open_ai["input"]["optional"]["custom_width"][1]["step"], 16);
        assert_eq!(
            open_ai["zed_schema"]["inputs"][1]["maximum"]["kind"],
            "preserved_expression"
        );
        assert_eq!(body["AddNoise"]["experimental"], true);
        Ok(())
    }

    #[test]
    fn native_object_info_projects_the_complete_source_registry() -> Result<(), NativeServiceError>
    {
        validate_native_object_info_fixture()
    }

    #[test]
    fn unavailable_object_info_uses_canonical_source_and_rejects_submission_before_controller()
    -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let calls = Arc::new(AtomicUsize::new(0));
        let controller = Arc::new(CountingExecutionController {
            calls: calls.clone(),
        });
        let mut presentation =
            sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                .map_err(presentation_error)?;
        presentation
            .initialize_profile(
                profile_id,
                sim_runtime::ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(presentation_error)?;
        let services = NativeRuntimeHttpServices::new(
            profile_id,
            sim_runtime::ExecutionPresentationOwner::ephemeral(presentation),
            controller,
            unavailable_registry(
                "AutogrowNamesTestNode",
                "inactive in the canonical source registry",
            )?,
        )?;

        let catalog = body_json(services.dispatch(request(
            HttpMethod::Get,
            "/object_info/AutogrowNamesTestNode",
            NativeServiceOperation::NodeCatalog,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(
            catalog["AutogrowNamesTestNode"]["python_module"],
            "comfy_extras.nodes_logic"
        );
        assert_eq!(
            catalog["AutogrowNamesTestNode"]["zed_native_binding"]["disposition"],
            "unavailable"
        );
        assert_eq!(
            catalog["AutogrowNamesTestNode"]["zed_native_binding"]["reason"],
            catalog["AutogrowNamesTestNode"]["zed_source"]["inactive_reason"]
        );
        assert_eq!(
            catalog["AutogrowNamesTestNode"]["zed_source"]["catalog_status"],
            "inactive"
        );
        assert!(catalog["AutogrowNamesTestNode"]["zed_source"]["inactive_reason"].is_string());
        assert!(catalog["AutogrowNamesTestNode"]["zed_source"]["availability"].is_string());
        assert!(catalog["AutogrowNamesTestNode"]["zed_source"]["feature_id"].is_string());
        assert!(catalog["AutogrowNamesTestNode"]["zed_schema"]["schema_source"].is_string());

        let error = services
            .dispatch(request(
                HttpMethod::Post,
                "/prompt",
                NativeServiceOperation::SubmitPrompt,
                profile_id,
                Some(json!({
                    "prompt": {
                        "1": {"class_type": "AutogrowNamesTestNode", "inputs": {}}
                    }
                })),
            )?)
            .expect_err("an unavailable node must fail before controller dispatch");
        assert_eq!(error.kind, NativeServiceErrorKind::Invalid);
        assert_eq!(error.code, "prompt_validation_failed");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(services.snapshot()?.queue.is_empty());

        let mut provider_presentation =
            sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                .map_err(presentation_error)?;
        provider_presentation
            .initialize_profile(
                profile_id,
                sim_runtime::ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(presentation_error)?;
        let provider_services = NativeRuntimeHttpServices::new(
            profile_id,
            sim_runtime::ExecutionPresentationOwner::ephemeral(provider_presentation),
            Arc::new(CountingExecutionController {
                calls: calls.clone(),
            }),
            provider_required_registry(
                "BeebleSwitchXImageEdit",
                "comfy_api_nodes.beeble",
                "requires a verified native provider",
            )?,
        )?;
        let provider_catalog = body_json(provider_services.dispatch(request(
            HttpMethod::Get,
            "/object_info/BeebleSwitchXImageEdit",
            NativeServiceOperation::NodeCatalog,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(
            provider_catalog["BeebleSwitchXImageEdit"]["python_module"],
            "comfy_api_nodes.nodes_beeble"
        );
        assert_eq!(
            provider_catalog["BeebleSwitchXImageEdit"]["zed_native_binding"],
            json!({
                "disposition": "provider_required",
                "reason": "requires a verified native provider",
            })
        );
        assert_eq!(
            provider_catalog["BeebleSwitchXImageEdit"]["zed_source"]["catalog_status"],
            "provider-required"
        );
        assert!(
            provider_catalog["BeebleSwitchXImageEdit"]["zed_source"]["inactive_reason"].is_null()
        );
        assert!(
            provider_catalog["BeebleSwitchXImageEdit"]["zed_source"]["availability"].is_string()
        );
        assert!(provider_catalog["BeebleSwitchXImageEdit"]["zed_source"]["feature_id"].is_string());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn component_object_info_fails_closed_without_checked_presentation()
    -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let mut registry = NativeNodeRegistry::default();
        let descriptor = RuntimeNodeDescriptor {
            schema_version: 1,
            class_type: "Presentationless".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: None,
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: Vec::new(),
            output_node: false,
            effect: sim_runtime::EffectClass::Pure,
            cache: sim_runtime::RuntimeCachePolicy::InputIdentity,
        };
        let node: Arc<dyn sim_runtime::NativeNode> = Arc::new(PresentationlessNode);
        registry
            .register_bound_batch([(descriptor, node)])
            .map_err(|error| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_registry_invalid",
                    error.to_string(),
                )
            })?;
        let mut presentation =
            sim_runtime::ExecutionPresentationService::new(MAXIMUM_HISTORY_PAGE_SIZE)
                .map_err(presentation_error)?;
        presentation
            .initialize_profile(
                profile_id,
                sim_runtime::ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(presentation_error)?;
        let error = NativeRuntimeHttpServices::new(
            profile_id,
            sim_runtime::ExecutionPresentationOwner::ephemeral(presentation),
            Arc::new(AcceptingExecutionController),
            registry,
        )
        .err()
        .ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_incomplete_registry_accepted",
                "the API accepted a registry without checked presentation metadata",
            )
        })?;
        assert_eq!(error.kind, NativeServiceErrorKind::Unavailable);
        assert_eq!(error.code, "native_execution_registry_incomplete");
        Ok(())
    }

    #[test]
    fn native_queue_and_job_routes_share_real_execution_state() -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let prompt_id = "00000000-0000-0000-0000-000000000202";
        services.dispatch(request(
            HttpMethod::Post,
            "/prompt",
            NativeServiceOperation::SubmitPrompt,
            profile_id,
            Some(json!({
                "prompt_id": prompt_id,
                "prompt": {
                    "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                    "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
                }
            })),
        )?)?;

        let queue = body_json(services.dispatch(request(
            HttpMethod::Get,
            "/queue",
            NativeServiceOperation::QueueSnapshot,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(queue["queue_pending"][0][1], prompt_id);

        let job = body_json(services.dispatch(request(
            HttpMethod::Get,
            &format!("/api/jobs/{prompt_id}"),
            NativeServiceOperation::Jobs,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(job["status"], "pending");

        let cancellation = body_json(services.dispatch(request(
            HttpMethod::Post,
            &format!("/api/jobs/{prompt_id}/cancel"),
            NativeServiceOperation::Jobs,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(cancellation["cancelled"], true);
        let snapshot = services.snapshot()?;
        assert!(snapshot.queue.is_empty());
        assert_eq!(snapshot.attempts[0].state, AttemptState::Cancelled);
        Ok(())
    }

    #[test]
    fn native_prompt_literals_reject_nested_process_local_handles() -> Result<(), Box<dyn Error>> {
        let identity: sim_runtime::NativeHandleStoreIdentity = serde_json::from_value(json!({
            "store_id": "00000000-0000-0000-0000-000000000001",
            "generation_id": "00000000-0000-0000-0000-000000000002",
        }))?;
        let handle = sim_runtime::NativeOpaqueHandle::new(
            sim_runtime::NativeHandleType::new(sim_runtime::NativeHandleKind::Artifact, "SVG")?,
            identity,
            "artifact-1",
            1,
            Some("a".repeat(64)),
        )?;
        let literal = NativeValue::List {
            values: vec![NativeValue::List {
                values: vec![NativeValue::Handle { value: handle }],
            }],
        };
        let error = native_literal_json(&literal).expect_err("handles must remain process-local");
        assert_eq!(error.code, "native_prompt_handle_not_serializable");
        Ok(())
    }

    #[test]
    fn authority_is_profile_bound_and_later_owned_routes_fail_explicitly()
    -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let other_profile = profile("00000000-0000-0000-0000-000000000002")?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let capabilities = services.http_capabilities()?;
        let catalog = http_route_catalog().map_err(|error| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_catalog_invalid",
                error.to_string(),
            )
        })?;
        let prompt_route = catalog
            .iter()
            .find(|route| route.feature_id() == "COMFY-API-0118")
            .ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_prompt_route_missing",
                    "prompt route is absent from the catalog",
                )
            })?;
        let asset_route = catalog
            .iter()
            .find(|route| route.feature_id() == "COMFY-API-0007")
            .ok_or_else(|| {
                NativeServiceError::new(
                    NativeServiceErrorKind::Internal,
                    "test_asset_route_missing",
                    "asset route is absent from the catalog",
                )
            })?;
        assert_eq!(
            capabilities.state_for(prompt_route),
            CapabilityState::Available
        );
        assert!(matches!(
            capabilities.state_for(asset_route),
            CapabilityState::Unavailable { .. }
        ));
        let cross_profile = services.dispatch(request(
            HttpMethod::Get,
            "/features",
            NativeServiceOperation::Features,
            other_profile,
            None,
        )?);
        assert!(matches!(
            cross_profile,
            Err(NativeServiceError {
                kind: NativeServiceErrorKind::Forbidden,
                ..
            })
        ));

        let assets = services.dispatch(request(
            HttpMethod::Get,
            "/api/assets",
            NativeServiceOperation::Assets,
            profile_id,
            None,
        )?);
        let error = assets.err().ok_or_else(|| {
            NativeServiceError::new(
                NativeServiceErrorKind::Internal,
                "test_expected_capability_error",
                "asset reference route unexpectedly succeeded",
            )
        })?;
        assert_eq!(error.kind, NativeServiceErrorKind::Unavailable);
        assert_eq!(error.code, "native_route_capability_unavailable");
        Ok(())
    }

    #[test]
    fn native_health_reads_the_live_profile_state() -> Result<(), NativeServiceError> {
        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?;
        let request = request(
            HttpMethod::Get,
            "/health",
            NativeServiceOperation::CatalogRoute {
                feature_id: "COMFY-API-0101".to_owned(),
            },
            profile_id,
            None,
        )?;
        let response = services.dispatch(request)?;
        assert_eq!(response.status, 200);
        assert!(matches!(response.body, HttpBody::Bytes(_)));
        Ok(())
    }

    #[test]
    fn model_wire_projection_preserves_logical_roots_and_deduplicates_legacy_names()
    -> Result<(), Box<dyn Error>> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "sim-api-model-wire-{}-{unique}",
            std::process::id()
        ));
        let profile_root = directory.join("profile");
        let first_model_root = directory.join("first-model-root");
        let second_model_root = directory.join("second-model-root");
        fs::create_dir_all(first_model_root.join("checkpoints"))?;
        fs::create_dir_all(second_model_root.join("checkpoints"))?;
        fs::write(
            first_model_root.join("checkpoints/shared.safetensors"),
            b"first",
        )?;
        fs::write(
            second_model_root.join("checkpoints/shared.safetensors"),
            b"second",
        )?;
        fs::write(
            second_model_root.join("checkpoints/unique.safetensors"),
            b"unique",
        )?;

        let profile_id = profile("00000000-0000-0000-0000-000000000001")?;
        let assets = open_native_profile_asset_service(
            profile_id.0.to_string(),
            &profile_root,
            &[first_model_root, second_model_root],
        )?;
        let model_reader = authorize_native_api_asset_reader(
            &PermissionPolicy::native_runtime_services(profile_id.0.to_string())?,
        )?;
        lock(&assets, "test_asset_state_poisoned")?.scan_namespaces(
            &[AssetNamespace::Model],
            &model_reader,
            &CancellationToken::default(),
        )?;
        let services = NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(AcceptingExecutionController),
        )?
        .with_assets(assets, model_reader)?;

        let legacy = body_json(services.dispatch(request(
            HttpMethod::Get,
            "/models/checkpoints",
            NativeServiceOperation::Models,
            profile_id,
            None,
        )?)?)?;
        assert_eq!(legacy, json!(["shared.safetensors", "unique.safetensors"]));

        let experimental = body_json(services.dispatch(request(
            HttpMethod::Get,
            "/experiment/models",
            NativeServiceOperation::Models,
            profile_id,
            None,
        )?)?)?;
        let entries = experimental
            .as_array()
            .ok_or("model folders are not an array")?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "checkpoints");
        let logical_roots = entries[0]["folders"]
            .as_array()
            .ok_or("logical model roots are not an array")?;
        assert_eq!(logical_roots.len(), 2);
        assert_ne!(logical_roots[0], logical_roots[1]);
        for logical_root in logical_roots {
            let logical_root = logical_root
                .as_str()
                .ok_or("logical model root is not a string")?;
            assert!(logical_root.starts_with("model-configured-"));
            assert!(!logical_root.contains(directory.to_string_lossy().as_ref()));
        }

        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn router_request_shape_is_compatible_with_the_concrete_services() {
        let request = HttpRequest::new(HttpMethod::Get, "/features");
        assert_eq!(request.path, "/features");
    }
}
