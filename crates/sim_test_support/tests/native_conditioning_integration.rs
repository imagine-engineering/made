use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    CLIP_VISION_SOURCE_SHA256, ClipVisionOutput, FLUX_REDUX_SOURCE_SHA256,
    GLIGEN_ATTENTION_SOURCE_SHA256, GLIGEN_OPENAIMODEL_SOURCE_SHA256,
    GLIGEN_SAMPLERS_SOURCE_SHA256, GLIGEN_SOURCE_SHA256, MappedModelWeights,
    NativeGligenCheckpoint, NativeGligenError, NativeGligenPositionParameter, NativeGligenRegion,
    NativeGligenResource, NativeModelPayload, NativeModelPayloadError, NativeModelResourceRole,
    NativePhotoMakerCheckpoint, NativePhotoMakerCheckpointEntry, NativePhotoMakerError,
    NativePhotoMakerResource, NativeStyleModelCheckpoint, NativeStyleModelError,
    NativeStyleModelResource, PHOTOMAKER_CLIP_VISION_SOURCE_SHA256, PHOTOMAKER_SOURCE_SHA256,
    PatchGraph, PatchGraphIdentity, PatchPayload, PatchTensor, PatchValueTransform,
    STYLE_ADAPTER_SOURCE_SHA256, STYLE_MODEL_NODES_SOURCE_SHA256, STYLE_MODEL_OPS_SOURCE_SHA256,
    STYLE_MODEL_SD_SOURCE_SHA256, SemanticPatchOperation,
    conditioning::{
        ClipVisionConditioning, ConditioningConstant, ConditioningControlBinding,
        ConditioningControlReference, ConditioningEntry, ConditioningEntryOptions,
        ConditioningHookReference, ConditioningIdentity, ConditioningMask, ConditioningMetadataKey,
        ConditioningMetadataMap, ConditioningMetadataValue, ConditioningReferences,
        ConditioningRegion, ConditioningSet, ConditioningTimestepRange, ConditioningValue,
        ConditioningWindow,
    },
    controlnet::{
        ControlBase, ControlChain, ControlHintLayout, ControlHintPreprocess, ControlModelBinding,
        ControlModelExecutor, ControlModelInput, ControlNet, ControlNetError,
        ControlNetRebindRequest, ControlNode, ControlPercentWindow, ControlResult,
        ControlTensorBinding, StrengthType,
    },
    generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
};
use sim_runtime::{
    AttemptEventKind, AttemptState, CanonicalConditioningCacheIdentities,
    CanonicalNativeDiffusionCacheIdentities, NativeDiffusionBundle, NativeDiffusionProvider,
    NativeHandleStoreGeneration, NativeImageExecutor, NativeImageRuntimeError,
    PreboundControlExecution, compile_native_diffusion_workflow,
    publish_native_conditioning_control_transaction,
};
use sim_sampler::{
    DiscreteSamplingProfile, GuidanceBranch, GuidanceDenoiser, GuidanceError, GuidanceEvaluation,
    GuidanceHook, GuidanceHookContext, GuidanceOptions, GuidancePredictions,
    NativeConditioningMetadata, NativeConditioningPayload, NativeControlExecution,
    NativeControlPayload, PredictionInterpretation, SamplingPlan, SamplingProfile,
    SamplingProfileIdentity, execute_guidance,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    NativeLatentBundle, NativeLatentMetadata, ResizeMode, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::{
        real_add_with_context_exact_native, sigmoid_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::full_like_with_context_exact_native,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_test_support::NativeDiffusionFixture;
use sim_types::{AttemptId, ProfileId};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use uuid::Uuid;

const WORKFLOW: &[u8] = include_bytes!("../fixtures/native_diffusion/workflow.json");
const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const STYLE_MODEL_FIXTURE_MEMORY: u64 = 64 * 1024 * 1024;
const GLIGEN_FIXTURE_MEMORY: u64 = 128 * 1024 * 1024;
const STYLE_MODEL_ORACLE: &[u8] =
    include_bytes!("../fixtures/models/conditioning-auxiliary-resource-foundation/oracle.json");
const STYLE_MODEL_MANIFEST: &[u8] =
    include_bytes!("../fixtures/models/conditioning-auxiliary-resource-foundation/manifest.json");
const STYLE_MODEL_PROVENANCE: &[u8] =
    include_bytes!("../fixtures/models/conditioning-auxiliary-resource-foundation/provenance.json");
const STYLE_MODEL_SOURCE_GRAPH: &[u8] =
    include_bytes!("../fixtures/models/conditioning-auxiliary-resource-foundation/source_graph.py");
const STYLE_MODEL_GENERATOR: &[u8] = include_bytes!(
    "../fixtures/models/conditioning-auxiliary-resource-foundation/generate_oracle.py"
);
const CONDITIONING_TASK: &str = "comfy-parity-conditioning-value-foundation";
const GUIDANCE_TASK: &str = "comfy-parity-conditioning-guidance-adapter";
const FIXTURE_CONTROL_EXECUTOR_DIGEST: &str =
    "749e2d208a8bab88e36f9b096bb23faca0a27fb66d8cccdba8f11e17feee5e75";

#[test]
fn task616_sampler_controlnet_text_override_preserves_primary_and_fallback_and_recovers()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let bundle =
        NativeDiffusionFixture::checked_in().load_bundle_with_context(backend.clone(), &context)?;
    let primary = tensor_from_f32(&backend, &[1, 1, 2], &[1.0, 2.0], &context)?;
    let override_attention = tensor_from_f32(&backend, &[1, 1, 2], &[3.0, 4.0], &context)?;
    let pooled = tensor_from_f32(&backend, &[1, 2], &[5.0, 6.0], &context)?;
    let latent = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.25; 4], &context)?;
    let control = ControlNet::checked(
        ControlBase::checked(
            1.0,
            StrengthType::Constant,
            ControlPercentWindow::checked(0.0, 1.0)?,
            false,
            None,
        )?,
        ControlModelBinding::checked(
            sd15_model_family_identity()?,
            bundle.conditioning().patch_graph().identity(),
            bundle.model_digest(),
            FIXTURE_CONTROL_EXECUTOR_DIGEST,
            DType::F32,
            DeviceId::CPU,
        )?,
        ControlTensorBinding::checked(latent.clone(), tensor_digest(&backend, &latent, &context)?)?,
        1,
        ResizeMode::NearestExact,
        ControlHintPreprocess::Identity,
        None,
        Vec::new(),
        false,
        Vec::new(),
    )?;
    let calls = Arc::new(AtomicUsize::new(0));
    let execution = NativeControlExecution::checked(
        Arc::new(ControlChain::checked(vec![ControlNode::ControlNet(
            control,
        )])?),
        Arc::new(FixtureControlExecutor {
            calls: calls.clone(),
            mode: FixtureControlMode::ReturnContext,
        }),
    )?;
    let payload = NativeConditioningPayload::checked_sd15(
        bundle.model_digest(),
        bundle.model(),
        bundle.conditioning().patch_graph().clone(),
        Some(execution),
    )?;
    let input = ConditioningSet::checked(
        payload.identity().clone(),
        vec![ConditioningEntry::checked(
            "primary",
            ConditioningValue::cross_attention(primary.clone())?,
            ConditioningEntryOptions::default(),
        )?],
        &cancellation,
    )?;
    let primary_digest = input.digest().to_owned();
    let evaluate = |set: &ConditioningSet, execution_context: &ExecutionContext<'_>| {
        let entry = set
            .entries()
            .first()
            .ok_or("missing conditioning entry")?
            .resolve(latent.descriptor(), &backend, execution_context)?;
        let output = payload
            .execute_control(
                &backend,
                &latent,
                1.0,
                &primary,
                0.5,
                &entry,
                execution_context,
            )?
            .ok_or("missing ControlNet output")?;
        let attention = output
            .middle()
            .first()
            .and_then(Option::as_ref)
            .ok_or("missing ControlNet attention")?;
        Ok::<_, Box<dyn Error>>(tensor_to_f32(&backend, attention, execution_context)?.to_vec())
    };
    assert_eq!(evaluate(&input, &context)?, [1.0, 2.0]);
    let with_none =
        input.with_controlnet_text_encoding(&override_attention, None, &cancellation)?;
    assert_eq!(evaluate(&with_none, &context)?, [3.0, 4.0]);
    let without_pooled = input.with_entry_metadata(
        vec![ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::CrossAttentionControlnet,
            ConditioningMetadataValue::Tensor(override_attention.clone()),
        )])?],
        &cancellation,
    )?;
    assert_ne!(without_pooled.digest(), with_none.digest());
    assert_eq!(evaluate(&without_pooled, &context)?, [3.0, 4.0]);
    let with_pooled =
        input.with_controlnet_text_encoding(&override_attention, Some(&pooled), &cancellation)?;
    assert_ne!(with_none.digest(), with_pooled.digest());
    assert_eq!(evaluate(&with_pooled, &context)?, [3.0, 4.0]);
    assert_eq!(input.digest(), primary_digest);
    assert_eq!(
        tensor_to_f32(&backend, &primary, &context)?.as_ref(),
        &[1.0, 2.0]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancelled,
    );
    assert!(evaluate(&with_pooled, &cancelled_context).is_err());
    let entry = with_pooled
        .entries()
        .first()
        .ok_or("missing conditioning entry")?
        .resolve(latent.descriptor(), &backend, &context)?;
    assert!(matches!(
        payload.execute_control(
            &backend,
            &latent,
            1.0,
            &primary,
            0.5,
            &entry,
            &cancelled_context,
        ),
        Err(ControlNetError::Cancelled)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(evaluate(&input, &context)?, [1.0, 2.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

const CONTRACT_CASES: [(&str, &str, &str); 17] = [
    (
        "conditioning-conditioning-value-conds-condregular-505e5b9e",
        CONDITIONING_TASK,
        "conditioning-conditioning-value-conds-condregular-505e5b9e:regular-repeat-concat-size",
    ),
    (
        "conditioning-conditioning-value-conds-condnoiseshape-7f11dbb1",
        CONDITIONING_TASK,
        "conditioning-conditioning-value-conds-condnoiseshape-7f11dbb1:noise-shape-region-repeat",
    ),
    (
        "conditioning-conditioning-value-conds-condcrossattn-4d921d69",
        CONDITIONING_TASK,
        "conditioning-conditioning-value-conds-condcrossattn-4d921d69:cross-attention-lcm-concat",
    ),
    (
        "conditioning-conditioning-value-conds-condconstant-0e559aad",
        CONDITIONING_TASK,
        "conditioning-conditioning-value-conds-condconstant-0e559aad:constant-equality-identity",
    ),
    (
        "conditioning-conditioning-value-conds-condlist-21ce2116",
        CONDITIONING_TASK,
        "conditioning-conditioning-value-conds-condlist-21ce2116:list-itemwise-process-concat-size",
    ),
    (
        "conditioning-guidance-samplers-get-area-and-mult-14d8dec2",
        GUIDANCE_TASK,
        "conditioning-guidance-samplers-get-area-and-mult-14d8dec2:resolved-area-mask-window-weight",
    ),
    (
        "conditioning-guidance-samplers-calc-cond-batch-23aa4a02",
        GUIDANCE_TASK,
        "conditioning-guidance-samplers-calc-cond-batch-23aa4a02:compatible-batch-regional-accumulation",
    ),
    (
        "conditioning-guidance-samplers-sampling-function-ef25ad1d",
        GUIDANCE_TASK,
        "conditioning-guidance-samplers-sampling-function-ef25ad1d:cfg-skip-and-hook-pipeline",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-prepare-mask-048488c7",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-prepare-mask-048488c7:mask-normalize-broadcast",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-get-models-from-cond-1be91d68",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-get-models-from-cond-1be91d68:typed-control-hook-reference-projection",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-convert-cond-e8752d85",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-convert-cond-e8752d85:typed-entry-set-conversion",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-get-additional-models-7ba596bf",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-get-additional-models-7ba596bf:prebound-additional-model-identity",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-prepare-sampling-b141c606",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-prepare-sampling-b141c606:prebound-bundle-load-and-execute",
    ),
    (
        "conditioning-guidance-hook-sampler-helpers-cleanup-models-6f147c97",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-sampler-helpers-cleanup-models-6f147c97:scope-drop-workspace-convergence",
    ),
    (
        "conditioning-guidance-hook-hooks-hook-536ff505",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-hooks-hook-536ff505:ordered-guidance-hook-phases",
    ),
    (
        "conditioning-guidance-hook-hooks-weighthook-03327446",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-hooks-weighthook-03327446:weight-hook-patchgraph-delegation",
    ),
    (
        "conditioning-guidance-hook-patcher-extension-patcherinjection-116374da",
        GUIDANCE_TASK,
        "conditioning-guidance-hook-patcher-extension-patcherinjection-116374da:injection-hook-lifecycle-cancellation",
    ),
];

#[derive(Clone, Copy)]
enum FixtureControlMode {
    Success,
    ReturnContext,
    Cancelled,
    AllocationFailed,
}

struct FixtureControlExecutor {
    calls: Arc<AtomicUsize>,
    mode: FixtureControlMode,
}

impl ControlModelExecutor for FixtureControlExecutor {
    fn execution_digest(&self) -> &str {
        FIXTURE_CONTROL_EXECUTOR_DIGEST
    }

    fn resident_bytes(&self) -> Result<u64, ControlNetError> {
        u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| ControlNetError::ResidentBytesOverflow)
    }

    fn execute_controlnet(
        &self,
        _binding: &ControlModelBinding,
        input: &ControlModelInput,
        context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            FixtureControlMode::ReturnContext => {
                context
                    .cancellation
                    .check()
                    .map_err(|_| ControlNetError::Cancelled)?;
                ControlResult::checked(Vec::new(), vec![Some(input.context.clone())], Vec::new())
            }
            FixtureControlMode::Success => {
                context
                    .cancellation
                    .check()
                    .map_err(|_| ControlNetError::Cancelled)?;
                ControlResult::checked(vec![Some(input.hint.clone())], Vec::new(), Vec::new())
            }
            FixtureControlMode::Cancelled => Err(ControlNetError::Cancelled),
            FixtureControlMode::AllocationFailed => Err(ControlNetError::CanonicalTensor(
                Box::new(TensorError::AllocationFailed {
                    requested: 4096,
                    reason: "injected ControlNet allocation failure".to_owned(),
                }),
            )),
        }
    }

    fn execute_t2i_adapter(
        &self,
        _binding: &ControlModelBinding,
        _hint: &Tensor,
        _context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError> {
        Err(ControlNetError::Invalid(
            "the SD15 integration fixture binds a ControlNet executor".to_owned(),
        ))
    }
}

fn tensor_digest(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<String, Box<dyn Error>> {
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(tensor.descriptor())?);
    for value in tensor_to_f32(backend, tensor, context)?.iter() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn assert_f32_tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    tensor: &Tensor,
    expected_shape: &[u64],
    expected_values: &[f32],
) -> Result<(), Box<dyn Error>> {
    assert_eq!(tensor.descriptor().shape(), expected_shape);
    let actual_values = tensor_to_f32(backend, tensor, context)?;
    assert_eq!(&*actual_values, expected_values);
    Ok(())
}

fn record_contract_case(
    executed_case_ids: &mut BTreeSet<&'static str>,
    contract_id: &str,
    expected_task: &str,
) -> Result<(), Box<dyn Error>> {
    let Some((_, task_id, case_id)) = CONTRACT_CASES
        .iter()
        .find(|(mapped_contract_id, _, _)| *mapped_contract_id == contract_id)
    else {
        return Err(format!("missing contract mapping for {contract_id}").into());
    };
    if *task_id != expected_task {
        return Err(format!("contract {contract_id} is associated with task {task_id}").into());
    }
    if !executed_case_ids.insert(*case_id) {
        return Err(format!("contract case {case_id} executed more than once").into());
    }
    Ok(())
}

fn assert_conditioning_value_contract_cases(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeSet<&'static str>, Box<dyn Error>> {
    let mut executed_case_ids = BTreeSet::new();

    let regular =
        ConditioningValue::regular(tensor_from_f32(backend, &[2, 1], &[1.0, 2.0], context)?)?;
    assert_eq!(regular.size()?, vec![2_u64, 1]);
    let processed_regular = regular.process(5, None, backend, context)?;
    let ConditioningValue::Regular(processed_regular_tensor) = &processed_regular else {
        return Err("processed regular conditioning changed kind".into());
    };
    assert_f32_tensor(
        backend,
        context,
        processed_regular_tensor,
        &[5, 1],
        &[1.0, 2.0, 1.0, 2.0, 1.0],
    )?;
    let other_regular =
        ConditioningValue::regular(tensor_from_f32(backend, &[2, 1], &[3.0, 4.0], context)?)?;
    assert!(regular.can_concat(&other_regular));
    let concatenated_regular = regular.concat(&[other_regular], backend, context)?;
    let ConditioningValue::Regular(concatenated_regular_tensor) = &concatenated_regular else {
        return Err("concatenated regular conditioning changed kind".into());
    };
    assert_f32_tensor(
        backend,
        context,
        concatenated_regular_tensor,
        &[4, 1],
        &[1.0, 2.0, 3.0, 4.0],
    )?;
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-conditioning-value-conds-condregular-505e5b9e",
        CONDITIONING_TASK,
    )?;

    let noise_values = (0..20).map(|value| value as f32).collect::<Vec<_>>();
    let noise = ConditioningValue::noise_shape(tensor_from_f32(
        backend,
        &[1, 1, 4, 5],
        &noise_values,
        context,
    )?)?;
    let region = ConditioningRegion::absolute(vec![2, 3], vec![1, 1])?.resolve(&[4, 5])?;
    let processed_noise = noise.process(2, Some(&region), backend, context)?;
    let ConditioningValue::NoiseShape(processed_noise_tensor) = &processed_noise else {
        return Err("processed noise-shape conditioning changed kind".into());
    };
    assert_f32_tensor(
        backend,
        context,
        processed_noise_tensor,
        &[2, 1, 2, 3],
        &[
            6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0,
        ],
    )?;
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-conditioning-value-conds-condnoiseshape-7f11dbb1",
        CONDITIONING_TASK,
    )?;

    let left_cross_attention = ConditioningValue::cross_attention(tensor_from_f32(
        backend,
        &[1, 2, 1],
        &[1.0, 2.0],
        context,
    )?)?;
    let right_cross_attention = ConditioningValue::cross_attention(tensor_from_f32(
        backend,
        &[1, 3, 1],
        &[3.0, 4.0, 5.0],
        context,
    )?)?;
    assert!(left_cross_attention.can_concat(&right_cross_attention));
    let concatenated_cross_attention =
        left_cross_attention.concat(&[right_cross_attention], backend, context)?;
    let ConditioningValue::CrossAttention(cross_attention_tensor) = &concatenated_cross_attention
    else {
        return Err("concatenated cross-attention conditioning changed kind".into());
    };
    assert_f32_tensor(
        backend,
        context,
        cross_attention_tensor,
        &[2, 6, 1],
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0],
    )?;
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-conditioning-value-conds-condcrossattn-4d921d69",
        CONDITIONING_TASK,
    )?;

    let constant = ConditioningValue::constant(ConditioningConstant::finite_f64(0.5)?)?;
    let same_constant = ConditioningValue::constant(ConditioningConstant::finite_f64(0.5)?)?;
    let different_constant = ConditioningValue::constant(ConditioningConstant::finite_f64(0.25)?)?;
    assert_eq!(constant.size()?, vec![1_u64]);
    assert!(constant.can_concat(&same_constant));
    assert!(!constant.can_concat(&different_constant));
    let concatenated_constant = constant.concat(&[same_constant], backend, context)?;
    let ConditioningValue::Constant(concatenated_constant) = concatenated_constant else {
        return Err("concatenated constant conditioning changed kind".into());
    };
    assert_eq!(
        concatenated_constant,
        ConditioningConstant::finite_f64(0.5)?
    );
    assert!(
        constant
            .concat(&[different_constant], backend, context)
            .is_err()
    );
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-conditioning-value-conds-condconstant-0e559aad",
        CONDITIONING_TASK,
    )?;

    let list = ConditioningValue::list(vec![
        tensor_from_f32(backend, &[1, 2], &[1.0, 2.0], context)?,
        tensor_from_f32(backend, &[1, 1], &[3.0], context)?,
    ])?;
    let other_list = ConditioningValue::list(vec![
        tensor_from_f32(backend, &[1, 2], &[4.0, 5.0], context)?,
        tensor_from_f32(backend, &[1, 1], &[6.0], context)?,
    ])?;
    assert_eq!(list.size()?, vec![1_u64, 1, 3]);
    assert!(list.can_concat(&other_list));
    let processed_list = list.process(2, None, backend, context)?;
    let ConditioningValue::List(processed_items) = &processed_list else {
        return Err("processed list conditioning changed kind".into());
    };
    let [processed_first, processed_second] = processed_items.as_slice() else {
        return Err("processed list conditioning did not contain two items".into());
    };
    assert_f32_tensor(
        backend,
        context,
        processed_first,
        &[2, 2],
        &[1.0, 2.0, 1.0, 2.0],
    )?;
    assert_f32_tensor(backend, context, processed_second, &[2, 1], &[3.0, 3.0])?;
    let concatenated_list = list.concat(&[other_list], backend, context)?;
    let ConditioningValue::List(concatenated_items) = &concatenated_list else {
        return Err("concatenated list conditioning changed kind".into());
    };
    let [concatenated_first, concatenated_second] = concatenated_items.as_slice() else {
        return Err("concatenated list conditioning did not contain two items".into());
    };
    assert_f32_tensor(
        backend,
        context,
        concatenated_first,
        &[2, 2],
        &[1.0, 2.0, 4.0, 5.0],
    )?;
    assert_f32_tensor(backend, context, concatenated_second, &[2, 1], &[3.0, 6.0])?;
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-conditioning-value-conds-condlist-21ce2116",
        CONDITIONING_TASK,
    )?;

    Ok(executed_case_ids)
}

fn guidance_contract_identity(namespace: &str) -> Result<ConditioningIdentity, Box<dyn Error>> {
    Ok(ConditioningIdentity::new(
        namespace,
        sd15_model_family_identity()?,
        sd15_latent_format_identity()?,
    )?)
}

fn assert_guidance_entry_contract_cases(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    executed_case_ids: &mut BTreeSet<&'static str>,
) -> Result<(), Box<dyn Error>> {
    let target = tensor_from_f32(backend, &[2, 2, 4], &[0.0; 16], context)?;
    let mask = tensor_from_f32(backend, &[4], &[0.25, 0.5, 0.75, 1.0], context)?;
    let references = ConditioningReferences::checked(
        Some(ConditioningControlReference::checked("control.primary")?),
        vec![
            ConditioningHookReference::checked("hook.first")?,
            ConditioningHookReference::checked("hook.second")?,
        ],
    )?;
    let entry = ConditioningEntry::checked_with_references(
        "source.guidance.entry",
        ConditioningValue::regular(tensor_from_f32(backend, &[1, 1], &[1.0], context)?)?,
        ConditioningEntryOptions {
            strength: 2.0,
            region: Some(ConditioningRegion::absolute(vec![2], vec![1])?),
            mask: Some(ConditioningMask::new(mask, 0.5, vec![0], false)?),
            window: ConditioningWindow::new(0.25, 0.75)?,
            default_region: false,
        },
        references,
    )?;
    let conditioning = ConditioningSet::checked(
        guidance_contract_identity("native.conditioning.guidance.entry")?,
        vec![entry],
        context.cancellation,
    )?;
    let resolved = conditioning.resolve(target.descriptor(), backend, context)?;
    let [resolved_entry] = resolved.as_slice() else {
        return Err("guidance entry fixture did not resolve exactly one entry".into());
    };
    let ConditioningValue::Regular(resolved_value) = resolved_entry.value() else {
        return Err("resolved guidance entry changed conditioning kind".into());
    };
    assert_f32_tensor(backend, context, resolved_value, &[2, 1], &[1.0, 1.0])?;
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-convert-cond-e8752d85",
        GUIDANCE_TASK,
    )?;

    assert_eq!(resolved_entry.region().sizes(), &[2]);
    assert_eq!(resolved_entry.region().offsets(), &[1]);
    assert_eq!(resolved_entry.strength(), 2.0);
    assert!(resolved_entry.window().contains(0.5));
    assert!(!resolved_entry.window().contains(0.8));
    let resolved_mask = resolved_entry
        .mask()
        .ok_or("guidance entry mask did not resolve")?;
    assert_eq!(resolved_mask.strength(), 0.5);
    assert_eq!(resolved_entry.contribution_weight(&[0], Some(0.5))?, 0.5);
    assert_eq!(resolved_entry.contribution_weight(&[1], Some(0.75))?, 0.75);
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-samplers-get-area-and-mult-14d8dec2",
        GUIDANCE_TASK,
    )?;

    assert_f32_tensor(
        backend,
        context,
        resolved_mask.tensor(),
        &[2, 2, 2],
        &[0.5, 0.75, 0.5, 0.75, 0.5, 0.75, 0.5, 0.75],
    )?;
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-prepare-mask-048488c7",
        GUIDANCE_TASK,
    )?;

    assert_eq!(
        resolved_entry
            .references()
            .control()
            .map(|reference| reference.identifier()),
        Some("control.primary")
    );
    let hook_identifiers = resolved_entry
        .references()
        .hooks()
        .iter()
        .map(ConditioningHookReference::identifier)
        .collect::<Vec<_>>();
    assert_eq!(hook_identifiers, vec!["hook.first", "hook.second"]);
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-get-models-from-cond-1be91d68",
        GUIDANCE_TASK,
    )?;
    Ok(())
}

struct GuidanceContractDenoiser<'a> {
    backend: &'a CpuBackend,
    batches: Vec<Vec<GuidanceBranch>>,
}

impl GuidanceDenoiser for GuidanceContractDenoiser<'_> {
    fn evaluate_batch(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        self.batches
            .push(evaluations.iter().map(GuidanceEvaluation::branch).collect());
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(evaluations.len())
            .map_err(|_| GuidanceError::ShapeOverflow("contract denoiser outputs"))?;
        for evaluation in evaluations {
            let value = match evaluation.entry().identifier() {
                "left" => 2.0,
                "strong" => 6.0,
                "negative" => 1.0,
                identifier => {
                    return Err(GuidanceError::Invalid(format!(
                        "unexpected contract denoiser entry {identifier}"
                    )));
                }
            };
            let count = usize::try_from(evaluation.latent().descriptor().element_count()?)
                .map_err(|_| GuidanceError::ShapeOverflow("contract denoiser output"))?;
            outputs.push(tensor_from_f32(
                self.backend,
                evaluation.latent().descriptor().shape(),
                &vec![value; count],
                context,
            )?);
        }
        Ok(outputs)
    }
}

struct OrderedGuidanceContractHook {
    name: &'static str,
    events: Rc<RefCell<Vec<String>>>,
    override_with_conditional: bool,
}

impl GuidanceHook for OrderedGuidanceContractHook {
    fn pre_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &mut GuidancePredictions,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), GuidanceError> {
        self.events.borrow_mut().push(format!("pre:{}", self.name));
        Ok(())
    }

    fn override_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        predictions: &GuidancePredictions,
        _current: &Tensor,
        _context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        self.events
            .borrow_mut()
            .push(format!("override:{}", self.name));
        Ok(self
            .override_with_conditional
            .then(|| predictions.conditional().clone()))
    }

    fn post_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        current: &Tensor,
        _context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        self.events.borrow_mut().push(format!("post:{}", self.name));
        Ok(Some(current.clone()))
    }
}

struct InjectionLifecycleHook {
    events: Rc<RefCell<Vec<&'static str>>>,
    active: bool,
}

impl GuidanceHook for InjectionLifecycleHook {
    fn pre_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &mut GuidancePredictions,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), GuidanceError> {
        if self.active {
            return Err(GuidanceError::Invalid(
                "injection hook was activated twice".to_owned(),
            ));
        }
        self.active = true;
        self.events.borrow_mut().push("inject");
        Ok(())
    }

    fn override_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        _current: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        self.events.borrow_mut().push("override");
        context.cancellation.cancel();
        Ok(None)
    }
}

impl Drop for InjectionLifecycleHook {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            self.events.borrow_mut().push("eject");
        }
    }
}

fn assert_guidance_execution_contract_cases(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    executed_case_ids: &mut BTreeSet<&'static str>,
) -> Result<(), Box<dyn Error>> {
    let conditioning_value = tensor_from_f32(backend, &[1, 1], &[1.0], context)?;
    let regional_options = |strength, offset| -> Result<_, Box<dyn Error>> {
        Ok(ConditioningEntryOptions {
            strength,
            region: Some(ConditioningRegion::absolute(vec![3], vec![offset])?),
            mask: None,
            window: ConditioningWindow::full(),
            default_region: false,
        })
    };
    let conditional = ConditioningSet::checked(
        guidance_contract_identity("native.conditioning.guidance.positive")?,
        vec![
            ConditioningEntry::checked(
                "left",
                ConditioningValue::regular(conditioning_value.clone())?,
                regional_options(1.0, 0)?,
            )?,
            ConditioningEntry::checked(
                "strong",
                ConditioningValue::regular(conditioning_value.clone())?,
                regional_options(3.0, 1)?,
            )?,
        ],
        context.cancellation,
    )?;
    let unconditional = ConditioningSet::checked(
        guidance_contract_identity("native.conditioning.guidance.negative")?,
        vec![ConditioningEntry::checked(
            "negative",
            ConditioningValue::regular(conditioning_value)?,
            ConditioningEntryOptions::default(),
        )?],
        context.cancellation,
    )?;
    let latent = tensor_from_f32(backend, &[1, 1, 4], &[0.0; 4], context)?;
    let profile = DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("conditioning-contract-profile")?,
        PredictionInterpretation::Epsilon,
        Arc::from([1.0_f32, 2.0_f32]),
    )?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        profile.identity().clone(),
        7,
        1,
        1.0,
        1.0,
    )?;

    let hook_events = Rc::new(RefCell::new(Vec::new()));
    let mut first_hook = OrderedGuidanceContractHook {
        name: "a",
        events: hook_events.clone(),
        override_with_conditional: true,
    };
    let mut second_hook = OrderedGuidanceContractHook {
        name: "b",
        events: hook_events.clone(),
        override_with_conditional: false,
    };
    let mut denoiser = GuidanceContractDenoiser {
        backend,
        batches: Vec::new(),
    };
    let result = {
        let mut hooks: [&mut dyn GuidanceHook; 2] = [&mut first_hook, &mut second_hook];
        execute_guidance(
            backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut hooks,
            context,
        )?
    };
    assert_f32_tensor(
        backend,
        context,
        result.guided(),
        &[1, 1, 4],
        &[2.0, 5.0, 5.0, 6.0],
    )?;
    assert_eq!(result.denoiser_evaluations(), 2);
    assert_eq!(result.denoiser_batches(), 1);
    assert_eq!(
        denoiser.batches,
        vec![vec![
            GuidanceBranch::Conditional,
            GuidanceBranch::Conditional,
        ]]
    );
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-samplers-calc-cond-batch-23aa4a02",
        GUIDANCE_TASK,
    )?;
    assert!(result.unconditional_skipped());
    assert_eq!(
        &*hook_events.borrow(),
        &[
            "pre:a",
            "pre:b",
            "override:a",
            "override:b",
            "post:a",
            "post:b",
        ]
    );
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-samplers-sampling-function-ef25ad1d",
        GUIDANCE_TASK,
    )?;
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-hook-hooks-hook-536ff505",
        GUIDANCE_TASK,
    )?;

    denoiser.batches.clear();
    let injection_events = Rc::new(RefCell::new(Vec::new()));
    let injection_cancellation = CancellationToken::default();
    let injection_context = backend.execution_context(
        context.stream,
        context.scratch.clone(),
        &injection_cancellation,
    );
    {
        let mut injection_hook = InjectionLifecycleHook {
            events: injection_events.clone(),
            active: false,
        };
        let mut injection_hooks: [&mut dyn GuidanceHook; 1] = [&mut injection_hook];
        assert!(matches!(
            execute_guidance(
                backend,
                &latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut injection_hooks,
                &injection_context,
            ),
            Err(GuidanceError::Cancelled)
        ));
        assert!(injection_cancellation.is_cancelled());
        assert_eq!(&*injection_events.borrow(), &["inject", "override"]);
    }
    assert_eq!(
        &*injection_events.borrow(),
        &["inject", "override", "eject"]
    );
    assert_eq!(injection_context.scratch.in_use_bytes(), 0);
    record_contract_case(
        executed_case_ids,
        "conditioning-guidance-hook-patcher-extension-patcherinjection-116374da",
        GUIDANCE_TASK,
    )?;
    Ok(())
}

fn assert_guidance_contract_cases(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeSet<&'static str>, Box<dyn Error>> {
    let mut executed_case_ids = BTreeSet::new();
    assert_guidance_entry_contract_cases(backend, context, &mut executed_case_ids)?;
    assert_guidance_execution_contract_cases(backend, context, &mut executed_case_ids)?;
    Ok(executed_case_ids)
}

fn prebound_fixture(
    backend: &Arc<CpuBackend>,
    context: &ExecutionContext<'_>,
    mode: FixtureControlMode,
) -> Result<(NativeDiffusionFixture, Arc<AtomicUsize>), Box<dyn Error>> {
    let fixture = NativeDiffusionFixture::checked_in();
    let cache_identities = fixture.cache_identities(context.cancellation)?;
    let model_digest = cache_identities.model_digest().to_owned();
    let patch_graph = Arc::new(PatchGraph::checked_semantic(
        &model_digest,
        vec![SemanticPatchOperation {
            identifier: "native-conditioning-output-bias".to_owned(),
            target_key: "model.diffusion_model.out.2.bias".to_owned(),
            expected_shape: vec![4],
            strength: 1.0,
            strength_model: 1.0,
            slices: Vec::new(),
            transform: PatchValueTransform::default(),
            payload: PatchPayload::Set {
                tensor: PatchTensor::checked(vec![4], vec![0.015, -0.01, 0.005, -0.02])?,
            },
        }],
    )?);
    let source_bias = tensor_from_f32(backend, &[4], &[0.0; 4], context)?;
    let mapped = MappedModelWeights::from_test_parts(
        model_digest.clone(),
        BTreeMap::from([("model.diffusion_model.out.2.bias".to_owned(), source_bias)]),
        Vec::new(),
    )?;
    let patched = patch_graph.apply(backend.as_ref(), &mapped, context)?;
    let patched_bias = patched
        .tensors()
        .get("model.diffusion_model.out.2.bias")
        .ok_or("patched fixture output bias is missing")?;
    assert_f32_tensor(
        backend.as_ref(),
        context,
        patched_bias,
        &[4],
        &[0.015, -0.01, 0.005, -0.02],
    )?;
    assert_ne!(patched.cache_identity(), mapped.cache_identity());

    let hint = tensor_from_f32(backend, &[1, 32, 4, 4], &[0.0025; 512], context)?;
    let hint = ControlTensorBinding::checked(
        hint.clone(),
        tensor_digest(backend.as_ref(), &hint, context)?,
    )?;
    let model = ControlModelBinding::checked(
        sd15_model_family_identity()?,
        patch_graph.identity(),
        model_digest,
        FIXTURE_CONTROL_EXECUTOR_DIGEST,
        DType::F32,
        DeviceId::CPU,
    )?;
    let control = ControlNet::checked(
        ControlBase::checked(
            0.125,
            StrengthType::Constant,
            ControlPercentWindow::checked(0.0, 1.0)?,
            false,
            Some(DType::F32),
        )?,
        model,
        hint,
        1,
        ResizeMode::NearestExact,
        ControlHintPreprocess::Identity,
        None,
        Vec::new(),
        false,
        Vec::new(),
    )?;
    let chain = Arc::new(ControlChain::checked(vec![ControlNode::ControlNet(
        control,
    )])?);
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(FixtureControlExecutor {
        calls: calls.clone(),
        mode,
    });
    let control_execution = PreboundControlExecution::checked(chain, executor)?;
    Ok((
        fixture.with_prebound_conditioning(
            patch_graph,
            Some(control_execution),
            patched.cache_identity().to_owned(),
        )?,
        calls,
    ))
}

struct MutableConditioningProvider {
    inner: NativeDiffusionFixture,
    cache_identities: Mutex<CanonicalNativeDiffusionCacheIdentities>,
    load_calls: AtomicUsize,
}

impl MutableConditioningProvider {
    fn checked(
        inner: NativeDiffusionFixture,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeImageRuntimeError> {
        let cache_identities = NativeDiffusionProvider::cache_identities(&inner, cancellation)?;
        Ok(Self {
            inner,
            cache_identities: Mutex::new(cache_identities),
            load_calls: AtomicUsize::new(0),
        })
    }

    fn replace_conditioning(
        &self,
        conditioning: CanonicalConditioningCacheIdentities,
    ) -> Result<(), NativeImageRuntimeError> {
        let mut identities = self.cache_identities.lock().map_err(|_| {
            NativeImageRuntimeError::Registry(
                "mutable conditioning fixture lock is poisoned".to_owned(),
            )
        })?;
        *identities = CanonicalNativeDiffusionCacheIdentities::checked(
            identities.model_digest(),
            identities.tokenizer_digest(),
            identities.clip().clone(),
            identities.vae().clone(),
            conditioning,
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        Ok(())
    }
}

impl NativeDiffusionProvider for MutableConditioningProvider {
    fn cache_identities(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<CanonicalNativeDiffusionCacheIdentities, NativeImageRuntimeError> {
        if cancellation.is_cancelled() {
            return Err(NativeImageRuntimeError::Cancelled);
        }
        let identities = self.cache_identities.lock().map_err(|_| {
            NativeImageRuntimeError::Registry(
                "mutable conditioning fixture lock is poisoned".to_owned(),
            )
        })?;
        if cancellation.is_cancelled() {
            return Err(NativeImageRuntimeError::Cancelled);
        }
        Ok(identities.clone())
    }

    fn load(
        &self,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDiffusionBundle, NativeImageRuntimeError> {
        self.load_calls.fetch_add(1, Ordering::SeqCst);
        NativeDiffusionProvider::load(&self.inner, backend, context)
    }
}

#[test]
fn val_conditioning_001() -> Result<(), Box<dyn Error>> {
    let cancellation: CancellationToken = Default::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let setup_workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let setup_context =
        backend.execution_context(StreamId::DEFAULT, setup_workspace.clone(), &cancellation);
    let mut executed_case_ids = assert_conditioning_value_contract_cases(&backend, &setup_context)?;
    executed_case_ids.extend(assert_guidance_contract_cases(&backend, &setup_context)?);
    assert_eq!(executed_case_ids.len(), 13);
    assert_eq!(setup_workspace.in_use_bytes(), 0);
    let (fixture, control_calls) =
        prebound_fixture(&backend, &setup_context, FixtureControlMode::Success)?;
    assert_eq!(setup_workspace.in_use_bytes(), 0);
    let base_fixture = NativeDiffusionFixture::checked_in();
    let fixture_identities = fixture.cache_identities(&cancellation)?;
    let base_identities = base_fixture.cache_identities(&cancellation)?;
    let fixture_conditioning = fixture_identities.conditioning();
    let base_conditioning = base_identities.conditioning();
    assert_ne!(fixture_conditioning, base_conditioning);
    assert_ne!(
        fixture_conditioning.model_patch(),
        base_conditioning.model_patch()
    );
    assert_ne!(fixture_conditioning.control(), base_conditioning.control());
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-guidance-hook-hooks-weighthook-03327446",
        GUIDANCE_TASK,
    )?;

    let mutable_provider = Arc::new(MutableConditioningProvider::checked(
        fixture.clone(),
        &cancellation,
    )?);
    let provider: Arc<dyn NativeDiffusionProvider> = mutable_provider.clone();
    let mut plan = compile_native_diffusion_workflow(WORKFLOW, &BTreeSet::new(), provider.clone())?;
    plan.prompt_id =
        sim_types::PromptId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3590));
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3591)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider.clone(),
    )?;
    let execution_workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let result = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x3592)),
        Default::default(),
        0,
        execution_workspace.clone(),
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "prebound native conditioning failed: {:?}",
        result.report.error
    );
    assert_eq!(result.executed_node_count, 7);
    assert!(control_calls.load(Ordering::SeqCst) > 0);
    assert_eq!(execution_workspace.in_use_bytes(), 0);
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-get-additional-models-7ba596bf",
        GUIDANCE_TASK,
    )?;
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-prepare-sampling-b141c606",
        GUIDANCE_TASK,
    )?;

    let exact_identities = mutable_provider.cache_identities(&cancellation)?;
    let exact_conditioning = exact_identities.conditioning();
    let replacement_control = if exact_conditioning.control().starts_with('f') {
        "e".repeat(64)
    } else {
        "f".repeat(64)
    };
    let replacement_conditioning = CanonicalConditioningCacheIdentities::checked(
        exact_conditioning.conditioning(),
        exact_conditioning.guidance(),
        exact_conditioning.model_patch(),
        exact_conditioning.model_execution(),
        replacement_control,
    )?;
    assert_ne!(
        replacement_conditioning.control(),
        exact_conditioning.control()
    );
    assert_ne!(
        replacement_conditioning.execution(),
        exact_conditioning.execution()
    );
    let load_calls_before_mutation = mutable_provider.load_calls.load(Ordering::SeqCst);
    let control_calls_before_mutation = control_calls.load(Ordering::SeqCst);
    mutable_provider.replace_conditioning(replacement_conditioning.clone())?;
    let mutated_identities = mutable_provider.cache_identities(&cancellation)?;
    assert_eq!(
        mutated_identities.model_digest(),
        exact_identities.model_digest()
    );
    assert_eq!(
        mutated_identities.tokenizer_digest(),
        exact_identities.tokenizer_digest()
    );
    assert_eq!(mutated_identities.clip(), exact_identities.clip());
    assert_eq!(mutated_identities.vae(), exact_identities.vae());
    assert_eq!(mutated_identities.conditioning(), &replacement_conditioning);
    let stale_workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let stale = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x3593)),
        Default::default(),
        0,
        stale_workspace.clone(),
    )?;
    assert_eq!(
        stale.report.state,
        AttemptState::Failed,
        "stale provider identity unexpectedly executed: {:?}",
        stale.report
    );
    assert!(
        stale
            .report
            .error
            .as_deref()
            .is_some_and(|error| error.contains(
                "native diffusion provider conditioning identity does not match its loaded bundle"
            ))
    );
    assert_eq!(stale.executed_node_count, 0);
    assert!(stale.report.outputs.is_empty());
    assert!(stale.output_proposals.is_empty());
    assert!(stale.report.events.iter().all(|event| !matches!(
        &event.kind,
        AttemptEventKind::OutputPrepared { .. } | AttemptEventKind::OutputAvailable { .. }
    )));
    assert_eq!(
        mutable_provider.load_calls.load(Ordering::SeqCst),
        load_calls_before_mutation
    );
    assert_eq!(
        control_calls.load(Ordering::SeqCst),
        control_calls_before_mutation
    );
    assert_eq!(stale_workspace.in_use_bytes(), 0);

    for (mode, expected_state, expected_code, attempt_suffix) in [
        (
            FixtureControlMode::Cancelled,
            AttemptState::Interrupted,
            "native_diffusion_cancelled",
            0x3594_u128,
        ),
        (
            FixtureControlMode::AllocationFailed,
            AttemptState::Failed,
            "native_diffusion_resource_exhausted",
            0x3595_u128,
        ),
    ] {
        let failure_setup_workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
        let failure_setup_context = backend.execution_context(
            StreamId::DEFAULT,
            failure_setup_workspace.clone(),
            &cancellation,
        );
        let (failure_fixture, failure_calls) =
            prebound_fixture(&backend, &failure_setup_context, mode)?;
        assert_eq!(failure_setup_workspace.in_use_bytes(), 0);
        let failure_provider: Arc<dyn NativeDiffusionProvider> = Arc::new(failure_fixture);
        let mut failure_plan = compile_native_diffusion_workflow(
            WORKFLOW,
            &BTreeSet::new(),
            failure_provider.clone(),
        )?;
        failure_plan.prompt_id = sim_types::PromptId(Uuid::from_u128(
            0x5349_4d00_0000_0000_0000_0000_0000_0000 | attempt_suffix,
        ));
        let failure_executor = NativeImageExecutor::new_with_diffusion_provider(
            ProfileId(Uuid::from_u128(
                0x5349_4d00_0000_0000_0000_0000_0001_0000 | attempt_suffix,
            )),
            BTreeMap::new(),
            true,
            backend.clone(),
            failure_provider,
        )?;
        let failure_workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
        let failure = failure_executor.execute_blocking(
            &failure_plan,
            AttemptId(Uuid::from_u128(attempt_suffix)),
            Default::default(),
            0,
            failure_workspace.clone(),
        )?;
        assert_eq!(failure.report.state, expected_state);
        assert!(
            failure
                .report
                .error
                .as_deref()
                .is_some_and(|error| error.contains(expected_code)),
            "typed ControlNet failure did not retain {expected_code}: {:?}",
            failure.report.error
        );
        if matches!(mode, FixtureControlMode::AllocationFailed) {
            assert!(
                failure
                    .report
                    .error
                    .as_deref()
                    .is_some_and(|error| error.contains("injected ControlNet allocation failure"))
            );
        }
        assert_eq!(failure_calls.load(Ordering::SeqCst), 1);
        assert_eq!(failure.executed_node_count, 0);
        assert!(failure.report.outputs.is_empty());
        assert!(failure.output_proposals.is_empty());
        assert!(failure.report.events.iter().all(|event| !matches!(
            &event.kind,
            AttemptEventKind::OutputPrepared { .. } | AttemptEventKind::OutputAvailable { .. }
        )));
        assert_eq!(failure_workspace.in_use_bytes(), 0);
    }
    record_contract_case(
        &mut executed_case_ids,
        "conditioning-guidance-hook-sampler-helpers-cleanup-models-6f147c97",
        GUIDANCE_TASK,
    )?;

    let constrained_provider: Arc<dyn NativeDiffusionProvider> = Arc::new(fixture);
    let constrained_executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3596)),
        BTreeMap::new(),
        true,
        backend,
        constrained_provider,
    )?;
    let constrained_workspace = workspace_authority.authorize_workspace(1024)?;
    let constrained = constrained_executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x3596)),
        Default::default(),
        0,
        constrained_workspace.clone(),
    )?;
    assert_eq!(constrained.report.state, AttemptState::Failed);
    assert!(
        constrained
            .report
            .error
            .as_deref()
            .is_some_and(|error| error.contains("native_diffusion_resource_exhausted"))
    );
    assert!(constrained.report.outputs.is_empty());
    assert!(constrained.output_proposals.is_empty());
    assert!(constrained.report.events.iter().all(|event| !matches!(
        &event.kind,
        AttemptEventKind::OutputPrepared { .. } | AttemptEventKind::OutputAvailable { .. }
    )));
    assert_eq!(constrained_workspace.in_use_bytes(), 0);

    write_artifact(&executed_case_ids)?;
    Ok(())
}

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let signatures = [
        format!("def {symbol}("),
        format!("async def {symbol}("),
        format!("class {symbol}("),
        format!("class {symbol}:"),
    ];
    let matches = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim_start_matches([' ', '\t']);
            signatures
                .iter()
                .any(|signature| trimmed.starts_with(signature))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let [start] = matches.as_slice() else {
        return Err(format!(
            "expected exactly one Python definition for {symbol}, found {}",
            matches.len()
        )
        .into());
    };
    let indentation = lines[*start].len() - lines[*start].trim_start_matches([' ', '\t']).len();
    let mut header_complete = lines[*start].trim_end().ends_with(':');
    let mut body_seen = false;
    let mut end = *start + 1;
    while let Some(line) = lines.get(end) {
        let trimmed = line.trim_start_matches([' ', '\t']);
        let content = trimmed.trim_end_matches(['\r', '\n']);
        if content.is_empty() || content.starts_with('#') {
            end += 1;
            continue;
        }
        let line_indentation = line.len() - trimmed.len();
        if !header_complete {
            header_complete = line_indentation == indentation && content.ends_with(':');
            end += 1;
            continue;
        }
        if body_seen && line_indentation <= indentation {
            break;
        }
        if line_indentation > indentation {
            body_seen = true;
        }
        end += 1;
    }
    if !body_seen {
        return Err(format!("Python definition {symbol} has no body").into());
    }
    while end > *start + 1 {
        let content = lines[end - 1].trim();
        if content.is_empty() || content.starts_with('#') {
            end -= 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(lines[*start..end].concat().as_bytes())
    ))
}

fn write_artifact(executed_case_ids: &BTreeSet<&str>) -> Result<(), Box<dyn Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = fs::read_to_string(
        repository
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut contracts = Vec::new();
    let mut counts = BTreeMap::new();
    let expected_cases = CONTRACT_CASES
        .iter()
        .map(|(contract_id, task_id, case_id)| (*contract_id, (*task_id, *case_id)))
        .collect::<BTreeMap<_, _>>();
    if expected_cases.len() != CONTRACT_CASES.len() {
        return Err("VAL-CONDITIONING-001 contract mapping contains duplicate IDs".into());
    }
    let expected_case_ids = CONTRACT_CASES
        .iter()
        .map(|(_, _, case_id)| *case_id)
        .collect::<BTreeSet<_>>();
    if expected_case_ids.len() != CONTRACT_CASES.len() {
        return Err("VAL-CONDITIONING-001 contract mapping contains duplicate case IDs".into());
    }
    if executed_case_ids != &expected_case_ids {
        return Err(format!(
            "VAL-CONDITIONING-001 executed cases differ from the exact contract mapping: expected {expected_case_ids:?}, observed {executed_case_ids:?}"
        )
        .into());
    }
    let mut seen_contract_ids = BTreeSet::new();
    for line in catalog.lines().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 15 || !matches!(columns[8], CONDITIONING_TASK | GUIDANCE_TASK) {
            continue;
        }
        let Some((expected_task, case_id)) = expected_cases.get(columns[0]).copied() else {
            return Err(format!(
                "conditioning catalog contains unexpected VAL-CONDITIONING-001 contract {}",
                columns[0]
            )
            .into());
        };
        if columns[8] != expected_task || !seen_contract_ids.insert(columns[0]) {
            return Err(format!(
                "conditioning contract {} has a duplicate or mismatched task",
                columns[0]
            )
            .into());
        }
        let source = fs::read(repository.join(columns[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), columns[5]);
        assert_eq!(python_symbol_sha256(&source, columns[3])?, columns[6]);
        *counts.entry(columns[8].to_owned()).or_insert(0_usize) += 1;
        contracts.push(json!({
            "contract_id": columns[0],
            "task_id": columns[8],
            "source_sha256": columns[5],
            "symbol_sha256": columns[6],
            "status": "passed",
            "case_ids": [case_id],
        }));
    }
    assert_eq!(counts.get(CONDITIONING_TASK), Some(&5));
    assert_eq!(counts.get(GUIDANCE_TASK), Some(&12));
    assert_eq!(contracts.len(), 17);
    if seen_contract_ids != expected_cases.keys().copied().collect() {
        return Err("conditioning catalog does not exactly cover the 17 contract mappings".into());
    }
    let implementation_path = "crates/sim_test_support/tests/native_conditioning_integration.rs";
    let implementation = fs::read(repository.join(implementation_path))?;
    let task_result = |path: &str, case_ids: &[&str], passed: usize| {
        let implementation = fs::read(repository.join(path))?;
        Ok::<_, Box<dyn Error>>(json!({
            "status": "passed",
            "passed": passed,
            "failed": 0,
            "skipped": 0,
            "case_ids": case_ids,
            "implementations": [{
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(implementation)),
            }],
        }))
    };
    let task_results = BTreeMap::from([
        (
            CONDITIONING_TASK,
            task_result(
                "crates/sim_model/src/conditioning.rs",
                &[
                    "conditioning:all-contracts",
                    "conditioning:values-regions-masks",
                    "conditioning:cancellation-oom-workspace-ownership",
                ],
                5,
            )?,
        ),
        (
            GUIDANCE_TASK,
            task_result(
                "crates/sim_sampler/src/guidance.rs",
                &[
                    "guidance:all-contracts",
                    "guidance:cfg-hooks-batching-regions",
                    "guidance:cancellation-oom-workspace-ownership",
                ],
                12,
            )?,
        ),
    ]);
    let artifact = json!({
        "schema_version": 1,
        "validation_id": "VAL-CONDITIONING-001",
        "overall_status": "passed",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": { "passed": 17, "failed": 0, "skipped": 0 },
        "implementation": {
            "path": implementation_path,
            "sha256": format!("{:x}", Sha256::digest(implementation)),
        },
        "task_results": task_results,
        "contracts": contracts,
    });
    let directory = repository.join("target/comfy-parity");
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("val-conditioning-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize)]
struct StyleStateFixture {
    key: String,
    shape: Vec<u64>,
    storage_bits: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleDtypeFixture {
    state: Vec<StyleStateFixture>,
    source_identity_sha256: String,
    projected_identity_sha256: String,
    output_shape: Vec<u64>,
    output_bits: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleProfileFixture {
    input_shape: Vec<u64>,
    input_bits: Vec<u32>,
    dtypes: BTreeMap<String, StyleDtypeFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct PhotoMakerDtypeFixture {
    state: Vec<StyleStateFixture>,
    source_identity_sha256: String,
    projected_identity_sha256: String,
    pooled_bits: Vec<u32>,
    first_projection_bits: Vec<u32>,
    second_projection_bits: Vec<u32>,
    identity_bits: Vec<u32>,
    output_shape: Vec<u64>,
    output_bits: Vec<u32>,
    canonical_output_bits: Vec<u32>,
    output_ulp_bound: u32,
    output_ulp_rejected_distance: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct PhotoMakerProfileFixture {
    image_shape: Vec<u64>,
    image_bits: Vec<u32>,
    prompt_shape: Vec<u64>,
    prompt_bits: Vec<u32>,
    mask_shape: Vec<u64>,
    mask: Vec<bool>,
    mask_positions: Vec<usize>,
    dtypes: BTreeMap<String, PhotoMakerDtypeFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenStateFixture {
    key: String,
    shape: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenDtypeFixture {
    prepared_shape: Vec<u64>,
    prepared_bits: Vec<u32>,
    visual_shape: Vec<u64>,
    visual_bits: Vec<u32>,
    output_bits: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenFuserFixture {
    namespace: String,
    region: String,
    block_index: u8,
    transformer_index: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenPositionFixture {
    embedding_bits: Vec<u32>,
    height: f32,
    width: f32,
    y: f32,
    x: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenHeadRuleFixture {
    key_dimension: usize,
    query_dimension: usize,
    heads: usize,
    head_dimension: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct GligenProfileFixture {
    key_dimension: usize,
    query_dimension: usize,
    state: Vec<GligenStateFixture>,
    fusers: Vec<GligenFuserFixture>,
    positions: Vec<GligenPositionFixture>,
    latent_shape: [u64; 4],
    dtypes: BTreeMap<String, GligenDtypeFixture>,
    head_rule_cases: Vec<GligenHeadRuleFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleMutationFixture {
    profile: String,
    key: String,
    index: usize,
    delta_bits: u32,
    source_identity_sha256: String,
    output_bits: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct PhotoMakerMutationFixture {
    key: String,
    index: usize,
    delta_bits: u32,
    source_identity_sha256: String,
    output_bits: Vec<u32>,
    canonical_output_bits: Vec<u32>,
    output_ulp_bound: u32,
    output_ulp_rejected_distance: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleStateModificationFixture {
    key: String,
    index: usize,
    value_bits: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleAttentionDiscriminatorFixture {
    state_modifications: Vec<StyleStateModificationFixture>,
    source_identity_sha256: String,
    input_shape: Vec<u64>,
    input_bits: Vec<u32>,
    output_shape: Vec<u64>,
    output_bits: Vec<u32>,
    batch_outputs_differ: bool,
    query_key_are_asymmetric: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct QuickGeluFixture {
    coefficient_bits: u32,
    input_bits: u32,
    scaled_bits: u32,
    sigmoid_bits: u32,
    output_bits: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleDiscriminators {
    signed_zero_input_bits: u32,
    signed_zero_after_add_bits: u32,
    quick_gelu: QuickGeluFixture,
    detection_precedence: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct StyleModelOracle {
    format: String,
    reduced_profiles_are_source_exact: bool,
    source_dimensions: serde_json::Value,
    reduced_dimensions: serde_json::Value,
    style: StyleProfileFixture,
    redux: StyleProfileFixture,
    photomaker: PhotoMakerProfileFixture,
    gligen: GligenProfileFixture,
    attention_discriminator: StyleAttentionDiscriminatorFixture,
    mutations: BTreeMap<String, StyleMutationFixture>,
    photomaker_mutations: BTreeMap<String, PhotoMakerMutationFixture>,
    discriminators: StyleDiscriminators,
    pinned_sources: BTreeMap<String, String>,
    generator_sha256: String,
    source_graph_sha256: String,
    generator_command: String,
}

fn style_model_oracle() -> Result<StyleModelOracle, Box<dyn Error>> {
    Ok(serde_json::from_slice(STYLE_MODEL_ORACLE)?)
}

fn fixture_dtype(name: &str) -> Result<DType, Box<dyn Error>> {
    match name {
        "float32" => Ok(DType::F32),
        "float16" => Ok(DType::F16),
        "bfloat16" => Ok(DType::Bf16),
        value => Err(format!("unsupported fixture dtype {value}").into()),
    }
}

fn storage_bytes(entry: &StyleStateFixture, dtype: DType) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    let byte_width = usize::try_from(dtype.byte_width())?;
    bytes.try_reserve_exact(
        entry
            .storage_bits
            .len()
            .checked_mul(byte_width)
            .ok_or("fixture storage byte count overflowed")?,
    )?;
    for value in &entry.storage_bits {
        match dtype {
            DType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            DType::F16 | DType::Bf16 => {
                bytes.extend_from_slice(&u16::try_from(*value)?.to_le_bytes())
            }
            _ => return Err("fixture storage dtype is not floating point".into()),
        }
    }
    Ok(bytes)
}

fn projected_f32_bits(value: u32, dtype: DType) -> Result<u32, Box<dyn Error>> {
    Ok(match dtype {
        DType::F32 => value,
        DType::Bf16 => value << 16,
        DType::F16 => {
            let decoded = DType::F16.decode_scalar(&u16::try_from(value)?.to_le_bytes())?;
            match decoded {
                sim_tensor::DecodedScalar::Real(value) => (value as f32).to_bits(),
                _ => return Err("F16 fixture decoded to a non-real scalar".into()),
            }
        }
        _ => return Err("fixture storage dtype is not floating point".into()),
    })
}

fn fixture_state_identity(
    state: &[StyleStateFixture],
    dtype_name: &str,
    projected: bool,
) -> Result<String, Box<dyn Error>> {
    let dtype = fixture_dtype(dtype_name)?;
    let mut hasher = Sha256::new();
    hasher.update(b"conditioning-auxiliary-state-v1\0");
    hasher.update(if projected {
        b"float32".as_slice()
    } else {
        dtype_name.as_bytes()
    });
    for entry in state {
        hasher.update(u64::try_from(entry.key.len())?.to_le_bytes());
        hasher.update(entry.key.as_bytes());
        for dimension in &entry.shape {
            hasher.update(dimension.to_le_bytes());
        }
        for value in &entry.storage_bits {
            if projected {
                hasher.update(projected_f32_bits(*value, dtype)?.to_le_bytes());
            } else if dtype == DType::F32 {
                hasher.update(value.to_le_bytes());
            } else {
                hasher.update(u16::try_from(*value)?.to_le_bytes());
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn upload_style_state(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    fixture: &StyleDtypeFixture,
    dtype_name: &str,
) -> Result<Vec<(String, Tensor)>, Box<dyn Error>> {
    let dtype = fixture_dtype(dtype_name)?;
    let mut state = Vec::new();
    state.try_reserve_exact(fixture.state.len())?;
    for entry in &fixture.state {
        let descriptor = TensorDescriptor::contiguous(
            entry.shape.clone(),
            dtype,
            DeviceId::CPU,
            context.stream,
        )?;
        let (tensor, event) =
            backend.upload_bytes(descriptor, &storage_bytes(entry, dtype)?, context)?;
        backend.wait_event(event, context)?;
        state.push((entry.key.clone(), tensor));
    }
    Ok(state)
}

fn upload_photomaker_state(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    fixture: &PhotoMakerDtypeFixture,
    dtype_name: &str,
) -> Result<Vec<(String, Tensor)>, Box<dyn Error>> {
    let dtype = fixture_dtype(dtype_name)?;
    let mut state = Vec::new();
    state.try_reserve_exact(fixture.state.len())?;
    for entry in &fixture.state {
        let descriptor = TensorDescriptor::contiguous(
            entry.shape.clone(),
            dtype,
            DeviceId::CPU,
            context.stream,
        )?;
        let (tensor, event) =
            backend.upload_bytes(descriptor, &storage_bytes(entry, dtype)?, context)?;
        backend.wait_event(event, context)?;
        state.push((entry.key.clone(), tensor));
    }
    Ok(state)
}

fn style_artifact_sha256(profile: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("task393:{profile}:reduced-v1"))
    )
}

fn style_checkpoint(
    profile: &str,
    state: Vec<(String, Tensor)>,
    memory_budget_bytes: u64,
) -> NativeStyleModelCheckpoint {
    NativeStyleModelCheckpoint {
        artifact_sha256: style_artifact_sha256(profile),
        ordered_state: state,
        memory_budget_bytes,
    }
}

fn fixture_clip_output(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    profile: &StyleProfileFixture,
) -> Result<ClipVisionOutput, Box<dyn Error>> {
    fixture_clip_output_values(backend, context, &profile.input_shape, &profile.input_bits)
}

fn fixture_clip_output_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    input_bits: &[u32],
) -> Result<ClipVisionOutput, Box<dyn Error>> {
    let values = input_bits
        .iter()
        .map(|value| f32::from_bits(*value))
        .collect::<Vec<_>>();
    let hidden = tensor_from_f32(backend, shape, &values, context)?;
    let embeds = tensor_from_f32(
        backend,
        &[shape[0], shape[2]],
        &vec![0.0; usize::try_from(shape[0] * shape[2])?],
        context,
    )?;
    Ok(ClipVisionOutput::checked(
        hidden,
        None,
        embeds,
        None,
        vec![[3, 16, 16]; usize::try_from(shape[0])?],
    )?)
}

fn assert_fixture_output(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    output: &Tensor,
    fixture: &StyleDtypeFixture,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(output.descriptor().shape(), fixture.output_shape);
    let bits = tensor_to_f32(backend, output, context)?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits, fixture.output_bits);
    Ok(())
}

fn assert_reconstruction(
    reconstructed: &NativeStyleModelCheckpoint,
    original: &[(String, Tensor)],
) -> Result<(), Box<dyn Error>> {
    assert_eq!(reconstructed.ordered_state.len(), original.len());
    for ((actual_key, actual), (expected_key, expected)) in
        reconstructed.ordered_state.iter().zip(original)
    {
        assert_eq!(actual_key, expected_key);
        assert_eq!(actual.descriptor(), expected.descriptor());
        assert_eq!(actual.storage_id(), expected.storage_id());
        assert_eq!(actual.contiguous_bytes()?, expected.contiguous_bytes()?);
    }
    Ok(())
}

fn assert_style_payload_cross_role_denial(payload: &NativeModelPayload) {
    assert!(payload.model().is_none());
    assert!(payload.native_family_model_resource().is_none());
    assert!(payload.clip().is_none());
    assert!(payload.vae().is_none());
    assert!(payload.structured_vae().is_none());
    assert!(payload.audio_encoder_resource().is_none());
    assert!(payload.optical_flow_resource().is_none());
    assert!(payload.clip_vision_resource().is_none());
    assert!(payload.decoder_clip_resource().is_none());
    assert!(payload.qwen_multimodal_resource().is_none());
    assert!(payload.gemma_multimodal_resource().is_none());
    assert!(payload.native_clip_resource().is_none());
    assert!(payload.sdpose_model_resource().is_none());
    assert!(payload.frame_interpolation_resource().is_none());
    assert!(payload.latent_upscale_model_resource().is_none());
    assert!(payload.background_removal_resource().is_none());
    assert!(payload.depth_anything_3_resource().is_none());
    assert!(payload.moge_resource().is_none());
    assert!(payload.gligen_resource().is_none());
    assert!(payload.photomaker_resource().is_none());
}

fn assert_photomaker_payload_cross_role_denial(payload: &NativeModelPayload) {
    assert!(payload.model().is_none());
    assert!(payload.native_family_model_resource().is_none());
    assert!(payload.clip().is_none());
    assert!(payload.vae().is_none());
    assert!(payload.structured_vae().is_none());
    assert!(payload.audio_encoder_resource().is_none());
    assert!(payload.optical_flow_resource().is_none());
    assert!(payload.clip_vision_resource().is_none());
    assert!(payload.decoder_clip_resource().is_none());
    assert!(payload.qwen_multimodal_resource().is_none());
    assert!(payload.gemma_multimodal_resource().is_none());
    assert!(payload.native_clip_resource().is_none());
    assert!(payload.sdpose_model_resource().is_none());
    assert!(payload.frame_interpolation_resource().is_none());
    assert!(payload.latent_upscale_model_resource().is_none());
    assert!(payload.background_removal_resource().is_none());
    assert!(payload.depth_anything_3_resource().is_none());
    assert!(payload.moge_resource().is_none());
    assert!(payload.gligen_resource().is_none());
    assert!(payload.style_model_resource().is_none());
}

fn assert_gligen_payload_cross_role_denial(payload: &NativeModelPayload) {
    assert!(payload.model().is_none());
    assert!(payload.native_family_model_resource().is_none());
    assert!(payload.clip().is_none());
    assert!(payload.vae().is_none());
    assert!(payload.structured_vae().is_none());
    assert!(payload.audio_encoder_resource().is_none());
    assert!(payload.optical_flow_resource().is_none());
    assert!(payload.clip_vision_resource().is_none());
    assert!(payload.decoder_clip_resource().is_none());
    assert!(payload.qwen_multimodal_resource().is_none());
    assert!(payload.gemma_multimodal_resource().is_none());
    assert!(payload.native_clip_resource().is_none());
    assert!(payload.sdpose_model_resource().is_none());
    assert!(payload.frame_interpolation_resource().is_none());
    assert!(payload.latent_upscale_model_resource().is_none());
    assert!(payload.background_removal_resource().is_none());
    assert!(payload.depth_anything_3_resource().is_none());
    assert!(payload.moge_resource().is_none());
    assert!(payload.photomaker_resource().is_none());
    assert!(payload.style_model_resource().is_none());
}

fn photomaker_artifact_sha256() -> String {
    format!("{:x}", Sha256::digest("task395:photomaker:reduced-v1"))
}

fn photomaker_checkpoint(
    state: Vec<(String, Tensor)>,
    nested: bool,
    memory_budget_bytes: u64,
) -> NativePhotoMakerCheckpoint {
    let ordered_entries = if nested {
        vec![NativePhotoMakerCheckpointEntry::Mapping {
            key: "id_encoder".to_owned(),
            ordered_state: state,
        }]
    } else {
        state
            .into_iter()
            .map(|(key, tensor)| NativePhotoMakerCheckpointEntry::Tensor { key, tensor })
            .collect()
    };
    NativePhotoMakerCheckpoint {
        artifact_sha256: photomaker_artifact_sha256(),
        ordered_entries,
        memory_budget_bytes,
    }
}

fn gligen_fixture_value(key: &str, index: usize, shape: &[u64]) -> Result<f32, Box<dyn Error>> {
    let width = shape.last().copied().unwrap_or(1);
    let width = usize::try_from(width)?;
    let output = index / width;
    let component = index % width;
    let variant = usize::from(key.contains("output_blocks"));
    let value = match key {
        "position_net.null_positive_feature" => (index + 1) as f32 * 0.03125,
        "position_net.null_position_feature" => (index % 9) as f32 * 0.0078125 - 0.03125,
        "position_net.linears.0.weight" => {
            if component == output % width {
                0.125 + (output % 3) as f32 * 0.015625
            } else {
                0.0
            }
        }
        "position_net.linears.0.bias" => (index % 7) as f32 * 0.00390625 - 0.01171875,
        "position_net.linears.2.weight" => {
            if component == output {
                0.5
            } else {
                0.0
            }
        }
        "position_net.linears.2.bias" => (index % 5) as f32 * 0.001953125 - 0.00390625,
        "position_net.linears.4.weight" => {
            if component == (output * 17 + 3) % width {
                0.25 + output as f32 * 0.03125
            } else {
                0.0
            }
        }
        "position_net.linears.4.bias" => index as f32 * 0.015625 - 0.015625,
        _ => {
            let suffix = key
                .split_once(".fuser.")
                .ok_or("GLIGEN fixture key has no fuser anchor")?
                .1;
            match suffix {
                "alpha_attn" => 0.375 + variant as f32 * 0.0625,
                "alpha_dense" => -0.3125 + variant as f32 * 0.03125,
                "linear.weight" => {
                    if component == output % width {
                        0.1875 + (output % 4) as f32 * 0.015625
                    } else {
                        0.0
                    }
                }
                "linear.bias" => (index % 5) as f32 * 0.0078125 - 0.015625,
                "attn.to_q.weight" => {
                    if component == output {
                        0.125 + variant as f32 * 0.015625
                    } else {
                        0.0
                    }
                }
                "attn.to_k.weight" => {
                    if component == (output + 1) % width {
                        0.09375
                    } else {
                        0.0
                    }
                }
                "attn.to_v.weight" => {
                    if component == output {
                        0.25
                    } else {
                        0.0
                    }
                }
                "attn.to_out.0.weight" => {
                    if component == output {
                        0.3125
                    } else {
                        0.0
                    }
                }
                "attn.to_out.0.bias" => (index % 3) as f32 * 0.00390625 - 0.00390625,
                "ff.net.0.proj.weight" => {
                    if component == output % width {
                        if output < width { 0.21875 } else { 0.15625 }
                    } else {
                        0.0
                    }
                }
                "ff.net.0.proj.bias" => (index % 7) as f32 * 0.002 - 0.006,
                "ff.net.2.weight" => {
                    if component == output {
                        0.28125
                    } else {
                        0.0
                    }
                }
                "ff.net.2.bias" => (index % 4) as f32 * 0.0025 - 0.0025,
                "norm1.weight" | "norm2.weight" => 1.0 + (index % 4) as f32 * 0.015625,
                "norm1.bias" | "norm2.bias" => (index % 5) as f32 * 0.001 - 0.002,
                _ => return Err(format!("unknown GLIGEN fixture suffix {suffix}").into()),
            }
        }
    };
    Ok(value)
}

fn upload_gligen_state(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    state_fixture: &[GligenStateFixture],
    dtype_name: &str,
) -> Result<Vec<(String, Tensor)>, Box<dyn Error>> {
    let dtype = fixture_dtype(dtype_name)?;
    let mut state = Vec::new();
    state.try_reserve_exact(state_fixture.len())?;
    for entry in state_fixture {
        let descriptor = TensorDescriptor::contiguous(
            entry.shape.clone(),
            dtype,
            DeviceId::CPU,
            context.stream,
        )?;
        let count = usize::try_from(descriptor.element_count()?)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(
            count
                .checked_mul(usize::try_from(dtype.byte_width())?)
                .ok_or("GLIGEN fixture byte count overflowed")?,
        )?;
        for index in 0..count {
            bytes.extend_from_slice(&dtype.encode_scalar(
                sim_tensor::Scalar::Float(f64::from(gligen_fixture_value(
                    &entry.key,
                    index,
                    &entry.shape,
                )?)),
                "task396.gligen-fixture",
                DeviceId::CPU,
            )?);
        }
        let (tensor, event) = backend.upload_bytes(descriptor, &bytes, context)?;
        backend.wait_event(event, context)?;
        state.push((entry.key.clone(), tensor));
    }
    Ok(state)
}

fn gligen_fixture_schema(
    key_dimension: usize,
    query_dimension: usize,
    namespaces: &[&str],
) -> Result<Vec<GligenStateFixture>, Box<dyn Error>> {
    let key = u64::try_from(key_dimension)?;
    let query = u64::try_from(query_dimension)?;
    let mut state = vec![
        GligenStateFixture {
            key: "position_net.null_positive_feature".to_owned(),
            shape: vec![key],
        },
        GligenStateFixture {
            key: "position_net.null_position_feature".to_owned(),
            shape: vec![64],
        },
        GligenStateFixture {
            key: "position_net.linears.0.weight".to_owned(),
            shape: vec![512, key + 64],
        },
        GligenStateFixture {
            key: "position_net.linears.0.bias".to_owned(),
            shape: vec![512],
        },
        GligenStateFixture {
            key: "position_net.linears.2.weight".to_owned(),
            shape: vec![512, 512],
        },
        GligenStateFixture {
            key: "position_net.linears.2.bias".to_owned(),
            shape: vec![512],
        },
        GligenStateFixture {
            key: "position_net.linears.4.weight".to_owned(),
            shape: vec![key, 512],
        },
        GligenStateFixture {
            key: "position_net.linears.4.bias".to_owned(),
            shape: vec![key],
        },
    ];
    for namespace in namespaces {
        let prefix = format!("{namespace}.fuser");
        for (suffix, shape) in [
            ("alpha_attn", vec![]),
            ("alpha_dense", vec![]),
            ("linear.weight", vec![query, key]),
            ("linear.bias", vec![query]),
            ("attn.to_q.weight", vec![query, query]),
            ("attn.to_k.weight", vec![query, query]),
            ("attn.to_v.weight", vec![query, query]),
            ("attn.to_out.0.weight", vec![query, query]),
            ("attn.to_out.0.bias", vec![query]),
            ("ff.net.0.proj.weight", vec![query * 2, query]),
            ("ff.net.0.proj.bias", vec![query * 2]),
            ("ff.net.2.weight", vec![query, query]),
            ("ff.net.2.bias", vec![query]),
            ("norm1.weight", vec![query]),
            ("norm1.bias", vec![query]),
            ("norm2.weight", vec![query]),
            ("norm2.bias", vec![query]),
        ] {
            state.push(GligenStateFixture {
                key: format!("{prefix}.{suffix}"),
                shape,
            });
        }
    }
    Ok(state)
}

fn gligen_checkpoint(
    state: Vec<(String, Tensor)>,
    memory_budget_bytes: u64,
) -> NativeGligenCheckpoint {
    NativeGligenCheckpoint {
        artifact_sha256: format!("{:x}", Sha256::digest("task396:gligen:reduced-v1")),
        ordered_state: state,
        memory_budget_bytes,
    }
}

fn gligen_positions(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    fixture: &GligenProfileFixture,
) -> Result<Vec<NativeGligenPositionParameter>, Box<dyn Error>> {
    fixture
        .positions
        .iter()
        .map(|position| {
            let values = position
                .embedding_bits
                .iter()
                .map(|value| f32::from_bits(*value))
                .collect::<Vec<_>>();
            Ok(NativeGligenPositionParameter {
                embedding: tensor_from_f32(
                    backend,
                    &[1, u64::try_from(values.len())?],
                    &values,
                    context,
                )?,
                height: position.height,
                width: position.width,
                y: position.y,
                x: position.x,
            })
        })
        .collect()
}

fn photomaker_inputs(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    fixture: &PhotoMakerProfileFixture,
) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error>> {
    let image = tensor_from_f32(
        backend,
        &fixture.image_shape,
        &fixture
            .image_bits
            .iter()
            .map(|value| f32::from_bits(*value))
            .collect::<Vec<_>>(),
        context,
    )?;
    let prompt = tensor_from_f32(
        backend,
        &fixture.prompt_shape,
        &fixture
            .prompt_bits
            .iter()
            .map(|value| f32::from_bits(*value))
            .collect::<Vec<_>>(),
        context,
    )?;
    let descriptor = TensorDescriptor::contiguous(
        fixture.mask_shape.clone(),
        DType::Bool,
        DeviceId::CPU,
        context.stream,
    )?;
    let bytes = fixture
        .mask
        .iter()
        .map(|value| u8::from(*value))
        .collect::<Vec<_>>();
    let (mask, event) = backend.upload_bytes(descriptor, &bytes, context)?;
    backend.wait_event(event, context)?;
    Ok((image, prompt, mask))
}

fn assert_photomaker_reconstruction(
    reconstructed: &NativePhotoMakerCheckpoint,
    original: &[(String, Tensor)],
    nested: bool,
) -> Result<(), Box<dyn Error>> {
    let actual = if nested {
        let [NativePhotoMakerCheckpointEntry::Mapping { key, ordered_state }] =
            reconstructed.ordered_entries.as_slice()
        else {
            return Err("PhotoMaker nested reconstruction lost its sole wrapper".into());
        };
        assert_eq!(key, "id_encoder");
        ordered_state.clone()
    } else {
        reconstructed
            .ordered_entries
            .iter()
            .map(|entry| match entry {
                NativePhotoMakerCheckpointEntry::Tensor { key, tensor } => {
                    Ok((key.clone(), tensor.clone()))
                }
                NativePhotoMakerCheckpointEntry::Mapping { .. } => {
                    Err("flat PhotoMaker reconstruction gained a wrapper")
                }
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    assert_eq!(actual.len(), original.len());
    for ((actual_key, actual), (expected_key, expected)) in actual.iter().zip(original) {
        assert_eq!(actual_key, expected_key);
        assert_eq!(actual.descriptor(), expected.descriptor());
        assert_eq!(actual.storage_id(), expected.storage_id());
        assert_eq!(actual.contiguous_bytes()?, expected.contiguous_bytes()?);
    }
    Ok(())
}

fn photomaker_ordered_f32_bits(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn photomaker_ulp_distance(left: u32, right: u32) -> u32 {
    photomaker_ordered_f32_bits(left).abs_diff(photomaker_ordered_f32_bits(right))
}

fn assert_photomaker_output_bound(
    actual: &[u32],
    expected: &[u32],
    canonical: &[u32],
    bound: u32,
    rejected_distance: u32,
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(canonical.len(), expected.len());
    assert_eq!(actual, canonical);
    let canonical_source_max = canonical
        .iter()
        .zip(expected)
        .map(|(left, right)| photomaker_ulp_distance(*left, *right))
        .max()
        .unwrap_or(0);
    assert_eq!(canonical_source_max, bound);
    assert_eq!(rejected_distance, bound + 1);
    let baseline = 0x3f00_0000;
    assert!(photomaker_ulp_distance(baseline, baseline + bound) <= bound);
    assert!(photomaker_ulp_distance(baseline, baseline + rejected_distance) > bound);
}

#[test]
fn conditioning_auxiliary_fixture_integrity() -> Result<(), Box<dyn Error>> {
    let oracle = style_model_oracle()?;
    assert_eq!(
        oracle.format,
        "conditioning-auxiliary-resource-foundation-v1"
    );
    assert!(!oracle.reduced_profiles_are_source_exact);
    assert_eq!(oracle.style.dtypes["float32"].state.len(), 42);
    assert_eq!(oracle.redux.dtypes["float32"].state.len(), 4);
    assert_eq!(oracle.photomaker.dtypes["float32"].state.len(), 407);
    assert_eq!(oracle.gligen.state.len(), 42);
    assert_eq!(oracle.gligen.fusers.len(), 2);
    assert_eq!(oracle.gligen.fusers[0].transformer_index, 0);
    assert_eq!(oracle.gligen.fusers[1].transformer_index, 1);
    assert_eq!(oracle.gligen.head_rule_cases[0].heads, 1);
    assert_eq!(oracle.gligen.head_rule_cases[0].head_dimension, 64);
    assert_eq!(oracle.gligen.head_rule_cases[1].key_dimension, 768);
    assert_eq!(oracle.gligen.head_rule_cases[1].query_dimension, 8);
    assert_eq!(oracle.gligen.head_rule_cases[1].heads, 8);
    assert_eq!(oracle.gligen.head_rule_cases[1].head_dimension, 1);
    assert_eq!(
        oracle.style.dtypes["float32"].state[0].key,
        "style_embedding"
    );
    assert_eq!(oracle.style.dtypes["float32"].state[1].key, "proj");
    assert_eq!(oracle.style.dtypes["float32"].state[41].key, "ln_pre.bias");
    assert_eq!(
        oracle.redux.dtypes["float32"].state[0].key,
        "redux_up.weight"
    );
    assert_eq!(
        oracle.redux.dtypes["float32"].state[3].key,
        "redux_down.bias"
    );
    assert_eq!(
        oracle.discriminators.detection_precedence,
        ["style_embedding", "redux_down.weight"]
    );
    assert_eq!(oracle.discriminators.signed_zero_input_bits, 0x8000_0000);
    assert_eq!(oracle.discriminators.signed_zero_after_add_bits, 0);
    assert_eq!(oracle.source_dimensions["style"]["width"], 1024);
    assert_eq!(oracle.source_dimensions["redux"]["hidden"], 12288);
    assert_eq!(oracle.reduced_dimensions["style"]["width"], 8);
    assert_eq!(oracle.reduced_dimensions["redux"]["hidden"], 12);
    assert_eq!(oracle.source_dimensions["photomaker"]["hidden"], 1024);
    assert_eq!(oracle.source_dimensions["photomaker"]["state_count"], 407);
    assert_eq!(oracle.reduced_dimensions["photomaker"]["hidden"], 4);
    assert_eq!(oracle.reduced_dimensions["photomaker"]["prompt"], 8);
    assert_eq!(oracle.photomaker.mask_positions, [1, 3]);
    assert_eq!(oracle.photomaker.image_shape, [1, 2, 3, 4, 4]);
    assert_eq!(oracle.photomaker.prompt_shape, [1, 4, 8]);
    for fixture in oracle.photomaker.dtypes.values() {
        assert!(fixture.output_ulp_bound > 0);
        assert_eq!(
            fixture.output_ulp_rejected_distance,
            fixture.output_ulp_bound + 1
        );
    }
    assert_eq!(oracle.attention_discriminator.input_shape, [2, 2, 8]);
    assert_eq!(oracle.attention_discriminator.output_shape, [2, 2, 6]);
    assert_eq!(oracle.attention_discriminator.state_modifications.len(), 36);
    assert!(oracle.attention_discriminator.batch_outputs_differ);
    assert!(oracle.attention_discriminator.query_key_are_asymmetric);
    assert_eq!(
        oracle.generator_command,
        "PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/conditioning-auxiliary-resource-foundation/generate_oracle.py --check"
    );

    let manifest: serde_json::Value = serde_json::from_slice(STYLE_MODEL_MANIFEST)?;
    let provenance: serde_json::Value = serde_json::from_slice(STYLE_MODEL_PROVENANCE)?;
    assert_eq!(
        manifest["oracle_sha256"],
        format!("{:x}", Sha256::digest(STYLE_MODEL_ORACLE))
    );
    assert_eq!(
        manifest["generator_sha256"],
        format!("{:x}", Sha256::digest(STYLE_MODEL_GENERATOR))
    );
    assert_eq!(
        manifest["source_graph_sha256"],
        format!("{:x}", Sha256::digest(STYLE_MODEL_SOURCE_GRAPH))
    );
    assert_eq!(provenance["reduced_profiles_are_source_exact"], false);
    let source_graph = std::str::from_utf8(STYLE_MODEL_SOURCE_GRAPH)?;
    for forbidden in [
        "import torch",
        "import numpy",
        "import ctypes",
        "import subprocess",
    ] {
        assert!(!source_graph.contains(forbidden));
    }
    assert_eq!(
        oracle.generator_sha256,
        format!("{:x}", Sha256::digest(STYLE_MODEL_GENERATOR))
    );
    assert_eq!(
        oracle.source_graph_sha256,
        format!("{:x}", Sha256::digest(STYLE_MODEL_SOURCE_GRAPH))
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/nodes.py"],
        STYLE_MODEL_NODES_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/sd.py"],
        STYLE_MODEL_SD_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/ops.py"],
        STYLE_MODEL_OPS_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/t2i_adapter/adapter.py"],
        STYLE_ADAPTER_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/ldm/flux/redux.py"],
        FLUX_REDUX_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy_extras/nodes_photomaker.py"],
        PHOTOMAKER_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/clip_model.py"],
        CLIP_VISION_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/clip_vision.py"],
        PHOTOMAKER_CLIP_VISION_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/gligen.py"],
        GLIGEN_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/ldm/modules/attention.py"],
        GLIGEN_ATTENTION_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/ldm/modules/diffusionmodules/openaimodel.py"],
        GLIGEN_OPENAIMODEL_SOURCE_SHA256
    );
    assert_eq!(
        oracle.pinned_sources["projects/comfy/ComfyUI/comfy/samplers.py"],
        GLIGEN_SAMPLERS_SOURCE_SHA256
    );
    Ok(())
}

#[test]
fn style_model_resource() -> Result<(), Box<dyn Error>> {
    let oracle = style_model_oracle()?;
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);

    for (profile_name, profile) in [("style", &oracle.style), ("redux", &oracle.redux)] {
        let baseline = &profile.dtypes["float32"].output_bits;
        for dtype_name in ["float32", "float16", "bfloat16"] {
            let fixture = &profile.dtypes[dtype_name];
            assert_eq!(
                fixture.source_identity_sha256,
                fixture_state_identity(&fixture.state, dtype_name, false)?
            );
            assert_eq!(
                fixture.projected_identity_sha256,
                fixture_state_identity(&fixture.state, dtype_name, true)?
            );
            let original = upload_style_state(&backend, &context, fixture, dtype_name)?;
            let resource = Arc::new(NativeStyleModelResource::from_reduced_fixture(
                &backend,
                style_checkpoint(profile_name, original.clone(), STYLE_MODEL_FIXTURE_MEMORY),
                &context,
            )?);
            assert!(!resource.is_source_exact_profile());
            assert_eq!(resource.source_dtype(), fixture_dtype(dtype_name)?);
            resource.validate(&cancellation)?;
            let reconstructed = resource.reconstruct_checkpoint(&cancellation)?;
            assert_reconstruction(&reconstructed, &original)?;
            let clip_output = fixture_clip_output(&backend, &context, profile)?;
            let output = resource.get_cond(&backend, &clip_output, &context)?;
            assert_fixture_output(&backend, &context, &output, fixture)?;
            let payload =
                NativeModelPayload::style_model_test_fixture(resource.clone(), &cancellation)?;
            assert!(Arc::ptr_eq(
                payload
                    .style_model_resource()
                    .ok_or("STYLE_MODEL payload accessor is missing")?,
                &resource
            ));
            assert_style_payload_cross_role_denial(&payload);
            payload.validate()?;
            assert!(NativeModelPayload::style_model(resource, &cancellation).is_err());
        }
        assert!(!baseline.is_empty());
    }

    let attention = &oracle.attention_discriminator;
    let mut fixture = oracle.style.dtypes["float32"].clone();
    for modification in &attention.state_modifications {
        let entry = fixture
            .state
            .iter_mut()
            .find(|entry| entry.key == modification.key)
            .ok_or("attention discriminator key is missing")?;
        *entry
            .storage_bits
            .get_mut(modification.index)
            .ok_or("attention discriminator index is missing")? = modification.value_bits;
    }
    assert_eq!(
        attention.source_identity_sha256,
        fixture_state_identity(&fixture.state, "float32", false)?
    );
    let state = upload_style_state(&backend, &context, &fixture, "float32")?;
    let resource = NativeStyleModelResource::from_reduced_fixture(
        &backend,
        style_checkpoint("style", state, STYLE_MODEL_FIXTURE_MEMORY),
        &context,
    )?;
    let output = resource.get_cond(
        &backend,
        &fixture_clip_output_values(
            &backend,
            &context,
            &attention.input_shape,
            &attention.input_bits,
        )?,
        &context,
    )?;
    assert_eq!(output.descriptor().shape(), attention.output_shape);
    assert_eq!(
        tensor_to_f32(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        attention.output_bits
    );

    for mutation in oracle.mutations.values() {
        let profile = if mutation.profile == "style" {
            &oracle.style
        } else {
            &oracle.redux
        };
        let mut fixture = profile.dtypes["float32"].clone();
        let entry = fixture
            .state
            .iter_mut()
            .find(|entry| entry.key == mutation.key)
            .ok_or("mutation key is missing")?;
        let value = entry
            .storage_bits
            .get_mut(mutation.index)
            .ok_or("mutation index is missing")?;
        *value = (f32::from_bits(*value) + f32::from_bits(mutation.delta_bits)).to_bits();
        assert_eq!(
            mutation.source_identity_sha256,
            fixture_state_identity(&fixture.state, "float32", false)?
        );
        let state = upload_style_state(&backend, &context, &fixture, "float32")?;
        let resource = NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint(&mutation.profile, state, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        )?;
        let output = resource.get_cond(
            &backend,
            &fixture_clip_output(&backend, &context, profile)?,
            &context,
        )?;
        let actual = tensor_to_f32(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, mutation.output_bits);
        assert_ne!(actual, profile.dtypes["float32"].output_bits);
    }

    let signed_zero = tensor_from_f32(
        &backend,
        &[1],
        &[f32::from_bits(oracle.discriminators.signed_zero_input_bits)],
        &context,
    )?;
    let zero = full_like_with_context_exact_native(
        &backend,
        &signed_zero,
        sim_tensor::Scalar::Float(0.0),
        Some(DType::F32),
        &context,
    )?;
    let added = real_add_with_context_exact_native(&backend, &signed_zero, &zero, &context)?;
    assert_eq!(
        tensor_to_f32(&backend, &added, &context)?[0].to_bits(),
        oracle.discriminators.signed_zero_after_add_bits
    );
    let quick = &oracle.discriminators.quick_gelu;
    assert_eq!(1.702_f32.to_bits(), quick.coefficient_bits);
    let quick_input = f32::from_bits(quick.input_bits);
    let quick_scaled = 1.702_f32 * quick_input;
    assert_eq!(quick_scaled.to_bits(), quick.scaled_bits);
    let quick_tensor = tensor_from_f32(&backend, &[1], &[quick_scaled], &context)?;
    let quick_sigmoid = sigmoid_with_context_exact_native(&backend, &quick_tensor, &context)?;
    let quick_sigmoid = tensor_to_f32(&backend, &quick_sigmoid, &context)?[0];
    assert_eq!(quick_sigmoid.to_bits(), quick.sigmoid_bits);
    assert_eq!((quick_input * quick_sigmoid).to_bits(), quick.output_bits);
    let implementation = include_str!("../../sim_model/src/conditioning_resources.rs");
    assert!(implementation.contains("scaled.push(QUICK_GELU_SCALE * value);"));
    assert!(implementation.contains("output.push(*value * sigmoid);"));

    let style_fixture = &oracle.style.dtypes["float32"];
    let style_state = upload_style_state(&backend, &context, style_fixture, "float32")?;
    let mut no_marker = upload_style_state(
        &backend,
        &context,
        &oracle.redux.dtypes["float32"],
        "float32",
    )?;
    let marker = no_marker
        .iter_mut()
        .find(|(key, _)| key == "redux_down.weight")
        .ok_or("Redux marker is missing")?;
    marker.0 = "not_a_style_model_marker".to_owned();
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("redux", no_marker, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::UnsupportedArchitecture)
    ));
    let mut ambiguous = style_state.clone();
    ambiguous[2].0 = "redux_down.weight".to_owned();
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", ambiguous, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::UnexpectedState(_))
    ));
    let mut aliased = style_state.clone();
    let weight = aliased
        .iter()
        .position(|(key, _)| key == "ln_post.weight")
        .ok_or("weight key is missing")?;
    let bias = aliased
        .iter()
        .position(|(key, _)| key == "ln_post.bias")
        .ok_or("bias key is missing")?;
    aliased[bias].1 = aliased[weight].1.clone();
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", aliased, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::InvalidCheckpoint(error)) if error.contains("aliases")
    ));
    let mut malformed_shape = style_state.clone();
    let malformed_descriptor =
        TensorDescriptor::contiguous(vec![1, 4, 4], DType::F32, DeviceId::CPU, context.stream)?;
    let (malformed_tensor, event) = backend.upload_bytes(
        malformed_descriptor,
        style_state[0].1.contiguous_bytes()?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    malformed_shape[0].1 = malformed_tensor;
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", malformed_shape, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::StateShape { .. })
    ));
    let mut malformed_dtype = style_state.clone();
    let bool_descriptor =
        TensorDescriptor::contiguous(vec![1, 2, 8], DType::Bool, DeviceId::CPU, context.stream)?;
    let (bool_tensor, event) = backend.upload_bytes(bool_descriptor, &[0; 16], &context)?;
    backend.wait_event(event, &context)?;
    malformed_dtype[0].1 = bool_tensor;
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", malformed_dtype, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::StateShape { .. })
    ));
    let noncanonical_artifact = NativeStyleModelCheckpoint {
        artifact_sha256: "A".repeat(64),
        ordered_state: style_state.clone(),
        memory_budget_bytes: STYLE_MODEL_FIXTURE_MEMORY,
    };
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            noncanonical_artifact,
            &context,
        ),
        Err(NativeStyleModelError::InvalidCheckpoint(error)) if error.contains("lowercase")
    ));
    let out_of_memory = NativeStyleModelResource::from_reduced_fixture(
        &backend,
        style_checkpoint("style", style_state.clone(), 1),
        &context,
    );
    let raw_required = match out_of_memory {
        Err(NativeStyleModelError::OutOfMemory {
            required,
            budget: 1,
        }) => required,
        result => return Err(format!("expected raw checkpoint OOM, got {result:?}").into()),
    };
    assert!(raw_required > 1);
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", style_state.clone(), raw_required - 1),
            &context,
        ),
        Err(NativeStyleModelError::OutOfMemory { required, budget })
            if required == raw_required && budget == raw_required - 1
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, cancelled_workspace.clone(), &cancelled);
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", style_state.clone(), STYLE_MODEL_FIXTURE_MEMORY),
            &cancelled_context,
        ),
        Err(NativeStyleModelError::Cancelled)
    ));
    assert_eq!(cancelled_workspace.in_use_bytes(), 0);

    let resource = NativeStyleModelResource::from_reduced_fixture(
        &backend,
        style_checkpoint("style", style_state.clone(), STYLE_MODEL_FIXTURE_MEMORY),
        &context,
    )?;
    let valid_clip = fixture_clip_output(&backend, &context, &oracle.style)?;
    assert!(matches!(
        resource.reconstruct_checkpoint(&cancelled),
        Err(NativeStyleModelError::Cancelled)
    ));
    let cancelled_invocation_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let cancelled_invocation_context = backend.execution_context(
        StreamId::DEFAULT,
        cancelled_invocation_workspace.clone(),
        &cancelled,
    );
    assert!(matches!(
        resource.get_cond(&backend, &valid_clip, &cancelled_invocation_context),
        Err(NativeStyleModelError::Cancelled)
    ));
    assert_eq!(cancelled_invocation_workspace.in_use_bytes(), 0);

    let mut f16_bytes = Vec::new();
    for value in &oracle.style.input_bits {
        f16_bytes.extend_from_slice(&DType::F16.encode_scalar(
            sim_tensor::Scalar::Float(f64::from(f32::from_bits(*value))),
            "task393-style-input-dtype-negative",
            DeviceId::CPU,
        )?);
    }
    let f16_descriptor = TensorDescriptor::contiguous(
        oracle.style.input_shape.clone(),
        DType::F16,
        DeviceId::CPU,
        context.stream,
    )?;
    let (f16_hidden, event) = backend.upload_bytes(f16_descriptor, &f16_bytes, &context)?;
    backend.wait_event(event, &context)?;
    let f16_embeds_descriptor = TensorDescriptor::contiguous(
        vec![oracle.style.input_shape[0], oracle.style.input_shape[2]],
        DType::F16,
        DeviceId::CPU,
        context.stream,
    )?;
    let f16_embeds_bytes = vec![
        0_u8;
        usize::try_from(
            oracle.style.input_shape[0] * oracle.style.input_shape[2] * DType::F16.byte_width(),
        )?
    ];
    let (f16_embeds, event) =
        backend.upload_bytes(f16_embeds_descriptor, &f16_embeds_bytes, &context)?;
    backend.wait_event(event, &context)?;
    let f16_clip = ClipVisionOutput::checked(
        f16_hidden,
        None,
        f16_embeds,
        None,
        valid_clip.image_sizes().to_vec(),
    )?;
    assert!(matches!(
        resource.get_cond(&backend, &f16_clip, &context),
        Err(NativeStyleModelError::InvalidInput(error)) if error.contains("contiguous F32 CPU")
    ));

    let mut admitted_budget = raw_required;
    let constrained_resource = loop {
        match NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", style_state.clone(), admitted_budget),
            &context,
        ) {
            Ok(resource) => break resource,
            Err(NativeStyleModelError::OutOfMemory { required, budget })
                if budget == admitted_budget && required > admitted_budget =>
            {
                admitted_budget = required;
            }
            result => {
                return Err(
                    format!("expected bounded construction admission, got {result:?}").into(),
                );
            }
        }
    };
    let oom_shape = [1, 256, 8];
    let oom_input_bits = oracle
        .style
        .input_bits
        .iter()
        .copied()
        .cycle()
        .take(usize::try_from(oom_shape.iter().product::<u64>())?)
        .collect::<Vec<_>>();
    let oom_clip = fixture_clip_output_values(&backend, &context, &oom_shape, &oom_input_bits)?;
    assert!(matches!(
        constrained_resource.get_cond(&backend, &oom_clip, &context),
        Err(NativeStyleModelError::OutOfMemory { required, budget })
            if budget == admitted_budget && required > budget
    ));

    let mut stale_clip = valid_clip;
    let mut changed = oracle
        .style
        .input_bits
        .iter()
        .map(|value| f32::from_bits(*value))
        .collect::<Vec<_>>();
    changed[0] += 0.25;
    stale_clip.last_hidden_state =
        tensor_from_f32(&backend, &oracle.style.input_shape, &changed, &context)?;
    assert!(matches!(
        resource.get_cond(&backend, &stale_clip, &context),
        Err(NativeStyleModelError::ClipVision(_))
    ));
    let foreign_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let foreign_context =
        backend.execution_context(StreamId::new(7), foreign_workspace.clone(), &cancellation);
    let foreign_clip = fixture_clip_output(&backend, &foreign_context, &oracle.style)?;
    assert!(matches!(
        resource.get_cond(&backend, &foreign_clip, &context),
        Err(NativeStyleModelError::InvalidInput(_))
    ));
    let foreign_state = upload_style_state(
        &backend,
        &foreign_context,
        &oracle.style.dtypes["float32"],
        "float32",
    )?;
    assert!(matches!(
        NativeStyleModelResource::from_reduced_fixture(
            &backend,
            style_checkpoint("style", foreign_state, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeStyleModelError::InvalidCheckpoint(error)) if error.contains("construction stream")
    ));
    assert_eq!(foreign_workspace.in_use_bytes(), 0);
    assert_eq!(workspace.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gligen_resource() -> Result<(), Box<dyn Error>> {
    let oracle = style_model_oracle()?;
    let fixture = &oracle.gligen;
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
    let positions = gligen_positions(&backend, &context, fixture)?;

    for dtype_name in ["float32", "float16", "bfloat16"] {
        let expected = &fixture.dtypes[dtype_name];
        let original = upload_gligen_state(&backend, &context, &fixture.state, dtype_name)?;
        let resource = Arc::new(NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(original.clone(), GLIGEN_FIXTURE_MEMORY),
            &context,
        )?);
        assert!(!resource.is_source_exact_profile());
        assert_eq!(resource.source_dtype(), fixture_dtype(dtype_name)?);
        assert_eq!(resource.key_dimension(), fixture.key_dimension);
        assert_eq!(resource.fuser_locations().len(), fixture.fusers.len());
        for (actual, expected_location) in resource.fuser_locations().iter().zip(&fixture.fusers) {
            assert_eq!(actual.namespace(), expected_location.namespace);
            assert_eq!(actual.block_index(), expected_location.block_index);
            assert_eq!(
                actual.transformer_index(),
                expected_location.transformer_index
            );
            assert_eq!(actual.query_dimension(), fixture.query_dimension);
            assert_eq!(
                actual.region(),
                match expected_location.region.as_str() {
                    "input_blocks" => NativeGligenRegion::InputBlock,
                    "middle_block" => NativeGligenRegion::MiddleBlock,
                    "output_blocks" => NativeGligenRegion::OutputBlock,
                    value => return Err(format!("unknown GLIGEN fixture region {value}").into()),
                }
            );
        }
        resource.validate(&cancellation)?;
        let reconstructed = resource.reconstruct_checkpoint(&cancellation)?;
        assert_eq!(reconstructed.ordered_state.len(), original.len());
        for ((actual_key, actual), (expected_key, expected_tensor)) in
            reconstructed.ordered_state.iter().zip(&original)
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(actual.descriptor(), expected_tensor.descriptor());
            assert_eq!(actual.storage_id(), expected_tensor.storage_id());
            assert_eq!(
                actual.contiguous_bytes()?,
                expected_tensor.contiguous_bytes()?
            );
        }
        let prepared =
            resource.prepare_positions(&backend, fixture.latent_shape, &positions, &context)?;
        assert_eq!(
            prepared.objects().descriptor().shape(),
            expected.prepared_shape
        );
        let prepared_bits = tensor_to_f32(&backend, prepared.objects(), &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            prepared_bits, expected.prepared_bits,
            "GLIGEN {dtype_name} PositionNet"
        );
        let visual = tensor_from_f32(
            &backend,
            &expected.visual_shape,
            &expected
                .visual_bits
                .iter()
                .map(|value| f32::from_bits(*value))
                .collect::<Vec<_>>(),
            &context,
        )?;
        let visual_before = visual.contiguous_bytes()?.to_vec();
        let output = resource.apply_fuser(&backend, 0, &visual, &prepared, &context)?;
        let output_bits = tensor_to_f32(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            output_bits, expected.output_bits,
            "GLIGEN {dtype_name} fuser"
        );
        assert_eq!(visual.contiguous_bytes()?, visual_before);
        assert_ne!(output.storage_id(), visual.storage_id());

        let payload = NativeModelPayload::gligen_test_fixture(resource.clone(), &cancellation)?;
        assert!(Arc::ptr_eq(
            payload
                .gligen_resource()
                .ok_or("GLIGEN payload accessor is missing")?,
            &resource,
        ));
        assert_gligen_payload_cross_role_denial(&payload);
        payload.validate()?;
        assert!(NativeModelPayload::gligen(resource, &cancellation).is_err());
    }

    let primary_state = upload_gligen_state(&backend, &context, &fixture.state, "float32")?;
    let primary = NativeGligenResource::from_reduced_fixture(
        &backend,
        gligen_checkpoint(primary_state.clone(), GLIGEN_FIXTURE_MEMORY),
        &context,
    )?;
    let empty = primary.prepare_positions(&backend, fixture.latent_shape, &[], &context)?;
    assert_eq!(empty.objects().descriptor().shape(), [1, 30, 4]);
    let thirty = (0..30).map(|_| positions[0].clone()).collect::<Vec<_>>();
    primary.prepare_positions(&backend, fixture.latent_shape, &thirty, &context)?;
    let thirty_one = (0..31).map(|_| positions[0].clone()).collect::<Vec<_>>();
    assert!(matches!(
        primary.prepare_positions(&backend, fixture.latent_shape, &thirty_one, &context),
        Err(NativeGligenError::InvalidInput(message)) if message.contains("at most 30")
    ));

    let alternate_state = gligen_fixture_schema(768, 8, &["middle_block.3.transformer_blocks.0"])?;
    let alternate = NativeGligenResource::from_reduced_fixture(
        &backend,
        gligen_checkpoint(
            upload_gligen_state(&backend, &context, &alternate_state, "float32")?,
            GLIGEN_FIXTURE_MEMORY,
        ),
        &context,
    )?;
    let alternate_location = &alternate.fuser_locations()[0];
    assert_eq!(alternate_location.heads(), fixture.head_rule_cases[1].heads);
    assert_eq!(
        alternate_location.head_dimension(),
        fixture.head_rule_cases[1].head_dimension
    );

    let prepared =
        primary.prepare_positions(&backend, fixture.latent_shape, &positions, &context)?;
    let visual_fixture = &fixture.dtypes["float32"];
    let visual = tensor_from_f32(
        &backend,
        &visual_fixture.visual_shape,
        &visual_fixture
            .visual_bits
            .iter()
            .map(|value| f32::from_bits(*value))
            .collect::<Vec<_>>(),
        &context,
    )?;
    let alternate_visual = tensor_from_f32(&backend, &[1, 2, 8], &[0.0; 16], &context)?;
    assert!(matches!(
        alternate.apply_fuser(&backend, 0, &alternate_visual, &prepared, &context),
        Err(NativeGligenError::InvalidInput(message)) if message.contains("do not belong")
    ));
    assert!(matches!(
        primary.apply_fuser(&backend, 2, &visual, &prepared, &context),
        Err(NativeGligenError::InvalidInput(message)) if message.contains("out of range")
    ));
    let batch_two_visual = tensor_from_f32(&backend, &[2, 2, 64], &[0.0; 256], &context)?;
    assert!(matches!(
        primary.apply_fuser(&backend, 0, &batch_two_visual, &prepared, &context),
        Err(NativeGligenError::InvalidInput(message)) if message.contains("do not belong")
    ));
    let f16_descriptor = TensorDescriptor::contiguous(
        visual_fixture.visual_shape.clone(),
        DType::F16,
        DeviceId::CPU,
        context.stream,
    )?;
    let mut f16_bytes = Vec::new();
    for value in &visual_fixture.visual_bits {
        f16_bytes.extend_from_slice(&DType::F16.encode_scalar(
            sim_tensor::Scalar::Float(f64::from(f32::from_bits(*value))),
            "task396.gligen-visual-f16",
            DeviceId::CPU,
        )?);
    }
    let (f16_visual, f16_event) = backend.upload_bytes(f16_descriptor, &f16_bytes, &context)?;
    backend.wait_event(f16_event, &context)?;
    assert!(matches!(
        primary.apply_fuser(&backend, 0, &f16_visual, &prepared, &context),
        Err(NativeGligenError::InvalidInput(message)) if message.contains("CPU F32")
    ));

    let mut partial = primary_state.clone();
    partial.pop();
    assert!(matches!(
        NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(partial, GLIGEN_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeGligenError::UnexpectedState(_))
    ));
    let mut out_of_range = primary_state.clone();
    for (key, _) in &mut out_of_range {
        if key.starts_with("output_blocks.7.") {
            *key = key.replacen("output_blocks.7.", "output_blocks.20.", 1);
        }
    }
    assert!(matches!(
        NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(out_of_range, GLIGEN_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeGligenError::UnexpectedState(message)) if message.contains("outside 0..19")
    ));
    let mut collision = primary_state.clone();
    for (key, _) in &mut collision {
        if key.starts_with("output_blocks.7.") {
            *key = key.replacen(
                "output_blocks.7.transformer_blocks.0",
                "input_blocks.2.transformer_blocks.1",
                1,
            );
        }
    }
    assert!(matches!(
        NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(collision, GLIGEN_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeGligenError::UnexpectedState(message)) if message.contains("multiple fuser namespaces")
    ));
    let mut aliases = primary_state.clone();
    let source = aliases
        .iter()
        .find(|(key, _)| key.ends_with("norm1.weight"))
        .ok_or("GLIGEN alias source is missing")?
        .1
        .clone();
    let target = aliases
        .iter_mut()
        .find(|(key, _)| key.ends_with("norm1.bias"))
        .ok_or("GLIGEN alias target is missing")?;
    assert_eq!(source.descriptor(), target.1.descriptor());
    target.1 = source;
    assert!(matches!(
        NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(aliases, GLIGEN_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativeGligenError::InvalidCheckpoint(message)) if message.contains("aliases")
    ));

    let raw_out_of_memory = NativeGligenResource::from_reduced_fixture(
        &backend,
        gligen_checkpoint(primary_state.clone(), 1),
        &context,
    );
    let required = match raw_out_of_memory {
        Err(NativeGligenError::OutOfMemory {
            required,
            budget: 1,
        }) => required,
        result => return Err(format!("expected GLIGEN construction OOM, got {result:?}").into()),
    };
    assert!(matches!(
        NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(primary_state.clone(), required - 1),
            &context,
        ),
        Err(NativeGligenError::OutOfMemory { required: actual, budget })
            if actual == required && budget == required - 1
    ));
    let mut admitted_budget = required;
    loop {
        match NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(primary_state.clone(), admitted_budget),
            &context,
        ) {
            Ok(_) => break,
            Err(NativeGligenError::OutOfMemory {
                required: next_required,
                budget,
            }) if budget == admitted_budget && next_required > admitted_budget => {
                admitted_budget = next_required;
            }
            result => {
                return Err(format!(
                    "GLIGEN monotonic construction admission failed at {admitted_budget}: {result:?}"
                )
                .into());
            }
        }
    }
    let (constrained, constrained_prepared) = loop {
        let candidate = NativeGligenResource::from_reduced_fixture(
            &backend,
            gligen_checkpoint(primary_state.clone(), admitted_budget),
            &context,
        )?;
        match candidate.prepare_positions(&backend, fixture.latent_shape, &positions, &context) {
            Ok(prepared) => break (candidate, prepared),
            Err(NativeGligenError::OutOfMemory {
                required: next_required,
                budget,
            }) if budget == admitted_budget && next_required > admitted_budget => {
                admitted_budget = next_required;
            }
            result => {
                return Err(format!(
                    "GLIGEN position admission failed at {admitted_budget}: {result:?}"
                )
                .into());
            }
        }
    };
    let large_visual = tensor_from_f32(&backend, &[1, 256, 64], &[0.0; 256 * 64], &context)?;
    match constrained.apply_fuser(&backend, 0, &large_visual, &constrained_prepared, &context) {
        Err(NativeGligenError::OutOfMemory {
            required: apply_required,
            budget,
        }) if budget == admitted_budget && apply_required > admitted_budget => {}
        result => {
            return Err(format!(
                "expected GLIGEN invocation OOM above construction/position budget {admitted_budget}, got {result:?}"
            )
            .into());
        }
    }
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancelled);
    assert!(matches!(
        alternate.prepare_positions(&backend, fixture.latent_shape, &[], &cancelled_context),
        Err(NativeGligenError::Cancelled)
    ));
    assert_eq!(workspace.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn photomaker_resource() -> Result<(), Box<dyn Error>> {
    let oracle = style_model_oracle()?;
    let fixture = &oracle.photomaker;
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
    let (image, prompt, mask) = photomaker_inputs(&backend, &context, fixture)?;
    let image_before = image.contiguous_bytes()?.to_vec();
    let prompt_before = prompt.contiguous_bytes()?.to_vec();
    let mask_before = mask.contiguous_bytes()?.to_vec();

    for (dtype_index, dtype_name) in ["float32", "float16", "bfloat16"].iter().enumerate() {
        let expected = &fixture.dtypes[*dtype_name];
        assert_eq!(
            expected.source_identity_sha256,
            fixture_state_identity(&expected.state, dtype_name, false)?
        );
        assert_eq!(
            expected.projected_identity_sha256,
            fixture_state_identity(&expected.state, dtype_name, true)?
        );
        assert_eq!(expected.pooled_bits.len(), 8);
        assert_eq!(
            expected.identity_bits.len(),
            expected.first_projection_bits.len() + expected.second_projection_bits.len()
        );
        let original = upload_photomaker_state(&backend, &context, expected, dtype_name)?;
        let nested = dtype_index != 0;
        let resource = Arc::new(
            NativePhotoMakerResource::from_reduced_fixture(
                &backend,
                photomaker_checkpoint(original.clone(), nested, STYLE_MODEL_FIXTURE_MEMORY),
                &context,
            )
            .map_err(|error| {
                format!(
                    "PhotoMaker {dtype_name} construction failed with budget {STYLE_MODEL_FIXTURE_MEMORY}: {error}"
                )
            })?,
        );
        assert!(!resource.is_source_exact_profile());
        assert_eq!(resource.source_dtype(), fixture_dtype(dtype_name)?);
        resource.validate(&cancellation)?;
        let reconstructed = resource
            .reconstruct_checkpoint(&cancellation)
            .map_err(|error| {
                format!(
                    "PhotoMaker {dtype_name} reconstruction failed with budget {STYLE_MODEL_FIXTURE_MEMORY}: {error}"
                )
            })?;
        assert_photomaker_reconstruction(&reconstructed, &original, nested)?;
        let output = resource
            .fuse_conditioning(&backend, &image, &prompt, &mask, &context)
            .map_err(|error| {
                format!(
                    "PhotoMaker {dtype_name} invocation failed with budget {STYLE_MODEL_FIXTURE_MEMORY}: {error}"
                )
            })?;
        assert_eq!(output.descriptor().shape(), expected.output_shape);
        let actual_bits = tensor_to_f32(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_photomaker_output_bound(
            &actual_bits,
            &expected.output_bits,
            &expected.canonical_output_bits,
            expected.output_ulp_bound,
            expected.output_ulp_rejected_distance,
        );
        for row in [0_usize, 2] {
            let start = row * 8;
            assert_eq!(
                &actual_bits[start..start + 8],
                &fixture.prompt_bits[start..start + 8]
            );
        }
        let payload = NativeModelPayload::photomaker_test_fixture(resource.clone(), &cancellation)?;
        assert_eq!(
            payload.identity().role(),
            sim_model::NativeModelResourceRole::Photomaker
        );
        assert!(Arc::ptr_eq(
            payload
                .photomaker_resource()
                .ok_or("PHOTOMAKER payload accessor is missing")?,
            &resource
        ));
        assert!(payload.style_model_resource().is_none());
        assert_photomaker_payload_cross_role_denial(&payload);
        payload.validate()?;
        assert!(NativeModelPayload::photomaker(resource, &cancellation).is_err());
    }
    assert_eq!(image.contiguous_bytes()?, image_before);
    assert_eq!(prompt.contiguous_bytes()?, prompt_before);
    assert_eq!(mask.contiguous_bytes()?, mask_before);

    let baseline = &fixture.dtypes["float32"];
    for (mutation_name, mutation) in &oracle.photomaker_mutations {
        let mut mutated = baseline.clone();
        let entry = mutated
            .state
            .iter_mut()
            .find(|entry| entry.key == mutation.key)
            .ok_or("PhotoMaker mutation key is missing")?;
        let value = entry
            .storage_bits
            .get_mut(mutation.index)
            .ok_or("PhotoMaker mutation index is missing")?;
        *value = (f32::from_bits(*value) + f32::from_bits(mutation.delta_bits)).to_bits();
        assert_eq!(
            mutation.source_identity_sha256,
            fixture_state_identity(&mutated.state, "float32", false)?
        );
        let resource = NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(
                upload_photomaker_state(&backend, &context, &mutated, "float32")?,
                false,
                STYLE_MODEL_FIXTURE_MEMORY,
            ),
            &context,
        )
        .map_err(|error| {
            format!(
                "PhotoMaker mutation {} construction failed with budget {STYLE_MODEL_FIXTURE_MEMORY}: {error}",
                mutation_name
            )
        })?;
        let output = resource
            .fuse_conditioning(&backend, &image, &prompt, &mask, &context)
            .map_err(|error| {
                format!(
                    "PhotoMaker mutation {} invocation failed with budget {STYLE_MODEL_FIXTURE_MEMORY}: {error}",
                    mutation_name
                )
            })?;
        let actual = tensor_to_f32(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_photomaker_output_bound(
            &actual,
            &mutation.output_bits,
            &mutation.canonical_output_bits,
            mutation.output_ulp_bound,
            mutation.output_ulp_rejected_distance,
        );
        assert_ne!(actual, baseline.canonical_output_bits);
    }

    let baseline_state = upload_photomaker_state(&backend, &context, baseline, "float32")?;
    let raw_out_of_memory = NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_checkpoint(baseline_state.clone(), false, 1),
        &context,
    );
    let raw_required = match raw_out_of_memory {
        Err(NativePhotoMakerError::OutOfMemory {
            required,
            budget: 1,
        }) => required,
        result => {
            return Err(format!("expected PhotoMaker construction OOM, got {result:?}").into());
        }
    };
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(
                baseline_state.clone(),
                false,
                raw_required - 1,
            ),
            &context,
        ),
        Err(NativePhotoMakerError::OutOfMemory { required, budget })
            if required == raw_required && budget == raw_required - 1
    ));
    let mut admitted_budget = raw_required;
    let constrained = loop {
        match NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(baseline_state.clone(), false, admitted_budget),
            &context,
        ) {
            Ok(resource) => break resource,
            Err(NativePhotoMakerError::OutOfMemory { required, budget })
                if budget == admitted_budget && required > admitted_budget =>
            {
                admitted_budget = required;
            }
            result => {
                return Err(format!(
                    "PhotoMaker bounded construction admission failed with budget {admitted_budget}: {result:?}"
                )
                .into());
            }
        }
    };
    assert!(matches!(
        constrained.fuse_conditioning(&backend, &image, &prompt, &mask, &context),
        Err(NativePhotoMakerError::OutOfMemory { required, budget })
            if required > budget && budget == admitted_budget
    ));

    let mut misordered = baseline_state.clone();
    misordered.swap(0, 1);
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(misordered, false, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativePhotoMakerError::UnexpectedState(_))
    ));
    let mut aliased = baseline_state.clone();
    let alias_source = aliased
        .iter()
        .find(|(key, _)| key == "vision_model.pre_layrnorm.weight")
        .map(|(_, tensor)| tensor.clone())
        .ok_or("PhotoMaker alias source is missing")?;
    let alias_target = aliased
        .iter_mut()
        .find(|(key, _)| key == "vision_model.pre_layrnorm.bias")
        .ok_or("PhotoMaker alias target is missing")?;
    assert_eq!(alias_source.descriptor(), alias_target.1.descriptor());
    alias_target.1 = alias_source;
    let alias_result = NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_checkpoint(aliased, false, STYLE_MODEL_FIXTURE_MEMORY),
        &context,
    );
    match alias_result {
        Err(NativePhotoMakerError::InvalidCheckpoint(error)) if error.contains("aliases") => {}
        result => {
            return Err(format!(
                "expected PhotoMaker source-storage alias rejection after matching-shape schema admission, got {result:?}"
            )
            .into());
        }
    }
    let wrong_wrapper = NativePhotoMakerCheckpoint {
        artifact_sha256: photomaker_artifact_sha256(),
        ordered_entries: vec![NativePhotoMakerCheckpointEntry::Mapping {
            key: "model".to_owned(),
            ordered_state: baseline_state.clone(),
        }],
        memory_budget_bytes: STYLE_MODEL_FIXTURE_MEMORY,
    };
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(&backend, wrong_wrapper, &context),
        Err(NativePhotoMakerError::InvalidCheckpoint(error)) if error.contains("id_encoder")
    ));
    let mixed_wrapper = NativePhotoMakerCheckpoint {
        artifact_sha256: photomaker_artifact_sha256(),
        ordered_entries: vec![
            NativePhotoMakerCheckpointEntry::Tensor {
                key: baseline_state[0].0.clone(),
                tensor: baseline_state[0].1.clone(),
            },
            NativePhotoMakerCheckpointEntry::Mapping {
                key: "id_encoder".to_owned(),
                ordered_state: baseline_state.clone(),
            },
        ],
        memory_budget_bytes: STYLE_MODEL_FIXTURE_MEMORY,
    };
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(&backend, mixed_wrapper, &context),
        Err(NativePhotoMakerError::InvalidCheckpoint(_))
    ));
    let mut inner_extra = baseline_state.clone();
    inner_extra.push(baseline_state[0].clone());
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(inner_extra, true, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativePhotoMakerError::UnexpectedState(_))
    ));
    let uppercase = NativePhotoMakerCheckpoint {
        artifact_sha256: "A".repeat(64),
        ordered_entries: baseline_state
            .iter()
            .cloned()
            .map(|(key, tensor)| NativePhotoMakerCheckpointEntry::Tensor { key, tensor })
            .collect(),
        memory_budget_bytes: STYLE_MODEL_FIXTURE_MEMORY,
    };
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(&backend, uppercase, &context),
        Err(NativePhotoMakerError::InvalidCheckpoint(error)) if error.contains("lowercase")
    ));

    let mut malformed_shape = baseline_state.clone();
    malformed_shape[0].1 = tensor_from_f32(&backend, &[1, 4], &[0.0; 4], &context)?;
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(malformed_shape, false, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativePhotoMakerError::StateShape { .. })
    ));
    let mut malformed_dtype = baseline_state.clone();
    let bool_descriptor =
        TensorDescriptor::contiguous(vec![4], DType::Bool, DeviceId::CPU, context.stream)?;
    let (bool_state, event) = backend.upload_bytes(bool_descriptor, &[0; 4], &context)?;
    backend.wait_event(event, &context)?;
    malformed_dtype[0].1 = bool_state;
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(malformed_dtype, false, STYLE_MODEL_FIXTURE_MEMORY),
            &context,
        ),
        Err(NativePhotoMakerError::StateShape { .. })
    ));

    let resource = NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_checkpoint(baseline_state.clone(), false, STYLE_MODEL_FIXTURE_MEMORY),
        &context,
    )?;
    let nested_resource = NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_checkpoint(baseline_state.clone(), true, STYLE_MODEL_FIXTURE_MEMORY),
        &context,
    )?;
    assert_ne!(
        resource.semantic_digest_sha256(),
        nested_resource.semantic_digest_sha256()
    );
    let short_mask_descriptor =
        TensorDescriptor::contiguous(vec![1, 4], DType::Bool, DeviceId::CPU, context.stream)?;
    let (short_mask, event) =
        backend.upload_bytes(short_mask_descriptor, &[0, 1, 0, 0], &context)?;
    backend.wait_event(event, &context)?;
    assert!(matches!(
        resource.fuse_conditioning(&backend, &image, &prompt, &short_mask, &context),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("true entries")
    ));
    let batch_two_image = tensor_from_f32(&backend, &[2, 1, 3, 4, 4], &[0.0; 96], &context)?;
    assert!(matches!(
        resource.fuse_conditioning(&backend, &batch_two_image, &prompt, &mask, &context),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("[1, N")
    ));
    let narrow_prompt = tensor_from_f32(&backend, &[1, 4, 7], &[0.0; 28], &context)?;
    assert!(matches!(
        resource.fuse_conditioning(&backend, &image, &narrow_prompt, &mask, &context),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("projection_width")
    ));
    let float_mask = tensor_from_f32(&backend, &[1, 4], &[0.0, 1.0, 0.0, 1.0], &context)?;
    assert!(matches!(
        resource.fuse_conditioning(&backend, &image, &prompt, &float_mask, &context),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("class token mask")
    ));
    let nonfinite_image = tensor_from_f32(
        &backend,
        &fixture.image_shape,
        &vec![f32::NAN; fixture.image_bits.len()],
        &context,
    )?;
    assert!(matches!(
        resource.fuse_conditioning(&backend, &nonfinite_image, &prompt, &mask, &context),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("non-finite")
    ));
    let foreign_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let foreign_context =
        backend.execution_context(StreamId::new(9), foreign_workspace.clone(), &cancellation);
    let (foreign_image, foreign_prompt, foreign_mask) =
        photomaker_inputs(&backend, &foreign_context, fixture)?;
    assert!(matches!(
        resource.fuse_conditioning(
            &backend,
            &foreign_image,
            &foreign_prompt,
            &foreign_mask,
            &context,
        ),
        Err(NativePhotoMakerError::InvalidInput(error)) if error.contains("execution stream")
    ));
    assert_eq!(foreign_workspace.in_use_bytes(), 0);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, cancelled_workspace.clone(), &cancelled);
    assert!(matches!(
        NativePhotoMakerResource::from_reduced_fixture(
            &backend,
            photomaker_checkpoint(baseline_state, false, STYLE_MODEL_FIXTURE_MEMORY),
            &cancelled_context,
        ),
        Err(NativePhotoMakerError::Cancelled)
    ));
    assert!(matches!(
        resource.reconstruct_checkpoint(&cancelled),
        Err(NativePhotoMakerError::Cancelled)
    ));
    assert!(matches!(
        resource.fuse_conditioning(&backend, &image, &prompt, &mask, &cancelled_context),
        Err(NativePhotoMakerError::Cancelled)
    ));
    assert_eq!(cancelled_workspace.in_use_bytes(), 0);
    assert_eq!(workspace.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn conditioning_auxiliary_resource_roles() -> Result<(), Box<dyn Error>> {
    let oracle = style_model_oracle()?;
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);

    let manifest: serde_json::Value = serde_json::from_slice(STYLE_MODEL_MANIFEST)?;
    assert_eq!(
        manifest["profiles"],
        json!(["style", "redux", "photomaker", "gligen"])
    );
    for (bytes, expected) in [
        (
            STYLE_MODEL_MANIFEST,
            "2c38283c6f54a76077cafd56dd653e4f1f9f7f943110db4c632ed7486e57c405",
        ),
        (
            STYLE_MODEL_PROVENANCE,
            "7149301049024767c50963bc25d7b00abd7e69979d532e28e788e6ac59e08c22",
        ),
        (
            STYLE_MODEL_ORACLE,
            "c4d20e79b630efea4917ba7416831596cd9756e6ce9d72e0ac51fdf064a0eb6e",
        ),
        (
            STYLE_MODEL_GENERATOR,
            "c7622869a2d8abf441dc45a8f147a9d7d73548dfcd652343405eb8f757aa249c",
        ),
        (
            STYLE_MODEL_SOURCE_GRAPH,
            "9663b57984f7fc64e8c4b5f6edf3a165894bf591c6eb7d56dea5ee1c6d6f364f",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    }

    let style_fixture = &oracle.style.dtypes["float32"];
    let style_resource = Arc::new(NativeStyleModelResource::from_reduced_fixture(
        &backend,
        style_checkpoint(
            "style",
            upload_style_state(&backend, &context, style_fixture, "float32")?,
            STYLE_MODEL_FIXTURE_MEMORY,
        ),
        &context,
    )?);
    let style_payload =
        NativeModelPayload::style_model_test_fixture(style_resource.clone(), &cancellation)?;

    let photomaker_fixture = &oracle.photomaker.dtypes["float32"];
    let photomaker_resource = Arc::new(NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_checkpoint(
            upload_photomaker_state(&backend, &context, photomaker_fixture, "float32")?,
            false,
            STYLE_MODEL_FIXTURE_MEMORY,
        ),
        &context,
    )?);
    let photomaker_payload =
        NativeModelPayload::photomaker_test_fixture(photomaker_resource.clone(), &cancellation)?;

    let gligen_resource = Arc::new(NativeGligenResource::from_reduced_fixture(
        &backend,
        gligen_checkpoint(
            upload_gligen_state(&backend, &context, &oracle.gligen.state, "float32")?,
            GLIGEN_FIXTURE_MEMORY,
        ),
        &context,
    )?);
    let gligen_payload =
        NativeModelPayload::gligen_test_fixture(gligen_resource.clone(), &cancellation)?;

    assert!(Arc::ptr_eq(
        style_payload
            .style_model_resource()
            .ok_or("STYLE_MODEL role is missing")?,
        &style_resource,
    ));
    assert!(style_payload.photomaker_resource().is_none());
    assert!(style_payload.gligen_resource().is_none());
    assert!(Arc::ptr_eq(
        photomaker_payload
            .photomaker_resource()
            .ok_or("PHOTOMAKER role is missing")?,
        &photomaker_resource,
    ));
    assert!(photomaker_payload.style_model_resource().is_none());
    assert!(photomaker_payload.gligen_resource().is_none());
    assert!(Arc::ptr_eq(
        gligen_payload
            .gligen_resource()
            .ok_or("GLIGEN role is missing")?,
        &gligen_resource,
    ));
    assert!(gligen_payload.style_model_resource().is_none());
    assert!(gligen_payload.photomaker_resource().is_none());

    assert!(matches!(
        NativeModelPayload::style_model(style_resource.clone(), &cancellation),
        Err(NativeModelPayloadError::ResourceMismatch(
            "STYLE_MODEL production source-exact profile"
        ))
    ));
    assert!(matches!(
        NativeModelPayload::photomaker(photomaker_resource.clone(), &cancellation),
        Err(NativeModelPayloadError::ResourceMismatch(
            "PHOTOMAKER production source-exact profile"
        ))
    ));
    assert!(matches!(
        NativeModelPayload::gligen(gligen_resource.clone(), &cancellation),
        Err(NativeModelPayloadError::ResourceMismatch(
            "GLIGEN production source-exact profile"
        ))
    ));

    let style_reconstructed = Arc::new(NativeStyleModelResource::from_reduced_fixture(
        &backend,
        style_payload
            .style_model_resource()
            .ok_or("STYLE_MODEL role is missing")?
            .reconstruct_checkpoint(&cancellation)?,
        &context,
    )?);
    let style_reconstructed_payload =
        NativeModelPayload::style_model_test_fixture(style_reconstructed.clone(), &cancellation)?;
    assert_eq!(
        style_reconstructed_payload.identity(),
        style_payload.identity()
    );
    assert_eq!(
        style_reconstructed_payload.resident_bytes(),
        style_payload.resident_bytes()
    );
    let style_input = fixture_clip_output(&backend, &context, &oracle.style)?;
    assert_eq!(
        style_resource
            .get_cond(&backend, &style_input, &context)?
            .contiguous_bytes()?,
        style_reconstructed
            .get_cond(&backend, &style_input, &context)?
            .contiguous_bytes()?,
    );

    let photomaker_reconstructed = Arc::new(NativePhotoMakerResource::from_reduced_fixture(
        &backend,
        photomaker_payload
            .photomaker_resource()
            .ok_or("PHOTOMAKER role is missing")?
            .reconstruct_checkpoint(&cancellation)?,
        &context,
    )?);
    let photomaker_reconstructed_payload = NativeModelPayload::photomaker_test_fixture(
        photomaker_reconstructed.clone(),
        &cancellation,
    )?;
    assert_eq!(
        photomaker_reconstructed_payload.identity(),
        photomaker_payload.identity()
    );
    assert_eq!(
        photomaker_reconstructed_payload.resident_bytes(),
        photomaker_payload.resident_bytes()
    );
    let (image, prompt, mask) = photomaker_inputs(&backend, &context, &oracle.photomaker)?;
    assert_eq!(
        photomaker_resource
            .fuse_conditioning(&backend, &image, &prompt, &mask, &context)?
            .contiguous_bytes()?,
        photomaker_reconstructed
            .fuse_conditioning(&backend, &image, &prompt, &mask, &context)?
            .contiguous_bytes()?,
    );

    let gligen_reconstructed = Arc::new(NativeGligenResource::from_reduced_fixture(
        &backend,
        gligen_payload
            .gligen_resource()
            .ok_or("GLIGEN role is missing")?
            .reconstruct_checkpoint(&cancellation)?,
        &context,
    )?);
    let gligen_reconstructed_payload =
        NativeModelPayload::gligen_test_fixture(gligen_reconstructed.clone(), &cancellation)?;
    assert_eq!(
        gligen_reconstructed_payload.identity(),
        gligen_payload.identity()
    );
    assert_eq!(
        gligen_reconstructed_payload.resident_bytes(),
        gligen_payload.resident_bytes()
    );
    let positions = gligen_positions(&backend, &context, &oracle.gligen)?;
    let prepared = gligen_resource.prepare_positions(
        &backend,
        oracle.gligen.latent_shape,
        &positions,
        &context,
    )?;
    let reconstructed_prepared = gligen_reconstructed.prepare_positions(
        &backend,
        oracle.gligen.latent_shape,
        &positions,
        &context,
    )?;
    let gligen_fixture = &oracle.gligen.dtypes["float32"];
    let visual = tensor_from_f32(
        &backend,
        &gligen_fixture.visual_shape,
        &gligen_fixture
            .visual_bits
            .iter()
            .map(|value| f32::from_bits(*value))
            .collect::<Vec<_>>(),
        &context,
    )?;
    assert_eq!(
        gligen_resource
            .apply_fuser(&backend, 0, &visual, &prepared, &context)?
            .contiguous_bytes()?,
        gligen_reconstructed
            .apply_fuser(&backend, 0, &visual, &reconstructed_prepared, &context)?
            .contiguous_bytes()?,
    );

    for payload in [style_payload, photomaker_payload, gligen_payload] {
        payload.clone().validate()?;
    }
    assert_eq!(workspace.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn native_clip_vision_context_construction() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let source = fs::read_to_string(workspace.join("crates/sim_model/src/clip_vision.rs"))?;
    let production = source
        .split_once("\n#[cfg(test)]\n")
        .map_or(source.as_str(), |(production, _)| production);
    assert_eq!(
        production.matches("pub struct NativeClipVision {").count(),
        1
    );
    assert_eq!(
        production.matches("pub fn new_with_cancellation(").count(),
        1
    );
    assert_eq!(production.matches("pub fn reconstruct(").count(), 1);
    assert_eq!(
        production.matches("pub(crate) fn forward_checked(").count(),
        1
    );
    assert!(production.contains("pub(crate) pooled_hidden: Tensor"));
    assert!(production.contains("Self::new_with_cancellation("));
    assert!(production.contains("ClipVisionCanonicalStateCursor::new"));
    assert!(production.contains("Self::new_with_cancellation_and_phase_hook("));
    for phase in [
        "CanonicalState",
        "ParameterProjection",
        "PatchEmbedding",
        "PreLayerNorm",
        "LayerNorm1",
        "AttentionQuery",
        "AttentionKey",
        "AttentionValue",
        "AttentionOutput",
        "LayerNorm2",
        "MlpInput",
        "MlpActivation",
        "MlpOutput",
        "PostLayerNorm",
        "VisualProjection",
        "LlavaProjection",
        "SemanticDigest",
        "ModuleDigest",
        "Validation",
        "Return",
    ] {
        assert!(
            production.contains(&format!("ClipVisionConstructionPhase::{phase}")),
            "missing caller-cancellation phase {phase}"
        );
    }
    assert!(!production.contains("pub struct NativeClipVisionCheckedForward"));

    let payload =
        fs::read_to_string(workspace.join("crates/sim_model/src/native_node_payload.rs"))?;
    assert!(!payload.contains("new_with_cancellation"));
    assert!(!payload.contains("forward_checked"));
    assert!(!payload.contains("pooled_hidden"));
    Ok(())
}

#[test]
fn task403_native_conditioning_metadata_preserves_cataloged_source_payloads()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1_048_576)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(16 * 1_048_576)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let tensor = tensor_from_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &context)?;
    let time_tensor = tensor_from_f32(&backend, &[1, 2], &[0.25, 0.75], &context)?;
    let audio_tensor = tensor_from_f32(&backend, &[1, 2, 2], &[0.5; 4], &context)?;
    let video_tensor = tensor_from_f32(&backend, &[1, 1, 1, 2, 2], &[0.5; 4], &context)?;
    let pixel_video = tensor_from_f32(&backend, &[1, 1, 2, 2, 3], &[0.5; 12], &context)?;
    let mask_28 = tensor_from_f32(&backend, &[1, 1, 2, 2, 28], &[1.0; 112], &context)?;
    let hidden = tensor_from_f32(&backend, &[1, 2, 3], &[0.0; 6], &context)?;
    let embeds = tensor_from_f32(&backend, &[1, 3], &[0.0; 3], &context)?;
    let vision = Arc::new(ClipVisionOutput::checked(
        hidden,
        None,
        embeds,
        None,
        vec![[3, 16, 16]],
    )?);
    let latent = Arc::new(NativeLatentBundle::single(
        tensor.clone(),
        None,
        None,
        NativeLatentMetadata::default(),
        &context,
    )?);
    let metadata = ConditioningMetadataMap::checked([
        (
            ConditioningMetadataKey::Region,
            ConditioningMetadataValue::Region(ConditioningRegion::percentage(
                vec![0.0, 0.5],
                vec![0.0, 0.25],
            )?),
        ),
        (
            ConditioningMetadataKey::Mask,
            ConditioningMetadataValue::Tensor(tensor.clone()),
        ),
        (
            ConditioningMetadataKey::MaskStrength,
            ConditioningMetadataValue::FiniteF64(0.75),
        ),
        (
            ConditioningMetadataKey::SetAreaToBounds,
            ConditioningMetadataValue::Boolean(false),
        ),
        (
            ConditioningMetadataKey::TimestepRange,
            ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(0.9, 0.1)?),
        ),
        (
            ConditioningMetadataKey::ConcatLatentImage,
            ConditioningMetadataValue::LatentBundle(latent),
        ),
        (
            ConditioningMetadataKey::ConcatMask,
            ConditioningMetadataValue::Tensor(tensor.clone()),
        ),
        (
            ConditioningMetadataKey::ConcatMaskIndex,
            ConditioningMetadataValue::Unsigned(0),
        ),
        (
            ConditioningMetadataKey::Control,
            ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                false,
            )?),
        ),
        (
            ConditioningMetadataKey::ControlApplyToUnconditioned,
            ConditioningMetadataValue::Boolean(false),
        ),
        (
            ConditioningMetadataKey::UnclipConditioning,
            ConditioningMetadataValue::ClipVisionConditioning(vec![
                ClipVisionConditioning::checked(vision.clone(), 1.0, 0.25)?,
            ]),
        ),
        (
            ConditioningMetadataKey::ClipVisionOutput,
            ConditioningMetadataValue::ClipVisionOutput(vision),
        ),
        (
            ConditioningMetadataKey::ReferenceLatents,
            ConditioningMetadataValue::TensorList(vec![tensor.clone()]),
        ),
        (
            ConditioningMetadataKey::VaceFrames,
            ConditioningMetadataValue::TensorList(vec![tensor.clone()]),
        ),
        (
            ConditioningMetadataKey::VaceMask,
            ConditioningMetadataValue::TensorList(vec![tensor.clone()]),
        ),
        (
            ConditioningMetadataKey::VaceStrength,
            ConditioningMetadataValue::FiniteF64List(vec![1.0]),
        ),
        (
            ConditioningMetadataKey::CameraConditions,
            ConditioningMetadataValue::Tensor(tensor.clone()),
        ),
        (
            ConditioningMetadataKey::TimeDimensionConcat,
            ConditioningMetadataValue::Tensor(time_tensor),
        ),
        (
            ConditioningMetadataKey::AudioEmbedding,
            ConditioningMetadataValue::Tensor(audio_tensor),
        ),
        (
            ConditioningMetadataKey::ReferenceMotion,
            ConditioningMetadataValue::Tensor(video_tensor.clone()),
        ),
        (
            ConditioningMetadataKey::ControlVideo,
            ConditioningMetadataValue::Tensor(video_tensor.clone()),
        ),
        (
            ConditioningMetadataKey::PoseVideoLatent,
            ConditioningMetadataValue::Tensor(video_tensor),
        ),
        (
            ConditioningMetadataKey::FaceVideoPixels,
            ConditioningMetadataValue::Tensor(pixel_video),
        ),
        (
            ConditioningMetadataKey::DrivingMask28Channels,
            ConditioningMetadataValue::Tensor(mask_28.clone()),
        ),
        (
            ConditioningMetadataKey::ReferenceMask28Channels,
            ConditioningMetadataValue::Tensor(mask_28),
        ),
        (
            ConditioningMetadataKey::ReferenceMaskFlag,
            ConditioningMetadataValue::Boolean(true),
        ),
        (
            ConditioningMetadataKey::SecondsStart,
            ConditioningMetadataValue::FiniteF64(0.5),
        ),
        (
            ConditioningMetadataKey::SecondsTotal,
            ConditioningMetadataValue::FiniteF64(5.0),
        ),
    ])?;
    assert_eq!(metadata.len(), 28);
    assert!(metadata.sampler_window().is_err());
    let entry = ConditioningEntry::checked(
        "entry",
        ConditioningValue::regular(tensor.clone())?,
        ConditioningEntryOptions::default(),
    )?;
    let set = ConditioningSet::checked(
        ConditioningIdentity::new(
            "task403",
            sd15_model_family_identity()?,
            sd15_latent_format_identity()?,
        )?,
        vec![entry],
        &cancellation,
    )?;
    let mut sampler_metadata = metadata.clone();
    sampler_metadata.insert(
        ConditioningMetadataKey::TimestepRange,
        ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(0.1, 0.9)?),
    )?;
    let sampler_entry = set.entries()[0].with_metadata(sampler_metadata)?;
    let consumed_entry = sampler_entry.resolve(tensor.descriptor(), &backend, &context)?;
    let consumed = NativeConditioningMetadata::checked(&consumed_entry)?;
    assert_eq!(consumed.entry_identifier(), "entry");
    assert_eq!(consumed.iter().count(), 28);
    assert!(matches!(
        consumed.get(ConditioningMetadataKey::SecondsTotal),
        Some(ConditioningMetadataValue::FiniteF64(5.0))
    ));
    let positive = set.with_entry_metadata(vec![metadata.clone()], &cancellation)?;
    let mut negative_metadata = metadata;
    negative_metadata.insert(
        ConditioningMetadataKey::Control,
        ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            false,
        )?),
    )?;
    let negative = set.with_entry_metadata(vec![negative_metadata], &cancellation)?;
    assert_eq!(positive.entries().len(), 1);
    assert_eq!(negative.entries().len(), 1);
    assert_ne!(positive.digest(), negative.digest());
    positive.validate()?;
    negative.validate()?;

    let model_digest = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let control = ControlNet::checked(
        ControlBase::checked(
            1.0,
            StrengthType::Constant,
            ControlPercentWindow::checked(0.0, 1.0)?,
            false,
            None,
        )?,
        ControlModelBinding::checked(
            sd15_model_family_identity()?,
            PatchGraphIdentity {
                schema_version: sim_model::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: model_digest.to_owned(),
                ordered_digest: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                    .to_owned(),
            },
            model_digest,
            FIXTURE_CONTROL_EXECUTOR_DIGEST,
            DType::F32,
            DeviceId::CPU,
        )?,
        ControlTensorBinding::checked(
            tensor.clone(),
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        )?,
        1,
        ResizeMode::NearestExact,
        ControlHintPreprocess::Identity,
        None,
        Vec::new(),
        false,
        Vec::new(),
    )?;
    let chain = Arc::new(ControlChain::checked(vec![ControlNode::ControlNet(
        control,
    )])?);
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(FixtureControlExecutor {
        calls: calls.clone(),
        mode: FixtureControlMode::Success,
    });
    let prebound = PreboundControlExecution::checked(chain.clone(), executor.clone())?;
    let rebind_request = ControlNetRebindRequest::checked(
        ControlTensorBinding::checked(
            tensor,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )?,
        ControlHintLayout::Nchw,
        0.5,
        ControlPercentWindow::checked(0.1, 0.9)?,
        None,
        Vec::new(),
        false,
        Some(2),
    )?;
    let rebound = prebound.rebound(&rebind_request, Some(&prebound), None, &cancellation)?;
    assert!(prebound.shares_executor_with(&rebound));
    assert_eq!(rebound.chain().nodes().len(), 2);
    let (fixture_backend, fixture_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let fixture_backend = Arc::new(fixture_backend);
    let fixture_context = fixture_backend.execution_context(
        StreamId::DEFAULT,
        fixture_authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let fixture_bundle = NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(fixture_backend, &fixture_context)?;
    let vae_rebind_request = ControlNetRebindRequest::checked(
        rebind_request.hint().clone(),
        rebind_request.hint_layout(),
        rebind_request.strength(),
        rebind_request.percent_window(),
        Some(fixture_bundle.vae().execution_digest()),
        rebind_request.extra_concat().to_vec(),
        rebind_request.concat_mask(),
        rebind_request.union_control_type(),
    )?;
    let vae_rebound = prebound.rebound(
        &vae_rebind_request,
        None,
        Some(fixture_bundle.vae().clone()),
        &cancellation,
    )?;
    assert_eq!(
        vae_rebound.vae_execution_digest(),
        Some(fixture_bundle.vae().execution_digest())
    );
    let execution = Arc::new(NativeControlExecution::checked(chain, executor)?);
    let sampling_tensor = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.25; 4], &context)?;
    let control_metadata = ConditioningMetadataMap::checked([
        (
            ConditioningMetadataKey::Control,
            ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
                execution.execution_digest(),
                false,
            )?),
        ),
        (
            ConditioningMetadataKey::ControlApplyToUnconditioned,
            ConditioningMetadataValue::Boolean(false),
        ),
        (
            ConditioningMetadataKey::ConcatLatentImage,
            ConditioningMetadataValue::Tensor(sampling_tensor.clone()),
        ),
        (
            ConditioningMetadataKey::ConcatMask,
            ConditioningMetadataValue::Tensor(sampling_tensor.clone()),
        ),
        (
            ConditioningMetadataKey::ConcatMaskIndex,
            ConditioningMetadataValue::Unsigned(0),
        ),
        (
            ConditioningMetadataKey::CameraConditions,
            ConditioningMetadataValue::Tensor(sampling_tensor.clone()),
        ),
    ])?;
    let control_entry = ConditioningEntry::checked(
        "sampler-control",
        ConditioningValue::regular(tensor_from_f32(&backend, &[1, 1], &[1.0], &context)?)?,
        ConditioningEntryOptions::default(),
    )?
    .with_metadata(control_metadata)?
    .resolve(sampling_tensor.descriptor(), &backend, &context)?;
    let cross_attention = tensor_from_f32(&backend, &[1, 1, 1], &[1.0], &context)?;
    let conditioning_payload = NativeConditioningPayload::checked_sd15(
        fixture_bundle.model_digest(),
        fixture_bundle.model(),
        fixture_bundle.conditioning().patch_graph().clone(),
        Some(execution.as_ref().clone()),
    )?;
    assert!(
        conditioning_payload
            .execute_control(
                &backend,
                &sampling_tensor,
                1.0,
                &cross_attention,
                0.5,
                &control_entry,
                &context,
            )?
            .is_some()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let control = Arc::new(NativeControlPayload::checked(
        NativeModelResourceRole::ControlNet,
        execution.clone(),
    )?);
    let publication_metadata = |seconds_start| {
        ConditioningMetadataMap::checked([
            (
                ConditioningMetadataKey::Control,
                ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
                    execution.execution_digest(),
                    false,
                )?),
            ),
            (
                ConditioningMetadataKey::SecondsStart,
                ConditioningMetadataValue::FiniteF64(seconds_start),
            ),
            (
                ConditioningMetadataKey::SecondsTotal,
                ConditioningMetadataValue::FiniteF64(5.0),
            ),
        ])
    };
    let positive = set.with_entry_metadata(vec![publication_metadata(0.5)?], &cancellation)?;
    let negative = set.with_entry_metadata(vec![publication_metadata(1.5)?], &cancellation)?;
    assert_ne!(positive.digest(), negative.digest());
    let incompatible_generation = NativeHandleStoreGeneration::with_capacities(3, 8 * 1_048_576)?;
    let incompatible_store =
        incompatible_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(402)));
    let incompatible = set.with_entry_metadata(
        vec![ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::Control,
            ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                false,
            )?),
        )])?],
        &cancellation,
    )?;
    assert!(
        publish_native_conditioning_control_transaction(
            incompatible_store.as_ref(),
            control.clone(),
            Arc::new(incompatible),
            Arc::new(negative.clone()),
            &cancellation,
        )
        .is_err()
    );
    assert!(incompatible_generation.is_empty());
    let failed_generation = NativeHandleStoreGeneration::with_capacities(2, 8 * 1_048_576)?;
    let failed_store = failed_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(403)));
    assert!(
        publish_native_conditioning_control_transaction(
            failed_store.as_ref(),
            control.clone(),
            Arc::new(positive.clone()),
            Arc::new(negative.clone()),
            &cancellation,
        )
        .is_err()
    );
    assert!(failed_generation.is_empty());

    let generation = NativeHandleStoreGeneration::with_capacities(3, 8 * 1_048_576)?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(404)));
    let published = publish_native_conditioning_control_transaction(
        store.as_ref(),
        control,
        Arc::new(positive),
        Arc::new(negative),
        &cancellation,
    )?;
    assert_eq!(generation.len(), 3);
    assert_ne!(
        published.positive.digest_sha256(),
        published.negative.digest_sha256()
    );
    let restarted_generation = generation.fresh_generation()?;
    let restarted = restarted_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(405)));
    let conditioning_type = sim_runtime::NativeHandleType::new(
        sim_runtime::NativeHandleKind::Conditioning,
        "CONDITIONING",
    )?;
    assert!(matches!(
        restarted.resolve(&published.positive, &conditioning_type, &cancellation),
        Err(sim_runtime::NativeHandleStoreError::WrongGeneration)
    ));
    Ok(())
}
