use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_media::NativePoseKeypoint;
use sim_model::{
    NativeModelPayload, NativeSdPoseHeatmapHead, NativeSdPoseModel, NativeSdPoseSd2Denoiser,
    SDPOSE_HEATMAP_CHANNELS, SDPOSE_HEATMAP_HEIGHT, SDPOSE_HEATMAP_WIDTH,
    SdPoseHeatmapHeadConfiguration, SdPoseModelError, SdPoseProjectionError, SdPoseRawKeypoint,
    SdPoseSd2Configuration, SdPoseSd2Error, decode_sdpose_heatmaps, project_sdpose_openpose_person,
    sdpose::{
        LOTUS_CONDITIONING_F16_SHA256, LOTUS_CONDITIONING_F32_SHA256, decode_sdpose_heatmap_tensor,
        prepare_lotus_sdpose_conditioning, project_sdpose_heatmap_tensor,
    },
    sdpose_heatmap_head_weight_manifest, sdpose_sd2_weight_manifest,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, ExecutionContext,
    StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::tensor_from_f32_with_context_exact_native,
};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

const WORKSPACE_BYTES: u64 = 4 * 1024 * 1024;
const SD2_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
struct ProductionManifest {
    simui_tree_sha256: String,
    tensor_count: usize,
    scalar_count: u64,
    capture: ProductionCapture,
    licensed_checkpoint_oracle: Option<String>,
}

#[derive(Deserialize)]
struct ProductionCapture {
    output_block: usize,
    shape: Vec<u64>,
}

#[derive(Deserialize)]
struct ResourceManifest {
    resource_contract: ResourceContract,
    head_state: Vec<ResourceHeadState>,
    licensed_checkpoint_oracle: Option<String>,
}

#[derive(Deserialize)]
struct ResourceContract {
    combined_tensor_count: usize,
    combined_scalar_count: u64,
    capture_shape: Vec<u64>,
    heatmap_shape: Vec<u64>,
}

#[derive(Deserialize)]
struct ResourceHeadState {
    key: String,
    shape: Vec<u64>,
}

fn fixture(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/sdpose/sd2_capture")
        .join(path)
}

fn resource_fixture(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/sdpose/resource")
        .join(path)
}

fn head_projection_fixture(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/sdpose/head_projection")
        .join(path)
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    tensor_with_dtype(backend, shape, values, DType::F32, context)
}

fn tensor_with_dtype(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        backend.device(),
        context,
    )?)
}

fn tensor_value(tensor: &Tensor, index: usize) -> Result<f32, Box<dyn Error>> {
    let index = u64::try_from(index)?;
    Ok(
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?
        {
            DecodedScalar::Boolean(value) => f32::from(u8::from(value)),
            DecodedScalar::Signed(value) => value as f32,
            DecodedScalar::Unsigned(value) => value as f32,
            DecodedScalar::Real(value) => value as f32,
            DecodedScalar::Complex { .. } => {
                return Err(std::io::Error::other("unexpected complex fixture tensor").into());
            }
        },
    )
}

fn tensor_sha256(tensor: &Tensor) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(tensor.contiguous_bytes()?)))
}

fn reduced_sd2_weights(
    backend: &CpuBackend,
    configuration: &SdPoseSd2Configuration,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn Error>> {
    let mut weights = BTreeMap::new();
    for specification in sdpose_sd2_weight_manifest(configuration)? {
        context.check()?;
        let element_count = specification
            .shape()
            .iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(usize::try_from(*dimension).ok()?)
            });
        let element_count = element_count
            .ok_or_else(|| std::io::Error::other("SD2 fixture weight is too large"))?;
        weights.insert(
            specification.key().to_owned(),
            tensor_with_dtype(
                backend,
                specification.shape(),
                &vec![0.0; element_count],
                dtype,
                context,
            )?,
        );
    }
    Ok(weights)
}

fn reduced_head_weights(
    backend: &CpuBackend,
    configuration: &SdPoseHeatmapHeadConfiguration,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn Error>> {
    let mut weights = BTreeMap::new();
    for specification in sdpose_heatmap_head_weight_manifest(configuration)? {
        context.check()?;
        let element_count = specification
            .shape()
            .iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(usize::try_from(*dimension).ok()?)
            })
            .ok_or_else(|| std::io::Error::other("SDPose head fixture weight is too large"))?;
        weights.insert(
            specification.key().to_owned(),
            tensor_with_dtype(
                backend,
                specification.shape(),
                &vec![0.0; element_count],
                dtype,
                context,
            )?,
        );
    }
    Ok(weights)
}

fn reduced_sdpose_model(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    cancellation: &CancellationToken,
) -> Result<NativeSdPoseModel, Box<dyn Error>> {
    let denoiser_configuration = SdPoseSd2Configuration::reduced_fixture(4, 3, 1, 1, 8, 8)?;
    let denoiser_weights =
        reduced_sd2_weights(backend, &denoiser_configuration, DType::F32, context)?;
    let denoiser = NativeSdPoseSd2Denoiser::from_reduced_fixture(
        denoiser_configuration,
        denoiser_weights,
        cancellation,
    )?;
    let head_configuration = SdPoseHeatmapHeadConfiguration::reduced_fixture(8, 8, 3)?;
    let head_weights = reduced_head_weights(backend, &head_configuration, DType::F32, context)?;
    let head = NativeSdPoseHeatmapHead::from_reduced_fixture(
        head_configuration,
        head_weights,
        cancellation,
    )?;
    Ok(NativeSdPoseModel::from_reduced_fixture(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        denoiser,
        head,
        cancellation,
    )?)
}

fn raw_points(score: f32) -> Result<Vec<SdPoseRawKeypoint>, SdPoseProjectionError> {
    (0..SDPOSE_HEATMAP_CHANNELS)
        .map(|index| {
            let index = f32::from(
                u16::try_from(index).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
            );
            SdPoseRawKeypoint::checked(index, index + 0.5, score)
        })
        .collect()
}

#[test]
fn sdpose_projection_preserves_raw_scores_and_openpose_cardinalities() -> Result<(), Box<dyn Error>>
{
    let mut raw = raw_points(1.25)?;
    raw[5] = SdPoseRawKeypoint::checked(10.0, 20.0, 0.5)?;
    raw[6] = SdPoseRawKeypoint::checked(14.0, 24.0, 0.4)?;
    let person = project_sdpose_openpose_person(&raw)?;

    assert_eq!(person.pose().len(), 18);
    assert_eq!(person.foot().len(), 6);
    assert_eq!(person.face().len(), 70);
    assert_eq!(person.hand_right().len(), 21);
    assert_eq!(person.hand_left().len(), 21);
    assert_eq!(person.pose()[1].x(), 12.0);
    assert_eq!(person.pose()[1].y(), 22.0);
    assert!((person.pose()[1].score() - 0.4).abs() < 1.0e-6);
    assert_eq!(person.face()[68], person.pose()[14]);
    assert_eq!(person.face()[69], person.pose()[15]);
    assert!(NativePoseKeypoint::checked(0.0, 0.0, -2.0).is_ok());
    assert!(NativePoseKeypoint::checked(0.0, 0.0, 3.0).is_ok());
    Ok(())
}

#[test]
fn sdpose_projection_uses_strict_neck_threshold() -> Result<(), Box<dyn Error>> {
    let mut raw = raw_points(0.5)?;
    raw[5] = SdPoseRawKeypoint::checked(10.0, 20.0, 0.3)?;
    raw[6] = SdPoseRawKeypoint::checked(14.0, 24.0, 0.9)?;
    let person = project_sdpose_openpose_person(&raw)?;
    assert_eq!(person.pose()[1].score(), 0.0);
    Ok(())
}

#[test]
fn sdpose_heatmap_decode_is_bounded_cancellable_and_first_index_stable()
-> Result<(), Box<dyn Error>> {
    let plane = SDPOSE_HEATMAP_HEIGHT * SDPOSE_HEATMAP_WIDTH;
    let mut heatmaps = vec![0.0; SDPOSE_HEATMAP_CHANNELS * plane];
    for channel in 0..SDPOSE_HEATMAP_CHANNELS - 2 {
        let offset = channel * plane;
        heatmaps[offset + 10 * SDPOSE_HEATMAP_WIDTH + 20] = 2.0;
        heatmaps[offset + 10 * SDPOSE_HEATMAP_WIDTH + 21] = 2.0;
    }
    heatmaps[0] = 3.0;
    let negative_offset = (SDPOSE_HEATMAP_CHANNELS - 1) * plane;
    heatmaps[negative_offset..].fill(-1.0);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let points = decode_sdpose_heatmaps(&heatmaps, 1, &backend, &context)?;
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].len(), SDPOSE_HEATMAP_CHANNELS);
    assert_eq!(points[0][0].score(), 3.0);
    assert!(points[0][0].x().is_finite());
    assert!(points[0][0].y().is_finite());
    assert!(points[0][1].x() < 21.0 * 767.0 / 191.0);
    assert_eq!(points[0][SDPOSE_HEATMAP_CHANNELS - 2].x(), -1.0);
    assert_eq!(points[0][SDPOSE_HEATMAP_CHANNELS - 2].score(), 0.0);
    assert_eq!(points[0][SDPOSE_HEATMAP_CHANNELS - 1].x(), -1.0);
    assert_eq!(points[0][SDPOSE_HEATMAP_CHANNELS - 1].score(), -1.0);
    assert_eq!(context.scratch.in_use_bytes(), 0);

    cancellation.cancel();
    assert!(matches!(
        decode_sdpose_heatmaps(&heatmaps, 1, &backend, &context),
        Err(SdPoseProjectionError::Tensor(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn lotus_conditioning_is_exact_broadcast_and_attempt_local() -> Result<(), Box<dyn Error>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(head_projection_fixture("manifest.json"))?)?;
    assert_eq!(
        manifest["simui"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(
        manifest["lotus_conditioning"]["f16le_sha256"],
        LOTUS_CONDITIONING_F16_SHA256
    );
    assert_eq!(
        manifest["lotus_conditioning"]["f32le_sha256"],
        LOTUS_CONDITIONING_F32_SHA256
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let (conditioning_f16, adm_f16) =
        prepare_lotus_sdpose_conditioning(&backend, 1, DType::F16, &context)?;
    assert_eq!(conditioning_f16.descriptor().shape(), &[1, 2, 1024]);
    assert_eq!(adm_f16.descriptor().shape(), &[1, 4]);
    assert_eq!(
        tensor_sha256(&conditioning_f16)?,
        LOTUS_CONDITIONING_F16_SHA256
    );
    assert_eq!(tensor_value(&conditioning_f16, 0)?, -0.31347656);
    assert_eq!(tensor_value(&conditioning_f16, 2047)?, 1.5009766);
    assert!((tensor_value(&adm_f16, 0)? - 1.0_f32.sin()).abs() < 5.0e-4);
    assert_eq!(tensor_value(&adm_f16, 1)?, 0.0);
    assert!((tensor_value(&adm_f16, 2)? - 1.0_f32.cos()).abs() < 5.0e-4);
    assert_eq!(tensor_value(&adm_f16, 3)?, 1.0);

    let (conditioning_f32, adm_f32) =
        prepare_lotus_sdpose_conditioning(&backend, 3, DType::F32, &context)?;
    assert_eq!(conditioning_f32.descriptor().shape(), &[3, 2, 1024]);
    assert_eq!(adm_f32.descriptor().shape(), &[3, 4]);
    let single_bytes = conditioning_f32
        .contiguous_bytes()?
        .get(..2048 * std::mem::size_of::<f32>())
        .ok_or_else(|| std::io::Error::other("missing first conditioning batch"))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(single_bytes)),
        LOTUS_CONDITIONING_F32_SHA256
    );
    for batch_index in 1..3 {
        assert_eq!(
            tensor_value(&conditioning_f32, batch_index * 2048)?,
            -0.31347656
        );
        assert_eq!(
            tensor_value(&conditioning_f32, batch_index * 2048 + 2047)?,
            1.5009766
        );
        assert!((tensor_value(&adm_f32, batch_index * 4)? - 1.0_f32.sin()).abs() < 1.0e-7);
    }
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_tensor_heatmap_projection_streams_one_plane_and_matches_slice_oracle()
-> Result<(), Box<dyn Error>> {
    let plane_length = SDPOSE_HEATMAP_HEIGHT * SDPOSE_HEATMAP_WIDTH;
    let mut values = vec![0.0; SDPOSE_HEATMAP_CHANNELS * plane_length];
    for channel in 0..SDPOSE_HEATMAP_CHANNELS {
        let offset = channel * plane_length;
        let channel_value = f32::from(u16::try_from(channel)?);
        values[offset + (channel % SDPOSE_HEATMAP_HEIGHT) * SDPOSE_HEATMAP_WIDTH] =
            1.0 + channel_value / 100.0;
    }
    let backend_bytes = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(backend_bytes)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let heatmaps = tensor(
        &backend,
        &[
            1,
            u64::try_from(SDPOSE_HEATMAP_CHANNELS)?,
            u64::try_from(SDPOSE_HEATMAP_HEIGHT)?,
            u64::try_from(SDPOSE_HEATMAP_WIDTH)?,
        ],
        &values,
        &context,
    )?;
    let slice = decode_sdpose_heatmaps(&values, 1, &backend, &context)?;
    let streamed = decode_sdpose_heatmap_tensor(&heatmaps, &backend, &context)?;
    assert_eq!(streamed, slice);
    assert_eq!(
        project_sdpose_heatmap_tensor(&heatmaps, &backend, &context)?,
        vec![project_sdpose_openpose_person(&slice[0])?]
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_heatmap_decode_rejects_shape_nonfinite_and_workspace_exhaustion()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        decode_sdpose_heatmaps(&[], 0, &backend, &context),
        Err(SdPoseProjectionError::InvalidHeatmapShape)
    ));

    let plane = SDPOSE_HEATMAP_HEIGHT * SDPOSE_HEATMAP_WIDTH;
    let mut heatmaps = vec![1.0; SDPOSE_HEATMAP_CHANNELS * plane];
    heatmaps[0] = f32::NAN;
    assert!(matches!(
        decode_sdpose_heatmaps(&heatmaps, 1, &backend, &context),
        Err(SdPoseProjectionError::NonFiniteInput)
    ));

    heatmaps[0] = 1.0;
    let (tiny_backend, tiny_authority) = CpuWorkspaceAuthority::create_backend(64)?;
    let tiny_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: tiny_authority.authorize_workspace(64)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        decode_sdpose_heatmaps(&heatmaps, 1, &tiny_backend, &tiny_context),
        Err(SdPoseProjectionError::Tensor(_))
    ));
    assert_eq!(tiny_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_sd2_manifest_is_complete_and_source_sized() -> Result<(), Box<dyn Error>> {
    let tracked: ProductionManifest =
        serde_json::from_slice(&fs::read(fixture("production_manifest/manifest.json"))?)?;
    assert_eq!(
        tracked.simui_tree_sha256,
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(tracked.tensor_count, 690);
    assert_eq!(tracked.scalar_count, 867_556_804);
    assert_eq!(tracked.capture.output_block, 9);
    assert_eq!(tracked.capture.shape, [1, 640, 128, 96]);
    assert!(tracked.licensed_checkpoint_oracle.is_none());
    let manifest = sdpose_sd2_weight_manifest(&SdPoseSd2Configuration::source())?;
    assert_eq!(manifest.len(), 690);
    let scalar_count = manifest.iter().try_fold(0u64, |count, specification| {
        let elements = specification
            .shape()
            .iter()
            .try_fold(1u64, |elements, dimension| elements.checked_mul(*dimension))?;
        count.checked_add(elements)
    });
    assert_eq!(scalar_count, Some(867_556_804));
    assert!(
        manifest
            .iter()
            .any(|specification| specification.key() == "native.output_blocks.9.1.proj_out.weight")
    );
    assert!(
        manifest
            .iter()
            .all(|specification| !specification.key().starts_with("native.heatmap_head."))
    );
    Ok(())
}

#[test]
fn sdpose_model_resource_manifest_is_exact_and_production_admission_is_closed()
-> Result<(), Box<dyn Error>> {
    let tracked: ResourceManifest = serde_json::from_slice(&fs::read(resource_fixture(
        "production_manifest/manifest.json",
    ))?)?;
    assert_eq!(tracked.resource_contract.combined_tensor_count, 695);
    assert_eq!(tracked.resource_contract.combined_scalar_count, 874_605_897);
    assert_eq!(tracked.resource_contract.capture_shape, [1, 640, 128, 96]);
    assert_eq!(tracked.resource_contract.heatmap_shape, [1, 133, 256, 192]);
    assert!(tracked.licensed_checkpoint_oracle.is_none());
    let head_manifest =
        sdpose_heatmap_head_weight_manifest(&SdPoseHeatmapHeadConfiguration::source())?;
    assert_eq!(head_manifest.len(), 5);
    let head_scalars = head_manifest.iter().try_fold(0u64, |count, specification| {
        let elements = specification
            .shape()
            .iter()
            .try_fold(1u64, |elements, dimension| elements.checked_mul(*dimension))?;
        count.checked_add(elements)
    });
    assert_eq!(head_scalars, Some(7_049_093));
    assert_eq!(
        head_manifest
            .first()
            .map(|specification| specification.shape()),
        Some([640, 640, 4, 4].as_slice())
    );
    assert_eq!(
        head_manifest
            .last()
            .map(|specification| specification.shape()),
        Some([133].as_slice())
    );
    assert_eq!(tracked.head_state.len(), head_manifest.len());
    for (tracked, specification) in tracked.head_state.iter().zip(&head_manifest) {
        assert_eq!(tracked.key, specification.key());
        assert_eq!(tracked.shape, specification.shape());
    }

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let resource = std::sync::Arc::new(reduced_sdpose_model(&backend, &context, &cancellation)?);
    assert!(!resource.is_source_exact_profile());
    assert!(matches!(
        NativeModelPayload::sdpose_model(resource),
        Err(sim_model::NativeModelPayloadError::ResourceMismatch(
            "SDPose production source-exact profile"
        ))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_model_resource_identity_residency_and_cancellation_are_transactional()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let first = reduced_sdpose_model(&backend, &context, &cancellation)?;
    let second = reduced_sdpose_model(&backend, &context, &cancellation)?;
    assert_eq!(
        first.semantic_state_digest_sha256(),
        second.semantic_state_digest_sha256()
    );
    assert_eq!(first.resident_tensor_allocations()?.len(), 695);
    let first_storage = first
        .resident_tensor_allocations()?
        .into_iter()
        .map(|(storage, _)| storage.get())
        .collect::<std::collections::BTreeSet<_>>();
    let second_storage = second
        .resident_tensor_allocations()?
        .into_iter()
        .map(|(storage, _)| storage.get())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(first_storage.is_disjoint(&second_storage));
    assert_eq!(first.resident_bytes()?, second.resident_bytes()?);
    first.validate(&cancellation)?;

    cancellation.cancel();
    assert!(matches!(
        first.validate(&cancellation),
        Err(SdPoseModelError::Cancellation(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_heatmap_head_rejects_missing_and_nonfinite_state() -> Result<(), Box<dyn Error>> {
    let configuration = SdPoseHeatmapHeadConfiguration::reduced_fixture(8, 8, 3)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let mut missing = reduced_head_weights(&backend, &configuration, DType::F32, &context)?;
    missing.remove("native.heatmap_head.final_layer.bias");
    assert!(matches!(
        NativeSdPoseHeatmapHead::from_reduced_fixture(
            configuration.clone(),
            missing,
            &cancellation,
        ),
        Err(SdPoseModelError::WeightKeys { .. })
    ));

    let mut nonfinite = reduced_head_weights(&backend, &configuration, DType::F32, &context)?;
    nonfinite.insert(
        "native.heatmap_head.final_layer.bias".to_owned(),
        tensor(&backend, &[3], &[f32::INFINITY, 0.0, 0.0], &context)?,
    );
    assert!(matches!(
        NativeSdPoseHeatmapHead::from_reduced_fixture(configuration, nonfinite, &cancellation),
        Err(SdPoseModelError::NonFinite)
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_heatmap_head_executes_retained_transpose_norm_activation_and_projection()
-> Result<(), Box<dyn Error>> {
    let configuration = SdPoseHeatmapHeadConfiguration::reduced_fixture(2, 2, 3)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let head = NativeSdPoseHeatmapHead::from_reduced_fixture(
        configuration,
        reduced_head_weights(
            &backend,
            &SdPoseHeatmapHeadConfiguration::reduced_fixture(2, 2, 3)?,
            DType::F32,
            &context,
        )?,
        &cancellation,
    )?;
    let feature = tensor(
        &backend,
        &[1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        &context,
    )?;
    let heatmaps = head.forward(&backend, &feature, &context)?;
    assert_eq!(heatmaps.descriptor().shape(), &[1, 3, 4, 4]);
    assert_eq!(heatmaps.descriptor().dtype(), DType::F32);
    assert_ne!(heatmaps.storage_id(), feature.storage_id());
    assert!(heatmaps.contiguous_bytes()?.iter().all(|value| *value == 0));
    assert_eq!(context.scratch.in_use_bytes(), 0);

    for dtype in [DType::F16, DType::Bf16] {
        let configuration = SdPoseHeatmapHeadConfiguration::reduced_fixture(2, 2, 3)?;
        let head = NativeSdPoseHeatmapHead::from_reduced_fixture(
            configuration.clone(),
            reduced_head_weights(&backend, &configuration, dtype, &context)?,
            &cancellation,
        )?;
        let feature = tensor_with_dtype(
            &backend,
            &[1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
            dtype,
            &context,
        )?;
        let heatmaps = head.forward(&backend, &feature, &context)?;
        assert_eq!(heatmaps.descriptor().shape(), &[1, 3, 4, 4]);
        assert_eq!(heatmaps.descriptor().dtype(), dtype);
        assert_ne!(heatmaps.storage_id(), feature.storage_id());
        assert!(heatmaps.contiguous_bytes()?.iter().all(|value| *value == 0));
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(matches!(
        head.forward(&backend, &feature, &cancelled_context),
        Err(SdPoseModelError::Cancellation(_))
            | Err(SdPoseModelError::Module(_))
            | Err(SdPoseModelError::Tensor(_))
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_sd2_reduced_execution_captures_last_preconcat_feature_transactionally()
-> Result<(), Box<dyn Error>> {
    let configuration = SdPoseSd2Configuration::reduced_fixture(4, 3, 1, 1, 8, 8)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let weights = reduced_sd2_weights(&backend, &configuration, DType::F32, &context)?;
    let denoiser =
        NativeSdPoseSd2Denoiser::from_reduced_fixture(configuration, weights, &cancellation)?;
    assert_eq!(denoiser.resident_tensor_allocations()?.len(), 690);

    let latent = tensor(&backend, &[1, 4, 8, 8], &vec![0.0; 256], &context)?;
    let conditioning = tensor(&backend, &[1, 2, 3], &[0.0; 6], &context)?;
    let adm = tensor(&backend, &[1, 4], &[0.0; 4], &context)?;
    let first = denoiser.forward(&backend, &latent, &[999.0], &conditioning, &adm, &context)?;
    assert_eq!(first.denoised().descriptor().shape(), [1, 4, 8, 8]);
    assert_eq!(first.feature_640().descriptor().shape(), [1, 8, 8, 8]);
    assert_eq!(first.capture_output_block(), 9);
    assert_ne!(first.feature_640().storage_id(), latent.storage_id());
    assert_eq!(context.scratch.in_use_bytes(), 0);

    let second = denoiser.forward(&backend, &latent, &[999.0], &conditioning, &adm, &context)?;
    assert_ne!(
        first.feature_640().storage_id(),
        second.feature_640().storage_id()
    );
    assert_eq!(
        first.feature_640().contiguous_bytes()?,
        second.feature_640().contiguous_bytes()?
    );

    cancellation.cancel();
    assert!(matches!(
        denoiser.forward(&backend, &latent, &[999.0], &conditioning, &adm, &context,),
        Err(SdPoseSd2Error::Cancellation(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_sd2_admission_rejects_missing_and_nonfinite_weights() -> Result<(), Box<dyn Error>> {
    let configuration = SdPoseSd2Configuration::reduced_fixture(4, 3, 1, 1, 8, 8)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
    let cancellation = CancellationToken::default();
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let mut missing = reduced_sd2_weights(&backend, &configuration, DType::F32, &context)?;
    missing.remove("native.out.2.bias");
    assert!(matches!(
        NativeSdPoseSd2Denoiser::from_reduced_fixture(
            configuration.clone(),
            missing,
            &cancellation,
        ),
        Err(SdPoseSd2Error::WeightKeys { .. })
    ));

    let mut nonfinite = reduced_sd2_weights(&backend, &configuration, DType::F32, &context)?;
    nonfinite.insert(
        "native.out.2.bias".to_owned(),
        tensor(&backend, &[4], &[f32::NAN, 0.0, 0.0, 0.0], &context)?,
    );
    assert!(matches!(
        NativeSdPoseSd2Denoiser::from_reduced_fixture(configuration, nonfinite, &cancellation,),
        Err(SdPoseSd2Error::NonFinite)
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn sdpose_sd2_reduced_execution_preserves_low_precision_dtype() -> Result<(), Box<dyn Error>> {
    let mut semantic_digests = Vec::new();
    for dtype in [DType::F16, DType::Bf16] {
        let configuration = SdPoseSd2Configuration::reduced_fixture(4, 3, 1, 1, 8, 8)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(SD2_WORKSPACE_BYTES)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(SD2_WORKSPACE_BYTES)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let weights = reduced_sd2_weights(&backend, &configuration, dtype, &context)?;
        let denoiser =
            NativeSdPoseSd2Denoiser::from_reduced_fixture(configuration, weights, &cancellation)?;
        semantic_digests.push(denoiser.semantic_state_digest_sha256().to_owned());
        let latent = tensor_with_dtype(&backend, &[1, 4, 8, 8], &vec![0.0; 256], dtype, &context)?;
        let conditioning = tensor_with_dtype(&backend, &[1, 2, 3], &[0.0; 6], dtype, &context)?;
        let adm = tensor_with_dtype(&backend, &[1, 4], &[0.0; 4], dtype, &context)?;
        let output =
            denoiser.forward(&backend, &latent, &[999.0], &conditioning, &adm, &context)?;
        assert_eq!(output.denoised().descriptor().dtype(), dtype);
        assert_eq!(output.feature_640().descriptor().dtype(), dtype);
        assert_eq!(output.capture_output_block(), 9);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }
    assert!(semantic_digests.windows(2).all(|pair| {
        matches!((pair.first(), pair.get(1)), (Some(first), Some(second)) if first != second)
    }));
    Ok(())
}
