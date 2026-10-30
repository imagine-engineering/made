use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionRequest, MathSdpSelection, QuantizationError, QuantizationKind,
    QuantizationMetadataV1, SdpaBackend, allow_fp16_bf16_reduction_math_sdp_exact_native,
    enable_flash_sdp_exact_native, enable_math_sdp_exact_native, quantize_matrix,
    scaled_dot_product_attention_with_context, sdpa_kernel_exact_native,
};
use sim_tensor::{
    ALL_DTYPES, BackendCapabilityMatrix, CATALOG_MODEL_DTYPES, DType, DecodedScalar, Layout,
    OperationSupport, Scalar, StreamId, TensorError, decode_float8,
    generated_accelerated_attention_kernel_01::AttentionKernelKind, promote_types,
};
use sim_types::CancellationToken;
use sim_types::WorkerDType;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

const FIXTURE_PATH: &str =
    "crates/sim_test_support/fixtures/numeric_formats/sim_numeric_formats_v1.json";
const FIXTURE_SHA256: &str = "38969f0a48ff3986dad10d0121e44a03545190ad48b08cb3c3e28593df7322e4";

fn run_attention(
    request: AttentionRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    cancellation: &CancellationToken,
) -> Result<sim_model::AttentionOutcome, AttentionError> {
    let (backend, workspace_authority) = sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(8)?,
        cancellation,
    );
    scaled_dot_product_attention_with_context(&backend, request, query, key, value, mask, &context)
}

#[test]
fn task_55_part_twelve_math_sdp_mapping_uses_the_canonical_model_attention_owner()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    assert_eq!(
        enable_math_sdp_exact_native(true, &cancellation)?,
        MathSdpSelection::Enabled(AttentionKernelKind::ReferenceSdp)
    );
    assert_eq!(
        enable_math_sdp_exact_native(false, &cancellation)?,
        MathSdpSelection::Disabled
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(enable_math_sdp_exact_native(true, &cancelled).is_err());
    Ok(())
}

#[test]
fn task_57_math_sdp_reduction_uses_the_canonical_attention_policy() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    assert!(
        allow_fp16_bf16_reduction_math_sdp_exact_native(true, &cancellation)?.allow_fp16_bf16()
    );
    assert!(
        !allow_fp16_bf16_reduction_math_sdp_exact_native(false, &cancellation)?.allow_fp16_bf16()
    );
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(allow_fp16_bf16_reduction_math_sdp_exact_native(true, &cancelled).is_err());
    Ok(())
}

#[test]
fn task_58_flash_sdp_uses_the_canonical_attention_policy() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    assert_eq!(
        enable_flash_sdp_exact_native(true, &cancellation)?,
        MathSdpSelection::Enabled(AttentionKernelKind::FlashAttention)
    );
    assert_eq!(
        enable_flash_sdp_exact_native(false, &cancellation)?,
        MathSdpSelection::Disabled
    );
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(enable_flash_sdp_exact_native(true, &cancelled).is_err());
    Ok(())
}

#[test]
fn task_60_sdpa_kernel_uses_the_canonical_attention_policy() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let backends = [SdpaBackend::FlashAttention, SdpaBackend::Math];
    let selection = sdpa_kernel_exact_native(&backends, true, &cancellation)?;
    assert_eq!(selection.backends(), &backends);
    assert!(selection.set_priority());
    assert!(sdpa_kernel_exact_native(&[], false, &cancellation).is_err());
    assert!(
        sdpa_kernel_exact_native(
            &[SdpaBackend::Math, SdpaBackend::Math],
            false,
            &cancellation,
        )
        .is_err()
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        sdpa_kernel_exact_native(&[], false, &cancelled),
        Err(AttentionError::Cancelled)
    ));
    Ok(())
}

#[test]
fn val_numeric_formats_001() -> Result<(), Box<dyn Error>> {
    let workspace = workspace_root()?;
    let fixture_bytes = fs::read(workspace.join(FIXTURE_PATH))?;
    let fixture_digest = sha256(&fixture_bytes);
    if fixture_digest != FIXTURE_SHA256 {
        return Err(io::Error::other(format!(
            "numeric fixture digest mismatch: expected {FIXTURE_SHA256}, got {fixture_digest}"
        ))
        .into());
    }
    let fixture: Value = serde_json::from_slice(&fixture_bytes)?;
    verify_source_digests(&workspace, &fixture)?;
    let mut cases = BTreeMap::new();

    let feature_ids = fixture
        .get("catalog_feature_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| io::Error::other("fixture catalog_feature_ids are missing"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| io::Error::other("fixture feature ID is not a string"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let dtype_ids = CATALOG_MODEL_DTYPES
        .iter()
        .map(|(_, feature_id)| *feature_id)
        .collect::<BTreeSet<_>>();
    cases.insert(
        "nine_catalog_dtypes_have_one_canonical_owner",
        dtype_ids.len() == 9 && dtype_ids.is_subset(&feature_ids),
    );
    cases.insert(
        "all_catalog_dtype_names_are_unique",
        CATALOG_MODEL_DTYPES
            .iter()
            .map(|(dtype, _)| dtype.catalog_name())
            .collect::<BTreeSet<_>>()
            .len()
            == CATALOG_MODEL_DTYPES.len(),
    );
    cases.insert("worker_dtype_boundary_round_trips_every_canonical_dtype", {
        ALL_DTYPES.iter().all(|dtype| {
            let wire = WorkerDType::from(*dtype);
            DType::from(wire) == *dtype
                && serde_json::to_vec(&wire)
                    .and_then(|bytes| serde_json::from_slice::<WorkerDType>(&bytes))
                    .is_ok_and(|decoded| decoded == wire)
        })
    });
    cases.insert("cpu_storage_capabilities_delegate_to_backend_matrix", {
        BackendCapabilityMatrix::for_native_device(sim_tensor::DeviceId::CPU).is_ok_and(|matrix| {
            CATALOG_MODEL_DTYPES.iter().all(|(dtype, _)| {
                matrix.supports(OperationSupport::copy_input(*dtype, Layout::Contiguous))
                    && matrix.supports(OperationSupport::copy_output(*dtype, Layout::Contiguous))
            }) && !matrix.supports(OperationSupport::fill(
                DType::Float8E8m0Fnu,
                Layout::Contiguous,
            ))
        })
    });
    cases.insert("float8_boundary_encodings_match_source_contract", {
        decode_float8(DType::Float8E4m3Fn, 126) == 448.0
            && decode_float8(DType::Float8E4m3Fnuz, 127) == 240.0
            && decode_float8(DType::Float8E5m2, 123) == 57_344.0
            && decode_float8(DType::Float8E5m2Fnuz, 127) == 57_344.0
            && decode_float8(DType::Float8E8m0Fnu, 127) == 1.0
    });
    cases.insert("nan_inf_and_rounding_are_typed", {
        matches!(
            DType::Float8E5m2.decode_scalar(&[0x7c]),
            Ok(DecodedScalar::Real(value)) if value.is_infinite()
        ) && matches!(
            DType::Float8E4m3Fn.decode_scalar(&[0x7f]),
            Ok(DecodedScalar::Real(value)) if value.is_nan()
        ) && matches!(
            DType::U8.encode_scalar(
                Scalar::Float(f64::NAN),
                "validation",
                sim_tensor::DeviceId::CPU
            ),
            Err(TensorError::InvalidNumeric { .. })
        )
    });
    cases.insert("promotion_matches_pytorch_contract", {
        promote_types(DType::U8, DType::I8) == Ok(DType::I16)
            && promote_types(DType::U32, DType::I32) == Ok(DType::I64)
            && promote_types(DType::U64, DType::I64) == Ok(DType::F64)
            && promote_types(DType::F16, DType::Bf16) == Ok(DType::F32)
            && promote_types(DType::F32, DType::Complex64) == Ok(DType::Complex64)
            && matches!(
                promote_types(DType::Float8E4m3Fn, DType::F16),
                Err(TensorError::UnsupportedCapability { .. })
            )
    });

    let values = [-6.0, -1.0, 0.0, 1.0, 6.0, 12.0];
    let token = CancellationToken::default();
    for (kind, name, tolerance) in [
        (QuantizationKind::Int8Tensorwise, "int8_tensorwise", 0.05),
        (QuantizationKind::MxFp8, "mxfp8", 0.5),
        (QuantizationKind::NvFp4, "nvfp4", 1.0),
    ] {
        let quantized = quantize_matrix(kind, DType::F32, &values, 2, 3, &token)?;
        let decoded = quantized.dequantize(&token)?;
        let maximum_error = decoded
            .iter()
            .zip(values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        cases.insert(name, maximum_error <= tolerance && quantized.kind() == kind);
    }
    let metadata = QuantizationMetadataV1::parse_json(
        br#"{"version":1,"layers":{"weight":{"algorithm":"int8_tensorwise","original_dtype":"f32"}}}"#,
    )?;
    cases.insert(
        "mixed_per_layer_metadata_v1_is_checked",
        metadata
            .quantize_layer("weight", &values, 2, 3, &token)?
            .kind()
            == QuantizationKind::Int8Tensorwise
            && matches!(
                metadata.quantize_layer("missing", &values, 2, 3, &token),
                Err(QuantizationError::MissingLayer { .. })
            ),
    );
    cases.insert("mixed_per_layer_metadata_is_bounded", {
        let oversized = vec![b' '; 1024 * 1024 + 1];
        matches!(
            QuantizationMetadataV1::parse_json(&oversized),
            Err(QuantizationError::InvalidMetadata { .. })
        )
    });
    cases.insert(
        "quantization_rejects_nonfinite_shape_and_cancellation",
        matches!(
            quantize_matrix(
                QuantizationKind::Int8Tensorwise,
                DType::F32,
                &[f32::NAN],
                1,
                1,
                &token,
            ),
            Err(QuantizationError::NonFinite { .. })
        ) && matches!(
            quantize_matrix(
                QuantizationKind::MxFp8,
                DType::F32,
                &[1.0],
                usize::MAX,
                2,
                &token,
            ),
            Err(QuantizationError::ShapeOverflow)
        ) && {
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            matches!(
                quantize_matrix(
                    QuantizationKind::NvFp4,
                    DType::F32,
                    &[1.0],
                    1,
                    1,
                    &cancelled,
                ),
                Err(QuantizationError::Cancelled)
            )
        },
    );

    let query = [1.0, 0.0, 0.0, 1.0];
    let key = query;
    let value = [2.0, 0.0, 0.0, 4.0];
    let boolean_mask = [true, false, true, true];
    let exact_request = attention_request(AttentionBackend::PytorchSdp);
    let exact = run_attention(
        exact_request,
        &query,
        &key,
        &value,
        Some(AttentionMask::Boolean {
            values: &boolean_mask,
            shape: AttentionMaskShape::QueryByKey,
        }),
        &token,
    )?;
    let expected = [2.0, 0.0, 0.537_882_8, 2.924_234_4];
    cases.insert(
        "native_sdp_matches_checked_in_sim_fixture",
        exact
            .values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 0.000_001),
    );
    let split = run_attention(
        attention_request(AttentionBackend::SplitOrSubQuadratic),
        &query,
        &key,
        &value,
        Some(AttentionMask::Boolean {
            values: &boolean_mask,
            shape: AttentionMaskShape::QueryByKey,
        }),
        &token,
    )?;
    cases.insert(
        "split_attention_is_memory_bounded_and_exact",
        split.values == exact.values
            && split.query_chunk_size == 1
            && split.peak_workspace_bytes == 8,
    );
    cases.insert("attention_additive_broadcast_and_cross_layout_are_exact", {
        let additive_mask = [0.0, f32::NEG_INFINITY, 0.0, 0.0];
        let additive = run_attention(
            exact_request,
            &query,
            &key,
            &value,
            Some(AttentionMask::Additive {
                values: &additive_mask,
                shape: AttentionMaskShape::QueryByKey,
            }),
            &token,
        );
        let mut cross_request = exact_request;
        cross_request.query_tokens = 1;
        let key_mask = [true, false];
        let cross = run_attention(
            cross_request,
            &[1.0, 0.0],
            &key,
            &value,
            Some(AttentionMask::Boolean {
                values: &key_mask,
                shape: AttentionMaskShape::KeyTokens,
            }),
            &token,
        );
        matches!(additive, Ok(outcome) if outcome.values == exact.values)
            && matches!(cross, Ok(outcome) if outcome.values == [2.0, 0.0])
    });
    cases.insert("optimized_backends_fallback_only_when_allowed", {
        let fallback = run_attention(
            attention_request(AttentionBackend::Xformers),
            &query,
            &key,
            &value,
            None,
            &token,
        );
        let mut forbidden = attention_request(AttentionBackend::SageOrFlash);
        forbidden.fallback = AttentionFallbackPolicy::Forbid;
        matches!(
            fallback,
            Ok(outcome)
                if outcome.effective_backend == AttentionBackend::PytorchSdp
                    && outcome.fallback_reason.is_some()
        ) && matches!(
            run_attention(forbidden, &query, &key, &value, None, &token),
            Err(AttentionError::UnsupportedBackend { .. })
        )
    });
    cases.insert("attention_masks_workspace_and_cancellation_fail_typed", {
        let mut small_workspace = exact_request;
        small_workspace.workspace_limit_bytes = 7;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        matches!(
            run_attention(small_workspace, &query, &key, &value, None, &token,),
            Err(AttentionError::WorkspaceTooSmall { .. })
        ) && matches!(
            run_attention(
                exact_request,
                &query,
                &key,
                &value,
                Some(AttentionMask::Boolean {
                    values: &[true],
                    shape: AttentionMaskShape::QueryByKey,
                }),
                &token,
            ),
            Err(AttentionError::MaskValueCount { .. })
        ) && matches!(
            run_attention(exact_request, &query, &key, &value, None, &cancelled,),
            Err(AttentionError::Cancelled)
        )
    });
    cases.insert(
        "all_four_attention_and_quantization_catalog_ids_are_exact",
        [
            AttentionBackend::PytorchSdp,
            AttentionBackend::SageOrFlash,
            AttentionBackend::SplitOrSubQuadratic,
            AttentionBackend::Xformers,
        ]
        .iter()
        .all(|backend| feature_ids.contains(backend.feature_id()))
            && [
                QuantizationKind::Int8Tensorwise,
                QuantizationKind::MxFp8,
                QuantizationKind::NvFp4,
                QuantizationKind::MixedPerLayerV1,
            ]
            .iter()
            .all(|kind| feature_ids.contains(kind.feature_id())),
    );

    if let Some((name, _)) = cases.iter().find(|(_, passed)| !**passed) {
        return Err(io::Error::other(format!("numeric validation case failed: {name}")).into());
    }
    write_artifact(&workspace, &fixture_digest, &cases)?;
    Ok(())
}

fn attention_request(backend: AttentionBackend) -> AttentionRequest {
    AttentionRequest {
        backend,
        fallback: AttentionFallbackPolicy::AllowExactNative,
        batch: 1,
        query_tokens: 2,
        key_tokens: 2,
        heads: 1,
        head_dimension: 2,
        value_dimension: 2,
        scale: Some(1.0),
        workspace_limit_bytes: 8,
    }
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| io::Error::other("workspace root is unavailable"))?
        .to_path_buf())
}

fn verify_source_digests(workspace: &Path, fixture: &Value) -> Result<(), Box<dyn Error>> {
    let sources = fixture
        .get("sources")
        .and_then(Value::as_object)
        .ok_or_else(|| io::Error::other("numeric fixture sources are missing"))?;
    for (path, expected) in sources {
        let expected = expected
            .as_str()
            .ok_or_else(|| io::Error::other("numeric source digest is not a string"))?;
        let actual = sha256(&fs::read(workspace.join(path))?);
        if actual != expected {
            return Err(io::Error::other(format!(
                "numeric source digest mismatch for {path}: expected {expected}, got {actual}"
            ))
            .into());
        }
    }
    Ok(())
}

fn write_artifact(
    workspace: &Path,
    fixture_digest: &str,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace.join(path)
            }
        })
        .unwrap_or_else(|| workspace.join("target"));
    let directory = target.join("sim-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation": "VAL-NUMERIC-FORMATS-001",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "native-rust-cpu"
        },
        "fixture_digests": { FIXTURE_PATH: fixture_digest },
        "cases": cases,
        "catalog_counts": {
            "dtype": 9,
            "attention": 4,
            "quantization": 4
        },
        "skipped": []
    });
    let bytes = serde_json::to_vec_pretty(&artifact)?;
    let temporary = directory.join("val-numeric-formats-001.json.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, directory.join("val-numeric-formats-001.json"))?;
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
