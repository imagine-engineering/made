use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Scalar, StreamId, Tensor,
    TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::TorchArchiveValue,
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, NativeAdamW, NativeBitwiseOperation,
        TorchArchiveLoadError, TorchArchiveLoader, bitwise_binary_with_context_exact_native,
        bitwise_xor_with_context_exact_native, clamp_jvp_with_context_exact_native,
        clamp_vjp_with_context_exact_native, clamp_with_context_exact_native,
        expit_jvp_with_context_exact_native, expit_vjp_with_context_exact_native,
        expit_with_context_exact_native, frombuffer_with_context_exact_native,
        full_like_with_context_exact_native, mul_jvp_with_context_exact_native,
        mul_vjp_with_context_exact_native, mul_with_context_exact_native,
        ndtri_jvp_with_context_exact_native, ndtri_vjp_with_context_exact_native,
        ndtri_with_context_exact_native, npu_current_device_exact_native,
        pow_jvp_with_context_exact_native, pow_vjp_with_context_exact_native,
        pow_with_context_exact_native, tensor_constructor_with_context_exact_native,
        torch_load_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[test]
fn inverse_normal_f64_preserves_source_schedule_tail_probabilities() {
    use sim_tensor::generated_elementwise_or_runtime_operation_09::inverse_normal_cdf_f64;
    for (probability, expected) in [
        (1e-12, -7.034483825301131),
        (0.02425, -1.972961051311885),
        (0.5, 0.0),
        (0.97575, 1.972961051311885),
        (0.999999999999, 7.0344869100478356),
    ] {
        let actual = inverse_normal_cdf_f64(probability);
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 1e-8, "{probability}: {actual}");
    }
    assert_eq!(inverse_normal_cdf_f64(0.0), f64::NEG_INFINITY);
    assert_eq!(inverse_normal_cdf_f64(1.0), f64::INFINITY);
    for probability in [-1.0, 2.0, f64::NAN] {
        assert!(inverse_normal_cdf_f64(probability).is_nan());
    }
}

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-6A39A98E4F68",
    "COMFY-TENSOR-OP-616CB031A549",
    "COMFY-TENSOR-OP-67B5EDD39C41",
    "COMFY-TENSOR-OP-6A4C6EDFC695",
    "COMFY-TENSOR-OP-6664BEC3F5BD",
    "COMFY-TENSOR-OP-65EF512E0143",
    "COMFY-TENSOR-OP-615251B481B7",
    "COMFY-TENSOR-OP-6311D94BE18A",
    "COMFY-TENSOR-OP-6238000D28B1",
    "COMFY-TENSOR-OP-67F181B603A1",
    "COMFY-TENSOR-OP-69E05601EAEA",
    "COMFY-TENSOR-OP-6520A75955CD",
];

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
}

fn upload_f32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn values(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            cancellation,
        ),
    )?)
}

fn decoded(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for index in 0..usize::try_from(tensor.descriptor().element_count()?)? {
        values.push(
            tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.element_bytes(&[u64::try_from(index)?])?)?,
        );
    }
    Ok(values)
}

struct FixedArchiveLoader(TorchArchiveValue);

impl TorchArchiveLoader for FixedArchiveLoader {
    fn load_weights_cpu(
        &self,
        _backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<TorchArchiveValue, TorchArchiveLoadError> {
        context
            .check()
            .map_err(|_| TorchArchiveLoadError::Cancelled)?;
        Ok(self.0.clone())
    }
}

#[test]
fn constructors_and_integer_adapter_use_canonical_storage() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64)?,
        &cancellation,
    );
    let tensor = tensor_constructor_with_context_exact_native(
        &backend,
        &[Scalar::Signed(1), Scalar::Signed(2), Scalar::Signed(3)],
        &[3],
        DType::I16,
        StreamId::DEFAULT,
        &execution,
    )?;
    assert_eq!(
        decoded(&tensor)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(3)
        ]
    );
    let right = tensor_constructor_with_context_exact_native(
        &backend,
        &[Scalar::Signed(3)],
        &[1],
        DType::I16,
        StreamId::DEFAULT,
        &execution,
    )?;
    assert_eq!(
        decoded(&bitwise_xor_with_context_exact_native(
            &backend, &tensor, &right, &execution
        )?)?,
        [
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0)
        ]
    );

    let source = [9_u8, 0, 10, 0, 11, 0, 12, 0];
    let from_buffer = frombuffer_with_context_exact_native(
        &backend,
        &source,
        DType::I16,
        Some(2),
        2,
        StreamId::DEFAULT,
        &execution,
    )?;
    assert_eq!(
        decoded(&from_buffer)?,
        [DecodedScalar::Signed(10), DecodedScalar::Signed(11)]
    );
    Ok(())
}

#[test]
fn tensor_constructor_workspace_is_exact_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let baseline = backend.memory_snapshot().current_bytes;
    let values = [Scalar::Signed(1), Scalar::Signed(2), Scalar::Signed(3)];
    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(5)?,
        &cancellation,
    );
    assert!(
        tensor_constructor_with_context_exact_native(
            &backend,
            &values,
            &[3],
            DType::I16,
            StreamId::DEFAULT,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(6)?,
        &cancellation,
    );
    let output = tensor_constructor_with_context_exact_native(
        &backend,
        &values,
        &[3],
        DType::I16,
        StreamId::DEFAULT,
        &exact,
    )?;
    assert_eq!(decoded(&output)?.len(), 3);
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(6)?,
        &cancelled,
    );
    assert!(
        tensor_constructor_with_context_exact_native(
            &backend,
            &values,
            &[3],
            DType::I16,
            StreamId::DEFAULT,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn clamp_full_like_and_binary_derivatives_preserve_broadcast_rules()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64)?,
        &cancellation,
    );
    let input = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[-1.0, 0.5, 2.0, 4.0],
        &cancellation,
    )?;
    let clamped = clamp_with_context_exact_native(
        &backend,
        &input,
        Some(Scalar::Float(0.0)),
        Some(Scalar::Float(2.0)),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &clamped, &cancellation)?,
        [0.0, 0.5, 2.0, 2.0]
    );
    assert_eq!(
        values(&backend, &authority, &input, &cancellation)?,
        [-1.0, 0.5, 2.0, 4.0]
    );
    let clamp_tangent = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &clamp_vjp_with_context_exact_native(
                &backend,
                &input,
                Some(0.0),
                Some(2.0),
                &clamp_tangent,
                &execution,
            )?,
            &cancellation,
        )?,
        [0.0, 2.0, 3.0, 0.0]
    );
    assert_eq!(
        clamp_jvp_with_context_exact_native(
            &backend,
            &input,
            Some(0.0),
            Some(2.0),
            &clamp_tangent,
            &execution,
        )?
        .contiguous_bytes()?,
        clamp_vjp_with_context_exact_native(
            &backend,
            &input,
            Some(0.0),
            Some(2.0),
            &clamp_tangent,
            &execution,
        )?
        .contiguous_bytes()?
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &full_like_with_context_exact_native(
                &backend,
                &input,
                Scalar::Float(7.0),
                None,
                &execution,
            )?,
            &cancellation
        )?,
        [7.0; 4]
    );

    let left = upload_f32(&backend, &authority, &[2, 1], &[2.0, 3.0], &cancellation)?;
    let right = upload_f32(&backend, &authority, &[1, 2], &[4.0, 5.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &mul_with_context_exact_native(&backend, &left, &right, &execution)?,
            &cancellation
        )?,
        [8.0, 10.0, 12.0, 15.0]
    );
    let upstream = upload_f32(&backend, &authority, &[2, 2], &[1.0; 4], &cancellation)?;
    let gradients =
        mul_vjp_with_context_exact_native(&backend, &left, &right, &upstream, &execution)?;
    assert_eq!(
        values(&backend, &authority, &gradients.left, &cancellation)?,
        [9.0, 9.0]
    );
    assert_eq!(
        values(&backend, &authority, &gradients.right, &cancellation)?,
        [5.0, 5.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &mul_jvp_with_context_exact_native(&backend, &left, &right, &left, &right, &execution,)?,
            &cancellation
        )?,
        [16.0, 20.0, 24.0, 30.0]
    );

    let base = upload_f32(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let exponent = upload_f32(&backend, &authority, &[2], &[3.0, 2.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &pow_with_context_exact_native(&backend, &base, &exponent, &execution)?,
            &cancellation
        )?,
        [8.0, 9.0]
    );
    let ones = upload_f32(&backend, &authority, &[2], &[1.0, 1.0], &cancellation)?;
    let pow_gradients =
        pow_vjp_with_context_exact_native(&backend, &base, &exponent, &ones, &execution)?;
    assert_eq!(
        values(&backend, &authority, &pow_gradients.left, &cancellation)?,
        [12.0, 6.0]
    );
    let jvp = values(
        &backend,
        &authority,
        &pow_jvp_with_context_exact_native(&backend, &base, &exponent, &ones, &ones, &execution)?,
        &cancellation,
    )?;
    for (actual, expected) in jvp
        .iter()
        .zip([12.0 + 8.0 * 2.0_f32.ln(), 6.0 + 9.0 * 3.0_f32.ln()])
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn archive_optimizer_device_and_special_functions_are_native_and_deterministic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let tensor = upload_f32(&backend, &authority, &[2], &[1.5, -2.0], &cancellation)?;
    let mut map = BTreeMap::new();
    map.insert("weight".to_owned(), TorchArchiveValue::Tensor(tensor));
    map.insert("step".to_owned(), TorchArchiveValue::Integer(7));
    let loader = FixedArchiveLoader(TorchArchiveValue::Map(map));
    let loaded =
        torch_load_with_context_exact_native(&loader, &backend, DeviceId::CPU, true, &execution)?;
    let TorchArchiveValue::Map(loaded) = loaded else {
        return Err("loaded archive is not a map".into());
    };
    assert!(matches!(
        loaded.get("step"),
        Some(TorchArchiveValue::Integer(7))
    ));
    let Some(TorchArchiveValue::Tensor(loaded_tensor)) = loaded.get("weight") else {
        return Err("loaded archive tensor is missing".into());
    };
    assert_eq!(
        values(&backend, &authority, loaded_tensor, &cancellation)?,
        [1.5, -2.0]
    );
    assert!(
        torch_load_with_context_exact_native(&loader, &backend, DeviceId::CPU, false, &execution,)
            .is_err()
    );

    let mut parameters = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[1.0],
        &cancellation,
    )?];
    let gradients = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[0.5],
        &cancellation,
    )?];
    let mut optimizer = NativeAdamW::new_with_context_exact_native(
        &backend,
        &parameters,
        0.1,
        0.9,
        0.999,
        1e-8,
        0.01,
        false,
        false,
        &execution,
    )?;
    optimizer.step_with_context_exact_native(&backend, &mut parameters, &gradients, &execution)?;
    assert_eq!(optimizer.steps(), [1]);
    assert!((values(&backend, &authority, &parameters[0], &cancellation)?[0] - 0.899).abs() < 1e-5);

    let npu =
        BackendCapabilityMatrix::new(DeviceId::new(DeviceKind::Npu, 4), Vec::new(), Vec::new())?;
    assert_eq!(npu_current_device_exact_native(&npu, &cancellation)?, 4);
    assert!(
        npu_current_device_exact_native(&CpuBackend::capability_matrix(), &cancellation).is_err()
    );

    let probabilities = upload_f32(
        &backend,
        &authority,
        &[3],
        &[0.25, 0.5, 0.75],
        &cancellation,
    )?;
    let quantiles = values(
        &backend,
        &authority,
        &ndtri_with_context_exact_native(&backend, &probabilities, &execution)?,
        &cancellation,
    )?;
    assert!((quantiles[0] + 0.674_489_74).abs() < 1e-6);
    assert_eq!(quantiles[1], 0.0);
    assert!((quantiles[2] - 0.674_489_74).abs() < 1e-6);
    let tangent = upload_f32(&backend, &authority, &[3], &[1.0; 3], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &ndtri_vjp_with_context_exact_native(&backend, &probabilities, &tangent, &execution,)?,
            &cancellation,
        )?,
        values(
            &backend,
            &authority,
            &ndtri_jvp_with_context_exact_native(&backend, &probabilities, &tangent, &execution,)?,
            &cancellation,
        )?
    );

    let logits = upload_f32(&backend, &authority, &[3], &[-1.0, 0.0, 1.0], &cancellation)?;
    let expit = values(
        &backend,
        &authority,
        &expit_with_context_exact_native(&backend, &logits, &execution)?,
        &cancellation,
    )?;
    assert!((expit[1] - 0.5).abs() < f32::EPSILON);
    assert_eq!(
        values(
            &backend,
            &authority,
            &expit_vjp_with_context_exact_native(&backend, &logits, &tangent, &execution)?,
            &cancellation,
        )?,
        values(
            &backend,
            &authority,
            &expit_jvp_with_context_exact_native(&backend, &logits, &tangent, &execution)?,
            &cancellation,
        )?
    );
    Ok(())
}

#[test]
fn resolution_contracts_are_unique_and_sealed_by_their_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let owner =
        "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-615251b481b7";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_09")
        .ok_or("elementwise/runtime part-nine resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    let ids = slice
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(ids, IDS.into_iter().collect());
    let mut overloads = BTreeSet::new();
    for contract in slice.iter() {
        assert_eq!(contract.owner_task_id, owner);
        assert!(overloads.insert(contract.overload_id));
        let fixture = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(contract.evidence_fixture),
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&fixture)),
            contract.evidence_fixture_sha256
        );
        let document: serde_json::Value = serde_json::from_slice(&fixture)?;
        assert_eq!(document["operation_id"], contract.operation_id);
        assert_eq!(document["overload_id"], contract.overload_id);
        assert_eq!(document["owner_task_id"], owner);
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-6A39A98E4F68" => "tensor_constructor_with_context_exact_native",
            "COMFY-TENSOR-OP-616CB031A549" => "bitwise_xor_with_context_exact_native",
            "COMFY-TENSOR-OP-67B5EDD39C41" => "clamp_with_context_exact_native",
            "COMFY-TENSOR-OP-6A4C6EDFC695" => "frombuffer_with_context_exact_native",
            "COMFY-TENSOR-OP-6664BEC3F5BD" => "full_like_with_context_exact_native",
            "COMFY-TENSOR-OP-65EF512E0143" => "torch_load_with_context_exact_native",
            "COMFY-TENSOR-OP-615251B481B7" => "mul_with_context_exact_native",
            "COMFY-TENSOR-OP-6311D94BE18A" => "npu_current_device_exact_native",
            "COMFY-TENSOR-OP-6238000D28B1" => "NativeAdamW::new_with_context_exact_native",
            "COMFY-TENSOR-OP-67F181B603A1" => "pow_with_context_exact_native",
            "COMFY-TENSOR-OP-69E05601EAEA" => "expit_with_context_exact_native",
            "COMFY-TENSOR-OP-6520A75955CD" => "ndtri_with_context_exact_native",
            _ => return Err("unexpected Task 52 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn every_local_task52_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[1], &[1.0], &live)?;
    let input_bytes = input.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );

    macro_rules! assert_cancelled {
        ($expression:expr) => {
            assert!(matches!(
                $expression,
                Err(ElementwiseRuntimePartNineError::Cancelled)
            ));
        };
    }

    assert_cancelled!(tensor_constructor_with_context_exact_native(
        &backend,
        &[],
        &[1],
        DType::F32,
        StreamId::DEFAULT,
        &execution,
    ));
    assert_cancelled!(bitwise_xor_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(bitwise_binary_with_context_exact_native(
        &backend,
        &input,
        &input,
        NativeBitwiseOperation::Xor,
        "cancelled-test",
        &execution,
    ));
    assert_cancelled!(clamp_with_context_exact_native(
        &backend, &input, None, None, &execution
    ));
    assert_cancelled!(clamp_vjp_with_context_exact_native(
        &backend, &input, None, None, &input, &execution
    ));
    assert_cancelled!(clamp_jvp_with_context_exact_native(
        &backend, &input, None, None, &input, &execution
    ));
    assert_cancelled!(frombuffer_with_context_exact_native(
        &backend,
        &[],
        DType::F32,
        None,
        1,
        StreamId::DEFAULT,
        &execution,
    ));
    assert_cancelled!(full_like_with_context_exact_native(
        &backend,
        &input,
        Scalar::Signed(-1),
        Some(DType::U8),
        &execution,
    ));
    let loader = FixedArchiveLoader(TorchArchiveValue::Integer(1));
    assert_cancelled!(torch_load_with_context_exact_native(
        &loader,
        &backend,
        DeviceId::new(DeviceKind::Cuda, 0),
        false,
        &execution,
    ));
    assert_cancelled!(mul_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(mul_vjp_with_context_exact_native(
        &backend, &input, &input, &input, &execution
    ));
    assert_cancelled!(mul_jvp_with_context_exact_native(
        &backend, &input, &input, &input, &input, &execution
    ));
    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert_cancelled!(npu_current_device_exact_native(&cpu, &cancelled));
    assert_cancelled!(NativeAdamW::new_with_context_exact_native(
        &backend,
        &[],
        f32::NAN,
        1.0,
        1.0,
        -1.0,
        -1.0,
        false,
        false,
        &execution,
    ));
    let live_execution = context(&backend, &authority, &live)?;
    let mut optimizer = NativeAdamW::new_with_context_exact_native(
        &backend,
        std::slice::from_ref(&input),
        0.1,
        0.9,
        0.999,
        1e-8,
        0.0,
        false,
        false,
        &live_execution,
    )?;
    let mut parameters = vec![input.clone()];
    assert_cancelled!(optimizer.step_with_context_exact_native(
        &backend,
        &mut parameters,
        &[],
        &execution,
    ));
    assert_eq!(optimizer.steps(), [0]);
    assert_cancelled!(pow_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(pow_vjp_with_context_exact_native(
        &backend, &input, &input, &input, &execution
    ));
    assert_cancelled!(pow_jvp_with_context_exact_native(
        &backend, &input, &input, &input, &input, &execution
    ));
    assert_cancelled!(expit_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(expit_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(expit_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(ndtri_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(ndtri_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(ndtri_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));

    assert_eq!(execution.scratch.peak_bytes(), 0);
    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(parameters[0].contiguous_bytes()?, input_bytes);
    assert_eq!(input.contiguous_bytes()?, input_bytes);
    Ok(())
}
