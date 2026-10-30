use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    ALL_DTYPES, BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DecodedScalar,
    DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    NativeDeviceProperties, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_07::{
        ElementwiseRuntimePartSevenError, NativeSgd, TensorPrintOptions, TensorPrintOptionsUpdate,
        addcdiv_jvp_with_context_exact_native, addcdiv_vjp_with_context_exact_native,
        addcdiv_with_context_exact_native, argwhere_with_context_exact_native,
        expm1_method_jvp_with_context_exact_native, expm1_method_vjp_with_context_exact_native,
        expm1_method_with_context_exact_native, log1p_jvp_with_context_exact_native,
        log1p_vjp_with_context_exact_native, log1p_with_context_exact_native,
        outer_jvp_with_context_exact_native, outer_vjp_with_context_exact_native,
        outer_with_context_exact_native, rsqrt_jvp_with_context_exact_native,
        rsqrt_vjp_with_context_exact_native, rsqrt_with_context_exact_native,
        set_printoptions_exact_native, tanh_function_jvp_with_context_exact_native,
        tanh_function_vjp_with_context_exact_native, tanh_function_with_context_exact_native,
        xpu_current_stream_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-5A1598AB1BFB",
    "COMFY-TENSOR-OP-5668EBF27561",
    "COMFY-TENSOR-OP-59C70700F28E",
    "COMFY-TENSOR-OP-56E8CFEB8E84",
    "COMFY-TENSOR-OP-58AE3CA27BFE",
    "COMFY-TENSOR-OP-594BD684E5EF",
    "COMFY-TENSOR-OP-59EBFDE56C4F",
    "COMFY-TENSOR-OP-54E28780B32B",
    "COMFY-TENSOR-OP-5547BE508AEE",
    "COMFY-TENSOR-OP-59AD8FFF431A",
    "COMFY-TENSOR-OP-576587FE2EAF",
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

fn decoded_flat(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut decoded = Vec::with_capacity(count);
    for linear in 0..count {
        let mut remainder = linear;
        let mut indices = vec![0; tensor.descriptor().rank()];
        for (index, dimension) in indices.iter_mut().zip(tensor.descriptor().shape()).rev() {
            let dimension = usize::try_from(*dimension)?;
            *index = u64::try_from(remainder % dimension)?;
            remainder /= dimension;
        }
        decoded.push(
            tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.element_bytes(&indices)?)?,
        );
    }
    Ok(decoded)
}

fn upload_zero_bits(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(vec![1], dtype, DeviceId::CPU, StreamId::DEFAULT)?;
    let (mut tensor, _) =
        backend.allocate(descriptor, &context(backend, authority, cancellation)?)?;
    tensor.write()?.bytes_mut()?.fill(0);
    Ok(tensor)
}

#[test]
fn unary_adapters_preserve_values_and_gradients() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let unary_input = upload_f32(&backend, &authority, &[3], &[-0.5, 0.0, 3.0], &cancellation)?;
    let expm1 = values(
        &backend,
        &authority,
        &expm1_method_with_context_exact_native(&backend, &unary_input, &execution)?,
        &cancellation,
    )?;
    assert!((expm1[0] - (-0.5_f32).exp_m1()).abs() < 1e-6);
    let ones = upload_f32(&backend, &authority, &[3], &[1.0; 3], &cancellation)?;
    let expm1_vjp = values(
        &backend,
        &authority,
        &expm1_method_vjp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
        &cancellation,
    )?;
    let expm1_jvp = values(
        &backend,
        &authority,
        &expm1_method_jvp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
        &cancellation,
    )?;
    for (index, input_value) in [-0.5_f32, 0.0, 3.0].into_iter().enumerate() {
        assert!((expm1_vjp[index] - input_value.exp()).abs() < 1e-5);
        assert!((expm1_jvp[index] - input_value.exp()).abs() < 1e-5);
    }
    let log1p = values(
        &backend,
        &authority,
        &log1p_with_context_exact_native(&backend, &unary_input, &execution)?,
        &cancellation,
    )?;
    assert!((log1p[0] - (-0.5_f32).ln_1p()).abs() < 1e-6);
    assert_eq!(
        values(
            &backend,
            &authority,
            &log1p_vjp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
            &cancellation,
        )?,
        [2.0, 1.0, 0.25]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &log1p_jvp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
            &cancellation,
        )?,
        [2.0, 1.0, 0.25]
    );
    let positive = upload_f32(&backend, &authority, &[3], &[0.25, 1.0, 4.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &rsqrt_with_context_exact_native(&backend, &positive, &execution)?,
            &cancellation,
        )?,
        [2.0, 1.0, 0.5]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &rsqrt_vjp_with_context_exact_native(&backend, &positive, &ones, &execution)?,
            &cancellation,
        )?,
        [-4.0, -0.5, -0.0625]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &rsqrt_jvp_with_context_exact_native(&backend, &positive, &ones, &execution)?,
            &cancellation,
        )?,
        [-4.0, -0.5, -0.0625]
    );
    let tanh = values(
        &backend,
        &authority,
        &tanh_function_with_context_exact_native(&backend, &unary_input, &execution)?,
        &cancellation,
    )?;
    assert!((tanh[2] - 3.0_f32.tanh()).abs() < 1e-6);
    let tanh_vjp = values(
        &backend,
        &authority,
        &tanh_function_vjp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
        &cancellation,
    )?;
    let tanh_jvp = values(
        &backend,
        &authority,
        &tanh_function_jvp_with_context_exact_native(&backend, &unary_input, &ones, &execution)?,
        &cancellation,
    )?;
    for (index, output) in tanh.iter().enumerate() {
        assert!((tanh_vjp[index] - (1.0 - output * output)).abs() < 1e-6);
        assert!((tanh_jvp[index] - tanh_vjp[index]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn workspace_authority_is_exact_bounded_and_converges_for_addcdiv()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[4], &[1.0; 4], &cancellation)?;
    let tensor1 = upload_f32(&backend, &authority, &[4], &[2.0; 4], &cancellation)?;
    let tensor2 = upload_f32(&backend, &authority, &[4], &[4.0; 4], &cancellation)?;
    let tangent = upload_f32(&backend, &authority, &[4], &[1.0; 4], &cancellation)?;
    let zero = upload_f32(&backend, &authority, &[4], &[0.0; 4], &cancellation)?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancellation,
    );
    addcdiv_jvp_with_context_exact_native(
        &backend, &input, &tensor1, &tensor2, 1.0, &tangent, &zero, &zero, &exact,
    )?;
    assert_eq!(exact.scratch.peak_bytes(), 16);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    assert!(addcdiv_with_context_exact_native(
        &backend,
        &input,
        &tensor1,
        &tensor2,
        1.0,
        &insufficient,
    )
    .is_err());
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(32)?,
        &cancelled,
    );
    assert!(argwhere_with_context_exact_native(&backend, &input, &cancelled_context).is_err());
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn addcdiv_argwhere_and_outer_cover_broadcast_and_reverse_rules()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = upload_f32(&backend, &authority, &[2, 1], &[1.0, 2.0], &cancellation)?;
    let tensor1 = upload_f32(&backend, &authority, &[1, 2], &[4.0, 8.0], &cancellation)?;
    let tensor2 = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[2.0, 4.0, 1.0, 2.0],
        &cancellation,
    )?;
    let output =
        addcdiv_with_context_exact_native(&backend, &input, &tensor1, &tensor2, 0.5, &execution)?;
    assert_eq!(
        values(&backend, &authority, &output, &cancellation)?,
        [2.0, 2.0, 4.0, 4.0]
    );
    let upstream = upload_f32(&backend, &authority, &[2, 2], &[1.0; 4], &cancellation)?;
    let gradients = addcdiv_vjp_with_context_exact_native(
        &backend, &input, &tensor1, &tensor2, 0.5, &upstream, &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &gradients.input, &cancellation)?,
        [2.0, 2.0]
    );
    assert_eq!(
        values(&backend, &authority, &gradients.tensor1, &cancellation)?,
        [0.75, 0.375]
    );
    let input_tangent = upload_f32(&backend, &authority, &[2, 1], &[1.0, 2.0], &cancellation)?;
    let tensor1_tangent = upload_f32(&backend, &authority, &[1, 2], &[1.0, 1.0], &cancellation)?;
    let tensor2_tangent = upload_f32(&backend, &authority, &[2, 2], &[0.0; 4], &cancellation)?;
    let tangent = addcdiv_jvp_with_context_exact_native(
        &backend,
        &input,
        &tensor1,
        &tensor2,
        0.5,
        &input_tangent,
        &tensor1_tangent,
        &tensor2_tangent,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &tangent, &cancellation)?,
        [1.25, 1.125, 2.5, 2.25]
    );
    assert!(
        values(
            &backend,
            &authority,
            &addcdiv_with_context_exact_native(
                &backend,
                &input,
                &tensor1,
                &tensor2,
                f32::INFINITY,
                &execution,
            )?,
            &cancellation,
        )?
        .into_iter()
        .all(f32::is_infinite)
    );

    let locations = argwhere_with_context_exact_native(&backend, &output, &execution)?;
    assert_eq!(locations.descriptor().shape(), [4, 2]);
    assert_eq!(
        decoded_flat(&locations)?,
        [
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(1),
        ]
    );
    for dtype in ALL_DTYPES {
        let input = upload_zero_bits(&backend, &authority, dtype, &cancellation)?;
        let decoded = decoded_flat(&input)?;
        let decoded = *decoded
            .first()
            .ok_or("single-element tensor decoded empty")?;
        let expected_rows = usize::from(match decoded {
            DecodedScalar::Boolean(value) => value,
            DecodedScalar::Signed(value) => value != 0,
            DecodedScalar::Unsigned(value) => value != 0,
            DecodedScalar::Real(value) => value != 0.0,
            DecodedScalar::Complex { real, imaginary } => real != 0.0 || imaginary != 0.0,
        });
        let locations = argwhere_with_context_exact_native(&backend, &input, &execution)?;
        assert_eq!(
            locations.descriptor().shape(),
            [u64::try_from(expected_rows)?, 1],
            "argwhere dtype {dtype:?}"
        );
    }
    let empty_input = upload_f32(&backend, &authority, &[0, 1], &[], &cancellation)?;
    let singleton = upload_f32(&backend, &authority, &[1, 1], &[1.0], &cancellation)?;
    let empty_divisor = upload_f32(&backend, &authority, &[0, 1], &[], &cancellation)?;
    assert_eq!(
        addcdiv_with_context_exact_native(
            &backend,
            &empty_input,
            &singleton,
            &empty_divisor,
            1.0,
            &execution,
        )?
        .descriptor()
        .shape(),
        [0, 1]
    );

    let left = upload_f32(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let right = upload_f32(&backend, &authority, &[3], &[4.0, 5.0, 6.0], &cancellation)?;
    let outer = outer_with_context_exact_native(&backend, &left, &right, &execution)?;
    assert_eq!(
        values(&backend, &authority, &outer, &cancellation)?,
        [8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
    );
    let gradient = upload_f32(&backend, &authority, &[2, 3], &[1.0; 6], &cancellation)?;
    let outer_gradients =
        outer_vjp_with_context_exact_native(&backend, &left, &right, &gradient, &execution)?;
    assert_eq!(
        values(&backend, &authority, &outer_gradients.input, &cancellation)?,
        [15.0, 15.0]
    );
    assert_eq!(
        values(&backend, &authority, &outer_gradients.other, &cancellation)?,
        [5.0, 5.0, 5.0]
    );
    let left_tangent = upload_f32(&backend, &authority, &[2], &[1.0, 0.0], &cancellation)?;
    let right_tangent = upload_f32(&backend, &authority, &[3], &[0.0, 1.0, 0.0], &cancellation)?;
    let outer_tangent = outer_jvp_with_context_exact_native(
        &backend,
        &left,
        &right,
        &left_tangent,
        &right_tangent,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &outer_tangent, &cancellation)?,
        [4.0, 7.0, 6.0, 0.0, 3.0, 0.0]
    );
    let scalar = upload_f32(&backend, &authority, &[], &[2.0], &cancellation)?;
    assert!(
        outer_vjp_with_context_exact_native(&backend, &scalar, &right, &gradient, &execution)
            .is_err()
    );
    Ok(())
}

#[test]
fn sgd_print_options_and_xpu_stream_are_explicit_and_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let mut parameters = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[1.0, -1.0],
        &cancellation,
    )?];
    let gradients = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.5, -0.25],
        &cancellation,
    )?];
    let mut optimizer =
        NativeSgd::new_exact_native(1, 0.1, 0.9, 0.0, 0.0, false, false, &cancellation)?;
    optimizer.step_with_context_exact_native(&backend, &mut parameters, &gradients, &execution)?;
    assert_eq!(
        values(&backend, &authority, &parameters[0], &cancellation)?,
        [0.95, -0.975]
    );
    optimizer.step_with_context_exact_native(&backend, &mut parameters, &gradients, &execution)?;
    let second = values(&backend, &authority, &parameters[0], &cancellation)?;
    assert!((second[0] - 0.855).abs() < 1e-6);
    assert!((second[1] - (-0.9275)).abs() < 1e-6);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_execution = context(&backend, &authority, &cancelled)?;
    let before = parameters[0].contiguous_bytes()?.to_vec();
    assert!(
        optimizer
            .step_with_context_exact_native(
                &backend,
                &mut parameters,
                &gradients,
                &cancelled_execution,
            )
            .is_err()
    );
    assert_eq!(parameters[0].contiguous_bytes()?, before);
    let mut atomic_parameters = vec![
        upload_f32(&backend, &authority, &[1], &[1.0], &cancellation)?,
        upload_f32(&backend, &authority, &[1], &[2.0], &cancellation)?,
    ];
    let invalid_gradients = vec![
        upload_f32(&backend, &authority, &[1], &[0.5], &cancellation)?,
        upload_f32(&backend, &authority, &[2], &[0.5, 0.5], &cancellation)?,
    ];
    let atomic_before = atomic_parameters
        .iter()
        .map(|parameter| parameter.contiguous_bytes().map(<[u8]>::to_vec))
        .collect::<Result<Vec<_>, _>>()?;
    let mut atomic_optimizer =
        NativeSgd::new_exact_native(2, 0.1, 0.9, 0.0, 0.0, false, false, &cancellation)?;
    assert!(
        atomic_optimizer
            .step_with_context_exact_native(
                &backend,
                &mut atomic_parameters,
                &invalid_gradients,
                &execution,
            )
            .is_err()
    );
    assert_eq!(
        atomic_parameters
            .iter()
            .map(|parameter| parameter.contiguous_bytes().map(<[u8]>::to_vec))
            .collect::<Result<Vec<_>, _>>()?,
        atomic_before
    );

    let defaults = TensorPrintOptions::default();
    let preview = set_printoptions_exact_native(
        &defaults,
        Some(TensorPrintOptionsUpdate {
            edge_items: Some(6),
            ..TensorPrintOptionsUpdate::default()
        }),
        &cancellation,
    )?;
    assert_eq!(preview.edge_items, 6);
    assert_eq!(
        set_printoptions_exact_native(&preview, None, &cancellation)?,
        defaults
    );
    assert!(
        set_printoptions_exact_native(
            &defaults,
            Some(TensorPrintOptionsUpdate {
                threshold: Some(10_000_001),
                ..TensorPrintOptionsUpdate::default()
            }),
            &cancellation,
        )
        .is_err()
    );

    let xpu = DeviceId::new(DeviceKind::Xpu, 2);
    let properties = NativeDeviceProperties::new(xpu, "xpu", 1024, 1, 0, None, true)?;
    let matrix = BackendCapabilityMatrix::new_with_properties(
        xpu,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        xpu_current_stream_exact_native(
            &matrix,
            xpu,
            &backend.execution_context(
                StreamId::new(17),
                authority.authorize_workspace(0)?,
                &cancellation,
            ),
        )?,
        StreamId::new(17)
    );
    assert!(
        xpu_current_stream_exact_native(
            &matrix,
            DeviceId::CPU,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(0)?,
                &cancellation,
            ),
        )
        .is_err()
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        xpu_current_stream_exact_native(
            &matrix,
            xpu,
            &backend.execution_context(
                StreamId::new(17),
                authority.authorize_workspace(0)?,
                &cancelled,
            ),
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn resolution_slice_seals_exactly_task_50_contracts() -> Result<(), Box<dyn std::error::Error>> {
    let owner =
        "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-54e28780b32b";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_07")
        .ok_or("elementwise/runtime part-seven resolution slice is missing")?;
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
            "COMFY-TENSOR-OP-5A1598AB1BFB" => "expm1_method_with_context_exact_native",
            "COMFY-TENSOR-OP-5668EBF27561" => "addcdiv_with_context_exact_native",
            "COMFY-TENSOR-OP-59C70700F28E" => "argwhere_with_context_exact_native",
            "COMFY-TENSOR-OP-56E8CFEB8E84" => "log1p_with_context_exact_native",
            "COMFY-TENSOR-OP-58AE3CA27BFE" => "weight_norm_exact_native",
            "COMFY-TENSOR-OP-594BD684E5EF" => "NativeSgd::new_exact_native",
            "COMFY-TENSOR-OP-59EBFDE56C4F" => "outer_with_context_exact_native",
            "COMFY-TENSOR-OP-54E28780B32B" => "rsqrt_with_context_exact_native",
            "COMFY-TENSOR-OP-5547BE508AEE" => "set_printoptions_exact_native",
            "COMFY-TENSOR-OP-59AD8FFF431A" => "tanh_function_with_context_exact_native",
            "COMFY-TENSOR-OP-576587FE2EAF" => "xpu_current_stream_exact_native",
            _ => return Err("unexpected Task 50 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
        if contract.operation_id == "COMFY-TENSOR-OP-58AE3CA27BFE" {
            assert!(
                contract
                    .rust_signature
                    .contains("weight_norm_exact_native<'a>")
            );
            assert!(
                contract
                    .rust_signature
                    .contains("Result<&'a mut NativeModule")
            );
        }
    }
    Ok(())
}

#[test]
fn every_local_task50_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[1], &[1.0], &live)?;
    let input_bytes = input.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let execution = context(&backend, &authority, &cancelled)?;

    assert!(matches!(
        expm1_method_with_context_exact_native(&backend, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        expm1_method_vjp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        expm1_method_jvp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        addcdiv_with_context_exact_native(&backend, &input, &input, &input, 1.0, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        addcdiv_vjp_with_context_exact_native(
            &backend, &input, &input, &input, 1.0, &input, &execution
        ),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        addcdiv_jvp_with_context_exact_native(
            &backend, &input, &input, &input, 1.0, &input, &input, &input, &execution
        ),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        argwhere_with_context_exact_native(&backend, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        log1p_with_context_exact_native(&backend, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        log1p_vjp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        log1p_jvp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        outer_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        outer_vjp_with_context_exact_native(&backend, &input, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        outer_jvp_with_context_exact_native(&backend, &input, &input, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        rsqrt_with_context_exact_native(&backend, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        rsqrt_vjp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        rsqrt_jvp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        tanh_function_with_context_exact_native(&backend, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        tanh_function_vjp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        tanh_function_jvp_with_context_exact_native(&backend, &input, &input, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert!(matches!(
        NativeSgd::new_exact_native(0, f32::NAN, 0.0, 0.0, 0.0, false, false, &cancelled),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    let mut optimizer = NativeSgd::new_exact_native(1, 0.1, 0.0, 0.0, 0.0, false, false, &live)?;
    let mut parameters = vec![input.clone()];
    assert!(matches!(
        optimizer.step_with_context_exact_native(&backend, &mut parameters, &[], &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert_eq!(parameters[0].contiguous_bytes()?, input_bytes);
    assert!(matches!(
        set_printoptions_exact_native(
            &TensorPrintOptions::default(),
            Some(TensorPrintOptionsUpdate {
                line_width: Some(0),
                ..TensorPrintOptionsUpdate::default()
            }),
            &cancelled
        ),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    let cpu_matrix = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        xpu_current_stream_exact_native(&cpu_matrix, DeviceId::CPU, &execution),
        Err(ElementwiseRuntimePartSevenError::Cancelled)
    ));
    assert_eq!(execution.scratch.peak_bytes(), 0);
    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, input_bytes);
    Ok(())
}
