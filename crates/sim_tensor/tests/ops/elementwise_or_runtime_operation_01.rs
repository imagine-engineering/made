use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, NativeDeviceProperties, OperationSupport,
    StreamId, Tensor, TensorDescriptor, UnaryOperation,
    generated_elementwise_or_runtime_operation_01::{
        ElementwiseRuntimeError, MeshgridIndexing, abs_jvp_with_context_exact_native,
        abs_vjp_with_context_exact_native, abs_with_context_exact_native,
        cuda_get_device_properties_exact_native, device_exact_native, dim_exact_native,
        finfo_exact_native, meshgrid_jvp_with_context_exact_native,
        meshgrid_vjp_with_context_exact_native, meshgrid_with_context_exact_native,
        signbit_with_context_exact_native, subtract_in_place_with_context_exact_native,
        subtract_jvp_with_context_exact_native, subtract_vjp_with_context_exact_native,
        triu_jvp_with_context_exact_native, triu_vjp_with_context_exact_native,
        triu_with_context_exact_native, vander_jvp_with_context_exact_native,
        vander_vjp_with_context_exact_native, vander_with_context_exact_native,
        xpu_current_device_exact_native, xpu_is_bf16_supported_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-010917B0D872",
    "COMFY-TENSOR-OP-01475E433DB3",
    "COMFY-TENSOR-OP-015751FC6965",
    "COMFY-TENSOR-OP-04A23E8A6156",
    "COMFY-TENSOR-OP-0546BABACDB9",
    "COMFY-TENSOR-OP-0678D863EEBA",
    "COMFY-TENSOR-OP-07B99A0B13EF",
    "COMFY-TENSOR-OP-0A5DBFB907FD",
    "COMFY-TENSOR-OP-0B05AB07BE66",
    "COMFY-TENSOR-OP-0B36DDFC0CD6",
    "COMFY-TENSOR-OP-0BDEE629B8C6",
    "COMFY-TENSOR-OP-0DB870AE36B5",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

#[test]
fn workspace_authorization_is_exact_bounded_and_convergent_for_part_one()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(
        &backend,
        &authority,
        &[4],
        &[-1.0, 0.0, 2.0, -0.0],
        &cancellation,
    )?;

    let authorization = authority.authorize_workspace(4)?;
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    signbit_with_context_exact_native(&backend, &input, &context)?;
    assert_eq!(authorization.peak_bytes(), 4);
    assert_eq!(authorization.in_use_bytes(), 0);

    let insufficient = authority.authorize_workspace(3)?;
    let context = backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(signbit_with_context_exact_native(&backend, &input, &context).is_err());
    assert_eq!(insufficient.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let authorization = authority.authorize_workspace(4)?;
    let context = backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancelled);
    assert!(signbit_with_context_exact_native(&backend, &input, &context).is_err());
    assert_eq!(authorization.in_use_bytes(), 0);
    Ok(())
}

fn tensor(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_f32(descriptor, values, &context)?.0)
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

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_01")
        .ok_or("elementwise/runtime part-one resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    for contract in slice.iter() {
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-010917b0d872"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-0B05AB07BE66" => "abs_with_context_exact_native",
            "COMFY-TENSOR-OP-0546BABACDB9" => "cuda_get_device_properties_exact_native",
            "COMFY-TENSOR-OP-0B36DDFC0CD6" => "device_exact_native",
            "COMFY-TENSOR-OP-0DB870AE36B5" => "dim_exact_native",
            "COMFY-TENSOR-OP-0A5DBFB907FD" => "finfo_exact_native",
            "COMFY-TENSOR-OP-07B99A0B13EF" => "meshgrid_with_context_exact_native",
            "COMFY-TENSOR-OP-015751FC6965" => "signbit_with_context_exact_native",
            "COMFY-TENSOR-OP-0BDEE629B8C6" => "subtract_in_place_with_context_exact_native",
            "COMFY-TENSOR-OP-0678D863EEBA" => "triu_with_context_exact_native",
            "COMFY-TENSOR-OP-010917B0D872" => "vander_with_context_exact_native",
            "COMFY-TENSOR-OP-01475E433DB3" => "xpu_current_device_exact_native",
            "COMFY-TENSOR-OP-04A23E8A6156" => "xpu_is_bf16_supported_exact_native",
            _ => return Err("unexpected Task 44 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn absolute_rank_and_subtraction_reuse_canonical_primitives_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(
        &backend,
        &authority,
        &[2, 2],
        &[-2.0, -0.0, 3.0, 0.0],
        &cancellation,
    )?;
    assert_eq!(dim_exact_native(&input, &cancellation)?, 2);
    let scalar = tensor(&backend, &authority, &[], &[4.0], &cancellation)?;
    assert_eq!(dim_exact_native(&scalar, &cancellation)?, 0);
    let absolute = abs_with_context_exact_native(
        &backend,
        &input,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &absolute, &cancellation)?,
        [2.0, 0.0, 3.0, 0.0]
    );

    let gradient = tensor(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let vjp = abs_vjp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let jvp = abs_jvp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp, &cancellation)?,
        [-1.0, 0.0, 3.0, 0.0]
    );
    assert_eq!(
        values(&backend, &authority, &jvp, &cancellation)?,
        [-1.0, 0.0, 3.0, 0.0]
    );

    let mut minuend = tensor(&backend, &authority, &[2], &[4.0, 5.0], &cancellation)?;
    let alias = minuend.clone();
    let subtrahend = tensor(&backend, &authority, &[2], &[4.0, 8.0], &cancellation)?;
    subtract_in_place_with_context_exact_native(
        &backend,
        &mut minuend,
        &subtrahend,
        0.25,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &minuend, &cancellation)?,
        [3.0, 3.0]
    );
    assert_eq!(
        values(&backend, &authority, &alias, &cancellation)?,
        [4.0, 5.0]
    );

    let output_gradient = tensor(&backend, &authority, &[2], &[2.0, 4.0], &cancellation)?;
    let vjp = subtract_vjp_with_context_exact_native(
        &backend,
        &output_gradient,
        0.25,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp.input, &cancellation)?,
        [2.0, 4.0]
    );
    assert_eq!(
        values(&backend, &authority, &vjp.other, &cancellation)?,
        [-0.5, -1.0]
    );
    let jvp = subtract_jvp_with_context_exact_native(
        &backend,
        &output_gradient,
        &subtrahend,
        0.25,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &jvp, &cancellation)?,
        [1.0, 2.0]
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let before = values(&backend, &authority, &minuend, &cancellation)?;
    assert!(
        subtract_in_place_with_context_exact_native(
            &backend,
            &mut minuend,
            &subtrahend,
            1.0,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancelled,
            ),
        )
        .is_err()
    );
    assert_eq!(
        values(&backend, &authority, &minuend, &cancellation)?,
        before
    );
    Ok(())
}

#[test]
fn dtype_and_device_adapters_delegate_to_their_canonical_domains()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    assert_eq!(device_exact_native("cpu", &cancellation)?, DeviceId::CPU);
    assert_eq!(
        device_exact_native("mps:2", &cancellation)?,
        DeviceId::new(DeviceKind::Metal, 2)
    );
    assert!(device_exact_native("cuda:-1", &cancellation).is_err());

    let info = finfo_exact_native(DType::F16, &cancellation)?;
    assert_eq!(info.bits(), 16);
    assert_eq!(info.epsilon(), 0.000_976_562_5);
    assert_eq!(info.maximum(), 65_504.0);
    assert!(finfo_exact_native(DType::I32, &cancellation).is_err());

    let xpu = DeviceId::new(DeviceKind::Xpu, 2);
    let support = OperationSupport::allocation(DType::Bf16, Layout::Contiguous);
    let xpu_capabilities = BackendCapabilityMatrix::new(xpu, vec![support], vec![support])?;
    assert_eq!(
        xpu_current_device_exact_native(&xpu_capabilities, &cancellation)?,
        2
    );
    assert!(xpu_is_bf16_supported_exact_native(
        &xpu_capabilities,
        &cancellation
    )?);

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let properties = NativeDeviceProperties::new(
        cuda,
        "fixture cuda",
        8 * 1024 * 1024 * 1024,
        8,
        9,
        Some("sm_89".to_owned()),
        true,
    )?;
    let cuda_capabilities = BackendCapabilityMatrix::new_with_properties(
        cuda,
        vec![OperationSupport::unary_input(
            UnaryOperation::Absolute,
            DType::F32,
            Layout::Contiguous,
        )],
        Vec::new(),
        Some(properties),
    )?;
    let properties =
        cuda_get_device_properties_exact_native(&cuda_capabilities, cuda, &cancellation)?;
    assert_eq!(properties.name(), "fixture cuda");
    assert_eq!(properties.major(), 8);
    assert_eq!(properties.minor(), 9);
    assert_eq!(properties.architecture(), Some("sm_89"));
    assert!(properties.has_fp16());

    let cpu_capabilities = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(xpu_current_device_exact_native(&cpu_capabilities, &cancellation).is_err());
    assert!(
        cuda_get_device_properties_exact_native(&cpu_capabilities, DeviceId::CPU, &cancellation,)
            .is_err()
    );
    Ok(())
}

#[test]
fn meshgrid_and_signbit_preserve_indexing_signed_zero_and_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let x = tensor(&backend, &authority, &[2], &[1.0, 2.0], &cancellation)?;
    let y = tensor(
        &backend,
        &authority,
        &[3],
        &[10.0, 20.0, 30.0],
        &cancellation,
    )?;
    let outputs = meshgrid_with_context_exact_native(
        &backend,
        &[x.clone(), y.clone()],
        MeshgridIndexing::Xy,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(outputs[0].descriptor().shape(), [3, 2]);
    assert_eq!(
        values(&backend, &authority, &outputs[0], &cancellation)?,
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
    assert_eq!(
        values(&backend, &authority, &outputs[1], &cancellation)?,
        [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
    );
    let ones = tensor(&backend, &authority, &[3, 2], &[1.0; 6], &cancellation)?;
    let input_gradients = meshgrid_vjp_with_context_exact_native(
        &backend,
        &[x, y],
        &[ones.clone(), ones],
        MeshgridIndexing::Xy,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &input_gradients[0], &cancellation)?,
        [3.0, 3.0]
    );
    assert_eq!(
        values(&backend, &authority, &input_gradients[1], &cancellation)?,
        [2.0, 2.0, 2.0]
    );
    let x_tangent = tensor(&backend, &authority, &[2], &[0.5, 1.0], &cancellation)?;
    let y_tangent = tensor(&backend, &authority, &[3], &[2.0, 3.0, 4.0], &cancellation)?;
    let output_tangents = meshgrid_jvp_with_context_exact_native(
        &backend,
        &[x_tangent, y_tangent],
        MeshgridIndexing::Xy,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &output_tangents[0], &cancellation)?,
        [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    );
    assert_eq!(
        values(&backend, &authority, &output_tangents[1], &cancellation)?,
        [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );

    let signs = tensor(
        &backend,
        &authority,
        &[4],
        &[-0.0, 0.0, -2.0, f32::from_bits(0xffc0_0000)],
        &cancellation,
    )?;
    let sign_bits = signbit_with_context_exact_native(
        &backend,
        &signs,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(sign_bits.contiguous_bytes()?, [1, 0, 1, 1]);
    Ok(())
}

#[test]
fn triangular_and_vandermonde_kernels_share_forward_and_gradient_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let matrix = tensor(
        &backend,
        &authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let upper = triu_with_context_exact_native(
        &backend,
        &matrix,
        1,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &upper, &cancellation)?,
        [0.0, 2.0, 3.0, 0.0, 0.0, 6.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &triu_vjp_with_context_exact_native(
                &backend,
                &matrix,
                1,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation,
        )?,
        [0.0, 2.0, 3.0, 0.0, 0.0, 6.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &triu_jvp_with_context_exact_native(
                &backend,
                &matrix,
                1,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation,
        )?,
        [0.0, 2.0, 3.0, 0.0, 0.0, 6.0]
    );

    let input = tensor(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let vander = vander_with_context_exact_native(
        &backend,
        &input,
        Some(3),
        false,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &vander, &cancellation)?,
        [4.0, 2.0, 1.0, 9.0, 3.0, 1.0]
    );
    let output_gradient = tensor(&backend, &authority, &[2, 3], &[1.0; 6], &cancellation)?;
    let vjp = vander_vjp_with_context_exact_native(
        &backend,
        &input,
        &output_gradient,
        Some(3),
        false,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp, &cancellation)?,
        [5.0, 7.0]
    );
    let tangent = tensor(&backend, &authority, &[2], &[0.5, 2.0], &cancellation)?;
    let jvp = vander_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        Some(3),
        false,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &jvp, &cancellation)?,
        [2.0, 0.5, 0.0, 12.0, 2.0, 0.0]
    );
    let increasing = vander_with_context_exact_native(
        &backend,
        &input,
        Some(3),
        true,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &increasing, &cancellation)?,
        [1.0, 2.0, 4.0, 1.0, 3.0, 9.0]
    );
    let empty = vander_with_context_exact_native(
        &backend,
        &input,
        Some(0),
        false,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        ),
    )?;
    assert_eq!(empty.descriptor().shape(), [2, 0]);
    assert!(empty.contiguous_bytes()?.is_empty());
    Ok(())
}

#[test]
fn every_task44_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let input = tensor(&backend, &authority, &[2], &[1.0, -2.0], &live)?;
    let matrix = tensor(&backend, &authority, &[1, 2], &[1.0, 2.0], &live)?;
    let mut minuend = input.clone();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        abs_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        dim_exact_native(&input, &cancelled),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        subtract_in_place_with_context_exact_native(
            &backend,
            &mut minuend,
            &input,
            f32::NAN,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert_eq!(minuend.storage_id(), input.storage_id());
    assert!(matches!(
        device_exact_native("invalid::device", &cancelled),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        finfo_exact_native(DType::I32, &cancelled),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        meshgrid_with_context_exact_native(
            &backend,
            std::slice::from_ref(&matrix),
            MeshgridIndexing::Ij,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        signbit_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        triu_with_context_exact_native(&backend, &input, 0, &cancelled_context),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        vander_with_context_exact_native(&backend, &matrix, None, false, &cancelled_context,),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    let cpu_capabilities = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        cuda_get_device_properties_exact_native(&cpu_capabilities, DeviceId::CPU, &cancelled,),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        xpu_current_device_exact_native(&cpu_capabilities, &cancelled),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert!(matches!(
        xpu_is_bf16_supported_exact_native(&cpu_capabilities, &cancelled),
        Err(ElementwiseRuntimeError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}
