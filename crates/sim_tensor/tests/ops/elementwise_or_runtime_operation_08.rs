use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    AutogradError, AutogradInput, AutogradTape, BackendCapabilityMatrix, BackwardRule,
    CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradientMode, GradientReducer, LeafId,
    SavedTensor, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError, autograd_grad_exact_native,
        byte_method_with_context_exact_native, concatenate_jvp_with_context_exact_native,
        concatenate_vjp_with_context_exact_native, concatenate_with_context_exact_native,
        cos_jvp_with_context_exact_native, cos_vjp_with_context_exact_native,
        cos_with_context_exact_native, expm1_function_jvp_with_context_exact_native,
        expm1_function_vjp_with_context_exact_native, expm1_function_with_context_exact_native,
        index_select_jvp_with_context_exact_native, index_select_vjp_with_context_exact_native,
        index_select_with_context_exact_native, log_method_jvp_with_context_exact_native,
        log_method_vjp_with_context_exact_native, log_method_with_context_exact_native,
        median_jvp_with_context_exact_native, median_vjp_with_context_exact_native,
        median_with_context_exact_native, mlu_current_device_exact_native,
        mps_is_available_exact_native, rot90_jvp_with_context_exact_native,
        rot90_vjp_with_context_exact_native, rot90_with_context_exact_native,
        square_jvp_with_context_exact_native, square_vjp_with_context_exact_native,
        square_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path, sync::Arc};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-5DFBC70338A1",
    "COMFY-TENSOR-OP-5ED49DCA2F78",
    "COMFY-TENSOR-OP-5CEC7CF2D62D",
    "COMFY-TENSOR-OP-5BF1DE9DA499",
    "COMFY-TENSOR-OP-5C52F193416C",
    "COMFY-TENSOR-OP-5C00AB949613",
    "COMFY-TENSOR-OP-5D7C103AB024",
    "COMFY-TENSOR-OP-5AB4376A79B5",
    "COMFY-TENSOR-OP-5BA79209BB02",
    "COMFY-TENSOR-OP-5EECCD4F0130",
    "COMFY-TENSOR-OP-5CDFF9F97B6F",
    "COMFY-TENSOR-OP-60A72EC2F5DD",
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

fn upload_i64(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    let (mut tensor, _) =
        backend.allocate(descriptor, &context(backend, authority, cancellation)?)?;
    let mut write = tensor.write()?;
    for (chunk, value) in write.bytes_mut()?.chunks_exact_mut(8).zip(values) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
    drop(write);
    Ok(tensor)
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

#[test]
fn byte_and_unary_adapters_preserve_exact_forward_and_derivative_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let byte_input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[-1.8, 255.9, 256.1, 511.0],
        &cancellation,
    )?;
    assert_eq!(
        decoded_flat(&byte_method_with_context_exact_native(
            &backend,
            &byte_input,
            &execution,
        )?)?,
        [
            DecodedScalar::Unsigned(255),
            DecodedScalar::Unsigned(255),
            DecodedScalar::Unsigned(0),
            DecodedScalar::Unsigned(255),
        ]
    );

    let input = upload_f32(&backend, &authority, &[3], &[0.5, 1.0, 2.0], &cancellation)?;
    let upstream = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let logarithm = values(
        &backend,
        &authority,
        &log_method_with_context_exact_native(&backend, &input, &execution)?,
        &cancellation,
    )?;
    assert!((logarithm[0] - 0.5_f32.ln()).abs() < 1e-6);
    assert_eq!(
        values(
            &backend,
            &authority,
            &log_method_vjp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?,
        [2.0, 2.0, 1.5]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &log_method_jvp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?,
        [2.0, 2.0, 1.5]
    );

    let cosine = values(
        &backend,
        &authority,
        &cos_with_context_exact_native(&backend, &input, &execution)?,
        &cancellation,
    )?;
    for (actual, expected) in cosine
        .iter()
        .zip([0.5_f32.cos(), 1.0_f32.cos(), 2.0_f32.cos()])
    {
        assert!((*actual - expected).abs() < 1e-6);
    }
    let cosine_vjp = values(
        &backend,
        &authority,
        &cos_vjp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
        &cancellation,
    )?;
    let cosine_jvp = values(
        &backend,
        &authority,
        &cos_jvp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
        &cancellation,
    )?;
    assert_eq!(cosine_vjp, cosine_jvp);
    for (actual, expected) in
        cosine_vjp
            .iter()
            .zip([-0.5_f32.sin(), -2.0 * 1.0_f32.sin(), -3.0 * 2.0_f32.sin()])
    {
        assert!((*actual - expected).abs() < 1e-6);
    }

    let exponential = values(
        &backend,
        &authority,
        &expm1_function_with_context_exact_native(&backend, &input, &execution)?,
        &cancellation,
    )?;
    assert!((exponential[0] - 0.5_f32.exp_m1()).abs() < 1e-6);
    assert_eq!(
        values(
            &backend,
            &authority,
            &expm1_function_vjp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?,
        values(
            &backend,
            &authority,
            &expm1_function_jvp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &square_with_context_exact_native(&backend, &input, &execution)?,
            &cancellation,
        )?,
        [0.25, 1.0, 4.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &square_vjp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?,
        [1.0, 4.0, 12.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &square_jvp_with_context_exact_native(&backend, &input, &upstream, &execution)?,
            &cancellation,
        )?,
        [1.0, 4.0, 12.0]
    );
    Ok(())
}

#[test]
fn square_workspace_authority_is_exact_and_failure_atomic() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let baseline = backend.memory_snapshot().current_bytes;
    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(11)?,
        &cancellation,
    );
    assert!(square_with_context_exact_native(&backend, &input, &underauthorized).is_err());
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(12)?,
        &cancellation,
    );
    let output = square_with_context_exact_native(&backend, &input, &exact)?;
    assert_eq!(
        values(&backend, &authority, &output, &cancellation)?,
        [1.0, 4.0, 9.0]
    );
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(12)?,
        &cancelled,
    );
    assert!(square_with_context_exact_native(&backend, &input, &cancelled_context).is_err());
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn concatenate_and_index_select_preserve_order_and_reverse_rules()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let left = upload_f32(&backend, &authority, &[2, 1], &[1.0, 2.0], &cancellation)?;
    let right = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let concatenated = concatenate_with_context_exact_native(
        &backend,
        &[left.clone(), right.clone()],
        -1,
        &execution,
    )?;
    assert_eq!(concatenated.descriptor().shape(), [2, 3]);
    assert_eq!(
        values(&backend, &authority, &concatenated, &cancellation)?,
        [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
    );
    let upstream = upload_f32(
        &backend,
        &authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let gradients = concatenate_vjp_with_context_exact_native(
        &backend,
        &[left.clone(), right.clone()],
        -1,
        &upstream,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &gradients[0], &cancellation)?,
        [1.0, 4.0]
    );
    assert_eq!(
        values(&backend, &authority, &gradients[1], &cancellation)?,
        [2.0, 3.0, 5.0, 6.0]
    );
    let tangent = concatenate_jvp_with_context_exact_native(
        &backend,
        &[left.clone(), right.clone()],
        &[left, right],
        -1,
        &execution,
    )?;
    assert_eq!(
        tangent.contiguous_bytes()?,
        concatenated.contiguous_bytes()?
    );

    let source = upload_f32(
        &backend,
        &authority,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let indices = upload_i64(&backend, &authority, &[2], &[2, 0], &cancellation)?;
    let selected =
        index_select_with_context_exact_native(&backend, &source, 0, &indices, &execution)?;
    assert_eq!(
        values(&backend, &authority, &selected, &cancellation)?,
        [5.0, 6.0, 1.0, 2.0]
    );
    let selected_upstream = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[10.0, 20.0, 30.0, 40.0],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &index_select_vjp_with_context_exact_native(
                &backend,
                &source,
                0,
                &indices,
                &selected_upstream,
                &execution,
            )?,
            &cancellation,
        )?,
        [30.0, 40.0, 0.0, 0.0, 10.0, 20.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &index_select_jvp_with_context_exact_native(
                &backend, &source, &source, 0, &indices, &execution,
            )?,
            &cancellation,
        )?,
        [5.0, 6.0, 1.0, 2.0]
    );
    let invalid_indices = upload_i64(&backend, &authority, &[1], &[-1], &cancellation)?;
    assert!(
        index_select_with_context_exact_native(&backend, &source, 0, &invalid_indices, &execution,)
            .is_err()
    );
    Ok(())
}

#[test]
fn median_and_rot90_have_deterministic_indices_values_and_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4, 2],
        &[1.0, 10.0, 4.0, 20.0, 2.0, 30.0, 3.0, 40.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let median = median_with_context_exact_native(&backend, &input, 0, false, &execution)?;
    assert_eq!(
        values(&backend, &authority, &median.values, &cancellation)?,
        [2.0, 20.0]
    );
    assert_eq!(
        decoded_flat(&median.indices)?,
        [DecodedScalar::Signed(2), DecodedScalar::Signed(1)]
    );
    let upstream = upload_f32(&backend, &authority, &[2], &[5.0, 7.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &median_vjp_with_context_exact_native(
                &backend, &input, 0, false, &upstream, &execution,
            )?,
            &cancellation,
        )?,
        [0.0, 0.0, 0.0, 7.0, 5.0, 0.0, 0.0, 0.0]
    );
    let tangent = upload_f32(
        &backend,
        &authority,
        &[4, 2],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &median_jvp_with_context_exact_native(
                &backend, &input, &tangent, 0, false, &execution,
            )?,
            &cancellation,
        )?,
        [4.0, 3.0]
    );

    let image = upload_f32(
        &backend,
        &authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let rotated = rot90_with_context_exact_native(&backend, &image, 1, [0, 1], &execution)?;
    assert_eq!(rotated.descriptor().shape(), [3, 2]);
    assert_eq!(
        values(&backend, &authority, &rotated, &cancellation)?,
        [3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
    );
    let rotated_upstream = upload_f32(
        &backend,
        &authority,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &rot90_vjp_with_context_exact_native(
                &backend,
                &image,
                1,
                [0, 1],
                &rotated_upstream,
                &execution,
            )?,
            &cancellation,
        )?,
        [5.0, 3.0, 1.0, 6.0, 4.0, 2.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &rot90_jvp_with_context_exact_native(&backend, &image, &image, 1, [0, 1], &execution,)?,
            &cancellation,
        )?,
        [3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
    );
    assert!(rot90_with_context_exact_native(&backend, &image, 1, [0, 0], &execution).is_err());
    Ok(())
}

struct PassThroughRule;

impl BackwardRule for PassThroughRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Ok(vec![output_gradients.first().cloned().flatten()])
    }
}

struct NoAccumulation;

impl GradientReducer for NoAccumulation {
    fn add(
        &self,
        _left: Tensor,
        _right: Tensor,
        _cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError> {
        Err(AutogradError::InvalidGraph {
            reason: "test reducer should not be called".to_owned(),
        })
    }
}

#[test]
fn autograd_and_device_adapters_project_only_canonical_owners_and_seal_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let used = LeafId::new("used")?;
    let unused = LeafId::new("unused")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let output = tape
        .record(
            vec![AutogradInput::Leaf(used.clone())],
            1,
            Vec::new(),
            Arc::new(PassThroughRule),
        )?
        .ok_or("enabled tape did not return an output slot")?
        .remove(0);
    let seed = upload_f32(&backend, &authority, &[1], &[3.0], &cancellation)?;
    let gradients = autograd_grad_exact_native(
        &mut tape,
        vec![(output, seed)],
        &[used, unused],
        &NoAccumulation,
        true,
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            gradients[0].as_ref().ok_or("used gradient is missing")?,
            &cancellation,
        )?,
        [3.0]
    );
    assert!(gradients[1].is_none());

    let metal =
        BackendCapabilityMatrix::new(DeviceId::new(DeviceKind::Metal, 0), Vec::new(), Vec::new())?;
    let cpu = BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?;
    assert!(mps_is_available_exact_native(
        &[cpu.clone(), metal],
        &cancellation
    )?);
    assert!(!mps_is_available_exact_native(
        std::slice::from_ref(&cpu),
        &cancellation
    )?);
    let mlu =
        BackendCapabilityMatrix::new(DeviceId::new(DeviceKind::Mlu, 3), Vec::new(), Vec::new())?;
    assert_eq!(mlu_current_device_exact_native(&mlu, &cancellation)?, 3);
    assert!(mlu_current_device_exact_native(&cpu, &cancellation).is_err());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(mps_is_available_exact_native(&[cpu], &cancelled).is_err());

    let owner =
        "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-5ab4376a79b5";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_08")
        .ok_or("elementwise/runtime part-eight resolution slice is missing")?;
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
            "COMFY-TENSOR-OP-5DFBC70338A1" => "byte_method_with_context_exact_native",
            "COMFY-TENSOR-OP-5ED49DCA2F78" => "log_method_with_context_exact_native",
            "COMFY-TENSOR-OP-5CEC7CF2D62D" => "autograd_grad_exact_native",
            "COMFY-TENSOR-OP-5BF1DE9DA499" => "mps_is_available_exact_native",
            "COMFY-TENSOR-OP-5C52F193416C" => "concatenate_with_context_exact_native",
            "COMFY-TENSOR-OP-5C00AB949613" => "cos_with_context_exact_native",
            "COMFY-TENSOR-OP-5D7C103AB024" => "expm1_function_with_context_exact_native",
            "COMFY-TENSOR-OP-5AB4376A79B5" => "index_select_with_context_exact_native",
            "COMFY-TENSOR-OP-5BA79209BB02" => "median_with_context_exact_native",
            "COMFY-TENSOR-OP-5EECCD4F0130" => "mlu_current_device_exact_native",
            "COMFY-TENSOR-OP-5CDFF9F97B6F" => "rot90_with_context_exact_native",
            "COMFY-TENSOR-OP-60A72EC2F5DD" => "square_with_context_exact_native",
            _ => return Err("unexpected Task 51 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn every_local_task51_adapter_honors_pre_cancellation_before_validation_or_publication()
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
                Err(ElementwiseRuntimePartEightError::Cancelled)
            ));
        };
    }

    assert_cancelled!(byte_method_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(log_method_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(log_method_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(log_method_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(autograd_grad_exact_native(
        &mut AutogradTape::new(GradientMode::Enabled),
        Vec::new(),
        &[],
        &NoAccumulation,
        false,
        &cancelled,
    ));
    assert_cancelled!(mps_is_available_exact_native(&[], &cancelled));
    assert_cancelled!(concatenate_with_context_exact_native(
        &backend,
        &[],
        8,
        &execution,
    ));
    assert_cancelled!(concatenate_vjp_with_context_exact_native(
        &backend,
        &[],
        8,
        &input,
        &execution,
    ));
    assert_cancelled!(concatenate_jvp_with_context_exact_native(
        &backend,
        &[],
        std::slice::from_ref(&input),
        8,
        &execution,
    ));
    assert_cancelled!(cos_with_context_exact_native(&backend, &input, &execution));
    assert_cancelled!(cos_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(cos_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(expm1_function_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(expm1_function_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(expm1_function_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(index_select_with_context_exact_native(
        &backend, &input, 8, &input, &execution
    ));
    assert_cancelled!(index_select_vjp_with_context_exact_native(
        &backend, &input, 8, &input, &input, &execution
    ));
    assert_cancelled!(index_select_jvp_with_context_exact_native(
        &backend, &input, &input, 8, &input, &execution
    ));
    assert_cancelled!(median_with_context_exact_native(
        &backend, &input, 8, false, &execution
    ));
    assert_cancelled!(median_vjp_with_context_exact_native(
        &backend, &input, 8, false, &input, &execution
    ));
    assert_cancelled!(median_jvp_with_context_exact_native(
        &backend, &input, &input, 8, false, &execution
    ));
    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert_cancelled!(mlu_current_device_exact_native(&cpu, &cancelled));
    assert_cancelled!(rot90_with_context_exact_native(
        &backend,
        &input,
        1,
        [0, 0],
        &execution,
    ));
    assert_cancelled!(rot90_vjp_with_context_exact_native(
        &backend,
        &input,
        1,
        [0, 0],
        &input,
        &execution,
    ));
    assert_cancelled!(rot90_jvp_with_context_exact_native(
        &backend,
        &input,
        &input,
        1,
        [0, 0],
        &execution,
    ));
    assert_cancelled!(square_with_context_exact_native(
        &backend, &input, &execution
    ));
    assert_cancelled!(square_vjp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));
    assert_cancelled!(square_jvp_with_context_exact_native(
        &backend, &input, &input, &execution
    ));

    assert_eq!(execution.scratch.peak_bytes(), 0);
    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, input_bytes);
    Ok(())
}
