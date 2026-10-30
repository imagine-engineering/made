use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor,
    generated_neural_network_module_01::LossReduction,
    generated_neural_network_module_04::{
        AVG_POOL_1D_OPERATION_ID, DROPOUT_OPERATION_ID, ELU_OPERATION_ID, IDENTITY_OPERATION_ID,
        MODULE_OPERATION_ID, MSE_LOSS_OPERATION_ID, NeuralNetworkModulePartFourError,
        SIGMOID_OPERATION_ID, average_pool_1d_jvp_with_context_exact_native,
        average_pool_1d_vjp_with_context_exact_native,
        average_pool_1d_with_context_exact_native, dropout_jvp_with_context_exact_native,
        dropout_vjp_with_context_exact_native, dropout_with_context_exact_native,
        elu_module_jvp_with_context_exact_native, elu_module_vjp_with_context_exact_native,
        elu_module_with_context_exact_native, identity_with_context_exact_native,
        mse_loss_jvp_with_context_exact_native, mse_loss_vjp_with_context_exact_native,
        mse_loss_with_context_exact_native, sigmoid_module_jvp_with_context_exact_native,
        sigmoid_module_vjp_with_context_exact_native, sigmoid_module_with_context_exact_native,
    },
    rng::{RngAlgorithm, RngError, RngProfileVersion, RngStream, RngStreamAddress, RetryRngPolicy},
};
use sim_types::DeviceKind;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(8 * 1024 * 1024)?,
            cancellation,
        ))
    }
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(descriptor, values, &backend.execution(cancellation)?)?
        .0)
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= tolerance, "{actual} != {expected}");
    }
}

fn stream() -> Result<RngStream, Box<dyn std::error::Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V1,
        RngAlgorithm::Philox4x32_10,
        19,
        RngStreamAddress::new(
            "workflow",
            "attempt",
            "dropout",
            0,
            "forward",
            0,
            0,
            RetryRngPolicy::Replay,
        )?,
    )?)
}

#[test]
fn average_pool_1d_forward_vjp_and_jvp_share_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let output = average_pool_1d_with_context_exact_native(
        &backend,
        &[1.0, 3.0, 5.0, 7.0],
        &[1, 4],
        2,
        2,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(output.shape, [1, 2]);
    close(&output.values, &[2.0, 6.0], 0.0);
    let vjp = average_pool_1d_vjp_with_context_exact_native(
        &backend,
        &[1.0, 3.0, 5.0, 7.0],
        &[1, 4],
        2,
        2,
        &[2.0, 4.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[1.0, 1.0, 2.0, 2.0], 0.0);
    let jvp = average_pool_1d_jvp_with_context_exact_native(
        &backend,
        &[1.0, 1.0, 3.0, 3.0],
        &[1, 4],
        2,
        2,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[1.0, 3.0], 0.0);
    Ok(())
}

#[test]
fn dropout_replays_and_commits_only_the_canonical_rng_transaction()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let stream = stream()?;
    let first = dropout_with_context_exact_native(
        &[1.0; 16],
        0.5,
        true,
        stream.begin(None)?,
        DeviceId::CPU,
        &context,
    )?;
    let replay = dropout_with_context_exact_native(
        &[1.0; 16],
        0.5,
        true,
        stream.begin(None)?,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(first.mask, replay.mask);
    assert_eq!(first.values, replay.values);
    let checkpoint = first.transaction.commit();
    let advanced = dropout_with_context_exact_native(
        &[1.0; 16],
        0.5,
        true,
        stream.begin(Some(checkpoint))?,
        DeviceId::CPU,
        &context,
    )?;
    assert_ne!(replay.mask, advanced.mask);
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let mismatched_stream = RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        19,
        RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "dropout",
            0,
            "forward",
            0,
            0,
            RetryRngPolicy::Replay,
            cuda,
        )?,
    )?;
    assert!(matches!(
        dropout_with_context_exact_native(
            &[1.0; 4],
            0.5,
            true,
            mismatched_stream.begin(None)?,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartFourError::Rng(
            RngError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual,
            }
        )) if actual == cuda
    ));
    close(
        &dropout_vjp_with_context_exact_native(
            &[1.0; 16],
            &replay.mask,
            0.5,
            true,
            DeviceId::CPU,
            &context,
        )?,
        &replay.values,
        0.0,
    );
    close(
        &dropout_jvp_with_context_exact_native(
            &[1.0; 16],
            &replay.mask,
            0.5,
            true,
            DeviceId::CPU,
            &context,
        )?,
        &replay.values,
        0.0,
    );
    Ok(())
}

#[test]
fn elu_and_mse_loss_have_matching_analytical_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let elu = elu_module_with_context_exact_native(
        &backend,
        &[-1.0, 0.0, 2.0],
        1.0,
        DeviceId::CPU,
        &context,
    )?;
    close(&elu, &[-0.63212055, 0.0, 2.0], 1.0e-6);
    let elu_vjp = elu_module_vjp_with_context_exact_native(
        &backend,
        &[-1.0, 0.0, 2.0],
        &[1.0; 3],
        1.0,
        DeviceId::CPU,
        &context,
    )?;
    let elu_jvp = elu_module_jvp_with_context_exact_native(
        &backend,
        &[-1.0, 0.0, 2.0],
        &[1.0; 3],
        1.0,
        DeviceId::CPU,
        &context,
    )?;
    close(&elu_vjp, &elu_jvp, 0.0);
    close(
        &mse_loss_with_context_exact_native(
            &[0.0, 2.0],
            &[1.0, 0.0],
            LossReduction::Mean,
            DeviceId::CPU,
            &context,
        )?,
        &[2.5],
        0.0,
    );
    let mse_vjp = mse_loss_vjp_with_context_exact_native(
        &[0.0, 2.0],
        &[1.0, 0.0],
        LossReduction::Mean,
        &[1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&mse_vjp.input, &[-1.0, 2.0], 0.0);
    close(&mse_vjp.target, &[1.0, -2.0], 0.0);
    close(
        &mse_loss_jvp_with_context_exact_native(
            &[0.0, 2.0],
            &[1.0, 1.0],
            &[1.0, 0.0],
            &[0.0, 0.0],
            LossReduction::Mean,
            DeviceId::CPU,
            &context,
        )?,
        &[1.0],
        0.0,
    );
    Ok(())
}

#[test]
fn identity_and_sigmoid_preserve_canonical_tensor_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[3], &[-1.0, 0.0, 1.0], &cancellation)?;
    let identity = identity_with_context_exact_native(&input, &context)?;
    assert_eq!(identity.storage_id(), input.storage_id());
    assert_eq!(identity.descriptor(), input.descriptor());
    let sigmoid = sigmoid_module_with_context_exact_native(&backend, &input, &context)?;
    close(&values(&sigmoid)?, &[0.26894143, 0.5, 0.7310586], 1.0e-6);
    let tangent = upload_f32(&backend, &[3], &[1.0; 3], &cancellation)?;
    let vjp = sigmoid_module_vjp_with_context_exact_native(&backend, &input, &tangent, &context)?;
    let jvp = sigmoid_module_jvp_with_context_exact_native(&backend, &input, &tangent, &context)?;
    close(&values(&vjp)?, &values(&jvp)?, 0.0);
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_arguments_and_rng_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    assert!(matches!(
        average_pool_1d_with_context_exact_native(
            &backend,
            &[],
            &[],
            0,
            0,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    assert!(matches!(
        dropout_with_context_exact_native(
            &[],
            f32::NAN,
            true,
            stream()?.begin(None)?,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    assert!(matches!(
        elu_module_with_context_exact_native(
            &backend,
            &[],
            f32::NAN,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    let input = upload_f32(&backend, &[0], &[], &CancellationToken::default())?;
    assert!(matches!(
        identity_with_context_exact_native(&input, &context),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    assert!(matches!(
        mse_loss_with_context_exact_native(
            &[],
            &[1.0],
            LossReduction::Mean,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    assert!(matches!(
        sigmoid_module_with_context_exact_native(&backend, &input, &context),
        Err(NeuralNetworkModulePartFourError::Cancelled)
    ));
    Ok(())
}

#[test]
fn operation_contracts_are_unique_and_evidence_is_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let expected = BTreeSet::from([
        AVG_POOL_1D_OPERATION_ID,
        DROPOUT_OPERATION_ID,
        ELU_OPERATION_ID,
        IDENTITY_OPERATION_ID,
        MSE_LOSS_OPERATION_ID,
        MODULE_OPERATION_ID,
        SIGMOID_OPERATION_ID,
    ]);
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|contracts| contracts.iter())
        .filter(|contract| expected.contains(contract.operation_id))
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), expected.len());
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        expected
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for contract in contracts {
        let fixture = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(fixture)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
