use sim_tensor::{
    AutogradError, AutogradTape, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType,
    DecodedScalar, DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    GradientMode, Layout, LeafId, Scalar, StreamId, TapeState, Tensor, TensorDescriptor,
    ViewAccess,
    generated_tensor_creation_01::{
        ARANGE_OPERATION_ID, AS_TENSOR_OPERATION_ID, EMPTY_OPERATION_ID, EYE_OPERATION_ID,
        FROM_NUMPY_OPERATION_ID, FULL_OPERATION_ID, LINSPACE_OPERATION_ID, NativeArray,
        NativeTensorInput, ONES_OPERATION_ID, TENSOR_OPERATION_ID, TensorCreationPartOneError,
        ZEROS_OPERATION_ID, arange_with_context_exact_native,
        as_tensor_jvp_with_context_exact_native, as_tensor_vjp_with_context_exact_native,
        as_tensor_with_context_exact_native, empty_with_context_exact_native,
        eye_with_context_exact_native, from_numpy_exact_native, full_with_context_exact_native,
        linspace_jvp_with_context_exact_native, linspace_vjp_exact_native,
        linspace_with_context_exact_native, ones_with_context_exact_native,
        tensor_with_context_exact_native, zeros_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use sha2::{Digest, Sha256};
use std::{fs, ops::Deref, path::Path, sync::Arc};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(16 * 1024 * 1024)?,
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

fn decoded_values(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    (0..count)
        .map(|index| {
            Ok(tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.linear_element_bytes(index)?)?)
        })
        .collect()
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

fn assert_leaf_binding(tape: &AutogradTape, tensor: &Tensor, expected_leaf: &str) {
    assert!(tape.requires_grad(tensor));
    assert_eq!(
        tape.leaf_binding(tensor).map(LeafId::as_str),
        Some(expected_leaf)
    );
}

#[test]
fn arange_matches_integer_float_empty_and_dtype_contracts() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let integers = arange_with_context_exact_native(
        &backend,
        Scalar::Signed(-3),
        Scalar::Signed(5),
        Scalar::Signed(2),
        None,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(integers.descriptor().dtype(), DType::I64);
    assert_eq!(integers.descriptor().shape(), [4]);
    assert_eq!(
        decoded_values(&integers)?,
        [
            DecodedScalar::Signed(-3),
            DecodedScalar::Signed(-1),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(3),
        ]
    );

    let descending = arange_with_context_exact_native(
        &backend,
        Scalar::Float(1.0),
        Scalar::Float(-0.1),
        Scalar::Float(-0.25),
        Some(DType::F64),
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(descending.descriptor().shape(), [5]);
    assert_eq!(
        decoded_values(&descending)?,
        [
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(0.75),
            DecodedScalar::Real(0.5),
            DecodedScalar::Real(0.25),
            DecodedScalar::Real(0.0),
        ]
    );
    let empty = arange_with_context_exact_native(
        &backend,
        Scalar::Signed(4),
        Scalar::Signed(2),
        Scalar::Signed(1),
        None,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), [0]);
    assert!(empty.contiguous_bytes()?.is_empty());
    assert!(matches!(
        arange_with_context_exact_native(
            &backend,
            Scalar::Signed(0),
            Scalar::Signed(3),
            Scalar::Signed(0),
            None,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Invalid { operation, .. })
            if operation == ARANGE_OPERATION_ID
    ));
    assert!(matches!(
        arange_with_context_exact_native(
            &backend,
            Scalar::Signed(0),
            Scalar::Signed(3),
            Scalar::Signed(1),
            Some(DType::Bool),
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype: DType::Bool,
            ..
        }) if operation == ARANGE_OPERATION_ID
    ));
    Ok(())
}

#[test]
fn tensor_and_as_tensor_distinguish_aliasing_copying_inference_and_casts()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let source = upload_f32(&backend, &[2], &[1.5, -2.25], &cancellation)?;

    let alias = as_tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Tensor(&source),
        None,
        None,
        &context,
    )?;
    assert_eq!(alias.storage_id(), source.storage_id());
    let cast = as_tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Tensor(&source),
        Some(DType::F64),
        None,
        &context,
    )?;
    assert_ne!(cast.storage_id(), source.storage_id());
    assert_eq!(cast.descriptor().dtype(), DType::F64);

    let copy = tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Tensor(&source),
        None,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_ne!(copy.storage_id(), source.storage_id());
    assert_eq!(decoded_values(&copy)?, decoded_values(&source)?);

    let literal_values = [Scalar::Boolean(true), Scalar::Boolean(false)];
    let literal = as_tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Literal {
            values: &literal_values,
            shape: &[2],
        },
        None,
        None,
        &context,
    )?;
    assert_eq!(literal.descriptor().dtype(), DType::Bool);
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    assert!(matches!(
        tensor_with_context_exact_native(
            &backend,
            NativeTensorInput::Literal {
                values: &[Scalar::Signed(1)],
                shape: &[1],
            },
            None,
            DeviceId::CPU,
            true,
            Some((&mut tape, LeafId::new("invalid-integer-tensor")?)),
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype: DType::I64,
            ..
        }) if operation == TENSOR_OPERATION_ID
    ));
    Ok(())
}

#[test]
fn from_numpy_is_a_zero_copy_read_only_strided_native_array_adapter()
-> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    for value in [10_i32, 20, 30, 40, 50, 60] {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let bytes: Arc<[u8]> = Arc::from(bytes);
    let array = NativeArray::new(
        bytes.clone(),
        vec![2, 2],
        vec![3, 1],
        0,
        DType::I32,
        StreamId::DEFAULT,
    )?;
    let cancellation = CancellationToken::default();
    let mut tensor = from_numpy_exact_native(&array, &cancellation)?;
    assert_eq!(tensor.access(), ViewAccess::ReadOnly);
    assert_eq!(tensor.descriptor().device(), DeviceId::CPU);
    assert_eq!(tensor.descriptor().strides(), [3, 1]);
    assert_eq!(tensor.host_storage_bytes()?.as_ptr(), bytes.as_ptr());
    assert_eq!(
        decoded_values(&tensor)?,
        [
            DecodedScalar::Signed(10),
            DecodedScalar::Signed(20),
            DecodedScalar::Signed(40),
            DecodedScalar::Signed(50),
        ]
    );
    assert!(matches!(
        tensor.write(),
        Err(sim_tensor::TensorError::ReadOnlyView)
    ));

    let invalid = NativeArray::new(
        Arc::from(vec![0_u8; 4]),
        vec![2, 2],
        vec![2, 1],
        0,
        DType::I32,
        StreamId::DEFAULT,
    );
    assert!(matches!(
        invalid,
        Err(TensorCreationPartOneError::Invalid { operation, .. })
            if operation == FROM_NUMPY_OPERATION_ID
    ));

    let reversed = NativeArray::new(bytes, vec![2], vec![-1], 1, DType::I32, StreamId::DEFAULT)?;
    let reversed = from_numpy_exact_native(&reversed, &cancellation)?;
    assert_eq!(reversed.descriptor().strides(), [-1]);
    assert_eq!(
        decoded_values(&reversed)?,
        [DecodedScalar::Signed(20), DecodedScalar::Signed(10)]
    );
    Ok(())
}

#[test]
fn empty_eye_full_ones_and_zeros_use_checked_canonical_allocations()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;

    let empty = empty_with_context_exact_native(
        &backend,
        &[2, 0, 3],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), [2, 0, 3]);
    assert!(empty.contiguous_bytes()?.is_empty());

    let eye = eye_with_context_exact_native(
        &backend,
        2,
        Some(3),
        DType::I16,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(
        decoded_values(&eye)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
        ]
    );

    let full = full_with_context_exact_native(
        &backend,
        &[2, 2],
        Scalar::Float(-1.5),
        None,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(full.descriptor().dtype(), DType::F32);
    assert_eq!(decoded_values(&full)?, [DecodedScalar::Real(-1.5); 4]);
    let ones = ones_with_context_exact_native(
        &backend,
        &[3],
        DType::U8,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(decoded_values(&ones)?, [DecodedScalar::Unsigned(1); 3]);
    let zeros = zeros_with_context_exact_native(
        &backend,
        &[3],
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(
        decoded_values(&zeros)?,
        [DecodedScalar::Complex {
            real: 0.0,
            imaginary: 0.0,
        }; 3]
    );
    Ok(())
}

#[test]
fn requires_grad_factories_bind_fresh_floating_and_complex_storage()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);

    let arange = arange_with_context_exact_native(
        &backend,
        Scalar::Float(-1.0),
        Scalar::Float(1.0),
        Scalar::Float(0.5),
        Some(DType::F32),
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-arange")?)),
        &context,
    )?;
    let empty = empty_with_context_exact_native(
        &backend,
        &[2],
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-empty")?)),
        &context,
    )?;
    let eye = eye_with_context_exact_native(
        &backend,
        2,
        None,
        DType::F64,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-eye")?)),
        &context,
    )?;
    let full = full_with_context_exact_native(
        &backend,
        &[2],
        Scalar::Float(2.5),
        Some(DType::Complex64),
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-full")?)),
        &context,
    )?;
    let linspace = linspace_with_context_exact_native(
        &backend,
        0.0,
        1.0,
        3,
        DType::F64,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-linspace")?)),
        &context,
    )?;
    let ones = ones_with_context_exact_native(
        &backend,
        &[2],
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-ones")?)),
        &context,
    )?;
    let complex_source = zeros_with_context_exact_native(
        &backend,
        &[2],
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    let tensor = tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Tensor(&complex_source),
        None,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-tensor")?)),
        &context,
    )?;
    let zeros = zeros_with_context_exact_native(
        &backend,
        &[2],
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, LeafId::new("factory-zeros")?)),
        &context,
    )?;

    let outputs = [
        (&arange, "factory-arange"),
        (&empty, "factory-empty"),
        (&eye, "factory-eye"),
        (&full, "factory-full"),
        (&linspace, "factory-linspace"),
        (&ones, "factory-ones"),
        (&tensor, "factory-tensor"),
        (&zeros, "factory-zeros"),
    ];
    for (output, expected_leaf) in outputs {
        assert_leaf_binding(&tape, output, expected_leaf);
    }
    assert_eq!(
        outputs
            .into_iter()
            .map(|(output, _)| output.storage_id())
            .collect::<std::collections::HashSet<_>>()
            .len(),
        outputs.len()
    );
    assert_ne!(tensor.storage_id(), complex_source.storage_id());
    assert!(!tape.requires_grad(&complex_source));
    Ok(())
}

#[test]
fn autograd_registration_must_match_and_failure_paths_bind_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;

    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Invalid { operation, reason })
            if operation == ZEROS_OPERATION_ID
                && reason == "requires_grad=true requires a canonical AutogradTape and checked LeafId"
    ));

    let mut mismatch_tape = AutogradTape::new(GradientMode::Enabled);
    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            Some((&mut mismatch_tape, LeafId::new("mismatched-registration")?)),
            &context,
        ),
        Err(TensorCreationPartOneError::Invalid { operation, reason })
            if operation == ZEROS_OPERATION_ID
                && reason == "an autograd leaf registration requires requires_grad=true"
    ));
    assert_eq!(mismatch_tape.state(), &TapeState::Active);
    assert_eq!(mismatch_tape.retained_node_count(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution(&cancelled)?;
    let mut cancelled_tape = AutogradTape::new(GradientMode::Enabled);
    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            true,
            Some((&mut cancelled_tape, LeafId::new("cancelled-registration")?)),
            &cancelled_context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == ZEROS_OPERATION_ID
    ));
    assert_eq!(cancelled_tape.state(), &TapeState::Active);
    assert_eq!(cancelled_tape.retained_node_count(), 0);

    let sentinel = upload_f32(&backend, &[1], &[1.0], &cancellation)?;
    let mut terminal_tape = AutogradTape::new(GradientMode::Enabled);
    terminal_tape.set_requires_grad(
        &sentinel,
        Some(LeafId::new("sentinel")?),
        true,
        &cancellation,
    )?;
    terminal_tape.cancel("test terminal tape")?;
    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            true,
            Some((&mut terminal_tape, LeafId::new("must-not-bind")?)),
            &context,
        ),
        Err(TensorCreationPartOneError::Autograd {
            operation,
            source: AutogradError::TerminalTape { .. },
        }) if operation == ZEROS_OPERATION_ID
    ));
    assert_eq!(
        terminal_tape.state(),
        &TapeState::Cancelled("test terminal tape".to_owned())
    );
    assert_leaf_binding(&terminal_tape, &sentinel, "sentinel");
    assert_eq!(terminal_tape.retained_node_count(), 0);
    Ok(())
}

#[test]
fn linspace_f32_preserves_source_rounding_for_parameterized_karras()
-> Result<(), Box<dyn std::error::Error>> {
    let value = sim_tensor::generated_tensor_creation_01::linspace_f32_value(0.0, 1.0, 20, 15)?;
    assert_eq!(
        value.to_bits(),
        0x3f4a_1af2,
        "source torch.linspace(0, 1, 20) ramp used by Karras rho=100"
    );
    Ok(())
}

#[test]
fn linspace_matches_endpoint_rules_and_analytical_vjp_jvp() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let output = linspace_with_context_exact_native(
        &backend,
        -1.0,
        1.0,
        5,
        DType::F64,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(
        decoded_values(&output)?,
        [
            DecodedScalar::Real(-1.0),
            DecodedScalar::Real(-0.5),
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(0.5),
            DecodedScalar::Real(1.0),
        ]
    );
    let one = linspace_with_context_exact_native(
        &backend,
        3.0,
        9.0,
        1,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert_eq!(decoded_values(&one)?, [DecodedScalar::Real(3.0)]);
    let empty = linspace_with_context_exact_native(
        &backend,
        3.0,
        9.0,
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert!(empty.contiguous_bytes()?.is_empty());

    let gradient = upload_f32(&backend, &[5], &[1.0; 5], &cancellation)?;
    assert_eq!(
        linspace_vjp_exact_native(&gradient, &cancellation)?,
        [2.5, 2.5]
    );
    let tangent = linspace_jvp_with_context_exact_native(
        &backend,
        2.0,
        4.0,
        3,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(
        decoded_values(&tangent)?,
        [
            DecodedScalar::Real(2.0),
            DecodedScalar::Real(3.0),
            DecodedScalar::Real(4.0),
        ]
    );
    let integer_gradient = ones_with_context_exact_native(
        &backend,
        &[3],
        DType::I64,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        &context,
    )?;
    assert!(matches!(
        linspace_vjp_exact_native(&integer_gradient, &cancellation),
        Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype: DType::I64,
            ..
        }) if operation == LINSPACE_OPERATION_ID
    ));
    Ok(())
}

#[test]
fn as_tensor_reuses_canonical_identity_and_cast_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let primal = upload_f32(&backend, &[2], &[1.0, 2.0], &cancellation)?;
    let gradient = upload_f32(&backend, &[2], &[3.0, 4.0], &cancellation)?;
    let identity = as_tensor_vjp_with_context_exact_native(
        &backend,
        &primal,
        &gradient,
        DType::F32,
        &context,
    )?;
    assert_eq!(decoded_values(&identity)?, decoded_values(&gradient)?);
    let tangent = as_tensor_jvp_with_context_exact_native(
        &backend,
        &gradient,
        DType::F64,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(tangent.descriptor().dtype(), DType::F64);
    assert_eq!(decoded_values(&tangent)?, decoded_values(&gradient)?);
    let output_gradient = as_tensor_with_context_exact_native(
        &backend,
        NativeTensorInput::Tensor(&gradient),
        Some(DType::F64),
        None,
        &context,
    )?;
    let pullback = as_tensor_vjp_with_context_exact_native(
        &backend,
        &primal,
        &output_gradient,
        DType::F64,
        &context,
    )?;
    assert_eq!(pullback.descriptor().dtype(), DType::F32);
    assert_eq!(decoded_values(&pullback)?, decoded_values(&gradient)?);
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_factory_arguments_and_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let native_array = NativeArray::new(
        Arc::from(vec![0_u8; 4]),
        vec![1],
        vec![1],
        0,
        DType::I32,
        StreamId::DEFAULT,
    )?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        arange_with_context_exact_native(
            &backend,
            Scalar::Float(f64::NAN),
            Scalar::Float(f64::INFINITY),
            Scalar::Float(0.0),
            Some(DType::Bool),
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == ARANGE_OPERATION_ID
    ));
    assert!(matches!(
        as_tensor_with_context_exact_native(
            &backend,
            NativeTensorInput::Literal {
                values: &[Scalar::Signed(1)],
                shape: &[2],
            },
            Some(DType::Bool),
            Some(cuda),
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == AS_TENSOR_OPERATION_ID
    ));
    assert!(matches!(
        empty_with_context_exact_native(
            &backend,
            &[u64::MAX, 2],
            DType::I64,
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == EMPTY_OPERATION_ID
    ));
    assert!(matches!(
        eye_with_context_exact_native(
            &backend,
            u64::MAX,
            Some(u64::MAX),
            DType::Bool,
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == EYE_OPERATION_ID
    ));
    assert!(matches!(
        from_numpy_exact_native(&native_array, &cancellation),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == FROM_NUMPY_OPERATION_ID
    ));
    assert!(matches!(
        full_with_context_exact_native(
            &backend,
            &[u64::MAX, 2],
            Scalar::Float(f64::NAN),
            Some(DType::Bool),
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == FULL_OPERATION_ID
    ));
    assert!(matches!(
        linspace_with_context_exact_native(
            &backend,
            f64::NAN,
            f64::INFINITY,
            u64::MAX,
            DType::Bool,
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == LINSPACE_OPERATION_ID
    ));
    assert!(matches!(
        ones_with_context_exact_native(
            &backend,
            &[u64::MAX, 2],
            DType::Bool,
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == ONES_OPERATION_ID
    ));
    assert!(matches!(
        tensor_with_context_exact_native(
            &backend,
            NativeTensorInput::Literal {
                values: &[Scalar::Signed(1)],
                shape: &[2],
            },
            Some(DType::Bool),
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == TENSOR_OPERATION_ID
    ));
    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[u64::MAX, 2],
            DType::Bool,
            Layout::ChannelsLast,
            cuda,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::Cancelled { operation })
            if operation == ZEROS_OPERATION_ID
    ));
    Ok(())
}

#[test]
fn unsupported_devices_layouts_and_grad_dtypes_are_typed_per_operation()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        arange_with_context_exact_native(
            &backend,
            Scalar::Signed(0),
            Scalar::Signed(1),
            Scalar::Signed(1),
            None,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == ARANGE_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        as_tensor_with_context_exact_native(
            &backend,
            NativeTensorInput::Literal {
                values: &[Scalar::Signed(1)],
                shape: &[1],
            },
            None,
            Some(cuda),
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == AS_TENSOR_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        empty_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == EMPTY_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        eye_with_context_exact_native(
            &backend,
            1,
            None,
            DType::F32,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == EYE_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        full_with_context_exact_native(
            &backend,
            &[1],
            Scalar::Signed(1),
            None,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == FULL_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        linspace_with_context_exact_native(
            &backend,
            0.0,
            1.0,
            2,
            DType::F32,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == LINSPACE_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        ones_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == ONES_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        tensor_with_context_exact_native(
            &backend,
            NativeTensorInput::Literal {
                values: &[Scalar::Signed(1)],
                shape: &[1],
            },
            None,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == TENSOR_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        zeros_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Strided,
            cuda,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDevice { operation, device })
            if operation == ZEROS_OPERATION_ID && device == cuda
    ));
    assert!(matches!(
        ones_with_context_exact_native(
            &backend,
            &[1],
            DType::F32,
            Layout::Contiguous,
            DeviceId::CPU,
            false,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedLayout { operation, .. })
            if operation == ONES_OPERATION_ID
    ));
    assert!(matches!(
        eye_with_context_exact_native(
            &backend,
            2,
            None,
            DType::I32,
            Layout::Strided,
            DeviceId::CPU,
            true,
            None,
            &context,
        ),
        Err(TensorCreationPartOneError::UnsupportedDType { operation, .. })
            if operation == EYE_OPERATION_ID
    ));
    Ok(())
}

#[test]
fn all_ten_operation_identifiers_are_distinct() {
    let identifiers = [
        ARANGE_OPERATION_ID,
        AS_TENSOR_OPERATION_ID,
        EMPTY_OPERATION_ID,
        EYE_OPERATION_ID,
        FROM_NUMPY_OPERATION_ID,
        FULL_OPERATION_ID,
        LINSPACE_OPERATION_ID,
        ONES_OPERATION_ID,
        TENSOR_OPERATION_ID,
        ZEROS_OPERATION_ID,
    ];
    let unique = identifiers
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), 10);
}

#[test]
fn all_ten_resolutions_are_unique_runtime_sealed_and_fixture_backed()
-> Result<(), Box<dyn std::error::Error>> {
    let expected = [
        (
            ARANGE_OPERATION_ID,
            "arange.json",
            "b4ae28c242c7cae3a7371934bca2faab25e402cf8a4d52adca367cc56dd3cddd",
        ),
        (
            AS_TENSOR_OPERATION_ID,
            "as_tensor.json",
            "232ab61bc27db5f02f5daa181d71a37e8ad23f48c7f62d80774cddcd988488b5",
        ),
        (
            EMPTY_OPERATION_ID,
            "empty.json",
            "eba5348b19ad2079ee6607477db875189dc197b6793a3613c0226f6282f3601d",
        ),
        (
            EYE_OPERATION_ID,
            "eye.json",
            "924df8a84bb2891654a46f1f11e5a96dc1ede54bb782c8df65bee770102dacac",
        ),
        (
            FROM_NUMPY_OPERATION_ID,
            "from_numpy.json",
            "e3ff781868acad4eab1ab45461f7e3ecea14079e278d912c3bf1dff0ff5ad846",
        ),
        (
            FULL_OPERATION_ID,
            "full.json",
            "3cfa23106bc5f824d0e941657493b08a8d379cad0c6bfb20f96fde08d2288cb2",
        ),
        (
            LINSPACE_OPERATION_ID,
            "linspace.json",
            "ebfc64825e3ff7da77ec9a9bd480bc6e7e7c3ffee0bf9b21d00361ecc5a9625e",
        ),
        (
            ONES_OPERATION_ID,
            "ones.json",
            "d7f803a4928a4c66b77ebb2bffdc41ab3587c71a3b7c5e75e3105313b6aee184",
        ),
        (
            TENSOR_OPERATION_ID,
            "tensor.json",
            "e92b3bf42aafca511e6d4007304641b4409a3fd60023e9e5edff3d1e60a3be6e",
        ),
        (
            ZEROS_OPERATION_ID,
            "zeros.json",
            "ad8cf0dca20a572f10f2377eb9d6ffd86d63a42430ab251455b78e58179608f5",
        ),
    ];
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let all_contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.iter())
        .collect::<Vec<_>>();
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "tensor_creation_01")
        .ok_or("tensor_creation_01 resolution slice was not generated")?;
    assert_eq!(slice.len(), expected.len());
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<std::collections::BTreeSet<_>>(),
        expected
            .iter()
            .map(|(operation_id, _, _)| *operation_id)
            .collect::<std::collections::BTreeSet<_>>()
    );
    for (operation_id, file_name, expected_digest) in expected {
        let contracts = all_contracts
            .iter()
            .filter(|contract| contract.operation_id == operation_id)
            .collect::<Vec<_>>();
        assert_eq!(contracts.len(), 1, "{operation_id}");
        let contract = contracts
            .first()
            .ok_or("generated resolution disappeared after uniqueness check")?;
        assert_eq!(contract.resolution_module, "tensor_creation_01");
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-tensor-creation-comfy-tensor-op-00009bb729df"
        );
        assert!(contract.evidence_fixture.ends_with(file_name));
        assert_eq!(contract.evidence_fixture_sha256, expected_digest);
        assert_ne!(
            contract.evidence_fixture_sha256,
            contract.baseline_fixture_sha256
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_digest);
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(fixture["operation_id"], operation_id);
        assert_eq!(
            fixture["source_observations"].as_array().map(Vec::len),
            Some(if operation_id == LINSPACE_OPERATION_ID {
                5
            } else {
                4
            })
        );
    }
    Ok(())
}
