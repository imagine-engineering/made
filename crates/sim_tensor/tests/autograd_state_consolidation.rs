use sim_tensor::{
    AutogradError, AutogradInput, AutogradTape, BackwardRule, CancellationToken, CpuBackend,
    CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, GradientMode, GradientReducer,
    GradientStore, Layout, LeafId, OutputSlot, SavedTensor, StreamId, TapeState, Tensor,
    TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    autograd::breadth::{AutogradBreadthError, CheckpointCallable, CheckpointFunction},
    generated_elementwise_or_runtime_operation_06::{
        ElementwiseRuntimePartSixError, checkpoint_exact_native,
    },
    generated_elementwise_or_runtime_operation_14::detach_exact_native,
    generated_tensor_creation_01::ones_with_context_exact_native,
};
use std::{collections::HashMap, error::Error, sync::Arc};

fn fixture() -> Result<(CpuBackend, CpuWorkspaceAuthority), TensorError> {
    CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)
}

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        cancellation,
    ))
}

fn tensor(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn integer_tensor(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    value: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(vec![1], DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    let (mut tensor, _) =
        backend.allocate(descriptor, &context(backend, authority, cancellation)?)?;
    tensor
        .write()?
        .bytes_mut()?
        .copy_from_slice(&value.to_ne_bytes());
    Ok(tensor)
}

fn scalar_value(tensor: &Tensor) -> Result<f32, Box<dyn Error>> {
    let bytes: [u8; 4] = tensor
        .contiguous_bytes()?
        .try_into()
        .map_err(|_| "expected one f32")?;
    Ok(f32::from_ne_bytes(bytes))
}

fn overwrite_first(tensor: &mut Tensor, value: f32) -> Result<(), TensorError> {
    tensor
        .write()?
        .bytes_mut()?
        .get_mut(..4)
        .ok_or(TensorError::StorageLength {
            expected: 4,
            actual: 0,
        })?
        .copy_from_slice(&value.to_ne_bytes());
    Ok(())
}

struct IdentityRule;

impl BackwardRule for IdentityRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Ok(vec![output_gradients.first().cloned().flatten()])
    }
}

struct AddReducer;

impl GradientReducer for AddReducer {
    fn add(
        &self,
        mut left: Tensor,
        right: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError> {
        if cancellation.is_cancelled() {
            return Err(AutogradError::Cancelled);
        }
        let left_value = scalar_value(&left).map_err(|error| AutogradError::InvalidGraph {
            reason: error.to_string(),
        })?;
        let right_value = scalar_value(&right).map_err(|error| AutogradError::InvalidGraph {
            reason: error.to_string(),
        })?;
        overwrite_first(&mut left, left_value + right_value)?;
        Ok(left)
    }
}

struct FailingReducer;

impl GradientReducer for FailingReducer {
    fn add(
        &self,
        _left: Tensor,
        _right: Tensor,
        _cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError> {
        Err(AutogradError::InvalidGraph {
            reason: "injected reducer publication failure".to_owned(),
        })
    }
}

fn record_identity(tape: &mut AutogradTape, leaf: LeafId) -> Result<OutputSlot, AutogradError> {
    tape.record(
        vec![AutogradInput::Leaf(leaf)],
        1,
        Vec::new(),
        Arc::new(IdentityRule),
    )?
    .and_then(|mut outputs| outputs.pop())
    .ok_or_else(|| AutogradError::InvalidGraph {
        reason: "enabled tape did not record an identity node".to_owned(),
    })
}

#[test]
fn cow_mutation_preserves_logical_identity_and_invalidates_saved_witnesses()
-> Result<(), Box<dyn Error>> {
    let fixture_document: serde_json::Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/autograd/state-v1.json"
    ))?;
    assert_eq!(fixture_document["schema_version"], 1);
    assert_eq!(
        fixture_document["logical_identity"]["copy_on_write"],
        "same_tensor_id_shared_lineage_new_storage"
    );
    assert_eq!(
        fixture_document["serialization"]["tensor_id"],
        "request_local_not_serialized"
    );
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let mut value = tensor(&backend, &authority, &[1.0], &cancellation)?;
    let handle = value.clone();
    let saved = SavedTensor::capture(&value);
    let original_tensor = value.tensor_id();
    let original_storage = value.storage_id();
    let original_version = value.mutation_version();
    let leaf = LeafId::new("cow-leaf")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&value, Some(leaf.clone()), true, &cancellation)?;

    overwrite_first(&mut value, 2.0)?;

    assert_eq!(value.tensor_id(), original_tensor);
    assert_eq!(handle.tensor_id(), original_tensor);
    assert_ne!(value.storage_id(), original_storage);
    assert_eq!(handle.storage_id(), original_storage);
    assert_eq!(value.mutation_version(), original_version + 1);
    assert_eq!(handle.mutation_version(), value.mutation_version());
    assert_eq!(tape.leaf_binding(&value), Some(&leaf));
    assert!(matches!(
        saved.validate(),
        Err(AutogradError::SavedTensorModified { tensor, .. }) if tensor == original_tensor
    ));
    Ok(())
}

#[test]
fn detach_data_and_views_share_mutation_lineage() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let mut source = tensor(&backend, &authority, &[3.0], &cancellation)?;
    let leaf = LeafId::new("source")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&source, Some(leaf), true, &cancellation)?;
    let detached = detach_exact_native(&source, &cancellation)?;
    let data = source.data_alias()?;
    let view = source.view(source.descriptor().clone(), ViewAccess::ReadOnly)?;

    assert_eq!(detached.storage_id(), source.storage_id());
    assert_eq!(data.storage_id(), source.storage_id());
    assert_eq!(view.storage_id(), source.storage_id());
    assert_ne!(detached.tensor_id(), source.tensor_id());
    assert_ne!(data.tensor_id(), source.tensor_id());
    assert_ne!(view.tensor_id(), source.tensor_id());
    assert!(!tape.requires_grad(&detached));
    assert!(!tape.requires_grad(&data));
    assert!(!tape.requires_grad(&view));

    let saved = SavedTensor::capture(&source);
    let source_id = source.tensor_id();
    overwrite_first(source.detached_in_place(), 4.0)?;
    assert_eq!(source.tensor_id(), source_id);
    assert!(matches!(
        saved.validate(),
        Err(AutogradError::SavedTensorModified { .. })
    ));
    assert_eq!(detached.mutation_version(), source.mutation_version());
    assert_eq!(data.mutation_version(), source.mutation_version());
    assert_eq!(view.mutation_version(), source.mutation_version());

    let replacement = tensor(&backend, &authority, &[8.0], &cancellation)?;
    source.data_in_place().replace_data(replacement)?;
    assert_eq!(source.tensor_id(), source_id);
    assert_eq!(scalar_value(&source)?, 8.0);
    Ok(())
}

#[test]
fn factory_requires_grad_binds_logical_identity() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let leaf = LeafId::new("factory")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let mut output = ones_with_context_exact_native(
        &backend,
        &[1],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        true,
        Some((&mut tape, leaf.clone())),
        &execution,
    )?;
    let logical_id = output.tensor_id();
    let storage_id = output.storage_id();
    let handle = output.clone();
    overwrite_first(&mut output, 9.0)?;

    assert_eq!(output.tensor_id(), logical_id);
    assert_eq!(handle.tensor_id(), logical_id);
    assert_ne!(output.storage_id(), storage_id);
    assert_eq!(tape.leaf_binding(&output), Some(&leaf));
    assert_eq!(tape.leaf_binding(&handle), Some(&leaf));
    Ok(())
}

#[test]
fn requires_grad_rejects_integral_tensor_without_mutating_leaf_state() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let integer = integer_tensor(&backend, &authority, 7, &cancellation)?;
    let leaf = LeafId::new("integral")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);

    assert!(matches!(
        tape.set_requires_grad(&integer, Some(leaf), true, &cancellation),
        Err(AutogradError::InvalidRequiresGradDType { dtype: DType::I64 })
    ));
    assert!(!tape.requires_grad(&integer));
    assert_eq!(tape.state(), &TapeState::Active);
    Ok(())
}

#[test]
fn gradient_publication_and_zeroing_use_canonical_store() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let leaf = LeafId::new("gradient")?;
    let gradient = tensor(&backend, &authority, &[5.0], &cancellation)?;
    let mut store = GradientStore::default();
    store.publish(HashMap::from([(leaf.clone(), gradient)]), &cancellation)?;
    assert_eq!(
        scalar_value(store.gradient(&leaf).ok_or("missing gradient")?)?,
        5.0
    );
    store.zero_grad(&backend, &execution, false)?;
    assert_eq!(
        scalar_value(store.gradient(&leaf).ok_or("missing zero")?)?,
        0.0
    );
    store.zero_grad(&backend, &execution, true)?;
    assert!(store.is_empty());
    Ok(())
}

struct IdentityCheckpoint;

impl CheckpointCallable for IdentityCheckpoint {
    fn forward(
        &self,
        _backend: &CpuBackend,
        inputs: &[Tensor],
        _mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        execution.check()?;
        Ok(inputs.to_vec())
    }

    fn recompute_vjp(
        &self,
        _backend: &CpuBackend,
        _inputs: &[Tensor],
        _parameters: &[Tensor],
        output_gradients: &[Option<Tensor>],
        _mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        execution.check()?;
        Ok(output_gradients.to_vec())
    }
}

#[test]
fn checkpoint_adapters_share_tape_record_and_reverse_path() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;

    let mut operation_input = tensor(&backend, &authority, &[1.0], &cancellation)?;
    let execution_record = checkpoint_exact_native(
        std::slice::from_ref(&operation_input),
        false,
        &cancellation,
        |inputs, _mode, cancellation| {
            cancellation.check()?;
            Ok(vec![inputs[0].clone()])
        },
    )?;
    overwrite_first(&mut operation_input, 2.0)?;
    assert!(matches!(
        execution_record.recompute_exact_native(&cancellation, |inputs, _mode, _| {
            Ok(vec![inputs[0].clone()])
        }),
        Err(ElementwiseRuntimePartSixError::Autograd(
            AutogradError::SavedTensorModified { .. }
        ))
    ));

    let mut breadth_input = tensor(&backend, &authority, &[3.0], &cancellation)?;
    let (checkpoint, outputs) = CheckpointFunction::forward(
        &backend,
        Arc::new(IdentityCheckpoint),
        std::slice::from_ref(&breadth_input),
        &[],
        vec![true],
        sim_tensor::AutocastPolicy::new(false, DType::F32, true)?,
        &execution,
    )?;
    overwrite_first(&mut breadth_input, 4.0)?;
    assert!(matches!(
        checkpoint.backward(&backend, &[Some(outputs[0].clone())], &execution),
        Err(AutogradBreadthError::Autograd(
            AutogradError::SavedTensorModified { .. }
        ))
    ));
    Ok(())
}

#[test]
fn retained_backward_cancellation_and_terminal_release_are_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let leaf = LeafId::new("retained")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let output = record_identity(&mut tape, leaf.clone())?;
    let seed = tensor(&backend, &authority, &[1.0], &cancellation)?;
    let mut store = GradientStore::default();

    tape.reverse_and_publish(
        vec![(output, seed.clone())],
        &AddReducer,
        &cancellation,
        true,
        None,
        &mut store,
    )?;
    assert_eq!(tape.state(), &TapeState::Active);
    assert_eq!(tape.retained_node_count(), 1);
    tape.reverse_and_publish(
        vec![(output, seed)],
        &AddReducer,
        &cancellation,
        false,
        None,
        &mut store,
    )?;
    assert_eq!(
        scalar_value(store.gradient(&leaf).ok_or("missing gradient")?)?,
        2.0
    );
    assert_eq!(tape.state(), &TapeState::Completed);
    assert_eq!(tape.retained_node_count(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_leaf = LeafId::new("cancelled")?;
    let mut cancelled_tape = AutogradTape::new(GradientMode::Enabled);
    let cancelled_output = record_identity(&mut cancelled_tape, cancelled_leaf)?;
    let before = scalar_value(store.gradient(&leaf).ok_or("missing prior gradient")?)?;
    assert!(matches!(
        cancelled_tape.reverse_and_publish(
            vec![(
                cancelled_output,
                tensor(&backend, &authority, &[7.0], &cancellation)?
            )],
            &AddReducer,
            &cancelled,
            false,
            None,
            &mut store,
        ),
        Err(AutogradError::Cancelled)
    ));
    assert_eq!(
        scalar_value(store.gradient(&leaf).ok_or("lost prior gradient")?)?,
        before
    );
    assert!(matches!(cancelled_tape.state(), TapeState::Cancelled(_)));
    assert_eq!(cancelled_tape.retained_node_count(), 0);

    let failing_leaf = LeafId::new("publication-failure")?;
    let mut failing_tape = AutogradTape::new(GradientMode::Enabled);
    let failing_output = record_identity(&mut failing_tape, failing_leaf.clone())?;
    let prior_gradient = tensor(&backend, &authority, &[11.0], &cancellation)?;
    let mut failing_store = GradientStore::default();
    failing_store.publish(
        HashMap::from([(failing_leaf.clone(), prior_gradient)]),
        &cancellation,
    )?;
    assert!(matches!(
        failing_tape.reverse_and_publish(
            vec![(
                failing_output,
                tensor(&backend, &authority, &[3.0], &cancellation)?
            )],
            &FailingReducer,
            &cancellation,
            false,
            None,
            &mut failing_store,
        ),
        Err(AutogradError::InvalidGraph { reason })
            if reason == "injected reducer publication failure"
    ));
    assert!(matches!(failing_tape.state(), TapeState::Faulted(_)));
    assert_eq!(failing_tape.retained_node_count(), 0);
    assert_eq!(
        scalar_value(
            failing_store
                .gradient(&failing_leaf)
                .ok_or("lost transactional prior gradient")?
        )?,
        11.0
    );
    Ok(())
}
