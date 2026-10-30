use sim_model::{ParserLimits, TorchArchiveFileLoader};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, StreamId,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::{TorchArchiveValue, torch_save_exact_native},
    generated_elementwise_or_runtime_operation_09::torch_load_with_context_exact_native,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use std::{collections::BTreeMap, fs};

#[test]
fn canonical_model_parser_loads_native_torch_archive_without_tensor_side_security()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let descriptor =
        TensorDescriptor::contiguous(vec![2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: workspace_authority.authorize_workspace(16 * 1024 * 1024)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let tensor = backend.upload_f32(descriptor, &[1.5, -2.0], &context)?.0;
    let mut values = BTreeMap::new();
    values.insert("weight".to_owned(), TorchArchiveValue::Tensor(tensor));
    values.insert("step".to_owned(), TorchArchiveValue::Integer(7));
    let archive = torch_save_exact_native(&TorchArchiveValue::Map(values), &cancellation)?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("weights.pt");
    fs::write(&path, &archive)?;
    let limits = ParserLimits::default();
    let loader = TorchArchiveFileLoader::new(&path, &limits);
    let loaded =
        torch_load_with_context_exact_native(&loader, &backend, DeviceId::CPU, true, &context)?;
    let TorchArchiveValue::Map(loaded) = loaded else {
        return Err("loaded value is not a map".into());
    };
    assert!(matches!(
        loaded.get("step"),
        Some(TorchArchiveValue::Integer(7))
    ));
    let Some(TorchArchiveValue::Tensor(tensor)) = loaded.get("weight") else {
        return Err("loaded tensor is missing".into());
    };
    assert_eq!(
        &*tensor_to_f32_with_context_exact_native(&backend, tensor, &context)?,
        [1.5, -2.0]
    );

    let mut corrupted = archive;
    let index = corrupted
        .iter()
        .position(|byte| *byte == b'l')
        .ok_or("archive has no mutable fixture byte")?;
    corrupted[index] ^= 1;
    fs::write(&path, corrupted)?;
    let loader = TorchArchiveFileLoader::new(&path, &limits);
    assert!(
        torch_load_with_context_exact_native(&loader, &backend, DeviceId::CPU, true, &context)
            .is_err()
    );
    Ok(())
}
