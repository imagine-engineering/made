use sim_model::{ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, ParserLimits};
use sim_model::{
    model_family::{
        MAX_MODEL_WEIGHT_STATISTIC_REQUESTS, ModelFamilyError, ModelWeightStatisticRequest,
    },
    model_store::{ModelStoreError, ModelWeightStatisticError},
};
use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError};
use sim_types::{CancellationToken, DeviceKind};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const SD2_STATISTIC_TENSOR: &str =
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias";

#[test]
fn real_sd2_weight_bytes_produce_exact_unspoofable_population_standard_deviation()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd2.safetensors");
    write_safetensors(
        &path,
        BTreeMap::from([
            ("model_type", "eps"),
            ("population_standard_deviation", "0.0"),
        ]),
        &[
            (
                SD2_STATISTIC_TENSOR,
                "F32",
                vec![4],
                f32_bytes(&[-0.15, -0.05, 0.05, 0.15]),
            ),
            (
                "model.diffusion_model.input_blocks.0.0.weight",
                "F32",
                vec![1],
                vec![0; 4],
            ),
        ],
    )?;
    let (index, store, loaded, cancellation) = load(&path)?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(
        probe.metadata().get("model_type").map(String::as_str),
        Some("eps")
    );
    assert_eq!(
        probe
            .metadata()
            .get("population_standard_deviation")
            .map(String::as_str),
        Some("0.0")
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(16)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let request = ModelWeightStatisticRequest::population_standard_deviation(
        SD2_STATISTIC_TENSOR,
        DeviceKind::Cpu,
    )?;
    let observations = store.observe_weight_statistics_with_context(
        &backend,
        &index,
        &loaded,
        std::slice::from_ref(&request),
        &context,
    )?;
    let observation = observations.first().ok_or("missing SD2 statistic")?;
    let expected = 0.111_803_404_986_858_37_f64;
    assert_eq!(observation.value().to_bits(), expected.to_bits());
    assert!(observation.exceeds_checked(0.09)?);
    assert!(matches!(
        observation.exceeds_checked(f64::NAN),
        Err(ModelFamilyError::NonFiniteWeightStatisticThreshold(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);

    let source = fs::read_to_string(
        workspace_root()?
            .join("projects")
            .join("sim")
            .join("ComfyUI/sim/supported_models.py"),
    )?;
    assert!(source.contains("torch.std(out, unbiased=False) > 0.09"));
    Ok(())
}

#[test]
fn invalid_dtype_device_request_count_empty_tensor_and_cancellation_fail_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("invalid.safetensors");
    write_safetensors(
        &path,
        BTreeMap::new(),
        &[
            (
                SD2_STATISTIC_TENSOR,
                "I64",
                vec![1],
                1_i64.to_le_bytes().to_vec(),
            ),
            ("empty.bias", "F32", vec![0], Vec::new()),
        ],
    )?;
    let (index, store, loaded, cancellation) = load(&path)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(64)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let unsupported_dtype = ModelWeightStatisticRequest::population_standard_deviation(
        SD2_STATISTIC_TENSOR,
        DeviceKind::Cpu,
    )?;
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend,
            &index,
            &loaded,
            std::slice::from_ref(&unsupported_dtype),
            &context,
        ),
        Err(ModelWeightStatisticError::Family(
            ModelFamilyError::WeightStatisticDType { .. }
        ))
    ));
    let unsupported_device = ModelWeightStatisticRequest::population_standard_deviation(
        SD2_STATISTIC_TENSOR,
        DeviceKind::Metal,
    )?;
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend,
            &index,
            &loaded,
            &[unsupported_device],
            &context,
        ),
        Err(ModelWeightStatisticError::Family(
            ModelFamilyError::UnsupportedDevice(DeviceKind::Metal)
        ))
    ));
    let excessive = vec![unsupported_dtype; MAX_MODEL_WEIGHT_STATISTIC_REQUESTS + 1];
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend, &index, &loaded, &excessive, &context,
        ),
        Err(ModelWeightStatisticError::Family(
            ModelFamilyError::WeightStatisticRequestLimit { .. }
        ))
    ));
    let empty =
        ModelWeightStatisticRequest::population_standard_deviation("empty.bias", DeviceKind::Cpu)?;
    assert!(matches!(
        store
            .observe_weight_statistics_with_context(&backend, &index, &loaded, &[empty], &context,),
        Err(ModelWeightStatisticError::Family(
            ModelFamilyError::NonFiniteWeightStatistic { .. }
        ))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(64)?,
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend,
            &index,
            &loaded,
            &[],
            &cancelled_context,
        ),
        Err(ModelWeightStatisticError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}

#[test]
fn workspace_oom_and_artifact_mutation_publish_no_observation()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("failure.safetensors");
    write_safetensors(
        &path,
        BTreeMap::new(),
        &[(
            SD2_STATISTIC_TENSOR,
            "F32",
            vec![4],
            f32_bytes(&[0.0, 0.1, 0.2, 0.3]),
        )],
    )?;
    let (mut index, store, loaded, cancellation) = load(&path)?;
    let request = ModelWeightStatisticRequest::population_standard_deviation(
        SD2_STATISTIC_TENSOR,
        DeviceKind::Cpu,
    )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let insufficient = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(15)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend,
            &index,
            &loaded,
            std::slice::from_ref(&request),
            &insufficient,
        ),
        Err(ModelWeightStatisticError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    write_safetensors(
        &path,
        BTreeMap::new(),
        &[(
            SD2_STATISTIC_TENSOR,
            "F32",
            vec![4],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )],
    )?;
    index.refresh(&cancellation)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(16)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        store.observe_weight_statistics_with_context(
            &backend,
            &index,
            &loaded,
            &[request],
            &context,
        ),
        Err(ModelWeightStatisticError::Store(
            ModelStoreError::ArtifactChanged { .. }
        ))
    ));
    Ok(())
}

fn load(
    path: &Path,
) -> Result<
    (
        ArtifactIndex,
        ModelStore,
        std::sync::Arc<sim_model::LoadedModel>,
        CancellationToken,
    ),
    Box<dyn std::error::Error>,
> {
    let directory = path.parent().ok_or("fixture parent missing")?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "weight-statistic",
        "checkpoints",
        directory,
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new(
        "weight-statistic",
        path.file_name().ok_or("fixture name missing")?,
    )?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok((index, store, loaded, cancellation))
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn write_safetensors(
    path: &Path,
    metadata: BTreeMap<&str, &str>,
    tensors: &[(&str, &str, Vec<u64>, Vec<u8>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::to_value(metadata)?);
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
