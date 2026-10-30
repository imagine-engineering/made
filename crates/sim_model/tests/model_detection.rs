use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelConfigurationKind, ModelConfigurationValue,
    ModelDetectionOutcome, ModelDetectionPolicy, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProbeError, ModelFamilyProbeErrorKind,
    ModelFamilyRegistry, ModelFormatError, ModelForwardOperation, ModelForwardStep,
    ModelStorageDType, ModelStore, ModelStoreError, ModelWeightRule, ParserLimits,
};
use sim_tensor::DType;
use sim_types::{CancellationToken, DeviceKind};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const FIXTURE_MANIFEST: &str = "../sim_test_support/fixtures/model_detection/manifest-v1.json";

const COMPONENTS: [ModelFamilyComponent; 1] = [ModelFamilyComponent {
    identifier: "denoiser",
    role: "diffusion",
    required: true,
}];
const CLIP_CANDIDATES: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sd1_clip.SD1Tokenizer",
        clip_model: "sd1_clip.SD1ClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
static CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &CLIP_CANDIDATES,
    dynamic_selection: false,
};
const WEIGHT_RULES: [ModelWeightRule; 1] = [ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "model.",
    required: true,
}];
const REQUIRED_KEYS: [&str; 1] = ["model.input"];
const DTYPES: [DType; 1] = [DType::U8];
const DEVICES: [DeviceKind; 1] = [DeviceKind::Cpu];
const FORWARD: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "identity",
    operation: ModelForwardOperation::MultiplyScalar(1.0),
}];
const NATIVE_RULES: [ModelDetectionRule; 2] = [
    ModelDetectionRule::ExactShape {
        key: "model.diffusion_model.input_blocks.0.0.weight",
        shape: &[320, 4, 1, 1],
        score: 100,
    },
    ModelDetectionRule::ExactShape {
        key: "model.diffusion_model.out.2.weight",
        shape: &[4, 320, 1, 1],
        score: 100,
    },
];
static NATIVE_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9701",
    identifier: "DetectionNative",
    architecture_version: "detection-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &CLIP_TARGET,
    components: &COMPONENTS,
    detection_rules: &NATIVE_RULES,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED_KEYS,
    optional_keys: &[],
    supported_dtypes: &DTYPES,
    supported_devices: &DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 1,
        bytes_per_parameter: 1,
        activation_bytes_per_element: 1,
    },
    forward_program: &FORWARD,
};
static TIED_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9702",
    identifier: "DetectionTie",
    ..NATIVE_FAMILY
};
static NATIVE_REGISTRY: [ModelFamilyDefinition; 1] = [NATIVE_FAMILY];
static TIED_REGISTRY: [ModelFamilyDefinition; 2] = [NATIVE_FAMILY, TIED_FAMILY];

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    schema_version: u16,
    fixture_id: String,
    oracle: FixtureOracle,
    cases: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FixtureOracle {
    executed_in_release_tests: bool,
    source: String,
    source_sha256: String,
}

fn load_model(
    path: &Path,
    extensions: &[&str],
) -> Result<
    (
        ArtifactIndex,
        ModelStore,
        std::sync::Arc<sim_model::LoadedModel>,
    ),
    Box<dyn Error>,
> {
    let root_path = path.parent().ok_or("model fixture has no parent")?;
    let file_name = path.file_name().ok_or("model fixture has no file name")?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "model-detection",
        "checkpoints",
        root_path,
        extensions.iter().copied(),
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("model-detection", PathBuf::from(file_name))?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let model = store.load(&index, &key, &cancellation)?;
    Ok((index, store, model))
}

fn rejected_model_load(
    path: &Path,
    extensions: &[&str],
) -> Result<(ArtifactIndex, ModelStore, ArtifactKey, ModelStoreError), Box<dyn Error>> {
    let root_path = path.parent().ok_or("model fixture has no parent")?;
    let file_name = path.file_name().ok_or("model fixture has no file name")?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "rejected-model-detection",
        "checkpoints",
        root_path,
        extensions.iter().copied(),
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("rejected-model-detection", PathBuf::from(file_name))?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let error = store
        .load(&index, &key, &cancellation)
        .expect_err("invalid fixture unexpectedly entered the model cache");
    Ok((index, store, key, error))
}

fn write_safetensors(
    path: &Path,
    metadata: BTreeMap<&str, &str>,
    tensors: &[(String, String, Vec<u64>, Vec<u8>)],
) -> Result<(), Box<dyn Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::to_value(metadata)?);
    let mut data = Vec::new();
    for (name, data_type, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            json!({
                "dtype": data_type,
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
    Ok(())
}

fn safetensor(
    name: &str,
    data_type: &str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
) -> (String, String, Vec<u64>, Vec<u8>) {
    (name.to_owned(), data_type.to_owned(), shape, bytes)
}

fn tensor_pickle(name: &str, storage_key: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = vec![0x80, 0x02, b'}', b'('];
    binunicode(&mut bytes, name)?;
    bytes.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
    binunicode(&mut bytes, "storage")?;
    bytes.extend_from_slice(b"ctorch\nFloatStorage\n");
    binunicode(&mut bytes, storage_key)?;
    binunicode(&mut bytes, "cpu")?;
    bytes.extend_from_slice(&[
        b'K', 1, b't', b'Q', b'K', 0, b'(', b'K', 1, b't', b'(', b'K', 1, b't', 0x89, b'}', b't',
        b'R', b'u', b'.',
    ]);
    Ok(bytes)
}

fn binunicode(bytes: &mut Vec<u8>, value: &str) -> Result<(), Box<dyn Error>> {
    bytes.push(b'X');
    bytes.extend_from_slice(&u32::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_stored_zip(path: &Path, entries: &[(&str, &[u8])]) -> Result<(), Box<dyn Error>> {
    let mut output = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = u32::try_from(output.len())?;
        let name_bytes = name.as_bytes();
        let crc = crc32(data);
        output.extend_from_slice(&0x0403_4b50_u32.to_le_bytes());
        output.extend_from_slice(&20_u16.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&crc.to_le_bytes());
        output.extend_from_slice(&u32::try_from(data.len())?.to_le_bytes());
        output.extend_from_slice(&u32::try_from(data.len())?.to_le_bytes());
        output.extend_from_slice(&u16::try_from(name_bytes.len())?.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(name_bytes);
        output.extend_from_slice(data);

        central.extend_from_slice(&0x0201_4b50_u32.to_le_bytes());
        central.extend_from_slice(&0x0314_u16.to_le_bytes());
        central.extend_from_slice(&20_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&u32::try_from(data.len())?.to_le_bytes());
        central.extend_from_slice(&u32::try_from(data.len())?.to_le_bytes());
        central.extend_from_slice(&u16::try_from(name_bytes.len())?.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&(0o100644_u32 << 16).to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name_bytes);
    }
    let central_offset = u32::try_from(output.len())?;
    let central_size = u32::try_from(central.len())?;
    output.extend_from_slice(&central);
    output.extend_from_slice(&0x0605_4b50_u32.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&u16::try_from(entries.len())?.to_le_bytes());
    output.extend_from_slice(&u16::try_from(entries.len())?.to_le_bytes());
    output.extend_from_slice(&central_size.to_le_bytes());
    output.extend_from_slice(&central_offset.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    fs::write(path, output)?;
    Ok(())
}

fn write_gguf(path: &Path, name: &str) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    push_gguf_string(&mut bytes, "general.alignment")?;
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    push_gguf_string(&mut bytes, "general.architecture")?;
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_string(&mut bytes, "foundation")?;
    push_gguf_string(&mut bytes, name)?;
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&24_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.push(7);
    fs::write(path, bytes)?;
    Ok(())
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) -> Result<(), Box<dyn Error>> {
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut state = u32::MAX;
    for byte in bytes {
        state ^= u32::from(*byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(state & 1)));
        }
    }
    !state
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validation_target() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
        .join("sim-parity/val-model-detection-001.json")
}

#[test]
fn val_model_detection_001() -> Result<(), Box<dyn Error>> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root.join("../..");
    let manifest_path = crate_root.join(FIXTURE_MANIFEST);
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: FixtureManifest = serde_json::from_slice(&manifest_bytes)?;
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.fixture_id, "model-detection-foundation-v1");
    assert!(!manifest.oracle.executed_in_release_tests);
    let oracle_source = repository_root.join(&manifest.oracle.source);
    assert_eq!(
        sha256(&fs::read(oracle_source)?),
        manifest.oracle.source_sha256
    );

    let directory = tempfile::tempdir()?;
    let native_path = directory.path().join("native.safetensors");
    let native_tensors = vec![
        safetensor(
            "model.diffusion_model.input_blocks.0.0.weight",
            "U8",
            vec![320, 4, 1, 1],
            vec![0; 320 * 4],
        ),
        safetensor(
            "model.diffusion_model.out.2.weight",
            "U8",
            vec![4, 320, 1, 1],
            vec![0; 4 * 320],
        ),
        safetensor(
            "model.diffusion_model.label_emb.0.0.weight",
            "U8",
            vec![1, 2816],
            vec![0; 2816],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
            "U8",
            vec![320, 1],
            vec![0; 320],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.1.proj_in.weight",
            "U8",
            vec![1, 1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "U8",
            vec![1, 768],
            vec![0; 768],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.1.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.2.0.op.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.3.0.in_layers.0.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.input_blocks.3.0.out_layers.3.weight",
            "U8",
            vec![640, 1],
            vec![0; 640],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.2.0.in_layers.0.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.2.1.proj_in.weight",
            "U8",
            vec![1, 1, 1, 1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "U8",
            vec![1, 768],
            vec![0; 768],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.1.1.proj_in.weight",
            "U8",
            vec![1, 1, 1, 1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "U8",
            vec![1, 768],
            vec![0; 768],
        ),
        safetensor(
            "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.middle_block.1.proj_in.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
        safetensor(
            "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_q.weight",
            "U8",
            vec![1],
            vec![0],
        ),
    ];
    write_safetensors(
        &native_path,
        BTreeMap::from([("config", r#"{"transformer":{"family":"native"}}"#)]),
        &native_tensors,
    )?;
    let (_native_index, native_store, native_model) = load_model(&native_path, &["safetensors"])?;
    let operations_before_probe = native_store.operations().to_vec();
    fs::remove_file(&native_path)?;
    let native_probe = native_store.family_probe(&native_model, &CancellationToken::default())?;
    assert_eq!(native_store.operations(), operations_before_probe);
    assert_eq!(
        native_probe.storage_dtype("model.diffusion_model.input_blocks.0.0.weight"),
        Some(ModelStorageDType::Tensor(DType::U8))
    );
    assert_eq!(native_probe.format_identities(), vec!["safetensors"]);
    assert_eq!(
        native_probe.metadata().get("config").map(String::as_str),
        Some(r#"{"transformer":{"family":"native"}}"#)
    );
    assert_eq!(
        native_probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    assert_eq!(
        native_probe.consecutive_block_count("model.diffusion_model.input_blocks.{}.")?,
        4
    );
    let native_configuration = native_probe.normalized_configuration()?;
    assert_eq!(native_configuration.kind(), ModelConfigurationKind::Native);
    assert_eq!(
        native_configuration.fact("model_channels"),
        Some(&ModelConfigurationValue::Unsigned(320))
    );
    assert_eq!(
        native_configuration.fact("in_channels"),
        Some(&ModelConfigurationValue::Unsigned(4))
    );
    assert_eq!(
        native_configuration.fact("input_block_count"),
        Some(&ModelConfigurationValue::Unsigned(4))
    );
    assert_eq!(
        native_configuration.fact("out_channels"),
        Some(&ModelConfigurationValue::Unsigned(4))
    );
    assert_eq!(
        native_configuration.fact("num_classes"),
        Some(&ModelConfigurationValue::Text("sequential".to_owned()))
    );
    assert_eq!(
        native_configuration.fact("adm_in_channels"),
        Some(&ModelConfigurationValue::Unsigned(2816))
    );
    assert_eq!(
        native_configuration.fact("num_res_blocks"),
        Some(&ModelConfigurationValue::UnsignedList(vec![1, 1]))
    );
    assert_eq!(
        native_configuration.fact("channel_mult"),
        Some(&ModelConfigurationValue::UnsignedList(vec![1, 2]))
    );
    assert_eq!(
        native_configuration.fact("transformer_depth"),
        Some(&ModelConfigurationValue::UnsignedList(vec![2, 0]))
    );
    assert_eq!(
        native_configuration.fact("transformer_depth_output"),
        Some(&ModelConfigurationValue::UnsignedList(vec![2, 1, 0]))
    );
    assert_eq!(
        native_configuration.fact("transformer_depth_middle"),
        Some(&ModelConfigurationValue::Signed(3))
    );
    assert_eq!(
        native_configuration.fact("context_dim"),
        Some(&ModelConfigurationValue::Unsigned(768))
    );
    assert_eq!(
        native_configuration.fact("use_linear_in_transformer"),
        Some(&ModelConfigurationValue::Boolean(true))
    );
    assert_eq!(
        native_configuration.fact("use_temporal_resblock"),
        Some(&ModelConfigurationValue::Boolean(true))
    );
    assert_eq!(
        native_configuration.fact("use_temporal_attention"),
        Some(&ModelConfigurationValue::Boolean(true))
    );
    assert_eq!(
        native_configuration.fact("disable_temporal_crossattention"),
        Some(&ModelConfigurationValue::Boolean(false))
    );
    let registry = ModelFamilyRegistry::checked(&NATIVE_REGISTRY)?;
    let ModelDetectionOutcome::Registered(native_detection) =
        registry.detect_with_policy(&native_probe, ModelDetectionPolicy::RegisteredOnly)?
    else {
        return Err("registered native detection returned the base fallback".into());
    };
    assert_eq!(native_detection.identity.identifier(), "DetectionNative");

    let native_defaults_path = directory.path().join("native-defaults.safetensors");
    let mut native_default_tensors = vec![safetensor(
        "model.diffusion_model.input_blocks.0.0.weight",
        "U8",
        vec![320, 4, 1, 1],
        vec![0; 320 * 4],
    )];
    for index in 0..5 {
        native_default_tensors.push(safetensor(
            &format!("model.diffusion_model.auxiliary.{index}.weight"),
            "U8",
            vec![1],
            vec![u8::try_from(index)?],
        ));
    }
    write_safetensors(
        &native_defaults_path,
        BTreeMap::new(),
        &native_default_tensors,
    )?;
    let (_native_defaults_index, native_defaults_store, native_defaults_model) =
        load_model(&native_defaults_path, &["safetensors"])?;
    let native_defaults_probe = native_defaults_store
        .family_probe(&native_defaults_model, &CancellationToken::default())?;
    let native_defaults_configuration = native_defaults_probe.normalized_configuration()?;
    assert_eq!(
        native_defaults_configuration.fact("out_channels"),
        Some(&ModelConfigurationValue::Unsigned(4))
    );
    assert_eq!(
        native_defaults_configuration.fact("adm_in_channels"),
        Some(&ModelConfigurationValue::None)
    );
    assert_eq!(
        native_defaults_configuration.fact("transformer_depth_middle"),
        Some(&ModelConfigurationValue::Signed(-2))
    );
    assert_eq!(
        native_defaults_configuration.fact("use_temporal_attention"),
        Some(&ModelConfigurationValue::Boolean(false))
    );

    let partial_path = directory.path().join("partial.safetensors");
    write_safetensors(
        &partial_path,
        BTreeMap::new(),
        &[safetensor(
            "model.diffusion_model.input_blocks.0.0.weight",
            "U8",
            vec![320, 4, 1, 1],
            vec![0; 320 * 4],
        )],
    )?;
    let (_partial_index, partial_store, partial_model) =
        load_model(&partial_path, &["safetensors"])?;
    let partial_probe =
        partial_store.family_probe(&partial_model, &CancellationToken::default())?;
    assert!(matches!(
        registry.detect_with_policy(&partial_probe, ModelDetectionPolicy::RegisteredOnly),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let diffusers_path = directory.path().join("diffusers.safetensors");
    write_safetensors(
        &diffusers_path,
        BTreeMap::new(),
        &[
            safetensor(
                "conv_in.weight",
                "F32",
                vec![320, 4, 1, 1],
                vec![0; 320 * 4 * 4],
            ),
            safetensor("down_blocks.0.resnets.0.weight", "F32", vec![1], vec![0; 4]),
            safetensor("down_blocks.1.resnets.0.weight", "F32", vec![1], vec![0; 4]),
        ],
    )?;
    let (_diffusers_index, diffusers_store, diffusers_model) =
        load_model(&diffusers_path, &["safetensors"])?;
    let diffusers_probe =
        diffusers_store.family_probe(&diffusers_model, &CancellationToken::default())?;
    let diffusers_configuration = diffusers_probe.normalized_configuration()?;
    assert_eq!(
        diffusers_configuration.kind(),
        ModelConfigurationKind::Diffusers
    );
    assert_eq!(diffusers_configuration.unet_prefix(), "");
    assert_eq!(
        diffusers_configuration.fact("model_channels"),
        Some(&ModelConfigurationValue::Unsigned(320))
    );

    let empty_registry = ModelFamilyRegistry::checked(&[])?;
    assert!(matches!(
        empty_registry.detect_with_policy(&diffusers_probe, ModelDetectionPolicy::RegisteredOnly),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let ModelDetectionOutcome::BaseFallback(fallback) = empty_registry
        .detect_with_policy(&diffusers_probe, ModelDetectionPolicy::AllowBaseFallback)?
    else {
        return Err(
            "unregistered diffusers configuration did not use the explicit fallback".into(),
        );
    };
    assert_eq!(fallback.configuration(), &diffusers_configuration);

    let sam3_path = directory.path().join("sam3.safetensors");
    write_safetensors(
        &sam3_path,
        BTreeMap::new(),
        &[
            safetensor("detector.weight", "F16", vec![1], vec![0; 2]),
            safetensor("tracker.weight", "F16", vec![1], vec![0; 2]),
        ],
    )?;
    let (_sam3_index, sam3_store, sam3_model) = load_model(&sam3_path, &["safetensors"])?;
    let sam3_probe = sam3_store.family_probe(&sam3_model, &CancellationToken::default())?;
    let sam3_prefix = sam3_probe.unet_prefix_selection()?;
    assert!(sam3_prefix.is_sam3_top_level());
    assert_eq!(sam3_prefix.prefix(), "");

    let pytorch_path = directory.path().join("restricted.ckpt");
    let pickle = tensor_pickle("model.diffusion_model.weight", "0")?;
    write_stored_zip(
        &pytorch_path,
        &[
            ("archive/data.pkl", pickle.as_slice()),
            ("archive/data/0", &[0, 0, 128, 63]),
            ("archive/version", b"3\n"),
        ],
    )?;
    let (_pytorch_index, pytorch_store, pytorch_model) = load_model(&pytorch_path, &["ckpt"])?;
    let pytorch_probe =
        pytorch_store.family_probe(&pytorch_model, &CancellationToken::default())?;
    assert_eq!(pytorch_probe.format_identities(), vec!["pytorch_archive"]);
    assert_eq!(
        pytorch_probe.storage_dtype("model.diffusion_model.weight"),
        Some(ModelStorageDType::Tensor(DType::F32))
    );

    let gguf_path = directory.path().join("model.gguf");
    write_gguf(&gguf_path, "model.diffusion_model.weight")?;
    let (_gguf_index, gguf_store, gguf_model) = load_model(&gguf_path, &["gguf"])?;
    let gguf_probe = gguf_store.family_probe(&gguf_model, &CancellationToken::default())?;
    assert_eq!(gguf_probe.format_identities(), vec!["gguf"]);
    assert_eq!(
        gguf_probe.storage_dtype("model.diffusion_model.weight"),
        Some(ModelStorageDType::Ggml(24))
    );
    assert_eq!(
        gguf_probe
            .metadata()
            .get("general.architecture")
            .map(String::as_str),
        Some("foundation")
    );

    let malformed_path = directory.path().join("malformed.safetensors");
    fs::write(&malformed_path, [0_u8; 8])?;
    let (mut malformed_index, mut malformed_store, malformed_key, malformed_error) =
        rejected_model_load(&malformed_path, &["safetensors"])?;
    assert!(
        matches!(
            &malformed_error,
            ModelStoreError::Format(ModelFormatError::Invalid { .. })
        ),
        "{malformed_error:?}"
    );
    assert!(matches!(
        malformed_store
            .operations()
            .last()
            .map(|record| &record.stage),
        Some(sim_model::ModelOperationStage::Failed)
    ));
    write_safetensors(
        &malformed_path,
        BTreeMap::new(),
        &[safetensor("repaired.weight", "U8", vec![1], vec![7])],
    )?;
    malformed_index.refresh(&CancellationToken::default())?;
    let repaired = malformed_store.load(
        &malformed_index,
        &malformed_key,
        &CancellationToken::default(),
    )?;
    assert!(repaired.tensors().contains_key("repaired.weight"));
    let unknown_dtype_path = directory.path().join("unknown-dtype.safetensors");
    write_safetensors(
        &unknown_dtype_path,
        BTreeMap::new(),
        &[safetensor("weight", "UNKNOWN", vec![1], vec![0])],
    )?;
    let (_unknown_index, unknown_store, _unknown_key, unknown_error) =
        rejected_model_load(&unknown_dtype_path, &["safetensors"])?;
    assert!(
        matches!(
            &unknown_error,
            ModelStoreError::Format(ModelFormatError::Invalid {
                format: "safetensors",
                ..
            })
        ),
        "{unknown_error:?}"
    );
    assert!(matches!(
        unknown_store
            .operations()
            .last()
            .map(|record| &record.stage),
        Some(sim_model::ModelOperationStage::Failed)
    ));
    let overflow_path = directory.path().join("overflow.safetensors");
    write_safetensors(
        &overflow_path,
        BTreeMap::new(),
        &[safetensor("weight", "U8", vec![u64::MAX, 2], Vec::new())],
    )?;
    let (_overflow_index, overflow_store, _overflow_key, overflow_error) =
        rejected_model_load(&overflow_path, &["safetensors"])?;
    assert!(
        matches!(
            &overflow_error,
            ModelStoreError::Format(ModelFormatError::Overflow(_))
        ),
        "{overflow_error:?}"
    );
    assert!(matches!(
        overflow_store
            .operations()
            .last()
            .map(|record| &record.stage),
        Some(sim_model::ModelOperationStage::Failed)
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let operations_before_cancelled_probe = partial_store.operations().to_vec();
    assert!(matches!(
        partial_store.family_probe(&partial_model, &cancelled),
        Err(ModelStoreError::Cancelled)
    ));
    assert_eq!(
        partial_store.operations(),
        operations_before_cancelled_probe
    );
    let foreign_store = ModelStore::new(ParserLimits::default())?;
    assert!(matches!(
        foreign_store.family_probe(&partial_model, &CancellationToken::default()),
        Err(ModelStoreError::ForeignModelHandle)
    ));

    let tied_registry = ModelFamilyRegistry::checked(&TIED_REGISTRY)?;
    assert!(matches!(
        tied_registry.detect_with_policy(&native_probe, ModelDetectionPolicy::RegisteredOnly),
        Err(ModelFamilyError::AmbiguousDetection { score: 200, .. })
    ));

    let probe_error =
        ModelFamilyProbeError::from(ModelFamilyError::UnknownStorageDType("UNKNOWN".to_owned()));
    assert_eq!(probe_error.kind(), ModelFamilyProbeErrorKind::StorageDType);

    let cases = BTreeMap::from([
        ("ambiguous", true),
        ("base-fallback", true),
        ("cache-and-io-isolation", true),
        ("cancellation", true),
        ("diffusers-prefix", true),
        ("dimension-overflow", true),
        ("gguf-metadata", true),
        ("malformed", true),
        ("no-match", true),
        ("partial", true),
        ("restricted-pytorch-native-prefix", true),
        ("safetensors-native-prefix", true),
        ("sam3-empty-prefix", true),
        ("unknown-dtype", true),
    ]);
    let mut fixture_cases = manifest.cases.clone();
    fixture_cases.sort();
    assert_eq!(
        fixture_cases,
        cases
            .keys()
            .map(|case| (*case).to_owned())
            .collect::<Vec<_>>()
    );
    let target = validation_target();
    fs::create_dir_all(target.parent().ok_or("validation target has no parent")?)?;
    let artifact = json!({
        "schema_version": 1,
        "validation_id": "VAL-MODEL-DETECTION-001",
        "scope": "native parsed-model detection foundation",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "native-rust-parsed-model-metadata",
            "development_oracle_executed": false,
            "network_used": false,
            "external_processes": []
        },
        "fixture_digests": {
            "fixture_manifest": sha256(&manifest_bytes),
            "model_detection_source": manifest.oracle.source_sha256,
            "test_source": sha256(&fs::read(crate_root.join("tests/model_detection.rs"))?)
        },
        "ownership": {
            "artifact_verification": "sim_model::ArtifactIndex",
            "parsing_and_cache": "sim_model::ModelStore",
            "parsed_fact_projection": "sim_model::ModelStore::family_probe",
            "normalization_and_resolution": "sim_model::model_family"
        },
        "boundary_notes": {
            "unknown_storage_dtype": "The bounded format parser rejects an unknown source dtype before ModelStore can cache it; ModelProbe::from_parsed_facts independently maps the same condition to the typed StorageDType projection error.",
            "rejected_detection_side_effects": "Detection consumes only immutable parsed facts and performs no artifact read, tensor allocation, cache mutation, state mapping, build, or publication."
        },
        "summary": {"passed": cases.len(), "failed": 0, "skipped": 0},
        "cases": cases,
        "skipped": []
    });
    let mut artifact_bytes = serde_json::to_vec_pretty(&artifact)?;
    artifact_bytes.push(b'\n');
    fs::write(target, artifact_bytes)?;
    Ok(())
}
