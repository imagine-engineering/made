use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
    build_model_family_for_probe, describe_model_family,
    generated_sam3_sim_model_0115 as sam3, generated_sam31_sim_model_0116 as sam31,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write, path::Path};

const DIGEST: &str = "0116011601160116011601160116011601160116011601160116011601160116";
const MEMORY_BYTES: u64 = 80;

#[test]
fn val_model_family_row_001_sam31_source_configuration_provenance_and_discrimination()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sam31::MODEL_FAMILY_IDENTIFIER, "SAM31");
    assert_eq!(sam31::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0116");
    assert_eq!(sam31::MODEL_FAMILY_SOURCE_ORDINAL, 88);
    assert_eq!(sam31::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.SAM3");
    assert_eq!(sam31::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);
    let descriptor = describe_model_family(&sam31::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "LatentFormat");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        sam3::MODEL_FAMILY_REGISTRATION,
        sam31::MODEL_FAMILY_REGISTRATION,
    ])?;
    for (layout, expected) in [
        ("native", sam3::Sam3Layout::SourceNative),
        ("source", sam3::Sam3Layout::SourceCheckpoint),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(layout, DType::F32, true))?;
        let configuration = sam31::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, expected);
        assert!(configuration.propagation_convolutions);
        assert_eq!(configuration.hidden_size, 2);
        assert_eq!(configuration.tracker_layer_count, 1);
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), sam31::MODEL_FAMILY_FEATURE_ID);
        let candidate = &resolved.clip_target().candidates()[0];
        assert_eq!(candidate.tokenizer().identifier(), "sim.text_encoders.sam3_clip.SAM3TokenizerWrapper");
        assert_eq!(candidate.clip_model().target().as_str(), "sim.text_encoders.sam3_clip.SAM3ClipModelWrapper");
    }
    let sam3_probe = ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false))?;
    assert_eq!(registry.resolve(&sam3_probe)?.detection().identity.feature_id(), sam3::MODEL_FAMILY_FEATURE_ID);

    let mut partial = parsed_facts("native", DType::F32, true);
    partial.tensors.remove(sam3::SAM31_PROPAGATION_KEY);
    assert!(sam31::configuration_for_probe(&ModelProbe::from_parsed_facts(partial)?).is_err());
    let mut malformed = parsed_facts("native", DType::F32, true);
    malformed.tensors.get_mut(sam3::SAM31_PROPAGATION_KEY).ok_or("propagation")?.shape = vec![2, 2, 3, 3];
    assert!(matches!(
        sam31::configuration_for_probe(&ModelProbe::from_parsed_facts(malformed)?),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("propagation convolution shape")
    ));
    let mut misleading = parsed_facts("native", DType::F32, true);
    misleading.formats[0].metadata.insert("image_model".into(), "SAM3".into());
    assert_eq!(registry.resolve(&ModelProbe::from_parsed_facts(misleading)?)?.detection().identity.feature_id(), sam31::MODEL_FAMILY_FEATURE_ID);

    verify_provenance_and_catalog()?;
    let source = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/sam31_sim_model_0116.rs"))?;
    for forbidden in ["struct Tensor", "struct ModelStore", "struct ModelProbe", "struct PatchGraph", "std::fs", "unsafe "] {
        assert!(!source.contains(forbidden));
    }
    super::write_model_family_row_artifact(
        sam31::MODEL_FAMILY_FIXTURE,
        sam31::MODEL_FAMILY_FEATURE_ID,
        sam31::MODEL_FAMILY_IDENTIFIER,
        sam31::MODEL_FAMILY_SOURCE_ORDINAL,
        "sam31_comfy_model_0116",
        &[
            "source-provenance-inheritance-and-ownership",
            "source-native-and-source-checkpoint-configuration",
            "sam3-versus-sam31-key-discrimination",
            "typed-inherited-sam3-clip-target",
            "inherited-exact-qkv-and-tracker-transforms",
            "persisted-store-forward-memory-and-dtypes",
            "partial-malformed-unexpected-cancelled-and-oom-failures",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sam31_transform_store_forward_and_failure_contracts()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        sam3::MODEL_FAMILY_REGISTRATION,
        sam31::MODEL_FAMILY_REGISTRATION,
    ])?;
    let source_probe = ModelProbe::from_parsed_facts(parsed_facts("source", DType::F32, true))?;
    let source_resolved = registry.resolve(&source_probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(StreamId::DEFAULT, authority.authorize_workspace(2 * 1024 * 1024)?, &cancellation);
    let source = raw_source_tensors(&backend, &context)?;
    let mapped = source_resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &context), DIGEST, &source)?;
    let model = mapped.component("model").ok_or("model")?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight"], &[1.0,2.0,3.0,4.0])?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.weight"], &[5.0,6.0,7.0,8.0])?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.weight"], &[9.0,10.0,11.0,12.0])?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.bias"], &[1.0,2.0])?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.bias"], &[3.0,4.0])?;
    assert_values(&backend, &context, &model["native.tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.bias"], &[5.0,6.0])?;
    assert!(model.contains_key("native.detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight"));
    assert!(model.contains_key("native.tracker.sam_decoder.transformer.layers.0.mlp.0.weight"));
    assert!(model.contains_key("native.tracker.sam_decoder.transformer.layers.0.mlp.2.weight"));
    assert!(model.contains_key("native.tracker.sam_decoder.transformer.layers.0.norm_final.weight"));
    assert!(!model.keys().any(|key| key.contains("freqs_cis")));
    assert!(mapped.component("text_encoder").ok_or("text")?.contains_key("text_encoder.sam3_clip.transformer.projection.weight"));

    let mut unexpected = source.clone();
    unexpected.insert("python.extension.weight".into(), tensor(&backend, &[1], &[1.0], DType::F32, &context)?);
    let mut unexpected_facts = parsed_facts("source", DType::F32, true);
    unexpected_facts.tensors.insert(
        "python.extension.weight".into(),
        ModelParsedTensorFact { shape: vec![1], storage_dtype: DType::F32.catalog_name().into() },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &context), DIGEST, &unexpected),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["python.extension.weight"]
    ));
    let cancelled = CancellationToken::default(); cancelled.cancel();
    let cancelled_context = backend.execution_context(StreamId::DEFAULT, authority.authorize_workspace(2 * 1024 * 1024)?, &cancelled);
    assert!(matches!(
        source_resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &cancelled_context), DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));

    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    let native = native_source_tensors(&backend, &context, DType::F32)?;
    let weights = resolved.map_primary_weights(&ModelStateTransaction::new(&backend, &context), DIGEST, &native)?;
    let model = build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, MEMORY_BYTES))?;
    assert_eq!(model.memory_estimate().total_bytes, MEMORY_BYTES);
    let input = tensor(&backend, &[1,2], &[0.25,-0.5], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    let output = &checkpoints.iter().find(|checkpoint| checkpoint.name == "segmentation_embedding").ok_or("output")?.tensor;
    assert_values_close(&backend, &context, output, &[0.24491866,-0.46211717])?;
    for dtype in [DType::F16, DType::Bf16] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts("native", dtype, true))?;
        let resolved = registry.resolve(&probe)?;
        let source = native_source_tensors(&backend, &context, dtype)?;
        let weights = resolved.map_primary_weights(&ModelStateTransaction::new(&backend, &context), DIGEST, &source)?;
        build_model_family_for_probe(&registry, &probe, weights, options(dtype, MEMORY_BYTES))?;
    }
    let weights = resolved.map_primary_weights(&ModelStateTransaction::new(&backend, &context), DIGEST, &native)?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, MEMORY_BYTES - 1)),
        Err(ModelFamilyError::OutOfMemory { required: MEMORY_BYTES, .. })
    ));
    let weights = resolved.map_primary_weights(&ModelStateTransaction::new(&backend, &context), DIGEST, &native)?;
    let mut metal = options(DType::F32, MEMORY_BYTES); metal.device = DeviceKind::Metal;
    assert!(matches!(build_model_family_for_probe(&registry, &probe, weights, metal), Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))));
    Ok(())
}

fn parsed_facts(layout: &str, dtype: DType, propagation: bool) -> ModelParsedFacts {
    let mut shapes = native_shapes(propagation);
    if layout == "source" {
        for key in [
            "tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight",
            "tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.weight",
            "tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.weight",
            "tracker.sam_decoder.transformer.layers.0.mlp.0.weight",
            "tracker.sam_decoder.transformer.layers.0.norm_final.weight",
        ] { shapes.remove(key); }
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_weight".into(), vec![6,2]);
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_bias".into(), vec![6]);
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.mlp.lin1.weight".into(), vec![2,2]);
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.mlp.lin2.weight".into(), vec![2,2]);
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.norm_final_attn.weight".into(), vec![2]);
        shapes.insert("tracker.model.sam_decoder.transformer.layers.0.attn.freqs_cis".into(), vec![1]);
        shapes.insert("detector.backbone.language_backbone.encoder.projection.weight".into(), vec![2, 2]);
        shapes.insert("detector.backbone.language_backbone.resizer.weight".into(), vec![1]);
    }
    ModelParsedFacts {
        tensors: shapes.into_iter().map(|(key,shape)| (key, ModelParsedTensorFact { shape, storage_dtype: dtype.catalog_name().into() })).collect(),
        formats: vec![ModelParsedFormatFact { identity: "safetensors".into(), metadata: BTreeMap::new() }],
    }
}

fn native_shapes(propagation: bool) -> BTreeMap<String, Vec<u64>> {
    let mut shapes = BTreeMap::from([
        ("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight".into(), vec![6,2]),
        ("detector.transformer.decoder.query_embed.weight".into(), vec![2,2]),
        ("tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight".into(), vec![2,2]),
        ("tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.weight".into(), vec![2,2]),
        ("tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.weight".into(), vec![2,2]),
        ("tracker.sam_decoder.transformer.layers.0.mlp.0.weight".into(), vec![2,2]),
        ("tracker.sam_decoder.transformer.layers.0.norm_final.weight".into(), vec![2]),
    ]);
    if propagation { shapes.insert(sam3::SAM31_PROPAGATION_KEY.into(), vec![2,2,1,1]); }
    shapes
}

fn native_source_tensors(backend: &CpuBackend, context: &ExecutionContext<'_>, dtype: DType) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    native_shapes(true).into_iter().map(|(key,shape)| {
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key.ends_with("query_embed.weight") || key.ends_with("q_proj.weight") || key.ends_with("k_proj.weight") || key.ends_with("v_proj.weight") || key.ends_with("mlp.0.weight") || key == sam3::SAM31_PROPAGATION_KEY {
            (0..elements).map(|index| if index % 3 == 0 {1.0} else {0.0}).collect::<Vec<_>>()
        } else if key.ends_with("norm_final.weight") { vec![1.0; elements] } else { vec![0.0; elements] };
        Ok((key, tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
}

fn raw_source_tensors(backend: &CpuBackend, context: &ExecutionContext<'_>) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    Ok(BTreeMap::from([
        ("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight".into(), tensor(backend,&[6,2],&[0.0;12],DType::F32,context)?),
        (sam3::SAM31_PROPAGATION_KEY.into(), tensor(backend,&[2,2,1,1],&[1.0,0.0,0.0,1.0],DType::F32,context)?),
        ("detector.transformer.decoder.query_embed.weight".into(), tensor(backend,&[2,2],&[1.0,0.0,0.0,1.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_weight".into(), tensor(backend,&[6,2],&[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_bias".into(), tensor(backend,&[6],&[1.0,2.0,3.0,4.0,5.0,6.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.mlp.lin1.weight".into(), tensor(backend,&[2,2],&[1.0,0.0,0.0,1.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.mlp.lin2.weight".into(), tensor(backend,&[2,2],&[1.0,0.0,0.0,1.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.norm_final_attn.weight".into(), tensor(backend,&[2],&[1.0,1.0],DType::F32,context)?),
        ("tracker.model.sam_decoder.transformer.layers.0.attn.freqs_cis".into(), tensor(backend,&[1],&[1.0],DType::F32,context)?),
        ("detector.backbone.language_backbone.encoder.projection.weight".into(), tensor(backend,&[2,2],&[1.0,0.0,0.0,1.0],DType::F32,context)?),
        ("detector.backbone.language_backbone.resizer.weight".into(), tensor(backend,&[1],&[1.0],DType::F32,context)?),
    ]))
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?; let path = directory.path().join("sam31.safetensors");
    write_safetensors(&path, parsed_facts("native",DType::F32,true))?;
    let mut index = ArtifactIndex::default(); index.add_root(ArtifactRoot::canonical("sam31-row","checkpoints",directory.path(),["safetensors"])?)?;
    let cancellation = CancellationToken::default(); index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sam31-row","sam31.safetensors")?; let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index,&key,&cancellation)?; Ok(store.family_probe(&loaded,&cancellation)?)
}
fn write_safetensors(path: &Path, facts: ModelParsedFacts) -> Result<(), Box<dyn std::error::Error>> {
    let mut header=serde_json::Map::new(); let mut data=Vec::new();
    for (key,tensor) in facts.tensors { let start=data.len(); let elements=tensor.shape.iter().try_fold(1_u64,|a,b|a.checked_mul(*b).ok_or("overflow"))?; for _ in 0..elements {data.extend_from_slice(&0.0_f32.to_le_bytes());} header.insert(key,serde_json::json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]})); }
    let header=serde_json::to_vec(&header)?; let mut file=std::fs::File::create(path)?; file.write_all(&u64::try_from(header.len())?.to_le_bytes())?; file.write_all(&header)?; file.write_all(&data)?; Ok(())
}
fn tensor(backend:&CpuBackend,shape:&[u64],values:&[f32],dtype:DType,context:&ExecutionContext<'_>)->Result<Tensor,Box<dyn std::error::Error>>{Ok(tensor_from_f32_with_context_exact_native(backend,shape,values,dtype,backend.device(),context)?)}
fn options(dtype:DType,memory_budget_bytes:u64)->NativeFamilyBuildOptions{NativeFamilyBuildOptions{dtype,device:DeviceKind::Cpu,activation_elements:2,memory_budget_bytes,allow_unexpected_weights:false}}
fn assert_values(backend:&CpuBackend,context:&ExecutionContext<'_>,tensor:&Tensor,expected:&[f32])->Result<(),Box<dyn std::error::Error>>{assert_eq!(tensor_to_f32_with_context_exact_native(backend,tensor,context)?,expected);Ok(())}
fn assert_values_close(backend:&CpuBackend,context:&ExecutionContext<'_>,tensor:&Tensor,expected:&[f32])->Result<(),Box<dyn std::error::Error>>{let actual=tensor_to_f32_with_context_exact_native(backend,tensor,context)?;for(a,e)in actual.iter().zip(expected){assert!((a-e).abs()<=1.0e-5);}Ok(())}
fn verify_provenance_and_catalog()->Result<(),Box<dyn std::error::Error>>{
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."); let provenance:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("crates/sim_test_support/fixtures/models").join(sam31::MODEL_FAMILY_FIXTURE).join("provenance.json"))?)?;
    assert_eq!(sha256(provenance["source_projection"].as_str().ok_or("projection")?.as_bytes()),sam31::MODEL_FAMILY_PROJECTION_SHA256); for source in provenance["source_files"].as_array().ok_or("sources")?{assert_eq!(sha256(&std::fs::read(root.join(source["path"].as_str().ok_or("path")?))?),source["sha256"]);}
    let catalog:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("crates/sim_model/catalog/model-families-v1.json"))?)?;let row=catalog["models"].as_array().ok_or("models")?.iter().find(|row|row["feature_id"]==sam31::MODEL_FAMILY_FEATURE_ID).ok_or("row")?;assert_eq!(row["source_ordinal"],88);assert_eq!(row["static"]["unet_config"]["value"]["image_model"],"SAM31");assert_eq!(row["inheritance_chain"][1],"SAM3");Ok(())
}
fn sha256(bytes:&[u8])->String{format!("{:x}",Sha256::digest(bytes))}
