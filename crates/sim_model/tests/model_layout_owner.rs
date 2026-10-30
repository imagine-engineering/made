use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, GENERATED_MODEL_FAMILY_REGISTRATIONS,
    ModelFamilyError, ModelFamilyRegistry, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStore, ParserLimits,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.genre_embedder.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["net.genre_embedder.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["genre_embedder.weight"],
        required_prefixes: &[],
    },
];

#[test]
fn model_store_keys_are_the_only_layout_and_state_plan_authority()
-> Result<(), Box<dyn std::error::Error>> {
    let native = model_store_probe(
        &["model.diffusion_model.genre_embedder.weight"],
        BTreeMap::from([("audio_model", "ace")]),
    )?;
    assert_eq!(
        native.select_layout(LAYOUT_SIGNATURES)?,
        ModelStateLayout::PrefixedNative
    );
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let native_resolved = registry.resolve(&native)?;
    let native_plan_identity = native_resolved
        .state_plan()
        .ok_or("native ACEStep layout did not select a state plan")?
        .identity()
        .to_owned();

    let spoofed = model_store_probe(
        &["model.diffusion_model.genre_embedder.weight"],
        BTreeMap::from([("audio_model", "ace"), ("model_layout", "diffusers")]),
    )?;
    assert_eq!(
        spoofed.select_layout(LAYOUT_SIGNATURES)?,
        ModelStateLayout::PrefixedNative
    );
    let spoofed_resolved = registry.resolve(&spoofed)?;
    assert_eq!(
        spoofed_resolved
            .state_plan()
            .ok_or("spoofed ACEStep layout did not select a state plan")?
            .identity(),
        native_plan_identity
    );

    let standalone = model_store_probe(
        &["net.genre_embedder.weight"],
        BTreeMap::from([("fixture", "standalone")]),
    )?;
    assert_eq!(
        standalone.select_layout(LAYOUT_SIGNATURES)?,
        ModelStateLayout::StandaloneNative
    );
    let diffusers = model_store_probe(
        &["genre_embedder.weight"],
        BTreeMap::from([("fixture", "diffusers")]),
    )?;
    assert_eq!(
        diffusers.select_layout(LAYOUT_SIGNATURES)?,
        ModelStateLayout::Diffusers
    );
    let diffusers_ace = model_store_probe(
        &["genre_embedder.weight"],
        BTreeMap::from([("audio_model", "ace")]),
    )?;
    let diffusers_resolved = registry.resolve(&diffusers_ace)?;
    assert_ne!(
        diffusers_resolved
            .state_plan()
            .ok_or("Diffusers ACEStep layout did not select a state plan")?
            .identity(),
        native_plan_identity
    );
    Ok(())
}

#[test]
fn partial_mixed_unsupported_and_excessive_layout_signatures_fail_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let partial = model_store_probe(
        &["model.diffusion_model.genre_embedder.bias"],
        BTreeMap::from([("audio_model", "ace")]),
    )?;
    assert_no_supported_layout(partial.select_layout(LAYOUT_SIGNATURES));
    assert!(matches!(
        registry.resolve(&partial),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mixed = model_store_probe(
        &[
            "model.diffusion_model.genre_embedder.weight",
            "genre_embedder.weight",
        ],
        BTreeMap::from([("audio_model", "ace")]),
    )?;
    assert!(matches!(
        mixed.select_layout(LAYOUT_SIGNATURES),
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("ambiguously match")
    ));
    assert!(matches!(
        registry.resolve(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("ambiguously match")
    ));

    let unsupported = model_store_probe(
        &["unrelated.weight"],
        BTreeMap::from([("fixture", "unsupported")]),
    )?;
    assert_no_supported_layout(unsupported.select_layout(LAYOUT_SIGNATURES));

    let excessive = [LAYOUT_SIGNATURES[0]; 4];
    assert!(matches!(
        native_probe().select_layout(&excessive),
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("expected 1..=3")
    ));
    Ok(())
}

fn assert_no_supported_layout(result: Result<ModelStateLayout, ModelFamilyError>) {
    assert!(matches!(
        result,
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("no supported layout")
    ));
}

fn native_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "model.diffusion_model.genre_embedder.weight".to_owned(),
            vec![2, 2],
        )]),
        metadata: BTreeMap::new(),
    }
}

fn model_store_probe(
    tensor_names: &[&str],
    metadata: BTreeMap<&str, &str>,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("layout.safetensors");
    write_safetensors(&model_path, tensor_names, metadata)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "layout-owner",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("layout-owner", "layout.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(
    path: &Path,
    tensor_names: &[&str],
    metadata: BTreeMap<&str, &str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::to_value(metadata)?);
    let mut data = Vec::new();
    for tensor_name in tensor_names {
        let start = data.len();
        data.extend_from_slice(&[0_u8; 16]);
        header.insert(
            (*tensor_name).to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let encoded_header = serde_json::to_vec(&header)?;
    let mut file = File::create(path)?;
    file.write_all(&u64::try_from(encoded_header.len())?.to_le_bytes())?;
    file.write_all(&encoded_header)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}
