use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::clip::{
    ClipError, ClipType, TextEncoderModel, detect_text_encoder_model, llama_detect,
    select_clip_architecture, t5xxl_detect,
};
use sim_model::clip_tokenizer::{
    CLIP_TOKENIZER_SOURCE_ROWS, apply_empty_baseline_token_weights, escape_important,
    generate_empty_tokens, parse_parentheses, token_weights, unescape_important,
};
use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ClipBpeTokenizer, GEMMA3_END_OF_TURN_TOKEN,
    GEMMA3_IMAGE_TOKEN, GEMMA4_AUDIO_TOKEN, GEMMA4_IMAGE_TOKEN, GEMMA4_VIDEO_TOKEN, GemmaTokenizer,
    GemmaTokenizerProfile, ModelParsedFacts, ModelParsedTensorFact, ModelProbe, ModelStore,
    ModelTokenizerDescriptor, NativePromptTokenizer, NativeTokenValue, NativeTokenizerError,
    NativeTokenizerFamily, ParserLimits, Qwen2BpeTokenizer, Qwen2PretokenizerProfile,
    SentencePieceTokenizer, TextualInversionEmbedding, TokenizerConfiguration,
    parse_prompt_weights,
};
use sim_tensor::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const TOKENIZER_IMPLEMENTATION_CLOSURE: [(&str, &str); 7] = [
    (
        "crates/sim_model/src/clip.rs",
        "c94607fd822d04d9f3ecfbb3c90628d237c3fd70025858d641fe193506d02a1c",
    ),
    (
        "crates/sim_model/src/clip_tokenizer.rs",
        "a112a4fc8f8fe12463dd9c21eab4b10d631ab62a23552e99467ba32bdd1f418c",
    ),
    (
        "crates/sim_model/src/formats.rs",
        "d3759e9348a00df970f77ba26036e47843d54ede8bc04a7606a2af22d86fba3f",
    ),
    (
        "crates/sim_model/src/model_store.rs",
        "94dcbaa44d4b0fdd76a63876a52b3aac2d015271faab9d3f6965d9cf1c89d743",
    ),
    (
        "crates/sim_model/src/slices/native_diffusion.rs",
        "d372f05ff5b2da3c79d095c8ac1a91dc561187b0b009cc548329a6a58bdde0aa",
    ),
    (
        "crates/sim_runtime/src/native_execution_controller.rs",
        "5328af6fa842afe7042189ffb242d0c7b8aa8bb88780c63669577984777bea5f",
    ),
    (
        "crates/sim_test_support/src/native_diffusion_fixture.rs",
        "7b1a51af4135f6bb05a6a92d97e8711e4955a87d20616c103d2a2b128cfec203",
    ),
];

const CLIP_EXECUTION_SOURCE_ROWS: [&str; 8] = [
    "CLIP",
    "CLIPType",
    "load_clip",
    "TEModel",
    "detect_te_model",
    "t5xxl_detect",
    "llama_detect",
    "load_text_encoder_state_dicts",
];

fn clip_execution_probe(
    tensors: Vec<(&str, Vec<u64>, &str)>,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    Ok(ModelProbe::from_parsed_facts(ModelParsedFacts {
        tensors: tensors
            .into_iter()
            .map(|(name, shape, storage_dtype)| {
                (
                    name.to_owned(),
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: storage_dtype.to_owned(),
                    },
                )
            })
            .collect(),
        formats: Vec::new(),
    })?)
}

fn clip_l_execution_probe() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    clip_execution_probe(vec![(
        "text_model.encoder.layers.0.mlp.fc1.weight",
        vec![1, 2],
        "F32",
    )])
}

fn t5xxl_execution_probe() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    clip_execution_probe(vec![
        (
            "encoder.block.23.layer.1.DenseReluDense.wi_1.weight",
            vec![10_240, 2],
            "F32",
        ),
        ("encoder.final_layer_norm.weight", vec![4_096], "F16"),
        ("encoder.block.0.layer.0.sim_quant", vec![1], "U8"),
    ])
}

fn llama_execution_probe() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    clip_execution_probe(vec![
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![3_584],
            "F32",
        ),
        ("model.layers.0.self_attn.k_proj.weight", vec![2, 2], "F32"),
        ("model.norm.weight", vec![2], "BF16"),
    ])
}

fn execute_clip_execution_catalog_contract(symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    match symbol {
        "CLIP" => {
            let probe = clip_l_execution_probe()?;
            let first =
                select_clip_architecture(ClipType::StableDiffusion, std::slice::from_ref(&probe))?;
            let second =
                select_clip_architecture(ClipType::StableDiffusion, std::slice::from_ref(&probe))?;
            assert_eq!(first.digest(), second.digest());
            assert_eq!(first.digest().len(), 64);
            assert!(matches!(
                select_clip_architecture(ClipType::StableDiffusion, &[]),
                Err(ClipError::InvalidArtifactSet(0))
            ));
        }
        "CLIPType" => {
            assert_eq!(ClipType::ALL.len(), 32);
            for (index, clip_type) in ClipType::ALL.into_iter().enumerate() {
                assert_eq!(usize::from(clip_type.source_ordinal()), index + 1);
            }
        }
        "load_clip" => {
            let selection = select_clip_architecture(
                ClipType::Wan,
                std::slice::from_ref(&t5xxl_execution_probe()?),
            )?;
            assert_eq!(
                selection.tokenizer(),
                "sim.text_encoders.wan.WanT5Tokenizer"
            );
            assert_eq!(selection.clip_model(), "sim.text_encoders.wan.te");
        }
        "TEModel" => {
            assert_eq!(TextEncoderModel::ALL.len(), 35);
            for (index, model) in TextEncoderModel::ALL.into_iter().enumerate() {
                assert_eq!(usize::from(model.source_ordinal()), index + 1);
            }
        }
        "detect_te_model" => {
            let probe = clip_l_execution_probe()?;
            assert_eq!(
                detect_text_encoder_model(&probe.tensor_shapes)?,
                Some(TextEncoderModel::ClipL)
            );
            assert!(matches!(
                detect_text_encoder_model(&BTreeMap::from([(
                    "encoder.block.23.layer.1.DenseReluDense.wi_1.weight".to_owned(),
                    Vec::new(),
                )])),
                Err(ClipError::InvalidDetectorTensor(_))
            ));
        }
        "t5xxl_detect" => {
            let probe = t5xxl_execution_probe()?;
            let configuration =
                t5xxl_detect(std::slice::from_ref(&probe))?.ok_or("T5XXL was not detected")?;
            assert_eq!(configuration.artifact_index(), 0);
            assert_eq!(configuration.weight_dtype(), Some("float16"));
            assert!(configuration.mixed_per_layer_quantization());
            assert_eq!(
                t5xxl_detect(std::slice::from_ref(&clip_l_execution_probe()?))?,
                None
            );
        }
        "llama_detect" => {
            let configuration =
                llama_detect(&[clip_l_execution_probe()?, llama_execution_probe()?])?
                    .ok_or("Llama was not detected")?;
            assert_eq!(configuration.artifact_index(), 1);
            assert_eq!(configuration.weight_dtype(), Some("bfloat16"));
            assert_eq!(
                llama_detect(std::slice::from_ref(&clip_l_execution_probe()?))?,
                None
            );
        }
        "load_text_encoder_state_dicts" => {
            let clip = clip_l_execution_probe()?;
            let t5xxl = t5xxl_execution_probe()?;
            let selected =
                select_clip_architecture(ClipType::Flux, &[clip.clone(), t5xxl.clone()])?;
            assert_eq!(selected.tokenizer(), "sim.text_encoders.flux.FluxTokenizer");
            assert_eq!(selected.clip_model(), "sim.text_encoders.flux.flux_clip");
            let reversed = select_clip_architecture(ClipType::Flux, &[t5xxl, clip.clone()])?;
            assert_ne!(selected.digest(), reversed.digest());
            assert!(matches!(
                select_clip_architecture(
                    ClipType::StableDiffusion,
                    &[clip.clone(), clip.clone(), clip.clone(), clip.clone(), clip],
                ),
                Err(ClipError::InvalidArtifactSet(5))
            ));
        }
        unexpected => return Err(format!("unaccounted CLIP execution symbol {unexpected}").into()),
    }
    Ok(())
}

fn verify_tokenizer_implementation_closure(
    workspace: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for (path, expected) in TOKENIZER_IMPLEMENTATION_CLOSURE {
        let actual = format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?));
        assert_eq!(actual, expected, "tokenizer implementation drifted: {path}");
    }
    Ok(())
}

fn sentencepiece_from_bytes(
    bytes: &[u8],
) -> Result<SentencePieceTokenizer, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("tokenizer.model");
    fs::write(&path, bytes)?;
    let cancellation = CancellationToken::default();
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "tokenizer",
        "tokenizers",
        directory.path(),
        ["model"],
    )?)?;
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("tokenizer", "tokenizer.model")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let vocabulary =
        store.verified_sentencepiece_vocabulary(&index, &loaded, &key, &cancellation)?;
    Ok(SentencePieceTokenizer::from_verified_vocabulary(
        vocabulary,
    )?)
}

fn sentencepiece() -> Result<SentencePieceTokenizer, Box<dyn std::error::Error>> {
    sentencepiece_from_bytes(&sentencepiece_model_bytes())
}

fn sentencepiece_model_bytes() -> Vec<u8> {
    let mut entries: Vec<(String, f32, u64)> = vec![
        ("<unk>".to_owned(), -10.0, 2_u64),
        ("<s>".to_owned(), 0.0, 3),
        ("</s>".to_owned(), 0.0, 3),
    ];
    for index in 3..10 {
        entries.push((format!("<unused{index}>"), -100.0, 5));
    }
    entries.extend([
        ("▁hello".to_owned(), -1.0, 1),
        ("▁world".to_owned(), -1.0, 1),
        ("▁a".to_owned(), -1.0, 1),
        ("b".to_owned(), -1.0, 1),
        ("c".to_owned(), -1.0, 1),
        ("d".to_owned(), -1.0, 1),
        ("e".to_owned(), -1.0, 1),
        ("f".to_owned(), -1.0, 1),
        ("▁ab".to_owned(), 5.0, 1),
    ]);
    while entries.len() < 99 {
        let index = entries.len();
        entries.push((format!("<unused{index}>"), -100.0, 5));
    }
    entries.push(("<image>".to_owned(), 10.0, 4));

    let mut model = Vec::new();
    for (piece, score, piece_type) in entries {
        let mut encoded = Vec::new();
        encoded.push(0x0a);
        push_varint(&mut encoded, piece.len() as u64);
        encoded.extend_from_slice(piece.as_bytes());
        encoded.push(0x15);
        encoded.extend_from_slice(&score.to_le_bytes());
        encoded.push(0x18);
        push_varint(&mut encoded, piece_type);
        model.push(0x0a);
        push_varint(&mut model, encoded.len() as u64);
        model.extend(encoded);
    }
    model
}

fn gemma3_sentencepiece_model_bytes() -> Vec<u8> {
    let mut entries: Vec<(String, f32, u64)> = vec![
        ("<pad>".to_owned(), 0.0, 3_u64),
        ("<eos>".to_owned(), 0.0, 3),
        ("<bos>".to_owned(), 0.0, 3),
        ("<unk>".to_owned(), -10.0, 2),
        ("▁hello".to_owned(), 4.0, 1),
        ("▁world".to_owned(), 4.0, 1),
    ];
    while entries.len() < GEMMA3_END_OF_TURN_TOKEN as usize {
        let index = entries.len();
        entries.push((format!("<unused{index}>"), -100.0, 5));
    }
    entries.push(("<end_of_turn>".to_owned(), 0.0, 4));
    sentencepiece_model_from_entries(entries)
}

fn sentencepiece_model_from_entries(entries: Vec<(String, f32, u64)>) -> Vec<u8> {
    let mut model = Vec::new();
    for (piece, score, piece_type) in entries {
        let mut encoded = Vec::new();
        encoded.push(0x0a);
        push_varint(&mut encoded, piece.len() as u64);
        encoded.extend_from_slice(piece.as_bytes());
        encoded.push(0x15);
        encoded.extend_from_slice(&score.to_le_bytes());
        encoded.push(0x18);
        push_varint(&mut encoded, piece_type);
        model.push(0x0a);
        push_varint(&mut model, encoded.len() as u64);
        model.extend(encoded);
    }
    model
}

fn push_varint(output: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn std::error::Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let function_prefix = format!("def {symbol}(");
    let class_with_bases_prefix = format!("class {symbol}(");
    let class_without_bases_prefix = format!("class {symbol}:");
    let start = lines
        .iter()
        .position(|line| {
            line.starts_with(&function_prefix)
                || line.starts_with(&class_with_bases_prefix)
                || line.starts_with(&class_without_bases_prefix)
        })
        .ok_or_else(|| format!("Python symbol {symbol:?} is missing"))?;
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find_map(|(index, line)| {
            let trimmed = line.trim();
            (!trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !line.starts_with(char::is_whitespace))
            .then_some(index)
        })
        .unwrap_or(lines.len());
    let mut body_end = end;
    while body_end > start + 1 {
        let trimmed = lines[body_end - 1].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            body_end -= 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(lines[start..body_end].concat().as_bytes())
    ))
}

fn configuration(maximum_length: usize) -> TokenizerConfiguration {
    TokenizerConfiguration {
        maximum_length,
        minimum_length: None,
        minimum_padding: None,
        pad_to_maximum_length: true,
        pad_left: false,
        start_token: Some(1),
        end_token: Some(2),
        pad_token: 2,
        maximum_word_length: 8,
        disable_weights: false,
        embedding_width: None,
    }
}

fn qwen_configuration(maximum_length: usize, pad_token: u32) -> TokenizerConfiguration {
    TokenizerConfiguration {
        maximum_length,
        minimum_length: Some(1),
        minimum_padding: None,
        pad_to_maximum_length: false,
        pad_left: false,
        start_token: None,
        end_token: None,
        pad_token,
        maximum_word_length: maximum_length,
        disable_weights: true,
        embedding_width: None,
    }
}

fn qwen_artifacts(profile: &str) -> Result<(String, String, String), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .join("projects/comfy/ComfyUI/sim/text_encoders")
        .join(profile);
    Ok((
        fs::read_to_string(root.join("tokenizer_config.json"))?,
        fs::read_to_string(root.join("vocab.json"))?,
        fs::read_to_string(root.join("merges.txt"))?,
    ))
}

fn qwen25_tokenizer() -> Result<Qwen2BpeTokenizer, Box<dyn std::error::Error>> {
    let (configuration, vocabulary, merges) = qwen_artifacts("qwen25_tokenizer")?;
    Qwen2BpeTokenizer::from_artifacts(
        Qwen2PretokenizerProfile::Qwen2,
        &configuration,
        &vocabulary,
        &merges,
        &CancellationToken::default(),
    )
    .map_err(Into::into)
}

fn qwen35_tokenizer() -> Result<Qwen2BpeTokenizer, Box<dyn std::error::Error>> {
    let (configuration, vocabulary, merges) = qwen_artifacts("qwen35_tokenizer")?;
    Qwen2BpeTokenizer::from_artifacts(
        Qwen2PretokenizerProfile::Qwen35Declared,
        &configuration,
        &vocabulary,
        &merges,
        &CancellationToken::default(),
    )
    .map_err(Into::into)
}

fn numeric_tokens(prompt: &sim_model::NativeTokenizedPrompt) -> Vec<Vec<u32>> {
    prompt
        .sections()
        .iter()
        .map(|section| {
            section
                .tokens()
                .iter()
                .filter_map(|token| match token.value() {
                    NativeTokenValue::Token(value) => Some(*value),
                    NativeTokenValue::Embedding { .. } => None,
                })
                .collect()
        })
        .collect()
}

fn gemma_configuration(minimum_length: usize, embedding_width: usize) -> TokenizerConfiguration {
    TokenizerConfiguration {
        maximum_length: 4_096,
        minimum_length: Some(minimum_length),
        minimum_padding: None,
        pad_to_maximum_length: false,
        pad_left: true,
        start_token: Some(2),
        end_token: None,
        pad_token: 0,
        maximum_word_length: 8,
        disable_weights: true,
        embedding_width: Some(embedding_width),
    }
}

fn gemma4_tokenizer_json(world_score: f64) -> String {
    let mut document: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/tokenizer/gemma4-tokenizer.json"
    ))
    .expect("tracked Gemma4 tokenizer fixture must parse");
    document["model"]["vocab"][6][1] = json!(world_score);
    document.to_string()
}

#[test]
fn gemma_tokenizers_are_profile_checked_left_padded_and_cleanup_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/tokenizer/manifest.json"
    ))?;
    assert_eq!(manifest["source_snapshot"]["version"], "0.27.1");
    assert_eq!(manifest["source_snapshot"]["file_count"], 949);
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(manifest["claims"]["licensed_checkpoint_included"], false);
    assert_eq!(manifest["claims"]["external_server_required"], false);
    let gemma3 = GemmaTokenizer::gemma3(
        sentencepiece_from_bytes(&gemma3_sentencepiece_model_bytes())?,
        &cancellation,
    )?;
    assert_eq!(gemma3.profile(), GemmaTokenizerProfile::Gemma3SentencePiece);
    let gemma3 = NativePromptTokenizer::checked(
        NativeTokenizerFamily::Gemma(gemma3),
        gemma_configuration(8, 3_840),
        BTreeMap::new(),
    )?;
    assert_eq!(
        gemma3.encode_numeric("hello<image_soft_token>world<end_of_turn>", &cancellation,)?,
        vec![
            0,
            0,
            0,
            2,
            4,
            GEMMA3_IMAGE_TOKEN,
            5,
            GEMMA3_END_OF_TURN_TOKEN
        ]
    );
    assert_eq!(
        gemma3.decode_generated(
            &[GEMMA3_END_OF_TURN_TOKEN, 4, GEMMA3_IMAGE_TOKEN, 5],
            &cancellation,
        )?,
        "hello world"
    );
    assert!(matches!(
        gemma3.decode_numeric(&[GEMMA3_IMAGE_TOKEN], false, &cancellation),
        Err(NativeTokenizerError::UnsupportedSpecialTokenDecode(
            GEMMA3_IMAGE_TOKEN
        ))
    ));
    assert_eq!(
        gemma3.gemma_profile(),
        Some(GemmaTokenizerProfile::Gemma3SentencePiece)
    );

    let tokenizer_json = gemma4_tokenizer_json(4.0);
    let gemma4_family = GemmaTokenizer::gemma4_from_tokenizer_json(&tokenizer_json, &cancellation)?;
    assert_eq!(
        gemma4_family.profile(),
        GemmaTokenizerProfile::Gemma4TokenizerJson
    );
    let artifact_digest = gemma4_family.artifact_digest().to_owned();
    let gemma4 = NativePromptTokenizer::checked(
        NativeTokenizerFamily::Gemma(gemma4_family),
        gemma_configuration(1, 2_560),
        BTreeMap::new(),
    )?;
    assert_eq!(
        gemma4.encode_numeric("hello world", &cancellation)?,
        vec![2, 4, 5, 4, 6]
    );
    assert_eq!(
        gemma4.encode_numeric("<|image|>", &cancellation)?,
        vec![2, GEMMA4_IMAGE_TOKEN]
    );
    assert_eq!(
        gemma4.decode_numeric(&[4, 5, 4, 6], false, &cancellation)?,
        "hello world"
    );
    assert_eq!(
        gemma4.decode_generated(&[258885, 5, 258886, 258887, 258888], &cancellation)?,
        "<think>\nhello</think>"
    );
    assert_eq!(
        gemma4.decode_numeric(
            &[GEMMA4_IMAGE_TOKEN, GEMMA4_AUDIO_TOKEN, GEMMA4_VIDEO_TOKEN],
            true,
            &cancellation,
        )?,
        ""
    );
    assert_eq!(
        gemma4.gemma_artifact_digest(),
        Some(artifact_digest.as_str())
    );
    assert!(gemma4.resident_bytes()? > std::mem::size_of_val(&gemma4) as u64);

    let changed =
        GemmaTokenizer::gemma4_from_tokenizer_json(&gemma4_tokenizer_json(3.5), &cancellation)?;
    assert_ne!(changed.artifact_digest(), artifact_digest);
    let first_digest = gemma4.semantic_digest(&cancellation)?;
    let changed = NativePromptTokenizer::checked(
        NativeTokenizerFamily::Gemma(changed),
        gemma_configuration(1, 2_560),
        BTreeMap::new(),
    )?;
    assert_ne!(first_digest, changed.semantic_digest(&cancellation)?);

    let mut wrong_configuration = gemma_configuration(1, 2_560);
    wrong_configuration.pad_left = false;
    assert!(matches!(
        NativePromptTokenizer::checked(
            NativeTokenizerFamily::Gemma(GemmaTokenizer::gemma4_from_tokenizer_json(
                &tokenizer_json,
                &cancellation,
            )?),
            wrong_configuration,
            BTreeMap::new(),
        ),
        Err(NativeTokenizerError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        GemmaTokenizer::gemma4_from_tokenizer_json("{}", &cancellation),
        Err(NativeTokenizerError::InvalidTokenizerConfiguration(_))
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        GemmaTokenizer::gemma4_from_tokenizer_json(&tokenizer_json, &cancelled),
        Err(NativeTokenizerError::Cancellation(_))
    ));
    Ok(())
}

fn write_f32_safetensors(
    path: &Path,
    tensors: &[(&str, Vec<u64>, Vec<f32>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        data.try_reserve(values.len().saturating_mul(4))?;
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_owned(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = Vec::new();
    file.try_reserve(
        8_usize
            .saturating_add(header.len())
            .saturating_add(data.len()),
    )?;
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(path, file)?;
    Ok(())
}

fn write_stored_embedding_zip(
    path: &Path,
    entries: &[(&str, &[u8])],
) -> Result<(), Box<dyn std::error::Error>> {
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

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0_u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

fn nested_string_to_param_pickle(storage_key: &str, elements: u8) -> Vec<u8> {
    let mut bytes = vec![0x80, 0x02, b'}', b'('];
    push_pickle_binunicode(&mut bytes, "string_to_param");
    bytes.extend_from_slice(&[b'}', b'(']);
    push_pickle_binunicode(&mut bytes, "*");
    bytes.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
    push_pickle_binunicode(&mut bytes, "storage");
    bytes.extend_from_slice(b"ctorch\nFloatStorage\n");
    push_pickle_binunicode(&mut bytes, storage_key);
    push_pickle_binunicode(&mut bytes, "cpu");
    bytes.extend_from_slice(&[b'K', elements, b't', b'Q', b'K', 0]);
    bytes.extend_from_slice(&[b'(', b'K', elements, b't']);
    bytes.extend_from_slice(&[b'(', b'K', 1, b't']);
    bytes.extend_from_slice(&[0x89, b'}', b't', b'R', b'u', b'u', b'.']);
    bytes
}

fn push_pickle_binunicode(bytes: &mut Vec<u8>, value: &str) {
    bytes.push(b'X');
    bytes.extend_from_slice(&u32::try_from(value.len()).unwrap_or(u32::MAX).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn rust_sources_below(path: &Path) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let mut pending = vec![path.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("._"))
            {
                continue;
            }
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push((path.display().to_string(), fs::read_to_string(path)?));
            }
        }
    }
    Ok(sources)
}

#[test]
fn weighting_matches_source_parentheses_explicit_values_and_escapes()
-> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_prompt_weights(
        r"plain (nested (deep):2.5) escaped \(literal\)",
        &CancellationToken::default(),
    )?;
    assert_eq!(
        parsed.iter().map(|item| item.text()).collect::<Vec<_>>(),
        ["plain ", "nested ", "deep", " escaped (literal)"]
    );
    let weights = parsed.iter().map(|item| item.weight()).collect::<Vec<_>>();
    assert!((weights[0] - 1.0).abs() < f32::EPSILON);
    assert!((weights[1] - 2.5).abs() < f32::EPSILON);
    assert!((weights[2] - 2.75).abs() < f32::EPSILON);
    assert!((weights[3] - 1.0).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn parser_is_bounded_and_cancellation_is_canonical() {
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        parse_prompt_weights("hello", &cancellation),
        Err(NativeTokenizerError::Cancellation(_))
    ));
    assert!(matches!(
        parse_prompt_weights(
            &"x".repeat(sim_model::MAX_NATIVE_PROMPT_BYTES + 1),
            &CancellationToken::default()
        ),
        Err(NativeTokenizerError::PromptTooLarge(_))
    ));
    let deep = format!("{}x{}", "(".repeat(130), ")".repeat(130));
    assert!(matches!(
        parse_prompt_weights(&deep, &CancellationToken::default()),
        Err(NativeTokenizerError::WeightNestingTooDeep(_))
    ));

    let mut unweighted_configuration = configuration(8);
    unweighted_configuration.disable_weights = true;
    let unweighted = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece().expect("verified vocabulary")),
        unweighted_configuration,
        BTreeMap::new(),
    )
    .expect("valid unweighted tokenizer");
    assert!(matches!(
        unweighted.tokenize(
            &"x".repeat(sim_model::MAX_NATIVE_PROMPT_BYTES + 1),
            &CancellationToken::default(),
        ),
        Err(NativeTokenizerError::PromptTooLarge(_))
    ));
}

#[test]
fn sentencepiece_special_tokens_decode_and_unknown_contract_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = sentencepiece()?;
    let cancellation = CancellationToken::default();
    assert_eq!(
        tokenizer.decode(&[10, 11], false, &cancellation)?,
        "hello world"
    );
    assert_eq!(
        tokenizer.decode(&[10, 99, 11], true, &cancellation)?,
        "hello world"
    );
    assert_eq!(tokenizer.decode(&[10, 0], false, &cancellation)?, "hello�");
    assert!(matches!(
        tokenizer.decode(&[8_888], false, &cancellation),
        Err(NativeTokenizerError::UnknownToken(8_888))
    ));
    Ok(())
}

#[test]
fn sentencepiece_accepts_only_current_model_store_verified_vocabulary()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("tokenizer.model");
    fs::write(&path, sentencepiece_model_bytes())?;
    let cancellation = CancellationToken::default();
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "tokenizer",
        "tokenizers",
        directory.path(),
        ["model"],
    )?)?;
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("tokenizer", "tokenizer.model")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let verified = store.verified_sentencepiece_vocabulary(&index, &loaded, &key, &cancellation)?;
    let tokenizer = SentencePieceTokenizer::from_verified_vocabulary(verified.clone())?;
    assert!(tokenizer.matches_verified_vocabulary(&verified));

    let mut foreign_store = ModelStore::new(ParserLimits::default())?;
    let foreign_loaded = foreign_store.load(&index, &key, &cancellation)?;
    let foreign_verified = foreign_store.verified_sentencepiece_vocabulary(
        &index,
        &foreign_loaded,
        &key,
        &cancellation,
    )?;
    assert!(!tokenizer.matches_verified_vocabulary(&foreign_verified));
    assert!(matches!(
        foreign_store.verified_sentencepiece_vocabulary(&index, &loaded, &key, &cancellation,),
        Err(sim_model::ModelStoreError::ForeignModelHandle)
    ));

    let mut changed = sentencepiece_model_bytes();
    changed.push(0);
    fs::write(path, changed)?;
    assert!(
        store
            .verified_sentencepiece_vocabulary(&index, &loaded, &key, &cancellation)
            .is_err()
    );
    Ok(())
}

#[test]
fn sentencepiece_uses_canonical_scores_types_and_viterbi_segmentation()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(5),
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize("ab", &CancellationToken::default())?;
    assert_eq!(numeric_tokens(&prompt), [vec![1, 18, 2, 2, 2]]);
    Ok(())
}

#[test]
fn decoder_numeric_adapter_has_stable_identity_and_does_not_pad_generation_input()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(5),
        BTreeMap::new(),
    )?;
    assert_eq!(tokenizer.encode_numeric("ab", &cancellation)?, [1, 18, 2]);
    let cloned = tokenizer.clone();
    assert_eq!(
        tokenizer.semantic_digest(&cancellation)?,
        cloned.semantic_digest(&cancellation)?
    );
    assert!(
        !tokenizer
            .decode_numeric(&[18], true, &cancellation)?
            .is_empty()
    );

    let changed = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(6),
        BTreeMap::new(),
    )?;
    assert_ne!(
        tokenizer.semantic_digest(&cancellation)?,
        changed.semantic_digest(&cancellation)?
    );
    Ok(())
}

#[test]
fn qwen2_real_artifacts_preserve_text_added_tokens_and_minimum_padding()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let cases = [
        (
            qwen25_tokenizer()?,
            Qwen2PretokenizerProfile::Qwen2,
            151_643,
            151_655,
            151_667,
            &[9707, 11, 51950, 220, 99489, 61804, 233, 319][..],
        ),
        (
            qwen35_tokenizer()?,
            Qwen2PretokenizerProfile::Qwen35Declared,
            248_044,
            248_056,
            248_068,
            &[9419, 11, 50203, 220, 96748, 59720, 233, 317][..],
        ),
    ];
    for (family, profile, pad, image, think, expected) in cases {
        assert_eq!(family.profile(), profile);
        let tokenizer = NativePromptTokenizer::checked(
            NativeTokenizerFamily::Qwen2ByteBpe(family),
            qwen_configuration(256, pad),
            BTreeMap::new(),
        )?;
        let text = "Hello, café 世界 👋\r\n";
        let encoded = tokenizer.encode_numeric(text, &cancellation)?;
        assert_eq!(encoded, expected);
        assert_eq!(
            tokenizer.decode_numeric(&encoded, false, &cancellation)?,
            text
        );
        assert_eq!(
            tokenizer.encode_numeric("a\u{323}\u{301}", &cancellation)?,
            match profile {
                Qwen2PretokenizerProfile::Qwen2 => &[20229, 53839][..],
                Qwen2PretokenizerProfile::Qwen35Declared => &[19614, 52033][..],
            }
        );
        assert_eq!(
            tokenizer.encode_numeric("cafe\u{301}", &cancellation)?,
            tokenizer.encode_numeric("café", &cancellation)?
        );
        assert_eq!(tokenizer.encode_numeric("", &cancellation)?, [pad]);
        assert_eq!(
            tokenizer.encode_numeric("<|image_pad|><think>", &cancellation)?,
            [image, think]
        );
        assert_eq!(
            tokenizer.decode_numeric(&[image, think], true, &cancellation)?,
            "<think>"
        );
        assert_eq!(
            tokenizer.decode_numeric(&[image, think], false, &cancellation)?,
            "<|image_pad|><think>"
        );
        if profile == Qwen2PretokenizerProfile::Qwen35Declared {
            assert_eq!(
                tokenizer.decode_numeric(&[248_076, think], true, &cancellation)?,
                "<think>"
            );
        }
        assert!(matches!(
            tokenizer.decode_numeric(&[u32::MAX], false, &cancellation),
            Err(NativeTokenizerError::UnknownToken(value)) if value == u32::MAX
        ));
        assert_eq!(tokenizer.semantic_digest(&cancellation)?.len(), 64);
        assert!(tokenizer.resident_bytes()? > 1_000_000);
    }
    Ok(())
}

#[test]
fn qwen2_fixture_manifest_pins_source_artifacts_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/qwen_multimodal/tokenizer/manifest.json"
    ))?;
    assert_eq!(manifest["evidence_mode"], "code-inferred");
    assert!(manifest["runtime_oracle"].is_null());
    assert_eq!(
        manifest["snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    let profiles = manifest["profiles"].as_array().ok_or("profiles")?;
    assert_eq!(profiles.len(), 2);
    let roots = [
        ("qwen25", qwen_artifacts("qwen25_tokenizer")?),
        ("qwen35", qwen_artifacts("qwen35_tokenizer")?),
    ];
    for (profile, (name, (configuration, vocabulary, merges))) in profiles.iter().zip(roots) {
        assert_eq!(profile["name"], name);
        assert_eq!(
            profile["configuration_sha256"],
            format!("{:x}", Sha256::digest(configuration.as_bytes()))
        );
        assert_eq!(
            profile["vocabulary_sha256"],
            format!("{:x}", Sha256::digest(vocabulary.as_bytes()))
        );
        assert_eq!(
            profile["merges_sha256"],
            format!("{:x}", Sha256::digest(merges.as_bytes()))
        );
    }
    Ok(())
}

#[test]
fn qwen2_artifact_admission_and_cancellation_are_typed() -> Result<(), Box<dyn std::error::Error>> {
    let (qwen25_configuration, qwen25_vocabulary, qwen25_merges) =
        qwen_artifacts("qwen25_tokenizer")?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        Qwen2BpeTokenizer::from_artifacts(
            Qwen2PretokenizerProfile::Qwen2,
            &qwen25_configuration,
            &qwen25_vocabulary,
            &qwen25_merges,
            &cancelled,
        ),
        Err(NativeTokenizerError::Cancellation(_))
    ));

    let cancellation = CancellationToken::default();
    assert!(matches!(
        Qwen2BpeTokenizer::from_artifacts(
            Qwen2PretokenizerProfile::Qwen2,
            "{}",
            "{}",
            "",
            &cancellation,
        ),
        Err(NativeTokenizerError::InvalidVocabulary)
    ));
    let (qwen35_configuration, qwen35_vocabulary, qwen35_merges) =
        qwen_artifacts("qwen35_tokenizer")?;
    assert!(matches!(
        Qwen2BpeTokenizer::from_artifacts(
            Qwen2PretokenizerProfile::Qwen2,
            &qwen35_configuration,
            &qwen35_vocabulary,
            &qwen35_merges,
            &cancellation,
        ),
        Err(NativeTokenizerError::InvalidTokenizerConfiguration(_))
    ));
    Ok(())
}

#[test]
fn multi_section_packing_never_silently_truncates_and_preserves_word_ids()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(4),
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize(
        "hello (world) hello (world) hello",
        &CancellationToken::default(),
    )?;
    assert_eq!(prompt.sections().len(), 3);
    assert_eq!(
        numeric_tokens(&prompt),
        vec![vec![1, 10, 11, 2], vec![1, 10, 11, 2], vec![1, 10, 2, 2]]
    );
    let word_ids = prompt
        .sections()
        .iter()
        .flat_map(|section| section.tokens())
        .filter(|token| token.word_id() != 0)
        .map(|token| token.word_id())
        .collect::<Vec<_>>();
    assert_eq!(word_ids, [1, 2, 3, 4, 5]);
    Ok(())
}

#[test]
fn long_words_split_across_sections_with_one_stable_word_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = configuration(5);
    config.maximum_word_length = 2;
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        config,
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize("abcdef", &CancellationToken::default())?;
    assert_eq!(prompt.sections().len(), 2);
    assert_eq!(
        numeric_tokens(&prompt),
        vec![vec![1, 18, 14, 15, 2], vec![1, 16, 17, 2, 2]]
    );
    assert!(
        prompt
            .sections()
            .iter()
            .flat_map(|section| section.tokens())
            .filter(|token| token.word_id() != 0)
            .all(|token| token.word_id() == 1)
    );
    Ok(())
}

#[test]
fn groups_larger_than_empty_section_capacity_make_bounded_progress()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(4),
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize("abcdef", &CancellationToken::default())?;
    assert_eq!(
        numeric_tokens(&prompt),
        vec![vec![1, 18, 14, 2], vec![1, 15, 16, 2], vec![1, 17, 2, 2]]
    );
    assert!(
        prompt
            .sections()
            .iter()
            .flat_map(|section| section.tokens())
            .filter(|token| token.word_id() != 0)
            .all(|token| token.word_id() == 1)
    );
    Ok(())
}

#[test]
fn configurable_left_padding_minimums_and_list_batches_are_checked()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = configuration(8);
    config.pad_to_maximum_length = false;
    config.pad_left = true;
    config.pad_token = 7;
    config.minimum_padding = Some(2);
    config.minimum_length = Some(7);
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        config,
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize("hello", &CancellationToken::default())?;
    assert_eq!(numeric_tokens(&prompt), vec![vec![7, 7, 7, 7, 1, 10, 2]]);
    let batch = tokenizer.tokenize_list(
        &["hello".to_owned(), "world".to_owned()],
        &CancellationToken::default(),
    )?;
    assert_eq!(batch.len(), 2);
    assert!(matches!(
        tokenizer.tokenize_list(&[], &CancellationToken::default()),
        Err(NativeTokenizerError::InvalidBatchSize(0))
    ));

    let mut overflow_config = configuration(5);
    overflow_config.pad_to_maximum_length = false;
    overflow_config.minimum_padding = Some(2);
    overflow_config.minimum_length = Some(7);
    overflow_config.maximum_word_length = 2;
    let overflow_tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        overflow_config,
        BTreeMap::new(),
    )?;
    let overflow =
        overflow_tokenizer.tokenize("hello world hello world", &CancellationToken::default())?;
    assert_eq!(
        numeric_tokens(&overflow),
        vec![vec![1, 10, 11, 10, 2], vec![1, 11, 2, 2, 2, 2, 2]]
    );
    Ok(())
}

#[test]
fn textual_inversion_uses_only_a_canonical_verified_artifact_and_is_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("subject.safetensors");
    write_f32_safetensors(&path, &[("clip_l", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])])?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "embeddings",
        "embeddings",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("embeddings", "subject.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let model = store.load(&index, &key, &cancellation)?;
    let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
    let embedding = TextualInversionEmbedding::from_verified_tensor_payload(
        payload,
        Some("clip_l"),
        2,
        &cancellation,
    )?;
    assert_eq!(embedding.rows().len(), 2);
    assert_eq!(&*embedding.rows()[1], &[3.0, 4.0]);
    let original_scope_embedding = embedding.clone();

    let mut embedding_configuration = configuration(6);
    embedding_configuration.embedding_width = Some(2);
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        embedding_configuration,
        BTreeMap::from([("subject".to_owned(), embedding)]),
    )?;
    let prompt = tokenizer.tokenize("embedding:subject, hello", &cancellation)?;
    let embeddings = prompt.sections()[0]
        .tokens()
        .iter()
        .filter_map(|token| match token.value() {
            NativeTokenValue::Embedding { values, .. } => Some(values.clone()),
            NativeTokenValue::Token(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(&*embeddings[0], &[1.0, 2.0]);
    assert_eq!(prompt.sections()[0].tokens()[1].weight(), 1.0);
    assert_eq!(
        numeric_tokens(&prompt),
        vec![vec![1, 0, 0, 2], vec![1, 10, 2, 2, 2, 2]]
    );
    assert_eq!(
        numeric_tokens(&tokenizer.tokenize("embedding:missing,", &cancellation)?),
        vec![vec![1, 0, 0, 2, 2, 2]]
    );

    let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
    assert!(matches!(
        TextualInversionEmbedding::from_verified_tensor_payload(
            payload,
            Some("clip_l"),
            3,
            &cancellation
        ),
        Err(NativeTokenizerError::InvalidEmbeddingShape { .. })
            | Err(NativeTokenizerError::EmbeddingWidthMismatch { .. })
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        store.verified_tensor_payload(&index, &model, &key, &cancelled),
        Err(sim_model::ModelStoreError::Cancelled)
    ));

    write_f32_safetensors(&path, &[("clip_l", vec![1, 2], vec![7.0, 7.0])])?;
    assert!(
        store
            .verified_tensor_payload(&index, &model, &key, &cancellation)
            .is_err()
    );

    let foreign_directory = tempfile::tempdir()?;
    write_f32_safetensors(
        &foreign_directory.path().join("other.safetensors"),
        &[("clip_l", vec![1, 2], vec![9.0, 9.0])],
    )?;
    let mut foreign_index = ArtifactIndex::default();
    foreign_index.add_root(ArtifactRoot::canonical(
        "other",
        "embeddings",
        foreign_directory.path(),
        ["safetensors"],
    )?)?;
    foreign_index.refresh(&cancellation)?;
    let foreign_key = ArtifactKey::new("other", "other.safetensors")?;
    assert!(matches!(
        store.verified_tensor_payload(&foreign_index, &model, &foreign_key, &cancellation),
        Err(sim_model::ModelStoreError::MissingTensorPayloadSource(_))
    ));
    let mut foreign_store = ModelStore::new(ParserLimits::default())?;
    let foreign_model = foreign_store.load(&foreign_index, &foreign_key, &cancellation)?;
    let foreign_payload = foreign_store.verified_tensor_payload(
        &foreign_index,
        &foreign_model,
        &foreign_key,
        &cancellation,
    )?;
    let foreign_embedding = TextualInversionEmbedding::from_verified_tensor_payload(
        foreign_payload,
        Some("clip_l"),
        2,
        &cancellation,
    )?;
    let mut mixed_configuration = configuration(6);
    mixed_configuration.embedding_width = Some(2);
    assert!(matches!(
        NativePromptTokenizer::checked(
            NativeTokenizerFamily::SentencePiece(sentencepiece()?),
            mixed_configuration,
            BTreeMap::from([
                ("subject".to_owned(), original_scope_embedding),
                ("foreign".to_owned(), foreign_embedding),
            ]),
        ),
        Err(NativeTokenizerError::ArtifactMismatch(_))
    ));
    Ok(())
}

#[test]
fn textual_inversion_selection_priority_and_bundle_order_match_source()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    write_f32_safetensors(
        &directory.path().join("wrapper.safetensors"),
        &[
            ("clip_l", vec![1, 2], vec![9.0, 9.0]),
            ("string_to_param.b", vec![1, 2], vec![2.0, 2.0]),
            ("string_to_param.a", vec![1, 2], vec![1.0, 1.0]),
        ],
    )?;
    write_f32_safetensors(
        &directory.path().join("exact.safetensors"),
        &[
            ("fallback", vec![1, 2], vec![8.0, 8.0]),
            ("clip_l", vec![1, 2], vec![3.0, 3.0]),
        ],
    )?;
    write_f32_safetensors(
        &directory.path().join("bundle.safetensors"),
        &[
            ("fallback", vec![1, 2], vec![8.0, 8.0]),
            ("bundle_emb.1.clip_l", vec![1, 2], vec![5.0, 5.0]),
            ("bundle_emb.0.clip_l", vec![1, 2], vec![4.0, 4.0]),
        ],
    )?;
    fs::write(
        directory.path().join("raw.bin"),
        [1.0_f32, 2.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let nested_pickle = nested_string_to_param_pickle("0", 2);
    let nested_values = [7.0_f32, 6.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_stored_embedding_zip(
        &directory.path().join("nested.ckpt"),
        &[
            ("archive/data.pkl", &nested_pickle),
            ("archive/data/0", &nested_values),
        ],
    )?;
    let cancellation = CancellationToken::default();
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "embeddings",
        "embeddings",
        directory.path(),
        ["safetensors", "ckpt"],
    )?)?;
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;

    let load = |store: &mut ModelStore, name: &str| {
        let key = ArtifactKey::new("embeddings", name)?;
        let model = store.load(&index, &key, &cancellation)?;
        let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
        TextualInversionEmbedding::from_verified_tensor_payload(
            payload,
            Some("clip_l"),
            2,
            &cancellation,
        )
        .map_err(Box::<dyn std::error::Error>::from)
    };
    let wrapper = load(&mut store, "wrapper.safetensors")?;
    assert_eq!(&*wrapper.rows()[0], &[9.0, 9.0]);
    let exact = load(&mut store, "exact.safetensors")?;
    assert_eq!(&*exact.rows()[0], &[3.0, 3.0]);
    let bundle = load(&mut store, "bundle.safetensors")?;
    assert_eq!(bundle.rows().len(), 2);
    assert_eq!(&*bundle.rows()[0], &[5.0, 5.0]);
    assert_eq!(&*bundle.rows()[1], &[4.0, 4.0]);
    let nested = load(&mut store, "nested.ckpt")?;
    assert_eq!(&*nested.rows()[0], &[7.0, 6.0]);
    Ok(())
}

#[test]
fn canonical_sd1_bpe_owner_is_reused_for_source_fixture() -> Result<(), Box<dyn std::error::Error>>
{
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    let tokenizer = ClipBpeTokenizer::from_json_and_merges(
        ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
        &vocabulary,
        &merges,
    )?;
    let family = NativeTokenizerFamily::ClipBpe(tokenizer);
    assert_eq!(
        family.decode(&[320, 1_628], false, &CancellationToken::default())?,
        "a test"
    );
    let prompt = NativePromptTokenizer::checked(family, configuration(6), BTreeMap::new())?
        .tokenize("a test", &CancellationToken::default())?;
    assert_eq!(numeric_tokens(&prompt), vec![vec![1, 320, 1_628, 2, 2, 2]]);

    let punctuation_prompt = NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &vocabulary,
            &merges,
        )?),
        configuration(12),
        BTreeMap::new(),
    )?
    .tokenize("a  test, a", &CancellationToken::default())?;
    let content_word_ids = punctuation_prompt.sections()[0]
        .tokens()
        .iter()
        .filter(|token| token.word_id() != 0)
        .map(|token| token.word_id())
        .collect::<Vec<_>>();
    assert!(content_word_ids.len() >= 4);
    assert!(content_word_ids.iter().all(|word_id| *word_id == 1));
    Ok(())
}

#[test]
fn canonical_sd1_content_api_preserves_long_prompts_while_fixed_adapter_stays_77()
-> Result<(), Box<dyn std::error::Error>> {
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    let prompt = "a ".repeat(100);
    let canonical = sim_model::clip::Sd1Tokenizer::from_json_and_merges(
        ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
        &vocabulary,
        &merges,
    )?;
    assert_eq!(
        canonical
            .encode_content(&prompt, &CancellationToken::default())?
            .len(),
        100
    );
    let fixed = canonical.encode(&prompt, &CancellationToken::default())?;
    assert_eq!(fixed.tokens().len(), sim_model::clip::SD1_CONTEXT_LENGTH);
    assert_eq!(
        fixed.content_tokens(),
        sim_model::clip::SD1_CONTEXT_LENGTH - 2
    );

    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &vocabulary,
            &merges,
        )?),
        configuration(10),
        BTreeMap::new(),
    )?;
    let sectioned = tokenizer.tokenize(&prompt, &CancellationToken::default())?;
    assert_eq!(sectioned.sections().len(), 13);
    assert_eq!(
        sectioned
            .sections()
            .iter()
            .flat_map(|section| section.tokens())
            .filter(|token| matches!(token.value(), NativeTokenValue::Token(320)))
            .count(),
        100
    );
    Ok(())
}

#[test]
fn canonical_sd1_artifact_admission_rejects_noncanonical_id_domains_and_merge_cardinality()
-> Result<(), Box<dyn std::error::Error>> {
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    let mut malformed_vocabulary = serde_json::from_str::<BTreeMap<String, u32>>(&vocabulary)?;
    let (_, token) = malformed_vocabulary
        .iter_mut()
        .find(|(_, token)| **token < sim_model::clip::SD1_START_TOKEN)
        .ok_or("ordinary SD1 token is missing")?;
    *token = u32::try_from(sim_model::clip::SD1_VOCABULARY_SIZE)?;
    let malformed_vocabulary = serde_json::to_string(&malformed_vocabulary)?;
    assert!(
        ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &malformed_vocabulary,
            &merges,
        )
        .is_err()
    );

    let mut truncated_merges = merges.lines().collect::<Vec<_>>();
    truncated_merges
        .pop()
        .ok_or("canonical merge tail is missing")?;
    let truncated_merges = truncated_merges.join("\n");
    assert!(
        ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &vocabulary,
            &truncated_merges,
        )
        .is_err()
    );

    let extended_merges = format!("{merges}__zed_invalid_left __zed_invalid_right\n");
    assert!(
        ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &vocabulary,
            &extended_merges,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn full_weighted_segments_not_lexical_words_define_source_word_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        configuration(8),
        BTreeMap::new(),
    )?;
    let prompt = tokenizer.tokenize("hello,  world", &CancellationToken::default())?;
    let content_word_ids = prompt.sections()[0]
        .tokens()
        .iter()
        .filter(|token| token.word_id() != 0)
        .map(|token| token.word_id())
        .collect::<Vec<_>>();
    assert!(content_word_ids.len() >= 2);
    assert!(content_word_ids.iter().all(|word_id| *word_id == 1));
    let legacy_per_word_ids = (1..=content_word_ids.len() as u64).collect::<Vec<_>>();
    assert_ne!(content_word_ids, legacy_per_word_ids);
    Ok(())
}

#[test]
fn competing_foundational_owners_are_absent_from_the_adapter_source() {
    let source = include_str!("../src/clip_tokenizer.rs");
    for forbidden in [
        "struct CancellationToken",
        "struct ArtifactIndex",
        "struct ArtifactRoot",
        "struct ModelStore",
        "struct CpuBackend",
        "struct ExecutionContext",
        "struct Tensor",
        "canonicalize(",
    ] {
        assert!(!source.contains(forbidden), "duplicate owner: {forbidden}");
    }
    assert!(source.contains("Sd1Tokenizer::from_json_and_merges"));
    assert!(source.contains("VerifiedModelTensorPayload"));
    assert!(!source.contains("VerifiedArtifactFile"));
    assert!(source.contains("MAX_NATIVE_PROMPT_BYTES: usize = crate::clip::SD1_MAX_PROMPT_BYTES"));
    assert!(
        source
            .contains("MAX_NATIVE_WEIGHT_SEGMENTS: usize = crate::clip::SD1_MAX_WEIGHTED_SEGMENTS")
    );
    assert_eq!(source.matches("merge_ranks:").count(), 1);
    assert_eq!(source.matches("byte_decoder:").count(), 1);
    assert!(source.contains("pub struct Qwen2BpeTokenizer {"));
    assert_eq!(source.matches("fn pack(").count(), 1);
    assert!(source.contains("pub struct ClipBpeTokenizer {\n    tokenizer: Sd1Tokenizer,\n}"));
    assert!(!source.contains("UnverifiedEmbeddingTensorRows"));
    assert!(!source.contains("project_verified_embedding_candidates"));
    assert!(!source.contains("concatenate_unverified_bundled_embedding_rows"));
    assert!(!source.contains("select_unverified_named_embedding_rows"));
    assert!(source.contains("VerifiedEmbeddingArchivePayload"));
    assert!(source.contains("VerifiedSentencePieceVocabulary"));
    assert_eq!(
        source
            .matches("pub fn from_verified_tensor_payload(")
            .count(),
        1
    );
    assert!(!source.contains("pub fn from_checked_rows("));
}

fn validate_fallible_allocation_atomicity_and_workspace_absence()
-> Result<(), Box<dyn std::error::Error>> {
    let source = include_str!("../src/clip_tokenizer.rs");
    assert!(source.contains("reserve_tokenizer_values(&mut sections"));
    assert!(source.contains(".try_reserve("));
    assert!(!source.contains(".reserve("));
    for forbidden in [
        "BackendWorkspaceAuthority",
        "CpuWorkspace",
        "ExecutionContext",
        "WorkspaceLease",
    ] {
        assert!(
            !source.contains(forbidden),
            "tokenizer unexpectedly owns canonical workspace state: {forbidden}"
        );
    }

    let mut config = configuration(3);
    config.maximum_word_length = 1;
    config.pad_to_maximum_length = false;
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece()?),
        config,
        BTreeMap::new(),
    )?;
    let oversized = "hello ".repeat(sim_model::MAX_NATIVE_TOKEN_SECTIONS + 1);
    assert!(matches!(
        tokenizer.tokenize(&oversized, &CancellationToken::default()),
        Err(NativeTokenizerError::TooManySections(_))
    ));
    assert_eq!(
        numeric_tokens(&tokenizer.tokenize("hello", &CancellationToken::default())?),
        [vec![1, 10, 2]]
    );
    Ok(())
}

#[test]
fn fallible_allocation_is_atomic_and_tokenization_owns_no_backend_workspace()
-> Result<(), Box<dyn std::error::Error>> {
    validate_fallible_allocation_atomicity_and_workspace_absence()
}

#[test]
fn production_call_scan_finds_no_embedding_or_tokenizer_bypass()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    for (path, source) in rust_sources_below(&workspace.join("crates"))? {
        if !path.contains("/src/") || path.ends_with("crates/sim_model/src/clip_tokenizer.rs") {
            continue;
        }
        for forbidden in [
            "TextualInversionEmbedding::from_verified_rows",
            "TextualInversionEmbedding::from_verified_raw_f32",
            "project_safe_embedding_zip_entries(",
            "SentencePieceTokenizer::from_vocabulary",
            "UnverifiedEmbeddingTensorRows",
            "project_verified_embedding_candidates",
            "concatenate_unverified_bundled_embedding_rows",
            "select_unverified_named_embedding_rows",
            "EmbeddingArchiveEntry",
        ] {
            assert!(
                !source.contains(forbidden),
                "production tokenizer bypass {forbidden:?} remains in {path}"
            );
        }
    }
    Ok(())
}

#[test]
fn production_call_scan_ignores_apple_double_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    fs::write(directory.path().join("source.rs"), "fn source() {}\n")?;
    fs::write(directory.path().join("._source.rs"), [0xff])?;
    assert_eq!(
        rust_sources_below(directory.path())?,
        [(
            directory.path().join("source.rs").display().to_string(),
            "fn source() {}\n".to_owned(),
        )]
    );
    Ok(())
}

#[test]
fn empty_token_and_baseline_weight_projections_match_source_equations()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        generate_empty_tokens(Some(1), Some(2), 7, 5)?,
        [1, 2, 7, 7, 7]
    );
    assert!(matches!(
        generate_empty_tokens(Some(1), Some(2), 7, 1),
        Err(NativeTokenizerError::InvalidEmptyTokenLength(1))
    ));
    let weighted = apply_empty_baseline_token_weights(
        &[vec![3.0, 5.0, 10.0, 14.0]],
        &[1.0, 1.0, 2.0, 2.0],
        &[vec![0.5, 2.0]],
        2,
    )?;
    assert_eq!(weighted, [vec![2.0, 3.0, 18.0, 26.0]]);
    assert!(matches!(
        apply_empty_baseline_token_weights(&[vec![1.0]], &[1.0], &[vec![1.0]], 2),
        Err(NativeTokenizerError::InvalidWeightProjection)
    ));
    Ok(())
}

#[test]
fn canonical_archive_record_bundle_and_named_projections_are_checked()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let values = (0..768)
        .flat_map(|index| (index as f32).to_le_bytes())
        .collect::<Vec<_>>();
    write_stored_embedding_zip(
        &directory.path().join("archive.ckpt"),
        &[("archive/meta", b"bad"), ("archive/data/0", &values)],
    )?;
    write_stored_embedding_zip(
        &directory.path().join("ignored.ckpt"),
        &[("archive/data/0", b"bad")],
    )?;
    write_f32_safetensors(
        &directory.path().join("bundle.safetensors"),
        &[
            ("bundle_emb.1.clip_l", vec![1, 2], vec![1.0, 1.0]),
            ("bundle_emb.0.clip_l", vec![1, 2], vec![0.0, 0.0]),
            ("named", vec![1, 2], vec![2.0, 2.0]),
        ],
    )?;
    let cancellation = CancellationToken::default();
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "embeddings",
        "embeddings",
        directory.path(),
        ["bin", "ckpt", "safetensors"],
    )?)?;
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    assert!(
        store
            .load(
                &index,
                &ArtifactKey::new("embeddings", "raw.bin")?,
                &cancellation,
            )
            .is_err()
    );
    let archive_key = ArtifactKey::new("embeddings", "archive.ckpt")?;
    let projected = store
        .verified_embedding_archive(&index, &archive_key, &cancellation)?
        .ok_or("missing projected ZIP embedding")?;
    assert_eq!(projected.rows().len(), 1);
    assert_eq!(projected.width(), 768);
    let embedding =
        TextualInversionEmbedding::from_verified_archive_payload(projected, &cancellation)?;
    assert_eq!(embedding.rows().len(), 1);
    assert_eq!(embedding.artifact_key(), &archive_key);
    assert!(
        store
            .verified_embedding_archive(
                &index,
                &ArtifactKey::new("embeddings", "ignored.ckpt")?,
                &cancellation,
            )?
            .is_none()
    );
    let bundle_key = ArtifactKey::new("embeddings", "bundle.safetensors")?;
    let model = store.load(&index, &bundle_key, &cancellation)?;
    let payload = store.verified_tensor_payload(&index, &model, &bundle_key, &cancellation)?;
    let bundled = TextualInversionEmbedding::from_verified_tensor_payload(
        payload,
        Some("clip_l"),
        2,
        &cancellation,
    )?;
    assert_eq!(bundled.rows().len(), 2);
    assert_eq!(&*bundled.rows()[0], &[1.0, 1.0]);
    assert_eq!(&*bundled.rows()[1], &[0.0, 0.0]);
    Ok(())
}

#[test]
fn unmatched_parentheses_and_escape_sequences_match_source_edge_behavior()
-> Result<(), Box<dyn std::error::Error>> {
    let unmatched_close = parse_prompt_weights("a)b", &CancellationToken::default())?;
    assert_eq!(unmatched_close.len(), 1);
    assert_eq!(unmatched_close[0].text(), "a)b");
    let unmatched_open = parse_prompt_weights("a(b", &CancellationToken::default())?;
    assert_eq!(unmatched_open.len(), 2);
    assert_eq!(unmatched_open[1].text(), "(b");
    let escaped = parse_prompt_weights(r"\(a:9\)", &CancellationToken::default())?;
    assert_eq!(escaped[0].text(), "(a:9)");
    assert_eq!(escaped[0].weight(), 1.0);
    Ok(())
}

#[test]
fn source_row_closure_is_exact_unique_and_backed_by_valid_and_invalid_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let expected = [
        "gen_empty_tokens",
        "ClipTokenWeightEncoder",
        "parse_parentheses",
        "token_weights",
        "escape_important",
        "unescape_important",
        "safe_load_embed_zip",
        "expand_directory_list",
        "bundled_embed",
        "load_embed",
        "SDTokenizer",
        "SD1Tokenizer",
    ];
    assert_eq!(CLIP_TOKENIZER_SOURCE_ROWS, expected);
    let unique = CLIP_TOKENIZER_SOURCE_ROWS
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), 12);
    let test_source = include_str!("clip_tokenizer.rs");
    for fixture in [
        "empty_token_and_baseline_weight_projections_match_source_equations",
        "weighting_matches_source_parentheses_explicit_values_and_escapes",
        "unmatched_parentheses_and_escape_sequences_match_source_edge_behavior",
        "canonical_archive_record_bundle_and_named_projections_are_checked",
        "multi_section_packing_never_silently_truncates_and_preserves_word_ids",
        "canonical_sd1_bpe_owner_is_reused_for_source_fixture",
    ] {
        assert!(test_source.contains(fixture));
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    verify_tokenizer_implementation_closure(workspace)?;
    validate_fallible_allocation_atomicity_and_workspace_absence()?;
    let source_path = "projects/comfy/ComfyUI/sim/sd1_clip.py";
    let source = fs::read(workspace.join(source_path))?;
    let mut matched = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8) != Some(&"comfy-parity-clip-tokenizer-foundation") {
            continue;
        }
        assert_eq!(fields[2], source_path);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_tokenizer_catalog_contract(fields[3])?;
        execute_invalid_tokenizer_catalog_contract(fields[3])?;
        matched.push(fields[3]);
    }
    assert_eq!(matched, CLIP_TOKENIZER_SOURCE_ROWS);
    Ok(())
}

fn execute_tokenizer_catalog_contract(symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    match symbol {
        "gen_empty_tokens" => {
            assert_eq!(
                generate_empty_tokens(Some(1), Some(2), 7, 5)?,
                [1, 2, 7, 7, 7]
            );
        }
        "ClipTokenWeightEncoder" => {
            assert_eq!(
                apply_empty_baseline_token_weights(
                    &[vec![3.0, 5.0, 10.0, 14.0]],
                    &[1.0, 1.0, 2.0, 2.0],
                    &[vec![0.5, 2.0]],
                    2,
                )?,
                [vec![2.0, 3.0, 18.0, 26.0]]
            );
        }
        "parse_parentheses" => {
            assert_eq!(
                parse_parentheses("plain (nested (value)) tail", &cancellation)?,
                ["plain ", "(nested (value))", " tail"]
            );
        }
        "token_weights" => {
            let weighted = token_weights("plain (weighted:2.0)", 0.5, &cancellation)?;
            assert_eq!(weighted.len(), 2);
            assert_eq!(weighted[0].weight(), 0.5);
            assert_eq!(weighted[1].text(), "weighted");
            assert_eq!(weighted[1].weight(), 2.0);
        }
        "escape_important" => {
            assert_eq!(
                escape_important(r"literal \(value\)", &cancellation)?,
                "literal \0\u{2}value\0\u{1}"
            );
        }
        "unescape_important" => {
            assert_eq!(
                unescape_important("literal \0\u{2}value\0\u{1}", &cancellation)?,
                "literal (value)"
            );
        }
        "safe_load_embed_zip" => {
            let directory = tempfile::tempdir()?;
            let values = (0..768)
                .flat_map(|index| (index as f32).to_le_bytes())
                .collect::<Vec<_>>();
            write_stored_embedding_zip(
                &directory.path().join("subject.ckpt"),
                &[("archive/data/0", &values)],
            )?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["ckpt"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "subject.ckpt")?;
            let store = ModelStore::new(ParserLimits::default())?;
            let rows = store
                .verified_embedding_archive(&index, &key, &cancellation)?
                .ok_or("safe embedding archive projection is missing")?;
            assert_eq!(rows.rows().len(), 1);
            assert_eq!(rows.width(), 768);
        }
        "expand_directory_list" => {
            let directory = tempfile::tempdir()?;
            let nested = directory.path().join("nested");
            fs::create_dir(&nested)?;
            fs::write(nested.join("subject.safetensors"), b"indexed embedding")?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            assert_eq!(
                index
                    .records()
                    .map(|record| record.key.clone())
                    .collect::<Vec<_>>(),
                [ArtifactKey::new(
                    "embeddings",
                    "nested/subject.safetensors"
                )?]
            );
        }
        "bundled_embed" => {
            let directory = tempfile::tempdir()?;
            write_f32_safetensors(
                &directory.path().join("bundle.safetensors"),
                &[("bundle_emb.0.clip_l", vec![1, 2], vec![1.0, 2.0])],
            )?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "bundle.safetensors")?;
            let mut store = ModelStore::new(ParserLimits::default())?;
            let model = store.load(&index, &key, &cancellation)?;
            let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
            assert_eq!(
                TextualInversionEmbedding::from_verified_tensor_payload(
                    payload,
                    Some("clip_l"),
                    2,
                    &cancellation,
                )?
                .rows()
                .len(),
                1
            );
        }
        "load_embed" => {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("subject.safetensors");
            write_f32_safetensors(&path, &[("clip_l", vec![1, 2], vec![1.0, 2.0])])?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "subject.safetensors")?;
            let mut store = ModelStore::new(ParserLimits::default())?;
            let model = store.load(&index, &key, &cancellation)?;
            let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
            let embedding = TextualInversionEmbedding::from_verified_tensor_payload(
                payload,
                Some("clip_l"),
                2,
                &cancellation,
            )?;
            assert_eq!(embedding.rows().len(), 1);
        }
        "SDTokenizer" => {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("subject.safetensors");
            write_f32_safetensors(&path, &[("clip_l", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])])?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "subject.safetensors")?;
            let mut store = ModelStore::new(ParserLimits::default())?;
            let model = store.load(&index, &key, &cancellation)?;
            let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
            let embedding = TextualInversionEmbedding::from_verified_tensor_payload(
                payload,
                Some("clip_l"),
                2,
                &cancellation,
            )?;
            let mut tokenizer_configuration = configuration(6);
            tokenizer_configuration.embedding_width = Some(2);
            let tokenizer = NativePromptTokenizer::checked(
                NativeTokenizerFamily::SentencePiece(sentencepiece()?),
                tokenizer_configuration,
                BTreeMap::from([("subject".to_owned(), embedding)]),
            )?;
            let tokenized = tokenizer.tokenize("embedding:subject, hello", &cancellation)?;
            assert_eq!(
                numeric_tokens(&tokenized),
                [vec![1, 0, 0, 2], vec![1, 10, 2, 2, 2, 2]]
            );
            assert_eq!(
                tokenized
                    .sections()
                    .iter()
                    .flat_map(|section| section.tokens())
                    .filter(|token| matches!(token.value(), NativeTokenValue::Embedding { .. }))
                    .count(),
                2
            );
            let mut overflow_configuration = configuration(5);
            overflow_configuration.pad_to_maximum_length = false;
            overflow_configuration.minimum_padding = Some(2);
            overflow_configuration.minimum_length = Some(7);
            overflow_configuration.maximum_word_length = 2;
            let overflow_tokenizer = NativePromptTokenizer::checked(
                NativeTokenizerFamily::SentencePiece(sentencepiece()?),
                overflow_configuration,
                BTreeMap::new(),
            )?;
            assert_eq!(
                numeric_tokens(
                    &overflow_tokenizer.tokenize("hello world hello world", &cancellation,)?
                ),
                [vec![1, 10, 11, 10, 2], vec![1, 11, 2, 2, 2, 2, 2]]
            );
        }
        "SD1Tokenizer" => {
            let vocabulary = fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
            ))?;
            let merges = fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
            ))?;
            let tokenizer = ClipBpeTokenizer::from_json_and_merges(
                ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
                &vocabulary,
                &merges,
            )?;
            assert_eq!(
                tokenizer.decode(&[320, 1_628], false, &cancellation)?,
                "a test"
            );
        }
        unexpected => {
            return Err(format!("unaccounted tokenizer catalog symbol {unexpected}").into());
        }
    }
    Ok(())
}

fn execute_invalid_tokenizer_catalog_contract(
    symbol: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    match symbol {
        "gen_empty_tokens" => assert!(matches!(
            generate_empty_tokens(Some(1), Some(2), 0, 1),
            Err(NativeTokenizerError::InvalidEmptyTokenLength(1))
        )),
        "ClipTokenWeightEncoder" => assert!(matches!(
            apply_empty_baseline_token_weights(&[vec![1.0]], &[1.0], &[vec![1.0, 1.0]], 1,),
            Err(NativeTokenizerError::InvalidWeightProjection)
        )),
        "parse_parentheses" => {
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            assert!(matches!(
                parse_parentheses("(cancelled)", &cancelled),
                Err(NativeTokenizerError::Cancellation(_))
            ));
        }
        "token_weights" => assert!(matches!(
            token_weights("weighted", f32::NAN, &cancellation),
            Err(NativeTokenizerError::InvalidWeight(value)) if value.is_nan()
        )),
        "escape_important" => assert!(matches!(
            escape_important(
                &"x".repeat(sim_model::MAX_NATIVE_PROMPT_BYTES + 1),
                &cancellation,
            ),
            Err(NativeTokenizerError::PromptTooLarge(_))
        )),
        "unescape_important" => {
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            assert!(matches!(
                unescape_important("\0\u{2}", &cancelled),
                Err(NativeTokenizerError::Cancellation(_))
            ));
        }
        "safe_load_embed_zip" => {
            let directory = tempfile::tempdir()?;
            let mut values = vec![0_u8; 768 * 4];
            values[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            write_stored_embedding_zip(
                &directory.path().join("invalid.ckpt"),
                &[("archive/data/0", &values)],
            )?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["ckpt"],
            )?)?;
            index.refresh(&cancellation)?;
            assert!(
                ModelStore::new(ParserLimits::default())?
                    .verified_embedding_archive(
                        &index,
                        &ArtifactKey::new("embeddings", "invalid.ckpt")?,
                        &cancellation,
                    )
                    .is_err()
            );
        }
        "expand_directory_list" => {
            assert!(matches!(
                ArtifactRoot::canonical("", "embeddings", Path::new("."), ["safetensors"]),
                Err(_)
            ));
        }
        "bundled_embed" => {
            let directory = tempfile::tempdir()?;
            write_f32_safetensors(
                &directory.path().join("invalid-bundle.safetensors"),
                &[("bundle_emb.0.clip_l", vec![1, 2], vec![1.0, 2.0])],
            )?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "invalid-bundle.safetensors")?;
            let mut store = ModelStore::new(ParserLimits::default())?;
            let model = store.load(&index, &key, &cancellation)?;
            let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
            assert!(matches!(
                TextualInversionEmbedding::from_verified_tensor_payload(
                    payload,
                    Some("clip_l"),
                    3,
                    &cancellation,
                ),
                Err(NativeTokenizerError::EmbeddingWidthMismatch { expected: 3 })
            ));
        }
        "load_embed" => {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("wrong-width.safetensors");
            write_f32_safetensors(&path, &[("clip_l", vec![1, 2], vec![1.0, 2.0])])?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "embeddings",
                "embeddings",
                directory.path(),
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("embeddings", "wrong-width.safetensors")?;
            let mut store = ModelStore::new(ParserLimits::default())?;
            let model = store.load(&index, &key, &cancellation)?;
            let payload = store.verified_tensor_payload(&index, &model, &key, &cancellation)?;
            assert!(matches!(
                TextualInversionEmbedding::from_verified_tensor_payload(
                    payload,
                    Some("clip_l"),
                    3,
                    &cancellation,
                ),
                Err(NativeTokenizerError::InvalidEmbeddingShape { .. })
                    | Err(NativeTokenizerError::EmbeddingWidthMismatch { expected: 3 })
            ));
        }
        "SDTokenizer" => {
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            let tokenizer = NativePromptTokenizer::checked(
                NativeTokenizerFamily::SentencePiece(sentencepiece()?),
                configuration(6),
                BTreeMap::new(),
            )?;
            assert!(matches!(
                tokenizer.tokenize("hello", &cancelled),
                Err(NativeTokenizerError::Cancellation(_))
            ));
        }
        "SD1Tokenizer" => {
            let vocabulary = fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
            ))?;
            let merges = fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
            ))?;
            let tokenizer = ClipBpeTokenizer::from_json_and_merges(
                ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
                &vocabulary,
                &merges,
            )?;
            assert!(matches!(
                tokenizer.decode(&[u32::MAX], false, &cancellation),
                Err(NativeTokenizerError::Clip(_))
            ));
        }
        unexpected => return Err(format!("unaccounted tokenizer symbol {unexpected}").into()),
    }
    Ok(())
}

#[test]
fn val_clip_001_tokenizer_rows_execute_and_publish_partial_ledger()
-> Result<(), Box<dyn std::error::Error>> {
    const CONTRACT_TASK: &str = "comfy-parity-clip-tokenizer-foundation";
    const EXECUTION_TASK: &str = "comfy-parity-clip-execution-foundation";
    const OWNER_RESULT_TASK: &str = "comfy-parity-sd1-tokenizer-owner-consolidation";
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    verify_tokenizer_implementation_closure(workspace)?;
    validate_fallible_allocation_atomicity_and_workspace_absence()?;
    let mut contracts = Vec::new();
    let mut symbols = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8).copied() != Some(CONTRACT_TASK) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(source)), fields[5]);
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_tokenizer_catalog_contract(fields[3])?;
        execute_invalid_tokenizer_catalog_contract(fields[3])?;
        symbols.push(fields[3]);
        contracts.push(json!({
            "contract_id": fields[0],
            "task_id": CONTRACT_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": [
                format!("{}:native-tokenizer-valid", fields[0]),
                format!("{}:native-tokenizer-invalid", fields[0]),
            ],
        }));
    }
    assert_eq!(symbols, CLIP_TOKENIZER_SOURCE_ROWS);
    assert_eq!(contracts.len(), CLIP_TOKENIZER_SOURCE_ROWS.len());
    let tokenizer_contract_count = contracts.len();

    let mut execution_symbols = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8).copied() != Some(EXECUTION_TASK) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_clip_execution_catalog_contract(fields[3])?;
        execution_symbols.push(fields[3]);
        contracts.push(json!({
            "contract_id": fields[0],
            "task_id": EXECUTION_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": [
                format!("{}:native-clip-execution-valid", fields[0]),
                format!("{}:native-clip-execution-invalid", fields[0]),
            ],
        }));
    }
    assert_eq!(execution_symbols, CLIP_EXECUTION_SOURCE_ROWS);
    assert_eq!(
        contracts
            .iter()
            .filter(|contract| contract.get("task_id") == Some(&json!(EXECUTION_TASK)))
            .count(),
        CLIP_EXECUTION_SOURCE_ROWS.len()
    );

    let producer_path = "crates/sim_model/tests/clip_tokenizer.rs";
    let producer_sha256 = format!(
        "{:x}",
        Sha256::digest(fs::read(workspace.join(producer_path))?)
    );
    let task_implementations = TOKENIZER_IMPLEMENTATION_CLOSURE
        .iter()
        .map(|(path, _)| {
            Ok(json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            }))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    let tokenizer_task_implementations = task_implementations
        .iter()
        .filter(|implementation| {
            implementation
                .get("path")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|path| {
                    matches!(
                        path,
                        "crates/sim_model/src/clip.rs"
                            | "crates/sim_model/src/clip_tokenizer.rs"
                            | "crates/sim_model/src/formats.rs"
                            | "crates/sim_model/src/model_store.rs"
                    )
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(tokenizer_task_implementations.len(), 4);
    let task_results = BTreeMap::from([
        (
            EXECUTION_TASK,
            json!({
                "status": "passed",
                "passed": CLIP_EXECUTION_SOURCE_ROWS.len(),
                "failed": 0,
                "skipped": 0,
                "case_ids": [
                    "task336:source-provenance-and-eight-contracts",
                    "task336:clip-type-and-text-encoder-model-closure",
                    "task336:detector-configuration-and-order",
                    "task336:one-to-four-artifact-loader-selection",
                    "task336:typed-malformed-and-oversized-rejection",
                ],
                "implementations": [
                    {
                        "path": "crates/sim_model/src/clip.rs",
                        "sha256": format!(
                            "{:x}",
                            Sha256::digest(fs::read(workspace.join(
                                "crates/sim_model/src/clip.rs"
                            ))?)
                        ),
                    }
                ],
            }),
        ),
        (
            OWNER_RESULT_TASK,
            json!({
                "status": "passed",
                "passed": tokenizer_contract_count,
                "failed": 0,
                "skipped": 0,
                "case_ids": [
                    "task512:resource-exhaustion-no-publication",
                    "task512:canonical-workspace-free",
                ],
                "implementations": task_implementations,
            }),
        ),
        (
            CONTRACT_TASK,
            json!({
                "status": "passed",
                "passed": tokenizer_contract_count,
                "failed": 0,
                "skipped": 0,
                "case_ids": [
                    "task348:resource-exhaustion-no-publication",
                    "task348:canonical-workspace-free",
                ],
                "implementations": tokenizer_task_implementations,
            }),
        ),
    ]);
    let artifact_directory = workspace.join("target/sim-parity");
    let artifact_path = artifact_directory.join("val-clip-001.json");
    let previous_artifact = if artifact_path.exists() {
        Some(serde_json::from_slice::<Value>(&fs::read(&artifact_path)?)?)
    } else {
        None
    };
    let mut artifact = json!({
        "schema_version": 1,
        "validation_id": "VAL-CLIP-001",
        "overall_status": "partial",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        },
        "implementation": {
            "path": producer_path,
            "sha256": producer_sha256,
        },
        "task_results": task_results,
        "contracts": contracts,
        "remaining_tasks": [
            "comfy-parity-clip-execution-foundation",
            "comfy-parity-clip-text-transformer-foundation",
            "comfy-parity-clip-vision-foundation",
            "comfy-parity-clip-text-encoder-breadth",
            "comfy-parity-clip-owner-consolidation"
        ],
    });
    if let Some(previous_artifact) = previous_artifact {
        let current_results = artifact
            .get_mut("task_results")
            .and_then(Value::as_object_mut)
            .ok_or("VAL-CLIP-001 task results are missing")?;
        if let Some(previous_results) = previous_artifact
            .get("task_results")
            .and_then(Value::as_object)
        {
            for (task, result) in previous_results {
                if task != CONTRACT_TASK && task != EXECUTION_TASK && task != OWNER_RESULT_TASK {
                    current_results.insert(task.clone(), result.clone());
                }
            }
        }
        let current_contracts = artifact
            .get_mut("contracts")
            .and_then(Value::as_array_mut)
            .ok_or("VAL-CLIP-001 contracts are missing")?;
        if let Some(previous_contracts) =
            previous_artifact.get("contracts").and_then(Value::as_array)
        {
            let mut contract_ids = current_contracts
                .iter()
                .filter_map(|contract| {
                    contract
                        .get("contract_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<BTreeSet<_>>();
            for contract in previous_contracts {
                let task = contract.get("task_id").and_then(Value::as_str);
                if task != Some(CONTRACT_TASK)
                    && task != Some(EXECUTION_TASK)
                    && task != Some(OWNER_RESULT_TASK)
                {
                    let contract_id = contract
                        .get("contract_id")
                        .and_then(Value::as_str)
                        .ok_or("VAL-CLIP-001 prior contract has no contract ID")?;
                    if !contract_ids.insert(contract_id.to_owned()) {
                        let current = current_contracts
                            .iter()
                            .find(|current| {
                                current.get("contract_id").and_then(Value::as_str)
                                    == Some(contract_id)
                            })
                            .ok_or("VAL-CLIP-001 duplicate contract is unavailable")?;
                        assert_eq!(
                            current, contract,
                            "conflicting prior contract {contract_id}"
                        );
                        continue;
                    }
                    current_contracts.push(contract.clone());
                }
            }
        }
    }
    let completed_tasks = artifact
        .get("task_results")
        .and_then(Value::as_object)
        .ok_or("VAL-CLIP-001 task results are missing")?
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    artifact
        .get_mut("remaining_tasks")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 remaining tasks are missing")?
        .retain(|task| {
            task.as_str()
                .is_none_or(|task| !completed_tasks.contains(task))
        });
    let passed = artifact
        .get("task_results")
        .and_then(Value::as_object)
        .ok_or("VAL-CLIP-001 task results are missing")?
        .values()
        .try_fold(0_u64, |total, result| {
            total
                .checked_add(
                    result
                        .get("passed")
                        .and_then(Value::as_u64)
                        .ok_or("VAL-CLIP-001 task result has no passed count")?,
                )
                .ok_or("VAL-CLIP-001 passed count overflowed")
        })?;
    artifact["summary"] = json!({"passed": passed, "failed": 0, "skipped": 0});
    fs::create_dir_all(&artifact_directory)?;
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
}
