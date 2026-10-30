use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    ALLOWED_PICKLE_TARGETS, ArtifactAvailability, ArtifactChangeKind, ArtifactIndex, ArtifactKey,
    ArtifactRoot, ModelFormat, ModelFormatError, ModelStore, PARSER_LIMITS_VERSION, ParserLimits,
    RESTRICTED_PICKLE_ALLOWLIST_VERSION, RESTRICTED_PICKLE_DECODED_ALLOCATION_MULTIPLIER,
    RestrictedPickleError, add_safe_globals_exact_native, parse_model_file,
    parse_restricted_pickle, parse_restricted_pickle_cancellable,
};
use sim_tensor::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

#[test]
fn allowlist_fixture_is_versioned_unique_and_explicit() {
    assert_eq!(RESTRICTED_PICKLE_ALLOWLIST_VERSION, 1);
    assert_eq!(RESTRICTED_PICKLE_DECODED_ALLOCATION_MULTIPLIER, 8);
    assert_eq!(PARSER_LIMITS_VERSION, 1);
    let mut targets = BTreeSet::new();
    for entry in ALLOWED_PICKLE_TARGETS {
        assert!(!entry.target.is_empty());
        assert!(entry.global);
        assert!(entry.global || entry.reduce || entry.build);
        assert!(targets.insert(entry.target));
        if entry.build {
            assert!(entry.reduce);
        }
    }
}

#[test]
fn task_59_safe_globals_admission_is_immutable_and_fail_closed() -> Result<(), Box<dyn Error>> {
    let before = ALLOWED_PICKLE_TARGETS
        .iter()
        .map(|target| target.target)
        .collect::<Vec<_>>();
    let cancellation = CancellationToken::default();
    let admission = add_safe_globals_exact_native(
        &[
            "torch.nn.parameter.Parameter",
            "torch.nn.parameter.Parameter",
            "collections.OrderedDict",
        ],
        &cancellation,
    )?;
    assert_eq!(
        admission.targets(),
        &["collections.OrderedDict", "torch.nn.parameter.Parameter"]
    );
    assert!(add_safe_globals_exact_native(&["subprocess.Popen"], &cancellation).is_err());
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(add_safe_globals_exact_native(&["collections.OrderedDict"], &cancelled).is_err());
    assert_eq!(
        before,
        ALLOWED_PICKLE_TARGETS
            .iter()
            .map(|target| target.target)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn every_global_target_is_data_only_and_unknown_targets_fail_closed() {
    for entry in ALLOWED_PICKLE_TARGETS {
        let (module, name) = entry.target.rsplit_once('.').unwrap_or((entry.target, ""));
        if name.is_empty() {
            continue;
        }
        let mut bytes = vec![0x80, 0x02, b'c'];
        bytes.extend_from_slice(module.as_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(b"\n.");
        assert!(
            parse_restricted_pickle(&bytes, &ParserLimits::default()).is_ok(),
            "{}",
            entry.target
        );
        if entry.reduce {
            let mut reduce = bytes;
            reduce.truncate(reduce.len().saturating_sub(1));
            reduce.extend_from_slice(b")R.");
            assert!(
                parse_restricted_pickle(&reduce, &ParserLimits::default()).is_ok(),
                "REDUCE {}",
                entry.target
            );
            if entry.build {
                reduce.truncate(reduce.len().saturating_sub(1));
                reduce.extend_from_slice(b"}b.");
                assert!(
                    parse_restricted_pickle(&reduce, &ParserLimits::default()).is_ok(),
                    "BUILD {}",
                    entry.target
                );
            }
        }
    }
    let hostile =
        parse_restricted_pickle(b"\x80\x02csubprocess\nPopen\n.", &ParserLimits::default());
    assert!(matches!(
        hostile,
        Err(RestrictedPickleError::ForbiddenTarget {
            operation: "GLOBAL",
            ..
        })
    ));
    for opcode in [0x81, 0x82, 0x83, 0x84, 0x92, 0x97, 0x98] {
        let bytes = [opcode, b'.'];
        assert!(matches!(
            parse_restricted_pickle(&bytes, &ParserLimits::default()),
            Err(RestrictedPickleError::ForbiddenOpcode { .. })
        ));
    }
}

#[test]
fn pytorch_parameter_and_quantized_archive_fixtures_are_data_only() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let limits = ParserLimits::default();
    let cancellation = CancellationToken::default();

    for target in [
        "torch.nn.parameter.Parameter",
        "torch._utils._rebuild_parameter",
        "torch._utils._rebuild_parameter_with_state",
    ] {
        let pickle = parameter_pickle("weight", "0", target);
        let path = directory.path().join(format!(
            "{}.ckpt",
            target
                .rsplit('.')
                .next()
                .ok_or("parameter target is empty")?
        ));
        write_stored_zip(
            &path,
            &[
                ("archive/data.pkl", pickle.as_slice()),
                ("archive/data/0", &[0, 0, 128, 63]),
            ],
        )?;
        let parsed = parse_model_file(&path, &limits, &cancellation)?;
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].name, "weight");
        assert_eq!(parsed.tensors[0].storage.length(), 4);
    }

    let quantized = quantized_pickle("weight", "0", 0.25);
    let quantized_path = directory.path().join("quantized.ckpt");
    write_stored_zip(
        &quantized_path,
        &[
            ("archive/data.pkl", quantized.as_slice()),
            ("archive/data/0", &[0, 1, 2, 3]),
        ],
    )?;
    let parsed = parse_model_file(&quantized_path, &limits, &cancellation)?;
    assert_eq!(parsed.tensors.len(), 1);
    assert_eq!(parsed.tensors[0].data_type, "torch.QInt8Storage");
    assert_eq!(parsed.tensors[0].shape, [4]);

    let hostile_quantized = quantized_pickle("weight", "0", f64::NAN);
    let hostile_path = directory.path().join("hostile-quantized.ckpt");
    write_stored_zip(
        &hostile_path,
        &[
            ("archive/data.pkl", hostile_quantized.as_slice()),
            ("archive/data/0", &[0, 1, 2, 3]),
        ],
    )?;
    assert!(parse_model_file(&hostile_path, &limits, &cancellation).is_err());
    Ok(())
}

#[test]
fn val_model_format_001() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let limits = ParserLimits::default();
    let cancellation = CancellationToken::default();
    let mut cases = BTreeMap::new();
    let mut fixture_digests = BTreeMap::new();

    let safetensors = directory.path().join("model.safetensors");
    write_safetensors(&safetensors, "weight", "F32", &[2], &[0; 8])?;
    fixture_digests.insert("safetensors", sha256(&fs::read(&safetensors)?));
    let parsed = parse_model_file(&safetensors, &limits, &cancellation)?;
    assert_eq!(parsed.format, ModelFormat::Safetensors);
    assert_eq!(parsed.tensors[0].storage.length(), 8);
    cases.insert("valid_safetensors", true);

    let pytorch = directory.path().join("model.ckpt");
    let pickle = tensor_pickle("weight", "0");
    write_stored_zip(
        &pytorch,
        &[
            ("archive/data.pkl", pickle.as_slice()),
            ("archive/data/0", &[0, 0, 128, 63]),
            ("archive/version", b"3\n"),
        ],
    )?;
    fixture_digests.insert("pytorch", sha256(&fs::read(&pytorch)?));
    let parsed = parse_model_file(&pytorch, &limits, &cancellation)?;
    assert_eq!(parsed.format, ModelFormat::PytorchArchive);
    assert_eq!(parsed.tensors[0].name, "weight");
    assert_eq!(parsed.tensors[0].storage.length(), 4);
    cases.insert("valid_restricted_pytorch", true);

    let parameter = directory.path().join("parameter.ckpt");
    let parameter_bytes = parameter_pickle(
        "parameter",
        "0",
        "torch._utils._rebuild_parameter_with_state",
    );
    write_stored_zip(
        &parameter,
        &[
            ("archive/data.pkl", parameter_bytes.as_slice()),
            ("archive/data/0", &[0, 0, 128, 63]),
        ],
    )?;
    fixture_digests.insert("pytorch_parameter", sha256(&fs::read(&parameter)?));
    assert_eq!(
        parse_model_file(&parameter, &limits, &cancellation)?.tensors[0].name,
        "parameter"
    );
    let quantized = directory.path().join("quantized.ckpt");
    let quantized_bytes = quantized_pickle("quantized", "0", 0.25);
    write_stored_zip(
        &quantized,
        &[
            ("archive/data.pkl", quantized_bytes.as_slice()),
            ("archive/data/0", &[0, 1, 2, 3]),
        ],
    )?;
    fixture_digests.insert("pytorch_quantized", sha256(&fs::read(&quantized)?));
    assert_eq!(
        parse_model_file(&quantized, &limits, &cancellation)?.tensors[0].data_type,
        "torch.QInt8Storage"
    );
    let hostile_quantized = directory.path().join("hostile-quantized.ckpt");
    let hostile_quantized_pickle = quantized_pickle("quantized", "0", f64::NAN);
    write_stored_zip(
        &hostile_quantized,
        &[
            ("archive/data.pkl", hostile_quantized_pickle.as_slice()),
            ("archive/data/0", &[0, 1, 2, 3]),
        ],
    )?;
    assert!(parse_model_file(&hostile_quantized, &limits, &cancellation).is_err());
    cases.insert("parameter_and_quantized_rebuilds", true);

    let pytorch_zip64 = directory.path().join("model-zip64.ckpt");
    write_stored_zip64(
        &pytorch_zip64,
        &[
            ("archive/data.pkl", pickle.as_slice()),
            ("archive/data/0", &[0, 0, 128, 63]),
            ("archive/version", b"3\n"),
        ],
    )?;
    fixture_digests.insert("pytorch_zip64", sha256(&fs::read(&pytorch_zip64)?));
    let parsed = parse_model_file(&pytorch_zip64, &limits, &cancellation)?;
    assert_eq!(parsed.format, ModelFormat::PytorchArchive);
    assert_eq!(parsed.tensors[0].storage.length(), 4);
    cases.insert("valid_zip64_restricted_pytorch", true);

    let gguf = directory.path().join("model.gguf");
    write_gguf(&gguf)?;
    fixture_digests.insert("gguf", sha256(&fs::read(&gguf)?));
    let parsed = parse_model_file(&gguf, &limits, &cancellation)?;
    assert_eq!(parsed.format, ModelFormat::Gguf);
    assert_eq!(parsed.tensors[0].name, "weight");
    assert_eq!(parsed.tensors[0].storage.length(), 1);
    cases.insert("valid_gguf", true);

    let yaml = directory.path().join("extra_model_paths.yaml");
    fs::write(
        &yaml,
        "zed:\n  base_path: /models\n  is_default: true\n  checkpoints: |\n    checkpoints\n    imported\n",
    )?;
    let parsed = parse_model_file(&yaml, &limits, &cancellation)?;
    assert_eq!(parsed.format, ModelFormat::YamlConfig);
    cases.insert("bounded_yaml_config", true);

    let tokenizer = directory.path().join("tokenizer.json");
    fs::write(&tokenizer, br#"{"model":{"type":"BPE","vocab":{"a":0}}}"#)?;
    assert_eq!(
        parse_model_file(&tokenizer, &limits, &cancellation)?.format,
        ModelFormat::JsonTokenizer
    );
    let sentencepiece = directory.path().join("tokenizer.model");
    fs::write(
        &sentencepiece,
        [
            0x0a, 0x0a, 0x0a, 0x01, b'a', 0x15, 0x00, 0x00, 0x00, 0x00, 0x18, 0x01,
        ],
    )?;
    assert_eq!(
        parse_model_file(&sentencepiece, &limits, &cancellation)?.format,
        ModelFormat::SentencePiece
    );
    let tiktoken = directory.path().join("vocab.tiktoken");
    fs::write(&tiktoken, b"YQ== 0\nYg== 1\n")?;
    assert_eq!(
        parse_model_file(&tiktoken, &limits, &cancellation)?.format,
        ModelFormat::Tiktoken
    );
    cases.insert("bounded_tokenizers", true);

    let truncated = directory.path().join("truncated.safetensors");
    fs::write(&truncated, 100_u64.to_le_bytes())?;
    assert!(parse_model_file(&truncated, &limits, &cancellation).is_err());
    let duplicate = directory.path().join("duplicate.json");
    fs::write(&duplicate, br#"{"x":1,"x":2}"#)?;
    assert!(parse_model_file(&duplicate, &limits, &cancellation).is_err());
    let unknown_gguf = directory.path().join("unknown.gguf");
    let mut unknown_bytes = fs::read(&gguf)?;
    let tensor_type_offset = gguf_tensor_type_offset(&unknown_bytes)?;
    unknown_bytes
        .get_mut(tensor_type_offset..tensor_type_offset + 4)
        .ok_or("GGUF tensor type offset is invalid")?
        .copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(&unknown_gguf, unknown_bytes)?;
    assert!(parse_model_file(&unknown_gguf, &limits, &cancellation).is_err());
    let hostile_pickle = directory.path().join("hostile.pkl");
    fs::write(&hostile_pickle, b"cos\nsystem\n(S'bad'\ntR.")?;
    assert!(matches!(
        parse_model_file(&hostile_pickle, &limits, &cancellation),
        Err(ModelFormatError::RestrictedPickle(
            RestrictedPickleError::ForbiddenTarget { .. }
        ))
    ));
    let unknown_pickle = directory.path().join("unknown.pkl");
    fs::write(&unknown_pickle, [0xff, b'.'])?;
    assert!(matches!(
        parse_model_file(&unknown_pickle, &limits, &cancellation),
        Err(ModelFormatError::RestrictedPickle(
            RestrictedPickleError::UnknownOpcode { .. }
        ))
    ));
    let amplified_pickle = directory.path().join("amplified.pkl");
    let mut amplified_bytes = vec![b']', b'q', 0];
    for _ in 0..8 {
        amplified_bytes.extend_from_slice(&[b'h', 0]);
    }
    amplified_bytes.push(b'.');
    fs::write(&amplified_pickle, amplified_bytes)?;
    let amplification_limits = ParserLimits {
        maximum_metadata_values: 8,
        ..limits
    };
    assert!(matches!(
        parse_model_file(&amplified_pickle, &amplification_limits, &cancellation),
        Err(ModelFormatError::RestrictedPickle(
            RestrictedPickleError::Limit(_)
        ))
    ));
    let duplicate_archive = directory.path().join("duplicate.ckpt");
    write_stored_zip(
        &duplicate_archive,
        &[("archive/data.pkl", b"N."), ("archive/data.pkl", b"N.")],
    )?;
    assert!(parse_model_file(&duplicate_archive, &limits, &cancellation).is_err());
    let traversal_archive = directory.path().join("traversal.ckpt");
    write_stored_zip(&traversal_archive, &[("../data.pkl", b"N.")])?;
    assert!(parse_model_file(&traversal_archive, &limits, &cancellation).is_err());
    let link_archive = directory.path().join("link.ckpt");
    write_stored_zip_with_modes(&link_archive, &[("archive/data.pkl", b"N.", 0o120777)])?;
    assert!(parse_model_file(&link_archive, &limits, &cancellation).is_err());
    let multiple_roots = directory.path().join("multiple-roots.ckpt");
    write_stored_zip(
        &multiple_roots,
        &[
            ("first/data.pkl", b"N."),
            ("first/data/0", &[0, 0, 0, 0]),
            ("second/data/0", &[0, 0, 0, 0]),
        ],
    )?;
    assert!(parse_model_file(&multiple_roots, &limits, &cancellation).is_err());

    let corrupt_storage = directory.path().join("corrupt-storage.ckpt");
    let mut corrupt_storage_bytes = fs::read(&pytorch)?;
    let (_, storage_offset) = stored_zip_local_entry(&corrupt_storage_bytes, "archive/data/0")?;
    let storage_byte = corrupt_storage_bytes
        .get_mut(storage_offset)
        .ok_or("storage byte is missing")?;
    *storage_byte ^= 0xff;
    fs::write(&corrupt_storage, corrupt_storage_bytes)?;
    assert!(parse_model_file(&corrupt_storage, &limits, &cancellation).is_err());

    let local_mismatch = directory.path().join("local-mismatch.ckpt");
    let mut local_mismatch_bytes = fs::read(&pytorch)?;
    let (local_header, _) = stored_zip_local_entry(&local_mismatch_bytes, "archive/data.pkl")?;
    local_mismatch_bytes
        .get_mut(local_header + 8..local_header + 10)
        .ok_or("local compression method is missing")?
        .copy_from_slice(&1_u16.to_le_bytes());
    fs::write(&local_mismatch, local_mismatch_bytes)?;
    assert!(parse_model_file(&local_mismatch, &limits, &cancellation).is_err());
    cases.insert("hostile_and_truncated_rejected", true);
    cases.insert("memo_amplification_rejected", true);
    cases.insert("archive_root_crc_and_header_integrity", true);

    let tiny_limits = ParserLimits {
        manifest_bytes: 16,
        ..limits
    };
    assert!(parse_model_file(&safetensors, &tiny_limits, &cancellation).is_err());
    let memory_limits = ParserLimits {
        maximum_tensor_bytes: 4,
        maximum_aggregate_tensor_bytes: 4,
        ..limits
    };
    assert!(parse_model_file(&safetensors, &memory_limits, &cancellation).is_err());
    cases.insert("deterministic_memory_limit", true);
    let depth_limits = ParserLimits {
        maximum_depth: 1,
        ..limits
    };
    let nested = directory.path().join("nested.json");
    fs::write(&nested, br#"{"a":{"b":1}}"#)?;
    assert!(parse_model_file(&nested, &depth_limits, &cancellation).is_err());
    let yaml_anchor = directory.path().join("anchor.yaml");
    fs::write(&yaml_anchor, b"base_path: &root /models\n")?;
    assert!(parse_model_file(&yaml_anchor, &limits, &cancellation).is_err());
    let invalid_tiktoken = directory.path().join("invalid.tiktoken");
    fs::write(&invalid_tiktoken, b"Y=== 0\n")?;
    assert!(parse_model_file(&invalid_tiktoken, &limits, &cancellation).is_err());
    let duplicate_tiktoken = directory.path().join("duplicate-token.tiktoken");
    fs::write(&duplicate_tiktoken, b"YQ== 0\nYQ== 1\n")?;
    assert!(parse_model_file(&duplicate_tiktoken, &limits, &cancellation).is_err());
    let invalid_sentencepiece = directory.path().join("invalid.model");
    fs::write(&invalid_sentencepiece, [0x0a, 0x03, b'a', b'b', b'c'])?;
    assert!(parse_model_file(&invalid_sentencepiece, &limits, &cancellation).is_err());
    let wrong_alignment_type = directory.path().join("wrong-alignment.gguf");
    let mut wrong_alignment_bytes = fs::read(&gguf)?;
    let alignment_type = gguf_alignment_type_offset(&wrong_alignment_bytes)?;
    wrong_alignment_bytes
        .get_mut(alignment_type..alignment_type + 4)
        .ok_or("GGUF alignment type offset is invalid")?
        .copy_from_slice(&10_u32.to_le_bytes());
    fs::write(&wrong_alignment_type, wrong_alignment_bytes)?;
    assert!(parse_model_file(&wrong_alignment_type, &limits, &cancellation).is_err());
    let overflow = directory.path().join("overflow.safetensors");
    let overflow_header = serde_json::to_vec(&json!({
        "x": {
            "dtype": "U8",
            "shape": [u64::MAX, 2_u64],
            "data_offsets": [0, 0]
        }
    }))?;
    let mut overflow_file = File::create(&overflow)?;
    overflow_file.write_all(&u64::try_from(overflow_header.len())?.to_le_bytes())?;
    overflow_file.write_all(&overflow_header)?;
    assert!(parse_model_file(&overflow, &limits, &cancellation).is_err());
    let count_limited = ParserLimits {
        maximum_tensors: 1,
        maximum_archive_entries: 2,
        ..limits
    };
    let excessive_count = directory.path().join("count.gguf");
    let mut excessive_count_bytes = fs::read(&gguf)?;
    excessive_count_bytes
        .get_mut(8..16)
        .ok_or("GGUF tensor count offset is invalid")?
        .copy_from_slice(&2_u64.to_le_bytes());
    fs::write(&excessive_count, excessive_count_bytes)?;
    assert!(parse_model_file(&excessive_count, &count_limited, &cancellation).is_err());
    cases.insert("bounds_precede_allocation", true);
    cases.insert("strict_config_tokenizer_and_gguf_types", true);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert_eq!(
        parse_model_file(&safetensors, &limits, &cancelled),
        Err(ModelFormatError::Cancelled)
    );
    cases.insert("cancellation", true);
    assert_eq!(
        parse_restricted_pickle_cancellable(b"N.", &limits, &cancelled),
        Err(RestrictedPickleError::Cancelled)
    );
    cases.insert("pickle_structural_cancellation", true);

    let unapproved_sibling = directory.path().join("unapproved.safetensors");
    fs::write(&unapproved_sibling, b"not a model")?;
    let mut import_index = ArtifactIndex::default();
    import_index.add_root(ArtifactRoot::approved_import(
        "approved",
        "checkpoints",
        &safetensors,
    )?)?;
    import_index.refresh(&cancellation)?;
    assert_eq!(import_index.records().count(), 1);
    let canonical_safetensors = fs::canonicalize(&safetensors)?;
    assert_eq!(
        import_index
            .records()
            .next()
            .map(|record| record.canonical_path.as_path()),
        Some(canonical_safetensors.as_path())
    );
    fs::remove_file(unapproved_sibling)?;
    cases.insert("approved_import_isolation", true);

    let root = ArtifactRoot::canonical(
        "models",
        "checkpoints",
        directory.path(),
        [
            "safetensors",
            "ckpt",
            "gguf",
            "json",
            "yaml",
            "model",
            "tiktoken",
        ],
    )?;
    let mut index = ArtifactIndex::default();
    index.add_root(root)?;
    index.refresh(&cancellation)?;
    let index_snapshot = index.snapshot()?;
    let trusted_roots = index.roots().cloned().collect::<Vec<_>>();
    let substituted_directory = tempfile::tempdir()?;
    let substituted_root = ArtifactRoot::canonical(
        "models",
        "checkpoints",
        substituted_directory.path(),
        [
            "safetensors",
            "ckpt",
            "gguf",
            "json",
            "yaml",
            "model",
            "tiktoken",
        ],
    )?;
    assert!(ArtifactIndex::from_snapshot(&index_snapshot, [substituted_root]).is_err());
    let mut malformed_snapshot: serde_json::Value = serde_json::from_slice(&index_snapshot)?;
    let uppercase_digest = malformed_snapshot
        .pointer("/records/0/sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or("artifact snapshot digest is missing")?
        .to_ascii_uppercase();
    *malformed_snapshot
        .pointer_mut("/records/0/sha256")
        .ok_or("artifact snapshot digest is missing")? =
        serde_json::Value::String(uppercase_digest);
    assert!(
        ArtifactIndex::from_snapshot(
            &serde_json::to_vec(&malformed_snapshot)?,
            trusted_roots.clone()
        )
        .is_err()
    );
    let mut restarted_index = ArtifactIndex::from_snapshot(&index_snapshot, trusted_roots)?;
    assert_eq!(restarted_index.snapshot()?, index_snapshot);
    cases.insert("snapshot_trusted_root_binding", true);
    cases.insert("snapshot_record_canonicalization", true);
    let key = ArtifactKey::new("models", "model.safetensors")?;
    let mut store = ModelStore::new(limits)?;
    let loaded = store.load(&restarted_index, &key, &cancellation)?;
    assert_eq!(loaded.accounting().resident_bytes, 0);
    assert_eq!(
        store
            .read_tensor(&restarted_index, &loaded, "weight", &cancellation)?
            .len(),
        8
    );
    #[cfg(unix)]
    {
        let mapping = store.map_tensor(&restarted_index, &loaded, "weight", &cancellation)?;
        assert_eq!(
            mapping.as_bytes(),
            store.read_tensor(&restarted_index, &loaded, "weight", &cancellation)?
        );
    }
    write_safetensors(&safetensors, "weight", "F32", &[2], &[9; 8])?;
    assert!(
        store
            .read_tensor(&restarted_index, &loaded, "weight", &cancellation)
            .is_err()
    );
    restarted_index.refresh(&cancellation)?;
    assert!(
        store
            .read_tensor(&restarted_index, &loaded, "weight", &cancellation)
            .is_err()
    );
    let replacement = store.load(&restarted_index, &key, &cancellation)?;
    assert_eq!(
        store.read_tensor(&restarted_index, &replacement, "weight", &cancellation)?,
        [9; 8]
    );
    cases.insert("verified_handle_change_detection", true);
    fs::remove_file(&safetensors)?;
    let changes = restarted_index.refresh(&cancellation)?;
    assert!(
        changes
            .iter()
            .any(|change| change.key == key && change.kind == ArtifactChangeKind::Missing)
    );
    assert_eq!(
        restarted_index
            .record(&key)
            .map(|record| &record.availability),
        Some(&ArtifactAvailability::Missing)
    );
    write_safetensors(&safetensors, "weight", "F32", &[2], &[0; 8])?;
    let changes = restarted_index.refresh(&cancellation)?;
    assert!(
        changes
            .iter()
            .any(|change| change.key == key && change.kind == ArtifactChangeKind::Restored)
    );
    assert_eq!(
        restarted_index
            .record(&key)
            .map(|record| &record.availability),
        Some(&ArtifactAvailability::Present)
    );
    cases.insert("lazy_index_watch_missing_recovery", true);
    cases.insert("model_store_read_side_index_adapter", true);
    cases.insert("snapshot_restart", true);

    let model_store_source = include_str!("../src/model_store.rs");
    assert!(!model_store_source.contains("index: ArtifactIndex"));
    assert!(!model_store_source.contains("File::open("));
    assert!(!model_store_source.contains(".resolve("));
    assert!(model_store_source.contains("open_verified("));
    assert!(model_store_source.contains("parse_verified_model_file("));
    cases.insert("authoritative_index_owns_security_and_identity", true);

    assert!(cases.values().all(|passed| *passed));
    write_validation_artifact(&fixture_digests, &cases)?;
    Ok(())
}

fn write_safetensors(
    path: &Path,
    name: &str,
    data_type: &str,
    shape: &[u64],
    data: &[u8],
) -> Result<(), Box<dyn Error>> {
    let header = serde_json::to_vec(&json!({
        "__metadata__": {"format": "pt"},
        name: {
            "dtype": data_type,
            "shape": shape,
            "data_offsets": [0, data.len()]
        }
    }))?;
    let mut file = File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(data)?;
    Ok(())
}

fn tensor_pickle(name: &str, storage_key: &str) -> Vec<u8> {
    let mut bytes = vec![0x80, 0x02, b'}', b'('];
    binunicode(&mut bytes, name);
    push_plain_tensor_rebuild(&mut bytes, "torch.FloatStorage", storage_key, 1);
    bytes.extend_from_slice(&[b'u', b'.']);
    bytes
}

fn push_plain_tensor_rebuild(
    bytes: &mut Vec<u8>,
    storage_type: &str,
    storage_key: &str,
    elements: u8,
) {
    bytes.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
    bytes.push(b'(');
    bytes.push(b'(');
    binunicode(bytes, "storage");
    let (module, name) = storage_type
        .rsplit_once('.')
        .unwrap_or(("torch", storage_type));
    bytes.push(b'c');
    bytes.extend_from_slice(module.as_bytes());
    bytes.push(b'\n');
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(b'\n');
    binunicode(bytes, storage_key);
    binunicode(bytes, "cpu");
    bytes.extend_from_slice(&[b'K', elements, b't', b'Q', b'K', 0]);
    bytes.extend_from_slice(&[b'(', b'K', elements, b't']);
    bytes.extend_from_slice(&[b'(', b'K', 1, b't']);
    bytes.extend_from_slice(&[0x89, b'}', b't', b'R']);
}

fn parameter_pickle(name: &str, storage_key: &str, target: &str) -> Vec<u8> {
    let mut bytes = vec![0x80, 0x02, b'}', b'('];
    binunicode(&mut bytes, name);
    let (module, name) = target.rsplit_once('.').unwrap_or((target, ""));
    bytes.push(b'c');
    bytes.extend_from_slice(module.as_bytes());
    bytes.push(b'\n');
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&[b'\n', b'(']);
    push_plain_tensor_rebuild(&mut bytes, "torch.FloatStorage", storage_key, 1);
    bytes.push(0x89);
    if target != "torch.nn.parameter.Parameter" {
        bytes.push(b'}');
        if target == "torch._utils._rebuild_parameter_with_state" {
            bytes.push(b'}');
        }
    }
    bytes.extend_from_slice(&[b't', b'R']);
    if target == "torch.nn.parameter.Parameter" {
        bytes.extend_from_slice(&[b'}', b'b']);
    }
    bytes.extend_from_slice(&[b'u', b'.']);
    bytes
}

fn quantized_pickle(name: &str, storage_key: &str, scale: f64) -> Vec<u8> {
    let mut bytes = vec![0x80, 0x02, b'}', b'('];
    binunicode(&mut bytes, name);
    bytes.extend_from_slice(b"ctorch._utils\n_rebuild_qtensor\n(");
    bytes.push(b'(');
    binunicode(&mut bytes, "storage");
    bytes.extend_from_slice(b"ctorch\nQInt8Storage\n");
    binunicode(&mut bytes, storage_key);
    binunicode(&mut bytes, "cpu");
    bytes.extend_from_slice(&[b'K', 4, b't', b'Q', b'K', 0]);
    bytes.extend_from_slice(&[b'(', b'K', 4, b't']);
    bytes.extend_from_slice(&[b'(', b'K', 1, b't']);
    bytes.extend_from_slice(b"(ctorch\nper_tensor_affine\nG");
    bytes.extend_from_slice(&scale.to_be_bytes());
    bytes.extend_from_slice(&[b'K', 0, b't', 0x89, b'}', b't', b'R', b'u', b'.']);
    bytes
}

fn binunicode(bytes: &mut Vec<u8>, value: &str) {
    bytes.push(b'X');
    bytes.extend_from_slice(&u32::try_from(value.len()).unwrap_or(u32::MAX).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_gguf(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    push_gguf_string(&mut bytes, "general.alignment")?;
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    push_gguf_string(&mut bytes, "weight")?;
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&24_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    let aligned = bytes.len().next_multiple_of(32);
    bytes.resize(aligned, 0);
    bytes.push(7);
    fs::write(path, bytes)?;
    Ok(())
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) -> Result<(), Box<dyn Error>> {
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn gguf_tensor_type_offset(bytes: &[u8]) -> Result<usize, Box<dyn Error>> {
    let mut cursor = 24_usize;
    let key_length = read_u64_slice(bytes, &mut cursor)?;
    cursor = cursor
        .checked_add(usize::try_from(key_length)?)
        .ok_or("overflow")?;
    cursor = cursor.checked_add(8).ok_or("overflow")?;
    let name_length = read_u64_slice(bytes, &mut cursor)?;
    cursor = cursor
        .checked_add(usize::try_from(name_length)?)
        .ok_or("overflow")?;
    cursor = cursor.checked_add(4 + 8).ok_or("overflow")?;
    Ok(cursor)
}

fn gguf_alignment_type_offset(bytes: &[u8]) -> Result<usize, Box<dyn Error>> {
    let mut cursor = 24_usize;
    let key_length = read_u64_slice(bytes, &mut cursor)?;
    cursor = cursor
        .checked_add(usize::try_from(key_length)?)
        .ok_or("overflow")?;
    Ok(cursor)
}

fn read_u64_slice(bytes: &[u8], cursor: &mut usize) -> Result<u64, Box<dyn Error>> {
    let value: [u8; 8] = bytes
        .get(*cursor..cursor.saturating_add(8))
        .ok_or("truncated u64")?
        .try_into()?;
    *cursor = cursor.saturating_add(8);
    Ok(u64::from_le_bytes(value))
}

fn stored_zip_local_entry(
    bytes: &[u8],
    expected_name: &str,
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut cursor = 0_usize;
    while bytes.get(cursor..cursor.saturating_add(4)) == Some(&0x0403_4b50_u32.to_le_bytes()) {
        let name_length = usize::from(read_u16_at(bytes, cursor + 26)?);
        let extra_length = usize::from(read_u16_at(bytes, cursor + 28)?);
        let data_length = usize::try_from(read_u32_at(bytes, cursor + 18)?)?;
        let name_start = cursor.checked_add(30).ok_or("local name overflow")?;
        let name_end = name_start
            .checked_add(name_length)
            .ok_or("local name overflow")?;
        let name = std::str::from_utf8(
            bytes
                .get(name_start..name_end)
                .ok_or("local name missing")?,
        )?;
        let data_start = name_end
            .checked_add(extra_length)
            .ok_or("local data overflow")?;
        let next = data_start
            .checked_add(data_length)
            .ok_or("local data overflow")?;
        if next > bytes.len() {
            return Err("local data missing".into());
        }
        if name == expected_name {
            return Ok((cursor, data_start));
        }
        cursor = next;
    }
    Err(format!("ZIP entry {expected_name:?} is missing").into())
}

fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, Box<dyn Error>> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset.saturating_add(2))
            .ok_or("truncated u16")?
            .try_into()?,
    ))
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn Error>> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset.saturating_add(4))
            .ok_or("truncated u32")?
            .try_into()?,
    ))
}

fn write_stored_zip(path: &Path, entries: &[(&str, &[u8])]) -> Result<(), Box<dyn Error>> {
    let entries = entries
        .iter()
        .map(|(name, data)| (*name, *data, 0o100644))
        .collect::<Vec<_>>();
    write_stored_zip_with_modes(path, &entries)
}

fn write_stored_zip_with_modes(
    path: &Path,
    entries: &[(&str, &[u8], u32)],
) -> Result<(), Box<dyn Error>> {
    let mut output = Vec::new();
    let mut central = Vec::new();
    for (name, data, mode) in entries {
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
        central.extend_from_slice(&(mode << 16).to_le_bytes());
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

fn write_stored_zip64(path: &Path, entries: &[(&str, &[u8])]) -> Result<(), Box<dyn Error>> {
    let mut output = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = u64::try_from(output.len())?;
        let name_bytes = name.as_bytes();
        let crc = crc32(data);
        output.extend_from_slice(&0x0403_4b50_u32.to_le_bytes());
        output.extend_from_slice(&45_u16.to_le_bytes());
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
        central.extend_from_slice(&0x032d_u16.to_le_bytes());
        central.extend_from_slice(&45_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&u32::MAX.to_le_bytes());
        central.extend_from_slice(&u32::MAX.to_le_bytes());
        central.extend_from_slice(&u16::try_from(name_bytes.len())?.to_le_bytes());
        central.extend_from_slice(&28_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&(0o100644_u32 << 16).to_le_bytes());
        central.extend_from_slice(&u32::MAX.to_le_bytes());
        central.extend_from_slice(name_bytes);
        central.extend_from_slice(&0x0001_u16.to_le_bytes());
        central.extend_from_slice(&24_u16.to_le_bytes());
        central.extend_from_slice(&u64::try_from(data.len())?.to_le_bytes());
        central.extend_from_slice(&u64::try_from(data.len())?.to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
    }
    let central_offset = u64::try_from(output.len())?;
    let central_size = u64::try_from(central.len())?;
    output.extend_from_slice(&central);
    let zip64_directory_offset = u64::try_from(output.len())?;
    output.extend_from_slice(&0x0606_4b50_u32.to_le_bytes());
    output.extend_from_slice(&44_u64.to_le_bytes());
    output.extend_from_slice(&45_u16.to_le_bytes());
    output.extend_from_slice(&45_u16.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
    output.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
    output.extend_from_slice(&central_size.to_le_bytes());
    output.extend_from_slice(&central_offset.to_le_bytes());
    output.extend_from_slice(&0x0706_4b50_u32.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&zip64_directory_offset.to_le_bytes());
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(&0x0605_4b50_u32.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&u16::MAX.to_le_bytes());
    output.extend_from_slice(&u16::MAX.to_le_bytes());
    output.extend_from_slice(&u32::MAX.to_le_bytes());
    output.extend_from_slice(&u32::MAX.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    fs::write(path, output)?;
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(digest.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn write_validation_artifact(
    fixture_digests: &BTreeMap<&str, String>,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    let directory = target.join("sim-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation": "VAL-MODEL-FORMAT-001",
        "parser_limits_version": PARSER_LIMITS_VERSION,
        "pickle_allowlist_version": RESTRICTED_PICKLE_ALLOWLIST_VERSION,
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "native-rust-cpu-file-slice"
        },
        "fixture_digests": fixture_digests,
        "cases": cases,
        "skipped": []
    });
    fs::write(
        directory.join("val-model-format-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
