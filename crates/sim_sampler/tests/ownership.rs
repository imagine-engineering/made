use std::{
    collections::BTreeSet,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn rust_sources(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    rust_sources_below(&root.join("crates"), false)
}

fn rust_sources_below(root: &Path, reject_symlinks: bool) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if reject_symlinks && file_type.is_symlink() {
                return Err(format!(
                    "generated sampler algorithm tree contains forbidden symlink {}",
                    path.display()
                )
                .into());
            }
            if file_type.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    Ok(sources)
}

fn rust_code_tokens(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if bytes[index..].starts_with(b"//") {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index..].starts_with(b"/*") {
            index += 2;
            let mut depth = 1_usize;
            while index < bytes.len() && depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            continue;
        }

        if bytes[index..].starts_with(b"r#")
            && bytes
                .get(index + 2)
                .is_some_and(|byte| !matches!(byte, b'#' | b'"'))
        {
            index += 2;
            let start = index;
            if bytes[index].is_ascii() {
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
            } else {
                while index < bytes.len() {
                    let Some(character) = source[index..].chars().next() else {
                        break;
                    };
                    if character.is_ascii()
                        && !(character.is_ascii_alphanumeric() || character == '_')
                    {
                        break;
                    }
                    if character.is_whitespace() {
                        break;
                    }
                    index += character.len_utf8();
                }
            }
            let identifier = &source[start..index];
            // Raw keywords may name macros, unlike control-flow keywords before negation.
            tokens.push(if negation_can_follow_keyword(identifier) {
                format!("r#{identifier}")
            } else {
                identifier.to_owned()
            });
            continue;
        }

        let raw_prefix = if bytes[index..].starts_with(b"br") {
            Some(2)
        } else if bytes[index] == b'r' {
            Some(1)
        } else {
            None
        };
        if let Some(prefix_length) = raw_prefix {
            let mut quote = index + prefix_length;
            while quote < bytes.len() && bytes[quote] == b'#' {
                quote += 1;
            }
            if quote < bytes.len() && bytes[quote] == b'"' {
                let hash_count = quote - index - prefix_length;
                index = quote + 1;
                while index < bytes.len() {
                    if bytes[index] == b'"'
                        && index + hash_count < bytes.len()
                        && bytes[index + 1..index + 1 + hash_count]
                            .iter()
                            .all(|byte| *byte == b'#')
                    {
                        index += hash_count + 1;
                        break;
                    }
                    index += 1;
                }
                continue;
            }
        }

        let string_prefix_length =
            usize::from(bytes[index] == b'b' && bytes.get(index + 1) == Some(&b'"'));
        if bytes[index + string_prefix_length] == b'"' {
            index += string_prefix_length + 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == b'"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
            continue;
        }

        if !bytes[index].is_ascii() {
            let start = index;
            while index < bytes.len() {
                let Some(character) = source[index..].chars().next() else {
                    break;
                };
                if character.is_ascii() {
                    if !(character.is_ascii_alphanumeric() || character == '_') {
                        break;
                    }
                } else if character.is_whitespace() {
                    break;
                }
                index += character.len_utf8();
            }
            tokens.push(source[start..index].to_owned());
            continue;
        }
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            tokens.push(source[start..index].to_owned());
            continue;
        }
        tokens.push(char::from(bytes[index]).to_string());
        index += 1;
    }
    tokens
}

fn declared_type_names(tokens: &[String]) -> Vec<&str> {
    tokens
        .windows(2)
        .filter_map(|window| {
            ["struct", "enum", "type", "trait", "union"]
                .contains(&window[0].as_str())
                .then_some(window[1].as_str())
        })
        .collect()
}

fn competing_shared_alias(tokens: &[String]) -> Option<&str> {
    tokens.windows(2).find_map(|window| {
        if window[0] != "as" {
            return None;
        }
        let alias = window[1].as_str();
        (alias == "CancellationToken"
            || ["Trace", "Progress", "Observation", "NoiseRequest"]
                .iter()
                .any(|fragment| alias.contains(fragment)))
        .then_some(alias)
    })
}

fn contains_qualified_call(tokens: &[String], owner: &str, method: &str) -> bool {
    let mut aliases = BTreeSet::from([owner]);
    loop {
        let mut changed = false;
        for (index, token) in tokens.iter().enumerate() {
            if aliases.contains(token.as_str())
                && tokens.get(index + 1).is_some_and(|token| token == "as")
                && let Some(alias) = tokens.get(index + 2)
            {
                changed |= aliases.insert(alias);
            }
            if token == "type"
                && let (Some(alias), Some(equals)) = (tokens.get(index + 1), tokens.get(index + 2))
                && equals == "="
                && tokens[index + 3..]
                    .iter()
                    .take_while(|token| token.as_str() != ";")
                    .any(|target| aliases.contains(target.as_str()))
            {
                changed |= aliases.insert(alias);
            }
        }
        if !changed {
            break;
        }
    }
    tokens.iter().enumerate().any(|(index, token)| {
        aliases.contains(token.as_str())
            && tokens[index + 1..]
                .iter()
                .take_while(|token| ![";", "{", "}"].contains(&token.as_str()))
                .collect::<Vec<_>>()
                .windows(3)
                .any(|window| window[0] == ":" && window[1] == ":" && window[2] == method)
    })
}

fn negation_can_follow_keyword(token: &str) -> bool {
    matches!(
        token,
        "if" | "while" | "return" | "break" | "yield" | "match" | "in"
    )
}

fn unsupported_macro(tokens: &[String]) -> Option<&str> {
    const ALLOWED: &[&str] = &[
        "assert",
        "assert_eq",
        "concat",
        "env",
        "format",
        "include_str",
        "matches",
        "vec",
    ];
    tokens.windows(3).enumerate().find_map(|(index, window)| {
        let identifier = window[0].chars().next().is_some_and(|character| {
            character.is_ascii_alphabetic() || character == '_' || !character.is_ascii()
        });
        let invocation = identifier
            && !negation_can_follow_keyword(&window[0])
            && window[1] == "!"
            && ["(", "[", "{"].contains(&window[2].as_str());
        if window[0] == "macro_rules" || (invocation && !ALLOWED.contains(&window[0].as_str())) {
            return Some(window[0].as_str());
        }
        if !invocation {
            return None;
        }
        let qualified = index >= 2 && tokens[index - 2] == ":" && tokens[index - 1] == ":";
        let imported = tokens.iter().enumerate().any(|(use_index, token)| {
            token == "use"
                && tokens[use_index + 1..]
                    .iter()
                    .take_while(|token| token.as_str() != ";")
                    .any(|token| token == &window[0])
        });
        (qualified || imported).then_some(window[0].as_str())
    })
}

fn contains_any_associated_open(tokens: &[String]) -> bool {
    tokens
        .windows(3)
        .any(|window| window[0] == ":" && window[1] == ":" && window[2] == "open")
}

fn unsupported_attribute(tokens: &[String]) -> Option<&str> {
    const ALLOWED_ATTRIBUTES: &[&str] =
        &["allow", "cfg", "default", "derive", "error", "from", "test"];
    const ALLOWED_DERIVES: &[&str] = &[
        "Clone",
        "Copy",
        "Debug",
        "Default",
        "Eq",
        "Error",
        "PartialEq",
    ];
    for (index, window) in tokens.windows(3).enumerate() {
        if window[0] != "#" || window[1] != "[" {
            continue;
        }
        let attribute = window[2].as_str();
        if !ALLOWED_ATTRIBUTES.contains(&attribute) {
            return Some(attribute);
        }
        if attribute != "derive" {
            continue;
        }
        for token in tokens[index + 3..]
            .iter()
            .take_while(|token| token.as_str() != "]")
        {
            if token
                .as_bytes()
                .first()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
                && !ALLOWED_DERIVES.contains(&token.as_str())
            {
                return Some(token);
            }
        }
    }
    None
}

#[test]
fn authoritative_sampling_owners_have_no_competing_production_definitions()
-> Result<(), Box<dyn Error>> {
    let root = workspace()?;
    let sources = rust_sources(&root)?;
    for (symbol, owner) in [
        ("SamplerIdentity", "crates/sim_sampler/src/sampler.rs"),
        ("SamplerDefinition", "crates/sim_sampler/src/sampler.rs"),
        (
            "CfgPpDenoiserContractError",
            "crates/sim_sampler/src/sampler.rs",
        ),
        ("SamplerRegistry", "crates/sim_sampler/src/sampler.rs"),
        ("SamplingPlan", "crates/sim_sampler/src/sampler.rs"),
        ("SamplingProgress", "crates/sim_sampler/src/sampler.rs"),
        ("SamplingTrace", "crates/sim_sampler/src/sampler.rs"),
        (
            "AdaptiveSamplingProgress",
            "crates/sim_sampler/src/sampler.rs",
        ),
        (
            "AdaptiveSamplingAttempt",
            "crates/sim_sampler/src/sampler.rs",
        ),
        (
            "AdaptiveSamplingAttemptTrace",
            "crates/sim_sampler/src/sampler.rs",
        ),
        ("AdaptiveSamplingTrace", "crates/sim_sampler/src/sampler.rs"),
        (
            "AdaptiveSamplingSession",
            "crates/sim_sampler/src/sampler.rs",
        ),
        ("SamplingSession", "crates/sim_sampler/src/sampler.rs"),
        ("ObservedSamplingStep", "crates/sim_sampler/src/sampler.rs"),
        ("SchedulerIdentity", "crates/sim_sampler/src/scheduler.rs"),
        ("SchedulerDefinition", "crates/sim_sampler/src/scheduler.rs"),
        ("SchedulerRegistry", "crates/sim_sampler/src/scheduler.rs"),
        ("SchedulerRequest", "crates/sim_sampler/src/scheduler.rs"),
        (
            "SamplingProfileIdentity",
            "crates/sim_sampler/src/sampling_profile.rs",
        ),
        (
            "PredictionInterpretation",
            "crates/sim_sampler/src/sampling_profile.rs",
        ),
        (
            "SamplingSnrMode",
            "crates/sim_sampler/src/sampling_profile.rs",
        ),
        (
            "SamplingProfile",
            "crates/sim_sampler/src/sampling_profile.rs",
        ),
        (
            "DiscreteSamplingProfile",
            "crates/sim_sampler/src/sampling_profile.rs",
        ),
        ("NoisePhaseIdentity", "crates/sim_sampler/src/noise.rs"),
        ("NoiseRequest", "crates/sim_sampler/src/noise.rs"),
        (
            "CompatibilityNoiseRequest",
            "crates/sim_sampler/src/noise.rs",
        ),
        (
            "BrownianNoiseIntervalAddress",
            "crates/sim_sampler/src/noise.rs",
        ),
        ("NoiseTrace", "crates/sim_sampler/src/noise.rs"),
    ] {
        let expected_owner = root.join(owner);
        let definitions = sources
            .iter()
            .filter(|path| {
                !path
                    .components()
                    .any(|component| component.as_os_str() == "tests")
            })
            .filter_map(|path| {
                let source = fs::read_to_string(path).ok()?;
                declared_type_names(&rust_code_tokens(&source))
                    .contains(&symbol)
                    .then_some(path)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            definitions,
            [&expected_owner],
            "{symbol} must have exactly one production owner"
        );
    }

    let model = fs::read_to_string(root.join("crates/sim_model/src/slices/native_diffusion.rs"))?;
    assert!(!model.contains("fn sigma_to_timestep"));
    assert!(!model.contains("0.00085_f64.sqrt()"));
    assert!(!model.contains("DiscreteSamplingProfile"));
    assert!(model.contains("detect_model_family_rules("));
    assert!(!model.contains("guided_prediction_at_model_time"));
    assert!(!model.contains("InvalidGuidance"));
    assert!(!model.contains("mul_add(cfg"));

    let runtime =
        fs::read_to_string(root.join("crates/sim_runtime/src/native_execution_controller.rs"))?;
    assert!(!runtime.contains("noise.mul_add(initial_sigma"));
    assert!(!runtime.contains("RngStreamAddress::new("));
    assert!(!runtime.contains("\"initial-noise-v1"));
    assert!(runtime.contains("INITIAL_NOISE_PHASE_ID"));
    assert!(runtime.contains("checked_native_diffusion_plan_for_profile("));
    assert!(runtime.contains("normal_sigmas_for_profile("));
    assert!(runtime.contains("scale_initial_noise_for_profile("));
    assert!(runtime.contains("sampler_payload.execute_checked("));
    assert!(runtime.contains("NoiseRequest::native_diffusion("));
    assert!(runtime.contains("model_time_for_profile("));
    assert!(runtime.contains("scale_model_input_for_profile("));
    assert!(runtime.contains("interpret_prediction_for_profile("));
    assert!(runtime.contains("impl GuidanceDenoiser for Sd15GuidanceDenoiser"));
    assert!(runtime.contains("pub struct Sd15GuidanceAdapter"));
    assert!(runtime.contains("NativeFamilyGuidanceDenoiser::checked("));
    assert!(runtime.contains("execute_guidance("));
    assert!(runtime.contains("[GUIDANCE_ADAPTER_ID.as_bytes()]"));
    assert!(runtime.contains("artifact_digests: identities.conditioning().artifact_digests()"));
    let ksampler = runtime
        .split("struct KSamplerNode")
        .nth(1)
        .and_then(|source| source.split("struct VaeDecodeNode").next())
        .ok_or("KSampler implementation slice is unavailable")?;
    let guidance_position = ksampler
        .find("guider.value.execute_with_modifier_session(")
        .ok_or("KSampler does not call canonical guidance")?;
    let publication_position = ksampler
        .find("publish_latent_bundle(&context, final_latent)")
        .ok_or("KSampler final latent publication is unavailable")?;
    assert!(guidance_position < publication_position);
    assert!(
        ksampler[guidance_position..publication_position]
            .contains("guider.value.execute_with_modifier_session(")
    );
    assert!(
        ksampler[guidance_position..publication_position]
            .contains("guidance.execute_with_modifier_session(")
    );
    assert_eq!(
        ksampler
            .matches("publish_latent_bundle(&context, final_latent)")
            .count(),
        1
    );

    let fixture_generator = fs::read_to_string(
        root.join("crates/sim_test_support/src/bin/generate_native_diffusion_fixture.rs"),
    )?;
    assert!(!fixture_generator.contains("value * sigmas[0]"));
    assert!(!fixture_generator.contains("RngStreamAddress::new("));
    assert!(fixture_generator.contains("NoiseRequest::native_diffusion("));
    assert!(fixture_generator.contains("scale_initial_noise("));
    assert!(fixture_generator.contains("scale_model_input("));
    assert!(fixture_generator.contains("Sd15GuidanceAdapter::checked("));
    let fixture_guidance_position = fixture_generator
        .find("let prediction = guidance")
        .ok_or("fixture generator does not call canonical guidance")?;
    assert!(fixture_generator[fixture_guidance_position..].contains(".execute("));

    for relative in [
        "crates/sim_test_support/tests/native_diffusion_foundation.rs",
        "crates/sim_test_support/tests/native_diffusion_e2e.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))?;
        assert!(source.contains("Sd15GuidanceAdapter::checked("));
        let guidance_position = source
            .find("let prediction = guidance")
            .ok_or("native diffusion fixture does not call canonical guidance")?;
        assert!(source[guidance_position..].contains(".execute("));
        assert!(!source.contains("guided_prediction_at_model_time"));
    }

    for relative in [
        "crates/sim_sampler/src/sampler.rs",
        "crates/sim_sampler/src/scheduler.rs",
        "crates/sim_sampler/src/sampling_profile.rs",
        "crates/sim_sampler/src/noise.rs",
        "crates/sim_sampler/src/algorithms/native_diffusion.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))?;
        for forbidden in [
            "ExecutionQueue",
            "NativeQueue",
            "OutputCommitter",
            "SerializableItem",
            "TypedRoot",
            "AssetIndex",
            "ModelIndex",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative} contains forbidden owner {forbidden}"
            );
        }
    }

    let algorithm_directory = root.join("crates/sim_sampler/src/algorithms");
    let native_diffusion = algorithm_directory.join("native_diffusion.rs");
    for path in rust_sources_below(&algorithm_directory, true)? {
        if path == native_diffusion {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        let tokens = rust_code_tokens(&source);
        let competing_type = declared_type_names(&tokens).into_iter().find(|name| {
            ["Trace", "Progress", "Observation", "NoiseRequest"]
                .iter()
                .any(|fragment| name.contains(fragment))
        });
        assert!(
            competing_type.is_none(),
            "generated sampler row {} owns competing shared type {}",
            path.display(),
            competing_type.unwrap_or_default()
        );
        assert!(
            competing_shared_alias(&tokens).is_none(),
            "generated sampler row {} aliases a competing shared type as {}",
            path.display(),
            competing_shared_alias(&tokens).unwrap_or_default()
        );
        assert!(
            !contains_qualified_call(&tokens, "CompatibilityRngTransaction", "open"),
            "generated sampler row {} bypasses the canonical compatibility-noise adapter with CompatibilityRngTransaction::open",
            path.display()
        );
        assert!(
            !contains_any_associated_open(&tokens),
            "generated sampler row {} calls an associated open constructor instead of a canonical injected service",
            path.display()
        );
        for forbidden in ["RngCompatibilityRequest", "RngStreamAddress", "RngStream"] {
            assert!(
                !tokens.iter().any(|token| token == forbidden),
                "generated sampler row {} bypasses the canonical compatibility-noise adapter by using {forbidden}",
                path.display()
            );
        }
        assert!(
            unsupported_macro(&tokens).is_none(),
            "generated sampler row {} invokes unsupported macro {} that could conceal a competing owner or RNG constructor",
            path.display(),
            unsupported_macro(&tokens).unwrap_or_default()
        );
        assert!(
            unsupported_attribute(&tokens).is_none(),
            "generated sampler row {} uses unsupported attribute or derive macro {} that could conceal a competing owner or RNG constructor",
            path.display(),
            unsupported_attribute(&tokens).unwrap_or_default()
        );
        assert!(
            !tokens
                .windows(2)
                .any(|window| window[0] == "struct" && window[1] == "CancellationToken"),
            "generated sampler row {} owns a competing cancellation token",
            path.display()
        );
    }

    let scheduler_directory = root.join("crates/sim_sampler/src/schedulers");
    for path in rust_sources_below(&scheduler_directory, true)? {
        let source = fs::read_to_string(&path)?;
        let tokens = rust_code_tokens(&source);
        assert!(
            tokens.iter().any(|token| {
                [
                    "build_scheduler_schedule",
                    "build_scheduler_schedule_with_capacity",
                    "normal_schedule",
                    "normal_schedule_with_mode",
                ]
                .contains(&token.as_str())
            }),
            "generated scheduler row {} bypasses the canonical scheduler builder or family adapter",
            path.display()
        );
        if tokens
            .iter()
            .any(|token| token == "build_scheduler_schedule_with_capacity")
        {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("ddim_uniform_sim_model_0204.rs"),
                "only DDIM's source-extra branch may request the checked two-slot equation bound"
            );
        }
        assert!(
            declared_type_names(&tokens).is_empty(),
            "generated scheduler row {} owns a shared type instead of only an immutable definition and equation",
            path.display()
        );
        for forbidden in [
            "validate_bounds",
            "workspace_vec",
            "try_reserve",
            "start_step",
            "end_step",
            "penultimate_sigma_policy",
            "saturating_sub",
        ] {
            assert!(
                !tokens.iter().any(|token| token == forbidden),
                "generated scheduler row {} duplicates canonical scheduler behavior through {forbidden}",
                path.display()
            );
        }
        assert!(
            unsupported_macro(&tokens).is_none(),
            "generated scheduler row {} invokes unsupported macro {}",
            path.display(),
            unsupported_macro(&tokens).unwrap_or_default()
        );
        assert!(
            unsupported_attribute(&tokens).is_none(),
            "generated scheduler row {} uses unsupported attribute or derive macro {}",
            path.display(),
            unsupported_attribute(&tokens).unwrap_or_default()
        );
    }
    Ok(())
}

#[test]
fn ownership_lexer_is_literal_resistant_and_fail_closed() {
    let harmless = r###"
        // struct CommentTrace;
        /* enum CommentProgress { Value } */
        const TEXT: &str = "struct StringNoiseRequest; CompatibilityRngTransaction::open";
        const RAW: &str = r#"struct RawObservation; RngStream::new"#;
        let invalid = !value.is_finite() || !(0.1..=20.0).contains(&value);
        if !(1..=4).contains(&order) { return !(valid); }
        assert!(true);
    "###;
    let harmless_tokens = rust_code_tokens(harmless);
    assert!(declared_type_names(&harmless_tokens).is_empty());
    assert!(!contains_qualified_call(
        &harmless_tokens,
        "CompatibilityRngTransaction",
        "open"
    ));
    assert!(unsupported_macro(&harmless_tokens).is_none());
    assert_eq!(
        unsupported_macro(&rust_code_tokens("r#if!(hidden_owner);")),
        Some("r#if")
    );

    let raw_identifier_tokens = rust_code_tokens(
        "struct r#SamplingTrace; r#CompatibilityRngTransaction::open(); struct λNoiseRequest; struct r#λProgress;",
    );
    assert!(declared_type_names(&raw_identifier_tokens).contains(&"SamplingTrace"));
    assert!(
        declared_type_names(&raw_identifier_tokens)
            .iter()
            .any(|name| name.contains("NoiseRequest"))
    );
    assert!(
        declared_type_names(&raw_identifier_tokens)
            .iter()
            .any(|name| name.contains("Progress"))
    );
    assert!(contains_qualified_call(
        &raw_identifier_tokens,
        "CompatibilityRngTransaction",
        "open"
    ));

    for source in [
        "use crate::CompatibilityRngTransaction as Transaction; Transaction::open();",
        "type Transaction = CompatibilityRngTransaction; Transaction::open();",
        "type Transaction = sim_tensor::CompatibilityRngTransaction; Transaction::open();",
        "<CompatibilityRngTransaction>::open();",
        "<CompatibilityRngTransaction as crate::OpenCompatibilityTransaction>::open();",
        "use crate::RngStream as Stream; type Alias = Stream; Alias::new();",
    ] {
        let tokens = rust_code_tokens(source);
        let (owner, method) = if source.contains("RngStream") {
            ("RngStream", "new")
        } else {
            ("CompatibilityRngTransaction", "open")
        };
        assert!(contains_qualified_call(&tokens, owner, method), "{source}");
    }

    assert_eq!(
        unsupported_macro(&rust_code_tokens(
            "macro_rules! hidden { () => { struct HiddenTrace; } } hidden!();"
        )),
        Some("macro_rules")
    );
    assert_eq!(
        unsupported_macro(&rust_code_tokens("custom::format!(\"hidden\");")),
        Some("format")
    );
    assert_eq!(
        unsupported_macro(&rust_code_tokens(
            "use custom_macros::assert; assert!(true);"
        )),
        Some("assert")
    );
    assert!(contains_any_associated_open(&rust_code_tokens(
        "type Transaction = <Provider as Trait>::Transaction; Transaction::open();"
    )));
    assert_eq!(
        competing_shared_alias(&rust_code_tokens(
            "pub struct RowState; pub use self::RowState as r#SamplingTrace;"
        )),
        Some("SamplingTrace")
    );
    assert_eq!(
        competing_shared_alias(&rust_code_tokens(
            "pub struct RowCancellation; pub use self::RowCancellation as CancellationToken;"
        )),
        Some("CancellationToken")
    );
    assert_eq!(
        unsupported_attribute(&rust_code_tokens("#[path = \"elsewhere.rs\"] mod helper;")),
        Some("path")
    );
    assert_eq!(
        unsupported_attribute(&rust_code_tokens("#[custom_owner] struct Equation;")),
        Some("custom_owner")
    );
    assert_eq!(
        unsupported_attribute(&rust_code_tokens(
            "#[derive(Clone, CustomOwner)] struct Equation;"
        )),
        Some("CustomOwner")
    );
}

#[test]
fn algorithm_tree_scan_is_recursive_and_rejects_symlinks() -> Result<(), Box<dyn Error>> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "sim-sampler-ownership-{}-{nonce}",
        std::process::id()
    ));
    let nested = root.join("helper/nested");
    fs::create_dir_all(&nested)?;
    let nested_source = nested.join("row.rs");
    fs::write(&nested_source, "struct NestedTrace;")?;
    assert_eq!(rust_sources_below(&root, true)?, [nested_source]);

    #[cfg(unix)]
    {
        let symlink = root.join("helper-link");
        std::os::unix::fs::symlink(root.join("helper"), &symlink)?;
        let error = rust_sources_below(&root, true)
            .expect_err("algorithm symlinks must fail closed")
            .to_string();
        assert!(error.contains("forbidden symlink"));
    }

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn sampler_family_equations_have_one_owner_and_focused_adapters() -> Result<(), Box<dyn Error>> {
    let root = workspace()?;
    let algorithms = root.join("crates/sim_sampler/src/algorithms");
    let profile_path = root.join("crates/sim_sampler/src/sampling_profile.rs");
    let sources = rust_sources(&root)?;
    for symbol in ["standard_ancestral_step", "rectified_flow_ancestral_step"] {
        let mut definitions = Vec::new();
        for path in sources.iter().filter(|path| {
            !path
                .components()
                .any(|component| component.as_os_str() == "tests")
        }) {
            let tokens = rust_code_tokens(&fs::read_to_string(path)?);
            if tokens
                .windows(2)
                .any(|window| window[0] == "fn" && window[1] == symbol)
            {
                definitions.push(path);
            }
        }
        assert_eq!(
            definitions,
            [&profile_path],
            "{symbol} must have exactly one production owner"
        );
    }

    let euler_owner = algorithms.join("native_diffusion.rs");
    let euler_definitions = sources
        .iter()
        .filter(|path| {
            !path
                .components()
                .any(|component| component.as_os_str() == "tests")
        })
        .filter_map(|path| {
            let tokens = rust_code_tokens(&fs::read_to_string(path).ok()?);
            tokens
                .windows(2)
                .any(|window| window[0] == "fn" && window[1] == "sample_euler_canonical")
                .then_some(path)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        euler_definitions,
        [&euler_owner],
        "canonical Euler traversal must have exactly one production owner"
    );
    let ddim_source = fs::read_to_string(algorithms.join("ddim_sim_model_0159.rs"))?;
    assert!(ddim_source.contains("sample_euler_canonical("));
    for forbidden in [
        "sigmas.windows(2)",
        "derivative.mul_add",
        "(current - denoised) / sigma",
        "SamplingSession::new",
        "session.commit_step",
    ] {
        assert!(
            !ddim_source.contains(forbidden),
            "DDIM duplicates canonical Euler ownership through {forbidden}"
        );
    }

    let dpm_solver_owner = algorithms.join("dpm_solver.rs");
    for symbol in [
        "dpm_solver_first_order",
        "dpm_solver_first_intermediate",
        "dpm_solver_second_order",
        "dpm_solver_third_order",
    ] {
        let definitions = sources
            .iter()
            .filter(|path| {
                !path
                    .components()
                    .any(|component| component.as_os_str() == "tests")
            })
            .filter_map(|path| {
                let tokens = rust_code_tokens(&fs::read_to_string(path).ok()?);
                tokens
                    .windows(2)
                    .any(|window| window[0] == "fn" && window[1] == symbol)
                    .then_some(path)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            definitions,
            [&dpm_solver_owner],
            "{symbol} must have exactly one production owner"
        );
    }
    for relative in [
        "dpm_adaptive_sim_model_0164.rs",
        "dpm_fast_sim_model_0165.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        for required in [
            "dpm_solver_first_order",
            "dpm_solver_first_intermediate",
            "dpm_solver_second_order",
            "dpm_solver_third_order",
        ] {
            assert!(
                source.contains(required),
                "{relative} does not delegate {required}"
            );
        }
        assert!(
            !source.contains(".exp_m1()"),
            "{relative} retains a DPMSolver step equation"
        );
    }

    for relative in [
        "dpm_2_ancestral_sim_model_0163.rs",
        "dpm_adaptive_sim_model_0164.rs",
        "dpm_fast_sim_model_0165.rs",
        "dpmpp_2s_ancestral_sim_model_0172.rs",
        "dpmpp_2s_ancestral_cfg_pp_sim_model_0173.rs",
        "dpmpp_sde_sim_model_0176.rs",
        "euler_ancestral_sim_model_0180.rs",
        "euler_ancestral_cfg_pp_sim_model_0181.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        let tokens = rust_code_tokens(&source);
        assert!(
            tokens
                .iter()
                .any(|token| token == "standard_ancestral_step"),
            "{relative} does not call the standard ancestral owner"
        );
        assert!(
            !tokens
                .windows(2)
                .any(|window| window[0] == "fn" && window[1] == "standard_ancestral_step"),
            "{relative} retains a standard ancestral definition"
        );
        assert!(
            !tokens
                .windows(2)
                .any(|window| window[0] == "fn" && window[1] == "ancestral_step"),
            "{relative} retains a renamed standard ancestral definition"
        );
    }
    for relative in [
        "dpm_2_ancestral_sim_model_0163.rs",
        "dpmpp_2s_ancestral_sim_model_0172.rs",
        "euler_ancestral_sim_model_0180.rs",
    ] {
        let tokens = rust_code_tokens(&fs::read_to_string(algorithms.join(relative))?);
        assert!(
            tokens
                .iter()
                .any(|token| token == "rectified_flow_ancestral_step"),
            "{relative} does not call the rectified-flow ancestral owner"
        );
        assert!(
            !tokens.iter().any(|token| token == "downstep_ratio"),
            "{relative} retains the rectified-flow coefficient formula"
        );
    }

    for relative in [
        "dpmpp_2m_sde_sim_model_0168.rs",
        "dpmpp_sde_sim_model_0176.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        assert!(!source.contains("GenerationDeviceUnavailable"));
        assert!(!source.contains("generation_device() != DeviceId::CPU"));
        assert!(source.contains("placement.output_device() != output_device"));
    }
    for relative in [
        "dpmpp_2m_sde_gpu_sim_model_0169.rs",
        "dpmpp_3m_sde_gpu_sim_model_0175.rs",
        "dpmpp_sde_gpu_sim_model_0177.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        let tokens = rust_code_tokens(&source);
        assert!(source.contains("BackendCapabilityMatrix::for_native_device"));
        for forbidden in [
            "SamplingSession",
            "BrownianTree",
            "CompatibilityRngTransaction",
        ] {
            assert!(
                !tokens.iter().any(|token| token == forbidden),
                "{relative} retains family mechanic {forbidden}"
            );
        }
    }
    Ok(())
}

#[test]
fn cfg_pp_output_contract_has_one_owner_and_compatibility_aliases() -> Result<(), Box<dyn Error>> {
    let root = workspace()?;
    let sources = rust_sources(&root)?;
    let sampler_path = root.join("crates/sim_sampler/src/sampler.rs");
    let definitions = sources
        .iter()
        .filter(|path| {
            !path
                .components()
                .any(|component| component.as_os_str() == "tests")
        })
        .filter_map(|path| {
            let source = fs::read_to_string(path).ok()?;
            rust_code_tokens(&source)
                .windows(2)
                .any(|window| window[0] == "struct" && window[1] == "CfgPpDenoiserOutput")
                .then_some(path)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        definitions,
        [&sampler_path],
        "CfgPpDenoiserOutput must have exactly one production owner"
    );

    let sampler = fs::read_to_string(&sampler_path)?;
    assert!(sampler.contains("pub struct CfgPpDenoiserOutput"));
    assert!(sampler.contains("pub struct CfgPpDenoiserContractError"));
    assert!(sampler.contains("pub fn validate_cfg_pp_denoiser_output"));

    let algorithms = root.join("crates/sim_sampler/src/algorithms");
    for (relative, alias) in [
        (
            "dpmpp_2m_cfg_pp_sim_model_0167.rs",
            "Dpmpp2mCfgPpDenoiserOutput",
        ),
        (
            "dpmpp_2s_ancestral_cfg_pp_sim_model_0173.rs",
            "Dpmpp2sAncestralCfgPpDenoiserOutput",
        ),
        (
            "euler_ancestral_cfg_pp_sim_model_0181.rs",
            "EulerAncestralCfgPpDenoiserOutput",
        ),
        ("euler_cfg_pp_sim_model_0182.rs", "EulerCfgPpDenoiserOutput"),
        (
            "res_multistep_ancestral_cfg_pp_sim_model_0195.rs",
            "ResMultistepAncestralCfgPpDenoiserOutput",
        ),
        (
            "res_multistep_cfg_pp_sim_model_0196.rs",
            "ResMultistepCfgPpDenoiserOutput",
        ),
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        let tokens = rust_code_tokens(&source);
        assert!(
            source.contains(&format!("pub type {alias} = CfgPpDenoiserOutput;")),
            "{relative} must expose only a canonical compatibility alias"
        );
        assert!(
            !tokens
                .windows(2)
                .any(|window| window[0] == "struct" && window[1] == alias),
            "{relative} retains a competing CFG++ output definition"
        );
    }

    for relative in [
        "dpmpp_2m_cfg_pp_sim_model_0167.rs",
        "dpmpp_2s_ancestral_cfg_pp_sim_model_0173.rs",
        "euler_ancestral_cfg_pp_sim_model_0181.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        assert!(
            source.contains("validate_cfg_pp_denoiser_output"),
            "{relative} bypasses the canonical CFG++ output validator"
        );
    }

    let native_diffusion = fs::read_to_string(algorithms.join("native_diffusion.rs"))?;
    assert!(native_diffusion.contains("fn validate_euler_noise_generation_device"));
    assert!(native_diffusion.contains("BackendCapabilityMatrix::for_native_device"));
    for relative in [
        "euler_ancestral_sim_model_0180.rs",
        "euler_ancestral_cfg_pp_sim_model_0181.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        assert!(source.contains("validate_euler_noise_generation_device"));
        assert!(!source.contains("BackendCapabilityMatrix::for_native_device"));
    }
    Ok(())
}

#[test]
fn res_multistep_family_has_one_equation_and_transaction_owner() -> Result<(), Box<dyn Error>> {
    let algorithms = workspace()?.join("crates/sim_sampler/src/algorithms");
    let family = fs::read_to_string(algorithms.join("res_multistep_sim_model_0193.rs"))?;
    for owned in [
        "pub fn sample_res_multistep_family",
        "fn multistep(",
        "fn euler_step(",
        "fn add_source_noise(",
        "SamplingSession::new",
        "standard_ancestral_step",
        "open_transaction(",
    ] {
        assert!(
            family.contains(owned),
            "RES family is missing owner {owned}"
        );
    }

    for relative in [
        "res_multistep_ancestral_sim_model_0194.rs",
        "res_multistep_ancestral_cfg_pp_sim_model_0195.rs",
        "res_multistep_cfg_pp_sim_model_0196.rs",
    ] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        let tokens = rust_code_tokens(&source);
        assert!(source.contains("sample_res_multistep_family("));
        for forbidden in [
            "SamplingSession",
            "CompatibilityRngTransaction",
            "standard_ancestral_step",
            "tensor_to_f32",
            "multistep",
            "euler_step",
            "add_source_noise",
            "open_transaction",
            "draw_normal",
        ] {
            assert!(
                !tokens.iter().any(|token| token == forbidden),
                "{relative} retains RES family mechanic {forbidden}"
            );
        }
    }
    Ok(())
}

#[test]
fn sa_solver_family_has_one_traversal_equation_and_transaction_owner() -> Result<(), Box<dyn Error>>
{
    let algorithms = workspace()?.join("crates/sim_sampler/src/algorithms");
    let family = fs::read_to_string(algorithms.join("sa_solver_sim_model_0197.rs"))?;
    for owned in [
        "pub fn sample_sa_solver_family",
        "fn stochastic_adams_coefficients(",
        "fn solve_linear_system(",
        "SamplingSession::new",
        ".open_transaction(",
        ".draw_normal(",
        ".observe_step(",
    ] {
        assert!(
            family.contains(owned),
            "SA-Solver family is missing owner {owned}"
        );
    }
    let adapter = fs::read_to_string(algorithms.join("sa_solver_pece_sim_model_0198.rs"))?;
    assert!(adapter.contains("sample_sa_solver_family("));
    assert!(adapter.contains("SaSolverFamilyOptions::new(options, true)"));
    for forbidden in [
        "SamplingSession::new",
        "stochastic_adams_coefficients(",
        "solve_linear_system(",
        ".open_transaction(",
        ".draw_normal(",
        ".observe_step(",
        ".commit(",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "PECE adapter retains {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn seeds_two_family_and_phi_helpers_have_one_authoritative_owner() -> Result<(), Box<dyn Error>> {
    let root = workspace()?;
    let algorithms = root.join("crates/sim_sampler/src/algorithms");
    let family = fs::read_to_string(algorithms.join("seeds_2_sim_model_0199.rs"))?;
    for owned in [
        "fn sample_seeds_2_family<",
        "SamplingSession::new",
        ".open_transaction(",
        ".draw_normal(",
        ".observe_step(",
    ] {
        assert!(
            family.contains(owned),
            "SEEDS-2 family is missing owner {owned}"
        );
    }
    assert!(family.contains("validate_euler_noise_generation_device("));
    assert!(!family.contains("BackendCapabilityMatrix::for_native_device"));
    for relative in [
        "exp_heun_2_x0_sim_model_0183.rs",
        "exp_heun_2_x0_sde_sim_model_0184.rs",
    ] {
        let adapter = fs::read_to_string(algorithms.join(relative))?;
        assert!(adapter.contains("sample_seeds_2_"));
        for forbidden in [
            "SamplingSession::new",
            ".open_transaction(",
            ".draw_normal(",
            ".observe_step(",
            "BackendCapabilityMatrix::for_native_device",
        ] {
            assert!(
                !adapter.contains(forbidden),
                "{relative} retains {forbidden}"
            );
        }
    }

    let profile = fs::read_to_string(root.join("crates/sim_sampler/src/sampling_profile.rs"))?;
    for helper in [
        "pub fn exponential_integrator_phi_one",
        "pub fn exponential_integrator_phi_two",
    ] {
        assert_eq!(profile.matches(helper).count(), 1);
    }
    for relative in ["seeds_2_sim_model_0199.rs", "seeds_3_sim_model_0200.rs"] {
        let source = fs::read_to_string(algorithms.join(relative))?;
        assert!(source.contains("exponential_integrator_phi_one"));
        assert!(source.contains("exponential_integrator_phi_two"));
        assert!(!source.contains("fn exponential_integrator_phi_"));
    }
    Ok(())
}

#[test]
fn uni_pc_family_has_one_equation_traversal_and_commit_owner() -> Result<(), Box<dyn Error>> {
    let source_root = workspace()?.join("crates/sim_sampler/src");
    let algorithms = source_root.join("algorithms");
    let owner_path = algorithms.join("uni_pc_sim_model_0201.rs");
    let adapter_path = algorithms.join("uni_pc_bh2_sim_model_0202.rs");
    let owner = fs::read_to_string(&owner_path)?;
    let adapter = fs::read_to_string(&adapter_path)?;

    for symbol in [
        "sample_uni_pc_variant",
        "multistep_update",
        "bh_system_rhs",
        "solve_vandermonde",
        "shift_history",
    ] {
        let definitions = rust_sources_below(&source_root, false)?
            .into_iter()
            .filter_map(|path| {
                let tokens = rust_code_tokens(&fs::read_to_string(&path).ok()?);
                tokens
                    .windows(2)
                    .any(|window| window[0] == "fn" && window[1] == symbol)
                    .then_some(path)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            definitions.as_slice(),
            std::slice::from_ref(&owner_path),
            "{symbol} must have exactly one production owner"
        );
    }

    for owned in [
        "SamplingSession::new",
        ".observe_step(",
        ".commit(",
        "UniPcVariant::Bh1",
        "UniPcVariant::Bh2",
        "exponential_integrator_phi_one(",
    ] {
        assert!(owner.contains(owned), "UniPC owner is missing {owned}");
    }
    assert!(adapter.contains("sample_uni_pc_variant("));
    assert!(adapter.contains("UniPcVariant::Bh2"));
    for forbidden in [
        "fn multistep_update",
        "fn bh_system_rhs",
        "fn solve_vandermonde",
        "fn shift_history",
        "fn sigma_",
        "SamplingSession::new",
        ".observe_step(",
        ".commit(",
        "CancellationToken",
        "exponential_integrator_phi_one(",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "UniPC BH2 adapter retains {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn scheduler_request_has_one_sampler_to_penultimate_policy_owner() -> Result<(), Box<dyn Error>> {
    let source_root = workspace()?.join("crates/sim_sampler/src");
    let scheduler_path = source_root.join("scheduler.rs");
    let scheduler = fs::read_to_string(&scheduler_path)?;
    assert_eq!(scheduler.matches("pub fn for_sampling_plan(").count(), 1);
    assert!(scheduler.contains("\"dpm_2\" | \"dpm_2_ancestral\" | \"uni_pc\" | \"uni_pc_bh2\""));
    for path in rust_sources_below(&source_root, false)? {
        if path == scheduler_path {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        assert!(
            !source.contains("PenultimateSigmaPolicy::Discard"),
            "{} retains a competing sampler-to-penultimate-policy decision",
            path.display()
        );
    }
    Ok(())
}

#[test]
fn ancestral_coefficient_owner_preserves_pinned_boundaries() -> Result<(), Box<dyn Error>> {
    use sim_sampler::{rectified_flow_ancestral_step, standard_ancestral_step};

    let sigma_from = 2.0_f32;
    let sigma_to = 1.0_f32;
    let eta = 0.5_f32;
    let from_squared = sigma_from * sigma_from;
    let to_squared = sigma_to * sigma_to;
    let expected_up =
        sigma_to.min(eta * (to_squared * (from_squared - to_squared) / from_squared).sqrt());
    let expected_down = (to_squared - expected_up * expected_up).sqrt();
    let (down, up) = standard_ancestral_step(sigma_from, sigma_to, eta)?;
    assert_eq!(down.to_bits(), expected_down.to_bits());
    assert_eq!(up.to_bits(), expected_up.to_bits());

    assert_eq!(standard_ancestral_step(2.0, 0.0, 1.0)?, (0.0, 0.0));
    assert_eq!(standard_ancestral_step(1.0, 1.0, 1.0)?, (1.0, 0.0));
    assert_eq!(
        standard_ancestral_step(f32::NAN, 0.25, 0.0)?,
        (0.25, 0.0),
        "eta zero must return before sigma admission like pinned get_ancestral_step"
    );
    assert!(standard_ancestral_step(1.0, 2.0, 1.0).is_err());
    assert!(standard_ancestral_step(1.0, 0.5, f32::NAN).is_err());

    let negative_eta = -0.5_f32;
    let expected_negative_up = sigma_to
        .min(negative_eta * (to_squared * (from_squared - to_squared) / from_squared).sqrt());
    let expected_negative_down = (to_squared - expected_negative_up * expected_negative_up).sqrt();
    let (negative_down, negative_up) = standard_ancestral_step(sigma_from, sigma_to, negative_eta)?;
    assert_eq!(negative_down.to_bits(), expected_negative_down.to_bits());
    assert_eq!(negative_up.to_bits(), expected_negative_up.to_bits());
    assert!(negative_up < 0.0);

    let flow_eta = 0.75_f32;
    let ratio = 1.0 + (sigma_to / sigma_from - 1.0) * flow_eta;
    let flow_down = sigma_to * ratio;
    let alpha_to = 1.0 - sigma_to;
    let alpha_down = 1.0 - flow_down;
    let flow_renoise = (sigma_to * sigma_to
        - flow_down * flow_down * alpha_to * alpha_to / (alpha_down * alpha_down))
        .sqrt();
    let (actual_flow_down, actual_flow_renoise) =
        rectified_flow_ancestral_step(sigma_from, sigma_to, flow_eta)?;
    assert_eq!(actual_flow_down.to_bits(), flow_down.to_bits());
    assert_eq!(actual_flow_renoise.to_bits(), flow_renoise.to_bits());
    assert_eq!(rectified_flow_ancestral_step(2.0, 0.0, 1.0)?, (0.0, 0.0));
    assert!(rectified_flow_ancestral_step(1.0, 2.0, 1.0).is_err());
    Ok(())
}
