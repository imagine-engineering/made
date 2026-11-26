include!(concat!(env!("OUT_DIR"), "/generated_sampler_tests.rs"));

#[test]
fn generated_sampler_manifest_is_sorted_and_unique() {
    let modules = sim_sampler::GENERATED_MODULES
        .iter()
        .filter(|module| module.starts_with("algorithms/"))
        .copied()
        .collect::<Vec<_>>();
    assert!(modules.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn val_sampler_001_catalog_trajectories() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use smol::process::Command;
    use std::{
        collections::{BTreeMap, BTreeSet},
        env, fs,
        path::{Path, PathBuf},
    };

    let executable = env::current_exe()?;
    let row_filter = [
        "generated_",
        "--skip",
        "generated_sampler_manifest_is_sorted_and_unique",
    ];
    let inventory = smol::block_on(
        Command::new(&executable)
            .args(row_filter)
            .arg("--list")
            .output(),
    )?;
    assert!(inventory.status.success(), "sampler test inventory failed");
    let inventory = String::from_utf8(inventory.stdout)?;
    let listed_tests = inventory
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect::<BTreeSet<_>>();

    // Execute registered fixtures even when the caller selects only val_sampler_001.
    let execution = smol::block_on(
        Command::new(executable)
            .args(row_filter)
            .args([
                "--format",
                "pretty",
                "--color",
                "never",
                "--test-threads",
                "1",
            ])
            .output(),
    )?;
    let results = String::from_utf8(execution.stdout)?;
    assert!(
        execution.status.success(),
        "sampler row tests failed:\n{results}\n{}",
        String::from_utf8_lossy(&execution.stderr)
    );
    let mut passed = BTreeSet::new();
    let mut cases_by_module = BTreeMap::<_, Vec<_>>::new();
    for test in results
        .lines()
        .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... ok"))
    {
        assert!(passed.insert(test), "duplicate sampler result {test}");
        let (module, _) = test.split_once("::").ok_or("unscoped sampler result")?;
        cases_by_module.entry(module).or_default().push(test);
    }
    assert_eq!(
        passed, listed_tests,
        "sampler tests missing a successful result"
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let catalog_path = "crates/sim_test_support/parity/catalogs/native-compute-closure.json";
    let catalog_bytes = fs::read(workspace.join(catalog_path))?;
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    let rows = catalog["samplers"]
        .as_array()
        .ok_or("sampler closure rows are missing")?;
    let definitions = sim_sampler::GENERATED_SAMPLER_DEFINITIONS;
    assert_eq!(rows.len(), 44);
    assert_eq!(definitions.len(), rows.len());
    let mut feature_ids = BTreeSet::new();
    let mut evidence_rows = Vec::new();
    let mut fixture_digests = BTreeMap::from([(
        catalog_path.to_owned(),
        format!("{:x}", Sha256::digest(&catalog_bytes)),
    )]);
    let fixture_root = workspace.join("crates/sim_test_support/fixtures/samplers");
    for row in rows {
        let feature_id = row["feature_id"]
            .as_str()
            .ok_or("sampler feature ID is missing")?;
        let definition = definitions
            .iter()
            .find(|definition| definition.feature_id == feature_id)
            .ok_or("sampler feature is absent from the compiled registry")?;
        assert!(
            feature_ids.insert(feature_id),
            "duplicate sampler feature {feature_id}"
        );
        assert_eq!(row["identity"].as_str(), Some(definition.identity));
        assert_eq!(
            row["implementation_module"].as_str(),
            Some(definition.implementation_module)
        );
        assert_eq!(
            row["source_ordinal"].as_u64(),
            Some(u64::from(definition.source_ordinal))
        );
        assert_eq!(row["status"].as_str(), Some("compiled"));
        let module = row["row_module"]
            .as_str()
            .ok_or("sampler row module is missing")?;
        assert!(sim_sampler::GENERATED_MODULES.contains(&module));
        let module_name = module
            .strip_prefix("algorithms/")
            .ok_or("invalid sampler namespace")?;
        let model_number = feature_id
            .strip_prefix("COMFY-MODEL-")
            .ok_or("invalid sampler ID")?;
        let module_suffix = format!("_sim_model_{model_number}");
        let fixture_stem = module_name
            .strip_suffix(&module_suffix)
            .ok_or("sampler module ID mismatch")?;
        let cases = cases_by_module
            .remove(format!("generated_{module_name}").as_str())
            .ok_or_else(|| format!("no executed sampler cases for {feature_id}"))?;

        let fixture = row["fixture"]
            .as_str()
            .ok_or("sampler fixture is missing")?;
        let fixture_directory =
            fixture_root.join(format!("{fixture_stem}_comfy_model_{model_number}"));
        let fixture_path = workspace.join(fixture);
        assert_eq!(fixture_path.parent(), Some(fixture_directory.as_path()));
        assert_eq!(fixture_path.canonicalize()?, fixture_path);
        for (file, digest_field) in [
            (
                format!("crates/sim_sampler/src/{module}.rs"),
                "source_sha256",
            ),
            (
                format!("crates/sim_sampler/tests/{module}.rs"),
                "test_sha256",
            ),
            (fixture.to_owned(), "fixture_sha256"),
        ] {
            let digest = format!("{:x}", Sha256::digest(fs::read(workspace.join(&file))?));
            assert_eq!(
                row[digest_field].as_str(),
                Some(digest.as_str()),
                "stale sampler input {file}"
            );
            assert!(fixture_digests.insert(file, digest).is_none());
        }
        evidence_rows.push(json!({
            "feature_id": feature_id,
            "sampler": definition.identity,
            "fixture": fixture,
            "passed_tests": cases,
        }));
    }
    assert!(
        cases_by_module.is_empty(),
        "unmapped sampler test modules: {cases_by_module:?}"
    );

    let report = json!({
        "validation_id": "VAL-SAMPLER-001",
        "scope": "catalog sampler trajectories through generated native integration fixtures",
        "environment": {"os": env::consts::OS, "arch": env::consts::ARCH, "backend": "native-rust-fixtures"},
        "fixture_digests": fixture_digests,
        "rows": evidence_rows,
        "summary": {"samplers": rows.len(), "passed": passed.len(), "failed": 0, "skipped": 0},
        "skipped": [],
    });
    let target =
        env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    let directory = workspace.join(target).join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let destination = directory.join("val-sampler-001.json");
    let temporary = destination.with_extension(format!("{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, destination)?;
    Ok(())
}
