use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use smol::process::Command;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
};

pub struct CatalogDefinition {
    pub feature_id: &'static str,
    pub identity: &'static str,
    pub implementation_module: &'static str,
    pub source_ordinal: u16,
}

pub fn write_catalog_report(
    kind: &str,
    namespace: &str,
    scope: &str,
    expected_rows: usize,
    definitions: impl IntoIterator<Item = CatalogDefinition>,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog_key = format!("{kind}s");
    let validation_id = format!("VAL-{}-001", kind.to_ascii_uppercase());
    let manifest_test = format!("generated_{kind}_manifest_is_sorted_and_unique");
    let executable = env::current_exe()?;
    let row_filter = ["generated_", "--skip", &manifest_test];
    let inventory = smol::block_on(
        Command::new(&executable)
            .args(row_filter)
            .arg("--list")
            .output(),
    )?;
    assert!(inventory.status.success(), "{kind} test inventory failed");
    let inventory = String::from_utf8(inventory.stdout)?;
    let listed_tests = inventory
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect::<BTreeSet<_>>();

    // Execute registered fixtures even when the caller selects only the aggregate test.
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
        "{kind} row tests failed:\n{results}\n{}",
        String::from_utf8_lossy(&execution.stderr)
    );
    let mut passed = BTreeSet::new();
    let mut cases_by_module = BTreeMap::<_, Vec<_>>::new();
    for test in results
        .lines()
        .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... ok"))
    {
        assert!(passed.insert(test), "duplicate {kind} result {test}");
        let (module, _) = test.split_once("::").ok_or("unscoped row result")?;
        cases_by_module.entry(module).or_default().push(test);
    }
    assert_eq!(
        passed, listed_tests,
        "{kind} tests missing a successful result"
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let catalog_path = "crates/sim_test_support/parity/catalogs/native-compute-closure.json";
    let catalog_bytes = fs::read(workspace.join(catalog_path))?;
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    let rows = catalog[&catalog_key]
        .as_array()
        .ok_or("compute closure rows are missing")?;
    assert_eq!(rows.len(), expected_rows);
    let mut definitions_by_feature = BTreeMap::new();
    for definition in definitions {
        let feature_id = definition.feature_id;
        assert!(
            definitions_by_feature
                .insert(feature_id, definition)
                .is_none(),
            "duplicate compiled {kind} feature {feature_id}"
        );
    }
    assert_eq!(definitions_by_feature.len(), rows.len());
    let mut evidence_rows = Vec::new();
    let mut fixture_digests = BTreeMap::from([(
        catalog_path.to_owned(),
        format!("{:x}", Sha256::digest(&catalog_bytes)),
    )]);
    let fixture_root = workspace
        .join("crates/sim_test_support/fixtures")
        .join(&catalog_key);
    let module_prefix = format!("{namespace}/");
    for row in rows {
        let feature_id = row["feature_id"].as_str().ok_or("feature ID is missing")?;
        let definition = definitions_by_feature.remove(feature_id).ok_or_else(|| {
            format!("{kind} feature {feature_id} is repeated or absent from the compiled registry")
        })?;
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
        let module = row["row_module"].as_str().ok_or("row module is missing")?;
        assert!(sim_sampler::GENERATED_MODULES.contains(&module));
        let module_name = module
            .strip_prefix(&module_prefix)
            .ok_or("invalid row module namespace")?;
        let model_number = feature_id
            .strip_prefix("COMFY-MODEL-")
            .ok_or("invalid feature ID")?;
        let module_suffix = format!("_sim_model_{model_number}");
        let fixture_stem = module_name
            .strip_suffix(&module_suffix)
            .ok_or("row module ID mismatch")?;
        let cases = cases_by_module
            .remove(format!("generated_{module_name}").as_str())
            .ok_or_else(|| format!("no executed {kind} cases for {feature_id}"))?;

        let fixture = row["fixture"].as_str().ok_or("row fixture is missing")?;
        let fixture_directory =
            fixture_root.join(format!("{fixture_stem}_comfy_model_{model_number}"));
        let fixture_path = workspace.join(fixture);
        assert_eq!(fixture_path.parent(), Some(fixture_directory.as_path()));
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
            let input = workspace.join(&file);
            assert_eq!(input.canonicalize()?, input);
            let digest = format!("{:x}", Sha256::digest(fs::read(input)?));
            assert_eq!(
                row[digest_field].as_str(),
                Some(digest.as_str()),
                "stale {kind} input {file}"
            );
            assert!(fixture_digests.insert(file, digest).is_none());
        }
        evidence_rows.push(json!({
            "feature_id": feature_id,
            (kind): definition.identity,
            "fixture": fixture,
            "passed_tests": cases,
        }));
    }
    assert!(
        cases_by_module.is_empty(),
        "unmapped {kind} test modules: {cases_by_module:?}"
    );

    let report = json!({
        "validation_id": validation_id,
        "scope": scope,
        "environment": {"os": env::consts::OS, "arch": env::consts::ARCH, "backend": "native-rust-fixtures"},
        "fixture_digests": fixture_digests,
        "rows": evidence_rows,
        "summary": {(catalog_key): rows.len(), "passed": passed.len(), "failed": 0, "skipped": 0},
        "skipped": [],
    });
    let target =
        env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    let directory = workspace.join(target).join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let destination = directory.join(format!("{}.json", validation_id.to_ascii_lowercase()));
    let temporary = destination.with_extension(format!("{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, destination)?;
    Ok(())
}
