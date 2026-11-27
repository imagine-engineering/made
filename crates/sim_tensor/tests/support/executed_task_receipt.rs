use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
};

pub(super) type TestResult = Result<(), Box<dyn Error>>;
type Cases = BTreeMap<&'static str, bool>;
type Case = (&'static str, fn() -> TestResult);

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?
        .canonicalize()?)
}

pub(super) fn verify_resolutions(
    module: &str,
    owner: &str,
    operation_ids: &[&str],
) -> Result<BTreeMap<&'static str, &'static str>, Box<dyn Error>> {
    let workspace = workspace()?;
    let fixture_directory = Path::new("crates/sim_test_support/fixtures/tensor_operations");
    let fixture_directory = workspace.join(fixture_directory).join(module);
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.iter())
        .filter(|contract| contract.resolution_module == module)
        .collect::<Vec<_>>();
    let expected = operation_ids.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(expected.len(), operation_ids.len());
    assert_eq!(contracts.len(), expected.len());
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        expected
    );
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.overload_id)
            .collect::<BTreeSet<_>>()
            .len(),
        expected.len()
    );
    let mut digests = BTreeMap::new();
    for contract in contracts {
        assert_eq!(contract.owner_task_id, owner);
        assert_ne!(
            contract.baseline_fixture_sha256,
            contract.evidence_fixture_sha256
        );
        assert_ne!(contract.baseline_overload_id, contract.overload_id);
        let path = workspace.join(contract.evidence_fixture);
        assert_eq!(path.parent(), Some(fixture_directory.as_path()));
        assert_eq!(path.canonicalize()?, path);
        let bytes = fs::read(path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            contract.evidence_fixture_sha256
        );
        let fixture: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            fixture.get("operation_id").and_then(Value::as_str),
            Some(contract.operation_id)
        );
        digests.insert(contract.operation_id, contract.evidence_fixture_sha256);
    }
    Ok(digests)
}

pub(super) fn execute(cases: &[Case]) -> Result<Cases, Box<dyn Error>> {
    let mut results = BTreeMap::new();
    assert!(!cases.is_empty());
    for (name, run) in cases {
        run().map_err(|error| format!("{name}: {error}"))?;
        assert!(
            results.insert(*name, true).is_none(),
            "duplicate validation case {name}"
        );
    }
    Ok(results)
}

pub(super) fn publish(
    artifact_name: &str,
    validation_id: &str,
    scope: &str,
    source: (&str, &[u8]),
    fixture_digests: &BTreeMap<&str, &str>,
    cases: &Cases,
) -> TestResult {
    let workspace = workspace()?;
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || workspace.join("target"),
        |directory| workspace.join(directory),
    );
    fs::create_dir_all(&target)?;
    let directory = target.canonicalize()?.join("comfy-parity");
    fs::create_dir_all(&directory)?;
    if directory.canonicalize()? != directory {
        return Err("invalid tensor task receipt directory".into());
    }

    let lock_path = directory.join("tensor-task-receipts.lock");
    let destination = directory.join(artifact_name);
    let temporary = destination.with_extension("json.tmp");
    if [&lock_path, &destination, &temporary]
        .into_iter()
        .any(|path| path.is_symlink())
    {
        return Err("invalid tensor task receipt output path".into());
    }
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    lock.lock()?;
    let artifact = json!({
        "schema_version": 1,
        "validation_id": validation_id,
        "validation": validation_id,
        "scope": scope,
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "development_oracle_executed": false,
        },
        "validation_source": {
            "path": source.0,
            "sha256": format!("{:x}", Sha256::digest(source.1)),
        },
        "fixture_digests": fixture_digests,
        "summary": {"passed": cases.len(), "failed": 0, "skipped": 0},
        "cases": cases,
        "skipped": [],
        "validation_closure": {"claimed": true, "stage": "task", "validated_scope": scope},
        "release_closure_claimed": false,
        "release_closure_required": true,
        "remaining_release_gates": ["remaining tensor and autograd leaves", "release qualification"],
    });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, destination)?;
    Ok(())
}
