#[path = "support/executed_task_receipt.rs"]
mod executed_task_receipt;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_tensor::{
    GENERATED_MODULES, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS,
    validate_generated_operation_release_closure, validate_operation_contracts,
    validate_operation_resolution_evidence,
};
use smol::process::Command;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
};

const EXPECTED_CALLABLE_COUNT: usize = 511;
const EXPECTED_EXTERNAL_DISPOSITION_COUNT: usize = 7;
const EXPECTED_REFERENCE_COUNT: usize = 82;
const EXPECTED_RESOLUTION_MODULE_COUNT: usize = 51;
const EXPECTED_REFERENCE_MODULE_COUNT: usize = 9;
const BASELINES: &str = "crates/sim_test_support/fixtures/tensor_signatures/contracts";
const EVIDENCE: &str = "crates/sim_test_support/fixtures/tensor_operations";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CatalogDisposition {
    Callable,
    TypedReference,
    External,
}

struct CatalogInput {
    operation_id: &'static str,
    overload_id: &'static str,
    baseline_sha256: &'static str,
    owner: &'static str,
    module: &'static str,
    disposition: CatalogDisposition,
}

include!(concat!(env!("OUT_DIR"), "/generated_catalog_tests.rs"));

type TestCases = BTreeMap<String, Vec<String>>;

#[test]
fn val_tensor_001_exact_native_operation_breadth_closure() -> Result<(), Box<dyn std::error::Error>>
{
    assert_eq!(OPERATION_CONTRACTS.len(), 600);
    assert_eq!(CATALOG_INPUTS.len(), OPERATION_CONTRACTS.len());
    validate_operation_contracts(OPERATION_CONTRACTS)?;
    validate_generated_operation_release_closure(OPERATION_CONTRACTS)?;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let mut references = OPERATION_CONTRACTS
        .iter()
        .filter_map(|contract| contract.typed_reference())
        .map(|reference| (reference.operation_id(), reference))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(references.len(), EXPECTED_REFERENCE_COUNT);
    let mut resolutions = BTreeMap::new();
    let mut overloads = BTreeSet::new();
    assert_eq!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES.len(),
        EXPECTED_RESOLUTION_MODULE_COUNT
    );
    for slice in GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES {
        assert!(!slice.is_empty());
        for resolution in slice.iter() {
            assert_eq!(resolution.resolution_module, slice.module_name);
            assert!(overloads.insert(resolution.overload_id));
            assert!(
                resolutions
                    .insert(resolution.operation_id, resolution)
                    .is_none()
            );
        }
    }
    assert_eq!(resolutions.len(), EXPECTED_CALLABLE_COUNT);

    let mut fixture_digests = BTreeMap::new();
    let reference_modules = CATALOG_INPUTS
        .iter()
        .filter(|input| input.disposition == CatalogDisposition::TypedReference)
        .map(|input| input.module)
        .collect::<BTreeSet<_>>();
    assert_eq!(reference_modules.len(), EXPECTED_REFERENCE_MODULE_COUNT);
    let mut reference_evidence = BTreeMap::new();
    for module in reference_modules {
        let directory = workspace.join(EVIDENCE).join(module);
        assert_eq!(directory.canonicalize()?, directory);
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            assert!(
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension == "json")
            );
            let relative = path
                .strip_prefix(&workspace)?
                .to_str()
                .ok_or("invalid reference path")?;
            let evidence: Value =
                serde_json::from_slice(&read_input(&workspace, relative, &mut fixture_digests)?)?;
            let operation_id = evidence["operation_id"]
                .as_str()
                .ok_or("reference evidence must have an operation ID")?
                .to_owned();
            assert_eq!(evidence["resolution_module"], module);
            assert!(
                reference_evidence
                    .insert(operation_id, (relative.to_owned(), evidence))
                    .is_none()
            );
        }
    }
    assert_eq!(reference_evidence.len(), EXPECTED_REFERENCE_COUNT);

    let mut rows = Vec::with_capacity(CATALOG_INPUTS.len());
    let mut modules = BTreeSet::new();
    let mut external = 0;
    for input in CATALOG_INPUTS {
        let baseline_path = format!(
            "{BASELINES}/{}.json",
            input.operation_id.to_ascii_lowercase()
        );
        let baseline: Value = serde_json::from_slice(&read_input(
            &workspace,
            &baseline_path,
            &mut fixture_digests,
        )?)?;
        assert_eq!(fixture_digests[&baseline_path], input.baseline_sha256);
        assert_eq!(baseline["operation_id"], input.operation_id);
        assert_eq!(
            baseline["resolution"]["expected_resolution_module"],
            input.module
        );
        assert_eq!(
            baseline["resolution"]["resolution_owner_task_id"],
            input.owner
        );
        let (status, evidence) = match input.disposition {
            CatalogDisposition::Callable => {
                let resolution = resolutions
                    .remove(input.operation_id)
                    .ok_or("callable catalog entry lacks a compiled resolution")?;
                assert_eq!(resolution.resolution_module, input.module);
                assert_eq!(resolution.owner_task_id, input.owner);
                assert_eq!(resolution.baseline_overload_id, input.overload_id);
                assert_eq!(resolution.baseline_fixture_sha256, input.baseline_sha256);
                assert_ne!(resolution.evidence_fixture_sha256, input.baseline_sha256);
                validate_operation_resolution_evidence(&workspace, resolution)?;
                let evidence_path = Path::new(resolution.evidence_fixture);
                assert_eq!(
                    evidence_path.parent(),
                    Some(Path::new(EVIDENCE).join(input.module).as_path())
                );
                read_input(
                    &workspace,
                    resolution.evidence_fixture,
                    &mut fixture_digests,
                )?;
                assert_eq!(
                    fixture_digests[resolution.evidence_fixture],
                    resolution.evidence_fixture_sha256
                );
                modules.insert(input.module);
                ("compiled", Some(resolution.evidence_fixture.to_owned()))
            }
            CatalogDisposition::TypedReference => {
                let reference = references
                    .remove(input.operation_id)
                    .ok_or("reference catalog entry lacks a compiled typed value")?;
                let (path, evidence) = reference_evidence
                    .remove(input.operation_id)
                    .ok_or("reference catalog entry lacks evidence")?;
                assert_eq!(evidence["canonical_target"], reference.canonical_target());
                assert_eq!(evidence["executable"], false);
                assert_eq!(evidence["baseline_overload_id"], input.overload_id);
                assert_eq!(evidence["baseline_fixture_sha256"], input.baseline_sha256);
                assert_eq!(evidence["owner_task_id"], input.owner);
                assert_eq!(evidence["resolution_module"], input.module);
                assert_eq!(
                    evidence["reference_semantic"],
                    baseline["reference_semantic"]
                );
                modules.insert(input.module);
                ("typed-reference", Some(path))
            }
            CatalogDisposition::External => {
                assert_eq!(baseline["resolution"]["state"], "reclassified_external");
                external += 1;
                ("reclassified-external", None)
            }
        };
        rows.push(json!({
            "operation_id": input.operation_id,
            "module": input.module,
            "owner": input.owner,
            "baseline": baseline_path,
            "evidence": evidence,
            "status": status,
        }));
    }
    assert!(resolutions.is_empty() && references.is_empty() && reference_evidence.is_empty());
    assert_eq!(external, EXPECTED_EXTERNAL_DISPOSITION_COUNT);

    let cases = run_catalog_tests()?;
    assert_eq!(
        cases.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        modules
    );
    for module in modules {
        assert!(GENERATED_MODULES.contains(&format!("ops/{module}").as_str()));
        for directory in ["src/ops", "tests/ops"] {
            read_input(
                &workspace,
                &format!("crates/sim_tensor/{directory}/{module}.rs"),
                &mut fixture_digests,
            )?;
        }
    }
    for file in [
        "crates/sim_tensor/build.rs",
        "crates/sim_tensor/src/operation_contract_records.rs",
        "crates/sim_tensor/tests/breadth_closure.rs",
    ] {
        read_input(&workspace, file, &mut fixture_digests)?;
    }
    let expected_backends = [
        "amd_rocm_sim_model_0014",
        "apple_metal_mps_sim_model_0015",
        "cambricon_mlu_sim_model_0017",
        "cpu_sim_model_0016",
        "directml_sim_model_0018",
        "huawei_ascend_npu_sim_model_0019",
        "intel_xpu_sim_model_0021",
        "nvidia_cuda_sim_model_0022",
    ];
    for backend in expected_backends {
        let wrapper = format!("ops/backend_{backend}");
        assert!(GENERATED_MODULES.contains(&wrapper.as_str()));
        assert!(
            workspace
                .join(format!("crates/sim_tensor/src/ops/backend_{backend}.rs"))
                .is_file()
        );
        assert!(
            workspace
                .join(format!("crates/sim_tensor/src/backends/{backend}.rs"))
                .is_file()
        );
    }
    let passed = cases.values().map(Vec::len).sum::<usize>();
    let report = json!({
        "validation_id": "VAL-TENSOR-001",
        "validation": "VAL-TENSOR-001",
        "scope": "compiled operation contracts and typed references exercised by the native catalog fixtures",
        "environment": {"os": env::consts::OS, "arch": env::consts::ARCH, "backend": "native-rust-catalog-fixtures"},
        "fixture_digests": fixture_digests,
        "rows": rows,
        "module_cases": cases,
        "summary": {
            "callable_operations": EXPECTED_CALLABLE_COUNT,
            "typed_references": EXPECTED_REFERENCE_COUNT,
            "external_dispositions": external,
            "passed": passed,
            "failed": 0,
            "skipped": 0,
        },
        "skipped": [],
        "release_closure_claimed": false,
    });
    let target =
        env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    let directory = workspace.join(target).join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let destination = directory.join("val-tensor-001.json");
    let temporary = destination.with_extension(format!("{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, destination)?;

    Ok(())
}

fn read_input(
    workspace: &Path,
    relative: &str,
    digests: &mut BTreeMap<String, String>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let path = workspace.join(relative);
    assert!(path.starts_with(workspace));
    assert_eq!(path.canonicalize()?, path);
    let bytes = fs::read(path)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    assert!(digests.insert(relative.to_owned(), digest).is_none());
    Ok(bytes)
}

fn run_catalog_tests() -> Result<TestCases, Box<dyn std::error::Error>> {
    let executable = env::current_exe()?;
    let inventory = smol::block_on(
        Command::new(&executable)
            .args(["catalog_", "--list"])
            .output(),
    )?;
    assert!(inventory.status.success(), "tensor test inventory failed");
    let inventory = String::from_utf8(inventory.stdout)?;
    let listed = inventory
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect::<BTreeSet<_>>();

    // This runs the leaf tests even when the outer Cargo command selects only VAL-TENSOR-001.
    let execution = smol::block_on(
        Command::new(executable)
            .args([
                "catalog_",
                "--format",
                "pretty",
                "--color",
                "never",
                "--test-threads",
                "1",
            ])
            .output(),
    )?;
    let output = String::from_utf8(execution.stdout)?;
    assert!(
        execution.status.success(),
        "tensor leaf tests failed:\n{output}\n{}",
        String::from_utf8_lossy(&execution.stderr)
    );
    let mut passed = BTreeSet::new();
    let mut cases = BTreeMap::<_, Vec<_>>::new();
    for case in output
        .lines()
        .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... ok"))
    {
        assert!(passed.insert(case), "duplicate tensor test result {case}");
        let (module, _) = case.split_once("::").ok_or("unscoped tensor test result")?;
        let module = module
            .strip_prefix("catalog_")
            .ok_or("unexpected tensor test module")?;
        cases
            .entry(module.to_owned())
            .or_default()
            .push(case.to_owned());
    }
    assert_eq!(passed, listed, "tensor tests missing a successful result");
    Ok(cases)
}
