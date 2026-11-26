use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    io::{self, ErrorKind},
    path::{Component, Path},
};

const VALIDATION: &str = "VAL-CLIP-001";
const PRODUCER: &str = "crates/sim_model/tests/support/clip_receipt.rs";
const CATALOG: &str = "crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv";
const TOKENIZER: &str = "comfy-parity-clip-tokenizer-foundation";
const SD1_TOKENIZER: &str = "comfy-parity-sd1-tokenizer-owner-consolidation";
const TEXT_ENCODER_BREADTH: &str = "comfy-parity-clip-text-encoder-breadth";
const ARCHITECTURE_OWNERS: [&str; 4] = [
    "comfy-parity-clip-text-encoder-t5-foundation",
    "comfy-parity-clip-text-encoder-decoder-foundation",
    "comfy-parity-clip-text-encoder-multimodal-foundation",
    "comfy-parity-clip-text-encoder-composite-adapters",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourceDigest {
    pub path: String,
    pub sha256: String,
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
pub struct TaskResult {
    pub status: String,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    case_ids: Vec<String>,
    implementations: Vec<SourceDigest>,
    #[serde(default)]
    validation_source: Option<SourceDigest>,
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
pub struct ContractResult {
    pub contract_id: String,
    pub task_id: String,
    pub source_sha256: String,
    pub symbol_sha256: String,
    pub status: String,
    pub case_ids: Vec<String>,
}

pub struct ClipEvidence {
    pub task_results: BTreeMap<String, TaskResult>,
    pub contracts: Vec<ContractResult>,
    witness: SourceDigest,
}

impl ClipEvidence {
    pub fn record(
        &mut self,
        results: impl IntoIterator<Item = (&'static str, Value)>,
        contracts: Vec<Value>,
    ) -> Result<(), Box<dyn Error>> {
        let results = results
            .into_iter()
            .map(|(task, result)| {
                let mut result: TaskResult = serde_json::from_value(result)?;
                result.validation_source = Some(self.witness.clone());
                Ok::<_, serde_json::Error>((task, result))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let contracts = contracts
            .into_iter()
            .map(serde_json::from_value::<ContractResult>)
            .collect::<Result<Vec<_>, _>>()?;
        if contracts
            .iter()
            .any(|contract| !results.contains_key(contract.task_id.as_str()))
        {
            return Err("CLIP contract has no contributing task result".into());
        }
        if ARCHITECTURE_OWNERS.iter().any(|task| {
            let Some(incoming) = results.get(task) else {
                return false;
            };
            self.task_results.get(*task) != Some(incoming)
                || contracts_for_task(&self.contracts, task) != contracts_for_task(&contracts, task)
        }) {
            self.task_results.remove(TEXT_ENCODER_BREADTH);
        }
        self.contracts
            .retain(|contract| !results.contains_key(contract.task_id.as_str()));
        self.contracts.extend(contracts);
        self.task_results.extend(
            results
                .into_iter()
                .map(|(task, result)| (task.to_owned(), result)),
        );
        Ok(())
    }
}

fn contracts_for_task<'a>(records: &'a [ContractResult], task: &str) -> Vec<&'a ContractResult> {
    let mut selected = records
        .iter()
        .filter(|contract| contract.task_id == task)
        .collect::<Vec<_>>();
    selected.sort_unstable_by(|left, right| left.contract_id.cmp(&right.contract_id));
    selected
}

pub fn update(
    workspace: &Path,
    witness: &str,
    apply: impl FnOnce(&mut ClipEvidence) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let workspace = workspace.canonicalize()?;
    let directory = workspace.join("target/comfy-parity");
    fs::create_dir_all(&directory)?;
    let lock_path = directory.join("val-clip-001.lock");
    let destination = directory.join("val-clip-001.json");
    if directory.canonicalize()? != directory || lock_path.is_symlink() || destination.is_symlink()
    {
        return Err("invalid CLIP receipt output path".into());
    }
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    lock.lock()?;

    let producer = source_digest(&workspace, PRODUCER)?;
    let mut evidence = ClipEvidence {
        task_results: BTreeMap::new(),
        contracts: Vec::new(),
        witness: source_digest(&workspace, witness)?,
    };
    let catalog = Catalog::read(&workspace)?;
    match fs::read(&destination) {
        Ok(bytes) => {
            let mut prior: Value = serde_json::from_slice(&bytes)?;
            if prior["schema_version"] != 1 || prior["validation_id"] != VALIDATION {
                return Err("unexpected CLIP validation receipt schema".into());
            }
            if prior["implementation"] == json!(producer) {
                evidence.task_results = serde_json::from_value(prior["task_results"].take())?;
                evidence.contracts = serde_json::from_value(prior["contracts"].take())?;
                let current = catalog.current_tasks(&workspace, &evidence)?;
                evidence
                    .task_results
                    .retain(|task, _| current.contains(task));
                evidence
                    .contracts
                    .retain(|contract| current.contains(&contract.task_id));
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    apply(&mut evidence)?;
    let current = catalog.current_tasks(&workspace, &evidence)?;
    if current.len() != evidence.task_results.len() || current.is_empty() {
        return Err(
            "CLIP contribution does not match the current source and implementation evidence"
                .into(),
        );
    }
    evidence
        .contracts
        .sort_by(|left, right| left.contract_id.cmp(&right.contract_id));
    let passed = evidence
        .task_results
        .values()
        .try_fold(0usize, |total, result| {
            total
                .checked_add(result.passed)
                .ok_or("CLIP passed count overflowed")
        })?;
    let mut expected_tasks = catalog
        .contracts
        .values()
        .map(|contract| contract.task_id.as_str())
        .collect::<BTreeSet<_>>();
    expected_tasks.extend([
        SD1_TOKENIZER,
        TEXT_ENCODER_BREADTH,
        "comfy-parity-clip-owner-consolidation",
    ]);
    let remaining = expected_tasks
        .into_iter()
        .filter(|task| !current.contains(*task))
        .collect::<Vec<_>>();
    let artifact = json!({
        "schema_version": 1,
        "validation_id": VALIDATION,
        "overall_status": if remaining.is_empty() { "passed" } else { "partial" },
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": {"passed": passed, "failed": 0, "skipped": 0},
        "implementation": producer,
        "task_results": evidence.task_results,
        "contracts": evidence.contracts,
        "remaining_tasks": remaining,
    });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    let temporary = destination.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, destination)?;
    Ok(())
}

struct Catalog {
    contracts: BTreeMap<String, CatalogContract>,
}

struct CatalogContract {
    task_id: String,
    source: SourceDigest,
    symbol_sha256: String,
}

impl Catalog {
    fn read(workspace: &Path) -> Result<Self, Box<dyn Error>> {
        let records = parse_csv(&fs::read_to_string(workspace.join(CATALOG))?)?;
        let mut contracts = BTreeMap::new();
        for row in records.into_iter().skip(1) {
            let [
                contract_id,
                _,
                source_path,
                _,
                _,
                source_sha256,
                symbol_sha256,
                _,
                task_id,
                _,
                _,
                _,
                _,
                _,
                artifact,
            ] = row.as_slice()
            else {
                return Err("unexpected conditioning catalog field count".into());
            };
            if artifact == VALIDATION
                && contracts
                    .insert(
                        contract_id.clone(),
                        CatalogContract {
                            task_id: task_id.clone(),
                            source: SourceDigest {
                                path: source_path.clone(),
                                sha256: source_sha256.clone(),
                            },
                            symbol_sha256: symbol_sha256.clone(),
                        },
                    )
                    .is_some()
            {
                return Err(format!("duplicate CLIP catalog contract {contract_id}").into());
            }
        }
        Ok(Self { contracts })
    }

    fn current_tasks(
        &self,
        workspace: &Path,
        evidence: &ClipEvidence,
    ) -> Result<BTreeSet<String>, Box<dyn Error>> {
        let mut contracts = BTreeMap::<_, Vec<_>>::new();
        let mut contract_ids = BTreeSet::new();
        let mut case_ids = BTreeSet::new();
        for contract in &evidence.contracts {
            if !evidence.task_results.contains_key(&contract.task_id)
                || !contract_ids.insert(&contract.contract_id)
                || !unique_case_ids(&contract.case_ids)
                || !contract.case_ids.iter().all(|case| case_ids.insert(case))
            {
                return Err(format!(
                    "inconsistent CLIP contract evidence for {}",
                    contract.contract_id
                )
                .into());
            }
            contracts
                .entry(contract.task_id.as_str())
                .or_default()
                .push(contract);
        }
        let mut digests = BTreeMap::new();
        let mut current = BTreeSet::new();
        for (task, result) in &evidence.task_results {
            let task_contracts = contracts.get(task.as_str()).map_or(&[][..], Vec::as_slice);
            let expected = self
                .contracts
                .values()
                .filter(|contract| &contract.task_id == task)
                .count();
            let expected_passed = match task.as_str() {
                SD1_TOKENIZER => self
                    .contracts
                    .values()
                    .filter(|contract| contract.task_id == TOKENIZER)
                    .count(),
                TEXT_ENCODER_BREADTH => self
                    .contracts
                    .values()
                    .filter(|contract| ARCHITECTURE_OWNERS.contains(&contract.task_id.as_str()))
                    .count(),
                _ => expected,
            };
            if expected_passed == 0
                || result.passed != expected_passed
                || task_contracts.len() != expected
                || result.status != "passed"
                || result.failed != 0
                || result.skipped != 0
                || !unique_case_ids(&result.case_ids)
                || result.implementations.is_empty()
            {
                continue;
            }
            let Some(witness) = &result.validation_source else {
                continue;
            };
            let mut valid = true;
            for seal in result.implementations.iter().chain([witness]) {
                valid &= current_digest(workspace, seal, &mut digests)?;
            }
            for contract in task_contracts {
                let Some(row) = self.contracts.get(&contract.contract_id) else {
                    valid = false;
                    break;
                };
                valid &= row.task_id == contract.task_id
                    && row.source.sha256 == contract.source_sha256
                    && row.symbol_sha256 == contract.symbol_sha256
                    && contract.status == "passed"
                    && current_digest(workspace, &row.source, &mut digests)?;
            }
            if valid {
                current.insert(task.clone());
            }
        }
        if !current.contains(TOKENIZER) {
            current.remove(SD1_TOKENIZER);
        }
        if !ARCHITECTURE_OWNERS
            .iter()
            .all(|task| current.contains(*task))
        {
            current.remove(TEXT_ENCODER_BREADTH);
        }
        Ok(current)
    }
}

fn unique_case_ids(cases: &[String]) -> bool {
    !cases.is_empty()
        && cases.iter().all(|case| !case.is_empty())
        && cases.iter().collect::<BTreeSet<_>>().len() == cases.len()
}

pub fn implementation_digests(workspace: &Path, paths: &[&str]) -> io::Result<Vec<SourceDigest>> {
    let workspace = workspace.canonicalize()?;
    paths
        .iter()
        .map(|path| source_digest(&workspace, path))
        .collect()
}

fn source_digest(workspace: &Path, relative: &str) -> io::Result<SourceDigest> {
    let path = Path::new(relative);
    if !path
        .components()
        .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid CLIP receipt path",
        ));
    }
    let path = workspace.join(path);
    if path.canonicalize()? != path || !path.is_file() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid CLIP receipt input",
        ));
    }
    Ok(SourceDigest {
        path: relative.to_owned(),
        sha256: format!("{:x}", Sha256::digest(fs::read(path)?)),
    })
}

fn current_digest(
    workspace: &Path,
    seal: &SourceDigest,
    digests: &mut BTreeMap<String, Option<String>>,
) -> io::Result<bool> {
    use std::collections::btree_map::Entry;

    let digest = match digests.entry(seal.path.clone()) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            let digest = match source_digest(workspace, &seal.path) {
                Ok(source) => Some(source.sha256),
                Err(error)
                    if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::InvalidData) =>
                {
                    None
                }
                Err(error) => return Err(error),
            };
            entry.insert(digest)
        }
    };
    Ok(digest.as_ref() == Some(&seal.sha256))
}

pub fn parse_csv(source: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = source.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => record.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            '\r' if !quoted => {}
            value => field.push(value),
        }
    }
    if quoted {
        return Err("CSV ended inside a quoted field".into());
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}
