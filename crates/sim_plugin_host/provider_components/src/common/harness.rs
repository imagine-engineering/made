use std::{
    collections::BTreeSet,
    fmt, fs,
    path::{Path, PathBuf},
};

use super::{
    DiscoveredProviderComponent, FixtureError, Materialization, ProviderCase, TerminalDisposition,
    parse_fixture,
};

#[derive(Debug)]
pub struct ProviderRegistration {
    pub provider: &'static str,
    pub fixture_names: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActuationToken {
    operation: u64,
    worker_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionEvidence {
    pub provider: String,
    pub case: String,
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub retry_count: usize,
    pub poll_count: usize,
    pub cost_microunits: u64,
    pub progress_count: usize,
    pub materializations: Vec<Materialization>,
    pub worker_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConformanceError {
    Io {
        path: PathBuf,
        message: String,
    },
    Fixture(FixtureError),
    Discovery(String),
    ProviderMismatch {
        expected: String,
        actual: String,
    },
    FixtureMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    SourceChanged,
    FixtureChanged,
    Backpressure,
    InvalidToken,
    Cancelled,
    WorkerLost,
    WorkerUnavailable,
    Provider(String),
}

impl fmt::Display for ConformanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ConformanceError {}

impl From<FixtureError> for ConformanceError {
    fn from(error: FixtureError) -> Self {
        Self::Fixture(error)
    }
}

#[derive(Debug, Default)]
pub struct ScriptedActuator {
    next_operation: u64,
    worker_generation: u64,
    active: Option<ActiveActuation>,
    worker_available: bool,
    request_count: usize,
    published: Vec<Materialization>,
}

#[derive(Debug, Eq, PartialEq)]
struct ActiveActuation {
    token: ActuationToken,
    provider: String,
    case: String,
}

impl ScriptedActuator {
    pub fn new() -> Self {
        Self {
            next_operation: 0,
            worker_generation: 1,
            active: None,
            worker_available: true,
            request_count: 0,
            published: Vec::new(),
        }
    }

    pub fn request_count(&self) -> usize {
        self.request_count
    }

    pub fn published(&self) -> &[Materialization] {
        &self.published
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin(&mut self, case: &ProviderCase) -> Result<ActuationToken, ConformanceError> {
        validate_case(case)?;
        if !self.worker_available {
            return Err(ConformanceError::WorkerUnavailable);
        }
        if self.active.is_some() {
            return Err(ConformanceError::Backpressure);
        }
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or(ConformanceError::InvalidToken)?;
        let token = ActuationToken {
            operation: self.next_operation,
            worker_generation: self.worker_generation,
        };
        self.active = Some(ActiveActuation {
            token,
            provider: case.provider.clone(),
            case: case.name.clone(),
        });
        self.request_count += 1;
        Ok(token)
    }

    pub fn complete(
        &mut self,
        token: ActuationToken,
        case: &ProviderCase,
    ) -> Result<ExecutionEvidence, ConformanceError> {
        if self.active.as_ref().is_none_or(|active| {
            active.token != token
                || active.provider != case.provider
                || active.case != case.name
                || token.worker_generation != self.worker_generation
        }) {
            return Err(ConformanceError::InvalidToken);
        }
        let result = match &case.terminal {
            TerminalDisposition::Success => {
                let materializations = case.materializations.clone();
                Ok(ExecutionEvidence {
                    provider: case.provider.clone(),
                    case: case.name.clone(),
                    request_bytes: request_bytes(case)?,
                    response_bytes: aggregate_bytes(&case.response_chunks)?,
                    retry_count: case.retry_statuses.len(),
                    poll_count: case.poll_states.len(),
                    cost_microunits: case.cost_microunits,
                    progress_count: case.progress.len(),
                    materializations,
                    worker_generation: self.worker_generation,
                })
            }
            TerminalDisposition::Cancelled => Err(ConformanceError::Cancelled),
            TerminalDisposition::WorkerLost => {
                self.worker_available = false;
                Err(ConformanceError::WorkerLost)
            }
            TerminalDisposition::ProviderError(code) => {
                Err(ConformanceError::Provider(code.clone()))
            }
        };
        self.active = None;
        if let Ok(evidence) = &result {
            self.published.extend(evidence.materializations.clone());
        }
        result
    }

    pub fn execute(&mut self, case: &ProviderCase) -> Result<ExecutionEvidence, ConformanceError> {
        let token = self.begin(case)?;
        self.complete(token, case)
    }

    pub fn cancel(&mut self, token: ActuationToken) -> Result<(), ConformanceError> {
        if self.active.as_ref().map(|active| active.token) != Some(token) {
            return Err(ConformanceError::InvalidToken);
        }
        self.active = None;
        Ok(())
    }

    pub fn lose_worker(&mut self, token: ActuationToken) -> Result<(), ConformanceError> {
        if self.active.as_ref().map(|active| active.token) != Some(token) {
            return Err(ConformanceError::InvalidToken);
        }
        self.active = None;
        self.worker_available = false;
        Ok(())
    }

    pub fn restart_worker(&mut self) -> Result<u64, ConformanceError> {
        if self.worker_available || self.active.is_some() {
            return Err(ConformanceError::InvalidToken);
        }
        self.worker_generation = self
            .worker_generation
            .checked_add(1)
            .ok_or(ConformanceError::InvalidToken)?;
        self.worker_available = true;
        Ok(self.worker_generation)
    }
}

pub fn run_discovered_provider(
    component: &DiscoveredProviderComponent,
) -> Result<Vec<ExecutionEvidence>, ConformanceError> {
    validate_provider_name(component.registration.provider)?;
    let fixture_names = component
        .fixture_files
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    let expected_names = component
        .registration
        .fixture_names
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if fixture_names != expected_names {
        return Err(ConformanceError::FixtureMismatch {
            expected: expected_names,
            actual: fixture_names,
        });
    }

    let source_before = read_file(Path::new(component.source_file))?;
    let source_identity = conformance_fingerprint(&source_before);
    let mut fixture_identities = Vec::new();
    let mut cases = Vec::new();
    for fixture_name in component.fixture_files {
        validate_fixture_name(fixture_name)?;
        let fixture_path = Path::new(component.fixture_root).join(fixture_name);
        let fixture_bytes = read_file(&fixture_path)?;
        fixture_identities.push((fixture_path, conformance_fingerprint(&fixture_bytes)));
        let case = parse_fixture(&fixture_bytes)?;
        if case.provider != component.registration.provider {
            return Err(ConformanceError::ProviderMismatch {
                expected: component.registration.provider.to_string(),
                actual: case.provider,
            });
        }
        cases.push(case);
    }

    let mut actuator = ScriptedActuator::new();
    let mut evidence = Vec::new();
    for case in &cases {
        match &case.terminal {
            TerminalDisposition::Success => evidence.push(actuator.execute(case)?),
            TerminalDisposition::Cancelled => {
                if actuator.execute(case) != Err(ConformanceError::Cancelled) {
                    return Err(ConformanceError::Discovery(format!(
                        "{}:{} did not cancel",
                        case.provider, case.name
                    )));
                }
            }
            TerminalDisposition::WorkerLost => {
                if actuator.execute(case) != Err(ConformanceError::WorkerLost) {
                    return Err(ConformanceError::Discovery(format!(
                        "{}:{} did not report worker loss",
                        case.provider, case.name
                    )));
                }
                actuator.restart_worker()?;
            }
            TerminalDisposition::ProviderError(expected) => {
                if actuator.execute(case) != Err(ConformanceError::Provider(expected.clone())) {
                    return Err(ConformanceError::Discovery(format!(
                        "{}:{} returned the wrong provider error",
                        case.provider, case.name
                    )));
                }
            }
        }
    }

    if conformance_fingerprint(&read_file(Path::new(component.source_file))?) != source_identity {
        return Err(ConformanceError::SourceChanged);
    }
    for (fixture_path, identity) in fixture_identities {
        if conformance_fingerprint(&read_file(&fixture_path)?) != identity {
            return Err(ConformanceError::FixtureChanged);
        }
    }
    Ok(evidence)
}

pub fn conformance_fingerprint(bytes: &[u8]) -> u64 {
    // FNV-1a is fixed here so fixture provenance is reproducible across Rust toolchains.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn validate_case(case: &ProviderCase) -> Result<(), ConformanceError> {
    validate_provider_name(&case.provider)?;
    if case.path.is_empty() || !case.path.starts_with('/') {
        return Err(ConformanceError::Discovery(
            "invalid request path".to_string(),
        ));
    }
    let mut ports = BTreeSet::new();
    for materialization in &case.materializations {
        if !ports.insert(materialization.port.as_str()) {
            return Err(ConformanceError::Discovery(
                "duplicate materialization port".to_string(),
            ));
        }
    }
    request_bytes(case)?;
    aggregate_bytes(&case.response_chunks)?;
    Ok(())
}

fn request_bytes(case: &ProviderCase) -> Result<usize, ConformanceError> {
    let chunks = aggregate_bytes(&case.request_chunks)?;
    let multipart = case.multipart.iter().try_fold(0_usize, |total, part| {
        total
            .checked_add(part.bytes.len())
            .ok_or_else(|| ConformanceError::Discovery("request byte total overflowed".to_string()))
    })?;
    case.body
        .len()
        .checked_add(chunks)
        .and_then(|total| total.checked_add(multipart))
        .ok_or_else(|| ConformanceError::Discovery("request byte total overflowed".to_string()))
}

fn aggregate_bytes(chunks: &[Vec<u8>]) -> Result<usize, ConformanceError> {
    chunks.iter().try_fold(0_usize, |total, chunk| {
        total
            .checked_add(chunk.len())
            .ok_or_else(|| ConformanceError::Discovery("chunk byte total overflowed".to_string()))
    })
}

fn read_file(path: &Path) -> Result<Vec<u8>, ConformanceError> {
    fs::read(path).map_err(|error| ConformanceError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn validate_provider_name(provider: &str) -> Result<(), ConformanceError> {
    if provider.is_empty()
        || !provider
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ConformanceError::Discovery(format!(
            "invalid provider name {provider:?}"
        )));
    }
    Ok(())
}

fn validate_fixture_name(fixture_name: &str) -> Result<(), ConformanceError> {
    let path = Path::new(fixture_name);
    if path.components().count() != 1
        || path.extension().and_then(|extension| extension.to_str()) != Some("case")
    {
        return Err(ConformanceError::Discovery(format!(
            "invalid fixture name {fixture_name:?}"
        )));
    }
    Ok(())
}
