use gpui::BackgroundExecutor;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use smol::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Command, ExitStatus, Stdio},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    ffi::{OsStr, OsString},
    fmt,
    fs::File,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    time::Duration,
};

pub const ORACLE_FIXTURE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub enum OracleError {
    InvalidFixture(String),
    Json(serde_json::Error),
    Io(io::Error),
    Launch(String),
    Timeout(Duration),
    OutputLimit { limit: usize },
}

impl fmt::Display for OracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFixture(message) => write!(formatter, "invalid oracle fixture: {message}"),
            Self::Json(error) => write!(formatter, "oracle fixture JSON error: {error}"),
            Self::Io(error) => write!(formatter, "oracle I/O error: {error}"),
            Self::Launch(message) => write!(formatter, "source oracle launch failed: {message}"),
            Self::Timeout(duration) => {
                write!(
                    formatter,
                    "source oracle exceeded {} ms",
                    duration.as_millis()
                )
            }
            Self::OutputLimit { limit } => {
                write!(formatter, "source oracle output exceeded {limit} bytes")
            }
        }
    }
}

impl Error for OracleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for OracleError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<io::Error> for OracleError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type OracleResult<T> = Result<T, OracleError>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceProvenance {
    pub product: String,
    pub declared_version: Option<String>,
    pub tree_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandProvenance {
    pub adapter: String,
    pub program: String,
    pub arguments: Vec<String>,
    pub configuration: Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputDigest {
    pub name: String,
    pub sha256: String,
}

impl InputDigest {
    pub fn from_bytes(name: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            name: name.into(),
            sha256: sha256_hex(bytes),
        }
    }

    pub fn from_file(name: impl Into<String>, path: impl AsRef<Path>) -> OracleResult<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        Ok(Self {
            name: name.into(),
            sha256: format!("{:x}", hasher.finalize()),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentProvenance {
    pub operating_system: String,
    pub architecture: String,
    pub device: String,
    pub device_details: BTreeMap<String, String>,
    pub dependencies: BTreeMap<String, String>,
    pub network_access: bool,
    pub account_access: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NumericTolerance {
    Exact,
    Absolute { value: f64 },
    AbsoluteRelative { absolute: f64, relative: f64 },
}

impl NumericTolerance {
    fn validate(&self) -> OracleResult<()> {
        let valid = match self {
            Self::Exact => true,
            Self::Absolute { value } => value.is_finite() && *value >= 0.0,
            Self::AbsoluteRelative { absolute, relative } => {
                absolute.is_finite() && *absolute >= 0.0 && relative.is_finite() && *relative >= 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(OracleError::InvalidFixture(
                "numeric tolerances must be finite and non-negative".into(),
            ))
        }
    }

    fn matches(&self, expected: f64, actual: f64) -> bool {
        let difference = (expected - actual).abs();
        match self {
            Self::Exact => false,
            Self::Absolute { value } => difference <= *value,
            Self::AbsoluteRelative { absolute, relative } => {
                difference <= *absolute + *relative * expected.abs().max(actual.abs())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TolerancePolicy {
    pub default: NumericTolerance,
    pub json_pointer_overrides: BTreeMap<String, NumericTolerance>,
}

impl Default for TolerancePolicy {
    fn default() -> Self {
        Self {
            default: NumericTolerance::Exact,
            json_pointer_overrides: BTreeMap::new(),
        }
    }
}

impl TolerancePolicy {
    fn validate(&self) -> OracleResult<()> {
        self.default.validate()?;
        for (pointer, tolerance) in &self.json_pointer_overrides {
            validate_json_pointer(pointer)?;
            tolerance.validate()?;
        }
        Ok(())
    }

    fn at(&self, pointer: &str) -> &NumericTolerance {
        self.json_pointer_overrides
            .get(pointer)
            .unwrap_or(&self.default)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizationPolicy {
    pub remove_json_pointers: BTreeSet<String>,
    pub replacements: BTreeMap<String, Value>,
    pub unordered_array_pointers: BTreeSet<String>,
}

impl NormalizationPolicy {
    fn validate(&self) -> OracleResult<()> {
        if self.remove_json_pointers.contains("") || self.replacements.contains_key("") {
            return Err(OracleError::InvalidFixture(
                "normalization cannot remove or replace the document root".into(),
            ));
        }
        for pointer in &self.remove_json_pointers {
            validate_json_pointer(pointer)?;
        }
        for pointer in self
            .replacements
            .keys()
            .chain(self.unordered_array_pointers.iter())
        {
            validate_json_pointer(pointer)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnresolvedNondeterminism {
    pub json_pointer: String,
    pub cause: String,
    pub impact: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationBlocker {
    Dependency,
    Hardware,
    Credential,
    PaidService,
    Platform,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Observation {
    Observed {
        normalized_output: Value,
        output_sha256: String,
    },
    NotObserved {
        blocker: ObservationBlocker,
        detail: String,
        evidence: Vec<String>,
        uncertainty: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OracleFixture {
    pub schema_version: u32,
    pub fixture_id: String,
    pub feature_ids: Vec<String>,
    pub observation_kind: String,
    pub source: SourceProvenance,
    pub command: CommandProvenance,
    pub inputs: Vec<InputDigest>,
    pub environment: EnvironmentProvenance,
    pub normalization: NormalizationPolicy,
    pub tolerance: TolerancePolicy,
    pub unresolved_nondeterminism: Vec<UnresolvedNondeterminism>,
    pub observation: Observation,
}

impl OracleFixture {
    pub fn validate(&self) -> OracleResult<()> {
        if self.schema_version != ORACLE_FIXTURE_SCHEMA_VERSION {
            return Err(OracleError::InvalidFixture(format!(
                "fixture {} uses unsupported schema version {}",
                self.fixture_id, self.schema_version
            )));
        }
        validate_identifier(&self.fixture_id, "fixture ID")?;
        validate_non_empty(&self.observation_kind, "observation kind")?;
        validate_non_empty(&self.source.product, "source product")?;
        validate_sha256(&self.source.tree_sha256, "source tree fingerprint")?;
        if self
            .source
            .declared_version
            .as_ref()
            .is_some_and(|version| version.trim().is_empty())
        {
            return Err(OracleError::InvalidFixture(
                "declared source version cannot be blank".into(),
            ));
        }
        validate_non_empty(&self.command.adapter, "oracle adapter")?;
        validate_non_empty(&self.command.program, "oracle program")?;
        validate_non_empty(&self.environment.operating_system, "operating system")?;
        validate_non_empty(&self.environment.architecture, "architecture")?;
        validate_non_empty(&self.environment.device, "device")?;
        if self.environment.dependencies.is_empty()
            || self
                .environment
                .dependencies
                .iter()
                .any(|(name, version)| name.trim().is_empty() || version.trim().is_empty())
        {
            return Err(OracleError::InvalidFixture(
                "fixture environment must record non-empty dependency state".into(),
            ));
        }
        self.normalization.validate()?;
        self.tolerance.validate()?;

        let mut feature_ids = BTreeSet::new();
        for feature_id in &self.feature_ids {
            if !feature_id.starts_with("COMFY-") || !feature_ids.insert(feature_id) {
                return Err(OracleError::InvalidFixture(format!(
                    "invalid or duplicate feature ID {feature_id}"
                )));
            }
        }

        if self.inputs.is_empty() {
            return Err(OracleError::InvalidFixture(
                "fixtures must name at least one hashed input".into(),
            ));
        }
        let mut input_names = BTreeSet::new();
        for input in &self.inputs {
            validate_non_empty(&input.name, "input name")?;
            validate_sha256(&input.sha256, "input digest")?;
            if !input_names.insert(&input.name) {
                return Err(OracleError::InvalidFixture(format!(
                    "duplicate input name {}",
                    input.name
                )));
            }
        }

        for nondeterminism in &self.unresolved_nondeterminism {
            validate_json_pointer(&nondeterminism.json_pointer)?;
            validate_non_empty(&nondeterminism.cause, "nondeterminism cause")?;
            validate_non_empty(&nondeterminism.impact, "nondeterminism impact")?;
        }

        match &self.observation {
            Observation::Observed {
                normalized_output,
                output_sha256,
            } => {
                validate_sha256(output_sha256, "normalized output digest")?;
                let expected_digest = sha256_hex(&canonical_json_bytes(normalized_output)?);
                if output_sha256 != &expected_digest {
                    return Err(OracleError::InvalidFixture(format!(
                        "fixture {} normalized output digest does not match its content",
                        self.fixture_id
                    )));
                }
            }
            Observation::NotObserved {
                detail,
                evidence,
                uncertainty,
                ..
            } => {
                validate_non_empty(detail, "not-observed detail")?;
                validate_non_empty(uncertainty, "not-observed uncertainty")?;
                if evidence.is_empty() || evidence.iter().any(|item| item.trim().is_empty()) {
                    return Err(OracleError::InvalidFixture(
                        "not-observed fixtures require explicit non-empty evidence".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> OracleResult<Vec<u8>> {
        self.validate()?;
        let value = serde_json::to_value(self)?;
        let mut bytes = serde_json::to_vec_pretty(&canonicalize_json(&value))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn from_slice(bytes: &[u8]) -> OracleResult<Self> {
        let fixture: Self = serde_json::from_slice(bytes)?;
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn compare(&self, actual: &Value) -> OracleResult<ComparisonReport> {
        self.validate()?;
        let expected = match &self.observation {
            Observation::Observed {
                normalized_output, ..
            } => normalized_output,
            Observation::NotObserved { .. } => {
                return Err(OracleError::InvalidFixture(format!(
                    "fixture {} was not observed and cannot be used as a golden output",
                    self.fixture_id
                )));
            }
        };
        let actual = normalize_json(actual, &self.normalization)?;
        let mut mismatches = Vec::new();
        compare_json(expected, &actual, "", &self.tolerance, &mut mismatches);
        Ok(ComparisonReport { mismatches })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixtureBundle {
    pub schema_version: u32,
    pub fixtures: Vec<OracleFixture>,
}

impl FixtureBundle {
    pub fn validate(&self) -> OracleResult<()> {
        if self.schema_version != ORACLE_FIXTURE_SCHEMA_VERSION {
            return Err(OracleError::InvalidFixture(format!(
                "bundle uses unsupported schema version {}",
                self.schema_version
            )));
        }
        if self.fixtures.is_empty() {
            return Err(OracleError::InvalidFixture(
                "fixture bundle cannot be empty".into(),
            ));
        }
        let mut fixture_ids = BTreeSet::new();
        for fixture in &self.fixtures {
            fixture.validate()?;
            if !fixture_ids.insert(&fixture.fixture_id) {
                return Err(OracleError::InvalidFixture(format!(
                    "duplicate fixture ID {}",
                    fixture.fixture_id
                )));
            }
        }
        Ok(())
    }

    pub fn from_slice(bytes: &[u8]) -> OracleResult<Self> {
        let bundle: Self = serde_json::from_slice(bytes)?;
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn to_canonical_bytes(&self) -> OracleResult<Vec<u8>> {
        self.validate()?;
        let value = serde_json::to_value(self)?;
        let mut bytes = serde_json::to_vec_pretty(&canonicalize_json(&value))?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixtureDraft {
    pub fixture_id: String,
    pub feature_ids: Vec<String>,
    pub observation_kind: String,
    pub source: SourceProvenance,
    pub command: CommandProvenance,
    pub inputs: Vec<InputDigest>,
    pub environment: EnvironmentProvenance,
    pub normalization: NormalizationPolicy,
    pub tolerance: TolerancePolicy,
    pub unresolved_nondeterminism: Vec<UnresolvedNondeterminism>,
}

#[derive(Clone, Debug)]
pub struct OracleRecorder {
    draft: FixtureDraft,
}

impl OracleRecorder {
    pub fn new(draft: FixtureDraft) -> Self {
        Self { draft }
    }

    pub fn record_observed(self, raw_output: &Value) -> OracleResult<OracleFixture> {
        let normalized_output = normalize_json(raw_output, &self.draft.normalization)?;
        let output_sha256 = sha256_hex(&canonical_json_bytes(&normalized_output)?);
        self.finish(Observation::Observed {
            normalized_output,
            output_sha256,
        })
    }

    pub fn record_not_observed(
        self,
        blocker: ObservationBlocker,
        detail: impl Into<String>,
        evidence: Vec<String>,
        uncertainty: impl Into<String>,
    ) -> OracleResult<OracleFixture> {
        self.finish(Observation::NotObserved {
            blocker,
            detail: detail.into(),
            evidence,
            uncertainty: uncertainty.into(),
        })
    }

    fn finish(self, observation: Observation) -> OracleResult<OracleFixture> {
        let fixture = OracleFixture {
            schema_version: ORACLE_FIXTURE_SCHEMA_VERSION,
            fixture_id: self.draft.fixture_id,
            feature_ids: self.draft.feature_ids,
            observation_kind: self.draft.observation_kind,
            source: self.draft.source,
            command: self.draft.command,
            inputs: self.draft.inputs,
            environment: self.draft.environment,
            normalization: self.draft.normalization,
            tolerance: self.draft.tolerance,
            unresolved_nondeterminism: self.draft.unresolved_nondeterminism,
            observation,
        };
        fixture.validate()?;
        Ok(fixture)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonMismatch {
    pub json_pointer: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonReport {
    pub mismatches: Vec<ComparisonMismatch>,
}

impl ComparisonReport {
    pub fn matches(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub struct LaunchCommand {
    executable: OsString,
    arguments: Vec<OsString>,
    environment: Vec<(OsString, OsString)>,
    current_directory: PathBuf,
    stdin: Vec<u8>,
    inherit_environment: bool,
}

impl LaunchCommand {
    pub fn new(executable: impl AsRef<OsStr>, current_directory: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.as_ref().to_os_string(),
            arguments: Vec::new(),
            environment: Vec::new(),
            current_directory: current_directory.into(),
            stdin: Vec::new(),
            inherit_environment: false,
        }
    }

    pub fn argument(mut self, argument: impl AsRef<OsStr>) -> Self {
        self.arguments.push(argument.as_ref().to_os_string());
        self
    }

    pub fn environment(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.environment
            .push((key.as_ref().to_os_string(), value.as_ref().to_os_string()));
        self
    }

    pub fn stdin(mut self, stdin: Vec<u8>) -> Self {
        self.stdin = stdin;
        self
    }

    pub fn inherit_environment(mut self, inherit: bool) -> Self {
        self.inherit_environment = inherit;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessObservation {
    pub status_code: Option<i32>,
    pub success: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl ProcessObservation {
    pub fn stdout_json(&self) -> OracleResult<Value> {
        Ok(serde_json::from_slice(&self.stdout)?)
    }
}

#[derive(Clone, Debug)]
pub struct OracleLauncher {
    timeout: Duration,
    max_output_bytes: usize,
}

impl OracleLauncher {
    pub fn new(timeout: Duration, max_output_bytes: usize) -> OracleResult<Self> {
        if timeout.is_zero() || max_output_bytes == 0 {
            return Err(OracleError::Launch(
                "oracle timeout and output limit must be non-zero".into(),
            ));
        }
        Ok(Self {
            timeout,
            max_output_bytes,
        })
    }

    pub async fn run(
        &self,
        launch: LaunchCommand,
        executor: &BackgroundExecutor,
    ) -> OracleResult<ProcessObservation> {
        let (cancellation_sender, cancellation_receiver) = smol::channel::bounded(1);
        let process_task = smol::spawn(run_process(
            launch,
            self.max_output_bytes,
            cancellation_receiver,
        ));
        wait_for_process(process_task, cancellation_sender, self.timeout, executor).await
    }
}

async fn wait_for_process(
    process_task: smol::Task<OracleResult<ProcessObservation>>,
    cancellation_sender: smol::channel::Sender<()>,
    timeout: Duration,
    executor: &BackgroundExecutor,
) -> OracleResult<ProcessObservation> {
    let deadline = executor.now().checked_add(timeout).ok_or_else(|| {
        OracleError::Launch("oracle timeout exceeds the platform clock range".into())
    })?;
    let mut cancellation_sent = false;
    loop {
        if process_task.is_finished() {
            let result = process_task.await;
            if cancellation_sent {
                return Err(OracleError::Timeout(timeout));
            }
            return result;
        }
        if !cancellation_sent && executor.now() >= deadline {
            if cancellation_sender.try_send(()).is_ok() {
                cancellation_sent = true;
            } else if !process_task.is_finished() {
                return Err(OracleError::Launch(
                    "oracle cancellation channel closed before process completion".into(),
                ));
            }
        }
        executor.timer(Duration::from_millis(2)).await;
    }
}

async fn run_process(
    launch: LaunchCommand,
    max_output_bytes: usize,
    cancellation: smol::channel::Receiver<()>,
) -> OracleResult<ProcessObservation> {
    let current_directory = launch.current_directory.canonicalize()?;
    let mut command = Command::new(&launch.executable);
    command
        .args(&launch.arguments)
        .current_dir(current_directory)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if !launch.inherit_environment {
        command.env_clear();
    }
    command.envs(launch.environment);

    let mut child = command.spawn().map_err(|error| {
        OracleError::Launch(format!("could not start {:?}: {error}", launch.executable))
    })?;
    let mut child_stdin = child
        .stdin
        .take()
        .ok_or_else(|| OracleError::Launch("child stdin was unavailable".into()))?;
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| OracleError::Launch("child stdout was unavailable".into()))?;
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| OracleError::Launch("child stderr was unavailable".into()))?;

    let writer = smol::spawn(async move {
        child_stdin.write_all(&launch.stdin).await?;
        child_stdin.flush().await?;
        child_stdin.close().await
    });
    let stdout_reader = smol::spawn(read_bounded(child_stdout, max_output_bytes));
    let stderr_reader = smol::spawn(read_bounded(child_stderr, max_output_bytes));

    enum Completion {
        Status(io::Result<ExitStatus>),
        Cancelled,
    }
    let completion =
        smol::future::race(async { Completion::Status(child.status().await) }, async {
            match cancellation.recv().await {
                Ok(()) | Err(_) => Completion::Cancelled,
            }
        })
        .await;
    let status = match completion {
        Completion::Status(status) => status?,
        Completion::Cancelled => {
            match child.kill() {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::InvalidInput => {}
                Err(error) => return Err(OracleError::Io(error)),
            }
            child.status().await?
        }
    };

    writer.await?;
    let stdout = enforce_output_limit(stdout_reader.await?, max_output_bytes)?;
    let stderr = enforce_output_limit(stderr_reader.await?, max_output_bytes)?;
    Ok(process_observation(status, stdout, stderr))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReleaseBoundaryPolicy {
    pub schema_version: u32,
    pub development_only_packages: BTreeSet<String>,
    pub source_launcher_paths: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseBoundaryReport {
    pub development_packages_found: BTreeSet<String>,
    pub forbidden_normal_or_build_dependents: Vec<String>,
}

impl ReleaseBoundaryPolicy {
    pub fn validate(&self) -> OracleResult<()> {
        if self.schema_version != ORACLE_FIXTURE_SCHEMA_VERSION {
            return Err(OracleError::InvalidFixture(format!(
                "release boundary uses unsupported schema version {}",
                self.schema_version
            )));
        }
        if self.development_only_packages.is_empty() || self.source_launcher_paths.is_empty() {
            return Err(OracleError::InvalidFixture(
                "release boundary must name development packages and launcher paths".into(),
            ));
        }
        for path in &self.source_launcher_paths {
            if Path::new(path).is_absolute()
                || path.split('/').any(|component| component == "..")
                || !path.starts_with("crates/sim_test_support/")
            {
                return Err(OracleError::InvalidFixture(format!(
                    "source launcher path is outside development-only crates: {path}"
                )));
            }
        }
        Ok(())
    }

    pub fn verify_cargo_metadata(&self, metadata: &Value) -> OracleResult<ReleaseBoundaryReport> {
        self.validate()?;
        let packages = metadata
            .get("packages")
            .and_then(Value::as_array)
            .ok_or_else(|| OracleError::InvalidFixture("Cargo metadata lacks packages".into()))?;
        let mut development_packages_found = BTreeSet::new();
        let mut forbidden_normal_or_build_dependents = Vec::new();

        for package in packages {
            let package_name = package
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| OracleError::InvalidFixture("Cargo package lacks a name".into()))?;
            if self.development_only_packages.contains(package_name) {
                development_packages_found.insert(package_name.to_owned());
                let publish_disabled = package
                    .get("publish")
                    .is_some_and(|publish| publish.as_array().is_some_and(Vec::is_empty));
                if !publish_disabled {
                    forbidden_normal_or_build_dependents
                        .push(format!("{package_name} is publishable"));
                }
            }

            let dependencies = package
                .get("dependencies")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    OracleError::InvalidFixture(format!(
                        "Cargo package {package_name} lacks dependencies"
                    ))
                })?;
            for dependency in dependencies {
                let dependency_name = dependency.get("name").and_then(Value::as_str);
                let dependency_kind = dependency.get("kind").and_then(Value::as_str);
                if dependency_name.is_some_and(|name| self.development_only_packages.contains(name))
                    && dependency_kind != Some("dev")
                    && !self.development_only_packages.contains(package_name)
                {
                    forbidden_normal_or_build_dependents.push(format!(
                        "{package_name} has a normal/build dependency on {}",
                        dependency_name.unwrap_or("unknown")
                    ));
                }
            }
        }
        forbidden_normal_or_build_dependents.sort();
        if development_packages_found != self.development_only_packages {
            return Err(OracleError::InvalidFixture(format!(
                "Cargo metadata is missing development-only packages: {:?}",
                self.development_only_packages
                    .difference(&development_packages_found)
                    .collect::<Vec<_>>()
            )));
        }
        Ok(ReleaseBoundaryReport {
            development_packages_found,
            forbidden_normal_or_build_dependents,
        })
    }

    pub fn verify_launcher_layout(&self, workspace_root: impl AsRef<Path>) -> OracleResult<()> {
        self.validate()?;
        let workspace_root = workspace_root.as_ref().canonicalize()?;
        for launcher_path in &self.source_launcher_paths {
            let launcher_path = workspace_root.join(launcher_path).canonicalize()?;
            if !launcher_path.starts_with(&workspace_root) || !launcher_path.is_file() {
                return Err(OracleError::InvalidFixture(format!(
                    "source launcher is missing or outside the workspace: {}",
                    launcher_path.display()
                )));
            }
        }
        Ok(())
    }
}

impl ReleaseBoundaryReport {
    pub fn is_clean(&self) -> bool {
        self.forbidden_normal_or_build_dependents.is_empty()
    }
}

pub fn load_embedded_fixtures() -> OracleResult<FixtureBundle> {
    FixtureBundle::from_slice(include_bytes!("../fixtures/oracle-fixtures.json"))
}

pub fn load_tensor_signature_resolution_fixture() -> OracleResult<OracleFixture> {
    OracleFixture::from_slice(include_bytes!(
        "../fixtures/tensor_signatures/resolution-environment.json"
    ))
}

pub fn load_release_boundary_policy() -> OracleResult<ReleaseBoundaryPolicy> {
    let policy: ReleaseBoundaryPolicy =
        serde_json::from_slice(include_bytes!("../fixtures/release-boundary.json"))?;
    policy.validate()?;
    Ok(policy)
}

pub fn normalize_json(value: &Value, policy: &NormalizationPolicy) -> OracleResult<Value> {
    policy.validate()?;
    normalize_at(value, "", policy)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn normalize_at(value: &Value, pointer: &str, policy: &NormalizationPolicy) -> OracleResult<Value> {
    if let Some(replacement) = policy.replacements.get(pointer) {
        return Ok(canonicalize_json(replacement));
    }
    match value {
        Value::Object(object) => {
            let mut entries = BTreeMap::new();
            for (key, value) in object {
                let child_pointer = json_pointer_child(pointer, key);
                if policy.remove_json_pointers.contains(&child_pointer) {
                    continue;
                }
                entries.insert(key.clone(), normalize_at(value, &child_pointer, policy)?);
            }
            Ok(Value::Object(entries.into_iter().collect()))
        }
        Value::Array(array) => {
            let mut normalized = array
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    normalize_at(
                        value,
                        &json_pointer_child(pointer, &index.to_string()),
                        policy,
                    )
                })
                .collect::<OracleResult<Vec<_>>>()?;
            if policy.unordered_array_pointers.contains(pointer) {
                normalized.sort_by_cached_key(Value::to_string);
            }
            Ok(Value::Array(normalized))
        }
        _ => Ok(value.clone()),
    }
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        Value::Array(array) => Value::Array(array.iter().map(canonicalize_json).collect()),
        _ => value.clone(),
    }
}

fn canonical_json_bytes(value: &Value) -> OracleResult<Vec<u8>> {
    Ok(serde_json::to_vec(&canonicalize_json(value))?)
}

fn compare_json(
    expected: &Value,
    actual: &Value,
    pointer: &str,
    tolerance: &TolerancePolicy,
    mismatches: &mut Vec<ComparisonMismatch>,
) {
    if expected == actual {
        return;
    }
    match (expected, actual) {
        (Value::Number(expected), Value::Number(actual)) => {
            let matches = expected
                .as_f64()
                .zip(actual.as_f64())
                .is_some_and(|(expected, actual)| tolerance.at(pointer).matches(expected, actual));
            if !matches {
                push_mismatch(
                    pointer,
                    &Value::Number(expected.clone()),
                    &Value::Number(actual.clone()),
                    mismatches,
                );
            }
        }
        (Value::Object(expected), Value::Object(actual)) => {
            for key in expected
                .keys()
                .chain(actual.keys())
                .collect::<BTreeSet<_>>()
            {
                let child_pointer = json_pointer_child(pointer, key);
                match (expected.get(key), actual.get(key)) {
                    (Some(expected), Some(actual)) => {
                        compare_json(expected, actual, &child_pointer, tolerance, mismatches)
                    }
                    (Some(expected), None) => push_mismatch(
                        &child_pointer,
                        expected,
                        &Value::String("<missing>".into()),
                        mismatches,
                    ),
                    (None, Some(actual)) => push_mismatch(
                        &child_pointer,
                        &Value::String("<missing>".into()),
                        actual,
                        mismatches,
                    ),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(expected), Value::Array(actual)) => {
            if expected.len() != actual.len() {
                push_mismatch(
                    pointer,
                    &Value::Array(expected.clone()),
                    &Value::Array(actual.clone()),
                    mismatches,
                );
                return;
            }
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                compare_json(
                    expected,
                    actual,
                    &json_pointer_child(pointer, &index.to_string()),
                    tolerance,
                    mismatches,
                );
            }
        }
        _ => push_mismatch(pointer, expected, actual, mismatches),
    }
}

fn push_mismatch(
    pointer: &str,
    expected: &Value,
    actual: &Value,
    mismatches: &mut Vec<ComparisonMismatch>,
) {
    mismatches.push(ComparisonMismatch {
        json_pointer: pointer.to_owned(),
        expected: expected.to_string(),
        actual: actual.to_string(),
    });
}

fn json_pointer_child(parent: &str, component: &str) -> String {
    let component = component.replace('~', "~0").replace('/', "~1");
    format!("{parent}/{component}")
}

fn validate_json_pointer(pointer: &str) -> OracleResult<()> {
    if pointer.is_empty() || pointer.starts_with('/') {
        Ok(())
    } else {
        Err(OracleError::InvalidFixture(format!(
            "invalid JSON pointer {pointer}"
        )))
    }
}

fn validate_identifier(value: &str, label: &str) -> OracleResult<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(OracleError::InvalidFixture(format!(
            "{label} must contain only ASCII letters, digits, dash, underscore, or dot"
        )));
    }
    Ok(())
}

fn validate_non_empty(value: &str, label: &str) -> OracleResult<()> {
    if value.trim().is_empty() {
        Err(OracleError::InvalidFixture(format!(
            "{label} cannot be blank"
        )))
    } else {
        Ok(())
    }
}

fn validate_sha256(value: &str, label: &str) -> OracleResult<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(OracleError::InvalidFixture(format!(
            "{label} must be a 64-character SHA-256 digest"
        )))
    }
}

async fn read_bounded(reader: impl AsyncRead + Unpin, limit: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .await?;
    Ok(bytes)
}

fn enforce_output_limit(bytes: Vec<u8>, limit: usize) -> OracleResult<Vec<u8>> {
    if bytes.len() > limit {
        Err(OracleError::OutputLimit { limit })
    } else {
        Ok(bytes)
    }
}

fn process_observation(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> ProcessObservation {
    ProcessObservation {
        status_code: status.code(),
        success: status.success(),
        stdout,
        stderr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn fixture_draft() -> FixtureDraft {
        FixtureDraft {
            fixture_id: "recorder-test".into(),
            feature_ids: vec!["COMFY-API-0001".into()],
            observation_kind: "protocol".into(),
            source: SourceProvenance {
                product: "ComfyUI".into(),
                declared_version: Some("0.27.1".into()),
                tree_sha256: "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
                    .into(),
            },
            command: CommandProvenance {
                adapter: "unit-test".into(),
                program: "synthetic".into(),
                arguments: Vec::new(),
                configuration: serde_json::json!({}),
            },
            inputs: vec![InputDigest::from_bytes("input", b"input")],
            environment: EnvironmentProvenance {
                operating_system: std::env::consts::OS.into(),
                architecture: std::env::consts::ARCH.into(),
                device: "cpu".into(),
                device_details: BTreeMap::new(),
                dependencies: BTreeMap::from([("synthetic-runtime".into(), "1".into())]),
                network_access: false,
                account_access: false,
            },
            normalization: NormalizationPolicy::default(),
            tolerance: TolerancePolicy::default(),
            unresolved_nondeterminism: Vec::new(),
        }
    }

    #[test]
    fn sha256_matches_the_standard_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn recorder_normalizes_hashes_and_compares_outputs() -> OracleResult<()> {
        let mut draft = fixture_draft();
        draft
            .normalization
            .remove_json_pointers
            .insert("/timestamp".into());
        draft
            .normalization
            .unordered_array_pointers
            .insert("/items".into());
        draft.tolerance.json_pointer_overrides.insert(
            "/value".into(),
            NumericTolerance::AbsoluteRelative {
                absolute: 0.01,
                relative: 0.0,
            },
        );
        let fixture = OracleRecorder::new(draft).record_observed(&serde_json::json!({
            "timestamp": 123,
            "items": ["b", "a"],
            "value": 1.0
        }))?;
        assert!(
            fixture
                .compare(&serde_json::json!({
                    "timestamp": 999,
                    "items": ["a", "b"],
                    "value": 1.005
                }))?
                .matches()
        );
        assert!(
            !fixture
                .compare(&serde_json::json!({
                    "items": ["a", "b"],
                    "value": 1.02
                }))?
                .matches()
        );
        Ok(())
    }

    #[test]
    fn canonical_fixture_round_trip_is_byte_stable() -> OracleResult<()> {
        let fixture = OracleRecorder::new(fixture_draft())
            .record_observed(&serde_json::json!({"b": 2, "a": 1}))?;
        let first = fixture.to_canonical_bytes()?;
        let decoded = OracleFixture::from_slice(&first)?;
        assert_eq!(first, decoded.to_canonical_bytes()?);
        Ok(())
    }

    #[test]
    fn exact_numeric_comparison_preserves_integer_identity() -> OracleResult<()> {
        let fixture = OracleRecorder::new(fixture_draft())
            .record_observed(&serde_json::json!({"value": 9_007_199_254_740_992_u64}))?;
        assert!(
            !fixture
                .compare(&serde_json::json!({"value": 9_007_199_254_740_993_u64}))?
                .matches()
        );

        let maximum = OracleRecorder::new(fixture_draft())
            .record_observed(&serde_json::json!({"value": u64::MAX}))?;
        assert!(
            !maximum
                .compare(&serde_json::json!({"value": u64::MAX - 1}))?
                .matches()
        );
        Ok(())
    }

    #[test]
    fn normalization_cannot_replace_the_observed_document() {
        let mut policy = NormalizationPolicy::default();
        policy
            .replacements
            .insert(String::new(), serde_json::json!({"fabricated": true}));
        assert!(matches!(
            normalize_json(&serde_json::json!({"actual": false}), &policy),
            Err(OracleError::InvalidFixture(_))
        ));
    }

    #[test]
    fn fixture_environment_requires_dependency_state() {
        let mut draft = fixture_draft();
        draft.environment.dependencies.clear();
        assert!(matches!(
            OracleRecorder::new(draft).record_observed(&serde_json::json!({"value": 1})),
            Err(OracleError::InvalidFixture(_))
        ));
    }

    #[test]
    fn unavailable_observations_are_explicit_and_have_no_output() -> OracleResult<()> {
        let fixture = OracleRecorder::new(fixture_draft()).record_not_observed(
            ObservationBlocker::Hardware,
            "No CUDA device was installed on the recording host.",
            vec!["system_profiler reported no NVIDIA device".into()],
            "CUDA runtime behavior remains not observed.",
        )?;
        assert!(matches!(
            fixture.observation,
            Observation::NotObserved {
                blocker: ObservationBlocker::Hardware,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn checked_in_fixtures_are_complete_and_release_consumable() -> OracleResult<()> {
        let bundle = load_embedded_fixtures()?;
        assert!(
            bundle
                .fixtures
                .iter()
                .any(|fixture| matches!(fixture.observation, Observation::Observed { .. }))
        );
        assert!(bundle.fixtures.iter().any(|fixture| matches!(
            fixture.observation,
            Observation::NotObserved {
                blocker: ObservationBlocker::Hardware,
                ..
            }
        )));
        assert!(bundle.fixtures.iter().any(|fixture| matches!(
            fixture.observation,
            Observation::NotObserved {
                blocker: ObservationBlocker::Credential | ObservationBlocker::PaidService,
                ..
            }
        )));
        let canonical = bundle.to_canonical_bytes()?;
        assert_eq!(
            canonical,
            FixtureBundle::from_slice(&canonical)?.to_canonical_bytes()?
        );

        let reroute_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("frontend_reroute");
        let provenance: Value =
            serde_json::from_slice(&std::fs::read(reroute_root.join("provenance.json"))?)?;
        assert_eq!(provenance["normalization"], "byte_exact_copy");
        assert_eq!(provenance["tolerance"], "exact");
        let inputs = provenance["inputs"].as_array().ok_or_else(|| {
            OracleError::InvalidFixture("reroute provenance inputs are missing".into())
        })?;
        assert_eq!(inputs.len(), 8);
        for input in inputs {
            let path = input["path"].as_str().ok_or_else(|| {
                OracleError::InvalidFixture("reroute provenance path is missing".into())
            })?;
            if Path::new(path).is_absolute() || path.split('/').any(|component| component == "..") {
                return Err(OracleError::InvalidFixture(format!(
                    "reroute fixture path is unsafe: {path}"
                )));
            }
            let expected_digest = input["sha256"].as_str().ok_or_else(|| {
                OracleError::InvalidFixture("reroute provenance digest is missing".into())
            })?;
            validate_sha256(expected_digest, "reroute fixture digest")?;
            let bytes = std::fs::read(reroute_root.join(path))?;
            assert_eq!(sha256_hex(&bytes), expected_digest, "fixture {path}");
        }
        Ok(())
    }

    #[test]
    fn tensor_signature_blocker_is_structural_and_has_no_fabricated_output() -> OracleResult<()> {
        let fixture = load_tensor_signature_resolution_fixture()?;
        assert_eq!(
            fixture.fixture_id,
            "tensor-signature-resolution-environment-v1"
        );
        assert!(matches!(
            fixture.observation,
            Observation::NotObserved {
                blocker: ObservationBlocker::Dependency,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn cargo_metadata_proves_development_only_reverse_dependencies() -> OracleResult<()> {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| OracleError::Launch("workspace root is unavailable".into()))?;
        let output = smol::block_on(async {
            Command::new(env!("CARGO"))
                .args(["metadata", "--no-deps", "--format-version", "1"])
                .current_dir(workspace_root)
                .output()
                .await
        })?;
        if !output.status.success() {
            return Err(OracleError::Launch(format!(
                "cargo metadata failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)?;
        let policy = load_release_boundary_policy()?;
        assert_eq!(
            policy.development_only_packages,
            BTreeSet::from(["sim_test_support".to_owned()])
        );
        policy.verify_launcher_layout(workspace_root)?;
        let report = policy.verify_cargo_metadata(&metadata)?;
        assert!(
            report.is_clean(),
            "release dependency violations: {:?}",
            report.forbidden_normal_or_build_dependents
        );
        assert!(metadata["packages"].as_array().is_some_and(|packages| {
            packages
                .iter()
                .all(|package| package["name"].as_str() != Some("comfy_oracle"))
        }));
        assert!(!workspace_root.join("crates/comfy_oracle").exists());
        let workspace_manifest = std::fs::read_to_string(workspace_root.join("Cargo.toml"))?;
        assert!(!workspace_manifest.contains("comfy_oracle"));
        Ok(())
    }

    #[test]
    fn source_launcher_is_bounded_and_does_not_use_a_shell() -> OracleResult<()> {
        if std::env::var_os("SIM_ORACLE_HELPER").is_some() {
            return Ok(());
        }
        let executable = std::env::current_exe()?;
        let command = LaunchCommand::new(executable, env!("CARGO_MANIFEST_DIR"))
            .argument("--exact")
            .argument("oracle::tests::oracle_helper_process")
            .argument("--nocapture")
            .environment("SIM_ORACLE_HELPER", "1")
            .inherit_environment(true);
        let (_cancellation_sender, cancellation_receiver) = smol::channel::bounded(1);
        let output = smol::block_on(run_process(command, 64 * 1024, cancellation_receiver))?;
        assert!(output.success);
        assert!(String::from_utf8_lossy(&output.stdout).contains("oracle-helper-output"));
        Ok(())
    }

    #[test]
    fn source_process_is_killed_on_cancellation() -> OracleResult<()> {
        if std::env::var_os("SIM_ORACLE_HELPER").is_some() {
            return Ok(());
        }
        let executable = std::env::current_exe()?;
        let command = LaunchCommand::new(executable, env!("CARGO_MANIFEST_DIR"))
            .argument("--exact")
            .argument("oracle::tests::oracle_helper_process")
            .argument("--nocapture")
            .environment("SIM_ORACLE_HELPER", "1")
            .environment("SIM_ORACLE_SLEEP_MS", "250")
            .inherit_environment(true);
        let (cancellation_sender, cancellation_receiver) = smol::channel::bounded(1);
        let cancellation_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            smol::block_on(cancellation_sender.send(()))
                .map_err(|error| OracleError::Launch(error.to_string()))
        });
        let output = smol::block_on(run_process(command, 64 * 1024, cancellation_receiver))?;
        match cancellation_thread.join() {
            Ok(result) => result?,
            Err(_) => {
                return Err(OracleError::Launch(
                    "cancellation helper thread panicked".into(),
                ));
            }
        }
        assert!(!output.success);
        Ok(())
    }

    #[test]
    fn source_launcher_rejects_excess_output() -> OracleResult<()> {
        if std::env::var_os("SIM_ORACLE_HELPER").is_some() {
            return Ok(());
        }
        let executable = std::env::current_exe()?;
        let command = LaunchCommand::new(executable, env!("CARGO_MANIFEST_DIR"))
            .argument("--exact")
            .argument("oracle::tests::oracle_helper_process")
            .argument("--nocapture")
            .environment("SIM_ORACLE_HELPER", "1")
            .inherit_environment(true);
        let (_cancellation_sender, cancellation_receiver) = smol::channel::bounded(1);
        let result = smol::block_on(run_process(command, 16, cancellation_receiver));
        assert!(
            matches!(result, Err(OracleError::OutputLimit { limit: 16 })),
            "{result:?}"
        );
        Ok(())
    }

    #[gpui::test]
    async fn source_launcher_timeout_uses_the_gpui_executor(executor: BackgroundExecutor) {
        let (cancellation_sender, cancellation_receiver) = smol::channel::bounded(1);
        let process_task = smol::spawn(async move {
            match cancellation_receiver.recv().await {
                Ok(()) | Err(_) => Ok(ProcessObservation {
                    status_code: None,
                    success: false,
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                }),
            }
        });
        let result = wait_for_process(
            process_task,
            cancellation_sender,
            Duration::from_millis(10),
            &executor,
        )
        .await;
        assert!(matches!(result, Err(OracleError::Timeout(_))), "{result:?}");
    }

    #[test]
    fn oracle_helper_process() -> OracleResult<()> {
        if std::env::var_os("SIM_ORACLE_HELPER").is_none() {
            return Ok(());
        }
        if let Some(milliseconds) = std::env::var_os("SIM_ORACLE_SLEEP_MS") {
            let milliseconds = milliseconds
                .to_string_lossy()
                .parse::<u64>()
                .map_err(|error| OracleError::Launch(error.to_string()))?;
            thread::sleep(Duration::from_millis(milliseconds));
        }
        println!("oracle-helper-output");
        Ok(())
    }
}
