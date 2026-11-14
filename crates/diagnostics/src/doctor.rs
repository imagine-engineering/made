use anyhow::{Result, ensure};
use gpui::TaskExt as _;
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt::Write as _;

pub const MAX_CHECKS: usize = 64;
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct DiagnosticText(String);

impl DiagnosticText {
    pub fn new(value: impl AsRef<str>) -> Self {
        let value = util::redact::redact_command(value.as_ref());
        let mut result = String::new();
        let mut redact_next = false;
        let mut private_quote = None;
        for word in value.split_whitespace() {
            let normalized = word.trim_matches(['"', '\'', ',', '(', ')', '[', ']', '{', '}']);
            let lower = normalized.to_ascii_lowercase();
            let assignment = word.split_once(['=', ':']);
            let sensitive_assignment = assignment.is_some_and(|(key, _)| {
                let key = key.trim_matches(['"', '\'', ',', '{', '[']);
                util::redact::should_redact(&key.to_ascii_uppercase())
                    || key.eq_ignore_ascii_case("authorization")
            });
            let private_path = normalized.starts_with('/')
                || lower.contains("\\users\\")
                || normalized.starts_with("~/");
            let sensitive = redact_next
                || private_quote.is_some()
                || sensitive_assignment
                || lower.contains("://")
                || lower.starts_with("sk-")
                || private_path;
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(if sensitive { "[redacted]" } else { word });
            if let Some(quote) = private_quote {
                if word.ends_with(quote) {
                    private_quote = None;
                }
            } else if sensitive {
                for quote in ['"', '\''] {
                    if word.starts_with(quote) && !word.ends_with(quote) {
                        private_quote = Some(quote);
                    }
                }
            }
            redact_next = (sensitive_assignment
                && assignment.is_some_and(|(_, value)| {
                    value.trim_matches(['"', '\'', ',', '}']).is_empty()
                }))
                || lower == "bearer"
                || lower == "authorization:"
                || lower.ends_with("password:")
                || lower.ends_with("token:")
                || lower.ends_with("secret:");
        }
        Self(bounded(&result, 1024))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for DiagnosticText {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Ok(Self::new(value))
    }
}

fn bounded(value: &str, maximum: usize) -> String {
    if value.len() <= maximum {
        return value.to_owned();
    }
    let end = value.floor_char_boundary(maximum.saturating_sub(3));
    format!("{}…", &value[..end])
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CheckOutcome {
    Ready,
    Warning,
    Blocked,
    Failed,
    Untested,
    Unsupported,
    Cancelled,
}

impl CheckOutcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Passed",
            Self::Warning => "Warning",
            Self::Blocked => "Blocked",
            Self::Failed => "Failed",
            Self::Untested => "Unverified",
            Self::Unsupported => "Unsupported",
            Self::Cancelled => "Cancelled",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DoctorScope {
    pub product: DiagnosticText,
    pub project: DiagnosticText,
    pub worktree: DiagnosticText,
    pub host: DiagnosticText,
}

pub(crate) fn project_authority_description(
    observation: &project::DoctorProjectObservation,
) -> String {
    let host = match observation.host {
        project::DoctorExecutionHost::Local => "Local development host",
        project::DoctorExecutionHost::RemoteConnected => "Connected project development host",
        project::DoctorExecutionHost::RemoteUnavailable => "Project development host unavailable",
        project::DoctorExecutionHost::CollaborationGuest => {
            "Collaboration guest — host checks unavailable"
        }
    };
    let trust = match observation.trusted {
        Some(true) => "granted",
        Some(false) => "not granted",
        None => "unknown",
    };
    format!(
        "Host: {host}. Execution trust: {trust}. Project read-only mode: {}.{}",
        if observation.read_only { "on" } else { "off" },
        if observation.isolated_recipe_test {
            " Isolated recipe-test host."
        } else {
            ""
        }
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DoctorCheck {
    pub id: DiagnosticText,
    pub title: DiagnosticText,
    pub outcome: CheckOutcome,
    pub detail: DiagnosticText,
    pub remedy: Option<DiagnosticText>,
    pub observed_at: u64,
    pub historical: bool,
}

impl DoctorCheck {
    pub(crate) fn scope(&self) -> DoctorInvestigationScope {
        let identifier = self.id.as_str();
        if identifier.starts_with("rust.")
            || identifier.starts_with("jvm.")
            || identifier.starts_with("development.")
            || identifier.starts_with("coverage.rust.")
            || identifier.starts_with("coverage.jvm.")
            || identifier.starts_with("coverage.development.")
        {
            DoctorInvestigationScope::Development
        } else {
            DoctorInvestigationScope::Product
        }
    }

    pub(crate) fn state_label(&self) -> &'static str {
        if self.historical {
            "Stale"
        } else {
            self.outcome.label()
        }
    }

    pub(crate) fn priority(&self) -> u8 {
        if self.historical {
            return 3;
        }
        match self.outcome {
            CheckOutcome::Failed => 0,
            CheckOutcome::Blocked => 1,
            CheckOutcome::Warning => 2,
            CheckOutcome::Untested => 4,
            CheckOutcome::Unsupported => 5,
            CheckOutcome::Cancelled => 6,
            CheckOutcome::Ready => 7,
        }
    }

    pub(crate) fn evidence_source(&self) -> &'static str {
        let identifier = self.id.as_str();
        if identifier == "project.authority" {
            "Project trust and execution-host owner"
        } else if identifier == "rust.pin" {
            "Selected worktree's toolchain declaration"
        } else if identifier == "rust.selection" {
            "Project toolchain selection"
        } else if identifier.contains("lsp") || identifier.starts_with("jvm.") {
            "Project language-server and imported-project owners"
        } else if identifier.contains("debug") {
            "Project debugger session metadata"
        } else if identifier.contains("tests") || identifier.contains("run.") {
            "Project structured test and Task results"
        } else if identifier.starts_with("rust.") || identifier.starts_with("development.cargo") {
            "Cargo workspace owner's cached metadata"
        } else if identifier.starts_with("made.provider") {
            "Configured provider discovery and credential status"
        } else if identifier.starts_with("made.mcp") {
            "Project context-server status and authority"
        } else if identifier.starts_with("made.extension") || identifier.starts_with("made.plugin")
        {
            "Product extension and plugin owners"
        } else if identifier.starts_with("made.recipe") || identifier.starts_with("made.schedule") {
            "Project recipe and schedule owners"
        } else {
            "Selected Doctor collector; see the evidence identifier and details"
        }
    }

    pub(crate) fn next_step(&self) -> &str {
        if self.historical {
            return "Collect checks again for the current inputs. This retained observation cannot establish current health.";
        }
        if let Some(remedy) = &self.remedy {
            return remedy.as_str();
        }
        if self.outcome == CheckOutcome::Unsupported {
            return "Use a supported product or execution host. This check did not run; no alternate host was used.";
        }
        if self.id.as_str() == "project.authority" {
            return "Inspect project trust and host availability in the existing project controls. Doctor cannot grant execution permission.";
        }
        if self.id.as_str().starts_with("made.provider") {
            return "Inspect the selected model's settings. A separately reviewed connectivity probe can test a real response; configuration alone does not prove inference works.";
        }
        if self.scope() == DoctorInvestigationScope::Development {
            return "Inspect the relevant Cargo, toolchain or Tests surface. For a real build/test result, set up a development investigation and review the exact command and effects before running it.";
        }
        "Inspect the evidence and relevant settings. Advanced actions offer separately reviewed probes and recovery; collecting checks applies no repairs."
    }

    pub fn new(
        id: &str,
        title: &str,
        outcome: CheckOutcome,
        detail: &str,
        observed_at: u64,
    ) -> Self {
        Self {
            id: DiagnosticText::new(id),
            title: DiagnosticText::new(title),
            outcome,
            detail: DiagnosticText::new(detail),
            remedy: None,
            observed_at,
            historical: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DoctorReport {
    pub scope: DoctorScope,
    pub generation: u64,
    pub observed_at: u64,
    checks: Vec<DoctorCheck>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    investigation: Option<DoctorInvestigationSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparison: Option<DoctorComparisonRecord>,
}

impl DoctorReport {
    pub fn new(scope: DoctorScope, generation: u64, observed_at: u64) -> Self {
        Self {
            scope,
            generation,
            observed_at,
            checks: Vec::new(),
            investigation: None,
            comparison: None,
        }
    }

    pub fn with_checks(
        scope: DoctorScope,
        generation: u64,
        observed_at: u64,
        checks: Vec<DoctorCheck>,
    ) -> Result<Self> {
        let mut identities = std::collections::HashSet::new();
        for check in &checks {
            ensure!(
                check.id.as_str() != "coverage.report",
                "Reserved Doctor coverage identity"
            );
            ensure!(
                identities.insert(check.id.as_str().to_owned()),
                "Doctor check identity is duplicated"
            );
        }
        let total = checks.len();
        let mut report = Self::new(scope, generation, observed_at);
        report.validate()?;
        let mut omitted = 0;
        for check in checks {
            if report.checks.len() >= MAX_CHECKS {
                omitted += 1;
                continue;
            }
            report.checks.push(check);
            if serde_json::to_vec(&report)?.len() > MAX_REPORT_BYTES {
                report.checks.pop();
                omitted += 1;
            }
        }
        if omitted > 0 {
            loop {
                report.checks.push(DoctorCheck::new(
                    "coverage.report",
                    "Incomplete report coverage",
                    CheckOutcome::Untested,
                    &format!("{omitted} of {total} collected observations omitted by the report row or byte limit. Omitted observations are not represented by this report; no complete-health conclusion is available."),
                    observed_at,
                ));
                if report.checks.len() <= MAX_CHECKS
                    && serde_json::to_vec(&report)?.len() <= MAX_REPORT_BYTES
                {
                    break;
                }
                report.checks.pop();
                ensure!(
                    report.checks.pop().is_some(),
                    "Doctor scope leaves no room for coverage evidence"
                );
                omitted += 1;
            }
        }
        report.validate()?;
        Ok(report)
    }

    pub fn investigation(&self) -> Option<&DoctorInvestigationSummary> {
        self.investigation.as_ref()
    }

    pub fn set_investigation(&mut self, summary: DoctorInvestigationSummary) -> Result<()> {
        summary.validate()?;
        ensure!(
            summary.scope == self.scope && summary.generation == self.generation,
            "Investigation belongs to another report scope"
        );
        let previous = self.investigation.replace(summary);
        if let Err(error) = self.validate() {
            self.investigation = previous;
            return Err(error);
        }
        Ok(())
    }

    pub fn clear_comparison(&mut self) {
        self.comparison = None;
    }

    pub fn comparison(&self) -> Option<&DoctorComparisonRecord> {
        self.comparison.as_ref()
    }

    pub fn set_comparison(&mut self, comparison: DoctorComparisonRecord) -> Result<()> {
        let previous = self.comparison.replace(comparison);
        if let Err(error) = self.validate() {
            self.comparison = previous;
            return Err(error);
        }
        Ok(())
    }

    pub fn checks(&self) -> &[DoctorCheck] {
        &self.checks
    }

    pub fn push(&mut self, check: DoctorCheck) -> Result<()> {
        ensure!(self.checks.len() < MAX_CHECKS, "Doctor check limit reached");
        ensure!(
            !self.checks.iter().any(|existing| existing.id == check.id),
            "Doctor check identity is duplicated"
        );
        self.checks.push(check);
        if self.validate().is_err() {
            self.checks.pop();
            anyhow::bail!("Doctor report exceeds its size limit");
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.checks.len() <= MAX_CHECKS,
            "Doctor check limit exceeded"
        );
        for (index, check) in self.checks.iter().enumerate() {
            ensure!(
                !self
                    .checks
                    .iter()
                    .take(index)
                    .any(|existing| existing.id == check.id),
                "Duplicate Doctor check"
            );
        }
        if let Some(summary) = &self.investigation {
            summary.validate()?;
            ensure!(
                summary.scope == self.scope && summary.generation == self.generation,
                "Investigation summary scope is invalid"
            );
        }
        if let Some(comparison) = &self.comparison {
            comparison.validate()?;
            let investigation = self
                .investigation
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Comparison has no relevant investigation"))?;
            ensure!(
                comparison.verification_selection == investigation.selection,
                "Comparison belongs to another validation target"
            );
            ensure!(
                comparison
                    .verification_evidence
                    .iter()
                    .all(|identifier| investigation
                        .evidence()
                        .iter()
                        .any(|row| row.id == *identifier)),
                "Comparison verification evidence is unavailable"
            );
        }
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_REPORT_BYTES,
            "Doctor report exceeds its size limit"
        );
        Ok(())
    }

    pub fn summary(&self) -> String {
        let mut states = std::collections::BTreeMap::new();
        for check in &self.checks {
            *states.entry(check.state_label()).or_insert(0_usize) += 1;
        }
        let summary = states
            .into_iter()
            .map(|(state, count)| format!("{count} {}", state.to_lowercase()))
            .collect::<Vec<_>>()
            .join(" · ");
        format!(
            "{summary} · {} total checks{}",
            self.checks.len(),
            if self
                .checks
                .iter()
                .any(|check| check.id.as_str().starts_with("coverage."))
            {
                " · incomplete coverage"
            } else {
                ""
            }
        )
    }

    pub fn export(&self) -> Result<String> {
        self.validate()?;
        let mut output = String::new();
        writeln!(output, "# {} Doctor", self.scope.product.as_str())?;
        writeln!(
            output,
            "\nProject: {}\nWorktree: {}\nHost: {}\nObserved: {} UTC (Unix seconds)\n\n{}",
            self.scope.project.as_str(),
            self.scope.worktree.as_str(),
            self.scope.host.as_str(),
            self.observed_at,
            self.summary()
        )?;
        for check in &self.checks {
            writeln!(
                output,
                "\n- {} — {}{}: {}",
                check.state_label(),
                check.title.as_str(),
                if check.historical {
                    " (retained observation)"
                } else {
                    ""
                },
                check.detail.as_str()
            )?;
            if let Some(remedy) = &check.remedy {
                writeln!(output, "  Action: {}", remedy.as_str())?;
            }
        }
        if let Some(summary) = &self.investigation {
            writeln!(
                output,
                "\nSelected investigation evidence (redacted; observations are not proof of a verified fix):\n{}",
                serde_json::to_string(summary)?
            )?;
        }
        ensure!(
            output.len() <= MAX_REPORT_BYTES,
            "Doctor export exceeds its size limit"
        );
        Ok(output)
    }
}

#[cfg(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools"))]
pub(crate) fn append_collector_coverage(
    checks: &mut Vec<DoctorCheck>,
    id: &str,
    label: &str,
    total: usize,
    maximum: usize,
    observed_at: u64,
    historical: bool,
) {
    let omitted = total.saturating_sub(maximum);
    if omitted == 0 {
        return;
    }
    let mut check = DoctorCheck::new(
        id,
        "Incomplete collector coverage",
        CheckOutcome::Untested,
        &format!(
            "{omitted} of {total} scoped {label} entries omitted by the {maximum}-entry detail limit. Those entries were not checked; this inventory is incomplete."
        ),
        observed_at,
    );
    check.historical = historical;
    checks.push(check);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeExecutable {
    requested_path: std::path::PathBuf,
    canonical_path: std::path::PathBuf,
    inode: u64,
    modified: fs::MTime,
    size: u64,
    kind: DependencyProbe,
}

impl ProbeExecutable {
    pub fn path(&self) -> &std::path::Path {
        &self.canonical_path
    }
    pub fn kind(&self) -> DependencyProbe {
        self.kind
    }

    pub async fn inspect(
        kind: DependencyProbe,
        path: std::path::PathBuf,
        project_roots: &[std::path::PathBuf],
        filesystem: &dyn fs::Fs,
    ) -> Result<Self> {
        ensure!(
            path.is_absolute(),
            "Dependency executable must have an absolute path"
        );
        let canonical = filesystem.canonicalize(&path).await?;
        let metadata = filesystem
            .metadata(&canonical)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Dependency executable is missing"))?;
        ensure!(
            !metadata.is_dir && !metadata.is_fifo && metadata.is_executable,
            "Dependency path is not an executable regular file"
        );
        let mut header = [0; 4];
        let mut file = filesystem.open_sync(&canonical).await?;
        std::io::Read::read_exact(&mut file, &mut header)?;
        let mut canonical_roots = Vec::new();
        for root in project_roots {
            canonical_roots.push(filesystem.canonicalize(root).await?);
        }
        kind.validate_executable(&path, project_roots, &header)?;
        kind.validate_executable(&canonical, &canonical_roots, &header)?;
        ensure!(
            !matches!(kind, DependencyProbe::Java | DependencyProbe::JavaCompiler)
                || !matches!(canonical.to_str(), Some("/usr/bin/java" | "/usr/bin/javac")),
            "System Java launchers may trigger installation; select an installed JDK instead"
        );
        Ok(Self {
            requested_path: path,
            canonical_path: canonical,
            inode: metadata.inode,
            modified: metadata.mtime,
            size: metadata.len,
            kind,
        })
    }

    pub async fn revalidate(
        &self,
        project_roots: &[std::path::PathBuf],
        filesystem: &dyn fs::Fs,
    ) -> Result<()> {
        let current = Self::inspect(
            self.kind,
            self.requested_path.clone(),
            project_roots,
            filesystem,
        )
        .await?;
        ensure!(&current == self, "Reviewed dependency executable changed");
        Ok(())
    }

    pub fn shell_command(&self, shell: util::shell::ShellKind) -> Result<String> {
        let path = self.canonical_path.to_str().ok_or_else(|| {
            anyhow::anyhow!("Dependency executable has an unsupported path encoding")
        })?;
        let executable = shell
            .try_quote_prefix_aware(path)
            .ok_or_else(|| anyhow::anyhow!("Dependency executable cannot be safely quoted"))?;
        let arguments = self
            .kind
            .version_arguments()
            .iter()
            .map(|arg| {
                shell
                    .try_quote(arg)
                    .ok_or_else(|| anyhow::anyhow!("Dependency argument cannot be safely quoted"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(std::iter::once(executable.as_ref())
            .chain(arguments.iter().map(|arg| arg.as_ref()))
            .collect::<Vec<_>>()
            .join(" "))
    }
}

pub fn dependency_probe_host_supported(observation: &project::DoctorProjectObservation) -> bool {
    observation.supports_reviewed_probe() && observation.host == project::DoctorExecutionHost::Local
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub(crate) fn launch_dependency_probe(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    executable: ProbeExecutable,
    cancellation: async_channel::Sender<()>,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<ProbeResult>> {
    let observation = project
        .read(cx)
        .doctor_observation(directory.worktree_id, cx);
    if !dependency_probe_host_supported(&observation) {
        return gpui::Task::ready(Err(anyhow::anyhow!(
            "Dependency probe requires a trusted local host; remote executable identity and process-tree cancellation are not available through this probe path"
        )));
    }
    let roots = project
        .read(cx)
        .worktrees(cx)
        .map(|tree| tree.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    let filesystem = project.read(cx).fs().clone();
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let executor = cx.background_executor().clone();
        let preparation = async {
            executable.revalidate(&roots, filesystem.as_ref()).await?;
            let command = project.read_with(cx, |project, cx| {
                project.exec_read_only_probe_for_path(
                    executable.path(),
                    executable.kind().version_arguments(),
                    &directory,
                    cx,
                )
            })??;
            executable.revalidate(&roots, filesystem.as_ref()).await?;
            Ok(command)
        };
        let command =
            match prepare_probe_until_stop(preparation, cancelled.clone(), executor.clone()).await?
            {
                Ok(command) => command,
                Err(end) => {
                    return Ok(ProbeResult {
                        end,
                        detail: DiagnosticText::new(
                            "Probe setup stopped before process launch; availability is unverified",
                        ),
                    });
                }
            };
        let current = project.read_with(cx, |project, cx| {
            project.doctor_observation(directory.worktree_id, cx)
        })?;
        ensure!(
            dependency_probe_host_supported(&current),
            "Dependency probe authority changed before launch"
        );
        let executor = cx.background_executor().clone();
        let monitor = cx.spawn({
            let project = project.clone();
            let executor = executor.clone();
            async move |cx| {
                loop {
                    executor.timer(std::time::Duration::from_millis(100)).await;
                    let allowed = project.read_with(cx, |project, cx| {
                        dependency_probe_host_supported(
                            &project.doctor_observation(directory.worktree_id, cx),
                        )
                    });
                    if !matches!(allowed, Ok(true)) {
                        cancellation.close();
                        return;
                    }
                }
            }
        });
        let result = executor
            .spawn(run_bounded_probe(
                command,
                executor.clone(),
                cancelled,
                std::time::Duration::from_secs(10),
            ))
            .await;
        drop(monitor);
        result
    })
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
async fn prepare_probe_until_stop<T>(
    preparation: impl std::future::Future<Output = Result<T>>,
    cancelled: async_channel::Receiver<()>,
    executor: gpui::BackgroundExecutor,
) -> Result<std::result::Result<T, ProbeEnd>> {
    prepare_probe_until_stop_with_limit(
        preparation,
        cancelled,
        executor,
        std::time::Duration::from_secs(10),
    )
    .await
}

async fn prepare_probe_until_stop_with_limit<T>(
    preparation: impl std::future::Future<Output = Result<T>>,
    cancelled: async_channel::Receiver<()>,
    executor: gpui::BackgroundExecutor,
    duration: std::time::Duration,
) -> Result<std::result::Result<T, ProbeEnd>> {
    use futures::FutureExt;
    if cancelled.is_closed() || !cancelled.is_empty() {
        return Ok(Err(ProbeEnd::Cancelled));
    }
    let stop = wait_for_probe_stop(cancelled, executor, duration);
    match futures::future::select(preparation.boxed_local(), stop.boxed_local()).await {
        futures::future::Either::Left((result, _)) => result.map(Ok),
        futures::future::Either::Right((end, _)) => Ok(Err(end)),
    }
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
const MAX_PROBE_BYTES: usize = 16 * 1024;

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub struct DoctorDependencyProbeAction {
    executable: ProbeExecutable,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DoctorDependencyProbeAction {
    pub fn executable(&self) -> &ProbeExecutable {
        &self.executable
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub fn prepare_dependency_probe(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    kind: DependencyProbe,
    path: std::path::PathBuf,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorDependencyProbeAction>>> {
    let project = project.clone();
    let filesystem = project.read(cx).fs().clone();
    let roots = project
        .read(cx)
        .worktrees(cx)
        .map(|root| root.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    cx.spawn(async move |cx| {
        let binding = cx.update(|cx| {
            let observation = project.read(cx).doctor_observation(directory.worktree_id, cx);
            ensure!(dependency_probe_host_supported(&observation), "Only a trusted local host supports this reviewed version probe; remote execution is unavailable");
            if matches!(kind, DependencyProbe::Java | DependencyProbe::JavaCompiler) {
                ensure!(cfg!(feature = "jvm-tools"), "This product integration is disabled");
            } else {
                ensure!(cfg!(feature = "rust-tools"), "This product integration is disabled");
            }
            project_repair_binding(&project, directory.clone(), generation, cx)
        })?;
        let (cancel, cancelled) = async_channel::bounded(1);
        let inspection = prepare_probe_until_stop(
            ProbeExecutable::inspect(kind, path, &roots, filesystem.as_ref()),
            cancelled, cx.background_executor().clone(),
        ).await?;
        drop(cancel);
        let executable = inspection.map_err(|end| anyhow::anyhow!("Executable inspection stopped: {end:?}; no process ran"))?;
        cx.update(|cx| { ensure!(project_repair_binding(&project, directory, generation, cx)? == binding, "Selected scope changed during inspection"); Ok::<_, anyhow::Error>(()) })?;
        DoctorRepairReview::new(
            DoctorDependencyProbeAction { executable }, binding,
            DoctorCheck::new("development.tool.version", "Selected tool version availability", CheckOutcome::Untested,
                "No selected tool has been invoked", observed_at),
            DiagnosticText::new("Invoke only the selected installed native executable's fixed version arguments on the trusted selected local worktree. Direct execution with minimal host environment, no shell startup, project scripts or shims, no installation, downloads or settings changes. Ten-second setup/process deadlines and 16 KiB combined output limit; cancel the owned process. The native binary runs with user permissions; this is not a sandbox. Availability does not establish build/test/application correctness."),
        )
    })
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub fn launch_reviewed_dependency_probe(
    mut review: DoctorRepairReview<DoctorDependencyProbeAction>,
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancellation: async_channel::Sender<()>,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorDependencyProbeAction>>> {
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let binding = cx.update(|cx| project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx))?;
        review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
        let executor = cx.background_executor().clone();
        let started = executor.now();
        let operation = cx.update(|cx| {
            let owner = project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?;
            review.revalidate(&project_repair_binding(&owner, directory.clone(), generation, cx)?)?;
            Ok::<_, anyhow::Error>(launch_dependency_probe(&owner, directory.clone(), review.action().executable.clone(), cancellation.clone(), cancelled, cx))
        });
        let revoked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let monitor = cx.spawn({
            let project = project.clone(); let directory = directory.clone(); let executor = executor.clone(); let revoked = revoked.clone();
            async move |cx| {
                loop {
                    let current = cx.update(|cx| project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx));
                    if !matches!(current, Ok(current) if current == binding) { revoked.store(true, std::sync::atomic::Ordering::Release); cancellation.close(); return; }
                    executor.timer(std::time::Duration::from_millis(100)).await;
                }
            }
        });
        let result = match operation { Ok(operation) => operation.await, Err(error) => Err(error) };
        drop(monitor);
        let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
        let (mut outcome, mut detail) = match result {
            Ok(result) => (match result.end {
                ProbeEnd::Completed if result.detail.as_str() != "No version output observed" => CheckOutcome::Ready,
                ProbeEnd::Completed | ProbeEnd::TimedOut | ProbeEnd::OutputLimit => CheckOutcome::Untested,
                ProbeEnd::Failed => CheckOutcome::Failed,
                ProbeEnd::Cancelled => CheckOutcome::Cancelled,
            }, format!("Observed version probe: {}. This proves only selected-host availability/version; build, tests and application behavior remain unverified.", result.detail.as_str())),
            Err(error) => (CheckOutcome::Blocked, format!("Selected version probe blocked or failed: {error}; availability is unverified")),
        };
        if revoked.load(std::sync::atomic::Ordering::Acquire) {
            outcome = CheckOutcome::Untested;
            detail = "Selected authority changed; the owned version probe stopped and availability is incomplete".to_owned();
        }
        let operation_outcome = match outcome {
            CheckOutcome::Ready => DoctorRepairOperationOutcome::Completed,
            CheckOutcome::Cancelled => DoctorRepairOperationOutcome::Cancelled,
            _ => DoctorRepairOperationOutcome::Failed,
        };
        review.operation_finished(operation_outcome, DiagnosticText::new(&detail))?;
        if operation_outcome != DoctorRepairOperationOutcome::Cancelled {
            let current = cx.update(|cx| project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory, generation, cx));
            match current {
                Ok(binding) if &binding == review.binding() => review.observe_recheck(&binding, DoctorCheck::new("development.tool.version", "Selected tool version availability", outcome, &detail, observed_at))?,
                Ok(_) => review.recheck_unavailable(DiagnosticText::new("Scope or inputs changed; version availability remains unverified"))?,
                Err(error) => review.recheck_unavailable(DiagnosticText::new(error.to_string()))?,
            }
        }
        review.persist_outcome(product, &history, None, observed_at).await?;
        Ok(review)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeEnd {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    OutputLimit,
}

#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub end: ProbeEnd,
    pub detail: DiagnosticText,
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
struct ProbeChild {
    process: util::process::Child,
    terminated: bool,
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
impl Drop for ProbeChild {
    fn drop(&mut self) {
        if !self.terminated
            && let Err(error) = self.process.kill()
        {
            log::error!(
                "Doctor could not terminate its owned dependency probe: {}",
                DiagnosticText::new(error.to_string()).as_str()
            );
        }
    }
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
async fn read_probe_output(
    mut stream: impl futures_lite::AsyncRead + Unpin,
    count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    maximum_bytes: usize,
) -> Result<Vec<u8>> {
    use futures_lite::AsyncReadExt as _;
    let mut output = Vec::new();
    let mut buffer = [0; 1024];
    loop {
        let length = stream.read(&mut buffer).await?;
        if length == 0 {
            return Ok(output);
        }
        let previous = count.fetch_add(length, std::sync::atomic::Ordering::Relaxed);
        ensure!(
            previous.saturating_add(length) <= maximum_bytes,
            "Dependency probe output limit reached"
        );
        output.extend_from_slice(&buffer[..length]);
    }
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
pub(crate) async fn run_bounded_probe(
    command: std::process::Command,
    executor: gpui::BackgroundExecutor,
    cancelled: async_channel::Receiver<()>,
    timeout: std::time::Duration,
) -> Result<ProbeResult> {
    ensure!(
        timeout > std::time::Duration::ZERO && timeout <= std::time::Duration::from_secs(30),
        "Dependency probe deadline is invalid"
    );
    if cancelled.is_closed() || !cancelled.is_empty() {
        return Ok(ProbeResult {
            end: ProbeEnd::Cancelled,
            detail: DiagnosticText::new("Dependency probe cancelled before launch"),
        });
    }
    let stop = wait_for_probe_stop(cancelled, executor, timeout);
    run_probe_until_stop(command, stop).await
}

fn wait_for_probe_stop(
    cancelled: async_channel::Receiver<()>,
    executor: gpui::BackgroundExecutor,
    timeout: std::time::Duration,
) -> impl std::future::Future<Output = ProbeEnd> + Send {
    use futures::{FutureExt as _, future::Either};
    // Arm the deadline before dispatch: a fast child can become observable
    // before this future is first polled.
    let deadline = executor.timer(timeout);
    async move {
        match futures::future::select(cancelled.recv().boxed(), deadline.boxed()).await {
            Either::Left(_) => ProbeEnd::Cancelled,
            Either::Right(_) => ProbeEnd::TimedOut,
        }
    }
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
async fn run_probe_until_stop(
    command: std::process::Command,
    stop: impl std::future::Future<Output = ProbeEnd> + Send,
) -> Result<ProbeResult> {
    run_owned_command_until_stop(command, stop, MAX_PROBE_BYTES).await
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub(crate) async fn run_toolchain_command(
    command: std::process::Command,
    executor: gpui::BackgroundExecutor,
    cancelled: async_channel::Receiver<()>,
    timeout: std::time::Duration,
) -> Result<ProbeResult> {
    ensure!(
        timeout > std::time::Duration::ZERO && timeout <= std::time::Duration::from_secs(300),
        "Toolchain operation deadline is invalid"
    );
    if cancelled.is_closed() || !cancelled.is_empty() {
        return Ok(ProbeResult {
            end: ProbeEnd::Cancelled,
            detail: DiagnosticText::new("Toolchain operation cancelled before launch"),
        });
    }
    run_owned_command_until_stop(
        command,
        wait_for_probe_stop(cancelled, executor, timeout),
        10 * 1024 * 1024,
    )
    .await
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
async fn run_owned_command_until_stop(
    command: std::process::Command,
    stop: impl std::future::Future<Output = ProbeEnd> + Send,
    maximum_bytes: usize,
) -> Result<ProbeResult> {
    use futures::{FutureExt as _, future::Either};
    use std::{
        process::Stdio,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    let mut child = ProbeChild {
        process: util::process::Child::spawn(
            command,
            Stdio::null(),
            Stdio::piped(),
            Stdio::piped(),
        )?,
        terminated: false,
    };
    let stdout = child
        .process
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Dependency probe stdout is unavailable"))?;
    let stderr = child
        .process
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("Dependency probe stderr is unavailable"))?;
    let count = Arc::new(AtomicUsize::new(0));
    let completion = async {
        futures::try_join!(
            child.process.status(),
            async {
                read_probe_output(stdout, count.clone(), maximum_bytes)
                    .await
                    .map_err(std::io::Error::other)
            },
            async {
                read_probe_output(stderr, count.clone(), maximum_bytes)
                    .await
                    .map_err(std::io::Error::other)
            }
        )
    }
    .boxed();
    let stop = stop.boxed();
    let result = match futures::future::select(completion, stop).await {
        Either::Left((result, _)) => match result {
            Ok((status, stdout, stderr)) => {
                let bytes = if stdout.is_empty() { stderr } else { stdout };
                let text = String::from_utf8_lossy(&bytes);
                ProbeResult {
                    end: if status.success() {
                        ProbeEnd::Completed
                    } else {
                        ProbeEnd::Failed
                    },
                    detail: DiagnosticText::new(
                        text.lines().next().unwrap_or("No version output observed"),
                    ),
                }
            }
            Err(_) if count.load(Ordering::Relaxed) > maximum_bytes => ProbeResult {
                end: ProbeEnd::OutputLimit,
                detail: DiagnosticText::new(
                    "Dependency probe exceeded its output limit; result is incomplete",
                ),
            },
            Err(error) => return Err(error.into()),
        },
        Either::Right((end, completion)) => {
            drop(completion);
            ProbeResult {
                end,
                detail: DiagnosticText::new(
                    "Dependency probe stopped; availability remains unverified",
                ),
            }
        }
    };
    child.process.kill()?;
    child.terminated = true;
    child.process.status().await?;
    Ok(result)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyProbe {
    RustCompiler,
    RustFormatter,
    ClippyDriver,
    RustAnalyzer,
    Cargo,
    Java,
    JavaCompiler,
}

impl DependencyProbe {
    pub fn version_arguments(self) -> &'static [&'static str] {
        match self {
            Self::RustCompiler
            | Self::RustFormatter
            | Self::ClippyDriver
            | Self::RustAnalyzer
            | Self::Cargo => &["--version"],
            Self::Java | Self::JavaCompiler => &["-version"],
        }
    }

    pub fn validate_executable(
        self,
        executable: &std::path::Path,
        project_roots: &[std::path::PathBuf],
        header: &[u8],
    ) -> Result<()> {
        ensure!(
            executable.is_absolute(),
            "Dependency executable must be an absolute reviewed path"
        );
        ensure!(
            !project_roots
                .iter()
                .any(|root| executable.starts_with(root)),
            "Project executables cannot be dependency probes"
        );
        let name = executable
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let expected = match self {
            Self::RustCompiler => "rustc",
            Self::RustFormatter => "rustfmt",
            Self::ClippyDriver => "clippy-driver",
            Self::RustAnalyzer => "rust-analyzer",
            Self::Cargo => "cargo",
            Self::Java => "java",
            Self::JavaCompiler => "javac",
        };
        ensure!(
            name == expected || name == format!("{expected}.exe"),
            "Dependency executable name does not match its reviewed tool"
        );
        ensure!(
            !executable
                .components()
                .any(|part| part.as_os_str() == ".cargo" || part.as_os_str() == "shims"),
            "Provisioning shims cannot be dependency probes"
        );
        let binary = header.starts_with(b"\x7fELF")
            || header.starts_with(b"MZ")
            || [
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf",
                b"\xca\xfe\xba\xbe",
                b"\xbe\xba\xfe\xca",
            ]
            .iter()
            .any(|magic| header.starts_with(*magic));
        ensure!(
            binary,
            "Scripts and unknown executable formats cannot be dependency probes"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoctorRepairBinding {
    project: gpui::EntityId,
    directory: project::ProjectPath,
    host: project::DoctorExecutionHost,
    connection: Option<gpui::EntityId>,
    generation: u64,
    input_revision: u64,
}

impl DoctorRepairBinding {
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    pub(crate) fn directory(&self) -> &project::ProjectPath {
        &self.directory
    }

    pub(crate) fn revalidate_project(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            project_repair_binding(project, self.directory.clone(), self.generation, cx)? == *self,
            "Reviewed project authority changed"
        );
        Ok(())
    }

    pub fn capture(
        project: &gpui::Entity<project::Project>,
        directory: project::ProjectPath,
        generation: u64,
        input_revision: u64,
        cx: &gpui::App,
    ) -> Result<Self> {
        let owner = project.read(cx);
        let observation = owner.doctor_observation(directory.worktree_id, cx);
        ensure!(
            observation.supports_reviewed_probe(),
            "Repair scope is unavailable or unauthorized"
        );
        ensure!(
            directory.path.is_empty(),
            "Select a worktree root for this repair"
        );
        Ok(Self {
            project: project.entity_id(),
            directory,
            host: observation.host,
            connection: owner.remote_client().map(|client| client.entity_id()),
            generation,
            input_revision,
        })
    }
    fn admission_key(&self, check: &DiagnosticText, effects: &DiagnosticText) -> String {
        use sha2::{Digest as _, Sha256};
        let mut digest = Sha256::new();
        // Refreshing observations must not bypass an active equivalent action.
        for value in [
            format!("{:?}", self.project),
            format!("{:?}", self.directory),
            format!("{:?}", self.host),
            format!("{:?}", self.connection),
            check.as_str().to_owned(),
            effects.as_str().to_owned(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn same_scope(&self, current: &Self) -> bool {
        self.project == current.project
            && self.directory == current.directory
            && self.host == current.host
            && self.connection == current.connection
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorInvestigationScope {
    Product,
    Development,
    Application,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorInvestigationProblem {
    ProductReadiness,
    Build,
    Test,
    Dependencies,
    Configuration,
    Pipeline,
    Crash,
    IncorrectBehavior,
    Cpu,
    Memory,
    SlowOperation,
    Concurrency,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct DoctorInvestigationInputIdentity(String);

impl DoctorInvestigationInputIdentity {
    pub fn from_fields(fields: &[&str]) -> Result<Self> {
        use sha2::{Digest as _, Sha256};
        ensure!(
            !fields.is_empty(),
            "Investigation requires actual selected owner inputs"
        );
        let bytes = fields.iter().try_fold(0_usize, |total, value| {
            total
                .checked_add(value.len())
                .ok_or_else(|| anyhow::anyhow!("Investigation inputs exceed their bound"))
        })?;
        ensure!(
            bytes > 0 && bytes <= MAX_REPORT_BYTES,
            "Investigation input fingerprint is empty or exceeds its bound"
        );
        let mut digest = Sha256::new();
        for value in fields {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        Ok(Self(format!("{:x}", digest.finalize())))
    }
}

impl<'de> Deserialize<'de> for DoctorInvestigationInputIdentity {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let digest = String::deserialize(deserializer)?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom(
                "Investigation input identity is invalid",
            ));
        }
        Ok(Self(digest))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorInvestigationSelection {
    pub input_identity: DoctorInvestigationInputIdentity,
    pub scope: DoctorInvestigationScope,
    pub problem: DoctorInvestigationProblem,
    pub symptom: DiagnosticText,
    pub validation_target: DiagnosticText,
    pub workload: DiagnosticText,
    pub configuration: DiagnosticText,
    pub source_revision: DiagnosticText,
    pub cache_conditions: DiagnosticText,
}

impl DoctorInvestigationSelection {
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("symptom", &self.symptom),
            ("validation target", &self.validation_target),
            ("workload", &self.workload),
            ("configuration", &self.configuration),
            ("source revision", &self.source_revision),
            ("cache conditions", &self.cache_conditions),
        ] {
            ensure!(
                !value.as_str().is_empty() && value.as_str().len() <= 4096,
                "Investigation {label} is missing or exceeds its bound"
            );
        }
        let supported = match self.scope {
            DoctorInvestigationScope::Product => {
                self.problem == DoctorInvestigationProblem::ProductReadiness
            }
            DoctorInvestigationScope::Development => matches!(
                self.problem,
                DoctorInvestigationProblem::Build
                    | DoctorInvestigationProblem::Test
                    | DoctorInvestigationProblem::Dependencies
                    | DoctorInvestigationProblem::Configuration
                    | DoctorInvestigationProblem::Pipeline
            ),
            DoctorInvestigationScope::Application => matches!(
                self.problem,
                DoctorInvestigationProblem::Test
                    | DoctorInvestigationProblem::Crash
                    | DoctorInvestigationProblem::IncorrectBehavior
                    | DoctorInvestigationProblem::Cpu
                    | DoctorInvestigationProblem::Memory
                    | DoctorInvestigationProblem::SlowOperation
                    | DoctorInvestigationProblem::Concurrency
            ),
        };
        ensure!(
            supported,
            "Choose a problem belonging to the selected investigation scope"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorEvidenceClass {
    Observed,
    UserReported,
    Hypothesis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorEvidenceOwner {
    Project,
    Task,
    Test,
    Debugger,
    Profile,
    NativeAgent,
    User,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorEvidenceObservation {
    Unverified,
    CommandExit {
        code: Option<i32>,
    },
    FunctionalFailure {
        assertion: DiagnosticText,
    },
    FunctionalSuccess {
        assertion: DiagnosticText,
    },
    Exception {
        kind: DiagnosticText,
    },
    AggregateCounter {
        metric: DiagnosticText,
        value: DiagnosticText,
        unit: DiagnosticText,
    },
    Cancelled,
    Incomplete,
    Unsupported,
    InsufficientEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorInvestigationEvidence {
    pub id: uuid::Uuid,
    pub owner: DoctorEvidenceOwner,
    pub owner_reference: DiagnosticText,
    pub classification: DoctorEvidenceClass,
    pub observation: DoctorEvidenceObservation,
    pub detail: DiagnosticText,
    pub observed_at: u64,
    pub historical: bool,
    selection: DoctorInvestigationSelection,
    generation: u64,
}

#[derive(Clone)]
pub struct DoctorEvidenceInput {
    pub owner: DoctorEvidenceOwner,
    pub owner_reference: DiagnosticText,
    pub classification: DoctorEvidenceClass,
    pub observation: DoctorEvidenceObservation,
    pub detail: DiagnosticText,
    pub observed_at: u64,
    pub historical: bool,
}

impl DoctorInvestigationEvidence {
    fn validate(&self) -> Result<()> {
        self.selection.validate()?;
        ensure!(
            !self.owner_reference.as_str().is_empty() && !self.detail.as_str().is_empty(),
            "Evidence requires an owner reference and a redacted summary"
        );
        match self.classification {
            DoctorEvidenceClass::Observed => ensure!(
                !matches!(
                    self.owner,
                    DoctorEvidenceOwner::User | DoctorEvidenceOwner::NativeAgent
                ),
                "User reports and model assertions are not owner observations"
            ),
            DoctorEvidenceClass::UserReported => ensure!(
                self.owner == DoctorEvidenceOwner::User
                    && self.observation == DoctorEvidenceObservation::Unverified,
                "A user symptom cannot establish an observed result"
            ),
            DoctorEvidenceClass::Hypothesis => ensure!(
                self.observation == DoctorEvidenceObservation::Unverified,
                "A hypothesis cannot establish an observed result"
            ),
        }
        match &self.observation {
            DoctorEvidenceObservation::FunctionalFailure { assertion }
            | DoctorEvidenceObservation::FunctionalSuccess { assertion } => {
                ensure!(
                    self.classification == DoctorEvidenceClass::Observed
                        && matches!(
                            self.owner,
                            DoctorEvidenceOwner::Task | DoctorEvidenceOwner::Test
                        )
                        && !assertion.as_str().is_empty(),
                    "Functional evidence requires an observed selected Task/test assertion"
                );
            }
            DoctorEvidenceObservation::Exception { kind } => ensure!(
                self.classification == DoctorEvidenceClass::Observed
                    && self.owner == DoctorEvidenceOwner::Debugger
                    && !kind.as_str().is_empty(),
                "Exception evidence requires the selected debugger observation"
            ),
            DoctorEvidenceObservation::AggregateCounter {
                metric,
                value,
                unit,
            } => {
                let number = value.as_str().parse::<f64>()?;
                ensure!(
                    self.classification == DoctorEvidenceClass::Observed
                        && matches!(
                            self.owner,
                            DoctorEvidenceOwner::Profile | DoctorEvidenceOwner::Task
                        )
                        && !metric.as_str().is_empty()
                        && !unit.as_str().is_empty()
                        && number.is_finite()
                        && number >= 0.0,
                    "Aggregate evidence requires a valid observed counter, unit and supported owner"
                );
            }
            DoctorEvidenceObservation::CommandExit { .. } => ensure!(
                self.classification == DoctorEvidenceClass::Observed
                    && self.owner == DoctorEvidenceOwner::Task,
                "Command exit evidence requires the selected Task owner"
            ),
            _ => {}
        }
        Ok(())
    }

    pub fn reproduced_failure(&self) -> bool {
        self.classification == DoctorEvidenceClass::Observed
            && !self.historical
            && matches!(
                self.observation,
                DoctorEvidenceObservation::FunctionalFailure { .. }
                    | DoctorEvidenceObservation::Exception { .. }
            )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorInvestigationSummary {
    pub selection: DoctorInvestigationSelection,
    pub scope: DoctorScope,
    pub generation: u64,
    pub started_at: u64,
    evidence: Vec<DoctorInvestigationEvidence>,
}

impl DoctorInvestigationSummary {
    pub fn evidence(&self) -> &[DoctorInvestigationEvidence] {
        &self.evidence
    }
    pub fn validate(&self) -> Result<()> {
        self.selection.validate()?;
        ensure!(
            self.evidence.len() <= MAX_CHECKS,
            "Investigation evidence exceeds its row limit"
        );
        let mut identities = std::collections::HashSet::new();
        let mut owners = std::collections::HashSet::new();
        for evidence in &self.evidence {
            evidence.validate()?;
            ensure!(
                evidence.historical || evidence.observed_at >= self.started_at,
                "Saved evidence predates its investigation"
            );
            ensure!(
                evidence.selection == self.selection && evidence.generation == self.generation,
                "Investigation evidence belongs to another target or generation"
            );
            ensure!(
                identities.insert(evidence.id)
                    && owners.insert(format!(
                        "{:?}:{}",
                        evidence.owner,
                        evidence.owner_reference.as_str()
                    )),
                "Duplicate investigation evidence"
            );
        }
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_REPORT_BYTES,
            "Investigation summary exceeds its byte limit"
        );
        Ok(())
    }
}

pub struct DoctorInvestigation {
    binding: DoctorRepairBinding,
    summary: DoctorInvestigationSummary,
}

impl DoctorInvestigation {
    pub fn new(
        selection: DoctorInvestigationSelection,
        scope: DoctorScope,
        binding: DoctorRepairBinding,
        started_at: u64,
    ) -> Result<Self> {
        selection.validate()?;
        Ok(Self {
            summary: DoctorInvestigationSummary {
                selection,
                scope,
                generation: binding.generation,
                started_at,
                evidence: Vec::new(),
            },
            binding,
        })
    }

    pub fn summary(&self) -> &DoctorInvestigationSummary {
        &self.summary
    }

    pub fn record(
        &mut self,
        current: &DoctorRepairBinding,
        selection: &DoctorInvestigationSelection,
        input: DoctorEvidenceInput,
    ) -> Result<uuid::Uuid> {
        ensure!(
            current == &self.binding && selection == &self.summary.selection,
            "Investigation inputs, target, permissions or execution host changed; this result is stale"
        );
        ensure!(
            input.historical || input.observed_at >= self.summary.started_at,
            "Evidence predates the selected investigation"
        );
        let evidence = DoctorInvestigationEvidence {
            id: uuid::Uuid::new_v4(),
            owner: input.owner,
            owner_reference: input.owner_reference,
            classification: input.classification,
            observation: input.observation,
            detail: input.detail,
            observed_at: input.observed_at,
            historical: input.historical,
            selection: selection.clone(),
            generation: self.binding.generation,
        };
        evidence.validate()?;
        let id = evidence.id;
        self.summary.evidence.push(evidence);
        if let Err(error) = self.summary.validate() {
            self.summary.evidence.pop();
            return Err(error);
        }
        Ok(id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorReviewedChange {
    pub before_input: DoctorInvestigationInputIdentity,
    pub after_input: DoctorInvestigationInputIdentity,
    pub description: DiagnosticText,
    pub session_id: Option<DiagnosticText>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorPerformanceCriterion {
    pub metric: DiagnosticText,
    pub unit: DiagnosticText,
    pub protocol: DiagnosticText,
    pub minimum_repetitions: u16,
    pub maximum_after: DiagnosticText,
    pub minimum_improvement_percent: DiagnosticText,
}

impl DoctorPerformanceCriterion {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.metric.as_str().is_empty()
                && !self.unit.as_str().is_empty()
                && !self.protocol.as_str().is_empty(),
            "Select the exact metric, units and capture protocol"
        );
        ensure!(
            (2..=32).contains(&self.minimum_repetitions),
            "Select two to thirty-two repetitions; a single sample cannot verify improvement"
        );
        let maximum = self.maximum_after.as_str().parse::<f64>()?;
        let improvement = self.minimum_improvement_percent.as_str().parse::<f64>()?;
        ensure!(
            maximum.is_finite()
                && maximum >= 0.0
                && improvement.is_finite()
                && (0.0..=100.0).contains(&improvement),
            "Performance acceptance criterion is invalid"
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct DoctorComparisonRequest {
    pub baseline_functional: uuid::Uuid,
    pub verification_functional: uuid::Uuid,
    pub change: DoctorReviewedChange,
    pub performance: Option<DoctorPerformanceCriterion>,
    pub baseline_measurements: Vec<uuid::Uuid>,
    pub verification_measurements: Vec<uuid::Uuid>,
    pub baseline_protocol: DiagnosticText,
    pub verification_protocol: DiagnosticText,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DoctorComparisonOutcome {
    VerifiedSelectedTarget,
    CriterionNotMet,
    Incomplete,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorComparisonRecord {
    pub outcome: DoctorComparisonOutcome,
    pub change: DoctorReviewedChange,
    pub criterion: Option<DoctorPerformanceCriterion>,
    pub baseline_selection: DoctorInvestigationSelection,
    pub verification_selection: DoctorInvestigationSelection,
    pub baseline_evidence: Vec<uuid::Uuid>,
    pub verification_evidence: Vec<uuid::Uuid>,
    pub baseline_median: Option<DiagnosticText>,
    pub verification_median: Option<DiagnosticText>,
    pub observed_at: u64,
    pub detail: DiagnosticText,
}

impl DoctorComparisonRecord {
    fn validate(&self) -> Result<()> {
        self.baseline_selection.validate()?;
        self.verification_selection.validate()?;
        ensure!(
            !self.change.description.as_str().is_empty() && !self.detail.as_str().is_empty(),
            "Comparison requires a reviewed change and remaining uncertainty"
        );
        for identifiers in [&self.baseline_evidence, &self.verification_evidence] {
            ensure!(
                !identifiers.is_empty()
                    && identifiers.len() <= 33
                    && identifiers
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        == identifiers.len(),
                "Comparison evidence is missing, duplicated or unbounded"
            );
        }
        if let Some(criterion) = &self.criterion {
            criterion.validate()?;
        }
        for value in [&self.baseline_median, &self.verification_median]
            .into_iter()
            .flatten()
        {
            let number = value.as_str().parse::<f64>()?;
            ensure!(
                number.is_finite() && number >= 0.0,
                "Comparison median is invalid"
            );
        }
        if self.outcome == DoctorComparisonOutcome::VerifiedSelectedTarget {
            ensure!(
                self.change.before_input == self.baseline_selection.input_identity
                    && self.change.after_input == self.verification_selection.input_identity
                    && self.change.before_input != self.change.after_input,
                "Verified comparison has no attributable changed inputs"
            );
            if self.criterion.is_some() {
                ensure!(
                    self.baseline_median.is_some() && self.verification_median.is_some(),
                    "Verified performance has no comparable measurements"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DoctorComparisonSnapshot {
    binding: DoctorRepairBinding,
    summary: DoctorInvestigationSummary,
    source_snapshot: Option<(usize, Vec<String>)>,
    source_identity: Option<DoctorInvestigationInputIdentity>,
    persisted_source: bool,
}

impl DoctorInvestigation {
    pub fn comparison_snapshot(&self) -> DoctorComparisonSnapshot {
        DoctorComparisonSnapshot {
            binding: self.binding.clone(),
            summary: self.summary.clone(),
            source_snapshot: None,
            source_identity: None,
            persisted_source: false,
        }
    }

    pub fn comparison_snapshot_at(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<DoctorComparisonSnapshot> {
        ensure!(
            project.entity_id() == self.binding.project,
            "Comparison project changed"
        );
        let worktree = project
            .read(cx)
            .worktree_for_id(self.binding.directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Comparison worktree is unavailable"))?;
        let mut snapshot = self.comparison_snapshot();
        snapshot.persisted_source =
            !project
                .read(cx)
                .opened_buffers(cx)
                .into_iter()
                .any(|buffer| {
                    let buffer = buffer.read(cx);
                    buffer.is_dirty()
                        && project::File::from_dyn(buffer.file()).is_some_and(|file| {
                            file.worktree_id(cx) == self.binding.directory.worktree_id
                        })
                });
        snapshot.source_identity = Some(comparison_source_identity(
            project,
            &self.binding.directory,
            cx,
        )?);
        snapshot.source_snapshot = Some((
            worktree.read(cx).scan_id(),
            reproduction_buffer_versions(project, self.binding.directory.worktree_id, cx),
        ));
        Ok(snapshot)
    }
}

pub fn comparison_source_identity(
    project: &gpui::Entity<project::Project>,
    directory: &project::ProjectPath,
    cx: &gpui::App,
) -> Result<DoctorInvestigationInputIdentity> {
    comparison_source_identity_excluding(project, directory, None, cx)
}

pub(crate) fn comparison_source_identity_excluding(
    project: &gpui::Entity<project::Project>,
    directory: &project::ProjectPath,
    excluded: Option<&project::ProjectPath>,
    cx: &gpui::App,
) -> Result<DoctorInvestigationInputIdentity> {
    use language::File as _;
    use sha2::{Digest as _, Sha256};
    let worktree = project
        .read(cx)
        .worktree_for_id(directory.worktree_id, cx)
        .ok_or_else(|| anyhow::anyhow!("Selected source worktree is unavailable"))?;
    let snapshot = worktree.read(cx).snapshot();
    let mut digest = Sha256::new();
    let mut bytes = 0_usize;
    for (index, entry) in snapshot
        .files(false, 0)
        .filter(|entry| {
            !entry.is_ignored
                && excluded.is_none_or(|path| {
                    path.worktree_id != directory.worktree_id || path.path != entry.path
                })
        })
        .enumerate()
    {
        ensure!(
            index < 100_000 && !entry.is_external && !entry.is_fifo && entry.mtime.is_some(),
            "Indexed source metadata is incomplete, unsupported or exceeds its bound"
        );
        let metadata = format!(
            "{:?}:{:?}:{}:{}",
            entry.path, entry.mtime, entry.inode, entry.size
        );
        bytes = bytes
            .checked_add(metadata.len())
            .ok_or_else(|| anyhow::anyhow!("Source metadata exceeds its bound"))?;
        ensure!(
            bytes <= 10 * 1024 * 1024,
            "Source metadata exceeds the reviewed ten-MiB ceiling"
        );
        digest.update((metadata.len() as u64).to_le_bytes());
        digest.update(metadata.as_bytes());
    }
    let mut buffers = project
        .read(cx)
        .opened_buffers(cx)
        .into_iter()
        .filter_map(|buffer| {
            let buffer = buffer.read(cx);
            let file = project::File::from_dyn(buffer.file())?;
            (file.worktree_id(cx) == directory.worktree_id && buffer.is_dirty())
                .then(|| format!("{:?}:{:?}", file.path(), buffer.version()))
        })
        .collect::<Vec<_>>();
    buffers.sort();
    for buffer in buffers {
        bytes = bytes
            .checked_add(buffer.len())
            .ok_or_else(|| anyhow::anyhow!("Source metadata exceeds its bound"))?;
        ensure!(
            bytes <= 10 * 1024 * 1024,
            "Source metadata exceeds its bound"
        );
        digest.update((buffer.len() as u64).to_le_bytes());
        digest.update(buffer.as_bytes());
    }
    Ok(DoctorInvestigationInputIdentity(format!(
        "{:x}",
        digest.finalize()
    )))
}

impl DoctorComparisonSnapshot {
    pub fn summary(&self) -> &DoctorInvestigationSummary {
        &self.summary
    }
    pub fn selection(&self) -> &DoctorInvestigationSelection {
        &self.summary.selection
    }
    pub fn evidence(&self) -> &[DoctorInvestigationEvidence] {
        self.summary.evidence()
    }

    fn revalidate_current(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            project.entity_id() == self.binding.project,
            "Verification project changed"
        );
        self.binding.revalidate_project(project, cx)?;
        let source = project
            .read(cx)
            .worktree_for_id(self.binding.directory.worktree_id, cx)
            .map(|worktree| {
                (
                    worktree.read(cx).scan_id(),
                    reproduction_buffer_versions(project, self.binding.directory.worktree_id, cx),
                )
            });
        ensure!(
            self.source_snapshot.is_some() && source == self.source_snapshot,
            "Verification source inputs are unavailable or changed; rerun the relevant target"
        );
        Ok(())
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    pub(crate) fn verify_dependency_inputs_current(
        &self,
        inputs: &DoctorInvestigationInputIdentity,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<()> {
        self.revalidate_current(project, cx)?;
        ensure!(
            self.persisted_source && self.source_identity.as_ref() == Some(inputs),
            "Selected Task/test and dependency owner did not observe the same saved project inputs"
        );
        Ok(())
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    pub(crate) fn verify_dependency_change_current(
        &self,
        baseline: &Self,
        change: &DoctorReviewedChange,
        evidence_id: uuid::Uuid,
        observed_at: u64,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<(DoctorComparisonOutcome, DiagnosticText)> {
        self.revalidate_current(project, cx)?;
        self.verify_dependency_change(baseline, change, evidence_id, observed_at)
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    fn verify_dependency_change(
        &self,
        baseline: &Self,
        change: &DoctorReviewedChange,
        evidence_id: uuid::Uuid,
        observed_at: u64,
    ) -> Result<(DoctorComparisonOutcome, DiagnosticText)> {
        use DoctorComparisonOutcome as Outcome;
        self.summary.validate()?;
        baseline.summary.validate()?;
        let baseline_binding = &baseline.binding;
        let baseline_selection = baseline.selection();
        let incomplete = |reason| (Outcome::Incomplete, DiagnosticText::new(reason));
        let selection = self.selection();
        if !baseline_binding.same_scope(&self.binding)
            || selection.scope != DoctorInvestigationScope::Development
            || selection.problem != DoctorInvestigationProblem::Dependencies
            || baseline_selection.scope != selection.scope
            || baseline_selection.problem != selection.problem
            || baseline_selection.validation_target != selection.validation_target
            || baseline_selection.workload != selection.workload
            || baseline_selection.configuration != selection.configuration
        {
            return Ok(incomplete(
                "Incomplete: selected dependency verification project, host, target, workload or execution configuration differs",
            ));
        }
        if !baseline.persisted_source
            || baseline.source_identity.is_none()
            || baseline.source_identity == self.source_identity
            || !self.persisted_source
            || self.source_identity.is_none()
            || change.before_input != baseline_selection.input_identity
            || change.after_input != selection.input_identity
            || change.before_input == change.after_input
            || change.description.as_str().is_empty()
            || baseline_selection.source_revision == selection.source_revision
        {
            return Ok(incomplete(
                "Incomplete: no saved, attributable reviewed before/after dependency change; editor labels and unsaved changes cannot establish executed inputs",
            ));
        }
        let Some(evidence) = self
            .evidence()
            .iter()
            .find(|evidence| evidence.id == evidence_id)
        else {
            return Ok(incomplete(
                "Incomplete: selected relevant Task/test verification is unavailable",
            ));
        };
        if evidence.observed_at > observed_at
            || observed_at < self.summary.started_at
            || evidence.historical
            || evidence.classification != DoctorEvidenceClass::Observed
        {
            return Ok(incomplete(
                "Incomplete: relevant tests are historical, unobserved or outside the verification clock",
            ));
        }
        if matches!(evidence.observation, DoctorEvidenceObservation::Cancelled) {
            return Ok((
                Outcome::Cancelled,
                DiagnosticText::new(
                    "Cancelled: relevant dependency verification did not finish; changes remain reviewable",
                ),
            ));
        }
        let DoctorEvidenceObservation::FunctionalSuccess { .. } = &evidence.observation else {
            return Ok(match evidence.observation {
                DoctorEvidenceObservation::FunctionalFailure { .. }
                | DoctorEvidenceObservation::Exception { .. } => (
                    Outcome::CriterionNotMet,
                    DiagnosticText::new(
                        "Still failing: relevant owner-observed tests failed after the dependency change",
                    ),
                ),
                _ => incomplete(
                    "Incomplete: command exit, a launch or a model assertion is not relevant owner-observed test verification",
                ),
            });
        };
        if self.evidence().iter().any(|row| {
            !row.historical
                && row.classification == DoctorEvidenceClass::Observed
                && matches!(
                    &row.observation,
                    DoctorEvidenceObservation::FunctionalFailure { .. }
                        | DoctorEvidenceObservation::Exception { .. }
                )
        }) {
            return Ok((
                Outcome::CriterionNotMet,
                DiagnosticText::new(
                    "Still failing: current relevant failure evidence contradicts the selected passing test",
                ),
            ));
        }
        Ok((
            Outcome::VerifiedSelectedTarget,
            DiagnosticText::new(
                "Observed relevant selected-target tests after the saved reviewed change; advisory and resolved-graph verification is still required",
            ),
        ))
    }

    pub fn combine_repetition(&mut self, other: &Self) -> Result<()> {
        let first = &self.summary.selection;
        let next = &other.summary.selection;
        ensure!(
            self.binding == other.binding
                && self.summary.scope == other.summary.scope
                && first.scope == next.scope
                && first.problem == next.problem
                && first.validation_target == next.validation_target
                && first.workload == next.workload
                && first.configuration == next.configuration
                && first.cache_conditions == next.cache_conditions
                && first.source_revision == next.source_revision
                && self.source_identity.is_some()
                && self.source_identity == other.source_identity,
            "Repeated measurements have different source inputs, execution configuration, host, target or authority"
        );
        let mut combined = self.summary.clone();
        for original in &other.summary.evidence {
            let mut row = original.clone();
            row.selection = combined.selection.clone();
            combined.evidence.push(row);
        }
        combined.validate()?;
        self.summary = combined;
        self.source_snapshot = other.source_snapshot.clone();
        Ok(())
    }

    pub fn compare_current(
        &self,
        verification: &Self,
        request: DoctorComparisonRequest,
        observed_at: u64,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<DoctorComparisonRecord> {
        verification.revalidate_current(project, cx)?;
        let mut result = self.compare(verification, request, observed_at)?;
        if result.outcome != DoctorComparisonOutcome::Cancelled
            && ((self.source_snapshot.is_some() && !self.persisted_source)
                || (verification.source_snapshot.is_some() && !verification.persisted_source))
        {
            result.outcome = DoctorComparisonOutcome::Incomplete;
            result.detail = DiagnosticText::new(
                "Incomplete: selected-worktree inputs include unsaved editor changes. External Tasks/test reports do not establish that this pending source was executed. Review and explicitly save relevant changes through the normal editor before a separately approved fresh rerun; independent edits remain untouched.",
            );
            return Ok(result);
        }
        if self.source_snapshot.is_none()
            || self.source_identity.is_none()
            || self.source_identity == verification.source_identity
        {
            result.outcome = DoctorComparisonOutcome::Incomplete;
            result.detail = DiagnosticText::new(
                "Incomplete: no attributable live before/after source input change was captured; historical reports or changed labels cannot establish a verified repair",
            );
        }
        Ok(result)
    }

    fn compare(
        &self,
        verification: &Self,
        request: DoctorComparisonRequest,
        observed_at: u64,
    ) -> Result<DoctorComparisonRecord> {
        self.summary.validate()?;
        verification.summary.validate()?;
        ensure!(
            !request.change.description.as_str().is_empty(),
            "Review the concrete source/configuration change first"
        );
        if let Some(criterion) = &request.performance {
            criterion.validate()?;
        }
        let mut baseline_evidence = vec![request.baseline_functional];
        baseline_evidence.extend(&request.baseline_measurements);
        let mut verification_evidence = vec![request.verification_functional];
        verification_evidence.extend(&request.verification_measurements);
        let mut record = DoctorComparisonRecord {
            outcome: DoctorComparisonOutcome::Incomplete,
            change: request.change,
            criterion: request.performance,
            baseline_selection: self.summary.selection.clone(),
            verification_selection: verification.summary.selection.clone(),
            baseline_evidence,
            verification_evidence,
            baseline_median: None,
            verification_median: None,
            observed_at,
            detail: DiagnosticText::new(
                "Incomplete: relevant reproduction, reviewed change and independent verification have not all been observed",
            ),
        };
        record.validate()?;
        let baseline = &record.baseline_selection;
        let after = &record.verification_selection;
        let incomplete = |mut record: DoctorComparisonRecord, detail: &str| {
            record.detail = DiagnosticText::new(detail);
            record
        };
        if !self.binding.same_scope(&verification.binding)
            || self.summary.scope != verification.summary.scope
            || baseline.scope != after.scope
            || baseline.problem != after.problem
            || baseline.validation_target != after.validation_target
            || baseline.workload != after.workload
            || (record.criterion.is_some()
                && (baseline.configuration != after.configuration
                    || baseline.cache_conditions != after.cache_conditions))
        {
            return Ok(incomplete(
                record,
                "Incomplete: project, worktree, execution host, target, workload, toolchain/runtime/build configuration or cache conditions differ",
            ));
        }
        if record.change.before_input != baseline.input_identity
            || record.change.after_input != after.input_identity
            || baseline.input_identity == after.input_identity
            || baseline.source_revision == after.source_revision
        {
            return Ok(incomplete(
                record,
                "Incomplete: no matching reviewed before/after source/configuration change identity",
            ));
        }
        if observed_at < verification.summary.started_at
            || verification.summary.started_at < self.summary.started_at
        {
            return Ok(incomplete(
                record,
                "Incomplete: verification clock or operation order is invalid",
            ));
        }
        let before = self
            .summary
            .evidence
            .iter()
            .find(|row| row.id == request.baseline_functional);
        let after = verification
            .summary
            .evidence
            .iter()
            .find(|row| row.id == request.verification_functional);
        let (Some(before), Some(after)) = (before, after) else {
            return Ok(incomplete(
                record,
                "Incomplete: selected baseline or relevant verification evidence is unavailable",
            ));
        };
        if matches!(after.observation, DoctorEvidenceObservation::Cancelled) {
            record.outcome = DoctorComparisonOutcome::Cancelled;
            record.detail = DiagnosticText::new(
                "Cancelled: the relevant verification did not complete; reached changes remain reviewable",
            );
            return Ok(record);
        }
        if before.observed_at > after.observed_at || after.observed_at > observed_at {
            return Ok(incomplete(
                record,
                "Incomplete: functional evidence timestamp is outside the observed baseline/verification order",
            ));
        }
        if before.historical
            || after.historical
            || before.classification != DoctorEvidenceClass::Observed
            || after.classification != DoctorEvidenceClass::Observed
        {
            return Ok(incomplete(
                record,
                "Incomplete: historical results, user reports and hypotheses cannot verify a current repair",
            ));
        }
        let assertion = match &after.observation {
            DoctorEvidenceObservation::FunctionalSuccess { assertion } => assertion,
            DoctorEvidenceObservation::FunctionalFailure { .. }
            | DoctorEvidenceObservation::Exception { .. } => {
                record.outcome = DoctorComparisonOutcome::CriterionNotMet;
                record.detail = DiagnosticText::new(
                    "Still failing: the relevant owner observed a functional failure or exception after the selected change",
                );
                return Ok(record);
            }
            _ => {
                return Ok(incomplete(
                    record,
                    "Incomplete: command exit, debugger launch, profile import or model assertion is not relevant functional verification",
                ));
            }
        };
        let same_failure = matches!(&before.observation, DoctorEvidenceObservation::FunctionalFailure { assertion: original } if original == assertion);
        let performance_problem = matches!(
            baseline.problem,
            DoctorInvestigationProblem::Cpu
                | DoctorInvestigationProblem::Memory
                | DoctorInvestigationProblem::SlowOperation
                | DoctorInvestigationProblem::Pipeline
        );
        if !same_failure && !performance_problem {
            return Ok(incomplete(
                record,
                "Incomplete: no reproduced failure matching the verified functional assertion",
            ));
        }
        if verification.summary.evidence.iter().any(|row| !row.historical && row.classification == DoctorEvidenceClass::Observed && matches!(&row.observation, DoctorEvidenceObservation::FunctionalFailure { assertion: failed } if failed == assertion)) {
            record.outcome = DoctorComparisonOutcome::CriterionNotMet;
            record.detail = DiagnosticText::new("Still failing: retained current failure evidence contradicts the selected passing assertion");
            return Ok(record);
        }
        if baseline.problem == DoctorInvestigationProblem::Concurrency {
            return Ok(incomplete(
                record,
                "Incomplete: a selected passing rerun does not establish race freedom or concurrency correctness; a reviewed relevant concurrency protocol is unavailable",
            ));
        }
        if let Some(criterion) = &record.criterion {
            if matches!(
                baseline.problem,
                DoctorInvestigationProblem::Cpu | DoctorInvestigationProblem::Memory
            ) || criterion.metric.as_str() != "reviewed Task end-to-end wall time"
                || criterion.unit.as_str() != "ms"
                || criterion.protocol.as_str() != DOCTOR_TASK_TIMING_PROTOCOL
            {
                return Ok(incomplete(
                    record,
                    "Incomplete: the current reviewed Task owner measures end-to-end wall time only. CPU/memory counters, artifact sample counts and unproven capture protocols cannot verify runtime performance",
                ));
            }
            if criterion.protocol != request.baseline_protocol
                || criterion.protocol != request.verification_protocol
            {
                return Ok(incomplete(
                    record,
                    "Incomplete: measurement capture protocols differ from the reviewed criterion",
                ));
            }
            let collect = |summary: &DoctorInvestigationSummary,
                           identifiers: &[uuid::Uuid]|
             -> Result<Option<f64>> {
                if identifiers.len() < usize::from(criterion.minimum_repetitions) {
                    return Ok(None);
                }
                let mut values = Vec::new();
                for identifier in identifiers {
                    let Some(row) = summary.evidence.iter().find(|row| row.id == *identifier)
                    else {
                        return Ok(None);
                    };
                    if row.historical
                        || row.classification != DoctorEvidenceClass::Observed
                        || row.owner != DoctorEvidenceOwner::Task
                        || row.detail.as_str() != DOCTOR_TASK_TIMING_PROTOCOL
                        || row.observed_at > observed_at
                    {
                        return Ok(None);
                    }
                    let DoctorEvidenceObservation::AggregateCounter {
                        metric,
                        value,
                        unit,
                    } = &row.observation
                    else {
                        return Ok(None);
                    };
                    if metric != &criterion.metric || unit != &criterion.unit {
                        return Ok(None);
                    }
                    values.push(value.as_str().parse::<f64>()?);
                }
                values.sort_by(f64::total_cmp);
                let middle = values.len() / 2;
                let value = values
                    .get(middle)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("No measurement samples"))?;
                let median = if values.len() % 2 == 0 {
                    value / 2.0
                        + values
                            .get(middle.saturating_sub(1))
                            .copied()
                            .ok_or_else(|| anyhow::anyhow!("Missing paired median"))?
                            / 2.0
                } else {
                    value
                };
                Ok(Some(median))
            };
            let (Some(before), Some(after)) = (
                collect(&self.summary, &request.baseline_measurements)?,
                collect(&verification.summary, &request.verification_measurements)?,
            ) else {
                return Ok(incomplete(
                    record,
                    "Incomplete: valid current samples, exact metric/units or reviewed repetition count are missing",
                ));
            };
            record.baseline_median = Some(DiagnosticText::new(before.to_string()));
            record.verification_median = Some(DiagnosticText::new(after.to_string()));
            let improvement = if before > 0.0 {
                (before - after) / before * 100.0
            } else {
                0.0
            };
            if after > criterion.maximum_after.as_str().parse::<f64>()?
                || improvement
                    < criterion
                        .minimum_improvement_percent
                        .as_str()
                        .parse::<f64>()?
            {
                record.outcome = DoctorComparisonOutcome::CriterionNotMet;
                record.detail = DiagnosticText::new(
                    "Criterion not met: observed comparable medians do not meet the selected absolute limit and improvement threshold; functional verification alone does not establish performance improvement",
                );
                return Ok(record);
            }
        } else if performance_problem {
            return Ok(incomplete(
                record,
                "Incomplete: a performance claim requires an explicitly selected criterion and comparable repeated measurements",
            ));
        }
        record.outcome = DoctorComparisonOutcome::VerifiedSelectedTarget;
        record.detail = DiagnosticText::new(
            "Observed selected target meets the reviewed criterion after the recorded change. This is bounded workload evidence, not proof of causation, overall application correctness, security, race freedom or statistical significance. Wall time is not CPU time and sample counts are not memory consumption.",
        );
        record.validate()?;
        Ok(record)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DoctorReproductionEffects {
    Development {
        file_writes: String,
        subprocesses: String,
    },
    Unknown,
    ExternalServices,
    ProductionOrDestructive,
    SensitiveCapture,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
#[derive(Clone)]
enum DoctorTestPlan {
    #[cfg(feature = "rust-tools")]
    Rust(project::rust_test_provider::RustTestActionPlan),
    #[cfg(feature = "jvm-tools")]
    Java(project::jvm_test_provider::JvmSelectedTestPlan),
    #[cfg(feature = "jvm-tools")]
    RemoteJava(project::jvm_test_provider::JvmRemoteTestActionPlan),
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DoctorTestPlan {
    fn provider_and_nodes(
        &self,
    ) -> Result<(
        project::structured_execution::StructuredProviderId,
        project::structured_execution::DiscoveryGeneration,
        Vec<project::structured_execution::StructuredNodeId>,
    )> {
        match self {
            #[cfg(feature = "rust-tools")]
            Self::Rust(project::rust_test_provider::RustTestActionPlan::Task {
                discovery_generation,
                scope_node_ids,
                ..
            }) => Ok((
                project::rust_test_provider::RustTestProviderStore::provider_id(),
                *discovery_generation,
                scope_node_ids.clone(),
            )),
            #[cfg(feature = "rust-tools")]
            Self::Rust(_) => {
                anyhow::bail!("Reproduction requires a test Task, not a debugger launch")
            }
            #[cfg(feature = "jvm-tools")]
            Self::RemoteJava(plan) => Ok((
                project::structured_execution::StructuredProviderId(
                    project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
                ),
                plan.discovery_generation,
                vec![plan.node_id.clone()],
            )),
            #[cfg(feature = "jvm-tools")]
            Self::Java(plan) => Ok((
                project::structured_execution::StructuredProviderId(
                    project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
                ),
                plan.discovery_generation,
                vec![plan.node_id.clone()],
            )),
        }
    }

    fn revalidate(
        &self,
        project: &gpui::Entity<project::Project>,
        worktree: settings::WorktreeId,
        cx: &gpui::App,
    ) -> Result<()> {
        use project::structured_execution::StructuredProviderStatus;
        let (provider, generation, nodes) = self.provider_and_nodes()?;
        let owner = project.read(cx);
        let store = owner.structured_execution_store().read(cx);
        let snapshot = store
            .state()
            .provider(&provider)
            .ok_or_else(|| anyhow::anyhow!("Selected test provider is unavailable"))?;
        ensure!(
            snapshot.discovery_generation == generation
                && (snapshot.status == StructuredProviderStatus::Current || {
                    #[cfg(feature = "jvm-tools")]
                    {
                        matches!(self, Self::Java(_) | Self::RemoteJava(_))
                            && snapshot.status == StructuredProviderStatus::Stale
                    }
                    #[cfg(not(feature = "jvm-tools"))]
                    {
                        false
                    }
                })
                && !snapshot.partial
                && !nodes.is_empty()
                && nodes
                    .iter()
                    .all(|selected| structured_node_in_worktree(snapshot, selected, worktree)),
            "Selected test discovery is stale, partial or outside this worktree"
        );
        ensure!(
            !snapshot
                .current_run
                .as_ref()
                .is_some_and(|run| !run.phase.is_terminal()),
            "Another test run is active; Doctor will not replace or cancel unrelated test work"
        );
        match self {
            #[cfg(feature = "rust-tools")]
            Self::Rust(_) => ensure!(
                (owner.is_remote()
                    || owner.rust_test_provider_store().read(cx).generation() == generation)
                    && !owner.rust_test_provider_store().read(cx).has_active_runs(),
                "Rust test inputs or active run changed"
            ),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteJava(plan) => owner
                .jvm_test_provider_store()
                .ok_or_else(|| anyhow::anyhow!("Remote JVM owner unavailable"))?
                .read(cx)
                .validate_bounded_remote_plan(plan, 5 * 1024 * 1024, cx)?,
            #[cfg(feature = "jvm-tools")]
            Self::Java(plan) => {
                let provider = owner
                    .jvm_test_provider_store()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Java test actions are unsupported on this host")
                    })?
                    .read(cx);
                ensure!(
                    !provider.has_active_runs()
                        && !provider.is_remote()
                        && provider.plan_selected_case(
                            &plan.node_id,
                            plan.discovery_generation,
                            cx
                        )? == *plan,
                    "Java test inputs or active run changed"
                );
            }
        }
        Ok(())
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
fn structured_node_in_worktree(
    snapshot: &project::structured_execution::StructuredProviderSnapshot,
    selected: &project::structured_execution::StructuredNodeId,
    worktree: settings::WorktreeId,
) -> bool {
    let mut node = snapshot.nodes.iter().find(|node| &node.id == selected);
    for _ in 0..32 {
        let Some(current) = node else {
            return false;
        };
        if let Some(path) = &current.path {
            return path.worktree_id == worktree;
        }
        node = current
            .parent_id
            .as_ref()
            .and_then(|parent| snapshot.nodes.iter().find(|node| &node.id == parent));
    }
    false
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub fn prepare_test_reproduction(
    project: gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    provider_id: project::structured_execution::StructuredProviderId,
    node_id: project::structured_execution::StructuredNodeId,
    discovery_generation: project::structured_execution::DiscoveryGeneration,
    selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorReproductionAction>>> {
    cx.spawn(async move |cx| {
        let binding =
            cx.update(|cx| project_repair_binding(&project, directory.clone(), generation, cx))?;
        let plan: DoctorTestPlan;
        #[cfg(feature = "rust-tools")]
        if provider_id == project::rust_test_provider::RustTestProviderStore::provider_id() {
            let preparation = cx.update(|cx| {
                project
                    .read(cx)
                    .rust_test_provider_store()
                    .clone()
                    .update(cx, |store, cx| {
                        store.plan_action(
                            &node_id,
                            discovery_generation,
                            project::rust_test_provider::RustTestAction::Run,
                            cx,
                        )
                    })
            });
            plan = DoctorTestPlan::Rust(preparation.await?);
        } else {
            #[cfg(feature = "jvm-tools")]
            {
                plan = prepare_java_test_plan(
                    &project,
                    &provider_id,
                    &node_id,
                    discovery_generation,
                    cx,
                )
                .await?;
            }
            #[cfg(not(feature = "jvm-tools"))]
            {
                anyhow::bail!("Selected test provider is unsupported with these product features");
            }
        }
        #[cfg(not(feature = "rust-tools"))]
        {
            plan =
                prepare_java_test_plan(&project, &provider_id, &node_id, discovery_generation, cx)
                    .await?;
        }
        cx.update(|cx| {
            ensure!(
                project_repair_binding(&project, directory.clone(), generation, cx)? == binding,
                "Test review authority changed during planning"
            );
            plan.revalidate(&project, directory.worktree_id, cx)?;
            let (source, template) = match &plan {
                #[cfg(feature = "rust-tools")]
                DoctorTestPlan::Rust(project::rust_test_provider::RustTestActionPlan::Task {
                    template,
                    worktree_id,
                    ..
                }) => {
                    ensure!(
                        *worktree_id == directory.worktree_id,
                        "Selected test belongs to another worktree"
                    );
                    (
                        project::TaskSourceKind::Language {
                            name: "Rust Tests".into(),
                        },
                        template.clone(),
                    )
                }
                #[cfg(feature = "rust-tools")]
                DoctorTestPlan::Rust(_) => anyhow::bail!("Unsupported test action"),
                #[cfg(feature = "jvm-tools")]
                DoctorTestPlan::RemoteJava(_) => {
                    anyhow::bail!("Remote Java requires its reviewed inspection receipt")
                }
                #[cfg(feature = "jvm-tools")]
                DoctorTestPlan::Java(plan) => {
                    ensure!(
                        plan.worktree_id == directory.worktree_id,
                        "Selected Java test belongs to another worktree"
                    );
                    (
                        project::TaskSourceKind::groovy_jvm_test(plan.worktree_id),
                        plan.template.clone(),
                    )
                }
            };
            let root = project
                .read(cx)
                .absolute_path(&directory, cx)
                .ok_or_else(|| anyhow::anyhow!("Selected test root unavailable"))?;
            let mut context = task::TaskContext {
                cwd: Some(root.clone()),
                ..Default::default()
            };
            context.task_variables.insert(
                task::VariableName::WorktreeRoot,
                root.to_string_lossy().into_owned(),
            );
            let task = template
                .resolve_task(&source.to_id_base(), &context)
                .ok_or_else(|| anyhow::anyhow!("Existing selected test Task could not resolve"))?;
            let mut review = prepare_reproduction(
                &project,
                directory,
                source,
                task,
                selection,
                effects,
                duration_seconds,
                generation,
                observed_at,
                cx,
            )?;
            review.action.selection.input_identity =
                DoctorInvestigationInputIdentity::from_fields(&[
                    &review.action.selection.input_identity.0,
                    &provider_id.0,
                    &node_id.0,
                    &discovery_generation.0.to_string(),
                ])?;
            review.before.id = DiagnosticText::new(format!(
                "investigation.task.{}",
                review.action.selection.input_identity.0
            ));
            review.action.test_plan = Some(plan);
            Ok(review)
        })
    })
}

#[cfg(feature = "jvm-tools")]
async fn prepare_java_test_plan(
    project: &gpui::Entity<project::Project>,
    provider_id: &project::structured_execution::StructuredProviderId,
    node_id: &project::structured_execution::StructuredNodeId,
    generation: project::structured_execution::DiscoveryGeneration,
    cx: &mut gpui::AsyncApp,
) -> Result<DoctorTestPlan> {
    ensure!(
        provider_id.0 == project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID,
        "Selected test provider is unsupported"
    );
    let provider = cx
        .update(|cx| project.read(cx).jvm_test_provider_store().cloned())
        .ok_or_else(|| anyhow::anyhow!("Java test owner is unavailable"))?;
    if cx.update(|cx| provider.read(cx).is_remote()) {
        anyhow::bail!(
            "Review bounded remote JVM metadata inspection first, then review its exact selected test execution; plain selected-test preparation does not grant report-read consent"
        );
    } else {
        cx.update(|cx| {
            provider
                .read(cx)
                .plan_selected_case(node_id, generation, cx)
        })
        .map(DoctorTestPlan::Java)
    }
}

#[cfg(feature = "jvm-tools")]
#[derive(Clone)]
pub struct DoctorRemoteTestInspectionAction {
    node: project::structured_execution::StructuredNodeId,
    discovery: project::structured_execution::DiscoveryGeneration,
    scan_id: usize,
    buffer_versions: Vec<String>,
}

#[cfg(feature = "jvm-tools")]
pub struct DoctorInspectedRemoteTest {
    plan: project::jvm_test_provider::JvmRemoteTestActionPlan,
    binding: DoctorRepairBinding,
    inputs: DoctorRemoteTestInspectionAction,
}

#[cfg(feature = "jvm-tools")]
pub fn prepare_remote_test_inspection(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    node: project::structured_execution::StructuredNodeId,
    discovery: project::structured_execution::DiscoveryGeneration,
    generation: u64,
    now: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorRemoteTestInspectionAction>> {
    let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
    let owner = project.read(cx);
    ensure!(
        !owner.is_shared(),
        "Shared development host is unavailable for Doctor reproduction"
    );
    let provider = owner
        .jvm_test_provider_store()
        .ok_or_else(|| anyhow::anyhow!("JVM test owner unavailable"))?
        .read(cx);
    ensure!(
        provider.is_remote() && !provider.has_active_runs(),
        "Select an idle owned remote JVM test host"
    );
    let snapshot = owner
        .structured_execution_store()
        .read(cx)
        .state()
        .provider(&project::structured_execution::StructuredProviderId(
            project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
        ))
        .ok_or_else(|| anyhow::anyhow!("Remote Java discovery unavailable"))?;
    ensure!(
        snapshot.discovery_generation == discovery
            && matches!(
                snapshot.status,
                project::structured_execution::StructuredProviderStatus::Current
                    | project::structured_execution::StructuredProviderStatus::Stale
            )
            && !snapshot.partial
            && structured_node_in_worktree(snapshot, &node, directory.worktree_id)
            && !snapshot
                .current_run
                .as_ref()
                .is_some_and(|run| !run.phase.is_terminal()),
        "Remote selected Java test scope is stale, partial or active"
    );
    let tree = owner
        .worktree_for_id(directory.worktree_id, cx)
        .ok_or_else(|| anyhow::anyhow!("Selected worktree unavailable"))?;
    let tree = tree.read(cx);
    let inputs = DoctorRemoteTestInspectionAction {
        node,
        discovery,
        scan_id: tree.scan_id(),
        buffer_versions: reproduction_buffer_versions(project, directory.worktree_id, cx),
    };
    let identity =
        DoctorInvestigationInputIdentity::from_fields(&[&inputs.node.0, &discovery.0.to_string()])?;
    DoctorRepairReview::new(
        inputs,
        binding,
        DoctorCheck::new(
            &format!("investigation.java.inspect.{}", identity.0),
            "Inspect selected remote Java test metadata",
            CheckOutcome::Untested,
            "A reviewed host report baseline is needed before exact execution review",
            now,
        ),
        DiagnosticText::new(
            "Read the selected remote JVM test owner’s exact Task metadata and bounded existing JUnit baseline (at most five MiB, 10-second deadline) on this trusted owned development host. The compatible host must acknowledge this byte limit; missing acknowledgement blocks execution. No wrapper, dependency download, Task, application, debugger or agent will run. Raw reports stay with the existing host owner, with no Doctor history/model upload. Cancellation or changed inputs discard the result; reached readonly collection may have completed. Exact reproduction requires a separate review and approval.",
        ),
    )
}

#[cfg(feature = "jvm-tools")]
impl DoctorRemoteTestInspectionAction {
    fn revalidate(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        cx: &gpui::App,
    ) -> Result<()> {
        let owner = project.read(cx);
        ensure!(
            !owner.is_shared()
                && owner
                    .worktree_for_id(directory.worktree_id, cx)
                    .is_some_and(|tree| tree.read(cx).scan_id() == self.scan_id)
                && reproduction_buffer_versions(project, directory.worktree_id, cx)
                    == self.buffer_versions,
            "Remote Java inspection source or development authority changed"
        );
        let store = owner
            .jvm_test_provider_store()
            .ok_or_else(|| anyhow::anyhow!("JVM owner unavailable"))?
            .read(cx);
        ensure!(
            store.is_remote() && !store.has_active_runs(),
            "Remote JVM test owner became unavailable or active"
        );
        let snapshot = owner
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&project::structured_execution::StructuredProviderId(
                project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
            ))
            .ok_or_else(|| anyhow::anyhow!("Remote Java discovery unavailable"))?;
        ensure!(
            snapshot.discovery_generation == self.discovery
                && matches!(
                    snapshot.status,
                    project::structured_execution::StructuredProviderStatus::Current
                        | project::structured_execution::StructuredProviderStatus::Stale
                )
                && !snapshot.partial
                && structured_node_in_worktree(snapshot, &self.node, directory.worktree_id)
                && !snapshot
                    .current_run
                    .as_ref()
                    .is_some_and(|run| !run.phase.is_terminal()),
            "Remote Java test scope changed"
        );
        Ok(())
    }
}

#[cfg(feature = "jvm-tools")]
pub fn inspect_remote_test(
    mut review: DoctorRepairReview<DoctorRemoteTestInspectionAction>,
    project: gpui::Entity<project::Project>,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<
    Result<(
        DoctorRepairReview<DoctorRemoteTestInspectionAction>,
        Option<DoctorInspectedRemoteTest>,
    )>,
> {
    let executor = cx.background_executor().clone();
    let (completed, completion) = futures::channel::oneshot::channel();
    cx.spawn(async move |cx| {
        let result = async {
            let binding = cx.update(|cx| {
                review.action.revalidate(&project, &review.binding.directory, cx)?;
                project_repair_binding(&project, review.binding.directory.clone(), review.binding.generation, cx)
            })?;
            review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
            let operation = async {
                ensure!(!completed.is_canceled(), "Remote inspection caller closed before collection");
                let plan = cx.update(|cx| {
                    review.action.revalidate(&project, &review.binding.directory, cx)?;
                    review.revalidate(&project_repair_binding(&project, review.binding.directory.clone(), review.binding.generation, cx)?)?;
                    let store = project.read(cx).jvm_test_provider_store().cloned().ok_or_else(|| anyhow::anyhow!("JVM owner unavailable"))?;
                    Ok::<_, anyhow::Error>(store.update(cx, |store, cx| store.plan_remote_action_with_limit(&review.action.node, review.action.discovery, None, Some(5 * 1024 * 1024), cx)))
                })?;
                use futures::FutureExt as _;
                let plan = plan.fuse();
                futures::pin_mut!(plan);
                let deadline = executor.now() + std::time::Duration::from_secs(10);
                let plan = loop {
                    ensure!(!completed.is_canceled() && !cancelled.is_closed() && cancelled.is_empty(), "Remote Java inspection cancelled; reached readonly collection may have completed");
                    cx.update(|cx| {
                        review.action.revalidate(&project, &review.binding.directory, cx)?;
                        review.revalidate(&project_repair_binding(&project, review.binding.directory.clone(), review.binding.generation, cx)?)
                    })?;
                    ensure!(executor.now() < deadline, "Bounded remote Java inspection timed out; no execution");
                    let timer = executor.timer(std::time::Duration::from_millis(100).min(deadline.saturating_duration_since(executor.now()))).fuse();
                    let cancellation = cancelled.recv().fuse();
                    futures::pin_mut!(timer, cancellation);
                    futures::select_biased! {
                        result = cancellation => { match result {Ok(()) | Err(_) => anyhow::bail!("Remote Java inspection cancelled; no execution")} },
                        result = plan => break result?,
                        () = timer => {},
                    }
                };
                let current = cx.update(|cx| {
                    ensure!(!completed.is_canceled() && !cancelled.is_closed() && cancelled.is_empty(), "Remote inspection cancelled after collection");
                    review.action.revalidate(&project, &review.binding.directory, cx)?;
                    review.revalidate(&project_repair_binding(&project, review.binding.directory.clone(), review.binding.generation, cx)?)?;
                    project.read(cx).jvm_test_provider_store().ok_or_else(|| anyhow::anyhow!("JVM owner unavailable"))?.read(cx).validate_bounded_remote_plan(&plan, 5 * 1024 * 1024, cx)
                });
                if let Err(error) = current {
                    cx.update(|cx| project.read(cx).jvm_test_provider_store().ok_or_else(|| anyhow::anyhow!("JVM owner unavailable"))?.read(cx).abandon_bounded_remote_plan(&plan, cx)).map_err(|cleanup| anyhow::anyhow!("{error}; plan cleanup incomplete: {cleanup}"))?;
                    return Err(error);
                }
                Ok::<_, anyhow::Error>(DoctorInspectedRemoteTest {plan, binding, inputs: review.action.clone()})
            }.await;
            let inspected = match operation {
                Ok(inspected) => {
                    review.operation_finished(DoctorRepairOperationOutcome::Completed, DiagnosticText::new("Observed an acknowledged bounded remote Java Task plan; no execution or verified repair"))?;
                    review.recheck_unavailable(DiagnosticText::new("Exact Task execution requires its separate reviewed consent"))?;
                    Some(inspected)
                }
                Err(error) => {
                    review.operation_finished(if completed.is_canceled() || cancelled.is_closed() || !cancelled.is_empty() {DoctorRepairOperationOutcome::Cancelled} else {DoctorRepairOperationOutcome::Failed}, DiagnosticText::new(format!("Remote Java inspection incomplete: {error}; no execution or replay")))?;
                    None
                }
            };
            if review.phase() == DoctorRepairPhase::Rechecking { review.recheck_unavailable(DiagnosticText::new("Remote inspection has no acknowledged current result; exact execution remains blocked until a fresh review"))?; }
            review.persist_outcome(product, &history, None, now).await?;
            Ok::<_, anyhow::Error>((review, inspected))
        }.await;
        if completed.send(result).is_err() { log::debug!("Remote Java inspection caller closed after its retained receipt"); }
    }).detach();
    cx.spawn(async move |_| {
        completion
            .await
            .map_err(|_| anyhow::anyhow!("Remote Java inspection observer closed; no replay"))?
    })
}

#[cfg(feature = "jvm-tools")]
pub fn prepare_inspected_remote_test_reproduction(
    inspected: DoctorInspectedRemoteTest,
    project: &gpui::Entity<project::Project>,
    selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    now: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
    let binding = &inspected.binding;
    ensure!(
        project_repair_binding(project, binding.directory.clone(), binding.generation, cx)?
            == *binding,
        "Remote inspection authority changed before exact execution review"
    );
    inspected
        .inputs
        .revalidate(project, &binding.directory, cx)?;
    project
        .read(cx)
        .jvm_test_provider_store()
        .ok_or_else(|| anyhow::anyhow!("JVM owner unavailable"))?
        .read(cx)
        .validate_bounded_remote_plan(&inspected.plan, 5 * 1024 * 1024, cx)?;
    let root = project
        .read(cx)
        .absolute_path(&binding.directory, cx)
        .ok_or_else(|| anyhow::anyhow!("Selected remote root unavailable"))?;
    let source = project::TaskSourceKind::groovy_jvm_test(inspected.plan.worktree_id);
    let mut context = task::TaskContext {
        cwd: Some(root.clone()),
        ..Default::default()
    };
    context.task_variables.insert(
        task::VariableName::WorktreeRoot,
        root.to_string_lossy().into_owned(),
    );
    let task = inspected
        .plan
        .template
        .resolve_task(&source.to_id_base(), &context)
        .ok_or_else(|| anyhow::anyhow!("Exact remote test Task could not resolve"))?;
    let mut review = prepare_reproduction(
        project,
        binding.directory.clone(),
        source,
        task,
        selection,
        effects,
        duration_seconds,
        binding.generation,
        now,
        cx,
    )?;
    review.action.selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
        &review.action.selection.input_identity.0,
        &inspected.plan.node_id.0,
        &inspected.plan.discovery_generation.0.to_string(),
        &inspected.plan.run_id.0,
    ])?;
    review.before.id = DiagnosticText::new(format!(
        "investigation.task.{}",
        review.action.selection.input_identity.0
    ));
    review.action.test_plan = Some(DoctorTestPlan::RemoteJava(inspected.plan));
    Ok(review)
}

#[derive(Clone)]
pub struct DoctorReproductionAction {
    source: project::TaskSourceKind,
    task: task::ResolvedTask,
    selection: DoctorInvestigationSelection,
    scan_id: usize,
    buffer_versions: Vec<String>,
    duration_seconds: u64,
    effects: DoctorReproductionEffects,
    dependency_collection: Option<crate::doctor_dependencies::DoctorDependencyCollection>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    test_plan: Option<DoctorTestPlan>,
}

impl DoctorReproductionAction {
    pub fn selection(&self) -> &DoctorInvestigationSelection {
        &self.selection
    }
    pub fn task(&self) -> &task::ResolvedTask {
        &self.task
    }
    pub fn exact_preview(&self) -> Vec<String> {
        let launch = &self.task.resolved;
        let mut keys = launch.env.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        let mut preview = vec![
            format!("Existing Task source: {:?}; task: {}", self.source, self.task.resolved_label),
            format!("Command: {:?}; arguments: {:?}; shell: {:?}", launch.command, launch.args, launch.shell),
            format!("Working directory: {:?}; explicit environment keys: {:?}", launch.cwd, keys),
            format!("Reviewed development effects: {:?}", self.effects),
            format!("Deadline: {} seconds; Doctor collects bounded exit/lifecycle metadata{}.", self.duration_seconds, if self.dependency_collection.is_some() { " and the explicitly selected bounded dependency report; no terminal output" } else { ", not terminal output or artifacts" }),
            "Project code and wrappers execute with the existing Task host, shell environment and user permissions. This is not a filesystem, network, CPU or memory sandbox. No additional credential or external-service authority is granted. Unknown, production, destructive, stress-test and sensitive-capture effects require a separate decision and are blocked here.".into(),
            "Uses a separate existing Task terminal; no buffers are automatically saved, unrelated runs are untouched, and Doctor does not replay or provide an unattended rerun. Cancellation requests termination through the Task owner; reached writes cannot be undone and termination may be unconfirmed.".into(),
        ];
        if let Some(collection) = &self.dependency_collection {
            preview.push(collection.exact_preview());
        }
        preview
    }

    fn revalidate_inputs(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        cx: &gpui::App,
    ) -> Result<()> {
        let owner = project.read(cx);
        let worktree = owner
            .worktree_for_id(directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Selected worktree is unavailable"))?;
        ensure!(
            worktree.read(cx).scan_id() == self.scan_id
                && reproduction_buffer_versions(project, directory.worktree_id, cx)
                    == self.buffer_versions,
            "Project files or open-buffer inputs changed; review a fresh reproduction"
        );
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        if let Some(plan) = &self.test_plan {
            plan.revalidate(project, directory.worktree_id, cx)?;
        }
        if let Some(collection) = &self.dependency_collection {
            collection.revalidate(project, directory, cx)?;
        }
        validate_reproduction_task(project, directory, &self.source, &self.task, cx)
    }
}

pub(crate) fn reproduction_buffer_versions(
    project: &gpui::Entity<project::Project>,
    worktree: settings::WorktreeId,
    cx: &gpui::App,
) -> Vec<String> {
    let mut versions = project
        .read(cx)
        .opened_buffers(cx)
        .into_iter()
        .filter_map(|buffer| {
            let buffer = buffer.read(cx);
            let file = project::File::from_dyn(buffer.file())?;
            (file.worktree_id(cx) == worktree)
                .then(|| format!("{:?}:{:?}", buffer.remote_id(), buffer.version()))
        })
        .collect::<Vec<_>>();
    versions.sort();
    versions
}

fn validate_reproduction_task(
    project: &gpui::Entity<project::Project>,
    directory: &project::ProjectPath,
    source: &project::TaskSourceKind,
    task: &task::ResolvedTask,
    cx: &gpui::App,
) -> Result<()> {
    ensure!(
        task.resolved_artifact().is_none() && task.resolved_artifact_group().is_none(),
        "Reproduction does not authorize artifact, dump or profile ingestion; use the separately reviewed evidence owner"
    );
    let launch = &task.resolved;
    ensure!(
        launch
            .command
            .as_ref()
            .is_some_and(|command| !command.is_empty())
            && !launch.args.iter().any(|argument| argument.contains('\0')),
        "A concrete valid Task command is required"
    );
    ensure!(
        format!("{task:?}").len() <= 32 * 1024,
        "Task input exceeds the reviewed input bound"
    );
    let cwd = launch
        .cwd
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Select an explicit project working directory"))?;
    ensure!(
        cwd.is_absolute()
            && !cwd
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
        "Task working directory must be an absolute confined project path"
    );
    let owner = project.read(cx);
    let path = owner
        .find_project_path(cwd, cx)
        .ok_or_else(|| anyhow::anyhow!("Task working directory is outside the selected project"))?;
    ensure!(
        path.worktree_id == directory.worktree_id,
        "Task belongs to another worktree"
    );
    if !path.path.is_empty() {
        let entry = owner
            .entry_for_path(&path, cx)
            .ok_or_else(|| anyhow::anyhow!("Task working directory is unavailable"))?;
        ensure!(
            entry.is_dir() && !entry.is_external && !entry.is_private,
            "Task directory is not an authorized contained development directory"
        );
    }
    if let project::TaskSourceKind::Worktree { id, .. } = source {
        ensure!(
            *id == directory.worktree_id,
            "Task source belongs to another worktree"
        );
    }
    Ok(())
}

#[cfg(feature = "rust-tools")]
pub fn prepare_cargo_dependency_collection(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    artifact: task::TaskArtifact,
    selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
    ensure!(
        cfg!(unix),
        "Reviewed Cargo output redirection requires the supported POSIX Task host"
    );
    let owner = project.read(cx);
    let manifest = project::ProjectPath {
        worktree_id: directory.worktree_id,
        path: std::sync::Arc::from(
            directory
                .path
                .join(util::rel_path::RelPath::from_unix_str("Cargo.toml")?),
        ),
    };
    let store = owner.cargo_workspace_store();
    let inputs = store
        .read(cx)
        .prepare_reviewed_metadata_import(manifest.clone(), cx)?;
    let artifact = artifact.with_resolved_path(artifact.path.clone())?;
    let cwd = owner
        .absolute_path(&directory, cx)
        .ok_or_else(|| anyhow::anyhow!("Cargo directory unavailable"))?;
    let manifest = owner
        .absolute_path(&manifest, cx)
        .ok_or_else(|| anyhow::anyhow!("Cargo manifest unavailable"))?;
    let output = cwd.join(&artifact.path);
    let quote = |path: &std::path::Path| -> Result<String> {
        let path = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Cargo Task path is not UTF-8"))?;
        util::shell::ShellKind::Posix
            .try_quote(path)
            .map(|quoted| quoted.into_owned())
            .ok_or_else(|| anyhow::anyhow!("Cargo Task path cannot be quoted"))
    };
    let template = task::TaskTemplate {
        label: "Collect reviewed offline Cargo dependency report".into(),
        command: format!(
            "set -C; cargo metadata --format-version=1 --manifest-path {} --locked --offline > {}",
            quote(&manifest)?,
            quote(&output)?
        ),
        cwd: Some(cwd.to_string_lossy().into_owned()),
        shell: util::shell::Shell::Program("/bin/sh".into()),
        env: collections::HashMap::from_iter([("RUSTUP_AUTO_INSTALL".into(), "0".into())]),
        ..Default::default()
    };
    let selected = template
        .resolve_task(
            "made-doctor-cargo-dependency-report",
            &task::TaskContext::default(),
        )
        .ok_or_else(|| anyhow::anyhow!("Cargo report Task could not resolve"))?;
    let mut review = prepare_dependency_collection(
        project,
        directory,
        project::TaskSourceKind::UserInput,
        selected,
        selection,
        effects,
        artifact,
        crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata,
        duration_seconds,
        generation,
        observed_at,
        cx,
    )?;
    let collection = review
        .action
        .dependency_collection
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Cargo collection descriptor is unavailable"))?;
    collection.bind_cargo_import(inputs);
    Ok(review)
}

#[cfg(feature = "jvm-tools")]
pub fn prepare_jvm_dependency_collection(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    artifact: task::TaskArtifact,
    format: project::jvm_project::JvmReviewedDependencyFormat,
    selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
    ensure!(
        cfg!(unix),
        "Reviewed JVM output collection requires the supported POSIX host"
    );
    let inputs = project
        .read(cx)
        .lsp_store()
        .read(cx)
        .prepare_reviewed_jvm_dependency_import(directory.clone(), format.clone(), cx)?;
    let artifact = artifact.with_resolved_path(artifact.path.clone())?;
    let cwd = project
        .read(cx)
        .absolute_path(&directory, cx)
        .ok_or_else(|| anyhow::anyhow!("JVM directory unavailable"))?;
    let output = cwd.join(&artifact.path);
    let quote = |value: &str| -> Result<String> {
        util::shell::ShellKind::Posix
            .try_quote(value)
            .map(|value| value.into_owned())
            .ok_or_else(|| anyhow::anyhow!("JVM Task argument cannot be quoted"))
    };
    let output = output
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("JVM report path is not UTF-8"))?;
    let adapter = if matches!(
        format,
        project::jvm_project::JvmReviewedDependencyFormat::GradleResolution { .. }
    ) {
        Some(crate::doctor_dependencies::DoctorGradleAdapter::planned(
            &format!("{}:{}", inputs.operation_identity(), artifact.path),
        )?)
    } else {
        None
    };
    let (command, report_format, environment) = match &format {
        project::jvm_project::JvmReviewedDependencyFormat::MavenSelectedTree => {
            let pom = cwd.join("pom.xml");
            let pom = pom
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Maven POM path is not UTF-8"))?;
            (
                format!(
                    "set -C; : > {} && mvn --offline --batch-mode --non-recursive --file {} org.apache.maven.plugins:maven-dependency-plugin:3.7.0:tree -DoutputType=json -Dverbose=false -Dscope= -Dincludes= -Dexcludes= -Dskip=false -DappendOutput=false {}",
                    quote(output)?,
                    quote(pom)?,
                    quote(&format!("-DoutputFile={output}"))?
                ),
                crate::doctor_dependencies::DoctorDependencyReportFormat::MavenSelectedTree,
                collections::HashMap::from_iter([("MAVEN_ARGS".into(), "".into())]),
            )
        }
        project::jvm_project::JvmReviewedDependencyFormat::GradleResolution {
            project_path,
            configuration,
        } => {
            let adapter_path = adapter
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Bundled Gradle input unavailable"))?
                .path()
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Bundled adapter path is not UTF-8"))?;
            let selected_task = if project_path == ":" {
                ":_madeDoctorDependencies".to_string()
            } else {
                format!("{project_path}:_madeDoctorDependencies")
            };
            (
                format!(
                    "gradle --offline --no-daemon --console=plain --init-script {} {} {} {}",
                    quote(adapter_path)?,
                    quote(&selected_task)?,
                    quote(&format!("-PmadeDoctorConfiguration={configuration}"))?,
                    quote(&format!("-PmadeDoctorReport={output}"))?
                ),
                crate::doctor_dependencies::DoctorDependencyReportFormat::GradleResolution {
                    project_path: project_path.clone(),
                    configuration: configuration.clone(),
                },
                collections::HashMap::default(),
            )
        }
    };
    let template = task::TaskTemplate {
        label: "Collect reviewed offline JVM dependency report".into(),
        command,
        cwd: Some(cwd.to_string_lossy().into_owned()),
        shell: util::shell::Shell::Program("/bin/sh".into()),
        env: environment,
        ..Default::default()
    };
    let selected = template
        .resolve_task(
            "made-doctor-jvm-dependency-report",
            &task::TaskContext::default(),
        )
        .ok_or_else(|| anyhow::anyhow!("JVM report Task could not resolve"))?;
    let mut review = prepare_dependency_collection(
        project,
        directory,
        project::TaskSourceKind::UserInput,
        selected,
        selection,
        effects,
        artifact,
        report_format,
        duration_seconds,
        generation,
        observed_at,
        cx,
    )?;
    review
        .action
        .dependency_collection
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("JVM descriptor unavailable"))?
        .bind_jvm_import(inputs);
    if let Some(adapter) = adapter {
        review
            .action
            .dependency_collection
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("JVM descriptor unavailable"))?
            .bind_gradle_adapter(adapter);
    }
    Ok(review)
}

pub fn prepare_dependency_collection(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    source: project::TaskSourceKind,
    task: task::ResolvedTask,
    selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    artifact: task::TaskArtifact,
    format: crate::doctor_dependencies::DoctorDependencyReportFormat,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
    use crate::doctor_dependencies::{DoctorDependencyCollection, DoctorDependencyReportFormat};
    ensure!(
        selection.scope == DoctorInvestigationScope::Development
            && selection.problem == DoctorInvestigationProblem::Dependencies,
        "Select development dependency scope and its exact validation target"
    );
    ensure!(
        match &format {
            DoctorDependencyReportFormat::CargoMetadata => cfg!(feature = "rust-tools"),
            DoctorDependencyReportFormat::MavenSelectedTree
            | DoctorDependencyReportFormat::GradleResolution { .. } => cfg!(feature = "jvm-tools"),
        },
        "This product does not include the selected dependency integration"
    );
    let collection = DoctorDependencyCollection::new(project, &directory, artifact, format, cx)?;
    let preview = collection.exact_preview();
    let mut review = prepare_reproduction(
        project,
        directory,
        source,
        task,
        selection,
        effects,
        duration_seconds,
        generation,
        observed_at,
        cx,
    )?;
    let identity = DoctorInvestigationInputIdentity::from_fields(&[
        &review.action.selection.input_identity.0,
        &preview,
    ])?;
    review.before.id = DiagnosticText::new(format!("investigation.dependencies.{}", identity.0));
    review.before.title = DiagnosticText::new("Reviewed dependency report collection");
    review.action.selection.input_identity = identity;
    review.effects = DiagnosticText::new(format!("{} {}", review.effects.as_str(), preview));
    review.action.dependency_collection = Some(collection);
    Ok(review)
}

pub fn prepare_reproduction(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    source: project::TaskSourceKind,
    mut task: task::ResolvedTask,
    mut selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
    selection.validate()?;
    ensure!(
        selection.scope != DoctorInvestigationScope::Product,
        "Select development or application scope for reproduction"
    );
    ensure!(
        (1..=300).contains(&duration_seconds),
        "Reproduction deadline must be between one second and five minutes"
    );
    let DoctorReproductionEffects::Development {
        file_writes,
        subprocesses,
    } = &effects
    else {
        anyhow::bail!(
            "Unknown, external-service, production/destructive or sensitive collection effects are not approved for development reproduction"
        );
    };
    ensure!(
        !file_writes.trim().is_empty()
            && !subprocesses.trim().is_empty()
            && file_writes.len() + subprocesses.len() <= 8192,
        "Disclose bounded expected development writes and subprocesses"
    );
    let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
    ensure!(
        !project.read(cx).is_shared(),
        "Shared/collaborative projects do not establish exclusive development-host authority"
    );
    validate_reproduction_task(project, &directory, &source, &task, cx)?;
    task.resolved.save = task::SaveStrategy::None;
    task.resolved.use_new_terminal = true;
    task.resolved.allow_concurrent_runs = false;
    task.resolved.show_rerun = false;
    let scan_id = project
        .read(cx)
        .worktree_for_id(directory.worktree_id, cx)
        .ok_or_else(|| anyhow::anyhow!("Selected worktree closed"))?
        .read(cx)
        .scan_id();
    let buffer_versions = reproduction_buffer_versions(project, directory.worktree_id, cx);
    let mut environment = task.resolved.env.iter().collect::<Vec<_>>();
    environment.sort_by(|first, second| first.0.cmp(second.0));
    let execution = format!(
        "{source:?}:{:?}:{:?}:{:?}:{:?}:{environment:?}:{effects:?}:{duration_seconds}",
        task.resolved.command, task.resolved.args, task.resolved.cwd, task.resolved.shell
    );
    let execution_identity = DoctorInvestigationInputIdentity::from_fields(&[&execution])?;
    selection.configuration = DiagnosticText::new(format!(
        "Task configuration {} — {}",
        execution_identity.0,
        selection.configuration.as_str()
    ));
    let inputs =
        format!("{source:?}:{task:?}:{effects:?}:{duration_seconds}:{scan_id}:{buffer_versions:?}");
    selection.input_identity =
        DoctorInvestigationInputIdentity::from_fields(&[&selection.input_identity.0, &inputs])?;
    let before = DoctorCheck::new(
        &format!("investigation.task.{}", selection.input_identity.0),
        "Reviewed development reproduction",
        CheckOutcome::Untested,
        "No reproduction has run. Task exit status alone cannot verify an application repair or establish a functional assertion.",
        observed_at,
    );
    let disclosure = DiagnosticText::new(format!(
        "Task {}; development writes {}; subprocesses {}; {} second deadline; existing shell/environment/host permissions; no automatic save, external authority, raw evidence ingestion, verified-fix claim or replay",
        task.resolved_label, file_writes, subprocesses, duration_seconds
    ));
    let action = DoctorReproductionAction {
        source,
        task,
        selection,
        scan_id,
        buffer_versions,
        duration_seconds,
        effects,
        dependency_collection: None,
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        test_plan: None,
    };
    DoctorRepairReview::new(action, binding, before, disclosure)
}

#[cfg(any(test, feature = "rust-tools", feature = "jvm-tools"))]
fn functional_assertion_identity(
    provider: &str,
    node: &str,
    label: &str,
) -> Result<DiagnosticText> {
    let identity = DoctorInvestigationInputIdentity::from_fields(&[provider, node])?;
    Ok(DiagnosticText::new(format!(
        "target {} — {label}",
        identity.0
    )))
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
fn selected_test_observation(
    plan: &DoctorTestPlan,
    run_id: &project::structured_execution::StructuredRunId,
    project: &gpui::Entity<project::Project>,
    exit_code: Option<i32>,
    cx: &gpui::App,
) -> Result<Option<(DoctorEvidenceObservation, DiagnosticText)>> {
    use project::structured_execution::{
        StructuredNodeState, StructuredProviderStatus, StructuredRunPhase,
    };
    #[cfg(not(feature = "rust-tools"))]
    let _ = exit_code;
    let (provider, generation, nodes) = plan.provider_and_nodes()?;
    let store = project.read(cx).structured_execution_store().read(cx);
    let snapshot = store
        .state()
        .provider(&provider)
        .ok_or_else(|| anyhow::anyhow!("Selected test result owner unavailable"))?;
    ensure!(
        snapshot.discovery_generation == generation
            && !snapshot.partial
            && matches!(
                snapshot.status,
                StructuredProviderStatus::Current | StructuredProviderStatus::Stale
            ),
        "Selected test discovery changed during execution"
    );
    let run = snapshot
        .current_run
        .iter()
        .chain(snapshot.completed_runs.iter())
        .find(|run| &run.id == run_id)
        .ok_or_else(|| anyhow::anyhow!("Selected test run is no longer retained"))?;
    ensure!(
        run.discovery_generation == generation && run.scope_node_ids == nodes && !run.truncated,
        "Selected test run scope or evidence coverage changed"
    );
    if !run.phase.is_terminal() {
        return Ok(None);
    }
    #[cfg(feature = "rust-tools")]
    if matches!(plan, DoctorTestPlan::Rust(_)) {
        return Ok(Some((
            DoctorEvidenceObservation::CommandExit { code: exit_code },
            DiagnosticText::new(
                "The existing Rust test owner attributes this selected run to Task lifecycle status. It does not retain an independent harness assertion distinguishing compilation, panic, no matching cases or application behavior; inspect the test terminal and supply relevant bounded functional evidence. No application fix is verified.",
            ),
        )));
    }
    if run.phase != StructuredRunPhase::Completed {
        return Ok(Some((
            DoctorEvidenceObservation::Incomplete,
            DiagnosticText::new(
                "Selected test run did not complete with an authoritative functional result",
            ),
        )));
    }
    let selected = nodes
        .first()
        .ok_or_else(|| anyhow::anyhow!("Selected test is missing"))?;
    let state = run
        .events
        .iter()
        .rev()
        .find(|event| &event.node_id == selected)
        .map(|event| event.state);
    let label = snapshot
        .nodes
        .iter()
        .find(|node| &node.id == selected)
        .map(|node| node.label.as_str())
        .ok_or_else(|| anyhow::anyhow!("Selected test identity is missing"))?;
    let assertion = functional_assertion_identity(&provider.0, &selected.0, label)?;
    let observation = match state {
        Some(StructuredNodeState::Passed) => {
            DoctorEvidenceObservation::FunctionalSuccess { assertion }
        }
        Some(StructuredNodeState::Failed) => {
            DoctorEvidenceObservation::FunctionalFailure { assertion }
        }
        _ => DoctorEvidenceObservation::InsufficientEvidence,
    };
    Ok(Some((
        observation,
        DiagnosticText::new(
            "Observed the existing JVM test owner's fresh selected JUnit case result, bound to the reviewed node, discovery and run. Raw XML, failure messages and source are not collected by Doctor. This single observation is not an independently verified repair.",
        ),
    )))
}

pub const DOCTOR_TASK_TIMING_PROTOCOL: &str = "Doctor reviewed Task end-to-end monotonic elapsed v1; admission, launch and bounded result collection included";

pub struct DoctorReproductionResult {
    pub review: DoctorRepairReview<DoctorReproductionAction>,
    pub evidence: DoctorEvidenceInput,
    pub elapsed: Option<DoctorEvidenceInput>,
    pub comparison: DoctorComparisonSnapshot,
    pub task: Option<task::StructuredTaskHandle>,
    pub dependency_report: Option<crate::doctor_dependencies::DoctorCollectedDependencyReport>,
}

pub fn run_reproduction(
    mut review: DoctorRepairReview<DoctorReproductionAction>,
    project: gpui::Entity<project::Project>,
    workspace: gpui::WeakEntity<workspace::Workspace>,
    window: gpui::AnyWindowHandle,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorReproductionResult>> {
    use gpui::AppContext as _;
    let executor = cx.background_executor().clone();
    let directory = review.binding.directory.clone();
    let generation = review.binding.generation;
    let (completed, completion) = futures::channel::oneshot::channel();
    cx.spawn(async move |cx| {
      let result = async {
        let started = executor.now();
        let current = cx.update(|cx| {
            review.action.revalidate_inputs(&project, &directory, cx)?;
            project_repair_binding(&project, directory.clone(), generation, cx)
        })?;
        review.record_before_execution(&current, report, product, &history, now, &cancelled).await?;
        let receipt = review.receipt.ok_or_else(|| anyhow::anyhow!("Reproduction has no durable receipt"))?;
        let deadline = executor.now() + std::time::Duration::from_secs(review.action.duration_seconds);
        let mut result_task = None;
        let mut dependency_report = None;
        let operation: Result<(DoctorEvidenceObservation, DiagnosticText)> = async {
            ensure!(!completed.is_canceled(), "Reproduction view was closed before dispatch");
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            let test_plan = review.action.test_plan.clone();
            #[cfg(not(any(feature = "rust-tools", feature = "jvm-tools")))]
            let is_test = false;
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            let is_test = test_plan.is_some();
            if !is_test && review.action.source != project::TaskSourceKind::UserInput {
                let templates = cx.update(|cx| {
                    let store = project.read(cx).task_store();
                    let inventory = store.read(cx).task_inventory().ok_or_else(|| anyhow::anyhow!("Task inventory is unavailable"))?;
                    Ok::<_, anyhow::Error>(inventory.read(cx).list_tasks(None, None, Some(directory.worktree_id), cx))
                })?.await;
                ensure!(templates.iter().any(|(source, template)| source == &review.action.source && template == review.action.task.original_task()), "Selected Task configuration changed or cannot be verified; review a fresh existing Task");
            }
            #[cfg(feature = "jvm-tools")]
            let previous_report = if let Some(DoctorTestPlan::Java(plan)) = &test_plan {
                let capture = cx.update(|cx| {
                    let provider = project.read(cx).jvm_test_provider_store().cloned().ok_or_else(|| anyhow::anyhow!("Java test owner unavailable"))?;
                    Ok::<_, anyhow::Error>(provider.update(cx, |store, cx| store.capture_report_version_with_limit(plan, 5 * 1024 * 1024, cx)))
                })?;
                Some(match prepare_probe_until_stop_with_limit(capture, cancelled.clone(), executor.clone(), deadline.saturating_duration_since(executor.now())).await? {
                    Ok(version) => version,
                    Err(ProbeEnd::Cancelled) => anyhow::bail!("Selected JUnit baseline capture cancelled"),
                    Err(ProbeEnd::TimedOut) => anyhow::bail!("Selected JUnit baseline capture timed out"),
                    Err(end) => anyhow::bail!("Selected JUnit baseline collection ended without a usable bounded result: {end:?}"),
                })
            } else { None };
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            let mut selected_run = None;
            if let Some(collection) = &review.action.dependency_collection {
                let preflight = cx.update(|cx| collection.preflight(&project, &directory, cx))?;
                match prepare_probe_until_stop_with_limit(preflight, cancelled.clone(), executor.clone(), deadline.saturating_duration_since(executor.now())).await? {
                    Ok(()) => {},
                    Err(end) => anyhow::bail!("Dependency collection preflight did not complete: {end:?}"),
                }
            }
            let handle = cx.update_window(window, |_, window, cx| {
                review.action.revalidate_inputs(&project, &directory, cx)?;
                review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Reproduction cancelled before Task dispatch");
                ensure!(!completed.is_canceled() && !project.read(cx).is_shared(), "Reproduction caller closed or development host became shared before dispatch");
                ensure!(executor.now() < deadline, "Reviewed reproduction deadline reached before dispatch");
                workspace.update(cx, |workspace, cx| {
                    ensure!(workspace.project().entity_id() == project.entity_id(), "Workspace project changed");
                    let mut task = review.action.task.clone();
                    task.resolved.id = task::TaskId(format!("made-doctor-{receipt}"));
                    let handle = workspace.schedule_resolved_task_with_structured_handle(review.action.source.clone(), task, true, window, cx);
                    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                    if let Some(plan) = &test_plan {
                        let registered: Result<_> = match plan {
                            #[cfg(feature = "rust-tools")]
                            DoctorTestPlan::Rust(project::rust_test_provider::RustTestActionPlan::Task { run_id, discovery_generation, scope_node_ids, .. }) => {
                                project.read(cx).rust_test_provider_store().clone().update(cx, |store, cx| store.register_run_handle(run_id.clone(), *discovery_generation, scope_node_ids.clone(), handle.clone(), cx)).map(|()| run_id.clone())
                            }
                            #[cfg(feature = "rust-tools")]
                            DoctorTestPlan::Rust(_) => Err(anyhow::anyhow!("Unsupported Rust test action")),
                            #[cfg(feature = "jvm-tools")]
                            DoctorTestPlan::Java(plan) => {
                                let version = previous_report.clone().ok_or_else(|| anyhow::anyhow!("Selected JUnit baseline receipt missing"))?;
                                project.read(cx).jvm_test_provider_store().cloned().ok_or_else(|| anyhow::anyhow!("Java test owner unavailable"))?.update(cx, |store, cx| store.register_run_handle_with_limit(plan.clone(), version, handle.clone(), 5 * 1024 * 1024, cx))
                            }
                            #[cfg(feature = "jvm-tools")]
                            DoctorTestPlan::RemoteJava(plan) => {
                                project.read(cx).jvm_test_provider_store().cloned().ok_or_else(|| anyhow::anyhow!("Remote JVM owner unavailable"))?.update(cx, |store, cx| store.register_remote_run_handle(plan.clone(), handle.clone(), cx))?;
                                Ok(plan.run_id.clone())
                            }
                        };
                        match registered { Ok(run) => selected_run = Some(run), Err(error) => { handle.cancel(cx); return Err(error); } }
                    }
                    Ok::<_, anyhow::Error>(handle)
                })?
            })??;
            result_task = Some(handle.clone());
            let _authority_subscriptions = cx.update(|cx| {
                let binding = review.binding.clone();
                let project = project.downgrade();
                let observed_project = project.clone();
                let workspace = workspace.clone();
                let cancellation_handle = handle.clone();
                let invalidate: std::rc::Rc<dyn Fn(&mut gpui::App)> = std::rc::Rc::new(move |cx| {
                    let current = observed_project.upgrade().and_then(|project| project_repair_binding(&project, binding.directory.clone(), binding.generation, cx).ok());
                    if current.as_ref() != Some(&binding) || workspace.read_with(cx, |workspace, _| workspace.project().entity_id()).ok() != Some(binding.project) {
                        cancellation_handle.cancel(cx);
                    }
                });
                let mut subscriptions = vec![cx.observe_global::<settings::SettingsStore>({
                    let invalidate = invalidate.clone(); move |cx| invalidate(cx)
                })];
                subscriptions.push(cx.subscribe(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Reproduction project closed"))?, {
                    let invalidate = invalidate.clone(); let handle = handle.clone();
                    move |_, event: &project::Event, cx| {
                        if matches!(event, project::Event::Closed | project::Event::WorktreeRemoved(_) | project::Event::DisconnectedFromHost | project::Event::DisconnectedFromRemote { .. } | project::Event::RemoteIdChanged(_)) {
                            handle.cancel(cx);
                        } else { invalidate(cx); }
                    }
                }));
                if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
                    subscriptions.push(cx.subscribe(&trust, move |_, _, cx| invalidate(cx)));
                }
                Ok::<_, anyhow::Error>(subscriptions)
            })?;
            loop {
                let authority = cx.update(|cx| {
                    let current = project_repair_binding(&project, directory.clone(), generation, cx)?;
                    review.revalidate(&current)?;
                    ensure!(!project.read(cx).is_shared(), "Development host became shared");
                    ensure!(workspace.read_with(cx, |workspace, _| workspace.project().entity_id())? == project.entity_id(), "Workspace project changed");
                    Ok::<_, anyhow::Error>(())
                });
                let stop = if completed.is_canceled() || cancelled.is_closed() || !cancelled.is_empty() { Some((DoctorEvidenceObservation::Cancelled, "Cancelled".to_owned())) }
                    else if let Err(error) = authority { Some((DoctorEvidenceObservation::Incomplete, format!("Authority/host changed: {error}"))) }
                    else if executor.now() >= deadline { Some((DoctorEvidenceObservation::Incomplete, "Reviewed reproduction deadline reached".to_owned())) } else { None };
                if let Some((observation, reason)) = stop {
                    let state = cx.update(|cx| { handle.cancel(cx); handle.state() });
                    return Ok((observation, DiagnosticText::new(format!("{reason}; existing Task cancellation requested; reached development writes remain. Owner state: {state:?}. No reproduced failure or verified repair is claimed."))));
                }
                match handle.state() {
                    task::StructuredTaskState::Completed { exit_code, .. } => {
                        if let Some(collection) = &review.action.dependency_collection {
                            ensure!(exit_code == Some(0), "Failed or unknown Task exit cannot establish a dependency report");
                            let capture = cx.update(|cx| collection.capture(&project, &directory, review.binding.clone(), receipt, cx))?;
                            let captured = match prepare_probe_until_stop_with_limit(capture, cancelled.clone(), executor.clone(), deadline.saturating_duration_since(executor.now())).await? {
                                Ok(report) => report,
                                Err(end) => anyhow::bail!("Dependency report collection did not complete: {end:?}"),
                            };
                            cx.update(|cx| {
                                collection.revalidate(&project, &directory, cx)?;
                                review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                                ensure!(!completed.is_canceled() && !cancelled.is_closed() && cancelled.is_empty()
                                    && executor.now() < deadline, "Dependency report authority, cancellation or deadline changed");
                                Ok::<_, anyhow::Error>(())
                            })?;
                            let count = captured.byte_count();
                            dependency_report = Some(captured);
                            return Ok((DoctorEvidenceObservation::CommandExit { code: exit_code }, DiagnosticText::new(format!("Observed selected Task exit and collected {count} bounded dependency-report bytes. Existing ecosystem ownership must validate provenance and freshness before graph admission. No vulnerability absence or verified repair is claimed; raw report data is excluded from history/model/export."))));
                        }
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        if let (Some(plan), Some(run)) = (&test_plan, &selected_run) {
                            match cx.update(|cx| selected_test_observation(plan, run, &project, exit_code, cx))? {
                                Some(observation) => return Ok(observation),
                                None => { executor.timer(std::time::Duration::from_millis(100)).await; continue; }
                            }
                        }
                        return Ok((DoctorEvidenceObservation::CommandExit { code: exit_code }, DiagnosticText::new("Observed only the selected Task exit status. Inspect selected test/debug evidence for functional assertions; this is not a verified repair.")));
                    },
                    task::StructuredTaskState::SpawnError { message } => return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new(format!("Existing Task could not launch: {message}")))),
                    task::StructuredTaskState::Cancelled { termination_confirmed, .. } => return Ok((DoctorEvidenceObservation::Cancelled, DiagnosticText::new(format!("Task owner cancelled; termination confirmed {termination_confirmed}; reached writes remain.")))),
                    _ => executor.timer(std::time::Duration::from_millis(100)).await,
                }
            }
        }.await;
        #[cfg(feature = "jvm-tools")]
        let operation = if let Some(collection) = &review.action.dependency_collection {
            match collection.cleanup_runtime_inputs() {
                Ok(()) => operation,
                Err(error) => { dependency_report = None; Err(anyhow::anyhow!("Reached Task effects remain; disposable input cleanup failed: {error}")) }
            }
        } else { operation };
        let (observation, detail) = operation.unwrap_or_else(|error| {
            let observation = if completed.is_canceled() || cancelled.is_closed() || !cancelled.is_empty() { DoctorEvidenceObservation::Cancelled } else { DoctorEvidenceObservation::Incomplete };
            (observation, DiagnosticText::new(format!("Reproduction did not reach an attributable result: {error}")))
        });
        review.operation_finished(if observation == DoctorEvidenceObservation::Cancelled { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Completed }, detail.clone())?;
        if review.phase == DoctorRepairPhase::Rechecking { review.recheck_unavailable(DiagnosticText::new(format!("{} Relevant reproduction and independent verification are required before a fix can be established.", detail.as_str())))?; }
        let finished_at = now.saturating_add(executor.now().duration_since(started).as_secs());
        let measurement_complete = matches!(&observation, DoctorEvidenceObservation::FunctionalFailure { .. } | DoctorEvidenceObservation::FunctionalSuccess { .. } | DoctorEvidenceObservation::CommandExit { code: Some(_) });
        let evidence = DoctorEvidenceInput { owner: if matches!(observation, DoctorEvidenceObservation::FunctionalFailure { .. } | DoctorEvidenceObservation::FunctionalSuccess { .. }) { DoctorEvidenceOwner::Test } else { DoctorEvidenceOwner::Task }, owner_reference: DiagnosticText::new(format!("task:{receipt}")), classification: DoctorEvidenceClass::Observed, observation, detail, observed_at: finished_at, historical: false };
        let mut investigation = DoctorInvestigation::new(review.action.selection.clone(), review.report_scope.clone().ok_or_else(|| anyhow::anyhow!("Reproduction scope receipt is missing"))?, review.binding.clone(), now)?;
        investigation.record(&review.binding, &review.action.selection, evidence.clone())?;
        let elapsed = measurement_complete.then(|| DoctorEvidenceInput {
            owner: DoctorEvidenceOwner::Task, owner_reference: DiagnosticText::new(format!("task-elapsed:{receipt}")),
            classification: DoctorEvidenceClass::Observed,
            observation: DoctorEvidenceObservation::AggregateCounter { metric: DiagnosticText::new("reviewed Task end-to-end wall time"), value: DiagnosticText::new(executor.now().duration_since(started).as_millis().to_string()), unit: DiagnosticText::new("ms") },
            detail: DiagnosticText::new(DOCTOR_TASK_TIMING_PROTOCOL), observed_at: finished_at, historical: false,
        });
        if let Some(elapsed) = &elapsed { investigation.record(&review.binding, &review.action.selection, elapsed.clone())?; }
        let comparison = cx.update(|cx| investigation.comparison_snapshot_at(&project, cx)).unwrap_or_else(|error| {
            log::debug!("Doctor comparison source identity unavailable: {error}; verification remains incomplete");
            investigation.comparison_snapshot()
        });
        let mut final_report = DoctorReport::new(investigation.summary.scope.clone(), generation, finished_at);
        final_report.push(review.before.clone())?;
        final_report.set_investigation(investigation.summary().clone())?;
        history.finish(product, receipt, if review.phase == DoctorRepairPhase::Cancelled { ReceiptOutcome::Cancelled } else { ReceiptOutcome::Incomplete }, final_report, None, finished_at).await?;
        Ok(DoctorReproductionResult { review, evidence, elapsed, comparison, task: result_task, dependency_report })
      }.await;
      if completed.send(result).is_err() { log::debug!("Doctor reproduction caller closed; durable observed receipt remains available"); }
    }).detach();
    cx.background_spawn(async move {
        completion
            .await
            .map_err(|_| anyhow::anyhow!("Doctor reproduction completion was interrupted"))?
    })
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub struct DoctorDebugAction {
    launch: Option<project::debugger::dap_store::DiagnosticDebugLaunch>,
    selection: DoctorInvestigationSelection,
    scan_id: usize,
    buffer_versions: Vec<String>,
    duration_seconds: u64,
    preview: Vec<String>,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DoctorDebugAction {
    pub fn selection(&self) -> &DoctorInvestigationSelection {
        &self.selection
    }
    pub fn exact_preview(&self) -> &[String] {
        &self.preview
    }

    fn revalidate_inputs(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        cx: &gpui::App,
    ) -> Result<()> {
        let owner = project.read(cx);
        ensure!(
            !owner.is_shared() && !owner.is_isolated_recipe_test(),
            "Diagnostic debugging requires the selected exclusive development project"
        );
        let worktree = owner
            .worktree_for_id(directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Selected worktree is unavailable"))?;
        ensure!(
            worktree.read(cx).scan_id() == self.scan_id
                && reproduction_buffer_versions(project, directory.worktree_id, cx)
                    == self.buffer_versions,
            "Diagnostic source or open-buffer inputs changed; review a fresh investigation"
        );
        Ok(())
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub fn prepare_debug(
    project: gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    launch: project::debugger::dap_store::DiagnosticDebugLaunch,
    mut selection: DoctorInvestigationSelection,
    effects: DoctorReproductionEffects,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorDebugAction>>> {
    let preparation = (|| {
        selection.validate()?;
        ensure!(
            selection.scope != DoctorInvestigationScope::Product
                && matches!(
                    selection.problem,
                    DoctorInvestigationProblem::Crash
                        | DoctorInvestigationProblem::Test
                        | DoctorInvestigationProblem::IncorrectBehavior
                        | DoctorInvestigationProblem::Concurrency
                ),
            "Selected metadata-only debugger cannot establish CPU/memory or pipeline evidence; select supported instrumentation"
        );
        ensure!(
            (1..=300).contains(&duration_seconds),
            "Debugger collection deadline must be between one second and five minutes"
        );
        let DoctorReproductionEffects::Development {
            file_writes,
            subprocesses,
        } = &effects
        else {
            anyhow::bail!(
                "Unknown, production, external-service, destructive or sensitive collection effects are not approved"
            );
        };
        ensure!(
            !file_writes.trim().is_empty()
                && !subprocesses.trim().is_empty()
                && file_writes.len() + subprocesses.len() <= 8192,
            "Disclose bounded expected development writes and subprocesses"
        );
        ensure!(
            launch.program_path().worktree_id == directory.worktree_id,
            "Diagnostic launch belongs to another selected worktree"
        );
        let binding = project_repair_binding(&project, directory.clone(), generation, cx)?;
        let owner = project.read(cx);
        ensure!(
            !owner.is_shared() && !owner.is_isolated_recipe_test(),
            "Diagnostic debugging requires an exclusive normal development project"
        );
        let worktree = owner
            .worktree_for_id(directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Selected worktree is unavailable"))?;
        let scan_id = worktree.read(cx).scan_id();
        let buffers = reproduction_buffer_versions(&project, directory.worktree_id, cx);
        let binary = launch.binary();
        let preview = vec![
            format!("Installed adapter: {:?}; arguments: {:?}; exact explicit environment: {:?}", binary.command, binary.arguments, binary.envs),
            format!("Program and launch configuration: {}", binary.request_args.configuration),
            format!("Working directory: {:?}; selected host: {:?}", binary.cwd, binding.host),
            format!("Development effects: {effects:?}; deadline: {duration_seconds} seconds"),
            "Collection: finite stop/exception counts and exit metadata; at most 10 MiB of inbound DAP data. No variables, memory, evaluation, raw output, source, payloads, dumps, protocol logs, reverse terminal/child launch or external attachment. Provider gaps remain unsupported.".into(),
            "Project code executes with the user's existing host permissions; this is not an OS resource/filesystem/network sandbox. No new credential, service or production authority is granted. No implicit download, build, automatic save, unattended consent or replay. Cancellation stops only this owned session; reached writes remain. Launch, exit zero and absence of an exception cannot verify a repair.".into(),
        ];
        let identity = format!("{binary:?}:{effects:?}:{duration_seconds}:{scan_id}:{buffers:?}");
        selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
            &selection.input_identity.0,
            &identity,
        ])?;
        let before = DoctorCheck::new(
            &format!("investigation.debug.{}", selection.input_identity.0),
            "Reviewed debugger investigation",
            CheckOutcome::Untested,
            "No selected debugger reproduction has run; a launch or exit status alone cannot establish a verified fix",
            observed_at,
        );
        let disclosure = DiagnosticText::new(format!(
            "Selected installed adapter {}; reviewed development writes {file_writes}; subprocesses {subprocesses}; metadata only; {duration_seconds} second deadline; ten MiB inbound bound; scoped stop, no replay or repair proof",
            launch.definition().adapter
        ));
        let pending = owner.dap_store().update(cx, |store, cx| {
            store.revalidate_diagnostic_launch(&launch, cx)
        });
        Ok::<_, anyhow::Error>((
            pending, binding, before, disclosure, scan_id, buffers, preview,
        ))
    })();
    let (pending, binding, before, disclosure, scan_id, buffer_versions, preview) =
        match preparation {
            Ok(prepared) => prepared,
            Err(error) => return gpui::Task::ready(Err(error)),
        };
    cx.spawn(async move |cx| {
        pending.await?;
        let action = DoctorDebugAction {
            launch: Some(launch),
            selection,
            scan_id,
            buffer_versions,
            duration_seconds,
            preview,
        };
        cx.update(|cx| {
            action.revalidate_inputs(&project, &directory, cx)?;
            ensure!(
                project_repair_binding(&project, directory, generation, cx)? == binding,
                "Diagnostic authority changed during preparation"
            );
            DoctorRepairReview::new(action, binding, before, disclosure)
        })
    })
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub struct DoctorDebugResult {
    pub review: DoctorRepairReview<DoctorDebugAction>,
    pub evidence: DoctorEvidenceInput,
    pub session: Option<gpui::Entity<project::debugger::session::Session>>,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub fn run_debug(
    mut review: DoctorRepairReview<DoctorDebugAction>,
    project: gpui::Entity<project::Project>,
    workspace: gpui::WeakEntity<workspace::Workspace>,
    window: gpui::AnyWindowHandle,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorDebugResult>> {
    use gpui::AppContext as _;
    let executor = cx.background_executor().clone();
    let directory = review.binding.directory.clone();
    let generation = review.binding.generation;
    let (completed, completion) = futures::channel::oneshot::channel();
    let completed = std::rc::Rc::new(std::cell::RefCell::new(Some(completed)));
    cx.spawn(async move |cx| {
        let result = async {
            let started = executor.now();
            let current = cx.update(|cx| {
                review.action.revalidate_inputs(&project, &directory, cx)?;
                project_repair_binding(&project, directory.clone(), generation, cx)
            })?;
            review.record_before_execution(&current, report, product, &history, now, &cancelled).await?;
            let receipt = review.receipt.ok_or_else(|| anyhow::anyhow!("Diagnostic action has no durable receipt"))?;
            let deadline = executor.now() + std::time::Duration::from_secs(review.action.duration_seconds);
            let mut selected = None;
            let operation: Result<(DoctorEvidenceObservation, DiagnosticText)> = async {
                let pending = cx.update_window(window, |_, window, cx| {
                    review.action.revalidate_inputs(&project, &directory, cx)?;
                    review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                    ensure!(!completed.borrow().as_ref().is_none_or(|sender| sender.is_canceled()) && !cancelled.is_closed() && cancelled.is_empty(), "Diagnostic action cancelled before session dispatch");
                    workspace.update(cx, |workspace, cx| {
                        ensure!(workspace.project().entity_id() == project.entity_id(), "Diagnostic workspace project changed");
                        let launch = review.action.launch.take().ok_or_else(|| anyhow::anyhow!("Diagnostic launch was already consumed"))?;
                        let authority: project::debugger::dap_store::DiagnosticDebugAuthority = std::rc::Rc::new({
                            let completed = completed.clone();
                            let cancelled = cancelled.clone();
                            let project = project.clone();
                            let workspace = cx.entity().downgrade();
                            let directory = directory.clone();
                            let binding = review.binding.clone();
                            let buffer_versions = review.action.buffer_versions.clone();
                            let scan_id = review.action.scan_id;
                            let executor = executor.clone();
                            move |cx| {
                                ensure!(!completed.borrow().as_ref().is_none_or(|sender| sender.is_canceled())
                                    && !cancelled.is_closed() && cancelled.is_empty() && executor.now() < deadline,
                                    "Diagnostic caller closed, action cancelled or review deadline elapsed before boot");
                                ensure!(project_repair_binding(&project, directory.clone(), generation, cx)? == binding,
                                    "Diagnostic approval inputs or permissions changed before boot");
                                ensure!(!project.read(cx).is_shared() && !project.read(cx).is_isolated_recipe_test(),
                                    "Diagnostic development authority changed");
                                let worktree = project.read(cx).worktree_for_id(directory.worktree_id, cx)
                                    .ok_or_else(|| anyhow::anyhow!("Selected worktree became unavailable"))?;
                                ensure!(worktree.read(cx).scan_id() == scan_id
                                    && reproduction_buffer_versions(&project, directory.worktree_id, cx) == buffer_versions,
                                    "Diagnostic source inputs changed before boot");
                                workspace.read_with(cx, |workspace, _| { ensure!(workspace.project().entity_id() == project.entity_id(), "Diagnostic workspace project changed"); Ok::<_, anyhow::Error>(()) })?
                            }
                        });
                        Ok::<_, anyhow::Error>(workspace.start_diagnostic_debug_session(launch, task::SharedTaskContext::default(), authority, window, cx))
                    })?
                })??;
                let handle = match prepare_probe_until_stop_with_limit(pending, cancelled.clone(), executor.clone(), deadline.saturating_duration_since(executor.now())).await? {
                    Ok(handle) => handle,
                    Err(ProbeEnd::Cancelled) => return Ok((DoctorEvidenceObservation::Cancelled, DiagnosticText::new("Diagnostic preparation cancelled before session admission"))),
                    Err(_) => return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Diagnostic native-input preparation exceeded the reviewed deadline"))),
                };
                selected = Some(handle.session.clone());
                let initialized = prepare_probe_until_stop_with_limit(handle.boot, cancelled.clone(), executor.clone(), deadline.saturating_duration_since(executor.now())).await?;
                match initialized {
                    Ok(()) => {},
                    Err(ProbeEnd::Cancelled) => return Ok((DoctorEvidenceObservation::Cancelled, DiagnosticText::new("Diagnostic initialization cancelled; no complete runtime evidence"))),
                    Err(_) => return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Diagnostic initialization exceeded the reviewed deadline"))),
                }
                loop {
                    let authority = cx.update(|cx| {
                        review.action.revalidate_inputs(&project, &directory, cx)?;
                        review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                        workspace.read_with(cx, |workspace, _| { ensure!(workspace.project().entity_id() == project.entity_id(), "Workspace changed"); Ok::<_, anyhow::Error>(()) })?
                    });
                    if completed.borrow().as_ref().is_none_or(|sender| sender.is_canceled()) || cancelled.is_closed() || !cancelled.is_empty() {
                        return Ok((DoctorEvidenceObservation::Cancelled, DiagnosticText::new("Diagnostic collection cancelled; partial metadata is not a completed test and reached development writes remain")));
                    }
                    if authority.is_err() {
                        return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Diagnostic authority or inputs changed; captured results cannot establish the selected target")));
                    }
                    let (observation, terminated) = handle.session.read_with(cx, |session, _| (session.diagnostic_observation().cloned(), session.is_terminated()));
                    let observation = observation.ok_or_else(|| anyhow::anyhow!("Immutable collection restriction is unavailable"))?;
                    if observation.truncated || observation.blocked_reverse_request {
                        return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Diagnostic metadata was truncated or the workflow requested an unapproved child/terminal dependency; collection is incomplete")));
                    }
                    if observation.exception_count > 0 {
                        return Ok((DoctorEvidenceObservation::Exception { kind: DiagnosticText::new("Selected adapter reported an exception stop") }, DiagnosticText::new(format!("Observed {} exception stops in {} bounded stops from this selected session. No exception message, variable, memory, source or stack data was collected. This reproduces an exception indication, not its cause or a verified repair.", observation.exception_count, observation.stop_count))));
                    }
                    if terminated || observation.exit_code.is_some() {
                        return Ok((DoctorEvidenceObservation::InsufficientEvidence, DiagnosticText::new(format!("Selected debugger reported exit {:?} and {} stop indications without an observed exception. Launch or exit alone does not establish correctness, race freedom or a verified fix.", observation.exit_code, observation.stop_count))));
                    }
                    if executor.now() >= deadline {
                        return Ok((DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Reviewed diagnostic collection deadline reached; absence of an observed exception is insufficient evidence")));
                    }
                    executor.timer(std::time::Duration::from_millis(100)).await;
                }
            }.await;
            let cleanup = if let Some(session) = &selected {
                let pending = cx.update(|cx| {
                    let session_id = session.read(cx).session_id();
                    project.read(cx).dap_store().update(cx, |store, cx| {
                        store.session_by_id(session_id).is_some().then(|| store.shutdown_session(session_id, cx))
                    })
                });
                match pending { Some(pending) => pending.await, None => Ok(()) }
            } else { Ok(()) };
            let (mut observation, mut detail) = operation.unwrap_or_else(|_| (DoctorEvidenceObservation::Incomplete, DiagnosticText::new("Selected diagnostic owner failed; no complete attributable runtime result was observed")));
            if completed.borrow().as_ref().is_none_or(|sender| sender.is_canceled()) || cancelled.is_closed() || !cancelled.is_empty() {
                observation = DoctorEvidenceObservation::Cancelled;
                detail = DiagnosticText::new("Diagnostic action cancelled or its view closed; the selected session was stopped, reached effects remain and no result is verified");
            }
            if cleanup.is_err() {
                observation = DoctorEvidenceObservation::Incomplete;
                detail = DiagnosticText::new("Diagnostic owner cleanup failed; inspect the selected Debugger session. Termination is unconfirmed and the investigation remains incomplete");
            }
            review.operation_finished(if observation == DoctorEvidenceObservation::Cancelled { DoctorRepairOperationOutcome::Cancelled } else if observation == DoctorEvidenceObservation::Incomplete { DoctorRepairOperationOutcome::Failed } else { DoctorRepairOperationOutcome::Completed }, detail.clone())?;
            if review.phase == DoctorRepairPhase::Rechecking {
                review.recheck_unavailable(DiagnosticText::new("Diagnostic evidence is a reproduction observation; an independently reviewed relevant functional verification is still required"))?;
            }
            let finished_at = now.saturating_add(executor.now().duration_since(started).as_secs());
            let evidence = DoctorEvidenceInput { owner: DoctorEvidenceOwner::Debugger, owner_reference: DiagnosticText::new(format!("debug:{receipt}")), classification: DoctorEvidenceClass::Observed, observation, detail, observed_at: finished_at, historical: false };
            let mut investigation = DoctorInvestigation::new(review.action.selection.clone(), review.report_scope.clone().ok_or_else(|| anyhow::anyhow!("Diagnostic report scope is missing"))?, review.binding.clone(), now)?;
            investigation.record(&review.binding, &review.action.selection, evidence.clone())?;
            let mut final_report = DoctorReport::new(investigation.summary.scope.clone(), generation, finished_at);
            final_report.push(review.before.clone())?;
            final_report.set_investigation(investigation.summary().clone())?;
            history.finish(product, receipt, if review.phase == DoctorRepairPhase::Cancelled { ReceiptOutcome::Cancelled } else { ReceiptOutcome::Incomplete }, final_report, None, finished_at).await?;
            Ok(DoctorDebugResult { review, evidence, session: selected })
        }.await;
        if let Some(sender) = completed.borrow_mut().take() {
            if sender.send(result).is_err() { log::debug!("Doctor debugger view closed; durable receipt remains and no action is replayed"); }
        } else {
            log::error!("Doctor debugger completion was already consumed");
        }
    }).detach();
    cx.background_spawn(async move {
        completion
            .await
            .map_err(|_| anyhow::anyhow!("Doctor debugger operation was interrupted"))?
    })
}

#[cfg(feature = "rust-tools")]
pub struct DoctorProfileAction {
    plan: Option<project::profile_results_store::DiagnosticProfileImport>,
    selection: DoctorInvestigationSelection,
    duration_seconds: u64,
    preview: Vec<String>,
}

#[cfg(feature = "rust-tools")]
impl DoctorProfileAction {
    pub fn selection(&self) -> &DoctorInvestigationSelection {
        &self.selection
    }
    pub fn exact_preview(&self) -> &[String] {
        &self.preview
    }
}

#[cfg(feature = "rust-tools")]
pub fn prepare_profile(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    paths: [project::ProjectPath; 2],
    mut selection: DoctorInvestigationSelection,
    duration_seconds: u64,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> Result<DoctorRepairReview<DoctorProfileAction>> {
    selection.validate()?;
    ensure!(
        selection.scope == DoctorInvestigationScope::Application
            && matches!(
                selection.problem,
                DoctorInvestigationProblem::Cpu | DoctorInvestigationProblem::SlowOperation
            ),
        "Native Samply sample import supports CPU/slow-operation inspection, not memory, crash, heap/core/JFR, environment or payload collection"
    );
    ensure!(
        (1..=300).contains(&duration_seconds),
        "Profile import deadline must be between one second and five minutes"
    );
    ensure!(
        paths
            .iter()
            .all(|path| path.worktree_id == directory.worktree_id),
        "Select both profile artifacts from the reviewed worktree"
    );
    let binding = project_repair_binding(project, directory, generation, cx)?;
    let plan = project.update(cx, |owner, cx| {
        owner.prepare_diagnostic_profile_import(paths.clone(), cx)
    })?;
    selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
        &selection.input_identity.0,
        &plan.input_identity(),
        &duration_seconds.to_string(),
    ])?;
    let preview = vec![
        format!("Selected profile and explicit symbol sidecar: {paths:?}; selected host: {:?}", binding.host),
        format!("Read/normalize only selected confined local macOS Samply 0.13.1 artifacts; combined 10 MiB ceiling before allocation; import deadline {duration_seconds} seconds. No writes, network, installation or collector process."),
        "Only aggregate sample counts enter Doctor evidence/history/model handoffs. Normalized call-tree data stays in the existing ProfileView with project privacy/trust revocation. No heap/core/environment/request payloads, JVM JFR, remote attachment, or remote artifact transfer.".into(),
        "Import is not a new capture and cannot establish the workload, CPU time, memory consumption, a comparable baseline, confirmed cause or verified repair. Use a separately reviewed existing collector Task where supported. Cancellation stops this import only; no automatic replay.".into(),
    ];
    let before = DoctorCheck::new(
        &format!("investigation.profile.{}", selection.input_identity.0),
        "Reviewed profile inspection",
        CheckOutcome::Untested,
        "No selected profile has been inspected; artifact import alone cannot establish runtime measurement or repair",
        observed_at,
    );
    DoctorRepairReview::new(
        DoctorProfileAction {
            plan: Some(plan),
            selection,
            duration_seconds,
            preview,
        },
        binding,
        before,
        DiagnosticText::new(
            "Read only explicitly selected local native profile artifacts under a combined ten MiB limit; aggregate sample counts only; no collector, upload, sensitive dump, unattended consent or repair proof",
        ),
    )
}

#[cfg(feature = "rust-tools")]
pub struct DoctorProfileResult {
    pub review: DoctorRepairReview<DoctorProfileAction>,
    pub evidence: DoctorEvidenceInput,
    pub profile: Option<project::profile_results_store::DiagnosticProfileResult>,
}

#[cfg(feature = "rust-tools")]
pub fn run_profile(
    mut review: DoctorRepairReview<DoctorProfileAction>,
    project: gpui::Entity<project::Project>,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorProfileResult>> {
    use futures::FutureExt as _;
    use gpui::AppContext as _;
    let executor = cx.background_executor().clone();
    let directory = review.binding.directory.clone();
    let generation = review.binding.generation;
    let (completed, completion) = futures::channel::oneshot::channel();
    cx.spawn(async move |cx| {
        let result = async {
            let current = cx.update(|cx| project_repair_binding(&project, directory.clone(), generation, cx))?;
            review.record_before_execution(&current, report, product, &history, now, &cancelled).await?;
            let receipt = review.receipt.ok_or_else(|| anyhow::anyhow!("Profile action has no durable receipt"))?;
            let started = executor.now();
            let deadline = started + std::time::Duration::from_secs(review.action.duration_seconds);
            let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let operation = async {
                let pending = cx.update(|cx| {
                    review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                    ensure!(!completed.is_canceled() && !cancelled.is_closed() && cancelled.is_empty(), "Profile import cancelled before dispatch");
                    let plan = review.action.plan.take().ok_or_else(|| anyhow::anyhow!("Profile action already consumed"))?;
                    project.update(cx, |owner, cx| owner.import_diagnostic_profile(plan, cancellation.clone(), cx))
                })?;
                let mut pending = pending.boxed_local();
                loop {
                    let authority = cx.update(|cx| review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?));
                    if completed.is_canceled() || cancelled.is_closed() || !cancelled.is_empty()
                        || authority.is_err() || executor.now() >= deadline {
                        cancellation.store(true, std::sync::atomic::Ordering::Relaxed);
                        anyhow::bail!("Profile import stopped because caller, authority or reviewed deadline changed");
                    }
                    let timer = executor.timer(std::time::Duration::from_millis(100)).boxed_local();
                    match futures::future::select(pending, timer).await {
                        futures::future::Either::Left((result, _)) => return result,
                        futures::future::Either::Right((_, remaining)) => pending = remaining,
                    }
                }
            }.await;
            let operation = operation.and_then(|profile| {
                cx.update(|cx| {
                    review.revalidate(&project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                    project.read(cx).revalidate_diagnostic_profile_result(&profile, cx)
                })?;
                Ok(profile)
            });
            let cancelled_by_caller = completed.is_canceled() || cancelled.is_closed() || !cancelled.is_empty();
            let (profile, observation, detail) = match operation {
                Ok(profile) if !cancelled_by_caller => {
                    let detail = DiagnosticText::new(format!("Imported {} samples, {} unattributed samples and {} unresolved frames from reviewed artifacts. This does not establish capture provenance, CPU/memory consumption or improvement; relevant verification remains required.", profile.total_samples(), profile.unattributed_samples(), profile.unresolved_frames()));
                    let observation = DoctorEvidenceObservation::AggregateCounter {
                        metric: DiagnosticText::new("Imported samples"), value: DiagnosticText::new(profile.total_samples().to_string()), unit: DiagnosticText::new("samples"),
                    };
                    (Some(profile), observation, detail)
                },
                _ => (None, if cancelled_by_caller { DoctorEvidenceObservation::Cancelled } else { DoctorEvidenceObservation::Incomplete },
                    DiagnosticText::new("Profile import cancelled, expired, failed or became unavailable. No complete runtime measurement or repair is claimed; inspect the current native owner and prepare a fresh review")),
            };
            review.operation_finished(if cancelled_by_caller { DoctorRepairOperationOutcome::Cancelled }
                else if profile.is_some() { DoctorRepairOperationOutcome::Completed } else { DoctorRepairOperationOutcome::Failed }, detail.clone())?;
            if review.phase() == DoctorRepairPhase::Rechecking {
                review.recheck_unavailable(DiagnosticText::new("Import establishes artifact sample counts only; relevant functional verification and a comparable reviewed measurement are still required"))?;
            }
            let finished_at = now.saturating_add(executor.now().duration_since(started).as_secs());
            let evidence = DoctorEvidenceInput { owner: DoctorEvidenceOwner::Profile,
                owner_reference: DiagnosticText::new(format!("profile:{receipt}")), classification: DoctorEvidenceClass::Observed,
                observation, detail, observed_at: finished_at, historical: false };
            let scope = review.report_scope.clone().ok_or_else(|| anyhow::anyhow!("Profile report scope missing"))?;
            let mut investigation = DoctorInvestigation::new(review.action.selection.clone(), scope.clone(), review.binding.clone(), now)?;
            investigation.record(&review.binding, &review.action.selection, evidence.clone())?;
            let mut report = DoctorReport::new(scope, generation, finished_at);
            report.push(review.before.clone())?;
            report.set_investigation(investigation.summary().clone())?;
            history.finish(product, receipt, if cancelled_by_caller { ReceiptOutcome::Cancelled } else { ReceiptOutcome::Incomplete }, report, None, finished_at).await?;
            Ok(DoctorProfileResult { review, evidence, profile })
        }.await;
        if completed.send(result).is_err() { log::debug!("Doctor profile caller closed; receipt persists without replay"); }
    }).detach();
    cx.background_spawn(async move {
        completion
            .await
            .map_err(|_| anyhow::anyhow!("Doctor profile operation interrupted"))?
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoctorRepairPhase {
    Proposed,
    Approved,
    Recording,
    Running,
    Rechecking,
    Verified,
    StillFailing,
    Incomplete,
    Cancelled,
    AwaitingUserAction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoctorRepairOperationOutcome {
    Completed,
    Failed,
    Cancelled,
    AwaitingUserAction,
}

pub struct DoctorRepairReview<Action> {
    action: Action,
    binding: DoctorRepairBinding,
    before: DoctorCheck,
    effects: DiagnosticText,
    phase: DoctorRepairPhase,
    operation: Option<DoctorRepairOperationOutcome>,
    operation_detail: Option<DiagnosticText>,
    after: Option<DoctorCheck>,
    receipt: Option<uuid::Uuid>,
    started_at: Option<u64>,
    report_scope: Option<DoctorScope>,
}

impl<Action> DoctorRepairReview<Action> {
    pub fn new(
        action: Action,
        binding: DoctorRepairBinding,
        before: DoctorCheck,
        effects: DiagnosticText,
    ) -> Result<Self> {
        ensure!(
            !before.id.as_str().is_empty() && !effects.as_str().is_empty(),
            "Repair must identify its check and concrete effects"
        );
        Ok(Self {
            action,
            binding,
            before,
            effects,
            phase: DoctorRepairPhase::Proposed,
            operation: None,
            operation_detail: None,
            after: None,
            receipt: None,
            started_at: None,
            report_scope: None,
        })
    }
    pub fn action(&self) -> &Action {
        &self.action
    }
    pub fn effects(&self) -> &DiagnosticText {
        &self.effects
    }
    pub fn before(&self) -> &DoctorCheck {
        &self.before
    }
    pub fn after(&self) -> Option<&DoctorCheck> {
        self.after.as_ref()
    }
    pub(crate) fn binding(&self) -> &DoctorRepairBinding {
        &self.binding
    }
    pub fn phase(&self) -> DoctorRepairPhase {
        self.phase
    }
    pub fn receipt(&self) -> Option<uuid::Uuid> {
        self.receipt
    }
    pub fn operation_detail(&self) -> Option<&DiagnosticText> {
        self.operation_detail.as_ref()
    }

    pub fn approve(&mut self, current: &DoctorRepairBinding) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Proposed,
            "This repair has already been reviewed; duplicate approval is rejected"
        );
        ensure!(
            &self.binding == current,
            "Repair inputs changed; review a new proposal"
        );
        self.phase = DoctorRepairPhase::Approved;
        Ok(())
    }

    pub fn revalidate(&self, current: &DoctorRepairBinding) -> Result<()> {
        ensure!(
            &self.binding == current,
            "Repair project, host, configuration or generation changed"
        );
        ensure!(
            self.phase == DoctorRepairPhase::Approved || self.phase == DoctorRepairPhase::Running,
            "Repair is not authorized to dispatch an owner action"
        );
        Ok(())
    }

    pub async fn record_before_execution(
        &mut self,
        current: &DoctorRepairBinding,
        report: DoctorReport,
        product: DoctorProduct,
        history: &DoctorHistoryStore,
        now: u64,
        cancelled: &async_channel::Receiver<()>,
    ) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Approved,
            "Repair is not awaiting execution; duplicate dispatch is rejected"
        );
        self.revalidate(current)?;
        ensure!(
            report.generation == self.binding.generation
                && report.checks().iter().any(|check| check == &self.before),
            "Repair report or diagnosed check changed"
        );
        report.validate()?;
        if cancelled.is_closed() || !cancelled.is_empty() {
            self.phase = DoctorRepairPhase::Cancelled;
            anyhow::bail!("Repair cancelled before recording or execution");
        }
        self.phase = DoctorRepairPhase::Recording;
        self.report_scope = Some(report.scope.clone());
        let mut record = DoctorHistoryRecord::new(
            report.clone(),
            Some(self.effects.clone()),
            ReceiptOutcome::Running,
            now,
        );
        record.admission_key = Some(self.binding.admission_key(&self.before.id, &self.effects));
        let receipt = match history.record(product, record, now).await {
            Ok(receipt) => receipt,
            Err(error) => {
                self.phase = DoctorRepairPhase::Incomplete;
                return Err(error);
            }
        };
        self.receipt = Some(receipt);
        self.started_at = Some(now);
        if cancelled.is_closed() || !cancelled.is_empty() {
            self.phase = DoctorRepairPhase::Cancelled;
            history
                .finish(
                    product,
                    receipt,
                    ReceiptOutcome::Cancelled,
                    report,
                    None,
                    now,
                )
                .await?;
            anyhow::bail!("Repair cancelled before owner dispatch");
        }
        self.phase = DoctorRepairPhase::Running;
        Ok(())
    }

    pub fn operation_finished(
        &mut self,
        outcome: DoctorRepairOperationOutcome,
        detail: DiagnosticText,
    ) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Running,
            "Repair is not running"
        );
        self.operation = Some(outcome);
        self.operation_detail = Some(detail);
        self.phase = match outcome {
            DoctorRepairOperationOutcome::Completed | DoctorRepairOperationOutcome::Failed => {
                DoctorRepairPhase::Rechecking
            }
            DoctorRepairOperationOutcome::Cancelled => DoctorRepairPhase::Cancelled,
            DoctorRepairOperationOutcome::AwaitingUserAction => {
                DoctorRepairPhase::AwaitingUserAction
            }
        };
        Ok(())
    }

    pub fn observe_recheck(
        &mut self,
        current: &DoctorRepairBinding,
        after: DoctorCheck,
    ) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Rechecking,
            "Repair is not awaiting a relevant recheck"
        );
        let relevant = self.binding.same_scope(current)
            && current.input_revision == self.binding.input_revision
            && current.generation >= self.binding.generation
            && after.id == self.before.id
            && !after.historical
            && self
                .started_at
                .is_some_and(|start| after.observed_at >= start);
        self.phase = if !relevant {
            DoctorRepairPhase::Incomplete
        } else {
            match after.outcome {
                CheckOutcome::Ready
                    if self.operation == Some(DoctorRepairOperationOutcome::Completed) =>
                {
                    DoctorRepairPhase::Verified
                }
                CheckOutcome::Failed | CheckOutcome::Blocked => DoctorRepairPhase::StillFailing,
                CheckOutcome::Cancelled => DoctorRepairPhase::Cancelled,
                _ => DoctorRepairPhase::Incomplete,
            }
        };
        self.after = Some(after);
        Ok(())
    }

    pub fn recheck_unavailable(&mut self, detail: DiagnosticText) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Rechecking,
            "Repair is not awaiting revalidation"
        );
        self.operation_detail = Some(detail);
        self.phase = DoctorRepairPhase::Incomplete;
        Ok(())
    }

    #[cfg(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools"))]
    pub(crate) fn cancel_recheck(&mut self, detail: DiagnosticText) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Rechecking,
            "Repair is not awaiting revalidation"
        );
        self.phase = DoctorRepairPhase::Cancelled;
        self.operation_detail = Some(detail);
        Ok(())
    }

    pub fn cancel_before_execution(&mut self) -> Result<()> {
        ensure!(
            matches!(
                self.phase,
                DoctorRepairPhase::Proposed | DoctorRepairPhase::Approved
            ),
            "A running action must be cancelled through its existing owner and report reached effects"
        );
        self.phase = DoctorRepairPhase::Cancelled;
        Ok(())
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) fn note_cleanup(&mut self, detail: DiagnosticText) -> Result<()> {
        ensure!(
            matches!(
                self.phase,
                DoctorRepairPhase::Verified
                    | DoctorRepairPhase::StillFailing
                    | DoctorRepairPhase::Cancelled
                    | DoctorRepairPhase::Incomplete
                    | DoctorRepairPhase::AwaitingUserAction
            ),
            "Cleanup notes require an observed terminal operation"
        );
        self.operation_detail = Some(DiagnosticText::new(format!(
            "{}; {}",
            self.operation_detail
                .as_ref()
                .map_or("Observed operation", DiagnosticText::as_str),
            detail.as_str()
        )));
        Ok(())
    }

    pub async fn persist_outcome(
        &self,
        product: DoctorProduct,
        history: &DoctorHistoryStore,
        session_id: Option<DiagnosticText>,
        now: u64,
    ) -> Result<()> {
        let outcome = match self.phase {
            DoctorRepairPhase::Verified => ReceiptOutcome::Completed,
            DoctorRepairPhase::StillFailing => ReceiptOutcome::Failed,
            DoctorRepairPhase::Cancelled => ReceiptOutcome::Cancelled,
            DoctorRepairPhase::Incomplete => ReceiptOutcome::Incomplete,
            DoctorRepairPhase::AwaitingUserAction => ReceiptOutcome::AwaitingUserAction,
            _ => anyhow::bail!("Repair has not reached an observed terminal result"),
        };
        let receipt = self
            .receipt
            .ok_or_else(|| anyhow::anyhow!("Repair has no durable operation receipt"))?;
        let scope = self
            .report_scope
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Repair has no recorded scope"))?;
        let mut report = DoctorReport::new(scope, self.binding.generation, now);
        report.push(self.before.clone())?;
        if let Some(after) = &self.after {
            let mut after = after.clone();
            after.id = DiagnosticText::new(format!("{}.recheck", after.id.as_str()));
            report.push(after)?;
        }
        if let Some(detail) = &self.operation_detail {
            report.push(DoctorCheck::new(
                "doctor.repair.operation",
                "Observed repair operation",
                CheckOutcome::Untested,
                detail.as_str(),
                now,
            ))?;
        }
        history
            .finish(product, receipt, outcome, report, session_id, now)
            .await
    }
}

pub struct DoctorNativeRepairResult {
    pub session_id: Option<String>,
    pub outcome: DoctorRepairOperationOutcome,
    pub detail: DiagnosticText,
}

pub struct DoctorNativeRepairRequest {
    origin: gpui::WeakEntity<crate::doctor_view::DoctorView>,
    pub workspace: gpui::WeakEntity<workspace::Workspace>,
    pub window: gpui::AnyWindowHandle,
    pub project: gpui::Entity<project::Project>,
    pub directory: project::ProjectPath,
    pub prompt: String,
    pub provider: String,
    pub model: String,
    pub cancelled: async_channel::Receiver<()>,
    authority: Box<dyn Fn(&gpui::App) -> Result<()>>,
    prompt_admission: Box<dyn Fn(&gpui::App) -> Result<()>>,
    history: DoctorHistoryStore,
    product: DoctorProduct,
    receipt: uuid::Uuid,
    claimed: std::cell::Cell<bool>,
    completion: std::cell::RefCell<
        Option<futures::channel::oneshot::Sender<Result<DoctorNativeRepairResult>>>,
    >,
}

impl DoctorNativeRepairRequest {
    #[cfg(feature = "agentic")]
    pub(crate) fn new(
        origin: gpui::WeakEntity<crate::doctor_view::DoctorView>,
        workspace: gpui::WeakEntity<workspace::Workspace>,
        window: gpui::AnyWindowHandle,
        project: gpui::Entity<project::Project>,
        directory: project::ProjectPath,
        action: crate::doctor_agent::DoctorNativeRepairAction,
        binding: DoctorRepairBinding,
        history: DoctorHistoryStore,
        product: DoctorProduct,
        receipt: uuid::Uuid,
        cancelled: async_channel::Receiver<()>,
        completion: futures::channel::oneshot::Sender<Result<DoctorNativeRepairResult>>,
    ) -> Self {
        let owner = project.clone();
        let scope = directory.clone();
        let prompt_action = action.clone();
        Self {
            prompt_admission: Box::new(move |cx| prompt_action.mark_prompt_submitted(cx)),
            origin,
            workspace,
            window,
            project,
            directory,
            prompt: action.prompt().to_owned(),
            provider: action.provider(),
            model: action.model(),
            authority: Box::new(move |cx| {
                action.revalidate(cx)?;
                ensure!(
                    project_repair_binding(&owner, scope.clone(), binding.generation, cx)?
                        == binding,
                    "Reviewed native repair project, host, permissions or settings changed"
                );
                ensure!(
                    !owner.read(cx).is_isolated_recipe_test(),
                    "Isolated recipe tests cannot launch Doctor repair sessions"
                );
                Ok(())
            }),
            cancelled,
            history,
            product,
            receipt,
            claimed: std::cell::Cell::new(false),
            completion: std::cell::RefCell::new(Some(completion)),
        }
    }

    pub fn revalidate(&self, cx: &gpui::App) -> Result<()> {
        ensure!(
            !self.cancelled.is_closed() && self.cancelled.is_empty(),
            "Doctor repair cancelled; no new work is authorized"
        );
        (self.authority)(cx)
    }

    pub fn mark_prompt_submitted(&self, cx: &gpui::App) -> Result<()> {
        self.revalidate(cx)?;
        (self.prompt_admission)(cx)
    }

    pub async fn record_session(
        &self,
        session: &str,
        now: u64,
        cx: &mut gpui::AsyncApp,
    ) -> Result<()> {
        self.history
            .link_native_session(self.product, self.receipt, session, now)
            .await?;
        self.origin.update(cx, |view, cx| {
            #[cfg(feature = "agentic")]
            view.native_session_available(session, cx);
            #[cfg(not(feature = "agentic"))]
            {
                let _ = view;
                cx.notify();
            }
        })?;
        Ok(())
    }

    pub fn claim(&self) -> bool {
        !self.claimed.replace(true)
    }

    pub fn complete(&self, result: Result<DoctorNativeRepairResult>, cx: &mut gpui::App) {
        if let Some(completion) = self.completion.borrow_mut().take()
            && let Err(unobserved) = completion.send(result)
        {
            let history = self.history.clone();
            let product = self.product;
            let receipt = self.receipt;
            cx.spawn(async move |_| {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                history
                    .finish_unobserved_native_session(product, receipt, unobserved, now)
                    .await
            })
            .detach_and_log_err(cx);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoctorProjectHandoff {
    Settings,
    ToolchainSelection,
    TaskConfiguration,
    TestSelection,
    DebugConfiguration,
}

impl DoctorProjectHandoff {
    pub fn action(self, cx: &gpui::App) -> Result<Box<dyn gpui::Action>> {
        let name = match self {
            Self::Settings => "zed::OpenProjectSettings",
            Self::ToolchainSelection => "toolchain::Select",
            Self::TaskConfiguration => "zed::OpenProjectTasks",
            Self::TestSelection => "test_explorer::ToggleTestsPanel",
            Self::DebugConfiguration => "debugger::OpenProjectDebugTasks",
        };
        cx.build_action(name, None).map_err(|error| {
            anyhow::anyhow!("This existing product surface is unavailable: {error}")
        })
    }

    pub fn result(self, observed_at: u64) -> DoctorCheck {
        DoctorCheck::new(
            "project.repair.handoff",
            "Awaiting action in existing project controls",
            CheckOutcome::Untested,
            "Opening settings, toolchain, Tasks, Tests or debugger configuration performs no repair or validation. Review the selected project/host in the existing surface, approve any actual operation there, then return to Doctor for a relevant fresh recheck. No pins, wrappers, dependencies or permissions have been changed by this handoff.",
            observed_at,
        )
    }
}

#[derive(Clone, Debug)]
pub enum DoctorProjectRepairAction {
    RestartLanguageServer {
        server: lsp::LanguageServerId,
        name: lsp::LanguageServerName,
    },
    #[cfg(feature = "jvm-tools")]
    RefreshJavaProject {
        server: lsp::LanguageServerId,
        generation: Option<u64>,
    },
    #[cfg(feature = "jvm-tools")]
    RefreshKotlinProject {
        server: lsp::LanguageServerId,
        generation: Option<u64>,
    },
}

pub(crate) fn project_repair_binding(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairBinding> {
    let revision = cx
        .try_global::<settings::SettingsStore>()
        .and_then(|store| store.observed_revision())
        .ok_or_else(|| {
            anyhow::anyhow!("Settings revision is unavailable; repair review is blocked")
        })?;
    DoctorRepairBinding::capture(project, directory, generation, revision, cx)
}

pub struct DoctorApplicationRestartAction {
    observation: workspace::ApplicationRestartObservation,
}

impl DoctorApplicationRestartAction {
    pub fn affected_workspace_count(&self) -> usize {
        self.observation.workspaces.len()
    }

    fn observe(cx: &gpui::App) -> Result<workspace::ApplicationRestartObservation> {
        let observation = workspace::application_restart_observation(cx)?;
        ensure!(
            !observation.workspaces.is_empty(),
            "No affected Made workspace is available"
        );
        ensure!(
            observation.dirty_items.is_empty(),
            "Save or explicitly resolve dirty files in every workspace before reviewing restart"
        );
        ensure!(
            observation.blockers.is_empty(),
            "Application restart is blocked: {}",
            observation.blockers.join("; ")
        );
        #[cfg(feature = "agentic")]
        ensure!(
            !agent::recipes::RecipeStore::try_global(cx)
                .is_some_and(|owner| owner.read(cx).has_active_work(cx)),
            "Finish active native-agent, recipe, schedule or isolated scan work before restarting"
        );
        Ok(observation)
    }

    fn revalidate(&self, cx: &gpui::App) -> Result<()> {
        ensure!(
            Self::observe(cx)? == self.observation,
            "Affected application workspaces or activity changed after review"
        );
        Ok(())
    }
}

pub fn prepare_application_restart(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorApplicationRestartAction>>> {
    let project = project.downgrade();
    cx.spawn(async move |cx| cx.update(|cx| {
        let project = project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?;
        let binding = project_repair_binding(&project, directory, generation, cx)?;
        let action = DoctorApplicationRestartAction { observation: DoctorApplicationRestartAction::observe(cx)? };
        DoctorRepairReview::new(action, binding, DoctorCheck::new("made.application.restart", "Application restart", CheckOutcome::Untested, "Application recovery requires fresh observations after restart", observed_at), DiagnosticText::new("Request normal Made application save/shutdown/restart across all retained workspaces. Resolve dirty files, Tasks, debugger, terminals, agent/recipe/schedule and collection work first. No OS/service restart or forced process killing. Cancelled save prompts and changed inputs stop this request. Persist requested/incomplete; reopen without replay and perform fresh checks."))
    }))
}

pub fn launch_application_restart(
    mut review: DoctorRepairReview<DoctorApplicationRestartAction>,
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorApplicationRestartAction>>> {
    use futures::FutureExt as _;
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let binding = cx.update(|cx| {
            review.action().revalidate(cx)?;
            project_repair_binding(&project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx)
        })?;
        review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
        let preparation = cx.update(|cx| {
            review.action().revalidate(cx)?;
            let binding = project_repair_binding(&project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx)?;
            review.revalidate(&binding)?;
            ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Restart cancelled before save preparation");
            Ok::<_,anyhow::Error>(workspace::prepare_to_reload(cx))
        });
        let mut prepared = None;
        let (outcome, detail) = match preparation {
            Ok(task) => match futures::future::select(task.boxed(), cancelled.recv().boxed()).await {
                futures::future::Either::Left((Ok(result), _)) if result.accepted => {
                    prepared = Some(result);
                    (DoctorRepairOperationOutcome::Completed, DiagnosticText::new("Normal all-workspace save preparation accepted; application restart and problem recovery remain unobserved"))
                }
                futures::future::Either::Left((Ok(_), _)) => (DoctorRepairOperationOutcome::Cancelled, DiagnosticText::new("A normal restart/save prompt was cancelled; no application restart requested")),
                futures::future::Either::Right(_) => (DoctorRepairOperationOutcome::Cancelled, DiagnosticText::new("Doctor waiting was cancelled; no application restart requested; dismiss any outstanding normal save prompt")),
                futures::future::Either::Left((Err(error), _)) => (DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("Normal restart preparation failed: {error}"))),
            },
            Err(error) => (DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("Restart authority/activity changed before normal preparation: {error}"))),
        };
        review.operation_finished(outcome, detail)?;
        if review.phase() == DoctorRepairPhase::Rechecking {
            let detail = review.operation_detail().map_or("Restart was not observed", DiagnosticText::as_str);
            review.recheck_unavailable(DiagnosticText::new(format!("{detail}; fresh checks in reopened Made are required; no replay")))?;
        }
        review.persist_outcome(product, &history, None, now).await?;
        if prepared.as_ref().is_some_and(|prepared|prepared.accepted) {
            let request = cx.update(|cx| {
                review.action().revalidate(cx)?;
                let binding = project_repair_binding(&project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?, directory, generation, cx)?;
                ensure!(&binding == review.binding(), "Restart scope changed during receipt persistence");
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Restart cancelled before application shutdown");
                cx.restart();
                Ok::<_,anyhow::Error>(())
            });
            if let Err(error) = request {
                review.operation_detail = Some(DiagnosticText::new(format!("No application restart requested: {error}; normal save preparation may have been reached")));
                if cancelled.is_closed() || !cancelled.is_empty() {
                    review.phase = DoctorRepairPhase::Cancelled;
                    review.operation = Some(DoctorRepairOperationOutcome::Cancelled);
                }
                review.persist_outcome(product, &history, None, now).await?;
            }
        }
        Ok(review)
    })
}

#[derive(Clone)]
pub struct DoctorRemoteRestartAction {
    client: gpui::Entity<remote::RemoteClient>,
    server: rpc::proto::DoctorRemoteServerResponse,
    workspaces: workspace::ApplicationRestartObservation,
    requesting_item: Option<gpui::EntityId>,
}

impl DoctorRemoteRestartAction {
    pub(crate) fn requesting_item(&self) -> Option<gpui::EntityId> {
        self.requesting_item
    }
    pub fn server_instance(&self) -> &str {
        &self.server.instance
    }
    pub fn server_process_id(&self) -> u32 {
        self.server.process_id
    }
    pub fn affected_workspaces(&self) -> &[(gpui::EntityId, gpui::EntityId)] {
        &self.workspaces.workspaces
    }
}

impl DoctorRepairReview<DoctorRemoteRestartAction> {
    pub fn note_project_reopening(&mut self, detail: DiagnosticText) -> Result<()> {
        ensure!(
            self.phase == DoctorRepairPhase::Incomplete,
            "Only an observed incomplete restart can record reopening"
        );
        self.operation_detail = Some(detail);
        Ok(())
    }
}

fn remote_restart_workspaces(
    project: &gpui::Entity<project::Project>,
    after_shutdown: bool,
    requesting_item: Option<gpui::EntityId>,
    cx: &gpui::App,
) -> Result<workspace::ApplicationRestartObservation> {
    let remote = project
        .read(cx)
        .remote_client()
        .ok_or_else(|| anyhow::anyhow!("No owned remote connection"))?;
    project
        .read(cx)
        .doctor_remote_restart_idle(after_shutdown, cx)?;
    ensure!(
        project.read(cx).visible_worktrees(cx).all(|worktree| {
            project
                .read(cx)
                .doctor_observation(worktree.read(cx).id(), cx)
                .trusted
                == Some(true)
        }),
        "Every affected remote worktree must remain trusted before server restart"
    );
    let mut observation = workspace::ApplicationRestartObservation {
        workspaces: Vec::new(),
        dirty_items: Vec::new(),
        blockers: Vec::new(),
    };
    for window in cx
        .windows()
        .into_iter()
        .filter_map(|window| window.downcast::<workspace::MultiWorkspace>())
    {
        for affected in window.read(cx)?.workspaces() {
            let owner = affected.read(cx);
            if owner
                .project()
                .read(cx)
                .remote_client()
                .is_none_or(|client| client.entity_id() != remote.entity_id())
            {
                continue;
            }
            observation
                .workspaces
                .push((affected.entity_id(), owner.project().entity_id()));
            if let Some(blocker) = owner.restart_setup_blocker() {
                observation.blockers.push(blocker);
            }
            for item in owner.items(cx) {
                if item.is_dirty(cx) {
                    observation.dirty_items.push(item.item_id());
                }
                if Some(item.item_id()) != requesting_item {
                    if let Some(blocker) = item.restart_blocker(cx) {
                        observation.blockers.push(blocker);
                    }
                }
            }
            for dock in owner.all_docks() {
                observation
                    .blockers
                    .extend(dock.read(cx).restart_blockers(cx));
            }
        }
    }
    ensure!(
        observation.workspaces.len() == 1
            && observation
                .workspaces
                .first()
                .is_some_and(|(_, identity)| *identity == project.entity_id()),
        "Remote server must have exactly one current owned workspace; shared or unobservable use is blocked"
    );
    ensure!(
        observation.dirty_items.is_empty() && observation.blockers.is_empty(),
        "Resolve dirty files and active/unknown remote work before restart review"
    );
    #[cfg(feature = "agentic")]
    ensure!(
        !agent::recipes::RecipeStore::try_global(cx)
            .is_some_and(|owner| owner.read(cx).has_active_work(cx)),
        "Finish native-agent/recipe/schedule work before remote restart"
    );
    Ok(observation)
}

pub fn prepare_remote_restart(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorRemoteRestartAction>>> {
    prepare_remote_restart_for_item(project, directory, generation, observed_at, None, cx)
}

pub(crate) fn prepare_remote_restart_for_item(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    observed_at: u64,
    requesting_item: Option<gpui::EntityId>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorRemoteRestartAction>>> {
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let (binding, client, workspaces, status) = cx.update(|cx| {
            let project = project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?;
            let binding = project_repair_binding(&project, directory.clone(), generation, cx)?;
            let workspaces = remote_restart_workspaces(&project, false, requesting_item, cx)?;
            let client = project.read(cx).remote_client().ok_or_else(||anyhow::anyhow!("Owned server unavailable"))?;
            let status = client.read(cx).doctor_restart_status(cx);
            Ok::<_,anyhow::Error>((binding, client, workspaces, status))
        })?;
        let server = status.await?;
        ensure!(server.exclusive && server.idle, "Remote restart blocked: {}", server.reason);
        cx.update(|cx| {
            let project = project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?;
            ensure!(project_repair_binding(&project, directory, generation, cx)? == binding && remote_restart_workspaces(&project, false, requesting_item, cx)? == workspaces, "Remote scope changed during observation");
            DoctorRepairReview::new(DoctorRemoteRestartAction {client, server, workspaces, requesting_item}, binding, DoctorCheck::new("made.remote.restart", "Owned remote development-server restart", CheckOutcome::Untested, "Problem recovery requires fresh relevant observations", observed_at), DiagnosticText::new("Gracefully shut down only the current exclusive idle development server, observe its instance/PID stop and start a different ready instance through the same installed connection owner. The configured remote shell may execute its existing startup files. Reopen with a fresh Project; no old request replay, server download, credential/provider switch, force killing or OS/service restart. Cancellation after shutdown may leave a stopped host or idle replacement; the original problem remains unverified until relevant fresh checks."))
        })
    })
}

pub fn launch_remote_restart(
    mut review: DoctorRepairReview<DoctorRemoteRestartAction>,
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    generation: u64,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<
    Result<(
        DoctorRepairReview<DoctorRemoteRestartAction>,
        Option<gpui::Entity<remote::RemoteClient>>,
    )>,
> {
    use futures::FutureExt as _;
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let binding = cx.update(|cx| {
            let project = project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?;
            ensure!(remote_restart_workspaces(&project, false, review.action().requesting_item, cx)? == review.action().workspaces, "Affected remote workspace changed");
            project_repair_binding(&project, directory.clone(), generation, cx)
        })?;
        review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
        let action = review.action().clone();
        let binding = binding.clone();
        let project = project.clone();
        let operation = cx.update(|cx| action.client.update(cx, |client, cx| {
            client.doctor_restart(action.server, async move { match cancelled.recv().await { Ok(()) | Err(_) => {} } }.boxed(), move |cx, after_shutdown| {
                let project = project.upgrade().ok_or_else(||anyhow::anyhow!("Doctor project closed"))?;
                if after_shutdown {
                    let observed = project.read(cx).doctor_observation(directory.worktree_id, cx);
                    ensure!(observed.worktree_available && observed.trusted == Some(true) && !observed.isolated_recipe_test && project.read(cx).remote_client().is_some_and(|client|Some(client.entity_id()) == binding.connection) && cx.global::<settings::SettingsStore>().observed_revision() == Some(binding.input_revision), "Remote authority changed during shutdown");
                } else { ensure!(project_repair_binding(&project, directory.clone(), generation, cx)? == binding, "Remote authority changed before dispatch"); }
                ensure!(remote_restart_workspaces(&project, after_shutdown, action.requesting_item, cx)? == action.workspaces, "Remote workspace/activity changed during restart");
                Ok(())
            }, cx)
        }));
        let (outcome, detail, replacement) = match operation.await {
            Ok(result) => {
                let outcome = match result.outcome {
                    remote::remote_client::DoctorRemoteRestartOutcome::Ready => DoctorRepairOperationOutcome::Completed,
                    remote::remote_client::DoctorRemoteRestartOutcome::Incomplete => DoctorRepairOperationOutcome::Failed,
                    remote::remote_client::DoctorRemoteRestartOutcome::Cancelled => DoctorRepairOperationOutcome::Cancelled,
                };
                (outcome, DiagnosticText::new(result.detail), result.replacement)
            },
            Err(error) => (DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("Remote restart incomplete: {error}; no verified repair or automatic replay")), None),
        };
        review.operation_finished(outcome, detail)?;
        if review.phase() == DoctorRepairPhase::Rechecking { review.recheck_unavailable(DiagnosticText::new("A fresh Project and relevant new check are required; server replacement is not proof of problem recovery"))?; }
        // The native view still owns fresh Project reopening. Leave its durable
        // receipt active until that observation finishes, so a crash is interrupted
        // rather than terminal history being rewritten or interpreted as replay.
        if replacement.is_none() || review.action().requesting_item.is_none() {
            review.persist_outcome(product, &history, None, now).await?;
        }
        Ok((review, replacement))
    })
}

pub fn prepare_project_repair(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    action: DoctorProjectRepairAction,
    generation: u64,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorRepairReview<DoctorProjectRepairAction>> {
    let binding = project_repair_binding(project, directory, generation, cx)?;
    let owner = project.read(cx);
    let observation = owner.doctor_observation(binding.directory.worktree_id, cx);
    let (server_id, name) = match &action {
        DoctorProjectRepairAction::RestartLanguageServer { server, name } => {
            (*server, name.clone())
        }
        #[cfg(feature = "jvm-tools")]
        DoctorProjectRepairAction::RefreshJavaProject { server, .. } => {
            (*server, lsp::LanguageServerName::new_static("jdtls"))
        }
        #[cfg(feature = "jvm-tools")]
        DoctorProjectRepairAction::RefreshKotlinProject { server, .. } => {
            (*server, lsp::LanguageServerName::new_static("kotlin-lsp"))
        }
    };
    ensure!(
        observation
            .language_servers
            .iter()
            .any(|server| server.id == server_id && server.name == name),
        "The selected language server is unavailable on this worktree"
    );
    ensure!(
        observation
            .language_servers
            .iter()
            .filter(|server| server.name == name)
            .count()
            == 1,
        "Multiple server instances prevent an unambiguous repair receipt; use existing language-server controls"
    );
    let before = project_repair_observation(
        owner,
        &action,
        binding.directory.worktree_id,
        false,
        observed_at,
        cx,
    )?;
    let effects = DiagnosticText::new(match &action {
        DoctorProjectRepairAction::RestartLanguageServer { .. } => {
            "Restart only the selected existing language-server instance through Project. It may affect buffers served by that instance and may retry setup/downloads under existing permissions. No provider choice, settings, project files, pins or locks are changed by Doctor. Review existing server configuration before approval. Once accepted by the owner, restart cannot be rolled back; cancellation stops Doctor waiting/rechecking. A running replacement proves server readiness only, not build or application correctness."
        }
        #[cfg(feature = "jvm-tools")]
        _ => {
            "Refresh only the selected existing JVM provider's project metadata through LspStore. The configured provider may inspect project/build metadata under its existing permissions; Doctor does not run wrappers, add extensions or change provider settings. Once accepted, cancellation stops Doctor waiting; it cannot undo the provider's query. Fresh complete metadata proves import observation only, not build/test/application correctness."
        }
    });
    DoctorRepairReview::new(action, binding, before, effects)
}

fn project_repair_observation(
    owner: &project::Project,
    action: &DoctorProjectRepairAction,
    worktree: settings::WorktreeId,
    after_dispatch: bool,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<DoctorCheck> {
    let observation = owner.doctor_observation(worktree, cx);
    match action {
        DoctorProjectRepairAction::RestartLanguageServer { server, name } => {
            let matching = observation
                .language_servers
                .iter()
                .filter(|candidate| candidate.name == *name)
                .collect::<Vec<_>>();
            let replacement = matching.first().filter(|candidate| {
                matching.len() == 1 && (!after_dispatch || candidate.id != *server)
            });
            let outcome = match replacement.map(|candidate| candidate.running) {
                Some(Some(true)) => CheckOutcome::Ready,
                Some(Some(false)) => CheckOutcome::Warning,
                _ => CheckOutcome::Untested,
            };
            Ok(DoctorCheck::new(
                &format!("project.lsp.{}", server.0),
                "Selected language-server readiness",
                outcome,
                &format!(
                    "Provider {name}; {} scoped instances; an identified running replacement {}. Server availability does not verify builds or application behavior.",
                    matching.len(),
                    replacement.is_some_and(|candidate| candidate.running == Some(true))
                ),
                observed_at,
            ))
        }
        #[cfg(feature = "jvm-tools")]
        DoctorProjectRepairAction::RefreshJavaProject { server, generation } => {
            let store = owner.lsp_store();
            let store = store.read(cx);
            let snapshot = store.jdtls_project_snapshot(worktree);
            Ok(jvm_refresh_check(
                "project.jvm.java.import",
                *server,
                *generation,
                snapshot.map(|(snapshot, current)| {
                    (
                        snapshot.server_id,
                        snapshot.generation,
                        current,
                        snapshot.partial,
                        snapshot.projects.len(),
                    )
                }),
                store.jdtls_project_query_failed(worktree),
                after_dispatch,
                observed_at,
            ))
        }
        #[cfg(feature = "jvm-tools")]
        DoctorProjectRepairAction::RefreshKotlinProject { server, generation } => {
            let store = owner.lsp_store();
            let store = store.read(cx);
            let snapshot = store.kotlin_workspace_snapshot(worktree);
            Ok(jvm_refresh_check(
                "project.jvm.kotlin.import",
                *server,
                *generation,
                snapshot.map(|(snapshot, current)| {
                    (
                        snapshot.server_id,
                        snapshot.generation,
                        current,
                        snapshot.partial,
                        snapshot.projects.len(),
                    )
                }),
                store.kotlin_workspace_query_failed(worktree)
                    || store.kotlin_workspace_import_failed(worktree),
                after_dispatch,
                observed_at,
            ))
        }
    }
}

#[cfg(feature = "jvm-tools")]
fn jvm_refresh_check(
    id: &str,
    expected_server: lsp::LanguageServerId,
    previous: Option<u64>,
    snapshot: Option<(lsp::LanguageServerId, u64, bool, bool, usize)>,
    failed: bool,
    after_dispatch: bool,
    observed_at: u64,
) -> DoctorCheck {
    let fresh = snapshot.is_some_and(|(server, generation, current, partial, count)| {
        server == expected_server
            && current
            && !partial
            && count > 0
            && (!after_dispatch || previous.is_none_or(|previous| generation > previous))
    });
    DoctorCheck::new(
        id,
        "Selected JVM import observation",
        if failed {
            CheckOutcome::Failed
        } else if fresh {
            CheckOutcome::Ready
        } else {
            CheckOutcome::Untested
        },
        &format!(
            "Current complete nonempty selected-provider metadata {}; failed query/import {}. A prior, partial or unrelated provider snapshot cannot verify this refresh. No build/test success is inferred.",
            fresh, failed
        ),
        observed_at,
    )
}

pub fn launch_project_repair(
    mut review: DoctorRepairReview<DoctorProjectRepairAction>,
    project: gpui::Entity<project::Project>,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancelled: async_channel::Receiver<()>,
    cx: &mut gpui::App,
) -> gpui::Task<Result<DoctorRepairReview<DoctorProjectRepairAction>>> {
    let directory = review.binding.directory.clone();
    let generation = review.binding.generation;
    let project = project.downgrade();
    let executor = cx.background_executor().clone();
    cx.spawn(async move |cx| {
        let started = executor.now();
        let current = cx.update(|cx| {
            let project = project.upgrade().ok_or_else(|| anyhow::anyhow!("Repair project closed"))?;
            project_repair_binding(&project, directory.clone(), generation, cx)
        })?;
        review.record_before_execution(&current, report, product, &history, now, &cancelled).await?;
        let operation: Result<()> = async {
            cx.update(|cx| {
                let project = project.upgrade().ok_or_else(|| anyhow::anyhow!("Repair project closed"))?;
                let current = project_repair_binding(&project, directory.clone(), generation, cx)?;
                review.revalidate(&current)?;
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Repair cancelled before dispatch");
                project.update(cx, |owner, cx| {
                let observation = owner.doctor_observation(directory.worktree_id, cx);
                match review.action() {
                    DoctorProjectRepairAction::RestartLanguageServer { server, name } => {
                        ensure!(observation.language_servers.iter().any(|current| current.id == *server && current.name == *name), "Reviewed server changed before dispatch");
                        let buffers = owner.opened_buffers(cx).into_iter().filter(|buffer| {
                            buffer.read(cx).file().and_then(|file| project::File::from_dyn(Some(file))).is_some_and(|file| file.worktree_id(cx) == directory.worktree_id)
                        }).collect::<Vec<_>>();
                        ensure!(!buffers.is_empty(), "No scoped open buffers can restart the selected server; use existing server controls");
                        owner.restart_language_servers_for_buffers(buffers, collections::HashSet::from_iter([lsp::LanguageServerSelector::Id(*server)]), true, cx);
                    }
                    #[cfg(feature = "jvm-tools")]
                    DoctorProjectRepairAction::RefreshJavaProject { server, generation: previous } => {
                        let store = owner.lsp_store();
                        store.update(cx, |store, cx| {
                            ensure!(store.jdtls_project_snapshot(directory.worktree_id).map(|(snapshot, _)| snapshot.generation) == *previous, "JVM import inputs changed after review");
                            ensure!(observation.language_servers.iter().any(|current| current.id == *server), "Selected JDTLS server changed");
                            store.refresh_jdtls_project(directory.worktree_id, cx);
                            anyhow::Ok(())
                        })?;
                    }
                    #[cfg(feature = "jvm-tools")]
                    DoctorProjectRepairAction::RefreshKotlinProject { server, generation: previous } => {
                        let store = owner.lsp_store();
                        store.update(cx, |store, cx| {
                            ensure!(store.kotlin_workspace_snapshot(directory.worktree_id).map(|(snapshot, _)| snapshot.generation) == *previous, "Kotlin import inputs changed after review");
                            ensure!(observation.language_servers.iter().any(|current| current.id == *server), "Selected Kotlin server changed");
                            store.refresh_kotlin_workspace(directory.worktree_id, cx);
                            anyhow::Ok(())
                        })?;
                    }
                }
                anyhow::Ok(())
                })
            })?;
            Ok(())
        }.await;
        if let Err(error) = operation {
            review.operation_finished(if cancelled.is_closed() || !cancelled.is_empty() { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed }, DiagnosticText::new(format!("Owner dispatch did not complete: {error}")))?;
            if review.phase() == DoctorRepairPhase::Rechecking { review.recheck_unavailable(DiagnosticText::new("No approved operation completed; readiness remains incomplete"))?; }
        } else {
            review.operation_finished(DoctorRepairOperationOutcome::Completed, DiagnosticText::new("Existing owner accepted the action; cancellation cannot undo accepted owner work. Revalidation is required."))?;
            let deadline = executor.now() + std::time::Duration::from_secs(30);
            loop {
                if cancelled.is_closed() || !cancelled.is_empty() {
                    review.phase = DoctorRepairPhase::Cancelled;
                    review.operation_detail = Some(DiagnosticText::new("Doctor stopped waiting; accepted owner work may still finish. No fix is claimed."));
                    break;
                }
                let observation = cx.update(|cx| {
                    let project = project.upgrade().ok_or_else(|| anyhow::anyhow!("Repair project closed"))?;
                    let current = project_repair_binding(&project, directory.clone(), generation, cx)?;
                    ensure!(current == review.binding, "Repair inputs or authority changed during revalidation");
                    let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                    let after = project_repair_observation(project.read(cx), review.action(), directory.worktree_id, true, observed_at, cx)?;
                    anyhow::Ok((current, after))
                });
                match observation {
                    Ok((current, after)) if matches!(after.outcome, CheckOutcome::Ready | CheckOutcome::Failed | CheckOutcome::Blocked) => {
                        review.observe_recheck(&current, after)?;
                        break;
                    }
                    Err(error) => { review.recheck_unavailable(DiagnosticText::new(format!("Revalidation unavailable: {error}")))?; break; }
                    _ if executor.now() >= deadline => { review.recheck_unavailable(DiagnosticText::new("No fresh relevant owner result was reached within 30 seconds; accepted owner work may still finish"))?; break; }
                    _ => executor.timer(std::time::Duration::from_millis(50)).await,
                }
            }
        }
        let finished_at = now.saturating_add(executor.now().duration_since(started).as_secs());
        review.persist_outcome(product, &history, None, finished_at).await?;
        Ok(review)
    })
}

pub const MAX_HISTORY_RECORDS: usize = 100;
pub const MAX_HISTORY_BYTES: usize = 5 * 1024 * 1024;
pub const HISTORY_RETENTION_SECONDS: u64 = 30 * 24 * 60 * 60;
const HISTORY_NAMESPACE: &str = "made.doctor.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum DoctorProduct {
    Made,
    Copper,
    Groovy,
}

impl DoctorProduct {
    pub fn current() -> Self {
        match product_flavor::ID {
            "rust" => Self::Copper,
            "jvm" => Self::Groovy,
            _ => Self::Made,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Made => "Made",
            Self::Copper => "Copper",
            Self::Groovy => "Groovy",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReceiptOutcome {
    Running,
    Completed,
    Failed,
    Cancelled,
    Interrupted,
    Incomplete,
    AwaitingUserAction,
}

impl ReceiptOutcome {
    fn is_active(self) -> bool {
        self == Self::Running
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorHistoryRecord {
    pub id: uuid::Uuid,
    pub created_at: u64,
    pub updated_at: u64,
    pub report: DoctorReport,
    pub action: Option<DiagnosticText>,
    pub outcome: ReceiptOutcome,
    pub session_id: Option<DiagnosticText>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    admission_key: Option<String>,
}

impl DoctorHistoryRecord {
    pub fn new(
        report: DoctorReport,
        action: Option<DiagnosticText>,
        outcome: ReceiptOutcome,
        now: u64,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            created_at: now,
            updated_at: now,
            report,
            action,
            outcome,
            session_id: None,
            admission_key: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DoctorHistorySnapshot {
    version: u32,
    records: Vec<DoctorHistoryRecord>,
}

impl Default for DoctorHistorySnapshot {
    fn default() -> Self {
        Self {
            version: 1,
            records: Vec::new(),
        }
    }
}

impl DoctorHistorySnapshot {
    fn validate(&self, product: DoctorProduct) -> Result<()> {
        ensure!(self.version == 1, "Doctor history version is unsupported");
        ensure!(
            self.records.len() <= MAX_HISTORY_RECORDS,
            "Doctor history record limit exceeded"
        );
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_HISTORY_BYTES,
            "Doctor history byte limit exceeded"
        );
        for (index, record) in self.records.iter().enumerate() {
            ensure!(
                !self
                    .records
                    .iter()
                    .take(index)
                    .any(|other| other.id == record.id),
                "Doctor history identity is duplicated"
            );
            ensure!(
                record.updated_at >= record.created_at,
                "Doctor history timestamp is invalid"
            );
            ensure!(
                record.report.scope.product.as_str() == product.name(),
                "Doctor history product scope does not match"
            );
            if let Some(session) = &record.session_id {
                ensure!(
                    session.as_str().len() <= 128 && !session.as_str().is_empty(),
                    "Doctor session identity is invalid"
                );
            }
            if let Some(key) = &record.admission_key {
                ensure!(
                    key.len() == 64
                        && key
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')),
                    "Doctor admission identity is invalid"
                );
                ensure!(
                    !record.outcome.is_active()
                        || !self
                            .records
                            .iter()
                            .take(index)
                            .any(|other| other.outcome.is_active()
                                && other.admission_key.as_ref() == Some(key)),
                    "Active Doctor admission identity is duplicated"
                );
            }
            record.report.validate()?;
        }
        Ok(())
    }

    fn prune(&mut self, now: u64) -> Result<()> {
        let cutoff = now.saturating_sub(HISTORY_RETENTION_SECONDS);
        self.records
            .retain(|record| record.outcome.is_active() || record.updated_at >= cutoff);
        while self.records.len() > MAX_HISTORY_RECORDS
            || serde_json::to_vec(self)?.len() > MAX_HISTORY_BYTES
        {
            let candidate = self
                .records
                .iter()
                .enumerate()
                .filter(|(_, record)| !record.outcome.is_active())
                .min_by_key(|(_, record)| (record.updated_at, record.created_at))
                .map(|(index, _)| index);
            let index = candidate.ok_or_else(|| {
                anyhow::anyhow!(
                    "Active Doctor receipts prevent history admission; no action was started"
                )
            })?;
            self.records.remove(index);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DoctorHistoryStore {
    database: std::sync::Arc<db::kvp::KeyValueStore>,
    snapshots: std::sync::Arc<
        async_lock::Mutex<std::collections::BTreeMap<DoctorProduct, DoctorHistorySnapshot>>,
    >,
}

impl gpui::Global for DoctorHistoryStore {}

impl DoctorHistoryStore {
    pub fn for_app(cx: &mut gpui::App) -> Result<Self> {
        if let Some(store) = cx.try_global::<Self>() {
            return Ok(store.clone());
        }
        let database = cx.try_global::<db::AppDatabase>().ok_or_else(|| {
            anyhow::anyhow!("Application database is unavailable; Doctor repairs are blocked")
        })?;
        let store = Self::from_database(database);
        cx.set_global(store.clone());
        Ok(store)
    }

    #[cfg(test)]
    pub(crate) fn for_test_database(database: &db::AppDatabase) -> Self {
        Self::from_database(database)
    }

    fn from_database(database: &db::AppDatabase) -> Self {
        Self {
            database: std::sync::Arc::new(db::kvp::KeyValueStore::from_app_db(database)),
            snapshots: std::sync::Arc::new(async_lock::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
        }
    }

    async fn update<T>(
        &self,
        product: DoctorProduct,
        now: u64,
        update: impl FnOnce(&mut DoctorHistorySnapshot) -> Result<T>,
    ) -> Result<T> {
        let mut cache = self.snapshots.lock().await;
        // AppDB writes can commit after their awaiting caller is dropped. Drain
        // earlier queued writes and reread durable state before applying a delta.
        self.database.write(|_| ()).await;
        let previous = match self
            .database
            .scoped(HISTORY_NAMESPACE)
            .read(product.name())?
        {
            Some(value) => {
                ensure!(
                    value.len() <= MAX_HISTORY_BYTES,
                    "Persisted Doctor history is oversized; it was not replaced"
                );
                let snapshot: DoctorHistorySnapshot = serde_json::from_str(&value)?;
                snapshot.validate(product)?;
                snapshot
            }
            None => DoctorHistorySnapshot::default(),
        };
        let mut next = previous.clone();
        if !cache.contains_key(&product) {
            for record in &mut next.records {
                if record.outcome.is_active() {
                    record.outcome = ReceiptOutcome::Interrupted;
                    record.updated_at = now.max(record.created_at);
                }
            }
        }
        let result = update(&mut next)?;
        next.prune(now)?;
        next.validate(product)?;
        let encoded = serde_json::to_string(&next)?;
        if encoded != serde_json::to_string(&previous)? {
            self.database
                .scoped(HISTORY_NAMESPACE)
                .write(product.name().to_owned(), encoded)
                .await?;
        }
        cache.insert(product, next);
        Ok(result)
    }

    pub async fn records(
        &self,
        product: DoctorProduct,
        now: u64,
    ) -> Result<Vec<DoctorHistoryRecord>> {
        self.update(product, now, |_| Ok(())).await?;
        let cache = self.snapshots.lock().await;
        Ok(cache
            .get(&product)
            .ok_or_else(|| anyhow::anyhow!("Doctor history was not loaded"))?
            .records
            .clone())
    }

    pub async fn record(
        &self,
        product: DoctorProduct,
        record: DoctorHistoryRecord,
        now: u64,
    ) -> Result<uuid::Uuid> {
        let id = record.id;
        self.update(product, now, move |snapshot| {
            ensure!(
                !snapshot.records.iter().any(|other| other.id == id),
                "Doctor receipt already exists"
            );
            if record.outcome.is_active() && let Some(key) = &record.admission_key {
                ensure!(!snapshot.records.iter().any(|other| other.outcome.is_active() && other.admission_key.as_ref() == Some(key)), "An equivalent Doctor action is already active; no duplicate action was admitted");
            }
            snapshot.records.push(record);
            Ok(id)
        })
        .await
    }

    async fn link_native_session(
        &self,
        product: DoctorProduct,
        receipt: uuid::Uuid,
        session: &str,
        now: u64,
    ) -> Result<()> {
        ensure!(
            !session.is_empty() && session.len() <= 4096 && !session.chars().any(char::is_control),
            "Native session identity is invalid"
        );
        let session = DiagnosticText::new(session);
        self.update(product, now, |snapshot| {
            let record = snapshot
                .records
                .iter_mut()
                .find(|record| record.id == receipt)
                .ok_or_else(|| anyhow::anyhow!("Native repair receipt is unavailable"))?;
            ensure!(
                record.outcome.is_active(),
                "Native repair receipt is no longer active"
            );
            ensure!(
                record
                    .session_id
                    .as_ref()
                    .is_none_or(|previous| previous == &session),
                "Native repair receipt already belongs to another session"
            );
            record.session_id = Some(session);
            record.updated_at = now.max(record.updated_at);
            Ok(())
        })
        .await
    }

    async fn finish_unobserved_native_session(
        &self,
        product: DoctorProduct,
        receipt: uuid::Uuid,
        result: Result<DoctorNativeRepairResult>,
        now: u64,
    ) -> Result<()> {
        self.update(product, now, |snapshot| {
            let record = snapshot.records.iter_mut().find(|record| record.id == receipt)
                .ok_or_else(|| anyhow::anyhow!("Native repair receipt is unavailable"))?;
            if !record.outcome.is_active() {
                return Ok(());
            }
            let (outcome, detail) = match result {
                Ok(result) => {
                    let outcome = match result.outcome {
                        DoctorRepairOperationOutcome::Cancelled => ReceiptOutcome::Cancelled,
                        DoctorRepairOperationOutcome::AwaitingUserAction => ReceiptOutcome::AwaitingUserAction,
                        DoctorRepairOperationOutcome::Failed | DoctorRepairOperationOutcome::Completed => ReceiptOutcome::Incomplete,
                    };
                    (outcome, result.detail)
                }
                Err(error) => (ReceiptOutcome::Incomplete, DiagnosticText::new(error.to_string())),
            };
            let mut report = DoctorReport::new(record.report.scope.clone(), record.report.generation, now);
            report.push(DoctorCheck::new("doctor.native.repair.unobserved", "Native repair view closed", CheckOutcome::Untested,
                &format!("{}. Inspect the retained native session and reached changes; no independent verification was observed and no action will be replayed.", detail.as_str()), now))?;
            record.report = report;
            record.outcome = outcome;
            record.updated_at = now.max(record.updated_at);
            Ok(())
        }).await
    }

    pub async fn finish(
        &self,
        product: DoctorProduct,
        id: uuid::Uuid,
        outcome: ReceiptOutcome,
        report: DoctorReport,
        session_id: Option<DiagnosticText>,
        now: u64,
    ) -> Result<()> {
        ensure!(
            !outcome.is_active(),
            "Doctor completion cannot remain running"
        );
        self.update(product, now, move |snapshot| {
            let record = snapshot
                .records
                .iter_mut()
                .find(|record| record.id == id)
                .ok_or_else(|| anyhow::anyhow!("Doctor receipt is unavailable"))?;
            ensure!(
                record.outcome.is_active(),
                "Doctor receipt is already terminal and cannot be replayed"
            );
            ensure!(
                record.report.scope == report.scope,
                "Doctor completion scope changed"
            );
            ensure!(
                now >= record.updated_at,
                "Doctor completion timestamp is stale"
            );
            record.outcome = outcome;
            record.updated_at = now;
            record.report = report;
            record.session_id = session_id;
            Ok(())
        })
        .await
    }

    pub(crate) async fn delete_reviewed(
        &self,
        product: DoctorProduct,
        expected: &DoctorHistoryRecord,
        now: u64,
    ) -> Result<()> {
        self.update(product, now, |snapshot| {
            let index = snapshot
                .records
                .iter()
                .position(|record| record.id == expected.id)
                .ok_or_else(|| anyhow::anyhow!("Doctor history record is unavailable"))?;
            let current = snapshot
                .records
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("Doctor history record is unavailable"))?;
            ensure!(
                !current.outcome.is_active(),
                "An active Doctor receipt cannot be deleted"
            );
            ensure!(
                serde_json::to_string(current)? == serde_json::to_string(expected)?,
                "Doctor history record changed after review"
            );
            snapshot.records.remove(index);
            Ok(())
        })
        .await
    }

    pub async fn delete(&self, product: DoctorProduct, id: uuid::Uuid, now: u64) -> Result<()> {
        self.update(product, now, |snapshot| {
            let index = snapshot
                .records
                .iter()
                .position(|record| record.id == id)
                .ok_or_else(|| anyhow::anyhow!("Doctor history record is unavailable"))?;
            ensure!(
                !snapshot
                    .records
                    .get(index)
                    .is_some_and(|record| record.outcome.is_active()),
                "An active Doctor receipt cannot be deleted"
            );
            snapshot.records.remove(index);
            Ok(())
        })
        .await
    }

    pub(crate) async fn review_clear(&self, product: DoctorProduct) -> Result<String> {
        let cache = self.snapshots.lock().await;
        self.database.write(|_| ()).await;
        let value = self
            .database
            .scoped(HISTORY_NAMESPACE)
            .read(product.name())?;
        Self::ensure_clearable(product, value.as_deref(), cache.get(&product))?;
        Ok(Self::history_fingerprint(value.as_deref()))
    }

    fn history_fingerprint(value: Option<&str>) -> String {
        use sha2::{Digest as _, Sha256};
        format!("{:x}", Sha256::digest(value.unwrap_or_default().as_bytes()))
    }

    fn ensure_clearable(
        product: DoctorProduct,
        value: Option<&str>,
        cached: Option<&DoctorHistorySnapshot>,
    ) -> Result<()> {
        ensure!(
            !cached.is_some_and(|snapshot| snapshot
                .records
                .iter()
                .any(|record| record.outcome.is_active())),
            "Stop active Doctor actions before clearing history"
        );
        if let Some(value) = value.filter(|value| value.len() <= MAX_HISTORY_BYTES)
            && let Ok(snapshot) = serde_json::from_str::<DoctorHistorySnapshot>(value)
            && snapshot.validate(product).is_ok()
        {
            ensure!(
                !snapshot
                    .records
                    .iter()
                    .any(|record| record.outcome.is_active()),
                "Stop active Doctor actions before clearing history"
            );
        }
        Ok(())
    }

    pub(crate) async fn clear_reviewed(
        &self,
        product: DoctorProduct,
        expected: &str,
    ) -> Result<()> {
        let mut cache = self.snapshots.lock().await;
        self.database.write(|_| ()).await;
        let value = self
            .database
            .scoped(HISTORY_NAMESPACE)
            .read(product.name())?;
        ensure!(
            Self::history_fingerprint(value.as_deref()) == expected,
            "Doctor history changed after review; review clearing again"
        );
        Self::ensure_clearable(product, value.as_deref(), cache.get(&product))?;
        let empty = DoctorHistorySnapshot::default();
        self.database
            .scoped(HISTORY_NAMESPACE)
            .write(product.name().to_owned(), serde_json::to_string(&empty)?)
            .await?;
        cache.insert(product, empty);
        Ok(())
    }

    pub async fn clear(&self, product: DoctorProduct, now: u64) -> Result<()> {
        self.update(product, now, |snapshot| {
            ensure!(
                !snapshot
                    .records
                    .iter()
                    .any(|record| record.outcome.is_active()),
                "Stop active Doctor actions before clearing history"
            );
            snapshot.records.clear();
            Ok(())
        })
        .await
    }
}

#[cfg(feature = "rust-tools")]
pub fn collect_copper_checks(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<Vec<DoctorCheck>>> {
    use gpui::AppContext as _;
    use project::DoctorExecutionHost;
    let owner = project.read(cx);
    let observation = owner.doctor_observation(directory.worktree_id, cx);
    let mut checks = copper_owner_checks(owner, directory.worktree_id, observed_at, cx);
    if !directory.path.is_empty() {
        return gpui::Task::ready(Err(anyhow::anyhow!(
            "Doctor requires a selected worktree root"
        )));
    }
    if observation.host != DoctorExecutionHost::Local {
        checks.push(DoctorCheck::new("rust.pin", "Rust toolchain declaration", CheckOutcome::Unsupported,
            "Direct pin inspection is available only on the local host; remote facts require an authoritative Cargo snapshot.", observed_at));
        return gpui::Task::ready(Ok(checks));
    }
    let Some(worktree) = owner.worktree_for_id(directory.worktree_id, cx) else {
        return gpui::Task::ready(Err(anyhow::anyhow!("Selected worktree is unavailable")));
    };
    let selected_toolchain =
        owner.active_toolchain(directory.clone(), language::LanguageName::new("Rust"), cx);
    let snapshot = worktree.read(cx).snapshot();
    let filesystem = owner.fs().clone();
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let selected_toolchain = selected_toolchain.await;
        checks.push(DoctorCheck::new("rust.selection", "Selected Rust toolchain", if selected_toolchain.is_some() { CheckOutcome::Ready } else { CheckOutcome::Untested },
            &selected_toolchain.map(|toolchain| format!("The existing toolchain owner selected {}. Installed components and compatibility still require their relevant checks.", toolchain.name)).unwrap_or_else(|| "No explicit Rust selection is recorded by the toolchain owner. A pin or manager default may determine the effective compiler.".to_string()), observed_at));
        let root = snapshot.root_entry().ok_or_else(|| anyhow::anyhow!("Selected worktree root is unavailable"))?;
        let declaration = ["rust-toolchain.toml", "rust-toolchain"].into_iter().find_map(|name| {
            let path = util::rel_path::RelPath::from_unix_str(name).ok()?;
            snapshot.entry_for_path(path.as_ref()).map(|entry| (path, entry.clone()))
        });
        let check = if let Some((path, entry)) = declaration {
            if !entry.is_file() || entry.is_private || entry.is_external || entry.is_fifo || entry.size > 32 * 1024 {
                DoctorCheck::new("rust.pin", "Rust toolchain declaration", CheckOutcome::Blocked,
                    "The pin file is private, outside the worktree, nonregular, or exceeds 32 KiB. No file was read.", observed_at)
            } else {
                let filesystem = filesystem.clone();
                let root_path = snapshot.abs_path().to_path_buf();
                let relative_path = path.as_std_path().to_path_buf();
                let root_inode = root.inode;
                let entry_inode = entry.inode;
                let entry_mtime = entry.mtime;
                let entry_size = entry.size;
                let source_path = project::ProjectPath { worktree_id: directory.worktree_id, path: path.into() };
                let parsed = cx.background_spawn(async move {
                    let contents = read_confined_declaration(filesystem.as_ref(), &root_path, root_inode, &relative_path, (entry_inode, entry_mtime, entry_size), 32 * 1024).await?;
                    project::cargo_workspace::parse_rust_toolchain(source_path, &contents)
                }).await;
                match parsed {
                    Ok(toolchain) => copper_declaration_check(&toolchain, observed_at),
                    Err(error) => DoctorCheck::new("rust.pin", "Rust toolchain declaration", CheckOutcome::Failed,
                        &format!("Toolchain declaration could not be inspected: {error}"), observed_at),
                }
            }
        } else {
            DoctorCheck::new("rust.pin", "Rust toolchain declaration", CheckOutcome::Untested,
                "No root pin file was observed. Ancestor configuration and installed manager defaults may determine the toolchain.", observed_at)
        };
        project.read_with(cx, |project, cx| {
            ensure!(project.doctor_observation(directory.worktree_id, cx).host == DoctorExecutionHost::Local,
                "The execution host changed during diagnosis");
            let current = project.worktree_for_id(directory.worktree_id, cx).ok_or_else(|| anyhow::anyhow!("Selected worktree closed during diagnosis"))?;
            let current_snapshot = current.read(cx).snapshot();
            ensure!(snapshot.root_entry().map(|entry| (entry.id, entry.inode)) == current_snapshot.root_entry().map(|entry| (entry.id, entry.inode)), "Selected worktree root changed during diagnosis");
            for name in ["rust-toolchain.toml", "rust-toolchain"] {
                let path = util::rel_path::RelPath::from_unix_str(name)?;
                let before = snapshot.entry_for_path(path.as_ref()).map(|entry| (entry.id, entry.inode, entry.mtime, entry.size, entry.is_private));
                let current_snapshot = current.read(cx).snapshot();
                let after = current_snapshot.entry_for_path(path.as_ref()).map(|entry| (entry.id, entry.inode, entry.mtime, entry.size, entry.is_private));
                ensure!(before == after, "Toolchain declaration changed during diagnosis; run Doctor again");
            }
            Ok::<(), anyhow::Error>(())
        })??;
        checks.push(check);
        Ok(checks)
    })
}

#[cfg(feature = "rust-tools")]
fn copper_declaration_check(
    toolchain: &project::cargo_workspace::CargoToolchainModel,
    observed_at: u64,
) -> DoctorCheck {
    DoctorCheck::new(
        "rust.pin",
        "Rust toolchain declaration",
        CheckOutcome::Ready,
        &format!(
            "Declared channel: {}; components: {}; targets: {}. This validates the declaration, not installation or effective selection.",
            toolchain.channel.as_deref().unwrap_or("unspecified"),
            toolchain.components.join(", "),
            toolchain.targets.join(", ")
        ),
        observed_at,
    )
}

pub fn collect_development(
    project: &gpui::Entity<project::Project>,
    directory: &project::ProjectPath,
    product: DoctorProduct,
    observed_at: u64,
    cx: &gpui::App,
) -> Result<Vec<DoctorCheck>> {
    let owner = project.read(cx);
    let observation = owner.doctor_observation(directory.worktree_id, cx);
    ensure!(
        directory.path.is_empty() && observation.worktree_available,
        "Select an available worktree root for development evidence"
    );
    let mut checks = vec![DoctorCheck::new(
        "development.scope",
        "Project development evidence scope",
        CheckOutcome::Untested,
        &format!(
            "{} Cached owner metadata only: no shell, wrapper, dependency refresh, build, test, download, lockfile rewrite or permission change. Recorded observations require a separately reviewed relevant reproduction/verification target.",
            project_authority_description(&observation)
        ),
        observed_at,
    )];
    match product {
        DoctorProduct::Copper => {
            #[cfg(feature = "rust-tools")]
            {
                let cached = owner.cargo_workspace_store().read(cx).cached_snapshot(cx);
                checks.extend(copper_snapshot_checks(cached, directory.worktree_id, observed_at));
                checks.extend(development_cargo_dependency_checks(cached, directory.worktree_id, observed_at));
            }
            #[cfg(not(feature = "rust-tools"))]
            checks.push(DoctorCheck::new("development.rust.disabled", "Rust development integration", CheckOutcome::Unsupported, "Rust development evidence requires the existing rust-tools feature; no provider was initialized or substituted.", observed_at));
        }
        DoctorProduct::Groovy => {
            #[cfg(feature = "jvm-tools")]
            {
                let store = owner.lsp_store().read(cx);
                let java = store.jdtls_project_snapshot(directory.worktree_id);
                let kotlin = store.kotlin_workspace_snapshot(directory.worktree_id);
                checks.extend(groovy_project_checks(java, kotlin, observed_at));
                checks.extend(development_jvm_dependency_checks(java, kotlin, observed_at));
            }
            #[cfg(not(feature = "jvm-tools"))]
            checks.push(DoctorCheck::new("development.jvm.disabled", "JVM development integration", CheckOutcome::Unsupported, "JVM development evidence requires the existing jvm-tools feature. Groovy release readiness is not inferred.", observed_at));
        }
        DoctorProduct::Made => checks.push(DoctorCheck::new("development.family", "Select development language integration", CheckOutcome::Untested, "Select Copper/Rust or Groovy/JVM checks for authoritative project facts. Generic Made product health does not establish build, test or application correctness.", observed_at)),
    }
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    {
        let providers = owner.structured_execution_store().read(cx);
        let scoped = providers.state().providers().filter(|provider| {
            provider.nodes.iter().any(|node| {
                node.path
                    .as_ref()
                    .is_some_and(|path| path.worktree_id == directory.worktree_id)
            })
        });
        let mut total = 0;
        for provider in scoped {
            if total < 8 {
                checks.push(recorded_run_check(
                    provider,
                    total,
                    observed_at,
                    directory.worktree_id,
                    "development",
                ));
            }
            total += 1;
        }
        append_collector_coverage(
            &mut checks,
            "coverage.development.runs",
            "Development workflow provider",
            total,
            8,
            observed_at,
            true,
        );
        if total == 0 {
            checks.push(DoctorCheck::new("development.tests", "Selected build/test evidence", CheckOutcome::Untested, "No selected-worktree structured workflow result is cached. No build/test success or failure has been reproduced by this collection.", observed_at));
        }
    }
    checks.push(DoctorCheck::new("development.timing", "Pipeline timing baseline", CheckOutcome::Untested, "Cached status and test durations lack a reviewed comparable workload, source/configuration revision, host and cache protocol. Review an existing Task/test target to collect a baseline; slowness explanations remain hypotheses.", observed_at));
    Ok(checks)
}

#[cfg(feature = "rust-tools")]
fn development_cargo_dependency_checks(
    cached: Option<(&project::cargo_workspace::CargoWorkspaceSnapshot, bool)>,
    worktree: settings::WorktreeId,
    observed_at: u64,
) -> Vec<DoctorCheck> {
    use project::cargo_workspace::CargoSnapshotCompleteness;
    let Some((snapshot, current)) = cached else {
        return vec![DoctorCheck::new(
            "development.cargo.dependencies",
            "Cargo dependency facts",
            CheckOutcome::Untested,
            "No authoritative cached Cargo model exists. Diagnosis did not resolve or fetch dependencies; use a separately reviewed existing Cargo Task if needed.",
            observed_at,
        )];
    };
    let dependencies = snapshot
        .workspaces
        .iter()
        .filter(|workspace| workspace.key.root.worktree_id == worktree)
        .flat_map(|workspace| workspace.members.iter())
        .flat_map(|package| {
            package
                .dependencies
                .iter()
                .map(move |dependency| (package, dependency))
        });
    let mut checks = Vec::new();
    let mut total = 0;
    for (package, dependency) in dependencies {
        if total < 24 {
            let resolved = dependency
                .resolved_instances
                .iter()
                .take(4)
                .map(|instance| {
                    format!(
                        "{} {} ({:?}, {:?})",
                        instance.name, instance.version, instance.source_kind, instance.lock_status
                    )
                })
                .collect::<Vec<_>>();
            let mut check = DoctorCheck::new(
                &format!("development.cargo.dependency.{total}"),
                "Declared versus recorded Cargo dependency",
                CheckOutcome::Untested,
                &format!(
                    "Package {} {}; declaration {} (alias {:?}), requirement {}, origin {:?}, kind {:?}, source {:?}, target {:?}; recorded selected version {:?}; instances [{}]; current cache {}, partial cache {}, truncated instances {}. These are declaration/resolution facts, not an advisory scan or complete transitive graph; absent resolution is incomplete and no lockfile or source was changed.",
                    package.name,
                    package.version,
                    dependency.declaration_name,
                    dependency.rename,
                    dependency.version_requirement,
                    dependency.declaration_origin,
                    dependency.kind,
                    dependency.source_kind,
                    dependency.target,
                    dependency.resolved_version,
                    resolved.join("; "),
                    current,
                    snapshot.completeness == CargoSnapshotCompleteness::Partial,
                    dependency.resolution_truncated || dependency.resolved_instances.len() > 4
                ),
                observed_at,
            );
            check.historical = !current;
            checks.push(check);
        }
        total += 1;
    }
    append_collector_coverage(
        &mut checks,
        "coverage.development.cargo.dependencies",
        "Declared Cargo dependency",
        total,
        24,
        observed_at,
        !current,
    );
    if total == 0 {
        checks.push(DoctorCheck::new("development.cargo.dependencies", "Cargo dependency facts", CheckOutcome::Untested, "No declared dependency entries are retained for the selected worktree. This does not establish an empty resolved graph, absence of vulnerabilities or build correctness.", observed_at));
    }
    checks
}

#[cfg(feature = "jvm-tools")]
fn development_jvm_dependency_checks(
    java: Option<(&project::jvm_project::JdtlsProjectSnapshot, bool)>,
    kotlin: Option<(&project::jvm_project::KotlinWorkspaceSnapshot, bool)>,
    observed_at: u64,
) -> Vec<DoctorCheck> {
    let mut checks = Vec::new();
    let mut total = 0;
    if let Some((snapshot, current)) = java {
        for module in &snapshot.projects {
            for artifact in &module.library_artifacts {
                if total < 24 {
                    let mut check = DoctorCheck::new(
                        &format!("development.jvm.dependency.{total}"),
                        "Recorded Java classpath coordinate",
                        CheckOutcome::Untested,
                        &format!(
                            "Provider JDTLS, generation {}, coordinate {}, scope {:?}, current {}, partial {}. This is classpath evidence; declared/resolved build graphs, transitive paths and selected Maven/Gradle configurations require their authoritative owner evidence. No wrapper or import was executed.",
                            snapshot.generation,
                            artifact.coordinate,
                            artifact.scope,
                            current,
                            snapshot.partial
                        ),
                        observed_at,
                    );
                    check.historical = !current;
                    checks.push(check);
                }
                total += 1;
            }
        }
    }
    if let Some((snapshot, current)) = kotlin {
        for module in &snapshot.projects {
            for dependency in &module.dependencies {
                if total < 24 {
                    let mut check = DoctorCheck::new(
                        &format!("development.jvm.dependency.{total}"),
                        "Recorded Kotlin dependency fact",
                        CheckOutcome::Untested,
                        &format!(
                            "Provider Kotlin, generation {}, build system {:?}, dependency {:?}, current {}, partial {}, unidentified libraries {}. Declared requirements, build configuration and complete resolved paths are not established by these provider facts; no dependency fetch or build was approved.",
                            snapshot.generation,
                            module.build_system,
                            dependency,
                            current,
                            snapshot.partial,
                            module.unidentified_libraries
                        ),
                        observed_at,
                    );
                    check.historical = !current;
                    checks.push(check);
                }
                total += 1;
            }
        }
    }
    append_collector_coverage(
        &mut checks,
        "coverage.development.jvm.dependencies",
        "Recorded JVM dependency fact",
        total,
        24,
        observed_at,
        java.is_some_and(|(_, current)| !current) || kotlin.is_some_and(|(_, current)| !current),
    );
    if total == 0 {
        checks.push(DoctorCheck::new("development.jvm.dependencies", "JVM dependency evidence", CheckOutcome::Untested, "No Java/Kotlin dependency coordinates are retained. Missing/offline/partial import is not an empty Maven/Gradle graph or a clean security scan. Groovy remains a planned product.", observed_at));
    }
    checks
}

#[cfg(feature = "rust-tools")]
fn copper_owner_checks(
    project: &project::Project,
    worktree_id: settings::WorktreeId,
    observed_at: u64,
    cx: &gpui::App,
) -> Vec<DoctorCheck> {
    let mut checks = copper_snapshot_checks(
        project.cargo_workspace_store().read(cx).cached_snapshot(cx),
        worktree_id,
        observed_at,
    );
    let observation = project.doctor_observation(worktree_id, cx);
    let servers: Vec<_> = observation
        .language_servers
        .iter()
        .filter(|server| server.name == lsp::LanguageServerName::new_static("rust-analyzer"))
        .collect();
    let server_outcome = if servers.is_empty() {
        CheckOutcome::Untested
    } else if servers.iter().any(|server| server.running == Some(false)) {
        CheckOutcome::Warning
    } else if servers.iter().all(|server| server.running == Some(true)) {
        CheckOutcome::Ready
    } else {
        CheckOutcome::Untested
    };
    checks.push(DoctorCheck::new("rust.lsp", "Rust language server", server_outcome,
        &format!("{} rust-analyzer status records on the selected worktree. A running server does not prove build or test success.", servers.len()), observed_at));
    let results = project.structured_execution_store().read(cx);
    let mut included = 0;
    let mut omitted = 0;
    for provider in results.state().providers() {
        if !provider.nodes.iter().any(|node| {
            node.path
                .as_ref()
                .is_some_and(|path| path.worktree_id == worktree_id)
        }) {
            continue;
        }
        if included >= 8 {
            omitted += 1;
            continue;
        }
        checks.push(recorded_run_check(
            provider,
            included,
            observed_at,
            worktree_id,
            "rust",
        ));
        included += 1;
    }
    append_collector_coverage(
        &mut checks,
        "coverage.rust.runs",
        "Rust workflow provider",
        included + omitted,
        8,
        observed_at,
        true,
    );
    if included == 0 {
        checks.push(DoctorCheck::new("rust.tests", "Recorded workflow results", CheckOutcome::Untested,
            "No scoped structured run is recorded. Missing compiler diagnostics do not prove a successful build.", observed_at));
    }
    let sessions = project.dap_store();
    let scoped_sessions = sessions
        .read(cx)
        .sessions()
        .filter(|session| {
            session
                .read(cx)
                .worktree()
                .is_some_and(|worktree| worktree.read(cx).id() == worktree_id)
        })
        .count();
    checks.push(DoctorCheck::new("rust.debug", "Debugger evidence", CheckOutcome::Untested,
        &format!("{scoped_sessions} debugger sessions belong to this worktree. Session presence or termination alone does not establish application correctness; select a reviewed debugger investigation for evidence."), observed_at));
    checks
}

#[cfg(feature = "rust-tools")]
fn copper_snapshot_checks(
    cached: Option<(&project::cargo_workspace::CargoWorkspaceSnapshot, bool)>,
    worktree_id: settings::WorktreeId,
    observed_at: u64,
) -> Vec<DoctorCheck> {
    use project::cargo_workspace::{CargoHostCompilerStatus, CargoSnapshotCompleteness};
    let Some((snapshot, current)) = cached else {
        return vec![DoctorCheck::new(
            "rust.cargo",
            "Cargo workspace evidence",
            CheckOutcome::Untested,
            "No Cargo owner snapshot exists. Refreshing Cargo can execute build scripts, download dependencies and write caches; it requires review.",
            observed_at,
        )];
    };
    let workspaces: Vec<_> = snapshot
        .workspaces
        .iter()
        .filter(|workspace| workspace.key.root.worktree_id == worktree_id)
        .collect();
    let failures = snapshot
        .failures
        .iter()
        .filter(|failure| failure.manifest_path.worktree_id == worktree_id)
        .count();
    let mut checks = vec![DoctorCheck::new(
        "rust.cargo",
        "Cargo workspace evidence",
        if !current || snapshot.completeness == CargoSnapshotCompleteness::Partial {
            CheckOutcome::Warning
        } else if failures > 0 {
            CheckOutcome::Failed
        } else if workspaces.is_empty() {
            CheckOutcome::Untested
        } else {
            CheckOutcome::Ready
        },
        &format!(
            "{} scoped workspaces; {failures} recorded collection failures; snapshot revision {}. {}",
            workspaces.len(),
            snapshot.revision,
            if current {
                "Owner inputs are current; resolution does not prove build success."
            } else {
                "Historical snapshot: inputs or request changed."
            }
        ),
        observed_at,
    )];
    if let Some(check) = checks.first_mut() {
        check.historical = !current;
    }
    for (index, failure) in snapshot
        .failures
        .iter()
        .filter(|failure| failure.manifest_path.worktree_id == worktree_id)
        .take(4)
        .enumerate()
    {
        let mut check = DoctorCheck::new(
            &format!("rust.failure.{index}"),
            &format!(
                "Cargo metadata — {}",
                failure.manifest_path.path.as_unix_str()
            ),
            if !current {
                CheckOutcome::Untested
            } else if failure.category
                == project::cargo_workspace::CargoWorkspaceErrorCategory::Restricted
            {
                CheckOutcome::Blocked
            } else {
                CheckOutcome::Failed
            },
            &format!(
                "Manifest {}; category {:?}; stale cached model {}. {}",
                failure.manifest_path.path.as_unix_str(),
                failure.category,
                failure.has_stale_model,
                failure.message
            ),
            observed_at,
        );
        if failure.category == project::cargo_workspace::CargoWorkspaceErrorCategory::Restricted {
            check.remedy = Some(DiagnosticText::new(
                "Review this worktree's trust in the existing project controls, then refresh Cargo and collect again. Doctor cannot grant trust or run a build automatically.",
            ));
        }
        check.historical = !current;
        checks.push(check);
    }
    append_collector_coverage(
        &mut checks,
        "coverage.rust.failures",
        "Cargo collection failure",
        failures,
        4,
        observed_at,
        !current,
    );
    append_collector_coverage(
        &mut checks,
        "coverage.rust.workspaces",
        "Cargo workspace",
        workspaces.len(),
        8,
        observed_at,
        !current,
    );
    for (index, workspace) in workspaces.iter().take(8).enumerate() {
        let compiler = &workspace.configuration.host_compiler;
        let outcome = if !current || compiler.stale {
            CheckOutcome::Warning
        } else {
            match compiler.status {
                CargoHostCompilerStatus::Available => CheckOutcome::Ready,
                CargoHostCompilerStatus::Restricted => CheckOutcome::Blocked,
                CargoHostCompilerStatus::Missing | CargoHostCompilerStatus::Failed => {
                    CheckOutcome::Failed
                }
                CargoHostCompilerStatus::Unknown => CheckOutcome::Untested,
            }
        };
        let mut check = DoctorCheck::new(
            &format!("rust.compiler.{index}"),
            "Recorded host compiler",
            outcome,
            &format!(
                "Workspace {}; manifest {}. Observed compiler: {}; host target: {}. Required components and targets need a reviewed manager inventory check; no installation was performed.",
                workspace.display_name,
                workspace
                    .root_manifest
                    .as_ref()
                    .map(|manifest| manifest.path.as_unix_str())
                    .unwrap_or("not recorded"),
                compiler.release.as_deref().unwrap_or("unknown"),
                compiler.host_target.as_deref().unwrap_or("unknown")
            ),
            observed_at,
        );
        check.historical = !current || compiler.stale;
        checks.push(check);
        for (diagnostic_index, diagnostic) in workspace
            .configuration
            .diagnostics
            .iter()
            .take(4)
            .enumerate()
        {
            let mut check = DoctorCheck::new(
                &format!("rust.configuration.{index}.{diagnostic_index}"),
                "Cargo configuration observation",
                if current {
                    CheckOutcome::Warning
                } else {
                    CheckOutcome::Untested
                },
                &diagnostic.message,
                observed_at,
            );
            check.historical = !current;
            checks.push(check);
        }
        append_collector_coverage(
            &mut checks,
            &format!("coverage.rust.configuration.{index}"),
            "Cargo configuration diagnostic",
            workspace.configuration.diagnostics.len(),
            4,
            observed_at,
            !current,
        );
    }
    checks
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
fn recorded_run_check(
    provider: &project::structured_execution::StructuredProviderSnapshot,
    index: usize,
    observed_at: u64,
    worktree_id: settings::WorktreeId,
    prefix: &str,
) -> DoctorCheck {
    use project::structured_execution::{StructuredProviderStatus, StructuredRunPhase};
    let run = provider
        .current_run
        .iter()
        .chain(provider.completed_runs.iter().rev())
        .chain(provider.last_complete_run.iter())
        .find(|run| {
            !run.scope_node_ids.is_empty()
                && run.scope_node_ids.iter().all(|node_id| {
                    let mut node = provider.nodes.iter().find(|node| &node.id == node_id);
                    for _ in 0..32 {
                        let Some(current) = node else {
                            return false;
                        };
                        if let Some(path) = &current.path {
                            return path.worktree_id == worktree_id;
                        }
                        node = current.parent_id.as_ref().and_then(|parent| {
                            provider.nodes.iter().find(|node| &node.id == parent)
                        });
                    }
                    false
                })
        });
    let (outcome, detail) = match run {
        None => (
            CheckOutcome::Untested,
            "No recorded run has a verifiable scope confined to this worktree.".to_string(),
        ),
        Some(run) => {
            let outcome = match run.phase {
                StructuredRunPhase::Queued | StructuredRunPhase::Running => CheckOutcome::Untested,
                StructuredRunPhase::Cancelled => CheckOutcome::Cancelled,
                StructuredRunPhase::SpawnError => CheckOutcome::Failed,
                StructuredRunPhase::Completed if run.summary.failed > 0 => CheckOutcome::Failed,
                StructuredRunPhase::Completed
                    if run.truncated
                        || provider.partial
                        || provider.status != StructuredProviderStatus::Current
                        || run.discovery_generation != provider.discovery_generation =>
                {
                    CheckOutcome::Warning
                }
                StructuredRunPhase::Completed
                    if run.summary.passed > 0
                        && run.summary.skipped == 0
                        && run.summary.cancelled == 0 =>
                {
                    CheckOutcome::Ready
                }
                StructuredRunPhase::Completed => CheckOutcome::Untested,
            };
            (
                outcome,
                format!(
                    "Provider {}; run {}; phase {:?}; passed {}, failed {}, skipped {}, cancelled {}; run generation {}, provider generation {}, partial {}, truncated {}. This is recorded scope evidence, not proof of a current project-wide fix.",
                    provider.provider_id.0,
                    run.id.0,
                    run.phase,
                    run.summary.passed,
                    run.summary.failed,
                    run.summary.skipped,
                    run.summary.cancelled,
                    run.discovery_generation.0,
                    provider.discovery_generation.0,
                    provider.partial,
                    run.truncated
                ),
            )
        }
    };
    let mut check = DoctorCheck::new(
        &format!("{prefix}.run.{index}"),
        "Recorded workflow run",
        outcome,
        &detail,
        observed_at,
    );
    check.historical = true;
    check
}

pub(crate) async fn read_confined_declaration(
    filesystem: &dyn fs::Fs,
    root: &std::path::Path,
    root_inode: u64,
    path: &std::path::Path,
    version: (u64, Option<fs::MTime>, u64),
    maximum: u64,
) -> Result<String> {
    use std::io::Read as _;
    let absolute_path = root.join(path);
    let before = filesystem
        .metadata(&absolute_path)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Declaration disappeared"))?;
    ensure!(
        before.inode == version.0
            && Some(before.mtime) == version.1
            && before.len == version.2
            && !before.is_dir
            && !before.is_fifo
            && !before.is_symlink
            && before.len <= maximum,
        "Declaration changed or is not a bounded regular file"
    );
    let mut reader = filesystem
        .open_sync_confined_file(root, path, root_inode, version.0)
        .await?;
    let mut contents = Vec::new();
    reader
        .by_ref()
        .take(maximum + 1)
        .read_to_end(&mut contents)?;
    ensure!(
        contents.len() as u64 <= maximum,
        "Declaration exceeded the byte limit"
    );
    let after = filesystem
        .metadata(&absolute_path)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Declaration disappeared during diagnosis"))?;
    ensure!(
        before.inode == after.inode
            && before.mtime == after.mtime
            && before.len == after.len
            && !after.is_symlink,
        "Declaration changed during diagnosis"
    );
    Ok(String::from_utf8(contents)?)
}

#[cfg(feature = "jvm-tools")]
pub fn collect_groovy_checks(
    project: &gpui::Entity<project::Project>,
    directory: project::ProjectPath,
    observed_at: u64,
    cx: &mut gpui::App,
) -> gpui::Task<Result<Vec<DoctorCheck>>> {
    use gpui::AppContext as _;
    use project::ObservedJavaHome;
    let owner = project.read(cx);
    let observation = owner.doctor_observation(directory.worktree_id, cx);
    let mut checks = groovy_owner_checks(owner, directory.worktree_id, observed_at, cx);
    if !directory.path.is_empty() {
        return gpui::Task::ready(Err(anyhow::anyhow!(
            "Doctor requires a selected worktree root"
        )));
    }
    if observation.host != project::DoctorExecutionHost::Local {
        checks.push(DoctorCheck::new("jvm.java_home", "Host JAVA_HOME candidate", CheckOutcome::Unsupported,
            "Local filesystem and environment cannot validate a remote JDK. Authoritative provider snapshots remain available; no local fallback was used.", observed_at));
        return gpui::Task::ready(Ok(checks));
    }
    let Some(worktree) = owner.worktree_for_id(directory.worktree_id, cx) else {
        return gpui::Task::ready(Err(anyhow::anyhow!("Selected worktree is unavailable")));
    };
    let snapshot = worktree.read(cx).snapshot();
    let java_home = owner
        .environment()
        .read(cx)
        .observed_java_home(&task::Shell::System, snapshot.abs_path());
    let filesystem = owner.fs().clone();
    let project = project.downgrade();
    cx.spawn(async move |cx| {
        let candidate = match java_home.clone() {
            ObservedJavaHome::Unavailable => DoctorCheck::new("jvm.java_home", "Host JAVA_HOME candidate", CheckOutcome::Untested,
                "No selected-directory environment has been loaded. Doctor did not load a shell or infer JAVA_HOME from the application process.", observed_at),
            ObservedJavaHome::Unset => DoctorCheck::new("jvm.java_home", "Host JAVA_HOME candidate", CheckOutcome::Untested,
                "JAVA_HOME is unset in the recorded project environment; PATH, build configuration or provider settings may select a JDK.", observed_at),
            ObservedJavaHome::Path(path) => {
                let filesystem = filesystem.clone();
                match cx.background_spawn(async move { inspect_groovy_jdk(filesystem.as_ref(), &path, observed_at).await }).await {
                    Ok(check) => check,
                    Err(error) => DoctorCheck::new("jvm.java_home", "Host JAVA_HOME candidate", CheckOutcome::Failed, &format!("Configured JDK could not be inspected: {error}"), observed_at),
                }
            }
        };
        checks.push(candidate);
        for (index, (script, properties)) in [
            ("gradlew", "gradle/wrapper/gradle-wrapper.properties"),
            ("mvnw", ".mvn/wrapper/maven-wrapper.properties"),
        ].into_iter().enumerate() {
            let path = util::rel_path::RelPath::from_unix_str(properties)?;
            let script_path = util::rel_path::RelPath::from_unix_str(script)?;
            let script_present = snapshot.entry_for_path(script_path).is_some_and(|entry| entry.is_file() && !entry.is_external && !entry.is_private);
            let entry = snapshot.entry_for_path(path).cloned();
            let check = match entry {
                None => DoctorCheck::new(&format!("jvm.wrapper.{index}"), "Build wrapper declaration", CheckOutcome::Untested,
                    &format!("{script} script present: {script_present}; no recorded wrapper properties. Global tools or other build systems may be configured."), observed_at),
                Some(entry) if !entry.is_file() || entry.is_private || entry.is_external || entry.is_fifo || entry.size > 32 * 1024 =>
                    DoctorCheck::new(&format!("jvm.wrapper.{index}"), "Build wrapper declaration", CheckOutcome::Blocked, "Wrapper properties are private, nonregular, external or oversized; no file was read.", observed_at),
                Some(entry) => {
                    let filesystem = filesystem.clone();
                    let root_path = snapshot.abs_path().to_path_buf();
                    let root_inode = snapshot.root_entry().ok_or_else(|| anyhow::anyhow!("Worktree root unavailable"))?.inode;
                    let relative_path = path.as_std_path().to_path_buf();
                    let text = cx.background_spawn(async move {
                        read_confined_declaration(filesystem.as_ref(), &root_path, root_inode, &relative_path, (entry.inode, entry.mtime, entry.size), 32 * 1024).await
                    }).await;
                    match text {
                        Ok(text) => groovy_wrapper_check(index, script_present, &text, observed_at),
                        Err(error) => DoctorCheck::new(&format!("jvm.wrapper.{index}"), "Build wrapper declaration", CheckOutcome::Failed, &format!("Wrapper could not be inspected: {error}"), observed_at),
                    }
                }
            };
            checks.push(check);
        }
        project.read_with(cx, |owner, cx| {
            ensure!(owner.doctor_observation(directory.worktree_id, cx).host == project::DoctorExecutionHost::Local, "Execution host changed during diagnosis");
            let current = owner.worktree_for_id(directory.worktree_id, cx).ok_or_else(|| anyhow::anyhow!("Selected worktree closed"))?;
            let current_snapshot = current.read(cx).snapshot();
            ensure!(snapshot.root_entry().map(|entry| (entry.id, entry.inode)) == current_snapshot.root_entry().map(|entry| (entry.id, entry.inode)), "Worktree root changed during diagnosis");
            ensure!(java_home == owner.environment().read(cx).observed_java_home(&task::Shell::System, current_snapshot.abs_path()), "JAVA_HOME observation changed during diagnosis");
            for name in ["gradlew", "mvnw", "gradle/wrapper/gradle-wrapper.properties", ".mvn/wrapper/maven-wrapper.properties"] {
                let path = util::rel_path::RelPath::from_unix_str(name)?;
                let before = snapshot.entry_for_path(path).map(|entry| (entry.id, entry.inode, entry.mtime, entry.size, entry.is_private));
                let after = current_snapshot.entry_for_path(path).map(|entry| (entry.id, entry.inode, entry.mtime, entry.size, entry.is_private));
                ensure!(before == after, "Wrapper changed during diagnosis");
            }
            Ok::<(), anyhow::Error>(())
        })??;
        Ok(checks)
    })
}

#[cfg(feature = "jvm-tools")]
async fn inspect_groovy_jdk(
    filesystem: &dyn fs::Fs,
    path: &std::path::Path,
    observed_at: u64,
) -> Result<DoctorCheck> {
    ensure!(path.is_absolute(), "JAVA_HOME is empty or relative");
    let root = filesystem.canonicalize(path).await?;
    let root_metadata = filesystem
        .metadata(&root)
        .await?
        .ok_or_else(|| anyhow::anyhow!("JAVA_HOME directory is missing"))?;
    ensure!(
        root_metadata.is_dir && !root_metadata.is_symlink,
        "JAVA_HOME does not identify a directory"
    );
    let mut available = Vec::new();
    for name in ["java", "javac"] {
        let name = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        available.push(
            filesystem
                .metadata(&root.join("bin").join(name))
                .await?
                .is_some_and(|metadata| {
                    metadata.is_executable
                        && !metadata.is_dir
                        && !metadata.is_fifo
                        && !metadata.is_symlink
                }),
        );
    }
    let java = available.first().copied().unwrap_or(false);
    let javac = available.get(1).copied().unwrap_or(false);
    if !java || !javac {
        return Ok(DoctorCheck::new(
            "jvm.java_home",
            "Host JAVA_HOME candidate",
            CheckOutcome::Failed,
            if !java {
                "JAVA_HOME lacks a verified executable-file candidate for java. No runtime was executed."
            } else {
                "JAVA_HOME has java but lacks javac; a runtime-only candidate does not establish a build JDK."
            },
            observed_at,
        ));
    }
    let release_path = std::path::Path::new("release");
    let release = filesystem.metadata(&root.join(release_path)).await?;
    let major = if let Some(metadata) = release {
        let contents = read_confined_declaration(
            filesystem,
            &root,
            root_metadata.inode,
            release_path,
            (metadata.inode, Some(metadata.mtime), metadata.len),
            4096,
        )
        .await?;
        project::jvm_project::java_major_from_release(&contents)
    } else {
        None
    };
    Ok(DoctorCheck::new(
        "jvm.java_home",
        "Host JAVA_HOME candidate",
        if major.is_some_and(|major| major >= 21) {
            CheckOutcome::Ready
        } else {
            CheckOutcome::Warning
        },
        &format!(
            "java/javac candidate files exist; declared JDK release: {}. JDTLS requires a separate compatible launch JDK; build and target selection may differ. This is metadata evidence, not an executed version probe.",
            major
                .map(|version| version.to_string())
                .unwrap_or_else(|| "unknown".into())
        ),
        observed_at,
    ))
}

#[cfg(feature = "jvm-tools")]
fn groovy_owner_checks(
    owner: &project::Project,
    worktree_id: settings::WorktreeId,
    observed_at: u64,
    cx: &gpui::App,
) -> Vec<DoctorCheck> {
    use settings::Settings as _;
    let lsp_store = owner.lsp_store();
    let lsp_store = lsp_store.read(cx);
    let mut checks = groovy_project_checks(
        lsp_store.jdtls_project_snapshot(worktree_id),
        lsp_store.kotlin_workspace_snapshot(worktree_id),
        observed_at,
    );
    let location = Some(settings::SettingsLocation {
        worktree_id,
        path: util::rel_path::RelPath::empty(),
    });
    for (index, name) in ["Java", "Kotlin"].into_iter().enumerate() {
        let language_name = language::LanguageName::new(name);
        let preferences = language::language_settings::AllLanguageSettings::get(location, cx)
            .language(location, Some(&language_name), cx);
        checks.push(DoctorCheck::new(&format!("jvm.preferences.{index}"), "Effective language-server preferences", CheckOutcome::Ready,
            &format!("{name}: language server enabled {}; configured preferences {:?}. These settings preserve explicit selections and exclusions; they do not establish extension installation or grant process/download permission.", preferences.enable_language_server, preferences.language_servers), observed_at));
    }
    for (index, (name, failed, pending)) in [
        (
            "jdtls",
            lsp_store.jdtls_project_query_failed(worktree_id),
            lsp_store.jdtls_project_query_pending(worktree_id),
        ),
        (
            "kotlin-lsp",
            lsp_store.kotlin_workspace_query_failed(worktree_id)
                || lsp_store.kotlin_workspace_import_failed(worktree_id),
            lsp_store.kotlin_workspace_query_pending(worktree_id),
        ),
        ("kotlin-language-server", false, false),
        ("gradle-language-server", false, false),
    ]
    .into_iter()
    .enumerate()
    {
        let observation = owner.doctor_observation(worktree_id, cx);
        let servers: Vec<_> = observation
            .language_servers
            .iter()
            .filter(|server| server.name == lsp::LanguageServerName::new_static(name))
            .collect();
        let outcome = if failed {
            CheckOutcome::Failed
        } else if pending
            || servers.is_empty()
            || servers.iter().any(|server| server.running.is_none())
        {
            CheckOutcome::Untested
        } else if servers.iter().all(|server| server.running == Some(true)) {
            CheckOutcome::Ready
        } else {
            CheckOutcome::Warning
        };
        checks.push(DoctorCheck::new(&format!("jvm.provider.{index}"), "JVM language provider", outcome, &format!("Provider {name}; recorded servers {}; failed query/import: {failed}; pending: {pending}. Missing status does not authorize extension installation or replacement of an explicit provider choice.", servers.len()), observed_at));
    }
    let results = owner.structured_execution_store().read(cx);
    let scoped_provider_count = results
        .state()
        .providers()
        .filter(|provider| {
            provider.nodes.iter().any(|node| {
                node.path
                    .as_ref()
                    .is_some_and(|path| path.worktree_id == worktree_id)
            })
        })
        .count();
    append_collector_coverage(
        &mut checks,
        "coverage.jvm.runs",
        "JVM workflow provider",
        scoped_provider_count,
        8,
        observed_at,
        true,
    );
    for (index, provider) in results
        .state()
        .providers()
        .filter(|provider| {
            provider.nodes.iter().any(|node| {
                node.path
                    .as_ref()
                    .is_some_and(|path| path.worktree_id == worktree_id)
            })
        })
        .take(8)
        .enumerate()
    {
        checks.push(recorded_run_check(
            provider,
            index,
            observed_at,
            worktree_id,
            "jvm",
        ));
    }
    checks
}

#[cfg(feature = "jvm-tools")]
fn groovy_project_checks(
    jdtls: Option<(&project::jvm_project::JdtlsProjectSnapshot, bool)>,
    kotlin: Option<(&project::jvm_project::KotlinWorkspaceSnapshot, bool)>,
    observed_at: u64,
) -> Vec<DoctorCheck> {
    let mut checks = Vec::new();
    let (java_count, java_current, java_partial) = jdtls
        .map(|(snapshot, current)| (snapshot.projects.len(), current, snapshot.partial))
        .unwrap_or((0, false, false));
    let java_unqualified_count = jdtls
        .map(|(snapshot, _)| {
            snapshot
                .projects
                .iter()
                .filter(|project| !project.membership.is_imported())
                .count()
        })
        .unwrap_or(0);
    let (kotlin_count, kotlin_current, kotlin_partial) = kotlin
        .map(|(snapshot, current)| (snapshot.projects.len(), current, snapshot.partial))
        .unwrap_or((0, false, false));
    checks.push(DoctorCheck::new("jvm.import", "JVM project import evidence", if java_partial || kotlin_partial || java_unqualified_count > 0 || jdtls.is_some() && !java_current || kotlin.is_some() && !kotlin_current { CheckOutcome::Warning } else if java_count > 0 || kotlin_count > 0 { CheckOutcome::Ready } else { CheckOutcome::Untested },
        &format!("JDTLS modules {java_count} (current {java_current}, partial {java_partial}, unknown/metadata-only membership {java_unqualified_count}); Kotlin modules {kotlin_count} (current {kotlin_current}, partial {kotlin_partial}). Groovy remains a planned product; imported metadata does not certify build, test or release readiness."), observed_at));
    if let Some((snapshot, current)) = jdtls {
        append_collector_coverage(
            &mut checks,
            "coverage.jvm.projects",
            "JDTLS project",
            snapshot.projects.len(),
            8,
            observed_at,
            !current,
        );
        for (index, project) in snapshot.projects.iter().take(8).enumerate() {
            let mut check = DoctorCheck::new(
                &format!("jvm.jdk.{index}"),
                "Server, project and target JDKs",
                if !current || snapshot.partial {
                    CheckOutcome::Warning
                } else if project.membership
                    != project::jvm_project::JdtlsProjectMembership::ImportedJava
                {
                    CheckOutcome::Untested
                } else if project
                    .server_launch_java_major
                    .is_some_and(|major| major < 21)
                {
                    CheckOutcome::Failed
                } else if project.project_vm_java_major.is_none()
                    || project.java_target_level.is_none()
                    || project.server_launch_java_major.is_none()
                {
                    CheckOutcome::Untested
                } else if project
                    .project_vm_java_major
                    .zip(project.java_target_level)
                    .is_some_and(|(actual, target)| actual < target)
                {
                    CheckOutcome::Warning
                } else {
                    CheckOutcome::Ready
                },
                &format!(
                    "JDTLS launch JDK {}; LSP-reported project JDK {}; source level {}; target level {}; observed membership {:?}; imported build kind {:?}. Maven/Gradle execution may select another build JDK; it requires validation through that owner.",
                    project
                        .server_launch_java_major
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    project
                        .project_vm_java_major
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    project
                        .java_source_level
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    project
                        .java_target_level
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    project.membership,
                    project.imported_build_kind
                ),
                observed_at,
            );
            check.historical = !current;
            checks.push(check);
        }
    }
    checks
}

#[cfg(feature = "jvm-tools")]
fn groovy_wrapper_check(
    index: usize,
    script_present: bool,
    text: &str,
    observed_at: u64,
) -> DoctorCheck {
    let mut distribution = None;
    let mut checksum = None;
    let mut unsupported = false;
    for line in text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with(['#', '!']))
    {
        if line.ends_with('\\') || line.contains("\\u") {
            unsupported = true;
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "distributionUrl" => {
                    if distribution.is_some() {
                        unsupported = true;
                    }
                    distribution = Some(value.trim().replace("\\:", ":"));
                }
                "distributionSha256Sum" => {
                    if checksum.is_some() {
                        unsupported = true;
                    }
                    checksum = Some(value.trim());
                }
                _ => {}
            }
        } else {
            unsupported = true;
        }
    }
    let safe_url = distribution
        .as_ref()
        .and_then(|value| url::Url::parse(value).ok())
        .is_some_and(|url| {
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        });
    let checksum_valid = checksum.is_some_and(|value| {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    });
    DoctorCheck::new(
        &format!("jvm.wrapper.{index}"),
        "Build wrapper declaration",
        if unsupported {
            CheckOutcome::Unsupported
        } else if !script_present || !safe_url || checksum.is_some() && !checksum_valid {
            CheckOutcome::Failed
        } else if !checksum_valid {
            CheckOutcome::Warning
        } else {
            CheckOutcome::Ready
        },
        &format!(
            "Wrapper script present: {script_present}; reviewed-form HTTPS distribution declaration: {safe_url}; valid SHA-256 declaration: {checksum_valid}; supported property syntax: {}. Metadata only: no script, download, build import or checksum verification was executed.",
            !unsupported
        ),
        observed_at,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gpui::{AppContext as _, BorrowAppContext as _};

    #[cfg(feature = "rust-tools")]
    fn copper_path(worktree: usize, path: &str) -> project::ProjectPath {
        project::ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(worktree),
            path: util::rel_path::RelPath::from_unix_str(path)
                .expect("Fixture path")
                .into(),
        }
    }

    #[cfg(feature = "rust-tools")]
    #[test]
    fn doctor_copper_projects_owner_scope_stale_compiler_and_configuration() {
        use project::cargo_workspace::*;
        let mut configuration = CargoWorkspaceConfiguration::unresolved();
        configuration.host_compiler = CargoHostCompilerModel {
            status: CargoHostCompilerStatus::Available,
            release: Some("1.90.0".into()),
            host_target: Some("aarch64-apple-darwin".into()),
            stale: false,
        };
        let mut snapshot = CargoWorkspaceSnapshot {
            revision: 4,
            input_fingerprint: 2,
            omitted: Default::default(),
            completeness: CargoSnapshotCompleteness::Complete,
            failures: Vec::new(),
            workspaces: vec![CargoWorkspaceModel {
                key: CargoWorkspaceKey {
                    root: copper_path(1, ""),
                },
                root_manifest: Some(copper_path(1, "Cargo.toml")),
                display_name: "example".into(),
                is_virtual: true,
                members: Vec::new(),
                dependency_graph: None,
                configuration,
            }],
        };
        let worktree = settings::WorktreeId::from_usize(1);
        let checks = copper_snapshot_checks(Some((&snapshot, true)), worktree, 10);
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "rust.compiler.0"
                    && check.outcome == CheckOutcome::Ready)
        );
        let stale = copper_snapshot_checks(Some((&snapshot, false)), worktree, 10);
        assert!(stale.iter().all(|check| check.historical));
        assert!(
            stale
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        let unrelated = copper_snapshot_checks(
            Some((&snapshot, true)),
            settings::WorktreeId::from_usize(2),
            10,
        );
        assert_eq!(unrelated.len(), 1);
        assert_eq!(
            unrelated.first().expect("Scope summary").outcome,
            CheckOutcome::Untested
        );
        let configuration = &mut snapshot
            .workspaces
            .first_mut()
            .expect("Fixture workspace")
            .configuration;
        configuration.host_compiler.status = CargoHostCompilerStatus::Missing;
        configuration.add_diagnostic(
            Some(copper_path(1, "rust-toolchain.toml")),
            CargoConfigurationDiagnosticCategory::Toolchain,
            "invalid declared toolchain",
        );
        let checks = copper_snapshot_checks(Some((&snapshot, true)), worktree, 10);
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "rust.compiler.0"
                    && check.outcome == CheckOutcome::Failed)
        );
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "rust.configuration.0.0")
        );
    }

    #[cfg(feature = "rust-tools")]
    #[test]
    fn doctor_copper_declarations_never_infer_component_installation() {
        let source = copper_path(1, "rust-toolchain.toml");
        assert!(
            project::cargo_workspace::parse_rust_toolchain(
                source.clone(),
                "[toolchain]\nchannel = ["
            )
            .is_err()
        );
        let toolchain = project::cargo_workspace::parse_rust_toolchain(source, "[toolchain]\nchannel = \"1.90.0\"\ncomponents = [\"clippy\"]\ntargets = [\"wasm32-unknown-unknown\"]").expect("Valid pin fixture");
        let check = copper_declaration_check(&toolchain, 10);
        assert!(check.detail.as_str().contains("clippy"));
        assert!(check.detail.as_str().contains("wasm32-unknown-unknown"));
        assert!(check.detail.as_str().contains("not installation"));
        assert_eq!(
            copper_snapshot_checks(None, settings::WorktreeId::from_usize(1), 10)
                .first()
                .expect("Absent owner evidence")
                .outcome,
            CheckOutcome::Untested
        );
    }

    #[cfg(feature = "rust-tools")]
    #[test]
    fn doctor_copper_run_provenance_rejects_false_health_claims() {
        use project::structured_execution::*;
        let mut provider = StructuredProviderSnapshot::discovery(
            StructuredProviderId("cargo.tests".into()),
            DiscoveryGeneration(1),
            StructuredProviderStatus::Current,
            vec![StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                id: StructuredNodeId("case-1".into()),
                parent_id: None,
                label: "test".into(),
                kind: StructuredNodeKind::Case,
                path: Some(copper_path(1, "test.rs")),
            }],
        );
        let mut run = StructuredRun::new(
            StructuredRunId("run-1".into()),
            DiscoveryGeneration(1),
            vec![StructuredNodeId("case-1".into())],
        );
        run.phase = StructuredRunPhase::Completed;
        provider.current_run = Some(run.clone());
        assert_eq!(
            recorded_run_check(
                &provider,
                0,
                10,
                settings::WorktreeId::from_usize(1),
                "rust"
            )
            .outcome,
            CheckOutcome::Untested
        );
        run.summary.passed = 1;
        provider.current_run = Some(run.clone());
        let passed = recorded_run_check(
            &provider,
            0,
            10,
            settings::WorktreeId::from_usize(1),
            "rust",
        );
        assert_eq!(passed.outcome, CheckOutcome::Ready);
        assert!(passed.historical);
        assert!(passed.detail.as_str().contains("not proof"));
        assert_eq!(
            recorded_run_check(
                &provider,
                0,
                10,
                settings::WorktreeId::from_usize(2),
                "rust"
            )
            .outcome,
            CheckOutcome::Untested,
            "An unrelated worktree cannot inherit a passing run"
        );
        provider.discovery_generation = DiscoveryGeneration(2);
        let stale = recorded_run_check(
            &provider,
            0,
            10,
            settings::WorktreeId::from_usize(1),
            "rust",
        );
        assert_eq!(stale.outcome, CheckOutcome::Warning);
        assert!(stale.historical);
        assert!(
            stale
                .detail
                .as_str()
                .contains("run generation 1, provider generation 2")
        );
        provider.discovery_generation = DiscoveryGeneration(1);
        run.summary.failed = 1;
        provider.current_run = Some(run.clone());
        assert_eq!(
            recorded_run_check(
                &provider,
                0,
                10,
                settings::WorktreeId::from_usize(1),
                "rust"
            )
            .outcome,
            CheckOutcome::Failed
        );
        run.phase = StructuredRunPhase::Cancelled;
        provider.current_run = Some(run.clone());
        assert_eq!(
            recorded_run_check(
                &provider,
                0,
                10,
                settings::WorktreeId::from_usize(1),
                "rust"
            )
            .outcome,
            CheckOutcome::Cancelled
        );
        run.phase = StructuredRunPhase::Completed;
        run.summary.failed = 0;
        run.truncated = true;
        provider.current_run = Some(run);
        assert_eq!(
            recorded_run_check(
                &provider,
                0,
                10,
                settings::WorktreeId::from_usize(1),
                "rust"
            )
            .outcome,
            CheckOutcome::Warning
        );
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_groovy_import_membership_does_not_invent_java_readiness() {
        use project::jvm_project::*;
        let mut snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id: settings::WorktreeId::from_usize(1),
            generation: 1,
            partial: false,
            projects: vec![JdtlsProject {
                module_root: project::ProjectPath {
                    worktree_id: settings::WorktreeId::from_usize(1),
                    path: util::rel_path::RelPath::empty_arc(),
                },
                membership: JdtlsProjectMembership::Unknown,
                imported_build_kind: JdtlsImportedBuildKind::Unknown,
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 0,
                library_files: Vec::new(),
                library_artifacts: Vec::new(),
                project_vm_known: false,
                project_vm_java_major: None,
                server_launch_java_major: None,
                java_source_level: None,
                java_target_level: None,
                gradle: None,
                maven: None,
            }],
        };
        for membership in [
            JdtlsProjectMembership::Unknown,
            JdtlsProjectMembership::MetadataOnly,
            JdtlsProjectMembership::ImportedNonJava,
        ] {
            snapshot
                .projects
                .first_mut()
                .expect("Observed module")
                .membership = membership;
            let checks = groovy_project_checks(Some((&snapshot, true)), None, 10);
            let import = checks
                .iter()
                .find(|check| check.id.as_str() == "jvm.import")
                .expect("Import evidence");
            assert_eq!(
                import.outcome,
                if membership.is_imported() {
                    CheckOutcome::Ready
                } else {
                    CheckOutcome::Warning
                }
            );
            let jdk = checks
                .iter()
                .find(|check| check.id.as_str() == "jvm.jdk.0")
                .expect("JDK evidence");
            assert_eq!(jdk.outcome, CheckOutcome::Untested);
            assert!(jdk.detail.as_str().contains("imported build kind Unknown"));
        }
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_groovy_preserves_distinct_jdks_partial_and_stale_imports() {
        use project::jvm_project::*;
        let path = project::ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(1),
            path: util::rel_path::RelPath::empty_arc(),
        };
        let mut snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id: path.worktree_id,
            generation: 1,
            partial: false,
            projects: vec![JdtlsProject {
                membership: JdtlsProjectMembership::ImportedJava,
                imported_build_kind: JdtlsImportedBuildKind::Unknown,
                module_root: path,
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 0,
                library_files: Vec::new(),
                library_artifacts: Vec::new(),
                project_vm_known: true,
                project_vm_java_major: Some(17),
                server_launch_java_major: Some(21),
                java_source_level: Some(17),
                java_target_level: Some(17),
                gradle: None,
                maven: None,
            }],
        };
        let checks = groovy_project_checks(Some((&snapshot, true)), None, 10);
        let jdk = checks
            .iter()
            .find(|check| check.id.as_str() == "jvm.jdk.0")
            .expect("JDK observation");
        assert_eq!(jdk.outcome, CheckOutcome::Ready);
        assert!(jdk.detail.as_str().contains("launch JDK 21"));
        assert!(jdk.detail.as_str().contains("project JDK 17"));
        assert!(jdk.detail.as_str().contains("another build JDK"));
        snapshot.partial = true;
        assert!(
            groovy_project_checks(Some((&snapshot, true)), None, 10)
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        snapshot.partial = false;
        let stale = groovy_project_checks(Some((&snapshot, false)), None, 10);
        assert!(
            stale
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        assert!(stale.iter().any(|check| check.historical));
        snapshot
            .projects
            .first_mut()
            .expect("Fixture module")
            .server_launch_java_major = Some(17);
        assert!(
            groovy_project_checks(Some((&snapshot, true)), None, 10)
                .iter()
                .any(|check| check.outcome == CheckOutcome::Failed)
        );
        assert_eq!(
            groovy_project_checks(None, None, 10)
                .first()
                .expect("Import absence")
                .outcome,
            CheckOutcome::Untested
        );
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_groovy_wrapper_metadata_does_not_authorize_execution_or_downloads() {
        let properties = format!(
            "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.14-bin.zip\ndistributionSha256Sum={}\n",
            "a".repeat(64)
        );
        let check = groovy_wrapper_check(0, true, &properties, 10);
        assert_eq!(check.outcome, CheckOutcome::Ready);
        assert!(check.detail.as_str().contains("no script, download"));
        assert_eq!(
            groovy_wrapper_check(0, false, &properties, 10).outcome,
            CheckOutcome::Failed
        );
        assert_eq!(
            groovy_wrapper_check(
                0,
                true,
                "distributionUrl=https://example.org/archive.zip",
                10
            )
            .outcome,
            CheckOutcome::Warning
        );
        for invalid in [
            "distributionUrl=http://example.org/archive.zip",
            "distributionUrl=https://user:secret@example.org/archive.zip",
            "distributionUrl=https://example.org/archive.zip?token=private",
            "distributionUrl=https://example.org/archive.zip\ndistributionSha256Sum=short",
        ] {
            let check = groovy_wrapper_check(0, true, invalid, 10);
            assert_eq!(check.outcome, CheckOutcome::Failed);
            assert!(!check.detail.as_str().contains("secret"));
            assert!(!check.detail.as_str().contains("private"));
        }
        assert_eq!(
            groovy_wrapper_check(0, true, "distributionUrl=x\\\ndistributionUrl=y", 10).outcome,
            CheckOutcome::Unsupported
        );
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    #[gpui::test]
    async fn doctor_groovy_inspects_real_filesystem_jdk_fixture_without_running_it(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        let filesystem = fs::RealFs::new(None, cx.background_executor.clone());
        let directory =
            std::env::temp_dir().join(format!("made-doctor-jdk-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("bin")).expect("Create disposable JDK fixture");
        for name in ["java", "javac"] {
            std::fs::write(
                directory.join("bin").join(name),
                "this fixture must never execute",
            )
            .expect("Write inert candidate");
            std::fs::set_permissions(
                directory.join("bin").join(name),
                std::fs::Permissions::from_mode(0o755),
            )
            .expect("Candidate metadata");
        }
        std::fs::write(directory.join("release"), "JAVA_VERSION=\"21.0.7\"")
            .expect("Release fixture");
        let check = inspect_groovy_jdk(&filesystem, &directory, 10)
            .await
            .expect("Bounded metadata check");
        assert_eq!(check.outcome, CheckOutcome::Ready);
        assert!(
            check
                .detail
                .as_str()
                .contains("not an executed version probe")
        );
        std::fs::remove_file(directory.join("bin/javac"))
            .expect("Simulate runtime-only installation");
        assert_eq!(
            inspect_groovy_jdk(&filesystem, &directory, 11)
                .await
                .expect("Runtime-only check")
                .outcome,
            CheckOutcome::Failed
        );
        assert!(
            inspect_groovy_jdk(&filesystem, std::path::Path::new("relative"), 12)
                .await
                .is_err()
        );
        std::fs::remove_dir_all(directory).expect("Clean up disposable fixture");
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_groovy_collects_wrapper_failure_without_running_scripts(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/groovy", serde_json::json!({
            "pom.xml": "<project/>", "mvnw": "must never execute", "gradlew": "must never execute",
            ".mvn": { "wrapper": { "maven-wrapper.properties": "distributionUrl=https://user:private@example.org/maven.zip" } }
        })).await;
        let project =
            project::Project::test(filesystem.clone(), [std::path::Path::new("/groovy")], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("Selected JVM root")
                .read(cx)
                .id()
        });
        let directory = project::ProjectPath {
            worktree_id,
            path: util::rel_path::RelPath::empty_arc(),
        };
        let checks = cx
            .update(|cx| collect_groovy_checks(&project, directory, 10, cx))
            .await
            .expect("Passive Groovy collection");
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "jvm.wrapper.1"
                    && check.outcome == CheckOutcome::Failed)
        );
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "jvm.java_home"
                    && check.outcome == CheckOutcome::Untested)
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/groovy/mvnw"))
                .await
                .expect("Original wrapper"),
            "must never execute"
        );
        assert!(
            !serde_json::to_string(&checks)
                .expect("Safe serialization")
                .contains("private@example")
        );
    }

    pub(crate) async fn application_restart_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        gpui::WindowHandle<workspace::MultiWorkspace>,
    ) {
        use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
        cx.executor().allow_parking();
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(r#"{"confirm_quit":false}"#, cx)
                    .expect("No implicit test prompt");
            });
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/doctor-restart",
                serde_json::json!({"Cargo.toml":"preserved", "source.rs":"unrelated"}),
            )
            .await;
        let project = project::Project::test_with_worktree_trust(
            filesystem,
            [std::path::Path::new("/doctor-restart")],
            cx,
        )
        .await;
        let directory = project.read_with(cx, |project, cx| project::ProjectPath {
            worktree_id: project
                .visible_worktrees(cx)
                .next()
                .expect("Root")
                .read(cx)
                .id(),
            path: util::rel_path::RelPath::empty_arc(),
        });
        let store = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        (project, directory, window)
    }

    fn application_restart_report(
        review: &DoctorRepairReview<DoctorApplicationRestartAction>,
    ) -> DoctorReport {
        let mut report = report();
        report
            .push(review.before().clone())
            .expect("Selected evidence");
        report
    }

    #[gpui::test]
    async fn doctor_remote_restart_local_scope_never_dispatches_host_changes(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _window) = application_restart_fixture(cx).await;
        let result = cx
            .update(|cx| prepare_remote_restart(&project, directory.clone(), 1, 10, cx))
            .await;
        assert!(result.is_err());
        project.read_with(cx, |project, cx| {
            assert!(project.is_local());
            assert!(project.remote_client().is_none());
            assert_eq!(
                project
                    .doctor_observation(directory.worktree_id, cx)
                    .trusted,
                Some(true)
            );
            assert!(!project.has_active_structured_execution(cx));
        });
    }

    #[gpui::test]
    async fn doctor_app_restart_requested_is_durable_incomplete_and_never_verified(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _window) = application_restart_fixture(cx).await;
        let mut review = cx
            .update(|cx| prepare_application_restart(&project, directory.clone(), 1, 10, cx))
            .await
            .expect("Explicit proposal");
        assert_eq!(review.action().affected_workspace_count(), 1);
        cx.update(|cx| {
            review.approve(
                &project_repair_binding(&project, directory.clone(), 1, cx).expect("Binding"),
            )
        })
        .expect("Explicit approval");
        let report = application_restart_report(&review);
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let restart = cx.expect_restart();
        let result = cx
            .update(|cx| {
                launch_application_restart(
                    review,
                    &project,
                    directory,
                    1,
                    report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Normal owner restart request");
        restart
            .await
            .expect("Observed test platform restart request");
        assert_eq!(result.phase(), DoctorRepairPhase::Incomplete);
        let records = history
            .records(DoctorProduct::Copper, 20)
            .await
            .expect("Persisted before restart");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            ReceiptOutcome::Incomplete
        );
        assert!(
            records
                .first()
                .expect("Receipt")
                .report
                .checks()
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        assert_eq!(
            project
                .read_with(cx, |project, _| project.fs().clone())
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Preserved source"),
            "unrelated"
        );
    }

    #[gpui::test]
    async fn doctor_app_restart_cancelled_prompt_and_precancel_do_not_restart(
        cx: &mut gpui::TestAppContext,
    ) {
        for precancel in [false, true] {
            let (project, directory, _window) = application_restart_fixture(cx).await;
            cx.update(|cx| {
                cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                    store
                        .set_user_settings(r#"{"confirm_quit":true}"#, cx)
                        .expect("Normal confirmation");
                })
            });
            let mut review = cx
                .update(|cx| prepare_application_restart(&project, directory.clone(), 1, 10, cx))
                .await
                .expect("Proposal");
            cx.update(|cx| {
                review.approve(
                    &project_repair_binding(&project, directory.clone(), 1, cx).expect("Binding"),
                )
            })
            .expect("Approval");
            let report = application_restart_report(&review);
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::from_database(&database);
            let (cancel, cancelled) = async_channel::bounded(1);
            if precancel {
                cancel.close();
            }
            let mut restart = cx.expect_restart();
            let task = cx.update(|cx| {
                launch_application_restart(
                    review,
                    &project,
                    directory,
                    1,
                    report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            });
            if !precancel {
                cx.run_until_parked();
                assert!(cx.has_pending_prompt());
                cx.simulate_prompt_answer("Cancel");
            }
            let result = task.await;
            if precancel {
                assert!(result.is_err());
                assert!(
                    history
                        .records(DoctorProduct::Copper, 20)
                        .await
                        .expect("No receipt")
                        .is_empty()
                );
            } else {
                assert_eq!(
                    result.expect("Observed cancellation").phase(),
                    DoctorRepairPhase::Cancelled
                );
                assert_eq!(
                    history
                        .records(DoctorProduct::Copper, 20)
                        .await
                        .expect("Cancelled receipt")
                        .first()
                        .expect("Receipt")
                        .outcome,
                    ReceiptOutcome::Cancelled
                );
            }
            assert!(restart.try_recv().expect("Receiver remains open").is_none());
            assert!(!cx.has_pending_prompt());
        }
    }

    #[gpui::test]
    async fn doctor_app_restart_stale_workspaces_and_dirty_items_block_effects(
        cx: &mut gpui::TestAppContext,
    ) {
        use workspace::item::test::TestItem;
        let (project, directory, window) = application_restart_fixture(cx).await;
        let mut review = cx
            .update(|cx| prepare_application_restart(&project, directory.clone(), 1, 10, cx))
            .await
            .expect("Proposal");
        cx.update(|cx| {
            review.approve(
                &project_repair_binding(&project, directory.clone(), 1, cx).expect("Binding"),
            )
        })
        .expect("Approval");
        let filesystem = project.read_with(cx, |project, _| project.fs().clone());
        let other_project =
            project::Project::test(filesystem, [std::path::Path::new("/doctor-restart")], cx).await;
        window
            .update(cx, |multi, window, cx| {
                multi.test_add_workspace(other_project, window, cx);
            })
            .expect("New affected workspace");
        let report = application_restart_report(&review);
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (_cancel, cancelled) = async_channel::bounded(1);
        assert!(
            cx.update(|cx| launch_application_restart(
                review,
                &project,
                directory.clone(),
                1,
                report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err()
        );
        assert!(
            history
                .records(DoctorProduct::Copper, 20)
                .await
                .expect("No stale receipt")
                .is_empty()
        );
        window
            .update(cx, |multi, window, cx| {
                let background = multi
                    .workspaces()
                    .next()
                    .expect("Background workspace")
                    .clone();
                let item = cx.new(|cx| TestItem::new(cx).with_dirty(true));
                background.update(cx, |workspace, cx| {
                    workspace.add_item_to_active_pane(Box::new(item), None, true, window, cx)
                });
            })
            .expect("Dirty background editor");
        assert!(
            cx.update(|cx| prepare_application_restart(&project, directory, 2, 20, cx))
                .await
                .is_err()
        );
        assert!(
            !cx.has_pending_prompt(),
            "Opening a proposal never implicitly saves or discards"
        );
    }

    async fn project_repair_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
        (
            gpui::Entity<language::Buffer>,
            project::lsp_store::OpenLspBufferHandle,
        ),
    ) {
        use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/repair",
                serde_json::json!({ "main.rs": "fn main() {}", "Cargo.lock": "original" }),
            )
            .await;
        let project = project::Project::test_with_worktree_trust(
            filesystem,
            [std::path::Path::new("/repair")],
            cx,
        )
        .await;
        let directory = project.read_with(cx, |project, cx| project::ProjectPath {
            worktree_id: project
                .visible_worktrees(cx)
                .next()
                .expect("Root")
                .read(cx)
                .id(),
            path: util::rel_path::RelPath::empty_arc(),
        });
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let trust =
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Existing trust owner"));
        trust.update(cx, |store, cx| {
            store.trust(
                &worktree_store,
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(std::sync::Arc::new(language::Language::new(
            language::LanguageConfig {
                name: "Doctor Fixture".into(),
                matcher: language::LanguageMatcher {
                    path_suffixes: vec!["rs".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let servers = languages.register_fake_lsp(
            "Doctor Fixture",
            language::FakeLspAdapter {
                name: "doctor-rust",
                ..Default::default()
            },
        );
        let path = project::ProjectPath {
            worktree_id: directory.worktree_id,
            path: util::rel_path::RelPath::from_unix_str("main.rs")
                .expect("Path")
                .into(),
        };
        let (buffer, handle) = project
            .update(cx, |owner, cx| owner.open_buffer_with_lsp(path, cx))
            .await
            .expect("Open selected buffer through LSP owner");
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer
                .language()
                .expect("Detected fixture language")
                .name()
                .to_string()),
            "Doctor Fixture"
        );
        assert!(
            project.read_with(cx, |project, cx| project
                .doctor_observation(directory.worktree_id, cx)
                .trusted
                == Some(true)),
            "Selected fixture is trusted"
        );
        (project, directory, servers, (buffer, handle))
    }

    #[gpui::test]
    async fn doctor_project_repair_restarts_selected_owner_and_observes_new_identity(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::StreamExt as _;
        let (project, directory, mut servers, _buffer) = project_repair_fixture(cx).await;
        let old_server = servers.next().await.expect("Original server");
        cx.run_until_parked();
        let mut review = cx
            .update(|cx| {
                prepare_project_repair(
                    &project,
                    directory.clone(),
                    DoctorProjectRepairAction::RestartLanguageServer {
                        server: old_server.server.server_id(),
                        name: lsp::LanguageServerName::new_static("doctor-rust"),
                    },
                    1,
                    10,
                    cx,
                )
            })
            .expect("Exact proposal");
        cx.update(|cx| {
            review.approve(
                &project_repair_binding(&project, directory.clone(), 1, cx)
                    .expect("Current authority"),
            )
        })
        .expect("Explicit approval");
        let mut report = report();
        report
            .push(review.before().clone())
            .expect("Relevant check");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let result = cx.update(|cx| {
            launch_project_repair(
                review,
                project.clone(),
                report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        let result = result.await.expect("Existing owner repair");
        assert_eq!(result.phase(), DoctorRepairPhase::Verified);
        let replacement = servers.next().await.expect("Actual replacement server");
        assert_ne!(
            old_server.server.server_id(),
            replacement.server.server_id()
        );
        let records = history
            .records(DoctorProduct::Copper, 40)
            .await
            .expect("Durable observed outcome");
        assert_eq!(
            records.first().expect("Receipt").outcome,
            ReceiptOutcome::Completed
        );
        assert!(
            records
                .first()
                .expect("Receipt")
                .report
                .checks()
                .iter()
                .any(|check| check.id.as_str().ends_with(".recheck")
                    && check.outcome == CheckOutcome::Ready)
        );
        assert_eq!(
            project
                .read_with(cx, |project, _| project.fs().clone())
                .load(std::path::Path::new("/repair/Cargo.lock"))
                .await
                .expect("Preserved unrelated lock"),
            "original"
        );
    }

    #[gpui::test]
    async fn doctor_project_repair_stale_settings_cancellation_and_revoked_trust_do_not_restart(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::StreamExt as _;
        use gpui::BorrowAppContext as _;
        use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
        let (project, directory, mut servers, _buffer) = project_repair_fixture(cx).await;
        let original = servers.next().await.expect("Original server");
        cx.run_until_parked();
        let action = DoctorProjectRepairAction::RestartLanguageServer {
            server: original.server.server_id(),
            name: lsp::LanguageServerName::new_static("doctor-rust"),
        };
        let mut review = cx
            .update(|cx| {
                prepare_project_repair(&project, directory.clone(), action.clone(), 1, 10, cx)
            })
            .expect("Preview");
        cx.update(|cx| {
            review.approve(
                &project_repair_binding(&project, directory.clone(), 1, cx).expect("Authority"),
            )
        })
        .expect("Approval");
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_server_settings("{}", cx)
                    .expect("Intervening settings")
            })
        });
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let mut scoped_report = report();
        scoped_report
            .push(review.before().clone())
            .expect("Finding");
        assert!(
            cx.update(|cx| launch_project_repair(
                review,
                project.clone(),
                scoped_report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err()
        );
        assert!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("No stale dispatch receipt")
                .is_empty()
        );
        let mut review = cx
            .update(|cx| {
                prepare_project_repair(&project, directory.clone(), action.clone(), 1, 10, cx)
            })
            .expect("Fresh preview");
        cx.update(|cx| {
            review.approve(
                &project_repair_binding(&project, directory.clone(), 1, cx).expect("Authority"),
            )
        })
        .expect("Approval");
        let (cancel, cancelled) = async_channel::bounded(1);
        cancel.close();
        let mut scoped_report = report();
        scoped_report
            .push(review.before().clone())
            .expect("Finding");
        assert!(
            cx.update(|cx| launch_project_repair(
                review,
                project.clone(),
                scoped_report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err()
        );
        assert!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("No cancelled receipt")
                .is_empty()
        );
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        trust.update(cx, |store, cx| {
            store.restrict(
                worktree_store.downgrade(),
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        assert!(
            cx.update(|cx| prepare_project_repair(&project, directory, action, 1, 10, cx))
                .is_err()
        );
        assert!(
            servers.try_recv().is_err(),
            "No repeated or unauthorized replacement"
        );
    }

    #[gpui::test]
    async fn doctor_project_repair_cancel_after_dispatch_retains_reached_effects(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::StreamExt as _;
        let (project, directory, mut servers, _buffer) = project_repair_fixture(cx).await;
        let old_server = servers.next().await.expect("Original server");
        let (release, released) = async_channel::bounded::<()>(1);
        let mut shutdowns =
            old_server.set_request_handler::<lsp::request::Shutdown, _, _>(move |_, _| {
                let released = released.clone();
                async move {
                    released.recv().await?;
                    Ok(())
                }
            });
        let mut review = cx
            .update(|cx| {
                prepare_project_repair(
                    &project,
                    directory.clone(),
                    DoctorProjectRepairAction::RestartLanguageServer {
                        server: old_server.server.server_id(),
                        name: lsp::LanguageServerName::new_static("doctor-rust"),
                    },
                    1,
                    10,
                    cx,
                )
            })
            .expect("Preview");
        cx.update(|cx| {
            review.approve(&project_repair_binding(&project, directory, 1, cx).expect("Authority"))
        })
        .expect("Approval");
        let mut scoped_report = report();
        scoped_report
            .push(review.before().clone())
            .expect("Relevant check");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = cx.update(|cx| {
            launch_project_repair(
                review,
                project.clone(),
                scoped_report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        shutdowns
            .next()
            .await
            .expect("Actual owner accepted restart");
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Receipt before effects")
                .first()
                .expect("Receipt")
                .outcome,
            ReceiptOutcome::Running
        );
        cancel.close();
        let result = result.await.expect("Cancelled waiting result");
        assert_eq!(result.phase(), DoctorRepairPhase::Cancelled);
        assert!(
            result
                .operation_detail()
                .expect("Reached effects")
                .as_str()
                .contains("may still finish")
        );
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 11)
                .await
                .expect("Persisted cancellation")
                .first()
                .expect("Receipt")
                .outcome,
            ReceiptOutcome::Cancelled
        );
        release
            .send(())
            .await
            .expect("Let existing atomic owner finish");
        servers
            .next()
            .await
            .expect("Accepted owner work can finish after Doctor cancellation");
    }

    #[gpui::test]
    async fn doctor_project_repair_timeout_is_incomplete_not_verified(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::StreamExt as _;
        let (project, directory, mut servers, _buffer) = project_repair_fixture(cx).await;
        let old_server = servers.next().await.expect("Original server");
        let (release, released) = async_channel::bounded::<()>(1);
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        let mut replacements = languages.register_fake_lsp_server(
            lsp::LanguageServerName::new_static("doctor-rust"),
            lsp::LanguageServer::full_capabilities(),
            Some(Box::new(move |server| {
                let released = released.clone();
                server.set_request_handler::<lsp::request::Initialize, _, _>(move |_, _| {
                    let released = released.clone();
                    async move {
                        released.recv().await?;
                        Ok(lsp::InitializeResult {
                            capabilities: lsp::LanguageServer::full_capabilities(),
                            ..Default::default()
                        })
                    }
                });
            })),
        );
        let mut review = cx
            .update(|cx| {
                prepare_project_repair(
                    &project,
                    directory.clone(),
                    DoctorProjectRepairAction::RestartLanguageServer {
                        server: old_server.server.server_id(),
                        name: lsp::LanguageServerName::new_static("doctor-rust"),
                    },
                    1,
                    10,
                    cx,
                )
            })
            .expect("Preview");
        cx.update(|cx| {
            review.approve(&project_repair_binding(&project, directory, 1, cx).expect("Authority"))
        })
        .expect("Approval");
        let mut scoped_report = report();
        scoped_report
            .push(review.before().clone())
            .expect("Relevant check");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let result = cx
            .update(|cx| {
                launch_project_repair(
                    review,
                    project.clone(),
                    scoped_report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Bounded recheck");
        assert_eq!(result.phase(), DoctorRepairPhase::Incomplete);
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 40)
                .await
                .expect("Incomplete receipt")
                .first()
                .expect("Receipt")
                .outcome,
            ReceiptOutcome::Incomplete
        );
        release.send(()).await.expect("Let owner finish");
        replacements
            .next()
            .await
            .expect("Owner replacement after observation deadline");
    }

    #[gpui::test]
    fn doctor_project_repair_guidance_uses_existing_actions_and_never_claims_a_fix(
        cx: &mut gpui::App,
    ) {
        for handoff in [
            DoctorProjectHandoff::Settings,
            DoctorProjectHandoff::TaskConfiguration,
            DoctorProjectHandoff::DebugConfiguration,
        ] {
            handoff.action(cx).expect("Existing configuration action");
            assert_eq!(handoff.result(10).outcome, CheckOutcome::Untested);
        }
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_project_repair_jvm_refresh_requires_fresh_selected_complete_metadata() {
        let expected = lsp::LanguageServerId(1);
        let check = |snapshot, failed| {
            jvm_refresh_check("java.import", expected, Some(3), snapshot, failed, true, 11)
        };
        assert_eq!(
            check(Some((expected, 4, true, false, 1)), false).outcome,
            CheckOutcome::Ready
        );
        for snapshot in [
            None,
            Some((expected, 3, true, false, 1)),
            Some((expected, 4, false, false, 1)),
            Some((expected, 4, true, true, 1)),
            Some((expected, 4, true, false, 0)),
            Some((lsp::LanguageServerId(2), 4, true, false, 1)),
        ] {
            assert_eq!(check(snapshot, false).outcome, CheckOutcome::Untested);
        }
        assert_eq!(
            check(Some((expected, 4, true, false, 1)), true).outcome,
            CheckOutcome::Failed
        );
    }

    fn repair_binding(cx: &mut gpui::TestAppContext) -> DoctorRepairBinding {
        use gpui::AppContext as _;
        let identity = cx.update(|cx| cx.new(|_| ()));
        DoctorRepairBinding {
            project: identity.entity_id(),
            directory: project::ProjectPath {
                worktree_id: settings::WorktreeId::from_usize(1),
                path: util::rel_path::RelPath::empty().into(),
            },
            host: project::DoctorExecutionHost::Local,
            connection: None,
            generation: 1,
            input_revision: 1,
        }
    }

    fn repair_report() -> (DoctorReport, DoctorCheck) {
        let mut report = report();
        let before = DoctorCheck::new(
            "rust.server",
            "Rust language server",
            CheckOutcome::Failed,
            "Observed server unavailable",
            10,
        );
        report.push(before.clone()).expect("Scoped finding");
        (report, before)
    }

    #[gpui::test]
    async fn doctor_admission_concurrent_views_share_one_durable_receipt(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let effects = DiagnosticText::new("Restart selected existing server; no permission grants");
        let mut first =
            DoctorRepairReview::new((), binding.clone(), before.clone(), effects.clone())
                .expect("First view");
        let mut second =
            DoctorRepairReview::new((), binding.clone(), before, effects).expect("Second view");
        first.approve(&binding).expect("First explicit approval");
        second.approve(&binding).expect("Second explicit approval");
        let (_cancel, cancelled) = async_channel::bounded(1);
        let results = futures::join!(
            first.record_before_execution(
                &binding,
                report.clone(),
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled
            ),
            second.record_before_execution(
                &binding,
                report,
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled
            )
        );
        assert_ne!(
            results.0.is_ok(),
            results.1.is_ok(),
            "Exactly one existing owner may dispatch"
        );
        assert_eq!(
            usize::from(first.receipt().is_some()) + usize::from(second.receipt().is_some()),
            1
        );
        let records = history
            .records(DoctorProduct::Copper, 10)
            .await
            .expect("Durable admission");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            ReceiptOutcome::Running
        );
    }

    #[gpui::test]
    async fn doctor_admission_refresh_cannot_bypass_active_action_but_distinct_scopes_can(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let effects = DiagnosticText::new("Exact selected action");
        let mut first =
            DoctorRepairReview::new((), binding.clone(), before.clone(), effects.clone())
                .expect("View");
        first.approve(&binding).expect("Approval");
        let (_cancel, cancelled) = async_channel::bounded(1);
        first
            .record_before_execution(
                &binding,
                report.clone(),
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled,
            )
            .await
            .expect("Admission");
        let mut refreshed = binding.clone();
        refreshed.generation = 2;
        refreshed.input_revision = 2;
        let mut refreshed_report = report.clone();
        refreshed_report.generation = 2;
        let mut duplicate =
            DoctorRepairReview::new((), refreshed.clone(), before.clone(), effects.clone())
                .expect("Refreshed view");
        duplicate
            .approve(&refreshed)
            .expect("Fresh explicit approval");
        assert!(
            duplicate
                .record_before_execution(
                    &refreshed,
                    refreshed_report,
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled
                )
                .await
                .is_err()
        );
        assert_eq!(duplicate.phase(), DoctorRepairPhase::Incomplete);
        assert!(duplicate.receipt().is_none());
        let mut another_scope = binding.clone();
        another_scope.directory.worktree_id = settings::WorktreeId::from_usize(2);
        let mut independent =
            DoctorRepairReview::new((), another_scope.clone(), before.clone(), effects)
                .expect("Different authoritative root");
        independent
            .approve(&another_scope)
            .expect("Independent approval");
        independent
            .record_before_execution(
                &another_scope,
                report.clone(),
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled,
            )
            .await
            .expect("Distinct scope remains independent even with identical displayed labels");
        let mut distinct_action = DoctorRepairReview::new(
            (),
            binding.clone(),
            before,
            DiagnosticText::new("Exact different action"),
        )
        .expect("Different concrete action");
        distinct_action
            .approve(&binding)
            .expect("Different action approval");
        distinct_action
            .record_before_execution(
                &binding,
                report,
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled,
            )
            .await
            .expect("Distinct action admitted");
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Receipts")
                .len(),
            3
        );
    }

    #[gpui::test]
    async fn doctor_admission_terminal_outcomes_release_without_automatic_replay(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let effects = DiagnosticText::new("Existing owner operation");
        let (_cancel, cancelled) = async_channel::bounded(1);
        for outcome in [
            DoctorRepairOperationOutcome::Cancelled,
            DoctorRepairOperationOutcome::Failed,
            DoctorRepairOperationOutcome::Completed,
        ] {
            let mut review =
                DoctorRepairReview::new((), binding.clone(), before.clone(), effects.clone())
                    .expect("Fresh review");
            assert!(
                review
                    .record_before_execution(
                        &binding,
                        report.clone(),
                        DoctorProduct::Copper,
                        &history,
                        10,
                        &cancelled
                    )
                    .await
                    .is_err(),
                "No replay without new approval"
            );
            review
                .approve(&binding)
                .expect("Separate explicit approval");
            review
                .record_before_execution(
                    &binding,
                    report.clone(),
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled,
                )
                .await
                .expect("Previous terminal action releases admission");
            review
                .operation_finished(
                    outcome,
                    DiagnosticText::new("Observed terminal owner result"),
                )
                .expect("Owner outcome");
            if review.phase() == DoctorRepairPhase::Rechecking {
                review
                    .recheck_unavailable(DiagnosticText::new(
                        "No relevant recheck; do not claim a fix",
                    ))
                    .expect("Incomplete verification");
            }
            review
                .persist_outcome(DoctorProduct::Copper, &history, None, 10)
                .await
                .expect("Terminal receipt");
        }
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("History")
                .len(),
            3
        );
    }

    #[gpui::test]
    async fn doctor_admission_dropped_review_stays_active_until_interrupted_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let effects = DiagnosticText::new("Explicit owner action");
        let (_cancel, cancelled) = async_channel::bounded(1);
        let mut abandoned =
            DoctorRepairReview::new((), binding.clone(), before.clone(), effects.clone())
                .expect("Review");
        abandoned.approve(&binding).expect("Approval");
        abandoned
            .record_before_execution(
                &binding,
                report.clone(),
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled,
            )
            .await
            .expect("Receipt");
        drop(abandoned);
        let mut fresh =
            DoctorRepairReview::new((), binding.clone(), before.clone(), effects.clone())
                .expect("Another view");
        fresh
            .approve(&binding)
            .expect("Approval does not cancel active work");
        assert!(
            fresh
                .record_before_execution(
                    &binding,
                    report.clone(),
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled
                )
                .await
                .is_err()
        );
        let reopened = DoctorHistoryStore::from_database(&database);
        assert_eq!(
            reopened
                .records(DoctorProduct::Copper, 11)
                .await
                .expect("Reopen")
                .first()
                .expect("Interrupted receipt")
                .outcome,
            ReceiptOutcome::Interrupted
        );
        let mut new_review = DoctorRepairReview::new((), binding.clone(), before, effects)
            .expect("Explicit new review");
        assert_eq!(new_review.phase(), DoctorRepairPhase::Proposed);
        new_review
            .approve(&binding)
            .expect("Separate post-recovery consent");
        new_review
            .record_before_execution(
                &binding,
                report,
                DoctorProduct::Copper,
                &reopened,
                11,
                &cancelled,
            )
            .await
            .expect("New action after interruption");
    }

    #[gpui::test]
    fn doctor_admission_identity_is_bounded_redacted_and_old_history_compatible(
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let effects = DiagnosticText::new(
            "Reviewed /Users/private-person/project/settings password=private-secret",
        );
        let mut record =
            DoctorHistoryRecord::new(report, Some(effects.clone()), ReceiptOutcome::Completed, 10);
        record.admission_key = Some(binding.admission_key(&before.id, &effects));
        let encoded = serde_json::to_string(&record).expect("Bounded receipt");
        assert!(!encoded.contains("private-person") && !encoded.contains("private-secret"));
        let mut old: serde_json::Value =
            serde_json::from_str(&encoded).expect("Old schema fixture");
        old.as_object_mut()
            .expect("Receipt object")
            .remove("admission_key");
        let older: DoctorHistoryRecord =
            serde_json::from_value(old).expect("Existing history without new optional metadata");
        assert!(older.admission_key.is_none());
        let mut snapshot = DoctorHistorySnapshot {
            version: 1,
            records: vec![record],
        };
        snapshot
            .validate(DoctorProduct::Copper)
            .expect("Current metadata");
        snapshot
            .records
            .first_mut()
            .expect("Fixture record")
            .admission_key = Some("not-an-opaque-identity".into());
        assert!(snapshot.validate(DoctorProduct::Copper).is_err());
    }

    #[gpui::test]
    async fn doctor_admission_dropped_queued_receipt_still_blocks_duplicate() {
        use futures::FutureExt as _;
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        };
        let paused = Arc::new(AtomicBool::new(false));
        let writes: Arc<Mutex<Vec<Box<dyn FnOnce() + Send>>>> = Arc::new(Mutex::new(Vec::new()));
        let connection = db::sqlez::thread_safe_connection::ThreadSafeConnection::builder::<()>(
            &format!("doctor-admission-{}", uuid::Uuid::new_v4()),
            false,
        )
        .with_write_queue_constructor(Box::new({
            let paused = paused.clone();
            let writes = writes.clone();
            move || {
                Box::new({
                    let paused = paused.clone();
                    let writes = writes.clone();
                    move |write| {
                        if paused.load(Ordering::SeqCst) {
                            writes.lock().expect("Fixture queue").push(write);
                        } else {
                            write();
                        }
                    }
                })
            }
        }))
        .build()
        .await
        .expect("Delayed AppDB fixture");
        connection.write(|connection| connection.exec("CREATE TABLE scoped_kv_store(namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(namespace, key)) STRICT;")?()).await.expect("Existing KVP schema");
        let database = db::AppDatabase(connection);
        let history = DoctorHistoryStore::from_database(&database);
        history
            .records(DoctorProduct::Copper, 10)
            .await
            .expect("Initialized owner");
        let mut first = DoctorHistoryRecord::new(
            report(),
            Some(DiagnosticText::new("Approved exact action")),
            ReceiptOutcome::Running,
            10,
        );
        first.admission_key = Some("a".repeat(64));
        let mut duplicate = first.clone();
        duplicate.id = uuid::Uuid::new_v4();
        paused.store(true, Ordering::SeqCst);
        let mut pending = history.record(DoctorProduct::Copper, first, 10).boxed();
        assert!(futures::poll!(&mut pending).is_pending());
        for write in std::mem::take(&mut *writes.lock().expect("Fixture queue")) {
            write();
        }
        if history
            .database
            .scoped(HISTORY_NAMESPACE)
            .read("Copper")
            .expect("Persisted receipt")
            .is_none()
        {
            assert!(futures::poll!(&mut pending).is_pending());
        }
        drop(pending);
        for write in std::mem::take(&mut *writes.lock().expect("Fixture queue")) {
            write();
        }
        paused.store(false, Ordering::SeqCst);
        assert!(
            history
                .record(DoctorProduct::Copper, duplicate, 11)
                .await
                .is_err(),
            "A committed receipt cannot be bypassed by dropping its waiter"
        );
        let records = history
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Authoritative owner state");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Original receipt").outcome,
            ReceiptOutcome::Running
        );
        let reopened = DoctorHistoryStore::from_database(&database);
        assert_eq!(
            reopened
                .records(DoctorProduct::Copper, 12)
                .await
                .expect("Interrupted recovery")
                .first()
                .expect("Receipt")
                .outcome,
            ReceiptOutcome::Interrupted
        );
    }

    #[gpui::test]
    fn doctor_repair_state_rejects_stale_duplicate_and_cancelled_approvals(
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = repair_binding(cx);
        let (_, before) = repair_report();
        let mut review = DoctorRepairReview::new(
            "existing-owner-action",
            binding.clone(),
            before,
            DiagnosticText::new("Restart selected server; may download under existing permissions"),
        )
        .expect("Concrete preview");
        for changed in 0..4 {
            let mut current = binding.clone();
            match changed {
                0 => current.generation += 1,
                1 => current.input_revision += 1,
                2 => current.host = project::DoctorExecutionHost::RemoteConnected,
                _ => current.directory.worktree_id = settings::WorktreeId::from_usize(2),
            }
            assert!(review.approve(&current).is_err());
            assert_eq!(review.phase(), DoctorRepairPhase::Proposed);
        }
        review.approve(&binding).expect("Exact user review");
        assert!(review.approve(&binding).is_err());
        assert!(review.receipt().is_none());
        review
            .cancel_before_execution()
            .expect("Cancel without changes");
        assert!(review.revalidate(&binding).is_err());
        assert!(review.approve(&binding).is_err());
    }

    #[gpui::test]
    async fn doctor_repair_state_requires_relevant_current_recheck(cx: &mut gpui::TestAppContext) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        for case in 0..7 {
            let (report, before) = repair_report();
            let mut review = DoctorRepairReview::new(
                (),
                binding.clone(),
                before,
                DiagnosticText::new("Reviewed owner operation"),
            )
            .expect("Preview");
            review.approve(&binding).expect("Approval");
            let (_cancel, cancelled) = async_channel::bounded(1);
            review
                .record_before_execution(
                    &binding,
                    report,
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled,
                )
                .await
                .expect("Durable receipt before effects");
            assert!(
                review
                    .record_before_execution(
                        &binding,
                        repair_report().0,
                        DoctorProduct::Copper,
                        &history,
                        10,
                        &cancelled
                    )
                    .await
                    .is_err()
            );
            assert!(
                review
                    .persist_outcome(DoctorProduct::Copper, &history, None, 11)
                    .await
                    .is_err()
            );
            review
                .operation_finished(
                    DoctorRepairOperationOutcome::Completed,
                    DiagnosticText::new("Command exited successfully; not a verified fix"),
                )
                .expect("Observed operation");
            assert_eq!(review.phase(), DoctorRepairPhase::Rechecking);
            let mut after = DoctorCheck::new(
                "rust.server",
                "Rust language server",
                CheckOutcome::Ready,
                "Observed current ready server",
                11,
            );
            match case {
                0 => after.historical = true,
                1 => after.id = DiagnosticText::new("unrelated.tests"),
                2 => after.observed_at = 9,
                3 => after.outcome = CheckOutcome::Failed,
                _ => {}
            }
            let mut current = binding.clone();
            if case == 5 {
                current.input_revision += 1;
            }
            if case == 6 {
                current.host = project::DoctorExecutionHost::RemoteConnected;
            }
            review
                .observe_recheck(&current, after)
                .expect("Record observed recheck");
            assert_eq!(
                review.phase(),
                match case {
                    3 => DoctorRepairPhase::StillFailing,
                    4 => DoctorRepairPhase::Verified,
                    _ => DoctorRepairPhase::Incomplete,
                }
            );
            review
                .persist_outcome(
                    DoctorProduct::Copper,
                    &history,
                    Some(DiagnosticText::new("native-session")),
                    11,
                )
                .await
                .expect("Persist bounded before/after outcome");
            assert!(
                review
                    .persist_outcome(DoctorProduct::Copper, &history, None, 12)
                    .await
                    .is_err()
            );
        }
        let records = history
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("History");
        assert_eq!(records.len(), 7);
        let verified = records
            .iter()
            .find(|record| record.outcome == ReceiptOutcome::Completed)
            .expect("Verified record");
        assert_eq!(verified.outcome, ReceiptOutcome::Completed);
        assert_eq!(verified.report.checks().len(), 3);
        assert!(
            verified
                .report
                .checks()
                .iter()
                .any(|check| check.id.as_str() == "rust.server.recheck")
        );
    }

    #[gpui::test]
    async fn doctor_repair_state_failed_or_cancelled_operation_never_verifies(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        for outcome in [
            DoctorRepairOperationOutcome::Failed,
            DoctorRepairOperationOutcome::Cancelled,
            DoctorRepairOperationOutcome::AwaitingUserAction,
        ] {
            let (report, before) = repair_report();
            let mut review = DoctorRepairReview::new(
                (),
                binding.clone(),
                before,
                DiagnosticText::new("Reviewed partial action"),
            )
            .expect("Preview");
            review.approve(&binding).expect("Approval");
            let (_cancel, cancelled) = async_channel::bounded(1);
            review
                .record_before_execution(
                    &binding,
                    report,
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled,
                )
                .await
                .expect("Receipt");
            assert!(
                review.cancel_before_execution().is_err(),
                "Running work must use its owner's cancellation"
            );
            review
                .operation_finished(
                    outcome,
                    DiagnosticText::new("Partial effects; Authorization: Bearer REPAIR_SECRET"),
                )
                .expect("Observed partial result");
            if outcome == DoctorRepairOperationOutcome::Failed {
                review
                    .observe_recheck(
                        &binding,
                        DoctorCheck::new(
                            "rust.server",
                            "Server",
                            CheckOutcome::Ready,
                            "Observed ready",
                            11,
                        ),
                    )
                    .expect("Recheck");
                assert_eq!(review.phase(), DoctorRepairPhase::Incomplete);
            }
            assert_ne!(review.phase(), DoctorRepairPhase::Verified);
            review
                .persist_outcome(DoctorProduct::Copper, &history, None, 11)
                .await
                .expect("Persist terminal summary");
        }
        assert!(
            !serde_json::to_string(
                &history
                    .records(DoctorProduct::Copper, 11)
                    .await
                    .expect("Redacted history")
            )
            .expect("Bounded serialization")
            .contains("REPAIR_SECRET")
        );
    }

    #[gpui::test]
    async fn doctor_repair_state_storage_failure_blocks_dispatch_and_drop_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        history
            .records(DoctorProduct::Copper, 10)
            .await
            .expect("Initialize schema");
        database.0.write(|connection| connection.exec("CREATE TRIGGER reject_doctor_receipt BEFORE INSERT ON scoped_kv_store BEGIN SELECT RAISE(ABORT, 'fixture failure'); END;")?()).await.expect("Storage failure fixture");
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let mut review = DoctorRepairReview::new(
            (),
            binding.clone(),
            before,
            DiagnosticText::new("Concrete action"),
        )
        .expect("Preview");
        review.approve(&binding).expect("Approval");
        let (_cancel, cancelled) = async_channel::bounded(1);
        assert!(
            review
                .record_before_execution(
                    &binding,
                    report,
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled
                )
                .await
                .is_err()
        );
        assert_eq!(review.phase(), DoctorRepairPhase::Incomplete);
        assert!(review.receipt().is_none());
        assert!(review.revalidate(&binding).is_err());
        database
            .0
            .write(|connection| connection.exec("DROP TRIGGER reject_doctor_receipt")?())
            .await
            .expect("Restore persistence");
        let (report, before) = repair_report();
        let mut interrupted = DoctorRepairReview::new(
            (),
            binding.clone(),
            before,
            DiagnosticText::new("No replay"),
        )
        .expect("New reviewed proposal");
        interrupted
            .approve(&binding)
            .expect("Explicit new approval");
        interrupted
            .record_before_execution(
                &binding,
                report,
                DoctorProduct::Copper,
                &history,
                10,
                &cancelled,
            )
            .await
            .expect("Durable pending receipt");
        drop(interrupted);
        let reopened = DoctorHistoryStore::from_database(&database);
        let records = reopened
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Recovered owner history");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Interrupted receipt").outcome,
            ReceiptOutcome::Interrupted
        );
    }

    #[gpui::test]
    async fn doctor_repair_state_precancel_has_no_receipt(cx: &mut gpui::TestAppContext) {
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let binding = repair_binding(cx);
        let (report, before) = repair_report();
        let mut review = DoctorRepairReview::new(
            (),
            binding.clone(),
            before,
            DiagnosticText::new("Concrete operation"),
        )
        .expect("Preview");
        review.approve(&binding).expect("Approval");
        let (cancel, cancelled) = async_channel::bounded(1);
        cancel.close();
        assert!(
            review
                .record_before_execution(
                    &binding,
                    report,
                    DoctorProduct::Copper,
                    &history,
                    10,
                    &cancelled
                )
                .await
                .is_err()
        );
        assert_eq!(review.phase(), DoctorRepairPhase::Cancelled);
        assert!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Unchanged history")
                .is_empty()
        );
    }

    fn report() -> DoctorReport {
        DoctorReport::new(
            DoctorScope {
                product: DiagnosticText::new("Copper"),
                project: DiagnosticText::new("project-key"),
                worktree: DiagnosticText::new("worktree-key"),
                host: DiagnosticText::new("Local"),
            },
            1,
            10,
        )
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_copper_reads_confined_pin_without_executing_or_changing_lockfile(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/doctor", serde_json::json!({
            "Cargo.toml": "[workspace]", "Cargo.lock": "version = 4",
            "rust-toolchain.toml": "[toolchain]\nchannel = \"1.90.0\"\ncomponents = [\"clippy\"]"
        })).await;
        let project =
            project::Project::test(filesystem.clone(), [std::path::Path::new("/doctor")], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("Selected fixture root")
                .read(cx)
                .id()
        });
        let directory = project::ProjectPath {
            worktree_id,
            path: util::rel_path::RelPath::empty().into(),
        };
        let calls = project::cargo_workspace_store::local_cargo_process_calls_for_test();
        let checks = cx
            .update(|cx| collect_copper_checks(&project, directory.clone(), 10, cx))
            .await
            .expect("Passive collection");
        assert!(
            checks.iter().any(
                |check| check.id.as_str() == "rust.pin" && check.outcome == CheckOutcome::Ready
            )
        );
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            calls
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor/Cargo.lock"))
                .await
                .expect("Original lockfile"),
            "version = 4"
        );
        filesystem
            .insert_tree(
                "/doctor",
                serde_json::json!({ "rust-toolchain.toml": "[toolchain]\nchannel = [" }),
            )
            .await;
        cx.run_until_parked();
        let checks = cx
            .update(|cx| collect_copper_checks(&project, directory, 11, cx))
            .await
            .expect("Malformed pin is a visible check");
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "rust.pin"
                    && check.outcome == CheckOutcome::Failed)
        );
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            calls
        );
        filesystem
            .insert_tree(
                "/doctor",
                serde_json::json!({ "rust-toolchain.toml": "x".repeat(32 * 1024 + 1) }),
            )
            .await;
        cx.run_until_parked();
        let directory = project::ProjectPath {
            worktree_id,
            path: util::rel_path::RelPath::empty().into(),
        };
        let checks = cx
            .update(|cx| collect_copper_checks(&project, directory, 12, cx))
            .await
            .expect("Oversized pin is visible");
        assert!(
            checks
                .iter()
                .any(|check| check.id.as_str() == "rust.pin"
                    && check.outcome == CheckOutcome::Blocked)
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn doctor_probe_inspects_and_rejects_replaced_executable(cx: &mut gpui::TestAppContext) {
        let filesystem = fs::RealFs::new(None, cx.background_executor.clone());
        let path = std::env::temp_dir().join(format!("doctor-probe-{}", std::process::id()));
        std::fs::create_dir_all(&path).expect("Create executable fixture directory");
        let executable = path.join("java");
        std::fs::write(&executable, b"\x7fELForiginal").expect("Write executable fixture");
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))
            .expect("Set fixture executable permission");
        let observed =
            ProbeExecutable::inspect(DependencyProbe::Java, executable.clone(), &[], &filesystem)
                .await
                .expect("Inspect known fixture format");
        observed
            .revalidate(&[], &filesystem)
            .await
            .expect("Unchanged fixture revalidates");
        std::fs::write(&executable, b"\x7fELFreplaced-longer").expect("Replace fixture contents");
        assert!(observed.revalidate(&[], &filesystem).await.is_err());
        assert!(
            ProbeExecutable::inspect(
                DependencyProbe::Java,
                executable.clone(),
                std::slice::from_ref(&path),
                &filesystem
            )
            .await
            .is_err()
        );
        std::fs::remove_file(executable).expect("Remove fixture executable");
        std::fs::remove_dir(path).expect("Remove fixture directory");
    }

    #[test]
    fn doctor_probe_rejects_project_scripts_and_provisioning_shims() -> Result<()> {
        use std::path::{Path, PathBuf};
        let roots = [PathBuf::from("/project")];
        for path in [
            "/project/bin/rustc",
            "/home/user/.cargo/bin/rustc",
            "/tools/shims/rustc",
            "rustc",
        ] {
            assert!(
                DependencyProbe::RustCompiler
                    .validate_executable(Path::new(path), &roots, b"\x7fELF")
                    .is_err()
            );
        }
        assert!(
            DependencyProbe::Java
                .validate_executable(Path::new("/jdk/bin/java"), &roots, b"#!/bin/sh")
                .is_err()
        );
        assert!(
            DependencyProbe::Java
                .validate_executable(Path::new("/jdk/bin/javac"), &roots, b"\x7fELF")
                .is_err()
        );
        DependencyProbe::RustCompiler.validate_executable(
            Path::new("/tools/toolchains/stable/bin/rustc"),
            &roots,
            b"\xcf\xfa\xed\xfe",
        )?;
        assert_eq!(DependencyProbe::Java.version_arguments(), &["-version"]);
        Ok(())
    }

    #[test]
    fn doctor_probe_enforces_combined_output_limit() -> Result<()> {
        use std::sync::{Arc, atomic::AtomicUsize};
        let count = Arc::new(AtomicUsize::new(0));
        smol::block_on(async {
            read_probe_output(
                futures_lite::io::Cursor::new(vec![b'a'; MAX_PROBE_BYTES]),
                count.clone(),
                MAX_PROBE_BYTES,
            )
            .await?;
            assert!(
                read_probe_output(
                    futures_lite::io::Cursor::new(vec![b'b']),
                    count,
                    MAX_PROBE_BYTES
                )
                .await
                .is_err()
            );
            anyhow::Ok(())
        })
    }

    #[cfg(unix)]
    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "Real child-process tests need wall-clock cutoffs; seeded GPUI timer tests cover virtual time separately"
    )]
    fn doctor_probe_observes_real_process_success_failure_and_output_limit() -> Result<()> {
        smol::block_on(async {
            for (program, expected) in [
                ("/usr/bin/true", ProbeEnd::Completed),
                ("/usr/bin/false", ProbeEnd::Failed),
            ] {
                let stop = async {
                    smol::Timer::after(std::time::Duration::from_secs(5)).await;
                    ProbeEnd::TimedOut
                };
                assert_eq!(
                    run_probe_until_stop(std::process::Command::new(program), stop)
                        .await?
                        .end,
                    expected
                );
            }
            let mut command = std::process::Command::new("/usr/bin/yes");
            command.arg("bounded-sentinel");
            let stop = async {
                smol::Timer::after(std::time::Duration::from_secs(5)).await;
                ProbeEnd::TimedOut
            };
            assert_eq!(
                run_probe_until_stop(command, stop).await?.end,
                ProbeEnd::OutputLimit
            );
            anyhow::Ok(())
        })
    }

    #[cfg(unix)]
    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "Real child-process tests need wall-clock cutoffs; seeded GPUI timer tests cover virtual time separately"
    )]
    fn doctor_probe_terminates_real_owned_process_on_timeout_and_cancel() -> Result<()> {
        smol::block_on(async {
            for end in [ProbeEnd::TimedOut, ProbeEnd::Cancelled] {
                let mut command = std::process::Command::new("/bin/sleep");
                command.arg("30");
                let stop = async move {
                    smol::Timer::after(std::time::Duration::from_millis(20)).await;
                    end
                };
                let result = run_probe_until_stop(command, stop).await?;
                assert_eq!(result.end, end);
                assert!(result.detail.as_str().contains("unverified"));
            }
            anyhow::Ok(())
        })
    }

    #[gpui::test]
    async fn doctor_probe_setup_is_bounded_cancellable_and_preserves_errors(
        cx: &mut gpui::TestAppContext,
    ) {
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = prepare_probe_until_stop(
            futures::future::pending::<Result<()>>(),
            cancelled.clone(),
            cx.background_executor.clone(),
        )
        .await
        .expect("Deadline result");
        assert_eq!(result, Err(ProbeEnd::TimedOut));
        let polled = std::rc::Rc::new(std::cell::Cell::new(false));
        cancel.close();
        let result = prepare_probe_until_stop(
            async {
                polled.set(true);
                Ok(())
            },
            cancelled,
            cx.background_executor.clone(),
        )
        .await
        .expect("Pre-cancel result");
        assert_eq!(result, Err(ProbeEnd::Cancelled));
        assert!(
            !polled.get(),
            "Cancelled setup must not initialize execution"
        );
        let (_cancel, cancelled) = async_channel::bounded(1);
        assert_eq!(
            prepare_probe_until_stop(
                async { Ok(42) },
                cancelled.clone(),
                cx.background_executor.clone()
            )
            .await
            .expect("Prepared value"),
            Ok(42)
        );
        assert!(
            prepare_probe_until_stop(
                async { Err::<(), _>(anyhow::anyhow!("Preparation failed")) },
                cancelled,
                cx.background_executor.clone()
            )
            .await
            .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_probe_uses_deterministic_deadline_and_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        let (cancel, cancelled) = async_channel::bounded(1);
        assert_eq!(
            wait_for_probe_stop(
                cancelled.clone(),
                cx.background_executor.clone(),
                std::time::Duration::from_secs(2)
            )
            .await,
            ProbeEnd::TimedOut
        );
        cancel.close();
        assert_eq!(
            wait_for_probe_stop(
                cancelled.clone(),
                cx.background_executor.clone(),
                std::time::Duration::from_secs(2)
            )
            .await,
            ProbeEnd::Cancelled
        );
        let result = run_bounded_probe(
            std::process::Command::new("/does-not-exist"),
            cx.background_executor.clone(),
            cancelled,
            std::time::Duration::from_secs(5),
        )
        .await
        .expect("Cancelled probe never spawns");
        assert_eq!(result.end, ProbeEnd::Cancelled);
    }

    #[gpui::test]
    async fn doctor_native_repair_history_closed_view_preserves_session_without_false_verification()
    {
        for operation in [
            DoctorRepairOperationOutcome::Cancelled,
            DoctorRepairOperationOutcome::Completed,
            DoctorRepairOperationOutcome::Failed,
            DoctorRepairOperationOutcome::AwaitingUserAction,
        ] {
            let database = db::AppDatabase::test_new();
            let store = DoctorHistoryStore::from_database(&database);
            let receipt = store
                .record(
                    DoctorProduct::Copper,
                    DoctorHistoryRecord::new(
                        report(),
                        Some(DiagnosticText::new("Native repair")),
                        ReceiptOutcome::Running,
                        10,
                    ),
                    10,
                )
                .await
                .expect("Durable approval");
            store
                .link_native_session(DoctorProduct::Copper, receipt, "native-session", 11)
                .await
                .expect("Session persisted before prompt");
            store
                .finish_unobserved_native_session(
                    DoctorProduct::Copper,
                    receipt,
                    Ok(DoctorNativeRepairResult {
                        session_id: Some("native-session".into()),
                        outcome: operation,
                        detail: DiagnosticText::new("API_KEY=never-store-the-secret observed turn"),
                    }),
                    12,
                )
                .await
                .expect("Caller closure receipt");
            let records = store
                .records(DoctorProduct::Copper, 12)
                .await
                .expect("Existing history owner");
            let record = records.first().expect("Terminal record");
            assert_eq!(
                record.session_id.as_ref().map(DiagnosticText::as_str),
                Some("native-session")
            );
            assert_eq!(
                record.outcome,
                match operation {
                    DoctorRepairOperationOutcome::Cancelled => ReceiptOutcome::Cancelled,
                    DoctorRepairOperationOutcome::AwaitingUserAction =>
                        ReceiptOutcome::AwaitingUserAction,
                    _ => ReceiptOutcome::Incomplete,
                }
            );
            assert!(
                !serde_json::to_string(record)
                    .expect("Bounded report")
                    .contains("never-store-the-secret")
            );
            store
                .finish_unobserved_native_session(
                    DoctorProduct::Copper,
                    receipt,
                    Err(anyhow::anyhow!("Repeated reply")),
                    13,
                )
                .await
                .expect("Repeated reply cannot rewrite terminal outcome");
            assert_eq!(
                store
                    .records(DoctorProduct::Copper, 13)
                    .await
                    .expect("Terminal remains")
                    .first()
                    .expect("Record")
                    .outcome,
                record.outcome
            );
            let reopened = DoctorHistoryStore::from_database(&database);
            assert_eq!(
                reopened
                    .records(DoctorProduct::Copper, 14)
                    .await
                    .expect("Restart observes terminal receipt without replay")
                    .first()
                    .expect("Record")
                    .outcome,
                record.outcome
            );
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) async fn doctor_selected_java_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        gpui::WindowHandle<workspace::MultiWorkspace>,
        project::structured_execution::StructuredNodeId,
        project::structured_execution::DiscoveryGeneration,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        for path in [
            "/doctor-restart/src/test/java",
            "/doctor-restart/target/surefire-reports",
        ] {
            filesystem
                .create_dir(std::path::Path::new(path))
                .await
                .expect("Development fixture directory");
        }
        for (path, contents) in [
            ("/doctor-restart/pom.xml", "<project/>"),
            ("/doctor-restart/mvnw", "#!/bin/sh\n"),
            (
                "/doctor-restart/src/test/java/GreetingTest.java",
                "class GreetingTest { @Test messageIsAvailable() {} }",
            ),
            (
                "/doctor-restart/target/surefire-reports/TEST-GreetingTest.xml",
                "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>",
            ),
        ] {
            filesystem
                .atomic_write(std::path::PathBuf::from(path), contents.into())
                .await
                .expect("Native-owner fixture input");
        }
        let java = language::Language::new(language::LanguageConfig {
            name: "Java".into(), matcher: language::LanguageMatcher { path_suffixes: vec!["java".into()], ..Default::default() }.into(), ..Default::default()
        }, Some(language::tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())).with_queries(language::LanguageQueries {
            runnables: Some(std::borrow::Cow::Borrowed("(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))")), ..Default::default()
        }).expect("Existing Java test-owner runnable fixture query");
        project
            .read_with(cx, |owner, _| owner.languages().clone())
            .add(std::sync::Arc::new(java));
        cx.run_until_parked();
        let provider = project
            .read_with(cx, |owner, _| owner.jvm_test_provider_store().cloned())
            .expect("Actual JVM test owner");
        let snapshot = provider
            .update(cx, |owner, cx| owner.refresh(cx))
            .await
            .expect("Native selected test discovery");
        let node = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "messageIsAvailable")
            .expect("Verified source-backed method")
            .id
            .clone();
        (
            project,
            directory,
            window,
            node,
            snapshot.discovery_generation,
        )
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_reproduction_selected_java_observes_fresh_assertion_without_raw_messages(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::{Arc, atomic::AtomicUsize};
        let (project, directory, window, node, discovery) = doctor_selected_java_fixture(cx).await;
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(1);
        let (completion, results) = async_channel::bounded(1);
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: Arc::new(AtomicUsize::new(0)),
            })
        });
        let provider = project::structured_execution::StructuredProviderId(
            project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
        );
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Application;
        selection.problem = DoctorInvestigationProblem::Test;
        let mut review = cx
            .update(|cx| {
                prepare_test_reproduction(
                    project.clone(),
                    directory.clone(),
                    provider,
                    node,
                    discovery,
                    selection,
                    reproduction_effects(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .await
            .expect("Actual owner plan, no Task launch");
        assert!(launches.is_empty());
        assert_eq!(
            review.action.task.resolved.command.as_deref(),
            Some("./mvnw")
        );
        assert_eq!(
            review.action.task.resolved.args,
            vec!["-Dtest=GreetingTest#messageIsAvailable", "test"]
        );
        review
            .approve(
                &cx.update(|cx| project_repair_binding(&project, directory, 1, cx))
                    .expect("Authority"),
            )
            .expect("Explicit exact execution review");
        let mut source = report();
        source.scope.product = DiagnosticText::new("Groovy");
        source.push(review.before.clone()).expect("Report");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Groovy,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        launches
            .recv()
            .await
            .expect("Existing Task owner launched selected JUnit method");
        project.read_with(cx, |owner, _| owner.fs().clone()).atomic_write(std::path::PathBuf::from("/doctor-restart/target/surefire-reports/TEST-GreetingTest.xml"), "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"PRIVATE_JUNIT_MESSAGE\"/></testcase></testsuite>".into()).await.expect("Fresh observed host result");
        completion
            .send(0)
            .await
            .expect("Task success is independent of the assertion failure");
        cx.run_until_parked();
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        let result = run.await.expect("Attributable native JUnit observation");
        assert_eq!(result.evidence.owner, DoctorEvidenceOwner::Test);
        assert!(matches!(
            result.evidence.observation,
            DoctorEvidenceObservation::FunctionalFailure { .. }
        ));
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        let retained = history
            .records(DoctorProduct::Groovy, 11)
            .await
            .expect("Redacted history");
        let encoded = serde_json::to_string(&retained).expect("Bounded serialization");
        assert!(!encoded.contains("PRIVATE_JUNIT_MESSAGE") && !encoded.contains("<testsuite>"));
        assert!(
            retained[0]
                .report
                .investigation()
                .expect("Evidence")
                .evidence()[0]
                .reproduced_failure()
        );
        assert_ne!(
            retained[0].outcome,
            ReceiptOutcome::Completed,
            "Reproduction is not a verified repair"
        );
        cancel.close();
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_reproduction_selected_test_rejects_active_owner_and_advisory_only_results(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _window, node, discovery) = doctor_selected_java_fixture(cx).await;
        let provider = project::structured_execution::StructuredProviderId(
            project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.into(),
        );
        let review = cx
            .update(|cx| {
                prepare_test_reproduction(
                    project.clone(),
                    directory.clone(),
                    provider,
                    node.clone(),
                    discovery,
                    investigation_selection(),
                    reproduction_effects(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .await
            .expect("Native selected plan");
        let DoctorTestPlan::Java(plan) = review.action.test_plan.as_ref().expect("Native plan")
        else {
            panic!("Expected actual Java owner plan")
        };
        let owner = project
            .read_with(cx, |owner, _| owner.jvm_test_provider_store().cloned())
            .expect("Owner");
        let version = owner
            .update(cx, |owner, cx| {
                owner.capture_report_version_with_limit(plan, 1024, cx)
            })
            .await
            .expect("Bounded current report");
        assert!(
            owner
                .update(cx, |owner, cx| owner
                    .capture_report_version_with_limit(plan, 1, cx))
                .await
                .is_err()
        );
        let handle = task::StructuredTaskHandle::new(task::TaskId("unrelated-native-test".into()));
        owner
            .update(cx, |owner, cx| {
                owner.register_run_handle_with_limit(
                    plan.clone(),
                    version,
                    handle.clone(),
                    1024,
                    cx,
                )
            })
            .expect("Unrelated existing run");
        assert!(
            cx.update(|cx| review.action.revalidate_inputs(&project, &directory, cx))
                .is_err()
        );
        assert!(
            !handle.state().is_terminal(),
            "Doctor did not replace or cancel unrelated Tests"
        );
        cx.update(|cx| {
            handle.cancel(cx);
        });
        cx.run_until_parked();
    }

    #[cfg(unix)]
    fn doctor_task_exit_status(code: i32) -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt as _;
        std::process::ExitStatus::from_raw(code << 8)
    }
    #[cfg(windows)]
    fn doctor_task_exit_status(code: i32) -> std::process::ExitStatus {
        use std::os::windows::process::ExitStatusExt as _;
        std::process::ExitStatus::from_raw(code as u32)
    }

    pub(crate) struct DoctorTaskFixture {
        pub(crate) started: async_channel::Sender<task::SpawnInTerminal>,
        pub(crate) completion: async_channel::Receiver<i32>,
        pub(crate) cancel_completion: async_channel::Sender<i32>,
        pub(crate) cancellation_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl workspace::TerminalProvider for DoctorTaskFixture {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut gpui::Window,
            _: &mut gpui::App,
        ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
            panic!("Doctor must use the existing cancellable structured Task path")
        }
        fn spawn_structured(
            &self,
            task: task::SpawnInTerminal,
            handle: task::StructuredTaskHandle,
            _: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
            use gpui::AppContext as _;
            self.started
                .try_send(task)
                .expect("Bounded selected Task observation");
            let cancelled = self.cancel_completion.clone();
            let count = self.cancellation_count.clone();
            handle.bind_terminal(
                task::StructuredTerminalId(1),
                move |_| {
                    count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cancelled.close();
                    true
                },
                cx,
            );
            let completion = self.completion.clone();
            cx.background_spawn(async move {
                completion
                    .recv()
                    .await
                    .ok()
                    .map(|code| Ok(doctor_task_exit_status(code)))
            })
        }
    }

    fn reproduction_task(root: &str) -> task::ResolvedTask {
        task::TaskTemplate {
            label: "Selected regression".into(),
            command: "cargo".into(),
            args: vec![
                "test".into(),
                "--offline".into(),
                "selected_regression".into(),
            ],
            cwd: Some(root.into()),
            save: task::SaveStrategy::All,
            ..Default::default()
        }
        .resolve_task("doctor-existing-task", &task::TaskContext::default())
        .expect("Existing Task resolution")
    }

    fn reproduction_effects() -> DoctorReproductionEffects {
        DoctorReproductionEffects::Development {
            file_writes: "project target and development test outputs".into(),
            subprocesses: "selected existing Cargo build/test subprocesses; no external services"
                .into(),
        }
    }

    #[cfg(feature = "rust-tools")]
    struct DoctorCargoFixtureRunner;

    #[cfg(feature = "rust-tools")]
    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoMetadataRunner for DoctorCargoFixtureRunner {
        async fn run(
            &self,
            _: project::cargo_workspace_store::CargoMetadataRequest,
        ) -> Result<Vec<u8>> {
            Ok(
                include_str!("../../project/test_data/cargo_workspace/workspace-v1.json")
                    .replace("/workspace", "/doctor-restart")
                    .into_bytes(),
            )
        }
    }

    #[cfg(feature = "rust-tools")]
    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoConfigurationProbe for DoctorCargoFixtureRunner {
        async fn run(
            &self,
            _: project::cargo_workspace_store::CargoConfigurationProbeRequest,
        ) -> Result<Vec<u8>> {
            anyhow::bail!("Compiler evidence is not produced by this deterministic fixture")
        }
    }

    #[cfg(feature = "rust-tools")]
    impl project::rust_test_provider::RustTestDiscoveryRunner for DoctorCargoFixtureRunner {
        fn discover(
            &self,
            _: project::rust_test_provider::RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<
            'static,
            Result<project::rust_test_provider::RustTestDiscoveryCapture>,
        > {
            Box::pin(async {
                anyhow::bail!("No test executable is run by this dependency collection fixture")
            })
        }
    }

    #[cfg(feature = "rust-tools")]
    pub(crate) async fn doctor_cargo_collection_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        gpui::WindowHandle<workspace::MultiWorkspace>,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/Cargo.toml"),
                "[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n"
                    .into(),
            )
            .await
            .expect("Manifest fixture");
        for (name, version) in [("member-one", "0.1.0"), ("member-two", "0.2.0")] {
            filesystem
                .create_dir(
                    &std::path::PathBuf::from("/doctor-restart")
                        .join(name)
                        .join("src"),
                )
                .await
                .expect("Existing member");
            filesystem
                .atomic_write(
                    std::path::PathBuf::from("/doctor-restart")
                        .join(name)
                        .join("Cargo.toml"),
                    format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n"),
                )
                .await
                .expect("Manifest");
            filesystem
                .atomic_write(
                    std::path::PathBuf::from("/doctor-restart")
                        .join(name)
                        .join("src/lib.rs"),
                    "preserved member source".into(),
                )
                .await
                .expect("Source");
        }
        cx.run_until_parked();
        let (store, worktrees, environment, tests) = project.read_with(cx, |owner, _| {
            (
                owner.cargo_workspace_store().clone(),
                owner.worktree_store(),
                owner.environment().clone(),
                owner.rust_test_provider_store().clone(),
            )
        });
        let install_fixture = |cx: &mut gpui::TestAppContext| {
            store.update(cx, |store, _| {
                *store = project::cargo_workspace_store::CargoWorkspaceStore::local_with_runners(
                    worktrees.clone(),
                    environment.clone(),
                    std::sync::Arc::new(DoctorCargoFixtureRunner),
                    std::sync::Arc::new(DoctorCargoFixtureRunner),
                );
            });
        };
        install_fixture(cx);
        tests
            .update(cx, |store, _| {
                store.set_discovery_runner_for_test(std::sync::Arc::new(DoctorCargoFixtureRunner))
            })
            .expect("Existing fixture discovery runner");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        cx.run_until_parked();
        assert!(
            tests.read_with(cx, |owner, _| owner.discovery_idle_for_test()),
            "Initial owner discovery settled before proposal"
        );
        install_fixture(cx);
        (project, directory, window)
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) async fn doctor_jvm_collection_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        gpui::WindowHandle<workspace::MultiWorkspace>,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/build.gradle"),
                "plugins { id 'java' }".into(),
            )
            .await
            .expect("Current build fixture");
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/settings.gradle"),
                "rootProject.name = 'doctor-fixture'".into(),
            )
            .await
            .expect("Current settings fixture");
        filesystem
            .remove_file(
                std::path::Path::new("/doctor-restart/Cargo.toml"),
                fs::RemoveOptions::default(),
            )
            .await
            .expect("Remove only unrelated synthetic Cargo declaration");
        cx.run_until_parked();
        (project, directory, window)
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_jvm_native_task_report_and_owner_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = doctor_jvm_collection_fixture(cx).await;
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(1);
        let (completion, results) = async_channel::bounded(1);
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })
        });
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Dependencies;
        let mut review = cx
            .update(|cx| {
                prepare_jvm_dependency_collection(
                    &project,
                    directory.clone(),
                    task::TaskArtifact {
                        path: "jvm-report.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 65536,
                    },
                    project::jvm_project::JvmReviewedDependencyFormat::GradleResolution { project_path: ":".into(), configuration: "runtimeClasspath".into() },
                    selection,
                    DoctorReproductionEffects::Development { file_writes: "New selected graph and existing Gradle development caches".into(), subprocesses: "Installed Gradle 8.13, offline dependency resolution, build scripts and JVM processes; no wrapper bootstrap or network".into() },
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Typed exact owner review");
        let command = review
            .action
            .task
            .resolved
            .command
            .as_deref()
            .expect("Exact Task");
        assert!(command.starts_with("gradle --offline "));
        assert!(command.contains("gradle --offline --no-daemon --console=plain --init-script"));
        assert!(command.contains(":_madeDoctorDependencies"));
        assert_eq!(
            review.action.task.resolved.shell,
            util::shell::Shell::Program("/bin/sh".into())
        );
        review
            .approve(
                &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                    .expect("Binding"),
            )
            .expect("Per-action consent");
        let mut source = report();
        source.scope.product = DiagnosticText::new("Groovy");
        source.push(review.before.clone()).expect("Report");
        let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Groovy,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        let run = match futures::future::select(Box::pin(launches.recv()), run).await {
            futures::future::Either::Left((launch, run)) => {
                launch.expect("Existing native Task path");
                run
            }
            futures::future::Either::Right((result, _)) => match result {
                Err(error) => panic!("Reviewed operation failed before Task dispatch: {error}"),
                Ok(result) => panic!(
                    "Reviewed operation ended before Task dispatch in {:?}: {:?}",
                    result.review.phase(),
                    result.evidence.detail
                ),
            },
        };
        let raw = include_str!(
            "../../project/test_data/groovy_jvm_gradle/doctor-selected-dependencies-v1.json"
        )
        .to_string();
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/jvm-report.json"),
                raw,
            )
            .await
            .expect("Selected fixture Task output");
        cx.run_until_parked();
        completion.send(0).await.expect("Successful Task exit");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        let result = run.await.expect("Captured result");
        assert!(
            !result
                .review
                .action
                .dependency_collection
                .as_ref()
                .expect("Descriptor")
                .runtime_input_exists_for_test(),
            "Successful Task completion removes only owned adapter input"
        );
        let captured = result
            .dependency_report
            .expect("Bounded owner-bound report");
        let owner_workspace = cx
            .update(|cx| captured.admit_jvm(project.clone(), cx))
            .await
            .expect("Existing JVM owner admission");
        assert_eq!(owner_workspace.nodes.len(), 4);
        assert!(owner_workspace.partial && !owner_workspace.repository_origins_known);
        project.read_with(cx, |project, cx| {
            assert!(
                project
                    .lsp_store()
                    .read(cx)
                    .reviewed_jvm_dependency_snapshot(
                        &directory,
                        &project::jvm_project::JvmReviewedDependencyFormat::GradleResolution {
                            project_path: ":".into(),
                            configuration: "runtimeClasspath".into()
                        },
                        cx
                    )
                    .is_some()
            );
        });
        assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
        assert_eq!(
            history
                .records(DoctorProduct::Groovy, 12)
                .await
                .expect("History")[0]
                .outcome,
            ReceiptOutcome::Incomplete
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source"),
            "unrelated"
        );
        cancel.close();
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_jvm_late_result_rejects_changed_review_authority(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _) = doctor_jvm_collection_fixture(cx).await;
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Dependencies;
        let review = cx
            .update(|cx| {
                prepare_jvm_dependency_collection(
                    &project,
                    directory.clone(),
                    task::TaskArtifact {
                        path: "late-report.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 65536,
                    },
                    project::jvm_project::JvmReviewedDependencyFormat::GradleResolution { project_path: ":".into(), configuration: "runtimeClasspath".into() },
                    selection,
                    DoctorReproductionEffects::Development { file_writes: "New selected graph and existing Gradle development caches".into(), subprocesses: "Installed Gradle 8.13, offline dependency resolution, build scripts and JVM processes; no wrapper bootstrap or network".into() },
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Bounded proposal");
        let collection = review
            .action
            .dependency_collection
            .as_ref()
            .expect("Descriptor");
        let raw = include_str!(
            "../../project/test_data/groovy_jvm_gradle/doctor-selected-dependencies-v1.json"
        )
        .to_string();
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/late-report.json"),
                raw,
            )
            .await
            .expect("Transient fixture report");
        cx.run_until_parked();
        let result = cx
            .update(|cx| {
                collection.capture(
                    &project,
                    &directory,
                    review.binding.clone(),
                    uuid::Uuid::new_v4(),
                    cx,
                )
            })
            .expect("Confined capture")
            .await
            .expect("Report");
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"vim_mode":true}"#, cx)
                    .expect("Revoked review inputs");
            })
        });
        assert!(
            cx.update(|cx| result.admit_jvm(project.clone(), cx))
                .await
                .is_err()
        );
        project.read_with(cx, |project, cx| {
            assert!(
                project
                    .lsp_store()
                    .read(cx)
                    .reviewed_jvm_dependency_snapshot(
                        &directory,
                        &project::jvm_project::JvmReviewedDependencyFormat::GradleResolution {
                            project_path: ":".into(),
                            configuration: "runtimeClasspath".into()
                        },
                        cx
                    )
                    .is_none()
            );
        });
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_native_task_report_and_owner_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = doctor_cargo_collection_fixture(cx).await;
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(1);
        let (completion, results) = async_channel::bounded(1);
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })
        });
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Dependencies;
        let mut review = cx
            .update(|cx| {
                prepare_cargo_dependency_collection(
                    &project,
                    directory.clone(),
                    task::TaskArtifact {
                        path: "cargo-report.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 65536,
                    },
                    selection,
                    reproduction_effects(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Typed exact owner review");
        let command = review
            .action
            .task
            .resolved
            .command
            .as_deref()
            .expect("Exact Task");
        assert!(command.starts_with("set -C; cargo metadata "));
        assert!(command.contains("--locked --offline > "));
        assert_eq!(
            review.action.task.resolved.shell,
            util::shell::Shell::Program("/bin/sh".into())
        );
        review
            .approve(
                &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                    .expect("Binding"),
            )
            .expect("Per-action consent");
        let mut source = report();
        source.push(review.before.clone()).expect("Report");
        let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        launches.recv().await.expect("Existing native Task path");
        let raw = include_str!("../../project/test_data/cargo_workspace/workspace-v1.json")
            .replace("/workspace", "/doctor-restart");
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/cargo-report.json"),
                raw,
            )
            .await
            .expect("Selected fixture Task output");
        cx.run_until_parked();
        completion.send(0).await.expect("Successful Task exit");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        let result = run.await.expect("Captured result");
        let captured = result
            .dependency_report
            .expect("Bounded owner-bound report");
        let owner_workspace = cx
            .update(|cx| captured.admit_cargo(project.clone(), cx))
            .await
            .expect("Existing Cargo cache admission");
        assert_eq!(owner_workspace.members.len(), 2);
        assert!(owner_workspace.dependency_graph.is_some());
        project.read_with(cx, |project, cx| {
            assert!(
                project
                    .cargo_workspace_store()
                    .read(cx)
                    .cached_snapshot(cx)
                    .is_some_and(|(_, current)| current)
            );
        });
        assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 12)
                .await
                .expect("History")[0]
                .outcome,
            ReceiptOutcome::Incomplete
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source"),
            "unrelated"
        );
        cancel.close();
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_late_result_rejects_changed_review_authority(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _) = doctor_cargo_collection_fixture(cx).await;
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Dependencies;
        let review = cx
            .update(|cx| {
                prepare_cargo_dependency_collection(
                    &project,
                    directory.clone(),
                    task::TaskArtifact {
                        path: "late-report.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 65536,
                    },
                    selection,
                    reproduction_effects(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Bounded proposal");
        let collection = review
            .action
            .dependency_collection
            .as_ref()
            .expect("Descriptor");
        let raw = include_str!("../../project/test_data/cargo_workspace/workspace-v1.json")
            .replace("/workspace", "/doctor-restart");
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/late-report.json"),
                raw,
            )
            .await
            .expect("Transient fixture report");
        cx.run_until_parked();
        let result = cx
            .update(|cx| {
                collection.capture(
                    &project,
                    &directory,
                    review.binding.clone(),
                    uuid::Uuid::new_v4(),
                    cx,
                )
            })
            .expect("Confined capture")
            .await
            .expect("Report");
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"vim_mode":true}"#, cx)
                    .expect("Revoked review inputs");
            })
        });
        assert!(
            cx.update(|cx| result.admit_cargo(project.clone(), cx))
                .await
                .is_err()
        );
        project.read_with(cx, |project, cx| {
            assert!(
                project
                    .cargo_workspace_store()
                    .read(cx)
                    .cached_snapshot(cx)
                    .is_none()
            );
        });
    }

    #[cfg(feature = "rust-tools")]
    fn dependency_collection_review(
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        name: &str,
        duration: u64,
        cx: &gpui::App,
    ) -> Result<DoctorRepairReview<DoctorReproductionAction>> {
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Dependencies;
        let selected_task = task::TaskTemplate {
            label: "Reviewed dependency report".into(),
            command: "cargo".into(),
            args: vec![
                "metadata".into(),
                "--format-version=1".into(),
                "--locked".into(),
                "--offline".into(),
                format!("> {name}"),
            ],
            cwd: Some("/doctor-restart".into()),
            ..Default::default()
        }
        .resolve_task("doctor-dependency-fixture", &task::TaskContext::default())
        .expect("Existing Task fixture");
        prepare_dependency_collection(
            project,
            directory.clone(),
            project::TaskSourceKind::UserInput,
            selected_task,
            selection,
            DoctorReproductionEffects::Development {
                file_writes: format!("New report {name}; existing development resolver caches"),
                subprocesses:
                    "Reviewed existing Cargo resolver and its existing environment; offline".into(),
            },
            task::TaskArtifact {
                path: name.into(),
                kind: task::TaskArtifactKind::Data,
                max_bytes: 1024,
            },
            crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata,
            duration,
            1,
            10,
            cx,
        )
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_approval_duplicate_success_and_redacted_history(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(4);
        let (completion, results) = async_channel::bounded(4);
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: count.clone(),
            })
        });
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let prepare = |cx: &gpui::App| {
            dependency_collection_review(&project, &directory, "report.json", 60, cx)
        };
        let proposal = cx.update(|cx| prepare(cx)).expect("Proposed collection");
        let mut source = report();
        source.push(proposal.before.clone()).expect("Report");
        let (cancel, cancelled) = async_channel::bounded(1);
        let unapproved = cx.update(|cx| {
            run_reproduction(
                proposal,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source.clone(),
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        assert!(unapproved.await.is_err(), "A proposal never grants consent");
        cancel.close();
        assert!(launches.is_empty());
        assert!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("No unapproved receipt")
                .is_empty()
        );

        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Binding");
        let mut review = cx.update(|cx| prepare(cx)).expect("Fresh review");
        review
            .approve(&binding)
            .expect("Concrete explicit approval");
        assert!(
            review
                .action
                .exact_preview()
                .iter()
                .any(|line| line.contains("no raw report"))
        );
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source.clone(),
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        launches.recv().await.expect("Existing Task launch");
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Durable before dispatch")[0]
                .outcome,
            ReceiptOutcome::Running
        );
        let mut duplicate = cx.update(|cx| prepare(cx)).expect("Duplicate proposal");
        duplicate
            .approve(&binding)
            .expect("Approval cannot bypass admission");
        let (duplicate_cancel, duplicate_cancelled) = async_channel::bounded(1);
        let duplicate_run = cx.update(|cx| {
            run_reproduction(
                duplicate,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Copper,
                history.clone(),
                10,
                duplicate_cancelled,
                cx,
            )
        });
        assert!(duplicate_run.await.is_err());
        duplicate_cancel.close();
        assert!(launches.is_empty());

        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        let raw = r#"{"version":1,"packages":[],"metadata":"PRIVATE_DEPENDENCY_SENTINEL"}"#;
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/report.json"),
                raw.into(),
            )
            .await
            .expect("Task output");
        cx.run_until_parked();
        completion.send(0).await.expect("Observed exit");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        let result = run.await.expect("Bounded report collection");
        let captured = result.dependency_report.expect("Transient raw report");
        assert_eq!(captured.bytes, raw.as_bytes());
        assert_eq!(captured.binding(), &binding);
        assert_eq!(captured.byte_count(), raw.len());
        assert_eq!(
            captured.task_receipt(),
            result.review.receipt().expect("Receipt")
        );
        assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 0);
        let records = DoctorHistoryStore::from_database(&database)
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Reopen");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, ReceiptOutcome::Incomplete);
        let export = records[0].report.export().expect("Explicit bounded export");
        assert!(!export.contains("PRIVATE_DEPENDENCY_SENTINEL"));
        assert!(!export.contains("packages"));
        assert!(!export.contains("/doctor-restart"));
        assert!(launches.is_empty(), "Reopen never replays a resolver");
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source"),
            "unrelated"
        );
        cancel.close();
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_missing_failed_malformed_oversized_and_changed_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        for scenario in [
            "missing",
            "failed",
            "malformed",
            "oversized",
            "wrong-format",
            "symlink",
            "changed",
        ] {
            let name = format!("{scenario}.json");
            let (started, launches) = async_channel::bounded(1);
            let (completion, results) = async_channel::bounded(1);
            workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(DoctorTaskFixture {
                    started,
                    completion: results,
                    cancel_completion: completion.clone(),
                    cancellation_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                })
            });
            let mut review = cx
                .update(|cx| dependency_collection_review(&project, &directory, &name, 60, cx))
                .expect("Review");
            review
                .approve(
                    &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                        .expect("Binding"),
                )
                .expect("Consent");
            let mut source = report();
            source.push(review.before.clone()).expect("Report");
            let (cancel, cancelled) = async_channel::bounded(1);
            let run = cx.update(|cx| {
                run_reproduction(
                    review,
                    project.clone(),
                    workspace.downgrade(),
                    window.into(),
                    source,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            });
            launches.recv().await.expect("Only reviewed Task");
            let path = std::path::Path::new("/doctor-restart").join(&name);
            match scenario {
                "missing" => {}
                "symlink" => filesystem
                    .create_symlink(&path, std::path::PathBuf::from("/doctor-restart/source.rs"))
                    .await
                    .expect("Untrusted output"),
                _ => {
                    let contents = match scenario {
                        "malformed" => "{".to_owned(),
                        "oversized" => "x".repeat(1025),
                        "wrong-format" => r#"{"version":2,"packages":[]}"#.into(),
                        _ => r#"{"version":1,"packages":[]}"#.into(),
                    };
                    filesystem
                        .atomic_write(path, contents)
                        .await
                        .expect("Output");
                }
            }
            if scenario == "changed" {
                filesystem
                    .atomic_write(
                        std::path::PathBuf::from("/doctor-restart/source.rs"),
                        "changed during collection".into(),
                    )
                    .await
                    .expect("Intervening edit");
            }
            cx.run_until_parked();
            completion
                .send(if scenario == "failed" { 1 } else { 0 })
                .await
                .expect("Task result");
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(200));
            let result = run.await.expect("Failure is reported");
            assert!(
                result.dependency_report.is_none(),
                "Invalid {scenario} report was admitted"
            );
            assert_eq!(
                result.evidence.observation,
                DoctorEvidenceObservation::Incomplete
            );
            assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
            cancel.close();
        }
        assert_eq!(
            history
                .records(DoctorProduct::Copper, 12)
                .await
                .expect("Bounded history")
                .len(),
            7
        );
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_cancel_timeout_revocation_and_no_replay(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        for scenario in ["cancel", "timeout", "revocation", "drop", "trust"] {
            let (started, launches) = async_channel::bounded(1);
            let (completion, results) = async_channel::bounded(1);
            let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(DoctorTaskFixture {
                    started,
                    completion: results,
                    cancel_completion: completion,
                    cancellation_count: count.clone(),
                })
            });
            let mut review = cx
                .update(|cx| {
                    dependency_collection_review(
                        &project,
                        &directory,
                        &format!("{scenario}.json"),
                        1,
                        cx,
                    )
                })
                .expect("Exact review");
            review
                .approve(
                    &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                        .expect("Binding"),
                )
                .expect("Consent");
            let mut source = report();
            source.push(review.before.clone()).expect("Report");
            let (cancel, cancelled) = async_channel::bounded(1);
            let mut pending = Some(cx.update(|cx| {
                run_reproduction(
                    review,
                    project.clone(),
                    workspace.downgrade(),
                    window.into(),
                    source,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            }));
            launches.recv().await.expect("Existing owner");
            match scenario {
                "cancel" => {
                    cancel.close();
                }
                "timeout" => cx
                    .executor()
                    .advance_clock(std::time::Duration::from_secs(2)),
                "revocation" => cx.update(|cx| {
                    cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                        store
                            .set_user_settings(r#"{"vim_mode":true}"#, cx)
                            .expect("Changed authority");
                    })
                }),
                "drop" => drop(pending.take()),
                "trust" => {
                    let trust = cx.update(|cx| {
                        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                            .expect("Trust owner")
                    });
                    let worktrees = project.read_with(cx, |owner, _| owner.worktree_store());
                    trust.update(cx, |owner, cx| {
                        owner.restrict(
                            worktrees.downgrade(),
                            collections::HashSet::from_iter([
                                project::trusted_worktrees::PathTrust::Worktree(
                                    directory.worktree_id,
                                ),
                            ]),
                            cx,
                        )
                    });
                }
                _ => unreachable!(),
            }
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(200));
            cx.run_until_parked();
            if let Some(run) = pending {
                let result = run.await.expect("Observed stop");
                assert!(result.dependency_report.is_none());
                assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
            }
            assert_eq!(
                count.load(std::sync::atomic::Ordering::Relaxed),
                1,
                "Only owned Task cancelled"
            );
            cancel.close();
        }
        let records = DoctorHistoryStore::from_database(&database)
            .records(DoctorProduct::Copper, 15)
            .await
            .expect("Recovered history");
        assert_eq!(records.len(), 5);
        assert!(records.iter().all(|record| !record.outcome.is_active()));
    }

    #[cfg(feature = "rust-tools")]
    #[gpui::test]
    async fn doctor_dependency_collection_preexisting_private_excluded_and_symlink_parent(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/existing.json"),
                "unrelated".into(),
            )
            .await
            .expect("Existing report");
        cx.run_until_parked();
        assert!(
            cx.update(|cx| dependency_collection_review(
                &project,
                &directory,
                "existing.json",
                60,
                cx
            ))
            .is_err()
        );
        for (key, path) in [
            ("private_files", "private.json"),
            ("file_scan_exclusions", "excluded.json"),
            ("read_only_files", "readonly.json"),
        ] {
            cx.update(|cx| {
                cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                    store
                        .set_user_settings(&format!(r#"{{"{key}":["{path}"]}}"#), cx)
                        .expect("Path policy");
                })
            });
            cx.run_until_parked();
            assert!(
                cx.update(|cx| dependency_collection_review(&project, &directory, path, 60, cx))
                    .is_err(),
                "{key} cannot authorize collection"
            );
        }
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.set_user_settings("{}", cx).expect("Reset fixture");
            })
        });
        cx.run_until_parked();
        for name in ["../outside.json", "/absolute.json"] {
            assert!(
                cx.update(|cx| dependency_collection_review(&project, &directory, name, 60, cx))
                    .is_err()
            );
        }
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(1);
        let (completion, results) = async_channel::bounded(1);
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion,
                cancellation_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })
        });
        let mut review = cx
            .update(|cx| {
                dependency_collection_review(&project, &directory, "parent/report.json", 60, cx)
            })
            .expect("Reviewed new path");
        review
            .approve(
                &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                    .expect("Binding"),
            )
            .expect("Consent");
        filesystem
            .create_symlink(
                std::path::Path::new("/doctor-restart/parent"),
                std::path::PathBuf::from("/"),
            )
            .await
            .expect("Hostile parent");
        let mut source = report();
        source.push(review.before.clone()).expect("Report");
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Copper,
                DoctorHistoryStore::from_database(&db::AppDatabase::test_new()),
                10,
                cancelled,
                cx,
            )
        });
        match run.await {
            Ok(result) => assert!(result.dependency_report.is_none()),
            Err(_) => {}
        }
        assert!(
            launches.is_empty(),
            "Symlink parent must block before Task dispatch"
        );
        cancel.close();
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/existing.json"))
                .await
                .expect("Preserved file"),
            "unrelated"
        );
    }

    #[gpui::test]
    async fn doctor_dependency_collection_disabled_products_reject_before_execution(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        for (enabled, format) in [
            (
                cfg!(feature = "rust-tools"),
                crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata,
            ),
            (
                cfg!(feature = "jvm-tools"),
                crate::doctor_dependencies::DoctorDependencyReportFormat::MavenSelectedTree,
            ),
        ] {
            if enabled {
                continue;
            }
            let mut selection = investigation_selection();
            selection.scope = DoctorInvestigationScope::Development;
            selection.problem = DoctorInvestigationProblem::Dependencies;
            assert!(
                cx.update(|cx| prepare_dependency_collection(
                    &project,
                    directory.clone(),
                    project::TaskSourceKind::UserInput,
                    reproduction_task("/doctor-restart"),
                    selection,
                    reproduction_effects(),
                    task::TaskArtifact {
                        path: "disabled.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 1024
                    },
                    format,
                    60,
                    1,
                    10,
                    cx
                ))
                .is_err()
            );
        }
    }

    #[gpui::test]
    async fn doctor_reproduction_selected_task_receipt_duplicate_and_exit_are_not_fix_proof(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::{Arc, atomic::AtomicUsize};
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let (started, launches) = async_channel::bounded(4);
        let (completion, results) = async_channel::bounded(4);
        let count = Arc::new(AtomicUsize::new(0));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: count.clone(),
            })
        });
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let prepare = |cx: &gpui::App| {
            prepare_reproduction(
                &project,
                directory.clone(),
                project::TaskSourceKind::UserInput,
                reproduction_task("/doctor-restart"),
                investigation_selection(),
                reproduction_effects(),
                60,
                1,
                10,
                cx,
            )
        };
        let mut review = cx.update(|cx| prepare(cx)).expect("Concrete proposal");
        assert!(
            review
                .action
                .exact_preview()
                .iter()
                .any(|line| line.contains("not a filesystem, network"))
        );
        assert_eq!(review.action.task.resolved.save, task::SaveStrategy::None);
        assert!(!review.action.task.resolved.show_rerun);
        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Authority");
        review.approve(&binding).expect("Explicit approval");
        let mut source = report();
        source.push(review.before.clone()).expect("Action report");
        let (cancel, cancelled) = async_channel::bounded(1);
        let run = cx.update(|cx| {
            run_reproduction(
                review,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source.clone(),
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        let launch = launches.recv().await.expect("Existing Task owner launched");
        assert!(launch.id.0.starts_with("made-doctor-"));
        assert_eq!(launch.command.as_deref(), Some("cargo"));
        assert_eq!(
            launch.args,
            vec!["test", "--offline", "selected_regression"]
        );
        assert!(launch.use_new_terminal);
        let retained = history
            .records(DoctorProduct::Copper, 10)
            .await
            .expect("Receipt before launch");
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].outcome, ReceiptOutcome::Running);
        let mut duplicate = cx.update(|cx| prepare(cx)).expect("Same proposed Task");
        duplicate
            .approve(&binding)
            .expect("Separate review cannot bypass admission");
        let (duplicate_cancel, duplicate_cancelled) = async_channel::bounded(1);
        let duplicate_run = cx.update(|cx| {
            run_reproduction(
                duplicate,
                project.clone(),
                workspace.downgrade(),
                window.into(),
                source,
                DoctorProduct::Copper,
                history.clone(),
                10,
                duplicate_cancelled,
                cx,
            )
        });
        match duplicate_run.await {
            Err(error) => assert!(error.to_string().contains("equivalent Doctor action")),
            Ok(_) => panic!("Equivalent active action must be rejected"),
        };
        duplicate_cancel.close();
        assert!(launches.is_empty(), "No duplicate terminal");
        completion
            .send(1)
            .await
            .expect("Selected observed failing command");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(100));
        let observed = run.await.expect("Reached Task result");
        assert_eq!(observed.review.phase(), DoctorRepairPhase::Incomplete);
        assert_eq!(
            observed.evidence.observation,
            DoctorEvidenceObservation::CommandExit { code: Some(1) }
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 0);
        let retained = history
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Observed receipt");
        assert_eq!(retained[0].outcome, ReceiptOutcome::Incomplete);
        let summary = retained[0]
            .report
            .investigation()
            .expect("Persisted bounded attribution");
        assert!(
            !summary.evidence()[0].reproduced_failure(),
            "A failing command is not a reproduced functional assertion"
        );
        assert!(
            !retained[0]
                .report
                .export()
                .expect("Explicit redacted export")
                .contains("/doctor-restart")
        );
        cancel.close();
        assert_eq!(
            project
                .read_with(cx, |owner, _| owner.fs().clone())
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source retained"),
            "unrelated"
        );
    }

    #[gpui::test]
    async fn doctor_reproduction_cancellation_timeout_revocation_and_dropped_caller(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (project, directory, window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        for (index, scenario) in ["cancel", "timeout", "revocation", "drop", "trust"]
            .into_iter()
            .enumerate()
        {
            let (started, launches) = async_channel::bounded(1);
            let (completion, results) = async_channel::bounded(1);
            let count = Arc::new(AtomicUsize::new(0));
            workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(DoctorTaskFixture {
                    started,
                    completion: results,
                    cancel_completion: completion,
                    cancellation_count: count.clone(),
                })
            });
            let generation = index as u64 + 1;
            let mut selected_task = reproduction_task("/doctor-restart");
            selected_task.resolved.args.push(scenario.into());
            let mut review = cx
                .update(|cx| {
                    prepare_reproduction(
                        &project,
                        directory.clone(),
                        project::TaskSourceKind::UserInput,
                        selected_task,
                        investigation_selection(),
                        reproduction_effects(),
                        1,
                        generation,
                        10,
                        cx,
                    )
                })
                .expect("Concrete reproduction");
            review
                .approve(
                    &cx.update(|cx| {
                        project_repair_binding(&project, directory.clone(), generation, cx)
                    })
                    .expect("Authority"),
                )
                .expect("Reviewed action");
            let mut source = report();
            source.generation = generation;
            source.push(review.before.clone()).expect("Report");
            let (cancel, cancelled) = async_channel::bounded(1);
            let run = cx.update(|cx| {
                run_reproduction(
                    review,
                    project.clone(),
                    workspace.downgrade(),
                    window.into(),
                    source,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            });
            launches.recv().await.expect("Only selected owner launched");
            let mut pending = Some(run);
            match scenario {
                "cancel" => {
                    cancel.close();
                }
                "timeout" => cx
                    .executor()
                    .advance_clock(std::time::Duration::from_secs(2)),
                "revocation" => cx.update(|cx| {
                    cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                        store
                            .set_user_settings("{\"vim_mode\":true}", cx)
                            .expect("Inputs changed");
                    })
                }),
                "drop" => drop(pending.take()),
                "trust" => {
                    let trust = cx.update(|cx| {
                        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                            .expect("Existing trust owner")
                    });
                    let worktree_store =
                        project.read_with(cx, |project, _| project.worktree_store());
                    trust.update(cx, |store, cx| {
                        store.restrict(
                            worktree_store.downgrade(),
                            collections::HashSet::from_iter([
                                project::trusted_worktrees::PathTrust::Worktree(
                                    directory.worktree_id,
                                ),
                            ]),
                            cx,
                        )
                    });
                    assert_eq!(
                        count.load(Ordering::Relaxed),
                        1,
                        "Trust revocation cancels before the monitor timer"
                    );
                }
                _ => unreachable!(),
            }
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(100));
            cx.run_until_parked();
            if let Some(run) = pending {
                let observed = run
                    .await
                    .expect("Cancellation reaches observed owner state");
                assert_ne!(observed.review.phase(), DoctorRepairPhase::Verified);
                assert_eq!(
                    observed.evidence.observation,
                    if scenario == "cancel" {
                        DoctorEvidenceObservation::Cancelled
                    } else {
                        DoctorEvidenceObservation::Incomplete
                    }
                );
            }
            assert_eq!(
                count.load(Ordering::Relaxed),
                1,
                "Only this Task's cancellation callback ran"
            );
            let retained = history
                .records(DoctorProduct::Copper, 13)
                .await
                .expect("Reached durable receipt");
            assert_eq!(retained.len(), index + 1);
            assert!(
                !retained
                    .last()
                    .expect("Current receipt")
                    .outcome
                    .is_active()
            );
            assert_ne!(
                retained.last().expect("Current receipt").outcome,
                ReceiptOutcome::Completed
            );
        }
        let reopened = DoctorHistoryStore::from_database(&database);
        assert_eq!(
            reopened
                .records(DoctorProduct::Copper, 14)
                .await
                .expect("No replay on reopen")
                .len(),
            5
        );
    }

    #[gpui::test]
    async fn doctor_reproduction_precancelled_terminal_has_no_environment_or_process_effects(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, _directory, _window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let handle =
            task::StructuredTaskHandle::new(task::TaskId("already-cancelled-doctor-task".into()));
        cx.update(|cx| {
            assert!(handle.cancel(cx));
        });
        let result = project
            .update(cx, |owner, cx| {
                owner.create_terminal_task_with_structured_handle(
                    reproduction_task("/doctor-restart").resolved,
                    Some(handle.clone()),
                    cx,
                )
            })
            .await;
        match result {
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("before environment or terminal preparation")
            ),
            Ok(_) => panic!("A cancelled Task cannot create a terminal"),
        }
        assert!(project.read_with(cx, |owner, _| owner.local_terminal_handles().is_empty()));
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Cancelled {
                termination_confirmed: true,
                ..
            }
        ));
    }

    #[gpui::test]
    async fn doctor_reproduction_review_rejects_unknown_capture_host_and_stale_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _window) = application_restart_fixture(cx).await;
        cx.run_until_parked();
        let task = reproduction_task("/doctor-restart");
        for effects in [
            DoctorReproductionEffects::Unknown,
            DoctorReproductionEffects::ExternalServices,
            DoctorReproductionEffects::ProductionOrDestructive,
            DoctorReproductionEffects::SensitiveCapture,
        ] {
            assert!(
                cx.update(|cx| prepare_reproduction(
                    &project,
                    directory.clone(),
                    project::TaskSourceKind::UserInput,
                    task.clone(),
                    investigation_selection(),
                    effects,
                    60,
                    1,
                    10,
                    cx
                ))
                .is_err()
            );
        }
        for duration in [0, 301] {
            assert!(
                cx.update(|cx| prepare_reproduction(
                    &project,
                    directory.clone(),
                    project::TaskSourceKind::UserInput,
                    task.clone(),
                    investigation_selection(),
                    reproduction_effects(),
                    duration,
                    1,
                    10,
                    cx
                ))
                .is_err()
            );
        }
        let foreign = reproduction_task("/other");
        assert!(
            cx.update(|cx| prepare_reproduction(
                &project,
                directory.clone(),
                project::TaskSourceKind::UserInput,
                foreign,
                investigation_selection(),
                reproduction_effects(),
                60,
                1,
                10,
                cx
            ))
            .is_err()
        );
        let review = cx
            .update(|cx| {
                prepare_reproduction(
                    &project,
                    directory.clone(),
                    project::TaskSourceKind::UserInput,
                    task,
                    investigation_selection(),
                    reproduction_effects(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Reviewed metadata");
        let filesystem = project.read_with(cx, |owner, _| owner.fs().clone());
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/doctor-restart/source.rs"),
                "user intervening edit".into(),
            )
            .await
            .expect("Intervening edit");
        cx.run_until_parked();
        assert!(
            cx.update(|cx| review.action.revalidate_inputs(&project, &directory, cx))
                .is_err()
        );
        project.update(cx, |owner, _| owner.mark_as_collab_for_testing());
        assert!(
            cx.update(|cx| prepare_reproduction(
                &project,
                directory,
                project::TaskSourceKind::UserInput,
                reproduction_task("/doctor-restart"),
                investigation_selection(),
                reproduction_effects(),
                60,
                1,
                10,
                cx
            ))
            .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_development_passive_scope_preserves_files_and_execution_owners(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/development", serde_json::json!({"Cargo.lock": "UNRELATED_LOCKFILE_CONTENT", "build.gradle": "Unrelated development edits", "file.rs": "fn main() {}"})).await;
        let project = project::Project::test(
            filesystem.clone(),
            [std::path::Path::new("/development")],
            cx,
        )
        .await;
        let directory = project.read_with(cx, |owner, cx| project::ProjectPath {
            worktree_id: owner
                .visible_worktrees(cx)
                .next()
                .expect("Selected root")
                .read(cx)
                .id(),
            path: util::rel_path::RelPath::empty_arc(),
        });
        cx.run_until_parked();
        #[cfg(feature = "rust-tools")]
        let before = project::cargo_workspace_store::local_cargo_process_calls_for_test();
        for product in [
            DoctorProduct::Copper,
            DoctorProduct::Groovy,
            DoctorProduct::Made,
        ] {
            let checks = cx
                .update(|cx| collect_development(&project, &directory, product, 10, cx))
                .expect("Passive selected-owner collection");
            assert!(
                checks
                    .iter()
                    .any(|check| check.id.as_str() == "development.timing"
                        && check.outcome == CheckOutcome::Untested)
            );
            assert!(
                checks
                    .iter()
                    .all(|check| !check.detail.as_str().contains("UNRELATED_LOCKFILE_CONTENT"))
            );
            #[cfg(not(feature = "rust-tools"))]
            if product == DoctorProduct::Copper {
                assert!(
                    checks
                        .iter()
                        .any(|check| check.id.as_str() == "development.rust.disabled"
                            && check.outcome == CheckOutcome::Unsupported)
                );
            }
            #[cfg(not(feature = "jvm-tools"))]
            if product == DoctorProduct::Groovy {
                assert!(
                    checks
                        .iter()
                        .any(|check| check.id.as_str() == "development.jvm.disabled"
                            && check.outcome == CheckOutcome::Unsupported)
                );
            }
        }
        #[cfg(feature = "rust-tools")]
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            before,
            "Passive Doctor collection never starts Cargo"
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/development/Cargo.lock"))
                .await
                .expect("Lockfile preserved"),
            "UNRELATED_LOCKFILE_CONTENT"
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/development/build.gradle"))
                .await
                .expect("Unrelated build edits preserved"),
            "Unrelated development edits"
        );
        assert!(project.read_with(cx, |owner, _| owner.local_terminal_handles().is_empty()));
        let nested = project::ProjectPath {
            path: util::rel_path::RelPath::from_unix_str("nested")
                .expect("Nested path")
                .into(),
            ..directory.clone()
        };
        assert!(
            cx.update(|cx| collect_development(&project, &nested, DoctorProduct::Made, 11, cx))
                .is_err()
        );
        let missing = project::ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(9999),
            ..directory.clone()
        };
        assert!(
            cx.update(|cx| collect_development(&project, &missing, DoctorProduct::Made, 11, cx))
                .is_err()
        );
        project.update(cx, |owner, _| owner.mark_as_collab_for_testing());
        let guest = cx
            .update(|cx| collect_development(&project, &directory, DoctorProduct::Made, 12, cx))
            .expect("Guest passive metadata");
        assert!(
            guest
                .first()
                .expect("Selected host")
                .detail
                .as_str()
                .contains("Collaboration guest — host checks unavailable")
        );
        assert!(
            project.read_with(cx, |owner, _| owner.local_terminal_handles().is_empty()),
            "Guest evidence never falls back to local execution"
        );
    }

    #[cfg(feature = "rust-tools")]
    #[test]
    fn doctor_development_cargo_declarations_resolution_staleness_and_bounds() {
        use project::cargo_workspace::*;
        let dependency = CargoDependencyModel {
            declaration_name: "serde".into(),
            rename: Some("local_alias".into()),
            kind: CargoDependencyKind::Normal,
            version_requirement: "^1".into(),
            optional: false,
            uses_default_features: true,
            requested_features: vec!["derive".into()],
            target: None,
            source_kind: CargoDependencySourceKind::Registry,
            resolved_name: Some("serde".into()),
            resolved_version: Some("1.0.0".into()),
            resolved_workspace_member: None,
            local_manifest: None,
            declaration_manifest: Some(copper_path(1, "Cargo.toml")),
            declaration_origin: CargoDependencyDeclarationOrigin::WorkspaceInherited,
            resolved_instances: vec![
                CargoResolvedDependencyModel {
                    name: "serde".into(),
                    version: "1.0.0".into(),
                    source_kind: CargoDependencySourceKind::Registry,
                    enabled_features: vec!["derive".into()],
                    lock_status: CargoDependencyLockStatus::Locked,
                    workspace_member: None,
                    local_manifest: None,
                },
                CargoResolvedDependencyModel {
                    name: "serde".into(),
                    version: "1.1.0".into(),
                    source_kind: CargoDependencySourceKind::Registry,
                    enabled_features: Vec::new(),
                    lock_status: CargoDependencyLockStatus::NotLocked,
                    workspace_member: None,
                    local_manifest: None,
                },
            ],
            resolution_truncated: false,
            feature_causality: CargoDependencyFeatureCausality::Ambiguous,
            cycle_detected: false,
        };
        let mut snapshot = CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 2,
            omitted: Default::default(),
            completeness: CargoSnapshotCompleteness::Complete,
            failures: Vec::new(),
            workspaces: vec![CargoWorkspaceModel {
                key: CargoWorkspaceKey {
                    root: copper_path(1, ""),
                },
                root_manifest: Some(copper_path(1, "Cargo.toml")),
                display_name: "fixture".into(),
                is_virtual: true,
                dependency_graph: None,
                configuration: CargoWorkspaceConfiguration::unresolved(),
                members: vec![CargoPackageModel {
                    id: "workspace-member".into(),
                    name: "fixture".into(),
                    version: "0.1.0".into(),
                    manifest_path: copper_path(1, "Cargo.toml"),
                    is_default_member: true,
                    targets: Vec::new(),
                    features: Vec::new(),
                    dependencies: vec![dependency.clone()],
                }],
            }],
        };
        let worktree = settings::WorktreeId::from_usize(1);
        let checks = development_cargo_dependency_checks(Some((&snapshot, true)), worktree, 10);
        let row = checks.first().expect("Dependency row");
        for expected in [
            "local_alias",
            "requirement ^1",
            "WorkspaceInherited",
            "serde 1.0.0 (Registry, Locked)",
            "serde 1.1.0 (Registry, NotLocked)",
        ] {
            assert!(row.detail.as_str().contains(expected), "Missing {expected}");
        }
        assert_eq!(
            row.outcome,
            CheckOutcome::Untested,
            "Cached facts are not a complete graph/security scan or verified fix"
        );
        assert!(
            development_cargo_dependency_checks(Some((&snapshot, false)), worktree, 11)
                .first()
                .expect("Stale dependency")
                .historical
        );
        let unrelated = development_cargo_dependency_checks(
            Some((&snapshot, true)),
            settings::WorktreeId::from_usize(2),
            11,
        );
        assert!(
            unrelated
                .iter()
                .all(|check| !check.detail.as_str().contains("local_alias"))
        );
        let package = snapshot
            .workspaces
            .first_mut()
            .expect("Workspace")
            .members
            .first_mut()
            .expect("Package");
        package.dependencies = vec![dependency; 30];
        let bounded = development_cargo_dependency_checks(Some((&snapshot, true)), worktree, 12);
        assert!(bounded.iter().any(|check| check.id.as_str()
            == "coverage.development.cargo.dependencies"
            && check.detail.as_str().contains("omitted")));
        assert!(
            bounded
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        assert_eq!(
            development_cargo_dependency_checks(None, worktree, 12)
                .first()
                .expect("Missing resolution")
                .outcome,
            CheckOutcome::Untested
        );
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_development_jvm_classpath_is_not_an_authoritative_build_graph() {
        use project::jvm_project::*;
        let root = project::ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(1),
            path: util::rel_path::RelPath::empty_arc(),
        };
        let java = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id: root.worktree_id,
            generation: 7,
            partial: true,
            projects: vec![JdtlsProject {
                membership: JdtlsProjectMembership::ImportedJava,
                imported_build_kind: JdtlsImportedBuildKind::Unknown,
                module_root: root.clone(),
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 1,
                library_files: Vec::new(),
                library_artifacts: vec![JdtlsLibraryArtifact {
                    coordinate: "org.example:library:1.2.3".into(),
                    scope: Some("test".into()),
                }],
                project_vm_known: true,
                project_vm_java_major: Some(17),
                server_launch_java_major: Some(21),
                java_source_level: Some(17),
                java_target_level: Some(17),
                gradle: None,
                maven: None,
            }],
        };
        let kotlin = KotlinWorkspaceSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(2),
            worktree_id: root.worktree_id,
            generation: 8,
            partial: true,
            projects: vec![KotlinWorkspaceProject {
                name: "module".into(),
                build_system: KotlinBuildSystem::Gradle,
                module_root: root,
                source_roots: Vec::new(),
                dependencies: vec![
                    KotlinWorkspaceDependency::Module {
                        name: "another-module".into(),
                        scope: None,
                    },
                    KotlinWorkspaceDependency::Artifact {
                        coordinate: "org.example:kotlin-library:2.0".into(),
                        scope: None,
                    },
                ],
                unidentified_libraries: 2,
            }],
        };
        let checks =
            development_jvm_dependency_checks(Some((&java, true)), Some((&kotlin, false)), 10);
        assert!(checks.iter().any(|check| {
            check.detail.as_str().contains("org.example:library:1.2.3")
                && check.detail.as_str().contains("partial true")
        }));
        assert!(checks.iter().any(|check| {
            check
                .detail
                .as_str()
                .contains("org.example:kotlin-library:2.0")
                && check.historical
        }));
        assert!(
            checks
                .iter()
                .all(|check| check.outcome != CheckOutcome::Ready)
        );
        let missing = development_jvm_dependency_checks(None, None, 11);
        assert!(
            missing
                .first()
                .expect("Incomplete import")
                .detail
                .as_str()
                .contains("not an empty")
        );
        assert_eq!(
            missing.first().expect("Missing evidence").outcome,
            CheckOutcome::Untested
        );
    }

    fn investigation_selection() -> DoctorInvestigationSelection {
        DoctorInvestigationSelection {
            input_identity: DoctorInvestigationInputIdentity::from_fields(&[
                "selected test",
                "fixture-1",
                "debug",
                "warm caches",
            ])
            .expect("Exact selected owner inputs"),
            scope: DoctorInvestigationScope::Application,
            problem: DoctorInvestigationProblem::Test,
            symptom: DiagnosticText::new("Selected regression fails"),
            validation_target: DiagnosticText::new("selected regression"),
            workload: DiagnosticText::new("one selected test"),
            configuration: DiagnosticText::new("debug / selected features"),
            source_revision: DiagnosticText::new("worktree revision fixture-1"),
            cache_conditions: DiagnosticText::new("existing warm development caches"),
        }
    }

    #[test]
    fn doctor_comparison_functional_identity_distinguishes_same_named_tests() {
        let first = functional_assertion_identity("junit", "module-one:case", "same name")
            .expect("First selected owner");
        let other = functional_assertion_identity("junit", "module-two:case", "same name")
            .expect("Different selected owner");
        assert_ne!(first, other);
        assert_eq!(
            first,
            functional_assertion_identity("junit", "module-one:case", "same name")
                .expect("Stable target")
        );
        assert_ne!(
            first,
            functional_assertion_identity("different provider", "module-one:case", "same name")
                .expect("Different provider")
        );
    }

    #[gpui::test]
    async fn doctor_comparison_deterministic_clock_and_repetition_scope(
        cx: &mut gpui::TestAppContext,
    ) {
        let (mut before, mut after, request) = comparison_fixture(cx, true);
        let executor = cx.background_executor.clone();
        for (snapshot, durations) in [(&mut before, [90, 100, 110]), (&mut after, [70, 80, 90])] {
            for (index, duration) in durations.into_iter().enumerate() {
                let started = executor.now();
                executor
                    .timer(std::time::Duration::from_millis(duration))
                    .await;
                let measured = executor.now().duration_since(started).as_millis();
                let row = snapshot
                    .summary
                    .evidence
                    .get_mut(index + 1)
                    .expect("Selected aggregate sample");
                if let DoctorEvidenceObservation::AggregateCounter { value, .. } =
                    &mut row.observation
                {
                    *value = DiagnosticText::new(measured.to_string());
                } else {
                    panic!("Expected owner counter");
                }
            }
        }
        let result = before
            .compare(&after, request, 30)
            .expect("Deterministic observed protocol");
        assert_eq!(
            result.outcome,
            DoctorComparisonOutcome::VerifiedSelectedTarget
        );
        assert_eq!(
            result
                .verification_median
                .as_ref()
                .map(DiagnosticText::as_str),
            Some("80")
        );
        before.source_identity = Some(
            DoctorInvestigationInputIdentity::from_fields(&["fixture source inputs"])
                .expect("Source metadata fixture"),
        );
        let unchanged = before.clone();
        assert!(
            before.combine_repetition(&unchanged).is_err(),
            "Duplicate samples cannot increase repetition count"
        );
        let mut unrelated = unchanged;
        unrelated.summary.selection.workload = DiagnosticText::new("unrelated workload");
        assert!(
            before.combine_repetition(&unrelated).is_err(),
            "Unrelated runs cannot supply comparable repetitions"
        );
    }

    #[gpui::test]
    async fn doctor_comparison_unsaved_editor_change_cannot_be_verified_as_task_repair(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, _servers, (buffer, _handle)) = project_repair_fixture(cx).await;
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .expect("Explicit fixture baseline save through existing owner");
        cx.run_until_parked();
        let (before_fixture, after_fixture, request) = comparison_fixture(cx, false);
        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Current authority");
        let before = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding: binding.clone(),
                    summary: before_fixture.summary,
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Sealed baseline");
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "unsaved proposed source repair\n")], None, cx)
        });
        assert!(buffer.read_with(cx, |buffer, _| buffer.is_dirty()));
        let after = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding,
                    summary: after_fixture.summary,
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Current owner observation fixture");
        let result = cx
            .update(|cx| before.compare_current(&after, request, 30, &project, cx))
            .expect("Incomplete comparison remains inspectable");
        assert_eq!(
            result.outcome,
            DoctorComparisonOutcome::Incomplete,
            "External Tasks cannot test unsaved editor content"
        );
        assert!(result.detail.as_str().contains("unsaved"));
        assert!(
            buffer.read_with(cx, |buffer, _| buffer.is_dirty()),
            "Diagnosis must not save or discard edits"
        );
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_verify_live_relevant_tests_saved_change_and_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
        let (project, directory, _servers, (buffer, _handle)) = project_repair_fixture(cx).await;
        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Current authority");
        let mut selected = investigation_selection();
        selected.scope = DoctorInvestigationScope::Development;
        selected.problem = DoctorInvestigationProblem::Dependencies;
        selected.input_identity = cx
            .update(|cx| comparison_source_identity(&project, &directory, cx))
            .expect("Saved baseline");
        let baseline = cx
            .update(|cx| {
                DoctorInvestigation::new(selected.clone(), report().scope, binding.clone(), 10)
                    .expect("Reviewed selected validation target")
                    .comparison_snapshot_at(&project, cx)
            })
            .expect("Saved baseline source seal");
        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(0..0, "reviewed dependency compatibility change\n")],
                None,
                cx,
            )
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .expect("Explicit saved change");
        cx.run_until_parked();
        let mut after_selection = selected.clone();
        after_selection.input_identity = cx
            .update(|cx| comparison_source_identity(&project, &directory, cx))
            .expect("Changed source");
        after_selection.source_revision = DiagnosticText::new("current changed source");
        let mut investigation =
            DoctorInvestigation::new(after_selection.clone(), report().scope, binding.clone(), 20)
                .expect("Current selected tests");
        let evidence_id = investigation
            .record(
                &binding,
                &after_selection,
                DoctorEvidenceInput {
                    owner: DoctorEvidenceOwner::Test,
                    owner_reference: DiagnosticText::new("relevant-owner-test:1"),
                    classification: DoctorEvidenceClass::Observed,
                    observation: DoctorEvidenceObservation::FunctionalSuccess {
                        assertion: DiagnosticText::new("selected provider assertion"),
                    },
                    detail: DiagnosticText::new("Observed selected regression"),
                    observed_at: 21,
                    historical: false,
                },
            )
            .expect("Relevant owner result");
        let snapshot = cx
            .update(|cx| investigation.comparison_snapshot_at(&project, cx))
            .expect("Current saved test input seal");
        let change = DoctorReviewedChange {
            before_input: selected.input_identity.clone(),
            after_input: after_selection.input_identity.clone(),
            description: DiagnosticText::new("Reviewed selected change"),
            session_id: Some(DiagnosticText::new("native-repair-session")),
        };
        let result = cx
            .update(|cx| {
                snapshot.verify_dependency_change_current(
                    &baseline,
                    &change,
                    evidence_id,
                    30,
                    &project,
                    cx,
                )
            })
            .expect("Current authority");
        assert_eq!(
            result.0,
            DoctorComparisonOutcome::VerifiedSelectedTarget,
            "Security baseline does not require a failed application test"
        );
        let mut unrelated = baseline.clone();
        unrelated.summary.selection.validation_target =
            DiagnosticText::new("different selected test");
        assert_eq!(
            cx.update(|cx| snapshot.verify_dependency_change_current(
                &unrelated,
                &change,
                evidence_id,
                30,
                &project,
                cx
            ))
            .expect("Inspectable mismatch")
            .0,
            DoctorComparisonOutcome::Incomplete
        );
        let mut historical = snapshot.clone();
        historical
            .summary
            .evidence
            .first_mut()
            .expect("Selected receipt")
            .historical = true;
        assert_eq!(
            historical
                .verify_dependency_change(&baseline, &change, evidence_id, 30)
                .expect("Historical receipt")
                .0,
            DoctorComparisonOutcome::Incomplete
        );
        let mut command = snapshot.clone();
        command
            .summary
            .evidence
            .first_mut()
            .expect("Selected receipt")
            .owner = DoctorEvidenceOwner::Task;
        command
            .summary
            .evidence
            .first_mut()
            .expect("Selected receipt")
            .observation = DoctorEvidenceObservation::CommandExit { code: Some(0) };
        assert_eq!(
            command
                .verify_dependency_change(&baseline, &change, evidence_id, 30)
                .expect("Command receipt")
                .0,
            DoctorComparisonOutcome::Incomplete
        );
        let mut cancelled = snapshot.clone();
        cancelled
            .summary
            .evidence
            .first_mut()
            .expect("Selected receipt")
            .observation = DoctorEvidenceObservation::Cancelled;
        assert_eq!(
            cancelled
                .verify_dependency_change(&baseline, &change, evidence_id, 30)
                .expect("Cancellation receipt")
                .0,
            DoctorComparisonOutcome::Cancelled
        );
        let mut failure = snapshot.clone();
        failure
            .summary
            .evidence
            .first_mut()
            .expect("Selected receipt")
            .observation = DoctorEvidenceObservation::FunctionalFailure {
            assertion: DiagnosticText::new("selected provider assertion"),
        };
        assert_eq!(
            failure
                .verify_dependency_change(&baseline, &change, evidence_id, 30)
                .expect("Failure receipt")
                .0,
            DoctorComparisonOutcome::CriterionNotMet
        );
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "independent unsaved edit\n")], None, cx)
        });
        assert!(
            cx.update(|cx| snapshot.verify_dependency_change_current(
                &baseline,
                &change,
                evidence_id,
                30,
                &project,
                cx
            ))
            .is_err(),
            "Source edits reject stale test evidence"
        );
        let trees = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                trees.downgrade(),
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        assert!(
            cx.update(|cx| snapshot.verify_dependency_change_current(
                &baseline,
                &change,
                evidence_id,
                30,
                &project,
                cx
            ))
            .is_err()
        );
        assert!(
            buffer.read_with(cx, |buffer, _| buffer.is_dirty()),
            "Verification preserves independent edits"
        );
    }

    #[gpui::test]
    async fn doctor_comparison_live_source_permissions_and_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
        let (project, directory, _servers, (buffer, _handle)) = project_repair_fixture(cx).await;
        let (before_fixture, after_fixture, request) = comparison_fixture(cx, false);
        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Actual trusted scope");
        let before = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding: binding.clone(),
                    summary: before_fixture.summary,
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Live baseline inputs");
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "reviewed source change\n")], None, cx)
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .expect("Explicit reviewed fixture save before fresh evidence");
        cx.run_until_parked();
        let after = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding: binding.clone(),
                    summary: after_fixture.summary,
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Live changed inputs");
        assert_eq!(
            cx.update(|cx| before.compare_current(&after, request.clone(), 30, &project, cx))
                .expect("Observed selected fixture target")
                .outcome,
            DoctorComparisonOutcome::VerifiedSelectedTarget
        );
        let mut historical = before.clone();
        historical.source_snapshot = None;
        assert_eq!(
            cx.update(|cx| historical.compare_current(&after, request.clone(), 30, &project, cx))
                .expect("Historical comparison remains visible")
                .outcome,
            DoctorComparisonOutcome::Incomplete
        );
        let mut unchanged = before.clone();
        unchanged.source_snapshot = after.source_snapshot.clone();
        unchanged.source_identity = after.source_identity.clone();
        assert_eq!(
            cx.update(|cx| unchanged.compare_current(&after, request.clone(), 30, &project, cx))
                .expect("Unchanged source receipt")
                .outcome,
            DoctorComparisonOutcome::Incomplete
        );
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "later unrelated edit\n")], None, cx)
        });
        assert!(
            cx.update(|cx| before.compare_current(&after, request.clone(), 30, &project, cx))
                .is_err(),
            "Later unsaved input changes require new relevant verification"
        );
        let after = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding: binding.clone(),
                    summary: after.summary.clone(),
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Fresh reviewed source snapshot");
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_server_settings("{}", cx)
                    .expect("Changed permission configuration")
            })
        });
        assert!(
            cx.update(|cx| before.compare_current(&after, request.clone(), 30, &project, cx))
                .is_err(),
            "Changed authority invalidates pending comparison"
        );
        let binding = cx
            .update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
            .expect("Fresh authority");
        let after = cx
            .update(|cx| {
                DoctorInvestigation {
                    binding,
                    summary: after.summary.clone(),
                }
                .comparison_snapshot_at(&project, cx)
            })
            .expect("Fresh current authority snapshot");
        cx.update(|cx| {
            let store = project.read(cx).worktree_store();
            TrustedWorktrees::try_get_global(cx)
                .expect("Trust owner")
                .update(cx, |trust, cx| {
                    trust.restrict(
                        store.downgrade(),
                        collections::HashSet::from_iter([PathTrust::Worktree(
                            directory.worktree_id,
                        )]),
                        cx,
                    )
                });
        });
        assert!(
            cx.update(|cx| before.compare_current(&after, request, 30, &project, cx))
                .is_err(),
            "Revoked project trust cannot verify a fix"
        );
    }

    #[gpui::test]
    async fn doctor_comparison_repetition_ignores_build_outputs_preserves_provenance_and_rejects_source_changes(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/series", serde_json::json!({".git": {}, ".gitignore": "target/\n", "main.rs": "source", "target": {"cache": "old output"}})).await;
        let project =
            project::Project::test(filesystem.clone(), [std::path::Path::new("/series")], cx).await;
        let directory = project.read_with(cx, |project, cx| project::ProjectPath {
            worktree_id: project
                .visible_worktrees(cx)
                .next()
                .expect("Root")
                .read(cx)
                .id(),
            path: util::rel_path::RelPath::empty_arc(),
        });
        cx.run_until_parked();
        let first = cx
            .update(|cx| comparison_source_identity(&project, &directory, cx))
            .expect("Bounded source metadata");
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/series/target/cache"),
                "new output".to_owned(),
            )
            .await
            .expect("Ignored build output");
        cx.run_until_parked();
        let second = cx
            .update(|cx| comparison_source_identity(&project, &directory, cx))
            .expect("Current source metadata");
        assert_eq!(
            first, second,
            "Ignored generated output is not a source change"
        );
        let (mut baseline, _, _) = comparison_fixture(cx, false);
        baseline.source_identity = Some(first);
        let mut repetition = baseline.clone();
        repetition.summary.selection.input_identity =
            DoctorInvestigationInputIdentity::from_fields(&["new per-attempt review identity"])
                .expect("Attempt identity");
        let row = repetition.summary.evidence.first_mut().expect("Owner row");
        row.id = uuid::Uuid::new_v4();
        row.owner_reference = DiagnosticText::new("independent repeated owner run");
        row.selection = repetition.summary.selection.clone();
        let original = repetition.summary.clone();
        baseline
            .combine_repetition(&repetition)
            .expect("Equivalent selected inputs may combine independent runs");
        assert_eq!(baseline.evidence().len(), 2);
        assert_eq!(
            repetition.summary, original,
            "Individual owner evidence remains immutable"
        );
        baseline
            .summary
            .validate()
            .expect("Bounded combined context");
        assert!(
            baseline.combine_repetition(&repetition).is_err(),
            "Duplicate runs do not increase sample count"
        );
        let mut different = repetition.clone();
        different.summary.selection.configuration =
            DiagnosticText::new("different exact Task definition");
        assert!(
            baseline.combine_repetition(&different).is_err(),
            "Changed execution input cannot hide behind the same target label"
        );
        filesystem
            .atomic_write(
                std::path::PathBuf::from("/series/main.rs"),
                "changed source".to_owned(),
            )
            .await
            .expect("Actual source change");
        cx.run_until_parked();
        let changed = cx
            .update(|cx| comparison_source_identity(&project, &directory, cx))
            .expect("Changed source metadata");
        assert_ne!(
            Some(changed),
            baseline.source_identity,
            "Tracked source changes affect the owner fingerprint"
        );
    }

    fn comparison_fixture(
        cx: &mut gpui::TestAppContext,
        performance: bool,
    ) -> (
        DoctorComparisonSnapshot,
        DoctorComparisonSnapshot,
        DoctorComparisonRequest,
    ) {
        let binding = repair_binding(cx);
        let mut before_selection = investigation_selection();
        if performance {
            before_selection.problem = DoctorInvestigationProblem::SlowOperation;
        }
        let mut after_selection = before_selection.clone();
        after_selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
            "reviewed source after",
            "selected test",
            "debug",
            "warm caches",
        ])
        .expect("Changed input identity");
        after_selection.source_revision = DiagnosticText::new("reviewed source revision 2");
        let mut before = DoctorInvestigation::new(
            before_selection.clone(),
            report().scope,
            binding.clone(),
            10,
        )
        .expect("Baseline scope");
        let mut after =
            DoctorInvestigation::new(after_selection.clone(), report().scope, binding.clone(), 20)
                .expect("Verification scope");
        let before_id = before
            .record(
                &binding,
                &before_selection,
                DoctorEvidenceInput {
                    owner: DoctorEvidenceOwner::Test,
                    owner_reference: DiagnosticText::new("baseline:test"),
                    classification: DoctorEvidenceClass::Observed,
                    observation: DoctorEvidenceObservation::FunctionalFailure {
                        assertion: DiagnosticText::new("selected regression"),
                    },
                    detail: DiagnosticText::new(
                        "API_KEY=COMPARISON_PRIVATE selected reproduced assertion",
                    ),
                    observed_at: 11,
                    historical: false,
                },
            )
            .expect("Observed failure");
        let after_id = after
            .record(
                &binding,
                &after_selection,
                DoctorEvidenceInput {
                    owner: DoctorEvidenceOwner::Test,
                    owner_reference: DiagnosticText::new("verification:test"),
                    classification: DoctorEvidenceClass::Observed,
                    observation: DoctorEvidenceObservation::FunctionalSuccess {
                        assertion: DiagnosticText::new("selected regression"),
                    },
                    detail: DiagnosticText::new("Observed relevant selected assertion"),
                    observed_at: 21,
                    historical: false,
                },
            )
            .expect("Observed verification");
        let mut request = DoctorComparisonRequest {
            baseline_functional: before_id,
            verification_functional: after_id,
            change: DoctorReviewedChange {
                before_input: before_selection.input_identity.clone(),
                after_input: after_selection.input_identity.clone(),
                description: DiagnosticText::new(
                    "Reviewed source proposal, unrelated edits retained",
                ),
                session_id: Some(DiagnosticText::new("existing-native-session")),
            },
            performance: None,
            baseline_measurements: Vec::new(),
            verification_measurements: Vec::new(),
            baseline_protocol: DiagnosticText::new(DOCTOR_TASK_TIMING_PROTOCOL),
            verification_protocol: DiagnosticText::new(DOCTOR_TASK_TIMING_PROTOCOL),
        };
        if performance {
            request.performance = Some(DoctorPerformanceCriterion {
                metric: DiagnosticText::new("reviewed Task end-to-end wall time"),
                unit: DiagnosticText::new("ms"),
                protocol: request.baseline_protocol.clone(),
                minimum_repetitions: 3,
                maximum_after: DiagnosticText::new("90"),
                minimum_improvement_percent: DiagnosticText::new("20"),
            });
            for (investigation, selection, identifiers, values) in [
                (
                    &mut before,
                    &before_selection,
                    &mut request.baseline_measurements,
                    [90, 100, 110],
                ),
                (
                    &mut after,
                    &after_selection,
                    &mut request.verification_measurements,
                    [70, 80, 90],
                ),
            ] {
                for (index, value) in values.into_iter().enumerate() {
                    let input = DoctorEvidenceInput {
                        owner: DoctorEvidenceOwner::Task,
                        owner_reference: DiagnosticText::new(format!("measurement-{index}")),
                        classification: DoctorEvidenceClass::Observed,
                        observation: DoctorEvidenceObservation::AggregateCounter {
                            metric: DiagnosticText::new("reviewed Task end-to-end wall time"),
                            value: DiagnosticText::new(value.to_string()),
                            unit: DiagnosticText::new("ms"),
                        },
                        detail: DiagnosticText::new(DOCTOR_TASK_TIMING_PROTOCOL),
                        observed_at: investigation.summary.started_at + 2,
                        historical: false,
                    };
                    identifiers.push(
                        investigation
                            .record(&binding, selection, input)
                            .expect("Observed repeated measurement"),
                    );
                }
            }
        }
        (
            before.comparison_snapshot(),
            after.comparison_snapshot(),
            request,
        )
    }

    #[gpui::test]
    async fn doctor_comparison_observed_functional_change_redacted_history_and_legacy(
        cx: &mut gpui::TestAppContext,
    ) {
        let (before, after, request) = comparison_fixture(cx, false);
        let comparison = before
            .compare(&after, request, 30)
            .expect("Relevant comparison");
        assert_eq!(
            comparison.outcome,
            DoctorComparisonOutcome::VerifiedSelectedTarget
        );
        assert!(
            comparison
                .detail
                .as_str()
                .contains("not proof of causation")
        );
        let mut report =
            DoctorReport::new(after.summary.scope.clone(), after.summary.generation, 30);
        report
            .set_investigation(after.summary.clone())
            .expect("Actual observed result");
        report
            .set_comparison(comparison)
            .expect("Bounded comparison receipt");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let mut receipt = DoctorHistoryRecord::new(
            report.clone(),
            Some(DiagnosticText::new("Compared selected repair")),
            ReceiptOutcome::Completed,
            30,
        );
        receipt.session_id = Some(DiagnosticText::new("existing-native-session"));
        history
            .record(DoctorProduct::Copper, receipt, 30)
            .await
            .expect("Existing AppDB owner");
        let records = history
            .records(DoctorProduct::Copper, 31)
            .await
            .expect("Persisted history");
        let serialized = serde_json::to_string(&records).expect("Redacted history");
        assert!(!serialized.contains("COMPARISON_PRIVATE"));
        assert!(
            serialized.contains("VerifiedSelectedTarget")
                && serialized.contains("existing-native-session")
        );
        let mut legacy = serde_json::to_value(report).expect("Legacy fixture");
        legacy.as_object_mut().expect("Report").remove("comparison");
        let legacy: DoctorReport =
            serde_json::from_value(legacy).expect("Compatible legacy report");
        assert!(legacy.comparison().is_none());
        legacy.validate().expect("Legacy still bounded");
    }

    #[gpui::test]
    fn doctor_comparison_rejects_unrelated_historical_hypothesis_cancelled_and_retained_failure(
        cx: &mut gpui::TestAppContext,
    ) {
        let (before, after, request) = comparison_fixture(cx, false);
        for case in 0..10 {
            let mut before = before.clone();
            let mut after = after.clone();
            let mut request = request.clone();
            match case {
                0 => {
                    after.summary.selection.validation_target =
                        DiagnosticText::new("unrelated passing test")
                }
                1 => after.binding.directory.worktree_id = settings::WorktreeId::from_usize(2),
                2 => after.binding.connection = Some(repair_binding(cx).project),
                3 => after.summary.evidence.first_mut().expect("Row").historical = true,
                4 => {
                    let row = after.summary.evidence.first_mut().expect("Row");
                    row.owner = DoctorEvidenceOwner::NativeAgent;
                    row.classification = DoctorEvidenceClass::Hypothesis;
                    row.observation = DoctorEvidenceObservation::Unverified;
                }
                5 => {
                    let row = after.summary.evidence.first_mut().expect("Row");
                    row.owner = DoctorEvidenceOwner::Task;
                    row.observation = DoctorEvidenceObservation::CommandExit { code: Some(0) };
                }
                6 => request.change.after_input = request.change.before_input.clone(),
                7 => {
                    before
                        .summary
                        .evidence
                        .first_mut()
                        .expect("Row")
                        .observation = DoctorEvidenceObservation::FunctionalFailure {
                        assertion: DiagnosticText::new("unrelated failure"),
                    }
                }
                8 => request.verification_functional = uuid::Uuid::new_v4(),
                9 => after.summary.selection.problem = DoctorInvestigationProblem::Concurrency,
                _ => unreachable!("Finite fixture cases"),
            }
            for row in &mut after.summary.evidence {
                row.selection = after.summary.selection.clone();
            }
            let result = before
                .compare(&after, request, 30)
                .expect("Incomplete diagnosis remains readable");
            assert_eq!(
                result.outcome,
                DoctorComparisonOutcome::Incomplete,
                "Case {case}"
            );
        }
        let mut cancelled = after.clone();
        cancelled
            .summary
            .evidence
            .first_mut()
            .expect("Row")
            .observation = DoctorEvidenceObservation::Cancelled;
        assert_eq!(
            before
                .compare(&cancelled, request.clone(), 30)
                .expect("Cancelled receipt")
                .outcome,
            DoctorComparisonOutcome::Cancelled
        );
        let mut failing = after.clone();
        failing
            .summary
            .evidence
            .first_mut()
            .expect("Row")
            .observation = DoctorEvidenceObservation::FunctionalFailure {
            assertion: DiagnosticText::new("selected regression"),
        };
        assert_eq!(
            before
                .compare(&failing, request.clone(), 30)
                .expect("Observed failure")
                .outcome,
            DoctorComparisonOutcome::CriterionNotMet
        );
        let mut retained = after;
        let mut failure = retained.summary.evidence.first().expect("Row").clone();
        failure.id = uuid::Uuid::new_v4();
        failure.owner_reference = DiagnosticText::new("retained current failure");
        failure.observation = DoctorEvidenceObservation::FunctionalFailure {
            assertion: DiagnosticText::new("selected regression"),
        };
        retained.summary.evidence.push(failure);
        assert_eq!(
            before
                .compare(&retained, request, 30)
                .expect("Retained failures")
                .outcome,
            DoctorComparisonOutcome::CriterionNotMet
        );
    }

    #[gpui::test]
    fn doctor_comparison_performance_exact_protocol_samples_thresholds_and_incomparable_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        let (before, after, request) = comparison_fixture(cx, true);
        let result = before
            .compare(&after, request.clone(), 30)
            .expect("Comparable observed medians");
        assert_eq!(
            result.outcome,
            DoctorComparisonOutcome::VerifiedSelectedTarget
        );
        assert_eq!(
            result.baseline_median.as_ref().map(DiagnosticText::as_str),
            Some("100")
        );
        assert_eq!(
            result
                .verification_median
                .as_ref()
                .map(DiagnosticText::as_str),
            Some("80")
        );
        for case in 0..12 {
            let mut after = after.clone();
            let mut request = request.clone();
            match case {
                0 => {
                    request.verification_protocol =
                        DiagnosticText::new("different capture protocol")
                }
                1 => {
                    request.verification_measurements.pop();
                }
                2 => after.summary.selection.workload = DiagnosticText::new("different workload"),
                3 => {
                    after.summary.selection.configuration =
                        DiagnosticText::new("different runtime or build features")
                }
                4 => after.summary.selection.cache_conditions = DiagnosticText::new("cold caches"),
                5 => {
                    after
                        .summary
                        .evidence
                        .get_mut(1)
                        .expect("Sample")
                        .historical = true
                }
                6 => {
                    let row = after.summary.evidence.get_mut(1).expect("Sample");
                    row.observation = DoctorEvidenceObservation::Incomplete;
                }
                7 => request.performance = None,
                8 => {
                    request.performance.as_mut().expect("Criterion").metric =
                        DiagnosticText::new("CPU time")
                }
                9 => {
                    request.performance.as_mut().expect("Criterion").unit =
                        DiagnosticText::new("seconds")
                }
                10 => {
                    after.summary.evidence.get_mut(1).expect("Sample").owner =
                        DoctorEvidenceOwner::Profile
                }
                11 => {
                    after.summary.evidence.get_mut(1).expect("Sample").detail = DiagnosticText::new(
                        "User-declared capture protocol without owner provenance",
                    )
                }
                _ => unreachable!("Finite fixture cases"),
            }
            for row in &mut after.summary.evidence {
                row.selection = after.summary.selection.clone();
            }
            assert_eq!(
                before
                    .compare(&after, request, 30)
                    .expect("Incomplete comparison")
                    .outcome,
                DoctorComparisonOutcome::Incomplete,
                "Case {case}"
            );
        }
        let mut threshold = request.clone();
        threshold
            .performance
            .as_mut()
            .expect("Criterion")
            .maximum_after = DiagnosticText::new("79");
        assert_eq!(
            before
                .compare(&after, threshold, 30)
                .expect("Threshold not met")
                .outcome,
            DoctorComparisonOutcome::CriterionNotMet
        );
        for invalid in ["NaN", "inf", "-1"] {
            let mut request = request.clone();
            request
                .performance
                .as_mut()
                .expect("Criterion")
                .maximum_after = DiagnosticText::new(invalid);
            assert!(before.compare(&after, request, 30).is_err());
        }
        let mut duplicates = request.clone();
        duplicates
            .verification_measurements
            .push(duplicates.verification_functional);
        assert!(before.compare(&after, duplicates, 30).is_err());
        assert_eq!(
            before
                .compare(&after, request, 19)
                .expect("Invalid clock evidence")
                .outcome,
            DoctorComparisonOutcome::Incomplete
        );
    }

    fn investigation_input(
        reference: &str,
        observation: DoctorEvidenceObservation,
    ) -> DoctorEvidenceInput {
        DoctorEvidenceInput {
            owner: DoctorEvidenceOwner::Task,
            owner_reference: DiagnosticText::new(reference),
            classification: DoctorEvidenceClass::Observed,
            observation,
            detail: DiagnosticText::new(
                "API_KEY=INVESTIGATION_PRIVATE_SECRET observed selected owner outcome",
            ),
            observed_at: 11,
            historical: false,
        }
    }

    #[gpui::test]
    fn doctor_investigation_rejects_stale_unrelated_and_duplicate_evidence(
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = repair_binding(cx);
        let selection = investigation_selection();
        let mut investigation =
            DoctorInvestigation::new(selection.clone(), report().scope, binding.clone(), 10)
                .expect("Selected scope");
        let mut unrelated = selection.clone();
        unrelated.validation_target = DiagnosticText::new("unrelated passing tests");
        assert!(
            investigation
                .record(
                    &binding,
                    &unrelated,
                    investigation_input(
                        "unrelated",
                        DoctorEvidenceObservation::FunctionalSuccess {
                            assertion: DiagnosticText::new("unrelated test")
                        }
                    )
                )
                .is_err()
        );
        for changed in [
            DoctorRepairBinding {
                generation: 2,
                ..binding.clone()
            },
            DoctorRepairBinding {
                input_revision: 2,
                ..binding.clone()
            },
            DoctorRepairBinding {
                host: project::DoctorExecutionHost::RemoteConnected,
                ..binding.clone()
            },
        ] {
            assert!(
                investigation
                    .record(
                        &changed,
                        &selection,
                        investigation_input(
                            "stale",
                            DoctorEvidenceObservation::CommandExit { code: Some(0) }
                        )
                    )
                    .is_err()
            );
        }
        let mut changed_inputs = selection.clone();
        changed_inputs.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
            "different actual workload behind the same display label",
        ])
        .expect("Changed input identity");
        assert!(
            investigation
                .record(
                    &binding,
                    &changed_inputs,
                    investigation_input(
                        "changed-workload",
                        DoctorEvidenceObservation::CommandExit { code: Some(0) }
                    )
                )
                .is_err()
        );
        investigation
            .record(
                &binding,
                &selection,
                investigation_input(
                    "selected-run",
                    DoctorEvidenceObservation::CommandExit { code: Some(0) },
                ),
            )
            .expect("Observed command outcome");
        assert!(
            !investigation
                .summary()
                .evidence()
                .first()
                .expect("Command evidence")
                .reproduced_failure(),
            "Command success is not a reproduced problem or a verified repair"
        );
        assert!(
            investigation
                .record(
                    &binding,
                    &selection,
                    investigation_input(
                        "selected-run",
                        DoctorEvidenceObservation::CommandExit { code: Some(0) }
                    )
                )
                .is_err()
        );
        assert_eq!(
            investigation.summary().evidence().len(),
            1,
            "Duplicate rejection preserves existing evidence"
        );
        let mut old = investigation_input(
            "old-run",
            DoctorEvidenceObservation::FunctionalFailure {
                assertion: DiagnosticText::new("selected test assertion"),
            },
        );
        old.observed_at = 1;
        assert!(investigation.record(&binding, &selection, old).is_err());
        let mut historical = investigation_input(
            "historical-run",
            DoctorEvidenceObservation::FunctionalFailure {
                assertion: DiagnosticText::new("selected test assertion"),
            },
        );
        historical.historical = true;
        historical.observed_at = 1;
        investigation
            .record(&binding, &selection, historical)
            .expect("Explicit historical facts");
        assert!(
            !investigation
                .summary()
                .evidence()
                .last()
                .expect("Historical evidence")
                .reproduced_failure()
        );
    }

    #[gpui::test]
    async fn doctor_investigation_bounds_redaction_history_export_and_uncertainty(
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = repair_binding(cx);
        let selection = investigation_selection();
        let mut source = report();
        let mut investigation =
            DoctorInvestigation::new(selection.clone(), source.scope.clone(), binding.clone(), 10)
                .expect("Selected scope");
        let mut observed = investigation_input(
            "selected-failure",
            DoctorEvidenceObservation::FunctionalFailure {
                assertion: DiagnosticText::new("selected test expected 3, observed 4"),
            },
        );
        observed.owner = DoctorEvidenceOwner::Test;
        investigation
            .record(&binding, &selection, observed)
            .expect("Relevant observed failure");
        assert!(
            investigation
                .summary()
                .evidence()
                .first()
                .expect("Observed failure")
                .reproduced_failure()
        );
        let mut assertion = investigation_input(
            "model-assertion",
            DoctorEvidenceObservation::FunctionalSuccess {
                assertion: DiagnosticText::new("all tests pass"),
            },
        );
        assertion.owner = DoctorEvidenceOwner::NativeAgent;
        assertion.classification = DoctorEvidenceClass::Hypothesis;
        assert!(
            investigation
                .record(&binding, &selection, assertion)
                .is_err(),
            "Model assertions cannot masquerade as observed verification"
        );
        for observation in [
            DoctorEvidenceObservation::Cancelled,
            DoctorEvidenceObservation::Incomplete,
            DoctorEvidenceObservation::Unsupported,
            DoctorEvidenceObservation::InsufficientEvidence,
        ] {
            investigation
                .record(
                    &binding,
                    &selection,
                    investigation_input(&format!("status-{observation:?}"), observation),
                )
                .expect("Incomplete states remain distinct");
        }
        source
            .set_investigation(investigation.summary().clone())
            .expect("Same bounded report owner");
        let encoded = serde_json::to_string(&source).expect("Redacted serialization");
        assert!(!encoded.contains("INVESTIGATION_PRIVATE_SECRET"));
        let decoded: DoctorReport =
            serde_json::from_str(&encoded).expect("Compatible report schema");
        decoded
            .validate()
            .expect("Evidence revalidated on history load");
        assert_eq!(
            decoded
                .investigation()
                .expect("Provenance")
                .evidence()
                .len(),
            5
        );
        let exported = decoded.export().expect("Explicit bounded export");
        assert!(
            exported.contains("selected regression") && exported.contains("InsufficientEvidence")
        );
        assert!(!exported.contains("INVESTIGATION_PRIVATE_SECRET"));
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        store
            .record(
                DoctorProduct::Copper,
                DoctorHistoryRecord::new(decoded, None, ReceiptOutcome::Incomplete, 12),
                12,
            )
            .await
            .expect("Existing redacted history owner");
        let reopened = DoctorHistoryStore::from_database(&database);
        let retained = reopened
            .records(DoctorProduct::Copper, 13)
            .await
            .expect("Durable bounded provenance");
        assert_eq!(
            retained.first().expect("History").outcome,
            ReceiptOutcome::Incomplete
        );
        assert!(
            retained
                .first()
                .expect("History")
                .report
                .investigation()
                .is_some()
        );
        let mut tampered: DoctorReport = serde_json::from_str(&encoded).expect("Fixture report");
        tampered
            .investigation
            .as_mut()
            .expect("Fixture summary")
            .evidence
            .first_mut()
            .expect("Fixture observation")
            .classification = DoctorEvidenceClass::Hypothesis;
        assert!(
            tampered.validate().is_err(),
            "Invalid saved evidence cannot acquire observed status"
        );
        let mut rejected = false;
        for index in 0..128 {
            let mut input = investigation_input(
                &format!("bounded-{index}"),
                DoctorEvidenceObservation::Incomplete,
            );
            input.detail = DiagnosticText::new("bounded summary ".repeat(200));
            if investigation.record(&binding, &selection, input).is_err() {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "Evidence admission must enforce aggregate limits");
        investigation
            .summary()
            .validate()
            .expect("Failed admission preserves a valid bounded summary");
        assert!(
            serde_json::to_vec(investigation.summary())
                .expect("Serialization")
                .len()
                <= MAX_REPORT_BYTES
        );
    }

    #[gpui::test]
    fn doctor_investigation_scope_and_counter_claims_are_capability_aware(
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = repair_binding(cx);
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Product;
        assert!(selection.validate().is_err());
        selection.problem = DoctorInvestigationProblem::ProductReadiness;
        selection.validate().expect("Product scope");
        selection.scope = DoctorInvestigationScope::Development;
        selection.problem = DoctorInvestigationProblem::Pipeline;
        selection.validate().expect("Development scope");
        selection.scope = DoctorInvestigationScope::Application;
        selection.problem = DoctorInvestigationProblem::Cpu;
        let mut investigation =
            DoctorInvestigation::new(selection.clone(), report().scope, binding.clone(), 10)
                .expect("Application scope");
        for value in ["NaN", "inf", "-1", "not-a-counter"] {
            let mut input = investigation_input(
                "invalid-counter",
                DoctorEvidenceObservation::AggregateCounter {
                    metric: DiagnosticText::new("wall time"),
                    value: DiagnosticText::new(value),
                    unit: DiagnosticText::new("milliseconds"),
                },
            );
            input.owner = DoctorEvidenceOwner::Profile;
            assert!(investigation.record(&binding, &selection, input).is_err());
        }
        let mut counter = investigation_input(
            "profile-capture",
            DoctorEvidenceObservation::AggregateCounter {
                metric: DiagnosticText::new("samples"),
                value: DiagnosticText::new("100"),
                unit: DiagnosticText::new("count"),
            },
        );
        counter.owner = DoctorEvidenceOwner::Profile;
        investigation
            .record(&binding, &selection, counter)
            .expect("Actual aggregate observation");
        assert!(
            !investigation
                .summary()
                .evidence()
                .first()
                .expect("Counter")
                .reproduced_failure(),
            "Samples do not establish CPU time, memory usage or a functional fix"
        );
    }

    #[gpui::test]
    async fn doctor_history_recovers_interrupted_receipts_without_replay() {
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        let record = DoctorHistoryRecord::new(
            report(),
            Some(DiagnosticText::new("API_KEY=history-sentinel repair")),
            ReceiptOutcome::Running,
            10,
        );
        let id = store
            .record(DoctorProduct::Copper, record, 10)
            .await
            .expect("Receipt persists before effects");
        assert!(store.clear(DoctorProduct::Copper, 10).await.is_err());
        let persisted = store
            .database
            .scoped(HISTORY_NAMESPACE)
            .read("Copper")
            .expect("Read receipt")
            .expect("Receipt is durable");
        assert!(!persisted.contains("history-sentinel"));
        assert!(persisted.contains("Running"));
        drop(store);
        let reopened = DoctorHistoryStore::from_database(&database);
        let records = reopened
            .records(DoctorProduct::Copper, 20)
            .await
            .expect("Recover durable receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, id);
        assert_eq!(records[0].outcome, ReceiptOutcome::Interrupted);
        assert!(
            reopened
                .finish(
                    DoctorProduct::Copper,
                    id,
                    ReceiptOutcome::Completed,
                    report(),
                    None,
                    21
                )
                .await
                .is_err()
        );
        assert!(
            reopened
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("Independent product history")
                .is_empty()
        );
        assert!(
            reopened
                .database
                .scoped(HISTORY_NAMESPACE)
                .read("Copper")
                .expect("Recovered receipt persisted")
                .expect("Recovered receipt exists")
                .contains("Interrupted")
        );
    }

    #[gpui::test]
    async fn doctor_history_retains_scope_links_and_preserves_other_namespaces() {
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        store
            .database
            .scoped("native.sessions")
            .write("sentinel".into(), "unchanged".into())
            .await
            .expect("Persist unrelated fixture");
        let id = store
            .record(
                DoctorProduct::Copper,
                DoctorHistoryRecord::new(
                    report(),
                    Some(DiagnosticText::new("Repair")),
                    ReceiptOutcome::Running,
                    10,
                ),
                10,
            )
            .await
            .expect("Persist action receipt");
        let mut wrong_scope = report();
        wrong_scope.scope.host = DiagnosticText::new("Other host");
        assert!(
            store
                .finish(
                    DoctorProduct::Copper,
                    id,
                    ReceiptOutcome::Completed,
                    wrong_scope,
                    None,
                    11
                )
                .await
                .is_err()
        );
        store
            .finish(
                DoctorProduct::Copper,
                id,
                ReceiptOutcome::Completed,
                report(),
                Some(DiagnosticText::new("native-session-123")),
                11,
            )
            .await
            .expect("Record actual outcome");
        assert!(
            store
                .finish(
                    DoctorProduct::Copper,
                    id,
                    ReceiptOutcome::Completed,
                    report(),
                    None,
                    12
                )
                .await
                .is_err()
        );
        let records = store
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Inspect linked outcome");
        assert_eq!(
            records[0].session_id.as_ref().map(DiagnosticText::as_str),
            Some("native-session-123")
        );
        store
            .delete(DoctorProduct::Copper, id, 12)
            .await
            .expect("Delete only Doctor history");
        assert_eq!(
            store
                .database
                .scoped("native.sessions")
                .read("sentinel")
                .expect("Unrelated session remains")
                .as_deref(),
            Some("unchanged")
        );
    }

    #[gpui::test]
    async fn doctor_history_prunes_age_and_count_but_refuses_active_overflow() {
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        for index in 0..=MAX_HISTORY_RECORDS {
            store
                .record(
                    DoctorProduct::Copper,
                    DoctorHistoryRecord::new(
                        report(),
                        None,
                        ReceiptOutcome::Completed,
                        index as u64,
                    ),
                    index as u64,
                )
                .await
                .expect("Admit bounded terminal history");
        }
        let records = store
            .records(DoctorProduct::Copper, 100)
            .await
            .expect("Inspect pruned count");
        assert_eq!(records.len(), MAX_HISTORY_RECORDS);
        assert!(!records.iter().any(|record| record.created_at == 0));
        assert!(
            store
                .records(DoctorProduct::Copper, HISTORY_RETENTION_SECONDS + 101)
                .await
                .expect("Prune expired records")
                .is_empty()
        );
        for index in 0..MAX_HISTORY_RECORDS {
            store
                .record(
                    DoctorProduct::Copper,
                    DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, index as u64),
                    index as u64,
                )
                .await
                .expect("Active receipt is retained");
        }
        assert!(
            store
                .record(
                    DoctorProduct::Copper,
                    DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, 100),
                    100
                )
                .await
                .is_err()
        );
        assert_eq!(
            store
                .records(DoctorProduct::Copper, HISTORY_RETENTION_SECONDS + 101)
                .await
                .expect("Active receipts never pruned")
                .len(),
            MAX_HISTORY_RECORDS
        );
    }

    #[gpui::test]
    async fn doctor_history_rejects_corruption_and_storage_failure() {
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        store
            .database
            .scoped(HISTORY_NAMESPACE)
            .write("Copper".into(), "not-json".into())
            .await
            .expect("Corrupt fixture");
        assert!(store.records(DoctorProduct::Copper, 10).await.is_err());
        assert_eq!(
            store
                .database
                .scoped(HISTORY_NAMESPACE)
                .read("Copper")
                .expect("Corruption remains")
                .as_deref(),
            Some("not-json")
        );
        database
            .0
            .write(|connection| connection.exec("DROP TABLE scoped_kv_store")?())
            .await
            .expect("Simulate storage failure");
        assert!(
            store
                .record(
                    DoctorProduct::Groovy,
                    DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, 10),
                    10
                )
                .await
                .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_history_dropped_queued_write_cannot_be_overwritten_by_cached_state() {
        use futures::FutureExt;
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        };
        let paused = Arc::new(AtomicBool::new(false));
        let writes: Arc<Mutex<Vec<Box<dyn FnOnce() + Send>>>> = Arc::new(Mutex::new(Vec::new()));
        let connection = db::sqlez::thread_safe_connection::ThreadSafeConnection::builder::<()>(
            &format!("doctor-delayed-history-{}", uuid::Uuid::new_v4()),
            false,
        )
        .with_write_queue_constructor(Box::new({
            let paused = paused.clone();
            let writes = writes.clone();
            move || {
                Box::new({
                    let paused = paused.clone();
                    let writes = writes.clone();
                    move |write| {
                        if paused.load(Ordering::SeqCst) {
                            writes.lock().expect("Fixture queue").push(write);
                        } else {
                            write();
                        }
                    }
                })
            }
        }))
        .build()
        .await
        .expect("Delayed-write database fixture");
        connection.write(|connection| connection.exec("CREATE TABLE scoped_kv_store(namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(namespace, key)) STRICT;")?()).await.expect("Existing KVP schema fixture");
        let database = db::AppDatabase(connection);
        let history = DoctorHistoryStore::from_database(&database);
        history
            .records(DoctorProduct::Copper, 10)
            .await
            .expect("Initialized view cache");
        paused.store(true, Ordering::SeqCst);
        let interrupted_record = DoctorHistoryRecord::new(
            report(),
            Some(DiagnosticText::new("Pending no-replay receipt")),
            ReceiptOutcome::Running,
            10,
        );
        let interrupted_id = interrupted_record.id;
        let mut pending = history
            .record(DoctorProduct::Copper, interrupted_record, 10)
            .boxed();
        assert!(futures::poll!(&mut pending).is_pending());
        for write in std::mem::take(&mut *writes.lock().expect("Fixture queue")) {
            write();
        }
        if history
            .database
            .scoped(HISTORY_NAMESPACE)
            .read("Copper")
            .expect("Inspect committed state")
            .is_none()
        {
            assert!(futures::poll!(&mut pending).is_pending());
        }
        drop(pending);
        for write in std::mem::take(&mut *writes.lock().expect("Fixture queue")) {
            write();
        }
        paused.store(false, Ordering::SeqCst);
        history
            .record(
                DoctorProduct::Copper,
                DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Completed, 11),
                11,
            )
            .await
            .expect("Later independent operation");
        let reopened = DoctorHistoryStore::from_database(&database);
        let records = reopened
            .records(DoctorProduct::Copper, 12)
            .await
            .expect("Authoritative reopened history");
        assert_eq!(
            records.len(),
            2,
            "A committed queued receipt must survive the next cached update"
        );
        assert_eq!(
            records
                .iter()
                .find(|record| record.id == interrupted_id)
                .expect("Preserved receipt")
                .outcome,
            ReceiptOutcome::Interrupted
        );
    }

    #[gpui::test]
    async fn doctor_history_write_failure_blocks_receipt_without_cache_success() {
        let database = db::AppDatabase::test_new();
        let store = DoctorHistoryStore::from_database(&database);
        assert!(
            store
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Load available history")
                .is_empty()
        );
        database.0.write(|connection| connection.exec("CREATE TRIGGER reject_doctor_receipt BEFORE INSERT ON scoped_kv_store BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END;")?()).await.expect("Install failing-write fixture");
        assert!(
            store
                .record(
                    DoctorProduct::Copper,
                    DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, 10),
                    10
                )
                .await
                .is_err()
        );
        assert!(
            store
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Failed write never enters cache")
                .is_empty()
        );
        assert!(
            store
                .database
                .scoped(HISTORY_NAMESPACE)
                .read("Copper")
                .expect("Database stays readable")
                .is_none()
        );
    }

    #[gpui::test]
    async fn doctor_history_serializes_concurrent_receipts_and_reuses_app_owner(
        cx: &mut gpui::TestAppContext,
    ) {
        let (first, second) = cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            (
                DoctorHistoryStore::for_app(cx).expect("Create application history owner"),
                DoctorHistoryStore::for_app(cx).expect("Reuse application history owner"),
            )
        });
        assert!(std::sync::Arc::ptr_eq(&first.snapshots, &second.snapshots));
        let one = first.record(
            DoctorProduct::Copper,
            DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, 10),
            10,
        );
        let two = second.record(
            DoctorProduct::Copper,
            DoctorHistoryRecord::new(report(), None, ReceiptOutcome::Running, 10),
            10,
        );
        let (one, two) = futures::join!(one, two);
        assert_ne!(
            one.expect("First durable receipt"),
            two.expect("Second durable receipt")
        );
        assert_eq!(
            first
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("No lost concurrent update")
                .len(),
            2
        );
    }

    #[test]
    fn doctor_history_enforces_byte_cap_without_pruning_active_receipts() -> Result<()> {
        let mut large = report();
        for index in 0..20 {
            let mut check = DoctorCheck::new(
                &format!("large-{index}"),
                &"t".repeat(1024),
                CheckOutcome::Warning,
                &"d".repeat(1024),
                10,
            );
            check.remedy = Some(DiagnosticText::new("r".repeat(1024)));
            large.push(check)?;
        }
        let mut history = DoctorHistorySnapshot::default();
        for _ in 0..100 {
            history.records.push(DoctorHistoryRecord::new(
                large.clone(),
                None,
                ReceiptOutcome::Completed,
                10,
            ));
        }
        history.prune(10)?;
        assert!(history.records.len() < 100);
        assert!(serde_json::to_vec(&history)?.len() <= MAX_HISTORY_BYTES);
        history.records = (0..100)
            .map(|_| DoctorHistoryRecord::new(large.clone(), None, ReceiptOutcome::Running, 10))
            .collect();
        assert!(history.prune(10).is_err());
        Ok(())
    }

    #[test]
    fn doctor_coverage_report_reserves_rows_and_preserves_redaction() -> Result<()> {
        let source = report();
        let checks = (0..100)
            .map(|index| {
                DoctorCheck::new(
                    &format!("check.{index}"),
                    "Observation",
                    CheckOutcome::Ready,
                    "token=private-sentinel /Users/private/source.rs",
                    10,
                )
            })
            .collect();
        let report = DoctorReport::with_checks(source.scope, 1, 10, checks)?;
        assert_eq!(report.checks().len(), MAX_CHECKS);
        let coverage = report.checks().last().expect("Reserved coverage row");
        assert_eq!(coverage.outcome, CheckOutcome::Untested);
        assert!(coverage.detail.as_str().contains("37 of 100"));
        assert!(report.summary().contains("63 passed"));
        assert!(report.summary().contains("incomplete coverage"));
        let serialized = serde_json::to_string(&report)?;
        assert!(!serialized.contains("private-sentinel"));
        assert!(!serialized.contains("/Users/private"));
        let reopened: DoctorReport = serde_json::from_str(&serialized)?;
        assert_eq!(report, reopened);
        assert!(reopened.export()?.contains("37 of 100"));
        Ok(())
    }

    #[test]
    fn doctor_coverage_report_reserves_byte_space_and_rejects_hidden_duplicates() -> Result<()> {
        let source = report();
        let checks = (0..MAX_CHECKS)
            .map(|index| {
                let mut check = DoctorCheck::new(
                    &format!("check.{index}"),
                    &"title".repeat(300),
                    CheckOutcome::Ready,
                    &"evidence".repeat(300),
                    10,
                );
                check.remedy = Some(DiagnosticText::new("remedy".repeat(300)));
                check
            })
            .collect();
        let bounded = DoctorReport::with_checks(source.scope.clone(), 1, 10, checks)?;
        assert!(bounded.checks().len() < MAX_CHECKS);
        assert_eq!(
            bounded.checks().last().expect("Byte coverage").id.as_str(),
            "coverage.report"
        );
        assert!(serde_json::to_vec(&bounded)?.len() <= MAX_REPORT_BYTES);
        bounded.validate()?;
        let mut checks: Vec<_> = (0..100)
            .map(|index| {
                DoctorCheck::new(
                    &format!("check.{index}"),
                    "Observation",
                    CheckOutcome::Ready,
                    "",
                    10,
                )
            })
            .collect();
        checks.push(checks.last().expect("Omitted duplicate").clone());
        assert!(DoctorReport::with_checks(source.scope.clone(), 1, 10, checks).is_err());
        assert!(
            DoctorReport::with_checks(
                source.scope,
                1,
                10,
                vec![DoctorCheck::new(
                    "coverage.report",
                    "Forged coverage",
                    CheckOutcome::Ready,
                    "",
                    10
                )]
            )
            .is_err()
        );
        Ok(())
    }

    #[cfg(feature = "rust-tools")]
    #[test]
    fn doctor_coverage_cargo_counts_only_selected_owner_evidence() {
        use project::cargo_workspace::*;
        let selected = settings::WorktreeId::from_usize(1);
        let mut configuration = CargoWorkspaceConfiguration::unresolved();
        for _ in 0..6 {
            configuration.add_diagnostic(
                None,
                CargoConfigurationDiagnosticCategory::Toolchain,
                "Recorded failure",
            );
        }
        let workspace = CargoWorkspaceModel {
            key: CargoWorkspaceKey {
                root: copper_path(1, ""),
            },
            root_manifest: Some(copper_path(1, "Cargo.toml")),
            display_name: "fixture".into(),
            is_virtual: true,
            members: Vec::new(),
            dependency_graph: None,
            configuration,
        };
        let mut snapshot = CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 1,
            omitted: Default::default(),
            completeness: CargoSnapshotCompleteness::Partial,
            failures: (0..5)
                .map(|index| CargoCandidateFailure {
                    manifest_path: copper_path(1, &format!("member-{index}/Cargo.toml")),
                    category: CargoWorkspaceErrorCategory::CargoFailed,
                    message: "Recorded failure".into(),
                    has_stale_model: false,
                })
                .collect(),
            workspaces: vec![workspace; 9],
        };
        let mut unrelated = snapshot.workspaces.first().expect("Workspace").clone();
        unrelated.key.root = copper_path(2, "");
        snapshot.workspaces.push(unrelated);
        snapshot.failures.push(CargoCandidateFailure {
            manifest_path: copper_path(2, "Cargo.toml"),
            category: CargoWorkspaceErrorCategory::CargoFailed,
            message: "Unrelated failure".into(),
            has_stale_model: false,
        });
        let checks = copper_snapshot_checks(Some((&snapshot, false)), selected, 10);
        for index in 0..4 {
            let check = checks
                .iter()
                .find(|check| check.id.as_str() == format!("rust.failure.{index}"))
                .expect("Selected failure");
            let manifest = format!("member-{index}/Cargo.toml");
            assert!(check.title.as_str().contains(&manifest));
            assert!(check.detail.as_str().contains(&manifest));
            assert!(check.detail.as_str().contains("Recorded failure"));
        }
        for (id, count) in [
            ("coverage.rust.workspaces", "1 of 9"),
            ("coverage.rust.failures", "1 of 5"),
            ("coverage.rust.configuration.0", "2 of 6"),
        ] {
            let coverage = checks
                .iter()
                .find(|check| check.id.as_str() == id)
                .expect("Scoped coverage");
            assert!(coverage.detail.as_str().contains(count));
            assert_eq!(coverage.outcome, CheckOutcome::Untested);
            assert!(coverage.historical);
        }
        assert!(
            !checks
                .iter()
                .any(|check| check.detail.as_str().contains("Unrelated failure"))
        );
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_coverage_jvm_import_does_not_claim_unchecked_modules_ready() {
        use project::jvm_project::*;
        let path = project::ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(1),
            path: util::rel_path::RelPath::empty_arc(),
        };
        let mut snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id: path.worktree_id,
            generation: 1,
            partial: false,
            projects: vec![JdtlsProject {
                membership: JdtlsProjectMembership::ImportedJava,
                imported_build_kind: JdtlsImportedBuildKind::Unknown,
                module_root: path,
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 0,
                library_files: Vec::new(),
                library_artifacts: Vec::new(),
                project_vm_known: true,
                project_vm_java_major: Some(17),
                server_launch_java_major: Some(21),
                java_source_level: Some(17),
                java_target_level: Some(17),
                gradle: None,
                maven: None,
            }],
        };
        let module = snapshot.projects.first().expect("Module").clone();
        snapshot.projects = vec![module; 10];
        snapshot
            .projects
            .last_mut()
            .expect("Unchecked module")
            .server_launch_java_major = Some(8);
        let checks = groovy_project_checks(Some((&snapshot, true)), None, 10);
        let coverage = checks
            .iter()
            .find(|check| check.id.as_str() == "coverage.jvm.projects")
            .expect("Module coverage");
        assert!(coverage.detail.as_str().contains("2 of 10"));
        assert_eq!(coverage.outcome, CheckOutcome::Untested);
        assert_eq!(
            checks
                .iter()
                .filter(|check| check.id.as_str().starts_with("jvm.jdk."))
                .count(),
            8
        );
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    async fn profile_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (
        gpui::Entity<project::Project>,
        project::ProjectPath,
        [project::ProjectPath; 2],
    ) {
        let (project, directory, _servers, _buffer) = project_repair_fixture(cx).await;
        let fs = project.read_with(cx, |owner, _| owner.fs().clone());
        fs.atomic_write(
            "/repair/profile.json".into(),
            include_str!(
                "../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
            )
            .into(),
        )
        .await
        .expect("Profile fixture");
        fs.atomic_write("/repair/profile.syms.json".into(), include_str!("../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json").into()).await.expect("Symbols fixture");
        cx.run_until_parked();
        let paths = ["profile.json", "profile.syms.json"].map(|path| project::ProjectPath {
            worktree_id: directory.worktree_id,
            path: util::rel_path::RelPath::from_unix_str(path)
                .expect("Path")
                .into(),
        });
        (project, directory, paths)
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    fn profile_selection() -> DoctorInvestigationSelection {
        let mut selection = investigation_selection();
        selection.scope = DoctorInvestigationScope::Application;
        selection.problem = DoctorInvestigationProblem::Cpu;
        selection.validation_target = DiagnosticText::new("Selected workload samples");
        selection.input_identity =
            DoctorInvestigationInputIdentity::from_fields(&["profile-fixture"]).expect("Identity");
        selection
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    fn profile_report(review: &DoctorRepairReview<DoctorProfileAction>) -> DoctorReport {
        DoctorReport::with_checks(
            DoctorScope {
                product: DiagnosticText::new("Copper"),
                project: DiagnosticText::new("Selected project"),
                worktree: DiagnosticText::new("Selected worktree"),
                host: DiagnosticText::new("Local"),
            },
            review.binding.generation,
            10,
            vec![review.before.clone()],
        )
        .expect("Report")
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    #[gpui::test]
    async fn doctor_profile_import_is_reviewed_bounded_redacted_and_never_verified(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, paths) = profile_fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        let mut review = cx
            .update(|cx| {
                prepare_profile(
                    &project,
                    directory.clone(),
                    paths.clone(),
                    profile_selection(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Exact proposal");
        assert!(
            review
                .action
                .exact_preview()
                .iter()
                .any(|line| line.contains("10 MiB"))
        );
        review
            .approve(&review.binding.clone())
            .expect("Explicit approval");
        assert!(review.approve(&review.binding.clone()).is_err());
        let report = profile_report(&review);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let pending = cx.update(|cx| {
            run_profile(
                review,
                project.clone(),
                report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        let result = pending.await.expect("Existing profile owner completed");
        assert!(matches!(
            result.evidence.observation,
            DoctorEvidenceObservation::AggregateCounter { .. }
        ));
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        assert!(
            result
                .profile
                .expect("Existing normalized result")
                .total_samples()
                > 0
        );
        let records = DoctorHistoryStore::from_database(&database)
            .records(DoctorProduct::Copper, 11)
            .await
            .expect("Durable receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, ReceiptOutcome::Incomplete);
        let json = serde_json::to_string(&records).expect("History");
        assert!(
            !json.contains("frameTable")
                && !json.contains("/repair")
                && !json.contains("string_table")
        );
        assert_eq!(
            project
                .read_with(cx, |owner, _| owner.fs().clone())
                .load(std::path::Path::new("/repair/Cargo.lock"))
                .await
                .expect("Unrelated lock"),
            "original"
        );
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    #[gpui::test]
    async fn doctor_profile_denies_unapproved_stale_sensitive_and_oversized_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        let (project, directory, paths) = profile_fixture(cx).await;
        let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
        for problem in [
            DoctorInvestigationProblem::Memory,
            DoctorInvestigationProblem::Crash,
        ] {
            let mut selection = profile_selection();
            selection.problem = problem;
            assert!(
                cx.update(|cx| prepare_profile(
                    &project,
                    directory.clone(),
                    paths.clone(),
                    selection,
                    60,
                    1,
                    10,
                    cx
                ))
                .is_err()
            );
        }
        let mut dump_paths = paths.clone();
        dump_paths[0].path = util::rel_path::RelPath::from_unix_str("heap.hprof")
            .expect("Path")
            .into();
        assert!(
            cx.update(|cx| prepare_profile(
                &project,
                directory.clone(),
                dump_paths,
                profile_selection(),
                60,
                1,
                10,
                cx
            ))
            .is_err()
        );
        assert!(
            cx.update(|cx| prepare_profile(
                &project,
                directory.clone(),
                paths.clone(),
                profile_selection(),
                301,
                1,
                10,
                cx
            ))
            .is_err()
        );
        let review = cx
            .update(|cx| {
                prepare_profile(
                    &project,
                    directory.clone(),
                    paths.clone(),
                    profile_selection(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Proposal");
        let report = profile_report(&review);
        let (_cancel, cancelled) = async_channel::bounded(1);
        assert!(
            cx.update(|cx| run_profile(
                review,
                project.clone(),
                report,
                DoctorProduct::Copper,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err()
        );
        assert!(
            history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("No admission")
                .is_empty()
        );
        let mut review = cx
            .update(|cx| {
                prepare_profile(
                    &project,
                    directory.clone(),
                    paths.clone(),
                    profile_selection(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Fresh review");
        review.approve(&review.binding.clone()).expect("Approval");
        let report = profile_report(&review);
        project
            .read_with(cx, |owner, _| owner.fs().clone())
            .atomic_write("/repair/profile.json".into(), "{}".into())
            .await
            .expect("Changed input");
        cx.run_until_parked();
        let (_cancel, cancelled) = async_channel::bounded(1);
        let result = cx
            .update(|cx| {
                run_profile(
                    review,
                    project.clone(),
                    report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Attributable stale result");
        assert!(result.profile.is_none());
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        let mut malformed_review = cx
            .update(|cx| {
                prepare_profile(
                    &project,
                    directory.clone(),
                    paths.clone(),
                    profile_selection(),
                    60,
                    1,
                    10,
                    cx,
                )
            })
            .expect("Malformed artifacts are not parsed before consent");
        malformed_review
            .approve(&malformed_review.binding.clone())
            .expect("Explicit inspection");
        let malformed_report = profile_report(&malformed_review);
        let (_cancel, cancelled) = async_channel::bounded(1);
        let malformed = cx
            .update(|cx| {
                run_profile(
                    malformed_review,
                    project.clone(),
                    malformed_report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Observed failed import");
        assert!(malformed.profile.is_none());
        assert_eq!(
            malformed.evidence.observation,
            DoctorEvidenceObservation::Incomplete
        );
        project
            .read_with(cx, |owner, _| owner.fs().clone())
            .atomic_write(
                "/repair/profile.json".into(),
                " ".repeat(10 * 1024 * 1024 + 1),
            )
            .await
            .expect("Oversized fixture");
        cx.run_until_parked();
        assert!(
            cx.update(|cx| prepare_profile(
                &project,
                directory,
                paths,
                profile_selection(),
                60,
                1,
                10,
                cx
            ))
            .is_err()
        );
    }

    #[cfg(all(feature = "rust-tools", target_os = "macos"))]
    #[gpui::test]
    async fn doctor_profile_cancellation_revocation_deadline_and_view_loss_clean_owned_import(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::UpdateGlobal as _;
        let (project, directory, paths) = profile_fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::from_database(&database);
        for mode in 0..4 {
            let mut review = cx
                .update(|cx| {
                    prepare_profile(
                        &project,
                        directory.clone(),
                        paths.clone(),
                        profile_selection(),
                        60,
                        1,
                        10,
                        cx,
                    )
                })
                .expect("Fresh review");
            review.approve(&review.binding.clone()).expect("Approval");
            let report = profile_report(&review);
            let store = project.read_with(cx, |owner, _| {
                owner
                    .profile_results_store_for_test()
                    .expect("Existing owner")
                    .clone()
            });
            let (started, waiting) = futures::channel::oneshot::channel();
            let (resume, release) = futures::channel::oneshot::channel();
            store.update(cx, |store, _| {
                store.gate_next_import_for_test(started, release)
            });
            let (cancel, cancelled) = async_channel::bounded(1);
            let pending = cx.update(|cx| {
                run_profile(
                    review,
                    project.clone(),
                    report,
                    DoctorProduct::Copper,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            });
            waiting
                .await
                .expect("Existing importer admitted after receipt");
            assert!(store.read_with(cx, |store, _| store.has_active_work()));
            match mode {
                0 => {
                    cancel.close();
                }
                1 => {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_secs(61));
                }
                2 => {
                    cx.update(|cx| {
                        settings::SettingsStore::update_global(cx, |store, cx| {
                            store.update_user_settings(cx, |settings| {
                                settings
                                    .project
                                    .dap
                                    .entry("changed".into())
                                    .or_default()
                                    .args = Some(vec!["changed".into()]);
                            })
                        })
                    });
                }
                _ => {}
            }
            if mode == 3 {
                drop(pending);
                cx.run_until_parked();
            } else {
                let result = pending.await.expect("Observed stop");
                assert!(result.profile.is_none());
                assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
            }
            resume.send(()).expect("Release deterministic gate");
            cx.run_until_parked();
            assert!(!store.read_with(cx, |store, _| store.has_active_work()));
        }
        let records = DoctorHistoryStore::from_database(&database)
            .records(DoctorProduct::Copper, 100)
            .await
            .expect("Reopened history");
        assert_eq!(records.len(), 4);
        assert!(records.iter().all(|record| matches!(
            record.outcome,
            ReceiptOutcome::Incomplete | ReceiptOutcome::Cancelled
        )));
    }

    #[test]
    fn doctor_report_redacts_before_serialization_and_export() -> Result<()> {
        let mut report = report();
        report.push(DoctorCheck::new("provider", "Provider", CheckOutcome::Blocked,
            "API_KEY=super-secret Bearer other-secret https://user:password@example.com/?token=private /Users/person/private.rs", 10))?;
        let serialized = serde_json::to_string(&report)?;
        let exported = report.export()?;
        for value in [
            "super-secret",
            "other-secret",
            "password",
            "token=private",
            "/Users/person",
        ] {
            assert!(!serialized.contains(value));
            assert!(!exported.contains(value));
        }
        let restored: DoctorReport = serde_json::from_str(&serialized)?;
        assert_eq!(restored, report);
        Ok(())
    }

    #[test]
    fn doctor_report_redacts_quoted_paths_and_json_credentials() -> Result<()> {
        let mut report = report();
        report.push(DoctorCheck::new(
            "details", "Details", CheckOutcome::Blocked,
            r#"at "/Users/person/private.rs" {"password": "json-sentinel"} C:\users\person\private.rs"#, 10,
        ))?;
        for text in [serde_json::to_string(&report)?, report.export()?] {
            for secret in ["/Users/person", "json-sentinel", "person\\private.rs"] {
                assert!(!text.contains(secret), "Private value leaked: {secret}");
            }
        }
        Ok(())
    }

    #[test]
    fn doctor_report_prioritizes_problems_and_never_counts_stale_passes_as_current() -> Result<()> {
        let mut report = report();
        for (identifier, outcome, historical) in [
            ("made.provider", CheckOutcome::Ready, true),
            ("project.authority", CheckOutcome::Ready, false),
            ("rust.failure.0", CheckOutcome::Failed, false),
            (
                "development.cargo.dependencies",
                CheckOutcome::Blocked,
                false,
            ),
            ("rust.selection", CheckOutcome::Untested, false),
        ] {
            let mut check = DoctorCheck::new(
                identifier,
                "Selected observation",
                outcome,
                "Bounded evidence",
                10,
            );
            check.historical = historical;
            report.push(check)?;
        }
        assert!(report.summary().contains("1 passed"));
        assert!(report.summary().contains("1 stale"));
        assert!(report.summary().contains("1 failed"));
        assert!(report.summary().contains("1 blocked"));
        assert!(report.summary().contains("1 unverified"));
        assert!(
            report
                .export()?
                .contains("Stale — Selected observation (retained observation)")
        );
        let mut ordered = report.checks().to_vec();
        ordered.sort_by_key(DoctorCheck::priority);
        assert_eq!(
            ordered.first().map(|check| check.id.as_str()),
            Some("rust.failure.0")
        );
        assert_eq!(
            ordered.last().map(|check| check.id.as_str()),
            Some("project.authority")
        );
        let reopened: DoctorReport = serde_json::from_str(&serde_json::to_string(&report)?)?;
        assert_eq!(report, reopened);
        assert_eq!(
            reopened.checks().first().map(DoctorCheck::state_label),
            Some("Stale")
        );
        Ok(())
    }

    #[test]
    fn doctor_report_distinguishes_configuration_development_and_unperformed_application_checks() {
        for (identifier, scope) in [
            ("made.provider", DoctorInvestigationScope::Product),
            ("project.authority", DoctorInvestigationScope::Product),
            ("rust.cargo", DoctorInvestigationScope::Development),
            ("jvm.import", DoctorInvestigationScope::Development),
            ("development.timing", DoctorInvestigationScope::Development),
            (
                "coverage.rust.failures",
                DoctorInvestigationScope::Development,
            ),
        ] {
            let check = DoctorCheck::new(
                identifier,
                "Owner evidence",
                CheckOutcome::Untested,
                "No runtime test",
                10,
            );
            assert_eq!(check.scope(), scope);
            assert_ne!(check.scope(), DoctorInvestigationScope::Application);
            assert_eq!(check.state_label(), "Unverified");
            assert!(!check.evidence_source().is_empty());
            assert!(!check.next_step().is_empty());
        }
    }

    #[test]
    fn doctor_report_does_not_count_unknown_or_cancelled_as_ready() -> Result<()> {
        let mut report = report();
        for (index, outcome) in [
            CheckOutcome::Ready,
            CheckOutcome::Untested,
            CheckOutcome::Unsupported,
            CheckOutcome::Cancelled,
        ]
        .into_iter()
        .enumerate()
        {
            report.push(DoctorCheck::new(
                &index.to_string(),
                "Check",
                outcome,
                "",
                10,
            ))?;
        }
        assert_eq!(
            report.summary(),
            "1 cancelled · 1 passed · 1 unsupported · 1 unverified · 4 total checks"
        );
        Ok(())
    }

    #[test]
    fn doctor_report_preserves_scope_and_rejects_duplicate_and_oversized_data() -> Result<()> {
        let mut report = report();
        let check = DoctorCheck::new(
            "scope",
            "Scope",
            CheckOutcome::Warning,
            &"界".repeat(5000),
            10,
        );
        assert!(check.detail.as_str().len() <= 1024);
        report.push(check.clone())?;
        assert!(report.push(check).is_err());
        for index in 1..MAX_CHECKS {
            report.push(DoctorCheck::new(
                &index.to_string(),
                "Check",
                CheckOutcome::Untested,
                "",
                10,
            ))?;
        }
        assert!(
            report
                .push(DoctorCheck::new(
                    "extra",
                    "Extra",
                    CheckOutcome::Ready,
                    "",
                    10
                ))
                .is_err()
        );
        let value: DiagnosticText = serde_json::from_str("\"token: leaked\"")?;
        assert!(!value.as_str().contains("leaked"));
        assert_eq!(report.scope.host.as_str(), "Local");
        Ok(())
    }
    #[cfg(all(unix, any(feature = "rust-tools", feature = "jvm-tools")))]
    mod debugger_owner {
        use super::*;
        use gpui::UpdateGlobal as _;
        use std::{
            os::unix::fs::PermissionsExt as _,
            path::{Path, PathBuf},
            sync::Arc,
        };

        struct Adapter;
        #[async_trait::async_trait(?Send)]
        impl dap::adapters::DebugAdapter for Adapter {
            fn name(&self) -> dap::adapters::DebugAdapterName {
                "DoctorOwnerFixture".into()
            }
            fn dap_schema(&self) -> serde_json::Value {
                serde_json::Value::Null
            }
            async fn config_from_zed_format(
                &self,
                _: task::ZedDebugConfig,
            ) -> Result<task::DebugScenario> {
                anyhow::bail!("Opaque formatter denied")
            }
            async fn get_binary(
                &self,
                _: &Arc<dyn dap::adapters::DapDelegate>,
                _: &dap::adapters::DebugTaskDefinition,
                _: Option<PathBuf>,
                _: Option<Vec<String>>,
                _: Option<collections::HashMap<String, String>>,
                _: &mut gpui::AsyncApp,
            ) -> Result<dap::adapters::DebugAdapterBinary> {
                anyhow::bail!("Opaque provisioning denied")
            }
            fn diagnostic_installed_binary(
                &self,
                definition: &dap::adapters::DebugTaskDefinition,
                root: &Path,
                installed: Option<&Path>,
                _: Option<&[String]>,
                environment: Option<&collections::HashMap<String, String>>,
            ) -> Result<dap::adapters::DebugAdapterBinary> {
                Ok(dap::adapters::DebugAdapterBinary {
                    command: Some(
                        installed
                            .expect("Installed fixture")
                            .to_string_lossy()
                            .into(),
                    ),
                    arguments: Vec::new(),
                    envs: environment.cloned().unwrap_or_default(),
                    cwd: Some(root.into()),
                    connection: None,
                    request_args: dap::StartDebuggingRequestArguments {
                        request: dap::StartDebuggingRequestArgumentsRequest::Launch,
                        configuration: definition.config.clone(),
                    },
                })
            }
        }
        struct Provider(gpui::Entity<project::debugger::dap_store::DapStore>);
        impl workspace::DebuggerProvider for Provider {
            fn start_diagnostic_session(
                &self,
                launch: project::debugger::dap_store::DiagnosticDebugLaunch,
                context: task::SharedTaskContext,
                authority: project::debugger::dap_store::DiagnosticDebugAuthority,
                _: &mut gpui::Window,
                cx: &mut gpui::App,
            ) -> gpui::Task<Result<project::debugger::dap_store::DiagnosticDebugSession>>
            {
                self.0.update(cx, |store, cx| {
                    store.start_diagnostic_session(launch, context, authority, cx)
                })
            }
            fn start_session(
                &self,
                _: task::DebugScenario,
                _: task::SharedTaskContext,
                _: Option<gpui::Entity<language::Buffer>>,
                _: Option<settings::WorktreeId>,
                _: &mut gpui::Window,
                _: &mut gpui::App,
            ) {
            }
            fn spawn_task_or_modal(
                &self,
                _: &mut workspace::Workspace,
                _: &zed_actions::Spawn,
                _: &mut gpui::Window,
                _: &mut gpui::Context<workspace::Workspace>,
            ) {
            }
            fn task_scheduled(&self, _: &mut gpui::App) {}
            fn debug_scenario_scheduled(&self, _: &mut gpui::App) {}
            fn debug_scenario_scheduled_last(&self, _: &gpui::App) -> bool {
                false
            }
            fn active_thread_state(
                &self,
                _: &gpui::App,
            ) -> Option<project::debugger::session::ThreadStatus> {
                None
            }
        }
        struct Fixture {
            directory: util::test::TempTree,
            project: gpui::Entity<project::Project>,
            path: project::ProjectPath,
            window: gpui::WindowHandle<workspace::MultiWorkspace>,
            workspace: gpui::Entity<workspace::Workspace>,
            store: gpui::Entity<project::debugger::dap_store::DapStore>,
        }
        async fn fixture(cx: &mut gpui::TestAppContext) -> Fixture {
            cx.executor().allow_parking();
            let directory = util::test::TempTree::new(
                serde_json::json!({"project":{"program":"native", "untouched":"original"},"adapter":"native"}),
            );
            let root = directory.path().join("project");
            let adapter = directory.path().join("adapter");
            for path in [&adapter, &root.join("program")] {
                std::fs::write(path, b"\x7fELFbounded native fixture").expect("Native input");
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .expect("Fixture mode");
            }
            cx.update(|cx| {
                let store = settings::SettingsStore::test(cx);
                cx.set_global(store);
                theme_settings::init(theme::LoadThemes::JustBase, cx);
                project::trusted_worktrees::init(Default::default(), cx);
                settings::SettingsStore::update_global(cx, |store, cx| {
                    store.set_user_settings(&serde_json::json!({"confirm_quit":false, "session":{"trust_all_worktrees":true}, "dap":{"DoctorOwnerFixture":{"binary":adapter}}}).to_string(), cx).expect("Exact host settings");
                });
                dap::DapRegistry::global(cx).add_adapter(Arc::new(Adapter));
            });
            let filesystem = fs::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    &root,
                    serde_json::json!({"program":"native", "untouched":"original"}),
                )
                .await;
            let project = project::Project::test(filesystem, [&root as &Path], cx).await;
            let store = project.read_with(cx, |project, _| project.dap_store());
            let native = Arc::new(fs::RealFs::new(None, cx.executor()));
            store
                .update(cx, |store, _| store.diagnostic_filesystem_for_test(native))
                .expect("Native confined inspector");
            let path = project.read_with(cx, |project, cx| project::ProjectPath {
                worktree_id: project
                    .visible_worktrees(cx)
                    .next()
                    .expect("Root")
                    .read(cx)
                    .id(),
                path: util::rel_path::RelPath::empty_arc(),
            });
            let window = cx.add_window(|window, cx| {
                workspace::MultiWorkspace::test_new(project.clone(), window, cx)
            });
            let workspace = window
                .read_with(cx, |multi, _| multi.workspace().clone())
                .expect("Workspace");
            workspace.update(cx, |workspace, _| {
                workspace.set_debugger_provider(Provider(store.clone()))
            });
            Fixture {
                directory,
                project,
                path,
                window,
                workspace,
                store,
            }
        }
        async fn review(
            fixture: &Fixture,
            effects: DoctorReproductionEffects,
            duration: u64,
            cx: &mut gpui::TestAppContext,
        ) -> Result<DoctorRepairReview<DoctorDebugAction>> {
            let root = fixture.directory.path().join("project");
            let worktree = fixture.project.read_with(cx, |project, cx| {
                project
                    .worktree_for_id(fixture.path.worktree_id, cx)
                    .expect("Root")
            });
            let launch = fixture.store.update(cx, |store, cx| store.prepare_diagnostic_launch(dap::adapters::DebugTaskDefinition { label: "Selected debugger".into(), adapter: "DoctorOwnerFixture".into(), tcp_connection: None, config: serde_json::json!({"request":"launch", "program":root.join("program"), "cwd":root}) }, worktree, cx)).await?;
            let mut selection = investigation_selection();
            selection.problem = DoctorInvestigationProblem::Crash;
            cx.update(|cx| {
                prepare_debug(
                    fixture.project.clone(),
                    fixture.path.clone(),
                    launch,
                    selection,
                    effects,
                    duration,
                    1,
                    10,
                    cx,
                )
            })
            .await
        }
        fn action_report(review: &DoctorRepairReview<DoctorDebugAction>) -> DoctorReport {
            let mut report = DoctorReport::new(
                DoctorScope {
                    product: DiagnosticText::new("Copper"),
                    project: DiagnosticText::new("Selected project"),
                    worktree: DiagnosticText::new("Selected worktree"),
                    host: DiagnosticText::new("Local"),
                },
                1,
                10,
            );
            report.push(review.before.clone()).expect("Action report");
            report
        }
        fn run(
            review: DoctorRepairReview<DoctorDebugAction>,
            fixture: &Fixture,
            history: DoctorHistoryStore,
            cancelled: async_channel::Receiver<()>,
            cx: &mut gpui::TestAppContext,
        ) -> gpui::Task<Result<DoctorDebugResult>> {
            let report = action_report(&review);
            cx.update(|cx| {
                run_debug(
                    review,
                    fixture.project.clone(),
                    fixture.workspace.downgrade(),
                    fixture.window.into(),
                    report,
                    DoctorProduct::Copper,
                    history,
                    10,
                    cancelled,
                    cx,
                )
            })
        }
        #[gpui::test]
        async fn doctor_debug_exception_is_observed_redacted_and_not_a_verified_fix(
            cx: &mut gpui::TestAppContext,
        ) {
            let fixture = fixture(cx).await;
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::from_database(&database);
            let mut review = review(&fixture, reproduction_effects(), 60, cx)
                .await
                .expect("Exact installed proposal");
            assert!(
                review
                    .action
                    .exact_preview()
                    .iter()
                    .any(|line| line.contains("No variables"))
            );
            let binding = review.binding.clone();
            review.approve(&binding).expect("Explicit action approval");
            let (started, sessions) = async_channel::bounded(1);
            let _interceptor =
                project::debugger::test::intercept_debug_sessions(cx, move |client| {
                    started
                        .try_send(client.clone())
                        .expect("One selected client");
                });
            let (cancel, cancelled) = async_channel::bounded(1);
            let pending = run(review, &fixture, history.clone(), cancelled, cx);
            let client = sessions.recv().await.expect("Existing DapStore session");
            let records = history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Durable admission");
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].outcome, ReceiptOutcome::Running);
            client
                .fake_event(dap::messages::Events::Output(dap::OutputEvent {
                    output: "API_KEY=RUNTIME_SECRET_SENTINEL private raw output".into(),
                    category: None,
                    group: None,
                    variables_reference: None,
                    source: None,
                    line: None,
                    column: None,
                    data: None,
                    location_reference: None,
                }))
                .await;
            client
                .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
                    reason: dap::StoppedEventReason::Exception,
                    description: Some("RUNTIME_SECRET_SENTINEL".into()),
                    thread_id: Some(1),
                    preserve_focus_hint: None,
                    text: None,
                    all_threads_stopped: None,
                    hit_breakpoint_ids: None,
                }))
                .await;
            let result = pending.await.expect("Observed selected exception");
            assert!(matches!(
                result.evidence.observation,
                DoctorEvidenceObservation::Exception { .. }
            ));
            assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
            assert!(
                result
                    .session
                    .expect("Current-session link")
                    .read_with(cx, |session, _| session.is_terminated())
            );
            let records = DoctorHistoryStore::from_database(&database)
                .records(DoctorProduct::Copper, 11)
                .await
                .expect("Reopened history");
            assert_eq!(records[0].outcome, ReceiptOutcome::Incomplete);
            let encoded = serde_json::to_string(&records).expect("Bounded redacted receipt");
            assert!(
                !encoded.contains("RUNTIME_SECRET_SENTINEL")
                    && !encoded.contains(&fixture.directory.path().to_string_lossy().to_string())
            );
            assert_eq!(
                fixture
                    .store
                    .read_with(cx, |store, _| store.sessions().count()),
                0
            );
            assert_eq!(
                std::fs::read_to_string(fixture.directory.path().join("project/untouched"))
                    .expect("Original file"),
                "original"
            );
            drop(cancel);
        }
        #[gpui::test]
        async fn doctor_debug_zero_exit_and_unrelated_stops_are_insufficient(
            cx: &mut gpui::TestAppContext,
        ) {
            let fixture = fixture(cx).await;
            let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
            let mut review = review(&fixture, reproduction_effects(), 60, cx)
                .await
                .expect("Review");
            review.approve(&review.binding.clone()).expect("Approval");
            let (started, sessions) = async_channel::bounded(1);
            let _interceptor =
                project::debugger::test::intercept_debug_sessions(cx, move |client| {
                    started.try_send(client.clone()).expect("One client");
                });
            let (cancel, cancelled) = async_channel::bounded(1);
            let pending = run(review, &fixture, history, cancelled, cx);
            let client = sessions.recv().await.expect("Selected client");
            client
                .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
                    reason: dap::StoppedEventReason::Pause,
                    description: None,
                    thread_id: Some(1),
                    preserve_focus_hint: None,
                    text: None,
                    all_threads_stopped: None,
                    hit_breakpoint_ids: None,
                }))
                .await;
            client
                .fake_event(dap::messages::Events::Exited(dap::ExitedEvent {
                    exit_code: 0,
                }))
                .await;
            let result = pending.await.expect("Bounded lifecycle observation");
            assert_eq!(
                result.evidence.observation,
                DoctorEvidenceObservation::InsufficientEvidence
            );
            assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
            assert_eq!(
                fixture
                    .store
                    .read_with(cx, |store, _| store.sessions().count()),
                0
            );
            drop(cancel);
        }
        #[gpui::test]
        async fn doctor_debug_cancellation_deadline_revocation_and_caller_loss_are_scoped(
            cx: &mut gpui::TestAppContext,
        ) {
            let fixture = fixture(cx).await;
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::from_database(&database);
            for mode in 0..4 {
                let mut review = review(&fixture, reproduction_effects(), 60, cx)
                    .await
                    .expect("Fresh review");
                review
                    .approve(&review.binding.clone())
                    .expect("Explicit approval");
                let (started, sessions) = async_channel::bounded(1);
                let interceptor =
                    project::debugger::test::intercept_debug_sessions(cx, move |client| {
                        started
                            .try_send(client.clone())
                            .expect("Selected owned client");
                    });
                let (cancel, cancelled) = async_channel::bounded(1);
                let pending = run(review, &fixture, history.clone(), cancelled, cx);
                let _client = sessions.recv().await.expect("Owned session");
                match mode {
                    0 => {
                        cancel.close();
                    }
                    1 => {
                        cx.executor()
                            .advance_clock(std::time::Duration::from_secs(61));
                    }
                    2 => {
                        cx.update(|cx| {
                            settings::SettingsStore::update_global(cx, |store, cx| {
                                store.update_user_settings(cx, |settings| {
                                    settings
                                        .project
                                        .dap
                                        .get_mut("DoctorOwnerFixture")
                                        .expect("Fixture adapter")
                                        .args = Some(Vec::new());
                                })
                            })
                        });
                    }
                    _ => {}
                }
                if mode == 3 {
                    drop(pending);
                    cx.run_until_parked();
                } else {
                    let result = pending
                        .await
                        .expect("Reached action has an attributable incomplete result");
                    assert_eq!(
                        result.evidence.observation,
                        if mode == 0 {
                            DoctorEvidenceObservation::Cancelled
                        } else {
                            DoctorEvidenceObservation::Incomplete
                        }
                    );
                    assert_ne!(result.review.phase(), DoctorRepairPhase::Verified);
                }
                cx.run_until_parked();
                assert_eq!(
                    fixture
                        .store
                        .read_with(cx, |store, _| store.sessions().count()),
                    0
                );
                drop(interceptor);
            }
            let records = DoctorHistoryStore::from_database(&database)
                .records(DoctorProduct::Copper, 100)
                .await
                .expect("Durable outcomes");
            assert_eq!(records.len(), 4);
            assert!(records.iter().all(|record| matches!(
                record.outcome,
                ReceiptOutcome::Incomplete | ReceiptOutcome::Cancelled
            )));
        }
        #[gpui::test]
        async fn doctor_debug_rejects_unapproved_unknown_and_stale_actions(
            cx: &mut gpui::TestAppContext,
        ) {
            let fixture = fixture(cx).await;
            assert!(
                review(&fixture, DoctorReproductionEffects::Unknown, 60, cx)
                    .await
                    .is_err()
            );
            assert!(
                review(
                    &fixture,
                    DoctorReproductionEffects::SensitiveCapture,
                    60,
                    cx
                )
                .await
                .is_err()
            );
            assert!(
                review(&fixture, reproduction_effects(), 301, cx)
                    .await
                    .is_err()
            );
            let history = DoctorHistoryStore::from_database(&db::AppDatabase::test_new());
            let proposed = review(&fixture, reproduction_effects(), 60, cx)
                .await
                .expect("Proposed review");
            let (cancel, cancelled) = async_channel::bounded(1);
            assert!(
                run(proposed, &fixture, history.clone(), cancelled.clone(), cx)
                    .await
                    .is_err()
            );
            assert!(
                history
                    .records(DoctorProduct::Copper, 10)
                    .await
                    .expect("History")
                    .is_empty()
            );
            let mut approved = review(&fixture, reproduction_effects(), 60, cx)
                .await
                .expect("Review");
            approved
                .approve(&approved.binding.clone())
                .expect("One approval");
            assert!(approved.approve(&approved.binding.clone()).is_err());
            std::fs::write(
                fixture.directory.path().join("project/program"),
                b"\x7fELFchanged native input",
            )
            .expect("Changed executable");
            let result = run(approved, &fixture, history.clone(), cancelled, cx)
                .await
                .expect("Durable stale owner failure");
            assert_eq!(
                result.evidence.observation,
                DoctorEvidenceObservation::Incomplete
            );
            assert_eq!(
                fixture
                    .store
                    .read_with(cx, |store, _| store.sessions().count()),
                0
            );
            drop(cancel);
        }
    }
}
