use std::{
    collections::{BTreeMap, BTreeSet},
    io::{ErrorKind, Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, bail, ensure};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use crate::spec_review::{ReviewPass, ReviewResult, Severity};

pub(crate) const WORKFLOW_INSTRUCTIONS: &str = include_str!("templates/spec_workflow.md");
const PRINCIPLES: &str = ".specify/memory/constitution.md";
const FEATURE_SELECTION: &str = ".specify/feature.json";
const MAX_DOCUMENT: usize = 96 * 1024;
const MAX_STATE: usize = 2 * 1024 * 1024;
const MAX_REFERENCES: usize = 12;
const MAX_REFERENCE_BYTES: usize = 48 * 1024;
static ARTIFACT_REPLACEMENT: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

pub fn specification_for_editor(
    root: PathBuf,
    path: &Path,
) -> Result<Option<SpecificationOverview>> {
    let specification = match path.file_name().and_then(|name| name.to_str()) {
        Some("plan.md" | "tasks.md" | "verification.md") => path.with_file_name("spec.md"),
        Some("spec.md") => path.to_path_buf(),
        _ => return Ok(None),
    };
    let Some(relative) = specification
        .strip_prefix(&root)
        .ok()
        .and_then(Path::to_str)
    else {
        return Ok(None);
    };
    let relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
    let project = SpecProject::open(root)?;
    if project.read(&relative, MAX_DOCUMENT)?.is_none() {
        return Ok(None);
    }
    let snapshot = project.snapshot(&relative)?;
    project.overview(&snapshot).map(Some)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecificationPhase {
    Draft,
    NeedsReview,
    ResolveFindings,
    AwaitingApproval,
    ReadyToPlan,
    Implementation,
    Verification,
    Verified,
}

#[derive(Clone)]
pub struct SpecificationOverview {
    pub path: PathBuf,
    pub title: String,
    pub phase: SpecificationPhase,
    pub revision: String,
    pub reviewed: Option<String>,
    pub approved: Option<String>,
    pub findings: usize,
    pub review_session: Option<SpecificationSession>,
    pub implementation_ready: bool,
    pub plan_path: Option<PathBuf>,
    pub tasks_path: Option<PathBuf>,
    pub verification_path: Option<PathBuf>,
    pub tasks: Vec<SpecificationTask>,
    pub criteria: Vec<SpecificationCriterion>,
}

#[derive(Clone)]
pub struct SpecificationTask {
    pub id: String,
    pub description: String,
    pub complete: bool,
    pub blocked_by: Vec<String>,
    pub verified: bool,
    pub latest_check: Option<String>,
    pub session: Option<SpecificationSession>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpecificationSession {
    pub session_id: String,
    pub tool_call_id: String,
}

pub struct SpecificationTaskRow {
    pub offset: usize,
    pub title_end: usize,
    pub end: usize,
    pub references: Vec<(std::ops::Range<usize>, String)>,
    pub indent: usize,
    pub id: String,
    pub description: String,
    pub complete: bool,
    pub dependencies: BTreeSet<String>,
    pub criteria: BTreeSet<String>,
    pub properties: BTreeSet<String>,
}

#[derive(Clone)]
pub struct SpecificationCriterion {
    pub id: String,
    pub passed: Option<bool>,
}

#[derive(Clone, Serialize)]
pub(crate) struct Document {
    pub path: String,
    pub text: String,
}

impl Document {
    pub fn new(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            text: text.into(),
        }
    }

    fn digest(&self) -> String {
        fingerprint(&self.text)
    }
}

fn fingerprint(text: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(text.replace("\r\n", "\n").as_bytes())
    )
}

#[derive(Clone)]
pub(crate) struct SpecProject {
    root: PathBuf,
    directory: Arc<Dir>,
}

#[derive(Clone)]
pub(crate) struct SpecBinding {
    pub project: SpecProject,
    pub path: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkflowState {
    pub version: u32,
    pub specification: String,
    pub reviews: Vec<ReviewRecord>,
    pub approvals: Vec<Approval>,
    pub plan: Option<PreparedPlan>,
    pub verification: Vec<Verification>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub task_sessions: BTreeMap<String, TaskSession>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct TaskSession {
    pub revision: String,
    pub session: SpecificationSession,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReviewRecord {
    pub id: Uuid,
    pub revision: String,
    pub reviewed_at: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub references: BTreeMap<String, String>,
    pub passes: Vec<(ReviewPass, ReviewResult)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SpecificationSession>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Approval {
    pub revision: String,
    pub review: Uuid,
    pub approved_at: DateTime<Utc>,
    pub accepted_trade_offs: BTreeMap<String, String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedPlan {
    revision: String,
    plan_digest: String,
    tasks_digest: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TerminalEvidence {
    pub id: String,
    pub command: String,
    pub working_directory: String,
    pub exit_code: Option<u32>,
    pub passed: bool,
    pub output: String,
    pub observed_at: DateTime<Utc>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Verification {
    pub revision: String,
    pub task: String,
    pub criteria: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<String>,
    pub evidence: TerminalEvidence,
}

pub(crate) struct Snapshot {
    pub document: Document,
    pub principles: Vec<Document>,
    pub revision: String,
    pub state: WorkflowState,
    state_source: Option<String>,
}

#[derive(Deserialize)]
struct FeatureSelection {
    feature_directory: String,
}

impl SpecProject {
    pub fn open(root: PathBuf) -> Result<Self> {
        let directory = Arc::new(
            Dir::open_ambient_dir(&root, ambient_authority())
                .context("Open specification project")?,
        );
        Ok(Self { root, directory })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn read(&self, path: &str, limit: usize) -> Result<Option<Document>> {
        let relative = checked_path(path)?;
        let metadata = match self.directory.symlink_metadata(relative) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("Inspect {path}")),
        };
        ensure!(
            metadata.is_file() && !metadata.is_symlink() && metadata.len() <= limit as u64,
            "{path} must be a regular project file of at most {limit} bytes"
        );
        // The capability confines races and symlink traversal to the retained project root.
        ensure!(
            self.directory.canonicalize(relative)? == relative,
            "{path} uses a path alias"
        );
        let mut bytes = Vec::new();
        self.directory
            .open(relative)?
            .take((limit + 1) as u64)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= limit, "{path} exceeds {limit} bytes");
        Ok(Some(Document::new(
            path,
            String::from_utf8(bytes).with_context(|| format!("{path} is not UTF-8"))?,
        )))
    }

    fn require(&self, path: &str) -> Result<Document> {
        self.read(path, MAX_DOCUMENT)?
            .with_context(|| format!("Cannot find {path}"))
    }

    fn replace(
        &self,
        path: &str,
        expected: Option<&str>,
        content: &str,
        limit: usize,
    ) -> Result<()> {
        ensure!(content.len() <= limit, "{path} exceeds {limit} bytes");
        let target = checked_path(path)?;
        let _replacement = ARTIFACT_REPLACEMENT.lock();
        let temporary = target.with_file_name(format!(".made-spec-{}.tmp", Uuid::new_v4()));
        let mut file = self
            .directory
            .open_with(&temporary, OpenOptions::new().write(true).create_new(true))?;
        let result = (|| {
            file.write_all(content.as_bytes())?;
            if expected.is_some() {
                file.set_permissions(self.directory.metadata(target)?.permissions())?;
            }
            file.sync_all()?;
            drop(file);
            let actual = self.read(path, limit)?;
            ensure!(
                actual.as_ref().map(|document| document.text.as_str()) == expected,
                "{path} changed on disk. Reload the specification before continuing."
            );
            if expected.is_none() {
                self.directory
                    .hard_link(&temporary, &self.directory, target)?;
            } else {
                self.directory.rename(&temporary, &self.directory, target)?;
            }
            Ok(())
        })();
        match self.directory.remove_file(&temporary) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => log::warn!("Cannot clean up a specification temporary file: {error}"),
        }
        result
    }

    pub fn locate(&self, explicit: Option<&str>) -> Result<String> {
        if let Some(path) = explicit {
            ensure!(
                checked_path(path)?
                    .extension()
                    .is_some_and(|extension| extension == "md"),
                "Select a project-relative Markdown specification"
            );
            self.require(path)?;
            return Ok(path.to_owned());
        }
        if let Some(selection) = self.read(FEATURE_SELECTION, MAX_DOCUMENT)? {
            let selection: FeatureSelection =
                serde_json::from_str(&selection.text).context("Read Spec Kit feature selection")?;
            let path = format!(
                "{}/spec.md",
                selection.feature_directory.trim_end_matches('/')
            );
            return self.locate(Some(&path));
        }
        let mut specs = self.specs()?;
        match specs.len() {
            0 => bail!(
                "No specification found. Start with a feature request or select a Markdown file."
            ),
            1 => specs.pop().context("Specification selection changed"),
            _ => bail!(
                "Choose a specification by its project-relative path: {}",
                specs.join(", ")
            ),
        }
    }

    fn specs(&self) -> Result<Vec<String>> {
        let entries = match self.directory.read_dir("specs") {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error).context("List project specifications"),
        };
        let mut paths = Vec::new();
        for entry in entries.take(512) {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                let path = format!("specs/{name}/spec.md");
                if self
                    .directory
                    .symlink_metadata(&path)
                    .is_ok_and(|metadata| metadata.is_file())
                {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        Ok(paths)
    }

    pub fn principles(&self) -> Result<Vec<Document>> {
        [PRINCIPLES, "AGENTS.md", ".rules"]
            .into_iter()
            .map(|path| self.read(path, MAX_DOCUMENT))
            .filter_map(Result::transpose)
            .collect()
    }

    pub fn references(&self, paths: &[String], specification: &str) -> Result<Vec<Document>> {
        ensure!(
            paths.len() <= MAX_REFERENCES,
            "Select at most {MAX_REFERENCES} reference files"
        );
        let mut total = 0;
        let mut selected = BTreeSet::new();
        let mut references = Vec::new();
        for path in paths {
            ensure!(
                path != specification,
                "The specification is already included"
            );
            if selected.insert(path) {
                let document = self
                    .read(path, MAX_REFERENCE_BYTES)?
                    .with_context(|| format!("Cannot find reference {path}"))?;
                total += document.text.len();
                ensure!(
                    total <= MAX_REFERENCE_BYTES,
                    "Selected references exceed {MAX_REFERENCE_BYTES} bytes"
                );
                references.push(document);
            }
        }
        Ok(references)
    }

    pub fn draft(&self, slug: &str, text: &str) -> Result<String> {
        ensure!(
            (1..=72).contains(&slug.len())
                && slug
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && slug
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
            "Use a lowercase feature name with letters, numbers and hyphens"
        );
        validate_spec(text)?;
        match self.directory.create_dir("specs") {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                ensure!(
                    self.directory.symlink_metadata("specs")?.is_dir(),
                    "specs must be a project directory"
                );
            }
            Err(error) => return Err(error).context("Create specifications directory"),
        }
        let existing = self.specs()?;
        let next = existing
            .iter()
            .filter_map(|path| {
                path.split('/')
                    .nth(1)?
                    .split('-')
                    .next()?
                    .parse::<u32>()
                    .ok()
            })
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .context("No feature number is available")?;
        let folder = format!("specs/{next:03}-{slug}");
        self.directory
            .create_dir(&folder)
            .with_context(|| format!("Create {folder}"))?;
        let path = format!("{folder}/spec.md");
        if let Err(error) = self.replace(&path, None, text, MAX_DOCUMENT) {
            if let Err(cleanup) = self.directory.remove_dir(&folder) {
                log::warn!("Cannot remove empty feature directory {folder}: {cleanup}");
            }
            return Err(error);
        }
        Ok(path)
    }

    pub fn snapshot(&self, path: &str) -> Result<Snapshot> {
        let document = self.require(path)?;
        let principles = self.principles()?;
        let revision = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                document.digest(),
                principles
                    .iter()
                    .map(|principle| (&principle.path, principle.digest()))
                    .collect::<Vec<_>>(),
            ))?)
        );
        let source = self.read(&state_path(path)?, MAX_STATE)?;
        let state = if let Some(source) = &source {
            let state: WorkflowState =
                serde_json::from_str(&source.text).context("Read Made workflow state")?;
            ensure!(
                state.version == 1 && state.specification == path,
                "Workflow state belongs to a different specification or uses an unsupported format"
            );
            state
        } else {
            WorkflowState {
                version: 1,
                specification: path.to_owned(),
                reviews: Vec::new(),
                approvals: Vec::new(),
                plan: None,
                verification: Vec::new(),
                task_sessions: BTreeMap::new(),
            }
        };
        Ok(Snapshot {
            document,
            principles,
            revision,
            state,
            state_source: source.map(|source| source.text),
        })
    }

    pub fn check_current(&self, snapshot: &Snapshot) -> Result<()> {
        let current = self.snapshot(&snapshot.document.path)?;
        ensure!(
            current.revision == snapshot.revision && current.state_source == snapshot.state_source,
            "The specification, principles, or workflow state changed. Reload before continuing."
        );
        Ok(())
    }

    pub fn save(&self, snapshot: &Snapshot) -> Result<()> {
        self.check_current(snapshot)?;
        self.replace(
            &state_path(&snapshot.document.path)?,
            snapshot.state_source.as_deref(),
            &(serde_json::to_string_pretty(&snapshot.state)? + "\n"),
            MAX_STATE,
        )
    }

    pub fn revise(&self, snapshot: &Snapshot, text: &str) -> Result<()> {
        if snapshot.document.path.rsplit('/').next() == Some("spec.md") {
            validate_spec(text)?;
        } else {
            ensure!(!text.trim().is_empty(), "The revised document is empty");
        }
        self.check_current(snapshot)?;
        self.replace(
            &snapshot.document.path,
            Some(&snapshot.document.text),
            text,
            MAX_DOCUMENT,
        )
    }

    pub fn prepare(
        &self,
        snapshot: &mut Snapshot,
        plan: Option<&str>,
        tasks: Option<&str>,
    ) -> Result<()> {
        snapshot.require_approval()?;
        let plan_path = companion(&snapshot.document.path, "plan.md")?;
        let tasks_path = companion(&snapshot.document.path, "tasks.md")?;
        let current_plan = self.read(&plan_path, MAX_DOCUMENT)?;
        let current_tasks = self.read(&tasks_path, MAX_DOCUMENT)?;
        ensure!(
            current_plan.is_none() || plan.is_none(),
            "Existing plan is user-owned. Review it in place."
        );
        ensure!(
            current_tasks.is_none() || tasks.is_none(),
            "Existing tasks are user-owned. Review them in place."
        );
        let plan = plan
            .or_else(|| current_plan.as_ref().map(|document| document.text.as_str()))
            .context("Provide a plan")?;
        let tasks = tasks
            .or_else(|| {
                current_tasks
                    .as_ref()
                    .map(|document| document.text.as_str())
            })
            .context("Provide a task list")?;
        ensure!(!plan.trim().is_empty(), "The plan is empty");
        let criteria = criteria_ids(&snapshot.document.text);
        let properties = specification_properties(&snapshot.document.text)?;
        let task_list = parse_tasks(tasks)?;
        ensure!(
            !criteria.is_empty(),
            "Give the specification stable AC-001 or SC-001 acceptance IDs"
        );
        ensure!(
            !task_list.is_empty(),
            "Use Spec Kit task checkboxes with T001-style IDs"
        );
        ensure!(
            criteria.iter().all(|criterion| task_list
                .values()
                .any(|task| criteria_ids(task).contains(criterion))),
            "Every acceptance criterion must appear in at least one task"
        );
        let mut assigned_properties = BTreeSet::new();
        for (task, description) in &task_list {
            let criteria = criteria_ids(description);
            for property in property_ids(description) {
                let linked = properties.get(&property).with_context(|| {
                    format!("Task {task} references unknown property {property}")
                })?;
                ensure!(
                    !linked.is_disjoint(&criteria),
                    "Task {task} must cite an acceptance criterion linked to {property}"
                );
                assigned_properties.insert(property);
            }
        }
        ensure!(
            properties.keys().all(|id| assigned_properties.contains(id)),
            "Every declared correctness property must be assigned to a task with its acceptance criterion"
        );
        self.check_current(snapshot)?;
        if current_plan.is_none() {
            self.replace(&plan_path, None, plan, MAX_DOCUMENT)?;
        }
        if current_tasks.is_none() {
            self.replace(&tasks_path, None, tasks, MAX_DOCUMENT)?;
        }
        snapshot.state.plan = Some(PreparedPlan {
            revision: snapshot.revision.clone(),
            plan_digest: fingerprint(plan),
            tasks_digest: task_digest(tasks),
        });
        self.save(snapshot)
    }

    pub fn write_supporting_documents(
        &self,
        snapshot: &Snapshot,
        documents: BTreeMap<String, String>,
    ) -> Result<()> {
        snapshot.require_approval()?;
        ensure!(
            documents.len() <= 16,
            "Include at most 16 supporting artifacts"
        );
        let mut new_documents = Vec::new();
        for (name, text) in documents {
            let path = checked_path(&name)?;
            let allowed = matches!(
                name.as_str(),
                "research.md" | "data-model.md" | "quickstart.md"
            ) || path.parent() == Some(Path::new("contracts"))
                && path.extension().is_some_and(|extension| {
                    matches!(extension.to_str(), Some("md" | "json" | "yaml" | "yml"))
                });
            ensure!(allowed, "{name} is not a supported planning artifact");
            ensure!(
                !text.trim().is_empty() && text.len() <= MAX_DOCUMENT,
                "{name} must be nonempty and at most {MAX_DOCUMENT} bytes"
            );
            let relative = companion(&snapshot.document.path, &name)?;
            if let Some(existing) = self.read(&relative, MAX_DOCUMENT)? {
                ensure!(existing.text == text, "Existing {relative} is user-owned");
            } else {
                new_documents.push((relative, text));
            }
        }
        self.check_current(snapshot)?;
        if new_documents
            .iter()
            .any(|(path, _)| path.contains("/contracts/") || path.starts_with("contracts/"))
        {
            let directory = companion(&snapshot.document.path, "contracts")?;
            match self.directory.create_dir(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                    ensure!(
                        self.directory.symlink_metadata(&directory)?.is_dir()
                            && self.directory.canonicalize(&directory)? == Path::new(&directory),
                        "Contracts must be stored in the feature's own directory"
                    );
                }
                Err(error) => return Err(error).context("Create contracts directory"),
            }
        }
        for (path, text) in new_documents {
            self.replace(&path, None, &text, MAX_DOCUMENT)?;
        }
        Ok(())
    }

    pub fn implementation_ready(&self, snapshot: &Snapshot) -> Result<Document> {
        snapshot.require_approval()?;
        let plan = snapshot
            .state
            .plan
            .as_ref()
            .filter(|plan| plan.revision == snapshot.revision)
            .context(
                "Prepare the plan and tasks from the approved revision before implementation",
            )?;
        let current_plan = self.require(&companion(&snapshot.document.path, "plan.md")?)?;
        let tasks = self.require(&companion(&snapshot.document.path, "tasks.md")?)?;
        ensure!(
            current_plan.digest() == plan.plan_digest
                && task_digest(&tasks.text) == plan.tasks_digest,
            "Plan or task content changed. Review and register the current files against the approval."
        );
        Ok(tasks)
    }

    pub fn registered_task(&self, snapshot: &Snapshot, id: &str) -> Result<String> {
        let tasks = self.implementation_ready(snapshot)?;
        let rows = specification_task_rows(&tasks.text);
        let task = rows
            .iter()
            .find(|task| task.id == id)
            .context("Task was not found")?;
        let blocked = blocked_dependencies(task, &rows);
        ensure!(
            blocked.is_empty(),
            "Task {id} is blocked by {}",
            blocked.join(", ")
        );
        Ok(task.description.clone())
    }

    pub fn complete_task(&self, snapshot: &Snapshot, id: &str) -> Result<()> {
        let tasks = self.implementation_ready(snapshot)?;
        self.registered_task(snapshot, id)?;
        let rows = specification_task_rows(&tasks.text);
        let task = rows
            .iter()
            .find(|task| task.id == id && !task.complete)
            .with_context(|| format!("No unchecked task {id}"))?;
        let criteria = &task.criteria;
        let declared = criteria_ids(&snapshot.document.text);
        ensure!(
            criteria
                .iter()
                .all(|criterion| declared.contains(criterion)),
            "Task {id} has an unknown acceptance criterion"
        );
        ensure!(
            task_evidence_passed(snapshot, task),
            "Record a passing observed check for every acceptance criterion and property on {id}"
        );
        self.check_current(snapshot)?;
        let checkbox = task.offset + task.indent + 3;
        let mut completed = tasks.text.clone();
        completed.replace_range(checkbox..checkbox + 1, "x");
        self.replace(&tasks.path, Some(&tasks.text), &completed, MAX_DOCUMENT)
    }

    pub fn record_verification(&self, snapshot: &mut Snapshot, check: Verification) -> Result<()> {
        let description = self.registered_task(snapshot, &check.task)?;
        let declared = criteria_ids(&snapshot.document.text);
        let assigned = criteria_ids(&description);
        ensure!(
            !check.criteria.is_empty()
                && check
                    .criteria
                    .iter()
                    .all(|id| declared.contains(id) && assigned.contains(id)),
            "Evidence can only cover acceptance criteria assigned to {}",
            check.task
        );
        if !check.properties.is_empty() {
            let properties = specification_properties(&snapshot.document.text)?;
            let assigned = property_ids(&description);
            for property in &check.properties {
                ensure!(
                    assigned.contains(property)
                        && properties.get(property).is_some_and(|criteria| check
                            .criteria
                            .iter()
                            .any(|id| criteria.contains(id))),
                    "Evidence for {property} must cite its acceptance criterion and be recorded on an assigned task"
                );
            }
        }
        ensure!(
            !snapshot
                .state
                .verification
                .iter()
                .any(|item| item.evidence.id == check.evidence.id),
            "This terminal receipt was already recorded"
        );
        snapshot.state.verification.push(check);
        self.save(snapshot)
    }

    pub fn overview(&self, snapshot: &Snapshot) -> Result<SpecificationOverview> {
        let document_path = &snapshot.document.path;
        let plan = companion(document_path, "plan.md")
            .ok()
            .map(|path| self.read(&path, MAX_DOCUMENT))
            .transpose()?
            .flatten();
        let task_document = companion(document_path, "tasks.md")
            .ok()
            .map(|path| self.read(&path, MAX_DOCUMENT))
            .transpose()?
            .flatten();
        let verification = companion(document_path, "verification.md")
            .ok()
            .map(|path| self.read(&path, MAX_DOCUMENT))
            .transpose()?
            .flatten();
        let implementation_ready = self.implementation_ready(snapshot).is_ok();
        let rows = task_document
            .as_ref()
            .map_or_else(Vec::new, |document| specification_task_rows(&document.text));
        let tasks = rows
            .iter()
            .map(|row| {
                let latest_check = snapshot
                    .state
                    .verification
                    .iter()
                    .rev()
                    .find(|check| check.revision == snapshot.revision && check.task == row.id)
                    .map(|check| check.evidence.observed_at.to_string());
                let verified = snapshot.approved()
                    && implementation_ready
                    && row.complete
                    && !row.criteria.is_empty()
                    && task_evidence_passed(snapshot, row);
                SpecificationTask {
                    id: row.id.clone(),
                    description: row
                        .description
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned(),
                    complete: row.complete,
                    blocked_by: blocked_dependencies(row, &rows),
                    verified,
                    latest_check,
                    session: snapshot
                        .state
                        .task_sessions
                        .get(&row.id)
                        .filter(|entry| entry.revision == snapshot.revision)
                        .map(|entry| entry.session.clone()),
                }
            })
            .collect::<Vec<_>>();
        let criteria = criteria_ids(&snapshot.document.text)
            .into_iter()
            .map(|id| {
                let passed = snapshot
                    .state
                    .verification
                    .iter()
                    .rev()
                    .find(|item| item.revision == snapshot.revision && item.criteria.contains(&id))
                    .map(|item| item.evidence.passed);
                SpecificationCriterion { id, passed }
            })
            .collect::<Vec<_>>();
        let review = snapshot.current_review();
        let last_design_review = snapshot.state.reviews.iter().rev().find(|record| {
            record
                .passes
                .iter()
                .map(|(pass, _)| *pass)
                .eq(ReviewPass::DESIGN)
        });
        let reviewed = last_design_review.map(|review| review.revision.clone());
        let approved = snapshot
            .state
            .approvals
            .last()
            .map(|approval| approval.revision.clone());
        let phase = if snapshot.approved() {
            if !implementation_ready {
                SpecificationPhase::ReadyToPlan
            } else if tasks.iter().any(|task| !task.complete) {
                SpecificationPhase::Implementation
            } else if criteria.is_empty()
                || criteria
                    .iter()
                    .any(|criterion| criterion.passed != Some(true))
            {
                SpecificationPhase::Verification
            } else {
                SpecificationPhase::Verified
            }
        } else if let Some(review) = review {
            if review.has_blockers() {
                SpecificationPhase::ResolveFindings
            } else {
                SpecificationPhase::AwaitingApproval
            }
        } else if reviewed.is_some() || approved.is_some() {
            SpecificationPhase::NeedsReview
        } else {
            SpecificationPhase::Draft
        };
        let title = snapshot
            .document
            .text
            .lines()
            .find_map(|line| line.strip_prefix("# "))
            .map(|title| {
                title
                    .trim_start_matches("Feature Specification: ")
                    .to_owned()
            })
            .unwrap_or_else(|| document_path.to_owned());
        Ok(SpecificationOverview {
            path: self.root.join(document_path),
            title,
            phase,
            revision: snapshot.revision.clone(),
            reviewed,
            approved,
            findings: review.map(|review| review.findings().count()).unwrap_or(0),
            review_session: review
                .or(last_design_review)
                .and_then(|review| review.session.clone()),
            implementation_ready,
            plan_path: plan.map(|document| self.root.join(document.path)),
            tasks_path: task_document.map(|document| self.root.join(document.path)),
            verification_path: verification.map(|document| self.root.join(document.path)),
            tasks,
            criteria,
        })
    }

    pub fn report(&self, snapshot: &Snapshot) -> Result<String> {
        let tasks = companion(&snapshot.document.path, "tasks.md")
            .ok()
            .map(|path| self.read(&path, MAX_DOCUMENT))
            .transpose()?
            .flatten();
        let progress = tasks
            .as_ref()
            .map(|document| summarize_tasks(&document.text))
            .unwrap_or_default();
        let review = snapshot.current_review();
        let principles = snapshot
            .principles
            .iter()
            .map(|principle| principle.path.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let principles = if principles.is_empty() {
            String::new()
        } else {
            format!("\n\nProject principles · {principles}")
        };
        let state = if snapshot.approved() {
            if snapshot.state.plan.is_some() && self.implementation_ready(snapshot).is_err() {
                "Approved. Review and register the current plan and task files before continuing."
            } else {
                "Approved"
            }
        } else if let Some(review) = review {
            if review.has_blockers() {
                "Resolve blocking review findings and review the revised specification."
            } else {
                "Awaiting user approval"
            }
        } else if !snapshot.state.reviews.is_empty() || !snapshot.state.approvals.is_empty() {
            "Specification or principles changed. Review the current revision."
        } else {
            "Needs independent review"
        };
        Ok(format!(
            "**{}**\n\nRevision `{}`\n\n{state}{progress}{principles}{}",
            snapshot.document.path,
            snapshot.revision,
            review.map(ReviewRecord::markdown).unwrap_or_default()
        ))
    }

    pub fn verification(&self, snapshot: &Snapshot) -> Result<String> {
        let mut report = format!(
            "# Verification\n\nSpecification: [spec.md](spec.md)\n\nRevision: `{}`\n",
            snapshot.revision
        );
        let mut latest_checks = BTreeMap::new();
        for check in &snapshot.state.verification {
            if check.revision != snapshot.revision {
                continue;
            }
            for criterion in check.criteria.iter().chain(&check.properties) {
                latest_checks.insert((check.task.as_str(), criterion.as_str()), check);
            }
        }
        for criterion in criteria_ids(&snapshot.document.text) {
            let check = snapshot.state.verification.iter().rev().find(|check| {
                check.revision == snapshot.revision && check.criteria.contains(&criterion)
            });
            let status = match check {
                Some(check) if check.evidence.passed => "Passed",
                Some(_) => "Failed or interrupted",
                None => "Unverified",
            };
            report.push_str(&format!("\n## {criterion} · {status}\n"));
        }
        for (property, criteria) in specification_properties(&snapshot.document.text)? {
            let check = snapshot.state.verification.iter().rev().find(|check| {
                check.revision == snapshot.revision && check.properties.contains(&property)
            });
            let status = match check {
                Some(check) if check.evidence.passed => "Passed",
                Some(_) => "Failed or interrupted",
                None => "Unverified",
            };
            report.push_str(&format!(
                "\n## {property} · {status}\n\nAcceptance criteria: {}\n",
                criteria.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        let mut tasks = BTreeMap::<&str, Vec<&Verification>>::new();
        for ((task, _), check) in latest_checks {
            let checks = tasks.entry(task).or_default();
            if !checks
                .iter()
                .any(|existing| existing.evidence.id == check.evidence.id)
            {
                checks.push(check);
            }
        }
        for (task, checks) in tasks {
            report.push_str(&format!("\n## Task {task}\n"));
            for check in checks {
                let exit = check
                    .evidence
                    .exit_code
                    .map_or_else(|| "unavailable".to_owned(), |code| code.to_string());
                report.push_str(&format!(
                    "\n{} · {} · exit code {exit}\n\n    {}\n",
                    check
                        .criteria
                        .iter()
                        .chain(&check.properties)
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", "),
                    check.evidence.observed_at,
                    check.evidence.command.replace('\n', "\n    ")
                ));
                let output = &check.evidence.output;
                let output = output
                    .strip_prefix("```\n")
                    .and_then(|content| content.strip_suffix("\n```"))
                    .unwrap_or(output);
                for line in output.lines() {
                    report.push_str("    ");
                    report.push_str(line);
                    report.push('\n');
                }
            }
        }
        Ok(report)
    }

    pub fn write_verification(&self, snapshot: &Snapshot) -> Result<String> {
        self.implementation_ready(snapshot)?;
        let report = self.verification(snapshot)?;
        let path = companion(&snapshot.document.path, "verification.md")?;
        let source = self.read(&path, MAX_DOCUMENT)?;
        const PREFIX: &str = "<!-- Generated by Made from made-workflow.json -->\n";
        ensure!(
            source
                .as_ref()
                .is_none_or(|source| source.text.starts_with(PREFIX)),
            "Existing verification.md is user-owned. The observed records are available in made-workflow.json."
        );
        self.check_current(snapshot)?;
        self.replace(
            &path,
            source.as_ref().map(|source| source.text.as_str()),
            &(PREFIX.to_owned() + &report),
            MAX_DOCUMENT,
        )?;
        Ok(report)
    }
}

impl Snapshot {
    pub fn current_review(&self) -> Option<&ReviewRecord> {
        self.state.reviews.iter().rev().find(|review| {
            review.revision == self.revision
                && review
                    .passes
                    .iter()
                    .map(|(pass, _)| *pass)
                    .eq(ReviewPass::DESIGN)
        })
    }

    pub fn approved(&self) -> bool {
        self.current_review().is_some_and(|review| {
            self.state
                .approvals
                .iter()
                .any(|approval| approval.revision == self.revision && approval.review == review.id)
        })
    }

    pub fn require_approval(&self) -> Result<()> {
        ensure!(
            self.approved(),
            "The current specification revision needs independent review and explicit approval"
        );
        Ok(())
    }
}

impl ReviewRecord {
    fn has_blockers(&self) -> bool {
        self.passes.iter().any(|(_, result)| {
            result
                .findings
                .iter()
                .any(|finding| finding.severity == Severity::Blocker)
        })
    }

    pub fn findings(&self) -> impl Iterator<Item = (String, &crate::spec_review::Finding)> {
        self.passes
            .iter()
            .enumerate()
            .flat_map(|(pass, (_, result))| {
                result
                    .findings
                    .iter()
                    .enumerate()
                    .map(move |(finding, item)| (format!("R{}-{}", pass + 1, finding + 1), item))
            })
    }

    pub fn approval(&self, accepted: BTreeMap<String, String>) -> Result<Approval> {
        ensure!(
            !self.has_blockers(),
            "Resolve blocking findings in the spec and run a fresh review"
        );
        let findings = self.findings().collect::<BTreeMap<_, _>>();
        ensure!(
            accepted.keys().all(|id| findings.contains_key(id)),
            "An accepted finding does not belong to the current review"
        );
        ensure!(
            findings
                .iter()
                .all(|(id, finding)| finding.severity != Severity::Major
                    || accepted
                        .get(id)
                        .is_some_and(|reason| !reason.trim().is_empty())),
            "Ask the user to resolve or explicitly accept every major finding with a reason"
        );
        Ok(Approval {
            revision: self.revision.clone(),
            review: self.id,
            approved_at: Utc::now(),
            accepted_trade_offs: accepted,
        })
    }

    pub fn markdown(&self) -> String {
        let mut result = String::new();
        for (stage, (pass, response)) in self.passes.iter().enumerate() {
            result.push_str(&format!("\n\n### {}\n\n{}", pass.title(), response.summary));
            for (index, finding) in response.findings.iter().enumerate() {
                result.push_str(&format!(
                    "\n- **R{}-{} · {:?}** · {} — {}",
                    stage + 1,
                    index + 1,
                    finding.severity,
                    finding.location,
                    finding.finding
                ));
                if !finding.trade_off.is_empty() {
                    result.push_str(&format!(" Trade-off: {}", finding.trade_off));
                }
            }
        }
        result
    }
}

fn checked_path(path: &str) -> Result<&Path> {
    let relative = Path::new(path);
    ensure!(
        !path.is_empty()
            && !path.contains('\\')
            && !path.contains(':')
            && path.split('/').all(|part| !part.is_empty()
                && !matches!(part, "." | ".." | ".git" | ".env")
                && !part.starts_with(".env."))
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "Use a project-relative file path without aliases or traversal"
    );
    Ok(relative)
}

fn companion(path: &str, name: &str) -> Result<String> {
    let path = checked_path(path)?;
    ensure!(
        path.file_name().is_some_and(|file| file == "spec.md"),
        "Planning and implementation require a feature's spec.md"
    );
    sibling(path, name)
}

pub(crate) fn is_workflow_input(specification: &str, path: &str) -> bool {
    path == specification
        || matches!(path, PRINCIPLES | "AGENTS.md" | ".rules")
        || state_path(specification).is_ok_and(|state| state == path)
        || ["plan.md", "tasks.md", "verification.md"]
            .iter()
            .any(|name| companion(specification, name).is_ok_and(|artifact| artifact == path))
}

fn state_path(path: &str) -> Result<String> {
    let path = checked_path(path)?;
    if path.file_name().is_some_and(|name| name == "spec.md") {
        sibling(path, "made-workflow.json")
    } else {
        Ok(format!("{path}.made-workflow.json", path = path.display()))
    }
}

fn sibling(path: &Path, name: &str) -> Result<String> {
    let parent = path
        .parent()
        .context("Specification has no parent directory")?;
    let parent = parent.to_str().context("Specification path is not UTF-8")?;
    Ok(if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    })
}

fn validate_spec(text: &str) -> Result<()> {
    ensure!(
        !text.trim().is_empty() && text.len() <= MAX_DOCUMENT,
        "Provide a nonempty specification under {MAX_DOCUMENT} bytes"
    );
    ensure!(
        !criteria_ids(text).is_empty(),
        "Include stable acceptance IDs such as AC-001"
    );
    specification_properties(text)?;
    Ok(())
}

pub(crate) fn reference_digests(references: &[Document]) -> BTreeMap<String, String> {
    references
        .iter()
        .map(|document| (document.path.clone(), document.digest()))
        .collect()
}

pub(crate) fn criteria_ids(text: &str) -> BTreeSet<String> {
    let acceptance = ids(text, "AC-");
    if acceptance.is_empty() {
        ids(text, "SC-")
    } else {
        acceptance
    }
}

fn property_ids(text: &str) -> BTreeSet<String> {
    ids(text, "PROP-")
}

fn specification_properties(text: &str) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let mut properties = BTreeMap::new();
    let mut references = BTreeSet::new();
    let mut criteria = BTreeSet::new();
    for line in text.lines() {
        let declaration = line.trim_start_matches([' ', '\t', '-', '*', '+', '#', '`']);
        let property = declaration
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '-')
            .next()
            .filter(|id| !property_ids(id).is_empty());
        if let Some(property) = property {
            let links = criteria_ids(line);
            ensure!(
                !links.is_empty(),
                "{property} must link to an acceptance criterion"
            );
            references.extend(links.iter().cloned());
            ensure!(
                properties.insert(property.to_owned(), links).is_none(),
                "Property {property} appears more than once"
            );
        } else {
            criteria.extend(criteria_ids(line));
        }
    }
    ensure!(
        references.is_subset(&criteria),
        "Every correctness property must link to an existing acceptance criterion"
    );
    Ok(properties)
}

fn task_evidence_passed(snapshot: &Snapshot, task: &SpecificationTaskRow) -> bool {
    let checks = snapshot
        .state
        .verification
        .iter()
        .rev()
        .filter(|check| check.revision == snapshot.revision && check.task == task.id);
    task.criteria.iter().all(|id| {
        checks
            .clone()
            .find(|check| check.criteria.contains(id))
            .is_some_and(|check| check.evidence.passed)
    }) && task.properties.iter().all(|id| {
        checks
            .clone()
            .find(|check| check.properties.contains(id))
            .is_some_and(|check| check.evidence.passed)
    })
}

fn ids(text: &str, prefix: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_ascii_alphanumeric() && character != '-')
        .filter(|word| is_specification_id(word, prefix))
        .map(str::to_owned)
        .collect()
}

pub(crate) fn parse_tasks(text: &str) -> Result<BTreeMap<String, String>> {
    let mut tasks = BTreeMap::new();
    for row in specification_task_rows(text) {
        ensure!(
            !tasks.contains_key(&row.id),
            "Task {} appears more than once",
            row.id
        );
        tasks.insert(row.id, row.description);
    }
    Ok(tasks)
}

fn is_specification_id(word: &str, prefix: &str) -> bool {
    word.strip_prefix(prefix)
        .is_some_and(|suffix| suffix.len() >= 3 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn task_id(id: &str) -> bool {
    if let Some(suffix) = id.strip_prefix('T') {
        suffix.len() >= 3 && suffix.bytes().all(|byte| byte.is_ascii_digit())
    } else {
        id.split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }
}

fn task_line(line: &str) -> Option<(&str, &str, bool)> {
    let line = line.trim_start_matches([' ', '\t']);
    let marker = line.get(..6)?;
    let complete = match marker.as_bytes() {
        [b'-' | b'*' | b'+', b' ', b'[', b' ', b']', b' '] => false,
        [b'-' | b'*' | b'+', b' ', b'[', b'x' | b'X', b']', b' '] => true,
        _ => return None,
    };
    let (id, description) = line.get(6..)?.split_once(' ')?;
    let id = id.trim_end_matches('.');
    task_id(id).then_some((id, description.trim(), complete))
}

pub fn specification_task_rows(text: &str) -> Vec<SpecificationTaskRow> {
    let mut tasks: Vec<SpecificationTaskRow> = Vec::new();
    let mut offset = 0;
    let mut last_content_end = 0;
    let mut open_tasks: Vec<usize> = Vec::new();
    let mut fence = None;
    for line in text.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        let content = line.trim_start_matches([' ', '\t']);
        let indent = line.len() - content.len();
        if content.trim().is_empty() {
            continue;
        }
        if fence.is_none() {
            while let Some(&index) = open_tasks.last() {
                let task = &mut tasks[index];
                if indent > task.indent {
                    break;
                }
                task.end = last_content_end;
                open_tasks.pop();
            }
        }
        last_content_end = start + line.trim_end_matches(['\r', '\n']).len();
        if let Some(marker) = content
            .chars()
            .next()
            .filter(|character| matches!(character, '`' | '~'))
        {
            let count = content
                .chars()
                .take_while(|character| *character == marker)
                .count();
            if count >= 3 {
                if fence.is_some_and(|(open, size)| open == marker && count >= size) {
                    fence = None;
                } else if fence.is_none() {
                    fence = Some((marker, count));
                }
                continue;
            }
        }
        if fence.is_some() {
            continue;
        }
        let (task, reference_start) = if let Some((id, description, complete)) = task_line(line) {
            open_tasks.push(tasks.len());
            tasks.push(SpecificationTaskRow {
                offset: start,
                title_end: last_content_end,
                end: last_content_end,
                indent,
                id: id.to_owned(),
                description: description.to_owned(),
                complete,
                dependencies: BTreeSet::new(),
                criteria: BTreeSet::new(),
                properties: BTreeSet::new(),
                references: Vec::new(),
            });
            (tasks.last_mut(), start + indent + 6 + id.len())
        } else {
            let mut task = open_tasks.last().and_then(|&index| tasks.get_mut(index));
            if let Some(task) = task.as_mut() {
                task.description.push('\n');
                task.description.push_str(content.trim());
            }
            (task, start + indent)
        };
        if let Some(task) = task {
            let mut word_start = reference_start;
            let delimiter =
                |character: char| !character.is_ascii_alphanumeric() && character != '-';
            if let Some(detail) = text.get(reference_start..last_content_end) {
                for chunk in detail.split_inclusive(delimiter) {
                    let word = chunk.trim_end_matches(delimiter);
                    if ["AC-", "SC-", "PROP-"]
                        .iter()
                        .any(|prefix| is_specification_id(word, prefix))
                    {
                        task.references
                            .push((word_start..word_start + word.len(), word.to_owned()));
                    }
                    word_start += chunk.len();
                }
            }
        }
    }
    for index in open_tasks {
        tasks[index].end = last_content_end;
    }
    for task in &mut tasks {
        task.criteria = criteria_ids(&task.description);
        task.properties = property_ids(&task.description);
        for line in task.description.lines() {
            let lower = line.to_ascii_lowercase();
            if let Some((_, dependencies)) = lower.split_once("depends on:") {
                task.dependencies.extend(
                    dependencies
                        .split(|character: char| {
                            character.is_ascii_whitespace()
                                || matches!(character, ',' | '[' | ']' | '(' | ')' | '\x60')
                        })
                        .map(|word| word.trim_end_matches('.'))
                        .filter(|word| !word.is_empty())
                        .filter_map(|word| {
                            let id = word.to_ascii_uppercase();
                            task_id(&id).then_some(id)
                        }),
                );
            }
        }
    }
    tasks
}

fn blocked_dependencies(task: &SpecificationTaskRow, rows: &[SpecificationTaskRow]) -> Vec<String> {
    task.dependencies
        .iter()
        .filter(|dependency| {
            *dependency == &task.id
                || !rows
                    .iter()
                    .any(|row| &row.id == *dependency && row.complete)
        })
        .cloned()
        .collect()
}

fn task_digest(tasks: &str) -> String {
    let mut normalized = tasks.to_owned();
    for task in specification_task_rows(tasks) {
        let checkbox = task.offset + task.indent + 3;
        normalized.replace_range(checkbox..checkbox + 1, " ");
    }
    fingerprint(&normalized)
}

fn summarize_tasks(tasks: &str) -> String {
    let rows = specification_task_rows(tasks);
    let total = rows.len();
    let complete = rows.iter().filter(|row| row.complete).count();
    format!("\n\nTasks: {complete}/{total} complete")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_workflow_csv_invariant_requires_explicit_evidence() -> Result<()> {
        let root = tempfile::tempdir()?;
        let project = SpecProject::open(root.path().to_path_buf())?;
        let spec = "# CSV import\n\n## Acceptance criteria\n\n- AC-001 When duplicate IDs appear in a CSV import, the importer shall keep the first valid row for each ID.\n\n## Correctness properties\n\n- PROP-001 (AC-001) For any ordered sequence of rows, an ID not already in the store maps to its earliest valid row, if one exists. An invalid row never reserves an ID.\n";
        assert!(validate_spec(&spec.replace("(AC-001)", "(AC-099)")).is_err());
        let path = project.draft("csv", spec)?;
        let mut snapshot = project.snapshot(&path)?;
        let review = review_record(&snapshot.revision, None);
        snapshot
            .state
            .approvals
            .push(review.approval(BTreeMap::new())?);
        snapshot.state.reviews.push(review);
        project.save(&snapshot)?;
        let mut snapshot = project.snapshot(&path)?;
        assert!(
            project
                .prepare(
                    &mut snapshot,
                    Some("# Plan"),
                    Some("- [ ] T001 Check duplicate IDs (AC-001)")
                )
                .is_err()
        );
        project.prepare(&mut snapshot, Some("# Plan"), Some("- [ ] T001 Exercise sequences of valid and invalid duplicate rows (AC-001, PROP-001)"))?;
        let mut snapshot = project.snapshot(&path)?;
        let check = |receipt: &str, properties: Vec<String>| Verification {
            revision: snapshot.revision.clone(),
            task: "T001".into(),
            criteria: vec!["AC-001".into()],
            properties,
            evidence: TerminalEvidence {
                id: receipt.into(),
                command: "python3 -m unittest test_duplicate_sequences".into(),
                working_directory: "".into(),
                exit_code: Some(0),
                passed: true,
                output: "All duplicate sequences passed".into(),
                observed_at: Utc::now(),
            },
        };
        let invalid = check("invalid", vec!["PROP-002".into()]);
        let example_only = check("criterion", Vec::new());
        let property = check("property", vec!["PROP-001".into()]);
        assert!(project.record_verification(&mut snapshot, invalid).is_err());
        project.record_verification(&mut snapshot, example_only)?;
        assert!(project.complete_task(&snapshot, "T001").is_err());
        let mut snapshot = project.snapshot(&path)?;
        project.record_verification(&mut snapshot, property)?;
        let snapshot = project.snapshot(&path)?;
        project.complete_task(&snapshot, "T001")?;
        let snapshot = project.snapshot(&path)?;
        let report = project.write_verification(&snapshot)?;
        assert!(report.contains("PROP-001 · Passed") && report.contains("AC-001, PROP-001"));
        assert!(
            project
                .overview(&snapshot)?
                .tasks
                .iter()
                .any(|task| task.id == "T001" && task.verified)
        );
        Ok(())
    }

    #[test]
    fn spec_workflow_editor_task_ranges_stop_before_siblings_and_fenced_examples() {
        let tasks = "- [x] T001 Parent (AC-001)\n  - [ ] T002 Child (PROP-001)\n    - Example é (AC-002)\n  - Parent note (AC-003)\n\n- [ ] T003 Leaf\n\n```md\n- [ ] T004 Example\n```\n";
        let rows = specification_task_rows(tasks);
        let [parent, child, leaf] = rows.as_slice() else {
            panic!("Expected the three editable task rows");
        };
        assert_eq!(
            &tasks[parent.title_end..parent.end],
            "\n  - [ ] T002 Child (PROP-001)\n    - Example é (AC-002)\n  - Parent note (AC-003)"
        );
        assert_eq!(
            &tasks[child.title_end..child.end],
            "\n    - Example é (AC-002)"
        );
        assert_eq!(leaf.title_end, leaf.end);
        for row in [parent, child] {
            for (range, id) in &row.references {
                assert_eq!(tasks.get(range.clone()), Some(id.as_str()));
            }
        }
        assert_eq!(
            parent
                .references
                .iter()
                .map(|(_, id)| id.as_str())
                .collect::<Vec<_>>(),
            ["AC-001", "AC-003"]
        );
        assert_eq!(
            child
                .references
                .iter()
                .map(|(_, id)| id.as_str())
                .collect::<Vec<_>>(),
            ["PROP-001", "AC-002"]
        );
    }

    #[test]
    fn spec_workflow_editor_tasks_preserve_nesting_and_gate_dependencies() -> Result<()> {
        let root = tempfile::tempdir()?;
        let project = SpecProject::open(root.path().to_path_buf())?;
        let path = project.draft(
            "nested",
            "# Nested feature\n\n## Acceptance criteria\n\n- AC-001 Run the test.\n",
        )?;
        let mut snapshot = project.snapshot(&path)?;
        let review = review_record(&snapshot.revision, None);
        snapshot
            .state
            .approvals
            .push(review.approval(BTreeMap::new())?);
        snapshot.state.reviews.push(review);
        project.save(&snapshot)?;
        let mut snapshot = project.snapshot(&path)?;
        let tasks = "# Tasks\n\n- [x] T001 Manually finished (AC-001)\n  - [ ] T002 Verify nested work (Depends on: T001)\n    - AC-001\n- [ ] T003 Wait for prerequisite (Depends on: T004)\n- [ ] T004 Run prerequisite (Depends on: T002)\n\n```md\n- [ ] T005 Example only\n```\n";
        project.prepare(&mut snapshot, Some("# Plan"), Some(tasks))?;
        let mut snapshot = project.snapshot(&path)?;
        assert!(project.registered_task(&snapshot, "T003").is_err());
        assert!(project.registered_task(&snapshot, "T004").is_err());
        assert!(
            project
                .registered_task(&snapshot, "T002")?
                .contains("AC-001")
        );
        let overview = project.overview(&snapshot)?;
        assert_eq!(overview.tasks.len(), 4);
        assert!(
            overview
                .tasks
                .iter()
                .any(|task| task.id == "T001" && task.complete && !task.verified)
        );
        assert!(project.complete_task(&snapshot, "T002").is_err());
        snapshot.state.task_sessions.insert(
            "T002".into(),
            TaskSession {
                revision: snapshot.revision.clone(),
                session: SpecificationSession {
                    session_id: "local-session".into(),
                    tool_call_id: "task-turn".into(),
                },
            },
        );
        snapshot.state.verification.push(Verification {
            revision: snapshot.revision.clone(),
            task: "T002".into(),
            criteria: vec!["AC-001".into()],
            properties: Vec::new(),
            evidence: TerminalEvidence {
                id: "fixture-check".into(),
                command: "cargo test".into(),
                working_directory: "".into(),
                exit_code: Some(0),
                passed: true,
                output: "passed".into(),
                observed_at: Utc::now(),
            },
        });
        project.save(&snapshot)?;
        let snapshot = project.snapshot(&path)?;
        project.complete_task(&snapshot, "T002")?;
        let snapshot = project.snapshot(&path)?;
        assert!(
            project
                .implementation_ready(&snapshot)?
                .text
                .contains("  - [x] T002")
        );
        assert!(project.registered_task(&snapshot, "T004").is_ok());
        assert!(project.registered_task(&snapshot, "T003").is_err());
        assert!(
            project
                .write_verification(&snapshot)?
                .contains("## Task T002")
        );
        let tasks_path = root.path().join(&path).with_file_name("tasks.md");
        let overview = specification_for_editor(root.path().to_path_buf(), &tasks_path)?
            .context("Recognized task document")?;
        let finished = overview
            .tasks
            .iter()
            .find(|task| task.id == "T002")
            .context("Nested task")?;
        assert!(finished.complete && finished.verified);
        assert_eq!(
            finished
                .session
                .as_ref()
                .map(|session| session.tool_call_id.as_str()),
            Some("task-turn")
        );
        Ok(())
    }
    use crate::spec_review::Finding;

    #[test]
    fn spec_workflow_recovers_project_and_invalidates_changed_revision() -> Result<()> {
        let root = tempfile::tempdir()?;
        let project = SpecProject::open(root.path().to_path_buf())?;
        let path = project.draft(
            "csv-import",
            "# CSV import\n\n- AC-001 Keep the first valid occurrence of each ID.\n",
        )?;
        let mut snapshot = project.snapshot(&path)?;
        assert!(
            project
                .prepare(
                    &mut snapshot,
                    Some("# Plan"),
                    Some("- [ ] T001 Import AC-001")
                )
                .is_err()
        );
        let first_revision = snapshot.revision.clone();
        let review = review_record(
            &first_revision,
            Some(Finding {
                severity: Severity::Major,
                location: "AC-001".into(),
                finding: "The order of validity checks is missing".into(),
                trade_off: "Checking validity first preserves a later valid row".into(),
            }),
        );
        assert!(review.approval(BTreeMap::new()).is_err());
        snapshot.state.reviews.push(review);
        assert_eq!(
            project.overview(&snapshot)?.phase,
            SpecificationPhase::AwaitingApproval
        );
        let blocker = review_record(
            &first_revision,
            Some(Finding {
                severity: Severity::Blocker,
                location: "AC-001".into(),
                finding: "The specification never defines a valid row".into(),
                trade_off: "Define validation before implementing duplicate handling".into(),
            }),
        );
        assert!(blocker.approval(BTreeMap::new()).is_err());
        snapshot.state.reviews.push(blocker);
        assert_eq!(
            project.overview(&snapshot)?.phase,
            SpecificationPhase::ResolveFindings
        );
        assert!(
            project
                .report(&snapshot)?
                .contains("Resolve blocking review findings")
        );
        project.save(&snapshot)?;
        let snapshot = project.snapshot(&path)?;
        project.revise(
            &snapshot,
            "# CSV import\n\n- AC-001 Validate before deduplicating.\n",
        )?;
        let mut snapshot = project.snapshot(&path)?;
        assert_ne!(snapshot.revision, first_revision);
        let review = review_record(&snapshot.revision, None);
        snapshot
            .state
            .approvals
            .push(review.approval(BTreeMap::new())?);
        snapshot.state.reviews.push(review);
        project.save(&snapshot)?;
        let mut snapshot = project.snapshot(&path)?;
        project.prepare(
            &mut snapshot,
            Some("# Plan\n"),
            Some("- [ ] T001 Implement import (AC-001)\n- [ ] T002 Document CSV import\n"),
        )?;
        let snapshot = project.snapshot(&path)?;
        assert!(project.complete_task(&snapshot, "T001").is_err());
        let mut snapshot = snapshot;
        snapshot.state.verification.push(Verification {
            revision: snapshot.revision.clone(),
            task: "T001".into(),
            criteria: vec!["AC-001".into()],
            properties: Vec::new(),
            evidence: TerminalEvidence {
                id: "1".into(),
                command: "cargo test".into(),
                working_directory: "".into(),
                exit_code: Some(0),
                passed: true,
                output: "1 passed".into(),
                observed_at: Utc::now(),
            },
        });
        project.save(&snapshot)?;
        let snapshot = project.snapshot(&path)?;
        project.complete_task(&snapshot, "T001")?;
        assert!(
            project
                .write_verification(&snapshot)?
                .contains("AC-001 · Passed")
        );
        drop(project);

        let reopened = SpecProject::open(root.path().to_path_buf())?;
        let path = reopened.locate(None)?;
        let current = reopened.snapshot(&path)?;
        assert!(current.approved());
        assert_eq!(current.state.reviews.len(), 3);
        assert!(
            current
                .state
                .reviews
                .get(1)
                .is_some_and(ReviewRecord::has_blockers)
        );
        assert!(reopened.report(&current)?.contains("1/2 complete"));
        assert!(reopened.registered_task(&current, "T002").is_ok());
        std::fs::write(
            root.path().join(&path),
            "# CSV import\n\n- AC-001 Keep the last valid row.\n",
        )?;
        let edited = reopened.snapshot(&path)?;
        assert!(!edited.approved());
        assert!(reopened.implementation_ready(&edited).is_err());
        assert!(
            reopened
                .report(&edited)?
                .contains("Review the current revision")
        );
        Ok(())
    }

    #[test]
    fn spec_workflow_uses_existing_spec_kit_artifacts() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::create_dir_all(root.path().join(".specify/memory"))?;
        std::fs::create_dir_all(root.path().join("features/custom"))?;
        std::fs::write(
            root.path().join(FEATURE_SELECTION),
            r#"{"feature_directory":"features/custom"}"#,
        )?;
        std::fs::write(
            root.path().join(PRINCIPLES),
            "# Constitution\nKeep the implementation local.\n",
        )?;
        let original_spec = "# Existing project\n\n- **SC-001** Accept a valid row.\n";
        let original_tasks = "- [x] T001 Fixture (SC-001)\n- [ ] T002 Import (SC-001)\n";
        std::fs::write(root.path().join("features/custom/spec.md"), original_spec)?;
        std::fs::write(
            root.path().join("features/custom/plan.md"),
            "# Existing plan\n",
        )?;
        std::fs::write(root.path().join("features/custom/tasks.md"), original_tasks)?;
        let project = SpecProject::open(root.path().to_path_buf())?;
        let path = project.locate(None)?;
        assert_eq!(path, "features/custom/spec.md");
        let mut snapshot = project.snapshot(&path)?;
        assert_eq!(snapshot.principles.len(), 1);
        let review = review_record(&snapshot.revision, None);
        snapshot
            .state
            .approvals
            .push(review.approval(BTreeMap::new())?);
        snapshot.state.reviews.push(review);
        project.save(&snapshot)?;
        let mut snapshot = project.snapshot(&path)?;
        assert!(
            project
                .prepare(&mut snapshot, Some("Overwrite"), None)
                .is_err()
        );
        project.prepare(&mut snapshot, None, None)?;
        let registered = project.snapshot(&path)?;
        assert!(project.registered_task(&registered, "T001").is_ok());
        assert_eq!(
            std::fs::read_to_string(root.path().join(&path))?,
            original_spec
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join("features/custom/tasks.md"))?,
            original_tasks
        );
        Ok(())
    }

    fn review_record(revision: &str, finding: Option<Finding>) -> ReviewRecord {
        ReviewRecord {
            id: Uuid::new_v4(),
            revision: revision.into(),
            reviewed_at: Utc::now(),
            provider: "fake".into(),
            model: "fake".into(),
            references: BTreeMap::new(),
            session: None,
            passes: ReviewPass::DESIGN
                .into_iter()
                .map(|pass| {
                    let findings = if pass == ReviewPass::Critique {
                        finding.clone().into_iter().collect()
                    } else {
                        Vec::new()
                    };
                    (
                        pass,
                        ReviewResult {
                            summary: "Reviewed".into(),
                            findings,
                        },
                    )
                })
                .collect(),
        }
    }
}
