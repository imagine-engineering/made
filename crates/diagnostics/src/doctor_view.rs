use crate::{
    Doctor,
    doctor::{CheckOutcome, DiagnosticText, DoctorCheck, DoctorProduct, DoctorReport, DoctorScope},
};
use gpui::{
    Action, App, Context, Entity, EventEmitter, FocusHandle, Focusable, Render, SharedString,
    Subscription, Task, WeakEntity, Window,
};
use project::{DoctorExecutionHost, Project, ProjectPath};
use settings::WorktreeId;
use std::time::{SystemTime, UNIX_EPOCH};
use ui::prelude::*;
use ui::{TintColor, TreeViewItem, WithScrollbar as _};
use workspace::{
    Workspace,
    item::{Item, ItemEvent},
};

enum OwnerRepair {
    #[cfg(feature = "agentic")]
    Github(crate::doctor::DoctorRepairReview<crate::doctor_dependencies::DoctorGithubAction>),
    Reproduction(crate::doctor::DoctorRepairReview<crate::doctor::DoctorReproductionAction>),
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    Debug(crate::doctor::DoctorRepairReview<crate::doctor::DoctorDebugAction>),
    #[cfg(feature = "rust-tools")]
    Profile(crate::doctor::DoctorRepairReview<crate::doctor::DoctorProfileAction>),
    #[cfg(feature = "jvm-tools")]
    RemoteTestInspection {
        review: crate::doctor::DoctorRepairReview<crate::doctor::DoctorRemoteTestInspectionAction>,
        selection: crate::doctor::DoctorInvestigationSelection,
        effects: crate::doctor::DoctorReproductionEffects,
        duration: u64,
    },
    #[cfg(feature = "agentic")]
    Native(crate::doctor::DoctorRepairReview<crate::doctor_agent::DoctorNativeRepairAction>),
    ApplicationRestart(
        crate::doctor::DoctorRepairReview<crate::doctor::DoctorApplicationRestartAction>,
    ),
    RemoteRestart(crate::doctor::DoctorRepairReview<crate::doctor::DoctorRemoteRestartAction>),
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    Dependency(crate::doctor::DoctorRepairReview<crate::doctor::DoctorDependencyProbeAction>),
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    Advisory(crate::doctor::DoctorRepairReview<crate::doctor_dependencies::AdvisoryRefreshAction>),
    Project(crate::doctor::DoctorRepairReview<crate::doctor::DoctorProjectRepairAction>),
    #[cfg(feature = "rust-tools")]
    Rustup(crate::doctor::DoctorRepairReview<crate::doctor_install::RustupAction>),
    #[cfg(feature = "jvm-tools")]
    Mise(crate::doctor::DoctorRepairReview<crate::doctor_jvm_install::MiseAction>),
    #[cfg(feature = "jvm-tools")]
    Sdkman(crate::doctor::DoctorRepairReview<crate::doctor_jvm_install::SdkmanAction>),
    #[cfg(feature = "agentic")]
    Agent(crate::doctor::DoctorRepairReview<crate::doctor_agent::DoctorAgentRepairAction>),
    #[cfg(feature = "agentic")]
    Probe(crate::doctor::DoctorRepairReview<crate::doctor_agent::DoctorProbeAction>),
}

impl OwnerRepair {
    fn conversation_identity(&self) -> DiagnosticText {
        let identity = match self {
            Self::Reproduction(review) => format!(
                "{}; Task command: {}",
                self.target(),
                review
                    .action()
                    .task()
                    .resolved
                    .command
                    .as_deref()
                    .unwrap_or("Command unavailable")
            ),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => format!(
                "{}; arguments: {}",
                self.target(),
                review
                    .action()
                    .executable()
                    .kind()
                    .version_arguments()
                    .join(" ")
            ),
            _ => self.target(),
        };
        DiagnosticText::new(identity)
    }

    fn before(&self) -> &DoctorCheck {
        match self {
            Self::Reproduction(review) => review.before(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.before(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.before(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.before(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.before(),
            Self::Project(review) => review.before(),
            Self::ApplicationRestart(review) => review.before(),
            Self::RemoteRestart(review) => review.before(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.before(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.before(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.before(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.before(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.before(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.before(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.before(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.before(),
        }
    }
    fn effects(&self) -> &DiagnosticText {
        match self {
            Self::Reproduction(review) => review.effects(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.effects(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.effects(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.effects(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.effects(),
            Self::Project(review) => review.effects(),
            Self::ApplicationRestart(review) => review.effects(),
            Self::RemoteRestart(review) => review.effects(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.effects(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.effects(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.effects(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.effects(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.effects(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.effects(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.effects(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.effects(),
        }
    }
    fn phase(&self) -> crate::doctor::DoctorRepairPhase {
        match self {
            Self::Reproduction(review) => review.phase(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.phase(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.phase(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.phase(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.phase(),
            Self::Project(review) => review.phase(),
            Self::ApplicationRestart(review) => review.phase(),
            Self::RemoteRestart(review) => review.phase(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.phase(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.phase(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.phase(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.phase(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.phase(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.phase(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.phase(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.phase(),
        }
    }
    fn after(&self) -> Option<&DoctorCheck> {
        match self {
            Self::Reproduction(review) => review.after(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.after(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.after(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.after(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.after(),
            Self::Project(review) => review.after(),
            Self::ApplicationRestart(review) => review.after(),
            Self::RemoteRestart(review) => review.after(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.after(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.after(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.after(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.after(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.after(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.after(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.after(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.after(),
        }
    }
    fn operation_detail(&self) -> Option<&DiagnosticText> {
        match self {
            Self::Reproduction(review) => review.operation_detail(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.operation_detail(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.operation_detail(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.operation_detail(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.operation_detail(),
            Self::Project(review) => review.operation_detail(),
            Self::ApplicationRestart(review) => review.operation_detail(),
            Self::RemoteRestart(review) => review.operation_detail(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.operation_detail(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.operation_detail(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.operation_detail(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.operation_detail(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.operation_detail(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.operation_detail(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.operation_detail(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.operation_detail(),
        }
    }
    fn approve(&mut self, binding: &crate::doctor::DoctorRepairBinding) -> anyhow::Result<()> {
        match self {
            Self::Reproduction(review) => review.approve(binding),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.approve(binding),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.approve(binding),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.approve(binding),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.approve(binding),
            Self::Project(review) => review.approve(binding),
            Self::ApplicationRestart(review) => review.approve(binding),
            Self::RemoteRestart(review) => review.approve(binding),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.approve(binding),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.approve(binding),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.approve(binding),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.approve(binding),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => {
                anyhow::ensure!(
                    !review.action().requires_first_default()
                        || review.action().has_first_default_consent(),
                    "Review and separately approve first-default creation before install approval"
                );
                review.approve(binding)
            }
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.approve(binding),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.approve(binding),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.approve(binding),
        }
    }
    fn cancel(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Reproduction(review) => review.cancel_before_execution(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => review.cancel_before_execution(),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => review.cancel_before_execution(),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { review, .. } => review.cancel_before_execution(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => review.cancel_before_execution(),
            Self::Project(review) => review.cancel_before_execution(),
            Self::ApplicationRestart(review) => review.cancel_before_execution(),
            Self::RemoteRestart(review) => review.cancel_before_execution(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => review.cancel_before_execution(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.cancel_before_execution(),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => review.cancel_before_execution(),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => review.cancel_mise_proposal(),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => review.cancel_before_execution(),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => review.cancel_before_execution(),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.cancel_before_execution(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.cancel_before_execution(),
        }
    }
    fn target(&self) -> String {
        match self {
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => format!(
                "Selected dependency evidence: {:?}",
                review.action().request()
            ),
            Self::Reproduction(review) => format!(
                "Reproduce selected target: {}",
                review.action().selection().validation_target.as_str()
            ),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => format!(
                "Metadata-only debugger: {}",
                review.action().selection().validation_target.as_str()
            ),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => format!(
                "Inspect selected profile: {}",
                review.action().selection().validation_target.as_str()
            ),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { selection, .. } => format!(
                "Inspect remote test metadata before execution review: {}",
                selection.validation_target.as_str()
            ),
            #[cfg(feature = "agentic")]
            Self::Native(review) => format!(
                "Native proposal: {} / {}",
                review.action().provider(),
                review.action().model()
            ),
            Self::ApplicationRestart(review) => format!(
                "Normal Made application restart across {} affected workspaces",
                review.action().affected_workspace_count()
            ),
            Self::RemoteRestart(review) => format!(
                "Exclusive idle development server {} (PID {}); {} affected workspaces",
                review.action().server_instance(),
                review.action().server_process_id(),
                review.action().affected_workspaces().len()
            ),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => format!(
                "Selected {:?} version availability",
                review.action().executable().kind()
            ),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => format!(
                "Rustup {}: {:?}",
                if review.action().is_inspection() {
                    "offline inspection"
                } else {
                    "installation"
                },
                review.action().request()
            ),
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => format!(
                "mise {:?}: {}",
                review.action().stage(),
                review.action().request().version
            ),
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => format!(
                "SDKMAN Java installation: {}",
                review.action().request().version
            ),
            Self::Project(review) => match review.action() {
                crate::doctor::DoctorProjectRepairAction::RestartLanguageServer {
                    server,
                    name,
                } => format!("Restart language server {name}, instance {}", server.0),
                #[cfg(feature = "jvm-tools")]
                crate::doctor::DoctorProjectRepairAction::RefreshJavaProject {
                    server,
                    generation,
                } => format!(
                    "Refresh JDTLS instance {}, previous generation {generation:?}",
                    server.0
                ),
                #[cfg(feature = "jvm-tools")]
                crate::doctor::DoctorProjectRepairAction::RefreshKotlinProject {
                    server,
                    generation,
                } => format!(
                    "Refresh Kotlin LSP instance {}, previous generation {generation:?}",
                    server.0
                ),
            },
            #[cfg(feature = "agentic")]
            Self::Probe(review) => review.action().target().to_owned(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => format!(
                "Read existing GitHub evidence: {} / {}",
                review.action().selection().account(),
                review.action().selection().repository().full_name()
            ),
            #[cfg(feature = "agentic")]
            Self::Agent(review) => match review.action() {
                crate::doctor_agent::DoctorAgentRepairAction::RestartMcp { server, .. } => format!(
                    "Restart MCP server {} through its existing configuration and credential owner",
                    server.id()
                ),
                crate::doctor_agent::DoctorAgentRepairAction::RetryProductExtension {
                    id,
                    download_origin,
                    ..
                } => format!("Retry missing product extension {id} from {download_origin}"),
            },
        }
    }
    fn exact_preview(&self) -> anyhow::Result<Vec<String>> {
        match self {
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Advisory(review) => review.action().exact_preview(),
            #[cfg(feature = "agentic")]
            Self::Github(review) => review.action().exact_preview(),
            #[cfg(feature = "agentic")]
            Self::Native(review) => Ok(vec![format!("Exact selected native proposal prompt to {} / {}: {}",review.action().provider(),review.action().model(),review.action().prompt())]),
            Self::Reproduction(review) => Ok(review.action().exact_preview()),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Debug(review) => Ok(review.action().exact_preview().to_vec()),
            #[cfg(feature = "rust-tools")]
            Self::Profile(review) => Ok(review.action().exact_preview().to_vec()),
            #[cfg(feature = "jvm-tools")]
            Self::RemoteTestInspection { .. } => Ok(vec!["Read only bounded existing remote JVM report metadata (at most five MiB per phase). No test execution. Host protocol/permissions must support the requested limit. A second review and approval is required for the exact test command.".into()]),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            Self::Dependency(review) => Ok(vec![
                format!(
                    "Exact executable: {}",
                    review.action().executable().path().display()
                ),
                format!(
                    "Fixed arguments: {:?}",
                    review.action().executable().kind().version_arguments()
                ),
            ]),
            #[cfg(feature = "agentic")]
            Self::Probe(review) => Ok(review.action().exact_preview().to_vec()),
            #[cfg(feature = "rust-tools")]
            Self::Rustup(review) => {
                let action = review.action();
                Ok(vec![
                    format!("Executable: {}", action.executable().display()),
                    format!("Arguments: {:?}", action.command_arguments()),
                    format!("Download source: {}", action.source()),
                    format!(
                        "Installation destination: {}",
                        action.destination().display()
                    ),
                ])
            }
            #[cfg(feature = "jvm-tools")]
            Self::Mise(review) => {
                let action = review.action();
                let mut preview = vec![
                    format!("Executable: {}", action.executable().display()),
                    format!("Arguments: {:?}", action.command_arguments()),
                    format!("Metadata source: {}", action.metadata_source()?),
                    format!(
                        "Installation destination: {}",
                        action.destination().display()
                    ),
                ];
                if let Some(artifact) = action.artifact() {
                    preview.extend([
                        format!("Artifact: {}", artifact.source),
                        format!("SHA-256: {}", artifact.checksum),
                        format!("Expected Java release: {}", artifact.java_version),
                    ]);
                }
                Ok(preview)
            }
            #[cfg(feature = "jvm-tools")]
            Self::Sdkman(review) => {
                let action = review.action();
                let mut preview = vec![format!("Executable: {}", action.executable().display()), format!("Arguments: {:?}", action.command_arguments()), format!("Installed initialization script: {}", action.initialization_script().display()), format!("Installation destination: {}", action.destination().display()), "Official downloaded post-install hooks execute with your user permissions. This is not a sandbox.".to_owned()];
                preview.extend(
                    action
                        .sources()?
                        .into_iter()
                        .map(|source| format!("Download/hook source: {source}")),
                );
                preview.push("Five minutes / 10 MiB maximum; download, cache, metadata, official hook and JDK installation writes are approved only by this action. Failure/cancellation can leave partial files. No automatic replay or destructive rollback.".to_owned());
                if action.requires_first_default() {
                    preview.push(format!("First default: candidates/java/current will select {}. This requires separate approval.", action.request().version));
                }
                Ok(preview)
            }
            _ => Ok(Vec::new()),
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
struct SecurityObservation {
    resolved_commit: Option<String>,
    snapshot: Option<crate::doctor_dependencies::AdvisorySnapshot>,
    registry_metadata: Option<crate::doctor_dependencies::RegistryMetadata>,
    cleared_dataset: Option<crate::doctor_dependencies::AdvisoryDataset>,
    scan: Option<crate::doctor_dependencies::AdvisoryScan>,
}

struct OwnerRepairResult {
    #[cfg(feature = "agentic")]
    github: Option<(
        crate::doctor_dependencies::DoctorGithubAction,
        crate::doctor_dependencies::GithubEvidenceSnapshot,
    )>,
    review: OwnerRepair,
    proposal: Option<OwnerRepair>,
    cleanup_error: Option<DiagnosticText>,
    investigation: Option<crate::doctor::DoctorInvestigationSummary>,
    comparison: Option<crate::doctor::DoctorComparisonSnapshot>,
    task: Option<task::StructuredTaskHandle>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    advisory: Option<SecurityObservation>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    projection: Option<crate::doctor_dependencies::DoctorDependencyProjection>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    recommendations: Vec<crate::doctor_dependencies::UpdateRecommendation>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    debug_session: Option<Entity<project::debugger::session::Session>>,
    #[cfg(feature = "rust-tools")]
    profile: Option<project::profile_results_store::DiagnosticProfileResult>,
}
impl OwnerRepairResult {
    fn observed(review: OwnerRepair) -> Self {
        Self {
            review,
            proposal: None,
            cleanup_error: None,
            investigation: None,
            #[cfg(feature = "agentic")]
            github: None,
            comparison: None,
            task: None,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            advisory: None,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            projection: None,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            recommendations: Vec::new(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            debug_session: None,
            #[cfg(feature = "rust-tools")]
            profile: None,
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum SecurityKind {
    #[cfg(feature = "rust-tools")]
    Cargo,
    #[cfg(feature = "jvm-tools")]
    Maven,
    #[cfg(feature = "jvm-tools")]
    Gradle,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
struct DependencySecurityForm {
    open: bool,
    kind: SecurityKind,
    output: Entity<editor::Editor>,
    module: Entity<editor::Editor>,
    configuration: Entity<editor::Editor>,
    rustsec_commit: Entity<editor::Editor>,
    public_coordinate: Entity<editor::Editor>,
    replacement_name: Entity<editor::Editor>,
    replacement_version: Entity<editor::Editor>,
    page: usize,
    selected: Option<usize>,
    finding_filter: usize,
    finding_page: usize,
    severity: Option<String>,
    update_filter: usize,
    projection: Option<crate::doctor_dependencies::DoctorDependencyProjection>,
    resolved_commit: Option<String>,
    snapshot: Option<crate::doctor_dependencies::AdvisorySnapshot>,
    scan: Option<crate::doctor_dependencies::AdvisoryScan>,
    metadata: Option<crate::doctor_dependencies::RegistryMetadata>,
    recommendations: Vec<crate::doctor_dependencies::UpdateRecommendation>,
    baseline: Option<crate::doctor_dependencies::DoctorDependencyVerificationBaseline>,
    verification: Option<crate::doctor_dependencies::DoctorDependencyVerification>,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DependencySecurityForm {
    fn new(window: &mut Window, cx: &mut Context<DoctorView>) -> Self {
        let mut input = |placeholder: &str, value: &str, cx: &mut Context<DoctorView>| {
            cx.new(|cx| {
                let mut editor = editor::Editor::single_line(window, cx);
                editor.set_placeholder_text(placeholder, window, cx);
                editor.set_text(value, window, cx);
                editor
            })
        };
        Self {
            open: false,
            #[cfg(feature = "rust-tools")]
            kind: SecurityKind::Cargo,
            #[cfg(all(not(feature = "rust-tools"), feature = "jvm-tools"))]
            kind: SecurityKind::Maven,
            output: input(
                "New contained report path (existing files are never overwritten)",
                "",
                cx,
            ),
            module: input("Exact Gradle project path", ":", cx),
            configuration: input("Exact Gradle configuration", "runtimeClasspath", cx),
            rustsec_commit: input("Exact reviewed 40-character RustSec commit", "", cx),
            public_coordinate: input("Confirm exact selected Maven coordinate is public", "", cx),
            replacement_name: input(
                "Explicit public replacement package / group:artifact",
                "",
                cx,
            ),
            replacement_version: input(
                "Exact replacement version; availability is unverified",
                "",
                cx,
            ),
            page: 0,
            selected: None,
            finding_filter: 0,
            finding_page: 0,
            severity: None,
            update_filter: 0,
            projection: None,
            resolved_commit: None,
            snapshot: None,
            scan: None,
            metadata: None,
            recommendations: Vec::new(),
            baseline: None,
            verification: None,
        }
    }
    fn fields(&self) -> Vec<Entity<editor::Editor>> {
        vec![
            self.output.clone(),
            self.module.clone(),
            self.configuration.clone(),
            self.rustsec_commit.clone(),
            self.public_coordinate.clone(),
            self.replacement_name.clone(),
            self.replacement_version.clone(),
        ]
    }
    fn clear_current(&mut self) {
        self.projection = None;
        self.scan = None;
        self.metadata = None;
        self.recommendations.clear();
        self.selected = None;
        self.page = 0;
        self.verification = None;
        self.finding_page = 0;
        self.severity = None;
    }
    fn clear_all(&mut self) {
        self.clear_current();
        self.resolved_commit = None;
        self.snapshot = None;
        self.baseline = None;
    }
    fn set_projection(
        &mut self,
        projection: crate::doctor_dependencies::DoctorDependencyProjection,
    ) {
        self.clear_current();
        self.projection = Some(projection);
    }
    fn dataset(&self) -> crate::doctor_dependencies::AdvisoryDataset {
        match self.kind {
            #[cfg(feature = "rust-tools")]
            SecurityKind::Cargo => crate::doctor_dependencies::AdvisoryDataset::RustSec,
            #[cfg(feature = "jvm-tools")]
            SecurityKind::Maven | SecurityKind::Gradle => {
                crate::doctor_dependencies::AdvisoryDataset::OsvMaven
            }
        }
    }
    fn format(&self, cx: &App) -> crate::doctor_dependencies::DoctorDependencyReportFormat {
        #[cfg(not(feature = "jvm-tools"))]
        let _ = cx;
        match self.kind {
            #[cfg(feature = "rust-tools")]
            SecurityKind::Cargo => {
                crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata
            }
            #[cfg(feature = "jvm-tools")]
            SecurityKind::Maven => {
                crate::doctor_dependencies::DoctorDependencyReportFormat::MavenSelectedTree
            }
            #[cfg(feature = "jvm-tools")]
            SecurityKind::Gradle => {
                crate::doctor_dependencies::DoctorDependencyReportFormat::GradleResolution {
                    project_path: self.module.read(cx).text(cx).trim().to_owned(),
                    configuration: self.configuration.read(cx).text(cx).trim().to_owned(),
                }
            }
        }
    }
    fn project(
        project: &Entity<Project>,
        directory: ProjectPath,
        format: &crate::doctor_dependencies::DoctorDependencyReportFormat,
        generation: u64,
        now: u64,
        cx: &App,
    ) -> anyhow::Result<crate::doctor_dependencies::DoctorDependencyProjection> {
        use crate::doctor_dependencies::{
            DoctorDependencyProjection, DoctorDependencyReportFormat,
        };
        match format {
            #[cfg(feature = "rust-tools")]
            DoctorDependencyReportFormat::CargoMetadata => DoctorDependencyProjection::cargo(
                project,
                directory.clone(),
                directory,
                generation,
                now,
                cx,
            ),
            #[cfg(feature = "jvm-tools")]
            DoctorDependencyReportFormat::MavenSelectedTree => DoctorDependencyProjection::jvm(
                project,
                directory.clone(),
                directory,
                project::jvm_project::JvmReviewedDependencyFormat::MavenSelectedTree,
                generation,
                now,
                cx,
            ),
            #[cfg(feature = "jvm-tools")]
            DoctorDependencyReportFormat::GradleResolution {
                project_path,
                configuration,
            } => DoctorDependencyProjection::jvm(
                project,
                directory.clone(),
                directory,
                project::jvm_project::JvmReviewedDependencyFormat::GradleResolution {
                    project_path: project_path.clone(),
                    configuration: configuration.clone(),
                },
                generation,
                now,
                cx,
            ),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("Selected dependency ecosystem is disabled"),
        }
    }
    fn accept_advisory(
        &mut self,
        result: SecurityObservation,
        recommendations: Vec<crate::doctor_dependencies::UpdateRecommendation>,
    ) {
        if let Some(commit) = result.resolved_commit {
            self.resolved_commit = Some(commit);
        }
        if result.cleared_dataset == Some(self.dataset()) {
            self.snapshot = None;
            self.scan = None;
            self.recommendations.clear();
            self.verification = None;
        }
        if let Some(snapshot) = result.snapshot {
            self.snapshot = Some(snapshot);
            self.scan = None;
            self.metadata = None;
            self.recommendations.clear();
            self.verification = None;
            self.finding_page = 0;
        }
        if let Some(scan) = result.scan {
            self.scan = Some(scan);
        }
        if let Some(metadata) = result.registry_metadata {
            self.metadata = Some(metadata);
            self.recommendations = recommendations;
        }
    }
    fn advisory_freshness(
        &self,
    ) -> (
        Option<crate::doctor_dependencies::AdvisoryFreshness>,
        Option<crate::doctor_dependencies::AdvisoryFreshness>,
    ) {
        let Some((snapshot, now)) = self.snapshot.as_ref().zip(DoctorView::now().ok()) else {
            return (None, None);
        };
        (
            Some(snapshot.freshness(now)),
            snapshot.upstream_freshness(now),
        )
    }
    fn finding_visible(&self, finding: &crate::doctor_dependencies::AdvisoryFinding) -> bool {
        use crate::doctor_dependencies::DependencyAdvisoryState;
        if self.severity.as_deref().is_some_and(|selected| {
            selected
                != finding
                    .evidence
                    .severity
                    .as_ref()
                    .map(DiagnosticText::as_str)
                    .unwrap_or("not provided")
        }) {
            return false;
        }
        match self.finding_filter {
            1 => finding.evidence.state == DependencyAdvisoryState::KnownAffectedDependency,
            2 => finding.evidence.state == DependencyAdvisoryState::Withdrawn,
            3 => !matches!(
                finding.evidence.state,
                DependencyAdvisoryState::KnownAffectedDependency
                    | DependencyAdvisoryState::Withdrawn
            ),
            4 => {
                let (download, publication) = self.advisory_freshness();
                download != Some(crate::doctor_dependencies::AdvisoryFreshness::Fresh)
                    || publication != Some(crate::doctor_dependencies::AdvisoryFreshness::Fresh)
            }
            5 => self.scan.as_ref().is_none_or(|scan| {
                scan.coverage != crate::doctor_dependencies::DependencyCoverage::Complete
            }),
            _ => true,
        }
    }
    fn update_visible(&self, choice: &crate::doctor_dependencies::UpdateRecommendation) -> bool {
        use crate::doctor_dependencies::DependencyUpdateKind;
        match self.update_filter {
            1 => choice.evidence.kind == DependencyUpdateKind::AdvisoryFix,
            2 => choice.constraint_compatibility == DependencyUpdateKind::ConstraintCompatible,
            3 => choice.constraint_compatibility == DependencyUpdateKind::PotentiallyBreaking,
            4 => choice.constraint_compatibility == DependencyUpdateKind::UnknownCompatibility,
            _ => true,
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
#[derive(Clone)]
enum SecurityAction {
    Refresh(crate::doctor_dependencies::AdvisoryRefreshRequest),
    Scan,
    Query { confirmed_catalog: bool },
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DoctorView {
    fn inspect_dependencies(&mut self, cx: &mut Context<Self>) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            DependencySecurityForm::project(
                &self.project,
                directory,
                &self.security.format(cx),
                self.generation,
                Self::now()?,
                cx,
            )
        })();
        match result {
            Ok(projection) => {
                self.security.set_projection(projection);
                self.error = None;
            }
            Err(error) => {
                self.security.clear_current();
                self.error = Some(DiagnosticText::new(error.to_string()));
            }
        }
        cx.notify();
    }

    fn review_dependency_collection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| {
            let (directory, selection, effects, duration) = self.investigation_selection(cx)?;
            anyhow::ensure!(
                selection.scope == crate::doctor::DoctorInvestigationScope::Development
                    && selection.problem == crate::doctor::DoctorInvestigationProblem::Dependencies,
                "Select Project development / Dependencies and fill the relevant validation target and development effects before collecting"
            );
            let path = self.security.output.read(cx).text(cx).trim().to_owned();
            anyhow::ensure!(
                !path.is_empty() && path.len() <= 4096,
                "Select a new contained report path"
            );
            let artifact = task::TaskArtifact {
                path,
                kind: task::TaskArtifactKind::Data,
                max_bytes: crate::doctor_dependencies::MAX_DEPENDENCY_GRAPH_BYTES as u64,
            };
            let now = Self::now()?;
            let review = match self.security.format(cx) {
                #[cfg(feature = "rust-tools")]
                crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata => {
                    crate::doctor::prepare_cargo_dependency_collection(
                        &self.project,
                        directory,
                        artifact,
                        selection,
                        effects,
                        duration,
                        self.generation,
                        now,
                        cx,
                    )?
                }
                #[cfg(feature = "jvm-tools")]
                crate::doctor_dependencies::DoctorDependencyReportFormat::MavenSelectedTree => {
                    crate::doctor::prepare_jvm_dependency_collection(
                        &self.project,
                        directory,
                        artifact,
                        project::jvm_project::JvmReviewedDependencyFormat::MavenSelectedTree,
                        selection,
                        effects,
                        duration,
                        self.generation,
                        now,
                        cx,
                    )?
                }
                #[cfg(feature = "jvm-tools")]
                crate::doctor_dependencies::DoctorDependencyReportFormat::GradleResolution {
                    project_path,
                    configuration,
                } => crate::doctor::prepare_jvm_dependency_collection(
                    &self.project,
                    directory,
                    artifact,
                    project::jvm_project::JvmReviewedDependencyFormat::GradleResolution {
                        project_path,
                        configuration,
                    },
                    selection,
                    effects,
                    duration,
                    self.generation,
                    now,
                    cx,
                )?,
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Selected dependency ecosystem is disabled"),
            };
            Ok(OwnerRepair::Reproduction(review))
        })();
        self.accept_proposal(result, window, cx);
    }

    fn review_security(
        &mut self,
        action: SecurityAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.preparation.is_some() || self.repair_task.is_some() {
            return;
        }
        use crate::doctor_dependencies::*;
        let prepared = (|| {
            let directory = self.review_directory(cx)?;
            let now = Self::now()?;
            let operation = match action {
                SecurityAction::Refresh(request) => prepare_advisory_refresh(
                    &self.project,
                    directory,
                    self.generation,
                    request,
                    now,
                    cx,
                ),
                SecurityAction::Scan => prepare_project_advisory_scan(
                    &self.project,
                    self.security.projection.clone().ok_or_else(|| {
                        anyhow::anyhow!("Inspect a current owner inventory first")
                    })?,
                    self.security.dataset(),
                    self.generation,
                    now,
                    cx,
                ),
                SecurityAction::Query { confirmed_catalog } => {
                    let projection = self.security.projection.clone().ok_or_else(|| {
                        anyhow::anyhow!("Inspect current resolved dependencies first")
                    })?;
                    projection.revalidate(&self.project, cx)?;
                    let node = self.security.selected.ok_or_else(|| {
                        anyhow::anyhow!("Select an exact resolved dependency instance")
                    })?;
                    let query = if confirmed_catalog {
                        RegistryQuery::for_confirmed_public_catalog(
                            projection.graph(),
                            node,
                            self.security.public_coordinate.read(cx).text(cx).trim(),
                        )?
                    } else {
                        RegistryQuery::for_resolved(projection.graph(), node)?
                    };
                    prepare_project_registry_query(
                        &self.project,
                        projection,
                        query,
                        self.generation,
                        now,
                        cx,
                    )
                }
            };
            Ok::<_, anyhow::Error>(operation)
        })();
        match prepared {
            Err(error) => self.accept_proposal(Err(error), window, cx),
            Ok(operation) => {
                let generation = self.generation;
                let window_handle = self.window_handle;
                self.preparation = Some(cx.spawn(async move |this, cx| {
                    let result = operation.await;
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.preparation = None;
                        if this.generation == generation {
                            this.accept_proposal(result.map(OwnerRepair::Advisory), window, cx);
                        }
                    }) {
                        log::debug!(
                            "Doctor security review view unavailable in {window_handle:?}: {error}"
                        );
                    }
                }));
                cx.notify();
            }
        }
    }

    #[cfg(feature = "agentic")]
    fn review_security_fix(
        &mut self,
        candidate: Option<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        use crate::doctor_dependencies::*;
        let prepared = (|| {
            let projection = self
                .security
                .projection
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Inspect current resolved dependencies"))?;
            projection.revalidate(&self.project, cx)?;
            let node = self
                .security
                .selected
                .ok_or_else(|| anyhow::anyhow!("Select an exact instance and introducing paths"))?;
            let proposal = if let Some(candidate) = candidate {
                let choice = self
                    .security
                    .recommendations
                    .get(candidate)
                    .ok_or_else(|| anyhow::anyhow!("Selected candidate changed"))?;
                DoctorDependencyRepairProposal::update(projection, node, choice)?
            } else {
                let mut package = projection
                    .graph()
                    .nodes()
                    .get(node)
                    .ok_or_else(|| anyhow::anyhow!("Selected dependency changed"))?
                    .clone();
                package.name = self
                    .security
                    .replacement_name
                    .read(cx)
                    .text(cx)
                    .trim()
                    .to_owned();
                package.version = self
                    .security
                    .replacement_version
                    .read(cx)
                    .text(cx)
                    .trim()
                    .to_owned();
                package.source = DependencySource::public_registry(package.ecosystem);
                DoctorDependencyRepairProposal::replacement(projection, node, package)?
            };
            let validation = if let Some(validation) =
                self.current_comparison.as_ref().filter(|validation| {
                    validation.selection().scope
                        == crate::doctor::DoctorInvestigationScope::Development
                        && validation.selection().problem
                            == crate::doctor::DoctorInvestigationProblem::Dependencies
                }) {
                validation.clone()
            } else {
                let (_, selection, _, _) = self.investigation_selection(cx)?;
                anyhow::ensure!(
                    selection.scope == crate::doctor::DoctorInvestigationScope::Development
                        && selection.problem
                            == crate::doctor::DoctorInvestigationProblem::Dependencies,
                    "Select Project development / Dependencies and the relevant validation target"
                );
                let scope = self
                    .report
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Collect the selected scope"))?
                    .scope
                    .clone();
                crate::doctor::DoctorInvestigation::new(
                    selection,
                    scope,
                    proposal.binding().clone(),
                    Self::now()?,
                )?
                .comparison_snapshot_at(&self.project, cx)?
            };
            let advisory = self.security.snapshot.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "First review retained advisory evaluation to establish the security baseline"
                )
            })?;
            let baseline = DoctorDependencyVerificationBaseline::capture(
                proposal.clone(),
                &validation,
                advisory,
                Self::now()?,
                &self.project,
                cx,
            )?;
            Ok::<_, anyhow::Error>((
                proposal,
                baseline,
                validation.selection().validation_target.as_str().to_owned(),
            ))
        })();
        match prepared {
            Ok((proposal, baseline, target)) => {
                self.security.baseline = Some(baseline);
                self.security.verification = None;
                self.review_dependency_native_repair(proposal, &target, window, cx);
            }
            Err(error) => {
                self.error = Some(DiagnosticText::new(error.to_string()));
                cx.notify();
            }
        }
    }

    fn verify_security_fix(&mut self, cx: &mut Context<Self>) {
        if self.preparation.is_some() || self.repair_task.is_some() {
            return;
        }
        let prepared = (|| {
            let baseline = self
                .security
                .baseline
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No reviewed dependency repair baseline"))?;
            let current = self.security.projection.clone().ok_or_else(|| {
                anyhow::anyhow!("Inspect newly collected owner graph after the change")
            })?;
            let tests = self.verification_comparison.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "Select a fresh relevant after-change test result using the comparison controls"
                )
            })?;
            let evidence_id = self
                .comparison_form
                .after_functional
                .ok_or_else(|| anyhow::anyhow!("Select the relevant observed test evidence"))?;
            let description = self
                .comparison_form
                .change
                .read(cx)
                .text(cx)
                .trim()
                .to_owned();
            anyhow::ensure!(
                !description.is_empty() && description.len() <= 4096,
                "Describe the reviewed change without source or secrets"
            );
            #[cfg(feature = "agentic")]
            let session_id = self
                .change_session
                .clone()
                .or(self.native_session.clone())
                .map(DiagnosticText::new);
            #[cfg(not(feature = "agentic"))]
            let session_id = None;
            let change = crate::doctor::DoctorReviewedChange {
                before_input: baseline.selection().input_identity.clone(),
                after_input: tests.selection().input_identity.clone(),
                description: DiagnosticText::new(description),
                session_id,
            };
            let (cancel, cancelled) = async_channel::bounded(1);
            let operation = baseline.verify_retained_current(
                current,
                tests,
                evidence_id,
                change,
                Self::now()?,
                cancelled,
                &self.project,
                cx,
            );
            Ok::<_, anyhow::Error>((cancel, operation))
        })();
        match prepared {
            Err(error) => {
                self.error = Some(DiagnosticText::new(error.to_string()));
                cx.notify();
            }
            Ok((cancel, operation)) => {
                let generation = self.generation;
                self.repair_cancel = Some(cancel);
                self.repair_status = Some(DiagnosticText::new(
                    "Comparing selected current graph, retained advisories and relevant owner test observations; no command execution",
                ));
                self.repair_task = Some(cx.spawn(async move |this, cx| {
                    let result = operation.await;
                    if let Err(error) = this.update(cx, |this, cx| {
                        this.repair_task = None; this.repair_cancel = None;
                        if this.generation != generation { return; }
                        match result {
                            Ok(result) => {
                                if let Some(report) = this.report.as_mut() {
                                    if let Err(error) = report.push(result.report_check()) { this.error = Some(DiagnosticText::new(error.to_string())); }
                                }
                                this.security.verification = Some(result); this.saved_generation = None;
                                this.repair_status = Some(DiagnosticText::new("Observed selected verification. Save/export only its bounded summary; this is not proof the whole application is secure."));
                            }
                            Err(error) => this.error = Some(DiagnosticText::new(format!("Dependency verification incomplete: {error}"))),
                        }
                        cx.notify();
                    }) { log::debug!("Doctor verification view unavailable: {error}"); }
                }));
                cx.notify();
            }
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
impl DoctorView {
    fn security_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor_dependencies::*;
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        let mut controls = v_flex().gap_2().child(
            div()
                .debug_selector(|| "doctor-security-open".into())
                .child(
                    Button::new(
                        "doctor-security-open",
                        if self.security.open {
                            "Hide dependency security"
                        } else {
                            "Dependency inventory, advisories and updates…"
                        },
                    )
                    .disabled(busy)
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.security.open = !this.security.open;
                        cx.notify();
                    })),
                ),
        );
        if !self.security.open {
            return controls.into_any_element();
        }
        controls = controls.child(Label::new("Local owner evidence remains available without GitHub. No hidden resolution, downloads, updates or source upload. An empty or incomplete scan never proves security. Select an existing worktree root; nested Gradle module/configuration is explicit."));
        let kinds = vec![
            #[cfg(feature = "rust-tools")]
            (SecurityKind::Cargo, "Cargo workspace"),
            #[cfg(feature = "jvm-tools")]
            (SecurityKind::Maven, "Maven selected tree"),
            #[cfg(feature = "jvm-tools")]
            (SecurityKind::Gradle, "Gradle selected configuration"),
        ];
        controls = controls.child(h_flex().gap_2().flex_wrap().children(
            kinds.into_iter().enumerate().map(|(index, (kind, label))| {
                Button::new(("doctor-security-kind", index), label)
                    .toggle_state(self.security.kind == kind)
                    .disabled(busy)
                    .tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.cancel_repair(cx);
                        this.security.clear_all();
                        this.security.kind = kind;
                        cx.notify();
                    }))
            }),
        ));
        if !busy {
            #[cfg(feature = "jvm-tools")]
            if self.security.kind == SecurityKind::Gradle {
                controls = controls
                    .child(Label::new("Selected Gradle project and configuration"))
                    .child(
                        div()
                            .id("doctor-security-module")
                            .aria_label("Gradle project path")
                            .child(self.security.module.clone()),
                    )
                    .child(
                        div()
                            .id("doctor-security-configuration")
                            .aria_label("Gradle configuration")
                            .child(self.security.configuration.clone()),
                    );
            }
            controls = controls
                .child(Label::new(
                    "New report output path; existing files cannot be overwritten",
                ))
                .child(
                    div()
                        .id("doctor-security-output")
                        .aria_label("New contained dependency report path")
                        .child(self.security.output.clone()),
                );
        }
        controls = controls.child(
            h_flex()
                .gap_2()
                .flex_wrap()
                .child(
                    div()
                        .debug_selector(|| "doctor-security-inventory".into())
                        .child(
                            Button::new(
                                "doctor-security-inventory",
                                "Inspect existing owner inventory",
                            )
                            .disabled(busy || self.report.is_none())
                            .tab_index(0_isize)
                            .on_click(cx.listener(|this, _, _, cx| this.inspect_dependencies(cx))),
                        ),
                )
                .child(
                    div()
                        .debug_selector(|| "doctor-security-collect".into())
                        .child(
                            Button::new(
                                "doctor-security-collect",
                                "Review exact dependency collection Task…",
                            )
                            .disabled(busy || self.report.is_none())
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                |this, _, window, cx| this.review_dependency_collection(window, cx),
                            )),
                        ),
                ),
        );
        controls = controls.child(Label::new("Collection accepts at most eight MiB through the existing dependency owner. It uses the investigation's explicit Project development / Dependencies scope, relevant validation target, deadline and development effects. Existing Cargo/Maven/Gradle commands can execute project build scripts and write caches; review exact effects. Missing installed/cache prerequisites stay blocked."));
        #[cfg(feature = "rust-tools")]
        if self.security.kind == SecurityKind::Cargo {
            if !busy {
                controls = controls.child(Label::new("Exact RustSec revision (lookup and archive download are separate approvals)"))
                .child(div().id("doctor-security-commit")
                            .aria_label("RustSec archive commit").child(self.security.rustsec_commit.clone()));
            }
            controls = controls
                .child(
                    Button::new(
                        "doctor-security-revision",
                        "Review official RustSec revision lookup…",
                    )
                    .disabled(busy || self.report.is_none())
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.review_security(
                            SecurityAction::Refresh(AdvisoryRefreshRequest::ResolveRustSecRevision),
                            window,
                            cx,
                        )
                    })),
                )
                .child(
                    Button::new(
                        "doctor-security-rustsec-refresh",
                        "Review exact RustSec archive refresh…",
                    )
                    .disabled(busy || self.report.is_none())
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, window, cx| {
                        let commit = this
                            .security
                            .rustsec_commit
                            .read(cx)
                            .text(cx)
                            .trim()
                            .to_owned();
                        this.review_security(
                            SecurityAction::Refresh(AdvisoryRefreshRequest::RustSec { commit }),
                            window,
                            cx,
                        );
                    }))
                    .map(|button| {
                        div()
                            .debug_selector(|| "doctor-security-rustsec-refresh".into())
                            .child(button)
                    }),
                );
            if let Some(commit) = self.security.resolved_commit.clone() {
                controls = controls.child(Label::new(format!("Observed official revision: {commit}. Lookup has not downloaded the dataset.")))
                    .child(Button::new("doctor-security-resolved-archive", "Review archive at the observed revision…").disabled(busy).tab_index(0_isize)
                        .on_click(cx.listener(move |this, _, window, cx| this.review_security(SecurityAction::Refresh(AdvisoryRefreshRequest::RustSec { commit: commit.clone() }), window, cx))));
            }
        }
        #[cfg(feature = "jvm-tools")]
        if matches!(
            self.security.kind,
            SecurityKind::Maven | SecurityKind::Gradle
        ) {
            controls = controls.child(
                Button::new(
                    "doctor-security-osv-refresh",
                    "Review official OSV Maven archive refresh…",
                )
                .disabled(busy || self.report.is_none())
                .tab_index(0_isize)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.review_security(
                        SecurityAction::Refresh(AdvisoryRefreshRequest::OsvMaven),
                        window,
                        cx,
                    )
                }))
                .map(|button| {
                    div()
                        .debug_selector(|| "doctor-security-osv-refresh".into())
                        .child(button)
                }),
            );
        }
        controls = controls.child(
            h_flex()
                .gap_2()
                .flex_wrap()
                .child(
                    div()
                        .debug_selector(|| "doctor-security-scan".into())
                        .child(
                            Button::new(
                                "doctor-security-scan",
                                "Review local retained advisory evaluation…",
                            )
                            .disabled(busy || self.security.projection.is_none())
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                |this, _, window, cx| {
                                    this.review_security(SecurityAction::Scan, window, cx)
                                },
                            )),
                        ),
                )
                .child(
                    Button::new(
                        "doctor-security-clear",
                        "Review clearing this shared host dataset…",
                    )
                    .disabled(busy || self.report.is_none())
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.review_security(
                            SecurityAction::Refresh(AdvisoryRefreshRequest::Clear {
                                dataset: this.security.dataset(),
                            }),
                            window,
                            cx,
                        )
                    })),
                ),
        );
        controls = controls.child(Label::new("Manual refresh only; data older than 24 hours is stale. Shared host cache expires after 30 days and is capped at 256 MiB. Current graph, retained revision, trust, permissions and host must remain valid. Remote advisory cache operations are unsupported; no local substitution."));
        let Some(projection) = self.security.projection.as_ref() else {
            return controls.into_any_element();
        };
        let current = projection.revalidate(&self.project, cx).is_ok();
        let graph = projection.graph();
        controls = controls
            .child(Label::new(format!(
                "Inventory: {} instances; {:?} coverage; observed at {}. {}",
                graph.nodes().len(),
                graph.provenance().coverage,
                graph.provenance().observed_at,
                if current {
                    "Current owner evidence"
                } else {
                    "Stale authority: inspect or review new collection before any action"
                }
            )))
            .children(
                graph
                    .provenance()
                    .limitations
                    .iter()
                    .map(|detail| Label::new(detail.as_str().to_owned())),
            );
        let count = graph.nodes().len();
        let start = self.security.page.saturating_mul(20);
        for (index, node) in graph.nodes().iter().enumerate().skip(start).take(20) {
            controls = controls.child(
                Button::new(
                    ("doctor-security-instance", index),
                    format!(
                        "{} {} — {:?}; configuration {}; classifier {}",
                        DiagnosticText::new(&node.name).as_str(),
                        DiagnosticText::new(&node.version).as_str(),
                        node.source.kind(),
                        node.configuration.hex(),
                        node.classifier.as_deref().unwrap_or("none")
                    ),
                )
                .toggle_state(self.security.selected == Some(index))
                .disabled(busy || !current)
                .tab_index(0_isize)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.cancel_repair(cx);
                    this.security.selected = Some(index);
                    this.security.metadata = None;
                    this.security.recommendations.clear();
                    cx.notify();
                }))
                .map(|button| {
                    div()
                        .debug_selector(move || format!("doctor-security-instance-{index}"))
                        .child(button)
                }),
            );
        }
        controls = controls.child(
            h_flex()
                .gap_2()
                .child(
                    Button::new("doctor-security-previous", "Previous instances")
                        .disabled(busy || self.security.page == 0)
                        .tab_index(0_isize)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.security.page = this.security.page.saturating_sub(1);
                            cx.notify();
                        })),
                )
                .child(Label::new(format!(
                    "Instances {}–{} of {count}",
                    start.saturating_add(1).min(count),
                    start.saturating_add(20).min(count)
                )))
                .child(
                    Button::new("doctor-security-next", "Next instances")
                        .disabled(busy || start.saturating_add(20) >= count)
                        .tab_index(0_isize)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.security.page = this.security.page.saturating_add(1);
                            cx.notify();
                        })),
                ),
        );
        if let Some(node) = self
            .security
            .selected
            .and_then(|index| graph.nodes().get(index))
        {
            let index = self.security.selected.unwrap_or_default();
            controls = controls.child(Label::new(format!(
                "Selected {} {} — source {:?}; configuration {}. {}",
                DiagnosticText::new(&node.name).as_str(),
                DiagnosticText::new(&node.version).as_str(),
                node.source.kind(),
                node.configuration.hex(),
                if graph.roots().contains(&index) {
                    "Workspace/root instance"
                } else {
                    "Resolved direct or transitive instance"
                }
            )));
            match graph.introducing_paths(index) {
                Ok(paths) => {
                    for (path_index, path) in paths.paths.iter().enumerate().take(32) {
                        let labels = path
                            .nodes
                            .iter()
                            .filter_map(|index| graph.nodes().get(*index))
                            .map(|node| {
                                format!(
                                    "{} {}",
                                    DiagnosticText::new(&node.name).as_str(),
                                    DiagnosticText::new(&node.version).as_str()
                                )
                            })
                            .collect::<Vec<_>>();
                        controls = controls.child(Label::new(format!(
                            "Introducing path {}: {}",
                            path_index + 1,
                            labels.join(" → ")
                        )));
                        for edge in path
                            .edges
                            .iter()
                            .filter_map(|index| graph.edges().get(*index))
                        {
                            controls = controls.child(Label::new(format!("Declared alias {}; constraint {}; scope {}; target {}; workspace inherited {}", DiagnosticText::new(&edge.alias).as_str(), DiagnosticText::new(edge.requirement.as_deref().unwrap_or("unknown")).as_str(), DiagnosticText::new(&edge.scope).as_str(), DiagnosticText::new(edge.target.as_deref().unwrap_or("all / unknown")).as_str(), edge.workspace_inherited)));
                        }
                    }
                    if paths.truncated || paths.cycle_observed {
                        controls = controls.child(Label::new("Introducing paths are incomplete or cyclic; omitted paths remain unknown."));
                    }
                }
                Err(error) => {
                    controls = controls.child(
                        Label::new(DiagnosticText::new(error.to_string()).as_str().to_owned())
                            .color(Color::Warning),
                    )
                }
            }
            controls = controls.child(
                Button::new(
                    "doctor-security-query",
                    "Review public source version request…",
                )
                .disabled(busy || !current || !node.source.is_public_registry())
                .tab_index(0_isize)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.review_security(
                        SecurityAction::Query {
                            confirmed_catalog: false,
                        },
                        window,
                        cx,
                    )
                }))
                .map(|button| {
                    div()
                        .debug_selector(|| "doctor-security-query".into())
                        .child(button)
                }),
            );
            if node.ecosystem == DependencyEcosystem::Maven
                && node.source.kind() == DependencySourceKind::Unknown
            {
                controls = controls.child(Label::new("Repository origin is unknown. Separately confirm this exact coordinate is public before a Maven Central catalog request. Candidates remain unverified for this artifact; origin never changes."));
                if !busy {
                    controls = controls.child(
                        div()
                            .id("doctor-security-public-coordinate")
                            .aria_label("Exact public Maven catalog coordinate confirmation")
                            .child(self.security.public_coordinate.clone()),
                    );
                }
                controls = controls.child(
                    Button::new(
                        "doctor-security-public-catalog",
                        "I confirm the exact coordinate is public: review catalog request…",
                    )
                    .disabled(busy || !current)
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.review_security(
                            SecurityAction::Query {
                                confirmed_catalog: true,
                            },
                            window,
                            cx,
                        )
                    }))
                    .map(|button| {
                        div()
                            .debug_selector(|| "doctor-security-public-catalog".into())
                            .child(button)
                    }),
                );
            }
        }
        if let Some(scan) = &self.security.scan {
            let (download_freshness, publication_freshness) = self.security.advisory_freshness();
            controls = controls.child(Label::new(format!("{} Graph {}; advisory {}; downloaded {}; observed {}; current download freshness {:?}; current publication freshness {:?}; observed match coverage {:?}.", scan.conclusion().as_str(), scan.graph_revision.hex(), scan.advisory_revision.hex(), scan.fetched_at, scan.observed_at, download_freshness, publication_freshness, scan.coverage)))
                .children(scan.limitations.iter().map(|detail| Label::new(detail.as_str().to_owned())))
                .child(h_flex().gap_2().flex_wrap().children(["All findings", "Known affected", "Withdrawn", "Uncertain / informational", "Stale / uncertain data", "Incomplete coverage"].into_iter().enumerate().map(|(index, label)| Button::new(("doctor-security-finding-filter", index), label).toggle_state(self.security.finding_filter == index).tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, _, cx| { this.security.finding_filter = index; this.security.finding_page = 0; cx.notify(); })))));
            for (index, finding) in scan
                .findings
                .iter()
                .enumerate()
                .filter(|(_, finding)| self.security.finding_visible(finding))
                .skip(self.security.finding_page.saturating_mul(32))
                .take(32)
            {
                controls = controls.child(
                    v_flex()
                        .gap_1()
                        .p_2()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .child(Label::new(format!(
                            "{} — {:?}; severity {}; source {:?}; {}",
                            finding.evidence.identifier.as_str(),
                            finding.evidence.state,
                            finding
                                .evidence
                                .severity
                                .as_ref()
                                .map(DiagnosticText::as_str)
                                .unwrap_or("not provided"),
                            finding.evidence.authority,
                            finding.source_url
                        )))
                        .child(Label::new(format!(
                            "Affected: {}; fixed: {}; modified {:?}; withdrawn {:?}",
                            finding.evidence.affected_versions.as_str(),
                            finding
                                .evidence
                                .fixed_versions
                                .iter()
                                .map(DiagnosticText::as_str)
                                .collect::<Vec<_>>()
                                .join(", "),
                            finding.modified_at,
                            finding.withdrawn_at
                        )))
                        .child(Label::new(finding.evidence.application_exploitability()))
                        .children(
                            finding
                                .conditions
                                .iter()
                                .map(|detail| Label::new(detail.as_str().to_owned())),
                        )
                        .child(
                            Button::new(
                                ("doctor-security-finding-instance", index),
                                "Inspect affected instance and introducing paths",
                            )
                            .disabled(busy || !current)
                            .tab_index(0_isize)
                            .on_click(cx.listener({
                                let node = finding.node;
                                move |this, _, _, cx| {
                                    this.cancel_repair(cx);
                                    this.security.selected = Some(node);
                                    this.security.page = node / 20;
                                    this.security.metadata = None;
                                    this.security.recommendations.clear();
                                    cx.notify();
                                }
                            })),
                        ),
                );
            }
            let filtered = scan
                .findings
                .iter()
                .filter(|finding| self.security.finding_visible(finding))
                .count();
            let finding_start = self.security.finding_page.saturating_mul(32);
            let mut severities = scan
                .findings
                .iter()
                .map(|finding| {
                    finding
                        .evidence
                        .severity
                        .as_ref()
                        .map(DiagnosticText::as_str)
                        .unwrap_or("not provided")
                        .to_owned()
                })
                .collect::<Vec<_>>();
            severities.sort();
            severities.dedup();
            controls = controls
                .child(
                    h_flex()
                        .gap_2()
                        .flex_wrap()
                        .child(
                            Button::new("doctor-security-severity-all", "All severities")
                                .toggle_state(self.security.severity.is_none())
                                .tab_index(0_isize)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.security.severity = None;
                                    this.security.finding_page = 0;
                                    cx.notify();
                                })),
                        )
                        .children(severities.into_iter().enumerate().map(|(index, severity)| {
                            Button::new(("doctor-security-severity", index), severity.clone())
                                .toggle_state(self.security.severity.as_ref() == Some(&severity))
                                .tab_index(0_isize)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.security.severity = Some(severity.clone());
                                    this.security.finding_page = 0;
                                    cx.notify();
                                }))
                        })),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            Button::new("doctor-security-finding-previous", "Previous findings")
                                .disabled(self.security.finding_page == 0)
                                .tab_index(0_isize)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.security.finding_page =
                                        this.security.finding_page.saturating_sub(1);
                                    cx.notify();
                                })),
                        )
                        .child(Label::new(format!(
                            "Findings {}–{} of {filtered}",
                            finding_start.saturating_add(1).min(filtered),
                            finding_start.saturating_add(32).min(filtered)
                        )))
                        .child(
                            Button::new("doctor-security-finding-next", "Next findings")
                                .disabled(finding_start.saturating_add(32) >= filtered)
                                .tab_index(0_isize)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.security.finding_page =
                                        this.security.finding_page.saturating_add(1);
                                    cx.notify();
                                })),
                        ),
                );
            controls = controls.child(Label::new("Known dependency findings are not proof of application exploitability. Missing paths, unknown versions, stale publication and incomplete graph/data coverage remain explicit."));
        }
        if let Some(metadata) = &self.security.metadata {
            controls = controls.child(Label::new(format!("Catalog response {}: fetched {}; freshness {:?}; coverage {:?}; catalog-only {}.", metadata.body_identity.hex(), metadata.fetched_at, Self::now().ok().and_then(|now| metadata.freshness(now)), metadata.coverage, metadata.query.is_catalog_only())))
                .children(metadata.limitations.iter().map(|detail| Label::new(detail.as_str().to_owned())))
                .child(h_flex().gap_2().flex_wrap().children(["All updates", "Advisory fixes", "Constraint compatible", "Potentially breaking", "Unknown compatibility"].into_iter().enumerate().map(|(index, label)| Button::new(("doctor-security-update-filter", index), label).toggle_state(self.security.update_filter == index).tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, _, cx| { this.security.update_filter = index; cx.notify(); })))));
            for (_index, choice) in self
                .security
                .recommendations
                .iter()
                .enumerate()
                .filter(|(_, choice)| self.security.update_visible(choice))
            {
                controls = controls.child(Label::new(format!("Candidate {} {} — {:?}; constraint {:?}; catalog-only {}; required Rust {}", choice.evidence.candidate.name, choice.evidence.candidate.version, choice.evidence.kind, choice.constraint_compatibility, choice.catalog_only, choice.required_rust.as_ref().map(DiagnosticText::as_str).unwrap_or("unknown"))))
                    .children(choice.limitations.iter().map(|detail| Label::new(detail.as_str().to_owned())));
                #[cfg(feature = "agentic")]
                {
                    controls = controls.child(
                        Button::new(
                            ("doctor-security-fix", _index),
                            "Select this alternative and review exact native repair proposal…",
                        )
                        .disabled(busy || !current)
                        .tab_index(0_isize)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.review_security_fix(Some(_index), window, cx)
                        }))
                        .map(|button| {
                            div()
                                .debug_selector(move || format!("doctor-security-fix-{_index}"))
                                .child(button)
                        }),
                    );
                }
            }
        }
        #[cfg(feature = "agentic")]
        {
            controls = controls.child(Label::new("Replacement requests explicitly select crates.io or Maven Central in the same ecosystem. They are unverified alternatives; availability, API/runtime compatibility and resolution require review and relevant verification. Repository configuration changes require separate review."));
            if !busy {
                controls = controls
                    .child(
                        div()
                            .id("doctor-security-replacement-name")
                            .aria_label("Explicit replacement package")
                            .child(self.security.replacement_name.clone()),
                    )
                    .child(
                        div()
                            .id("doctor-security-replacement-version")
                            .aria_label("Exact replacement version")
                            .child(self.security.replacement_version.clone()),
                    );
            }
            controls = controls.child(
                Button::new(
                    "doctor-security-replacement",
                    "Select replacement and review native repair proposal…",
                )
                .disabled(busy || !current || self.security.selected.is_none())
                .tab_index(0_isize)
                .on_click(
                    cx.listener(|this, _, window, cx| this.review_security_fix(None, window, cx)),
                ),
            );
        }
        controls = controls.child(Label::new("Before repair, select Project development / Dependencies and the relevant validation target in investigation controls. Current graph/advisory observations can establish the security baseline without a fabricated failed application test. Review the exact proposal and ordinary native changes/session. Afterwards collect a fresh resolved graph, explicitly rerun the same relevant tests, select their observed functional evidence and describe the reviewed change in comparison controls. Rust command exit alone remains incomplete; no unrelated test or model assertion proves a fix."))
            .child(div().debug_selector(|| "doctor-security-verify".into()).child(Button::new("doctor-security-verify", "Compare current advisories and relevant after-change tests")
                .disabled(busy || self.security.baseline.is_none() || !current).tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| this.verify_security_fix(cx)))));
        if let Some(result) = &self.security.verification {
            controls = controls.child(Label::new(format!(
                "Selected verification {:?}: {}",
                result.outcome,
                result.detail.as_str()
            )));
        }
        controls.child(Label::new("Use existing Save report, history, selected export and session links for bounded redacted summaries. Full inventory, raw reports, advisory bodies and credentials are transient and never exported to Doctor history or model context." )).into_any_element()
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum InstallKind {
    #[cfg(feature = "rust-tools")]
    RustToolchain,
    #[cfg(feature = "rust-tools")]
    RustComponent,
    #[cfg(feature = "rust-tools")]
    RustTarget,
    #[cfg(feature = "jvm-tools")]
    Mise,
    #[cfg(feature = "jvm-tools")]
    Sdkman,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
struct InstallForm {
    kind: InstallKind,
    executable: Entity<editor::Editor>,
    version: Entity<editor::Editor>,
    package: Entity<editor::Editor>,
    open: bool,
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
struct DependencyInputs {
    kind: crate::doctor::DependencyProbe,
    executable: Entity<editor::Editor>,
    open: bool,
}

#[cfg(feature = "agentic")]
struct ProbeInputs {
    endpoint: Entity<editor::Editor>,
    image: Entity<editor::Editor>,
    runtime: Entity<editor::Editor>,
    open: bool,
}

#[derive(Eq, PartialEq)]
enum RestartCoordination {
    Application,
    Remote,
}

enum ExportSource {
    Current { generation: u64, stale: bool },
    History(crate::doctor::DoctorHistoryRecord),
}

enum HistoryChange {
    Save(DoctorReport),
    Delete(crate::doctor::DoctorHistoryRecord),
    Clear { fingerprint: String },
}

#[derive(Clone)]
enum InvestigationAction {
    Command,
    Task(usize),
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    Test {
        provider: project::structured_execution::StructuredProviderId,
        node: project::structured_execution::StructuredNodeId,
        discovery: project::structured_execution::DiscoveryGeneration,
    },
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    Debug(usize),
    #[cfg(feature = "rust-tools")]
    Profile,
}

struct InvestigationForm {
    open: bool,
    scope: crate::doctor::DoctorInvestigationScope,
    problem: crate::doctor::DoctorInvestigationProblem,
    symptom: Entity<editor::Editor>,
    target: Entity<editor::Editor>,
    workload: Entity<editor::Editor>,
    configuration: Entity<editor::Editor>,
    cache: Entity<editor::Editor>,
    command: Entity<editor::Editor>,
    writes: Entity<editor::Editor>,
    subprocesses: Entity<editor::Editor>,
    duration: Entity<editor::Editor>,
    development_only: bool,
    tasks: Vec<(project::TaskSourceKind, task::TaskTemplate)>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    debug_scenarios: Vec<(project::TaskSourceKind, task::DebugScenario)>,
    #[cfg(feature = "rust-tools")]
    profile: Entity<editor::Editor>,
    #[cfg(feature = "rust-tools")]
    symbols: Entity<editor::Editor>,
}

impl InvestigationForm {
    fn new(window: &mut Window, cx: &mut Context<DoctorView>) -> Self {
        let mut input = |placeholder: &str, initial: &str, cx: &mut Context<DoctorView>| {
            cx.new(|cx| {
                let mut editor = editor::Editor::single_line(window, cx);
                editor.set_placeholder_text(placeholder, window, cx);
                editor.set_text(initial, window, cx);
                editor
            })
        };
        Self {
            open: false,
            scope: crate::doctor::DoctorInvestigationScope::Product,
            problem: crate::doctor::DoctorInvestigationProblem::ProductReadiness,
            symptom: input("Observed problem; exclude secrets and payloads", "", cx),
            target: input("Exact relevant build, test or application target", "", cx),
            workload: input("Workload and expected functional behavior", "", cx),
            configuration: input("Build/runtime/toolchain configuration", "", cx),
            cache: input("Cache and warm-up conditions for this operation", "", cx),
            command: input(
                "Exact development reproduction command (existing Task shell)",
                "",
                cx,
            ),
            writes: input(
                "Expected development file writes (or explicitly none)",
                "",
                cx,
            ),
            subprocesses: input("Expected subprocesses and their effects", "", cx),
            duration: input("Deadline in seconds (1–300)", "60", cx),
            development_only: false,
            tasks: Vec::new(),
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            debug_scenarios: Vec::new(),
            #[cfg(feature = "rust-tools")]
            profile: input("Contained project .json.gz Samply profile path", "", cx),
            #[cfg(feature = "rust-tools")]
            symbols: input("Contained project .syms.json sidecar path", "", cx),
        }
    }

    fn fields(&self) -> Vec<Entity<editor::Editor>> {
        let fields = vec![
            self.symptom.clone(),
            self.target.clone(),
            self.workload.clone(),
            self.configuration.clone(),
            self.cache.clone(),
            self.command.clone(),
            self.writes.clone(),
            self.subprocesses.clone(),
            self.duration.clone(),
        ];
        #[cfg(feature = "rust-tools")]
        let fields = fields
            .into_iter()
            .chain([self.profile.clone(), self.symbols.clone()])
            .collect();
        fields
    }

    fn problems(&self) -> Vec<(crate::doctor::DoctorInvestigationProblem, &'static str)> {
        use crate::doctor::{
            DoctorInvestigationProblem as Problem, DoctorInvestigationScope as Scope,
        };
        match self.scope {
            Scope::Product => vec![(Problem::ProductReadiness, "Product health")],
            Scope::Development => vec![
                (Problem::Build, "Build failure"),
                (Problem::Test, "Failing tests"),
                (Problem::Dependencies, "Dependencies"),
                (Problem::Configuration, "Configuration / toolchain"),
                (Problem::Pipeline, "Slow build pipeline"),
            ],
            Scope::Application => vec![
                (Problem::Test, "Failing tests"),
                (Problem::Crash, "Crash / exception"),
                (Problem::IncorrectBehavior, "Incorrect behavior"),
                (Problem::Cpu, "CPU"),
                (Problem::Memory, "Memory"),
                (Problem::SlowOperation, "Slow operation"),
                (Problem::Concurrency, "Concurrency"),
            ],
        }
    }
}

struct ComparisonForm {
    change: Entity<editor::Editor>,
    repetitions: Entity<editor::Editor>,
    maximum: Entity<editor::Editor>,
    improvement: Entity<editor::Editor>,
    performance: bool,
    before_functional: Option<uuid::Uuid>,
    after_functional: Option<uuid::Uuid>,
    before_measurements: Vec<uuid::Uuid>,
    after_measurements: Vec<uuid::Uuid>,
}
impl ComparisonForm {
    fn new(window: &mut Window, cx: &mut Context<DoctorView>) -> Self {
        let mut input = |placeholder: &str, value: &str, cx: &mut Context<DoctorView>| {
            cx.new(|cx| {
                let mut editor = editor::Editor::single_line(window, cx);
                editor.set_placeholder_text(placeholder, window, cx);
                editor.set_text(value, window, cx);
                editor
            })
        };
        Self {
            change: input(
                "Reviewed change description; exclude source and secrets",
                "",
                cx,
            ),
            repetitions: input("Minimum repetitions (2–32)", "2", cx),
            maximum: input("Maximum acceptable after-run median (ms)", "", cx),
            improvement: input("Minimum improvement (%)", "", cx),
            performance: false,
            before_functional: None,
            after_functional: None,
            before_measurements: Vec::new(),
            after_measurements: Vec::new(),
        }
    }
    fn clear_selections(&mut self) {
        self.before_functional = None;
        self.after_functional = None;
        self.before_measurements.clear();
        self.after_measurements.clear();
    }
}

#[cfg(feature = "agentic")]
struct GithubEvidenceForm {
    open: bool,
    account: Entity<editor::Editor>,
    owner: Entity<editor::Editor>,
    repository: Entity<editor::Editor>,
    target: Entity<editor::Editor>,
    credential_selected: bool,
    dependabot: bool,
    code_scanning: bool,
    page: usize,
    snapshot: Option<crate::doctor_dependencies::GithubEvidenceSnapshot>,
    authority: Option<crate::doctor_dependencies::DoctorGithubAction>,
    #[cfg(any(test, feature = "test-support"))]
    credential_reader: Option<std::sync::Arc<dyn Fn() -> Option<String> + Send + Sync>>,
}

#[cfg(feature = "agentic")]
impl GithubEvidenceForm {
    fn new(window: &mut Window, cx: &mut Context<DoctorView>) -> Self {
        let mut input = |placeholder: &str, cx: &mut Context<DoctorView>| {
            cx.new(|cx| {
                let mut input = editor::Editor::single_line(window, cx);
                input.set_placeholder_text(placeholder, window, cx);
                input
            })
        };
        Self {
            open: false,
            account: input("Expected existing GitHub account", cx),
            owner: input("Exact repository owner", cx),
            repository: input("Exact repository name", cx),
            target: input(
                "Relevant build/test validation target for a source proposal",
                cx,
            ),
            credential_selected: false,
            dependabot: false,
            code_scanning: false,
            page: 0,
            snapshot: None,
            authority: None,
            #[cfg(any(test, feature = "test-support"))]
            credential_reader: None,
        }
    }
    fn fields(&self) -> [Entity<editor::Editor>; 4] {
        [
            self.account.clone(),
            self.owner.clone(),
            self.repository.clone(),
            self.target.clone(),
        ]
    }
    fn clear_observations(&mut self) {
        self.snapshot = None;
        self.authority = None;
        self.page = 0;
    }
    fn selection(
        &self,
        cx: &App,
    ) -> anyhow::Result<http_client::github_security::GithubSecuritySelection> {
        anyhow::ensure!(
            self.credential_selected,
            "Explicitly select the existing GITHUB_TOKEN credential for this account/repository"
        );
        http_client::github_security::GithubSecuritySelection::new(
            &self.account.read(cx).text(cx),
            http_client::github_security::GithubSecurityRepository::new(
                &self.owner.read(cx).text(cx),
                &self.repository.read(cx).text(cx),
            )?,
        )
        .map_err(Into::into)
    }
    fn sources(&self) -> crate::doctor_dependencies::DoctorGithubSources {
        crate::doctor_dependencies::DoctorGithubSources {
            dependabot: self.dependabot,
            code_scanning: self.code_scanning,
        }
    }
    fn matches(
        &self,
        authority: &crate::doctor_dependencies::DoctorGithubAction,
        cx: &App,
    ) -> bool {
        self.selection(cx)
            .is_ok_and(|selection| selection == *authority.selection())
            && self.sources() == authority.sources()
    }
    fn page_count(&self) -> usize {
        self.snapshot
            .as_ref()
            .map(|snapshot| {
                [
                    snapshot.dependabot.observations.len(),
                    snapshot.pull_requests.observations.len(),
                    snapshot.code_scanning.observations.len(),
                    snapshot.analyses.observations.len(),
                ]
                .into_iter()
                .max()
                .unwrap_or(0)
                .div_ceil(10)
                .max(1)
            })
            .unwrap_or(1)
    }
}

#[cfg(feature = "agentic")]
impl DoctorView {
    fn review_github(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.repair_task.is_some() {
            return;
        }
        self.github.clear_observations();
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let selection = self.github.selection(cx)?;
            let sources = self.github.sources();
            let now = Self::now()?;
            #[cfg(any(test, feature = "test-support"))]
            if let Some(reader) = &self.github.credential_reader {
                return crate::doctor_dependencies::prepare_github_evidence_fixture(
                    &self.project,
                    directory,
                    self.generation,
                    selection,
                    sources,
                    now,
                    reader.clone(),
                    cx,
                )
                .map(OwnerRepair::Github);
            }
            crate::doctor_dependencies::prepare_github_evidence(
                &self.project,
                directory,
                self.generation,
                selection,
                sources,
                now,
                cx,
            )
            .map(OwnerRepair::Github)
        })();
        self.accept_proposal(result, window, cx);
    }
    fn review_github_source_repair(
        &mut self,
        selection: crate::doctor_agent::DoctorGithubFindingSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let report = self.report.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Collect current checks before requesting a source proposal")
            })?;
            let snapshot = self
                .github
                .snapshot
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Collect reviewed GitHub evidence first"))?;
            let authority = self
                .github
                .authority
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Selected GitHub authority is unavailable"))?;
            anyhow::ensure!(
                self.github.matches(authority, cx),
                "Selected GitHub scope changed"
            );
            crate::doctor_agent::prepare_github_native_repair(
                &self.project,
                directory,
                report,
                authority.clone(),
                snapshot,
                selection,
                &self.github.target.read(cx).text(cx),
                Self::now()?,
                cx,
            )
            .map(OwnerRepair::Native)
        })();
        self.accept_proposal(result, window, cx);
    }
    fn github_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut controls = v_flex().gap_2().child(
            div().debug_selector(|| "doctor-github-open".into()).child(
                Button::new(
                    "doctor-github-open",
                    if self.github.open {
                        "Hide optional GitHub evidence"
                    } else {
                        "Optional GitHub security evidence"
                    },
                )
                .tab_index(0_isize)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.github.open = !this.github.open;
                    cx.notify();
                })),
            ),
        );
        if !self.github.open {
            return controls.into_any_element();
        }
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        controls = controls.child(Label::new("Read-only GitHub.com metadata from an explicitly selected existing credential. Local/offline diagnosis remains available. No source/SARIF/secret retrieval, new permission grants, alert changes, service enablement or workflow execution."));
        for (identifier, label, field) in [
            (
                "doctor-github-account",
                "Expected account",
                &self.github.account,
            ),
            (
                "doctor-github-owner",
                "Repository owner",
                &self.github.owner,
            ),
            (
                "doctor-github-repository",
                "Repository name",
                &self.github.repository,
            ),
            (
                "doctor-github-target",
                "Relevant source-repair validation target",
                &self.github.target,
            ),
        ] {
            controls = controls.child(
                v_flex().gap_1().child(Label::new(label)).child(
                    div()
                        .debug_selector(move || identifier.into())
                        .child(field.clone()),
                ),
            );
        }
        controls = controls.child(h_flex().flex_wrap().gap_2().children([
            (0_usize, "Select existing GITHUB_TOKEN", self.github.credential_selected),
            (1, "Dependabot alerts and PR candidates", self.github.dependabot),
            (2, "Existing code-scanning alerts and analyses", self.github.code_scanning),
        ].into_iter().map(|(index, label, selected)| div().debug_selector(move || format!("doctor-github-source-{index}")).child(Button::new(("doctor-github-source", index), label)
            .toggle_state(selected).disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| {
                this.cancel_repair(cx); this.github.clear_observations();
                match index { 0 => this.github.credential_selected = !this.github.credential_selected, 1 => this.github.dependabot = !this.github.dependabot, _ => this.github.code_scanning = !this.github.code_scanning }
                cx.notify();
            }))))))
            .child(Label::new("Minimum selected access: repository Metadata read; Dependabot alerts and Pull requests read when requested; Code scanning alerts read when requested. Missing/denied access stays blocked or incomplete. Account identity and exact project origin are verified after approval."))
            .child(div().debug_selector(|| "doctor-github-review".into()).child(Button::new("doctor-github-review", "Review exact GitHub read")
                .disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.review_github(window, cx)))));
        let Some(snapshot) = &self.github.snapshot else {
            return controls
                .child(Label::new(
                    "No reviewed GitHub evidence observed. This is not a zero-finding result.",
                ))
                .into_any_element();
        };
        let now = Self::now().ok();
        let current = self.github.authority.as_ref().is_some_and(|authority| {
            self.github.matches(authority, cx)
                && self.review_directory(cx).is_ok_and(|directory| {
                    authority.revalidate(&self.project, &directory, cx).is_ok()
                })
        });
        let fresh =
            now.is_some_and(|now| snapshot.fetched_at <= now && now - snapshot.fetched_at <= 86400);
        controls = controls.child(Label::new(format!("Verified account ID {}; repository {} (ID {}, private {}); fetched {}. Current local authority inputs: {}; retrieval freshness: {}. Selected checkout ref {:?}, commit {:?}, dirty {:?}", snapshot.authority.account_id, snapshot.repository.full_name(), snapshot.authority.repository_id, snapshot.authority.private_repository, Self::timestamp(snapshot.fetched_at), current, if fresh { "within 24 hours" } else { "stale or unknown" }, snapshot.checkout.reference, snapshot.checkout.commit, snapshot.checkout.dirty)))
            .child(Label::new("Complete pagination means this source was read to exhaustion. Empty findings, unknown/disabled coverage and no current successful analysis are distinct; none proves project security. Dependabot has no observed resolved version/analyzed commit; PRs are candidates, not verified fixes."));
        for summary in [
            format!(
                "Dependabot: {:?}; {} observations; {} pages / {} bytes",
                snapshot.dependabot.coverage,
                snapshot.dependabot.observations.len(),
                snapshot.dependabot.pages_completed,
                snapshot.dependabot.bytes_collected
            ),
            format!(
                "Update PR candidates: {:?}; {} observations; {} pages / {} bytes",
                snapshot.pull_requests.coverage,
                snapshot.pull_requests.observations.len(),
                snapshot.pull_requests.pages_completed,
                snapshot.pull_requests.bytes_collected
            ),
            format!(
                "Code scanning: {:?}; {} observations; {} pages / {} bytes",
                snapshot.code_scanning.coverage,
                snapshot.code_scanning.observations.len(),
                snapshot.code_scanning.pages_completed,
                snapshot.code_scanning.bytes_collected
            ),
            format!(
                "Analysis provenance: {:?}; {} observations; {} pages / {} bytes",
                snapshot.analyses.coverage,
                snapshot.analyses.observations.len(),
                snapshot.analyses.pages_completed,
                snapshot.analyses.bytes_collected
            ),
        ] {
            controls = controls.child(Label::new(summary));
        }
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let local = self
            .security
            .projection
            .as_ref()
            .filter(|projection| projection.revalidate(&self.project, cx).is_ok())
            .and_then(|projection| {
                self.security
                    .scan
                    .as_ref()
                    .filter(|scan| scan.graph_revision == projection.graph().identity())
            })
            .map(|scan| {
                scan.findings
                    .iter()
                    .map(|finding| &finding.evidence)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        #[cfg(not(any(feature = "rust-tools", feature = "jvm-tools")))]
        let local = Vec::new();
        match crate::doctor_dependencies::correlate_github_evidence(
            snapshot,
            &local,
            now.unwrap_or(0),
        ) {
            Ok(correlation) => {
                controls = controls.child(Label::new(format!("Preserved occurrences: {} local / {} GitHub. {} candidate package/advisory links; {} code groups; {} verified dependency duplicate groups", correlation.local_occurrences, correlation.github_occurrences, correlation.dependency_links.len(), correlation.code_groups.len(), correlation.verified_dependency_groups)));
                for limitation in &correlation.limitations {
                    controls = controls.child(Label::new(limitation.as_str().to_owned()));
                }
                for link in correlation
                    .dependency_links
                    .iter()
                    .skip(self.github.page.saturating_mul(10))
                    .take(10)
                {
                    controls = controls.child(Label::new(format!("Candidate local observation {} ↔ GitHub alert {}: {:?}; {}. Inspect the local resolved dependency path and explicitly select a fix in Dependency Security", link.local_observation, link.github_alert, link.checkout_relevance, link.limitation.as_str())));
                }
                for group in correlation
                    .code_groups
                    .iter()
                    .skip(self.github.page.saturating_mul(10))
                    .take(10)
                {
                    controls = controls.child(Label::new(format!("Code group occurrences {:?}; checkout {:?}; matching analysis IDs {:?}. All source observations below remain preserved", group.alert_numbers, group.checkout_relevance, group.matching_analysis_ids)));
                }
            }
            Err(error) => {
                controls = controls.child(Label::new(format!(
                    "Correlation incomplete: {}",
                    DiagnosticText::new(error.to_string()).as_str()
                )))
            }
        }
        let offset = self.github.page.saturating_mul(10);
        for finding in snapshot
            .dependabot
            .observations
            .iter()
            .skip(offset)
            .take(10)
        {
            let number = finding.number;
            let source = finding.url.clone();
            controls = controls.child(v_flex().p_2().gap_1().border_1().border_color(cx.theme().colors().border)
                .child(Label::new(DiagnosticText::new(format!("Dependabot #{number}: {} / {}, advisory {}, aliases {:?}, {:?}, severity {:?}, affected {:?}, first fixed {:?}, withdrawn {:?}; manifest {:?}, relationship {:?}, scope {:?}. Published {:?}, updated {:?}: {}. Resolved version/commit and exploitability unknown", finding.ecosystem, finding.package, finding.advisory_id, finding.aliases, finding.state, finding.severity, finding.affected_range, finding.first_fixed_version, finding.withdrawn_at, finding.manifest_path, finding.relationship, finding.scope, finding.advisory_published_at, finding.updated_at, crate::doctor_dependencies::GithubEvidenceSnapshot::source_freshness(finding.updated_at.as_deref(), now.unwrap_or(0)).as_str())).as_str().to_owned()))
                .child(Button::new(("doctor-github-dependabot-link", number), "Open source alert").tab_index(0_isize).on_click(move |_, _, cx| cx.open_url(&source)))
                .child(div().debug_selector(move || format!("doctor-github-dependabot-repair-{number}")).child(Button::new(("doctor-github-dependabot-repair", number), "Review source mitigation proposal (no dependency upgrade)")
                    .disabled(busy || !current || !fresh).tab_index(0_isize).on_click(cx.listener(move |this, _, window, cx| this.review_github_source_repair(crate::doctor_agent::DoctorGithubFindingSelection::Dependabot(number), window, cx))))));
        }
        for finding in snapshot
            .code_scanning
            .observations
            .iter()
            .skip(offset)
            .take(10)
        {
            let number = finding.number;
            let source = finding.url.clone();
            controls = controls.child(v_flex().p_2().gap_1().border_1().border_color(cx.theme().colors().border)
                .child(Label::new(DiagnosticText::new(format!("Code alert #{number}: rule {:?}, scanner {} / {:?}, GUID {:?}, state {}, severity {:?}/{:?}; {:?}:{:?}-{:?}; ref {:?}, commit {:?}, updated {:?}: {}; checkout {:?}", finding.rule_id, finding.tool.name, finding.tool.version, finding.tool.guid, finding.state, finding.severity, finding.security_severity, finding.path, finding.start_line, finding.end_line, finding.reference, finding.commit, finding.updated_at, crate::doctor_dependencies::GithubEvidenceSnapshot::source_freshness(finding.updated_at.as_deref(), now.unwrap_or(0)).as_str(), snapshot.checkout.relevance(finding.reference.as_deref(), finding.commit.as_deref()))).as_str().to_owned()))
                .child(Button::new(("doctor-github-code-link", number), "Open source code-scanning alert").tab_index(0_isize).on_click(move |_, _, cx| cx.open_url(&source)))
                .child(div().debug_selector(move || format!("doctor-github-code-repair-{number}")).child(Button::new(("doctor-github-code-repair", number), "Review selected source repair proposal")
                    .disabled(busy || !current || !fresh).tab_index(0_isize).on_click(cx.listener(move |this, _, window, cx| this.review_github_source_repair(crate::doctor_agent::DoctorGithubFindingSelection::CodeScanning(number), window, cx))))));
        }
        for analysis in snapshot.analyses.observations.iter().skip(offset).take(10) {
            controls = controls.child(Label::new(DiagnosticText::new(format!("Analysis {}: {} / {:?}, ref {}, commit {}, scan time {:?}: {}, rules {:?}, results {:?}, category {:?}, error {}, warning {}, checkout {:?}. Missing/zero rules or observed errors cannot establish successful current coverage", analysis.id, analysis.tool.name, analysis.tool.version, analysis.reference, analysis.commit, analysis.created_at, crate::doctor_dependencies::GithubEvidenceSnapshot::source_freshness(analysis.created_at.as_deref(), now.unwrap_or(0)).as_str(), analysis.rules_count, analysis.results_count, analysis.category, analysis.observed_error, analysis.observed_warning, snapshot.checkout.relevance(Some(&analysis.reference), Some(&analysis.commit)))).as_str().to_owned()));
        }
        for pull_request in snapshot
            .pull_requests
            .observations
            .iter()
            .skip(offset)
            .take(10)
        {
            let source = pull_request.url.clone();
            let number = pull_request.number;
            controls = controls.child(Label::new(format!("Dependabot PR candidate #{}: {}; base {}@{}, head {}@{}, updated {:?}. Association to a specific alert/fix is unverified", pull_request.number, pull_request.state, pull_request.base_ref, pull_request.base_commit, pull_request.head_ref, pull_request.head_commit, pull_request.updated_at)))
                .child(div().debug_selector(move || format!("doctor-github-pr-{number}")).child(Button::new(("doctor-github-pr", pull_request.number), "Open existing update PR (no merge)").tab_index(0_isize).on_click(move |_, _, cx| cx.open_url(&source))));
        }
        controls
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("doctor-github-previous", "Previous evidence page")
                            .disabled(self.github.page == 0)
                            .tab_index(0_isize)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.github.page = this.github.page.saturating_sub(1);
                                cx.notify();
                            })),
                    )
                    .child(Label::new(format!(
                        "Page {} / {}; at most ten rows per source",
                        self.github.page + 1,
                        self.github.page_count()
                    )))
                    .child(
                        Button::new("doctor-github-next", "Next evidence page")
                            .disabled(
                                self.github.page.saturating_add(1) >= self.github.page_count(),
                            )
                            .tab_index(0_isize)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.github.page = this
                                    .github
                                    .page
                                    .saturating_add(1)
                                    .min(this.github.page_count().saturating_sub(1));
                                cx.notify();
                            })),
                    ),
            )
            .into_any_element()
    }
}

#[cfg(feature = "agentic")]
struct SecuritySetupForm {
    open: bool,
    kind: crate::doctor_agent::DoctorSecuritySetupKind,
    fields: Vec<(&'static str, &'static str, Entity<editor::Editor>)>,
}
#[cfg(feature = "agentic")]
impl SecuritySetupForm {
    fn new(window: &mut Window, cx: &mut Context<DoctorView>) -> Self {
        let fields = [
            ("doctor-setup-path", "Exact target configuration/workflow path", ".github/dependabot.yml"),
            ("doctor-setup-instructions", "Unpublished source-proposal instructions", "Inspect observed project manifests and existing policy; draft exact compatible configuration changes for review. Do not activate or publish"),
            ("doctor-setup-version", "Scanner/service version or explicit uncertainty", "GitHub-hosted Dependabot; schema version 2; service build unknown"),
            ("doctor-setup-triggers", "Proposed triggers/jobs for final review", "Weekly update proposals; existing CI effects must be inspected and separately approved"),
            ("doctor-setup-runner", "Runner/service and entitlement", "GitHub-hosted service; private eligibility and runner authority unknown"),
            ("doctor-setup-network", "CI network/download destinations and effects", "Project package registries; exact destinations and downloads must be established before publication"),
            ("doctor-setup-credentials", "Existing credential references and least permissions", "No new secrets or grants; existing least-privilege references must be reviewed before activation"),
            ("doctor-setup-costs", "Billing/cost effects", "Unknown; separately review account plan and runner/service costs"),
            ("doctor-setup-validation", "Relevant validation target", "Inspect exact configuration/workflow diff and rerun selected project build/tests under separate review"),
        ].into_iter().map(|(identifier, label, value)| {
            let input = cx.new(|cx| { let mut input = editor::Editor::single_line(window, cx); input.set_placeholder_text(label, window, cx); input.set_text(value, window, cx); input });
            (identifier, label, input)
        }).collect();
        Self {
            open: false,
            kind: crate::doctor_agent::DoctorSecuritySetupKind::Dependabot,
            fields,
        }
    }
    fn request(&self, cx: &App) -> anyhow::Result<crate::doctor_agent::DoctorSecuritySetupRequest> {
        let value = |identifier: &str| {
            self.fields
                .iter()
                .find(|(id, _, _)| *id == identifier)
                .map(|(_, _, input)| input.read(cx).text(cx))
                .ok_or_else(|| anyhow::anyhow!("Setup input is unavailable"))
        };
        let request = crate::doctor_agent::DoctorSecuritySetupRequest {
            kind: self.kind,
            configuration_path: value("doctor-setup-path")?,
            instructions: value("doctor-setup-instructions")?,
            scanner_version: value("doctor-setup-version")?,
            triggers: value("doctor-setup-triggers")?,
            runner: value("doctor-setup-runner")?,
            network: value("doctor-setup-network")?,
            credential_permissions: value("doctor-setup-credentials")?,
            costs: value("doctor-setup-costs")?,
            validation_target: value("doctor-setup-validation")?,
        };
        request.validate()?;
        Ok(request)
    }
}

#[cfg(feature = "agentic")]
impl DoctorView {
    fn review_security_setup(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let report = self
                .report
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Collect current local checks first"))?;
            let snapshot = self.github.snapshot.as_ref().ok_or_else(|| anyhow::anyhow!("Explicitly select and review existing GitHub account/repository evidence first; unavailable source permissions do not grant setup authority"))?;
            let authority = self.github.authority.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Current selected repository authority is unavailable")
            })?;
            anyhow::ensure!(
                self.github.matches(authority, cx),
                "Selected setup account/repository scope changed"
            );
            crate::doctor_agent::prepare_security_setup_native_repair(
                &self.project,
                directory,
                report,
                authority.clone(),
                snapshot,
                self.security_setup.request(cx)?,
                Self::now()?,
                cx,
            )
            .map(OwnerRepair::Native)
        })();
        self.accept_proposal(result, window, cx);
    }
    fn security_setup_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut controls = v_flex().gap_2().child(
            div()
                .debug_selector(|| "doctor-security-setup-open".into())
                .child(
                    Button::new(
                        "doctor-security-setup-open",
                        if self.security_setup.open {
                            "Hide unpublished security setup"
                        } else {
                            "Prepare separately reviewed security setup"
                        },
                    )
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.security_setup.open = !this.security_setup.open;
                        cx.notify();
                    })),
                ),
        );
        if !self.security_setup.open {
            return controls.into_any_element();
        }
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        controls = controls.child(Label::new("Source proposal only. NativeAgent drafts exact configuration/workflow changes in normal Made review. Repository metadata does not establish private eligibility, billing, enabled services, CI/network/credential effects or runner authority. Unknown effects block activation; no setup-complete claim is made."))
            .child(Label::new("Publication can trigger workflows. Service enablement, publication and CI execution require separate explicit approval for the final files, triggers/jobs, scanner/version, destinations/downloads, credentials/permissions, eligibility and costs. This view has no activation or publication action."))
            .child(h_flex().gap_2().children([
                (0_usize, crate::doctor_agent::DoctorSecuritySetupKind::Dependabot, "Dependabot configuration"),
                (1, crate::doctor_agent::DoctorSecuritySetupKind::ExistingScanningWorkflow, "Existing scanning workflow"),
            ].into_iter().map(|(index, kind, label)| div().debug_selector(move || format!("doctor-setup-kind-{index}")).child(Button::new(("doctor-setup-kind", index), label).toggle_state(self.security_setup.kind == kind).disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, window, cx| {
                this.cancel_repair(cx); this.security_setup.kind = kind;
                for (identifier, _, field) in &this.security_setup.fields {
                    let value = match (*identifier, kind) {
                        ("doctor-setup-path", crate::doctor_agent::DoctorSecuritySetupKind::Dependabot) => Some(".github/dependabot.yml"),
                        ("doctor-setup-path", crate::doctor_agent::DoctorSecuritySetupKind::ExistingScanningWorkflow) => Some(".github/workflows/security.yml"),
                        ("doctor-setup-version", crate::doctor_agent::DoctorSecuritySetupKind::Dependabot) => Some("GitHub-hosted Dependabot; schema version 2; service build unknown"),
                        ("doctor-setup-version", crate::doctor_agent::DoctorSecuritySetupKind::ExistingScanningWorkflow) => Some("Existing project scanner/version unknown; inspect current workflow, do not introduce a new scanner"),
                        _ => None,
                    };
                    if let Some(value) = value { field.update(cx, |input, cx| input.set_text(value, window, cx)); }
                }
                cx.notify();
            }))))));
        for (identifier, label, field) in &self.security_setup.fields {
            let identifier = *identifier;
            controls = controls.child(
                v_flex().gap_1().child(Label::new(*label)).child(
                    div()
                        .debug_selector(move || identifier.into())
                        .child(field.clone()),
                ),
            );
        }
        if let Some(snapshot) = &self.github.snapshot {
            controls = controls.child(Label::new(format!("Selected verified repository {} (ID {}, private {}); private/service eligibility and billing remain unknown. Account/repository/credential and source authority are rechecked before model submission", snapshot.repository.full_name(), snapshot.authority.repository_id, snapshot.authority.private_repository)));
            match self
                .security_setup
                .request(cx)
                .and_then(|request| request.exact_preview(snapshot.authority.private_repository))
            {
                Ok(preview) => {
                    for line in preview {
                        controls = controls.child(Label::new(line));
                    }
                }
                Err(error) => {
                    controls = controls.child(
                        Label::new(format!(
                            "Setup proposal is invalid: {}",
                            DiagnosticText::new(error.to_string()).as_str()
                        ))
                        .color(Color::Error),
                    )
                }
            }
            let settings_url = format!(
                "https://github.com/{}/settings/security_analysis",
                snapshot.repository.full_name()
            );
            controls = controls.child(
                div()
                    .debug_selector(|| "doctor-setup-settings".into())
                    .child(
                        Button::new(
                            "doctor-setup-settings",
                            "Open existing repository security settings (manual handoff)",
                        )
                        .tab_index(0_isize)
                        .on_click(move |_, _, cx| cx.open_url(&settings_url)),
                    ),
            );
        } else {
            controls = controls.child(Label::new("No verified selected GitHub repository evidence. Local/offline diagnosis and ordinary source-repair flows remain available."));
        }
        controls
            .child(
                div().debug_selector(|| "doctor-setup-review".into()).child(
                    Button::new(
                        "doctor-setup-review",
                        "Review unpublished NativeAgent setup draft",
                    )
                    .disabled(busy || self.report.is_none() || self.github.snapshot.is_none())
                    .tab_index(0_isize)
                    .on_click(
                        cx.listener(|this, _, window, cx| this.review_security_setup(window, cx)),
                    ),
                ),
            )
            .into_any_element()
    }
}

pub struct DoctorEvidenceUpdated;

pub struct DoctorView {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    window_handle: gpui::AnyWindowHandle,
    focus_handle: FocusHandle,
    scroll_handle: gpui::ScrollHandle,
    review_anchor: gpui::ScrollAnchor,
    selected_worktree: Option<WorktreeId>,
    selected_product: DoctorProduct,
    generation: u64,
    report: Option<DoctorReport>,
    report_stale: bool,
    collection_revision: Option<u64>,
    collection_cancelled: bool,
    collection_failed: bool,
    advanced_open: bool,
    error: Option<DiagnosticText>,
    collection: Option<Task<()>>,
    closed: bool,
    proposal: Option<OwnerRepair>,
    owner_result: Option<OwnerRepair>,
    reviewed_action: Option<DiagnosticText>,
    repair_status: Option<DiagnosticText>,
    repair_task: Option<Task<()>>,
    repair_cancel: Option<async_channel::Sender<()>>,
    preparation: Option<Task<()>>,
    restart_preparation: bool,
    restart_active: Option<RestartCoordination>,
    history_open: bool,
    history_records: Vec<crate::doctor::DoctorHistoryRecord>,
    history_selected: Option<uuid::Uuid>,
    history_change: Option<HistoryChange>,
    history_task: Option<Task<()>>,
    history_status: Option<DiagnosticText>,
    saved_generation: Option<u64>,
    export_task: Option<Task<()>>,
    export_status: Option<DiagnosticText>,
    export_writing: bool,
    #[cfg(feature = "agentic")]
    probe: ProbeInputs,
    #[cfg(feature = "agentic")]
    github: GithubEvidenceForm,
    #[cfg(feature = "agentic")]
    security_setup: SecuritySetupForm,
    #[cfg(feature = "agentic")]
    native_target: Entity<editor::Editor>,
    #[cfg(feature = "agentic")]
    native_evidence: Vec<String>,
    #[cfg(feature = "agentic")]
    native_investigation_evidence: Vec<uuid::Uuid>,
    #[cfg(feature = "agentic")]
    native_repair_open: bool,
    #[cfg(feature = "agentic")]
    native_session: Option<String>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    install: InstallForm,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    dependency: DependencyInputs,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    security: DependencySecurityForm,
    investigation: InvestigationForm,
    current_comparison: Option<crate::doctor::DoctorComparisonSnapshot>,
    baseline_comparison: Option<crate::doctor::DoctorComparisonSnapshot>,
    verification_comparison: Option<crate::doctor::DoctorComparisonSnapshot>,
    comparison_form: ComparisonForm,
    #[cfg(feature = "agentic")]
    change_session: Option<String>,
    #[cfg(feature = "rust-tools")]
    profile_result: Option<project::profile_results_store::DiagnosticProfileResult>,
    task_result: Option<task::StructuredTaskHandle>,
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    debug_result: Option<Entity<project::debugger::session::Session>>,
    _subscriptions: Vec<Subscription>,
}

impl DoctorView {
    pub fn register(workspace: &mut Workspace, _: Option<&mut Window>, _: &mut Context<Workspace>) {
        workspace.register_action(Self::deploy);
    }

    pub fn deploy(
        workspace: &mut Workspace,
        _: &Doctor,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.reveal_center_pane(window, cx);
        let view = if let Some(view) = workspace.item_of_type::<Self>(cx) {
            workspace.activate_item(&view, true, true, window, cx);
            view
        } else {
            let project = workspace.project().clone();
            let workspace_handle = cx.entity().downgrade();
            let view = cx.new(|cx| Self::new(project, workspace_handle, window, cx));
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
            view
        };
        // The pane may have been hidden behind a zoomed panel in this frame.
        window.on_next_frame(move |window, cx| view.focus_handle(cx).focus(window, cx));
    }

    fn new(
        project: Entity<Project>,
        workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let roots: Vec<_> = project
            .read(cx)
            .visible_worktrees(cx)
            .map(|root| root.read(cx).id())
            .collect();
        let selected_worktree = if roots.len() == 1 {
            roots.first().copied()
        } else {
            None
        };
        let subscriptions = vec![cx.subscribe(&project, |this, _, event, cx| match event {
            project::Event::Closed => {
                this.closed = true;
                this.invalidate(cx);
            }
            project::Event::WorktreeRemoved(id) if this.selected_worktree == Some(*id) => {
                this.selected_worktree = None;
                this.invalidate(cx);
            }
            project::Event::WorktreeUpdatedEntries(id, _)
                if this.selected_worktree == Some(*id)
                    && (this.report.is_some() || this.collection.is_some())
                    && !this.report_stale =>
            {
                this.report_stale = true;
                cx.emit(DoctorEvidenceUpdated);
                cx.notify();
            }
            project::Event::DisconnectedFromRemote { .. }
            | project::Event::DisconnectedFromHost
                if this.restart_active.is_some() =>
            {
                cx.notify()
            }
            project::Event::DisconnectedFromRemote { .. }
            | project::Event::DisconnectedFromHost => this.invalidate(cx),
            _ => cx.notify(),
        })];
        let mut subscriptions = subscriptions;
        subscriptions.push(cx.observe_global::<settings::SettingsStore>(|this, cx| {
            if (this.report.is_some() || this.collection.is_some())
                && !this.report_stale
                && this.report_is_stale(cx)
            {
                this.report_stale = true;
                cx.emit(DoctorEvidenceUpdated);
                cx.notify();
            }
        }));
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let install = {
            let input = |placeholder: &str, cx: &mut Context<Self>| {
                cx.new(|cx| {
                    let mut editor = editor::Editor::single_line(window, cx);
                    editor.set_placeholder_text(placeholder, window, cx);
                    editor
                })
            };
            let mut input = input;
            let executable = input("Absolute path to already installed manager", cx);
            let version = input("Exact version (no moving channels)", cx);
            let package = input("Component or target when selected", cx);
            for field in [&executable, &version, &package] {
                subscriptions.push(cx.subscribe(field, |this, _, event, cx| {
                    if matches!(event, editor::EditorEvent::Edited { .. })
                        && this.repair_task.is_none()
                    {
                        this.preparation = None;
                        this.cancel_repair(cx);
                        cx.notify();
                    }
                }));
            }
            InstallForm {
                #[cfg(feature = "rust-tools")]
                kind: InstallKind::RustToolchain,
                #[cfg(all(not(feature = "rust-tools"), feature = "jvm-tools"))]
                kind: InstallKind::Mise,
                executable,
                version,
                package,
                open: false,
            }
        };
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let dependency = {
            let executable = cx.new(|cx| {
                let mut editor = editor::Editor::single_line(window, cx);
                editor.set_placeholder_text(
                    "Absolute installed native tool executable (no shims)",
                    window,
                    cx,
                );
                editor
            });
            subscriptions.push(cx.subscribe(&executable, |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) && this.repair_task.is_none()
                {
                    this.preparation = None;
                    this.cancel_repair(cx);
                }
            }));
            DependencyInputs {
                #[cfg(feature = "rust-tools")]
                kind: crate::doctor::DependencyProbe::RustCompiler,
                #[cfg(all(not(feature = "rust-tools"), feature = "jvm-tools"))]
                kind: crate::doctor::DependencyProbe::Java,
                executable,
                open: false,
            }
        };
        #[cfg(feature = "agentic")]
        let probe = {
            let mut input = |placeholder: &str, cx: &mut Context<Self>| {
                cx.new(|cx| {
                    let mut editor = editor::Editor::single_line(window, cx);
                    editor.set_placeholder_text(placeholder, window, cx);
                    editor
                })
            };
            let endpoint = input("Exact local engine endpoint", cx);
            let image = input("Immutable installed image reference", cx);
            let runtime = input("Absolute Made Linux runtime path", cx);
            for field in [&endpoint, &image, &runtime] {
                subscriptions.push(cx.subscribe(field, |this, _, event, cx| {
                    if matches!(event, editor::EditorEvent::Edited { .. })
                        && this.repair_task.is_none()
                    {
                        this.cancel_repair(cx);
                        cx.notify();
                    }
                }));
            }
            ProbeInputs {
                endpoint,
                image,
                runtime,
                open: false,
            }
        };
        #[cfg(not(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools")))]
        let _ = window;
        #[cfg(feature = "agentic")]
        let native_target = {
            let target = cx.new(|cx| {
                let mut target = editor::Editor::single_line(window, cx);
                target.set_placeholder_text(
                    "Relevant build/test target to review separately",
                    window,
                    cx,
                );
                target
            });
            subscriptions.push(cx.subscribe(&target, |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) {
                    this.cancel_repair(cx);
                }
            }));
            target
        };
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let security = DependencySecurityForm::new(window, cx);
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        for field in security.fields() {
            let scope_input = field == security.module || field == security.configuration;
            subscriptions.push(cx.subscribe(&field, move |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) && this.repair_task.is_none()
                {
                    this.cancel_repair(cx);
                    this.preparation = None;
                    if scope_input {
                        this.security.clear_all();
                    }
                    cx.notify();
                }
            }));
        }
        #[cfg(feature = "agentic")]
        let github = GithubEvidenceForm::new(window, cx);
        #[cfg(feature = "agentic")]
        for field in github.fields() {
            subscriptions.push(cx.subscribe(&field, |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) {
                    this.cancel_repair(cx);
                    this.github.clear_observations();
                    cx.notify();
                }
            }));
        }
        #[cfg(feature = "agentic")]
        let security_setup = SecuritySetupForm::new(window, cx);
        #[cfg(feature = "agentic")]
        for (_, _, field) in &security_setup.fields {
            subscriptions.push(cx.subscribe(field, |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) {
                    this.cancel_repair(cx);
                    cx.notify();
                }
            }));
        }
        let investigation = InvestigationForm::new(window, cx);
        for field in investigation.fields() {
            subscriptions.push(cx.subscribe(&field, |this, _, event, cx| {
                if matches!(event, editor::EditorEvent::Edited { .. }) && this.repair_task.is_none()
                {
                    this.cancel_repair(cx);
                    cx.notify();
                }
            }));
        }
        let comparison_form = ComparisonForm::new(window, cx);
        let scroll_handle = gpui::ScrollHandle::new();
        let review_anchor = gpui::ScrollAnchor::for_handle(scroll_handle.clone());
        Self {
            project,
            workspace,
            window_handle: window.window_handle(),
            focus_handle: cx.focus_handle(),
            scroll_handle,
            review_anchor,
            selected_worktree,
            selected_product: match product_flavor::ID {
                "rust" => DoctorProduct::Copper,
                "jvm" => DoctorProduct::Groovy,
                _ => DoctorProduct::Made,
            },
            generation: 0,
            report: None,
            report_stale: false,
            collection_revision: None,
            collection_cancelled: false,
            collection_failed: false,
            advanced_open: false,
            error: None,
            collection: None,
            closed: false,
            proposal: None,
            owner_result: None,
            reviewed_action: None,
            repair_status: None,
            repair_task: None,
            repair_cancel: None,
            preparation: None,
            restart_preparation: false,
            restart_active: None,
            history_open: false,
            history_records: Vec::new(),
            history_selected: None,
            history_change: None,
            history_task: None,
            history_status: None,
            saved_generation: None,
            export_task: None,
            export_status: None,
            export_writing: false,
            #[cfg(feature = "agentic")]
            probe,
            #[cfg(feature = "agentic")]
            github,
            #[cfg(feature = "agentic")]
            security_setup,
            #[cfg(feature = "agentic")]
            native_target,
            #[cfg(feature = "agentic")]
            native_evidence: Vec::new(),
            #[cfg(feature = "agentic")]
            native_investigation_evidence: Vec::new(),
            #[cfg(feature = "agentic")]
            native_repair_open: false,
            #[cfg(feature = "agentic")]
            native_session: None,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            install,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            dependency,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            security,
            investigation,
            current_comparison: None,
            baseline_comparison: None,
            verification_comparison: None,
            comparison_form,
            #[cfg(feature = "agentic")]
            change_session: None,
            #[cfg(feature = "rust-tools")]
            profile_result: None,
            task_result: None,
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            debug_result: None,
            _subscriptions: subscriptions,
        }
    }

    fn invalidate(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        self.cancel_repair(cx);
        self.owner_result = None;
        self.reviewed_action = None;
        self.generation = self.generation.saturating_add(1);
        self.collection = None;
        self.report = None;
        self.report_stale = false;
        self.collection_revision = None;
        self.collection_cancelled = false;
        self.collection_failed = false;
        self.investigation.tasks.clear();
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        self.investigation.debug_scenarios.clear();
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        self.security.clear_all();
        #[cfg(feature = "agentic")]
        self.github.clear_observations();
        self.current_comparison = None;
        self.task_result = None;
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        {
            self.debug_result = None;
        }
        self.baseline_comparison = None;
        self.verification_comparison = None;
        self.comparison_form.clear_selections();
        #[cfg(feature = "agentic")]
        {
            self.change_session = None;
        }
        #[cfg(feature = "rust-tools")]
        {
            self.profile_result = None;
        }
        #[cfg(feature = "agentic")]
        {
            self.native_evidence.clear();
            self.native_investigation_evidence.clear();
            self.native_session = None;
        }
        self.history_change = None;
        self.history_records.clear();
        self.history_selected = None;
        self.saved_generation = None;
        cx.notify();
    }

    fn select_worktree(&mut self, worktree: WorktreeId, cx: &mut Context<Self>) {
        if self
            .project
            .read(cx)
            .worktree_for_id(worktree, cx)
            .is_none()
        {
            self.error = Some(DiagnosticText::new(
                "Selected worktree is no longer available",
            ));
        } else {
            self.selected_worktree = Some(worktree);
            self.invalidate(cx);
        }
        cx.notify();
    }

    fn directory(&self, cx: &App) -> Option<ProjectPath> {
        if self.closed || self.workspace.upgrade().is_none() {
            return None;
        }
        let worktree_id = self.selected_worktree?;
        self.project.read(cx).worktree_for_id(worktree_id, cx)?;
        Some(ProjectPath {
            worktree_id,
            path: util::rel_path::RelPath::empty_arc(),
        })
    }

    fn host_label(&self, cx: &App) -> &'static str {
        let Some(directory) = self.directory(cx) else {
            return "Select a project worktree";
        };
        match self
            .project
            .read(cx)
            .doctor_observation(directory.worktree_id, cx)
            .host
        {
            DoctorExecutionHost::Local => "Local development host",
            DoctorExecutionHost::RemoteConnected => "Connected project development host",
            DoctorExecutionHost::RemoteUnavailable => "Project development host unavailable",
            DoctorExecutionHost::CollaborationGuest => {
                "Collaboration guest — host checks unavailable"
            }
        }
    }

    fn report_is_stale(&self, cx: &App) -> bool {
        self.report_stale
            || self.collection_revision.is_some_and(|revision| {
                cx.try_global::<settings::SettingsStore>()
                    .and_then(|store| store.observed_revision())
                    != Some(revision)
            })
    }

    fn cancel_collection(&mut self, cx: &mut Context<Self>) {
        if self.collection.is_some() {
            self.generation = self.generation.saturating_add(1);
            self.collection = None;
            self.collection_cancelled = true;
            cx.emit(DoctorEvidenceUpdated);
            cx.notify();
        }
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.repair_task.is_some() {
            self.error = Some(DiagnosticText::new(
                "Wait for the current reviewed action or cancel its observation before collecting a new scope",
            ));
            cx.notify();
            return;
        }
        let Some(directory) = self.directory(cx) else {
            self.error = Some(DiagnosticText::new(
                "Select an available project worktree before collecting checks",
            ));
            cx.notify();
            return;
        };
        self.invalidate(cx);
        self.collection_revision = cx
            .try_global::<settings::SettingsStore>()
            .and_then(|store| store.observed_revision());
        let generation = self.generation;
        let observed_at = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(time) => time.as_secs(),
            Err(error) => {
                self.collection_failed = true;
                self.error = Some(DiagnosticText::new(format!(
                    "Observation clock is unavailable: {error}"
                )));
                cx.emit(DoctorEvidenceUpdated);
                cx.notify();
                return;
            }
        };
        let owner = self.project.read(cx);
        let observation = owner.doctor_observation(directory.worktree_id, cx);
        let roots: Vec<_> = owner
            .visible_worktrees(cx)
            .map(|root| root.read(cx).root_name_str().to_owned())
            .collect();
        let Some(root) = owner.worktree_for_id(directory.worktree_id, cx) else {
            return;
        };
        let scope = DoctorScope {
            product: DiagnosticText::new(product_flavor::DISPLAY_NAME),
            project: DiagnosticText::new(roots.join(", ")),
            worktree: DiagnosticText::new(root.read(cx).root_name_str()),
            host: DiagnosticText::new(self.host_label(cx)),
        };
        let mut checks = vec![DoctorCheck::new(
            "project.authority",
            "Selected project authority",
            if observation.supports_reviewed_probe() {
                CheckOutcome::Ready
            } else {
                CheckOutcome::Blocked
            },
            &format!(
                "{} This observes existing authority and grants no permissions or production access.",
                crate::doctor::project_authority_description(&observation)
            ),
            observed_at,
        )];
        #[cfg(feature = "agentic")]
        checks.extend(crate::doctor_agent::collect_agent_checks(
            &self.project,
            None,
            directory.worktree_id,
            observed_at,
            cx,
        ));
        #[cfg(not(feature = "agentic"))]
        checks.push(DoctorCheck::new("made.agent", "Made agent checks", CheckOutcome::Unsupported, "Agent functionality is disabled in this build. Doctor did not initialize an agent, MCP server or schedule owner.", observed_at));
        let product = self.selected_product;
        #[cfg(feature = "rust-tools")]
        let copper = (product == DoctorProduct::Copper).then(|| {
            crate::doctor::collect_copper_checks(&self.project, directory.clone(), observed_at, cx)
        });
        #[cfg(not(feature = "rust-tools"))]
        if product == DoctorProduct::Copper {
            checks.push(DoctorCheck::new(
                "rust.unavailable",
                "Copper checks",
                CheckOutcome::Unsupported,
                "Rust tools are disabled in this build.",
                observed_at,
            ));
        }
        #[cfg(feature = "jvm-tools")]
        let groovy = (product == DoctorProduct::Groovy).then(|| {
            crate::doctor::collect_groovy_checks(&self.project, directory.clone(), observed_at, cx)
        });
        #[cfg(not(feature = "jvm-tools"))]
        if product == DoctorProduct::Groovy {
            checks.push(DoctorCheck::new(
                "jvm.unavailable",
                "Groovy checks",
                CheckOutcome::Unsupported,
                "JVM tools are disabled in this build. Groovy remains a planned product.",
                observed_at,
            ));
        }
        self.collection = Some(cx.spawn(async move |this, cx| {
            let result: anyhow::Result<DoctorReport> = async {
                #[cfg(feature = "rust-tools")]
                if let Some(task) = copper {
                    checks.extend(task.await?);
                }
                #[cfg(feature = "jvm-tools")]
                if let Some(task) = groovy {
                    checks.extend(task.await?);
                }
                let development = this.read_with(cx, |this, cx| {
                    crate::doctor::collect_development(
                        &this.project,
                        &directory,
                        product,
                        observed_at,
                        cx,
                    )
                })??;
                // Product and development projections share some Cargo facts.
                // Retain one observation per owner identity, not duplicate results.
                let mut identifiers: std::collections::HashSet<_> = checks
                    .iter()
                    .map(|check| check.id.as_str().to_owned())
                    .collect();
                checks.extend(
                    development
                        .into_iter()
                        .filter(|check| identifiers.insert(check.id.as_str().to_owned())),
                );
                DoctorReport::with_checks(scope, generation, observed_at, checks)
            }
            .await;
            if let Err(error) = this.update(cx, |this, cx| {
                if this.generation != generation || this.directory(cx).as_ref() != Some(&directory)
                {
                    return;
                }
                this.collection = None;
                match result {
                    Ok(report) => this.report = Some(report),
                    Err(error) => {
                        this.collection_failed = true;
                        this.error = Some(DiagnosticText::new(error.to_string()));
                    }
                }
                cx.emit(DoctorEvidenceUpdated);
                cx.notify();
            }) {
                log::debug!("Doctor view closed during collection: {error}");
            }
        }));
        cx.emit(DoctorEvidenceUpdated);
        cx.notify();
    }

    pub fn conversation_context(&self, cx: &App) -> Option<String> {
        let state = if self.collection.is_some() {
            "Running"
        } else if self.collection_cancelled {
            "Cancelled"
        } else if self.report.is_some() && self.report_is_stale(cx) {
            "Stale"
        } else if self.report.is_some() {
            "Collected"
        } else if self.collection_failed {
            "Failed"
        } else {
            return None;
        };
        let project = self.project.read(cx);
        let roots = project
            .visible_worktrees(cx)
            .map(|worktree| worktree.read(cx).root_name_str().to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let worktree = self
            .selected_worktree
            .and_then(|identifier| project.worktree_for_id(identifier, cx))
            .map(|worktree| worktree.read(cx).root_name_str().to_owned())
            .unwrap_or_else(|| "No selected worktree".to_owned());
        let mut summary = format!(
            "collection status: {state}. Project: {roots}. Worktree: {worktree}. Check family: {}. Host: {}. ",
            self.selected_product.name(),
            self.host_label(cx)
        );
        if let Some(report) = &self.report {
            summary.push_str(if state == "Collected" {
                "Latest collection replaces earlier collection snapshots for this scope. Current counts: "
            } else {
                "Retained observation counts, not a new current collection: "
            });
            for label in [
                "Blocked",
                "Failed",
                "Warning",
                "Passed",
                "Stale",
                "Unverified",
                "Unsupported",
                "Cancelled",
            ] {
                let count = report
                    .checks()
                    .iter()
                    .filter(|check| check.state_label() == label)
                    .count();
                summary.push_str(&format!("{label} {count}; "));
            }
        }
        if let Some(identity) = &self.reviewed_action {
            let identity: String = identity.as_str().chars().take(220).collect();
            summary.push_str(&format!("Reviewed action: {identity}. "));
            if self.repair_task.is_some() {
                if self
                    .repair_cancel
                    .as_ref()
                    .is_some_and(|sender| sender.is_closed())
                {
                    summary.push_str("Action status: Stop requested; awaiting the owned outcome. ");
                } else {
                    summary.push_str("Action status: Running; no completed result yet. ");
                }
            } else if let Some(result) = &self.owner_result {
                summary.push_str(&format!("Action status: {:?}. ", result.phase()));
                if let Some(detail) = result.operation_detail() {
                    summary.push_str(&format!(
                        "Observed operation: {}. ",
                        detail.as_str().chars().take(160).collect::<String>()
                    ));
                }
                if let Some(after) = result.after() {
                    summary.push_str(&format!(
                        "Relevant recheck: {} — {}. ",
                        after.state_label(),
                        after.detail.as_str().chars().take(160).collect::<String>()
                    ));
                }
            } else if let Some(status) = &self.repair_status {
                summary.push_str(&format!("Action status: {}. ", status.as_str()));
            }
            if self.repair_task.is_none()
                && matches!(
                    self.owner_result.as_ref(),
                    Some(OwnerRepair::Reproduction(_))
                )
                && let Some(investigation) =
                    self.report.as_ref().and_then(DoctorReport::investigation)
            {
                for row in investigation.evidence().iter().rev().take(3) {
                    summary.push_str(&format!(
                        "Evidence {:?}: {:?}; {:?}; {}. ",
                        row.owner,
                        row.classification,
                        row.observation,
                        if row.historical {
                            "historical observation"
                        } else {
                            "observed for this reviewed run"
                        }
                    ));
                }
            }
        }
        if let Some(report) = &self.report {
            summary.push_str(&format!(
                "Collected {}. {}. ",
                Self::timestamp(report.observed_at),
                report.summary()
            ));
            let mut checks: Vec<_> = report.checks().iter().collect();
            checks.sort_by_key(|check| check.priority());
            for check in checks.into_iter().take(4) {
                let detail: String = check.detail.as_str().chars().take(180).collect();
                summary.push_str(&format!(
                    "[{}; {}; {}] {}: {}. ",
                    check.state_label(),
                    check.id.as_str(),
                    check.evidence_source(),
                    check.title.as_str(),
                    detail
                ));
            }
        }
        if let Some(error) = &self.error {
            summary.push_str(&format!("Error: {}. ", error.as_str()));
        }
        // The full report remains in Doctor; conversation context is selected,
        // redacted owner metadata rather than raw logs or project source.
        let bounded = DiagnosticText::new(summary);
        Some(format!(
            "/doctor {} Only a bounded problem-first excerpt is included. Named reviewed-action observations do not establish general correctness or a verified repair. Collection reads configuration and cached development evidence; it does not execute the selected application, builds, tests, repairs or grant permissions. Inspect Doctor for the complete report; unperformed application checks remain Not run.",
            bounded.as_str()
        ))
    }

    pub fn conversation_context_failed(&mut self, cx: &mut Context<Self>) {
        self.error = Some(DiagnosticText::new(
            "Doctor cannot update its originating conversation. Results remain in Doctor; invoke /doctor from an available native conversation to discuss them.",
        ));
        cx.notify();
    }

    fn export_selected(&mut self, historical: bool, cx: &mut Context<Self>) {
        if self.export_task.is_some() {
            return;
        }
        let preparation = (|| {
            let (source, contents) = if historical {
                let record = self
                    .history_records
                    .iter()
                    .find(|record| Some(record.id) == self.history_selected)
                    .ok_or_else(|| anyhow::anyhow!("Select a retained record before exporting"))?
                    .clone();
                let text = format!(
                    "Historical record {}\nOutcome: {:?}\nRecorded: {}\nUpdated: {}\nAction: {}\nNative session: {}\nHistorical evidence does not verify current behavior.\n\n{}",
                    record.id,
                    record.outcome,
                    Self::timestamp(record.created_at),
                    Self::timestamp(record.updated_at),
                    record
                        .action
                        .as_ref()
                        .map(|value| value.as_str())
                        .unwrap_or("No action"),
                    record
                        .session_id
                        .as_ref()
                        .map(|value| value.as_str())
                        .unwrap_or("No session"),
                    record.report.export()?
                );
                (ExportSource::History(record), text)
            } else {
                let report = self
                    .report
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Collect checks before exporting"))?;
                (
                    ExportSource::Current {
                        generation: report.generation,
                        stale: self.report_is_stale(cx),
                    },
                    if self.report_is_stale(cx) {
                        format!(
                            "Stale report — selected inputs changed after collection. Original observations below do not establish current health; collect again.\n\n{}",
                            report.export()?
                        )
                    } else {
                        report.export()?
                    },
                )
            };
            anyhow::ensure!(
                contents.len() <= crate::doctor::MAX_REPORT_BYTES,
                "Selected Doctor export exceeds 64 KiB; it was not written"
            );
            let filesystem = self
                .workspace
                .read_with(cx, |workspace, _| workspace.app_state().fs.clone())?;
            let product = Self::product();
            let generation = self.generation;
            let directory = std::env::home_dir().unwrap_or_default();
            let path = cx.prompt_for_new_path(&directory, Some("made-doctor-report.md"));
            Ok::<_, anyhow::Error>((source, contents, filesystem, product, generation, path))
        })();
        let (source, contents, filesystem, product, generation, path) = match preparation {
            Ok(value) => value,
            Err(error) => {
                self.export_status =
                    Some(DiagnosticText::new(format!("Export unavailable: {error}")));
                cx.notify();
                return;
            }
        };
        self.export_status = Some(DiagnosticText::new(
            "Choose an explicit local destination for only the selected redacted report. Cancelling the save dialog writes nothing.",
        ));
        self.export_task = Some(cx.spawn(async move |this, cx| {
            let result: anyhow::Result<bool> = async {
                let Some(path) = path.await?? else { return Ok(false); };
                anyhow::ensure!(path.is_absolute(), "Export requires an absolute selected destination");
                this.read_with(cx, |this, cx| {
                    anyhow::ensure!(!this.closed && this.generation == generation, "Doctor scope changed while choosing an export destination; choose again");
                    if let ExportSource::Current { generation, stale } = &source {
                        anyhow::ensure!(this.report.as_ref().is_some_and(|report| report.generation == *generation), "Selected checks changed before export");
                        anyhow::ensure!(this.report_is_stale(cx) == *stale, "Report freshness changed while choosing an export destination; review it again");
                    }
                    Ok::<_, anyhow::Error>(())
                })??;
                if let ExportSource::History(expected) = &source {
                    let store = cx.update(crate::doctor::DoctorHistoryStore::for_app)?;
                    let records = store.records(product, Self::now()?).await?;
                    let current = records.iter().find(|record| record.id == expected.id).ok_or_else(|| anyhow::anyhow!("Selected history expired or was deleted; refresh before export"))?;
                    anyhow::ensure!(serde_json::to_string(current)? == serde_json::to_string(expected)?, "Selected history changed before export; review it again");
                }
                if let Some(metadata) = filesystem.metadata(&path).await? {
                    anyhow::ensure!(!metadata.is_dir && !metadata.is_symlink, "Choose an ordinary file destination; directories and symbolic links are not overwritten");
                }
                this.update(cx, |this, cx| {
                    anyhow::ensure!(!this.closed && this.generation == generation, "Doctor scope changed before writing");
                    if let ExportSource::Current { stale, .. } = &source {
                        anyhow::ensure!(this.report_is_stale(cx) == *stale, "Report freshness changed before writing; review it again");
                    }
                    this.export_writing = true;
                    this.export_status = Some(DiagnosticText::new("Writing only the explicitly selected redacted report. The accepted atomic file write may finish."));
                    cx.notify();
                    Ok::<_, anyhow::Error>(())
                })??;
                filesystem.atomic_write(path, contents).await?;
                Ok(true)
            }.await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.export_task = None;
                this.export_writing = false;
                this.export_status = Some(DiagnosticText::new(match result {
                    Ok(true) => "Selected redacted report exported. Export does not verify a repair.".to_owned(),
                    Ok(false) => "Export cancelled; no file was written.".to_owned(),
                    Err(error) => format!("Export failed: {error}; no successful export is claimed"),
                }));
                cx.notify();
            }) { log::debug!("Doctor export view closed: {error}"); }
        }));
        cx.notify();
    }

    fn cancel_export(&mut self, cx: &mut Context<Self>) {
        if !self.export_writing {
            self.export_task = None;
            self.export_status = Some(DiagnosticText::new(
                "Export cancelled before the file write; no successful export is claimed",
            ));
        } else {
            self.export_status = Some(DiagnosticText::new(
                "The approved atomic file write was accepted and may finish; wait for its observed result",
            ));
        }
        cx.notify();
    }

    fn load_history(&mut self, cx: &mut Context<Self>) {
        if self.history_task.is_some() {
            return;
        }
        self.history_open = true;
        self.history_change = None;
        let family = self.selected_product;
        let product = Self::product();
        let preparation = crate::doctor::DoctorHistoryStore::for_app(cx)
            .and_then(|store| Ok((store, Self::now()?)));
        let (store, now) = match preparation {
            Ok(value) => value,
            Err(error) => {
                self.history_status =
                    Some(DiagnosticText::new(format!("History unavailable: {error}")));
                cx.notify();
                return;
            }
        };
        self.history_task = Some(cx.spawn(async move |this, cx| {
            let result = store.records(product, now).await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.history_task = None;
                if this.selected_product == family {
                    match result {
                        Ok(mut records) => {
                            records.reverse();
                            this.history_records = records;
                            if !this.history_records.iter().any(|record| Some(record.id) == this.history_selected) { this.history_selected = None; }
                            this.history_status = Some(DiagnosticText::new("Retained redacted records loaded. Historical outcomes do not verify the current project."));
                        }
                        Err(error) => { this.history_records.clear(); this.history_selected = None; this.history_status = Some(DiagnosticText::new(format!("History could not be read: {error}. It was not replaced. Explicitly review clearing to discard this product's records."))); }
                    }
                }
                cx.notify();
            }) { log::debug!("Doctor history view closed: {error}"); }
        }));
        cx.notify();
    }

    fn review_history_clear(&mut self, cx: &mut Context<Self>) {
        if self.history_task.is_some() || self.repair_task.is_some() {
            return;
        }
        self.history_change = None;
        let family = self.selected_product;
        let product = Self::product();
        let store = match crate::doctor::DoctorHistoryStore::for_app(cx) {
            Ok(store) => store,
            Err(error) => {
                self.history_status =
                    Some(DiagnosticText::new(format!("History unavailable: {error}")));
                cx.notify();
                return;
            }
        };
        self.history_task = Some(cx.spawn(async move |this, cx| {
            let result = store.review_clear(product).await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.history_task = None;
                if this.selected_product == family {
                    match result {
                        Ok(fingerprint) => {
                            this.history_change = Some(HistoryChange::Clear { fingerprint })
                        }
                        Err(error) => {
                            this.history_status = Some(DiagnosticText::new(format!(
                                "Cannot prepare history clearing: {error}"
                            )))
                        }
                    }
                }
                cx.notify();
            }) {
                log::debug!("Doctor history clear view closed: {error}");
            }
        }));
        cx.notify();
    }

    fn approve_history_change(&mut self, cx: &mut Context<Self>) {
        if self.history_task.is_some() || self.repair_task.is_some() {
            return;
        }
        let Some(change) = self.history_change.take() else {
            return;
        };
        let family = self.selected_product;
        let product = Self::product();
        let preparation = crate::doctor::DoctorHistoryStore::for_app(cx)
            .and_then(|store| Ok((store, Self::now()?)));
        let (store, now) = match preparation {
            Ok(value) => value,
            Err(error) => {
                self.history_status =
                    Some(DiagnosticText::new(format!("History unavailable: {error}")));
                cx.notify();
                return;
            }
        };
        if let HistoryChange::Save(report) = &change
            && (self.report.as_ref() != Some(report)
                || report.generation != self.generation
                || self.report.as_ref().map(|current| current.generation)
                    != Some(report.generation)
                || self.saved_generation == Some(report.generation))
        {
            self.history_status = Some(DiagnosticText::new(
                "Collected report changed or was already saved; collect and review again",
            ));
            cx.notify();
            return;
        }
        let approved_report = match &change {
            HistoryChange::Save(report) => Some(report.clone()),
            _ => None,
        };
        self.history_task = Some(cx.spawn(async move |this, cx| {
            let mut saved_generation = None;
            let result: anyhow::Result<()> = async {
                match change {
                    HistoryChange::Save(report) => {
                        saved_generation = Some(report.generation);
                        store
                            .record(
                                product,
                                crate::doctor::DoctorHistoryRecord::new(
                                    report,
                                    None,
                                    crate::doctor::ReceiptOutcome::Completed,
                                    now,
                                ),
                                now,
                            )
                            .await?;
                    }
                    HistoryChange::Delete(record) => {
                        store.delete_reviewed(product, &record, now).await?
                    }
                    HistoryChange::Clear { fingerprint } => {
                        store.clear_reviewed(product, &fingerprint).await?
                    }
                }
                Ok(())
            }
            .await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.history_task = None;
                if this.selected_product == family {
                    match result {
                        Ok(()) => {
                            if approved_report
                                .as_ref()
                                .is_none_or(|approved| this.report.as_ref() == Some(approved))
                            {
                                this.saved_generation = saved_generation.or(this.saved_generation);
                            }
                            this.history_selected = None;
                            this.load_history(cx);
                        }
                        Err(error) => {
                            this.history_status = Some(DiagnosticText::new(format!(
                                "History change failed: {error}; no success is claimed"
                            )))
                        }
                    }
                }
                cx.notify();
            }) {
                log::debug!("Doctor history change view closed after database action: {error}");
            }
        }));
        cx.notify();
    }

    fn history_controls(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let busy = self.history_task.is_some();
        let mut content = v_flex()
            .gap_2()
            .child(
                h_flex()
                    .flex_wrap()
                    .gap_2()
                    .child(
                        div().debug_selector(|| "doctor-history-open".into()).child(
                            Button::new("doctor-history-open", "Inspect retained history")
                                .disabled(busy)
                                .tab_index(0_isize)
                                .on_click(cx.listener(|this, _, _, cx| this.load_history(cx))),
                        ),
                    )
                    .child(
                        div().debug_selector(|| "doctor-history-save".into()).child(
                            Button::new("doctor-history-save", "Review saving these checks")
                                .disabled(
                                    busy || self.repair_task.is_some()
                                        || self.report.is_none()
                                        || self.report.as_ref().is_some_and(|report| {
                                            self.saved_generation == Some(report.generation)
                                        }),
                                )
                                .tab_index(0_isize)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(report) = this.report.clone() {
                                        this.history_open = true;
                                        this.history_change = Some(HistoryChange::Save(report));
                                        cx.notify();
                                    }
                                })),
                        ),
                    ),
            )
            .when_some(self.history_status.as_ref(), |element, status| {
                element.child(
                    div()
                        .id("doctor-history-status")
                        .role(gpui::Role::Status)
                        .aria_live(gpui::accesskit::Live::Polite)
                        .child(Label::new(status.as_str().to_owned())),
                )
            });
        if !self.history_open {
            return content;
        }
        content = content.child(Label::new(format!("{} history: 30 days, at most 100 records and 5 MiB. Only redacted summaries are retained. Clearing does not delete agent sessions.", Self::product().name())))
            .child(div().debug_selector(|| "doctor-history-clear".into()).child(Button::new("doctor-history-clear", "Review clearing this product's history").disabled(busy || self.repair_task.is_some()).tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| this.review_history_clear(cx)))));
        if let Some(change) = &self.history_change {
            let preview = match change {
                HistoryChange::Save(report) => format!(
                    "Save these {} redacted check summaries collected {}. Completed means collection finished; it does not prove an issue fixed.",
                    report.checks().len(),
                    Self::timestamp(report.observed_at)
                ),
                HistoryChange::Delete(record) => format!(
                    "Delete only Doctor record {} from {}. The linked native agent session will remain.",
                    record.id,
                    Self::timestamp(record.created_at)
                ),
                HistoryChange::Clear { .. } => format!(
                    "Discard all {} Doctor history, including unreadable data. Native sessions, files, credentials and caches remain. Changed history will require a new review.",
                    Self::product().name()
                ),
            };
            content = content.child(Label::new(preview)).child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .debug_selector(|| "doctor-history-confirm".into())
                            .child(
                                Button::new(
                                    "doctor-history-confirm",
                                    "Confirm this history change",
                                )
                                .disabled(busy || self.repair_task.is_some())
                                .tab_index(0_isize)
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.approve_history_change(cx)),
                                ),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "doctor-history-cancel".into())
                            .child(
                                Button::new("doctor-history-cancel", "Cancel history change")
                                    .tab_index(0_isize)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.history_change = None;
                                        cx.notify();
                                    })),
                            ),
                    ),
            );
        }
        content = content.children(self.history_records.iter().enumerate().map(
            |(index, record)| {
                let id = record.id;
                div()
                    .debug_selector(move || format!("doctor-history-record-{index}"))
                    .child(
                        Button::new(
                            ("doctor-history-record", index),
                            format!(
                                "{} · {:?} · {} · {} · {}",
                                Self::timestamp(record.created_at),
                                record.outcome,
                                record.report.scope.project.as_str(),
                                record.report.scope.worktree.as_str(),
                                record.report.scope.host.as_str()
                            ),
                        )
                        .toggle_state(self.history_selected == Some(id))
                        .disabled(busy)
                        .tab_index(0_isize)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.history_selected = Some(id);
                            this.history_change = None;
                            cx.notify();
                        })),
                    )
            },
        ));
        if let Some(record) = self
            .history_records
            .iter()
            .find(|record| Some(record.id) == self.history_selected)
        {
            let selected = record.clone();
            if let Some(summary) = record.report.investigation() {
                content = content.child(Self::investigation_summary_element(summary, true));
            }
            if let Some(comparison) = record.report.comparison() {
                content = content.child(Self::comparison_result_element(comparison, true));
            }
            content = content.child(Label::new("Past evidence — never rerun automatically and never used as current verification"))
                .child(Label::new(format!("Recorded {:?}; updated {}", record.outcome, Self::timestamp(record.updated_at))))
                .when_some(record.action.as_ref(), |element, action| element.child(Label::new(action.as_str().to_owned())))
                .child(div().debug_selector(|| "doctor-history-delete".into()).child(Button::new("doctor-history-delete", "Review deleting this record").disabled(busy || self.repair_task.is_some() || record.outcome == crate::doctor::ReceiptOutcome::Running).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| { this.history_change = Some(HistoryChange::Delete(selected.clone())); cx.notify(); }))));
            content = content.child(
                div()
                    .debug_selector(|| "doctor-export-history".into())
                    .child(
                        Button::new(
                            "doctor-export-history",
                            "Export this redacted historical record…",
                        )
                        .disabled(self.export_task.is_some())
                        .tab_index(0_isize)
                        .on_click(cx.listener(|this, _, _, cx| this.export_selected(true, cx))),
                    ),
            );
            #[cfg(feature = "agentic")]
            if let Some(session) = record.session_id.as_ref().or_else(|| {
                record
                    .report
                    .comparison()
                    .and_then(|comparison| comparison.change.session_id.as_ref())
            }) {
                let session = session.as_str().to_owned();
                content = content.child(
                    div()
                        .debug_selector(|| "doctor-history-session".into())
                        .child(
                            Button::new(
                                "doctor-history-session",
                                "Open existing native repair session",
                            )
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                move |this, _, window, cx| match cx.build_action(
                                    "agent::OpenDoctorSession",
                                    Some(serde_json::json!({"session_id": session})),
                                ) {
                                    Ok(action) => window.dispatch_action(action, cx),
                                    Err(error) => {
                                        this.history_status = Some(DiagnosticText::new(format!(
                                            "Native session navigation is unavailable: {error}"
                                        )));
                                        cx.notify();
                                    }
                                },
                            )),
                        ),
                );
            }
            #[cfg(not(feature = "agentic"))]
            if record.session_id.is_some() {
                content = content.child(Label::new("Session navigation is unavailable because agentic-tools is disabled. The session is preserved."));
            }
            content = content.children(record.report.checks().iter().map(|check| {
                v_flex()
                    .gap_1()
                    .child(Label::new(format!(
                        "{} — {}",
                        check.outcome.label(),
                        check.title.as_str()
                    )))
                    .child(Label::new(check.detail.as_str().to_owned()))
            }));
        }
        content
    }

    fn timestamp(value: u64) -> String {
        match i64::try_from(value)
            .ok()
            .and_then(|value| time::OffsetDateTime::from_unix_timestamp(value).ok())
        {
            Some(timestamp) => format!(
                "{} UTC",
                time_format::format_localized_timestamp(
                    timestamp,
                    timestamp,
                    time::UtcOffset::UTC,
                    time_format::TimestampFormat::Absolute
                )
            ),
            None => "Observation timestamp unavailable".to_owned(),
        }
    }

    fn product() -> DoctorProduct {
        DoctorProduct::current()
    }

    fn now() -> anyhow::Result<u64> {
        Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
    }

    fn review_directory(&self, cx: &App) -> anyhow::Result<ProjectPath> {
        anyhow::ensure!(
            self.repair_task.is_none(),
            "A reviewed action is already running"
        );
        let report = self
            .report
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Collect selected checks before reviewing a repair"))?;
        anyhow::ensure!(
            report.generation == self.generation,
            "Collected scope is stale"
        );
        self.directory(cx)
            .ok_or_else(|| anyhow::anyhow!("Selected project scope is unavailable"))
    }

    #[cfg(all(feature = "agentic", any(test, feature = "test-support")))]
    pub fn configure_github_inputs_fixture(
        &mut self,
        account: &str,
        owner: &str,
        repository: &str,
        target: &str,
        credential_reader: std::sync::Arc<dyn Fn() -> Option<String> + Send + Sync>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for (field, value) in [
            (self.github.account.clone(), account),
            (self.github.owner.clone(), owner),
            (self.github.repository.clone(), repository),
            (self.github.target.clone(), target),
        ] {
            field.update(cx, |input, cx| input.set_text(value, window, cx));
        }
        self.github.credential_reader = Some(credential_reader);
        cx.notify();
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn operation_busy_fixture(&self) -> bool {
        self.collection.is_some() || self.repair_task.is_some() || self.preparation.is_some()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn configure_native_repair_fixture(
        &mut self,
        target: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<bool> {
        #[cfg(feature = "agentic")]
        {
            let check = self
                .report
                .as_ref()
                .and_then(|report| report.checks().first())
                .ok_or_else(|| {
                    anyhow::anyhow!("Collect current checks before fixture selection")
                })?;
            self.native_evidence = vec![check.id.as_str().to_owned()];
            self.native_investigation_evidence.clear();
            self.native_repair_open = true;
            self.native_target
                .update(cx, |editor, cx| editor.set_text(target, window, cx));
            cx.notify();
            Ok(true)
        }
        #[cfg(not(feature = "agentic"))]
        {
            let _ = (target, window, cx);
            Ok(false)
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn configure_investigation_fixture(
        &mut self,
        selection: crate::doctor::DoctorInvestigationSelection,
        inputs: Vec<crate::doctor::DoctorEvidenceInput>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<bool> {
        #[cfg(feature = "agentic")]
        {
            let directory = self.review_directory(cx)?;
            let report = self
                .report
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Collect current checks first"))?;
            let binding = crate::doctor::project_repair_binding(
                &self.project,
                directory,
                report.generation,
                cx,
            )?;
            let mut investigation = crate::doctor::DoctorInvestigation::new(
                selection.clone(),
                report.scope.clone(),
                binding.clone(),
                report.observed_at,
            )?;
            for input in inputs {
                investigation.record(&binding, &selection, input)?;
            }
            self.current_comparison =
                Some(investigation.comparison_snapshot_at(&self.project, cx)?);
            report.set_investigation(investigation.summary().clone())?;
            self.native_evidence.clear();
            self.native_investigation_evidence.clear();
            self.native_repair_open = true;
            self.native_target.update(cx, |editor, cx| {
                editor.set_text(selection.validation_target.as_str(), window, cx)
            });
            cx.notify();
            Ok(true)
        }
        #[cfg(not(feature = "agentic"))]
        {
            let _ = (selection, inputs, window, cx);
            Ok(false)
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn configure_comparison_change_fixture(
        &mut self,
        description: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.comparison_form
            .change
            .update(cx, |editor, cx| editor.set_text(description, window, cx));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn report_fixture(&self) -> Option<&DoctorReport> {
        self.report.as_ref()
    }

    #[cfg(all(
        any(test, feature = "test-support"),
        feature = "rust-tools",
        feature = "jvm-tools"
    ))]
    pub fn dependency_security_busy_fixture(&self) -> bool {
        self.preparation.is_some() || self.repair_task.is_some()
    }

    #[cfg(all(
        any(test, feature = "test-support"),
        feature = "rust-tools",
        feature = "jvm-tools"
    ))]
    pub fn dependency_security_instance_fixture(&self, name: &str) -> Option<usize> {
        self.security
            .projection
            .as_ref()?
            .graph()
            .nodes()
            .iter()
            .position(|node| node.name == name)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn error_fixture(&self) -> Option<&str> {
        self.error.as_ref().map(DiagnosticText::as_str)
    }

    pub fn retain_native_repair_subscription(&mut self, subscription: Subscription) {
        self._subscriptions.push(subscription);
    }

    #[cfg(feature = "agentic")]
    pub(crate) fn native_session_available(&mut self, session: &str, cx: &mut Context<Self>) {
        self.native_session = Some(session.to_owned());
        cx.notify();
    }

    #[cfg(feature = "agentic")]
    fn native_session_control(&self, session: String, cx: &mut Context<Self>) -> gpui::AnyElement {
        div()
            .debug_selector(|| "doctor-native-session".into())
            .child(
                Button::new(
                    "doctor-native-session",
                    "Open native repair session and normal change review",
                )
                .tab_index(0_isize)
                .on_click(cx.listener(move |this, _, window, cx| {
                    match cx.build_action(
                        "agent::OpenDoctorSession",
                        Some(serde_json::json!({"session_id": session})),
                    ) {
                        Ok(action) => window.dispatch_action(action, cx),
                        Err(error) => {
                            this.error = Some(DiagnosticText::new(error.to_string()));
                            cx.notify();
                        }
                    }
                })),
            )
            .into_any_element()
    }

    #[cfg(all(
        feature = "agentic",
        any(feature = "rust-tools", feature = "jvm-tools")
    ))]
    pub(crate) fn review_dependency_native_repair(
        &mut self,
        proposal: crate::doctor_dependencies::DoctorDependencyRepairProposal,
        target: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let report = self.report.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Collect current checks before selecting a dependency repair")
            })?;
            crate::doctor_agent::prepare_dependency_native_repair(
                &self.project,
                directory,
                report,
                proposal,
                target,
                cx,
            )
            .map(OwnerRepair::Native)
        })();
        self.accept_proposal(result, window, cx);
    }

    #[cfg(all(
        any(test, feature = "test-support"),
        feature = "rust-tools",
        feature = "jvm-tools"
    ))]
    pub fn configure_dependency_security_inputs_fixture(
        &mut self,
        maven: bool,
        target: &str,
        public_coordinate: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.cancel_repair(cx);
        self.security.kind = if maven {
            SecurityKind::Maven
        } else {
            SecurityKind::Cargo
        };
        self.investigation.open = true;
        self.investigation.scope = crate::doctor::DoctorInvestigationScope::Development;
        self.investigation.problem = crate::doctor::DoctorInvestigationProblem::Dependencies;
        for (field, text) in [
            (
                self.investigation.symptom.clone(),
                "Selected dependency needs review",
            ),
            (self.investigation.target.clone(), target),
            (
                self.investigation.command.clone(),
                "cargo test --workspace --locked --offline",
            ),
            (
                self.comparison_form.change.clone(),
                "Reviewed selected dependency manifest change",
            ),
            (
                self.investigation.workload.clone(),
                "Selected build and affected tests",
            ),
            (
                self.investigation.configuration.clone(),
                "Development fixture configuration",
            ),
            (
                self.investigation.cache.clone(),
                "Existing development caches",
            ),
            (
                self.investigation.writes.clone(),
                "Only reviewed project changes and build caches",
            ),
            (
                self.investigation.subprocesses.clone(),
                "Existing selected build and test commands",
            ),
            (self.security.public_coordinate.clone(), public_coordinate),
            (
                self.security.rustsec_commit.clone(),
                "1111111111111111111111111111111111111111",
            ),
        ] {
            field.update(cx, |editor, cx| editor.set_text(text, window, cx));
        }
        cx.notify();
    }

    #[cfg(all(
        any(test, feature = "test-support"),
        feature = "rust-tools",
        feature = "jvm-tools"
    ))]
    pub async fn dependency_security_archive_fixture(
        maven: bool,
        withdrawn: bool,
    ) -> anyhow::Result<Vec<u8>> {
        let (name, document) = if maven {
            if withdrawn {
                (
                    "GHSA-dddd-eeee-ffff.json",
                    r#"{"id":"GHSA-dddd-eeee-ffff","modified":"2026-10-01T00:00:00Z","withdrawn":"2026-10-01T00:00:00Z"}"#,
                )
            } else {
                (
                    "GHSA-aaaa-bbbb-cccc.json",
                    r#"{"id":"GHSA-aaaa-bbbb-cccc","modified":"2026-10-01T00:00:00Z","summary":"PUBLIC_BODY_NOT_HISTORY","affected":[{"package":{"ecosystem":"Maven","name":"org.junit.jupiter:junit-jupiter-api"},"versions":["5.11.4"]}]}"#,
                )
            }
        } else {
            (
                "advisory-db-fixture/crates/public-example/RUSTSEC-2026-0001.md",
                "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"public-example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [\">=2.0.0\"]\n```\n# PUBLIC_BODY_NOT_HISTORY",
            )
        };
        let mut writer = async_zip::base::write::ZipFileWriter::new(Vec::new());
        writer
            .write_entry_whole(
                async_zip::ZipEntryBuilder::new(name.into(), async_zip::Compression::Deflate),
                document.as_bytes(),
            )
            .await?;
        Ok(writer.close().await?)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn configure_dependency_repair_fixture(
        &mut self,
        target: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<bool> {
        #[cfg(all(feature = "agentic", feature = "rust-tools"))]
        {
            use crate::doctor_dependencies::*;
            let directory = self.review_directory(cx)?;
            let report = self
                .report
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Current fixture report is unavailable"))?;
            let projection = DoctorDependencyProjection::cargo(
                &self.project,
                directory.clone(),
                directory,
                report.generation,
                report.observed_at,
                cx,
            )?;
            let node = projection
                .graph()
                .nodes()
                .iter()
                .position(|node| node.source.kind() == DependencySourceKind::CratesIo)
                .ok_or_else(|| {
                    anyhow::anyhow!("Current fixture has no public resolved dependency")
                })?;
            let metadata = RegistryMetadata {
                query: RegistryQuery::for_resolved(projection.graph(), node)?,
                versions: vec![RegistryVersion {
                    version: "2.0.0".into(),
                    yanked: false,
                    required_rust: None,
                }],
                fetched_at: report.observed_at,
                response_date: Some(report.observed_at),
                response_age: None,
                modified_at: None,
                body_identity: DependencyIdentity::from_fields(&[
                    "Explicit reviewed registry fixture, not live retrieval",
                ])?,
                coverage: DependencyCoverage::Complete,
                limitations: vec![],
            };
            let choice = recommend_updates(
                projection.graph(),
                &metadata,
                None,
                report.observed_at,
                || Ok(()),
            )?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No selected fixture candidate"))?;
            let proposal = DoctorDependencyRepairProposal::update(projection, node, &choice)?;
            self.review_dependency_native_repair(proposal, target, window, cx);
            Ok(true)
        }
        #[cfg(not(all(feature = "agentic", feature = "rust-tools")))]
        {
            let _ = (target, window, cx);
            Ok(false)
        }
    }

    #[cfg(feature = "agentic")]
    fn review_native_repair(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let report = self
                .report
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Collect current checks first"))?;
            let target = self.native_target.read(cx).text(cx);
            let review = if self.native_investigation_evidence.is_empty() {
                crate::doctor_agent::prepare_native_repair(
                    &self.project,
                    directory,
                    report,
                    &self.native_evidence,
                    &target,
                    cx,
                )
            } else {
                anyhow::ensure!(
                    self.native_evidence.is_empty(),
                    "Select either product checks or investigation observations"
                );
                let investigation = report.investigation().ok_or_else(|| {
                    anyhow::anyhow!("Current investigation evidence is unavailable")
                })?;
                anyhow::ensure!(
                    DiagnosticText::new(&target) == investigation.selection.validation_target,
                    "Validation target changed; collect current investigation evidence"
                );
                crate::doctor_agent::prepare_investigation_native_repair(
                    &self.project,
                    directory,
                    report,
                    &investigation.selection,
                    &self.native_investigation_evidence,
                    cx,
                )
            };
            review.map(OwnerRepair::Native)
        })();
        self.accept_proposal(result, window, cx);
    }

    #[cfg(feature = "agentic")]
    fn native_repair_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        if !self.native_repair_open {
            return div()
                .debug_selector(|| "doctor-native-open".into())
                .child(
                    Button::new(
                        "doctor-native-open",
                        "Select evidence for a native repair proposal",
                    )
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.native_repair_open = true;
                        cx.notify();
                    })),
                )
                .into_any_element();
        }

        let mut controls =
            v_flex()
                .gap_2()
                .child(Label::new(
                    "Native repair proposal — select redacted checks or investigation observations below",
                ))
                .child(Label::new(
                    "Validation target (label only; execution requires separate review)",
                ))
                .child(self.native_target.clone())
                .child(
                    div()
                        .debug_selector(|| "doctor-native-review".into())
                        .child(
                            Button::new(
                                "doctor-native-review",
                                "Review selected evidence and native repair handoff",
                            )
                            .disabled(
                                (self.native_evidence.is_empty() && self.native_investigation_evidence.is_empty())
                                    || self.report.is_none()
                                    || self.repair_task.is_some(),
                            )
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                |this, _, window, cx| this.review_native_repair(window, cx),
                            )),
                        ),
                );
        if let Some(investigation) = self.report.as_ref().and_then(DoctorReport::investigation) {
            controls = controls.child(Label::new(format!(
                "Selected investigation target: {}",
                investigation.selection.validation_target.as_str()
            )));
            for (index, evidence) in investigation.evidence().iter().enumerate() {
                let identifier = evidence.id;
                controls = controls.child(
                    v_flex()
                        .gap_1()
                        .child(Label::new(format!(
                            "{:?} / {:?}{} — {}",
                            evidence.classification,
                            evidence.observation,
                            if evidence.historical {
                                " (historical)"
                            } else {
                                ""
                            },
                            evidence.detail.as_str()
                        )))
                        .child(
                            div()
                                .debug_selector(move || {
                                    format!("doctor-investigation-evidence-{index}")
                                })
                                .child(
                                    Button::new(
                                        ("doctor-investigation-evidence", index),
                                        "Select observation for native repair",
                                    )
                                    .toggle_state(
                                        self.native_investigation_evidence.contains(&identifier),
                                    )
                                    .disabled(self.repair_task.is_some())
                                    .tab_index(0_isize)
                                    .on_click(cx.listener(
                                        move |this, _, _, cx| {
                                            this.cancel_repair(cx);
                                            this.native_evidence.clear();
                                            if this
                                                .native_investigation_evidence
                                                .contains(&identifier)
                                            {
                                                this.native_investigation_evidence
                                                    .retain(|selected| *selected != identifier);
                                            } else if this.native_investigation_evidence.len() < 8 {
                                                this.native_investigation_evidence.push(identifier);
                                            } else {
                                                this.error = Some(DiagnosticText::new(
                                                    "Select at most eight observations",
                                                ));
                                            }
                                            cx.notify();
                                        },
                                    )),
                                ),
                        ),
                );
            }
        }
        if let Some(OwnerRepair::Native(review)) = &self.proposal {
            controls = controls
                .child(Label::new("Exact redacted prompt to the selected provider"))
                .child(Label::new(review.action().prompt().to_owned()));
        }
        if self.repair_task.is_none()
            && let Some(OwnerRepair::Native(review)) = &self.owner_result
            && let Some(session) = review.action().session_id()
        {
            controls = controls.child(self.native_session_control(session, cx));
        }
        controls.into_any_element()
    }

    fn review_project_action(
        &mut self,
        action: crate::doctor::DoctorProjectRepairAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let review = crate::doctor::prepare_project_repair(
                &self.project,
                directory,
                action,
                self.generation,
                Self::now()?,
                cx,
            )?;
            Ok::<_, anyhow::Error>(OwnerRepair::Project(review))
        })();
        self.accept_proposal(result, window, cx);
    }

    #[cfg(feature = "agentic")]
    fn review_agent_action(
        &mut self,
        request: crate::doctor_agent::DoctorAgentRepairRequest,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let review = crate::doctor_agent::prepare_agent_owner_repair(
                &self.project,
                directory,
                request,
                self.generation,
                Self::now()?,
                cx,
            )?;
            Ok::<_, anyhow::Error>(OwnerRepair::Agent(review))
        })();
        self.accept_proposal(result, window, cx);
    }

    fn accept_proposal(
        &mut self,
        result: anyhow::Result<OwnerRepair>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match result {
            Ok(review) => {
                if let Some(mut previous) = self.proposal.take() {
                    if let Err(error) = previous.cancel() {
                        self.error = Some(DiagnosticText::new(error.to_string()));
                        cx.notify();
                        return;
                    }
                }
                self.proposal = Some(review);
                self.review_anchor.scroll_to(window, cx);
                self.owner_result = None;
                self.error = None;
                self.repair_status = Some(DiagnosticText::new(
                    "Proposed only — review the exact owner action and effects before approval",
                ));
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn cancel_repair(&mut self, cx: &mut Context<Self>) {
        self.preparation = None;
        self.restart_preparation = false;
        if let Some(mut review) = self.proposal.take() {
            match review.cancel() {
                Ok(()) => {
                    self.repair_status = Some(DiagnosticText::new(
                        "Proposal cancelled; no repair was executed",
                    ))
                }
                Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
            }
        }
        if let Some(cancel) = &self.repair_cancel {
            // Setup and execution both observe cloned receivers; cancellation must
            // remain visible after either waiter observes it.
            cancel.close();
            self.repair_status = Some(DiagnosticText::new(
                "Cancellation requested; accepted owner effects may continue or remain. Waiting for the observed outcome.",
            ));
        }
        if self.reviewed_action.is_some() {
            cx.emit(DoctorEvidenceUpdated);
        }
        cx.notify();
    }

    fn investigation_selection(
        &self,
        cx: &App,
    ) -> anyhow::Result<(
        ProjectPath,
        crate::doctor::DoctorInvestigationSelection,
        crate::doctor::DoctorReproductionEffects,
        u64,
    )> {
        let directory = self.review_directory(cx)?;
        let text = |field: &Entity<editor::Editor>| -> anyhow::Result<String> {
            let value = field.read(cx).text(cx).trim().to_owned();
            anyhow::ensure!(
                !value.is_empty() && value.len() <= 4096,
                "Fill each investigation field with at most 4096 bytes; exclude secrets and sensitive runtime data"
            );
            Ok(value)
        };
        let duration = text(&self.investigation.duration)?.parse::<u64>()?;
        anyhow::ensure!(
            (1..=300).contains(&duration),
            "Select a deadline between one second and five minutes"
        );
        let source = crate::doctor::comparison_source_identity(&self.project, &directory, cx)?;
        let source_revision = serde_json::to_string(&source)?;
        let symptom = text(&self.investigation.symptom)?;
        let target = text(&self.investigation.target)?;
        let workload = text(&self.investigation.workload)?;
        let configuration = text(&self.investigation.configuration)?;
        let cache = text(&self.investigation.cache)?;
        let selection = crate::doctor::DoctorInvestigationSelection {
            input_identity: crate::doctor::DoctorInvestigationInputIdentity::from_fields(&[
                &source_revision,
                &symptom,
                &target,
                &workload,
                &configuration,
                &cache,
            ])?,
            scope: self.investigation.scope,
            problem: self.investigation.problem,
            symptom: DiagnosticText::new(symptom),
            validation_target: DiagnosticText::new(target),
            workload: DiagnosticText::new(workload),
            configuration: DiagnosticText::new(configuration),
            source_revision: DiagnosticText::new(source_revision),
            cache_conditions: DiagnosticText::new(cache),
        };
        selection.validate()?;
        let effects = if self.investigation.development_only {
            crate::doctor::DoctorReproductionEffects::Development {
                file_writes: text(&self.investigation.writes)?,
                subprocesses: text(&self.investigation.subprocesses)?,
            }
        } else {
            crate::doctor::DoctorReproductionEffects::Unknown
        };
        Ok((directory, selection, effects, duration))
    }

    fn investigation_task_context(
        &self,
        directory: &ProjectPath,
        cx: &App,
    ) -> anyhow::Result<task::TaskContext> {
        let root = self
            .project
            .read(cx)
            .absolute_path(directory, cx)
            .ok_or_else(|| anyhow::anyhow!("Selected development root is unavailable"))?;
        let mut context = task::TaskContext {
            cwd: Some(root.clone()),
            ..Default::default()
        };
        context.task_variables.insert(
            task::VariableName::WorktreeRoot,
            root.to_string_lossy().into_owned(),
        );
        Ok(context)
    }

    fn load_investigation_targets(&mut self, cx: &mut Context<Self>) {
        if self.preparation.is_some() || self.repair_task.is_some() {
            return;
        }
        let operation = (|| {
            let directory = self.review_directory(cx)?;
            let inventory = self
                .project
                .read(cx)
                .task_store()
                .read(cx)
                .task_inventory()
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("Existing Task inventory is unavailable on this host")
                })?;
            let tasks = inventory
                .read(cx)
                .list_tasks(None, None, Some(directory.worktree_id), cx);
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            let debug = {
                let contexts = project::TaskContexts {
                    active_worktree_context: Some((
                        directory.worktree_id,
                        self.investigation_task_context(&directory, cx)?,
                    )),
                    ..Default::default()
                };
                inventory.update(cx, |inventory, cx| {
                    inventory.list_debug_scenarios(&contexts, Vec::new(), Vec::new(), false, cx)
                })
            };
            let generation = self.generation;
            Ok::<_, anyhow::Error>(cx.spawn(async move |view, cx| {
                let tasks = tasks.await;
                #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                let (_, debug) = debug.await;
                if let Err(error) = view.update(cx, |view, cx| {
                    view.preparation = None;
                    if generation != view.generation { return; }
                    view.investigation.tasks = tasks.into_iter().take(64).collect();
                    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                    { view.investigation.debug_scenarios = debug.into_iter().take(32).collect(); }
                    view.repair_status = Some(DiagnosticText::new("Loaded bounded existing Task/debug targets without execution or provisioning. Additional targets remain available in their normal surfaces."));
                    cx.notify();
                }) { log::debug!("Doctor target selection view closed: {error}"); }
            }))
        })();
        match operation {
            Ok(task) => self.preparation = Some(task),
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn review_investigation(
        &mut self,
        action: InvestigationAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.preparation.is_some() || self.repair_task.is_some() {
            return;
        }
        let preparation = (|| {
            let (directory, selection, effects, duration) = self.investigation_selection(cx)?;
            let observed = Self::now()?;
            let generation = self.generation;
            let operation = match action {
                InvestigationAction::Command | InvestigationAction::Task(_) => {
                    let (source, template) = match action {
                        InvestigationAction::Task(index) => self
                            .investigation
                            .tasks
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                anyhow::anyhow!("Selected Task is unavailable; reload targets")
                            })?,
                        _ => {
                            let command = self
                                .investigation
                                .command
                                .read(cx)
                                .text(cx)
                                .trim()
                                .to_owned();
                            anyhow::ensure!(
                                !command.is_empty()
                                    && command.len() <= 4096
                                    && !command.contains('\0'),
                                "Enter a valid bounded development command"
                            );
                            (
                                project::TaskSourceKind::UserInput,
                                task::TaskTemplate {
                                    label: format!(
                                        "Doctor: {}",
                                        selection.validation_target.as_str()
                                    ),
                                    command,
                                    ..Default::default()
                                },
                            )
                        }
                    };
                    let resolved = template.resolve_task(&source.to_id_base(), &self.investigation_task_context(&directory, cx)?).ok_or_else(|| anyhow::anyhow!("Existing Task resolution requires unavailable variables; use the normal Task surface to select a complete target"))?;
                    Task::ready(
                        crate::doctor::prepare_reproduction(
                            &self.project,
                            directory,
                            source,
                            resolved,
                            selection,
                            effects,
                            duration,
                            generation,
                            observed,
                            cx,
                        )
                        .map(OwnerRepair::Reproduction),
                    )
                }
                #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                InvestigationAction::Test {
                    provider,
                    node,
                    discovery,
                } => {
                    #[cfg(feature = "jvm-tools")]
                    if self
                        .project
                        .read(cx)
                        .doctor_observation(directory.worktree_id, cx)
                        .host
                        == DoctorExecutionHost::RemoteConnected
                        && provider.0 == project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID
                    {
                        return Ok(Task::ready(
                            crate::doctor::prepare_remote_test_inspection(
                                &self.project,
                                directory,
                                node,
                                discovery,
                                generation,
                                observed,
                                cx,
                            )
                            .map(|review| {
                                OwnerRepair::RemoteTestInspection {
                                    review,
                                    selection,
                                    effects,
                                    duration,
                                }
                            }),
                        ));
                    }
                    let operation = crate::doctor::prepare_test_reproduction(
                        self.project.clone(),
                        directory,
                        provider,
                        node,
                        discovery,
                        selection,
                        effects,
                        duration,
                        generation,
                        observed,
                        cx,
                    );
                    cx.spawn(async move |_, _| operation.await.map(OwnerRepair::Reproduction))
                }
                #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                InvestigationAction::Debug(index) => {
                    let (_, scenario) = self
                        .investigation
                        .debug_scenarios
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            anyhow::anyhow!("Selected debugger configuration is unavailable")
                        })?;
                    anyhow::ensure!(
                        scenario.build.is_none() && scenario.tcp_connection.is_none(),
                        "Doctor does not build, attach, download or connect while preparing a debugger review. Run a separately reviewed build Task and select a fully resolved local launch configuration"
                    );
                    let definition = dap::adapters::DebugTaskDefinition {
                        label: scenario.label,
                        adapter: dap::adapters::DebugAdapterName(scenario.adapter),
                        config: scenario.config,
                        tcp_connection: None,
                    };
                    let worktree = self
                        .project
                        .read(cx)
                        .worktree_for_id(directory.worktree_id, cx)
                        .ok_or_else(|| anyhow::anyhow!("Selected worktree closed"))?;
                    let launch = self.project.read(cx).dap_store().update(cx, |store, cx| {
                        store.prepare_diagnostic_launch(definition, worktree, cx)
                    });
                    let project = self.project.clone();
                    cx.spawn(async move |_, cx| {
                        let launch = launch.await?;
                        let preparation = cx.update(|cx| {
                            crate::doctor::prepare_debug(
                                project, directory, launch, selection, effects, duration,
                                generation, observed, cx,
                            )
                        });
                        preparation.await.map(OwnerRepair::Debug)
                    })
                }
                #[cfg(feature = "rust-tools")]
                InvestigationAction::Profile => {
                    let path = |editor: &Entity<editor::Editor>| -> anyhow::Result<ProjectPath> {
                        let text = editor.read(cx).text(cx);
                        anyhow::ensure!(
                            !text.is_empty() && text.len() <= 4096,
                            "Select an explicit contained profile artifact"
                        );
                        let path = self
                            .project
                            .read(cx)
                            .find_project_path(&text, cx)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Artifact is outside the selected project")
                            })?;
                        anyhow::ensure!(
                            path.worktree_id == directory.worktree_id,
                            "Artifact belongs to another selected worktree"
                        );
                        Ok(path)
                    };
                    let paths = [
                        path(&self.investigation.profile)?,
                        path(&self.investigation.symbols)?,
                    ];
                    Task::ready(
                        crate::doctor::prepare_profile(
                            &self.project,
                            directory,
                            paths,
                            selection,
                            duration,
                            generation,
                            observed,
                            cx,
                        )
                        .map(OwnerRepair::Profile),
                    )
                }
            };
            Ok::<_, anyhow::Error>(operation)
        })();
        match preparation {
            Ok(operation) => {
                let generation = self.generation;
                self.preparation = Some(cx.spawn_in(window, async move |view, cx| {
                    let result = operation.await;
                    if let Err(error) = view.update_in(cx, |view, window, cx| {
                        view.preparation = None;
                        if generation == view.generation {
                            view.accept_proposal(result, window, cx);
                        }
                    }) {
                        log::debug!("Doctor investigation review view closed: {error}");
                    }
                }));
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn investigation_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor::DoctorInvestigationScope as Scope;
        let busy =
            self.repair_task.is_some() || self.preparation.is_some() || self.collection.is_some();
        let mut controls = v_flex().gap_2().child(
            div()
                .debug_selector(|| "doctor-investigation-form".into())
                .child(
                    Button::new(
                        "doctor-investigation-form",
                        if self.investigation.open {
                            "Hide project/application investigation"
                        } else {
                            "Investigate a project or application…"
                        },
                    )
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.investigation.open = !this.investigation.open;
                        cx.notify();
                    })),
                ),
        );
        if !self.investigation.open {
            return controls.into_any_element();
        }
        controls = controls.child(Label::new("This investigation uses the section, worktree and execution host selected above. Each action is reviewed independently; this form grants no production, external-service, credential or unattended authority."));
        if self.investigation.scope == Scope::Product {
            return controls
                .child(Label::new(
                    "Use the existing product checks and reviewed repairs above.",
                ))
                .into_any_element();
        }
        controls = controls.child(h_flex().flex_wrap().gap_2().children(self.investigation.problems().into_iter().enumerate().map(|(index, (problem, label))| Button::new(("doctor-investigation-problem", index), label).toggle_state(self.investigation.problem == problem).disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| { this.cancel_repair(cx); this.investigation.problem = problem; cx.notify(); })))))
            .child(Label::new("Inputs and diagnostics are bounded and redacted. Do not enter secrets, request payloads or environment dumps. Source identity uses indexed nonignored metadata and dirty buffers; ignored/external inputs remain unverified."));
        if !busy {
            for (id, label, field) in [
                ("symptom", "Problem / symptom", &self.investigation.symptom),
                (
                    "target",
                    "Relevant validation target",
                    &self.investigation.target,
                ),
                (
                    "workload",
                    "Workload / expected behavior",
                    &self.investigation.workload,
                ),
                (
                    "configuration",
                    "Toolchain / build / runtime configuration",
                    &self.investigation.configuration,
                ),
                (
                    "cache",
                    "Cache / warm-up conditions",
                    &self.investigation.cache,
                ),
                (
                    "duration",
                    "Capture / execution deadline (seconds; maximum 300)",
                    &self.investigation.duration,
                ),
                (
                    "writes",
                    "Expected development writes",
                    &self.investigation.writes,
                ),
                (
                    "subprocesses",
                    "Expected subprocesses",
                    &self.investigation.subprocesses,
                ),
                (
                    "command",
                    "Exact reproduction command through existing Tasks",
                    &self.investigation.command,
                ),
            ] {
                controls = controls.child(Label::new(label)).child(
                    div()
                        .id(format!("doctor-investigation-{id}"))
                        .debug_selector(move || format!("doctor-investigation-{id}"))
                        .aria_label(label)
                        .child(field.clone()),
                );
            }
        }
        controls = controls.child(Button::new("doctor-investigation-development-only", "Confirm disclosed development-only effects; no external services or sensitive capture").toggle_state(self.investigation.development_only).disabled(busy).tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| { this.cancel_repair(cx); this.investigation.development_only = !this.investigation.development_only; cx.notify(); })).map(|button| div().debug_selector(|| "doctor-investigation-development-only".into()).child(button)))
            .child(Label::new("Default limit is 60 seconds; maximum 5 minutes and 10 MiB of bounded evidence. Existing Task/debug processes use normal host permissions, not an OS resource or network sandbox. Unknown effects are blocked. No automatic save, replay or inferred permission."))
            .child(div().debug_selector(|| "doctor-investigation-review-command".into()).child(Button::new("doctor-investigation-review-command", "Review exact reproduction command").disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.review_investigation(InvestigationAction::Command, window, cx)))))
            .child(Button::new("doctor-investigation-load-targets", "Load existing Tasks / debugger configurations (no execution)").disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| this.load_investigation_targets(cx))));
        for (index, (_, template)) in self.investigation.tasks.iter().enumerate() {
            controls = controls.child(
                Button::new(
                    ("doctor-investigation-task", index),
                    format!(
                        "Review Task: {}",
                        DiagnosticText::new(&template.label).as_str()
                    ),
                )
                .disabled(busy)
                .tab_index(0_isize)
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.review_investigation(InvestigationAction::Task(index), window, cx)
                })),
            );
        }
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        {
            let store = self.project.read(cx).structured_execution_store().read(cx);
            let mut choices = Vec::new();
            for provider in store.state().providers() {
                #[cfg(feature = "jvm-tools")]
                let jvm =
                    provider.provider_id.0 == project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID;
                #[cfg(not(feature = "jvm-tools"))]
                let jvm = false;
                let stale_result = jvm
                    && provider.status
                        == project::structured_execution::StructuredProviderStatus::Stale;
                if provider.partial
                    || !(provider.status
                        == project::structured_execution::StructuredProviderStatus::Current
                        || stale_result)
                {
                    continue;
                }
                for node in &provider.nodes {
                    if choices.len() >= 64 {
                        break;
                    }
                    if jvm && node.kind != project::structured_execution::StructuredNodeKind::Case {
                        continue;
                    }
                    if provider
                        .project_path_for_node(&node.id)
                        .is_some_and(|path| Some(path.worktree_id) == self.selected_worktree)
                    {
                        choices.push((
                            provider.provider_id.clone(),
                            node.id.clone(),
                            provider.discovery_generation,
                            DiagnosticText::new(if stale_result { format!("{} (previous report is stale; execution requires source verification)", node.label) } else { node.label.clone() }),
                        ));
                    }
                }
            }
            if choices.is_empty() {
                controls = controls.child(Label::new("No current selected-worktree test provider. Discover/select tests in the existing Tests surface; missing or partial discovery is not a successful check."));
            }
            for (index, (provider, node, discovery, label)) in choices.into_iter().enumerate() {
                controls = controls.child(
                    div()
                        .debug_selector(move || format!("doctor-investigation-test-{index}"))
                        .child(
                            Button::new(
                                ("doctor-investigation-test", index),
                                format!("Review test: {}", label.as_str()),
                            )
                            .disabled(busy)
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                move |this, _, window, cx| {
                                    this.review_investigation(
                                        InvestigationAction::Test {
                                            provider: provider.clone(),
                                            node: node.clone(),
                                            discovery,
                                        },
                                        window,
                                        cx,
                                    )
                                },
                            )),
                        ),
                );
            }
            for (index, (_, scenario)) in self.investigation.debug_scenarios.iter().enumerate() {
                controls = controls.child(
                    Button::new(
                        ("doctor-investigation-debug", index),
                        format!(
                            "Review metadata-only debugger: {}",
                            DiagnosticText::new(&scenario.label).as_str()
                        ),
                    )
                    .disabled(busy)
                    .tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.review_investigation(InvestigationAction::Debug(index), window, cx)
                    })),
                );
            }
        }
        #[cfg(feature = "rust-tools")]
        {
            controls = controls.child(Label::new("Local macOS Samply 0.13.1 artifact inspection only. Import is not a new capture, CPU time or memory measurement. JVM JFR / heap / core / remote runtime capture are unsupported."));
            if !busy {
                controls = controls
                    .child(Label::new("Contained profile .json.gz path"))
                    .child(self.investigation.profile.clone())
                    .child(Label::new("Explicit .syms.json sidecar path"))
                    .child(self.investigation.symbols.clone());
            }
            controls = controls.child(
                Button::new(
                    "doctor-investigation-review-profile",
                    "Review selected profile inspection",
                )
                .disabled(busy || self.report.is_none())
                .tab_index(0_isize)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.review_investigation(InvestigationAction::Profile, window, cx)
                })),
            );
        }
        controls = controls.child(
            Button::new(
                "doctor-investigation-restricted-data",
                "Report insufficient evidence: this investigation requires excluded runtime data",
            )
            .disabled(busy || self.report.is_none())
            .tab_index(0_isize)
            .on_click(cx.listener(|this, _, _, cx| this.record_restricted_investigation(cx))),
        );
        controls.child(Label::new("Command exit alone cannot verify a fix. Selected fresh JVM JUnit assertions can support functional verification; current Rust execution status alone remains incomplete. Unsupported memory/concurrency evidence must not become a confirmed cause or successful repair." )).into_any_element()
    }

    fn select_comparison_series(&mut self, baseline: bool, append: bool, cx: &mut Context<Self>) {
        let result = (|| {
            anyhow::ensure!(
                self.repair_task.is_none(),
                "Wait for the selected owner result"
            );
            let current = self.current_comparison.as_ref().ok_or_else(|| anyhow::anyhow!("No sealed live reproduction result is available; historical reports cannot become current verification"))?;
            let selected = if baseline {
                &mut self.baseline_comparison
            } else {
                &mut self.verification_comparison
            };
            if append {
                selected
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Select the first result in this series"))?
                    .combine_repetition(current)?;
            } else {
                *selected = Some(current.clone());
            }
            self.comparison_form.clear_selections();
            self.saved_generation = None;
            Ok::<_, anyhow::Error>(())
        })();
        match result {
            Ok(()) => {
                self.repair_status = Some(DiagnosticText::new(
                    "Selected live result series. Each additional run requires its own exact action review; grouping does not execute or replay work.",
                ))
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn compare_investigation(&mut self, cx: &mut Context<Self>) {
        let result = (|| {
            anyhow::ensure!(
                self.repair_task.is_none(),
                "Wait for the current action before comparing"
            );
            let before = self
                .baseline_comparison
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Select an observed baseline"))?;
            let after = self
                .verification_comparison
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Select relevant after-change results"))?;
            let description = self
                .comparison_form
                .change
                .read(cx)
                .text(cx)
                .trim()
                .to_owned();
            anyhow::ensure!(
                !description.is_empty() && description.len() <= 4096,
                "Describe the explicitly reviewed change with at most 4096 bytes; do not include source or secrets"
            );
            let performance = if self.comparison_form.performance {
                Some(crate::doctor::DoctorPerformanceCriterion {
                    metric: DiagnosticText::new("reviewed Task end-to-end wall time"),
                    unit: DiagnosticText::new("ms"),
                    protocol: DiagnosticText::new(crate::doctor::DOCTOR_TASK_TIMING_PROTOCOL),
                    minimum_repetitions: self
                        .comparison_form
                        .repetitions
                        .read(cx)
                        .text(cx)
                        .trim()
                        .parse()?,
                    maximum_after: DiagnosticText::new(
                        self.comparison_form.maximum.read(cx).text(cx),
                    ),
                    minimum_improvement_percent: DiagnosticText::new(
                        self.comparison_form.improvement.read(cx).text(cx),
                    ),
                })
            } else {
                None
            };
            #[cfg(feature = "agentic")]
            let session_id = self.change_session.clone().map(DiagnosticText::new);
            #[cfg(not(feature = "agentic"))]
            let session_id = None;
            let request = crate::doctor::DoctorComparisonRequest {
                baseline_functional: self.comparison_form.before_functional.ok_or_else(|| {
                    anyhow::anyhow!("Explicitly select relevant baseline functional evidence")
                })?,
                verification_functional: self.comparison_form.after_functional.ok_or_else(
                    || {
                        anyhow::anyhow!(
                            "Explicitly select relevant verification functional evidence"
                        )
                    },
                )?,
                change: crate::doctor::DoctorReviewedChange {
                    before_input: before.selection().input_identity.clone(),
                    after_input: after.selection().input_identity.clone(),
                    description: DiagnosticText::new(description),
                    session_id,
                },
                performance,
                baseline_measurements: self.comparison_form.before_measurements.clone(),
                verification_measurements: self.comparison_form.after_measurements.clone(),
                baseline_protocol: DiagnosticText::new(crate::doctor::DOCTOR_TASK_TIMING_PROTOCOL),
                verification_protocol: DiagnosticText::new(
                    crate::doctor::DOCTOR_TASK_TIMING_PROTOCOL,
                ),
            };
            let record = before.compare_current(after, request, Self::now()?, &self.project, cx)?;
            let report = self
                .report
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Current report is unavailable"))?;
            let mut proposed = report.clone();
            proposed.clear_comparison();
            proposed.set_investigation(after.summary().clone())?;
            proposed.set_comparison(record)?;
            *report = proposed;
            self.saved_generation = None;
            self.history_change = None;
            Ok::<_, anyhow::Error>(())
        })();
        if let Err(error) = result {
            self.error = Some(DiagnosticText::new(error.to_string()));
        }
        cx.notify();
    }

    fn record_restricted_investigation(&mut self, cx: &mut Context<Self>) {
        let result = (|| {
            anyhow::ensure!(
                self.repair_task.is_none(),
                "Wait for the current owner action"
            );
            let (directory, selection, _, _) = self.investigation_selection(cx)?;
            anyhow::ensure!(
                selection.scope == crate::doctor::DoctorInvestigationScope::Application,
                "Select the application problem and relevant target"
            );
            let now = Self::now()?;
            let binding = crate::doctor::project_repair_binding(
                &self.project,
                directory,
                self.generation,
                cx,
            )?;
            let report = self
                .report
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Collect current scope first"))?;
            let mut investigation = crate::doctor::DoctorInvestigation::new(
                selection.clone(),
                report.scope.clone(),
                binding.clone(),
                now,
            )?;
            investigation.record(&binding, &selection, crate::doctor::DoctorEvidenceInput {
                owner: crate::doctor::DoctorEvidenceOwner::Project, owner_reference: DiagnosticText::new("doctor.approved-data-boundary"), classification: crate::doctor::DoctorEvidenceClass::Observed,
                observation: crate::doctor::DoctorEvidenceObservation::InsufficientEvidence,
                detail: DiagnosticText::new("Observed collection boundary: requested heap/core/JFR/environment/request-payload or sensitive-runtime data is outside the approved initial scope. No excluded artifact was read, uploaded or retained. Cause and repair remain unverified; selected bounded development/test/debug evidence may be collected separately. Broader collection requires a separate decision."), observed_at: now, historical: false,
            })?;
            report.clear_comparison();
            report.set_investigation(investigation.summary().clone())?;
            self.current_comparison = None;
            self.saved_generation = None;
            self.history_change = None;
            #[cfg(feature = "agentic")]
            {
                self.native_evidence.clear();
                self.native_investigation_evidence.clear();
            }
            Ok::<_, anyhow::Error>(())
        })();
        if let Err(error) = result {
            self.error = Some(DiagnosticText::new(error.to_string()));
        }
        cx.notify();
    }

    #[cfg(feature = "rust-tools")]
    fn open_investigation_profile(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(result) = self.profile_result.take() else {
            return;
        };
        match self.workspace.update(cx, |workspace, cx| {
            language_tools::profile_view::open_diagnostic_profile_result(
                workspace, result, window, cx,
            )
        }) {
            Ok(Ok(_)) => {}
            Ok(Err(error)) | Err(error) => {
                self.error = Some(DiagnosticText::new(format!(
                    "Profile view handoff failed: {error}; no runtime cause or repair was verified. Review a fresh import if authority changed."
                )))
            }
        }
        cx.notify();
    }

    fn authorize_result_navigation(&self, cx: &App) -> anyhow::Result<()> {
        let directory = self.review_directory(cx)?;
        crate::doctor::project_repair_binding(&self.project, directory, self.generation, cx)?;
        anyhow::ensure!(
            self.workspace
                .read_with(cx, |workspace, _| workspace.project().entity_id())?
                == self.project.entity_id(),
            "The result belongs to a different project workspace"
        );
        Ok(())
    }

    fn investigation_summary_element(
        summary: &crate::doctor::DoctorInvestigationSummary,
        historical: bool,
    ) -> gpui::AnyElement {
        v_flex().debug_selector(|| "doctor-investigation-evidence-pane".into()).gap_2().child(Label::new(format!("{} evidence — {:?} / {:?}; target: {}", if historical { "Historical" } else { "Recorded selected" }, summary.selection.scope, summary.selection.problem, summary.selection.validation_target.as_str())))
            .child(Label::new(format!("Workload: {}; configuration: {}; cache: {}; source metadata identity: {}", summary.selection.workload.as_str(), summary.selection.configuration.as_str(), summary.selection.cache_conditions.as_str(), summary.selection.source_revision.as_str())))
            .children(summary.evidence().iter().enumerate().map(|(index, row)| v_flex().id(("doctor-observation", index)).p_2().gap_1().child(Label::new(format!("{:?} · {:?} · {:?} · {}{}", row.classification, row.owner, row.observation, Self::timestamp(row.observed_at), if historical || row.historical { " · historical; not current verification" } else { "" }))).child(Label::new(row.detail.as_str().to_owned())).child(Label::new(format!("Owner reference: {}; reproduced failure: {}", row.owner_reference.as_str(), !historical && row.reproduced_failure())))))
            .child(Label::new("User reports and model hypotheses are not observations. A command exit or successful launch is not a verified fix. Coverage, supported collection and relevant functional evidence constrain every conclusion." )).into_any_element()
    }

    fn comparison_result_element(
        record: &crate::doctor::DoctorComparisonRecord,
        historical: bool,
    ) -> gpui::AnyElement {
        v_flex().gap_2().child(Label::new(format!("{} comparison recorded {}: {:?}", if historical { "Historical" } else { "Selected-target" }, Self::timestamp(record.observed_at), record.outcome)))
            .child(Label::new(record.change.description.as_str().to_owned()))
            .child(Label::new(format!("Target: {}; baseline source: {}; verification source: {}", record.verification_selection.validation_target.as_str(), record.baseline_selection.source_revision.as_str(), record.verification_selection.source_revision.as_str())))
            .when_some(record.criterion.as_ref(), |element, criterion| element.child(Label::new(format!("Criterion: {} {}; minimum {} repetitions; maximum after {}; minimum improvement {}%; protocol {}", criterion.metric.as_str(), criterion.unit.as_str(), criterion.minimum_repetitions, criterion.maximum_after.as_str(), criterion.minimum_improvement_percent.as_str(), criterion.protocol.as_str()))))
            .child(Label::new(format!("Baseline median: {}; after median: {}", record.baseline_median.as_ref().map_or("unavailable", DiagnosticText::as_str), record.verification_median.as_ref().map_or("unavailable", DiagnosticText::as_str))))
            .child(Label::new(record.detail.as_str().to_owned()))
            .child(Label::new("This records the selected evidence at that time. It is not a new rerun, proof of present project health, exploitability, general correctness, causation or race freedom." )).into_any_element()
    }

    fn investigation_evidence_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        let mut controls = v_flex().gap_2();
        if let Some(summary) = self.report.as_ref().and_then(DoctorReport::investigation) {
            controls = controls.child(Self::investigation_summary_element(summary, false));
            #[cfg(feature = "agentic")]
            {
                let target = summary.selection.validation_target.as_str().to_owned();
                controls = controls.child(
                    Button::new(
                        "doctor-investigation-native-select",
                        "Select these observations for a reviewed native repair proposal",
                    )
                    .disabled(busy)
                    .tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.advanced_open = true;
                        this.investigation.open = false;
                        this.native_repair_open = true;
                        this.native_target.update(cx, |editor, cx| {
                            editor.set_text(target.as_str(), window, cx)
                        });
                        this.native_evidence.clear();
                        this.native_investigation_evidence.clear();
                        this.scroll_handle
                            .set_offset(gpui::point(gpui::px(0.), gpui::px(0.)));
                        cx.notify();
                    }))
                    .map(|button| {
                        div()
                            .debug_selector(|| "doctor-investigation-native-select".into())
                            .child(button)
                    }),
                );
            }
        }
        if let Some(handle) = self.task_result.clone() {
            controls = controls.child(div().debug_selector(|| "doctor-investigation-open-task".into()).child(Button::new("doctor-investigation-open-task", "Inspect this Task in its existing terminal").disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, window, cx| {
                if let Err(error) = this.authorize_result_navigation(cx) { this.error = Some(DiagnosticText::new(error.to_string())); cx.notify(); return; }
                if !handle.reveal_terminal(window, cx) { this.error = Some(DiagnosticText::new("The original Task terminal is no longer available; opening evidence never reruns it")); cx.notify(); }
            }))));
        }
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        if let Some(session) = self.debug_result.clone() {
            controls = controls.child(div().debug_selector(|| "doctor-investigation-open-debug".into()).child(Button::new("doctor-investigation-open-debug", "Inspect this session in the existing Debugger").disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, window, cx| {
                if let Err(error) = this.authorize_result_navigation(cx) { this.error = Some(DiagnosticText::new(error.to_string())); cx.notify(); return; }
                let provider = this.workspace.read_with(cx, |workspace, _| workspace.debugger_provider()).ok().flatten();
                if !provider.is_some_and(|provider| provider.reveal_diagnostic_session(&session, window, cx)) { this.error = Some(DiagnosticText::new("The original debugger session is no longer available; opening evidence never launches another session")); cx.notify(); }
            }))));
        }
        if let Some(record) = self.report.as_ref().and_then(DoctorReport::comparison) {
            controls = controls.child(Self::comparison_result_element(record, false));
        }
        #[cfg(feature = "rust-tools")]
        if self.profile_result.is_some() {
            controls = controls.child(
                Button::new(
                    "doctor-investigation-open-profile",
                    "Inspect bounded result in existing Profile view",
                )
                .disabled(busy)
                .tab_index(0_isize)
                .on_click(
                    cx.listener(|this, _, window, cx| this.open_investigation_profile(window, cx)),
                ),
            );
        }
        if self.current_comparison.is_none() {
            return controls.into_any_element();
        }
        controls = controls.child(Label::new("Retain live baseline / verification series. Each repetition is a separately approved existing Task/test action. Historical reports cannot be promoted into current proof."))
            .child(h_flex().flex_wrap().gap_2().children([(true, false, "Use latest result as baseline"), (true, true, "Add latest repetition to baseline"), (false, false, "Use latest result as verification"), (false, true, "Add latest repetition to verification")].into_iter().enumerate().map(|(index, (baseline, append, label))| div().debug_selector(move || format!("doctor-investigation-series-{index}")).child(Button::new(("doctor-investigation-series", index), label).disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| this.select_comparison_series(baseline, append, cx)))))));
        for (baseline, snapshot) in [
            (true, self.baseline_comparison.as_ref()),
            (false, self.verification_comparison.as_ref()),
        ] {
            let Some(snapshot) = snapshot else {
                continue;
            };
            controls = controls.child(Label::new(if baseline { "Baseline: explicitly select functional evidence and any elapsed samples" } else { "Verification: explicitly select relevant functional evidence and any elapsed samples" }));
            for (index, row) in snapshot.evidence().iter().enumerate() {
                let identifier = row.id;
                let selected = if baseline {
                    self.comparison_form.before_functional == Some(identifier)
                } else {
                    self.comparison_form.after_functional == Some(identifier)
                };
                controls = controls.child(
                    div()
                        .debug_selector(move || {
                            format!(
                                "doctor-comparison-{}-functional-{index}",
                                if baseline { "before" } else { "after" }
                            )
                        })
                        .child(
                            Button::new(
                                (
                                    if baseline {
                                        "doctor-comparison-before-functional"
                                    } else {
                                        "doctor-comparison-after-functional"
                                    },
                                    index,
                                ),
                                format!(
                                    "Functional selection: {:?} ({:?}, {})",
                                    row.observation,
                                    row.owner,
                                    row.owner_reference.as_str()
                                ),
                            )
                            .toggle_state(selected)
                            .disabled(busy)
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                move |this, _, _, cx| {
                                    if baseline {
                                        this.comparison_form.before_functional = Some(identifier);
                                    } else {
                                        this.comparison_form.after_functional = Some(identifier);
                                    }
                                    cx.notify();
                                },
                            )),
                        ),
                );
                if matches!(
                    row.observation,
                    crate::doctor::DoctorEvidenceObservation::AggregateCounter { .. }
                ) {
                    let selected = if baseline {
                        self.comparison_form
                            .before_measurements
                            .contains(&identifier)
                    } else {
                        self.comparison_form
                            .after_measurements
                            .contains(&identifier)
                    };
                    controls = controls.child(
                        Button::new(
                            (
                                if baseline {
                                    "doctor-comparison-before-sample"
                                } else {
                                    "doctor-comparison-after-sample"
                                },
                                index,
                            ),
                            "Select elapsed measurement",
                        )
                        .toggle_state(selected)
                        .disabled(busy)
                        .tab_index(0_isize)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let samples = if baseline {
                                &mut this.comparison_form.before_measurements
                            } else {
                                &mut this.comparison_form.after_measurements
                            };
                            if samples.contains(&identifier) {
                                samples.retain(|sample| *sample != identifier);
                            } else if samples.len() < 32 {
                                samples.push(identifier);
                            } else {
                                this.error = Some(DiagnosticText::new(
                                    "Select at most thirty-two measurements",
                                ));
                            }
                            cx.notify();
                        })),
                    );
                }
            }
        }
        if !busy {
            controls = controls
                .child(Label::new(
                    "Reviewed source/configuration/build change description",
                ))
                .child(self.comparison_form.change.clone())
                .child(
                    Button::new(
                        "doctor-comparison-performance",
                        "Include explicit repeated Task elapsed criterion",
                    )
                    .toggle_state(self.comparison_form.performance)
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.comparison_form.performance = !this.comparison_form.performance;
                        cx.notify();
                    })),
                );
            if self.comparison_form.performance {
                controls = controls.child(Label::new("Reviewed Task end-to-end monotonic elapsed, including admission/launch/result collection; milliseconds. Same selected workload, host, source, execution configuration and cache protocol. This is not CPU/memory or statistical significance."))
                    .child(Label::new("Minimum repetitions (2–32)")).child(self.comparison_form.repetitions.clone()).child(Label::new("Maximum acceptable after median (ms)")).child(self.comparison_form.maximum.clone()).child(Label::new("Minimum improvement (%)")).child(self.comparison_form.improvement.clone());
            }
        }
        controls
            .child(
                div()
                    .debug_selector(|| "doctor-investigation-compare".into())
                    .child(
                        Button::new(
                            "doctor-investigation-compare",
                            "Evaluate selected relevant evidence (no execution)",
                        )
                        .disabled(
                            busy || self.baseline_comparison.is_none()
                                || self.verification_comparison.is_none(),
                        )
                        .tab_index(0_isize)
                        .on_click(cx.listener(|this, _, _, cx| this.compare_investigation(cx))),
                    ),
            )
            .into_any_element()
    }

    fn approve_repair(&mut self, cx: &mut Context<Self>) {
        if self.repair_task.is_some() {
            cx.notify();
            return;
        }
        let preparation = (|| {
            let directory = self.review_directory(cx)?;
            let binding = crate::doctor::project_repair_binding(
                &self.project,
                directory.clone(),
                self.generation,
                cx,
            )?;
            let history = crate::doctor::DoctorHistoryStore::for_app(cx)?;
            let now = Self::now()?;
            let scope = self
                .report
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Collected report is unavailable"))?
                .scope
                .clone();
            Ok::<_, anyhow::Error>((directory, binding, history, now, scope))
        })();
        let (directory, binding, history, now, scope) = match preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                self.error = Some(DiagnosticText::new(error.to_string()));
                cx.notify();
                return;
            }
        };
        #[cfg(not(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools")))]
        let _ = &directory;
        let Some(mut review) = self.proposal.take() else {
            self.error = Some(DiagnosticText::new(
                "No current repair proposal is awaiting approval",
            ));
            cx.notify();
            return;
        };
        if let Err(error) = review.approve(&binding) {
            self.repair_status = Some(DiagnosticText::new(
                "Proposal rejected before execution; collect and review current inputs",
            ));
            self.error = Some(DiagnosticText::new(error.to_string()));
            cx.notify();
            return;
        }
        let mut report = DoctorReport::new(scope, self.generation, now);
        if let Err(error) = report.push(review.before().clone()) {
            self.error = Some(DiagnosticText::new(error.to_string()));
            cx.notify();
            return;
        }
        self.reviewed_action = Some(review.conversation_identity());
        self.owner_result = None;
        let product = Self::product();
        let (cancel, cancelled) = async_channel::bounded(1);
        #[cfg(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools"))]
        let repair_cancel = cancel.clone();
        #[cfg(not(any(feature = "agentic", feature = "rust-tools", feature = "jvm-tools")))]
        let repair_cancel = cancel;
        self.repair_cancel = Some(repair_cancel);
        self.error = None;
        self.repair_status = Some(DiagnosticText::new(
            "Recording exact approval, then running the existing owner action and its relevant recheck",
        ));
        let generation = self.generation;
        self.restart_active = match &review {
            OwnerRepair::ApplicationRestart(_) => Some(RestartCoordination::Application),
            OwnerRepair::RemoteRestart(_) => Some(RestartCoordination::Remote),
            _ => None,
        };
        #[cfg(feature = "agentic")]
        {
            if self.native_session.is_some() {
                self.change_session = self.native_session.clone();
            }
            self.native_session = None;
        }
        let operation = match review {
            OwnerRepair::Reproduction(review) => {
                #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                let selected_project = self.project.clone();
                let operation = crate::doctor::run_reproduction(
                    review,
                    self.project.clone(),
                    self.workspace.clone(),
                    self.window_handle,
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, cx| {
                    #[cfg(not(any(feature = "rust-tools", feature = "jvm-tools")))]
                    let _ = cx;
                    let result = operation.await?;
                    let mut observed =
                        OwnerRepairResult::observed(OwnerRepair::Reproduction(result.review));
                    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                    if let Some(collected) = result.dependency_report {
                        let root = collected.binding().directory().clone();
                        let format = collected.format().clone();
                        let admission = match format {
                            #[cfg(feature = "rust-tools")]
                            crate::doctor_dependencies::DoctorDependencyReportFormat::CargoMetadata => {
                                cx.update(|cx| collected.admit_cargo(selected_project.clone(), cx)).await.map(|_| ())
                            }
                            #[cfg(feature = "jvm-tools")]
                            crate::doctor_dependencies::DoctorDependencyReportFormat::MavenSelectedTree | crate::doctor_dependencies::DoctorDependencyReportFormat::GradleResolution { .. } => {
                                cx.update(|cx| collected.admit_jvm(selected_project.clone(), cx)).await.map(|_| ())
                            }
                            #[allow(unreachable_patterns)]
                            _ => Err(anyhow::anyhow!("Selected dependency ecosystem is disabled")),
                        };
                        match admission {
                            Ok(()) => {
                                observed.projection = Some(cx.update(|cx| DependencySecurityForm::project(&selected_project, root, &format, generation, Self::now()?, cx))?);
                            }
                            Err(error) => observed.cleanup_error = Some(DiagnosticText::new(format!("Reached Task effects remain; graph admission is incomplete: {error}"))),
                        }
                    }
                    observed.investigation = Some(result.comparison.summary().clone());
                    observed.comparison = Some(result.comparison);
                    observed.task = result.task;
                    Ok(observed)
                })
            }
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            OwnerRepair::Debug(review) => {
                let selection = review.action().selection().clone();
                let selected_binding = binding.clone();
                let selected_scope = report.scope.clone();
                let operation = crate::doctor::run_debug(
                    review,
                    self.project.clone(),
                    self.workspace.clone(),
                    self.window_handle,
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    let result = operation.await?;
                    let mut investigation = crate::doctor::DoctorInvestigation::new(
                        selection.clone(),
                        selected_scope,
                        selected_binding.clone(),
                        now,
                    )?;
                    investigation.record(&selected_binding, &selection, result.evidence)?;
                    let mut observed =
                        OwnerRepairResult::observed(OwnerRepair::Debug(result.review));
                    observed.debug_session = result.session;
                    observed.investigation = Some(investigation.summary().clone());
                    Ok(observed)
                })
            }
            #[cfg(feature = "rust-tools")]
            OwnerRepair::Profile(review) => {
                let selection = review.action().selection().clone();
                let selected_binding = binding.clone();
                let selected_scope = report.scope.clone();
                let operation = crate::doctor::run_profile(
                    review,
                    self.project.clone(),
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    let result = operation.await?;
                    let mut investigation = crate::doctor::DoctorInvestigation::new(
                        selection.clone(),
                        selected_scope,
                        selected_binding.clone(),
                        now,
                    )?;
                    investigation.record(&selected_binding, &selection, result.evidence)?;
                    let mut observed =
                        OwnerRepairResult::observed(OwnerRepair::Profile(result.review));
                    observed.investigation = Some(investigation.summary().clone());
                    observed.profile = result.profile;
                    Ok(observed)
                })
            }
            #[cfg(feature = "jvm-tools")]
            OwnerRepair::RemoteTestInspection {
                review,
                selection,
                effects,
                duration,
            } => {
                let project = self.project.clone();
                let operation = crate::doctor::inspect_remote_test(
                    review,
                    project.clone(),
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, cx| {
                    let (review, inspected) = operation.await?;
                    let mut result =
                        OwnerRepairResult::observed(OwnerRepair::RemoteTestInspection {
                            review,
                            selection: selection.clone(),
                            effects: effects.clone(),
                            duration,
                        });
                    if let Some(inspected) = inspected {
                        result.proposal = Some(
                            cx.update(|cx| {
                                crate::doctor::prepare_inspected_remote_test_reproduction(
                                    inspected,
                                    &project,
                                    selection,
                                    effects,
                                    duration,
                                    Self::now()?,
                                    cx,
                                )
                            })
                            .map(OwnerRepair::Reproduction)?,
                        );
                    }
                    Ok(result)
                })
            }
            #[cfg(feature = "agentic")]
            OwnerRepair::Native(mut review) => {
                let project = self.project.clone();
                let workspace = self.workspace.clone();
                let window_handle = self.window_handle;
                let sender = cx.weak_entity();
                cx.spawn(async move |_, cx| {
                    let binding = cx.update(|cx| crate::doctor::project_repair_binding(&project, directory.clone(), generation, cx))?;
                    cx.update(|cx| review.action().revalidate(cx))?;
                    review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
                    let validation = cx.update(|cx| {
                        review.action().revalidate(cx)?;
                        review.revalidate(&crate::doctor::project_repair_binding(&project, directory.clone(), generation, cx)?)?;
                        Ok::<_, anyhow::Error>(())
                    });
                    let outcome = if let Err(error) = validation {
                        Err(error)
                    } else {
                        let (completion, response) = futures::channel::oneshot::channel();
                        let request = std::rc::Rc::new(crate::doctor::DoctorNativeRepairRequest::new(sender.clone(), workspace, window_handle, project, directory,
                            review.action().clone(), binding, history.clone(), product, review.receipt().ok_or_else(|| anyhow::anyhow!("Native repair requires its durable receipt"))?, cancelled.clone(), completion));
                        sender.update(cx, |_, cx| cx.emit(request))?;
                        response.await.unwrap_or_else(|_| Err(anyhow::anyhow!("Native repair desktop host is unavailable; no successful handoff was observed")))
                    };
                    let (session, operation, detail) = match outcome {
                        Ok(result) => (result.session_id, result.outcome, result.detail),
                        Err(error) => (None, crate::doctor::DoctorRepairOperationOutcome::Failed, DiagnosticText::new(error.to_string())),
                    };
                    *review.action().session.borrow_mut() = session.clone();
                    review.operation_finished(operation, detail)?;
                    if review.phase() == crate::doctor::DoctorRepairPhase::Rechecking {
                        review.recheck_unavailable(DiagnosticText::new("Native repair handoff did not establish a relevant independently observed recheck; inspect the session and review the selected validation target"))?;
                    }
                    let observed = DoctorView::now()?;
                    review.persist_outcome(product, &history, session.map(DiagnosticText::new), observed).await?;
                    Ok(OwnerRepairResult::observed(OwnerRepair::Native(review)))
                })
            }
            OwnerRepair::ApplicationRestart(review) => {
                let operation = crate::doctor::launch_application_restart(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation.await.map(|review| {
                        OwnerRepairResult::observed(OwnerRepair::ApplicationRestart(review))
                    })
                })
            }
            OwnerRepair::RemoteRestart(review) => {
                use futures::FutureExt as _;
                let reopen_cancelled = cancelled.clone();
                let previous_project = self.project.clone();
                let history_after = history.clone();
                let operation = crate::doctor::launch_remote_restart(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, cx| {
                    let (mut review, replacement) = operation.await?;
                    if let Some(replacement) = replacement.filter(|_| review.phase() == crate::doctor::DoctorRepairPhase::Incomplete) {
                        let expected = review.action().affected_workspaces().to_vec();
                        let requesting_item = review.action().requesting_item();
                        let cancellation = async move { match reopen_cancelled.recv().await { Ok(()) | Err(_) => {} } }.boxed().shared();
                        let reopening = cx.update(|cx| workspace::doctor_reopen_remote_workspaces(previous_project, replacement, expected, requesting_item, Doctor.boxed_clone(), cancellation, cx));
                        let detail = match reopening.await {
                            Ok(count) => format!("Observed fresh connection and reopened {count} workspace(s) with new Project stores through existing owners. No buffered request replay. Relevant problem checks remain unverified; collect them in the reopened Doctor."),
                            Err(error) => format!("Server restart/reopening is incomplete: {error}. A stopped host, partial fresh workspace or idle replacement may remain. No automatic replay or original-problem verification."),
                        };
                        review.note_project_reopening(DiagnosticText::new(detail))?;
                        review.persist_outcome(product, &history_after, None, Self::now()?).await?;
                    }
                    Ok(OwnerRepairResult::observed(OwnerRepair::RemoteRestart(review)))
                })
            }
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            OwnerRepair::Dependency(review) => {
                let operation = crate::doctor::launch_reviewed_dependency_probe(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation
                        .await
                        .map(|review| OwnerRepairResult::observed(OwnerRepair::Dependency(review)))
                })
            }
            #[cfg(feature = "agentic")]
            OwnerRepair::Github(review) => {
                let authority = review.action().clone();
                let (updates, progress) = async_channel::bounded(32);
                let operation = crate::doctor_dependencies::launch_github_evidence(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    updates,
                    cx,
                );
                cx.spawn(async move |this, cx| {
                    let progress_task = cx.spawn(async move |cx| {
                        while let Ok(progress) = progress.recv().await {
                            if let Err(error) = this.update(cx, |this, cx| {
                                if this.generation == generation {
                                    this.repair_status = Some(progress);
                                    cx.notify();
                                }
                            }) {
                                log::debug!("Doctor GitHub progress view unavailable: {error}");
                                break;
                            }
                        }
                    });
                    let result = operation.await?;
                    drop(progress_task);
                    let mut observed =
                        OwnerRepairResult::observed(OwnerRepair::Github(result.review));
                    observed.github = result.snapshot.map(|snapshot| (authority, snapshot));
                    Ok(observed)
                })
            }
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            OwnerRepair::Advisory(review) => {
                let (updates, progress) = async_channel::bounded(32);
                let selected_project = self.project.clone();
                let projection = self.security.projection.clone();
                let snapshot = self.security.snapshot.clone();
                let computation_cancelled = cancelled.clone();
                let computation_cancellation = cancel.clone();
                let operation = crate::doctor_dependencies::launch_advisory_refresh(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    updates,
                    cx,
                );
                cx.spawn(async move |this, cx| {
                    let progress_task = cx.spawn(async move |cx| {
                        while let Ok(progress) = progress.recv().await {
                            if let Err(error) = this.update(cx, |this, cx| {
                                if this.generation == generation {
                                    this.repair_status = Some(DiagnosticText::new(format!("{}: {} bytes", progress.stage.as_str(), progress.received_bytes)));
                                    cx.notify();
                                }
                            }) { log::debug!("Doctor progress view unavailable: {error}"); break; }
                        }
                    });
                    let result = operation.await?;
                    drop(progress_task);
                    let mut recommendations = Vec::new();
                    if let Some(metadata) = result.registry_metadata.as_ref() {
                        let projection = projection.ok_or_else(|| anyhow::anyhow!("Current selected graph is unavailable"))?;
                        cx.update(|cx| projection.revalidate(&selected_project, cx))?;
                        let graph = projection.graph().clone();
                        let metadata = metadata.clone();
                        let executor = cx.background_executor().clone();
                        let computation = executor.spawn(async move {
                            crate::doctor_dependencies::recommend_updates(&graph, &metadata, snapshot.as_ref(), Self::now()?, || {
                                anyhow::ensure!(!computation_cancelled.is_closed() && computation_cancelled.is_empty(), "Candidate interpretation cancelled");
                                Ok(())
                            })
                        });
                        use futures::FutureExt as _;
                        let deadline = executor.timer(std::time::Duration::from_secs(60));
                        futures::pin_mut!(computation, deadline);
                        recommendations = futures::select_biased! {
                            result = computation.fuse() => result?,
                            _ = deadline.fuse() => { computation_cancellation.close(); anyhow::bail!("Candidate interpretation timed out; retrieval receipt remains, no update proof") },
                        };
                        cx.update(|cx| projection.revalidate(&selected_project, cx))?;
                    }
                    let mut observed = OwnerRepairResult::observed(OwnerRepair::Advisory(result.review));
                    observed.recommendations = recommendations;
                    observed.advisory = Some(SecurityObservation { resolved_commit: result.resolved_commit, snapshot: result.snapshot, registry_metadata: result.registry_metadata, cleared_dataset: result.cleared_dataset, scan: result.scan });
                    Ok(observed)
                })
            }
            OwnerRepair::Project(review) => {
                let operation = crate::doctor::launch_project_repair(
                    review,
                    self.project.clone(),
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation
                        .await
                        .map(|review| OwnerRepairResult::observed(OwnerRepair::Project(review)))
                })
            }
            #[cfg(feature = "rust-tools")]
            OwnerRepair::Rustup(review) => {
                let operation = crate::doctor_install::launch_rustup_operation(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation.await.map(|result| OwnerRepairResult {
                        review: OwnerRepair::Rustup(result.review),
                        proposal: result.proposed_install.map(OwnerRepair::Rustup),
                        cleanup_error: None,
                        investigation: None,
                        #[cfg(feature = "agentic")]
                        github: None,
                        comparison: None,
                        task: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        advisory: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        projection: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        recommendations: Vec::new(),
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        debug_session: None,
                        #[cfg(feature = "rust-tools")]
                        profile: None,
                    })
                })
            }
            #[cfg(feature = "jvm-tools")]
            OwnerRepair::Mise(review) => {
                let operation = crate::doctor_jvm_install::launch_mise_operation(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation.await.map(|result| OwnerRepairResult {
                        review: OwnerRepair::Mise(result.review),
                        proposal: result.proposal.map(OwnerRepair::Mise),
                        cleanup_error: result.cleanup_error,
                        investigation: None,
                        #[cfg(feature = "agentic")]
                        github: None,
                        comparison: None,
                        task: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        advisory: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        projection: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        recommendations: Vec::new(),
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        debug_session: None,
                        #[cfg(feature = "rust-tools")]
                        profile: None,
                    })
                })
            }
            #[cfg(feature = "jvm-tools")]
            OwnerRepair::Sdkman(review) => {
                let operation = crate::doctor_jvm_install::launch_sdkman_installation(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation.await.map(|result| OwnerRepairResult {
                        review: OwnerRepair::Sdkman(result.review),
                        proposal: None,
                        cleanup_error: result.cleanup_error,
                        investigation: None,
                        #[cfg(feature = "agentic")]
                        github: None,
                        comparison: None,
                        task: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        advisory: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        projection: None,
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        recommendations: Vec::new(),
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        debug_session: None,
                        #[cfg(feature = "rust-tools")]
                        profile: None,
                    })
                })
            }
            #[cfg(feature = "agentic")]
            OwnerRepair::Probe(review) => {
                let operation = crate::doctor_agent::launch_owner_probe(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancel,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation
                        .await
                        .map(|review| OwnerRepairResult::observed(OwnerRepair::Probe(review)))
                })
            }
            #[cfg(feature = "agentic")]
            OwnerRepair::Agent(review) => {
                let operation = crate::doctor_agent::launch_agent_owner_repair(
                    review,
                    &self.project,
                    directory,
                    generation,
                    report,
                    product,
                    history,
                    now,
                    cancelled,
                    cx,
                );
                cx.spawn(async move |_, _| {
                    operation
                        .await
                        .map(|review| OwnerRepairResult::observed(OwnerRepair::Agent(review)))
                })
            }
        };
        self.repair_task = Some(cx.spawn(async move |this, cx| {
            let result = operation.await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.repair_task = None;
                this.restart_active = None;
                this.repair_cancel = None;
                if this.generation != generation {
                    match result {
                        Ok(mut result) => {
                            this.error = result.cleanup_error;
                            if let Some(mut proposal) = result.proposal.take() {
                                if let Err(error) = proposal.cancel() { this.error = Some(DiagnosticText::new(error.to_string())); }
                            }
                        }
                        Err(error) => this.error = Some(DiagnosticText::new(error.to_string())),
                    }
                    this.repair_status = Some(DiagnosticText::new("Previous-scope action ended; inspect its retained receipt and collect fresh checks")); cx.notify(); return;
                }
                match result {
                    Ok(result) => {
                        #[cfg(feature = "agentic")]
                        if matches!(&result.review, OwnerRepair::Github(_)) {
                            this.github.clear_observations();
                            if let Some((authority, snapshot)) = result.github {
                                if this.github.matches(&authority, cx) {
                                    this.github.authority = Some(authority); this.github.snapshot = Some(snapshot);
                                } else { this.error = Some(DiagnosticText::new("GitHub input selection changed; previous-scope receipt retained, metadata is not current evidence")); }
                            }
                        }
                        let investigation_result = result.investigation.is_some();
                        if let Some(investigation) = result.investigation {
                            if let Some(report) = this.report.as_mut() {
                                report.clear_comparison();
                                if let Err(error) = report.set_investigation(investigation) { this.error = Some(DiagnosticText::new(error.to_string())); }
                            }
                            #[cfg(feature = "agentic")]
                            { this.native_evidence.clear(); this.native_investigation_evidence.clear(); }
                        }
                        if investigation_result { this.current_comparison = result.comparison; }
                        else if let Some(comparison) = result.comparison { this.current_comparison = Some(comparison); }
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        {
                            if let Some(projection) = result.projection { this.security.set_projection(projection); }
                            if let Some(advisory) = result.advisory {
                                this.security.accept_advisory(advisory, result.recommendations);
                                if let Some(check) = result.review.after().cloned() {
                                    if let Some(report) = this.report.as_mut() {
                                        if let Err(error) = report.push(check) { this.error = Some(DiagnosticText::new(error.to_string())); }
                                    }
                                }
                            }
                        }
                        if investigation_result { this.task_result = result.task; }
                        else if let Some(task) = result.task { this.task_result = Some(task); }
                        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                        { if investigation_result { this.debug_result = result.debug_session; } else if let Some(session) = result.debug_session { this.debug_result = Some(session); } }
                        this.saved_generation = None;
                        this.history_change = None;
                        #[cfg(feature = "rust-tools")]
                        { if investigation_result { this.profile_result = result.profile; } else if let Some(profile) = result.profile { this.profile_result = Some(profile); } }
                        let review = result.review;
                        if result.cleanup_error.is_some() { this.error = result.cleanup_error; }
                        this.proposal = result.proposal;
                        this.repair_status = Some(DiagnosticText::new(format!("Observed owner readiness result: {:?}. This does not prove a project build or application issue is fixed; inspect the relevant before/after evidence.", review.phase())));
                        this.owner_result = Some(review);
                    }
                    Err(error) => { this.error = Some(DiagnosticText::new(error.to_string())); this.repair_status = Some(DiagnosticText::new("Repair admission or observation failed; no verified fix. Reached receipts are retained without replay.")); }
                }
                cx.emit(DoctorEvidenceUpdated);
                cx.notify();
            }) { log::debug!("Doctor view closed after repair observation: {error}"); }
        }));
        cx.emit(DoctorEvidenceUpdated);
        cx.notify();
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    fn prepare_install(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.preparation.is_some() || self.repair_task.is_some() {
            return;
        }
        let preparation = (|| {
            let directory = self.review_directory(cx)?;
            let now = Self::now()?;
            let executable = self.install.executable.read(cx).text(cx).trim().to_owned();
            let version = self.install.version.read(cx).text(cx).trim().to_owned();
            let package = self.install.package.read(cx).text(cx).trim().to_owned();
            anyhow::ensure!(
                executable.len() <= 4096 && version.len() <= 128 && package.len() <= 128,
                "Manager proposal fields exceed their limits"
            );
            let operation = match self.install.kind {
                #[cfg(feature = "rust-tools")]
                kind @ (InstallKind::RustToolchain
                | InstallKind::RustComponent
                | InstallKind::RustTarget) => {
                    let request = match kind {
                        InstallKind::RustToolchain => {
                            crate::doctor_install::RustupInstallRequest::Toolchain { version }
                        }
                        InstallKind::RustComponent => {
                            crate::doctor_install::RustupInstallRequest::Component {
                                version,
                                component: package,
                            }
                        }
                        InstallKind::RustTarget => {
                            crate::doctor_install::RustupInstallRequest::Target {
                                version,
                                target: package,
                            }
                        }
                        #[cfg(feature = "jvm-tools")]
                        _ => anyhow::bail!("Unsupported Rust installation selection"),
                    };
                    let operation = crate::doctor_install::prepare_rustup_inspection(
                        &self.project,
                        directory,
                        executable.into(),
                        request,
                        self.generation,
                        now,
                        cx,
                    );
                    cx.spawn(async move |_, _| operation.await.map(OwnerRepair::Rustup))
                }
                #[cfg(feature = "jvm-tools")]
                InstallKind::Mise => {
                    let operation = crate::doctor_jvm_install::prepare_mise_inspection(
                        &self.project,
                        directory,
                        executable.into(),
                        crate::doctor_jvm_install::MiseJdkRequest { version },
                        self.generation,
                        now,
                        cx,
                    );
                    cx.spawn(async move |_, _| operation.await.map(OwnerRepair::Mise))
                }
                #[cfg(feature = "jvm-tools")]
                InstallKind::Sdkman => {
                    let operation = crate::doctor_jvm_install::prepare_sdkman_installation(
                        &self.project,
                        directory,
                        crate::doctor_jvm_install::SdkmanJdkRequest { version },
                        self.generation,
                        now,
                        cx,
                    );
                    cx.spawn(async move |_, _| operation.await.map(OwnerRepair::Sdkman))
                }
            };
            Ok::<_, anyhow::Error>(operation)
        })();
        match preparation {
            Ok(operation) => {
                self.cancel_repair(cx);
                let generation = self.generation;
                self.repair_status = Some(DiagnosticText::new(
                    "Inspecting bounded installed-manager metadata only; no command, download or installation has been approved",
                ));
                self.preparation = Some(cx.spawn_in(window, async move |this, cx| {
                    let result = operation.await;
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.preparation = None;
                        if this.generation == generation {
                            this.accept_proposal(result, window, cx);
                        }
                    }) {
                        log::debug!("Doctor manager metadata review view closed: {error}");
                    }
                }));
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    #[cfg(feature = "jvm-tools")]
    fn approve_first_default(&mut self, cx: &mut Context<Self>) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let binding = crate::doctor::project_repair_binding(
                &self.project,
                directory,
                self.generation,
                cx,
            )?;
            let Some(OwnerRepair::Sdkman(review)) = &self.proposal else {
                anyhow::bail!("No current SDKMAN first-default proposal");
            };
            review.approve_sdkman_first_default(&binding)
        })();
        match result {
            Ok(()) => {
                self.repair_status = Some(DiagnosticText::new(
                    "First-default creation acknowledged for this exact proposal. Installation still requires its separate action approval.",
                ))
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn review_restart(&mut self, remote: bool, window: &mut Window, cx: &mut Context<Self>) {
        let preparation = (|| {
            let directory = self.review_directory(cx)?;
            let now = Self::now()?;
            let operation = if remote {
                let operation = crate::doctor::prepare_remote_restart_for_item(
                    &self.project,
                    directory,
                    self.generation,
                    now,
                    Some(cx.entity_id()),
                    cx,
                );
                cx.spawn(async move |_, _| operation.await.map(OwnerRepair::RemoteRestart))
            } else {
                let operation = crate::doctor::prepare_application_restart(
                    &self.project,
                    directory,
                    self.generation,
                    now,
                    cx,
                );
                cx.spawn(async move |_, _| operation.await.map(OwnerRepair::ApplicationRestart))
            };
            Ok::<_, anyhow::Error>(operation)
        })();
        match preparation {
            Ok(operation) => {
                self.cancel_repair(cx);
                self.restart_preparation = true;
                let generation = self.generation;
                self.repair_status = Some(DiagnosticText::new(
                    "Checking affected workspace activity and restart authority only; no restart is approved",
                ));
                self.preparation = Some(cx.spawn_in(window, async move |this, cx| {
                    let result = operation.await;
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.preparation = None;
                        this.restart_preparation = false;
                        if this.generation == generation {
                            this.accept_proposal(result, window, cx);
                        }
                    }) {
                        log::debug!("Doctor restart review view closed: {error}");
                    }
                }));
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    fn restart_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        v_flex().gap_2()
            .child(Label::new("Restart review checks every affected workspace first. Resolve dirty files, active Tasks/debugger/terminals/agent/recipe work and diagnostic collection. No OS, Docker-engine or arbitrary service restart."))
            .child(h_flex().flex_wrap().gap_2()
                .child(div().debug_selector(|| "doctor-review-application-restart".into()).child(Button::new("doctor-review-application-restart", "Review normal Made restart…").disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.review_restart(false, window, cx)))))
                .child(div().debug_selector(|| "doctor-review-remote-restart".into()).child(Button::new("doctor-review-remote-restart", "Review owned development server restart…").disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.review_restart(true, window, cx))))))
            .into_any_element()
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    fn review_dependency(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let preparation = (|| {
            let directory = self.review_directory(cx)?;
            let path = self.dependency.executable.read(cx).text(cx);
            anyhow::ensure!(path.len() <= 4096, "Selected executable path is too long");
            Ok::<_, anyhow::Error>(crate::doctor::prepare_dependency_probe(
                &self.project,
                directory,
                self.dependency.kind,
                path.into(),
                self.generation,
                Self::now()?,
                cx,
            ))
        })();
        match preparation {
            Ok(operation) => {
                self.cancel_repair(cx);
                let generation = self.generation;
                self.repair_status = Some(DiagnosticText::new(
                    "Inspecting selected native executable metadata only; no process is approved",
                ));
                self.preparation = Some(cx.spawn_in(window, async move |this, cx| {
                    let result = operation.await;
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.preparation = None;
                        if this.generation == generation {
                            if result.is_ok() {
                                this.dependency.open = false;
                            }
                            this.accept_proposal(result.map(OwnerRepair::Dependency), window, cx);
                        }
                    }) {
                        log::debug!("Doctor version review view closed: {error}");
                    }
                }));
            }
            Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
        }
        cx.notify();
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    fn dependency_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor::DependencyProbe;
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        let mut controls = v_flex().gap_2().child(
            Button::new(
                "doctor-dependency-form",
                if self.dependency.open {
                    "Hide version probe form"
                } else {
                    "Review installed tool version…"
                },
            )
            .disabled(busy)
            .tab_index(0_isize)
            .on_click(cx.listener(|this, _, _, cx| {
                this.dependency.open = !this.dependency.open;
                cx.notify();
            })),
        );
        if !self.dependency.open {
            return controls.into_any_element();
        }
        let mut kinds = Vec::new();
        #[cfg(feature = "rust-tools")]
        kinds.extend([
            (DependencyProbe::RustCompiler, "rustc"),
            (DependencyProbe::Cargo, "Cargo"),
            (DependencyProbe::RustFormatter, "rustfmt"),
            (DependencyProbe::ClippyDriver, "clippy-driver"),
            (DependencyProbe::RustAnalyzer, "rust-analyzer"),
        ]);
        #[cfg(feature = "jvm-tools")]
        kinds.extend([
            (DependencyProbe::Java, "java"),
            (DependencyProbe::JavaCompiler, "javac"),
        ]);
        controls = controls.child(Label::new("Select an installed native binary. Preparing review only inspects metadata; scripts, provisioning shims, project executables and automatic installs are blocked. Remote version probes are unsupported."))
            .child(h_flex().flex_wrap().gap_2().children(kinds.into_iter().enumerate().map(|(index, (kind, label))|
                Button::new(("doctor-dependency-kind", index), label).toggle_state(self.dependency.kind == kind).disabled(busy).tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, _, cx| { this.cancel_repair(cx); this.dependency.kind = kind; cx.notify(); })))));
        if !busy {
            controls = controls
                .child(Label::new("Absolute installed executable"))
                .child(
                    div()
                        .id("doctor-dependency-path")
                        .aria_label("Absolute installed native executable path")
                        .child(self.dependency.executable.clone()),
                );
        }
        controls
            .child(
                div()
                    .debug_selector(|| "doctor-review-dependency".into())
                    .child(
                        Button::new(
                            "doctor-review-dependency",
                            "Inspect and prepare version review (no execution)",
                        )
                        .disabled(busy || self.report.is_none())
                        .tab_index(0_isize)
                        .on_click(
                            cx.listener(|this, _, window, cx| this.review_dependency(window, cx)),
                        ),
                    ),
            )
            .into_any_element()
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    fn installation_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut kinds = Vec::new();
        #[cfg(feature = "rust-tools")]
        kinds.extend([
            (InstallKind::RustToolchain, "Rust toolchain"),
            (InstallKind::RustComponent, "Rust component"),
            (InstallKind::RustTarget, "Rust target"),
        ]);
        #[cfg(feature = "jvm-tools")]
        kinds.extend([
            (InstallKind::Mise, "mise Java"),
            (InstallKind::Sdkman, "SDKMAN Java"),
        ]);
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        let controls = v_flex().gap_2().child(
            Button::new(
                "doctor-install-form",
                if self.install.open {
                    "Hide manager form"
                } else {
                    "Review existing-manager installation…"
                },
            )
            .disabled(busy)
            .tab_index(0_isize)
            .on_click(cx.listener(|this, _, _, cx| {
                this.install.open = !this.install.open;
                cx.notify();
            })),
        );
        if !self.install.open {
            return controls.into_any_element();
        }
        let mut form = controls
            .child(Label::new("Already-installed managers only. Every stage has a separate approval; opening this form executes nothing. Remote installation is currently unsupported."))
            .child(h_flex().flex_wrap().gap_2().children(kinds.into_iter().enumerate().map(|(index, (kind, label))| Button::new(("doctor-install-kind", index), label).toggle_state(self.install.kind == kind).disabled(busy).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| { this.cancel_repair(cx); this.install.kind = kind; cx.notify(); })))))
            .child(Label::new("Exact version / vendor-qualified Java ID"));
        if !busy {
            form = form.child(
                div()
                    .id("doctor-install-version")
                    .aria_label("Exact toolchain or vendor-qualified Java version")
                    .child(self.install.version.clone()),
            );
            #[cfg(feature = "jvm-tools")]
            let requires_executable = self.install.kind != InstallKind::Sdkman;
            #[cfg(not(feature = "jvm-tools"))]
            let requires_executable = true;
            if requires_executable {
                form = form
                    .child(Label::new("Absolute existing manager executable"))
                    .child(
                        div()
                            .id("doctor-install-executable")
                            .aria_label("Absolute installed manager executable")
                            .child(self.install.executable.clone()),
                    );
            }
            #[cfg(feature = "jvm-tools")]
            if self.install.kind == InstallKind::Sdkman {
                form = form.child(Label::new("SDKMAN is validated from the existing user home. Official post-install hooks execute with user permissions; first-default creation is separately reviewed."));
            }
            #[cfg(feature = "rust-tools")]
            if matches!(
                self.install.kind,
                InstallKind::RustComponent | InstallKind::RustTarget
            ) {
                form = form.child(Label::new("Exact component or target")).child(
                    div()
                        .id("doctor-install-package")
                        .aria_label("Exact Rust component or target")
                        .child(self.install.package.clone()),
                );
            }
        } else {
            form = form.child(Label::new(
                "Manager inputs are fixed while their reviewed operation is active.",
            ));
        }
        form.child(
            div()
                .debug_selector(|| "doctor-prepare-install".into())
                .child(
                    Button::new(
                        "doctor-prepare-install",
                        "Prepare exact review (no execution)",
                    )
                    .disabled(busy || self.report.is_none())
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, window, cx| this.prepare_install(window, cx))),
                ),
        )
        .when(self.preparation.is_some(), |form| {
            form.child(
                Button::new("doctor-cancel-preparation", "Cancel metadata review")
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| this.cancel_repair(cx))),
            )
        })
        .into_any_element()
    }

    #[cfg(feature = "agentic")]
    fn review_probe(&mut self, docker: bool, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| {
            let directory = self.review_directory(cx)?;
            let preview = if docker {
                let endpoint = self.probe.endpoint.read(cx).text(cx).trim().to_owned();
                let image = self.probe.image.read(cx).text(cx).trim().to_owned();
                let runtime = self.probe.runtime.read(cx).text(cx).trim().to_owned();
                anyhow::ensure!(
                    endpoint.len() <= 4096 && image.len() <= 1024 && runtime.len() <= 4096,
                    "Docker probe fields exceed their limits"
                );
                crate::doctor_agent::DoctorProbePreview::Docker(
                    crate::doctor_agent::DockerPrerequisitePreview::prepare(
                        endpoint,
                        image,
                        runtime.into(),
                        self.generation,
                        cx,
                    )?,
                )
            } else {
                crate::doctor_agent::DoctorProbePreview::Provider(
                    crate::doctor_agent::ProviderProbePreview::prepare(self.generation, cx)?,
                )
            };
            crate::doctor_agent::prepare_owner_probe(
                &self.project,
                directory,
                preview,
                self.generation,
                Self::now()?,
                cx,
            )
            .map(OwnerRepair::Probe)
        })();
        self.accept_proposal(result, window, cx);
    }

    #[cfg(feature = "agentic")]
    fn probe_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        let controls = v_flex()
            .gap_2()
            .child(
                div()
                    .debug_selector(|| "doctor-review-provider-probe".into())
                    .child(
                        Button::new(
                            "doctor-review-provider-probe",
                            "Review selected provider connectivity probe…",
                        )
                        .disabled(busy || self.report.is_none())
                        .tab_index(0_isize)
                        .on_click(
                            cx.listener(|this, _, window, cx| this.review_probe(false, window, cx)),
                        ),
                    ),
            )
            .child(
                Button::new(
                    "doctor-docker-probe-form",
                    if self.probe.open {
                        "Hide Docker prerequisite form"
                    } else {
                        "Review local Docker prerequisites…"
                    },
                )
                .disabled(busy)
                .tab_index(0_isize)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.probe.open = !this.probe.open;
                    cx.notify();
                })),
            );
        if !self.probe.open {
            return controls.into_any_element();
        }
        controls.child(Label::new("Specify the exact local engine, installed immutable image and Made Linux runtime. Preparing a review performs no Docker query, image pull, container creation or recipe execution."))
            .when(!busy, |controls| controls
                .child(Label::new("Local engine endpoint")).child(div().id("doctor-probe-endpoint").aria_label("Exact local Docker engine endpoint").child(self.probe.endpoint.clone()))
                .child(Label::new("Installed immutable image")).child(div().id("doctor-probe-image").aria_label("Immutable installed Docker image").child(self.probe.image.clone()))
                .child(Label::new("Made Linux runtime")).child(div().id("doctor-probe-runtime").aria_label("Absolute Made Linux runtime path").child(self.probe.runtime.clone())))
            .child(div().debug_selector(|| "doctor-review-docker-probe".into()).child(Button::new("doctor-review-docker-probe", "Prepare exact Docker review (no queries)").disabled(busy || self.report.is_none()).tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.review_probe(true, window, cx)))))
            .into_any_element()
    }

    fn repair_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut controls = v_flex().gap_2();
        if let Some(directory) = self.directory(cx) {
            let observation = self
                .project
                .read(cx)
                .doctor_observation(directory.worktree_id, cx);
            for server in observation.language_servers {
                let action = crate::doctor::DoctorProjectRepairAction::RestartLanguageServer {
                    server: server.id,
                    name: server.name.clone(),
                };
                controls = controls.child(
                    Button::new(
                        ("doctor-review-lsp", server.id.0),
                        format!("Review restart: {} (instance {})", server.name, server.id.0),
                    )
                    .disabled(
                        self.report.is_none()
                            || self.repair_task.is_some()
                            || self.preparation.is_some(),
                    )
                    .tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.review_project_action(action.clone(), window, cx)
                    })),
                );
                #[cfg(feature = "jvm-tools")]
                {
                    let store = self.project.read(cx).lsp_store();
                    let store = store.read(cx);
                    let action = if server.name == lsp::LanguageServerName::new_static("jdtls") {
                        Some(
                            crate::doctor::DoctorProjectRepairAction::RefreshJavaProject {
                                server: server.id,
                                generation: store
                                    .jdtls_project_snapshot(directory.worktree_id)
                                    .map(|(snapshot, _)| snapshot.generation),
                            },
                        )
                    } else if server.name == lsp::LanguageServerName::new_static("kotlin-lsp") {
                        Some(
                            crate::doctor::DoctorProjectRepairAction::RefreshKotlinProject {
                                server: server.id,
                                generation: store
                                    .kotlin_workspace_snapshot(directory.worktree_id)
                                    .map(|(snapshot, _)| snapshot.generation),
                            },
                        )
                    } else {
                        None
                    };
                    if let Some(action) = action {
                        controls = controls.child(
                            Button::new(
                                ("doctor-review-import", server.id.0),
                                format!("Review metadata refresh: {}", server.name),
                            )
                            .disabled(
                                self.report.is_none()
                                    || self.repair_task.is_some()
                                    || self.preparation.is_some(),
                            )
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                move |this, _, window, cx| {
                                    this.review_project_action(action.clone(), window, cx)
                                },
                            )),
                        );
                    }
                }
            }
            #[cfg(feature = "agentic")]
            {
                let store = self.project.read(cx).context_server_store();
                let ids = store.read(cx).configured_server_ids();
                for (index, id) in ids.into_iter().take(12).enumerate() {
                    let request =
                        crate::doctor_agent::DoctorAgentRepairRequest::RestartMcp(id.clone());
                    controls = controls.child(
                        div().debug_selector(|| "doctor-review-mcp".into()).child(
                            Button::new(
                                ("doctor-review-mcp", index),
                                format!("Review restart: MCP {id}"),
                            )
                            .disabled(
                                self.report.is_none()
                                    || self.repair_task.is_some()
                                    || self.preparation.is_some(),
                            )
                            .tab_index(0_isize)
                            .on_click(cx.listener(
                                move |this, _, window, cx| {
                                    this.review_agent_action(request.clone(), window, cx)
                                },
                            )),
                        ),
                    );
                }
                for (index, id) in product_flavor::DEFAULT_EXTENSIONS.iter().enumerate() {
                    let request =
                        crate::doctor_agent::DoctorAgentRepairRequest::RetryProductExtension(
                            (*id).into(),
                        );
                    controls = controls.child(
                        Button::new(
                            ("doctor-review-extension", index),
                            format!("Review retry: missing {id} extension"),
                        )
                        .disabled(
                            self.report.is_none()
                                || self.repair_task.is_some()
                                || self.preparation.is_some(),
                        )
                        .tab_index(0_isize)
                        .on_click(cx.listener(
                            move |this, _, window, cx| {
                                this.review_agent_action(request.clone(), window, cx)
                            },
                        )),
                    );
                }
            }
        }
        if let Some(review) = &self.proposal {
            #[cfg(feature = "jvm-tools")]
            let first_default_control = if let OwnerRepair::Sdkman(review) = review {
                review.action().requires_first_default().then(|| {
                    div()
                        .debug_selector(|| "doctor-approve-first-default".into())
                        .child(
                            Button::new(
                                "doctor-approve-first-default",
                                "Approve creation of this first Java default",
                            )
                            .disabled(review.action().has_first_default_consent())
                            .tab_index(0_isize)
                            .on_click(cx.listener(|this, _, _, cx| this.approve_first_default(cx))),
                        )
                        .into_any_element()
                })
            } else {
                None
            };
            #[cfg(not(feature = "jvm-tools"))]
            let first_default_control: Option<gpui::AnyElement> = None;
            #[cfg(feature = "jvm-tools")]
            let requires_default_consent = matches!(review, OwnerRepair::Sdkman(review) if review.action().requires_first_default() && !review.action().has_first_default_consent());
            #[cfg(not(feature = "jvm-tools"))]
            let requires_default_consent = false;
            let exact_preview = review.exact_preview();
            let invalid_preview = exact_preview.is_err();
            controls = controls.child(
                v_flex()
                    .id("doctor-review-preview")
                    .anchor_scroll(Some(self.review_anchor.clone()))
                    .p_3()
                    .gap_2()
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .child(Label::new(review.target()))
                    .child(Label::new(format!(
                        "Scope: {}; {}",
                        self.report
                            .as_ref()
                            .map(|report| report.scope.worktree.as_str())
                            .unwrap_or("unavailable"),
                        self.host_label(cx)
                    )))
                    .child(Label::new(review.before().detail.as_str().to_owned()))
                    .child(Label::new(review.effects().as_str().to_owned()))
                    .children(match exact_preview {
                        Ok(lines) => lines
                            .into_iter()
                            .map(|line| Label::new(line).into_any_element())
                            .collect::<Vec<_>>(),
                        Err(error) => vec![
                            Label::new(DiagnosticText::new(error.to_string()).as_str().to_owned())
                                .color(Color::Error)
                                .into_any_element(),
                        ],
                    })
                    .when_some(first_default_control, |element, control| {
                        element.child(control)
                    })
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                div()
                                    .debug_selector(|| "doctor-approve-repair".into())
                                    .child(
                                        Button::new(
                                            "doctor-approve-repair",
                                            "Approve this exact action",
                                        )
                                        .disabled(requires_default_consent || invalid_preview)
                                        .tab_index(0_isize)
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.approve_repair(cx)),
                                        ),
                                    ),
                            )
                            .child(
                                div()
                                    .debug_selector(|| "doctor-cancel-repair".into())
                                    .child(
                                        Button::new("doctor-cancel-proposal", "Cancel proposal")
                                            .tab_index(0_isize)
                                            .on_click(
                                                cx.listener(|this, _, _, cx| {
                                                    this.cancel_repair(cx)
                                                }),
                                            ),
                                    ),
                            ),
                    ),
            );
        }
        if let Some(status) = &self.repair_status {
            controls = controls.child(
                div()
                    .id("doctor-repair-status")
                    .role(gpui::Role::Status)
                    .aria_value(status.as_str().to_owned())
                    .aria_label("Doctor reviewed-action status")
                    .aria_live(gpui::accesskit::Live::Polite)
                    .child(Label::new(status.as_str().to_owned())),
            );
        }
        if let Some(result) = &self.owner_result {
            if let Some(detail) = result.operation_detail() {
                controls = controls.child(Label::new(format!(
                    "Observed operation and limits: {}",
                    detail.as_str()
                )));
            }
            controls = controls.child(Label::new(format!(
                "Before: {} — {}",
                result.before().outcome.label(),
                result.before().detail.as_str()
            )));
            if let Some(after) = result.after() {
                controls = controls.child(Label::new(format!(
                    "Relevant recheck: {} — {}",
                    after.outcome.label(),
                    after.detail.as_str()
                )));
            }
        }
        controls.into_any_element()
    }

    fn open_handoff(
        &mut self,
        handoff: crate::doctor::DoctorProjectHandoff,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.directory(cx).is_none() {
            self.error = Some(DiagnosticText::new("Selected project scope is unavailable"));
        } else {
            match handoff.action(cx) {
                Ok(action) => window.dispatch_action(action, cx),
                Err(error) => self.error = Some(DiagnosticText::new(error.to_string())),
            }
        }
        cx.notify();
    }
}

impl Focusable for DoctorView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}
impl EventEmitter<ItemEvent> for DoctorView {}
impl EventEmitter<DoctorEvidenceUpdated> for DoctorView {}
impl EventEmitter<std::rc::Rc<crate::doctor::DoctorNativeRepairRequest>> for DoctorView {}
impl Drop for DoctorView {
    fn drop(&mut self) {
        if let Some(cancel) = &self.repair_cancel {
            cancel.close();
        }
    }
}
impl Item for DoctorView {
    type Event = ItemEvent;
    fn tab_content_text(&self, _: usize, _: &App) -> SharedString {
        "Doctor".into()
    }
    fn tab_tooltip_text(&self, _: &App) -> Option<SharedString> {
        Some("Made Doctor — scoped diagnosis and reviewed repairs".into())
    }
    fn active_project_path(&self, cx: &App) -> Option<ProjectPath> {
        self.directory(cx)
    }
    fn restart_blocker(&self, _: &App) -> Option<&'static str> {
        if self.collection.is_some() {
            return Some("Finish Doctor diagnostic collection before restarting");
        }
        if self.history_task.is_some() {
            return Some("Finish the accepted Doctor history database operation before restarting");
        }
        if self.export_task.is_some() {
            return Some("Finish or cancel the selected Doctor export before restarting");
        }
        if (self.preparation.is_some() && !self.restart_preparation)
            || (self.repair_task.is_some()
                && self.restart_active != Some(RestartCoordination::Application))
        {
            return Some("Finish or cancel the other Doctor operation before restarting");
        }
        None
    }
}

impl DoctorView {
    fn navigation_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor::DoctorInvestigationScope as Scope;
        let busy = self.repair_task.is_some() || self.preparation.is_some();
        v_flex()
            .id("doctor-navigation")
            .role(gpui::Role::Tree)
            .aria_label("Doctor sections")
            .w(rems(12.))
            .h_full()
            .flex_shrink_0()
            .p_2()
            .gap_1()
            .border_r_1()
            .border_color(cx.theme().colors().border)
            .bg(cx.theme().colors().panel_background)
            .children([
                (Scope::Product, "Configuration health"),
                (Scope::Development, "Development health"),
                (Scope::Application, "Application behavior"),
            ].into_iter().enumerate().map(|(index, (scope, label))| {
                div().debug_selector(move || format!("doctor-investigation-scope-{index}")).child(
                    TreeViewItem::new(("doctor-section", index), label)
                        .toggle_state(!self.advanced_open && self.investigation.scope == scope)
                        .disabled(busy)
                        .tab_index(0_isize)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.cancel_repair(cx);
                            this.advanced_open = false;
                            this.investigation.scope = scope;
                            this.investigation.problem = match scope {
                                Scope::Product => crate::doctor::DoctorInvestigationProblem::ProductReadiness,
                                Scope::Development => crate::doctor::DoctorInvestigationProblem::Build,
                                Scope::Application => crate::doctor::DoctorInvestigationProblem::IncorrectBehavior,
                            };
                            this.scroll_handle.set_offset(gpui::Point::default());
                            cx.notify();
                        })),
                )
            }))
            .child(div().debug_selector(|| "doctor-advanced".into()).child(
                TreeViewItem::new("doctor-advanced-section", "Advanced actions")
                    .toggle_state(self.advanced_open)
                    .disabled(busy)
                    .tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.advanced_open = true;
                        this.scroll_handle.set_offset(gpui::Point::default());
                        cx.notify();
                    })),
            ))
            .into_any_element()
    }

    fn scope_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor::DoctorInvestigationScope as Scope;
        let (title, inputs, effects, next) = match self.investigation.scope {
            Scope::Product => (
                format!("{} configuration health", product_flavor::DISPLAY_NAME),
                "Examines the configured model/provider, extensions, context servers, project authority and automation owners.",
                "Inputs: existing application settings and owner status. Collection reads these facts; it does not send a model request or prove connectivity.",
                "Next: Collect checks, inspect problems below, then use a separately reviewed advanced probe or recovery action if needed.",
            ),
            Scope::Development => (
                "Project development environment and build/test health".to_owned(),
                "Examines the selected project's toolchain declaration, cached Cargo/JVM metadata, language servers and recorded build/test evidence.",
                "Inputs: this worktree and its existing development owners. Collection does not run Cargo, build scripts, builds or tests, download dependencies, or edit files.",
                "Next: Collect checks. To reproduce a build/test failure, set up an investigation and review its exact command and development effects.",
            ),
            Scope::Application => (
                "Application correctness and performance".to_owned(),
                "Examines the program you develop, using a selected symptom, validation target and reviewed Task, test, debugger or supported profile evidence.",
                "Inputs: an explicit reproduction and attributable results. No application check has run merely because configuration collection passed. Memory, race and performance claims require suitable measurements.",
                "Next: Set up an investigation below. Review each execution separately; source changes, installations and repairs require their own approval.",
            ),
        };
        v_flex()
            .min_w_0()
            .gap_2()
            .child(
                v_flex()
                    .min_w_0()
                    .p_3()
                    .gap_1()
                    .rounded_md()
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .child(Label::new(title).weight(gpui::FontWeight::SEMIBOLD))
                    .child(Label::new(inputs))
                    .child(
                        Label::new(effects)
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    )
                    .child(Label::new(next).size(LabelSize::Small)),
            )
            .into_any_element()
    }

    fn result_controls(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        use crate::doctor::DoctorInvestigationScope as Scope;
        let scope = self.investigation.scope;
        if scope == Scope::Application {
            return v_flex().gap_2()
                .when(self.report.as_ref().and_then(DoctorReport::investigation).is_none(), |element| {
                    element.child(div().id("doctor-application-not-run").role(gpui::Role::Status)
                        .aria_label("Application behavior: Not run")
                        .child(Label::new("Not run — no application reproduction or performance baseline has been collected.")))
                })
                .into_any_element();
        }
        let Some(report) = &self.report else {
            return div().into_any_element();
        };
        let stale = self.report_is_stale(cx);
        let mut checks: Vec<_> = report
            .checks()
            .iter()
            .enumerate()
            .filter(|(_, check)| check.scope() == scope)
            .collect();
        checks.sort_by_key(|(_, check)| check.priority());
        let mut results = v_flex().min_w_0().gap_2()
            .child(Label::new("Results — problems first").weight(gpui::FontWeight::SEMIBOLD))
            .child(Label::new("Passed means the stated observation passed. It does not certify a build, test, model response or application behavior that was not performed.").size(LabelSize::Small).color(Color::Muted));
        if checks.is_empty() {
            results = results.child(Label::new(
                "Not run — no evidence for this section was collected.",
            ));
        }
        for (index, check) in checks {
            let state = if stale { "Stale" } else { check.state_label() };
            let color = match state {
                "Failed" => Color::Error,
                "Warning" | "Blocked" | "Stale" => Color::Warning,
                "Passed" => Color::Success,
                _ => Color::Muted,
            };
            let next = if stale {
                "Collect checks again: the selected inputs changed after this observation."
            } else {
                check.next_step()
            };
            let card = v_flex()
                .id(("doctor-check", index))
                .debug_selector(move || format!("doctor-result-{index}"))
                .min_w_0()
                .flex_shrink_0()
                .p_3()
                .gap_1()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border)
                .role(gpui::Role::Group)
                .aria_label(format!(
                    "{state}: {}. Evidence: {}. Next step: {next}",
                    check.title.as_str(),
                    check.detail.as_str()
                ))
                .child(Label::new(format!("{state} — {}", check.title.as_str())).color(color))
                .child(Label::new(check.detail.as_str().to_owned()))
                .child(
                    Label::new(format!(
                        "Source: {} · {} · {}",
                        check.evidence_source(),
                        check.id.as_str(),
                        Self::timestamp(check.observed_at)
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .child(Label::new(format!("Next: {next}")).size(LabelSize::Small));
            #[cfg(feature = "agentic")]
            let card = if self.advanced_open {
                let identifier = check.id.as_str().to_owned();
                card.child(
                    Button::new(
                        ("doctor-native-evidence", index),
                        "Select for native repair",
                    )
                    .toggle_state(self.native_evidence.contains(&identifier))
                    .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                    .disabled(stale || self.repair_task.is_some())
                    .tab_index(0_isize)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.cancel_repair(cx);
                        this.native_investigation_evidence.clear();
                        if this.native_evidence.contains(&identifier) {
                            this.native_evidence
                                .retain(|selected| selected != &identifier);
                        } else if this.native_evidence.len() < 8 {
                            this.native_evidence.push(identifier.clone());
                        } else {
                            this.error = Some(DiagnosticText::new("Select at most eight checks"));
                        }
                        cx.notify();
                    })),
                )
            } else {
                card
            };
            results = results.child(card);
        }
        results.into_any_element()
    }
}

impl Render for DoctorView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let roots: Vec<_> = self
            .project
            .read(cx)
            .visible_worktrees(cx)
            .map(|root| (root.read(cx).id(), root.read(cx).root_name_str().to_owned()))
            .collect();
        let stale = self.report.is_some() && self.report_is_stale(cx);
        let status = if self.collection.is_some() {
            "Running — reading selected owner state; no build, test or repair is running".to_owned()
        } else if self.collection_cancelled {
            "Cancelled — collection stopped; collect again to obtain current results".to_owned()
        } else if self.collection_failed {
            "Failed — collection did not produce a report; inspect the error and retry".to_owned()
        } else if let Some(report) = &self.report {
            format!(
                "{}{}",
                if stale {
                    "Stale — inputs changed; collect again. "
                } else {
                    ""
                },
                report.summary()
            )
        } else {
            "Not run — choose Collect checks to read configuration and cached development evidence"
                .to_owned()
        };
        let selected_path = self.directory(cx).and_then(|directory| {
            self.project
                .read(cx)
                .worktree_for_id(directory.worktree_id, cx)
                .map(|worktree| worktree.read(cx).abs_path().display().to_string())
        });
        let scope_label = format!(
            "Project: {} · Check family: {} · Host: {}",
            roots
                .iter()
                .map(|(_, name)| name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            self.selected_product.name(),
            self.host_label(cx)
        );
        let content = v_flex().w_full().min_w_0().flex_shrink_0().p_4().gap_3()
            .child(Label::new(format!("{} Doctor", product_flavor::DISPLAY_NAME)).size(LabelSize::Large))
            .child(Label::new("Collect evidence, inspect problems, then separately review any execution or repair.").color(Color::Muted))
            .child(v_flex().id("doctor-selected-scope").min_w_0().gap_1().role(gpui::Role::Group).aria_label(scope_label.clone())
                .child(Label::new(scope_label).size(LabelSize::Small))
                .child(Label::new(format!("Worktree: {}", selected_path.as_deref().unwrap_or("Select a worktree below"))).size(LabelSize::Small)))
            .child(h_flex().flex_wrap().gap_2().children(roots.into_iter().map(|(id, label)| div().debug_selector(|| "doctor-worktree".into()).child(
                Button::new(("doctor-worktree", id.to_usize()), label.clone())
                    .style(ButtonStyle::Outlined).selected_style(ButtonStyle::Tinted(TintColor::Accent))
                    .toggle_state(self.selected_worktree == Some(id)).disabled(self.repair_task.is_some()).tab_index(0_isize)
                    .aria_label(format!("Select Doctor worktree {label}"))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if this.selected_worktree != Some(id) { this.select_worktree(id, cx); }
                    })))))
                .child(Button::new("doctor-choose-host", "Change execution host…").tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| {
                    match cx.build_action("projects::OpenRemote", None) {
                        Ok(action) => window.dispatch_action(action, cx),
                        Err(error) => { this.error = Some(DiagnosticText::new(format!("Existing remote-project controls are unavailable: {error}"))); cx.notify(); }
                    }
                }))))
            .when(!self.advanced_open, |element| element.child(self.scope_controls(cx)))
            .when(self.advanced_open, |element| element.child(
                v_flex().gap_1()
                    .child(Label::new("Advanced actions").weight(gpui::FontWeight::SEMIBOLD))
                    .child(Label::new("Choose a diagnostic probe or recovery action below. Opening a form performs no execution, repair, installation or permission grant.").size(LabelSize::Small).color(Color::Muted))))
            .child(h_flex().flex_wrap().gap_2()
                .child(div().debug_selector(|| "doctor-refresh".into()).child(
                    Button::new("doctor-refresh", if self.error.is_some() || self.collection_cancelled { "Retry checks" } else { "Collect checks" })
                        .style(ButtonStyle::Filled).disabled(self.directory(cx).is_none() || self.collection.is_some() || self.repair_task.is_some() || self.preparation.is_some())
                        .tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| this.refresh(cx)))))
                .when(self.collection.is_some(), |element| element.child(
                    div().debug_selector(|| "doctor-cancel-collection".into()).child(
                        Button::new("doctor-cancel-collection", "Cancel collection").style(ButtonStyle::Outlined).tab_index(0_isize)
                            .on_click(cx.listener(|this, _, _, cx| this.cancel_collection(cx))))))
                .child(div().debug_selector(|| "doctor-export-current".into()).child(
                    Button::new("doctor-export-current", "Export redacted report…").disabled(self.report.is_none() || self.export_task.is_some()).tab_index(0_isize)
                        .on_click(cx.listener(|this, _, _, cx| this.export_selected(false, cx)))))
)
            .when(self.directory(cx).is_none(), |element| element.child(Label::new("Collection is disabled until an available worktree is selected. Doctor never chooses another host as a fallback.").color(Color::Warning)))
            .when(self.repair_task.is_some() || self.preparation.is_some(), |element| element.child(Label::new("Collection is disabled while an action is being prepared or observed. Finish or cancel that action first.").color(Color::Warning)))
            .child(div().id("doctor-status").role(gpui::Role::Status).aria_label("Doctor collection status").aria_value(status.clone()).aria_live(gpui::accesskit::Live::Polite).child(Label::new(status)))
            .when_some(self.report.as_ref(), |element, report| element.child(Label::new(format!("Last collection: {} · configuration and cached development evidence only", Self::timestamp(report.observed_at))).size(LabelSize::Small).color(Color::Muted)))
            .when_some(self.error.as_ref(), |element, error| element.child(div().id("doctor-error").role(gpui::Role::Alert).aria_value(error.as_str().to_owned()).aria_label("Doctor error").child(Label::new(error.as_str().to_owned()).color(Color::Error))))
            .when_some(self.export_status.as_ref(), |element, status| element.child(div().id("doctor-export-status").role(gpui::Role::Status).aria_live(gpui::accesskit::Live::Polite).child(Label::new(status.as_str().to_owned()))))
            .child(self.investigation_controls(cx))
            .when(self.advanced_open || self.proposal.is_some() || self.owner_result.is_some() || self.repair_status.is_some(), |element| element.child(self.repair_controls(cx)))
            .when(self.advanced_open, |element| {
                let element = element
                    .child(Label::new("Advanced actions — opening a form performs no execution or repair").weight(gpui::FontWeight::SEMIBOLD))
                    .child(h_flex().flex_wrap().gap_2().children([DoctorProduct::Made, DoctorProduct::Copper, DoctorProduct::Groovy].into_iter().enumerate().map(|(index, product)| div().debug_selector(move || format!("doctor-product-{index}")).child(
                        Button::new(("doctor-product", index), format!("{} checks", product.name()))
                            .selected_style(ButtonStyle::Tinted(TintColor::Accent)).toggle_state(self.selected_product == product)
                            .disabled(self.repair_task.is_some()).tab_index(0_isize).on_click(cx.listener(move |this, _, _, cx| {
                                if this.selected_product != product { this.selected_product = product; this.invalidate(cx); }
                            }))))))
                    .child(Label::new("The check family selects language integration; it does not change the running product. Integrations absent from this build report Unsupported.").size(LabelSize::Small).color(Color::Muted))
                    .child(Button::new("doctor-settings", "Inspect project settings").tab_index(0_isize).on_click(cx.listener(|this, _, window, cx| this.open_handoff(crate::doctor::DoctorProjectHandoff::Settings, window, cx))))
                    .child(self.history_controls(cx))
                    .child(self.restart_controls(cx));
                #[cfg(feature = "agentic")]
                let element = element.child(self.native_repair_controls(cx)).child(self.probe_controls(cx)).child(self.github_controls(cx)).child(self.security_setup_controls(cx));
                #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
                let element = element.child(self.dependency_controls(cx)).child(self.security_controls(cx)).child(self.installation_controls(cx));
                element
            })
            .child(self.investigation_evidence_controls(cx))
            .when(!self.advanced_open, |element| element.child(self.result_controls(cx)))
            .map(|element| {
                #[cfg(feature = "agentic")]
                let element = element.when(self.advanced_open && self.native_repair_open, |element| element.child(self.result_controls(cx)));
                element
            });
        v_flex()
            .id("made-doctor")
            .debug_selector(|| "made-doctor".into())
            .track_focus(&self.focus_handle)
            .key_context("Doctor")
            .size_full()
            .min_h_0()
            .overflow_hidden()
            .when(self.repair_task.is_some(), |element| {
                element.child(
                    h_flex()
                        .flex_shrink_0()
                        .flex_wrap()
                        .p_2()
                        .gap_2()
                        .child(Label::new("Reviewed action running"))
                        .map(|element| {
                            #[cfg(feature = "agentic")]
                            let element = element.children(
                                self.native_session
                                    .clone()
                                    .map(|session| self.native_session_control(session, cx)),
                            );
                            element
                        })
                        .child(
                            div()
                                .debug_selector(|| "doctor-cancel-repair".into())
                                .child(
                                    Button::new(
                                        "doctor-cancel-operation",
                                        "Cancel waiting / owned operation",
                                    )
                                    .tab_index(0_isize)
                                    .on_click(cx.listener(|this, _, _, cx| this.cancel_repair(cx))),
                                ),
                        ),
                )
            })
            .when(self.export_task.is_some(), |element| {
                element.child(
                    h_flex()
                        .flex_shrink_0()
                        .p_2()
                        .gap_2()
                        .child(Label::new(if self.export_writing {
                            "Writing selected redacted report"
                        } else {
                            "Waiting for an explicit export destination"
                        }))
                        .child(
                            div()
                                .debug_selector(|| "doctor-export-cancel".into())
                                .child(
                                    Button::new(
                                        "doctor-export-cancel",
                                        "Cancel export before writing",
                                    )
                                    .disabled(self.export_writing)
                                    .tab_index(0_isize)
                                    .on_click(cx.listener(|this, _, _, cx| this.cancel_export(cx))),
                                ),
                        ),
                )
            })
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .overflow_hidden()
                    .child(self.navigation_controls(cx))
                    .child(
                        v_flex()
                            .id("doctor-content-frame")
                            .relative()
                            .flex_1()
                            .min_w_0()
                            .min_h_0()
                            .h_full()
                            .vertical_scrollbar_for(&self.scroll_handle, window, cx)
                            .child(
                                // Keep long pages out of the parent pane's intrinsic height.
                                div()
                                    .id("doctor-evidence-scroll")
                                    .absolute()
                                    .inset_0()
                                    .min_w_0()
                                    .min_h_0()
                                    .track_scroll(&self.scroll_handle)
                                    .overflow_y_scroll()
                                    .child(content),
                            ),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_github_flow_partial_pages_revoked_sources_and_duplicate_provenance_remain_explicit(
        cx: &mut TestAppContext,
    ) {
        use http_client::github_security::GithubSecurityCoverage;
        use std::sync::atomic::Ordering;
        let (project, view, mut visual, calls, _, _) = doctor_github_ui_fixture(cx, false).await;
        let (model, _) = crate::doctor_agent::tests::doctor_probe_fixture(cx);
        model.forbid_requests();
        project.read_with(&visual, |project, _| project.client().http_client()).as_fake().replace_handler({
            let calls = calls.clone();
            move |_, request| { let calls = calls.clone(); async move {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request.method(), http_client::Method::GET);
                let path = request.uri().path();
                if path.ends_with("/code-scanning/analyses") { return Ok(http_client::Response::builder().status(403).body("RAW_API_ERROR_NOT_EVIDENCE".into())?); }
                if request.uri().query().is_some_and(|query| query.contains("page=2")) { return Ok(http_client::Response::builder().status(503).body("RAW_API_ERROR_NOT_EVIDENCE".into())?); }
                let value = if path == "/user" { serde_json::json!({"id":42,"login":"selected-user"}) }
                    else if path == "/repos/selected-owner/selected-repository" { serde_json::json!({"id":123,"full_name":"selected-owner/selected-repository","private":true,"default_branch":"dev"}) }
                    else if path.ends_with("/code-scanning/alerts") { serde_json::Value::Array([1_u64,2].into_iter().map(|number| serde_json::json!({
                        "number":number,"state":"open","rule":{"id":"rust/example","severity":"warning","security_severity_level":"high","description":"RAW_SOURCE_NOT_HANDOFF"},
                        "tool":{"name":"Selected Scanner","version":"1.2.3"},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z",
                        "most_recent_instance":{"ref":"refs/heads/other","commit_sha":"b".repeat(40),"location":{"path":"source.txt","start_line":1,"end_line":1},"message":{"text":"RAW_SECRET_NOT_HANDOFF"}},
                        "url":format!("https://api.github.com/repos/selected-owner/selected-repository/code-scanning/alerts/{number}"),
                        "html_url":format!("https://github.com/selected-owner/selected-repository/security/code-scanning/{number}")
                    })).collect()) }
                    else { panic!("Unexpected collection {path}"); };
                let mut response = http_client::Response::builder().status(200).header("Content-Type", "application/json");
                if path.ends_with("/code-scanning/alerts") { response = response.header("Link", "<https://api.github.com/repos/selected-owner/selected-repository/code-scanning/alerts?per_page=100&ref=refs%2Fheads%2Fdev&sort=updated&direction=desc&page=2>; rel=\"next\""); }
                Ok(response.body(serde_json::to_vec(&value)?.into())?)
            }}
        });
        click_investigation_control(&view, &mut visual, "doctor-github-source-0");
        click_investigation_control(&view, &mut visual, "doctor-github-source-2");
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            let snapshot = view
                .github
                .snapshot
                .as_ref()
                .expect("Verified repository with incomplete sources");
            assert!(matches!(
                snapshot.code_scanning.coverage,
                GithubSecurityCoverage::Partial(
                    http_client::github_security::GithubSecurityError::ServiceUnavailable
                )
            ));
            assert_eq!(snapshot.code_scanning.observations.len(), 2);
            assert!(matches!(
                snapshot.analyses.coverage,
                GithubSecurityCoverage::Unavailable(
                    http_client::github_security::GithubSecurityError::AccessUnavailable
                )
            ));
            assert_eq!(
                snapshot.dependabot.coverage,
                GithubSecurityCoverage::NotRequested
            );
            let correlation = crate::doctor_dependencies::correlate_github_evidence(
                snapshot,
                &[],
                DoctorView::now().expect("Clock"),
            )
            .expect("Pure correlation");
            assert_eq!(correlation.code_groups.len(), 1);
            let group = correlation
                .code_groups
                .first()
                .expect("Duplicate evidence group");
            assert_eq!(
                group.alert_numbers.len(),
                2,
                "All duplicate occurrences retain provenance"
            );
            assert_eq!(
                group.checkout_relevance,
                crate::doctor_dependencies::GithubCheckoutRelevance::BranchMismatch
            );
            assert!(
                group.matching_analysis_ids.is_empty(),
                "Denied analyses cannot prove scanner coverage"
            );
            assert!(view.report.is_some());
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed outcome")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Incomplete
            );
        });
        click_investigation_control(&view, &mut visual, "doctor-github-code-repair-1");
        view.read_with(&visual, |view, _| {
            let OwnerRepair::Native(review) = view
                .proposal
                .as_ref()
                .expect("Explicit selected bounded source proposal")
            else {
                panic!("Wrong owner");
            };
            let prompt = review.action().prompt();
            assert!(prompt.contains("BranchMismatch") && prompt.contains("Partial"));
            assert!(
                !prompt.contains("RAW_SECRET")
                    && !prompt.contains("RAW_SOURCE")
                    && !prompt.contains("RAW_API_ERROR")
            );
        });
        click_investigation_control(&view, &mut visual, "doctor-cancel-repair");
        assert!(view.read_with(&visual, |view, _| view.report.is_some()
            && view.proposal.is_none()));
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_security_setup_review_is_unpublished_cancelled_and_authority_bound(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (_, view, mut visual, calls, credential, history) =
            doctor_github_ui_fixture(cx, false).await;
        let (model, _) = crate::doctor_agent::tests::doctor_probe_fixture(cx);
        model.forbid_requests();
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security_setup(window, cx)
        });
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.error.is_some()));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-github-source-0");
        click_investigation_control(&view, &mut visual, "doctor-github-source-1");
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        click_investigation_control(&view, &mut visual, "doctor-security-setup-open");
        click_investigation_control(&view, &mut visual, "doctor-setup-review");
        view.read_with(&visual, |view, _| {
            let OwnerRepair::Native(review) =
                view.proposal.as_ref().expect("Unpublished setup review")
            else {
                panic!("Wrong owner");
            };
            let prompt = review.action().prompt();
            for expected in [
                ".github/dependabot.yml",
                "Unknown",
                "cost",
                "trigger",
                "Separate explicit",
                "Do not",
            ] {
                assert!(
                    prompt.contains(expected),
                    "Missing setup disclosure {expected}"
                );
            }
            assert!(!prompt.contains("FIXTURE_GITHUB_TOKEN"));
            assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Proposed);
        });
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "Setup review has no additional network or model effects"
        );
        click_investigation_control(&view, &mut visual, "doctor-cancel-repair");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.report.is_some()));
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Receipts")
                .len(),
            1,
            "Cancelled unpublished review has no execution receipt"
        );
        click_investigation_control(&view, &mut visual, "doctor-setup-kind-1");
        click_investigation_control(&view, &mut visual, "doctor-setup-review");
        assert!(view.read_with(&visual, |view, _| {
            view.proposal.is_none()
                && view
                    .error
                    .as_ref()
                    .expect("Missing existing workflow")
                    .as_str()
                    .contains("existing project scanning workflow")
        }));
        click_investigation_control(&view, &mut visual, "doctor-setup-kind-0");
        click_investigation_control(&view, &mut visual, "doctor-setup-review");
        *credential.lock().expect("Credential fixture") = None;
        view.read_with(&visual, |view, cx| {
            let OwnerRepair::Native(review) = view.proposal.as_ref().expect("Current review")
            else {
                panic!("Wrong owner");
            };
            assert!(
                review.action().revalidate(cx).is_err(),
                "Revocation rejects source setup handoff"
            );
        });
        view.update(&mut visual, |view, cx| view.cancel_repair(cx));
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_security_setup_validation_rejects_unsafe_inputs_and_preserves_unknowns(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (_, view, mut visual, calls, _, _) = doctor_github_ui_fixture(cx, false).await;
        use crate::doctor_agent::DoctorSecuritySetupKind;
        let original = view
            .read_with(&visual, |view, cx| view.security_setup.request(cx))
            .expect("Bounded initial proposal");
        let preview = original
            .exact_preview(true)
            .expect("Unknown private eligibility disclosed")
            .join("\n");
        assert!(
            preview.contains("unknown")
                && preview.contains("requires separate concrete approval")
                && preview.contains("private: true")
        );
        for path in [
            "../.github/dependabot.yml",
            ".github/workflows/../security.yml",
            ".github/workflows/a/b.yml",
            "https://example.invalid/setup.yml",
            ".github/workflows/evil\\path.yml",
        ] {
            let mut request = original.clone();
            request.kind = DoctorSecuritySetupKind::ExistingScanningWorkflow;
            request.configuration_path = path.into();
            assert!(request.validate().is_err(), "Rejected path {path}");
        }
        for value in [
            "GITHUB_TOKEN=DO_NOT_HANDOFF",
            "github_token: DO_NOT_HANDOFF",
            "authorization: DO_NOT_HANDOFF",
            "Bearer DO_NOT_HANDOFF",
            "ghp_DO_NOT_HANDOFF",
            "github_pat_DO_NOT_HANDOFF",
            "https://user:password@example.invalid",
            "https://example.invalid?token=DO_NOT_HANDOFF",
            "sk-DO_NOT_HANDOFF",
            "line\nnext",
            "",
        ] {
            let mut request = original.clone();
            request.credential_permissions = value.into();
            assert!(request.validate().is_err(), "Rejected unsafe setup input");
        }
        let mut request = original.clone();
        request.instructions = "x".repeat(4097);
        assert!(request.validate().is_err());
        let mut request = original;
        request.kind = DoctorSecuritySetupKind::ExistingScanningWorkflow;
        request.configuration_path = ".github/workflows/existing-security.yaml".into();
        assert!(request.validate().is_ok());
        view.update_in(&mut visual, |view, window, cx| {
            let field = view
                .security_setup
                .fields
                .iter()
                .find(|(id, _, _)| *id == "doctor-setup-credentials")
                .expect("Existing input")
                .2
                .clone();
            field.update(cx, |input, cx| {
                input.set_text("GITHUB_TOKEN=DO_NOT_HANDOFF", window, cx)
            });
            view.review_security_setup(window, cx);
        });
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "Invalid setup never reads remote evidence or submits a model prompt"
        );
    }
    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_github_view_source_proposal_preserves_provenance_and_current_authority(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (project, view, mut visual, calls, credential, _) =
            doctor_github_ui_fixture(cx, false).await;
        let (model, _) = crate::doctor_agent::tests::doctor_probe_fixture(cx);
        model.forbid_requests();
        let old_timestamp = time::OffsetDateTime::from_unix_timestamp(
            i64::try_from(DoctorView::now().expect("Clock").saturating_sub(172_800))
                .expect("Fixture timestamp"),
        )
        .expect("Fixture time")
        .format(&time::format_description::well_known::Rfc3339)
        .expect("Fixture timestamp format");
        project.read_with(&visual, |project, _| project.client().http_client()).as_fake().replace_handler({
            let calls = calls.clone();
            move |_, request| { let calls = calls.clone(); let old_timestamp = old_timestamp.clone(); async move {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request.method(), http_client::Method::GET);
                assert_eq!(request.extensions().get::<http_client::RedirectPolicy>(), Some(&http_client::RedirectPolicy::NoFollow));
                let path = request.uri().path();
                let value = if path == "/user" { serde_json::json!({"id":42,"login":"selected-user"}) }
                    else if path == "/repos/selected-owner/selected-repository" { serde_json::json!({"id":123,"full_name":"selected-owner/selected-repository","private":true,"default_branch":"dev"}) }
                    else if path.ends_with("/code-scanning/alerts") { serde_json::json!([{
                        "number":2,"state":"open","rule":{"id":"rust/example","severity":"warning","security_severity_level":"high","description":"RAW_SOURCE_NOT_HANDOFF"},
                        "tool":{"name":"Selected Scanner","version":"1.2.3"},"created_at":old_timestamp,"updated_at":old_timestamp,
                        "most_recent_instance":{"ref":"refs/heads/other","commit_sha":"b".repeat(40),"location":{"path":"source.txt","start_line":1,"end_line":1},"message":{"text":"RAW_SECRET_NOT_HANDOFF"}},
                        "url":"https://api.github.com/repos/selected-owner/selected-repository/code-scanning/alerts/2",
                        "html_url":"https://github.com/selected-owner/selected-repository/security/code-scanning/2"
                    }]) }
                    else if path.ends_with("/code-scanning/analyses") { serde_json::json!([{
                        "id":3,"ref":"refs/heads/other","commit_sha":"b".repeat(40),"created_at":old_timestamp,"results_count":1,"rules_count":0,
                        "tool":{"name":"Selected Scanner","version":"1.2.3"},"error":"RAW_ERROR_NOT_HANDOFF","warning":"","environment":"RAW_CREDENTIAL_NOT_HANDOFF",
                        "url":"https://api.github.com/repos/selected-owner/selected-repository/code-scanning/analyses/3"
                    }]) }
                    else { panic!("Unexpected source endpoint {path}"); };
                Ok(http_client::Response::builder().status(200).header("Content-Type", "application/json").body(serde_json::to_vec(&value)?.into())?)
            }}
        });
        click_investigation_control(&view, &mut visual, "doctor-github-source-0");
        click_investigation_control(&view, &mut visual, "doctor-github-source-2");
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            let snapshot = view
                .github
                .snapshot
                .as_ref()
                .expect("Observed exact source");
            let finding = snapshot
                .code_scanning
                .observations
                .first()
                .expect("Finding");
            assert_eq!(
                snapshot
                    .checkout
                    .relevance(finding.reference.as_deref(), finding.commit.as_deref()),
                crate::doctor_dependencies::GithubCheckoutRelevance::BranchMismatch
            );
            let analysis = snapshot
                .analyses
                .observations
                .first()
                .expect("Analysis provenance");
            assert_eq!(
                analysis.rules_count, None,
                "Zero rules remain unknown coverage"
            );
            assert!(analysis.observed_error);
            assert!(
                crate::doctor_dependencies::GithubEvidenceSnapshot::source_freshness(
                    analysis.created_at.as_deref(),
                    DoctorView::now().expect("Clock")
                )
                .as_str()
                .contains("Stale")
            );
        });
        click_investigation_control(&view, &mut visual, "doctor-github-code-repair-2");
        view.read_with(&visual, |view, _| {
            let OwnerRepair::Native(review) = view
                .proposal
                .as_ref()
                .expect("Current selected NativeAgent proposal")
            else {
                panic!("Wrong owner");
            };
            let prompt = review.action().prompt();
            assert!(
                prompt.contains("BranchMismatch")
                    && prompt.contains("rust/example")
                    && prompt.contains("1.2.3")
            );
            assert!(
                prompt.contains("Do not upgrade dependencies") && prompt.contains("verification")
            );
            for marker in [
                "RAW_SECRET",
                "RAW_SOURCE",
                "RAW_CREDENTIAL",
                "FIXTURE_GITHUB_TOKEN",
            ] {
                assert!(!prompt.contains(marker));
            }
            assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Proposed);
        });
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "Preparing a source proposal does not reread GitHub or launch a model"
        );
        *credential.lock().expect("Fixture credential") = None;
        view.read_with(&visual, |view, cx| {
            let OwnerRepair::Native(review) = view.proposal.as_ref().expect("Review") else {
                panic!("Wrong owner");
            };
            assert!(
                review.action().revalidate(cx).is_err(),
                "Revoked GitHub authority rejects stale source handoff"
            );
        });
        view.update(&mut visual, |view, cx| view.cancel_repair(cx));
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.report.is_some()));
    }
    #[cfg(feature = "agentic")]
    async fn doctor_github_ui_fixture(
        cx: &mut TestAppContext,
        pending: bool,
    ) -> (
        Entity<Project>,
        Entity<DoctorView>,
        VisualTestContext,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::Mutex<Option<String>>>,
        crate::doctor::DoctorHistoryStore,
    ) {
        use std::sync::{Arc, Mutex, atomic::AtomicUsize};
        let (project, _, _) = crate::doctor_dependencies::doctor_github_project_fixture(cx).await;
        let calls = Arc::new(AtomicUsize::new(0));
        let credential = Arc::new(Mutex::new(Some(
            "FIXTURE_GITHUB_TOKEN_NOT_HISTORY".to_owned(),
        )));
        let history =
            crate::doctor::DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
            cx.set_global(history.clone());
            crate::doctor_dependencies::doctor_github_transport_fixture(
                &project,
                calls.clone(),
                pending,
                cx,
            );
        });
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Actual workspace");
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Native Doctor Item");
        view.update_in(&mut visual, |view, window, cx| {
            view.refresh(cx);
            for (field, value) in [
                (view.github.account.clone(), "selected-user"),
                (view.github.owner.clone(), "selected-owner"),
                (view.github.repository.clone(), "selected-repository"),
                (
                    view.github.target.clone(),
                    "selected project build and affected tests",
                ),
            ] {
                field.update(cx, |editor, cx| editor.set_text(value, window, cx));
            }
            let credential = credential.clone();
            view.github.credential_reader = Some(Arc::new(move || {
                credential.lock().expect("Fixture credential").clone()
            }));
        });
        visual
            .condition(&view, |view, _| view.collection.is_none())
            .await;
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-github-open");
        (project, view, visual, calls, credential, history)
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_github_view_explicit_selection_review_no_early_effects_and_local_availability(
        cx: &mut TestAppContext,
    ) {
        use http_client::github_security::GithubSecurityCoverage;
        use std::sync::atomic::Ordering;
        let (_, view, mut visual, calls, _, history) = doctor_github_ui_fixture(cx, false).await;
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(view.error.is_some());
            assert!(view.report.is_some());
        });
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-github-source-0");
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "Credential selection is not permission to read"
        );
        click_investigation_control(&view, &mut visual, "doctor-github-source-1");
        click_investigation_control(&view, &mut visual, "doctor-github-review");
        view.read_with(&visual, |view, _| {
            let OwnerRepair::Github(review) =
                view.proposal.as_ref().expect("Ordinary reviewed action")
            else {
                panic!("Wrong owner");
            };
            assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Proposed);
            let preview = review
                .action()
                .exact_preview()
                .expect("Exact request preview")
                .join("\n");
            assert!(
                preview.contains("api.github.com/user") && preview.contains("dependabot/alerts")
            );
            assert!(!preview.contains(
                "GET https://api.github.com/repos/selected-owner/selected-repository/code-scanning"
            ));
            assert!(!preview.contains("FIXTURE_GITHUB_TOKEN"));
        });
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        view.update(&mut visual, |view, cx| view.approve_repair(cx));
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            let snapshot = view
                .github
                .snapshot
                .as_ref()
                .expect("Observed selected evidence");
            assert_eq!(
                snapshot.dependabot.coverage,
                GithubSecurityCoverage::Complete
            );
            assert_eq!(
                snapshot.analyses.coverage,
                GithubSecurityCoverage::NotRequested
            );
            assert_eq!(
                view.owner_result.as_ref().expect("Owner result").phase(),
                crate::doctor::DoctorRepairPhase::Incomplete
            );
            assert!(view.report.is_some(), "Local evidence remains available");
        });
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable history");
        assert_eq!(records.len(), 1);
        assert!(
            !serde_json::to_string(&records)
                .expect("Redacted history")
                .contains("FIXTURE_GITHUB_TOKEN")
        );
        view.update_in(&mut visual, |view, window, cx| {
            view.github.account.update(cx, |editor, cx| {
                editor.set_text("other-account", window, cx)
            })
        });
        visual.run_until_parked();
        assert!(
            view.read_with(&visual, |view, _| view.github.snapshot.is_none()
                && view.report.is_some())
        );
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_github_view_cancel_revocation_stale_inputs_and_interrupted_history(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        for scenario in [
            "cancel_proposal",
            "revoke_before",
            "edit_before",
            "cancel_pending",
            "revoke_pending",
        ] {
            let mut application = cx.new_app();
            let (_, view, mut visual, calls, credential, history) =
                doctor_github_ui_fixture(&mut application, true).await;
            click_investigation_control(&view, &mut visual, "doctor-github-source-0");
            click_investigation_control(&view, &mut visual, "doctor-github-source-1");
            click_investigation_control(&view, &mut visual, "doctor-github-review");
            match scenario {
                "cancel_proposal" => view.update(&mut visual, |view, cx| view.cancel_repair(cx)),
                "revoke_before" => {
                    *credential.lock().expect("Fixture credential") = None;
                }
                "edit_before" => view.update_in(&mut visual, |view, window, cx| {
                    view.github.repository.update(cx, |editor, cx| {
                        editor.set_text("other-repository", window, cx)
                    })
                }),
                _ => {}
            }
            visual.run_until_parked();
            if matches!(scenario, "cancel_proposal" | "edit_before") {
                assert!(view.read_with(&visual, |view, _| view.proposal.is_none()));
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                continue;
            }
            click_investigation_control(&view, &mut visual, "doctor-approve-repair");
            visual.run_until_parked();
            if scenario == "revoke_before" {
                visual
                    .condition(&view, |view, _| view.repair_task.is_none())
                    .await;
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert!(
                    history
                        .records(compiled_product(), DoctorView::now().expect("Clock"))
                        .await
                        .expect("No effect")
                        .is_empty()
                );
                continue;
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            if scenario == "cancel_pending" {
                let bounds = visual
                    .debug_bounds("doctor-cancel-repair")
                    .expect("Fixed cancel control");
                visual.simulate_click(bounds.center(), gpui::Modifiers::default());
                visual.run_until_parked();
            } else {
                *credential.lock().expect("Fixture credential") = None;
                visual
                    .executor()
                    .advance_clock(std::time::Duration::from_millis(100));
            }
            visual
                .condition(&view, |view, _| view.repair_task.is_none())
                .await;
            assert_eq!(calls.load(Ordering::SeqCst), 1, "No replay");
            view.read_with(&visual, |view, _| {
                assert!(view.github.snapshot.is_none());
                assert!(view.report.is_some());
                assert_ne!(
                    view.owner_result.as_ref().expect("Observed stop").phase(),
                    crate::doctor::DoctorRepairPhase::Verified
                );
            });
            assert_eq!(
                history
                    .records(compiled_product(), DoctorView::now().expect("Clock"))
                    .await
                    .expect("Reached receipt")
                    .len(),
                1
            );
        }
    }
    use super::*;
    use gpui::{TestAppContext, VisualTestContext};

    #[cfg(all(feature = "agentic", feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_flow_native_collection_owner_admission_and_cancel(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor::*;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        for jvm in [false, true] {
            let (project, _directory, window) = if jvm {
                crate::doctor::tests::doctor_jvm_collection_fixture(cx).await
            } else {
                crate::doctor::tests::doctor_cargo_collection_fixture(cx).await
            };
            let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
            cx.update(|cx| {
                editor::init(cx);
                crate::init(cx);
                cx.set_global(history.clone());
            });
            let workspace = window
                .read_with(cx, |multi, _| multi.workspace().clone())
                .expect("Actual workspace");
            let mut visual = VisualTestContext::from_window(window.into(), cx);
            let view = workspace.update_in(&mut visual, |workspace, window, cx| {
                DoctorView::deploy(workspace, &Doctor, window, cx);
                workspace
                    .item_of_type::<DoctorView>(cx)
                    .expect("Native Doctor Item")
            });
            view.update_in(&mut visual, |view, window, cx| {
                view.selected_product = if jvm {
                    DoctorProduct::Groovy
                } else {
                    DoctorProduct::Copper
                };
                view.refresh(cx);
                view.configure_dependency_security_inputs_fixture(
                    jvm,
                    "selected project build and affected tests",
                    "",
                    window,
                    cx,
                );
                if jvm {
                    view.security.kind = SecurityKind::Gradle;
                }
                view.security.output.update(cx, |editor, cx| {
                    editor.set_text("doctor-owner-report.json", window, cx)
                });
            });
            visual
                .condition(&view, |view, _| view.collection.is_none())
                .await;
            visual.run_until_parked();
            let (started, launches) = async_channel::bounded(2);
            let (completion, results) = async_channel::bounded(2);
            let cancelled = Arc::new(AtomicUsize::new(0));
            workspace.update(&mut visual, |workspace, _| {
                workspace.set_terminal_provider(crate::doctor::tests::DoctorTaskFixture {
                    started,
                    completion: results,
                    cancel_completion: completion.clone(),
                    cancellation_count: cancelled.clone(),
                })
            });
            click_investigation_control(
                &view,
                &mut visual,
                "doctor-investigation-development-only",
            );
            click_investigation_control(&view, &mut visual, "doctor-security-open");
            click_investigation_control(&view, &mut visual, "doctor-security-collect");
            assert!(
                launches.is_empty(),
                "Collection preview is not execution permission"
            );
            view.read_with(&visual, |view, _| {
                let preview = view
                    .proposal
                    .as_ref()
                    .unwrap_or_else(|| panic!("Exact collection review: {:?}", view.error))
                    .exact_preview()
                    .expect("Preview")
                    .join("\n");
                assert!(
                    preview.contains("offline") && preview.contains("doctor-owner-report.json")
                );
            });
            click_investigation_control(&view, &mut visual, "doctor-approve-repair");
            let launch = launches.recv().await.expect("Existing structured Task");
            assert_eq!(launch.save, task::SaveStrategy::None);
            let raw = if jvm {
                include_str!(
                    "../../project/test_data/groovy_jvm_gradle/doctor-selected-dependencies-v1.json"
                )
                .to_owned()
            } else {
                include_str!("../../project/test_data/cargo_workspace/workspace-v1.json")
                    .replace("/workspace", "/doctor-restart")
            };
            let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
            filesystem
                .atomic_write(
                    std::path::PathBuf::from("/doctor-restart/doctor-owner-report.json"),
                    raw,
                )
                .await
                .expect("Only the approved fixture Task report");
            visual.run_until_parked();
            completion.send(0).await.expect("Observed Task exit");
            visual
                .condition(&view, |view, _| view.repair_task.is_none())
                .await;
            view.read_with(&visual, |view, cx| {
                let projection = view.security.projection.as_ref().unwrap_or_else(|| {
                    panic!(
                        "Actual captured report owner admission (JVM {jvm}): error {:?}, outcome {:?}",
                        view.error,
                        view.owner_result.as_ref().and_then(OwnerRepair::after)
                    )
                });
                projection
                    .revalidate(&project, cx)
                    .expect("Current owner graph");
                assert!(!projection.graph().nodes().is_empty());
                assert_ne!(
                    view.owner_result
                        .as_ref()
                        .expect("Observed collection")
                        .phase(),
                    DoctorRepairPhase::Verified,
                    "Successful collection never verifies an application repair"
                );
                assert!(
                    view.task_result.is_some(),
                    "Existing Task navigation handle retained"
                );
            });
            click_investigation_control(&view, &mut visual, "doctor-security-inventory");
            let output = view.read_with(&visual, |view, _| view.security.output.clone());
            output.update_in(&mut visual, |editor, window, cx| {
                editor.set_text("doctor-cancelled-report.json", window, cx)
            });
            visual.run_until_parked();
            click_investigation_control(&view, &mut visual, "doctor-security-collect");
            click_investigation_control(&view, &mut visual, "doctor-approve-repair");
            launches
                .recv()
                .await
                .expect("Separately approved second collection");
            let cancel_bounds = visual
                .debug_bounds("doctor-cancel-repair")
                .expect("Fixed native progress cancellation control");
            assert!(
                visual
                    .debug_bounds("made-doctor")
                    .expect("Native Doctor viewport")
                    .contains(&cancel_bounds.center())
            );
            visual.simulate_click(cancel_bounds.center(), gpui::Modifiers::default());
            visual
                .condition(&view, |view, _| view.repair_task.is_none())
                .await;
            assert_eq!(cancelled.load(Ordering::SeqCst), 1);
            assert!(
                !filesystem
                    .is_file(std::path::Path::new(
                        "/doctor-restart/doctor-cancelled-report.json"
                    ))
                    .await
            );
            let records = history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Bounded durable owner receipts");
            assert_eq!(records.len(), 2);
            assert_eq!(
                records.last().expect("Cancellation").outcome,
                ReceiptOutcome::Cancelled
            );
            assert!(
                records
                    .iter()
                    .all(|record| record.outcome != ReceiptOutcome::Completed)
            );
            assert!(
                !serde_json::to_string(&records)
                    .expect("Redacted history")
                    .contains("workspace_members")
            );
            assert_eq!(
                filesystem
                    .load(std::path::Path::new("/doctor-restart/source.rs"))
                    .await
                    .expect("Unrelated source"),
                "unrelated"
            );
        }
    }

    #[cfg(all(feature = "agentic", feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_flow_groovy_offline_partial_cache_cancellation_and_revocation(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor_dependencies::*;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        filesystem.as_fake().insert_tree("/doctor-restart",serde_json::json!({
            "pom.xml":"<project><groupId>evidence</groupId><artifactId>doctor-report</artifactId><version>1.0</version></project>",
            "src":{"Main.java":"unrelated application source"}
        })).await;
        visual.run_until_parked();
        let root = project.read_with(&visual, |project, cx| {
            project
                .find_project_path(std::path::Path::new("/doctor-restart"), cx)
                .expect("Selected project")
        });
        let store = project.read_with(&visual, |project, _| project.lsp_store());
        let inputs = store
            .read_with(&visual, |store, cx| {
                store.prepare_reviewed_jvm_dependency_import(
                    root,
                    project::jvm_project::JvmReviewedDependencyFormat::MavenSelectedTree,
                    cx,
                )
            })
            .expect("Existing owner plan");
        store.update(&mut visual,|store,cx|store.admit_reviewed_jvm_dependency_import(inputs,include_bytes!("../../project/test_data/groovy_jvm_maven/doctor-selected-dependencies-v1.json").to_vec(),cx)).await.expect("Existing reviewed graph admission");
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_dependency_security_inputs_fixture(
                true,
                "selected Maven tests",
                "org.junit.jupiter:junit-jupiter-api",
                window,
                cx,
            )
        });
        visual.run_until_parked();
        let archive = DoctorView::dependency_security_archive_fixture(true, false)
            .await
            .expect("Official-format fixture");
        let withdrawn = DoctorView::dependency_security_archive_fixture(true, true)
            .await
            .expect("Withdrawn fixture");
        let requests = Arc::new(AtomicUsize::new(0));
        let offline = Arc::new(AtomicBool::new(false));
        project.read_with(&visual,|project,_|project.client().http_client()).as_fake().replace_handler({
            let requests=requests.clone();let offline=offline.clone();
            move |_,request| {let requests=requests.clone();let offline=offline.clone();let archive=archive.clone();let withdrawn=withdrawn.clone();async move {
                requests.fetch_add(1,Ordering::SeqCst);
                if offline.load(Ordering::SeqCst) {anyhow::bail!("RAW_SECRET_NETWORK_FAILURE");}
                let body=match request.uri().to_string().as_str() {
                    "https://storage.googleapis.com/osv-vulnerabilities/Maven/all.zip"=>archive,
                    "https://storage.googleapis.com/osv-vulnerabilities/%5BEMPTY%5D/all.zip"=>withdrawn,
                    _=>anyhow::bail!("Unexpected destination"),
                };
                assert!(request.headers().get("Authorization").is_none());
                Ok(http_client::Response::builder().status(200).header("Date",chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()).header("Last-Modified",chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()).body(http_client::AsyncBody::from(body))?)
            }}
        });
        click_investigation_control(&view, &mut visual, "doctor-security-open");
        click_investigation_control(&view, &mut visual, "doctor-security-inventory");
        let selected = view
            .read_with(&visual, |view, _| {
                view.dependency_security_instance_fixture("org.junit.jupiter:junit-jupiter-api")
            })
            .expect("Resolved selected component");
        let selector: &'static str =
            Box::leak(format!("doctor-security-instance-{selected}").into_boxed_str());
        click_investigation_control(&view, &mut visual, selector);
        click_investigation_control(&view, &mut visual, "doctor-security-osv-refresh");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-cancel-repair");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()));
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-security-osv-refresh");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        let revision = view.read_with(&visual, |view, _| {
            view.security
                .snapshot
                .as_ref()
                .expect("Observed retained data")
                .sources
                .clone()
        });
        click_investigation_control(&view, &mut visual, "doctor-security-scan");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            let scan = view
                .security
                .scan
                .as_ref()
                .expect("Observed partial matching");
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert!(
                scan.findings.iter().all(|finding| finding.evidence.state
                    != DependencyAdvisoryState::KnownAffectedDependency),
                "Unknown artifact source never proves a public artifact match"
            );
            assert_eq!(
                view.security
                    .projection
                    .as_ref()
                    .expect("Inventory")
                    .graph()
                    .nodes()
                    .get(selected)
                    .expect("Component")
                    .source
                    .kind(),
                DependencySourceKind::Unknown
            );
        });
        offline.store(true, Ordering::SeqCst);
        click_investigation_control(&view, &mut visual, "doctor-security-osv-refresh");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.security
                    .snapshot
                    .as_ref()
                    .expect("Offline failure preserves previous evidence")
                    .sources,
                revision
            )
        });
        click_investigation_control(&view, &mut visual, "doctor-security-osv-refresh");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"tab_size":7}"#, cx)
                    .expect("Changed authority")
            })
        });
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            3,
            "Changed permission/configuration authority blocks stale review"
        );
        assert!(view.read_with(&visual, |view, _| view.error.is_some()));
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Durable bounded receipts");
        let persisted = serde_json::to_string(&records).expect("Redacted history");
        assert!(
            !persisted.contains("PUBLIC_BODY_NOT_HISTORY")
                && !persisted.contains("RAW_SECRET_NETWORK_FAILURE")
                && !persisted.contains("Main.java")
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/src/Main.java"))
                .await
                .expect("Preserved application"),
            "unrelated application source"
        );
    }

    #[cfg(all(feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_view_passive_missing_inventory_and_invalid_collection(
        cx: &mut TestAppContext,
    ) {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let requests = Arc::new(AtomicUsize::new(0));
        project
            .read_with(&visual, |project, _| project.client().http_client())
            .as_fake()
            .replace_handler({
                let requests = requests.clone();
                move |_, _| {
                    requests.fetch_add(1, Ordering::SeqCst);
                    async { anyhow::bail!("No implicit HTTP") }
                }
            });
        let processes = project::cargo_workspace_store::local_cargo_process_calls_for_test();
        click_investigation_control(&view, &mut visual, "doctor-security-open");
        assert!(view.read_with(&visual, |view, _| view.security.open));
        click_investigation_control(&view, &mut visual, "doctor-security-inventory");
        view.read_with(&visual, |view, _| {
            assert!(view.security.projection.is_none());
            assert!(
                view.error.is_some(),
                "Missing owner graph is incomplete, not zero dependencies"
            );
            assert!(
                view.preparation.is_none() && view.repair_task.is_none() && view.proposal.is_none()
            );
        });
        click_investigation_control(&view, &mut visual, "doctor-security-collect");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.error.is_some()));
        let output = view.read_with(&visual, |view, _| view.security.output.clone());
        output.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("../outside.json", window, cx)
        });
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-security-collect");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.repair_task.is_none()));
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            processes
        );
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("History")
                .is_empty()
        );
        let fs = project.read_with(&visual, |project, _| project.fs().clone());
        assert!(!fs.is_file(std::path::Path::new("/outside.json")).await);
        visual.simulate_keystrokes("tab");
        visual.run_until_parked();
        assert!(visual.debug_bounds("made-doctor").is_some());
    }

    #[cfg(all(feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_view_review_cancel_stale_clear_and_redacted_history(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor_dependencies::*;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let requests = Arc::new(AtomicUsize::new(0));
        project
            .read_with(&visual, |project, _| project.client().http_client())
            .as_fake()
            .replace_handler({
                let requests = requests.clone();
                move |_, _| {
                    requests.fetch_add(1, Ordering::SeqCst);
                    async { anyhow::bail!("Not authorized to fetch in this fixture") }
                }
            });
        view.update(&mut visual, |view, cx| {
            view.security.open = true;
            view.security.kind = SecurityKind::Maven;
            cx.notify();
        });
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security(
                SecurityAction::Refresh(AdvisoryRefreshRequest::OsvMaven),
                window,
                cx,
            )
        });
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Prepared review");
        }
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            let proposal = view.proposal.as_ref().expect("Concrete reviewed source");
            assert_eq!(proposal.phase(), crate::doctor::DoctorRepairPhase::Proposed);
            assert!(
                proposal
                    .exact_preview()
                    .expect("Exact effects")
                    .join("\n")
                    .contains("storage.googleapis.com")
            );
            assert!(view.repair_task.is_none());
        });
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        click_investigation_control(&view, &mut visual, "doctor-cancel-repair");
        assert!(view.read_with(&visual, |view, _| view.proposal.is_none()
            && view.repair_task.is_none()));
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security(
                SecurityAction::Refresh(AdvisoryRefreshRequest::OsvMaven),
                window,
                cx,
            )
        });
        while view.read_with(&visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Fresh review");
        }
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{\"tab_size\":3}", cx)
                    .expect("Changed inputs")
            })
        });
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        assert!(view.read_with(&visual, |view, _| view.repair_task.is_none()
            && view.error.is_some()));
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("Unapproved history")
                .is_empty()
        );
        fill_investigation_form(&view, &mut visual);
        view.update(&mut visual, |view, cx| {
            let (directory, selection, _, _) = view
                .investigation_selection(cx)
                .expect("Current selected target");
            let binding = crate::doctor::project_repair_binding(
                &view.project,
                directory,
                view.generation,
                cx,
            )
            .expect("Current authority");
            let scope = view.report.as_ref().expect("Current report").scope.clone();
            view.current_comparison = Some(
                crate::doctor::DoctorInvestigation::new(
                    selection,
                    scope,
                    binding,
                    DoctorView::now().expect("Time"),
                )
                .expect("Selected investigation")
                .comparison_snapshot_at(&view.project, cx)
                .expect("Sealed target"),
            );
        });
        let target_identity = view.read_with(&visual, |view, _| {
            view.current_comparison
                .as_ref()
                .expect("Selected target")
                .selection()
                .input_identity
                .clone()
        });
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security(
                SecurityAction::Refresh(AdvisoryRefreshRequest::Clear {
                    dataset: AdvisoryDataset::OsvMaven,
                }),
                window,
                cx,
            )
        });
        while view.read_with(&visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Clear review");
        }
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        while view.read_with(&visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Observed absence");
        }
        view.read_with(&visual, |view, _| {
            assert!(view.security.snapshot.is_none() && view.security.scan.is_none());
            assert_eq!(
                view.owner_result.as_ref().expect("Owner result").phase(),
                crate::doctor::DoctorRepairPhase::Verified
            );
            assert_eq!(
                view.current_comparison
                    .as_ref()
                    .expect("Unrelated scan preserves selected evidence")
                    .selection()
                    .input_identity,
                target_identity
            );
        });
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Explicit action history");
        assert_eq!(records.len(), 1);
        let persisted = serde_json::to_string(&records).expect("Bounded history");
        assert!(!persisted.contains("Authorization") && !persisted.contains("token"));
        assert!(persisted.contains("clear"));
    }

    #[cfg(all(feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_dependency_view_current_owner_paths_reviewed_catalog_and_stale_selection(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor_dependencies::*;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        filesystem.as_fake().insert_tree("/doctor-restart", serde_json::json!({
            "Cargo.toml":"[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n",
            "member-one":{"Cargo.toml":"[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n", "src":{"lib.rs":"preserved source"}},
            "member-two":{"Cargo.toml":"[package]\nname = \"member-two\"\nversion = \"0.2.0\"\n", "src":{"lib.rs":"other source"}}
        })).await;
        visual.run_until_parked();
        let tests = project.read_with(&visual, |project, _| {
            project.rust_test_provider_store().clone()
        });
        let pending = tests.update(&mut visual, |tests, cx| tests.refresh(cx));
        drop(pending);
        visual.run_until_parked();
        let manifest = project.read_with(&visual, |project, cx| {
            project
                .find_project_path(std::path::Path::new("/doctor-restart/Cargo.toml"), cx)
                .expect("Existing manifest")
        });
        let store = project.read_with(&visual, |project, _| {
            project.cargo_workspace_store().clone()
        });
        let inputs = store
            .read_with(&visual, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("Owner import review");
        inputs
            .preflight_filesystem(filesystem.as_ref())
            .await
            .expect("Containment");
        let bytes = String::from_utf8_lossy(include_bytes!(
            "../../project/test_data/cargo_workspace/workspace-v1.json"
        ))
        .replace("/workspace", "/doctor-restart");
        let mut metadata: serde_json::Value =
            serde_json::from_str(&bytes).expect("Actual Cargo fixture");
        let mut package = metadata["packages"][1].clone();
        let identity = "registry+https://github.com/rust-lang/crates.io-index#public-example@1.0.0";
        package["name"] = "public-example".into();
        package["version"] = "1.0.0".into();
        package["id"] = identity.into();
        package["source"] = "registry+https://github.com/rust-lang/crates.io-index".into();
        package["manifest_path"] = "/cargo-cache/public-example/Cargo.toml".into();
        metadata["packages"]
            .as_array_mut()
            .expect("Packages")
            .push(package);
        metadata["resolve"]["nodes"]
            .as_array_mut()
            .expect("Nodes")
            .push(serde_json::json!({"id":identity,"dependencies":[],"deps":[],"features":[]}));
        metadata["resolve"]["nodes"][0]["deps"].as_array_mut().expect("Introducing edges").push(serde_json::json!({"name":"public_example","pkg":identity,"dep_kinds":[{"kind":null,"target":null}]}));
        store
            .update(&mut visual, |store, cx| {
                store.admit_reviewed_metadata_import(
                    inputs,
                    serde_json::to_vec(&metadata).expect("Bytes"),
                    cx,
                )
            })
            .await
            .expect("Existing owner admission");
        let requests = Arc::new(AtomicUsize::new(0));
        project.read_with(&visual, |project, _| project.client().http_client()).as_fake().replace_handler({
            let requests = requests.clone();
            move |_, request| { requests.fetch_add(1, Ordering::SeqCst); async move {
                assert_eq!(request.uri().to_string(), "https://crates.io/api/v1/crates/public-example/versions");
                assert_eq!(request.method(), "GET");
                assert!(request.headers().get("Authorization").is_none());
                Ok(http_client::Response::builder().status(200).body(http_client::AsyncBody::from(r#"{"versions":[{"crate":"public-example","num":"1.1.0","yanked":false}],"meta":{"total":1,"next_page":null}}"#))?)
            }}
        });
        click_investigation_control(&view, &mut visual, "doctor-security-open");
        click_investigation_control(&view, &mut visual, "doctor-security-inventory");
        let selected = view.read_with(&visual, |view, _| {
            let graph = view
                .security
                .projection
                .as_ref()
                .expect("Current owner evidence")
                .graph();
            let selected = graph
                .nodes()
                .iter()
                .position(|node| node.name == "public-example")
                .expect("Public instance");
            assert!(
                !graph
                    .introducing_paths(selected)
                    .expect("Introducing paths")
                    .paths
                    .is_empty()
            );
            selected
        });
        view.update(&mut visual, |view, cx| {
            view.security.selected = Some(selected);
            cx.notify();
        });
        // These are ordinary editable fields; editing a proposal must not erase the current inventory.
        let confirmation =
            view.read_with(&visual, |view, _| view.security.public_coordinate.clone());
        confirmation.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("never sent for Cargo", window, cx)
        });
        visual.run_until_parked();
        assert!(
            view.read_with(&visual, |view, _| view.security.projection.is_some()
                && view.security.selected == Some(selected))
        );
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security(
                SecurityAction::Query {
                    confirmed_catalog: false,
                },
                window,
                cx,
            )
        });
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Exact query review");
        }
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        view.read_with(&visual, |view, _| {
            assert!(
                view.proposal
                    .as_ref()
                    .expect("Reviewed public query")
                    .exact_preview()
                    .expect("Exact URL")
                    .join("\n")
                    .contains("public-example/versions")
            )
        });
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        while view.read_with(&visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Observed catalog");
        }
        view.read_with(&visual, |view, _| {
            assert!(view.error.is_none(), "{:?}", view.error);
            let choice = view
                .security
                .recommendations
                .first()
                .expect("Interpreted candidate");
            assert_eq!(choice.evidence.candidate.version, "1.1.0");
            assert_ne!(choice.evidence.kind, DependencyUpdateKind::AdvisoryFix);
            assert!(
                choice
                    .limitations
                    .iter()
                    .any(|detail| detail.as_str().contains("freshness"))
            );
            assert!(view.security.baseline.is_none() && view.security.verification.is_none());
        });
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{\"tab_size\":7}", cx)
                    .expect("Changed authority")
            })
        });
        view.update_in(&mut visual, |view, window, cx| {
            view.review_security(
                SecurityAction::Query {
                    confirmed_catalog: false,
                },
                window,
                cx,
            )
        });
        assert!(view.read_with(&visual, |view, _| view.error.is_some()
            && view.proposal.is_none()));
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "Stale selection cannot disclose again"
        );
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Receipts");
        assert_eq!(records.len(), 1);
        assert!(
            !serde_json::to_string(&records)
                .expect("Bounded summaries")
                .contains("\"versions\"")
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source"),
            "unrelated"
        );
    }

    async fn fixture(
        cx: &mut TestAppContext,
        multiple: bool,
    ) -> (
        Entity<Project>,
        Entity<Workspace>,
        gpui::WindowHandle<workspace::MultiWorkspace>,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .insert_tree("/first", serde_json::json!({"keep.txt":"untouched"}))
            .await;
        filesystem
            .insert_tree("/second", serde_json::json!({"keep.txt":"untouched"}))
            .await;
        let roots = if multiple {
            vec![
                std::path::Path::new("/first"),
                std::path::Path::new("/second"),
            ]
        } else {
            vec![std::path::Path::new("/first")]
        };
        let project = Project::test(filesystem, roots, cx).await;
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Workspace");
        (project, workspace, window)
    }

    async fn restart_ui(
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
        Entity<Workspace>,
    ) {
        let (_project, _directory, window) =
            crate::doctor::tests::application_restart_fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = crate::doctor::DoctorHistoryStore::for_test_database(&database);
        cx.update(|cx| {
            editor::init(cx);
            crate::init(cx);
            cx.set_global(history.clone());
        });
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Affected Workspace");
        workspace.update(cx, |workspace, cx| {
            DoctorView::register(workspace, None, cx);
            cx.notify();
        });
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual
            .refresh()
            .expect("Render registered Workspace action");
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Native Doctor");
        let collect = visual
            .debug_bounds("doctor-refresh")
            .expect("Selected collection control");
        visual.simulate_click(collect.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.collection.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Scoped collection");
        }
        visual
            .refresh()
            .expect("Completed collection and disclosure");
        let advanced = visual
            .debug_bounds("doctor-advanced")
            .expect("Advanced action disclosure");
        visual.simulate_click(advanced.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        (view, visual, history, workspace)
    }

    async fn prepare_restart_ui(
        view: &Entity<DoctorView>,
        visual: &mut VisualTestContext,
        remote: bool,
    ) {
        view.update(visual, |_, cx| cx.notify());
        visual.run_until_parked();
        click_investigation_control(
            view,
            visual,
            if remote {
                "doctor-review-remote-restart"
            } else {
                "doctor-review-application-restart"
            },
        );
        let mut notifications = visual.notifications(view);
        while view.read_with(visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Review remains available");
        }
        view.update(visual, |_, cx| cx.notify());
        visual.run_until_parked();
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_normal_application_requires_approval_and_remains_incomplete(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        let mut restart = cx.expect_restart();
        prepare_restart_ui(&view, &mut visual, false).await;
        view.read_with(&visual, |view, _| {
            let review = view.proposal.as_ref().expect("Exact reviewed scope");
            assert!(matches!(review, OwnerRepair::ApplicationRestart(_)));
            assert!(review.target().contains("1 affected workspaces"));
        });
        assert!(restart.try_recv().expect("No implicit restart").is_none());
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Explicit restart consent");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Normal lifecycle observation");
        }
        restart
            .await
            .expect("Existing normal application restart requested");
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed requested outcome")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Incomplete
            )
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable restart receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Incomplete
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_cancelled_proposal_never_restarts_or_records(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        let mut restart = cx.expect_restart();
        prepare_restart_ui(&view, &mut visual, false).await;
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Native proposal cancel");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(restart.try_recv().expect("No restart").is_none());
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No consent receipt")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_dirty_workspace_blocks_without_save_or_discard(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        workspace.update_in(&mut visual, |workspace, window, cx| {
            let dirty = cx.new(|cx| workspace::item::test::TestItem::new(cx).with_dirty(true));
            workspace.add_item_to_active_pane(Box::new(dirty), None, false, window, cx);
            workspace.activate_item(&view, true, true, window, cx);
        });
        prepare_restart_ui(&view, &mut visual, false).await;
        view.read_with(&visual, |view, _| {
            assert!(
                view.error
                    .as_ref()
                    .expect("Actionable dirty-file denial")
                    .as_str()
                    .contains("dirty")
            );
            assert!(view.proposal.is_none());
        });
        assert!(
            !cx.has_pending_prompt(),
            "Review does not implicitly save or discard edits"
        );
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No effects")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_local_project_never_substitutes_remote_host(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        prepare_restart_ui(&view, &mut visual, true).await;
        view.read_with(&visual, |view, _| {
            assert!(view.error.is_some());
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No host mutation")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_changed_settings_reject_old_approval(cx: &mut TestAppContext) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        let mut restart = cx.expect_restart();
        prepare_restart_ui(&view, &mut visual, false).await;
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(r#"{"confirm_quit":true}"#, cx)
                    .expect("Changed reviewed inputs");
            });
        });
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Old exact review");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(restart.try_recv().expect("No restart").is_none());
        view.read_with(&visual, |view, _| {
            assert!(view.error.is_some());
            assert!(view.repair_task.is_none());
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No durable action")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_new_dirty_item_denies_approved_restart(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let mut restart = cx.expect_restart();
        prepare_restart_ui(&view, &mut visual, false).await;
        workspace.update_in(&mut visual, |workspace, window, cx| {
            let dirty = cx.new(|cx| workspace::item::test::TestItem::new(cx).with_dirty(true));
            workspace.add_item_to_active_pane(Box::new(dirty), None, false, window, cx);
            workspace.activate_item(&view, true, true, window, cx);
        });
        visual.run_until_parked();
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Original exact preview");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Denial delivered to UI");
        }
        assert!(restart.try_recv().expect("No restart").is_none());
        assert!(
            !cx.has_pending_prompt(),
            "Doctor never infers save/discard authority"
        );
        assert!(view.read_with(&visual, |view, _| view.error.is_some()));
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No reached effects")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_restart_flow_normal_confirmation_cancel_is_recorded(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(r#"{"confirm_quit":true}"#, cx)
                    .expect("Normal confirmation retained");
            });
        });
        let mut restart = cx.expect_restart();
        prepare_restart_ui(&view, &mut visual, false).await;
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Reviewed action consent");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(cx.has_pending_prompt());
        cx.simulate_prompt_answer("Cancel");
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Observed normal prompt cancellation");
        }
        assert!(restart.try_recv().expect("No restart").is_none());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Cancelled receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(
            view.read_with(&visual, |view, _| view
                .owner_result
                .as_ref()
                .expect("Observed cancellation")
                .phase()),
            crate::doctor::DoctorRepairPhase::Cancelled
        );
    }

    async fn history_ui(
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
        db::AppDatabase,
    ) {
        let (view, visual, _, _) = restart_ui(cx).await;
        let database = db::AppDatabase::test_new();
        let history = crate::doctor::DoctorHistoryStore::for_test_database(&database);
        cx.update(|cx| cx.set_global(history.clone()));
        (view, visual, history, database)
    }

    async fn history_click(
        view: &Entity<DoctorView>,
        visual: &mut VisualTestContext,
        selector: &'static str,
    ) {
        visual.refresh().expect("Rendered history controls");
        let bounds = visual
            .debug_bounds(selector)
            .expect("Visible native history control");
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        let mut notifications = visual.notifications(view);
        while view.read_with(visual, |view, _| view.history_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Observed history completion");
        }
        visual.refresh().expect("Observed history rendering");
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_explicit_save_cancel_select_delete_preserves_current_checks(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, database) = history_ui(cx).await;
        let current = view.read_with(&visual, |view, _| {
            view.report.clone().expect("Current checks")
        });
        let store = db::kvp::KeyValueStore::from_app_db(&database);
        store
            .scoped("native-session-sentinel")
            .write("keep".into(), "unrelated existing session".into())
            .await
            .expect("Unrelated persisted data");
        history_click(&view, &mut visual, "doctor-history-save").await;
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No implicit record")
                .is_empty()
        );
        history_click(&view, &mut visual, "doctor-history-cancel").await;
        assert!(view.read_with(&visual, |view, _| view.history_change.is_none()));
        history_click(&view, &mut visual, "doctor-history-save").await;
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        view.update(&mut visual, |view, cx| view.approve_history_change(cx));
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Saved summary");
        assert_eq!(records.len(), 1);
        assert!(records.first().expect("Summary").action.is_none());
        history_click(&view, &mut visual, "doctor-history-record-0").await;
        assert_eq!(
            view.read_with(&visual, |view, _| view
                .report
                .as_ref()
                .expect("Current checks")
                .generation),
            current.generation
        );
        history_click(&view, &mut visual, "doctor-history-delete").await;
        history_click(&view, &mut visual, "doctor-history-cancel").await;
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Cancellation preserves record")
                .len(),
            1
        );
        history_click(&view, &mut visual, "doctor-history-delete").await;
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Deleted only selected record")
                .is_empty()
        );
        assert_eq!(
            store
                .scoped("native-session-sentinel")
                .read("keep")
                .expect("Unrelated data"),
            Some("unrelated existing session".into())
        );
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_changed_clear_and_active_records_are_rejected(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = history_ui(cx).await;
        history_click(&view, &mut visual, "doctor-history-open").await;
        history_click(&view, &mut visual, "doctor-history-clear").await;
        let report = view.read_with(&visual, |view, _| view.report.clone().expect("Checks"));
        let now = DoctorView::now().expect("Clock");
        let record = crate::doctor::DoctorHistoryRecord::new(
            report.clone(),
            Some(DiagnosticText::new("Approved exact owner operation")),
            crate::doctor::ReceiptOutcome::Completed,
            now,
        );
        history
            .record(compiled_product(), record, now)
            .await
            .expect("Concurrent record");
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        assert!(view.read_with(&visual, |view, _| {
            view.history_status
                .as_ref()
                .expect("Stale error")
                .as_str()
                .contains("changed after review")
        }));
        assert_eq!(
            history
                .records(compiled_product(), now)
                .await
                .expect("Preserved concurrent record")
                .len(),
            1
        );
        history
            .record(
                compiled_product(),
                crate::doctor::DoctorHistoryRecord::new(
                    report,
                    Some(DiagnosticText::new("Active owner operation")),
                    crate::doctor::ReceiptOutcome::Running,
                    now,
                ),
                now,
            )
            .await
            .expect("Running receipt");
        history_click(&view, &mut visual, "doctor-history-clear").await;
        assert!(view.read_with(&visual, |view, _| {
            view.history_change.is_none()
                && view
                    .history_status
                    .as_ref()
                    .expect("Active refusal")
                    .as_str()
                    .contains("active Doctor")
        }));
        assert_eq!(
            history
                .records(compiled_product(), now)
                .await
                .expect("Active preserved")
                .len(),
            2
        );
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_corruption_explicit_clear_preserves_other_product(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, database) = history_ui(cx).await;
        let now = DoctorView::now().expect("Clock");
        let report = view.read_with(&visual, |view, _| view.report.clone().expect("Checks"));
        let other = if compiled_product() == DoctorProduct::Groovy {
            DoctorProduct::Copper
        } else {
            DoctorProduct::Groovy
        };
        let mut other_report = report;
        other_report.scope.product = DiagnosticText::new(other.name());
        history
            .record(
                other,
                crate::doctor::DoctorHistoryRecord::new(
                    other_report,
                    None,
                    crate::doctor::ReceiptOutcome::Completed,
                    now,
                ),
                now,
            )
            .await
            .expect("Other product");
        let store = db::kvp::KeyValueStore::from_app_db(&database);
        store
            .scoped("made.doctor.v1")
            .write(compiled_product().name().into(), "{invalid".into())
            .await
            .expect("Corrupt fixture");
        history_click(&view, &mut visual, "doctor-history-open").await;
        assert!(view.read_with(&visual, |view, _| {
            view.history_status
                .as_ref()
                .expect("Visible corruption")
                .as_str()
                .contains("not replaced")
        }));
        history_click(&view, &mut visual, "doctor-history-clear").await;
        history_click(&view, &mut visual, "doctor-history-cancel").await;
        assert_eq!(
            store
                .scoped("made.doctor.v1")
                .read(compiled_product().name())
                .expect("Preserved corrupt history"),
            Some("{invalid".into())
        );
        history_click(&view, &mut visual, "doctor-history-clear").await;
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        assert!(
            history
                .records(compiled_product(), now)
                .await
                .expect("Explicit recovered history")
                .is_empty()
        );
        assert_eq!(
            history
                .records(other, now)
                .await
                .expect("Other product preserved")
                .len(),
            1
        );
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_selected_product_matches_retained_partition(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = history_ui(cx).await;
        let other = if compiled_product() == DoctorProduct::Groovy {
            DoctorProduct::Copper
        } else {
            DoctorProduct::Groovy
        };
        view.update(&mut visual, |view, cx| {
            view.selected_product = other;
            view.refresh(cx);
        });
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.collection.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Selected product checks");
        }
        assert_eq!(
            view.read_with(&visual, |view, _| view
                .report
                .as_ref()
                .expect("Checks")
                .scope
                .product
                .as_str()
                .to_owned()),
            compiled_product().name()
        );
        history_click(&view, &mut visual, "doctor-history-save").await;
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Selected partition")
                .len(),
            1
        );
        assert!(
            history
                .records(other, DoctorView::now().expect("Clock"))
                .await
                .expect("Other partition preserved")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_storage_failure_never_reports_saved(cx: &mut TestAppContext) {
        let (view, mut visual, _, database) = history_ui(cx).await;
        history_click(&view, &mut visual, "doctor-history-save").await;
        database
            .0
            .write(|connection| connection.exec("DROP TABLE scoped_kv_store")?())
            .await
            .expect("Simulated storage failure");
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        view.read_with(&visual, |view, _| {
            assert!(view.saved_generation.is_none());
            assert!(
                view.history_status
                    .as_ref()
                    .expect("Visible storage failure")
                    .as_str()
                    .contains("History change failed")
            );
            assert!(view.report.is_some());
        });
    }

    #[gpui::test]
    async fn doctor_ui_history_flow_interruption_is_not_replayed_or_current_verification(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, database) = history_ui(cx).await;
        let now = DoctorView::now().expect("Clock");
        let mut report = view.read_with(&visual, |view, _| {
            view.report.clone().expect("Current checks")
        });
        report.generation = report.generation.saturating_add(100);
        let mut record = crate::doctor::DoctorHistoryRecord::new(
            report,
            Some(DiagnosticText::new("Interrupted approved owner action")),
            crate::doctor::ReceiptOutcome::Running,
            now,
        );
        record.session_id = Some(DiagnosticText::new("existing-session-id"));
        history
            .record(compiled_product(), record, now)
            .await
            .expect("Crash boundary");
        let reopened = crate::doctor::DoctorHistoryStore::for_test_database(&database);
        cx.update(|cx| cx.set_global(reopened));
        history_click(&view, &mut visual, "doctor-history-open").await;
        history_click(&view, &mut visual, "doctor-history-record-0").await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.history_records
                    .first()
                    .expect("Interrupted record")
                    .outcome,
                crate::doctor::ReceiptOutcome::Interrupted
            );
            assert_ne!(
                view.report.as_ref().expect("Current report").generation,
                view.history_records
                    .first()
                    .expect("History")
                    .report
                    .generation
            );
            assert!(view.proposal.is_none() && view.repair_task.is_none());
        });
    }

    async fn export_ui(
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        std::sync::Arc<fs::FakeFs>,
        crate::doctor::DoctorHistoryStore,
    ) {
        let (view, visual, history, _) = history_ui(cx).await;
        let filesystem = view.read_with(&visual, |view, cx| {
            view.workspace
                .read_with(cx, |workspace, _| workspace.app_state().fs.as_fake())
                .expect("Local export Fs owner")
        });
        filesystem
            .insert_tree("/exports", serde_json::json!({}))
            .await;
        (view, visual, filesystem, history)
    }

    fn export_click(
        view: &Entity<DoctorView>,
        visual: &mut VisualTestContext,
        selector: &'static str,
    ) {
        visual.refresh().expect("Native export controls");
        let bounds = visual
            .debug_bounds(selector)
            .expect("Rendered export control");
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        assert!(view.read_with(visual, |view, _| view.export_task.is_some()));
    }

    async fn export_observe(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        let mut notifications = visual.notifications(view);
        while view.read_with(visual, |view, _| view.export_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Observed export result");
        }
        visual.refresh().expect("Rendered export result");
    }

    #[gpui::test]
    async fn doctor_export_native_dialog_cancellation_and_own_control_write_nothing(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        export_click(&view, &mut visual, "doctor-export-current");
        cx.simulate_new_path_selection(|_| None);
        export_observe(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Visible cancellation")
                .as_str()
                .contains("cancelled")
        }));
        export_click(&view, &mut visual, "doctor-export-current");
        let bounds = visual
            .debug_bounds("doctor-export-cancel")
            .expect("Reachable export cancellation");
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/cancelled.md")));
        visual.run_until_parked();
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/cancelled.md")
            )
            .await
            .expect("No export")
            .is_none()
        );
    }

    #[gpui::test]
    async fn doctor_export_native_selected_report_redacts_and_preserves_project(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        view.update(&mut visual, |view, _| {
            view.report.as_mut().expect("Current report").push(DoctorCheck::new("export.security", "Redacted evidence", CheckOutcome::Warning, "API_KEY=export-secret-sentinel https://private.invalid?token=export-url-sentinel /private/source/sensitive.rs", 10)).expect("Bounded selected evidence");
        });
        export_click(&view, &mut visual, "doctor-export-current");
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/report.md")));
        export_observe(&view, &mut visual).await;
        let contents = fs::Fs::load(
            filesystem.as_ref(),
            std::path::Path::new("/exports/report.md"),
        )
        .await
        .expect("Observed local export");
        assert!(contents.contains("[redacted]") && contents.contains("Redacted evidence"));
        for secret in [
            "export-secret-sentinel",
            "export-url-sentinel",
            "private.invalid",
            "/private/source",
        ] {
            assert!(!contents.contains(secret));
        }
        assert!(contents.len() <= crate::doctor::MAX_REPORT_BYTES);
        assert_eq!(
            fs::Fs::load(
                filesystem.as_ref(),
                std::path::Path::new("/doctor-restart/source.rs")
            )
            .await
            .expect("Unrelated source"),
            "unrelated"
        );
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Observed export")
                .as_str()
                .contains("exported")
        }));
    }

    #[gpui::test]
    async fn doctor_export_changed_scope_and_write_failure_never_report_success(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        export_click(&view, &mut visual, "doctor-export-current");
        view.update(&mut visual, |view, cx| view.invalidate(cx));
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/stale.md")));
        export_observe(&view, &mut visual).await;
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/stale.md")
            )
            .await
            .expect("Stale refusal")
            .is_none()
        );
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Stale error")
                .as_str()
                .contains("scope changed")
        }));
        view.update(&mut visual, |view, cx| view.refresh(cx));
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.collection.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Current collection");
        }
        fs::Fs::atomic_write(
            filesystem.as_ref(),
            std::path::PathBuf::from("/exports/blocked-parent"),
            "Preserved ordinary file".into(),
        )
        .await
        .expect("Existing file blocks a child destination");
        export_click(&view, &mut visual, "doctor-export-current");
        cx.simulate_new_path_selection(|_| {
            Some(std::path::PathBuf::from(
                "/exports/blocked-parent/report.md",
            ))
        });
        export_observe(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Write failure")
                .as_str()
                .contains("Export failed")
        }));
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/blocked-parent/report.md")
            )
            .await
            .expect("No success")
            .is_none()
        );
    }

    #[gpui::test]
    async fn doctor_export_freshness_change_rejects_dialog_and_stale_report_is_labelled(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        export_click(&view, &mut visual, "doctor-export-current");
        filesystem
            .insert_tree(
                "/doctor-restart",
                serde_json::json!({"source.rs":"changed fixture input"}),
            )
            .await;
        visual.run_until_parked();
        assert!(view.read_with(&visual, |view, cx| view.report_is_stale(cx)));
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/fresh.md")));
        export_observe(&view, &mut visual).await;
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/fresh.md")
            )
            .await
            .expect("Rejected stale export")
            .is_none()
        );
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Freshness error")
                .as_str()
                .contains("freshness changed")
        }));
        export_click(&view, &mut visual, "doctor-export-current");
        cx.simulate_new_path_selection(|_| {
            Some(std::path::PathBuf::from("/exports/stale-labelled.md"))
        });
        export_observe(&view, &mut visual).await;
        let contents = fs::Fs::load(
            filesystem.as_ref(),
            std::path::Path::new("/exports/stale-labelled.md"),
        )
        .await
        .expect("Explicit stale export");
        assert!(contents.starts_with("Stale report"));
        assert!(contents.contains("Original observations below do not establish current health"));
    }

    #[gpui::test]
    async fn doctor_export_symbolic_link_does_not_overwrite_original_source(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        fs::Fs::create_symlink(
            filesystem.as_ref(),
            std::path::Path::new("/exports/link.md"),
            std::path::PathBuf::from("/doctor-restart/source.rs"),
        )
        .await
        .expect("Selected symlink fixture");
        export_click(&view, &mut visual, "doctor-export-current");
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/link.md")));
        export_observe(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Refusal")
                .as_str()
                .contains("symbolic links")
        }));
        assert_eq!(
            fs::Fs::load(
                filesystem.as_ref(),
                std::path::Path::new("/doctor-restart/source.rs")
            )
            .await
            .expect("Source preserved"),
            "unrelated"
        );
    }

    #[gpui::test]
    async fn doctor_export_selected_history_metadata_and_deleted_selection_refusal(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, history) = export_ui(cx).await;
        history_click(&view, &mut visual, "doctor-history-save").await;
        history_click(&view, &mut visual, "doctor-history-confirm").await;
        history_click(&view, &mut visual, "doctor-history-record-0").await;
        export_click(&view, &mut visual, "doctor-export-history");
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/history.md")));
        export_observe(&view, &mut visual).await;
        let contents = fs::Fs::load(
            filesystem.as_ref(),
            std::path::Path::new("/exports/history.md"),
        )
        .await
        .expect("Historical export");
        assert!(
            contents.contains("Historical record")
                && contents.contains("does not verify current behavior")
        );
        assert!(contents.len() <= crate::doctor::MAX_REPORT_BYTES);
        export_click(&view, &mut visual, "doctor-export-history");
        let record = view.read_with(&visual, |view, _| {
            view.history_records
                .first()
                .expect("Selected record")
                .clone()
        });
        history
            .delete(
                compiled_product(),
                record.id,
                DoctorView::now().expect("Clock"),
            )
            .await
            .expect("Concurrent deletion");
        cx.simulate_new_path_selection(|_| Some(std::path::PathBuf::from("/exports/deleted.md")));
        export_observe(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| {
            view.export_status
                .as_ref()
                .expect("Deleted error")
                .as_str()
                .contains("deleted")
        }));
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/deleted.md")
            )
            .await
            .expect("No hidden export")
            .is_none()
        );
    }

    #[gpui::test]
    async fn doctor_export_invalid_oversized_report_is_rejected_before_dialog(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, filesystem, _) = export_ui(cx).await;
        view.update(&mut visual, |view, _| {
            let mut encoded = serde_json::to_value(view.report.as_ref().expect("Current checks"))
                .expect("Fixture serialization");
            let checks = encoded
                .get_mut("checks")
                .and_then(serde_json::Value::as_array_mut)
                .expect("Report checks");
            for index in 0..100 {
                checks.push(
                    serde_json::to_value(DoctorCheck::new(
                        &format!("oversized-{index}"),
                        "Oversized selected fixture",
                        CheckOutcome::Warning,
                        &"x".repeat(1024),
                        10,
                    ))
                    .expect("Fixture check"),
                );
            }
            view.report = Some(serde_json::from_value(encoded).expect("Unvalidated fixture"));
        });
        visual.refresh().expect("Export refusal control");
        let bounds = visual
            .debug_bounds("doctor-export-current")
            .expect("Explicit export");
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        assert!(view.read_with(&visual, |view, _| {
            view.export_task.is_none()
                && view
                    .export_status
                    .as_ref()
                    .expect("Bound error")
                    .as_str()
                    .contains("Export unavailable")
        }));
        assert!(
            fs::Fs::metadata(
                filesystem.as_ref(),
                std::path::Path::new("/exports/report.md")
            )
            .await
            .expect("No file")
            .is_none()
        );
    }

    #[cfg(target_os = "macos")]
    struct NativeRestartUiFixture {
        directory: tempfile::TempDir,
        client: Entity<remote::RemoteClient>,
        project: Entity<Project>,
        workspace: Entity<Workspace>,
        window: gpui::WindowHandle<workspace::MultiWorkspace>,
    }

    #[cfg(target_os = "macos")]
    impl NativeRestartUiFixture {
        async fn new(
            cx: &mut TestAppContext,
        ) -> (
            Self,
            Entity<DoctorView>,
            VisualTestContext,
            crate::doctor::DoctorHistoryStore,
        ) {
            Self::with_stop_delay(false, cx).await
        }
        async fn with_stop_delay(
            delay: bool,
            cx: &mut TestAppContext,
        ) -> (
            Self,
            Entity<DoctorView>,
            VisualTestContext,
            crate::doctor::DoctorHistoryStore,
        ) {
            use std::os::unix::fs::PermissionsExt as _;
            let (_local_view, _local_visual, history, local_workspace) = restart_ui(cx).await;
            let app_state =
                local_workspace.read_with(cx, |workspace, _| workspace.app_state().clone());
            let directory =
                tempfile::tempdir_in("/private/tmp").expect("Private native host/project fixture");
            let root = directory.path().join("project");
            std::fs::create_dir(&root).expect("Disposable project");
            std::fs::write(
                root.join("sentinel.txt"),
                "preserved unrelated project content",
            )
            .expect("Source sentinel");
            let binary = std::env::var_os("MADE_DOCTOR_SERVER_BINARY")
                .map(std::path::PathBuf::from)
                .expect("Current native remote_server binary");
            assert!(binary.is_file());
            let wrapper = directory.path().join("server-wrapper");
            let quote = |path: &std::path::Path| {
                format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
            };
            let stop_delay = if delay {
                format!(
                    "if [ \"$1\" = doctor-server-stopped ]; then printf '%s' \"$$\" > {}; exec /bin/sleep 60; fi\n",
                    quote(&directory.path().join("stop-observation.pid"))
                )
            } else {
                String::new()
            };
            std::fs::write(
                &wrapper,
                format!(
                    "#!/bin/sh\nprintf '%s %s\\n' \"$$\" \"$*\" >> {}\n{stop_delay}exec {} --test-data-dir {} \"$@\"\n",
                    quote(&directory.path().join("wrapper-invocations.log")),
                    quote(&binary),
                    quote(directory.path())
                ),
            )
            .expect("Exact native fixture launcher");
            std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
                .expect("Private launcher permissions");
            cx.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
            let connection = std::sync::Arc::new(remote::ProcessRemoteConnection::new(
                wrapper,
                directory.path().join("server_state"),
            ));
            let (_cancel, cancelled) = futures::channel::oneshot::channel();
            let client = cx
                .update(|cx| {
                    remote::RemoteClient::new(
                        remote::ConnectionIdentifier::setup(),
                        connection,
                        cancelled,
                        std::sync::Arc::new(remote::MockDelegate),
                        cx,
                    )
                })
                .await
                .expect("Native server connection")
                .expect("Observed native client");
            let project = cx.update(|cx| {
                Project::remote(
                    client.clone(),
                    app_state.client.clone(),
                    app_state.node_runtime.clone(),
                    app_state.user_store.clone(),
                    app_state.languages.clone(),
                    app_state.fs.clone(),
                    true,
                    cx,
                )
            });
            project
                .update(cx, |project, cx| {
                    project.find_or_create_worktree(&root, true, cx)
                })
                .await
                .expect("Native project root");
            project
                .read_with(cx, |project, cx| project.wait_for_initial_scan(cx))
                .await;
            let store = project.read_with(cx, |project, _| project.worktree_store());
            let worktree_id = project.read_with(cx, |project, cx| {
                project
                    .visible_worktrees(cx)
                    .next()
                    .expect("Selected root")
                    .read(cx)
                    .id()
            });
            cx.update(|cx| {
                use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
                TrustedWorktrees::try_get_global(cx)
                    .expect("Existing trust owner")
                    .update(cx, |trust, cx| {
                        trust.trust(
                            &store,
                            collections::HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                            cx,
                        )
                    });
            });
            let window = cx.add_window(|window, cx| {
                let workspace = cx
                    .new(|cx| Workspace::new(None, project.clone(), app_state.clone(), window, cx));
                workspace::MultiWorkspace::new(workspace, window, cx)
            });
            let workspace = window
                .read_with(cx, |multi, _| multi.workspace().clone())
                .expect("Owned remote Workspace");
            let mut visual = VisualTestContext::from_window(window.into(), cx);
            visual.dispatch_action(Doctor);
            visual.run_until_parked();
            let view = workspace
                .read_with(&visual, |workspace, cx| {
                    workspace.item_of_type::<DoctorView>(cx)
                })
                .expect("Remote Doctor through existing action");
            view.update(&mut visual, |view, cx| view.refresh(cx));
            let mut notifications = visual.notifications(&view);
            while view.read_with(&visual, |view, _| view.collection.is_some()) {
                futures::StreamExt::next(&mut notifications)
                    .await
                    .expect("Remote collection observation");
            }
            #[cfg(feature = "rust-tools")]
            {
                let discovery =
                    project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(200));
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while !discovery.read_with(cx, |owner, _| owner.discovery_idle_for_test()) {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "Existing Rust discovery did not settle before idle-host fixture review"
                    );
                    smol::unblock(|| std::thread::sleep(std::time::Duration::from_millis(10)))
                        .await;
                }
            }
            (
                Self {
                    directory,
                    client,
                    project,
                    workspace,
                    window,
                },
                view,
                visual,
                history,
            )
        }
        async fn stop(client: &Entity<remote::RemoteClient>, cx: &mut TestAppContext) {
            use futures::FutureExt as _;
            let status = client
                .read_with(cx, |client, cx| client.doctor_restart_status(cx))
                .await
                .expect("Current owned native server");
            let result = client
                .update(cx, |client, cx| {
                    client.doctor_restart(
                        status,
                        futures::future::pending().boxed(),
                        |_, after_shutdown| {
                            anyhow::ensure!(
                                !after_shutdown,
                                "Private fixture cleanup leaves its exact server stopped"
                            );
                            Ok(())
                        },
                        cx,
                    )
                })
                .await
                .expect("Graceful exact-instance fixture cleanup");
            assert!(result.shutdown_requested && result.replacement.is_none());
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test(iterations = 1)]
    async fn doctor_ui_restart_flow_native_server_reopens_fresh_project_without_replay(
        cx: &mut TestAppContext,
    ) {
        let (fixture, view, mut visual, history) = NativeRestartUiFixture::new(cx).await;
        prepare_restart_ui(&view, &mut visual, true).await;
        view.read_with(&visual, |view, _| {
            let review = view
                .proposal
                .as_ref()
                .expect("Exact observed owned-server preview");
            assert!(matches!(review, OwnerRepair::RemoteRestart(_)));
            assert!(review.effects().as_str().contains("no old request replay"));
        });
        click_native_process_control(&view, &mut visual, "doctor-approve-repair");
        wait_for_native_repair(&view, &mut visual).await;
        let fresh_workspace = fixture
            .window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Reopened native Workspace");
        view.read_with(&visual, |view, _| {
            assert_ne!(
                fresh_workspace,
                fixture.workspace,
                "Reopening error {:?}; outcome {:?}; detail {:?}",
                view.error,
                view.owner_result.as_ref().map(OwnerRepair::phase),
                view.owner_result
                    .as_ref()
                    .and_then(OwnerRepair::operation_detail)
            )
        });
        let fresh_project =
            fresh_workspace.read_with(cx, |workspace, _| workspace.project().clone());
        assert_ne!(fresh_project, fixture.project);
        let fresh_client = fresh_project.read_with(cx, |project, _| {
            project.remote_client().expect("Fresh connected owner")
        });
        assert_ne!(fresh_client, fixture.client);
        fresh_workspace.read_with(cx, |workspace, cx| {
            let doctor = workspace
                .item_of_type::<DoctorView>(cx)
                .expect("Doctor reopened by existing action");
            assert!(doctor.read(cx).proposal.is_none());
            assert!(
                doctor.read(cx).report.is_none(),
                "Prior observations cannot become fresh verification"
            );
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable incomplete restart receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Incomplete
        );
        view.read_with(&visual, |view, _| {
            assert!(
                view.error.is_none(),
                "Reopening observation error: {:?}; status {:?}",
                view.error,
                view.repair_status
            )
        });
        assert_eq!(
            std::fs::read_to_string(fixture.directory.path().join("project/sentinel.txt"))
                .expect("Preserved original project"),
            "preserved unrelated project content"
        );
        NativeRestartUiFixture::stop(&fresh_client, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test(iterations = 1)]
    async fn doctor_ui_restart_flow_native_revoked_trust_and_cancelled_proposal_do_not_stop_server(
        cx: &mut TestAppContext,
    ) {
        let (fixture, view, mut visual, history) = NativeRestartUiFixture::new(cx).await;
        let original = fixture
            .client
            .read_with(cx, |client, cx| client.doctor_restart_status(cx))
            .await
            .expect("Initial server identity");
        prepare_restart_ui(&view, &mut visual, true).await;
        click_native_process_control(&view, &mut visual, "doctor-cancel-repair");
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No consent receipt")
                .is_empty()
        );
        prepare_restart_ui(&view, &mut visual, true).await;
        view.read_with(&visual, |view, _| {
            assert!(
                view.proposal.is_some(),
                "Native review failed before trust revocation: {:?}",
                view.error
            )
        });
        let store = fixture
            .project
            .read_with(cx, |project, _| project.worktree_store());
        let selected = view.read_with(&visual, |view, _| {
            view.selected_worktree.expect("Selected root")
        });
        visual.update(|_, cx| {
            use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
            TrustedWorktrees::try_get_global(cx)
                .expect("Existing trust owner")
                .update(cx, |trust, cx| {
                    trust.restrict(
                        store.downgrade(),
                        collections::HashSet::from_iter([PathTrust::Worktree(selected)]),
                        cx,
                    )
                });
        });
        click_native_process_control(&view, &mut visual, "doctor-approve-repair");
        wait_for_native_repair(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| view.error.is_some()));
        let current = fixture
            .client
            .read_with(cx, |client, cx| client.doctor_restart_status(cx))
            .await
            .expect("Original server remains connected");
        assert_eq!(current.instance, original.instance);
        assert_eq!(current.process_id, original.process_id);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No granted action")
                .is_empty()
        );
        NativeRestartUiFixture::stop(&fixture.client, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test(iterations = 1)]
    async fn doctor_ui_restart_flow_native_active_cancellation_blocks_other_restart_and_never_replays(
        cx: &mut TestAppContext,
    ) {
        let (fixture, view, mut visual, history) =
            NativeRestartUiFixture::with_stop_delay(true, cx).await;
        prepare_restart_ui(&view, &mut visual, true).await;
        click_native_process_control(&view, &mut visual, "doctor-approve-repair");
        let marker = fixture.directory.path().join("stop-observation.pid");
        let started = smol::unblock(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !marker.is_file() {
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            true
        })
        .await;
        view.read_with(&visual, |view, _| {
            assert!(
                started,
                "Owned stop helper unavailable; error {:?}; outcome {:?}; detail {:?}; wrapper invocations {:?}",
                view.error,
                view.owner_result.as_ref().map(OwnerRepair::phase),
                view.owner_result
                    .as_ref()
                    .and_then(OwnerRepair::operation_detail),
                std::fs::read_to_string(fixture.directory.path().join("wrapper-invocations.log"))
            )
        });
        let observation = cx
            .update(|cx| workspace::application_restart_observation(cx))
            .expect("All existing workspace blockers");
        assert!(
            observation
                .blockers
                .iter()
                .any(|blocker| blocker.contains("Doctor operation")),
            "Another application restart cannot bypass active remote coordination"
        );
        click_native_process_control(&view, &mut visual, "doctor-cancel-repair");
        wait_for_native_repair(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            let result = view
                .owner_result
                .as_ref()
                .expect("Observed partial cancellation");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Cancelled);
            assert!(
                result
                    .operation_detail()
                    .expect("Reached effects")
                    .as_str()
                    .contains("no replacement")
            );
        });
        assert!(
            fixture
                .client
                .read_with(cx, |client, _| client.connection().is_none())
        );
        let active = fixture
            .window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("No replacement workspace");
        assert_eq!(active, fixture.workspace);
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable cancelled receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(
            std::fs::read_to_string(fixture.directory.path().join("project/sentinel.txt"))
                .expect("Preserved unrelated source"),
            "preserved unrelated project content"
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test(iterations = 1)]
    async fn doctor_ui_restart_flow_native_reopening_rejects_cancellation_and_changed_authority(
        cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        let (fixture, view, visual, history) = NativeRestartUiFixture::new(cx).await;
        let (directory, generation, scope) = view.read_with(&visual, |view, cx| {
            (
                view.directory(cx).expect("Selected root"),
                view.generation,
                view.report.as_ref().expect("Bounded report").scope.clone(),
            )
        });
        let now = DoctorView::now().expect("Clock");
        let mut review = cx
            .update(|cx| {
                crate::doctor::prepare_remote_restart(
                    &fixture.project,
                    directory.clone(),
                    generation,
                    now,
                    cx,
                )
            })
            .await
            .expect("Existing owner review");
        cx.update(|cx| {
            review.approve(
                &crate::doctor::project_repair_binding(
                    &fixture.project,
                    directory.clone(),
                    generation,
                    cx,
                )
                .expect("Selected authority"),
            )
        })
        .expect("Exact reviewed action consent");
        let report =
            DoctorReport::with_checks(scope, generation, now, vec![review.before().clone()])
                .expect("Relevant original observation");
        let (_cancel, cancelled) = async_channel::bounded(1);
        let (review, replacement) = cx
            .update(|cx| {
                crate::doctor::launch_remote_restart(
                    review,
                    &fixture.project,
                    directory,
                    generation,
                    report,
                    compiled_product(),
                    history.clone(),
                    now,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Observed native replacement");
        assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Incomplete);
        let replacement = replacement.expect("Fresh ready connection");
        let expected = vec![(fixture.workspace.entity_id(), fixture.project.entity_id())];
        let result = cx
            .update(|cx| {
                workspace::doctor_reopen_remote_workspaces(
                    fixture.project.clone(),
                    replacement.clone(),
                    expected.clone(),
                    None,
                    Doctor.boxed_clone(),
                    futures::future::ready(()).boxed().shared(),
                    cx,
                )
            })
            .await;
        assert!(
            result
                .expect_err("Pre-cancelled opening cannot create a Project")
                .to_string()
                .contains("cancelled")
        );
        let store = fixture
            .project
            .read_with(cx, |project, _| project.worktree_store());
        let selected = view.read_with(&visual, |view, _| {
            view.selected_worktree.expect("Retained root")
        });
        cx.update(|cx| {
            use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
            TrustedWorktrees::try_get_global(cx)
                .expect("Existing authority owner")
                .update(cx, |trust, cx| {
                    trust.restrict(
                        store.downgrade(),
                        collections::HashSet::from_iter([PathTrust::Worktree(selected)]),
                        cx,
                    )
                });
        });
        let result = cx
            .update(|cx| {
                workspace::doctor_reopen_remote_workspaces(
                    fixture.project.clone(),
                    replacement.clone(),
                    expected,
                    None,
                    Doctor.boxed_clone(),
                    futures::future::pending::<()>().boxed().shared(),
                    cx,
                )
            })
            .await;
        assert!(
            result
                .expect_err("Changed authority cannot implicitly create a fresh trusted scope")
                .to_string()
                .contains("authority")
        );
        assert_eq!(
            fixture
                .window
                .read_with(cx, |multi, _| multi.workspace().clone())
                .expect("Preserved Workspace"),
            fixture.workspace
        );
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("No replay receipts");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Incomplete
        );
        NativeRestartUiFixture::stop(&replacement, cx).await;
    }

    #[cfg(feature = "agentic")]
    async fn repair_fixture(
        cx: &mut TestAppContext,
        fail: bool,
        hold: bool,
    ) -> (
        Entity<Project>,
        Entity<DoctorView>,
        VisualTestContext,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        crate::doctor::DoctorHistoryStore,
    ) {
        let (project, _, _, calls) =
            crate::doctor_agent::tests::doctor_owner_fixture(cx, fail, hold).await;
        let database = db::AppDatabase::test_new();
        let history = crate::doctor::DoctorHistoryStore::for_test_database(&database);
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
            cx.set_global(history.clone());
        });
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Repair workspace");
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Native repair view");
        view.update(&mut visual, |view, cx| {
            view.selected_product = DoctorProduct::Made;
            view.refresh(cx);
        });
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-advanced");
        let review = visual
            .debug_bounds("doctor-review-mcp")
            .expect("Native owner review control");
        visual.simulate_click(review.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_some(), "Concrete owner proposal")
        });
        (project, view, visual, calls, history)
    }

    fn compiled_product() -> DoctorProduct {
        match product_flavor::ID {
            "rust" => DoctorProduct::Copper,
            "jvm" => DoctorProduct::Groovy,
            _ => DoctorProduct::Made,
        }
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_exact_approval_rechecks_and_deduplicates(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, false, false).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1, "Review must not restart");
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("No approval history")
                .is_empty()
        );
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Exact approval control");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        view.update(&mut visual, |view, cx| view.approve_repair(cx));
        visual.run_until_parked();
        visual
            .executor()
            .advance_clock(std::time::Duration::from_millis(100));
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 2, "Only one owner restart");
        view.read_with(&visual, |view, _| {
            let result = view.owner_result.as_ref().expect("Observed owner result");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Verified);
            assert_eq!(
                result.after().expect("Relevant fresh recheck").outcome,
                CheckOutcome::Ready
            );
            assert!(
                view.repair_status
                    .as_ref()
                    .expect("Limits")
                    .as_str()
                    .contains("does not prove")
            );
            assert!(view.repair_task.is_none());
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Durable history");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("One approved action").outcome,
            crate::doctor::ReceiptOutcome::Completed
        );
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_stale_inputs_never_dispatch(cx: &mut TestAppContext) {
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, false, false).await;
        visual.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(r#"{"confirm_quit":false}"#, cx)
                    .expect("Changed input")
            })
        });
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Stale approval control");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
            assert!(
                view.error
                    .as_ref()
                    .expect("Stale error")
                    .as_str()
                    .contains("changed")
            );
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("No stale receipt")
                .is_empty()
        );
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_cancellation_preserves_reached_owner_effects(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, false, true).await;
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Approval");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "Owner accepted restart");
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .unwrap_or_else(|| {
                panic!(
                    "Native cancel operation: {:?}",
                    view.read_with(&visual, |view, _| (
                        view.repair_task.is_some(),
                        view.proposal.is_some(),
                        view.owner_result.as_ref().map(OwnerRepair::phase),
                        view.error.clone(),
                        view.repair_status.clone(),
                        view.generation
                    ))
                )
            });
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .executor()
            .advance_clock(std::time::Duration::from_millis(100));
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Cancellation observation")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Cancelled
            )
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Cancellation receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2, "No rollback or replay");
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_cancelled_proposal_has_no_receipt_or_effect(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, false, false).await;
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Proposal cancellation");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("No cancelled proposal receipt")
                .is_empty()
        );
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_failed_owner_never_reports_verified(cx: &mut TestAppContext) {
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, true, false).await;
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Exact approval");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .executor()
            .advance_clock(std::time::Duration::from_millis(100));
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        view.read_with(&visual, |view, _| {
            assert_ne!(
                view.owner_result
                    .as_ref()
                    .expect("Observed failure")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Verified
            )
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Time"))
            .await
            .expect("Failure history");
        assert_eq!(records.len(), 1);
        assert_ne!(
            records.first().expect("Failure receipt").outcome,
            crate::doctor::ReceiptOutcome::Completed
        );
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_repair_flow_revoked_mcp_authority_never_restarts(cx: &mut TestAppContext) {
        use settings::Settings as _;
        use std::sync::atomic::Ordering;
        let (_project, view, mut visual, calls, history) = repair_fixture(cx, false, false).await;
        visual.update(|_, cx| {
            let mut preferences =
                project::project_settings::ProjectSettings::get_global(cx).clone();
            preferences.context_servers.clear();
            project::project_settings::ProjectSettings::override_global(preferences, cx);
        });
        visual.run_until_parked();
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .unwrap_or_else(|| {
                panic!(
                    "Prior proposal remains explicit: {:?}",
                    view.read_with(&visual, |view, _| (
                        view.proposal.is_some(),
                        view.error.clone(),
                        view.closed,
                        view.generation
                    ))
                )
            });
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .executor()
            .advance_clock(std::time::Duration::from_millis(100));
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        view.read_with(&visual, |view, _| {
            assert!(
                view.owner_result.as_ref().is_none_or(
                    |result| result.phase() != crate::doctor::DoctorRepairPhase::Verified
                )
            );
            assert!(view.owner_result.is_some() || view.error.is_some());
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Time"))
                .await
                .expect("Revoked history")
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(all(unix, any(feature = "rust-tools", feature = "jvm-tools")))]
    async fn install_ui(
        project: Entity<Project>,
        review: OwnerRepair,
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
    ) {
        let database = db::AppDatabase::test_new();
        let history = crate::doctor::DoctorHistoryStore::for_test_database(&database);
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
            cx.set_global(history.clone());
        });
        let window =
            cx.add_window(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Manager workspace");
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Doctor view");
        let collect = visual
            .debug_bounds("doctor-refresh")
            .expect("Native collection control");
        visual.simulate_click(collect.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.collection.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Native collection view remains available");
        }
        click_investigation_control(&view, &mut visual, "doctor-advanced");
        view.update_in(&mut visual, |view, window, cx| {
            assert_eq!(view.generation, 1);
            view.accept_proposal(Ok(review), window, cx);
        });
        visual.run_until_parked();
        (view, visual, history)
    }

    #[cfg(all(unix, any(feature = "rust-tools", feature = "jvm-tools")))]
    async fn approve_install_ui(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        view.update(visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Native exact manager action approval");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        wait_for_native_repair(view, visual).await;
    }

    #[cfg(unix)]
    async fn wait_for_native_repair(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        // OS process completion is not controlled by GPUI's virtual clock.
        let mut notifications = visual.notifications(view);
        while view.read_with(visual, |view, _| view.repair_task.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("Native repair view remains available");
        }
    }

    #[cfg(unix)]
    fn click_native_process_control(
        view: &Entity<DoctorView>,
        visual: &mut VisualTestContext,
        selector: &'static str,
    ) {
        // Draining every runnable can spin on real process I/O before the user
        // event is delivered. Dispatch both native input events before awaiting it.
        view.update(visual, |_, cx| cx.notify());
        visual.refresh().expect("Refresh actual native controls");
        // Refresh only schedules drawing; bounds can still belong to the prior non-running frame.
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let position = visual
            .debug_bounds(selector)
            .expect("Rendered current native process control")
            .center();
        visual.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position,
                    modifiers: gpui::Modifiers::default(),
                    button: gpui::MouseButton::Left,
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position,
                    modifiers: gpui::Modifiers::default(),
                    button: gpui::MouseButton::Left,
                    click_count: 1,
                }),
                cx,
            );
        });
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_rustup_requires_separate_install_and_rechecks(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::doctor_install::tests::fixture(cx).await;
        let review = crate::doctor_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Rustup(review), cx).await;
        let marker = fixture.home.join(".rustup/started");
        assert!(!marker.exists(), "Opening exact review executes nothing");
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert!(matches!(view.proposal.as_ref(), Some(OwnerRepair::Rustup(review)) if !review.action().is_inspection()));
            assert!(view.owner_result.is_some());
        });
        assert!(
            !marker.exists(),
            "Inspection approval is not install consent"
        );
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            let result = view
                .owner_result
                .as_ref()
                .expect("Observed manager installation");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Verified);
            assert_eq!(
                result
                    .after()
                    .expect("Fresh compiler/inventory recheck")
                    .outcome,
                CheckOutcome::Ready
            );
            assert!(view.proposal.is_none());
        });
        assert!(marker.exists());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Stage receipts");
        assert_eq!(records.len(), 2);
        assert_eq!(
            std::fs::read_to_string(fixture.home.join(".rustup/settings.toml"))
                .expect("Preserved defaults"),
            "version = '12'\ndefault_toolchain = 'preserved-default'\nprofile = 'complete'\n"
        );
    }

    #[cfg(all(unix, feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_mise_each_stage_and_cancelled_cache(cx: &mut TestAppContext) {
        let fixture = crate::doctor_jvm_install::tests::fixture(cx).await;
        let review = crate::doctor_jvm_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Mise(review), cx).await;
        let marker = fixture.home.join(".local/share/mise/started");
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| assert!(matches!(view.proposal.as_ref(), Some(OwnerRepair::Mise(review)) if review.action().stage() == crate::doctor_jvm_install::MiseStage::Resolve)));
        assert!(!marker.exists());
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            let Some(OwnerRepair::Mise(review)) = &view.proposal else {
                panic!("Separate artifact installation proposal");
            };
            assert_eq!(
                review.action().stage(),
                crate::doctor_jvm_install::MiseStage::Install
            );
            assert!(review.action().artifact().is_some());
            assert!(
                view.proposal
                    .as_ref()
                    .expect("Proposal")
                    .exact_preview()
                    .expect("Exact transient inputs")
                    .iter()
                    .any(|line| line.contains("SHA-256:"))
            );
        });
        assert!(!marker.exists(), "Resolution is not installation consent");
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Cancel staged install");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(!marker.exists());
        assert!(
            !std::fs::read_dir(&fixture.home)
                .expect("Private cache scope")
                .any(|entry| entry
                    .expect("Entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".made-doctor-mise-"))
        );
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Only approved stages")
                .len(),
            2
        );
    }

    #[cfg(all(unix, feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_mise_install_rechecks_runtime_and_compiler(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::doctor_jvm_install::tests::fixture(cx).await;
        let review = crate::doctor_jvm_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Mise(review), cx).await;
        for _ in 0..3 {
            approve_install_ui(&view, &mut visual).await;
        }
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert_eq!(
                view.owner_result.as_ref().expect("Rechecked Java").phase(),
                crate::doctor::DoctorRepairPhase::Verified
            );
        });
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Distinct stage receipts")
                .len(),
            3
        );
    }

    #[cfg(all(unix, feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_sdkman_first_default_is_distinct(cx: &mut TestAppContext) {
        let (fixture, review) = crate::doctor_jvm_install::tests::sdkman_fixture(false, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Sdkman(review), cx).await;
        let marker = fixture.home.join(".sdkman/started");
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Disabled install approval");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(!marker.exists());
        view.read_with(&visual, |view, _| assert!(view.proposal.is_some()));
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let acknowledge = visual
            .debug_bounds("doctor-approve-first-default")
            .expect("Separate first-default consent");
        visual.simulate_click(acknowledge.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(
            !marker.exists(),
            "Default acknowledgment alone cannot install"
        );
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed release/runtime/compiler/default")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Verified
            )
        });
        assert!(marker.exists());
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Install receipt")
                .len(),
            1
        );
    }

    #[cfg(all(unix, feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_sdkman_changed_config_never_executes(cx: &mut TestAppContext) {
        let (fixture, review) = crate::doctor_jvm_install::tests::sdkman_fixture(true, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Sdkman(review), cx).await;
        std::fs::write(
            fixture.home.join(".sdkman/etc/config"),
            "sdkman_auto_env=true\n",
        )
        .expect("Revoked manager configuration");
        approve_install_ui(&view, &mut visual).await;
        assert!(!fixture.home.join(".sdkman/started").exists());
        view.read_with(&visual, |view, _| {
            assert_ne!(
                view.owner_result
                    .as_ref()
                    .expect("Observed blocked effects")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Verified
            )
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Observed failure receipt")
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_invalid_form_and_edits_cancel_pending_consent(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::doctor_install::tests::fixture(cx).await;
        let review = crate::doctor_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Rustup(review), cx).await;
        let version = view.read_with(&visual, |view, _| view.install.version.clone());
        version.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("stable", window, cx)
        });
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(
                view.proposal.is_none(),
                "Changed form invalidates previous consent"
            )
        });
        view.update(&mut visual, |view, cx| {
            view.install.open = true;
            cx.notify();
        });
        visual.run_until_parked();
        click_investigation_control(&view, &mut visual, "doctor-prepare-install");
        visual
            .condition(&view, |view, _| view.preparation.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
            assert!(
                view.error
                    .as_ref()
                    .expect("Invalid input error")
                    .as_str()
                    .contains("exact")
            );
        });
        assert!(!fixture.home.join(".rustup/started").exists());
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No effects receipt")
                .is_empty()
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_cancelled_install_preserves_reached_partial_effects(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::doctor_install::tests::fixture(cx).await;
        let review = crate::doctor_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Rustup(review), cx).await;
        approve_install_ui(&view, &mut visual).await;
        std::fs::write(fixture.home.join(".rustup/fixture-mode"), "wait")
            .expect("Pending native installation fixture");
        click_native_process_control(&view, &mut visual, "doctor-approve-repair");
        let marker = fixture.home.join(".rustup/started");
        crate::doctor_install::tests::wait_for_started(&fixture).await;
        click_native_process_control(&view, &mut visual, "doctor-cancel-repair");
        view.read_with(&visual, |view, _| {
            assert!(
                view.repair_task.is_none()
                    || view
                        .repair_status
                        .as_ref()
                        .is_some_and(|status| status.as_str().contains("Cancellation requested")),
                "Native cancellation click must reach the reviewed operation: {:?}",
                view.repair_status
            )
        });
        wait_for_native_repair(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed cancellation")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Cancelled
            );
            assert!(view.proposal.is_none());
        });
        assert!(
            marker.exists(),
            "Reached effects are not falsely rolled back"
        );
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Cancellation history");
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .any(|record| record.outcome == crate::doctor::ReceiptOutcome::Cancelled)
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_install_flow_successful_command_with_wrong_compiler_is_incomplete(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::doctor_install::tests::fixture(cx).await;
        let review = crate::doctor_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Rustup(review), cx).await;
        approve_install_ui(&view, &mut visual).await;
        std::fs::write(fixture.home.join(".rustup/fixture-mode"), "wrong-version")
            .expect("Wrong installed compiler fixture");
        approve_install_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed compiler mismatch")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Incomplete
            )
        });
        assert!(fixture.home.join(".rustup/started").exists());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Relevant mismatch receipt");
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .any(|record| record.outcome == crate::doctor::ReceiptOutcome::Incomplete)
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    async fn dependency_ui(
        mode: &str,
        kind: crate::doctor::DependencyProbe,
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
        crate::doctor_install::tests::Fixture,
        std::path::PathBuf,
    ) {
        let fixture = crate::doctor_install::tests::fixture(cx).await;
        let executable =
            fixture
                .home
                .join(if matches!(kind, crate::doctor::DependencyProbe::Java) {
                    "java"
                } else {
                    "rustc"
                });
        let source = fixture.home.join("version-probe.c");
        std::fs::write(&source, r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int count, char **arguments) {
    const char *name = strrchr(arguments[0], '/'); name = name ? name + 1 : arguments[0];
    const char *version_argument = !strcmp(name, "java") ? "-version" : "--version";
    if (count != 2 || strcmp(arguments[1], version_argument) || getenv("GITHUB_TOKEN") || getenv("JAVA_HOME") || getenv("BASH_ENV") || getenv("ENV")) return 91;
    char path[4096], mode[64] = "success";
    snprintf(path, sizeof(path), "%s.mode", arguments[0]);
    FILE *input = fopen(path, "r"); if (input) { if (!fgets(mode, sizeof(mode), input)) return 92; fclose(input); }
    snprintf(path, sizeof(path), "%s.started", arguments[0]);
    FILE *marker = fopen(path, "a"); if (!marker) return 93; fputs("started\n", marker); if (fclose(marker)) return 94;
    if (!strcmp(mode, "wait")) { for (;;) pause(); }
    if (!strcmp(mode, "large")) { for (int index = 0; index < 20000; index++) putchar('x'); return 0; }
    if (!strcmp(mode, "empty")) return 0;
    if (!strcmp(mode, "fail")) { puts("observed tool failure"); return 9; }
    puts("rustc 99.9.9 (fixture) GITHUB_TOKEN=DOCTOR_PROBE_RAW_SECRET");
    return 0;
}
"#).expect("Native installed-tool fixture source");
        let compiled = smol::process::Command::new("/usr/bin/cc")
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .await
            .expect("Existing compiler creates fixture");
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        std::fs::write(executable.with_extension("mode"), mode).expect("Private fixture behavior");
        let review = crate::doctor_install::tests::inspection(&fixture, cx).await;
        let (view, mut visual, history) =
            install_ui(fixture.project.clone(), OwnerRepair::Rustup(review), cx).await;
        view.update_in(&mut visual, |view, window, cx| {
            view.cancel_repair(cx);
            view.dependency.open = true;
            view.dependency.kind = kind;
            view.dependency.executable.update(cx, |editor, cx| {
                editor.set_text(executable.to_string_lossy(), window, cx)
            });
            view.scroll_handle
                .set_offset(gpui::point(gpui::px(0.), gpui::px(0.)));
            cx.notify();
        });
        visual.run_until_parked();
        let prepare = visual
            .debug_bounds("doctor-review-dependency")
            .expect("Native metadata-only preparation");
        visual.simulate_click(prepare.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let mut notifications = visual.notifications(&view);
        while view.read_with(&visual, |view, _| view.preparation.is_some()) {
            futures::StreamExt::next(&mut notifications)
                .await
                .expect("View remains available during inspection");
        }
        view.read_with(&visual, |view, _| {
            assert!(
                matches!(view.proposal, Some(OwnerRepair::Dependency(_))),
                "{:?}",
                view.error
            )
        });
        assert!(
            !executable.with_extension("started").exists(),
            "Metadata review must not execute the tool"
        );
        (view, visual, history, fixture, executable)
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_requires_consent_deduplicates_and_redacts(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _fixture, executable) =
            dependency_ui("success", crate::doctor::DependencyProbe::RustCompiler, cx).await;
        view.read_with(&visual, |view, _| {
            let review = view
                .proposal
                .as_ref()
                .expect("Exact proposed version request");
            assert!(
                review
                    .exact_preview()
                    .expect("Preview")
                    .iter()
                    .any(|line| line.contains("--version"))
            );
        });
        approve_install_ui(&view, &mut visual).await;
        view.update(&mut visual, |view, cx| view.approve_repair(cx));
        view.read_with(&visual, |view, _| {
            let result = view.owner_result.as_ref().expect("Observed version result");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Verified);
            assert!(
                result
                    .after()
                    .expect("Relevant observation")
                    .detail
                    .as_str()
                    .contains("remain unverified")
            );
        });
        assert_eq!(
            std::fs::read_to_string(executable.with_extension("started"))
                .expect("Only one execution")
                .lines()
                .count(),
            1
        );
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Observed history");
        assert_eq!(records.len(), 1);
        assert!(
            !serde_json::to_string(&records)
                .expect("Redacted records")
                .contains("DOCTOR_PROBE_RAW_SECRET")
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_failure_empty_and_output_limit_are_not_fixes(
        cx: &mut TestAppContext,
    ) {
        let mut fixtures = Vec::new();
        for (mode, phase) in [
            ("fail", crate::doctor::DoctorRepairPhase::StillFailing),
            ("empty", crate::doctor::DoctorRepairPhase::Incomplete),
            ("large", crate::doctor::DoctorRepairPhase::Incomplete),
        ] {
            let (view, mut visual, history, fixture, _) =
                dependency_ui(mode, crate::doctor::DependencyProbe::RustCompiler, cx).await;
            approve_install_ui(&view, &mut visual).await;
            view.read_with(&visual, |view, _| {
                assert_eq!(
                    view.owner_result
                        .as_ref()
                        .expect("Observed limited result")
                        .phase(),
                    phase
                )
            });
            let records = history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Failure records");
            assert!(
                records
                    .iter()
                    .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
            );
            fixtures.push(fixture);
        }
    }

    #[cfg(all(unix, feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_java_uses_fixed_jdk_arguments(cx: &mut TestAppContext) {
        let (view, mut visual, history, _fixture, executable) =
            dependency_ui("success", crate::doctor::DependencyProbe::Java, cx).await;
        view.read_with(&visual, |view, _| {
            let Some(OwnerRepair::Dependency(review)) = view.proposal.as_ref() else {
                panic!("Reviewed installed JDK version query")
            };
            assert_eq!(
                review.action().executable().kind().version_arguments(),
                &["-version"]
            );
        });
        approve_install_ui(&view, &mut visual).await;
        assert!(executable.with_extension("started").exists());
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Selected JDK query observed")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Verified
            )
        });
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("JDK receipt")
                .len(),
            1
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_replaced_executable_never_runs(cx: &mut TestAppContext) {
        let (view, mut visual, history, _fixture, executable) =
            dependency_ui("success", crate::doctor::DependencyProbe::RustCompiler, cx).await;
        std::fs::write(&executable, "#!/bin/sh\nexit 0\n").expect("Intervening input change");
        approve_install_ui(&view, &mut visual).await;
        assert!(!executable.with_extension("started").exists());
        view.read_with(&visual, |view, _| {
            assert!(
                view.owner_result.as_ref().is_none_or(
                    |result| result.phase() != crate::doctor::DoctorRepairPhase::Verified
                )
            )
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Changed input records");
        assert!(
            records
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_revoked_trust_never_executes(cx: &mut TestAppContext) {
        use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
        let (view, mut visual, history, fixture, executable) =
            dependency_ui("success", crate::doctor::DependencyProbe::RustCompiler, cx).await;
        let trust = visual
            .update(|_, cx| TrustedWorktrees::try_get_global(cx).expect("Existing trust owner"));
        let store = fixture
            .project
            .read_with(&visual, |project, _| project.worktree_store());
        let selected = view.read_with(&visual, |view, _| {
            view.selected_worktree.expect("Selected root")
        });
        trust.update(&mut visual, |trust, cx| {
            trust.restrict(
                store.downgrade(),
                collections::HashSet::from_iter([PathTrust::Worktree(selected)]),
                cx,
            )
        });
        approve_install_ui(&view, &mut visual).await;
        assert!(!executable.with_extension("started").exists());
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No revoked execution")
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_cancelled_review_has_no_effects(cx: &mut TestAppContext) {
        let (view, mut visual, history, _fixture, executable) =
            dependency_ui("success", crate::doctor::DependencyProbe::RustCompiler, cx).await;
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Proposal cancel");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert!(!executable.with_extension("started").exists());
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No receipt without consent")
                .is_empty()
        );
    }

    #[cfg(all(unix, feature = "rust-tools"))]
    #[gpui::test]
    async fn doctor_ui_dependency_probe_owned_process_cancellation_is_incomplete(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _fixture, executable) =
            dependency_ui("wait", crate::doctor::DependencyProbe::RustCompiler, cx).await;
        click_native_process_control(&view, &mut visual, "doctor-approve-repair");
        let marker = executable.with_extension("started");
        smol::unblock(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !marker.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert!(marker.exists(), "Native process starts before cancel");
        })
        .await;
        click_native_process_control(&view, &mut visual, "doctor-cancel-repair");
        view.read_with(&visual, |view, _| {
            assert!(
                view.repair_status
                    .as_ref()
                    .is_some_and(|status| status.as_str().contains("Cancellation requested")),
                "Native cancellation must dispatch"
            );
        });
        wait_for_native_repair(&view, &mut visual).await;
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("No automatic replay");
        assert_eq!(
            records.first().expect("Reached cancellation").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
    }

    #[cfg(feature = "agentic")]
    async fn probe_ui(
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
        std::sync::Arc<language_model::fake_provider::FakeLanguageModel>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        let (_, view, mut visual, _, history) = repair_fixture(cx, false, false).await;
        view.update(&mut visual, |view, cx| view.cancel_repair(cx));
        let (model, authenticated) = crate::doctor_agent::tests::doctor_probe_fixture(cx);
        view.update(&mut visual, |view, cx| view.refresh(cx));
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.collection.is_none())
            .await;
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let review = visual
            .debug_bounds("doctor-review-provider-probe")
            .expect("Native selected provider review");
        visual.simulate_click(review.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(matches!(view.proposal, Some(OwnerRepair::Probe(_))))
        });
        (view, visual, history, model, authenticated)
    }

    #[cfg(feature = "agentic")]
    fn click_probe_approval(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        view.update(visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let approve = visual
            .debug_bounds("doctor-approve-repair")
            .expect("Exact probe consent control");
        visual.simulate_click(approve.center(), gpui::Modifiers::default());
        visual.run_until_parked();
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_cancelled_review_never_requests_or_records(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        model.forbid_requests();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Native proposal cancellation");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        assert_eq!(model.completion_count(), 0);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No effects receipt")
                .is_empty()
        );
        view.read_with(&visual, |view, _| assert!(view.proposal.is_none()));
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_approval_deduplicates_records_and_limits_success(
        cx: &mut TestAppContext,
    ) {
        use language_model::{LanguageModelCompletionEvent, StopReason};
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        assert_eq!(
            model.completion_count(),
            0,
            "Preparation cannot make a request"
        );
        click_probe_approval(&view, &mut visual);
        view.update(&mut visual, |view, cx| view.approve_repair(cx));
        visual.run_until_parked();
        let requests = model.pending_completions();
        assert_eq!(requests.len(), 1);
        let request = requests.first().expect("One reviewed request");
        assert_eq!(request.messages.len(), 1);
        assert!(request.tools.is_empty());
        assert!(request.thread_id.is_none());
        assert!(
            !request
                .messages
                .first()
                .expect("Fixed non-project content")
                .string_contents()
                .contains("doctor-owner")
        );
        let running = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable receipt precedes request");
        assert_eq!(running.len(), 1);
        assert_eq!(
            running.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Running
        );
        model.send_completion_stream_text_chunk(request, "DOCTOR_RAW_PROVIDER_SECRET_RESPONSE");
        model.send_completion_stream_event(
            request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            let result = view
                .owner_result
                .as_ref()
                .expect("Observed provider result");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Verified);
            assert_eq!(
                result
                    .after()
                    .expect("Actual connectivity observation")
                    .outcome,
                CheckOutcome::Ready
            );
            assert!(
                view.repair_status
                    .as_ref()
                    .expect("Limits")
                    .as_str()
                    .contains("does not prove")
            );
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Observed history");
        assert_eq!(records.len(), 1);
        assert!(
            !serde_json::to_string(&records)
                .expect("Redacted history")
                .contains("DOCTOR_RAW_PROVIDER_SECRET_RESPONSE")
        );
        model.end_completion_stream(request);
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_revoked_credentials_never_dispatch(cx: &mut TestAppContext) {
        let (view, mut visual, history, model, authenticated) = probe_ui(cx).await;
        model.forbid_requests();
        authenticated.store(false, std::sync::atomic::Ordering::Relaxed);
        click_probe_approval(&view, &mut visual);
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(model.completion_count(), 0);
        view.read_with(&visual, |view, _| {
            assert_ne!(
                view.owner_result.as_ref().expect("Observed denial").phase(),
                crate::doctor::DoctorRepairPhase::Verified
            )
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Denial receipt")
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_active_cancellation_retains_receipt_and_no_replay(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        click_probe_approval(&view, &mut visual);
        let request = model
            .pending_completions()
            .first()
            .expect("Reviewed request")
            .clone();
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Reachable probe cancellation");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed cancellation")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Cancelled
            )
        });
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Cancellation history");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(
            model.completion_count(),
            1,
            "Cancellation cannot replay the charged request"
        );
        model.end_completion_stream(&request);
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_deadline_is_incomplete_and_never_switches_provider(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        click_probe_approval(&view, &mut visual);
        let request = model
            .pending_completions()
            .first()
            .expect("Reviewed request")
            .clone();
        visual
            .executor()
            .advance_clock(std::time::Duration::from_secs(10));
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.owner_result
                    .as_ref()
                    .expect("Observed timeout")
                    .phase(),
                crate::doctor::DoctorRepairPhase::Incomplete
            )
        });
        assert_eq!(model.completion_count(), 1);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Incomplete history")
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
        model.end_completion_stream(&request);
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn doctor_ui_probe_flow_invalid_docker_proposal_never_queries_or_grants_consent(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        view.update(&mut visual, |view, cx| {
            view.cancel_repair(cx);
            view.probe.open = true;
            cx.notify();
        });
        let endpoint = view.read_with(&visual, |view, _| view.probe.endpoint.clone());
        let image = view.read_with(&visual, |view, _| view.probe.image.clone());
        endpoint.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("tcp://external.invalid:2375", window, cx)
        });
        image.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("alpine:latest", window, cx)
        });
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let review = visual
            .debug_bounds("doctor-review-docker-probe")
            .expect("Docker prepare review control");
        visual.simulate_click(review.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(view.error.is_some());
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
        });
        assert_eq!(model.completion_count(), 0);
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No execution receipt")
                .is_empty()
        );
    }

    #[cfg(all(unix, feature = "agentic"))]
    async fn docker_probe_ui(
        invalid_runtime: bool,
        wait_for_cancellation: bool,
        cx: &mut TestAppContext,
    ) -> (
        Entity<DoctorView>,
        VisualTestContext,
        crate::doctor::DoctorHistoryStore,
        util::test::TempTree,
    ) {
        let (view, mut visual, history, model, _) = probe_ui(cx).await;
        model.forbid_requests();
        let generation = view.read_with(&visual, |view, _| view.generation);
        let (tree, preview) = crate::doctor_agent::tests::doctor_docker_fixture(
            generation,
            invalid_runtime,
            wait_for_cancellation,
            cx,
        );
        view.update_in(&mut visual, |view, window, cx| {
            view.cancel_repair(cx);
            let directory = view.review_directory(cx).expect("Selected project scope");
            let proposal = crate::doctor_agent::prepare_owner_probe(
                &view.project,
                directory,
                crate::doctor_agent::DoctorProbePreview::Docker(preview),
                generation,
                DoctorView::now().expect("Clock"),
                cx,
            )
            .map(OwnerRepair::Probe);
            view.accept_proposal(proposal, window, cx);
        });
        visual.run_until_parked();
        assert!(
            !tree.path().join("queries").exists(),
            "Review must not query Docker"
        );
        (view, visual, history, tree)
    }

    #[cfg(all(unix, feature = "agentic"))]
    #[gpui::test]
    async fn doctor_ui_probe_flow_docker_observes_only_readonly_prerequisites(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, tree) = docker_probe_ui(false, false, cx).await;
        click_probe_approval(&view, &mut visual);
        wait_for_native_repair(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            let result = view.owner_result.as_ref().expect("Observed prerequisites");
            assert_eq!(result.phase(), crate::doctor::DoctorRepairPhase::Verified);
            let after = result.after().expect("Actual observation");
            assert_eq!(after.outcome, CheckOutcome::Ready);
            assert!(after.detail.as_str().contains("No workflow executed"));
            assert!(after.detail.as_str().contains("runtime handshake"));
        });
        assert!(tree.path().join("queries").exists());
        assert!(!tree.path().join("unexpected-command").exists());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable prerequisite receipt");
        assert_eq!(records.len(), 1);
    }

    #[cfg(all(unix, feature = "agentic"))]
    #[gpui::test]
    async fn doctor_ui_probe_flow_docker_invalid_runtime_is_failed_not_ready(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, tree) = docker_probe_ui(true, false, cx).await;
        click_probe_approval(&view, &mut visual);
        wait_for_native_repair(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            let result = view.owner_result.as_ref().expect("Observed failure");
            assert_eq!(
                result.phase(),
                crate::doctor::DoctorRepairPhase::StillFailing
            );
            assert_eq!(
                result.after().expect("Failure observation").outcome,
                CheckOutcome::Failed
            );
        });
        assert!(!tree.path().join("unexpected-command").exists());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Failure receipt");
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Failed
        );
    }

    #[cfg(all(unix, feature = "agentic"))]
    #[gpui::test]
    async fn doctor_ui_probe_flow_docker_replaced_file_blocks_before_queries(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, tree) = docker_probe_ui(false, false, cx).await;
        std::fs::write(tree.path().join("runtime"), "intervening replacement")
            .expect("Changed selected input");
        click_probe_approval(&view, &mut visual);
        wait_for_native_repair(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert!(
                view.error.is_some()
                    || view.owner_result.as_ref().is_some_and(
                        |result| result.phase() != crate::doctor::DoctorRepairPhase::Verified
                    )
            );
        });
        assert!(!tree.path().join("queries").exists());
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Blocked or interrupted receipt");
        assert!(
            records
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
    }

    #[cfg(all(unix, feature = "agentic"))]
    #[gpui::test]
    async fn doctor_ui_probe_flow_docker_active_cancel_preserves_incomplete_receipt(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, tree) = docker_probe_ui(false, true, cx).await;
        click_probe_approval(&view, &mut visual);
        let marker = tree.path().join("queries");
        smol::unblock(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !marker.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert!(marker.exists(), "Owned CLI must start before cancellation");
        })
        .await;
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Active native cancellation");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        wait_for_native_repair(&view, &mut visual).await;
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Cancellation receipt");
        assert_eq!(
            records.first().expect("Receipt").outcome,
            crate::doctor::ReceiptOutcome::Cancelled
        );
        assert!(!tree.path().join("unexpected-command").exists());
    }

    #[gpui::test]
    async fn doctor_conversation_context_latest_snapshot_clears_resolved_block(
        cx: &mut TestAppContext,
    ) {
        let (_, workspace, window) = fixture(cx, false).await;
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Doctor");
        for (outcome, expected) in [
            (CheckOutcome::Blocked, "Blocked 1;"),
            (CheckOutcome::Ready, "Blocked 0;"),
        ] {
            view.update(&mut visual, |view, cx| {
                view.report = Some(
                    DoctorReport::with_checks(
                        DoctorScope {
                            product: DiagnosticText::new("Copper"),
                            project: DiagnosticText::new("fixture"),
                            worktree: DiagnosticText::new("first"),
                            host: DiagnosticText::new("Local"),
                        },
                        view.generation,
                        10,
                        vec![DoctorCheck::new(
                            "made.provider",
                            "Resolved default model/provider",
                            outcome,
                            "Observed provider configuration only",
                            10,
                        )],
                    )
                    .expect("Observed report"),
                );
                let context = view.conversation_context(cx).expect("Latest collection");
                assert!(
                    context.contains("Latest collection replaces earlier collection snapshots")
                );
                assert!(context.contains(expected), "{context}");
                assert!(context.contains("Failed 0;"), "{context}");
                if outcome == CheckOutcome::Ready {
                    assert!(context.contains("Passed 1;"), "{context}");
                }
            });
        }
    }

    #[gpui::test]
    async fn doctor_view_scope_guides_collection_exposes_development_and_preserves_repeated_open(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor::DoctorInvestigationScope;
        let (project, workspace, window) = fixture(cx, false).await;
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Doctor");
        view.read_with(&visual, |view, cx| {
            assert!(!view.advanced_open);
            assert!(view.report.is_none());
            assert!(view.conversation_context(cx).is_none());
        });
        assert!(
            visual
                .debug_bounds("doctor-investigation-scope-0")
                .is_some()
        );
        assert!(
            visual
                .debug_bounds("doctor-investigation-scope-1")
                .is_some()
        );
        assert!(
            visual
                .debug_bounds("doctor-investigation-scope-2")
                .is_some()
        );
        assert!(
            visual.debug_bounds("doctor-product-0").is_none(),
            "Advanced operations start disclosed, not expanded"
        );
        view.update(&mut visual, |view, cx| {
            view.error = Some(DiagnosticText::new("Execution-host navigation failed"));
            assert!(view.conversation_context(cx).is_none());
            assert!(!view.collection_failed);
        });
        click_investigation_control(&view, &mut visual, "doctor-refresh");
        visual
            .condition(&view, |view, _| view.collection.is_none())
            .await;
        view.read_with(&visual, |view, cx| {
            let report = view.report.as_ref().expect("Real owner collection");
            assert!(
                report
                    .checks()
                    .iter()
                    .any(|check| check.id.as_str() == "development.scope")
            );
            assert!(
                report.investigation().is_none(),
                "Passive collection does not perform an application reproduction"
            );
            let summary = view
                .conversation_context(cx)
                .expect("Attributable collection context");
            assert!(summary.starts_with("/doctor collection status: Collected."));
            assert!(summary.contains("Project:"));
            assert!(summary.contains("Worktree:"));
            assert!(summary.contains("does not execute"));
            assert!(!view.report_is_stale(cx));
        });
        click_investigation_control(&view, &mut visual, "doctor-investigation-scope-1");
        let generation = view.read_with(&visual, |view, _| view.generation);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.investigation.scope,
                DoctorInvestigationScope::Development
            );
            assert_eq!(view.generation, generation);
            assert!(view.report.is_some());
        });
        let filesystem = project.read_with(&visual, |project, _| project.fs().as_fake());
        filesystem
            .insert_tree(
                "/first",
                serde_json::json!({"keep.txt":"changed fixture input"}),
            )
            .await;
        visual.run_until_parked();
        view.read_with(&visual, |view, cx| {
            assert!(
                view.report_is_stale(cx),
                "Changed worktree inputs cannot leave a current passing report"
            );
            assert!(
                view.conversation_context(cx)
                    .expect("Stale context")
                    .contains("Stale")
            );
        });
    }

    #[gpui::test]
    async fn doctor_view_scope_narrow_results_wrap_and_scroll_without_overlapping(
        cx: &mut TestAppContext,
    ) {
        let (_project, workspace, window) = fixture(cx, false).await;
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_resize(gpui::size(gpui::px(520.), gpui::px(700.)));
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Doctor");
        click_investigation_control(&view, &mut visual, "doctor-advanced");
        visual.refresh().expect("Advanced page layout");
        let visible_height = visual.update(|window, _| window.viewport_size().height);
        assert!(
            view.read_with(&visual, |view, _| view.scroll_handle.bounds().bottom()
                <= visible_height),
            "Advanced content must remain inside the actual window"
        );
        click_investigation_control(&view, &mut visual, "doctor-investigation-scope-1");
        view.update(&mut visual, |view, cx| {
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Development;
            view.report = Some(DoctorReport::with_checks(DoctorScope {
                product: DiagnosticText::new("Copper"), project: DiagnosticText::new("fixture"),
                worktree: DiagnosticText::new("first"), host: DiagnosticText::new("Local"),
            }, view.generation, 10, (0..3).map(|index| DoctorCheck::new(
                &format!("rust.failure.{index}"), "A failed Cargo collection with a long diagnostic",
                CheckOutcome::Failed, &"Missing dependency; offline restriction requires explicit recovery. ".repeat(20), 10,
            )).collect()).expect("Bounded diagnostic fixtures"));
            cx.notify();
        });
        visual.refresh().expect("Narrow result layout");
        let first = visual
            .debug_bounds("doctor-result-0")
            .expect("First diagnostic");
        let second = visual
            .debug_bounds("doctor-result-1")
            .expect("Second diagnostic");
        assert!(
            first.size.height > gpui::px(100.),
            "Long evidence must grow its card"
        );
        assert!(
            first.bottom() <= second.top(),
            "Diagnostic rows must not overlap"
        );
        let navigation = visual
            .debug_bounds("doctor-advanced")
            .expect("Always visible navigation");
        let (viewport, before) = view.read_with(&visual, |view, _| {
            (view.scroll_handle.bounds(), view.scroll_handle.offset())
        });
        visual.simulate_event(gpui::ScrollWheelEvent {
            position: viewport.center(),
            delta: gpui::ScrollDelta::Pixels(gpui::point(gpui::px(0.), gpui::px(-500.))),
            modifiers: gpui::Modifiers::default(),
            touch_phase: gpui::TouchPhase::Moved,
        });
        visual.run_until_parked();
        visual
            .refresh()
            .expect("Settled layout after precise scrolling");
        assert!(
            view.read_with(&visual, |view, _| view.scroll_handle.offset().y < before.y),
            "Real scroll input must make lower results reachable"
        );
        view.update(&mut visual, |view, cx| {
            view.scroll_handle
                .set_offset(gpui::point(gpui::px(0.), gpui::px(0.)));
            cx.notify();
        });
        visual.run_until_parked();
        visual.simulate_event(gpui::ScrollWheelEvent {
            position: viewport.center(),
            delta: gpui::ScrollDelta::Lines(gpui::point(0., -3.)),
            modifiers: gpui::Modifiers::default(),
            touch_phase: gpui::TouchPhase::default(),
        });
        visual.run_until_parked();
        visual
            .refresh()
            .expect("Settled layout after discrete scrolling");
        assert!(
            view.read_with(&visual, |view, _| view.scroll_handle.offset().y
                < gpui::px(0.)),
            "Discrete native wheel input must make lower results reachable"
        );
        assert_eq!(
            navigation,
            visual
                .debug_bounds("doctor-advanced")
                .expect("Fixed navigation survives scrolling")
        );
        click_investigation_control(&view, &mut visual, "doctor-advanced");
        assert!(view.read_with(&visual, |view, _| view.advanced_open
            && view.report.is_some()));
        assert!(visual.debug_bounds("doctor-product-0").is_some());
        assert!(
            visual.debug_bounds("doctor-result-0").is_none(),
            "Advanced actions must not bury their controls beneath health results"
        );
        click_investigation_control(&view, &mut visual, "doctor-investigation-scope-1");
        assert!(visual.debug_bounds("doctor-result-0").is_some());
        assert_eq!(
            view.read_with(&visual, |view, _| view.scroll_handle.offset().y),
            gpui::px(0.)
        );
    }

    #[gpui::test]
    async fn doctor_view_scope_native_action_requires_root_and_collects_only_on_click(
        cx: &mut TestAppContext,
    ) {
        let (project, workspace, window) = fixture(cx, true).await;
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        cx.dispatch_action(Doctor);
        cx.run_until_parked();
        let view = workspace
            .read_with(cx, |workspace, cx| workspace.item_of_type::<DoctorView>(cx))
            .expect("Native Doctor item");
        view.read_with(cx, |view, _| {
            assert!(view.selected_worktree.is_none());
            assert!(view.report.is_none());
            assert!(view.collection.is_none());
        });
        assert!(
            cx.debug_bounds("made-doctor").is_some(),
            "Production Doctor item must render in Workspace"
        );
        let root = cx
            .debug_bounds("doctor-worktree")
            .expect("Native root control");
        cx.simulate_click(root.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        let selected = view.read_with(cx, |view, _| {
            view.selected_worktree.expect("Explicitly chosen worktree")
        });
        assert!(project.read_with(cx, |owner, cx| {
            owner.worktree_for_id(selected, cx).is_some()
        }));
        assert!(view.read_with(cx, |view, _| view.report.is_none()));
        view.update(cx, |view, cx| {
            view.selected_product = DoctorProduct::Made;
            view.invalidate(cx);
        });
        cx.run_until_parked();
        let collect = cx
            .debug_bounds("doctor-refresh")
            .expect("Native collect control");
        cx.simulate_click(collect.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            let report = view.report.as_ref().expect("Observed checks");
            assert_eq!(report.scope.product.as_str(), product_flavor::DISPLAY_NAME);
            assert!(report.scope.host.as_str().contains("Local"));
            assert_eq!(report.scope.worktree.as_str(), "second");
            assert!(
                report
                    .checks()
                    .iter()
                    .any(|check| check.id.as_str() == "project.authority")
            );
            assert!(view.collection.is_none());
        });
        let first = view.entity_id();
        cx.dispatch_action(Doctor);
        cx.run_until_parked();
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .item_of_type::<DoctorView>(cx)
                .expect("Reused view")
                .entity_id()),
            first
        );
        #[cfg(feature = "agentic")]
        cx.update(|_, cx| {
            assert!(language_model::LanguageModelRegistry::try_global(cx).is_none());
            assert!(agent::recipes::RecipeStore::try_global(cx).is_none());
            assert!(extension_host::ExtensionStore::try_global(cx).is_none());
        });
    }

    #[gpui::test]
    async fn doctor_view_scope_closed_project_and_removed_root_reject_collection(
        cx: &mut TestAppContext,
    ) {
        let (project, workspace, window) = fixture(cx, false).await;
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        cx.dispatch_action(Doctor);
        cx.run_until_parked();
        let view = workspace
            .read_with(cx, |workspace, cx| workspace.item_of_type::<DoctorView>(cx))
            .expect("Doctor");
        let root = view.read_with(cx, |view, _| view.selected_worktree.expect("Only root"));
        project.update(cx, |project, cx| project.remove_worktree(root, cx));
        cx.run_until_parked();
        view.update(cx, |view, cx| view.refresh(cx));
        view.read_with(cx, |view, _| {
            assert!(view.error.is_some());
            assert!(view.collection.is_none());
            assert!(view.report.is_none());
        });
        project.update(cx, |_, cx| cx.emit(project::Event::Closed));
        cx.run_until_parked();
        view.read_with(cx, |view, cx| {
            assert!(view.closed);
            assert!(view.directory(cx).is_none());
        });
    }

    #[gpui::test]
    async fn doctor_view_scope_selection_invalidates_results_and_disabled_checks_are_explicit(
        cx: &mut TestAppContext,
    ) {
        let (_project, workspace, window) = fixture(cx, false).await;
        let cx = &mut VisualTestContext::from_window(window.into(), cx);
        cx.dispatch_action(Doctor);
        cx.run_until_parked();
        let view = workspace
            .read_with(cx, |workspace, cx| workspace.item_of_type::<DoctorView>(cx))
            .expect("Doctor");
        for product in [DoctorProduct::Copper, DoctorProduct::Groovy] {
            view.update(cx, |view, cx| {
                view.selected_product = product;
                view.invalidate(cx);
                assert!(view.report.is_none());
                view.refresh(cx);
            });
            cx.run_until_parked();
            view.read_with(cx, |view, _| {
                let report = view.report.as_ref().expect("Explicit collection");
                assert_eq!(report.generation, view.generation);
                #[cfg(not(feature = "rust-tools"))]
                if product == DoctorProduct::Copper {
                    assert!(
                        report
                            .checks()
                            .iter()
                            .any(|check| check.id.as_str() == "rust.unavailable"
                                && check.outcome == CheckOutcome::Unsupported)
                    );
                }
                #[cfg(not(feature = "jvm-tools"))]
                if product == DoctorProduct::Groovy {
                    assert!(
                        report
                            .checks()
                            .iter()
                            .any(|check| check.id.as_str() == "jvm.unavailable"
                                && check.outcome == CheckOutcome::Unsupported)
                    );
                }
                #[cfg(not(feature = "agentic"))]
                assert!(
                    report
                        .checks()
                        .iter()
                        .any(|check| check.id.as_str() == "made.agent"
                            && check.outcome == CheckOutcome::Unsupported)
                );
            });
        }
    }
    fn fill_investigation_form(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        let fields = view.read_with(visual, |view, _| {
            vec![
                (view.investigation.symptom.clone(), "Selected test fails"),
                (view.investigation.target.clone(), "selected regression"),
                (
                    view.investigation.workload.clone(),
                    "one deterministic assertion",
                ),
                (
                    view.investigation.configuration.clone(),
                    "reviewed development configuration",
                ),
                (
                    view.investigation.cache.clone(),
                    "warm cache; no automatic cache deletion",
                ),
                (
                    view.investigation.writes.clone(),
                    "development build outputs only",
                ),
                (
                    view.investigation.subprocesses.clone(),
                    "existing test subprocesses; no external services",
                ),
                (
                    view.investigation.command.clone(),
                    "cargo test --offline selected_regression",
                ),
            ]
        });
        for (field, value) in fields {
            field.update_in(visual, |editor, window, cx| {
                editor.set_text(value, window, cx)
            });
        }
        visual.run_until_parked();
    }

    async fn prepare_investigation_ui(view: &Entity<DoctorView>, visual: &mut VisualTestContext) {
        view.update(visual, |view, cx| {
            view.scroll_handle
                .set_offset(gpui::point(gpui::px(0.), gpui::px(-900.)));
            cx.notify();
        });
        visual.run_until_parked();
        let review = visual
            .debug_bounds("doctor-investigation-review-command")
            .expect("Exact reproduction review control");
        visual.simulate_click(review.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .condition(view, |view, _| view.preparation.is_none())
            .await;
        view.update(visual, |_, cx| cx.notify());
        visual.run_until_parked();
    }

    #[gpui::test]
    async fn doctor_investigation_selection_scopes_keyboard_review_and_cancel_are_passive(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor::{
            DoctorInvestigationProblem as Problem, DoctorInvestigationScope as Scope,
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let form = visual
            .debug_bounds("doctor-investigation-form")
            .expect("Native investigation form control");
        visual.simulate_click(form.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let scope = visual
            .debug_bounds("doctor-investigation-scope-1")
            .expect("Development scope");
        visual.simulate_click(scope.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert_eq!(view.investigation.scope, Scope::Development);
            assert_eq!(view.investigation.problem, Problem::Build);
            assert!(view.proposal.is_none());
        });
        fill_investigation_form(&view, &mut visual);
        view.update(&mut visual, |view, cx| {
            view.investigation.development_only = true;
            cx.notify();
        });
        prepare_investigation_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            let Some(OwnerRepair::Reproduction(review)) = view.proposal.as_ref() else {
                panic!("Existing Task reproduction review: {:?}", view.error);
            };
            assert_eq!(
                review.action().task().resolved.command.as_deref(),
                Some("cargo test --offline selected_regression")
            );
            assert_eq!(
                review.action().task().resolved.save,
                task::SaveStrategy::None
            );
            assert!(
                review
                    .action()
                    .exact_preview()
                    .iter()
                    .any(|line| line.contains("not a filesystem, network"))
            );
            assert!(view.repair_task.is_none());
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Passive history")
                .is_empty()
        );
        click_investigation_control(&view, &mut visual, "doctor-cancel-repair");
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
        });
        view.update(&mut visual, |view, cx| {
            view.scroll_handle
                .set_offset(gpui::point(gpui::px(0.), gpui::px(0.)));
            cx.notify();
        });
        visual.run_until_parked();
        let scope = visual
            .debug_bounds("doctor-investigation-scope-2")
            .expect("Application scope");
        visual.simulate_click(scope.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert_eq!(view.investigation.scope, Scope::Application);
            assert_eq!(view.investigation.problem, Problem::IncorrectBehavior);
        });
        let target = view.read_with(&visual, |view, _| view.investigation.target.clone());
        target.update_in(&mut visual, |editor, window, cx| {
            editor.focus_handle(cx).focus(window, cx);
        });
        visual.update(|_, cx| {
            cx.bind_keys([gpui::KeyBinding::new(
                "cmd-a",
                editor::actions::SelectAll,
                Some("Editor"),
            )])
        });
        visual.simulate_keystrokes("cmd-a");
        visual.simulate_input("a different relevant target");
        visual.run_until_parked();
        view.read_with(&visual, |view, cx| {
            assert_eq!(
                view.investigation.target.read(cx).text(cx),
                "a different relevant target"
            )
        });
        assert_eq!(
            workspace.read_with(&visual, |workspace, _| workspace.project().entity_id()),
            view.read_with(&visual, |view, _| view.project.entity_id())
        );
    }

    #[gpui::test]
    async fn doctor_investigation_selection_invalid_effects_edits_and_scope_never_execute(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        view.update(&mut visual, |view, cx| {
            view.investigation.open = true;
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Development;
            view.investigation.problem = crate::doctor::DoctorInvestigationProblem::Test;
            cx.notify();
        });
        fill_investigation_form(&view, &mut visual);
        prepare_investigation_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(
                view.error
                    .as_ref()
                    .expect("Unknown effects remain blocked")
                    .as_str()
                    .contains("not approved")
            );
        });
        view.update(&mut visual, |view, cx| {
            view.investigation.development_only = true;
            cx.notify();
        });
        prepare_investigation_ui(&view, &mut visual).await;
        assert!(view.read_with(&visual, |view, _| view.proposal.is_some()));
        let command = view.read_with(&visual, |view, _| view.investigation.command.clone());
        command.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("changed reproduction", window, cx)
        });
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(
                view.proposal.is_none(),
                "Changed exact command requires new review"
            );
            assert!(view.repair_task.is_none());
        });
        let duration = view.read_with(&visual, |view, _| view.investigation.duration.clone());
        duration.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("301", window, cx)
        });
        prepare_investigation_ui(&view, &mut visual).await;
        view.read_with(&visual, |view, _| {
            assert!(view.proposal.is_none());
            assert!(
                view.error
                    .as_ref()
                    .expect("Bounded duration")
                    .as_str()
                    .contains("five minutes")
            );
        });
        view.update(&mut visual, |view, cx| {
            view.selected_worktree = None;
            view.invalidate(cx);
        });
        view.read_with(&visual, |view, cx| {
            assert!(
                view.investigation_selection(cx).is_err(),
                "No default host/worktree fallback"
            )
        });
        assert!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No admitted actions")
                .is_empty()
        );
    }
    #[gpui::test]
    async fn doctor_investigation_selection_multiroot_missing_providers_and_removed_host(
        cx: &mut TestAppContext,
    ) {
        let (project, workspace, window) = fixture(cx, true).await;
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.dispatch_action(Doctor);
        visual.run_until_parked();
        let view = workspace
            .read_with(&visual, |workspace, cx| {
                workspace.item_of_type::<DoctorView>(cx)
            })
            .expect("Actual Item");
        view.read_with(&visual, |view, cx| {
            assert!(view.directory(cx).is_none());
            assert!(view.investigation_selection(cx).is_err());
        });
        let root = visual
            .debug_bounds("doctor-worktree")
            .expect("Explicit second worktree choice");
        visual.simulate_click(root.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let directory = view.read_with(&visual, |view, cx| {
            view.directory(cx).expect("Explicit selected host")
        });
        assert!(project.read_with(&visual, |project, cx| {
            project
                .absolute_path(&directory, cx)
                .expect("Selected root")
                .ends_with("second")
        }));
        view.update(&mut visual, |view, cx| {
            view.investigation.open = true;
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Application;
            view.investigation.problem = crate::doctor::DoctorInvestigationProblem::Memory;
            view.refresh(cx);
        });
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.collection.is_none())
            .await;
        view.read_with(&visual, |view, _| {
            assert!(view.report.is_some());
            assert!(view.proposal.is_none());
            assert!(view.repair_task.is_none());
        });
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        view.read_with(&visual, |view, cx| {
            assert!(
                !view
                    .project
                    .read(cx)
                    .structured_execution_store()
                    .read(cx)
                    .state()
                    .providers()
                    .any(|provider| provider.status
                        == project::structured_execution::StructuredProviderStatus::Current)
            );
        });
        project.update(&mut visual, |project, cx| {
            project.remove_worktree(directory.worktree_id, cx)
        });
        visual.run_until_parked();
        view.read_with(&visual, |view, cx| {
            assert!(view.directory(cx).is_none());
            assert!(view.report.is_none());
            assert!(view.proposal.is_none());
            assert!(view.investigation_selection(cx).is_err());
        });
    }
    fn click_investigation_control(
        view: &Entity<DoctorView>,
        visual: &mut VisualTestContext,
        selector: &'static str,
    ) {
        visual.refresh().expect("Current Doctor controls");
        if visual.debug_bounds("made-doctor").is_none() {
            visual.dispatch_action(Doctor);
            visual
                .refresh()
                .expect("Return to the existing Doctor Item");
        }
        if visual.debug_bounds(selector).is_none()
            && !view.read_with(visual, |view, _| view.advanced_open)
        {
            click_investigation_control(view, visual, "doctor-advanced");
        }
        let bounds = visual.debug_bounds(selector).expect("Rendered control");
        if selector == "doctor-advanced" || selector.starts_with("doctor-investigation-scope-") {
            assert!(
                visual
                    .debug_bounds("made-doctor")
                    .expect("Doctor Item")
                    .contains(&bounds.center())
            );
            visual.simulate_click(bounds.center(), gpui::Modifiers::default());
            visual.run_until_parked();
            return;
        }
        view.update(visual, |view, cx| {
            let viewport = view.scroll_handle.bounds();
            let offset = view.scroll_handle.offset();
            view.scroll_handle.set_offset(gpui::point(
                offset.x,
                offset.y + viewport.center().y - bounds.center().y,
            ));
            cx.notify();
        });
        visual.run_until_parked();
        let bounds = visual.debug_bounds(selector).expect("Visible control");
        assert!(view.read_with(visual, |view, _| {
            view.scroll_handle.bounds().contains(&bounds.center())
        }));
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        visual.run_until_parked();
    }

    #[gpui::test]
    async fn doctor_investigation_view_progress_task_outcomes_cancel_and_history(
        cx: &mut TestAppContext,
    ) {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let (view, mut visual, history, workspace) = restart_ui(cx).await;
        let contexts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let _subscription = visual.update(|_, cx| {
            let contexts = contexts.clone();
            cx.subscribe(&view, move |view, _: &DoctorEvidenceUpdated, cx| {
                if let Some(context) = view.read(cx).conversation_context(cx) {
                    contexts.borrow_mut().push(context);
                }
            })
        });
        view.update(&mut visual, |view, cx| {
            view.investigation.open = true;
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Development;
            view.investigation.problem = crate::doctor::DoctorInvestigationProblem::Test;
            cx.notify();
        });
        fill_investigation_form(&view, &mut visual);
        view.update(&mut visual, |view, cx| {
            view.investigation.development_only = true;
            cx.notify();
        });
        let (started, launches) = async_channel::bounded(4);
        let (completion, results) = async_channel::bounded(4);
        let cancellations = Arc::new(AtomicUsize::new(0));
        workspace.update(&mut visual, |workspace, _| {
            workspace.set_terminal_provider(crate::doctor::tests::DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: cancellations.clone(),
            })
        });
        for code in [1, 0] {
            prepare_investigation_ui(&view, &mut visual).await;
            click_investigation_control(&view, &mut visual, "doctor-approve-repair");
            let launch = {
                let raced = futures::future::select(
                    Box::pin(launches.recv()),
                    Box::pin(visual.condition(&view, |view, _| view.repair_task.is_none())),
                )
                .await;
                match raced {
                    futures::future::Either::Left((launch, pending)) => {
                        drop(pending);
                        Some(launch)
                    }
                    futures::future::Either::Right(((), pending)) => {
                        drop(pending);
                        None
                    }
                }
            };
            let launch = launch
                .unwrap_or_else(|| {
                    panic!(
                        "Action ended before launch: {}",
                        view.read_with(&visual, |view, _| format!(
                            "error={:?}; status={:?}; proposal={}; phase={:?}",
                            view.error,
                            view.repair_status,
                            view.proposal.is_some(),
                            view.owner_result.as_ref().map(OwnerRepair::phase)
                        ))
                    )
                })
                .expect("Existing structured Task owner");
            assert!(
                launch
                    .command
                    .as_deref()
                    .expect("Exact command")
                    .contains("selected_regression")
            );
            assert_eq!(launch.save, task::SaveStrategy::None);
            assert!(!launch.show_rerun);
            view.read_with(&visual, |view, _| {
                assert!(view.repair_task.is_some());
                assert!(view.repair_cancel.is_some());
            });
            visual.run_until_parked();
            {
                let contexts = contexts.borrow();
                let context = contexts
                    .last()
                    .expect("Running action delivered to conversation");
                assert!(context.contains("selected_regression"), "{context}");
                assert!(context.contains("Action status: Running"), "{context}");
                assert!(
                    !context.contains("CommandExit"),
                    "Previous exit cannot describe a new run: {context}"
                );
            }
            completion.send(code).await.expect("Observed fixture exit");
            visual
                .condition(&view, |view, _| view.repair_task.is_none())
                .await;
            visual.run_until_parked();
            {
                let contexts = contexts.borrow();
                let context = contexts
                    .last()
                    .expect("Observed result delivered to conversation");
                assert!(context.contains("selected_regression"), "{context}");
                assert!(
                    context.contains(&format!("CommandExit {{ code: Some({code}) }}")),
                    "{context}"
                );
                assert!(
                    context.contains("do not establish general correctness"),
                    "{context}"
                );
            }
            view.read_with(&visual, |view, _| {
                assert_eq!(
                    view.owner_result.as_ref().expect("Observed result").phase(),
                    crate::doctor::DoctorRepairPhase::Incomplete
                );
                let evidence = view
                    .report
                    .as_ref()
                    .expect("Report")
                    .investigation()
                    .expect("Bounded observations")
                    .evidence();
                assert!(evidence.iter().any(|row| row.observation
                    == crate::doctor::DoctorEvidenceObservation::CommandExit { code: Some(code) }));
                assert!(view.current_comparison.is_some());
                assert!(view.report.as_ref().expect("Report").comparison().is_none());
            });
            view.update(&mut visual, |view, cx| {
                view.select_comparison_series(code == 1, false, cx)
            });
        }
        let revealed = Arc::new(AtomicUsize::new(0));
        let task = view.read_with(&visual, |view, _| {
            view.task_result.clone().expect("Original Task handle")
        });
        task.set_reveal({
            let revealed = revealed.clone();
            move |_, _| {
                revealed.fetch_add(1, Ordering::Relaxed);
                true
            }
        });
        click_investigation_control(&view, &mut visual, "doctor-investigation-open-task");
        assert_eq!(revealed.load(Ordering::Relaxed), 1);
        assert!(launches.is_empty(), "Navigation cannot rerun the Task");
        task.set_reveal(|_, _| false);
        click_investigation_control(&view, &mut visual, "doctor-investigation-open-task");
        view.read_with(&visual, |view, _| {
            assert!(
                view.error
                    .as_ref()
                    .expect("Missing original terminal")
                    .as_str()
                    .contains("no longer available")
            )
        });
        assert!(
            launches.is_empty(),
            "Missing result cannot trigger a replacement execution"
        );
        view.update(&mut visual, |view, cx| {
            view.select_comparison_series(false, true, cx)
        });
        view.read_with(&visual, |view, _| {
            assert!(
                view.error
                    .as_ref()
                    .expect("Duplicate cannot count as repetition")
                    .as_str()
                    .contains("Duplicate")
            )
        });
        prepare_investigation_ui(&view, &mut visual).await;
        click_investigation_control(&view, &mut visual, "doctor-approve-repair");
        launches.recv().await.expect("Third action launched");
        view.update(&mut visual, |_, cx| cx.notify());
        visual.run_until_parked();
        let cancel = visual
            .debug_bounds("doctor-cancel-repair")
            .expect("Accessible persistent cancel control");
        visual.simulate_click(cancel.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| view.repair_task.is_none())
            .await;
        assert_eq!(cancellations.load(Ordering::Relaxed), 1);
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Observed durable receipts");
        assert_eq!(records.len(), 3);
        assert!(
            records
                .iter()
                .all(|record| record.outcome != crate::doctor::ReceiptOutcome::Completed)
        );
        assert!(
            records
                .iter()
                .any(|record| record.outcome == crate::doctor::ReceiptOutcome::Cancelled)
        );
        assert!(
            records
                .iter()
                .all(|record| record.report.investigation().is_some())
        );
        assert!(launches.is_empty(), "No replay or duplicate dispatch");
        assert_eq!(
            view.read_with(&visual, |view, _| view.project.clone())
                .read_with(&visual, |project, _| project.fs().clone())
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Unrelated source"),
            "unrelated"
        );
    }

    #[gpui::test]
    async fn doctor_investigation_view_restricted_capture_redaction_stale_save_and_historical_data(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        view.update(&mut visual, |view, cx| {
            view.investigation.open = true;
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Application;
            view.investigation.problem = crate::doctor::DoctorInvestigationProblem::Memory;
            cx.notify();
        });
        fill_investigation_form(&view, &mut visual);
        let symptom = view.read_with(&visual, |view, _| view.investigation.symptom.clone());
        symptom.update_in(&mut visual, |editor, window, cx| {
            editor.set_text(
                "Heap dump needed; API_KEY=DOCTOR_EXCLUDED_SECRET",
                window,
                cx,
            )
        });
        visual.run_until_parked();
        view.update(&mut visual, |view, cx| {
            view.record_restricted_investigation(cx)
        });
        let report = view.read_with(&visual, |view, _| {
            view.report.clone().expect("Policy disposition")
        });
        let summary = report
            .investigation()
            .expect("Selected restricted investigation");
        assert_eq!(
            summary
                .evidence()
                .first()
                .expect("Observed collection limit")
                .observation,
            crate::doctor::DoctorEvidenceObservation::InsufficientEvidence
        );
        assert!(
            summary
                .evidence()
                .first()
                .expect("Restriction")
                .detail
                .as_str()
                .contains("No excluded artifact was read")
        );
        assert!(report.comparison().is_none());
        assert!(
            !report
                .export()
                .expect("Explicit redacted export")
                .contains("DOCTOR_EXCLUDED_SECRET")
        );
        view.update(&mut visual, |view, cx| {
            view.history_change = Some(HistoryChange::Save(report));
            view.record_restricted_investigation(cx);
        });
        view.read_with(&visual, |view, _| {
            assert!(
                view.history_change.is_none(),
                "Changed observations revoke pending report approval"
            )
        });
        view.update(&mut visual, |view, cx| {
            view.history_open = true;
            view.history_change = Some(HistoryChange::Save(
                view.report.clone().expect("Current summary"),
            ));
            view.approve_history_change(cx);
        });
        visual
            .condition(&view, |view, _| view.history_task.is_none())
            .await;
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Bounded history");
        assert_eq!(records.len(), 1);
        assert!(
            records
                .first()
                .expect("Record")
                .report
                .comparison()
                .is_none()
        );
        assert!(
            !serde_json::to_string(&records)
                .expect("Persisted summaries")
                .contains("DOCTOR_EXCLUDED_SECRET")
        );
        view.update(&mut visual, |view, cx| {
            view.history_selected = records.first().map(|record| record.id);
            view.history_records = records;
            cx.notify();
        });
        visual.run_until_parked();
        view.read_with(&visual, |view, _| {
            assert!(view.current_comparison.is_none());
            assert!(view.repair_task.is_none());
            assert!(view.proposal.is_none());
        });
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn doctor_investigation_flow_groovy_fresh_junit_rerun_and_relevant_comparison(
        cx: &mut TestAppContext,
    ) {
        use crate::doctor::*;
        use std::sync::{Arc, atomic::AtomicUsize};
        let (project, directory, window, _selected_node, _) =
            crate::doctor::tests::doctor_selected_java_fixture(cx).await;
        let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
        cx.update(|cx| {
            editor::init(cx);
            crate::init(cx);
            cx.set_global(history.clone());
        });
        let workspace = window
            .read_with(cx, |multi, _| multi.workspace().clone())
            .expect("Actual workspace");
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        let view = workspace.update_in(&mut visual, |workspace, window, cx| {
            DoctorView::deploy(workspace, &Doctor, window, cx);
            workspace
                .item_of_type::<DoctorView>(cx)
                .expect("Native Doctor Item")
        });
        view.update(&mut visual, |view, cx| {
            view.selected_product = DoctorProduct::Groovy;
            view.refresh(cx);
            view.investigation.open = true;
            view.investigation.scope = DoctorInvestigationScope::Application;
            view.investigation.problem = DoctorInvestigationProblem::Test;
        });
        visual.run_until_parked();
        fill_investigation_form(&view, &mut visual);
        view.update(&mut visual, |view, cx| {
            view.investigation.development_only = true;
            cx.notify();
        });
        let (started, launches) = async_channel::bounded(4);
        let (completion, results) = async_channel::bounded(4);
        workspace.update(&mut visual, |workspace, _| {
            workspace.set_terminal_provider(crate::doctor::tests::DoctorTaskFixture {
                started,
                completion: results,
                cancel_completion: completion.clone(),
                cancellation_count: Arc::new(AtomicUsize::new(0)),
            })
        });
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        let source = project
            .update(&mut visual, |project, cx| {
                project.open_local_buffer("/doctor-restart/src/test/java/GreetingTest.java", cx)
            })
            .await
            .expect("Existing source buffer");
        for failure in [true, false] {
            let provider = project
                .read_with(&visual, |project, _| {
                    project.jvm_test_provider_store().cloned()
                })
                .expect("Existing discovery owner");
            let discovery = provider
                .update(&mut visual, |provider, cx| provider.refresh(cx))
                .await
                .expect("Explicit selected-test discovery through the existing owner");
            let selected_node = discovery
                .nodes
                .iter()
                .find(|node| node.label == "messageIsAvailable")
                .expect("Current selected method")
                .id
                .clone();
            assert_eq!(
                discovery.status,
                project::structured_execution::StructuredProviderStatus::Stale,
                "Historical report discovery is not fresh verification"
            );
            let index = discovery
                .nodes
                .iter()
                .filter(|node| {
                    node.kind == project::structured_execution::StructuredNodeKind::Case
                        && discovery
                            .project_path_for_node(&node.id)
                            .is_some_and(|path| path.worktree_id == directory.worktree_id)
                })
                .position(|node| node.id == selected_node)
                .expect("Selected inherited-scope case");
            let historical_picker_index = project.read_with(&visual, |project, cx| {
                project
                    .structured_execution_store()
                    .read(cx)
                    .state()
                    .providers()
                    .filter(|provider| {
                        provider.status
                            == project::structured_execution::StructuredProviderStatus::Current
                            && !provider.partial
                    })
                    .flat_map(|provider| provider.nodes.iter())
                    .filter(|node| {
                        node.path
                            .as_ref()
                            .is_some_and(|path| path.worktree_id == directory.worktree_id)
                    })
                    .position(|node| node.id == selected_node)
            });
            assert!(
                historical_picker_index.is_none(),
                "Historical Current-only filtering would hide this runnable source-backed case"
            );
            let selector = match index {
                0 => "doctor-investigation-test-0",
                1 => "doctor-investigation-test-1",
                2 => "doctor-investigation-test-2",
                3 => "doctor-investigation-test-3",
                _ => panic!("Unexpected fixture node index {index}"),
            };
            click_investigation_control(&view, &mut visual, selector);
            visual
                .condition(&view, |view, _| view.preparation.is_none())
                .await;
            assert!(launches.is_empty(), "Review does not execute project code");
            view.read_with(&visual, |view, _| {
                let Some(OwnerRepair::Reproduction(review)) = view.proposal.as_ref() else {
                    panic!("Actual JVM owner plan: {:?}", view.error);
                };
                assert_eq!(
                    review.action().task().resolved.command.as_deref(),
                    Some("./mvnw")
                );
                assert_eq!(
                    review.action().task().resolved.args,
                    vec!["-Dtest=GreetingTest#messageIsAvailable", "test"]
                );
            });
            click_investigation_control(&view, &mut visual, "doctor-approve-repair");
            let task = launches.recv().await.expect("Existing Task owner launch");
            assert_eq!(task.save, task::SaveStrategy::None);
            let xml = if failure {
                "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"PRIVATE_FLOW_ASSERTION\"/></testcase></testsuite>"
            } else {
                "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            };
            filesystem
                .atomic_write(
                    std::path::PathBuf::from(
                        "/doctor-restart/target/surefire-reports/TEST-GreetingTest.xml",
                    ),
                    xml.into(),
                )
                .await
                .expect("Fresh fixture output read by actual JVM owner");
            completion
                .send(0)
                .await
                .expect("Task exit does not establish assertion outcome");
            visual.run_until_parked();
            visual
                .executor()
                .advance_clock(std::time::Duration::from_millis(200));
            visual
                .condition(&view, |view, _| view.repair_task.is_none())
                .await;
            view.read_with(&visual, |view, _| {
                let evidence = view
                    .report
                    .as_ref()
                    .expect("Report")
                    .investigation()
                    .expect("Owner observations")
                    .evidence();
                assert!(evidence.iter().any(|row| if failure {
                    matches!(
                        row.observation,
                        DoctorEvidenceObservation::FunctionalFailure { .. }
                    )
                } else {
                    matches!(
                        row.observation,
                        DoctorEvidenceObservation::FunctionalSuccess { .. }
                    )
                }));
                assert!(view.report.as_ref().expect("Report").comparison().is_none());
            });
            click_investigation_control(
                &view,
                &mut visual,
                if failure {
                    "doctor-investigation-series-0"
                } else {
                    "doctor-investigation-series-2"
                },
            );
            if failure {
                source.update(&mut visual, |source, cx| {
                    source.edit([(0..0, "// reviewed fixture source change\n")], None, cx)
                });
                project
                    .update(&mut visual, |project, cx| {
                        project.save_buffer(source.clone(), cx)
                    })
                    .await
                    .expect("Explicit reviewed fixture source save before the next Task");
                visual.run_until_parked();
            }
        }
        click_investigation_control(&view, &mut visual, "doctor-comparison-before-functional-0");
        click_investigation_control(&view, &mut visual, "doctor-comparison-after-functional-0");
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_comparison_change_fixture(
                "Reviewed source change followed by the same selected JUnit assertion",
                window,
                cx,
            )
        });
        click_investigation_control(&view, &mut visual, "doctor-investigation-compare");
        view.read_with(&visual, |view, _| {
            let comparison = view
                .report
                .as_ref()
                .expect("Report")
                .comparison()
                .unwrap_or_else(|| panic!("Relevant comparison: {:?}", view.error));
            assert_eq!(
                comparison.outcome,
                DoctorComparisonOutcome::VerifiedSelectedTarget
            );
        });
        click_investigation_control(&view, &mut visual, "doctor-history-save");
        click_investigation_control(&view, &mut visual, "doctor-history-confirm");
        visual
            .condition(&view, |view, _| view.history_task.is_none())
            .await;
        let records = history
            .records(compiled_product(), DoctorView::now().expect("Clock"))
            .await
            .expect("Durable observations");
        assert_eq!(records.len(), 3);
        let encoded = serde_json::to_string(&records).expect("Bounded history");
        assert!(!encoded.contains("PRIVATE_FLOW_ASSERTION") && !encoded.contains("<testsuite>"));
        assert!(
            records.iter().any(|record| record
                .report
                .comparison()
                .is_some_and(|comparison| comparison.outcome
                    == DoctorComparisonOutcome::VerifiedSelectedTarget))
        );
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/doctor-restart/source.rs"))
                .await
                .expect("Independent edits"),
            "unrelated"
        );
        assert!(
            launches.is_empty(),
            "Comparison, history and navigation cannot replay work"
        );
    }

    #[gpui::test]
    async fn doctor_investigation_view_selected_comparison_recheck_history_and_changed_source(
        cx: &mut TestAppContext,
    ) {
        let (view, mut visual, history, _) = restart_ui(cx).await;
        view.update(&mut visual, |view, cx| {
            view.investigation.open = true;
            view.investigation.scope = crate::doctor::DoctorInvestigationScope::Application;
            view.investigation.problem = crate::doctor::DoctorInvestigationProblem::Test;
            cx.notify();
        });
        fill_investigation_form(&view, &mut visual);
        let (project, directory, scope, generation) = view.read_with(&visual, |view, cx| {
            (
                view.project.clone(),
                view.directory(cx).expect("Selected root"),
                view.report.as_ref().expect("Report").scope.clone(),
                view.generation,
            )
        });
        let source = ProjectPath {
            path: util::rel_path::RelPath::from_unix_str("source.rs")
                .expect("Source")
                .into(),
            ..directory.clone()
        };
        let buffer = project
            .update(&mut visual, |project, cx| project.open_buffer(source, cx))
            .await
            .expect("Normal source owner");
        let make_snapshot = |view: &mut DoctorView, cx: &mut Context<DoctorView>, failure: bool| {
            let (_, selection, _, _) = view
                .investigation_selection(cx)
                .expect("Current owner input identity");
            let binding =
                crate::doctor::project_repair_binding(&project, directory.clone(), generation, cx)
                    .expect("Current authority");
            let now = DoctorView::now().expect("Clock");
            let mut investigation = crate::doctor::DoctorInvestigation::new(
                selection.clone(),
                scope.clone(),
                binding.clone(),
                now,
            )
            .expect("Selected fixture owner observation");
            let identifier = investigation
                .record(
                    &binding,
                    &selection,
                    crate::doctor::DoctorEvidenceInput {
                        owner: crate::doctor::DoctorEvidenceOwner::Test,
                        owner_reference: DiagnosticText::new(if failure {
                            "fixture:selected-before"
                        } else {
                            "fixture:selected-after"
                        }),
                        classification: crate::doctor::DoctorEvidenceClass::Observed,
                        observation: if failure {
                            crate::doctor::DoctorEvidenceObservation::FunctionalFailure {
                                assertion: DiagnosticText::new("same:selected:regression"),
                            }
                        } else {
                            crate::doctor::DoctorEvidenceObservation::FunctionalSuccess {
                                assertion: DiagnosticText::new("same:selected:regression"),
                            }
                        },
                        detail: DiagnosticText::new(
                            "Deterministic owner observation fixture, not live application proof",
                        ),
                        observed_at: now,
                        historical: false,
                    },
                )
                .expect("Selected observation");
            (
                investigation
                    .comparison_snapshot_at(&project, cx)
                    .expect("Live source/authority seal"),
                identifier,
            )
        };
        let (before, before_id) =
            view.update(&mut visual, |view, cx| make_snapshot(view, cx, true));
        view.update(&mut visual, |view, cx| {
            view.current_comparison = Some(before);
            view.select_comparison_series(true, false, cx);
        });
        buffer.update(&mut visual, |buffer, cx| {
            buffer.edit([(0..0, "reviewed fixture source change\n")], None, cx)
        });
        project
            .update(&mut visual, |project, cx| {
                project.save_buffer(buffer.clone(), cx)
            })
            .await
            .expect("Explicit reviewed fixture save before fresh observation");
        visual.run_until_parked();
        let (after, after_id) = view.update(&mut visual, |view, cx| make_snapshot(view, cx, false));
        view.update(&mut visual, |view, cx| {
            view.current_comparison = Some(after);
            view.select_comparison_series(false, false, cx);
            view.comparison_form.before_functional = Some(before_id);
            view.comparison_form.after_functional = Some(after_id);
        });
        let description = view.read_with(&visual, |view, _| view.comparison_form.change.clone());
        description.update_in(&mut visual, |editor, window, cx| {
            editor.set_text("Reviewed selected fixture source change", window, cx)
        });
        view.update(&mut visual, |view, cx| view.compare_investigation(cx));
        view.read_with(&visual, |view, _| {
            assert_eq!(
                view.report
                    .as_ref()
                    .expect("Compared report")
                    .comparison()
                    .expect("Relevant selected evidence")
                    .outcome,
                crate::doctor::DoctorComparisonOutcome::VerifiedSelectedTarget
            )
        });
        view.update(&mut visual, |view, cx| {
            view.history_change = Some(HistoryChange::Save(
                view.report.clone().expect("Current comparison"),
            ));
            view.approve_history_change(cx);
        });
        visual
            .condition(&view, |view, _| view.history_task.is_none())
            .await;
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("Comparison receipt")
                .first()
                .expect("Record")
                .report
                .comparison()
                .expect("Bounded saved comparison")
                .outcome,
            crate::doctor::DoctorComparisonOutcome::VerifiedSelectedTarget
        );
        buffer.update(&mut visual, |buffer, cx| {
            buffer.edit([(0..0, "later unrelated change\n")], None, cx)
        });
        view.update(&mut visual, |view, cx| view.compare_investigation(cx));
        view.read_with(&visual, |view, _| {
            assert!(
                view.error
                    .as_ref()
                    .expect("Changed live verification rejected")
                    .as_str()
                    .contains("changed")
            )
        });
        assert_eq!(
            history
                .records(compiled_product(), DoctorView::now().expect("Clock"))
                .await
                .expect("No fabricated new receipt")
                .len(),
            1
        );
    }
}
