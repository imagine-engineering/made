use crate::{EnumFeatureFlag, FeatureFlag, PresenceFlag, register_feature_flag};

pub struct NotebookFeatureFlag;

impl FeatureFlag for NotebookFeatureFlag {
    const NAME: &'static str = "notebooks";
    type Value = PresenceFlag;
}
register_feature_flag!(NotebookFeatureFlag);

pub struct PanicFeatureFlag;

impl FeatureFlag for PanicFeatureFlag {
    const NAME: &'static str = "panic";
    type Value = PresenceFlag;
}
register_feature_flag!(PanicFeatureFlag);

/// A feature flag for granting access to beta ACP features.
///
/// We reuse this feature flag for new betas, so don't delete it if it is not currently in use.
pub struct AcpBetaFeatureFlag;

impl FeatureFlag for AcpBetaFeatureFlag {
    const NAME: &'static str = "acp-beta";
    type Value = PresenceFlag;
}
register_feature_flag!(AcpBetaFeatureFlag);

pub struct DiffReviewFeatureFlag;

impl FeatureFlag for DiffReviewFeatureFlag {
    const NAME: &'static str = "diff-review";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }
}
register_feature_flag!(DiffReviewFeatureFlag);

/// Gates the `create_thread` and `list_agents_and_models` tools, which let
/// the agent spawn independent sibling threads that show up in the agent
/// panel sidebar.
pub struct CreateThreadToolFeatureFlag;

impl FeatureFlag for CreateThreadToolFeatureFlag {
    const NAME: &'static str = "create-thread-tool";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        true
    }
}
register_feature_flag!(CreateThreadToolFeatureFlag);

pub struct LspToolFeatureFlag;

impl FeatureFlag for LspToolFeatureFlag {
    const NAME: &'static str = "lsp-tool";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }
}
register_feature_flag!(LspToolFeatureFlag);

pub struct RenameToolFeatureFlag;

impl FeatureFlag for RenameToolFeatureFlag {
    const NAME: &'static str = "rename-tool";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        true
    }
}
register_feature_flag!(RenameToolFeatureFlag);

/// Controls how agent thread worktree chips are labeled in the sidebar.
#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumFeatureFlag)]
pub enum AgentThreadWorktreeLabel {
    #[default]
    Both,
    Worktree,
    Branch,
}

pub struct AgentThreadWorktreeLabelFlag;

impl FeatureFlag for AgentThreadWorktreeLabelFlag {
    const NAME: &'static str = "agent-thread-worktree-label";
    type Value = AgentThreadWorktreeLabel;

    fn enabled_for_staff() -> bool {
        false
    }
}
register_feature_flag!(AgentThreadWorktreeLabelFlag);

/// Wraps agent-run terminal commands in an OS-level sandbox where supported,
/// and applies the shared per-host network grants to the `fetch` tool and the
/// out-of-project write grants to the `create_directory` tool. When off,
/// these tools run with the agent's full ambient permissions, as they always
/// have.
pub struct SandboxingFeatureFlag;

impl FeatureFlag for SandboxingFeatureFlag {
    const NAME: &'static str = "sandboxing";
    type Value = PresenceFlag;

    fn enabled_for_all() -> bool {
        true
    }

    fn enabled_for_staff() -> bool {
        false
    }
}
register_feature_flag!(SandboxingFeatureFlag);

pub struct RustCoverageExperimentalFeatureFlag;

impl FeatureFlag for RustCoverageExperimentalFeatureFlag {
    const NAME: &'static str = "rust-coverage-experimental";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }

    fn enabled_for_all() -> bool {
        false
    }

    fn allow_local_override() -> bool {
        true
    }

    fn requires_local_opt_in() -> bool {
        true
    }

    fn description() -> &'static str {
        "Experimental: Rust coverage capture and presentation. Producing-Task authority, complete UI/lifecycle acceptance, remote and Windows producers, and installed collector certification remain incomplete."
    }
}
register_feature_flag!(RustCoverageExperimentalFeatureFlag);

pub struct RustNativeProfilingExperimentalFeatureFlag;

impl FeatureFlag for RustNativeProfilingExperimentalFeatureFlag {
    const NAME: &'static str = "rust-native-profiling-experimental";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }

    fn enabled_for_all() -> bool {
        false
    }

    fn allow_local_override() -> bool {
        true
    }

    fn requires_local_opt_in() -> bool {
        true
    }

    fn description() -> &'static str {
        "Experimental: macOS native Cargo capture and profile review. Selected-workload planner options, installed collector/shipping qualification, Intel and older-host support, and production viewer certification remain incomplete."
    }
}
register_feature_flag!(RustNativeProfilingExperimentalFeatureFlag);

pub struct RustNamedCargoPresetsExperimentalFeatureFlag;

impl FeatureFlag for RustNamedCargoPresetsExperimentalFeatureFlag {
    const NAME: &'static str = "rust-named-cargo-presets-experimental";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }

    fn enabled_for_all() -> bool {
        false
    }

    fn allow_local_override() -> bool {
        true
    }

    fn requires_local_opt_in() -> bool {
        true
    }

    fn description() -> &'static str {
        "Experimental: saved named Cargo presets and project-scoped selection. Complete preset resolution, native Test/Bench combinations, and current app/platform verification remain incomplete."
    }
}
register_feature_flag!(RustNamedCargoPresetsExperimentalFeatureFlag);

pub struct RustExactDoctestActionsExperimentalFeatureFlag;

impl FeatureFlag for RustExactDoctestActionsExperimentalFeatureFlag {
    const NAME: &'static str = "rust-exact-doctest-actions-experimental";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }

    fn enabled_for_all() -> bool {
        false
    }

    fn allow_local_override() -> bool {
        true
    }

    fn requires_local_opt_in() -> bool {
        true
    }

    fn description() -> &'static str {
        "Experimental: individually selected Rust doctest actions. Unique rustdoc selection, literal Task/DAP receiving processes, and current app/platform verification remain incomplete."
    }
}
register_feature_flag!(RustExactDoctestActionsExperimentalFeatureFlag);

pub struct CopperDoctor;

impl FeatureFlag for CopperDoctor {
    const NAME: &'static str = "copper-doctor";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }
    fn allow_local_override() -> bool {
        true
    }
    fn requires_local_opt_in() -> bool {
        true
    }
    fn description() -> &'static str {
        "Enable Copper Doctor and the /doctor command."
    }
}
register_feature_flag!(CopperDoctor);

pub struct CopperSchedules;

impl FeatureFlag for CopperSchedules {
    const NAME: &'static str = "copper-schedules";
    type Value = PresenceFlag;

    fn enabled_for_staff() -> bool {
        false
    }
    fn allow_local_override() -> bool {
        true
    }
    fn requires_local_opt_in() -> bool {
        true
    }
    fn description() -> &'static str {
        "Enable /schedules and its schedule management commands."
    }
}
register_feature_flag!(CopperSchedules);
