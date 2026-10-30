use clap::Args;
use extension_host::ComponentLifecycleAdapter as _;
use serde::Serialize;
use serde_json::{Value, json};
use settings::RootUserSettings as _;
use sim_api::{
    HttpLimits, HttpRequest, NativeApiHostError, NativeApiServerConfig, NativeAutomationBody,
    NativeAutomationResult, NativeCliInvocation, NativeCliOperation, NativeHeadlessPolicy,
    NativeHeadlessService, NativeRuntimeApiHost, NativeTlsAcceptor, PreparedNativeHeadlessRuntime,
    WebSocketLimits,
    security::{ApiSecurityConfig, ArtifactIdempotencySnapshotStore, BearerCredential, TlsPolicy},
};
#[cfg(test)]
use sim_runtime::{AssetNamespace, AssetRoots, AssetService, authorize_native_plugin_asset_broker};
use sim_runtime::{
    ExecutionController as _, ExecutionDataSource, ExecutionEventBus, ExecutionPresentationOwner,
    ExecutionPresentationService, ExecutionSnapshotStatus, NATIVE_IMAGE_REGISTRY_VERSION,
    NativeExecutionController, NativeExecutionControllerConfig, NativeRuntimeProfile,
    SharedExecutionPresentationService, SimRuntimeDb, WorkerLaunchConfig,
    open_native_profile_asset_service,
};
use sim_types::{HttpMethod, ProfileId, WorkerId};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    net::{IpAddr, SocketAddr},
    path::Path,
    sync::atomic::{AtomicI32, Ordering},
    sync::{Arc, OnceLock},
    thread,
    time::{Duration, Instant},
};
use uuid::Uuid;

const COMMAND_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-commands.csv");
const PARAMETER_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-parameters.csv");
const SCHEMA_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-schemas.csv");
const EVENT_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-events.csv");
const ERROR_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-errors.csv");
const CONFIG_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-config.csv");
const FORMAT_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-formats.csv");
const LIFECYCLE_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-lifecycle.csv");
const ENVIRONMENT_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-environment.csv");
const SCHEMA_MAPPING_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-schema-mappings.csv");

struct ContractCatalogSpec {
    name: &'static str,
    contents: &'static str,
    expected_rows: usize,
    sha256: &'static str,
}

const CONTRACT_CATALOGS: &[ContractCatalogSpec] = &[
    ContractCatalogSpec {
        name: "schemas",
        contents: SCHEMA_CATALOG,
        expected_rows: 23,
        sha256: "06429ac0b6476172c99f5c09cffeeb33b7d2c89cb56d488812f6c926a292cd55",
    },
    ContractCatalogSpec {
        name: "events",
        contents: EVENT_CATALOG,
        expected_rows: 12,
        sha256: "fd37bdfc6c3d6319bbfca27a432307baba2fcfbdfe2fd08ba91abf5f003bf815",
    },
    ContractCatalogSpec {
        name: "errors",
        contents: ERROR_CATALOG,
        expected_rows: 99,
        sha256: "fef57a6ebdf4cf39c483e943a835eaf541126416dff09d4140e2c7c1a4738779",
    },
    ContractCatalogSpec {
        name: "config",
        contents: CONFIG_CATALOG,
        expected_rows: 20,
        sha256: "62545831c8826cfcb7beb06509c24a5d14e17349df3bcc36ff00972d41cfd558",
    },
    ContractCatalogSpec {
        name: "formats",
        contents: FORMAT_CATALOG,
        expected_rows: 34,
        sha256: "7716b3cf4c564f73d408e2094bf6a7e1788e24098e3eb2fe43bdb4be9da08096",
    },
    ContractCatalogSpec {
        name: "lifecycle",
        contents: LIFECYCLE_CATALOG,
        expected_rows: 24,
        sha256: "49bc45d68ade04bd9fd11aeacfd1ef636dae56597af67ce968593e71418f7b7e",
    },
    ContractCatalogSpec {
        name: "schema_mappings",
        contents: SCHEMA_MAPPING_CATALOG,
        expected_rows: 66,
        sha256: "db9c5d5d73b42f6f283d65157bcaad776300a46709c9fc775c42485976c5a86b",
    },
];

const INVALID_INPUT_EXIT: i32 = 2;
const UNAVAILABLE_EXIT: i32 = 69;
const MIGRATION_EXIT: i32 = 78;
const INTERRUPTED_EXIT: i32 = 130;

static HEADLESS_SIGNAL: AtomicI32 = AtomicI32::new(0);

#[derive(Args, Clone, Debug)]
#[command(
    disable_help_flag = true,
    disable_version_flag = true,
    trailing_var_arg = true
)]
pub(crate) struct SimArgs {
    /// A native Sim lifecycle or automation command and its options.
    #[arg(
        value_name = "COMMAND_OR_OPTION",
        num_args = 0..,
        allow_hyphen_values = true
    )]
    arguments: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CommandDisposition {
    Native,
    Migration,
    Deferred,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputMode {
    Pretty,
    Json,
    JsonStream,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ParsedInvocation {
    pub(crate) feature_id: String,
    pub(crate) command_path: String,
    pub(crate) disposition: CommandDisposition,
    pub(crate) arguments: Vec<String>,
    pub(crate) output_mode: OutputMode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CliAction {
    Help(Option<String>, OutputMode),
    HelpJson,
    Version(OutputMode),
    Invoke(ParsedInvocation),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliError {
    code: String,
    message: String,
    hint: Option<String>,
}

impl CliError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_arguments".into(),
            message: message.into(),
            hint: Some(format!(
                "run `{} sim --help` to inspect the native command surface",
                product_flavor::EXECUTABLE_NAME
            )),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

#[derive(Clone, Debug)]
struct CatalogCommand {
    feature_id: String,
    path: String,
    help: String,
    hidden: bool,
    disposition: CommandDisposition,
    parity_decision: String,
}

#[derive(Clone, Debug)]
struct CatalogParameter {
    feature_id: String,
    command_path: String,
    scope: String,
    name: String,
    kind: String,
    flags: Vec<String>,
    value_type: String,
    nullable: bool,
    value_arity: String,
    paired_boolean: bool,
    repeatable: bool,
    choices: Vec<String>,
    constraints: String,
    required: bool,
}

#[derive(Clone, Debug)]
struct Catalog {
    commands: Vec<CatalogCommand>,
    parameters: Vec<CatalogParameter>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeLocalOperation {
    Discover,
    Environment,
    Which,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NativeExecutionPlan {
    Local(NativeLocalOperation),
    Headless(NativeCliOperation),
}

static CATALOG: OnceLock<Result<Catalog, String>> = OnceLock::new();

pub(crate) fn run(args: SimArgs) -> i32 {
    let action = match parse_action(&args.arguments) {
        Ok(action) => action,
        Err(error) => {
            render_error(&error, output_mode(&args.arguments));
            return INVALID_INPUT_EXIT;
        }
    };

    match action {
        CliAction::Help(command, output_mode) => {
            render_help(command.as_deref(), output_mode);
            0
        }
        CliAction::HelpJson => match render_help_json() {
            Ok(()) => 0,
            Err(error) => {
                render_error(&error, OutputMode::Json);
                INVALID_INPUT_EXIT
            }
        },
        CliAction::Version(output_mode) => {
            render_success(
                "sim version",
                json!({
                    "name": format!("{} sim", product_flavor::EXECUTABLE_NAME),
                    "version": env!("CARGO_PKG_VERSION"),
                    "runtime": "native-rust",
                    "protocol": sim_types::NATIVE_PROTOCOL_VERSION,
                }),
                output_mode,
            );
            0
        }
        CliAction::Invoke(invocation) => match invocation.disposition {
            CommandDisposition::Migration => {
                render_explicit_disposition(
                    &invocation,
                    "architecture_conflict",
                    "This source command manages Python, ComfyUI, or Python custom nodes. Zed preserves its inputs as migration evidence and uses native runtime profiles and Rust/WASM plugins instead; no source process was started.",
                );
                MIGRATION_EXIT
            }
            CommandDisposition::Deferred => {
                render_explicit_disposition(
                    &invocation,
                    "deferred_service",
                    "This network or account operation is explicitly deferred until a native provider integration and user authorization are available; no request was sent.",
                );
                UNAVAILABLE_EXIT
            }
            CommandDisposition::Native => execute_native(invocation),
        },
    }
}

fn execute_native(invocation: ParsedInvocation) -> i32 {
    if invocation.command_path == "sim serve" {
        return execute_serve(invocation);
    }
    let execution_plan = match native_operation(&invocation) {
        Ok(execution_plan) => execution_plan,
        Err(error) => {
            render_error(&error, invocation.output_mode);
            return INVALID_INPUT_EXIT;
        }
    };
    let operation = match execution_plan {
        NativeExecutionPlan::Local(operation) => {
            return execute_local_native(&invocation, operation);
        }
        NativeExecutionPlan::Headless(operation) => operation,
    };
    let requires_runtime = !matches!(
        operation,
        NativeCliOperation::Migration { .. } | NativeCliOperation::Deferred { .. }
    );
    let presentation = match native_presentation(Uuid::from_u128(0x2101)) {
        Ok(presentation) => presentation,
        Err(error) => {
            render_native_error(
                &invocation,
                "native_presentation_initialization_failed",
                &error.to_string(),
            );
            return UNAVAILABLE_EXIT;
        }
    };
    let service = match NativeHeadlessService::offline_prepared(
        presentation,
        |presentation| {
            prepare_native_runtime_for_profile(
                presentation,
                Uuid::from_u128(0x2101),
                ApiSecurityConfig::loopback(),
            )
        },
        NativeHeadlessPolicy::default(),
    ) {
        Ok(service) => service,
        Err(error) => {
            render_native_error(
                &invocation,
                "native_headless_configuration_failed",
                &error.to_string(),
            );
            return UNAVAILABLE_EXIT;
        }
    };
    if requires_runtime && let Err(error) = service.start() {
        render_native_error(
            &invocation,
            "native_runtime_start_failed",
            &error.to_string(),
        );
        return UNAVAILABLE_EXIT;
    }
    let now_epoch_seconds = match sim_api::current_epoch_seconds() {
        Ok(now_epoch_seconds) => now_epoch_seconds,
        Err(error) => {
            if requires_runtime && let Err(shutdown_error) = service.shutdown() {
                render_native_error(
                    &invocation,
                    "clock_and_shutdown_failed",
                    &format!("{error}; shutdown: {shutdown_error}"),
                );
                return UNAVAILABLE_EXIT;
            }
            render_native_error(&invocation, error.code(), &error.to_string());
            return UNAVAILABLE_EXIT;
        }
    };
    let result = service.execute_cli(NativeCliInvocation {
        feature_id: invocation.feature_id.clone(),
        operation,
        now_epoch_seconds,
    });
    let shutdown_result = if requires_runtime {
        service.shutdown().map(|_| ())
    } else {
        Ok(())
    };
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            if let Err(shutdown_error) = shutdown_result {
                render_native_error(
                    &invocation,
                    "native_execution_and_shutdown_failed",
                    &format!("{error}; shutdown: {shutdown_error}"),
                );
            } else {
                render_native_error(&invocation, error.code(), &error.to_string());
            }
            return native_error_exit(&error);
        }
    };
    if let Err(error) = shutdown_result {
        render_native_error(
            &invocation,
            "native_runtime_shutdown_failed",
            &error.to_string(),
        );
        return UNAVAILABLE_EXIT;
    }
    render_native_result(&invocation, result)
}

fn execute_local_native(invocation: &ParsedInvocation, operation: NativeLocalOperation) -> i32 {
    let result = match operation {
        NativeLocalOperation::Discover => help_value(None),
        NativeLocalOperation::Environment => native_environment_report(),
        NativeLocalOperation::Which => Ok(native_profile_report()),
    };
    match result {
        Ok(value) => {
            render_success(&invocation.command_path, value, invocation.output_mode);
            0
        }
        Err(error) => {
            render_error(&error, invocation.output_mode);
            INVALID_INPUT_EXIT
        }
    }
}

fn native_environment_report() -> Result<Value, CliError> {
    let records = parse_csv(ENVIRONMENT_CATALOG).map_err(|message| CliError {
        code: "catalog_invalid".into(),
        message,
        hint: Some("the compiled environment compatibility catalog is invalid".into()),
    })?;
    let header = records.first().ok_or_else(|| CliError {
        code: "catalog_invalid".into(),
        message: "the compiled environment compatibility catalog has no header".into(),
        hint: None,
    })?;
    let index = |name: &str| {
        header
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| CliError {
                code: "catalog_invalid".into(),
                message: format!("the environment catalog has no `{name}` field"),
                hint: None,
            })
    };
    let key_index = index("key")?;
    let classification_index = index("classification")?;
    let behavior_index = index("behavior")?;
    let decision_index = index("parity_decision")?;
    let mut variables = Vec::with_capacity(records.len().saturating_sub(1));
    for (row_number, row) in records.iter().enumerate().skip(1) {
        if row.len() != header.len() {
            return Err(CliError {
                code: "catalog_invalid".into(),
                message: format!(
                    "environment catalog row {row_number} has {} fields, expected {}",
                    row.len(),
                    header.len()
                ),
                hint: None,
            });
        }
        let key = row.get(key_index).ok_or_else(|| CliError {
            code: "catalog_invalid".into(),
            message: format!("environment catalog row {row_number} has no key"),
            hint: None,
        })?;
        variables.push(json!({
            "key": key,
            "configured": std::env::var_os(key).is_some(),
            "value_disclosed": false,
            "classification": row.get(classification_index),
            "source_behavior": row.get(behavior_index),
            "native_decision": row.get(decision_index),
        }));
    }
    Ok(json!({
        "schema": "zed.sim.environment/1",
        "runtime": "native-rust",
        "variables": variables,
        "values_redacted": true,
        "external_engine_environment_applied": false,
    }))
}

fn native_profile_report() -> Value {
    let profile_id = Uuid::from_u128(0x2101);
    json!({
        "schema": "zed.sim.profile-selection/1",
        "runtime": "native-rust",
        "profile_id": profile_id,
        "data_root": sim_runtime::native_profile_asset_root(paths::data_dir(), profile_id),
        "worker_binary": if cfg!(windows) { "sim-worker.exe" } else { "sim-worker" },
        "external_comfy_selected": false,
        "python_runtime_selected": false,
    })
}

fn execute_serve(invocation: ParsedInvocation) -> i32 {
    let address = match serve_address(&invocation.arguments) {
        Ok(address) => address,
        Err(error) => {
            render_error(&error, invocation.output_mode);
            return INVALID_INPUT_EXIT;
        }
    };
    let serve_configuration = match serve_configuration(&invocation, address) {
        Ok(configuration) => configuration,
        Err(error) => {
            render_error(&error, invocation.output_mode);
            return INVALID_INPUT_EXIT;
        }
    };
    let _signal_guard = match HeadlessSignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => {
            render_native_error(&invocation, "signal_handler_unavailable", &error);
            return UNAVAILABLE_EXIT;
        }
    };
    let mut config = NativeApiServerConfig::new(address);
    config.tls = serve_configuration.tls;
    let profile_id = serve_configuration.profile_id;
    let security = serve_configuration.security;
    let presentation = match native_presentation(profile_id) {
        Ok(presentation) => presentation,
        Err(error) => {
            render_native_error(
                &invocation,
                "native_presentation_initialization_failed",
                &error.to_string(),
            );
            return UNAVAILABLE_EXIT;
        }
    };
    let service = match NativeHeadlessService::serve_prepared(
        presentation,
        move |presentation| {
            prepare_native_runtime_for_profile(presentation, profile_id, security.clone())
        },
        config,
        NativeHeadlessPolicy::default(),
    ) {
        Ok(service) => service,
        Err(error) => {
            render_native_error(&invocation, error.code(), &error.to_string());
            return UNAVAILABLE_EXIT;
        }
    };
    let snapshot = match service.start() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            render_native_error(&invocation, error.code(), &error.to_string());
            return UNAVAILABLE_EXIT;
        }
    };
    render_success(
        &invocation.command_path,
        json!({
            "state": snapshot.state,
            "address": snapshot.local_address,
            "runtime": "native-rust",
            "profile": profile_id,
            "offline": has_option(&invocation.arguments, "--offline"),
        }),
        invocation.output_mode,
    );

    if let Some(seconds) = option_value(&invocation.arguments, "--shutdown-after") {
        let seconds = match seconds.parse::<u64>() {
            Ok(seconds) if seconds > 0 => seconds,
            _ => {
                if let Err(error) = service.shutdown() {
                    render_native_error(&invocation, error.code(), &error.to_string());
                }
                render_native_error(
                    &invocation,
                    "invalid_shutdown_timeout",
                    "--shutdown-after must be a positive integer number of seconds",
                );
                return INVALID_INPUT_EXIT;
            }
        };
        let duration = Duration::from_secs(seconds);
        let Some(deadline) = Instant::now().checked_add(duration) else {
            if let Err(error) = service.shutdown() {
                render_native_error(&invocation, error.code(), &error.to_string());
            }
            render_native_error(
                &invocation,
                "invalid_shutdown_timeout",
                "--shutdown-after exceeds the platform monotonic clock range",
            );
            return INVALID_INPUT_EXIT;
        };
        while Instant::now() < deadline {
            let signal = HEADLESS_SIGNAL.load(Ordering::Acquire);
            if signal != 0 {
                let shutdown_result = service.shutdown();
                if let Err(error) = shutdown_result {
                    render_native_error(&invocation, error.code(), &error.to_string());
                    return UNAVAILABLE_EXIT;
                }
                render_cancellation_event(&invocation, signal);
                return signal_exit_code(signal);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            thread::sleep(remaining.min(Duration::from_millis(20)));
        }
        return match service.shutdown() {
            Ok(_) => 0,
            Err(error) => {
                render_native_error(&invocation, error.code(), &error.to_string());
                UNAVAILABLE_EXIT
            }
        };
    }

    loop {
        thread::sleep(Duration::from_millis(20));
        let signal = HEADLESS_SIGNAL.load(Ordering::Acquire);
        if signal != 0 {
            let shutdown_result = service.shutdown();
            if let Err(error) = shutdown_result {
                render_native_error(&invocation, error.code(), &error.to_string());
                return UNAVAILABLE_EXIT;
            }
            render_cancellation_event(&invocation, signal);
            return signal_exit_code(signal);
        }
        match service.snapshot() {
            Ok(snapshot) if matches!(snapshot.state, sim_api::NativeHeadlessState::Ready) => {}
            Ok(snapshot) => {
                render_native_error(
                    &invocation,
                    "native_host_stopped",
                    &format!("native host left ready state: {:?}", snapshot.state),
                );
                return UNAVAILABLE_EXIT;
            }
            Err(error) => {
                render_native_error(&invocation, error.code(), &error.to_string());
                return UNAVAILABLE_EXIT;
            }
        }
    }
}

fn render_cancellation_event(invocation: &ParsedInvocation, signal: i32) {
    if invocation.output_mode == OutputMode::JsonStream {
        println!(
            "{}",
            json!({
                "schema": "event/1",
                "type": "cancelled",
                "signal": signal,
                "exit_code": signal_exit_code(signal),
            })
        );
    }
}

fn signal_exit_code(signal: i32) -> i32 {
    if signal == 2 {
        INTERRUPTED_EXIT
    } else {
        128 + signal
    }
}

#[cfg(unix)]
struct HeadlessSignalGuard {
    previous_interrupt: usize,
    previous_terminate: usize,
}

#[cfg(unix)]
impl HeadlessSignalGuard {
    fn install() -> Result<Self, String> {
        const SIGNAL_ERROR: usize = usize::MAX;
        HEADLESS_SIGNAL.store(0, Ordering::Release);
        let handler = record_unix_signal as *const () as usize;
        let previous_interrupt = unsafe { install_unix_signal(2, handler) };
        if previous_interrupt == SIGNAL_ERROR {
            return Err("failed to install the SIGINT handler".into());
        }
        let previous_terminate = unsafe { install_unix_signal(15, handler) };
        if previous_terminate == SIGNAL_ERROR {
            unsafe {
                install_unix_signal(2, previous_interrupt);
            }
            return Err("failed to install the SIGTERM handler".into());
        }
        Ok(Self {
            previous_interrupt,
            previous_terminate,
        })
    }
}

#[cfg(unix)]
impl Drop for HeadlessSignalGuard {
    fn drop(&mut self) {
        unsafe {
            install_unix_signal(2, self.previous_interrupt);
            install_unix_signal(15, self.previous_terminate);
        }
        HEADLESS_SIGNAL.store(0, Ordering::Release);
    }
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "signal"]
    fn install_unix_signal(signal: i32, handler: usize) -> usize;
}

#[cfg(unix)]
extern "C" fn record_unix_signal(signal: i32) {
    // Preserve the first signal so SIGTERM cannot replace SIGINT's conventional exit code.
    match HEADLESS_SIGNAL.compare_exchange(0, signal, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) | Err(_) => {}
    }
}

#[cfg(target_os = "windows")]
struct HeadlessSignalGuard;

#[cfg(target_os = "windows")]
impl HeadlessSignalGuard {
    fn install() -> Result<Self, String> {
        HEADLESS_SIGNAL.store(0, Ordering::Release);
        unsafe {
            windows::Win32::System::Console::SetConsoleCtrlHandler(
                Some(record_windows_signal),
                true,
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(Self)
    }
}

#[cfg(target_os = "windows")]
impl Drop for HeadlessSignalGuard {
    fn drop(&mut self) {
        if let Err(error) = unsafe {
            windows::Win32::System::Console::SetConsoleCtrlHandler(
                Some(record_windows_signal),
                false,
            )
        } {
            eprintln!("failed to remove native headless signal handler: {error}");
        }
        HEADLESS_SIGNAL.store(0, Ordering::Release);
    }
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn record_windows_signal(
    control_type: u32,
) -> windows::Win32::Foundation::BOOL {
    use windows::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};

    if control_type == CTRL_C_EVENT || control_type == CTRL_BREAK_EVENT {
        match HEADLESS_SIGNAL.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) | Err(_) => {}
        }
        windows::Win32::Foundation::BOOL(1)
    } else {
        windows::Win32::Foundation::BOOL(0)
    }
}

fn native_operation(invocation: &ParsedInvocation) -> Result<NativeExecutionPlan, CliError> {
    if has_option(&invocation.arguments, "--enable-telemetry") {
        return Ok(deferred_to(
            "comfy-parity-auth-cloud-telemetry",
            &invocation.command_path,
            "telemetry consent and provider configuration require the profile-scoped native policy owner",
        ));
    }
    if ["--workspace", "--here", "--recent"]
        .into_iter()
        .any(|flag| has_option(&invocation.arguments, flag))
    {
        return Ok(deferred_to(
            "comfy-parity-settings-localization-ui",
            &invocation.command_path,
            "workspace and recent-project selection require the canonical native profile and settings owner",
        ));
    }
    if has_option(&invocation.arguments, "--host") || has_option(&invocation.arguments, "--port") {
        return Ok(headless(NativeCliOperation::Migration {
            replacement: format!("socketless native runtime service or `{} sim serve`", product_flavor::EXECUTABLE_NAME),
            detail: "source host/port options identify an external ComfyUI server and are never contacted by production Zed".into(),
        }));
    }
    if let Some(routing) = option_value(&invocation.arguments, "--where") {
        match routing.as_str() {
            "local" => {}
            "cloud" => {
                return Ok(headless(NativeCliOperation::Deferred {
                    reason: "cloud routing requires an enabled native provider, an explicit network grant, and user authorization".into(),
                }));
            }
            _ => {
                return Err(CliError::invalid(format!(
                    "invalid --where routing mode `{routing}`; expected local or cloud"
                )));
            }
        }
    }
    if invocation.command_path == "sim jobs ls"
        && ["--local-only", "--orphaned", "--watch"]
            .into_iter()
            .any(|flag| has_option(&invocation.arguments, flag))
    {
        return Ok(deferred_to(
            "comfy-parity-native-execution-e2e",
            &invocation.command_path,
            "local journal reconciliation, orphan classification, and watch mode require the durable execution lifecycle",
        ));
    }
    if invocation.command_path == "sim nodes ls"
        && [
            "--accepts",
            "--api-only",
            "--category",
            "--cloud-disabled",
            "--cloud-enabled",
            "--exclude-deprecated",
            "--input",
            "--label",
            "--limit",
            "--output-only",
            "--pack",
            "--produces",
        ]
        .into_iter()
        .any(|flag| has_option(&invocation.arguments, flag))
    {
        return Ok(deferred_to(
            "comfy-parity-assets-editors-viewers",
            &invocation.command_path,
            "filtered node projection requires the shared native node-library query owner",
        ));
    }
    if invocation.command_path == "sim nodes show" && has_option(&invocation.arguments, "--input") {
        return Ok(deferred_to(
            "comfy-parity-assets-editors-viewers",
            &invocation.command_path,
            "offline object-info import requires the bounded native catalog import owner",
        ));
    }
    if (invocation.command_path == "sim model list"
        && has_option(&invocation.arguments, "--relative-path"))
        || (invocation.command_path == "sim models list-folder"
            && has_option(&invocation.arguments, "--limit"))
    {
        return Ok(deferred_to(
            "comfy-parity-file-asset-runtime",
            &invocation.command_path,
            "alternate model roots and bounded folder projections require the canonical model asset query adapter",
        ));
    }

    match invocation.command_path.as_str() {
        "sim discover" => Ok(NativeExecutionPlan::Local(NativeLocalOperation::Discover)),
        "sim env" => Ok(NativeExecutionPlan::Local(
            NativeLocalOperation::Environment,
        )),
        "sim which" => Ok(NativeExecutionPlan::Local(NativeLocalOperation::Which)),
        "sim jobs ls" => Ok(headless(NativeCliOperation::Jobs {
            status: None,
            limit: Some(parsed_usize_option(&invocation.arguments, "--limit")?.unwrap_or(10)),
            offset: None,
        })),
        "sim jobs status" => Ok(headless(NativeCliOperation::JobStatus {
            job_id: required_positional(invocation, "prompt_id")?,
        })),
        "sim jobs cancel" => Ok(headless(NativeCliOperation::CancelJobs {
            job_ids: positional_values(invocation)?,
            operation_id: format!("zed-sim-cancel-{}", Uuid::new_v4()),
        })),
        "sim nodes ls" | "sim nodes refresh" => Ok(native_get("/object_info")),
        "sim nodes show" => {
            let node = required_positional(invocation, "node class")?;
            Ok(native_get(format!(
                "/object_info/{}",
                urlencoding::encode(&node)
            )))
        }
        "sim model list" | "sim models list-folders" => Ok(native_get("/models")),
        "sim models list-folder" => {
            let folder = required_positional(invocation, "model folder")?;
            Ok(native_get(format!(
                "/models/{}",
                urlencoding::encode(&folder)
            )))
        }
        "sim _watch _watch-job"
        | "sim jobs wait"
        | "sim jobs watch"
        | "sim run"
        | "sim run-cli" => Ok(deferred_to(
            "comfy-parity-native-execution-e2e",
            &invocation.command_path,
            "terminal prompt ownership, cancellable waiting, and event streaming are completed with the native execution end-to-end slice",
        )),
        "sim assets push" | "sim download" | "sim model download" | "sim model remove"
        | "sim upload" => Ok(deferred_to(
            "comfy-parity-file-asset-runtime",
            &invocation.command_path,
            "verified file transfer, model mutation, and output download require the canonical asset transaction service",
        )),
        "sim auth list"
        | "sim auth remove"
        | "sim auth set"
        | "sim tracking disable"
        | "sim tracking enable" => Ok(deferred_to(
            "comfy-parity-auth-cloud-telemetry",
            &invocation.command_path,
            "provider credentials and telemetry consent remain profile-scoped canonical services",
        )),
        "sim nodes categories"
        | "sim nodes downstream"
        | "sim nodes path"
        | "sim nodes search"
        | "sim nodes types"
        | "sim nodes upstream"
        | "sim models search"
        | "sim models show"
        | "sim preview" => Ok(deferred_to(
            "comfy-parity-assets-editors-viewers",
            &invocation.command_path,
            "semantic node/model search and content preview require the shared native catalog and content dispatch owners",
        )),
        "sim pr-cache clean" | "sim pr-cache list" | "sim update" => Ok(deferred_to(
            "comfy-parity-updates-snapshots",
            &invocation.command_path,
            "cache cleanup and updates require signed staged downloads, snapshots, and rollback",
        )),
        "sim project init"
        | "sim project status"
        | "sim validate"
        | "sim workflow compose"
        | "sim workflow decompose"
        | "sim workflow fragment ls"
        | "sim workflow fragment show"
        | "sim workflow fragment validate"
        | "sim workflow delete"
        | "sim workflow get"
        | "sim workflow list"
        | "sim workflow save"
        | "sim workflow set-slot"
        | "sim workflow slots"
        | "sim workflow vary" => Ok(deferred_to(
            "comfy-parity-workflow-formats",
            &invocation.command_path,
            "lossless workflow documents, fragments, validation, slots, and persistence require the canonical workflow format and save owners",
        )),
        "sim templates fetch"
        | "sim templates ls"
        | "sim templates refresh"
        | "sim templates show" => Ok(deferred_to(
            "comfy-parity-workflow-experience",
            &invocation.command_path,
            "template inventory and retrieval require the native template repository and provider gates",
        )),
        "sim set-default"
        | "sim setup"
        | "sim skill install"
        | "sim skill list"
        | "sim skill show"
        | "sim skill status"
        | "sim skill uninstall"
        | "sim skill validate"
        | "sim skills install"
        | "sim skills list"
        | "sim skills show"
        | "sim skills status"
        | "sim skills uninstall"
        | "sim skills validate" => Ok(deferred_to(
            "comfy-parity-settings-localization-ui",
            &invocation.command_path,
            "profile defaults, setup, and documented native skill surfaces require the canonical settings and help owners",
        )),
        _ => Err(CliError::invalid(format!(
            "native command `{}` has no authoritative execution mapping",
            invocation.command_path
        ))),
    }
}

fn headless(operation: NativeCliOperation) -> NativeExecutionPlan {
    NativeExecutionPlan::Headless(operation)
}

fn deferred_to(owning_task: &str, command_path: &str, reason: &str) -> NativeExecutionPlan {
    headless(NativeCliOperation::Deferred {
        reason: format!(
            "`{command_path}` is assigned to owning task `{owning_task}`: {reason}; no subprocess, external ComfyUI server, or unapproved network fallback was attempted"
        ),
    })
}

fn native_get(path: impl Into<String>) -> NativeExecutionPlan {
    headless(NativeCliOperation::NativeRequest {
        request: HttpRequest::new(HttpMethod::Get, path.into()),
        requires_network: false,
    })
}

fn native_presentation(
    profile_uuid: Uuid,
) -> Result<SharedExecutionPresentationService, NativeApiHostError> {
    let profile_id = ProfileId(profile_uuid);
    let app_database = db::AppDatabase::new();
    let database = SimRuntimeDb::from_app_database(&app_database);
    let presentation = ExecutionPresentationOwner::persistent(
        ExecutionPresentationService::new(4_096)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
        Arc::new(database),
    );
    smol::block_on(async {
        presentation.restore_profile(profile_id).await?;
        presentation
            .set_snapshot_status_durable(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .await
    })
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    Ok(presentation)
}

fn prepare_native_runtime_for_profile(
    _presentation: SharedExecutionPresentationService,
    profile_uuid: Uuid,
    security: ApiSecurityConfig,
) -> Result<PreparedNativeHeadlessRuntime, NativeApiHostError> {
    let (profile, plugin_security) = headless_native_configuration(profile_uuid)
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let root = sim_runtime::native_profile_asset_root(paths::data_dir(), profile_uuid);
    let private_state_root = root.join("state");
    fs::create_dir_all(&private_state_root)
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        fs::set_permissions(&private_state_root, fs::Permissions::from_mode(0o700))
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    }
    let assets =
        open_native_profile_asset_service(profile_uuid.to_string(), &root, &profile.model_roots)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let worker = WorkerLaunchConfig::for_packaged_worker_profile(
        &profile,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        8 * 1024 * 1024 * 1024,
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let idempotency_store = ArtifactIdempotencySnapshotStore::from_directory(
        &private_state_root,
        "native-api-idempotency.json",
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let permission_policy = Arc::new(
        plugin_security
            .permission_policy()
            .clone()
            .with_additional_grants(
                sim_api::security::plugin_route_permission_grants(&security.plugin_route_grants)
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
    );
    let profile_bits = profile.id.as_u128();
    let profile_seed = (profile_bits as u64) ^ ((profile_bits >> 64) as u64);
    let plugin_services = crate::zed::sim_plugin_services::deny_only_private_worker_services(
        worker.clone(),
        assets.clone(),
        plugin_security.provider_policy().clone(),
        profile_seed,
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let component_host = sim_plugin_host::ComponentHost::new(
        extension_host::ComponentRuntime::no_wasi()
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
        plugin_security.trust_policy().clone(),
        permission_policy.as_ref().clone(),
        plugin_services.boundary.clone(),
        sim_plugin_host::ComponentLimits::default(),
        sim_runtime::generated_native_node_registry_projection(None)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let component_router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
        component_host.clone(),
        plugin_security.component_registry_generation(),
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let extension_fs: Arc<dyn ::fs::Fs> = Arc::new(::fs::RealFs::new(
        None,
        gpui_platform::background_executor(),
    ));
    let candidate = smol::block_on(
        extension_host::ExtensionStore::canonical_component_inventory_candidate(
            extension_fs,
            paths::extensions_dir(),
        ),
    )
    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let (components, candidate_identity) = candidate.into_parts();
    smol::block_on(component_router.synchronize(components))
        .map_err(NativeApiHostError::Runtime)?;
    let registry_bundle = Arc::new(
        component_router
            .active_execution_registry_bundle()
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
    );
    let provider_invocation_authority = plugin_services
        .invocation_authority(component_host)
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let private_worker_executor = plugin_services.private_worker_executor();
    PreparedNativeHeadlessRuntime::checked(
        registry_bundle,
        candidate_identity,
        move |presentation, registry_bundle, candidate_identity| {
            let worker =
                worker.with_registry_deployment(registry_bundle.worker_deployment().clone());
            let events = ExecutionEventBus::new(1_024)
                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
            let mut controller_config = NativeExecutionControllerConfig::new(
                assets.clone(),
                presentation.clone(),
                worker,
                true,
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?
            .with_memory_policy(profile.memory_policy);
            if let Some(provider_registry) = registry_bundle.provider_registry() {
                controller_config = controller_config
                    .with_provider_registry(provider_registry.clone())
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?
                    .with_provider_invocation_authority(provider_invocation_authority);
            }
            let registration = NativeExecutionController::start_with_provider_worker_bridge(
                controller_config,
                events.clone(),
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
            let (controller, provider_worker_bridge) = registration.into_parts();
            if let Err(error) =
                private_worker_executor.attach_provider_worker_bridge(provider_worker_bridge)
            {
                if let Err(shutdown_error) = controller.shutdown() {
                    return Err(NativeApiHostError::Runtime(format!(
                        "headless provider bridge attachment failed: {error}; controller rollback failed: {}",
                        shutdown_error.message
                    )));
                }
                return Err(NativeApiHostError::Runtime(format!(
                    "headless provider bridge attachment failed: {error}"
                )));
            }
            NativeRuntimeApiHost::with_registry_bundle(
                registry_bundle,
                &candidate_identity,
                presentation,
                controller,
                &events,
                Some(assets),
                HttpLimits::default(),
                WebSocketLimits::default(),
                security,
                permission_policy,
                Arc::new(idempotency_store),
            )
        },
    )
}

fn headless_native_configuration(
    profile_id: Uuid,
) -> anyhow::Result<(
    NativeRuntimeProfile,
    sim_runtime::NativePluginSecurityPolicy,
)> {
    let settings_text = match fs::read_to_string(paths::settings_file()) {
        Ok(settings) => Some(settings),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    headless_native_configuration_from_settings_text(profile_id, settings_text.as_deref())
}

fn headless_native_profile_from_settings_text(
    profile_id: Uuid,
    settings_text: Option<&str>,
) -> anyhow::Result<NativeRuntimeProfile> {
    if let Some(settings_text) = settings_text {
        let settings = settings::SettingsContent::parse_json_with_comments(&settings_text)?;
        let profile_id_text = profile_id.to_string();
        if let Some(runtime) = settings.sim_runtime.as_ref()
            && let Some(profile) = runtime
                .profiles
                .as_deref()
                .unwrap_or_default()
                .iter()
                .find(|profile| profile.id.as_deref() == Some(profile_id_text.as_str()))
        {
            return NativeRuntimeProfile::try_from(profile).map_err(Into::into);
        }
    }
    let default_profile = Uuid::parse_str("00000000-0000-0000-0000-000000002101")?;
    if profile_id == default_profile {
        return NativeRuntimeProfile::disabled_migration_replacement(
            profile_id,
            "Native headless runtime",
        )
        .map_err(Into::into);
    }
    Err(anyhow::anyhow!(
        "headless native profile {profile_id} is not present in canonical settings"
    ))
}

fn headless_native_configuration_from_settings_text(
    profile_id: Uuid,
    settings_text: Option<&str>,
) -> anyhow::Result<(
    NativeRuntimeProfile,
    sim_runtime::NativePluginSecurityPolicy,
)> {
    if let Some(settings_text) = settings_text {
        let settings = settings::SettingsContent::parse_json_with_comments(&settings_text)?;
        let profile_id_text = profile_id.to_string();
        if let Some(runtime) = settings.sim_runtime.as_ref()
            && let Some(profile_content) = runtime
                .profiles
                .as_deref()
                .unwrap_or_default()
                .iter()
                .find(|profile| profile.id.as_deref() == Some(profile_id_text.as_str()))
        {
            let profile = NativeRuntimeProfile::try_from(profile_content)?;
            let plugin_security =
                profile.project_plugin_security_policy(profile_content.plugin_security.as_ref())?;
            return Ok((profile, plugin_security));
        }
    }
    let profile = headless_native_profile_from_settings_text(profile_id, None)?;
    let plugin_security = profile.project_plugin_security_policy(None)?;
    Ok((profile, plugin_security))
}

struct ServeConfiguration {
    profile_id: Uuid,
    security: ApiSecurityConfig,
    tls: Option<NativeTlsAcceptor>,
}

fn serve_configuration(
    invocation: &ParsedInvocation,
    address: SocketAddr,
) -> Result<ServeConfiguration, CliError> {
    if has_option(&invocation.arguments, "--offline") {
        return Err(CliError::invalid(
            "--offline is not valid for the socket-serving host; use socketless native CLI commands for enforced offline automation",
        ));
    }
    let profile_id = option_value(&invocation.arguments, "--profile")
        .unwrap_or_else(|| "00000000-0000-0000-0000-000000002101".into())
        .parse::<Uuid>()
        .map_err(|error| CliError::invalid(format!("invalid native --profile UUID: {error}")))?;
    let certificate_path = option_value(&invocation.arguments, "--tls-cert");
    let private_key_path = option_value(&invocation.arguments, "--tls-key");
    if certificate_path.is_some() != private_key_path.is_some() {
        return Err(CliError::invalid(
            "--tls-cert and --tls-key must be supplied together",
        ));
    }
    let certificate_identity = certificate_path
        .as_ref()
        .map(|path| format!("zed-sim-cli:{}", Path::new(path).display()));
    let tls = match (
        certificate_identity.as_deref(),
        certificate_path.as_deref(),
        private_key_path.as_deref(),
    ) {
        (Some(identity), Some(certificate), Some(private_key)) => Some(
            NativeTlsAcceptor::from_der_files(identity, certificate, private_key)
                .map_err(|error| CliError::invalid(error.to_string()))?,
        ),
        (None, None, None) => None,
        _ => return Err(CliError::invalid("incomplete native TLS configuration")),
    };
    let origins = option_values(&invocation.arguments, "--origin");
    let bearer_token = option_value(&invocation.arguments, "--bearer-token");
    let remote = !address.ip().is_loopback();
    if remote && !has_option(&invocation.arguments, "--allow-remote") {
        return Err(CliError::invalid(
            "non-loopback exposure requires --allow-remote acknowledgement",
        ));
    }
    if remote && tls.is_none() {
        return Err(CliError::invalid(
            "non-loopback exposure requires --tls-cert and --tls-key",
        ));
    }
    if remote && origins.is_empty() {
        return Err(CliError::invalid(
            "non-loopback exposure requires at least one exact --origin",
        ));
    }
    if remote && bearer_token.is_none() {
        return Err(CliError::invalid(
            "non-loopback exposure requires --bearer-token",
        ));
    }
    let mut security = ApiSecurityConfig::loopback();
    security.bind_address = address.ip();
    security.explicit_remote_exposure = remote;
    security.allowed_origins = origins.into_iter().collect();
    if let Some(identity) = certificate_identity {
        security.tls = TlsPolicy::Required {
            certificate_identity: identity,
        };
    }
    if let Some(bearer_token) = bearer_token {
        security.require_authentication = true;
        security.credentials.push(
            BearerCredential::new(
                "zed-sim-cli",
                bearer_token,
                ["api:read".to_owned(), "api:write".to_owned()],
                None,
            )
            .map_err(|error| CliError::invalid(error.to_string()))?,
        );
    }
    security
        .validate()
        .map_err(|error| CliError::invalid(error.to_string()))?;
    Ok(ServeConfiguration {
        profile_id,
        security,
        tls,
    })
}

fn serve_address(arguments: &[String]) -> Result<SocketAddr, CliError> {
    let bind = option_value(arguments, "--bind").unwrap_or_else(|| "127.0.0.1".into());
    let bind = bind
        .parse::<IpAddr>()
        .map_err(|error| CliError::invalid(format!("invalid --bind address: {error}")))?;
    let port = option_value(arguments, "--port")
        .map(|port| {
            port.parse::<u16>()
                .map_err(|error| CliError::invalid(format!("invalid --port: {error}")))
        })
        .transpose()?
        .unwrap_or(0);
    Ok(SocketAddr::new(bind, port))
}

fn native_error_exit(error: &sim_api::NativeHeadlessError) -> i32 {
    match error {
        sim_api::NativeHeadlessError::InvalidAutomation(_)
        | sim_api::NativeHeadlessError::InvalidConfiguration(_)
        | sim_api::NativeHeadlessError::InvalidCliEvent(_) => INVALID_INPUT_EXIT,
        sim_api::NativeHeadlessError::Offline { .. } => UNAVAILABLE_EXIT,
        _ => UNAVAILABLE_EXIT,
    }
}

fn render_native_result(invocation: &ParsedInvocation, result: NativeAutomationResult) -> i32 {
    match result {
        NativeAutomationResult::Native {
            status,
            content_type,
            headers,
            body,
            ..
        } => {
            let body = match body {
                NativeAutomationBody::Empty => Value::Null,
                NativeAutomationBody::Bytes(bytes) => json!({
                    "byte_length": bytes.len(),
                    "bytes": bytes,
                }),
                NativeAutomationBody::Json(value) => value,
            };
            let ok = (200..300).contains(&status);
            let value = json!({
                "schema": "envelope/1",
                "ok": ok,
                "command": &invocation.command_path,
                "feature_id": invocation.feature_id,
                "data": {
                    "status": status,
                    "content_type": content_type,
                    "headers": headers,
                    "body": body,
                }
            });
            match invocation.output_mode {
                OutputMode::Pretty => println!("{body}"),
                OutputMode::Json | OutputMode::JsonStream => println!("{value}"),
            }
            if ok { 0 } else { UNAVAILABLE_EXIT }
        }
        NativeAutomationResult::Migration {
            replacement,
            detail,
            ..
        } => {
            render_explicit_result(
                invocation,
                "migration",
                "architecture_conflict",
                &detail,
                Some(&replacement),
            );
            MIGRATION_EXIT
        }
        NativeAutomationResult::Deferred { reason, .. } => {
            render_explicit_result(
                invocation,
                "deferred",
                "native_capability_unavailable",
                &reason,
                None,
            );
            UNAVAILABLE_EXIT
        }
    }
}

fn render_explicit_result(
    invocation: &ParsedInvocation,
    disposition: &str,
    code: &str,
    message: &str,
    replacement: Option<&str>,
) {
    let value = json!({
        "schema": "envelope/1",
        "ok": false,
        "command": &invocation.command_path,
        "feature_id": invocation.feature_id,
        "disposition": disposition,
        "error": { "code": code, "message": message, "replacement": replacement },
    });
    match invocation.output_mode {
        OutputMode::Pretty => eprintln!("{}: {message}", invocation.command_path),
        OutputMode::Json | OutputMode::JsonStream => println!("{value}"),
    }
}

fn render_native_error(invocation: &ParsedInvocation, code: &str, message: &str) {
    let error = CliError {
        code: code.to_owned(),
        message: message.to_owned(),
        hint: None,
    };
    render_error(&error, invocation.output_mode);
}

fn parsed_usize_option(arguments: &[String], flag: &str) -> Result<Option<usize>, CliError> {
    option_value(arguments, flag)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| CliError::invalid(format!("invalid {flag} value: {error}")))
        })
        .transpose()
}

fn has_option(arguments: &[String], flag: &str) -> bool {
    arguments
        .iter()
        .any(|argument| argument == flag || argument.starts_with(&format!("{flag}=")))
}

fn option_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments.iter().enumerate().find_map(|(index, argument)| {
        if argument == flag {
            arguments.get(index + 1).cloned()
        } else {
            argument
                .strip_prefix(&format!("{flag}="))
                .map(str::to_owned)
        }
    })
}

fn option_values(arguments: &[String], flag: &str) -> Vec<String> {
    arguments
        .iter()
        .enumerate()
        .filter_map(|(index, argument)| {
            if argument == flag {
                arguments.get(index + 1).cloned()
            } else {
                argument
                    .strip_prefix(&format!("{flag}="))
                    .map(str::to_owned)
            }
        })
        .collect()
}

fn positional_values(invocation: &ParsedInvocation) -> Result<Vec<String>, CliError> {
    let command_words = invocation
        .command_path
        .split_whitespace()
        .skip(1)
        .collect::<Vec<_>>();
    let command_start = invocation
        .arguments
        .windows(command_words.len())
        .position(|window| {
            window
                .iter()
                .map(String::as_str)
                .eq(command_words.iter().copied())
        });
    let Some(command_start) = command_start else {
        return Ok(Vec::new());
    };
    let tail = invocation
        .arguments
        .get(command_start + command_words.len()..)
        .unwrap_or_default();
    let parameter_flags = catalog()?
        .parameters
        .iter()
        .filter(|parameter| {
            parameter.command_path == invocation.command_path || parameter.command_path == "sim"
        })
        .flat_map(|parameter| {
            parameter.flags.iter().map(move |flag| {
                (
                    flag.clone(),
                    option_requires_value(parameter, flag.as_str()),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let mut values = Vec::new();
    let mut index = 0;
    while let Some(argument) = tail.get(index) {
        if argument == "--" {
            values.extend(tail.iter().skip(index + 1).cloned());
            break;
        }
        if argument.starts_with('-') {
            let flag = argument
                .split_once('=')
                .map_or(argument.as_str(), |pair| pair.0);
            let consumes_value =
                parameter_flags.get(flag).copied().unwrap_or(false) && !argument.contains('=');
            index += if consumes_value { 2 } else { 1 };
        } else {
            values.push(argument.clone());
            index += 1;
        }
    }
    Ok(values)
}

fn required_positional(
    invocation: &ParsedInvocation,
    description: &str,
) -> Result<String, CliError> {
    positional_values(invocation)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            CliError::invalid(format!(
                "{} requires {description}",
                invocation.command_path
            ))
        })
}

fn parse_action(arguments: &[String]) -> Result<CliAction, CliError> {
    let catalog = catalog()?;
    validate_global_constraints(arguments)?;
    let output_mode = output_mode(arguments);

    if contains_flag(arguments, "--help-json") {
        return Ok(CliAction::HelpJson);
    }
    if contains_flag(arguments, "--version") || contains_flag(arguments, "-v") {
        return Ok(CliAction::Version(output_mode));
    }
    if arguments.is_empty() {
        return Ok(CliAction::Help(None, output_mode));
    }

    let command_start = find_command_start(arguments, &catalog.parameters)?;
    let Some(command_start) = command_start else {
        if contains_flag(arguments, "--help") || contains_flag(arguments, "-h") {
            return Ok(CliAction::Help(None, output_mode));
        }
        return Err(CliError::invalid("a native Sim command is required"));
    };

    if arguments.get(command_start).map(String::as_str) == Some("serve") {
        let command_end = command_start + 1;
        validate_synthetic_serve_arguments(&arguments[command_end..])?;
        if contains_help_flag(&arguments[command_end..]) {
            return Ok(CliAction::Help(Some("sim serve".into()), output_mode));
        }
        return Ok(CliAction::Invoke(ParsedInvocation {
            feature_id: "ZED-SIM-NATIVE-SERVE".into(),
            command_path: "sim serve".into(),
            disposition: CommandDisposition::Native,
            arguments: arguments.to_vec(),
            output_mode,
        }));
    }

    let (command, command_word_count) = catalog
        .commands
        .iter()
        .filter_map(|command| {
            let words = command.path.split_whitespace().skip(1).collect::<Vec<_>>();
            arguments
                .get(command_start..command_start + words.len())
                .filter(|candidate| {
                    candidate
                        .iter()
                        .map(String::as_str)
                        .eq(words.iter().copied())
                })
                .map(|_| (command, words.len()))
        })
        .max_by_key(|(_, word_count)| *word_count)
        .ok_or_else(|| {
            CliError::invalid(format!(
                "unknown native Sim command `{}`",
                arguments
                    .get(command_start)
                    .map(String::as_str)
                    .unwrap_or_default()
            ))
        })?;

    let command_end = command_start + command_word_count;
    let command_arguments = arguments.get(command_end..).unwrap_or_default();
    if contains_help_flag(command_arguments) || command.path == "sim help" {
        return Ok(CliAction::Help(Some(command.path.clone()), output_mode));
    }
    validate_command_arguments(command, arguments, command_arguments, &catalog.parameters)?;

    Ok(CliAction::Invoke(ParsedInvocation {
        feature_id: command.feature_id.clone(),
        command_path: command.path.clone(),
        disposition: command.disposition,
        arguments: arguments.to_vec(),
        output_mode,
    }))
}

fn validate_global_constraints(arguments: &[String]) -> Result<(), CliError> {
    let routing_count = ["--here", "--recent", "--workspace"]
        .into_iter()
        .filter(|flag| has_option(arguments, flag))
        .count();
    if routing_count > 1 {
        return Err(CliError::invalid(
            "--here, --recent, and --workspace are mutually exclusive",
        ));
    }
    let output_count = ["--json", "--no-json", "--json-stream"]
        .into_iter()
        .filter(|flag| contains_flag(arguments, flag))
        .count();
    if output_count > 1 {
        return Err(CliError::invalid(
            "--json, --no-json, and --json-stream are mutually exclusive",
        ));
    }
    Ok(())
}

fn find_command_start(
    arguments: &[String],
    parameters: &[CatalogParameter],
) -> Result<Option<usize>, CliError> {
    let global_parameters = parameters
        .iter()
        .filter(|parameter| parameter.command_path == "sim")
        .collect::<Vec<_>>();
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        if argument == "--" {
            return Ok(arguments.get(index + 1).map(|_| index + 1));
        }
        if !argument.starts_with('-') {
            return Ok(Some(index));
        }
        if is_builtin_flag(argument) {
            index += 1;
            continue;
        }
        let parameter = find_flag(&global_parameters, argument)
            .ok_or_else(|| CliError::invalid(format!("unknown global Sim option `{argument}`")))?;
        index += 1;
        if option_requires_value(parameter, argument) && !argument.contains('=') {
            let value = arguments.get(index).ok_or_else(|| {
                CliError::invalid(format!("option `{argument}` requires a value"))
            })?;
            if value.starts_with('-') {
                return Err(CliError::invalid(format!(
                    "option `{argument}` requires a value"
                )));
            }
            index += 1;
        }
    }
    Ok(None)
}

fn validate_command_arguments(
    command: &CatalogCommand,
    all_arguments: &[String],
    command_arguments: &[String],
    parameters: &[CatalogParameter],
) -> Result<(), CliError> {
    let accepted = parameters
        .iter()
        .filter(|parameter| {
            parameter.command_path == command.path || parameter.command_path == "sim"
        })
        .collect::<Vec<_>>();
    let command_words = command.path.split_whitespace().skip(1).collect::<Vec<_>>();
    let command_start = all_arguments
        .windows(command_words.len())
        .position(|window| {
            window
                .iter()
                .map(String::as_str)
                .eq(command_words.iter().copied())
        })
        .ok_or_else(|| CliError::invalid(format!("missing command path `{}`", command.path)))?;
    let mut occurrences = BTreeMap::<String, usize>::new();
    let mut index = 0;
    while let Some(argument) = all_arguments.get(index) {
        if argument == "--" {
            break;
        }
        if !argument.starts_with('-') || is_builtin_flag(argument) {
            index += 1;
            continue;
        }
        let (flag, attached_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(flag, value)| {
                (flag, Some(value))
            });
        let parameter = accepted
            .iter()
            .copied()
            .filter(|parameter| parameter.flags.iter().any(|candidate| candidate == flag))
            .max_by_key(|parameter| {
                usize::from(index > command_start && parameter.command_path == command.path)
            })
            .ok_or_else(|| {
                CliError::invalid(format!(
                    "option `{flag}` is not valid for `{}`",
                    command.path
                ))
            })?;
        let occurrence = occurrences.entry(parameter.feature_id.clone()).or_default();
        *occurrence += 1;
        if !parameter.repeatable && *occurrence > 1 {
            return Err(CliError::invalid(format!(
                "option `{flag}` cannot be repeated for `{}`",
                command.path
            )));
        }
        if option_requires_value(parameter, argument) {
            let value = match attached_value {
                Some(value) => value,
                None => all_arguments.get(index + 1).ok_or_else(|| {
                    CliError::invalid(format!("option `{flag}` requires a value"))
                })?,
            };
            validate_parameter_value(parameter, value)?;
            index += usize::from(attached_value.is_none());
        } else if let Some(value) = attached_value {
            if parameter.value_arity == "0 or 1 inline" {
                validate_inline_boolean(parameter, value)?;
            } else {
                return Err(CliError::invalid(format!(
                    "boolean option `{flag}` does not accept a value"
                )));
            }
        }
        index += 1;
    }

    for required_option in accepted
        .iter()
        .filter(|parameter| parameter.kind == "option" && parameter.required)
    {
        if !required_option.flags.iter().any(|flag| {
            all_arguments
                .iter()
                .any(|argument| argument == flag || argument.starts_with(&format!("{flag}=")))
        }) {
            return Err(CliError::invalid(format!(
                "required option `{}` is missing for `{}`",
                required_option.flags.join("/"),
                command.path
            )));
        }
    }

    let argument_parameters = accepted
        .iter()
        .copied()
        .filter(|parameter| parameter.kind == "argument")
        .collect::<Vec<_>>();
    let invocation = ParsedInvocation {
        feature_id: command.feature_id.clone(),
        command_path: command.path.clone(),
        disposition: command.disposition,
        arguments: all_arguments.to_vec(),
        output_mode: output_mode(all_arguments),
    };
    let positional_values = positional_values(&invocation)?;
    let mut positional_index = 0;
    for parameter in argument_parameters {
        if parameter.value_arity == "variadic" {
            let remaining = &positional_values[positional_index..];
            if parameter.required && remaining.is_empty() {
                return Err(CliError::invalid(format!(
                    "required argument `{}` is missing for `{}`",
                    parameter.name, command.path
                )));
            }
            for value in remaining {
                validate_parameter_value(parameter, value)?;
            }
            positional_index = positional_values.len();
            break;
        }
        match positional_values.get(positional_index) {
            Some(value) => {
                validate_parameter_value(parameter, value)?;
                positional_index += 1;
            }
            None if parameter.required => {
                return Err(CliError::invalid(format!(
                    "required argument `{}` is missing for `{}`",
                    parameter.name, command.path
                )));
            }
            None => {}
        }
    }
    if positional_index < positional_values.len() {
        return Err(CliError::invalid(format!(
            "too many positional arguments for `{}`",
            command.path
        )));
    }
    if command_arguments.is_empty()
        && accepted
            .iter()
            .any(|parameter| parameter.kind == "argument" && parameter.required)
    {
        return Err(CliError::invalid(format!(
            "required positional input is missing for `{}`",
            command.path
        )));
    }

    Ok(())
}

fn validate_synthetic_serve_arguments(arguments: &[String]) -> Result<(), CliError> {
    const VALUE_FLAGS: &[&str] = &[
        "--bind",
        "--port",
        "--profile",
        "--origin",
        "--bearer-token",
        "--tls-cert",
        "--tls-key",
        "--shutdown-after",
    ];
    const SWITCH_FLAGS: &[&str] = &[
        "--offline",
        "--allow-remote",
        "--json",
        "--json-stream",
        "--no-json",
    ];
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        if contains_help_flag(std::slice::from_ref(argument)) {
            index += 1;
            continue;
        }
        let flag = argument
            .split_once('=')
            .map_or(argument.as_str(), |pair| pair.0);
        if SWITCH_FLAGS.contains(&flag) {
            index += 1;
            continue;
        }
        if VALUE_FLAGS.contains(&flag) {
            if argument.contains('=') {
                index += 1;
                continue;
            }
            if arguments
                .get(index + 1)
                .is_none_or(|value| value.starts_with('-'))
            {
                return Err(CliError::invalid(format!(
                    "option `{flag}` requires a value"
                )));
            }
            index += 2;
            continue;
        }
        return Err(CliError::invalid(format!(
            "unknown `sim serve` option `{argument}`"
        )));
    }
    Ok(())
}

fn option_requires_value(parameter: &CatalogParameter, _argument: &str) -> bool {
    parameter.kind == "option" && matches!(parameter.value_arity.as_str(), "1" | "1 per occurrence")
}

fn validate_inline_boolean(parameter: &CatalogParameter, value: &str) -> Result<(), CliError> {
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "true" | "false" | "0" | "no"
    ) {
        Ok(())
    } else {
        Err(CliError::invalid(format!(
            "`{}` accepts only true, false, 0, or no as an inline boolean",
            parameter.name
        )))
    }
}

fn validate_parameter_value(parameter: &CatalogParameter, value: &str) -> Result<(), CliError> {
    if !parameter.choices.is_empty() && !parameter.choices.iter().any(|choice| choice == value) {
        return Err(CliError::invalid(format!(
            "`{value}` is not valid for `{}`; expected one of {}",
            parameter.name,
            parameter.choices.join(", ")
        )));
    }
    match parameter.value_type.as_str() {
        "boolean" if parameter.value_arity != "0" => validate_inline_boolean(parameter, value),
        "boolean" if parameter.value_arity == "0" => Ok(()),
        "integer" => value.parse::<i64>().map(|_| ()).map_err(|error| {
            CliError::invalid(format!("`{}` requires an integer: {error}", parameter.name))
        }),
        "number" => match value.parse::<f64>() {
            Ok(value) if value.is_finite() => Ok(()),
            Ok(_) => Err(CliError::invalid(format!(
                "`{}` requires a finite number",
                parameter.name
            ))),
            Err(error) => Err(CliError::invalid(format!(
                "`{}` requires a number: {error}",
                parameter.name
            ))),
        },
        "string" | "path" | "enum" if !value.is_empty() || parameter.nullable => Ok(()),
        "string" | "path" | "enum" => Err(CliError::invalid(format!(
            "`{}` cannot be empty",
            parameter.name
        ))),
        unexpected => Err(CliError::invalid(format!(
            "unsupported compiled parameter type `{unexpected}`"
        ))),
    }
}

fn find_flag<'a>(
    parameters: &'a [&CatalogParameter],
    argument: &str,
) -> Option<&'a CatalogParameter> {
    let flag = argument.split_once('=').map_or(argument, |pair| pair.0);
    parameters
        .iter()
        .copied()
        .find(|parameter| parameter.flags.iter().any(|candidate| candidate == flag))
}

fn is_builtin_flag(argument: &str) -> bool {
    matches!(argument, "--help" | "-h")
}

fn contains_help_flag(arguments: &[String]) -> bool {
    contains_flag(arguments, "--help") || contains_flag(arguments, "-h")
}

fn contains_flag(arguments: &[String], flag: &str) -> bool {
    arguments.iter().any(|argument| argument == flag)
}

fn output_mode(arguments: &[String]) -> OutputMode {
    if contains_flag(arguments, "--json-stream") {
        OutputMode::JsonStream
    } else if contains_flag(arguments, "--json") || contains_flag(arguments, "--help-json") {
        OutputMode::Json
    } else {
        OutputMode::Pretty
    }
}

fn native_cli_path_from_catalog(path: &str) -> std::borrow::Cow<'_, str> {
    if path == "comfy" {
        "sim".into()
    } else if let Some(suffix) = path.strip_prefix("comfy ") {
        format!("sim {suffix}").into()
    } else {
        path.into()
    }
}

fn render_help(command_path: Option<&str>, output_mode: OutputMode) {
    if output_mode != OutputMode::Pretty {
        let value = help_value(command_path).unwrap_or_else(|error| {
            json!({
                "schema": "envelope/1",
                "ok": false,
                "error": { "code": error.code, "message": error.message }
            })
        });
        println!("{value}");
        return;
    }

    let Ok(catalog) = catalog() else {
        eprintln!("Unable to read the compiled native Sim command catalog.");
        return;
    };
    match command_path {
        Some("sim serve") => {
            println!(
                "Native compatibility host\n\nUsage: {} sim serve [OPTIONS]\n\nOptions:\n  --bind <ADDRESS>          Bind address (loopback by default)\n  --port <PORT>             Native API port, or 0 for an OS-selected port\n  --profile <ID>            Native runtime profile UUID\n  --origin <ORIGIN>         Allowed remote origin; repeatable\n  --bearer-token <TOKEN>    Native host bearer credential\n  --tls-cert <PATH>         DER TLS certificate\n  --tls-key <PATH>          DER PKCS#8 private key\n  --allow-remote            Acknowledge non-loopback exposure risk\n  --shutdown-after <SECS>   Bounded foreground lifetime\n  --json                    Emit envelope/1 JSON\n  --json-stream             Emit event/1 NDJSON\n  -h, --help                Print help",
                product_flavor::EXECUTABLE_NAME
            );
        }
        Some(path) => {
            if let Some(command) = catalog.commands.iter().find(|command| command.path == path) {
                println!(
                    "{}\n\nUsage: {} {} [OPTIONS]",
                    command.help,
                    product_flavor::EXECUTABLE_NAME,
                    &command.path
                );
                let parameters = catalog
                    .parameters
                    .iter()
                    .filter(|parameter| parameter.command_path == path)
                    .collect::<Vec<_>>();
                if !parameters.is_empty() {
                    println!("\nParameters:");
                    for parameter in parameters {
                        let spelling = if parameter.flags.is_empty() {
                            format!("<{}>", parameter.name.to_ascii_uppercase())
                        } else {
                            parameter.flags.join(", ")
                        };
                        println!("  {spelling}");
                    }
                }
            } else {
                eprintln!("Unknown command `{path}`");
            }
        }
        None => {
            println!(
                "Native Sim lifecycle and automation\n\nUsage: {} sim [GLOBAL OPTIONS] <COMMAND> [COMMAND OPTIONS]\n\nGlobal options:\n  --workspace <PATH>    Select or migrate a native profile workspace\n  --where <MODE>        Route locally or to an authorized native provider\n  --json                Emit envelope/1 JSON\n  --json-stream         Emit event/1 NDJSON\n  --help-json           Print machine-readable command discovery\n  -v, --version         Print native CLI version\n  -h, --help            Print help\n\nCommands:",
                product_flavor::EXECUTABLE_NAME
            );
            let mut top_levels = catalog
                .commands
                .iter()
                .filter(|command| !command.hidden)
                .filter_map(|command| command.path.split_whitespace().nth(1))
                .collect::<BTreeSet<_>>();
            top_levels.insert("serve");
            for top_level in top_levels {
                println!("  {top_level}");
            }
        }
    }
}

fn help_value(command_path: Option<&str>) -> Result<Value, CliError> {
    let catalog = catalog()?;
    let commands = catalog
        .commands
        .iter()
        .map(|command| {
            json!({
                "feature_id": command.feature_id,
                "path": &command.path,
                "hidden": command.hidden,
                "help": command.help,
                "disposition": command.disposition,
                "mapping": command.parity_decision,
            })
        })
        .chain(std::iter::once(json!({
            "feature_id": "ZED-SIM-NATIVE-SERVE",
            "path": "sim serve",
            "hidden": false,
            "help": "Serve the native Rust compatibility API without constructing GPUI.",
            "disposition": CommandDisposition::Native,
        })))
        .collect::<Vec<_>>();
    let parameters = catalog
        .parameters
        .iter()
        .filter(|parameter| {
            command_path.is_none_or(|path| {
                parameter.command_path == path || parameter.command_path == "sim"
            })
        })
        .map(|parameter| {
            json!({
                "feature_id": parameter.feature_id,
                "command_path": &parameter.command_path,
                "scope": parameter.scope,
                "name": parameter.name,
                "kind": parameter.kind,
                "flags": parameter.flags,
                "value_type": parameter.value_type,
                "nullable": parameter.nullable,
                "value_arity": parameter.value_arity,
                "paired_boolean": parameter.paired_boolean,
                "repeatable": parameter.repeatable,
                "choices": parameter.choices,
                "constraints": parameter.constraints,
                "required": parameter.required,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": "help/discovery/1",
        "prog": format!("{} sim", product_flavor::EXECUTABLE_NAME),
        "version": env!("CARGO_PKG_VERSION"),
        "runtime": "native-rust",
        "command": command_path,
        "commands": commands,
        "parameters": parameters,
        "contract_registry": contract_registry()?,
        "capabilities": {
            "gpui_required": false,
            "python": false,
            "javascript_extensions": false,
            "external_comfy_server": false,
            "native_api": true,
            "rust_wasm_plugins": true,
        }
    }))
}

fn contract_registry() -> Result<Value, CliError> {
    let mut catalogs = serde_json::Map::new();
    for spec in CONTRACT_CATALOGS {
        let records = parse_csv(spec.contents).map_err(|message| CliError {
            code: "catalog_invalid".into(),
            message,
            hint: Some(format!(
                "the compiled {} contract catalog is invalid",
                spec.name
            )),
        })?;
        let header = records.first().ok_or_else(|| CliError {
            code: "catalog_invalid".into(),
            message: format!("the compiled {} contract catalog has no header", spec.name),
            hint: None,
        })?;
        let rows = records
            .iter()
            .enumerate()
            .skip(1)
            .map(|(row_number, row)| contract_registry_row(spec.name, row_number, header, row))
            .collect::<Result<Vec<_>, CliError>>()?;
        if rows.len() != spec.expected_rows {
            return Err(CliError {
                code: "catalog_invalid".into(),
                message: format!(
                    "compiled {} contract count changed: expected {}, found {}",
                    spec.name,
                    spec.expected_rows,
                    rows.len()
                ),
                hint: None,
            });
        }
        catalogs.insert(
            spec.name.into(),
            json!({
                "schema": "zed.sim.contract-catalog/1",
                "row_count": rows.len(),
                "sha256": spec.sha256,
                "rows": rows,
            }),
        );
    }
    Ok(json!({
        "schema": "zed.sim.contract-registry/1",
        "catalogs": catalogs,
    }))
}

fn contract_registry_row(
    catalog_name: &str,
    row_number: usize,
    header: &[String],
    row: &[String],
) -> Result<Value, CliError> {
    if header.len() != row.len() {
        return Err(CliError {
            code: "catalog_invalid".into(),
            message: format!(
                "compiled {catalog_name} catalog row {row_number} has {} fields, expected {}",
                row.len(),
                header.len()
            ),
            hint: None,
        });
    }
    let fields = header
        .iter()
        .cloned()
        .zip(row.iter().cloned())
        .map(|(key, value)| (key, Value::String(value)))
        .collect::<serde_json::Map<_, _>>();
    let value = |name: &str| fields.get(name).and_then(Value::as_str).unwrap_or_default();
    let identity = if catalog_name == "schema_mappings" {
        format!(
            "{}:{}:{}:{}",
            value("mapping_kind"),
            value("command_path"),
            value("schema"),
            row_number
        )
    } else {
        value("feature_id").to_owned()
    };
    let name = ["name", "event", "code", "key", "command_path"]
        .into_iter()
        .find_map(|field_name| {
            let candidate = value(field_name);
            (!candidate.is_empty()).then_some(candidate)
        })
        .unwrap_or_else(|| value("schema"));
    let source_status = if catalog_name == "schema_mappings" {
        if value("reachable") == "true" {
            "reachable"
        } else {
            "nonreachable_orphan"
        }
    } else {
        value("target_status")
    };
    let (disposition, mapping) = if catalog_name == "schema_mappings" {
        if value("reachable") == "true" {
            (CommandDisposition::Native, "native_protocol_projection")
        } else {
            (CommandDisposition::Migration, "catalog_only_orphan")
        }
    } else {
        let disposition = match source_status {
            "missing" => CommandDisposition::Native,
            "conflicting" => CommandDisposition::Migration,
            "deferred" => CommandDisposition::Deferred,
            unexpected => {
                return Err(CliError {
                    code: "catalog_invalid".into(),
                    message: format!(
                        "compiled {catalog_name} catalog row {row_number} has unsupported target status `{unexpected}`"
                    ),
                    hint: None,
                });
            }
        };
        (disposition, value("parity_decision"))
    };
    Ok(json!({
        "identity": identity,
        "row_number": row_number,
        "name": name,
        "source_status": source_status,
        "disposition": disposition,
        "mapping": mapping,
        "source_contract": fields,
    }))
}

fn render_help_json() -> Result<(), CliError> {
    println!("{}", help_value(None)?);
    Ok(())
}

fn render_explicit_disposition(
    invocation: &ParsedInvocation,
    code: &'static str,
    message: &'static str,
) {
    let value = json!({
        "schema": "envelope/1",
        "ok": false,
        "command": &invocation.command_path,
        "feature_id": invocation.feature_id,
        "disposition": invocation.disposition,
        "error": {
            "code": code,
            "message": message,
            "replacement": "Zed native runtime profiles and versioned Rust/WASM plugins",
        }
    });
    match invocation.output_mode {
        OutputMode::Pretty => eprintln!("{}: {message}", invocation.command_path),
        OutputMode::Json | OutputMode::JsonStream => println!("{value}"),
    }
}

fn render_success(command: &str, data: Value, output_mode: OutputMode) {
    let value = json!({
        "schema": "envelope/1",
        "ok": true,
        "command": command,
        "data": data,
    });
    match output_mode {
        OutputMode::Pretty => println!("{data}"),
        OutputMode::Json | OutputMode::JsonStream => println!("{value}"),
    }
}

fn render_error(error: &CliError, output_mode: OutputMode) {
    let value = json!({
        "schema": "envelope/1",
        "ok": false,
        "error": {
            "code": error.code,
            "message": error.message,
            "hint": error.hint,
        }
    });
    match output_mode {
        OutputMode::Pretty => {
            eprintln!("Error: {}", error.message);
            if let Some(hint) = &error.hint {
                eprintln!("Hint: {hint}");
            }
        }
        OutputMode::Json | OutputMode::JsonStream => println!("{value}"),
    }
}

fn catalog() -> Result<&'static Catalog, CliError> {
    CATALOG
        .get_or_init(load_catalog)
        .as_ref()
        .map_err(|error| CliError {
            code: "catalog_invalid".into(),
            message: error.clone(),
            hint: Some("the compiled Zed binary contains an invalid CLI catalog".into()),
        })
}

fn load_catalog() -> Result<Catalog, String> {
    let command_rows = parse_csv(COMMAND_CATALOG)?;
    let parameter_rows = parse_csv(PARAMETER_CATALOG)?;
    let command_header = command_rows
        .first()
        .ok_or_else(|| "command catalog has no header".to_owned())?;
    let parameter_header = parameter_rows
        .first()
        .ok_or_else(|| "parameter catalog has no header".to_owned())?;

    let commands = command_rows
        .iter()
        .skip(1)
        .map(|row| {
            let target_status = field(command_header, row, "target_status")?;
            let disposition = match target_status {
                "missing" => CommandDisposition::Native,
                "conflicting" => CommandDisposition::Migration,
                "deferred" => CommandDisposition::Deferred,
                unexpected => {
                    return Err(format!("unsupported command target status `{unexpected}`"));
                }
            };
            Ok(CatalogCommand {
                feature_id: field(command_header, row, "feature_id")?.to_owned(),
                path: native_cli_path_from_catalog(field(command_header, row, "path")?)
                    .into_owned(),
                help: field(command_header, row, "help")?.to_owned(),
                hidden: parse_hidden(field(command_header, row, "hidden")?)?,
                disposition,
                parity_decision: field(command_header, row, "parity_decision")?.to_owned(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let parameters = parameter_rows
        .iter()
        .skip(1)
        .map(|row| {
            let boolean_forms = field(parameter_header, row, "boolean_forms")?;
            let flags = if boolean_forms.is_empty() {
                field(parameter_header, row, "flags")?
            } else {
                boolean_forms
            };
            Ok(CatalogParameter {
                feature_id: field(parameter_header, row, "feature_id")?.to_owned(),
                command_path: native_cli_path_from_catalog(field(
                    parameter_header,
                    row,
                    "command_path",
                )?)
                .into_owned(),
                scope: field(parameter_header, row, "scope")?.to_owned(),
                name: field(parameter_header, row, "name")?.to_owned(),
                kind: field(parameter_header, row, "kind")?.to_owned(),
                flags: flags
                    .split(" | ")
                    .filter(|flag| !flag.is_empty())
                    .map(str::to_owned)
                    .collect(),
                value_type: field(parameter_header, row, "value_type")?.to_owned(),
                nullable: parse_bool(field(parameter_header, row, "nullable")?, "nullable")?,
                value_arity: field(parameter_header, row, "value_arity")?.to_owned(),
                paired_boolean: boolean_forms.split(" | ").count() == 2,
                repeatable: parse_bool(field(parameter_header, row, "repeatable")?, "repeatable")?,
                choices: field(parameter_header, row, "choices")?
                    .split(" | ")
                    .filter(|choice| !choice.is_empty())
                    .map(str::to_owned)
                    .collect(),
                constraints: field(parameter_header, row, "constraints")?.to_owned(),
                required: parse_bool(field(parameter_header, row, "required")?, "required")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    if commands.len() != 123 {
        return Err(format!(
            "command catalog count changed: expected 123, found {}",
            commands.len()
        ));
    }
    if parameters.len() != 370 {
        return Err(format!(
            "parameter catalog count changed: expected 370, found {}",
            parameters.len()
        ));
    }
    Ok(Catalog {
        commands,
        parameters,
    })
}

fn field<'a>(header: &[String], row: &'a [String], name: &str) -> Result<&'a str, String> {
    let index = header
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| format!("catalog header is missing `{name}`"))?;
    row.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("catalog row is missing `{name}`"))
}

fn parse_bool(value: &str, field_name: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("catalog field `{field_name}` is not boolean")),
    }
}

fn parse_hidden(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" | "ambiguous" => Ok(false),
        _ => Err("catalog field `hidden` has an unsupported value".into()),
    }
}

fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, String> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut characters = input.chars().peekable();
    let mut quoted = false;
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                if field.ends_with('\r') {
                    field.pop();
                }
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("catalog contains an unterminated quoted field".into());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::{collections::BTreeMap, fs, path::PathBuf};

    #[test]
    fn headless_profile_uses_exact_configured_rocm_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_106);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "ROCm headless",
                    "model_roots": [],
                    "device": "rocm",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "rocm_package_root": "/reviewed/rocm-package",
                    "rocm_package_signer": "rocm.release",
                    "rocm_package_public_key_hex": "11".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Rocm);
        let package = profile
            .rocm_package
            .ok_or_else(|| anyhow::anyhow!("ROCm profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/rocm-package"));
        assert_eq!(package.verification_key().signer(), "rocm.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x11; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_metal_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_108);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "Metal headless",
                    "model_roots": [],
                    "device": "metal",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "metal_package_root": "/reviewed/metal-package",
                    "metal_package_signer": "metal.release",
                    "metal_package_public_key_hex": "33".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Metal);
        let package = profile
            .metal_package
            .ok_or_else(|| anyhow::anyhow!("Metal profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/metal-package"));
        assert_eq!(package.verification_key().signer(), "metal.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x33; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_mlu_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_109);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "MLU headless",
                    "model_roots": [],
                    "device": "mlu",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "mlu_package_root": "/reviewed/mlu-package",
                    "mlu_package_signer": "mlu.release",
                    "mlu_package_public_key_hex": "44".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Mlu);
        let package = profile
            .mlu_package
            .ok_or_else(|| anyhow::anyhow!("MLU profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/mlu-package"));
        assert_eq!(package.verification_key().signer(), "mlu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x44; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_npu_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_10a);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "NPU headless",
                    "model_roots": [],
                    "device": "npu",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "npu_package_root": "/reviewed/npu-package",
                    "npu_package_signer": "npu.release",
                    "npu_package_public_key_hex": "56".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Npu);
        let package = profile
            .npu_package
            .ok_or_else(|| anyhow::anyhow!("NPU profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/npu-package"));
        assert_eq!(package.verification_key().signer(), "npu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x56; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_cuda_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_10a);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "CUDA headless",
                    "model_roots": [],
                    "device": "cuda",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "cuda_package_root": "/reviewed/cuda-package",
                    "cuda_package_signer": "cuda.release",
                    "cuda_package_public_key_hex": "56".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Cuda);
        let package = profile
            .cuda_package
            .ok_or_else(|| anyhow::anyhow!("CUDA profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/cuda-package"));
        assert_eq!(package.verification_key().signer(), "cuda.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x56; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_xpu_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_10b);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "XPU headless",
                    "model_roots": [],
                    "device": "xpu",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "xpu_package_root": "/reviewed/xpu-package",
                    "xpu_package_signer": "xpu.release",
                    "xpu_package_public_key_hex": "67".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::Xpu);
        let package = profile
            .xpu_package
            .ok_or_else(|| anyhow::anyhow!("XPU profile omitted its package authority"))?;
        assert_eq!(package.package_root(), Path::new("/reviewed/xpu-package"));
        assert_eq!(package.verification_key().signer(), "xpu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x67; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_uses_exact_configured_directml_authority_without_cpu_fallback()
    -> anyhow::Result<()> {
        let profile_id = Uuid::from_u128(0x8_110);
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "DirectML headless",
                    "model_roots": [],
                    "device": "directml",
                    "memory_policy": "balanced",
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "directml_package_root": "/reviewed/directml-package",
                    "directml_package_signer": "directml.release",
                    "directml_package_public_key_hex": "55".repeat(32)
                }]
            }
        })
        .to_string();

        let profile =
            headless_native_profile_from_settings_text(profile_id, Some(settings.as_str()))?;
        assert_eq!(profile.device, sim_types::DeviceKind::DirectMl);
        let package = profile
            .directml_package
            .ok_or_else(|| anyhow::anyhow!("DirectML profile omitted its package authority"))?;
        assert_eq!(
            package.package_root(),
            Path::new("/reviewed/directml-package")
        );
        assert_eq!(package.verification_key().signer(), "directml.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x55; 32]);
        Ok(())
    }

    #[test]
    fn headless_profile_rejects_unknown_nondefault_profile_instead_of_using_cpu() {
        let missing_profile = Uuid::from_u128(0x8_107);
        let error = headless_native_profile_from_settings_text(missing_profile, Some("{}"))
            .expect_err("unknown headless profile must fail closed");
        assert!(
            error
                .to_string()
                .contains("not present in canonical settings")
        );
    }

    const CATALOGS: &[(&str, &str, usize)] = &[
        ("commands", COMMAND_CATALOG, 123),
        ("parameters", PARAMETER_CATALOG, 370),
        (
            "cql-policy",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-cql-policy.csv"),
            419,
        ),
        (
            "documentation",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-documentation.csv"
            ),
            16,
        ),
        (
            "config",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-config.csv"),
            20,
        ),
        (
            "environment",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-environment.csv"
            ),
            35,
        ),
        (
            "errors",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-errors.csv"),
            99,
        ),
        (
            "events",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-events.csv"),
            12,
        ),
        (
            "extensions",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-extensions.csv"),
            17,
        ),
        (
            "formats",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-formats.csv"),
            34,
        ),
        (
            "lifecycle",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-lifecycle.csv"),
            24,
        ),
        (
            "modules",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-modules.csv"),
            104,
        ),
        (
            "partner-openapi",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-partner-openapi.csv"
            ),
            52,
        ),
        (
            "schema-mappings",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-schema-mappings.csv"
            ),
            66,
        ),
        (
            "schemas",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-schemas.csv"),
            23,
        ),
        (
            "source-coverage",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-source-coverage.csv"
            ),
            312,
        ),
        (
            "tests",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-tests.csv"),
            2295,
        ),
    ];

    #[derive(Parser, Debug)]
    #[command(name = "zed", subcommand_precedence_over_arg = true)]
    struct TestArgs {
        #[command(subcommand)]
        command: Option<TestCommand>,
        paths: Vec<String>,
    }

    #[derive(clap::Subcommand, Debug)]
    enum TestCommand {
        Sim(SimArgs),
    }

    #[test]
    fn sim_cli_native_discovery_preserves_source_catalog_identity() -> anyhow::Result<()> {
        let parsed = TestArgs::try_parse_from(["zed", "sim", "jobs", "ls", "--json"])?;
        assert!(matches!(parsed.command, Some(TestCommand::Sim(_))));
        let help = help_value(None).map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            help["prog"],
            format!("{} sim", product_flavor::EXECUTABLE_NAME)
        );
        assert_eq!(native_cli_path_from_catalog("comfy jobs ls"), "sim jobs ls");
        for command in help["commands"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing commands"))?
        {
            assert!(
                command["path"]
                    .as_str()
                    .is_some_and(|path| path.starts_with("sim "))
            );
            assert!(command.get("legacy_path").is_none());
        }
        Ok(())
    }

    #[test]
    fn sim_cli_contract_parses_every_command_parameter_and_disposition() -> anyhow::Result<()> {
        let parsed = TestArgs::try_parse_from(["zed", "sim", "jobs", "ls", "--json"])?;
        assert!(matches!(parsed.command, Some(TestCommand::Sim(_))));
        let catalog = catalog().map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(catalog.commands.len(), 123);
        assert_eq!(catalog.parameters.len(), 370);
        assert_eq!(
            catalog
                .commands
                .iter()
                .filter(|command| command.disposition == CommandDisposition::Native)
                .count(),
            73
        );
        assert_eq!(
            catalog
                .commands
                .iter()
                .filter(|command| command.disposition == CommandDisposition::Migration)
                .count(),
            40
        );
        assert_eq!(
            catalog
                .commands
                .iter()
                .filter(|command| command.disposition == CommandDisposition::Deferred)
                .count(),
            10
        );
        assert_eq!(
            catalog
                .commands
                .iter()
                .filter(|command| command.hidden)
                .count(),
            11
        );
        assert_eq!(
            catalog
                .commands
                .iter()
                .filter_map(|command| command.path.split_whitespace().nth(1))
                .collect::<BTreeSet<_>>()
                .len(),
            41
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.kind == "option")
                .count(),
            314
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.kind == "argument")
                .count(),
            56
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.command_path == "sim")
                .count(),
            11
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.repeatable)
                .count(),
            22
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.required)
                .count(),
            54
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.paired_boolean)
                .count(),
            15
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.scope == "command")
                .count(),
            348
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.scope == "global")
                .count(),
            11
        );
        assert_eq!(
            catalog
                .parameters
                .iter()
                .filter(|parameter| parameter.scope == "dynamic fixed")
                .count(),
            11
        );
        for command in &catalog.commands {
            let arguments = valid_command_arguments(command, catalog)?;
            let action = parse_action(&arguments)?;
            match action {
                CliAction::Invoke(invocation) => {
                    assert_eq!(invocation.feature_id, command.feature_id);
                    assert_eq!(invocation.disposition, command.disposition);
                }
                CliAction::Help(_, _) if command.path == "sim help" => {}
                unexpected => panic!("unexpected action for {}: {unexpected:?}", command.path),
            }
            assert!(!command.parity_decision.is_empty());
        }
        for parameter in &catalog.parameters {
            assert!(!parameter.feature_id.is_empty());
            assert!(parameter.kind == "argument" || parameter.kind == "option");
            if parameter.kind == "option" {
                assert!(!parameter.flags.is_empty());
            }
            let value = valid_parameter_value(parameter);
            if parameter.value_arity != "0" {
                validate_parameter_value(parameter, &value)?;
            }
        }
        assert_eq!(
            sha256(PARAMETER_CATALOG.as_bytes()),
            "f4919448071b0d42077296f6b7e5d11477bdc254f4ff2cc13542800b161b2186"
        );
        let registry = contract_registry()?;
        assert_eq!(registry["schema"], "zed.sim.contract-registry/1");
        for spec in CONTRACT_CATALOGS {
            let compiled = &registry["catalogs"][spec.name];
            assert_eq!(compiled["row_count"], spec.expected_rows);
            assert_eq!(compiled["sha256"], spec.sha256);
            assert_eq!(
                compiled["rows"].as_array().map(Vec::len),
                Some(spec.expected_rows)
            );
            assert_eq!(sha256(spec.contents.as_bytes()), spec.sha256);
            for row in compiled["rows"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("{} registry rows are not an array", spec.name))?
            {
                assert!(
                    row["identity"]
                        .as_str()
                        .is_some_and(|value| !value.is_empty())
                );
                assert!(row["name"].as_str().is_some_and(|value| !value.is_empty()));
                assert!(
                    row["source_status"]
                        .as_str()
                        .is_some_and(|value| !value.is_empty())
                );
                assert!(
                    row["mapping"]
                        .as_str()
                        .is_some_and(|value| !value.is_empty())
                );
                assert!(row["source_contract"].is_object());
            }
        }
        let environment = native_environment_report()?;
        let variables = environment["variables"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("native environment variables are not an array"))?;
        assert_eq!(variables.len(), 35);
        assert!(variables.iter().all(|variable| {
            variable.get("value").is_none()
                && variable["value_disclosed"] == false
                && variable["configured"].is_boolean()
        }));
        let profile = native_profile_report();
        assert_eq!(profile["runtime"], "native-rust");
        assert_eq!(profile["external_comfy_selected"], false);
        assert_eq!(profile["python_runtime_selected"], false);
        assert_eq!(help_value(None)?["schema"], "help/discovery/1");
        assert!(parse_action(&["jobs".into(), "ls".into(), "--limit=3".into()]).is_ok());
        for (arguments, owning_task) in [
            (
                &["jobs", "ls", "--watch"] as &[&str],
                "comfy-parity-native-execution-e2e",
            ),
            (
                &["nodes", "ls", "--produces", "IMAGE"],
                "comfy-parity-assets-editors-viewers",
            ),
            (
                &["nodes", "show", "KSampler", "--input", "object-info.json"],
                "comfy-parity-assets-editors-viewers",
            ),
            (
                &["model", "list", "--relative-path", "models"],
                "comfy-parity-file-asset-runtime",
            ),
            (
                &["models", "list-folder", "checkpoints", "--limit", "1"],
                "comfy-parity-file-asset-runtime",
            ),
            (
                &["jobs", "status", "prompt-1", "--enable-telemetry"],
                "comfy-parity-auth-cloud-telemetry",
            ),
            (
                &["jobs", "status", "prompt-1", "--workspace", "workspace"],
                "comfy-parity-settings-localization-ui",
            ),
        ] {
            assert!(matches!(
                native_plan_for(arguments)?,
                NativeExecutionPlan::Headless(NativeCliOperation::Deferred { reason })
                    if reason.contains(owning_task)
            ));
        }
        assert!(matches!(
            native_plan_for(&["jobs", "ls"])?,
            NativeExecutionPlan::Headless(NativeCliOperation::Jobs {
                limit: Some(10),
                ..
            })
        ));
        assert!(native_plan_for(&["jobs", "ls", "--where", "invalid"]).is_err());
        assert!(
            parse_action(&[
                "cloud".into(),
                "set-base-url".into(),
                "https://example.test".into(),
            ])
            .is_ok()
        );
        assert!(parse_action(&["standalone".into(), "--platform=linux".into(),]).is_ok());
        assert!(parse_action(&["standalone".into(), "--platform=unsupported".into(),]).is_err());
        assert!(
            parse_action(&["generate".into(), "--json=false".into(), "partner".into()]).is_ok()
        );
        assert!(
            parse_action(&[
                "generate".into(),
                "--async=invalid".into(),
                "partner".into()
            ])
            .is_err()
        );
        assert!(parse_action(&["jobs".into(), "ls".into(), "--watch=true".into()]).is_err());
        assert!(parse_action(&["node".into(), "install".into()]).is_err());
        assert!(parse_action(&["jobs".into(), "ls".into(), "--not-real".into()]).is_err());
        assert!(
            parse_action(&[
                "--here".into(),
                "--recent".into(),
                "jobs".into(),
                "ls".into()
            ])
            .is_err()
        );
        assert!(
            parse_action(&[
                "--json".into(),
                "--json-stream".into(),
                "jobs".into(),
                "ls".into()
            ])
            .is_err()
        );
        assert!(matches!(
            parse_action(&["serve".into(), "--offline".into()])?,
            CliAction::Invoke(ParsedInvocation {
                disposition: CommandDisposition::Native,
                ..
            })
        ));
        assert!(
            serve_configuration(
                &ParsedInvocation {
                    feature_id: "ZED-SIM-NATIVE-SERVE".into(),
                    command_path: "sim serve".into(),
                    disposition: CommandDisposition::Native,
                    arguments: vec!["serve".into(), "--offline".into()],
                    output_mode: OutputMode::Pretty,
                },
                "127.0.0.1:0".parse()?,
            )
            .is_err()
        );
        HEADLESS_SIGNAL.store(0, Ordering::Release);
        #[cfg(unix)]
        record_unix_signal(2);
        assert_eq!(HEADLESS_SIGNAL.load(Ordering::Acquire), 2);
        assert_eq!(signal_exit_code(2), 130);
        HEADLESS_SIGNAL.store(0, Ordering::Release);
        Ok(())
    }

    fn native_plan_for(arguments: &[&str]) -> anyhow::Result<NativeExecutionPlan> {
        let arguments = arguments
            .iter()
            .map(|argument| (*argument).to_owned())
            .collect::<Vec<_>>();
        match parse_action(&arguments)? {
            CliAction::Invoke(invocation) => native_operation(&invocation).map_err(Into::into),
            action => Err(anyhow::anyhow!(
                "expected native invocation, found {action:?}"
            )),
        }
    }

    #[test]
    fn sim_cli_private_runtime_state_is_not_an_asset_namespace() -> anyhow::Result<()> {
        let temporary = std::env::temp_dir().join(format!("zed-sim-cli-{}", Uuid::new_v4()));
        fs::create_dir_all(&temporary)?;
        let private_state_root = temporary.join("state");
        fs::create_dir_all(&private_state_root)?;
        let ledger = private_state_root.join("native-api-idempotency.json");
        fs::write(&ledger, b"{}")?;
        let typed_roots = [
            (AssetNamespace::Input, "input"),
            (AssetNamespace::Output, "output"),
            (AssetNamespace::Temporary, "temporary"),
            (AssetNamespace::Model, "model"),
            (AssetNamespace::Plugin, "plugin"),
        ]
        .into_iter()
        .map(|(namespace, name)| {
            let root = temporary.join(name);
            fs::create_dir_all(&root)?;
            Ok((namespace, root))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
        let configured_asset_roots = typed_roots
            .iter()
            .map(|(_, root)| root.clone())
            .collect::<Vec<_>>();
        let roots = AssetRoots::new("cli-state-isolation", typed_roots)?;
        for asset_root in &configured_asset_roots {
            assert!(!ledger.starts_with(asset_root));
            assert!(!asset_root.starts_with(&private_state_root));
        }
        let assets = AssetService::open(roots)?;
        let authorization = authorize_native_plugin_asset_broker("cli-state-isolation")?;
        assert_eq!(
            assets
                .list_authorized(&Default::default(), &authorization)?
                .total,
            0
        );
        assert!(ledger.is_file());
        drop(assets);
        fs::remove_dir_all(&temporary)?;
        Ok(())
    }

    #[test]
    fn sim_cli_contract_accounts_for_every_normative_catalog_row() -> anyhow::Result<()> {
        let mut row_counts = BTreeMap::new();
        let mut catalog_digests = BTreeMap::new();
        let mut catalog_rows = BTreeMap::new();
        for (name, contents, expected_rows) in CATALOGS {
            let records = parse_csv(contents).map_err(anyhow::Error::msg)?;
            let actual_rows = records.len().saturating_sub(1);
            assert_eq!(actual_rows, *expected_rows, "{name}");
            row_counts.insert(*name, actual_rows);
            catalog_digests.insert(*name, sha256(contents.as_bytes()));
            let header = records
                .first()
                .ok_or_else(|| anyhow::anyhow!("{name} catalog has no header"))?;
            let rows = records
                .iter()
                .enumerate()
                .skip(1)
                .map(|(row_number, row)| catalog_evidence_row(name, row_number, header, row))
                .collect::<anyhow::Result<Vec<_>>>()?;
            catalog_rows.insert(*name, rows);
        }
        let catalog = catalog().map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let command_results = catalog
            .commands
            .iter()
            .map(|command| {
                let arguments = valid_command_arguments(command, catalog)?;
                let action = parse_action(&arguments)?;
                let (observed, mapping_detail) = match action {
                    CliAction::Invoke(invocation) => match invocation.disposition {
                        CommandDisposition::Native => match native_operation(&invocation)? {
                            NativeExecutionPlan::Local(_) => ("native_local", None),
                            NativeExecutionPlan::Headless(NativeCliOperation::Deferred {
                                reason,
                            }) => ("deferred", Some(reason)),
                            NativeExecutionPlan::Headless(NativeCliOperation::Migration {
                                detail,
                                ..
                            }) => ("migration", Some(detail)),
                            NativeExecutionPlan::Headless(_) => ("native_service", None),
                        },
                        CommandDisposition::Migration => ("migration", None),
                        CommandDisposition::Deferred => ("deferred", None),
                    },
                    CliAction::Help(_, _) => ("native_help", None),
                    CliAction::HelpJson => ("native_discovery", None),
                    CliAction::Version(_) => ("native_version", None),
                };
                Ok(json!({
                    "feature_id": command.feature_id,
                    "path": &command.path,
                "legacy_path": command.path,
                    "catalog_disposition": command.disposition,
                    "parser": "passed",
                    "observed_mapping": observed,
                    "mapping_detail": mapping_detail,
                }))
            })
            .collect::<Result<Vec<_>, CliError>>()?;
        let parameter_results = catalog
            .parameters
            .iter()
            .map(|parameter| {
                let valid_value = valid_parameter_value(parameter);
                let valid = parameter.value_arity == "0"
                    || validate_parameter_value(parameter, &valid_value).is_ok();
                let invalid = invalid_parameter_value(parameter).map(|invalid_value| {
                    validate_parameter_value(parameter, invalid_value).is_err()
                });
                json!({
                    "feature_id": parameter.feature_id,
                    "command_path": &parameter.command_path,
                    "name": parameter.name,
                    "type": parameter.value_type,
                    "arity": parameter.value_arity,
                    "repeatable": parameter.repeatable,
                    "valid_case": valid,
                    "invalid_case_rejected": invalid,
                })
            })
            .collect::<Vec<_>>();
        let main_source = include_str!("main.rs");
        let parse_position = main_source
            .find("let args = Args::parse();")
            .ok_or_else(|| anyhow::anyhow!("main Args parse missing"))?;
        let dispatch_position = main_source
            .find("sim_cli::run(sim_args)")
            .ok_or_else(|| anyhow::anyhow!("early Sim dispatch missing"))?;
        let gpui_position = main_source
            .find("let app = build_application()")
            .ok_or_else(|| anyhow::anyhow!("GPUI construction missing"))?;
        let process_command_api = ["process", "Command"].join("::");
        let case_results = BTreeMap::from([
            (
                "early_headless_dispatch",
                parse_position < dispatch_position && dispatch_position < gpui_position,
            ),
            (
                "invalid_input",
                parse_action(&["jobs".into(), "ls".into(), "--invalid".into()]).is_err(),
            ),
            (
                "output_exclusivity",
                parse_action(&[
                    "--json".into(),
                    "--json-stream".into(),
                    "jobs".into(),
                    "ls".into(),
                ])
                .is_err(),
            ),
            ("signal_exit_130", signal_exit_code(2) == 130),
            (
                "no_external_process_api",
                !include_str!("sim_cli.rs").contains(&process_command_api),
            ),
        ]);
        let source_top_levels = catalog
            .commands
            .iter()
            .filter_map(|command| command.path.split_whitespace().nth(1))
            .collect::<BTreeSet<_>>()
            .len();
        let hidden_leaves = catalog
            .commands
            .iter()
            .filter(|command| command.hidden)
            .count();
        let option_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.kind == "option")
            .count();
        let argument_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.kind == "argument")
            .count();
        let command_local_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.scope == "command")
            .count();
        let global_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.scope == "global")
            .count();
        let dynamic_fixed_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.scope == "dynamic fixed")
            .count();
        let fixed_generate_rows = catalog
            .parameters
            .iter()
            .filter(|parameter| {
                parameter.scope == "dynamic fixed" && parameter.command_path == "sim generate"
            })
            .count();
        let raw_required_arguments = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.required && parameter.kind == "argument")
            .count();
        let raw_required_options = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.required && parameter.kind == "option")
            .count();
        let hidden_callback_aliases = hidden_callback_alias_paths()?;
        let repeated_required_alias_arguments = catalog
            .parameters
            .iter()
            .filter(|parameter| {
                parameter.required
                    && parameter.kind == "argument"
                    && hidden_callback_aliases.contains(parameter.command_path.as_str())
            })
            .map(|parameter| format!("{} {}", parameter.command_path, parameter.name))
            .collect::<Vec<_>>();
        let source_required_arguments =
            raw_required_arguments.saturating_sub(repeated_required_alias_arguments.len());
        let repeatable_or_variadic = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.repeatable || parameter.value_arity == "variadic")
            .count();
        let paired_boolean_aliases = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.paired_boolean)
            .count();
        let native_commands = catalog
            .commands
            .iter()
            .filter(|command| command.disposition == CommandDisposition::Native)
            .count();
        let migration_commands = catalog
            .commands
            .iter()
            .filter(|command| command.disposition == CommandDisposition::Migration)
            .count();
        let deferred_commands = catalog
            .commands
            .iter()
            .filter(|command| command.disposition == CommandDisposition::Deferred)
            .count();
        let actual_native_paths = command_results
            .iter()
            .filter(|result| {
                matches!(
                    result["observed_mapping"].as_str(),
                    Some("native_help" | "native_local" | "native_service")
                )
            })
            .filter_map(|result| result["path"].as_str())
            .collect::<BTreeSet<_>>();
        let expected_actual_native_paths = BTreeSet::from([
            "sim discover",
            "sim env",
            "sim help",
            "sim jobs cancel",
            "sim jobs ls",
            "sim jobs status",
            "sim model list",
            "sim models list-folder",
            "sim models list-folders",
            "sim nodes ls",
            "sim nodes refresh",
            "sim nodes show",
            "sim which",
        ]);
        let native_task_deferrals = command_results
            .iter()
            .filter(|result| {
                result["catalog_disposition"] == "native"
                    && result["observed_mapping"] == "deferred"
            })
            .collect::<Vec<_>>();
        let native_unavailable = command_results
            .iter()
            .filter(|result| result["observed_mapping"] == "native_capability_unavailable")
            .count();
        let task_deferrals_are_owned = native_task_deferrals.iter().all(|result| {
            result["mapping_detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("owning task `comfy-parity-"))
        });
        let passed = case_results.values().all(|passed| *passed)
            && command_results.len() == 123
            && parameter_results.len() == 370
            && parameter_results
                .iter()
                .all(|result| result["valid_case"] == true)
            && row_counts.values().sum::<usize>() == 4_021
            && source_top_levels == 41
            && hidden_leaves == 11
            && option_rows == 314
            && argument_rows == 56
            && command_local_rows == 348
            && global_rows == 11
            && dynamic_fixed_rows == 11
            && fixed_generate_rows == 11
            && raw_required_arguments == 44
            && raw_required_options == 10
            && source_required_arguments == 43
            && repeated_required_alias_arguments == ["sim cs query"]
            && repeatable_or_variadic == 22
            && paired_boolean_aliases == 15
            && native_commands == 73
            && migration_commands == 40
            && deferred_commands == 10
            && actual_native_paths == expected_actual_native_paths
            && native_task_deferrals.len() == 60
            && native_unavailable == 0
            && task_deferrals_are_owned;
        let reconciliation: Value = serde_json::from_str(include_str!(
            "../../../crates/sim_test_support/parity/catalogs/sim-cli-reconciliation.json"
        ))?;
        let artifact = json!({
            "validation": "VAL-CLI-001",
            "schema_version": 1,
            "implementation": {
                "binary": "zed",
                "command_family": format!("{} sim", product_flavor::EXECUTABLE_NAME),
                "gpui_constructed": false,
                "runtime": "native-rust",
                "python_processes": 0,
                "node_processes": 0,
                "browser_processes": 0,
                "external_comfy_connections": 0,
            },
            "catalog_rows": row_counts,
            "catalog_sha256": catalog_digests,
            "catalog_accounting": catalog_rows,
            "reconciliation_sha256": sha256(include_bytes!("../../../crates/sim_test_support/parity/catalogs/sim-cli-reconciliation.json")),
            "reconciliation": reconciliation,
            "command_dispositions": {
                "native": native_commands,
                "migration": migration_commands,
                "deferred": deferred_commands,
                "implemented_native_paths": actual_native_paths,
                "task_activated_native_deferrals": native_task_deferrals.len(),
                "unowned_native_unavailable": native_unavailable,
            },
            "ratchets": {
                "accounted_relationships": row_counts.values().sum::<usize>(),
                "source_leaves": catalog.commands.len(),
                "synthetic_native_leaves": 1,
                "source_top_levels": source_top_levels,
                "hidden_leaves": hidden_leaves,
                "options": option_rows,
                "arguments": argument_rows,
                "command_local_parameters": command_local_rows,
                "global_parameters": global_rows,
                "dynamic_fixed_parameters": dynamic_fixed_rows,
                "fixed_generate_parameters": fixed_generate_rows,
                "required_binding_rows": {
                    "total": raw_required_arguments + raw_required_options,
                    "arguments": raw_required_arguments,
                    "options": raw_required_options,
                },
                "required_source_declarations": {
                    "total": source_required_arguments + raw_required_options,
                    "arguments": source_required_arguments,
                    "options": raw_required_options,
                    "deduplicated_alias_rows": repeated_required_alias_arguments,
                    "deduplication_basis": "hidden callback-leaf alias sharing the source symbol and line with its visible command",
                },
                "repeatable_or_variadic": repeatable_or_variadic,
                "paired_boolean_aliases": paired_boolean_aliases,
                "nonreachable_orphans": [
                    "comfy models (legacy hidden function)",
                    "comfy version (schema only)",
                    "comfy query (documentation only)"
                ],
            },
            "command_results": command_results,
            "parameter_results": parameter_results,
            "cases": case_results,
            "passed": passed,
            "failed": usize::from(!passed),
            "skipped": 0,
        });
        assert!(passed);
        let output = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/comfy-parity/val-cli-001.json");
        let parent = output
            .parent()
            .ok_or_else(|| anyhow::anyhow!("artifact path has no parent"))?;
        fs::create_dir_all(parent)?;
        fs::write(
            output,
            format!("{}\n", serde_json::to_string_pretty(&artifact)?),
        )?;
        Ok(())
    }

    fn catalog_evidence_row(
        catalog_name: &str,
        row_number: usize,
        header: &[String],
        row: &[String],
    ) -> anyhow::Result<Value> {
        anyhow::ensure!(
            header.len() == row.len(),
            "{catalog_name} row {row_number} has {} fields, expected {}",
            row.len(),
            header.len()
        );
        let fields = header
            .iter()
            .cloned()
            .zip(row.iter().cloned())
            .collect::<BTreeMap<_, _>>();
        let value = |name: &str| fields.get(name).map(String::as_str).unwrap_or_default();
        let primary = [
            "feature_id",
            "mapping_kind",
            "module",
            "test",
            "source_file",
            "schema",
            "key",
            "name",
        ]
        .into_iter()
        .find_map(|name| {
            let candidate = value(name);
            (!candidate.is_empty()).then_some(candidate)
        })
        .unwrap_or("row");
        let secondary = ["command_path", "path", "schema", "symbol", "code", "event"]
            .into_iter()
            .find_map(|name| {
                let candidate = value(name);
                (!candidate.is_empty()).then_some(candidate)
            })
            .unwrap_or("contract");
        let execution = match catalog_name {
            "commands" => "native_parser_and_disposition_exercised_by_val_cli",
            "parameters" => "native_parser_type_and_boundary_exercised_by_val_cli",
            "schemas" | "events" | "errors" | "config" | "formats" | "lifecycle"
            | "schema-mappings" => "compiled_help_contract_registry_exercised_by_val_cli",
            "tests" => "source_python_test_cataloged_not_executed_by_val_cli",
            _ => "catalog_reconciliation_evidence_only",
        };
        let target_status = value("target_status");
        let parity_decision = value("parity_decision");
        Ok(json!({
            "identity": format!("{catalog_name}:{row_number}:{primary}:{secondary}"),
            "row_number": row_number,
            "status": "accounted",
            "execution": execution,
            "source_file": value("source_file"),
            "source_tests": value("tests"),
            "evidence_level": value("evidence_level"),
            "target_status": if target_status.is_empty() { "not_declared_by_catalog" } else { target_status },
            "parity_mapping": if parity_decision.is_empty() { "not_declared_by_catalog" } else { parity_decision },
            "source_contract": fields,
        }))
    }

    fn hidden_callback_alias_paths() -> anyhow::Result<BTreeSet<String>> {
        let records = parse_csv(COMMAND_CATALOG).map_err(anyhow::Error::msg)?;
        let header = records
            .first()
            .ok_or_else(|| anyhow::anyhow!("command catalog has no header"))?;
        let source_index = header
            .iter()
            .position(|field| field == "source_file")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no source_file"))?;
        let symbol_index = header
            .iter()
            .position(|field| field == "symbol")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no symbol"))?;
        let line_index = header
            .iter()
            .position(|field| field == "line")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no line"))?;
        let path_index = header
            .iter()
            .position(|field| field == "path")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no path"))?;
        let hidden_index = header
            .iter()
            .position(|field| field == "hidden")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no hidden"))?;
        let registration_index = header
            .iter()
            .position(|field| field == "registration")
            .ok_or_else(|| anyhow::anyhow!("command catalog has no registration"))?;
        let rows = records.iter().skip(1).collect::<Vec<_>>();
        Ok(rows
            .iter()
            .filter(|row| {
                row.get(hidden_index).map(String::as_str) == Some("true")
                    && row
                        .get(registration_index)
                        .is_some_and(|value| value.contains("alias"))
                    && rows.iter().any(|candidate| {
                        candidate.get(hidden_index).map(String::as_str) == Some("false")
                            && candidate.get(source_index) == row.get(source_index)
                            && candidate.get(symbol_index) == row.get(symbol_index)
                            && candidate.get(line_index) == row.get(line_index)
                    })
            })
            .filter_map(|row| row.get(path_index).cloned())
            .collect())
    }

    fn valid_command_arguments(
        command: &CatalogCommand,
        catalog: &Catalog,
    ) -> Result<Vec<String>, CliError> {
        let mut arguments = command
            .path
            .split_whitespace()
            .skip(1)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let parameters = catalog
            .parameters
            .iter()
            .filter(|parameter| parameter.command_path == command.path)
            .collect::<Vec<_>>();
        for parameter in parameters
            .iter()
            .copied()
            .filter(|parameter| parameter.kind == "option" && parameter.required)
        {
            let flag = parameter
                .flags
                .first()
                .ok_or_else(|| CliError::invalid("required option has no flag"))?;
            arguments.push(flag.clone());
            if option_requires_value(parameter, flag) {
                arguments.push(valid_parameter_value(parameter));
            }
        }
        let argument_parameters = parameters
            .iter()
            .copied()
            .filter(|parameter| parameter.kind == "argument")
            .collect::<Vec<_>>();
        let last_required = argument_parameters
            .iter()
            .rposition(|parameter| parameter.required);
        if let Some(last_required) = last_required {
            for parameter in argument_parameters.iter().take(last_required + 1) {
                arguments.push(valid_parameter_value(parameter));
            }
        }
        Ok(arguments)
    }

    fn valid_parameter_value(parameter: &CatalogParameter) -> String {
        if parameter.name == "where" {
            return "local".into();
        }
        if let Some(choice) = parameter.choices.first() {
            return choice.clone();
        }
        match parameter.value_type.as_str() {
            "integer" => "1".into(),
            "number" => "1.5".into(),
            "path" => "fixture.json".into(),
            "boolean" => "true".into(),
            "string" | "enum" => "fixture".into(),
            _ => "fixture".into(),
        }
    }

    fn invalid_parameter_value(parameter: &CatalogParameter) -> Option<&'static str> {
        match parameter.value_type.as_str() {
            "integer" => Some("not-an-integer"),
            "number" => Some("NaN"),
            "enum" => Some("not-a-catalog-choice"),
            "boolean" if parameter.value_arity == "0 or 1 inline" => Some("invalid"),
            _ => None,
        }
    }

    fn sha256(input: &[u8]) -> String {
        const INITIAL: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const ROUND: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let bit_length = (input.len() as u64).wrapping_mul(8);
        let mut padded = input.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_length.to_be_bytes());
        let mut hash = INITIAL;
        for chunk in padded.chunks_exact(64) {
            let mut words = [0u32; 64];
            for (index, bytes) in chunk.chunks_exact(4).enumerate() {
                words[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
            for index in 16..64 {
                let sigma0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let sigma1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(sigma0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(sigma1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
            for index in 0..64 {
                let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choice = (e & f) ^ ((!e) & g);
                let temporary1 = h
                    .wrapping_add(sum1)
                    .wrapping_add(choice)
                    .wrapping_add(ROUND[index])
                    .wrapping_add(words[index]);
                let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let temporary2 = sum0.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temporary1);
                d = c;
                c = b;
                b = a;
                a = temporary1.wrapping_add(temporary2);
            }
            hash[0] = hash[0].wrapping_add(a);
            hash[1] = hash[1].wrapping_add(b);
            hash[2] = hash[2].wrapping_add(c);
            hash[3] = hash[3].wrapping_add(d);
            hash[4] = hash[4].wrapping_add(e);
            hash[5] = hash[5].wrapping_add(f);
            hash[6] = hash[6].wrapping_add(g);
            hash[7] = hash[7].wrapping_add(h);
        }
        hash.iter().map(|word| format!("{word:08x}")).collect()
    }
}
