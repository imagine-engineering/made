use std::{env::consts, path::PathBuf, sync::OnceLock};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::{
    io::{Read as _, Write as _},
    os::unix::fs::PermissionsExt as _,
};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use collections::HashMap;
use dap::adapters::{DebugTaskDefinition, latest_github_release};
use futures::StreamExt;
use gpui::AsyncApp;
use serde_json::Value;
use task::{DebugRequest, DebugScenario, ZedDebugConfig};
use util::fs::remove_matching;

use crate::*;

#[derive(Default)]
pub(crate) struct CodeLldbDebugAdapter {
    path_to_codelldb: OnceLock<String>,
}

impl CodeLldbDebugAdapter {
    const ADAPTER_NAME: &'static str = "CodeLLDB";

    async fn request_args(
        &self,
        delegate: &Arc<dyn DapDelegate>,
        mut configuration: Value,
        label: &str,
    ) -> Result<dap::StartDebuggingRequestArguments> {
        let obj = configuration
            .as_object_mut()
            .context("CodeLLDB is not a valid json object")?;

        // CodeLLDB uses `name` for a terminal label.
        obj.entry("name")
            .or_insert(Value::String(String::from(label)));

        obj.entry("cwd")
            .or_insert(delegate.worktree_root_path().to_string_lossy().into());

        let request = self.request_kind(&configuration).await?;

        Ok(dap::StartDebuggingRequestArguments {
            request,
            configuration,
        })
    }

    async fn fetch_latest_adapter_version(
        &self,
        delegate: &Arc<dyn DapDelegate>,
    ) -> Result<AdapterVersion> {
        let release =
            latest_github_release("vadimcn/codelldb", true, false, delegate.http_client()).await?;

        let arch = match std::env::consts::ARCH {
            "aarch64" => "arm64",
            "x86_64" => "x64",
            unsupported => {
                anyhow::bail!("unsupported architecture {unsupported}");
            }
        };
        let platform = match std::env::consts::OS {
            "macos" => "darwin",
            "linux" => "linux",
            "windows" => "win32",
            unsupported => {
                anyhow::bail!("unsupported operating system {unsupported}");
            }
        };
        let asset_name = format!("codelldb-{platform}-{arch}.vsix");
        let ret = AdapterVersion {
            tag_name: release.tag_name,
            url: release
                .assets
                .iter()
                .find(|asset| asset.name == asset_name)
                .with_context(|| format!("no asset found matching {asset_name:?}"))?
                .browser_download_url
                .clone(),
        };

        Ok(ret)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn is_native_arm64_program(configuration: &Value) -> bool {
    let Some(program) = configuration.get("program").and_then(Value::as_str) else {
        return false;
    };
    let path = std::path::Path::new(program);
    if !path.is_absolute() {
        return false;
    }
    let Ok(mut executable) = std::fs::File::open(path) else {
        return false;
    };
    let mut header = [0; 8];
    if executable.read_exact(&mut header).is_err() {
        return false;
    }
    header[..4] == [0xcf, 0xfa, 0xed, 0xfe]
        && u32::from_le_bytes([header[4], header[5], header[6], header[7]]) == 0x0100_000c
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn native_arm64_debugserver_launcher() -> Result<PathBuf> {
    const LAUNCHER: &str = r#"#!/bin/sh
developer_directory=$(/usr/bin/xcode-select -p) || exit 127
case "$developer_directory" in
  */Contents/Developer)
    debugserver="${developer_directory%/Contents/Developer}/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver"
    ;;
  *)
    debugserver="$developer_directory/Library/PrivateFrameworks/LLDB.framework/Versions/A/Resources/debugserver"
    ;;
esac
if [ ! -x "$debugserver" ]; then
  echo "CodeLLDB: Apple debugserver is unavailable at $debugserver" >&2
  exit 127
fi
exec /usr/bin/arch -arm64 "$debugserver" "$@"
"#;

    let directory = paths::debug_adapters_dir().join("CodeLLDB");
    std::fs::create_dir_all(&directory).context("creating CodeLLDB adapter directory")?;
    let mut launcher = tempfile::NamedTempFile::new_in(&directory)
        .context("creating native CodeLLDB debugserver launcher")?;
    launcher
        .write_all(LAUNCHER.as_bytes())
        .context("writing native CodeLLDB debugserver launcher")?;
    std::fs::set_permissions(launcher.path(), std::fs::Permissions::from_mode(0o700))
        .context("marking native CodeLLDB debugserver launcher executable")?;
    let path = directory.join("zed-arm64-debugserver");
    launcher
        .persist(&path)
        .context("installing native CodeLLDB debugserver launcher")?;
    Ok(path)
}

#[async_trait(?Send)]
impl DebugAdapter for CodeLldbDebugAdapter {
    fn name(&self) -> DebugAdapterName {
        DebugAdapterName(Self::ADAPTER_NAME.into())
    }

    async fn config_from_zed_format(&self, zed_scenario: ZedDebugConfig) -> Result<DebugScenario> {
        let mut configuration = json!({
            "request": match zed_scenario.request {
                DebugRequest::Launch(_) => "launch",
                DebugRequest::Attach(_) => "attach",
            },
        });
        let map = configuration.as_object_mut().unwrap();
        // CodeLLDB uses `name` for a terminal label.
        map.insert(
            "name".into(),
            Value::String(String::from(zed_scenario.label.as_ref())),
        );
        match &zed_scenario.request {
            DebugRequest::Attach(attach) => {
                map.insert("pid".into(), attach.process_id.into());
            }
            DebugRequest::Launch(launch) => {
                map.insert("program".into(), launch.program.clone().into());

                if !launch.args.is_empty() {
                    map.insert("args".into(), launch.args.clone().into());
                }
                if !launch.env.is_empty() {
                    map.insert("env".into(), launch.env_json());
                }
                if let Some(stop_on_entry) = zed_scenario.stop_on_entry {
                    map.insert("stopOnEntry".into(), stop_on_entry.into());
                }
                if let Some(cwd) = launch.cwd.as_ref() {
                    map.insert("cwd".into(), cwd.to_string_lossy().into_owned().into());
                }
            }
        }

        Ok(DebugScenario {
            adapter: zed_scenario.adapter,
            label: zed_scenario.label,
            config: configuration,
            build: None,
            tcp_connection: None,
        })
    }

    fn dap_schema(&self) -> serde_json::Value {
        json!({
            "properties": {
                "request": {
                    "type": "string",
                    "enum": ["attach", "launch"],
                    "description": "Debug adapter request type"
                },
                "program": {
                    "type": "string",
                    "description": "Path to the program to debug or attach to"
                },
                "args": {
                    "type": ["array", "string"],
                    "description": "Program arguments"
                },
                "cwd": {
                    "type": "string",
                    "description": "Program working directory"
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables",
                    "patternProperties": {
                        ".*": {
                            "type": "string"
                        }
                    }
                },
                "envFile": {
                    "type": "string",
                    "description": "File to read the environment variables from"
                },
                "stdio": {
                    "type": ["null", "string", "array", "object"],
                    "description": "Destination for stdio streams: null = send to debugger console or a terminal, \"<path>\" = attach to a file/tty/fifo"
                },
                "terminal": {
                    "type": "string",
                    "enum": ["integrated", "console"],
                    "description": "Terminal type to use",
                    "default": "integrated"
                },
                "console": {
                    "type": "string",
                    "enum": ["integratedTerminal", "internalConsole"],
                    "description": "Terminal type to use (compatibility alias of 'terminal')"
                },
                "stopOnEntry": {
                    "type": "boolean",
                    "description": "Automatically stop debuggee after launch",
                    "default": false
                },
                "initCommands": {
                    "type": "array",
                    "description": "Initialization commands executed upon debugger startup",
                    "items": {
                        "type": "string"
                    }
                },
                "targetCreateCommands": {
                    "type": "array",
                    "description": "Commands that create the debug target",
                    "items": {
                        "type": "string"
                    }
                },
                "preRunCommands": {
                    "type": "array",
                    "description": "Commands executed just before the program is launched",
                    "items": {
                        "type": "string"
                    }
                },
                "processCreateCommands": {
                    "type": "array",
                    "description": "Commands that create the debuggee process",
                    "items": {
                        "type": "string"
                    }
                },
                "postRunCommands": {
                    "type": "array",
                    "description": "Commands executed just after the program has been launched",
                    "items": {
                        "type": "string"
                    }
                },
                "preTerminateCommands": {
                    "type": "array",
                    "description": "Commands executed just before the debuggee is terminated or disconnected from",
                    "items": {
                        "type": "string"
                    }
                },
                "exitCommands": {
                    "type": "array",
                    "description": "Commands executed at the end of debugging session",
                    "items": {
                        "type": "string"
                    }
                },
                "expressions": {
                    "type": "string",
                    "enum": ["simple", "python", "native"],
                    "description": "The default evaluator type used for expressions"
                },
                "sourceMap": {
                    "type": "object",
                    "description": "Source path remapping between the build machine and the local machine",
                    "patternProperties": {
                        ".*": {
                            "type": ["string", "null"]
                        }
                    }
                },
                "relativePathBase": {
                    "type": "string",
                    "description": "Base directory used for resolution of relative source paths. Defaults to the workspace folder"
                },
                "sourceLanguages": {
                    "type": "array",
                    "description": "A list of source languages to enable language-specific features for",
                    "items": {
                        "type": "string"
                    }
                },
                "reverseDebugging": {
                    "type": "boolean",
                    "description": "Enable reverse debugging",
                    "default": false
                },
                "breakpointMode": {
                    "type": "string",
                    "enum": ["path", "file"],
                    "description": "Specifies how source breakpoints should be set"
                },
                "pid": {
                    "type": ["integer", "string"],
                    "description": "Process id to attach to"
                },
                "waitFor": {
                    "type": "boolean",
                    "description": "Wait for the process to launch (MacOS only)",
                    "default": false
                }
            },
            "required": ["request"],
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "request": {
                                "enum": ["launch"]
                            }
                        }
                    },
                    "then": {
                        "oneOf": [
                            {
                                "required": ["program"]
                            },
                            {
                                "required": ["targetCreateCommands"]
                            }
                        ]
                    }
                },
                {
                    "if": {
                        "properties": {
                            "request": {
                                "enum": ["attach"]
                            }
                        }
                    },
                    "then": {
                        "oneOf": [
                            {
                                "required": ["pid"]
                            },
                            {
                                "required": ["program"]
                            }
                        ]
                    }
                }
            ]
        })
    }

    fn diagnostic_installed_binary(
        &self,
        definition: &DebugTaskDefinition,
        root: &std::path::Path,
        installed_path: Option<&std::path::Path>,
        arguments: Option<&[String]>,
        environment: Option<&HashMap<String, String>>,
    ) -> Result<DebugAdapterBinary> {
        anyhow::ensure!(
            definition.tcp_connection.is_none()
                && arguments.is_none_or(|arguments| arguments.is_empty()),
            "Diagnostic CodeLLDB requires stdio without custom adapter startup arguments"
        );
        let command = installed_path
            .map(|path| path.to_string_lossy().into_owned())
            .or_else(|| self.path_to_codelldb.get().cloned())
            .context("Select an already installed CodeLLDB binary; Doctor will not download it")?;
        let mut configuration = definition.config.clone();
        let object = configuration
            .as_object_mut()
            .context("Debug configuration must be an object")?;
        anyhow::ensure!(
            object.keys().all(|key| matches!(
                key.as_str(),
                "request"
                    | "program"
                    | "cwd"
                    | "args"
                    | "env"
                    | "name"
                    | "stopOnEntry"
                    | "sourceLanguages"
                    | "terminal"
                    | "console"
            )),
            "Diagnostic CodeLLDB does not permit launch hooks, environment files, source remapping or other unreviewed configuration"
        );
        anyhow::ensure!(
            object.get("request").is_some_and(|value| value == "launch"),
            "Diagnostic debugging supports a new development process, not attachment"
        );
        anyhow::ensure!(
            object.get("program").and_then(Value::as_str).is_some(),
            "Select a concrete program"
        );
        object.entry("cwd").or_insert(root.to_string_lossy().into());
        object
            .entry("name")
            .or_insert(definition.label.to_string().into());
        anyhow::ensure!(
            object
                .get("terminal")
                .is_none_or(|value| value == "console")
                && object
                    .get("console")
                    .is_none_or(|value| value == "internalConsole"),
            "Diagnostic launches cannot create a terminal or collect terminal output"
        );
        object.insert("terminal".into(), "console".into());
        if let Some(value) = object.get("stopOnEntry") {
            anyhow::ensure!(value.is_boolean(), "stopOnEntry must be a boolean");
        }
        if let Some(arguments) = object.get("args") {
            anyhow::ensure!(
                arguments
                    .as_array()
                    .is_some_and(|arguments| arguments.len() <= 256
                        && arguments.iter().all(Value::is_string)),
                "Program arguments must be a bounded array of explicit strings"
            );
        }
        if let Some(environment) = object.get("env") {
            anyhow::ensure!(
                environment
                    .as_object()
                    .is_some_and(|environment| environment.len() <= 64
                        && environment.values().all(Value::is_string)),
                "Program environment must be an explicit bounded string map"
            );
        }
        let mut adapter_arguments = Vec::new();
        if let Some(languages) = object.remove("sourceLanguages") {
            anyhow::ensure!(
                languages.as_array().is_some_and(
                    |languages| languages.len() <= 8 && languages.iter().all(Value::is_string)
                ),
                "Source languages must be a bounded string array"
            );
            adapter_arguments = vec![
                "--settings".into(),
                json!({"sourceLanguages": languages}).to_string(),
            ];
        }
        Ok(DebugAdapterBinary {
            command: Some(command),
            arguments: adapter_arguments,
            envs: environment.cloned().unwrap_or_default(),
            cwd: Some(root.to_path_buf()),
            connection: None,
            request_args: dap::StartDebuggingRequestArguments {
                request: dap::StartDebuggingRequestArgumentsRequest::Launch,
                configuration,
            },
        })
    }

    async fn get_binary(
        &self,
        delegate: &Arc<dyn DapDelegate>,
        config: &DebugTaskDefinition,
        user_installed_path: Option<PathBuf>,
        user_args: Option<Vec<String>>,
        user_env: Option<HashMap<String, String>>,
        _: &mut AsyncApp,
    ) -> Result<DebugAdapterBinary> {
        let mut command = user_installed_path
            .map(|p| p.to_string_lossy().into_owned())
            .or(self.path_to_codelldb.get().cloned());

        if command.is_none() {
            delegate.output_to_console(format!("Checking latest version of {}...", self.name()));
            let adapter_path = paths::debug_adapters_dir().join(&Self::ADAPTER_NAME);
            let version_path = match self.fetch_latest_adapter_version(delegate).await {
                Ok(version) => {
                    adapters::download_adapter_from_github(
                        self.name(),
                        version.clone(),
                        adapters::DownloadedFileType::Vsix,
                        delegate.as_ref(),
                    )
                    .await?;
                    let version_path =
                        adapter_path.join(format!("{}_{}", Self::ADAPTER_NAME, version.tag_name));
                    remove_matching(&adapter_path, |entry| entry != version_path).await;
                    version_path
                }
                Err(e) => {
                    delegate.output_to_console("Unable to fetch latest version".to_string());
                    log::error!("Error fetching latest version of {}: {}", self.name(), e);
                    delegate.output_to_console(format!(
                        "Searching for adapters in: {}",
                        adapter_path.display()
                    ));
                    let mut paths = delegate
                        .fs()
                        .read_dir(&adapter_path)
                        .await
                        .context("No cached adapter directory")?;
                    paths
                        .next()
                        .await
                        .context("No cached adapter found")?
                        .context("No cached adapter found")?
                }
            };
            let adapter_dir = version_path.join("extension").join("adapter");
            let path = adapter_dir
                .join(format!("codelldb{}", consts::EXE_SUFFIX))
                .to_string_lossy()
                .into_owned();
            self.path_to_codelldb.set(path.clone()).ok();
            command = Some(path);
        };
        let mut json_config = config.config.clone();

        // Auto-detect Rust projects and add sourceLanguages if not present.
        // This enables panic breakpoints to work correctly with CodeLLDB.
        if let Some(config_obj) = json_config.as_object_mut() {
            if !config_obj.contains_key("sourceLanguages") {
                // Check if this looks like a Rust binary (Cargo build output)
                if let Some(program) = config_obj.get("program").and_then(|p| p.as_str()) {
                    let path_str = program.replace('\\', "/");
                    if path_str.contains("/target/debug/") || path_str.contains("/target/release/")
                    {
                        config_obj.insert("sourceLanguages".to_owned(), json!(["rust"]));
                    }
                }
            }
        }

        let envs = user_env.unwrap_or_default();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let envs = {
            let mut envs = envs;
            if !envs.contains_key("LLDB_DEBUGSERVER_PATH")
                && std::env::var_os("LLDB_DEBUGSERVER_PATH").is_none()
                && is_native_arm64_program(&json_config)
            {
                envs.insert(
                    "LLDB_DEBUGSERVER_PATH".into(),
                    native_arm64_debugserver_launcher()?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            envs
        };

        Ok(DebugAdapterBinary {
            command: Some(command.unwrap()),
            cwd: Some(delegate.worktree_root_path().to_path_buf()),
            arguments: user_args.unwrap_or_else(|| {
                if let Some(config) = json_config.as_object_mut()
                    && let Some(source_languages) = config.get("sourceLanguages").filter(|value| {
                        value
                            .as_array()
                            .is_some_and(|array| array.iter().all(Value::is_string))
                    })
                {
                    let ret = vec![
                        "--settings".into(),
                        json!({"sourceLanguages": source_languages}).to_string(),
                    ];
                    config.remove("sourceLanguages");
                    ret
                } else {
                    vec![]
                }
            }),
            request_args: self
                .request_args(delegate, json_config, &config.label)
                .await?,
            envs,
            connection: None,
        })
    }
}

#[cfg(test)]
mod diagnostic_debug_tests {
    use super::*;
    use std::path::Path;

    fn definition() -> DebugTaskDefinition {
        DebugTaskDefinition {
            label: "Reviewed development launch".into(),
            adapter: CodeLldbDebugAdapter::ADAPTER_NAME.into(),
            config: json!({
                "request":"launch", "program":"/diagnostic/program",
                "args":["selected-input"], "sourceLanguages":["rust"]
            }),
            tcp_connection: None,
        }
    }

    #[test]
    fn diagnostic_debug_installed_codelldb_preview_has_no_provisioning_or_terminal() {
        let adapter = CodeLldbDebugAdapter::default();
        let definition = definition();
        assert!(
            adapter
                .diagnostic_installed_binary(
                    &definition,
                    Path::new("/diagnostic"),
                    None,
                    None,
                    None,
                )
                .is_err()
        );
        let environment = HashMap::from_iter([("REVIEWED_MODE".into(), "development".into())]);
        let binary = adapter
            .diagnostic_installed_binary(
                &definition,
                Path::new("/diagnostic"),
                Some(Path::new("/installed/codelldb")),
                None,
                Some(&environment),
            )
            .expect("Explicit installed preview");
        assert_eq!(binary.command.as_deref(), Some("/installed/codelldb"));
        assert_eq!(binary.envs, environment);
        assert_eq!(
            binary.arguments,
            vec!["--settings", r#"{"sourceLanguages":["rust"]}"#]
        );
        assert_eq!(binary.request_args.configuration["cwd"], "/diagnostic");
        assert_eq!(binary.request_args.configuration["terminal"], "console");
        assert!(
            binary
                .request_args
                .configuration
                .get("sourceLanguages")
                .is_none()
        );
        assert!(binary.connection.is_none());
        assert!(adapter.path_to_codelldb.get().is_none());
        assert_eq!(definition.config["sourceLanguages"], json!(["rust"]));
    }

    #[test]
    fn diagnostic_debug_installed_codelldb_blocks_opaque_configuration() {
        let adapter = CodeLldbDebugAdapter::default();
        for (key, value) in [
            ("request", json!("attach")),
            ("initCommands", json!(["unapproved command"])),
            ("envFile", json!("private environment")),
            ("terminal", json!("integrated")),
            ("console", json!("integratedTerminal")),
            ("args", json!("opaque argument string")),
            ("env", json!({"VALUE": false})),
            ("sourceLanguages", json!([false])),
            ("stopOnEntry", json!("unknown")),
        ] {
            let mut definition = definition();
            definition.config[key] = value;
            assert!(
                adapter
                    .diagnostic_installed_binary(
                        &definition,
                        Path::new("/diagnostic"),
                        Some(Path::new("/installed/codelldb")),
                        None,
                        None,
                    )
                    .is_err(),
                "Configuration {key} must be blocked"
            );
        }
        assert!(
            adapter
                .diagnostic_installed_binary(
                    &definition(),
                    Path::new("/diagnostic"),
                    Some(Path::new("/installed/codelldb")),
                    Some(&["--port".into(), "1234".into()]),
                    None,
                )
                .is_err()
        );
    }
}
