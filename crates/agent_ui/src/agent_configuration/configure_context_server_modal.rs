use anyhow::{Context as _, Result};
use collections::HashMap;
use context_server::{ContextServerCommand, ContextServerId};
use editor::{Editor, EditorElement, EditorStyle};

use extension_host::ExtensionStore;
use gpui::{
    AsyncWindowContext, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, ScrollHandle,
    Subscription, Task, TextStyle, TextStyleRefinement, UnderlineStyle, WeakEntity, prelude::*,
};
use language::{Language, LanguageRegistry};
use markdown::{Markdown, MarkdownElement, MarkdownStyle};
use notifications::status_toast::StatusToast;
use parking_lot::Mutex;
use project::{
    context_server_store::{
        ContextServerStatus, ContextServerStore, ServerStatusChangedEvent,
        registry::ContextServerDescriptorRegistry,
    },
    project_settings::{ContextServerSettings, OAuthClientSettings, ProjectSettings},
    worktree_store::WorktreeStore,
};
use serde::Deserialize;
use settings::{Settings as _, update_settings_file};
use std::sync::Arc;
use theme_settings::ThemeSettings;
use ui::{
    CommonAnimationExt, KeyBinding, Modal, ModalFooter, ModalHeader, Section, Tooltip,
    WithScrollbar, prelude::*,
};
use util::ResultExt as _;
use workspace::{ModalView, Workspace};

enum ConfigurationTarget {
    Existing {
        id: ContextServerId,
        command: ContextServerCommand,
    },
    ExistingHttp {
        id: ContextServerId,
        url: String,
        headers: HashMap<String, String>,
        oauth: Option<OAuthClientSettings>,
    },

    Extension {
        id: ContextServerId,
        repository_url: Option<SharedString>,
        installation: Option<extension::ContextServerConfiguration>,
    },
}

enum ExistingServerType {
    Local,
    Remote,
}

enum ConfigurationSource {
    Existing {
        editor: Entity<Editor>,
        server_type: ExistingServerType,
    },
    Extension {
        id: ContextServerId,
        editor: Option<Entity<Editor>>,
        repository_url: Option<SharedString>,
        installation_instructions: Option<Entity<markdown::Markdown>>,
        settings_validator: Option<jsonschema::Validator>,
    },
}

impl ConfigurationSource {
    fn has_configuration_options(&self) -> bool {
        !matches!(self, ConfigurationSource::Extension { editor: None, .. })
    }

    fn from_target(
        target: ConfigurationTarget,
        language_registry: Arc<LanguageRegistry>,
        jsonc_language: Option<Arc<Language>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        fn create_editor(
            json: String,
            jsonc_language: Option<Arc<Language>>,
            window: &mut Window,
            cx: &mut App,
        ) -> Entity<Editor> {
            cx.new(|cx| {
                let mut editor = Editor::auto_height(4, 16, window, cx);
                editor.set_text(json, window, cx);
                editor.set_show_gutter(false, cx);
                editor.set_soft_wrap_mode(language::language_settings::SoftWrap::None, cx);
                if let Some(buffer) = editor.buffer().read(cx).as_singleton() {
                    buffer.update(cx, |buffer, cx| buffer.set_language(jsonc_language, cx))
                }
                editor
            })
        }

        match target {
            ConfigurationTarget::Existing { id, command } => ConfigurationSource::Existing {
                editor: create_editor(
                    context_server_input(Some((id, command))),
                    jsonc_language,
                    window,
                    cx,
                ),
                server_type: ExistingServerType::Local,
            },
            ConfigurationTarget::ExistingHttp {
                id,
                url,
                headers: auth,
                oauth,
            } => ConfigurationSource::Existing {
                editor: create_editor(
                    context_server_http_input(Some((id, url, auth, oauth))),
                    jsonc_language,
                    window,
                    cx,
                ),
                server_type: ExistingServerType::Remote,
            },

            ConfigurationTarget::Extension {
                id,
                repository_url,
                installation,
            } => {
                let settings_validator = installation.as_ref().and_then(|installation| {
                    jsonschema::validator_for(&installation.settings_schema)
                        .context("Failed to load JSON schema for context server settings")
                        .log_err()
                });
                let installation_instructions = installation.as_ref().map(|installation| {
                    cx.new(|cx| {
                        Markdown::new(
                            installation.installation_instructions.clone().into(),
                            Some(language_registry.clone()),
                            None,
                            cx,
                        )
                    })
                });
                ConfigurationSource::Extension {
                    id,
                    repository_url,
                    installation_instructions,
                    settings_validator,
                    editor: installation.map(|installation| {
                        create_editor(installation.default_settings, jsonc_language, window, cx)
                    }),
                }
            }
        }
    }

    fn output(&self, cx: &mut App) -> Result<(ContextServerId, ContextServerSettings)> {
        match self {
            ConfigurationSource::Existing {
                editor,
                server_type,
            } => match *server_type {
                ExistingServerType::Remote => {
                    parse_http_input(&editor.read(cx).text(cx)).map(|(id, url, auth, oauth)| {
                        (
                            id,
                            ContextServerSettings::Http {
                                enabled: true,
                                url,
                                headers: auth,
                                timeout: None,
                                oauth,
                            },
                        )
                    })
                }
                ExistingServerType::Local => {
                    parse_input(&editor.read(cx).text(cx)).map(|(id, command)| {
                        (
                            id,
                            ContextServerSettings::Stdio {
                                enabled: true,
                                remote: false,
                                command,
                            },
                        )
                    })
                }
            },
            ConfigurationSource::Extension {
                id,
                editor,
                settings_validator,
                ..
            } => {
                let text = editor
                    .as_ref()
                    .context("No output available")?
                    .read(cx)
                    .text(cx);
                let settings = serde_json_lenient::from_str::<serde_json::Value>(&text)?;
                if let Some(settings_validator) = settings_validator
                    && let Err(error) = settings_validator.validate(&settings)
                {
                    return Err(anyhow::anyhow!(error.to_string()));
                }
                Ok((
                    id.clone(),
                    ContextServerSettings::Extension {
                        enabled: true,
                        remote: false,
                        settings,
                    },
                ))
            }
        }
    }
}

fn context_server_input(existing: Option<(ContextServerId, ContextServerCommand)>) -> String {
    let (name, command, args, env, cwd) = match existing {
        Some((id, cmd)) => {
            let args = serde_json::to_string(&cmd.args).unwrap();
            let env = serde_json::to_string(&cmd.env.unwrap_or_default()).unwrap();
            let cmd_path = serde_json::to_string(&cmd.path).unwrap();
            (
                id.0.to_string(),
                cmd_path,
                args,
                env,
                cmd.cwd
                    .map(|cwd| format!(",\n    \"cwd\": {}", serde_json::json!(cwd)))
                    .unwrap_or_default(),
            )
        }
        None => (
            "some-mcp-server".to_string(),
            "".to_string(),
            "[]".to_string(),
            "{}".to_string(),
            String::new(),
        ),
    };

    format!(
        r#"{{
  /// Configure an MCP server that runs locally via stdin/stdout
  ///
  /// The name of your MCP server
  "{name}": {{
    /// The command which runs the MCP server
    "command": {command},
    /// The arguments to pass to the MCP server
    "args": {args},
    /// The environment variables to set
    "env": {env}{cwd}
  }}
}}"#
    )
}

fn context_server_http_input(
    existing: Option<(
        ContextServerId,
        String,
        HashMap<String, String>,
        Option<OAuthClientSettings>,
    )>,
) -> String {
    let (name, url, headers, oauth) = match existing {
        Some((id, url, headers, oauth)) => {
            let headers = if headers.is_empty() {
                r#"// "Authorization": "Bearer <token>"#.to_string()
            } else {
                let json = serde_json::to_string_pretty(&headers).unwrap();
                let mut lines = json.split("\n").collect::<Vec<_>>();
                if lines.len() > 1 {
                    lines.remove(0);
                    lines.pop();
                }
                lines
                    .into_iter()
                    .map(|line| format!("  {}", line))
                    .collect::<String>()
            };
            (id.0.to_string(), url, headers, oauth)
        }
        None => (
            "some-remote-server".to_string(),
            "https://example.com/mcp".to_string(),
            r#"// "Authorization": "Bearer <token>"#.to_string(),
            None,
        ),
    };

    let oauth = oauth.map_or_else(
        || {
            r#"
    /// Uncomment to use a pre-registered OAuth client. You can include the client secret here as well, otherwise it will be prompted interactively and saved in the system keychain.
    // "oauth": {
    //   "client_id": "your-client-id",
    // },"#
                .to_string()
        },

        |oauth| {
            let mut lines = vec![
                String::from("\n    \"oauth\": {"),

                format!("      \"client_id\": {},", serde_json::to_string(&oauth.client_id).unwrap()),
            ];
            if let Some(client_secret) = oauth.client_secret {
                lines.push(format!(
                    "      \"client_secret\": {}",
                    serde_json::to_string(&client_secret).unwrap()
                ));
            } else {
                lines.push(String::from(
                    "      /// Optional client secret for confidential clients\n      // \"client_secret\": \"your-client-secret\"",
                ));
            }
            lines.push(String::from("    },"));

            lines.join("\n")
        },
    );

    format!(
        r#"{{
  /// Configure an MCP server that you connect to over HTTP
  ///
  /// The name of your remote MCP server
  "{name}": {{
    /// The URL of the remote MCP server
    "url": "{url}",{oauth}
    "headers": {{
     /// Any headers to send along
     {headers}
    }}
  }}
}}"#
    )
}

fn parse_http_input(
    text: &str,
) -> Result<(
    ContextServerId,
    String,
    HashMap<String, String>,
    Option<OAuthClientSettings>,
)> {
    #[derive(Deserialize)]
    struct Temp {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
        #[serde(default)]
        oauth: Option<OAuthClientSettings>,
    }
    let value: HashMap<String, Temp> = serde_json_lenient::from_str(text)?;
    if value.len() != 1 {
        anyhow::bail!("Expected exactly one context server configuration");
    }

    let (key, value) = value.into_iter().next().unwrap();

    Ok((
        ContextServerId(key.into()),
        value.url,
        value.headers,
        value.oauth,
    ))
}

fn resolve_context_server_extension(
    id: ContextServerId,
    worktree_store: Entity<WorktreeStore>,
    cx: &mut App,
) -> Task<Option<ConfigurationTarget>> {
    let registry = ContextServerDescriptorRegistry::default_global(cx).read(cx);

    let Some(descriptor) = registry.context_server_descriptor(&id.0) else {
        return Task::ready(None);
    };

    let extension = ExtensionStore::global(cx)
        .read(cx)
        .installed_extensions()
        .iter()
        .find(|(_, entry)| entry.manifest.context_servers.contains_key(&id.0))
        .map(|(id, entry)| (id.clone(), entry.manifest.clone()));
    cx.spawn(async move |cx| {
        let installation = descriptor
            .configuration(worktree_store, cx)
            .await
            .context("Failed to resolve context server configuration")
            .log_err()
            .flatten();

        Some(ConfigurationTarget::Extension {
            id,
            repository_url: extension
                .and_then(|(_, manifest)| manifest.repository.clone().map(SharedString::from)),
            installation,
        })
    })
}

enum State {
    Idle,
    Waiting,
    AuthRequired {
        server_id: ContextServerId,
    },
    ClientSecretRequired {
        server_id: ContextServerId,
        error: Option<SharedString>,
    },
    Authenticating {
        server_id: ContextServerId,
    },
    Error(SharedString),
}

#[derive(Clone)]
struct PackageReview {
    package_id: String,
    package_name: String,
    revision: String,
    server_name: String,
    settings: ContextServerSettings,
}

fn package_review_target(
    id: ContextServerId,
    settings: ContextServerSettings,
) -> Result<ConfigurationTarget> {
    match settings {
        ContextServerSettings::Stdio { mut command, .. } => {
            if let Some(environment) = &mut command.env {
                for (name, value) in environment {
                    if name != "PLUGIN_ROOT" && name != "PLUGIN_DATA" {
                        *value = "<value hidden>".into();
                    }
                }
            }
            Ok(ConfigurationTarget::Existing { id, command })
        }
        ContextServerSettings::Http {
            url,
            mut headers,
            mut oauth,
            ..
        } => {
            extension_host::agent_plugins::ServerDeclaration::StreamableHttp {
                url: url.clone(),
                headers: headers.clone().into_iter().collect(),
            }
            .validate()?;
            for value in headers.values_mut() {
                *value = "<value hidden>".into();
            }
            if let Some(oauth) = &mut oauth {
                oauth.client_secret = oauth
                    .client_secret
                    .as_ref()
                    .map(|_| "<value hidden>".into());
            }
            Ok(ConfigurationTarget::ExistingHttp {
                id,
                url,
                headers,
                oauth,
            })
        }
        ContextServerSettings::Extension { .. } => {
            anyhow::bail!("Portable packages do not use native extension configuration")
        }
    }
}

pub struct ConfigureContextServerModal {
    context_server_store: Entity<ContextServerStore>,
    workspace: WeakEntity<Workspace>,
    source: ConfigurationSource,
    state: State,
    original_server_id: Option<ContextServerId>,
    scroll_handle: ScrollHandle,
    secret_editor: Entity<Editor>,
    _auth_subscription: Option<Subscription>,
    package_review: Option<PackageReview>,
    package_operation: u64,
}

impl ConfigureContextServerModal {
    pub fn show_modal_for_package_server(
        package_id: String,
        server_name: String,
        language_registry: Arc<LanguageRegistry>,
        workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let owner = ExtensionStore::global(cx);
        let Some(package) = owner.read(cx).agent_plugins.get(&package_id).cloned() else {
            return Task::ready(Err(anyhow::anyhow!("Package is no longer installed")));
        };
        let filesystem = owner.read(cx).fs.clone();
        let directory = owner.read(cx).agent_plugins_directory.clone();
        window.spawn(cx, async move |cx| {
            let base = extension_host::agent_plugins::prepare_server_settings(
                filesystem.as_ref(),
                &directory,
                &package,
                &server_name,
            )
            .await?;
            let key =
                extension_host::agent_plugins::package_configuration_key(&package_id, &server_name);
            let settings = workspace.update(cx, |workspace, cx| {
                workspace
                    .project()
                    .read(cx)
                    .context_server_store()
                    .read(cx)
                    .resolved_settings_for_server(&ContextServerId(key.clone().into()), cx)
                    .or_else(|| {
                        ProjectSettings::get_global(cx)
                            .context_servers
                            .get(key.as_str())
                            .cloned()
                    })
                    .unwrap_or_else(|| base.clone())
            })?;
            extension_host::agent_plugins::validate_effective_settings(
                filesystem.as_ref(),
                &base,
                &settings,
            )
            .await?;
            let id = ContextServerId(package.server_id(&server_name).into());
            let review = PackageReview {
                package_id,
                package_name: package.manifest.name,
                revision: package.revision,
                server_name,
                settings: settings.clone(),
            };
            let target = package_review_target(id, settings)?;
            Self::show_modal_with_package(target, Some(review), language_registry, workspace, cx)
                .await
        })
    }

    fn initial_state(
        context_server_store: &Entity<ContextServerStore>,
        target: &ConfigurationTarget,
        cx: &App,
    ) -> State {
        let server_id = match target {
            ConfigurationTarget::Existing { id, .. }
            | ConfigurationTarget::ExistingHttp { id, .. }
            | ConfigurationTarget::Extension { id, .. } => id,
        };

        match context_server_store.read(cx).status_for_server(server_id) {
            Some(ContextServerStatus::AuthRequired) => State::AuthRequired {
                server_id: server_id.clone(),
            },
            Some(ContextServerStatus::ClientSecretRequired { error }) => {
                State::ClientSecretRequired {
                    server_id: server_id.clone(),
                    error: error.map(SharedString::from),
                }
            }
            Some(ContextServerStatus::Authenticating) => State::Authenticating {
                server_id: server_id.clone(),
            },
            Some(ContextServerStatus::Error(error)) => State::Error(error.into()),

            Some(ContextServerStatus::Starting)
            | Some(ContextServerStatus::Running)
            | Some(ContextServerStatus::Stopped)
            | None => State::Idle,
        }
    }

    pub fn show_modal_for_existing_server(
        server_id: ContextServerId,
        language_registry: Arc<LanguageRegistry>,
        workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<()>> {
        if server_id.0.starts_with("agent-plugin/") {
            let owner = ExtensionStore::global(cx);
            let package_server = owner.read(cx).agent_plugins.values().find_map(|package| {
                package.servers.keys().find_map(|name| {
                    (package.server_id(name) == server_id.0.as_ref()
                        || extension_host::agent_plugins::package_configuration_key(
                            &package.id,
                            name,
                        ) == server_id.0.as_ref())
                    .then(|| (package.id.clone(), name.clone()))
                })
            });
            return match package_server {
                Some((id, name)) => Self::show_modal_for_package_server(
                    id,
                    name,
                    language_registry,
                    workspace,
                    window,
                    cx,
                ),
                None => Task::ready(Err(anyhow::anyhow!(
                    "Package server changed; reopen its review from Extensions"
                ))),
            };
        }
        let Some(settings) = ProjectSettings::get_global(cx)
            .context_servers
            .get(&server_id.0)
            .cloned()
            .or_else(|| {
                ContextServerDescriptorRegistry::default_global(cx)
                    .read(cx)
                    .context_server_descriptor(&server_id.0)
                    .map(|_| ContextServerSettings::default_extension())
            })
        else {
            return Task::ready(Err(anyhow::anyhow!("Context server not found")));
        };

        window.spawn(cx, async move |cx| {
            let target = match settings {
                ContextServerSettings::Stdio {
                    enabled: _,
                    command,
                    ..
                } => Some(ConfigurationTarget::Existing {
                    id: server_id,
                    command,
                }),
                ContextServerSettings::Http {
                    enabled: _,
                    url,
                    headers,
                    timeout: _,
                    oauth,
                } => Some(ConfigurationTarget::ExistingHttp {
                    id: server_id,
                    url,
                    headers,
                    oauth,
                }),

                ContextServerSettings::Extension { .. } => {
                    match workspace
                        .update(cx, |workspace, cx| {
                            resolve_context_server_extension(
                                server_id,
                                workspace.project().read(cx).worktree_store(),
                                cx,
                            )
                        })
                        .ok()
                    {
                        Some(task) => task.await,
                        None => None,
                    }
                }
            };

            match target {
                Some(target) => Self::show_modal(target, language_registry, workspace, cx).await,
                None => Err(anyhow::anyhow!("Failed to resolve context server")),
            }
        })
    }

    fn show_modal(
        target: ConfigurationTarget,
        language_registry: Arc<LanguageRegistry>,
        workspace: WeakEntity<Workspace>,
        cx: &mut AsyncWindowContext,
    ) -> Task<Result<()>> {
        Self::show_modal_with_package(target, None, language_registry, workspace, cx)
    }

    fn show_modal_with_package(
        target: ConfigurationTarget,
        package_review: Option<PackageReview>,
        language_registry: Arc<LanguageRegistry>,
        workspace: WeakEntity<Workspace>,
        cx: &mut AsyncWindowContext,
    ) -> Task<Result<()>> {
        cx.spawn(async move |cx| {
            let jsonc_language = language_registry.language_for_name("jsonc").await.ok();
            workspace.update_in(cx, |workspace, window, cx| {
                let workspace_handle = cx.weak_entity();
                let context_server_store = workspace.project().read(cx).context_server_store();
                workspace.toggle_modal(window, cx, |window, cx| Self {
                    context_server_store: context_server_store.clone(),
                    workspace: workspace_handle,
                    state: Self::initial_state(&context_server_store, &target, cx),

                    original_server_id: Some(match &target {
                        ConfigurationTarget::Existing { id, .. }
                        | ConfigurationTarget::ExistingHttp { id, .. }
                        | ConfigurationTarget::Extension { id, .. } => id.clone(),
                    }),
                    source: {
                        let source = ConfigurationSource::from_target(
                            target,
                            language_registry,
                            jsonc_language,
                            window,
                            cx,
                        );
                        if package_review.is_some()
                            && let ConfigurationSource::Existing { editor, .. } = &source
                        {
                            editor.update(cx, |editor, _| editor.set_read_only(true));
                        }
                        source
                    },
                    scroll_handle: ScrollHandle::new(),
                    secret_editor: cx.new(|cx| {
                        let mut editor = Editor::single_line(window, cx);
                        editor.set_placeholder_text(
                            "Enter client secret (leave empty for public clients)",
                            window,
                            cx,
                        );
                        editor.set_masked(true, cx);
                        editor
                    }),
                    _auth_subscription: None,
                    package_review,
                    package_operation: 0,
                })
            })
        })
    }

    fn set_error(&mut self, err: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.state = State::Error(err.into());
        cx.notify();
    }

    fn approve_package_server(&mut self, cx: &mut Context<Self>) {
        let Some(review) = self.package_review.clone() else {
            return;
        };
        if !review.settings.enabled() {
            self.set_error("This server is disabled in MCP settings. Enable that setting, then reopen this review.", cx);
            return;
        }
        let key = extension_host::agent_plugins::package_configuration_key(
            &review.package_id,
            &review.server_name,
        );
        let current = self
            .context_server_store
            .read(cx)
            .resolved_settings_for_server(&ContextServerId(key.into()), cx);
        if current.is_some_and(|settings| settings != review.settings) {
            self.set_error(
                "MCP settings changed. Reopen the server review before approving.",
                cx,
            );
            return;
        }
        let owner = ExtensionStore::global(cx);
        let operation = owner.update(cx, |owner, cx| {
            owner.change_agent_plugin(
                &review.package_id,
                extension_host::agent_plugins::PackageChange::ApproveServerConfiguration {
                    name: review.server_name.clone(),
                    revision: review.revision.clone(),
                    settings: review.settings,
                },
                cx,
            )
        });
        self.package_operation = self.package_operation.wrapping_add(1);
        let generation = self.package_operation;
        self.state = State::Waiting;
        cx.notify();
        cx.spawn(async move |this, cx| {
            if let Err(error) = operation.await {
                this.update(cx, |this, cx| {
                    if this.package_operation == generation {
                        this.set_error(format!("{error:#}"), cx);
                    }
                })
                .log_err();
                return;
            }
            let id = owner.read_with(cx, |owner, _| {
                owner
                    .agent_plugins
                    .get(&review.package_id)
                    .map(|package| ContextServerId(package.server_id(&review.server_name).into()))
            });
            let Some(id) = id else {
                return;
            };
            let waiting = this
                .update(cx, |this, cx| {
                    if this.package_operation != generation {
                        return None;
                    }
                    this.original_server_id = Some(id.clone());
                    Some(wait_for_context_server(
                        &this.context_server_store,
                        id.clone(),
                        cx,
                    ))
                })
                .ok()
                .flatten();
            let Some(waiting) = waiting else {
                return;
            };
            let result = waiting.await;
            this.update(cx, |this, cx| {
                if this.package_operation != generation {
                    return;
                }
                match result {
                    Ok(ContextServerStatus::Running) => {
                        this.state = State::Idle;
                        this.show_configured_context_server_toast(id, cx);
                        cx.emit(DismissEvent);
                    }
                    Ok(ContextServerStatus::AuthRequired) => {
                        this.state = State::AuthRequired { server_id: id }
                    }
                    Ok(ContextServerStatus::ClientSecretRequired { error }) => {
                        this.state = State::ClientSecretRequired {
                            server_id: id,
                            error: error.map(SharedString::from),
                        }
                    }
                    Err(error) => this.set_error(error, cx),
                    Ok(_) => this.state = State::Idle,
                }
                cx.notify();
            })
            .log_err();
        })
        .detach();
    }

    fn revoke_package_server(&mut self, cx: &mut Context<Self>) {
        let Some(review) = self.package_review.clone() else {
            return;
        };
        let operation = ExtensionStore::global(cx).update(cx, |owner, cx| {
            owner.change_agent_plugin(
                &review.package_id,
                extension_host::agent_plugins::PackageChange::RevokeServer(review.server_name),
                cx,
            )
        });
        self.package_operation = self.package_operation.wrapping_add(1);
        let generation = self.package_operation;
        self.state = State::Waiting;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = operation.await;
            this.update(cx, |this, cx| {
                if this.package_operation != generation {
                    return;
                }
                match result {
                    Ok(()) => this.state = State::Idle,
                    Err(error) => this.set_error(format!("{error:#}"), cx),
                }
                cx.notify();
            })
            .log_err();
        })
        .detach();
    }

    fn confirm(&mut self, _: &menu::Confirm, cx: &mut Context<Self>) {
        if matches!(self.state, State::Waiting | State::Authenticating { .. }) {
            return;
        }
        if self.package_review.is_some() {
            self.approve_package_server(cx);
            return;
        }

        self._auth_subscription = None;

        self.state = State::Idle;
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        let (id, settings) = match self.source.output(cx) {
            Ok(val) => val,
            Err(error) => {
                self.set_error(error.to_string(), cx);
                return;
            }
        };

        self.state = State::Waiting;

        let existing_server = self.context_server_store.read(cx).get_server(&id);
        if existing_server.is_some() {
            self.context_server_store.update(cx, |store, cx| {
                store.stop_server(&id, cx).log_err();
            });
        }

        let wait_for_context_server_task =
            wait_for_context_server(&self.context_server_store, id.clone(), cx);
        cx.spawn({
            let id = id.clone();
            async move |this, cx| {
                let result = wait_for_context_server_task.await;
                this.update(cx, |this, cx| match result {
                    Ok(ContextServerStatus::Running) => {
                        this.state = State::Idle;
                        this.show_configured_context_server_toast(id, cx);
                        cx.emit(DismissEvent);
                    }
                    Ok(ContextServerStatus::AuthRequired) => {
                        this.state = State::AuthRequired { server_id: id };
                        cx.notify();
                    }
                    Ok(ContextServerStatus::ClientSecretRequired { error }) => {
                        this.state = State::ClientSecretRequired {
                            server_id: id,
                            error: error.map(SharedString::from),
                        };
                        cx.notify();
                    }
                    Err(err) => {
                        this.set_error(err, cx);
                    }
                    Ok(_) => {}
                })
            }
        })
        .detach();

        let settings_changed =
            ProjectSettings::get_global(cx).context_servers.get(&id.0) != Some(&settings);

        if settings_changed {
            // When we write the settings to the file, the context server will be restarted.
            workspace.update(cx, |workspace, cx| {
                let fs = workspace.app_state().fs.clone();
                let original_server_id = self.original_server_id.clone();
                update_settings_file(fs.clone(), cx, move |current, _| {
                    if let Some(original_id) = original_server_id {
                        if original_id != id {
                            current.project.context_servers.remove(&original_id.0);
                        }
                    }
                    current
                        .project
                        .context_servers
                        .insert(id.0, settings.into());
                });
            });
        } else if let Some(existing_server) = existing_server {
            self.context_server_store
                .update(cx, |store, cx| store.start_server(existing_server, cx));
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn cancel_authentication(&mut self, server_id: &ContextServerId, cx: &mut Context<Self>) {
        self._auth_subscription = None;
        self.context_server_store.update(cx, |store, cx| {
            store.stop_server(server_id, cx).log_err();
        });
        self.state = State::Idle;
        cx.notify();
    }

    fn authenticate(&mut self, server_id: ContextServerId, cx: &mut Context<Self>) {
        self.context_server_store.update(cx, |store, cx| {
            store.authenticate_server(&server_id, cx).log_err();
        });
        self.await_auth_outcome(server_id, cx);
    }

    fn submit_client_secret(&mut self, server_id: ContextServerId, cx: &mut Context<Self>) {
        let secret = self.secret_editor.read(cx).text(cx);
        self.context_server_store.update(cx, |store, cx| {
            store.submit_client_secret(&server_id, secret, cx).log_err();
        });
        self.await_auth_outcome(server_id, cx);
    }

    fn await_auth_outcome(&mut self, server_id: ContextServerId, cx: &mut Context<Self>) {
        self.state = State::Authenticating {
            server_id: server_id.clone(),
        };

        self._auth_subscription = Some(cx.subscribe(
            &self.context_server_store,
            move |this, _, event: &ServerStatusChangedEvent, cx| {
                if event.server_id != server_id {
                    return;
                }
                match &event.status {
                    ContextServerStatus::Running => {
                        this._auth_subscription = None;
                        this.state = State::Idle;
                        this.show_configured_context_server_toast(event.server_id.clone(), cx);
                        cx.emit(DismissEvent);
                    }
                    ContextServerStatus::AuthRequired => {
                        this._auth_subscription = None;
                        this.state = State::AuthRequired {
                            server_id: event.server_id.clone(),
                        };
                        cx.notify();
                    }
                    ContextServerStatus::ClientSecretRequired { error } => {
                        this._auth_subscription = None;
                        this.state = State::ClientSecretRequired {
                            server_id: event.server_id.clone(),
                            error: error.clone().map(SharedString::from),
                        };
                        cx.notify();
                    }
                    ContextServerStatus::Error(error) => {
                        this._auth_subscription = None;
                        this.set_error(error.clone(), cx);
                    }
                    ContextServerStatus::Authenticating
                    | ContextServerStatus::Starting
                    | ContextServerStatus::Stopped => {}
                }
            },
        ));

        cx.notify();
    }

    fn show_configured_context_server_toast(&self, id: ContextServerId, cx: &mut App) {
        self.workspace
            .update(cx, {
                |workspace, cx| {
                    let status_toast = StatusToast::new(
                        format!("{} configured successfully.", id.0),
                        cx,
                        |this, _cx| {
                            this.icon(
                                Icon::new(IconName::ToolHammer)
                                    .size(IconSize::Small)
                                    .color(Color::Muted),
                            )
                            .action("Dismiss", |_, _| {})
                        },
                    );

                    workspace.toggle_status_toast(status_toast, cx);
                }
            })
            .log_err();
    }
}

fn parse_input(text: &str) -> Result<(ContextServerId, ContextServerCommand)> {
    let value: serde_json::Value = serde_json_lenient::from_str(text)?;
    let object = value.as_object().context("Expected object")?;
    anyhow::ensure!(object.len() == 1, "Expected exactly one key-value pair");
    let (context_server_name, value) = object.into_iter().next().unwrap();
    let command: ContextServerCommand = serde_json::from_value(value.clone())?;
    Ok((ContextServerId(context_server_name.clone().into()), command))
}

impl ModalView for ConfigureContextServerModal {}

impl Focusable for ConfigureContextServerModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        match &self.source {
            ConfigurationSource::Existing { editor, .. } => editor.focus_handle(cx),
            ConfigurationSource::Extension { editor, .. } => editor
                .as_ref()
                .map(|editor| editor.focus_handle(cx))
                .unwrap_or_else(|| cx.focus_handle()),
        }
    }
}

impl EventEmitter<DismissEvent> for ConfigureContextServerModal {}

impl ConfigureContextServerModal {
    fn render_modal_header(&self) -> ModalHeader {
        if let Some(review) = &self.package_review {
            return ModalHeader::new().headline(format!(
                "Review {} · {}",
                review.package_name, review.server_name
            ));
        }
        let text: SharedString = match &self.source {
            ConfigurationSource::Existing { .. } => "Configure MCP Server".into(),
            ConfigurationSource::Extension { id, .. } => format!("Configure {}", id.0).into(),
        };
        ModalHeader::new().headline(text)
    }

    fn render_modal_description(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        if let Some(review) = &self.package_review {
            return v_flex().gap_1().pb_2()
                .child(Label::new(format!("Package revision: {}", review.revision)))
                .child(Label::new("Approve this exact server configuration to connect. Tool permissions remain separate. Scheduling consent is unchanged."))
                .child(Label::new("Environment and header names are shown; secret values are hidden. Edit the source package and inspect an update to change its configuration. Existing MCP settings overrides require renewed review.").color(Color::Muted))
                .when_some(match &review.settings { ContextServerSettings::Stdio { command, .. } => command.cwd.as_ref(), _ => None }, |element, cwd| element.child(Label::new(format!("Working directory: {}", cwd.display()))))
                .child(Button::new("package-open-settings", "Open MCP Settings").tab_index(0_isize).on_click(|_, window, cx| {
                    window.dispatch_action(Box::new(zed_actions::OpenSettingsAt { path: "context_servers".into(), target: None }), cx);
                }))
                .into_any_element();
        }
        const MODAL_DESCRIPTION: &str =
            "Check the server docs for required arguments and environment variables.";

        if let ConfigurationSource::Extension {
            installation_instructions: Some(installation_instructions),
            ..
        } = &self.source
        {
            div()
                .pb_2()
                .text_sm()
                .child(MarkdownElement::new(
                    installation_instructions.clone(),
                    default_markdown_style(window, cx),
                ))
                .into_any_element()
        } else {
            Label::new(MODAL_DESCRIPTION)
                .color(Color::Muted)
                .into_any_element()
        }
    }

    fn render_modal_content(&self, cx: &App) -> AnyElement {
        let editor = match &self.source {
            ConfigurationSource::Existing { editor, .. } => editor,
            ConfigurationSource::Extension { editor, .. } => {
                let Some(editor) = editor else {
                    return div().into_any_element();
                };
                editor
            }
        };

        div()
            .p_2()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().colors().border_variant)
            .bg(cx.theme().colors().editor_background)
            .child({
                let settings = ThemeSettings::get_global(cx);
                let text_style = TextStyle {
                    color: cx.theme().colors().text,
                    font_family: settings.buffer_font.family.clone(),
                    font_fallbacks: settings.buffer_font.fallbacks.clone(),
                    font_size: settings.buffer_font_size(cx).into(),
                    font_weight: settings.buffer_font.weight,
                    line_height: relative(settings.buffer_line_height.value()),
                    ..Default::default()
                };
                EditorElement::new(
                    editor,
                    EditorStyle {
                        background: cx.theme().colors().editor_background,
                        local_player: cx.theme().players().local(),
                        text: text_style,
                        syntax: cx.theme().syntax().clone(),
                        ..Default::default()
                    },
                )
            })
            .into_any_element()
    }

    fn render_modal_footer(&self, cx: &mut Context<Self>) -> ModalFooter {
        let focus_handle = self.focus_handle(cx);
        let is_busy = matches!(self.state, State::Waiting | State::Authenticating { .. });

        ModalFooter::new()
            .start_slot::<Button>(if self.package_review.is_some() {
                Some(
                    Button::new("package-revoke-server", "Revoke Approval")
                        .tab_index(0_isize)
                        .disabled(ExtensionStore::global(cx).read(cx).agent_plugins_busy)
                        .on_click(cx.listener(|this, _, _, cx| this.revoke_package_server(cx))),
                )
            } else if let ConfigurationSource::Extension {
                repository_url: Some(repository_url),
                ..
            } = &self.source
            {
                Some(
                    Button::new("open-repository", "Open Repository")
                        .end_icon(
                            Icon::new(IconName::ArrowUpRight)
                                .size(IconSize::Small)
                                .color(Color::Muted),
                        )
                        .tooltip({
                            let repository_url = repository_url.clone();
                            move |_window, cx| {
                                Tooltip::with_meta(
                                    "Open Repository",
                                    None,
                                    repository_url.clone(),
                                    cx,
                                )
                            }
                        })
                        .on_click({
                            let repository_url = repository_url.clone();
                            move |_, _, cx| cx.open_url(&repository_url)
                        }),
                )
            } else {
                None
            })
            .end_slot(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new(
                            "cancel",
                            if self.package_review.is_some() && is_busy {
                                "Close"
                            } else if self.source.has_configuration_options() {
                                "Cancel"
                            } else {
                                "Dismiss"
                            },
                        )
                        .key_binding(
                            KeyBinding::for_action_in(&menu::Cancel, &focus_handle, cx)
                                .map(|kb| kb.size(rems_from_px(12_f32))),
                        )
                        .on_click(
                            cx.listener(|this, _event, _window, cx| this.cancel(&menu::Cancel, cx)),
                        ),
                    )
                    .children(self.source.has_configuration_options().then(|| {
                        Button::new(
                            "configure-server",
                            if self.package_review.is_some() {
                                "Approve and Connect"
                            } else {
                                "Configure Server"
                            },
                        )
                        .tab_index(0_isize)
                        .disabled(is_busy)
                        .key_binding(
                            KeyBinding::for_action_in(&menu::Confirm, &focus_handle, cx)
                                .map(|kb| kb.size(rems_from_px(12_f32))),
                        )
                        .on_click(
                            cx.listener(|this, _event, _window, cx| {
                                this.confirm(&menu::Confirm, cx)
                            }),
                        )
                    })),
            )
    }

    fn render_loading(&self, label: impl Into<SharedString>) -> Div {
        h_flex()
            .h_8()
            .gap_1p5()
            .justify_center()
            .child(
                Icon::new(IconName::LoadCircle)
                    .size(IconSize::XSmall)
                    .color(Color::Muted)
                    .with_rotate_animation(3),
            )
            .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
    }

    fn render_auth_required(&self, server_id: &ContextServerId, cx: &mut Context<Self>) -> Div {
        h_flex()
            .h_8()
            .min_w_0()
            .w_full()
            .gap_2()
            .justify_center()
            .child(
                h_flex()
                    .gap_1p5()
                    .child(
                        Icon::new(IconName::Info)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    )
                    .child(
                        Label::new("Authenticate to connect this server")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
            )
            .child(
                Button::new("authenticate-server", "Authenticate")
                    .style(ButtonStyle::Outlined)
                    .label_size(LabelSize::Small)
                    .on_click({
                        let server_id = server_id.clone();
                        cx.listener(move |this, _event, _window, cx| {
                            this.authenticate(server_id.clone(), cx);
                        })
                    }),
            )
    }

    fn render_client_secret_required(
        &self,
        server_id: &ContextServerId,
        error: Option<SharedString>,
        cx: &mut Context<Self>,
    ) -> Div {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.buffer_font.family.clone(),
            font_fallbacks: settings.buffer_font.fallbacks.clone(),
            font_size: settings.buffer_font_size(cx).into(),
            font_weight: settings.buffer_font.weight,
            line_height: relative(settings.buffer_line_height.value()),
            ..Default::default()
        };

        v_flex()
            .w_full()
            .gap_2()
            .when_some(error, |this, error| {
                this.child(Self::render_modal_error(error))
            })
            .child(
                h_flex()
                    .gap_1p5()
                    .child(
                        Icon::new(IconName::Info)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    )
                    .child(
                        Label::new(
                            "Enter your OAuth client secret, or leave empty for public clients",
                        )
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    ),
            )
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .capture_action({
                        let server_id = server_id.clone();
                        cx.listener(move |this, _: &editor::actions::Newline, _window, cx| {
                            this.submit_client_secret(server_id.clone(), cx);
                        })
                    })
                    .child(div().flex_1().child(EditorElement::new(
                        &self.secret_editor,
                        EditorStyle {
                            background: cx.theme().colors().editor_background,
                            local_player: cx.theme().players().local(),
                            text: text_style,
                            syntax: cx.theme().syntax().clone(),
                            ..Default::default()
                        },
                    )))
                    .child(
                        Button::new("submit-client-secret", "Submit")
                            .style(ButtonStyle::Outlined)
                            .label_size(LabelSize::Small)
                            .on_click({
                                let server_id = server_id.clone();
                                cx.listener(move |this, _event, _window, cx| {
                                    this.submit_client_secret(server_id.clone(), cx);
                                })
                            }),
                    ),
            )
    }

    fn render_authenticating(&self, server_id: &ContextServerId, cx: &mut Context<Self>) -> Div {
        h_flex()
            .h_8()
            .gap_2()
            .justify_center()
            .child(
                h_flex()
                    .gap_1p5()
                    .child(
                        Icon::new(IconName::LoadCircle)
                            .size(IconSize::XSmall)
                            .color(Color::Muted)
                            .with_rotate_animation(3),
                    )
                    .child(
                        Label::new("Authenticating…")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
            )
            .child(
                Button::new("cancel-authentication", "Cancel")
                    .style(ButtonStyle::Outlined)
                    .label_size(LabelSize::Small)
                    .on_click({
                        let server_id = server_id.clone();
                        cx.listener(move |this, _event, _window, cx| {
                            this.cancel_authentication(&server_id, cx);
                        })
                    }),
            )
    }

    fn render_modal_error(error: SharedString) -> Div {
        h_flex()
            .h_8()
            .gap_1p5()
            .justify_center()
            .child(
                Icon::new(IconName::Warning)
                    .size(IconSize::Small)
                    .color(Color::Warning),
            )
            .child(
                div()
                    .w_full()
                    .child(Label::new(error).size(LabelSize::Small).color(Color::Muted)),
            )
    }
}

impl Render for ConfigureContextServerModal {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .elevation_3(cx)
            .w(rems(40.))
            .key_context("ConfigureContextServerModal")
            .on_action(
                cx.listener(|this, _: &menu::Cancel, _window, cx| this.cancel(&menu::Cancel, cx)),
            )
            .on_action(
                cx.listener(|this, _: &menu::Confirm, _window, cx| {
                    this.confirm(&menu::Confirm, cx)
                }),
            )
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle(cx).focus(window, cx);
            }))
            .child(
                Modal::new("configure-context-server", None)
                    .header(self.render_modal_header())
                    .section(
                        Section::new().child(
                            div()
                                .size_full()
                                .child(
                                    div()
                                        .id("modal-content")
                                        .max_h(vh(0.7, window))
                                        .overflow_y_scroll()
                                        .track_scroll(&self.scroll_handle)
                                        .child(self.render_modal_description(window, cx))
                                        .child(self.render_modal_content(cx))
                                        .child(match &self.state {
                                            State::Idle => div(),
                                            State::Waiting => {
                                                self.render_loading("Connecting Server…")
                                            }
                                            State::AuthRequired { server_id } => {
                                                self.render_auth_required(&server_id.clone(), cx)
                                            }
                                            State::ClientSecretRequired { server_id, error } => {
                                                self.render_client_secret_required(
                                                    &server_id.clone(),
                                                    error.clone(),
                                                    cx,
                                                )
                                            }
                                            State::Authenticating { server_id } => {
                                                self.render_authenticating(&server_id.clone(), cx)
                                            }
                                            State::Error(error) => {
                                                Self::render_modal_error(error.clone())
                                            }
                                        }),
                                )
                                .vertical_scrollbar_for(&self.scroll_handle, window, cx),
                        ),
                    )
                    .footer(self.render_modal_footer(cx)),
            )
    }
}

fn wait_for_context_server(
    context_server_store: &Entity<ContextServerStore>,
    context_server_id: ContextServerId,
    cx: &mut App,
) -> Task<Result<ContextServerStatus, Arc<str>>> {
    use std::time::Duration;

    const WAIT_TIMEOUT: Duration = Duration::from_secs(120);

    match context_server_store
        .read(cx)
        .status_for_server(&context_server_id)
    {
        Some(
            status @ (ContextServerStatus::Running
            | ContextServerStatus::AuthRequired
            | ContextServerStatus::ClientSecretRequired { .. }),
        ) => return Task::ready(Ok(status)),
        Some(ContextServerStatus::Error(error)) => return Task::ready(Err(error)),
        _ => {}
    }

    let (tx, rx) = futures::channel::oneshot::channel();
    let tx = Arc::new(Mutex::new(Some(tx)));

    let context_server_id_for_timeout = context_server_id.clone();
    let subscription = cx.subscribe(context_server_store, move |_, event, _cx| {
        let ServerStatusChangedEvent { server_id, status } = event;

        if server_id != &context_server_id {
            return;
        }

        match status {
            ContextServerStatus::Running
            | ContextServerStatus::AuthRequired
            | ContextServerStatus::ClientSecretRequired { .. } => {
                if let Some(tx) = tx.lock().take() {
                    let _ = tx.send(Ok(status.clone()));
                }
            }
            ContextServerStatus::Stopped => {
                if let Some(tx) = tx.lock().take() {
                    let _ = tx.send(Err("Context server stopped running".into()));
                }
            }
            ContextServerStatus::Error(error) => {
                if let Some(tx) = tx.lock().take() {
                    let _ = tx.send(Err(error.clone()));
                }
            }
            ContextServerStatus::Starting | ContextServerStatus::Authenticating => {}
        }
    });

    cx.spawn(async move |cx| {
        let timeout = cx.background_executor().timer(WAIT_TIMEOUT);
        let result = futures::future::select(rx, timeout).await;
        drop(subscription);
        match result {
            futures::future::Either::Left((Ok(inner), _)) => inner,
            futures::future::Either::Left((Err(_), _)) => {
                Err(Arc::from("Context server store was dropped"))
            }
            futures::future::Either::Right(_) => Err(Arc::from(format!(
                "Timed out waiting for context server `{}` to start. Check the Zed log for details.",
                context_server_id_for_timeout
            ))),
        }
    })
}

pub(crate) fn default_markdown_style(window: &Window, cx: &App) -> MarkdownStyle {
    let theme_settings = ThemeSettings::get_global(cx);
    let colors = cx.theme().colors();
    let mut text_style = window.text_style();
    text_style.refine(&TextStyleRefinement {
        font_family: Some(theme_settings.ui_font.family.clone()),
        font_fallbacks: theme_settings.ui_font.fallbacks.clone(),
        font_features: Some(theme_settings.ui_font.features.clone()),
        font_size: Some(TextSize::XSmall.rems(cx).into()),
        color: Some(colors.text_muted),
        ..Default::default()
    });

    MarkdownStyle {
        base_text_style: text_style.clone(),
        selection_background_color: colors.element_selection_background,
        link: TextStyleRefinement {
            background_color: Some(colors.editor_foreground.opacity(0.025)),
            underline: Some(UnderlineStyle {
                color: Some(colors.text_accent.opacity(0.5)),
                thickness: px(1.),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[gpui::test]
    async fn plugin_context_server_desktop_permission_and_revocation_flow(
        cx: &mut gpui::TestAppContext,
    ) {
        use extension_host::agent_plugins::{
            MANIFEST_SCHEMA, MCP_SCHEMA, PackageChange, capture_package,
        };
        use gpui::UpdateGlobal as _;
        use project::{FakeFs, Project};
        use std::path::Path;
        crate::conversation_view::tests::init_test(cx);
        cx.update(|cx| {
            extension::init(cx);
            gpui_tokio::init(cx);
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .extension
                        .auto_install_extensions
                        .insert("html".into(), false);
                })
            });
        });
        cx.update(|cx| {
            language_model::LanguageModelRegistry::test(cx);
            agent::ThreadStore::init_global(cx);
        });
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_http = calls.clone();
        cx.update(|cx| cx.set_http_client(http_client::FakeHttpClient::create(move |request| {
            let calls = calls_for_http.clone();
            async move {
                use futures::AsyncReadExt as _;
                use http_client::{AsyncBody, Response};
                if request.method() != http_client::Method::POST {
                    return Ok(Response::builder().status(405).body(AsyncBody::empty())?);
                }
                let mut body = String::new();
                request.into_body().read_to_string(&mut body).await?;
                let message: serde_json::Value = serde_json::from_str(&body)?;
                let Some(id) = message.get("id") else {
                    return Ok(Response::builder().status(202).body(AsyncBody::empty())?);
                };
                let result = match message["method"].as_str() {
                    Some("initialize") => serde_json::json!({"protocolVersion":"2024-11-05", "capabilities":{"tools":{}}, "serverInfo":{"name":"review", "version":"1"}}),
                    Some("tools/list") => serde_json::json!({"tools":[{"name":"review", "description":"Review project", "inputSchema":{"type":"object", "properties":{}}}]}),
                    Some("tools/call") => {
                        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        serde_json::json!({"content":[{"type":"text", "text":"Reviewed"}]})
                    }
                    _ => serde_json::json!({}),
                };
                Ok(Response::builder().status(200).header("Content-Type", "application/json")
                    .body(AsyncBody::from(serde_json::json!({"jsonrpc":"2.0", "id":id, "result":result}).to_string()))?)
            }
        })));
        let filesystem = FakeFs::new(cx.executor());
        cx.update(|cx| {
            <dyn fs::Fs>::set_global(filesystem.clone(), cx);
            agent::recipes::RecipeStore::init_global(cx);
        });
        filesystem
            .insert_tree("/project", serde_json::json!({}))
            .await;
        filesystem.insert_tree("/package", serde_json::json!({
            "plugin.json": serde_json::json!({"$schema": MANIFEST_SCHEMA, "name": "review-package"}).to_string(),
            "skills": {"review": {"SKILL.md": "---\nname: review\ndescription: Review project changes\n---\nInspect risky changes"}},
            "mcp.json": serde_json::json!({"$schema": MCP_SCHEMA, "mcpServers": {"remote": {"type": "streamable-http", "url": "https://example.com/mcp", "headers": {"Authorization": "private-value"}}}}).to_string()
        })).await;
        let project = Project::test(filesystem.clone(), [Path::new("/project")], cx).await;
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        workspace.update(cx, |workspace, cx| {
            let state = workspace.app_state();
            extension_host::init(
                Arc::new(extension::ExtensionHostProxy::new()),
                filesystem.clone(),
                state.client.clone(),
                state.node_runtime.clone(),
                cx,
            );
        });
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(200));
        cx.run_until_parked();
        let owner = cx.update(|_, cx| ExtensionStore::global(cx));
        owner.update(cx, |owner, cx| owner.init_agent_plugins(cx));
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let tools = cx.update(|_, cx| {
            cx.new(|cx| agent::ContextServerRegistry::new(server_store.clone(), cx))
        });
        let snapshot = capture_package(
            filesystem.as_ref(),
            Path::new("/package"),
            Default::default(),
        )
        .await
        .expect("inspect");
        owner
            .update(cx, |owner, cx| owner.install_agent_plugin(snapshot, cx))
            .await
            .expect("install");
        let id = owner.read_with(cx, |owner, _| {
            owner
                .agent_plugins
                .keys()
                .next()
                .expect("installed")
                .clone()
        });
        use acp_thread::AgentConnection as _;
        let native = cx.update(|_, cx| {
            agent::NativeAgent::new(
                agent::ThreadStore::global(cx),
                agent::Templates::new(),
                filesystem.clone(),
                cx,
            )
        });
        let connection = std::rc::Rc::new(agent::NativeAgentConnection(native));
        let session = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    project
                        .read(cx)
                        .worktree_paths(cx)
                        .folder_path_list()
                        .clone(),
                    cx,
                )
            })
            .await
            .expect("native session");
        cx.run_until_parked();
        assert!(cx.update(|_, cx| {
            connection
                .available_skills(session.read(cx).session_id(), cx)
                .iter()
                .any(|skill| skill.name == "review" && skill.source.starts_with("plugin/"))
        }));
        for approve in [false, true] {
            workspace
                .update_in(cx, |workspace, window, cx| {
                    ConfigureContextServerModal::show_modal_for_package_server(
                        id.clone(),
                        "remote".into(),
                        workspace.app_state().languages.clone(),
                        workspace.weak_handle(),
                        window,
                        cx,
                    )
                })
                .await
                .expect("review modal");
            let modal = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<ConfigureContextServerModal>(cx)
                })
                .expect("modal");
            assert!(owner.read_with(cx, |owner, _| {
                owner
                    .agent_plugins
                    .get(&id)
                    .expect("installed")
                    .approved_servers
                    .is_empty()
            }));
            let text = modal.read_with(cx, |modal, cx| match &modal.source {
                ConfigurationSource::Existing { editor, .. } => editor.read(cx).text(cx),
                _ => panic!("HTTP review editor"),
            });
            assert!(!text.contains("private-value"));
            if !approve {
                cx.dispatch_action(menu::Cancel);
                cx.run_until_parked();
                continue;
            }
            cx.dispatch_action(menu::Confirm);
            cx.dispatch_action(menu::Confirm);
            cx.run_until_parked();
            assert!(owner.read_with(cx, |owner, _| {
                owner
                    .agent_plugins
                    .get(&id)
                    .expect("installed")
                    .approved_servers
                    .contains("remote")
            }));
            let server_id = owner.read_with(cx, |owner, _| {
                ContextServerId(
                    owner
                        .agent_plugins
                        .get(&id)
                        .expect("installed")
                        .server_id("remote")
                        .into(),
                )
            });
            assert!(server_store.read_with(cx, |store, _| matches!(
                store.status_for_server(&server_id),
                Some(ContextServerStatus::Running)
            )));
            let tool = tools
                .read_with(cx, |tools, _| {
                    tools.tools_for_server(&server_id).next().cloned()
                })
                .expect("published MCP tool");
            for allow in [false, true] {
                let (mut sender, input) = agent::ToolInput::<serde_json::Value>::test();
                sender.send_full(serde_json::json!({}));
                let (events, mut receiver) = agent::ToolCallEventStream::test();
                let task = cx.update(|_, cx| tool.clone().run(input, events, cx));
                let authorization = receiver.expect_authorization().await;
                authorization
                    .response
                    .send(acp_thread::SelectedPermissionOutcome::new(
                        agent_client_protocol::schema::v1::PermissionOptionId::new(if allow {
                            "allow"
                        } else {
                            "deny"
                        }),
                        if allow {
                            agent_client_protocol::schema::v1::PermissionOptionKind::AllowOnce
                        } else {
                            agent_client_protocol::schema::v1::PermissionOptionKind::RejectOnce
                        },
                    ))
                    .expect("permission response");
                assert_eq!(task.await.is_ok(), allow);
            }
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            workspace
                .update_in(cx, |workspace, window, cx| {
                    ConfigureContextServerModal::show_modal_for_package_server(
                        id.clone(),
                        "remote".into(),
                        workspace.app_state().languages.clone(),
                        workspace.weak_handle(),
                        window,
                        cx,
                    )
                })
                .await
                .expect("reopen connected server review");
            let modal = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<ConfigureContextServerModal>(cx)
                })
                .expect("active revocation UI");
            let (mut sender, input) = agent::ToolInput::<serde_json::Value>::test();
            sender.send_full(serde_json::json!({}));
            let (events, mut receiver) = agent::ToolCallEventStream::test();
            let pending = cx.update(|_, cx| tool.clone().run(input, events, cx));
            let authorization = receiver.expect_authorization().await;
            modal.update(cx, |modal, cx| modal.revoke_package_server(cx));
            cx.run_until_parked();
            assert!(owner.read_with(cx, |owner, _| {
                owner
                    .agent_plugins
                    .get(&id)
                    .expect("installed")
                    .approved_servers
                    .is_empty()
            }));
            authorization
                .response
                .send(acp_thread::SelectedPermissionOutcome::new(
                    agent_client_protocol::schema::v1::PermissionOptionId::new("allow"),
                    agent_client_protocol::schema::v1::PermissionOptionKind::AllowOnce,
                ))
                .expect("late permission response");
            assert!(pending.await.is_err());
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(server_store.read_with(cx, |store, _| {
                store.get_running_server(&server_id).is_none()
            }));
            owner
                .update(cx, |owner, cx| {
                    owner.change_agent_plugin(&id, PackageChange::Enabled(false), cx)
                })
                .await
                .expect("disable");
            cx.run_until_parked();
            assert!(!cx.update(|_, cx| {
                connection
                    .available_skills(session.read(cx).session_id(), cx)
                    .iter()
                    .any(|skill| skill.source.starts_with("plugin/"))
            }));
            assert!(cx.update(|_, cx| {
                agent::recipes::RecipeStore::try_global(cx)
                    .expect("recipe owner")
                    .read(cx)
                    .state
                    .schedules
                    .is_empty()
            }));
            modal.update(cx, |modal, cx| modal.confirm(&menu::Confirm, cx));
            cx.run_until_parked();
            assert!(modal.read_with(cx, |modal, _| matches!(modal.state, State::Error(_))));
            assert!(owner.read_with(cx, |owner, _| {
                owner
                    .agent_plugins
                    .get(&id)
                    .expect("installed")
                    .approved_servers
                    .is_empty()
            }));
        }
    }

    #[test]
    fn package_context_server_review_redacts_secrets_and_preserves_launch_details() {
        let settings = ContextServerSettings::Stdio {
            enabled: true,
            remote: false,
            command: ContextServerCommand {
                path: "/managed/server".into(),
                args: vec!["--safe".into()],
                env: Some(HashMap::from_iter([
                    ("TOKEN".into(), "private-token".into()),
                    ("PLUGIN_ROOT".into(), "/managed".into()),
                ])),
                cwd: Some("/managed/resources".into()),
                timeout: None,
            },
        };
        let target =
            package_review_target(ContextServerId("package".into()), settings).expect("review");
        let ConfigurationTarget::Existing { command, .. } = target else {
            panic!("stdio review");
        };
        assert_eq!(command.path, std::path::PathBuf::from("/managed/server"));
        assert_eq!(command.args, ["--safe"]);
        assert_eq!(command.cwd, Some("/managed/resources".into()));
        let environment = command.env.expect("environment");
        assert_eq!(
            environment.get("TOKEN").map(String::as_str),
            Some("<value hidden>")
        );
        assert_eq!(
            environment.get("PLUGIN_ROOT").map(String::as_str),
            Some("/managed")
        );
        let target = package_review_target(
            ContextServerId("package".into()),
            ContextServerSettings::Http {
                enabled: true,
                url: "https://example.com/mcp".into(),
                headers: HashMap::from_iter([("Authorization".into(), "secret".into())]),
                oauth: Some(OAuthClientSettings {
                    client_id: "client".into(),
                    client_secret: Some("secret".into()),
                }),
                timeout: None,
            },
        )
        .expect("HTTP review");
        let ConfigurationTarget::ExistingHttp { headers, oauth, .. } = target else {
            panic!("HTTP review");
        };
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("<value hidden>")
        );
        assert_eq!(
            oauth.expect("OAuth").client_secret.as_deref(),
            Some("<value hidden>")
        );
        assert!(
            package_review_target(
                ContextServerId("package".into()),
                ContextServerSettings::Http {
                    enabled: true,
                    url: "http://example.com/mcp".into(),
                    headers: HashMap::default(),
                    oauth: None,
                    timeout: None,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn parse_http_input_reads_oauth_settings() {
        let (id, url, headers, oauth) = parse_http_input(
            r#"{
  "figma": {
    "url": "https://mcp.figma.com/mcp",
    "oauth": {
      "client_id": "client-id",
      "client_secret": "client-secret"
    },
    "headers": {
      "X-Test": "test"
    }
  }
}"#,
        )
        .unwrap();

        assert_eq!(id, ContextServerId("figma".into()));
        assert_eq!(url, "https://mcp.figma.com/mcp");
        assert_eq!(headers.get("X-Test"), Some(&String::from("test")));
        let oauth = oauth.expect("oauth should be present");
        assert_eq!(oauth.client_id, "client-id");
        assert_eq!(oauth.client_secret.as_deref(), Some("client-secret"));
    }

    #[test]
    fn context_server_http_input_preserves_existing_oauth_settings() {
        let text = context_server_http_input(Some((
            ContextServerId("figma".into()),
            String::from("https://mcp.figma.com/mcp"),
            HashMap::default(),
            Some(OAuthClientSettings {
                client_id: String::from("client-id"),
                client_secret: Some(String::from("client-secret")),
            }),
        )));

        let (_, _, _, oauth) = parse_http_input(&text).unwrap();
        let oauth = oauth.expect("oauth should be present");
        assert_eq!(oauth.client_id, "client-id");
        assert_eq!(oauth.client_secret.as_deref(), Some("client-secret"));
    }
}
