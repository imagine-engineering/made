use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::io::Read as _;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};
use std::time::Duration;

use gpui::{App, Context, Entity, Task, WeakEntity};
use language::{Buffer, PointUtf16, ToOffset as _};
use lsp::{LanguageServer, LanguageServerId};
use rpc::proto;
use util::{ResultExt as _, paths::UrlExt as _};

use crate::jvm_project::{
    JdtlsImportedBuildKind, KotlinBuildSystem, KotlinWorkspaceDependency, jdtls_nested_preference,
    jdtls_preference,
};
use crate::project_settings::ProjectSettings;
use crate::trusted_worktrees::TrustedWorktrees;
use crate::{File, LspStore, ProjectPath};
use settings::Settings as _;

const COMPANION_SIZE: u64 = 122_117;
const COMPANION_SHA256: &str = "6c0f070ca799ce6e89d859cb317044b354f3cae1b0e3509d9be64a335ccd990e";
pub(super) const PINNED_COMPANION_JDTLS_VERSION: &str = "1.61.0-SNAPSHOT";
const COMPANION_PROBE: &str = "groovy.gradle.importedBuildRoots.v2";
const COMPANION_COMMANDS: &[&str] = &[
    "groovy.gradle.projects",
    "groovy.gradle.projectDetails",
    "groovy.gradle.projectDetails.v3",
    "groovy.gradle.projectDetails.v4",
    "groovy.gradle.sourceOutputs",
    "groovy.gradle.dependencyGraph",
    "groovy.maven.projects",
    "groovy.maven.projectDetails",
    "groovy.maven.goalDetails",
    "groovy.maven.dependencyGraph",
    "groovy.gradle.defaultImportRoots",
    "groovy.gradle.defaultImportRoots.v2",
    "groovy.gradle.importSelected",
    "groovy.gradle.importSelected.v2",
    "groovy.gradle.applyModuleStates",
    "groovy.gradle.moduleStates",
    "groovy.gradle.rootState",
    "groovy.gradle.moduleStates.v2",
    "groovy.gradle.applyModuleStates.v2",
    "groovy.gradle.rootState.v2",
    "groovy.gradle.applyRootState.v2",
    "groovy.gradle.applyRootState",
    "groovy.maven.defaultImportRoots",
    "groovy.maven.defaultImportRoots.v2",
    "groovy.maven.importSelected",
    "groovy.maven.importReactor",
    "groovy.maven.importReactorWithMasks",
    "groovy.java.importPlainProjects",
    "groovy.java.addLooseSource",
    "groovy.maven.importedProjects",
    "groovy.gradle.importedProjects",
    "groovy.gradle.importedBuildRoots",
    COMPANION_PROBE,
    "groovy.maven.unlinkSelected",
    "groovy.gradle.unlinkSelected",
];

pub(super) fn apply_build_import_settings_at_startup(
    workspace_configuration: &Value,
    initialization_options: &mut Option<Value>,
) {
    // JDTLS can start importing builds before receiving workspace/didChangeConfiguration.
    const PREFERENCES: &[&str] = &[
        "java.import.gradle.annotationProcessing.enabled",
        "java.import.gradle.arguments",
        "java.import.gradle.enabled",
        "java.import.gradle.home",
        "java.import.gradle.java.home",
        "java.import.gradle.jvmArguments",
        "java.import.gradle.offline.enabled",
        "java.import.gradle.user.home",
        "java.import.gradle.version",
        "java.import.gradle.wrapper.enabled",
        "java.import.maven.enabled",
        "java.import.maven.offline.enabled",
        "java.configuration.maven.globalSettings",
        "java.configuration.maven.userSettings",
        "java.jdt.ls.kotlinSupport.enabled",
    ];
    let mut preferences = PREFERENCES
        .iter()
        .filter_map(|&key| jdtls_preference(workspace_configuration, key).map(|value| (key, value)))
        .peekable();
    if preferences.peek().is_none() {
        return;
    }
    let Some(options) = initialization_options
        .get_or_insert_with(|| json!({}))
        .as_object_mut()
    else {
        return;
    };
    let settings = options.entry("settings").or_insert_with(|| json!({}));
    for (key, value) in preferences {
        if settings.get(key).is_none()
            && jdtls_nested_preference(settings, key).is_none()
            && let Some(settings) = settings.as_object_mut()
        {
            settings.insert(key.to_owned(), value.clone());
        }
    }
}

pub(super) async fn has_pinned_companion(
    fs: &dyn fs::Fs,
    options: Option<&serde_json::Value>,
) -> bool {
    let Some(bundles) = options
        .and_then(|options| options.get("bundles"))
        .and_then(serde_json::Value::as_array)
        .filter(|bundles| bundles.len() <= 32)
    else {
        return false;
    };
    for bundle in bundles.iter().filter_map(|bundle| bundle.as_str()) {
        let path = Path::new(bundle);
        if !path.is_absolute() || path.extension().is_none_or(|extension| extension != "jar") {
            continue;
        }
        match fs.metadata(path).await {
            Ok(Some(metadata))
                if !metadata.is_dir && !metadata.is_fifo && metadata.len == COMPANION_SIZE => {}
            Ok(_) => continue,
            Err(error) => {
                log::debug!("Could not check a configured JDTLS bundle: {error:#}");
                continue;
            }
        }
        let Some(file) = fs.open_sync(path).await.log_err() else {
            continue;
        };
        let mut bytes = Vec::with_capacity(COMPANION_SIZE as usize);
        if file
            .take(COMPANION_SIZE + 1)
            .read_to_end(&mut bytes)
            .log_err()
            .is_some()
            && bytes.len() as u64 == COMPANION_SIZE
            && format!("{:x}", Sha256::digest(bytes)) == COMPANION_SHA256
        {
            return true;
        }
    }
    false
}

pub(crate) async fn recover_pinned_companion_commands(
    server: &LanguageServer,
    worktree_root: &Path,
    request_timeout: Duration,
) -> bool {
    if server.name() != lsp::LanguageServerName::new_static("jdtls")
        || server.version().as_deref() != Some(PINNED_COMPANION_JDTLS_VERSION)
        || server
            .capabilities()
            .execute_command_provider
            .is_some_and(|provider| {
                COMPANION_COMMANDS
                    .iter()
                    .all(|command| provider.commands.iter().any(|item| item == command))
            })
    {
        return false;
    }
    let initialized_folders = server.workspace_folders();
    let Ok(scopes) = crate::jvm_project::imported_build_read_roots(
        initialized_folders
            .iter()
            .map(|folder| folder.as_str().to_owned()),
        worktree_root,
    ) else {
        return false;
    };
    let Some(root) = scopes
        .first()
        .and_then(|root| url::Url::from_directory_path(root).ok())
    else {
        return false;
    };

    // The pinned JDTLS retries its Eclipse registry lookup on an unknown command.
    // The cached-only probe establishes that the checked bundle's handler can answer.
    let response = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: COMPANION_PROBE.into(),
                arguments: vec![serde_json::json!({"rootUri": root, "cachedOnly": true})],
                ..Default::default()
            },
            request_timeout.min(Duration::from_secs(4)),
        )
        .await
        .into_response();
    let Ok(Some(reply)) = response else {
        log::debug!("Pinned JDTLS companion did not answer its cached capability probe");
        return false;
    };
    if initialized_folders != server.workspace_folders() || !is_companion_probe_reply(&reply, &root)
    {
        return false;
    }
    server.update_capabilities(|capabilities| {
        let provider = capabilities
            .execute_command_provider
            .get_or_insert_with(Default::default);
        for command in COMPANION_COMMANDS {
            if !provider.commands.iter().any(|item| item == command) {
                provider.commands.push((*command).to_owned());
            }
        }
    });
    true
}

fn is_companion_probe_reply(reply: &serde_json::Value, root: &url::Url) -> bool {
    reply.get("version").and_then(serde_json::Value::as_u64) == Some(2)
        && reply.get("rootUri").and_then(serde_json::Value::as_str) == Some(root.as_str())
        && reply.get("cachedOnly").and_then(serde_json::Value::as_bool) == Some(true)
        && reply.get("readOnly").and_then(serde_json::Value::as_bool) == Some(true)
        && reply
            .get("fullSynchronizationKnown")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        && reply
            .get("projects")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|rows| rows.len() <= 128)
        && reply
            .get("topologies")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|rows| rows.len() <= 128)
}

#[derive(PartialEq, Eq)]
struct MavenKotlinDefinitionScope {
    java_generation: u64,
    kotlin_generation: u64,
    kotlin_server_id: LanguageServerId,
    roots: Vec<ProjectPath>,
}

impl LspStore {
    fn maven_kotlin_definition_scope(
        &self,
        source: &ProjectPath,
        java_server: LanguageServerId,
        cx: &App,
    ) -> Option<MavenKotlinDefinitionScope> {
        self.as_local()?;
        let trusted = TrustedWorktrees::try_get_global(cx)?;
        if trusted
            .read(cx)
            .observed_worktree_trust(&self.worktree_store, source.worktree_id, cx)
            != Some(true)
            || TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
        {
            return None;
        }
        let (java, java_current) = self.jdtls_project_snapshot(source.worktree_id)?;
        let (kotlin, kotlin_current) = self.kotlin_workspace_snapshot(source.worktree_id)?;
        if !java_current || !kotlin_current || java.server_id != java_server {
            return None;
        }
        let origin = java
            .projects
            .iter()
            .filter(|project| {
                project.membership.is_imported()
                    && project.imported_build_kind == JdtlsImportedBuildKind::Maven
                    && source.path.starts_with(&project.module_root.path)
                    && project
                        .source_roots
                        .iter()
                        .any(|root| root.navigable && source.path.starts_with(&root.path.path))
            })
            .max_by_key(|project| project.module_root.path.as_unix_str().len())?;
        let kotlin_origin = kotlin.projects.iter().find(|project| {
            project.build_system == KotlinBuildSystem::Maven
                && project.module_root == origin.module_root
        })?;
        let roots = kotlin_origin
            .dependencies
            .iter()
            .filter_map(|dependency| {
                let KotlinWorkspaceDependency::Module { name, scope } = dependency else {
                    return None;
                };
                if scope.as_deref() != Some("compile") || !origin.project_references.contains(name)
                {
                    return None;
                }
                let mut matches = kotlin.projects.iter().filter(|project| {
                    project.build_system == KotlinBuildSystem::Maven
                        && project.name == *name
                        && java.projects.iter().any(|project_java| {
                            project_java.module_root == project.module_root
                                && project_java.membership.is_imported()
                                && project_java.imported_build_kind == JdtlsImportedBuildKind::Maven
                        })
                });
                let target = matches.next()?;
                matches.next().is_none().then_some(target)
            })
            .flat_map(|project| {
                project
                    .source_roots
                    .iter()
                    .filter(|root| root.navigable && !root.generated)
                    .map(|root| root.path.clone())
            })
            .collect::<Vec<_>>();
        (!roots.is_empty()).then_some(MavenKotlinDefinitionScope {
            java_generation: java.generation,
            kotlin_generation: kotlin.generation,
            kotlin_server_id: kotlin.server_id,
            roots,
        })
    }

    pub(crate) fn maven_kotlin_definition(
        &self,
        buffer: &Entity<Buffer>,
        position: PointUtf16,
        java_server: LanguageServerId,
        cx: &mut Context<Self>,
    ) -> Option<Task<anyhow::Result<Option<lsp::Location>>>> {
        let source_buffer = buffer.read(cx);
        let file = File::from_dyn(source_buffer.file())?;
        if !file.is_local || file.is_private {
            return None;
        }
        let source = ProjectPath {
            worktree_id: file.worktree.read(cx).id(),
            path: file.path.clone(),
        };
        let scope = self.maven_kotlin_definition_scope(&source, java_server, cx)?;
        let (class_name, package) = java_type_name(source_buffer, position)?;
        let query = format!("{package}.{class_name}");
        let server = self.language_server_for_id(scope.kotlin_server_id)?;
        let request = server.request::<lsp::request::WorkspaceSymbolRequest>(
            lsp::WorkspaceSymbolParams {
                query: query.clone(),
                ..Default::default()
            },
            ProjectSettings::get_global(cx)
                .global_lsp_settings
                .get_request_timeout(),
        );
        let buffer = buffer.clone();
        let source_version = source_buffer.version();
        Some(cx.spawn(async move |store, cx| {
            let Some(response) = request.await.into_response()? else {
                return Ok(None);
            };
            let candidates = match response {
                lsp::WorkspaceSymbolResponse::Flat(symbols) => symbols
                    .into_iter()
                    .map(|symbol| {
                        (
                            symbol.name,
                            symbol.container_name,
                            symbol.kind,
                            symbol.location,
                        )
                    })
                    .collect::<Vec<_>>(),
                lsp::WorkspaceSymbolResponse::Nested(symbols) => symbols
                    .into_iter()
                    .filter_map(|symbol| {
                        let lsp::OneOf::Left(location) = symbol.location else {
                            return None;
                        };
                        Some((symbol.name, symbol.container_name, symbol.kind, location))
                    })
                    .collect(),
            };
            store.read_with(cx, |store, cx| {
                if candidates.len() > 256
                    || buffer.read(cx).version() != source_version
                    || store
                        .maven_kotlin_definition_scope(&source, java_server, cx)
                        .as_ref()
                        != Some(&scope)
                {
                    return None;
                }
                let worktrees = store.worktree_store.read(cx);
                let mut matches =
                    candidates
                        .into_iter()
                        .filter_map(|(name, container, kind, location)| {
                            if (name != class_name && name != query)
                                || container.as_deref() != Some(&package)
                                || !matches!(
                                    kind,
                                    lsp::SymbolKind::CLASS
                                        | lsp::SymbolKind::INTERFACE
                                        | lsp::SymbolKind::ENUM
                                )
                            {
                                return None;
                            }
                            let path =
                                location.uri.to_file_path_ext(worktrees.path_style()).ok()?;
                            let (worktree, relative) = worktrees.find_worktree(&path, cx)?;
                            let worktree = worktree.read(cx);
                            let entry = worktree.entry_for_path(&relative)?;
                            (worktree.id() == source.worktree_id
                                && worktree.is_visible()
                                && path.extension().is_some_and(|extension| extension == "kt")
                                && entry.is_file()
                                && !entry.is_external
                                && !entry.is_private
                                && !entry.is_ignored
                                && !entry.is_fifo
                                && scope
                                    .roots
                                    .iter()
                                    .any(|root| relative.starts_with(&root.path)))
                            .then_some(location)
                        });
                let result = matches.next()?;
                matches.all(|other| other == result).then_some(result)
            })
        }))
    }
}

fn java_type_name(buffer: &Buffer, position: PointUtf16) -> Option<(String, String)> {
    if buffer.language()?.name() != language::LanguageName::new_static("Java") {
        return None;
    }
    let snapshot = buffer.snapshot();
    let offset = position.to_offset(&snapshot);
    let node = snapshot.syntax_ancestor(offset..offset)?;
    if node.kind() != "type_identifier"
        || !node.byte_range().contains(&offset)
        || node
            .parent()
            .is_some_and(|parent| parent.kind() == "scoped_type_identifier")
    {
        return None;
    }
    let class_name = snapshot
        .text_for_range(node.byte_range())
        .collect::<String>();
    let mut root = node;
    while let Some(parent) = root.parent() {
        root = parent;
    }
    let mut cursor = root.walk();
    let mut package = None;
    for child in root.named_children(&mut cursor) {
        let mut names = child.walk();
        match child.kind() {
            "package_declaration" => {
                package = child
                    .named_children(&mut names)
                    .find(|name| matches!(name.kind(), "identifier" | "scoped_identifier"))
                    .map(|name| {
                        snapshot
                            .text_for_range(name.byte_range())
                            .collect::<String>()
                    });
            }
            "import_declaration" => {
                if child
                    .children(&mut names)
                    .any(|token| matches!(token.kind(), "static" | "asterisk" | "*"))
                {
                    continue;
                }
                let Some(name) = child
                    .named_children(&mut names)
                    .find(|name| name.kind() == "scoped_identifier")
                else {
                    continue;
                };
                let declaration = snapshot
                    .text_for_range(name.byte_range())
                    .collect::<String>();
                let Some((owner, name)) = declaration.rsplit_once('.') else {
                    continue;
                };
                if name == class_name
                    && owner
                        .split('.')
                        .all(|part| !part.is_empty() && !part.contains(char::is_whitespace))
                {
                    return Some((class_name, owner.to_owned()));
                }
            }
            _ => {}
        }
    }
    Some((class_name, package?))
}

#[derive(Debug)]
struct StatusReport;

impl lsp::notification::Notification for StatusReport {
    type Params = StatusReportParams;
    const METHOD: &'static str = "language/status";
}

#[derive(Debug)]
struct ProjectEvent;

impl lsp::notification::Notification for ProjectEvent {
    type Params = ProjectEventParams;
    const METHOD: &'static str = "language/eventNotification";
}

#[derive(Debug, Deserialize, Serialize)]
struct ProjectEventParams {
    #[serde(rename = "eventType")]
    event_type: u32,
}

const CLASSPATH_UPDATED_EVENT: u32 = 100;
const PROJECTS_IMPORTED_EVENT: u32 = 200;
const PROJECTS_DELETED_EVENT: u32 = 210;

#[derive(Debug, Deserialize, Serialize)]
struct StatusReportParams {
    #[serde(rename = "type")]
    kind: String,
    message: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(super) enum ImportFailure {
    MavenOfflineCache = 1,
    GradlePlugin = 2,
}

impl ImportFailure {
    pub(super) fn from_log(message: &str) -> Option<Self> {
        if message.contains("in offline mode") && message.contains("has not been downloaded") {
            Some(Self::MavenOfflineCache)
        } else if super::gradle_plugin_resolution_failed(message) {
            Some(Self::GradlePlugin)
        } else {
            None
        }
    }

    fn from_status(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::MavenOfflineCache),
            2 => Some(Self::GradlePlugin),
            _ => None,
        }
    }

    pub(super) fn message(self) -> &'static str {
        match self {
            Self::MavenOfflineCache => {
                "Java Maven import needs cached dependencies; populate the host Maven cache or disable Maven offline mode, then reimport"
            }
            Self::GradlePlugin => {
                "Java Gradle import could not resolve a build plugin; check the host plugin repositories and cache, then reimport"
            }
        }
    }
}

fn status_health(
    params: &StatusReportParams,
    import_failure: Option<ImportFailure>,
) -> Option<(proto::ServerHealth, &'static str)> {
    match (params.kind.as_str(), params.message.as_deref()) {
        ("Error", _) => Some((
            proto::ServerHealth::Error,
            "Java language server reported an error; view language server logs",
        )),
        ("ProjectStatus", Some("WARNING")) => Some(match import_failure {
            Some(failure) => (proto::ServerHealth::Error, failure.message()),
            None => (
                proto::ServerHealth::Warning,
                "Java project reports errors; view Problems and language server logs",
            ),
        }),
        _ => None,
    }
}

pub(super) fn register_notifications(
    lsp_store: WeakEntity<LspStore>,
    language_server: &LanguageServer,
    pending_import_failure: Arc<AtomicU8>,
) {
    let server_id: LanguageServerId = language_server.server_id();
    let name = language_server.name();
    let project_events_store = lsp_store.clone();
    language_server
        .on_notification::<StatusReport, _>(move |params, cx| {
            let import_failure = if params.kind == "ProjectStatus" {
                match params.message.as_deref() {
                    Some("WARNING") => {
                        ImportFailure::from_status(pending_import_failure.swap(0, Ordering::AcqRel))
                    }
                    Some("OK") => {
                        pending_import_failure.store(0, Ordering::Release);
                        lsp_store
                            .update(cx, |store, cx| {
                                let worktree_id = store
                                    .language_server_statuses
                                    .get(&server_id)
                                    .and_then(|status| status.worktree);
                                if let Some(worktree_id) = worktree_id
                                    && store.jdtls_project_snapshot(worktree_id).is_some_and(
                                        |(snapshot, current)| {
                                            snapshot.server_id == server_id && !current
                                        },
                                    )
                                {
                                    store.schedule_jdtls_project_refresh(
                                        server_id,
                                        worktree_id,
                                        cx,
                                    );
                                }
                            })
                            .log_err();
                        None
                    }
                    _ => None,
                }
            } else {
                None
            };
            let Some((health, message)) = status_health(&params, import_failure) else {
                return;
            };
            super::emit_jvm_provider_health(&lsp_store, server_id, &name, health, message, cx);
        })
        .detach();

    language_server
        .on_notification::<ProjectEvent, _>(move |params, cx| {
            if !matches!(
                params.event_type,
                CLASSPATH_UPDATED_EVENT | PROJECTS_IMPORTED_EVENT | PROJECTS_DELETED_EVENT
            ) {
                return;
            }
            let result = project_events_store.update(cx, |store, cx| {
                let worktree_id = store
                    .language_server_statuses
                    .get(&server_id)
                    .filter(|status| status.name == lsp::LanguageServerName::new_static("jdtls"))
                    .and_then(|status| status.worktree);
                if let Some(worktree_id) = worktree_id {
                    if let Some(state) = store.jdtls_projects.get_mut(&worktree_id) {
                        if state.import_operation().is_some()
                            || state.failed_import() == Some(server_id)
                        {
                            return;
                        }
                        if let Err(error) = state.mark_stale() {
                            log::warn!(
                                "Could not invalidate changed JDTLS project metadata: {error:#}"
                            );
                            return;
                        }
                    }
                    store.schedule_jdtls_project_refresh(server_id, worktree_id, cx);
                }
            });
            if let Err(error) = result {
                log::debug!("JDTLS project event arrived after LSP store teardown: {error:#}");
            }
        })
        .detach();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_import_startup_preserves_explicit_initialization_settings() {
        let home = "java.import.gradle.java.home";
        let gradle_offline = "java.import.gradle.offline.enabled";
        let maven_offline = "java.import.maven.offline.enabled";
        let maven_user_settings = "java.configuration.maven.userSettings";
        let workspace_configuration = json!({
            "java": {"import": {"gradle": {
                "java": {"home": "/gradle/live"},
                "offline": {"enabled": true},
                "wrapper": {"enabled": false},
            }}},
            "java.import.gradle.home": "/gradle/distribution",
            "java.import.maven.offline.enabled": true,
            "java.configuration.maven.userSettings": "/maven/live.xml",
            "java.format.enabled": false,
        });
        let mut options = Some(json!({
            "bundles": ["fixture.jar"],
            "settings": {
                "java.import.gradle.offline.enabled": false,
                "java.configuration.maven.userSettings": "/maven/explicit.xml",
                "java": {"import": {
                    "gradle": {"java": {"home": null}},
                    "maven": {"offline": {"enabled": false}},
                }},
            },
        }));

        apply_build_import_settings_at_startup(&workspace_configuration, &mut options);

        let options = options.expect("initialization options");
        assert_eq!(options["bundles"], json!(["fixture.jar"]));
        let settings = &options["settings"];
        assert!(settings.get(home).is_none());
        assert_eq!(jdtls_nested_preference(settings, home), Some(&Value::Null));
        assert_eq!(settings[gradle_offline], false);
        assert!(settings.get(maven_offline).is_none());
        assert_eq!(
            jdtls_nested_preference(settings, maven_offline),
            Some(&Value::Bool(false))
        );
        assert_eq!(settings[maven_user_settings], "/maven/explicit.xml");
        assert_eq!(settings["java.import.gradle.wrapper.enabled"], false);
        assert_eq!(settings["java.import.gradle.home"], "/gradle/distribution");
        assert!(settings.get("java.format.enabled").is_none());
    }

    #[test]
    fn pinned_companion_recovery_uses_only_the_reviewed_manifest_commands() {
        let manifest = include_str!("../../../../tooling/jvm/gradle-model/plugin.xml");
        let commands: Vec<_> = manifest
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix("<command id=\"")?
                    .strip_suffix("\"/>")
            })
            .collect();
        assert_eq!(COMPANION_COMMANDS, commands);
    }

    #[test]
    fn pinned_companion_probe_requires_the_exact_cached_read_reply() {
        let root = url::Url::parse("file:///trusted/").expect("initialized folder");
        let reply = serde_json::json!({
            "version": 2,
            "rootUri": root,
            "cachedOnly": true,
            "readOnly": true,
            "fullSynchronizationKnown": false,
            "partial": true,
            "projects": [],
            "topologies": [{"topologyKnown": false, "projects": []}]
        });
        assert!(is_companion_probe_reply(&reply, &root));
        for (key, value) in [
            ("version", serde_json::json!(1)),
            ("rootUri", serde_json::json!("file:///different/")),
            ("cachedOnly", serde_json::json!(false)),
            ("readOnly", serde_json::json!(false)),
            ("fullSynchronizationKnown", serde_json::json!(true)),
            ("projects", serde_json::json!(vec![0; 129])),
            ("topologies", serde_json::Value::Null),
        ] {
            let mut rejected = reply.clone();
            rejected[key] = value;
            assert!(!is_companion_probe_reply(&rejected, &root), "{key}");
        }
    }

    #[test]
    fn jdtls_status_reports_failure_without_exposing_host_paths_or_claiming_import_success() {
        let status: StatusReportParams = serde_json::from_str(
            r#"{"type":"Error","message":"Import failed at /private/customer/project"}"#,
        )
        .expect("JDTLS error status");
        let (health, message) = status_health(&status, None).expect("visible failure");
        assert_eq!(health, proto::ServerHealth::Error);
        assert!(!message.contains("/private/customer"));

        let warning: StatusReportParams =
            serde_json::from_str(r#"{"type":"ProjectStatus","message":"WARNING"}"#)
                .expect("JDTLS project warning");
        assert_eq!(
            status_health(&warning, None),
            Some((
                proto::ServerHealth::Warning,
                "Java project reports errors; view Problems and language server logs"
            ))
        );
        for payload in [
            r#"{"type":"ServiceReady","message":"ServiceReady"}"#,
            r#"{"type":"ProjectStatus","message":"OK"}"#,
            r#"{"type":"Starting","message":"Starting"}"#,
        ] {
            let status: StatusReportParams = serde_json::from_str(payload).expect("JDTLS status");
            assert_eq!(status_health(&status, None), None);
        }
    }

    #[test]
    fn jdtls_offline_maven_warning_reports_host_cache_action_without_log_paths() {
        assert_eq!(
            ImportFailure::from_log(
                "Could not calculate build plan: Cannot access central in offline mode and the artifact org.apache.maven.plugins:maven-resources-plugin:jar:3.4.0 has not been downloaded from it before at /private/host/path"
            ),
            Some(ImportFailure::MavenOfflineCache)
        );
        assert_eq!(
            ImportFailure::from_log(
                "Could not calculate build plan: dependency missing at /private/host/path"
            ),
            None
        );
        let warning: StatusReportParams =
            serde_json::from_str(r#"{"type":"ProjectStatus","message":"WARNING"}"#)
                .expect("JDTLS project warning");
        let (health, message) = status_health(&warning, Some(ImportFailure::MavenOfflineCache))
            .expect("offline cache failure");
        assert_eq!(health, proto::ServerHealth::Error);
        assert_eq!(message, ImportFailure::MavenOfflineCache.message());
        assert!(!message.contains("/private/host"));
    }

    #[test]
    fn jdtls_gradle_plugin_resolution_reports_recovery_without_log_paths() {
        let log = "Synchronize project /private/host/project failed: Caused by: org.gradle.api.plugins.UnknownPluginException: Plugin [id: 'org.jetbrains.kotlin.jvm', version: '2.0.21', apply: false] was not found; Plugin Repositories (could not resolve plugin artifact 'org.jetbrains.kotlin.jvm:org.jetbrains.kotlin.jvm.gradle.plugin:2.0.21')";
        assert_eq!(
            ImportFailure::from_log(log),
            Some(ImportFailure::GradlePlugin)
        );
        assert_eq!(
            ImportFailure::from_log("Gradle daemon stopped at /private/host/project"),
            None
        );
        let warning: StatusReportParams =
            serde_json::from_str(r#"{"type":"ProjectStatus","message":"WARNING"}"#)
                .expect("JDTLS project warning");
        let (health, message) = status_health(&warning, Some(ImportFailure::GradlePlugin))
            .expect("Gradle plugin failure");
        assert_eq!(health, proto::ServerHealth::Error);
        assert_eq!(message, ImportFailure::GradlePlugin.message());
        assert!(!message.contains("/private/host"));
    }
}
