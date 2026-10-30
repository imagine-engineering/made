use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use gpui::{AsyncApp, WeakEntity};
use lsp::{LanguageServer, LanguageServerId};
use rpc::proto;

use crate::LspStore;

#[derive(Debug)]
struct WorkspaceImportState;

impl lsp::notification::Notification for WorkspaceImportState {
    type Params = WorkspaceImportStateParams;
    const METHOD: &'static str = "intellij/workspaceImportState";
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkspaceImportStateParams {
    phase: String,
    folders: Vec<ImportFolder>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ImportFolder {
    status: String,
    message: Option<String>,
    tool: Option<String>,
}

#[derive(Debug)]
struct ImportLog;

impl lsp::notification::Notification for ImportLog {
    type Params = ImportLogParams;
    const METHOD: &'static str = "intellij/importLog";
}

#[derive(Debug, Deserialize, Serialize)]
struct ImportLogParams {
    #[serde(default)]
    failed: bool,
    #[serde(default)]
    started: bool,
    tool: Option<String>,
    message: Option<String>,
}

#[derive(Default)]
struct ImportHealthState {
    maven_import_active: bool,
    gradle_import_active: bool,
    maven_offline_cache_incomplete: bool,
    maven_dependency_fetch_failed: bool,
    gradle_plugin_resolution_failed: bool,
    gradle_plugin_exception_seen: bool,
    gradle_plugin_artifact_unavailable_seen: bool,
}

const MAVEN_OFFLINE_CACHE_MESSAGE: &str =
    "Kotlin Maven import needs cached dependencies; fetch them or disable Maven offline mode";
const MAVEN_FETCH_MESSAGE: &str = "Kotlin Maven import could not fetch dependencies; check network or repository access, or populate the cache";
const GRADLE_PLUGIN_MESSAGE: &str = "Kotlin Gradle import cannot resolve a build plugin; check host plugin repository access or restore the cache, then reimport";

#[derive(Clone, Copy)]
pub(super) enum KotlinImportEvent {
    Started,
    Finished { successful: bool },
}

fn import_state_health(
    params: &WorkspaceImportStateParams,
    state: &mut ImportHealthState,
) -> Option<(proto::ServerHealth, &'static str)> {
    if params.phase != "FINISHED" {
        return None;
    }
    let maven_offline_cache_incomplete = state.maven_offline_cache_incomplete;
    let maven_dependency_fetch_failed = state.maven_dependency_fetch_failed;
    let gradle_plugin_resolution_failed = state.gradle_plugin_resolution_failed;
    state.maven_import_active = false;
    state.gradle_import_active = false;
    state.maven_offline_cache_incomplete = false;
    state.maven_dependency_fetch_failed = false;
    state.gradle_plugin_resolution_failed = false;
    state.gradle_plugin_exception_seen = false;
    state.gradle_plugin_artifact_unavailable_seen = false;
    let failed_folders = params
        .folders
        .iter()
        .filter(|folder| folder.status == "FAILED")
        .collect::<Vec<_>>();
    if !failed_folders.is_empty() {
        if maven_offline_cache_incomplete
            && matches!(failed_folders.as_slice(), [folder] if folder.tool.as_deref() == Some("maven"))
        {
            return Some((proto::ServerHealth::Error, MAVEN_OFFLINE_CACHE_MESSAGE));
        }
        if maven_dependency_fetch_failed
            && matches!(failed_folders.as_slice(), [folder] if folder.tool.as_deref() == Some("maven"))
        {
            return Some((proto::ServerHealth::Error, MAVEN_FETCH_MESSAGE));
        }
        if gradle_plugin_resolution_failed
            && matches!(failed_folders.as_slice(), [folder] if folder.tool.as_deref() == Some("gradle"))
        {
            return Some((proto::ServerHealth::Error, GRADLE_PLUGIN_MESSAGE));
        }
        return Some((
            proto::ServerHealth::Error,
            "Kotlin project import failed; view language server logs",
        ));
    }
    if params
        .folders
        .iter()
        .any(|folder| folder.status == "BLOCKED")
    {
        let message = if params.folders.iter().any(|folder| {
            folder.status == "BLOCKED" && folder.message.as_deref() == Some("noBuildSystemFound")
        }) {
            "Kotlin project import needs a Maven or Gradle build in the project root"
        } else {
            "Kotlin project import is blocked; view language server logs"
        };
        return Some((proto::ServerHealth::Warning, message));
    }
    if !params.folders.is_empty()
        && params
            .folders
            .iter()
            .all(|folder| folder.status == "SUCCESS")
    {
        return Some((proto::ServerHealth::Ok, "Kotlin project imported"));
    }
    None
}

fn import_log_health(
    params: &ImportLogParams,
    state: &mut ImportHealthState,
) -> Option<(proto::ServerHealth, &'static str)> {
    if params.started {
        state.gradle_import_active = params
            .tool
            .as_deref()
            .is_some_and(|tool| tool.eq_ignore_ascii_case("gradle"))
            || params
                .message
                .as_deref()
                .is_some_and(|message| message.starts_with("Importing with Gradle:"));
        state.maven_import_active = params
            .tool
            .as_deref()
            .is_some_and(|tool| tool.eq_ignore_ascii_case("maven"))
            || params
                .message
                .as_deref()
                .is_some_and(|message| message.starts_with("Importing with Maven:"));
        state.maven_offline_cache_incomplete = false;
        state.maven_dependency_fetch_failed = false;
        state.gradle_plugin_resolution_failed = false;
        state.gradle_plugin_exception_seen = false;
        state.gradle_plugin_artifact_unavailable_seen = false;
    }
    let missing_offline_dependency = params.message.as_deref().is_some_and(|message| {
        state.maven_import_active
            && message.contains("in offline mode")
            && message.contains("has not been downloaded")
    });
    if missing_offline_dependency {
        state.maven_offline_cache_incomplete = true;
    }
    let missing_fetch_dependency = params.message.as_deref().is_some_and(|message| {
        state.maven_import_active && message.contains("Could not transfer artifact")
    });
    if missing_fetch_dependency {
        state.maven_dependency_fetch_failed = true;
    }
    // Kotlin's Gradle importer emits the exception and artifact detail as separate log lines.
    if state.gradle_import_active
        && let Some(message) = params.message.as_deref()
    {
        state.gradle_plugin_exception_seen |= super::gradle_unknown_plugin(message);
        state.gradle_plugin_artifact_unavailable_seen |=
            super::gradle_plugin_artifact_unavailable(message);
    }
    let missing_gradle_plugin = !state.gradle_plugin_resolution_failed
        && state.gradle_plugin_exception_seen
        && state.gradle_plugin_artifact_unavailable_seen;
    if missing_gradle_plugin {
        state.gradle_plugin_resolution_failed = true;
    }
    if params.failed {
        if state.maven_offline_cache_incomplete {
            return Some((proto::ServerHealth::Error, MAVEN_OFFLINE_CACHE_MESSAGE));
        }
        if state.maven_dependency_fetch_failed {
            return Some((proto::ServerHealth::Error, MAVEN_FETCH_MESSAGE));
        }
        if state.gradle_plugin_resolution_failed {
            return Some((proto::ServerHealth::Error, GRADLE_PLUGIN_MESSAGE));
        }
        Some((
            proto::ServerHealth::Error,
            "Kotlin project import failed; view language server logs",
        ))
    } else if params.started {
        Some((
            proto::ServerHealth::Warning,
            "Kotlin project import in progress",
        ))
    } else if missing_offline_dependency {
        Some((proto::ServerHealth::Warning, MAVEN_OFFLINE_CACHE_MESSAGE))
    } else if missing_fetch_dependency {
        Some((proto::ServerHealth::Warning, MAVEN_FETCH_MESSAGE))
    } else if missing_gradle_plugin {
        Some((proto::ServerHealth::Warning, GRADLE_PLUGIN_MESSAGE))
    } else {
        None
    }
}

pub(super) fn fallback_log_health(message: &str) -> Option<(proto::ServerHealth, &'static str)> {
    if message.contains("KotlinFrontEndException") {
        return Some((
            proto::ServerHealth::Error,
            "Kotlin fallback analysis failed; view language server logs or select another server",
        ));
    }
    if message.contains("Could not resolve classpath using Maven")
        && message.contains("Unable to find the 'mvn' command or suitable wrapper")
    {
        return Some((
            proto::ServerHealth::Warning,
            "Kotlin fallback needs Maven on the host or a wrapper beside each module POM",
        ));
    }
    if message.contains("Could not resolve kotlin-stdlib using Gradle") {
        return Some((
            proto::ServerHealth::Error,
            "Kotlin fallback cannot resolve the Gradle standard library; restore the host cache or allow dependency downloads, then restart the server",
        ));
    }
    None
}

fn update_workspace_snapshot(
    lsp_store: &WeakEntity<LspStore>,
    server_id: LanguageServerId,
    event: KotlinImportEvent,
    cx: &mut AsyncApp,
) {
    if let Err(error) = lsp_store.update(cx, |store, cx| {
        store.on_kotlin_workspace_import_event(server_id, event, cx);
    }) {
        log::debug!("Kotlin workspace update arrived after LSP store teardown: {error:#}");
    }
}

pub(super) fn register_notifications(
    lsp_store: WeakEntity<LspStore>,
    language_server: &LanguageServer,
) {
    let server_id = language_server.server_id();
    let name = language_server.name();
    let import_state = Arc::new(Mutex::new(ImportHealthState::default()));
    language_server
        .on_notification::<WorkspaceImportState, _>({
            let lsp_store = lsp_store.clone();
            let name = name.clone();
            let import_state = import_state.clone();
            move |params, cx| {
                let update = import_state_health(&params, &mut import_state.lock());
                if let Some((health, message)) = update {
                    super::emit_jvm_provider_health(
                        &lsp_store, server_id, &name, health, message, cx,
                    );
                }
                if params.phase == "FINISHED" {
                    let successful = !params.folders.is_empty()
                        && params
                            .folders
                            .iter()
                            .all(|folder| folder.status == "SUCCESS");
                    update_workspace_snapshot(
                        &lsp_store,
                        server_id,
                        KotlinImportEvent::Finished { successful },
                        cx,
                    );
                }
            }
        })
        .detach();
    language_server
        .on_notification::<ImportLog, _>({
            move |params, cx| {
                if params.started {
                    update_workspace_snapshot(
                        &lsp_store,
                        server_id,
                        KotlinImportEvent::Started,
                        cx,
                    );
                }
                let update = import_log_health(&params, &mut import_state.lock());
                if let Some((health, message)) = update {
                    super::emit_jvm_provider_health(
                        &lsp_store, server_id, &name, health, message, cx,
                    );
                }
            }
        })
        .detach();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kotlin_gradle_plugin_failure_is_scoped_to_the_current_import() {
        let mut state = ImportHealthState::default();
        let plugin_error: ImportLogParams = serde_json::from_str(
            r#"{"message":"/private/host/build.gradle.kts: org.gradle.api.plugins.UnknownPluginException: Plugin [id: 'org.jetbrains.kotlin.jvm', version: '2.0.21'] was not found; Plugin Repositories (could not resolve plugin artifact 'org.jetbrains.kotlin.jvm:org.jetbrains.kotlin.jvm.gradle.plugin:2.0.21')"}"#,
        )
        .expect("observed Gradle plugin error");
        assert_eq!(import_log_health(&plugin_error, &mut state), None);
        let started: ImportLogParams = serde_json::from_str(
            r#"{"message":"Importing with Gradle: /private/host","tool":"Gradle","started":true}"#,
        )
        .expect("Gradle import start");
        assert!(import_log_health(&started, &mut state).is_some());
        let unrelated: ImportLogParams = serde_json::from_str(
            r#"{"message":"org.gradle.api.plugins.UnknownPluginException: Plugin unavailable"}"#,
        )
        .expect("unrelated plugin failure");
        assert_eq!(import_log_health(&unrelated, &mut state), None);
        assert_eq!(
            import_log_health(&plugin_error, &mut state),
            Some((proto::ServerHealth::Warning, GRADLE_PLUGIN_MESSAGE))
        );
        let failed_log: ImportLogParams = serde_json::from_str(
            r#"{"failed":true,"tool":"Gradle","message":"Failed to import /private/host"}"#,
        )
        .expect("Gradle failed log");
        assert_eq!(
            import_log_health(&failed_log, &mut state),
            Some((proto::ServerHealth::Error, GRADLE_PLUGIN_MESSAGE))
        );
        let failed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"gradle","status":"FAILED"}]}"#,
        )
        .expect("Gradle terminal failure");
        assert_eq!(
            import_state_health(&failed, &mut state),
            Some((proto::ServerHealth::Error, GRADLE_PLUGIN_MESSAGE))
        );
        assert!(!state.gradle_plugin_resolution_failed);
        assert!(!state.gradle_import_active);
        assert!(!state.gradle_plugin_exception_seen);
        assert!(!state.gradle_plugin_artifact_unavailable_seen);
        assert!(!GRADLE_PLUGIN_MESSAGE.contains("/private/host"));
        assert!(import_log_health(&started, &mut state).is_some());
        assert_eq!(
            import_state_health(
                &serde_json::from_str(
                    r#"{"phase":"FINISHED","folders":[{"tool":"gradle","status":"SUCCESS"}]}"#,
                )
                .expect("Gradle recovery"),
                &mut state,
            ),
            Some((proto::ServerHealth::Ok, "Kotlin project imported"))
        );
        assert!(import_log_health(&started, &mut state).is_some());
        assert_eq!(import_log_health(&unrelated, &mut state), None);
        assert!(import_log_health(&started, &mut state).is_some());
        let artifact_only: ImportLogParams = serde_json::from_str(
            r#"{"message":"could not resolve plugin artifact in /private/host"}"#,
        )
        .expect("split artifact log from a new import");
        assert_eq!(import_log_health(&artifact_only, &mut state), None);
        assert!(import_log_health(&plugin_error, &mut state).is_some());
        assert_eq!(
            import_state_health(
                &serde_json::from_str(
                    r#"{"phase":"FINISHED","folders":[{"tool":"gradle","status":"FAILED"},{"tool":"maven","status":"FAILED"}]}"#,
                )
                .expect("mixed import failure"),
                &mut state,
            ),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );
    }

    #[test]
    fn fallback_logs_report_bounded_import_and_analysis_failures() {
        assert_eq!(
            fallback_log_health(
                "Could not resolve classpath using Maven: Unable to find the 'mvn' command or suitable wrapper at /private/host/path"
            ),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin fallback needs Maven on the host or a wrapper beside each module POM",
            ))
        );
        assert_eq!(
            fallback_log_health("KotlinFrontEndException in /private/host/path"),
            Some((
                proto::ServerHealth::Error,
                "Kotlin fallback analysis failed; view language server logs or select another server",
            ))
        );
        assert_eq!(
            fallback_log_health(
                "Could not resolve kotlin-stdlib using Gradle: /private/host/path/.gradle/caches"
            ),
            Some((
                proto::ServerHealth::Error,
                "Kotlin fallback cannot resolve the Gradle standard library; restore the host cache or allow dependency downloads, then restart the server",
            ))
        );
        assert_eq!(
            fallback_log_health("No diagnostics in file:///private/host/path"),
            None
        );
    }

    #[test]
    fn kotlin_import_status_distinguishes_blocked_failed_and_successful_projects() {
        let mut state = ImportHealthState::default();
        let blocked: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"folderUri":"file:///outside/private","status":"BLOCKED","message":"noBuildSystemFound"}]}"#,
        )
        .expect("blocked import notification");
        assert_eq!(
            import_state_health(&blocked, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import needs a Maven or Gradle build in the project root"
            ))
        );
        let failed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"folderUri":"file:///outside/private","tool":"gradle","status":"FAILED","message":"Gradle sync failed"}]}"#,
        )
        .expect("failed import notification");
        assert_eq!(
            import_state_health(&failed, &mut state),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );
        let current: WorkspaceImportStateParams =
            serde_json::from_str(r#"{"phase":"IMPORTING","folders":[{"status":"SUCCESS"}]}"#)
                .expect("in-progress notification");
        assert_eq!(import_state_health(&current, &mut state), None);
        let success: WorkspaceImportStateParams =
            serde_json::from_str(r#"{"phase":"FINISHED","folders":[{"status":"SUCCESS"}]}"#)
                .expect("successful import notification");
        assert_eq!(
            import_state_health(&success, &mut state),
            Some((proto::ServerHealth::Ok, "Kotlin project imported"))
        );
        let failed_log: ImportLogParams = serde_json::from_str(
            r#"{"type":1,"message":"Unable to import a Gradle project","tool":"Gradle","failed":true,"succeeded":false,"started":false}"#,
        )
        .expect("Kotlin LSP import log notification");
        assert_eq!(
            import_log_health(&failed_log, &mut state),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );
        let started_log: ImportLogParams = serde_json::from_str(
            r#"{"type":3,"message":"Importing with Maven: /outside/private","tool":"Maven","failed":false,"succeeded":false,"started":true}"#,
        )
        .expect("Kotlin LSP import start notification");
        assert_eq!(
            import_log_health(&started_log, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        let succeeded_log: ImportLogParams =
            serde_json::from_str(r#"{"failed":false,"succeeded":true,"started":false}"#)
                .expect("Kotlin LSP import success log notification");
        assert_eq!(import_log_health(&succeeded_log, &mut state), None);
    }

    #[test]
    fn kotlin_import_explains_an_incomplete_maven_offline_cache_without_host_paths() {
        let mut state = ImportHealthState::default();
        let started: ImportLogParams = serde_json::from_str(
            r#"{"message":"Importing with Maven: /outside/private","started":true}"#,
        )
        .expect("Maven import start");
        assert_eq!(
            import_log_health(&started, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        let missing_plugin: ImportLogParams = serde_json::from_str(
            r#"{"message":"Cannot access central (https://repo.maven.apache.org/maven2) in offline mode and the artifact org.apache.maven.plugins:maven-install-plugin:jar:3.1.4 has not been downloaded from it before."}"#,
        )
        .expect("Maven offline-cache diagnostic");
        assert_eq!(
            import_log_health(&missing_plugin, &mut state),
            Some((proto::ServerHealth::Warning, MAVEN_OFFLINE_CACHE_MESSAGE))
        );
        let failed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"folderUri":"file:///outside/private","tool":"maven","status":"FAILED","message":"Failed to import Maven project"}]}"#,
        )
        .expect("terminal Maven import failure");
        assert_eq!(
            import_state_health(&failed, &mut state),
            Some((proto::ServerHealth::Error, MAVEN_OFFLINE_CACHE_MESSAGE))
        );
        assert!(!MAVEN_OFFLINE_CACHE_MESSAGE.contains("/outside/private"));
        assert!(!state.maven_offline_cache_incomplete);

        assert_eq!(
            import_log_health(&started, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        let recovered: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"maven","status":"SUCCESS"}]}"#,
        )
        .expect("Maven import recovery");
        assert_eq!(
            import_state_health(&recovered, &mut state),
            Some((proto::ServerHealth::Ok, "Kotlin project imported"))
        );

        state.maven_offline_cache_incomplete = true;
        let mixed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"maven","status":"FAILED"},{"tool":"gradle","status":"FAILED"}]}"#,
        )
        .expect("mixed workspace import failure");
        assert_eq!(
            import_state_health(&mixed, &mut state),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );
    }

    #[test]
    fn kotlin_import_explains_a_denied_maven_fetch_without_host_paths() {
        let mut state = ImportHealthState::default();
        let started: ImportLogParams = serde_json::from_str(
            r#"{"message":"Importing with Maven: /outside/private","started":true}"#,
        )
        .expect("Maven import start");
        assert_eq!(
            import_log_health(&started, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        let missing_artifact: ImportLogParams = serde_json::from_str(
            r#"{"message":"Could not transfer artifact org.apache.maven.plugins:maven-install-plugin:pom:3.1.4 from/to central (https://repo.maven.apache.org/maven2): repo.maven.apache.org: nodename nor servname provided, or not known; /outside/private"}"#,
        )
        .expect("denied-network Maven diagnostic");
        assert_eq!(
            import_log_health(&missing_artifact, &mut state),
            Some((proto::ServerHealth::Warning, MAVEN_FETCH_MESSAGE))
        );
        let failed_log: ImportLogParams = serde_json::from_str(
            r#"{"message":"Failed to import Maven project in /outside/private","failed":true}"#,
        )
        .expect("Maven import failure log");
        assert_eq!(
            import_log_health(&failed_log, &mut state),
            Some((proto::ServerHealth::Error, MAVEN_FETCH_MESSAGE))
        );
        let failed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"folderUri":"file:///outside/private","tool":"maven","status":"FAILED","message":"Failed to import Maven project"}]}"#,
        )
        .expect("terminal Maven import failure");
        assert_eq!(
            import_state_health(&failed, &mut state),
            Some((proto::ServerHealth::Error, MAVEN_FETCH_MESSAGE))
        );
        assert!(!MAVEN_FETCH_MESSAGE.contains("/outside/private"));
        assert!(!state.maven_dependency_fetch_failed);

        assert_eq!(
            import_log_health(&started, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        let recovered: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"maven","status":"SUCCESS"}]}"#,
        )
        .expect("Maven import recovery");
        assert_eq!(
            import_state_health(&recovered, &mut state),
            Some((proto::ServerHealth::Ok, "Kotlin project imported"))
        );

        state.maven_dependency_fetch_failed = true;
        let mixed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"maven","status":"FAILED"},{"tool":"gradle","status":"FAILED"}]}"#,
        )
        .expect("mixed workspace import failure");
        assert_eq!(
            import_state_health(&mixed, &mut state),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );

        let gradle_started: ImportLogParams = serde_json::from_str(
            r#"{"message":"Importing with Gradle: /outside/private","tool":"Gradle","started":true}"#,
        )
        .expect("Gradle import start");
        assert_eq!(
            import_log_health(&gradle_started, &mut state),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin project import in progress"
            ))
        );
        assert_eq!(import_log_health(&missing_artifact, &mut state), None);
        let gradle_failed: WorkspaceImportStateParams = serde_json::from_str(
            r#"{"phase":"FINISHED","folders":[{"tool":"gradle","status":"FAILED"}]}"#,
        )
        .expect("terminal Gradle import failure");
        assert_eq!(
            import_state_health(&gradle_failed, &mut state),
            Some((
                proto::ServerHealth::Error,
                "Kotlin project import failed; view language server logs"
            ))
        );
    }
}
