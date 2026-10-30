use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use gpui::WeakEntity;
use lsp::{LanguageServer, LanguageServerId};
use rpc::proto;

use crate::LspStore;

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
