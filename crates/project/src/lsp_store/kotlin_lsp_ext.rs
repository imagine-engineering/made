use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result, ensure};
use futures::{AsyncReadExt as _, channel::oneshot};
use http_client::{HttpClient, HttpRequestExt as _, RedirectPolicy, Request};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use gpui::{App, AsyncApp, Context, Entity, Task, WeakEntity};
use lsp::{IoKind, LanguageServer, LanguageServerId};
use rpc::{TypedEnvelope, proto};
use semver::Version;

use crate::LspStore;

const LATEST_RELEASE_API: &str = "https://api.github.com/repos/Kotlin/kotlin-lsp/releases/latest";
const MAX_RELEASE_METADATA: u64 = 128 * 1024;
const RELEASE_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) fn standalone_binary_version(binary: &Path) -> Option<Version> {
    if !matches!(
        binary.file_name()?.to_str()?,
        "intellij-server" | "intellij-server.exe"
    ) {
        return None;
    }
    let bin = binary.parent()?;
    if bin.file_name()?.to_str()? != "bin" {
        return None;
    }
    let directory = bin.parent()?.file_name()?.to_str()?;
    let version = directory
        .strip_prefix("kotlin-server-")
        .or_else(|| directory.strip_prefix("kotlin-lsp-"))?;
    super::kotlin_lsp_version(version)
}

#[derive(Deserialize)]
struct LatestRelease {
    tag_name: String,
    body: String,
}

fn standalone_release(release: LatestRelease, os: &str, arch: &str) -> Option<Version> {
    let version = super::kotlin_lsp_version(release.tag_name.strip_prefix("kotlin-lsp/v")?)?;
    let suffix = match arch {
        "x86_64" => "",
        "aarch64" => "-aarch64",
        _ => return None,
    };
    let extension = match os {
        "macos" => "sit",
        "linux" => "tar.gz",
        "windows" => "win.zip",
        _ => return None,
    };
    let url = format!(
        "https://download.jetbrains.com/language-server/kotlin-server/{version}/kotlin-server-{version}{suffix}.{extension}"
    );
    let standalone = release
        .body
        .split_once("**Standalone Kotlin LSP Archive**")?
        .1;
    standalone
        .lines()
        .take_while(|line| !line.starts_with('#'))
        .any(|line| {
            line.contains(&format!("]({url})")) && line.contains(&format!("]({url}.sha256)"))
        })
        .then_some(version)
}

async fn fetch_standalone_release(http: Arc<dyn HttpClient>) -> Result<Option<Version>> {
    let request = Request::get(LATEST_RELEASE_API)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "Made-Kotlin-LSP")
        .follow_redirects(RedirectPolicy::NoFollow)
        .timeout(RELEASE_TIMEOUT)
        .body(Default::default())?;
    let mut response = http
        .send(request)
        .await
        .context("Could not reach the official Kotlin LSP release from the project host")?;
    ensure!(
        response.status().is_success(),
        "The Kotlin LSP release check returned HTTP {}",
        response.status().as_u16()
    );
    let mut body = Vec::new();
    response
        .body_mut()
        .take(MAX_RELEASE_METADATA + 1)
        .read_to_end(&mut body)
        .await?;
    ensure!(
        body.len() as u64 <= MAX_RELEASE_METADATA,
        "Kotlin LSP release metadata exceeded the permitted size"
    );
    let release = serde_json::from_slice(&body)
        .context("Could not read the official Kotlin LSP release metadata")?;
    Ok(standalone_release(
        release,
        std::env::consts::OS,
        std::env::consts::ARCH,
    ))
}

impl LspStore {
    fn ensure_kotlin_release_check_scope(&self, cx: &App) -> Result<()> {
        ensure!(
            !self.isolated_recipe_test,
            "Kotlin LSP release checks are disabled in this execution profile"
        );
        ensure!(
            crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx).is_some()
                && !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &self.worktree_store,
                    cx
                )
                && self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .next()
                    .is_some(),
            "Trust the project before checking for a Kotlin LSP release"
        );
        Ok(())
    }

    pub(super) fn remote_kotlin_release_connection(&self) -> Option<(&rpc::AnyProtoClient, u64)> {
        let remote = self.as_remote().filter(|remote| {
            remote.upstream_project_id == proto::REMOTE_SERVER_PROJECT_ID
                && remote.jvm_consumers.connected
        })?;
        let client = remote
            .upstream_client
            .as_ref()
            .filter(|client| !client.is_via_collab())?;
        Some((client, remote.jvm_consumers.generation))
    }

    pub(super) fn check_kotlin_lsp_release_on_host(
        &self,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<Option<Version>>>> {
        self.ensure_kotlin_release_check_scope(cx)?;
        if let Some(local) = self.as_local() {
            let http = local.http_client.clone();
            return Ok(cx.spawn(async move |store, cx| {
                let release = fetch_standalone_release(http).await?;
                store.read_with(cx, |store, cx| store.ensure_kotlin_release_check_scope(cx))??;
                Ok(release)
            }));
        }
        let (client, generation) = self
            .remote_kotlin_release_connection()
            .context("Connect directly to the Kotlin project host to check for a release")?;
        let client = client.clone();
        Ok(cx.spawn(async move |store, cx| {
            let request = client.request(proto::CheckKotlinLspRelease {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
            });
            let timeout = cx
                .background_executor()
                .timer(RELEASE_TIMEOUT + Duration::from_secs(3));
            let response = match futures::future::select(Box::pin(request), Box::pin(timeout)).await
            {
                futures::future::Either::Left((response, _)) => response?,
                futures::future::Either::Right(_) => {
                    anyhow::bail!("The project host's Kotlin LSP release check timed out")
                }
            };
            store.read_with(cx, |store, cx| {
                store.ensure_kotlin_release_check_scope(cx)?;
                let (current, current_generation) =
                    store.remote_kotlin_release_connection().context(
                        "Connect directly to the Kotlin project host to check for a release",
                    )?;
                ensure!(
                    current.is_same_instance(&client) && generation == current_generation,
                    "The Kotlin project host changed during the release check"
                );
                Ok::<_, anyhow::Error>(())
            })??;
            response
                .version
                .as_deref()
                .map(|version| {
                    super::kotlin_lsp_version(version)
                        .context("The project host returned an invalid Kotlin LSP release version")
                })
                .transpose()
        }))
    }

    pub(super) async fn handle_check_kotlin_lsp_release(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CheckKotlinLspRelease>,
        cx: AsyncApp,
    ) -> Result<proto::KotlinLspReleaseResponse> {
        ensure!(
            envelope.original_sender_id.is_none(),
            "Kotlin LSP release checks require a direct project host connection"
        );
        let scope = |store: &Self, cx: &App| {
            store
                .ensure_host_jvm_consumer_scope(envelope.payload.project_id, envelope.sender_id)?;
            store.ensure_kotlin_release_check_scope(cx)
        };
        let http = this.read_with(&cx, |store, cx| {
            scope(store, cx)?;
            Ok::<_, anyhow::Error>(
                store
                    .as_local()
                    .context("The Kotlin project host is unavailable")?
                    .http_client
                    .clone(),
            )
        })?;
        let version = fetch_standalone_release(http).await?;
        this.read_with(&cx, scope)?;
        Ok(proto::KotlinLspReleaseResponse {
            version: version.map(|version| version.to_string()),
        })
    }
}

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

const FALLBACK_MEMORY_MESSAGE: &str = "Kotlin fallback ran out of memory. Review its JVM memory settings on the project host, then restart it or select another Kotlin server.";

pub(super) fn register_fallback_memory_health(
    lsp_store: WeakEntity<LspStore>,
    language_server: &LanguageServer,
    cx: &AsyncApp,
) -> Arc<AtomicBool> {
    let server_id = language_server.server_id();
    let name = language_server.name();
    let memory_failed = Arc::new(AtomicBool::new(false));
    let (memory_signal, received) = oneshot::channel();
    let mut memory_signal = Some(memory_signal);
    language_server
        .on_io({
            let memory_failed = memory_failed.clone();
            move |kind, line| {
                if matches!(kind, IoKind::StdErr)
                    && fallback_memory_failure(line)
                    && !memory_failed.swap(true, Ordering::Relaxed)
                    && let Some(memory_signal) = memory_signal.take()
                    && memory_signal.send(()).is_err()
                {
                    log::debug!(
                        "Kotlin fallback memory failure arrived after health monitoring ended"
                    );
                }
            }
        })
        .detach();
    cx.spawn(async move |cx| {
        if received.await.is_ok() {
            super::emit_jvm_provider_health(
                &lsp_store,
                server_id,
                &name,
                proto::ServerHealth::Error,
                FALLBACK_MEMORY_MESSAGE,
                cx,
            );
        }
    })
    .detach();
    memory_failed
}

fn fallback_memory_failure(line: &str) -> bool {
    let line = line.trim();
    let line = [
        "SEVERE: Internal error:",
        "java.util.concurrent.CompletionException:",
        "Caused by:",
    ]
    .into_iter()
    .fold(line, |line, prefix| {
        line.strip_prefix(prefix)
            .map(str::trim_start)
            .unwrap_or(line)
    });
    let exception = if let Some(exception) = line.strip_prefix("Exception: ") {
        exception
    } else if let Some(thread) = line.strip_prefix("Exception in thread \"") {
        let Some((_, exception)) = thread.split_once("\" ") else {
            return false;
        };
        exception
    } else {
        line
    };
    exception
        .strip_prefix("java.lang.OutOfMemoryError")
        .is_some_and(|details| {
            details.is_empty() || details.starts_with(':') || details.starts_with(" thrown from ")
        })
}

pub(super) fn fallback_log_health(
    message: &str,
    memory_failed: &AtomicBool,
) -> Option<(proto::ServerHealth, &'static str)> {
    if memory_failed.load(Ordering::Relaxed) {
        return None;
    }
    if message.lines().any(fallback_memory_failure) {
        return (!memory_failed.swap(true, Ordering::Relaxed))
            .then_some((proto::ServerHealth::Error, FALLBACK_MEMORY_MESSAGE));
    }
    if message.contains(
        "Internal error: java.lang.NullPointerException: null cannot be cast to non-null type V of org.jetbrains.kotlin.util.slicedMap.OpenAddressLinearProbingHashTable",
    ) {
        return Some((
            proto::ServerHealth::Error,
            "Kotlin fallback analysis failed. Restart the server for this project, view its logs, or select another Kotlin server.",
        ));
    }
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
    #[test]
    fn standalone_release_requires_the_exact_archive_for_the_host() {
        let tag = "kotlin-lsp/v271.2085.0";
        let url = "https://download.jetbrains.com/language-server/kotlin-server/271.2085.0/kotlin-server-271.2085.0-aarch64.sit";
        let links =
            format!("  * [Download for macOS-arm64]({url}) | [SHA-256 checksum]({url}.sha256)");
        let release = |body| super::LatestRelease {
            tag_name: tag.into(),
            body,
        };
        let standalone = || {
            release(format!(
                "### v271.2085.0\r\n- **Standalone Kotlin LSP Archive**\r\n{links}"
            ))
        };
        assert_eq!(
            super::standalone_release(standalone(), "macos", "aarch64")
                .map(|v| v.to_string())
                .as_deref(),
            Some("271.2085.0")
        );
        for (os, arch) in [
            ("macos", "x86_64"),
            ("linux", "aarch64"),
            ("windows", "aarch64"),
        ] {
            assert!(super::standalone_release(standalone(), os, arch).is_none());
        }
        assert!(super::standalone_release(release(links.clone()), "macos", "aarch64").is_none());
        assert!(
            super::standalone_release(
                release(format!(
                    "**Standalone Kotlin LSP Archive**\n### Changelog\n{links}"
                )),
                "macos",
                "aarch64"
            )
            .is_none()
        );
    }

    #[test]
    fn expired_binary_version_accepts_a_standard_standalone_layout() {
        use std::path::Path;
        for path in [
            "/local/kotlin-server-263.4702.0/bin/intellij-server",
            "/local/kotlin-lsp-263.4702.0/bin/intellij-server.exe",
        ] {
            assert_eq!(
                super::standalone_binary_version(Path::new(path))
                    .map(|v| v.to_string())
                    .as_deref(),
                Some("263.4702.0")
            );
        }
        assert!(
            super::standalone_binary_version(Path::new("/custom/kotlin/bin/intellij-server"))
                .is_none()
        );
        assert!(
            super::standalone_binary_version(Path::new(
                "/custom/kotlin-server-263.4702.0/bin/wrapper"
            ))
            .is_none()
        );
    }

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
        let memory_failed = AtomicBool::new(false);
        assert_eq!(
            fallback_log_health(
                "Could not resolve classpath using Maven: Unable to find the 'mvn' command or suitable wrapper at /private/host/path",
                &memory_failed
            ),
            Some((
                proto::ServerHealth::Warning,
                "Kotlin fallback needs Maven on the host or a wrapper beside each module POM",
            ))
        );
        assert_eq!(
            fallback_log_health(
                "KotlinFrontEndException in /private/host/path",
                &memory_failed
            ),
            Some((
                proto::ServerHealth::Error,
                "Kotlin fallback analysis failed; view language server logs or select another server",
            ))
        );
        assert_eq!(
            fallback_log_health(
                "Could not resolve kotlin-stdlib using Gradle: /private/host/path/.gradle/caches",
                &memory_failed
            ),
            Some((
                proto::ServerHealth::Error,
                "Kotlin fallback cannot resolve the Gradle standard library; restore the host cache or allow dependency downloads, then restart the server",
            ))
        );
        assert_eq!(
            fallback_log_health(
                "No diagnostics in file:///private/host/path",
                &memory_failed
            ),
            None
        );
    }

    #[test]
    fn fallback_reported_memory_failure_takes_precedence_over_analysis_errors() {
        for message in [
            "SEVERE: Internal error: java.lang.OutOfMemoryError: Java heap space",
            "KotlinFrontEndException at /private/root\nCaused by: java.lang.OutOfMemoryError: Java heap space",
        ] {
            let memory_failed = AtomicBool::new(false);
            assert_eq!(
                fallback_log_health(message, &memory_failed),
                Some((proto::ServerHealth::Error, FALLBACK_MEMORY_MESSAGE))
            );
            assert_eq!(fallback_log_health(message, &memory_failed), None);
            assert_eq!(
                fallback_log_health("KotlinFrontEndException at /private/root", &memory_failed),
                None
            );
        }
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
