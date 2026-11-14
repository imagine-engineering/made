use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    io::Read as _,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, ensure};
use client::proto;
use gpui::BackgroundExecutor;
use lsp::{LanguageServer, LanguageServerBinary, LanguageServerId};
use path::PathStyle;
use proto::Message as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use util::{
    paths::{self, SanitizedPath},
    rel_path::RelPath,
};
use worktree::{Snapshot, WorktreeId};

use crate::ProjectPath;

const MAX_JDTLS_MODULES: usize = 128;
const MAX_JDTLS_SOURCE_ROOTS: usize = 256;
const MAX_JDTLS_CLASSPATH_ENTRIES: usize = 512;
const MAX_KOTLIN_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_KOTLIN_LIBRARIES: usize = 4096;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct JvmBuildSelectionIntent<'a> {
    pub maven: Option<MavenSelectionIntent<'a>>,
    pub gradle: Option<GradleSelectionIntent<'a>>,
}

const MAX_MAVEN_POM_MASKS: usize = 128;
const MAX_MAVEN_POM_MASK_BYTES: usize = 4096;
const MAX_MAVEN_POM_MASK_TOTAL_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct MavenPomMaskContract {
    patterns: Vec<String>,
    expression: Option<regex::Regex>,
}

impl MavenPomMaskContract {
    pub fn new(patterns: &[String]) -> Result<Self> {
        validate_maven_pom_masks(patterns)?;
        let expression = if patterns.is_empty() {
            None
        } else {
            let alternatives = patterns
                .iter()
                .map(|pattern| {
                    // The pinned IDEA converter discards empty backslash-delimited pieces.
                    pattern
                        .split('\\')
                        .filter(|piece| !piece.is_empty())
                        .map(|piece| {
                            regex::escape(piece)
                                .replace(r"\*", ".*")
                                .replace(r"\?", ".")
                        })
                        .collect::<Vec<_>>()
                        .join(r"\\")
                })
                .collect::<Vec<_>>()
                .join("|");
            Some(regex::RegexBuilder::new(&format!(r"\A(?:{alternatives})\z"))
                .size_limit(8 * 1024 * 1024)
                .dfa_size_limit(256 * 1024)
                .build()
                .map_err(|_| anyhow::anyhow!("Maven ignored_pom_patterns exceed the matcher resource bound; reduce patterns and refresh"))?)
        };
        Ok(Self {
            patterns: patterns.to_vec(),
            expression,
        })
    }

    pub fn from_request(value: &Value) -> Result<Self> {
        let failure = || {
            anyhow::anyhow!(
                "Maven ignored_pom_patterns contract is unsupported or malformed; check provider setup and refresh"
            )
        };
        let object = value.as_object().ok_or_else(failure)?;
        ensure!(
            object.len() == 4
                && object.get("version").and_then(Value::as_u64) == Some(1)
                && object.get("syntax").and_then(Value::as_str) == Some("idea-mask-v1")
                && object.get("pathFormat").and_then(Value::as_str) == Some("portable-absolute-v1"),
            "{}",
            failure()
        );
        let patterns = object
            .get("patterns")
            .and_then(Value::as_array)
            .ok_or_else(failure)?;
        ensure!(patterns.len() <= MAX_MAVEN_POM_MASKS, "{}", failure());
        let mut total = 0usize;
        let mut admitted = Vec::with_capacity(patterns.len());
        for value in patterns {
            let pattern = value.as_str().ok_or_else(failure)?;
            total = total.checked_add(pattern.len()).ok_or_else(failure)?;
            ensure!(
                pattern.len() <= MAX_MAVEN_POM_MASK_BYTES
                    && total <= MAX_MAVEN_POM_MASK_TOTAL_BYTES,
                "{}",
                failure()
            );
            admitted.push(pattern.to_owned());
        }
        Self::new(&admitted)
    }

    pub fn to_request(&self) -> Value {
        json!({"version": 1, "syntax": "idea-mask-v1", "pathFormat": "portable-absolute-v1", "patterns": self.patterns})
    }

    pub fn matches_portable_host_path(&self, path: &str, style: PathStyle) -> Result<bool> {
        let failure = || {
            anyhow::anyhow!(
                "Maven POM path normalization is unavailable; refresh from the authoritative project host"
            )
        };
        ensure!(
            !path.is_empty()
                && path.len() <= MAX_MAVEN_POM_MASK_BYTES
                && !path.chars().any(invalid_maven_mask_character)
                && !path.ends_with('/'),
            "{}",
            failure()
        );
        let relative = match style {
            PathStyle::Unix => path
                .strip_prefix('/')
                .filter(|path| !path.starts_with('/'))
                .ok_or_else(failure)?,
            PathStyle::Windows => {
                ensure!(!path.contains('\\'), "{}", failure());
                if let Some(relative) = path.strip_prefix("//") {
                    ensure!(relative.split('/').count() >= 3, "{}", failure());
                    relative
                } else {
                    ensure!(
                        path.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
                            && path.get(1..3) == Some(":/"),
                        "{}",
                        failure()
                    );
                    path.get(3..).ok_or_else(failure)?
                }
            }
        };
        ensure!(
            relative
                .split('/')
                .all(|piece| !piece.is_empty() && piece != "." && piece != ".."),
            "{}",
            failure()
        );
        if style == PathStyle::Windows {
            ensure!(
                relative.split('/').all(|piece| {
                    !piece.chars().any(|character| {
                        matches!(character, ':' | '*' | '?' | '"' | '<' | '>' | '|')
                    }) && !piece.ends_with(['.', ' '])
                }),
                "{}",
                failure()
            );
        }
        Ok(self
            .expression
            .as_ref()
            .is_some_and(|expression| expression.is_match(path)))
    }
}

fn invalid_maven_mask_character(character: char) -> bool {
    character.is_control() || matches!(character, '\u{2028}' | '\u{2029}')
}

fn validate_maven_pom_masks(patterns: &[String]) -> Result<()> {
    let failure = || {
        anyhow::anyhow!(
            "Maven ignored_pom_patterns must be unique nonempty bounded masks without control characters; reduce or correct patterns and refresh"
        )
    };
    ensure!(patterns.len() <= MAX_MAVEN_POM_MASKS, "{}", failure());
    let mut total = 0usize;
    let mut distinct = HashSet::new();
    for pattern in patterns {
        total = total.checked_add(pattern.len()).ok_or_else(failure)?;
        ensure!(
            !pattern.is_empty()
                && pattern.len() <= MAX_MAVEN_POM_MASK_BYTES
                && total <= MAX_MAVEN_POM_MASK_TOTAL_BYTES
                && !pattern.chars().any(invalid_maven_mask_character)
                && distinct.insert(pattern.as_str()),
            "{}",
            failure()
        );
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub struct MavenSelectionIntent<'a> {
    pub ignored_pom_patterns: &'a [String],
    pub selected_poms: Option<&'a [String]>,
    pub ignored_projects: &'a [String],
}

#[derive(Debug, PartialEq, Eq)]
pub struct GradleSelectionIntent<'a> {
    pub selected_build_files: Option<&'a [String]>,
    pub ignored_projects: &'a [String],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SavedGradleSelection {
    pub selected_roots: Vec<ProjectPath>,
    pub ignored_projects: Vec<ProjectPath>,
    pub excluded_projects: Vec<ProjectPath>,
}

impl SavedGradleSelection {
    pub(crate) fn new(
        intent: &GradleSelectionIntent<'_>,
        worktree_id: WorktreeId,
        projects: &[JdtlsProject],
    ) -> Result<Option<Self>> {
        validate_saved_build_paths(
            intent.selected_build_files,
            128,
            "Gradle",
            Some(&[
                "settings.gradle",
                "settings.gradle.kts",
                "build.gradle",
                "build.gradle.kts",
            ]),
        )?;
        validate_saved_build_paths(Some(intent.ignored_projects), 128, "Gradle", None)?;
        let Some(selected_build_files) = intent.selected_build_files else {
            ensure!(
                intent.ignored_projects.is_empty(),
                "Gradle ignore requires explicit selected_build_files until provider default-root discovery is available; select build roots or remove the ignore override"
            );
            return Ok(None);
        };
        ensure!(
            projects.len() <= MAX_JDTLS_MODULES,
            "Gradle ownership exceeds the supported bound; refresh and reduce project selection before applying saved settings"
        );
        let project_path = |text: &str| -> Result<ProjectPath> {
            Ok(ProjectPath {
                worktree_id,
                path: RelPath::from_unix_str(text)?.into(),
            })
        };
        let ignored_projects = intent
            .ignored_projects
            .iter()
            .map(|directory| project_path(directory))
            .collect::<Result<Vec<_>>>()?;
        let is_ignored = |path: &ProjectPath| {
            path.worktree_id == worktree_id
                && ignored_projects
                    .iter()
                    .any(|directory| path.path.starts_with(&directory.path))
        };
        let mut selected_roots = Vec::new();
        let mut directories = HashSet::new();
        for build_file in selected_build_files {
            let build_file = project_path(build_file)?;
            let root = ProjectPath {
                worktree_id,
                path: build_file.path.parent().unwrap_or(RelPath::empty()).into(),
            };
            if !is_ignored(&root) && directories.insert(root.clone()) {
                selected_roots.push(root);
            }
        }
        let mut identities = HashSet::new();
        let mut excluded_projects = Vec::new();
        for project in projects {
            if !is_ignored(&project.module_root) || !project.membership.is_imported() {
                continue;
            }
            ensure!(
                identities.insert(project.module_root.clone())
                    && project.imported_build_kind != JdtlsImportedBuildKind::Unknown,
                "Ignored Gradle project ownership is unavailable or ambiguous; check provider setup and refresh before applying saved settings"
            );
            if project.imported_build_kind == JdtlsImportedBuildKind::Gradle {
                excluded_projects.push(project.module_root.clone());
            }
        }
        Ok(Some(Self {
            selected_roots,
            ignored_projects,
            excluded_projects,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SavedMavenSelection {
    pub selected_poms: Vec<ProjectPath>,
    pub ignored_projects: Vec<ProjectPath>,
    pub unlink_projects: Vec<ProjectPath>,
}

impl SavedMavenSelection {
    pub(crate) fn new(
        intent: &MavenSelectionIntent<'_>,
        worktree_id: WorktreeId,
        projects: &[JdtlsProject],
    ) -> Result<Option<Self>> {
        Self::with_default_snapshot(intent, worktree_id, projects, None)
    }

    pub(crate) fn with_default_snapshot(
        intent: &MavenSelectionIntent<'_>,
        worktree_id: WorktreeId,
        projects: &[JdtlsProject],
        entries: Option<&Snapshot>,
    ) -> Result<Option<Self>> {
        validate_saved_build_paths(intent.selected_poms, 32, "Maven", Some(&["pom.xml"]))?;
        validate_maven_pom_masks(intent.ignored_pom_patterns)?;
        validate_saved_build_paths(Some(intent.ignored_projects), 128, "Maven", None)?;
        let default_poms = if intent.selected_poms.is_none()
            && (!intent.ignored_projects.is_empty() || !intent.ignored_pom_patterns.is_empty())
        {
            let entries = entries.context(
                "Maven ignore requires explicit selected_poms until provider default-root discovery is available; select build roots or remove the ignore override"
            )?;
            Some(
                default_maven_import_inputs(entries, worktree_id)?
                    .into_iter()
                    .map(|pom| pom.path.as_unix_str().to_owned())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let Some(selected_poms) = intent.selected_poms.or(default_poms.as_deref()) else {
            return Ok(None);
        };
        validate_saved_build_paths(Some(selected_poms), 32, "Maven", Some(&["pom.xml"]))?;
        ensure!(
            projects.len() <= MAX_JDTLS_MODULES,
            "Maven ownership exceeds the supported bound; refresh and reduce project selection before applying saved settings"
        );
        let project_path = |text: &str| -> Result<ProjectPath> {
            Ok(ProjectPath {
                worktree_id,
                path: RelPath::from_unix_str(text)?.into(),
            })
        };
        let ignored_projects = intent
            .ignored_projects
            .iter()
            .map(|directory| project_path(directory))
            .collect::<Result<Vec<_>>>()?;
        let is_ignored = |path: &ProjectPath| {
            path.worktree_id == worktree_id
                && ignored_projects
                    .iter()
                    .any(|directory| path.path.starts_with(&directory.path))
        };
        let selected_poms = selected_poms
            .iter()
            .map(|pom| project_path(pom))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|pom| !is_ignored(pom))
            .collect();
        let mut identities = HashSet::new();
        let mut unlink_projects = Vec::new();
        for project in projects {
            if !is_ignored(&project.module_root) || !project.membership.is_imported() {
                continue;
            }
            ensure!(
                identities.insert(project.module_root.clone())
                    && project.imported_build_kind != JdtlsImportedBuildKind::Unknown,
                "Ignored Maven project ownership is unavailable or ambiguous; check provider setup and refresh before applying saved settings"
            );
            if project.imported_build_kind == JdtlsImportedBuildKind::Maven {
                unlink_projects.push(project.module_root.clone());
            }
        }
        Ok(Some(Self {
            selected_poms,
            ignored_projects,
            unlink_projects,
        }))
    }
}

impl<'a> JvmBuildSelectionIntent<'a> {
    pub fn from_settings(settings: Option<&'a settings::JvmBuildSettingsContent>) -> Result<Self> {
        let Some(settings) = settings else {
            return Ok(Self::default());
        };
        let maven = settings
            .maven
            .as_ref()
            .map(|settings| {
                validate_saved_build_paths(
                    settings.selected_poms.as_deref(),
                    32,
                    "Maven",
                    Some(&["pom.xml"]),
                )?;
                validate_saved_build_paths(
                    settings.ignored_projects.as_deref(),
                    128,
                    "Maven",
                    None,
                )?;
                validate_maven_pom_masks(
                    settings.ignored_pom_patterns.as_deref().unwrap_or_default(),
                )?;
                Ok::<_, anyhow::Error>(MavenSelectionIntent {
                    ignored_pom_patterns: settings
                        .ignored_pom_patterns
                        .as_deref()
                        .unwrap_or_default(),
                    selected_poms: settings.selected_poms.as_deref(),
                    ignored_projects: settings.ignored_projects.as_deref().unwrap_or_default(),
                })
            })
            .transpose()?;
        let gradle = settings
            .gradle
            .as_ref()
            .map(|settings| {
                validate_saved_build_paths(
                    settings.selected_build_files.as_deref(),
                    128,
                    "Gradle",
                    Some(&[
                        "settings.gradle",
                        "settings.gradle.kts",
                        "build.gradle",
                        "build.gradle.kts",
                    ]),
                )?;
                validate_saved_build_paths(
                    settings.ignored_projects.as_deref(),
                    128,
                    "Gradle",
                    None,
                )?;
                Ok::<_, anyhow::Error>(GradleSelectionIntent {
                    selected_build_files: settings.selected_build_files.as_deref(),
                    ignored_projects: settings.ignored_projects.as_deref().unwrap_or_default(),
                })
            })
            .transpose()?;
        Ok(Self { maven, gradle })
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct JvmStartupImportControls {
    maven: bool,
    gradle: bool,
}

impl JvmBuildSelectionIntent<'_> {
    pub(crate) fn startup_import_controls(&self) -> JvmStartupImportControls {
        JvmStartupImportControls {
            maven: self.maven.as_ref().is_some_and(|intent| {
                intent.selected_poms.is_some()
                    || !intent.ignored_projects.is_empty()
                    || !intent.ignored_pom_patterns.is_empty()
            }),
            gradle: self.gradle.as_ref().is_some_and(|intent| {
                intent.selected_build_files.is_some() || !intent.ignored_projects.is_empty()
            }),
        }
    }
}

const JVM_STARTUP_IMPORT_CONFLICT: &str = "Saved JVM build selection conflicts with advanced JDTLS import settings. Remove projectConfigurations and enabled importer overrides from lsp.jdtls initialization_options/settings, or remove jvm_builds selection; then restart the language server. Existing imported projects still require reconciliation.";

impl JvmStartupImportControls {
    fn controlled_providers(self) -> impl Iterator<Item = &'static str> {
        [(self.maven, "maven"), (self.gradle, "gradle")]
            .into_iter()
            .filter_map(|(controlled, provider)| controlled.then_some(provider))
    }

    pub(crate) fn validate_overrides(
        self,
        settings: &crate::project_settings::LspSettings,
    ) -> Result<()> {
        if self == Self::default() {
            return Ok(());
        }
        if let Some(options) = &settings.initialization_options {
            ensure!(
                options.is_object() && options.get("projectConfigurations").is_none(),
                JVM_STARTUP_IMPORT_CONFLICT
            );
            if let Some(configuration) = options.get("settings") {
                self.validate_configuration(configuration)?;
            }
        }
        if let Some(configuration) = &settings.settings {
            self.validate_configuration(configuration)?;
        }
        Ok(())
    }

    fn validate_configuration(self, configuration: &Value) -> Result<()> {
        ensure!(configuration.is_object(), JVM_STARTUP_IMPORT_CONFLICT);
        for provider in self.controlled_providers() {
            validate_import_setting_aliases(
                configuration,
                &["java", "import", provider, "enabled"],
            )?;
        }
        Ok(())
    }

    pub(crate) fn apply_initialization(self, options: &mut Option<Value>) -> Result<()> {
        if self == Self::default() {
            return Ok(());
        }
        let options = options.get_or_insert_with(|| json!({}));
        ensure!(
            options.is_object() && options.get("projectConfigurations").is_none(),
            JVM_STARTUP_IMPORT_CONFLICT
        );
        let object = options
            .as_object_mut()
            .context(JVM_STARTUP_IMPORT_CONFLICT)?;
        let configuration = object.entry("settings").or_insert_with(|| json!({}));
        self.apply_configuration(configuration)?;
        if self.maven && self.gradle {
            // Importer enabled flags do not suppress the SDK's Eclipse metadata importer.
            object.insert("projectConfigurations".to_owned(), json!([]));
        }
        Ok(())
    }

    pub(crate) fn apply_configuration(self, configuration: &mut Value) -> Result<()> {
        if self == Self::default() {
            return Ok(());
        }
        if configuration.is_null() {
            *configuration = json!({});
        }
        let object = configuration
            .as_object_mut()
            .context(JVM_STARTUP_IMPORT_CONFLICT)?;
        for provider in self.controlled_providers() {
            suppress_import_setting_aliases(object, &["java", "import", provider, "enabled"])?;
            let java = object
                .entry("java")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .context(JVM_STARTUP_IMPORT_CONFLICT)?;
            let import = java
                .entry("import")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .context(JVM_STARTUP_IMPORT_CONFLICT)?;
            let provider = import
                .entry(provider)
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .context(JVM_STARTUP_IMPORT_CONFLICT)?;
            provider.insert("enabled".to_owned(), Value::Bool(false));
        }
        Ok(())
    }
}

fn validate_import_setting_aliases(configuration: &Value, components: &[&str]) -> Result<()> {
    ensure!(configuration.is_object(), JVM_STARTUP_IMPORT_CONFLICT);
    for count in 1..=components.len() {
        let key = components
            .get(..count)
            .context(JVM_STARTUP_IMPORT_CONFLICT)?
            .join(".");
        if let Some(value) = configuration.get(&key) {
            if count == components.len() {
                ensure!(
                    value == &Value::Bool(false)
                        || value
                            .as_str()
                            .is_some_and(|value| value.eq_ignore_ascii_case("false")),
                    JVM_STARTUP_IMPORT_CONFLICT
                );
            } else {
                let remaining = components
                    .get(count..)
                    .context(JVM_STARTUP_IMPORT_CONFLICT)?;
                validate_import_setting_aliases(value, remaining)?;
            }
        }
    }
    Ok(())
}

fn suppress_import_setting_aliases(
    configuration: &mut serde_json::Map<String, Value>,
    components: &[&str],
) -> Result<()> {
    for count in 1..=components.len() {
        let key = components
            .get(..count)
            .context(JVM_STARTUP_IMPORT_CONFLICT)?
            .join(".");
        if let Some(value) = configuration.get_mut(&key) {
            if count == components.len() {
                *value = Value::Bool(false);
            } else {
                let remaining = components
                    .get(count..)
                    .context(JVM_STARTUP_IMPORT_CONFLICT)?;
                suppress_import_setting_aliases(
                    value.as_object_mut().context(JVM_STARTUP_IMPORT_CONFLICT)?,
                    remaining,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_saved_build_paths(
    paths: Option<&[String]>,
    limit: usize,
    provider: &str,
    build_files: Option<&[&str]>,
) -> Result<()> {
    let Some(paths) = paths else {
        return Ok(());
    };
    let mut unique = BTreeSet::new();
    let valid = paths.len() <= limit
        && paths.iter().all(|path| {
            path.len() <= 4096
                && !path.chars().any(|character| {
                    character.is_control()
                        || matches!(character, '\\' | ':' | '*' | '?' | '[' | ']' | '{' | '}')
                })
                && (path.is_empty() && build_files.is_none()
                    || !path.is_empty()
                        && path
                            .split('/')
                            .all(|component| !matches!(component, "" | "." | "..")))
                && build_files.is_none_or(|names| {
                    path.rsplit('/')
                        .next()
                        .is_some_and(|name| names.contains(&name))
                })
                && unique.insert(path)
        });
    ensure!(
        valid,
        "Saved {provider} build selection is invalid; review jvm_builds settings for unique bounded worktree-relative identities before importing"
    );
    Ok(())
}

const MAX_DEFAULT_JVM_DISCOVERY_ENTRIES: usize = 65_536;

pub(crate) fn default_jvm_build_inputs(
    snapshot: &Snapshot,
    worktree_id: WorktreeId,
    settings: Option<&settings::JvmBuildSettingsContent>,
) -> Result<Vec<ProjectPath>> {
    ensure!(
        snapshot.id() == worktree_id,
        "Default JVM build inputs belong to a different project folder; reopen the current host project"
    );
    let intent = JvmBuildSelectionIntent::from_settings(settings)?;
    let providers = [
        (
            "Maven",
            &["pom.xml"][..],
            intent
                .maven
                .as_ref()
                .and_then(|selection| selection.selected_poms),
            intent
                .maven
                .as_ref()
                .map(|selection| selection.ignored_projects)
                .unwrap_or_default(),
        ),
        (
            "Gradle",
            &[
                "settings.gradle",
                "settings.gradle.kts",
                "build.gradle",
                "build.gradle.kts",
            ][..],
            intent
                .gradle
                .as_ref()
                .and_then(|selection| selection.selected_build_files),
            intent
                .gradle
                .as_ref()
                .map(|selection| selection.ignored_projects)
                .unwrap_or_default(),
        ),
    ];
    for (provider, _, selected, ignored) in &providers {
        ensure!(
            *provider == "Maven" || selected.is_some() || ignored.is_empty(),
            "{provider} ignore requires explicit build-file selection until provider default-root import discovery is available; select roots or remove the ignore override"
        );
    }
    ensure!(
        providers
            .iter()
            .all(|(_, _, selected, _)| selected.is_some())
            || snapshot.visible_entry_count() <= MAX_DEFAULT_JVM_DISCOVERY_ENTRIES,
        "Default JVM input discovery exceeds its entry budget; configure explicit jvm_builds Maven/Gradle build-file selections and refresh"
    );
    let manifest_is_current = |path: &RelPath| {
        remote_path_is_visible(path, snapshot)
            && snapshot
                .entry_for_path(path)
                .is_some_and(|entry| entry.is_file())
            && path.ancestors().all(|ancestor| {
                snapshot.entry_for_path(ancestor).is_some_and(|entry| {
                    !entry.is_ignored
                        && entry.canonical_path.is_none()
                        && (ancestor == path || entry.is_dir())
                })
            })
    };
    let mut candidates = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    for (provider, names, selected, ignored) in providers {
        let ignored = ignored
            .iter()
            .map(|path| RelPath::from_unix_str(path))
            .collect::<Result<Vec<_>, _>>()?;
        let excluded = |path: &RelPath| ignored.iter().any(|directory| path.starts_with(directory));
        let mut admit = |path: &RelPath| -> Result<()> {
            if excluded(path) {
                return Ok(());
            }
            ensure!(
                manifest_is_current(path),
                "Selected {provider} build input is unavailable, ignored, private or aliased; select a current visible host build file and refresh"
            );
            inputs.insert(ProjectPath {
                worktree_id,
                path: Arc::from(path),
            });
            let parent = path.parent().unwrap_or(RelPath::empty());
            candidates.insert(ProjectPath {
                worktree_id,
                path: Arc::from(parent),
            });
            ensure!(
                candidates.len() <= MAX_JDTLS_MODULES,
                "Default JVM build roots exceed their admission budget; narrow jvm_builds selection and refresh"
            );
            Ok(())
        };
        if let Some(selected) = selected {
            for path in selected {
                admit(RelPath::from_unix_str(path)?)?;
            }
        } else {
            for entry in snapshot.entries(false, 0) {
                if entry.is_file()
                    && !entry.is_hidden
                    && entry
                        .path
                        .file_name()
                        .is_some_and(|name| names.contains(&name))
                    && manifest_is_current(&entry.path)
                {
                    admit(&entry.path)?;
                }
            }
        }
    }
    Ok(inputs.into_iter().collect())
}

pub(crate) fn default_maven_import_inputs(
    snapshot: &Snapshot,
    worktree_id: WorktreeId,
) -> Result<Vec<ProjectPath>> {
    // Gradle selection and errors must not change Maven's candidate intent.
    let settings = settings::JvmBuildSettingsContent {
        maven: None,
        gradle: Some(settings::GradleBuildSettingsContent {
            selected_build_files: Some(Vec::new()),
            ..Default::default()
        }),
    };
    let inputs = default_jvm_build_inputs(snapshot, worktree_id, Some(&settings))?;
    ensure!(
        inputs.len() <= 32,
        "Default Maven inputs exceed the supported candidate bound; narrow selected_poms and refresh"
    );
    let paths = inputs
        .iter()
        .map(|input| input.path.as_unix_str().to_owned())
        .collect::<Vec<_>>();
    validate_saved_build_paths(Some(&paths), 32, "Maven", Some(&["pom.xml"])).context(
        "Default Maven input names are unsupported; select portable bounded POM paths and refresh",
    )?;
    Ok(inputs)
}

pub(crate) fn default_gradle_import_inputs(
    snapshot: &Snapshot,
    worktree_id: WorktreeId,
) -> Result<Vec<ProjectPath>> {
    let settings = settings::JvmBuildSettingsContent {
        maven: Some(settings::MavenBuildSettingsContent {
            selected_poms: Some(Vec::new()),
            ..Default::default()
        }),
        gradle: None,
    };
    let inputs = default_jvm_build_inputs(snapshot, worktree_id, Some(&settings))?;
    ensure!(
        inputs.len() <= 128,
        "Default Gradle inputs exceed the supported candidate bound; narrow selected_build_files and refresh"
    );
    let paths = inputs
        .iter()
        .map(|input| input.path.as_unix_str().to_owned())
        .collect::<Vec<_>>();
    validate_saved_build_paths(
        Some(&paths),
        128,
        "Gradle",
        Some(&[
            "settings.gradle",
            "settings.gradle.kts",
            "build.gradle",
            "build.gradle.kts",
        ]),
    )
    .context("Default Gradle input names are unsupported; select portable bounded build paths and refresh")?;
    Ok(inputs)
}

pub fn default_jvm_build_roots(
    snapshot: &Snapshot,
    worktree_id: WorktreeId,
    settings: Option<&settings::JvmBuildSettingsContent>,
) -> Result<Vec<ProjectPath>> {
    let candidates = default_jvm_build_inputs(snapshot, worktree_id, settings)?
        .into_iter()
        .map(|input| ProjectPath {
            worktree_id,
            path: Arc::from(input.path.parent().unwrap_or(RelPath::empty())),
        })
        .collect::<BTreeSet<_>>();
    let mut roots: Vec<ProjectPath> = Vec::new();
    for candidate in candidates {
        if !roots
            .iter()
            .any(|root| candidate.path.starts_with(&root.path))
        {
            roots.push(candidate);
        }
    }
    Ok(roots)
}

pub const JDTLS_SNAPSHOT_PROTOCOL_VERSION: u32 = 8;

pub fn jdtls_snapshot_size_limit(version: u32) -> Result<usize> {
    match version {
        1 => Ok(1_048_576),
        2..=JDTLS_SNAPSHOT_PROTOCOL_VERSION => Ok(8 * 1_048_576),
        _ => anyhow::bail!("Java project snapshot protocol mismatch"),
    }
}

pub const MAX_JVM_DEPENDENCY_REPORT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_JVM_DEPENDENCY_NODES: usize = 4096;
pub const MAX_JVM_DEPENDENCY_EDGES: usize = 32768;
const MAX_JVM_DEPENDENCY_FIELD_BYTES: usize = 512;
const MAX_MAVEN_DEPENDENCY_DEPTH: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JvmReviewedDependencyFormat {
    MavenSelectedTree,
    GradleResolution {
        project_path: String,
        configuration: String,
    },
}

#[derive(Clone)]
pub struct JvmReviewedDependencyImport {
    pub(crate) owner_nonce: u128,
    pub(crate) settings_revision: u64,
    pub(crate) provider_generations: [(Option<u64>, Option<LanguageServerId>, bool); 2],
    pub(crate) module_root: ProjectPath,
    pub(crate) format: JvmReviewedDependencyFormat,
    pub(crate) root: PathBuf,
    pub(crate) root_inode: u64,
    pub(crate) files: Vec<(Arc<RelPath>, u64, Option<fs::MTime>, u64)>,
}

impl JvmReviewedDependencyImport {
    pub fn operation_identity(&self) -> String {
        let identity = format!(
            "{}:{:?}:{:?}:{}:{:?}:{:?}",
            self.owner_nonce,
            self.module_root,
            self.format,
            self.settings_revision,
            self.provider_generations,
            self.files
        );
        format!("{:x}", Sha256::digest(identity.as_bytes()))
    }

    pub fn module_root(&self) -> &ProjectPath {
        &self.module_root
    }
    pub fn format(&self) -> &JvmReviewedDependencyFormat {
        &self.format
    }

    pub(crate) fn retained_input_bytes(&self) -> Result<usize> {
        self.files.iter().try_fold(
            self.root.as_os_str().len()
                + self.module_root.path.as_std_path().as_os_str().len()
                + 2048,
            |bytes, (path, ..)| {
                bytes
                    .checked_add(path.as_std_path().as_os_str().len() + 64)
                    .context("JVM retained input budget overflow")
            },
        )
    }

    pub async fn preflight_filesystem(&self, filesystem: &dyn fs::Fs) -> Result<()> {
        self.read_inputs(filesystem).await.map(|_| ())
    }

    async fn read_inputs(&self, filesystem: &dyn fs::Fs) -> Result<BTreeMap<Arc<RelPath>, String>> {
        let root = filesystem
            .metadata(&self.root)
            .await?
            .context("JVM root is unavailable")?;
        ensure!(
            root.is_dir && !root.is_symlink && root.inode == self.root_inode,
            "Reviewed JVM root changed"
        );
        let mut inputs = BTreeMap::new();
        for (path, inode, mtime, size) in &self.files {
            let absolute = self.root.join(path.as_std_path());
            let metadata = filesystem
                .metadata(&absolute)
                .await?
                .context("JVM source disappeared")?;
            ensure!(
                !metadata.is_symlink
                    && !metadata.is_dir
                    && !metadata.is_fifo
                    && metadata.inode == *inode
                    && Some(metadata.mtime) == *mtime
                    && metadata.len == *size,
                "Reviewed JVM input changed or is not ordinary"
            );
            let mut confined = filesystem
                .open_sync_confined_file(&self.root, path.as_std_path(), self.root_inode, *inode)
                .await?;
            let mut contents = String::new();
            confined
                .by_ref()
                .take(size.saturating_add(1))
                .read_to_string(&mut contents)
                .map_err(|_| anyhow::anyhow!("Reviewed JVM input is not bounded UTF-8"))?;
            ensure!(
                contents.len() as u64 == *size,
                "JVM input changed while reading"
            );
            let after = filesystem
                .metadata(&absolute)
                .await?
                .context("JVM source disappeared while reading")?;
            ensure!(
                after.inode == *inode
                    && Some(after.mtime) == *mtime
                    && after.len == *size
                    && !after.is_symlink,
                "JVM input changed while reading"
            );
            inputs.insert(path.clone(), contents);
        }
        if self.format == JvmReviewedDependencyFormat::MavenSelectedTree {
            let pom = self
                .module_root
                .path
                .join(RelPath::from_unix_str("pom.xml")?);
            let contents = inputs
                .get::<RelPath>(pom.as_ref())
                .context("Reviewed Maven POM unavailable")?;
            validate_doctor_maven_mode(contents)?;
            ensure!(
                !self.files.iter().any(|(path, ..)| matches!(
                    path.file_name(),
                    Some("maven.config" | "jvm.config" | "extensions.xml")
                )),
                "Maven project option or extension effects cannot establish the supported selected report mode"
            );
        }
        Ok(inputs)
    }

    pub(crate) async fn interpret(
        &self,
        contents: &[u8],
        filesystem: &dyn fs::Fs,
    ) -> Result<JvmDependencyGraph> {
        let inputs = self.read_inputs(filesystem).await?;
        let graph = match &self.format {
            JvmReviewedDependencyFormat::MavenSelectedTree => {
                let graph = parse_maven_dependency_report(
                    contents,
                    self.module_root.clone(),
                    &MavenDependencyReportMode {
                        plugin_version: "3.7.0".into(),
                        verbose: false,
                        scope: None,
                        includes: Vec::new(),
                        excludes: Vec::new(),
                    },
                )
                .map_err(|_| {
                    anyhow::anyhow!("Reviewed Maven dependency report is invalid or unsupported")
                })?;
                let pom = self
                    .module_root
                    .path
                    .join(RelPath::from_unix_str("pom.xml")?);
                let expected = validate_doctor_maven_mode(
                    inputs
                        .get::<RelPath>(pom.as_ref())
                        .context("Current Maven POM unavailable")?,
                )?;
                let root = graph
                    .roots
                    .first()
                    .and_then(|index| graph.nodes.get(*index as usize))
                    .context("Maven report has no root")?;
                ensure!(
                    (&root.group_id, &root.artifact_id, &root.version)
                        == (&expected.0, &expected.1, &expected.2),
                    "Maven dependency report does not describe the selected current POM"
                );
                graph
            }
            JvmReviewedDependencyFormat::GradleResolution {
                project_path,
                configuration,
            } => parse_gradle_dependency_report(
                contents,
                self.module_root.clone(),
                project_path,
                configuration,
            )
            .map_err(|_| {
                anyhow::anyhow!("Reviewed Gradle dependency report is invalid or unsupported")
            })?,
        };
        graph.validate()?;
        Ok(graph)
    }
}

fn validate_doctor_maven_mode(contents: &str) -> Result<(String, String, String)> {
    use quick_xml::{Reader, events::Event};
    ensure!(
        contents.len() <= 1024 * 1024,
        "Maven POM exceeds the reviewed limit"
    );
    let mut reader = Reader::from_str(contents);
    let mut stack = Vec::<String>::new();
    let mut fields = BTreeMap::new();
    let mut elements = 0_usize;
    loop {
        let event = reader
            .read_event()
            .map_err(|_| anyhow::anyhow!("Maven POM is malformed"))?;
        let empty = matches!(&event, Event::Empty(_));
        match event {
            Event::Start(start) | Event::Empty(start) => {
                elements += 1;
                ensure!(
                    elements <= 4096 && stack.len() < 64,
                    "Maven POM exceeds structural limits"
                );
                let name = std::str::from_utf8(start.name().as_ref())?.to_string();
                ensure!(
                    !name.contains(':'),
                    "Qualified Maven element semantics are unsupported"
                );
                ensure!(
                    !matches!(
                        name.as_str(),
                        "parent" | "profiles" | "extensions" | "configuration"
                    ),
                    "Inherited, profiled, extension or plugin configuration requires effective-mode proof unavailable to this collection"
                );
                if stack.is_empty() {
                    ensure!(name == "project", "Maven document is not a project");
                }
                if !empty {
                    stack.push(name);
                }
            }
            Event::End(_) => {
                ensure!(stack.pop().is_some(), "Malformed Maven structure");
            }
            Event::Text(text)
                if stack.len() == 2 && stack.first().is_some_and(|name| name == "project") =>
            {
                let name = stack.last().context("Maven field missing")?;
                if matches!(name.as_str(), "groupId" | "artifactId" | "version") {
                    let value = text
                        .decode()
                        .map_err(|_| anyhow::anyhow!("Invalid Maven field"))?
                        .trim()
                        .to_string();
                    ensure!(
                        !value.contains(['$', '&']) && fields.insert(name.clone(), value).is_none(),
                        "Interpolated or duplicate Maven identity requires unavailable effective-mode proof"
                    );
                }
            }
            Event::DocType(_) | Event::CData(_) | Event::GeneralRef(_) => {
                anyhow::bail!("Maven entity or literal semantics are unsupported")
            }
            Event::Eof => break,
            _ => {}
        }
    }
    ensure!(stack.is_empty(), "Malformed Maven POM structure");
    let field = |name| {
        fields
            .get(name)
            .cloned()
            .context("Maven selected POM needs a literal group, artifact and version")
    };
    let group = field("groupId")?;
    let artifact = field("artifactId")?;
    let version = field("version")?;
    jvm_dependency_coordinate(&group)?;
    jvm_dependency_coordinate(&artifact)?;
    jvm_dependency_text(&version, false)?;
    Ok((group, artifact, version))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JvmDependencyBuildSystem {
    Maven,
    Gradle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JvmDependencyComponentKind {
    Workspace,
    Module,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JvmDependencyNode {
    pub identity: String,
    pub kind: JvmDependencyComponentKind,
    pub group_id: String,
    pub artifact_id: String,
    pub version: String,
    pub artifact_type: Option<String>,
    pub classifier: Option<String>,
    pub project_path: Option<String>,
    pub variants: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JvmDependencyEdge {
    pub parent: u32,
    pub child: u32,
    pub requested: Option<String>,
    pub scope: String,
    pub optional: Option<bool>,
    pub constraint: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JvmDependencyGraph {
    pub format_version: u32,
    pub build_system: JvmDependencyBuildSystem,
    pub module_root: ProjectPath,
    pub provider_version: String,
    pub configuration: String,
    pub report_identity: String,
    pub nodes: Vec<JvmDependencyNode>,
    pub edges: Vec<JvmDependencyEdge>,
    pub roots: Vec<u32>,
    pub unresolved_dependencies: u32,
    pub selected_versions_known: bool,
    pub introducing_paths_complete: bool,
    pub repository_origins_known: bool,
    pub declarations_known: bool,
    pub partial: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MavenDependencyReportMode {
    pub plugin_version: String,
    pub verbose: bool,
    pub scope: Option<String>,
    pub includes: Vec<String>,
    pub excludes: Vec<String>,
}

fn jvm_dependency_identity(fields: &[&str]) -> String {
    let mut digest = Sha256::new();
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn jvm_dependency_text(value: &str, allow_empty: bool) -> Result<()> {
    ensure!(
        (allow_empty || !value.is_empty())
            && value.len() <= MAX_JVM_DEPENDENCY_FIELD_BYTES
            && !value.chars().any(char::is_control),
        "JVM dependency field is invalid or too large"
    );
    Ok(())
}

fn jvm_dependency_coordinate(value: &str) -> Result<()> {
    jvm_dependency_text(value, false)?;
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)),
        "JVM dependency coordinate is invalid"
    );
    Ok(())
}

impl JvmDependencyGraph {
    pub fn validate(&self) -> Result<()> {
        self.retained_field_bytes().map(|_| ())
    }

    pub fn retained_field_bytes(&self) -> Result<usize> {
        ensure!(
            self.format_version == 1,
            "Unsupported JVM dependency graph format"
        );
        ensure!(
            !self.nodes.is_empty()
                && self.nodes.len() <= MAX_JVM_DEPENDENCY_NODES
                && self.edges.len() <= MAX_JVM_DEPENDENCY_EDGES
                && !self.roots.is_empty()
                && self.roots.len() <= 128,
            "JVM dependency graph count limit exceeded"
        );
        ensure!(
            !self.repository_origins_known && !self.declarations_known && self.partial,
            "This report cannot establish repository or declaration provenance"
        );
        jvm_dependency_text(&self.provider_version, false)?;
        jvm_dependency_text(&self.configuration, false)?;
        ensure!(
            valid_jvm_dependency_hash(&self.report_identity),
            "Invalid report identity"
        );
        let mut identities = BTreeSet::new();
        let mut field_bytes =
            self.provider_version.len() + self.configuration.len() + self.report_identity.len();
        for node in &self.nodes {
            ensure!(
                valid_jvm_dependency_hash(&node.identity) && identities.insert(&node.identity),
                "Invalid or duplicate JVM dependency identity"
            );
            jvm_dependency_text(
                &node.group_id,
                node.kind != JvmDependencyComponentKind::Module,
            )?;
            jvm_dependency_text(&node.artifact_id, false)?;
            jvm_dependency_text(
                &node.version,
                node.kind != JvmDependencyComponentKind::Module,
            )?;
            ensure!(
                node.variants.len() <= 64,
                "JVM dependency variant limit exceeded"
            );
            for value in [
                &node.identity,
                &node.group_id,
                &node.artifact_id,
                &node.version,
            ]
            .into_iter()
            .chain(node.artifact_type.iter())
            .chain(node.classifier.iter())
            .chain(node.project_path.iter())
            .chain(node.variants.iter())
            {
                jvm_dependency_text(value, true)?;
                field_bytes = field_bytes
                    .checked_add(value.len())
                    .context("JVM graph size overflow")?;
            }
        }
        let mut roots = BTreeSet::new();
        for root in &self.roots {
            ensure!(
                (*root as usize) < self.nodes.len() && roots.insert(root),
                "Invalid or duplicate JVM graph root"
            );
        }
        for edge in &self.edges {
            ensure!(
                (edge.parent as usize) < self.nodes.len()
                    && (edge.child as usize) < self.nodes.len(),
                "JVM dependency edge has a missing node"
            );
            jvm_dependency_text(&edge.scope, true)?;
            field_bytes = field_bytes
                .checked_add(edge.scope.len())
                .context("JVM graph size overflow")?;
            if let Some(requested) = &edge.requested {
                jvm_dependency_text(requested, false)?;
                field_bytes = field_bytes
                    .checked_add(requested.len())
                    .context("JVM graph size overflow")?;
            }
        }
        ensure!(
            field_bytes <= MAX_JVM_DEPENDENCY_REPORT_BYTES,
            "JVM graph aggregate field limit exceeded"
        );
        Ok(field_bytes)
    }
}

fn valid_jvm_dependency_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub fn parse_maven_dependency_report(
    contents: &[u8],
    module_root: ProjectPath,
    mode: &MavenDependencyReportMode,
) -> Result<JvmDependencyGraph> {
    ensure!(
        contents.len() <= MAX_JVM_DEPENDENCY_REPORT_BYTES,
        "Maven dependency report exceeds the byte limit"
    );
    ensure!(
        mode.plugin_version == "3.7.0"
            && !mode.verbose
            && mode.scope.is_none()
            && mode.includes.is_empty()
            && mode.excludes.is_empty(),
        "Only the reviewed pinned unfiltered nonverbose Maven selected-tree mode is supported"
    );
    let value: serde_json::Value =
        serde_json::from_slice(contents).context("Maven dependency report is malformed")?;
    let report_identity = jvm_dependency_identity(&[std::str::from_utf8(contents)?]);
    let mut graph = JvmDependencyGraph {
        format_version: 1,
        build_system: JvmDependencyBuildSystem::Maven,
        module_root,
        provider_version: "maven-dependency-plugin 3.7.0 selected tree".into(),
        configuration: "all effective dependency scopes".into(),
        report_identity,
        nodes: Vec::new(),
        edges: Vec::new(),
        roots: Vec::new(),
        unresolved_dependencies: 0,
        selected_versions_known: true,
        introducing_paths_complete: false,
        repository_origins_known: false,
        declarations_known: false,
        partial: true,
    };
    let mut pending = vec![(&value, None, 0_usize)];
    let mut instances = BTreeMap::new();
    let mut occurrences = 0_usize;
    while let Some((value, parent, depth)) = pending.pop() {
        ensure!(
            depth <= MAX_MAVEN_DEPENDENCY_DEPTH,
            "Maven dependency path exceeds the depth limit"
        );
        occurrences += 1;
        ensure!(
            occurrences <= MAX_JVM_DEPENDENCY_EDGES,
            "Maven dependency occurrence limit exceeded"
        );
        let object = value
            .as_object()
            .context("Maven dependency node is not an object")?;
        ensure!(
            object.keys().all(|key| matches!(
                key.as_str(),
                "groupId"
                    | "artifactId"
                    | "version"
                    | "type"
                    | "scope"
                    | "classifier"
                    | "optional"
                    | "children"
            )),
            "Unsupported Maven dependency node fields"
        );
        let text = |name: &str| -> Result<&str> {
            object
                .get(name)
                .and_then(serde_json::Value::as_str)
                .context("Maven dependency node has missing or invalid fields")
        };
        let group = text("groupId")?;
        let artifact = text("artifactId")?;
        let version = text("version")?;
        let artifact_type = text("type")?;
        let scope = text("scope")?;
        let classifier = text("classifier")?;
        jvm_dependency_coordinate(group)?;
        jvm_dependency_coordinate(artifact)?;
        jvm_dependency_text(version, false)?;
        jvm_dependency_text(artifact_type, false)?;
        jvm_dependency_text(scope, true)?;
        jvm_dependency_text(classifier, true)?;
        let optional = match text("optional")? {
            "true" => true,
            "false" => false,
            _ => anyhow::bail!("Invalid Maven optional dependency flag"),
        };
        let identity = jvm_dependency_identity(&[
            &graph.report_identity,
            if parent.is_none() {
                "workspace"
            } else {
                "unknown repository"
            },
            group,
            artifact,
            version,
            artifact_type,
            classifier,
            scope,
        ]);
        let index = if let Some(index) = instances.get(&identity) {
            *index
        } else {
            ensure!(
                graph.nodes.len() < MAX_JVM_DEPENDENCY_NODES,
                "Maven dependency node limit exceeded"
            );
            let index = u32::try_from(graph.nodes.len())?;
            graph.nodes.push(JvmDependencyNode {
                identity: identity.clone(),
                kind: if parent.is_none() {
                    JvmDependencyComponentKind::Workspace
                } else {
                    JvmDependencyComponentKind::Module
                },
                group_id: group.into(),
                artifact_id: artifact.into(),
                version: version.into(),
                artifact_type: Some(artifact_type.into()),
                classifier: (!classifier.is_empty()).then(|| classifier.into()),
                project_path: None,
                variants: Vec::new(),
            });
            instances.insert(identity, index);
            index
        };
        if let Some(parent) = parent {
            ensure!(
                graph.edges.len() < MAX_JVM_DEPENDENCY_EDGES,
                "Maven dependency edge limit exceeded"
            );
            graph.edges.push(JvmDependencyEdge {
                parent,
                child: index,
                requested: None,
                scope: scope.into(),
                optional: Some(optional),
                constraint: false,
            });
        } else {
            graph.roots.push(index);
        }
        if let Some(children) = object.get("children") {
            let children = children
                .as_array()
                .context("Maven dependency children are invalid")?;
            ensure!(
                pending
                    .len()
                    .checked_add(children.len())
                    .is_some_and(|count| count <= MAX_JVM_DEPENDENCY_EDGES),
                "Maven pending dependency limit exceeded"
            );
            pending.extend(
                children
                    .iter()
                    .rev()
                    .map(|child| (child, Some(index), depth + 1)),
            );
        }
    }
    graph.validate()?;
    Ok(graph)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GradleDependencyReport {
    version: u32,
    gradle_version: String,
    configuration: String,
    project_path: String,
    root: u32,
    nodes: Vec<GradleDependencyReportNode>,
    edges: Vec<GradleDependencyReportEdge>,
    unresolved_dependencies: u32,
    partial: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GradleDependencyReportNode {
    component: String,
    kind: String,
    group: String,
    module: String,
    version: String,
    project_path: Option<String>,
    variants: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GradleDependencyReportEdge {
    parent: u32,
    child: u32,
    requested: String,
    constraint: bool,
}

pub fn parse_gradle_dependency_report(
    contents: &[u8],
    module_root: ProjectPath,
    expected_project_path: &str,
    expected_configuration: &str,
) -> Result<JvmDependencyGraph> {
    ensure!(
        contents.len() <= MAX_JVM_DEPENDENCY_REPORT_BYTES,
        "Gradle dependency report exceeds the byte limit"
    );
    let report: GradleDependencyReport =
        serde_json::from_slice(contents).context("Gradle dependency report is malformed")?;
    ensure!(
        report.version == 1 && report.gradle_version == "8.13",
        "Unsupported Gradle resolver report version"
    );
    ensure!(
        report.configuration == expected_configuration
            && report.project_path == expected_project_path
            && gradle_path_is_valid(&report.project_path, true),
        "Gradle report differs from the reviewed project/configuration"
    );
    ensure!(
        report.nodes.len() <= MAX_JVM_DEPENDENCY_NODES
            && report.edges.len() <= MAX_JVM_DEPENDENCY_EDGES,
        "Gradle dependency graph count limit exceeded"
    );
    jvm_dependency_text(&report.configuration, false)?;
    let report_identity = jvm_dependency_identity(&[std::str::from_utf8(contents)?]);
    let mut graph = JvmDependencyGraph {
        format_version: 1,
        build_system: JvmDependencyBuildSystem::Gradle,
        module_root,
        provider_version: format!("Gradle {} ResolutionResult", report.gradle_version),
        configuration: report.configuration,
        report_identity,
        nodes: Vec::new(),
        edges: Vec::new(),
        roots: vec![report.root],
        unresolved_dependencies: report.unresolved_dependencies,
        selected_versions_known: true,
        introducing_paths_complete: !report.partial && report.unresolved_dependencies == 0,
        repository_origins_known: false,
        declarations_known: false,
        partial: true,
    };
    for node in report.nodes {
        ensure!(
            valid_jvm_dependency_hash(&node.component),
            "Invalid opaque Gradle component identity"
        );
        let kind = match node.kind.as_str() {
            "project" => JvmDependencyComponentKind::Workspace,
            "module" => JvmDependencyComponentKind::Module,
            "unknown" => JvmDependencyComponentKind::Unknown,
            _ => anyhow::bail!("Unsupported Gradle component kind"),
        };
        if kind == JvmDependencyComponentKind::Module {
            jvm_dependency_coordinate(&node.group)?;
            jvm_dependency_coordinate(&node.module)?;
        }
        if let Some(project_path) = &node.project_path {
            ensure!(
                kind == JvmDependencyComponentKind::Workspace
                    && gradle_path_is_valid(project_path, true),
                "Invalid Gradle project component path"
            );
        }
        graph.nodes.push(JvmDependencyNode {
            identity: jvm_dependency_identity(&[&graph.report_identity, &node.component]),
            kind,
            group_id: node.group,
            artifact_id: node.module,
            version: node.version,
            artifact_type: None,
            classifier: None,
            project_path: node.project_path,
            variants: node.variants,
        });
    }
    for edge in report.edges {
        graph.edges.push(JvmDependencyEdge {
            parent: edge.parent,
            child: edge.child,
            requested: Some(edge.requested),
            scope: graph.configuration.clone(),
            optional: None,
            constraint: edge.constraint,
        });
    }
    ensure!(
        graph.unresolved_dependencies as usize <= MAX_JVM_DEPENDENCY_EDGES,
        "Gradle unresolved dependency count limit exceeded"
    );
    let root = graph
        .roots
        .first()
        .and_then(|root| graph.nodes.get(*root as usize))
        .context("Gradle report root is unavailable")?;
    ensure!(
        root.kind == JvmDependencyComponentKind::Workspace
            && root.project_path.as_deref() == Some(expected_project_path),
        "Gradle resolver root differs from the reviewed project"
    );
    if graph.introducing_paths_complete {
        let mut reachable = BTreeSet::new();
        let mut pending = graph.roots.clone();
        while let Some(node) = pending.pop() {
            if reachable.insert(node) {
                pending.extend(
                    graph
                        .edges
                        .iter()
                        .filter(|edge| edge.parent == node)
                        .map(|edge| edge.child),
                );
            }
        }
        ensure!(
            reachable.len() == graph.nodes.len(),
            "Gradle report has disconnected dependency nodes"
        );
    }
    graph.validate()?;
    Ok(graph)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KotlinWorkspaceProject {
    pub name: String,
    pub build_system: KotlinBuildSystem,
    pub module_root: ProjectPath,
    pub source_roots: Vec<KotlinWorkspaceSourcePath>,
    pub dependencies: Vec<KotlinWorkspaceDependency>,
    pub unidentified_libraries: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KotlinBuildSystem {
    Unknown,
    Maven,
    Gradle,
}

impl KotlinBuildSystem {
    fn from_provider(value: &str) -> Option<Self> {
        match value {
            "MAVEN" => Some(Self::Maven),
            "GRADLE" => Some(Self::Gradle),
            _ => None,
        }
    }

    fn from_remote(value: &str) -> Option<Self> {
        if value.is_empty() {
            Some(Self::Unknown)
        } else {
            Self::from_provider(value)
        }
    }

    fn wire_name(self) -> &'static str {
        match self {
            Self::Unknown => "",
            Self::Maven => "MAVEN",
            Self::Gradle => "GRADLE",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KotlinWorkspaceSourcePath {
    pub path: ProjectPath,
    pub navigable: bool,
    pub generated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KotlinWorkspaceDependency {
    Module {
        name: String,
        scope: Option<String>,
    },
    Artifact {
        coordinate: String,
        scope: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceData {
    #[serde(rename = "externalSystem")]
    external_system: Option<String>,
    #[serde(default)]
    modules: Vec<KotlinWorkspaceModule>,
    #[serde(default)]
    libraries: Vec<KotlinWorkspaceLibrary>,
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceModule {
    name: String,
    #[serde(rename = "externalProjectPath")]
    external_project_path: Option<String>,
    #[serde(default, rename = "contentRoots")]
    content_roots: Vec<KotlinWorkspaceContentRoot>,
    #[serde(default)]
    dependencies: Vec<KotlinWorkspaceDependencyData>,
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceContentRoot {
    #[serde(default, rename = "sourceRoots")]
    source_roots: Vec<KotlinWorkspaceSourceRoot>,
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceSourceRoot {
    path: String,
    #[serde(default)]
    generated: bool,
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceDependencyData {
    #[serde(rename = "type")]
    kind: String,
    name: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KotlinWorkspaceLibrary {
    name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JdtlsProject {
    pub module_root: ProjectPath,
    pub membership: JdtlsProjectMembership,
    pub imported_build_kind: JdtlsImportedBuildKind,
    pub source_roots: Vec<JdtlsSourceRoot>,
    pub project_references: Vec<String>,
    pub library_entries: usize,
    pub library_files: Vec<JdtlsLibraryFile>,
    pub library_artifacts: Vec<JdtlsLibraryArtifact>,
    pub project_vm_known: bool,
    pub project_vm_java_major: Option<u32>,
    pub server_launch_java_major: Option<u32>,
    pub java_source_level: Option<u32>,
    pub java_target_level: Option<u32>,
    pub gradle: Option<GradleProject>,
    pub maven: Option<MavenProject>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JdtlsImportedBuildKind {
    #[default]
    Unknown,
    Maven,
    Gradle,
}

impl JdtlsImportedBuildKind {
    fn to_proto(self) -> proto::JdtlsImportedBuildKind {
        match self {
            Self::Unknown => proto::JdtlsImportedBuildKind::Unknown,
            Self::Maven => proto::JdtlsImportedBuildKind::Maven,
            Self::Gradle => proto::JdtlsImportedBuildKind::Gradle,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JdtlsProjectMembership {
    #[default]
    Unknown,
    MetadataOnly,
    ImportedJava,
    ImportedNonJava,
}

impl JdtlsProjectMembership {
    pub fn is_imported(self) -> bool {
        matches!(self, Self::ImportedJava | Self::ImportedNonJava)
    }

    fn to_proto(self) -> proto::JdtlsProjectMembership {
        match self {
            Self::Unknown => proto::JdtlsProjectMembership::Unknown,
            Self::MetadataOnly => proto::JdtlsProjectMembership::MetadataOnly,
            Self::ImportedJava => proto::JdtlsProjectMembership::ImportedJava,
            Self::ImportedNonJava => proto::JdtlsProjectMembership::ImportedNonJava,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MavenProject {
    pub pom: ProjectPath,
    pub modules: Vec<ProjectPath>,
    pub group_id: String,
    pub artifact_id: String,
    pub version: String,
    pub packaging: String,
    pub profiles: BTreeMap<String, bool>,
    pub profile_definitions_known: bool,
    pub profile_definitions: Vec<MavenProfileDefinition>,
    pub reactor_position: Option<usize>,
    pub reactor_order_problem: Option<String>,
    pub plugins: Vec<MavenPlugin>,
    pub lifecycles: Vec<MavenLifecycle>,
    pub dependency_graph: Option<Arc<MavenDependencyGraph>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MavenDependencyGraph {
    version: u32,
    #[serde(default, skip_serializing)]
    root_uri: String,
    #[serde(default, skip_serializing)]
    pom_uri: String,
    pub occurrences_known: bool,
    pub root_node: String,
    pub root_component: String,
    pub nodes: Vec<MavenDependencyNode>,
    pub components: Vec<MavenDependencyComponent>,
    pub edges: Vec<MavenDependencyEdge>,
    pub unavailable_descriptors: Vec<MavenUnavailableDescriptor>,
    pub collection_complete: bool,
    pub structural_coverage_known: bool,
    pub problem: Option<String>,
    pub origins_known: bool,
    pub artifacts_known: bool,
    pub offline: bool,
    pub cached_only: bool,
    pub cycle_count: usize,
    pub cycles: Vec<MavenDependencyCycle>,
    pub cycle_occurrences_known: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MavenDependencyComponent {
    pub id: String,
    pub group_id: String,
    pub artifact_id: String,
    pub extension: String,
    pub classifier: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MavenDependencyNode {
    pub id: String,
    pub component: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MavenDependencyEdge {
    pub from: String,
    pub from_node: String,
    pub target_node: String,
    pub selected_node: String,
    pub position: usize,
    pub component: String,
    pub selected: String,
    pub omitted: bool,
    pub direct: bool,
    pub version_constraint: Option<String>,
    pub premanaged_version: Option<String>,
    pub premanaged_scope: Option<String>,
    pub premanaged_optional: Option<bool>,
    pub managed_bits: u32,
    pub original_scope: Option<String>,
    pub original_optional: Option<bool>,
    pub scope: String,
    pub optional: bool,
    pub exclusions: Vec<MavenDependencyExclusion>,
    pub premanaged_exclusions: Option<Vec<MavenDependencyExclusion>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MavenDependencyExclusion {
    pub group_id: String,
    pub artifact_id: String,
    pub extension: String,
    pub classifier: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MavenDependencyCycle {
    pub preceding: Vec<MavenCycleDependency>,
    pub cyclic: Vec<MavenCycleDependency>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MavenCycleDependency {
    pub component: String,
    pub scope: String,
    pub optional: bool,
    pub exclusions: Vec<MavenDependencyExclusion>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MavenUnavailableDescriptor {
    pub component: String,
    pub problem: String,
}

#[derive(Clone, Debug)]
pub struct SelectedMavenDependencyGraph {
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedMavenDependencyGraph {
    pub fn new(root: &Path, pom: &Path) -> Result<Self> {
        SelectedMavenImport::new(root, &[pom.to_path_buf()])?;
        Ok(Self {
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.maven.dependencyGraph".into(),
                arguments: vec![json!({
                    "rootUri": selected_import_uri(root, true)?,
                    "pomUri": selected_import_uri(pom, false)?,
                    "offline": true, "cachedOnly": true,
                })],
                ..Default::default()
            },
        })
    }

    pub async fn query(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<MavenDependencyGraph> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Maven", "dependency graph",
            "Maven dependency graph failed; check host offline settings, cached metadata and project import, then refresh before retrying",
        ).await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: Value) -> Result<MavenDependencyGraph> {
        let request = self
            .parameters
            .arguments
            .first()
            .context("Missing Maven graph selection")?;
        ensure!(
            response["rootUri"] == request["rootUri"] && response["pomUri"] == request["pomUri"],
            "Maven dependency graph identity changed; refresh project data before retrying"
        );
        decode_maven_dependency_graph(response)
    }
}

fn decode_maven_dependency_graph(response: Value) -> Result<MavenDependencyGraph> {
    let invalid = || {
        anyhow::anyhow!(
            "Maven dependency graph is invalid or incomplete; refresh project data and check cached metadata before retrying"
        )
    };
    let mut pending = vec![(&response, 0usize)];
    let mut text_bytes = 0usize;
    let mut values = 0usize;
    while let Some((value, depth)) = pending.pop() {
        values += 1;
        ensure!(
            values + pending.len() <= 1_000_000 && depth <= 32,
            "{}",
            invalid()
        );
        match value {
            Value::String(text) => {
                text_bytes = text_bytes.checked_add(text.len()).ok_or_else(invalid)?;
                ensure!(
                    text.len() <= 4096
                        && !text.chars().any(char::is_control)
                        && text_bytes <= 1024 * 1024,
                    "{}",
                    invalid()
                );
            }
            Value::Array(array) => {
                ensure!(array.len() <= 16384, "{}", invalid());
                pending.extend(array.iter().map(|value| (value, depth + 1)));
            }
            Value::Object(object) => {
                ensure!(object.len() <= 64, "{}", invalid());
                for (key, value) in object {
                    text_bytes = text_bytes.checked_add(key.len()).ok_or_else(invalid)?;
                    ensure!(
                        key.len() <= 4096 && text_bytes <= 1024 * 1024,
                        "{}",
                        invalid()
                    );
                    pending.push((value, depth + 1));
                }
            }
            _ => {}
        }
    }
    ensure!(
        serde_json::to_vec(&response).map_err(|_| invalid())?.len() <= 8 * 1024 * 1024,
        "{}",
        invalid()
    );
    let graph: MavenDependencyGraph = serde_json::from_value(response).map_err(|_| invalid())?;
    ensure!(
        graph.version == 2
            && graph.offline
            && graph.cached_only
            && graph.occurrences_known
            && !graph.origins_known
            && !graph.artifacts_known
            && !graph.cycle_occurrences_known
            && graph.collection_complete == graph.structural_coverage_known
            && graph.problem.as_deref()
                == if graph.collection_complete {
                    None
                } else {
                    Some("offline-metadata-unavailable-check-cache-and-refresh")
                }
            && (!graph.collection_complete || graph.unavailable_descriptors.is_empty())
            && !graph.nodes.is_empty()
            && graph.nodes.len() <= 4096
            && !graph.components.is_empty()
            && graph.components.len() <= 4096
            && graph.edges.len() <= 16384
            && graph.unavailable_descriptors.len() <= 4096
            && graph.cycles.len() <= 4096
            && graph.cycle_count == graph.cycles.len(),
        "{}",
        invalid()
    );
    let mut components = HashSet::new();
    let mut coordinate_identities = HashSet::new();
    for component in &graph.components {
        let coordinates: Vec<String> =
            serde_json::from_str(&component.id).map_err(|_| invalid())?;
        ensure!(
            coordinates
                == [
                    component.group_id.clone(),
                    component.artifact_id.clone(),
                    component.extension.clone(),
                    component.classifier.clone(),
                    component.version.clone()
                ]
                && !component.group_id.is_empty()
                && !component.artifact_id.is_empty()
                && !component.extension.is_empty()
                && components.insert(component.id.as_str())
                && coordinate_identities.insert(coordinates),
            "{}",
            invalid()
        );
    }
    let mut nodes = HashMap::new();
    for node in &graph.nodes {
        ensure!(
            node.id
                .strip_prefix("node:")
                .is_some_and(|identity| identity.len() == 64
                    && identity
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
                && components.contains(node.component.as_str())
                && nodes
                    .insert(node.id.as_str(), node.component.as_str())
                    .is_none(),
            "{}",
            invalid()
        );
    }
    ensure!(
        nodes.get(graph.root_node.as_str()) == Some(&graph.root_component.as_str()),
        "{}",
        invalid()
    );
    let mut children: HashMap<&str, BTreeMap<usize, &str>> = HashMap::new();
    for edge in &graph.edges {
        ensure!(
            nodes.get(edge.from_node.as_str()) == Some(&edge.from.as_str())
                && nodes.get(edge.target_node.as_str()) == Some(&edge.component.as_str())
                && nodes.get(edge.selected_node.as_str()) == Some(&edge.selected.as_str())
                && (edge.omitted || edge.selected_node == edge.target_node)
                && edge.direct == (edge.from_node == graph.root_node)
                && edge.exclusions.len() <= 128
                && edge
                    .premanaged_exclusions
                    .as_ref()
                    .is_none_or(|values| values.len() <= 128)
                && children
                    .entry(&edge.from_node)
                    .or_default()
                    .insert(edge.position, &edge.target_node)
                    .is_none(),
            "{}",
            invalid()
        );
    }
    for positions in children.values() {
        ensure!(
            positions.keys().copied().eq(0..positions.len()),
            "{}",
            invalid()
        );
    }
    let mut reachable = HashSet::new();
    let mut pending = vec![graph.root_node.as_str()];
    while let Some(node) = pending.pop() {
        if reachable.insert(node)
            && let Some(children) = children.get(node)
        {
            pending.extend(children.values().copied());
        }
    }
    ensure!(reachable.len() == nodes.len(), "{}", invalid());
    let mut cycle_steps = 0usize;
    for cycle in &graph.cycles {
        ensure!(
            cycle.preceding.len() <= 256 && !cycle.cyclic.is_empty() && cycle.cyclic.len() <= 256,
            "{}",
            invalid()
        );
        cycle_steps += cycle.preceding.len() + cycle.cyclic.len();
        ensure!(
            cycle_steps <= 16384
                && cycle
                    .preceding
                    .iter()
                    .chain(&cycle.cyclic)
                    .all(|step| components.contains(step.component.as_str())
                        && step.exclusions.len() <= 128),
            "{}",
            invalid()
        );
    }
    ensure!(
        graph
            .unavailable_descriptors
            .iter()
            .all(
                |descriptor| components.contains(descriptor.component.as_str())
                    && descriptor.problem == "offline-metadata-unavailable-check-cache-and-refresh"
            ),
        "{}",
        invalid()
    );
    Ok(graph)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MavenProfileDefinition {
    pub id: String,
    pub source_group_id: String,
    pub source_artifact_id: String,
    pub source_version: String,
    pub source_pom: Option<ProjectPath>,
    pub inherited: bool,
    pub active_at_source: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MavenModelProfileDefinition {
    id: String,
    source_group_id: String,
    source_artifact_id: String,
    source_version: String,
    source_pom_uri: Option<String>,
    inherited: bool,
    active_at_source: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MavenPlugin {
    pub group_id: String,
    pub artifact_id: String,
    pub version: Option<String>,
    pub managed_only: bool,
    pub goals_known: bool,
    pub goal_prefix: Option<String>,
    pub goals: Vec<MavenGoal>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MavenGoal {
    pub name: String,
    pub phase: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub parameters_known: bool,
    #[serde(default)]
    pub parameters: Vec<MavenGoalParameter>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MavenGoalParameter {
    pub name: String,
    pub alias: Option<String>,
    pub declared_type: Option<String>,
    pub required: bool,
    pub editable: bool,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct MavenLifecycle {
    pub id: String,
    pub phases: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MavenModel {
    version: u32,
    root_uri: String,
    projects: Vec<MavenModelProject>,
    unavailable_pom_uris: Vec<String>,
    lifecycles: Vec<MavenLifecycle>,
    cached_only: bool,
    reactor_order_known: bool,
    reactor_pom_uris: Option<Vec<String>>,
    reactor_order_problem: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MavenModelProject {
    pom_uri: String,
    group_id: String,
    artifact_id: String,
    version: String,
    packaging: String,
    module_uris: Vec<String>,
    profiles: BTreeMap<String, bool>,
    inherited_profile_definitions_known: bool,
    profile_definitions: Option<Vec<MavenModelProfileDefinition>>,
    #[serde(skip)]
    reactor_position: Option<usize>,
    #[serde(skip)]
    reactor_order_problem: Option<String>,
    plugins: Vec<MavenPlugin>,
    #[serde(skip)]
    lifecycles: Vec<MavenLifecycle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GradleProject {
    pub dependency_graph: Option<Arc<GradleDependencyGraph>>,
    pub build_root: ProjectPath,
    pub project_path: String,
    pub tasks: Vec<GradleTask>,
    pub source_sets_known: bool,
    pub source_sets: Vec<GradleSourceSet>,
    pub configurations_known: bool,
    pub configurations: Vec<GradleConfiguration>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GradleDependencyGraph {
    #[serde(skip)]
    root_uri: String,
    #[serde(skip)]
    build_root_uri: String,
    #[serde(skip)]
    module_uri: String,
    pub project_path: String,
    pub name: String,
    pub resolution: GradleResolutionGraph,
}

impl GradleDependencyGraph {
    pub(crate) fn matches_project(&self, root: &Path, project: &GradleProject) -> bool {
        self.project_path == project.project_path
            && gradle_graph_binding_path(&self.build_root_uri).is_ok_and(|build_root| {
                build_root == root.join(project.build_root.path.as_unix_str())
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GradleResolutionGraph {
    pub configuration: String,
    pub root_component: String,
    pub structural_coverage_known: bool,
    pub resolution_complete: bool,
    pub origins_known: bool,
    pub artifacts_known: bool,
    pub components: Vec<GradleDependencyComponent>,
    pub edges: Vec<GradleDependencyEdge>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum GradleDependencyKind {
    Module,
    Project,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GradleDependencyComponent {
    pub id: String,
    pub kind: GradleDependencyKind,
    pub group: Option<String>,
    pub module: Option<String>,
    pub version: Option<String>,
    pub build_tree_path: Option<String>,
    pub project_path: Option<String>,
    pub conflict_resolution: bool,
    pub forced: bool,
    pub selected_by_rule: bool,
    pub composite_substitution: bool,
    pub constrained: bool,
    pub selection_causes: Vec<GradleSelectionCause>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GradleSelectionCause {
    Root,
    Requested,
    SelectedByRule,
    Forced,
    ConflictResolution,
    CompositeBuild,
    Rejection,
    Constraint,
    ByAncestor,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum GradleDependencySelector {
    Module {
        group: String,
        module: String,
        version: String,
        required_version: String,
        preferred_version: String,
        strict_version: String,
        rejected_versions: Vec<String>,
    },
    Project {
        build_path: String,
        project_path: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GradleDependencyEdge {
    pub from: String,
    pub requested: GradleDependencySelector,
    pub constraint: bool,
    pub origin_known: bool,
    pub resolved: bool,
    pub selected: Option<String>,
    pub selected_variant: Option<String>,
    pub failure_category: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SelectedGradleDependencyGraph {
    root: PathBuf,
    module: PathBuf,
    configuration: String,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedGradleDependencyGraph {
    pub fn new(root: &Path, module: &Path, configuration: &str) -> Result<Self> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle graph selection is invalid; select an imported module and resolvable configuration before retrying"
            )
        };
        let root_uri = selected_import_uri(root, true).map_err(|_| invalid())?;
        let module_uri = selected_import_uri(module, true).map_err(|_| invalid())?;
        ensure!(
            module.starts_with(root) && gradle_configuration_name_is_valid(configuration),
            "{}",
            invalid()
        );
        Ok(Self {
            root: root.into(),
            module: module.into(),
            configuration: configuration.into(),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.dependencyGraph".into(),
                arguments: vec![
                    json!({"rootUri": root_uri, "moduleUri": module_uri, "configuration": configuration, "offline": true}),
                ],
                ..Default::default()
            },
        })
    }

    pub async fn query(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<GradleDependencyGraph> {
        let response = request_selected_jdtls_phase(server, &self.parameters, request_timeout, "Gradle", "dependency graph",
            "Gradle dependency graph failed; check host offline settings, cached installation and metadata, then refresh before retrying").await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: Value) -> Result<GradleDependencyGraph> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle dependency graph is invalid or incomplete; refresh project data and check cached metadata before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 3
                && object
                    .get("projects")
                    .and_then(Value::as_array)
                    .is_some_and(|projects| projects.len() == 1),
            "{}",
            invalid()
        );
        let project = object
            .get("projects")
            .and_then(Value::as_array)
            .and_then(|projects| projects.first())
            .and_then(Value::as_object)
            .ok_or_else(invalid)?;
        ensure!(
            project.len() == 5
                && project
                    .get("graphJson")
                    .and_then(Value::as_str)
                    .is_some_and(|graph| graph.len() <= 2 * 1024 * 1024),
            "{}",
            invalid()
        );
        ensure!(
            object
                .get("rootUri")
                .and_then(Value::as_str)
                .is_some_and(graph_binding_text_is_valid)
                && ["buildRootUri", "moduleUri", "projectPath", "name"]
                    .iter()
                    .all(|key| project
                        .get(*key)
                        .and_then(Value::as_str)
                        .is_some_and(graph_binding_text_is_valid)),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Response {
            version: u32,
            root_uri: String,
            projects: Vec<Project>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Project {
            build_root_uri: String,
            module_uri: String,
            project_path: String,
            name: String,
            graph_json: String,
        }
        let response: Response = serde_json::from_value(response).map_err(|_| invalid())?;
        let project = response.projects.into_iter().next().ok_or_else(invalid)?;
        let root = gradle_graph_binding_path(&response.root_uri).map_err(|_| invalid())?;
        let build_root =
            gradle_graph_binding_path(&project.build_root_uri).map_err(|_| invalid())?;
        let module = gradle_graph_binding_path(&project.module_uri).map_err(|_| invalid())?;
        ensure!(
            response.version == 1
                && root == self.root
                && module == self.module
                && build_root.starts_with(&root)
                && module.starts_with(&build_root)
                && gradle_path_is_valid(&project.project_path, true)
                && !project.name.is_empty(),
            "{}",
            invalid()
        );
        let value: Value = serde_json::from_str(&project.graph_json).map_err(|_| invalid())?;
        let graph =
            decode_gradle_resolution_graph(value, &self.configuration, &project.project_path)?;
        Ok(GradleDependencyGraph {
            root_uri: response.root_uri,
            build_root_uri: project.build_root_uri,
            module_uri: project.module_uri,
            project_path: project.project_path,
            name: project.name,
            resolution: graph,
        })
    }
}

fn decode_gradle_resolution_graph(
    value: Value,
    configuration: &str,
    project_path: &str,
) -> Result<GradleResolutionGraph> {
    let invalid = || {
        anyhow::anyhow!(
            "Gradle dependency graph is invalid or incomplete; refresh project data and check cached metadata before retrying"
        )
    };
    validate_gradle_graph_json_budget(&value).map_err(|_| invalid())?;
    let graph: GradleResolutionGraph = serde_json::from_value(value).map_err(|_| invalid())?;
    ensure!(
        graph.configuration == configuration
            && graph.structural_coverage_known
            && !graph.origins_known
            && !graph.artifacts_known
            && !graph.components.is_empty()
            && graph.components.len() <= 4096
            && graph.edges.len() <= 16384,
        "{}",
        invalid()
    );
    let mut components = HashMap::new();
    for component in &graph.components {
        let identity: Vec<String> = serde_json::from_str(&component.id).map_err(|_| invalid())?;
        let valid = match component.kind {
            GradleDependencyKind::Module => {
                let group = component.group.as_deref().ok_or_else(invalid)?;
                let module = component.module.as_deref().ok_or_else(invalid)?;
                let version = component.version.as_deref().ok_or_else(invalid)?;
                !module.is_empty()
                    && component.build_tree_path.is_none()
                    && component.project_path.is_none()
                    && identity == ["module", group, module, version]
            }
            GradleDependencyKind::Project => {
                let build_tree_path = component.build_tree_path.as_deref().ok_or_else(invalid)?;
                let project_path = component.project_path.as_deref().ok_or_else(invalid)?;
                component.group.is_none()
                    && component.module.is_none()
                    && component.version.is_none()
                    && gradle_path_is_valid(build_tree_path, true)
                    && gradle_path_is_valid(project_path, true)
                    && identity == ["project", build_tree_path]
            }
        };
        ensure!(
            valid
                && component.selection_causes.len() <= 128
                && components
                    .insert(component.id.as_str(), component)
                    .is_none(),
            "{}",
            invalid()
        );
    }
    ensure!(
        components
            .get(graph.root_component.as_str())
            .is_some_and(|root| root.kind == GradleDependencyKind::Project
                && root.project_path.as_deref() == Some(project_path)),
        "{}",
        invalid()
    );
    let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut complete = true;
    for edge in &graph.edges {
        ensure!(
            components.contains_key(edge.from.as_str()) && !edge.origin_known,
            "{}",
            invalid()
        );
        let selector_valid = match &edge.requested {
            GradleDependencySelector::Module {
                module,
                rejected_versions,
                ..
            } => !module.is_empty() && rejected_versions.len() <= 128,
            GradleDependencySelector::Project {
                build_path,
                project_path,
            } => gradle_path_is_valid(build_path, true) && gradle_path_is_valid(project_path, true),
        };
        ensure!(selector_valid, "{}", invalid());
        if edge.resolved {
            let selected = edge.selected.as_deref().ok_or_else(invalid)?;
            ensure!(
                components.contains_key(selected)
                    && edge
                        .selected_variant
                        .as_ref()
                        .is_some_and(|variant| !variant.is_empty())
                    && edge.failure_category.is_none(),
                "{}",
                invalid()
            );
            outgoing.entry(&edge.from).or_default().push(selected);
        } else {
            complete = false;
            ensure!(
                edge.selected.is_none()
                    && edge.selected_variant.is_none()
                    && edge.failure_category.as_deref()
                        == Some(
                            "Dependency metadata unavailable. Check offline cache and build settings; refresh after repair."
                        ),
                "{}",
                invalid()
            );
        }
    }
    ensure!(graph.resolution_complete == complete, "{}", invalid());
    let mut reachable = HashSet::new();
    let mut pending = vec![graph.root_component.as_str()];
    while let Some(component) = pending.pop() {
        if reachable.insert(component)
            && let Some(children) = outgoing.get(component)
        {
            pending.extend(children.iter().copied());
        }
    }
    ensure!(reachable.len() == components.len(), "{}", invalid());
    Ok(graph)
}

fn gradle_graph_binding_path(value: &str) -> Result<PathBuf> {
    let path = file_uri_path(value)?;
    let canonical = selected_import_uri(&path, true)?;
    ensure!(
        value == canonical.as_str()
            || value.strip_prefix("file:") == canonical.as_str().strip_prefix("file://"),
        "Gradle graph URI contains an alias"
    );
    Ok(path)
}

fn graph_binding_text_is_valid(text: &str) -> bool {
    text.len() <= 4096 && !text.chars().any(char::is_control)
}

fn validate_gradle_graph_json_budget(response: &Value) -> Result<()> {
    let mut pending = vec![(response, 0usize)];
    let mut text_bytes = 0usize;
    let mut values = 0usize;
    while let Some((value, depth)) = pending.pop() {
        values += 1;
        ensure!(
            values + pending.len() <= 1_000_000 && depth <= 32,
            "Gradle graph exceeds value or depth bounds"
        );
        match value {
            Value::String(text) => {
                text_bytes = text_bytes
                    .checked_add(text.len())
                    .context("Gradle graph text limit exceeded")?;
                ensure!(
                    graph_binding_text_is_valid(text) && text_bytes <= 1024 * 1024,
                    "Gradle graph text limit exceeded"
                );
            }
            Value::Array(array) => {
                ensure!(array.len() <= 16384, "Gradle graph array limit exceeded");
                pending.extend(array.iter().map(|value| (value, depth + 1)));
            }
            Value::Object(object) => {
                ensure!(object.len() <= 64, "Gradle graph object limit exceeded");
                for (key, value) in object {
                    text_bytes = text_bytes
                        .checked_add(key.len())
                        .context("Gradle graph text limit exceeded")?;
                    ensure!(
                        graph_binding_text_is_valid(key) && text_bytes <= 1024 * 1024,
                        "Gradle graph text limit exceeded"
                    );
                    pending.push((value, depth + 1));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GradleConfiguration {
    pub name: String,
    pub description: Option<String>,
    pub resolvable: bool,
    pub consumable: bool,
    pub extends_from: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GradleSourceSet {
    pub name: String,
    pub roots: Vec<JdtlsSourceRoot>,
    pub annotation_processor_output: GradleAnnotationProcessorOutput,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum GradleAnnotationProcessorOutput {
    #[default]
    Unknown,
    Absent,
    Declared(JdtlsSourceRoot),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GradleTask {
    pub path: String,
    pub name: String,
    pub group: Option<String>,
    pub description: Option<String>,
    pub options_known: bool,
    #[serde(default)]
    pub options: Vec<GradleTaskOption>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GradleTaskOption {
    pub name: String,
    pub description: String,
    pub declared_type: String,
    pub enum_constants: Option<Vec<String>>,
}

fn gradle_enum_constants_are_valid(constants: &[String]) -> bool {
    let mut names = HashSet::new();
    constants.len() <= 128
        && constants
            .iter()
            .all(|constant| model_identifier_is_valid(constant) && names.insert(constant))
}

fn deserialize_present_gradle_field<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

impl<'de> Deserialize<'de> for GradleTaskOption {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Declaration {
            name: String,
            description: String,
            declared_type: String,
            #[serde(default, deserialize_with = "deserialize_present_gradle_field")]
            enum_constants_known: Option<bool>,
            #[serde(default, deserialize_with = "deserialize_present_gradle_field")]
            enum_constants: Option<Vec<String>>,
        }

        let declaration = Declaration::deserialize(deserializer)?;
        let enum_constants = match (declaration.enum_constants_known, declaration.enum_constants) {
            (None, None) => None,
            (Some(false), Some(constants)) if constants.is_empty() => None,
            (Some(true), Some(constants)) => {
                if !gradle_enum_constants_are_valid(&constants) {
                    return Err(serde::de::Error::custom(
                        "Gradle enum declarations are invalid, duplicated or oversized",
                    ));
                }
                Some(constants)
            }
            _ => {
                return Err(serde::de::Error::custom(
                    "Gradle enum declaration coverage is inconsistent",
                ));
            }
        };
        Ok(Self {
            name: declaration.name,
            description: declaration.description,
            declared_type: declaration.declared_type,
            enum_constants,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradleModel {
    version: u32,
    root_uri: String,
    source_sets_known: bool,
    projects: Vec<GradleModelProject>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradleModelProject {
    build_root_uri: String,
    module_uri: String,
    project_path: String,
    name: String,
    tasks: Vec<GradleTask>,
    details_json: Option<String>,
    #[serde(skip)]
    details: Option<GradleModelDetails>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradleModelDetails {
    java_source_sets_known: bool,
    source_sets: Vec<GradleModelSourceSet>,
    tasks: Vec<GradleModelTaskDetails>,
    #[serde(default)]
    configurations_known: bool,
    #[serde(default)]
    configurations: Vec<GradleConfiguration>,
    #[serde(default)]
    dependency_graph_known: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradleModelSourceSet {
    name: String,
    roots: Vec<String>,
    generatedness_known: bool,
    #[serde(default, deserialize_with = "deserialize_present_gradle_field")]
    annotation_processor_output_known: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_gradle_field")]
    annotation_processor_output_uri: Option<Option<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradleModelTaskDetails {
    path: String,
    options: Vec<GradleTaskOption>,
    options_complete: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JdtlsLibraryArtifact {
    pub coordinate: String,
    pub scope: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JdtlsLibraryFile {
    pub name: String,
    pub gradle_source_sets: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JdtlsSourceRoot {
    pub path: ProjectPath,
    pub navigable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderProjectSnapshot<Project, Metadata = ()> {
    pub server_id: LanguageServerId,
    pub worktree_id: WorktreeId,
    pub generation: u64,
    pub projects: Vec<Project>,
    pub partial: bool,
    pub metadata: Metadata,
}

pub type JdtlsProjectSnapshot = ProviderProjectSnapshot<JdtlsProject, Option<MavenSelectionFacts>>;
pub type KotlinWorkspaceSnapshot = ProviderProjectSnapshot<KotlinWorkspaceProject>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MavenSelectionFacts {
    pub imported_poms: Vec<ProjectPath>,
    pub ignored_poms: Vec<ProjectPath>,
    settings_fingerprint: Option<[u8; 32]>,
    default_inputs_fingerprint: Option<[u8; 32]>,
}

impl MavenSelectionFacts {
    pub(crate) fn from_phase(
        phase: MavenReactorPhaseFacts,
        root: &Path,
        worktree_id: WorktreeId,
        entries: &Snapshot,
        settings: Option<&settings::MavenBuildSettingsContent>,
    ) -> Result<Self> {
        Self::from_phase_with_default_inputs(phase, root, worktree_id, entries, settings, None)
    }

    pub(crate) fn from_phase_with_default_inputs(
        phase: MavenReactorPhaseFacts,
        root: &Path,
        worktree_id: WorktreeId,
        entries: &Snapshot,
        settings: Option<&settings::MavenBuildSettingsContent>,
        default_inputs: Option<&[ProjectPath]>,
    ) -> Result<Self> {
        let default_inputs_fingerprint = default_inputs
            .map(|inputs| {
                ensure!(
                    inputs == default_maven_import_inputs(entries, worktree_id)?,
                    "Default Maven inputs changed during import; import may be partial, refresh before retrying"
                );
                Ok::<_, anyhow::Error>(Self::inputs_fingerprint(inputs))
            })
            .transpose()?;
        let paths = |poms: Vec<PathBuf>| -> Result<Vec<ProjectPath>> {
            poms.into_iter()
                .map(|pom| {
                    let relative = pom
                        .strip_prefix(root)
                        .ok()
                        .and_then(|path| RelPath::new(path, PathStyle::local()).ok())
                        .context("Maven selection identity changed; refresh before retrying")?;
                    Ok(ProjectPath {
                        worktree_id,
                        path: relative.into(),
                    })
                })
                .collect()
        };
        let facts = Self {
            imported_poms: paths(phase.imported_poms)?,
            ignored_poms: paths(phase.ignored_poms)?,
            settings_fingerprint: Some(Self::settings_fingerprint(settings)?),
            default_inputs_fingerprint,
        };
        facts.validate(worktree_id, entries)?;
        Ok(facts)
    }

    fn inputs_fingerprint(inputs: &[ProjectPath]) -> [u8; 32] {
        let mut digest = Sha256::new();
        for input in inputs {
            let path = input.path.as_unix_str();
            digest.update(input.worktree_id.to_proto().to_le_bytes());
            digest.update((path.len() as u64).to_le_bytes());
            digest.update(path.as_bytes());
        }
        digest.finalize().into()
    }

    fn settings_fingerprint(
        settings: Option<&settings::MavenBuildSettingsContent>,
    ) -> Result<[u8; 32]> {
        let bytes = serde_json::to_vec(&settings).map_err(|_| {
            anyhow::anyhow!("Maven selection settings are unavailable; refresh before retrying")
        })?;
        Ok(Sha256::digest(bytes).into())
    }

    pub(crate) fn matches_settings(
        &self,
        settings: Option<&settings::MavenBuildSettingsContent>,
    ) -> Result<bool> {
        Ok(self.settings_fingerprint == Some(Self::settings_fingerprint(settings)?))
    }

    pub fn to_proto_for_version(
        &self,
        version: u32,
        worktree_id: WorktreeId,
        entries: &Snapshot,
        projects: &[JdtlsProject],
    ) -> Result<Option<proto::JdtlsMavenSelectionFacts>> {
        jdtls_snapshot_size_limit(version)?;
        if version < 7 {
            return Ok(None);
        }
        self.validate(worktree_id, entries)?;
        self.validate_membership(projects)?;
        let paths = |poms: &[ProjectPath]| {
            poms.iter()
                .map(|pom| pom.path.as_unix_str().to_owned())
                .collect()
        };
        Ok(Some(proto::JdtlsMavenSelectionFacts {
            imported_poms: paths(&self.imported_poms),
            ignored_poms: paths(&self.ignored_poms),
        }))
    }

    pub fn from_proto_for_version(
        raw: Option<proto::JdtlsMavenSelectionFacts>,
        version: u32,
        worktree_id: WorktreeId,
        entries: &Snapshot,
        projects: &[JdtlsProject],
    ) -> Result<Option<Self>> {
        jdtls_snapshot_size_limit(version)?;
        let Some(raw) = raw else { return Ok(None) };
        ensure!(
            version >= 7,
            "Remote Maven selection facts need a current host; update the host and refresh"
        );
        ensure!(
            raw.imported_poms
                .len()
                .checked_add(raw.ignored_poms.len())
                .is_some_and(|count| (1..=128).contains(&count)),
            "Remote Maven selection facts are empty or oversized; refresh before retrying"
        );
        let paths = |poms: Vec<String>| -> Result<Vec<ProjectPath>> {
            poms.into_iter()
                .map(|pom| parse_remote_path(&pom, worktree_id, entries))
                .collect()
        };
        let facts = Self {
            imported_poms: paths(raw.imported_poms)?,
            ignored_poms: paths(raw.ignored_poms)?,
            settings_fingerprint: None,
            default_inputs_fingerprint: None,
        };
        facts.validate(worktree_id, entries)?;
        facts.validate_membership(projects)?;
        Ok(Some(facts))
    }

    pub(crate) fn admit_current_snapshot(
        self,
        worktree_id: WorktreeId,
        entries: &Snapshot,
        projects: &[JdtlsProject],
        settings: Option<&settings::MavenBuildSettingsContent>,
    ) -> Result<Option<Self>> {
        ensure!(
            self.matches_settings(settings)?,
            "Saved Maven settings changed during refresh; import may be partial, review settings and refresh before retrying"
        );
        match self
            .validate(worktree_id, entries)
            .and_then(|()| self.validate_membership(projects))
        {
            Ok(()) => Ok(Some(self)),
            Err(error) => {
                // A current read can supersede the phase; retain its data without obsolete classification.
                log::debug!("Discarding obsolete Maven selection facts: {error:#}");
                Ok(None)
            }
        }
    }

    pub fn validate(&self, worktree_id: WorktreeId, entries: &Snapshot) -> Result<()> {
        if let Some(expected) = self.default_inputs_fingerprint {
            ensure!(
                expected
                    == Self::inputs_fingerprint(&default_maven_import_inputs(
                        entries,
                        worktree_id
                    )?),
                "Default Maven inputs changed; import may be partial, refresh before retrying"
            );
        }
        let mut identities = HashSet::new();
        ensure!(
            !self.imported_poms.is_empty() || !self.ignored_poms.is_empty(),
            "Maven selection facts are empty; refresh before retrying"
        );
        for pom in self.imported_poms.iter().chain(&self.ignored_poms) {
            ensure!(
                identities.len() < 128
                    && pom.worktree_id == worktree_id
                    && pom.path.as_unix_str().len() <= 4096
                    && !pom.path.as_unix_str().chars().any(char::is_control)
                    && pom.path.file_name() == Some("pom.xml")
                    && remote_path_is_visible(&pom.path, entries)
                    && entries
                        .entry_for_path(&pom.path)
                        .is_some_and(|entry| entry.is_file())
                    && identities.insert(pom),
                "Maven selection facts are stale, restricted, contradictory or oversized; refresh before retrying"
            );
        }
        Ok(())
    }

    pub(crate) fn validate_membership(&self, projects: &[JdtlsProject]) -> Result<()> {
        for (poms, imported) in [(&self.imported_poms, true), (&self.ignored_poms, false)] {
            for pom in poms {
                let parent = pom.path.parent().unwrap_or(RelPath::empty());
                let matching = projects
                    .iter()
                    .filter(|project| {
                        project.module_root.worktree_id == pom.worktree_id
                            && project.module_root.path.as_ref() == parent
                            && project.membership.is_imported()
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    if imported {
                        matching.len() == 1
                            && matching.first().is_some_and(|project| {
                                project.imported_build_kind == JdtlsImportedBuildKind::Maven
                            })
                    } else {
                        matching.is_empty()
                    },
                    "Maven selection membership is not current; import may be partial, refresh before retrying"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum JvmBuildOperation {
    MavenPhase {
        lifecycle: String,
        phase: String,
    },
    MavenGoal {
        group: String,
        artifact: String,
        version: String,
        goal: String,
    },
    GradleTask {
        path: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JvmBuildOperationReceipt {
    pub server_id: LanguageServerId,
    pub worktree_id: WorktreeId,
    pub generation: u64,
    pub module_root: ProjectPath,
    pub operations: Vec<JvmBuildOperation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JvmBuildOperationPlan {
    pub template: task::TaskTemplate,
    pub wrapper: ProjectPath,
    pub build_files: Vec<ProjectPath>,
}

impl JvmBuildOperationReceipt {
    pub fn new(
        snapshot: &JdtlsProjectSnapshot,
        current: bool,
        module_root: ProjectPath,
        operations: Vec<JvmBuildOperation>,
        path_style: PathStyle,
    ) -> Result<Self> {
        let receipt = Self {
            server_id: snapshot.server_id,
            worktree_id: snapshot.worktree_id,
            generation: snapshot.generation,
            module_root,
            operations,
        };
        receipt.plan(snapshot, current, path_style)?;
        Ok(receipt)
    }

    pub fn plan(
        &self,
        snapshot: &JdtlsProjectSnapshot,
        current: bool,
        path_style: PathStyle,
    ) -> Result<JvmBuildOperationPlan> {
        ensure!(
            current
                && self.server_id == snapshot.server_id
                && self.worktree_id == snapshot.worktree_id
                && self.module_root.worktree_id == snapshot.worktree_id
                && self.generation == snapshot.generation,
            "Build operation metadata is stale; refresh and select the operation again"
        );
        ensure!(
            !self.operations.is_empty() && self.operations.len() <= 32,
            "Select between one and 32 build operations"
        );
        let mut matching = snapshot
            .projects
            .iter()
            .filter(|project| project.module_root == self.module_root);
        let project = matching
            .next()
            .context("The selected module is unavailable; refresh and select it again")?;
        ensure!(
            matching.next().is_none() && project.membership.is_imported(),
            "The selected module is not uniquely imported; refresh before running operations"
        );
        let valid_token = |text: &str| {
            model_identifier_is_valid(text)
                && text.len() <= 4096
                && !text.starts_with('-')
                && !text.contains(':')
        };
        let mut arguments = vec!["--offline".to_string()];
        let (root, wrapper_name, build_files, label) = match project.imported_build_kind {
            JdtlsImportedBuildKind::Maven => {
                let maven = project.maven.as_ref().context(
                    "Current Maven operation metadata is unavailable; refresh the module",
                )?;
                ensure!(
                    maven.pom.worktree_id == self.worktree_id
                        && maven.pom.path.parent() == Some(self.module_root.path.as_ref())
                        && maven.pom.path.file_name() == Some("pom.xml"),
                    "The selected Maven POM does not match its module; refresh before running operations"
                );
                arguments.extend([
                    "--file".into(),
                    maven.pom.path.display(path_style).to_string(),
                ]);
                for operation in &self.operations {
                    let argument = match operation {
                        JvmBuildOperation::MavenPhase { lifecycle, phase } => {
                            ensure!(
                                valid_token(lifecycle)
                                    && valid_token(phase)
                                    && maven.lifecycles.iter().any(|known| known.id == *lifecycle
                                        && known.phases.contains(phase)),
                                "The selected Maven phase is unavailable; refresh and select a known phase"
                            );
                            phase.clone()
                        }
                        JvmBuildOperation::MavenGoal {
                            group,
                            artifact,
                            version,
                            goal,
                        } => {
                            ensure!(
                                [group, artifact, version, goal]
                                    .into_iter()
                                    .all(|text| valid_token(text))
                                    && maven.plugins.iter().any(|plugin| !plugin.managed_only
                                        && plugin.goals_known
                                        && plugin.group_id == *group
                                        && plugin.artifact_id == *artifact
                                        && plugin.version.as_ref() == Some(version)
                                        && plugin.goals.iter().any(|known| known.name == *goal)),
                                "The selected Maven goal needs a known plugin version and goal; refresh or repair provider setup"
                            );
                            format!("{group}:{artifact}:{version}:{goal}")
                        }
                        JvmBuildOperation::GradleTask { .. } => {
                            anyhow::bail!("Select operations from one build system")
                        }
                    };
                    arguments.push(argument);
                }
                (
                    RelPath::empty_arc(),
                    match path_style {
                        PathStyle::Unix => "mvnw",
                        PathStyle::Windows => "mvnw.cmd",
                    },
                    vec![maven.pom.clone()],
                    "Run Maven operations offline",
                )
            }
            JdtlsImportedBuildKind::Gradle => {
                let gradle = project
                    .gradle
                    .as_ref()
                    .context("Current Gradle task metadata is unavailable; refresh the module")?;
                ensure!(
                    gradle.build_root.worktree_id == self.worktree_id
                        && self.module_root.path.starts_with(&gradle.build_root.path),
                    "The selected Gradle build root does not own the module; refresh before running tasks"
                );
                for operation in &self.operations {
                    let JvmBuildOperation::GradleTask { path } = operation else {
                        anyhow::bail!("Select operations from one build system");
                    };
                    ensure!(
                        path.starts_with(':')
                            && path.len() <= 4096
                            && !path.chars().any(char::is_control)
                            && gradle.tasks.iter().any(|task| task.path == *path),
                        "The selected Gradle task is unavailable; refresh and select a discovered task"
                    );
                    arguments.push(path.clone());
                }
                let build_files = [
                    "settings.gradle",
                    "settings.gradle.kts",
                    "build.gradle",
                    "build.gradle.kts",
                ]
                .into_iter()
                .map(|name| {
                    Ok(ProjectPath {
                        worktree_id: self.worktree_id,
                        path: gradle
                            .build_root
                            .path
                            .join(RelPath::from_unix_str(name)?)
                            .into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
                (
                    gradle.build_root.path.clone(),
                    match path_style {
                        PathStyle::Unix => "gradlew",
                        PathStyle::Windows => "gradlew.bat",
                    },
                    build_files,
                    "Run Gradle tasks offline",
                )
            }
            JdtlsImportedBuildKind::Unknown => {
                anyhow::bail!("The selected build system is unavailable; refresh provider setup")
            }
        };
        let wrapper = ProjectPath {
            worktree_id: self.worktree_id,
            path: root.join(RelPath::from_unix_str(wrapper_name)?).into(),
        };
        ensure!(
            [root.as_unix_str(), wrapper.path.as_unix_str()]
                .into_iter()
                .all(|path| path.len() <= 4096
                    && !path.contains('$')
                    && !path.chars().any(char::is_control)),
            "This build path cannot be resolved safely; use a path without controls or dollar signs"
        );
        ensure!(
            arguments.iter().map(String::len).sum::<usize>() <= 64 * 1024
                && arguments
                    .iter()
                    .all(|text| !text.chars().any(char::is_control)),
            "Selected build arguments exceed the supported size or contain controls; reduce the selection"
        );
        let separator = match path_style {
            PathStyle::Unix => "/",
            PathStyle::Windows => "\\",
        };
        let cwd = if root.is_empty() {
            "$ZED_WORKTREE_ROOT".to_string()
        } else {
            format!("$ZED_WORKTREE_ROOT{separator}{}", root.display(path_style))
        };
        Ok(JvmBuildOperationPlan {
            template: task::TaskTemplate {
                label: label.into(),
                command: format!(
                    "$ZED_WORKTREE_ROOT{separator}{}",
                    wrapper.path.display(path_style)
                ),
                args: arguments,
                cwd: Some(cwd),
                ..Default::default()
            },
            wrapper,
            build_files,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderRefreshToken {
    server_id: LanguageServerId,
    worktree_id: WorktreeId,
    generation: u64,
}

pub type JdtlsRefreshToken = ProviderRefreshToken;

impl ProviderRefreshToken {
    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn server_id(self) -> LanguageServerId {
        self.server_id
    }

    pub fn worktree_id(self) -> WorktreeId {
        self.worktree_id
    }
}

pub struct ProviderProjectState<Project, Metadata = ()> {
    generation: u64,
    stale: bool,
    failed_import: Option<LanguageServerId>,
    pending_import_refresh: Option<LanguageServerId>,
    import_operation: Option<ProviderRefreshToken>,
    failed_query: Option<ProviderRefreshToken>,
    pending: Option<ProviderRefreshToken>,
    snapshot: Option<ProviderProjectSnapshot<Project, Metadata>>,
    pending_metadata: Metadata,
}

pub type JdtlsProjectState = ProviderProjectState<JdtlsProject, Option<MavenSelectionFacts>>;
pub type KotlinWorkspaceState = ProviderProjectState<KotlinWorkspaceProject>;

impl<Project, Metadata: Default> Default for ProviderProjectState<Project, Metadata> {
    fn default() -> Self {
        Self {
            generation: 0,
            stale: false,
            failed_import: None,
            pending_import_refresh: None,
            import_operation: None,
            failed_query: None,
            pending: None,
            snapshot: None,
            pending_metadata: Metadata::default(),
        }
    }
}

impl<Project, Metadata: Default> ProviderProjectState<Project, Metadata> {
    pub fn begin(
        &mut self,
        server_id: LanguageServerId,
        worktree_id: WorktreeId,
    ) -> Result<ProviderRefreshToken> {
        ensure!(
            self.import_operation.is_none(),
            "Java project import is in progress; wait or cancel before refreshing"
        );
        self.generation = self
            .generation
            .checked_add(1)
            .context("JVM provider project generation exhausted")?;
        let token = ProviderRefreshToken {
            server_id,
            worktree_id,
            generation: self.generation,
        };
        self.stale = true;
        self.failed_import = None;
        self.pending_import_refresh = None;
        self.failed_query = None;
        self.pending = Some(token);
        self.pending_metadata = Metadata::default();
        Ok(token)
    }

    pub fn begin_with_metadata(
        &mut self,
        server_id: LanguageServerId,
        worktree_id: WorktreeId,
        metadata: Metadata,
    ) -> Result<ProviderRefreshToken> {
        let token = self.begin(server_id, worktree_id)?;
        self.pending_metadata = metadata;
        Ok(token)
    }

    pub fn pending_metadata(&self, server_id: LanguageServerId) -> Option<&Metadata> {
        self.pending_query()
            .filter(|token| token.server_id == server_id)
            .map(|_| &self.pending_metadata)
    }

    pub fn begin_import_operation(
        &mut self,
        server_id: LanguageServerId,
        worktree_id: WorktreeId,
    ) -> Result<ProviderRefreshToken> {
        let token = self.begin(server_id, worktree_id)?;
        self.import_operation = Some(token);
        Ok(token)
    }

    pub fn import_operation(&self) -> Option<ProviderRefreshToken> {
        self.import_operation
    }

    pub fn finish_import_operation(&mut self, token: ProviderRefreshToken) -> bool {
        if self.import_operation != Some(token) || self.pending != Some(token) {
            return false;
        }
        self.import_operation = None;
        self.pending = None;
        self.pending_metadata = Metadata::default();
        true
    }

    pub fn finish(
        &mut self,
        token: ProviderRefreshToken,
        projects: Vec<Project>,
        partial: bool,
    ) -> bool {
        self.finish_with_metadata(token, projects, partial, Metadata::default())
    }

    pub fn finish_with_metadata(
        &mut self,
        token: ProviderRefreshToken,
        projects: Vec<Project>,
        partial: bool,
        metadata: Metadata,
    ) -> bool {
        if self.pending != Some(token) || self.import_operation.is_some() {
            return false;
        }
        self.pending = None;
        self.pending_metadata = Metadata::default();
        self.stale = false;
        self.failed_import = None;
        self.pending_import_refresh = None;
        self.failed_query = None;
        self.snapshot = Some(ProviderProjectSnapshot {
            server_id: token.server_id,
            worktree_id: token.worktree_id,
            generation: token.generation,
            projects,
            partial,
            metadata,
        });
        true
    }

    pub fn fail(&mut self, token: ProviderRefreshToken) -> bool {
        if self.pending != Some(token) {
            return false;
        }
        self.import_operation = None;
        self.pending = None;
        self.pending_metadata = Metadata::default();
        self.failed_query = Some(token);
        true
    }

    pub fn invalidate_server(&mut self, server_id: LanguageServerId) {
        if self
            .import_operation
            .is_some_and(|token| token.server_id == server_id)
        {
            self.import_operation = None;
        }
        if self
            .pending
            .is_some_and(|token| token.server_id == server_id)
        {
            self.pending = None;
            self.pending_metadata = Metadata::default();
        }
        if self
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.server_id == server_id)
        {
            self.snapshot = None;
        }
        if self
            .failed_query
            .is_some_and(|token| token.server_id == server_id)
        {
            self.failed_query = None;
        }
        if self.failed_import == Some(server_id) {
            self.failed_import = None;
        }
        if self.pending_import_refresh == Some(server_id) {
            self.pending_import_refresh = None;
        }
    }

    pub fn invalidate_worktree(&mut self, worktree_id: WorktreeId) {
        if self
            .import_operation
            .is_some_and(|token| token.worktree_id == worktree_id)
        {
            self.import_operation = None;
        }
        if self
            .pending
            .is_some_and(|token| token.worktree_id == worktree_id)
        {
            self.pending = None;
            self.pending_metadata = Metadata::default();
        }
        if self
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.worktree_id == worktree_id)
        {
            self.snapshot = None;
        }
        if self
            .failed_query
            .is_some_and(|token| token.worktree_id == worktree_id)
        {
            self.failed_query = None;
        }
        self.failed_import = None;
        self.pending_import_refresh = None;
    }

    pub fn invalidate_remote_connection(&mut self, worktree_id: WorktreeId) -> Result<()> {
        self.invalidate_worktree(worktree_id);
        self.mark_stale()
    }

    pub fn mark_stale(&mut self) -> Result<()> {
        self.import_operation = None;
        self.pending = None;
        self.pending_metadata = Metadata::default();
        self.stale = true;
        self.failed_import = None;
        self.pending_import_refresh = None;
        self.generation = self
            .generation
            .checked_add(1)
            .context("JVM provider project generation exhausted")?;
        Ok(())
    }

    pub fn mark_import_failed(&mut self, server_id: LanguageServerId) -> Result<()> {
        self.mark_stale()?;
        self.failed_import = Some(server_id);
        Ok(())
    }

    pub fn mark_import_succeeded(&mut self, server_id: LanguageServerId) -> Result<()> {
        self.mark_stale()?;
        self.pending_import_refresh = Some(server_id);
        Ok(())
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn snapshot(&self) -> Option<&ProviderProjectSnapshot<Project, Metadata>> {
        self.snapshot.as_ref()
    }

    pub fn snapshot_is_current(&self) -> bool {
        !self.stale
            && self.pending.is_none()
            && self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.generation == self.generation)
    }

    pub fn is_refreshing(&self) -> bool {
        self.pending.is_some()
    }

    pub fn pending_query(&self) -> Option<ProviderRefreshToken> {
        self.pending.filter(|_| self.import_operation.is_none())
    }

    pub fn failed_query(&self) -> Option<ProviderRefreshToken> {
        self.failed_query
    }

    pub fn failed_import(&self) -> Option<LanguageServerId> {
        self.failed_import
    }

    pub fn pending_import_refresh(&self) -> Option<LanguageServerId> {
        self.pending_import_refresh
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct JdtlsProjectSettings {
    #[serde(skip)]
    membership: JdtlsProjectMembership,
    #[serde(skip)]
    imported_build_kind: JdtlsImportedBuildKind,
    #[serde(skip)]
    imported_build_kinds_incomplete: bool,
    #[serde(skip)]
    imported_resources_incomplete: bool,
    #[serde(rename = "org.eclipse.jdt.ls.core.sourcePaths")]
    source_paths: Option<Vec<String>>,
    #[serde(rename = "org.eclipse.jdt.ls.core.classpathEntries")]
    classpath_entries: Option<Vec<JdtlsClasspathEntry>>,
    #[serde(rename = "org.eclipse.jdt.ls.core.vm.location")]
    project_vm_location: Option<String>,
    #[serde(rename = "org.eclipse.jdt.core.compiler.source")]
    java_source_level: Option<String>,
    #[serde(rename = "org.eclipse.jdt.core.compiler.codegen.targetPlatform")]
    java_target_level: Option<String>,
    #[serde(skip)]
    project_vm_java_major: Option<u32>,
    #[serde(skip)]
    server_launch_java_major: Option<u32>,
    #[serde(skip)]
    gradle: Option<GradleModelProject>,
    #[serde(skip)]
    maven: Option<MavenModelProject>,
    #[serde(skip)]
    maven_metadata_incomplete: bool,
}

fn jdtls_launch_java_home(binary: &LanguageServerBinary) -> Option<&Path> {
    let command = binary.path.file_name()?.to_str()?;
    let java = match command {
        "java" | "java.exe" => binary.path.as_path(),
        // Java extension 6.8.27 gives the proxy its working directory first,
        // then either Java or a configured JDTLS launcher. A launcher does not
        // identify the JVM it will eventually select.
        "java-lsp-proxy" | "java-lsp-proxy.exe" => Path::new(binary.arguments.get(1)?),
        _ => return None,
    };
    if !binary
        .arguments
        .iter()
        .any(|argument| argument == "-Declipse.application=org.eclipse.jdt.ls.core.id1")
        || !java.is_absolute()
        || !matches!(java.file_name()?.to_str()?, "java" | "java.exe")
    {
        return None;
    }
    let bin = java.parent()?;
    if bin.file_name()?.to_str()? != "bin" {
        return None;
    }
    bin.parent()
}

fn parse_java_compatibility_level(value: &str) -> Option<u32> {
    let version = value.strip_prefix("1.").unwrap_or(value);
    if version.is_empty() || version.len() > 2 || !version.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let version = version.parse::<u32>().ok()?;
    (1..=99).contains(&version).then_some(version)
}

fn project_vm_java_major_from_location(location: &str) -> Option<u32> {
    if location.is_empty() || location.len() > 4096 {
        return None;
    }
    let mut release = String::new();
    std::fs::File::open(Path::new(location).join("release"))
        .ok()?
        .take(4096)
        .read_to_string(&mut release)
        .ok()?;
    java_major_from_release(&release)
}

pub fn java_major_from_release(release: &str) -> Option<u32> {
    if release.len() > 4096 {
        return None;
    }
    let version = release
        .lines()
        .find_map(|line| line.strip_prefix("JAVA_VERSION="))?
        .trim_matches('"');
    let major = version.split('.').next()?;
    let major = if major == "1" {
        version.split('.').nth(1)?
    } else {
        major
    };
    parse_java_compatibility_level(major)
}

#[derive(Debug, Deserialize)]
struct JdtlsClasspathEntry {
    kind: i32,
    path: Option<String>,
    attributes: Option<serde_json::Value>,
}

pub fn project_from_settings(
    module_uri: &str,
    settings: JdtlsProjectSettings,
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
) -> Result<(JdtlsProject, bool)> {
    if matches!(
        settings.membership,
        JdtlsProjectMembership::ImportedNonJava | JdtlsProjectMembership::MetadataOnly
    ) {
        ensure!(
            settings.source_paths.as_ref().is_none_or(Vec::is_empty)
                && settings
                    .classpath_entries
                    .as_ref()
                    .is_none_or(Vec::is_empty)
                && settings.project_vm_location.is_none()
                && settings.project_vm_java_major.is_none()
                && settings.java_source_level.is_none()
                && settings.java_target_level.is_none(),
            "Non-Java resource contains unqualified Java semantics"
        );
    }
    ensure!(
        settings.membership.is_imported()
            || settings.imported_build_kind == JdtlsImportedBuildKind::Unknown,
        "JVM build ownership lacks imported resource evidence"
    );
    ensure!(
        module_uri.len() <= 4096 && !module_uri.chars().any(char::is_control),
        "JDTLS module identity exceeds its bound or contains controls"
    );
    let module_path = file_uri_path(module_uri)?;
    let module_root = confined_project_path(&module_path, worktree_root, worktree_id, path_style)
        .context("JDTLS returned a module outside the current worktree")?;
    let mut partial = settings.source_paths.is_none()
        || settings.classpath_entries.is_none()
        || settings.maven_metadata_incomplete
        || settings.imported_resources_incomplete
        || settings.imported_build_kinds_incomplete;
    let gradle = settings
        .gradle
        .map(|project| {
            let build_path = file_uri_path(&project.build_root_uri)?;
            let build_root =
                confined_project_path(&build_path, worktree_root, worktree_id, path_style)
                    .context("Gradle build is outside the current worktree")?;
            ensure!(
                module_path.starts_with(&build_path),
                "Gradle module is outside its build"
            );
            partial = true;
            let (source_sets_known, source_records, configurations_known, configurations) = project
                .details
                .map(|details| {
                    (
                        details.java_source_sets_known,
                        details.source_sets,
                        details.configurations_known,
                        details.configurations,
                    )
                })
                .unwrap_or_default();
            let source_sets = source_records
                .into_iter()
                .map(|source| {
                    let roots = source
                        .roots
                        .into_iter()
                        .map(|uri| {
                            let path = file_uri_path(&uri)?;
                            let path = confined_project_path(
                                &path,
                                worktree_root,
                                worktree_id,
                                path_style,
                            )
                            .context("Gradle source root is outside the current worktree")?;
                            Ok(JdtlsSourceRoot {
                                path,
                                navigable: false,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(GradleSourceSet {
                        name: source.name,
                        roots,
                        annotation_processor_output: match (
                            source.annotation_processor_output_known,
                            source.annotation_processor_output_uri.flatten(),
                        ) {
                            (Some(true), Some(uri)) => {
                                let path = confined_project_path(
                                    &file_uri_path(&uri)?,
                                    worktree_root,
                                    worktree_id,
                                    path_style,
                                )
                                .context(
                                    "Gradle compiler output is outside the current worktree",
                                )?;
                                GradleAnnotationProcessorOutput::Declared(JdtlsSourceRoot {
                                    path,
                                    navigable: false,
                                })
                            }
                            (Some(true), None) => GradleAnnotationProcessorOutput::Absent,
                            _ => GradleAnnotationProcessorOutput::Unknown,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GradleProject {
                dependency_graph: None,
                build_root,
                project_path: project.project_path,
                tasks: project.tasks,
                source_sets_known,
                source_sets,
                configurations_known,
                configurations,
            })
        })
        .transpose()?;
    let maven = settings
        .maven
        .map(|project| -> Result<MavenProject> {
            let pom = confined_project_path(
                &file_uri_path(&project.pom_uri)?,
                worktree_root,
                worktree_id,
                path_style,
            )
            .context("Maven POM is outside the current worktree")?;
            let modules = project
                .module_uris
                .iter()
                .map(|uri| {
                    confined_project_path(
                        &file_uri_path(uri)?,
                        worktree_root,
                        worktree_id,
                        path_style,
                    )
                    .context("Maven module is outside the current worktree")
                })
                .collect::<Result<Vec<_>>>()?;
            partial = true;
            Ok(MavenProject {
                dependency_graph: None,
                pom,
                modules,
                group_id: project.group_id,
                artifact_id: project.artifact_id,
                version: project.version,
                packaging: project.packaging,
                profiles: project.profiles,
                profile_definitions_known: project.inherited_profile_definitions_known,
                profile_definitions: project
                    .profile_definitions
                    .unwrap_or_default()
                    .into_iter()
                    .map(|definition| {
                        Ok(MavenProfileDefinition {
                            id: definition.id,
                            source_group_id: definition.source_group_id,
                            source_artifact_id: definition.source_artifact_id,
                            source_version: definition.source_version,
                            source_pom: definition
                                .source_pom_uri
                                .map(|uri| {
                                    confined_project_path(
                                        &file_uri_path(&uri)?,
                                        worktree_root,
                                        worktree_id,
                                        path_style,
                                    )
                                    .context("Maven profile origin is outside the worktree")
                                })
                                .transpose()?,
                            inherited: definition.inherited,
                            active_at_source: definition.active_at_source,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                reactor_position: project.reactor_position,
                reactor_order_problem: project.reactor_order_problem,
                plugins: project.plugins,
                lifecycles: project.lifecycles,
            })
        })
        .transpose()?;
    let java_source_level = settings
        .java_source_level
        .as_deref()
        .and_then(parse_java_compatibility_level);
    partial |= settings.java_source_level.is_some() && java_source_level.is_none();
    let java_target_level = settings
        .java_target_level
        .as_deref()
        .and_then(parse_java_compatibility_level);
    partial |= settings.java_target_level.is_some() && java_target_level.is_none();
    let source_paths = settings.source_paths.unwrap_or_default();
    ensure!(
        source_paths.len() <= MAX_JDTLS_SOURCE_ROOTS,
        "JDTLS returned too many source roots"
    );
    let mut source_roots = BTreeSet::new();
    for source_path in source_paths {
        let Some(source) = confined_project_path(
            Path::new(&source_path),
            worktree_root,
            worktree_id,
            path_style,
        ) else {
            partial = true;
            continue;
        };
        source_roots.insert(source);
    }

    let classpath_entries = settings.classpath_entries.unwrap_or_default();
    ensure!(
        classpath_entries.len() <= MAX_JDTLS_CLASSPATH_ENTRIES,
        "JDTLS returned too many classpath entries"
    );
    let mut project_references = BTreeSet::new();
    let mut library_entries = 0;
    let mut library_files = Vec::new();
    let mut library_artifacts = Vec::new();
    for entry in classpath_entries {
        match entry.kind {
            1 => {
                library_entries += 1;
                if let Some(attributes) = entry.attributes.as_ref()
                    && has_maven_artifact_attributes(attributes)
                {
                    if let Some(artifact) = jdtls_maven_artifact(attributes) {
                        partial |=
                            attributes.get("maven.scope").is_some() && artifact.scope.is_none();
                        library_artifacts.push(artifact);
                        continue;
                    }
                    partial = true;
                }
                if let Some(file_name) = entry.path.as_deref().and_then(|path| {
                    external_library_file_name(path, worktree_root, worktree_id, path_style)
                }) {
                    let gradle_source_sets = match entry
                        .attributes
                        .as_ref()
                        .and_then(|attributes| attributes.get("gradle_used_by_scope"))
                    {
                        Some(value) => match value.as_str().and_then(parse_gradle_source_sets) {
                            Some(source_sets) => source_sets,
                            None => {
                                partial = true;
                                Vec::new()
                            }
                        },
                        None => Vec::new(),
                    };
                    library_files.push(JdtlsLibraryFile {
                        name: file_name.to_string(),
                        gradle_source_sets,
                    });
                }
            }
            2 => {
                if let Some(reference) = entry.path.as_deref().and_then(virtual_project_name) {
                    project_references.insert(reference.to_string());
                } else {
                    partial = true;
                }
            }
            _ => {}
        }
    }
    library_files.sort_unstable_by(|left, right| {
        (&left.name, &left.gradle_source_sets).cmp(&(&right.name, &right.gradle_source_sets))
    });
    library_artifacts.sort_unstable_by(|left, right| {
        (&left.coordinate, &left.scope).cmp(&(&right.coordinate, &right.scope))
    });
    Ok((
        JdtlsProject {
            module_root,
            membership: settings.membership,
            imported_build_kind: settings.imported_build_kind,
            source_roots: source_roots
                .into_iter()
                .map(|path| JdtlsSourceRoot {
                    path,
                    navigable: false,
                })
                .collect(),
            project_references: project_references.into_iter().collect(),
            library_entries,
            library_files,
            library_artifacts,
            project_vm_known: settings
                .project_vm_location
                .is_some_and(|location| !location.is_empty()),
            project_vm_java_major: settings.project_vm_java_major,
            server_launch_java_major: settings.server_launch_java_major,
            java_source_level,
            java_target_level,
            gradle,
            maven,
        },
        partial,
    ))
}

pub fn check_module_count(count: usize) -> Result<()> {
    ensure!(
        count <= MAX_JDTLS_MODULES,
        "JDTLS returned too many modules"
    );
    Ok(())
}

pub fn projects_from_provider(
    raw_projects: Vec<(String, JdtlsProjectSettings)>,
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
    snapshot: &Snapshot,
    sharing: bool,
) -> (Vec<JdtlsProject>, bool) {
    if check_module_count(raw_projects.len()).is_err() {
        log::warn!("JDTLS project admission exceeded the module bound; check provider setup");
        return (Vec::new(), true);
    }
    let mut projects = Vec::new();
    let mut identities = HashSet::new();
    let mut partial = raw_projects.is_empty();
    for (uri, settings) in raw_projects {
        match project_from_settings(&uri, settings, worktree_root, worktree_id, path_style) {
            Ok((project, project_partial)) => {
                if !identities.insert(project.module_root.clone()) {
                    partial = true;
                    continue;
                }
                partial |= project_partial;
                projects.push(project);
            }
            Err(error) => {
                log::debug!("Ignoring invalid JDTLS project response: {error:#}");
                partial = true;
            }
        }
    }
    filter_provider_projects(projects, partial, snapshot, sharing)
}

pub fn filter_provider_projects(
    projects: Vec<JdtlsProject>,
    mut partial: bool,
    snapshot: &Snapshot,
    sharing: bool,
) -> (Vec<JdtlsProject>, bool) {
    let mut visible_projects = Vec::with_capacity(projects.len());
    for mut project in projects {
        let module_visible = remote_path_is_visible(&project.module_root.path, snapshot)
            && snapshot
                .entry_for_path(&project.module_root.path)
                .is_some_and(|entry| entry.is_dir());
        if !module_visible {
            partial = true;
            continue;
        }
        let mut filter_source = |source: &mut JdtlsSourceRoot| {
            if source.path.path.ancestors().any(|ancestor| {
                snapshot
                    .entry_for_path(ancestor)
                    .is_some_and(|entry| entry.is_private || entry.is_external || entry.is_fifo)
            }) {
                partial = true;
                return false;
            }
            match snapshot.entry_for_path(source.path.path.as_ref()) {
                Some(_) => {
                    source.navigable = true;
                    true
                }
                None => {
                    partial = true;
                    !sharing
                }
            }
        };
        project.source_roots.retain_mut(&mut filter_source);
        if let Some(gradle) = &mut project.gradle {
            for source in &mut gradle.source_sets {
                source.roots.retain_mut(&mut filter_source);
                if let GradleAnnotationProcessorOutput::Declared(output) =
                    &mut source.annotation_processor_output
                    && !filter_source(output)
                {
                    source.annotation_processor_output = GradleAnnotationProcessorOutput::Unknown;
                }
            }
            if sharing && !remote_path_is_visible(&gradle.build_root.path, snapshot) {
                project.gradle = None;
                partial = true;
            }
        }
        if let Some(maven) = &mut project.maven {
            let hidden = |path: &ProjectPath| {
                path.path.ancestors().any(|ancestor| {
                    snapshot
                        .entry_for_path(ancestor)
                        .is_some_and(|entry| entry.is_private || entry.is_external || entry.is_fifo)
                })
            };
            if hidden(&maven.pom) || (sharing && !remote_path_is_visible(&maven.pom.path, snapshot))
            {
                project.maven = None;
                partial = true;
            } else {
                maven.modules.retain(|module| {
                    if hidden(module) {
                        partial = true;
                        return false;
                    }
                    if snapshot.entry_for_path(module.path.as_ref()).is_none() {
                        partial = true;
                        return !sharing;
                    }
                    true
                });
                maven.profile_definitions.retain(|definition| {
                    if let Some(source) = &definition.source_pom
                        && (hidden(source)
                            || (sharing && snapshot.entry_for_path(source.path.as_ref()).is_none()))
                    {
                        maven.profile_definitions_known = false;
                        partial = true;
                        return false;
                    }
                    true
                });
                partial |= snapshot.entry_for_path(maven.pom.path.as_ref()).is_none();
            }
        }
        visible_projects.push(project);
    }
    let mut projects = visible_projects;
    if sharing {
        let visible_names = projects
            .iter()
            .filter_map(|project| project.module_root.path.file_name())
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        for project in &mut projects {
            let previous_count = project.project_references.len();
            project
                .project_references
                .retain(|reference| visible_names.contains(reference.as_str()));
            partial |= project.project_references.len() != previous_count;
        }
    }
    (projects, partial)
}

pub fn project_to_proto(project: &JdtlsProject) -> proto::JdtlsProjectInfo {
    proto::JdtlsProjectInfo {
        module_root: project.module_root.path.as_unix_str().to_string(),
        membership: project.membership.to_proto().into(),
        imported_build_kind: project.imported_build_kind.to_proto().into(),
        source_roots: project
            .source_roots
            .iter()
            .map(|source| proto::JdtlsSourceInfo {
                path: source.path.path.as_unix_str().to_string(),
                navigable: source.navigable,
            })
            .collect(),
        project_references: project.project_references.clone(),
        library_entries: project.library_entries as u32,
        project_vm_known: project.project_vm_known,
        project_vm_java_major: project.project_vm_java_major.unwrap_or_default(),
        server_launch_java_major: project.server_launch_java_major.unwrap_or_default(),
        java_source_level: project.java_source_level.unwrap_or_default(),
        java_target_level: project.java_target_level.unwrap_or_default(),
        library_file_names: project
            .library_files
            .iter()
            .map(|file| file.name.clone())
            .collect(),
        library_artifacts: project
            .library_artifacts
            .iter()
            .map(|artifact| proto::JdtlsLibraryArtifactInfo {
                coordinate: artifact.coordinate.clone(),
                scope: artifact.scope.clone().unwrap_or_default(),
            })
            .collect(),
        gradle: project
            .gradle
            .as_ref()
            .map(|gradle| proto::JdtlsGradleProjectInfo {
                dependency_graph_json: None,
                build_root: gradle.build_root.path.as_unix_str().to_string(),
                project_path: gradle.project_path.clone(),
                configurations_known: gradle.configurations_known,
                configurations: gradle
                    .configurations
                    .iter()
                    .map(|configuration| proto::JdtlsGradleConfigurationInfo {
                        name: configuration.name.clone(),
                        description: configuration.description.clone(),
                        resolvable: configuration.resolvable,
                        consumable: configuration.consumable,
                        extends_from: configuration.extends_from.clone(),
                    })
                    .collect(),
                source_sets_known: gradle.source_sets_known,
                source_sets: gradle
                    .source_sets
                    .iter()
                    .map(|source| proto::JdtlsGradleSourceSetInfo {
                        name: source.name.clone(),
                        annotation_processor_output_known: Some(!matches!(
                            source.annotation_processor_output,
                            GradleAnnotationProcessorOutput::Unknown
                        )),
                        annotation_processor_output: match &source.annotation_processor_output {
                            GradleAnnotationProcessorOutput::Declared(output) => {
                                Some(proto::JdtlsSourceInfo {
                                    path: output.path.path.as_unix_str().to_string(),
                                    navigable: output.navigable,
                                })
                            }
                            _ => None,
                        },
                        roots: source
                            .roots
                            .iter()
                            .map(|root| proto::JdtlsSourceInfo {
                                path: root.path.path.as_unix_str().to_string(),
                                navigable: root.navigable,
                            })
                            .collect(),
                    })
                    .collect(),
                tasks: gradle
                    .tasks
                    .iter()
                    .map(|task| proto::JdtlsGradleTaskInfo {
                        path: task.path.clone(),
                        name: task.name.clone(),
                        group: task.group.clone(),
                        description: task.description.clone(),
                        options_known: task.options_known,
                        options: task
                            .options
                            .iter()
                            .map(|option| proto::JdtlsGradleTaskOptionInfo {
                                name: option.name.clone(),
                                description: option.description.clone(),
                                declared_type: option.declared_type.clone(),
                                enum_constants_known: option.enum_constants.as_ref().map(|_| true),
                                enum_constants: option.enum_constants.clone().unwrap_or_default(),
                            })
                            .collect(),
                    })
                    .collect(),
            }),
        maven: project
            .maven
            .as_ref()
            .map(|maven| proto::JdtlsMavenProjectInfo {
                dependency_graph_json: None,
                pom: maven.pom.path.as_unix_str().to_string(),
                modules: maven
                    .modules
                    .iter()
                    .map(|module| module.path.as_unix_str().to_string())
                    .collect(),
                group_id: maven.group_id.clone(),
                artifact_id: maven.artifact_id.clone(),
                version: maven.version.clone(),
                packaging: maven.packaging.clone(),
                profiles: maven
                    .profiles
                    .iter()
                    .map(|(id, active)| proto::JdtlsMavenProfileInfo {
                        id: id.clone(),
                        active: *active,
                    })
                    .collect(),
                profile_definitions_known: maven.profile_definitions_known,
                profile_definitions: maven
                    .profile_definitions
                    .iter()
                    .map(|definition| proto::JdtlsMavenProfileDefinitionInfo {
                        id: definition.id.clone(),
                        source_group_id: definition.source_group_id.clone(),
                        source_artifact_id: definition.source_artifact_id.clone(),
                        source_version: definition.source_version.clone(),
                        source_pom: definition
                            .source_pom
                            .as_ref()
                            .map(|path| path.path.as_unix_str().to_string()),
                        inherited: definition.inherited,
                        active_at_source: definition.active_at_source,
                    })
                    .collect(),
                reactor_position: maven.reactor_position.map(|position| position as u32),
                reactor_order_problem: maven.reactor_order_problem.clone(),
                plugins: maven
                    .plugins
                    .iter()
                    .map(|plugin| proto::JdtlsMavenPluginInfo {
                        group_id: plugin.group_id.clone(),
                        artifact_id: plugin.artifact_id.clone(),
                        version: plugin.version.clone(),
                        managed_only: plugin.managed_only,
                        goals_known: plugin.goals_known,
                        goal_prefix: plugin.goal_prefix.clone(),
                        goals: plugin
                            .goals
                            .iter()
                            .map(|goal| proto::JdtlsMavenGoalInfo {
                                name: goal.name.clone(),
                                phase: goal.phase.clone(),
                                description: goal.description.clone(),
                                parameters_known: goal.parameters_known,
                                parameters: goal
                                    .parameters
                                    .iter()
                                    .map(|parameter| proto::JdtlsMavenGoalParameterInfo {
                                        name: parameter.name.clone(),
                                        alias: parameter.alias.clone(),
                                        declared_type: parameter.declared_type.clone(),
                                        required: parameter.required,
                                        editable: parameter.editable,
                                        description: parameter.description.clone(),
                                    })
                                    .collect(),
                            })
                            .collect(),
                    })
                    .collect(),
                lifecycles: maven
                    .lifecycles
                    .iter()
                    .map(|lifecycle| proto::JdtlsMavenLifecycleInfo {
                        id: lifecycle.id.clone(),
                        phases: lifecycle.phases.clone(),
                    })
                    .collect(),
            }),
        library_source_sets: project
            .library_files
            .iter()
            .enumerate()
            .filter(|(_, file)| !file.gradle_source_sets.is_empty())
            .map(|(file_index, file)| proto::JdtlsLibrarySourceSetInfo {
                file_index: file_index as u32,
                source_sets: file.gradle_source_sets.clone(),
            })
            .collect(),
    }
}

pub fn project_to_proto_for_version(
    project: &JdtlsProject,
    version: u32,
) -> Result<proto::JdtlsProjectInfo> {
    jdtls_snapshot_size_limit(version)?;
    let mut projected = project_to_proto(project);
    if let Some(gradle) = projected.gradle.as_mut() {
        if version < 8 {
            for source in &mut gradle.source_sets {
                source.annotation_processor_output_known = None;
                source.annotation_processor_output = None;
            }
        }
        for option in gradle.tasks.iter_mut().flat_map(|task| &mut task.options) {
            if version < 6 {
                option.enum_constants_known = None;
                option.enum_constants.clear();
            } else {
                ensure!(
                    gradle_enum_constants_are_valid(&option.enum_constants),
                    "Gradle enum declarations are invalid, duplicated or oversized"
                );
            }
        }
    }
    if version >= 5 {
        if let (Some(metadata), Some(graph)) = (
            projected.maven.as_mut(),
            project
                .maven
                .as_ref()
                .and_then(|maven| maven.dependency_graph.as_ref()),
        ) {
            let value =
                serde_json::to_value(graph.as_ref()).context("Maven graph encoding failed")?;
            decode_maven_dependency_graph(value.clone())?;
            let text = serde_json::to_string(&value).context("Maven graph encoding failed")?;
            ensure!(
                text.len() <= 8 * 1_048_576,
                "Maven graph exceeds the remote size limit"
            );
            metadata.dependency_graph_json = Some(text);
        }
        if let (Some(metadata), Some(graph)) = (
            projected.gradle.as_mut(),
            project
                .gradle
                .as_ref()
                .and_then(|gradle| gradle.dependency_graph.as_ref()),
        ) {
            let value =
                serde_json::to_value(graph.as_ref()).context("Gradle graph encoding failed")?;
            validate_gradle_graph_json_budget(&value)?;
            decode_gradle_resolution_graph(
                value["resolution"].clone(),
                &graph.resolution.configuration,
                &graph.project_path,
            )?;
            let text = serde_json::to_string(&value).context("Gradle graph encoding failed")?;
            ensure!(
                text.len() <= 2 * 1_048_576,
                "Gradle graph exceeds the remote size limit"
            );
            metadata.dependency_graph_json = Some(text);
        }
    }
    if version < 4 {
        projected.imported_build_kind = proto::JdtlsImportedBuildKind::Unknown.into();
    }
    if version < 3 {
        projected.membership = proto::JdtlsProjectMembership::Unknown.into();
    }
    if version == 1 {
        projected.gradle = None;
        projected.maven = None;
    }
    ensure!(
        projected.encoded_len() <= jdtls_snapshot_size_limit(version)?,
        "Java project snapshot exceeds the size limit"
    );
    Ok(projected)
}

pub fn projects_from_proto_for_version(
    raw_projects: Vec<proto::JdtlsProjectInfo>,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
    version: u32,
) -> Result<Vec<JdtlsProject>> {
    let limit = jdtls_snapshot_size_limit(version)?;
    let size = raw_projects
        .iter()
        .try_fold(0usize, |size, project| {
            size.checked_add(project.encoded_len())
        })
        .context("Remote Java project snapshot size overflow")?;
    ensure!(
        size <= limit
            && (version >= 5
                || raw_projects.iter().all(|project| project
                    .maven
                    .as_ref()
                    .is_none_or(|maven| maven.dependency_graph_json.is_none())
                    && project
                        .gradle
                        .as_ref()
                        .is_none_or(|gradle| gradle.dependency_graph_json.is_none()))),
        "Remote Java project snapshot protocol mismatch or size limit exceeded"
    );
    ensure!(
        version >= 6
            || raw_projects.iter().all(|project| {
                project.gradle.as_ref().is_none_or(|gradle| {
                    gradle
                        .tasks
                        .iter()
                        .flat_map(|task| &task.options)
                        .all(|option| {
                            option.enum_constants_known.is_none()
                                && option.enum_constants.is_empty()
                        })
                })
            }),
        "Remote Gradle enum declarations require a matching snapshot protocol"
    );
    ensure!(
        version >= 8
            || raw_projects.iter().all(|project| {
                project.gradle.as_ref().is_none_or(|gradle| {
                    gradle.source_sets.iter().all(|source| {
                        source.annotation_processor_output_known.is_none()
                            && source.annotation_processor_output.is_none()
                    })
                })
            }),
        "Remote Gradle compiler outputs require a matching snapshot protocol"
    );
    projects_from_proto(raw_projects, worktree_id, snapshot)
}

pub fn projects_from_proto(
    raw_projects: Vec<proto::JdtlsProjectInfo>,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
) -> Result<Vec<JdtlsProject>> {
    check_module_count(raw_projects.len())?;
    let encoded_bytes = raw_projects
        .iter()
        .try_fold(0usize, |bytes, project| {
            bytes.checked_add(project.encoded_len())
        })
        .context("Remote Java project snapshot size overflow")?;
    ensure!(
        encoded_bytes <= jdtls_snapshot_size_limit(JDTLS_SNAPSHOT_PROTOCOL_VERSION)?,
        "Remote Java project snapshot exceeds the size limit"
    );

    let mut modules = HashSet::new();
    let mut gradle_identities = HashSet::new();
    let mut reactor_positions = HashSet::new();
    let mut task_count = 0usize;
    let mut goal_count = 0usize;
    raw_projects
        .into_iter()
        .map(|project| {
            let module_root = parse_remote_path(&project.module_root, worktree_id, snapshot)?;
            ensure!(
                snapshot
                    .entry_for_path(&module_root.path)
                    .is_some_and(|entry| entry.is_dir()),
                "Remote JVM module identity is not a project directory"
            );
            let membership = match proto::JdtlsProjectMembership::from_i32(project.membership)
                .context("Remote JVM imported membership is invalid")?
            {
                proto::JdtlsProjectMembership::Unknown => JdtlsProjectMembership::Unknown,
                proto::JdtlsProjectMembership::MetadataOnly => JdtlsProjectMembership::MetadataOnly,
                proto::JdtlsProjectMembership::ImportedJava => JdtlsProjectMembership::ImportedJava,
                proto::JdtlsProjectMembership::ImportedNonJava => {
                    JdtlsProjectMembership::ImportedNonJava
                }
            };
            let imported_build_kind =
                match proto::JdtlsImportedBuildKind::from_i32(project.imported_build_kind)
                    .context("Remote JVM imported build kind is invalid")?
                {
                    proto::JdtlsImportedBuildKind::Unknown => JdtlsImportedBuildKind::Unknown,
                    proto::JdtlsImportedBuildKind::Maven => JdtlsImportedBuildKind::Maven,
                    proto::JdtlsImportedBuildKind::Gradle => JdtlsImportedBuildKind::Gradle,
                };
            ensure!(
                membership.is_imported() || imported_build_kind == JdtlsImportedBuildKind::Unknown,
                "Remote JVM build ownership lacks imported resource evidence"
            );
            if matches!(
                membership,
                JdtlsProjectMembership::ImportedNonJava | JdtlsProjectMembership::MetadataOnly
            ) {
                ensure!(
                    project.source_roots.is_empty()
                        && project.project_references.is_empty()
                        && project.library_entries == 0
                        && project.library_file_names.is_empty()
                        && project.library_artifacts.is_empty()
                        && project.library_source_sets.is_empty()
                        && !project.project_vm_known
                        && project.project_vm_java_major == 0
                        && project.java_source_level == 0
                        && project.java_target_level == 0,
                    "Remote non-Java resource contains unqualified Java semantics"
                );
            }
            ensure!(
                modules.insert(module_root.clone()),
                "Remote Java modules are duplicated"
            );
            let gradle = project
                .gradle
                .map(|gradle| {
                    let gradle = gradle_from_proto(gradle, &module_root, worktree_id, snapshot)?;
                    ensure!(
                        gradle_identities
                            .insert((gradle.build_root.clone(), gradle.project_path.clone())),
                        "Remote Gradle project identity is duplicated"
                    );
                    task_count = task_count
                        .checked_add(gradle.tasks.len())
                        .context("Remote Gradle task count exceeded")?;
                    ensure!(task_count <= 4096, "Remote Gradle task count exceeded");
                    Ok::<_, anyhow::Error>(gradle)
                })
                .transpose()?;
            let maven = project
                .maven
                .map(|maven| {
                    let maven = maven_from_proto(maven, &module_root, worktree_id, snapshot)?;
                    if let Some(position) = maven.reactor_position {
                        ensure!(
                            reactor_positions.insert(position),
                            "Remote Maven reactor position is duplicated"
                        );
                    }
                    for plugin in &maven.plugins {
                        goal_count = goal_count
                            .checked_add(plugin.goals.len())
                            .context("Remote Maven goal count exceeded")?;
                    }
                    ensure!(goal_count <= 4096, "Remote Maven goal count exceeded");
                    Ok::<_, anyhow::Error>(maven)
                })
                .transpose()?;
            ensure!(
                (maven
                    .as_ref()
                    .is_none_or(|maven| maven.dependency_graph.is_none())
                    || membership.is_imported()
                        && imported_build_kind == JdtlsImportedBuildKind::Maven)
                    && (gradle
                        .as_ref()
                        .is_none_or(|gradle| gradle.dependency_graph.is_none())
                        || membership.is_imported()
                            && imported_build_kind == JdtlsImportedBuildKind::Gradle),
                "Remote dependency graph lacks imported provider ownership"
            );
            ensure!(
                project.source_roots.len() <= MAX_JDTLS_SOURCE_ROOTS,
                "remote JDTLS snapshot has too many source roots"
            );
            ensure!(
                project
                    .project_references
                    .len()
                    .checked_add(project.library_entries as usize)
                    .is_some_and(|count| count <= MAX_JDTLS_CLASSPATH_ENTRIES),
                "remote JDTLS snapshot has too many classpath entries"
            );
            let source_roots = project
                .source_roots
                .into_iter()
                .map(|source| {
                    let path = parse_remote_path(&source.path, worktree_id, snapshot)?;
                    Ok(JdtlsSourceRoot {
                        path,
                        navigable: source.navigable,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                project
                    .project_references
                    .iter()
                    .all(|reference| virtual_project_name(&format!("/{reference}"))
                        == Some(reference.as_str())),
                "remote JDTLS snapshot has an invalid project reference"
            );
            ensure!(
                project
                    .library_file_names
                    .len()
                    .checked_add(project.library_artifacts.len())
                    .is_some_and(|count| count <= project.library_entries as usize)
                    && project
                        .library_file_names
                        .iter()
                        .all(|name| valid_library_file_name(name)),
                "remote JDTLS snapshot has an invalid library file name"
            );
            let mut library_file_source_sets = vec![Vec::new(); project.library_file_names.len()];
            let mut seen_source_set_indices = std::collections::HashSet::new();
            for usage in project.library_source_sets {
                let index = usage.file_index as usize;
                ensure!(
                    seen_source_set_indices.insert(index)
                        && library_file_source_sets.get(index).is_some()
                        && valid_gradle_source_sets(&usage.source_sets),
                    "remote JDTLS snapshot has invalid Gradle source-set use"
                );
                library_file_source_sets[index] = usage.source_sets;
            }
            let library_files = project
                .library_file_names
                .into_iter()
                .zip(library_file_source_sets)
                .map(|(name, gradle_source_sets)| JdtlsLibraryFile {
                    name,
                    gradle_source_sets,
                })
                .collect();
            let library_artifacts = project
                .library_artifacts
                .into_iter()
                .map(|artifact| {
                    ensure!(
                        valid_artifact_coordinate(&artifact.coordinate),
                        "remote JDTLS snapshot has an invalid Maven artifact coordinate"
                    );
                    let scope = if artifact.scope.is_empty() {
                        None
                    } else {
                        ensure!(
                            jdtls_maven_scope(&artifact.scope).is_some(),
                            "remote JDTLS snapshot has an invalid Maven artifact scope"
                        );
                        Some(artifact.scope)
                    };
                    Ok(JdtlsLibraryArtifact {
                        coordinate: artifact.coordinate,
                        scope,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                project.java_source_level <= 99,
                "remote JDTLS snapshot has an invalid Java source level"
            );
            ensure!(
                project.java_target_level <= 99,
                "remote JDTLS snapshot has an invalid Java target level"
            );
            ensure!(
                project.project_vm_java_major <= 99,
                "remote JDTLS snapshot has an invalid project VM Java version"
            );
            ensure!(
                project.server_launch_java_major <= 99,
                "remote JDTLS snapshot has an invalid launch Java version"
            );
            ensure!(
                project.project_vm_java_major == 0 || project.project_vm_known,
                "remote JDTLS snapshot has a Java version without a project VM"
            );
            Ok(JdtlsProject {
                module_root,
                membership,
                imported_build_kind,
                source_roots,
                project_references: project.project_references,
                library_entries: project.library_entries as usize,
                library_files,
                library_artifacts,
                gradle,
                maven,
                project_vm_known: project.project_vm_known,
                project_vm_java_major: (project.project_vm_java_major != 0)
                    .then_some(project.project_vm_java_major),
                server_launch_java_major: (project.server_launch_java_major != 0)
                    .then_some(project.server_launch_java_major),
                java_source_level: (project.java_source_level != 0)
                    .then_some(project.java_source_level),
                java_target_level: (project.java_target_level != 0)
                    .then_some(project.java_target_level),
            })
        })
        .collect()
}

fn gradle_configuration_name_is_valid(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 4096
        && !name.starts_with('.')
        && !name.ends_with('.')
        && name.chars().all(|character| {
            !character.is_control()
                && !matches!(
                    character,
                    '/' | '\\' | ':' | '<' | '>' | '"' | '?' | '*' | '|'
                )
        })
}

fn validate_gradle_configurations(
    known: bool,
    configurations: &[GradleConfiguration],
) -> Result<()> {
    ensure!(
        configurations.len() <= 512 && (known || configurations.is_empty()),
        "Gradle configuration count or availability is invalid"
    );
    let mut names = HashSet::new();
    for configuration in configurations {
        ensure!(
            gradle_configuration_name_is_valid(&configuration.name)
                && names.insert(&configuration.name)
                && configuration
                    .description
                    .as_deref()
                    .is_none_or(model_text_is_valid),
            "Gradle configuration metadata is invalid or duplicated"
        );
    }
    let mut edge_count = 0usize;
    for configuration in configurations {
        edge_count = edge_count
            .checked_add(configuration.extends_from.len())
            .context("Gradle configuration edge count exceeded")?;
        let mut parents = HashSet::new();
        ensure!(
            edge_count <= 4096
                && configuration.extends_from.len() <= 128
                && configuration.extends_from.iter().all(|parent| {
                    parent != &configuration.name
                        && names.contains(parent)
                        && parents.insert(parent)
                }),
            "Gradle configuration parents are invalid, duplicated or oversized"
        );
    }
    Ok(())
}

fn gradle_from_proto(
    project: proto::JdtlsGradleProjectInfo,
    module: &ProjectPath,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
) -> Result<GradleProject> {
    let build_root = parse_remote_path(&project.build_root, worktree_id, snapshot)?;
    let configurations = project
        .configurations
        .into_iter()
        .map(|configuration| GradleConfiguration {
            name: configuration.name,
            description: configuration.description,
            resolvable: configuration.resolvable,
            consumable: configuration.consumable,
            extends_from: configuration.extends_from,
        })
        .collect::<Vec<_>>();
    validate_gradle_configurations(project.configurations_known, &configurations)?;
    ensure!(
        module.path.starts_with(&build_root.path)
            && gradle_path_is_valid(&project.project_path, true)
            && project.tasks.len() <= 4096
            && project.source_sets.len() <= 128
            && (project.source_sets_known || project.source_sets.is_empty()),
        "Remote Gradle project metadata is invalid"
    );
    let mut task_paths = HashSet::new();
    let tasks = project
        .tasks
        .into_iter()
        .map(|task| {
            let expected = if project.project_path == ":" {
                format!(":{}", task.name)
            } else {
                format!("{}:{}", project.project_path, task.name)
            };
            ensure!(
                task.path == expected
                    && gradle_path_is_valid(&task.path, false)
                    && task_paths.insert(task.path.clone())
                    && task.options.len() <= 128
                    && task.group.as_deref().is_none_or(model_text_is_valid)
                    && task.description.as_deref().is_none_or(model_text_is_valid),
                "Remote Gradle task metadata is invalid or duplicated"
            );
            let mut names = HashSet::new();
            let options = task
                .options
                .into_iter()
                .map(|option| {
                    ensure!(
                        model_identifier_is_valid(&option.name)
                            && names.insert(option.name.clone())
                            && model_text_is_valid(&option.description)
                            && !option.declared_type.is_empty()
                            && model_text_is_valid(&option.declared_type),
                        "Remote Gradle option metadata is invalid or duplicated"
                    );
                    let enum_constants = match option.enum_constants_known {
                        Some(true) if gradle_enum_constants_are_valid(&option.enum_constants) => {
                            Some(option.enum_constants)
                        }
                        None | Some(false) if option.enum_constants.is_empty() => None,
                        _ => anyhow::bail!(
                            "Remote Gradle enum declarations are invalid, inconsistent, duplicated or oversized"
                        ),
                    };
                    Ok(GradleTaskOption {
                        name: option.name,
                        description: option.description,
                        declared_type: option.declared_type,
                        enum_constants,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GradleTask {
                path: task.path,
                name: task.name,
                group: task.group,
                description: task.description,
                options_known: task.options_known,
                options,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut names = HashSet::new();
    let mut root_count = 0usize;
    let source_sets = project
        .source_sets
        .into_iter()
        .map(|source| {
            root_count = root_count
                .checked_add(source.roots.len())
                .and_then(|count| {
                    count.checked_add(usize::from(source.annotation_processor_output.is_some()))
                })
                .context("Remote Gradle source-root count exceeded")?;
            ensure!(
                root_count <= MAX_JDTLS_SOURCE_ROOTS
                    && model_identifier_is_valid(&source.name)
                    && names.insert(source.name.clone()),
                "Remote Gradle source sets are invalid or duplicated"
            );
            let mut paths = HashSet::new();
            let roots = source
                .roots
                .into_iter()
                .map(|root| {
                    let path = parse_remote_path(&root.path, worktree_id, snapshot)?;
                    ensure!(
                        paths.insert(path.clone()),
                        "Remote Gradle source root is duplicated"
                    );
                    Ok(JdtlsSourceRoot {
                        path,
                        navigable: root.navigable,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let annotation_processor_output = match (
                source.annotation_processor_output_known,
                source.annotation_processor_output,
            ) {
                (None | Some(false), None) => GradleAnnotationProcessorOutput::Unknown,
                (Some(true), None) => GradleAnnotationProcessorOutput::Absent,
                (Some(true), Some(output)) => {
                    let path = parse_remote_path(&output.path, worktree_id, snapshot)?;
                    ensure!(
                        path.path.starts_with(&build_root.path),
                        "Remote Gradle compiler output is outside its build"
                    );
                    GradleAnnotationProcessorOutput::Declared(JdtlsSourceRoot {
                        path,
                        navigable: output.navigable,
                    })
                }
                _ => anyhow::bail!("Remote Gradle compiler-output coverage is inconsistent"),
            };
            Ok(GradleSourceSet {
                name: source.name,
                roots,
                annotation_processor_output,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dependency_graph = project
        .dependency_graph_json
        .map(|text| {
            ensure!(
                text.len() <= 2 * 1024 * 1024,
                "Remote Gradle graph exceeds the size limit"
            );
            let value: Value = serde_json::from_str(&text).map_err(|_| {
                anyhow::anyhow!(
                    "Remote Gradle graph is invalid; refresh host metadata before retrying"
                )
            })?;
            validate_gradle_graph_json_budget(&value)?;
            let mut graph: GradleDependencyGraph =
                serde_json::from_value(value.clone()).map_err(|_| {
                    anyhow::anyhow!(
                        "Remote Gradle graph is invalid; refresh host metadata before retrying"
                    )
                })?;
            ensure!(
                graph.project_path == project.project_path
                    && graph_binding_text_is_valid(&graph.name)
                    && !graph.name.is_empty()
                    && project.configurations_known
                    && configurations.iter().any(|configuration| configuration.name
                        == graph.resolution.configuration
                        && configuration.resolvable),
                "Remote Gradle graph selection differs from imported metadata"
            );
            graph.resolution = decode_gradle_resolution_graph(
                value["resolution"].clone(),
                &graph.resolution.configuration,
                &project.project_path,
            )?;
            Ok(Arc::new(graph))
        })
        .transpose()?;
    Ok(GradleProject {
        dependency_graph,
        build_root,
        project_path: project.project_path,
        tasks,
        source_sets_known: project.source_sets_known,
        source_sets,
        configurations_known: project.configurations_known,
        configurations,
    })
}

fn maven_from_proto(
    project: proto::JdtlsMavenProjectInfo,
    module: &ProjectPath,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
) -> Result<MavenProject> {
    let pom = parse_remote_path(&project.pom, worktree_id, snapshot)?;
    ensure!(
        pom.path.file_name().is_some()
            && pom.path.parent() == Some(module.path.as_ref())
            && [
                &project.group_id,
                &project.artifact_id,
                &project.version,
                &project.packaging
            ]
            .into_iter()
            .all(|text| model_identifier_is_valid(text))
            && project.modules.len() <= MAX_JDTLS_MODULES
            && project.profiles.len() <= 128
            && project.profile_definitions.len() <= 128
            && project.plugins.len() <= 128
            && project.lifecycles.len() <= 32
            && project
                .reactor_position
                .is_none_or(|position| position < MAX_JDTLS_MODULES as u32)
            && !(project.reactor_position.is_some() && project.reactor_order_problem.is_some())
            && project
                .reactor_order_problem
                .as_deref()
                .is_none_or(|problem| matches!(
                    problem,
                    "missing-cached-model" | "duplicate-coordinates" | "invalid-cached-reactor"
                )),
        "Remote Maven project metadata is invalid"
    );
    let mut paths = HashSet::new();
    let modules = project
        .modules
        .into_iter()
        .map(|path| {
            let path = parse_remote_path(&path, worktree_id, snapshot)?;
            ensure!(
                paths.insert(path.clone()),
                "Remote Maven module link is duplicated"
            );
            Ok(path)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut profiles = BTreeMap::new();
    for profile in project.profiles {
        ensure!(
            model_identifier_is_valid(&profile.id)
                && profiles.insert(profile.id, profile.active).is_none(),
            "Remote Maven profile is invalid or duplicated"
        );
    }
    let mut origins = HashSet::new();
    let profile_definitions = project
        .profile_definitions
        .into_iter()
        .map(|definition| {
            ensure!(
                [
                    &definition.id,
                    &definition.source_group_id,
                    &definition.source_artifact_id,
                    &definition.source_version
                ]
                .into_iter()
                .all(|text| model_identifier_is_valid(text))
                    && origins.insert((
                        definition.id.clone(),
                        definition.source_group_id.clone(),
                        definition.source_artifact_id.clone(),
                        definition.source_version.clone(),
                        definition.source_pom.clone()
                    )),
                "Remote Maven profile origin is invalid or duplicated"
            );
            let source_pom = definition
                .source_pom
                .map(|path| parse_remote_path(&path, worktree_id, snapshot))
                .transpose()?;
            ensure!(
                source_pom
                    .as_ref()
                    .is_none_or(|path| path.path.file_name().is_some())
                    && if definition.inherited {
                        source_pom.as_ref() != Some(&pom)
                    } else {
                        source_pom.as_ref() == Some(&pom)
                            && definition.source_group_id == project.group_id
                            && definition.source_artifact_id == project.artifact_id
                            && definition.source_version == project.version
                    },
                "Remote Maven profile origin differs from its module"
            );
            Ok(MavenProfileDefinition {
                id: definition.id,
                source_group_id: definition.source_group_id,
                source_artifact_id: definition.source_artifact_id,
                source_version: definition.source_version,
                source_pom,
                inherited: definition.inherited,
                active_at_source: definition.active_at_source,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut identities = HashSet::new();
    let mut goal_count = 0usize;
    let plugins = project
        .plugins
        .into_iter()
        .map(|plugin| {
            ensure!(
                model_identifier_is_valid(&plugin.group_id)
                    && model_identifier_is_valid(&plugin.artifact_id)
                    && plugin
                        .version
                        .as_deref()
                        .is_none_or(model_identifier_is_valid)
                    && plugin
                        .goal_prefix
                        .as_deref()
                        .is_none_or(model_identifier_is_valid)
                    && identities.insert((plugin.group_id.clone(), plugin.artifact_id.clone()))
                    && (plugin.goals_known || plugin.goals.is_empty()),
                "Remote Maven plugin is invalid or duplicated"
            );
            goal_count = goal_count
                .checked_add(plugin.goals.len())
                .context("Remote Maven goal count exceeded")?;
            ensure!(goal_count <= 4096, "Remote Maven goal count exceeded");
            let mut names = HashSet::new();
            let goals = plugin
                .goals
                .into_iter()
                .map(|goal| {
                    ensure!(
                        model_identifier_is_valid(&goal.name)
                            && names.insert(goal.name.clone())
                            && goal.phase.as_deref().is_none_or(model_identifier_is_valid)
                            && goal.description.as_deref().is_none_or(model_text_is_valid),
                        "Remote Maven goal is invalid or duplicated"
                    );
                    let goal = MavenGoal {
                        name: goal.name,
                        phase: goal.phase,
                        description: goal.description,
                        parameters_known: goal.parameters_known,
                        parameters: goal
                            .parameters
                            .into_iter()
                            .map(|parameter| MavenGoalParameter {
                                name: parameter.name,
                                alias: parameter.alias,
                                declared_type: parameter.declared_type,
                                required: parameter.required,
                                editable: parameter.editable,
                                description: parameter.description,
                            })
                            .collect(),
                    };
                    ensure!(
                        maven_goal_parameters_are_valid(&goal),
                        "Remote Maven parameters are invalid or duplicated"
                    );
                    Ok(goal)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(MavenPlugin {
                group_id: plugin.group_id,
                artifact_id: plugin.artifact_id,
                version: plugin.version,
                managed_only: plugin.managed_only,
                goals_known: plugin.goals_known,
                goal_prefix: plugin.goal_prefix,
                goals,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut names = HashSet::new();
    let lifecycles = project
        .lifecycles
        .into_iter()
        .map(|lifecycle| {
            let mut phases = HashSet::new();
            ensure!(
                model_identifier_is_valid(&lifecycle.id)
                    && names.insert(lifecycle.id.clone())
                    && !lifecycle.phases.is_empty()
                    && lifecycle.phases.len() <= 128
                    && lifecycle
                        .phases
                        .iter()
                        .all(|phase| model_identifier_is_valid(phase) && phases.insert(phase)),
                "Remote Maven lifecycle is invalid or duplicated"
            );
            Ok(MavenLifecycle {
                id: lifecycle.id,
                phases: lifecycle.phases,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dependency_graph = project
        .dependency_graph_json
        .map(|text| {
            ensure!(
                text.len() <= 8 * 1024 * 1024,
                "Remote Maven graph exceeds the size limit"
            );
            let value: Value = serde_json::from_str(&text).map_err(|_| {
                anyhow::anyhow!(
                    "Remote Maven graph is invalid; refresh host metadata before retrying"
                )
            })?;
            ensure!(
                value.get("rootUri").is_none() && value.get("pomUri").is_none(),
                "Remote Maven graph contains host path bindings"
            );
            let graph = decode_maven_dependency_graph(value)?;
            ensure!(
                graph
                    .components
                    .iter()
                    .any(|component| component.id == graph.root_component
                        && component.group_id == project.group_id
                        && component.artifact_id == project.artifact_id
                        && component.version == project.version),
                "Remote Maven graph root differs from imported coordinates"
            );
            Ok(Arc::new(graph))
        })
        .transpose()?;
    Ok(MavenProject {
        dependency_graph,
        pom,
        modules,
        group_id: project.group_id,
        artifact_id: project.artifact_id,
        version: project.version,
        packaging: project.packaging,
        profiles,
        profile_definitions_known: project.profile_definitions_known,
        profile_definitions,
        reactor_position: project.reactor_position.map(|position| position as usize),
        reactor_order_problem: project.reactor_order_problem,
        plugins,
        lifecycles,
    })
}

pub fn kotlin_project_to_proto(
    project: &KotlinWorkspaceProject,
) -> proto::KotlinWorkspaceProjectInfo {
    proto::KotlinWorkspaceProjectInfo {
        name: project.name.clone(),
        build_system: project.build_system.wire_name().to_string(),
        module_root: project.module_root.path.as_unix_str().to_string(),
        source_roots: project
            .source_roots
            .iter()
            .map(|source| proto::KotlinWorkspaceSourceInfo {
                path: source.path.path.as_unix_str().to_string(),
                navigable: source.navigable,
                generated: source.generated,
            })
            .collect(),
        dependencies: project
            .dependencies
            .iter()
            .map(|dependency| match dependency {
                KotlinWorkspaceDependency::Module { name, scope } => {
                    proto::KotlinWorkspaceDependencyInfo {
                        is_module: true,
                        name: name.clone(),
                        scope: scope.clone(),
                    }
                }
                KotlinWorkspaceDependency::Artifact { coordinate, scope } => {
                    proto::KotlinWorkspaceDependencyInfo {
                        is_module: false,
                        name: coordinate.clone(),
                        scope: scope.clone(),
                    }
                }
            })
            .collect(),
        unidentified_libraries: project.unidentified_libraries as u32,
    }
}

pub fn kotlin_projects_from_proto(
    raw_projects: Vec<proto::KotlinWorkspaceProjectInfo>,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
) -> Result<Vec<KotlinWorkspaceProject>> {
    check_module_count(raw_projects.len())?;
    let mut names = HashSet::new();
    for project in &raw_projects {
        ensure!(
            valid_kotlin_module_name(&project.name) && names.insert(project.name.clone()),
            "remote Kotlin snapshot has an invalid module name"
        );
    }
    raw_projects
        .into_iter()
        .map(|project| {
            let module_root = parse_remote_path(&project.module_root, worktree_id, snapshot)?;
            ensure!(
                project.source_roots.len() <= MAX_JDTLS_SOURCE_ROOTS,
                "remote Kotlin snapshot has too many source roots"
            );
            ensure!(
                project
                    .dependencies
                    .len()
                    .checked_add(project.unidentified_libraries as usize)
                    .is_some_and(|count| count <= MAX_JDTLS_CLASSPATH_ENTRIES),
                "remote Kotlin snapshot has too many classpath entries"
            );
            let source_roots = project
                .source_roots
                .into_iter()
                .map(|source| {
                    Ok(KotlinWorkspaceSourcePath {
                        path: parse_remote_path(&source.path, worktree_id, snapshot)?,
                        navigable: source.navigable,
                        generated: source.generated,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let dependencies = project
                .dependencies
                .into_iter()
                .map(|dependency| {
                    ensure!(
                        dependency.scope.as_deref()
                            == kotlin_dependency_scope(dependency.scope.as_deref()).as_deref(),
                        "remote Kotlin snapshot has an invalid classpath scope"
                    );
                    if dependency.is_module {
                        ensure!(
                            names.contains(&dependency.name),
                            "remote Kotlin snapshot has an unknown module reference"
                        );
                        Ok(KotlinWorkspaceDependency::Module {
                            name: dependency.name,
                            scope: dependency.scope,
                        })
                    } else {
                        ensure!(
                            valid_artifact_coordinate(&dependency.name),
                            "remote Kotlin snapshot has an invalid artifact coordinate"
                        );
                        Ok(KotlinWorkspaceDependency::Artifact {
                            coordinate: dependency.name,
                            scope: dependency.scope,
                        })
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(KotlinWorkspaceProject {
                name: project.name,
                build_system: KotlinBuildSystem::from_remote(&project.build_system)
                    .context("remote Kotlin snapshot has an invalid build system")?,
                module_root,
                source_roots,
                dependencies,
                unidentified_libraries: project.unidentified_libraries as usize,
            })
        })
        .collect()
}

fn remote_path_is_visible(path: &RelPath, snapshot: &Snapshot) -> bool {
    !path.as_unix_str().chars().any(char::is_control)
        && !path.ancestors().any(|ancestor| {
            snapshot
                .entry_for_path(ancestor)
                .is_some_and(|entry| entry.is_private || entry.is_external || entry.is_fifo)
        })
        && (path.is_empty()
            || snapshot
                .entry_for_path(path)
                .is_some_and(|entry| !entry.is_private && !entry.is_external && !entry.is_fifo))
}

fn parse_remote_path(
    relative_path: &str,
    worktree_id: WorktreeId,
    snapshot: &Snapshot,
) -> Result<ProjectPath> {
    ensure!(relative_path.len() <= 4096, "remote JVM path is too long");
    let path = RelPath::from_unix_str(relative_path)
        .map_err(|_| anyhow::anyhow!("remote JVM path is not project-relative"))?;
    ensure!(
        remote_path_is_visible(path, snapshot),
        "remote JVM path is not visible in this project"
    );
    Ok(ProjectPath {
        worktree_id,
        path: Arc::from(path),
    })
}

pub fn is_project_input(path: &RelPath) -> bool {
    let build_configuration = matches!(
        path.file_name(),
        Some(
            "pom.xml"
                | "build.gradle"
                | "build.gradle.kts"
                | "settings.gradle"
                | "settings.gradle.kts"
                | "gradle.properties"
                | "gradle-wrapper.properties"
                | "maven-wrapper.properties"
                | "maven.config"
                | "jvm.config"
                | "settings.xml"
                | "profiles.xml"
                | "extensions.xml"
                | "mvnw"
                | "mvnw.cmd"
                | "gradlew"
                | "gradlew.bat"
                | ".classpath"
                | ".project"
        )
    );
    let version_catalog = path.extension() == Some("toml")
        && path.components().any(|component| component == "gradle");
    let build_source = path
        .components()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|components| components == ["buildSrc", "src"])
        && matches!(path.extension(), Some("java" | "kt" | "groovy"));
    build_configuration || version_catalog || build_source
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportedOwnershipCoverage {
    pub(crate) root: Option<PathBuf>,
    pub(crate) initialized_folders: Vec<String>,
    pub(crate) complete: bool,
    // Plain Eclipse projects have no build kind, but can remain untouched in a complete namespace.
    pub(crate) namespace_complete: bool,
    pub(crate) imported_count: usize,
}

impl ImportedOwnershipCoverage {
    pub(crate) fn scope_is_current(&self, root: &Path, server: &LanguageServer) -> bool {
        self.root.as_deref() == Some(root)
            && imported_ownership_scope_folders(
                server
                    .workspace_folders()
                    .iter()
                    .map(|folder| folder.as_str().to_owned()),
                root,
            )
            .is_ok_and(|folders| folders == self.initialized_folders)
    }
}

pub(crate) struct JdtlsProjectQuery {
    pub projects: Vec<(String, JdtlsProjectSettings)>,
    pub ownership: ImportedOwnershipCoverage,
}

fn imported_ownership_scope_folders(
    workspace_folders: impl IntoIterator<Item = String>,
    root: &Path,
) -> Result<Vec<String>> {
    let mut folders = BTreeSet::new();
    for folder in workspace_folders {
        ensure!(
            folder.len() <= 4096 && !folder.chars().any(char::is_control),
            "JVM initialized scope exceeds its bound or contains controls"
        );
        let path = file_uri_path(&folder)?;
        let canonical = url::Url::from_directory_path(&path)
            .map_err(|()| anyhow::anyhow!("JVM initialized scope is not an absolute directory"))?;
        ensure!(
            canonical.as_str() == folder
                || canonical.as_str().strip_suffix('/') == Some(folder.as_str()),
            "JVM initialized scope is not canonical"
        );
        if path.starts_with(root) {
            folders.insert(folder);
            ensure!(
                folders.len() <= 32,
                "Too many initialized JVM folders; narrow the project before refreshing"
            );
        }
    }
    Ok(folders.into_iter().collect())
}

pub(crate) async fn query_jdtls_projects(
    server: Arc<LanguageServer>,
    request_timeout: Duration,
    background_executor: BackgroundExecutor,
    maven_root: Option<PathBuf>,
    imported_build_root: Option<PathBuf>,
) -> Result<JdtlsProjectQuery> {
    // The installed Java proxy answered these read-only commands in a direct
    // probe, while Groovy's capability snapshot omitted them. Validate the
    // responses rather than relying on that snapshot.
    let projects = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: "java.project.getAll".to_string(),
                ..Default::default()
            },
            request_timeout,
        )
        .await
        .into_response()
        .context("query JDTLS projects")?
        .context("JDTLS returned no project list")?;
    let project_uris = imported_project_uris(projects)?;
    let resources = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: "java.project.getAll".into(),
                // JSONUtility.toModel accepts a JSON string; LSP4J maps object
                // arguments to a Map that the pinned JDTLS converter rejects.
                arguments: vec![json!(r#"{"includeNonJava":true}"#)],
                ..Default::default()
            },
            request_timeout,
        )
        .await
        .into_response();
    let resources = match resources {
        Ok(Some(resources)) => Some(imported_project_uris(resources)?),
        Ok(None) => {
            log::warn!(
                "JDTLS imported resource enumeration unavailable; refresh or check provider setup"
            );
            None
        }
        Err(error) => {
            log::warn!(
                "JDTLS imported resource enumeration failed; refresh or check provider setup: {error:#}"
            );
            None
        }
    };
    let resource_enumeration_complete = resources.is_some();
    let mut ownership = ImportedOwnershipCoverage::default();
    let java_paths = project_uris
        .iter()
        .map(|uri| file_uri_path(uri))
        .collect::<Result<HashSet<_>>>()?;
    if let Some(resources) = &resources {
        let resource_paths = resources
            .iter()
            .map(|uri| file_uri_path(uri))
            .collect::<Result<HashSet<_>>>()?;
        ensure!(
            java_paths.is_subset(&resource_paths),
            "JDTLS imported identities changed during enumeration; refresh again"
        );
    }
    let mut result = Vec::with_capacity(project_uris.len());
    for project_uri in project_uris {
        let settings = server
            .request::<lsp::request::ExecuteCommand>(
                lsp::ExecuteCommandParams {
                    command: "java.project.getSettings".to_string(),
                    arguments: vec![
                        json!(project_uri),
                        json!([
                            "org.eclipse.jdt.ls.core.sourcePaths",
                            "org.eclipse.jdt.ls.core.classpathEntries",
                            "org.eclipse.jdt.ls.core.vm.location",
                            "org.eclipse.jdt.core.compiler.source",
                            "org.eclipse.jdt.core.compiler.codegen.targetPlatform"
                        ]),
                    ],
                    ..Default::default()
                },
                request_timeout,
            )
            .await
            .into_response()
            .context("query JDTLS project settings")?
            .context("JDTLS returned no project settings")?;
        let mut settings: JdtlsProjectSettings =
            serde_json::from_value(settings).context("JDTLS returned invalid project settings")?;
        settings.membership = JdtlsProjectMembership::ImportedJava;
        settings.imported_resources_incomplete = resources.is_none();
        result.push((project_uri, settings));
    }
    if let Some(resources) = resources {
        for uri in resources {
            if !java_paths.contains(&file_uri_path(&uri)?) {
                result.push((
                    uri,
                    JdtlsProjectSettings {
                        membership: JdtlsProjectMembership::ImportedNonJava,
                        ..Default::default()
                    },
                ));
            }
        }
    }
    check_module_count(result.len())?;
    if let Some(root) = imported_build_root {
        let folders = imported_ownership_scope_folders(
            server
                .workspace_folders()
                .iter()
                .map(|folder| folder.as_str().to_owned()),
            &root,
        )?;
        let (namespace_kinds_complete, kinds_complete) =
            query_imported_build_kinds(&server, request_timeout, &root, &mut result).await?;
        let mut complete = resource_enumeration_complete && kinds_complete;
        let mut imported_count = 0;
        for (uri, settings) in &result {
            if settings.membership.is_imported() && file_uri_path(uri)?.starts_with(&root) {
                imported_count += 1;
                complete &= settings.imported_build_kind != JdtlsImportedBuildKind::Unknown;
            }
        }
        ensure!(
            folders
                == imported_ownership_scope_folders(
                    server
                        .workspace_folders()
                        .iter()
                        .map(|folder| folder.as_str().to_owned()),
                    &root,
                )?,
            "Initialized JVM ownership scope changed during refresh; retry"
        );
        ownership = ImportedOwnershipCoverage {
            root: Some(root),
            initialized_folders: folders,
            complete,
            namespace_complete: resource_enumeration_complete && namespace_kinds_complete,
            imported_count,
        };
    }
    let maven_command = server
        .capabilities()
        .execute_command_provider
        .as_ref()
        .and_then(|provider| {
            provider
                .commands
                .iter()
                .find(|command| *command == "groovy.maven.goalDetails")
                .or_else(|| {
                    provider
                        .commands
                        .iter()
                        .find(|command| *command == "groovy.maven.projectDetails")
                })
                .or_else(|| {
                    provider
                        .commands
                        .iter()
                        .find(|command| *command == "groovy.maven.projects")
                })
                .cloned()
        });
    if let Some(root) = maven_root
        && let Some(command) = maven_command
    {
        let root_uri = url::Url::from_directory_path(&root)
            .map_err(|()| anyhow::anyhow!("Maven project root is unavailable"))?;
        let response = server
            .request::<lsp::request::ExecuteCommand>(
                lsp::ExecuteCommandParams {
                    command,
                    arguments: vec![json!({"rootUri": root_uri, "cachedOnly": true})],
                    ..Default::default()
                },
                request_timeout,
            )
            .await
            .into_response()
            .context(
                "Could not read cached Maven metadata; check import/cache settings and refresh",
            )?
            .context("Maven model provider returned no data")?;
        merge_maven_model(&mut result, response, &root)?;
    }
    let launch_java_home = jdtls_launch_java_home(server.binary()).map(Path::to_path_buf);
    let projects = background_executor
        .spawn(async move {
            let launch_java_major = launch_java_home
                .as_deref()
                .and_then(Path::to_str)
                .and_then(project_vm_java_major_from_location);
            for (_, settings) in &mut result {
                settings.server_launch_java_major = launch_java_major;
                settings.project_vm_java_major = settings
                    .project_vm_location
                    .as_deref()
                    .and_then(project_vm_java_major_from_location);
            }
            result
        })
        .await;
    Ok(JdtlsProjectQuery {
        projects,
        ownership,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportedBuildKindsReply {
    version: u32,
    root_uri: String,
    projects: Vec<ImportedBuildKindRecord>,
    partial: bool,
    full_synchronization_known: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportedBuildKindRecord {
    project_uri: String,
    java_nature: bool,
}

fn imported_build_identity_path(uri: &str) -> Result<PathBuf> {
    ensure!(
        uri.len() <= 4096,
        "JVM imported build identity exceeds its bound"
    );
    let path = file_uri_path(uri)?;
    let canonical = url::Url::from_directory_path(&path).map_err(|()| {
        anyhow::anyhow!("JVM imported build identity is not an absolute directory")
    })?;
    ensure!(
        canonical.as_str() == uri || selected_import_uri(&path, true)?.as_str() == uri,
        "JVM imported build identity is not canonical"
    );
    Ok(path)
}

fn imported_build_read_roots(
    workspace_folders: impl IntoIterator<Item = String>,
    worktree_root: &Path,
) -> Result<Vec<PathBuf>> {
    let mut folders = BTreeSet::new();
    for folder in workspace_folders {
        ensure!(
            folder.len() <= 4096,
            "JVM initialized folder exceeds its bound"
        );
        let folder = file_uri_path(&folder)?;
        if folder.starts_with(worktree_root) {
            folders.insert(folder);
            ensure!(
                folders.len() <= 32,
                "Too many initialized JVM folders; narrow the project before refreshing"
            );
        }
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    for folder in folders {
        if !roots.iter().any(|root| folder.starts_with(root)) {
            roots.push(folder);
        }
    }
    Ok(roots)
}

fn admit_imported_build_kinds(
    projects: &mut [(String, JdtlsProjectSettings)],
    replies: Vec<(PathBuf, JdtlsImportedBuildKind, Value)>,
) -> Result<bool> {
    ensure!(
        replies.len() <= 64,
        "Too many JVM imported build identity replies"
    );
    let mut observations = HashMap::new();
    for (uri, settings) in projects.iter() {
        ensure!(
            observations
                .insert(file_uri_path(uri)?, settings.membership)
                .is_none(),
            "JVM imported resource identities are duplicated"
        );
    }
    let mut contexts = BTreeMap::<PathBuf, (bool, bool)>::new();
    let mut kinds = HashMap::new();
    let mut partial = false;
    let mut record_count = 0usize;
    for (root, kind, value) in replies {
        ensure!(
            kind != JdtlsImportedBuildKind::Unknown,
            "JVM imported build provider is unknown"
        );
        ensure!(
            serde_json::to_vec(&value)?.len() <= 1_048_576,
            "JVM imported build identity reply exceeds its bound"
        );
        let reply: ImportedBuildKindsReply = serde_json::from_value(value).map_err(|_| {
            anyhow::anyhow!(
                "JVM imported build identity reply is malformed; check provider setup and refresh"
            )
        })?;
        ensure!(
            reply.version == 1
                && !reply.full_synchronization_known
                && imported_build_identity_path(&reply.root_uri)? == root,
            "JVM imported build identity root or version changed; refresh again"
        );
        let context = contexts.entry(root.clone()).or_default();
        let present = match kind {
            JdtlsImportedBuildKind::Maven => &mut context.0,
            JdtlsImportedBuildKind::Gradle => &mut context.1,
            JdtlsImportedBuildKind::Unknown => {
                anyhow::bail!("JVM imported build provider is unknown")
            }
        };
        ensure!(
            !*present,
            "JVM imported build scope is duplicated; refresh again"
        );
        *present = true;
        check_module_count(reply.projects.len())?;
        record_count = record_count
            .checked_add(reply.projects.len())
            .context("JVM imported build identity count overflow")?;
        check_module_count(record_count)?;
        partial |= reply.partial;
        for record in reply.projects {
            let path = imported_build_identity_path(&record.project_uri)?;
            ensure!(
                path.starts_with(&root),
                "JVM imported build identity is outside its initialized folder"
            );
            ensure!(
                kinds.insert(path.clone(), kind).is_none(),
                "JVM imported build identities are duplicated or have conflicting providers; refresh again"
            );
            match observations.get(&path) {
                Some(JdtlsProjectMembership::ImportedJava) => ensure!(
                    record.java_nature,
                    "JVM imported Java nature changed during refresh; retry"
                ),
                Some(JdtlsProjectMembership::ImportedNonJava) => ensure!(
                    !record.java_nature,
                    "JVM imported non-Java nature changed during refresh; retry"
                ),
                Some(_) => anyhow::bail!("JVM build ownership lacks imported resource evidence"),
                None => partial = true,
            }
        }
    }
    for (uri, settings) in projects {
        let kind = kinds.get(&file_uri_path(uri)?).copied().unwrap_or_default();
        settings.imported_build_kind = kind;
        settings.imported_build_kinds_incomplete = partial;
    }
    Ok(!partial
        && !contexts.is_empty()
        && contexts.values().all(|(maven, gradle)| *maven && *gradle))
}

async fn query_imported_build_kinds(
    server: &Arc<LanguageServer>,
    request_timeout: Duration,
    worktree_root: &Path,
    projects: &mut [(String, JdtlsProjectSettings)],
) -> Result<(bool, bool)> {
    let folders = server.workspace_folders();
    let roots = imported_build_read_roots(
        folders.iter().map(|folder| folder.as_str().to_owned()),
        worktree_root,
    )?;
    let capabilities = server.capabilities();
    let commands = capabilities.execute_command_provider.as_ref();
    let mut replies = Vec::new();
    for (command, kind) in [
        (
            "groovy.maven.importedProjects",
            JdtlsImportedBuildKind::Maven,
        ),
        (
            "groovy.gradle.importedProjects",
            JdtlsImportedBuildKind::Gradle,
        ),
    ] {
        if !commands.is_some_and(|provider| {
            provider
                .commands
                .iter()
                .any(|candidate| candidate == command)
        }) {
            continue;
        }
        for root in &roots {
            let root_uri = url::Url::from_directory_path(root)
                .map_err(|()| anyhow::anyhow!("JVM initialized folder is unavailable"))?;
            let value = server.request::<lsp::request::ExecuteCommand>(
                lsp::ExecuteCommandParams {
                    command: command.into(),
                    arguments: vec![json!({"rootUri": root_uri, "cachedOnly": true})],
                    ..Default::default()
                },
                request_timeout,
            ).await.into_response()
                .map_err(|_| anyhow::anyhow!("Could not read cached JVM imported build identities; check provider setup and refresh"))?
                .context("JVM imported build provider returned no data; check provider setup and refresh")?;
            replies.push((root.clone(), kind, value));
        }
    }
    ensure!(
        folders == server.workspace_folders(),
        "Initialized JVM folders changed during refresh; retry"
    );
    let complete = admit_imported_build_kinds(projects, replies)?;
    Ok((
        complete,
        complete && roots.iter().any(|root| root == worktree_root),
    ))
}

fn imported_project_uris(value: serde_json::Value) -> Result<Vec<String>> {
    let uris: Vec<String> = serde_json::from_value(value)
        .context("JDTLS returned an invalid imported resource list")?;
    check_module_count(uris.len())?;
    let mut paths = HashSet::new();
    for uri in &uris {
        ensure!(
            uri.len() <= 4096 && !uri.chars().any(char::is_control),
            "JDTLS imported identity exceeds its bound or contains controls"
        );
        ensure!(
            paths.insert(file_uri_path(uri)?),
            "JDTLS imported resource identities are duplicated"
        );
    }
    Ok(uris)
}

pub async fn query_gradle_project_metadata(
    server: Arc<LanguageServer>,
    request_timeout: Duration,
    mut result: Vec<(String, JdtlsProjectSettings)>,
    gradle_root: PathBuf,
) -> Result<Vec<(String, JdtlsProjectSettings)>> {
    let gradle_command = server
        .capabilities()
        .execute_command_provider
        .as_ref()
        .and_then(|provider| {
            provider
                .commands
                .iter()
                .find(|command| *command == "groovy.gradle.projectDetails")
                .or_else(|| {
                    provider
                        .commands
                        .iter()
                        .find(|command| *command == "groovy.gradle.projects")
                })
                .cloned()
        });
    let command =
        gradle_command.context("Gradle metadata provider is unavailable; check extension setup")?;
    {
        let root = gradle_root;
        let root_uri = url::Url::from_directory_path(&root)
            .map_err(|()| anyhow::anyhow!("Gradle project root is unavailable"))?;
        let response = server
            .request::<lsp::request::ExecuteCommand>(
                lsp::ExecuteCommandParams {
                    command,
                    arguments: vec![json!({"rootUri": root_uri, "offline": true})],
                    ..Default::default()
                },
                request_timeout,
            )
            .await
            .into_response()
            .context("Could not discover Gradle tasks; check build settings and refresh")?
            .context("Gradle model provider returned no data")?;
        merge_gradle_model(&mut result, response, &root)?;
    }
    Ok(result)
}

pub(crate) async fn query_scoped_gradle_project_metadata(
    server: Arc<LanguageServer>,
    request_timeout: Duration,
    mut projects: Vec<(String, JdtlsProjectSettings)>,
    worktree_root: &Path,
    workspace_root: &Path,
    build_root: &Path,
    admitted_modules: &BTreeSet<PathBuf>,
) -> Result<Vec<(String, JdtlsProjectSettings)>> {
    ensure!(
        server
            .capabilities()
            .execute_command_provider
            .as_ref()
            .is_some_and(|provider| provider
                .commands
                .iter()
                .any(|command| command == "groovy.gradle.projectDetails.v4")),
        "Module-fenced Gradle metadata is unavailable; update the Groovy JVM provider and refresh before synchronizing nested builds"
    );
    let invalid = || {
        anyhow::anyhow!(
            "Scoped Gradle metadata is invalid, incomplete or stale; review current imported builds and provider setup, then refresh before retrying"
        )
    };
    ensure!(
        workspace_root.starts_with(worktree_root)
            && build_root.starts_with(workspace_root)
            && !admitted_modules.is_empty()
            && admitted_modules.iter().all(|module| {
                module.starts_with(build_root) && selected_import_uri(module, true).is_ok()
            }),
        "{}",
        invalid()
    );
    let mut expected_modules = admitted_modules.clone();
    // Eclipse can import only the children of a Gradle build with no Java root project.
    expected_modules.insert(build_root.to_path_buf());
    check_module_count(expected_modules.len()).map_err(|_| invalid())?;
    selected_import_uri(worktree_root, true).map_err(|_| invalid())?;
    let workspace_uri = selected_import_uri(workspace_root, true).map_err(|_| invalid())?;
    let build_uri = selected_import_uri(build_root, true).map_err(|_| invalid())?;
    let parameters = lsp::ExecuteCommandParams {
        command: "groovy.gradle.projectDetails.v4".into(),
        arguments: vec![json!({
            "workspaceRootUri": workspace_uri,
            "buildRootUri": build_uri,
            "offline": true,
        })],
        ..Default::default()
    };
    let value = request_selected_jdtls_phase(
        server,
        &parameters,
        request_timeout,
        "Gradle",
        "scoped metadata discovery",
        "Scoped Gradle metadata failed; check cached host tools and current imported builds, then refresh before retrying",
    )
    .await?;
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Reply {
        version: u32,
        workspace_root_uri: String,
        build_root_uri: String,
        model: Value,
        model_query_complete: bool,
        full_synchronization_known: bool,
    }
    let admit = || -> Result<Value> {
        ensure!(
            serde_json::to_vec(&value)?.len() <= 8 * 1024 * 1024,
            "{}",
            invalid()
        );
        let reply: Reply = serde_json::from_value(value)?;
        ensure!(
            reply.version == 4
                && reply.model_query_complete
                && !reply.full_synchronization_known
                && imported_build_identity_path(&reply.workspace_root_uri)? == workspace_root
                && imported_build_identity_path(&reply.build_root_uri)? == build_root
                && reply.model.get("version").and_then(Value::as_u64) == Some(2)
                && reply
                    .model
                    .get("rootUri")
                    .and_then(Value::as_str)
                    .is_some_and(|uri| imported_build_identity_path(uri)
                        .is_ok_and(|root| root == build_root)),
            "{}",
            invalid()
        );
        Ok(reply.model)
    };
    let model = admit().map_err(|_| invalid())?;
    merge_gradle_model_with_admission(
        &mut projects,
        model,
        build_root,
        worktree_root,
        Some(&expected_modules),
    )
    .map_err(|_| invalid())?;
    Ok(projects)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JdtlsUnlinkProvider {
    Maven,
    Gradle,
}

impl JdtlsUnlinkProvider {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Maven => "Maven",
            Self::Gradle => "Gradle",
        }
    }

    pub(crate) fn command(self) -> &'static str {
        match self {
            Self::Maven => "groovy.maven.unlinkSelected",
            Self::Gradle => "groovy.gradle.unlinkSelected",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedProjectUnlink {
    provider: JdtlsUnlinkProvider,
    root: PathBuf,
    projects: BTreeSet<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedProjectUnlink {
    pub(crate) fn new(
        provider: JdtlsUnlinkProvider,
        root: &Path,
        projects: &[PathBuf],
    ) -> Result<Self> {
        let invalid = || {
            anyhow::anyhow!(
                "Selected {} unlink identities are invalid; refresh and select current imported projects again",
                provider.name()
            )
        };
        ensure!(
            !projects.is_empty() && projects.len() <= 128,
            "{}",
            invalid()
        );
        let root_uri = selected_import_uri(root, true).map_err(|_| invalid())?;
        let mut identities = BTreeSet::new();
        let mut project_uris = Vec::with_capacity(projects.len());
        for project in projects {
            ensure!(
                project.starts_with(root) && identities.insert(project.clone()),
                "{}",
                invalid()
            );
            project_uris.push(selected_import_uri(project, true).map_err(|_| invalid())?);
        }
        Ok(Self {
            provider,
            root: root.to_path_buf(),
            projects: identities,
            parameters: lsp::ExecuteCommandParams {
                command: provider.command().into(),
                arguments: vec![json!({"rootUri": root_uri, "projectUris": project_uris})],
                ..Default::default()
            },
        })
    }

    pub(crate) async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<()> {
        let failure = match self.provider {
            JdtlsUnlinkProvider::Maven => {
                "Selected Maven unlink failed or may be partial; refresh current imported projects and check host permissions before retrying"
            }
            JdtlsUnlinkProvider::Gradle => {
                "Selected Gradle unlink failed or may be partial; refresh current imported projects and check host permissions before retrying"
            }
        };
        let response = request_selected_jdtls_phase(
            server,
            &self.parameters,
            request_timeout,
            self.provider.name(),
            "unlink",
            failure,
        )
        .await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: serde_json::Value) -> Result<()> {
        let invalid = || {
            anyhow::anyhow!(
                "Selected {} unlink receipt is invalid or may be partial; refresh current imported projects before retrying",
                self.provider.name()
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        let bounded_uri = |value: &serde_json::Value| {
            value
                .as_str()
                .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control))
        };
        ensure!(
            object.len() == 5
                && object.get("rootUri").is_some_and(bounded_uri)
                && object
                    .get("unlinkedProjectUris")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(
                        |uris| uris.len() == self.projects.len() && uris.iter().all(bounded_uri)
                    ),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            unlinked_project_uris: Vec<String>,
            selected_unlink_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        ensure!(
            receipt.version == 1
                && receipt.selected_unlink_phase_complete
                && !receipt.full_synchronization_known
                && receipt.unlinked_project_uris.len() == self.projects.len(),
            "{}",
            invalid()
        );
        let root = selected_import_path(&receipt.root_uri, true).map_err(|_| invalid())?;
        ensure!(root == self.root, "{}", invalid());
        let mut projects = BTreeSet::new();
        for uri in receipt.unlinked_project_uris {
            let project = selected_import_path(&uri, true).map_err(|_| invalid())?;
            ensure!(
                project.starts_with(&self.root) && projects.insert(project),
                "{}",
                invalid()
            );
        }
        ensure!(projects == self.projects, "{}", invalid());
        Ok(())
    }
}

fn gradle_module_state_path(uri: &str) -> Result<PathBuf> {
    ensure!(
        uri.len() <= 4096 && !uri.chars().any(char::is_control),
        "Invalid Gradle module state URI"
    );
    let path = selected_import_path(uri, true)?;
    ensure!(
        selected_import_uri(&path, true)?.as_str() == uri,
        "Invalid Gradle module state URI"
    );
    Ok(path)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GradleWorkspaceState {
    Open,
    Closed,
    Absent,
}

#[derive(Clone, Debug)]
struct GradleModuleState {
    path: PathBuf,
    workspace_state: GradleWorkspaceState,
    description_available: bool,
}

#[derive(Clone, Debug)]
pub struct GradleRootState {
    root: PathBuf,
    workspace_state: GradleWorkspaceState,
    description_available: bool,
}

impl GradleRootState {
    pub fn retained_ignore_phase(&self) -> Result<SelectedGradleRootState> {
        ensure!(
            self.description_available,
            "Ignoring a Gradle build without existing project metadata is not yet supported. Keep the ignore setting and check provider support before retrying; this action did not request an import"
        );
        // A no-op close still runs the SDK active-child and orphan preflight.
        SelectedGradleRootState::new(&self.root, false)
    }

    pub(crate) fn ignore_phase_if_retained(&self) -> Result<Option<SelectedGradleRootState>> {
        if self.workspace_state == GradleWorkspaceState::Absent && !self.description_available {
            return Ok(None);
        }
        self.retained_ignore_phase().map(Some)
    }

    pub fn validate_policy(&self, ignored_root: bool) -> Result<()> {
        ensure!(
            if ignored_root {
                self.workspace_state != GradleWorkspaceState::Open
            } else {
                self.workspace_state == GradleWorkspaceState::Open
            },
            "Gradle root policy was not confirmed by the provider; import may be partial, refresh before retrying"
        );
        Ok(())
    }

    pub fn plan(&self, ignored_root: bool) -> Result<Option<SelectedGradleRootState>> {
        let desired_state = if ignored_root {
            GradleWorkspaceState::Closed
        } else {
            GradleWorkspaceState::Open
        };
        if self.workspace_state == GradleWorkspaceState::Absent && !self.description_available
            || self.workspace_state == desired_state
        {
            return Ok(None);
        }
        SelectedGradleRootState::new(&self.root, !ignored_root).map(Some)
    }
}

fn scoped_gradle_state_parameters(
    workspace: &Path,
    root: &Path,
    mut parameters: lsp::ExecuteCommandParams,
) -> Result<lsp::ExecuteCommandParams> {
    let invalid = || {
        anyhow::anyhow!(
            "Gradle state authority is invalid; review initialized workspace/build roots and refresh"
        )
    };
    let workspace_uri = selected_import_uri(workspace, true).map_err(|_| invalid())?;
    let build_uri = selected_import_uri(root, true).map_err(|_| invalid())?;
    ensure!(
        root.starts_with(workspace) && parameters.arguments.len() == 1,
        "{}",
        invalid()
    );
    let command = match parameters.command.as_str() {
        "groovy.gradle.rootState" => "groovy.gradle.rootState.v2",
        "groovy.gradle.applyRootState" => "groovy.gradle.applyRootState.v2",
        "groovy.gradle.moduleStates" => "groovy.gradle.moduleStates.v2",
        "groovy.gradle.applyModuleStates" => "groovy.gradle.applyModuleStates.v2",
        _ => return Err(invalid()),
    };
    let arguments = parameters
        .arguments
        .first_mut()
        .and_then(Value::as_object_mut)
        .ok_or_else(invalid)?;
    let previous_root = arguments.remove("rootUri").ok_or_else(invalid)?;
    ensure!(previous_root == json!(build_uri), "{}", invalid());
    arguments.insert("workspaceRootUri".into(), json!(workspace_uri));
    arguments.insert("buildRootUri".into(), json!(build_uri));
    parameters.command = command.into();
    Ok(parameters)
}

fn normalize_scoped_gradle_state_reply(
    parameters: &lsp::ExecuteCommandParams,
    mut response: Value,
) -> Result<Value> {
    if !matches!(
        parameters.command.as_str(),
        "groovy.gradle.rootState.v2"
            | "groovy.gradle.applyRootState.v2"
            | "groovy.gradle.moduleStates.v2"
            | "groovy.gradle.applyModuleStates.v2"
    ) {
        return Ok(response);
    }
    let invalid = || {
        anyhow::anyhow!(
            "Scoped Gradle state data is invalid or stale; refresh current project data before retrying"
        )
    };
    let arguments = parameters
        .arguments
        .first()
        .and_then(Value::as_object)
        .ok_or_else(invalid)?;
    let object = response.as_object_mut().ok_or_else(invalid)?;
    ensure!(
        object.get("version").and_then(Value::as_u64) == Some(2) && !object.contains_key("rootUri"),
        "{}",
        invalid()
    );
    let workspace = object.remove("workspaceRootUri").ok_or_else(invalid)?;
    let build = object.remove("buildRootUri").ok_or_else(invalid)?;
    ensure!(
        Some(&workspace) == arguments.get("workspaceRootUri")
            && Some(&build) == arguments.get("buildRootUri"),
        "{}",
        invalid()
    );
    object.insert("rootUri".into(), build);
    object.insert("version".into(), json!(1));
    Ok(response)
}

#[derive(Clone, Debug)]
pub struct GradleRootStateQuery {
    root: PathBuf,
    parameters: lsp::ExecuteCommandParams,
}

impl GradleRootStateQuery {
    pub fn new(root: &Path) -> Result<Self> {
        let root_uri = selected_import_uri(root, true).map_err(|_| {
            anyhow::anyhow!(
                "Gradle root state identity is invalid; open its build root and refresh"
            )
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.rootState".into(),
                arguments: vec![json!({"rootUri": root_uri, "offline": true})],
                ..Default::default()
            },
        })
    }

    pub(crate) fn new_scoped(workspace: &Path, root: &Path) -> Result<Self> {
        let mut query = Self::new(root)?;
        query.parameters = scoped_gradle_state_parameters(workspace, root, query.parameters)?;
        Ok(query)
    }

    pub async fn query(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<GradleRootState> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "root state read",
            "Gradle root state unavailable; check provider setup and host build settings, then refresh before retrying",
        ).await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: Value) -> Result<GradleRootState> {
        let response = normalize_scoped_gradle_state_reply(&self.parameters, response)?;
        let invalid = || {
            anyhow::anyhow!(
                "Gradle root state data is invalid or stale; check provider setup and refresh before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 5
                && object.get("version").and_then(Value::as_u64) == Some(1)
                && object
                    .get("fullSynchronizationKnown")
                    .and_then(Value::as_bool)
                    == Some(false)
                && object
                    .get("descriptionAvailable")
                    .is_some_and(Value::is_boolean)
                && object
                    .get("workspaceState")
                    .and_then(Value::as_str)
                    .is_some_and(|state| matches!(state, "open" | "closed" | "absent"))
                && object
                    .get("rootUri")
                    .and_then(Value::as_str)
                    .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control)),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Response {
            version: u32,
            root_uri: String,
            workspace_state: GradleWorkspaceState,
            description_available: bool,
            full_synchronization_known: bool,
        }
        let response: Response = serde_json::from_value(response).map_err(|_| invalid())?;
        let root = gradle_module_state_path(&response.root_uri).map_err(|_| invalid())?;
        ensure!(
            response.version == 1
                && !response.full_synchronization_known
                && root == self.root
                && (response.workspace_state == GradleWorkspaceState::Absent
                    || response.description_available),
            "{}",
            invalid()
        );
        Ok(GradleRootState {
            root,
            workspace_state: response.workspace_state,
            description_available: response.description_available,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SelectedGradleRootState {
    root: PathBuf,
    open: bool,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedGradleRootState {
    pub fn new(root: &Path, open: bool) -> Result<Self> {
        let root_uri = selected_import_uri(root, true).map_err(|_| {
            anyhow::anyhow!("Gradle root state target is invalid; open its build root and refresh")
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            open,
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.applyRootState".into(),
                arguments: vec![json!({"rootUri": root_uri, "offline": true, "open": open})],
                ..Default::default()
            },
        })
    }

    pub(crate) fn with_scoped_authority(mut self, workspace: &Path) -> Result<Self> {
        self.parameters = scoped_gradle_state_parameters(workspace, &self.root, self.parameters)?;
        Ok(self)
    }

    pub async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<()> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "root state change",
            "Gradle root state change failed or may be partial; check provider setup, retained root description, closed children and host settings, then refresh before retrying",
        ).await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: Value) -> Result<()> {
        let response = normalize_scoped_gradle_state_reply(&self.parameters, response)?;
        let invalid = || {
            anyhow::anyhow!(
                "Gradle root state receipt is invalid or may be partial; refresh project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 5
                && object.get("version").and_then(Value::as_u64) == Some(1)
                && object.get("open").and_then(Value::as_bool) == Some(self.open)
                && object
                    .get("rootStatePhaseComplete")
                    .and_then(Value::as_bool)
                    == Some(true)
                && object
                    .get("fullSynchronizationKnown")
                    .and_then(Value::as_bool)
                    == Some(false)
                && object
                    .get("rootUri")
                    .and_then(Value::as_str)
                    .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control)),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            open: bool,
            root_state_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        let root = gradle_module_state_path(&receipt.root_uri).map_err(|_| invalid())?;
        ensure!(
            receipt.version == 1
                && receipt.open == self.open
                && receipt.root_state_phase_complete
                && !receipt.full_synchronization_known
                && root == self.root,
            "{}",
            invalid()
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GradleModuleStates {
    root: PathBuf,
    modules: Vec<GradleModuleState>,
}

impl GradleModuleStates {
    pub(crate) fn module_paths(&self) -> impl Iterator<Item = &Path> {
        self.modules.iter().map(|module| module.path.as_path())
    }

    pub(crate) fn validate_ignored_targets(&self, ignored: &[PathBuf]) -> Result<()> {
        ensure!(
            ignored.iter().all(|directory| self
                .modules
                .iter()
                .any(|module| module.path.starts_with(directory))),
            "Ignored Gradle directory has no current provider module; preserve settings, review the selection and refresh before retrying"
        );
        Ok(())
    }

    pub fn plan(
        &self,
        ignored_directories: &[PathBuf],
    ) -> Result<Option<SelectedGradleModuleStates>> {
        self.plan_policy(ignored_directories, false)
    }

    pub fn plan_closed_build(&self) -> Result<Option<SelectedGradleModuleStates>> {
        self.plan_policy(&[], true)
    }

    fn plan_policy(
        &self,
        ignored_directories: &[PathBuf],
        ignored_root: bool,
    ) -> Result<Option<SelectedGradleModuleStates>> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle module ignore selection is invalid; choose non-root directories inside this build and refresh. Root ignore requires the project-selection lifecycle"
            )
        };
        ensure!(
            ignored_directories.len() <= MAX_JDTLS_MODULES,
            "{}",
            invalid()
        );
        let mut ignored = BTreeSet::new();
        for path in ignored_directories {
            selected_import_uri(path, true).map_err(|_| invalid())?;
            ensure!(
                path != &self.root && path.starts_with(&self.root) && ignored.insert(path),
                "{}",
                invalid()
            );
        }
        let mut closed = BTreeSet::new();
        let mut restored = BTreeSet::new();
        for module in &self.modules {
            if ignored_root
                || ignored
                    .iter()
                    .any(|directory| module.path.starts_with(directory))
            {
                if module.workspace_state != GradleWorkspaceState::Closed {
                    closed.insert(module.path.clone());
                }
            } else if module.workspace_state == GradleWorkspaceState::Closed
                || module.workspace_state == GradleWorkspaceState::Absent
                    && module.description_available
            {
                restored.insert(module.path.clone());
            }
        }
        if closed.is_empty() && restored.is_empty() {
            return Ok(None);
        }
        SelectedGradleModuleStates::new(
            &self.root,
            &closed.into_iter().collect::<Vec<_>>(),
            &restored.into_iter().collect::<Vec<_>>(),
        )
        .map(Some)
    }
}

#[derive(Clone, Debug)]
pub struct GradleModuleStateQuery {
    root: PathBuf,
    parameters: lsp::ExecuteCommandParams,
}

impl GradleModuleStateQuery {
    pub fn new(root: &Path) -> Result<Self> {
        let root_uri = selected_import_uri(root, true).map_err(|_| {
            anyhow::anyhow!("Gradle module state root is invalid; open its build root and refresh")
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.moduleStates".into(),
                arguments: vec![json!({"rootUri":root_uri,"offline":true})],
                ..Default::default()
            },
        })
    }

    pub(crate) fn new_scoped(workspace: &Path, root: &Path) -> Result<Self> {
        let mut query = Self::new(root)?;
        query.parameters = scoped_gradle_state_parameters(workspace, root, query.parameters)?;
        Ok(query)
    }

    pub async fn query(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<GradleModuleStates> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "module state read",
            "Gradle module states unavailable; check provider setup and host build settings, then refresh before retrying",
        ).await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: serde_json::Value) -> Result<GradleModuleStates> {
        let response = normalize_scoped_gradle_state_reply(&self.parameters, response)?;
        let invalid = || {
            anyhow::anyhow!(
                "Gradle module state data is invalid or stale; check provider setup and refresh before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        let modules = object
            .get("modules")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(invalid)?;
        let bounded_uri = |value: &serde_json::Value| {
            value
                .as_str()
                .is_some_and(|text| text.len() <= 4096 && !text.chars().any(char::is_control))
        };
        ensure!(
            object.len() == 4
                && object.get("version").and_then(serde_json::Value::as_u64) == Some(1)
                && object
                    .get("fullSynchronizationKnown")
                    .and_then(serde_json::Value::as_bool)
                    == Some(false)
                && modules.len() < MAX_JDTLS_MODULES
                && object.get("rootUri").is_some_and(bounded_uri)
                && modules
                    .iter()
                    .all(
                        |module| module.as_object().is_some_and(|record| record.len() == 4
                            && record.get("moduleUri").is_some_and(bounded_uri)
                            && record.get("buildRootUri").is_some_and(bounded_uri)
                            && record
                                .get("workspaceState")
                                .and_then(serde_json::Value::as_str)
                                .is_some_and(|state| matches!(
                                    state,
                                    "open" | "closed" | "absent"
                                ))
                            && record
                                .get("descriptionAvailable")
                                .is_some_and(serde_json::Value::is_boolean))
                    ),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Module {
            module_uri: String,
            build_root_uri: String,
            workspace_state: GradleWorkspaceState,
            description_available: bool,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Response {
            version: u32,
            root_uri: String,
            modules: Vec<Module>,
            full_synchronization_known: bool,
        }
        let response: Response = serde_json::from_value(response).map_err(|_| invalid())?;
        let root = gradle_module_state_path(&response.root_uri).map_err(|_| invalid())?;
        ensure!(
            response.version == 1 && !response.full_synchronization_known && root == self.root,
            "{}",
            invalid()
        );
        let mut identities = BTreeSet::new();
        let mut modules = Vec::with_capacity(response.modules.len());
        for record in response.modules {
            let path = gradle_module_state_path(&record.module_uri).map_err(|_| invalid())?;
            let build = gradle_module_state_path(&record.build_root_uri).map_err(|_| invalid())?;
            ensure!(
                path != root
                    && build.starts_with(&root)
                    && path.starts_with(&build)
                    && identities.insert(path.clone())
                    && (record.workspace_state == GradleWorkspaceState::Absent
                        || record.description_available),
                "{}",
                invalid()
            );
            modules.push(GradleModuleState {
                path,
                workspace_state: record.workspace_state,
                description_available: record.description_available,
            });
        }
        Ok(GradleModuleStates { root, modules })
    }
}

#[derive(Clone, Debug)]
pub struct SelectedGradleModuleStates {
    root: PathBuf,
    closed: BTreeSet<PathBuf>,
    restored: BTreeSet<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedGradleModuleStates {
    pub(crate) fn module_paths(&self) -> impl Iterator<Item = &Path> {
        self.closed
            .iter()
            .chain(&self.restored)
            .map(PathBuf::as_path)
    }

    pub fn new(root: &Path, closed: &[PathBuf], restored: &[PathBuf]) -> Result<Self> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle module state identities are invalid; refresh and select current modules inside one build root"
            )
        };
        ensure!(
            closed.len() <= MAX_JDTLS_MODULES
                && restored.len() <= MAX_JDTLS_MODULES
                && (1..=MAX_JDTLS_MODULES).contains(&(closed.len() + restored.len())),
            "{}",
            invalid()
        );
        let root_uri = selected_import_uri(root, true).map_err(|_| invalid())?;
        let mut identities = BTreeSet::new();
        let mut convert = |paths: &[PathBuf]| -> Result<Vec<url::Url>> {
            paths
                .iter()
                .map(|path| {
                    let uri = selected_import_uri(path, true).map_err(|_| invalid())?;
                    ensure!(
                        path != root && path.starts_with(root) && identities.insert(path.clone()),
                        "{}",
                        invalid()
                    );
                    Ok(uri)
                })
                .collect()
        };
        let closed_uris = convert(closed)?;
        let restored_uris = convert(restored)?;
        Ok(Self {
            root: root.to_path_buf(),
            closed: closed.iter().cloned().collect(),
            restored: restored.iter().cloned().collect(),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.applyModuleStates".into(),
                arguments: vec![json!({"rootUri": root_uri, "offline": true,
                    "closedProjectUris": closed_uris, "restoredProjectUris": restored_uris})],
                ..Default::default()
            },
        })
    }

    pub(crate) fn with_scoped_authority(mut self, workspace: &Path) -> Result<Self> {
        self.parameters = scoped_gradle_state_parameters(workspace, &self.root, self.parameters)?;
        Ok(self)
    }

    pub async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<()> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "module state change",
            "Gradle module state change failed or may be partial; check provider setup and host build settings, then refresh before retrying",
        ).await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: serde_json::Value) -> Result<()> {
        let response = normalize_scoped_gradle_state_reply(&self.parameters, response)?;
        let invalid = || {
            anyhow::anyhow!(
                "Gradle module state receipt is invalid or may be partial; refresh project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        let bounded_uri = |value: &serde_json::Value| {
            value
                .as_str()
                .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control))
        };
        let closed = object
            .get("closedProjectUris")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(invalid)?;
        let restored = object
            .get("restoredProjectUris")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(invalid)?;
        ensure!(
            object.len() == 6
                && closed.len() <= MAX_JDTLS_MODULES
                && restored.len() <= MAX_JDTLS_MODULES
                && (1..=MAX_JDTLS_MODULES).contains(&(closed.len() + restored.len()))
                && closed.iter().chain(restored).all(bounded_uri)
                && object.get("rootUri").is_some_and(bounded_uri),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            closed_project_uris: Vec<String>,
            restored_project_uris: Vec<String>,
            module_state_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        let canonical_path = |uri: &str| gradle_module_state_path(uri).map_err(|_| invalid());
        let root = canonical_path(&receipt.root_uri)?;
        ensure!(
            receipt.version == 1
                && receipt.module_state_phase_complete
                && !receipt.full_synchronization_known
                && root == self.root,
            "{}",
            invalid()
        );
        let paths = |uris: Vec<String>| -> Result<BTreeSet<PathBuf>> {
            let mut paths = BTreeSet::new();
            for uri in uris {
                let path = canonical_path(&uri)?;
                ensure!(
                    path != self.root && path.starts_with(&self.root) && paths.insert(path),
                    "{}",
                    invalid()
                );
            }
            Ok(paths)
        };
        ensure!(
            paths(receipt.closed_project_uris)? == self.closed
                && paths(receipt.restored_project_uris)? == self.restored,
            "{}",
            invalid()
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SelectedGradleImport {
    root: PathBuf,
    workspace_root: Option<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedGradleImport {
    pub(crate) fn scoped_roots(&self) -> Result<(&Path, &Path)> {
        let workspace = self.workspace_root.as_deref().context(
            "Gradle import has no initialized scoped authority; refresh before applying policy",
        )?;
        Ok((workspace, &self.root))
    }

    pub fn new(root: &Path) -> Result<Self> {
        let root_uri = selected_import_uri(root, true).map_err(|_| {
            anyhow::anyhow!(
                "Selected Gradle build identity is invalid; open its build root and retry"
            )
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            workspace_root: None,
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.importSelected".into(),
                arguments: vec![json!({"rootUri": root_uri, "offline": true})],
                ..Default::default()
            },
        })
    }

    pub(crate) fn new_scoped(workspace_root: &Path, build_root: &Path) -> Result<Self> {
        let invalid = || {
            anyhow::anyhow!(
                "Scoped Gradle build identity is invalid; review current initialized folders and selected build files, then refresh"
            )
        };
        ensure!(build_root.starts_with(workspace_root), "{}", invalid());
        let workspace_uri = selected_import_uri(workspace_root, true).map_err(|_| invalid())?;
        let build_uri = selected_import_uri(build_root, true).map_err(|_| invalid())?;
        Ok(Self {
            root: build_root.to_path_buf(),
            workspace_root: Some(workspace_root.to_path_buf()),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.importSelected.v2".into(),
                arguments: vec![json!({
                    "workspaceRootUri": workspace_uri,
                    "buildRootUri": build_uri,
                    "offline": true,
                })],
                ..Default::default()
            },
        })
    }

    pub async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<()> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "import",
            "Selected Gradle import failed or may be partial; check host offline settings, cached installation and build selection, then refresh before retrying",
        ).await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: serde_json::Value) -> Result<()> {
        if let Some(workspace_root) = &self.workspace_root {
            return self.validate_scoped_receipt(response, workspace_root);
        }
        let invalid = || {
            anyhow::anyhow!(
                "Selected Gradle import receipt is invalid or may be partial; refresh project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 4
                && object
                    .get("rootUri")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control)),
            "{invalid}",
            invalid = invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            selected_import_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        let root = selected_import_path(&receipt.root_uri, true).map_err(|_| invalid())?;
        ensure!(
            receipt.version == 1
                && receipt.selected_import_phase_complete
                && !receipt.full_synchronization_known
                && root == self.root,
            "{invalid}",
            invalid = invalid()
        );
        Ok(())
    }

    fn validate_scoped_receipt(&self, response: Value, workspace_root: &Path) -> Result<()> {
        let invalid = || {
            anyhow::anyhow!(
                "Scoped Gradle import receipt is invalid or may be partial; refresh current project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 5
                && ["workspaceRootUri", "buildRootUri"].iter().all(|name| {
                    object
                        .get(*name)
                        .and_then(Value::as_str)
                        .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control))
                }),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            workspace_root_uri: String,
            build_root_uri: String,
            selected_import_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        ensure!(
            receipt.version == 2
                && receipt.selected_import_phase_complete
                && !receipt.full_synchronization_known
                && selected_import_path(&receipt.workspace_root_uri, true).map_err(|_| invalid())?
                    == workspace_root
                && selected_import_path(&receipt.build_root_uri, true).map_err(|_| invalid())?
                    == self.root,
            "{}",
            invalid()
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SelectedMavenImport {
    root: PathBuf,
    poms: Vec<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

impl SelectedMavenImport {
    pub fn new(root: &Path, poms: &[PathBuf]) -> Result<Self> {
        ensure!(
            !poms.is_empty() && poms.len() <= 32,
            "Select between one and 32 Maven POM files before importing"
        );
        let root_uri = selected_import_uri(root, true)?;
        let mut identities = HashSet::new();
        let mut pom_uris = Vec::with_capacity(poms.len());
        for pom in poms {
            ensure!(
                pom.starts_with(root)
                    && pom.file_name().is_some_and(|name| name == "pom.xml")
                    && identities.insert(pom),
                "Selected Maven import requires distinct POM files inside the chosen project folder"
            );
            pom_uris.push(selected_import_uri(pom, false)?);
        }
        Ok(Self {
            root: root.to_path_buf(),
            poms: poms.to_vec(),
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.maven.importSelected".into(),
                arguments: vec![json!({"rootUri": root_uri, "pomUris": pom_uris, "offline": true})],
                ..Default::default()
            },
        })
    }

    // This receipt completes only the provider's selected-import phase, not build synchronization.
    pub async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<()> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Maven", "import",
            "Selected Maven import failed or may be partial; check host offline settings, cache and project selection, then refresh before retrying",
        ).await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: serde_json::Value) -> Result<()> {
        let invalid = || {
            anyhow::anyhow!(
                "Selected Maven import receipt is invalid or may be partial; refresh project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        let poms = object
            .get("importedPomUris")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(invalid)?;
        ensure!(
            object.len() == 5
                && !poms.is_empty()
                && poms.len() <= 32
                && poms.iter().all(|pom| pom.as_str().is_some_and(|uri| {
                    uri.len() <= 4096 && !uri.chars().any(char::is_control)
                }))
                && object
                    .get("rootUri")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control)),
            "Selected Maven import receipt is invalid or may be partial; refresh project data before retrying"
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            imported_pom_uris: Vec<String>,
            selected_import_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        let root = selected_import_path(&receipt.root_uri, true).map_err(|_| invalid())?;
        let mut imported = HashSet::new();
        for uri in receipt.imported_pom_uris {
            let path = selected_import_path(&uri, false).map_err(|_| invalid())?;
            ensure!(imported.insert(path), "{invalid}", invalid = invalid());
        }
        ensure!(
            receipt.version == 1
                && receipt.selected_import_phase_complete
                && !receipt.full_synchronization_known
                && root == self.root
                && imported.len() == self.poms.len()
                && self.poms.iter().all(|pom| imported.contains(pom)),
            "{invalid}",
            invalid = invalid()
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ScopedGradleAssociation {
    scope: PathBuf,
    project: PathBuf,
    build: PathBuf,
    java_nature: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ScopedGradleMetadataQuery {
    worktree_root: PathBuf,
    workspace: PathBuf,
    build: PathBuf,
    modules: BTreeSet<PathBuf>,
}

impl ScopedGradleMetadataQuery {
    pub(crate) async fn query(
        &self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
        projects: Vec<(String, JdtlsProjectSettings)>,
    ) -> Result<Vec<(String, JdtlsProjectSettings)>> {
        query_scoped_gradle_project_metadata(
            server,
            request_timeout,
            projects,
            &self.worktree_root,
            &self.workspace,
            &self.build,
            &self.modules,
        )
        .await
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScopedGradleImports {
    root: PathBuf,
    worktree_id: WorktreeId,
    settings: settings::GradleBuildSettingsContent,
    default_query: Option<GradleDefaultImportRootsQuery>,
    cached_associations: Option<Vec<ScopedGradleAssociation>>,
    inputs: Vec<ProjectPath>,
    entries: Vec<worktree::Entry>,
    metadata_inputs: Vec<(Arc<RelPath>, Option<worktree::Entry>)>,
    ignored: Vec<PathBuf>,
    directory_entries: Vec<worktree::Entry>,
    roots: Vec<PathBuf>,
    initialized_roots: Vec<PathBuf>,
    imports: Vec<SelectedGradleImport>,
    path_style: PathStyle,
}

impl ScopedGradleImports {
    pub(crate) fn for_selection(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        settings: &settings::GradleBuildSettingsContent,
    ) -> Result<Option<Self>> {
        Self::for_selection_mode(snapshot, worktree_id, settings, false)
    }

    fn for_selection_mode(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        settings: &settings::GradleBuildSettingsContent,
        allow_root_selection: bool,
    ) -> Result<Option<Self>> {
        if settings.selected_build_files.is_none() {
            return Ok(None);
        }
        if settings
            .selected_build_files
            .as_ref()
            .is_some_and(Vec::is_empty)
        {
            return Self::for_empty(snapshot, worktree_id, settings).map(Some);
        }
        let Some(selection) = SavedGradleSelection::new(
            &GradleSelectionIntent {
                selected_build_files: settings.selected_build_files.as_deref(),
                ignored_projects: &[],
            },
            worktree_id,
            &[],
        )?
        else {
            return Ok(None);
        };
        if !allow_root_selection
            && selection
                .selected_roots
                .iter()
                .all(|root| root.path.is_empty())
        {
            return Ok(None);
        }
        let ignored_directories = settings.ignored_projects.as_deref().unwrap_or_default();
        validate_saved_build_paths(Some(ignored_directories), 128, "Gradle", None)?;
        let root = snapshot.abs_path().as_ref().to_path_buf();
        let roots = selection
            .selected_roots
            .iter()
            .map(|path| root.join(path.path.as_std_path()))
            .collect::<Vec<_>>();
        ensure!(
            !roots.is_empty()
                && roots.iter().enumerate().all(|(index, root)| {
                    roots.iter().enumerate().all(|(other_index, other)| {
                        index == other_index
                            || (!root.starts_with(other) && !other.starts_with(root))
                    })
                }),
            "Selected Gradle roots overlap; choose independent build roots and refresh before importing"
        );
        let ignored = ignored_directories
            .iter()
            .map(|directory| Ok(root.join(RelPath::from_unix_str(directory)?.as_std_path())))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            ignored.iter().all(|directory| roots
                .iter()
                .any(|build| { directory.starts_with(build) || build.starts_with(directory) })),
            "Ignored Gradle directory is outside the selected builds; review settings and refresh before applying policy"
        );
        let directory_entries = ignored_directories.iter().map(|directory| {
            let path = RelPath::from_unix_str(directory)?;
            ensure!(Self::directory_is_visible(snapshot, path),
                "Ignored Gradle directory is unavailable, private or aliased; review settings and refresh before applying policy");
            snapshot.entry_for_path(path).cloned().context("Ignored Gradle directory is unavailable; refresh before applying policy")
        }).collect::<Result<Vec<_>>>()?;
        let inputs = Self::selected_inputs(snapshot, worktree_id, settings)?;
        let entries = inputs
            .iter()
            .map(|path| {
                snapshot
                    .entry_for_path(&path.path)
                    .cloned()
                    .context("Selected Gradle input is unavailable; refresh before importing")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Self {
            root,
            worktree_id,
            settings: settings.clone(),
            default_query: None,
            cached_associations: None,
            inputs,
            entries,
            metadata_inputs: Vec::new(),
            ignored,
            directory_entries,
            roots,
            initialized_roots: Vec::new(),
            imports: Vec::new(),
            path_style: snapshot.path_style(),
        }))
    }

    pub(crate) fn for_default(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        settings: &settings::GradleBuildSettingsContent,
    ) -> Result<Self> {
        ensure!(
            settings.selected_build_files.is_none(),
            "Default Gradle policy requires default build selection; refresh before retrying"
        );
        let root = snapshot.abs_path().as_ref().to_path_buf();
        let directories = settings.ignored_projects.as_deref().unwrap_or_default();
        validate_saved_build_paths(Some(directories), 128, "Gradle", None)?;
        let mut ignored = Vec::new();
        let mut directory_entries = Vec::new();
        for directory in directories {
            let path = RelPath::from_unix_str(directory)?;
            ensure!(
                Self::directory_is_visible(snapshot, path),
                "Ignored Gradle directory is unavailable, private or aliased; review settings and refresh before applying policy"
            );
            directory_entries.push(snapshot.entry_for_path(path).cloned().context(
                "Ignored Gradle directory is unavailable; refresh before applying policy",
            )?);
            ignored.push(root.join(path.as_std_path()));
        }
        Ok(Self {
            root,
            worktree_id,
            settings: settings.clone(),
            default_query: Some(GradleDefaultImportRootsQuery::new(snapshot, worktree_id)?),
            cached_associations: None,
            inputs: Vec::new(),
            entries: Vec::new(),
            metadata_inputs: Vec::new(),
            ignored,
            directory_entries,
            roots: Vec::new(),
            initialized_roots: Vec::new(),
            imports: Vec::new(),
            path_style: snapshot.path_style(),
        })
    }

    pub(crate) fn for_metadata(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        settings: &settings::GradleBuildSettingsContent,
    ) -> Result<Self> {
        if let Some(plan) = Self::for_selection_mode(snapshot, worktree_id, settings, true)? {
            return Ok(plan);
        }
        let mut plan = Self::for_default(snapshot, worktree_id, settings)?;
        let candidates = plan
            .default_query
            .take()
            .context("Gradle input authority is unavailable")?;
        plan.inputs = candidates.inputs;
        plan.entries = plan
            .inputs
            .iter()
            .map(|input| {
                snapshot.entry_for_path(&input.path).cloned().context(
                    "Current Gradle input is unavailable; refresh before synchronizing metadata",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(plan)
    }

    pub(crate) fn metadata_commands_available(server: &LanguageServer) -> bool {
        server
            .capabilities()
            .execute_command_provider
            .as_ref()
            .is_some_and(|provider| {
                [
                    "groovy.gradle.projectDetails.v4",
                    "groovy.gradle.importedBuildRoots",
                    "groovy.gradle.importedProjects",
                ]
                .iter()
                .all(|required| provider.commands.iter().any(|command| command == required))
            })
    }

    pub(crate) fn resolve_metadata_associations(
        &mut self,
        rows: Vec<ScopedGradleAssociation>,
        snapshot: &Snapshot,
        server: &LanguageServer,
    ) -> Result<()> {
        if self.settings.selected_build_files.is_none() {
            let roots = rows
                .iter()
                .map(|row| row.build.clone())
                .collect::<BTreeSet<_>>();
            ensure!(
                roots.iter().all(|root| self.inputs.iter().any(|input| {
                    self.root.join(
                        input
                            .path
                            .parent()
                            .unwrap_or(RelPath::empty())
                            .as_std_path(),
                    ) == *root
                })),
                "Current Gradle build inputs are unavailable; refresh or reimport before synchronizing metadata"
            );
            self.roots = roots.into_iter().collect();
            self.bind_initialized_folders(server)?;
        }
        self.resolve_cached_associations(rows, snapshot, server)?;
        let mut paths = BTreeSet::new();
        for entry in &self.directory_entries {
            for name in [
                "build.gradle",
                "build.gradle.kts",
                "settings.gradle",
                "settings.gradle.kts",
            ] {
                paths.insert(entry.path.join(RelPath::from_unix_str(name)?));
            }
        }
        ensure!(
            paths.len() <= MAX_JDTLS_MODULES * 8,
            "Gradle input authority exceeds its bound; narrow the project before synchronizing metadata"
        );
        self.metadata_inputs = paths.into_iter().map(|path| {
            let entry = snapshot.entry_for_path(&path).cloned();
            ensure!(entry.as_ref().is_none_or(|entry| entry.is_file() && !entry.is_ignored
                && entry.canonical_path.is_none() && remote_path_is_visible(&path, snapshot)),
                "Gradle module input is private, aliased or unavailable; review visibility and refresh before synchronizing metadata");
            Ok((path.into(), entry))
        }).collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub(crate) fn metadata_queries(&self) -> Result<Vec<ScopedGradleMetadataQuery>> {
        let rows = self.cached_associations.as_ref().context(
            "Gradle cached ownership is unresolved; refresh before synchronizing metadata",
        )?;
        if self.is_empty_selection() {
            return Ok(Vec::new());
        }
        self.imports.iter().filter(|import| !self.ignored.iter().any(|ignored| import.root.starts_with(ignored)))
            .map(|import| {
                let (workspace, build) = import.scoped_roots()?;
                let modules = rows.iter().filter(|row| row.build == build)
                    .map(|row| row.project.clone()).collect::<BTreeSet<_>>();
                ensure!(!modules.is_empty(),
                    "Selected Gradle build is not currently imported; apply its selection and refresh before synchronizing metadata");
                Ok(ScopedGradleMetadataQuery {worktree_root:self.root.clone(),workspace:workspace.to_path_buf(),build:build.to_path_buf(),modules})
            }).collect()
    }

    pub(crate) fn default_query(&self) -> Option<&GradleDefaultImportRootsQuery> {
        self.default_query.as_ref()
    }

    pub(crate) fn resolve_defaults(
        &mut self,
        roots: BTreeSet<PathBuf>,
        server: &LanguageServer,
    ) -> Result<()> {
        self.validate_initialized_folders(server)?;
        ensure!(
            self.default_query.is_some() && self.roots.is_empty() && !roots.is_empty(),
            "Gradle default roots are empty or unresolved; preserve settings, select explicit builds or refresh before retrying"
        );
        ensure!(
            self.default_query
                .as_ref()
                .is_some_and(|query| roots.len() <= 128
                    && roots.iter().all(|root| {
                        query.candidates.contains(root)
                            && roots.iter().all(|other| {
                                root == other
                                    || (!root.starts_with(other) && !other.starts_with(root))
                            })
                    })),
            "Gradle default roots are invalid or ambiguous; refresh before retrying"
        );
        ensure!(
            self.ignored.iter().all(|directory| roots
                .iter()
                .any(|build| { directory.starts_with(build) || build.starts_with(directory) })),
            "Ignored Gradle directory is outside the provider default builds; review settings and refresh before applying policy"
        );
        self.roots = roots.into_iter().collect();
        self.bind_initialized_folders(server)
    }

    pub(crate) fn supports_default_policy(server: &LanguageServer) -> bool {
        Self::policy_commands_available(server)
            && server
                .capabilities()
                .execute_command_provider
                .as_ref()
                .is_some_and(|provider| {
                    provider
                        .commands
                        .iter()
                        .any(|command| command == "groovy.gradle.defaultImportRoots")
                })
    }

    fn for_empty(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        settings: &settings::GradleBuildSettingsContent,
    ) -> Result<Self> {
        SavedGradleSelection::new(
            &GradleSelectionIntent {
                selected_build_files: settings.selected_build_files.as_deref(),
                ignored_projects: settings.ignored_projects.as_deref().unwrap_or_default(),
            },
            worktree_id,
            &[],
        )?;
        ensure!(
            Self::directory_is_visible(snapshot, RelPath::empty()),
            "Gradle project directory is unavailable, private or aliased; refresh before applying selection"
        );
        let directory_entries = vec![snapshot.entry_for_path(RelPath::empty()).cloned().context(
            "Gradle project directory is unavailable; refresh before applying selection",
        )?];
        let inputs = Self::selected_inputs(snapshot, worktree_id, settings)?;
        let entries = inputs
            .iter()
            .map(|input| {
                snapshot.entry_for_path(&input.path).cloned().context(
                    "Current Gradle input is unavailable; refresh before applying selection",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            root: snapshot.abs_path().as_ref().to_path_buf(),
            worktree_id,
            settings: settings.clone(),
            default_query: None,
            cached_associations: None,
            inputs,
            entries,
            metadata_inputs: Vec::new(),
            ignored: Vec::new(),
            directory_entries,
            roots: Vec::new(),
            initialized_roots: Vec::new(),
            imports: Vec::new(),
            path_style: snapshot.path_style(),
        })
    }

    pub(crate) fn is_empty_selection(&self) -> bool {
        self.settings
            .selected_build_files
            .as_ref()
            .is_some_and(Vec::is_empty)
    }

    pub(crate) fn required_commands(&self) -> &'static [&'static str] {
        if self.is_empty_selection() {
            &[
                "groovy.gradle.importedBuildRoots",
                "groovy.gradle.importedProjects",
                "groovy.gradle.unlinkSelected",
            ]
        } else {
            &[
                "groovy.gradle.importSelected.v2",
                "groovy.gradle.importedProjects",
            ]
        }
    }

    pub(crate) fn supports_cached_reconciliation(server: &LanguageServer) -> bool {
        server
            .capabilities()
            .execute_command_provider
            .as_ref()
            .is_some_and(|provider| {
                [
                    "groovy.gradle.importedBuildRoots",
                    "groovy.gradle.importedProjects",
                    "groovy.gradle.unlinkSelected",
                ]
                .iter()
                .all(|required| provider.commands.iter().any(|command| command == required))
            })
    }

    fn association_is_deselected(&self, row: &ScopedGradleAssociation) -> bool {
        self.is_empty_selection() || !self.roots.contains(&row.build)
    }

    pub(crate) async fn query_cached_associations(
        &self,
        scope: &Path,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<Vec<ScopedGradleAssociation>> {
        self.validate_initialized_folders(&server)?;
        ensure!(
            self.initialized_roots.iter().any(|root| root == scope),
            "Gradle cached association scope changed; refresh before applying selection"
        );
        let parameters = lsp::ExecuteCommandParams {
            command: "groovy.gradle.importedBuildRoots".into(),
            arguments: vec![
                json!({"rootUri": selected_import_uri(scope, true)?, "cachedOnly": true}),
            ],
            ..Default::default()
        };
        let value = request_selected_jdtls_phase(server, &parameters, request_timeout, "Gradle", "cached association read",
            "Current Gradle build ownership is unavailable; check extension setup and refresh before applying selection").await?;
        self.validate_cached_association_reply(scope, value)
    }

    fn validate_cached_association_reply(
        &self,
        scope: &Path,
        value: Value,
    ) -> Result<Vec<ScopedGradleAssociation>> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle cached association data is invalid, incomplete or stale; preserve settings and refresh before applying selection"
            )
        };
        let bounded = |value: &Value| {
            value
                .as_str()
                .is_some_and(|text| text.len() <= 4096 && !text.chars().any(char::is_control))
        };
        let object = value.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 7
                && object.get("rootUri").is_some_and(bounded)
                && object
                    .get("projects")
                    .and_then(Value::as_array)
                    .is_some_and(|projects| projects.len() <= 128
                        && projects.iter().all(|project| project
                            .as_object()
                            .is_some_and(|object| object.len() == 3)
                            && project.get("projectUri").is_some_and(bounded)
                            && project.get("buildRootUri").is_some_and(bounded))),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Reply {
            version: u32,
            root_uri: String,
            cached_only: bool,
            read_only: bool,
            partial: bool,
            full_synchronization_known: bool,
            projects: Vec<Record>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Record {
            project_uri: String,
            build_root_uri: String,
            java_nature: bool,
        }
        let reply: Reply = serde_json::from_value(value).map_err(|_| invalid())?;
        ensure!(
            reply.version == 1
                && reply.cached_only
                && reply.read_only
                && !reply.partial
                && !reply.full_synchronization_known
                && imported_build_identity_path(&reply.root_uri).map_err(|_| invalid())? == scope,
            "{}",
            invalid()
        );
        let mut identities = BTreeSet::new();
        let mut rows = Vec::with_capacity(reply.projects.len());
        for record in reply.projects {
            let project =
                imported_build_identity_path(&record.project_uri).map_err(|_| invalid())?;
            let build =
                imported_build_identity_path(&record.build_root_uri).map_err(|_| invalid())?;
            ensure!(
                scope.starts_with(&self.root)
                    && build.starts_with(scope)
                    && project.starts_with(&build)
                    && identities.insert(project.clone()),
                "{}",
                invalid()
            );
            rows.push(ScopedGradleAssociation {
                scope: scope.to_path_buf(),
                project,
                build,
                java_nature: record.java_nature,
            });
        }
        rows.sort_by(|left, right| left.project.cmp(&right.project));
        Ok(rows)
    }

    pub(crate) fn resolve_cached_associations(
        &mut self,
        rows: Vec<ScopedGradleAssociation>,
        snapshot: &Snapshot,
        server: &LanguageServer,
    ) -> Result<()> {
        self.validate_initialized_folders(server)?;
        self.validate_current(snapshot, self.worktree_id)?;
        ensure!(
            self.cached_associations.is_none() && rows.len() <= MAX_JDTLS_MODULES,
            "Gradle cached ownership authority changed; refresh before retrying"
        );
        let mut identities = BTreeSet::new();
        let mut directories = self
            .directory_entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect::<BTreeSet<_>>();
        let mut roots = BTreeSet::new();
        for row in &rows {
            ensure!(
                self.initialized_roots.contains(&row.scope)
                    && row.build.starts_with(&row.scope)
                    && row.project.starts_with(&row.build)
                    && identities.insert(row.project.clone()),
                "Gradle cached association identities overlap or are outside this host; refresh before applying selection"
            );
            ensure!(
                self.is_empty_selection()
                    || self.roots.iter().all(|selected| selected == &row.build
                        || !selected.starts_with(&row.build) && !row.build.starts_with(selected)),
                "Selected and deselected Gradle build ownership overlaps; review composite and nested build selection before refreshing"
            );
            roots.insert(row.build.clone());
            ensure!(
                roots.len() <= 128,
                "Gradle build ownership exceeds its bound; narrow the project before applying selection"
            );
            for path in [&row.project, &row.build] {
                let project = confined_project_path(path, &self.root, self.worktree_id, self.path_style)
                    .context("Gradle cached association is outside this project; refresh before applying selection")?;
                ensure!(
                    Self::directory_is_visible(snapshot, &project.path),
                    "Gradle cached ownership is private, aliased or unavailable; review visibility and refresh before applying selection"
                );
                if directories.insert(project.path.clone()) {
                    self.directory_entries
                        .push(snapshot.entry_for_path(&project.path).cloned().context(
                        "Gradle cached project is unavailable; refresh before applying selection",
                    )?);
                }
            }
        }
        if self.is_empty_selection() {
            self.roots = roots.into_iter().collect();
        }
        self.cached_associations = Some(rows);
        Ok(())
    }

    pub(crate) fn validate_cached_associations(
        &self,
        scope: &Path,
        current: &[ScopedGradleAssociation],
    ) -> Result<()> {
        let expected = self
            .cached_associations
            .as_ref()
            .context("Gradle cached ownership is unresolved; refresh before retrying")?;
        ensure!(
            expected
                .iter()
                .filter(|row| row.scope == scope)
                .eq(current.iter()),
            "Gradle cached build ownership changed; selection may be partial, preserve settings and refresh before retrying"
        );
        Ok(())
    }

    pub(crate) fn validate_cached_namespace(
        &self,
        scope: &Path,
        paths: &[ProjectPath],
    ) -> Result<()> {
        let rows = self
            .cached_associations
            .as_ref()
            .context("Gradle cached ownership is unresolved; refresh before retrying")?;
        let expected = rows
            .iter()
            .filter(|row| row.scope == scope)
            .map(|row| row.project.clone())
            .collect::<BTreeSet<_>>();
        let actual = paths
            .iter()
            .map(|path| {
                ensure!(
                    path.worktree_id == self.worktree_id,
                    "Gradle namespace host changed; refresh before applying selection"
                );
                Ok(self.root.join(path.path.as_std_path()))
            })
            .collect::<Result<BTreeSet<_>>>()?;
        ensure!(
            expected == actual,
            "Gradle cached associations do not cover current imported identities; preserve settings and refresh before applying selection"
        );
        Ok(())
    }

    pub(crate) fn cached_unlink_scopes(&self) -> Result<Vec<PathBuf>> {
        let rows = self
            .cached_associations
            .as_ref()
            .context("Gradle cached ownership is unresolved; refresh before retrying")?;
        Ok(self
            .initialized_roots
            .iter()
            .filter(|scope| {
                rows.iter()
                    .any(|row| &row.scope == *scope && self.association_is_deselected(row))
            })
            .cloned()
            .collect())
    }

    pub(crate) fn cached_unlink(&self, scope: &Path) -> Result<SelectedProjectUnlink> {
        let rows = self
            .cached_associations
            .as_ref()
            .context("Gradle cached ownership is unresolved; refresh before retrying")?;
        let paths = rows
            .iter()
            .filter(|row| row.scope == scope && self.association_is_deselected(row))
            .map(|row| row.project.clone())
            .collect::<Vec<_>>();
        SelectedProjectUnlink::new(JdtlsUnlinkProvider::Gradle, scope, &paths)
    }

    fn selected_inputs(
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
        gradle: &settings::GradleBuildSettingsContent,
    ) -> Result<Vec<ProjectPath>> {
        let mut gradle = gradle.clone();
        if gradle
            .selected_build_files
            .as_ref()
            .is_some_and(Vec::is_empty)
        {
            gradle.selected_build_files = None;
        }
        gradle.ignored_projects = None;
        let settings = settings::JvmBuildSettingsContent {
            maven: Some(settings::MavenBuildSettingsContent {
                selected_poms: Some(Vec::new()),
                ..Default::default()
            }),
            gradle: Some(gradle),
        };
        default_jvm_build_inputs(snapshot, worktree_id, Some(&settings))
    }

    fn directory_is_visible(snapshot: &Snapshot, path: &RelPath) -> bool {
        remote_path_is_visible(path, snapshot)
            && path.ancestors().all(|ancestor| {
                snapshot.entry_for_path(ancestor).is_some_and(|entry| {
                    entry.is_dir() && !entry.is_ignored && entry.canonical_path.is_none()
                })
            })
    }

    pub(crate) fn supports_policy(&self, server: &LanguageServer) -> bool {
        (!self.imports.is_empty()
            || (self.default_query.is_some() && !self.initialized_roots.is_empty()))
            && Self::policy_commands_available(server)
    }

    fn policy_commands_available(server: &LanguageServer) -> bool {
        server
            .capabilities()
            .execute_command_provider
            .as_ref()
            .is_some_and(|provider| {
                [
                    "groovy.gradle.rootState.v2",
                    "groovy.gradle.applyRootState.v2",
                    "groovy.gradle.moduleStates.v2",
                    "groovy.gradle.applyModuleStates.v2",
                ]
                .iter()
                .all(|command| {
                    provider
                        .commands
                        .iter()
                        .any(|candidate| candidate.as_str() == *command)
                })
            })
    }

    pub(crate) fn requires_policy(&self) -> bool {
        !self.ignored.is_empty()
    }

    pub(crate) fn policy_for_build(&self, build: &Path) -> Result<(bool, Vec<PathBuf>)> {
        ensure!(
            self.roots.iter().any(|root| root == build),
            "Gradle policy build is no longer selected; refresh before retrying"
        );
        let ignored_root = self
            .ignored
            .iter()
            .any(|directory| build.starts_with(directory));
        let ignored = self
            .ignored
            .iter()
            .filter(|directory| directory.starts_with(build) && directory.as_path() != build)
            .cloned()
            .collect();
        Ok((ignored_root, ignored))
    }

    pub(crate) fn validate_ignored_namespace(&self, paths: &[ProjectPath]) -> Result<()> {
        ensure!(
            !self.is_empty_selection() || paths.is_empty(),
            "Gradle resources remain active after empty selection; unlink may be partial, preserve settings and refresh before retrying"
        );
        ensure!(
            self.cached_associations.is_none()
                || paths.iter().all(|path| self
                    .roots
                    .iter()
                    .any(|root| self.root.join(path.path.as_std_path()).starts_with(root))),
            "Deselected Gradle resources remain active; unlink may be partial, preserve settings and refresh before retrying"
        );
        ensure!(
            paths.iter().all(|path| path.worktree_id == self.worktree_id
                && !self.ignored.iter().any(|directory| self
                    .root
                    .join(path.path.as_std_path())
                    .starts_with(directory))),
            "Ignored Gradle projects remain active or an orphan is unresolved; import may be partial, review provider state and refresh before retrying"
        );
        Ok(())
    }

    pub(crate) fn bind_initialized_folders(&mut self, server: &LanguageServer) -> Result<()> {
        let roots = imported_build_read_roots(
            server
                .workspace_folders()
                .iter()
                .map(|folder| folder.as_str().to_owned()),
            &self.root,
        )?;
        ensure!(
            self.default_query.is_none() || roots.iter().any(|root| root == &self.root),
            "Java tooling does not own the default Gradle query workspace; review initialized folders and restart before retrying"
        );
        ensure!(
            self.default_query.is_none()
                || self.initialized_roots.is_empty()
                || self.initialized_roots == roots,
            "Java initialized folders changed; scoped import may be partial, refresh before retrying"
        );
        ensure!(
            !self.is_empty_selection() || !roots.is_empty(),
            "Java tooling does not own a Gradle scope; review initialized folders and refresh before applying selection"
        );
        let imports = if self.is_empty_selection() {
            Vec::new()
        } else {
            self.roots.iter().map(|build| {
            let workspace = roots.iter().find(|root| build.starts_with(root)).context(
                "Java tooling does not own a selected Gradle build; review initialized folders and restart the provider before retrying"
            )?;
            SelectedGradleImport::new_scoped(workspace, build)
        }).collect::<Result<Vec<_>>>()?
        };
        self.initialized_roots = roots;
        self.imports = imports;
        Ok(())
    }

    pub(crate) fn validate_current(
        &self,
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
    ) -> Result<()> {
        ensure!(
            snapshot.id() == worktree_id
                && worktree_id == self.worktree_id
                && snapshot.abs_path().as_ref() == self.root
                && snapshot.path_style() == self.path_style
                && (self.default_query.is_some()
                    || Self::selected_inputs(snapshot, worktree_id, &self.settings)?
                        == self.inputs)
                && self
                    .entries
                    .iter()
                    .all(|entry| snapshot.entry_for_path(&entry.path) == Some(entry))
                && self.metadata_inputs.iter().all(|(path, expected)| {
                    snapshot.entry_for_path(path) == expected.as_ref()
                        && path
                            .parent()
                            .is_some_and(|parent| Self::directory_is_visible(snapshot, parent))
                        && expected.as_ref().is_none_or(|entry| {
                            entry.is_file()
                                && !entry.is_ignored
                                && entry.canonical_path.is_none()
                                && remote_path_is_visible(path, snapshot)
                        })
                })
                && self.directory_entries.iter().all(|entry| {
                    Self::directory_is_visible(snapshot, &entry.path)
                        && snapshot.entry_for_path(&entry.path).is_some_and(|current| {
                            current.id == entry.id
                                && current.inode == entry.inode
                                && current.is_dir()
                        })
                }),
            "Selected Gradle inputs changed; import may be partial, refresh before retrying"
        );
        if let Some(query) = &self.default_query {
            query.validate_current(snapshot, worktree_id)?;
        }
        Ok(())
    }

    pub(crate) fn validate_initialized_folders(&self, server: &LanguageServer) -> Result<()> {
        ensure!(
            !self.initialized_roots.is_empty()
                && (if self.is_empty_selection() {
                    self.imports.is_empty()
                } else {
                    self.imports.len() == self.roots.len()
                })
                && imported_build_read_roots(
                    server
                        .workspace_folders()
                        .iter()
                        .map(|folder| folder.as_str().to_owned()),
                    &self.root,
                )? == self.initialized_roots,
            "Java initialized folders changed; scoped import may be partial, refresh before retrying"
        );
        Ok(())
    }

    pub(crate) fn covers_imported_projects(&self, projects: &[JdtlsProject]) -> bool {
        projects
            .iter()
            .filter(|project| project.membership.is_imported())
            .all(|project| {
                project.module_root.worktree_id == self.worktree_id
                    && self.initialized_roots.iter().any(|root| {
                        self.root
                            .join(project.module_root.path.as_std_path())
                            .starts_with(root)
                    })
            })
    }

    pub(crate) fn imports(&self) -> &[SelectedGradleImport] {
        &self.imports
    }

    pub(crate) fn initialized_roots(&self) -> &[PathBuf] {
        &self.initialized_roots
    }

    pub(crate) async fn query_current_ownership(
        &self,
        root: &Path,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
        identities: &mut BTreeSet<PathBuf>,
    ) -> Result<Vec<ProjectPath>> {
        self.validate_initialized_folders(&server)?;
        ensure!(
            self.initialized_roots.iter().any(|scope| scope == root),
            "Gradle imported scope is no longer initialized; refresh before retrying"
        );
        let parameters = lsp::ExecuteCommandParams {
            command: "groovy.gradle.importedProjects".into(),
            arguments: vec![
                json!({"rootUri": selected_import_uri(root, true)?, "cachedOnly": true}),
            ],
            ..Default::default()
        };
        let value = request_selected_jdtls_phase(
            server, &parameters, request_timeout, "Gradle", "imported scope read",
            "Gradle imported scope is unavailable; check current provider setup and refresh before importing",
        ).await?;
        self.validate_ownership_reply(root, value, identities)
    }

    fn validate_ownership_reply(
        &self,
        root: &Path,
        value: Value,
        identities: &mut BTreeSet<PathBuf>,
    ) -> Result<Vec<ProjectPath>> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle imported scope is incomplete, invalid or changed; import may be partial, refresh before retrying"
            )
        };
        let object = value.as_object().ok_or_else(invalid)?;
        ensure!(
            object.len() == 5
                && object
                    .get("projects")
                    .and_then(Value::as_array)
                    .is_some_and(|projects| {
                        projects.len() <= MAX_JDTLS_MODULES
                            && projects.iter().all(|project| {
                                project.as_object().is_some_and(|object| object.len() == 2)
                                    && project
                                        .get("projectUri")
                                        .and_then(Value::as_str)
                                        .is_some_and(|uri| uri.len() <= 4096)
                            })
                    })
                && object
                    .get("rootUri")
                    .and_then(Value::as_str)
                    .is_some_and(|uri| uri.len() <= 4096),
            "{}",
            invalid()
        );
        let reply: ImportedBuildKindsReply =
            serde_json::from_value(value).map_err(|_| invalid())?;
        ensure!(
            reply.version == 1
                && !reply.partial
                && !reply.full_synchronization_known
                && imported_build_identity_path(&reply.root_uri).map_err(|_| invalid())? == root,
            "{}",
            invalid()
        );
        let mut projects = Vec::with_capacity(reply.projects.len());
        for record in reply.projects {
            let path = imported_build_identity_path(&record.project_uri).map_err(|_| invalid())?;
            ensure!(
                path.starts_with(root)
                    && identities.len() < MAX_JDTLS_MODULES
                    && identities.insert(path.clone()),
                "{}",
                invalid()
            );
            ensure!(
                self.roots.iter().any(|root| path.starts_with(root))
                    || self.cached_associations.as_ref().is_some_and(|rows| rows
                        .iter()
                        .any(|row| row.scope == root && row.project == path)),
                "An active Gradle project is outside the selected builds; review current unlink policy and refresh before importing another selection"
            );
            let project =
                confined_project_path(&path, &self.root, self.worktree_id, self.path_style)
                    .ok_or_else(invalid)?;
            ensure!(
                !project.path.as_unix_str().chars().any(char::is_control),
                "{}",
                invalid()
            );
            projects.push(project);
        }
        Ok(projects)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MavenDefaultImportRootsQuery {
    root: PathBuf,
    worktree_id: WorktreeId,
    inputs: Vec<ProjectPath>,
    candidates: BTreeSet<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct MavenDefaultImportRoots {
    pub poms: BTreeSet<PathBuf>,
    pub retained_poms: BTreeSet<PathBuf>,
}

impl MavenDefaultImportRootsQuery {
    pub(crate) fn new(snapshot: &Snapshot, worktree_id: WorktreeId) -> Result<Self> {
        ensure!(
            snapshot.id() == worktree_id,
            "Maven candidate host changed; refresh before retrying"
        );
        let root = snapshot.abs_path().as_ref().to_path_buf();
        let inputs = default_maven_import_inputs(snapshot, worktree_id)?;
        let candidates = inputs
            .iter()
            .map(|input| root.join(input.path.as_std_path()))
            .collect::<BTreeSet<_>>();
        let root_uri = selected_import_uri(&root, true)?;
        let candidate_uris = candidates
            .iter()
            .map(|pom| selected_import_uri(pom, false))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            root,
            worktree_id,
            inputs,
            candidates,
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.maven.defaultImportRoots".into(),
                arguments: vec![
                    json!({"rootUri": root_uri, "candidatePomUris": candidate_uris, "cachedOnly": true}),
                ],
                ..Default::default()
            },
        })
    }

    pub(crate) fn validate_current(
        &self,
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
    ) -> Result<()> {
        ensure!(
            snapshot.id() == worktree_id
                && worktree_id == self.worktree_id
                && snapshot.abs_path().as_ref() == self.root
                && default_maven_import_inputs(snapshot, worktree_id)? == self.inputs,
            "Maven candidate inputs changed; refresh before retrying"
        );
        Ok(())
    }

    pub(crate) async fn query(
        &self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<MavenDefaultImportRoots> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Maven", "default-root discovery",
            "Maven default-root discovery failed; check extension/JDTLS setup and current POMs, then refresh before retrying",
        ).await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: Value) -> Result<MavenDefaultImportRoots> {
        let invalid = || {
            anyhow::anyhow!(
                "Maven default-root response is invalid or stale; refresh project data and check extension setup before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(object.len() == 8, "{}", invalid());
        for name in ["rootPomUris", "retainedPomUris"] {
            ensure!(
                object
                    .get(name)
                    .and_then(Value::as_array)
                    .is_some_and(|values| values.len() <= 32
                        && values.iter().all(|value| value
                            .as_str()
                            .is_some_and(
                                |text| text.len() <= 4096 && !text.chars().any(char::is_control)
                            ))),
                "{}",
                invalid()
            );
        }
        ensure!(
            object
                .get("rootUri")
                .and_then(Value::as_str)
                .is_some_and(|text| text.len() <= 4096 && !text.chars().any(char::is_control)),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Response {
            version: u32,
            root_uri: String,
            root_pom_uris: Vec<String>,
            retained_pom_uris: Vec<String>,
            cached_only: bool,
            read_only: bool,
            partial: bool,
            full_synchronization_known: bool,
        }
        let response: Response = serde_json::from_value(response).map_err(|_| invalid())?;
        ensure!(
            response.version == 1
                && response.cached_only
                && response.read_only
                && !response.partial
                && !response.full_synchronization_known
                && selected_import_path(&response.root_uri, true).map_err(|_| invalid())?
                    == self.root,
            "{}",
            invalid()
        );
        let decode = |uris: Vec<String>| -> Result<BTreeSet<PathBuf>> {
            let mut paths = BTreeSet::new();
            for uri in uris {
                let path = selected_import_path(&uri, false).map_err(|_| invalid())?;
                ensure!(
                    self.candidates.contains(&path) && paths.insert(path),
                    "{}",
                    invalid()
                );
            }
            Ok(paths)
        };
        let poms = decode(response.root_pom_uris)?;
        let retained_poms = decode(response.retained_pom_uris)?;
        ensure!(retained_poms.is_subset(&poms), "{}", invalid());
        Ok(MavenDefaultImportRoots {
            poms,
            retained_poms,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GradleDefaultImportRootsQuery {
    root: PathBuf,
    worktree_id: WorktreeId,
    inputs: Vec<ProjectPath>,
    entries: Vec<worktree::Entry>,
    candidates: BTreeSet<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
}

impl GradleDefaultImportRootsQuery {
    pub(crate) fn new(snapshot: &Snapshot, worktree_id: WorktreeId) -> Result<Self> {
        ensure!(
            snapshot.id() == worktree_id,
            "Gradle candidate host changed; refresh before retrying"
        );
        let invalid = || {
            anyhow::anyhow!(
                "Gradle default-root inputs are invalid or unavailable; check current visible build files and refresh"
            )
        };
        let root = snapshot.abs_path().as_ref().to_path_buf();
        let inputs = default_gradle_import_inputs(snapshot, worktree_id)?;
        let entries = inputs
            .iter()
            .map(|input| {
                snapshot
                    .entry_for_path(&input.path)
                    .cloned()
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidates = BTreeSet::new();
        let mut candidate_uris = Vec::new();
        for input in &inputs {
            let file = root.join(input.path.as_std_path());
            let directory = file.parent().ok_or_else(invalid)?.to_path_buf();
            selected_import_uri(&directory, true).map_err(|_| invalid())?;
            candidates.insert(directory);
            candidate_uris.push(selected_import_uri(&file, false).map_err(|_| invalid())?);
        }
        let root_uri = selected_import_uri(&root, true).map_err(|_| invalid())?;
        Ok(Self {
            root,
            worktree_id,
            inputs,
            entries,
            candidates,
            parameters: lsp::ExecuteCommandParams {
                command: "groovy.gradle.defaultImportRoots".into(),
                arguments: vec![json!({
                    "rootUri": root_uri,
                    "candidateBuildFileUris": candidate_uris,
                    "cachedOnly": true,
                })],
                ..Default::default()
            },
        })
    }

    pub(crate) fn validate_current(
        &self,
        snapshot: &Snapshot,
        worktree_id: WorktreeId,
    ) -> Result<()> {
        ensure!(
            snapshot.id() == worktree_id
                && worktree_id == self.worktree_id
                && snapshot.abs_path().as_ref() == self.root
                && default_gradle_import_inputs(snapshot, worktree_id)? == self.inputs
                && self
                    .entries
                    .iter()
                    .all(|entry| { snapshot.entry_for_path(&entry.path) == Some(entry) }),
            "Gradle candidate inputs changed; refresh before retrying"
        );
        Ok(())
    }

    pub(crate) async fn query(
        &self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<BTreeSet<PathBuf>> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Gradle", "default-root discovery",
            "Gradle default-root discovery failed; check extension/JDTLS setup and current build files, then refresh before retrying",
        ).await?;
        self.validate_response(response)
    }

    fn validate_response(&self, response: Value) -> Result<BTreeSet<PathBuf>> {
        let invalid = || {
            anyhow::anyhow!(
                "Gradle default-root response is invalid or stale; refresh project data and check extension setup before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        ensure!(object.len() == 7, "{}", invalid());
        ensure!(
            object
                .get("buildRootUris")
                .and_then(Value::as_array)
                .is_some_and(|values| {
                    values.len() <= 128
                        && values.iter().all(|value| {
                            value.as_str().is_some_and(|text| {
                                text.len() <= 4096 && !text.chars().any(char::is_control)
                            })
                        })
                })
                && object
                    .get("rootUri")
                    .and_then(Value::as_str)
                    .is_some_and(|text| {
                        text.len() <= 4096 && !text.chars().any(char::is_control)
                    }),
            "{}",
            invalid()
        );
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Response {
            version: u32,
            root_uri: String,
            build_root_uris: Vec<String>,
            cached_only: bool,
            read_only: bool,
            partial: bool,
            full_synchronization_known: bool,
        }
        let response: Response = serde_json::from_value(response).map_err(|_| invalid())?;
        ensure!(
            response.version == 1
                && response.cached_only
                && response.read_only
                && !response.partial
                && !response.full_synchronization_known
                && selected_import_path(&response.root_uri, true).map_err(|_| invalid())?
                    == self.root,
            "{}",
            invalid()
        );
        let mut roots = BTreeSet::new();
        for uri in response.build_root_uris {
            let path = selected_import_path(&uri, true).map_err(|_| invalid())?;
            ensure!(
                self.candidates.contains(&path)
                    && roots.iter().all(|root: &PathBuf| {
                        !path.starts_with(root) && !root.starts_with(&path)
                    })
                    && roots.insert(path),
                "{}",
                invalid()
            );
        }
        Ok(roots)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MavenReactorImport {
    root: PathBuf,
    pom: PathBuf,
    excluded_projects: BTreeSet<PathBuf>,
    parameters: lsp::ExecuteCommandParams,
    pom_masks: Option<MavenPomMaskContract>,
}

impl MavenReactorImport {
    pub(crate) fn root_pom(&self) -> &Path {
        &self.pom
    }

    pub(crate) fn new(root: &Path, pom: &Path, excluded_projects: &[PathBuf]) -> Result<Self> {
        Self::new_with_masks(root, pom, excluded_projects, None)
    }

    pub(crate) fn new_with_masks(
        root: &Path,
        pom: &Path,
        excluded_projects: &[PathBuf],
        patterns: Option<&[String]>,
    ) -> Result<Self> {
        let invalid = || {
            anyhow::anyhow!(
                "Selected Maven reactor requires a visible root POM and distinct excluded directories inside the project folder; review selection and retry"
            )
        };
        let root_uri = selected_import_uri(root, true).map_err(|_| invalid())?;
        let pom_uri = selected_import_uri(pom, false).map_err(|_| invalid())?;
        ensure!(
            pom.starts_with(root)
                && pom.file_name().is_some_and(|name| name == "pom.xml")
                && excluded_projects.len() <= 128,
            "{}",
            invalid()
        );
        let mut exclusions = BTreeSet::new();
        let mut excluded_uris = Vec::with_capacity(excluded_projects.len());
        for directory in excluded_projects {
            ensure!(
                directory.starts_with(root)
                    && !pom.starts_with(directory)
                    && exclusions.insert(directory.clone()),
                "{}",
                invalid()
            );
            excluded_uris.push(selected_import_uri(directory, true).map_err(|_| invalid())?);
        }
        let pom_masks = patterns.map(MavenPomMaskContract::new).transpose()?;
        let mut request = json!({"rootUri": root_uri, "pomUri": pom_uri,
            "excludedProjectUris": excluded_uris, "offline": true});
        if let Some(masks) = &pom_masks {
            matches_maven_host_pom(masks, pom).map_err(|_| invalid())?;
            request
                .as_object_mut()
                .context("Maven request is unavailable; refresh before retrying")?
                .insert("pomMasks".into(), masks.to_request());
        }
        let command = if pom_masks.is_some() {
            "groovy.maven.importReactorWithMasks"
        } else {
            "groovy.maven.importReactor"
        };
        Ok(Self {
            root: root.to_path_buf(),
            pom: pom.to_path_buf(),
            excluded_projects: exclusions,
            parameters: lsp::ExecuteCommandParams {
                command: command.into(),
                arguments: vec![request],
                ..Default::default()
            },
            pom_masks,
        })
    }

    pub(crate) async fn execute_phase(
        self,
        server: Arc<LanguageServer>,
        request_timeout: Duration,
    ) -> Result<MavenReactorPhaseFacts> {
        let response = request_selected_jdtls_phase(
            server, &self.parameters, request_timeout, "Maven", "import",
            "Selected Maven reactor import failed or may be partial; check host offline settings, cache and project selection, then refresh before retrying",
        ).await?;
        self.validate_receipt(response)
    }

    fn validate_receipt(&self, response: serde_json::Value) -> Result<MavenReactorPhaseFacts> {
        let invalid = || {
            anyhow::anyhow!(
                "Selected Maven reactor receipt is invalid or may be partial; refresh project data before retrying"
            )
        };
        let object = response.as_object().ok_or_else(invalid)?;
        let bounded_uri = |value: &serde_json::Value| {
            value
                .as_str()
                .is_some_and(|uri| uri.len() <= 4096 && !uri.chars().any(char::is_control))
        };
        let masked = self.pom_masks.is_some();
        let expected_fields = if masked { 8 } else { 7 };
        let expected_version = if masked { 2 } else { 1 };
        ensure!(
            object.len() == expected_fields
                && ["rootUri", "reactorPomUri"]
                    .iter()
                    .all(|field| object.get(*field).is_some_and(bounded_uri)),
            "{}",
            invalid()
        );
        for (field, minimum, maximum) in [
            ("importedPomUris", usize::from(!masked), 128),
            (
                "excludedProjectUris",
                self.excluded_projects.len(),
                self.excluded_projects.len(),
            ),
        ] {
            ensure!(
                object
                    .get(field)
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|values| {
                        values.len() >= minimum
                            && values.len() <= maximum
                            && values.iter().all(bounded_uri)
                    }),
                "{}",
                invalid()
            );
        }
        if masked {
            ensure!(
                object
                    .get("ignoredPomUris")
                    .and_then(Value::as_array)
                    .is_some_and(|values| {
                        values.len() <= 128
                            && values.iter().all(bounded_uri)
                            && object
                                .get("importedPomUris")
                                .and_then(Value::as_array)
                                .is_some_and(|imported| !values.is_empty() || !imported.is_empty())
                            && object
                                .get("importedPomUris")
                                .and_then(Value::as_array)
                                .is_some_and(|imported| values.len() + imported.len() <= 128)
                    }),
                "{}",
                invalid()
            );
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Receipt {
            version: u32,
            root_uri: String,
            reactor_pom_uri: String,
            imported_pom_uris: Vec<String>,
            #[serde(default)]
            ignored_pom_uris: Vec<String>,
            excluded_project_uris: Vec<String>,
            selected_import_phase_complete: bool,
            full_synchronization_known: bool,
        }
        let receipt: Receipt = serde_json::from_value(response).map_err(|_| invalid())?;
        ensure!(
            receipt.version == expected_version
                && receipt.selected_import_phase_complete
                && !receipt.full_synchronization_known
                && selected_import_path(&receipt.root_uri, true).map_err(|_| invalid())?
                    == self.root
                && selected_import_path(&receipt.reactor_pom_uri, false).map_err(|_| invalid())?
                    == self.pom,
            "{}",
            invalid()
        );
        let mut exclusions = BTreeSet::new();
        for uri in receipt.excluded_project_uris {
            ensure!(
                exclusions.insert(selected_import_path(&uri, true).map_err(|_| invalid())?),
                "{}",
                invalid()
            );
        }
        ensure!(exclusions == self.excluded_projects, "{}", invalid());
        let mut identities = BTreeSet::new();
        let mut facts = MavenReactorPhaseFacts::default();
        for (uris, ignored) in [
            (receipt.imported_pom_uris, false),
            (receipt.ignored_pom_uris, true),
        ] {
            for uri in uris {
                let pom = selected_import_path(&uri, false).map_err(|_| invalid())?;
                ensure!(
                    pom.starts_with(&self.root)
                        && pom.file_name().is_some_and(|name| name == "pom.xml")
                        && !self
                            .excluded_projects
                            .iter()
                            .any(|directory| pom.starts_with(directory))
                        && identities.insert(pom.clone()),
                    "{}",
                    invalid()
                );
                let matches = self
                    .pom_masks
                    .as_ref()
                    .map(|masks| matches_maven_host_pom(masks, &pom))
                    .transpose()
                    .map_err(|_| invalid())?
                    .unwrap_or(false);
                ensure!(matches == ignored, "{}", invalid());
                if ignored {
                    facts.ignored_poms.push(pom);
                } else {
                    facts.imported_poms.push(pom);
                }
            }
        }
        ensure!(identities.contains(&self.pom), "{}", invalid());
        Ok(facts)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MavenReactorPhaseFacts {
    pub imported_poms: Vec<PathBuf>,
    pub ignored_poms: Vec<PathBuf>,
}

fn matches_maven_host_pom(masks: &MavenPomMaskContract, pom: &Path) -> Result<bool> {
    let path = pom.to_str().context(
        "Maven POM path normalization is unavailable; refresh from the authoritative project host",
    )?;
    let portable = if PathStyle::local() == PathStyle::Windows {
        path.replace('\\', "/")
    } else {
        path.to_owned()
    };
    masks.matches_portable_host_path(&portable, PathStyle::local())
}

async fn request_selected_jdtls_phase(
    server: Arc<LanguageServer>,
    parameters: &lsp::ExecuteCommandParams,
    request_timeout: Duration,
    name: &'static str,
    operation: &'static str,
    failure_message: &'static str,
) -> Result<serde_json::Value> {
    ensure!(
        server
            .capabilities()
            .execute_command_provider
            .as_ref()
            .is_some_and(|provider| provider.commands.contains(&parameters.command)),
        "Selected {name} {operation} provider is unavailable; check extension setup and retry"
    );
    server
        .request::<lsp::request::ExecuteCommand>(parameters.clone(), request_timeout)
        .await
        .into_response()
        .map_err(|_| anyhow::anyhow!(failure_message))?
        .with_context(|| {
            format!("Selected {name} {operation} returned no receipt; refresh before retrying")
        })
}

fn selected_import_uri(path: &Path, directory: bool) -> Result<url::Url> {
    let path_text = path.to_str().context(
        "Selected Maven import file identity is invalid; select the project files again",
    )?;
    ensure!(
        path_text.len() <= 4096 && !path_text.chars().any(char::is_control),
        "Selected Maven import file identity is invalid or too long; select the project files again"
    );
    let normalized_separators;
    let path = if cfg!(windows) {
        normalized_separators = PathBuf::from(path_text.replace('/', "\\"));
        normalized_separators.as_path()
    } else {
        path
    };
    let normalized = path.components().collect::<PathBuf>();
    ensure!(
        path.is_absolute()
            && (normalized.as_os_str() == path.as_os_str()
                || directory && normalized.join("").as_os_str() == path.as_os_str())
            && !path.components().any(|component| matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )),
        "Selected Maven import paths must be absolute without aliases; select the project files again"
    );
    let mut uri = if directory {
        url::Url::from_directory_path(path)
    } else {
        url::Url::from_file_path(path)
    }
    .map_err(|()| {
        anyhow::anyhow!(
            "Selected Maven import file identity is unavailable; select the project files again"
        )
    })?;
    // The JVM providers use NIO URI paths, whose reserved set differs from WHATWG URLs.
    let java_path = uri
        .path()
        .replace('^', "%5E")
        .replace('[', "%5B")
        .replace(']', "%5D")
        .replace('|', "%7C");
    uri.set_path(&java_path);
    ensure!(
        uri.as_str().len() <= 4096
            && path
                .to_str()
                .is_some_and(|text| !text.chars().any(char::is_control)),
        "Selected Maven import file identity is invalid or too long; select the project files again"
    );
    Ok(uri)
}

fn selected_import_path(value: &str, directory: bool) -> Result<PathBuf> {
    let uri = url::Url::parse(value)?;
    ensure!(uri.as_str() == value, "Maven import URI contains an alias");
    let path = file_uri_path(value)?;
    selected_import_uri(&path, directory)?;
    Ok(path)
}

fn file_uri_path(uri: &str) -> Result<PathBuf> {
    url::Url::parse(uri)
        .ok()
        .filter(|uri| uri.query().is_none() && uri.fragment().is_none())
        .and_then(|uri| uri.to_file_path().ok())
        .filter(|path| {
            path.to_str()
                .is_some_and(|text| !text.chars().any(char::is_control))
        })
        .context("JVM build provider returned an invalid file URI")
}

fn gradle_model_identity_path(uri: &str) -> Result<PathBuf> {
    ensure!(
        uri.len() <= 4096,
        "Gradle model directory identity exceeds its bound"
    );
    if let Ok(path) = imported_build_identity_path(uri) {
        return Ok(path);
    }
    let path_text = uri
        .strip_prefix("file:/")
        .filter(|path| !path.starts_with('/'))
        .context("Gradle model directory identity is not a supported file URI")?;
    ensure!(
        !uri.chars().any(|character| character.is_control()
            || matches!(
                character,
                ' ' | '\\' | '"' | '<' | '>' | '`' | '{' | '}' | '[' | ']' | '^' | '|'
            ))
            && path_text.split('/').all(|segment| {
                let dots = segment.to_ascii_lowercase().replace("%2e", ".");
                dots != "." && dots != ".."
            }),
        "Gradle model directory identity contains an alias"
    );
    let path = file_uri_path(uri)?;
    let canonical = selected_import_uri(&path, true)?;
    // Gradle's unchanged inner model uses File.toURI(), including raw non-ASCII text.
    ensure!(
        url::Url::parse(uri)?.as_str() == canonical.as_str(),
        "Gradle model directory identity is not canonical"
    );
    Ok(path)
}

fn gradle_path_is_valid(path: &str, project: bool) -> bool {
    path.len() <= 4096
        && path.starts_with(':')
        && (project && path == ":"
            || path[1..].split(':').all(|part| {
                !part.is_empty()
                    && !part
                        .chars()
                        .any(|character| character.is_control() || matches!(character, '/' | '\\'))
            }))
}

fn model_text_is_valid(text: &str) -> bool {
    text.len() <= 4096
        && !text
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
}

fn model_identifier_is_valid(text: &str) -> bool {
    !text.is_empty()
        && model_text_is_valid(text)
        && !text
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
}

fn maven_goal_parameters_are_valid(goal: &MavenGoal) -> bool {
    let name_is_valid = |text: &str| {
        model_text_is_valid(text)
            && text.chars().enumerate().all(|(index, character)| {
                character.is_ascii_alphabetic()
                    || matches!(character, '_' | '$')
                    || (index > 0 && (character.is_ascii_digit() || matches!(character, '.' | '-')))
            })
            && !text.is_empty()
    };
    let mut names = HashSet::new();
    goal.parameters.len() <= 256
        && (goal.parameters_known || goal.parameters.is_empty())
        && goal.parameters.iter().all(|parameter| {
            name_is_valid(&parameter.name)
                && names.insert(&parameter.name)
                && parameter.alias.as_deref().is_none_or(name_is_valid)
                && parameter
                    .declared_type
                    .as_deref()
                    .is_none_or(model_text_is_valid)
                && parameter
                    .description
                    .as_deref()
                    .is_none_or(model_text_is_valid)
        })
}

fn merge_gradle_model(
    projects: &mut Vec<(String, JdtlsProjectSettings)>,
    value: serde_json::Value,
    root: &Path,
) -> Result<()> {
    merge_gradle_model_with_admission(projects, value, root, root, None)
}

fn merge_gradle_model_with_admission(
    projects: &mut Vec<(String, JdtlsProjectSettings)>,
    value: serde_json::Value,
    root: &Path,
    worktree_root: &Path,
    admitted_modules: Option<&BTreeSet<PathBuf>>,
) -> Result<()> {
    ensure!(
        serde_json::to_vec(&value)?.len() <= 8 * 1024 * 1024,
        "Gradle model exceeds the size limit"
    );
    let mut model: GradleModel =
        serde_json::from_value(value).context("Gradle model response is malformed")?;
    ensure!(
        matches!(model.version, 1 | 2)
            && !model.source_sets_known
            && file_uri_path(&model.root_uri)? == root,
        "Gradle model protocol or project root differs"
    );
    check_module_count(model.projects.len())?;
    let mut identities = HashSet::new();
    let mut modules = HashSet::new();
    let mut task_count = 0usize;
    for project in &mut model.projects {
        let build = file_uri_path(&project.build_root_uri)?;
        let module = file_uri_path(&project.module_uri)?;
        if let Some(admitted_modules) = admitted_modules {
            ensure!(
                (build == root || admitted_modules.contains(&build))
                    && gradle_model_identity_path(&project.build_root_uri)? == build
                    && gradle_model_identity_path(&project.module_uri)? == module
                    && admitted_modules.contains(&module)
                    && (project.project_path != ":" || module == build),
                "Gradle model contains a project outside current cached build ownership"
            );
        }
        ensure!(
            build.starts_with(root)
                && module.starts_with(&build)
                && !build
                    .components()
                    .any(|component| component == std::path::Component::ParentDir)
                && !module
                    .components()
                    .any(|component| component == std::path::Component::ParentDir)
                && identities.insert((build.clone(), project.project_path.clone()))
                && modules.insert(module)
                && gradle_path_is_valid(&project.project_path, true)
                && model_text_is_valid(&project.name),
            "Gradle model contains an invalid or duplicate project"
        );
        task_count = task_count
            .checked_add(project.tasks.len())
            .context("Gradle task count exceeded")?;
        ensure!(task_count <= 4096, "Gradle model has too many tasks");
        let mut task_paths = HashSet::new();
        ensure!(
            project.tasks.iter().all(|task| {
                let expected = if project.project_path == ":" {
                    format!(":{}", task.name)
                } else {
                    format!("{}:{}", project.project_path, task.name)
                };
                task.path == expected
                    && gradle_path_is_valid(&task.path, false)
                    && task_paths.insert(&task.path)
                    && !task.options_known
                    && task.options.is_empty()
                    && task.group.as_deref().is_none_or(model_text_is_valid)
                    && task.description.as_deref().is_none_or(model_text_is_valid)
            }),
            "Gradle model contains invalid or duplicate task metadata"
        );
        if model.version == 1 {
            ensure!(
                project.details_json.is_none(),
                "Basic Gradle model contains extended facts"
            );
        } else {
            let json = project
                .details_json
                .take()
                .context("Gradle project metadata is missing")?;
            ensure!(
                json.len() <= 2 * 1024 * 1024,
                "Gradle project metadata exceeds the size limit"
            );
            let details: GradleModelDetails = serde_json::from_str(&json).map_err(|error| {
                anyhow::anyhow!(
                    "Gradle project metadata is malformed ({:?} at line {}, column {}); check provider setup and refresh",
                    error.classify(),
                    error.line(),
                    error.column()
                )
            })?;
            validate_gradle_configurations(details.configurations_known, &details.configurations)?;
            ensure!(
                !details.dependency_graph_known,
                "Gradle resolved dependency graph is not supported by this declaration contract"
            );
            ensure!(
                details.source_sets.len() <= 128
                    && (details.java_source_sets_known || details.source_sets.is_empty())
                    && details.tasks.len() == project.tasks.len(),
                "Gradle project metadata count or source-set availability differs"
            );
            let mut names = HashSet::new();
            let mut root_count = 0usize;
            for source in &details.source_sets {
                let output_uri = match (
                    source.annotation_processor_output_known,
                    &source.annotation_processor_output_uri,
                ) {
                    (None, None) | (Some(false), Some(None)) | (Some(true), Some(None)) => None,
                    (Some(true), Some(Some(uri))) => Some(uri),
                    _ => anyhow::bail!("Gradle compiler-output coverage is inconsistent"),
                };
                root_count = root_count
                    .checked_add(source.roots.len())
                    .and_then(|count| count.checked_add(usize::from(output_uri.is_some())))
                    .context("Gradle source-root count exceeded")?;
                ensure!(
                    root_count <= MAX_JDTLS_SOURCE_ROOTS
                        && !source.generatedness_known
                        && !source.name.is_empty()
                        && model_text_is_valid(&source.name)
                        && !source.name.chars().any(char::is_control)
                        && names.insert(&source.name),
                    "Gradle source-set metadata is invalid or duplicated"
                );
                let mut roots = HashSet::new();
                for uri in &source.roots {
                    let path = file_uri_path(uri)?;
                    ensure!(
                        uri.len() <= 4096
                            && path.starts_with(worktree_root)
                            && !path
                                .components()
                                .any(|part| part == std::path::Component::ParentDir)
                            && roots.insert(path),
                        "Gradle source root is outside the worktree or duplicated"
                    );
                }
                if let Some(uri) = output_uri {
                    let path = file_uri_path(uri)?;
                    ensure!(
                        uri.len() <= 4096
                            && path.starts_with(&build)
                            && !path
                                .components()
                                .any(|part| part == std::path::Component::ParentDir),
                        "Gradle compiler output is outside its build or exceeds the path limit"
                    );
                }
            }
            let mut paths = HashSet::new();
            for task in &details.tasks {
                ensure!(
                    paths.insert(&task.path) && task.options.len() <= 128,
                    "Gradle task option metadata is duplicated or oversized"
                );
                let target = project
                    .tasks
                    .iter_mut()
                    .find(|target| target.path == task.path)
                    .context("Gradle task options reference an undiscovered task")?;
                let mut names = HashSet::new();
                ensure!(
                    task.options.iter().all(|option| {
                        !option.name.is_empty()
                            && names.insert(&option.name)
                            && model_text_is_valid(&option.name)
                            && !option.name.chars().any(char::is_control)
                            && model_text_is_valid(&option.description)
                            && !option.declared_type.is_empty()
                            && model_text_is_valid(&option.declared_type)
                    }),
                    "Gradle task option declaration is invalid or duplicated"
                );
                target.options_known = task.options_complete;
                target.options = task.options.clone();
            }
            project.details = Some(details);
        }
    }
    if let Some(admitted_modules) = admitted_modules {
        ensure!(
            admitted_modules.len() == modules.len()
                && admitted_modules
                    .iter()
                    .all(|module| modules.contains(module)),
            "Gradle model does not cover current cached build ownership"
        );
        ensure!(
            identities.iter().all(|(build, _)| {
                build == root || identities.contains(&(build.clone(), ":".to_owned()))
            }),
            "Gradle included build has no current captured root project"
        );
    }
    let additional = model
        .projects
        .iter()
        .filter(|project| {
            !projects.iter().any(|(uri, _)| {
                file_uri_path(uri).is_ok_and(|existing| {
                    file_uri_path(&project.module_uri).is_ok_and(|module| existing == module)
                })
            })
        })
        .count();
    check_module_count(
        projects
            .len()
            .checked_add(additional)
            .context("Gradle module count exceeded")?,
    )?;
    for project in model.projects {
        let module = file_uri_path(&project.module_uri)?;
        if let Some((_, settings)) = projects
            .iter_mut()
            .find(|(uri, _)| file_uri_path(uri).is_ok_and(|existing| existing == module))
        {
            settings.gradle = Some(project);
        } else {
            projects.push((
                project.module_uri.clone(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::MetadataOnly,
                    gradle: Some(project),
                    ..Default::default()
                },
            ));
        }
    }
    check_module_count(projects.len())?;
    Ok(())
}

fn merge_maven_model(
    projects: &mut Vec<(String, JdtlsProjectSettings)>,
    value: serde_json::Value,
    root: &Path,
) -> Result<()> {
    ensure!(
        serde_json::to_vec(&value)?.len() <= 8 * 1024 * 1024,
        "Maven model exceeds the size limit"
    );
    let model: MavenModel =
        serde_json::from_value(value).context("Maven model response is malformed")?;
    ensure!(
        matches!(model.version, 1..=3)
            && model.cached_only
            && file_uri_path(&model.root_uri)? == root,
        "Maven model protocol or project root differs"
    );
    check_module_count(
        model
            .projects
            .len()
            .checked_add(model.unavailable_pom_uris.len())
            .context("Maven module count exceeded")?,
    )?;
    let confined = |uri: &str| -> Result<PathBuf> {
        let path = file_uri_path(uri)?;
        ensure!(
            uri.len() <= 4096
                && path.starts_with(root)
                && !path
                    .components()
                    .any(|part| part == std::path::Component::ParentDir),
            "Maven model path is outside the worktree"
        );
        Ok(path)
    };
    let identifier = model_identifier_is_valid;
    let mut lifecycles = HashSet::new();
    ensure!(
        model.lifecycles.len() <= 32,
        "Maven lifecycle count exceeded"
    );
    for lifecycle in &model.lifecycles {
        let mut phases = HashSet::new();
        ensure!(
            identifier(&lifecycle.id)
                && lifecycles.insert(&lifecycle.id)
                && !lifecycle.phases.is_empty()
                && lifecycle.phases.len() <= 128
                && lifecycle
                    .phases
                    .iter()
                    .all(|phase| identifier(phase) && phases.insert(phase)),
            "Maven lifecycle contains invalid or duplicate phases"
        );
    }
    let mut poms = HashSet::new();
    for uri in &model.unavailable_pom_uris {
        let path = confined(uri)?;
        ensure!(
            path.file_name().is_some() && poms.insert(path),
            "Maven unavailable POM identity is invalid or duplicated"
        );
    }
    let mut goal_count = 0usize;
    for project in &model.projects {
        let pom = confined(&project.pom_uri)?;
        ensure!(
            pom.file_name().is_some()
                && poms.insert(pom)
                && identifier(&project.group_id)
                && identifier(&project.artifact_id)
                && identifier(&project.version)
                && identifier(&project.packaging)
                && project.module_uris.len() <= 128
                && project.profiles.len() <= 128
                && (model.version >= 2
                    || (!project.inherited_profile_definitions_known
                        && project.profile_definitions.is_none()))
                && project.plugins.len() <= 128
                && project.profiles.keys().all(|profile| identifier(profile)),
            "Maven project metadata is invalid or duplicated"
        );
        if model.version >= 2 {
            let definitions = project
                .profile_definitions
                .as_ref()
                .context("Maven profile definitions are missing")?;
            ensure!(
                definitions.len() <= 128,
                "Maven profile definition count exceeded"
            );
            let mut origins = HashSet::new();
            for definition in definitions {
                ensure!(
                    identifier(&definition.id)
                        && identifier(&definition.source_group_id)
                        && identifier(&definition.source_artifact_id)
                        && identifier(&definition.source_version)
                        && origins.insert((
                            &definition.id,
                            &definition.source_group_id,
                            &definition.source_artifact_id,
                            &definition.source_version,
                            &definition.source_pom_uri
                        )),
                    "Maven profile origin is invalid or duplicated"
                );
                let source = definition
                    .source_pom_uri
                    .as_deref()
                    .map(confined)
                    .transpose()?;
                ensure!(
                    source
                        .as_ref()
                        .is_none_or(|path| path.file_name().is_some()),
                    "Maven profile origin POM is invalid"
                );
                if !definition.inherited {
                    ensure!(
                        source.as_ref() == Some(&confined(&project.pom_uri)?)
                            && definition.source_group_id == project.group_id
                            && definition.source_artifact_id == project.artifact_id
                            && definition.source_version == project.version,
                        "Maven local profile origin differs from its module"
                    );
                } else {
                    ensure!(
                        source.as_ref() != Some(&confined(&project.pom_uri)?),
                        "Maven inherited profile origin is the current module"
                    );
                }
            }
        }
        let mut modules = HashSet::new();
        for uri in &project.module_uris {
            ensure!(
                modules.insert(confined(uri)?),
                "Maven module link is duplicated"
            );
        }
        let mut plugins = HashSet::new();
        for plugin in &project.plugins {
            ensure!(
                identifier(&plugin.group_id)
                    && identifier(&plugin.artifact_id)
                    && plugin.version.as_deref().is_none_or(identifier)
                    && plugin.goal_prefix.as_deref().is_none_or(identifier)
                    && plugins.insert((&plugin.group_id, &plugin.artifact_id))
                    && (plugin.goals_known || plugin.goals.is_empty()),
                "Maven plugin metadata is invalid or duplicated"
            );
            goal_count = goal_count
                .checked_add(plugin.goals.len())
                .context("Maven goal count exceeded")?;
            ensure!(goal_count <= 4096, "Maven goal count exceeded");
            let mut goals = HashSet::new();
            ensure!(
                plugin.goals.iter().all(|goal| identifier(&goal.name)
                    && goals.insert(&goal.name)
                    && goal.phase.as_deref().is_none_or(identifier)
                    && goal.description.as_deref().is_none_or(model_text_is_valid)
                    && maven_goal_parameters_are_valid(goal)
                    && (model.version == 3
                        || (!goal.parameters_known && goal.parameters.is_empty()))),
                "Maven goal metadata is invalid or duplicated"
            );
        }
    }
    let mut reactor_positions = HashMap::new();
    if model.version == 1 {
        ensure!(
            !model.reactor_order_known
                && model.reactor_pom_uris.is_none()
                && model.reactor_order_problem.is_none(),
            "Maven basic protocol cannot claim reactor order"
        );
    } else {
        let order = model
            .reactor_pom_uris
            .as_ref()
            .context("Maven reactor order is missing")?;
        check_module_count(order.len())?;
        if model.reactor_order_known {
            ensure!(
                model.unavailable_pom_uris.is_empty()
                    && model.reactor_order_problem.is_none()
                    && order.len() == model.projects.len(),
                "Maven known order has incomplete models"
            );
            for (position, uri) in order.iter().enumerate() {
                let pom = confined(uri)?;
                ensure!(
                    poms.contains(&pom) && reactor_positions.insert(pom, position).is_none(),
                    "Maven reactor order identity differs or is duplicated"
                );
            }
        } else {
            ensure!(
                order.is_empty()
                    && matches!(
                        model.reactor_order_problem.as_deref(),
                        Some(
                            "missing-cached-model"
                                | "duplicate-coordinates"
                                | "invalid-cached-reactor"
                        )
                    ),
                "Maven unknown reactor order category is invalid"
            );
        }
    }
    let modules = model
        .projects
        .iter()
        .map(|project| {
            let pom = confined(&project.pom_uri)?;
            pom.parent()
                .map(Path::to_path_buf)
                .context("Maven POM has no project folder")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut unique = HashSet::new();
    ensure!(
        modules.iter().all(|module| unique.insert(module)),
        "Maven project folders are duplicated"
    );
    let additional = modules
        .iter()
        .filter(|module| {
            !projects
                .iter()
                .any(|(uri, _)| file_uri_path(uri).is_ok_and(|existing| existing == **module))
        })
        .count();
    check_module_count(
        projects
            .len()
            .checked_add(additional)
            .context("Maven module count exceeded")?,
    )?;
    if model.projects.is_empty() || !model.unavailable_pom_uris.is_empty() {
        for (_, settings) in projects.iter_mut() {
            settings.maven_metadata_incomplete = true;
        }
    }
    for (mut project, module) in model.projects.into_iter().zip(modules) {
        project.lifecycles = model.lifecycles.clone();
        project.reactor_position = reactor_positions.get(&confined(&project.pom_uri)?).copied();
        project.reactor_order_problem = model.reactor_order_problem.clone();
        if let Some((_, settings)) = projects
            .iter_mut()
            .find(|(uri, _)| file_uri_path(uri).is_ok_and(|existing| existing == module))
        {
            settings.maven = Some(project);
        } else {
            let uri = url::Url::from_directory_path(module)
                .map_err(|()| anyhow::anyhow!("Maven module URI is invalid"))?;
            projects.push((
                uri.to_string(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::MetadataOnly,
                    maven: Some(project),
                    ..Default::default()
                },
            ));
        }
    }
    Ok(())
}

pub async fn query_kotlin_workspace(
    server: Arc<LanguageServer>,
    request_timeout: Duration,
    background_executor: BackgroundExecutor,
) -> Result<Vec<u8>> {
    let capabilities = server.capabilities();
    let commands = capabilities
        .execute_command_provider
        .as_ref()
        .map(|provider| provider.commands.as_slice())
        .unwrap_or_default();
    ensure!(
        commands.iter().any(|command| command == "exportWorkspace"),
        "Kotlin LSP does not expose workspace export"
    );
    let directory = background_executor
        .spawn(async { tempfile::tempdir() })
        .await
        .context("create a temporary Kotlin workspace export directory")?;
    let export_path = directory
        .path()
        .to_str()
        .context("Kotlin workspace export path is not UTF-8")?
        .to_string();
    let response = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: "exportWorkspace".to_string(),
                arguments: vec![json!(export_path)],
                ..Default::default()
            },
            request_timeout,
        )
        .await
        .into_response()
        .context("export Kotlin workspace model")?;
    ensure!(
        response.is_none(),
        "Kotlin workspace export returned an unexpected result"
    );
    background_executor
        .spawn(async move {
            let path = directory.path().join("workspace.json");
            let metadata = path
                .symlink_metadata()
                .context("Kotlin workspace export is missing")?;
            ensure!(
                metadata.is_file(),
                "Kotlin workspace export is not a regular file"
            );
            let mut bytes = Vec::new();
            std::fs::File::open(&path)
                .context("open Kotlin workspace export")?
                .take(MAX_KOTLIN_WORKSPACE_BYTES + 1)
                .read_to_end(&mut bytes)
                .context("read Kotlin workspace export")?;
            ensure!(
                bytes.len() as u64 <= MAX_KOTLIN_WORKSPACE_BYTES,
                "Kotlin workspace export exceeds the size limit"
            );
            Ok(bytes)
        })
        .await
}

pub fn projects_from_kotlin_workspace(
    bytes: &[u8],
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
    snapshot: &Snapshot,
    sharing: bool,
) -> Result<(Vec<KotlinWorkspaceProject>, bool)> {
    ensure!(
        bytes.len() as u64 <= MAX_KOTLIN_WORKSPACE_BYTES,
        "Kotlin workspace export exceeds the size limit"
    );
    let workspace: KotlinWorkspaceData =
        serde_json::from_slice(bytes).context("Kotlin LSP returned an invalid workspace export")?;
    let build_system = workspace
        .external_system
        .as_deref()
        .and_then(KotlinBuildSystem::from_provider)
        .context("Kotlin workspace export has no supported Maven or Gradle import")?;
    ensure!(
        workspace.modules.len() <= MAX_JDTLS_MODULES,
        "Kotlin workspace export has too many modules"
    );
    ensure!(
        workspace.libraries.len() <= MAX_KOTLIN_LIBRARIES,
        "Kotlin workspace export has too many libraries"
    );
    let library_names = workspace
        .libraries
        .into_iter()
        .filter_map(|library| {
            kotlin_artifact_coordinate(&library.name)
                .is_some()
                .then_some(library.name)
        })
        .collect::<HashSet<_>>();
    let mut modules = Vec::new();
    let mut visible_names = HashSet::new();
    let mut partial = false;
    for module in workspace.modules {
        let Some(module_root) = module.external_project_path.as_deref().and_then(|path| {
            confined_kotlin_export_path(path, worktree_root, worktree_id, path_style)
        }) else {
            partial = true;
            continue;
        };
        let visible = module_root.path.is_empty()
            || snapshot
                .entry_for_path(module_root.path.as_ref())
                .is_some_and(|entry| !entry.is_private && !entry.is_external && !entry.is_fifo);
        if !visible
            || !valid_kotlin_module_name(&module.name)
            || !visible_names.insert(module.name.clone())
        {
            partial = true;
            continue;
        }
        let mut source_roots = Vec::new();
        for source in module
            .content_roots
            .iter()
            .flat_map(|root| &root.source_roots)
            .take(MAX_JDTLS_SOURCE_ROOTS + 1)
        {
            if source_roots.len() == MAX_JDTLS_SOURCE_ROOTS {
                partial = true;
                break;
            }
            let Some(path) =
                confined_kotlin_export_path(&source.path, worktree_root, worktree_id, path_style)
            else {
                partial = true;
                continue;
            };
            match snapshot.entry_for_path(path.path.as_ref()) {
                Some(entry) if entry.is_private || entry.is_external || entry.is_fifo => {
                    partial = true;
                }
                Some(_) => source_roots.push(KotlinWorkspaceSourcePath {
                    path,
                    navigable: true,
                    generated: source.generated,
                }),
                None => {
                    partial = true;
                    if !sharing {
                        source_roots.push(KotlinWorkspaceSourcePath {
                            path,
                            navigable: false,
                            generated: source.generated,
                        });
                    }
                }
            }
        }
        modules.push((module, module_root, source_roots));
    }
    let projects = modules
        .into_iter()
        .map(|(module, module_root, source_roots)| {
            let mut dependencies = Vec::new();
            let mut unidentified_libraries = 0;
            if module.dependencies.len() > MAX_JDTLS_CLASSPATH_ENTRIES {
                partial = true;
            }
            for dependency in module
                .dependencies
                .into_iter()
                .take(MAX_JDTLS_CLASSPATH_ENTRIES)
            {
                let scope = kotlin_dependency_scope(dependency.scope.as_deref());
                match (dependency.kind.as_str(), dependency.name) {
                    ("module", Some(name)) if visible_names.contains(&name) => {
                        dependencies.push(KotlinWorkspaceDependency::Module { name, scope });
                    }
                    ("module", _) => partial = true,
                    ("library", Some(name)) if library_names.contains(&name) => {
                        if let Some(coordinate) = kotlin_artifact_coordinate(&name) {
                            dependencies.push(KotlinWorkspaceDependency::Artifact {
                                coordinate: coordinate.to_string(),
                                scope,
                            });
                        } else {
                            unidentified_libraries += 1;
                            partial = true;
                        }
                    }
                    ("library", _) => {
                        unidentified_libraries += 1;
                        partial = true;
                    }
                    ("sdk" | "inheritedSdk" | "moduleSource", _) => {}
                    _ => partial = true,
                }
            }
            KotlinWorkspaceProject {
                name: module.name,
                build_system,
                module_root,
                source_roots,
                dependencies,
                unidentified_libraries,
            }
        })
        .collect();
    Ok((projects, partial))
}

fn valid_kotlin_module_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && !name
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
}

fn kotlin_dependency_scope(scope: Option<&str>) -> Option<String> {
    match scope {
        Some("compile" | "test" | "runtime" | "provided") => scope.map(str::to_string),
        _ => None,
    }
}

fn kotlin_artifact_coordinate(name: &str) -> Option<&str> {
    let coordinate = name
        .strip_prefix("Maven: ")
        .or_else(|| name.strip_prefix("Gradle: "))?;
    valid_artifact_coordinate(coordinate).then_some(coordinate)
}

fn has_maven_artifact_attributes(attributes: &serde_json::Value) -> bool {
    ["maven.groupId", "maven.artifactId", "maven.version"]
        .into_iter()
        .any(|key| attributes.get(key).is_some())
}

fn jdtls_maven_artifact(attributes: &serde_json::Value) -> Option<JdtlsLibraryArtifact> {
    let group = attributes.get("maven.groupId")?.as_str()?;
    let artifact = attributes.get("maven.artifactId")?.as_str()?;
    let version = attributes.get("maven.version")?.as_str()?;
    if [group, artifact, version]
        .into_iter()
        .any(|component| component.len() > 128)
    {
        return None;
    }
    let coordinate = format!("{group}:{artifact}:{version}");
    valid_artifact_coordinate(&coordinate).then(|| JdtlsLibraryArtifact {
        coordinate,
        scope: attributes
            .get("maven.scope")
            .and_then(serde_json::Value::as_str)
            .and_then(jdtls_maven_scope)
            .map(str::to_string),
    })
}

fn jdtls_maven_scope(scope: &str) -> Option<&str> {
    matches!(
        scope,
        "compile" | "test" | "runtime" | "provided" | "system"
    )
    .then_some(scope)
}

fn parse_gradle_source_sets(value: &str) -> Option<Vec<String>> {
    if value.is_empty() {
        return Some(Vec::new());
    }
    if value.len() > 256 {
        return None;
    }
    let source_sets = value
        .split(',')
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();
    valid_gradle_source_sets(&source_sets).then_some(source_sets)
}

fn valid_gradle_source_sets(source_sets: &[String]) -> bool {
    !source_sets.is_empty()
        && source_sets.len() <= 16
        && source_sets.iter().all(|name| {
            !name.is_empty()
                && name.len() <= 64
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        })
}

fn valid_artifact_coordinate(coordinate: &str) -> bool {
    let mut components = coordinate.split(':');
    for _ in 0..3 {
        let Some(component) = components.next() else {
            return false;
        };
        if component.is_empty()
            || component.len() > 128
            || !component.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+')
            })
        {
            return false;
        }
    }
    components.next().is_none()
}

fn confined_project_path(
    absolute_path: &Path,
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
) -> Option<ProjectPath> {
    if !path_style.is_absolute(absolute_path.to_str()?) {
        return None;
    }
    let absolute_path = SanitizedPath::new(absolute_path);
    let worktree_root = SanitizedPath::new(worktree_root);
    let relative = path_style.strip_prefix(absolute_path.as_path(), worktree_root.as_path())?;
    Some(ProjectPath {
        worktree_id,
        path: Arc::<RelPath>::from(relative),
    })
}

fn confined_kotlin_export_path(
    path: &str,
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
) -> Option<ProjectPath> {
    // Kotlin LSP writes host-home paths as <HOME>/... when exporting to a temporary directory.
    let absolute_path = if let Some(relative) = path.strip_prefix("<HOME>/") {
        paths::home_dir().join(relative)
    } else {
        PathBuf::from(path)
    };
    confined_project_path(&absolute_path, worktree_root, worktree_id, path_style)
}

fn virtual_project_name(path: &str) -> Option<&str> {
    let name = path.strip_prefix('/')?;
    if !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Some(name)
    } else {
        None
    }
}

fn external_library_file_name<'a>(
    path: &'a str,
    worktree_root: &Path,
    worktree_id: WorktreeId,
    path_style: PathStyle,
) -> Option<&'a str> {
    if !path_style.is_absolute(path)
        || confined_project_path(Path::new(path), worktree_root, worktree_id, path_style).is_some()
    {
        return None;
    }
    let name = path
        .rsplit(|character| path_style.separators_ch().contains(&character))
        .next()?;
    valid_library_file_name(name).then_some(name)
}

fn valid_library_file_name(name: &str) -> bool {
    name.len() <= 128
        && name.ends_with(".jar")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'+'))
}

#[cfg(test)]
mod tests {

    #[test]
    fn maven_pom_mask_pinned_reference_language() {
        let cases: &[(&[&str], &str, bool)] = &[
            (&["*"], "/fixture/module/pom.xml", true),
            (&["*"], "/fixture/pom.xml", true),
            (&[], "/fixture/pom.xml", false),
            (&["pom.xml"], "/fixture/pom.xml", false),
            (&["/fixture/pom.xml"], "/fixture/pom.xml", true),
            (&["/fixture/POM.xml"], "/fixture/pom.xml", false),
            (&["/fixture/pom.xml"], "/fixture/module/pom.xml", false),
            (&["*/pom.xml"], "/fixture/deep/module/pom.xml", true),
            (&["*/pom.xml"], "pom.xml", false),
            (&["**/pom.xml"], "pom.xml", false),
            (
                &["/fixture/m?dule/pom.xml"],
                "/fixture/module/pom.xml",
                true,
            ),
            (
                &["/fixture/m?dule/pom.xml"],
                "/fixture/mdule/pom.xml",
                false,
            ),
            (
                &["/fixture/m?dule/pom.xml"],
                "/fixture/moddule/pom.xml",
                false,
            ),
            (&["*/[ab]/pom.xml"], "/fixture/[ab]/pom.xml", true),
            (&["*/[ab]/pom.xml"], "/fixture/a/pom.xml", false),
            (&["*/{api,app}/pom.xml"], "/fixture/{api,app}/pom.xml", true),
            (&["*/{api,app}/pom.xml"], "/fixture/api/pom.xml", false),
            (&["*/a+b/pom.xml"], "/fixture/a+b/pom.xml", true),
            (&["*/pom.xml"], "/fixture/pomXxml", false),
            (&["*/a(b)/pom.xml"], "/fixture/a(b)/pom.xml", true),
            (&["*/a|b/pom.xml"], "/fixture/a|b/pom.xml", true),
            (&["*/a$/pom.xml"], "/fixture/a$/pom.xml", true),
            (&["*/^a/pom.xml"], "/fixture/^a/pom.xml", true),
            (&["!*/pom.xml"], "/fixture/pom.xml", false),
            (&["*/other.xml", "*/pom.xml"], "/fixture/pom.xml", true),
            (&["/fixture/?/pom.xml"], "/fixture/λ/pom.xml", true),
            (&["*"], "/fixture/a\nb/pom.xml", false),
        ];
        for &(patterns, path, expected) in cases {
            let patterns: Vec<String> = patterns
                .iter()
                .map(|pattern| (*pattern).to_owned())
                .collect();
            let contract = MavenPomMaskContract::new(&patterns).expect("Bounded reference masks");
            assert_eq!(
                contract
                    .expression
                    .as_ref()
                    .is_some_and(|expression| expression.is_match(path)),
                expected,
                "patterns={patterns:?}, path={path:?}"
            );
        }
        for (pattern, path, expected) in [
            (r"\*/pom.xml\", "/fixture/pom.xml", true),
            (r"*/x\\y/pom.xml", r"/fixture/x\y/pom.xml", true),
            (r"*/x\y/pom.xml", r"/fixture/x\y/pom.xml", true),
            (r"*\pom.xml", "C:/fixture/pom.xml", false),
            ("/fixture/?/pom.xml", "/fixture/😀/pom.xml", true),
            ("/fixture/é/pom.xml", "/fixture/e\u{301}/pom.xml", false),
        ] {
            let contract =
                MavenPomMaskContract::new(&[pattern.to_owned()]).expect("Reference boundary mask");
            assert_eq!(
                contract
                    .matches_portable_host_path(
                        path,
                        if path.starts_with("C:/") {
                            PathStyle::Windows
                        } else {
                            PathStyle::Unix
                        }
                    )
                    .expect("Known portable host spelling"),
                expected
            );
        }
    }

    #[test]
    fn maven_pom_mask_contract_is_versioned_atomic_and_bounded() {
        let original = MavenPomMaskContract::new(&["*/module/pom.xml".into()]).expect("Mask");
        let request = original.to_request();
        let roundtrip =
            MavenPomMaskContract::from_request(&request).expect("Exact request contract");
        assert_eq!(roundtrip.to_request(), request);
        for (key, value) in [
            ("version", json!(2)),
            ("version", json!(true)),
            ("syntax", json!("glob")),
            ("pathFormat", json!("native")),
            ("patterns", json!([1])),
            ("patterns", json!(["private-canary\n"])),
        ] {
            let mut invalid = request.clone();
            invalid[key] = value;
            let before = invalid.clone();
            let error = MavenPomMaskContract::from_request(&invalid)
                .expect_err("Malformed or unknown contract")
                .to_string();
            assert!(error.contains("ignored_pom_patterns") && error.contains("refresh"));
            assert!(!error.contains("private-canary"));
            assert_eq!(invalid, before);
        }
        let mut extra = request;
        extra["extra"] = json!(true);
        assert!(MavenPomMaskContract::from_request(&extra).is_err());
        assert!(MavenPomMaskContract::from_request(&Value::Null).is_err());
        let exact_count: Vec<_> = (0..128).map(|index| format!("*/{index}/pom.xml")).collect();
        MavenPomMaskContract::new(&exact_count).expect("Exact mask count");
        let mut oversized_count = exact_count;
        oversized_count.push("*/overflow/pom.xml".into());
        assert!(MavenPomMaskContract::new(&oversized_count).is_err());
        let exact_bytes: Vec<_> = (0..16)
            .map(|index| format!("{index:05}{}", "a".repeat(4091)))
            .collect();
        MavenPomMaskContract::new(&exact_bytes).expect("Exact per-mask and aggregate bytes");
        let mut oversized_bytes = exact_bytes;
        oversized_bytes.push("z".into());
        assert!(MavenPomMaskContract::new(&oversized_bytes).is_err());
        for invalid in [
            vec!["a".repeat(4097)],
            vec!["λ".repeat(2049)],
            vec!["".into()],
            vec!["*".into(), "*".into()],
            vec!["private-canary\u{2028}".into()],
            vec!["private-canary\u{2029}".into()],
            vec!["private-canary\0".into()],
        ] {
            let error = MavenPomMaskContract::new(&invalid)
                .expect_err("Invalid mask")
                .to_string();
            assert!(!error.contains("private-canary"));
        }
    }

    #[test]
    fn maven_pom_mask_paths_require_known_host_spelling() {
        let masks = MavenPomMaskContract::new(&["*/pom.xml".into()]).expect("Mask");
        for path in ["/fixture/pom.xml", "/fixture/λ/pom.xml"] {
            assert!(
                masks
                    .matches_portable_host_path(path, PathStyle::Unix)
                    .expect("POSIX path")
            );
        }
        for path in [
            "C:/fixture/pom.xml",
            "c:/fixture/pom.xml",
            "//server/share/pom.xml",
        ] {
            assert!(
                masks
                    .matches_portable_host_path(path, PathStyle::Windows)
                    .expect("Portable Windows path")
            );
        }
        let exact =
            MavenPomMaskContract::new(&["C:/fixture/pom.xml".into()]).expect("Case-sensitive mask");
        assert!(
            !exact
                .matches_portable_host_path("c:/fixture/pom.xml", PathStyle::Windows)
                .expect("No case folding")
        );
        for style in [PathStyle::Unix, PathStyle::Windows] {
            for path in [
                "pom.xml",
                "file:///fixture/pom.xml",
                "/fixture/../pom.xml",
                "/fixture/./pom.xml",
                "/fixture//pom.xml",
                "/fixture/pom.xml/",
                "/fixture/private-canary\u{2028}/pom.xml",
            ] {
                let error = masks
                    .matches_portable_host_path(path, style)
                    .expect_err("Unknown or aliased host spelling")
                    .to_string();
                assert!(!error.contains("private-canary"));
            }
        }
        for path in [
            r"C:ixture\pom.xml",
            "C:fixture/pom.xml",
            "/fixture/pom.xml",
            "//?/fixture/pom.xml",
            "//server/pom.xml",
            "C:/fixture/pom.xml:stream",
            "C:/fixture./pom.xml",
            "C:/fixture /pom.xml",
        ] {
            assert!(
                masks
                    .matches_portable_host_path(path, PathStyle::Windows)
                    .is_err(),
                "{path}"
            );
        }
        for pattern in [r"\", r"\\"] {
            let empty_translation = MavenPomMaskContract::new(&[pattern.to_owned()])
                .expect("Pinned converter drops empty backslash pieces");
            assert!(
                !empty_translation
                    .matches_portable_host_path("/fixture/pom.xml", PathStyle::Unix)
                    .expect("An empty translated mask must not match every POM")
            );
        }
        let none = MavenPomMaskContract::new(&[]).expect("Cleared masks");
        assert!(
            !none
                .matches_portable_host_path("/fixture/pom.xml", PathStyle::Unix)
                .expect("Known path")
        );
    }

    #[test]
    fn maven_pom_mask_saved_intent_validates_before_application() {
        let settings_for = |patterns: Vec<String>| {
            serde_json::from_value::<settings::JvmBuildSettingsContent>(json!({
                "maven": {"ignored_pom_patterns": patterns},
                "gradle": {"selected_build_files": ["settings.gradle"]},
            }))
            .expect("Saved provider-specific intent")
        };
        let valid_settings = settings_for(vec!["*".into()]);
        let valid = JvmBuildSelectionIntent::from_settings(Some(&valid_settings))
            .expect("Valid default-root mask intent");
        assert!(
            valid
                .maven
                .as_ref()
                .expect("Maven intent")
                .selected_poms
                .is_none()
        );
        assert_eq!(
            valid.gradle.expect("Gradle intent").selected_build_files,
            Some(["settings.gradle".into()].as_slice())
        );
        for patterns in [
            vec![String::new()],
            vec!["*".into(), "*".into()],
            vec!["private-canary\n".into()],
            vec!["private-canary\u{2028}".into()],
            vec!["a".repeat(4097)],
            (0..129).map(|index| format!("*/{index}/pom.xml")).collect(),
        ] {
            let settings = settings_for(patterns);
            let before = serde_json::to_value(&settings).expect("Original saved settings");
            let error = JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect_err("Invalid Maven intent cannot admit partial Gradle intent")
                .to_string();

            assert!(error.contains("ignored_pom_patterns") && error.contains("refresh"));
            assert!(!error.contains("private-canary"));
            assert_eq!(
                serde_json::to_value(&settings).expect("Retained settings"),
                before
            );
        }
    }

    #[test]
    fn doctor_dependency_collection_jvm_maven_mode_requires_visible_effective_proof() {
        let pom = "<project><modelVersion>4.0.0</modelVersion><groupId>dev.made</groupId><artifactId>sample</artifactId><version>1.0</version></project>";
        assert_eq!(
            validate_doctor_maven_mode(pom).expect("Literal selected mode"),
            ("dev.made".into(), "sample".into(), "1.0".into())
        );
        for inserted in [
            "<parent/>",
            "<profiles/>",
            "<build><extensions/></build>",
            "<build><plugins><plugin><configuration><verbose>true</verbose></configuration></plugin></plugins></build>",
            "<groupId>duplicate</groupId>",
        ] {
            assert!(
                validate_doctor_maven_mode(
                    &pom.replace("</project>", &format!("{inserted}</project>"))
                )
                .is_err()
            );
        }
        for invalid in [
            "<!DOCTYPE project [<!ENTITY secret SYSTEM 'file:///private/sentinel'>]><project/>",
            "<project><groupId>${private}</groupId><artifactId>x</artifactId><version>1</version></project>",
            "<project><groupId>PRIVATE_SOURCE_SENTINEL",
            "<not-a-project/>",
        ] {
            let error = validate_doctor_maven_mode(invalid).expect_err("Unsupported configuration");
            assert!(!error.to_string().contains("PRIVATE_SOURCE_SENTINEL"));
        }
    }

    use super::*;

    fn operation_snapshot(gradle: bool) -> JdtlsProjectSnapshot {
        let mut raw = Vec::new();
        if gradle {
            merge_gradle_model(&mut raw, json!({"version":1,"rootUri":"file:///fixture/","sourceSetsKnown":false,
                "projects":[{"buildRootUri":"file:///fixture/included%20build/",
                    "moduleUri":"file:///fixture/included%20build/application/","projectPath":":application",
                    "name":"application","tasks":[
                        {"path":":application:verify","name":"verify","group":null,"description":null,"optionsKnown":false},
                        {"path":":application:日本語 $ZED_FILE $(printf x)","name":"日本語 $ZED_FILE $(printf x)","group":null,"description":null,"optionsKnown":false}]}]}), Path::new("/fixture"))
                .expect("captured Gradle metadata");
        } else {
            merge_maven_model(&mut raw, json!({"version":1,"rootUri":"file:///fixture/","cachedOnly":true,
                "reactorOrderKnown":false,"unavailablePomUris":[],
                "lifecycles":[{"id":"default","phases":["compile","verify"]},{"id":"clean","phases":["clean"]}],
                "projects":[{"pomUri":"file:///fixture/application/pom.xml","groupId":"evidence",
                    "artifactId":"application","version":"1","packaging":"jar","moduleUris":[],"profiles":{},
                    "inheritedProfileDefinitionsKnown":false,"plugins":[
                        {"groupId":"org.apache.maven.plugins","artifactId":"maven-surefire-plugin","version":"3.5.2",
                            "managedOnly":false,"goalsKnown":true,"goals":[{"name":"test","phase":"test","description":null}]}]}]}),Path::new("/fixture"))
                .expect("captured Maven metadata");
        }
        let (uri, settings) = raw.remove(0);
        let (mut project, _) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("admitted provider facts");
        project.membership = JdtlsProjectMembership::ImportedJava;
        project.imported_build_kind = if gradle {
            JdtlsImportedBuildKind::Gradle
        } else {
            JdtlsImportedBuildKind::Maven
        };
        JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: LanguageServerId(7),
            worktree_id: WorktreeId::from_usize(1),
            generation: 5,
            projects: vec![project],
            partial: true,
        }
    }

    #[test]
    fn current_operation_plans_preserve_order_and_build_system_roots() {
        let maven = operation_snapshot(false);
        let selection = vec![
            JvmBuildOperation::MavenPhase {
                lifecycle: "clean".into(),
                phase: "clean".into(),
            },
            JvmBuildOperation::MavenPhase {
                lifecycle: "default".into(),
                phase: "verify".into(),
            },
            JvmBuildOperation::MavenGoal {
                group: "org.apache.maven.plugins".into(),
                artifact: "maven-surefire-plugin".into(),
                version: "3.5.2".into(),
                goal: "test".into(),
            },
        ];
        let receipt = JvmBuildOperationReceipt::new(
            &maven,
            true,
            maven.projects[0].module_root.clone(),
            selection,
            PathStyle::Unix,
        )
        .expect("ordered Maven receipt");
        let plan = receipt
            .plan(&maven, true, PathStyle::Unix)
            .expect("current Maven plan");
        assert_eq!(
            plan.template.args,
            [
                "--offline",
                "--file",
                "application/pom.xml",
                "clean",
                "verify",
                "org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test"
            ]
        );
        assert_eq!(plan.template.command, "$ZED_WORKTREE_ROOT/mvnw");
        assert_eq!(plan.template.cwd.as_deref(), Some("$ZED_WORKTREE_ROOT"));
        assert_eq!(
            plan.build_files,
            [maven.projects[0].maven.as_ref().expect("Maven").pom.clone()]
        );
        assert_eq!(
            receipt
                .plan(&maven, true, PathStyle::Windows)
                .expect("Windows data plan")
                .template
                .args[2],
            "application\\pom.xml"
        );
        let gradle = operation_snapshot(true);
        let paths = gradle.projects[0]
            .gradle
            .as_ref()
            .expect("Gradle")
            .tasks
            .iter()
            .map(|task| JvmBuildOperation::GradleTask {
                path: task.path.clone(),
            })
            .collect();
        let receipt = JvmBuildOperationReceipt::new(
            &gradle,
            true,
            gradle.projects[0].module_root.clone(),
            paths,
            PathStyle::Unix,
        )
        .expect("composite root receipt");
        let plan = receipt
            .plan(&gradle, true, PathStyle::Unix)
            .expect("current Gradle plan");
        assert_eq!(
            plan.template.args,
            [
                "--offline",
                ":application:verify",
                ":application:日本語 $ZED_FILE $(printf x)"
            ]
        );
        assert_eq!(
            plan.template.command,
            "$ZED_WORKTREE_ROOT/included build/gradlew"
        );
        assert_eq!(
            plan.template.cwd.as_deref(),
            Some("$ZED_WORKTREE_ROOT/included build")
        );
        assert!(plan.build_files.iter().all(|path| {
            path.path.parent().expect("manifest parent").as_unix_str() == "included build"
        }));
        assert_eq!(
            receipt
                .plan(&gradle, true, PathStyle::Windows)
                .expect("Windows wrapper data")
                .wrapper
                .path
                .as_unix_str(),
            "included build/gradlew.bat"
        );
    }

    #[test]
    fn current_operation_plans_refuse_stale_foreign_unknown_and_unbounded_selections() {
        let snapshot = operation_snapshot(false);
        let operation = JvmBuildOperation::MavenPhase {
            lifecycle: "default".into(),
            phase: "verify".into(),
        };
        let receipt = JvmBuildOperationReceipt::new(
            &snapshot,
            true,
            snapshot.projects[0].module_root.clone(),
            vec![operation.clone()],
            PathStyle::Unix,
        )
        .expect("known phase");
        assert!(receipt.plan(&snapshot, false, PathStyle::Unix).is_err());
        for field in [0, 1, 2, 3, 4, 5] {
            let mut changed = snapshot.clone();
            match field {
                0 => changed.generation += 1,
                1 => changed.server_id = LanguageServerId(8),
                2 => changed.worktree_id = WorktreeId::from_usize(2),
                3 => changed.projects.clear(),
                4 => changed.projects[0].membership = JdtlsProjectMembership::MetadataOnly,
                _ => changed.projects.push(changed.projects[0].clone()),
            }
            assert!(
                receipt.plan(&changed, true, PathStyle::Unix).is_err(),
                "changed owner {field}"
            );
        }
        let module = snapshot.projects[0].module_root.clone();
        for selection in [
            vec![],
            vec![operation.clone(); 33],
            vec![JvmBuildOperation::MavenPhase {
                lifecycle: "unknown".into(),
                phase: "verify".into(),
            }],
            vec![JvmBuildOperation::MavenPhase {
                lifecycle: "default".into(),
                phase: "--offline".into(),
            }],
            vec![JvmBuildOperation::MavenGoal {
                group: "org.apache.maven.plugins".into(),
                artifact: "maven-surefire-plugin".into(),
                version: "unknown".into(),
                goal: "test".into(),
            }],
            vec![
                operation,
                JvmBuildOperation::GradleTask {
                    path: ":test".into(),
                },
            ],
        ] {
            let error = JvmBuildOperationReceipt::new(
                &snapshot,
                true,
                module.clone(),
                selection,
                PathStyle::Unix,
            )
            .expect_err("unsafe selection")
            .to_string();
            assert!(!error.contains("fixture") && !error.contains("unknown"));
        }
        let goal = JvmBuildOperation::MavenGoal {
            group: "org.apache.maven.plugins".into(),
            artifact: "maven-surefire-plugin".into(),
            version: "3.5.2".into(),
            goal: "test".into(),
        };
        for field in 0..3 {
            let mut changed = snapshot.clone();
            let plugin = &mut changed.projects[0].maven.as_mut().expect("Maven").plugins[0];
            match field {
                0 => plugin.version = None,
                1 => plugin.managed_only = true,
                _ => plugin.goals_known = false,
            }
            assert!(
                JvmBuildOperationReceipt::new(
                    &changed,
                    true,
                    module.clone(),
                    vec![goal.clone()],
                    PathStyle::Unix
                )
                .is_err()
            );
        }
        let mut gradle = operation_snapshot(true);
        let module = gradle.projects[0].module_root.clone();
        for path in [
            ":missing",
            "--offline",
            ":application:bad\n",
            ":application:bad\0",
        ] {
            assert!(
                JvmBuildOperationReceipt::new(
                    &gradle,
                    true,
                    module.clone(),
                    vec![JvmBuildOperation::GradleTask { path: path.into() }],
                    PathStyle::Unix
                )
                .is_err()
            );
        }
        gradle.projects[0]
            .gradle
            .as_mut()
            .expect("Gradle")
            .build_root
            .worktree_id = WorktreeId::from_usize(99);
        assert!(
            JvmBuildOperationReceipt::new(
                &gradle,
                true,
                module,
                vec![JvmBuildOperation::GradleTask {
                    path: ":application:verify".into()
                }],
                PathStyle::Unix
            )
            .is_err()
        );
    }

    #[test]
    fn current_operation_plans_bound_argv_and_reject_template_paths() {
        let mut gradle = operation_snapshot(true);
        let module = gradle.projects[0].module_root.clone();
        let task = &mut gradle.projects[0].gradle.as_mut().expect("Gradle").tasks[0];
        task.path = format!(":{}", "x".repeat(4095));
        let operation = JvmBuildOperation::GradleTask {
            path: task.path.clone(),
        };
        let create = |snapshot: &JdtlsProjectSnapshot, operations| {
            JvmBuildOperationReceipt::new(
                snapshot,
                true,
                module.clone(),
                operations,
                PathStyle::Unix,
            )
        };
        assert!(create(&gradle, vec![operation.clone(); 15]).is_ok());
        assert!(
            create(&gradle, vec![operation; 16]).is_err(),
            "offline flag counts toward 64 KiB"
        );
        gradle.projects[0].gradle.as_mut().expect("Gradle").tasks[0]
            .path
            .push('x');
        let operation = JvmBuildOperation::GradleTask {
            path: gradle.projects[0].gradle.as_ref().expect("Gradle").tasks[0]
                .path
                .clone(),
        };
        assert!(
            create(&gradle, vec![operation]).is_err(),
            "overlong known task rejected"
        );
        for root in ["included $ZED_FILE", "included\nprivate-canary"] {
            let mut changed = operation_snapshot(true);
            let root = RelPath::from_unix_str(root).expect("relative fixture");
            changed.projects[0].module_root.path = root
                .join(RelPath::from_unix_str("application").expect("module"))
                .into();
            changed.projects[0]
                .gradle
                .as_mut()
                .expect("Gradle")
                .build_root
                .path = root.into();
            let error = JvmBuildOperationReceipt::new(
                &changed,
                true,
                changed.projects[0].module_root.clone(),
                vec![JvmBuildOperation::GradleTask {
                    path: ":application:verify".into(),
                }],
                PathStyle::Unix,
            )
            .expect_err("unsafe template path")
            .to_string();
            assert!(!error.contains("canary") && !error.contains("ZED_FILE"));
        }
        let mut maven = operation_snapshot(false);
        maven.projects[0].module_root.path = RelPath::empty_arc();
        maven.projects[0].maven.as_mut().expect("Maven").pom.path =
            RelPath::from_unix_str("pom.xml").expect("root POM").into();
        let receipt = JvmBuildOperationReceipt::new(
            &maven,
            true,
            maven.projects[0].module_root.clone(),
            vec![JvmBuildOperation::MavenPhase {
                lifecycle: "default".into(),
                phase: "verify".into(),
            }],
            PathStyle::Unix,
        )
        .expect("root module");
        assert_eq!(
            receipt
                .plan(&maven, true, PathStyle::Unix)
                .expect("root plan")
                .template
                .args,
            ["--offline", "--file", "pom.xml", "verify"]
        );
    }

    async fn graph_transport_fixture(cx: &mut gpui::TestAppContext) -> Snapshot {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/graph-desktop", json!({"application": {"pom.xml": "<project/>", "build.gradle": "", ".env": "private"}})).await;
        let project = crate::Project::test(fs, [Path::new("/graph-desktop")], cx).await;
        project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("fixture worktree")
                .read(cx)
                .snapshot()
        })
    }

    async fn default_root_fixture(
        root: &str,
        entries: Value,
        cx: &mut gpui::TestAppContext,
    ) -> (Arc<fs::FakeFs>, gpui::Entity<crate::Project>) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree(root, entries).await;
        let project = crate::Project::test(fs.clone(), [Path::new(root)], cx).await;
        (fs, project)
    }

    fn default_root_snapshot(
        project: &gpui::Entity<crate::Project>,
        cx: &gpui::TestAppContext,
    ) -> Snapshot {
        project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("default root worktree")
                .read(cx)
                .snapshot()
        })
    }

    #[gpui::test]
    async fn default_maven_candidates_preserve_intent_and_retire_obsolete_phase_facts(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = Path::new("/maven-default-candidates");
        let (fs, project) = default_root_fixture(
            root.to_str().expect("Fixture root"),
            json!({
                "pom.xml":"<project/>", "child":{"pom.xml":"<project/>"},
                "other":{"pom.xml":"<project/>"}, "excluded":{"pom.xml":"<project/>"},
                ".env":{"pom.xml":"private"}, ".hidden":{"pom.xml":"hidden"},
                ".gitignore":"ignored/\n", "ignored":{"pom.xml":"ignored"},
                "gradle":{"settings.gradle.kts":"", "build.gradle.kts":""}, "alias":{}
            }),
            cx,
        )
        .await;
        fs.create_symlink(&root.join("alias/pom.xml"), root.join("pom.xml"))
            .await
            .expect("Owned in-memory alias");
        cx.run_until_parked();
        let entries = default_root_snapshot(&project, cx);
        let worktree_id = entries.id();
        let inputs =
            default_maven_import_inputs(&entries, worktree_id).expect("Current candidates");
        let paths = |inputs: &[ProjectPath]| {
            inputs
                .iter()
                .map(|input| input.path.as_unix_str().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            paths(&inputs),
            [
                "child/pom.xml",
                "excluded/pom.xml",
                "other/pom.xml",
                "pom.xml"
            ]
        );
        assert!(default_maven_import_inputs(&entries, WorktreeId::from_usize(9999)).is_err());
        let settings = settings::MavenBuildSettingsContent {
            selected_poms: None,
            ignored_projects: Some(vec!["excluded".into()]),
            ignored_pom_patterns: Some(vec!["*/other/pom.xml".into()]),
        };
        let original = serde_json::to_value(&settings).expect("Saved settings");
        let intent = MavenSelectionIntent {
            selected_poms: None,
            ignored_projects: settings.ignored_projects.as_deref().expect("Exclusions"),
            ignored_pom_patterns: settings.ignored_pom_patterns.as_deref().expect("Masks"),
        };
        let module = |directory: &str| {
            let mut project = project_from_settings(
                &format!(
                    "{}/",
                    lsp::Uri::from_file_path(root.join(directory)).expect("Module URI")
                ),
                super::tests::settings(),
                root,
                worktree_id,
                PathStyle::Unix,
            )
            .expect("Provider-owned module")
            .0;
            project.membership = JdtlsProjectMembership::ImportedNonJava;
            project.imported_build_kind = JdtlsImportedBuildKind::Maven;
            project
        };
        let ignored_project = module("excluded");
        let plan = SavedMavenSelection::with_default_snapshot(
            &intent,
            worktree_id,
            std::slice::from_ref(&ignored_project),
            Some(&entries),
        )
        .expect("Default candidate plan")
        .expect("Saved policy");
        assert_eq!(
            paths(&plan.selected_poms),
            ["child/pom.xml", "other/pom.xml", "pom.xml"]
        );
        assert_eq!(plan.unlink_projects, [ignored_project.module_root]);
        assert_eq!(
            serde_json::to_value(&settings).expect("Unchanged settings"),
            original
        );
        assert!(
            SavedMavenSelection::new(&intent, worktree_id, &[]).is_err(),
            "No snapshot cannot guess defaults"
        );
        assert!(
            SavedMavenSelection::with_default_snapshot(
                &MavenSelectionIntent {
                    selected_poms: None,
                    ignored_projects: &[],
                    ignored_pom_patterns: &[],
                },
                worktree_id,
                &[],
                Some(&entries)
            )
            .expect("Ordinary defaults")
            .is_none()
        );
        let empty = SavedMavenSelection::with_default_snapshot(
            &MavenSelectionIntent {
                selected_poms: Some(&[]),
                ignored_projects: &[],
                ignored_pom_patterns: intent.ignored_pom_patterns,
            },
            worktree_id,
            &[],
            Some(&entries),
        )
        .expect("Explicit empty")
        .expect("Explicit intent");
        assert!(empty.selected_poms.is_empty());
        let phase = MavenReactorPhaseFacts {
            imported_poms: vec![root.join("pom.xml"), root.join("child/pom.xml")],
            ignored_poms: vec![root.join("other/pom.xml")],
        };
        let facts = MavenSelectionFacts::from_phase_with_default_inputs(
            phase.clone(),
            root,
            worktree_id,
            &entries,
            Some(&settings),
            Some(&inputs),
        )
        .expect("Facts bound to current candidate intent");
        let imported = [module(""), module("child")];
        let wire = facts
            .to_proto_for_version(8, worktree_id, &entries, &imported)
            .expect("Current wire facts");
        assert_eq!(
            wire,
            MavenSelectionFacts::from_phase(
                phase.clone(),
                root,
                worktree_id,
                &entries,
                Some(&settings),
            )
            .expect("Explicit phase compatibility")
            .to_proto_for_version(8, worktree_id, &entries, &imported)
            .expect("Unchanged wire payload")
        );
        assert!(
            MavenSelectionFacts::from_phase_with_default_inputs(
                phase,
                root,
                worktree_id,
                &entries,
                Some(&settings),
                Some(&inputs[..1]),
            )
            .is_err(),
            "A partial candidate list cannot certify current defaults"
        );
        fs.insert_tree(&root.join("new"), json!({"pom.xml":"<project/>"}))
            .await;
        cx.run_until_parked();
        let changed = default_root_snapshot(&project, cx);
        let message = facts
            .validate(worktree_id, &changed)
            .expect_err("New candidate retires old facts")
            .to_string();
        assert!(message.contains("refresh") && !message.contains("maven-default-candidates"));
        assert!(
            facts
                .clone()
                .admit_current_snapshot(worktree_id, &changed, &imported, Some(&settings))
                .expect("Retain provider data without obsolete classification")
                .is_none()
        );
        assert!(
            facts
                .to_proto_for_version(8, worktree_id, &changed, &imported)
                .is_err()
        );
        fs.rename(
            &root.join("new/pom.xml"),
            &root.join(".env/retired-new-pom.xml"),
            Default::default(),
        )
        .await
        .expect("Move only the fake input into a private directory");
        cx.run_until_parked();
        facts
            .validate(worktree_id, &default_root_snapshot(&project, cx))
            .expect("Original public candidate set recovered");
        fs.remove_file(&root.join("excluded/pom.xml"), Default::default())
            .await
            .expect("Remove only the excluded fake candidate");
        cx.run_until_parked();
        assert!(
            facts
                .validate(worktree_id, &default_root_snapshot(&project, cx))
                .is_err(),
            "Candidate freshness covers inputs absent from phase membership facts"
        );
        fs.insert_tree(&root.join("excluded"), json!({"pom.xml":"<project/>"}))
            .await;
        cx.run_until_parked();
        facts
            .validate(worktree_id, &default_root_snapshot(&project, cx))
            .expect("Recovered default intent");
    }

    #[gpui::test]
    async fn default_maven_candidates_bound_discovery_without_changing_explicit_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let tree = (0..33)
            .map(|index| (format!("module-{index}"), json!({"pom.xml":"<project/>"})))
            .collect::<serde_json::Map<_, _>>();
        let (fs, project) =
            default_root_fixture("/maven-default-budget", Value::Object(tree), cx).await;
        let entries = default_root_snapshot(&project, cx);
        let worktree_id = entries.id();
        let intent = MavenSelectionIntent {
            selected_poms: None,
            ignored_projects: &[],
            ignored_pom_patterns: &["*".into()],
        };
        let error =
            SavedMavenSelection::with_default_snapshot(&intent, worktree_id, &[], Some(&entries))
                .expect_err("Default candidate budget")
                .to_string();
        assert!(
            error.contains("narrow selected_poms")
                && error.contains("refresh")
                && !error.contains("maven-default-budget")
        );
        let selected = ["module-0/pom.xml".into()];
        let explicit = SavedMavenSelection::with_default_snapshot(
            &MavenSelectionIntent {
                selected_poms: Some(&selected),
                ..intent
            },
            worktree_id,
            &[],
            Some(&entries),
        )
        .expect("Explicit choice bypasses default discovery")
        .expect("Explicit plan");
        assert_eq!(explicit.selected_poms.len(), 1);
        assert_eq!(
            explicit.selected_poms[0].path.as_unix_str(),
            "module-0/pom.xml"
        );
        for module in ["module-30", "module-31"] {
            fs.remove_file(
                &Path::new("/maven-default-budget")
                    .join(module)
                    .join("pom.xml"),
                Default::default(),
            )
            .await
            .expect("Remove only fake inputs for the name-bound check");
        }
        fs.insert_tree(
            "/maven-default-budget/literal[private-name-canary]",
            json!({"pom.xml":"<project/>"}),
        )
        .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        let error = default_maven_import_inputs(&current, worktree_id)
            .expect_err("Default candidates reuse portable explicit-path bounds")
            .to_string();
        assert!(
            error.contains("portable")
                && error.contains("refresh")
                && !error.contains("private-name-canary")
        );
    }

    #[gpui::test]
    async fn default_jvm_roots_use_visible_inputs_and_saved_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let (fs, project) = default_root_fixture("/default-jvm", json!({
            "kotlin-maven":{"pom.xml":"<project/>","Main.kt":"class Main"},
            "kotlin-gradle":{"settings.gradle.kts":"rootProject.name = \"kotlin\"", "build.gradle.kts":"", "Main.kt":"class Main"},
            "parent":{"pom.xml":"<project/>","nested":{"pom.xml":"<project/>"}},
            ".env":{"private-root-canary":{"pom.xml":"<project/>"}},
            ".hidden-build":{"pom.xml":"<project/>"},
            ".gitignore":"ignored-root/\n",
            "ignored-root":{"pom.xml":"<project/>"},
            "alias":{}
        }), cx).await;
        fs.create_symlink(
            Path::new("/default-jvm/alias/pom.xml"),
            PathBuf::from("/default-jvm/kotlin-maven/pom.xml"),
        )
        .await
        .expect("fixture alias");
        cx.run_until_parked();
        let snapshot = default_root_snapshot(&project, cx);
        let root_names = |roots: Vec<ProjectPath>| {
            roots
                .into_iter()
                .map(|root| root.path.as_unix_str().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            root_names(
                default_jvm_build_roots(&snapshot, snapshot.id(), None).expect("default roots")
            ),
            ["kotlin-gradle", "kotlin-maven", "parent"]
        );
        let settings = |value| {
            serde_json::from_value::<settings::JvmBuildSettingsContent>(value)
                .expect("selection settings")
        };
        let explicit = settings(
            json!({"maven":{"selected_poms":["parent/nested/pom.xml","kotlin-maven/pom.xml"],"ignored_projects":["kotlin-maven"]},"gradle":{"selected_build_files":[]}}),
        );
        assert_eq!(
            root_names(
                default_jvm_build_roots(&snapshot, snapshot.id(), Some(&explicit))
                    .expect("explicit roots")
            ),
            ["parent/nested"]
        );
        let default_excluded = settings(json!({"maven":{"ignored_projects":["parent"]}}));
        assert_eq!(
            root_names(
                default_jvm_build_roots(&snapshot, snapshot.id(), Some(&default_excluded))
                    .expect("Maven exclusions preserve default input discovery")
            ),
            ["kotlin-gradle", "kotlin-maven"]
        );
        let empty =
            settings(json!({"maven":{"selected_poms":[]},"gradle":{"selected_build_files":[]}}));
        assert!(
            default_jvm_build_roots(&snapshot, snapshot.id(), Some(&empty))
                .expect("empty selection")
                .is_empty()
        );
        let hidden = settings(
            json!({"maven":{"selected_poms":[".hidden-build/pom.xml"]},"gradle":{"selected_build_files":[]}}),
        );
        assert_eq!(
            root_names(
                default_jvm_build_roots(&snapshot, snapshot.id(), Some(&hidden))
                    .expect("explicit visible hidden root")
            ),
            [".hidden-build"]
        );
        for value in [
            json!({"maven":{"selected_poms":["missing/pom.xml"]},"gradle":{"selected_build_files":[]}}),
            json!({"maven":{"selected_poms":[".env/private-root-canary/pom.xml"]},"gradle":{"selected_build_files":[]}}),
            json!({"maven":{"selected_poms":["alias/pom.xml"]},"gradle":{"selected_build_files":[]}}),
            json!({"maven":{"selected_poms":["ignored-root/pom.xml"]},"gradle":{"selected_build_files":[]}}),
            json!({"maven":{"selected_poms":["../private-root-canary/pom.xml"]}}),
            json!({"maven":{"ignored_projects":["../parent"]}}),
        ] {
            let error = default_jvm_build_roots(&snapshot, snapshot.id(), Some(&settings(value)))
                .expect_err("invalid or unavailable selection")
                .to_string();
            assert!(!error.contains("private-root-canary") && !error.contains("/default-jvm"));
        }
        assert!(default_jvm_build_roots(&snapshot, WorktreeId::from_usize(9999), None).is_err());
        let before = default_jvm_build_roots(&snapshot, snapshot.id(), None).expect("stable roots");
        assert_eq!(
            default_jvm_build_roots(&snapshot, snapshot.id(), None).expect("repeat"),
            before
        );
        fs.remove_file(
            Path::new("/default-jvm/kotlin-maven/pom.xml"),
            Default::default(),
        )
        .await
        .expect("remove fixture manifest");
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        assert_eq!(
            root_names(
                default_jvm_build_roots(&current, current.id(), None).expect("removed root")
            ),
            ["kotlin-gradle", "parent"]
        );
        let removed = settings(
            json!({"maven":{"selected_poms":["kotlin-maven/pom.xml"]},"gradle":{"selected_build_files":[]}}),
        );
        assert!(default_jvm_build_roots(&current, current.id(), Some(&removed)).is_err());
    }

    #[gpui::test]
    async fn default_jvm_roots_bound_distinct_candidates_before_return(
        cx: &mut gpui::TestAppContext,
    ) {
        let entries = (0..MAX_JDTLS_MODULES)
            .map(|index| {
                (
                    format!("module_{index:03}"),
                    json!({"pom.xml":"<project/>"}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let (fs, project) =
            default_root_fixture("/default-jvm-roots", Value::Object(entries), cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let roots =
            default_jvm_build_roots(&snapshot, snapshot.id(), None).expect("exact root budget");
        assert_eq!(roots.len(), MAX_JDTLS_MODULES);
        fs.insert_tree(
            "/default-jvm-roots/overflow",
            json!({"pom.xml":"<project/>"}),
        )
        .await;
        cx.run_until_parked();
        let overflow = default_root_snapshot(&project, cx);
        assert!(
            default_jvm_build_roots(&overflow, overflow.id(), None)
                .expect_err("root budget")
                .to_string()
                .contains("admission budget")
        );
        let explicit = serde_json::from_value::<settings::JvmBuildSettingsContent>(json!({"maven":{"selected_poms":["module_000/pom.xml"]},"gradle":{"selected_build_files":[]}})).expect("narrow selection");
        assert_eq!(
            default_jvm_build_roots(&overflow, overflow.id(), Some(&explicit))
                .expect("narrow recovery")
                .len(),
            1
        );
        assert_eq!(
            default_jvm_build_roots(&snapshot, snapshot.id(), None)
                .expect("original immutable snapshot"),
            roots
        );
    }

    #[gpui::test]
    async fn default_jvm_roots_bound_default_scan_and_allow_explicit_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut entries = (0..MAX_DEFAULT_JVM_DISCOVERY_ENTRIES - 2)
            .map(|index| (format!("source_{index}.txt"), Value::String(String::new())))
            .collect::<serde_json::Map<_, _>>();
        entries.insert("pom.xml".into(), Value::String("<project/>".into()));
        let (fs, project) =
            default_root_fixture("/default-jvm-budget", Value::Object(entries), cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        assert_eq!(
            snapshot.visible_entry_count(),
            MAX_DEFAULT_JVM_DISCOVERY_ENTRIES
        );
        assert_eq!(
            default_jvm_build_roots(&snapshot, snapshot.id(), None)
                .expect("exact scan budget")
                .len(),
            1
        );
        assert!(
            GradleDefaultImportRootsQuery::new(&snapshot, snapshot.id())
                .expect("exact entry budget")
                .inputs
                .is_empty()
        );
        fs.insert_file(Path::new("/default-jvm-budget/overflow.txt"), Vec::new())
            .await;
        cx.run_until_parked();
        let overflow = default_root_snapshot(&project, cx);
        assert_eq!(
            overflow.visible_entry_count(),
            MAX_DEFAULT_JVM_DISCOVERY_ENTRIES + 1
        );
        assert!(
            default_jvm_build_roots(&overflow, overflow.id(), None)
                .expect_err("bounded default scan")
                .to_string()
                .contains("entry budget")
        );
        assert!(
            GradleDefaultImportRootsQuery::new(&overflow, overflow.id())
                .expect_err("Gradle entry budget")
                .to_string()
                .contains("entry budget")
        );
        let explicit = serde_json::from_value::<settings::JvmBuildSettingsContent>(
            json!({"maven":{"selected_poms":["pom.xml"]},"gradle":{"selected_build_files":[]}}),
        )
        .expect("explicit selection");
        let recovered = default_jvm_build_roots(&overflow, overflow.id(), Some(&explicit))
            .expect("no default traversal");
        assert_eq!(recovered.len(), 1);
        assert!(recovered.first().expect("current root").path.is_empty());
    }

    fn graph_transport_wire(gradle: bool) -> proto::JdtlsProjectInfo {
        let mut wire = proto::JdtlsProjectInfo {
            module_root: "application".into(),
            membership: proto::JdtlsProjectMembership::ImportedNonJava as i32,
            imported_build_kind: if gradle {
                proto::JdtlsImportedBuildKind::Gradle
            } else {
                proto::JdtlsImportedBuildKind::Maven
            } as i32,
            ..Default::default()
        };
        if gradle {
            let (selection, response) = gradle_graph_fixture();
            let graph = selection
                .validate_response(response)
                .expect("captured Gradle graph");
            wire.gradle = Some(proto::JdtlsGradleProjectInfo {
                build_root: "".into(),
                project_path: ":application".into(),
                configurations_known: true,
                configurations: vec![proto::JdtlsGradleConfigurationInfo {
                    name: "parityGraph".into(),
                    resolvable: true,
                    ..Default::default()
                }],
                dependency_graph_json: Some(
                    serde_json::to_string(&graph).expect("path-free Gradle graph"),
                ),
                ..Default::default()
            });
        } else {
            let (selection, response) = maven_graph_fixture();
            let graph = selection
                .validate_response(response)
                .expect("captured Maven graph");
            wire.maven = Some(proto::JdtlsMavenProjectInfo {
                pom: "application/pom.xml".into(),
                group_id: "dev.made.groovy.fixture.graph".into(),
                artifact_id: "application".into(),
                version: "1".into(),
                packaging: "pom".into(),
                dependency_graph_json: Some(
                    serde_json::to_string(&graph).expect("path-free Maven graph"),
                ),
                ..Default::default()
            });
        }
        wire
    }

    #[gpui::test]
    async fn graph_snapshot_transport_preserves_provider_facts_and_legacy_unknown(
        cx: &mut gpui::TestAppContext,
    ) {
        let snapshot = graph_transport_fixture(cx).await;
        for gradle in [false, true] {
            let wire = graph_transport_wire(gradle);
            let admitted =
                projects_from_proto_for_version(vec![wire.clone()], snapshot.id(), &snapshot, 5)
                    .expect("remote facts");
            let encoded = project_to_proto_for_version(&admitted[0], 5).expect("new peer encode");
            assert_eq!(encoded, wire);
            let bytes = encoded.encode_to_vec();
            let text = String::from_utf8_lossy(&bytes);
            assert!(
                !text.contains("file:")
                    && !text.contains("/groovy-graph")
                    && !text.contains("/graph-desktop")
            );
            let roundtrip = projects_from_proto_for_version(
                vec![proto::JdtlsProjectInfo::decode(bytes.as_slice()).expect("protobuf")],
                snapshot.id(),
                &snapshot,
                5,
            )
            .expect("roundtrip");
            assert_eq!(roundtrip, admitted);
            if gradle {
                let graph = admitted[0]
                    .gradle
                    .as_ref()
                    .expect("Gradle metadata")
                    .dependency_graph
                    .as_ref()
                    .expect("Gradle graph");
                assert_eq!(graph.resolution.components.len(), 8);
                assert_eq!(graph.resolution.edges.len(), 11);
                assert!(
                    !graph.resolution.resolution_complete
                        && graph.resolution.structural_coverage_known
                        && !graph.resolution.origins_known
                        && !graph.resolution.artifacts_known
                );
            } else {
                let graph = admitted[0]
                    .maven
                    .as_ref()
                    .expect("Maven metadata")
                    .dependency_graph
                    .as_ref()
                    .expect("Maven graph");
                assert_eq!(graph.nodes.len(), 15);
                assert_eq!(graph.edges.len(), 14);
                assert!(
                    graph.occurrences_known
                        && !graph.cycle_occurrences_known
                        && !graph.origins_known
                        && !graph.artifacts_known
                );
            }
            for version in 1..=4 {
                assert!(
                    projects_from_proto_for_version(
                        vec![wire.clone()],
                        snapshot.id(),
                        &snapshot,
                        version
                    )
                    .is_err(),
                    "old protocol cannot admit new fields"
                );
                let legacy =
                    project_to_proto_for_version(&admitted[0], version).expect("old peer encode");
                let legacy = projects_from_proto_for_version(
                    vec![legacy],
                    snapshot.id(),
                    &snapshot,
                    version,
                )
                .expect("old peer metadata");
                assert!(
                    legacy[0]
                        .maven
                        .as_ref()
                        .is_none_or(|maven| maven.dependency_graph.is_none())
                );
                assert!(
                    legacy[0]
                        .gradle
                        .as_ref()
                        .is_none_or(|gradle| gradle.dependency_graph.is_none())
                );
            }
            assert!(
                projects_from_proto_for_version(
                    vec![wire],
                    snapshot.id(),
                    &snapshot,
                    JDTLS_SNAPSHOT_PROTOCOL_VERSION + 1,
                )
                .is_err()
            );
        }
    }

    #[gpui::test]
    async fn graph_snapshot_transport_rejects_selection_ownership_privacy_and_private_errors(
        cx: &mut gpui::TestAppContext,
    ) {
        let snapshot = graph_transport_fixture(cx).await;
        for gradle in [false, true] {
            let wire = graph_transport_wire(gradle);
            for mutation in 0..9 {
                let mut invalid = wire.clone();
                match mutation {
                    0 => {
                        invalid.imported_build_kind = proto::JdtlsImportedBuildKind::Unknown as i32
                    }
                    1 => {
                        invalid.membership = proto::JdtlsProjectMembership::MetadataOnly as i32;
                        invalid.imported_build_kind = proto::JdtlsImportedBuildKind::Unknown as i32;
                    }
                    2 => invalid.module_root = "application/.env/hidden".into(),
                    3 => {
                        if gradle {
                            invalid.gradle.as_mut().expect("Gradle").configurations[0].resolvable =
                                false;
                        } else {
                            invalid.maven.as_mut().expect("Maven").version = "different".into();
                        }
                    }
                    4 => {
                        if gradle {
                            invalid.gradle.as_mut().expect("Gradle").project_path = ":other".into();
                        } else {
                            invalid.maven.as_mut().expect("Maven").pom = "other/pom.xml".into();
                        }
                    }
                    _ => {
                        let text = if gradle {
                            invalid
                                .gradle
                                .as_mut()
                                .expect("Gradle")
                                .dependency_graph_json
                                .as_mut()
                                .expect("graph")
                        } else {
                            invalid
                                .maven
                                .as_mut()
                                .expect("Maven")
                                .dependency_graph_json
                                .as_mut()
                                .expect("graph")
                        };
                        let mut graph: Value = serde_json::from_str(text).expect("facts");
                        match mutation {
                            5 => {
                                graph["rootUri"] = json!("file:///private/SECRET");
                            }
                            6 => {
                                if gradle {
                                    graph["resolution"]["edges"][0]["selected"] = json!("SECRET");
                                } else {
                                    graph["edges"][0]["targetNode"] = json!("SECRET");
                                }
                            }
                            7 => {
                                if gradle {
                                    graph["resolution"]["components"][0]["kind"] = json!("SECRET");
                                } else {
                                    graph["components"][0]["id"] = json!("SECRET");
                                }
                            }
                            _ => {
                                graph["unknownSECRET"] = json!("SECRET");
                            }
                        }
                        *text = serde_json::to_string(&graph).expect("mutated facts");
                    }
                }
                let error =
                    projects_from_proto_for_version(vec![invalid], snapshot.id(), &snapshot, 5)
                        .expect_err("unsafe graph denied");
                assert!(
                    !format!("{error:#}").contains("SECRET"),
                    "no private parser detail: {error:#}"
                );
            }
            let mut projects =
                projects_from_proto_for_version(vec![wire], snapshot.id(), &snapshot, 5)
                    .expect("current facts");
            projects[0].module_root.path = RelPath::from_unix_str("application/.env")
                .expect("private module")
                .into();
            let (visible, partial) = filter_provider_projects(projects, false, &snapshot, true);
            assert!(
                visible.is_empty() && partial,
                "sharing removes graph with private owner"
            );
        }
    }

    #[gpui::test]
    async fn graph_snapshot_transport_rejects_budgets_and_retains_partial_facts(
        cx: &mut gpui::TestAppContext,
    ) {
        let snapshot = graph_transport_fixture(cx).await;
        for gradle in [false, true] {
            let wire = graph_transport_wire(gradle);
            let mut oversized = wire.clone();
            let payload = if gradle {
                &mut oversized
                    .gradle
                    .as_mut()
                    .expect("Gradle")
                    .dependency_graph_json
            } else {
                &mut oversized
                    .maven
                    .as_mut()
                    .expect("Maven")
                    .dependency_graph_json
            };
            *payload = Some("x".repeat(8 * 1_048_576));
            assert!(
                projects_from_proto_for_version(vec![oversized], snapshot.id(), &snapshot, 5)
                    .is_err()
            );
            let mut invalid = wire.clone();
            let text = if gradle {
                invalid
                    .gradle
                    .as_mut()
                    .expect("Gradle")
                    .dependency_graph_json
                    .as_mut()
                    .expect("graph")
            } else {
                invalid
                    .maven
                    .as_mut()
                    .expect("Maven")
                    .dependency_graph_json
                    .as_mut()
                    .expect("graph")
            };
            let mut graph: Value = serde_json::from_str(text).expect("facts");
            if gradle {
                graph["resolution"]["components"] =
                    json!(vec![graph["resolution"]["components"][0].clone(); 4097]);
            } else {
                graph["nodes"] = json!(vec![graph["nodes"][0].clone(); 4097]);
            }
            *text = serde_json::to_string(&graph).expect("count excess");
            assert!(
                projects_from_proto_for_version(vec![invalid], snapshot.id(), &snapshot, 5)
                    .is_err()
            );
            let mut partial = wire;
            if !gradle {
                let text = partial
                    .maven
                    .as_mut()
                    .expect("Maven")
                    .dependency_graph_json
                    .as_mut()
                    .expect("graph");
                let mut graph: Value = serde_json::from_str(text).expect("facts");
                graph["collectionComplete"] = json!(false);
                graph["structuralCoverageKnown"] = json!(false);
                graph["problem"] = json!("offline-metadata-unavailable-check-cache-and-refresh");
                *text = serde_json::to_string(&graph).expect("partial facts");
            }
            let admitted =
                projects_from_proto_for_version(vec![partial], snapshot.id(), &snapshot, 5)
                    .expect("current partial data");
            let projected =
                project_to_proto_for_version(&admitted[0], 5).expect("partial encoding");
            assert!(
                projected
                    .maven
                    .as_ref()
                    .is_some_and(|maven| maven.dependency_graph_json.is_some())
                    || projected
                        .gradle
                        .as_ref()
                        .is_some_and(|gradle| gradle.dependency_graph_json.is_some())
            );
        }
    }

    fn gradle_graph_fixture() -> (SelectedGradleDependencyGraph, Value) {
        let root = if cfg!(windows) {
            Path::new("C:/groovy-graph")
        } else {
            Path::new("/groovy-graph")
        };
        let selection =
            SelectedGradleDependencyGraph::new(root, &root.join("application"), "parityGraph")
                .expect("selected Gradle graph");
        let mut response: Value = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/dependency-graph-v1.json"
        ))
        .expect("captured provider graph");
        if cfg!(windows) {
            response["rootUri"] = selection.parameters.arguments[0]["rootUri"].clone();
            response["projects"][0]["buildRootUri"] = response["rootUri"].clone();
            response["projects"][0]["moduleUri"] =
                selection.parameters.arguments[0]["moduleUri"].clone();
        }
        (selection, response)
    }

    fn change_gradle_graph(response: &Value, change: impl FnOnce(&mut Value)) -> Value {
        let mut response = response.clone();
        let mut graph: Value = serde_json::from_str(
            response["projects"][0]["graphJson"]
                .as_str()
                .expect("graph JSON"),
        )
        .expect("graph");
        change(&mut graph);
        response["projects"][0]["graphJson"] =
            json!(serde_json::to_string(&graph).expect("graph JSON"));
        response
    }

    #[test]
    fn gradle_graph_decoder_preserves_configuration_resolution_semantics() {
        let (selection, response) = gradle_graph_fixture();
        let graph = selection
            .validate_response(response)
            .expect("actual pinned graph");
        assert_eq!(graph.project_path, ":application");
        assert_eq!(graph.resolution.configuration, "parityGraph");
        assert!(graph.resolution.structural_coverage_known);
        assert!(
            !graph.resolution.resolution_complete
                && !graph.resolution.origins_known
                && !graph.resolution.artifacts_known
        );
        assert_eq!(graph.resolution.components.len(), 8);
        assert_eq!(graph.resolution.edges.len(), 11);
        let shared = graph
            .resolution
            .components
            .iter()
            .find(|component| component.module.as_deref() == Some("shared"))
            .expect("conflict winner");
        assert_eq!(shared.version.as_deref(), Some("2.0"));
        assert!(shared.conflict_resolution && shared.constrained);
        assert!(
            shared
                .selection_causes
                .contains(&GradleSelectionCause::ConflictResolution)
        );
        assert!(graph.resolution.components.iter().any(
            |component| component.composite_substitution
                && component.build_tree_path.as_deref() == Some(":included")
                && component.project_path.as_deref() == Some(":")
        ));
        assert!(graph.resolution.edges.iter().any(|edge| edge.constraint && matches!(&edge.requested, GradleDependencySelector::Module{preferred_version,rejected_versions,..} if preferred_version=="1.0" && rejected_versions==&["3.0"])));
        let unresolved = graph
            .resolution
            .edges
            .iter()
            .filter(|edge| !edge.resolved)
            .collect::<Vec<_>>();
        assert_eq!(unresolved.len(), 1);
        assert!(unresolved[0].selected.is_none() && unresolved[0].selected_variant.is_none());
        let cycle_a = graph
            .resolution
            .components
            .iter()
            .find(|component| component.module.as_deref() == Some("cycle-a"))
            .expect("cycle a");
        let cycle_b = graph
            .resolution
            .components
            .iter()
            .find(|component| component.module.as_deref() == Some("cycle-b"))
            .expect("cycle b");
        assert!(
            graph
                .resolution
                .edges
                .iter()
                .any(|edge| edge.from == cycle_a.id
                    && edge.selected.as_deref() == Some(cycle_b.id.as_str()))
        );
        assert!(
            graph
                .resolution
                .edges
                .iter()
                .any(|edge| edge.from == cycle_b.id
                    && edge.selected.as_deref() == Some(cycle_a.id.as_str()))
        );
        let (selection, response) = gradle_graph_fixture();
        let changed = change_gradle_graph(&response, |graph| {
            let component = &mut graph["components"][0];
            let old_identity = component["id"].as_str().expect("identity").to_owned();
            component["version"] = json!("");
            let new_identity = serde_json::to_string(&[
                "module",
                component["group"].as_str().expect("group"),
                component["module"].as_str().expect("module"),
                "",
            ])
            .expect("identity");
            component["id"] = json!(new_identity);
            for edge in graph["edges"].as_array_mut().expect("edges") {
                for field in ["from", "selected"] {
                    if edge[field].as_str() == Some(old_identity.as_str()) {
                        edge[field] = json!(new_identity);
                    }
                }
            }
        });
        assert!(
            selection.validate_response(changed).is_ok(),
            "Gradle coordinates do not require a nonempty Maven-style version"
        );
    }

    #[test]
    fn gradle_graph_decoder_rejects_foreign_bindings_and_private_errors() {
        let (selection, response) = gradle_graph_fixture();
        let mut cases = Vec::new();
        for (field, value) in [
            ("version", json!(2)),
            ("rootUri", json!("file:///private-canary/")),
            ("projects", json!([])),
            ("private-field", json!("private-canary")),
        ] {
            let mut changed = response.clone();
            changed[field] = value;
            cases.push(changed);
        }
        for (field, value) in [
            ("moduleUri", json!("file:/groovy-graph/other/")),
            ("buildRootUri", json!("file:/private-canary/")),
            ("projectPath", json!("bad/path")),
            ("name", json!("secret\ncanary")),
            ("graphJson", json!("private-canary")),
        ] {
            let mut changed = response.clone();
            changed["projects"][0][field] = value;
            cases.push(changed);
        }
        for (field, value) in [
            ("configuration", json!("different")),
            ("originsKnown", json!(true)),
            ("artifactsKnown", json!(true)),
            ("structuralCoverageKnown", json!(false)),
            ("resolutionComplete", json!(true)),
            ("rootComponent", json!("private-canary")),
            ("private-field", json!("secret")),
        ] {
            cases.push(change_gradle_graph(&response, |graph| graph[field] = value));
        }
        for changed in cases {
            let error = selection
                .validate_response(changed)
                .expect_err("invalid graph");
            let message = format!("{error:#}");
            assert!(
                message.contains("refresh")
                    && !message.contains("private")
                    && !message.contains("secret")
                    && !message.contains("groovy-graph")
            );
        }
        let root = selection.root;
        for configuration in ["", "configuration\nsecret", "x/y", "runtime:α"] {
            assert!(
                SelectedGradleDependencyGraph::new(&root, &root.join("application"), configuration)
                    .is_err()
            );
        }
        assert!(
            SelectedGradleDependencyGraph::new(&root, &root.join("../escape"), "parityGraph")
                .is_err()
        );
        assert!(
            SelectedGradleDependencyGraph::new(&root, Path::new("relative"), "parityGraph")
                .is_err()
        );
        assert!(gradle_graph_binding_path("file:///groovy-graph/../private-canary/").is_err());
        assert!(gradle_graph_binding_path("file:///groovy-graph/%61pplication/").is_err());
    }

    #[test]
    fn gradle_graph_decoder_rejects_inconsistent_references_and_resolution() {
        let (selection, response) = gradle_graph_fixture();
        let mut cases = Vec::new();
        for (field, value) in [
            ("id", json!("unknown")),
            ("kind", json!("cargo")),
            ("version", json!("different")),
            ("projectPath", json!(":")),
            ("selectionCauses", json!(["private-description"])),
        ] {
            cases.push(change_gradle_graph(&response, |graph| {
                graph["components"][0][field] = value
            }));
        }
        for (field, value) in [
            ("from", json!("unknown")),
            ("selected", json!("unknown")),
            ("selectedVariant", Value::Null),
            ("originKnown", json!(true)),
            ("failureCategory", json!("secret")),
            ("resolved", json!(false)),
        ] {
            cases.push(change_gradle_graph(&response, |graph| {
                graph["edges"][0][field] = value
            }));
        }
        cases.push(change_gradle_graph(&response, |graph| {
            let duplicate = graph["components"][0].clone();
            graph["components"]
                .as_array_mut()
                .expect("components")
                .push(duplicate);
        }));
        cases.push(change_gradle_graph(&response, |graph| {
            let mut orphan = graph["components"][0].clone();
            orphan["module"] = json!("orphan");
            orphan["id"] = json!(r#"["module","evidence","orphan","1.0"]"#);
            graph["components"]
                .as_array_mut()
                .expect("components")
                .push(orphan);
        }));
        cases.push(change_gradle_graph(&response, |graph| {
            graph["components"]
                .as_array_mut()
                .expect("components")
                .retain(|component| {
                    component["kind"] != "module" || component["module"] != "cycle-b"
                });
        }));
        cases.push(change_gradle_graph(&response, |graph| {
            let edge = graph["edges"]
                .as_array_mut()
                .expect("edges")
                .iter_mut()
                .find(|edge| edge["resolved"] == false)
                .expect("unresolved");
            edge["failureCategory"] = json!("private-canary");
        }));
        for changed in cases {
            assert!(selection.validate_response(changed).is_err());
        }
        let complete = change_gradle_graph(&response, |graph| {
            graph["edges"]
                .as_array_mut()
                .expect("edges")
                .retain(|edge| edge["resolved"] == true);
            graph["resolutionComplete"] = json!(true);
        });
        assert!(
            selection
                .validate_response(complete)
                .expect("fully resolved graph")
                .resolution
                .resolution_complete
        );
    }

    #[test]
    fn gradle_graph_decoder_enforces_graph_text_count_and_depth_budgets() {
        let (selection, response) = gradle_graph_fixture();
        for (field, count) in [("components", 4097), ("edges", 16385)] {
            let changed = change_gradle_graph(&response, |graph| {
                let example = graph[field][0].clone();
                graph[field] = json!(vec![example; count]);
            });
            assert!(selection.validate_response(changed).is_err());
        }
        for field in ["configuration", "rootComponent"] {
            assert!(
                selection
                    .validate_response(change_gradle_graph(&response, |graph| graph[field] =
                        json!("x".repeat(4097))))
                    .is_err()
            );
        }
        assert!(
            selection
                .validate_response(change_gradle_graph(&response, |graph| {
                    graph["components"][0]["selectionCauses"] = json!(vec!["REQUESTED"; 129]);
                }))
                .is_err()
        );
        assert!(
            selection
                .validate_response(change_gradle_graph(&response, |graph| {
                    graph["edges"][0]["requested"]["rejectedVersions"] = json!(vec!["1.0"; 129]);
                }))
                .is_err()
        );
        let mut nested = json!(null);
        for _ in 0..33 {
            nested = json!([nested]);
        }
        assert!(validate_gradle_graph_json_budget(&nested).is_err());
        let text = json!(vec!["x".repeat(4096); 257]);
        assert!(validate_gradle_graph_json_budget(&text).is_err());
        let mut changed = response;
        changed["projects"][0]["graphJson"] = json!("x".repeat(2 * 1024 * 1024 + 1));
        assert!(selection.validate_response(changed).is_err());
    }

    #[gpui::test]
    async fn gradle_graph_query_reuses_advertised_lsp_command_and_sanitizes_failures(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for supported in [false, true] {
            let (selection, response) = gradle_graph_fixture();
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let expected = selection.parameters.clone();
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            if supported {
                                "groovy.gradle.dependencyGraph"
                            } else {
                                "java.project.getSettings"
                            }
                            .into(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                let mode = mode.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let mode = mode.load(Ordering::SeqCst);
                    let response = response.clone();
                    async move {
                        match mode {
                            0 => Ok(Some(response)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!("private-canary /secret/path token=value")),
                            _ => Ok(Some(json!({"private-canary": "secret"}))),
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized fake provider");
            if supported {
                selection
                    .clone()
                    .query(server.clone(), Duration::from_secs(10))
                    .await
                    .expect("graph query");
                for failure in 1..=3 {
                    mode.store(failure, Ordering::SeqCst);
                    let message = format!(
                        "{:#}",
                        selection
                            .clone()
                            .query(server.clone(), Duration::from_secs(10))
                            .await
                            .expect_err("graph failure")
                    );
                    assert!(
                        !message.contains("private")
                            && !message.contains("secret")
                            && !message.contains("token")
                    );
                }
                assert_eq!(requests.load(Ordering::SeqCst), 4);
            } else {
                assert!(
                    selection
                        .query(server, Duration::from_secs(10))
                        .await
                        .expect_err("unadvertised")
                        .to_string()
                        .contains("extension setup")
                );
                assert_eq!(requests.load(Ordering::SeqCst), 0);
            }
        }
    }

    fn maven_graph_fixture() -> (SelectedMavenDependencyGraph, Value) {
        let selection = SelectedMavenDependencyGraph::new(
            Path::new("/groovy-graph"),
            Path::new("/groovy-graph/application/pom.xml"),
        )
        .expect("selected graph");
        let response = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_maven/dependency-graph-v2.json"
        ))
        .expect("captured provider graph");
        (selection, response)
    }

    #[test]
    fn maven_graph_decoder_preserves_actual_occurrences_and_cut_cycles() {
        let (selection, response) = maven_graph_fixture();
        let graph = selection
            .validate_response(response)
            .expect("actual pinned graph");
        assert!(graph.collection_complete && graph.structural_coverage_known);
        assert!(!graph.origins_known && !graph.artifacts_known && !graph.cycle_occurrences_known);
        let branches = graph
            .edges
            .iter()
            .filter(|edge| {
                graph.components.iter().any(|component| {
                    component.id == edge.component && component.artifact_id == "branch"
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(branches.len(), 2);
        let omitted = branches
            .iter()
            .find(|edge| edge.omitted)
            .expect("omitted occurrence");
        let selected = branches
            .iter()
            .find(|edge| !edge.omitted)
            .expect("selected occurrence");
        assert_ne!(selected.target_node, omitted.target_node);
        assert_eq!(omitted.selected_node, selected.target_node);
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.from_node == selected.target_node)
        );
        assert!(
            !graph
                .edges
                .iter()
                .any(|edge| edge.from_node == omitted.target_node)
        );
        assert!(!graph.cycles.is_empty());
        assert_eq!(graph.cycles.len(), graph.cycle_count);
    }

    #[test]
    fn maven_graph_decoder_rejects_bad_binding_flags_and_private_errors() {
        let (selection, response) = maven_graph_fixture();
        for (field, value) in [
            ("version", json!(1)),
            ("rootUri", json!("file:///private-canary/")),
            ("pomUri", json!("file:///groovy-graph/other/pom.xml")),
            ("offline", json!(false)),
            ("cachedOnly", json!(false)),
            ("occurrencesKnown", json!(false)),
            ("originsKnown", json!(true)),
            ("artifactsKnown", json!(true)),
            ("cycleOccurrencesKnown", json!(true)),
            ("structuralCoverageKnown", json!(false)),
            ("cycleCount", json!(4097)),
            ("problem", json!("private-canary /secret/path token=value")),
            ("nodes", json!([])),
            ("private-field", json!("private-canary")),
        ] {
            let mut changed = response.clone();
            changed[field] = value;
            let error = selection
                .validate_response(changed)
                .expect_err("invalid graph");
            let message = format!("{error:#}");
            assert!(message.contains("refresh"));
            assert!(
                !message.contains("private")
                    && !message.contains("secret")
                    && !message.contains("groovy-graph")
            );
        }
        for field in [
            "version",
            "nodes",
            "rootNode",
            "rootComponent",
            "rootUri",
            "pomUri",
            "edges",
            "components",
            "offline",
            "cachedOnly",
            "occurrencesKnown",
            "cycles",
        ] {
            let mut changed = response.clone();
            changed.as_object_mut().expect("graph object").remove(field);
            assert!(
                selection.validate_response(changed).is_err(),
                "missing {field}"
            );
        }
    }

    #[test]
    fn maven_graph_decoder_rejects_inconsistent_occurrence_references() {
        let (selection, response) = maven_graph_fixture();
        let mut cases = Vec::new();
        for field in [
            "fromNode",
            "targetNode",
            "selectedNode",
            "from",
            "component",
            "selected",
        ] {
            let mut changed = response.clone();
            changed["edges"][0][field] = json!("private-canary");
            cases.push(changed);
        }
        for (field, value) in [
            ("position", json!(400)),
            ("direct", json!(false)),
            ("managedBits", json!(-1)),
        ] {
            let mut changed = response.clone();
            changed["edges"][0][field] = value;
            cases.push(changed);
        }
        let mut changed = response.clone();
        let duplicate = changed["nodes"][0].clone();
        changed["nodes"]
            .as_array_mut()
            .expect("nodes")
            .push(duplicate);
        cases.push(changed);
        let mut changed = response.clone();
        let duplicate = changed["components"][0].clone();
        changed["components"]
            .as_array_mut()
            .expect("components")
            .push(duplicate);
        cases.push(changed);
        let mut changed = response.clone();
        changed["components"][0]["version"] = json!("different");
        cases.push(changed);
        let mut changed = response.clone();
        let duplicate = changed["edges"][0].clone();
        changed["edges"]
            .as_array_mut()
            .expect("edges")
            .push(duplicate);
        cases.push(changed);
        let mut changed = response.clone();
        changed["cycles"][0]["cyclic"][0]["component"] = json!("dangling");
        cases.push(changed);
        let mut changed = response;
        let root_component = changed["rootComponent"].clone();
        changed["nodes"]
            .as_array_mut()
            .expect("nodes")
            .push(json!({"id": format!("node:{}", "a".repeat(64)), "component": root_component}));
        cases.push(changed);
        for changed in cases {
            assert!(selection.validate_response(changed).is_err());
        }
    }

    #[test]
    fn maven_graph_decoder_retains_partial_coverage_and_enforces_budgets() {
        let (selection, response) = maven_graph_fixture();
        let mut partial = response.clone();
        partial["collectionComplete"] = json!(false);
        partial["structuralCoverageKnown"] = json!(false);
        partial["problem"] = json!("offline-metadata-unavailable-check-cache-and-refresh");
        partial["unavailableDescriptors"] = json!([{"component": partial["rootComponent"], "problem": "offline-metadata-unavailable-check-cache-and-refresh"}]);
        let graph = selection
            .validate_response(partial.clone())
            .expect("explicit partial graph");
        assert!(!graph.collection_complete && !graph.structural_coverage_known);
        partial["unavailableDescriptors"][0]["problem"] = json!("private-provider-exception");
        assert!(selection.validate_response(partial).is_err());
        for field in [
            "nodes",
            "components",
            "cycles",
            "edges",
            "unavailableDescriptors",
        ] {
            let mut oversized = response.clone();
            let example = oversized[field][0].clone();
            let count = if field == "edges" { 16385 } else { 4097 };
            oversized[field] = json!(vec![example; count]);
            assert!(selection.validate_response(oversized).is_err());
        }
        let mut oversized = response.clone();
        oversized["components"][0]["classifier"] = json!("x".repeat(4097));
        assert!(selection.validate_response(oversized).is_err());
        let mut controls = response;
        controls["edges"][0]["versionConstraint"] = json!("private\ncanary");
        assert!(selection.validate_response(controls).is_err());
    }

    #[gpui::test]
    async fn maven_graph_query_reuses_advertised_lsp_command_and_sanitizes_failures(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for supported in [false, true] {
            let (selection, response) = maven_graph_fixture();
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let expected = selection.parameters.clone();
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            if supported {
                                "groovy.maven.dependencyGraph"
                            } else {
                                "java.project.getSettings"
                            }
                            .into(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                let mode = mode.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let mode = mode.load(Ordering::SeqCst);
                    let response = response.clone();
                    async move {
                        match mode {
                            0 => Ok(Some(response)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!("private-canary /secret/path token=value")),
                            _ => Ok(Some(json!({"private-canary": "secret"}))),
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized fake provider");
            if supported {
                selection
                    .clone()
                    .query(server.clone(), Duration::from_secs(10))
                    .await
                    .expect("graph query");
                for failure in 1..=3 {
                    mode.store(failure, Ordering::SeqCst);
                    let message = format!(
                        "{:#}",
                        selection
                            .clone()
                            .query(server.clone(), Duration::from_secs(10))
                            .await
                            .expect_err("graph failure")
                    );
                    assert!(
                        !message.contains("private")
                            && !message.contains("secret")
                            && !message.contains("token")
                    );
                }
                assert_eq!(requests.load(Ordering::SeqCst), 4);
            } else {
                assert!(
                    selection
                        .query(server, Duration::from_secs(10))
                        .await
                        .expect_err("unadvertised")
                        .to_string()
                        .contains("extension setup")
                );
                assert_eq!(requests.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn startup_controls_distinguish_absent_empty_selection_and_cleared_ignores() {
        for (value, maven, gradle) in [
            (json!({}), false, false),
            (json!({"maven": {}}), false, false),
            (json!({"maven": {"ignored_projects": []}}), false, false),
            (json!({"maven": {"ignored_pom_patterns": []}}), false, false),
            (
                json!({"maven": {"ignored_pom_patterns": ["*"]}}),
                true,
                false,
            ),
            (
                json!({"maven": {"ignored_projects": ["library"]}}),
                true,
                false,
            ),
            (json!({"maven": {"selected_poms": []}}), true, false),
            (json!({"gradle": {"selected_build_files": []}}), false, true),
            (json!({"gradle": {"ignored_projects": [""]}}), false, true),
            (
                json!({"maven": {"selected_poms": ["pom.xml"]}, "gradle": {"ignored_projects": ["library"]}}),
                true,
                true,
            ),
        ] {
            let settings = serde_json::from_value::<settings::JvmBuildSettingsContent>(value)
                .expect("fixture settings");
            let controls = JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("valid intent")
                .startup_import_controls();
            assert_eq!(controls, JvmStartupImportControls { maven, gradle });
        }
    }

    #[test]
    fn startup_controls_preserve_default_and_other_provider_options() {
        let original = json!({"bundles": ["fixture-bundle"], "settings": {"java": {"import": {
            "maven": {"enabled": true, "offline": {"enabled": true}},
            "gradle": {"enabled": true, "home": "fixture-home"}
        }}, "java.import.maven.enabled": true, "java_home": "fixture-jdk"}});
        let mut unchanged = Some(original.clone());
        JvmStartupImportControls::default()
            .apply_initialization(&mut unchanged)
            .expect("no intent");
        assert_eq!(unchanged, Some(original.clone()));
        let mut absent = None;
        JvmStartupImportControls::default()
            .apply_initialization(&mut absent)
            .expect("no projection");
        assert_eq!(absent, None);
        let mut controlled = Some(original);
        JvmStartupImportControls {
            maven: true,
            gradle: false,
        }
        .apply_initialization(&mut controlled)
        .expect("adapter defaults suppressed");
        assert_eq!(
            controlled,
            Some(
                json!({"bundles": ["fixture-bundle"], "settings": {"java": {"import": {
            "maven": {"enabled": false, "offline": {"enabled": true}},
            "gradle": {"enabled": true, "home": "fixture-home"}
        }}, "java.import.maven.enabled": false, "java_home": "fixture-jdk"}})
            )
        );
    }

    #[test]
    fn startup_controls_suppress_fresh_metadata_only_for_joint_saved_intent() {
        for (maven, gradle) in [(false, false), (true, false), (false, true), (true, true)] {
            let controls = JvmStartupImportControls { maven, gradle };
            let original = json!({"bundles": ["fixture-bundle"], "settings": {"java": {
                "import": {"maven": {"enabled": true}, "gradle": {"enabled": true}},
                "format": {"enabled": false}
            }}, "fixture": "retained"});
            let mut options = Some(original.clone());
            controls
                .apply_initialization(&mut options)
                .expect("projection");
            let options = options.expect("original options");
            assert_eq!(
                options.get("projectConfigurations"),
                (maven && gradle).then(|| json!([])).as_ref()
            );
            assert_eq!(options["bundles"], original["bundles"]);
            assert_eq!(options["fixture"], original["fixture"]);
            assert_eq!(
                options["settings"]["java"]["format"],
                original["settings"]["java"]["format"]
            );
            assert_eq!(
                options["settings"]["java"]["import"]["maven"]["enabled"],
                !maven
            );
            assert_eq!(
                options["settings"]["java"]["import"]["gradle"]["enabled"],
                !gradle
            );
            if !maven && !gradle {
                assert_eq!(options, original);
            }
        }
        for settings in [
            json!({"maven":{"selected_poms":[]},"gradle":{"selected_build_files":[]}}),
            json!({"maven":{"ignored_pom_patterns":["*/pom.xml"]},"gradle":{"ignored_projects":[""]}}),
        ] {
            let settings = serde_json::from_value::<settings::JvmBuildSettingsContent>(settings)
                .expect("saved intent");
            let controls = JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("admitted settings")
                .startup_import_controls();
            let mut options = None;
            controls
                .apply_initialization(&mut options)
                .expect("joint controls");
            assert_eq!(
                options.expect("joint initialization")["projectConfigurations"],
                json!([])
            );
            assert!(
                controls
                    .validate_overrides(&crate::project_settings::LspSettings {
                        initialization_options: Some(json!({"projectConfigurations":[]})),
                        ..Default::default()
                    })
                    .is_err(),
                "explicit advanced configuration still conflicts rather than being overwritten"
            );
        }
    }

    #[test]
    fn startup_controls_reject_conflicting_or_malformed_advanced_options() {
        let controls = JvmStartupImportControls {
            maven: true,
            gradle: false,
        };
        for options in [
            json!({"projectConfigurations": []}),
            json!({"projectConfigurations": null}),
            json!([]),
            json!({"settings": null}),
            json!({"settings": {"java": true}}),
            json!({"settings": {"java": {"import": {"maven": "private-canary"}}}}),
            json!({"settings": {"java.import.maven.enabled": true}}),
        ] {
            let error = controls
                .validate_overrides(&crate::project_settings::LspSettings {
                    initialization_options: Some(options),
                    ..Default::default()
                })
                .expect_err("conflict cannot fall back to default roots");
            assert!(error.to_string().contains("restart the language server"));
            assert!(!error.to_string().contains("private-canary"));
        }
        for disabled in [json!(false), json!("false"), json!("FALSE")] {
            controls
                .validate_overrides(&crate::project_settings::LspSettings {
                    settings: Some(json!({"java.import.maven.enabled": disabled})),
                    ..Default::default()
                })
                .expect("compatible disabled representation");
        }
        controls.validate_overrides(&crate::project_settings::LspSettings {
            initialization_options: Some(json!({"settings": {"java.import.maven.enabled": false, "java.import.gradle.enabled": true}})),
            settings: Some(json!({"java": {"import": {"maven": {"enabled": false}, "gradle": {"enabled": true}}}})),
            ..Default::default()
        }).expect("compatible explicit settings preserve other provider");
    }

    #[test]
    fn startup_controls_cover_nested_dotted_and_mixed_aliases() {
        let controls = JvmStartupImportControls {
            maven: false,
            gradle: true,
        };
        for configuration in [
            json!({"java.import.gradle.enabled": true}),
            json!({"java": {"import.gradle.enabled": true}}),
            json!({"java.import": {"gradle.enabled": true}}),
            json!({"java.import.gradle": {"enabled": true}}),
            json!({"java": {"import": {"gradle.enabled": true}}}),
            json!({"java": {"import.gradle": {"enabled": true}}}),
            json!({"java.import": {"gradle": {"enabled": true}}}),
            json!({"java": {"import": {"gradle": {"enabled": true}}}}),
        ] {
            controls
                .validate_overrides(&crate::project_settings::LspSettings {
                    settings: Some(configuration.clone()),
                    ..Default::default()
                })
                .expect_err("explicit enabled alias conflicts");
            let mut projected = configuration;
            controls
                .apply_configuration(&mut projected)
                .expect("adapter default projection");
            controls
                .validate_overrides(&crate::project_settings::LspSettings {
                    settings: Some(projected.clone()),
                    ..Default::default()
                })
                .expect("every existing alias is false");
            assert_eq!(
                projected.pointer("/java/import/gradle/enabled"),
                Some(&Value::Bool(false))
            );
        }
    }

    #[test]
    fn saved_maven_masks_preserve_default_roots_and_saved_settings() {
        for value in [
            json!({"maven": {"ignored_pom_patterns": ["*/private-canary/pom.xml"]}}),
            json!({"maven": {"selected_poms": null, "ignored_pom_patterns": ["*"]}}),
        ] {
            let settings = serde_json::from_value::<settings::JvmBuildSettingsContent>(value)
                .expect("Saved mask intent");
            let before = serde_json::to_value(&settings).expect("Original saved settings");
            let intent = JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("Saved masks can retain provider default roots");
            assert!(
                intent
                    .maven
                    .as_ref()
                    .expect("Maven intent")
                    .selected_poms
                    .is_none()
            );
            assert!(
                intent.startup_import_controls().maven,
                "Saved masks suppress automatic Maven import before controlled application"
            );
            assert_eq!(
                serde_json::to_value(&settings).expect("Retained settings"),
                before
            );
        }
        for masks in [None, Some(Vec::new())] {
            let mut settings = serde_json::from_value::<settings::JvmBuildSettingsContent>(json!({
                "maven": {"selected_poms": ["pom.xml"], "ignored_projects": ["library"]},
                "gradle": {"selected_build_files": ["settings.gradle.kts"]}
            }))
            .expect("Existing exact selections");
            settings.maven.as_mut().expect("Maven").ignored_pom_patterns = masks;
            let intent = JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("Absent, cleared and explicit masks preserve provider-specific selection");
            let maven = intent.maven.expect("Maven intent");
            assert_eq!(maven.selected_poms, Some(["pom.xml".into()].as_slice()));
            assert_eq!(maven.ignored_projects, &["library"]);
            assert_eq!(
                intent.gradle.expect("Gradle intent").selected_build_files,
                Some(["settings.gradle.kts".into()].as_slice())
            );
        }
    }

    #[gpui::test]
    async fn maven_selection_wire_preserves_phase_facts_without_host_settings(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut tree = json!({
            "pom.xml":"<project/>", "child":{"pom.xml":"<project/>"},
            ".env":{"pom.xml":"private"}, "not-pom.txt":"text",
        });
        for index in 0..127 {
            tree[format!("module-{index}")] = json!({"pom.xml":"<project/>"});
        }
        let (_fs, project) = default_root_fixture("/selection-wire", tree, cx).await;
        let entries = default_root_snapshot(&project, cx);
        let worktree_id = entries.id();
        let settings = settings::MavenBuildSettingsContent {
            selected_poms: Some(vec!["pom.xml".into()]),
            ignored_pom_patterns: Some(vec!["/selection-wire/pom.xml".into()]),
            ..Default::default()
        };
        let facts = MavenSelectionFacts::from_phase(
            MavenReactorPhaseFacts {
                imported_poms: vec![PathBuf::from("/selection-wire/child/pom.xml")],
                ignored_poms: vec![PathBuf::from("/selection-wire/pom.xml")],
            },
            Path::new("/selection-wire"),
            worktree_id,
            &entries,
            Some(&settings),
        )
        .expect("local phase facts");
        let mut module = project_from_settings(
            "file:///selection-wire/child/",
            super::tests::settings(),
            Path::new("/selection-wire"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("module")
        .0;
        module.membership = JdtlsProjectMembership::ImportedJava;
        module.imported_build_kind = JdtlsImportedBuildKind::Maven;
        let projects = std::slice::from_ref(&module);
        let wire = facts
            .to_proto_for_version(7, worktree_id, &entries, projects)
            .expect("current projection")
            .expect("known facts");
        assert_eq!(wire.imported_poms, ["child/pom.xml"]);
        assert_eq!(wire.ignored_poms, ["pom.xml"]);
        let encoded = proto::JdtlsProjectSnapshotResponse {
            protocol_version: 7,
            maven_selection: Some(wire.clone()),
            ..Default::default()
        }
        .encode_to_vec();
        let fingerprint = facts
            .settings_fingerprint
            .as_ref()
            .expect("host-private digest");
        assert!(
            !encoded
                .windows(fingerprint.len())
                .any(|bytes| bytes == fingerprint)
        );
        let response =
            proto::JdtlsProjectSnapshotResponse::decode(encoded.as_slice()).expect("real protobuf");
        let decoded = MavenSelectionFacts::from_proto_for_version(
            response.maven_selection,
            response.protocol_version,
            worktree_id,
            &entries,
            projects,
        )
        .expect("admitted remote facts")
        .expect("present");
        assert_eq!(decoded.imported_poms, facts.imported_poms);
        assert_eq!(decoded.ignored_poms, facts.ignored_poms);
        assert!(decoded.settings_fingerprint.is_none());
        assert!(
            !decoded
                .matches_settings(Some(&settings))
                .expect("remote facts grant no local provenance")
        );
        for version in 1..=6 {
            assert!(
                facts
                    .to_proto_for_version(version, worktree_id, &entries, projects)
                    .expect("legacy projection")
                    .is_none()
            );
            assert!(
                MavenSelectionFacts::from_proto_for_version(
                    None,
                    version,
                    worktree_id,
                    &entries,
                    projects
                )
                .expect("unknown is not known-empty")
                .is_none()
            );
            assert!(
                MavenSelectionFacts::from_proto_for_version(
                    Some(wire.clone()),
                    version,
                    worktree_id,
                    &entries,
                    projects
                )
                .is_err()
            );
        }
        assert!(
            MavenSelectionFacts::from_proto_for_version(
                None,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION + 1,
                worktree_id,
                &entries,
                projects
            )
            .is_err()
        );
        let empty_import = proto::JdtlsMavenSelectionFacts {
            imported_poms: vec![],
            ignored_poms: vec!["child/pom.xml".into(), "pom.xml".into()],
        };
        let known_empty = MavenSelectionFacts::from_proto_for_version(
            Some(empty_import),
            7,
            worktree_id,
            &entries,
            &[],
        )
        .expect("known empty import")
        .expect("known outcome");
        assert!(known_empty.imported_poms.is_empty());
        assert_eq!(known_empty.ignored_poms.len(), 2);
        let bounded = proto::JdtlsMavenSelectionFacts {
            imported_poms: vec![],
            ignored_poms: std::iter::once("pom.xml".to_string())
                .chain((0..127).map(|index| format!("module-{index}/pom.xml")))
                .collect(),
        };
        assert_eq!(
            MavenSelectionFacts::from_proto_for_version(
                Some(bounded),
                7,
                worktree_id,
                &entries,
                &[],
            )
            .expect("combined128 limit")
            .expect("known facts")
            .ignored_poms
            .len(),
            128
        );
        let invalid = [
            proto::JdtlsMavenSelectionFacts::default(),
            proto::JdtlsMavenSelectionFacts {
                imported_poms: vec!["child/pom.xml".into(); 129],
                ignored_poms: vec![],
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: wire.imported_poms.clone(),
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                imported_poms: vec!["child/pom.xml".into(); 2],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec![".env/pom.xml".into()],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec!["/foreign/pom.xml".into()],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec!["../pom.xml".into()],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec!["missing/pom.xml".into()],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec!["not-pom.txt".into()],
                ..wire.clone()
            },
            proto::JdtlsMavenSelectionFacts {
                ignored_poms: vec!["bad\npom.xml".into()],
                ..wire.clone()
            },
        ];
        for raw in invalid {
            assert!(
                MavenSelectionFacts::from_proto_for_version(
                    Some(raw),
                    7,
                    worktree_id,
                    &entries,
                    projects
                )
                .is_err()
            );
        }
        assert!(
            MavenSelectionFacts::from_proto_for_version(Some(wire), 7, worktree_id, &entries, &[])
                .is_err()
        );
    }

    #[gpui::test]
    async fn maven_selection_snapshot_rejects_stale_restricted_and_contradictory_facts(
        cx: &mut gpui::TestAppContext,
    ) {
        let (_fs, project) = default_root_fixture(
            "/selection-facts",
            json!({
                "pom.xml":"<project/>", "child":{"pom.xml":"<project/>"},
                ".env":{"pom.xml":"private"}, "not-pom.txt":"text",
            }),
            cx,
        )
        .await;
        let entries = default_root_snapshot(&project, cx);
        let worktree_id = entries.id();
        let settings = settings::MavenBuildSettingsContent {
            selected_poms: Some(vec!["pom.xml".into()]),
            ignored_pom_patterns: Some(vec!["/selection-facts/pom.xml".into()]),
            ..Default::default()
        };
        let facts = MavenSelectionFacts::from_phase(
            MavenReactorPhaseFacts {
                imported_poms: vec![PathBuf::from("/selection-facts/child/pom.xml")],
                ignored_poms: vec![PathBuf::from("/selection-facts/pom.xml")],
            },
            Path::new("/selection-facts"),
            worktree_id,
            &entries,
            Some(&settings),
        )
        .expect("Visible phase facts");
        assert!(
            facts
                .matches_settings(Some(&settings))
                .expect("Settings fingerprint")
        );
        assert!(!facts.matches_settings(None).expect("Removed override"));
        let mut cleared = settings;
        cleared.ignored_pom_patterns = Some(vec![]);
        assert!(
            !facts
                .matches_settings(Some(&cleared))
                .expect("Explicit clearing")
        );
        for invalid in [
            MavenSelectionFacts {
                imported_poms: facts.ignored_poms.clone(),
                ..facts.clone()
            },
            MavenSelectionFacts {
                ignored_poms: vec![ProjectPath {
                    worktree_id,
                    path: RelPath::from_unix_str(".env/pom.xml")
                        .expect("Private path")
                        .into(),
                }],
                ..facts.clone()
            },
            MavenSelectionFacts {
                ignored_poms: vec![ProjectPath {
                    worktree_id: WorktreeId::from_usize(9999),
                    ..facts.ignored_poms[0].clone()
                }],
                ..facts.clone()
            },
            MavenSelectionFacts {
                ignored_poms: vec![ProjectPath {
                    worktree_id,
                    path: RelPath::from_unix_str("not-pom.txt")
                        .expect("Non-POM")
                        .into(),
                }],
                ..facts.clone()
            },
        ] {
            let error = invalid
                .validate(worktree_id, &entries)
                .expect_err("Invalid snapshot facts")
                .to_string();
            assert!(
                error.contains("refresh")
                    && !error.contains(".env")
                    && !error.contains("selection-facts")
            );
        }
        let mut module = project_from_settings(
            "file:///selection-facts/child/",
            super::tests::settings(),
            Path::new("/selection-facts"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("Current module")
        .0;
        module.membership = JdtlsProjectMembership::ImportedJava;
        module.imported_build_kind = JdtlsImportedBuildKind::Maven;
        facts
            .validate_membership(std::slice::from_ref(&module))
            .expect("Actual membership");
        assert!(facts.validate_membership(&[]).is_err());
        module.imported_build_kind = JdtlsImportedBuildKind::Gradle;
        assert!(facts.validate_membership(&[module]).is_err());
        let mut state = JdtlsProjectState::default();
        let old = state
            .begin_with_metadata(LanguageServerId(4), worktree_id, Some(facts.clone()))
            .expect("Old generation");
        let pending = state
            .pending_metadata(LanguageServerId(4))
            .expect("Owned pending facts")
            .clone();
        assert_eq!(pending, Some(facts.clone()));
        assert!(state.pending_metadata(LanguageServerId(999)).is_none());
        let current = state
            .begin_with_metadata(LanguageServerId(4), worktree_id, pending)
            .expect("Superseding read preserves the same current phase facts");
        assert!(!state.finish_with_metadata(old, vec![], true, Some(facts.clone())));
        assert!(state.finish_with_metadata(current, vec![], true, Some(facts.clone())));
        assert!(state.pending_metadata(LanguageServerId(4)).is_none());
        assert_eq!(
            state.snapshot().expect("Bound metadata").metadata,
            Some(facts)
        );
        state.mark_stale().expect("Settings change");
        assert!(!state.snapshot_is_current());
        let next = state
            .begin(LanguageServerId(5), worktree_id)
            .expect("New server");
        assert!(state.finish(next, vec![], true));
        assert!(
            state
                .snapshot()
                .expect("No inherited phase receipt")
                .metadata
                .is_none()
        );
        state.invalidate_server(LanguageServerId(5));
        assert!(state.snapshot().is_none());
    }

    #[test]
    fn saved_build_intent_preserves_provider_selection_and_rejects_unsafe_paths() {
        use settings::{
            GradleBuildSettingsContent, JvmBuildSettingsContent, MavenBuildSettingsContent,
        };
        assert_eq!(
            JvmBuildSelectionIntent::from_settings(None).expect("no intent"),
            JvmBuildSelectionIntent::default()
        );
        let mut settings = JvmBuildSettingsContent {
            maven: Some(MavenBuildSettingsContent {
                selected_poms: Some(vec!["pom.xml".into(), "application/pom.xml".into()]),
                ignored_projects: Some(vec![String::new(), "library".into()]),
                ..Default::default()
            }),
            gradle: Some(GradleBuildSettingsContent {
                selected_build_files: Some(vec![
                    "settings.gradle.kts".into(),
                    "included/build.gradle".into(),
                ]),
                ignored_projects: Some(vec!["included/library".into()]),
            }),
        };
        let intent = JvmBuildSelectionIntent::from_settings(Some(&settings)).expect("intent");
        assert_eq!(
            intent
                .maven
                .expect("Maven")
                .selected_poms
                .expect("selection")
                .len(),
            2
        );
        assert_eq!(
            intent.gradle.expect("Gradle").ignored_projects,
            &["included/library"]
        );
        settings.maven.as_mut().expect("Maven").selected_poms = Some(vec![]);
        assert_eq!(
            JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("empty intent")
                .maven
                .expect("Maven")
                .selected_poms,
            Some([].as_slice())
        );
        settings.maven.as_mut().expect("Maven").selected_poms = None;
        assert!(
            JvmBuildSelectionIntent::from_settings(Some(&settings))
                .expect("default intent")
                .maven
                .expect("Maven")
                .selected_poms
                .is_none()
        );
        for provider in ["Maven", "Gradle"] {
            for invalid in [
                "/private-canary",
                "../private-canary",
                "nested/../private-canary",
                "nested//private-canary",
                "./private-canary",
                "file:///private-canary",
                "C:/private-canary",
                "nested\\private-canary",
                "private\ncanary",
                "**/private-canary",
            ] {
                let paths = Some(vec![invalid.to_string()]);
                if provider == "Maven" {
                    settings.maven.as_mut().expect("Maven").ignored_projects = paths;
                } else {
                    settings.maven.as_mut().expect("Maven").ignored_projects = None;
                    settings.gradle.as_mut().expect("Gradle").ignored_projects = paths;
                }
                let error = JvmBuildSelectionIntent::from_settings(Some(&settings))
                    .expect_err("unsafe intent")
                    .to_string();
                assert!(error.contains(provider) && error.contains("jvm_builds"));
                assert!(!error.contains("canary"));
            }
        }
        settings.gradle.as_mut().expect("Gradle").ignored_projects = None;
        for (paths, build_files) in [
            (
                vec!["pom.xml".to_string(); 33],
                Some(["pom.xml"].as_slice()),
            ),
            (vec!["pom.xml".to_string(); 2], Some(["pom.xml"].as_slice())),
            (
                vec!["build.gradle".to_string()],
                Some(["pom.xml"].as_slice()),
            ),
            (vec![String::new()], Some(["pom.xml"].as_slice())),
            (vec!["x".repeat(4097)], None),
        ] {
            assert!(validate_saved_build_paths(Some(&paths), 32, "Maven", build_files).is_err());
        }
        assert!(
            validate_saved_build_paths(Some(&vec![String::new(); 129]), 128, "Gradle", None)
                .is_err()
        );
        assert!(
            validate_saved_build_paths(
                Some(&["pom.xml".into()]),
                128,
                "Gradle",
                Some(&["settings.gradle"])
            )
            .is_err()
        );
        let encoded = serde_json::to_value(&settings).expect("serialize intent");
        let decoded: JvmBuildSettingsContent =
            serde_json::from_value(encoded).expect("restore intent");
        assert_eq!(decoded, settings);
    }

    #[test]
    fn saved_gradle_selection_preserves_defaults_and_ordered_build_roots() {
        let worktree_id = WorktreeId::from_usize(1);
        let intent = GradleSelectionIntent {
            selected_build_files: None,
            ignored_projects: &[],
        };
        assert_eq!(
            SavedGradleSelection::new(&intent, worktree_id, &[]).expect("provider defaults"),
            None
        );
        let ignored = ["private-canary".to_string()];
        let error = SavedGradleSelection::new(
            &GradleSelectionIntent {
                ignored_projects: &ignored,
                ..intent
            },
            worktree_id,
            &[],
        )
        .expect_err("no default-root guess")
        .to_string();
        assert!(error.contains("explicit selected_build_files"));
        assert!(!error.contains("private-canary"));
        let empty = SavedGradleSelection::new(
            &GradleSelectionIntent {
                selected_build_files: Some(&[]),
                ignored_projects: &[],
            },
            worktree_id,
            &[],
        )
        .expect("empty intent")
        .expect("explicit selection");
        assert!(empty.selected_roots.is_empty() && empty.excluded_projects.is_empty());
        let files = [
            "included/build.gradle.kts".into(),
            "settings.gradle.kts".into(),
            "included/settings.gradle.kts".into(),
            "build.gradle".into(),
            "ignored/build.gradle".into(),
            "other/build.gradle".into(),
        ];
        let ignored = ["ignored".into()];
        let selection = SavedGradleSelection::new(
            &GradleSelectionIntent {
                selected_build_files: Some(&files),
                ignored_projects: &ignored,
            },
            worktree_id,
            &[],
        )
        .expect("ordered intent")
        .expect("explicit selection");
        assert_eq!(
            selection
                .selected_roots
                .iter()
                .map(|root| root.path.as_unix_str())
                .collect::<Vec<_>>(),
            ["included", "", "other"]
        );
        for files in [
            vec!["../private-canary/build.gradle".into()],
            vec!["pom.xml".into()],
            vec!["build.gradle".into(); 129],
            vec!["build.gradle".into(), "build.gradle".into()],
        ] {
            let error = SavedGradleSelection::new(
                &GradleSelectionIntent {
                    selected_build_files: Some(&files),
                    ignored_projects: &[],
                },
                worktree_id,
                &[],
            )
            .expect_err("invalid intent")
            .to_string();
            assert!(!error.contains("private-canary"));
        }
    }

    #[test]
    fn saved_gradle_exclusions_require_unique_current_provider_facts() {
        let worktree_id = WorktreeId::from_usize(1);
        let project = |directory: &str, membership, imported_build_kind| {
            project_from_settings(
                &format!("file:///fixture/{directory}/"),
                JdtlsProjectSettings {
                    membership,
                    imported_build_kind,
                    ..Default::default()
                },
                Path::new("/fixture"),
                worktree_id,
                PathStyle::Unix,
            )
            .expect("provider project")
            .0
        };
        let gradle = JdtlsImportedBuildKind::Gradle;
        let imported = JdtlsProjectMembership::ImportedNonJava;
        let mut projects = vec![
            project("ignored", imported, gradle),
            project(
                "ignored/nested",
                JdtlsProjectMembership::ImportedJava,
                gradle,
            ),
            project("ignored/maven", imported, JdtlsImportedBuildKind::Maven),
            project(
                "ignored/metadata",
                JdtlsProjectMembership::MetadataOnly,
                JdtlsImportedBuildKind::Unknown,
            ),
            project("ignored-other", imported, gradle),
            project("required", imported, gradle),
            project("orphan", imported, gradle),
        ];
        let mut foreign = project("ignored/foreign", imported, gradle);
        foreign.module_root.worktree_id = WorktreeId::from_usize(2);
        projects.push(foreign);
        let before = projects.clone();
        let files = ["settings.gradle".into()];
        let ignored = ["ignored".into()];
        let intent = GradleSelectionIntent {
            selected_build_files: Some(&files),
            ignored_projects: &ignored,
        };
        let selection = SavedGradleSelection::new(&intent, worktree_id, &projects)
            .expect("owned intent")
            .expect("explicit selection");
        assert_eq!(
            selection
                .excluded_projects
                .iter()
                .map(|root| root.path.as_unix_str())
                .collect::<Vec<_>>(),
            ["ignored", "ignored/nested"]
        );
        assert_eq!(
            projects, before,
            "planning does not mutate project/source facts"
        );
        for invalid in [
            vec![projects[0].clone(), projects[0].clone()],
            vec![project(
                "ignored",
                imported,
                JdtlsImportedBuildKind::Unknown,
            )],
            vec![projects[0].clone(); MAX_JDTLS_MODULES + 1],
        ] {
            let error = SavedGradleSelection::new(&intent, worktree_id, &invalid)
                .expect_err("incomplete ownership")
                .to_string();
            assert!(error.contains("refresh"));
        }
    }

    #[test]
    fn saved_maven_selection_preserves_defaults_and_requires_explicit_ignore_roots() {
        let worktree_id = WorktreeId::from_usize(1);
        let intent = MavenSelectionIntent {
            ignored_pom_patterns: &[],
            selected_poms: None,
            ignored_projects: &[],
        };
        assert_eq!(
            SavedMavenSelection::new(&intent, worktree_id, &[]).expect("provider defaults"),
            None
        );
        let ignored = ["private-canary".to_string()];
        let error = SavedMavenSelection::new(
            &MavenSelectionIntent {
                ignored_projects: &ignored,
                ..intent
            },
            worktree_id,
            &[],
        )
        .expect_err("no client default-root guess")
        .to_string();
        assert!(error.contains("explicit selected_poms"));
        assert!(!error.contains("private-canary"));
        let empty = SavedMavenSelection::new(
            &MavenSelectionIntent {
                ignored_pom_patterns: &[],
                selected_poms: Some(&[]),
                ignored_projects: &[],
            },
            worktree_id,
            &[],
        )
        .expect("empty selection")
        .expect("explicit policy");
        assert!(empty.selected_poms.is_empty() && empty.unlink_projects.is_empty());
    }

    #[test]
    fn saved_maven_selection_ignores_only_owned_subtrees_and_retains_unmentioned_projects() {
        let worktree_id = WorktreeId::from_usize(1);
        let project = |directory: &str, membership, imported_build_kind| {
            project_from_settings(
                &format!("file:///fixture/{directory}/"),
                JdtlsProjectSettings {
                    membership,
                    imported_build_kind,
                    ..Default::default()
                },
                Path::new("/fixture"),
                worktree_id,
                PathStyle::Unix,
            )
            .expect("provider project")
            .0
        };
        let maven = JdtlsImportedBuildKind::Maven;
        let imported = JdtlsProjectMembership::ImportedNonJava;
        let mut projects = vec![
            project("ignored", imported, maven),
            project(
                "ignored/nested",
                JdtlsProjectMembership::ImportedJava,
                maven,
            ),
            project("ignored/gradle", imported, JdtlsImportedBuildKind::Gradle),
            project(
                "ignored/metadata",
                JdtlsProjectMembership::MetadataOnly,
                JdtlsImportedBuildKind::Unknown,
            ),
            project("ignored-other", imported, maven),
            project("orphan", imported, maven),
            project("required", imported, maven),
        ];
        let mut foreign = project("ignored/foreign", imported, maven);
        foreign.module_root.worktree_id = WorktreeId::from_usize(2);
        projects.push(foreign);
        let roots = [
            "pom.xml".into(),
            "nested/pom.xml".into(),
            "ignored/pom.xml".into(),
        ];
        let directories = ["ignored".into()];
        let intent = MavenSelectionIntent {
            ignored_pom_patterns: &[],
            selected_poms: Some(&roots),
            ignored_projects: &directories,
        };
        let selection = SavedMavenSelection::new(&intent, worktree_id, &projects)
            .expect("saved policy")
            .expect("explicit policy");
        assert_eq!(
            selection
                .selected_poms
                .iter()
                .map(|path| path.path.as_unix_str())
                .collect::<Vec<_>>(),
            ["pom.xml", "nested/pom.xml"]
        );
        assert_eq!(
            selection
                .unlink_projects
                .iter()
                .map(|path| path.path.as_unix_str())
                .collect::<Vec<_>>(),
            ["ignored", "ignored/nested"]
        );
        assert_eq!(selection.ignored_projects[0].path.as_unix_str(), "ignored");
        let whole_root = [String::new()];
        let selection = SavedMavenSelection::new(
            &MavenSelectionIntent {
                ignored_projects: &whole_root,
                ..intent
            },
            worktree_id,
            &projects,
        )
        .expect("root ignore")
        .expect("explicit policy");
        assert!(selection.selected_poms.is_empty());
        assert_eq!(selection.unlink_projects.len(), 5);
        let empty = SavedMavenSelection::new(
            &MavenSelectionIntent {
                ignored_pom_patterns: &[],
                selected_poms: Some(&[]),
                ignored_projects: &directories,
            },
            worktree_id,
            &projects,
        )
        .expect("empty imports with explicit ignore")
        .expect("explicit policy");
        assert!(empty.selected_poms.is_empty());
        assert_eq!(empty.unlink_projects.len(), 2);
        for invalid in [
            vec![projects[0].clone(), projects[0].clone()],
            vec![project(
                "ignored",
                imported,
                JdtlsImportedBuildKind::Unknown,
            )],
            vec![projects[0].clone(); MAX_JDTLS_MODULES + 1],
        ] {
            let error = SavedMavenSelection::new(&intent, worktree_id, &invalid)
                .expect_err("invalid ownership")
                .to_string();
            assert!(error.contains("refresh") && !error.contains("ignored"));
        }
        for roots in [
            vec!["../private-canary/pom.xml".into()],
            vec!["pom.xml".into(); 33],
            vec!["pom.xml".into(), "pom.xml".into()],
        ] {
            let error = SavedMavenSelection::new(
                &MavenSelectionIntent {
                    ignored_pom_patterns: &[],
                    selected_poms: Some(&roots),
                    ignored_projects: &directories,
                },
                worktree_id,
                &projects,
            )
            .expect_err("invalid saved selection")
            .to_string();
            assert!(!error.contains("private-canary"));
        }
    }

    #[test]
    fn selected_unlink_requires_exact_bounded_provider_identities_and_phase_receipt() {
        let root = selected_import_fixture().root;
        let projects = vec![root.join("application"), root.join("library")];
        for provider in [JdtlsUnlinkProvider::Maven, JdtlsUnlinkProvider::Gradle] {
            let selection =
                SelectedProjectUnlink::new(provider, &root, &projects).expect("selection");
            assert_eq!(selection.parameters.command, provider.command());
            assert_eq!(
                selection.parameters.arguments,
                vec![json!({
                    "rootUri": selected_import_uri(&root, true).expect("root URI"),
                    "projectUris": projects.iter().map(|project| selected_import_uri(project, true).expect("module URI")).collect::<Vec<_>>()
                })]
            );
            for invalid in [
                vec![],
                vec![root.join("application"); 129],
                vec![projects[0].clone(); 2],
                vec![PathBuf::from("relative")],
                vec![root.join("../private-canary")],
                vec![root.join("private\ncanary")],
                vec![root.join("x".repeat(4097))],
            ] {
                let error = SelectedProjectUnlink::new(provider, &root, &invalid)
                    .expect_err("invalid identities")
                    .to_string();
                assert!(error.contains(provider.name()));
                assert!(!error.contains("canary"));
            }
            let receipt = json!({"version": 1, "rootUri": selected_import_uri(&root, true).expect("root URI"),
                "unlinkedProjectUris": selection.parameters.arguments[0]["projectUris"], "selectedUnlinkPhaseComplete": true,
                "fullSynchronizationKnown": false});
            selection
                .validate_receipt(receipt.clone())
                .expect("receipt");
            let mut invalid = vec![json!(null), json!([]), json!({})];
            for (field, value) in [
                ("version", json!(2)),
                ("selectedUnlinkPhaseComplete", json!(false)),
                ("fullSynchronizationKnown", json!(true)),
                ("rootUri", json!("file:///private-canary")),
                ("rootUri", json!("x".repeat(4097))),
                ("unlinkedProjectUris", json!([])),
                (
                    "unlinkedProjectUris",
                    json!(["https://private-canary.invalid", "file:///private-canary"]),
                ),
                (
                    "unlinkedProjectUris",
                    json!(["x".repeat(4097), "x".repeat(4097)]),
                ),
                (
                    "unlinkedProjectUris",
                    json!([
                        selection.parameters.arguments[0]["projectUris"][0].clone(),
                        selection.parameters.arguments[0]["projectUris"][0].clone()
                    ]),
                ),
                ("privateCanary", json!(true)),
            ] {
                let mut changed = receipt.clone();
                changed[field] = value;
                invalid.push(changed);
            }
            for value in invalid {
                let error = selection
                    .validate_receipt(value)
                    .expect_err("invalid receipt")
                    .to_string();
                assert!(error.contains(provider.name()) && error.contains("refresh"));
                assert!(!error.contains("canary"));
            }
        }
    }

    fn gradle_root_fixture_root() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from("C:/groovy-root-state")
        } else {
            PathBuf::from("/groovy-root-state")
        }
    }

    fn gradle_root_fixture(label: &str) -> (GradleRootStateQuery, Value) {
        let fixture: Value = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/root-states-v1.json"
        ))
        .expect("captured root states");
        let mut response = fixture["snapshots"]
            .as_array()
            .expect("snapshots")
            .iter()
            .find(|entry| entry["label"] == label)
            .expect("state")["response"]
            .clone();
        let query = GradleRootStateQuery::new(&gradle_root_fixture_root()).expect("root query");
        response["rootUri"] = query.parameters.arguments[0]["rootUri"].clone();
        (query, response)
    }

    #[test]
    fn gradle_root_states_plan_only_needed_retained_phases() {
        for (label, expected) in [
            ("fresh", [None, None]),
            ("open", [None, Some(false)]),
            ("closed", [Some(true), None]),
            ("recoverable", [Some(true), Some(false)]),
        ] {
            let (query, response) = gradle_root_fixture(label);
            let root_uri = query.parameters.arguments[0]["rootUri"].clone();
            assert_eq!(query.parameters.command, "groovy.gradle.rootState");
            assert_eq!(
                query.parameters.arguments,
                vec![json!({"rootUri":root_uri,"offline":true})]
            );
            let state = query.validate_response(response).expect("SDK root state");
            for (ignored, expected_open) in [false, true].into_iter().zip(expected) {
                let phase = state.plan(ignored).expect("pure root plan");
                assert_eq!(phase.as_ref().map(|phase| phase.open), expected_open);
                if let Some(phase) = phase {
                    assert_eq!(phase.root, state.root);
                    assert_eq!(phase.parameters.command, "groovy.gradle.applyRootState");
                    assert_eq!(
                        phase.parameters.arguments,
                        vec![
                            json!({"rootUri":root_uri,"offline":true,"open":expected_open.expect("phase")})
                        ]
                    );
                }
            }
        }
    }

    #[test]
    fn gradle_root_policy_requires_retained_metadata_and_confirmed_state() {
        for label in ["fresh", "open", "closed", "recoverable"] {
            let (query, response) = gradle_root_fixture(label);
            let state = query.validate_response(response).expect("root state");
            let phase = state.retained_ignore_phase();
            assert_eq!(phase.is_ok(), label != "fresh");
            if label == "fresh" {
                let message = phase.as_ref().expect_err("fresh root").to_string();
                assert!(message.contains("not yet supported"));
                assert!(message.contains("Keep the ignore setting"));
                assert!(message.contains("this action did not request an import"));
            }
            if let Ok(phase) = phase {
                assert!(!phase.open, "even already-closed roots run SDK preflight");
                assert_eq!(phase.root, state.root);
            }
            assert_eq!(state.validate_policy(true).is_ok(), label != "open");
            assert_eq!(state.validate_policy(false).is_ok(), label == "open");
        }
    }

    #[test]
    fn gradle_root_policy_closes_all_current_children_without_restoring() {
        for label in ["fresh", "open", "closed", "recoverable"] {
            let (query, response) = gradle_module_policy_fixture(label);
            let states = query.validate_response(response).expect("module states");
            let expected = states
                .modules
                .iter()
                .filter(|module| module.workspace_state != GradleWorkspaceState::Closed)
                .map(|module| module.path.clone())
                .collect::<BTreeSet<_>>();
            let phase = states.plan_closed_build().expect("whole build policy");
            assert_eq!(phase.is_some(), !expected.is_empty());
            if let Some(phase) = phase {
                assert_eq!(
                    phase.closed.iter().cloned().collect::<BTreeSet<_>>(),
                    expected
                );
                assert!(phase.restored.is_empty());
                assert!(!phase.closed.contains(&states.root));
            }
        }
    }

    #[test]
    fn gradle_root_state_reads_reject_unbounded_foreign_and_inconsistent_data() {
        let (query, response) = gradle_root_fixture("fresh");
        let mut invalid = vec![json!(null), json!([]), json!({})];
        for (field, value) in [
            ("version", json!(true)),
            ("version", json!(2)),
            ("version", json!("private-canary")),
            ("rootUri", json!("file:///private-canary/")),
            ("rootUri", json!("file:///groovy-root-state")),
            ("rootUri", json!("file://localhost/groovy-root-state/")),
            ("rootUri", json!("file:///groovy-root-state/?private=value")),
            ("rootUri", json!("file:///groovy-root-state/../private/")),
            ("rootUri", json!("file:///%67roovy-root-state/")),
            ("rootUri", json!("https://private.invalid/")),
            ("rootUri", json!("x".repeat(4097))),
            ("rootUri", json!("file:///private\ncanary/")),
            ("workspaceState", json!("private-canary".repeat(8192))),
            ("workspaceState", json!(true)),
            ("workspaceState", json!("closed")),
            ("workspaceState", json!("open")),
            ("descriptionAvailable", json!("private-canary")),
            ("fullSynchronizationKnown", json!(true)),
            ("fullSynchronizationKnown", json!("private-canary")),
            ("private-field", json!("private-value")),
        ] {
            let mut changed = response.clone();
            changed[field] = value;
            invalid.push(changed);
        }
        for field in [
            "version",
            "rootUri",
            "workspaceState",
            "descriptionAvailable",
            "fullSynchronizationKnown",
        ] {
            let mut changed = response.clone();
            changed
                .as_object_mut()
                .expect("root response")
                .remove(field);
            invalid.push(changed);
        }
        for value in invalid {
            let message = query
                .validate_response(value)
                .expect_err("unsafe root data")
                .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("private")
                    && !message.contains("groovy-root-state")
            );
        }
        for root in [
            PathBuf::from("relative"),
            PathBuf::from("/private\ncanary"),
            PathBuf::from("/root/../private"),
            PathBuf::from(format!("/{}", "x".repeat(4097))),
        ] {
            for error in [
                GradleRootStateQuery::new(&root)
                    .expect_err("bad read root")
                    .to_string(),
                SelectedGradleRootState::new(&root, false)
                    .expect_err("bad phase root")
                    .to_string(),
            ] {
                assert!(error.contains("refresh") && !error.contains("private"));
            }
        }
        let unicode = GradleRootStateQuery::new(&gradle_root_fixture_root().join("日本語 space"))
            .expect("Unicode root");
        let mut encoded = response;
        encoded["rootUri"] = unicode.parameters.arguments[0]["rootUri"].clone();
        unicode
            .validate_response(encoded)
            .expect("canonical Unicode root");
    }

    #[test]
    fn gradle_root_state_receipts_require_exact_root_open_and_phase_echo() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/root-states-v1.json"
        ))
        .expect("captured root phases");
        let phases = fixture["phases"].as_array().expect("phases");
        assert_eq!(phases.len(), 6);
        for entry in phases {
            let mut response = entry["response"].clone();
            let open = response["open"].as_bool().expect("requested state");
            let phase = SelectedGradleRootState::new(&gradle_root_fixture_root(), open)
                .expect("root phase");
            response["rootUri"] = phase.parameters.arguments[0]["rootUri"].clone();
            phase
                .validate_receipt(response.clone())
                .expect("actual SDK phase");
            let mut invalid = vec![json!(null), json!([]), json!({})];
            for (field, value) in [
                ("version", json!(2)),
                ("version", json!(true)),
                ("version", json!("private")),
                ("open", json!(!open)),
                ("open", json!("private")),
                ("rootStatePhaseComplete", json!(false)),
                ("rootStatePhaseComplete", json!("private")),
                ("fullSynchronizationKnown", json!(true)),
                ("rootUri", json!("file:///private-canary/")),
                ("rootUri", json!("file:///groovy-root-state")),
                ("rootUri", json!("file:///groovy-root-state/?private=value")),
                ("rootUri", json!("x".repeat(4097))),
                ("private-field", json!("private-value")),
            ] {
                let mut changed = response.clone();
                changed[field] = value;
                invalid.push(changed);
            }
            for field in [
                "version",
                "rootUri",
                "open",
                "rootStatePhaseComplete",
                "fullSynchronizationKnown",
            ] {
                let mut changed = response.clone();
                changed.as_object_mut().expect("phase").remove(field);
                invalid.push(changed);
            }
            for value in invalid {
                let message = phase
                    .validate_receipt(value)
                    .expect_err("unsafe receipt")
                    .to_string();
                assert!(
                    message.contains("refresh")
                        && !message.contains("private")
                        && !message.contains("groovy-root-state")
                );
            }
        }
    }

    #[gpui::test]
    async fn gradle_root_state_commands_require_capabilities_and_cancel_with_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::AppContext as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        for reading in [true, false] {
            for supported in [false, true] {
                let (query, root_response) = gradle_root_fixture("closed");
                let phase = SelectedGradleRootState::new(&query.root, false).expect("phase");
                let response = if reading {
                    root_response
                } else {
                    json!({"version":1,"rootUri":query.parameters.arguments[0]["rootUri"],"open":false,"rootStatePhaseComplete":true,"fullSynchronizationKnown":false})
                };
                let expected = if reading {
                    query.parameters.clone()
                } else {
                    phase.parameters.clone()
                };
                let mut commands = vec![
                    "groovy.gradle.moduleStates".into(),
                    "groovy.gradle.applyModuleStates".into(),
                    "groovy.gradle.importSelected".into(),
                ];
                if supported {
                    commands.push(expected.command.clone());
                }
                let requests = Arc::new(AtomicUsize::new(0));
                let mode = Arc::new(AtomicUsize::new(0));
                let cancelled = Arc::new(AtomicUsize::new(0));
                let (entered_sender, entered_receiver) = async_channel::unbounded();
                let (server, fake) = lsp::FakeLanguageServer::new(
                    LanguageServerId(0),
                    LanguageServerBinary {
                        path: "unused-fake-jdtls".into(),
                        arguments: vec![],
                        env: None,
                    },
                    "jdtls".into(),
                    lsp::ServerCapabilities {
                        execute_command_provider: Some(lsp::ExecuteCommandOptions {
                            commands,
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    &mut cx.to_async(),
                );
                fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                    let requests = requests.clone();
                    let mode = mode.clone();
                    let cancelled = cancelled.clone();
                    move |parameters, _| {
                        assert_eq!(parameters, expected);
                        requests.fetch_add(1, Ordering::SeqCst);
                        let current = mode.load(Ordering::SeqCst);
                        let response = response.clone();
                        let cancelled = cancelled.clone();
                        let entered_sender = entered_sender.clone();
                        async move {
                            match current {
                                0 => Ok(Some(response)),
                                1 => Ok(None),
                                2 => {
                                    Err(anyhow::anyhow!("private credential at /groovy-root-state"))
                                }
                                3 => Ok(Some(json!({"private-field":"private-value"}))),
                                _ => {
                                    let _cancel_guard = util::defer(move || {
                                        cancelled.fetch_add(1, Ordering::SeqCst);
                                    });
                                    entered_sender.send(()).await.expect("entered observer");
                                    futures::future::pending().await
                                }
                            }
                        }
                    }
                });
                let server = cx
                    .update(|cx| {
                        let parameters = server.default_initialize_params(false, false, cx);
                        server.initialize(
                            parameters,
                            Arc::new(lsp::DidChangeConfigurationParams {
                                settings: json!({}),
                            }),
                            Duration::from_secs(10),
                            cx,
                        )
                    })
                    .await
                    .expect("initialized");
                let perform = |server: Arc<LanguageServer>, timeout: Duration| {
                    let query = query.clone();
                    let phase = phase.clone();
                    async move {
                        if reading {
                            query.query(server, timeout).await.map(|_| ())
                        } else {
                            phase.execute_phase(server, timeout).await
                        }
                    }
                };
                if !supported {
                    let message = perform(server, Duration::from_secs(10))
                        .await
                        .expect_err("no root fallback")
                        .to_string();
                    assert!(message.contains("extension setup"));
                    assert_eq!(requests.load(Ordering::SeqCst), 0);
                    continue;
                }
                perform(server.clone(), Duration::from_secs(10))
                    .await
                    .expect("root response");
                for value in 1..=4 {
                    mode.store(value, Ordering::SeqCst);
                    let message = perform(server.clone(), Duration::from_secs(1))
                        .await
                        .expect_err("bad or timed out root response")
                        .to_string();
                    assert!(
                        message.contains("refresh")
                            && !message.contains("private")
                            && !message.contains("groovy-root-state")
                    );
                }
                entered_receiver
                    .recv()
                    .await
                    .expect("timeout handler entered");
                cx.run_until_parked();
                assert_eq!(cancelled.load(Ordering::SeqCst), 1);
                let task = cx.background_spawn(perform(server.clone(), Duration::MAX));
                entered_receiver.recv().await.expect("drop handler entered");
                drop(task);
                cx.run_until_parked();
                assert_eq!(cancelled.load(Ordering::SeqCst), 2);
                mode.store(0, Ordering::SeqCst);
                perform(server, Duration::from_secs(10))
                    .await
                    .expect("root recovery");
                assert_eq!(requests.load(Ordering::SeqCst), 7);
            }
        }
    }

    fn gradle_module_policy_fixture(label: &str) -> (GradleModuleStateQuery, serde_json::Value) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/module-state-snapshots-v1.json"
        ))
        .expect("captured snapshots");
        assert_eq!(fixture["providerVersion"], "0.35.0");
        let root = Path::new(if cfg!(windows) {
            "C:\\groovy-module-state"
        } else {
            "/groovy-module-state"
        });
        let response = &fixture["snapshots"]
            .as_array()
            .expect("snapshots")
            .iter()
            .find(|item| item["label"] == label)
            .expect("snapshot label")["response"];
        let response = serde_json::from_str(&response.to_string().replace(
            "file:///groovy-module-state/",
            selected_import_uri(root, true).expect("root URI").as_str(),
        ))
        .expect("host paths");
        (GradleModuleStateQuery::new(root).expect("query"), response)
    }

    #[test]
    fn gradle_module_policy_admits_captured_states_and_derives_only_needed_deltas() {
        for (label, ignored, closed, restored) in [
            ("fresh", vec![], vec![], vec![]),
            ("fresh", vec!["library"], vec!["library"], vec![]),
            ("fresh", vec!["future-module"], vec![], vec![]),
            ("closed", vec![], vec![], vec!["included", "library"]),
            ("closed", vec!["library"], vec![], vec!["included"]),
            ("closed", vec!["library", "included"], vec![], vec![]),
            ("open", vec![], vec![], vec!["included"]),
            ("open", vec!["library"], vec!["library"], vec!["included"]),
            ("open", vec!["included"], vec![], vec![]),
            ("recoverable", vec![], vec![], vec!["included", "library"]),
            (
                "recoverable",
                vec!["library"],
                vec!["library"],
                vec!["included"],
            ),
            (
                "recoverable",
                vec!["library", "included"],
                vec!["included", "library"],
                vec![],
            ),
        ] {
            let (query, response) = gradle_module_policy_fixture(label);
            let states = query
                .validate_response(response)
                .expect("captured actual SDK states");
            let paths = |names: Vec<&str>| {
                names
                    .into_iter()
                    .map(|name| states.root.join(name))
                    .collect::<Vec<_>>()
            };
            let phase = states.plan(&paths(ignored)).expect("policy phase");
            let closed = paths(closed).into_iter().collect::<BTreeSet<_>>();
            let restored = paths(restored).into_iter().collect::<BTreeSet<_>>();
            if closed.is_empty() && restored.is_empty() {
                assert!(phase.is_none(), "{label}");
            } else {
                let phase = phase.expect("mutation needed");
                assert_eq!(phase.closed, closed, "{label}");
                assert_eq!(phase.restored, restored, "{label}");
            }
        }
        let (query, mut response) = gradle_module_policy_fixture("fresh");
        let uri =
            |path: &str| selected_import_uri(&query.root.join(path), true).expect("module URI");
        response["modules"]=json!(["library","library/nested","library-sibling"].map(|path| json!({"moduleUri":uri(path),"buildRootUri":selected_import_uri(&query.root,true).expect("build URI"),"workspaceState":"open","descriptionAvailable":true})));
        let states = query
            .validate_response(response)
            .expect("software nested prefix fixture");
        let phase = states
            .plan(&[states.root.join("library")])
            .expect("prefix")
            .expect("close nested modules");
        assert_eq!(
            phase.closed,
            [
                states.root.join("library"),
                states.root.join("library/nested")
            ]
            .into_iter()
            .collect()
        );
        assert!(phase.restored.is_empty());
        assert!(
            states
                .plan(&[states.root.join("library/src")])
                .expect("unmatched descendant prefix")
                .is_none()
        );
        for ignored in [
            vec![states.root.clone()],
            vec![PathBuf::from("relative")],
            vec![states.root.join("../private-canary")],
            vec![states.root.join("./private-canary")],
            vec![states.root.join("private\ncanary")],
            vec![states.root.join("library"); 2],
            vec![states.root.join("library"); 129],
        ] {
            let message = states
                .plan(&ignored)
                .expect_err("invalid policy")
                .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("canary")
                    && !message.contains("groovy-module-state")
            );
        }
    }

    #[test]
    fn gradle_module_policy_rejects_inconsistent_or_unbounded_provider_states() {
        let (query, response) = gradle_module_policy_fixture("closed");
        let mut cases = vec![json!(null), json!([]), json!({})];
        for (key, value) in [
            ("version", json!(2)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", json!("file:///private-canary/")),
            ("modules", json!(null)),
            ("modules", json!(vec![response["modules"][0].clone(); 128])),
            ("private-field", json!("private-value")),
        ] {
            let mut invalid = response.clone();
            invalid[key] = value;
            cases.push(invalid);
        }
        for (key, value) in [
            ("moduleUri", response["rootUri"].clone()),
            ("moduleUri", json!("file:///groovy-module-state/%6cibrary/")),
            (
                "moduleUri",
                json!("file:///groovy-module-state/library/?private=value"),
            ),
            ("moduleUri", json!("https://private-canary.invalid/")),
            ("moduleUri", json!("x".repeat(4097))),
            ("buildRootUri", json!("file:///private-canary/")),
            (
                "buildRootUri",
                json!(selected_import_uri(&query.root.join("sibling"), true).expect("URI")),
            ),
            ("workspaceState", json!("unknown")),
            ("workspaceState", json!("x".repeat(4097))),
            ("descriptionAvailable", json!("private-value")),
            ("workspaceState", json!(true)),
            ("descriptionAvailable", json!(false)),
            ("private-field", json!("private-value")),
        ] {
            let mut invalid = response.clone();
            invalid["modules"][1][key] = value;
            cases.push(invalid);
        }
        let mut duplicate = response.clone();
        duplicate["modules"][1] = duplicate["modules"][0].clone();
        cases.push(duplicate);
        for key in ["version", "rootUri", "modules", "fullSynchronizationKnown"] {
            let mut invalid = response.clone();
            invalid.as_object_mut().expect("object").remove(key);
            cases.push(invalid);
        }
        for invalid in cases {
            let message = query
                .validate_response(invalid)
                .expect_err("unsafe response")
                .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("private")
                    && !message.contains("groovy-module-state")
            );
        }
        let mut limit = response.clone();
        limit["modules"]=json!((0..127).map(|index| json!({"moduleUri":selected_import_uri(&query.root.join(format!("module{index}")),true).expect("URI"),"buildRootUri":response["rootUri"],"workspaceState":"absent","descriptionAvailable":false})).collect::<Vec<_>>());
        let states = query
            .validate_response(limit)
            .expect("software127 non-root record limit");
        assert_eq!(states.modules.len(), 127);
        let mut empty = response;
        empty["modules"] = json!([]);
        assert!(
            query
                .validate_response(empty)
                .expect("root-only build")
                .plan(&[])
                .expect("empty policy")
                .is_none()
        );
    }

    #[gpui::test]
    async fn gradle_module_policy_queries_only_the_advertised_provider_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        for supported in [false, true] {
            let (query, response) = gradle_module_policy_fixture("closed");
            let expected = query.parameters.clone();
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![if supported {
                            expected.command.clone()
                        } else {
                            "groovy.gradle.importedProjects".into()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let mode = mode.clone();
                let requests = requests.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let response = response.clone();
                    let mode = mode.load(Ordering::SeqCst);
                    async move {
                        match mode {
                            0 => Ok(Some(response)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!(
                                "private host credential at /groovy-module-state"
                            )),
                            _ => Ok(Some(json!({"private-field":"private-value"}))),
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized");
            if !supported {
                let message = query
                    .query(server, Duration::from_secs(10))
                    .await
                    .expect_err("no fallback")
                    .to_string();
                assert!(message.contains("extension setup"));
                assert_eq!(requests.load(Ordering::SeqCst), 0);
                continue;
            }
            query
                .clone()
                .query(server.clone(), Duration::from_secs(10))
                .await
                .expect("qualified reply");
            for value in 1..=3 {
                mode.store(value, Ordering::SeqCst);
                let message = query
                    .clone()
                    .query(server.clone(), Duration::from_secs(10))
                    .await
                    .expect_err("failed read")
                    .to_string();
                assert!(
                    message.contains("refresh")
                        && !message.contains("private")
                        && !message.contains("groovy-module-state")
                );
            }
            mode.store(0, Ordering::SeqCst);
            let states = query
                .query(server, Duration::from_secs(10))
                .await
                .expect("read recovery");
            assert!(states.plan(&[]).expect("restore default modules").is_some());
            assert_eq!(requests.load(Ordering::SeqCst), 5);
        }
    }

    fn gradle_module_state_fixture() -> (SelectedGradleModuleStates, serde_json::Value) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/module-states-v1.json"
        ))
        .expect("captured production receipt");
        assert_eq!(fixture["providerVersion"], "0.34.0");
        let root = Path::new(if cfg!(windows) {
            "C:\\groovy-module-state"
        } else {
            "/groovy-module-state"
        });
        let receipt: serde_json::Value =
            serde_json::from_str(&fixture["receipt"].to_string().replace(
                "file:///groovy-module-state/",
                selected_import_uri(root, true).expect("root URI").as_str(),
            ))
            .expect("host-remapped fixture");
        let closed = receipt["closedProjectUris"]
            .as_array()
            .expect("closed modules")
            .iter()
            .map(|uri| selected_import_path(uri.as_str().expect("URI"), true).expect("path"))
            .collect::<Vec<_>>();
        (
            SelectedGradleModuleStates::new(root, &closed, &[]).expect("module selection"),
            receipt,
        )
    }

    #[test]
    fn selected_gradle_module_states_reject_unsafe_or_ambiguous_mutation_sets() {
        let (selection, _) = gradle_module_state_fixture();
        let root = &selection.root;
        let module = root.join("library");
        for (closed, restored) in [
            (vec![], vec![]),
            (vec![root.clone()], vec![]),
            (vec![module.clone(); 2], vec![]),
            (vec![module.clone()], vec![module.clone()]),
            (vec![module.clone(); 129], vec![]),
            (vec![PathBuf::from("relative")], vec![]),
            (vec![root.join("../private-canary")], vec![]),
            (vec![root.join("./private-canary")], vec![]),
            (vec![root.join("module//private-canary")], vec![]),
            (vec![root.join("private\ncanary")], vec![]),
            (vec![root.join("x".repeat(4097))], vec![]),
            (
                vec![],
                vec![root.parent().expect("parent").join("private-canary")],
            ),
        ] {
            let message = SelectedGradleModuleStates::new(root, &closed, &restored)
                .expect_err("unsafe request")
                .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("canary")
                    && !message.contains("groovy-module-state")
            );
        }
        let paths = (0..128)
            .map(|index| root.join(format!("module{index}")))
            .collect::<Vec<_>>();
        SelectedGradleModuleStates::new(root, &paths[..64], &paths[64..])
            .expect("combined exact limit");
        let extra = root.join("extra");
        assert!(SelectedGradleModuleStates::new(root, &paths, &[extra]).is_err());
        let restore =
            SelectedGradleModuleStates::new(root, &[], &[module]).expect("restore-only phase");
        assert_eq!(
            restore.parameters.command,
            "groovy.gradle.applyModuleStates"
        );
        assert_eq!(restore.parameters.arguments[0]["offline"], true);
    }

    #[test]
    fn selected_gradle_module_states_admit_captured_receipts_without_sync_claims() {
        let (selection, receipt) = gradle_module_state_fixture();
        selection
            .validate_receipt(receipt.clone())
            .expect("captured actual0.34 reply");
        let mut reordered = receipt.clone();
        reordered["closedProjectUris"]
            .as_array_mut()
            .expect("modules")
            .reverse();
        selection
            .validate_receipt(reordered)
            .expect("set identity independent of provider order");
        let mut cases = vec![json!(null), json!([]), json!({})];
        let first = receipt["closedProjectUris"][0].clone();
        for (field, value) in [
            ("version", json!(2)),
            ("version", json!("1")),
            ("moduleStatePhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", first.clone()),
            ("rootUri", json!("file:///private-canary/")),
            ("private-field", json!("private-value")),
            ("closedProjectUris", json!([])),
            ("closedProjectUris", json!([first.clone(), first.clone()])),
            ("closedProjectUris", json!(vec![first.clone(); 129])),
            (
                "closedProjectUris",
                json!(["https://private-canary.invalid/module"]),
            ),
            (
                "closedProjectUris",
                json!(["file:///groovy-module-state/%6cibrary/"]),
            ),
            (
                "closedProjectUris",
                json!(["file:///groovy-module-state/library/?private=value"]),
            ),
            ("closedProjectUris", json!(["x".repeat(4097)])),
            ("closedProjectUris", json!([true])),
            ("restoredProjectUris", json!([first])),
        ] {
            let mut value_receipt = receipt.clone();
            value_receipt[field] = value;
            cases.push(value_receipt);
        }
        for field in [
            "version",
            "rootUri",
            "closedProjectUris",
            "restoredProjectUris",
            "moduleStatePhaseComplete",
            "fullSynchronizationKnown",
        ] {
            let mut missing = receipt.clone();
            missing.as_object_mut().expect("receipt").remove(field);
            cases.push(missing);
        }
        for invalid in cases {
            let message = selection
                .validate_receipt(invalid)
                .expect_err("unsafe phase receipt")
                .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("private")
                    && !message.contains("groovy-module-state")
            );
        }
        let root = &selection.root;
        let mixed = SelectedGradleModuleStates::new(
            root,
            &[root.join("library")],
            &[root.join("application")],
        )
        .expect("mixed phase");
        let request = &mixed.parameters.arguments[0];
        let reply = json!({"version":1,"rootUri":request["rootUri"],"closedProjectUris":request["closedProjectUris"],"restoredProjectUris":request["restoredProjectUris"],"moduleStatePhaseComplete":true,"fullSynchronizationKnown":false});
        mixed
            .validate_receipt(reply)
            .expect("exact disjoint mixed receipt");
    }

    #[gpui::test]
    async fn selected_gradle_module_states_use_advertised_command_and_cancel_requests(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::AppContext as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        for supported in [false, true] {
            let (selection, receipt) = gradle_module_state_fixture();
            let expected = selection.parameters.clone();
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let cancelled = Arc::new(AtomicUsize::new(0));
            let (entered_sender, entered_receiver) = async_channel::unbounded();
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![if supported {
                            expected.command.clone()
                        } else {
                            "java.project.changeImportedProjects".into()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                let mode = mode.clone();
                let cancelled = cancelled.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let receipt = receipt.clone();
                    let current_mode = mode.load(Ordering::SeqCst);
                    let cancelled = cancelled.clone();
                    let entered_sender = entered_sender.clone();
                    async move {
                        match current_mode {
                            0 => Ok(Some(receipt)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!(
                                "private host credential at /groovy-module-state failed"
                            )),
                            3 => Ok(Some(json!({"private-field":"private-value"}))),
                            _ => {
                                let _cancel_guard = util::defer(move || {
                                    cancelled.fetch_add(1, Ordering::SeqCst);
                                });
                                entered_sender.send(()).await.expect("entered observer");
                                futures::future::pending().await
                            }
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized fake");
            if !supported {
                let message = selection
                    .execute_phase(server, Duration::from_secs(10))
                    .await
                    .expect_err("no upstream fallback")
                    .to_string();
                assert!(message.contains("extension setup"));
                assert_eq!(requests.load(Ordering::SeqCst), 0);
                continue;
            }
            selection
                .clone()
                .execute_phase(server.clone(), Duration::from_secs(10))
                .await
                .expect("captured phase");
            for current_mode in 1..=4 {
                mode.store(current_mode, Ordering::SeqCst);
                let message = selection
                    .clone()
                    .execute_phase(server.clone(), Duration::from_secs(1))
                    .await
                    .expect_err("failed phase")
                    .to_string();
                assert!(
                    message.contains("refresh")
                        && !message.contains("private")
                        && !message.contains("groovy-module-state")
                );
            }
            entered_receiver
                .recv()
                .await
                .expect("timeout handler entered");
            cx.run_until_parked();
            assert_eq!(cancelled.load(Ordering::SeqCst), 1);
            let task = cx.background_spawn(
                selection
                    .clone()
                    .execute_phase(server.clone(), Duration::MAX),
            );
            entered_receiver.recv().await.expect("drop handler entered");
            drop(task);
            cx.run_until_parked();
            assert_eq!(cancelled.load(Ordering::SeqCst), 2);
            mode.store(0, Ordering::SeqCst);
            selection
                .execute_phase(server, Duration::from_secs(10))
                .await
                .expect("recovery after timeout/drop");
            assert_eq!(requests.load(Ordering::SeqCst), 7);
        }
    }

    #[test]
    fn selected_gradle_import_requires_bounded_absolute_root_and_exact_phase_receipt() {
        let root = selected_import_fixture().root;
        let selection = SelectedGradleImport::new(&root).expect("Gradle selection");
        assert_eq!(selection.parameters.command, "groovy.gradle.importSelected");
        assert_eq!(
            selection.parameters.arguments,
            vec![
                json!({"rootUri": selected_import_uri(&root, true).expect("root URI"), "offline": true})
            ]
        );
        for path in [
            PathBuf::from("relative"),
            root.join("../alias"),
            root.join("private\ncanary"),
            root.join("x".repeat(4097)),
        ] {
            let error = SelectedGradleImport::new(&path)
                .expect_err("invalid root")
                .to_string();
            assert!(!error.contains("canary"));
        }
        let receipt = json!({"version": 1, "rootUri": selected_import_uri(&root, true).expect("root URI"), "selectedImportPhaseComplete": true, "fullSynchronizationKnown": false});
        selection
            .validate_receipt(receipt.clone())
            .expect("phase receipt");
        let mut invalid = vec![json!(null), json!([]), json!({}), json!({"version": 1})];
        for (field, value) in [
            ("version", json!(true)),
            ("version", json!(2)),
            ("selectedImportPhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", json!("https://private.invalid/canary")),
            ("rootUri", json!("file:///private/canary")),
            ("rootUri", json!("x".repeat(4097))),
            ("privateCanary", json!(true)),
        ] {
            let mut changed = receipt.clone();
            changed[field] = value;
            invalid.push(changed);
        }
        for value in invalid {
            let error = selection
                .validate_receipt(value)
                .expect_err("invalid receipt")
                .to_string();
            assert!(!error.contains("canary"));
            assert!(error.contains("refresh"));
        }
    }

    #[test]
    fn scoped_gradle_state_codecs_preserve_authority_and_strict_legacy_admission() {
        let workspace = selected_import_fixture().root;
        let root = workspace.join("nested");
        let module = root.join("module");
        let workspace_uri = selected_import_uri(&workspace, true).expect("workspace");
        let root_uri = selected_import_uri(&root, true).expect("root");
        let module_uri = selected_import_uri(&module, true).expect("module");
        let root_query = GradleRootStateQuery::new_scoped(&workspace, &root).expect("root query");
        let root_phase = SelectedGradleRootState::new(&root, false)
            .expect("root phase")
            .with_scoped_authority(&workspace)
            .expect("root authority");
        let module_query =
            GradleModuleStateQuery::new_scoped(&workspace, &root).expect("module query");
        let module_phase =
            SelectedGradleModuleStates::new(&root, std::slice::from_ref(&module), &[])
                .expect("module phase")
                .with_scoped_authority(&workspace)
                .expect("module authority");
        let validate = |kind, value| -> Result<()> {
            match kind {
                0 => root_query.validate_response(value).map(|_| ()),
                1 => root_phase.validate_receipt(value),
                2 => module_query.validate_response(value).map(|_| ()),
                _ => module_phase.validate_receipt(value),
            }
        };
        let records = json!([{"moduleUri":module_uri,"buildRootUri":root_uri,"workspaceState":"closed","descriptionAvailable":true}]);
        let values = [
            json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,"workspaceState":"open","descriptionAvailable":true,"fullSynchronizationKnown":false}),
            json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,"open":false,"rootStatePhaseComplete":true,"fullSynchronizationKnown":false}),
            json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,"modules":records,"fullSynchronizationKnown":false}),
            json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,"closedProjectUris":[module_uri],"restoredProjectUris":[],"moduleStatePhaseComplete":true,"fullSynchronizationKnown":false}),
        ];
        for (kind, value) in values.into_iter().enumerate() {
            validate(kind, value.clone()).expect("exact scoped data");
            for (field, replacement) in [
                ("version", json!(1)),
                ("version", json!(true)),
                ("workspaceRootUri", json!("file:///foreign-private-canary/")),
                ("buildRootUri", json!(workspace_uri)),
                ("buildRootUri", json!("x".repeat(4097))),
                ("rootUri", json!(root_uri)),
                ("fullSynchronizationKnown", json!(true)),
                ("privateCanary", json!(true)),
            ] {
                let mut invalid = value.clone();
                invalid[field] = replacement;
                let message = validate(kind, invalid)
                    .expect_err("invalid scoped authority or schema")
                    .to_string();
                assert!(!message.contains("canary") && message.contains("refresh"));
            }
            for field in [
                "version",
                "workspaceRootUri",
                "buildRootUri",
                "fullSynchronizationKnown",
            ] {
                let mut invalid = value.clone();
                invalid
                    .as_object_mut()
                    .expect("object")
                    .remove(field)
                    .expect("present");
                validate(kind, invalid).expect_err("missing scoped field");
            }
            validate(kind, json!(null)).expect_err("missing data");
        }
        for parameters in [
            &root_query.parameters,
            &root_phase.parameters,
            &module_query.parameters,
            &module_phase.parameters,
        ] {
            assert!(parameters.command.ends_with(".v2"));
            let request = parameters.arguments.first().expect("argument");
            assert_eq!(request["workspaceRootUri"], json!(workspace_uri));
            assert_eq!(request["buildRootUri"], json!(root_uri));
            assert_eq!(request["offline"], true);
            assert!(request.get("rootUri").is_none());
        }
        GradleRootStateQuery::new_scoped(&root, &workspace)
            .expect_err("ancestor is not a scoped build");
        GradleModuleStateQuery::new_scoped(&workspace, &workspace.join("../private-canary"))
            .expect_err("lexical alias is not a scoped build");
        module_query.validate_response(json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,
            "modules":[{"moduleUri":module_uri,"buildRootUri":root_uri,"workspaceState":"open","descriptionAvailable":false}],
            "fullSynchronizationKnown":false})).expect_err("scoping cannot weaken existing state admission");
        module_phase.validate_receipt(json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":root_uri,
            "closedProjectUris":[module_uri,module_uri],"restoredProjectUris":[],"moduleStatePhaseComplete":true,
            "fullSynchronizationKnown":false})).expect_err("scoping cannot hide duplicate mutation identities");
    }

    #[test]
    fn scoped_gradle_module_policy_requires_current_targets_and_retained_root_admission() {
        let workspace = selected_import_fixture().root;
        let root = workspace.join("nested");
        let module = root.join("module");
        let query = GradleModuleStateQuery::new_scoped(&workspace, &root).expect("scope");
        let facts = query.validate_response(json!({"version":2,"workspaceRootUri":selected_import_uri(&workspace,true).expect("workspace"),
            "buildRootUri":selected_import_uri(&root,true).expect("root"),"modules":[{
                "moduleUri":selected_import_uri(&module,true).expect("module"),"buildRootUri":selected_import_uri(&root,true).expect("root"),
                "workspaceState":"open","descriptionAvailable":true}],"fullSynchronizationKnown":false})).expect("current facts");
        facts
            .validate_ignored_targets(std::slice::from_ref(&module))
            .expect("current target");
        facts
            .validate_ignored_targets(&[root.join("unresolved")])
            .expect_err("unknown target cannot silently succeed");
        let phase = facts
            .plan(std::slice::from_ref(&module))
            .expect("policy")
            .expect("close needed");
        assert_eq!(
            phase.module_paths().collect::<Vec<_>>(),
            vec![module.as_path()]
        );
        let fresh = GradleRootState {
            root: root.clone(),
            workspace_state: GradleWorkspaceState::Absent,
            description_available: false,
        };
        assert!(
            fresh
                .ignore_phase_if_retained()
                .expect("absent root is already inactive")
                .is_none()
        );
        let retained = GradleRootState {
            root,
            workspace_state: GradleWorkspaceState::Closed,
            description_available: true,
        };
        assert!(
            retained
                .ignore_phase_if_retained()
                .expect("retained no-op still preflights children/orphans")
                .is_some()
        );
    }

    #[test]
    fn scoped_gradle_import_requires_exact_authority_and_phase_receipt() {
        let root = selected_import_fixture().root;
        let build = root.join("nested");
        let selection = SelectedGradleImport::new_scoped(&root, &build).expect("scoped import");
        let workspace_uri = selected_import_uri(&root, true).expect("workspace URI");
        let build_uri = selected_import_uri(&build, true).expect("build URI");
        assert_eq!(
            selection.parameters.command,
            "groovy.gradle.importSelected.v2"
        );
        assert_eq!(
            selection.parameters.arguments,
            vec![json!({"workspaceRootUri":workspace_uri,"buildRootUri":build_uri,"offline":true})]
        );
        for (workspace, build) in [
            (root.clone(), root.join("../foreign")),
            (build, root.clone()),
            (root.clone(), root.join("private\ncanary")),
            (root.clone(), root.join("x".repeat(4097))),
        ] {
            let message = SelectedGradleImport::new_scoped(&workspace, &build)
                .expect_err("invalid scope")
                .to_string();
            assert!(!message.contains("canary"));
        }
        let receipt = json!({"version":2,"workspaceRootUri":workspace_uri,"buildRootUri":build_uri,"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        selection
            .validate_receipt(receipt.clone())
            .expect("phase receipt");
        SelectedGradleImport::new(&root)
            .expect("legacy")
            .validate_receipt(receipt.clone())
            .expect_err("legacy rejects scoped receipt");
        for (field, value) in [
            ("version", json!(1)),
            ("selectedImportPhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("workspaceRootUri", json!("file:///foreign-private-canary/")),
            ("buildRootUri", json!(workspace_uri)),
            ("buildRootUri", json!("x".repeat(4097))),
            ("privateCanary", json!(true)),
        ] {
            let mut invalid = receipt.clone();
            invalid[field] = value;
            let message = selection
                .validate_receipt(invalid)
                .expect_err("invalid receipt")
                .to_string();
            assert!(message.contains("refresh") && !message.contains("canary"));
        }
    }

    #[gpui::test]
    async fn scoped_gradle_selection_captures_inputs_and_initialized_authority(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = if cfg!(windows) {
            Path::new("C:/scoped-selection")
        } else {
            Path::new("/scoped-selection")
        };
        let (fs,project)=default_root_fixture(root.to_str().expect("root"),json!({
            "first":{"build.gradle":"unread","settings.gradle":"unread","nested":{"build.gradle":"unread"}},
            "second":{"build.gradle.kts":"unread"},"pom.xml":"unread",
            ".env":{"build.gradle":"private"},".gitignore":"ignored/\n","ignored":{"build.gradle":"unread"}
        }),cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let settings =
            |files: Option<Vec<&str>>, ignored: Vec<&str>| settings::GradleBuildSettingsContent {
                selected_build_files: files
                    .map(|files| files.into_iter().map(str::to_owned).collect()),
                ignored_projects: Some(ignored.into_iter().map(str::to_owned).collect()),
            };
        for intent in [settings(None, vec![]), settings(None, vec!["first"])] {
            assert!(
                ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &intent)
                    .expect("legacy path")
                    .is_none()
            );
        }
        for intent in [
            settings(
                Some(vec!["first/build.gradle", "first/nested/build.gradle"]),
                vec![],
            ),
            settings(Some(vec!["first/build.gradle"]), vec!["other"]),
            settings(Some(vec![".env/build.gradle"]), vec![]),
            settings(Some(vec!["ignored/build.gradle"]), vec![]),
            settings(Some(vec!["missing/build.gradle"]), vec![]),
        ] {
            let message = ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &intent)
                .expect_err("restricted policy")
                .to_string();
            assert!(message.contains("refresh") || message.contains("review"));
        }
        let intent = settings(
            Some(vec![
                "second/build.gradle.kts",
                "first/build.gradle",
                "first/settings.gradle",
            ]),
            vec![],
        );
        let mut plan = ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &intent)
            .expect("selection")
            .expect("nested roots");
        assert_eq!(plan.inputs.len(), 3);
        assert_eq!(plan.roots, vec![root.join("second"), root.join("first")]);
        plan.validate_current(&snapshot, snapshot.id())
            .expect("current");
        let (server, _fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            Default::default(),
            &mut cx.to_async(),
        );
        let uri = |path: &Path| selected_import_uri(path, true).expect("URI");
        let folders = |paths: Vec<PathBuf>| {
            paths
                .iter()
                .map(|path| uri(path).as_str().parse().expect("LSP URI"))
                .collect()
        };
        server.set_workspace_folders(folders(vec![root.join("first"), root.join("second")]));
        plan.bind_initialized_folders(&server)
            .expect("existing per-build authorities");
        for import in plan.imports() {
            assert_eq!(import.workspace_root.as_ref(), Some(&import.root));
        }
        plan.validate_initialized_folders(&server)
            .expect("unchanged folders");
        server.set_workspace_folders(folders(vec![root.to_path_buf(), root.join("first")]));
        plan.validate_initialized_folders(&server)
            .expect_err("authority changed");
        plan.bind_initialized_folders(&server)
            .expect("existing common parent");
        for import in plan.imports() {
            assert_eq!(import.workspace_root.as_deref(), Some(root));
        }
        server.set_workspace_folders(folders(vec![root.join("first")]));
        plan.bind_initialized_folders(&server)
            .expect_err("no invented second authority");
        fs.insert_tree(root, json!({"unselected":{"build.gradle":"new"}}))
            .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect("unselected input does not invalidate explicit capture");
        fs.insert_tree(&root.join("first"), json!({"build.gradle":"changed"}))
            .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect_err("captured entry changed");
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_captures_selected_modules_and_absent_descriptors(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = Path::new(util::path!("/metadata-inputs"));
        let (fs, project) = default_root_fixture(root.to_str().expect("root"), json!({
            "first":{"build.gradle":"root", "child":{"build.gradle":"child", "C.java":"class C {}"}},
            "second":{"settings.gradle":"other"}
        }), cx).await;
        let (server, _fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            Default::default(),
            &mut cx.to_async(),
        );
        server.set_workspace_folders(
            [selected_import_uri(root, true)
                .expect("URI")
                .as_str()
                .parse()
                .expect("folder")]
            .into(),
        );
        for mode in 0..4 {
            let snapshot = default_root_snapshot(&project, cx);
            let settings = settings::GradleBuildSettingsContent {
                selected_build_files: Some(vec!["first/build.gradle".into()]),
                ..Default::default()
            };
            let mut plan = ScopedGradleImports::for_metadata(&snapshot, snapshot.id(), &settings)
                .expect("metadata selection");
            plan.bind_initialized_folders(&server)
                .expect("existing folder");
            let rows = vec![ScopedGradleAssociation {
                scope: root.into(),
                project: root.join("first/child"),
                build: root.join("first"),
                java_nature: true,
            }];
            plan.resolve_metadata_associations(rows, &snapshot, &server)
                .expect("known active module");
            let queries = plan.metadata_queries().expect("queries");
            assert_eq!(queries.len(), 1);
            assert_eq!(queries[0].workspace, root);
            assert_eq!(queries[0].build, root.join("first"));
            assert_eq!(
                queries[0].modules,
                BTreeSet::from([root.join("first/child")])
            );
            assert_eq!(plan.metadata_inputs.len(), 8);
            plan.validate_current(&snapshot, snapshot.id())
                .expect("current witnesses");
            match mode {
                0 => {
                    fs.insert_tree(&root.join("first/child"), json!({"build.gradle":"edited"}))
                        .await
                }
                1 => {
                    fs.insert_tree(
                        &root.join("first/child"),
                        json!({"settings.gradle":"added"}),
                    )
                    .await
                }
                2 => fs
                    .remove_file(&root.join("first/child/build.gradle"), Default::default())
                    .await
                    .expect("remove descriptor"),
                3 => {
                    fs.rename(
                        &root.join("first/child"),
                        &root.join("first/old-child"),
                        Default::default(),
                    )
                    .await
                    .expect("move module");
                    fs.insert_tree(&root.join("first/child"), json!({"C.java":"class C {}"}))
                        .await;
                }
                _ => unreachable!(),
            }
            cx.run_until_parked();
            let current = default_root_snapshot(&project, cx);
            plan.validate_current(&current, current.id())
                .expect_err("late module change must refuse publication");
        }
        let snapshot = default_root_snapshot(&project, cx);
        let mut defaults =
            ScopedGradleImports::for_metadata(&snapshot, snapshot.id(), &Default::default())
                .expect("default known roots");
        defaults
            .bind_initialized_folders(&server)
            .expect("parent folder");
        defaults
            .resolve_metadata_associations(
                vec![
                    ScopedGradleAssociation {
                        scope: root.into(),
                        project: root.join("first/child"),
                        build: root.join("first"),
                        java_nature: true,
                    },
                    ScopedGradleAssociation {
                        scope: root.into(),
                        project: root.join("second"),
                        build: root.join("second"),
                        java_nature: true,
                    },
                ],
                &snapshot,
                &server,
            )
            .expect("provider roots");
        assert_eq!(
            defaults
                .metadata_queries()
                .expect("both known builds")
                .len(),
            2
        );
        let empty = settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec![]),
            ..Default::default()
        };
        let mut plan = ScopedGradleImports::for_metadata(&snapshot, snapshot.id(), &empty)
            .expect("empty intent");
        plan.bind_initialized_folders(&server).expect("folder");
        plan.resolve_metadata_associations(vec![], &snapshot, &server)
            .expect("no active metadata needed");
        assert!(plan.metadata_queries().expect("empty").is_empty());
    }

    #[gpui::test]
    async fn scoped_gradle_empty_selection_binds_cached_ids_and_current_directories(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = if cfg!(windows) {
            Path::new("C:/empty-scoped")
        } else {
            Path::new("/empty-scoped")
        };
        let (fs, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({
                "first":{"build.gradle":"unread", "child":{"C.java":"class C {}"}},
                "second":{"settings.gradle":"unread"},"maven":{"pom.xml":"unread"},
                ".env":{"private":{"build.gradle":"private"}}
            }),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        let settings = settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec![]),
            ignored_projects: Some(vec!["first".into()]),
        };
        let mut plan = ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &settings)
            .expect("empty intent")
            .expect("scoped plan");
        assert!(plan.is_empty_selection() && plan.imports().is_empty());
        assert_eq!(plan.inputs.len(), 2);
        assert!(!plan.requires_policy());
        let (server, _fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            Default::default(),
            &mut cx.to_async(),
        );
        let uri = |path: &Path| selected_import_uri(path, true).expect("URI");
        server.set_workspace_folders([uri(root).as_str().parse().expect("folder")].into());
        plan.bind_initialized_folders(&server).expect("authority");
        let valid = json!({"version":1,"rootUri":uri(root),"projects":[
            {"projectUri":uri(&root.join("first")),"buildRootUri":uri(&root.join("first")),"javaNature":true},
            {"projectUri":uri(&root.join("first/child")),"buildRootUri":uri(&root.join("first")),"javaNature":false}],
            "cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false});
        for (field, value) in [
            ("version", json!(2)),
            ("partial", json!(true)),
            ("readOnly", json!(false)),
            ("cachedOnly", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", json!("file:///private-canary/")),
            ("unknown", json!(true)),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            let message = plan
                .validate_cached_association_reply(root, invalid)
                .expect_err("strict contract")
                .to_string();
            assert!(message.contains("refresh") && !message.contains("canary"));
        }
        for records in [
            json!([valid["projects"][0].clone(), valid["projects"][0].clone()]),
            json!([{"projectUri":uri(&root.join("second")),"buildRootUri":uri(&root.join("first")),"javaNature":true}]),
            json!([{"projectUri":"file:///private-canary/","buildRootUri":"file:///private-canary/","javaNature":true}]),
            json!(vec![valid["projects"][0].clone(); 129]),
        ] {
            let mut invalid = valid.clone();
            invalid["projects"] = records;
            plan.validate_cached_association_reply(root, invalid)
                .expect_err("bounded current IDs");
        }
        let rows = plan
            .validate_cached_association_reply(root, valid)
            .expect("cached rows");
        plan.resolve_cached_associations(rows.clone(), &snapshot, &server)
            .expect("current directories");
        plan.validate_cached_associations(root, &rows)
            .expect("same cached rows");
        plan.validate_cached_associations(root, &rows[..1])
            .expect_err("association changed");
        let paths = rows
            .iter()
            .map(|row| {
                confined_project_path(&row.project, root, snapshot.id(), snapshot.path_style())
                    .expect("path")
            })
            .collect::<Vec<_>>();
        plan.validate_cached_namespace(root, &paths)
            .expect("v1 agrees");
        plan.validate_cached_namespace(root, &paths[..1])
            .expect_err("incomplete v1");
        plan.validate_ignored_namespace(&paths)
            .expect_err("must actually clear");
        plan.validate_ignored_namespace(&[])
            .expect("no active Gradle");
        let unlink = plan.cached_unlink(root).expect("existing unlink");
        assert_eq!(unlink.parameters.command, "groovy.gradle.unlinkSelected");
        assert_eq!(
            unlink.parameters.arguments[0]["projectUris"]
                .as_array()
                .expect("IDs")
                .len(),
            2
        );
        plan.resolve_cached_associations(rows, &snapshot, &server)
            .expect_err("resolved only once");
        fs.insert_tree(&root.join("second"), json!({"settings.gradle":"changed"}))
            .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect_err("inputs changed");
        let mut plan = ScopedGradleImports::for_selection(&current, current.id(), &settings)
            .expect("current")
            .expect("empty");
        plan.bind_initialized_folders(&server).expect("authority");
        let row = ScopedGradleAssociation {
            scope: root.to_path_buf(),
            project: root.join("first/child"),
            build: root.join("first"),
            java_nature: true,
        };
        plan.resolve_cached_associations(vec![row], &current, &server)
            .expect("capture directories");
        fs.rename(
            &root.join("first/child"),
            &root.join("first/old-child"),
            Default::default(),
        )
        .await
        .expect("move directory");
        fs.insert_tree(&root.join("first/child"), json!({"C.java":"class C {}"}))
            .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect_err("same-path directory replacement");
    }

    #[gpui::test]
    async fn scoped_gradle_transition_retains_selected_roots_and_refuses_overlap(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = if cfg!(windows) {
            Path::new("C:/transition-scoped")
        } else {
            Path::new("/transition-scoped")
        };
        let (_fs,project)=default_root_fixture(root.to_str().expect("root"),json!({
            "first":{"build.gradle":"unread", "child":{"build.gradle":"unread"}},"second":{"settings.gradle":"unread"}
        }),cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let settings = settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec!["second/settings.gradle".into()]),
            ..Default::default()
        };
        let mut plan = ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &settings)
            .expect("intent")
            .expect("scoped plan");
        let (server, _fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            Default::default(),
            &mut cx.to_async(),
        );
        server.set_workspace_folders(
            [selected_import_uri(root, true)
                .expect("URI")
                .as_str()
                .parse()
                .expect("folder")]
            .into(),
        );
        plan.bind_initialized_folders(&server).expect("authority");
        let row = |project: &str, build: &str| ScopedGradleAssociation {
            scope: root.to_path_buf(),
            project: root.join(project),
            build: root.join(build),
            java_nature: true,
        };
        let rows = vec![
            row("first", "first"),
            row("first/child", "first"),
            row("second", "second"),
        ];
        plan.resolve_cached_associations(rows.clone(), &snapshot, &server)
            .expect("independent current roots");
        assert_eq!(plan.roots, vec![root.join("second")]);
        let unlink = plan.cached_unlink(root).expect("omit only first");
        assert_eq!(
            unlink.projects,
            BTreeSet::from([root.join("first"), root.join("first/child")])
        );
        assert_eq!(plan.imports.len(), 1);
        let paths = rows
            .iter()
            .map(|row| {
                confined_project_path(&row.project, root, snapshot.id(), snapshot.path_style())
                    .expect("path")
            })
            .collect::<Vec<_>>();
        plan.validate_cached_namespace(root, &paths)
            .expect("before transition");
        plan.validate_ignored_namespace(&paths)
            .expect_err("omitted root still active");
        plan.validate_ignored_namespace(&paths[2..])
            .expect("selected membership only");
        let mut identities = BTreeSet::new();
        let reply = json!({"version":1,"rootUri":selected_import_uri(root,true).expect("URI"),"projects":rows.iter().map(|row|json!({"projectUri":selected_import_uri(&row.project,true).expect("URI"),"javaNature":true})).collect::<Vec<_>>(),"partial":false,"fullSynchronizationKnown":false});
        plan.validate_ownership_reply(root, reply, &mut identities)
            .expect("temporary admission only for exact cached IDs");
        let ambiguous = settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec!["first/child/build.gradle".into()]),
            ..Default::default()
        };
        let mut ambiguous =
            ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &ambiguous)
                .expect("intent")
                .expect("scoped");
        ambiguous
            .bind_initialized_folders(&server)
            .expect("authority");
        ambiguous
            .resolve_cached_associations(rows, &snapshot, &server)
            .expect_err("parent build is not an independent deselected root");
    }

    #[gpui::test]
    async fn scoped_gradle_default_policy_uses_one_candidate_snapshot_and_existing_authority(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = if cfg!(windows) {
            Path::new("C:/default-scoped")
        } else {
            Path::new("/default-scoped")
        };
        let (fs, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({
                "first":{"build.gradle":"unread","child":{"build.gradle":"unread"}},
                "second":{"settings.gradle":"unread"},"maven":{"pom.xml":"unread"},
                ".env":{"build.gradle":"private"}
            }),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        let settings = settings::GradleBuildSettingsContent {
            ignored_projects: Some(vec!["first/child".into()]),
            ..Default::default()
        };
        let mut plan = ScopedGradleImports::for_default(&snapshot, snapshot.id(), &settings)
            .expect("default plan");
        assert!(plan.inputs.is_empty() && plan.entries.is_empty());
        assert_eq!(
            plan.default_query()
                .expect("sole candidate owner")
                .inputs
                .len(),
            3
        );
        plan.validate_current(&snapshot, snapshot.id())
            .expect("current candidates");
        let (server, _fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            Default::default(),
            &mut cx.to_async(),
        );
        let folder = |path: &Path| {
            selected_import_uri(path, true)
                .expect("URI")
                .as_str()
                .parse()
                .expect("folder URI")
        };
        server.set_workspace_folders(BTreeSet::from([folder(&root.join("first"))]));
        plan.bind_initialized_folders(&server)
            .expect_err("default discovery must own worktree root");
        server.set_workspace_folders(BTreeSet::from([folder(root)]));
        plan.bind_initialized_folders(&server)
            .expect("existing workspace authority");
        for roots in [
            BTreeSet::new(),
            BTreeSet::from([root.join("second")]),
            BTreeSet::from([root.join("first"), root.join("first/child")]),
            BTreeSet::from([root.join("unknown")]),
        ] {
            plan.clone()
                .resolve_defaults(roots, &server)
                .expect_err("empty, unmatched or ambiguous roots");
        }
        plan.resolve_defaults(
            BTreeSet::from([root.join("first"), root.join("second")]),
            &server,
        )
        .expect("current provider roots");
        assert_eq!(plan.imports().len(), 2);
        plan.resolve_defaults(BTreeSet::from([root.join("first")]), &server)
            .expect_err("cannot replace resolved selection");
        server.set_workspace_folders(BTreeSet::from([folder(&root.join("first"))]));
        plan.bind_initialized_folders(&server)
            .expect_err("no authority replacement");
        fs.insert_tree(root, json!({"new":{"build.gradle":"new candidate"}}))
            .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect_err("new default candidate invalidates capture");
    }

    #[gpui::test]
    async fn scoped_gradle_policy_captures_ignored_roots_and_directory_identity(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = if cfg!(windows) {
            Path::new("C:/scoped-policy")
        } else {
            Path::new("/scoped-policy")
        };
        let (fs, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({
                "first":{"build.gradle":"plugins {}", "child":{"build.gradle":"plugins {}"}},
                "maven":{"pom.xml":"<project/>"}
            }),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        let settings = |ignored: &str| settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec!["first/build.gradle".into()]),
            ignored_projects: Some(vec![ignored.into()]),
        };
        let ignored_root =
            ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &settings("first"))
                .expect("ignored root intent")
                .expect("scoped policy");
        assert_eq!(ignored_root.roots, vec![root.join("first")]);
        assert_eq!(
            ignored_root
                .inputs
                .iter()
                .map(|path| path.path.as_unix_str())
                .collect::<Vec<_>>(),
            vec!["first/build.gradle"]
        );
        assert!(
            ignored_root
                .policy_for_build(&root.join("first"))
                .expect("policy")
                .0
        );
        assert!(ignored_root.requires_policy());
        let path = |relative: &str| ProjectPath {
            worktree_id: snapshot.id(),
            path: RelPath::from_unix_str(relative).expect("path").into(),
        };
        ignored_root
            .validate_ignored_namespace(&[path("maven")])
            .expect("Maven path remains unrelated");
        ignored_root
            .validate_ignored_namespace(&[path("first/child")])
            .expect_err("active ignored orphan cannot be hidden");
        let plan =
            ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &settings("first/child"))
                .expect("module ignore")
                .expect("scoped policy");
        assert_eq!(
            plan.policy_for_build(&root.join("first")).expect("policy"),
            (false, vec![root.join("first/child")])
        );
        plan.validate_current(&snapshot, snapshot.id())
            .expect("current identities");
        fs.insert_tree(
            &root.join("first/child"),
            json!({".project":"SDK metadata"}),
        )
        .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        assert!(
            current
                .entry_for_path(
                    RelPath::from_unix_str("first/child/.project").expect("metadata path")
                )
                .is_some()
        );
        plan.validate_current(&current, current.id())
            .expect("metadata changes directory mtime, not identity or authority");
        fs.rename(
            &root.join("first/child"),
            &root.join("first/retired-child"),
            Default::default(),
        )
        .await
        .expect("retire only fixture directory");
        fs.insert_tree(
            &root.join("first/child"),
            json!({"build.gradle":"plugins {}",".project":"SDK metadata"}),
        )
        .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        plan.validate_current(&current, current.id())
            .expect_err("same-path ignored directory replacement retires authority");
        ignored_root
            .validate_current(&current, current.id())
            .expect("unrelated child metadata does not replace ignored root");
        fs.insert_tree(
            &root.join("first"),
            json!({"build.gradle":"changed ignored root script"}),
        )
        .await;
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        ignored_root
            .validate_current(&current, current.id())
            .expect_err("ignored selected root inputs remain bound");
    }

    #[gpui::test]
    async fn scoped_gradle_selection_bounds_namespace_replies(cx: &mut gpui::TestAppContext) {
        let root = if cfg!(windows) {
            Path::new("C:/scoped-replies")
        } else {
            Path::new("/scoped-replies")
        };
        let (_, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({"first":{"build.gradle":""}}),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        let intent = settings::GradleBuildSettingsContent {
            selected_build_files: Some(vec!["first/build.gradle".into()]),
            ..Default::default()
        };
        let plan = ScopedGradleImports::for_selection(&snapshot, snapshot.id(), &intent)
            .expect("plan")
            .expect("nested");
        let uri = |path: &Path| selected_import_uri(path, true).expect("URI");
        let receipt = json!({"version":1,"rootUri":uri(root),"projects":[{"projectUri":uri(&root.join("first")),"javaNature":true}],"partial":false,"fullSynchronizationKnown":false});
        let mut identities = BTreeSet::new();
        assert_eq!(
            plan.validate_ownership_reply(root, receipt.clone(), &mut identities)
                .expect("complete namespace")
                .len(),
            1
        );
        plan.validate_ownership_reply(root, receipt.clone(), &mut identities)
            .expect_err("duplicate across scopes");
        for (field, value) in [
            ("version", json!(2)),
            ("partial", json!(true)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", json!(uri(&root.join("first")))),
            ("rootUri", json!("x".repeat(4097))),
            (
                "projects",
                json!([{"projectUri":uri(&root.join("first")),"javaNature":true,"extra":true}]),
            ),
            (
                "projects",
                json!([{"projectUri":uri(&root.join("outside")),"javaNature":true}]),
            ),
            (
                "projects",
                json!([{"projectUri":"file:///private-canary/","javaNature":true}]),
            ),
            (
                "projects",
                json!([{"projectUri":"file:///private%0Acanary/","javaNature":true}]),
            ),
            (
                "projects",
                json!(vec![
                    json!({"projectUri":uri(&root.join("first")),"javaNature":true});
                    129
                ]),
            ),
            ("extra", json!(true)),
        ] {
            let mut invalid = receipt.clone();
            invalid[field] = value;
            let message = plan
                .validate_ownership_reply(root, invalid, &mut BTreeSet::new())
                .expect_err("invalid scope")
                .to_string();
            assert!(message.contains("refresh") && !message.contains("canary"));
        }
    }

    fn selected_import_fixture() -> SelectedMavenImport {
        let root = if cfg!(windows) {
            Path::new("C:/groovy-selection")
        } else {
            Path::new("/groovy-selection")
        };
        SelectedMavenImport::new(
            root,
            &[
                root.join("application/pom.xml"),
                root.join("library/pom.xml"),
            ],
        )
        .expect("bounded selected import")
    }

    fn selected_import_receipt(selection: &SelectedMavenImport) -> serde_json::Value {
        let request = selection.parameters.arguments.first().expect("one request");
        json!({"version": 1, "rootUri": request["rootUri"], "importedPomUris": request["pomUris"],
            "selectedImportPhaseComplete": true, "fullSynchronizationKnown": false})
    }

    #[test]
    fn selected_maven_import_rejects_invalid_paths_before_dispatch() {
        let selection = selected_import_fixture();
        let root = selection.root;
        for poms in [
            vec![],
            vec![root.join("application/pom.xml"); 33],
            vec![root.join("application/pom.xml"); 2],
            vec![root.join("application/build.gradle")],
            vec![root.join("../foreign/pom.xml")],
            vec![PathBuf::from("relative/pom.xml")],
            vec![root.join("private\nvalue/pom.xml")],
            vec![root.join("x".repeat(4096)).join("pom.xml")],
            vec![root.join("./application/pom.xml")],
            vec![root.join("application//pom.xml")],
        ] {
            let error = SelectedMavenImport::new(&root, &poms).expect_err("invalid selection");
            assert!(!format!("{error:#}").contains("private"));
            assert!(!format!("{error:#}").contains("groovy-selection"));
        }
        let parent = root.parent().expect("fixture parent");
        assert!(SelectedMavenImport::new(&root, &[parent.join("foreign/pom.xml")]).is_err());
        assert!(
            SelectedMavenImport::new(Path::new("relative"), &[PathBuf::from("relative/pom.xml")])
                .is_err()
        );
        let poms = (0..32)
            .map(|index| root.join(format!("module{index}/pom.xml")))
            .collect::<Vec<_>>();
        SelectedMavenImport::new(&root, &poms).expect("exact count limit");
        SelectedMavenImport::new(&root, &[root.join("日本語 space/pom.xml")])
            .expect("encoded unicode identity");
    }

    #[test]
    fn selected_maven_import_receipt_is_exact_and_does_not_certify_synchronization() {
        let selection = selected_import_fixture();
        let receipt = selected_import_receipt(&selection);
        selection
            .validate_receipt(receipt.clone())
            .expect("matching phase receipt");
        let mut reordered = receipt.clone();
        reordered["importedPomUris"]
            .as_array_mut()
            .expect("POM array")
            .reverse();
        selection
            .validate_receipt(reordered)
            .expect("set identity is independent of response order");
        for (field, value) in [
            ("version", json!(2)),
            ("version", json!("private-value")),
            ("rootUri", json!("file:///foreign-root/")),
            (
                "rootUri",
                json!("file:///groovy-selection/../groovy-selection/"),
            ),
            ("rootUri", json!("x".repeat(4097))),
            ("selectedImportPhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("importedPomUris", json!([])),
            (
                "importedPomUris",
                json!(vec![receipt["importedPomUris"][0].clone(); 2]),
            ),
            ("importedPomUris", json!(["file:///foreign/pom.xml"])),
            ("importedPomUris", json!([17])),
            (
                "importedPomUris",
                json!(["file:///groovy-selection/application/pom.xml?private=value"]),
            ),
            (
                "importedPomUris",
                json!(["file:///groovy-selection/application/pom.xml#private-value"]),
            ),
            (
                "importedPomUris",
                json!(["file:///groovy-selection/application/../application/pom.xml"]),
            ),
            ("importedPomUris", json!(["x".repeat(4097)])),
            (
                "importedPomUris",
                json!(vec!["file:///groovy-selection/application/pom.xml"; 33]),
            ),
            (
                "private-secret-field",
                json!({"credential": "private-value"}),
            ),
        ] {
            let mut invalid = receipt.clone();
            invalid[field] = value;
            let error = selection
                .validate_receipt(invalid)
                .expect_err("receipt rejection");
            let message = format!("{error:#}");
            assert!(message.contains("refresh"));
            assert!(!message.contains("private"));
            assert!(!message.contains("foreign"));
            assert!(!message.contains("groovy-selection"));
        }
        for field in [
            "version",
            "rootUri",
            "importedPomUris",
            "selectedImportPhaseComplete",
            "fullSynchronizationKnown",
        ] {
            let mut invalid = receipt.clone();
            invalid.as_object_mut().expect("receipt").remove(field);
            assert!(selection.validate_receipt(invalid).is_err());
        }
        assert!(selection.validate_receipt(json!(null)).is_err());
    }

    #[test]
    fn gradle_default_root_query_empty_inputs_keep_worktree_identity() {
        let root = selected_import_fixture().root;
        let snapshot = |worktree_id| {
            Snapshot::new(
                worktree_id,
                Arc::from(RelPath::from_unix_str("fixture").expect("root name")),
                Arc::from(root.as_path()),
                PathStyle::local(),
            )
        };
        let original = snapshot(WorktreeId::from_usize(1));
        let replacement = snapshot(WorktreeId::from_usize(2));
        let query =
            GradleDefaultImportRootsQuery::new(&original, original.id()).expect("empty query");
        assert!(query.inputs.is_empty() && query.candidates.is_empty());
        query
            .validate_current(&original, original.id())
            .expect("same scope");
        assert!(
            query
                .validate_current(&replacement, replacement.id())
                .is_err()
        );
        assert!(GradleDefaultImportRootsQuery::new(&original, replacement.id()).is_err());
        let long = Snapshot::new(
            original.id(),
            Arc::from(RelPath::from_unix_str("fixture").expect("name")),
            Arc::from(root.join("x".repeat(4097)).as_path()),
            PathStyle::local(),
        );
        let error = GradleDefaultImportRootsQuery::new(&long, long.id())
            .expect_err("URI budget")
            .to_string();
        assert!(error.contains("Gradle") && !error.contains("xxxxx"));
    }

    #[gpui::test]
    async fn gradle_default_root_query_bounds_responses_and_current_candidates(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = if cfg!(windows) {
            Path::new("C:/gradle-default-query")
        } else {
            Path::new("/gradle-default-query")
        };
        let (fs, project) = default_root_fixture(root.to_str().expect("root"), json!({
            "settings.gradle":"unread", "settings.gradle.kts":"unread", "build.gradle":"unread", "build.gradle.kts":"unread",
            "child":{"build.gradle":"unread"}, "pom.xml":"invalid XML is not a Gradle candidate",
            ".env":{"build.gradle":"private"}, "ignored":{"settings.gradle":"ignored"}, ".gitignore":"ignored/\n"
        }), cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let query = GradleDefaultImportRootsQuery::new(&snapshot, snapshot.id()).expect("query");
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("fixture URI");
        assert_eq!(query.inputs.len(), 5);
        assert_eq!(
            query.candidates,
            BTreeSet::from([root.to_path_buf(), root.join("child")])
        );
        assert_eq!(query.parameters.command, "groovy.gradle.defaultImportRoots");
        let files = [
            "build.gradle",
            "build.gradle.kts",
            "child/build.gradle",
            "settings.gradle",
            "settings.gradle.kts",
        ];
        assert_eq!(
            query.parameters.arguments,
            vec![json!({"rootUri":uri(root,true),
            "candidateBuildFileUris":files.map(|path| uri(&root.join(path),false)),"cachedOnly":true})]
        );
        let response = json!({"version":1,"rootUri":uri(root,true),"buildRootUris":[uri(root,true)],
            "cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false});
        assert_eq!(
            query
                .validate_response(response.clone())
                .expect("root subset"),
            BTreeSet::from([root.to_path_buf()])
        );
        let mut empty = response.clone();
        empty["buildRootUris"] = json!([]);
        assert!(
            query
                .validate_response(empty)
                .expect("known empty")
                .is_empty()
        );
        let mut child = response.clone();
        child["buildRootUris"] = json!([uri(&root.join("child"), true)]);
        assert_eq!(
            query.validate_response(child).expect("captured directory"),
            BTreeSet::from([root.join("child")])
        );
        for (field, value) in [
            ("version", json!(2)),
            ("version", json!("1")),
            ("rootUri", json!("file:///private-canary/")),
            ("rootUri", json!("x".repeat(4097))),
            ("cachedOnly", json!(false)),
            ("readOnly", json!(false)),
            ("partial", json!(true)),
            ("fullSynchronizationKnown", json!(true)),
            ("extra", json!(true)),
            ("buildRootUris", json!([uri(root, true), uri(root, true)])),
            (
                "buildRootUris",
                json!([uri(root, true), uri(&root.join("child"), true)]),
            ),
            (
                "buildRootUris",
                json!([uri(&root.join("child"), true), uri(root, true)]),
            ),
            ("buildRootUris", json!(["file:///private-canary/"])),
            ("buildRootUris", json!([uri(&root.join("missing"), true)])),
            ("buildRootUris", json!([uri(&root.join(".env"), true)])),
            ("buildRootUris", json!([uri(&root.join("ignored"), true)])),
            (
                "buildRootUris",
                json!([uri(&root.join("build.gradle"), false)]),
            ),
            (
                "buildRootUris",
                json!([format!("{}child/../", uri(root, true))]),
            ),
            (
                "buildRootUris",
                json!([format!("{}?private-canary", uri(root, true))]),
            ),
            ("buildRootUris", json!(["file:///private%0Acanary/"])),
            ("buildRootUris", json!(["x".repeat(4097)])),
            ("buildRootUris", json!([17])),
            ("buildRootUris", json!(vec![uri(root, true); 129])),
        ] {
            let mut invalid = response.clone();
            invalid[field] = value;
            let error = format!(
                "{:#}",
                query
                    .validate_response(invalid)
                    .expect_err("invalid receipt")
            );
            assert!(
                error.contains("refresh")
                    && !error.contains("private")
                    && !error.contains("gradle-default-query")
            );
        }
        for field in [
            "version",
            "rootUri",
            "buildRootUris",
            "cachedOnly",
            "readOnly",
            "partial",
            "fullSynchronizationKnown",
        ] {
            let mut invalid = response.clone();
            invalid.as_object_mut().expect("object").remove(field);
            assert!(query.validate_response(invalid).is_err());
        }
        assert!(query.validate_response(json!(null)).is_err());
        query
            .validate_current(&snapshot, snapshot.id())
            .expect("unchanged");
        fs.insert_tree(root, json!({"new":{"settings.gradle.kts":"unread"}}))
            .await;
        cx.run_until_parked();
        let changed = default_root_snapshot(&project, cx);
        assert!(query.validate_current(&changed, changed.id()).is_err());
        fs.remove_file(&root.join("new/settings.gradle.kts"), Default::default())
            .await
            .expect("remove candidate");
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        query
            .validate_current(&current, current.id())
            .expect("original candidate identities");
        fs.insert_tree(root, json!({"child":{"build.gradle":"replacement"}}))
            .await;
        cx.run_until_parked();
        let changed = default_root_snapshot(&project, cx);
        assert!(query.validate_current(&changed, changed.id()).is_err());
    }

    #[gpui::test]
    async fn gradle_default_root_query_bounds_files_without_collapsing_descriptor_kinds(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = if cfg!(windows) {
            "C:/gradle-default-budget"
        } else {
            "/gradle-default-budget"
        };
        let tree = (0..32).map(|index| (format!("module{index:03}"),json!({
            "build.gradle":"unread", "build.gradle.kts":"unread", "settings.gradle":"unread", "settings.gradle.kts":"unread"
        }))).collect::<serde_json::Map<_,_>>();
        let (fs, project) = default_root_fixture(root, Value::Object(tree), cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let query =
            GradleDefaultImportRootsQuery::new(&snapshot, snapshot.id()).expect("128 exact files");
        assert_eq!(query.inputs.len(), 128);
        assert_eq!(query.candidates.len(), 32);
        fs.insert_tree(root, json!({"overflow":{"build.gradle":"unread"}}))
            .await;
        cx.run_until_parked();
        let snapshot = default_root_snapshot(&project, cx);
        let error = GradleDefaultImportRootsQuery::new(&snapshot, snapshot.id())
            .expect_err("129 files")
            .to_string();
        assert!(error.contains("candidate bound") && !error.contains("gradle-default-budget"));
    }

    #[gpui::test]
    async fn gradle_default_root_query_uses_advertised_command_and_cancels(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::AppContext as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let root = if cfg!(windows) {
            Path::new("C:/gradle-default-query")
        } else {
            Path::new("/gradle-default-query")
        };
        let (_, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({"build.gradle":"unread"}),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        for supported in [false, true] {
            let selection =
                GradleDefaultImportRootsQuery::new(&snapshot, snapshot.id()).expect("query");
            let receipt = json!({"version":1,"rootUri":selected_import_uri(root,true).expect("root URI"),
                "buildRootUris":[selected_import_uri(root,true).expect("build URI")],
                "cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false});
            let expected = selection.parameters.clone();
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let cancelled = Arc::new(AtomicUsize::new(0));
            let (entered_sender, entered_receiver) = async_channel::unbounded();
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![if supported {
                            expected.command.clone()
                        } else {
                            "java.project.changeImportedProjects".into()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                let mode = mode.clone();
                let cancelled = cancelled.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let receipt = receipt.clone();
                    let current_mode = mode.load(Ordering::SeqCst);
                    let cancelled = cancelled.clone();
                    let entered_sender = entered_sender.clone();
                    async move {
                        match current_mode {
                            0 => Ok(Some(receipt)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!(
                                "private host credential at /groovy-default-query failed"
                            )),
                            3 => Ok(Some(json!({"private-field":"private-value"}))),
                            _ => {
                                let _cancel_guard = util::defer(move || {
                                    cancelled.fetch_add(1, Ordering::SeqCst);
                                });
                                entered_sender.send(()).await.expect("entered observer");
                                futures::future::pending().await
                            }
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized fake");
            if !supported {
                let message = selection
                    .query(server, Duration::from_secs(10))
                    .await
                    .expect_err("no upstream fallback")
                    .to_string();
                assert!(message.contains("extension setup"));
                assert_eq!(requests.load(Ordering::SeqCst), 0);
                continue;
            }
            selection
                .query(server.clone(), Duration::from_secs(10))
                .await
                .expect("captured phase");
            for current_mode in 1..=4 {
                mode.store(current_mode, Ordering::SeqCst);
                let message = selection
                    .query(server.clone(), Duration::from_secs(1))
                    .await
                    .expect_err("failed phase")
                    .to_string();
                assert!(
                    message.contains("refresh")
                        && !message.contains("private")
                        && !message.contains("groovy-default-query")
                );
            }
            entered_receiver
                .recv()
                .await
                .expect("timeout handler entered");
            cx.run_until_parked();
            assert_eq!(cancelled.load(Ordering::SeqCst), 1);
            let task = cx.background_spawn({
                let selection = selection.clone();
                let server = server.clone();
                async move { selection.query(server, Duration::MAX).await }
            });
            entered_receiver.recv().await.expect("drop handler entered");
            drop(task);
            cx.run_until_parked();
            assert_eq!(cancelled.load(Ordering::SeqCst), 2);
            mode.store(0, Ordering::SeqCst);
            selection
                .query(server, Duration::from_secs(10))
                .await
                .expect("recovery after timeout/drop");
            assert_eq!(requests.load(Ordering::SeqCst), 7);
        }
    }

    #[test]
    fn maven_default_root_query_empty_inputs_keep_worktree_identity() {
        let root = selected_import_fixture().root;
        let snapshot = |worktree_id| {
            Snapshot::new(
                worktree_id,
                Arc::from(RelPath::from_unix_str("fixture").expect("root name")),
                Arc::from(root.as_path()),
                PathStyle::local(),
            )
        };
        let original = snapshot(WorktreeId::from_usize(1));
        let replacement = snapshot(WorktreeId::from_usize(2));
        let query =
            MavenDefaultImportRootsQuery::new(&original, original.id()).expect("empty query");
        assert!(query.candidates.is_empty());
        query
            .validate_current(&original, original.id())
            .expect("same scope");
        assert!(
            query
                .validate_current(&replacement, replacement.id())
                .is_err()
        );
        assert!(MavenDefaultImportRootsQuery::new(&original, replacement.id()).is_err());
    }

    #[gpui::test]
    async fn maven_default_root_query_bounds_responses_and_current_candidates(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let root = if cfg!(windows) {
            Path::new("C:/maven-default-query")
        } else {
            Path::new("/maven-default-query")
        };
        let (fs, project) = default_root_fixture(root.to_str().expect("root"), json!({
            "pom.xml":"<project/>", "child":{"pom.xml":"<project/>"},
            ".env":{"pom.xml":"private"}, "ignored":{"pom.xml":"ignored"}, ".gitignore":"ignored/\n"
        }), cx).await;
        let snapshot = default_root_snapshot(&project, cx);
        let query =
            MavenDefaultImportRootsQuery::new(&snapshot, snapshot.id()).expect("current query");
        let root_uri = selected_import_uri(root, true)
            .expect("root URI")
            .to_string();
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("fixture URI");
        assert_eq!(query.candidates.len(), 2);
        assert_eq!(query.parameters.command, "groovy.maven.defaultImportRoots");
        assert_eq!(
            query.parameters.arguments,
            vec![json!({"rootUri":uri(root,true),
            "candidatePomUris":[uri(&root.join("child/pom.xml"),false),uri(&root.join("pom.xml"),false)],"cachedOnly":true})]
        );
        let response = json!({"version":1,"rootUri":uri(root,true),"rootPomUris":[uri(&root.join("pom.xml"),false)],
            "retainedPomUris":[],"cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false});
        let roots = query.validate_response(response.clone()).expect("subset");
        assert_eq!(roots.poms, BTreeSet::from([root.join("pom.xml")]));
        assert!(roots.retained_poms.is_empty());
        let mut empty = response.clone();
        empty["rootPomUris"] = json!([]);
        assert!(
            query
                .validate_response(empty)
                .expect("known empty")
                .poms
                .is_empty()
        );
        let mut linked = response.clone();
        linked["retainedPomUris"] = linked["rootPomUris"].clone();
        assert_eq!(
            query
                .validate_response(linked)
                .expect("retained")
                .retained_poms,
            roots.poms
        );
        for (field, value) in [
            ("version", json!(2)),
            ("version", json!("1")),
            ("rootUri", json!("file:///foreign/")),
            ("rootUri", json!("x".repeat(4097))),
            ("cachedOnly", json!(false)),
            ("readOnly", json!(false)),
            ("partial", json!(true)),
            ("fullSynchronizationKnown", json!(true)),
            ("privateCanary", json!(true)),
            (
                "rootPomUris",
                json!([
                    uri(&root.join("pom.xml"), false),
                    uri(&root.join("pom.xml"), false)
                ]),
            ),
            ("rootPomUris", json!(["file:///foreign/pom.xml"])),
            ("rootPomUris", json!([format!("{root_uri}missing/pom.xml")])),
            ("rootPomUris", json!([format!("{root_uri}.env/pom.xml")])),
            ("rootPomUris", json!([format!("{root_uri}ignored/pom.xml")])),
            (
                "rootPomUris",
                json!([format!("{root_uri}child/../pom.xml")]),
            ),
            (
                "rootPomUris",
                json!([format!("{root_uri}pom.xml?privateCanary=true")]),
            ),
            ("rootPomUris", json!(["x".repeat(4097)])),
            ("rootPomUris", json!([17])),
            (
                "rootPomUris",
                json!(vec![uri(&root.join("pom.xml"), false); 33]),
            ),
            (
                "retainedPomUris",
                json!([uri(&root.join("child/pom.xml"), false)]),
            ),
            (
                "retainedPomUris",
                json!([
                    uri(&root.join("pom.xml"), false),
                    uri(&root.join("pom.xml"), false)
                ]),
            ),
        ] {
            let mut changed = response.clone();
            changed[field] = value;
            let error = format!(
                "{:#}",
                query
                    .validate_response(changed)
                    .expect_err("invalid response")
            );
            assert!(
                error.contains("refresh")
                    && !error.contains("private")
                    && !error.contains("foreign")
                    && !error.contains("maven-default-query")
            );
        }
        for field in [
            "version",
            "rootUri",
            "rootPomUris",
            "retainedPomUris",
            "cachedOnly",
            "readOnly",
            "partial",
            "fullSynchronizationKnown",
        ] {
            let mut changed = response.clone();
            changed.as_object_mut().expect("object").remove(field);
            assert!(query.validate_response(changed).is_err());
        }
        query
            .validate_current(&snapshot, snapshot.id())
            .expect("unchanged");
        fs.insert_tree(root, json!({"new":{"pom.xml":"<project/>"}}))
            .await;
        cx.run_until_parked();
        let changed = default_root_snapshot(&project, cx);
        assert!(query.validate_current(&changed, changed.id()).is_err());
        fs.remove_file(&root.join("new/pom.xml"), Default::default())
            .await
            .expect("remove candidate");
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        query
            .validate_current(&current, current.id())
            .expect("identity recovery");
        fs.remove_file(&root.join("child/pom.xml"), Default::default())
            .await
            .expect("remove original candidate");
        cx.run_until_parked();
        let current = default_root_snapshot(&project, cx);
        assert!(query.validate_current(&current, current.id()).is_err());
    }

    #[gpui::test]
    async fn maven_default_root_query_uses_advertised_read_only_command(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let root = if cfg!(windows) {
            Path::new("C:/maven-default-query")
        } else {
            Path::new("/maven-default-query")
        };
        let (_, project) = default_root_fixture(
            root.to_str().expect("root"),
            json!({"pom.xml":"<project/>"}),
            cx,
        )
        .await;
        let snapshot = default_root_snapshot(&project, cx);
        let query = MavenDefaultImportRootsQuery::new(&snapshot, snapshot.id()).expect("query");
        for supported in [false, true] {
            let response = json!({"version":1,"rootUri":selected_import_uri(root,true).expect("root URI"),"rootPomUris":[selected_import_uri(&root.join("pom.xml"),false).expect("POM URI")],
                "retainedPomUris":[],"cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false});
            let requests = Arc::new(AtomicUsize::new(0));
            let mode = Arc::new(AtomicUsize::new(0));
            let expected = query.parameters.clone();
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![if supported {
                            expected.command.clone()
                        } else {
                            "groovy.maven.importReactor".into()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                let mode = mode.clone();
                move |parameters, _| {
                    assert_eq!(parameters, expected);
                    requests.fetch_add(1, Ordering::SeqCst);
                    let mode = mode.load(Ordering::SeqCst);
                    let response = response.clone();
                    async move {
                        match mode {
                            0 => Ok(Some(response)),
                            1 => Ok(None),
                            2 => Err(anyhow::anyhow!("privateCanary /secret/path token=value")),
                            _ => Ok(Some(json!({"privateCanary":true}))),
                        }
                    }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized");
            if supported {
                query
                    .query(server.clone(), Duration::from_secs(10))
                    .await
                    .expect("read");
                for failure in 1..=3 {
                    mode.store(failure, Ordering::SeqCst);
                    let message = format!(
                        "{:#}",
                        query
                            .query(server.clone(), Duration::from_secs(10))
                            .await
                            .expect_err("failure")
                    );
                    assert!(
                        !message.contains("private")
                            && !message.contains("secret")
                            && !message.contains("token")
                    );
                }
                assert_eq!(requests.load(Ordering::SeqCst), 4);
            } else {
                assert!(
                    query
                        .query(server, Duration::from_secs(10))
                        .await
                        .expect_err("old provider")
                        .to_string()
                        .contains("extension setup")
                );
                assert_eq!(requests.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn maven_reactor_selection_bounds_and_confines_root_and_exclusions() {
        let root = selected_import_fixture().root;
        let pom = root.join("pom.xml");
        let excluded = root.join("ignored");
        let selection = MavenReactorImport::new(&root, &pom, std::slice::from_ref(&excluded))
            .expect("reactor selection");
        assert_eq!(selection.parameters.command, "groovy.maven.importReactor");
        assert_eq!(
            selection.parameters.arguments,
            vec![json!({
                "rootUri": selected_import_uri(&root, true).expect("root URI"),
                "pomUri": selected_import_uri(&pom, false).expect("POM URI"),
                "excludedProjectUris": [selected_import_uri(&excluded, true).expect("exclusion URI")], "offline": true,
            })]
        );
        for candidate in [
            root.join("build.gradle"),
            root.join("../foreign/pom.xml"),
            root.join("private\ncanary/pom.xml"),
            root.join("./pom.xml"),
            PathBuf::from("relative/pom.xml"),
        ] {
            let message = MavenReactorImport::new(&root, &candidate, &[])
                .expect_err("invalid root POM")
                .to_string();
            assert!(!message.contains("canary") && !message.contains("groovy-selection"));
        }
        for exclusions in [
            vec![root.clone()],
            vec![excluded.clone(), excluded.clone()],
            vec![excluded; 129],
            vec![root.join("../foreign")],
            vec![root.join("private\ncanary")],
            vec![PathBuf::from("relative")],
        ] {
            let message = MavenReactorImport::new(&root, &pom, &exclusions)
                .expect_err("invalid exclusions")
                .to_string();
            assert!(!message.contains("canary") && !message.contains("groovy-selection"));
        }
    }

    #[test]
    fn maven_reactor_receipts_require_exact_root_exclusions_and_bounded_current_identities() {
        let root = selected_import_fixture().root;
        let pom = root.join("pom.xml");
        let excluded = root.join("ignored");
        let selection =
            MavenReactorImport::new(&root, &pom, std::slice::from_ref(&excluded)).expect("reactor");
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("fixture URI");
        let poms = vec![
            pom.clone(),
            root.join("nested/pom.xml"),
            root.join("nested/leaf/pom.xml"),
        ];
        let receipt = json!({"version":1,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":poms.iter().map(|pom| uri(pom,false)).collect::<Vec<_>>(),
            "excludedProjectUris":[uri(&excluded,true)],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        assert_eq!(
            selection
                .validate_receipt(receipt.clone())
                .expect("bounded receipt"),
            MavenReactorPhaseFacts {
                imported_poms: poms,
                ignored_poms: Vec::new()
            }
        );
        let mut invalid = vec![json!(null), json!([]), json!({})];
        for (field, value) in [
            ("version", json!(2)),
            ("selectedImportPhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("rootUri", json!(uri(&root.join("foreign"), true))),
            (
                "reactorPomUri",
                json!(uri(&root.join("nested/pom.xml"), false)),
            ),
            ("importedPomUris", json!([])),
            (
                "importedPomUris",
                json!([uri(&root.join("nested/pom.xml"), false)]),
            ),
            (
                "importedPomUris",
                json!([uri(&pom, false), uri(&pom, false)]),
            ),
            (
                "importedPomUris",
                json!([uri(&pom, false), uri(&excluded.join("pom.xml"), false)]),
            ),
            (
                "importedPomUris",
                json!([uri(&pom, false), uri(&root.join("Main.java"), false)]),
            ),
            (
                "importedPomUris",
                json!([uri(&pom, false), "file:///private/canary/pom.xml"]),
            ),
            (
                "importedPomUris",
                json!([uri(&pom, false), "https://private.invalid/canary/pom.xml"]),
            ),
            (
                "importedPomUris",
                json!([
                    uri(&pom, false),
                    format!("{}?private=canary", uri(&pom, false))
                ]),
            ),
            ("importedPomUris", json!(vec![uri(&pom, false); 129])),
            ("importedPomUris", json!(["x".repeat(4097)])),
            ("excludedProjectUris", json!([])),
            (
                "excludedProjectUris",
                json!([uri(&excluded, true), uri(&excluded, true)]),
            ),
            (
                "excludedProjectUris",
                json!([uri(&root.join("other"), true)]),
            ),
            ("privateCanary", json!(true)),
        ] {
            let mut changed = receipt.clone();
            changed[field] = value;
            invalid.push(changed);
        }
        for field in [
            "version",
            "rootUri",
            "reactorPomUri",
            "importedPomUris",
            "excludedProjectUris",
            "selectedImportPhaseComplete",
            "fullSynchronizationKnown",
        ] {
            let mut changed = receipt.clone();
            changed
                .as_object_mut()
                .expect("receipt object")
                .remove(field);
            invalid.push(changed);
        }
        for value in invalid {
            let message = selection
                .validate_receipt(value)
                .expect_err("invalid phase receipt")
                .to_string();
            assert!(message.contains("refresh"));
            assert!(
                !message.contains("canary")
                    && !message.contains("private")
                    && !message.contains("groovy-selection")
            );
        }
    }

    #[test]
    fn maven_masked_reactor_preserves_classification_and_parent_child_independence() {
        let root = selected_import_fixture().root;
        let pom = root.join("pom.xml");
        let parent = root.join("nested/pom.xml");
        let child = root.join("nested/leaf/pom.xml");
        let patterns = ["*/nested/pom.xml".to_owned()];
        let selection = MavenReactorImport::new_with_masks(&root, &pom, &[], Some(&patterns))
            .expect("Bounded explicit mask request");
        assert_eq!(
            selection.parameters.command,
            "groovy.maven.importReactorWithMasks"
        );
        assert_eq!(selection.parameters.arguments.len(), 1);
        let request = &selection.parameters.arguments[0];
        assert_eq!(request.as_object().expect("Request object").len(), 5);
        assert_eq!(
            request["pomMasks"],
            json!({"version":1,"syntax":"idea-mask-v1",
            "pathFormat":"portable-absolute-v1","patterns":patterns})
        );
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("Fixture URI");
        let receipt = json!({"version":2,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":[uri(&pom,false),uri(&child,false)],"ignoredPomUris":[uri(&parent,false)],
            "excludedProjectUris":[],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        assert_eq!(
            selection
                .validate_receipt(receipt)
                .expect("Classified receipt"),
            MavenReactorPhaseFacts {
                imported_poms: vec![pom.clone(), child.clone()],
                ignored_poms: vec![parent.clone()],
            }
        );
        let all = MavenReactorImport::new_with_masks(&root, &pom, &[], Some(&["*".into()]))
            .expect("All ignored reactor request");
        let mut receipt = json!({"version":2,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":[],"ignoredPomUris":[uri(&pom,false),uri(&parent,false),uri(&child,false)],
            "excludedProjectUris":[],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        assert_eq!(
            all.validate_receipt(receipt.clone())
                .expect("Known empty import"),
            MavenReactorPhaseFacts {
                imported_poms: vec![],
                ignored_poms: vec![pom.clone(), parent, child],
            }
        );
        let cleared = MavenReactorImport::new_with_masks(&root, &pom, &[], Some(&[]))
            .expect("Explicit empty contract");
        receipt["importedPomUris"] = json!([uri(&pom, false)]);
        receipt["ignoredPomUris"] = json!([]);
        assert_eq!(
            cleared
                .validate_receipt(receipt)
                .expect("Empty contract imports root"),
            MavenReactorPhaseFacts {
                imported_poms: vec![pom],
                ignored_poms: vec![],
            }
        );
    }

    #[test]
    fn maven_masked_reactor_rejects_contradictory_or_unbound_receipts() {
        let root = selected_import_fixture().root;
        let pom = root.join("pom.xml");
        let ignored = root.join("nested/pom.xml");
        let excluded = root.join("excluded");
        let selection = MavenReactorImport::new_with_masks(
            &root,
            &pom,
            std::slice::from_ref(&excluded),
            Some(&["*/nested/pom.xml".into()]),
        )
        .expect("Mask request");
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("Fixture URI");
        let receipt = json!({"version":2,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":[uri(&pom,false)],"ignoredPomUris":[uri(&ignored,false)],
            "excludedProjectUris":[uri(&excluded,true)],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        for (field, value) in [
            ("version", json!(1)),
            ("version", json!(3)),
            ("version", json!("2")),
            ("rootUri", json!(uri(&root.join("foreign"), true))),
            ("reactorPomUri", json!(uri(&ignored, false))),
            ("selectedImportPhaseComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("importedPomUris", json!([])),
            (
                "importedPomUris",
                json!([uri(&pom, false), uri(&ignored, false)]),
            ),
            (
                "ignoredPomUris",
                json!([uri(&ignored, false), uri(&ignored, false)]),
            ),
            ("ignoredPomUris", json!([uri(&pom, false)])),
            (
                "ignoredPomUris",
                json!([uri(&root.join("other/pom.xml"), false)]),
            ),
            (
                "ignoredPomUris",
                json!([uri(&excluded.join("nested/pom.xml"), false)]),
            ),
            (
                "ignoredPomUris",
                json!([uri(&root.join("nested/build.gradle"), false)]),
            ),
            (
                "ignoredPomUris",
                json!(["file:///private-canary/nested/pom.xml"]),
            ),
            (
                "ignoredPomUris",
                json!(["https://private-canary.invalid/pom.xml"]),
            ),
            (
                "ignoredPomUris",
                json!([format!("{}?private-canary=1", uri(&ignored, false))]),
            ),
            ("ignoredPomUris", json!(["x".repeat(4097)])),
            ("ignoredPomUris", json!(null)),
            ("ignoredPomUris", json!("private-canary")),
            ("excludedProjectUris", json!([])),
            ("extra", json!("private-canary")),
        ] {
            let mut changed = receipt.clone();
            changed[field] = value;
            let error = selection
                .validate_receipt(changed)
                .expect_err("Invalid classified receipt")
                .to_string();
            assert!(error.contains("refresh"));
            assert!(!error.contains("private-canary") && !error.contains("groovy-selection"));
        }
        for field in [
            "version",
            "rootUri",
            "reactorPomUri",
            "importedPomUris",
            "ignoredPomUris",
            "excludedProjectUris",
            "selectedImportPhaseComplete",
            "fullSynchronizationKnown",
        ] {
            let mut changed = receipt.clone();
            changed.as_object_mut().expect("Receipt").remove(field);
            assert!(
                selection.validate_receipt(changed).is_err(),
                "Missing {field}"
            );
        }
        let legacy = MavenReactorImport::new(&root, &pom, &[excluded]).expect("Legacy request");
        assert!(
            legacy.validate_receipt(receipt).is_err(),
            "A legacy request cannot admit a masked receipt"
        );
    }

    #[test]
    fn maven_masked_reactor_bounds_the_combined_identity_set() {
        let root = selected_import_fixture().root;
        let pom = root.join("pom.xml");
        let uri =
            |path: &Path, directory| selected_import_uri(path, directory).expect("Fixture URI");
        let selection = MavenReactorImport::new_with_masks(
            &root,
            &pom,
            &[],
            Some(&["*/ignored-?/pom.xml".into()]),
        )
        .expect("Mask request");
        let mut imported = vec![uri(&pom, false)];
        imported.extend(
            (0..125).map(|index| uri(&root.join(format!("module-{index}/pom.xml")), false)),
        );
        let ignored = vec![
            uri(&root.join("ignored-0/pom.xml"), false),
            uri(&root.join("ignored-1/pom.xml"), false),
        ];
        let mut receipt = json!({"version":2,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":imported,"ignoredPomUris":ignored,"excludedProjectUris":[],
            "selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        let facts = selection
            .validate_receipt(receipt.clone())
            .expect("Exactly 128 total identities");
        assert_eq!(
            (facts.imported_poms.len(), facts.ignored_poms.len()),
            (126, 2)
        );
        imported.push(uri(&root.join("overflow/pom.xml"), false));
        receipt["importedPomUris"] = json!(imported);
        assert!(
            selection.validate_receipt(receipt).is_err(),
            "Each list is individually bounded but their union exceeds 128"
        );
        let all = MavenReactorImport::new_with_masks(&root, &pom, &[], Some(&["*".into()]))
            .expect("All masks");
        let mut ignored = vec![uri(&pom, false)];
        ignored.extend(
            (0..127).map(|index| uri(&root.join(format!("module-{index}/pom.xml")), false)),
        );
        let mut receipt = json!({"version":2,"rootUri":uri(&root,true),"reactorPomUri":uri(&pom,false),
            "importedPomUris":[],"ignoredPomUris":ignored,"excludedProjectUris":[],
            "selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
        assert_eq!(
            all.validate_receipt(receipt.clone())
                .expect("128 ignored identities")
                .ignored_poms
                .len(),
            128
        );
        ignored.push(uri(&root.join("overflow/pom.xml"), false));
        receipt["ignoredPomUris"] = json!(ignored);
        assert!(all.validate_receipt(receipt).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn maven_masked_reactor_refuses_unknown_host_spelling_before_dispatch() {
        use std::os::unix::ffi::OsStrExt as _;
        let root = selected_import_fixture().root;
        let pom = root
            .join(std::ffi::OsStr::from_bytes(b"private-canary-\xff"))
            .join("pom.xml");
        let error = MavenReactorImport::new_with_masks(&root, &pom, &[], Some(&["*".into()]))
            .expect_err("Unknown UTF-8 host identity")
            .to_string();
        assert!(!error.contains("private-canary") && !error.contains("groovy-selection"));
        for patterns in [
            vec![String::new()],
            vec!["*".into(), "*".into()],
            vec!["private-canary\n".into()],
        ] {
            let error = MavenReactorImport::new_with_masks(
                &root,
                &root.join("pom.xml"),
                &[],
                Some(&patterns),
            )
            .expect_err("Malformed mask request")
            .to_string();
            assert!(!error.contains("private-canary") && error.contains("refresh"));
        }
    }

    #[gpui::test]
    async fn selected_maven_import_uses_only_the_advertised_offline_phase_command(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        for (reactor, masked) in [(false, false), (true, false), (true, true)] {
            for supported in [false, true] {
                let selection = selected_import_fixture();
                let patterns = ["*".to_owned()];
                let reactor_selection = MavenReactorImport::new_with_masks(
                    &selection.root,
                    &selection.root.join("pom.xml"),
                    &[],
                    masked.then_some(patterns.as_slice()),
                )
                .expect("reactor selection");
                let expected = if reactor {
                    reactor_selection.parameters.clone()
                } else {
                    selection.parameters.clone()
                };
                let receipt = if masked {
                    json!({"version":2,"rootUri":expected.arguments[0]["rootUri"],"reactorPomUri":expected.arguments[0]["pomUri"],
                        "importedPomUris":[],"ignoredPomUris":[expected.arguments[0]["pomUri"]],"excludedProjectUris":[],
                        "selectedImportPhaseComplete":true,"fullSynchronizationKnown":false})
                } else if reactor {
                    json!({"version":1,"rootUri":expected.arguments[0]["rootUri"],"reactorPomUri":expected.arguments[0]["pomUri"],
                "importedPomUris":[expected.arguments[0]["pomUri"]],"excludedProjectUris":[],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false})
                } else {
                    selected_import_receipt(&selection)
                };
                let requests = Arc::new(AtomicUsize::new(0));
                let mode = Arc::new(AtomicUsize::new(0));
                let (server, fake) = lsp::FakeLanguageServer::new(
                    LanguageServerId(0),
                    LanguageServerBinary {
                        path: "unused-fake-jdtls".into(),
                        arguments: vec![],
                        env: None,
                    },
                    "jdtls".into(),
                    lsp::ServerCapabilities {
                        execute_command_provider: Some(lsp::ExecuteCommandOptions {
                            commands: vec![if supported {
                                expected.command.clone()
                            } else if masked {
                                "groovy.maven.importReactor".into()
                            } else {
                                "java.project.changeImportedProjects".into()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    &mut cx.to_async(),
                );
                fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                    let requests = requests.clone();
                    let mode = mode.clone();
                    move |parameters, _| {
                        assert_eq!(parameters, expected);
                        requests.fetch_add(1, Ordering::SeqCst);
                        let receipt = receipt.clone();
                        let mode = mode.load(Ordering::SeqCst);
                        async move {
                            match mode {
                                0 => Ok(Some(receipt)),
                                1 => Ok(None),
                                2 => Err(anyhow::anyhow!(
                                    "private host credential at /groovy-selection failed"
                                )),
                                _ => Ok(Some(json!({"private-field": "private-value"}))),
                            }
                        }
                    }
                });
                let server = cx
                    .update(|cx| {
                        let parameters = server.default_initialize_params(false, false, cx);
                        server.initialize(
                            parameters,
                            Arc::new(lsp::DidChangeConfigurationParams {
                                settings: json!({}),
                            }),
                            Duration::from_secs(10),
                            cx,
                        )
                    })
                    .await
                    .expect("fake initialized JDTLS");
                if supported {
                    if reactor {
                        reactor_selection
                            .clone()
                            .execute_phase(server.clone(), Duration::from_secs(10))
                            .await
                            .map(|_| ())
                    } else {
                        selection
                            .clone()
                            .execute_phase(server.clone(), Duration::from_secs(10))
                            .await
                    }
                    .expect("matching phase receipt");
                    for failure_mode in 1..=3 {
                        mode.store(failure_mode, Ordering::SeqCst);
                        let error = if reactor {
                            reactor_selection
                                .clone()
                                .execute_phase(server.clone(), Duration::from_secs(10))
                                .await
                                .map(|_| ())
                        } else {
                            selection
                                .clone()
                                .execute_phase(server.clone(), Duration::from_secs(10))
                                .await
                        }
                        .expect_err("failed phase");
                        let message = format!("{error:#}");
                        assert!(message.contains("refresh"));
                        assert!(!message.contains("private"));
                        assert!(!message.contains("groovy-selection"));
                    }
                    assert_eq!(requests.load(Ordering::SeqCst), 4);
                } else {
                    if reactor {
                        reactor_selection
                            .execute_phase(server, Duration::from_secs(10))
                            .await
                            .map(|_| ())
                    } else {
                        selection
                            .execute_phase(server, Duration::from_secs(10))
                            .await
                    }
                    .expect_err("no unsafe upstream fallback");
                    assert_eq!(requests.load(Ordering::SeqCst), 0);
                }
            }
        }
    }

    #[gpui::test]
    async fn remote_build_facts_preserve_host_paths_and_reject_invalid_metadata(
        cx: &mut gpui::TestAppContext,
    ) {
        use proto::Message as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        let tree = json!({"application": {"pom.xml": "<project/>", "src": {"Main.java": "class Main {}"},
            "generated": {"Generated.java": "class Generated {}"}, ".env": "private"},
            "library": {"pom.xml": "<project/>"}});
        fs.insert_tree("/host-build-facts", tree.clone()).await;
        fs.insert_tree("/desktop-build-facts", tree).await;
        let project = crate::Project::test(
            fs,
            [
                Path::new("/host-build-facts"),
                Path::new("/desktop-build-facts"),
            ],
            cx,
        )
        .await;
        let snapshots = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .map(|worktree| worktree.read(cx).snapshot())
                .collect::<Vec<_>>()
        });
        let host = snapshots
            .iter()
            .find(|snapshot| snapshot.abs_path().as_ref() == Path::new("/host-build-facts"))
            .expect("host snapshot");
        let desktop = snapshots
            .iter()
            .find(|snapshot| snapshot.abs_path().as_ref() == Path::new("/desktop-build-facts"))
            .expect("desktop snapshot");
        let mut raw = Vec::new();
        merge_gradle_model(&mut raw, json!({"version":2,"rootUri":"file:///host-build-facts/","sourceSetsKnown":false,
            "projects":[{"buildRootUri":"file:///host-build-facts/","moduleUri":"file:///host-build-facts/application/",
                "projectPath":":application","name":"application","tasks":[{"path":":application:test","name":"test",
                    "group":"verification","description":"Run tests","optionsKnown":false}],
                "detailsJson":json!({"javaSourceSetsKnown":true,"configurationsKnown":true,"dependencyGraphKnown":false,
                    "configurations":[{"name":"implementation","description":null,"resolvable":false,"consumable":false,"extendsFrom":[]},
                        {"name":"runtimeClasspath","description":"Runtime declaration","resolvable":true,"consumable":false,"extendsFrom":["implementation"]}],
                    "sourceSets":[{"name":"main","generatednessKnown":false,
                    "roots":["file:///host-build-facts/application/src","file:///host-build-facts/application/generated",
                        "file:///host-build-facts/application/.env/missing","file:///host-build-facts/application/missing"]}],
                    "tasks":[{"path":":application:test","optionsComplete":true,"options":[{"name":"tests",
                        "description":"Selected test filter","declaredType":"java.util.List<java.lang.String>"}]}]}).to_string()}]}),
            Path::new("/host-build-facts")).expect("host Gradle facts");
        let parameter_declarations = json!([{"name":"skipTests","alias":null,"declaredType":"boolean","required":false,"editable":true,"description":"Skip tests"}]);
        merge_maven_model(&mut raw, json!({"version":3,"rootUri":"file:///host-build-facts/","cachedOnly":true,
            "reactorOrderKnown":true,"reactorPomUris":["file:///host-build-facts/application/pom.xml"],"unavailablePomUris":[],
            "lifecycles":[{"id":"default","phases":["compile","test","verify"]}],
            "projects":[{"pomUri":"file:///host-build-facts/application/pom.xml","groupId":"evidence","artifactId":"application",
                "version":"1.0","packaging":"jar","moduleUris":["file:///host-build-facts/library/",
                    "file:///host-build-facts/application/.env/missing","file:///host-build-facts/application/missing"],
                "profiles":{"parity":true},"inheritedProfileDefinitionsKnown":true,"profileDefinitions":[
                    {"id":"parity","sourceGroupId":"evidence","sourceArtifactId":"application","sourceVersion":"1.0",
                        "sourcePomUri":"file:///host-build-facts/application/pom.xml","inherited":false,"activeAtSource":true},
                    {"id":"hidden","sourceGroupId":"evidence","sourceArtifactId":"parent","sourceVersion":"1.0",
                        "sourcePomUri":"file:///host-build-facts/application/.env/missing.xml","inherited":true,"activeAtSource":false}],
                "plugins":[{"groupId":"org.apache.maven.plugins","artifactId":"maven-surefire-plugin","version":"3.5.2",
                    "managedOnly":false,"goalsKnown":true,"goalPrefix":"surefire","goals":[{"name":"test","phase":"test","description":"Run tests","parametersKnown":true,"parameters":parameter_declarations}]}]}]}),
            Path::new("/host-build-facts")).expect("host Maven facts");
        let (projects, partial) = projects_from_provider(
            raw,
            Path::new("/host-build-facts"),
            host.id(),
            PathStyle::Unix,
            host,
            true,
        );
        assert!(partial);
        assert_eq!(projects.len(), 1);
        for constants in [
            Vec::new(),
            vec!["ALPHA".into(), "ZULU".into()],
            (0..128).map(|index| format!("MEMBER_{index}")).collect(),
        ] {
            let mut declared_projects = projects.clone();
            let option = declared_projects
                .first_mut()
                .expect("project")
                .gradle
                .as_mut()
                .expect("Gradle facts")
                .tasks
                .first_mut()
                .expect("task")
                .options
                .first_mut()
                .expect("option");
            option.declared_type = "fixture.Mode".into();
            option.enum_constants = Some(constants);
            let declared_wire = declared_projects
                .iter()
                .map(project_to_proto)
                .collect::<Vec<_>>();
            let encoded = proto::JdtlsProjectSnapshotResponse {
                protocol_version: JDTLS_SNAPSHOT_PROTOCOL_VERSION,
                projects: declared_wire.clone(),
                ..Default::default()
            }
            .encode_to_vec();
            let decoded = proto::JdtlsProjectSnapshotResponse::decode(encoded.as_slice())
                .expect("typed enum declaration wire");
            assert_eq!(
                projects_from_proto_for_version(
                    decoded.projects,
                    host.id(),
                    host,
                    JDTLS_SNAPSHOT_PROTOCOL_VERSION
                )
                .expect("current enum transport"),
                declared_projects
            );
            assert!(
                projects_from_proto_for_version(declared_wire, host.id(), host, 5).is_err(),
                "earlier protocol must refuse newer declaration fields"
            );
            let older_wire = declared_projects
                .iter()
                .map(|project| project_to_proto_for_version(project, 5))
                .collect::<Result<Vec<_>>>()
                .expect("earlier projection");
            let downgraded = projects_from_proto_for_version(older_wire, host.id(), host, 5)
                .expect("old enum transport downgrade");
            declared_projects
                .first_mut()
                .expect("project")
                .gradle
                .as_mut()
                .expect("Gradle facts")
                .tasks
                .first_mut()
                .expect("task")
                .options
                .first_mut()
                .expect("option")
                .enum_constants = None;
            assert_eq!(downgraded, declared_projects);
        }
        for (known, constants) in [
            (None, vec!["ALPHA".into()]),
            (Some(false), vec!["ALPHA".into()]),
            (Some(true), vec!["ALPHA".into(), "ALPHA".into()]),
            (Some(true), vec!["/private/enum-canary".into()]),
            (Some(true), vec!["PRIVATE\nCANARY".into()]),
            (
                Some(true),
                (0..129).map(|index| format!("MEMBER_{index}")).collect(),
            ),
        ] {
            let mut invalid = projects.iter().map(project_to_proto).collect::<Vec<_>>();
            let option = invalid
                .first_mut()
                .expect("project")
                .gradle
                .as_mut()
                .expect("Gradle")
                .tasks
                .first_mut()
                .expect("task")
                .options
                .first_mut()
                .expect("option");
            option.enum_constants_known = known;
            option.enum_constants = constants;
            let error = projects_from_proto_for_version(
                invalid,
                host.id(),
                host,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION,
            )
            .expect_err("malformed enum transport")
            .to_string();
            assert!(error.contains("Remote Gradle enum declarations"));
            assert!(
                !error.contains("canary")
                    && !error.contains("CANARY")
                    && !error.contains("/private")
            );
        }
        let wire = projects.iter().map(project_to_proto).collect::<Vec<_>>();
        let encoded = proto::JdtlsProjectSnapshotResponse {
            protocol_version: 2,
            projects: wire.clone(),
            partial,
            ..Default::default()
        }
        .encode_to_vec();
        assert!(!String::from_utf8_lossy(&encoded).contains("/host-build-facts"));
        assert!(!String::from_utf8_lossy(&encoded).contains(".env"));
        assert!(!String::from_utf8_lossy(&encoded).contains("missing"));
        let decoded = proto::JdtlsProjectSnapshotResponse::decode(encoded.as_slice())
            .expect("typed snapshot");
        assert_eq!(
            projects_from_proto(decoded.projects, host.id(), host).expect("host roundtrip"),
            projects
        );
        let remote = projects_from_proto(wire.clone(), desktop.id(), desktop)
            .expect("desktop worktree admission");
        let facts = remote.first().expect("provider facts");
        assert_eq!(facts.module_root.worktree_id, desktop.id());
        let gradle = facts.gradle.as_ref().expect("remote Gradle facts");
        assert_eq!(gradle.build_root.worktree_id, desktop.id());
        assert_eq!(gradle.tasks[0].options[0].name, "tests");
        assert!(gradle.tasks[0].options_known);
        assert_eq!(gradle.source_sets[0].roots.len(), 2);
        assert!(gradle.configurations_known);
        assert_eq!(gradle.configurations[1].extends_from, ["implementation"]);
        assert!(gradle.configurations[1].resolvable);
        let maven = facts.maven.as_ref().expect("remote Maven facts");
        assert_eq!(maven.pom.worktree_id, desktop.id());
        assert_eq!(maven.modules.len(), 1);
        assert_eq!(maven.reactor_position, Some(0));
        assert!(!maven.profile_definitions_known);
        assert_eq!(maven.profile_definitions.len(), 1);
        assert_eq!(maven.plugins[0].goals[0].name, "test");
        assert!(maven.plugins[0].goals[0].parameters_known);
        assert_eq!(maven.plugins[0].goals[0].parameters[0].name, "skipTests");
        let legacy_goal =
            proto::JdtlsMavenGoalInfo::decode([0x0a, 0x04, b't', b'e', b's', b't'].as_slice())
                .expect("legacy goal without new wire fields");
        let mut legacy_wire = wire.clone();
        legacy_wire[0].maven.as_mut().expect("Maven").plugins[0].goals[0] = legacy_goal;
        let legacy_facts = projects_from_proto(legacy_wire, desktop.id(), desktop)
            .expect("legacy host retains unknown coverage");
        let legacy_goal = &legacy_facts[0].maven.as_ref().expect("Maven").plugins[0].goals[0];
        assert!(!legacy_goal.parameters_known);
        assert!(legacy_goal.parameters.is_empty());
        let legacy_gradle = proto::JdtlsGradleProjectInfo::decode([0x12, 0x01, b':'].as_slice())
            .expect("legacy Gradle without configuration fields");
        let mut legacy_wire = wire.clone();
        legacy_wire[0].gradle = Some(legacy_gradle);
        let legacy_facts = projects_from_proto(legacy_wire, desktop.id(), desktop)
            .expect("legacy Gradle host retains unknown coverage");
        let legacy_gradle = legacy_facts[0].gradle.as_ref().expect("Gradle");
        assert!(!legacy_gradle.configurations_known);
        assert!(legacy_gradle.configurations.is_empty());
        let legacy = project_to_proto_for_version(&projects[0], 1).expect("legacy projection");
        assert!(legacy.gradle.is_none() && legacy.maven.is_none());
        assert_eq!(
            legacy.membership,
            proto::JdtlsProjectMembership::Unknown as i32
        );
        let legacy =
            project_to_proto_for_version(&projects[0], 2).expect("build metadata projection");
        assert!(legacy.gradle.is_some() && legacy.maven.is_some());
        assert_eq!(
            legacy.membership,
            proto::JdtlsProjectMembership::Unknown as i32
        );
        let raw_resources = vec![
            (
                "file:/host-build-facts/library/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedNonJava,
                    ..Default::default()
                },
            ),
            (
                "file:/host-build-facts/application/.env/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedNonJava,
                    ..Default::default()
                },
            ),
            (
                "file:/foreign/root/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedNonJava,
                    ..Default::default()
                },
            ),
            (
                "file:///host-build-facts/library/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedNonJava,
                    ..Default::default()
                },
            ),
        ];
        let (resources, resources_partial) = projects_from_provider(
            raw_resources,
            Path::new("/host-build-facts"),
            host.id(),
            PathStyle::Unix,
            host,
            true,
        );
        assert!(resources_partial);
        assert_eq!(
            resources.len(),
            1,
            "private, foreign and duplicate resources are not shared"
        );
        assert!(resources[0].membership.is_imported());
        let mut imported_resource = resources[0].clone();
        imported_resource.imported_build_kind = JdtlsImportedBuildKind::Maven;
        let resource_wire = project_to_proto(&imported_resource);
        for version in 1..=3 {
            let legacy = project_to_proto_for_version(&imported_resource, version)
                .expect("legacy kind projection");
            assert_eq!(
                legacy.imported_build_kind,
                proto::JdtlsImportedBuildKind::Unknown as i32
            );
        }
        assert_eq!(
            project_to_proto_for_version(&imported_resource, 4)
                .expect("current kind projection")
                .imported_build_kind,
            proto::JdtlsImportedBuildKind::Maven as i32
        );
        let remote_resources =
            projects_from_proto(vec![resource_wire.clone()], desktop.id(), desktop)
                .expect("resource admission");
        assert_eq!(
            remote_resources[0].membership,
            JdtlsProjectMembership::ImportedNonJava
        );
        assert_eq!(remote_resources[0].module_root.worktree_id, desktop.id());
        assert_eq!(
            remote_resources[0].imported_build_kind,
            JdtlsImportedBuildKind::Maven
        );
        for mutation in 0..3 {
            let mut invalid = resource_wire.clone();
            match mutation {
                0 => invalid.imported_build_kind = 99,
                1 => invalid.membership = proto::JdtlsProjectMembership::MetadataOnly as i32,
                _ => invalid.membership = proto::JdtlsProjectMembership::Unknown as i32,
            }
            assert!(projects_from_proto(vec![invalid], desktop.id(), desktop).is_err());
        }
        assert!(remote_resources[0].source_roots.is_empty());
        assert!(
            remote_resources[0].maven.is_none() && remote_resources[0].gradle.is_none(),
            "unknown build kind is not inferred from project files"
        );
        for membership in [
            JdtlsProjectMembership::Unknown,
            JdtlsProjectMembership::MetadataOnly,
        ] {
            assert!(!membership.is_imported());
        }
        for mutation in 0..3 {
            let mut invalid = resource_wire.clone();
            match mutation {
                0 => invalid.membership = 99,
                1 => invalid.project_vm_known = true,
                _ => invalid.source_roots.push(proto::JdtlsSourceInfo {
                    path: "library".into(),
                    navigable: true,
                }),
            }
            assert!(projects_from_proto(vec![invalid], desktop.id(), desktop).is_err());
        }
        assert_eq!(
            jdtls_snapshot_size_limit(1).expect("legacy bound"),
            1_048_576
        );
        assert_eq!(
            jdtls_snapshot_size_limit(2).expect("extended bound"),
            8 * 1_048_576
        );
        assert!(
            project_to_proto_for_version(&projects[0], JDTLS_SNAPSHOT_PROTOCOL_VERSION + 1)
                .is_err()
        );
        let original: serde_json::Value =
            serde_json::to_value(&wire[0]).expect("wire mutation fixture");
        for (pointer, value) in [
            ("/module_root", json!("/host-build-facts/application")),
            ("/gradle/build_root", json!("../foreign")),
            ("/gradle/project_path", json!(":foreign:project")),
            ("/gradle/tasks/0/path", json!(":other:test")),
            ("/gradle/tasks/0/description", json!("secret\u{1b}[0m")),
            ("/gradle/tasks/0/options/0/name", json!("tests\n/private")),
            ("/gradle/source_sets_known", json!(false)),
            ("/gradle/configurations_known", json!(false)),
            ("/gradle/configurations/1/name", json!("implementation")),
            ("/gradle/configurations/0/name", json!("")),
            ("/gradle/configurations/0/name", json!("x".repeat(4097))),
            (
                "/gradle/configurations/0/description",
                json!("secret\u{1b}"),
            ),
            ("/gradle/configurations/1/extends_from", json!(["missing"])),
            (
                "/gradle/configurations/1/extends_from",
                json!(["runtimeClasspath"]),
            ),
            (
                "/gradle/configurations/1/extends_from",
                json!(["implementation", "implementation"]),
            ),
            (
                "/gradle/configurations/1/extends_from",
                json!(vec!["implementation"; 129]),
            ),
            (
                "/gradle/configurations",
                json!(vec![original["gradle"]["configurations"][0].clone(); 513]),
            ),
            (
                "/gradle/source_sets/0/roots/0/path",
                json!("application/.env/missing"),
            ),
            ("/maven/pom", json!("library/pom.xml")),
            (
                "/maven/profiles",
                json!([{"id":"parity","active":true},{"id":"parity","active":false}]),
            ),
            ("/maven/reactor_position", json!(128)),
            ("/maven/reactor_order_problem", json!("raw /private/error")),
            (
                "/maven/profile_definitions/0/source_pom",
                json!("library/pom.xml"),
            ),
            ("/maven/plugins/0/goals/0/parameters_known", json!(false)),
            (
                "/maven/plugins/0/goals/0/parameters/0/name",
                json!("../private"),
            ),
            (
                "/maven/plugins/0/goals/0/parameters/0/alias",
                json!("bad\n"),
            ),
            (
                "/maven/plugins/0/goals/0/parameters/0/declared_type",
                json!("x".repeat(4097)),
            ),
            (
                "/maven/plugins/0/goals/0/parameters",
                json!(vec![
                    original
                        .pointer("/maven/plugins/0/goals/0/parameters/0")
                        .expect("declaration")
                        .clone();
                    257
                ]),
            ),
            ("/maven/plugins/0/goals_known", json!(false)),
            (
                "/maven/plugins/0/goals",
                json!(vec![
                    json!({"name":"test","phase":"test","description":null,"parameters_known":false,"parameters":[]});
                    4097
                ]),
            ),
            ("/maven/lifecycles/0/phases", json!(["test", "test"])),
        ] {
            let mut invalid = original.clone();
            *invalid.pointer_mut(pointer).expect("fixture field") = value;
            let invalid = serde_json::from_value(invalid).expect("typed malformed semantic facts");
            assert!(
                projects_from_proto(vec![invalid], desktop.id(), desktop).is_err(),
                "{pointer}"
            );
            assert!(projects_from_proto(wire.clone(), desktop.id(), desktop).is_ok());
        }
        assert!(
            projects_from_proto(
                vec![wire[0].clone(), wire[0].clone()],
                desktop.id(),
                desktop
            )
            .is_err()
        );
        let mut oversized = wire[0].clone();
        let task = oversized.gradle.as_ref().expect("Gradle").tasks[0].clone();
        oversized.gradle.as_mut().expect("Gradle").tasks = vec![task; 4097];
        assert!(projects_from_proto(vec![oversized], desktop.id(), desktop).is_err());
        let missing = Snapshot::new(
            desktop.id(),
            Arc::from(RelPath::from_unix_str("empty").expect("root name")),
            Arc::from(Path::new("/empty")),
            PathStyle::Unix,
        );
        assert!(projects_from_proto(wire, desktop.id(), &missing).is_err());
    }

    #[test]
    fn remote_build_facts_reject_encoded_payloads_before_semantic_projection() {
        let worktree_id = WorktreeId::from_usize(1);
        let snapshot = Snapshot::new(
            worktree_id,
            Arc::from(RelPath::from_unix_str("fixture").expect("root name")),
            Arc::from(Path::new("/fixture")),
            PathStyle::Unix,
        );
        let project = proto::JdtlsProjectInfo {
            gradle: Some(proto::JdtlsGradleProjectInfo {
                project_path: ":".into(),
                tasks: vec![proto::JdtlsGradleTaskInfo {
                    path: ":test".into(),
                    name: "test".into(),
                    description: Some("x".repeat(8 * 1_048_576)),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            projects_from_proto(vec![project], worktree_id, &snapshot)
                .expect_err("encoded admission bound")
                .to_string(),
            "Remote Java project snapshot exceeds the size limit"
        );
    }

    #[test]
    fn changed_build_inputs_invalidate_pending_provider_facts() {
        for input in [
            "gradle/wrapper/gradle-wrapper.properties",
            ".mvn/wrapper/maven-wrapper.properties",
            ".mvn/maven.config",
            ".mvn/jvm.config",
            "gradle/libs.versions.toml",
            "buildSrc/src/main/kotlin/Conventions.kt",
            "nested/buildSrc/src/main/groovy/Conventions.groovy",
            "application/pom.xml",
            "settings.xml",
        ] {
            assert!(is_project_input(
                RelPath::from_unix_str(input).expect("input path")
            ));
            let mut state = ProviderProjectState::<JdtlsProject>::default();
            let query = state
                .begin(LanguageServerId(1), WorktreeId::from_usize(1))
                .expect("current query");
            state.mark_stale().expect("input invalidation");
            assert!(!state.finish(query, Vec::new(), false));
        }
        for input in [
            "application/src/main/java/Main.java",
            "library/src/main/kotlin/Greeting.kt",
            "buildSrc/build/classes/Conventions.class",
            "application/build/generated/Greeting.kt",
            "Cargo.toml",
        ] {
            assert!(!is_project_input(
                RelPath::from_unix_str(input).expect("source path")
            ));
        }
    }

    fn settings() -> JdtlsProjectSettings {
        serde_json::from_value(serde_json::json!({
            "org.eclipse.jdt.ls.core.sourcePaths": [
                "/fixture/application/src/main/java",
                "/fixture/application/build/generated/sources/groovyFixture/java"
            ],
            "org.eclipse.jdt.ls.core.classpathEntries": [
                {"kind": 2, "path": "/library"},
                {"kind": 1, "path": "/private/cache/customer.jar"}
            ],
            "org.eclipse.jdt.ls.core.vm.location": "/private/jdk",
            "org.eclipse.jdt.core.compiler.source": "17",
            "org.eclipse.jdt.core.compiler.codegen.targetPlatform": "21"
        }))
        .expect("JDTLS fixture settings")
    }

    #[test]
    fn imported_identity_lists_reject_malformed_duplicate_and_unbounded_resources() {
        assert_eq!(
            imported_project_uris(json!(["file:/fixture/aggregate/", "file:/fixture/java/"]))
                .expect("qualified resource identities")
                .len(),
            2
        );
        for value in [
            json!(null),
            json!({"resource":"file:/fixture/aggregate/"}),
            json!(["https://host/private"]),
            json!(["file:/fixture/aggregate/?private"]),
            json!(["file:/fixture/aggregate/#private"]),
            json!(["file:/fixture/%0Aprivate"]),
            json!(["file:/fixture/aggregate/", "file:///fixture/aggregate/"]),
            json!([format!("file:/{}", "a".repeat(4097))]),
            json!(vec!["file:/fixture/aggregate/"; MAX_JDTLS_MODULES + 1]),
        ] {
            assert!(imported_project_uris(value).is_err());
        }
        for membership in [
            JdtlsProjectMembership::ImportedNonJava,
            JdtlsProjectMembership::MetadataOnly,
        ] {
            let mut fabricated = settings();
            fabricated.membership = membership;
            assert!(
                project_from_settings(
                    "file:/fixture/application/",
                    fabricated,
                    Path::new("/fixture"),
                    WorktreeId::from_usize(1),
                    PathStyle::Unix
                )
                .is_err()
            );
        }
    }

    fn imported_kind_fixture() -> Vec<(String, JdtlsProjectSettings)> {
        vec![
            (
                "file:///fixture/java/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedJava,
                    ..Default::default()
                },
            ),
            (
                "file:///fixture/aggregate/".into(),
                JdtlsProjectSettings {
                    membership: JdtlsProjectMembership::ImportedNonJava,
                    ..Default::default()
                },
            ),
        ]
    }

    fn imported_kind_reply(records: Value) -> Value {
        json!({"version":1,"rootUri":"file:///fixture/","projects":records,
            "partial":false,"fullSynchronizationKnown":false})
    }

    #[test]
    fn imported_build_read_roots_reuse_only_confined_initialized_folders() {
        let roots = imported_build_read_roots(
            [
                "file:///fixture/gradle/nested/".into(),
                "file:///fixture/gradle/".into(),
                "file:///foreign/".into(),
                "file:///fixture/maven/".into(),
            ],
            Path::new("/fixture"),
        )
        .expect("confined existing LSP folders");
        assert_eq!(
            roots,
            [
                PathBuf::from("/fixture/gradle"),
                PathBuf::from("/fixture/maven")
            ]
        );
        assert!(
            imported_build_read_roots(
                (0..33).map(|index| format!("file:///fixture/{index}/")),
                Path::new("/fixture")
            )
            .is_err()
        );
        assert!(
            imported_build_read_roots(
                ["file:///fixture/?private-canary".into()],
                Path::new("/fixture")
            )
            .is_err()
        );
        assert!(
            imported_build_read_roots(["file:///foreign/".into()], Path::new("/fixture"))
                .expect("no foreign query")
                .is_empty()
        );
    }

    #[test]
    fn imported_ownership_coverage_distinguishes_empty_and_unavailable_contexts() {
        let context = |kind| {
            (
                PathBuf::from("/fixture"),
                kind,
                imported_kind_reply(json!([])),
            )
        };
        for (replies, expected) in [
            (vec![], false),
            (vec![context(JdtlsImportedBuildKind::Maven)], false),
            (
                vec![
                    context(JdtlsImportedBuildKind::Maven),
                    context(JdtlsImportedBuildKind::Gradle),
                ],
                true,
            ),
        ] {
            assert_eq!(
                admit_imported_build_kinds(&mut [], replies).expect("bounded context"),
                expected
            );
        }
        let mut incomplete = context(JdtlsImportedBuildKind::Maven);
        incomplete.2["partial"] = json!(true);
        assert!(
            !admit_imported_build_kinds(
                &mut [],
                vec![incomplete, context(JdtlsImportedBuildKind::Gradle)]
            )
            .expect("partial context")
        );
        assert!(
            admit_imported_build_kinds(
                &mut [],
                vec![
                    context(JdtlsImportedBuildKind::Maven),
                    context(JdtlsImportedBuildKind::Maven)
                ]
            )
            .is_err()
        );
        let mut projects = imported_kind_fixture();
        assert!(
            admit_imported_build_kinds(
                &mut projects,
                vec![
                    context(JdtlsImportedBuildKind::Maven),
                    context(JdtlsImportedBuildKind::Maven)
                ]
            )
            .is_err()
        );
        assert!(
            projects.iter().all(
                |(_, settings)| settings.imported_build_kind == JdtlsImportedBuildKind::Unknown
            )
        );
    }

    #[test]
    fn imported_ownership_scope_is_bounded_and_preserves_initialized_uri_identity() {
        assert_eq!(
            imported_ownership_scope_folders(
                [
                    "file:///fixture".into(),
                    "file:///fixture/nested/".into(),
                    "file:///fixture".into(),
                    "file:///foreign/".into()
                ],
                Path::new("/fixture")
            )
            .expect("initialized folders"),
            ["file:///fixture", "file:///fixture/nested/"]
        );
        for invalid in [
            "file:///fixture/?private-canary",
            "file:///fixture/\n",
            "https://example.com/fixture",
            "file:///fixture/../foreign/",
        ] {
            assert!(
                imported_ownership_scope_folders([invalid.into()], Path::new("/fixture")).is_err()
            );
        }
        assert!(
            imported_ownership_scope_folders(
                (0..33).map(|index| format!("file:///fixture/{index}/")),
                Path::new("/fixture")
            )
            .is_err()
        );
        assert!(
            imported_ownership_scope_folders(
                [format!("file:///fixture/{}", "x".repeat(4096))],
                Path::new("/fixture")
            )
            .is_err()
        );
    }

    #[test]
    fn imported_build_kinds_are_independent_of_java_and_cached_model_facts() {
        let mut projects = imported_kind_fixture();
        admit_imported_build_kinds(
            &mut projects,
            vec![
                (
                    PathBuf::from("/fixture"),
                    JdtlsImportedBuildKind::Maven,
                    imported_kind_reply(json!([
                        {"projectUri":"file:///fixture/aggregate/","javaNature":false}
                    ])),
                ),
                (
                    PathBuf::from("/fixture"),
                    JdtlsImportedBuildKind::Gradle,
                    imported_kind_reply(json!([
                        {"projectUri":"file:///fixture/java/","javaNature":true}
                    ])),
                ),
            ],
        )
        .expect("qualified separate provider facts");
        assert_eq!(
            projects[0].1.imported_build_kind,
            JdtlsImportedBuildKind::Gradle
        );
        assert_eq!(
            projects[1].1.imported_build_kind,
            JdtlsImportedBuildKind::Maven
        );
        assert!(
            projects
                .iter()
                .all(|(_, settings)| !settings.imported_build_kinds_incomplete)
        );
        assert!(
            projects
                .iter()
                .all(|(_, settings)| settings.maven.is_none() && settings.gradle.is_none())
        );
        admit_imported_build_kinds(&mut projects, Vec::new())
            .expect("missing capability remains unknown");
        assert!(
            projects
                .iter()
                .all(|(_, settings)| settings.imported_build_kind
                    == JdtlsImportedBuildKind::Unknown
                    && !settings.imported_build_kinds_incomplete)
        );
        admit_imported_build_kinds(
            &mut projects,
            vec![(
                PathBuf::from("/fixture"),
                JdtlsImportedBuildKind::Maven,
                imported_kind_reply(json!([])),
            )],
        )
        .expect("known empty reply does not invent ownership");
        assert!(
            projects.iter().all(
                |(_, settings)| settings.imported_build_kind == JdtlsImportedBuildKind::Unknown
            )
        );
    }

    #[test]
    fn imported_build_kind_admission_rejects_invalid_replies_atomically() {
        let valid = imported_kind_reply(json!([
            {"projectUri":"file:///fixture/java/","javaNature":true}
        ]));
        let mut invalid = Vec::new();
        for mutation in 0..13 {
            let mut value = valid.clone();
            match mutation {
                0 => value["version"] = json!(2),
                1 => value["fullSynchronizationKnown"] = json!(true),
                2 => value["rootUri"] = json!("file:///foreign/"),
                3 => value["projects"][0]["projectUri"] = json!("file:///foreign/private-canary/"),
                4 => value["projects"][0]["projectUri"] = json!("file:///fixture/java/../java/"),
                5 => value["projects"][0]["javaNature"] = json!(false),
                6 => value["projects"][0]["javaNature"] = json!("true"),
                7 => {
                    value["projects"] =
                        json!([value["projects"][0].clone(), value["projects"][0].clone()])
                }
                8 => value["projects"] = json!(vec![value["projects"][0].clone(); 129]),
                9 => value["partial"] = json!(null),
                10 => value["rootUri"] = json!("file:/fixture/"),
                11 => {
                    value["projects"][0]["projectUri"] =
                        json!("file:///fixture/java/?private-canary")
                }
                _ => value["private-canary"] = json!("x".repeat(1_048_576)),
            }
            invalid.push(value);
        }
        for value in invalid {
            let mut projects = imported_kind_fixture();
            let error = admit_imported_build_kinds(
                &mut projects,
                vec![
                    (
                        PathBuf::from("/fixture"),
                        JdtlsImportedBuildKind::Gradle,
                        imported_kind_reply(json!([
                            {"projectUri":"file:///fixture/aggregate/","javaNature":false}
                        ])),
                    ),
                    (
                        PathBuf::from("/fixture"),
                        JdtlsImportedBuildKind::Maven,
                        value,
                    ),
                ],
            )
            .expect_err("malformed generation must be rejected");
            assert!(!format!("{error:#}").contains("private-canary"));
            assert!(projects.iter().all(
                |(_, settings)| settings.imported_build_kind == JdtlsImportedBuildKind::Unknown
            ));
        }
        let mut projects = imported_kind_fixture();
        let error = admit_imported_build_kinds(
            &mut projects,
            vec![
                (
                    PathBuf::from("/fixture"),
                    JdtlsImportedBuildKind::Maven,
                    valid.clone(),
                ),
                (
                    PathBuf::from("/fixture"),
                    JdtlsImportedBuildKind::Gradle,
                    valid,
                ),
            ],
        )
        .expect_err("conflicting provider ownership");
        assert!(error.to_string().contains("conflicting providers"));
        assert!(
            projects.iter().all(
                |(_, settings)| settings.imported_build_kind == JdtlsImportedBuildKind::Unknown
            )
        );
    }

    #[test]
    fn imported_build_kind_omissions_remain_partial_without_inventing_resources() {
        let mut projects = imported_kind_fixture();
        let mut reply = imported_kind_reply(json!([
            {"projectUri":"file:///fixture/java/","javaNature":true},
            {"projectUri":"file:///fixture/new-during-read/","javaNature":false}
        ]));
        reply["partial"] = json!(true);
        admit_imported_build_kinds(
            &mut projects,
            vec![(
                PathBuf::from("/fixture"),
                JdtlsImportedBuildKind::Maven,
                reply,
            )],
        )
        .expect("partial reads preserve observed resources");
        assert_eq!(projects.len(), 2);
        assert_eq!(
            projects[0].1.imported_build_kind,
            JdtlsImportedBuildKind::Maven
        );
        assert_eq!(
            projects[1].1.imported_build_kind,
            JdtlsImportedBuildKind::Unknown
        );
        assert!(
            projects
                .iter()
                .all(|(_, settings)| settings.imported_build_kinds_incomplete)
        );
        projects[0].1.membership = JdtlsProjectMembership::MetadataOnly;
        assert!(
            admit_imported_build_kinds(
                &mut projects,
                vec![(
                    PathBuf::from("/fixture"),
                    JdtlsImportedBuildKind::Maven,
                    imported_kind_reply(json!([
                        {"projectUri":"file:///fixture/java/","javaNature":true}
                    ]))
                )]
            )
            .is_err()
        );
        assert!(
            project_from_settings(
                "file:///fixture/java/",
                projects.remove(0).1,
                Path::new("/fixture"),
                WorktreeId::from_usize(1),
                PathStyle::Unix
            )
            .is_err()
        );
    }

    #[test]
    fn gradle_configuration_names_preserve_qualified_literal_identity() {
        let (selection, response) = gradle_graph_fixture();
        for name in [
            "härt-α".to_owned(),
            "runtime graph".to_owned(),
            "runtime📦".to_owned(),
            format!("r{}", "x".repeat(128)),
            "x".repeat(4096),
            "α".repeat(2048),
        ] {
            let selected =
                SelectedGradleDependencyGraph::new(&selection.root, &selection.module, &name)
                    .expect("qualified literal configuration");
            assert_eq!(
                selected.parameters.arguments[0]["configuration"],
                json!(name)
            );
            let changed =
                change_gradle_graph(&response, |graph| graph["configuration"] = json!(name));
            let graph = selected
                .validate_response(changed)
                .expect("exact graph identity");
            assert_eq!(graph.resolution.configuration, name);
            validate_gradle_configurations(
                true,
                &[GradleConfiguration {
                    name,
                    description: None,
                    resolvable: true,
                    consumable: false,
                    extends_from: Vec::new(),
                }],
            )
            .expect("qualified declaration");
        }
        for name in [
            "".to_owned(),
            "runtime:α".to_owned(),
            "x/y".to_owned(),
            "x\\y".to_owned(),
            "<x>".to_owned(),
            "x\"y".to_owned(),
            "x?".to_owned(),
            "x*".to_owned(),
            "x|".to_owned(),
            ".x".to_owned(),
            "x.".to_owned(),
            "x\nsecret".to_owned(),
            "x\0secret".to_owned(),
            "x\u{85}secret".to_owned(),
            "x".repeat(4097),
            "α".repeat(2049),
        ] {
            let error =
                SelectedGradleDependencyGraph::new(&selection.root, &selection.module, &name)
                    .expect_err("invalid configuration");
            assert!(!format!("{error:#}").contains("secret"));
            assert!(
                validate_gradle_configurations(
                    true,
                    &[GradleConfiguration {
                        name,
                        description: None,
                        resolvable: true,
                        consumable: false,
                        extends_from: Vec::new(),
                    }]
                )
                .is_err()
            );
        }
    }

    #[test]
    fn gradle_configuration_parent_budget_is_aggregate() {
        let names = (0..65)
            .map(|index| format!("configuration{index}"))
            .collect::<Vec<_>>();
        let mut configurations = names
            .iter()
            .map(|name| GradleConfiguration {
                name: name.clone(),
                description: None,
                resolvable: false,
                consumable: false,
                extends_from: names
                    .iter()
                    .filter(|parent| *parent != name)
                    .cloned()
                    .collect(),
            })
            .collect::<Vec<_>>();
        assert!(validate_gradle_configurations(true, &configurations).is_err());
        let (first, remaining) = configurations
            .split_first_mut()
            .expect("fixture declarations");
        first.extends_from.clear();
        assert_eq!(
            remaining
                .iter()
                .map(|item| item.extends_from.len())
                .sum::<usize>(),
            4096
        );
        validate_gradle_configurations(true, &configurations).expect("exact aggregate edge budget");
        let (first, remaining) = configurations
            .split_first_mut()
            .expect("fixture declarations");
        first
            .extends_from
            .push(remaining.first().expect("parent declaration").name.clone());
        assert!(validate_gradle_configurations(true, &configurations).is_err());
    }

    #[test]
    fn gradle_model_preserves_composite_identities_and_rejects_invalid_facts() {
        let model = json!({
            "version": 1, "rootUri": "file:///fixture/", "sourceSetsKnown": false,
            "projects": [
                {"buildRootUri": "file:///fixture/", "moduleUri": "file:///fixture/application/",
                 "projectPath": ":application", "name": "application", "tasks": [
                    {"path": ":application:test", "name": "test", "group": "verification",
                     "description": "Run tests", "optionsKnown": false}
                 ]},
                {"buildRootUri": "file:///fixture/included/", "moduleUri": "file:///fixture/included/",
                 "projectPath": ":", "name": "included", "tasks": [
                    {"path": ":test", "name": "test", "group": null, "description": null, "optionsKnown": false}
                 ]}
            ]
        });
        let mut projects = vec![("file:///fixture/application".to_string(), settings())];
        merge_gradle_model(&mut projects, model.clone(), Path::new("/fixture"))
            .expect("valid Gradle model");
        assert_eq!(projects.len(), 2);
        let (uri, settings) = projects.remove(0);
        let (project, partial) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("current model");
        assert!(partial);
        let gradle = project.gradle.expect("Gradle provider facts");
        assert_eq!(gradle.project_path, ":application");
        assert_eq!(gradle.tasks[0].path, ":application:test");
        assert!(!gradle.tasks[0].options_known);
        assert!(!gradle.configurations_known);
        assert!(gradle.configurations.is_empty());
        for (pointer, value) in [
            ("/version", json!(2)),
            ("/rootUri", json!("file:///foreign")),
            ("/sourceSetsKnown", json!(true)),
            (
                "/projects/0/buildRootUri",
                json!("file:///fixture/../foreign"),
            ),
            ("/projects/0/moduleUri", json!("file:///foreign")),
            ("/projects/0/tasks/0/path", json!(":other:test")),
            ("/projects/0/tasks/0/optionsKnown", json!(true)),
            ("/projects/0/tasks/0/group", json!("secret\u{1b}[0m")),
        ] {
            let mut invalid = model.clone();
            *invalid.pointer_mut(pointer).expect("fixture field") = value;
            assert!(
                merge_gradle_model(&mut Vec::new(), invalid, Path::new("/fixture")).is_err(),
                "{pointer}"
            );
        }
        let mut duplicate = model.clone();
        let repeated = duplicate["projects"][0].clone();
        duplicate["projects"]
            .as_array_mut()
            .expect("projects")
            .push(repeated);
        assert!(merge_gradle_model(&mut Vec::new(), duplicate, Path::new("/fixture")).is_err());
        let mut oversized = model;
        oversized["projects"][0]["tasks"] =
            json!(vec![oversized["projects"][0]["tasks"][0].clone(); 4097]);
        assert!(merge_gradle_model(&mut Vec::new(), oversized, Path::new("/fixture")).is_err());
    }

    #[test]
    fn gradle_enum_declarations_distinguish_unknown_empty_and_bounded_members() {
        let declaration =
            json!({"name":"mode", "description":"Fixture mode", "declaredType":"fixture.Mode"});
        let legacy: GradleTaskOption =
            serde_json::from_value(declaration.clone()).expect("legacy option");
        assert!(legacy.enum_constants.is_none());
        for (known, constants, expected) in [
            (false, Vec::<String>::new(), None),
            (true, Vec::new(), Some(Vec::new())),
            (
                true,
                vec!["ALPHA".into(), "ZULU".into()],
                Some(vec!["ALPHA".into(), "ZULU".into()]),
            ),
            (
                true,
                (0..128)
                    .map(|index| format!("V{index}"))
                    .collect::<Vec<_>>(),
                Some(
                    (0..128)
                        .map(|index| format!("V{index}"))
                        .collect::<Vec<_>>(),
                ),
            ),
        ] {
            let mut payload = declaration.clone();
            payload["enumConstantsKnown"] = json!(known);
            payload["enumConstants"] = json!(constants);
            let option: GradleTaskOption =
                serde_json::from_value(payload.clone()).expect("valid declarations");
            assert_eq!(option.enum_constants, expected);
            assert_eq!(option.name, legacy.name);
            assert_eq!(option.description, legacy.description);
            assert_eq!(option.declared_type, legacy.declared_type);
            assert_eq!(
                option,
                serde_json::from_value(payload).expect("stable repeated declaration")
            );
        }
    }

    fn invalid_gradle_enum_declarations() -> Vec<Value> {
        vec![
            json!({"enumConstantsKnown":true}),
            json!({"enumConstantsKnown":false}),
            json!({"enumConstants":[]}),
            json!({"enumConstantsKnown":null,"enumConstants":[]}),
            json!({"enumConstantsKnown":true,"enumConstants":null}),
            json!({"enumConstantsKnown":"private-value","enumConstants":[]}),
            json!({"enumConstantsKnown":true,"enumConstants":"private-value"}),
            json!({"enumConstantsKnown":false,"enumConstants":["ALPHA"]}),
            json!({"enumConstantsKnown":true,"enumConstants":["ALPHA","ALPHA"]}),
            json!({"enumConstantsKnown":true,"enumConstants":[""]}),
            json!({"enumConstantsKnown":true,"enumConstants":["/private-value/path"]}),
            json!({"enumConstantsKnown":true,"enumConstants":["private-value\u{1b}"]}),
            json!({"enumConstantsKnown":true,"enumConstants":["x".repeat(4097)]}),
            json!({"enumConstantsKnown":true,"enumConstants":[12]}),
            json!({"enumConstantsKnown":true,"enumConstants":(0..129).map(|index|format!("V{index}")).collect::<Vec<_>>()}),
        ]
    }

    #[test]
    fn gradle_enum_declarations_reject_malformed_wire_records() {
        let declaration =
            json!({"name":"mode", "description":"Fixture mode", "declaredType":"fixture.Mode"});
        for fields in invalid_gradle_enum_declarations() {
            let mut invalid = declaration.clone();
            invalid
                .as_object_mut()
                .expect("option")
                .extend(fields.as_object().expect("fields").clone());
            assert!(serde_json::from_value::<GradleTaskOption>(invalid).is_err());
        }
    }

    #[test]
    fn gradle_enum_declarations_reject_invalid_pairs_before_snapshot_mutation() {
        let declaration =
            json!({"name":"mode", "description":"Fixture mode", "declaredType":"fixture.Mode"});
        let details = json!({"javaSourceSetsKnown":false, "sourceSets":[],
            "tasks":[{"path":":application:message", "optionsComplete":false, "options":[declaration]}]});
        let model = json!({"version":2,"rootUri":"file:///fixture/","sourceSetsKnown":false,
            "projects":[{"buildRootUri":"file:///fixture/","moduleUri":"file:///fixture/application/",
                "projectPath":":application","name":"application","tasks":[{"path":":application:message",
                    "name":"message","group":null,"description":null,"optionsKnown":false}],"detailsJson":details.to_string()}]});
        let mut projects = Vec::new();
        merge_gradle_model(&mut projects, model.clone(), Path::new("/fixture"))
            .expect("current legacy snapshot");
        let original = format!("{projects:?}");
        for fields in invalid_gradle_enum_declarations() {
            let mut invalid_option = declaration.clone();
            invalid_option
                .as_object_mut()
                .expect("option")
                .extend(fields.as_object().expect("fields").clone());
            assert!(serde_json::from_value::<GradleTaskOption>(invalid_option.clone()).is_err());
            let mut invalid_details = details.clone();
            invalid_details["tasks"][0]["options"][0] = invalid_option;
            let mut invalid_model = model.clone();
            invalid_model["projects"][0]["detailsJson"] = json!(invalid_details.to_string());
            let error = merge_gradle_model(&mut projects, invalid_model, Path::new("/fixture"))
                .expect_err("invalid declarations rejected");
            assert!(!error.to_string().contains("private-value"));
            assert_eq!(format!("{projects:?}"), original);
        }
        let mut current_details = details;
        current_details["tasks"][0]["options"][0]["enumConstantsKnown"] = json!(true);
        current_details["tasks"][0]["options"][0]["enumConstants"] = json!(["ALPHA", "ZULU"]);
        let mut current_model = model;
        current_model["projects"][0]["detailsJson"] = json!(current_details.to_string());
        let mut current = Vec::new();
        merge_gradle_model(&mut current, current_model, Path::new("/fixture"))
            .expect("provider declarations");
        let (uri, settings) = current.pop().expect("module");
        let (project, _) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("current project");
        let gradle = project.gradle.expect("Gradle module");
        assert_eq!(gradle.project_path, ":application");
        assert_eq!(gradle.tasks[0].path, ":application:message");
        assert!(!gradle.tasks[0].options_known);
        assert_eq!(
            gradle.tasks[0].options[0].enum_constants.as_deref(),
            Some(["ALPHA".to_string(), "ZULU".to_string()].as_slice())
        );
    }

    fn compiler_output_model(root: &Path, fields: Value) -> Value {
        let mut source = json!({"name":"main", "generatednessKnown":false,
            "roots":[url::Url::from_file_path(root.join("application/src")).expect("Source URI")]});
        source
            .as_object_mut()
            .expect("Source record")
            .extend(fields.as_object().expect("Declaration fields").clone());
        json!({"version":2,
            "rootUri":url::Url::from_directory_path(root).expect("Build URI"), "sourceSetsKnown":false,
            "projects":[{"buildRootUri":url::Url::from_directory_path(root).expect("Build URI"),
                "moduleUri":url::Url::from_directory_path(root.join("application")).expect("Module URI"),
                "projectPath":":application", "name":"application", "tasks":[],
                "detailsJson":json!({"javaSourceSetsKnown":true,"sourceSets":[source],"tasks":[]}).to_string()}]})
    }

    fn scoped_gradle_composite_fixture() -> Value {
        let root = selected_import_uri(Path::new(util::path!("/groovy-composite-fixture")), true)
            .expect("fixture root")
            .to_string();
        let fixture = include_str!("../test_data/groovy_jvm_gradle/cached-composite-v1.json")
            .replace("file:///groovy-composite-fixture/", &root)
            .replace(
                "file:/groovy-composite-fixture/",
                &root.replacen("file://", "file:", 1),
            );
        serde_json::from_str(&fixture).expect("qualified SDK fixture with platform URI root")
    }

    #[test]
    fn scoped_gradle_composite_preserves_verified_connection_and_participant_identities() {
        let fixture = scoped_gradle_composite_fixture();
        let root = imported_build_identity_path(
            fixture["cached"]["rootUri"].as_str().expect("cached root"),
        )
        .expect("connection root");
        let records = fixture["cached"]["projects"]
            .as_array()
            .expect("cached modules");
        let admitted = records
            .iter()
            .map(|record| {
                assert_eq!(record["buildRootUri"], fixture["cached"]["rootUri"]);
                imported_build_identity_path(record["projectUri"].as_str().expect("module URI"))
                    .expect("cached module")
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(admitted.len(), 4);
        assert_eq!(
            records
                .iter()
                .filter(|record| record["javaNature"] == true)
                .count(),
            2
        );
        let namespace = fixture["namespace"]["projects"]
            .as_array()
            .expect("namespace");
        assert_eq!(namespace.len(), records.len());
        for (record, independent) in records.iter().zip(namespace) {
            assert_eq!(record["projectUri"], independent["projectUri"]);
            assert_eq!(record["javaNature"], independent["javaNature"]);
        }
        let model = fixture["details"]["model"].clone();
        let mut projects = Vec::new();
        merge_gradle_model_with_admission(
            &mut projects,
            model.clone(),
            &root,
            &root,
            Some(&admitted),
        )
        .expect("current captured composite");
        assert_eq!(projects.len(), 4);
        let included = root.join("included");
        let (_, settings) = projects
            .iter()
            .find(|(uri, _)| file_uri_path(uri).is_ok_and(|module| module == included))
            .expect("included non-Java resource");
        assert_eq!(settings.membership, JdtlsProjectMembership::MetadataOnly);
        let included_model = settings.gradle.as_ref().expect("included metadata");
        assert_eq!(
            file_uri_path(&included_model.build_root_uri).expect("participant root"),
            included
        );
        assert_eq!(included_model.project_path, ":");
        assert_eq!(
            projects
                .iter()
                .filter(|(_, settings)| settings
                    .gradle
                    .as_ref()
                    .is_some_and(|model| model.project_path == ":"))
                .count(),
            2
        );
        assert_eq!(
            projects
                .iter()
                .filter(|(_, settings)| settings
                    .gradle
                    .as_ref()
                    .and_then(|model| model.details.as_ref())
                    .is_some_and(|details| details.java_source_sets_known))
                .count(),
            2
        );
        let before = format!("{projects:?}");
        merge_gradle_model_with_admission(&mut projects, model, &root, &root, Some(&admitted))
            .expect("same captured composite refresh");
        assert_eq!(
            format!("{projects:?}"),
            before,
            "stable module/participant identities on refresh"
        );
    }

    #[test]
    fn scoped_gradle_composite_refuses_incomplete_or_forged_participants_atomically() {
        let fixture = scoped_gradle_composite_fixture();
        let root =
            imported_build_identity_path(fixture["cached"]["rootUri"].as_str().expect("root"))
                .expect("connection root");
        let admitted = fixture["cached"]["projects"]
            .as_array()
            .expect("cached modules")
            .iter()
            .map(|record| {
                imported_build_identity_path(record["projectUri"].as_str().expect("module"))
                    .expect("identity")
            })
            .collect::<BTreeSet<_>>();
        let model = fixture["details"]["model"].clone();
        for mode in 0..9 {
            let mut changed = model.clone();
            let projects = changed["projects"].as_array_mut().expect("model projects");
            let included_index = projects
                .iter()
                .position(|record| {
                    record["moduleUri"]
                        .as_str()
                        .is_some_and(|uri| uri.ends_with("/included/"))
                })
                .expect("included record");
            match mode {
                0 => {
                    projects.remove(included_index);
                }
                1 => {
                    projects.push(projects[included_index].clone());
                }
                2 => {
                    projects[included_index]["buildRootUri"] =
                        json!("file:/groovy-composite-fixture/primary/uncaptured/");
                }
                3 => {
                    projects[included_index]["moduleUri"] =
                        json!("file:/groovy-composite-fixture/primary/unknown/");
                }
                4 => {
                    projects[included_index]["buildRootUri"] = json!("file:/private-canary/");
                }
                5 => {
                    projects[included_index]["projectPath"] = json!(":included");
                }
                6 => {
                    let child = projects
                        .iter_mut()
                        .find(|record| record["projectPath"] == ":application")
                        .expect("child");
                    child["projectPath"] = json!(":");
                }
                7 => {
                    projects[included_index]["moduleUri"] =
                        json!("file:/groovy-composite-fixture/primary/included/../included/");
                }
                8 => {
                    projects[included_index]["buildRootUri"] =
                        json!("file:/groovy-composite-fixture/primary/application/");
                }
                _ => unreachable!(),
            }
            let mut original = vec![("file:///preserved-maven/".into(), settings())];
            let before = format!("{original:?}");
            merge_gradle_model_with_admission(
                &mut original,
                changed,
                &root,
                &root,
                Some(&admitted),
            )
            .expect_err("invalid participant must refuse before publication");
            assert_eq!(format!("{original:?}"), before, "atomic refusal for {mode}");
        }
        for removed in [&root.join("included"), &root.join("application")] {
            let mut current = admitted.clone();
            current.remove(removed);
            let mut original = Vec::new();
            merge_gradle_model_with_admission(
                &mut original,
                model.clone(),
                &root,
                &root,
                Some(&current),
            )
            .expect_err("closed or unknown cached module");
            assert!(original.is_empty());
        }
    }

    fn scoped_gradle_metadata_reply(workspace: &Path, build: &Path) -> Value {
        let mut model = compiler_output_model(build, json!({}));
        model["projects"].as_array_mut().expect("model projects").insert(0, json!({
            "buildRootUri":selected_import_uri(build,true).expect("build URI"),
            "moduleUri":selected_import_uri(build,true).expect("root URI"),
            "projectPath":":", "name":"build", "tasks":[],
            "detailsJson":json!({"javaSourceSetsKnown":false,"sourceSets":[],"tasks":[]}).to_string()
        }));
        for project in model["projects"].as_array_mut().expect("projects") {
            for field in ["buildRootUri", "moduleUri"] {
                let uri = project[field].as_str().expect("directory URI");
                project[field] = json!(uri.replacen("file://", "file:", 1));
            }
        }
        json!({"version":4,"workspaceRootUri":selected_import_uri(workspace,true).expect("workspace URI"),
            "buildRootUri":selected_import_uri(build,true).expect("build URI"),
            "model":model,"modelQueryComplete":true,"fullSynchronizationKnown":false})
    }

    #[test]
    fn scoped_gradle_metadata_uri_spellings_match_the_verified_java_contract() {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Case {
            name: String,
            nio_suffix: String,
            java_file_suffix: String,
        }
        #[derive(Deserialize)]
        struct Fixture {
            version: u32,
            cases: Vec<Case>,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../test_data/groovy_jvm_gradle/java-uri-spellings-v1.json"
        ))
        .expect("actual JBR spelling fixture");
        assert_eq!(fixture.version, 1);
        assert_eq!(fixture.cases.len(), 11);
        let root = Path::new(util::path!("/gradle-uri-forms"));
        let nio_root = selected_import_uri(root, true)
            .expect("NIO root")
            .to_string();
        let java_root = nio_root.replacen("file://", "file:", 1);
        for case in fixture.cases {
            let expected = root.join(&case.name);
            let nio = format!("{nio_root}{}", case.nio_suffix);
            let java = format!("{java_root}{}", case.java_file_suffix);
            assert_eq!(
                selected_import_uri(&expected, true)
                    .expect("directory URI")
                    .as_str(),
                nio
            );
            assert_eq!(
                selected_import_uri(&expected, false)
                    .expect("file URI")
                    .as_str(),
                nio.trim_end_matches('/')
            );
            assert_eq!(
                imported_build_identity_path(&nio).expect("provider NIO identity"),
                expected
            );
            assert_eq!(
                imported_build_identity_path(
                    url::Url::from_directory_path(&expected)
                        .expect("client URL")
                        .as_str()
                )
                .expect("existing client initialized identity"),
                expected
            );
            assert_eq!(
                gradle_model_identity_path(&nio).expect("NIO inner identity"),
                expected
            );
            assert_eq!(
                gradle_model_identity_path(&java).expect("File inner identity"),
                expected
            );
            assert!(
                imported_build_identity_path(&java).is_err(),
                "outer scoped envelopes retain their strict NIO/client directory contract"
            );
        }
        for uri in [
            format!("{java_root}application/../application/"),
            format!("{java_root}application/%2e%2e/application/"),
            format!("{java_root}%61pplication/"),
            format!("{java_root}application//"),
            format!("{java_root}application space/"),
            format!("{java_root}source[set]/"),
            format!("{java_root}application/?private-canary"),
            format!("{java_root}application/#private-canary"),
            format!("{nio_root}application/../application/"),
            nio_root.replacen("file:///", "file://localhost/", 1),
        ] {
            assert!(
                gradle_model_identity_path(&uri).is_err(),
                "URI alias must refuse"
            );
        }
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_admits_current_models_and_sanitizes_invalid_replies(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let workspace = Path::new(util::path!("/scoped-metadata"));
        let build = workspace.join("build");
        let application = build.join("application");
        let admitted = BTreeSet::from([application]);
        let value = Arc::new(std::sync::Mutex::new(scoped_gradle_metadata_reply(
            workspace, &build,
        )));
        let calls = Arc::new(AtomicUsize::new(0));
        let (server, fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: vec![
                        "groovy.gradle.projectDetails.v4".into(),
                        "groovy.gradle.projectDetails.v3".into(),
                        "groovy.gradle.projectDetails".into(),
                    ],
                    ..Default::default()
                }),
                ..Default::default()
            },
            &mut cx.to_async(),
        );
        fake.set_request_handler::<lsp::request::ExecuteCommand,_,_>({
            let value=value.clone();let calls=calls.clone();let build=build.clone();
            move |parameters,_| {
                assert_eq!(parameters.command,"groovy.gradle.projectDetails.v4");
                assert_eq!(parameters.arguments,vec![json!({"workspaceRootUri":selected_import_uri(workspace,true).expect("workspace"),
                    "buildRootUri":selected_import_uri(&build,true).expect("build"),"offline":true})]);
                calls.fetch_add(1,Ordering::SeqCst);
                let reply=value.lock().expect("response fixture").clone();
                async move {Ok(Some(reply))}
            }
        });
        let server = cx
            .update(|cx| {
                let parameters = server.default_initialize_params(false, false, cx);
                server.initialize(
                    parameters,
                    Arc::new(lsp::DidChangeConfigurationParams {
                        settings: json!({}),
                    }),
                    Duration::from_secs(10),
                    cx,
                )
            })
            .await
            .expect("initialized fake");
        let original = vec![
            (
                selected_import_uri(&workspace.join("maven"), true)
                    .expect("Maven URI")
                    .to_string(),
                settings(),
            ),
            (
                selected_import_uri(&workspace.join("plain"), true)
                    .expect("plain URI")
                    .to_string(),
                settings(),
            ),
        ];
        let original_debug = format!("{original:?}");
        let projects = query_scoped_gradle_project_metadata(
            server.clone(),
            Duration::from_secs(10),
            original,
            workspace,
            workspace,
            &build,
            &admitted,
        )
        .await
        .expect("current v4 details");
        assert_eq!(projects.len(), 4);
        assert_eq!(format!("{:?}", &projects[..2]), original_debug);
        assert_eq!(
            projects
                .iter()
                .filter(|(_, settings)| settings.gradle.is_some())
                .count(),
            2
        );
        assert!(
            projects
                .iter()
                .filter(|(_, settings)| settings.gradle.is_some())
                .all(|(_, settings)| settings.membership == JdtlsProjectMembership::MetadataOnly)
        );
        let baseline = scoped_gradle_metadata_reply(workspace, &build);
        let mut invalid = Vec::new();
        for (key, field) in [
            ("version", json!(3)),
            ("modelQueryComplete", json!(false)),
            ("fullSynchronizationKnown", json!(true)),
            ("workspaceRootUri", json!("file:///private-canary/")),
            ("buildRootUri", json!("file:///private-canary/")),
            ("private-canary", json!("must-not-leak")),
        ] {
            let mut changed = baseline.clone();
            changed[key] = field;
            invalid.push(changed);
        }
        let mut missing = baseline.clone();
        missing["model"]["projects"]
            .as_array_mut()
            .expect("projects")
            .pop();
        invalid.push(missing);
        let mut duplicate = baseline.clone();
        let record = duplicate["model"]["projects"][1].clone();
        duplicate["model"]["projects"]
            .as_array_mut()
            .expect("projects")
            .push(record);
        invalid.push(duplicate);
        let mut unknown = baseline.clone();
        unknown["model"]["projects"][1]["moduleUri"] =
            json!(selected_import_uri(&build.join("unknown"), true).expect("unknown URI"));
        invalid.push(unknown);
        let mut foreign = baseline.clone();
        foreign["model"]["projects"][1]["buildRootUri"] =
            json!(selected_import_uri(&workspace.join("other"), true).expect("other URI"));
        invalid.push(foreign);
        let mut wrong_version = baseline.clone();
        wrong_version["model"]["version"] = json!(1);
        invalid.push(wrong_version);
        let mut alias = baseline.clone();
        alias["model"]["rootUri"] = json!(format!(
            "{}../build/",
            selected_import_uri(&build, true).expect("build URI")
        ));
        invalid.push(alias);
        let count = invalid.len();
        for reply in invalid {
            *value.lock().expect("response fixture") = reply;
            let message = query_scoped_gradle_project_metadata(
                server.clone(),
                Duration::from_secs(10),
                Vec::new(),
                workspace,
                workspace,
                &build,
                &admitted,
            )
            .await
            .expect_err("invalid metadata must refuse")
            .to_string();
            assert!(
                message.contains("refresh")
                    && !message.contains("private-canary")
                    && !message.contains("must-not-leak")
                    && !message.contains(workspace.to_str().expect("root text"))
            );
        }
        *value.lock().expect("response fixture") = baseline;
        query_scoped_gradle_project_metadata(
            server,
            Duration::from_secs(10),
            Vec::new(),
            workspace,
            workspace,
            &build,
            &admitted,
        )
        .await
        .expect("recovery after rejected replies");
        assert_eq!(calls.load(Ordering::SeqCst), count + 2);
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_keeps_source_visibility_within_the_owning_worktree(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let worktree = Path::new(util::path!("/scoped-shared-sources"));
        let build = worktree.join("build");
        let modules = BTreeSet::from([build.join("application")]);
        let shared = selected_import_uri(&worktree.join("shared"), true).expect("shared source");
        let mut reply = scoped_gradle_metadata_reply(&build, &build);
        let mut details: Value = serde_json::from_str(
            reply["model"]["projects"][1]["detailsJson"]
                .as_str()
                .expect("details"),
        )
        .expect("details JSON");
        details["sourceSets"][0]["roots"] = json!([shared]);
        reply["model"]["projects"][1]["detailsJson"] = json!(details.to_string());
        let value = Arc::new(std::sync::Mutex::new(reply));
        let requests = Arc::new(AtomicUsize::new(0));
        let (server, fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: vec!["groovy.gradle.projectDetails.v4".into()],
                    ..Default::default()
                }),
                ..Default::default()
            },
            &mut cx.to_async(),
        );
        server.set_workspace_folders(
            [selected_import_uri(&build, true)
                .expect("scope")
                .as_str()
                .parse()
                .expect("folder")]
            .into(),
        );
        fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
            let build = build.clone();
            let value = value.clone();
            let requests = requests.clone();
            move |parameters, _| {
                assert_eq!(parameters.command, "groovy.gradle.projectDetails.v4");
                let uri = selected_import_uri(&build, true).expect("build");
                assert_eq!(
                    parameters.arguments,
                    [json!({"workspaceRootUri":uri,"buildRootUri":uri,"offline":true})]
                );
                requests.fetch_add(1, Ordering::SeqCst);
                let reply = value.lock().expect("reply").clone();
                async move { Ok(Some(reply)) }
            }
        });
        let server = cx
            .update(|cx| {
                let parameters = server.default_initialize_params(false, false, cx);
                server.initialize(
                    parameters,
                    Arc::new(lsp::DidChangeConfigurationParams {
                        settings: json!({}),
                    }),
                    Duration::from_secs(10),
                    cx,
                )
            })
            .await
            .expect("initialized provider");
        let projects = query_scoped_gradle_project_metadata(
            server.clone(),
            Duration::from_secs(10),
            Vec::new(),
            worktree,
            &build,
            &build,
            &modules,
        )
        .await
        .expect("shared source within trusted worktree");
        let application = projects
            .iter()
            .find(|(uri, _)| {
                gradle_model_identity_path(uri).is_ok_and(|path| path == build.join("application"))
            })
            .expect("application");
        let details = application
            .1
            .gradle
            .as_ref()
            .and_then(|project| project.details.as_ref())
            .expect("source details");
        assert_eq!(details.source_sets[0].roots, [shared.to_string()]);
        let mut invalid: Value = serde_json::from_str(
            value.lock().expect("reply")["model"]["projects"][1]["detailsJson"]
                .as_str()
                .expect("details"),
        )
        .expect("details JSON");
        invalid["sourceSets"][0]["roots"] =
            json!([
                selected_import_uri(&worktree.with_file_name("outside-source-canary"), true)
                    .expect("outside URI")
            ]);
        value.lock().expect("reply")["model"]["projects"][1]["detailsJson"] =
            json!(invalid.to_string());
        let error = query_scoped_gradle_project_metadata(
            server.clone(),
            Duration::from_secs(10),
            Vec::new(),
            worktree,
            &build,
            &build,
            &modules,
        )
        .await
        .expect_err("source outside worktree refuses")
        .to_string();
        assert!(error.contains("refresh") && !error.contains("outside-source-canary"));
        query_scoped_gradle_project_metadata(
            server,
            Duration::from_secs(10),
            Vec::new(),
            &worktree.join("unrelated"),
            &build,
            &build,
            &modules,
        )
        .await
        .expect_err("unrelated host bound refuses before RPC");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_never_falls_back_to_root_only_capabilities(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let workspace = Path::new(util::path!("/scoped-capabilities"));
        let build = workspace.join("build");
        for commands in [
            vec![],
            vec!["groovy.gradle.projectDetails.v3".into()],
            vec!["groovy.gradle.projectDetails".into()],
        ] {
            let requests = Arc::new(AtomicUsize::new(0));
            let (server, fake) = lsp::FakeLanguageServer::new(
                LanguageServerId(0),
                LanguageServerBinary {
                    path: "unused-fake-jdtls".into(),
                    arguments: vec![],
                    env: None,
                },
                "jdtls".into(),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                &mut cx.to_async(),
            );
            fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                let requests = requests.clone();
                move |_, _| {
                    requests.fetch_add(1, Ordering::SeqCst);
                    async { Ok(Some(Value::Null)) }
                }
            });
            let server = cx
                .update(|cx| {
                    let parameters = server.default_initialize_params(false, false, cx);
                    server.initialize(
                        parameters,
                        Arc::new(lsp::DidChangeConfigurationParams {
                            settings: json!({}),
                        }),
                        Duration::from_secs(10),
                        cx,
                    )
                })
                .await
                .expect("initialized fake");
            let error = query_scoped_gradle_project_metadata(
                server,
                Duration::from_secs(10),
                Vec::new(),
                workspace,
                workspace,
                &build,
                &BTreeSet::from([build.clone()]),
            )
            .await
            .expect_err("no v3 or legacy fallback")
            .to_string();
            assert!(error.contains("Module-fenced") && error.contains("update"));
            assert_eq!(requests.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn gradle_compiler_output_preserves_knowledge_and_legacy_roots() {
        let root = Path::new(util::path!("/compiler-output"));
        let output =
            url::Url::from_file_path(root.join("application/generated")).expect("Output URI");
        for (fields, expected) in [
            (json!({}), GradleAnnotationProcessorOutput::Unknown),
            (
                json!({"annotationProcessorOutputKnown":false,"annotationProcessorOutputUri":null}),
                GradleAnnotationProcessorOutput::Unknown,
            ),
            (
                json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":null}),
                GradleAnnotationProcessorOutput::Absent,
            ),
            (
                json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":output}),
                GradleAnnotationProcessorOutput::Declared(JdtlsSourceRoot {
                    path: ProjectPath {
                        worktree_id: WorktreeId::from_usize(1),
                        path: RelPath::from_unix_str("application/generated")
                            .expect("Output path")
                            .into(),
                    },
                    navigable: false,
                }),
            ),
        ] {
            let mut projects = Vec::new();
            merge_gradle_model(&mut projects, compiler_output_model(root, fields), root)
                .expect("Provider source declaration");
            let (uri, settings) = projects.pop().expect("Module");
            let (project, _) = project_from_settings(
                &uri,
                settings,
                root,
                WorktreeId::from_usize(1),
                PathStyle::local(),
            )
            .expect("Typed declaration");
            let gradle = project.gradle.expect("Gradle facts");
            let source = gradle.source_sets.first().expect("Main source set");
            assert_eq!(source.annotation_processor_output, expected);
            assert_eq!(source.name, "main");
            assert_eq!(source.roots.len(), 1);
            assert_eq!(source.roots[0].path.path.as_unix_str(), "application/src");
            assert!(!source.roots[0].navigable);
            assert_eq!(gradle.project_path, ":application");
        }
    }

    #[test]
    fn gradle_compiler_output_rejects_invalid_pairs_and_paths_atomically() {
        let root = Path::new(util::path!("/compiler-output"));
        let declared = url::Url::from_file_path(root.join("application/generated"))
            .expect("Output URI")
            .to_string();
        let foreign =
            url::Url::from_file_path(root.parent().expect("Root parent").join("private-canary"))
                .expect("Foreign URI");
        let mut projects = Vec::new();
        merge_gradle_model(&mut projects, compiler_output_model(root, json!({})), root)
            .expect("Existing legacy destination");
        let original = format!("{projects:?}");
        for fields in [
            json!({"annotationProcessorOutputKnown":true}),
            json!({"annotationProcessorOutputUri":null}),
            json!({"annotationProcessorOutputKnown":null,"annotationProcessorOutputUri":null}),
            json!({"annotationProcessorOutputKnown":"private-canary","annotationProcessorOutputUri":null}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":12}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":""}),
            json!({"annotationProcessorOutputKnown":false,"annotationProcessorOutputUri":declared}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":foreign}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":"private-canary"}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":"https://private-canary/path"}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":format!("{declared}?private-canary")}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":format!("{declared}#private-canary")}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":format!("{declared}/{}", "x".repeat(4096))}),
            json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":format!("{declared}/private-canary%1b")}),
        ] {
            let error =
                merge_gradle_model(&mut projects, compiler_output_model(root, fields), root)
                    .expect_err("Invalid declaration")
                    .to_string();
            assert!(!error.contains("private-canary"), "Sanitized category only");
            assert_eq!(format!("{projects:?}"), original);
        }
        let mut nested = compiler_output_model(
            root,
            json!({
            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":declared}),
        );
        nested["projects"][0]["buildRootUri"] = json!(
            url::Url::from_directory_path(root.join("application")).expect("Nested build URI")
        );
        let mut details: Value = serde_json::from_str(
            nested["projects"][0]["detailsJson"]
                .as_str()
                .expect("Details"),
        )
        .expect("JSON");
        details["sourceSets"][0]["annotationProcessorOutputUri"] = json!(
            url::Url::from_file_path(root.join("another-build/generated"))
                .expect("Foreign build output URI")
        );
        nested["projects"][0]["detailsJson"] = json!(details.to_string());
        assert!(merge_gradle_model(&mut projects, nested, root).is_err());
        assert_eq!(format!("{projects:?}"), original);
        let roots = (0..MAX_JDTLS_SOURCE_ROOTS)
            .map(|index| {
                url::Url::from_file_path(root.join(format!("application/source-{index}")))
                    .expect("Bounded source URI")
            })
            .collect::<Vec<_>>();
        let mut exact = Vec::new();
        merge_gradle_model(
            &mut exact,
            compiler_output_model(root, json!({"roots":roots})),
            root,
        )
        .expect("Exact legacy source budget");
        let overflow = compiler_output_model(
            root,
            json!({"roots":roots,
            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":declared}),
        );
        assert!(merge_gradle_model(&mut projects, overflow, root).is_err());
        assert_eq!(format!("{projects:?}"), original);
        let valid = compiler_output_model(
            root,
            json!({"roots":&roots[..MAX_JDTLS_SOURCE_ROOTS-1],
            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":declared}),
        );
        merge_gradle_model(&mut projects, valid, root)
            .expect("Declaration shares aggregate path budget");
    }

    #[gpui::test]
    async fn gradle_compiler_output_query_reuses_state_and_current_path_privacy(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let root = Path::new(util::path!("/compiler-output"));
        let (_filesystem, project) = default_root_fixture(
            root.to_str().expect("Root text"),
            json!({"application":{"build.gradle":"", "src":{"Main.java":"class Main {}"},
                "generated":{}, ".env":{"generated":{}}}}),
            cx,
        )
        .await;
        let entries = default_root_snapshot(&project, cx);
        let mode = Arc::new(AtomicUsize::new(0));
        let paths = [
            "application/generated",
            "application/missing",
            "application/.env/generated",
        ];
        let root_uri = url::Url::from_directory_path(root).expect("Root URI");
        let (server, fake) = lsp::FakeLanguageServer::new(
            LanguageServerId(0),
            LanguageServerBinary {
                path: "unused-fake-jdtls".into(),
                arguments: vec![],
                env: None,
            },
            "jdtls".into(),
            lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: vec!["groovy.gradle.projectDetails".into()],
                    ..Default::default()
                }),
                ..Default::default()
            },
            &mut cx.to_async(),
        );
        fake.set_request_handler::<lsp::request::ExecuteCommand,_,_>({
            let mode=mode.clone();
            move |parameters,_| {
                assert_eq!(parameters.command,"groovy.gradle.projectDetails");
                assert_eq!(parameters.arguments,[json!({"rootUri":root_uri,"offline":true})]);
                let output=url::Url::from_file_path(root.join(paths[mode.load(Ordering::SeqCst)]))
                    .expect("Output URI");
                let response=compiler_output_model(root,json!({"annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":output}));
                async move {Ok(Some(response))}
            }
        });
        let server = cx
            .update(|cx| {
                let parameters = server.default_initialize_params(false, false, cx);
                server.initialize(
                    parameters,
                    Arc::new(lsp::DidChangeConfigurationParams {
                        settings: json!({}),
                    }),
                    Duration::from_secs(10),
                    cx,
                )
            })
            .await
            .expect("Initialized provider");
        let mut state = JdtlsProjectState::default();
        for (index, path) in paths.iter().enumerate() {
            mode.store(index, Ordering::SeqCst);
            let token = state
                .begin(server.server_id(), entries.id())
                .expect("Current generation");
            let raw = query_gradle_project_metadata(
                server.clone(),
                Duration::from_secs(10),
                Vec::new(),
                root.to_owned(),
            )
            .await
            .expect("Actual advertised LSP query");
            let (projects, partial) = projects_from_provider(
                raw,
                root,
                entries.id(),
                PathStyle::local(),
                &entries,
                false,
            );
            assert_eq!(projects.len(), 1);
            let source = &projects[0]
                .gradle
                .as_ref()
                .expect("Gradle facts")
                .source_sets[0];
            match &source.annotation_processor_output {
                GradleAnnotationProcessorOutput::Declared(output) => {
                    assert!(index < 2);
                    assert_eq!(output.path.path.as_unix_str(), *path);
                    assert_eq!(output.navigable, index == 0);
                }
                GradleAnnotationProcessorOutput::Unknown => assert_eq!(index, 2),
                GradleAnnotationProcessorOutput::Absent => {
                    panic!("Omission must never invent known absence")
                }
            }
            if index == 1 {
                let (shared, _) =
                    filter_provider_projects(projects.clone(), partial, &entries, true);
                assert_eq!(
                    shared[0]
                        .gradle
                        .as_ref()
                        .expect("Shared module")
                        .source_sets[0]
                        .annotation_processor_output,
                    GradleAnnotationProcessorOutput::Unknown
                );
            }
            let current = state
                .begin(server.server_id(), entries.id())
                .expect("Replacement generation");
            assert!(!state.finish(token, projects.clone(), partial));
            assert!(state.finish(current, projects, partial));
            assert!(state.snapshot_is_current());
            state.mark_stale().expect("Existing invalidation owner");
            assert!(!state.snapshot_is_current());
            assert!(!state.finish(current, Vec::new(), false));
        }
    }

    async fn compiler_output_wire_fixture(
        cx: &mut gpui::TestAppContext,
    ) -> (Snapshot, JdtlsProject) {
        let root = Path::new(util::path!("/compiler-output-wire"));
        let mut tree = json!({"application":{"build.gradle":"", "src":{}, "generated":{}, ".env":"private"},
            "other-build":{"generated":{}}});
        for index in 0..MAX_JDTLS_SOURCE_ROOTS {
            tree["application"][format!("source-{index}")] = json!({});
        }
        let (_filesystem, project) =
            default_root_fixture(root.to_str().expect("Root text"), tree, cx).await;
        let entries = default_root_snapshot(&project, cx);
        let mut raw = Vec::new();
        merge_gradle_model(&mut raw, compiler_output_model(root, json!({
            "annotationProcessorOutputKnown":true,
            "annotationProcessorOutputUri":url::Url::from_file_path(root.join("application/generated")).expect("Output URI")
        })), root).expect("Local provider declaration");
        let (mut projects, _) =
            projects_from_provider(raw, root, entries.id(), PathStyle::local(), &entries, true);
        assert_eq!(projects.len(), 1);
        (entries, projects.pop().expect("Current visible project"))
    }

    #[gpui::test]
    async fn gradle_compiler_output_wire_preserves_states_and_older_unknown(
        cx: &mut gpui::TestAppContext,
    ) {
        let (entries, mut project) = compiler_output_wire_fixture(cx).await;
        let declared = project.gradle.as_ref().expect("Gradle").source_sets[0]
            .annotation_processor_output
            .clone();
        for knowledge in [
            GradleAnnotationProcessorOutput::Unknown,
            GradleAnnotationProcessorOutput::Absent,
            declared,
        ] {
            project.gradle.as_mut().expect("Gradle").source_sets[0].annotation_processor_output =
                knowledge;
            let wire = project_to_proto_for_version(&project, JDTLS_SNAPSHOT_PROTOCOL_VERSION)
                .expect("Current export");
            let bytes = proto::JdtlsProjectSnapshotResponse {
                protocol_version: JDTLS_SNAPSHOT_PROTOCOL_VERSION,
                projects: vec![wire.clone()],
                ..Default::default()
            }
            .encode_to_vec();
            let decoded = proto::JdtlsProjectSnapshotResponse::decode(bytes.as_slice())
                .expect("Actual protobuf roundtrip");
            assert_eq!(
                projects_from_proto_for_version(
                    decoded.projects,
                    entries.id(),
                    &entries,
                    JDTLS_SNAPSHOT_PROTOCOL_VERSION
                )
                .expect("Current admission"),
                [project.clone()]
            );
            for version in 2..JDTLS_SNAPSHOT_PROTOCOL_VERSION {
                assert!(
                    projects_from_proto_for_version(
                        vec![wire.clone()],
                        entries.id(),
                        &entries,
                        version
                    )
                    .is_err(),
                    "New fields in an older envelope must not be admitted"
                );
                let older =
                    project_to_proto_for_version(&project, version).expect("Older projection");
                let source = &older.gradle.as_ref().expect("Legacy Gradle").source_sets[0];
                assert!(source.annotation_processor_output_known.is_none());
                assert!(source.annotation_processor_output.is_none());
                let imported =
                    projects_from_proto_for_version(vec![older], entries.id(), &entries, version)
                        .expect("Older source facts")
                        .pop()
                        .expect("Module");
                assert_eq!(
                    imported.gradle.as_ref().expect("Legacy Gradle").source_sets[0]
                        .annotation_processor_output,
                    GradleAnnotationProcessorOutput::Unknown
                );
                assert_eq!(
                    imported.gradle.as_ref().expect("Legacy Gradle").source_sets[0].roots,
                    project.gradle.as_ref().expect("Current Gradle").source_sets[0].roots
                );
            }
            let mut legacy = wire;
            let source = &mut legacy.gradle.as_mut().expect("Gradle").source_sets[0];
            source.annotation_processor_output_known = None;
            source.annotation_processor_output = None;
            let legacy = projects_from_proto_for_version(
                vec![legacy],
                entries.id(),
                &entries,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION,
            )
            .expect("Absent legacy fields")
            .pop()
            .expect("Module");
            assert_eq!(
                legacy.gradle.as_ref().expect("Legacy Gradle").source_sets[0]
                    .annotation_processor_output,
                GradleAnnotationProcessorOutput::Unknown
            );
        }
    }

    #[gpui::test]
    async fn gradle_compiler_output_wire_rejects_invalid_private_and_over_budget_paths(
        cx: &mut gpui::TestAppContext,
    ) {
        let (entries, project) = compiler_output_wire_fixture(cx).await;
        let wire = project_to_proto_for_version(&project, JDTLS_SNAPSHOT_PROTOCOL_VERSION)
            .expect("Current wire");
        let output = proto::JdtlsSourceInfo {
            path: "application/generated".into(),
            navigable: true,
        };
        for (known, output) in [
            (None, Some(output.clone())),
            (Some(false), Some(output)),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: "application/.env/private-canary".into(),
                    navigable: true,
                }),
            ),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: "application/private-canary-missing".into(),
                    navigable: true,
                }),
            ),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: "../private-canary".into(),
                    navigable: true,
                }),
            ),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: "/private-canary".into(),
                    navigable: true,
                }),
            ),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: format!("application/{}", "x".repeat(4096)),
                    navigable: true,
                }),
            ),
            (
                Some(true),
                Some(proto::JdtlsSourceInfo {
                    path: "application/private-canary\n".into(),
                    navigable: true,
                }),
            ),
        ] {
            let mut invalid = wire.clone();
            let source = &mut invalid.gradle.as_mut().expect("Gradle").source_sets[0];
            source.annotation_processor_output_known = known;
            source.annotation_processor_output = output;
            let error = projects_from_proto_for_version(
                vec![invalid],
                entries.id(),
                &entries,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION,
            )
            .expect_err("Invalid wire declaration");
            let error = format!("{error:#}");
            assert!(!error.contains("private-canary"), "Sanitized category");
        }
        let mut foreign = wire.clone();
        let gradle = foreign.gradle.as_mut().expect("Gradle");
        gradle.build_root = "application".into();
        gradle.source_sets[0].annotation_processor_output = Some(proto::JdtlsSourceInfo {
            path: "other-build/generated".into(),
            navigable: true,
        });
        assert!(
            projects_from_proto_for_version(
                vec![foreign],
                entries.id(),
                &entries,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION
            )
            .is_err()
        );
        let mut bounded = wire.clone();
        let source = &mut bounded.gradle.as_mut().expect("Gradle").source_sets[0];
        source.roots = (0..MAX_JDTLS_SOURCE_ROOTS - 1)
            .map(|index| proto::JdtlsSourceInfo {
                path: format!("application/source-{index}"),
                navigable: true,
            })
            .collect();
        projects_from_proto_for_version(
            vec![bounded.clone()],
            entries.id(),
            &entries,
            JDTLS_SNAPSHOT_PROTOCOL_VERSION,
        )
        .expect("Exact combined source budget");
        bounded.gradle.as_mut().expect("Gradle").source_sets[0]
            .roots
            .push(proto::JdtlsSourceInfo {
                path: format!("application/source-{}", MAX_JDTLS_SOURCE_ROOTS - 1),
                navigable: true,
            });
        assert!(
            projects_from_proto_for_version(
                vec![bounded],
                entries.id(),
                &entries,
                JDTLS_SNAPSHOT_PROTOCOL_VERSION
            )
            .is_err()
        );
        assert_eq!(
            project_to_proto_for_version(&project, JDTLS_SNAPSHOT_PROTOCOL_VERSION)
                .expect("Unchanged original"),
            wire
        );
    }

    #[test]
    fn gradle_details_preserve_declarations_and_reject_foreign_or_inconsistent_facts() {
        let details = json!({
            "javaSourceSetsKnown": true,
            "sourceSets": [{"name": "main", "generatednessKnown": false,
                "roots": ["file:///fixture/application/src/main/java",
                    "file:///fixture/application/build/generated/sources/java"]}],
            "tasks": [{"path": ":application:message", "optionsComplete": false,
                "options": [{"name": "message", "description": "Fixture message",
                    "declaredType": "org.gradle.api.provider.Property<java.lang.String>"}]}],
            "configurationsKnown": true,
            "dependencyGraphKnown": false,
            "configurations": [
                {"name":"implementation", "description":null, "resolvable":false,
                    "consumable":false, "extendsFrom":[]},
                {"name":"runtimeClasspath", "description":"Runtime declaration", "resolvable":true,
                    "consumable":false, "extendsFrom":["implementation"]}
            ]
        });
        let model = json!({
            "version": 2, "rootUri": "file:///fixture/", "sourceSetsKnown": false,
            "projects": [{"buildRootUri": "file:///fixture/", "moduleUri": "file:///fixture/application/",
                "projectPath": ":application", "name": "application",
                "tasks": [{"path": ":application:message", "name": "message", "group": null,
                    "description": null, "optionsKnown": false}], "detailsJson": details.to_string()}]
        });
        let mut projects = Vec::new();
        merge_gradle_model(&mut projects, model.clone(), Path::new("/fixture"))
            .expect("valid extended metadata");
        let (uri, settings) = projects.remove(0);
        let (project, partial) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("current extended model");
        assert!(partial);
        let gradle = project.gradle.expect("extended Gradle facts");
        assert!(gradle.source_sets_known);
        assert_eq!(gradle.source_sets[0].name, "main");
        assert_eq!(
            gradle.source_sets[0].roots[1].path.path.as_unix_str(),
            "application/build/generated/sources/java"
        );
        assert!(!gradle.source_sets[0].roots[1].navigable);
        assert_eq!(gradle.tasks[0].options[0].name, "message");
        assert!(!gradle.tasks[0].options_known);
        assert!(gradle.configurations_known);
        assert_eq!(gradle.configurations.len(), 2);
        assert!(gradle.configurations[1].resolvable);
        assert_eq!(gradle.configurations[1].extends_from, ["implementation"]);
        for (pointer, value) in [
            ("/javaSourceSetsKnown", json!(false)),
            ("/configurationsKnown", json!(false)),
            ("/dependencyGraphKnown", json!(true)),
            ("/configurations/0/name", json!("")),
            ("/configurations/0/name", json!("secret\u{1b}")),
            ("/configurations/0/name", json!("x".repeat(4097))),
            ("/configurations/0/description", json!("secret\u{1b}")),
            ("/configurations/0/description", json!("x".repeat(4097))),
            ("/configurations/1/name", json!("implementation")),
            ("/configurations/1/extendsFrom", json!(["foreign"])),
            ("/configurations/1/extendsFrom", json!(["runtimeClasspath"])),
            (
                "/configurations/1/extendsFrom",
                json!(["implementation", "implementation"]),
            ),
            (
                "/configurations",
                json!(vec![details["configurations"][0].clone(); 513]),
            ),
            (
                "/configurations/1/extendsFrom",
                json!(vec!["implementation"; 129]),
            ),
            ("/sourceSets/0/generatednessKnown", json!(true)),
            ("/sourceSets/0/name", json!("main\u{1b}")),
            ("/sourceSets/0/roots/0", json!("file:///private/source")),
            ("/tasks/0/path", json!(":foreign:message")),
            ("/tasks/0/options/0/name", json!("message\n")),
            ("/tasks/0/options/0/description", json!("secret\u{1b}[0m")),
            ("/tasks/0/options/0/declaredType", json!("")),
            (
                "/tasks/0/options",
                json!(vec![details["tasks"][0]["options"][0].clone(); 129]),
            ),
            (
                "/sourceSets/0/roots",
                json!(vec!["file:///fixture/root"; 257]),
            ),
        ] {
            let mut invalid_details = details.clone();
            *invalid_details
                .pointer_mut(pointer)
                .expect("metadata fixture field") = value;
            let mut invalid = model.clone();
            invalid["projects"][0]["detailsJson"] = json!(invalid_details.to_string());
            let mut destination = vec![(
                "file:///fixture/existing".to_string(),
                JdtlsProjectSettings::default(),
            )];
            assert!(
                merge_gradle_model(&mut destination, invalid, Path::new("/fixture")).is_err(),
                "{pointer}"
            );
            assert_eq!(destination.len(), 1);
            assert!(destination[0].1.gradle.is_none());
        }
        for known in [false, true] {
            let mut empty_details = details.clone();
            empty_details["configurationsKnown"] = json!(known);
            empty_details["configurations"] = json!([]);
            let mut empty_model = model.clone();
            empty_model["projects"][0]["detailsJson"] = json!(empty_details.to_string());
            let mut records = Vec::new();
            merge_gradle_model(&mut records, empty_model, Path::new("/fixture"))
                .expect("empty declaration coverage");
            let admitted = records[0]
                .1
                .gradle
                .as_ref()
                .expect("Gradle metadata")
                .details
                .as_ref()
                .expect("extended metadata");
            assert_eq!(admitted.configurations_known, known);
            assert!(admitted.configurations.is_empty());
        }
        let mut legacy_details = details.clone();
        for field in [
            "configurationsKnown",
            "configurations",
            "dependencyGraphKnown",
        ] {
            legacy_details
                .as_object_mut()
                .expect("details")
                .remove(field);
        }
        let mut legacy_model = model.clone();
        legacy_model["projects"][0]["detailsJson"] = json!(legacy_details.to_string());
        let mut legacy_records = Vec::new();
        merge_gradle_model(&mut legacy_records, legacy_model, Path::new("/fixture"))
            .expect("legacy detail metadata");
        let legacy = legacy_records[0]
            .1
            .gradle
            .as_ref()
            .expect("Gradle metadata")
            .details
            .as_ref()
            .expect("extended metadata");
        assert!(!legacy.configurations_known);
        assert!(legacy.configurations.is_empty());
        let canary = "GROOVY_PRIVATE_DECLARATION_CANARY";
        for pointer in [
            "/configurations/0/resolvable",
            "/configurations/0/consumable",
        ] {
            let mut private_details = details.clone();
            *private_details.pointer_mut(pointer).expect("role field") = json!(canary);
            let mut invalid_model = model.clone();
            invalid_model["projects"][0]["detailsJson"] = json!(private_details.to_string());
            let error = merge_gradle_model(&mut Vec::new(), invalid_model, Path::new("/fixture"))
                .expect_err("malformed role");
            let diagnostic = format!("{error:#}");
            assert!(diagnostic.contains("check provider setup and refresh"));
            assert!(!diagnostic.contains(canary));
        }
        let mut unknown_field = details;
        unknown_field["configurations"][0][canary] = json!(true);
        let mut invalid_model = model.clone();
        invalid_model["projects"][0]["detailsJson"] = json!(unknown_field.to_string());
        let error = merge_gradle_model(&mut Vec::new(), invalid_model, Path::new("/fixture"))
            .expect_err("unknown declaration field");
        assert!(!format!("{error:#}").contains(canary));
        let mut missing = model.clone();
        missing["projects"][0]
            .as_object_mut()
            .expect("project object")
            .remove("detailsJson");
        assert!(merge_gradle_model(&mut Vec::new(), missing, Path::new("/fixture")).is_err());
        let mut full = (0..MAX_JDTLS_MODULES)
            .map(|index| {
                (
                    format!("file:///fixture/existing-{index}"),
                    JdtlsProjectSettings::default(),
                )
            })
            .collect::<Vec<_>>();
        assert!(merge_gradle_model(&mut full, model.clone(), Path::new("/fixture")).is_err());
        assert_eq!(full.len(), MAX_JDTLS_MODULES);
        assert!(full.iter().all(|(_, settings)| settings.gradle.is_none()));
        let mut malformed = model;
        malformed["projects"][0]["detailsJson"] = json!("invalid json");
        assert!(merge_gradle_model(&mut Vec::new(), malformed, Path::new("/fixture")).is_err());
    }

    #[test]
    fn maven_model_preserves_provider_concepts_and_rejects_invalid_snapshots_atomically() {
        let model = json!({
            "version": 1, "rootUri": "file:///fixture/", "cachedOnly": true,
            "reactorOrderKnown": false, "unavailablePomUris": [],
            "lifecycles": [{"id": "default", "phases": ["compile", "test", "verify"]}],
            "projects": [{"pomUri": "file:///fixture/application/pom.xml", "groupId": "evidence",
                "artifactId": "application", "version": "1.0-SNAPSHOT", "packaging": "jar",
                "moduleUris": ["file:///fixture/library/"], "profiles": {"parity": true},
                "inheritedProfileDefinitionsKnown": false, "plugins": [
                    {"groupId": "org.apache.maven.plugins", "artifactId": "maven-surefire-plugin",
                        "version": "3.5.2", "managedOnly": false, "goalsKnown": true,
                        "goalPrefix": "surefire", "goals": [{"name": "test", "phase": "test", "description": "Run tests"}]},
                    {"groupId": "evidence", "artifactId": "missing-plugin", "version": null,
                        "managedOnly": true, "goalsKnown": false, "goals": []}
                ]}]
        });
        let mut projects = vec![("file:///fixture/application/".to_string(), settings())];
        merge_maven_model(&mut projects, model.clone(), Path::new("/fixture"))
            .expect("valid Maven metadata");
        assert_eq!(projects.len(), 1);
        let (uri, settings) = projects.remove(0);
        let (project, partial) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("Maven snapshot");
        assert!(partial);
        let maven = project.maven.expect("Maven facts");
        assert_eq!(maven.pom.path.as_unix_str(), "application/pom.xml");
        assert_eq!(maven.modules[0].path.as_unix_str(), "library");
        assert_eq!(maven.profiles.get("parity"), Some(&true));
        assert_eq!(maven.plugins[0].goals[0].name, "test");
        assert!(maven.plugins[1].managed_only);
        assert!(!maven.plugins[1].goals_known);
        assert_eq!(maven.lifecycles[0].phases, ["compile", "test", "verify"]);
        for (pointer, value) in [
            ("/version", json!(2)),
            ("/cachedOnly", json!(false)),
            ("/reactorOrderKnown", json!(true)),
            ("/rootUri", json!("file:///foreign/")),
            ("/projects/0/pomUri", json!("file:///private/pom.xml")),
            ("/projects/0/moduleUris/0", json!("file:///foreign/module")),
            (
                "/projects/0/moduleUris/0",
                json!("file:///fixture/%1Bmodule"),
            ),
            ("/projects/0/inheritedProfileDefinitionsKnown", json!(true)),
            ("/lifecycles/0/phases", json!(["test", "test"])),
            ("/projects/0/plugins/0/goalPrefix", json!("secret\u{1b}[0m")),
            ("/projects/0/plugins/0/goals/0/name", json!("test\n")),
            ("/projects/0/plugins/0/goalsKnown", json!(false)),
            (
                "/projects/0/plugins/0/goals",
                json!(vec![
                    model["projects"][0]["plugins"][0]["goals"][0].clone();
                    4097
                ]),
            ),
        ] {
            let mut invalid = model.clone();
            *invalid.pointer_mut(pointer).expect("Maven fixture field") = value;
            let mut destination = vec![(
                "file:///fixture/application/".to_string(),
                JdtlsProjectSettings::default(),
            )];
            assert!(
                merge_maven_model(&mut destination, invalid, Path::new("/fixture")).is_err(),
                "{pointer}"
            );
            assert!(destination[0].1.maven.is_none());
            assert!(!destination[0].1.maven_metadata_incomplete);
        }
        let mut unavailable = model;
        unavailable["unavailablePomUris"] = json!(["file:///fixture/missing/pom.xml"]);
        let mut destination = vec![(
            "file:///fixture/application/".to_string(),
            JdtlsProjectSettings::default(),
        )];
        merge_maven_model(&mut destination, unavailable, Path::new("/fixture"))
            .expect("explicit missing cached model");
        assert!(destination[0].1.maven_metadata_incomplete);
    }

    #[test]
    fn maven_parameter_declarations_are_atomic_and_legacy_coverage_is_unknown() {
        let parameter = json!({"name":"skipTests","alias":null,"declaredType":"boolean",
            "required":false,"editable":true,"description":"Skip tests"});
        let model = json!({"version":3,"rootUri":"file:///fixture/","cachedOnly":true,
            "reactorOrderKnown":true,"reactorPomUris":["file:///fixture/application/pom.xml"],
            "unavailablePomUris":[],"lifecycles":[],"projects":[{
                "pomUri":"file:///fixture/application/pom.xml","groupId":"evidence","artifactId":"application",
                "version":"1.0","packaging":"jar","moduleUris":[],"profiles":{},
                "inheritedProfileDefinitionsKnown":true,"profileDefinitions":[],
                "plugins":[{"groupId":"org.apache.maven.plugins","artifactId":"maven-surefire-plugin",
                    "version":"3.5.2","managedOnly":false,"goalsKnown":true,"goalPrefix":"surefire",
                    "goals":[{"name":"test","phase":"test","description":"Run tests","parametersKnown":true,
                        "parameters":[parameter]}]}]}]});
        let mut admitted = Vec::new();
        merge_maven_model(&mut admitted, model.clone(), Path::new("/fixture"))
            .expect("current parameter declarations");
        let original = format!("{:?}", admitted[0].1.maven);
        let goal = &admitted[0]
            .1
            .maven
            .as_ref()
            .expect("Maven metadata")
            .plugins[0]
            .goals[0];
        assert!(goal.parameters_known);
        assert_eq!(goal.parameters[0].declared_type.as_deref(), Some("boolean"));
        assert!(!goal.parameters[0].required);
        assert!(goal.parameters[0].editable);
        for (pointer, value) in [
            ("/version", json!(2)),
            (
                "/projects/0/plugins/0/goals/0/parametersKnown",
                json!(false),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters/0/name",
                json!("../private"),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters/0/alias",
                json!("bad\n"),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters/0/required",
                json!("true"),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters/0/editable",
                json!(null),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters/0/description",
                json!("x".repeat(4097)),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters",
                json!([parameter.clone(), parameter.clone()]),
            ),
            (
                "/projects/0/plugins/0/goals/0/parameters",
                json!(vec![parameter; 257]),
            ),
        ] {
            let mut invalid = model.clone();
            *invalid.pointer_mut(pointer).expect("fixture field") = value;
            assert!(
                merge_maven_model(&mut admitted, invalid, Path::new("/fixture")).is_err(),
                "{pointer}"
            );
            assert_eq!(
                format!("{:?}", admitted[0].1.maven),
                original,
                "failed update preserves current facts"
            );
        }
        let mut foreign_field = model.clone();
        foreign_field["projects"][0]["plugins"][0]["goals"][0]["parameters"][0]["effectiveValue"] =
            json!("private");
        assert!(merge_maven_model(&mut admitted, foreign_field, Path::new("/fixture")).is_err());
        assert_eq!(format!("{:?}", admitted[0].1.maven), original);
        for version in [1, 2] {
            let mut legacy = model.clone();
            legacy["version"] = json!(version);
            let goal = legacy["projects"][0]["plugins"][0]["goals"][0]
                .as_object_mut()
                .expect("goal");
            goal.remove("parametersKnown");
            goal.remove("parameters");
            if version == 1 {
                let project = legacy["projects"][0].as_object_mut().expect("module");
                project.remove("profileDefinitions");
                project["inheritedProfileDefinitionsKnown"] = json!(false);
                let root = legacy.as_object_mut().expect("model");
                root.remove("reactorPomUris");
                root["reactorOrderKnown"] = json!(false);
            }
            let mut admitted = Vec::new();
            merge_maven_model(&mut admitted, legacy, Path::new("/fixture"))
                .expect("legacy unknown coverage");
            let goal = &admitted[0].1.maven.as_ref().expect("Maven").plugins[0].goals[0];
            assert!(!goal.parameters_known);
            assert!(goal.parameters.is_empty());
        }
    }

    #[test]
    fn extended_maven_order_and_profile_origins_validate_before_snapshot_mutation() {
        let model = json!({
            "version": 2, "rootUri": "file:///fixture/", "cachedOnly": true,
            "reactorOrderKnown": true, "reactorPomUris": ["file:///fixture/application/pom.xml"],
            "unavailablePomUris": [], "lifecycles": [],
            "projects": [{"pomUri": "file:///fixture/application/pom.xml", "groupId": "evidence",
                "artifactId": "application", "version": "1.0", "packaging": "jar", "moduleUris": [],
                "profiles": {"parity": false}, "inheritedProfileDefinitionsKnown": true,
                "profileDefinitions": [
                    {"id": "parity", "sourceGroupId": "evidence", "sourceArtifactId": "application", "sourceVersion": "1.0",
                        "sourcePomUri": "file:///fixture/application/pom.xml", "inherited": false, "activeAtSource": false},
                    {"id": "parity", "sourceGroupId": "evidence", "sourceArtifactId": "parent", "sourceVersion": "1.0",
                        "sourcePomUri": "file:///fixture/pom.xml", "inherited": true, "activeAtSource": true},
                    {"id": "external", "sourceGroupId": "evidence", "sourceArtifactId": "external", "sourceVersion": "1.0",
                        "inherited": true, "activeAtSource": false}
                ], "plugins": []}]
        });
        let mut projects = Vec::new();
        merge_maven_model(&mut projects, model.clone(), Path::new("/fixture"))
            .expect("extended model");
        let (uri, settings) = projects.remove(0);
        let (project, partial) = project_from_settings(
            &uri,
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("extended snapshot");
        assert!(partial, "cached metadata does not certify import freshness");
        let maven = project.maven.expect("Maven metadata");
        assert_eq!(maven.reactor_position, Some(0));
        assert!(maven.profile_definitions_known);
        assert_eq!(maven.profile_definitions.len(), 3);
        assert_ne!(
            maven.profile_definitions[0].source_pom,
            maven.profile_definitions[1].source_pom
        );
        assert!(maven.profile_definitions[1].active_at_source);
        assert!(maven.profile_definitions[2].source_pom.is_none());
        for (pointer, value) in [
            ("/reactorPomUris", json!([])),
            ("/reactorPomUris", json!(["file:///foreign/pom.xml"])),
            ("/reactorPomUris", json!(["file:///fixture/absent/pom.xml"])),
            (
                "/reactorPomUris",
                json!([
                    "file:///fixture/application/pom.xml",
                    "file:///fixture/application/pom.xml"
                ]),
            ),
            ("/reactorOrderKnown", json!(false)),
            (
                "/unavailablePomUris",
                json!(["file:///fixture/missing/pom.xml"]),
            ),
            (
                "/projects/0/profileDefinitions/1/sourcePomUri",
                json!("file:///foreign/pom.xml"),
            ),
            (
                "/projects/0/profileDefinitions/0/sourceArtifactId",
                json!("different"),
            ),
            ("/projects/0/profileDefinitions/1/inherited", json!(false)),
            ("/projects/0/profileDefinitions/2/id", json!("secret\u{1b}")),
            (
                "/projects/0/profileDefinitions",
                json!(vec![
                    model["projects"][0]["profileDefinitions"][0].clone();
                    129
                ]),
            ),
        ] {
            let mut invalid = model.clone();
            *invalid
                .pointer_mut(pointer)
                .expect("extended fixture field") = value;
            let mut destination = vec![(
                "file:///fixture/application/".into(),
                JdtlsProjectSettings::default(),
            )];
            assert!(
                merge_maven_model(&mut destination, invalid, Path::new("/fixture")).is_err(),
                "{pointer}"
            );
            assert!(destination[0].1.maven.is_none());
        }
        let mut unknown = model;
        unknown["reactorOrderKnown"] = json!(false);
        unknown["reactorPomUris"] = json!([]);
        unknown["reactorOrderProblem"] = json!("invalid-cached-reactor");
        let mut projects = Vec::new();
        merge_maven_model(&mut projects, unknown.clone(), Path::new("/fixture"))
            .expect("explicit invalid reactor");
        assert_eq!(
            projects[0]
                .1
                .maven
                .as_ref()
                .expect("metadata")
                .reactor_order_problem
                .as_deref(),
            Some("invalid-cached-reactor")
        );
        unknown["reactorOrderProblem"] = json!("private raw error");
        assert!(merge_maven_model(&mut Vec::new(), unknown, Path::new("/fixture")).is_err());
    }

    #[test]
    fn jdtls_maven_attributes_are_validated_before_projection() {
        let settings = serde_json::from_value(serde_json::json!({
            "org.eclipse.jdt.ls.core.sourcePaths": [],
            "org.eclipse.jdt.ls.core.classpathEntries": [
                {
                    "kind": 1,
                    "path": "/private/cache/junit.jar",
                    "attributes": {
                        "maven.groupId": "org.junit.jupiter",
                        "maven.artifactId": "junit-jupiter-api",
                        "maven.version": "5.11.4",
                        "maven.scope": "test"
                    }
                },
                {
                    "kind": 1,
                    "path": "/private/cache/unsafe.jar",
                    "attributes": {
                        "maven.groupId": "../private",
                        "maven.artifactId": "unsafe",
                        "maven.version": "1"
                    }
                },
                {"kind": 1, "path": "/private/cache/plain.jar"}
            ]
        }))
        .expect("Maven classpath settings");
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("validated Maven classpath");
        assert!(partial);
        assert_eq!(project.library_entries, 3);
        assert_eq!(
            project.library_artifacts,
            [JdtlsLibraryArtifact {
                coordinate: "org.junit.jupiter:junit-jupiter-api:5.11.4".to_string(),
                scope: Some("test".to_string()),
            }]
        );
        assert_eq!(
            project.library_files,
            [
                JdtlsLibraryFile {
                    name: "plain.jar".to_string(),
                    gradle_source_sets: Vec::new(),
                },
                JdtlsLibraryFile {
                    name: "unsafe.jar".to_string(),
                    gradle_source_sets: Vec::new(),
                },
            ]
        );
        let shared = project_to_proto(&project);
        assert!(!format!("{shared:?}").contains("/private"));
    }

    #[test]
    fn jdtls_gradle_source_sets_are_validated_before_projection() {
        let settings = serde_json::from_value(serde_json::json!({
            "org.eclipse.jdt.ls.core.sourcePaths": [],
            "org.eclipse.jdt.ls.core.classpathEntries": [
                {
                    "kind": 1,
                    "path": "/private/cache/main.jar",
                    "attributes": {"gradle_used_by_scope": "main,test"}
                },
                {
                    "kind": 1,
                    "path": "/private/cache/unsafe.jar",
                    "attributes": {"gradle_used_by_scope": "test\n/private"}
                }
            ]
        }))
        .expect("Gradle classpath settings");
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("validated Gradle classpath");
        assert!(partial);
        assert_eq!(project.library_entries, 2);
        assert_eq!(project.library_files[0].name, "main.jar");
        assert_eq!(
            project.library_files[0].gradle_source_sets,
            ["main", "test"]
        );
        assert!(project.library_files[1].gradle_source_sets.is_empty());
        let shared = project_to_proto(&project);
        assert_eq!(shared.library_source_sets.len(), 1);
        assert!(!format!("{shared:?}").contains("/private"));
    }

    #[test]
    fn jdtls_paths_and_dependencies_are_bounded_to_the_host_worktree() {
        let worktree_id = WorktreeId::from_usize(1);
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            settings(),
            Path::new("/fixture"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("current JDTLS response");
        assert!(!partial);
        assert_eq!(project.module_root.path.as_unix_str(), "application");
        assert_eq!(project.source_roots.len(), 2);
        assert_eq!(project.project_references, ["library"]);
        assert_eq!(project.library_entries, 1);
        assert_eq!(project.library_files[0].name, "customer.jar");
        assert!(project.project_vm_known);
        assert_eq!(project.project_vm_java_major, None);
        assert_eq!(project.java_source_level, Some(17));
        assert_eq!(project.java_target_level, Some(21));

        let mut invalid_target = settings();
        invalid_target.java_target_level = Some("/private/host/jdk".to_string());
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            invalid_target,
            Path::new("/fixture"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("invalid provider target level does not escape the snapshot");
        assert!(partial);
        assert_eq!(project.java_source_level, Some(17));
        assert_eq!(project.java_target_level, None);

        let mut invalid_level = settings();
        invalid_level.java_source_level = Some("/private/host/jdk".to_string());
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            invalid_level,
            Path::new("/fixture"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("invalid provider source level does not escape the snapshot");
        assert!(partial);
        assert_eq!(project.java_source_level, None);

        let mut outside = settings();
        outside.source_paths = Some(vec![
            "/fixture/application/../../private/source".to_string(),
            "/fixture/application/src/main/java".to_string(),
        ]);
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            outside,
            Path::new("/fixture"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("partially confined JDTLS response");
        assert!(partial);
        assert_eq!(project.source_roots.len(), 1);
        assert!(
            project_from_settings(
                "file:///private/application",
                settings(),
                Path::new("/fixture"),
                worktree_id,
                PathStyle::Unix,
            )
            .is_err()
        );

        let mut unsafe_libraries = settings();
        unsafe_libraries.classpath_entries = Some(vec![
            JdtlsClasspathEntry {
                kind: 1,
                path: Some("/fixture/private.jar".to_string()),
                attributes: None,
            },
            JdtlsClasspathEntry {
                kind: 1,
                path: Some("/private/cache/secret\nname.jar".to_string()),
                attributes: None,
            },
            JdtlsClasspathEntry {
                kind: 1,
                path: Some("/private/cache/junit-jupiter-5.11.4.jar".to_string()),
                attributes: None,
            },
        ]);
        let (project, _) = project_from_settings(
            "file:///fixture/application",
            unsafe_libraries,
            Path::new("/fixture"),
            worktree_id,
            PathStyle::Unix,
        )
        .expect("bounded library labels");
        assert_eq!(project.library_entries, 3);
        assert_eq!(project.library_files[0].name, "junit-jupiter-5.11.4.jar");
        assert!(valid_library_file_name("junit-jupiter-5.11.4.jar"));
        assert!(!valid_library_file_name("/private/cache/customer.jar"));
        assert!(!valid_library_file_name("customer\n.jar"));
        assert!(!valid_library_file_name("customer.zip"));
        assert_eq!(
            external_library_file_name(
                "C:\\cache\\junit-jupiter-5.11.4.jar",
                Path::new("C:\\fixture"),
                worktree_id,
                PathStyle::Windows,
            ),
            Some("junit-jupiter-5.11.4.jar")
        );
    }

    #[test]
    fn jdtls_java_compatibility_levels_are_bounded_numbers() {
        assert_eq!(parse_java_compatibility_level("17"), Some(17));
        assert_eq!(parse_java_compatibility_level("1.8"), Some(8));
        for invalid in ["", "0", "100", "17.0", "17\n/private", "/private/jdk"] {
            assert_eq!(parse_java_compatibility_level(invalid), None);
        }
    }

    #[test]
    fn jdtls_launch_java_is_distinct_from_project_compatibility() {
        let directory = tempfile::tempdir().expect("launch JDK fixture");
        std::fs::write(
            directory.path().join("release"),
            "JAVA_VERSION=\"25.0.1\"\n",
        )
        .expect("launch JDK release metadata");
        let java =
            directory
                .path()
                .join("bin")
                .join(if cfg!(windows) { "java.exe" } else { "java" });
        let marker = "-Declipse.application=org.eclipse.jdt.ls.core.id1";
        let mut binary = LanguageServerBinary {
            path: directory.path().join("java-lsp-proxy"),
            arguments: vec![
                directory.path().as_os_str().into(),
                java.as_os_str().into(),
                marker.into(),
            ],
            env: None,
        };
        let home = jdtls_launch_java_home(&binary).expect("managed Java proxy command");
        assert_eq!(home, directory.path());
        let mut settings = settings();
        settings.server_launch_java_major =
            home.to_str().and_then(project_vm_java_major_from_location);
        let (project, _) = project_from_settings(
            "file:///fixture/application",
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("independent JDK facts");
        assert_eq!(project.server_launch_java_major, Some(25));
        assert_eq!(project.project_vm_java_major, None);
        assert_eq!(project.java_source_level, Some(17));
        assert_eq!(project.java_target_level, Some(21));
        let shared = project_to_proto(&project);
        assert_eq!(shared.server_launch_java_major, 25);
        assert!(!format!("{shared:?}").contains(&directory.path().to_string_lossy().to_string()));

        binary.arguments[1] = directory.path().join("bin/jdtls").into_os_string();
        assert_eq!(jdtls_launch_java_home(&binary), None);
        binary.arguments[1] = "java".into();
        assert_eq!(jdtls_launch_java_home(&binary), None);
        binary.arguments[1] = java.as_os_str().into();
        binary.path = directory.path().join("custom-proxy");
        assert_eq!(jdtls_launch_java_home(&binary), None);
        binary.path = java;
        binary.arguments = vec![marker.into()];
        assert_eq!(jdtls_launch_java_home(&binary), Some(directory.path()));
        binary.arguments.clear();
        assert_eq!(jdtls_launch_java_home(&binary), None);
        binary.path = directory.path().join("java-lsp-proxy");
        binary.arguments = vec![marker.into()];
        assert_eq!(jdtls_launch_java_home(&binary), None);
        assert_eq!(
            project_vm_java_major_from_location("/missing/launch-jdk"),
            None
        );
    }

    #[test]
    fn jdtls_project_vm_java_version_is_reduced_on_the_host() {
        let directory = tempfile::tempdir().expect("temporary JDK fixture");
        std::fs::write(
            directory.path().join("release"),
            "JAVA_VERSION=\"17.0.18\"\n",
        )
        .expect("write JDK release metadata");
        let location = directory.path().to_str().expect("UTF-8 JDK fixture path");
        assert_eq!(project_vm_java_major_from_location(location), Some(17));

        let mut settings = settings();
        settings.project_vm_java_major = project_vm_java_major_from_location(location);
        let (project, partial) = project_from_settings(
            "file:///fixture/application",
            settings,
            Path::new("/fixture"),
            WorktreeId::from_usize(1),
            PathStyle::Unix,
        )
        .expect("project VM fixture");
        assert!(!partial);
        assert_eq!(project.project_vm_java_major, Some(17));
        let shared = project_to_proto(&project);
        assert_eq!(shared.project_vm_java_major, 17);
        assert!(!format!("{shared:?}").contains(location));
        std::fs::write(
            directory.path().join("release"),
            "JAVA_VERSION=\"1.8.0_452\"\n",
        )
        .expect("update JDK release metadata");
        assert_eq!(project_vm_java_major_from_location(location), Some(8));
        std::fs::write(
            directory.path().join("release"),
            "JAVA_VERSION=\"/private/jdk\"\n",
        )
        .expect("update invalid release metadata");
        assert_eq!(project_vm_java_major_from_location(location), None);
    }

    #[test]
    fn kotlin_export_home_paths_are_confined_to_the_worktree() {
        let worktree_id = WorktreeId::from_usize(1);
        let worktree_root = paths::home_dir().join("kotlin-export-fixture");
        let path_style = if cfg!(windows) {
            PathStyle::Windows
        } else {
            PathStyle::Unix
        };
        let module = confined_kotlin_export_path(
            "<HOME>/kotlin-export-fixture/application",
            &worktree_root,
            worktree_id,
            path_style,
        )
        .expect("home-relative provider module");
        assert_eq!(module.path.as_unix_str(), "application");
        assert!(
            confined_kotlin_export_path(
                "<HOME>/kotlin-export-fixture/../outside",
                &worktree_root,
                worktree_id,
                path_style,
            )
            .is_none()
        );
        assert!(
            confined_kotlin_export_path(
                "<HOME>/other-project/application",
                &worktree_root,
                worktree_id,
                path_style,
            )
            .is_none()
        );
        assert!(
            confined_kotlin_export_path(
                "<MAVEN_REPO>/library.jar",
                &worktree_root,
                worktree_id,
                path_style,
            )
            .is_none()
        );
    }

    #[test]
    fn stale_jdtls_generations_cannot_restore_removed_projects() {
        let mut state = JdtlsProjectState::default();
        let server = LanguageServerId(1);
        let worktree = WorktreeId::from_usize(2);
        let old = state.begin(server, worktree).expect("first generation");
        assert_eq!(state.pending_query(), Some(old));
        let current = state.begin(server, worktree).expect("second generation");
        assert_eq!(state.pending_query(), Some(current));
        assert!(!state.finish(old, Vec::new(), false));
        assert_eq!(state.pending_query(), Some(current));
        assert!(state.finish(current, Vec::new(), true));
        assert!(state.pending_query().is_none());
        assert_eq!(state.snapshot().expect("current state").generation, 2);
        assert!(state.snapshot_is_current());
        let failed = state.begin(server, worktree).expect("failed generation");
        assert_eq!(state.pending_query(), Some(failed));
        assert!(!state.snapshot_is_current());
        assert!(state.fail(failed));
        assert!(state.pending_query().is_none());
        assert_eq!(state.failed_query(), Some(failed));
        assert_eq!(state.snapshot().expect("stale state").generation, 2);
        assert!(!state.snapshot_is_current());
        state.invalidate_worktree(worktree);
        assert!(state.pending_query().is_none());
        assert!(state.snapshot().is_none());
        assert!(state.failed_query().is_none());
        assert!(!state.finish(old, Vec::new(), false));

        let later = state
            .begin(server, worktree)
            .expect("replacement generation");
        state
            .mark_stale()
            .expect("new import invalidates pending reply");
        assert!(!state.finish(later, Vec::new(), false));
        let later = state
            .begin(server, worktree)
            .expect("new import generation");
        state.invalidate_server(server);
        assert!(!state.finish(later, Vec::new(), false));
        assert!(state.snapshot().is_none());

        let before_disconnect = state
            .begin(server, worktree)
            .expect("pre-disconnect generation");
        assert!(state.finish(before_disconnect, Vec::new(), false));
        state
            .invalidate_remote_connection(worktree)
            .expect("connection loss invalidates provider generation");
        assert!(state.snapshot().is_none());
        assert!(!state.finish(before_disconnect, Vec::new(), false));
        let after_reconnect = state.begin(server, worktree).expect("new host generation");
        assert!(after_reconnect.generation() > before_disconnect.generation());
    }

    #[test]
    fn selected_import_phase_cannot_publish_or_replace_a_read_generation() {
        let mut state = JdtlsProjectState::default();
        let server = LanguageServerId(1);
        let worktree = WorktreeId::from_usize(3);
        let read = state.begin(server, worktree).expect("initial read");
        assert!(state.finish(read, Vec::new(), false));
        let phase = state
            .begin_import_operation(server, worktree)
            .expect("explicit import");
        assert!(state.is_refreshing());
        assert!(!state.snapshot_is_current());
        assert_eq!(state.pending_query(), None);
        assert!(state.begin(server, worktree).is_err());
        assert!(!state.finish(phase, Vec::new(), false));
        assert!(!state.finish_import_operation(read));
        assert!(state.finish_import_operation(phase));
        assert!(
            !state.snapshot_is_current(),
            "a receipt is not a current model"
        );
        let read = state.begin(server, worktree).expect("separate read");
        assert!(state.finish(read, Vec::new(), false));
        assert!(state.snapshot_is_current());
        for invalidation in 0..4 {
            let phase = state
                .begin_import_operation(server, worktree)
                .expect("new phase");
            match invalidation {
                0 => state.mark_stale().expect("changed build inputs"),
                1 => state.invalidate_server(server),
                2 => state.invalidate_worktree(worktree),
                _ => assert!(state.fail(phase)),
            }
            assert_eq!(state.import_operation(), None);
            assert!(!state.finish_import_operation(phase));
            assert!(!state.finish(phase, Vec::new(), false));
        }
    }

    #[test]
    fn kotlin_import_failure_stays_bound_to_its_server_and_clears_on_retry() {
        let mut state = KotlinWorkspaceState::default();
        let server = LanguageServerId(1);
        let other_server = LanguageServerId(2);
        let worktree = WorktreeId::from_usize(3);
        state
            .mark_import_failed(server)
            .expect("failed provider import");
        assert_eq!(state.failed_import(), Some(server));
        state.invalidate_server(other_server);
        assert_eq!(state.failed_import(), Some(server));
        state.begin(server, worktree).expect("manual retry");
        assert_eq!(state.failed_import(), None);
        state
            .mark_import_failed(server)
            .expect("second failed provider import");
        state.invalidate_server(server);
        assert_eq!(state.failed_import(), None);
    }

    #[test]
    fn early_kotlin_import_success_waits_for_a_current_server_refresh() {
        let mut state = KotlinWorkspaceState::default();
        let server = LanguageServerId(1);
        let other_server = LanguageServerId(2);
        let worktree = WorktreeId::from_usize(3);
        state
            .mark_import_succeeded(server)
            .expect("early successful import");
        assert_eq!(state.pending_import_refresh(), Some(server));
        state.invalidate_server(other_server);
        assert_eq!(state.pending_import_refresh(), Some(server));
        let refresh = state.begin(server, worktree).expect("server became ready");
        assert_eq!(state.pending_import_refresh(), None);
        assert!(state.finish(refresh, Vec::new(), false));
        assert!(state.snapshot_is_current());

        state
            .mark_import_succeeded(server)
            .expect("replacement successful import");
        state.invalidate_server(server);
        assert_eq!(state.pending_import_refresh(), None);
    }
}
