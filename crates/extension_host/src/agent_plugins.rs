use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};
use url::Url;

pub const MANIFEST_SCHEMA: &str = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";
pub const MCP_SCHEMA: &str = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackageManifest {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ServerDeclaration {
    #[serde(rename = "stdio")]
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        cwd: Option<String>,
    },
    #[serde(rename = "streamable-http")]
    StreamableHttp {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
    #[serde(rename = "sse")]
    Sse {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct ServerDeclarations {
    pub servers: BTreeMap<String, ServerDeclaration>,
    pub diagnostics: Vec<String>,
}

pub fn parse_manifest(bytes: &[u8]) -> Result<(PackageManifest, Vec<String>)> {
    let value: Value = serde_json::from_slice(bytes).context("Invalid plugin.json")?;
    let object = value
        .as_object()
        .context("plugin.json must contain an object")?;
    ensure!(
        object.get("$schema").and_then(Value::as_str) == Some(MANIFEST_SCHEMA),
        "Unsupported Agent Plugins manifest schema"
    );
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .context("Package name must be a string")?;
    ensure!(
        valid_package_name(name),
        "Invalid Agent Plugins package name"
    );
    for key in [
        "version",
        "description",
        "homepage",
        "repository",
        "license",
    ] {
        if let Some(value) = object.get(key) {
            ensure!(value.is_string(), "Manifest {key} must be a string");
        }
    }
    if let Some(author) = object.get("author") {
        let author = author
            .as_object()
            .context("Manifest author must be an object")?;
        for (key, value) in author {
            ensure!(
                ["name", "email", "url"].contains(&key.as_str()) && value.is_string(),
                "Invalid manifest author field {key}"
            );
        }
    }
    if let Some(keywords) = object.get("keywords") {
        ensure!(
            keywords
                .as_array()
                .is_some_and(|values| values.iter().all(Value::is_string)),
            "Manifest keywords must be an array of strings"
        );
    }
    let mut diagnostics = Vec::new();
    for key in object.keys() {
        if ![
            "$schema",
            "name",
            "version",
            "description",
            "author",
            "homepage",
            "repository",
            "license",
            "keywords",
            "extensions",
        ]
        .contains(&key.as_str())
        {
            diagnostics.push(format!("Ignoring unknown manifest field {key}"));
        }
    }
    // Unimplemented namespaces have no validation semantics in this client.
    if object
        .get("extensions")
        .is_some_and(|value| !value.is_object())
    {
        diagnostics.push("Ignoring non-object manifest extensions".into());
    }
    Ok((
        PackageManifest {
            name: name.into(),
            version: object
                .get("version")
                .and_then(Value::as_str)
                .map(str::to_owned),
            description: object
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
        },
        diagnostics,
    ))
}

fn valid_package_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
        })
        && name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && name
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
        && !name.contains("--")
        && !name.contains("..")
}

pub fn parse_servers(bytes: &[u8]) -> Result<ServerDeclarations> {
    let value: Value = serde_json::from_slice(bytes).context("Invalid mcp.json")?;
    let object = value
        .as_object()
        .context("mcp.json must contain an object")?;
    ensure!(
        object.get("$schema").and_then(Value::as_str) == Some(MCP_SCHEMA),
        "Unsupported MCP package schema"
    );
    ensure!(
        object
            .keys()
            .all(|key| ["$schema", "mcpServers"].contains(&key.as_str())),
        "Unknown mcp.json field"
    );
    let servers = object
        .get("mcpServers")
        .and_then(Value::as_object)
        .context("mcpServers must be an object")?;
    let mut result = ServerDeclarations::default();
    for (name, value) in servers {
        let parsed = validate_server_field_types(value)
            .and_then(|()| {
                serde_json::from_value::<ServerDeclaration>(value.clone())
                    .map_err(anyhow::Error::from)
            })
            .and_then(|server| {
                server.validate()?;
                Ok(server)
            });
        match parsed {
            Ok(ServerDeclaration::Sse { .. }) => result
                .diagnostics
                .push(format!("Server {name}: legacy HTTP+SSE is unsupported")),
            Ok(server) => {
                result.servers.insert(name.clone(), server);
            }
            Err(error) => result.diagnostics.push(format!("Server {name}: {error:#}")),
        }
    }
    Ok(result)
}

fn validate_server_field_types(value: &Value) -> Result<()> {
    let object = value.as_object().context("Server must be an object")?;
    for key in ["command", "cwd", "url"] {
        if let Some(value) = object.get(key) {
            ensure!(value.is_string(), "Server {key} must be a string");
        }
    }
    Ok(())
}

impl ServerDeclaration {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Stdio {
                command, env, cwd, ..
            } => {
                ensure!(
                    !command.is_empty()
                        && (!command.chars().any(char::is_whitespace) || command.starts_with("./")),
                    "Command must be one executable token"
                );
                if command.contains('/') || command.contains('\\') {
                    validate_relative_path(command)?;
                }
                ensure!(
                    !env.contains_key("PLUGIN_ROOT") && !env.contains_key("PLUGIN_DATA"),
                    "Reserved plugin environment variable"
                );
                if let Some(cwd) = cwd {
                    if cwd.starts_with("./") {
                        validate_relative_path(cwd)?;
                    } else {
                        let suffix = cwd
                            .strip_prefix("${PLUGIN_ROOT}")
                            .or_else(|| cwd.strip_prefix("${PLUGIN_DATA}"))
                            .context("Invalid plugin working directory")?;
                        ensure!(
                            suffix.is_empty() || suffix.starts_with('/'),
                            "Invalid plugin working directory"
                        );
                        validate_relative_path(&format!("./{}", suffix.trim_start_matches('/')))?;
                    }
                }
            }
            Self::StreamableHttp { url, headers } | Self::Sse { url, headers } => {
                ensure!(
                    !url.chars()
                        .any(|character| character.is_whitespace() || character.is_control())
                        && !url
                            .split_once("://")
                            .map(|(_, remainder)| remainder
                                .split('/')
                                .next()
                                .unwrap_or("")
                                .contains('@'))
                            .unwrap_or(false),
                    "MCP endpoint must not include whitespace or user information"
                );
                let endpoint = Url::parse(url).context("Invalid MCP endpoint")?;
                ensure!(
                    endpoint.username().is_empty()
                        && endpoint.password().is_none()
                        && endpoint.fragment().is_none(),
                    "MCP endpoint must not include credentials or a fragment"
                );
                let loopback = match endpoint.host() {
                    Some(url::Host::Domain(domain)) => domain == "localhost",
                    Some(url::Host::Ipv4(address)) => address.is_loopback(),
                    Some(url::Host::Ipv6(address)) => address.is_loopback(),
                    None => false,
                };
                ensure!(
                    endpoint.host().is_some()
                        && (endpoint.scheme() == "https"
                            || (endpoint.scheme() == "http" && loopback)),
                    "MCP endpoint requires HTTPS except on loopback"
                );
                let mut names = std::collections::BTreeSet::new();
                for (name, value) in headers {
                    ensure!(
                        !name.is_empty()
                            && name.bytes().all(|byte| byte.is_ascii_alphanumeric()
                                || b"!#$%&'*+-.^_`|~".contains(&byte)),
                        "Invalid HTTP header name"
                    );
                    ensure!(
                        names.insert(name.to_ascii_lowercase()),
                        "Duplicate case-insensitive HTTP header"
                    );
                    ensure!(
                        value
                            .bytes()
                            .all(|byte| byte == b'\t' || (byte >= 32 && byte != 127)),
                        "Invalid HTTP header value"
                    );
                }
            }
        }
        Ok(())
    }
}

pub fn validate_relative_path(path: &str) -> Result<PathBuf> {
    ensure!(
        path.starts_with("./") && !path.contains('\\'),
        "Package paths must start with ./"
    );
    let path = Path::new(path);
    ensure!(
        path.components().all(|part| matches!(
            part,
            Component::CurDir | Component::ParentDir | Component::Normal(_)
        )),
        "Package path escapes its root"
    );
    Ok(path.to_path_buf())
}

pub fn expand_value(value: &str, root: &Path, data: &Path) -> String {
    let mut expanded = String::new();
    let mut remainder = value;
    while let Some(index) = remainder.find("${PLUGIN_") {
        let (prefix, suffix) = remainder.split_at(index);
        expanded.push_str(prefix);
        if let Some(rest) = suffix.strip_prefix("${PLUGIN_ROOT}") {
            expanded.push_str(&root.to_string_lossy());
            remainder = rest;
        } else if let Some(rest) = suffix.strip_prefix("${PLUGIN_DATA}") {
            expanded.push_str(&data.to_string_lossy());
            remainder = rest;
        } else {
            expanded.push_str("${PLUGIN_");
            remainder = &suffix[9..];
        }
    }
    expanded.push_str(remainder);
    expanded
}

pub fn contained_path(root: &Path, path: &Path) -> Result<PathBuf> {
    let root = root.canonicalize().context("Cannot resolve package root")?;
    let resolved = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve {}", path.display()))?;
    ensure!(
        resolved.starts_with(&root),
        "Package path escapes its root: {}",
        path.display()
    );
    Ok(resolved)
}

#[derive(Clone, Debug)]
pub struct SnapshotFile {
    pub bytes: Vec<u8>,
    pub executable: bool,
}

#[derive(Clone, Debug)]
pub struct PackageSnapshot {
    pub manifest: PackageManifest,
    pub source: PathBuf,
    pub files: BTreeMap<PathBuf, SnapshotFile>,
    pub directories: std::collections::BTreeSet<PathBuf>,
    pub links: BTreeMap<PathBuf, PathBuf>,
    pub servers: BTreeMap<String, ServerDeclaration>,
    pub diagnostics: Vec<String>,
    pub revision: String,
}

#[derive(Clone, Copy)]
pub struct SnapshotLimits {
    pub file_bytes: u64,
    pub total_bytes: u64,
    pub entries: usize,
}

impl Default for SnapshotLimits {
    fn default() -> Self {
        Self {
            file_bytes: 64 * 1024 * 1024,
            total_bytes: 512 * 1024 * 1024,
            entries: 10_000,
        }
    }
}

async fn read_snapshot_file(
    filesystem: &dyn fs::Fs,
    root: &Path,
    path: &Path,
    limit: u64,
) -> Result<SnapshotFile> {
    use std::io::Read as _;
    let resolved = filesystem.canonicalize(path).await?;
    ensure!(
        resolved.starts_with(root),
        "Package path escapes its root: {}",
        path.display()
    );
    let before = filesystem
        .metadata(&resolved)
        .await?
        .context("Package file disappeared")?;
    ensure!(
        !before.is_dir && !before.is_fifo && before.len <= limit,
        "Package file is not regular or exceeds the size limit: {}",
        path.display()
    );
    let reader = filesystem.open_sync(&resolved).await?;
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "Package file exceeds the size limit"
    );
    let after = filesystem
        .metadata(&resolved)
        .await?
        .context("Package file disappeared during inspection")?;
    ensure!(
        filesystem.canonicalize(path).await? == resolved
            && before.inode == after.inode
            && before.len == after.len
            && before.mtime == after.mtime,
        "Package file changed during inspection: {}",
        path.display()
    );
    Ok(SnapshotFile {
        bytes,
        executable: before.is_executable,
    })
}

pub async fn capture_package(
    filesystem: &dyn fs::Fs,
    source: &Path,
    limits: SnapshotLimits,
) -> Result<PackageSnapshot> {
    use futures::StreamExt as _;
    use sha2::{Digest as _, Sha256};
    let root = filesystem
        .canonicalize(source)
        .await
        .context("Cannot resolve package directory")?;
    ensure!(
        filesystem.is_dir(&root).await,
        "Package source must be a directory"
    );
    let manifest_file = read_snapshot_file(
        filesystem,
        &root,
        &root.join("plugin.json"),
        limits.file_bytes.min(1024 * 1024),
    )
    .await?;
    let (manifest, mut diagnostics) = parse_manifest(&manifest_file.bytes)?;
    let mut files = BTreeMap::new();
    let mut directories = std::collections::BTreeSet::new();
    let mut links = BTreeMap::new();
    let mut pending = vec![(PathBuf::new(), Vec::<PathBuf>::new())];
    let mut total_bytes = 0u64;
    let mut count = 0usize;
    while let Some((relative, mut ancestors)) = pending.pop() {
        count += 1;
        ensure!(count <= limits.entries, "Package exceeds the entry limit");
        let path = root.join(&relative);
        let resolved = match filesystem.canonicalize(&path).await {
            Ok(resolved) if resolved.starts_with(&root) => resolved,
            Ok(_) => {
                diagnostics.push(format!(
                    "Ignoring escaped package path {}",
                    relative.display()
                ));
                continue;
            }
            Err(error) => {
                diagnostics.push(format!("Cannot resolve {}: {error}", relative.display()));
                continue;
            }
        };
        if filesystem
            .metadata(&path)
            .await?
            .is_some_and(|metadata| metadata.is_symlink)
        {
            links.insert(
                relative.clone(),
                resolved.strip_prefix(&root)?.to_path_buf(),
            );
        }
        let metadata = filesystem
            .metadata(&resolved)
            .await?
            .context("Package entry disappeared")?;
        if metadata.is_dir {
            if ancestors.contains(&resolved) {
                diagnostics.push(format!("Ignoring directory cycle {}", relative.display()));
                continue;
            }
            ancestors.push(resolved);
            directories.insert(relative.clone());
            let mut entries = filesystem.read_dir(&path).await?;
            while let Some(entry) = entries.next().await {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .context("Package entry has no file name")?;
                pending.push((relative.join(name), ancestors.clone()));
                ensure!(
                    pending.len().saturating_add(count) <= limits.entries,
                    "Package exceeds the entry limit"
                );
            }
        } else {
            let captured = if relative == Path::new("plugin.json") {
                let current =
                    read_snapshot_file(filesystem, &root, &path, limits.file_bytes).await?;
                ensure!(
                    current.bytes == manifest_file.bytes,
                    "Manifest changed during inspection"
                );
                current
            } else {
                match read_snapshot_file(filesystem, &root, &path, limits.file_bytes).await {
                    Ok(file) => file,
                    Err(error) => {
                        diagnostics
                            .push(format!("Cannot capture {}: {error:#}", relative.display()));
                        continue;
                    }
                }
            };
            total_bytes = total_bytes
                .checked_add(captured.bytes.len() as u64)
                .context("Package size overflow")?;
            ensure!(
                total_bytes <= limits.total_bytes,
                "Package exceeds the total size limit"
            );
            files.insert(relative, captured);
        }
    }
    ensure!(
        files.contains_key(Path::new("plugin.json")),
        "Manifest disappeared during inspection"
    );
    let mut servers = BTreeMap::new();
    if let Some(config) = files.get(Path::new("mcp.json")) {
        match parse_servers(&config.bytes) {
            Ok(parsed) => {
                servers = parsed.servers;
                diagnostics.extend(parsed.diagnostics);
            }
            Err(error) => diagnostics.push(format!("MCP component unavailable: {error:#}")),
        }
    } else if directories.contains(Path::new("mcp.json")) {
        diagnostics.push("MCP component must be a regular mcp.json file".into());
    }
    if files.contains_key(Path::new("skills")) {
        diagnostics.push("Skills component must be a directory".into());
    }
    let mut invalid_servers = Vec::new();
    for (name, server) in &servers {
        if let ServerDeclaration::Stdio { command, .. } = server
            && command.starts_with("./")
        {
            let resolved = filesystem.canonicalize(&root.join(command)).await;
            let captured = resolved
                .as_ref()
                .ok()
                .and_then(|path| path.strip_prefix(&root).ok())
                .is_some_and(|path| files.contains_key(path));
            if !captured {
                diagnostics.push(format!(
                    "Server {name}: executable was not captured within the package"
                ));
                invalid_servers.push(name.clone());
            }
        }
    }
    for name in invalid_servers {
        servers.remove(&name);
    }
    links.retain(|_, target| files.contains_key(target) || directories.contains(target));
    let mut fingerprint = Sha256::new();
    for (link, target) in &links {
        fingerprint.update([2]);
        for path in [link, target] {
            let bytes = path.as_os_str().as_encoded_bytes();
            fingerprint.update((bytes.len() as u64).to_le_bytes());
            fingerprint.update(bytes);
        }
    }
    for directory in &directories {
        let bytes = directory.as_os_str().as_encoded_bytes();
        fingerprint.update([0]);
        fingerprint.update((bytes.len() as u64).to_le_bytes());
        fingerprint.update(bytes);
    }
    for (path, file) in &files {
        let bytes = path.as_os_str().as_encoded_bytes();
        fingerprint.update([1]);
        fingerprint.update((bytes.len() as u64).to_le_bytes());
        fingerprint.update(bytes);
        fingerprint.update([u8::from(file.executable)]);
        fingerprint.update((file.bytes.len() as u64).to_le_bytes());
        fingerprint.update(&file.bytes);
    }
    Ok(PackageSnapshot {
        manifest,
        source: root,
        files,
        directories,
        links,
        servers,
        diagnostics,
        revision: format!("{:x}", fingerprint.finalize()),
    })
}

pub async fn prepare_server_settings(
    filesystem: &dyn fs::Fs,
    directory: &Path,
    package: &InstalledPackage,
    name: &str,
) -> Result<project::project_settings::ContextServerSettings> {
    use project::project_settings::ContextServerSettings;
    let root = filesystem
        .canonicalize(&package.snapshot_path(directory))
        .await?;
    let snapshot = capture_package(filesystem, &root, SnapshotLimits::default()).await?;
    ensure!(
        snapshot.revision == package.revision && snapshot.servers == package.servers,
        "Installed package changed; update and review it before activation"
    );
    let server = package
        .servers
        .get(name)
        .context("Package server is missing")?;
    match server {
        ServerDeclaration::Stdio {
            command,
            args,
            env,
            cwd,
        } => {
            let data_parent = directory.join("data");
            filesystem.create_dir(&data_parent).await?;
            let data_parent = filesystem.canonicalize(&data_parent).await?;
            let data = data_parent.join(&package.id);
            filesystem.create_dir(&data).await?;
            let data = filesystem.canonicalize(&data).await?;
            ensure!(
                data.starts_with(&data_parent),
                "Package data directory escapes managed storage"
            );
            let root_text = root.to_str().context("Package path is not valid Unicode")?;
            let data_text = data
                .to_str()
                .context("Package data path is not valid Unicode")?;
            let working_directory = match cwd.as_deref() {
                None => root.clone(),
                Some(value) => {
                    let expanded = expand_value(value, &root, &data);
                    let path = if expanded.starts_with("./") {
                        root.join(expanded)
                    } else {
                        PathBuf::from(expanded)
                    };
                    let path = filesystem
                        .canonicalize(&path)
                        .await
                        .context("Package working directory is missing")?;
                    let boundary = if value.starts_with("${PLUGIN_DATA}") {
                        &data
                    } else {
                        &root
                    };
                    ensure!(
                        path.starts_with(boundary),
                        "Package working directory escapes its root"
                    );
                    ensure!(
                        filesystem.is_dir(&path).await,
                        "Package working directory is not a directory"
                    );
                    path
                }
            };
            let path = if command.starts_with("./") {
                let path = filesystem
                    .canonicalize(&root.join(command))
                    .await
                    .context("Package executable is missing")?;
                ensure!(
                    path.starts_with(&root),
                    "Package executable escapes its root"
                );
                path
            } else {
                PathBuf::from(command)
            };
            let mut environment = collections::HashMap::default();
            for (name, value) in env {
                if cfg!(windows)
                    && (name.eq_ignore_ascii_case("PLUGIN_ROOT")
                        || name.eq_ignore_ascii_case("PLUGIN_DATA"))
                {
                    continue;
                }
                environment.insert(name.clone(), expand_value(value, &root, &data));
            }
            environment.insert("PLUGIN_ROOT".into(), root_text.into());
            environment.insert("PLUGIN_DATA".into(), data_text.into());
            Ok(ContextServerSettings::Stdio {
                enabled: true,
                remote: false,
                command: settings::ContextServerCommand {
                    path,
                    cwd: Some(working_directory),
                    args: args
                        .iter()
                        .map(|value| expand_value(value, &root, &data))
                        .collect(),
                    env: Some(environment),
                    timeout: None,
                },
            })
        }
        ServerDeclaration::StreamableHttp { url, headers } => Ok(ContextServerSettings::Http {
            enabled: true,
            url: url.clone(),
            headers: headers.clone().into_iter().collect(),
            timeout: None,
            oauth: None,
        }),
        ServerDeclaration::Sse { .. } => anyhow::bail!("Legacy HTTP+SSE is unsupported"),
    }
}

pub fn settings_fingerprint(
    settings: &project::project_settings::ContextServerSettings,
) -> Result<String> {
    use sha2::{Digest as _, Sha256};
    let mut settings = settings.clone();
    settings.set_enabled(true);
    let mut value = serde_json::to_value(settings)?;
    value.sort_all_objects();
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
}

#[derive(Clone)]
pub struct PackageServerDescriptor {
    pub owner: gpui::WeakEntity<crate::ExtensionStore>,
    pub package_id: String,
    pub revision: String,
    pub server_name: String,
    pub approval_token: Option<String>,
    pub settings: project::project_settings::ContextServerSettings,
}

pub async fn validate_effective_settings(
    filesystem: &dyn fs::Fs,
    defaults: &project::project_settings::ContextServerSettings,
    settings: &project::project_settings::ContextServerSettings,
) -> Result<()> {
    use project::project_settings::ContextServerSettings;
    match (defaults, settings) {
        (
            ContextServerSettings::Stdio {
                command: defaults, ..
            },
            ContextServerSettings::Stdio {
                command, remote, ..
            },
        ) => {
            ensure!(!remote, "Package servers must run on the local host");
            for name in ["PLUGIN_ROOT", "PLUGIN_DATA"] {
                ensure!(
                    command.env.as_ref().and_then(|env| env.get(name))
                        == defaults.env.as_ref().and_then(|env| env.get(name)),
                    "Package settings must preserve {name}"
                );
            }
            let cwd = command
                .cwd
                .as_ref()
                .context("Package working directory is required")?;
            let cwd = filesystem.canonicalize(cwd).await?;
            ensure!(
                filesystem.is_dir(&cwd).await,
                "Package working directory is not a directory"
            );
            ensure!(
                ["PLUGIN_ROOT", "PLUGIN_DATA"].iter().any(|name| defaults
                    .env
                    .as_ref()
                    .and_then(|env| env.get(*name))
                    .is_some_and(|root| cwd.starts_with(root))),
                "Package working directory escapes its roots"
            );
        }
        (ContextServerSettings::Http { .. }, ContextServerSettings::Http { url, headers, .. }) => {
            ServerDeclaration::StreamableHttp {
                url: url.clone(),
                headers: headers.clone().into_iter().collect(),
            }
            .validate()?;
        }
        _ => anyhow::bail!("Package settings cannot change the declared transport"),
    }
    Ok(())
}

impl project::context_server_store::registry::ContextServerDescriptor for PackageServerDescriptor {
    fn configuration_key(&self) -> Option<std::sync::Arc<str>> {
        Some(package_configuration_key(&self.package_id, &self.server_name).into())
    }

    fn validate_installation(
        &self,
        settings: &project::project_settings::ContextServerSettings,
        cx: &gpui::AsyncApp,
    ) -> gpui::Task<Result<()>> {
        let settings = settings.clone();
        let owner = self.owner.clone();
        let package_id = self.package_id.clone();
        let server_name = self.server_name.clone();
        let revision = self.revision.clone();
        cx.spawn(async move |cx| {
            let (filesystem, directory, package) = owner.read_with(cx, |owner, _| {
                Ok::<_, anyhow::Error>((
                    owner.fs.clone(),
                    owner.agent_plugins_directory.clone(),
                    owner
                        .agent_plugins
                        .get(&package_id)
                        .context("Package was removed")?
                        .clone(),
                ))
            })??;
            ensure!(package.revision == revision, "Package revision changed");
            let defaults =
                prepare_server_settings(filesystem.as_ref(), &directory, &package, &server_name)
                    .await?;
            validate_effective_settings(filesystem.as_ref(), &defaults, &settings).await
        })
    }

    fn provided_settings(&self) -> Option<project::project_settings::ContextServerSettings> {
        Some(self.settings.clone())
    }

    fn authorize_settings(
        &self,
        settings: &project::project_settings::ContextServerSettings,
        cx: &gpui::App,
    ) -> Result<()> {
        let owner = self
            .owner
            .upgrade()
            .context("Package owner is unavailable")?;
        let package = owner
            .read(cx)
            .agent_plugins
            .get(&self.package_id)
            .context("Package was removed")?;
        ensure!(
            package.enabled
                && package.revision == self.revision
                && package.approved_servers.contains(&self.server_name)
                && self
                    .approval_token
                    .as_ref()
                    .is_some_and(
                        |token| package.approval_tokens.get(&self.server_name) == Some(token)
                    ),
            "Package server approval was revoked"
        );
        if let project::project_settings::ContextServerSettings::Http { url, headers, .. } =
            settings
        {
            ServerDeclaration::StreamableHttp {
                url: url.clone(),
                headers: headers.clone().into_iter().collect(),
            }
            .validate()?;
        }
        let expected = package
            .approved_configurations
            .get(&self.server_name)
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| settings_fingerprint(&self.settings))?;
        ensure!(
            settings_fingerprint(settings)? == expected,
            "Package server configuration changed; review and approve it again"
        );
        Ok(())
    }

    fn command(
        &self,
        _worktree_store: gpui::Entity<project::worktree_store::WorktreeStore>,
        _cx: &gpui::AsyncApp,
    ) -> gpui::Task<Result<settings::ContextServerCommand>> {
        match &self.settings {
            project::project_settings::ContextServerSettings::Stdio { command, .. } => {
                gpui::Task::ready(Ok(command.clone()))
            }
            _ => gpui::Task::ready(Err(anyhow::anyhow!("This package server uses HTTP"))),
        }
    }

    fn configuration(
        &self,
        _worktree_store: gpui::Entity<project::worktree_store::WorktreeStore>,
        _cx: &gpui::AsyncApp,
    ) -> gpui::Task<Result<Option<extension::ContextServerConfiguration>>> {
        gpui::Task::ready(Ok(None))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InstalledPackage {
    pub id: String,
    pub manifest: PackageManifest,
    pub source: PathBuf,
    pub revision: String,
    pub enabled: bool,
    pub approved_servers: std::collections::BTreeSet<String>,
    #[serde(default)]
    pub approved_configurations: BTreeMap<String, String>,
    #[serde(default)]
    pub approval_tokens: BTreeMap<String, String>,
    pub servers: BTreeMap<String, ServerDeclaration>,
    pub diagnostics: Vec<String>,
}

#[cfg(feature = "agentic")]
impl crate::ExtensionStore {
    pub fn init_agent_plugins(&mut self, cx: &mut gpui::Context<Self>) {
        if self.agent_plugins_subscription.is_some() {
            return;
        }
        let entity = cx.entity();
        self.agent_plugins_subscription = Some(cx.subscribe(&entity, |this, _, event, cx| {
            if matches!(event, crate::Event::ExtensionsUpdated) {
                this.refresh_agent_plugin_contributions(cx);
            }
        }));
        self.refresh_agent_plugin_contributions(cx);
    }

    fn refresh_agent_plugin_contributions(&mut self, cx: &mut gpui::Context<Self>) {
        use agent_skills::{SkillIndex, SkillSource};
        use gpui::{AppContext as _, BorrowAppContext as _};
        use project::context_server_store::registry::ContextServerDescriptorRegistry;
        use util::ResultExt as _;
        let records = self.agent_plugins.clone();
        let directory = self.agent_plugins_directory.clone();
        let registry = ContextServerDescriptorRegistry::default_global(cx);
        let allowed = records
            .values()
            .filter(|package| package.enabled)
            .flat_map(|package| {
                package
                    .approved_servers
                    .iter()
                    .map(|name| std::sync::Arc::<str>::from(package.server_id(name)))
            })
            .collect::<std::collections::BTreeSet<_>>();
        let retired = self
            .agent_plugins_registered
            .difference(&allowed)
            .cloned()
            .collect::<Vec<_>>();
        for id in retired {
            registry.update(cx, |registry, cx| {
                registry.unregister_context_server_descriptor_by_id(&id, cx)
            });
            self.agent_plugins_registered.remove(&id);
        }
        if !cx.has_global::<SkillIndex>() {
            cx.set_global(SkillIndex::default());
        }
        cx.update_global(|index: &mut SkillIndex, _| {
            index.package_skills.retain(|skill| match &skill.source {
                SkillSource::Package { root, .. } => records.values().any(|package| {
                    package.enabled && package.snapshot_path(&directory) == root.as_ref()
                }),
                _ => false,
            });
        });
        let filesystem = self.fs.clone();
        self.agent_plugins_refresh = Some(cx.spawn(async move |this, cx| {
            let contributions = cx.background_spawn({
                let records = records.clone();
                async move { collect_package_contributions(filesystem, directory, records).await }
            }).await;
            this.update(cx, |this, cx| {
                if this.agent_plugins != records {
                    return;
                }
                let desired = contributions
                    .servers
                    .iter()
                    .filter_map(|(package_id, name, _)| {
                        records
                            .get(package_id)
                            .map(|package| std::sync::Arc::<str>::from(package.server_id(name)))
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                let retired = this
                    .agent_plugins_registered
                    .difference(&desired)
                    .cloned()
                    .collect::<Vec<_>>();
                for id in retired {
                    registry.update(cx, |registry, cx| {
                        registry.unregister_context_server_descriptor_by_id(&id, cx)
                    });
                    this.agent_plugins_registered.remove(&id);
                }
                for (package_id, server_name, settings) in contributions.servers {
                    let Some(package) = records.get(&package_id) else {
                        continue;
                    };
                    let id: std::sync::Arc<str> = package.server_id(&server_name).into();
                    if this.agent_plugins_registered.insert(id.clone()) {
                        let descriptor = PackageServerDescriptor {
                            owner: cx.weak_entity(),
                            package_id,
                            revision: package.revision.clone(),
                            approval_token: package.approval_tokens.get(&server_name).cloned(),
                            server_name,
                            settings,
                        };
                        registry.update(cx, |registry, cx| {
                            registry.register_context_server_descriptor(
                                id,
                                std::sync::Arc::new(descriptor),
                                cx,
                            )
                        });
                    }
                }
                cx.update_global(|index: &mut SkillIndex, _| {
                    index.package_skills = contributions.skills
                });
                this.agent_plugins_diagnostics = contributions.diagnostics;
                cx.notify();
            })
            .log_err();
        }));
    }
}

#[cfg(feature = "agentic")]
#[derive(Default)]
struct PackageContributions {
    skills: Vec<agent_skills::Skill>,
    servers: Vec<(
        String,
        String,
        project::project_settings::ContextServerSettings,
    )>,
    diagnostics: BTreeMap<String, Vec<String>>,
}

#[cfg(feature = "agentic")]
async fn collect_package_contributions(
    filesystem: std::sync::Arc<dyn fs::Fs>,
    directory: PathBuf,
    records: BTreeMap<String, InstalledPackage>,
) -> PackageContributions {
    let mut contributions = PackageContributions::default();
    for package in records.values().filter(|package| package.enabled) {
        let root = package.snapshot_path(&directory);
        let snapshot = capture_package(filesystem.as_ref(), &root, SnapshotLimits::default()).await;
        let error = match snapshot {
            Ok(snapshot) if snapshot.revision == package.revision => None,
            Ok(_) => Some(
                "Managed package content changed; inspect and update the package before using it"
                    .to_owned(),
            ),
            Err(error) => Some(format!("Cannot load managed package: {error:#}")),
        };
        if let Some(error) = error {
            contributions
                .diagnostics
                .entry(package.id.clone())
                .or_default()
                .push(error);
            continue;
        }
        for skill in
            agent_skills::load_package_skills(&filesystem, &root, &package.manifest.name).await
        {
            match skill {
                Ok(skill) => contributions.skills.push(skill),
                Err(error) => contributions
                    .diagnostics
                    .entry(package.id.clone())
                    .or_default()
                    .push(format!("{}: {}", error.path.display(), error.message)),
            }
        }
        for name in &package.approved_servers {
            match prepare_server_settings(filesystem.as_ref(), &directory, package, name).await {
                Ok(settings) => {
                    contributions
                        .servers
                        .push((package.id.clone(), name.clone(), settings))
                }
                Err(error) => contributions
                    .diagnostics
                    .entry(package.id.clone())
                    .or_default()
                    .push(format!("{name}: {error:#}")),
            }
        }
    }
    contributions
}

pub enum PackageChange {
    Enabled(bool),
    ApproveServer {
        name: String,
        revision: String,
    },
    ApproveServerConfiguration {
        name: String,
        revision: String,
        settings: project::project_settings::ContextServerSettings,
    },
    RevokeServer(String),
    Update(PackageSnapshot),
    Uninstall {
        delete_data: bool,
    },
}

impl PackageChange {
    pub fn retires_authority(&self) -> bool {
        matches!(
            self,
            Self::Enabled(false) | Self::RevokeServer(_) | Self::Uninstall { .. }
        )
    }
}

pub fn package_configuration_key(package_id: &str, server_name: &str) -> String {
    use sha2::{Digest as _, Sha256};
    format!(
        "agent-plugin/{package_id}/{:x}",
        Sha256::digest(server_name.as_bytes())
    )
}

impl InstalledPackage {
    pub fn server_id(&self, name: &str) -> String {
        format!(
            "{}/{}/{}/{}",
            package_configuration_key(&self.id, name),
            self.revision,
            self.approved_configurations
                .get(name)
                .map(String::as_str)
                .unwrap_or("default"),
            self.approval_tokens
                .get(name)
                .map(String::as_str)
                .unwrap_or("unapproved")
        )
    }

    pub fn from_snapshot(snapshot: &PackageSnapshot) -> Self {
        use sha2::{Digest as _, Sha256};
        let mut identity = Sha256::new();
        identity.update(snapshot.source.as_os_str().as_encoded_bytes());
        identity.update([0]);
        identity.update(snapshot.manifest.name.as_bytes());
        Self {
            id: format!("{:x}", identity.finalize()),
            manifest: snapshot.manifest.clone(),
            source: snapshot.source.clone(),
            revision: snapshot.revision.clone(),
            enabled: true,
            approved_servers: Default::default(),
            approved_configurations: Default::default(),
            approval_tokens: Default::default(),
            servers: snapshot.servers.clone(),
            diagnostics: snapshot.diagnostics.clone(),
        }
    }

    pub fn apply_change(&mut self, change: &PackageChange) -> Result<()> {
        match change {
            PackageChange::Enabled(enabled) => {
                self.enabled = *enabled;
                if !enabled {
                    self.approved_servers.clear();
                    self.approved_configurations.clear();
                    self.approval_tokens.clear();
                }
            }
            PackageChange::ApproveServer { name, revision } => {
                ensure!(
                    self.enabled && &self.revision == revision,
                    "Package changed or is disabled; review the server again"
                );
                ensure!(
                    self.servers.contains_key(name),
                    "Package server no longer exists"
                );
                if !self.approved_servers.contains(name)
                    || self.approved_configurations.contains_key(name)
                {
                    self.approval_tokens
                        .insert(name.clone(), uuid::Uuid::new_v4().to_string());
                }
                self.approved_servers.insert(name.clone());
                self.approved_configurations.remove(name);
            }
            PackageChange::ApproveServerConfiguration {
                name,
                revision,
                settings,
            } => {
                ensure!(
                    self.enabled && &self.revision == revision && self.servers.contains_key(name),
                    "Package changed; review the server again"
                );
                let fingerprint = settings_fingerprint(settings)?;
                if !self.approved_servers.contains(name)
                    || self.approved_configurations.get(name) != Some(&fingerprint)
                {
                    self.approval_tokens
                        .insert(name.clone(), uuid::Uuid::new_v4().to_string());
                }
                self.approved_servers.insert(name.clone());
                self.approved_configurations
                    .insert(name.clone(), fingerprint);
            }
            PackageChange::RevokeServer(name) => {
                self.approved_servers.remove(name);
                self.approved_configurations.remove(name);
                self.approval_tokens.remove(name);
            }
            PackageChange::Update(snapshot) => {
                ensure!(
                    self.manifest.name == snapshot.manifest.name,
                    "Update has a different package name"
                );
                if self.revision != snapshot.revision {
                    self.approved_servers.clear();
                    self.approved_configurations.clear();
                    self.approval_tokens.clear();
                }
                self.manifest = snapshot.manifest.clone();
                self.source = snapshot.source.clone();
                self.revision = snapshot.revision.clone();
                self.servers = snapshot.servers.clone();
                self.diagnostics = snapshot.diagnostics.clone();
            }
            PackageChange::Uninstall { .. } => {
                self.enabled = false;
                self.approved_servers.clear();
                self.approved_configurations.clear();
                self.approval_tokens.clear();
            }
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            valid_identity(&self.id) && valid_identity(&self.revision),
            "Invalid installed package identity"
        );
        ensure!(
            valid_package_name(&self.manifest.name) && self.source.is_absolute(),
            "Invalid installed package metadata"
        );
        ensure!(
            self.approved_servers
                .iter()
                .all(|name| self.servers.contains_key(name)),
            "Unknown approved package server"
        );
        ensure!(
            self.approved_configurations
                .iter()
                .all(
                    |(name, digest)| self.approved_servers.contains(name) && valid_identity(digest)
                ),
            "Invalid server approval fingerprint"
        );
        ensure!(
            self.approved_servers.iter().all(|name| self
                .approval_tokens
                .get(name)
                .is_some_and(|token| uuid::Uuid::parse_str(token).is_ok())),
            "Missing server approval identity"
        );
        for server in self.servers.values() {
            server.validate()?;
        }
        Ok(())
    }

    pub fn snapshot_path(&self, packages_directory: &Path) -> PathBuf {
        packages_directory
            .join("snapshots")
            .join(&self.id)
            .join(&self.revision)
    }
}

fn valid_identity(identity: &str) -> bool {
    identity.len() == 64
        && identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub async fn load_inventory(
    filesystem: &dyn fs::Fs,
    directory: &Path,
) -> Result<BTreeMap<String, InstalledPackage>> {
    let path = directory.join("inventory.json");
    if filesystem.metadata(&path).await?.is_none() {
        return Ok(BTreeMap::new());
    }
    let root = filesystem.canonicalize(directory).await?;
    let file = read_snapshot_file(filesystem, &root, &path, 16 * 1024 * 1024).await?;
    let records: BTreeMap<String, InstalledPackage> =
        serde_json::from_slice(&file.bytes).context("Cannot read Agent Plugins inventory")?;
    for (id, record) in &records {
        record.validate()?;
        ensure!(id == &record.id, "Package inventory identity mismatch");
    }
    Ok(records)
}

pub async fn save_inventory(
    filesystem: &dyn fs::Fs,
    directory: &Path,
    records: &BTreeMap<String, InstalledPackage>,
) -> Result<()> {
    for record in records.values() {
        record.validate()?;
    }
    filesystem.create_dir(directory).await?;
    filesystem
        .atomic_write(
            directory.join("inventory.json"),
            serde_json::to_string_pretty(records)?,
        )
        .await
}

pub async fn write_snapshot(
    filesystem: &dyn fs::Fs,
    directory: &Path,
    record: &InstalledPackage,
    snapshot: &PackageSnapshot,
) -> Result<()> {
    record.validate()?;
    let destination = record.snapshot_path(directory);
    if filesystem.is_dir(&destination).await {
        let existing = capture_package(filesystem, &destination, SnapshotLimits::default()).await?;
        ensure!(
            existing.revision == snapshot.revision,
            "Installed snapshot was modified"
        );
        return Ok(());
    }
    let staging_parent = directory.join("staging");
    filesystem.create_dir(&staging_parent).await?;
    let staging = staging_parent.join(&record.id);
    filesystem
        .remove_dir(
            &staging,
            fs::RemoveOptions {
                recursive: true,
                ignore_if_not_exists: true,
            },
        )
        .await?;
    filesystem.create_dir(&staging).await?;
    for relative in &snapshot.directories {
        if !snapshot.links.keys().any(|link| relative.starts_with(link)) {
            filesystem.create_dir(&staging.join(relative)).await?;
        }
    }
    for (relative, file) in &snapshot.files {
        if snapshot.links.keys().any(|link| relative.starts_with(link)) {
            continue;
        }
        let path = staging.join(relative);
        filesystem.write(&path, &file.bytes).await?;
        if file.executable && !filesystem.is_fake() {
            util::fs::make_file_executable(&path).await?;
        }
    }
    for (link, target) in &snapshot.links {
        if snapshot
            .links
            .keys()
            .any(|ancestor| ancestor != link && link.starts_with(ancestor))
        {
            continue;
        }
        let mut relative_target = PathBuf::new();
        if let Some(parent) = link.parent() {
            for _ in parent.components() {
                relative_target.push("..");
            }
        }
        relative_target.push(target);
        filesystem
            .create_symlink(&staging.join(link), relative_target)
            .await?;
    }
    filesystem
        .create_dir(destination.parent().context("Snapshot has no parent")?)
        .await?;
    filesystem
        .rename(&staging, &destination, fs::RenameOptions::default())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::Fs as _;
    use serde_json::json;

    #[test]
    fn agent_plugins_manifest_recovery_and_version() {
        let bytes = serde_json::to_vec(&json!({"$schema": MANIFEST_SCHEMA, "name":"made.test", "version":"development", "unknown":true,"extensions":{"org.unknown":false}})).expect("fixture");
        let (manifest, diagnostics) = parse_manifest(&bytes).expect("valid manifest");
        assert_eq!(manifest.version.as_deref(), Some("development"));
        assert_eq!(diagnostics.len(), 1);
        assert!(parse_manifest(br#"{"$schema":"https://evil/schema","name":"test"}"#).is_err());
        assert!(parse_manifest(br#"{"name":"test"}"#).is_err());
    }

    #[test]
    fn agent_plugins_independent_server_failures() {
        let document = json!({"$schema":MCP_SCHEMA,"mcpServers":{
            "valid":{"type":"stdio","command":"node","args":["./server.js"]},
            "invalid":{"type":"stdio","command":"node --eval"},
            "remote":{"type":"streamable-http","url":"http://example.com"},
            "legacy":{"type":"sse","url":"https://example.com"}
        }});
        let parsed =
            parse_servers(&serde_json::to_vec(&document).expect("fixture")).expect("document");
        assert_eq!(parsed.servers.len(), 1);
        assert_eq!(parsed.diagnostics.len(), 3);
    }

    #[test]
    fn agent_plugins_expansion_is_not_recursive() {
        assert_eq!(
            expand_value(
                "${PLUGIN_ROOT}/${PLUGIN_DATA}/${HOME}",
                Path::new("/x/${PLUGIN_DATA}"),
                Path::new("/data")
            ),
            "/x/${PLUGIN_DATA}//data/${HOME}"
        );
        assert_eq!(
            expand_value("${PLUGIN_UNKNOWN}", Path::new("/root"), Path::new("/data")),
            "${PLUGIN_UNKNOWN}"
        );
    }

    #[test]
    fn agent_plugins_closed_server_fields_and_nulls() {
        for invalid in [
            json!({"type":"stdio", "command":"node", "cwd":null}),
            json!({"type":"stdio", "command":"node", "url":"https://example.com"}),
            json!({"type":"stdio", "command":"node", "args":[3]}),
            json!({"type":"streamable-http", "url":"https://user:secret@example.com"}),
            json!({"type":"streamable-http", "url":"https://example.com#fragment"}),
        ] {
            let bytes =
                serde_json::to_vec(&json!({"$schema":MCP_SCHEMA,"mcpServers":{"bad":invalid}}))
                    .expect("fixture");
            let parsed = parse_servers(&bytes).expect("valid document");
            assert!(parsed.servers.is_empty());
            assert_eq!(parsed.diagnostics.len(), 1);
        }
    }

    #[test]
    fn agent_plugins_literal_executable_and_reserved_environment_rules() {
        let declaration = json!({"$schema": MCP_SCHEMA, "mcpServers": {
            "literal": {"type": "stdio", "command": "./bin/${LITERAL}", "env": {"plugin_root": "literal"}}
        }});
        let parsed = parse_servers(&serde_json::to_vec(&declaration).expect("fixture"))
            .expect("valid declaration");
        assert!(parsed.diagnostics.is_empty());
        let Some(ServerDeclaration::Stdio { command, env, .. }) = parsed.servers.get("literal")
        else {
            panic!("stdio server");
        };
        assert_eq!(command, "./bin/${LITERAL}");
        for name in ["PLUGIN_ROOT", "PLUGIN_DATA"] {
            let value = json!({"$schema": MCP_SCHEMA, "mcpServers": {"invalid": {"type":"stdio", "command":"node", "env": {name:"forbidden"}}}});
            assert!(
                parse_servers(&serde_json::to_vec(&value).expect("fixture"))
                    .expect("MCP document")
                    .servers
                    .is_empty()
            );
        }
        assert_eq!(env.get("plugin_root").map(String::as_str), Some("literal"));
    }

    #[test]
    fn agent_plugins_reject_normalized_unsafe_urls() {
        for url in [
            "https://@example.org/mcp",
            "https://example.org/\npath",
            " https://example.org",
            "https://example.org/#fragment",
        ] {
            assert!(
                ServerDeclaration::StreamableHttp {
                    url: url.into(),
                    headers: BTreeMap::new()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn agent_plugins_manifest_fatal_author_and_nonfatal_extensions() {
        let bytes = serde_json::to_vec(
            &json!({"$schema":MANIFEST_SCHEMA,"name":"valid", "author":{"unknown":"field"}}),
        )
        .expect("fixture");
        assert!(parse_manifest(&bytes).is_err());
        let bytes = serde_json::to_vec(
            &json!({"$schema":MANIFEST_SCHEMA,"name":"valid", "extensions":false}),
        )
        .expect("fixture");
        assert_eq!(parse_manifest(&bytes).expect("recoverable").1.len(), 1);
        for name in [
            "",
            "UPPER",
            "two--parts",
            "two..parts",
            "-prefix",
            "suffix.",
        ] {
            assert!(!valid_package_name(name));
        }
    }

    #[cfg(unix)]
    #[test]
    fn agent_plugins_resolved_containment() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().join("package");
        std::fs::create_dir(&root)?;
        std::fs::write(root.join("inside"), "safe")?;
        std::fs::write(directory.path().join("outside"), "private")?;
        std::os::unix::fs::symlink(root.join("inside"), root.join("internal"))?;
        std::os::unix::fs::symlink(directory.path().join("outside"), root.join("escape"))?;
        assert!(contained_path(&root, &root.join("internal")).is_ok());
        assert!(contained_path(&root, &root.join("escape")).is_err());
        Ok(())
    }

    #[gpui::test]
    async fn agent_plugins_prepare_launch_checks_paths_and_expands_arguments(
        cx: &mut gpui::TestAppContext,
    ) {
        async {
            let filesystem = fs::FakeFs::new(cx.executor());
            filesystem.insert_tree("/source", json!({
                "plugin.json": json!({"$schema":MANIFEST_SCHEMA,"name":"launch"}).to_string(),
                "mcp.json": json!({"$schema":MCP_SCHEMA,"mcpServers": {
                    "good": {"type":"stdio","command":"./bin/server", "args":["${PLUGIN_ROOT}/input", "${HOME}"], "env":{"DATA":"${PLUGIN_DATA}/config"}, "cwd":"./resources"},
                    "escape": {"type":"stdio","command":"./bin/server", "cwd":"./../outside"}
                }}).to_string(),
                "bin":{"server":"executable"}, "resources":{}
            })).await;
            let snapshot = capture_package(filesystem.as_ref(), Path::new("/source"), SnapshotLimits::default()).await?;
            let package = InstalledPackage::from_snapshot(&snapshot);
            let directory = Path::new("/managed");
            write_snapshot(filesystem.as_ref(), directory, &package, &snapshot).await?;
            let root = package.snapshot_path(directory);
            let settings = prepare_server_settings(filesystem.as_ref(), directory, &package, "good").await?;
            validate_effective_settings(filesystem.as_ref(), &settings, &settings).await?;
            let mut overridden = settings.clone();
            if let project::project_settings::ContextServerSettings::Stdio { command, .. } = &mut overridden {
                command.env.as_mut().context("environment")?.remove("PLUGIN_ROOT");
            }
            assert!(validate_effective_settings(filesystem.as_ref(), &settings, &overridden).await.is_err());
            let mut overridden = settings.clone();
            filesystem.create_symlink(&root.join("outside"), PathBuf::from("/source")).await?;
            if let project::project_settings::ContextServerSettings::Stdio { command, .. } = &mut overridden {
                command.cwd = Some(root.join("outside"));
            }
            assert!(validate_effective_settings(filesystem.as_ref(), &settings, &overridden).await.is_err());
            let project::project_settings::ContextServerSettings::Stdio { command, .. } = settings else { anyhow::bail!("expected stdio"); };
            assert_eq!(command.path, root.join("bin/server"));
            assert_eq!(command.cwd, Some(root.join("resources")));
            assert_eq!(command.args, vec![format!("{}/input", root.display()), "${HOME}".into()]);
            assert!(fs::Fs::is_dir(filesystem.as_ref(), &directory.join("data").join(&package.id)).await);
            assert!(prepare_server_settings(filesystem.as_ref(), directory, &package, "escape").await.is_err());
            let mut package = package;
            let approval = PackageChange::ApproveServer {name: "good".into(), revision: package.revision.clone()};
            package.apply_change(&approval)?;
            let first_id = package.server_id("good");
            package.apply_change(&approval)?;
            assert_eq!(first_id, package.server_id("good"));
            package.apply_change(&PackageChange::RevokeServer("good".into()))?;
            package.apply_change(&approval)?;
            assert_ne!(first_id, package.server_id("good"), "old tool grants must not regain authority");
            let restored: InstalledPackage = serde_json::from_slice(&serde_json::to_vec(&package)?)?;
            assert_eq!(package.server_id("good"), restored.server_id("good"));
            Ok::<(), anyhow::Error>(())
        }.await.expect("prepare portable launch");
    }

    #[gpui::test]
    async fn agent_plugins_snapshot_preserves_reviewed_bytes(cx: &mut gpui::TestAppContext) {
        async {
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/package", json!({
            "plugin.json": json!({"$schema":MANIFEST_SCHEMA,"name":"example"}).to_string(),
            "mcp.json": json!({"$schema":MCP_SCHEMA,"mcpServers":{"local":{"type":"stdio","command":"./server"}}}).to_string(),
            "server":"original"
        })).await;
        let snapshot = capture_package(
            filesystem.as_ref(),
            Path::new("/package"),
            SnapshotLimits::default(),
        )
        .await?;
        fs::Fs::write(
            filesystem.as_ref(),
            Path::new("/package/server"),
            b"changed",
        )
        .await?;
        assert_eq!(
            snapshot
                .files
                .get(Path::new("server"))
                .context("server")?
                .bytes,
            b"original"
        );
        let changed = capture_package(
            filesystem.as_ref(),
            Path::new("/package"),
            SnapshotLimits::default(),
        )
        .await?;
        assert_ne!(snapshot.revision, changed.revision);
        assert_eq!(snapshot.servers.len(), 1);
        assert!(
            capture_package(
                filesystem.as_ref(),
                Path::new("/package"),
                SnapshotLimits {
                    entries: 1,
                    ..SnapshotLimits::default()
                }
            )
            .await
            .is_err()
        );
        Ok::<(), anyhow::Error>(())
        }.await.expect("snapshot scenario");
    }

    #[gpui::test]
    async fn agent_plugins_snapshot_isolates_component_escape(cx: &mut gpui::TestAppContext) {
        async {
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/package", json!({"plugin.json":json!({"$schema":MANIFEST_SCHEMA,"name":"example"}).to_string(),"skills":{"safe":{"SKILL.md":"safe"}}})).await;
        filesystem
            .insert_tree("/external", json!({"config":"private"}))
            .await;
        fs::Fs::create_symlink(
            filesystem.as_ref(),
            Path::new("/package/mcp.json"),
            PathBuf::from("/external/config"),
        )
        .await?;
        let snapshot = capture_package(
            filesystem.as_ref(),
            Path::new("/package"),
            SnapshotLimits::default(),
        )
        .await?;
        assert!(
            snapshot
                .files
                .contains_key(Path::new("skills/safe/SKILL.md"))
        );
        assert!(!snapshot.files.contains_key(Path::new("mcp.json")));
        assert!(!snapshot.diagnostics.is_empty());
        Ok::<(), anyhow::Error>(())
        }.await.expect("snapshot scenario");
    }

    #[gpui::test]
    async fn agent_plugins_inventory_restart_and_corruption(cx: &mut gpui::TestAppContext) {
        async {
            let filesystem = fs::FakeFs::new(cx.executor());
            filesystem.insert_tree("/source", json!({"plugin.json":json!({"$schema":MANIFEST_SCHEMA,"name":"persisted"}).to_string(),"asset":"content"})).await;
            let snapshot = capture_package(filesystem.as_ref(), Path::new("/source"), SnapshotLimits::default()).await?;
            let record = InstalledPackage::from_snapshot(&snapshot);
            assert!(record.approved_servers.is_empty());
            let directory = Path::new("/extensions/agent-plugins");
            write_snapshot(filesystem.as_ref(), directory, &record, &snapshot).await?;
            let records = BTreeMap::from([(record.id.clone(), record.clone())]);
            save_inventory(filesystem.as_ref(), directory, &records).await?;
            let restored = load_inventory(filesystem.as_ref(), directory).await?;
            assert_eq!(restored.get(&record.id).context("record")?.revision, record.revision);
            assert_eq!(fs::Fs::load(filesystem.as_ref(), &record.snapshot_path(directory).join("asset")).await?, "content");
            fs::Fs::write(filesystem.as_ref(), &directory.join("inventory.json"), b"broken").await?;
            assert!(load_inventory(filesystem.as_ref(), directory).await.is_err());
            Ok::<(), anyhow::Error>(())
        }.await.expect("inventory scenario");
    }

    #[gpui::test]
    async fn agent_plugins_snapshot_preserves_internal_links(cx: &mut gpui::TestAppContext) {
        async {
            let filesystem = fs::FakeFs::new(cx.executor());
            filesystem.insert_tree("/linked", json!({
                "plugin.json":json!({"$schema":MANIFEST_SCHEMA,"name":"linked"}).to_string(),
                "mcp.json":json!({"$schema":MCP_SCHEMA,"mcpServers":{"server":{"type":"stdio","command":"./bin/../entry"}}}).to_string(),
                "bin":{"server":"executable","resource":"data"}
            })).await;
            fs::Fs::create_symlink(filesystem.as_ref(), Path::new("/linked/entry"), PathBuf::from("bin/server")).await?;
            let snapshot = capture_package(filesystem.as_ref(), Path::new("/linked"), SnapshotLimits::default()).await?;
            assert_eq!(snapshot.servers.len(), 1);
            let record = InstalledPackage::from_snapshot(&snapshot);
            let directory = Path::new("/installed");
            write_snapshot(filesystem.as_ref(), directory, &record, &snapshot).await?;
            let installed = record.snapshot_path(directory);
            assert_eq!(fs::Fs::read_link(filesystem.as_ref(), &installed.join("entry")).await?, PathBuf::from("bin/server"));
            let restored = capture_package(filesystem.as_ref(), &installed, SnapshotLimits::default()).await?;
            assert_eq!(snapshot.revision, restored.revision);
            Ok::<(), anyhow::Error>(())
        }.await.expect("internal link scenario");
    }

    #[test]
    fn agent_plugins_paths_and_headers() {
        assert!(validate_relative_path("../escape").is_err());
        assert!(validate_relative_path("/absolute").is_err());
        let server = ServerDeclaration::StreamableHttp {
            url: "http://127.0.0.1/mcp".into(),
            headers: BTreeMap::from([
                ("X-Test".into(), "one".into()),
                ("x-test".into(), "two".into()),
            ]),
        };
        assert!(server.validate().is_err());
    }
}
