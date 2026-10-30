#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::AtomicU64;
use std::{
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::{Context as _, Result, ensure};
use collections::HashMap;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::ProjectPath;

pub const MAX_PROFILE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_SYMBOL_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PROFILE_NODES: usize = 100_000;
const MAX_SAMPLES: usize = 1_000_000;
const MAX_DEPTH: usize = 256;
#[cfg(any(test, feature = "test-support"))]
static SAMPLY_PROFILE_LOAD_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(any(test, feature = "test-support"))]
pub fn samply_profile_load_calls_for_test() -> u64 {
    SAMPLY_PROFILE_LOAD_CALLS.load(Ordering::Relaxed)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileFrame {
    pub label: String,
    pub source: Option<(ProjectPath, u32)>,
    pub resolved: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileNode {
    pub frame: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub inclusive_samples: u64,
    pub self_samples: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileResults {
    pub frames: Vec<ProfileFrame>,
    pub nodes: Vec<ProfileNode>,
    pub roots: Vec<usize>,
    pub total_samples: u64,
    pub unattributed_samples: u64,
    pub unresolved_frames: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileHost {
    MacOs,
    Linux,
    Windows,
}

fn array(value: &Value) -> Result<&Vec<Value>> {
    value.as_array().context("Profile column is not an array")
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value.get(name).context("Required profile field is absent")
}

fn integer(value: &Value) -> Result<usize> {
    usize::try_from(value.as_u64().context("Profile integer is invalid")?)
        .context("Profile integer exceeds the supported range")
}

fn at(values: &[Value], index: usize) -> Result<&Value> {
    values.get(index).context("Profile index is out of range")
}

fn column<'a>(table: &'a Value, name: &str) -> Result<&'a Vec<Value>> {
    let length = integer(field(table, "length")?)?;
    ensure!(
        length <= MAX_PROFILE_NODES,
        "Profile table exceeds the entry limit"
    );
    let values = array(field(table, name)?)?;
    ensure!(
        values.len() == length,
        "Profile table column length mismatch"
    );
    Ok(values)
}

fn text(value: &Value) -> Result<&str> {
    value.as_str().context("Profile string is invalid")
}

fn string_at(values: &[Value], value: &Value) -> Result<String> {
    Ok(text(at(values, integer(value)?)?)?.to_owned())
}

fn label(value: &str) -> String {
    if value.contains(['/', '\\', '"', '\'', '\0']) || value.chars().any(char::is_control) {
        return "<redacted symbol>".to_string();
    }
    let mut end = value.len().min(512);
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    value.get(..end).unwrap_or_default().to_owned()
}

fn identity(value: &str) -> String {
    value
        .chars()
        .filter(|character| *character != '-')
        .flat_map(char::to_uppercase)
        .collect()
}

fn cancelled(cancel: &AtomicBool) -> Result<()> {
    ensure!(
        !cancel.load(Ordering::Relaxed),
        "Profile import was cancelled"
    );
    Ok(())
}

struct CancellableJsonReader<'a, R> {
    reader: R,
    cancel: &'a AtomicBool,
}

impl<R: Read> Read for CancellableJsonReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(std::io::Error::other("Profile import was cancelled"));
        }
        self.reader.read(buffer)
    }
}

fn parse_json(
    reader: impl Read,
    cancel: &AtomicBool,
    malformed_message: &'static str,
) -> Result<Value> {
    cancelled(cancel)?;
    let reader = BufReader::with_capacity(64 * 1024, CancellableJsonReader { reader, cancel });
    #[allow(
        clippy::disallowed_methods,
        reason = "bounded buffered decoding must check cancellation during parsing; from_slice cannot"
    )]
    let result = serde_json::from_reader(reader);
    cancelled(cancel)?;
    result.context(malformed_message)
}

fn read_profile_input(mut reader: impl Read, limit: usize, cancel: &AtomicBool) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        cancelled(cancel)?;
        let remaining = (limit - bytes.len()).saturating_add(1).min(buffer.len());
        let count = match reader.read(&mut buffer[..remaining]) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => anyhow::bail!("Cannot read the selected profile input"),
        };
        cancelled(cancel)?;
        if count == 0 {
            return Ok(bytes);
        }
        ensure!(
            count <= limit - bytes.len(),
            "Profile input exceeds the byte limit"
        );
        bytes.extend_from_slice(&buffer[..count]);
    }
}

pub struct ProfileInput {
    root: PathBuf,
    path: PathBuf,
    root_inode: u64,
    file_inode: u64,
    fingerprint: Option<[u8; 32]>,
}

impl ProfileInput {
    pub fn from_snapshot(
        snapshot: &worktree::Snapshot,
        path: &util::rel_path::RelPath,
    ) -> Result<Self> {
        let root = snapshot
            .root_entry()
            .context("Profile worktree is missing")?;
        let file = snapshot
            .entry_for_path(path)
            .context("Profile input is missing")?;
        ensure!(
            root.is_dir()
                && file.is_file()
                && !file.is_private
                && !file.is_fifo
                && !file.is_external,
            "Profile input is not permitted"
        );
        Ok(Self {
            root: snapshot.abs_path().to_path_buf(),
            path: path.as_std_path().to_path_buf(),
            root_inode: root.inode,
            file_inode: file.inode,
            fingerprint: None,
        })
    }

    async fn open(&self, fs: &dyn fs::Fs) -> Result<Box<dyn Read + Send + Sync>> {
        fs.open_sync_confined_file(&self.root, &self.path, self.root_inode, self.file_inode)
            .await
    }

    pub(crate) async fn read_with_fingerprint(
        &self,
        filesystem: &dyn fs::Fs,
        limit: usize,
        cancellation: &AtomicBool,
    ) -> Result<(Vec<u8>, [u8; 32])> {
        ensure!(
            (1..=task::MAX_TASK_ARTIFACT_BYTES as usize).contains(&limit),
            "Artifact input byte limit is invalid"
        );
        cancelled(cancellation)?;
        let bytes = read_profile_input(self.open(filesystem).await?, limit, cancellation)?;
        let fingerprint = profile_fingerprint(&bytes, cancellation)?;
        if let Some(expected) = self.fingerprint {
            ensure!(fingerprint == expected, "Sealed artifact bytes changed");
        }
        Ok((bytes, fingerprint))
    }

    pub fn with_fingerprint(mut self, fingerprint: [u8; 32]) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }
}

pub(crate) fn profile_fingerprint(bytes: &[u8], cancel: &AtomicBool) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    for chunk in bytes.chunks(64 * 1024) {
        cancelled(cancel)?;
        digest.update(chunk);
    }
    Ok(digest.finalize().into())
}

#[cfg(target_os = "macos")]
const MAX_TASK_HELPER_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct TaskHelperIdentity {
    requested: PathBuf,
    canonical: PathBuf,
    root: PathBuf,
    root_inode: u64,
    relative: PathBuf,
    metadata: fs::Metadata,
    fingerprint: [u8; 32],
    retained_file: Option<RetainedTaskHelperFile>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct RetainedTaskHelperFile {
    file: std::sync::Arc<std::fs::File>,
    metadata: std::fs::Metadata,
}

#[cfg(target_os = "macos")]
impl RetainedTaskHelperFile {
    fn revalidate(&self) -> Result<()> {
        use std::os::unix::fs::MetadataExt as _;
        let current = self.file.metadata()?;
        let expected = &self.metadata;
        ensure!(
            current.is_file()
                && current.dev() == expected.dev()
                && current.ino() == expected.ino()
                && current.mode() == expected.mode()
                && current.uid() == expected.uid()
                && current.gid() == expected.gid()
                && current.nlink() == expected.nlink()
                && current.len() == expected.len()
                && current.mtime() == expected.mtime()
                && current.mtime_nsec() == expected.mtime_nsec()
                && current.ctime() == expected.ctime()
                && current.ctime_nsec() == expected.ctime_nsec(),
            "Retained Task helper file identity changed"
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl TaskHelperIdentity {
    pub(crate) fn canonical_path(&self) -> &Path {
        &self.canonical
    }
    pub(crate) fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

#[cfg(target_os = "macos")]
fn helper_metadata_matches(left: fs::Metadata, right: fs::Metadata) -> bool {
    left.inode == right.inode
        && left.mtime == right.mtime
        && left.len == right.len
        && left.is_dir == right.is_dir
        && left.is_fifo == right.is_fifo
        && left.is_symlink == right.is_symlink
        && left.is_executable == right.is_executable
        && left.is_writable == right.is_writable
}

#[cfg(target_os = "macos")]
fn fingerprint_task_helper_reader(
    mut reader: impl Read,
    expected_bytes: u64,
    limit: u64,
    cancel: &AtomicBool,
) -> Result<[u8; 32]> {
    ensure!(
        expected_bytes <= limit,
        "Task helper exceeds its byte limit"
    );
    use sha2_helper::Digest as _;

    let mut digest = sha2_helper::Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut total = 0_u64;
    loop {
        cancelled(cancel)?;
        let remaining = limit
            .saturating_sub(total)
            .saturating_add(1)
            .min(buffer.len() as u64) as usize;
        let count = match reader.read(&mut buffer[..remaining]) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => anyhow::bail!("Cannot read the selected Task helper"),
        };
        cancelled(cancel)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("Task helper size overflow")?;
        ensure!(total <= limit, "Task helper grew beyond its byte limit");
        digest.update(&buffer[..count]);
    }
    ensure!(total == expected_bytes, "Task helper length changed");
    Ok(digest.finalize().into())
}

#[cfg(target_os = "macos")]
async fn fingerprint_task_helper_file(
    filesystem: &dyn fs::Fs,
    identity: &TaskHelperIdentity,
    initial_file: Option<&std::fs::File>,
    cancel: &AtomicBool,
) -> Result<[u8; 32]> {
    cancelled(cancel)?;
    let root_before = filesystem
        .metadata(&identity.root)
        .await?
        .context("Task helper root disappeared")?;
    ensure!(
        root_before.is_dir && !root_before.is_symlink && root_before.inode == identity.root_inode,
        "Task helper root identity changed"
    );
    let before = filesystem
        .metadata(&identity.canonical)
        .await?
        .context("Task helper disappeared")?;
    ensure!(
        helper_metadata_matches(before, identity.metadata),
        "Task helper identity changed"
    );
    let reader: Box<dyn Read + Send + Sync + '_> = match initial_file {
        Some(file) => Box::new(file),
        None => {
            filesystem
                .open_sync_confined_file(
                    &identity.root,
                    &identity.relative,
                    identity.root_inode,
                    identity.metadata.inode,
                )
                .await?
        }
    };
    let digest = fingerprint_task_helper_reader(
        reader,
        identity.metadata.len,
        MAX_TASK_HELPER_BYTES,
        cancel,
    )?;
    let after = filesystem
        .metadata(&identity.canonical)
        .await?
        .context("Task helper disappeared")?;
    ensure!(
        helper_metadata_matches(after, identity.metadata),
        "Task helper changed while reading"
    );
    let root_after = filesystem
        .metadata(&identity.root)
        .await?
        .context("Task helper root disappeared")?;
    ensure!(
        root_after.is_dir && !root_after.is_symlink && root_after.inode == identity.root_inode,
        "Task helper root identity changed while reading"
    );
    cancelled(cancel)?;
    Ok(digest)
}

#[cfg(target_os = "macos")]
pub(crate) async fn prepare_task_helper_identity(
    filesystem: &dyn fs::Fs,
    requested: PathBuf,
    cancel: &AtomicBool,
) -> Result<TaskHelperIdentity> {
    ensure!(
        requested.is_absolute()
            && requested
                .to_str()
                .is_some_and(|value| value.len() <= 4096 && !value.chars().any(char::is_control)),
        "Task helper path is invalid"
    );
    cancelled(cancel)?;
    let canonical = filesystem.canonicalize(&requested).await?;
    ensure!(
        canonical
            .to_str()
            .is_some_and(|value| value.len() <= 4096 && !value.chars().any(char::is_control)),
        "Canonical Task helper path is invalid"
    );
    let root = canonical
        .parent()
        .context("Task helper has no parent")?
        .to_path_buf();
    let relative = PathBuf::from(
        canonical
            .file_name()
            .context("Task helper has no filename")?,
    );
    let root_metadata = filesystem
        .metadata(&root)
        .await?
        .context("Task helper root disappeared")?;
    let metadata = filesystem
        .metadata(&canonical)
        .await?
        .context("Task helper disappeared")?;
    ensure!(
        root_metadata.is_dir
            && !root_metadata.is_symlink
            && !metadata.is_dir
            && !metadata.is_symlink
            && !metadata.is_fifo
            && metadata.is_executable
            && (1..=MAX_TASK_HELPER_BYTES).contains(&metadata.len),
        "Task helper is not a bounded ordinary executable"
    );
    let mut identity = TaskHelperIdentity {
        requested,
        canonical,
        root,
        root_inode: root_metadata.inode,
        relative,
        metadata,
        fingerprint: [0; 32],
        retained_file: None,
    };
    if !filesystem.is_fake() {
        use std::os::unix::fs::MetadataExt as _;
        let file = filesystem
            .open_sync_confined_file_handle(
                &identity.root,
                &identity.relative,
                identity.root_inode,
                identity.metadata.inode,
            )
            .await?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file()
                && metadata.ino() == identity.metadata.inode
                && metadata.len() == identity.metadata.len,
            "Retained Task helper file differs from the admitted identity"
        );
        identity.retained_file = Some(RetainedTaskHelperFile {
            file: std::sync::Arc::new(file),
            metadata,
        });
    }
    // FakeFs keeps its simulated reader; it must never mint a real image handle.
    identity.fingerprint = fingerprint_task_helper_file(
        filesystem,
        &identity,
        identity
            .retained_file
            .as_ref()
            .map(|retained| retained.file.as_ref()),
        cancel,
    )
    .await?;
    if let Some(retained) = &identity.retained_file {
        retained.revalidate()?;
    }
    ensure!(
        filesystem.canonicalize(&identity.requested).await? == identity.canonical,
        "Configured Task helper path changed"
    );
    Ok(identity)
}

#[cfg(target_os = "macos")]
pub(crate) async fn revalidate_task_helper_identity(
    filesystem: &dyn fs::Fs,
    identity: &TaskHelperIdentity,
    cancel: &AtomicBool,
) -> Result<()> {
    if let Some(retained) = &identity.retained_file {
        retained.revalidate()?;
    }
    ensure!(
        filesystem.canonicalize(&identity.requested).await? == identity.canonical,
        "Configured Task helper path changed"
    );
    ensure!(
        fingerprint_task_helper_file(filesystem, identity, None, cancel).await?
            == identity.fingerprint,
        "Task helper bytes changed"
    );
    if let Some(retained) = &identity.retained_file {
        retained.revalidate()?;
    }
    Ok(())
}

pub(crate) async fn fingerprint_profile_input(
    filesystem: &dyn fs::Fs,
    input: &ProfileInput,
    limit: usize,
    cancellation: &AtomicBool,
) -> Result<[u8; 32]> {
    Ok(input
        .read_with_fingerprint(filesystem, limit, cancellation)
        .await?
        .1)
}

pub async fn fingerprint_profile_inputs(
    fs: &dyn fs::Fs,
    inputs: &[ProfileInput; 2],
    cancel: &AtomicBool,
) -> Result<[[u8; 32]; 2]> {
    let mut fingerprints = [[0; 32]; 2];
    for ((input, limit), fingerprint) in inputs
        .iter()
        .zip([MAX_PROFILE_BYTES, MAX_SYMBOL_BYTES])
        .zip(&mut fingerprints)
    {
        *fingerprint = fingerprint_profile_input(fs, input, limit, cancel).await?;
    }
    Ok(fingerprints)
}

pub async fn load_samply_profile(
    fs: &dyn fs::Fs,
    inputs: [&ProfileInput; 2],
    host: ProfileHost,
    cancel: &AtomicBool,
    resolve_source: impl FnMut(&Path, u32) -> Option<ProjectPath>,
) -> Result<ProfileResults> {
    load_samply_profile_with_byte_limit(
        fs,
        inputs,
        host,
        cancel,
        MAX_PROFILE_BYTES + MAX_SYMBOL_BYTES,
        resolve_source,
    )
    .await
}

pub fn validate_profile_byte_limit(limit: usize) -> Result<()> {
    ensure!(
        (1..=MAX_PROFILE_BYTES + MAX_SYMBOL_BYTES).contains(&limit),
        "Profile byte limit is outside the supported range"
    );
    Ok(())
}

pub async fn load_samply_profile_with_byte_limit(
    fs: &dyn fs::Fs,
    inputs: [&ProfileInput; 2],
    host: ProfileHost,
    cancel: &AtomicBool,
    total_byte_limit: usize,
    mut resolve_source: impl FnMut(&Path, u32) -> Option<ProjectPath>,
) -> Result<ProfileResults> {
    validate_profile_byte_limit(total_byte_limit)?;
    #[cfg(any(test, feature = "test-support"))]
    SAMPLY_PROFILE_LOAD_CALLS.fetch_add(1, Ordering::Relaxed);
    ensure!(
        host == ProfileHost::MacOs,
        "Native profiling requires a macOS host"
    );
    cancelled(cancel)?;
    let profile = inputs[0]
        .open(fs)
        .await
        .map_err(|_| anyhow::anyhow!("Cannot open the selected profile file"))?;
    let profile = read_profile_input(profile, MAX_PROFILE_BYTES.min(total_byte_limit), cancel)?;
    let symbols = inputs[1]
        .open(fs)
        .await
        .map_err(|_| anyhow::anyhow!("Cannot open the selected symbol file"))?;
    let symbols = read_profile_input(
        symbols,
        MAX_SYMBOL_BYTES.min(total_byte_limit - profile.len()),
        cancel,
    )?;
    for (input, bytes) in inputs.iter().zip([profile.as_slice(), symbols.as_slice()]) {
        if let Some(expected) = input.fingerprint {
            ensure!(
                profile_fingerprint(bytes, cancel)? == expected,
                "Task profile artifact changed after completion"
            );
        }
    }
    let mut source_paths = HashMap::default();
    for path in samply_source_paths(&symbols, cancel)? {
        cancelled(cancel)?;
        // Missing source files are expected in portable captures and never become links.
        if let Ok(canonical) = fs.canonicalize(Path::new(&path)).await {
            source_paths.insert(path, canonical);
        }
    }
    cancelled(cancel)?;
    parse_samply_profile(&profile, &symbols, host, cancel, |path, line| {
        resolve_source(source_paths.get(path)?, line)
    })
}

pub fn samply_source_paths(symbols: &[u8], cancel: &AtomicBool) -> Result<Vec<String>> {
    ensure!(
        symbols.len() <= MAX_SYMBOL_BYTES,
        "Symbol input exceeds the byte limit"
    );
    cancelled(cancel)?;
    let symbols = parse_json(symbols, cancel, "Malformed symbol JSON")?;
    let strings = array(field(&symbols, "string_table")?)?;
    ensure!(
        strings.len() <= MAX_PROFILE_NODES,
        "Profile string table exceeds the limit"
    );
    let libraries = array(field(&symbols, "data")?)?;
    ensure!(libraries.len() <= 512, "Profile library limit exceeded");
    let mut paths = collections::HashSet::default();
    let mut count = 0usize;
    for library in libraries {
        for symbol in array(field(library, "symbol_table")?)? {
            cancelled(cancel)?;
            count = count.saturating_add(1);
            ensure!(count <= MAX_PROFILE_NODES, "Profile symbol limit exceeded");
            if let Some(frames) = symbol.get("frames") {
                let frames = array(frames)?;
                ensure!(
                    frames.len() <= MAX_DEPTH,
                    "Profile inline depth exceeds the limit"
                );
                for frame in frames {
                    if let Some(file) = frame.get("file").filter(|value| !value.is_null()) {
                        let path = text(at(strings, integer(file)?)?)?;
                        ensure!(path.len() <= 4096, "Profile string exceeds the byte limit");
                        if std::path::Path::new(path).is_absolute()
                            && !path.chars().any(char::is_control)
                            && !path.split(['/', '\\']).any(|component| component == "..")
                        {
                            paths.insert(path.to_owned());
                        }
                    }
                }
            }
        }
    }
    Ok(paths.into_iter().collect())
}

pub fn parse_samply_profile(
    profile: &[u8],
    symbols: &[u8],
    host: ProfileHost,
    cancel: &AtomicBool,
    mut source: impl FnMut(&str, u32) -> Option<ProjectPath>,
) -> Result<ProfileResults> {
    ensure!(
        host == ProfileHost::MacOs,
        "Native profiling is supported only on macOS"
    );
    ensure!(
        profile.len() <= MAX_PROFILE_BYTES && symbols.len() <= MAX_SYMBOL_BYTES,
        "Profile input exceeds the byte limit"
    );
    cancelled(cancel)?;
    let profile = parse_json(profile, cancel, "Malformed profile JSON")?;
    let symbols = parse_json(symbols, cancel, "Malformed symbol JSON")?;
    let meta = field(&profile, "meta")?;
    ensure!(
        integer(field(meta, "version")?)? == 24
            && integer(field(meta, "preprocessedProfileVersion")?)? == 49,
        "Unsupported processed profile version"
    );
    let threads = array(field(&profile, "threads")?)?;
    let libraries = array(field(&profile, "libs")?)?;
    let symbol_libraries = array(field(&symbols, "data")?)?;
    ensure!(
        threads.len() <= 64 && libraries.len() <= 512 && symbol_libraries.len() <= 512,
        "Profile thread or library limit exceeded"
    );
    let symbol_strings = array(field(&symbols, "string_table")?)?;
    let mut string_count = 0usize;
    let mut string_bytes = 0usize;
    for strings in std::iter::once(symbol_strings).chain(
        threads
            .iter()
            .map(|thread| array(field(thread, "stringArray")?))
            .collect::<Result<Vec<_>>>()?,
    ) {
        string_count = string_count.saturating_add(strings.len());
        for (index, value) in strings.iter().enumerate() {
            if index % 256 == 0 {
                cancelled(cancel)?;
            }
            let value = text(value)?;
            ensure!(value.len() <= 4096, "Profile string exceeds the byte limit");
            string_bytes = string_bytes.saturating_add(value.len());
        }
    }
    ensure!(
        string_count <= MAX_PROFILE_NODES && string_bytes <= 16 * 1024 * 1024,
        "Profile string table exceeds the limit"
    );
    let mut symbol_maps = HashMap::default();
    let mut symbol_count = 0usize;
    let mut address_count = 0usize;
    for library in symbol_libraries {
        cancelled(cancel)?;
        let key = identity(text(field(library, "code_id")?)?);
        ensure!(
            !key.is_empty() && !symbol_maps.contains_key(&key),
            "Duplicate or invalid symbol library identity"
        );
        let debug_id = identity(text(field(library, "debug_id")?)?);
        ensure!(
            key.len() == 32
                && debug_id.len() == 32
                && key.chars().all(|c| c.is_ascii_hexdigit())
                && debug_id.chars().all(|c| c.is_ascii_hexdigit()),
            "Unsupported macOS library identity"
        );
        ensure!(
            libraries.iter().any(|profile_library| {
                profile_library
                    .get("codeId")
                    .and_then(Value::as_str)
                    .is_some_and(|value| identity(value) == key)
                    && profile_library
                        .get("breakpadId")
                        .and_then(Value::as_str)
                        .is_some_and(|value| identity(value) == format!("{debug_id}0"))
                    && profile_library.get("debugName").and_then(Value::as_str)
                        == library.get("debug_name").and_then(Value::as_str)
            }),
            "Symbol library identity does not match the profile"
        );
        let table = array(field(library, "symbol_table")?)?;
        symbol_count = symbol_count.saturating_add(table.len());
        ensure!(
            symbol_count <= MAX_PROFILE_NODES,
            "Profile symbol limit exceeded"
        );
        let mut frames = Vec::with_capacity(table.len());
        for symbol in table {
            cancelled(cancel)?;
            let address = integer(field(symbol, "rva")?)?;
            ensure!(
                address <= u32::MAX as usize,
                "Symbol address exceeds the supported range"
            );
            if let Some(size) = symbol.get("size").filter(|value| !value.is_null()) {
                ensure!(
                    address
                        .checked_add(integer(size)?)
                        .is_some_and(|end| end <= u32::MAX as usize),
                    "Symbol size exceeds the supported range"
                );
            }
            let name = string_at(symbol_strings, field(symbol, "symbol")?)?;
            let mut location = None;
            if let Some(inline) = symbol.get("frames") {
                let inline = array(inline)?;
                ensure!(
                    inline.len() <= MAX_DEPTH,
                    "Profile inline depth exceeds the limit"
                );
                for frame in inline.iter().rev() {
                    if let Some(function) = frame.get("function").filter(|value| !value.is_null()) {
                        string_at(symbol_strings, function)?;
                    }
                    if let (Some(file), Some(line)) = (
                        frame.get("file").filter(|value| !value.is_null()),
                        frame.get("line").filter(|value| !value.is_null()),
                    ) {
                        let path = string_at(symbol_strings, file)?;
                        let line = u32::try_from(integer(line)?).context("Invalid source line")?;
                        ensure!(line > 0, "Source lines must be positive");
                        if location.is_none()
                            && !path.chars().any(char::is_control)
                            && !path.split(['/', '\\']).any(|component| component == "..")
                        {
                            location = source(&path, line).map(|path| (path, line));
                        }
                    }
                }
            }
            frames.push(ProfileFrame {
                label: label(&name),
                source: location,
                resolved: true,
            });
        }
        let addresses = array(field(library, "known_addresses")?)?;
        address_count = address_count.saturating_add(addresses.len());
        ensure!(
            address_count <= MAX_PROFILE_NODES,
            "Profile address limit exceeded"
        );
        let mut mapping = HashMap::default();
        for address in addresses {
            cancelled(cancel)?;
            let pair = array(address)?;
            ensure!(pair.len() == 2, "Invalid symbol address pair");
            let address = integer(at(pair, 0)?)?;
            ensure!(
                address <= u32::MAX as usize,
                "Symbol address exceeds the supported range"
            );
            let symbol = at(table, integer(at(pair, 1)?)?)?;
            let start = integer(field(symbol, "rva")?)?;
            ensure!(address >= start, "Symbol address precedes its function");
            if let Some(size) = symbol.get("size").filter(|value| !value.is_null()) {
                ensure!(
                    address.saturating_sub(start) < integer(size)?,
                    "Symbol address exceeds its function"
                );
            }
            let frame = frames
                .get(integer(at(pair, 1)?)?)
                .context("Symbol index is out of range")?
                .clone();
            ensure!(
                mapping.insert(address, frame).is_none(),
                "Duplicate symbol address"
            );
        }
        symbol_maps.insert(key, mapping);
    }
    for key in symbol_maps.keys() {
        ensure!(
            libraries.iter().any(|library| library
                .get("codeId")
                .and_then(Value::as_str)
                .is_some_and(|value| identity(value) == *key)),
            "Symbol sidecar does not match the profile"
        );
    }
    let mut result = ProfileResults {
        frames: Vec::new(),
        nodes: Vec::new(),
        roots: Vec::new(),
        total_samples: 0,
        unattributed_samples: 0,
        unresolved_frames: 0,
    };
    let mut frame_ids = HashMap::default();
    let mut nodes = HashMap::default();
    let mut samples_seen = 0usize;
    let mut table_entries = [0usize; 4];
    for thread in threads {
        cancelled(cancel)?;
        let frames = field(thread, "frameTable")?;
        let functions = field(thread, "funcTable")?;
        let resources = field(thread, "resourceTable")?;
        let stacks = field(thread, "stackTable")?;
        let frame_functions = column(frames, "func")?;
        let addresses = column(frames, "address")?;
        let names = column(functions, "name")?;
        let function_resources = column(functions, "resource")?;
        let resource_libraries = column(resources, "lib")?;
        let prefixes = column(stacks, "prefix")?;
        let stack_frames = column(stacks, "frame")?;
        let strings = array(field(thread, "stringArray")?)?;
        for (total, count) in table_entries.iter_mut().zip([
            frame_functions.len(),
            prefixes.len(),
            names.len(),
            resource_libraries.len(),
        ]) {
            *total = total.saturating_add(count);
            ensure!(
                *total <= MAX_PROFILE_NODES,
                "Profile tables exceed the total limit"
            );
        }
        let mut mapped_frames = Vec::with_capacity(frame_functions.len());
        for (index, function) in frame_functions.iter().enumerate() {
            if index % 256 == 0 {
                cancelled(cancel)?;
            }
            let function = integer(function)?;
            let fallback = string_at(strings, at(names, function)?)?;
            let resource = at(function_resources, function)?;
            let address = integer(at(addresses, index)?)?;
            let mut library_identity = None;
            let resolved = if resource.as_i64().is_some_and(|value| value < 0) {
                None
            } else {
                let library = at(resource_libraries, integer(resource)?)?;
                if library.is_null() {
                    None
                } else {
                    let library_index = integer(library)?;
                    library_identity = Some(library_index);
                    let library = at(libraries, library_index)?;
                    let key = identity(text(field(library, "codeId")?)?);
                    symbol_maps
                        .get(&key)
                        .and_then(|symbols| symbols.get(&address))
                        .cloned()
                }
            };
            let frame = resolved.unwrap_or_else(|| ProfileFrame {
                label: label(&fallback),
                source: None,
                resolved: false,
            });
            let key = (
                library_identity,
                frame.label.clone(),
                frame.source.clone(),
                frame.resolved,
            );
            let id = if let Some(id) = frame_ids.get(&key) {
                *id
            } else {
                ensure!(
                    result.frames.len() < MAX_PROFILE_NODES,
                    "Profile frame limit exceeded"
                );
                let id = result.frames.len();
                result.unresolved_frames += usize::from(!frame.resolved);
                result.frames.push(frame);
                frame_ids.insert(key, id);
                id
            };
            mapped_frames.push(id);
        }
        let mut depths = Vec::with_capacity(prefixes.len());
        for (index, prefix) in prefixes.iter().enumerate() {
            if index % 256 == 0 {
                cancelled(cancel)?;
            }
            let depth = if prefix.is_null() {
                1
            } else {
                let prefix = integer(prefix)?;
                ensure!(prefix < index, "Profile stack prefix is cyclic or forward");
                depths
                    .get(prefix)
                    .copied()
                    .context("Invalid stack prefix")?
                    + 1
            };
            ensure!(depth <= MAX_DEPTH, "Profile stack depth exceeds the limit");
            ensure!(
                integer(at(stack_frames, index)?)? < frame_functions.len(),
                "Profile stack frame is out of range"
            );
            depths.push(depth);
        }
        let samples = field(thread, "samples")?;
        ensure!(
            text(field(samples, "weightType")?)? == "samples",
            "Unsupported profile sample units"
        );
        let sample_stacks = array(field(samples, "stack")?)?;
        let weights = array(field(samples, "weight")?)?;
        ensure!(
            integer(field(samples, "length")?)? == sample_stacks.len()
                && weights.len() == sample_stacks.len(),
            "Profile sample columns differ in length"
        );
        samples_seen = samples_seen.saturating_add(sample_stacks.len());
        ensure!(
            samples_seen <= MAX_SAMPLES,
            "Profile sample count exceeds the limit"
        );
        for (index, stack) in sample_stacks.iter().enumerate() {
            if index % 256 == 0 {
                cancelled(cancel)?;
            }
            let weight = at(weights, index)?
                .as_u64()
                .context("Invalid profile sample weight")?;
            if stack.is_null() {
                result.unattributed_samples = result
                    .unattributed_samples
                    .checked_add(weight)
                    .context("Profile weight overflow")?;
                result.total_samples = result
                    .total_samples
                    .checked_add(weight)
                    .context("Profile weight overflow")?;
                continue;
            }
            let mut current = Some(integer(stack)?);
            let mut chain = Vec::new();
            while let Some(stack) = current {
                chain.push(
                    *mapped_frames
                        .get(integer(at(stack_frames, stack)?)?)
                        .context("Invalid mapped frame")?,
                );
                let prefix = at(prefixes, stack)?;
                current = if prefix.is_null() {
                    None
                } else {
                    Some(integer(prefix)?)
                };
            }
            let mut parent = None;
            for frame in chain.into_iter().rev() {
                let key = (parent, frame);
                let node = if let Some(node) = nodes.get(&key) {
                    *node
                } else {
                    ensure!(
                        result.nodes.len() < MAX_PROFILE_NODES,
                        "Profile aggregate node limit exceeded"
                    );
                    let node = result.nodes.len();
                    result.nodes.push(ProfileNode {
                        frame,
                        parent,
                        children: Vec::new(),
                        inclusive_samples: 0,
                        self_samples: 0,
                    });
                    if let Some(parent) = parent {
                        result
                            .nodes
                            .get_mut(parent)
                            .context("Invalid aggregate parent")?
                            .children
                            .push(node);
                    } else {
                        result.roots.push(node);
                    }
                    nodes.insert(key, node);
                    node
                };
                let entry = result
                    .nodes
                    .get_mut(node)
                    .context("Invalid aggregate node")?;
                entry.inclusive_samples = entry
                    .inclusive_samples
                    .checked_add(weight)
                    .context("Profile weight overflow")?;
                parent = Some(node);
            }
            if let Some(node) = parent {
                let entry = result
                    .nodes
                    .get_mut(node)
                    .context("Invalid aggregate leaf")?;
                entry.self_samples = entry
                    .self_samples
                    .checked_add(weight)
                    .context("Profile weight overflow")?;
            }
            result.total_samples = result
                .total_samples
                .checked_add(weight)
                .context("Profile weight overflow")?;
        }
    }
    cancelled(cancel)?;
    let bytes = result
        .frames
        .iter()
        .map(|frame| {
            frame.label.len()
                + frame
                    .source
                    .as_ref()
                    .map_or(0, |(path, _)| path.path.as_unix_str().len())
                + 64
        })
        .sum::<usize>()
        + result.nodes.len() * 64;
    ensure!(
        bytes <= 16 * 1024 * 1024,
        "Normalized profile exceeds the result limit"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn load_test_samply_profile(
        fs: &dyn fs::Fs,
        paths: [&Path; 2],
        host: ProfileHost,
        cancel: &AtomicBool,
        resolve_source: impl FnMut(&Path, u32) -> Option<ProjectPath>,
    ) -> Result<ProfileResults> {
        ensure!(
            host == ProfileHost::MacOs,
            "Native profiling requires a macOS host"
        );
        let mut inputs = Vec::new();
        for (path, error) in paths.into_iter().zip([
            "Cannot open the selected profile file",
            "Cannot open the selected symbol file",
        ]) {
            let root = path.parent().context(error)?;
            inputs.push(ProfileInput {
                fingerprint: None,
                root: root.to_path_buf(),
                path: path.strip_prefix(root)?.to_path_buf(),
                root_inode: fs.metadata(root).await?.context(error)?.inode,
                file_inode: fs.metadata(path).await?.context(error)?.inode,
            });
        }
        load_samply_profile(fs, [&inputs[0], &inputs[1]], host, cancel, resolve_source).await
    }

    #[gpui::test]
    async fn doctor_profile_reader_enforces_combined_limit_and_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/doctor-profile", serde_json::json!({}))
            .await;
        fs.insert_file("/doctor-profile/profile.json", PROFILE.to_vec())
            .await;
        fs.insert_file("/doctor-profile/profile.syms.json", SYMBOLS.to_vec())
            .await;
        let root = Path::new("/doctor-profile");
        let mut inputs = Vec::new();
        for path in ["profile.json", "profile.syms.json"] {
            inputs.push(ProfileInput {
                root: root.into(),
                path: path.into(),
                root_inode: fs
                    .metadata(root)
                    .await
                    .expect("metadata")
                    .expect("root")
                    .inode,
                file_inode: fs
                    .metadata(&root.join(path))
                    .await
                    .expect("metadata")
                    .expect("file")
                    .inode,
                fingerprint: None,
            });
        }
        let cancel = AtomicBool::new(false);
        let size = PROFILE.len() + SYMBOLS.len();
        assert!(
            load_samply_profile_with_byte_limit(
                fs.as_ref(),
                [&inputs[0], &inputs[1]],
                ProfileHost::MacOs,
                &cancel,
                size,
                |_, _| None
            )
            .await
            .is_ok()
        );
        for limit in [0, 1, size - 1, MAX_PROFILE_BYTES + MAX_SYMBOL_BYTES + 1] {
            assert!(
                load_samply_profile_with_byte_limit(
                    fs.as_ref(),
                    [&inputs[0], &inputs[1]],
                    ProfileHost::MacOs,
                    &cancel,
                    limit,
                    |_, _| None
                )
                .await
                .is_err()
            );
        }
        cancel.store(true, Ordering::Relaxed);
        assert!(
            load_samply_profile_with_byte_limit(
                fs.as_ref(),
                [&inputs[0], &inputs[1]],
                ProfileHost::MacOs,
                &cancel,
                size,
                |_, _| None
            )
            .await
            .is_err()
        );
        assert!(
            load_samply_profile_with_byte_limit(
                fs.as_ref(),
                [&inputs[0], &inputs[1]],
                ProfileHost::Linux,
                &AtomicBool::new(false),
                size,
                |_, _| None
            )
            .await
            .is_err()
        );
    }

    const PROFILE: &[u8] = include_bytes!(
        "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
    );
    const SYMBOLS: &[u8] = include_bytes!(
        "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"
    );

    #[test]
    fn profile_results_input_reader_bounds_cancellation_and_error_privacy() {
        let cancel = AtomicBool::new(false);
        assert_eq!(
            read_profile_input(&b"abcd"[..], 4, &cancel).expect("exact limit"),
            b"abcd"
        );
        assert!(read_profile_input(&b"abcde"[..], 4, &cancel).is_err());
        assert!(read_profile_input(&b"x"[..], 0, &cancel).is_err());
        assert!(
            read_profile_input(&b""[..], 0, &cancel)
                .expect("empty input")
                .is_empty()
        );

        struct InterruptedReader(bool);
        impl Read for InterruptedReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                if std::mem::replace(&mut self.0, false) {
                    Err(std::io::ErrorKind::Interrupted.into())
                } else {
                    Err(std::io::Error::other("/private/secret/profile.json"))
                }
            }
        }
        let error =
            read_profile_input(InterruptedReader(true), 4, &cancel).expect_err("read failure");
        assert_eq!(
            format!("{error:#}"),
            "Cannot read the selected profile input"
        );

        struct CancellingReader<'a>(&'a AtomicBool);
        impl Read for CancellingReader<'_> {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                self.0.store(true, Ordering::Relaxed);
                bytes.fill(0);
                Ok(bytes.len())
            }
        }
        let error = read_profile_input(CancellingReader(&cancel), 100_000, &cancel)
            .expect_err("mid-read cancellation");
        assert_eq!(error.to_string(), "Profile import was cancelled");
        assert!(read_profile_input(&b""[..], 0, &cancel).is_err());
    }

    #[gpui::test]
    async fn profile_results_single_artifact_fingerprint_obeys_declared_budget(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let filesystem = fs::FakeFs::new(cx.background_executor.clone());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"target": {"coverage.json": "abcd"}}),
            )
            .await;
        let root = filesystem
            .metadata(Path::new("/workspace"))
            .await
            .expect("root metadata")
            .expect("root exists");
        let file = filesystem
            .metadata(Path::new("/workspace/target/coverage.json"))
            .await
            .expect("report metadata")
            .expect("report exists");
        let input = ProfileInput {
            root: "/workspace".into(),
            path: "target/coverage.json".into(),
            root_inode: root.inode,
            file_inode: file.inode,
            fingerprint: None,
        };
        let cancel = AtomicBool::new(false);
        let fingerprint = fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
            .await
            .expect("exact declared single-artifact budget");
        assert_eq!(
            fingerprint,
            profile_fingerprint(b"abcd", &cancel).expect("expected digest")
        );
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 3, &cancel)
                .await
                .is_err()
        );
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 0, &cancel)
                .await
                .is_err()
        );
        assert!(
            fingerprint_profile_input(
                filesystem.as_ref(),
                &input,
                task::MAX_TASK_ARTIFACT_BYTES as usize + 1,
                &cancel,
            )
            .await
            .is_err()
        );
        cancel.store(true, Ordering::Relaxed);
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
                .await
                .is_err()
        );
    }

    #[gpui::test]
    async fn profile_results_single_artifact_fingerprint_detects_same_inode_content_mutation(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let filesystem = fs::FakeFs::new(cx.background_executor.clone());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"target": {"coverage.json": "abcd"}}),
            )
            .await;
        let root = filesystem
            .metadata(Path::new("/workspace"))
            .await
            .expect("root metadata")
            .expect("root exists");
        let path = Path::new("/workspace/target/coverage.json");
        let before = filesystem
            .metadata(path)
            .await
            .expect("report metadata")
            .expect("report exists");
        let input = ProfileInput {
            root: "/workspace".into(),
            path: "target/coverage.json".into(),
            root_inode: root.inode,
            file_inode: before.inode,
            fingerprint: None,
        };
        let cancel = AtomicBool::new(false);
        let sealed = fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
            .await
            .expect("original confined report");
        filesystem
            .write(path, b"dcba")
            .await
            .expect("same-inode same-size rewrite");
        let after = filesystem
            .metadata(path)
            .await
            .expect("rewritten metadata")
            .expect("report remains");
        assert_eq!(before.inode, after.inode);
        assert_eq!(before.len, after.len);
        let changed = fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
            .await
            .expect("same admitted descriptor identity");
        assert_ne!(
            sealed, changed,
            "content digest rejects metadata-compatible bytes"
        );
        assert_eq!(
            changed,
            profile_fingerprint(b"dcba", &cancel).expect("actual changed digest")
        );
    }

    #[gpui::test]
    async fn profile_results_single_artifact_fingerprint_rejects_replaced_entry_and_symlink(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        let filesystem = fs::FakeFs::new(cx.background_executor.clone());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"target": {"coverage.json": "abcd"}}),
            )
            .await;
        filesystem
            .insert_tree("/outside", serde_json::json!({"coverage.json": "abcd"}))
            .await;
        let root = filesystem
            .metadata(Path::new("/workspace"))
            .await
            .expect("root metadata")
            .expect("root exists");
        let path = Path::new("/workspace/target/coverage.json");
        let before = filesystem
            .metadata(path)
            .await
            .expect("report metadata")
            .expect("report exists");
        let input = ProfileInput {
            root: "/workspace".into(),
            path: "target/coverage.json".into(),
            root_inode: root.inode,
            file_inode: before.inode,
            fingerprint: None,
        };
        let cancel = AtomicBool::new(false);
        let initial = fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
            .await
            .expect("admitted initial report");
        assert_eq!(
            initial,
            profile_fingerprint(b"abcd", &cancel).expect("initial digest")
        );
        filesystem
            .remove_file(path, fs::RemoveOptions::default())
            .await
            .expect("remove admitted report");
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
                .await
                .is_err()
        );
        filesystem.insert_file(path, b"abcd".to_vec()).await;
        let replaced = filesystem
            .metadata(path)
            .await
            .expect("replacement metadata")
            .expect("replacement exists");
        assert_ne!(replaced.inode, before.inode);
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
                .await
                .is_err()
        );
        filesystem
            .remove_file(path, fs::RemoveOptions::default())
            .await
            .expect("remove replacement");
        filesystem
            .create_symlink(path, "/outside/coverage.json".into())
            .await
            .expect("external alias");
        assert!(
            fingerprint_profile_input(filesystem.as_ref(), &input, 4, &cancel)
                .await
                .is_err()
        );
    }

    #[gpui::test]
    async fn profile_results_shared_loader_obeys_host_and_file_boundaries(
        cx: &mut gpui::TestAppContext,
    ) {
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_file("/profile.json", PROFILE.to_vec()).await;
        fs.insert_file("/profile.syms.json", SYMBOLS.to_vec()).await;
        let cancel = AtomicBool::new(false);
        let inputs = [Path::new("/profile.json"), Path::new("/profile.syms.json")];
        let result =
            load_test_samply_profile(fs.as_ref(), inputs, ProfileHost::MacOs, &cancel, |_, _| {
                None
            })
            .await
            .expect("captured input");
        assert_eq!(result.total_samples, 30);
        assert!(result.frames.iter().all(|frame| frame.source.is_none()));
        let missing_input = ProfileInput {
            fingerprint: None,
            root: "/secret".into(),
            path: "missing".into(),
            root_inode: 0,
            file_inode: 0,
        };
        for host in [ProfileHost::Linux, ProfileHost::Windows] {
            let error =
                load_samply_profile(fs.as_ref(), [&missing_input; 2], host, &cancel, |_, _| None)
                    .await
                    .expect_err("unsupported authoritative host");
            assert_eq!(error.to_string(), "Native profiling requires a macOS host");
        }
        let error = load_test_samply_profile(
            fs.as_ref(),
            [Path::new("/secret/missing"); 2],
            ProfileHost::MacOs,
            &cancel,
            |_, _| None,
        )
        .await
        .expect_err("missing input");
        assert_eq!(
            format!("{error:#}"),
            "Cannot open the selected profile file"
        );
        fs.insert_file("/profile.json", vec![0; MAX_PROFILE_BYTES + 1])
            .await;
        let error =
            load_test_samply_profile(fs.as_ref(), inputs, ProfileHost::MacOs, &cancel, |_, _| {
                None
            })
            .await
            .expect_err("oversized input");
        assert_eq!(error.to_string(), "Profile input exceeds the byte limit");
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_results_real_loader_rejects_fifo_and_symlink_replacements(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let directory = tempfile::TempDir::new().expect("profile fixture directory");
        let profile = directory.path().join("profile.json");
        let symbols = directory.path().join("profile.syms.json");
        std::fs::write(&profile, PROFILE).expect("profile fixture");
        std::fs::write(&symbols, SYMBOLS).expect("symbols fixture");
        let fs = fs::RealFs::new(None, cx.background_executor.clone());
        let cancel = AtomicBool::new(false);
        let imported = load_test_samply_profile(
            &fs,
            [&profile, &symbols],
            ProfileHost::MacOs,
            &cancel,
            |_, _| None,
        )
        .await
        .expect("regular capture files");
        assert_eq!(imported.total_samples, 30);

        for (path, bytes, expected) in [
            (&profile, PROFILE, "Cannot open the selected profile file"),
            (&symbols, SYMBOLS, "Cannot open the selected symbol file"),
        ] {
            std::fs::remove_file(path).expect("replace admitted regular file");
            assert!(
                smol::process::Command::new("/usr/bin/mkfifo")
                    .arg(path)
                    .status()
                    .await
                    .expect("create FIFO fixture")
                    .success()
            );
            let error = load_test_samply_profile(
                &fs,
                [&profile, &symbols],
                ProfileHost::MacOs,
                &cancel,
                |_, _| None,
            )
            .await
            .expect_err("FIFO without a writer must be rejected");
            assert_eq!(format!("{error:#}"), expected);
            std::fs::remove_file(path).expect("remove FIFO");
            let replacement = directory.path().join("replacement.json");
            std::fs::write(&replacement, bytes).expect("replacement contents");
            std::os::unix::fs::symlink(&replacement, path).expect("replacement symlink");
            let error = load_test_samply_profile(
                &fs,
                [&profile, &symbols],
                ProfileHost::MacOs,
                &cancel,
                |_, _| None,
            )
            .await
            .expect_err("replacement symlink must be rejected");
            assert_eq!(format!("{error:#}"), expected);
            std::fs::remove_file(path).expect("remove symlink");
            std::fs::write(path, bytes).expect("restore regular fixture");
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_results_confined_loader_rejects_replaced_parents_and_root(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::{MetadataExt, symlink};
        cx.executor().allow_parking();
        let directory = tempfile::TempDir::new().expect("fixture");
        let root = directory.path().join("workspace");
        let parent = root.join("target");
        std::fs::create_dir_all(&parent).expect("artifact directory");
        std::fs::write(parent.join("profile.json"), PROFILE).expect("profile");
        std::fs::write(parent.join("profile.syms.json"), SYMBOLS).expect("symbols");
        let inputs = ["profile.json", "profile.syms.json"].map(|name| ProfileInput {
            fingerprint: None,
            root: root.clone(),
            path: Path::new("target").join(name),
            root_inode: std::fs::metadata(&root).expect("root metadata").ino(),
            file_inode: std::fs::metadata(parent.join(name))
                .expect("file metadata")
                .ino(),
        });
        let fs = fs::RealFs::new(None, cx.background_executor.clone());
        let cancel = AtomicBool::new(false);
        let load = || {
            load_samply_profile(
                &fs,
                [&inputs[0], &inputs[1]],
                ProfileHost::MacOs,
                &cancel,
                |_, _| None,
            )
        };
        assert_eq!(load().await.expect("admitted files").total_samples, 30);

        let moved = directory.path().join("moved-artifacts");
        std::fs::rename(&parent, &moved).expect("move admitted directory");
        symlink(&moved, &parent).expect("replace parent with symlink");
        assert_eq!(
            format!("{:#}", load().await.expect_err("parent symlink")),
            "Cannot open the selected profile file"
        );
        std::fs::remove_file(&parent).expect("remove symlink");
        std::fs::create_dir(&parent).expect("replacement directory");
        std::fs::write(parent.join("profile.json"), PROFILE)
            .expect("different file with identical bytes");
        std::fs::write(parent.join("profile.syms.json"), SYMBOLS).expect("different symbols");
        assert_eq!(
            format!("{:#}", load().await.expect_err("different file identity")),
            "Cannot open the selected profile file"
        );
        std::fs::remove_dir_all(&parent).expect("remove replacement");
        std::fs::rename(&moved, &parent).expect("restore admitted files");
        assert_eq!(
            load().await.expect("original files restored").total_samples,
            30
        );

        let moved_root = directory.path().join("moved-workspace");
        std::fs::rename(&root, &moved_root).expect("move root");
        std::fs::create_dir_all(&parent).expect("replacement root");
        for name in ["profile.json", "profile.syms.json"] {
            std::fs::hard_link(moved_root.join("target").join(name), parent.join(name))
                .expect("retain artifact identities under a different root");
        }
        assert_eq!(
            format!("{:#}", load().await.expect_err("root identity changed")),
            "Cannot open the selected profile file"
        );
        std::fs::remove_dir_all(&root).expect("remove replacement root");
        symlink(&moved_root, &root).expect("root symlink");
        assert_eq!(
            format!("{:#}", load().await.expect_err("root symlink")),
            "Cannot open the selected profile file"
        );
    }

    #[test]
    fn profile_results_mac_capture_aggregates_and_discards_private_data() {
        let result = parse_samply_profile(
            PROFILE,
            SYMBOLS,
            ProfileHost::MacOs,
            &AtomicBool::new(false),
            |_, _| None,
        )
        .expect("captured macOS profile should parse");
        assert_eq!(result.total_samples, 30);
        assert!(
            result
                .frames
                .iter()
                .any(|frame| frame.label == "workload::leaf" && frame.resolved)
        );
        assert!(result.frames.iter().all(|frame| frame.source.is_none()));
        assert_eq!(
            result
                .nodes
                .iter()
                .map(|node| node.self_samples)
                .sum::<u64>(),
            30
        );
        assert!(!format!("{result:?}").contains("/fixture"));
        assert_eq!(
            result,
            parse_samply_profile(
                PROFILE,
                SYMBOLS,
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |_, _| None
            )
            .expect("repeat parse")
        );
    }

    #[test]
    fn profile_results_stops_when_cancelled_during_symbol_conversion() {
        let cancelled = AtomicBool::new(false);
        let mut source_lookups = 0;
        let error =
            parse_samply_profile(PROFILE, SYMBOLS, ProfileHost::MacOs, &cancelled, |_, _| {
                source_lookups += 1;
                cancelled.store(true, Ordering::Relaxed);
                None
            })
            .expect_err("cancellation during symbol conversion must discard the partial result");
        assert!(
            source_lookups > 0,
            "the parser did not enter source conversion"
        );
        assert_eq!(error.to_string(), "Profile import was cancelled");
    }

    #[test]
    fn profile_results_stops_when_cancelled_during_json_decoding() {
        struct CancelAfterFirstRead<'a> {
            bytes: &'a [u8],
            cancel: &'a AtomicBool,
            reads: usize,
        }

        impl Read for CancelAfterFirstRead<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let count = self.bytes.read(buffer)?;
                if count > 0 {
                    self.reads += 1;
                    if self.reads == 1 {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                }
                Ok(count)
            }
        }

        let cancel = AtomicBool::new(false);
        let bytes = format!("{{\"payload\":\"{}\"}}", "a".repeat(128 * 1024));
        let error = parse_json(
            CancelAfterFirstRead {
                bytes: bytes.as_bytes(),
                cancel: &cancel,
                reads: 0,
            },
            &cancel,
            "Malformed profile JSON",
        )
        .expect_err("cancellation during JSON decoding must discard the partial value");
        assert_eq!(error.to_string(), "Profile import was cancelled");
    }

    #[test]
    fn profile_results_validates_sources_without_retaining_raw_paths() {
        let original: Value = serde_json::from_slice(SYMBOLS).expect("symbol fixture");
        for unsafe_path in ["/fixture/../private.rs", "/fixture/private\nname.rs"] {
            let mut symbols = original.clone();
            for value in symbols["string_table"].as_array_mut().expect("strings") {
                if value.as_str().is_some_and(|value| value.starts_with('/')) {
                    *value = Value::from(unsafe_path);
                }
            }
            let mut mapped_paths = Vec::new();
            let result = parse_samply_profile(
                PROFILE,
                &serde_json::to_vec(&symbols).expect("JSON"),
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |path, _| {
                    mapped_paths.push(path.to_string());
                    None
                },
            )
            .expect("unsafe sources are omitted");
            assert!(!mapped_paths.iter().any(|path| path == unsafe_path));
            assert!(result.frames.iter().all(|frame| frame.source.is_none()));
            assert!(!format!("{result:?}").contains(unsafe_path));
        }
        let mut symbols = original;
        let source = symbols["data"]
            .as_array_mut()
            .expect("libraries")
            .iter_mut()
            .flat_map(|library| library["symbol_table"].as_array_mut().expect("symbols"))
            .find_map(|symbol| {
                symbol
                    .get_mut("frames")
                    .and_then(Value::as_array_mut)
                    .and_then(|frames| frames.first_mut())
            })
            .expect("source frame");
        source["line"] = Value::from(0);
        assert!(
            parse_samply_profile(
                PROFILE,
                &serde_json::to_vec(&symbols).expect("JSON"),
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |_, _| None
            )
            .is_err()
        );
    }

    #[test]
    fn profile_results_rejects_unsupported_cancelled_and_malformed_inputs() {
        for host in [ProfileHost::Linux, ProfileHost::Windows] {
            assert!(
                parse_samply_profile(PROFILE, SYMBOLS, host, &AtomicBool::new(false), |_, _| None)
                    .is_err()
            );
        }
        assert!(
            parse_samply_profile(
                PROFILE,
                SYMBOLS,
                ProfileHost::MacOs,
                &AtomicBool::new(true),
                |_, _| None
            )
            .is_err()
        );
        for input in [b"{".as_slice(), b"\xff", b"{}"] {
            assert!(
                parse_samply_profile(
                    input,
                    SYMBOLS,
                    ProfileHost::MacOs,
                    &AtomicBool::new(false),
                    |_, _| None
                )
                .is_err()
            );
        }
        assert!(
            parse_samply_profile(
                &vec![b' '; MAX_PROFILE_BYTES + 1],
                SYMBOLS,
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |_, _| None
            )
            .is_err()
        );
    }

    #[test]
    fn profile_results_rejects_forward_versions_cycles_indexes_and_sidecar_mismatch() {
        let original: Value = serde_json::from_slice(PROFILE).expect("fixture JSON");
        for invalid in [
            ("meta", "preprocessedProfileVersion", Value::from(50)),
            ("meta", "version", Value::from(25)),
        ] {
            let mut profile = original.clone();
            profile[invalid.0][invalid.1] = invalid.2;
            assert!(
                parse_samply_profile(
                    &serde_json::to_vec(&profile).expect("JSON"),
                    SYMBOLS,
                    ProfileHost::MacOs,
                    &AtomicBool::new(false),
                    |_, _| None
                )
                .is_err()
            );
        }
        for (table, column, value) in [
            ("stackTable", "prefix", Value::from(0)),
            ("stackTable", "frame", Value::from(usize::MAX)),
            ("samples", "weight", Value::from(-1)),
        ] {
            let mut profile = original.clone();
            profile["threads"][0][table][column][0] = value;
            assert!(
                parse_samply_profile(
                    &serde_json::to_vec(&profile).expect("JSON"),
                    SYMBOLS,
                    ProfileHost::MacOs,
                    &AtomicBool::new(false),
                    |_, _| None
                )
                .is_err()
            );
        }
        let mut symbols: Value = serde_json::from_slice(SYMBOLS).expect("fixture JSON");
        symbols["data"][0]["code_id"] = Value::from("unrelated-library");
        assert!(
            parse_samply_profile(
                PROFILE,
                &serde_json::to_vec(&symbols).expect("JSON"),
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |_, _| None
            )
            .is_err()
        );
    }
    #[test]
    fn profile_results_bounds_large_tables_and_redacts_symbol_literals() {
        let mut profile: Value = serde_json::from_slice(PROFILE).expect("fixture JSON");
        let count = 10_001usize;
        let thread = &mut profile["threads"][0];
        thread["stringArray"] = serde_json::json!(
            (0..count)
                .map(|index| format!("frame_{index}"))
                .collect::<Vec<_>>()
        );
        thread["frameTable"] = serde_json::json!({"length": count, "func": (0..count).collect::<Vec<_>>(), "address": (0..count).collect::<Vec<_>>()});
        thread["funcTable"] = serde_json::json!({"length": count, "name": (0..count).collect::<Vec<_>>(), "resource": vec![-1; count]});
        let mut prefixes = vec![Value::from(0); count];
        *prefixes.first_mut().expect("root") = Value::Null;
        thread["stackTable"] = serde_json::json!({"length": count, "prefix": prefixes, "frame": (0..count).collect::<Vec<_>>()});
        thread["samples"] = serde_json::json!({"length": count - 1, "weightType": "samples", "stack": (1..count).collect::<Vec<_>>(), "weight": vec![1; count - 1]});
        let bytes = serde_json::to_vec(&profile).expect("large fixture JSON");
        let result = parse_samply_profile(
            &bytes,
            SYMBOLS,
            ProfileHost::MacOs,
            &AtomicBool::new(false),
            |_, _| None,
        )
        .expect("bounded large profile");
        assert_eq!(result.total_samples, 10_000);
        assert_eq!(result.nodes.len(), count);
        assert_eq!(result.frames.len(), count);
        assert_eq!(label("function<\"secret-value\">"), "<redacted symbol>");
        assert_eq!(label("/private/source.rs"), "<redacted symbol>");
        assert!(!label(&"é".repeat(300)).ends_with('\u{fffd}'));
        profile["threads"][0]["frameTable"]["length"] = Value::from(MAX_PROFILE_NODES + 1);
        assert!(
            parse_samply_profile(
                &serde_json::to_vec(&profile).expect("JSON"),
                SYMBOLS,
                ProfileHost::MacOs,
                &AtomicBool::new(false),
                |_, _| None
            )
            .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_report_helper_fingerprint_bounds_changes_cancel_and_error_privacy() {
        let cancellation = AtomicBool::new(false);
        for (bytes, expected) in [
            (
                b"".as_slice(),
                [
                    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99,
                    0x6f, 0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95,
                    0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
                ],
            ),
            (
                b"abc".as_slice(),
                [
                    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d,
                    0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10,
                    0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
                ],
            ),
        ] {
            assert_eq!(
                fingerprint_task_helper_reader(
                    std::io::Cursor::new(bytes),
                    bytes.len() as u64,
                    bytes.len() as u64,
                    &cancellation,
                )
                .expect("known helper SHA-256"),
                expected,
            );
        }
        for length in [0, 3, 55, 56, 63, 64, 65, 65535, 65537, 1024 * 1024] {
            let bytes: Vec<_> = (0..length).map(|index| (index % 251) as u8).collect();
            assert_eq!(
                fingerprint_task_helper_reader(
                    std::io::Cursor::new(&bytes),
                    bytes.len() as u64,
                    bytes.len() as u64,
                    &cancellation,
                )
                .expect("bounded helper SHA-256"),
                profile_fingerprint(&bytes, &cancellation).expect("independent SHA-256 parity"),
                "helper SHA-256 parity at {length} bytes",
            );
        }
        let bytes = vec![b'x'; 65536];
        assert_eq!(
            fingerprint_task_helper_reader(
                std::io::Cursor::new(&bytes),
                bytes.len() as u64,
                bytes.len() as u64,
                &cancellation
            )
            .expect("exact helper limit"),
            profile_fingerprint(&bytes, &cancellation).expect("same digest")
        );
        assert!(
            fingerprint_task_helper_reader(
                std::io::Cursor::new(vec![b'x'; 65537]),
                65536,
                65536,
                &cancellation
            )
            .is_err()
        );
        assert!(
            fingerprint_task_helper_reader(std::io::Cursor::new(b"short"), 6, 6, &cancellation)
                .is_err()
        );
        cancellation.store(true, Ordering::Relaxed);
        assert!(
            fingerprint_task_helper_reader(std::io::Cursor::new(b"x"), 1, 1, &cancellation)
                .is_err()
        );
        struct PrivateError;
        impl Read for PrivateError {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("private path and secret arguments"))
            }
        }
        let error = fingerprint_task_helper_reader(PrivateError, 1, 1, &AtomicBool::new(false))
            .expect_err("private raw error denied");
        assert!(!error.to_string().contains("secret"));
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_helper_real_identity_rejects_replacement_alias_and_nonregular(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::{PermissionsExt as _, symlink};
        cx.executor().allow_parking();
        let fixture = tempfile::TempDir::new().expect("helper identity fixture");
        let helper = fixture.path().join("helper");
        std::fs::write(&helper, b"sealed helper bytes").expect("helper bytes");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700))
            .expect("executable helper");
        let filesystem = fs::RealFs::new(None, cx.background_executor.clone());
        let cancel = AtomicBool::new(false);
        let identity = prepare_task_helper_identity(&filesystem, helper.clone(), &cancel)
            .await
            .expect("ordinary executable identity");
        assert_eq!(
            identity.fingerprint(),
            profile_fingerprint(b"sealed helper bytes", &cancel).expect("byte digest")
        );
        revalidate_task_helper_identity(&filesystem, &identity, &cancel)
            .await
            .expect("unchanged helper identity");
        std::fs::write(&helper, b"changed helper data").expect("same-length changed helper");
        assert!(
            revalidate_task_helper_identity(&filesystem, &identity, &cancel)
                .await
                .is_err()
        );
        let changed = prepare_task_helper_identity(&filesystem, helper.clone(), &cancel)
            .await
            .expect("new configured helper data");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o600))
            .expect("remove execute permission");
        assert!(
            revalidate_task_helper_identity(&filesystem, &changed, &cancel)
                .await
                .is_err()
        );
        assert!(
            prepare_task_helper_identity(&filesystem, helper.clone(), &cancel)
                .await
                .is_err()
        );
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700))
            .expect("restore execute permission");
        let other = fixture.path().join("other");
        std::fs::write(&other, b"other helper").expect("second helper");
        std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o700))
            .expect("second executable");
        let alias = fixture.path().join("configured-helper");
        symlink(&helper, &alias).expect("configured path alias");
        let aliased = prepare_task_helper_identity(&filesystem, alias.clone(), &cancel)
            .await
            .expect("canonical configured alias");
        std::fs::remove_file(&alias).expect("replace configured alias");
        symlink(&other, &alias).expect("alias now chooses different helper");
        assert!(
            revalidate_task_helper_identity(&filesystem, &aliased, &cancel)
                .await
                .is_err()
        );
        assert!(
            prepare_task_helper_identity(&filesystem, fixture.path().to_path_buf(), &cancel)
                .await
                .is_err()
        );
        let oversized = fixture.path().join("oversized");
        let file = std::fs::File::create(&oversized).expect("sparse oversized fixture");
        file.set_len(MAX_TASK_HELPER_BYTES + 1)
            .expect("sparse advertised length");
        std::fs::set_permissions(&oversized, std::fs::Permissions::from_mode(0o700))
            .expect("executable oversized fixture");
        assert!(
            prepare_task_helper_identity(&filesystem, oversized, &cancel)
                .await
                .is_err(),
            "oversized identity is rejected before hashing"
        );
        cancel.store(true, Ordering::Relaxed);
        assert!(
            prepare_task_helper_identity(&filesystem, helper, &cancel)
                .await
                .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_helper_retains_initial_file_across_path_replacement(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::{FileExt as _, MetadataExt as _, PermissionsExt as _};
        cx.executor().allow_parking();
        let fixture = tempfile::TempDir::new().expect("retained helper fixture");
        let helper = fixture.path().join("helper");
        let original_bytes = b"retained original helper";
        std::fs::write(&helper, original_bytes).expect("original helper bytes");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700))
            .expect("executable original helper");
        let filesystem = fs::RealFs::new(None, cx.background_executor.clone());
        let cancellation = AtomicBool::new(false);
        let identity = prepare_task_helper_identity(&filesystem, helper.clone(), &cancellation)
            .await
            .expect("retained helper identity");
        let retained = identity.retained_file.as_ref().expect("real retained file");
        let original_inode = retained.file.metadata().expect("held metadata").ino();
        use std::io::Seek as _;
        let mut retained_reader = retained.file.as_ref();
        assert_eq!(
            retained_reader
                .stream_position()
                .expect("initial retained reader offset"),
            original_bytes.len() as u64,
            "the bounded initial fingerprint consumed this retained descriptor",
        );
        assert_eq!(
            identity.fingerprint(),
            profile_fingerprint(original_bytes, &cancellation)
                .expect("same-handle initial fingerprint")
        );
        assert_eq!(
            retained.file.metadata().expect("held metadata").len(),
            original_bytes.len() as u64
        );
        let copied_identity = identity.clone();
        assert!(std::sync::Arc::ptr_eq(
            &retained.file,
            &copied_identity
                .retained_file
                .as_ref()
                .expect("cloned retained file")
                .file
        ));
        revalidate_task_helper_identity(&filesystem, &identity, &cancellation)
            .await
            .expect("unchanged held and path identity");
        let replacement = fixture.path().join("replacement");
        std::fs::write(&replacement, b"different helper bytes").expect("replacement bytes");
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o700))
            .expect("replacement executable");
        std::fs::rename(&replacement, &helper).expect("replace helper pathname");
        assert_ne!(
            std::fs::metadata(&helper)
                .expect("replacement metadata")
                .ino(),
            original_inode
        );
        assert_eq!(
            retained
                .file
                .metadata()
                .expect("original descriptor remains")
                .ino(),
            original_inode
        );
        let mut observed = vec![0; original_bytes.len()];
        retained
            .file
            .read_exact_at(&mut observed, 0)
            .expect("read retained original descriptor");
        assert_eq!(observed, original_bytes);
        assert!(
            retained.revalidate().is_err(),
            "unlinked original metadata expires"
        );
        assert!(
            revalidate_task_helper_identity(&filesystem, &identity, &cancellation)
                .await
                .is_err()
        );
        let fresh = prepare_task_helper_identity(&filesystem, helper, &cancellation)
            .await
            .expect("fresh replacement identity");
        assert_ne!(fresh.fingerprint(), identity.fingerprint());
        revalidate_task_helper_identity(&filesystem, &fresh, &cancellation)
            .await
            .expect("fresh recovery");
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_helper_retained_file_rejects_metadata_only_change(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let fixture = tempfile::TempDir::new().expect("retained metadata fixture");
        let helper = fixture.path().join("helper");
        std::fs::write(&helper, b"unchanged helper bytes").expect("helper bytes");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700))
            .expect("original permissions");
        let filesystem = fs::RealFs::new(None, cx.background_executor.clone());
        let cancellation = AtomicBool::new(false);
        let identity = prepare_task_helper_identity(&filesystem, helper.clone(), &cancellation)
            .await
            .expect("held helper identity");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o500))
            .expect("different still-executable permissions");
        assert_eq!(
            profile_fingerprint(
                &std::fs::read(&helper).expect("unchanged bytes"),
                &cancellation
            )
            .expect("same bytes digest"),
            identity.fingerprint()
        );
        assert!(
            identity
                .retained_file
                .as_ref()
                .expect("retained file")
                .revalidate()
                .is_err()
        );
        assert!(
            revalidate_task_helper_identity(&filesystem, &identity, &cancellation)
                .await
                .is_err()
        );
    }
}
