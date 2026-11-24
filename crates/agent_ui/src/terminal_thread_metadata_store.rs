use std::path::{Path, PathBuf};

use agent_client_protocol::schema::v1 as acp;
use anyhow::Context as _;
use chrono::{DateTime, Utc};
use collections::{HashMap, HashSet};
use db::{
    sqlez::{
        bindable::Column, domain::Domain, statement::Statement,
        thread_safe_connection::ThreadSafeConnection,
    },
    sqlez_macros::sql,
};
use futures::{
    FutureExt,
    channel::oneshot,
    future::{BoxFuture, Shared},
};
use gpui::{AppContext as _, Entity, Global, Task};
use remote::{RemoteConnectionOptions, same_remote_connection_identity};
use sha2::{Digest as _, Sha256};
use ui::{App, Context, SharedString};
use util::ResultExt as _;
use workspace::PathList;

use crate::{
    TerminalId,
    thread_metadata_store::{ThreadId, WorktreePaths},
};

pub fn init(cx: &mut App) {
    TerminalThreadMetadataStore::init_global(cx);
}

struct GlobalTerminalThreadMetadataStore(Entity<TerminalThreadMetadataStore>);
impl Global for GlobalTerminalThreadMetadataStore {}

#[cfg(any(test, feature = "test-support"))]
pub struct TestTerminalMetadataDbName(pub String);
#[cfg(any(test, feature = "test-support"))]
impl Global for TestTerminalMetadataDbName {}

#[cfg(any(test, feature = "test-support"))]
impl TestTerminalMetadataDbName {
    pub fn global(cx: &App) -> String {
        cx.try_global::<Self>()
            .map(|global| global.0.clone())
            .unwrap_or_else(|| {
                let thread = std::thread::current();
                let test_name = thread.name().unwrap_or("unknown_test");
                format!("TERMINAL_THREAD_METADATA_DB_{}", test_name)
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InlineTerminalDraftPosition {
    byte_offset: usize,
    source_sha256: [u8; 32],
}

impl InlineTerminalDraftPosition {
    pub fn new(source: &str, byte_offset: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            source.is_char_boundary(byte_offset),
            "The inline terminal position is not a valid source character boundary"
        );
        Ok(Self {
            byte_offset,
            source_sha256: Sha256::digest(source.as_bytes()).into(),
        })
    }

    pub fn byte_offset_for_source(&self, source: &str) -> anyhow::Result<usize> {
        let source_sha256: [u8; 32] = Sha256::digest(source.as_bytes()).into();
        anyhow::ensure!(
            self.source_sha256 == source_sha256,
            "The inline terminal position belongs to a different draft"
        );
        anyhow::ensure!(
            source.is_char_boundary(self.byte_offset),
            "The inline terminal position is not a valid source character boundary"
        );
        Ok(self.byte_offset)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InlineTerminalContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurrence_id: Option<uuid::Uuid>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_occurrences: Vec<InlineTerminalContext>,
    pub thread_id: ThreadId,
    pub session_id: Option<acp::SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored_draft_source_session_id: Option<acp::SessionId>,
    pub preceding_entry_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_client_id: Option<acp_thread::ClientUserMessageId>,
    #[serde(default)]
    pub is_draft: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_order: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_position: Option<InlineTerminalDraftPosition>,
    pub collapsed: bool,
    pub output: String,
    #[serde(default)]
    pub snapshot: Option<terminal::TerminalSnapshot>,
}

struct InlineTerminalEncodedSizeBudget {
    remaining: usize,
}
impl std::io::Write for InlineTerminalEncodedSizeBudget {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "terminal history byte budget exceeded",
            ));
        }
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl InlineTerminalContext {
    pub(crate) fn validate_occurrences(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.previous_occurrences.len() < 256,
            "Terminal occurrence history is full; open an independent terminal instead"
        );
        let mut keys = HashSet::from_iter([self.occurrence_id]);
        for previous in &self.previous_occurrences {
            anyhow::ensure!(
                previous.previous_occurrences.is_empty(),
                "Nested terminal occurrence history is invalid"
            );
            anyhow::ensure!(
                keys.insert(previous.occurrence_id),
                "Duplicate terminal occurrence history is invalid"
            );
            anyhow::ensure!(
                previous.thread_id == self.thread_id,
                "Terminal occurrence belongs to another conversation"
            );
            anyhow::ensure!(
                !previous.is_draft || previous.message_client_id.is_none(),
                "A draft occurrence cannot also belong to a submitted message"
            );
            previous
                .snapshot
                .as_ref()
                .context("A frozen terminal occurrence has no styled snapshot")?
                .validate()
                .context("Invalid terminal history snapshot")?;
        }
        if let Some(snapshot) = &self.snapshot {
            snapshot
                .validate()
                .context("Invalid terminal history snapshot")?;
        }
        // Count encoding without allocating an over-budget aggregate or walking nested history.
        serde_json::to_writer(
            InlineTerminalEncodedSizeBudget {
                remaining: 64 * 1024 * 1024,
            },
            self,
        )
        .context("Terminal occurrence history exceeds its bounded storage budget")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalThreadMetadata {
    pub terminal_id: TerminalId,
    pub title: SharedString,
    pub custom_title: Option<SharedString>,
    pub created_at: DateTime<Utc>,
    pub worktree_paths: WorktreePaths,
    pub remote_connection: Option<RemoteConnectionOptions>,
    pub working_directory: Option<PathBuf>,
    pub inline_context: Option<InlineTerminalContext>,
}

impl TerminalThreadMetadata {
    pub fn folder_paths(&self) -> &PathList {
        self.worktree_paths.folder_path_list()
    }

    pub fn main_worktree_paths(&self) -> &PathList {
        self.worktree_paths.main_worktree_path_list()
    }

    pub fn display_title(&self) -> SharedString {
        compose_terminal_thread_title(
            self.title.as_ref(),
            self.custom_title.as_ref().map(|title| title.as_ref()),
        )
    }
}

pub(crate) fn compose_terminal_thread_title(
    terminal_title: &str,
    custom_title: Option<&str>,
) -> SharedString {
    let Some(custom_title) = custom_title.filter(|title| !title.trim().is_empty()) else {
        return SharedString::from(terminal_title.to_string());
    };

    if let Some(prefix) = terminal_title_prefix(terminal_title) {
        SharedString::from(format!("{prefix}{custom_title}"))
    } else {
        SharedString::from(custom_title.to_string())
    }
}

pub(crate) fn terminal_title_without_prefix(title: &str) -> &str {
    terminal_title_prefix(title)
        .map(|prefix| &title[prefix.len()..])
        .unwrap_or(title)
}

pub fn terminal_title_prefix(title: &str) -> Option<&str> {
    let mut prefix_byte_len = 0;
    let mut saw_prefix_character = false;
    let mut saw_whitespace_after_prefix = false;

    let mut chars = title.chars().peekable();
    while let Some(character) = chars.next() {
        if character.is_alphanumeric() {
            return None;
        }

        if character.is_whitespace() {
            if !saw_prefix_character {
                return None;
            }

            prefix_byte_len += character.len_utf8();
            saw_whitespace_after_prefix = true;

            while let Some(character) = chars.peek() {
                if !character.is_whitespace() {
                    break;
                }

                prefix_byte_len += character.len_utf8();
                chars.next();
            }

            break;
        }

        saw_prefix_character = true;
        prefix_byte_len += character.len_utf8();
    }

    if saw_whitespace_after_prefix {
        Some(&title[..prefix_byte_len])
    } else {
        None
    }
}

pub struct TerminalThreadMetadataStore {
    db: TerminalThreadMetadataDb,
    terminals: HashMap<TerminalId, TerminalThreadMetadata>,
    terminals_by_paths: HashMap<PathList, HashSet<TerminalId>>,
    terminals_by_main_paths: HashMap<PathList, HashSet<TerminalId>>,
    reload_task: Option<Shared<Task<()>>>,
    load_error: Option<SharedString>,
    pending_terminal_ops_tx: async_channel::Sender<TerminalDbRequest>,
    pending_write_revisions: HashMap<TerminalId, u64>,
    pending_delete_metadata: HashMap<TerminalId, TerminalThreadMetadata>,
    write_errors: HashMap<TerminalId, TerminalHistoryWriteError>,
    next_write_revision: u64,
    _db_operations_task: Task<()>,
    _db_completion_task: Task<()>,
}

struct TerminalHistoryWriteError {
    revision: Option<u64>,
    message: SharedString,
    retry_operation: Option<DbOperation>,
}

#[derive(Debug, PartialEq)]
struct QueuedDbOperation {
    revision: u64,
    operation: DbOperation,
}

enum TerminalDbRequest {
    Operation(QueuedDbOperation),
    Flush(oneshot::Sender<anyhow::Result<()>>),
}

struct TerminalDbWriteResult {
    terminal_id: TerminalId,
    revision: u64,
    retry_operation: DbOperation,
    error: Option<SharedString>,
}

#[derive(Debug, Clone, PartialEq)]
enum DbOperation {
    Upsert(TerminalThreadMetadata),
    Delete(TerminalId),
}

impl DbOperation {
    fn id(&self) -> TerminalId {
        match self {
            DbOperation::Upsert(metadata) => metadata.terminal_id,
            DbOperation::Delete(terminal_id) => *terminal_id,
        }
    }
}

impl TerminalThreadMetadataStore {
    #[cfg(not(any(test, feature = "test-support")))]
    pub fn init_global(cx: &mut App) {
        if cx.has_global::<GlobalTerminalThreadMetadataStore>() {
            return;
        }

        let db = TerminalThreadMetadataDb::global(cx);
        let terminal_store = cx.new(|cx| Self::new(db, cx));
        cx.set_global(GlobalTerminalThreadMetadataStore(terminal_store));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn init_global(cx: &mut App) {
        let db_name = TestTerminalMetadataDbName::global(cx);
        let db = gpui::block_on(db::open_test_db::<TerminalThreadMetadataDb>(&db_name));
        let terminal_store = cx.new(|cx| Self::new(TerminalThreadMetadataDb(db), cx));
        cx.set_global(GlobalTerminalThreadMetadataStore(terminal_store));
    }

    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalTerminalThreadMetadataStore>()
            .map(|store| store.0.clone())
    }

    pub fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalTerminalThreadMetadataStore>().0.clone()
    }

    pub fn entry(&self, terminal_id: TerminalId) -> Option<&TerminalThreadMetadata> {
        self.terminals.get(&terminal_id)
    }

    pub fn entries(&self) -> impl Iterator<Item = &TerminalThreadMetadata> + '_ {
        self.terminals.values()
    }

    pub fn entries_for_conversation<'a>(
        &'a self,
        thread_id: ThreadId,
        session_id: Option<&'a acp::SessionId>,
        remote_connection: Option<&'a RemoteConnectionOptions>,
    ) -> impl Iterator<Item = &'a TerminalThreadMetadata> + 'a {
        self.terminals.values().filter(move |terminal| {
            terminal.inline_context.as_ref().is_some_and(|context| {
                context.thread_id == thread_id && context.session_id.as_ref() == session_id
            }) && same_remote_connection_identity(
                terminal.remote_connection.as_ref(),
                remote_connection,
            )
        })
    }

    pub fn load_error(&self) -> Option<&SharedString> {
        self.load_error.as_ref()
    }

    pub fn write_error(&self, terminal_id: TerminalId) -> Option<&SharedString> {
        self.write_errors
            .get(&terminal_id)
            .map(|error| &error.message)
    }

    pub fn write_pending(&self, terminal_id: TerminalId) -> bool {
        self.pending_write_revisions.contains_key(&terminal_id)
    }

    pub fn flush(&self, cx: &mut Context<Self>) -> Task<anyhow::Result<()>> {
        if self.pending_write_revisions.is_empty() {
            return Task::ready(self.write_completion_result());
        }

        let (sender, receiver) = oneshot::channel();
        let mut sender = Some(sender);
        let subscription = cx.observe_self(move |store, _| {
            if store.pending_write_revisions.is_empty()
                && let Some(sender) = sender.take()
                && sender.send(store.write_completion_result()).is_err()
            {
                log::warn!("Terminal history flush receiver was dropped before completion");
            }
        });
        let owner = cx.entity();
        cx.spawn(async move |_, _| {
            let _subscription = subscription;
            let _owner = owner;
            receiver
                .await
                .context("Terminal history flush was interrupted")?
        })
    }

    pub fn flush_before_quit(&self) -> BoxFuture<'static, anyhow::Result<()>> {
        let error = self.write_errors.iter().find_map(|(terminal_id, error)| {
            let newer_write_pending =
                self.pending_write_revisions
                    .get(terminal_id)
                    .is_some_and(|revision| {
                        error
                            .revision
                            .is_some_and(|failed_revision| failed_revision < *revision)
                    });
            // Valid replacement data can repair a rejected snapshot even when
            // that rejection has no replayable operation. The fence still
            // reports an actual failure of the newer accepted write.
            (!newer_write_pending).then(|| error.message.clone())
        });
        let (sender, receiver) = oneshot::channel();
        let queued = self
            .pending_terminal_ops_tx
            .try_send(TerminalDbRequest::Flush(sender));
        async move {
            queued.context("Terminal history write queue is unavailable")?;
            let result = receiver
                .await
                .context("Terminal history durable fence was interrupted")?;
            result?;
            if let Some(error) = error {
                anyhow::bail!("Terminal history was not persisted: {error}");
            }
            Ok(())
        }
        .boxed()
    }

    fn write_completion_result(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.pending_terminal_ops_tx.is_closed(),
            "Terminal history write queue is unavailable"
        );
        if let Some(error) = self.write_errors.values().next() {
            anyhow::bail!("Terminal history was not persisted: {}", error.message);
        }
        Ok(())
    }

    pub fn retry_write(
        &mut self,
        terminal_id: TerminalId,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        if self.write_pending(terminal_id) {
            anyhow::bail!("A terminal history write is already pending");
        }
        let operation = self
            .write_errors
            .get(&terminal_id)
            .and_then(|error| error.retry_operation.clone())
            .context("This terminal history error requires updated valid data before retrying")?;
        match operation {
            DbOperation::Upsert(metadata) => self.save(metadata, cx),
            DbOperation::Delete(terminal_id) => self.delete(terminal_id, cx),
        }
        if !self.write_pending(terminal_id) {
            anyhow::bail!(
                "{}",
                self.write_error(terminal_id)
                    .map_or("Terminal history retry could not be queued", |error| error
                        .as_ref())
            );
        }
        Ok(())
    }

    pub fn reload_task(&self) -> Shared<Task<()>> {
        self.reload_task
            .clone()
            .unwrap_or_else(|| Task::ready(()).shared())
    }

    pub fn entries_for_path<'a>(
        &'a self,
        path_list: &PathList,
        remote_connection: Option<&'a RemoteConnectionOptions>,
    ) -> impl Iterator<Item = &'a TerminalThreadMetadata> + 'a {
        self.terminals_by_paths
            .get(path_list)
            .into_iter()
            .flatten()
            .filter_map(|id| self.terminals.get(id))
            .filter(move |terminal| {
                same_remote_connection_identity(
                    terminal.remote_connection.as_ref(),
                    remote_connection,
                )
            })
    }

    pub fn entries_for_main_worktree_path<'a>(
        &'a self,
        path_list: &PathList,
        remote_connection: Option<&'a RemoteConnectionOptions>,
    ) -> impl Iterator<Item = &'a TerminalThreadMetadata> + 'a {
        self.terminals_by_main_paths
            .get(path_list)
            .into_iter()
            .flatten()
            .filter_map(|id| self.terminals.get(id))
            .filter(move |terminal| {
                same_remote_connection_identity(
                    terminal.remote_connection.as_ref(),
                    remote_connection,
                )
            })
    }

    pub fn path_is_referenced_by_terminal(
        &self,
        terminal_id: Option<TerminalId>,
        path: &Path,
        remote_connection: Option<&RemoteConnectionOptions>,
    ) -> bool {
        self.entries().any(|terminal| {
            Some(terminal.terminal_id) != terminal_id
                && same_remote_connection_identity(
                    terminal.remote_connection.as_ref(),
                    remote_connection,
                )
                && terminal
                    .folder_paths()
                    .paths()
                    .iter()
                    .any(|folder_path| folder_path.as_path() == path)
        })
    }

    pub fn save(&mut self, metadata: TerminalThreadMetadata, cx: &mut Context<Self>) {
        self.save_internal(metadata);
        cx.notify();
    }

    pub fn change_worktree_paths(
        &mut self,
        current_folder_paths: &PathList,
        remote_connection: Option<&RemoteConnectionOptions>,
        mutate: impl Fn(&mut WorktreePaths),
        cx: &mut Context<Self>,
    ) {
        let terminal_ids: Vec<_> = self
            .terminals_by_paths
            .get(current_folder_paths)
            .into_iter()
            .flatten()
            .filter(|id| {
                self.terminals.get(id).is_some_and(|terminal| {
                    same_remote_connection_identity(
                        terminal.remote_connection.as_ref(),
                        remote_connection,
                    )
                })
            })
            .copied()
            .collect();

        if terminal_ids.is_empty() {
            return;
        }

        for terminal_id in terminal_ids {
            if let Some(mut terminal) = self.terminals.get(&terminal_id).cloned() {
                mutate(&mut terminal.worktree_paths);
                self.save_internal(terminal);
            }
        }

        cx.notify();
    }

    fn save_internal(&mut self, mut metadata: TerminalThreadMetadata) {
        if let Some(context) = metadata.inline_context.as_ref()
            && let Err(error) = context.validate_occurrences()
        {
            log::error!("Invalid terminal occurrence history: {error:#}");
            self.record_rejected_write(
                metadata.terminal_id,
                format!("Invalid terminal occurrence history: {error:#}")
                    .chars()
                    .take(512)
                    .collect::<String>()
                    .into(),
            );
            return;
        }
        if let Some(snapshot) = metadata
            .inline_context
            .as_ref()
            .and_then(|context| context.snapshot.as_ref())
            && let Err(error) = snapshot.validate()
        {
            log::error!("Invalid terminal history snapshot: {error:#}");
            self.record_rejected_write(
                metadata.terminal_id,
                format!("Invalid terminal history snapshot: {error:#}")
                    .chars()
                    .take(512)
                    .collect::<String>()
                    .into(),
            );
            return;
        }
        if let Some(existing) = self
            .terminals
            .get(&metadata.terminal_id)
            .or_else(|| self.pending_delete_metadata.get(&metadata.terminal_id))
        {
            // Standalone hosts refresh shell metadata without owning the
            // conversation association or retained history.
            if metadata.inline_context.is_none() {
                metadata.inline_context = existing.inline_context.clone();
            }
            if existing.folder_paths() != metadata.folder_paths()
                && let Some(ids) = self.terminals_by_paths.get_mut(existing.folder_paths())
            {
                ids.remove(&metadata.terminal_id);
            }

            if existing.main_worktree_paths() != metadata.main_worktree_paths()
                && let Some(ids) = self
                    .terminals_by_main_paths
                    .get_mut(existing.main_worktree_paths())
            {
                ids.remove(&metadata.terminal_id);
            }
        }

        self.cache_terminal_metadata(metadata.clone());
        self.pending_delete_metadata.remove(&metadata.terminal_id);
        self.enqueue_db_operation(DbOperation::Upsert(metadata));
    }

    fn cache_terminal_metadata(&mut self, metadata: TerminalThreadMetadata) {
        self.terminals
            .insert(metadata.terminal_id, metadata.clone());

        self.terminals_by_paths
            .entry(metadata.folder_paths().clone())
            .or_default()
            .insert(metadata.terminal_id);

        if !metadata.main_worktree_paths().is_empty() {
            self.terminals_by_main_paths
                .entry(metadata.main_worktree_paths().clone())
                .or_default()
                .insert(metadata.terminal_id);
        }
    }

    pub fn delete(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        if let Some(terminal) = self.terminals.remove(&terminal_id) {
            self.pending_delete_metadata
                .insert(terminal_id, terminal.clone());
            if let Some(ids) = self.terminals_by_paths.get_mut(terminal.folder_paths()) {
                ids.remove(&terminal_id);
            }
            if !terminal.main_worktree_paths().is_empty()
                && let Some(ids) = self
                    .terminals_by_main_paths
                    .get_mut(terminal.main_worktree_paths())
            {
                ids.remove(&terminal_id);
            }
        }
        self.enqueue_db_operation(DbOperation::Delete(terminal_id));
        cx.notify();
    }

    fn new(db: TerminalThreadMetadataDb, cx: &mut Context<Self>) -> Self {
        let (tx, rx) = async_channel::unbounded::<TerminalDbRequest>();
        let (completion_sender, completion_receiver) =
            async_channel::unbounded::<TerminalDbWriteResult>();
        // Quit holds the App borrow, so durable writes and their fence must not
        // depend on a foreground entity update making progress.
        let _db_operations_task = cx.background_spawn({
            let db = db.clone();
            async move {
                let mut write_errors = HashMap::<TerminalId, SharedString>::default();
                while let Ok(first_request) = rx.recv().await {
                    let mut updates = Vec::new();
                    let mut next = first_request;
                    let flush = loop {
                        match next {
                            TerminalDbRequest::Operation(operation) => updates.push(operation),
                            TerminalDbRequest::Flush(sender) => break Some(sender),
                        }
                        match rx.try_recv() {
                            Ok(request) => next = request,
                            Err(_) => break None,
                        }
                    };
                    for queued in Self::dedup_db_operations(updates) {
                        let terminal_id = queued.operation.id();
                        let revision = queued.revision;
                        let retry_operation = queued.operation.clone();
                        let result = match queued.operation {
                            DbOperation::Upsert(metadata) => db.save(metadata).await,
                            DbOperation::Delete(terminal_id) => db.delete(terminal_id).await,
                        };
                        let error: Option<SharedString> = result.err().map(|error| {
                            log::error!("Failed to persist terminal history: {error:#}");
                            format!("{error:#}")
                                .chars()
                                .take(512)
                                .collect::<String>()
                                .into()
                        });
                        if let Some(error) = &error {
                            write_errors.insert(terminal_id, error.clone());
                        } else {
                            write_errors.remove(&terminal_id);
                        }
                        completion_sender
                            .try_send(TerminalDbWriteResult {
                                terminal_id,
                                revision,
                                retry_operation,
                                error,
                            })
                            .log_err();
                    }
                    if let Some(sender) = flush {
                        let result = match write_errors.values().next() {
                            Some(error) => Err(anyhow::anyhow!(
                                "Terminal history was not persisted: {error}"
                            )),
                            None => Ok(()),
                        };
                        if sender.send(result).is_err() {
                            log::warn!("Terminal history durable fence receiver was dropped");
                        }
                    }
                }
            }
        });
        let _db_completion_task = cx.spawn(async move |this, cx| {
            while let Ok(completion) = completion_receiver.recv().await {
                let TerminalDbWriteResult {
                    terminal_id,
                    revision,
                    retry_operation,
                    error,
                } = completion;
                this.update(cx, |this, cx| {
                    // An older completion must not settle a newer queued write.
                    if this.pending_write_revisions.get(&terminal_id) != Some(&revision) {
                        return;
                    }
                    this.pending_write_revisions.remove(&terminal_id);
                    let can_settle_error =
                        this.write_errors.get(&terminal_id).is_none_or(|error| {
                            error
                                .revision
                                .is_some_and(|failed_revision| failed_revision <= revision)
                        });
                    if let Some(error) = error {
                        if can_settle_error {
                            this.write_errors.insert(
                                terminal_id,
                                TerminalHistoryWriteError {
                                    revision: Some(revision),
                                    message: error,
                                    retry_operation: Some(retry_operation),
                                },
                            );
                        }
                        if let Some(metadata) = this.pending_delete_metadata.remove(&terminal_id) {
                            this.cache_terminal_metadata(metadata);
                        }
                    } else {
                        if can_settle_error {
                            this.write_errors.remove(&terminal_id);
                        }
                        this.pending_delete_metadata.remove(&terminal_id);
                    }
                    cx.notify();
                    if this.pending_write_revisions.is_empty() {
                        this.reload(cx);
                    }
                })
                .log_err();
            }
        });
        cx.on_app_quit(|store, _| {
            let flush = store.flush_before_quit();
            async move {
                if let Err(error) = flush.await {
                    log::error!("Failed to flush terminal history before quit: {error:#}");
                }
            }
        })
        .detach();

        let mut this = Self {
            db,
            terminals: HashMap::default(),
            terminals_by_paths: HashMap::default(),
            terminals_by_main_paths: HashMap::default(),
            reload_task: None,
            load_error: None,
            pending_terminal_ops_tx: tx,
            pending_write_revisions: HashMap::default(),
            pending_delete_metadata: HashMap::default(),
            write_errors: HashMap::default(),
            next_write_revision: 0,
            _db_operations_task,
            _db_completion_task,
        };
        this.reload(cx);
        this
    }

    fn enqueue_db_operation(&mut self, operation: DbOperation) {
        let terminal_id = operation.id();
        let Some(revision) = self.next_write_revision.checked_add(1) else {
            self.record_queue_error(
                terminal_id,
                None,
                "Terminal history write revisions are exhausted".into(),
                Some(operation),
            );
            return;
        };
        self.next_write_revision = revision;
        self.pending_write_revisions.insert(terminal_id, revision);
        if let Err(error) = self
            .pending_terminal_ops_tx
            .try_send(TerminalDbRequest::Operation(QueuedDbOperation {
                revision,
                operation,
            }))
        {
            log::error!("Failed to queue terminal history write: {error}");
            let retry_operation = match error.into_inner() {
                TerminalDbRequest::Operation(queued) => Some(queued.operation),
                TerminalDbRequest::Flush(_) => None,
            };
            self.record_queue_error(
                terminal_id,
                Some(revision),
                "Terminal history write queue is unavailable".into(),
                retry_operation,
            );
        }
    }

    fn record_rejected_write(&mut self, terminal_id: TerminalId, message: SharedString) {
        let revision = self.next_write_revision.checked_add(1);
        if let Some(revision) = revision {
            self.next_write_revision = revision;
        }
        // Rejecting invalid data must not cancel an already accepted durable operation.
        self.write_errors.insert(
            terminal_id,
            TerminalHistoryWriteError {
                revision,
                message,
                retry_operation: None,
            },
        );
    }

    fn record_queue_error(
        &mut self,
        terminal_id: TerminalId,
        revision: Option<u64>,
        message: SharedString,
        retry_operation: Option<DbOperation>,
    ) {
        self.pending_write_revisions.remove(&terminal_id);
        self.write_errors.insert(
            terminal_id,
            TerminalHistoryWriteError {
                revision,
                message,
                retry_operation,
            },
        );
        if let Some(metadata) = self.pending_delete_metadata.remove(&terminal_id) {
            self.cache_terminal_metadata(metadata);
        }
    }

    fn dedup_db_operations(operations: Vec<QueuedDbOperation>) -> Vec<QueuedDbOperation> {
        let mut ops = HashMap::default();
        for operation in operations.into_iter().rev() {
            if ops.contains_key(&operation.operation.id()) {
                continue;
            }
            ops.insert(operation.operation.id(), operation);
        }
        ops.into_values().collect()
    }

    fn reload(&mut self, cx: &mut Context<Self>) {
        let db = self.db.clone();
        let write_revision = self.next_write_revision;
        let writes_pending = !self.pending_write_revisions.is_empty();
        self.reload_task = Some(
            cx.spawn(async move |this, cx| {
                let rows = cx
                    .background_spawn(async move {
                        db.list()
                            .context("Failed to fetch terminal thread metadata")
                    })
                    .await;

                this.update(cx, |this, cx| {
                    // A read racing a queued write can otherwise resurrect deleted rows.
                    if writes_pending || write_revision != this.next_write_revision {
                        if this.pending_write_revisions.is_empty() {
                            this.reload(cx);
                        }
                        return;
                    }
                    match rows {
                        Ok(rows) => {
                            let unsaved = this
                                .terminals
                                .values()
                                .filter(|metadata| {
                                    this.write_pending(metadata.terminal_id)
                                        || this.write_error(metadata.terminal_id).is_some()
                                })
                                .cloned()
                                .collect::<Vec<_>>();
                            let unsaved_ids = unsaved
                                .iter()
                                .map(|metadata| metadata.terminal_id)
                                .collect::<HashSet<_>>();
                            this.terminals.clear();
                            this.terminals_by_paths.clear();
                            this.terminals_by_main_paths.clear();
                            this.load_error = None;
                            for row in rows {
                                if !this.pending_delete_metadata.contains_key(&row.terminal_id)
                                    && !unsaved_ids.contains(&row.terminal_id)
                                {
                                    this.cache_terminal_metadata(row);
                                }
                            }
                            for row in unsaved {
                                this.cache_terminal_metadata(row);
                            }
                        }
                        Err(error) => {
                            // Failed reads must not turn retained history into
                            // an apparently successful empty conversation.
                            log::error!("{error:#}");
                            this.load_error = Some(format!("{error:#}").into());
                        }
                    }
                    cx.notify();
                })
                .log_err();
            })
            .shared(),
        );
    }
}

struct TerminalThreadMetadataDb(ThreadSafeConnection);

impl Domain for TerminalThreadMetadataDb {
    const NAME: &str = stringify!(TerminalThreadMetadataDb);

    const MIGRATIONS: &[&str] = &[
        sql!(
            CREATE TABLE IF NOT EXISTS sidebar_terminal_threads(
                terminal_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                custom_title TEXT,
                created_at TEXT NOT NULL,
                working_directory TEXT,
                folder_paths TEXT,
                folder_paths_order TEXT,
                main_worktree_paths TEXT,
                main_worktree_paths_order TEXT,
                remote_connection TEXT
            ) STRICT;
        ),
        sql!(
            ALTER TABLE sidebar_terminal_threads ADD COLUMN inline_context TEXT;
        ),
    ];
}

db::static_connection!(TerminalThreadMetadataDb, []);

impl TerminalThreadMetadataDb {
    pub fn list(&self) -> anyhow::Result<Vec<TerminalThreadMetadata>> {
        self.select::<TerminalThreadMetadata>(
            "SELECT terminal_id, title, custom_title, created_at, \
            working_directory, folder_paths, folder_paths_order, main_worktree_paths, \
            main_worktree_paths_order, remote_connection, inline_context \
            FROM sidebar_terminal_threads \
            ORDER BY created_at DESC",
        )?()
    }

    pub async fn save(&self, row: TerminalThreadMetadata) -> anyhow::Result<()> {
        let terminal_id = row.terminal_id.to_key_string();
        let title = row.title.to_string();
        let custom_title = row.custom_title.as_ref().map(ToString::to_string);
        let created_at = row.created_at.to_rfc3339();
        let working_directory = row
            .working_directory
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        let serialized = row.folder_paths().serialize();
        let (folder_paths, folder_paths_order) = if row.folder_paths().is_empty() {
            (None, None)
        } else {
            (Some(serialized.paths), Some(serialized.order))
        };
        let main_serialized = row.main_worktree_paths().serialize();
        let (main_worktree_paths, main_worktree_paths_order) =
            if row.main_worktree_paths().is_empty() {
                (None, None)
            } else {
                (Some(main_serialized.paths), Some(main_serialized.order))
            };
        let remote_connection = row
            .remote_connection
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .context("serialize terminal thread remote connection")?;

        let inline_context = row
            .inline_context
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .context("serialize inline terminal conversation context")?;

        self.write(move |conn| {
            let sql = "INSERT INTO sidebar_terminal_threads(terminal_id, title, custom_title, created_at, working_directory, folder_paths, folder_paths_order, main_worktree_paths, main_worktree_paths_order, remote_connection, inline_context) \
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) \
                       ON CONFLICT(terminal_id) DO UPDATE SET \
                           title = excluded.title, \
                           custom_title = excluded.custom_title, \
                           created_at = excluded.created_at, \
                           working_directory = excluded.working_directory, \
                           folder_paths = excluded.folder_paths, \
                           folder_paths_order = excluded.folder_paths_order, \
                           main_worktree_paths = excluded.main_worktree_paths, \
                           main_worktree_paths_order = excluded.main_worktree_paths_order, \
                           remote_connection = excluded.remote_connection, \
                           inline_context = COALESCE(excluded.inline_context, sidebar_terminal_threads.inline_context)";
            let mut stmt = Statement::prepare(conn, sql)?;
            let mut i = stmt.bind(&terminal_id, 1)?;
            i = stmt.bind(&title, i)?;
            i = stmt.bind(&custom_title, i)?;
            i = stmt.bind(&created_at, i)?;
            i = stmt.bind(&working_directory, i)?;
            i = stmt.bind(&folder_paths, i)?;
            i = stmt.bind(&folder_paths_order, i)?;
            i = stmt.bind(&main_worktree_paths, i)?;
            i = stmt.bind(&main_worktree_paths_order, i)?;
            i = stmt.bind(&remote_connection, i)?;
            stmt.bind(&inline_context, i)?;
            stmt.exec()
        })
        .await
    }

    pub async fn delete(&self, terminal_id: TerminalId) -> anyhow::Result<()> {
        let terminal_id = terminal_id.to_key_string();
        self.write(move |conn| {
            let mut stmt = Statement::prepare(
                conn,
                "DELETE FROM sidebar_terminal_threads WHERE terminal_id = ?",
            )?;
            stmt.bind(&terminal_id, 1)?;
            stmt.exec()
        })
        .await
    }
}

impl Column for TerminalThreadMetadata {
    fn column(statement: &mut Statement, start_index: i32) -> anyhow::Result<(Self, i32)> {
        let (terminal_id, next): (String, i32) = Column::column(statement, start_index)?;
        let (title, next): (String, i32) = Column::column(statement, next)?;
        let (custom_title, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (created_at, next): (String, i32) = Column::column(statement, next)?;
        let (working_directory, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (folder_paths_str, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (folder_paths_order_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (main_worktree_paths_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (main_worktree_paths_order_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (remote_connection_json, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (inline_context_json, next): (Option<String>, i32) = Column::column(statement, next)?;
        let inline_context = inline_context_json
            .as_deref()
            .map(|json| -> anyhow::Result<InlineTerminalContext> {
                anyhow::ensure!(
                    json.len() <= 64 * 1024 * 1024,
                    "Terminal occurrence history exceeds its bounded storage budget"
                );
                Ok(serde_json::from_str::<InlineTerminalContext>(json)?)
            })
            .transpose()
            .context("deserialize inline terminal conversation context")?;
        if let Some(snapshot) = inline_context
            .as_ref()
            .and_then(|context| context.snapshot.as_ref())
        {
            snapshot
                .validate()
                .context("validate inline terminal history snapshot")?;
        }

        let folder_paths = folder_paths_str
            .map(|paths| {
                PathList::deserialize(&util::path_list::SerializedPathList {
                    paths,
                    order: folder_paths_order_str.unwrap_or_default(),
                })
            })
            .unwrap_or_default();

        let main_worktree_paths = main_worktree_paths_str
            .map(|paths| {
                PathList::deserialize(&util::path_list::SerializedPathList {
                    paths,
                    order: main_worktree_paths_order_str.unwrap_or_default(),
                })
            })
            .unwrap_or_default();

        let remote_connection = remote_connection_json
            .as_deref()
            .map(serde_json::from_str::<RemoteConnectionOptions>)
            .transpose()
            .context("deserialize terminal thread remote connection")?;

        if let Some(context) = inline_context.as_ref() {
            context.validate_occurrences()?;
        }

        let worktree_paths = WorktreePaths::from_path_lists(main_worktree_paths, folder_paths)
            .unwrap_or_else(|_| WorktreePaths::default());

        Ok((
            TerminalThreadMetadata {
                terminal_id: TerminalId::from_key_string(&terminal_id)?,
                title: SharedString::from(title),
                custom_title: custom_title
                    .filter(|title| !title.trim().is_empty())
                    .map(SharedString::from),
                created_at: DateTime::parse_from_rfc3339(&created_at)?.with_timezone(&Utc),
                worktree_paths,
                remote_connection,
                working_directory: working_directory.map(PathBuf::from),
                inline_context,
            },
            next,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use std::path::Path;

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            TerminalThreadMetadataStore::init_global(cx);
        });
        cx.run_until_parked();
    }

    fn metadata(title: &str, worktree_paths: WorktreePaths) -> TerminalThreadMetadata {
        let now = Utc::now();
        TerminalThreadMetadata {
            terminal_id: TerminalId::new(),
            title: SharedString::from(title.to_string()),
            custom_title: None,
            created_at: now,
            worktree_paths,
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        }
    }

    #[test]
    fn test_inline_draft_position_preserves_unicode_boundaries() {
        let source = "Aλ\n🦀B";
        for offset in [0, 1, 3, 4, 8, 9] {
            let position = InlineTerminalDraftPosition::new(source, offset)
                .expect("valid Unicode source boundary");
            let json = serde_json::to_string(&position).expect("encode position");
            let restored: InlineTerminalDraftPosition =
                serde_json::from_str(&json).expect("restore position");
            assert_eq!(
                restored
                    .byte_offset_for_source(source)
                    .expect("same source"),
                offset
            );
        }
        let empty = InlineTerminalDraftPosition::new("", 0).expect("empty draft start");
        assert_eq!(
            empty.byte_offset_for_source("").expect("same empty draft"),
            0
        );
        for offset in [2, 5, 6, 7, 10, usize::MAX] {
            assert!(InlineTerminalDraftPosition::new(source, offset).is_err());
        }
        assert!(InlineTerminalDraftPosition::new("", 1).is_err());
    }

    #[test]
    fn test_inline_draft_position_rejects_changed_source_and_corruption() {
        let position = InlineTerminalDraftPosition::new("AB", 1).expect("valid position");
        for source in ["AC", "BA", "", "AB ", "λ"] {
            let error = position
                .byte_offset_for_source(source)
                .expect_err("changed draft");
            assert!(error.to_string().contains("different draft"));
        }
        let source = "Aλ\n🦀B";
        let position = InlineTerminalDraftPosition::new(source, 1).expect("valid position");
        for offset in [2, 10, usize::MAX] {
            let mut value = serde_json::to_value(&position).expect("encode position");
            value.as_object_mut().expect("position object").insert(
                "byte_offset".into(),
                serde_json::to_value(offset).expect("encode offset"),
            );
            let restored: InlineTerminalDraftPosition =
                serde_json::from_value(value).expect("typed but invalid offset");
            assert!(restored.byte_offset_for_source(source).is_err());
        }
        let mut value = serde_json::to_value(&position).expect("encode position");
        value.as_object_mut().expect("position object").insert(
            "source_sha256".into(),
            serde_json::to_value([0_u8; 31]).expect("encode truncated digest"),
        );
        assert!(serde_json::from_value::<InlineTerminalDraftPosition>(value).is_err());
    }

    #[gpui::test]
    async fn test_inline_draft_position_sqlite_reload_corruption_and_legacy(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let terminal = cx.new(|cx| {
            terminal::TerminalBuilder::new_display_only(
                terminal::terminal_settings::CursorShape::Block,
                settings::AlternateScroll::Off,
                Some(128),
                cx.entity_id().as_u64(),
                cx.background_executor(),
                util::paths::PathStyle::local(),
            )
            .subscribe(cx)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b[1;31mposition history\x1b[0m\r\n", cx);
        });
        cx.run_until_parked();
        assert!(
            terminal
                .read_with(cx, |terminal, _| terminal.get_content())
                .contains("position history")
        );
        let snapshot = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture styled canonical grid")
        });
        let source = "Aλ\n🦀B";
        let mut row = metadata("Inline draft position", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("position-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: Some(
                InlineTerminalDraftPosition::new(source, 4)
                    .expect("between actual Unicode prefix and suffix"),
            ),
            collapsed: true,
            output: String::new(),
            snapshot: Some(snapshot),
        });
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        database
            .save(row.clone())
            .await
            .expect("persist position and styled grid");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            let position = store
                .entry(row.terminal_id)
                .expect("retained row")
                .inline_context
                .as_ref()
                .expect("retained parent")
                .draft_position
                .as_ref()
                .expect("retained position");
            assert_eq!(
                position
                    .byte_offset_for_source(source)
                    .expect("exact draft"),
                4
            );
            assert!(position.byte_offset_for_source("Aλ\n🦀C").is_err());
        });

        let mut corrupt_context =
            serde_json::to_value(&row.inline_context).expect("encode trusted context");
        corrupt_context
            .get_mut("draft_position")
            .expect("position")
            .as_object_mut()
            .expect("position object")
            .insert(
                "source_sha256".into(),
                serde_json::to_value([0_u8; 31]).expect("encode truncated digest"),
            );
        let corrupt_json = serde_json::to_string(&corrupt_context).expect("encode corrupt digest");
        let terminal_id = row.terminal_id.to_key_string();
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "UPDATE sidebar_terminal_threads SET inline_context = ?1 WHERE terminal_id = ?2")?;
            statement.bind(&corrupt_json, 1)?;
            statement.bind(&terminal_id, 2)?;
            statement.exec()
        }).await.expect("write actual malformed position");
        let error = database
            .list()
            .expect_err("invalid position must fail canonical read");
        assert!(format!("{error:#}").contains("deserialize inline terminal conversation context"));
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_some_and(|error| {
                error.contains("deserialize inline terminal conversation context")
            }));
        });
        database
            .save(row.clone())
            .await
            .expect("repair original trusted position");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_none());
        });

        let mut legacy_context =
            serde_json::to_value(&row.inline_context).expect("encode actual positioned history");
        legacy_context
            .as_object_mut()
            .expect("context object")
            .remove("draft_position");
        let legacy_json = serde_json::to_string(&legacy_context).expect("encode legacy context");
        let terminal_id = row.terminal_id.to_key_string();
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "UPDATE sidebar_terminal_threads SET inline_context = ?1 WHERE terminal_id = ?2")?;
            statement.bind(&legacy_json, 1)?;
            statement.bind(&terminal_id, 2)?;
            statement.exec()
        }).await.expect("write legacy position-free styled row");
        row.inline_context.as_mut().expect("context").draft_position = None;
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_none());
        });
        assert!(
            serde_json::to_value(&row.inline_context)
                .expect("legacy encoding")
                .get("draft_position")
                .is_none()
        );
    }

    #[test]
    fn test_terminal_title_prefix_preserves_non_alphanumeric_prefixes() {
        assert_eq!(terminal_title_prefix("✳ Thinking"), Some("✳ "));
        assert_eq!(terminal_title_prefix(">>>   Thinking"), Some(">>>   "));
        assert_eq!(terminal_title_prefix("⠋ Running"), Some("⠋ "));
        assert_eq!(terminal_title_prefix("* Claude"), Some("* "));
        assert_eq!(terminal_title_prefix("✳Thinking"), None);
        assert_eq!(terminal_title_prefix("Thinking"), None);
        assert_eq!(terminal_title_prefix(" Thinking"), None);
        assert_eq!(terminal_title_prefix("✳"), None);
        assert_eq!(terminal_title_prefix("v1 Running"), None);
    }

    #[test]
    fn test_terminal_thread_display_title_combines_raw_and_custom_titles() {
        let mut metadata = metadata(
            "⠋ Thinking",
            WorktreePaths::from_folder_paths(&PathList::default()),
        );
        metadata.custom_title = Some("Fix bug".into());
        assert_eq!(metadata.display_title().as_ref(), "⠋ Fix bug");

        metadata.title = "Thinking".into();
        assert_eq!(metadata.display_title().as_ref(), "Fix bug");
    }

    #[gpui::test]
    async fn test_change_worktree_paths_reindexes_terminal_metadata(cx: &mut TestAppContext) {
        init_test(cx);

        let old_main_paths = PathList::new(&[Path::new("/repo")]);
        let old_folder_paths = PathList::new(&[Path::new("/repo-feature")]);
        let new_main_path = Path::new("/repo");
        let new_folder_path = Path::new("/repo-feature-renamed");
        let new_folder_paths = PathList::new(&[new_folder_path]);
        let metadata = metadata(
            "Dev Server",
            WorktreePaths::from_path_lists(old_main_paths.clone(), old_folder_paths.clone())
                .unwrap(),
        );
        let terminal_id = metadata.terminal_id;

        cx.update(|cx| {
            TerminalThreadMetadataStore::global(cx).update(cx, |store, cx| {
                store.save(metadata, cx);
            });
        });

        cx.update(|cx| {
            TerminalThreadMetadataStore::global(cx).update(cx, |store, cx| {
                store.change_worktree_paths(
                    &old_folder_paths,
                    None,
                    |paths| {
                        paths.add_path(new_main_path, new_folder_path);
                        paths.remove_folder_path(Path::new("/repo-feature"));
                    },
                    cx,
                );
            });
        });

        cx.update(|cx| {
            let store = TerminalThreadMetadataStore::global(cx);
            let store = store.read(cx);
            assert!(
                store
                    .entries_for_path(&old_folder_paths, None)
                    .next()
                    .is_none()
            );
            assert_eq!(
                store
                    .entries_for_path(&new_folder_paths, None)
                    .map(|entry| entry.terminal_id)
                    .collect::<Vec<_>>(),
                vec![terminal_id]
            );
            assert_eq!(
                store
                    .entry(terminal_id)
                    .unwrap()
                    .main_worktree_paths()
                    .paths(),
                old_main_paths.paths()
            );
        });
    }

    #[gpui::test]
    async fn test_inline_terminal_context_survives_database_reload(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        let mut row = metadata("Shell", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("parent-session")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 3,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: true,
            output: "\u{1b}[32mretained output: 界\u{1b}[0m\n".into(),
            snapshot: None,
        });
        database
            .save(row.clone())
            .await
            .expect("save inline context");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row.clone())
        );
        row.inline_context
            .as_mut()
            .expect("inline context")
            .collapsed = false;
        row.inline_context
            .as_mut()
            .expect("inline context")
            .output
            .push_str("next output\n");
        database
            .save(row.clone())
            .await
            .expect("update same terminal");
        let rows = database.list().expect("read terminal rows");
        assert_eq!(
            rows.iter()
                .filter(|item| item.terminal_id == row.terminal_id)
                .count(),
            1
        );
        assert_eq!(
            rows.into_iter()
                .find(|item| item.terminal_id == row.terminal_id),
            Some(row)
        );
    }

    #[gpui::test]
    async fn test_terminal_write_failure_survives_unrelated_success_and_reload(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        database.write(|connection| {
            Statement::prepare(connection, "CREATE TRIGGER remediation_reject_history BEFORE INSERT ON sidebar_terminal_threads WHEN NEW.title = 'Rejected history' BEGIN SELECT RAISE(ABORT, 'remediation rejected history'); END")?.exec()
        }).await.expect("install actual SQLite insert rejection");
        let persisted = metadata(
            "Previous durable history",
            WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from("/previous-history")])),
        );
        database
            .save(persisted.clone())
            .await
            .expect("persist last trustworthy row");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let mut rejected = persisted.clone();
        rejected.title = "Rejected history".into();
        rejected.worktree_paths =
            WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from("/rejected-history")]));
        let terminal_id = rejected.terminal_id;
        store.update(cx, |store, cx| {
            store.save(rejected.clone(), cx);
            assert!(store.write_pending(terminal_id));
            assert!(store.write_error(terminal_id).is_none());
        });
        cx.condition(&store, |store, _| {
            !store.write_pending(terminal_id) && store.write_error(terminal_id).is_some()
        })
        .await;
        store.read_with(cx, |store, _| {
            assert!(
                store
                    .write_error(terminal_id)
                    .expect("visible SQL failure")
                    .contains("remediation rejected history")
            );
            assert_eq!(store.entry(terminal_id), Some(&rejected));
        });
        assert_eq!(
            database
                .list()
                .expect("read last trustworthy persisted rows"),
            vec![persisted],
        );
        let unrelated = metadata("Unrelated durable history", WorktreePaths::default());
        store.update(cx, |store, cx| store.save(unrelated.clone(), cx));
        cx.condition(&store, |store, _| {
            !store.write_pending(unrelated.terminal_id)
        })
        .await;
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert!(
                store.write_error(terminal_id).is_some(),
                "unrelated success cannot clear the affected error"
            );
            assert_eq!(
                store.entry(terminal_id),
                Some(&rejected),
                "reload cannot discard unsaved history"
            );
            assert!(store.path_is_referenced_by_terminal(
                None,
                Path::new("/rejected-history"),
                None
            ));
            assert!(
                !store.path_is_referenced_by_terminal(None, Path::new("/previous-history"), None,),
                "reload cannot leave the persisted row's stale path index"
            );
        });
        database
            .write(|connection| {
                Statement::prepare(connection, "DROP TRIGGER remediation_reject_history")?.exec()
            })
            .await
            .expect("remove disposable rejection trigger");
        store.update(cx, |store, cx| {
            store
                .retry_write(terminal_id, cx)
                .expect("queue the actual failed upsert");
            assert!(store.write_pending(terminal_id));
            assert!(
                store.retry_write(terminal_id, cx).is_err(),
                "retry cannot replay an old operation over a pending write"
            );
            assert!(
                store.write_error(terminal_id).is_some(),
                "retry remains failed until the actual write succeeds"
            );
        });
        cx.condition(&store, |store, _| {
            !store.write_pending(terminal_id) && store.write_error(terminal_id).is_none()
        })
        .await;
        assert!(
            database
                .list()
                .expect("verify actual repaired SQLite row")
                .iter()
                .any(|row| row == &rejected)
        );
    }

    #[gpui::test]
    async fn test_failed_terminal_delete_restores_metadata_and_indexes_until_retry(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let retained = metadata(
            "Keep failed deletion",
            WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from("/delete-history")])),
        );
        let terminal_id = retained.terminal_id;
        database
            .save(retained.clone())
            .await
            .expect("persist actual retained history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        database.write(|connection| {
            Statement::prepare(connection, "CREATE TRIGGER remediation_reject_delete BEFORE DELETE ON sidebar_terminal_threads WHEN OLD.title = 'Keep failed deletion' BEGIN SELECT RAISE(ABORT, 'remediation rejected delete'); END")?.exec()
        }).await.expect("install actual SQLite delete rejection");
        store.update(cx, |store, cx| {
            store.delete(terminal_id, cx);
            store.delete(terminal_id, cx);
            assert!(store.write_pending(terminal_id));
            assert!(store.entry(terminal_id).is_none());
        });
        cx.condition(&store, |store, _| {
            !store.write_pending(terminal_id) && store.write_error(terminal_id).is_some()
        })
        .await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(terminal_id), Some(&retained));
            assert!(store.path_is_referenced_by_terminal(None, Path::new("/delete-history"), None));
            assert!(
                store
                    .write_error(terminal_id)
                    .expect("visible delete failure")
                    .contains("remediation rejected delete")
            );
        });
        assert_eq!(
            database
                .list()
                .expect("verify last trustworthy SQLite state"),
            vec![retained]
        );
        database
            .write(|connection| {
                Statement::prepare(connection, "DROP TRIGGER remediation_reject_delete")?.exec()
            })
            .await
            .expect("remove disposable delete trigger");
        store.update(cx, |store, cx| {
            store
                .retry_write(terminal_id, cx)
                .expect("queue the actual failed delete")
        });
        cx.condition(&store, |store, _| {
            !store.write_pending(terminal_id) && store.write_error(terminal_id).is_none()
        })
        .await;
        assert!(
            database
                .list()
                .expect("verify actual retried delete")
                .is_empty()
        );
        store.read_with(cx, |store, _| {
            assert!(store.entry(terminal_id).is_none());
            assert!(!store.path_is_referenced_by_terminal(
                None,
                Path::new("/delete-history"),
                None
            ));
        });
    }

    #[gpui::test]
    async fn test_inline_terminal_flush_waits_for_latest_sqlite_write(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let mut row = metadata("Initial history", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("flush-parent-session")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: "initial output".into(),
            snapshot: None,
        });
        let first_flush = store.update(cx, |store, cx| {
            store.save(row.clone(), cx);
            store.flush(cx)
        });
        row.title = "Latest history".into();
        row.inline_context
            .as_mut()
            .expect("canonical inline context")
            .output = "latest output".into();
        let second_flush = store.update(cx, |store, cx| {
            store.save(row.clone(), cx);
            store.flush(cx)
        });
        let (first_result, second_result) = futures::join!(first_flush, second_flush);
        first_result.expect("first flush must wait for all accepted revisions");
        second_result.expect("latest flush must complete actual SQLite persistence");
        assert!(!store.read_with(cx, |store, _| store.write_pending(row.terminal_id)));
        assert_eq!(
            database.list().expect("actual durable rows"),
            vec![row.clone()]
        );
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row)
        );
    }

    #[gpui::test]
    async fn test_inline_terminal_normal_quit_flushes_pending_sqlite_write(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let row = metadata("Pending normal quit", WorktreePaths::default());
        store.update(cx, |store, cx| store.save(row.clone(), cx));
        assert!(store.read_with(cx, |store, _| store.write_pending(row.terminal_id)));
        // Normal-quit cases must reach their durable fence rather than inject a
        // randomized scheduler tick timeout. The real shutdown deadline is unchanged.
        cx.background_executor
            .set_block_on_ticks(usize::MAX..=usize::MAX);
        cx.update(|cx| cx.shutdown());
        assert_eq!(
            database.list().expect("durable before any restart pumping"),
            vec![row]
        );
    }

    #[gpui::test]
    async fn test_inline_terminal_normal_quit_reports_actual_sqlite_failure(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        database.write(|connection| {
            Statement::prepare(connection, "CREATE TRIGGER remediation_reject_quit BEFORE INSERT ON sidebar_terminal_threads BEGIN SELECT RAISE(ABORT, 'remediation quit failure'); END")?.exec()
        }).await.expect("install actual quit-time SQLite failure");
        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        store.update(cx, |store, cx| {
            store.save(metadata("Failed normal quit", WorktreePaths::default()), cx);
            cx.on_app_quit({
                let result = result.clone();
                move |store, _| {
                    let flush = store.flush_before_quit();
                    let result = result.clone();
                    async move {
                        *result.lock().expect("quit receipt lock") = Some(flush.await);
                    }
                }
            })
            .detach();
        });
        cx.background_executor
            .set_block_on_ticks(usize::MAX..=usize::MAX);
        cx.update(|cx| cx.shutdown());
        let error = result
            .lock()
            .expect("quit receipt lock")
            .take()
            .expect("the quit fence must complete before shutdown returns")
            .expect_err("actual SQL failure must not become successful quit persistence");
        assert!(error.to_string().contains("remediation quit failure"));
        assert!(
            database
                .list()
                .expect("read durable SQL without pumping")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn test_inline_terminal_quit_accepts_valid_replacement_of_rejected_snapshot(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let (trusted, invalid) = trusted_and_invalid_styled_write_metadata(cx);
        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        store.update(cx, |store, cx| {
            store.save(invalid, cx);
            assert!(store.write_error(trusted.terminal_id).is_some());
            assert!(!store.write_pending(trusted.terminal_id));
            store.save(trusted.clone(), cx);
            assert!(store.write_pending(trusted.terminal_id));
            assert!(
                store.write_error(trusted.terminal_id).is_some(),
                "an accepted replacement is not yet a durable write"
            );
            cx.on_app_quit({
                let result = result.clone();
                move |store, _| {
                    let flush = store.flush_before_quit();
                    let result = result.clone();
                    async move {
                        *result.lock().expect("quit receipt lock") = Some(flush.await);
                    }
                }
            })
            .detach();
        });
        cx.background_executor
            .set_block_on_ticks(usize::MAX..=usize::MAX);
        cx.update(|cx| cx.shutdown());
        assert_eq!(
            database.list().expect("durable before any restart pumping"),
            vec![trusted]
        );
        result
            .lock()
            .expect("quit receipt lock")
            .take()
            .expect("quit fence must settle without foreground completion")
            .expect("actual newer valid write must repair the older snapshot rejection");
    }

    #[gpui::test]
    async fn test_inline_terminal_quit_keeps_snapshot_rejection_newer_than_accepted_write(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let (trusted, invalid) = trusted_and_invalid_styled_write_metadata(cx);
        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        store.update(cx, |store, cx| {
            store.save(trusted.clone(), cx);
            store.save(invalid, cx);
            assert!(store.write_pending(trusted.terminal_id));
            assert!(store.write_error(trusted.terminal_id).is_some());
            cx.on_app_quit({
                let result = result.clone();
                move |store, _| {
                    let flush = store.flush_before_quit();
                    let result = result.clone();
                    async move {
                        *result.lock().expect("quit receipt lock") = Some(flush.await);
                    }
                }
            })
            .detach();
        });
        cx.background_executor
            .set_block_on_ticks(usize::MAX..=usize::MAX);
        cx.update(|cx| cx.shutdown());
        let error = result
            .lock()
            .expect("quit receipt lock")
            .take()
            .expect("quit fence must settle without foreground completion")
            .expect_err("an older durable write cannot repair the newest rejected snapshot");
        assert!(
            error
                .to_string()
                .contains("Invalid terminal history snapshot")
        );
        assert_eq!(
            database
                .list()
                .expect("retain actual older accepted history"),
            vec![trusted]
        );
    }

    #[gpui::test]
    async fn test_inline_terminal_flush_reports_sqlite_failure_and_retry(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        database.write(|connection| {
            Statement::prepare(connection, "CREATE TRIGGER remediation_reject_flush BEFORE INSERT ON sidebar_terminal_threads BEGIN SELECT RAISE(ABORT, 'remediation flush failure'); END")?.exec()
        }).await.expect("install actual SQLite failure");
        let row = metadata("Pending history", WorktreePaths::default());
        let error = store
            .update(cx, |store, cx| {
                store.save(row.clone(), cx);
                store.flush(cx)
            })
            .await
            .expect_err("settled SQL failure must not become a successful flush");
        assert!(error.to_string().contains("remediation flush failure"));
        assert!(
            database
                .list()
                .expect("failed write stays absent")
                .is_empty()
        );
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row.clone())
        );
        database
            .write(|connection| {
                Statement::prepare(connection, "DROP TRIGGER remediation_reject_flush")?.exec()
            })
            .await
            .expect("remove disposable failure trigger");
        store
            .update(cx, |store, cx| {
                store.retry_write(row.terminal_id, cx).expect("queue retry");
                store.flush(cx)
            })
            .await
            .expect("retried actual SQL write must settle");
        assert_eq!(database.list().expect("actual retried row"), vec![row]);
    }

    #[gpui::test]
    async fn test_closed_terminal_write_queue_exposes_error_and_keeps_history(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let retained = metadata("Unsaved queue failure", WorktreePaths::default());
        store.update(cx, |store, cx| {
            store.pending_terminal_ops_tx.close();
            store.save(retained.clone(), cx);
            assert!(!store.write_pending(retained.terminal_id));
            assert!(store.write_error(retained.terminal_id).is_some());
            assert_eq!(store.entry(retained.terminal_id), Some(&retained));
            assert!(
                store.retry_write(retained.terminal_id, cx).is_err(),
                "closed-queue retry must propagate its failure"
            );
            assert!(!store.write_pending(retained.terminal_id));
            store.delete(retained.terminal_id, cx);
            assert_eq!(store.entry(retained.terminal_id), Some(&retained));
            assert!(
                store
                    .write_error(retained.terminal_id)
                    .expect("visible unavailable queue")
                    .contains("queue is unavailable")
            );
        });
        assert!(
            store.update(cx, |store, cx| store.flush(cx)).await.is_err(),
            "an unavailable queue must not report a successful quit flush"
        );
    }

    fn trusted_and_invalid_styled_write_metadata(
        cx: &mut TestAppContext,
    ) -> (TerminalThreadMetadata, TerminalThreadMetadata) {
        let terminal = cx.new(|cx| {
            terminal::TerminalBuilder::new_display_only(
                terminal::terminal_settings::CursorShape::Block,
                settings::AlternateScroll::Off,
                Some(128),
                cx.entity_id().as_u64(),
                cx.background_executor(),
                util::paths::PathStyle::local(),
            )
            .subscribe(cx)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"accepted durable history\r\n", cx)
        });
        cx.run_until_parked();
        let snapshot = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture actual accepted grid")
        });
        let mut trusted = metadata("Accepted durable history", WorktreePaths::default());
        trusted.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("accepted-write-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 0,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: String::new(),
            snapshot: Some(snapshot),
        });
        let mut encoded =
            serde_json::to_value(&trusted.inline_context).expect("encode actual context");
        let snapshot = encoded.get_mut("snapshot").expect("snapshot present");
        let columns = snapshot
            .get("columns")
            .and_then(serde_json::Value::as_u64)
            .expect("column count");
        snapshot
            .as_object_mut()
            .expect("snapshot object")
            .insert("cursor_column".into(), columns.into());
        let mut invalid = trusted.clone();
        invalid.inline_context =
            serde_json::from_value(encoded).expect("typed but semantically invalid context");
        (trusted, invalid)
    }

    #[gpui::test]
    async fn test_rejected_snapshot_does_not_cancel_accepted_write_or_erase_newer_error(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let (trusted, invalid) = trusted_and_invalid_styled_write_metadata(cx);
        let terminal_id = trusted.terminal_id;
        store.update(cx, |store, cx| {
            store.save(trusted.clone(), cx);
            store.save(invalid, cx);
            assert!(
                store.write_pending(terminal_id),
                "rejection cannot cancel the already accepted operation"
            );
            assert!(store.write_error(terminal_id).is_some());
            assert_eq!(store.entry(terminal_id), Some(&trusted));
        });
        cx.condition(&store, |store, _| !store.write_pending(terminal_id))
            .await;
        store.read_with(cx, |store, _| {
            assert!(
                store
                    .write_error(terminal_id)
                    .expect("newer rejection retained")
                    .contains("Invalid terminal history snapshot")
            );
            assert_eq!(store.entry(terminal_id), Some(&trusted));
        });
        assert_eq!(
            database
                .list()
                .expect("accepted actual SQLite write must complete"),
            vec![trusted.clone()]
        );
        store.update(cx, |store, cx| store.save(trusted, cx));
        cx.condition(&store, |store, _| {
            !store.write_pending(terminal_id) && store.write_error(terminal_id).is_none()
        })
        .await;
    }

    #[gpui::test]
    async fn test_rejected_snapshot_does_not_undo_accepted_terminal_delete(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let (trusted, invalid) = trusted_and_invalid_styled_write_metadata(cx);
        let terminal_id = trusted.terminal_id;
        database
            .save(trusted)
            .await
            .expect("persist accepted actual history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.update(cx, |store, cx| {
            store.delete(terminal_id, cx);
            store.save(invalid, cx);
            assert!(
                store.write_pending(terminal_id),
                "rejection cannot cancel accepted deletion"
            );
            assert!(
                store.entry(terminal_id).is_none(),
                "invalid replacement cannot resurrect the row"
            );
        });
        cx.condition(&store, |store, _| !store.write_pending(terminal_id))
            .await;
        assert!(
            database
                .list()
                .expect("actual accepted delete must complete")
                .is_empty()
        );
        store.read_with(cx, |store, _| {
            assert!(store.entry(terminal_id).is_none());
            assert!(
                store
                    .write_error(terminal_id)
                    .expect("newer rejected replacement retained")
                    .contains("Invalid terminal history snapshot")
            );
        });
    }

    #[gpui::test]
    async fn test_invalid_snapshot_write_preserves_trusted_cache_and_database(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let database = store.read_with(cx, |store, _| store.db.clone());
        let terminal = cx.new(|cx| {
            terminal::TerminalBuilder::new_display_only(
                terminal::terminal_settings::CursorShape::Block,
                settings::AlternateScroll::Off,
                Some(128),
                cx.entity_id().as_u64(),
                cx.background_executor(),
                util::paths::PathStyle::local(),
            )
            .subscribe(cx)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"trusted write history\r\n", cx)
        });
        cx.run_until_parked();
        let snapshot = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture actual canonical grid")
        });
        let mut trusted = metadata("Trusted write history", WorktreePaths::default());
        trusted.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("trusted-write-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 0,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: String::new(),
            snapshot: Some(snapshot),
        });
        database
            .save(trusted.clone())
            .await
            .expect("persist actual trusted history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let mut encoded =
            serde_json::to_value(&trusted.inline_context).expect("encode typed context");
        let snapshot = encoded.get_mut("snapshot").expect("snapshot present");
        let columns = snapshot
            .get("columns")
            .and_then(serde_json::Value::as_u64)
            .expect("column count");
        snapshot
            .as_object_mut()
            .expect("snapshot object")
            .insert("cursor_column".into(), columns.into());
        let mut invalid = trusted.clone();
        invalid.inline_context =
            serde_json::from_value(encoded).expect("syntactically valid typed context");
        store.update(cx, |store, cx| {
            store.save(invalid, cx);
            assert!(!store.write_pending(trusted.terminal_id));
            assert!(
                store
                    .write_error(trusted.terminal_id)
                    .expect("semantic failure exposed")
                    .contains("Invalid terminal history snapshot")
            );
            assert_eq!(store.entry(trusted.terminal_id), Some(&trusted));
            assert!(
                store.retry_write(trusted.terminal_id, cx).is_err(),
                "invalid snapshots require updated valid data, not silent replay"
            );
        });
        cx.run_until_parked();
        assert_eq!(
            database.list().expect("verify trusted SQL unchanged"),
            vec![trusted]
        );
    }

    #[gpui::test]
    async fn test_inline_styled_snapshot_survives_sqlite_reload(cx: &mut TestAppContext) {
        init_test(cx);
        let terminal = cx.new(|cx| {
            terminal::TerminalBuilder::new_display_only(
                terminal::terminal_settings::CursorShape::Block,
                settings::AlternateScroll::Off,
                Some(128),
                cx.entity_id().as_u64(),
                cx.background_executor(),
                util::paths::PathStyle::local(),
            )
            .subscribe(cx)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b[1;31mstyled retained grid\x1b[0m\r\n", cx);
        });
        cx.run_until_parked();
        assert!(
            terminal
                .read_with(cx, |terminal, _| terminal.get_content())
                .contains("styled retained grid")
        );
        let snapshot = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture canonical grid")
        });
        let mut row = metadata("Styled inline history", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("styled-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: true,
            output: String::new(),
            snapshot: Some(snapshot),
        });
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        database
            .save(row.clone())
            .await
            .expect("persist styled history through canonical store");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row)
        );
    }

    #[gpui::test]
    async fn test_invalid_inline_styled_snapshot_retains_trusted_history(cx: &mut TestAppContext) {
        init_test(cx);
        let terminal = cx.new(|cx| {
            terminal::TerminalBuilder::new_display_only(
                terminal::terminal_settings::CursorShape::Block,
                settings::AlternateScroll::Off,
                Some(128),
                cx.entity_id().as_u64(),
                cx.background_executor(),
                util::paths::PathStyle::local(),
            )
            .subscribe(cx)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b[1;31mtrusted styled history\x1b[0m\r\n", cx);
        });
        cx.run_until_parked();
        assert!(
            terminal
                .read_with(cx, |terminal, _| terminal.get_content())
                .contains("trusted styled history")
        );
        let snapshot = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture trusted canonical grid")
        });
        let mut row = metadata("Trusted styled history", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("trusted-styled-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: String::new(),
            snapshot: Some(snapshot),
        });
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        database
            .save(row.clone())
            .await
            .expect("persist trusted history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row.clone())
        );
        let mut corrupt_context =
            serde_json::to_value(&row.inline_context).expect("serialize valid context");
        let corrupt_snapshot = corrupt_context
            .get_mut("snapshot")
            .expect("snapshot present");
        let columns = corrupt_snapshot
            .get("columns")
            .and_then(serde_json::Value::as_u64)
            .expect("column count");
        corrupt_snapshot
            .as_object_mut()
            .expect("snapshot object")
            .insert("cursor_column".into(), columns.into());
        let terminal_id = row.terminal_id.to_key_string();
        let corrupt_json = serde_json::to_string(&corrupt_context).expect("serialize corrupt grid");
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "UPDATE sidebar_terminal_threads SET inline_context = ?1 WHERE terminal_id = ?2")?;
            statement.bind(&corrupt_json, 1)?;
            statement.bind(&terminal_id, 2)?;
            statement.exec()
        }).await.expect("persist syntactically valid but out-of-bounds snapshot");
        let error = database
            .list()
            .expect_err("invalid grid must fail canonical read");
        assert!(format!("{error:#}").contains("Invalid terminal snapshot cursor"));
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(
                store.load_error().is_some_and(
                    |error| error.contains("validate inline terminal history snapshot")
                )
            );
        });
        database
            .save(row.clone())
            .await
            .expect("repair trusted snapshot");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_none());
        });
    }

    #[test]
    fn test_legacy_inline_context_without_styled_snapshot_remains_readable() {
        let context = InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: None,
            restored_draft_source_session_id: None,
            preceding_entry_count: 0,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: "legacy output".into(),
            snapshot: None,
        };
        let mut value = serde_json::to_value(&context).expect("serialize context");
        value
            .as_object_mut()
            .expect("context object")
            .remove("snapshot");
        assert_eq!(
            serde_json::from_value::<InlineTerminalContext>(value).expect("legacy context"),
            context
        );
    }

    #[test]
    fn test_legacy_inline_context_does_not_claim_current_draft_ownership() {
        let context = InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("legacy-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: "retained legacy output".into(),
            snapshot: None,
        };
        let mut value = serde_json::to_value(&context).expect("serialize context");
        let object = value.as_object_mut().expect("context object");
        object.remove("is_draft");
        object.remove("message_client_id");
        let restored: InlineTerminalContext =
            serde_json::from_value(value).expect("legacy context");
        assert!(!restored.is_draft);
        assert!(restored.message_client_id.is_none());
        assert_eq!(restored.output, context.output);
        assert_eq!(
            restored.preceding_entry_count,
            context.preceding_entry_count
        );
    }

    #[gpui::test]
    async fn test_legacy_terminal_row_has_no_inline_context(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        let row = metadata("Legacy Shell", WorktreePaths::default());
        let terminal_id = row.terminal_id.to_key_string();
        let created_at = row.created_at.to_rfc3339();
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "INSERT INTO sidebar_terminal_threads(terminal_id, title, created_at) VALUES (?1, 'Legacy Shell', ?2)")?;
            statement.bind(&terminal_id, 1)?;
            statement.bind(&created_at, 2)?;
            statement.exec()
        }).await.expect("insert a legacy record without the new column");
        let loaded = database
            .list()
            .expect("load legacy metadata")
            .into_iter()
            .find(|entry| entry.terminal_id == row.terminal_id)
            .expect("retained legacy terminal");
        assert_eq!(loaded, row);
        assert!(loaded.inline_context.is_none());
    }

    #[gpui::test]
    async fn test_malformed_inline_context_is_an_explicit_read_error(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        let row = metadata("Malformed context", WorktreePaths::default());
        let terminal_id = row.terminal_id.to_key_string();
        database.save(row).await.expect("save initial terminal");
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "UPDATE sidebar_terminal_threads SET inline_context = '{broken' WHERE terminal_id = ?1")?;
            statement.bind(&terminal_id, 1)?;
            statement.exec()
        }).await.expect("set malformed persisted context");
        let error = database
            .list()
            .expect_err("malformed context must not become empty history");
        assert!(format!("{error:#}").contains("deserialize inline terminal conversation context"));
    }

    #[gpui::test]
    async fn test_inline_terminal_lookup_keeps_parent_session_and_host_scope(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let thread_id = ThreadId::new();
        let session_id = acp::SessionId::new("active-session");
        let other_session_id = acp::SessionId::new("other-session");
        let remote = RemoteConnectionOptions::Mock(remote::MockConnectionOptions { id: 1 });
        let other_remote = RemoteConnectionOptions::Mock(remote::MockConnectionOptions { id: 2 });
        let mut row = metadata("Scoped Shell", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id,
            session_id: Some(session_id.clone()),
            restored_draft_source_session_id: None,
            preceding_entry_count: 2,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: "preserved".into(),
            snapshot: None,
        });
        let expected = row.terminal_id;
        let mut other_session = row.clone();
        other_session.terminal_id = TerminalId::new();
        other_session
            .inline_context
            .as_mut()
            .expect("inline context")
            .session_id = Some(other_session_id);
        let mut other_parent = row.clone();
        other_parent.terminal_id = TerminalId::new();
        other_parent
            .inline_context
            .as_mut()
            .expect("inline context")
            .thread_id = ThreadId::new();
        let mut other_host = row.clone();
        other_host.terminal_id = TerminalId::new();
        other_host.remote_connection = Some(other_remote);
        let remote_id = other_host.terminal_id;
        let mut draft = row.clone();
        draft.terminal_id = TerminalId::new();
        draft
            .inline_context
            .as_mut()
            .expect("inline context")
            .session_id = None;
        let draft_id = draft.terminal_id;
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        store.update(cx, |store, cx| {
            for row in [row, other_session, other_parent, other_host, draft] {
                store.save(row, cx);
            }
        });
        store.read_with(cx, |store, _| {
            assert_eq!(
                store
                    .entries_for_conversation(thread_id, Some(&session_id), None)
                    .map(|entry| entry.terminal_id)
                    .collect::<Vec<_>>(),
                vec![expected]
            );
            assert_eq!(
                store
                    .entries_for_conversation(thread_id, None, None)
                    .map(|entry| entry.terminal_id)
                    .collect::<Vec<_>>(),
                vec![draft_id]
            );
            assert!(
                store
                    .entries_for_conversation(thread_id, Some(&session_id), Some(&remote))
                    .next()
                    .is_none()
            );
            assert_eq!(
                store
                    .entries_for_conversation(
                        thread_id,
                        Some(&session_id),
                        Some(&RemoteConnectionOptions::Mock(
                            remote::MockConnectionOptions { id: 2 }
                        ))
                    )
                    .map(|entry| entry.terminal_id)
                    .collect::<Vec<_>>(),
                vec![remote_id]
            );
        });
    }

    #[gpui::test]
    async fn test_standalone_refresh_preserves_inline_context(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        let mut row = metadata("Retained history", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("parent-session")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 5,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: true,
            output: "canonical retained history".into(),
            snapshot: None,
        });
        database
            .save(row.clone())
            .await
            .expect("save inline metadata");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let mut refresh = row.clone();
        refresh.title = "Refreshed title".into();
        refresh.inline_context = None;
        store.update(cx, |store, cx| store.save(refresh.clone(), cx));
        row.title = refresh.title.clone();
        assert_eq!(
            store.read_with(cx, |store, _| store.entry(row.terminal_id).cloned()),
            Some(row.clone())
        );
        database
            .save(refresh)
            .await
            .expect("save standalone host refresh");
        let loaded = database
            .list()
            .expect("load retained association")
            .into_iter()
            .find(|entry| entry.terminal_id == row.terminal_id);
        assert_eq!(loaded, Some(row));
    }
    #[gpui::test]
    async fn test_failed_reload_retains_history_and_exposes_error(cx: &mut TestAppContext) {
        init_test(cx);
        let store = cx.update(|cx| TerminalThreadMetadataStore::global(cx));
        let database = store.read_with(cx, |store, _| store.db.clone());
        let mut row = metadata("History retained on error", WorktreePaths::default());
        row.inline_context = Some(InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: ThreadId::new(),
            session_id: Some(acp::SessionId::new("retained-parent")),
            restored_draft_source_session_id: None,
            preceding_entry_count: 1,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: "known retained output".into(),
            snapshot: None,
        });
        database
            .save(row.clone())
            .await
            .expect("save valid history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        let terminal_id = row.terminal_id.to_key_string();
        database.write(move |connection| {
            let mut statement = Statement::prepare(connection,
                "UPDATE sidebar_terminal_threads SET inline_context = '{broken' WHERE terminal_id = ?1")?;
            statement.bind(&terminal_id, 1)?;
            statement.exec()
        }).await.expect("set malformed history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_some_and(|error| {
                error.contains("deserialize inline terminal conversation context")
            }));
        });
        database
            .save(row.clone())
            .await
            .expect("repair valid history");
        store.update(cx, |store, cx| store.reload(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            assert_eq!(store.entry(row.terminal_id), Some(&row));
            assert!(store.load_error().is_none());
        });
    }
}
