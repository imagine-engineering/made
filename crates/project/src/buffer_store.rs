use crate::{
    ProjectPath,
    lsp_store::OpenLspBufferHandle,
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};
use anyhow::{Context as _, Result, anyhow};
use client::{Client, Status as ClientStatus};
use collections::{HashMap, HashSet, hash_map};
use futures::{Future, FutureExt as _, channel::oneshot, future::Shared};
use gpui::{
    App, AppContext as _, AsyncApp, Context, Entity, EventEmitter, Subscription, Task, WeakEntity,
};
use language::{
    Buffer, BufferEvent, Capability, DiskState, File as _, Language, LineEnding, Operation,
    language_settings::{AllLanguageSettings, LineEndingSetting},
    proto::{
        deserialize_line_ending, deserialize_version, serialize_line_ending, serialize_version,
        split_operations,
    },
};
use rpc::{
    AnyProtoClient, ErrorCode, ErrorExt as _, TypedEnvelope,
    proto::{self, PeerId},
};

use settings::Settings;
use std::{io, sync::Arc, time::Instant};
use text::{BufferId, ReplicaId};
use util::{ResultExt as _, debug_panic, maybe, rel_path::RelPath};
use worktree::{File, PathChange, ProjectEntryId, Worktree, WorktreeId, WorktreeSettings};

/// A set of open buffers.
pub struct BufferStore {
    state: BufferStoreState,
    #[allow(clippy::type_complexity)]
    loading_buffers: HashMap<ProjectPath, Shared<Task<Result<Entity<Buffer>, Arc<anyhow::Error>>>>>,
    worktree_store: Entity<WorktreeStore>,
    opened_buffers: HashMap<BufferId, OpenBuffer>,
    path_to_buffer_id: HashMap<ProjectPath, BufferId>,
    downstream_client: Option<(AnyProtoClient, u64)>,
    shared_buffers: HashMap<proto::PeerId, HashMap<BufferId, SharedBuffer>>,
    pending_synchronizations: HashMap<proto::PeerId, std::sync::Weak<()>>,
    #[cfg(any(test, feature = "test-support"))]
    test_synchronization_preparation: Option<TestSynchronizationPreparationGate>,
    #[cfg(any(test, feature = "test-support"))]
    test_synchronization_commit_observer: Option<TestSynchronizationCommitObserver>,
    #[cfg(any(test, feature = "test-support"))]
    test_peer_save_wait: Option<TestPeerSaveWait>,
    non_searchable_buffers: HashSet<BufferId>,
    project_search: RemoteProjectSearchState,
}

#[derive(Default)]
struct RemoteProjectSearchState {
    // List of ongoing project search chunks from our remote host. Used by the side issuing a search RPC request.
    chunks: HashMap<u64, async_channel::Sender<BufferId>>,
    // Monotonously-increasing handle to hand out to remote host in order to identify the project search result chunk.
    next_id: u64,
    // Used by the side running the actual search for match candidates to potentially cancel the search prematurely.
    searches_in_progress: HashMap<(PeerId, u64), Task<Result<()>>>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct SharedBuffer {
    buffer: Entity<Buffer>,
    lsp_handle: Option<OpenLspBufferHandle>,
}

struct BufferSynchronizationRequest {
    project_id: u64,
    peer_id: proto::PeerId,
    publishing_to_collaboration: bool,
    client: Arc<Client>,
    client_status: ClientStatus,
    token: Arc<()>,
}

struct SynchronizedBuffer {
    buffer_id: BufferId,
    buffer: Entity<Buffer>,
    version: proto::BufferVersion,
    file_binding: Option<ProjectPath>,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct TestSynchronizationPrepared {
    pub project_id: u64,
    pub peer_id: proto::PeerId,
    pub version: proto::BufferVersion,
    pub chunks: Vec<Vec<proto::Operation>>,
}

#[cfg(any(test, feature = "test-support"))]
struct TestSynchronizationPreparationGate {
    project_id: u64,
    peer_id: proto::PeerId,
    buffer_id: u64,
    observed: oneshot::Sender<TestSynchronizationPrepared>,
    release: oneshot::Receiver<()>,
}

#[cfg(any(test, feature = "test-support"))]
struct TestSynchronizationCommitObserver {
    project_id: u64,
    peer_id: proto::PeerId,
    observed: oneshot::Sender<proto::SynchronizeBuffersResponse>,
}

#[cfg(any(test, feature = "test-support"))]
struct TestPeerSaveWait {
    project_id: u64,
    buffer_id: BufferId,
    observed: oneshot::Sender<()>,
}

pub(crate) struct PreparedSynchronization {
    request: BufferSynchronizationRequest,
    buffers: Vec<SynchronizedBuffer>,
}

enum BufferStoreState {
    Local(LocalBufferStore),
    Remote(RemoteBufferStore),
}

struct RemoteBufferStore {
    shared_with_me: HashSet<Entity<Buffer>>,
    upstream_client: AnyProtoClient,
    project_id: u64,
    loading_remote_buffers_by_id: HashMap<BufferId, Entity<Buffer>>,
    remote_buffer_listeners:
        HashMap<BufferId, Vec<oneshot::Sender<anyhow::Result<Entity<Buffer>>>>>,
    worktree_store: Entity<WorktreeStore>,
}

struct LocalBufferStore {
    local_buffer_ids_by_entry_id: HashMap<ProjectEntryId, BufferId>,
    worktree_store: Entity<WorktreeStore>,
    _subscription: Subscription,
}

enum OpenBuffer {
    Complete { buffer: WeakEntity<Buffer> },
    Operations(Vec<Operation>),
}

pub enum BufferStoreEvent {
    BufferAdded(Entity<Buffer>),
    SharedBufferClosed(proto::PeerId, BufferId),
    BufferDropped(BufferId),
    BufferChangedFilePath {
        buffer: Entity<Buffer>,
        old_file: Option<Arc<dyn language::File>>,
    },
}

#[derive(Default, Debug, Clone)]
pub struct ProjectTransaction(pub HashMap<Entity<Buffer>, language::Transaction>);

impl PartialEq for ProjectTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().all(|(buffer, transaction)| {
                other.0.get(buffer).is_some_and(|t| t.id == transaction.id)
            })
    }
}

impl EventEmitter<BufferStoreEvent> for BufferStore {}

impl RemoteBufferStore {
    pub fn wait_for_remote_buffer(
        &mut self,
        id: BufferId,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        let (tx, rx) = oneshot::channel();
        self.remote_buffer_listeners.entry(id).or_default().push(tx);

        cx.spawn(async move |this, cx| {
            if let Some(buffer) = this
                .read_with(cx, |buffer_store, _| buffer_store.get(id))
                .ok()
                .flatten()
            {
                return Ok(buffer);
            }

            cx.background_spawn(async move { rx.await? }).await
        })
    }

    fn save_remote_buffer(
        &self,
        buffer_handle: Entity<Buffer>,
        new_path: Option<proto::ProjectPath>,
        cx: &Context<BufferStore>,
    ) -> Task<Result<()>> {
        let buffer = buffer_handle.read(cx);
        let buffer_id = buffer.remote_id().into();
        let version = buffer.version();
        let rpc = self.upstream_client.clone();
        let project_id = self.project_id;
        cx.spawn(async move |_, cx| {
            let response = rpc
                .request(proto::SaveBuffer {
                    project_id,
                    buffer_id,
                    new_path,
                    version: serialize_version(&version),
                })
                .await?;
            let version = deserialize_version(&response.version);
            let mtime = response.mtime.map(|mtime| mtime.into());

            buffer_handle.update(cx, |buffer, cx| {
                buffer.did_save(version.clone(), mtime, cx);
            });

            Ok(())
        })
    }

    pub fn handle_create_buffer_for_peer(
        &mut self,
        envelope: TypedEnvelope<proto::CreateBufferForPeer>,
        replica_id: ReplicaId,
        capability: Capability,
        cx: &mut Context<BufferStore>,
    ) -> Result<Option<Entity<Buffer>>> {
        match envelope.payload.variant.context("missing variant")? {
            proto::create_buffer_for_peer::Variant::State(mut state) => {
                let buffer_id = BufferId::new(state.id)?;

                let buffer_result = maybe!({
                    let mut buffer_file = None;
                    if let Some(file) = state.file.take() {
                        let worktree_id = worktree::WorktreeId::from_proto(file.worktree_id);
                        let worktree = self
                            .worktree_store
                            .read(cx)
                            .worktree_for_id(worktree_id, cx)
                            .with_context(|| {
                                format!("no worktree found for id {}", file.worktree_id)
                            })?;
                        buffer_file = Some(Arc::new(File::from_proto(file, worktree, cx)?)
                            as Arc<dyn language::File>);
                    }
                    Buffer::from_proto(replica_id, capability, state, buffer_file)
                });

                match buffer_result {
                    Ok(buffer) => {
                        let buffer = cx.new(|_| buffer);
                        self.loading_remote_buffers_by_id.insert(buffer_id, buffer);
                    }
                    Err(error) => {
                        if let Some(listeners) = self.remote_buffer_listeners.remove(&buffer_id) {
                            for listener in listeners {
                                listener.send(Err(anyhow!(error.cloned()))).ok();
                            }
                        }
                    }
                }
            }
            proto::create_buffer_for_peer::Variant::Chunk(chunk) => {
                let buffer_id = BufferId::new(chunk.buffer_id)?;
                let buffer = self
                    .loading_remote_buffers_by_id
                    .get(&buffer_id)
                    .cloned()
                    .with_context(|| {
                        format!(
                            "received chunk for buffer {} without initial state",
                            chunk.buffer_id
                        )
                    })?;

                let result = maybe!({
                    let operations = chunk
                        .operations
                        .into_iter()
                        .map(language::proto::deserialize_operation)
                        .collect::<Result<Vec<_>>>()?;
                    buffer.update(cx, |buffer, cx| buffer.apply_ops(operations, cx));
                    anyhow::Ok(())
                });

                if let Err(error) = result {
                    self.loading_remote_buffers_by_id.remove(&buffer_id);
                    if let Some(listeners) = self.remote_buffer_listeners.remove(&buffer_id) {
                        for listener in listeners {
                            listener.send(Err(error.cloned())).ok();
                        }
                    }
                } else if chunk.is_last {
                    self.loading_remote_buffers_by_id.remove(&buffer_id);
                    if self.upstream_client.is_via_collab() {
                        // retain buffers sent by peers to avoid races.
                        self.shared_with_me.insert(buffer.clone());
                    }

                    if let Some(senders) = self.remote_buffer_listeners.remove(&buffer_id) {
                        for sender in senders {
                            sender.send(Ok(buffer.clone())).ok();
                        }
                    }
                    return Ok(Some(buffer));
                }
            }
        }
        Ok(None)
    }

    pub fn incomplete_buffer_ids(&self) -> Vec<BufferId> {
        self.loading_remote_buffers_by_id
            .keys()
            .copied()
            .collect::<Vec<_>>()
    }

    pub fn deserialize_project_transaction(
        &self,
        message: proto::ProjectTransaction,
        push_to_history: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<ProjectTransaction>> {
        cx.spawn(async move |this, cx| {
            let mut project_transaction = ProjectTransaction::default();
            for (buffer_id, transaction) in message.buffer_ids.into_iter().zip(message.transactions)
            {
                let buffer_id = BufferId::new(buffer_id)?;
                let buffer = this
                    .update(cx, |this, cx| this.wait_for_remote_buffer(buffer_id, cx))?
                    .await?;
                let transaction = language::proto::deserialize_transaction(transaction)?;
                project_transaction.0.insert(buffer, transaction);
            }

            for (buffer, transaction) in &project_transaction.0 {
                buffer
                    .update(cx, |buffer, _| {
                        buffer.wait_for_edits(transaction.edit_ids.iter().copied())
                    })
                    .await?;

                if push_to_history {
                    buffer.update(cx, |buffer, _| {
                        buffer.push_transaction(transaction.clone(), Instant::now());
                        buffer.finalize_last_transaction();
                    });
                }
            }

            Ok(project_transaction)
        })
    }

    fn open_buffer(
        &self,
        path: Arc<RelPath>,
        worktree: Entity<Worktree>,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        let worktree_id = worktree.read(cx).id().to_proto();
        let project_id = self.project_id;
        let client = self.upstream_client.clone();
        cx.spawn(async move |this, cx| {
            let response = client
                .request(proto::OpenBufferByPath {
                    project_id,
                    worktree_id,
                    path: path.as_unix_str().to_owned(),
                })
                .await?;
            let buffer_id = BufferId::new(response.buffer_id)?;

            let buffer = this
                .update(cx, {
                    |this, cx| this.wait_for_remote_buffer(buffer_id, cx)
                })?
                .await?;

            Ok(buffer)
        })
    }

    fn create_buffer(
        &self,
        language: Option<Arc<Language>>,
        project_searchable: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        let create = self.upstream_client.request(proto::OpenNewBuffer {
            project_id: self.project_id,
        });
        cx.spawn(async move |this, cx| {
            let response = create.await?;
            let buffer_id = BufferId::new(response.buffer_id)?;

            let buffer = this
                .update(cx, |this, cx| {
                    if !project_searchable {
                        this.non_searchable_buffers.insert(buffer_id);
                    }
                    this.wait_for_remote_buffer(buffer_id, cx)
                })?
                .await?;
            if let Some(language) = language {
                buffer.update(cx, |buffer, cx| {
                    buffer.set_language(Some(language), cx);
                });
            }
            Ok(buffer)
        })
    }

    fn reload_buffers(
        &self,
        buffers: HashSet<Entity<Buffer>>,
        push_to_history: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<ProjectTransaction>> {
        let request = self.upstream_client.request(proto::ReloadBuffers {
            project_id: self.project_id,
            buffer_ids: buffers
                .iter()
                .map(|buffer| buffer.read(cx).remote_id().to_proto())
                .collect(),
        });

        cx.spawn(async move |this, cx| {
            let response = request.await?.transaction.context("missing transaction")?;
            this.update(cx, |this, cx| {
                this.deserialize_project_transaction(response, push_to_history, cx)
            })?
            .await
        })
    }
}

impl LocalBufferStore {
    fn save_local_buffer(
        &self,
        buffer_handle: Entity<Buffer>,
        worktree: Entity<Worktree>,
        path: Arc<RelPath>,
        mut has_changed_file: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<()>> {
        let buffer = buffer_handle.read(cx);

        let text = buffer.as_rope().clone();
        let line_ending = buffer.line_ending();
        let encoding = buffer.encoding();
        let has_bom = buffer.has_bom();
        let version = buffer.version();
        let buffer_id = buffer.remote_id();
        let file = buffer.file().cloned();
        if file
            .as_ref()
            .is_some_and(|file| file.disk_state() == DiskState::New)
        {
            has_changed_file = true;
        }

        let save = worktree.update(cx, |worktree, cx| {
            worktree.write_file(path, text, line_ending, encoding, has_bom, cx)
        });

        cx.spawn(async move |this, cx| {
            let new_file = save.await?;
            let mtime = new_file.disk_state().mtime();
            this.update(cx, |this, cx| {
                if let Some((downstream_client, project_id)) = this.downstream_client.clone() {
                    if has_changed_file {
                        downstream_client
                            .send(proto::UpdateBufferFile {
                                project_id,
                                buffer_id: buffer_id.to_proto(),
                                file: Some(language::File::to_proto(&*new_file, cx)),
                            })
                            .log_err();
                    }
                    downstream_client
                        .send(proto::BufferSaved {
                            project_id,
                            buffer_id: buffer_id.to_proto(),
                            version: serialize_version(&version),
                            mtime: mtime.map(|time| time.into()),
                        })
                        .log_err();
                }
            })?;
            buffer_handle.update(cx, |buffer, cx| {
                if has_changed_file {
                    buffer.file_updated(new_file, cx);
                }
                buffer.did_save(version.clone(), mtime, cx);
            });
            Ok(())
        })
    }

    fn subscribe_to_worktree(
        &mut self,
        worktree: &Entity<Worktree>,
        cx: &mut Context<BufferStore>,
    ) {
        cx.subscribe(worktree, |this, worktree, event, cx| {
            if worktree.read(cx).is_local()
                && let worktree::Event::UpdatedEntries(changes) = event
            {
                Self::local_worktree_entries_changed(this, &worktree, changes, cx);
            }
        })
        .detach();
    }

    fn local_worktree_entries_changed(
        this: &mut BufferStore,
        worktree_handle: &Entity<Worktree>,
        changes: &[(Arc<RelPath>, ProjectEntryId, PathChange)],
        cx: &mut Context<BufferStore>,
    ) {
        let snapshot = worktree_handle.read(cx).snapshot();
        for (path, entry_id, _) in changes {
            Self::local_worktree_entry_changed(
                this,
                *entry_id,
                path,
                worktree_handle,
                &snapshot,
                cx,
            );
        }
    }

    fn local_worktree_entry_changed(
        this: &mut BufferStore,
        entry_id: ProjectEntryId,
        path: &Arc<RelPath>,
        worktree: &Entity<worktree::Worktree>,
        snapshot: &worktree::Snapshot,
        cx: &mut Context<BufferStore>,
    ) -> Option<()> {
        let project_path = ProjectPath {
            worktree_id: snapshot.id(),
            path: path.clone(),
        };

        let buffer_id = this
            .as_local_mut()
            .and_then(|local| local.local_buffer_ids_by_entry_id.get(&entry_id))
            .copied()
            .or_else(|| this.path_to_buffer_id.get(&project_path).copied())?;

        let buffer = if let Some(buffer) = this.get(buffer_id) {
            Some(buffer)
        } else {
            this.opened_buffers.remove(&buffer_id);
            this.non_searchable_buffers.remove(&buffer_id);
            None
        };

        let buffer = if let Some(buffer) = buffer {
            buffer
        } else {
            this.path_to_buffer_id.remove(&project_path);
            let this = this.as_local_mut()?;
            this.local_buffer_ids_by_entry_id.remove(&entry_id);
            return None;
        };

        let events = buffer.update(cx, |buffer, cx| {
            let file = buffer.file()?;
            let old_file = File::from_dyn(Some(file))?;
            if old_file.worktree != *worktree {
                return None;
            }

            let snapshot_entry = old_file
                .entry_id
                .and_then(|entry_id| snapshot.entry_for_id(entry_id))
                .or_else(|| snapshot.entry_for_path(old_file.path.as_ref()));

            let new_file = if let Some(entry) = snapshot_entry {
                File {
                    disk_state: match entry.mtime {
                        Some(mtime) => DiskState::Present {
                            mtime,
                            size: entry.size,
                        },
                        None => old_file.disk_state,
                    },
                    is_local: true,
                    entry_id: Some(entry.id),
                    path: entry.path.clone(),
                    worktree: worktree.clone(),
                    is_private: entry.is_private,
                }
            } else {
                File {
                    disk_state: DiskState::Deleted,
                    is_local: true,
                    entry_id: old_file.entry_id,
                    path: old_file.path.clone(),
                    worktree: worktree.clone(),
                    is_private: old_file.is_private,
                }
            };

            if new_file == *old_file {
                return None;
            }

            let mut events = Vec::new();
            if new_file.path != old_file.path {
                this.path_to_buffer_id.remove(&ProjectPath {
                    path: old_file.path.clone(),
                    worktree_id: old_file.worktree_id(cx),
                });
                this.path_to_buffer_id.insert(
                    ProjectPath {
                        worktree_id: new_file.worktree_id(cx),
                        path: new_file.path.clone(),
                    },
                    buffer_id,
                );
                events.push(BufferStoreEvent::BufferChangedFilePath {
                    buffer: cx.entity(),
                    old_file: buffer.file().cloned(),
                });
            }
            let local = this.as_local_mut()?;
            if new_file.entry_id != old_file.entry_id {
                if let Some(entry_id) = old_file.entry_id {
                    local.local_buffer_ids_by_entry_id.remove(&entry_id);
                }
                if let Some(entry_id) = new_file.entry_id {
                    local
                        .local_buffer_ids_by_entry_id
                        .insert(entry_id, buffer_id);
                }
            }

            if let Some((client, project_id)) = &this.downstream_client {
                client
                    .send(proto::UpdateBufferFile {
                        project_id: *project_id,
                        buffer_id: buffer_id.to_proto(),
                        file: Some(new_file.to_proto(cx)),
                    })
                    .ok();
            }

            buffer.file_updated(Arc::new(new_file), cx);
            Some(events)
        })?;

        for event in events {
            cx.emit(event);
        }

        None
    }

    fn save_buffer(
        &self,
        buffer: Entity<Buffer>,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<()>> {
        let Some(file) = File::from_dyn(buffer.read(cx).file()) else {
            return Task::ready(Err(anyhow!("buffer doesn't have a file")));
        };
        let worktree = file.worktree.clone();
        self.save_local_buffer(buffer, worktree, file.path.clone(), false, cx)
    }

    fn save_buffer_as(
        &self,
        buffer: Entity<Buffer>,
        path: ProjectPath,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<()>> {
        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(path.worktree_id, cx)
        else {
            return Task::ready(Err(anyhow!("no such worktree")));
        };
        self.save_local_buffer(buffer, worktree, path.path, true, cx)
    }

    #[ztracing::instrument(skip_all)]
    fn open_buffer_internal(
        &self,
        path: Arc<RelPath>,
        worktree: Entity<Worktree>,
        max_bytes: Option<usize>,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        let load_file = worktree.update(cx, |worktree, cx| match max_bytes {
            Some(limit) => worktree.load_file_with_limit(path.as_ref(), limit, cx),
            None => worktree.load_file(path.as_ref(), cx),
        });
        cx.spawn(async move |this, cx| {
            let path = path.clone();
            let buffer = match load_file.await {
                Ok(loaded) => {
                    let is_writable = loaded.is_writable;
                    let capability = if is_writable {
                        Capability::ReadWrite
                    } else {
                        Capability::Read
                    };
                    let reservation = cx.reserve_entity::<Buffer>();
                    let buffer_id = BufferId::from(reservation.entity_id().as_non_zero_u64());
                    let text_buffer = cx
                        .background_spawn(async move {
                            text::Buffer::new(ReplicaId::LOCAL, buffer_id, loaded.text)
                        })
                        .await;
                    cx.insert_entity(reservation, |_| {
                        let mut buffer = Buffer::build(text_buffer, Some(loaded.file), capability);
                        buffer.set_encoding(loaded.encoding);
                        buffer.set_has_bom(loaded.has_bom);
                        buffer
                    })
                }
                Err(error) if max_bytes.is_none() && is_not_found_error(&error) => cx.new(|cx| {
                    let buffer_id = BufferId::from(cx.entity_id().as_non_zero_u64());
                    let text_buffer = text::Buffer::new(ReplicaId::LOCAL, buffer_id, "");
                    let mut buffer = Buffer::build(
                        text_buffer,
                        Some(Arc::new(File {
                            worktree,
                            path,
                            disk_state: DiskState::New,
                            entry_id: None,
                            is_local: true,
                            is_private: false,
                        })),
                        Capability::ReadWrite,
                    );
                    apply_initial_line_ending(&mut buffer, cx);
                    buffer
                }),
                Err(e) => return Err(e),
            };
            this.update(cx, |this, cx| {
                this.add_buffer(buffer.clone(), cx)?;
                let buffer_id = buffer.read(cx).remote_id();
                if let Some(file) = File::from_dyn(buffer.read(cx).file()) {
                    let project_path = ProjectPath {
                        worktree_id: file.worktree_id(cx),
                        path: file.path.clone(),
                    };
                    let entry_id = file.entry_id;

                    // Check if the file should be read-only based on settings
                    let settings = WorktreeSettings::get(Some((&project_path).into()), cx);
                    let is_read_only = if project_path.path.is_empty() {
                        settings.is_std_path_read_only(&file.full_path(cx))
                    } else {
                        settings.is_path_read_only(&project_path.path)
                    };
                    if is_read_only {
                        buffer.update(cx, |buffer, cx| {
                            buffer.set_capability(Capability::Read, cx);
                        });
                    }

                    this.path_to_buffer_id.insert(project_path, buffer_id);
                    let this = this.as_local_mut().unwrap();
                    if let Some(entry_id) = entry_id {
                        this.local_buffer_ids_by_entry_id
                            .insert(entry_id, buffer_id);
                    }
                }

                anyhow::Ok(())
            })??;

            Ok(buffer)
        })
    }

    fn create_buffer(
        &self,
        language: Option<Arc<Language>>,
        project_searchable: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        cx.spawn(async move |buffer_store, cx| {
            let buffer = cx.new(|cx| {
                let mut buffer = Buffer::local("", cx)
                    .with_language(language.unwrap_or_else(|| language::PLAIN_TEXT.clone()), cx);
                apply_initial_line_ending(&mut buffer, cx);
                buffer
            });
            buffer_store.update(cx, |buffer_store, cx| {
                buffer_store.add_buffer(buffer.clone(), cx).log_err();
                if !project_searchable {
                    buffer_store
                        .non_searchable_buffers
                        .insert(buffer.read(cx).remote_id());
                }
            })?;
            Ok(buffer)
        })
    }

    fn reload_buffers(
        &self,
        buffers: HashSet<Entity<Buffer>>,
        push_to_history: bool,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<ProjectTransaction>> {
        cx.spawn(async move |_, cx| {
            let mut project_transaction = ProjectTransaction::default();
            for buffer in buffers {
                let transaction = buffer.update(cx, |buffer, cx| buffer.reload(cx)).await?;
                buffer.update(cx, |buffer, cx| {
                    if let Some(transaction) = transaction {
                        if !push_to_history {
                            buffer.forget_transaction(transaction.id);
                        }
                        project_transaction.0.insert(cx.entity(), transaction);
                    }
                });
            }

            Ok(project_transaction)
        })
    }
}

impl BufferStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_message_handler(Self::handle_buffer_reloaded);
        client.add_entity_message_handler(Self::handle_buffer_saved);
        client.add_entity_message_handler(Self::handle_update_buffer_file);
        client.add_entity_request_handler(Self::handle_save_buffer);
        client.add_entity_request_handler(Self::handle_reload_buffers);
    }

    /// Creates a buffer store, optionally retaining its buffers.
    pub fn local(worktree_store: Entity<WorktreeStore>, cx: &mut Context<Self>) -> Self {
        Self {
            state: BufferStoreState::Local(LocalBufferStore {
                local_buffer_ids_by_entry_id: Default::default(),
                worktree_store: worktree_store.clone(),
                _subscription: cx.subscribe(&worktree_store, |this, _, event, cx| {
                    if let WorktreeStoreEvent::WorktreeAdded(worktree) = event {
                        let this = this.as_local_mut().unwrap();
                        this.subscribe_to_worktree(worktree, cx);
                    }
                }),
            }),
            downstream_client: None,
            opened_buffers: Default::default(),
            path_to_buffer_id: Default::default(),
            shared_buffers: Default::default(),
            pending_synchronizations: Default::default(),
            #[cfg(any(test, feature = "test-support"))]
            test_synchronization_preparation: None,
            #[cfg(any(test, feature = "test-support"))]
            test_synchronization_commit_observer: None,
            #[cfg(any(test, feature = "test-support"))]
            test_peer_save_wait: None,
            loading_buffers: Default::default(),
            non_searchable_buffers: Default::default(),
            worktree_store,
            project_search: Default::default(),
        }
    }

    pub fn remote(
        worktree_store: Entity<WorktreeStore>,
        upstream_client: AnyProtoClient,
        remote_id: u64,
        _cx: &mut Context<Self>,
    ) -> Self {
        Self {
            state: BufferStoreState::Remote(RemoteBufferStore {
                shared_with_me: Default::default(),
                loading_remote_buffers_by_id: Default::default(),
                remote_buffer_listeners: Default::default(),
                project_id: remote_id,
                upstream_client,
                worktree_store: worktree_store.clone(),
            }),
            downstream_client: None,
            opened_buffers: Default::default(),
            path_to_buffer_id: Default::default(),
            loading_buffers: Default::default(),
            shared_buffers: Default::default(),
            pending_synchronizations: Default::default(),
            #[cfg(any(test, feature = "test-support"))]
            test_synchronization_preparation: None,
            #[cfg(any(test, feature = "test-support"))]
            test_synchronization_commit_observer: None,
            #[cfg(any(test, feature = "test-support"))]
            test_peer_save_wait: None,
            non_searchable_buffers: Default::default(),
            worktree_store,
            project_search: Default::default(),
        }
    }

    fn as_local_mut(&mut self) -> Option<&mut LocalBufferStore> {
        match &mut self.state {
            BufferStoreState::Local(state) => Some(state),
            _ => None,
        }
    }

    fn as_remote_mut(&mut self) -> Option<&mut RemoteBufferStore> {
        match &mut self.state {
            BufferStoreState::Remote(state) => Some(state),
            _ => None,
        }
    }

    fn as_remote(&self) -> Option<&RemoteBufferStore> {
        match &self.state {
            BufferStoreState::Remote(state) => Some(state),
            _ => None,
        }
    }

    #[ztracing::instrument(skip_all)]
    pub fn open_buffer(
        &mut self,
        project_path: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Buffer>>> {
        self.open_buffer_internal(project_path, None, cx)
    }

    pub fn open_buffer_with_limit(
        &mut self,
        project_path: ProjectPath,
        max_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Buffer>>> {
        if max_bytes == 0 || max_bytes as u64 > task::MAX_TASK_ARTIFACT_BYTES {
            return Task::ready(Err(anyhow!("Invalid artifact byte limit")));
        }
        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(project_path.worktree_id, cx)
        else {
            return Task::ready(Err(anyhow!("no such worktree")));
        };
        if !worktree.read(cx).is_visible() {
            return Task::ready(Err(anyhow!("Artifact worktree is not visible")));
        }
        let path = project_path.path.clone();
        if let BufferStoreState::Remote(remote) = &self.state {
            let client = remote.upstream_client.clone();
            let project_id = remote.project_id;
            let retained_buffer =
                Arc::new(parking_lot::Mutex::new(self.get_by_path(&project_path)));
            let subscription = cx.subscribe(&cx.entity(), {
                let path = project_path.clone();
                let retained_buffer = retained_buffer.clone();
                move |_, _, event, cx| {
                    if let BufferStoreEvent::BufferAdded(buffer) = event
                        && File::from_dyn(buffer.read(cx).file()).is_some_and(|file| {
                            file.worktree_id(cx) == path.worktree_id && file.path == path.path
                        })
                    {
                        *retained_buffer.lock() = Some(buffer.clone());
                    }
                }
            });
            return cx.spawn(async move |this, cx| {
                let _subscription = subscription;
                let response = client
                    .request(proto::OpenBoundedBufferByPath {
                        project_id,
                        worktree_id: project_path.worktree_id.to_proto(),
                        path: path.as_unix_str().to_owned(),
                        max_bytes: max_bytes as u64,
                    })
                    .await?;
                worktree
                    .update(cx, |worktree, _| {
                        worktree.wait_for_snapshot(response.worktree_scan_id as usize)
                    })
                    .await?;
                let buffer_id = BufferId::new(response.buffer_id)?;
                let retained = retained_buffer.lock().take();
                let retained = retained.filter(|buffer| {
                    buffer.read_with(cx, |buffer, _| buffer.remote_id()) == buffer_id
                });
                let buffer = match retained {
                    Some(buffer) => buffer,
                    None => {
                        this.update(cx, |this, cx| this.wait_for_remote_buffer(buffer_id, cx))?
                            .await?
                    }
                };
                buffer.read_with(cx, |buffer, _| {
                    anyhow::ensure!(
                        !buffer.is_dirty() && buffer.len() <= max_bytes,
                        "Remote artifact buffer is modified or oversized"
                    );
                    anyhow::Ok(())
                })?;
                Ok(buffer)
            });
        }
        let loading = self.open_buffer_internal(project_path, Some(max_bytes), cx);
        cx.spawn(async move |_, cx| {
            let buffer = loading.await?;
            let mut loaded = worktree
                .update(cx, |worktree, cx| {
                    worktree.load_file_with_limit(&path, max_bytes, cx)
                })
                .await?;
            let snapshot = buffer.read_with(cx, |buffer, _| {
                anyhow::ensure!(
                    !buffer.is_dirty() && buffer.len() <= max_bytes,
                    "Artifact buffer is modified or exceeds the declared byte limit"
                );
                anyhow::Ok(buffer.snapshot())
            })?;
            let version = snapshot.version().clone();
            cx.background_spawn(async move {
                LineEnding::normalize(&mut loaded.text);
                anyhow::ensure!(
                    snapshot.text() == loaded.text,
                    "Artifact buffer differs from the current host file"
                );
                anyhow::Ok(())
            })
            .await?;
            buffer.read_with(cx, |buffer, _| {
                anyhow::ensure!(
                    !buffer.is_dirty() && buffer.version() == version,
                    "Artifact buffer changed during admission"
                );
                anyhow::Ok(())
            })?;
            Ok(buffer)
        })
    }

    fn open_buffer_internal(
        &mut self,
        project_path: ProjectPath,
        max_bytes: Option<usize>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Buffer>>> {
        if let Some(buffer) = self.get_by_path(&project_path) {
            return Task::ready(Ok(buffer));
        }

        let task = match self.loading_buffers.entry(project_path.clone()) {
            hash_map::Entry::Occupied(e) => e.get().clone(),
            hash_map::Entry::Vacant(entry) => {
                let path = project_path.path.clone();
                let Some(worktree) = self
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(project_path.worktree_id, cx)
                else {
                    return Task::ready(Err(anyhow!("no such worktree")));
                };
                let load_buffer = match &self.state {
                    BufferStoreState::Local(this) => {
                        this.open_buffer_internal(path, worktree, max_bytes, cx)
                    }
                    BufferStoreState::Remote(this) => this.open_buffer(path, worktree, cx),
                };

                entry
                    .insert(
                        cx.spawn(async move |this, cx| {
                            let load_result = load_buffer.await;
                            this.update(cx, |this, _cx| {
                                // Record the fact that the buffer is no longer loading.
                                this.loading_buffers.remove(&project_path);

                                let buffer = load_result.map_err(Arc::new)?;
                                Ok(buffer)
                            })?
                        })
                        .shared(),
                    )
                    .clone()
            }
        };

        cx.background_spawn(async move {
            task.await.map_err(|e| {
                if e.error_code() != ErrorCode::Internal {
                    anyhow!(e.error_code())
                } else {
                    anyhow!("{e}")
                }
            })
        })
    }

    pub fn open_bounded_buffer_for_peer(
        &mut self,
        path: ProjectPath,
        max_bytes: usize,
        peer_id: proto::PeerId,
        cx: &mut Context<Self>,
    ) -> Task<Result<proto::OpenBoundedBufferResponse>> {
        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(path.worktree_id, cx)
        else {
            return Task::ready(Err(anyhow!("no such worktree")));
        };
        let buffer = self.open_buffer_with_limit(path, max_bytes, cx);
        cx.spawn(async move |this, cx| {
            let buffer = buffer.await?;
            this.update(cx, |this, cx| {
                this.create_bounded_buffer_for_peer(&buffer, peer_id, max_bytes, cx)
            })?
            .await?;
            Ok(proto::OpenBoundedBufferResponse {
                buffer_id: buffer.read_with(cx, |buffer, _| buffer.remote_id().to_proto()),
                worktree_scan_id: worktree.read_with(cx, |worktree, _| worktree.scan_id() as u64),
            })
        })
    }

    pub fn create_buffer(
        &mut self,
        language: Option<Arc<Language>>,
        project_searchable: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Buffer>>> {
        match &self.state {
            BufferStoreState::Local(this) => this.create_buffer(language, project_searchable, cx),
            BufferStoreState::Remote(this) => this.create_buffer(language, project_searchable, cx),
        }
    }

    pub fn save_buffer(
        &mut self,
        buffer: Entity<Buffer>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        match &mut self.state {
            BufferStoreState::Local(this) => this.save_buffer(buffer, cx),
            BufferStoreState::Remote(this) => this.save_remote_buffer(buffer, None, cx),
        }
    }

    pub fn save_buffer_as(
        &mut self,
        buffer: Entity<Buffer>,
        path: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let old_file = buffer.read(cx).file().cloned();
        let task = match &self.state {
            BufferStoreState::Local(this) => this.save_buffer_as(buffer.clone(), path, cx),
            BufferStoreState::Remote(this) => {
                this.save_remote_buffer(buffer.clone(), Some(path.to_proto()), cx)
            }
        };
        cx.spawn(async move |this, cx| {
            task.await?;
            this.update(cx, |this, cx| {
                old_file.clone().and_then(|file| {
                    this.path_to_buffer_id.remove(&ProjectPath {
                        worktree_id: file.worktree_id(cx),
                        path: file.path().clone(),
                    })
                });

                cx.emit(BufferStoreEvent::BufferChangedFilePath { buffer, old_file });
            })
        })
    }

    fn add_buffer(&mut self, buffer_entity: Entity<Buffer>, cx: &mut Context<Self>) -> Result<()> {
        let buffer = buffer_entity.read(cx);
        let remote_id = buffer.remote_id();
        let path = File::from_dyn(buffer.file()).map(|file| ProjectPath {
            path: file.path.clone(),
            worktree_id: file.worktree_id(cx),
        });
        let is_remote = buffer.replica_id().is_remote();
        let open_buffer = OpenBuffer::Complete {
            buffer: buffer_entity.downgrade(),
        };

        let handle = cx.entity().downgrade();
        buffer_entity.update(cx, move |_, cx| {
            cx.on_release(move |buffer, cx| {
                handle
                    .update(cx, |_, cx| {
                        cx.emit(BufferStoreEvent::BufferDropped(buffer.remote_id()))
                    })
                    .ok();
            })
            .detach()
        });
        let _expect_path_to_exist;
        match self.opened_buffers.entry(remote_id) {
            hash_map::Entry::Vacant(entry) => {
                entry.insert(open_buffer);
                _expect_path_to_exist = false;
            }
            hash_map::Entry::Occupied(mut entry) => {
                if let OpenBuffer::Operations(operations) = entry.get_mut() {
                    buffer_entity.update(cx, |b, cx| b.apply_ops(operations.drain(..), cx));
                } else if entry.get().upgrade().is_some() {
                    if is_remote {
                        return Ok(());
                    } else {
                        debug_panic!("buffer {remote_id} was already registered");
                        anyhow::bail!("buffer {remote_id} was already registered");
                    }
                }
                entry.insert(open_buffer);
                _expect_path_to_exist = true;
            }
        }

        if let Some(path) = path {
            self.path_to_buffer_id.insert(path, remote_id);
        }

        cx.subscribe(&buffer_entity, Self::on_buffer_event).detach();
        cx.emit(BufferStoreEvent::BufferAdded(buffer_entity));
        Ok(())
    }

    pub fn buffers(&self) -> impl '_ + Iterator<Item = Entity<Buffer>> {
        self.opened_buffers
            .values()
            .filter_map(|buffer| buffer.upgrade())
    }

    pub(crate) fn is_searchable(&self, id: &BufferId) -> bool {
        !self.non_searchable_buffers.contains(&id)
    }

    pub fn loading_buffers(
        &self,
    ) -> impl Iterator<Item = (&ProjectPath, impl Future<Output = Result<Entity<Buffer>>>)> {
        self.loading_buffers.iter().map(|(path, task)| {
            let task = task.clone();
            (path, async move {
                task.await.map_err(|e| {
                    if e.error_code() != ErrorCode::Internal {
                        anyhow!(e.error_code())
                    } else {
                        anyhow!("{e}")
                    }
                })
            })
        })
    }

    pub fn buffer_id_for_project_path(&self, project_path: &ProjectPath) -> Option<&BufferId> {
        self.path_to_buffer_id.get(project_path)
    }

    pub fn get_by_path(&self, path: &ProjectPath) -> Option<Entity<Buffer>> {
        self.path_to_buffer_id
            .get(path)
            .and_then(|buffer_id| self.get(*buffer_id))
    }

    pub fn get(&self, buffer_id: BufferId) -> Option<Entity<Buffer>> {
        self.opened_buffers.get(&buffer_id)?.upgrade()
    }

    pub fn get_existing(&self, buffer_id: BufferId) -> Result<Entity<Buffer>> {
        self.get(buffer_id)
            .with_context(|| format!("unknown buffer id {buffer_id}"))
    }

    pub fn get_possibly_incomplete(&self, buffer_id: BufferId) -> Option<Entity<Buffer>> {
        self.get(buffer_id).or_else(|| {
            self.as_remote()
                .and_then(|remote| remote.loading_remote_buffers_by_id.get(&buffer_id).cloned())
        })
    }

    pub fn buffer_version_info(&self, cx: &App) -> (Vec<proto::BufferVersion>, Vec<BufferId>) {
        let buffers = self
            .buffers()
            .map(|buffer| {
                let buffer = buffer.read(cx);
                proto::BufferVersion {
                    id: buffer.remote_id().into(),
                    version: language::proto::serialize_version(&buffer.version),
                }
            })
            .collect();
        let incomplete_buffer_ids = self
            .as_remote()
            .map(|remote| remote.incomplete_buffer_ids())
            .unwrap_or_default();
        (buffers, incomplete_buffer_ids)
    }

    pub fn disconnected_from_host(&mut self, cx: &mut App) {
        for open_buffer in self.opened_buffers.values_mut() {
            if let Some(buffer) = open_buffer.upgrade() {
                buffer.update(cx, |buffer, _| buffer.give_up_waiting());
            }
        }

        for buffer in self.buffers() {
            buffer.update(cx, |buffer, cx| {
                buffer.set_capability(Capability::ReadOnly, cx)
            });
        }

        if let Some(remote) = self.as_remote_mut() {
            // Wake up all futures currently waiting on a buffer to get opened,
            // to give them a chance to fail now that we've disconnected.
            remote.remote_buffer_listeners.clear()
        }
    }

    pub fn shared(&mut self, remote_id: u64, downstream_client: AnyProtoClient, _cx: &mut App) {
        if self
            .downstream_client
            .as_ref()
            .is_none_or(|(current, current_id)| {
                *current_id != remote_id || !current.is_same_instance(&downstream_client)
            })
        {
            self.forget_shared_buffers();
        } else {
            self.pending_synchronizations.clear();
        }
        self.downstream_client = Some((downstream_client, remote_id));
    }

    pub fn unshared(&mut self, _cx: &mut Context<Self>) {
        self.downstream_client.take();
        self.forget_shared_buffers();
    }

    pub fn discard_incomplete(&mut self) {
        self.opened_buffers
            .retain(|_, buffer| !matches!(buffer, OpenBuffer::Operations(_)));
    }

    fn buffer_changed_file(&mut self, buffer: Entity<Buffer>, cx: &mut App) -> Option<()> {
        let file = File::from_dyn(buffer.read(cx).file())?;

        let remote_id = buffer.read(cx).remote_id();
        if let Some(entry_id) = file.entry_id {
            if let Some(local) = self.as_local_mut() {
                match local.local_buffer_ids_by_entry_id.get(&entry_id) {
                    Some(_) => {
                        return None;
                    }
                    None => {
                        local
                            .local_buffer_ids_by_entry_id
                            .insert(entry_id, remote_id);
                    }
                }
            }
            self.path_to_buffer_id.insert(
                ProjectPath {
                    worktree_id: file.worktree_id(cx),
                    path: file.path.clone(),
                },
                remote_id,
            );
        };

        Some(())
    }

    fn on_buffer_event(
        &mut self,
        buffer: Entity<Buffer>,
        event: &BufferEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            BufferEvent::FileHandleChanged => {
                self.buffer_changed_file(buffer, cx);
            }
            BufferEvent::Reloaded => {
                let Some((downstream_client, project_id)) = self.downstream_client.as_ref() else {
                    return;
                };
                let buffer = buffer.read(cx);
                downstream_client
                    .send(proto::BufferReloaded {
                        project_id: *project_id,
                        buffer_id: buffer.remote_id().to_proto(),
                        version: serialize_version(&buffer.version()),
                        mtime: buffer.saved_mtime().map(|t| t.into()),
                        line_ending: serialize_line_ending(buffer.line_ending()) as i32,
                    })
                    .log_err();
            }
            BufferEvent::LanguageChanged(_) => {}
            _ => {}
        }
    }

    pub async fn handle_update_buffer_from_owner(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateBuffer>,
        cx: AsyncApp,
    ) -> Result<proto::Ack> {
        Self::apply_update_buffer(this, envelope, None, cx).await
    }

    pub(crate) async fn handle_update_buffer_from_peer(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateBuffer>,
        cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let project_id = envelope.payload.project_id;
        Self::apply_update_buffer(this, envelope, Some(project_id), cx).await
    }

    async fn apply_update_buffer(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateBuffer>,
        peer_project_id: Option<u64>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let payload = envelope.payload;
        let buffer_id = BufferId::new(payload.buffer_id)?;
        let operations = payload
            .operations
            .into_iter()
            .map(language::proto::deserialize_operation)
            .collect::<Result<Vec<_>, _>>()?;
        this.update(&mut cx, |this, cx| {
            if let Some(project_id) = peer_project_id {
                let buffer = this.ensure_peer_buffer_mutation(buffer_id, project_id, cx)?;
                buffer.update(cx, |buffer, cx| buffer.apply_ops(operations, cx));
            } else {
                match this.opened_buffers.entry(buffer_id) {
                    hash_map::Entry::Occupied(mut entry) => match entry.get_mut() {
                        OpenBuffer::Operations(queued) => queued.extend_from_slice(&operations),
                        OpenBuffer::Complete { buffer, .. } => {
                            if let Some(buffer) = buffer.upgrade() {
                                buffer.update(cx, |buffer, cx| buffer.apply_ops(operations, cx));
                            }
                        }
                    },
                    hash_map::Entry::Vacant(entry) => {
                        entry.insert(OpenBuffer::Operations(operations));
                    }
                }
            }
            Ok(proto::Ack {})
        })
    }

    pub fn register_shared_lsp_handle(
        &mut self,
        peer_id: proto::PeerId,
        buffer_id: BufferId,
        handle: OpenLspBufferHandle,
    ) {
        if let Some(shared_buffers) = self.shared_buffers.get_mut(&peer_id)
            && let Some(buffer) = shared_buffers.get_mut(&buffer_id)
        {
            buffer.lsp_handle = Some(handle);
            return;
        }
        debug_panic!("tried to register shared lsp handle, but buffer was not shared")
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn hold_synchronization_preparation_for_test(
        &mut self,
        project_id: u64,
        peer_id: proto::PeerId,
        buffer_id: u64,
    ) -> Result<(
        oneshot::Receiver<TestSynchronizationPrepared>,
        oneshot::Sender<()>,
    )> {
        anyhow::ensure!(
            self.test_synchronization_preparation.is_none(),
            "preparation probe already armed"
        );
        let (observed, receiver) = oneshot::channel();
        let (release, wait) = oneshot::channel();
        self.test_synchronization_preparation = Some(TestSynchronizationPreparationGate {
            project_id,
            peer_id,
            buffer_id,
            observed,
            release: wait,
        });
        Ok((receiver, release))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn has_live_synchronization_for_test(&self, peer_id: proto::PeerId) -> bool {
        self.pending_synchronizations
            .get(&peer_id)
            .and_then(std::sync::Weak::upgrade)
            .is_some()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn observe_synchronization_commit_for_test(
        &mut self,
        project_id: u64,
        peer_id: proto::PeerId,
    ) -> Result<oneshot::Receiver<proto::SynchronizeBuffersResponse>> {
        anyhow::ensure!(
            self.test_synchronization_commit_observer.is_none(),
            "commit observer already armed"
        );
        anyhow::ensure!(
            self.downstream_client
                .as_ref()
                .is_some_and(|(client, current_project)| {
                    client.is_via_collab() && *current_project == project_id
                }),
            ErrorCode::Disconnected
        );
        let (observed, receiver) = oneshot::channel();
        self.test_synchronization_commit_observer = Some(TestSynchronizationCommitObserver {
            project_id,
            peer_id,
            observed,
        });
        Ok(receiver)
    }

    pub(crate) fn handle_synchronize_buffers(
        &mut self,
        envelope: TypedEnvelope<proto::SynchronizeBuffers>,
        cx: &mut Context<Self>,
        client: Arc<Client>,
    ) -> Task<Result<PreparedSynchronization>> {
        let preflight = (|| -> Result<_> {
            let peer_id = envelope.original_sender_id()?;
            let project_id = envelope.payload.project_id;
            let requested_buffers = envelope
                .payload
                .buffers
                .into_iter()
                .map(|buffer| {
                    Ok((
                        BufferId::new(buffer.id)?,
                        deserialize_version(&buffer.version),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let (downstream_client, current_project_id) = self
                .downstream_client
                .as_ref()
                .context("Buffer synchronization transport is unavailable")?;
            anyhow::ensure!(
                *current_project_id == project_id,
                "Buffer synchronization project changed"
            );
            let publishing_to_collaboration = downstream_client.is_via_collab();
            anyhow::ensure!(
                publishing_to_collaboration == rpc::ProtoClient::is_via_collab(client.as_ref()),
                "Buffer synchronization transport role mismatch"
            );
            let client_status = *client.status().borrow();
            anyhow::ensure!(
                matches!(client_status, ClientStatus::Connected { .. }),
                ErrorCode::Disconnected
            );
            let token = Arc::new(());
            self.pending_synchronizations
                .insert(peer_id, Arc::downgrade(&token));
            Ok((
                BufferSynchronizationRequest {
                    project_id,
                    peer_id,
                    publishing_to_collaboration,
                    client,
                    client_status,
                    token,
                },
                requested_buffers,
            ))
        })();
        let (request, requested_buffers) = match preflight {
            Ok(preflight) => preflight,
            Err(error) => return Task::ready(Err(error)),
        };

        cx.spawn(async move |this, cx| {
            let mut completed = Vec::new();
            'buffers: for (buffer_id, remote_version) in requested_buffers {
                let preparation = this.update(cx, |this, cx| -> Result<_> {
                    this.ensure_synchronization_request_current(&request)?;
                    let Some(buffer) = this.get(buffer_id) else {
                        return Ok(None);
                    };
                    let current = buffer.read(cx);
                    match this.ensure_buffer_shareable_with_peer(current, cx) {
                        Ok(()) => {}
                        Err(error) if error.error_code() == ErrorCode::UnsharedItem => {
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    }
                    let synchronized = SynchronizedBuffer {
                        buffer_id,
                        buffer: buffer.clone(),
                        version: proto::BufferVersion {
                            id: buffer_id.to_proto(),
                            version: serialize_version(&current.version()),
                        },
                        file_binding: current
                            .file()
                            .map(|file| ProjectPath::from_file(file.as_ref(), cx)),
                    };
                    let operations = current.serialize_ops(Some(remote_version), cx);
                    Ok(Some((synchronized, operations)))
                })??;
                let Some((synchronized, operations)) = preparation else {
                    continue;
                };
                #[cfg(any(test, feature = "test-support"))]
                let gate = this.update(cx, |this, _| {
                    this.test_synchronization_preparation.take_if(|gate| {
                        gate.project_id == request.project_id
                            && gate.peer_id == request.peer_id
                            && gate.buffer_id == synchronized.version.id
                    })
                })?;
                #[cfg(any(test, feature = "test-support"))]
                let checkpoint = TestSynchronizationPrepared {
                    project_id: request.project_id,
                    peer_id: request.peer_id,
                    version: synchronized.version.clone(),
                    chunks: Vec::new(),
                };
                let chunks = cx
                    .background_spawn(async move {
                        let chunks = split_operations(operations.await).collect::<Vec<_>>();
                        #[cfg(any(test, feature = "test-support"))]
                        if let Some(gate) = gate {
                            let mut checkpoint = checkpoint;
                            checkpoint.chunks = chunks.clone();
                            gate.observed.send(checkpoint).map_err(|_| {
                                anyhow!("synchronization preparation observer dropped")
                            })?;
                            gate.release
                                .await
                                .context("synchronization preparation gate dropped")?;
                        }
                        anyhow::Ok(chunks)
                    })
                    .await?;
                let sent_metadata = this.update(cx, |this, cx| -> Result<bool> {
                    if !this.synchronization_buffer_is_shareable(&request, &synchronized, cx)? {
                        return Ok(false);
                    }
                    let current = synchronized.buffer.read(cx);
                    if let Some(file) = current.file() {
                        request.client.send(proto::UpdateBufferFile {
                            project_id: request.project_id,
                            buffer_id: synchronized.buffer_id.to_proto(),
                            file: Some(file.to_proto(cx)),
                        })?;
                    }
                    request.client.send(proto::BufferReloaded {
                        project_id: request.project_id,
                        buffer_id: synchronized.buffer_id.to_proto(),
                        version: serialize_version(current.saved_version()),
                        mtime: current.saved_mtime().map(Into::into),
                        line_ending: serialize_line_ending(current.line_ending()) as i32,
                    })?;
                    Ok(true)
                })??;
                if !sent_metadata {
                    continue;
                }
                for chunk in chunks {
                    let send = this.update(cx, |this, cx| -> Result<_> {
                        if !this.synchronization_buffer_is_shareable(&request, &synchronized, cx)? {
                            return Ok(None);
                        }
                        // Client::request enqueues before its future is polled.
                        Ok(Some(request.client.request(proto::UpdateBuffer {
                            project_id: request.project_id,
                            buffer_id: synchronized.version.id,
                            operations: chunk,
                        })))
                    })??;
                    let Some(send) = send else {
                        continue 'buffers;
                    };
                    send.await?;
                    let still_shareable = this.update(cx, |this, cx| {
                        this.synchronization_buffer_is_shareable(&request, &synchronized, cx)
                    })??;
                    if !still_shareable {
                        continue 'buffers;
                    }
                }
                completed.push(synchronized);
            }
            Ok(PreparedSynchronization {
                request,
                buffers: completed,
            })
        })
    }

    fn ensure_synchronization_request_current(
        &self,
        request: &BufferSynchronizationRequest,
    ) -> Result<()> {
        let (client, project_id) = self
            .downstream_client
            .as_ref()
            .context("Buffer synchronization transport is unavailable")?;
        anyhow::ensure!(
            *project_id == request.project_id
                && client.is_via_collab() == request.publishing_to_collaboration,
            "Buffer synchronization scope changed"
        );
        anyhow::ensure!(
            matches!(request.client_status, ClientStatus::Connected { .. })
                && *request.client.status().borrow() == request.client_status,
            ErrorCode::Disconnected
        );
        anyhow::ensure!(
            self.pending_synchronizations
                .get(&request.peer_id)
                .is_some_and(|current| current.ptr_eq(&Arc::downgrade(&request.token))),
            "Buffer synchronization request was superseded"
        );
        Ok(())
    }

    fn synchronization_buffer_is_shareable(
        &self,
        request: &BufferSynchronizationRequest,
        synchronized: &SynchronizedBuffer,
        cx: &App,
    ) -> Result<bool> {
        self.ensure_synchronization_request_current(request)?;
        let current = self
            .get(synchronized.buffer_id)
            .context("Buffer closed during synchronization")?;
        anyhow::ensure!(
            current == synchronized.buffer,
            "Buffer changed during synchronization"
        );
        let current = current.read(cx);
        anyhow::ensure!(
            current
                .file()
                .map(|file| ProjectPath::from_file(file.as_ref(), cx))
                == synchronized.file_binding,
            "Buffer file changed during synchronization"
        );
        match self.ensure_buffer_shareable_with_peer(current, cx) {
            Ok(()) => Ok(true),
            Err(error) if error.error_code() == ErrorCode::UnsharedItem => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn commit_synchronized_buffers(
        &mut self,
        prepared: PreparedSynchronization,
        client: &Arc<Client>,
        cx: &App,
    ) -> Result<proto::SynchronizeBuffersResponse> {
        anyhow::ensure!(
            Arc::ptr_eq(client, &prepared.request.client),
            "Buffer synchronization client changed"
        );
        self.ensure_synchronization_request_current(&prepared.request)?;
        let mut permitted = Vec::new();
        for synchronized in prepared.buffers {
            if self.synchronization_buffer_is_shareable(&prepared.request, &synchronized, cx)? {
                permitted.push(synchronized);
            }
        }
        let mut response = proto::SynchronizeBuffersResponse {
            buffers: Vec::with_capacity(permitted.len()),
        };
        for synchronized in permitted {
            self.shared_buffers
                .entry(prepared.request.peer_id)
                .or_default()
                .entry(synchronized.buffer_id)
                .or_insert_with(|| SharedBuffer {
                    buffer: synchronized.buffer,
                    lsp_handle: None,
                });
            response.buffers.push(synchronized.version);
        }
        self.pending_synchronizations
            .remove(&prepared.request.peer_id);
        #[cfg(any(test, feature = "test-support"))]
        if let Some(observer) = self
            .test_synchronization_commit_observer
            .take_if(|observer| {
                observer.project_id == prepared.request.project_id
                    && observer.peer_id == prepared.request.peer_id
            })
        {
            observer
                .observed
                .send(response.clone())
                .map_err(|_| anyhow!("synchronization commit observer dropped"))?;
        }
        Ok(response)
    }

    pub fn handle_create_buffer_for_peer(
        &mut self,
        envelope: TypedEnvelope<proto::CreateBufferForPeer>,
        replica_id: ReplicaId,
        capability: Capability,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let remote = self
            .as_remote_mut()
            .context("buffer store is not a remote")?;

        if let Some(buffer) =
            remote.handle_create_buffer_for_peer(envelope, replica_id, capability, cx)?
        {
            self.add_buffer(buffer, cx)?;
        }

        Ok(())
    }

    pub async fn handle_update_buffer_file(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateBufferFile>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let buffer_id = envelope.payload.buffer_id;
        let buffer_id = BufferId::new(buffer_id)?;

        this.update(&mut cx, |this, cx| {
            let payload = envelope.payload.clone();
            if let Some(buffer) = this.get_possibly_incomplete(buffer_id) {
                let file = payload.file.context("invalid file")?;
                let worktree = this
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(WorktreeId::from_proto(file.worktree_id), cx)
                    .context("no such worktree")?;
                let file = File::from_proto(file, worktree, cx)?;
                let old_file = buffer.update(cx, |buffer, cx| {
                    let old_file = buffer.file().cloned();
                    let new_path = file.path.clone();

                    buffer.file_updated(Arc::new(file), cx);
                    if old_file.as_ref().is_none_or(|old| *old.path() != new_path) {
                        Some(old_file)
                    } else {
                        None
                    }
                });
                if let Some(old_file) = old_file {
                    cx.emit(BufferStoreEvent::BufferChangedFilePath { buffer, old_file });
                }
            }
            if let Some((downstream_client, project_id)) = this.downstream_client.as_ref() {
                downstream_client
                    .send(proto::UpdateBufferFile {
                        project_id: *project_id,
                        buffer_id: buffer_id.into(),
                        file: envelope.payload.file,
                    })
                    .log_err();
            }
            Ok(())
        })
    }

    pub async fn handle_save_buffer(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::SaveBuffer>,
        mut cx: AsyncApp,
    ) -> Result<proto::BufferSaved> {
        let buffer_id = BufferId::new(envelope.payload.buffer_id)?;
        let new_path = envelope
            .payload
            .new_path
            .map(|path| ProjectPath::from_proto(path).context("invalid save destination"))
            .transpose()?;
        let (buffer, project_id, from_peer) = this.read_with(&cx, |this, cx| {
            let (client, project_id) = this
                .downstream_client
                .as_ref()
                .context("project is not shared")?;
            anyhow::ensure!(
                *project_id == envelope.payload.project_id,
                ErrorCode::UnsharedItem
            );
            let from_peer = client.is_via_collab();
            let buffer = if from_peer {
                let buffer = this.ensure_peer_buffer_mutation(buffer_id, *project_id, cx)?;
                if let Some(path) = &new_path {
                    this.ensure_peer_save_destination(path, cx)?;
                }
                buffer
            } else {
                this.get_existing(buffer_id)?
            };
            anyhow::Ok((buffer, *project_id, from_peer))
        })?;
        let wait_for_version = buffer.update(&mut cx, |buffer, _| {
            buffer.wait_for_version(deserialize_version(&envelope.payload.version))
        });
        #[cfg(any(test, feature = "test-support"))]
        if from_peer {
            this.update(&mut cx, |this, _| {
                if let Some(observer) = this.test_peer_save_wait.take_if(|observer| {
                    observer.project_id == project_id && observer.buffer_id == buffer_id
                }) {
                    observer
                        .observed
                        .send(())
                        .map_err(|_| anyhow!("peer save observer dropped"))?;
                }
                anyhow::Ok(())
            })?;
        }
        wait_for_version.await?;
        this.update(&mut cx, |this, cx| {
            if from_peer {
                let current = this.ensure_peer_buffer_mutation(buffer_id, project_id, cx)?;
                anyhow::ensure!(current == buffer, ErrorCode::UnsharedItem);
                if let Some(path) = &new_path {
                    this.ensure_peer_save_destination(path, cx)?;
                }
            }
            anyhow::Ok(if let Some(path) = new_path {
                this.save_buffer_as(buffer.clone(), path, cx)
            } else {
                this.save_buffer(buffer.clone(), cx)
            })
        })?
        .await?;
        Ok(buffer.read_with(&cx, |buffer, _| proto::BufferSaved {
            project_id,
            buffer_id: buffer_id.into(),
            version: serialize_version(buffer.saved_version()),
            mtime: buffer.saved_mtime().map(|time| time.into()),
        }))
    }

    pub async fn handle_close_buffer(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CloseBuffer>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let peer_id = envelope.sender_id;
        let buffer_id = BufferId::new(envelope.payload.buffer_id)?;
        this.update(&mut cx, |this, cx| {
            if let Some(shared) = this.shared_buffers.get_mut(&peer_id)
                && shared.remove(&buffer_id).is_some()
            {
                cx.emit(BufferStoreEvent::SharedBufferClosed(peer_id, buffer_id));
                if shared.is_empty() {
                    this.shared_buffers.remove(&peer_id);
                }
                return;
            }
            debug_panic!(
                "peer_id {} closed buffer_id {} which was either not open or already closed",
                peer_id,
                buffer_id
            )
        });
        Ok(())
    }

    pub async fn handle_buffer_saved(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::BufferSaved>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let buffer_id = BufferId::new(envelope.payload.buffer_id)?;
        let version = deserialize_version(&envelope.payload.version);
        let mtime = envelope.payload.mtime.clone().map(|time| time.into());
        this.update(&mut cx, move |this, cx| {
            if let Some(buffer) = this.get_possibly_incomplete(buffer_id) {
                buffer.update(cx, |buffer, cx| {
                    buffer.did_save(version, mtime, cx);
                });
            }

            if let Some((downstream_client, project_id)) = this.downstream_client.as_ref() {
                downstream_client
                    .send(proto::BufferSaved {
                        project_id: *project_id,
                        buffer_id: buffer_id.into(),
                        mtime: envelope.payload.mtime,
                        version: envelope.payload.version,
                    })
                    .log_err();
            }
        });
        Ok(())
    }

    pub async fn handle_buffer_reloaded(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::BufferReloaded>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let buffer_id = BufferId::new(envelope.payload.buffer_id)?;
        let version = deserialize_version(&envelope.payload.version);
        let mtime = envelope.payload.mtime.clone().map(|time| time.into());
        let line_ending = deserialize_line_ending(
            proto::LineEnding::from_i32(envelope.payload.line_ending)
                .context("missing line ending")?,
        );
        this.update(&mut cx, |this, cx| {
            if let Some(buffer) = this.get_possibly_incomplete(buffer_id) {
                buffer.update(cx, |buffer, cx| {
                    buffer.did_reload(version, line_ending, mtime, cx);
                });
            }

            if let Some((downstream_client, project_id)) = this.downstream_client.as_ref() {
                downstream_client
                    .send(proto::BufferReloaded {
                        project_id: *project_id,
                        buffer_id: buffer_id.into(),
                        mtime: envelope.payload.mtime,
                        version: envelope.payload.version,
                        line_ending: envelope.payload.line_ending,
                    })
                    .log_err();
            }
        });
        Ok(())
    }

    pub fn reload_buffers(
        &self,
        buffers: HashSet<Entity<Buffer>>,
        push_to_history: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<ProjectTransaction>> {
        if buffers.is_empty() {
            return Task::ready(Ok(ProjectTransaction::default()));
        }
        match &self.state {
            BufferStoreState::Local(this) => this.reload_buffers(buffers, push_to_history, cx),
            BufferStoreState::Remote(this) => this.reload_buffers(buffers, push_to_history, cx),
        }
    }

    async fn handle_reload_buffers(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ReloadBuffers>,
        mut cx: AsyncApp,
    ) -> Result<proto::ReloadBuffersResponse> {
        let sender_id = envelope.original_sender_id().unwrap_or_default();
        let reload = this.update(&mut cx, |this, cx| {
            let mut buffers = HashSet::default();
            for buffer_id in &envelope.payload.buffer_ids {
                let buffer_id = BufferId::new(*buffer_id)?;
                buffers.insert(this.get_existing(buffer_id)?);
            }
            anyhow::Ok(this.reload_buffers(buffers, false, cx))
        })?;

        let project_transaction = reload.await?;
        let project_transaction = this
            .update(&mut cx, |this, cx| {
                this.serialize_project_transaction_for_peer(project_transaction, sender_id, cx)
            })
            .await?;
        Ok(proto::ReloadBuffersResponse {
            transaction: Some(project_transaction),
        })
    }

    pub(crate) fn is_buffer_private_for_collaboration(buffer: &Buffer, cx: &App) -> bool {
        let Some(file) = buffer.file() else {
            return false;
        };
        let Some(file) = File::from_dyn(Some(file)) else {
            return file.is_private();
        };
        Self::is_worktree_path_private(file.worktree.read(cx), file.path.as_ref(), cx)
    }

    fn is_worktree_path_private(worktree: &Worktree, path: &RelPath, cx: &App) -> bool {
        let path = if path.is_empty() && worktree.is_single_file() {
            worktree.root_name()
        } else {
            path
        };
        if let Some(local) = worktree.as_local() {
            local.is_path_private(path)
        } else {
            WorktreeSettings::get(
                Some(settings::SettingsLocation {
                    worktree_id: worktree.id(),
                    path: RelPath::empty(),
                }),
                cx,
            )
            .is_path_private(path)
        }
    }

    pub(crate) fn ensure_peer_buffer_mutation(
        &self,
        buffer_id: BufferId,
        project_id: u64,
        cx: &App,
    ) -> Result<Entity<Buffer>> {
        anyhow::ensure!(
            self.downstream_client
                .as_ref()
                .is_some_and(|(client, current)| {
                    client.is_via_collab() && *current == project_id
                }),
            ErrorCode::UnsharedItem
        );
        let buffer = self
            .get_existing(buffer_id)
            .map_err(|error| anyhow::Error::from(ErrorCode::UnsharedItem).context(error))?;
        anyhow::ensure!(
            !Self::is_buffer_private_for_collaboration(buffer.read(cx), cx),
            ErrorCode::UnsharedItem
        );
        Ok(buffer)
    }

    fn ensure_peer_save_destination(&self, path: &ProjectPath, cx: &App) -> Result<()> {
        let worktree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(path.worktree_id, cx)
            .ok_or_else(|| anyhow::Error::from(ErrorCode::UnsharedItem))?;
        anyhow::ensure!(
            !Self::is_worktree_path_private(worktree.read(cx), &path.path, cx),
            ErrorCode::UnsharedItem
        );
        Ok(())
    }

    pub(crate) fn ensure_buffer_shareable_with_peer(
        &self,
        buffer: &Buffer,
        cx: &App,
    ) -> Result<()> {
        let publishing_to_collaboration = self
            .downstream_client
            .as_ref()
            .is_some_and(|(client, _)| client.is_via_collab());
        anyhow::ensure!(
            !publishing_to_collaboration || !Self::is_buffer_private_for_collaboration(buffer, cx),
            ErrorCode::UnsharedItem
        );
        Ok(())
    }

    pub fn create_buffer_for_peer(
        &mut self,
        buffer: &Entity<Buffer>,
        peer_id: proto::PeerId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Err(error) = self.ensure_buffer_shareable_with_peer(buffer.read(cx), cx) {
            return Task::ready(Err(error));
        }
        let buffer_id = buffer.read(cx).remote_id();
        if self
            .shared_buffers
            .get(&peer_id)
            .is_some_and(|buffers| buffers.contains_key(&buffer_id))
        {
            return Task::ready(Ok(()));
        }
        let Some((client, project_id)) = &self.downstream_client else {
            self.shared_buffers.entry(peer_id).or_default().insert(
                buffer_id,
                SharedBuffer {
                    buffer: buffer.clone(),
                    lsp_handle: None,
                },
            );
            return Task::ready(Ok(()));
        };
        let publishing_to_collaboration = client.is_via_collab();
        let originating_client = client.clone();
        let project_id = *project_id;
        let buffer = buffer.clone();
        let operations = buffer.update(cx, |buffer, cx| buffer.serialize_ops(None, cx));
        cx.spawn(async move |this, cx| {
            let chunks = cx
                .background_spawn(
                    async move { split_operations(operations.await).collect::<Vec<_>>() },
                )
                .await;
            let mut admitted = false;
            let result = this.update(cx, |this, cx| {
                this.ensure_buffer_shareable_with_peer(buffer.read(cx), cx)?;
                let (client, current_project_id) = this
                    .downstream_client
                    .clone()
                    .context("Peer buffer transport is no longer available")?;
                anyhow::ensure!(
                    current_project_id == project_id,
                    "Peer buffer project changed during serialization"
                );
                anyhow::ensure!(
                    client.is_same_instance(&originating_client),
                    "Peer buffer client changed during serialization"
                );
                anyhow::ensure!(
                    client.is_via_collab() == publishing_to_collaboration,
                    "Peer buffer transport role changed during serialization"
                );
                if this
                    .shared_buffers
                    .get(&peer_id)
                    .is_some_and(|buffers| buffers.contains_key(&buffer_id))
                {
                    return Ok(());
                }
                let state = buffer.read(cx).to_proto(cx);
                client.send(proto::CreateBufferForPeer {
                    project_id,
                    peer_id: Some(peer_id),
                    variant: Some(proto::create_buffer_for_peer::Variant::State(state)),
                })?;
                this.shared_buffers.entry(peer_id).or_default().insert(
                    buffer_id,
                    SharedBuffer {
                        buffer: buffer.clone(),
                        lsp_handle: None,
                    },
                );
                admitted = true;
                let mut chunks = chunks.into_iter().peekable();
                while let Some(chunk) = chunks.next() {
                    this.ensure_buffer_shareable_with_peer(buffer.read(cx), cx)?;
                    client.send(proto::CreateBufferForPeer {
                        project_id,
                        peer_id: Some(peer_id),
                        variant: Some(proto::create_buffer_for_peer::Variant::Chunk(
                            proto::BufferChunk {
                                buffer_id: buffer_id.into(),
                                operations: chunk,
                                is_last: chunks.peek().is_none(),
                            },
                        )),
                    })?;
                }
                anyhow::Ok(())
            })?;
            if admitted && result.is_err() {
                this.update(cx, |this, _| {
                    if let Some(buffers) = this.shared_buffers.get_mut(&peer_id)
                        && buffers
                            .get(&buffer_id)
                            .is_some_and(|shared| shared.buffer == buffer)
                    {
                        buffers.remove(&buffer_id);
                        if buffers.is_empty() {
                            this.shared_buffers.remove(&peer_id);
                        }
                    }
                })?;
            }
            result
        })
    }

    pub fn forget_shared_buffers(&mut self) {
        self.pending_synchronizations.clear();
        self.shared_buffers.clear();
    }

    pub fn create_bounded_buffer_for_peer(
        &mut self,
        buffer: &Entity<Buffer>,
        peer_id: proto::PeerId,
        max_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let admission = (|| {
            let buffer = buffer.read(cx);
            anyhow::ensure!(
                max_bytes > 0 && max_bytes as u64 <= task::MAX_TASK_ARTIFACT_BYTES,
                "Invalid artifact byte limit"
            );
            let mut content_bytes = buffer.base_text().len();
            anyhow::ensure!(
                content_bytes <= max_bytes,
                "Artifact base text exceeds its limit"
            );
            for (index, (_, operation)) in buffer.operations().iter().enumerate() {
                anyhow::ensure!(index < 4096, "Artifact buffer history is too large");
                if let text::Operation::Edit(edit) = operation {
                    for text in &edit.new_text {
                        content_bytes = content_bytes
                            .checked_add(text.len())
                            .context("Artifact history size overflow")?;
                        anyhow::ensure!(
                            content_bytes <= max_bytes,
                            "Artifact text and retained history exceed the declared byte limit"
                        );
                    }
                }
            }
            anyhow::Ok(())
        })();
        if let Err(error) = admission {
            return Task::ready(Err(error));
        }
        let buffer = buffer.clone();
        let version = buffer.read(cx).version();
        let operations = buffer.read(cx).serialize_ops(None, cx);
        cx.spawn(async move |this, cx| {
            let operations = operations.await;
            let transfer = this.update(cx, |this, cx| {
                let current = buffer.read(cx);
                anyhow::ensure!(
                    !current.is_dirty() && current.version() == version,
                    "Artifact buffer changed before transfer"
                );
                anyhow::ensure!(
                    current.file().is_some_and(|file| !file.is_private()),
                    "Artifact buffer is not shareable"
                );
                anyhow::ensure!(
                    current.len() <= max_bytes && current.base_text().len() <= max_bytes,
                    "Artifact buffer exceeds the declared byte limit"
                );
                let state = current.to_proto(cx);
                validate_bounded_buffer_transfer(&state, &operations, max_bytes)?;
                let buffer_id = current.remote_id();
                let (client, project_id) = this
                    .downstream_client
                    .clone()
                    .context("Artifact peer transport is unavailable")?;
                if this
                    .shared_buffers
                    .get(&peer_id)
                    .is_some_and(|buffers| buffers.contains_key(&buffer_id))
                {
                    return Ok(None);
                }
                client.send(proto::CreateBufferForPeer {
                    project_id,
                    peer_id: Some(peer_id),
                    variant: Some(proto::create_buffer_for_peer::Variant::State(state)),
                })?;
                this.shared_buffers.entry(peer_id).or_default().insert(
                    buffer_id,
                    SharedBuffer {
                        buffer: buffer.clone(),
                        lsp_handle: None,
                    },
                );
                anyhow::Ok(Some((client, project_id, buffer_id)))
            })??;
            if let Some((client, project_id, buffer_id)) = transfer {
                let result = cx
                    .background_spawn(async move {
                        let mut chunks = split_operations(operations).peekable();
                        while let Some(chunk) = chunks.next() {
                            client.send(proto::CreateBufferForPeer {
                                project_id,
                                peer_id: Some(peer_id),
                                variant: Some(proto::create_buffer_for_peer::Variant::Chunk(
                                    proto::BufferChunk {
                                        buffer_id: buffer_id.into(),
                                        operations: chunk,
                                        is_last: chunks.peek().is_none(),
                                    },
                                )),
                            })?;
                        }
                        anyhow::Ok(())
                    })
                    .await;
                if result.is_err() {
                    this.update(cx, |this, _| {
                        if let Some(buffers) = this.shared_buffers.get_mut(&peer_id) {
                            buffers.remove(&buffer_id);
                        }
                    })?;
                }
                result?;
            }
            Ok(())
        })
    }

    pub fn forget_shared_buffers_for(&mut self, peer_id: &proto::PeerId) {
        self.pending_synchronizations.remove(peer_id);
        self.shared_buffers.remove(peer_id);
    }

    pub fn update_peer_id(&mut self, old_peer_id: &proto::PeerId, new_peer_id: proto::PeerId) {
        self.pending_synchronizations.remove(old_peer_id);
        self.pending_synchronizations.remove(&new_peer_id);
        if let Some(buffers) = self.shared_buffers.remove(old_peer_id) {
            self.shared_buffers.insert(new_peer_id, buffers);
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn is_buffer_shared_with_peer(&self, peer_id: proto::PeerId, buffer_id: BufferId) -> bool {
        self.shared_buffers
            .get(&peer_id)
            .is_some_and(|buffers| buffers.contains_key(&buffer_id))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn has_buffer_or_pending_operations(&self, buffer_id: BufferId) -> bool {
        self.opened_buffers.contains_key(&buffer_id)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn queued_operation_count_for_test(&self, buffer_id: BufferId) -> Option<usize> {
        match self.opened_buffers.get(&buffer_id) {
            Some(OpenBuffer::Operations(operations)) => Some(operations.len()),
            _ => None,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn observe_peer_save_wait_for_test(
        &mut self,
        project_id: u64,
        buffer_id: BufferId,
    ) -> Result<oneshot::Receiver<()>> {
        anyhow::ensure!(
            self.test_peer_save_wait.is_none(),
            "peer save observer already armed"
        );
        let (observed, receiver) = oneshot::channel();
        self.test_peer_save_wait = Some(TestPeerSaveWait {
            project_id,
            buffer_id,
            observed,
        });
        Ok(receiver)
    }

    pub fn has_shared_buffers(&self) -> bool {
        !self.shared_buffers.is_empty()
    }

    pub fn create_local_buffer(
        &mut self,
        text: &str,
        language: Option<Arc<Language>>,
        project_searchable: bool,
        cx: &mut Context<Self>,
    ) -> Entity<Buffer> {
        let buffer = cx.new(|cx| {
            let mut buffer = Buffer::local(text, cx)
                .with_language(language.unwrap_or_else(|| language::PLAIN_TEXT.clone()), cx);
            apply_initial_line_ending(&mut buffer, cx);
            buffer
        });

        self.add_buffer(buffer.clone(), cx).log_err();
        let buffer_id = buffer.read(cx).remote_id();
        if !project_searchable {
            self.non_searchable_buffers.insert(buffer_id);
        }

        if let Some(file) = File::from_dyn(buffer.read(cx).file()) {
            self.path_to_buffer_id.insert(
                ProjectPath {
                    worktree_id: file.worktree_id(cx),
                    path: file.path.clone(),
                },
                buffer_id,
            );
            let this = self
                .as_local_mut()
                .expect("local-only method called in a non-local context");
            if let Some(entry_id) = file.entry_id {
                this.local_buffer_ids_by_entry_id
                    .insert(entry_id, buffer_id);
            }
        }
        buffer
    }

    pub fn deserialize_project_transaction(
        &mut self,
        message: proto::ProjectTransaction,
        push_to_history: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<ProjectTransaction>> {
        if let Some(this) = self.as_remote_mut() {
            this.deserialize_project_transaction(message, push_to_history, cx)
        } else {
            debug_panic!("not a remote buffer store");
            Task::ready(Err(anyhow!("not a remote buffer store")))
        }
    }

    pub fn wait_for_remote_buffer(
        &mut self,
        id: BufferId,
        cx: &mut Context<BufferStore>,
    ) -> Task<Result<Entity<Buffer>>> {
        if let Some(this) = self.as_remote_mut() {
            this.wait_for_remote_buffer(id, cx)
        } else {
            debug_panic!("not a remote buffer store");
            Task::ready(Err(anyhow!("not a remote buffer store")))
        }
    }

    fn ensure_peer_transaction_scope(
        &self,
        project_transaction: &ProjectTransaction,
        originating_client: &AnyProtoClient,
        project_id: u64,
        cx: &App,
    ) -> Result<()> {
        anyhow::ensure!(
            self.downstream_client
                .as_ref()
                .is_some_and(|(client, current_id)| {
                    *current_id == project_id && client.is_same_instance(originating_client)
                }),
            "Project sharing changed during transaction publication"
        );
        for buffer in project_transaction.0.keys() {
            self.ensure_buffer_shareable_with_peer(buffer.read(cx), cx)?;
        }
        Ok(())
    }

    pub fn serialize_project_transaction_for_peer(
        &mut self,
        project_transaction: ProjectTransaction,
        peer_id: proto::PeerId,
        cx: &mut Context<Self>,
    ) -> Task<Result<proto::ProjectTransaction>> {
        if project_transaction.0.is_empty() {
            return Task::ready(Ok(proto::ProjectTransaction::default()));
        }
        let Some((originating_client, project_id)) = self.downstream_client.clone() else {
            return Task::ready(Err(anyhow!(
                "Peer transaction transport is no longer available"
            )));
        };
        if let Err(error) = self.ensure_peer_transaction_scope(
            &project_transaction,
            &originating_client,
            project_id,
            cx,
        ) {
            return Task::ready(Err(error));
        }
        cx.spawn(async move |this, cx| {
            for buffer in project_transaction.0.keys() {
                let transfer = this.update(cx, |this, cx| {
                    this.ensure_peer_transaction_scope(
                        &project_transaction,
                        &originating_client,
                        project_id,
                        cx,
                    )?;
                    anyhow::Ok(this.create_buffer_for_peer(buffer, peer_id, cx))
                })??;
                transfer.await?;
                this.read_with(cx, |this, cx| {
                    this.ensure_peer_transaction_scope(
                        &project_transaction,
                        &originating_client,
                        project_id,
                        cx,
                    )
                })??;
            }
            this.update(cx, |this, cx| {
                this.ensure_peer_transaction_scope(
                    &project_transaction,
                    &originating_client,
                    project_id,
                    cx,
                )?;
                let mut serialized_transaction = proto::ProjectTransaction::default();
                for (buffer, transaction) in project_transaction.0 {
                    serialized_transaction
                        .buffer_ids
                        .push(buffer.read(cx).remote_id().into());
                    serialized_transaction
                        .transactions
                        .push(language::proto::serialize_transaction(&transaction));
                }
                Ok(serialized_transaction)
            })?
        })
    }

    pub(crate) fn register_project_search_result_handle(
        &mut self,
    ) -> (u64, async_channel::Receiver<BufferId>) {
        let (tx, rx) = async_channel::unbounded();
        let handle = util::post_inc(&mut self.project_search.next_id);
        let _old_entry = self.project_search.chunks.insert(handle, tx);
        debug_assert!(_old_entry.is_none());
        (handle, rx)
    }

    pub fn register_ongoing_project_search(
        &mut self,
        id: (PeerId, u64),
        search: Task<anyhow::Result<()>>,
    ) {
        let _old = self.project_search.searches_in_progress.insert(id, search);
        debug_assert!(_old.is_none());
    }

    pub async fn handle_find_search_candidates_cancel(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::FindSearchCandidatesCancelled>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let id = (
            envelope.original_sender_id.unwrap_or(envelope.sender_id),
            envelope.payload.handle,
        );
        let _ = this.update(&mut cx, |this, _| {
            this.project_search.searches_in_progress.remove(&id)
        });
        Ok(())
    }

    pub(crate) async fn handle_find_search_candidates_chunk(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::FindSearchCandidatesChunk>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        use proto::find_search_candidates_chunk::Variant;
        let handle = envelope.payload.handle;

        let buffer_ids = match envelope
            .payload
            .variant
            .context("Expected non-null variant")?
        {
            Variant::Matches(find_search_candidates_matches) => find_search_candidates_matches
                .buffer_ids
                .into_iter()
                .filter_map(|buffer_id| BufferId::new(buffer_id).ok())
                .collect::<Vec<_>>(),
            Variant::Done(_) => {
                this.update(&mut cx, |this, _| {
                    this.project_search.chunks.remove(&handle)
                });
                return Ok(proto::Ack {});
            }
        };
        let Some(sender) = this.read_with(&mut cx, |this, _| {
            this.project_search.chunks.get(&handle).cloned()
        }) else {
            return Ok(proto::Ack {});
        };

        for buffer_id in buffer_ids {
            let Ok(_) = sender.send(buffer_id).await else {
                this.update(&mut cx, |this, _| {
                    this.project_search.chunks.remove(&handle)
                });
                return Ok(proto::Ack {});
            };
        }
        Ok(proto::Ack {})
    }
}

impl OpenBuffer {
    fn upgrade(&self) -> Option<Entity<Buffer>> {
        match self {
            OpenBuffer::Complete { buffer, .. } => buffer.upgrade(),
            OpenBuffer::Operations(_) => None,
        }
    }
}

fn is_not_found_error(error: &anyhow::Error) -> bool {
    error
        .root_cause()
        .downcast_ref::<io::Error>()
        .is_some_and(|err| err.kind() == io::ErrorKind::NotFound)
}

fn validate_bounded_buffer_transfer(
    state: &proto::BufferState,
    operations: &[proto::Operation],
    max_bytes: usize,
) -> Result<()> {
    use proto::Message as _;

    anyhow::ensure!(
        operations.len() <= 4096,
        "Artifact buffer history is too large"
    );
    let mut content_bytes = state.base_text.len();
    let mut wire_bytes = state.encoded_len();
    anyhow::ensure!(
        content_bytes <= max_bytes,
        "Artifact base text exceeds its limit"
    );
    for operation in operations {
        if let Some(proto::operation::Variant::Edit(edit)) = &operation.variant {
            for text in &edit.new_text {
                content_bytes = content_bytes
                    .checked_add(text.len())
                    .context("Artifact history size overflow")?;
                anyhow::ensure!(
                    content_bytes <= max_bytes,
                    "Artifact text and retained history exceed the declared byte limit"
                );
            }
        }
        wire_bytes = wire_bytes
            .checked_add(operation.encoded_len())
            .context("Artifact transfer size overflow")?;
    }
    anyhow::ensure!(
        wire_bytes <= max_bytes.saturating_add(64 * 1024),
        "Artifact protocol metadata exceeds its limit"
    );
    Ok(())
}

#[cfg(test)]
mod bounded_transfer_tests {
    use super::validate_bounded_buffer_transfer;
    use rpc::proto;

    #[test]
    fn rejects_retained_text_beyond_artifact_limit() {
        let state = proto::BufferState {
            base_text: "abc".into(),
            ..Default::default()
        };
        assert!(validate_bounded_buffer_transfer(&state, &[], 3).is_ok());
        let edit = proto::Operation {
            variant: Some(proto::operation::Variant::Edit(proto::operation::Edit {
                new_text: vec!["deleted historical content".into()],
                ..Default::default()
            })),
        };
        assert!(validate_bounded_buffer_transfer(&state, &[edit], 3).is_err());
    }

    #[test]
    fn rejects_unbounded_protocol_overhead() {
        let state = proto::BufferState::default();
        let operation = proto::Operation {
            variant: Some(proto::operation::Variant::UpdateCompletionTriggers(
                proto::operation::UpdateCompletionTriggers {
                    triggers: vec!["x".repeat(64 * 1024 + 1)],
                    ..Default::default()
                },
            )),
        };
        assert!(validate_bounded_buffer_transfer(&state, &[operation], 1).is_err());
        assert!(
            validate_bounded_buffer_transfer(&state, &vec![proto::Operation::default(); 4097], 1)
                .is_err()
        );
    }
}

fn apply_initial_line_ending(buffer: &mut Buffer, cx: &mut Context<Buffer>) {
    // Only applies for empty rope or a single line with no trailing newline.
    if buffer.max_point().row > 0 {
        return;
    }
    let location = buffer.file().map(|file| settings::SettingsLocation {
        worktree_id: file.worktree_id(cx),
        path: file.path().as_ref(),
    });
    let language = buffer.language().map(|l| l.name());
    let settings = AllLanguageSettings::get(location, cx).language(location, language.as_ref(), cx);
    let desired = match settings.line_ending {
        LineEndingSetting::Detect => return,
        LineEndingSetting::PreferLf | LineEndingSetting::EnforceLf => LineEnding::Unix,
        LineEndingSetting::PreferCrlf | LineEndingSetting::EnforceCrlf => LineEnding::Windows,
    };
    if buffer.line_ending() != desired {
        buffer.set_line_ending(desired, cx);
    }
}
