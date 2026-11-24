use std::{collections::BTreeMap, sync::Arc, time::Duration};

use anyhow::{Context as _, Result, anyhow, ensure};
use channel::{Channel, ChannelStore};
use chrono::{DateTime, Utc};
use client::{Client, Status, Subscription, User};
use collaboration_domain::{
    AggregateId, CommunityId, OperationId, channel_id_for_legacy_channel,
    community_id_for_legacy_root_channel,
};
use editor::Editor;
use futures::{FutureExt as _, StreamExt as _};
use gpui::{
    App, AppContext as _, Context, Entity, EntityId, Focusable, Global, Render, Role, Task,
    WeakEntity, Window,
};
use menu::Confirm;
use multi_buffer::MultiBufferOffset;
use nostr_compat::SignedEvent;
use project::Project;
use rpc::TypedEnvelope;
use rpc::proto;
use ui::{ButtonStyle, Color, IconButton, IconName, Label, LabelSize, prelude::*};
use util::ResultExt as _;
use workspace::collaborative_composer::{
    CollaborativeComposerActionError, CollaborativeComposerProvider,
};
use workspace::collaborative_participants::{
    CollaborativeConnectionState, CollaborativeParticipant, CollaborativeParticipantPresence,
    CollaborativeParticipantProvider, CollaborativeParticipantProviderState,
    CollaborativeParticipantViewData,
};
use workspace::{Workspace, collaborative_navigation::CollaborativeNavigationTarget};
use zed_credentials_provider::channel_signing::{
    ChannelSigningIdentity, load_or_create_channel_signing_identity,
};

use crate::message_timeline::{
    MessageTimeline, MessageTimelineAuthor, MessageTimelineAuthorKind, MessageTimelineContext,
    MessageTimelineEntry, MessageTimelinePage, MessageTimelineReaction, OptimisticMessage,
};

const CONTRACT_VERSION: u32 = 1;
const PAGE_SIZE: u32 = 100;
const READ_ACKNOWLEDGEMENT_TIMEOUT: Duration = Duration::from_secs(30);
const MESSAGE_KIND: u16 = 40_002;

struct GlobalChannelMessaging(Entity<ChannelMessagingHub>);

struct ChannelMessagingHub {
    client: Arc<Client>,
    sessions: BTreeMap<
        EntityId,
        (
            CommunityId,
            AggregateId,
            WeakEntity<ChannelMessagingTransport>,
        ),
    >,
    _message_subscription: Subscription,
}

impl ChannelMessagingHub {
    async fn handle_stream_update(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CollaborativeMessageStreamUpdate>,
        mut cx: gpui::AsyncApp,
    ) -> Result<()> {
        let sessions = this.read_with(&cx, |this, _| {
            this.sessions
                .values()
                .map(|(_, _, session)| session.clone())
                .collect::<Vec<_>>()
        });
        for session in sessions {
            if let Some(session) = session.upgrade() {
                session.update(&mut cx, |session, cx| {
                    if let Err(error) = session.apply_stream_update(envelope.payload.clone(), cx) {
                        session.set_error(&format!("{error:#}. Retry to reconnect."), cx);
                    }
                });
            }
        }
        Ok(())
    }

    fn close_session(&mut self, session_id: EntityId, cx: &mut Context<Self>) {
        let Some((community_id, channel_id, _)) = self.sessions.remove(&session_id) else {
            return;
        };
        // The service subscription belongs to the client connection, so another window
        // displaying the same channel must retain it when this view closes.
        if self
            .sessions
            .values()
            .any(|(community, channel, _)| *community == community_id && *channel == channel_id)
        {
            return;
        }
        let client = self.client.clone();
        cx.spawn(async move |_, _| {
            client
                .request(proto::CloseCollaborativeChannel {
                    contract_version: CONTRACT_VERSION,
                    community_id: community_id.as_uuid().as_bytes().to_vec(),
                    channel_id: channel_id.as_uuid().as_bytes().to_vec(),
                })
                .await?;
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }
}

impl Global for GlobalChannelMessaging {}

#[derive(Clone)]
pub struct ChannelMessagingViews {
    pub timeline: Entity<MessageTimeline>,
    pub composer: CollaborativeComposerProvider,
    pub participants: CollaborativeParticipantProvider,
}

#[derive(Clone)]
struct ActiveChannel {
    generation: u64,
    community_id: CommunityId,
    channel_id: AggregateId,
    user: Arc<User>,
    timeline: Entity<MessageTimeline>,
    composer: Entity<ChannelMessageComposer>,
    signing_identity: Option<Arc<ChannelSigningIdentity>>,
    principal_id: Option<Vec<u8>>,
    next_cursor: Option<proto::CollaborativeMessageCursor>,
    next_cursor_token: Option<String>,
    authoritative_outbox_cursor: u64,
    history_head_loaded: bool,
    latest_target: Option<MessageTarget>,
    latest_own_target: Option<MessageTarget>,
    reply_target: Option<MessageTarget>,
}

#[derive(Clone)]
struct PendingOperation {
    request: proto::ApplyCollaborativeMessageOperation,
    optimistic: bool,
}

#[derive(Clone)]
struct MessageTarget {
    message_id: Vec<u8>,
    source_event_id: Vec<u8>,
    version: u64,
    reaction_version: u64,
}

struct PendingReadAcknowledgement {
    request: proto::ApplyCollaborativeMessageOperation,
    task: Option<Task<()>>,
}

pub struct ChannelMessagingTransport {
    client: Arc<Client>,
    active: Option<ActiveChannel>,
    pending: BTreeMap<String, PendingOperation>,
    active_submit: Option<(String, Task<()>)>,
    read_acknowledgement: Option<PendingReadAcknowledgement>,
    desired_read_sequence: u64,
    status: String,
    hub: Entity<ChannelMessagingHub>,
    _connection_task: Task<()>,
}

impl ChannelMessagingTransport {
    pub fn open(
        channel: Arc<Channel>,
        user: Arc<User>,
        client: Arc<Client>,
        project: Entity<Project>,
        workspace: Entity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<ChannelMessagingViews> {
        let hub = if let Some(global) = cx.try_global::<GlobalChannelMessaging>() {
            global.0.clone()
        } else {
            let hub = cx.new(|cx| ChannelMessagingHub {
                client: client.clone(),
                sessions: BTreeMap::new(),
                _message_subscription: client.add_message_handler(
                    cx.weak_entity(),
                    ChannelMessagingHub::handle_stream_update,
                ),
            });
            cx.set_global(GlobalChannelMessaging(hub.clone()));
            hub
        };
        if !Arc::ptr_eq(&hub.read(cx).client, &client) {
            return Err(anyhow!(
                "collaboration client changed while the workspace is active"
            ));
        }
        let transport = cx.new(|cx| {
            let mut connection_status = client.status();
            let connection_task = cx.spawn(async move |this: WeakEntity<Self>, cx| {
                while let Some(status) = connection_status.next().await {
                    let Some(this) = this.upgrade() else {
                        break;
                    };
                    this.update(cx, |this, cx| match status {
                        Status::Connected { .. } => this.reconnect(cx),
                        Status::ConnectionLost | Status::Reconnecting => {
                            this.set_status("Offline — reconnecting from server cursor", cx)
                        }
                        Status::SignedOut | Status::UpgradeRequired => {
                            this.set_status("Channel messaging requires authentication", cx)
                        }
                        _ => {}
                    });
                }
            });
            let session_id = cx.entity_id();
            cx.on_release(move |this: &mut Self, cx| {
                this.hub
                    .update(cx, |hub, cx| hub.close_session(session_id, cx));
            })
            .detach();
            Self {
                client: client.clone(),
                active: None,
                pending: BTreeMap::new(),
                active_submit: None,
                read_acknowledgement: None,
                desired_read_sequence: 0,
                status: String::new(),
                hub: hub.clone(),
                _connection_task: connection_task,
            }
        });

        let channel_store = ChannelStore::global(cx);
        let participant_channel = channel.clone();
        let participants = CollaborativeParticipantProvider::from_reader(
            project.clone(),
            channel_store.entity_id(),
            move |cx| {
                let store = channel_store.read(cx);
                let connection = call::ActiveCall::try_global(cx)
                    .and_then(|active_call| {
                        let active_call = active_call.read(cx);
                        (active_call.channel_id(cx) == Some(participant_channel.id))
                            .then(|| active_call.room().cloned())
                            .flatten()
                    })
                    .map(|room| {
                        let status = room.read(cx).status();
                        if status.is_online() {
                            CollaborativeConnectionState::Connected
                        } else if status.is_offline() {
                            CollaborativeConnectionState::Failed
                        } else {
                            CollaborativeConnectionState::Connecting
                        }
                    })
                    .unwrap_or_default();

                CollaborativeParticipantProviderState::Ready(CollaborativeParticipantViewData {
                    task_title: Some(participant_channel.name.clone()),
                    connection,
                    participants: store
                        .channel_participants(participant_channel.id)
                        .iter()
                        .map(|user| {
                            CollaborativeParticipant::human(
                                user,
                                CollaborativeParticipantPresence::Online,
                            )
                        })
                        .collect(),
                    ..Default::default()
                })
            },
        );
        let timeline = cx.new(MessageTimeline::new);
        let editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Message this channel…", window, cx);
            editor
        });
        let composer = cx.new(|cx| ChannelMessageComposer::new(editor, transport.downgrade(), cx));
        let generation = 1;
        let community_id = community_id_for_legacy_root_channel(channel.root_id().0);
        let channel_id = channel_id_for_legacy_channel(channel.id.0);
        hub.update(cx, |hub, _| {
            hub.sessions.insert(
                transport.entity_id(),
                (community_id, channel_id, transport.downgrade()),
            );
        });
        transport.update(cx, |this, cx| {
            this.pending.clear();
            this.active_submit = None;
            this.active = Some(ActiveChannel {
                generation,
                community_id,
                channel_id,
                user,
                timeline: timeline.clone(),
                composer: composer.clone(),
                signing_identity: None,
                principal_id: None,
                next_cursor: None,
                next_cursor_token: None,
                authoritative_outbox_cursor: 0,
                history_head_loaded: false,
                latest_target: None,
                latest_own_target: None,
                reply_target: None,
            });
            this.set_status("Connecting to channel…", cx);
            this.connect(generation, cx);
            cx.observe(&workspace, move |this, workspace, cx| {
                let selected_channel_id =
                    match workspace.read(cx).collaborative_navigation().current() {
                        Some(CollaborativeNavigationTarget::Channel { channel_id }) => {
                            channel_id.parse::<u64>().ok()
                        }
                        _ => None,
                    };
                if selected_channel_id != Some(channel.id.0)
                    && this
                        .active
                        .as_ref()
                        .is_some_and(|active| active.generation == generation)
                {
                    this.close_active(cx);
                    this.pending.clear();
                    this.active_submit = None;
                }
            })
            .detach();
            cx.observe_release(&workspace, move |this, _, cx| {
                if this
                    .active
                    .as_ref()
                    .is_some_and(|active| active.generation == generation)
                {
                    this.close_active(cx);
                    this.pending.clear();
                    this.active_submit = None;
                }
            })
            .detach();
        });

        let focus_handle = composer.read(cx).editor.read(cx).focus_handle(cx);
        let submit_transport = transport.clone();
        let cancel_transport = transport.clone();
        let provider = CollaborativeComposerProvider::new(
            project,
            composer.into(),
            move |cx| submit_transport.update(cx, |transport, cx| transport.submit(cx)),
            move |cx| cancel_transport.update(cx, |transport, cx| transport.cancel(cx)),
        )
        .with_focus_handle(focus_handle);
        Ok(ChannelMessagingViews {
            timeline,
            composer: provider,
            participants,
        })
    }

    fn connect(&mut self, generation: u64, cx: &mut Context<Self>) {
        let Some(active) = self
            .active
            .as_ref()
            .filter(|active| active.generation == generation)
        else {
            return;
        };
        let provider = zed_credentials_provider::global(cx);
        let client = self.client.clone();
        let community_id = active.community_id;
        let channel_id = active.channel_id;
        let account_id = active.user.legacy_id;
        let username = active.user.username.to_string();
        let after_outbox_sequence = active.authoritative_outbox_cursor;
        cx.spawn(async move |this, cx| {
            let result = async {
                let identity = Arc::new(
                    load_or_create_channel_signing_identity(
                        provider.as_ref(),
                        community_id,
                        account_id,
                        cx,
                    )
                    .await
                    .context("load protected channel signing identity")?,
                );
                let response = client
                    .request(proto::OpenCollaborativeChannel {
                        contract_version: CONTRACT_VERSION,
                        community_id: community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: channel_id.as_uuid().as_bytes().to_vec(),
                        page_size: PAGE_SIZE,
                        after_outbox_sequence,
                        signing_public_key: identity.public_key().to_vec(),
                    })
                    .await
                    .context("open collaborative channel")?;
                this.update(cx, |this, cx| {
                    this.apply_open_response(generation, identity, response, cx)
                })??;
                anyhow::Ok(())
            }
            .await;
            if let Err(error) = result {
                this.update(cx, |this, cx| {
                    if this
                        .active
                        .as_ref()
                        .is_some_and(|active| active.generation == generation)
                    {
                        this.set_error(&format!("{error:#}. Retry to reconnect."), cx);
                    }
                })?;
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
        self.set_status(&format!("Connecting as {username}…"), cx);
    }

    fn apply_open_response(
        &mut self,
        generation: u64,
        identity: Arc<ChannelSigningIdentity>,
        response: proto::OpenCollaborativeChannelResponse,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        ensure_contract_version(response.contract_version)?;
        ensure_success(response.error_code)?;
        let Some(active) = self
            .active
            .as_mut()
            .filter(|active| active.generation == generation)
        else {
            return Ok(());
        };
        ensure!(
            response.principal_id.len() == 16,
            "invalid channel principal identity"
        );
        let page = response
            .page
            .context("channel open response omitted its page")?;
        validate_page(active, &page)?;
        let previous_principal = active.principal_id.replace(response.principal_id);
        if let Err(error) = apply_page(active, None, page, &self.pending, cx) {
            active.principal_id = previous_principal;
            return Err(error);
        }
        active.signing_identity = Some(identity);
        self.set_status("Connected — history is server-backed", cx);
        self.start_read_acknowledgement(cx);
        Ok(())
    }

    fn reconnect(&mut self, cx: &mut Context<Self>) {
        if let Some(generation) = self.active.as_ref().map(|active| active.generation) {
            if let Some(pending) = &mut self.read_acknowledgement {
                pending.task = None;
            }
            self.set_status("Reconnected — replaying missed channel activity", cx);
            self.connect(generation, cx);
        }
    }

    fn submit(
        &mut self,
        cx: &mut Context<Self>,
    ) -> std::result::Result<(), CollaborativeComposerActionError> {
        let active = self
            .active
            .as_ref()
            .ok_or(CollaborativeComposerActionError::ThreadUnavailable)?;
        if self.active_submit.is_some() {
            return Err(CollaborativeComposerActionError::ProviderFailure(
                "a channel message is already being submitted".into(),
            ));
        }
        let body = active.composer.read(cx).editor.read(cx).text(cx);
        let body = body.trim().to_owned();
        if body.is_empty() {
            return Err(CollaborativeComposerActionError::EmptyInput);
        }
        let identity = active.signing_identity.clone().ok_or_else(|| {
            CollaborativeComposerActionError::ProviderFailure(
                "channel signing identity is still loading".into(),
            )
        })?;
        let operation_id = OperationId::new();
        let message_id = AggregateId::new();
        let operation_id_string = operation_id.to_string();
        let reply_target = active.reply_target.clone();
        let mut tags = vec![vec!["h".into(), active.channel_id.to_string()]];
        if let Some(reply_target) = &reply_target {
            tags.push(vec![
                "e".into(),
                hex::encode(&reply_target.source_event_id),
                String::new(),
                "reply".into(),
            ]);
        }
        let signed_event = identity
            .sign(unix_time_seconds(), MESSAGE_KIND, tags, body.clone())
            .map_err(|error| {
                CollaborativeComposerActionError::ProviderFailure(error.to_string())
            })?;
        active
            .timeline
            .update(cx, |timeline, cx| {
                timeline.begin_optimistic(
                    OptimisticMessage {
                        operation_id: operation_id_string.clone(),
                        author: MessageTimelineAuthor {
                            kind: MessageTimelineAuthorKind::Human,
                            id: active.user.legacy_id.to_string(),
                            label: active.user.username.to_string(),
                        },
                        content: body.clone(),
                        reply_to: reply_target
                            .as_ref()
                            .map(|target| hex::encode(&target.source_event_id)),
                        occurred_at: Utc::now(),
                        context: timeline_context(active),
                    },
                    cx,
                )
            })
            .map_err(|error| {
                CollaborativeComposerActionError::ProviderFailure(error.to_string())
            })?;
        let request = proto::ApplyCollaborativeMessageOperation {
            contract_version: CONTRACT_VERSION,
            community_id: active.community_id.as_uuid().as_bytes().to_vec(),
            channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
            message_id: message_id.as_uuid().as_bytes().to_vec(),
            operation_id: operation_id_string.clone(),
            kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate.into(),
            expected_version: 0,
            body,
            reply_to_event_id: reply_target.map_or_else(Vec::new, |target| target.source_event_id),
            reaction: String::new(),
            related_reaction_event_id: Vec::new(),
            signed_event: Some(signed_event_to_proto(signed_event)),
            acknowledged_outbox_sequence: 0,
        };
        self.pending.insert(
            operation_id_string.clone(),
            PendingOperation {
                request: request.clone(),
                optimistic: true,
            },
        );
        self.start_operation(operation_id_string, request, cx);
        Ok(())
    }

    fn start_operation(
        &mut self,
        operation_id: String,
        request: proto::ApplyCollaborativeMessageOperation,
        cx: &mut Context<Self>,
    ) {
        let client = self.client.clone();
        let task_operation_id = operation_id.clone();
        let task = cx.spawn(async move |this, cx| {
            let response = client.request(request).await;
            this.update(cx, |this, cx| {
                if let Err(error) = this.finish_operation(&task_operation_id, response, cx) {
                    this.set_error(&format!("{error:#}. Message retained for retry."), cx);
                }
            })
            .log_err();
        });
        self.active_submit = Some((operation_id, task));
        self.set_status("Sending channel message…", cx);
    }

    fn finish_operation(
        &mut self,
        operation_id: &str,
        response: Result<proto::ApplyCollaborativeMessageOperationResponse>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if self
            .active_submit
            .as_ref()
            .is_some_and(|(id, _)| id == operation_id)
        {
            self.active_submit = None;
        }
        let timeline = self
            .active
            .as_ref()
            .map(|active| active.timeline.clone())
            .context("channel is no longer active")?;
        let optimistic = self
            .pending
            .get(operation_id)
            .is_some_and(|operation| operation.optimistic);
        let response = response.and_then(|response| {
            ensure_contract_version(response.contract_version)?;
            if response.accepted {
                ensure_success(response.error_code)?;
                let record = response
                    .message
                    .as_ref()
                    .context("accepted channel operation omitted its message")?;
                let active = self
                    .active
                    .as_ref()
                    .context("channel is no longer active")?;
                validate_record_scope(active, record)?;
                record_to_entry_with_origin(record, &self.pending)?;
                ensure!(
                    record.accepted_operation_id == operation_id,
                    "channel operation receipt does not match the request"
                );
                let pending = self
                    .pending
                    .get(operation_id)
                    .context("channel operation is no longer pending")?;
                ensure!(
                    record.message_id == pending.request.message_id,
                    "channel operation receipt identifies a different message"
                );
            } else {
                ensure!(
                    response.error_code != 0,
                    "rejected channel operation omitted its error"
                );
            }
            Ok(response)
        });
        match response {
            Ok(response) if response.accepted => {
                let record = response
                    .message
                    .context("accepted channel operation omitted its message")?;
                let entry = record_to_entry_with_origin(&record, &self.pending)?;
                if optimistic {
                    timeline.update(cx, |timeline, cx| {
                        timeline.accept_optimistic(operation_id, entry, cx)
                    })?;
                    self.clear_delivered_draft(operation_id, cx);
                } else {
                    timeline.update(cx, |timeline, cx| timeline.upsert_replayed(entry, cx))?;
                }
                self.pending.remove(operation_id);
                self.set_status("Channel operation delivered", cx);
            }
            Ok(response) => {
                let reason = error_label(response.error_code).to_owned();
                if optimistic {
                    timeline.update(cx, |timeline, cx| {
                        timeline.reject_optimistic(operation_id, reason.clone(), cx)
                    })?;
                }
                self.set_error(&format!("{reason} — retry is available"), cx);
            }
            Err(error) => {
                if optimistic {
                    timeline.update(cx, |timeline, cx| {
                        timeline.reject_optimistic(
                            operation_id,
                            "Message was not confirmed — retained for retry",
                            cx,
                        )
                    })?;
                }
                self.set_error(
                    &format!("Message was not confirmed — retrying requires confirmation: {error}"),
                    cx,
                );
            }
        }
        Ok(())
    }

    fn retry(&mut self, cx: &mut Context<Self>) -> Result<()> {
        if self.active_submit.is_some() {
            return Ok(());
        }
        let (operation_id, pending) = self
            .pending
            .iter()
            .next_back()
            .map(|(operation_id, pending)| (operation_id.clone(), pending.clone()))
            .context("no failed channel message is available to retry")?;
        if pending.optimistic
            && let Some(active) = &self.active
        {
            active.timeline.update(cx, |timeline, cx| {
                timeline.retry_optimistic(&operation_id, cx)
            })?;
        }
        self.start_operation(operation_id, pending.request, cx);
        Ok(())
    }

    fn edit_latest(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let active = self
            .active
            .as_ref()
            .context("channel is no longer active")?;
        let target = active
            .latest_own_target
            .clone()
            .context("no authored message is available to edit")?;
        let body = active.composer.read(cx).editor.read(cx).text(cx);
        let body = body.trim().to_owned();
        if body.is_empty() {
            return Err(anyhow!(
                "type replacement text before editing the latest message"
            ));
        }
        self.start_target_operation(
            target,
            proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit,
            40_003,
            body,
            String::new(),
            cx,
        )
    }

    fn delete_latest(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let target = self
            .active
            .as_ref()
            .and_then(|active| active.latest_own_target.clone())
            .context("no authored message is available to delete")?;
        self.start_target_operation(
            target,
            proto::CollaborativeMessageOperationKind::CollaborativeMessageDelete,
            5,
            String::new(),
            String::new(),
            cx,
        )
    }

    fn react_to_latest(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let target = self
            .active
            .as_ref()
            .and_then(|active| active.latest_target.clone())
            .context("no message is available to react to")?;
        self.start_target_operation(
            target,
            proto::CollaborativeMessageOperationKind::CollaborativeMessageReactionAdd,
            7,
            "👍".into(),
            "👍".into(),
            cx,
        )
    }

    fn toggle_reply_to_latest(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let active = self
            .active
            .as_mut()
            .context("channel is no longer active")?;
        if active.reply_target.is_some() {
            active.reply_target = None;
            self.set_status("Reply target cleared", cx);
        } else {
            active.reply_target = Some(
                active
                    .latest_target
                    .clone()
                    .context("no message is available to reply to")?,
            );
            self.set_status("Replying to the latest message", cx);
        }
        Ok(())
    }

    fn start_target_operation(
        &mut self,
        target: MessageTarget,
        kind: proto::CollaborativeMessageOperationKind,
        event_kind: u16,
        body: String,
        reaction: String,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if self.active_submit.is_some() {
            return Err(anyhow!("a channel operation is already in progress"));
        }
        let active = self
            .active
            .as_ref()
            .context("channel is no longer active")?;
        let identity = active
            .signing_identity
            .clone()
            .context("channel signing identity is still loading")?;
        let target_event_id = hex::encode(&target.source_event_id);
        let signed_event = identity.sign(
            unix_time_seconds(),
            event_kind,
            vec![vec!["e".into(), target_event_id]],
            body.clone(),
        )?;
        let operation_id = OperationId::new().to_string();
        let expected_version =
            if kind == proto::CollaborativeMessageOperationKind::CollaborativeMessageReactionAdd {
                target.reaction_version
            } else {
                target.version
            };
        let request = proto::ApplyCollaborativeMessageOperation {
            contract_version: CONTRACT_VERSION,
            community_id: active.community_id.as_uuid().as_bytes().to_vec(),
            channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
            message_id: target.message_id,
            operation_id: operation_id.clone(),
            kind: kind.into(),
            expected_version,
            body,
            reply_to_event_id: Vec::new(),
            reaction,
            related_reaction_event_id: Vec::new(),
            signed_event: Some(signed_event_to_proto(signed_event)),
            acknowledged_outbox_sequence: 0,
        };
        self.pending.insert(
            operation_id.clone(),
            PendingOperation {
                request: request.clone(),
                optimistic: false,
            },
        );
        self.start_operation(operation_id, request, cx);
        Ok(())
    }

    fn acknowledge(&mut self, sequence: u64, cx: &mut Context<Self>) {
        self.desired_read_sequence = self.desired_read_sequence.max(sequence);
        if self.read_acknowledgement.is_none() {
            self.start_read_acknowledgement(cx);
        }
    }

    fn start_read_acknowledgement(&mut self, cx: &mut Context<Self>) {
        let Some(active) = self.active.as_ref() else {
            return;
        };
        if self.desired_read_sequence == 0
            || self
                .read_acknowledgement
                .as_ref()
                .is_some_and(|pending| pending.task.is_some())
        {
            return;
        }
        let generation = active.generation;
        if self.read_acknowledgement.is_none() {
            let request = proto::ApplyCollaborativeMessageOperation {
                contract_version: CONTRACT_VERSION,
                community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                message_id: AggregateId::new().as_uuid().as_bytes().to_vec(),
                operation_id: OperationId::new().to_string(),
                kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge
                    .into(),
                expected_version: 0,
                body: String::new(),
                reply_to_event_id: Vec::new(),
                reaction: String::new(),
                related_reaction_event_id: Vec::new(),
                signed_event: None,
                acknowledged_outbox_sequence: self.desired_read_sequence,
            };
            self.read_acknowledgement = Some(PendingReadAcknowledgement {
                request,
                task: None,
            });
        }
        let Some(pending) = self.read_acknowledgement.as_mut() else {
            return;
        };
        let request = pending.request.clone();
        let operation_id = request.operation_id.clone();
        let client = self.client.clone();
        let timeout = cx.background_executor().timer(READ_ACKNOWLEDGEMENT_TIMEOUT);
        pending.task = Some(cx.spawn(async move |this, cx| {
            let response = wait_for_read_acknowledgement(client.request(request), timeout).await;
            this.update(cx, |this, cx| {
                this.finish_read_acknowledgement(generation, &operation_id, response, cx);
            })
            .log_err();
        }));
    }

    fn finish_read_acknowledgement(
        &mut self,
        generation: u64,
        operation_id: &str,
        response: Result<proto::ApplyCollaborativeMessageOperationResponse>,
        cx: &mut Context<Self>,
    ) {
        let Some(active) = self
            .active
            .as_ref()
            .filter(|active| active.generation == generation)
        else {
            return;
        };
        let Some(pending) = self
            .read_acknowledgement
            .as_mut()
            .filter(|pending| pending.request.operation_id == operation_id)
        else {
            return;
        };
        pending.task = None;
        let response = response.and_then(|response| {
            ensure_contract_version(response.contract_version)?;
            ensure_success(response.error_code)?;
            ensure!(response.accepted, "read acknowledgement was not accepted");
            Ok(())
        });
        active.composer.update(cx, |composer, cx| {
            composer.read_error = response
                .is_err()
                .then_some("Read state not synchronized — reconnect to retry.");
            cx.notify();
        });
        if response.is_ok() {
            let sequence = pending.request.acknowledged_outbox_sequence;
            self.read_acknowledgement = None;
            if self.desired_read_sequence > sequence {
                self.start_read_acknowledgement(cx);
            } else {
                self.desired_read_sequence = 0;
            }
        }
    }

    fn cancel(
        &mut self,
        cx: &mut Context<Self>,
    ) -> std::result::Result<(), CollaborativeComposerActionError> {
        if let Some((operation_id, task)) = self.active_submit.take() {
            drop(task);
            if let Some(active) = &self.active {
                active
                    .timeline
                    .update(cx, |timeline, cx| {
                        timeline.reject_optimistic(
                            &operation_id,
                            "Cancelled locally; a committed server result will still reconcile",
                            cx,
                        )
                    })
                    .map_err(|error| {
                        CollaborativeComposerActionError::ProviderFailure(error.to_string())
                    })?;
            }
            self.set_status("Message submission cancelled", cx);
            return Ok(());
        }
        if let Some(active) = &self.active {
            active
                .composer
                .update(cx, |composer, cx| composer.clear(cx));
            self.set_status("Draft cleared", cx);
            return Ok(());
        }
        Err(CollaborativeComposerActionError::ThreadUnavailable)
    }

    fn load_more(&mut self, cx: &mut Context<Self>) {
        let Some(active) = self.active.clone() else {
            return;
        };
        let Some(cursor) = active.next_cursor.clone() else {
            self.set_status("All channel history is loaded", cx);
            return;
        };
        let request_cursor_token = active.next_cursor_token.clone();
        let generation = active.generation;
        let client = self.client.clone();
        cx.spawn(async move |this, cx| {
            let result = async {
                let response = client
                    .request(proto::GetCollaborativeMessageWindow {
                        contract_version: CONTRACT_VERSION,
                        community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                        thread_root_event_id: Vec::new(),
                        page_size: PAGE_SIZE,
                        cursor: Some(cursor),
                    })
                    .await?;
                ensure_contract_version(response.contract_version)?;
                ensure_success(response.error_code)?;
                let page = response
                    .page
                    .context("channel history response omitted its page")?;
                this.update(cx, |this, cx| {
                    let current = this
                        .active
                        .as_mut()
                        .filter(|current| current.generation == active.generation)
                        .context("channel changed while loading history")?;
                    apply_page(current, request_cursor_token, page, &this.pending, cx)?;
                    this.set_status("Loaded earlier channel messages", cx);
                    anyhow::Ok(())
                })??;
                anyhow::Ok(())
            }
            .await;
            if let Err(error) = result {
                this.update(cx, |this, cx| {
                    if this
                        .active
                        .as_ref()
                        .is_some_and(|active| active.generation == generation)
                    {
                        this.set_error(&format!("{error:#}. Retry to reconnect."), cx);
                    }
                })?;
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
        self.set_status("Loading earlier channel messages…", cx);
    }

    fn apply_stream_update(
        &mut self,
        update: proto::CollaborativeMessageStreamUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        ensure_contract_version(update.contract_version)?;
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        if update.community_id != active.community_id.as_uuid().as_bytes()
            || update.channel_id != active.channel_id.as_uuid().as_bytes()
        {
            return Ok(());
        }
        if let Some(status) = &update.status {
            ensure!(
                update.outbox_sequence == 0
                    && update.message.is_none()
                    && update.actor_principal_id.is_empty()
                    && update.acknowledged_outbox_sequence == 0,
                "channel status frame contains message progress"
            );
            let error_code = proto::CollaborativeMessageErrorCode::from_i32(status.error_code)
                .context("unsupported channel delivery status")?;
            let warning = match error_code {
                proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone => {
                    ensure!(
                        status.retry_after_millis == 0,
                        "ready status contains retry delay"
                    );
                    None
                }
                proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied => {
                    ensure!(
                        status.retry_after_millis == 0,
                        "denied status contains retry delay"
                    );
                    Some(
                        "Channel access revoked. Last synchronized messages retained; reconnect to recheck access.",
                    )
                }
                proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying => {
                    ensure!(
                        (1..=60_000).contains(&status.retry_after_millis),
                        "invalid channel retry delay"
                    );
                    Some(
                        "Channel delivery interrupted — retrying automatically. Showing last synchronized messages.",
                    )
                }
                _ => anyhow::bail!("unsupported channel delivery status"),
            };
            active.composer.update(cx, |composer, cx| {
                composer.delivery_error = warning;
                cx.notify();
            });
            if warning.is_none() && !active.composer.read(cx).status_is_error {
                self.set_status("Channel delivery recovered — messages synchronized", cx);
            }
            return Ok(());
        }
        if update.outbox_sequence <= active.authoritative_outbox_cursor {
            return Ok(());
        }
        let operation_kind =
            proto::CollaborativeMessageOperationKind::from_i32(update.operation_kind)
                .context("unsupported channel operation kind")?;
        if operation_kind
            == proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge
        {
            ensure!(
                update.actor_principal_id.len() == 16,
                "invalid channel acknowledgement actor"
            );
            ensure!(
                update.acknowledged_outbox_sequence > 0
                    && i64::try_from(update.acknowledged_outbox_sequence).is_ok(),
                "invalid channel acknowledgement cursor"
            );
            if let Some(record) = &update.message {
                validate_record_scope(active, record)?;
                record_to_entry(record)?;
            }
            active.authoritative_outbox_cursor = update.outbox_sequence;
            if !active.composer.read(cx).status_is_error {
                self.set_status("Read state synchronized", cx);
            }
            return Ok(());
        }
        let record = update
            .message
            .context("channel update omitted its message")?;
        let operation_id = (!record.accepted_operation_id.is_empty())
            .then(|| record.accepted_operation_id.clone());
        validate_record_scope(active, &record)?;
        if let Some(pending) = operation_id.as_ref().and_then(|id| self.pending.get(id)) {
            ensure!(
                record.message_id == pending.request.message_id,
                "channel stream receipt identifies a different message"
            );
        }
        let entry = record_to_entry_with_origin(&record, &self.pending)?;
        active
            .timeline
            .update(cx, |timeline, cx| timeline.upsert_replayed(entry, cx))?;
        remember_record(active, &record, true);
        active.authoritative_outbox_cursor = update.outbox_sequence;
        if let Some(operation_id) = operation_id {
            self.clear_delivered_draft(&operation_id, cx);
        }
        self.acknowledge(update.outbox_sequence, cx);
        self.set_status("Channel updated in real time", cx);
        Ok(())
    }

    fn clear_delivered_draft(&mut self, operation_id: &str, cx: &mut Context<Self>) {
        let Some(pending) = self
            .pending
            .get(operation_id)
            .filter(|pending| pending.optimistic)
        else {
            return;
        };
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let body = active.composer.read(cx).editor.read(cx).text(cx);
        let reply_to = active
            .reply_target
            .as_ref()
            .map_or(&[][..], |target| target.source_event_id.as_slice());
        // A delayed receipt must not erase a draft or reply selected after submission.
        if body.trim() == pending.request.body && reply_to == pending.request.reply_to_event_id {
            active
                .composer
                .update(cx, |composer, cx| composer.clear(cx));
            active.reply_target = None;
        }
    }

    fn close_active(&mut self, cx: &mut Context<Self>) {
        self.set_status("Channel closed — select it again to reconnect", cx);
        self.read_acknowledgement = None;
        self.desired_read_sequence = 0;
        self.active = None;
        let session_id = cx.entity_id();
        self.hub
            .update(cx, |hub, cx| hub.close_session(session_id, cx));
    }

    fn run_composer_action<E: std::fmt::Display>(
        &mut self,
        cx: &mut Context<Self>,
        action: impl FnOnce(&mut Self, &mut Context<Self>) -> std::result::Result<(), E>,
    ) {
        if let Err(error) = action(self, cx) {
            self.set_error(&error.to_string(), cx);
        }
    }

    fn set_error(&mut self, error: &str, cx: &mut Context<Self>) {
        self.set_status(error, cx);
        if let Some(active) = &self.active {
            active.composer.update(cx, |composer, cx| {
                composer.status_is_error = true;
                cx.notify();
            });
        }
    }

    fn set_status(&mut self, status: &str, cx: &mut Context<Self>) {
        self.status.clear();
        self.status.push_str(status);
        if let Some(active) = &self.active {
            active.composer.update(cx, |composer, cx| {
                composer.status_is_error = false;
                composer.status.clear();
                composer.status.push_str(status);
                cx.notify();
            });
        }
        cx.notify();
    }
}

pub struct ChannelMessageComposer {
    editor: Entity<Editor>,
    transport: WeakEntity<ChannelMessagingTransport>,
    status: String,
    status_is_error: bool,
    delivery_error: Option<&'static str>,
    read_error: Option<&'static str>,
}

impl ChannelMessageComposer {
    fn new(
        editor: Entity<Editor>,
        transport: WeakEntity<ChannelMessagingTransport>,
        _cx: &mut Context<Self>,
    ) -> Self {
        Self {
            editor,
            transport,
            status: "Connecting to channel…".into(),
            status_is_error: false,
            delivery_error: None,
            read_error: None,
        }
    }

    fn status_text(&self) -> String {
        let status = match self.delivery_error {
            Some(warning) if self.status_is_error => format!("{warning} {}", self.status),
            Some(warning) => warning.into(),
            None => self.status.clone(),
        };
        match self.read_error {
            Some(warning) => format!("{warning} {status}"),
            None => status,
        }
    }

    fn clear(&self, cx: &mut App) {
        let buffer = self.editor.read(cx).buffer().clone();
        buffer.update(cx, |buffer, cx| {
            let length = buffer.len(cx);
            buffer.edit([(MultiBufferOffset(0)..length, "")], None, cx);
        });
    }
}

impl Render for ChannelMessageComposer {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let submit_transport = self.transport.clone();
        let cancel_transport = self.transport.clone();
        let retry_transport = self.transport.clone();
        let page_transport = self.transport.clone();
        let edit_transport = self.transport.clone();
        let delete_transport = self.transport.clone();
        let reaction_transport = self.transport.clone();
        let reply_transport = self.transport.clone();
        let reconnect_transport = self.transport.clone();
        let status_is_error =
            self.status_is_error || self.delivery_error.is_some() || self.read_error.is_some();
        v_flex()
            .id("collaborative-channel-composer")
            .w_full()
            .gap_1()
            .p(DynamicSpacing::Base08.rems(cx))
            .rounded(rems_from_px(ui::fractal::RADIUS_ELEMENT))
            .border_1()
            .border_color(cx.theme().colors().border)
            .on_action(move |_: &Confirm, _window, cx| {
                submit_transport
                    .update(cx, |transport, cx| {
                        transport.run_composer_action(cx, ChannelMessagingTransport::submit)
                    })
                    .log_err();
            })
            .child(div().w_full().min_w_0().child(self.editor.clone()))
            .child(
                h_flex()
                    .w_full()
                    .flex_wrap()
                    .gap_1()
                    .when(status_is_error, |row| {
                        row.child(
                            IconButton::new("reconnect-channel-messages", IconName::RotateCw)
                                .aria_label("Reconnect channel messages")
                                .style(ButtonStyle::Subtle)
                                .tooltip(ui::Tooltip::text(
                                    "Reconnect without discarding messages or draft",
                                ))
                                .on_click(move |_, _, cx| {
                                    reconnect_transport
                                        .update(cx, |transport, cx| transport.reconnect(cx))
                                        .log_err();
                                }),
                        )
                    })
                    .child(
                        IconButton::new("cancel-channel-message", IconName::Stop)
                            .aria_label("Cancel send or clear draft")
                            .style(ButtonStyle::Subtle)
                            .tooltip(ui::Tooltip::text("Cancel send or clear draft"))
                            .on_click(move |_, _, cx| {
                                cancel_transport
                                    .update(cx, |transport, cx| {
                                        transport.run_composer_action(
                                            cx,
                                            ChannelMessagingTransport::cancel,
                                        )
                                    })
                                    .log_err();
                            }),
                    )
                    .child(
                        IconButton::new(
                            "reply-to-latest-channel-message",
                            IconName::ReplyArrowRight,
                        )
                        .style(ButtonStyle::Subtle)
                        .tooltip(ui::Tooltip::text("Reply to the latest message"))
                        .on_click(move |_, _, cx| {
                            reply_transport
                                .update(cx, |transport, cx| {
                                    transport.run_composer_action(
                                        cx,
                                        ChannelMessagingTransport::toggle_reply_to_latest,
                                    )
                                })
                                .log_err();
                        }),
                    )
                    .child(
                        IconButton::new("edit-latest-channel-message", IconName::Pencil)
                            .aria_label("Edit your latest message")
                            .style(ButtonStyle::Subtle)
                            .tooltip(ui::Tooltip::text(
                                "Replace your latest message with the composer text",
                            ))
                            .on_click(move |_, _, cx| {
                                edit_transport
                                    .update(cx, |transport, cx| {
                                        transport.run_composer_action(
                                            cx,
                                            ChannelMessagingTransport::edit_latest,
                                        )
                                    })
                                    .log_err();
                            }),
                    )
                    .child(
                        IconButton::new("react-latest-channel-message", IconName::ThumbsUp)
                            .aria_label("React to latest message")
                            .style(ButtonStyle::Subtle)
                            .tooltip(ui::Tooltip::text("React to the latest message"))
                            .on_click(move |_, _, cx| {
                                reaction_transport
                                    .update(cx, |transport, cx| {
                                        transport.run_composer_action(
                                            cx,
                                            ChannelMessagingTransport::react_to_latest,
                                        )
                                    })
                                    .log_err();
                            }),
                    )
                    .child(
                        IconButton::new("delete-latest-channel-message", IconName::Trash)
                            .aria_label("Delete your latest message")
                            .style(ButtonStyle::Subtle)
                            .tooltip(ui::Tooltip::text("Delete your latest message"))
                            .on_click(move |_, _, cx| {
                                delete_transport
                                    .update(cx, |transport, cx| {
                                        transport.run_composer_action(
                                            cx,
                                            ChannelMessagingTransport::delete_latest,
                                        )
                                    })
                                    .log_err();
                            }),
                    )
                    .child(
                        IconButton::new("send-channel-message", IconName::Send)
                            .aria_label("Send or retry message")
                            .style(ButtonStyle::Filled)
                            .icon_color(Color::Accent)
                            .tooltip(ui::Tooltip::text("Send channel message"))
                            .on_click(move |_, _, cx| {
                                retry_transport
                                    .update(cx, |transport, cx| {
                                        transport.run_composer_action(cx, |transport, cx| {
                                            if transport.pending.is_empty() {
                                                transport.submit(cx).map_err(anyhow::Error::msg)
                                            } else {
                                                transport.retry(cx)
                                            }
                                        });
                                    })
                                    .log_err();
                            }),
                    )
                    .child(
                        IconButton::new("load-earlier-channel-messages", IconName::ArrowUp)
                            .aria_label("Load earlier messages")
                            .style(ButtonStyle::Subtle)
                            .tooltip(ui::Tooltip::text("Load earlier messages"))
                            .on_click(move |_, _, cx| {
                                page_transport
                                    .update(cx, |transport, cx| transport.load_more(cx))
                                    .log_err();
                            }),
                    ),
            )
            .child(
                div()
                    .id("channel-composer-status")
                    .role(if status_is_error {
                        Role::Alert
                    } else {
                        Role::Status
                    })
                    .child(Label::new(self.status_text()).size(LabelSize::Small).color(
                        if status_is_error {
                            Color::Error
                        } else {
                            Color::Muted
                        },
                    )),
            )
    }
}

fn apply_page(
    active: &mut ActiveChannel,
    request_cursor: Option<String>,
    page: proto::CollaborativeMessagePage,
    pending: &BTreeMap<String, PendingOperation>,
    cx: &mut App,
) -> Result<()> {
    validate_page(active, &page)?;
    let next_cursor_token = page
        .next_cursor
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    let entries = page
        .messages
        .iter()
        .map(|record| record_to_entry_with_origin(record, pending))
        .collect::<Result<Vec<_>>>()?;
    let is_head = request_cursor.is_none();
    active.timeline.update(cx, |timeline, cx| {
        let page = MessageTimelinePage {
            request_cursor,
            next_cursor: next_cursor_token.clone(),
            entries,
        };
        if is_head {
            timeline.replace_history_head(page, cx)
        } else {
            timeline.apply_history_page(page, cx)
        }
    })?;
    for (index, record) in page.messages.iter().enumerate() {
        remember_record(active, record, index == 0 && active.latest_target.is_none());
    }
    active.next_cursor = page.next_cursor;
    active.next_cursor_token = next_cursor_token;
    // A head only covers a window. Once live entries exist, its watermark could
    // suppress missed edits to retained rows outside that window.
    if is_head && !active.history_head_loaded {
        if active.authoritative_outbox_cursor == 0 {
            active.authoritative_outbox_cursor = page.authoritative_outbox_cursor;
        }
        active.history_head_loaded = true;
    }
    Ok(())
}

fn remember_record(
    active: &mut ActiveChannel,
    record: &proto::CollaborativeMessageRecord,
    prefer_as_latest: bool,
) {
    let mut target = MessageTarget {
        message_id: record.message_id.clone(),
        source_event_id: record.source_event_id.clone(),
        version: record.version,
        reaction_version: record.reaction_version,
    };
    for current in [
        &active.latest_target,
        &active.latest_own_target,
        &active.reply_target,
    ]
    .into_iter()
    .flatten()
    {
        if current.message_id == target.message_id {
            target.version = target.version.max(current.version);
            target.reaction_version = target.reaction_version.max(current.reaction_version);
        }
    }
    if prefer_as_latest {
        active.latest_target = Some(target.clone());
    }
    if active
        .principal_id
        .as_ref()
        .is_some_and(|principal_id| principal_id == &record.author_principal_id)
    {
        if prefer_as_latest || active.latest_own_target.is_none() {
            active.latest_own_target = Some(target);
        }
    }
}

fn record_to_entry(record: &proto::CollaborativeMessageRecord) -> Result<MessageTimelineEntry> {
    ensure!(
        record.community_id.len() == 16
            && record.channel_id.len() == 16
            && record.message_id.len() == 16
            && record.author_principal_id.len() == 16,
        "invalid channel message identity"
    );
    ensure!(
        record.source_event_id.len() == 32
            && record.current_event_id.len() == 32
            && (record.reply_to_event_id.is_empty() || record.reply_to_event_id.len() == 32),
        "invalid channel event identity"
    );
    ensure!(record.version > 0, "invalid channel message version");
    for reaction in &record.reactions {
        ensure!(
            reaction.actor_principal_id.len() == 16 && reaction.source_event_id.len() == 32,
            "invalid channel reaction identity"
        );
    }
    let timestamp =
        i64::try_from(record.created_at).context("channel message timestamp overflow")?;
    let occurred_at = DateTime::<Utc>::from_timestamp(timestamp, 0)
        .context("channel message timestamp is outside the supported range")?;
    let mut reactions = BTreeMap::<String, u32>::new();
    for reaction in &record.reactions {
        *reactions.entry(reaction.value.clone()).or_default() += 1;
    }
    Ok(MessageTimelineEntry {
        event_id: hex::encode(&record.source_event_id),
        operation_id: None,
        source_version: record.outbox_sequence.max(1),
        author: MessageTimelineAuthor {
            kind: MessageTimelineAuthorKind::Human,
            id: hex::encode(&record.author_principal_id),
            label: record.author_display_name.clone(),
        },
        content: record.body.clone(),
        reply_to: (!record.reply_to_event_id.is_empty())
            .then(|| hex::encode(&record.reply_to_event_id)),
        edited: record.edited,
        deleted: record.deleted,
        reactions: reactions
            .into_iter()
            .map(|(value, count)| MessageTimelineReaction { value, count })
            .collect(),
        occurred_at,
        projected_at: occurred_at,
        context: MessageTimelineContext {
            community_id: Some(hex::encode(&record.community_id)),
            project_id: Some(hex::encode(&record.channel_id)),
            thread_id: (!record.reply_to_event_id.is_empty())
                .then(|| hex::encode(&record.reply_to_event_id)),
        },
    })
}

fn record_to_entry_with_origin(
    record: &proto::CollaborativeMessageRecord,
    pending: &BTreeMap<String, PendingOperation>,
) -> Result<MessageTimelineEntry> {
    let mut entry = record_to_entry(record)?;
    for (operation_id, operation) in pending {
        if !operation.optimistic || operation.request.message_id != record.message_id {
            continue;
        }
        ensure!(
            operation.request.community_id == record.community_id
                && operation.request.channel_id == record.channel_id
                && operation
                    .request
                    .signed_event
                    .as_ref()
                    .is_some_and(|signed| { signed.claimed_event_id == record.source_event_id }),
            "authoritative message does not match its optimistic source event"
        );
        // Mutation receipts identify edits/reactions too. Only a locally matched
        // signed create owns the optimistic timeline entry for this source event.
        entry.operation_id = Some(operation_id.clone());
        break;
    }
    Ok(entry)
}

fn timeline_context(active: &ActiveChannel) -> MessageTimelineContext {
    MessageTimelineContext {
        community_id: Some(active.community_id.to_string()),
        project_id: Some(active.channel_id.to_string()),
        thread_id: None,
    }
}

fn signed_event_to_proto(signed: SignedEvent) -> proto::CollaborativeSignedEvent {
    proto::CollaborativeSignedEvent {
        claimed_event_id: signed.claimed_id.as_bytes().to_vec(),
        public_key: signed.event.public_key.as_bytes().to_vec(),
        created_at: signed.event.created_at,
        kind: u32::from(signed.event.kind),
        tags: signed
            .event
            .tags
            .into_iter()
            .map(|values| proto::CollaborativeEventTag { values })
            .collect(),
        content: signed.event.content,
        signature: signed.signature.as_bytes().to_vec(),
    }
}

async fn wait_for_read_acknowledgement(
    response: impl std::future::Future<
        Output = Result<proto::ApplyCollaborativeMessageOperationResponse>,
    >,
    timeout: Task<()>,
) -> Result<proto::ApplyCollaborativeMessageOperationResponse> {
    let response = response.fuse();
    let timeout = timeout.fuse();
    futures::pin_mut!(response, timeout);
    futures::select_biased! {
        response = response => response,
        _ = timeout => Err(anyhow!("read acknowledgement timed out")),
    }
}

fn ensure_contract_version(version: u32) -> Result<()> {
    ensure!(
        version == CONTRACT_VERSION,
        "unsupported channel messaging contract version"
    );
    Ok(())
}

fn validate_record_scope(
    active: &ActiveChannel,
    record: &proto::CollaborativeMessageRecord,
) -> Result<()> {
    ensure!(
        record.community_id == active.community_id.as_uuid().as_bytes()
            && record.channel_id == active.channel_id.as_uuid().as_bytes(),
        "channel message belongs to a different channel"
    );
    Ok(())
}

fn validate_page(active: &ActiveChannel, page: &proto::CollaborativeMessagePage) -> Result<()> {
    ensure!(
        page.messages.len() <= PAGE_SIZE as usize,
        "channel history exceeded the requested page size"
    );
    ensure!(
        page.done == page.next_cursor.is_none(),
        "channel history has inconsistent pagination"
    );
    if let Some(cursor) = &page.next_cursor {
        ensure!(
            cursor.source_event_id.len() == 32,
            "invalid channel history cursor"
        );
    }
    for record in &page.messages {
        validate_record_scope(active, record)?;
        record_to_entry(record)?;
    }
    Ok(())
}

fn ensure_success(error_code: i32) -> Result<()> {
    if error_code == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone as i32 {
        Ok(())
    } else {
        Err(anyhow!(error_label(error_code)))
    }
}

fn error_label(error_code: i32) -> &'static str {
    match error_code {
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone as i32 =>
        {
            "Ready"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32 =>
        {
            "Access to this channel was denied"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorOffline
                    as i32 =>
        {
            "Offline"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying
                    as i32 =>
        {
            "The server committed the operation and is retrying delivery"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorStaleVersion
                    as i32 =>
        {
            "The message changed on another client"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorUnavailable
                    as i32 =>
        {
            "Channel messaging is temporarily unavailable"
        }
        value
            if value
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest
                    as i32 =>
        {
            "The server rejected the message operation"
        }
        _ => "The server does not support this messaging contract",
    }
}

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;

    #[test]
    fn mutation_receipts_do_not_replace_optimistic_create_ownership() {
        let mut record = proto::CollaborativeMessageRecord {
            community_id: vec![1; 16],
            channel_id: vec![2; 16],
            message_id: vec![3; 16],
            author_principal_id: vec![4; 16],
            source_event_id: vec![5; 32],
            current_event_id: vec![6; 32],
            version: 2,
            accepted_operation_id: "edit-operation".into(),
            body: "edited message".into(),
            outbox_sequence: 3,
            ..Default::default()
        };
        assert_eq!(
            record_to_entry(&record).expect("remote edit").operation_id,
            None
        );
        let pending = BTreeMap::from([(
            "create-operation".into(),
            PendingOperation {
                optimistic: true,
                request: proto::ApplyCollaborativeMessageOperation {
                    community_id: record.community_id.clone(),
                    channel_id: record.channel_id.clone(),
                    message_id: record.message_id.clone(),
                    signed_event: Some(proto::CollaborativeSignedEvent {
                        claimed_event_id: record.source_event_id.clone(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            },
        )]);
        assert_eq!(
            record_to_entry_with_origin(&record, &pending)
                .expect("matched create")
                .operation_id
                .as_deref(),
            Some("create-operation")
        );
        record.accepted_operation_id = "reaction-operation".into();
        assert_eq!(
            record_to_entry_with_origin(&record, &pending)
                .expect("reaction preserves origin")
                .operation_id
                .as_deref(),
            Some("create-operation")
        );
        record.source_event_id = vec![6; 32];
        assert!(record_to_entry_with_origin(&record, &pending).is_err());
    }

    #[gpui::test]
    async fn collaborative_channel_sessions_do_not_retarget_other_workspaces(
        cx: &mut TestAppContext,
    ) {
        let app_state = cx.update(|cx| {
            let app_state = workspace::AppState::test(cx);
            cx.set_global(db::AppDatabase::test_new());
            editor::init(cx);
            channel::init(&app_state.client, app_state.user_store.clone(), cx);
            app_state
        });
        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let first = cx.add_window(|window, cx| {
            Workspace::new(None, project.clone(), app_state.clone(), window, cx)
        });
        let second = cx.add_window(|window, cx| {
            Workspace::new(None, project.clone(), app_state.clone(), window, cx)
        });
        let user = Arc::new(User {
            legacy_id: 1,
            username: "test-user".into(),
            avatar_uri: "".into(),
            name: None,
        });
        let mut providers = Vec::new();
        for (window, channel_id) in [(first, 42), (second, 43)] {
            let workspace = window
                .update(cx, |_, _, cx| cx.entity())
                .expect("workspace exists");
            let views = window
                .update(cx, |_, window, cx| {
                    ChannelMessagingTransport::open(
                        Arc::new(Channel {
                            id: client::ChannelId(channel_id),
                            name: format!("channel-{channel_id}").into(),
                            visibility: proto::ChannelVisibility::Members,
                            parent_path: Vec::new(),
                            channel_order: 0,
                        }),
                        user.clone(),
                        app_state.client.clone(),
                        project.clone(),
                        workspace,
                        window,
                        cx,
                    )
                    .expect("channel view opens")
                })
                .expect("window exists");
            providers.push(views.composer);
        }
        second
            .update(cx, |_, window, cx| {
                let hub = cx.global::<GlobalChannelMessaging>().0.clone();
                let sessions = hub
                    .read(cx)
                    .sessions
                    .values()
                    .map(|(_, _, session)| session.upgrade().expect("provider retains its session"))
                    .collect::<Vec<_>>();
                assert_eq!(sessions.len(), 2);
                assert_ne!(sessions[0].entity_id(), sessions[1].entity_id());
                assert_ne!(
                    sessions[0]
                        .read(cx)
                        .active
                        .as_ref()
                        .map(|active| active.channel_id),
                    sessions[1]
                        .read(cx)
                        .active
                        .as_ref()
                        .map(|active| active.channel_id)
                );
                let second_channel = sessions[1]
                    .read(cx)
                    .active
                    .as_ref()
                    .expect("second channel active")
                    .channel_id;
                sessions[0].update(cx, |session, cx| {
                    let active = session.active.as_ref().expect("first channel active");
                    let record = proto::CollaborativeMessageRecord {
                        community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                        message_id: vec![3; 16],
                        author_principal_id: vec![4; 16],
                        source_event_id: vec![5; 32],
                        current_event_id: vec![5; 32],
                        author_display_name: "Author".into(),
                        body: "live message before the first history response".into(),
                        created_at: 1_900_000_001,
                        version: 1,
                        outbox_sequence: 11,
                        ..Default::default()
                    };
                    session.apply_stream_update(proto::CollaborativeMessageStreamUpdate {
                        contract_version: CONTRACT_VERSION,
                        community_id: record.community_id.clone(),
                        channel_id: record.channel_id.clone(),
                        outbox_sequence: 11,
                        message: Some(record.clone()),
                        operation_kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate.into(),
                        ..Default::default()
                    }, cx).expect("live delivery precedes initial head");
                    let active = session.active.as_mut().expect("first channel active");
                    apply_page(active, None, proto::CollaborativeMessagePage {
                        messages: vec![proto::CollaborativeMessageRecord {
                            message_id: vec![7; 16],
                            source_event_id: vec![8; 32],
                            current_event_id: vec![8; 32],
                            body: "newer head message".into(),
                            created_at: 1_900_000_100,
                            outbox_sequence: 99,
                            ..record.clone()
                        }],
                        next_cursor: Some(proto::CollaborativeMessageCursor {
                            source_event_id: vec![8; 32],
                            snapshot_micros: 1_900_000_200_000_000,
                            message_created_at: 1_900_000_100,
                            outbox_sequence: 99,
                        }),
                        authoritative_outbox_cursor: 99,
                        ..Default::default()
                    }, &BTreeMap::new(), cx).expect("initial head excludes retained older live message");
                    assert_eq!(active.authoritative_outbox_cursor, 11);
                    session.apply_stream_update(proto::CollaborativeMessageStreamUpdate {
                        contract_version: CONTRACT_VERSION,
                        community_id: record.community_id.clone(),
                        channel_id: record.channel_id.clone(),
                        outbox_sequence: 12,
                        message: Some(proto::CollaborativeMessageRecord {
                            body: "retained message updated below head watermark".into(),
                            current_event_id: vec![6; 32],
                            version: 2,
                            outbox_sequence: 12,
                            ..record
                        }),
                        operation_kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit.into(),
                        ..Default::default()
                    }, cx).expect("missed edit survives initial head ordering");
                    let active = session.active.as_ref().expect("first channel active");
                    let timeline = active.timeline.read(cx).timeline();
                    assert_eq!(active.authoritative_outbox_cursor, 12);
                    assert!(timeline.read(cx).items().iter().any(|item| {
                        item.object.id.as_deref() == Some(hex::encode([5; 32]).as_str())
                            && item.object.label == "retained message updated below head watermark"
                    }));
                });
                sessions[0].update(cx, |session, cx| session.close_active(cx));
                assert_eq!(hub.read(cx).sessions.len(), 1);
                assert_eq!(
                    sessions[1]
                        .read(cx)
                        .active
                        .as_ref()
                        .map(|active| active.channel_id),
                    Some(second_channel)
                );
                assert_eq!(
                    providers[0].submit(cx),
                    Err(CollaborativeComposerActionError::ThreadUnavailable)
                );
                sessions[1].update(cx, |session, cx| {
                    let active = session.active.as_mut().expect("active channel");
                    let before = active.authoritative_outbox_cursor;
                    let malformed = proto::CollaborativeMessagePage {
                        messages: vec![proto::CollaborativeMessageRecord::default()],
                        done: true,
                        authoritative_outbox_cursor: 99,
                        ..Default::default()
                    };
                    assert!(apply_page(active, None, malformed, &BTreeMap::new(), cx).is_err());
                    assert_eq!(active.authoritative_outbox_cursor, before);
                    assert!(active.latest_target.is_none());
                    assert!(active.next_cursor.is_none());
                    let oversized = proto::CollaborativeMessagePage {
                        messages: vec![
                            proto::CollaborativeMessageRecord::default();
                            PAGE_SIZE as usize + 1
                        ],
                        done: true,
                        authoritative_outbox_cursor: 100,
                        ..Default::default()
                    };
                    assert!(
                        apply_page(active, None, oversized, &BTreeMap::new(), cx)
                            .expect_err("oversized history rejected")
                            .to_string()
                            .contains("page size")
                    );
                    let inconsistent = proto::CollaborativeMessagePage::default();
                    assert!(
                        apply_page(active, None, inconsistent, &BTreeMap::new(), cx)
                            .expect_err("missing continuation rejected")
                            .to_string()
                            .contains("pagination")
                    );
                    assert_eq!(active.authoritative_outbox_cursor, before);
                    let empty = proto::CollaborativeMessagePage {
                        done: true,
                        authoritative_outbox_cursor: 10,
                        ..Default::default()
                    };
                    assert!(apply_page(active, None, empty, &BTreeMap::new(), cx).is_ok());
                    let cursor = proto::CollaborativeMessageCursor {
                        source_event_id: vec![7; 32],
                        snapshot_micros: 1_900_000_000_000_000,
                        message_created_at: 1_900_000_000,
                        outbox_sequence: 10,
                    };
                    let cursor_token = serde_json::to_string(&cursor).expect("cursor token");
                    apply_page(
                        active,
                        None,
                        proto::CollaborativeMessagePage {
                            next_cursor: Some(cursor),
                            authoritative_outbox_cursor: 10,
                            ..Default::default()
                        },
                        &BTreeMap::new(),
                        cx,
                    )
                    .expect("head establishes the live cursor");
                    apply_page(
                        active,
                        Some(cursor_token),
                        proto::CollaborativeMessagePage {
                            done: true,
                            authoritative_outbox_cursor: 99,
                            ..Default::default()
                        },
                        &BTreeMap::new(),
                        cx,
                    )
                    .expect("older page loads");
                    assert_eq!(active.authoritative_outbox_cursor, 10);
                    let record = proto::CollaborativeMessageRecord {
                        community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                        message_id: vec![3; 16],
                        author_principal_id: vec![4; 16],
                        source_event_id: vec![5; 32],
                        current_event_id: vec![5; 32],
                        author_display_name: "Author".into(),
                        body: "arrived while older history was loading".into(),
                        created_at: 1_900_000_001,
                        version: 1,
                        outbox_sequence: 11,
                        ..Default::default()
                    };
                    session
                        .apply_stream_update(
                            proto::CollaborativeMessageStreamUpdate {
                                contract_version: CONTRACT_VERSION,
                                community_id: record.community_id.clone(),
                                channel_id: record.channel_id.clone(),
                                outbox_sequence: 11,
                                message: Some(record.clone()),
                                operation_kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate.into(),
                                ..Default::default()
                            },
                            cx,
                        )
                        .expect("history watermark must not suppress the live message");
                    let active = session.active.as_mut().expect("active channel");
                    assert_eq!(active.authoritative_outbox_cursor, 11);
                    assert_eq!(
                        active.latest_target.as_ref().expect("live message applied").message_id,
                        vec![3; 16]
                    );
                    apply_page(
                        active,
                        None,
                        proto::CollaborativeMessagePage {
                            done: true,
                            authoritative_outbox_cursor: 5,
                            ..Default::default()
                        },
                        &BTreeMap::new(),
                        cx,
                    )
                    .expect("late head response");
                    assert_eq!(active.authoritative_outbox_cursor, 11);
                    apply_page(
                        active,
                        None,
                        proto::CollaborativeMessagePage {
                            messages: vec![proto::CollaborativeMessageRecord {
                                message_id: vec![7; 16],
                                source_event_id: vec![8; 32],
                                current_event_id: vec![8; 32],
                                body: "newer reconnect head message".into(),
                                created_at: 1_900_000_100,
                                outbox_sequence: 99,
                                ..record.clone()
                            }],
                            next_cursor: Some(proto::CollaborativeMessageCursor {
                                source_event_id: vec![8; 32],
                                snapshot_micros: 1_900_000_200_000_000,
                                message_created_at: 1_900_000_100,
                                outbox_sequence: 99,
                            }),
                            authoritative_outbox_cursor: 99,
                            ..Default::default()
                        },
                        &BTreeMap::new(),
                        cx,
                    )
                    .expect("reconnect head omits the older retained live message");
                    assert_eq!(active.authoritative_outbox_cursor, 11);
                    session
                        .apply_stream_update(
                            proto::CollaborativeMessageStreamUpdate {
                                contract_version: CONTRACT_VERSION,
                                community_id: record.community_id.clone(),
                                channel_id: record.channel_id.clone(),
                                outbox_sequence: 12,
                                message: Some(proto::CollaborativeMessageRecord {
                                    body: "edited while disconnected".into(),
                                    current_event_id: vec![6; 32],
                                    version: 2,
                                    outbox_sequence: 12,
                                    ..record
                                }),
                                operation_kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit.into(),
                                ..Default::default()
                            },
                            cx,
                        )
                        .expect("replay below the refreshed head watermark must update retained content");
                    let active = session.active.as_ref().expect("active channel");
                    assert_eq!(active.authoritative_outbox_cursor, 12);
                    assert_eq!(active.latest_target.as_ref().expect("edited target").version, 2);
                    let timeline = active.timeline.read(cx).timeline();
                    assert!(timeline.read(cx).items().iter().any(|item| {
                        item.object.id.as_deref() == Some(hex::encode([5; 32]).as_str())
                            && item.object.label == "edited while disconnected"
                    }));
                    let composer = active.composer.clone();
                    composer.read(cx).editor.clone().update(cx, |editor, cx| {
                        editor.set_text("retain this draft", window, cx);
                    });
                    let status_frame = proto::CollaborativeMessageStreamUpdate {
                        contract_version: CONTRACT_VERSION,
                        community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                        status: Some(proto::CollaborativeMessageStreamStatus {
                            error_code: proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying as i32,
                            retry_after_millis: 1000,
                        }),
                        ..Default::default()
                    };
                    session.apply_stream_update(status_frame.clone(), cx).expect("scoped recovery status");
                    session.set_status("Read state synchronized", cx);
                    assert!(composer.read(cx).status_text().contains("retrying automatically"));
                    assert_eq!(session.active.as_ref().expect("active").authoritative_outbox_cursor, 12);
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "retain this draft");
                    let mut malformed = status_frame.clone();
                    malformed.outbox_sequence = 99;
                    assert!(session.apply_stream_update(malformed, cx).is_err());
                    assert_eq!(session.active.as_ref().expect("active").authoritative_outbox_cursor, 12);
                    let mut foreign = status_frame.clone();
                    foreign.channel_id = vec![99; 16];
                    foreign.status = Some(proto::CollaborativeMessageStreamStatus::default());
                    session.apply_stream_update(foreign, cx).expect("foreign status ignored");
                    assert!(composer.read(cx).delivery_error.is_some());
                    let mut denied = status_frame.clone();
                    denied.status = Some(proto::CollaborativeMessageStreamStatus {
                        error_code: proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32,
                        retry_after_millis: 0,
                    });
                    session.apply_stream_update(denied, cx).expect("access revoked status");
                    assert!(composer.read(cx).status_text().contains("access revoked"));
                    let mut recovered = status_frame;
                    recovered.status = Some(proto::CollaborativeMessageStreamStatus::default());
                    session.apply_stream_update(recovered.clone(), cx).expect("recovered status");
                    assert!(composer.read(cx).delivery_error.is_none());
                    assert!(composer.read(cx).status_text().contains("delivery recovered"));
                    session.set_error("Message was not confirmed — retained for retry", cx);
                    session.apply_stream_update(recovered, cx).expect("duplicate recovered status");
                    assert!(composer.read(cx).status_is_error, "delivery recovery must preserve an independent message error");
                    assert!(composer.read(cx).status_text().contains("retained for retry"));
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "retain this draft");
                    let active = session.active.as_ref().expect("active channel");
                    assert_eq!(active.authoritative_outbox_cursor, 12);
                    assert!(timeline.read(cx).items().iter().any(|item| item.object.label == "edited while disconnected"));
                    let acknowledgement = proto::CollaborativeMessageStreamUpdate {
                        contract_version: CONTRACT_VERSION,
                        community_id: active.community_id.as_uuid().as_bytes().to_vec(),
                        channel_id: active.channel_id.as_uuid().as_bytes().to_vec(),
                        outbox_sequence: 13,
                        operation_kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge.into(),
                        actor_principal_id: vec![4; 16],
                        acknowledged_outbox_sequence: 12,
                        ..Default::default()
                    };
                    let acknowledgement_record = proto::CollaborativeMessageRecord {
                        community_id: acknowledgement.community_id.clone(),
                        channel_id: acknowledgement.channel_id.clone(),
                        message_id: vec![9; 16],
                        author_principal_id: vec![4; 16],
                        source_event_id: vec![9; 32],
                        current_event_id: vec![9; 32],
                        created_at: 1,
                        version: 1,
                        body: "read receipt must not replace timeline content".into(),
                        ..Default::default()
                    };
                    let mut malformed_acknowledgements = Vec::new();
                    for actor_length in [0, 15, 17] {
                        let mut malformed = acknowledgement.clone();
                        malformed.actor_principal_id = vec![4; actor_length];
                        malformed_acknowledgements.push(malformed);
                    }
                    for sequence in [0, u64::MAX] {
                        let mut malformed = acknowledgement.clone();
                        malformed.acknowledged_outbox_sequence = sequence;
                        malformed_acknowledgements.push(malformed);
                    }
                    let mut malformed = acknowledgement.clone();
                    malformed.message = Some(proto::CollaborativeMessageRecord::default());
                    malformed_acknowledgements.push(malformed);
                    let mut malformed = acknowledgement.clone();
                    malformed.message = Some(proto::CollaborativeMessageRecord {
                        created_at: u64::MAX,
                        ..acknowledgement_record.clone()
                    });
                    malformed_acknowledgements.push(malformed);
                    for malformed in malformed_acknowledgements {
                        assert!(session.apply_stream_update(malformed, cx).is_err(), "malformed acknowledgement must not advance replay");
                        assert_eq!(session.active.as_ref().expect("active channel").authoritative_outbox_cursor, 12);
                        assert!(composer.read(cx).status_is_error);
                        assert_eq!(composer.read(cx).editor.read(cx).text(cx), "retain this draft");
                    }
                    session.apply_stream_update(acknowledgement.clone(), cx).expect("valid acknowledgement advances replay");
                    assert_eq!(session.active.as_ref().expect("active channel").authoritative_outbox_cursor, 13);
                    session.apply_stream_update(proto::CollaborativeMessageStreamUpdate {
                        outbox_sequence: 14,
                        message: Some(acknowledgement_record),
                        ..acknowledgement
                    }, cx).expect("valid optional acknowledgement projection");
                    let active = session.active.as_ref().expect("active channel");
                    assert_eq!(active.authoritative_outbox_cursor, 14);
                    assert!(!timeline.read(cx).items().iter().any(|item| item.object.label == "read receipt must not replace timeline content"));
                    assert_eq!(active.latest_target.as_ref().expect("edited target retained").version, 2);
                    assert!(composer.read(cx).status_is_error, "read-state synchronization must preserve an independent message error");
                    assert!(composer.read(cx).status_text().contains("retained for retry"));
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "retain this draft");
                    let operation_id = OperationId::new().to_string();
                    active.timeline.update(cx, |timeline, cx| {
                        timeline
                            .begin_optimistic(
                                OptimisticMessage {
                                    operation_id: operation_id.clone(),
                                    author: MessageTimelineAuthor {
                                        kind: MessageTimelineAuthorKind::Human,
                                        id: "test-user".into(),
                                        label: "test-user".into(),
                                    },
                                    content: "retain this draft".into(),
                                    reply_to: None,
                                    occurred_at: Utc::now(),
                                    context: timeline_context(active),
                                },
                                cx,
                            )
                            .expect("optimistic message starts");
                    });
                    session.pending.insert(
                        operation_id.clone(),
                        PendingOperation {
                            request: proto::ApplyCollaborativeMessageOperation::default(),
                            optimistic: true,
                        },
                    );
                    session
                        .finish_operation(
                            &operation_id,
                            Ok(proto::ApplyCollaborativeMessageOperationResponse {
                                contract_version: CONTRACT_VERSION + 1,
                                accepted: true,
                                ..Default::default()
                            }),
                            cx,
                        )
                        .expect("invalid response becomes a visible retry state");
                    assert!(
                        session
                            .status
                            .contains("unsupported channel messaging contract version")
                    );
                    assert!(session.pending.contains_key(&operation_id));
                    assert_eq!(
                        composer.read(cx).editor.read(cx).text(cx),
                        "retain this draft"
                    );
                    session.pending.get_mut(&operation_id).expect("pending create").request.body = "submitted draft".into();
                    session.clear_delivered_draft(&operation_id, cx);
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "retain this draft");
                    composer.read(cx).editor.clone().update(cx, |editor, cx| {
                        editor.set_text("submitted draft", window, cx);
                    });
                    session.active.as_mut().expect("active").reply_target = Some(MessageTarget {
                        message_id: vec![8; 16],
                        source_event_id: vec![9; 32],
                        version: 1,
                        reaction_version: 0,
                    });
                    session.clear_delivered_draft(&operation_id, cx);
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "submitted draft");
                    assert!(session.active.as_ref().expect("active").reply_target.is_some());
                    session.active.as_mut().expect("active").reply_target = None;
                    session.clear_delivered_draft(&operation_id, cx);
                    assert_eq!(composer.read(cx).editor.read(cx).text(cx), "");
                    session.run_composer_action(cx, empty_action);
                    assert_eq!(session.status, "replacement text is empty");
                    let composer = &session.active.as_ref().expect("active channel").composer;
                    assert_eq!(composer.read(cx).status, "replacement text is empty");
                    assert!(composer.read(cx).status_is_error);
                    let composer = composer.clone();
                    session.read_acknowledgement = None;
                    session.desired_read_sequence = 0;
                    session.acknowledge(100, cx);
                    let first_request = session.read_acknowledgement.as_ref().expect("first read acknowledgement").request.clone();
                    session.acknowledge(200, cx);
                    assert_eq!(session.read_acknowledgement.as_ref().expect("one in flight").request, first_request);
                    assert_eq!(session.desired_read_sequence, 200);
                    let generation = session.active.as_ref().expect("active").generation;
                    session.finish_read_acknowledgement(generation, &first_request.operation_id, Err(anyhow!("offline")), cx);
                    assert!(composer.read(cx).status_text().contains("Read state not synchronized"));
                    assert!(composer.read(cx).status_is_error, "independent composer failure is retained");
                    assert!(session.read_acknowledgement.as_ref().expect("retry retained").task.is_none());
                    session.acknowledge(300, cx);
                    assert!(session.read_acknowledgement.as_ref().expect("no implicit retry storm").task.is_none());
                    session.set_status("Unrelated operation delivered", cx);
                    assert!(composer.read(cx).status_text().contains("Read state not synchronized"));
                    session.start_read_acknowledgement(cx);
                    assert_eq!(session.read_acknowledgement.as_ref().expect("same retry identity").request, first_request);
                    let accepted = proto::ApplyCollaborativeMessageOperationResponse {
                        contract_version: CONTRACT_VERSION,
                        accepted: true,
                        ..Default::default()
                    };
                    session.finish_read_acknowledgement(generation, &first_request.operation_id, Ok(proto::ApplyCollaborativeMessageOperationResponse { accepted: false, ..accepted.clone() }), cx);
                    assert!(composer.read(cx).read_error.is_some(), "error-code zero alone does not acknowledge a read");
                    session.start_read_acknowledgement(cx);
                    session.finish_read_acknowledgement(generation, &first_request.operation_id, Ok(accepted.clone()), cx);
                    let next_request = session.read_acknowledgement.as_ref().expect("coalesced read").request.clone();
                    assert_eq!(next_request.acknowledged_outbox_sequence, 300);
                    assert_ne!(next_request.operation_id, first_request.operation_id);
                    session.finish_read_acknowledgement(generation, &first_request.operation_id, Err(anyhow!("late response")), cx);
                    assert!(composer.read(cx).read_error.is_none());
                    assert_eq!(session.read_acknowledgement.as_ref().expect("next read unchanged").request, next_request);
                    session.finish_read_acknowledgement(generation, &next_request.operation_id, Ok(accepted.clone()), cx);
                    assert!(session.read_acknowledgement.is_none());
                    assert_eq!(session.desired_read_sequence, 0);
                    session.acknowledge(400, cx);
                    session.close_active(cx);
                    assert!(session.read_acknowledgement.is_none());
                    assert_eq!(session.desired_read_sequence, 0);
                    session.finish_read_acknowledgement(generation, &next_request.operation_id, Ok(accepted), cx);
                    assert_eq!(session.status, "Channel closed — select it again to reconnect");
                });
            })
            .expect("second window exists");
    }

    #[gpui::test]
    async fn read_acknowledgement_deadline_releases_pending_response(cx: &mut TestAppContext) {
        use std::sync::atomic::{AtomicBool, Ordering};

        let dropped = Arc::new(AtomicBool::new(false));
        let cleanup = util::defer({
            let dropped = dropped.clone();
            move || dropped.store(true, Ordering::SeqCst)
        });
        let response = async move {
            let _cleanup = cleanup;
            futures::future::pending().await
        };
        let started = cx.background_executor.now();
        let result = wait_for_read_acknowledgement(
            response,
            cx.background_executor.timer(READ_ACKNOWLEDGEMENT_TIMEOUT),
        )
        .await;
        assert!(
            result
                .expect_err("deadline must expire")
                .to_string()
                .contains("timed out")
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "timed-out request must be cancelled"
        );
        assert_eq!(
            cx.background_executor.now().duration_since(started),
            READ_ACKNOWLEDGEMENT_TIMEOUT
        );
        let accepted = proto::ApplyCollaborativeMessageOperationResponse {
            contract_version: CONTRACT_VERSION,
            accepted: true,
            ..Default::default()
        };
        assert_eq!(
            wait_for_read_acknowledgement(
                futures::future::ready(Ok(accepted.clone())),
                Task::ready(())
            )
            .await
            .expect("ready response takes precedence"),
            accepted
        );
    }

    #[test]
    fn channel_records_reject_malformed_identities_and_timestamps() {
        let record = proto::CollaborativeMessageRecord {
            community_id: vec![1; 16],
            channel_id: vec![2; 16],
            message_id: vec![3; 16],
            author_principal_id: vec![4; 16],
            source_event_id: vec![5; 32],
            current_event_id: vec![6; 32],
            created_at: 1,
            version: 1,
            ..Default::default()
        };
        assert!(record_to_entry(&record).is_ok());
        let mut invalid = record.clone();
        invalid.source_event_id.pop();
        assert!(record_to_entry(&invalid).is_err());
        invalid = record.clone();
        invalid.author_principal_id.clear();
        assert!(record_to_entry(&invalid).is_err());
        invalid = record.clone();
        invalid.created_at = u64::MAX;
        assert!(record_to_entry(&invalid).is_err());
        invalid = record;
        invalid
            .reactions
            .push(proto::CollaborativeMessageReaction::default());
        assert!(record_to_entry(&invalid).is_err());
        assert!(ensure_contract_version(CONTRACT_VERSION).is_ok());
        assert!(ensure_contract_version(CONTRACT_VERSION + 1).is_err());
    }

    fn empty_action(
        _: &mut ChannelMessagingTransport,
        _: &mut Context<ChannelMessagingTransport>,
    ) -> Result<()> {
        Err(anyhow!("replacement text is empty"))
    }
}
