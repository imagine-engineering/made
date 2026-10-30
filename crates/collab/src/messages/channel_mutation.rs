use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthorizationAction, AuthorizationScope, Channel,
    ChannelDescription, ChannelLifecycleState, ChannelName, ChannelRecordFields, ChannelType,
    ChannelVisibility, IntegrityAlgorithm, IntegrityReference, Message, MessageAuthor,
    MessageCommandOutcome, MessageContent, MessageCreateFields, MessageDeleteMetadata,
    MessageLifecycleState, MessageMutation, MessageMutationKind, MessageRecordFields,
    MessageSource, NostrEventId, OperationId, PrincipalId, Provenance, ReactionCommandOutcome,
    ReactionMutation, ReactionMutationKind, ReactionRecordFields, ReactionSet, ReactionValue,
    SourceRecordId, SourceSystem,
};
use nostr_compat::generated_kinds::{
    KIND_DELETION, KIND_NIP29_DELETE_EVENT, KIND_REACTION, KIND_STREAM_MESSAGE,
    KIND_STREAM_MESSAGE_EDIT, KIND_STREAM_MESSAGE_V2,
};
use nostr_compat::{EventId, TimestampPolicy, verify_signed_event};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, QueryResult,
    Statement,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{
    collaboration_command::{DomainCommand, DomainCommandSubmissionError},
    db::collaboration::{
        event_repository::{EventRepository, EventStoreOutcome, VerifiedEventRecord},
        outbox::{
            AppliedCommand, CommandFingerprint, OutboxOperation, TransactionalCommandMutation,
        },
        persistence_policy::{EventPersistencePolicy, PrivacyAdmission},
    },
    identity::profile_repository::IdentityProfileRepository,
    messages::channel_admission::{
        AuthorizedChannel, channel_scope, current_message_admission_in_transaction,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOperationKind {
    Create,
    Edit,
    Delete,
    ReactionAdd,
    ReactionRemove,
    Acknowledge,
}

pub struct MessageOperation {
    pub authorization: AuthorizedChannel,
    pub kind: MessageOperationKind,
    pub message_id: AggregateId,
    pub expected_version: Option<AggregateVersion>,
    pub signed_event: Option<VerifiedEventRecord>,
    pub reaction: Option<String>,
    pub related_reaction_event_id: Option<NostrEventId>,
    pub acknowledged_outbox_sequence: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MessageOutboxPayload {
    pub operation_id: OperationId,
    pub channel_id: AggregateId,
    pub message_id: AggregateId,
    pub kind: MessageOperationKind,
    pub source_event_id: Option<NostrEventId>,
    #[serde(default)]
    pub actor_principal_id: Option<PrincipalId>,
    #[serde(default)]
    pub acknowledged_outbox_sequence: Option<u64>,
}

pub struct CanonicalMessageMutation {
    event_repository: EventRepository,
    profiles: IdentityProfileRepository,
}

impl CanonicalMessageMutation {
    pub fn new(connection: DatabaseConnection) -> Result<Self, DomainCommandSubmissionError> {
        if !matches!(
            &connection,
            DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(DomainCommandSubmissionError::Unavailable);
        }
        Ok(Self {
            profiles: IdentityProfileRepository::new(DatabaseConnection::from(
                connection.get_postgres_connection_pool().clone(),
            ))
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
            event_repository: EventRepository::new(connection)
                .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        })
    }
}

#[async_trait]
impl TransactionalCommandMutation<MessageOperation> for CanonicalMessageMutation {
    fn transaction_isolation(&self) -> Option<sea_orm::IsolationLevel> {
        // Waiting for current policy locks must refresh subsequent membership and restriction reads.
        Some(sea_orm::IsolationLevel::ReadCommitted)
    }

    fn fingerprint(
        &self,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<CommandFingerprint, DomainCommandSubmissionError> {
        let payload = command.payload();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            payload
                .authorization
                .tenant
                .community_id()
                .as_uuid()
                .as_bytes(),
        );
        bytes.extend_from_slice(payload.authorization.channel_id.as_uuid().as_bytes());
        bytes.extend_from_slice(payload.message_id.as_uuid().as_bytes());
        bytes.extend_from_slice(format!("{:?}", payload.kind).as_bytes());
        if let Some(version) = payload.expected_version {
            bytes.extend_from_slice(&version.get().to_be_bytes());
        }
        if let Some(event) = &payload.signed_event {
            bytes.extend_from_slice(
                &event
                    .signed_event()
                    .event
                    .canonical_bytes()
                    .map_err(|_| DomainCommandSubmissionError::Rejected)?,
            );
            bytes.extend_from_slice(event.signed_event().signature.as_bytes());
        }
        if let Some(reaction) = &payload.reaction {
            bytes.extend_from_slice(reaction.as_bytes());
        }
        if let Some(event_id) = payload.related_reaction_event_id {
            bytes.extend_from_slice(event_id.as_bytes());
        }
        if let Some(sequence) = payload.acknowledged_outbox_sequence {
            bytes.extend_from_slice(&sequence.to_be_bytes());
        }
        CommandFingerprint::new(command_kind(payload.kind), &bytes)
    }

    async fn admit_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<(), DomainCommandSubmissionError> {
        let payload = command.payload();
        if payload.authorization.tenant.community_id() != command.tenant().community_id()
            || payload.authorization.principal != *command.principal()
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let current = current_message_admission_in_transaction(
            transaction,
            &self.profiles,
            command.tenant(),
            command.principal(),
            payload.authorization.channel_id,
            now_millis(),
        )
        .await
        .map_err(message_admission_error)?;
        if current.authorization.community_membership != payload.authorization.community_membership
            || current.authorization.channel_membership != payload.authorization.channel_membership
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let action = if payload.kind == MessageOperationKind::Acknowledge {
            AuthorizationAction::Read
        } else {
            AuthorizationAction::Write
        };
        current
            .authorize(action, now_millis())
            .map_err(|_| DomainCommandSubmissionError::Rejected)?;
        if let (Some(key), Some(source)) = (
            current.authorization.signing_public_key,
            payload.signed_event.as_ref(),
        ) {
            if source.signed_event().event.public_key.as_bytes() != &key {
                return Err(DomainCommandSubmissionError::Rejected);
            }
        }
        Ok(())
    }

    async fn apply(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AppliedCommand, DomainCommandSubmissionError> {
        let payload = command.payload();
        if payload.authorization.tenant.community_id() != command.tenant().community_id()
            || payload.authorization.principal != *command.principal()
            || payload.authorization.channel_id.as_uuid().is_nil()
            || payload.message_id.as_uuid().is_nil()
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let authoritative_version = match payload.kind {
            MessageOperationKind::Create => self.create_message(transaction, command).await?,
            MessageOperationKind::Edit => self.edit_message(transaction, command).await?,
            MessageOperationKind::Delete => self.delete_message(transaction, command).await?,
            MessageOperationKind::ReactionAdd | MessageOperationKind::ReactionRemove => {
                self.mutate_reaction(transaction, command).await?
            }
            MessageOperationKind::Acknowledge => self.acknowledge(transaction, command).await?,
        };
        let source_event_id = payload
            .signed_event
            .as_ref()
            .map(|event| domain_event_id(event.signed_event().claimed_id));
        let outbox_payload = MessageOutboxPayload {
            operation_id: command.operation_id(),
            channel_id: payload.authorization.channel_id,
            message_id: payload.message_id,
            kind: payload.kind,
            source_event_id,
            actor_principal_id: Some(command.principal().principal_id()),
            acknowledged_outbox_sequence: payload.acknowledged_outbox_sequence,
        };
        let encoded = serde_json::to_vec(&outbox_payload)
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let source_record = source_event_id
            .map(|event_id| hex::encode(event_id.as_bytes()))
            .unwrap_or_else(|| command.operation_id().to_string());
        let source_record_id =
            SourceRecordId::new(source_record).ok_or(DomainCommandSubmissionError::Rejected)?;
        let provenance = Provenance::new(SourceSystem::Zed, source_record_id, now_millis())
            .with_source_version(authoritative_version.to_string())
            .with_integrity(IntegrityReference {
                algorithm: if source_event_id.is_some() {
                    IntegrityAlgorithm::NostrEventId
                } else {
                    IntegrityAlgorithm::Sha256
                },
                value: if let Some(source_event_id) = source_event_id {
                    hex::encode(source_event_id.as_bytes())
                } else {
                    hex::encode(Sha256::digest(&encoded))
                },
            });
        let topic = channel_topic(payload.authorization.channel_id);
        let outbox = OutboxOperation::new(topic, provenance, encoded)?;
        Ok(AppliedCommand::new(authoritative_version, outbox))
    }
}

impl CanonicalMessageMutation {
    async fn create_message(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AggregateVersion, DomainCommandSubmissionError> {
        let payload = command.payload();
        if payload.expected_version.is_some() {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let event = require_event(payload, &[KIND_STREAM_MESSAGE, KIND_STREAM_MESSAGE_V2])?;
        let event_id = domain_event_id(event.signed_event().claimed_id);
        require_channel_tag(event, payload.authorization.channel_id)?;
        let channel = load_channel(transaction, &payload.authorization).await?;
        let scope = write_scope(&payload.authorization)?;
        let authorization = payload.authorization.message_authorization_request(
            payload.message_id,
            command.principal().principal_id(),
            AuthorizationAction::Write,
            &scope,
            now_millis(),
        );
        let admitted = current_message_admission_in_transaction(
            transaction,
            &self.profiles,
            command.tenant(),
            command.principal(),
            payload.authorization.channel_id,
            now_millis(),
        )
        .await
        .map_err(message_admission_error)?;
        let author = admitted
            .message_author
            .unwrap_or_else(|| MessageAuthor::principal(command.principal().principal_id()));
        let message = Message::create(
            MessageCreateFields {
                community_id: command.tenant().community_id(),
                channel_id: payload.authorization.channel_id,
                message_id: payload.message_id,
                author,
                content: MessageContent::new(event.signed_event().event.content.clone())
                    .map_err(|_| DomainCommandSubmissionError::Rejected)?,
                source: MessageSource {
                    event_id,
                    event_created_at: event.signed_event().event.created_at,
                },
            },
            &channel,
            &authorization,
        )
        .map_err(|_| DomainCommandSubmissionError::Rejected)?;
        store_event(&self.event_repository, transaction, command, event).await?;
        let attribution = admitted
            .agent_author
            .as_ref()
            .map(|author| {
                author
                    .attribution(
                        admitted
                            .owner_membership
                            .ok_or(DomainCommandSubmissionError::Unavailable)?,
                        admitted.authorization.channel_membership,
                    )
                    .map_err(|_| DomainCommandSubmissionError::Unavailable)
            })
            .transpose()?;
        let result = transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_messages (
    community_id, message_id, channel_id, source_event_id, current_event_id,
    author_principal_id, message_created_at, lifecycle_state, message_version,
    source_system, source_record_id, source_version, source_observed_at,
    integrity_algorithm, integrity_value, agent_author_profile_id,
    agent_author_owner_principal_id, agent_author_proof_event_id, agent_author_attribution
) VALUES ($1, $2, $3, $4, $4, $5, CAST($6 AS numeric), 'active', 1,
    'zed', $7, '1', clock_timestamp(), 'nostr_event_id', $7, $8, $9, $10, $11)
ON CONFLICT (community_id, source_event_id) DO NOTHING
"#,
                [
                    command.tenant().community_id().as_uuid().into(),
                    message.fields().message_id.as_uuid().into(),
                    message.fields().channel_id.as_uuid().into(),
                    event_id.as_bytes().to_vec().into(),
                    command.principal().principal_id().as_uuid().into(),
                    message.fields().source.event_created_at.to_string().into(),
                    hex::encode(event_id.as_bytes()).into(),
                    admitted
                        .agent_author
                        .as_ref()
                        .map(|author| author.profile_id().as_uuid())
                        .into(),
                    author.owner_principal_id().map(|id| id.as_uuid()).into(),
                    admitted
                        .agent_author
                        .as_ref()
                        .map(|author| author.proof_event_id().as_bytes().to_vec())
                        .into(),
                    attribution.into(),
                ],
            ))
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        if result.rows_affected() != 1 {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        Ok(AggregateVersion::FIRST)
    }

    async fn edit_message(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AggregateVersion, DomainCommandSubmissionError> {
        let payload = command.payload();
        let event = require_event(payload, &[KIND_STREAM_MESSAGE_EDIT])?;
        let mut message = load_message(
            &self.event_repository,
            &self.profiles,
            transaction,
            command,
            payload.message_id,
        )
        .await?;
        require_target_tag(event, message.fields().source.event_id)?;
        let expected_version = payload
            .expected_version
            .ok_or(DomainCommandSubmissionError::Rejected)?;
        // Exact retries return their outbox receipt before this fresh-operation check.
        if message.fields().version != expected_version {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let source = MessageSource {
            event_id: domain_event_id(event.signed_event().claimed_id),
            event_created_at: event.signed_event().event.created_at,
        };
        let scope = write_scope(&payload.authorization)?;
        let authorization = payload.authorization.message_authorization_request(
            payload.message_id,
            message.fields().author.principal_id(),
            AuthorizationAction::Write,
            &scope,
            now_millis(),
        );
        if message
            .edit(
                expected_version,
                MessageContent::new(event.signed_event().event.content.clone())
                    .map_err(|_| DomainCommandSubmissionError::Rejected)?,
                source,
                &authorization,
            )
            .map_err(|_| DomainCommandSubmissionError::Rejected)?
            == MessageCommandOutcome::Unchanged
        {
            return Ok(message.fields().version);
        }
        store_event(&self.event_repository, transaction, command, event).await?;
        insert_auxiliary(
            transaction,
            command,
            &message,
            source,
            "edit",
            false,
            None,
            None,
        )
        .await?;
        update_message_projection(transaction, command, &message).await?;
        Ok(message.fields().version)
    }

    async fn delete_message(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AggregateVersion, DomainCommandSubmissionError> {
        let payload = command.payload();
        let event = require_event(payload, &[KIND_DELETION, KIND_NIP29_DELETE_EVENT])?;
        let signed = &event.signed_event().event;
        if signed
            .tags
            .iter()
            .filter(|tag| tag.first().is_some_and(|name| name == "e" || name == "a"))
            .count()
            != 1
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        if u32::from(signed.kind) == KIND_NIP29_DELETE_EVENT {
            let signed_channel = signed
                .tags
                .iter()
                .filter(|tag| tag.first().is_some_and(|name| name == "h"))
                .filter_map(|tag| {
                    tag.get(1)
                        .and_then(|value| uuid::Uuid::parse_str(value).ok())
                })
                .next();
            if signed_channel != Some(payload.authorization.channel_id.as_uuid()) {
                return Err(DomainCommandSubmissionError::Rejected);
            }
        }
        let metadata = deletion_metadata(signed.kind, &signed.tags)?;
        if load_channel(transaction, &payload.authorization)
            .await?
            .fields()
            .lifecycle_state
            == ChannelLifecycleState::Archived
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let mut message = load_message(
            &self.event_repository,
            &self.profiles,
            transaction,
            command,
            payload.message_id,
        )
        .await?;
        require_target_tag(event, message.fields().source.event_id)?;
        let expected_version = payload
            .expected_version
            .ok_or(DomainCommandSubmissionError::Rejected)?;
        // Exact retries return their outbox receipt before this fresh-operation check.
        if message.fields().version != expected_version {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let source = MessageSource {
            event_id: domain_event_id(event.signed_event().claimed_id),
            event_created_at: event.signed_event().event.created_at,
        };
        let scope = write_scope(&payload.authorization)?;
        let author = message.fields().author;
        let actor = command.principal().principal_id();
        let author_delete =
            actor == author.principal_id() || author.owner_principal_id() == Some(actor);
        if u32::from(signed.kind) == KIND_DELETION && !author_delete {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let action = if author_delete && metadata.is_none() {
            AuthorizationAction::Write
        } else {
            AuthorizationAction::Manage
        };
        let authorization = payload.authorization.message_authorization_request(
            payload.message_id,
            message.fields().author.principal_id(),
            action,
            &scope,
            now_millis(),
        );
        if message
            .delete(expected_version, source, metadata, &authorization)
            .map_err(|_| DomainCommandSubmissionError::Rejected)?
            == MessageCommandOutcome::Unchanged
        {
            return Ok(message.fields().version);
        }
        store_event(&self.event_repository, transaction, command, event).await?;
        insert_auxiliary(
            transaction,
            command,
            &message,
            source,
            "delete",
            true,
            None,
            None,
        )
        .await?;
        update_message_projection(transaction, command, &message).await?;
        Ok(message.fields().version)
    }

    async fn mutate_reaction(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AggregateVersion, DomainCommandSubmissionError> {
        let payload = command.payload();
        let event = require_event(payload, &[KIND_REACTION])?;
        let message = load_message(
            &self.event_repository,
            &self.profiles,
            transaction,
            command,
            payload.message_id,
        )
        .await?;
        require_target_tag(event, message.fields().source.event_id)?;
        let mut reactions = load_reactions(transaction, command, &message).await?;
        let expected_version = payload
            .expected_version
            .ok_or(DomainCommandSubmissionError::Rejected)?;
        let value = ReactionValue::new(
            payload
                .reaction
                .clone()
                .ok_or(DomainCommandSubmissionError::Rejected)?,
        )
        .map_err(|_| DomainCommandSubmissionError::Rejected)?;
        let source = MessageSource {
            event_id: domain_event_id(event.signed_event().claimed_id),
            event_created_at: event.signed_event().event.created_at,
        };
        let scope = write_scope(&payload.authorization)?;
        let authorization = payload.authorization.message_authorization_request(
            payload.message_id,
            message.fields().author.principal_id(),
            AuthorizationAction::Write,
            &scope,
            now_millis(),
        );
        let outcome = match payload.kind {
            MessageOperationKind::ReactionAdd => reactions.add(
                expected_version,
                value.clone(),
                source,
                &message,
                &authorization,
            ),
            MessageOperationKind::ReactionRemove => reactions.remove(
                expected_version,
                value.clone(),
                payload
                    .related_reaction_event_id
                    .ok_or(DomainCommandSubmissionError::Rejected)?,
                source,
                &message,
                &authorization,
            ),
            _ => return Err(DomainCommandSubmissionError::Rejected),
        }
        .map_err(|_| DomainCommandSubmissionError::Rejected)?;
        if outcome == ReactionCommandOutcome::Unchanged {
            return Ok(reactions.fields().version);
        }
        store_event(&self.event_repository, transaction, command, event).await?;
        let (kind, tombstone, related) = match payload.kind {
            MessageOperationKind::ReactionAdd => ("reaction_add", false, None),
            MessageOperationKind::ReactionRemove => {
                ("reaction_remove", true, payload.related_reaction_event_id)
            }
            _ => return Err(DomainCommandSubmissionError::Rejected),
        };
        insert_auxiliary(
            transaction,
            command,
            &message,
            source,
            kind,
            tombstone,
            Some(value.as_str()),
            related,
        )
        .await?;
        Ok(reactions.fields().version)
    }

    async fn acknowledge(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<MessageOperation>,
    ) -> Result<AggregateVersion, DomainCommandSubmissionError> {
        let payload = command.payload();
        if payload.signed_event.is_some() {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let sequence = payload
            .acknowledged_outbox_sequence
            .ok_or(DomainCommandSubmissionError::Rejected)?;
        let database_sequence = i64::try_from(sequence)
            .ok()
            .filter(|sequence| *sequence > 0)
            .ok_or(DomainCommandSubmissionError::Rejected)?;
        let head = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT COALESCE(max(outbox_sequence), 0)::bigint AS sequence FROM public.collaboration_outbox WHERE community_id = $1 AND topic = $2",
                [
                    command.tenant().community_id().as_uuid().into(),
                    channel_topic(payload.authorization.channel_id).into(),
                ],
            ))
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?
            .ok_or(DomainCommandSubmissionError::Unavailable)?;
        let head = head
            .try_get::<i64>("", "sequence")
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        if database_sequence > head {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let result = transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_message_read_states (
    community_id, channel_id, principal_id, last_outbox_sequence, operation_id, agent_profile_id
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (community_id, channel_id, principal_id) DO UPDATE SET
    last_outbox_sequence = GREATEST(
        public.collaboration_message_read_states.last_outbox_sequence,
        EXCLUDED.last_outbox_sequence
    ),
    operation_id = EXCLUDED.operation_id,
    agent_profile_id = EXCLUDED.agent_profile_id,
    updated_at = clock_timestamp()
"#,
                [
                    command.tenant().community_id().as_uuid().into(),
                    payload.authorization.channel_id.as_uuid().into(),
                    command.principal().principal_id().as_uuid().into(),
                    database_sequence.into(),
                    command.operation_id().as_uuid().into(),
                    match command.principal().kind() {
                        collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent {
                            profile_id,
                            ..
                        } => Some(profile_id.as_uuid()),
                        _ => None,
                    }
                    .into(),
                ],
            ))
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        if result.rows_affected() != 1 {
            return Err(DomainCommandSubmissionError::Unavailable);
        }
        AggregateVersion::new(sequence).ok_or(DomainCommandSubmissionError::Rejected)
    }
}

async fn store_event(
    repository: &EventRepository,
    transaction: &DatabaseTransaction,
    command: &DomainCommand<MessageOperation>,
    event: &VerifiedEventRecord,
) -> Result<(), DomainCommandSubmissionError> {
    let persistence = EventPersistencePolicy::evaluate(
        event.signed_event().event.kind,
        PrivacyAdmission::community(),
    )
    .map_err(|_| DomainCommandSubmissionError::Rejected)?;
    match repository
        .store_in_transaction(transaction, command.tenant(), event, persistence)
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?
    {
        EventStoreOutcome::Inserted | EventStoreOutcome::Duplicate => Ok(()),
        EventStoreOutcome::Stale | EventStoreOutcome::EphemeralNotPersisted => {
            Err(DomainCommandSubmissionError::Rejected)
        }
    }
}

async fn load_channel(
    transaction: &DatabaseTransaction,
    authorization: &AuthorizedChannel,
) -> Result<Channel, DomainCommandSubmissionError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT name, channel_type, visibility, lifecycle_state, description,
       creator_principal_id, channel_version::text AS channel_version
FROM public.collaboration_channels
WHERE community_id = $1 AND channel_id = $2
"#,
            [
                authorization.tenant.community_id().as_uuid().into(),
                authorization.channel_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?
        .ok_or(DomainCommandSubmissionError::Rejected)?;
    Channel::from_record(ChannelRecordFields {
        community_id: authorization.tenant.community_id(),
        channel_id: authorization.channel_id,
        name: ChannelName::new(row_string(&row, "name")?)
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        channel_type: match row_string(&row, "channel_type")?.as_str() {
            "stream" => ChannelType::Stream,
            "forum" => ChannelType::Forum,
            "dm" => ChannelType::DirectMessage,
            "workflow" => ChannelType::Workflow,
            "ephemeral" => ChannelType::Ephemeral,
            "huddle" => ChannelType::Huddle,
            _ => return Err(DomainCommandSubmissionError::Unavailable),
        },
        visibility: match row_string(&row, "visibility")?.as_str() {
            "open" => ChannelVisibility::Open,
            "private" => ChannelVisibility::Private,
            _ => return Err(DomainCommandSubmissionError::Unavailable),
        },
        lifecycle_state: match row_string(&row, "lifecycle_state")?.as_str() {
            "active" => ChannelLifecycleState::Active,
            "archived" => ChannelLifecycleState::Archived,
            "deleted" => ChannelLifecycleState::Deleted,
            "expired" => ChannelLifecycleState::Expired,
            _ => return Err(DomainCommandSubmissionError::Unavailable),
        },
        description: row
            .try_get::<Option<String>>("", "description")
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?
            .map(ChannelDescription::new)
            .transpose()
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        creator_principal_id: PrincipalId::from_uuid(
            row.try_get("", "creator_principal_id")
                .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        ),
        expiration: None,
        version: version_from_row(&row, "channel_version")?,
    })
    .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

async fn load_message(
    event_repository: &EventRepository,
    profiles: &IdentityProfileRepository,
    transaction: &DatabaseTransaction,
    command: &DomainCommand<MessageOperation>,
    message_id: AggregateId,
) -> Result<Message, DomainCommandSubmissionError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT m.channel_id, m.source_event_id, m.current_event_id, m.author_principal_id,
       m.agent_author_profile_id, m.agent_author_owner_principal_id,
       m.agent_author_proof_event_id, m.agent_author_attribution,
       m.message_created_at::text AS message_created_at,
       m.lifecycle_state, m.message_version::text AS message_version,
       current_event.content, current_event.event_created_at::text AS current_created_at
FROM public.collaboration_messages AS m
JOIN public.collaboration_events AS current_event
  ON current_event.community_id = m.community_id
 AND current_event.event_id = m.current_event_id
WHERE m.community_id = $1 AND m.message_id = $2 AND m.channel_id = $3
FOR UPDATE OF m
"#,
            [
                command.tenant().community_id().as_uuid().into(),
                message_id.as_uuid().into(),
                command.payload().authorization.channel_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?
        .ok_or(DomainCommandSubmissionError::Rejected)?;
    let author_principal_id = PrincipalId::from_uuid(
        row.try_get("", "author_principal_id")
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
    );
    let stored_profile: Option<uuid::Uuid> = row
        .try_get("", "agent_author_profile_id")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let author = if let Some(profile_id) = stored_profile {
        let owner: uuid::Uuid = row
            .try_get("", "agent_author_owner_principal_id")
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let attribution: serde_json::Value = row
            .try_get("", "agent_author_attribution")
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        profiles
            .message_author_from_attribution_in_transaction(
                transaction,
                command.tenant(),
                author_principal_id,
                collaboration_domain::ProfileId::from_uuid(profile_id),
                PrincipalId::from_uuid(owner),
                event_id_from_row(&row, "agent_author_proof_event_id")?,
                event_id_from_row(&row, "source_event_id")?,
                attribution,
            )
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?
    } else {
        MessageAuthor::principal(author_principal_id)
    };
    let source_event_id = event_id_from_row(&row, "source_event_id")?;
    let current_event_id = event_id_from_row(&row, "current_event_id")?;
    let auxiliary = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT auxiliary.auxiliary_event_id, auxiliary.actor_principal_id, auxiliary.auxiliary_kind,
       auxiliary.event_created_at::text AS event_created_at
FROM public.collaboration_message_auxiliary_events AS auxiliary
WHERE auxiliary.community_id = $1 AND auxiliary.channel_id = $2 AND auxiliary.target_message_event_id = $3
  AND auxiliary.auxiliary_kind IN ('edit', 'delete')
ORDER BY auxiliary.event_created_at ASC, auxiliary.auxiliary_event_id ASC
"#,
            [
                command.tenant().community_id().as_uuid().into(),
                command.payload().authorization.channel_id.as_uuid().into(),
                source_event_id.as_bytes().to_vec().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let mut version = AggregateVersion::FIRST;
    let mut mutations = Vec::with_capacity(auxiliary.len());
    for auxiliary_row in auxiliary {
        version = version
            .next()
            .ok_or(DomainCommandSubmissionError::Unavailable)?;
        let actor_principal_id = PrincipalId::from_uuid(
            auxiliary_row
                .try_get("", "actor_principal_id")
                .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        );
        let event_id = event_id_from_row(&auxiliary_row, "auxiliary_event_id")?;
        let source = event_repository
            .exact_in_transaction(
                transaction,
                command.tenant(),
                EventId::from_bytes(*event_id.as_bytes()),
            )
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?
            .ok_or(DomainCommandSubmissionError::Unavailable)?;
        let signed = source.signed_event();
        verify_signed_event(signed, TimestampPolicy::Historical)
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        let created_at = row_u64(&auxiliary_row, "event_created_at")?;
        if signed.event.created_at != created_at {
            return Err(DomainCommandSubmissionError::Unavailable);
        }
        let kind = match row_string(&auxiliary_row, "auxiliary_kind")?.as_str() {
            "edit" if u32::from(signed.event.kind) == KIND_STREAM_MESSAGE_EDIT => {
                MessageMutationKind::Edit
            }
            "delete"
                if matches!(
                    u32::from(signed.event.kind),
                    KIND_DELETION | KIND_NIP29_DELETE_EVENT
                ) =>
            {
                let metadata = deletion_metadata(signed.event.kind, &signed.event.tags)
                    .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
                MessageMutationKind::Delete {
                    moderated: actor_principal_id != author_principal_id || metadata.is_some(),
                    metadata,
                }
            }
            _ => return Err(DomainCommandSubmissionError::Unavailable),
        };
        mutations.push(MessageMutation {
            source: MessageSource {
                event_id,
                event_created_at: created_at,
            },
            actor_principal_id,
            kind,
            resulting_version: version,
        });
    }
    let stored_version = version_from_row(&row, "message_version")?;
    if stored_version != version {
        return Err(DomainCommandSubmissionError::Unavailable);
    }
    Message::from_record(MessageRecordFields {
        community_id: command.tenant().community_id(),
        channel_id: command.payload().authorization.channel_id,
        message_id,
        author,
        content: MessageContent::new(row_string(&row, "content")?)
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
        lifecycle_state: match row_string(&row, "lifecycle_state")?.as_str() {
            "active" => MessageLifecycleState::Active,
            "edited" => MessageLifecycleState::Edited,
            "deleted" => MessageLifecycleState::Deleted,
            _ => return Err(DomainCommandSubmissionError::Unavailable),
        },
        source: MessageSource {
            event_id: source_event_id,
            event_created_at: row_u64(&row, "message_created_at")?,
        },
        current_source: MessageSource {
            event_id: current_event_id,
            event_created_at: row_u64(&row, "current_created_at")?,
        },
        mutations,
        version,
    })
    .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

async fn load_reactions(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<MessageOperation>,
    message: &Message,
) -> Result<ReactionSet, DomainCommandSubmissionError> {
    let rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT auxiliary_event_id, actor_principal_id, auxiliary_kind, related_event_id, emoji,
       event_created_at::text AS event_created_at
FROM public.collaboration_message_auxiliary_events
WHERE community_id = $1 AND channel_id = $2 AND target_message_event_id = $3
  AND auxiliary_kind IN ('reaction_add', 'reaction_remove')
ORDER BY event_created_at ASC, auxiliary_event_id ASC
"#,
            [
                command.tenant().community_id().as_uuid().into(),
                message.fields().channel_id.as_uuid().into(),
                message.fields().source.event_id.as_bytes().to_vec().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let mut version = AggregateVersion::FIRST;
    let mutations = rows
        .into_iter()
        .map(|row| {
            version = version
                .next()
                .ok_or(DomainCommandSubmissionError::Unavailable)?;
            let kind = match row_string(&row, "auxiliary_kind")?.as_str() {
                "reaction_add" => ReactionMutationKind::Add,
                "reaction_remove" => ReactionMutationKind::Remove {
                    added_event_id: event_id_from_row(&row, "related_event_id")?,
                },
                _ => return Err(DomainCommandSubmissionError::Unavailable),
            };
            Ok(ReactionMutation {
                source: MessageSource {
                    event_id: event_id_from_row(&row, "auxiliary_event_id")?,
                    event_created_at: row_u64(&row, "event_created_at")?,
                },
                actor_principal_id: PrincipalId::from_uuid(
                    row.try_get("", "actor_principal_id")
                        .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
                ),
                value: ReactionValue::new(row_string(&row, "emoji")?)
                    .map_err(|_| DomainCommandSubmissionError::Unavailable)?,
                kind,
                resulting_version: version,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ReactionSet::from_record(ReactionRecordFields {
        community_id: message.fields().community_id,
        channel_id: message.fields().channel_id,
        message_id: message.fields().message_id,
        target_message_event_id: message.fields().source.event_id,
        mutations,
        version,
    })
    .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

async fn insert_auxiliary(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<MessageOperation>,
    message: &Message,
    source: MessageSource,
    kind: &str,
    tombstone: bool,
    emoji: Option<&str>,
    related_event_id: Option<NostrEventId>,
) -> Result<(), DomainCommandSubmissionError> {
    let event_id_hex = hex::encode(source.event_id.as_bytes());
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
INSERT INTO public.collaboration_message_auxiliary_events (
    community_id, auxiliary_event_id, channel_id, target_message_event_id,
    actor_principal_id, auxiliary_kind, related_event_id, emoji,
    event_created_at, is_tombstone, source_system, source_record_id,
    source_version, source_observed_at, integrity_algorithm, integrity_value, agent_actor_profile_id
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CAST($9 AS numeric), $10,
    'zed', $11, $12, clock_timestamp(), 'nostr_event_id', $11, $13)
ON CONFLICT (community_id, auxiliary_event_id) DO NOTHING
"#,
            [
                command.tenant().community_id().as_uuid().into(),
                source.event_id.as_bytes().to_vec().into(),
                message.fields().channel_id.as_uuid().into(),
                message.fields().source.event_id.as_bytes().to_vec().into(),
                command.principal().principal_id().as_uuid().into(),
                kind.into(),
                related_event_id
                    .map(|event_id| event_id.as_bytes().to_vec())
                    .into(),
                emoji.map(ToOwned::to_owned).into(),
                source.event_created_at.to_string().into(),
                tombstone.into(),
                event_id_hex.into(),
                message.fields().version.to_string().into(),
                match command.principal().kind() {
                    collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent {
                        profile_id,
                        ..
                    } => Some(profile_id.as_uuid()),
                    _ => None,
                }
                .into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    if result.rows_affected() != 1 {
        return Err(DomainCommandSubmissionError::Rejected);
    }
    Ok(())
}

async fn update_message_projection(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<MessageOperation>,
    message: &Message,
) -> Result<(), DomainCommandSubmissionError> {
    let fields = message.fields();
    let (lifecycle, deleted_event) = match fields.lifecycle_state {
        MessageLifecycleState::Active => ("active", None),
        MessageLifecycleState::Edited => ("edited", None),
        MessageLifecycleState::Deleted => (
            "deleted",
            Some(fields.current_source.event_id.as_bytes().to_vec()),
        ),
    };
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
UPDATE public.collaboration_messages
SET current_event_id = $4,
    deleted_by_event_id = $5,
    lifecycle_state = $6,
    message_version = CAST($7 AS numeric),
    source_version = $7,
    projected_at = clock_timestamp()
WHERE community_id = $1 AND message_id = $2 AND channel_id = $3
"#,
            [
                command.tenant().community_id().as_uuid().into(),
                fields.message_id.as_uuid().into(),
                fields.channel_id.as_uuid().into(),
                fields.current_source.event_id.as_bytes().to_vec().into(),
                deleted_event.into(),
                lifecycle.into(),
                fields.version.to_string().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    if result.rows_affected() != 1 {
        return Err(DomainCommandSubmissionError::Unavailable);
    }
    Ok(())
}

fn deletion_metadata(
    kind: u16,
    tags: &[Vec<String>],
) -> Result<Option<MessageDeleteMetadata>, DomainCommandSubmissionError> {
    if u32::from(kind) != KIND_NIP29_DELETE_EVENT {
        return Ok(None);
    }
    let value = |name: &str| {
        tags.iter().find_map(|tag| {
            (tag.first().is_some_and(|item| item == name))
                .then(|| tag.get(1).cloned())
                .flatten()
        })
    };
    let action_id = value("action_id")
        .map(|value| uuid::Uuid::parse_str(&value).map(AggregateId::from_uuid))
        .transpose()
        .map_err(|_| DomainCommandSubmissionError::Rejected)?;
    let metadata =
        MessageDeleteMetadata::new(action_id, value("reason_code"), value("public_reason"))
            .map_err(|_| DomainCommandSubmissionError::Rejected)?;
    Ok((!metadata.is_empty()).then_some(metadata))
}

fn require_event<'a>(
    payload: &'a MessageOperation,
    kinds: &[u32],
) -> Result<&'a VerifiedEventRecord, DomainCommandSubmissionError> {
    let event = payload
        .signed_event
        .as_ref()
        .ok_or(DomainCommandSubmissionError::Rejected)?;
    if !kinds.contains(&u32::from(event.signed_event().event.kind))
        || event.community_id() != payload.authorization.tenant.community_id()
    {
        return Err(DomainCommandSubmissionError::Rejected);
    }
    Ok(event)
}

fn require_channel_tag(
    event: &VerifiedEventRecord,
    channel_id: AggregateId,
) -> Result<(), DomainCommandSubmissionError> {
    require_tag(event, "h", &channel_id.to_string())
}

fn require_target_tag(
    event: &VerifiedEventRecord,
    target_event_id: NostrEventId,
) -> Result<(), DomainCommandSubmissionError> {
    require_tag(event, "e", &hex::encode(target_event_id.as_bytes()))
}

fn require_tag(
    event: &VerifiedEventRecord,
    name: &str,
    value: &str,
) -> Result<(), DomainCommandSubmissionError> {
    event
        .signed_event()
        .event
        .tags
        .iter()
        .any(|tag| {
            tag.first().is_some_and(|item| item == name)
                && tag.get(1).is_some_and(|item| item == value)
        })
        .then_some(())
        .ok_or(DomainCommandSubmissionError::Rejected)
}

fn write_scope(
    authorization: &AuthorizedChannel,
) -> Result<AuthorizationScope, DomainCommandSubmissionError> {
    channel_scope(&authorization.principal, AuthorizationAction::Write)
        .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

fn message_admission_error(error: anyhow::Error) -> DomainCommandSubmissionError {
    if error.downcast_ref::<sea_orm::DbErr>().is_some()
        || error.downcast_ref::<crate::admin::moderation::ModerationRestrictionReadError>().is_some()
        || matches!(error.downcast_ref::<crate::identity::profile_repository::IdentityProfileRepositoryError>(),
            Some(crate::identity::profile_repository::IdentityProfileRepositoryError::Unavailable(_)
                | crate::identity::profile_repository::IdentityProfileRepositoryError::EventRepository(_)
                | crate::identity::profile_repository::IdentityProfileRepositoryError::InvalidRecord)) {
        DomainCommandSubmissionError::Unavailable
    } else { DomainCommandSubmissionError::Rejected }
}

fn domain_event_id(event_id: nostr_compat::EventId) -> NostrEventId {
    NostrEventId::from_bytes(*event_id.as_bytes())
}

pub fn channel_topic(channel_id: AggregateId) -> String {
    format!("channel:{channel_id}")
}

fn command_kind(kind: MessageOperationKind) -> &'static str {
    match kind {
        MessageOperationKind::Create => "collaborative_message.create.v1",
        MessageOperationKind::Edit => "collaborative_message.edit.v1",
        MessageOperationKind::Delete => "collaborative_message.delete.v1",
        MessageOperationKind::ReactionAdd => "collaborative_message.reaction_add.v1",
        MessageOperationKind::ReactionRemove => "collaborative_message.reaction_remove.v1",
        MessageOperationKind::Acknowledge => "collaborative_message.acknowledge.v1",
    }
}

fn event_id_from_row(
    row: &QueryResult,
    column: &str,
) -> Result<NostrEventId, DomainCommandSubmissionError> {
    let bytes: Vec<u8> = row
        .try_get("", column)
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let bytes =
        <[u8; 32]>::try_from(bytes).map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    Ok(NostrEventId::from_bytes(bytes))
}

fn row_string(row: &QueryResult, column: &str) -> Result<String, DomainCommandSubmissionError> {
    row.try_get("", column)
        .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

fn row_u64(row: &QueryResult, column: &str) -> Result<u64, DomainCommandSubmissionError> {
    row_string(row, column)?
        .parse()
        .map_err(|_| DomainCommandSubmissionError::Unavailable)
}

fn version_from_row(
    row: &QueryResult,
    column: &str,
) -> Result<AggregateVersion, DomainCommandSubmissionError> {
    AggregateVersion::new(row_u64(row, column)?).ok_or(DomainCommandSubmissionError::Unavailable)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}
