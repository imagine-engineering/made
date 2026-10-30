use collaboration_domain::{
    AggregateId, CommunityId, OperationId, TenantContext, TrustedTenantRoute,
    channel_id_for_legacy_channel, community_id_for_legacy_root_channel,
};
use futures::FutureExt as _;
use gpui::{AppContext as _, BackgroundExecutor, TestAppContext};
use nostr_compat::{CanonicalEvent, EventSignature, PublicKey, SignedEvent};
use rpc::{RECEIVE_TIMEOUT, TypedEnvelope, proto};
use sea_orm::{ConnectionTrait as _, DatabaseConnection, Statement, TransactionTrait as _};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use sqlx::migrate::MigrateDatabase as _;
use uuid::Uuid;

use crate::{TestServer, db_tests::run_database_migrations};
use collab::{
    pubsub::{
        envelope::FanoutEnvelope,
        postgres::PostgresFanoutReplayStore,
        redis::RedisFanoutTransport,
        subscription_bus::{FanoutReplayStore as _, FanoutTransport as _, SubscriptionBusError},
    },
    rpc::RECONNECT_TIMEOUT,
};

struct UpdateReceiver;

struct TestSigner {
    secret: SecretKey,
    public_key: [u8; 32],
}

impl TestSigner {
    fn new(seed: u8) -> Self {
        let mut bytes = [0; 32];
        bytes[31] = seed;
        let secret = SecretKey::from_slice(&bytes).expect("valid deterministic secret");
        let secp = Secp256k1::new();
        let keypair = Keypair::from_secret_key(&secp, &secret);
        let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
        Self {
            secret,
            public_key: public_key.serialize(),
        }
    }

    fn event(
        &self,
        created_at: u64,
        kind: u16,
        tags: Vec<Vec<String>>,
        content: &str,
    ) -> proto::CollaborativeSignedEvent {
        let event = CanonicalEvent::new(
            PublicKey::from_bytes(self.public_key),
            created_at,
            kind,
            tags,
            content.to_owned(),
        );
        let claimed_id = event.event_id().expect("canonical event id");
        let secp = Secp256k1::new();
        let keypair = Keypair::from_secret_key(&secp, &self.secret);
        let signature =
            secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
        let signed = SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string())
                .expect("canonical signature"),
        };
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
}

#[gpui::test]
async fn canonical_channel_messages_round_trip_between_two_clients(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_denied: &mut TestAppContext,
) {
    if std::env::var("CI").is_ok() && !cfg!(target_os = "linux") {
        return;
    }

    let mut server = TestServer::start_postgres(executor.clone()).await;
    let client_a = server.create_client(cx_a, "message_author").await;
    let client_b = server.create_client(cx_b, "message_reader").await;
    let denied_client = server.create_client(cx_denied, "message_outsider").await;
    let channel_id = server
        .make_channel(
            "canonical-messages",
            None,
            (&client_a, cx_a),
            &mut [(&client_b, cx_b)],
        )
        .await;
    executor.run_until_parked();

    let community_id = community_id_for_legacy_root_channel(channel_id.0);
    let canonical_channel_id = channel_id_for_legacy_channel(channel_id.0);
    let signer_a = TestSigner::new(1);
    let signer_b = TestSigner::new(2);

    for (version, community_bytes, page_size, expected) in [
        (
            2,
            community_id.as_uuid().as_bytes().to_vec(),
            100,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorUnsupportedVersion,
        ),
        (
            1,
            vec![1; 15],
            100,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest,
        ),
        (
            1,
            community_id.as_uuid().as_bytes().to_vec(),
            0,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest,
        ),
        (
            1,
            community_id.as_uuid().as_bytes().to_vec(),
            201,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest,
        ),
        (
            1,
            community_id.as_uuid().as_bytes().to_vec(),
            u32::MAX,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest,
        ),
    ] {
        let mut request = open_request(community_id, canonical_channel_id, signer_a.public_key, 0);
        request.contract_version = version;
        request.community_id = community_bytes.clone();
        request.page_size = page_size;
        let rejected = client_a
            .request(request)
            .await
            .expect("invalid open response");
        assert_eq!(rejected.error_code, expected as i32);
        assert!(rejected.principal_id.is_empty());
        assert!(rejected.page.is_none());
        let rejected = client_a
            .request(proto::GetCollaborativeMessageWindow {
                contract_version: version,
                community_id: community_bytes,
                channel_id: canonical_channel_id.as_uuid().as_bytes().to_vec(),
                page_size,
                ..Default::default()
            })
            .await
            .expect("invalid history response");
        assert_eq!(rejected.error_code, expected as i32);
        assert!(rejected.page.is_none());
    }

    let (updates_tx, updates_rx) = async_channel::bounded(32);
    let update_receiver = cx_b.new(|_| UpdateReceiver);
    let _update_subscription = client_b.add_message_handler(
        update_receiver.downgrade(),
        move |_, envelope: TypedEnvelope<proto::CollaborativeMessageStreamUpdate>, _| {
            let updates_tx = updates_tx.clone();
            async move {
                updates_tx.send(envelope.payload).await?;
                anyhow::Ok(())
            }
        },
    );

    let open_a = client_a
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer_a.public_key,
            0,
        ))
        .await
        .expect("author opens channel");
    assert_eq!(
        open_a.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone as i32
    );
    let open_b = client_b
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer_b.public_key,
            0,
        ))
        .await
        .expect("reader opens channel");
    assert_eq!(
        open_b.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone as i32
    );

    assert_channel_status(
        next_channel_update(&updates_rx, &executor).await,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
    );

    let denied = denied_client
        .request(open_request(
            community_id,
            canonical_channel_id,
            TestSigner::new(3).public_key,
            0,
        ))
        .await
        .expect("denied open has typed response");
    assert_eq!(
        denied.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32
    );
    let cross_tenant = client_a
        .request(open_request(
            CommunityId::new(),
            canonical_channel_id,
            signer_a.public_key,
            0,
        ))
        .await
        .expect("cross-tenant open has typed response");
    assert_eq!(
        cross_tenant.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32
    );

    let now = unix_time_seconds();
    let message_id = AggregateId::new();
    let create_operation = OperationId::new();
    let create_event = signer_a.event(
        now,
        40_002,
        vec![vec!["h".into(), canonical_channel_id.to_string()]],
        "hello from client A",
    );
    let create_request = operation_request(
        community_id,
        canonical_channel_id,
        message_id,
        create_operation,
        proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
        0,
        "hello from client A",
        "",
        create_event.clone(),
    );
    server
        .test_db
        .db()
        .transaction(|transaction| async move {
            transaction
                .execute(Statement::from_string(
                    sea_orm::DatabaseBackend::Postgres,
                    "ALTER TABLE public.collaboration_outbox ADD CONSTRAINT reject_test_message CHECK (false) NOT VALID".to_owned(),
                ))
                .await?;
            Ok(())
        })
        .await
        .expect("install failure at the final outbox write in the disposable database");
    let interrupted = client_a
        .request(create_request.clone())
        .await
        .expect("failed transaction has a typed response");
    assert!(!interrupted.accepted);
    assert!(!interrupted.duplicate);
    assert!(interrupted.message.is_none());
    assert_eq!(
        interrupted.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorUnavailable as i32
    );
    assert!(updates_rx.try_recv().is_err());
    server
        .test_db
        .db()
        .transaction(|transaction| async move {
            transaction
                .execute(Statement::from_sql_and_values(
                    sea_orm::DatabaseBackend::Postgres,
                    "SELECT set_config('app.community_id', $1, true)",
                    [community_id.to_string().into()],
                ))
                .await?;
            for table in [
                "collaboration_events",
                "collaboration_event_heads",
                "collaboration_messages",
                "collaboration_message_auxiliary_events",
                "collaboration_command_receipts",
                "collaboration_outbox",
            ] {
                let row = transaction
                    .query_one(Statement::from_sql_and_values(
                        sea_orm::DatabaseBackend::Postgres,
                        format!(
                            "SELECT count(*) AS count FROM public.{table} WHERE community_id = $1"
                        ),
                        [community_id.as_uuid().into()],
                    ))
                    .await?
                    .expect("count row");
                assert_eq!(
                    row.try_get::<i64>("", "count")?,
                    0,
                    "partial rows in {table}"
                );
            }
            transaction
                .execute(Statement::from_string(
                    sea_orm::DatabaseBackend::Postgres,
                    "ALTER TABLE public.collaboration_outbox DROP CONSTRAINT reject_test_message"
                        .to_owned(),
                ))
                .await?;
            Ok(())
        })
        .await
        .expect("failed command leaves no event, projection, receipt or outbox rows");
    let created = client_a
        .request(create_request.clone())
        .await
        .expect("create request");
    assert!(
        created.accepted,
        "create rejected with error code {} and authoritative version {}",
        created.error_code, created.authoritative_version
    );
    assert!(!created.duplicate);
    let created_record = created.message.expect("created message record");
    assert_eq!(created_record.body, "hello from client A");

    let duplicate = client_a
        .request(create_request)
        .await
        .expect("idempotent retry");
    assert!(duplicate.accepted);
    assert!(duplicate.duplicate);
    assert_eq!(
        duplicate.message.expect("duplicate record").message_id,
        created_record.message_id
    );

    let delivered = updates_rx.recv().await.expect("reader receives create");
    assert_eq!(delivered.outbox_sequence, created_record.outbox_sequence);
    assert_eq!(
        delivered.message.expect("delivered record").body,
        "hello from client A"
    );

    server.forbid_connections();
    server.disconnect_client(client_b.peer_id().expect("reader peer id"));
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert!(!client_b.status().borrow().is_connected());

    let edited = client_a
        .request(operation_request(
            community_id,
            canonical_channel_id,
            message_id,
            OperationId::new(),
            proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit,
            created_record.version,
            "edited by client A",
            "",
            signer_a.event(
                now + 1,
                40_003,
                vec![vec![
                    "e".into(),
                    hex::encode(&created_record.source_event_id),
                ]],
                "edited by client A",
            ),
        ))
        .await
        .expect("edit request");
    assert!(edited.accepted);
    let edited_record = edited.message.expect("edited message record");
    assert_eq!(edited_record.body, "edited by client A");
    assert_eq!(edited_record.version, created_record.version + 1);

    server.allow_connections();
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert!(client_b.status().borrow().is_connected());
    let recovered_after_disconnect = client_b
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer_b.public_key,
            created_record.outbox_sequence,
        ))
        .await
        .expect("reader reopens channel from authoritative cursor");
    let recovered_page = recovered_after_disconnect.page.expect("recovered page");
    assert_eq!(recovered_page.messages.len(), 1);
    assert_eq!(recovered_page.messages[0].body, "edited by client A");
    assert_eq!(
        recovered_page.messages[0].message_id,
        created_record.message_id
    );

    let stale = client_a
        .request(operation_request(
            community_id,
            canonical_channel_id,
            message_id,
            OperationId::new(),
            proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit,
            created_record.version,
            "stale edit",
            "",
            signer_a.event(
                now + 2,
                40_003,
                vec![vec![
                    "e".into(),
                    hex::encode(&created_record.source_event_id),
                ]],
                "stale edit",
            ),
        ))
        .await
        .expect("stale edit response");
    assert!(!stale.accepted);
    assert_eq!(
        stale.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorStaleVersion as i32
    );

    let reacted = client_b
        .request(operation_request(
            community_id,
            canonical_channel_id,
            message_id,
            OperationId::new(),
            proto::CollaborativeMessageOperationKind::CollaborativeMessageReactionAdd,
            edited_record.reaction_version,
            "👍",
            "👍",
            signer_b.event(
                now + 3,
                7,
                vec![vec![
                    "e".into(),
                    hex::encode(&created_record.source_event_id),
                ]],
                "👍",
            ),
        ))
        .await
        .expect("reaction request");
    assert!(reacted.accepted);
    let reacted_record = reacted.message.expect("reacted message record");
    assert_eq!(reacted_record.reactions.len(), 1);

    let acknowledgement_request = proto::ApplyCollaborativeMessageOperation {
        contract_version: 1,
        community_id: community_id.as_uuid().as_bytes().to_vec(),
        channel_id: canonical_channel_id.as_uuid().as_bytes().to_vec(),
        message_id: AggregateId::new().as_uuid().as_bytes().to_vec(),
        operation_id: OperationId::new().to_string(),
        kind: proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge.into(),
        expected_version: 0,
        body: String::new(),
        reply_to_event_id: Vec::new(),
        reaction: String::new(),
        related_reaction_event_id: Vec::new(),
        signed_event: None,
        acknowledged_outbox_sequence: reacted_record.outbox_sequence,
    };
    let mut rejected_read_operations = Vec::new();
    for sequence in [
        0,
        reacted_record
            .outbox_sequence
            .checked_add(1)
            .expect("next sequence"),
        i64::MAX as u64,
        u64::MAX,
    ] {
        let operation_id = OperationId::new();
        let rejected = client_b
            .request(proto::ApplyCollaborativeMessageOperation {
                operation_id: operation_id.to_string(),
                acknowledged_outbox_sequence: sequence,
                ..acknowledgement_request.clone()
            })
            .await
            .expect("invalid read cursor response");
        assert!(
            !rejected.accepted,
            "invalid read cursor must not advance read state"
        );
        assert_eq!(
            rejected.error_code,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest as i32
        );
        rejected_read_operations.push(operation_id);
    }
    let acknowledged = client_b
        .request(acknowledgement_request.clone())
        .await
        .expect("acknowledge request");
    assert!(acknowledged.accepted);
    assert_eq!(acknowledged.error_code, 0);
    assert!(acknowledged.message.is_none());
    assert!(!acknowledged.duplicate);
    let repeated_acknowledgement = client_b
        .request(acknowledgement_request)
        .await
        .expect("idempotent acknowledgement retry");
    assert!(repeated_acknowledgement.accepted);
    assert!(repeated_acknowledgement.duplicate);
    assert_eq!(repeated_acknowledgement.error_code, 0);
    assert!(repeated_acknowledgement.message.is_none());
    assert_eq!(
        repeated_acknowledgement.authoritative_version,
        acknowledged.authoritative_version
    );
    let reader_principal = Uuid::from_slice(&open_b.principal_id).expect("reader principal");
    let acknowledged_sequence = reacted_record.outbox_sequence;
    server
        .test_db
        .db()
        .transaction(|transaction| {
            let rejected_read_operations = rejected_read_operations.clone();
            async move {
                transaction
                    .execute(Statement::from_sql_and_values(
                        sea_orm::DatabaseBackend::Postgres,
                        "SELECT set_config('app.community_id', $1, true)",
                        [community_id.to_string().into()],
                    ))
                    .await?;
                let row = transaction
                    .query_one(Statement::from_sql_and_values(
                        sea_orm::DatabaseBackend::Postgres,
                        "SELECT last_outbox_sequence FROM public.collaboration_message_read_states WHERE community_id = $1 AND channel_id = $2 AND principal_id = $3",
                        [community_id.as_uuid().into(), canonical_channel_id.as_uuid().into(), reader_principal.into()],
                    ))
                    .await?
                    .expect("acknowledgement persisted independently of its response");
                assert_eq!(row.try_get::<i64>("", "last_outbox_sequence")?, i64::try_from(acknowledged_sequence).expect("bounded database sequence"));
                for operation_id in rejected_read_operations {
                    for table in ["collaboration_command_receipts", "collaboration_outbox"] {
                        let row = transaction.query_one(Statement::from_sql_and_values(
                            sea_orm::DatabaseBackend::Postgres,
                            format!("SELECT count(*) AS count FROM public.{table} WHERE community_id = $1 AND operation_id = $2"),
                            [community_id.as_uuid().into(), operation_id.as_uuid().into()],
                        )).await?.expect("rejected operation count");
                        assert_eq!(row.try_get::<i64>("", "count")?, 0, "rejected read cursor must not leave a receipt or outbox row");
                    }
                }
                Ok(())
            }
        })
        .await
        .expect("read state survives a separate transaction");

    let history = client_b
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer_b.public_key,
            reacted_record.outbox_sequence,
        ))
        .await
        .expect("history after reconnect");
    let page = history.page.expect("history page");
    assert_eq!(page.messages.len(), 1);
    assert_eq!(page.messages[0].body, "edited by client A");
    assert_eq!(page.messages[0].reactions.len(), 1);

    let deleted = client_a
        .request(operation_request(
            community_id,
            canonical_channel_id,
            message_id,
            OperationId::new(),
            proto::CollaborativeMessageOperationKind::CollaborativeMessageDelete,
            edited_record.version,
            "",
            "",
            signer_a.event(
                now + 4,
                5,
                vec![vec![
                    "e".into(),
                    hex::encode(&created_record.source_event_id),
                ]],
                "",
            ),
        ))
        .await
        .expect("delete request");
    assert!(deleted.accepted);
    assert!(deleted.message.expect("deleted record").deleted);

    client_a
        .channel_store()
        .update(cx_a, |store, cx| {
            store.remove_member(channel_id, client_b.user_id().expect("reader user id"), cx)
        })
        .await
        .expect("remove channel member");
    executor.advance_clock(std::time::Duration::from_secs(2));
    loop {
        let update = next_channel_update(&updates_rx, &executor).await;
        if let Some(status) = &update.status {
            if status.error_code
                == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32
            {
                assert_channel_status(
                    update,
                    proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied,
                );
                break;
            }
            assert_channel_status(
                update,
                proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
            );
        }
    }

    let after_revocation = client_a
        .request(operation_request(
            community_id,
            canonical_channel_id,
            AggregateId::new(),
            OperationId::new(),
            proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
            0,
            "not visible after revocation",
            "",
            signer_a.event(
                now + 5,
                40_002,
                vec![vec!["h".into(), canonical_channel_id.to_string()]],
                "not visible after revocation",
            ),
        ))
        .await
        .expect("authorized member sends after revocation");
    assert!(after_revocation.accepted);
    executor.run_until_parked();
    assert!(
        updates_rx.try_recv().is_err(),
        "removed member subscription must close before message hydration"
    );

    let removed_member = client_b
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer_b.public_key,
            0,
        ))
        .await
        .expect("removed member receives typed denial");
    assert_eq!(
        removed_member.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32
    );
}

#[gpui::test]
async fn canonical_channel_moderator_deletion_preserves_author_and_history(
    executor: BackgroundExecutor,
    cx_moderator: &mut TestAppContext,
    cx_member: &mut TestAppContext,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let moderator = server
        .create_client(cx_moderator, "deletion_moderator")
        .await;
    let member = server.create_client(cx_member, "deletion_member").await;
    let channel = server
        .make_channel(
            "moderator-deletion",
            None,
            (&moderator, cx_moderator),
            &mut [(&member, cx_member)],
        )
        .await;
    executor.run_until_parked();
    let community_id = community_id_for_legacy_root_channel(channel.0);
    let channel_id = channel_id_for_legacy_channel(channel.0);
    let moderator_signer = TestSigner::new(10);
    let member_signer = TestSigner::new(11);
    for (client, signer) in [(&moderator, &moderator_signer), (&member, &member_signer)] {
        let opened = client
            .request(open_request(community_id, channel_id, signer.public_key, 0))
            .await
            .expect("open signed canonical channel");
        assert_eq!(opened.error_code, 0);
    }
    assert_eq!(
        moderator
            .channel_store()
            .read_with(cx_moderator, |store, _| store.channel_role(channel)),
        proto::ChannelRole::Admin
    );
    assert_eq!(
        member
            .channel_store()
            .read_with(cx_member, |store, _| store.channel_role(channel)),
        proto::ChannelRole::Member
    );
    let moderator_message_id = AggregateId::new();
    let member_message_id = AggregateId::new();
    let self_deleted_message_id = AggregateId::new();
    let nip29_self_deleted_message_id = AggregateId::new();
    let created_at = unix_time_seconds();
    let mut records = Vec::new();
    for (client, signer, message_id, body) in [
        (
            &moderator,
            &moderator_signer,
            moderator_message_id,
            "moderator message",
        ),
        (&member, &member_signer, member_message_id, "member message"),
        (
            &member,
            &member_signer,
            self_deleted_message_id,
            "member self deletion",
        ),
        (
            &member,
            &member_signer,
            nip29_self_deleted_message_id,
            "member NIP-29 self deletion",
        ),
    ] {
        let created = client
            .request(operation_request(
                community_id,
                channel_id,
                message_id,
                OperationId::new(),
                proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
                0,
                body,
                "",
                signer.event(
                    created_at,
                    40_002,
                    vec![vec!["h".into(), channel_id.to_string()]],
                    body,
                ),
            ))
            .await
            .expect("create canonical message");
        assert!(created.accepted);
        records.push(created.message.expect("created canonical projection"));
    }
    let [
        moderator_record,
        member_record,
        self_deleted_record,
        nip29_self_deleted_record,
    ]: [_; 4] = records.try_into().expect("four created messages");
    let deletion = |community,
                    message_id,
                    operation_id,
                    version,
                    target: &[u8],
                    signer: &TestSigner,
                    kind: u16,
                    additional_tags: Vec<Vec<String>>| {
        let mut tags = vec![vec!["e".into(), hex::encode(target)]];
        if kind == 9_005 {
            tags.push(vec!["h".into(), channel_id.to_string()]);
        }
        tags.extend(additional_tags);
        operation_request(
            community,
            channel_id,
            message_id,
            operation_id,
            proto::CollaborativeMessageOperationKind::CollaborativeMessageDelete,
            version,
            "",
            "",
            signer.event(created_at + 1, kind, tags, ""),
        )
    };
    let denied_operation = OperationId::new();
    let denied = member
        .request(deletion(
            community_id,
            moderator_message_id,
            denied_operation,
            moderator_record.version,
            &moderator_record.source_event_id,
            &member_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("nonauthor member receives a typed response");
    assert!(!denied.accepted);
    assert!(denied.message.is_none());
    let foreign_operation = OperationId::new();
    let foreign = moderator
        .request(deletion(
            CommunityId::from_uuid(Uuid::new_v4()),
            member_message_id,
            foreign_operation,
            member_record.version,
            &member_record.source_event_id,
            &moderator_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("foreign community receives typed denial");
    assert!(!foreign.accepted);
    assert_eq!(
        foreign.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied as i32
    );
    let stale_operation = OperationId::new();
    let stale = moderator
        .request(deletion(
            community_id,
            member_message_id,
            stale_operation,
            member_record.version + 1,
            &member_record.source_event_id,
            &moderator_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("stale moderator deletion is fenced");
    assert!(!stale.accepted);
    assert_eq!(
        stale.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorStaleVersion as i32
    );
    let mut protocol_denials = Vec::new();
    let target = vec!["e".into(), hex::encode(&member_record.source_event_id)];
    let channel_tag = vec!["h".into(), channel_id.to_string()];
    for (kind, tags, label) in [
        (
            5,
            vec![target.clone()],
            "admin cannot moderate via standard deletion",
        ),
        (
            9_005,
            vec![target.clone()],
            "moderation requires its channel tag",
        ),
        (
            9_005,
            vec![
                target.clone(),
                vec!["h".into(), AggregateId::new().to_string()],
            ],
            "moderation target must belong to signed channel",
        ),
        (
            9_005,
            vec![
                target.clone(),
                vec!["h".into(), AggregateId::new().to_string()],
                channel_tag.clone(),
            ],
            "later channel tag cannot override first valid channel",
        ),
        (
            9_005,
            vec![target.clone(), target.clone(), channel_tag.clone()],
            "deletion cannot cover multiple targets",
        ),
        (
            9_005,
            vec![
                target.clone(),
                vec!["a".into(), "40002:invalid:target".into()],
                channel_tag.clone(),
            ],
            "deletion cannot mix target forms",
        ),
        (
            9_005,
            vec![
                target.clone(),
                channel_tag.clone(),
                vec!["action_id".into(), "invalid".into()],
            ],
            "moderation action identity must be valid",
        ),
    ] {
        let operation_id = OperationId::new();
        let signed_event = moderator_signer.event(created_at + 1, kind, tags, "");
        let event_id = signed_event.claimed_event_id.clone();
        let rejected = moderator
            .request(operation_request(
                community_id,
                channel_id,
                member_message_id,
                operation_id,
                proto::CollaborativeMessageOperationKind::CollaborativeMessageDelete,
                member_record.version,
                "",
                "",
                signed_event,
            ))
            .await
            .expect(label);
        assert!(!rejected.accepted, "{label}");
        assert!(rejected.message.is_none(), "{label}");
        protocol_denials.push((operation_id, event_id));
    }
    let spoof_operation = OperationId::new();
    let spoof = member
        .request(deletion(
            community_id,
            member_message_id,
            spoof_operation,
            member_record.version,
            &member_record.source_event_id,
            &moderator_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("bound signer cannot be borrowed from moderator");
    assert!(!spoof.accepted);
    assert_eq!(
        spoof.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorInvalidRequest as i32
    );
    let metadata_operation = OperationId::new();
    let metadata_denied = member
        .request(deletion(
            community_id,
            self_deleted_message_id,
            metadata_operation,
            self_deleted_record.version,
            &self_deleted_record.source_event_id,
            &member_signer,
            9_005,
            vec![vec!["public_reason".into(), "Moderation".into()]],
        ))
        .await
        .expect("author cannot claim moderator metadata with member role");
    assert!(!metadata_denied.accepted);
    let self_deleted = member
        .request(deletion(
            community_id,
            self_deleted_message_id,
            OperationId::new(),
            self_deleted_record.version,
            &self_deleted_record.source_event_id,
            &member_signer,
            5,
            vec![vec![
                "public_reason".into(),
                "Standard self deletion".into(),
            ]],
        ))
        .await
        .expect("member deletes their own message");
    assert!(self_deleted.accepted);
    assert!(self_deleted.message.expect("self tombstone").deleted);
    let request = deletion(
        community_id,
        member_message_id,
        OperationId::new(),
        member_record.version,
        &member_record.source_event_id,
        &moderator_signer,
        9_005,
        vec![
            vec!["action_id".into(), Uuid::new_v4().to_string()],
            vec!["reason_code".into(), "policy_violation".into()],
            vec!["public_reason".into(), "Removed by a moderator".into()],
        ],
    );
    let moderated = moderator
        .request(request.clone())
        .await
        .expect("moderator deletion response");
    assert!(
        moderated.accepted,
        "authorized moderator must delete another author's message"
    );
    let tombstone = moderated.message.expect("moderated tombstone");
    assert!(tombstone.deleted);
    assert!(tombstone.body.is_empty());
    assert_eq!(tombstone.source_event_id, member_record.source_event_id);
    assert_eq!(
        tombstone.author_principal_id,
        member_record.author_principal_id
    );
    assert_eq!(tombstone.version, member_record.version + 1);
    let retained_request = request.clone();
    let original_author = member_record.author_principal_id.clone();
    let actor_id = moderator_record.author_principal_id.clone();
    let retried = moderator
        .request(request.clone())
        .await
        .expect("retry committed deletion");
    assert!(retried.accepted);
    assert!(retried.duplicate);
    assert_eq!(retried.authoritative_version, tombstone.version);
    let changed_retry = moderator
        .request(proto::ApplyCollaborativeMessageOperation {
            expected_version: tombstone.version,
            ..request.clone()
        })
        .await
        .expect("changed same-operation payload is refused");
    assert!(!changed_retry.accepted);
    assert!(!changed_retry.duplicate);
    let stale_fresh_request = moderator
        .request(proto::ApplyCollaborativeMessageOperation {
            operation_id: OperationId::new().to_string(),
            ..request.clone()
        })
        .await
        .expect("stale fresh deletion is refused even with a previously applied event");
    assert!(!stale_fresh_request.accepted);
    assert_eq!(
        stale_fresh_request.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorStaleVersion as i32
    );
    assert_eq!(stale_fresh_request.authoritative_version, tombstone.version);
    let reloaded = moderator
        .request(proto::ApplyCollaborativeMessageOperation {
            operation_id: OperationId::new().to_string(),
            expected_version: tombstone.version,
            ..request
        })
        .await
        .expect("reload moderated history through another canonical command");
    assert!(
        reloaded.accepted,
        "persisted moderation must hydrate without rewriting the author"
    );
    assert_eq!(reloaded.authoritative_version, tombstone.version);
    let nip29_self_deleted = member
        .request(deletion(
            community_id,
            nip29_self_deleted_message_id,
            OperationId::new(),
            nip29_self_deleted_record.version,
            &nip29_self_deleted_record.source_event_id,
            &member_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("member author may use NIP-29 self deletion without moderation metadata");
    assert!(nip29_self_deleted.accepted);
    let nip29_self_tombstone = nip29_self_deleted.message.expect("NIP-29 self tombstone");
    assert!(nip29_self_tombstone.deleted);
    assert!(nip29_self_tombstone.body.is_empty());
    assert_eq!(
        nip29_self_tombstone.author_principal_id,
        nip29_self_deleted_record.author_principal_id
    );
    assert_eq!(
        nip29_self_tombstone.version,
        nip29_self_deleted_record.version + 1
    );
    let own_metadata = deletion(
        community_id,
        moderator_message_id,
        OperationId::new(),
        moderator_record.version,
        &moderator_record.source_event_id,
        &moderator_signer,
        9_005,
        vec![vec![
            "public_reason".into(),
            "Audited moderation of own message".into(),
        ]],
    );
    let own_deleted = moderator
        .request(own_metadata.clone())
        .await
        .expect("admin may moderate own message");
    assert!(own_deleted.accepted);
    let own_tombstone = own_deleted.message.expect("own moderated tombstone");
    assert!(own_tombstone.deleted);
    let own_reloaded = moderator
        .request(proto::ApplyCollaborativeMessageOperation {
            operation_id: OperationId::new().to_string(),
            expected_version: own_tombstone.version,
            ..own_metadata
        })
        .await
        .expect("reload own moderated metadata through canonical history");
    assert!(own_reloaded.accepted);
    assert_eq!(own_reloaded.authoritative_version, own_tombstone.version);
    let retained_event = retained_request
        .signed_event
        .as_ref()
        .expect("retained signed moderation source")
        .clone();
    let retained_tags: Vec<Vec<String>> = retained_event
        .tags
        .iter()
        .map(|tag| tag.values.clone())
        .collect();
    let mut corrupted_tags = retained_tags.clone();
    corrupted_tags.push(vec!["public_reason".into(), "Corrupted history".into()]);
    for (kind, tags, signature, label) in [
        (
            9_005,
            corrupted_tags,
            retained_event.signature.clone(),
            "tags",
        ),
        (
            5,
            retained_tags.clone(),
            retained_event.signature.clone(),
            "kind",
        ),
        (9_005, retained_tags.clone(), vec![0; 64], "signature"),
    ] {
        let event_id = retained_event.claimed_event_id.clone();
        server.test_db.db().transaction(|transaction| {
            let event_id = event_id.clone();
            let tags = tags.clone();
            let signature = signature.clone();
            async move {
            transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                "SELECT set_config('app.community_id', $1, true)", [community_id.to_string().into()])).await?;
            // The disposable fixture must bypass the immutability trigger to simulate damaged storage.
            transaction.execute(Statement::from_string(sea_orm::DatabaseBackend::Postgres,
                "ALTER TABLE public.collaboration_events DISABLE TRIGGER collaboration_events_immutable".to_owned())).await?;
            let updated = transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                "UPDATE public.collaboration_events SET kind = $3, tags = $4, signature = $5 WHERE community_id = $1 AND event_id = $2",
                [community_id.as_uuid().into(), event_id.into(), kind.into(), serde_json::to_value(tags)?.into(), signature.into()])).await?;
            assert_eq!(updated.rows_affected(), 1);
            transaction.execute(Statement::from_string(sea_orm::DatabaseBackend::Postgres,
                "ALTER TABLE public.collaboration_events ENABLE TRIGGER collaboration_events_immutable".to_owned())).await?;
            Ok(())
        }
        }).await.expect("corrupt only the owned retained event and restore trigger");
        let corrupt_operation = OperationId::new();
        let corrupted = moderator
            .request(proto::ApplyCollaborativeMessageOperation {
                operation_id: corrupt_operation.to_string(),
                expected_version: tombstone.version,
                ..retained_request.clone()
            })
            .await;
        let event_id = retained_event.claimed_event_id.clone();
        let signature = retained_event.signature.clone();
        let tags = retained_tags.clone();
        server.test_db.db().transaction(|transaction| {
            let event_id = event_id.clone();
            let tags = tags.clone();
            let signature = signature.clone();
            async move {
            transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                "SELECT set_config('app.community_id', $1, true)", [community_id.to_string().into()])).await?;
            transaction.execute(Statement::from_string(sea_orm::DatabaseBackend::Postgres,
                "ALTER TABLE public.collaboration_events DISABLE TRIGGER collaboration_events_immutable".to_owned())).await?;
            let restored = transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                "UPDATE public.collaboration_events SET kind = 9005, tags = $3, signature = $4 WHERE community_id = $1 AND event_id = $2",
                [community_id.as_uuid().into(), event_id.into(), serde_json::to_value(tags)?.into(), signature.into()])).await?;
            assert_eq!(restored.rows_affected(), 1);
            transaction.execute(Statement::from_string(sea_orm::DatabaseBackend::Postgres,
                "ALTER TABLE public.collaboration_events ENABLE TRIGGER collaboration_events_immutable".to_owned())).await?;
            let trigger = transaction.query_one(Statement::from_string(sea_orm::DatabaseBackend::Postgres,
                "SELECT tgenabled::text AS enabled FROM pg_trigger WHERE tgrelid = 'public.collaboration_events'::regclass AND tgname = 'collaboration_events_immutable'".to_owned())).await?.expect("event immutability trigger");
            assert_eq!(trigger.try_get::<String>("", "enabled")?, "O");
            Ok(())
        }
        }).await.expect("restore exact retained event before checking response");
        let corrupted = corrupted.expect("corrupt history receives a typed unavailable response");
        assert!(
            !corrupted.accepted,
            "corrupt auxiliary {label} must not hydrate"
        );
        assert_eq!(
            corrupted.error_code,
            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorUnavailable as i32,
            "corrupt auxiliary {label} is infrastructure failure, not user denial"
        );
        server.test_db.db().transaction(|transaction| async move {
            transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                "SELECT set_config('app.community_id', $1, true)", [community_id.to_string().into()])).await?;
            for table in ["collaboration_command_receipts", "collaboration_outbox"] {
                let row = transaction.query_one(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
                    format!("SELECT count(*) AS count FROM public.{table} WHERE community_id = $1 AND operation_id = $2"),
                    [community_id.as_uuid().into(), corrupt_operation.as_uuid().into()])).await?.expect("corrupt history operation count");
                assert_eq!(row.try_get::<i64>("", "count")?, 0);
            }
            Ok(())
        }).await.expect("corrupt history cannot commit receipts or outbox");
    }
    let restored = moderator
        .request(proto::ApplyCollaborativeMessageOperation {
            operation_id: OperationId::new().to_string(),
            expected_version: tombstone.version,
            ..retained_request.clone()
        })
        .await
        .expect("genuine history remains usable after corruption controls");
    assert!(restored.accepted);
    assert_eq!(restored.authoritative_version, tombstone.version);
    server.test_db.db().transaction(|transaction| {
            let protocol_denials = protocol_denials.clone();
            let retained_request = retained_request.clone();
            let actor_id = actor_id.clone();
            let original_author = original_author.clone();
            async move {
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        )).await?;
        let rejected_operations = [denied_operation, foreign_operation, stale_operation, spoof_operation, metadata_operation]
            .into_iter().chain(protocol_denials.iter().map(|(operation_id, _)| *operation_id));
        for operation_id in rejected_operations {
            for table in ["collaboration_command_receipts", "collaboration_outbox"] {
                let row = transaction.query_one(Statement::from_sql_and_values(
                    sea_orm::DatabaseBackend::Postgres,
                    format!("SELECT count(*) AS count FROM public.{table} WHERE community_id = $1 AND operation_id = $2"),
                    [community_id.as_uuid().into(), operation_id.as_uuid().into()],
                )).await?.expect("rejected operation count");
                assert_eq!(row.try_get::<i64>("", "count")?, 0);
            }
        }
        for (_, event_id) in &protocol_denials {
            let row = transaction.query_one(Statement::from_sql_and_values(
                sea_orm::DatabaseBackend::Postgres,
                "SELECT count(*) AS count FROM public.collaboration_events WHERE community_id = $1 AND event_id = $2",
                [community_id.as_uuid().into(), event_id.clone().into()],
            )).await?.expect("rejected signed event count");
            assert_eq!(row.try_get::<i64>("", "count")?, 0);
        }
        let row = transaction.query_one(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT payload FROM public.collaboration_outbox WHERE community_id = $1 AND operation_id = $2",
            [community_id.as_uuid().into(), Uuid::parse_str(&retained_request.operation_id).map_err(anyhow::Error::from)?.into()],
        )).await?.expect("moderation outbox");
        let payload: serde_json::Value = serde_json::from_slice(&row.try_get::<Vec<u8>>("", "payload")?)?;
        assert_eq!(payload["actor_principal_id"], Uuid::from_slice(&actor_id).map_err(anyhow::Error::from)?.to_string());
        let row = transaction.query_one(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT auxiliary.actor_principal_id, auxiliary.is_tombstone, event.kind, event.tags, message.author_principal_id FROM public.collaboration_message_auxiliary_events AS auxiliary JOIN public.collaboration_events AS event ON event.community_id = auxiliary.community_id AND event.event_id = auxiliary.auxiliary_event_id JOIN public.collaboration_messages AS message ON message.community_id = auxiliary.community_id AND message.source_event_id = auxiliary.target_message_event_id WHERE auxiliary.community_id = $1 AND auxiliary.auxiliary_event_id = $2",
            [community_id.as_uuid().into(), retained_request.signed_event.as_ref().expect("retained signed deletion").claimed_event_id.clone().into()],
        )).await?.expect("moderation immutable history");
        assert_eq!(row.try_get::<Uuid>("", "actor_principal_id")?, Uuid::from_slice(&actor_id).map_err(anyhow::Error::from)?);
        assert_eq!(row.try_get::<Uuid>("", "author_principal_id")?, Uuid::from_slice(&original_author).map_err(anyhow::Error::from)?);
        assert!(row.try_get::<bool>("", "is_tombstone")?);
        assert_eq!(row.try_get::<i32>("", "kind")?, 9_005);
        let tags: serde_json::Value = row.try_get("", "tags")?;
        let expected_tags: Vec<_> = retained_request.signed_event.as_ref().expect("retained metadata").tags.iter().map(|tag| tag.values.clone()).collect();
        assert_eq!(tags, serde_json::to_value(expected_tags)?);
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "UPDATE public.collaboration_channels SET lifecycle_state = 'archived' WHERE community_id = $1 AND channel_id = $2",
            [community_id.as_uuid().into(), channel_id.as_uuid().into()],
        )).await?;
        Ok(())
        }
    }).await.expect("denied operations have no side effects; moderator and original author remain distinct");
    let archived_operation = OperationId::new();
    let archived = moderator
        .request(deletion(
            community_id,
            member_message_id,
            archived_operation,
            tombstone.version,
            &member_record.source_event_id,
            &moderator_signer,
            9_005,
            Vec::new(),
        ))
        .await
        .expect("archived channel denies moderation");
    assert!(!archived.accepted);
    server.test_db.db().transaction(|transaction| async move {
        transaction.execute(Statement::from_sql_and_values(sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)", [community_id.to_string().into()])).await?;
        for table in ["collaboration_command_receipts", "collaboration_outbox"] {
            let row = transaction.query_one(Statement::from_sql_and_values(
                sea_orm::DatabaseBackend::Postgres,
                format!("SELECT count(*) AS count FROM public.{table} WHERE community_id = $1 AND operation_id = $2"),
                [community_id.as_uuid().into(), archived_operation.as_uuid().into()],
            )).await?.expect("archived operation count");
            assert_eq!(row.try_get::<i64>("", "count")?, 0);
        }
        Ok(())
    }).await.expect("archived rejection leaves no receipt or outbox");
}

#[gpui::test]
async fn canonical_channel_history_paginates_dense_timestamps_without_duplicates(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    if std::env::var("CI").is_ok() && !cfg!(target_os = "linux") {
        return;
    }

    let mut server = TestServer::start_postgres(executor.clone()).await;
    let client = server.create_client(cx, "dense_history_author").await;
    let channel_id = server
        .make_channel("dense-history", None, (&client, cx), &mut [])
        .await;
    executor.run_until_parked();
    let community_id = community_id_for_legacy_root_channel(channel_id.0);
    let canonical_channel_id = channel_id_for_legacy_channel(channel_id.0);
    let signer = TestSigner::new(4);
    let opened = client
        .request(open_request(
            community_id,
            canonical_channel_id,
            signer.public_key,
            0,
        ))
        .await
        .expect("author opens dense history channel");
    assert_eq!(
        opened.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone as i32
    );

    let created_at = unix_time_seconds();
    for index in 0..3 {
        let message_id = AggregateId::new();
        let body = format!("dense message {index}");
        let response = client
            .request(operation_request(
                community_id,
                canonical_channel_id,
                message_id,
                OperationId::new(),
                proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
                0,
                &body,
                "",
                signer.event(
                    created_at,
                    40_002,
                    vec![vec!["h".into(), canonical_channel_id.to_string()]],
                    &body,
                ),
            ))
            .await
            .expect("create dense timestamp message");
        assert!(response.accepted);
    }

    let first = client
        .request(proto::GetCollaborativeMessageWindow {
            contract_version: 1,
            community_id: community_id.as_uuid().as_bytes().to_vec(),
            channel_id: canonical_channel_id.as_uuid().as_bytes().to_vec(),
            thread_root_event_id: Vec::new(),
            page_size: 2,
            cursor: None,
        })
        .await
        .expect("first dense history page")
        .page
        .expect("first page payload");
    assert_eq!(first.messages.len(), 2);
    assert!(!first.done);
    let cursor = first.next_cursor.expect("dense history cursor");
    let second = client
        .request(proto::GetCollaborativeMessageWindow {
            contract_version: 1,
            community_id: community_id.as_uuid().as_bytes().to_vec(),
            channel_id: canonical_channel_id.as_uuid().as_bytes().to_vec(),
            thread_root_event_id: Vec::new(),
            page_size: 2,
            cursor: Some(cursor),
        })
        .await
        .expect("second dense history page")
        .page
        .expect("second page payload");
    assert_eq!(second.messages.len(), 1);
    assert!(second.done);
    let unique_message_ids = first
        .messages
        .iter()
        .chain(&second.messages)
        .map(|message| message.message_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique_message_ids.len(), 3);
}

#[gpui::test]
async fn committed_channel_receipt_retries_after_hydration_recovers(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    if std::env::var("CI").is_ok() && !cfg!(target_os = "linux") {
        return;
    }
    let mut server = TestServer::start_postgres(executor).await;
    let author = server.create_client(cx, "hydration_author").await;
    let channel = server
        .make_channel("hydration-recovery", None, (&author, cx), &mut [])
        .await;
    let community_id = community_id_for_legacy_root_channel(channel.0);
    let channel_id = channel_id_for_legacy_channel(channel.0);
    let signer = TestSigner::new(7);
    let opened = author
        .request(open_request(community_id, channel_id, signer.public_key, 0))
        .await
        .expect("open author");
    assert_eq!(opened.error_code, 0);
    author
        .request(proto::CloseCollaborativeChannel {
            contract_version: 1,
            community_id: community_id.as_uuid().as_bytes().to_vec(),
            channel_id: channel_id.as_uuid().as_bytes().to_vec(),
        })
        .await
        .expect("close stream before injecting projection faults");
    let message_id = AggregateId::new();
    let operation_id = OperationId::new();
    let request = operation_request(
        community_id,
        channel_id,
        message_id,
        operation_id,
        proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
        0,
        "durably committed",
        "",
        signer.event(
            unix_time_seconds(),
            40_002,
            vec![vec!["h".into(), channel_id.to_string()]],
            "durably committed",
        ),
    );
    let committed = author
        .request(request.clone())
        .await
        .expect("commit message");
    assert!(committed.accepted);
    assert!(!committed.duplicate);
    let original = committed.message.expect("original committed message");
    let database = server.test_db.db();
    let runtime = &database
        .test_options
        .as_ref()
        .expect("database runtime")
        .runtime;
    let saved_message = runtime.block_on(async {
        let transaction = database.pool.begin().await?;
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        )).await?;
        let row = transaction.query_one(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "DELETE FROM public.collaboration_messages AS message WHERE community_id = $1 AND message_id = $2 RETURNING to_jsonb(message) AS saved_message",
            [community_id.as_uuid().into(), message_id.as_uuid().into()],
        )).await?.expect("only the projection is removed");
        let saved: serde_json::Value = row.try_get("", "saved_message")?;
        transaction.commit().await?;
        anyhow::Ok(saved)
    }).expect("simulate missing projection without deleting the event or receipt");
    let missing = author
        .request(request.clone())
        .await
        .expect("retry missing projection");
    assert!(
        !missing.accepted,
        "missing hydration must not produce a malformed success"
    );
    assert_eq!(
        missing.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying as i32
    );
    assert!(missing.retry_after_millis > 0);
    assert!(missing.message.is_none());
    runtime.block_on(async {
        let transaction = database.pool.begin().await?;
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        )).await?;
        let restored = transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            r#"INSERT INTO public.collaboration_messages AS message (
                community_id, message_id, channel_id, source_event_id, current_event_id, deleted_by_event_id, author_principal_id, message_created_at, lifecycle_state, message_version, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value, projected_at, agent_author_profile_id, agent_author_owner_principal_id, agent_author_proof_event_id, agent_author_attribution
            ) SELECT
                saved.community_id, saved.message_id, saved.channel_id, saved.source_event_id, saved.current_event_id, saved.deleted_by_event_id, saved.author_principal_id, saved.message_created_at, saved.lifecycle_state, saved.message_version, saved.source_system, saved.source_record_id, saved.source_version, saved.source_observed_at, saved.integrity_algorithm, saved.integrity_value, saved.projected_at, saved.agent_author_profile_id, saved.agent_author_owner_principal_id, saved.agent_author_proof_event_id, saved.agent_author_attribution
            FROM jsonb_populate_record(NULL::public.collaboration_messages, $1) AS saved"#,
            [saved_message.clone().into()],
        )).await?;
        assert_eq!(restored.rows_affected(), 1);
        let restored = transaction.query_one(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT to_jsonb(message) AS restored_message FROM public.collaboration_messages AS message WHERE community_id = $1 AND message_id = $2",
            [community_id.as_uuid().into(), message_id.as_uuid().into()],
        )).await?.ok_or_else(|| anyhow::anyhow!("restored projection is missing"))?;
        let restored_message: serde_json::Value = restored.try_get("", "restored_message")?;
        assert_eq!(restored_message, saved_message, "restore every original projection field, including generated identity");
        transaction.execute_unprepared("ALTER TABLE public.collaboration_messages RENAME TO collaboration_messages_hydration_unavailable").await?;
        transaction.commit().await?;
        anyhow::Ok(())
    }).expect("restore projection and inject a query failure in this disposable database");
    let unavailable = author
        .request(request.clone())
        .await
        .expect("retry unavailable projection");
    assert!(
        !unavailable.accepted,
        "query errors must not produce a malformed success"
    );
    assert_eq!(
        unavailable.error_code,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying as i32
    );
    assert!(unavailable.retry_after_millis > 0);
    assert!(unavailable.message.is_none());
    runtime.block_on(database.pool.execute_unprepared(
        "ALTER TABLE public.collaboration_messages_hydration_unavailable RENAME TO collaboration_messages"
    )).expect("restore the disposable projection table");
    let recovered = author.request(request).await.expect("retry after recovery");
    assert!(recovered.accepted);
    assert!(
        recovered.duplicate,
        "recovery must reuse the original durable receipt"
    );
    assert_eq!(recovered.error_code, 0);
    assert_eq!(
        recovered.authoritative_version,
        committed.authoritative_version
    );
    let recovered_message = recovered.message.expect("recovered message");
    assert_eq!(recovered_message.message_id, original.message_id);
    assert_eq!(recovered_message.source_event_id, original.source_event_id);
    assert_eq!(recovered_message.outbox_sequence, original.outbox_sequence);
    assert_eq!(recovered_message.body, original.body);
    assert_eq!(
        recovered_message.accepted_operation_id,
        operation_id.to_string()
    );
    runtime.block_on(async {
        let transaction = database.pool.begin().await?;
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        )).await?;
        let row = transaction.query_one(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT count(*)::bigint AS count FROM public.collaboration_outbox WHERE community_id = $1 AND operation_id = $2",
            [community_id.as_uuid().into(), operation_id.as_uuid().into()],
        )).await?.expect("outbox count");
        assert_eq!(row.try_get::<i64>("", "count")?, 1, "retry must not append another mutation");
        transaction.commit().await?;
        anyhow::Ok(())
    }).expect("verify one durable mutation after recovery");
}

#[gpui::test]
async fn canonical_channel_replay_drains_multiple_batches_and_recovers_without_hints(
    executor: BackgroundExecutor,
    cx_author: &mut TestAppContext,
    cx_reader: &mut TestAppContext,
) {
    if std::env::var("CI").is_ok() && !cfg!(target_os = "linux") {
        return;
    }
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let author = server.create_client(cx_author, "backlog_author").await;
    let reader = server.create_client(cx_reader, "backlog_reader").await;
    let channel = server
        .make_channel(
            "replay-backlog",
            None,
            (&author, cx_author),
            &mut [(&reader, cx_reader)],
        )
        .await;
    let community_id = community_id_for_legacy_root_channel(channel.0);
    let channel_id = channel_id_for_legacy_channel(channel.0);
    let signer = TestSigner::new(5);
    let opened = author
        .request(open_request(community_id, channel_id, signer.public_key, 0))
        .await
        .expect("open author");
    assert_eq!(opened.error_code, 0);
    author
        .request(proto::CloseCollaborativeChannel {
            contract_version: 1,
            community_id: community_id.as_uuid().as_bytes().to_vec(),
            channel_id: channel_id.as_uuid().as_bytes().to_vec(),
        })
        .await
        .expect("close author stream before constructing backlog");
    let created_at = unix_time_seconds();
    let mut expected = Vec::new();
    for index in 0..513 {
        let body = format!("backlog message {index}");
        let response = author
            .request(operation_request(
                community_id,
                channel_id,
                AggregateId::new(),
                OperationId::new(),
                proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate,
                0,
                &body,
                "",
                signer.event(
                    created_at,
                    40_002,
                    vec![vec!["h".into(), channel_id.to_string()]],
                    &body,
                ),
            ))
            .await
            .expect("commit backlog message");
        assert!(
            response.accepted,
            "backlog message {index} rejected: {response:?}"
        );
        let record = response.message.expect("committed record");
        expected.push((record.outbox_sequence, record.message_id));
    }
    let (updates_tx, updates_rx) = async_channel::bounded(1024);
    let update_receiver = cx_reader.new(|_| UpdateReceiver);
    let _subscription = reader.add_message_handler(
        update_receiver.downgrade(),
        move |_, envelope: TypedEnvelope<proto::CollaborativeMessageStreamUpdate>, _| {
            let sender = updates_tx.clone();
            async move {
                sender.send(envelope.payload).await?;
                anyhow::Ok(())
            }
        },
    );
    let opened = reader
        .request(open_request(
            community_id,
            channel_id,
            TestSigner::new(6).public_key,
            0,
        ))
        .await
        .expect("open backlog reader");
    assert_eq!(opened.error_code, 0);
    for (sequence, message_id) in &expected {
        let update = next_channel_update(&updates_rx, &executor).await;
        assert_eq!(update.outbox_sequence, *sequence);
        assert_eq!(
            &update.message.expect("replayed message").message_id,
            message_id
        );
    }
    assert_channel_status(
        next_channel_update(&updates_rx, &executor).await,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
    );
    executor.advance_clock(std::time::Duration::from_secs(2));
    executor.run_until_parked();
    assert!(
        updates_rx.try_recv().is_err(),
        "periodic catch-up must not duplicate delivered records"
    );

    let user = server
        .app_state
        .user_service
        .get_user_by_github_login("backlog_author")
        .await
        .expect("load author")
        .expect("author exists");
    let operation_ids = [OperationId::new(), OperationId::new()];
    let acknowledged_sequence = expected.last().expect("backlog exists").0;
    let database = server.test_db.db();
    // Commit through the real mutation/outbox boundary without publishing. This
    // models a writer dying after commit, before either local or Redis delivery.
    let (original_payload, sequences) = database
        .test_options
        .as_ref()
        .expect("database runtime")
        .runtime
        .block_on(async {
            use collab::collaboration_command::{
                CommandAdapter, DomainCommand, DomainCommandSink as _,
            };
            use collab::messages::channel_mutation::{
                CanonicalMessageMutation, MessageOperation, MessageOperationKind,
            };
            let authorization = collab::messages::channel_admission::admit_channel(
                &database.pool,
                &user,
                community_id,
                channel_id,
            )
            .await?;
            let mutation = CanonicalMessageMutation::new(DatabaseConnection::from(
                database.pool.get_postgres_connection_pool().clone(),
            ))?;
            let sink = collab::db::collaboration::outbox::TransactionalCommandOutbox::new(
                DatabaseConnection::from(database.pool.get_postgres_connection_pool().clone()),
                mutation,
            )
            .map_err(|backend| anyhow::anyhow!("unexpected test backend: {backend:?}"))?;
            for operation_id in operation_ids {
                sink.submit(DomainCommand::new(
                    operation_id,
                    authorization.tenant.clone(),
                    authorization.principal.clone(),
                    None,
                    None,
                    CommandAdapter::ZedRpc,
                    MessageOperation {
                        authorization: authorization.clone(),
                        kind: MessageOperationKind::Acknowledge,
                        message_id: AggregateId::new(),
                        expected_version: None,
                        signed_event: None,
                        reaction: None,
                        related_reaction_event_id: None,
                        acknowledged_outbox_sequence: Some(acknowledged_sequence),
                    },
                ))
                .await?;
            }
            let transaction = database.pool.begin().await?;
            transaction.execute(Statement::from_sql_and_values(
                sea_orm::DatabaseBackend::Postgres,
                "SELECT set_config('app.community_id', $1, true)",
                [community_id.to_string().into()],
            )).await?;
            let mut sequences = Vec::new();
            let mut original_payload = Vec::new();
            for operation_id in operation_ids {
                let row = transaction.query_one(Statement::from_sql_and_values(
                    sea_orm::DatabaseBackend::Postgres,
                    "SELECT payload, outbox_sequence FROM public.collaboration_outbox WHERE community_id = $1 AND operation_id = $2",
                    [community_id.as_uuid().into(), operation_id.as_uuid().into()],
                )).await?.expect("committed operation exists");
                sequences.push(u64::try_from(row.try_get::<i64>("", "outbox_sequence")?)?);
                if original_payload.is_empty() {
                    original_payload = row.try_get::<Vec<u8>>("", "payload")?;
                    let damaged = transaction.execute(Statement::from_sql_and_values(
                        sea_orm::DatabaseBackend::Postgres,
                        "UPDATE public.collaboration_outbox SET payload = $3 WHERE community_id = $1 AND operation_id = $2",
                        [community_id.as_uuid().into(), operation_id.as_uuid().into(), b"invalid-json".to_vec().into()],
                    )).await?;
                    assert_eq!(damaged.rows_affected(), 1);
                }
            }
            transaction.commit().await?;
            anyhow::Ok((original_payload, sequences))
        })
        .expect("commit without any notification");
    assert_channel_status(
        next_channel_update(&updates_rx, &executor).await,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying,
    );
    executor.advance_clock(std::time::Duration::from_secs(2));
    executor.run_until_parked();
    assert!(
        updates_rx.try_recv().is_err(),
        "a later record must not bypass failed payload hydration"
    );
    database.test_options.as_ref().expect("database runtime").runtime.block_on(async {
        let transaction = database.pool.begin().await?;
        transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        )).await?;
        let restored = transaction.execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "UPDATE public.collaboration_outbox SET payload = $3 WHERE community_id = $1 AND operation_id = $2",
            [community_id.as_uuid().into(), operation_ids[0].as_uuid().into(), original_payload.into()],
        )).await?;
        assert_eq!(restored.rows_affected(), 1);
        transaction.commit().await?;
        anyhow::Ok(())
    }).expect("restore the deliberately damaged test payload");
    executor.advance_clock(std::time::Duration::from_secs(2));
    for sequence in sequences {
        let recovered = next_channel_update(&updates_rx, &executor).await;
        assert_eq!(recovered.outbox_sequence, sequence);
        assert_eq!(
            recovered.operation_kind,
            proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge as i32
        );
        assert_eq!(
            recovered.acknowledged_outbox_sequence,
            acknowledged_sequence
        );
    }
    assert_channel_status(
        next_channel_update(&updates_rx, &executor).await,
        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
    );
    executor.advance_clock(std::time::Duration::from_secs(2));
    executor.run_until_parked();
    assert!(
        updates_rx.try_recv().is_err(),
        "recovered record must be sent exactly once"
    );
    reader
        .request(proto::CloseCollaborativeChannel {
            contract_version: 1,
            community_id: community_id.as_uuid().as_bytes().to_vec(),
            channel_id: channel_id.as_uuid().as_bytes().to_vec(),
        })
        .await
        .expect("close recovered subscription");
}

fn assert_channel_status(
    update: proto::CollaborativeMessageStreamUpdate,
    expected: proto::CollaborativeMessageErrorCode,
) {
    assert_eq!(update.outbox_sequence, 0);
    assert!(update.message.is_none());
    let status = update.status.expect("explicit delivery status");
    assert_eq!(status.error_code, expected as i32);
    assert_eq!(
        status.retry_after_millis,
        if expected == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying {
            1000
        } else {
            0
        }
    );
}

async fn next_channel_update(
    updates: &async_channel::Receiver<proto::CollaborativeMessageStreamUpdate>,
    executor: &BackgroundExecutor,
) -> proto::CollaborativeMessageStreamUpdate {
    futures::select_biased! {
        update = updates.recv().fuse() => update.expect("reader stream open"),
        _ = executor.timer(std::time::Duration::from_secs(10)).fuse() => panic!("canonical stream did not catch up"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn postgres_replay_recovers_when_redis_notification_is_unavailable() -> anyhow::Result<()> {
    let Ok(base_url) = std::env::var("COLLAB_TEST_DATABASE_URL") else {
        return Ok(());
    };
    let mut database_url = url::Url::parse(&base_url)?;
    database_url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let database_url = database_url.to_string();
    sqlx::Postgres::create_database(&database_url).await?;

    let test_result = run_redis_loss_replay_test(&database_url).await;
    let drop_result = sqlx::Postgres::drop_database(&database_url).await;
    test_result?;
    drop_result?;
    Ok(())
}

async fn run_redis_loss_replay_test(database_url: &str) -> anyhow::Result<()> {
    sqlx::any::install_default_drivers();
    let options = sea_orm::ConnectOptions::new(database_url.to_owned());
    run_database_migrations(&options, concat!(env!("CARGO_MANIFEST_DIR"), "/migrations")).await?;
    let connection = sea_orm::Database::connect(options).await?;
    let community_id = CommunityId::new();
    let operation_id = OperationId::new();
    let topic = format!("community.{}.channel.test", community_id);
    let transaction = connection.begin().await?;
    transaction
        .execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        ))
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            r#"
INSERT INTO public.collaboration_command_receipts (
    community_id, operation_id, contract_version, principal_id, originating_adapter,
    command_kind, command_fingerprint, authoritative_version, accepted_at
) VALUES ($1, $2, 1, $3, 'zed_rpc', 'test.redis_loss', $4, 1, clock_timestamp())
"#,
            [
                community_id.as_uuid().into(),
                operation_id.as_uuid().into(),
                Uuid::new_v4().into(),
                vec![7_u8; 32].into(),
            ],
        ))
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            sea_orm::DatabaseBackend::Postgres,
            r#"
INSERT INTO public.collaboration_outbox (
    community_id, operation_id, authoritative_version, topic, source_system,
    source_record_id, source_version, source_observed_at, payload
) VALUES ($1, $2, 1, $3, 'zed', $4, '1', clock_timestamp(), $5)
"#,
            [
                community_id.as_uuid().into(),
                operation_id.as_uuid().into(),
                topic.clone().into(),
                operation_id.to_string().into(),
                b"committed-message".to_vec().into(),
            ],
        ))
        .await?;
    transaction.commit().await?;

    let route = TrustedTenantRoute::from_deployment(community_id, "test:redis-loss")?;
    let tenant = TenantContext::establish(Some(route), &[])?;
    let replay = PostgresFanoutReplayStore::new(DatabaseConnection::from(
        connection.get_postgres_connection_pool().clone(),
    ))?;
    let envelope = replay.envelope_for_operation(&tenant, operation_id).await?;
    replay.validate_notification(&tenant, &envelope).await?;
    let mut changed_provenance = envelope.provenance().clone();
    changed_provenance.source_version = Some("999".to_owned());
    let forged_notifications = [
        FanoutEnvelope::new(
            community_id,
            envelope.outbox_sequence(),
            format!("{topic}.retagged"),
            envelope.provenance().clone(),
            *envelope.payload_sha256(),
        )?,
        FanoutEnvelope::new(
            community_id,
            envelope.outbox_sequence(),
            topic.clone(),
            envelope.provenance().clone(),
            [0; 32],
        )?,
        FanoutEnvelope::new(
            community_id,
            envelope.outbox_sequence(),
            topic.clone(),
            changed_provenance,
            *envelope.payload_sha256(),
        )?,
        FanoutEnvelope::new(
            community_id,
            envelope.outbox_sequence().saturating_add(1),
            topic.clone(),
            envelope.provenance().clone(),
            *envelope.payload_sha256(),
        )?,
    ];
    for forged in forged_notifications {
        assert_eq!(
            replay.validate_notification(&tenant, &forged).await,
            Err(SubscriptionBusError::InvalidRequest)
        );
    }
    let foreign_notification = FanoutEnvelope::new(
        CommunityId::new(),
        envelope.outbox_sequence(),
        topic.clone(),
        envelope.provenance().clone(),
        *envelope.payload_sha256(),
    )?;
    assert_eq!(
        replay
            .validate_notification(&tenant, &foreign_notification)
            .await,
        Err(SubscriptionBusError::TenantMismatch)
    );
    replay.validate_notification(&tenant, &envelope).await?;
    let encoded = envelope.encode()?;

    if let Ok(redis_url) = std::env::var("COLLAB_TEST_REDIS_URL") {
        let redis = RedisFanoutTransport::new(&redis_url)?;
        let subscriber = redis.clone();
        let expected = encoded.clone();
        let (delivered_tx, delivered_rx) = tokio::sync::oneshot::channel();
        let mut delivered_tx = Some(delivered_tx);
        let (_community_registration, communities) =
            tokio::sync::watch::channel(vec![tenant.community_id()]);
        let subscription = tokio::spawn(async move {
            subscriber
                .subscribe(communities, move |payload| {
                    if payload == expected
                        && let Some(delivered_tx) = delivered_tx.take()
                    {
                        delivered_tx
                            .send(())
                            .expect("Redis delivery receiver remains active");
                    }
                    std::future::ready(())
                })
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        redis.publish(encoded.clone()).await?;
        tokio::time::timeout(std::time::Duration::from_secs(5), delivered_rx).await??;
        subscription.abort();
    }

    let unavailable_redis = RedisFanoutTransport::new("redis://127.0.0.1:1")?;
    assert!(unavailable_redis.publish(encoded).await.is_err());
    let recovered = replay.load_after(&tenant, &topic, 0, 16).await?;
    assert_eq!(recovered, vec![envelope]);
    assert_eq!(
        replay
            .payload(&tenant, recovered[0].outbox_sequence())
            .await?,
        b"committed-message"
    );
    connection.close().await?;
    Ok(())
}

fn open_request(
    community_id: collaboration_domain::CommunityId,
    channel_id: AggregateId,
    public_key: [u8; 32],
    after_outbox_sequence: u64,
) -> proto::OpenCollaborativeChannel {
    proto::OpenCollaborativeChannel {
        contract_version: 1,
        community_id: community_id.as_uuid().as_bytes().to_vec(),
        channel_id: channel_id.as_uuid().as_bytes().to_vec(),
        page_size: 100,
        after_outbox_sequence,
        signing_public_key: public_key.to_vec(),
    }
}

fn operation_request(
    community_id: collaboration_domain::CommunityId,
    channel_id: AggregateId,
    message_id: AggregateId,
    operation_id: OperationId,
    kind: proto::CollaborativeMessageOperationKind,
    expected_version: u64,
    body: &str,
    reaction: &str,
    signed_event: proto::CollaborativeSignedEvent,
) -> proto::ApplyCollaborativeMessageOperation {
    proto::ApplyCollaborativeMessageOperation {
        contract_version: 1,
        community_id: community_id.as_uuid().as_bytes().to_vec(),
        channel_id: channel_id.as_uuid().as_bytes().to_vec(),
        message_id: message_id.as_uuid().as_bytes().to_vec(),
        operation_id: operation_id.to_string(),
        kind: kind.into(),
        expected_version,
        body: body.to_owned(),
        reply_to_event_id: Vec::new(),
        reaction: reaction.to_owned(),
        related_reaction_event_id: Vec::new(),
        signed_event: Some(signed_event),
        acknowledged_outbox_sequence: 0,
    }
}

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
