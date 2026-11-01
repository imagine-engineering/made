use crate::{
    RoomParticipants, TestClient, TestServer, channel_id, following_tests::join_channel,
    room_participants,
};
use anyhow::{Result, anyhow};
use buffer_diff::{DiffHunkSecondaryStatus, DiffHunkStatus, assert_hunks};
use call::{ActiveCall, Room, room};
use client::{RECEIVE_TIMEOUT, User};
use collab::rpc::{CLEANUP_TIMEOUT, RECONNECT_TIMEOUT};
use collections::{BTreeMap, HashMap, HashSet};
use fs::{FakeFs, Fs as _, RemoveOptions};
use futures::{StreamExt as _, channel::mpsc};
use git::{
    repository::repo_path,
    status::{FileStatus, StatusCode, TrackedStatus, UnmergedStatus, UnmergedStatusCode},
};
use gpui::{
    App, AppContext as _, BackgroundExecutor, Entity, Modifiers, MouseButton, MouseDownEvent,
    TestAppContext, UpdateGlobal, px, size,
};
use language::{
    Diagnostic, DiagnosticEntry, DiagnosticSourceKind, FakeLspAdapter, Language, LanguageConfig,
    LanguageMatcher, LineEnding, OffsetRangeExt, Point, Rope,
    language_settings::{Formatter, FormatterList},
    rust_lang, tree_sitter_rust, tree_sitter_typescript,
};
use lsp::{DEFAULT_LSP_REQUEST_TIMEOUT, LanguageServerId, OneOf};
use parking_lot::Mutex;
use pretty_assertions::assert_eq;
use project::{
    DiagnosticSummary, HoverBlockKind, Project, ProjectPath,
    lsp_store::{FormatTrigger, LspFormatTarget, SymbolLocation},
    search::{SearchQuery, SearchResult},
};
use rand::prelude::*;
use rpc::proto;
use serde_json::json;
use settings::{LanguageServerFormatterSpecifier, PrettierSettingsContent, SettingsStore};
use std::{
    cell::{Cell, RefCell},
    env, future, mem,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    time::Duration,
};
use unindent::Unindent as _;
use util::{path, rel_path::rel_path, uri};
use workspace::{Pane, ParticipantLocation};

#[ctor::ctor(unsafe)]
fn init_logger() {
    zlog::init_test();
}

#[gpui::test(iterations = 10)]
async fn test_database_failure_during_client_reconnection(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client = server.create_client(cx, "user_a").await;

    // Keep disconnecting the client until a database failure prevents it from
    // reconnecting.
    server.test_db.set_query_failure_probability(0.3);
    loop {
        server.disconnect_client(client.peer_id().unwrap());
        executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
        if !client.status().borrow().is_connected() {
            break;
        }
    }

    // Make the database healthy again and ensure the client can finally connect.
    server.test_db.set_query_failure_probability(0.);
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert!(
        matches!(*client.status().borrow(), client::Status::Connected { .. }),
        "status was {:?}",
        *client.status().borrow()
    );
}

#[gpui::test(iterations = 10)]
async fn test_basic_calls(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_b2: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);

    // Call user B from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: vec!["user_b".to_string()]
        }
    );

    // User B receives the call.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    let call_b = incoming_call_b.next().await.unwrap().unwrap();
    assert_eq!(call_b.calling_user.username, "user_a");

    // User B connects via another client and also receives a ring on the newly-connected client.
    let _client_b2 = server.create_client(cx_b2, "user_b").await;
    let active_call_b2 = cx_b2.read(ActiveCall::global);

    let mut incoming_call_b2 = active_call_b2.read_with(cx_b2, |call, _| call.incoming());
    executor.run_until_parked();
    let call_b2 = incoming_call_b2.next().await.unwrap().unwrap();
    assert_eq!(call_b2.calling_user.username, "user_a");

    // User B joins the room using the first client.
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    assert!(incoming_call_b.next().await.unwrap().is_none());

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: Default::default()
        }
    );

    // Call user C from client B.

    let mut incoming_call_c = active_call_c.read_with(cx_c, |call, _| call.incoming());
    active_call_b
        .update(cx_b, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: vec!["user_c".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: vec!["user_c".to_string()]
        }
    );

    // User C receives the call, but declines it.
    let call_c = incoming_call_c.next().await.unwrap().unwrap();
    assert_eq!(call_c.calling_user.username, "user_b");
    active_call_c.update(cx_c, |call, cx| call.decline_incoming(cx).unwrap());
    assert!(incoming_call_c.next().await.unwrap().is_none());

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: Default::default()
        }
    );

    // Call user C again from user A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: vec!["user_c".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: vec!["user_c".to_string()]
        }
    );

    // User C accepts the call.
    let call_c = incoming_call_c.next().await.unwrap().unwrap();
    assert_eq!(call_c.calling_user.username, "user_a");
    active_call_c
        .update(cx_c, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    assert!(incoming_call_c.next().await.unwrap().is_none());

    let room_c = active_call_c.read_with(cx_c, |call, _| call.room().unwrap().clone());

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string(), "user_c".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_c".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_b".to_string()],
            pending: Default::default()
        }
    );

    // User A shares their screen
    let display = gpui::TestScreenCaptureSource::new();
    let events_b = active_call_events(cx_b);
    let events_c = active_call_events(cx_c);
    cx_a.set_screen_capture_sources(vec![display]);
    let screen_a = cx_a
        .update(|cx| cx.screen_capture_sources())
        .await
        .unwrap()
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    active_call_a
        .update(cx_a, |call, cx| {
            call.room()
                .unwrap()
                .update(cx, |room, cx| room.share_screen(screen_a, cx))
        })
        .await
        .unwrap();

    executor.run_until_parked();

    // User B observes the remote screen sharing track.
    assert_eq!(events_b.borrow().len(), 1);
    let event_b = events_b.borrow().first().unwrap().clone();
    if let call::room::Event::RemoteVideoTracksChanged { participant_id } = event_b {
        assert_eq!(participant_id, client_a.peer_id().unwrap());

        room_b.read_with(cx_b, |room, _| {
            assert_eq!(
                room.remote_participants()[&client_a.user_id().unwrap()]
                    .video_tracks
                    .len(),
                1
            );
        });
    } else {
        panic!("unexpected event")
    }

    // User C observes the remote screen sharing track.
    assert_eq!(events_c.borrow().len(), 1);
    let event_c = events_c.borrow().first().unwrap().clone();
    if let call::room::Event::RemoteVideoTracksChanged { participant_id } = event_c {
        assert_eq!(participant_id, client_a.peer_id().unwrap());

        room_c.read_with(cx_c, |room, _| {
            assert_eq!(
                room.remote_participants()[&client_a.user_id().unwrap()]
                    .video_tracks
                    .len(),
                1
            );
        });
    } else {
        panic!("unexpected event")
    }

    // User A leaves the room.
    active_call_a
        .update(cx_a, |call, cx| {
            let hang_up = call.hang_up(cx);
            assert!(call.room().is_none());
            hang_up
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_c".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );

    // User B gets disconnected from the LiveKit server, which causes them
    // to automatically leave the room. User C leaves the room as well because
    // nobody else is in there.
    server
        .test_livekit_server
        .disconnect_client(client_b.user_id().unwrap().to_string())
        .await;
    executor.run_until_parked();

    active_call_b.read_with(cx_b, |call, _| assert!(call.room().is_none()));

    active_call_c.read_with(cx_c, |call, _| assert!(call.room().is_none()));
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
}

#[gpui::test(iterations = 10)]
async fn test_calling_multiple_users_simultaneously(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    cx_d: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    let client_d = server.create_client(cx_d, "user_d").await;
    server
        .make_contacts(&mut [
            (&client_a, cx_a),
            (&client_b, cx_b),
            (&client_c, cx_c),
            (&client_d, cx_d),
        ])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);
    let active_call_d = cx_d.read(ActiveCall::global);

    // Simultaneously call user B and user C from client A.
    let b_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_b.user_id().unwrap(), None, cx)
    });
    let c_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_c.user_id().unwrap(), None, cx)
    });
    b_invite.await.unwrap();
    c_invite.await.unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: vec!["user_b".to_string(), "user_c".to_string()]
        }
    );

    // Call client D from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_d.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: vec![
                "user_b".to_string(),
                "user_c".to_string(),
                "user_d".to_string()
            ]
        }
    );

    // Accept the call on all clients simultaneously.
    let accept_b = active_call_b.update(cx_b, |call, cx| call.accept_incoming(cx));
    let accept_c = active_call_c.update(cx_c, |call, cx| call.accept_incoming(cx));
    let accept_d = active_call_d.update(cx_d, |call, cx| call.accept_incoming(cx));
    accept_b.await.unwrap();
    accept_c.await.unwrap();
    accept_d.await.unwrap();

    executor.run_until_parked();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());

    let room_c = active_call_c.read_with(cx_c, |call, _| call.room().unwrap().clone());

    let room_d = active_call_d.read_with(cx_d, |call, _| call.room().unwrap().clone());
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec![
                "user_b".to_string(),
                "user_c".to_string(),
                "user_d".to_string(),
            ],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec![
                "user_a".to_string(),
                "user_c".to_string(),
                "user_d".to_string(),
            ],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec![
                "user_a".to_string(),
                "user_b".to_string(),
                "user_d".to_string(),
            ],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_d, cx_d),
        RoomParticipants {
            remote: vec![
                "user_a".to_string(),
                "user_b".to_string(),
                "user_c".to_string(),
            ],
            pending: Default::default()
        }
    );
}

#[gpui::test(iterations = 10)]
async fn test_joining_channels_and_calling_multiple_users_simultaneously(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let channel_1 = server
        .make_channel(
            "channel1",
            None,
            (&client_a, cx_a),
            &mut [(&client_b, cx_b), (&client_c, cx_c)],
        )
        .await;

    let channel_2 = server
        .make_channel(
            "channel2",
            None,
            (&client_a, cx_a),
            &mut [(&client_b, cx_b), (&client_c, cx_c)],
        )
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);

    // Simultaneously join channel 1 and then channel 2
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_1, cx))
        .detach();
    let join_channel_2 = active_call_a.update(cx_a, |call, cx| call.join_channel(channel_2, cx));

    join_channel_2.await.unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();

    assert_eq!(channel_id(&room_a, cx_a), Some(channel_2));

    // Leave the room
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();

    // Initiating invites and then joining a channel should fail gracefully
    let b_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_b.user_id().unwrap(), None, cx)
    });
    let c_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_c.user_id().unwrap(), None, cx)
    });

    let join_channel = active_call_a.update(cx_a, |call, cx| call.join_channel(channel_1, cx));

    b_invite.await.unwrap();
    c_invite.await.unwrap();
    join_channel.await.unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();

    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: vec!["user_b".to_string(), "user_c".to_string()]
        }
    );

    assert_eq!(channel_id(&room_a, cx_a), None);

    // Leave the room
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();

    // Simultaneously join channel 1 and call user B and user C from client A.
    let join_channel = active_call_a.update(cx_a, |call, cx| call.join_channel(channel_1, cx));

    let b_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_b.user_id().unwrap(), None, cx)
    });
    let c_invite = active_call_a.update(cx_a, |call, cx| {
        call.invite(client_c.user_id().unwrap(), None, cx)
    });

    join_channel.await.unwrap();
    b_invite.await.unwrap();
    c_invite.await.unwrap();

    active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
}

#[gpui::test(iterations = 10)]
async fn test_room_uniqueness(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_a2: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_b2: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let _client_a2 = server.create_client(cx_a2, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let _client_b2 = server.create_client(cx_b2, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_a2 = cx_a2.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_b2 = cx_b2.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);

    // Call user B from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    // Ensure a new room can't be created given user A just created one.
    active_call_a2
        .update(cx_a2, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();

    active_call_a2.read_with(cx_a2, |call, _| assert!(call.room().is_none()));

    // User B receives the call from user A.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    let call_b1 = incoming_call_b.next().await.unwrap().unwrap();
    assert_eq!(call_b1.calling_user.username, "user_a");

    // Ensure calling users A and B from client C fails.
    active_call_c
        .update(cx_c, |call, cx| {
            call.invite(client_a.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();
    active_call_c
        .update(cx_c, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();

    // Ensure User B can't create a room while they still have an incoming call.
    active_call_b2
        .update(cx_b2, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();

    active_call_b2.read_with(cx_b2, |call, _| assert!(call.room().is_none()));

    // User B joins the room and calling them after they've joined still fails.
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    active_call_c
        .update(cx_c, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();

    // Ensure User B can't create a room while they belong to another room.
    active_call_b2
        .update(cx_b2, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap_err();

    active_call_b2.read_with(cx_b2, |call, _| assert!(call.room().is_none()));

    // Client C can successfully call client B after client B leaves the room.
    active_call_b
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    active_call_c
        .update(cx_c, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    let call_b2 = incoming_call_b.next().await.unwrap().unwrap();
    assert_eq!(call_b2.calling_user.username, "user_c");
}

#[gpui::test(iterations = 10)]
async fn test_client_disconnecting_from_room(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    // Call user B from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());

    // User B receives the call and joins the room.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    incoming_call_b.next().await.unwrap().unwrap();
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: Default::default()
        }
    );

    // User A automatically reconnects to the room upon disconnection.
    server.disconnect_client(client_a.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT);
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: Default::default()
        }
    );

    // When user A disconnects, both client A and B clear their room on the active call.
    server.forbid_connections();
    server.disconnect_client(client_a.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);

    active_call_a.read_with(cx_a, |call, _| assert!(call.room().is_none()));

    active_call_b.read_with(cx_b, |call, _| assert!(call.room().is_none()));
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );

    // Allow user A to reconnect to the server.
    server.allow_connections();
    executor.advance_clock(RECONNECT_TIMEOUT);

    // Call user B again from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());

    // User B receives the call and joins the room.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    incoming_call_b.next().await.unwrap().unwrap();
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: Default::default()
        }
    );

    // User B gets disconnected from the LiveKit server, which causes it
    // to automatically leave the room.
    server
        .test_livekit_server
        .disconnect_client(client_b.user_id().unwrap().to_string())
        .await;
    executor.run_until_parked();
    active_call_a.update(cx_a, |call, _| assert!(call.room().is_none()));
    active_call_b.update(cx_b, |call, _| assert!(call.room().is_none()));
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: Default::default(),
            pending: Default::default()
        }
    );
}

#[gpui::test(iterations = 10)]
async fn test_server_restarts(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    cx_d: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    client_a
        .fs()
        .insert_tree("/a", json!({ "a.txt": "a-contents" }))
        .await;

    // Invite client B to collaborate on a project
    let (project_a, _) = client_a.build_local_project("/a", cx_a).await;

    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    let client_d = server.create_client(cx_d, "user_d").await;
    server
        .make_contacts(&mut [
            (&client_a, cx_a),
            (&client_b, cx_b),
            (&client_c, cx_c),
            (&client_d, cx_d),
        ])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);
    let active_call_d = cx_d.read(ActiveCall::global);

    // User A calls users B, C, and D.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), Some(project_a.clone()), cx)
        })
        .await
        .unwrap();
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_c.user_id().unwrap(), Some(project_a.clone()), cx)
        })
        .await
        .unwrap();
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_d.user_id().unwrap(), Some(project_a.clone()), cx)
        })
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());

    // User B receives the call and joins the room.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    assert!(incoming_call_b.next().await.unwrap().is_some());
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());

    // User C receives the call and joins the room.

    let mut incoming_call_c = active_call_c.read_with(cx_c, |call, _| call.incoming());
    assert!(incoming_call_c.next().await.unwrap().is_some());
    active_call_c
        .update(cx_c, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_c = active_call_c.read_with(cx_c, |call, _| call.room().unwrap().clone());

    // User D receives the call but doesn't join the room yet.

    let mut incoming_call_d = active_call_d.read_with(cx_d, |call, _| call.incoming());
    assert!(incoming_call_d.next().await.unwrap().is_some());

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string(), "user_c".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_c".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_b".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );

    // The server is torn down.
    server.reset().await;

    // Users A and B reconnect to the call. User C has troubles reconnecting, so it leaves the room.
    client_c.override_establish_connection(|_, cx| cx.spawn(async |_| future::pending().await));
    executor.advance_clock(RECONNECT_TIMEOUT);
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string(), "user_c".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_c".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );

    // User D is notified again of the incoming call and accepts it.
    assert!(incoming_call_d.next().await.unwrap().is_some());
    active_call_d
        .update(cx_d, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    let room_d = active_call_d.read_with(cx_d, |call, _| call.room().unwrap().clone());
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec![
                "user_b".to_string(),
                "user_c".to_string(),
                "user_d".to_string(),
            ],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec![
                "user_a".to_string(),
                "user_c".to_string(),
                "user_d".to_string(),
            ],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_d, cx_d),
        RoomParticipants {
            remote: vec![
                "user_a".to_string(),
                "user_b".to_string(),
                "user_c".to_string(),
            ],
            pending: vec![]
        }
    );

    // The server finishes restarting, cleaning up stale connections.
    server.start().await.unwrap();
    executor.advance_clock(CLEANUP_TIMEOUT);
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string(), "user_d".to_string()],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_d".to_string()],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_d, cx_d),
        RoomParticipants {
            remote: vec!["user_a".to_string(), "user_b".to_string()],
            pending: vec![]
        }
    );

    // User D hangs up.
    active_call_d
        .update(cx_d, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_c, cx_c),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_d, cx_d),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );

    // User B calls user D again.
    active_call_b
        .update(cx_b, |call, cx| {
            call.invite(client_d.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    // User D receives the call but doesn't join the room yet.

    let mut incoming_call_d = active_call_d.read_with(cx_d, |call, _| call.incoming());
    assert!(incoming_call_d.next().await.unwrap().is_some());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: vec!["user_d".to_string()]
        }
    );

    // The server is torn down.
    server.reset().await;

    // Users A and B have troubles reconnecting, so they leave the room.
    client_a.override_establish_connection(|_, cx| cx.spawn(async |_| future::pending().await));
    client_b.override_establish_connection(|_, cx| cx.spawn(async |_| future::pending().await));
    client_c.override_establish_connection(|_, cx| cx.spawn(async |_| future::pending().await));
    executor.advance_clock(RECONNECT_TIMEOUT);
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec![],
            pending: vec![]
        }
    );

    // User D is notified again of the incoming call but doesn't accept it.
    assert!(incoming_call_d.next().await.unwrap().is_some());

    // The server finishes restarting, cleaning up stale connections and canceling the
    // call to user D because the room has become empty.
    server.start().await.unwrap();
    executor.advance_clock(CLEANUP_TIMEOUT);
    assert!(incoming_call_d.next().await.unwrap().is_none());
}

#[gpui::test(iterations = 10)]
async fn test_calls_on_multiple_connections(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b1: &mut TestAppContext,
    cx_b2: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b1 = server.create_client(cx_b1, "user_b").await;
    let client_b2 = server.create_client(cx_b2, "user_b").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b1, cx_b1)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b1 = cx_b1.read(ActiveCall::global);
    let active_call_b2 = cx_b2.read(ActiveCall::global);

    let mut incoming_call_b1 = active_call_b1.read_with(cx_b1, |call, _| call.incoming());

    let mut incoming_call_b2 = active_call_b2.read_with(cx_b2, |call, _| call.incoming());
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // Call user B from client A, ensuring both clients for user B ring.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User B declines the call on one of the two connections, causing both connections
    // to stop ringing.
    active_call_b2.update(cx_b2, |call, cx| call.decline_incoming(cx).unwrap());
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // Call user B again from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User B accepts the call on one of the two connections, causing both connections
    // to stop ringing.
    active_call_b2
        .update(cx_b2, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // User B disconnects the client that is not on the call. Everything should be fine.
    client_b1.disconnect(&cx_b1.to_async());
    executor.advance_clock(RECEIVE_TIMEOUT);
    client_b1
        .connect(false, &cx_b1.to_async())
        .await
        .into_response()
        .unwrap();

    // User B hangs up, and user A calls them again.
    active_call_b2
        .update(cx_b2, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User A cancels the call, causing both connections to stop ringing.
    active_call_a
        .update(cx_a, |call, cx| {
            call.cancel_invite(client_b1.user_id().unwrap(), cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // User A calls user B again.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User A hangs up, causing both connections to stop ringing.
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // User A calls user B again.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User A disconnects, causing both connections to stop ringing.
    server.forbid_connections();
    server.disconnect_client(client_a.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert!(incoming_call_b1.next().await.unwrap().is_none());
    assert!(incoming_call_b2.next().await.unwrap().is_none());

    // User A reconnects automatically, then calls user B again.
    server.allow_connections();
    executor.advance_clock(RECONNECT_TIMEOUT);
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b1.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(incoming_call_b1.next().await.unwrap().is_some());
    assert!(incoming_call_b2.next().await.unwrap().is_some());

    // User B disconnects all clients, causing user A to no longer see a pending call for them.
    server.forbid_connections();
    server.disconnect_client(client_b1.peer_id().unwrap());
    server.disconnect_client(client_b2.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);

    active_call_a.read_with(cx_a, |call, _| assert!(call.room().is_none()));
}

#[gpui::test(iterations = 10)]
async fn test_unshare_project(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/a",
            json!({
                "a.txt": "a-contents",
                "b.txt": "b-contents",
            }),
        )
        .await;

    let (project_a, worktree_id) = client_a.build_local_project("/a", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    let worktree_a = project_a.read_with(cx_a, |project, cx| project.worktrees(cx).next().unwrap());
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();

    assert!(worktree_a.read_with(cx_a, |tree, _| tree.has_update_observer()));

    project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    // When client B leaves the room, the project becomes read-only.
    active_call_b
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    assert!(project_b.read_with(cx_b, |project, cx| project.is_disconnected(cx)));

    // Client C opens the project.
    let project_c = client_c.join_remote_project(project_id, cx_c).await;

    // When client A unshares the project, client C's project becomes read-only.
    project_a
        .update(cx_a, |project, cx| project.unshare(cx))
        .unwrap();
    executor.run_until_parked();

    assert!(worktree_a.read_with(cx_a, |tree, _| !tree.has_update_observer()));

    assert!(project_c.read_with(cx_c, |project, cx| project.is_disconnected(cx)));

    // Client C can open the project again after client A re-shares.
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_c2 = client_c.join_remote_project(project_id, cx_c).await;
    executor.run_until_parked();

    assert!(worktree_a.read_with(cx_a, |tree, _| tree.has_update_observer()));
    project_c2
        .update(cx_c, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    // When client A (the host) leaves the room, the project gets unshared and guests are notified.
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| assert!(!project.is_shared()));

    project_c2.read_with(cx_c, |project, cx| {
        assert!(project.is_disconnected(cx));
        assert!(project.collaborators().is_empty());
    });
}

#[gpui::test(iterations = 10)]
async fn test_project_reconnect(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    cx_b.update(editor::init);

    client_a
        .fs()
        .insert_tree(
            path!("/root-1"),
            json!({
                "dir1": {
                    "a.txt": "a",
                    "b.txt": "b",
                    "subdir1": {
                        "c.txt": "c",
                        "d.txt": "d",
                        "e.txt": "e",
                    }
                },
                "dir2": {
                    "v.txt": "v",
                },
                "dir3": {
                    "w.txt": "w",
                    "x.txt": "x",
                    "y.txt": "y",
                },
                "dir4": {
                    "z.txt": "z",
                },
            }),
        )
        .await;
    client_a
        .fs()
        .insert_tree(
            path!("/root-2"),
            json!({
                "2.txt": "2",
            }),
        )
        .await;
    client_a
        .fs()
        .insert_tree(
            path!("/root-3"),
            json!({
                "3.txt": "3",
            }),
        )
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let (project_a1, _) = client_a
        .build_local_project(path!("/root-1/dir1"), cx_a)
        .await;
    let (project_a2, _) = client_a.build_local_project(path!("/root-2"), cx_a).await;
    let (project_a3, _) = client_a.build_local_project(path!("/root-3"), cx_a).await;
    let worktree_a1 =
        project_a1.read_with(cx_a, |project, cx| project.worktrees(cx).next().unwrap());
    let project1_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a1.clone(), cx))
        .await
        .unwrap();
    let project2_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a2.clone(), cx))
        .await
        .unwrap();
    let project3_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a3.clone(), cx))
        .await
        .unwrap();

    let project_b1 = client_b.join_remote_project(project1_id, cx_b).await;
    let project_b2 = client_b.join_remote_project(project2_id, cx_b).await;
    let project_b3 = client_b.join_remote_project(project3_id, cx_b).await;
    executor.run_until_parked();

    let worktree1_id = worktree_a1.read_with(cx_a, |worktree, _| {
        assert!(worktree.has_update_observer());
        worktree.id()
    });
    let (worktree_a2, _) = project_a1
        .update(cx_a, |p, cx| {
            p.find_or_create_worktree(path!("/root-1/dir2"), true, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();

    let worktree2_id = worktree_a2.read_with(cx_a, |tree, _| {
        assert!(tree.has_update_observer());
        tree.id()
    });
    executor.run_until_parked();

    project_b1.read_with(cx_b, |project, cx| {
        assert!(project.worktree_for_id(worktree2_id, cx).is_some())
    });

    let buffer_a1 = project_a1
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree1_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();
    let buffer_b1 = project_b1
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree1_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    // Drop client A's connection.
    server.forbid_connections();
    server.disconnect_client(client_a.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT);

    project_a1.read_with(cx_a, |project, _| {
        assert!(project.is_shared());
        assert_eq!(project.collaborators().len(), 1);
    });

    project_b1.read_with(cx_b, |project, cx| {
        assert!(!project.is_disconnected(cx));
        assert_eq!(project.collaborators().len(), 1);
    });

    worktree_a1.read_with(cx_a, |tree, _| assert!(tree.has_update_observer()));

    // While client A is disconnected, add and remove files from client A's project.
    client_a
        .fs()
        .insert_tree(
            path!("/root-1/dir1/subdir2"),
            json!({
                "f.txt": "f-contents",
                "g.txt": "g-contents",
                "h.txt": "h-contents",
                "i.txt": "i-contents",
            }),
        )
        .await;
    client_a
        .fs()
        .remove_dir(
            path!("/root-1/dir1/subdir1").as_ref(),
            RemoveOptions {
                recursive: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // While client A is disconnected, add and remove worktrees from client A's project.
    project_a1.update(cx_a, |project, cx| {
        project.remove_worktree(worktree2_id, cx)
    });
    let (worktree_a3, _) = project_a1
        .update(cx_a, |p, cx| {
            p.find_or_create_worktree(path!("/root-1/dir3"), true, cx)
        })
        .await
        .unwrap();
    worktree_a3
        .read_with(cx_a, |tree, _| tree.as_local().unwrap().scan_complete())
        .await;

    let worktree3_id = worktree_a3.read_with(cx_a, |tree, _| {
        assert!(!tree.has_update_observer());
        tree.id()
    });
    executor.run_until_parked();

    // While client A is disconnected, close project 2
    cx_a.update(|_| drop(project_a2));

    // While client A is disconnected, mutate a buffer on both the host and the guest.
    buffer_a1.update(cx_a, |buf, cx| buf.edit([(0..0, "W")], None, cx));
    buffer_b1.update(cx_b, |buf, cx| buf.edit([(1..1, "Z")], None, cx));
    executor.run_until_parked();

    // Client A reconnects. Their project is re-shared, and client B re-joins it.
    server.allow_connections();
    client_a
        .connect(false, &cx_a.to_async())
        .await
        .into_response()
        .unwrap();
    executor.run_until_parked();

    project_a1.read_with(cx_a, |project, cx| {
        assert!(project.is_shared());
        assert!(worktree_a1.read(cx).has_update_observer());
        assert_eq!(
            worktree_a1.read(cx).snapshot().paths().collect::<Vec<_>>(),
            vec![
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("subdir2"),
                rel_path("subdir2/f.txt"),
                rel_path("subdir2/g.txt"),
                rel_path("subdir2/h.txt"),
                rel_path("subdir2/i.txt")
            ]
        );
        assert!(worktree_a3.read(cx).has_update_observer());
        assert_eq!(
            worktree_a3.read(cx).snapshot().paths().collect::<Vec<_>>(),
            vec![rel_path("w.txt"), rel_path("x.txt"), rel_path("y.txt")]
        );
    });

    project_b1.read_with(cx_b, |project, cx| {
        assert!(!project.is_disconnected(cx));
        assert_eq!(
            project
                .worktree_for_id(worktree1_id, cx)
                .unwrap()
                .read(cx)
                .snapshot()
                .paths()
                .collect::<Vec<_>>(),
            vec![
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("subdir2"),
                rel_path("subdir2/f.txt"),
                rel_path("subdir2/g.txt"),
                rel_path("subdir2/h.txt"),
                rel_path("subdir2/i.txt")
            ]
        );
        assert!(project.worktree_for_id(worktree2_id, cx).is_none());
        assert_eq!(
            project
                .worktree_for_id(worktree3_id, cx)
                .unwrap()
                .read(cx)
                .snapshot()
                .paths()
                .collect::<Vec<_>>(),
            vec![rel_path("w.txt"), rel_path("x.txt"), rel_path("y.txt")]
        );
    });

    project_b2.read_with(cx_b, |project, cx| assert!(project.is_disconnected(cx)));

    project_b3.read_with(cx_b, |project, cx| assert!(!project.is_disconnected(cx)));

    buffer_a1.read_with(cx_a, |buffer, _| assert_eq!(buffer.text(), "WaZ"));

    buffer_b1.read_with(cx_b, |buffer, _| assert_eq!(buffer.text(), "WaZ"));

    // Drop client B's connection.
    server.forbid_connections();
    server.disconnect_client(client_b.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT);

    // While client B is disconnected, add and remove files from client A's project
    client_a
        .fs()
        .insert_file(path!("/root-1/dir1/subdir2/j.txt"), "j-contents".into())
        .await;
    client_a
        .fs()
        .remove_file(
            path!("/root-1/dir1/subdir2/i.txt").as_ref(),
            Default::default(),
        )
        .await
        .unwrap();

    // While client B is disconnected, add and remove worktrees from client A's project.
    let (worktree_a4, _) = project_a1
        .update(cx_a, |p, cx| {
            p.find_or_create_worktree(path!("/root-1/dir4"), true, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();

    let worktree4_id = worktree_a4.read_with(cx_a, |tree, _| {
        assert!(tree.has_update_observer());
        tree.id()
    });
    project_a1.update(cx_a, |project, cx| {
        project.remove_worktree(worktree3_id, cx)
    });
    executor.run_until_parked();

    // While client B is disconnected, mutate a buffer on both the host and the guest.
    buffer_a1.update(cx_a, |buf, cx| buf.edit([(1..1, "X")], None, cx));
    buffer_b1.update(cx_b, |buf, cx| buf.edit([(2..2, "Y")], None, cx));
    executor.run_until_parked();

    // While disconnected, close project 3
    cx_a.update(|_| drop(project_a3));
    executor.run_until_parked();

    // Client B reconnects. They re-join the room and the remaining shared project.
    server.allow_connections();
    client_b
        .connect(false, &cx_b.to_async())
        .await
        .into_response()
        .unwrap();
    executor.run_until_parked();

    project_b1.read_with(cx_b, |project, cx| {
        assert!(!project.is_disconnected(cx));
        assert_eq!(
            project
                .worktree_for_id(worktree1_id, cx)
                .unwrap()
                .read(cx)
                .snapshot()
                .paths()
                .collect::<Vec<_>>(),
            vec![
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("subdir2"),
                rel_path("subdir2/f.txt"),
                rel_path("subdir2/g.txt"),
                rel_path("subdir2/h.txt"),
                rel_path("subdir2/j.txt")
            ]
        );
        assert!(project.worktree_for_id(worktree2_id, cx).is_none());
        assert_eq!(
            project
                .worktree_for_id(worktree4_id, cx)
                .unwrap()
                .read(cx)
                .snapshot()
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            vec!["z.txt"]
        );
    });

    project_b3.read_with(cx_b, |project, cx| assert!(project.is_disconnected(cx)));

    buffer_a1.read_with(cx_a, |buffer, _| assert_eq!(buffer.text(), "WXaYZ"));

    buffer_b1.read_with(cx_b, |buffer, _| assert_eq!(buffer.text(), "WXaYZ"));
}

#[gpui::test(iterations = 10)]
async fn test_active_call_events(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    client_a.fs().insert_tree("/a", json!({})).await;
    client_b.fs().insert_tree("/b", json!({})).await;

    let (project_a, _) = client_a.build_local_project("/a", cx_a).await;
    let (project_b, _) = client_b.build_local_project("/b", cx_b).await;

    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    executor.run_until_parked();

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    let events_a = active_call_events(cx_a);
    let events_b = active_call_events(cx_b);

    let project_a_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(mem::take(&mut *events_a.borrow_mut()), vec![]);
    assert_eq!(
        mem::take(&mut *events_b.borrow_mut()),
        vec![room::Event::RemoteProjectShared {
            owner: Arc::new(User {
                legacy_id: client_a.user_id().unwrap(),
                username: "user_a".into(),
                avatar_uri: "avatar_a".into(),
                name: None,
            }),
            project_id: project_a_id,
            worktree_root_names: vec!["a".to_string()],
        }]
    );

    let project_b_id = active_call_b
        .update(cx_b, |call, cx| call.share_project(project_b.clone(), cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        mem::take(&mut *events_a.borrow_mut()),
        vec![room::Event::RemoteProjectShared {
            owner: Arc::new(User {
                legacy_id: client_b.user_id().unwrap(),
                username: "user_b".into(),
                avatar_uri: "avatar_b".into(),
                name: None,
            }),
            project_id: project_b_id,
            worktree_root_names: vec!["b".to_string()]
        }]
    );
    assert_eq!(mem::take(&mut *events_b.borrow_mut()), vec![]);

    // Sharing a project twice is idempotent.
    let project_b_id_2 = active_call_b
        .update(cx_b, |call, cx| call.share_project(project_b.clone(), cx))
        .await
        .unwrap();
    assert_eq!(project_b_id_2, project_b_id);
    executor.run_until_parked();
    assert_eq!(mem::take(&mut *events_a.borrow_mut()), vec![]);
    assert_eq!(mem::take(&mut *events_b.borrow_mut()), vec![]);

    // Unsharing a project should dispatch the RemoteProjectUnshared event.
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    assert_eq!(
        mem::take(&mut *events_a.borrow_mut()),
        vec![room::Event::RoomLeft { channel_id: None }]
    );
    assert_eq!(
        mem::take(&mut *events_b.borrow_mut()),
        vec![room::Event::RemoteProjectUnshared {
            project_id: project_a_id,
        }]
    );
}

fn active_call_events(cx: &mut TestAppContext) -> Rc<RefCell<Vec<room::Event>>> {
    let events = Rc::new(RefCell::new(Vec::new()));
    let active_call = cx.read(ActiveCall::global);
    cx.update({
        let events = events.clone();
        |cx| {
            cx.subscribe(&active_call, move |_, event, _| {
                events.borrow_mut().push(event.clone())
            })
            .detach()
        }
    });
    events
}

#[gpui::test]
async fn test_mute_deafen(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;

    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);

    // User A calls user B, B answers.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());

    room_a.read_with(cx_a, |room, _| assert!(!room.is_muted()));
    room_b.read_with(cx_b, |room, _| assert!(!room.is_muted()));

    // Users A and B are both unmuted.
    assert_eq!(
        participant_audio_state(&room_a, cx_a),
        &[ParticipantAudioState {
            user_id: client_b.user_id().unwrap(),
            is_muted: false,
            audio_tracks_playing: vec![true],
        }]
    );
    assert_eq!(
        participant_audio_state(&room_b, cx_b),
        &[ParticipantAudioState {
            user_id: client_a.user_id().unwrap(),
            is_muted: false,
            audio_tracks_playing: vec![true],
        }]
    );

    // User A mutes
    room_a.update(cx_a, |room, cx| room.toggle_mute(cx));
    executor.run_until_parked();

    // User A hears user B, but B doesn't hear A.
    room_a.read_with(cx_a, |room, _| assert!(room.is_muted()));
    room_b.read_with(cx_b, |room, _| assert!(!room.is_muted()));
    assert_eq!(
        participant_audio_state(&room_a, cx_a),
        &[ParticipantAudioState {
            user_id: client_b.user_id().unwrap(),
            is_muted: false,
            audio_tracks_playing: vec![true],
        }]
    );
    assert_eq!(
        participant_audio_state(&room_b, cx_b),
        &[ParticipantAudioState {
            user_id: client_a.user_id().unwrap(),
            is_muted: true,
            audio_tracks_playing: vec![true],
        }]
    );

    // User A deafens
    room_a.update(cx_a, |room, cx| room.toggle_deafen(cx));
    executor.run_until_parked();

    // User A does not hear user B.
    room_a.read_with(cx_a, |room, _| assert!(room.is_muted()));
    room_b.read_with(cx_b, |room, _| assert!(!room.is_muted()));
    assert_eq!(
        participant_audio_state(&room_a, cx_a),
        &[ParticipantAudioState {
            user_id: client_b.user_id().unwrap(),
            is_muted: false,
            audio_tracks_playing: vec![false],
        }]
    );
    assert_eq!(
        participant_audio_state(&room_b, cx_b),
        &[ParticipantAudioState {
            user_id: client_a.user_id().unwrap(),
            is_muted: true,
            audio_tracks_playing: vec![true],
        }]
    );

    // User B calls user C, C joins.
    active_call_b
        .update(cx_b, |call, cx| {
            call.invite(client_c.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    active_call_c
        .update(cx_c, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    // User A does not hear users B or C.
    assert_eq!(
        participant_audio_state(&room_a, cx_a),
        &[
            ParticipantAudioState {
                user_id: client_b.user_id().unwrap(),
                is_muted: false,
                audio_tracks_playing: vec![false],
            },
            ParticipantAudioState {
                user_id: client_c.user_id().unwrap(),
                is_muted: false,
                audio_tracks_playing: vec![false],
            }
        ]
    );
    assert_eq!(
        participant_audio_state(&room_b, cx_b),
        &[
            ParticipantAudioState {
                user_id: client_a.user_id().unwrap(),
                is_muted: true,
                audio_tracks_playing: vec![true],
            },
            ParticipantAudioState {
                user_id: client_c.user_id().unwrap(),
                is_muted: false,
                audio_tracks_playing: vec![true],
            }
        ]
    );

    #[derive(PartialEq, Eq, Debug)]
    struct ParticipantAudioState {
        user_id: u64,
        is_muted: bool,
        audio_tracks_playing: Vec<bool>,
    }

    fn participant_audio_state(
        room: &Entity<Room>,
        cx: &TestAppContext,
    ) -> Vec<ParticipantAudioState> {
        room.read_with(cx, |room, _| {
            room.remote_participants()
                .iter()
                .map(|(user_id, participant)| ParticipantAudioState {
                    user_id: *user_id,
                    is_muted: participant.muted,
                    audio_tracks_playing: participant
                        .audio_tracks
                        .values()
                        .map(|(track, _)| track.enabled())
                        .collect(),
                })
                .collect::<Vec<_>>()
        })
    }
}

#[gpui::test(iterations = 10)]
async fn test_room_location(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    client_a.fs().insert_tree("/a", json!({})).await;
    client_b.fs().insert_tree("/b", json!({})).await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    let a_notified = Rc::new(Cell::new(false));
    cx_a.update({
        let notified = a_notified.clone();
        |cx| {
            cx.observe(&active_call_a, move |_, _| notified.set(true))
                .detach()
        }
    });

    let b_notified = Rc::new(Cell::new(false));
    cx_b.update({
        let b_notified = b_notified.clone();
        |cx| {
            cx.observe(&active_call_b, move |_, _| b_notified.set(true))
                .detach()
        }
    });

    let (project_a, _) = client_a.build_local_project("/a", cx_a).await;
    active_call_a
        .update(cx_a, |call, cx| call.set_location(Some(&project_a), cx))
        .await
        .unwrap();
    let (project_b, _) = client_b.build_local_project("/b", cx_b).await;

    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert!(a_notified.take());
    assert_eq!(
        participant_locations(&room_a, cx_a),
        vec![("user_b".to_string(), ParticipantLocation::External)]
    );
    assert!(b_notified.take());
    assert_eq!(
        participant_locations(&room_b, cx_b),
        vec![("user_a".to_string(), ParticipantLocation::UnsharedProject)]
    );

    let project_a_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(a_notified.take());
    assert_eq!(
        participant_locations(&room_a, cx_a),
        vec![("user_b".to_string(), ParticipantLocation::External)]
    );
    assert!(b_notified.take());
    assert_eq!(
        participant_locations(&room_b, cx_b),
        vec![(
            "user_a".to_string(),
            ParticipantLocation::SharedProject {
                project_id: project_a_id
            }
        )]
    );

    let project_b_id = active_call_b
        .update(cx_b, |call, cx| call.share_project(project_b.clone(), cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(a_notified.take());
    assert_eq!(
        participant_locations(&room_a, cx_a),
        vec![("user_b".to_string(), ParticipantLocation::External)]
    );
    assert!(b_notified.take());
    assert_eq!(
        participant_locations(&room_b, cx_b),
        vec![(
            "user_a".to_string(),
            ParticipantLocation::SharedProject {
                project_id: project_a_id
            }
        )]
    );

    active_call_b
        .update(cx_b, |call, cx| call.set_location(Some(&project_b), cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(a_notified.take());
    assert_eq!(
        participant_locations(&room_a, cx_a),
        vec![(
            "user_b".to_string(),
            ParticipantLocation::SharedProject {
                project_id: project_b_id
            }
        )]
    );
    assert!(b_notified.take());
    assert_eq!(
        participant_locations(&room_b, cx_b),
        vec![(
            "user_a".to_string(),
            ParticipantLocation::SharedProject {
                project_id: project_a_id
            }
        )]
    );

    active_call_b
        .update(cx_b, |call, cx| call.set_location(None, cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert!(a_notified.take());
    assert_eq!(
        participant_locations(&room_a, cx_a),
        vec![("user_b".to_string(), ParticipantLocation::External)]
    );
    assert!(b_notified.take());
    assert_eq!(
        participant_locations(&room_b, cx_b),
        vec![(
            "user_a".to_string(),
            ParticipantLocation::SharedProject {
                project_id: project_a_id
            }
        )]
    );

    fn participant_locations(
        room: &Entity<Room>,
        cx: &TestAppContext,
    ) -> Vec<(String, ParticipantLocation)> {
        room.read_with(cx, |room, _| {
            room.remote_participants()
                .values()
                .map(|participant| (participant.user.username.to_string(), participant.location))
                .collect()
        })
    }
}

#[gpui::test(iterations = 10)]
async fn test_propagate_saves_and_fs_changes(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;

    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let rust = Arc::new(Language::new(
        LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".to_string()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_rust::LANGUAGE.into()),
    ));
    let javascript = Arc::new(Language::new(
        LanguageConfig {
            name: "JavaScript".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["js".to_string()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_rust::LANGUAGE.into()),
    ));
    for client in [&client_a, &client_b, &client_c] {
        client.language_registry().add(rust.clone());
        client.language_registry().add(javascript.clone());
    }

    client_a
        .fs()
        .insert_tree(
            path!("/a"),
            json!({
                "file1.rs": "",
                "file2": ""
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/a"), cx_a).await;

    let worktree_a = project_a.read_with(cx_a, |p, cx| p.worktrees(cx).next().unwrap());
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    // Join that worktree as clients B and C.
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let project_c = client_c.join_remote_project(project_id, cx_c).await;

    let worktree_b = project_b.read_with(cx_b, |p, cx| p.worktrees(cx).next().unwrap());

    let worktree_c = project_c.read_with(cx_c, |p, cx| p.worktrees(cx).next().unwrap());

    // Open and edit a buffer as both guests B and C.
    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("file1.rs")), cx)
        })
        .await
        .unwrap();
    let buffer_c = project_c
        .update(cx_c, |p, cx| {
            p.open_buffer((worktree_id, rel_path("file1.rs")), cx)
        })
        .await
        .unwrap();

    buffer_b.read_with(cx_b, |buffer, _| {
        assert_eq!(buffer.language().unwrap().name(), "Rust");
    });

    buffer_c.read_with(cx_c, |buffer, _| {
        assert_eq!(buffer.language().unwrap().name(), "Rust");
    });
    buffer_b.update(cx_b, |buf, cx| buf.edit([(0..0, "i-am-b, ")], None, cx));
    buffer_c.update(cx_c, |buf, cx| buf.edit([(0..0, "i-am-c, ")], None, cx));

    // Open and edit that buffer as the host.
    let buffer_a = project_a
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("file1.rs")), cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();

    buffer_a.read_with(cx_a, |buf, _| assert_eq!(buf.text(), "i-am-c, i-am-b, "));
    buffer_a.update(cx_a, |buf, cx| {
        buf.edit([(buf.len()..buf.len(), "i-am-a")], None, cx)
    });

    executor.run_until_parked();

    buffer_a.read_with(cx_a, |buf, _| {
        assert_eq!(buf.text(), "i-am-c, i-am-b, i-am-a");
    });

    buffer_b.read_with(cx_b, |buf, _| {
        assert_eq!(buf.text(), "i-am-c, i-am-b, i-am-a");
    });

    buffer_c.read_with(cx_c, |buf, _| {
        assert_eq!(buf.text(), "i-am-c, i-am-b, i-am-a");
    });

    // Edit the buffer as the host and concurrently save as guest B.
    let save_b = project_b.update(cx_b, |project, cx| {
        project.save_buffer(buffer_b.clone(), cx)
    });
    buffer_a.update(cx_a, |buf, cx| buf.edit([(0..0, "hi-a, ")], None, cx));
    save_b.await.unwrap();
    assert_eq!(
        client_a.fs().load("/a/file1.rs".as_ref()).await.unwrap(),
        "hi-a, i-am-c, i-am-b, i-am-a"
    );

    executor.run_until_parked();

    buffer_a.read_with(cx_a, |buf, _| assert!(!buf.is_dirty()));

    buffer_b.read_with(cx_b, |buf, _| assert!(!buf.is_dirty()));

    buffer_c.read_with(cx_c, |buf, _| assert!(!buf.is_dirty()));

    // Make changes on host's file system, see those changes on guest worktrees.
    client_a
        .fs()
        .rename(
            path!("/a/file1.rs").as_ref(),
            path!("/a/file1.js").as_ref(),
            Default::default(),
        )
        .await
        .unwrap();
    client_a
        .fs()
        .rename(
            path!("/a/file2").as_ref(),
            path!("/a/file3").as_ref(),
            Default::default(),
        )
        .await
        .unwrap();
    client_a
        .fs()
        .insert_file(path!("/a/file4"), "4".into())
        .await;
    executor.run_until_parked();

    worktree_a.read_with(cx_a, |tree, _| {
        assert_eq!(
            tree.paths().collect::<Vec<_>>(),
            [rel_path("file1.js"), rel_path("file3"), rel_path("file4")]
        )
    });

    worktree_b.read_with(cx_b, |tree, _| {
        assert_eq!(
            tree.paths().collect::<Vec<_>>(),
            [rel_path("file1.js"), rel_path("file3"), rel_path("file4")]
        )
    });

    worktree_c.read_with(cx_c, |tree, _| {
        assert_eq!(
            tree.paths().collect::<Vec<_>>(),
            [rel_path("file1.js"), rel_path("file3"), rel_path("file4")]
        )
    });

    // Ensure buffer files are updated as well.

    buffer_a.read_with(cx_a, |buffer, _| {
        assert_eq!(buffer.file().unwrap().path().as_ref(), rel_path("file1.js"));
        assert_eq!(buffer.language().unwrap().name(), "JavaScript");
    });

    buffer_b.read_with(cx_b, |buffer, _| {
        assert_eq!(buffer.file().unwrap().path().as_ref(), rel_path("file1.js"));
        assert_eq!(buffer.language().unwrap().name(), "JavaScript");
    });

    buffer_c.read_with(cx_c, |buffer, _| {
        assert_eq!(buffer.file().unwrap().path().as_ref(), rel_path("file1.js"));
        assert_eq!(buffer.language().unwrap().name(), "JavaScript");
    });

    let new_buffer_a = project_a
        .update(cx_a, |p, cx| p.create_buffer(None, false, cx))
        .await
        .unwrap();

    let new_buffer_id = new_buffer_a.read_with(cx_a, |buffer, _| buffer.remote_id());
    let new_buffer_b = project_b
        .update(cx_b, |p, cx| p.open_buffer_by_id(new_buffer_id, cx))
        .await
        .unwrap();

    new_buffer_b.read_with(cx_b, |buffer, _| {
        assert!(buffer.file().is_none());
    });

    new_buffer_a.update(cx_a, |buffer, cx| {
        buffer.edit([(0..0, "ok")], None, cx);
    });
    project_a
        .update(cx_a, |project, cx| {
            let path = ProjectPath {
                path: rel_path("file3.rs").into(),
                worktree_id: worktree_a.read(cx).id(),
            };

            project.save_buffer_as(new_buffer_a.clone(), path, cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();

    new_buffer_b.read_with(cx_b, |buffer_b, _| {
        assert_eq!(
            buffer_b.file().unwrap().path().as_ref(),
            rel_path("file3.rs")
        );

        new_buffer_a.read_with(cx_a, |buffer_a, _| {
            assert_eq!(buffer_b.saved_mtime(), buffer_a.saved_mtime());
            assert_eq!(buffer_b.saved_version(), buffer_a.saved_version());
        });
    });
}

#[gpui::test(iterations = 10)]
async fn test_git_diff_base_change(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/dir",
            json!({
            ".git": {},
            "sub": {
                ".git": {},
                "b.txt": "
                    one
                    two
                    three
                ".unindent(),
            },
            "a.txt": "
                    one
                    two
                    three
                ".unindent(),
            }),
        )
        .await;

    let (project_local, worktree_id) = client_a.build_local_project("/dir", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| {
            call.share_project(project_local.clone(), cx)
        })
        .await
        .unwrap();

    let project_remote = client_b.join_remote_project(project_id, cx_b).await;

    let staged_text = "
        one
        three
    "
    .unindent();

    let committed_text = "
        one
        TWO
        three
    "
    .unindent();

    let new_committed_text = "
        one
        TWO_HUNDRED
        three
    "
    .unindent();

    let new_staged_text = "
        one
        two
    "
    .unindent();

    client_a
        .fs()
        .set_index_for_repo(Path::new("/dir/.git"), &[("a.txt", staged_text.clone())]);
    client_a.fs().set_head_for_repo(
        Path::new("/dir/.git"),
        &[("a.txt", committed_text.clone())],
        "deadbeef",
    );

    // Create the buffer
    let buffer_local_a = project_local
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();
    let local_unstaged_diff_a = project_local
        .update(cx_a, |p, cx| {
            p.open_unstaged_diff(buffer_local_a.clone(), cx)
        })
        .await
        .unwrap();

    // Wait for it to catch up to the new diff
    executor.run_until_parked();
    local_unstaged_diff_a.read_with(cx_a, |diff, cx| {
        let buffer = buffer_local_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(1..2, "", "two\n", DiffHunkStatus::added_none())],
        );
    });

    // Create remote buffer
    let remote_buffer_a = project_remote
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();
    let remote_unstaged_diff_a = project_remote
        .update(cx_b, |p, cx| {
            p.open_unstaged_diff(remote_buffer_a.clone(), cx)
        })
        .await
        .unwrap();

    // Wait remote buffer to catch up to the new diff
    executor.run_until_parked();
    remote_unstaged_diff_a.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(1..2, "", "two\n", DiffHunkStatus::added_none())],
        );
    });

    // Open uncommitted changes on the guest, without opening them on the host first
    let remote_uncommitted_diff_a = project_remote
        .update(cx_b, |p, cx| {
            p.open_uncommitted_diff(remote_buffer_a.clone(), cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    remote_uncommitted_diff_a.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(committed_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(
                1..2,
                "TWO\n",
                "two\n",
                DiffHunkStatus::modified(DiffHunkSecondaryStatus::HasSecondaryHunk),
            )],
        );
    });

    // Update the index text of the open buffer
    client_a.fs().set_index_for_repo(
        Path::new("/dir/.git"),
        &[("a.txt", new_staged_text.clone())],
    );
    client_a.fs().set_head_for_repo(
        Path::new("/dir/.git"),
        &[("a.txt", new_committed_text.clone())],
        "deadbeef",
    );

    // Wait for buffer_local_a to receive it
    executor.run_until_parked();
    local_unstaged_diff_a.read_with(cx_a, |diff, cx| {
        let buffer = buffer_local_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(new_staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(2..3, "", "three\n", DiffHunkStatus::added_none())],
        );
    });

    // Guest receives index text update
    remote_unstaged_diff_a.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(new_staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(2..3, "", "three\n", DiffHunkStatus::added_none())],
        );
    });

    remote_uncommitted_diff_a.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_a.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(new_committed_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(
                1..2,
                "TWO_HUNDRED\n",
                "two\n",
                DiffHunkStatus::modified(DiffHunkSecondaryStatus::OverlapsWithSecondaryHunk),
            )],
        );
    });

    // Nested git dir
    let staged_text = "
        one
        three
    "
    .unindent();

    let new_staged_text = "
        one
        two
    "
    .unindent();

    client_a.fs().set_index_for_repo(
        Path::new("/dir/sub/.git"),
        &[("b.txt", staged_text.clone())],
    );

    // Create the buffer
    let buffer_local_b = project_local
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("sub/b.txt")), cx)
        })
        .await
        .unwrap();
    let local_unstaged_diff_b = project_local
        .update(cx_a, |p, cx| {
            p.open_unstaged_diff(buffer_local_b.clone(), cx)
        })
        .await
        .unwrap();

    // Wait for it to catch up to the new diff
    executor.run_until_parked();
    local_unstaged_diff_b.read_with(cx_a, |diff, cx| {
        let buffer = buffer_local_b.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(1..2, "", "two\n", DiffHunkStatus::added_none())],
        );
    });

    // Create remote buffer
    let remote_buffer_b = project_remote
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("sub/b.txt")), cx)
        })
        .await
        .unwrap();
    let remote_unstaged_diff_b = project_remote
        .update(cx_b, |p, cx| {
            p.open_unstaged_diff(remote_buffer_b.clone(), cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();
    remote_unstaged_diff_b.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_b.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &staged_text,
            &[(1..2, "", "two\n", DiffHunkStatus::added_none())],
        );
    });

    // Updatet the staged text
    client_a.fs().set_index_for_repo(
        Path::new("/dir/sub/.git"),
        &[("b.txt", new_staged_text.clone())],
    );

    // Wait for buffer_local_b to receive it
    executor.run_until_parked();
    local_unstaged_diff_b.read_with(cx_a, |diff, cx| {
        let buffer = buffer_local_b.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(new_staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &new_staged_text,
            &[(2..3, "", "three\n", DiffHunkStatus::added_none())],
        );
    });

    remote_unstaged_diff_b.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer_b.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(new_staged_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..4, buffer),
            buffer,
            &new_staged_text,
            &[(2..3, "", "three\n", DiffHunkStatus::added_none())],
        );
    });
}

#[gpui::test(iterations = 10)]
async fn test_git_diff_index_matches_head(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let committed_text = "
        one
        two
        three
    "
    .unindent();
    let file_contents = "
        one
        TWO
        three
    "
    .unindent();

    client_a
        .fs()
        .insert_tree(
            "/dir",
            json!({
                ".git": {},
                "a.txt": file_contents,
            }),
        )
        .await;
    client_a
        .fs()
        .set_head_and_index_for_repo(Path::new("/dir/.git"), &[("a.txt", committed_text.clone())]);

    let (project_local, worktree_id) = client_a.build_local_project("/dir", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| {
            call.share_project(project_local.clone(), cx)
        })
        .await
        .unwrap();
    let project_remote = client_b.join_remote_project(project_id, cx_b).await;

    // Open the uncommitted diff on the guest, without opening it on the host
    // first, so that the host loads the diff bases in response to the guest's
    // request.
    let remote_buffer = project_remote
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();
    let remote_uncommitted_diff = project_remote
        .update(cx_b, |p, cx| {
            p.open_uncommitted_diff(remote_buffer.clone(), cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();

    // The guest's index and head texts share one allocation, which is only
    // possible if the host detected that the index matches the head and sent
    // `Mode::IndexMatchesHead`.
    let buffer_id = remote_buffer.read_with(cx_b, |buffer, _| buffer.remote_id());
    project_remote.read_with(cx_b, |project, cx| {
        assert!(
            project
                .git_store()
                .read(cx)
                .index_matches_head_for_buffer(buffer_id, cx),
            "the host should send IndexMatchesHead when the index is clean"
        );
    });

    remote_uncommitted_diff.read_with(cx_b, |diff, cx| {
        let buffer = remote_buffer.read(cx);
        assert_eq!(
            diff.base_text_string(cx).as_deref(),
            Some(committed_text.as_str())
        );
        assert_hunks(
            diff.snapshot(cx).hunks_in_row_range(0..3, buffer),
            buffer,
            &diff.base_text_string(cx).unwrap(),
            &[(
                1..2,
                "two\n",
                "TWO\n",
                DiffHunkStatus::modified(DiffHunkSecondaryStatus::HasSecondaryHunk),
            )],
        );
    });
}

#[gpui::test(iterations = 10)]
async fn test_git_branch_name(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/dir",
            json!({
            ".git": {},
            }),
        )
        .await;

    let (project_local, _worktree_id) = client_a.build_local_project("/dir", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| {
            call.share_project(project_local.clone(), cx)
        })
        .await
        .unwrap();

    let project_remote = client_b.join_remote_project(project_id, cx_b).await;
    client_a
        .fs()
        .set_branch_name(Path::new("/dir/.git"), Some("branch-1"));

    // Wait for it to catch up to the new branch
    executor.run_until_parked();

    #[track_caller]
    fn assert_branch(branch_name: Option<impl Into<String>>, project: &Project, cx: &App) {
        let branch_name = branch_name.map(Into::into);
        let repositories = project.repositories(cx).values().collect::<Vec<_>>();
        assert_eq!(repositories.len(), 1);
        let repository = repositories[0].clone();
        assert_eq!(
            repository
                .read(cx)
                .branch
                .as_ref()
                .map(|branch| branch.name().to_owned()),
            branch_name
        )
    }

    // Smoke test branch reading

    project_local.read_with(cx_a, |project, cx| {
        assert_branch(Some("branch-1"), project, cx)
    });

    project_remote.read_with(cx_b, |project, cx| {
        assert_branch(Some("branch-1"), project, cx)
    });

    client_a
        .fs()
        .set_branch_name(Path::new("/dir/.git"), Some("branch-2"));

    // Wait for buffer_local_a to receive it
    executor.run_until_parked();

    // Smoke test branch reading

    project_local.read_with(cx_a, |project, cx| {
        assert_branch(Some("branch-2"), project, cx)
    });

    project_remote.read_with(cx_b, |project, cx| {
        assert_branch(Some("branch-2"), project, cx)
    });

    let project_remote_c = client_c.join_remote_project(project_id, cx_c).await;
    executor.run_until_parked();

    project_remote_c.read_with(cx_c, |project, cx| {
        assert_branch(Some("branch-2"), project, cx)
    });
}

#[gpui::test]
async fn test_git_status_sync(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/dir"),
            json!({
                ".git": {},
                "a.txt": "a",
                "b.txt": "b",
                "c.txt": "c",
            }),
        )
        .await;

    // Initially, a.txt is uncommitted, but present in the index,
    // and b.txt is unmerged.
    client_a.fs().set_head_for_repo(
        path!("/dir/.git").as_ref(),
        &[("b.txt", "B".into()), ("c.txt", "c".into())],
        "deadbeef",
    );
    client_a.fs().set_index_for_repo(
        path!("/dir/.git").as_ref(),
        &[
            ("a.txt", "".into()),
            ("b.txt", "B".into()),
            ("c.txt", "c".into()),
        ],
    );
    client_a.fs().set_unmerged_paths_for_repo(
        path!("/dir/.git").as_ref(),
        &[(
            repo_path("b.txt"),
            UnmergedStatus {
                first_head: UnmergedStatusCode::Updated,
                second_head: UnmergedStatusCode::Deleted,
            },
        )],
    );

    const A_STATUS_START: FileStatus = FileStatus::Tracked(TrackedStatus {
        index_status: StatusCode::Added,
        worktree_status: StatusCode::Modified,
    });
    const B_STATUS_START: FileStatus = FileStatus::Unmerged(UnmergedStatus {
        first_head: UnmergedStatusCode::Updated,
        second_head: UnmergedStatusCode::Deleted,
    });

    let (project_local, _worktree_id) = client_a.build_local_project(path!("/dir"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| {
            call.share_project(project_local.clone(), cx)
        })
        .await
        .unwrap();

    let project_remote = client_b.join_remote_project(project_id, cx_b).await;

    // Wait for it to catch up to the new status
    executor.run_until_parked();

    #[track_caller]
    fn assert_status(file: &str, status: Option<FileStatus>, project: &Project, cx: &App) {
        let file = repo_path(file);
        let repos = project
            .repositories(cx)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(repos.len(), 1);
        let repo = repos.into_iter().next().unwrap();
        assert_eq!(
            repo.read(cx)
                .status_for_path(&file)
                .map(|entry| entry.status),
            status
        );
    }

    project_local.read_with(cx_a, |project, cx| {
        assert_status("a.txt", Some(A_STATUS_START), project, cx);
        assert_status("b.txt", Some(B_STATUS_START), project, cx);
        assert_status("c.txt", None, project, cx);
    });

    project_remote.read_with(cx_b, |project, cx| {
        assert_status("a.txt", Some(A_STATUS_START), project, cx);
        assert_status("b.txt", Some(B_STATUS_START), project, cx);
        assert_status("c.txt", None, project, cx);
    });

    const A_STATUS_END: FileStatus = FileStatus::Tracked(TrackedStatus {
        index_status: StatusCode::Added,
        worktree_status: StatusCode::Unmodified,
    });
    const B_STATUS_END: FileStatus = FileStatus::Tracked(TrackedStatus {
        index_status: StatusCode::Deleted,
        worktree_status: StatusCode::Added,
    });
    const C_STATUS_END: FileStatus = FileStatus::Tracked(TrackedStatus {
        index_status: StatusCode::Unmodified,
        worktree_status: StatusCode::Modified,
    });

    // Delete b.txt from the index, mark conflict as resolved,
    // and modify c.txt in the working copy.
    client_a.fs().set_index_for_repo(
        path!("/dir/.git").as_ref(),
        &[("a.txt", "a".into()), ("c.txt", "c".into())],
    );
    client_a
        .fs()
        .set_unmerged_paths_for_repo(path!("/dir/.git").as_ref(), &[]);
    client_a
        .fs()
        .atomic_write(path!("/dir/c.txt").into(), "CC".into())
        .await
        .unwrap();

    // Wait for buffer_local_a to receive it
    executor.run_until_parked();

    // Smoke test status reading
    project_local.read_with(cx_a, |project, cx| {
        assert_status("a.txt", Some(A_STATUS_END), project, cx);
        assert_status("b.txt", Some(B_STATUS_END), project, cx);
        assert_status("c.txt", Some(C_STATUS_END), project, cx);
    });

    project_remote.read_with(cx_b, |project, cx| {
        assert_status("a.txt", Some(A_STATUS_END), project, cx);
        assert_status("b.txt", Some(B_STATUS_END), project, cx);
        assert_status("c.txt", Some(C_STATUS_END), project, cx);
    });

    // And synchronization while joining
    let project_remote_c = client_c.join_remote_project(project_id, cx_c).await;
    executor.run_until_parked();

    project_remote_c.read_with(cx_c, |project, cx| {
        assert_status("a.txt", Some(A_STATUS_END), project, cx);
        assert_status("b.txt", Some(B_STATUS_END), project, cx);
        assert_status("c.txt", Some(C_STATUS_END), project, cx);
    });

    // Now remove the original git repository and check that collaborators are notified.
    client_a
        .fs()
        .remove_dir(path!("/dir/.git").as_ref(), RemoveOptions::default())
        .await
        .unwrap();

    executor.run_until_parked();
    project_remote.update(cx_b, |project, cx| {
        pretty_assertions::assert_eq!(
            project.git_store().read(cx).repo_snapshots(cx),
            HashMap::default()
        );
    });
    project_remote_c.update(cx_c, |project, cx| {
        pretty_assertions::assert_eq!(
            project.git_store().read(cx).repo_snapshots(cx),
            HashMap::default()
        );
    });
}

#[gpui::test(iterations = 10)]
async fn test_fs_operations(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/dir"),
            json!({
                "a.txt": "a-contents",
                "b.txt": "b-contents",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/dir"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let worktree_a = project_a.read_with(cx_a, |project, cx| project.worktrees(cx).next().unwrap());
    let worktree_b = project_b.read_with(cx_b, |project, cx| project.worktrees(cx).next().unwrap());

    let entry = project_b
        .update(cx_b, |project, cx| {
            project.create_entry((worktree_id, rel_path("c.txt")), false, cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [rel_path("a.txt"), rel_path("b.txt"), rel_path("c.txt")]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [rel_path("a.txt"), rel_path("b.txt"), rel_path("c.txt")]
        );
    });

    project_b
        .update(cx_b, |project, cx| {
            project.rename_entry(entry.id, (worktree_id, rel_path("d.txt")).into(), cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [rel_path("a.txt"), rel_path("b.txt"), rel_path("d.txt")]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["a.txt", "b.txt", "d.txt"]
        );
    });

    let dir_entry = project_b
        .update(cx_b, |project, cx| {
            project.create_entry((worktree_id, rel_path("DIR")), true, cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["DIR", "a.txt", "b.txt", "d.txt"]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["DIR", "a.txt", "b.txt", "d.txt"]
        );
    });

    project_b
        .update(cx_b, |project, cx| {
            project.create_entry((worktree_id, rel_path("DIR/e.txt")), false, cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    project_b
        .update(cx_b, |project, cx| {
            project.create_entry((worktree_id, rel_path("DIR/SUBDIR")), true, cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    project_b
        .update(cx_b, |project, cx| {
            project.create_entry((worktree_id, rel_path("DIR/SUBDIR/f.txt")), false, cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [
                rel_path("DIR"),
                rel_path("DIR/SUBDIR"),
                rel_path("DIR/SUBDIR/f.txt"),
                rel_path("DIR/e.txt"),
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("d.txt")
            ]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [
                rel_path("DIR"),
                rel_path("DIR/SUBDIR"),
                rel_path("DIR/SUBDIR/f.txt"),
                rel_path("DIR/e.txt"),
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("d.txt")
            ]
        );
    });

    project_b
        .update(cx_b, |project, cx| {
            project.copy_entry(
                entry.id,
                (worktree_b.read(cx).id(), rel_path("f.txt")).into(),
                cx,
            )
        })
        .await
        .unwrap()
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [
                rel_path("DIR"),
                rel_path("DIR/SUBDIR"),
                rel_path("DIR/SUBDIR/f.txt"),
                rel_path("DIR/e.txt"),
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("d.txt"),
                rel_path("f.txt")
            ]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            [
                rel_path("DIR"),
                rel_path("DIR/SUBDIR"),
                rel_path("DIR/SUBDIR/f.txt"),
                rel_path("DIR/e.txt"),
                rel_path("a.txt"),
                rel_path("b.txt"),
                rel_path("d.txt"),
                rel_path("f.txt")
            ]
        );
    });

    project_b
        .update(cx_b, |project, cx| {
            project.delete_entry(dir_entry.id, cx).unwrap()
        })
        .await
        .unwrap();
    executor.run_until_parked();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["a.txt", "b.txt", "d.txt", "f.txt"]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["a.txt", "b.txt", "d.txt", "f.txt"]
        );
    });

    project_b
        .update(cx_b, |project, cx| {
            project.delete_entry(entry.id, cx).unwrap()
        })
        .await
        .unwrap();

    worktree_a.read_with(cx_a, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["a.txt", "b.txt", "f.txt"]
        );
    });

    worktree_b.read_with(cx_b, |worktree, _| {
        assert_eq!(
            worktree
                .paths()
                .map(|p| p.as_unix_str())
                .collect::<Vec<_>>(),
            ["a.txt", "b.txt", "f.txt"]
        );
    });
}

#[gpui::test(iterations = 10)]
async fn test_local_settings(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    // As client A, open a project that contains some local settings files
    client_a
        .fs()
        .insert_tree(
            "/dir",
            json!({
                ".zed": {
                    "settings.json": r#"{ "tab_size": 2 }"#
                },
                "a": {
                    ".zed": {
                        "settings.json": r#"{ "tab_size": 8 }"#
                    },
                    "a.txt": "a-contents",
                },
                "b": {
                    "b.txt": "b-contents",
                }
            }),
        )
        .await;
    let (project_a, _) = client_a.build_local_project("/dir", cx_a).await;
    executor.run_until_parked();
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    executor.run_until_parked();

    // As client B, join that project and observe the local settings.
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let worktree_b = project_b.read_with(cx_b, |project, cx| project.worktrees(cx).next().unwrap());
    executor.run_until_parked();
    cx_b.read(|cx| {
        let store = cx.global::<SettingsStore>();
        assert_eq!(
            store
                .local_settings(worktree_b.read(cx).id())
                .map(|(path, content)| (
                    path,
                    content.all_languages.defaults.tab_size.map(Into::into)
                ))
                .collect::<Vec<_>>(),
            &[
                (rel_path("").into(), Some(2)),
                (rel_path("a").into(), Some(8)),
            ]
        )
    });

    // As client A, update a settings file. As Client B, see the changed settings.
    client_a
        .fs()
        .insert_file("/dir/.zed/settings.json", r#"{}"#.into())
        .await;
    executor.run_until_parked();
    cx_b.read(|cx| {
        let store = cx.global::<SettingsStore>();
        assert_eq!(
            store
                .local_settings(worktree_b.read(cx).id())
                .map(|(path, content)| (
                    path,
                    content.all_languages.defaults.tab_size.map(Into::into)
                ))
                .collect::<Vec<_>>(),
            &[(rel_path("").into(), None), (rel_path("a").into(), Some(8)),]
        )
    });

    // As client A, create and remove some settings files. As client B, see the changed settings.
    client_a
        .fs()
        .remove_file("/dir/.zed/settings.json".as_ref(), Default::default())
        .await
        .unwrap();
    client_a
        .fs()
        .create_dir("/dir/b/.zed".as_ref())
        .await
        .unwrap();
    client_a
        .fs()
        .insert_file("/dir/b/.zed/settings.json", r#"{"tab_size": 4}"#.into())
        .await;
    executor.run_until_parked();
    cx_b.read(|cx| {
        let store = cx.global::<SettingsStore>();
        assert_eq!(
            store
                .local_settings(worktree_b.read(cx).id())
                .map(|(path, content)| (
                    path,
                    content.all_languages.defaults.tab_size.map(Into::into)
                ))
                .collect::<Vec<_>>(),
            &[
                (rel_path("a").into(), Some(8)),
                (rel_path("b").into(), Some(4)),
            ]
        )
    });

    // As client B, disconnect.
    server.forbid_connections();
    server.disconnect_client(client_b.peer_id().unwrap());

    // As client A, change and remove settings files while client B is disconnected.
    client_a
        .fs()
        .insert_file("/dir/a/.zed/settings.json", r#"{"hard_tabs":true}"#.into())
        .await;
    client_a
        .fs()
        .remove_file("/dir/b/.zed/settings.json".as_ref(), Default::default())
        .await
        .unwrap();
    executor.run_until_parked();

    // As client B, reconnect and see the changed settings.
    server.allow_connections();
    executor.advance_clock(RECEIVE_TIMEOUT);
    cx_b.read(|cx| {
        let store = cx.global::<SettingsStore>();
        assert_eq!(
            store
                .local_settings(worktree_b.read(cx).id())
                .map(|(path, content)| (path, content.all_languages.defaults.hard_tabs))
                .collect::<Vec<_>>(),
            &[(rel_path("a").into(), Some(true))],
        )
    });
}

#[gpui::test(iterations = 10)]
async fn test_buffer_conflict_after_save(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/dir"),
            json!({
                "a.txt": "a-contents",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/dir"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open a buffer as client B
    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    buffer_b.update(cx_b, |buf, cx| buf.edit([(0..0, "world ")], None, cx));

    buffer_b.read_with(cx_b, |buf, _| {
        assert!(buf.is_dirty());
        assert!(!buf.has_conflict());
    });

    project_b
        .update(cx_b, |project, cx| {
            project.save_buffer(buffer_b.clone(), cx)
        })
        .await
        .unwrap();

    buffer_b.read_with(cx_b, |buffer_b, _| assert!(!buffer_b.is_dirty()));

    buffer_b.read_with(cx_b, |buf, _| {
        assert!(!buf.has_conflict());
    });

    buffer_b.update(cx_b, |buf, cx| buf.edit([(0..0, "hello ")], None, cx));

    buffer_b.read_with(cx_b, |buf, _| {
        assert!(buf.is_dirty());
        assert!(!buf.has_conflict());
    });
}

#[gpui::test(iterations = 10)]
async fn test_buffer_reloading(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/dir"),
            json!({
                "a.txt": "a\nb\nc",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/dir"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open a buffer as client B
    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    buffer_b.read_with(cx_b, |buf, _| {
        assert!(!buf.is_dirty());
        assert!(!buf.has_conflict());
        assert_eq!(buf.line_ending(), LineEnding::Unix);
    });

    let new_contents = Rope::from("d\ne\nf");
    client_a
        .fs()
        .save(
            path!("/dir/a.txt").as_ref(),
            &new_contents,
            LineEnding::Windows,
        )
        .await
        .unwrap();

    executor.run_until_parked();

    buffer_b.read_with(cx_b, |buf, _| {
        assert_eq!(buf.text(), new_contents.to_string());
        assert!(!buf.is_dirty());
        assert!(!buf.has_conflict());
        assert_eq!(buf.line_ending(), LineEnding::Windows);
    });
}

#[gpui::test(iterations = 10)]
async fn test_editing_while_guest_opens_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(path!("/dir"), json!({ "a.txt": "a-contents" }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/dir"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open a buffer as client A
    let buffer_a = project_a
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    // Start opening the same buffer as client B
    let open_buffer = project_b.update(cx_b, |p, cx| {
        p.open_buffer((worktree_id, rel_path("a.txt")), cx)
    });
    let buffer_b = cx_b.executor().spawn(open_buffer);

    // Edit the buffer as client A while client B is still opening it.
    cx_b.executor().simulate_random_delay().await;
    buffer_a.update(cx_a, |buf, cx| buf.edit([(0..0, "X")], None, cx));
    cx_b.executor().simulate_random_delay().await;
    buffer_a.update(cx_a, |buf, cx| buf.edit([(1..1, "Y")], None, cx));

    let text = buffer_a.read_with(cx_a, |buf, _| buf.text());
    let buffer_b = buffer_b.await.unwrap();
    executor.run_until_parked();

    buffer_b.read_with(cx_b, |buf, _| assert_eq!(buf.text(), text));
}

#[gpui::test(iterations = 10)]
async fn test_leaving_worktree_while_opening_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree("/dir", json!({ "a.txt": "a-contents" }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project("/dir", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // See that a guest has joined as client A.
    executor.run_until_parked();

    project_a.read_with(cx_a, |p, _| assert_eq!(p.collaborators().len(), 1));

    // Begin opening a buffer as client B, but leave the project before the open completes.
    let open_buffer = project_b.update(cx_b, |p, cx| {
        p.open_buffer((worktree_id, rel_path("a.txt")), cx)
    });
    let buffer_b = cx_b.executor().spawn(open_buffer);
    cx_b.update(|_| drop(project_b));
    drop(buffer_b);

    // See that the guest has left.
    executor.run_until_parked();

    project_a.read_with(cx_a, |p, _| assert!(p.collaborators().is_empty()));
}

#[gpui::test(iterations = 10)]
async fn test_canceling_buffer_opening(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/dir",
            json!({
                "a.txt": "abc",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project("/dir", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let buffer_a = project_a
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    // Open a buffer as client B but cancel after a random amount of time.
    let buffer_b = project_b.update(cx_b, |p, cx| {
        p.open_buffer_by_id(buffer_a.read_with(cx_a, |a, _| a.remote_id()), cx)
    });
    executor.simulate_random_delay().await;
    drop(buffer_b);

    // Try opening the same buffer again as client B, and ensure we can
    // still do it despite the cancellation above.
    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_by_id(buffer_a.read_with(cx_a, |a, _| a.remote_id()), cx)
        })
        .await
        .unwrap();

    buffer_b.read_with(cx_b, |buf, _| assert_eq!(buf.text(), "abc"));
}

#[gpui::test(iterations = 10)]
async fn bounded_artifact_collaboration_admission_and_retry(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    outsider_cx: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(host_cx, "profile_host").await;
    let guest = server.create_client(guest_cx, "profile_guest").await;
    let outsider = server.create_client(outsider_cx, "profile_outsider").await;
    server
        .create_room(&mut [(&host, host_cx), (&guest, guest_cx)])
        .await;
    host.fs()
        .insert_tree(
            "/profile",
            json!({"profile.svg": "<svg>host</svg>", ".env": "private"}),
        )
        .await;
    let (host_project, worktree_id) = host.build_local_project("/profile", host_cx).await;
    let call = host_cx.read(ActiveCall::global);
    let project_id = call
        .update(host_cx, |call, cx| {
            call.share_project(host_project.clone(), cx)
        })
        .await
        .expect("share profiling fixture");
    let guest_project = guest.join_remote_project(project_id, guest_cx).await;
    executor.run_until_parked();
    assert!(
        outsider
            .request(proto::OpenBoundedBufferByPath {
                project_id,
                worktree_id: worktree_id.to_proto(),
                path: "profile.svg".into(),
                max_bytes: 1024,
            })
            .await
            .is_err(),
        "a non-member cannot obtain a shared artifact"
    );
    let path = ProjectPath {
        worktree_id,
        path: rel_path("profile.svg").into(),
    };
    let opening = guest_project.update(guest_cx, |project, cx| {
        project.open_buffer_with_limit(path.clone(), 1024, cx)
    });
    executor.simulate_random_delay().await;
    drop(opening);
    let buffer = guest_project
        .update(guest_cx, |project, cx| {
            project.open_buffer_with_limit(path.clone(), 1024, cx)
        })
        .await
        .expect("authenticated retry through the collaboration server");
    assert_eq!(
        buffer.read_with(guest_cx, |buffer, _| buffer.text()),
        "<svg>host</svg>"
    );
    for (relative_path, limit) in [(".env", 1024), ("profile.svg", 2)] {
        assert!(
            guest_project
                .update(guest_cx, |project, cx| {
                    project.open_buffer_with_limit(
                        ProjectPath {
                            worktree_id,
                            path: rel_path(relative_path).into(),
                        },
                        limit,
                        cx,
                    )
                })
                .await
                .is_err()
        );
    }
    host_project
        .update(host_cx, |project, cx| project.unshare(cx))
        .expect("unshare profiling fixture");
    executor.run_until_parked();
    assert!(
        guest_project
            .update(guest_cx, |project, cx| {
                project.open_buffer_with_limit(path, 1024, cx)
            })
            .await
            .is_err(),
        "unshared projects cannot reopen cached artifacts"
    );
}

#[gpui::test(iterations = 10)]
async fn profile_artifact_collaboration_rejects_completion_after_reshare(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    struct DelayedCompletion {
        started: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        completion: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }
    impl workspace::TerminalProvider for DelayedCompletion {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut gpui::Window,
            cx: &mut App,
        ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
            self.started
                .lock()
                .take()
                .expect("one Task start")
                .send(())
                .expect("start receiver");
            let completion = self.completion.lock().take().expect("one completion");
            cx.background_spawn(async move {
                completion.await.expect("release Task completion");
                Some(Ok(std::process::ExitStatus::default()))
            })
        }
    }
    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(host_cx, "artifact_reshare_host").await;
    let guest = server
        .create_client(guest_cx, "artifact_reshare_guest")
        .await;
    server
        .create_room(&mut [(&host, host_cx), (&guest, guest_cx)])
        .await;
    host_cx.update(editor::init);
    host.fs()
        .insert_tree("/profile", json!({"profile.svg": "<svg>host output</svg>"}))
        .await;
    let (project, _) = host.build_local_project("/profile", host_cx).await;
    let call = host_cx.read(ActiveCall::global);
    let original_id = call
        .update(host_cx, |call, cx| call.share_project(project.clone(), cx))
        .await
        .expect("share project");
    let original_guest = guest.join_remote_project(original_id, guest_cx).await;
    let (workspace, host_cx) = host.build_workspace(&project, host_cx);
    let (started, start) = futures::channel::oneshot::channel();
    let (release, completion) = futures::channel::oneshot::channel();
    let (finished, result) = futures::channel::oneshot::channel();
    let task = task::TaskTemplate {
        label: "Shared profile".into(),
        command: "explicit-profiler".into(),
        cwd: Some("/profile".into()),
        save: task::SaveStrategy::None,
        artifact: Some(task::TaskArtifact {
            path: "profile.svg".into(),
            kind: task::TaskArtifactKind::Svg,
            max_bytes: 1024,
        }),
        ..Default::default()
    }
    .resolve_task("shared-profile", &task::TaskContext::default())
    .expect("profile Task");
    workspace.update_in(host_cx, |workspace, window, cx| {
        workspace.set_terminal_provider(DelayedCompletion {
            started: Mutex::new(Some(started)),
            completion: Mutex::new(Some(completion)),
        });
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            task.clone(),
            true,
            move |result, _| finished.send(result).expect("completion receiver"),
            window,
            cx,
        );
    });
    match futures::future::select(start, executor.timer(Duration::from_secs(5))).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("Task started before sharing changes");
        }
        futures::future::Either::Right(_) => panic!("Task did not reach the completion gate"),
    }
    call.update(host_cx, |call, cx| {
        call.unshare_project(project.clone(), cx)
    })
    .expect("unshare through active call");
    executor.run_until_parked();
    let replacement_id = call
        .update(host_cx, |call, cx| call.share_project(project.clone(), cx))
        .await
        .expect("share with new identity");
    assert_ne!(original_id, replacement_id);
    let replacement_guest = guest.join_remote_project(replacement_id, guest_cx).await;
    executor.run_until_parked();
    release.send(()).expect("late successful completion");
    match futures::future::select(result, executor.timer(Duration::from_secs(5))).await {
        futures::future::Either::Left((result, _)) => {
            assert_eq!(
                result.expect("completion callback"),
                workspace::tasks::ScheduledTaskResult::Success
            );
        }
        futures::future::Either::Right(_) => panic!("Late Task completion did not finish"),
    }
    executor.run_until_parked();
    assert!(workspace.read_with(host_cx, |workspace, cx| {
        workspace.item_of_type::<editor::Editor>(cx).is_none()
    }));
    let (started, start) = futures::channel::oneshot::channel();
    let (release, completion) = futures::channel::oneshot::channel();
    let (finished, result) = futures::channel::oneshot::channel();
    workspace.update_in(host_cx, |workspace, window, cx| {
        workspace.set_terminal_provider(DelayedCompletion {
            started: Mutex::new(Some(started)),
            completion: Mutex::new(Some(completion)),
        });
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            task,
            true,
            move |result, _| finished.send(result).expect("fresh completion receiver"),
            window,
            cx,
        );
    });
    match futures::future::select(start, executor.timer(Duration::from_secs(5))).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("fresh Task started");
        }
        futures::future::Either::Right(_) => panic!("Fresh Task did not start after reshare"),
    }
    release.send(()).expect("fresh successful completion");
    match futures::future::select(result, executor.timer(Duration::from_secs(5))).await {
        futures::future::Either::Left((result, _)) => {
            assert_eq!(
                result.expect("fresh completion callback"),
                workspace::tasks::ScheduledTaskResult::Success
            );
        }
        futures::future::Either::Right(_) => panic!("Fresh Task completion did not finish"),
    }
    executor.run_until_parked();
    assert!(
        workspace.read_with(host_cx, |workspace, cx| {
            workspace.item_of_type::<editor::Editor>(cx).is_some()
        }),
        "a new Task in the replacement sharing session can open its artifact"
    );
    host_cx.deactivate_window();
    workspace.update_in(host_cx, |_, window, _| window.remove_window());
    drop(workspace);
    drop(original_guest);
    drop(replacement_guest);
    executor.run_until_parked();
}

#[gpui::test(iterations = 10)]
async fn test_leaving_project(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/a",
            json!({
                "a.txt": "a-contents",
                "b.txt": "b-contents",
            }),
        )
        .await;
    let (project_a, _) = client_a.build_local_project("/a", cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b1 = client_b.join_remote_project(project_id, cx_b).await;
    let project_c = client_c.join_remote_project(project_id, cx_c).await;

    // Client A sees that a guest has joined.
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    project_b1.read_with(cx_b, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    project_c.read_with(cx_c, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    // Client B opens a buffer.
    let buffer_b1 = project_b1
        .update(cx_b, |project, cx| {
            let worktree_id = project.worktrees(cx).next().unwrap().read(cx).id();
            project.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    buffer_b1.read_with(cx_b, |buffer, _| assert_eq!(buffer.text(), "a-contents"));

    // Drop client B's project and ensure client A and client C observe client B leaving.
    cx_b.update(|_| drop(project_b1));
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 1);
    });

    project_c.read_with(cx_c, |project, _| {
        assert_eq!(project.collaborators().len(), 1);
    });

    // Client B re-joins the project and can open buffers as before.
    let project_b2 = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    project_b2.read_with(cx_b, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    project_c.read_with(cx_c, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    let buffer_b2 = project_b2
        .update(cx_b, |project, cx| {
            let worktree_id = project.worktrees(cx).next().unwrap().read(cx).id();
            project.open_buffer((worktree_id, rel_path("a.txt")), cx)
        })
        .await
        .unwrap();

    buffer_b2.read_with(cx_b, |buffer, _| assert_eq!(buffer.text(), "a-contents"));

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 2);
    });

    // Drop client B's connection and ensure client A and client C observe client B leaving.
    client_b.disconnect(&cx_b.to_async());
    executor.advance_clock(RECONNECT_TIMEOUT);

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 1);
    });

    project_b2.read_with(cx_b, |project, cx| {
        assert!(project.is_disconnected(cx));
    });

    project_c.read_with(cx_c, |project, _| {
        assert_eq!(project.collaborators().len(), 1);
    });

    // Client B can't join the project, unless they re-join the room.
    cx_b.spawn(|cx| {
        Project::in_room(
            project_id,
            client_b.app_state.client.clone(),
            client_b.user_store().clone(),
            client_b.language_registry().clone(),
            FakeFs::new(cx.background_executor().clone()),
            cx,
        )
    })
    .await
    .unwrap_err();

    // Simulate connection loss for client C and ensure client A observes client C leaving the project.
    client_c.wait_for_current_user(cx_c).await;
    server.forbid_connections();
    server.disconnect_client(client_c.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| {
        assert_eq!(project.collaborators().len(), 0);
    });

    project_b2.read_with(cx_b, |project, cx| {
        assert!(project.is_disconnected(cx));
    });

    project_c.read_with(cx_c, |project, cx| {
        assert!(project.is_disconnected(cx));
    });
}

#[gpui::test(iterations = 10)]
async fn test_collaborating_with_diagnostics(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a.language_registry().add(Arc::new(Language::new(
        LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".to_string()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_rust::LANGUAGE.into()),
    )));
    let mut fake_language_servers = client_a
        .language_registry()
        .register_fake_lsp("Rust", Default::default());

    // Share a project as client A
    client_a
        .fs()
        .insert_tree(
            path!("/a"),
            json!({
                "a.rs": "let one = two",
                "other.rs": "",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/a"), cx_a).await;

    // Cause the language server to start.
    let _buffer = project_a
        .update(cx_a, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/a/other.rs"), cx)
        })
        .await
        .unwrap();

    // Simulate a language server reporting errors for a file.
    let mut fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server
        .receive_notification::<lsp::notification::DidOpenTextDocument>()
        .await;
    fake_language_server.notify::<lsp::notification::PublishDiagnostics>(
        lsp::PublishDiagnosticsParams {
            uri: lsp::Uri::from_file_path(path!("/a/a.rs")).unwrap(),
            version: None,
            diagnostics: vec![lsp::Diagnostic {
                severity: Some(lsp::DiagnosticSeverity::WARNING),
                range: lsp::Range::new(lsp::Position::new(0, 4), lsp::Position::new(0, 7)),
                message: "message 0".to_string(),
                ..Default::default()
            }],
        },
    );

    // Client A shares the project and, simultaneously, the language server
    // publishes a diagnostic. This is done to ensure that the server always
    // observes the latest diagnostics for a worktree.
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    fake_language_server.notify::<lsp::notification::PublishDiagnostics>(
        lsp::PublishDiagnosticsParams {
            uri: lsp::Uri::from_file_path(path!("/a/a.rs")).unwrap(),
            version: None,
            diagnostics: vec![lsp::Diagnostic {
                severity: Some(lsp::DiagnosticSeverity::ERROR),
                range: lsp::Range::new(lsp::Position::new(0, 4), lsp::Position::new(0, 7)),
                message: "message 1".to_string(),
                ..Default::default()
            }],
        },
    );

    // Join the worktree as client B.
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Wait for server to see the diagnostics update.
    executor.run_until_parked();

    // Ensure client B observes the new diagnostics.

    project_b.read_with(cx_b, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            &[(
                ProjectPath {
                    worktree_id,
                    path: rel_path("a.rs").into(),
                },
                LanguageServerId(0),
                DiagnosticSummary {
                    error_count: 1,
                    warning_count: 0,
                },
            )]
        )
    });

    // Join project as client C and observe the diagnostics.
    let project_c = client_c.join_remote_project(project_id, cx_c).await;
    executor.run_until_parked();
    let project_c_diagnostic_summaries =
        Rc::new(RefCell::new(project_c.read_with(cx_c, |project, cx| {
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>()
        })));
    project_c.update(cx_c, |_, cx| {
        let summaries = project_c_diagnostic_summaries.clone();
        cx.subscribe(&project_c, {
            move |p, _, event, cx| {
                if let project::Event::DiskBasedDiagnosticsFinished { .. } = event {
                    *summaries.borrow_mut() = p.diagnostic_summaries(false, cx).collect();
                }
            }
        })
        .detach();
    });

    executor.run_until_parked();
    assert_eq!(
        project_c_diagnostic_summaries.borrow().as_slice(),
        &[(
            ProjectPath {
                worktree_id,
                path: rel_path("a.rs").into(),
            },
            LanguageServerId(0),
            DiagnosticSummary {
                error_count: 1,
                warning_count: 0,
            },
        )]
    );

    // Simulate a language server reporting more errors for a file.
    fake_language_server.notify::<lsp::notification::PublishDiagnostics>(
        lsp::PublishDiagnosticsParams {
            uri: lsp::Uri::from_file_path(path!("/a/a.rs")).unwrap(),
            version: None,
            diagnostics: vec![
                lsp::Diagnostic {
                    severity: Some(lsp::DiagnosticSeverity::ERROR),
                    range: lsp::Range::new(lsp::Position::new(0, 4), lsp::Position::new(0, 7)),
                    message: "message 1".to_string(),
                    ..Default::default()
                },
                lsp::Diagnostic {
                    severity: Some(lsp::DiagnosticSeverity::WARNING),
                    range: lsp::Range::new(lsp::Position::new(0, 10), lsp::Position::new(0, 13)),
                    message: "message 2".to_string(),
                    ..Default::default()
                },
            ],
        },
    );

    // Clients B and C get the updated summaries
    executor.run_until_parked();

    project_b.read_with(cx_b, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            [(
                ProjectPath {
                    worktree_id,
                    path: rel_path("a.rs").into(),
                },
                LanguageServerId(0),
                DiagnosticSummary {
                    error_count: 1,
                    warning_count: 1,
                },
            )]
        );
    });

    project_c.read_with(cx_c, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            [(
                ProjectPath {
                    worktree_id,
                    path: rel_path("a.rs").into(),
                },
                LanguageServerId(0),
                DiagnosticSummary {
                    error_count: 1,
                    warning_count: 1,
                },
            )]
        );
    });

    // Open the file with the errors on client B. They should be present.
    let open_buffer = project_b.update(cx_b, |p, cx| {
        p.open_buffer((worktree_id, rel_path("a.rs")), cx)
    });
    let buffer_b = cx_b.executor().spawn(open_buffer).await.unwrap();

    buffer_b.read_with(cx_b, |buffer, _| {
        assert_eq!(
            buffer
                .snapshot()
                .diagnostics_in_range::<_, Point>(0..buffer.len(), false)
                .collect::<Vec<_>>(),
            &[
                DiagnosticEntry {
                    range: Point::new(0, 4)..Point::new(0, 7),
                    diagnostic: Diagnostic {
                        group_id: 2,
                        message: "message 1".to_string(),
                        severity: lsp::DiagnosticSeverity::ERROR,
                        is_primary: true,
                        source_kind: DiagnosticSourceKind::Pushed,
                        ..Diagnostic::default()
                    }
                },
                DiagnosticEntry {
                    range: Point::new(0, 10)..Point::new(0, 13),
                    diagnostic: Diagnostic {
                        group_id: 3,
                        severity: lsp::DiagnosticSeverity::WARNING,
                        message: "message 2".to_string(),
                        is_primary: true,
                        source_kind: DiagnosticSourceKind::Pushed,
                        ..Diagnostic::default()
                    }
                }
            ]
        );
    });

    // Simulate a language server reporting no errors for a file.
    fake_language_server.notify::<lsp::notification::PublishDiagnostics>(
        lsp::PublishDiagnosticsParams {
            uri: lsp::Uri::from_file_path(path!("/a/a.rs")).unwrap(),
            version: None,
            diagnostics: Vec::new(),
        },
    );
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            []
        )
    });

    project_b.read_with(cx_b, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            []
        )
    });

    project_c.read_with(cx_c, |project, cx| {
        assert_eq!(
            project.diagnostic_summaries(false, cx).collect::<Vec<_>>(),
            []
        )
    });
}

#[gpui::test(iterations = 10)]
async fn test_collaborating_with_lsp_progress_updates_and_diagnostics_ordering(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            disk_based_diagnostics_progress_token: Some("the-disk-based-token".into()),
            disk_based_diagnostics_sources: vec!["the-disk-based-diagnostics-source".into()],
            ..Default::default()
        },
    );

    let file_names = &["one.rs", "two.rs", "three.rs", "four.rs", "five.rs"];
    client_a
        .fs()
        .insert_tree(
            path!("/test"),
            json!({
                "one.rs": "const ONE: usize = 1;",
                "two.rs": "const TWO: usize = 2;",
                "three.rs": "const THREE: usize = 3;",
                "four.rs": "const FOUR: usize = 3;",
                "five.rs": "const FIVE: usize = 3;",
            }),
        )
        .await;

    let (project_a, worktree_id) = client_a.build_local_project(path!("/test"), cx_a).await;

    // Share a project as client A
    let active_call_a = cx_a.read(ActiveCall::global);
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    // Join the project as client B and open all three files.
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let guest_buffers = futures::future::try_join_all(file_names.iter().map(|file_name| {
        project_b.update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path(file_name)), cx)
        })
    }))
    .await
    .unwrap();

    // Simulate a language server reporting errors for a file.
    let fake_language_server = fake_language_servers.next().await.unwrap();
    executor.run_until_parked();
    fake_language_server
        .request::<lsp::request::WorkDoneProgressCreate>(
            lsp::WorkDoneProgressCreateParams {
                token: lsp::NumberOrString::String("the-disk-based-token".to_string()),
            },
            DEFAULT_LSP_REQUEST_TIMEOUT,
        )
        .await
        .into_response()
        .unwrap();
    fake_language_server.notify::<lsp::notification::Progress>(lsp::ProgressParams {
        token: lsp::NumberOrString::String("the-disk-based-token".to_string()),
        value: lsp::ProgressParamsValue::WorkDone(lsp::WorkDoneProgress::Begin(
            lsp::WorkDoneProgressBegin {
                title: "Progress Began".into(),
                ..Default::default()
            },
        )),
    });
    for file_name in file_names {
        fake_language_server.notify::<lsp::notification::PublishDiagnostics>(
            lsp::PublishDiagnosticsParams {
                uri: lsp::Uri::from_file_path(Path::new(path!("/test")).join(file_name)).unwrap(),
                version: None,
                diagnostics: vec![lsp::Diagnostic {
                    severity: Some(lsp::DiagnosticSeverity::WARNING),
                    source: Some("the-disk-based-diagnostics-source".into()),
                    range: lsp::Range::new(lsp::Position::new(0, 0), lsp::Position::new(0, 0)),
                    message: "message one".to_string(),
                    ..Default::default()
                }],
            },
        );
    }
    fake_language_server.notify::<lsp::notification::Progress>(lsp::ProgressParams {
        token: lsp::NumberOrString::String("the-disk-based-token".to_string()),
        value: lsp::ProgressParamsValue::WorkDone(lsp::WorkDoneProgress::End(
            lsp::WorkDoneProgressEnd { message: None },
        )),
    });

    // When the "disk base diagnostics finished" message is received, the buffers'
    // diagnostics are expected to be present.
    let disk_based_diagnostics_finished = Arc::new(AtomicBool::new(false));
    project_b.update(cx_b, {
        let project_b = project_b.clone();
        let disk_based_diagnostics_finished = disk_based_diagnostics_finished.clone();
        move |_, cx| {
            cx.subscribe(&project_b, move |_, _, event, cx| {
                if let project::Event::DiskBasedDiagnosticsFinished { .. } = event {
                    disk_based_diagnostics_finished.store(true, SeqCst);
                    for (buffer, _) in &guest_buffers {
                        assert_eq!(
                            buffer
                                .read(cx)
                                .snapshot()
                                .diagnostics_in_range::<_, usize>(0..5, false)
                                .count(),
                            1,
                            "expected a diagnostic for buffer {:?}",
                            buffer.read(cx).file().unwrap().path(),
                        );
                    }
                }
            })
            .detach();
        }
    });

    executor.run_until_parked();
    assert!(disk_based_diagnostics_finished.load(SeqCst));
}

#[gpui::test(iterations = 10)]
async fn test_reloading_buffer_manually(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(path!("/a"), json!({ "a.rs": "let one = 1;" }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/a"), cx_a).await;
    let buffer_a = project_a
        .update(cx_a, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let open_buffer = project_b.update(cx_b, |p, cx| {
        p.open_buffer((worktree_id, rel_path("a.rs")), cx)
    });
    let buffer_b = cx_b.executor().spawn(open_buffer).await.unwrap();
    buffer_b.update(cx_b, |buffer, cx| {
        buffer.edit([(4..7, "six")], None, cx);
        buffer.edit([(10..11, "6")], None, cx);
        assert_eq!(buffer.text(), "let six = 6;");
        assert!(buffer.is_dirty());
        assert!(!buffer.has_conflict());
    });
    executor.run_until_parked();

    buffer_a.read_with(cx_a, |buffer, _| assert_eq!(buffer.text(), "let six = 6;"));

    client_a
        .fs()
        .save(
            path!("/a/a.rs").as_ref(),
            &Rope::from("let seven = 7;"),
            LineEnding::Unix,
        )
        .await
        .unwrap();
    executor.run_until_parked();

    buffer_a.read_with(cx_a, |buffer, _| assert!(buffer.has_conflict()));

    buffer_b.read_with(cx_b, |buffer, _| assert!(buffer.has_conflict()));

    project_b
        .update(cx_b, |project, cx| {
            project.reload_buffers(HashSet::from_iter([buffer_b.clone()]), true, cx)
        })
        .await
        .unwrap();

    buffer_a.read_with(cx_a, |buffer, _| {
        assert_eq!(buffer.text(), "let seven = 7;");
        assert!(!buffer.is_dirty());
        assert!(!buffer.has_conflict());
    });

    buffer_b.read_with(cx_b, |buffer, _| {
        assert_eq!(buffer.text(), "let seven = 7;");
        assert!(!buffer.is_dirty());
        assert!(!buffer.has_conflict());
    });

    buffer_a.update(cx_a, |buffer, cx| {
        // Undoing on the host is a no-op when the reload was initiated by the guest.
        buffer.undo(cx);
        assert_eq!(buffer.text(), "let seven = 7;");
        assert!(!buffer.is_dirty());
        assert!(!buffer.has_conflict());
    });
    buffer_b.update(cx_b, |buffer, cx| {
        // Undoing on the guest rolls back the buffer to before it was reloaded but the conflict gets cleared.
        buffer.undo(cx);
        assert_eq!(buffer.text(), "let six = 6;");
        assert!(buffer.is_dirty());
        assert!(!buffer.has_conflict());
    });
}

#[gpui::test(iterations = 10)]
async fn test_formatting_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a
        .language_registry()
        .register_fake_lsp("Rust", FakeLspAdapter::default());

    // Here we insert a fake tree with a directory that exists on disk. This is needed
    // because later we'll invoke a command, which requires passing a working directory
    // that points to a valid location on disk.
    let directory = env::current_dir().unwrap();
    client_a
        .fs()
        .insert_tree(&directory, json!({ "a.rs": "let one = \"two\"" }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(&directory, cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();

    let _handle = project_b.update(cx_b, |project, cx| {
        project.register_buffer_with_language_servers(&buffer_b, cx)
    });
    let fake_language_server = fake_language_servers.next().await.unwrap();
    executor.run_until_parked();
    fake_language_server.set_request_handler::<lsp::request::Formatting, _, _>(|_, _| async move {
        Ok(Some(vec![
            lsp::TextEdit {
                range: lsp::Range::new(lsp::Position::new(0, 4), lsp::Position::new(0, 4)),
                new_text: "h".to_string(),
            },
            lsp::TextEdit {
                range: lsp::Range::new(lsp::Position::new(0, 7), lsp::Position::new(0, 7)),
                new_text: "y".to_string(),
            },
        ]))
    });

    project_b
        .update(cx_b, |project, cx| {
            project.format(
                HashSet::from_iter([buffer_b.clone()]),
                LspFormatTarget::Buffers,
                true,
                FormatTrigger::Save,
                cx,
            )
        })
        .await
        .unwrap();

    // The edits from the LSP are applied, and a final newline is added.
    assert_eq!(
        buffer_b.read_with(cx_b, |buffer, _| buffer.text()),
        "let honey = \"two\"\n"
    );

    // There is no `awk` command on Windows.
    #[cfg(not(target_os = "windows"))]
    {
        // Ensure buffer can be formatted using an external command. Notice how the
        // host's configuration is honored as opposed to using the guest's settings.
        cx_a.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |file| {
                    file.project.all_languages.defaults.formatter =
                        Some(FormatterList::Single(Formatter::External {
                            command: "awk".into(),
                            arguments: Some(vec!["{sub(/two/,\"{buffer_path}\")}1".to_string()]),
                        }));
                });
            });
        });

        executor.allow_parking();
        project_b
            .update(cx_b, |project, cx| {
                project.format(
                    HashSet::from_iter([buffer_b.clone()]),
                    LspFormatTarget::Buffers,
                    true,
                    FormatTrigger::Save,
                    cx,
                )
            })
            .await
            .unwrap();
        assert_eq!(
            buffer_b.read_with(cx_b, |buffer, _| buffer.text()),
            format!("let honey = \"{}/a.rs\"\n", directory.to_str().unwrap())
        );
    }
}

#[gpui::test(iterations = 10)]
async fn test_range_formatting_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: lsp::ServerCapabilities {
                document_range_formatting_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let directory = env::current_dir().unwrap();
    client_a
        .fs()
        .insert_tree(&directory, json!({ "a.rs": "one\ntwo\nthree\n" }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(&directory, cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let buffer_b = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();

    let _handle = project_b.update(cx_b, |project, cx| {
        project.register_buffer_with_language_servers(&buffer_b, cx)
    });
    let fake_language_server = fake_language_servers.next().await.unwrap();
    executor.run_until_parked();

    fake_language_server.set_request_handler::<lsp::request::RangeFormatting, _, _>(
        |params, _| async move {
            assert_eq!(params.range.start, lsp::Position::new(0, 0));
            assert_eq!(params.range.end, lsp::Position::new(1, 3));
            Ok(Some(vec![lsp::TextEdit::new(
                lsp::Range::new(lsp::Position::new(0, 3), lsp::Position::new(1, 0)),
                ", ".to_string(),
            )]))
        },
    );

    let buffer_id = buffer_b.read_with(cx_b, |buffer, _| buffer.remote_id());
    let ranges = buffer_b.read_with(cx_b, |buffer, _| {
        let start = buffer.anchor_before(0);
        let end = buffer.anchor_after(7);
        vec![start..end]
    });

    let mut ranges_map = BTreeMap::new();
    ranges_map.insert(buffer_id, ranges);

    project_b
        .update(cx_b, |project, cx| {
            project.format(
                HashSet::from_iter([buffer_b.clone()]),
                LspFormatTarget::Ranges(ranges_map),
                true,
                FormatTrigger::Manual,
                cx,
            )
        })
        .await
        .unwrap();

    assert_eq!(
        buffer_b.read_with(cx_b, |buffer, _| buffer.text()),
        "one, two\nthree\n"
    );
}

#[gpui::test(iterations = 10)]
async fn test_prettier_formatting_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let test_plugin = "test_plugin";

    client_a.language_registry().add(Arc::new(Language::new(
        LanguageConfig {
            name: "TypeScript".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["ts".to_string()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
    )));
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "TypeScript",
        FakeLspAdapter {
            prettier_plugins: vec![test_plugin],
            ..Default::default()
        },
    );

    // Here we insert a fake tree with a directory that exists on disk. This is needed
    // because later we'll invoke a command, which requires passing a working directory
    // that points to a valid location on disk.
    let directory = env::current_dir().unwrap();
    let buffer_text = "let one = \"two\"";
    client_a
        .fs()
        .insert_tree(&directory, json!({ "a.ts": buffer_text }))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(&directory, cx_a).await;
    let prettier_format_suffix = project::TEST_PRETTIER_FORMAT_SUFFIX;
    let open_buffer = project_a.update(cx_a, |p, cx| {
        p.open_buffer((worktree_id, rel_path("a.ts")), cx)
    });
    let buffer_a = cx_a.executor().spawn(open_buffer).await.unwrap();

    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let (buffer_b, _) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("a.ts")), cx)
        })
        .await
        .unwrap();

    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |file| {
                file.project.all_languages.defaults.formatter = Some(FormatterList::default());
                file.project.all_languages.defaults.prettier = Some(PrettierSettingsContent {
                    allowed: Some(true),
                    ..Default::default()
                });
            });
        });
    });
    cx_b.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |file| {
                file.project.all_languages.defaults.formatter = Some(FormatterList::Single(
                    Formatter::LanguageServer(LanguageServerFormatterSpecifier::Current),
                ));
                file.project.all_languages.defaults.prettier = Some(PrettierSettingsContent {
                    allowed: Some(true),
                    ..Default::default()
                });
            });
        });
    });
    let fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server.set_request_handler::<lsp::request::Formatting, _, _>(|_, _| async move {
        panic!(
            "Unexpected: prettier should be preferred since it's enabled and language supports it"
        )
    });

    project_b
        .update(cx_b, |project, cx| {
            project.format(
                HashSet::from_iter([buffer_b.clone()]),
                LspFormatTarget::Buffers,
                true,
                FormatTrigger::Save,
                cx,
            )
        })
        .await
        .unwrap();

    executor.run_until_parked();
    assert_eq!(
        buffer_b.read_with(cx_b, |buffer, _| buffer.text()),
        buffer_text.to_string() + "\n" + prettier_format_suffix,
        "Prettier formatting was not applied to client buffer after client's request"
    );

    project_a
        .update(cx_a, |project, cx| {
            project.format(
                HashSet::from_iter([buffer_a.clone()]),
                LspFormatTarget::Buffers,
                true,
                FormatTrigger::Manual,
                cx,
            )
        })
        .await
        .unwrap();

    executor.run_until_parked();
    assert_eq!(
        buffer_b.read_with(cx_b, |buffer, _| buffer.text()),
        buffer_text.to_string() + "\n" + prettier_format_suffix + "\n" + prettier_format_suffix,
        "Prettier formatting was not applied to client buffer after host's request"
    );
}

#[gpui::test]
async fn test_call_hierarchy_routes_through_the_authoritative_host(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let capabilities = lsp::ServerCapabilities {
        call_hierarchy_provider: Some(lsp::CallHierarchyServerCapability::Simple(true)),
        ..lsp::ServerCapabilities::default()
    };
    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/root"),
            json!({
                "root.rs": "fn root() {}\n",
                "caller.rs": "fn caller() { root(); }\n",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/root"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let (buffer_b, _handle) = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("root.rs")), cx)
        })
        .await
        .unwrap();
    let fake_language_server = fake_language_servers.next().await.unwrap();
    executor.run_until_parked();

    fake_language_server.set_request_handler::<lsp::request::CallHierarchyPrepare, _, _>(
        |_, _| async move {
            Ok(Some(vec![lsp::CallHierarchyItem {
                name: "root".to_string(),
                kind: lsp::SymbolKind::FUNCTION,
                tags: None,
                detail: Some("fn root()".to_string()),
                uri: lsp::Uri::from_file_path(path!("/root/root.rs")).unwrap(),
                range: lsp::Range::new(lsp::Position::new(0, 0), lsp::Position::new(0, 12)),
                selection_range: lsp::Range::new(
                    lsp::Position::new(0, 3),
                    lsp::Position::new(0, 7),
                ),
                data: Some(json!({"authoritative": true})),
            }]))
        },
    );
    let prepared = project_b
        .update(cx_b, |project, cx| {
            project.prepare_call_hierarchy(&buffer_b, 4, cx)
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(prepared.items.len(), 1);
    assert_eq!(prepared.items[0].name.as_ref(), "root");
    assert_eq!(prepared.items[0].data, Some(json!({"authoritative": true})));

    fake_language_server.set_request_handler::<lsp::request::CallHierarchyIncomingCalls, _, _>(
        |params, _| async move {
            assert_eq!(params.item.data, Some(json!({"authoritative": true})));
            Ok(Some(vec![lsp::CallHierarchyIncomingCall {
                from: lsp::CallHierarchyItem {
                    name: "caller".to_string(),
                    kind: lsp::SymbolKind::FUNCTION,
                    tags: None,
                    detail: Some("fn caller()".to_string()),
                    uri: lsp::Uri::from_file_path(path!("/root/caller.rs")).unwrap(),
                    range: lsp::Range::new(lsp::Position::new(0, 0), lsp::Position::new(0, 23)),
                    selection_range: lsp::Range::new(
                        lsp::Position::new(0, 3),
                        lsp::Position::new(0, 9),
                    ),
                    data: None,
                },
                from_ranges: vec![lsp::Range::new(
                    lsp::Position::new(0, 14),
                    lsp::Position::new(0, 18),
                )],
            }]))
        },
    );
    let incoming = project_b
        .update(cx_b, |project, cx| {
            project.incoming_calls(prepared.items[0].clone(), cx)
        })
        .await
        .unwrap();
    assert_eq!(incoming.calls.len(), 1);
    assert_eq!(incoming.calls[0].item.name.as_ref(), "caller");
    assert_eq!(
        incoming.calls[0]
            .item
            .buffer
            .read_with(cx_b, |buffer, _| buffer.text()),
        "fn caller() { root(); }\n"
    );
}

#[gpui::test(iterations = 10)]
async fn test_definition(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let capabilities = lsp::ServerCapabilities {
        definition_provider: Some(OneOf::Left(true)),
        type_definition_provider: Some(lsp::TypeDefinitionProviderCapability::Simple(true)),
        ..lsp::ServerCapabilities::default()
    };
    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/root"),
            json!({
                "dir-1": {
                    "a.rs": "const ONE: usize = b::TWO + b::THREE;",
                },
                "dir-2": {
                    "b.rs": "const TWO: c::T2 = 2;\nconst THREE: usize = 3;",
                    "c.rs": "type T2 = usize;",
                }
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a
        .build_local_project(path!("/root/dir-1"), cx_a)
        .await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open the file on client B.
    let (buffer_b, _handle) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();

    // Request the definition of a symbol as the guest.
    let fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server.set_request_handler::<lsp::request::GotoDefinition, _, _>(
        |_, _| async move {
            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                lsp::Location::new(
                    lsp::Uri::from_file_path(path!("/root/dir-2/b.rs")).unwrap(),
                    lsp::Range::new(lsp::Position::new(0, 6), lsp::Position::new(0, 9)),
                ),
            )))
        },
    );
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    let definitions_1 = project_b
        .update(cx_b, |p, cx| p.definitions(&buffer_b, 23, cx))
        .await
        .unwrap()
        .unwrap();
    cx_b.read(|cx| {
        assert_eq!(
            definitions_1.len(),
            1,
            "Unexpected definitions: {definitions_1:?}"
        );
        assert_eq!(project_b.read(cx).worktrees(cx).count(), 2);
        let target_buffer = definitions_1[0].target.buffer.read(cx);
        assert_eq!(
            target_buffer.text(),
            "const TWO: c::T2 = 2;\nconst THREE: usize = 3;"
        );
        assert_eq!(
            definitions_1[0].target.range.to_point(target_buffer),
            Point::new(0, 6)..Point::new(0, 9)
        );
    });

    // Try getting more definitions for the same buffer, ensuring the buffer gets reused from
    // the previous call to `definition`.
    fake_language_server.set_request_handler::<lsp::request::GotoDefinition, _, _>(
        |_, _| async move {
            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                lsp::Location::new(
                    lsp::Uri::from_file_path(path!("/root/dir-2/b.rs")).unwrap(),
                    lsp::Range::new(lsp::Position::new(1, 6), lsp::Position::new(1, 11)),
                ),
            )))
        },
    );

    let definitions_2 = project_b
        .update(cx_b, |p, cx| p.definitions(&buffer_b, 33, cx))
        .await
        .unwrap()
        .unwrap();
    cx_b.read(|cx| {
        assert_eq!(definitions_2.len(), 1);
        assert_eq!(project_b.read(cx).worktrees(cx).count(), 2);
        let target_buffer = definitions_2[0].target.buffer.read(cx);
        assert_eq!(
            target_buffer.text(),
            "const TWO: c::T2 = 2;\nconst THREE: usize = 3;"
        );
        assert_eq!(
            definitions_2[0].target.range.to_point(target_buffer),
            Point::new(1, 6)..Point::new(1, 11)
        );
    });
    assert_eq!(
        definitions_1[0].target.buffer,
        definitions_2[0].target.buffer
    );

    fake_language_server.set_request_handler::<lsp::request::GotoTypeDefinition, _, _>(
        |req, _| async move {
            assert_eq!(
                req.text_document_position_params.position,
                lsp::Position::new(0, 7)
            );
            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                lsp::Location::new(
                    lsp::Uri::from_file_path(path!("/root/dir-2/c.rs")).unwrap(),
                    lsp::Range::new(lsp::Position::new(0, 5), lsp::Position::new(0, 7)),
                ),
            )))
        },
    );

    let type_definitions = project_b
        .update(cx_b, |p, cx| p.type_definitions(&buffer_b, 7, cx))
        .await
        .unwrap()
        .unwrap();
    cx_b.read(|cx| {
        assert_eq!(
            type_definitions.len(),
            1,
            "Unexpected type definitions: {type_definitions:?}"
        );
        let target_buffer = type_definitions[0].target.buffer.read(cx);
        assert_eq!(target_buffer.text(), "type T2 = usize;");
        assert_eq!(
            type_definitions[0].target.range.to_point(target_buffer),
            Point::new(0, 5)..Point::new(0, 7)
        );
    });
}

#[gpui::test]
async fn test_edit_prediction_definition(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let capabilities = lsp::ServerCapabilities {
        definition_provider: Some(OneOf::Left(true)),
        ..lsp::ServerCapabilities::default()
    };
    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/root"),
            json!({
                "a.rs": "const ONE: usize = TWO;",
                "b.rs": "const TWO: usize = 2;",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/root"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let (buffer_b, _handle) = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();

    let fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server.set_request_handler::<lsp::request::GotoDefinition, _, _>(
        |_, _| async move {
            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                lsp::Location::new(
                    lsp::Uri::from_file_path(path!("/root/b.rs")).unwrap(),
                    lsp::Range::new(lsp::Position::new(0, 6), lsp::Position::new(0, 9)),
                ),
            )))
        },
    );
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    let definitions = project_b
        .update(cx_b, |project, cx| {
            project.edit_prediction_definitions(&buffer_b, 19, false, cx)
        })
        .await
        .unwrap();

    cx_b.read(|cx| {
        assert_eq!(definitions.len(), 1);
        assert_eq!(
            definitions[0].path,
            ProjectPath {
                worktree_id,
                path: rel_path("b.rs").into(),
            }
        );
        assert_eq!(
            definitions[0].range.start.0,
            language::PointUtf16::new(0, 6)
        );
        assert_eq!(definitions[0].range.end.0, language::PointUtf16::new(0, 9));
        assert!(
            project_b
                .read(cx)
                .get_open_buffer(&definitions[0].path, cx)
                .is_none()
        );
    });
}

#[gpui::test(iterations = 10)]
async fn test_references(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let capabilities = lsp::ServerCapabilities {
        references_provider: Some(lsp::OneOf::Left(true)),
        ..lsp::ServerCapabilities::default()
    };
    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "my-fake-lsp-adapter",
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            name: "my-fake-lsp-adapter",
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/root"),
            json!({
                "dir-1": {
                    "one.rs": "const ONE: usize = 1;",
                    "two.rs": "const TWO: usize = one::ONE + one::ONE;",
                },
                "dir-2": {
                    "three.rs": "const THREE: usize = two::TWO + one::ONE;",
                }
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a
        .build_local_project(path!("/root/dir-1"), cx_a)
        .await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open the file on client B.
    let (buffer_b, _handle) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("one.rs")), cx)
        })
        .await
        .unwrap();

    // Request references to a symbol as the guest.
    let fake_language_server = fake_language_servers.next().await.unwrap();
    let (lsp_response_tx, rx) = mpsc::unbounded::<Result<Option<Vec<lsp::Location>>>>();
    fake_language_server.set_request_handler::<lsp::request::References, _, _>({
        let rx = Arc::new(Mutex::new(Some(rx)));
        move |params, _| {
            assert_eq!(
                params.text_document_position.text_document.uri.as_str(),
                uri!("file:///root/dir-1/one.rs")
            );
            let rx = rx.clone();
            async move {
                let mut response_rx = rx.lock().take().unwrap();
                let result = response_rx.next().await.unwrap();
                *rx.lock() = Some(response_rx);
                result
            }
        }
    });
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    let references = project_b.update(cx_b, |p, cx| p.references(&buffer_b, 7, cx));

    // User is informed that a request is pending.
    executor.run_until_parked();
    project_b.read_with(cx_b, |project, cx| {
        let status = project.language_server_statuses(cx).next().unwrap().1;
        assert_eq!(status.name.0, "my-fake-lsp-adapter");
        assert_eq!(
            status.pending_work.values().next().unwrap().message,
            Some("Finding references...".into())
        );
    });

    // Cause the language server to respond.
    lsp_response_tx
        .unbounded_send(Ok(Some(vec![
            lsp::Location {
                uri: lsp::Uri::from_file_path(path!("/root/dir-1/two.rs")).unwrap(),
                range: lsp::Range::new(lsp::Position::new(0, 24), lsp::Position::new(0, 27)),
            },
            lsp::Location {
                uri: lsp::Uri::from_file_path(path!("/root/dir-1/two.rs")).unwrap(),
                range: lsp::Range::new(lsp::Position::new(0, 35), lsp::Position::new(0, 38)),
            },
            lsp::Location {
                uri: lsp::Uri::from_file_path(path!("/root/dir-2/three.rs")).unwrap(),
                range: lsp::Range::new(lsp::Position::new(0, 37), lsp::Position::new(0, 40)),
            },
        ])))
        .unwrap();

    let references = references.await.unwrap().unwrap();
    executor.run_until_parked();
    project_b.read_with(cx_b, |project, cx| {
        // User is informed that a request is no longer pending.
        let status = project.language_server_statuses(cx).next().unwrap().1;
        assert!(status.pending_work.is_empty());

        assert_eq!(references.len(), 3);
        assert_eq!(project.worktrees(cx).count(), 2);

        let two_buffer = references[0].buffer.read(cx);
        let three_buffer = references[2].buffer.read(cx);
        assert_eq!(
            two_buffer.file().unwrap().path().as_ref(),
            rel_path("two.rs")
        );
        assert_eq!(references[1].buffer, references[0].buffer);
        assert_eq!(
            three_buffer.file().unwrap().full_path(cx),
            Path::new(path!("/root/dir-2/three.rs"))
        );

        assert_eq!(references[0].range.to_offset(two_buffer), 24..27);
        assert_eq!(references[1].range.to_offset(two_buffer), 35..38);
        assert_eq!(references[2].range.to_offset(three_buffer), 37..40);
    });

    let references = project_b.update(cx_b, |p, cx| p.references(&buffer_b, 7, cx));

    // User is informed that a request is pending.
    executor.run_until_parked();
    project_b.read_with(cx_b, |project, cx| {
        let status = project.language_server_statuses(cx).next().unwrap().1;
        assert_eq!(status.name.0, "my-fake-lsp-adapter");
        assert_eq!(
            status.pending_work.values().next().unwrap().message,
            Some("Finding references...".into())
        );
    });

    // Cause the LSP request to fail.
    lsp_response_tx
        .unbounded_send(Err(anyhow!("can't find references")))
        .unwrap();
    assert_eq!(references.await.unwrap().unwrap(), []);

    // User is informed that the request is no longer pending.
    executor.run_until_parked();
    project_b.read_with(cx_b, |project, cx| {
        let status = project.language_server_statuses(cx).next().unwrap().1;
        assert!(status.pending_work.is_empty());
    });
}

#[gpui::test(iterations = 10)]
async fn test_project_search(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            "/root",
            json!({
                "dir-1": {
                    "a": "hello world",
                    "b": "goodnight moon",
                    "c": "a world of goo",
                    "d": "world champion of clown world",
                },
                "dir-2": {
                    "e": "disney world is fun",
                }
            }),
        )
        .await;
    let (project_a, _) = client_a.build_local_project("/root/dir-1", cx_a).await;
    let (worktree_2, _) = project_a
        .update(cx_a, |p, cx| {
            p.find_or_create_worktree("/root/dir-2", true, cx)
        })
        .await
        .unwrap();
    worktree_2
        .read_with(cx_a, |tree, _| tree.as_local().unwrap().scan_complete())
        .await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Perform a search as the guest.
    let mut results = HashMap::default();
    let search_rx = project_b.update(cx_b, |project, cx| {
        project.search(
            SearchQuery::text(
                "world",
                false,
                false,
                false,
                Default::default(),
                Default::default(),
                false,
                None,
            )
            .unwrap(),
            cx,
        )
    });
    while let Ok(result) = search_rx.rx.recv().await {
        match result {
            SearchResult::Buffer { buffer, ranges } => {
                results.entry(buffer).or_insert(ranges);
            }
            SearchResult::LimitReached => {
                panic!(
                    "Unexpectedly reached search limit in tests. If you do want to assert limit-reached, change this panic call."
                )
            }
            SearchResult::WaitingForScan | SearchResult::Searching => {}
        };
    }

    let mut ranges_by_path = results
        .into_iter()
        .map(|(buffer, ranges)| {
            buffer.read_with(cx_b, |buffer, cx| {
                let path = buffer.file().unwrap().full_path(cx);
                let offset_ranges = ranges
                    .into_iter()
                    .map(|range| range.to_offset(buffer))
                    .collect::<Vec<_>>();
                (path, offset_ranges)
            })
        })
        .collect::<Vec<_>>();
    ranges_by_path.sort_by_key(|(path, _)| path.clone());

    assert_eq!(
        ranges_by_path,
        &[
            (PathBuf::from("dir-1/a"), vec![6..11]),
            (PathBuf::from("dir-1/c"), vec![2..7]),
            (PathBuf::from("dir-1/d"), vec![0..5, 24..29]),
            (PathBuf::from("dir-2/e"), vec![7..12]),
        ]
    );
}

#[gpui::test(iterations = 10)]
async fn test_document_highlights(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/root-1"),
            json!({
                "main.rs": "fn double(number: i32) -> i32 { number + number }",
            }),
        )
        .await;

    client_a.language_registry().add(rust_lang());
    let capabilities = lsp::ServerCapabilities {
        document_highlight_provider: Some(lsp::OneOf::Left(true)),
        ..lsp::ServerCapabilities::default()
    };
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    let (project_a, worktree_id) = client_a.build_local_project(path!("/root-1"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open the file on client B.
    let (buffer_b, _handle) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("main.rs")), cx)
        })
        .await
        .unwrap();

    // Request document highlights as the guest.
    let fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server.set_request_handler::<lsp::request::DocumentHighlightRequest, _, _>(
        |params, _| async move {
            assert_eq!(
                params
                    .text_document_position_params
                    .text_document
                    .uri
                    .as_str(),
                uri!("file:///root-1/main.rs")
            );
            assert_eq!(
                params.text_document_position_params.position,
                lsp::Position::new(0, 34)
            );
            Ok(Some(vec![
                lsp::DocumentHighlight {
                    kind: Some(lsp::DocumentHighlightKind::WRITE),
                    range: lsp::Range::new(lsp::Position::new(0, 10), lsp::Position::new(0, 16)),
                },
                lsp::DocumentHighlight {
                    kind: Some(lsp::DocumentHighlightKind::READ),
                    range: lsp::Range::new(lsp::Position::new(0, 32), lsp::Position::new(0, 38)),
                },
                lsp::DocumentHighlight {
                    kind: Some(lsp::DocumentHighlightKind::READ),
                    range: lsp::Range::new(lsp::Position::new(0, 41), lsp::Position::new(0, 47)),
                },
            ]))
        },
    );
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    let highlights = project_b
        .update(cx_b, |p, cx| p.document_highlights(&buffer_b, 34, cx))
        .await
        .unwrap();

    buffer_b.read_with(cx_b, |buffer, _| {
        let snapshot = buffer.snapshot();

        let highlights = highlights
            .into_iter()
            .map(|highlight| (highlight.kind, highlight.range.to_offset(&snapshot)))
            .collect::<Vec<_>>();
        assert_eq!(
            highlights,
            &[
                (lsp::DocumentHighlightKind::WRITE, 10..16),
                (lsp::DocumentHighlightKind::READ, 32..38),
                (lsp::DocumentHighlightKind::READ, 41..47)
            ]
        )
    });
}

#[gpui::test(iterations = 10)]
async fn test_lsp_hover(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree(
            path!("/root-1"),
            json!({
                "main.rs": "use std::collections::HashMap;",
            }),
        )
        .await;

    client_a.language_registry().add(rust_lang());
    let language_server_names = ["rust-analyzer", "CrabLang-ls"];
    let capabilities_1 = lsp::ServerCapabilities {
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
        ..lsp::ServerCapabilities::default()
    };
    let capabilities_2 = lsp::ServerCapabilities {
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
        ..lsp::ServerCapabilities::default()
    };
    let mut language_servers = [
        client_a.language_registry().register_fake_lsp(
            "Rust",
            FakeLspAdapter {
                name: language_server_names[0],
                capabilities: capabilities_1.clone(),
                ..FakeLspAdapter::default()
            },
        ),
        client_a.language_registry().register_fake_lsp(
            "Rust",
            FakeLspAdapter {
                name: language_server_names[1],
                capabilities: capabilities_2.clone(),
                ..FakeLspAdapter::default()
            },
        ),
    ];
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            name: language_server_names[0],
            capabilities: capabilities_1,
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            name: language_server_names[1],
            capabilities: capabilities_2,
            ..FakeLspAdapter::default()
        },
    );

    let (project_a, worktree_id) = client_a.build_local_project(path!("/root-1"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Open the file as the guest
    let (buffer_b, _handle) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("main.rs")), cx)
        })
        .await
        .unwrap();

    let mut servers_with_hover_requests = HashMap::default();
    for i in 0..language_server_names.len() {
        let new_server = language_servers[i].next().await.unwrap_or_else(|| {
            panic!(
                "Failed to get language server #{i} with name {}",
                language_server_names[i]
            )
        });
        let new_server_name = new_server.server.name();
        assert!(
            !servers_with_hover_requests.contains_key(&new_server_name),
            "Unexpected: initialized server with the same name twice. Name: `{new_server_name}`"
        );
        match new_server_name.as_ref() {
            "CrabLang-ls" => {
                servers_with_hover_requests.insert(
                    new_server_name.clone(),
                    new_server.set_request_handler::<lsp::request::HoverRequest, _, _>(
                        move |params, _| {
                            assert_eq!(
                                params
                                    .text_document_position_params
                                    .text_document
                                    .uri
                                    .as_str(),
                                uri!("file:///root-1/main.rs")
                            );
                            let name = new_server_name.clone();
                            async move {
                                Ok(Some(lsp::Hover {
                                    contents: lsp::HoverContents::Scalar(
                                        lsp::MarkedString::String(format!("{name} hover")),
                                    ),
                                    range: None,
                                }))
                            }
                        },
                    ),
                );
            }
            "rust-analyzer" => {
                servers_with_hover_requests.insert(
                    new_server_name.clone(),
                    new_server.set_request_handler::<lsp::request::HoverRequest, _, _>(
                        |params, _| async move {
                            assert_eq!(
                                params
                                    .text_document_position_params
                                    .text_document
                                    .uri
                                    .as_str(),
                                uri!("file:///root-1/main.rs")
                            );
                            assert_eq!(
                                params.text_document_position_params.position,
                                lsp::Position::new(0, 22)
                            );
                            Ok(Some(lsp::Hover {
                                contents: lsp::HoverContents::Array(vec![
                                    lsp::MarkedString::String("Test hover content.".to_string()),
                                    lsp::MarkedString::LanguageString(lsp::LanguageString {
                                        language: "Rust".to_string(),
                                        value: "let foo = 42;".to_string(),
                                    }),
                                ]),
                                range: Some(lsp::Range::new(
                                    lsp::Position::new(0, 22),
                                    lsp::Position::new(0, 29),
                                )),
                            }))
                        },
                    ),
                );
            }
            unexpected => panic!("Unexpected server name: {unexpected}"),
        }
    }
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    // Request hover information as the guest.
    let mut hovers = project_b
        .update(cx_b, |p, cx| p.hover(&buffer_b, 22, cx))
        .await
        .unwrap();
    assert_eq!(
        hovers.len(),
        2,
        "Expected two hovers from both language servers, but got: {hovers:?}"
    );

    let _: Vec<()> = futures::future::join_all(servers_with_hover_requests.into_values().map(
        |mut hover_request| async move {
            hover_request
                .next()
                .await
                .expect("All hover requests should have been triggered")
        },
    ))
    .await;

    hovers.sort_by_key(|hover| hover.contents.len());
    let first_hover = hovers.first().cloned().unwrap();
    assert_eq!(
        first_hover.contents,
        vec![project::HoverBlock {
            text: "CrabLang-ls hover".to_string(),
            kind: HoverBlockKind::Markdown,
        },]
    );
    let second_hover = hovers.last().cloned().unwrap();
    assert_eq!(
        second_hover.contents,
        vec![
            project::HoverBlock {
                text: "Test hover content.".to_string(),
                kind: HoverBlockKind::Markdown,
            },
            project::HoverBlock {
                text: "let foo = 42;".to_string(),
                kind: HoverBlockKind::Code {
                    language: "Rust".to_string()
                },
            }
        ]
    );
    buffer_b.read_with(cx_b, |buffer, _| {
        let snapshot = buffer.snapshot();
        assert_eq!(second_hover.range.unwrap().to_offset(&snapshot), 22..29);
    });
}

#[gpui::test(iterations = 10)]
async fn test_project_symbols(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: lsp::ServerCapabilities {
                workspace_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/code"),
            json!({
                "crate-1": {
                    "one.rs": "const ONE: usize = 1;",
                },
                "crate-2": {
                    "two.rs": "const TWO: usize = 2; const THREE: usize = 3;",
                },
                "private": {
                    "passwords.txt": "the-password",
                }
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a
        .build_local_project(path!("/code/crate-1"), cx_a)
        .await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Cause the language server to start.
    let _buffer = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("one.rs")), cx)
        })
        .await
        .unwrap();

    let fake_language_server = fake_language_servers.next().await.unwrap();
    executor.run_until_parked();
    fake_language_server.set_request_handler::<lsp::WorkspaceSymbolRequest, _, _>(
        |_, _| async move {
            Ok(Some(lsp::WorkspaceSymbolResponse::Flat(vec![
                #[allow(deprecated)]
                lsp::SymbolInformation {
                    name: "TWO".into(),
                    location: lsp::Location {
                        uri: lsp::Uri::from_file_path(path!("/code/crate-2/two.rs")).unwrap(),
                        range: lsp::Range::new(lsp::Position::new(0, 6), lsp::Position::new(0, 9)),
                    },
                    kind: lsp::SymbolKind::CONSTANT,
                    tags: None,
                    container_name: None,
                    deprecated: None,
                },
            ])))
        },
    );

    // Request the definition of a symbol as the guest.
    let symbols = project_b
        .update(cx_b, |p, cx| p.symbols("two", cx))
        .await
        .unwrap();
    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].name, "TWO");
    assert_eq!(symbols[0].kind, language::SymbolKind::Constant);

    // Open one of the returned symbols.
    let buffer_b_2 = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer_for_symbol(&symbols[0], cx)
        })
        .await
        .unwrap();

    buffer_b_2.read_with(cx_b, |buffer, cx| {
        assert_eq!(
            buffer.file().unwrap().full_path(cx),
            Path::new(path!("/code/crate-2/two.rs"))
        );
    });

    // Attempt to craft a symbol and violate host's privacy by opening an arbitrary file.
    let mut fake_symbol = symbols[0].clone();
    fake_symbol.path = SymbolLocation::OutsideProject {
        abs_path: Path::new(path!("/code/secrets")).into(),
        signature: [0x17; 32],
    };
    let error = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer_for_symbol(&fake_symbol, cx)
        })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("invalid symbol signature"));
}

#[gpui::test(iterations = 10)]
async fn test_open_buffer_while_getting_definition_pointing_to_it(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    mut rng: StdRng,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let capabilities = lsp::ServerCapabilities {
        definition_provider: Some(OneOf::Left(true)),
        ..lsp::ServerCapabilities::default()
    };
    client_a.language_registry().add(rust_lang());
    let mut fake_language_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        },
    );
    client_b.language_registry().add(rust_lang());
    client_b.language_registry().register_fake_lsp_adapter(
        "Rust",
        FakeLspAdapter {
            capabilities,
            ..FakeLspAdapter::default()
        },
    );

    client_a
        .fs()
        .insert_tree(
            path!("/root"),
            json!({
                "a.rs": "const ONE: usize = b::TWO;",
                "b.rs": "const TWO: usize = 2",
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project(path!("/root"), cx_a).await;
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let (buffer_b1, _lsp) = project_b
        .update(cx_b, |p, cx| {
            p.open_buffer_with_lsp((worktree_id, rel_path("a.rs")), cx)
        })
        .await
        .unwrap();

    let fake_language_server = fake_language_servers.next().await.unwrap();
    fake_language_server.set_request_handler::<lsp::request::GotoDefinition, _, _>(
        |_, _| async move {
            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                lsp::Location::new(
                    lsp::Uri::from_file_path(path!("/root/b.rs")).unwrap(),
                    lsp::Range::new(lsp::Position::new(0, 6), lsp::Position::new(0, 9)),
                ),
            )))
        },
    );

    let definitions;
    let buffer_b2;
    if rng.random() {
        cx_a.run_until_parked();
        cx_b.run_until_parked();
        definitions = project_b.update(cx_b, |p, cx| p.definitions(&buffer_b1, 23, cx));
        (buffer_b2, _) = project_b
            .update(cx_b, |p, cx| {
                p.open_buffer_with_lsp((worktree_id, rel_path("b.rs")), cx)
            })
            .await
            .unwrap();
    } else {
        (buffer_b2, _) = project_b
            .update(cx_b, |p, cx| {
                p.open_buffer_with_lsp((worktree_id, rel_path("b.rs")), cx)
            })
            .await
            .unwrap();
        cx_a.run_until_parked();
        cx_b.run_until_parked();
        definitions = project_b.update(cx_b, |p, cx| p.definitions(&buffer_b1, 23, cx));
    }

    let definitions = definitions.await.unwrap().unwrap();
    assert_eq!(
        definitions.len(),
        1,
        "Unexpected definitions: {definitions:?}"
    );
    assert_eq!(definitions[0].target.buffer, buffer_b2);
}

#[gpui::test(iterations = 10)]
async fn test_contacts(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    cx_d: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    let client_d = server.create_client(cx_d, "user_d").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);
    let active_call_c = cx_c.read(ActiveCall::global);
    let _active_call_d = cx_d.read(ActiveCall::global);

    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(contacts(&client_d, cx_d), []);

    server.disconnect_client(client_c.peer_id().unwrap());
    server.forbid_connections();
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "offline", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_c".to_string(), "offline", "free")
        ]
    );
    assert_eq!(contacts(&client_c, cx_c), []);
    assert_eq!(contacts(&client_d, cx_d), []);

    server.allow_connections();
    client_c
        .connect(false, &cx_c.to_async())
        .await
        .into_response()
        .unwrap();

    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(contacts(&client_d, cx_d), []);

    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "busy")
        ]
    );
    assert_eq!(contacts(&client_d, cx_d), []);

    // Client B and client D become contacts while client B is being called.
    server
        .make_contacts(&mut [(&client_b, cx_b), (&client_d, cx_d)])
        .await;
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free"),
            ("user_d".to_string(), "online", "free"),
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "busy")]
    );

    active_call_b.update(cx_b, |call, cx| call.decline_incoming(cx).unwrap());
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "free")]
    );

    active_call_c
        .update(cx_c, |call, cx| {
            call.invite(client_a.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "busy"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "free")]
    );

    active_call_a
        .update(cx_a, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "busy"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "free")]
    );

    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "busy"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "busy")]
    );

    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_c".to_string(), "online", "free"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "free"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "free")]
    );

    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_a, cx_a),
        [
            ("user_b".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_c".to_string(), "online", "free"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "online", "busy"),
            ("user_b".to_string(), "online", "busy")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "busy")]
    );

    server.forbid_connections();
    server.disconnect_client(client_a.peer_id().unwrap());
    executor.advance_clock(RECEIVE_TIMEOUT + RECONNECT_TIMEOUT);
    assert_eq!(contacts(&client_a, cx_a), []);
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "offline", "free"),
            ("user_c".to_string(), "online", "free"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [
            ("user_a".to_string(), "offline", "free"),
            ("user_b".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_d, cx_d),
        [("user_b".to_string(), "online", "free")]
    );

    // Test removing a contact
    client_b
        .user_store()
        .update(cx_b, |store, cx| {
            store.remove_contact(client_c.user_id().unwrap(), cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();
    assert_eq!(
        contacts(&client_b, cx_b),
        [
            ("user_a".to_string(), "offline", "free"),
            ("user_d".to_string(), "online", "free")
        ]
    );
    assert_eq!(
        contacts(&client_c, cx_c),
        [("user_a".to_string(), "offline", "free"),]
    );

    fn contacts(
        client: &TestClient,
        cx: &TestAppContext,
    ) -> Vec<(String, &'static str, &'static str)> {
        client.user_store().read_with(cx, |store, _| {
            store
                .contacts()
                .iter()
                .map(|contact| {
                    (
                        contact.user.username.clone().to_string(),
                        if contact.online { "online" } else { "offline" },
                        if contact.busy { "busy" } else { "free" },
                    )
                })
                .collect()
        })
    }
}

#[gpui::test(iterations = 10)]
async fn test_contact_requests(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_a2: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_b2: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    cx_c2: &mut TestAppContext,
) {
    // Connect to a server as 3 clients.
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_a2 = server.create_client(cx_a2, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_b2 = server.create_client(cx_b2, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    let client_c2 = server.create_client(cx_c2, "user_c").await;

    assert_eq!(client_a.user_id().unwrap(), client_a2.user_id().unwrap());
    assert_eq!(client_b.user_id().unwrap(), client_b2.user_id().unwrap());
    assert_eq!(client_c.user_id().unwrap(), client_c2.user_id().unwrap());

    // User A and User C request that user B become their contact.
    client_a
        .user_store()
        .update(cx_a, |store, cx| {
            store.request_contact(client_b.user_id().unwrap(), cx)
        })
        .await
        .unwrap();
    client_c
        .user_store()
        .update(cx_c, |store, cx| {
            store.request_contact(client_b.user_id().unwrap(), cx)
        })
        .await
        .unwrap();
    executor.run_until_parked();

    // All users see the pending request appear in all their clients.
    assert_eq!(
        client_a.summarize_contacts(cx_a).outgoing_requests,
        &["user_b"]
    );
    assert_eq!(
        client_a2.summarize_contacts(cx_a2).outgoing_requests,
        &["user_b"]
    );
    assert_eq!(
        client_b.summarize_contacts(cx_b).incoming_requests,
        &["user_a", "user_c"]
    );
    assert_eq!(
        client_b2.summarize_contacts(cx_b2).incoming_requests,
        &["user_a", "user_c"]
    );
    assert_eq!(
        client_c.summarize_contacts(cx_c).outgoing_requests,
        &["user_b"]
    );
    assert_eq!(
        client_c2.summarize_contacts(cx_c2).outgoing_requests,
        &["user_b"]
    );

    // Contact requests are present upon connecting (tested here via disconnect/reconnect)
    disconnect_and_reconnect(&client_a, cx_a).await;
    disconnect_and_reconnect(&client_b, cx_b).await;
    disconnect_and_reconnect(&client_c, cx_c).await;
    executor.run_until_parked();
    assert_eq!(
        client_a.summarize_contacts(cx_a).outgoing_requests,
        &["user_b"]
    );
    assert_eq!(
        client_b.summarize_contacts(cx_b).incoming_requests,
        &["user_a", "user_c"]
    );
    assert_eq!(
        client_c.summarize_contacts(cx_c).outgoing_requests,
        &["user_b"]
    );

    // User B accepts the request from user A.
    client_b
        .user_store()
        .update(cx_b, |store, cx| {
            store.respond_to_contact_request(client_a.user_id().unwrap(), true, cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();

    // User B sees user A as their contact now in all client, and the incoming request from them is removed.
    let contacts_b = client_b.summarize_contacts(cx_b);
    assert_eq!(contacts_b.current, &["user_a"]);
    assert_eq!(contacts_b.incoming_requests, &["user_c"]);
    let contacts_b2 = client_b2.summarize_contacts(cx_b2);
    assert_eq!(contacts_b2.current, &["user_a"]);
    assert_eq!(contacts_b2.incoming_requests, &["user_c"]);

    // User A sees user B as their contact now in all clients, and the outgoing request to them is removed.
    let contacts_a = client_a.summarize_contacts(cx_a);
    assert_eq!(contacts_a.current, &["user_b"]);
    assert!(contacts_a.outgoing_requests.is_empty());
    let contacts_a2 = client_a2.summarize_contacts(cx_a2);
    assert_eq!(contacts_a2.current, &["user_b"]);
    assert!(contacts_a2.outgoing_requests.is_empty());

    // Contacts are present upon connecting (tested here via disconnect/reconnect)
    disconnect_and_reconnect(&client_a, cx_a).await;
    disconnect_and_reconnect(&client_b, cx_b).await;
    disconnect_and_reconnect(&client_c, cx_c).await;
    executor.run_until_parked();
    assert_eq!(client_a.summarize_contacts(cx_a).current, &["user_b"]);
    assert_eq!(client_b.summarize_contacts(cx_b).current, &["user_a"]);
    assert_eq!(
        client_b.summarize_contacts(cx_b).incoming_requests,
        &["user_c"]
    );
    assert!(client_c.summarize_contacts(cx_c).current.is_empty());
    assert_eq!(
        client_c.summarize_contacts(cx_c).outgoing_requests,
        &["user_b"]
    );

    // User B rejects the request from user C.
    client_b
        .user_store()
        .update(cx_b, |store, cx| {
            store.respond_to_contact_request(client_c.user_id().unwrap(), false, cx)
        })
        .await
        .unwrap();

    executor.run_until_parked();

    // User B doesn't see user C as their contact, and the incoming request from them is removed.
    let contacts_b = client_b.summarize_contacts(cx_b);
    assert_eq!(contacts_b.current, &["user_a"]);
    assert!(contacts_b.incoming_requests.is_empty());
    let contacts_b2 = client_b2.summarize_contacts(cx_b2);
    assert_eq!(contacts_b2.current, &["user_a"]);
    assert!(contacts_b2.incoming_requests.is_empty());

    // User C doesn't see user B as their contact, and the outgoing request to them is removed.
    let contacts_c = client_c.summarize_contacts(cx_c);
    assert!(contacts_c.current.is_empty());
    assert!(contacts_c.outgoing_requests.is_empty());
    let contacts_c2 = client_c2.summarize_contacts(cx_c2);
    assert!(contacts_c2.current.is_empty());
    assert!(contacts_c2.outgoing_requests.is_empty());

    // Incoming/outgoing requests are not present upon connecting (tested here via disconnect/reconnect)
    disconnect_and_reconnect(&client_a, cx_a).await;
    disconnect_and_reconnect(&client_b, cx_b).await;
    disconnect_and_reconnect(&client_c, cx_c).await;
    executor.run_until_parked();
    assert_eq!(client_a.summarize_contacts(cx_a).current, &["user_b"]);
    assert_eq!(client_b.summarize_contacts(cx_b).current, &["user_a"]);
    assert!(
        client_b
            .summarize_contacts(cx_b)
            .incoming_requests
            .is_empty()
    );
    assert!(client_c.summarize_contacts(cx_c).current.is_empty());
    assert!(
        client_c
            .summarize_contacts(cx_c)
            .outgoing_requests
            .is_empty()
    );

    async fn disconnect_and_reconnect(client: &TestClient, cx: &mut TestAppContext) {
        client.disconnect(&cx.to_async());
        client.clear_contacts(cx).await;
        client
            .connect(false, &cx.to_async())
            .await
            .into_response()
            .unwrap();
    }
}

#[gpui::test(iterations = 10)]
async fn test_join_call_after_screen_was_shared(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    // Call users B and C from client A.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: Default::default(),
            pending: vec!["user_b".to_string()]
        }
    );

    // User B receives the call.

    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    let call_b = incoming_call_b.next().await.unwrap().unwrap();
    assert_eq!(call_b.calling_user.username, "user_a");

    // User A shares their screen
    let display = gpui::TestScreenCaptureSource::new();
    cx_a.set_screen_capture_sources(vec![display]);
    let screen_a = cx_a
        .update(|cx| cx.screen_capture_sources())
        .await
        .unwrap()
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

    active_call_a
        .update(cx_a, |call, cx| {
            call.room()
                .unwrap()
                .update(cx, |room, cx| room.share_screen(screen_a, cx))
        })
        .await
        .unwrap();

    client_b.user_store().update(cx_b, |user_store, _| {
        user_store.clear_cache();
    });

    // User B joins the room
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    assert!(incoming_call_b.next().await.unwrap().is_none());

    executor.run_until_parked();
    assert_eq!(
        room_participants(&room_a, cx_a),
        RoomParticipants {
            remote: vec!["user_b".to_string()],
            pending: vec![],
        }
    );
    assert_eq!(
        room_participants(&room_b, cx_b),
        RoomParticipants {
            remote: vec!["user_a".to_string()],
            pending: vec![],
        }
    );

    // Ensure User B sees User A's screenshare.

    room_b.read_with(cx_b, |room, _| {
        assert_eq!(
            room.remote_participants()
                .get(&client_a.user_id().unwrap())
                .unwrap()
                .video_tracks
                .len(),
            1
        );
    });
}

#[cfg(target_os = "linux")]
#[gpui::test(iterations = 10)]
async fn test_share_screen_wayland(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    // User A calls user B.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    // User B accepts.
    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    executor.run_until_parked();
    incoming_call_b.next().await.unwrap().unwrap();
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    let room_b = active_call_b.read_with(cx_b, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();

    // User A shares their screen via the Wayland path.
    let events_b = active_call_events(cx_b);
    active_call_a
        .update(cx_a, |call, cx| {
            call.room()
                .unwrap()
                .update(cx, |room, cx| room.share_screen_wayland(cx))
        })
        .await
        .unwrap();

    executor.run_until_parked();

    // Room A is sharing and has a nonzero synthetic screen ID.
    room_a.read_with(cx_a, |room, _| {
        assert!(room.is_sharing_screen());
        let screen_id = room.shared_screen_id();
        assert!(screen_id.is_some(), "shared_screen_id should be Some");
        assert_ne!(screen_id.unwrap(), 0, "synthetic ID must be nonzero");
    });

    // User B observes the remote screen sharing track.
    assert_eq!(events_b.borrow().len(), 1);
    if let call::room::Event::RemoteVideoTracksChanged { participant_id } =
        events_b.borrow().first().unwrap()
    {
        assert_eq!(*participant_id, client_a.peer_id().unwrap());
        room_b.read_with(cx_b, |room, _| {
            assert_eq!(
                room.remote_participants()[&client_a.user_id().unwrap()]
                    .video_tracks
                    .len(),
                1
            );
        });
    } else {
        panic!("expected RemoteVideoTracksChanged event");
    }
}

#[cfg(target_os = "linux")]
#[gpui::test(iterations = 10)]
async fn test_unshare_screen_wayland(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;

    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .make_contacts(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    // User A calls user B.
    active_call_a
        .update(cx_a, |call, cx| {
            call.invite(client_b.user_id().unwrap(), None, cx)
        })
        .await
        .unwrap();

    // User B accepts.
    let mut incoming_call_b = active_call_b.read_with(cx_b, |call, _| call.incoming());
    executor.run_until_parked();
    incoming_call_b.next().await.unwrap().unwrap();
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();

    let room_a = active_call_a.read_with(cx_a, |call, _| call.room().unwrap().clone());
    executor.run_until_parked();

    // User A shares their screen via the Wayland path.
    active_call_a
        .update(cx_a, |call, cx| {
            call.room()
                .unwrap()
                .update(cx, |room, cx| room.share_screen_wayland(cx))
        })
        .await
        .unwrap();
    executor.run_until_parked();

    room_a.read_with(cx_a, |room, _| {
        assert!(room.is_sharing_screen());
    });

    // User A stops sharing.
    room_a
        .update(cx_a, |room, cx| room.unshare_screen(true, cx))
        .unwrap();
    executor.run_until_parked();

    // Room A is no longer sharing, screen ID is gone.
    room_a.read_with(cx_a, |room, _| {
        assert!(!room.is_sharing_screen());
        assert!(room.shared_screen_id().is_none());
    });
}

#[gpui::test]
async fn test_right_click_menu_behind_collab_panel(cx: &mut TestAppContext) {
    let mut server = TestServer::start(cx.executor().clone()).await;
    let client_a = server.create_client(cx, "user_a").await;
    let (workspace_a, cx) = client_a.build_test_workspace(cx).await;

    cx.simulate_resize(size(px(300.), px(300.)));

    cx.simulate_keystrokes("cmd-n cmd-n cmd-n");
    cx.update(|window, _cx| window.refresh());

    let new_tab_button_bounds = cx.debug_bounds("ICON-Plus").unwrap();

    cx.simulate_event(MouseDownEvent {
        button: MouseButton::Right,
        position: new_tab_button_bounds.center(),
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });

    // regression test that the right click menu for tabs does not open.
    assert!(cx.debug_bounds("MENU_ITEM-Close").is_none());

    let pane = workspace_a.update(cx, |workspace, _| workspace.active_pane().clone());
    pane.update_in(cx, |pane, window, cx| {
        pane.activate_item(1, true, true, window, cx);
    });
    cx.run_until_parked();
    assert_eq!(pane.read_with(cx, |pane, _| pane.active_item_index()), 1);
    let tab_bounds = cx.debug_bounds("TAB-1").unwrap();
    let click_position = tab_bounds.center();
    cx.update(|window, _| {
        assert!(
            gpui::Bounds::new(gpui::point(px(0.), px(0.)), window.viewport_size())
                .contains(&click_position)
        );
    });
    assert!(!new_tab_button_bounds.contains(&click_position));

    cx.simulate_event(MouseDownEvent {
        button: MouseButton::Right,
        position: tab_bounds.center(),
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    assert!(cx.debug_bounds("MENU_ITEM-Close").is_some());
}

#[gpui::test]
async fn test_pane_split_left(cx: &mut TestAppContext) {
    let (_, client) = TestServer::start1(cx).await;
    let (workspace, cx) = client.build_test_workspace(cx).await;

    cx.simulate_keystrokes("cmd-n");
    workspace.update(cx, |workspace, cx| {
        assert!(workspace.items(cx).collect::<Vec<_>>().len() == 1);
    });
    cx.simulate_keystrokes("cmd-k left");
    workspace.update(cx, |workspace, cx| {
        assert!(workspace.items(cx).collect::<Vec<_>>().len() == 2);
    });
    cx.simulate_keystrokes("cmd-k");
    // Sleep past the historical timeout to ensure the multi-stroke binding
    // still fires now that unambiguous prefixes no longer auto-expire.
    cx.executor().advance_clock(Duration::from_secs(2));
    cx.simulate_keystrokes("left");
    workspace.update(cx, |workspace, cx| {
        assert!(workspace.items(cx).collect::<Vec<_>>().len() == 3);
    });
}

#[gpui::test]
async fn test_join_after_restart(cx1: &mut TestAppContext, cx2: &mut TestAppContext) {
    let (mut server, client) = TestServer::start1(cx1).await;
    let channel1 = server.make_public_channel("channel1", &client, cx1).await;
    let channel2 = server.make_public_channel("channel2", &client, cx1).await;

    join_channel(channel1, &client, cx1).await.unwrap();
    drop(client);

    let client2 = server.create_client(cx2, "user_a").await;
    join_channel(channel2, &client2, cx2).await.unwrap();
}

#[gpui::test]
async fn test_preview_tabs(cx: &mut TestAppContext) {
    let (_server, client) = TestServer::start1(cx).await;
    let (workspace, cx) = client.build_test_workspace(cx).await;
    let project = workspace.read_with(cx, |workspace, _| workspace.project().clone());

    let worktree_id = project.update(cx, |project, cx| {
        project.worktrees(cx).next().unwrap().read(cx).id()
    });

    let path_1 = ProjectPath {
        worktree_id,
        path: rel_path("1.txt").into(),
    };
    let path_2 = ProjectPath {
        worktree_id,
        path: rel_path("2.js").into(),
    };
    let path_3 = ProjectPath {
        worktree_id,
        path: rel_path("3.rs").into(),
    };

    let pane = workspace.read_with(cx, |workspace, _| workspace.active_pane().clone());

    let get_path = |pane: &Pane, idx: usize, cx: &App| {
        pane.item_for_index(idx).unwrap().project_path(cx).unwrap()
    };

    // Opening item 3 as a "permanent" tab
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path(path_3.clone(), None, false, window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(pane.preview_item_id(), None);

        assert!(!pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Open item 1 as preview
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path_preview(path_1.clone(), None, true, true, true, window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(get_path(pane, 1, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Open item 2 as preview
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path_preview(path_2.clone(), None, true, true, true, window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(get_path(pane, 1, cx), path_2.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Going back should show item 1 as preview
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.go_back(pane.downgrade(), window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(get_path(pane, 1, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(pane.can_navigate_forward());
    });

    // Closing item 1
    pane.update_in(cx, |pane, window, cx| {
        pane.close_item_by_id(
            pane.active_item().unwrap().item_id(),
            workspace::SaveIntent::Skip,
            window,
            cx,
        )
    })
    .await
    .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(pane.preview_item_id(), None);

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Going back should show item 1 as preview
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.go_back(pane.downgrade(), window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_3.clone());
        assert_eq!(get_path(pane, 1, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(pane.can_navigate_forward());
    });

    // Close permanent tab
    pane.update_in(cx, |pane, window, cx| {
        let id = pane.items().next().unwrap().item_id();
        pane.close_item_by_id(id, workspace::SaveIntent::Skip, window, cx)
    })
    .await
    .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().next().unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(pane.can_navigate_forward());
    });

    // Split pane to the right
    pane.update_in(cx, |pane, window, cx| {
        pane.split(
            workspace::SplitDirection::Right,
            workspace::SplitMode::default(),
            window,
            cx,
        );
    });
    cx.run_until_parked();
    let right_pane = workspace.read_with(cx, |workspace, _| workspace.active_pane().clone());

    right_pane.update(cx, |pane, cx| {
        // Nav history is now cloned in an pane split, but that's inconvenient
        // for this test, which uses the presence of a backwards history item as
        // an indication that a preview item was successfully opened
        pane.nav_history_mut().clear(cx);
    });

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().next().unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(pane.can_navigate_forward());
    });

    right_pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(pane.preview_item_id(), None);

        assert!(!pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Open item 2 as preview in right pane
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path_preview(path_2.clone(), None, true, true, true, window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().next().unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(pane.can_navigate_forward());
    });

    right_pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(get_path(pane, 1, cx), path_2.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    // Focus left pane
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.activate_pane_in_direction(workspace::SplitDirection::Left, window, cx)
    });

    // Open item 2 as preview in left pane
    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path_preview(path_2.clone(), None, true, true, true, window, cx)
        })
        .await
        .unwrap();

    pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 1);
        assert_eq!(get_path(pane, 0, cx), path_2.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().next().unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });

    right_pane.update(cx, |pane, cx| {
        assert_eq!(pane.items_len(), 2);
        assert_eq!(get_path(pane, 0, cx), path_1.clone());
        assert_eq!(get_path(pane, 1, cx), path_2.clone());
        assert_eq!(
            pane.preview_item_id(),
            Some(pane.items().nth(1).unwrap().item_id())
        );

        assert!(pane.can_navigate_backward());
        assert!(!pane.can_navigate_forward());
    });
}

#[gpui::test]
async fn test_remote_git_branches(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    client_a
        .fs()
        .insert_tree("/project", serde_json::json!({ ".git":{} }))
        .await;
    let branches = ["main", "dev", "feature-1"];
    client_a
        .fs()
        .insert_branches(Path::new("/project/.git"), &branches);
    let branches_set = branches
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

    let (project_a, _) = client_a.build_local_project("/project", cx_a).await;

    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    // Client A sees that a guest has joined and the repo has been populated
    executor.run_until_parked();

    let repo_b = cx_b.update(|cx| project_b.read(cx).active_repository(cx).unwrap());

    let branches_b = cx_b
        .update(|cx| repo_b.update(cx, |repository, _| repository.branches()))
        .await
        .unwrap()
        .unwrap();

    let new_branch = branches[2];

    let branches_b = branches_b
        .branches
        .into_iter()
        .map(|branch| branch.name().to_string())
        .collect::<HashSet<_>>();

    assert_eq!(branches_b, branches_set);

    cx_b.update(|cx| {
        repo_b.update(cx, |repository, _cx| {
            repository.change_branch(new_branch.to_string())
        })
    })
    .await
    .unwrap()
    .unwrap();

    executor.run_until_parked();

    let host_branch = cx_a.update(|cx| {
        project_a.update(cx, |project, cx| {
            project
                .repositories(cx)
                .values()
                .next()
                .unwrap()
                .read(cx)
                .branch
                .as_ref()
                .unwrap()
                .clone()
        })
    });

    assert_eq!(host_branch.name(), branches[2]);

    // Also try creating a new branch
    cx_b.update(|cx| {
        repo_b.update(cx, |repository, _cx| {
            repository.create_branch("totally-new-branch".to_string(), None)
        })
    })
    .await
    .unwrap()
    .unwrap();

    cx_b.update(|cx| {
        repo_b.update(cx, |repository, _cx| {
            repository.change_branch("totally-new-branch".to_string())
        })
    })
    .await
    .unwrap()
    .unwrap();

    executor.run_until_parked();

    let host_branch = cx_a.update(|cx| {
        project_a.update(cx, |project, cx| {
            project
                .repositories(cx)
                .values()
                .next()
                .unwrap()
                .read(cx)
                .branch
                .as_ref()
                .unwrap()
                .clone()
        })
    });

    assert_eq!(host_branch.name(), "totally-new-branch");

    let default_branch_b = cx_b
        .update(|cx| repo_b.update(cx, |repository, _cx| repository.default_branch(false)))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(default_branch_b.as_deref(), Some("main"));

    let default_branch_with_remote_b = cx_b
        .update(|cx| repo_b.update(cx, |repository, _cx| repository.default_branch(true)))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(default_branch_with_remote_b.as_deref(), Some("origin/main"));
}

#[gpui::test]
async fn test_guest_can_rejoin_shared_project_after_leaving_call(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;

    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    client_a
        .fs()
        .insert_tree(
            path!("/project"),
            json!({
                "file.txt": "hello\n",
            }),
        )
        .await;

    let (project_a, _worktree_id) = client_a.build_local_project(path!("/project"), cx_a).await;
    let active_call_a = cx_a.read(ActiveCall::global);
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();

    let _project_b = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();

    // third client joins call to prevent room from being torn down
    let _project_c = client_c.join_remote_project(project_id, cx_c).await;
    executor.run_until_parked();

    let active_call_b = cx_b.read(ActiveCall::global);
    active_call_b
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    let user_id_b = client_b.current_user_id(cx_b).to_proto();
    let active_call_a = cx_a.read(ActiveCall::global);
    active_call_a
        .update(cx_a, |call, cx| call.invite(user_id_b, None, cx))
        .await
        .unwrap();
    executor.run_until_parked();
    let active_call_b = cx_b.read(ActiveCall::global);
    active_call_b
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .unwrap();
    executor.run_until_parked();

    let _project_b2 = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();

    project_a.read_with(cx_a, |project, _| {
        let guest_count = project
            .collaborators()
            .values()
            .filter(|c| !c.is_host)
            .count();

        assert_eq!(
            guest_count, 2,
            "host should have exactly one guest collaborator after rejoin"
        );
    });

    _project_b.read_with(cx_b, |project, _| {
        assert_eq!(
            project.client_subscriptions().len(),
            0,
            "We should clear all host subscriptions after leaving the project"
        );
    })
}

#[gpui::test]
async fn test_project_search_excludes_private_files(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);

    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |file| {
                file.project.worktree.private_files = Some(vec!["**/.env".to_string()].into());
            });
        });
    });

    client_a
        .fs()
        .insert_tree(
            "/root",
            json!({
                "dir-1": {
                    "a.txt": "the secret is out",
                    ".env": "API_KEY=private-secret-sentinel",
                }
            }),
        )
        .await;
    let (project_a, worktree_id) = client_a.build_local_project("/root/dir-1", cx_a).await;
    let private_buffer = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer(
                ProjectPath {
                    worktree_id,
                    path: rel_path(".env").into(),
                },
                cx,
            )
        })
        .await
        .expect("host can open its private file");
    assert!(private_buffer.read_with(cx_a, |buffer, _| {
        buffer.file().is_some_and(|file| file.is_private())
    }));
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share the private-search fixture");
    let project_b = client_b.join_remote_project(project_id, cx_b).await;

    let received_buffers = Rc::new(RefCell::new(Vec::new()));
    let _subscription = cx_b.update(|cx| {
        let store = project_b.read(cx).buffer_store().clone();
        let received_buffers = received_buffers.clone();
        cx.subscribe(&store, move |_, event, _| {
            if let project::buffer_store::BufferStoreEvent::BufferAdded(buffer) = event {
                received_buffers.borrow_mut().push(buffer.clone());
            }
        })
    });
    let mut results = HashMap::default();
    let search_rx = project_b.update(cx_b, |project, cx| {
        project.search(
            SearchQuery::text(
                "secret",
                false,
                false,
                false,
                Default::default(),
                Default::default(),
                false,
                None,
            )
            .expect("valid public/private search query"),
            cx,
        )
    });
    while let Ok(result) = search_rx.rx.recv().await {
        if let SearchResult::Buffer { buffer, ranges } = result {
            results.entry(buffer).or_insert(ranges);
        }
    }

    let mut paths = results
        .into_keys()
        .map(|buffer| {
            buffer.read_with(cx_b, |buffer, cx| {
                buffer
                    .file()
                    .expect("search result has a file")
                    .full_path(cx)
            })
        })
        .collect::<Vec<_>>();
    paths.sort();

    assert_eq!(
        paths,
        &[PathBuf::from("dir-1/a.txt")],
        "the guest's search returned a file the host marked private"
    );

    executor.run_until_parked();
    assert_eq!(
        received_buffers.borrow().len(),
        1,
        "only the public buffer is transferred"
    );
    for buffer in received_buffers.borrow().iter() {
        buffer.read_with(cx_b, |buffer, cx| {
            assert_eq!(
                buffer
                    .file()
                    .expect("transferred buffer has a file")
                    .full_path(cx),
                PathBuf::from("dir-1/a.txt")
            );
            assert!(!buffer.text().contains("private-secret-sentinel"));
        });
    }
}

async fn check_peer_open_current_policy(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    by_path: bool,
) {
    use rpc::ErrorExt as _;
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);
    client_a
        .fs()
        .insert_tree(
            "/root",
            json!({"peer-policy":{
                "secret.txt":"PEER_OPEN_PRIVATE_CANARY", "public.txt":"PEER_OPEN_PUBLIC_CONTROL"
            }}),
        )
        .await;
    let (project_a, worktree_id) = client_a
        .build_local_project("/root/peer-policy", cx_a)
        .await;
    let secret = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer(
                ProjectPath {
                    worktree_id,
                    path: rel_path("secret.txt").into(),
                },
                cx,
            )
        })
        .await
        .expect("owner opens initially public secret");
    let public = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer(
                ProjectPath {
                    worktree_id,
                    path: rel_path("public.txt").into(),
                },
                cx,
            )
        })
        .await
        .expect("owner opens public control");
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share project");
    let guest_project = client_b.join_remote_project(project_id, cx_b).await;
    let received_buffers = Rc::new(RefCell::new(Vec::new()));
    let _subscription = cx_b.update(|cx| {
        let store = guest_project.read(cx).buffer_store().clone();
        let received_buffers = received_buffers.clone();
        cx.subscribe(&store, move |_, event, _| {
            if let project::buffer_store::BufferStoreEvent::BufferAdded(buffer) = event {
                received_buffers.borrow_mut().push(buffer.clone());
            }
        })
    });
    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(vec!["**/secret.txt".into()].into());
            });
        })
    });
    executor.run_until_parked();
    secret.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("worktree-backed secret");
        assert!(
            !buffer.file().expect("cached file").is_private(),
            "fixture retains stale public flag"
        );
        assert!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("secret.txt")),
            "current policy denies secret"
        );
    });
    let secret_buffer_id = secret.read_with(cx_a, |buffer, _| buffer.remote_id());
    let secret_id = secret_buffer_id.to_proto();
    let response = if by_path {
        client_b
            .request(proto::OpenBufferByPath {
                project_id,
                worktree_id: worktree_id.to_proto(),
                path: "secret.txt".into(),
            })
            .await
    } else {
        client_b
            .request(proto::OpenBufferById {
                project_id,
                id: secret_id,
            })
            .await
    };
    let error = response
        .expect_err("peer open must return denial instead of an unpublished private buffer ID");
    assert_eq!(error.error_code(), proto::ErrorCode::UnsharedItem);
    let public_buffer_id = public.read_with(cx_a, |buffer, _| buffer.remote_id());
    let public_id = public_buffer_id.to_proto();
    let response = client_b
        .request(proto::OpenBufferById {
            project_id,
            id: public_id,
        })
        .await
        .expect("unrelated public buffer still opens");
    assert_eq!(response.buffer_id, public_id);
    let public_received = guest_project.update(cx_b, |project, cx| {
        project.buffer_store().update(cx, |store, cx| {
            store.wait_for_remote_buffer(public_buffer_id, cx)
        })
    });
    let public_received = resync_gate_within(&executor, public_received)
        .await
        .expect("public constructor receipt timeout")
        .expect("actual public constructor receipt");
    assert_eq!(
        public_received.read_with(cx_b, |buffer, _| buffer.text()),
        "PEER_OPEN_PUBLIC_CONTROL"
    );
    executor.run_until_parked();
    let guest_peer = client_b.peer_id().expect("connected guest");
    project_a.read_with(cx_a, |project, cx| {
        let store = project.buffer_store().read(cx);
        assert!(store.is_buffer_shared_with_peer(guest_peer, public_buffer_id));
        assert!(
            !store.is_buffer_shared_with_peer(guest_peer, secret_buffer_id),
            "denied peer open must not create host admission"
        );
    });
    guest_project.read_with(cx_b, |project, cx| {
        let store = project.buffer_store().read(cx);
        assert!(store.get_possibly_incomplete(secret_buffer_id).is_none());
        assert!(
            !store.has_buffer_or_pending_operations(secret_buffer_id),
            "denied peer open must not create a guest buffer or pending operations"
        );
    });
    assert_eq!(
        received_buffers.borrow().len(),
        1,
        "only the public constructor is transferred"
    );
    for buffer in received_buffers.borrow().iter() {
        buffer.read_with(cx_b, |buffer, _| {
            assert_eq!(buffer.remote_id(), public_buffer_id);
            assert!(!buffer.text().contains("PEER_OPEN_PRIVATE_CANARY"));
        });
    }
    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = Some(Vec::<String>::new().into());
            });
        })
    });
    let response = client_b
        .request(proto::OpenBufferById {
            project_id,
            id: secret_id,
        })
        .await
        .expect("policy correction permits a new peer open");
    assert_eq!(response.buffer_id, secret_id);
    let corrected_received = guest_project.update(cx_b, |project, cx| {
        project.buffer_store().update(cx, |store, cx| {
            store.wait_for_remote_buffer(secret_buffer_id, cx)
        })
    });
    let corrected_received = resync_gate_within(&executor, corrected_received)
        .await
        .expect("corrected constructor receipt timeout")
        .expect("actual corrected constructor receipt");
    assert_eq!(
        corrected_received.read_with(cx_b, |buffer, _| buffer.text()),
        "PEER_OPEN_PRIVATE_CANARY"
    );
    assert!(project_a.read_with(cx_a, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .is_buffer_shared_with_peer(guest_peer, secret_buffer_id)
    }));
}

#[gpui::test]
async fn test_peer_open_current_policy_by_id(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_open_current_policy(executor, cx_a, cx_b, false).await;
}

#[gpui::test]
async fn test_peer_open_current_policy_by_path(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_open_current_policy(executor, cx_a, cx_b, true).await;
}
async fn check_peer_resync_current_policy(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    explicit_opt_in: bool,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;
    client_a
        .fs()
        .insert_tree(
            "/root",
            json!({"peer-resync": {
                "secret.txt": "RESYNC_PREVIOUSLY_PUBLIC",
                "public.txt": "RESYNC_PUBLIC_CONTROL",
                "never-admitted-secret.txt": "RESYNC_NEVER_ADMITTED_BASE_CANARY"
            }}),
        )
        .await;
    let (project_a, worktree_id) = client_a
        .build_local_project("/root/peer-resync", cx_a)
        .await;
    let secret = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("owner opens initially public buffer");
    let public = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("owner opens public sibling");
    let never_admitted = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("never-admitted-secret.txt")), cx)
        })
        .await
        .expect("host opens fresh buffer without sharing it with the guest");
    let never_admitted_id = never_admitted.read_with(cx_a, |buffer, _| buffer.remote_id());
    let secret_id = secret.read_with(cx_a, |buffer, _| buffer.remote_id());
    let public_id = public.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share actual project");
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let guest_secret = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(secret_id, cx))
        .await
        .expect("guest receives initially public buffer");
    let guest_public = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(public_id, cx))
        .await
        .expect("guest receives public sibling");
    assert_eq!(
        guest_secret.read_with(cx_b, |buffer, _| buffer.text()),
        "RESYNC_PREVIOUSLY_PUBLIC"
    );
    let guest_peer = client_b.peer_id().expect("connected guest identity");
    project_a.read_with(cx_a, |project, cx| {
        let store = project.buffer_store().read(cx);
        assert!(store.is_buffer_shared_with_peer(guest_peer, secret_id));
        assert!(store.is_buffer_shared_with_peer(guest_peer, public_id));
        assert!(!store.is_buffer_shared_with_peer(guest_peer, never_admitted_id));
    });
    assert!(!project_b.read_with(cx_b, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .has_buffer_or_pending_operations(never_admitted_id)
    }));
    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(
                    vec![
                        "**/secret.txt".into(),
                        "**/never-admitted-secret.txt".into(),
                    ]
                    .into(),
                );
            });
        })
    });
    executor.run_until_parked();
    let worktree = secret.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("worktree file");
        assert!(
            !file.is_private,
            "fixture must retain stale public metadata"
        );
        assert!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("secret.txt")),
            "current owner must deny the private path"
        );
        file.worktree.clone()
    });
    secret.update(cx_a, |buffer, cx| {
        buffer.edit([(0..0, "RESYNC_AFTER_PRIVATE_CANARY ")], None, cx)
    });
    never_admitted.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("fresh worktree file");
        assert!(
            !file.is_private,
            "fresh fixture also retains public File metadata"
        );
        assert!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("never-admitted-secret.txt"))
        );
    });
    never_admitted.update(cx_a, |buffer, cx| {
        buffer.edit(
            [(0..0, "RESYNC_NEVER_ADMITTED_AFTER_PRIVATE_CANARY ")],
            None,
            cx,
        )
    });
    executor.run_until_parked();
    assert!(
        !guest_secret
            .read_with(cx_b, |buffer, _| buffer.text())
            .contains("RESYNC_AFTER_PRIVATE_CANARY"),
        "queued-operation prerequisite must prevent delivery before resynchronization"
    );
    let private_request = proto::SynchronizeBuffers {
        project_id,
        buffers: vec![proto::BufferVersion {
            id: secret_id.to_proto(),
            version: Vec::new(),
        }],
    };
    let response = client_b
        .request(private_request.clone())
        .await
        .expect("resync reports the permitted subset");
    assert!(
        response.buffers.is_empty(),
        "current private buffers must be omitted from resync response"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.buffer_store().read(cx);
        assert!(
            store.is_buffer_shared_with_peer(guest_peer, secret_id),
            "previously complete bookkeeping is not revoked by response omission"
        );
        assert!(
            store.is_buffer_shared_with_peer(guest_peer, public_id),
            "unrequested public sibling admission survives private-only resync"
        );
        assert!(!store.is_buffer_shared_with_peer(guest_peer, never_admitted_id));
    });
    let response = client_b
        .request(proto::SynchronizeBuffers {
            project_id,
            buffers: vec![proto::BufferVersion {
                id: never_admitted_id.to_proto(),
                version: Vec::new(),
            }],
        })
        .await
        .expect("fresh private resync reports the permitted subset");
    assert!(
        response.buffers.is_empty(),
        "never-admitted private buffer must not be advertised"
    );
    executor.run_until_parked();
    assert!(
        !project_a.read_with(cx_a, |project, cx| project
            .buffer_store()
            .read(cx)
            .is_buffer_shared_with_peer(guest_peer, never_admitted_id)),
        "denied fresh buffer must not acquire a new admission"
    );
    assert!(
        !project_b.read_with(cx_b, |project, cx| project
            .buffer_store()
            .read(cx)
            .has_buffer_or_pending_operations(never_admitted_id)),
        "denied fresh buffer must not leak content into an incomplete guest entry"
    );
    let response = client_b
        .request(proto::SynchronizeBuffers {
            project_id,
            buffers: vec![
                private_request
                    .buffers
                    .first()
                    .expect("private request")
                    .clone(),
                proto::BufferVersion {
                    id: never_admitted_id.to_proto(),
                    version: Vec::new(),
                },
                proto::BufferVersion {
                    id: public_id.to_proto(),
                    version: Vec::new(),
                },
            ],
        })
        .await
        .expect("unrelated public sibling still resynchronizes");
    assert_eq!(
        response
            .buffers
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        vec![public_id.to_proto()]
    );
    executor.run_until_parked();
    assert_eq!(
        guest_public.read_with(cx_b, |buffer, _| buffer.text()),
        public.read_with(cx_a, |buffer, _| buffer.text())
    );
    assert!(
        !guest_secret
            .read_with(cx_b, |buffer, _| buffer.text())
            .contains("RESYNC_AFTER_PRIVATE_CANARY"),
        "resync cannot transfer operations from a currently private buffer"
    );
    assert!(!project_b.read_with(cx_b, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .has_buffer_or_pending_operations(never_admitted_id)
    }));
    project_a.read_with(cx_a, |project, cx| {
        let store = project.buffer_store().read(cx);
        assert!(store.is_buffer_shared_with_peer(guest_peer, public_id));
        assert!(!store.is_buffer_shared_with_peer(guest_peer, never_admitted_id));
    });
    if explicit_opt_in {
        worktree.update(cx_a, |worktree, cx| {
            worktree
                .as_local_mut()
                .expect("local sharing authority")
                .share_private_files(cx)
        });
    } else {
        cx_a.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = Some(Vec::<String>::new().into());
                });
            })
        });
    }
    let response = client_b
        .request(private_request)
        .await
        .expect("policy correction or explicit owner opt-in permits resync");
    assert_eq!(
        response
            .buffers
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        vec![secret_id.to_proto()]
    );
    executor.run_until_parked();
    assert_eq!(
        guest_secret.read_with(cx_b, |buffer, _| buffer.text()),
        secret.read_with(cx_a, |buffer, _| buffer.text()),
        "accepted resync must deliver the withheld operation history"
    );
    let guest_fresh = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer_by_id(never_admitted_id, cx)
        })
        .await
        .expect("allowed ordinary peer open must send a complete fresh buffer after denied resync");
    assert_eq!(
        guest_fresh.read_with(cx_b, |buffer, _| buffer.text()),
        never_admitted.read_with(cx_a, |buffer, _| buffer.text()),
        "a ghost admission would incorrectly suppress the fresh State transfer"
    );
    assert!(project_a.read_with(cx_a, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .is_buffer_shared_with_peer(guest_peer, never_admitted_id)
    }));
}

#[gpui::test]
async fn test_peer_resync_current_policy_correction(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_resync_current_policy(executor, cx_a, cx_b, false).await;
}

#[gpui::test]
async fn test_peer_resync_current_policy_explicit_opt_in(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_resync_current_policy(executor, cx_a, cx_b, true).await;
}

#[derive(Clone, Copy)]
enum PeerBufferMetadataKind {
    File,
    Saved,
    Reloaded,
}

async fn check_peer_buffer_metadata_host_origin(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    kind: PeerBufferMetadataKind,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;
    client_a
        .fs()
        .insert_tree("/root/metadata", json!({"public.txt":"ORIGINAL\n"}))
        .await;
    let (project_a, worktree_id) = client_a.build_local_project("/root/metadata", cx_a).await;
    let buffer_a = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("host opens metadata fixture");
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share metadata fixture");
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let project_c = client_c.join_remote_project(project_id, cx_c).await;
    let buffer_b = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("attacker opens public fixture");
    let buffer_c = project_c
        .update(cx_c, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("observer opens public fixture");
    let buffer_id = buffer_a.read_with(cx_a, |buffer, _| buffer.remote_id().to_proto());
    assert_eq!(
        buffer_id,
        buffer_b.read_with(cx_b, |buffer, _| buffer.remote_id().to_proto())
    );
    assert_eq!(
        buffer_id,
        buffer_c.read_with(cx_c, |buffer, _| buffer.remote_id().to_proto())
    );
    if !matches!(kind, PeerBufferMetadataKind::File) {
        buffer_a.update(cx_a, |buffer, cx| {
            buffer.edit([(0..0, "UNSAVED ")], None, cx)
        });
        executor.run_until_parked();
        for (buffer, cx) in [
            (&buffer_a, &mut *cx_a),
            (&buffer_b, &mut *cx_b),
            (&buffer_c, &mut *cx_c),
        ] {
            buffer.read_with(cx, |buffer, _| {
                assert_eq!(buffer.text(), "UNSAVED ORIGINAL\n");
                assert!(buffer.is_dirty());
                assert_ne!(buffer.saved_version(), &buffer.version());
                assert_eq!(buffer.line_ending(), LineEnding::Unix);
            });
        }
    }
    let metadata = |buffer: &language::Buffer, cx: &gpui::App| {
        (
            buffer.file().expect("file metadata").to_proto(cx),
            buffer.text(),
            buffer.saved_version().clone(),
            buffer.saved_mtime(),
            buffer.is_dirty(),
            buffer.line_ending(),
            buffer.version(),
            buffer.has_conflict(),
        )
    };
    let before_a = buffer_a.read_with(cx_a, metadata);
    let before_c = buffer_c.read_with(cx_c, metadata);
    assert_eq!(before_a.1, before_c.1);
    let version = buffer_a.read_with(cx_a, |buffer, _| {
        language::proto::serialize_version(&buffer.version())
    });
    match kind {
        PeerBufferMetadataKind::File => {
            let mut forged_file = before_c.0.clone();
            forged_file.path = "forged-metadata.txt".into();
            client_b
                .send(proto::UpdateBufferFile {
                    project_id,
                    buffer_id,
                    file: Some(forged_file),
                })
                .expect("enqueue guest file metadata");
        }
        PeerBufferMetadataKind::Saved => {
            let mtime = proto::Timestamp {
                seconds: 9_876_543_210,
                nanos: 123,
            };
            assert_ne!(before_a.0.mtime.as_ref(), Some(&mtime));
            assert_ne!(before_c.0.mtime.as_ref(), Some(&mtime));
            client_b
                .send(proto::BufferSaved {
                    project_id,
                    buffer_id,
                    version,
                    mtime: Some(mtime),
                })
                .expect("enqueue guest saved metadata");
        }
        PeerBufferMetadataKind::Reloaded => {
            client_b
                .send(proto::BufferReloaded {
                    project_id,
                    buffer_id,
                    version,
                    mtime: Some(proto::Timestamp {
                        seconds: 9_876_543_210,
                        nanos: 456,
                    }),
                    line_ending: proto::LineEnding::Windows as i32,
                })
                .expect("enqueue guest reload metadata");
        }
    }
    executor.run_until_parked();
    let after_c = buffer_c.read_with(cx_c, metadata);
    assert_eq!(
        after_c, before_c,
        "guest metadata must not alter observing guest buffer state"
    );
    assert_eq!(
        buffer_a.read_with(cx_a, metadata),
        before_a,
        "guest metadata must not alter host buffer state"
    );
    match kind {
        PeerBufferMetadataKind::File => {
            let entry_id = project_a.read_with(cx_a, |project, cx| {
                project
                    .entry_for_path(&(worktree_id, rel_path("public.txt")).into(), cx)
                    .expect("source entry")
                    .id
            });
            project_a
                .update(cx_a, |project, cx| {
                    project.rename_entry(
                        entry_id,
                        (worktree_id, rel_path("renamed.txt")).into(),
                        cx,
                    )
                })
                .await
                .expect("host rename")
                .into_included()
                .expect("renamed included file");
            executor.run_until_parked();
            for (buffer, cx) in [
                (&buffer_a, &mut *cx_a),
                (&buffer_b, &mut *cx_b),
                (&buffer_c, &mut *cx_c),
            ] {
                buffer.read_with(cx, |buffer, _| {
                    assert_eq!(buffer.remote_id().to_proto(), buffer_id);
                    assert_eq!(
                        buffer.file().expect("renamed file").path().as_ref(),
                        rel_path("renamed.txt")
                    );
                    assert_eq!(buffer.text(), "ORIGINAL\n");
                });
            }
            assert!(
                !client_a
                    .fs()
                    .is_file(Path::new("/root/metadata/public.txt"))
                    .await
            );
            assert_eq!(
                client_a
                    .fs()
                    .load(Path::new("/root/metadata/renamed.txt"))
                    .await
                    .expect("renamed disk contents"),
                "ORIGINAL\n"
            );
        }
        PeerBufferMetadataKind::Saved => {
            project_a
                .update(cx_a, |project, cx| {
                    project.save_buffer(buffer_a.clone(), cx)
                })
                .await
                .expect("genuine host save");
            executor.run_until_parked();
            let saved = buffer_a.read_with(cx_a, metadata);
            assert!(!saved.4);
            for (buffer, cx) in [(&buffer_b, &mut *cx_b), (&buffer_c, &mut *cx_c)] {
                buffer.read_with(cx, |buffer, cx| {
                    assert_eq!(metadata(buffer, cx), saved);
                    assert_eq!(buffer.saved_version(), &buffer.version());
                });
            }
            assert_eq!(
                client_a
                    .fs()
                    .load(Path::new("/root/metadata/public.txt"))
                    .await
                    .expect("saved disk contents"),
                "UNSAVED ORIGINAL\n"
            );
        }
        PeerBufferMetadataKind::Reloaded => {
            client_a
                .fs()
                .save(
                    Path::new("/root/metadata/public.txt"),
                    &Rope::from("HOST_RELOAD_CONTROL\n"),
                    LineEnding::Windows,
                )
                .await
                .expect("change disk before genuine host reload");
            executor.run_until_parked();
            project_a
                .update(cx_a, |project, cx| {
                    project.reload_buffers(HashSet::from_iter([buffer_a.clone()]), true, cx)
                })
                .await
                .expect("genuine host reload");
            executor.run_until_parked();
            let reloaded = buffer_a.read_with(cx_a, metadata);
            assert_eq!(reloaded.1, "HOST_RELOAD_CONTROL\n");
            assert_eq!(reloaded.5, LineEnding::Windows);
            assert!(!reloaded.4);
            for (buffer, cx) in [(&buffer_b, &mut *cx_b), (&buffer_c, &mut *cx_c)] {
                buffer.read_with(cx, |buffer, cx| {
                    assert_eq!(metadata(buffer, cx), reloaded);
                    assert_eq!(buffer.saved_version(), &buffer.version());
                    assert!(!buffer.has_conflict());
                });
            }
        }
    }
}

#[gpui::test]
async fn test_peer_buffer_metadata_host_origin_file(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    check_peer_buffer_metadata_host_origin(
        executor,
        cx_a,
        cx_b,
        cx_c,
        PeerBufferMetadataKind::File,
    )
    .await;
}

#[gpui::test]
async fn test_peer_buffer_metadata_host_origin_saved(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    check_peer_buffer_metadata_host_origin(
        executor,
        cx_a,
        cx_b,
        cx_c,
        PeerBufferMetadataKind::Saved,
    )
    .await;
}

#[gpui::test]
async fn test_peer_buffer_metadata_host_origin_reloaded(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    check_peer_buffer_metadata_host_origin(
        executor,
        cx_a,
        cx_b,
        cx_c,
        PeerBufferMetadataKind::Reloaded,
    )
    .await;
}
async fn check_host_buffer_metadata_route_identity_and_compatibility(
    db: &Arc<collab::db::Database>,
) {
    use collab::db::{ChannelRole, RoomId};
    use rpc::ConnectionId;
    use sea_orm::{ColumnTrait as _, EntityTrait as _, QueryFilter as _};
    let owner_id = db
        .create_server("metadata-origin-test")
        .await
        .expect("routing server")
        .0 as u32;
    let host_user = db.create_user(false).await.expect("route host").user_id;
    let member_user = db.create_user(false).await.expect("route member").user_id;
    let host = ConnectionId { owner_id, id: 10 };
    let member = ConnectionId { owner_id, id: 11 };
    let room_id = RoomId::from_proto(
        db.create_room(host_user, host, "")
            .await
            .expect("route room")
            .id,
    );
    db.call(room_id, host_user, host, member_user, None)
        .await
        .expect("call member");
    db.join_room(room_id, member_user, member)
        .await
        .expect("member joins room");
    let project_id = db
        .share_project(room_id, host, &[], false, false, &[])
        .await
        .expect("share route project")
        .0;
    drop(
        db.join_project(project_id, member, member_user, None, None)
            .await
            .expect("member joins project"),
    );
    let expected = db
        .project_connection_ids(project_id, host, false)
        .await
        .expect("legacy recipients")
        .into_inner();
    assert!(expected.contains(&host) && expected.contains(&member));
    assert_eq!(
        *db.project_host_connection_ids(project_id, host)
            .await
            .expect("current host route"),
        expected
    );
    assert!(
        db.project_connection_ids(project_id, member, false)
            .await
            .is_ok(),
        "legacy membership route admits guest"
    );
    assert!(
        db.project_host_connection_ids(project_id, member)
            .await
            .is_err()
    );
    assert!(
        db.project_host_connection_ids(project_id, ConnectionId { owner_id, id: 12 })
            .await
            .is_err()
    );
    assert!(
        db.project_host_connection_ids(
            project_id,
            ConnectionId {
                owner_id: owner_id + 1,
                id: host.id
            }
        )
        .await
        .is_err(),
        "full host ConnectionId is required"
    );

    let replacement = ConnectionId { owner_id, id: 13 };
    drop(
        db.rejoin_room(
            proto::RejoinRoom {
                id: room_id.to_proto(),
                reshared_projects: vec![proto::UpdateProject {
                    project_id: project_id.to_proto(),
                    worktrees: vec![],
                    upstream_session_generation: None,
                }],
                rejoined_projects: vec![],
            },
            host_user,
            replacement,
        )
        .await
        .expect("public host replacement"),
    );
    assert!(
        db.project_host_connection_ids(project_id, host)
            .await
            .is_err(),
        "stale host is rejected"
    );
    let recipients = db
        .project_host_connection_ids(project_id, replacement)
        .await
        .expect("replacement host route")
        .into_inner();
    assert!(recipients.contains(&replacement) && recipients.contains(&member));
    assert!(!recipients.contains(&host));

    drop(
        db.set_room_participant_role(host_user, room_id, host_user, ChannelRole::Banned)
            .await
            .expect("synthetic banned host compatibility state"),
    );
    assert!(
        db.connections_for_buffer_update(project_id, replacement, collab::db::Capability::ReadOnly)
            .await
            .is_err(),
        "ReadOnly adds a room-role restriction"
    );
    assert_eq!(
        *db.project_host_connection_ids(project_id, replacement)
            .await
            .expect("host origin preserves legacy room-role semantics"),
        recipients
    );
    let query = collab::db::project::Entity::update_many()
        .filter(collab::db::project::Column::Id.eq(project_id))
        .col_expr(
            collab::db::project::Column::RoomId,
            sea_orm::sea_query::Expr::value(Option::<i32>::None),
        )
        .exec(&db.pool);
    let changed = db
        .test_options
        .as_ref()
        .expect("database fixture runtime")
        .runtime
        .block_on(query)
        .expect("synthetic schema-valid roomless project");
    assert_eq!(changed.rows_affected, 1);
    assert!(
        db.connections_for_buffer_update(project_id, replacement, collab::db::Capability::ReadOnly)
            .await
            .is_err(),
        "ReadOnly adds a room association restriction"
    );
    assert_eq!(
        *db.project_host_connection_ids(project_id, replacement)
            .await
            .expect("current host remains routable without a room"),
        recipients
    );
    assert!(
        db.project_host_connection_ids(project_id, member)
            .await
            .is_err()
    );
}

#[gpui::test]
async fn test_host_buffer_metadata_route_identity_and_compatibility_postgres(
    executor: BackgroundExecutor,
    _cx: &mut TestAppContext,
) {
    let server = TestServer::start_postgres(executor).await;
    check_host_buffer_metadata_route_identity_and_compatibility(server.test_db.db()).await;
}

#[gpui::test]
async fn test_host_buffer_metadata_route_identity_and_compatibility_sqlite(
    cx: &mut TestAppContext,
) {
    let database = crate::db_tests::TestDb::sqlite(cx.executor());
    check_host_buffer_metadata_route_identity_and_compatibility(database.db()).await;
}

#[derive(Clone, Copy)]
enum ResyncGateCase {
    PreparedPrivate,
    PreparedRebind,
    AckPrivate,
    AckRebind,
    AckFailure,
    ConcurrentPublic,
}

async fn resync_gate_within<T>(
    executor: &BackgroundExecutor,
    future: impl std::future::Future<Output = T>,
) -> anyhow::Result<T> {
    match futures::future::select(
        Box::pin(future),
        Box::pin(executor.timer(Duration::from_secs(5))),
    )
    .await
    {
        futures::future::Either::Left((output, _)) => Ok(output),
        futures::future::Either::Right(_) => Err(anyhow!("resync fixture checkpoint timed out")),
    }
}

fn resync_gate_set_private(cx: &mut TestAppContext, private: bool) {
    cx.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(
                    if private {
                        vec!["**/target.txt".to_string()]
                    } else {
                        Vec::new()
                    }
                    .into(),
                );
            });
        })
    });
}

fn resync_gate_has_text(operations: &[proto::Operation], marker: &str) -> bool {
    operations.iter().any(|operation| match &operation.variant {
        Some(proto::operation::Variant::Edit(edit)) => {
            edit.new_text.iter().any(|text| text.contains(marker))
        }
        _ => false,
    })
}

fn assert_resync_metadata_pair(messages: &[proto::Envelope], project_id: u64, buffer_id: u64) {
    assert_eq!(
        messages.len(),
        2,
        "exactly two authorized metadata messages"
    );
    assert_eq!(messages.iter().filter(|message| matches!(&message.payload,
        Some(proto::envelope::Payload::UpdateBufferFile(file)) if file.project_id == project_id && file.buffer_id == buffer_id
    )).count(), 1, "one file metadata message for the exact target");
    assert_eq!(messages.iter().filter(|message| matches!(&message.payload,
        Some(proto::envelope::Payload::BufferReloaded(reloaded)) if reloaded.project_id == project_id && reloaded.buffer_id == buffer_id
    )).count(), 1, "one saved/reloaded metadata message for the exact target");
}

async fn check_resync_gate(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    case: ResyncGateCase,
) {
    use rpc::ErrorExt as _;
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let host = server.create_client(cx_a, "resync_host").await;
    let guest = server.create_client(cx_b, "resync_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree(
            "/root",
            json!({"resync-gate": {
                "target.txt": "TARGET_BASE", "prior.txt": "PRIOR_PUBLIC_BASE"
            }}),
        )
        .await;
    let (project_a, worktree_id) = host.build_local_project("/root/resync-gate", cx_a).await;
    let target = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("target.txt")), cx)
        })
        .await
        .expect("host target");
    let target_id = target.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share real project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let guest_target = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(target_id, cx))
        .await
        .expect("guest has a complete target base");
    let guest_peer = guest.peer_id().expect("connected guest");
    let host_peer = host.peer_id().expect("connected host");
    let store = project_a.read_with(cx_a, |project, _| project.buffer_store().clone());
    assert!(store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    store.update(cx_a, |store, _| {
        store.forget_shared_buffers_for(&guest_peer)
    });
    // Actual lifecycle forget models a retained guest buffer with absent constructor bookkeeping.
    let prior = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("prior.txt")), cx)
        })
        .await
        .expect("host public sibling");
    let prior_id = prior.read_with(cx_a, |buffer, _| buffer.remote_id());
    let guest_prior = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(prior_id, cx))
        .await
        .expect("guest admits the public sibling after lifecycle forget");
    assert!(!store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    assert!(store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, prior_id)
    }));

    resync_gate_set_private(cx_a, true);
    executor.run_until_parked();
    target.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("worktree target");
        assert!(
            !file.is_private,
            "excluded scan must keep stale public metadata"
        );
        assert!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("target.txt"))
        );
    });
    for index in 0..12 {
        let text = format!("WITHHELD_{index} ");
        target.update(cx_a, |buffer, cx| {
            buffer.edit([(0..0, text.as_str())], None, cx)
        });
    }
    target.update(cx_a, |buffer, cx| {
        buffer.edit([(0..0, "RESYNC_BLOCKED_TAIL_CANARY ")], None, cx)
    });
    executor.run_until_parked();
    assert_eq!(
        guest_target.read_with(cx_b, |buffer, _| buffer.text()),
        "TARGET_BASE",
        "ordinary egress must have been denied while preparing withheld history"
    );
    resync_gate_set_private(cx_a, false);
    executor.run_until_parked();

    let (prepared, release_prepared) = store
        .update(cx_a, |store, _| {
            store.hold_synchronization_preparation_for_test(
                project_id,
                guest_peer,
                target_id.to_proto(),
            )
        })
        .expect("arm exact owner preparation checkpoint");
    let (mut ack, release_ack) = server
        .hold_update_buffer_ack(host_peer, project_id, target_id.to_proto())
        .expect("arm actual server Ack checkpoint");
    let metadata_observed = if matches!(
        case,
        ResyncGateCase::PreparedPrivate | ResyncGateCase::PreparedRebind
    ) {
        None
    } else {
        Some(
            server
                .observe_buffer_metadata_pair(host_peer, project_id, target_id.to_proto())
                .expect("observe actual authorized file and reload handlers independently of Ack"),
        )
    };
    let client = guest.client().clone();
    let (result_sender, mut result_receiver) = futures::channel::oneshot::channel();
    let synchronization = cx_b.spawn(async move |_| {
        let result = client
            .request(proto::SynchronizeBuffers {
                project_id,
                buffers: vec![proto::BufferVersion {
                    id: target_id.to_proto(),
                    version: Vec::new(),
                }],
            })
            .await;
        result_sender
            .send(result)
            .map_err(|_| anyhow!("test result observer dropped"))
    });
    let checkpoint = resync_gate_within(&executor, prepared)
        .await
        .expect("preparation checkpoint timeout")
        .expect("preparation observer");
    assert_eq!(checkpoint.version.id, target_id.to_proto());
    assert!(
        checkpoint.chunks.len() >= 3,
        "require real multi-chunk history plus final empty chunk"
    );
    assert!(checkpoint.chunks.last().expect("chunks").is_empty());
    assert!(!resync_gate_has_text(
        checkpoint.chunks.first().expect("first chunk"),
        "RESYNC_BLOCKED_TAIL_CANARY"
    ));
    assert!(
        checkpoint
            .chunks
            .iter()
            .skip(1)
            .any(|chunk| resync_gate_has_text(chunk, "RESYNC_BLOCKED_TAIL_CANARY"))
    );
    assert!(server.recorded_update_buffers().is_empty());
    assert!(server.recorded_buffer_metadata().is_empty());
    assert!(
        result_receiver
            .try_recv()
            .expect("live response observer")
            .is_none()
    );
    assert!(!store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    if matches!(case, ResyncGateCase::PreparedPrivate) {
        resync_gate_set_private(cx_a, true);
    }
    if matches!(case, ResyncGateCase::PreparedRebind) {
        let client = host.client().clone();
        store.update(cx_a, |store, cx| {
            store.shared(project_id, client.into(), cx)
        });
    }
    release_prepared
        .send(())
        .expect("owner still waiting at checkpoint");

    if !matches!(
        case,
        ResyncGateCase::PreparedPrivate | ResyncGateCase::PreparedRebind
    ) {
        let held = resync_gate_within(&executor, &mut ack)
            .await
            .expect("Ack checkpoint timeout")
            .expect("Ack checkpoint observer");
        assert_eq!(held.sender_id, host_peer);
        assert_eq!(held.request.project_id, project_id);
        assert_eq!(held.request.buffer_id, target_id.to_proto());
        assert_eq!(
            held.request.operations,
            checkpoint.chunks.first().expect("first chunk").clone()
        );
        assert!(
            result_receiver
                .try_recv()
                .expect("live response observer")
                .is_none(),
            "Sync cannot respond while its real UpdateBuffer Ack remains held"
        );
        assert!(!store.read_with(cx_a, |store, _| {
            store.is_buffer_shared_with_peer(guest_peer, target_id)
        }));
        resync_gate_within(
            &executor,
            metadata_observed.expect("metadata observer armed for published case"),
        )
        .await
        .expect("authorized metadata checkpoint timeout")
        .expect("authorized metadata observer");
        assert_eq!(server.recorded_buffer_metadata().len(), 2);
        assert_resync_metadata_pair(
            &server.recorded_buffer_metadata(),
            project_id,
            target_id.to_proto(),
        );
        if matches!(case, ResyncGateCase::AckPrivate) {
            resync_gate_set_private(cx_a, true);
        }
        if matches!(case, ResyncGateCase::AckRebind) {
            let client = host.client().clone();
            store.update(cx_a, |store, cx| {
                store.shared(project_id, client.into(), cx)
            });
        }
        if matches!(case, ResyncGateCase::ConcurrentPublic) {
            target.update(cx_a, |buffer, cx| {
                buffer.edit([(0..0, "CONCURRENT_PUBLIC_CANARY ")], None, cx)
            });
            executor.run_until_parked();
            assert!(
                server
                    .recorded_update_buffers()
                    .iter()
                    .any(|record| resync_gate_has_text(
                        &record.request.operations,
                        "CONCURRENT_PUBLIC_CANARY"
                    )),
                "ordinary public UpdateBuffer must bypass the consumed resync Ack gate"
            );
            let guest_received = guest_target
                .read_with(cx_b, |buffer, cx| buffer.serialize_ops(None, cx))
                .await;
            assert!(
                resync_gate_has_text(&guest_received, "CONCURRENT_PUBLIC_CANARY"),
                "joined guest receives the concurrent operation despite absent constructor admission"
            );
            assert!(
                guest_target.read_with(cx_b, |buffer, _| buffer.has_deferred_ops()),
                "concurrent edit can be deferred until missing captured history arrives"
            );
            assert!(
                result_receiver
                    .try_recv()
                    .expect("live response observer")
                    .is_none()
            );
            assert_ne!(
                target.read_with(cx_a, |buffer, _| language::proto::serialize_version(
                    &buffer.version()
                )),
                checkpoint.version.version,
                "force a meaningful concurrent version change"
            );
        }
        release_ack
            .send(if matches!(case, ResyncGateCase::AckFailure) {
                Err("RESYNC_ACK_FAILURE_CANARY".to_string())
            } else {
                Ok(())
            })
            .expect("server still awaits explicit Ack outcome");
    }
    let result = resync_gate_within(&executor, result_receiver)
        .await
        .expect("Sync response timeout")
        .expect("Sync response observer");
    synchronization.await.expect("result transfer task");
    executor.run_until_parked();
    assert!(
        store.read_with(cx_a, |store, _| store
            .is_buffer_shared_with_peer(guest_peer, prior_id)),
        "unrequested complete sibling bookkeeping survives"
    );
    assert_eq!(
        guest_prior.read_with(cx_b, |buffer, _| buffer.text()),
        "PRIOR_PUBLIC_BASE"
    );
    match case {
        ResyncGateCase::PreparedPrivate | ResyncGateCase::AckPrivate => {
            assert!(
                result
                    .expect("current private item is omitted")
                    .buffers
                    .is_empty()
            );
            assert!(!store.read_with(cx_a, |store, _| {
                store.is_buffer_shared_with_peer(guest_peer, target_id)
            }));
            assert!(
                !guest_target
                    .read_with(cx_b, |buffer, _| buffer.text())
                    .contains("RESYNC_BLOCKED_TAIL_CANARY")
            );
        }
        ResyncGateCase::PreparedRebind | ResyncGateCase::AckRebind => {
            let error = result.expect_err("same-ID/role rebind invalidates the request token");
            assert!(
                error
                    .to_string()
                    .contains("Buffer synchronization request was superseded")
            );
            assert!(!store.read_with(cx_a, |store, _| {
                store.is_buffer_shared_with_peer(guest_peer, target_id)
            }));
        }
        ResyncGateCase::AckFailure => {
            let error =
                result.expect_err("actual UpdateBuffer RPC error must propagate to Sync caller");
            assert_eq!(error.error_code(), rpc::ErrorCode::Internal);
            assert!(error.to_string().contains("RESYNC_ACK_FAILURE_CANARY"));
            assert!(!store.read_with(cx_a, |store, _| {
                store.is_buffer_shared_with_peer(guest_peer, target_id)
            }));
        }
        ResyncGateCase::ConcurrentPublic => {
            let response =
                result.expect("concurrent public edit does not trigger unsupported drift refusal");
            assert_eq!(response.buffers, vec![checkpoint.version.clone()]);
            assert!(store.read_with(cx_a, |store, _| {
                store.is_buffer_shared_with_peer(guest_peer, target_id)
            }));
            assert_eq!(
                guest_target.read_with(cx_b, |buffer, _| buffer.text()),
                target.read_with(cx_a, |buffer, _| buffer.text())
            );
            assert!(!guest_target.read_with(cx_b, |buffer, _| buffer.has_deferred_ops()));
            let sent_resync: Vec<_> = server
                .recorded_update_buffers()
                .into_iter()
                .filter(|record| {
                    !resync_gate_has_text(&record.request.operations, "CONCURRENT_PUBLIC_CANARY")
                })
                .map(|record| record.request.operations)
                .collect();
            assert_eq!(
                sent_resync, checkpoint.chunks,
                "every captured resync chunk, including empty tail, crosses real transport"
            );
        }
    }
    if matches!(
        case,
        ResyncGateCase::PreparedPrivate | ResyncGateCase::PreparedRebind
    ) {
        assert!(server.recorded_update_buffers().is_empty());
        assert!(server.recorded_buffer_metadata().is_empty());
        assert!(
            ack.try_recv()
                .expect("unused Ack observer remains live")
                .is_none()
        );
    } else if !matches!(case, ResyncGateCase::ConcurrentPublic) {
        assert_eq!(
            server.recorded_update_buffers().len(),
            1,
            "no resync chunks after policy/lifecycle change or injected request failure"
        );
        assert_eq!(server.recorded_buffer_metadata().len(), 2);
    }
}

#[gpui::test]
async fn test_resync_prepared_current_policy(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::PreparedPrivate).await;
}
#[gpui::test]
async fn test_resync_prepared_same_scope_rebind(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::PreparedRebind).await;
}
#[gpui::test]
async fn test_resync_ack_current_policy(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::AckPrivate).await;
}
#[gpui::test]
async fn test_resync_ack_same_scope_rebind(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::AckRebind).await;
}
#[gpui::test]
async fn test_resync_ack_failure_propagates(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::AckFailure).await;
}
#[gpui::test]
async fn test_resync_concurrent_public_broadcast(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_gate(executor, cx_a, cx_b, ResyncGateCase::ConcurrentPublic).await;
}

#[derive(Clone, Copy)]
enum ResyncHeldMetadataCase {
    Save,
    Reload,
}

async fn check_resync_held_current_metadata(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    case: ResyncHeldMetadataCase,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let host = server.create_client(cx_a, "metadata_host").await;
    let guest = server.create_client(cx_b, "metadata_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree("/root/resync-metadata", json!({"public.txt":"ORIGINAL\n"}))
        .await;
    let (project_a, worktree_id) = host
        .build_local_project("/root/resync-metadata", cx_a)
        .await;
    let buffer_a = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("host public file");
    let buffer_id = buffer_a.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share metadata project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let buffer_b = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(buffer_id, cx))
        .await
        .expect("guest complete public file");
    buffer_a.update(cx_a, |buffer, cx| {
        buffer.edit([(0..0, "UNSAVED ")], None, cx)
    });
    executor.run_until_parked();
    let metadata = |buffer: &language::Buffer, cx: &App| {
        (
            language::proto::serialize_version(buffer.saved_version()),
            buffer.saved_mtime(),
            buffer.line_ending(),
            buffer.is_dirty(),
            buffer.has_conflict(),
            buffer.file().map(|file| file.to_proto(cx)),
            buffer.text(),
        )
    };
    let before = buffer_a.read_with(cx_a, metadata);
    assert!(before.3, "real unsaved edit must predate serialization");
    assert_eq!(before.2, LineEnding::Unix);
    assert_eq!(buffer_b.read_with(cx_b, metadata), before);
    let store = project_a.read_with(cx_a, |project, _| project.buffer_store().clone());
    let guest_peer = guest.peer_id().expect("guest Connected");
    let host_peer = host.peer_id().expect("host Connected");
    let (prepared, release_prepared) = store
        .update(cx_a, |store, _| {
            store.hold_synchronization_preparation_for_test(
                project_id,
                guest_peer,
                buffer_id.to_proto(),
            )
        })
        .expect("hold real serialization result");
    let guest_version = buffer_b.read_with(cx_b, |buffer, _| {
        language::proto::serialize_version(&buffer.version())
    });
    let client = guest.client().clone();
    let (result_sender, mut result_receiver) = futures::channel::oneshot::channel();
    let synchronization = cx_b.spawn(async move |_| {
        let result = client
            .request(proto::SynchronizeBuffers {
                project_id,
                buffers: vec![proto::BufferVersion {
                    id: buffer_id.to_proto(),
                    version: guest_version,
                }],
            })
            .await;
        result_sender
            .send(result)
            .map_err(|_| anyhow!("metadata result observer dropped"))
    });
    let checkpoint = resync_gate_within(&executor, prepared)
        .await
        .expect("serialization timeout")
        .expect("serialization observer");
    assert_ne!(
        checkpoint.version.version, before.0,
        "captured operations version must differ from pre-save saved version"
    );
    match case {
        ResyncHeldMetadataCase::Save => {
            project_a
                .update(cx_a, |project, cx| {
                    project.save_buffer(buffer_a.clone(), cx)
                })
                .await
                .expect("genuine host save while serialization result held");
            assert_eq!(
                host.fs()
                    .load(Path::new("/root/resync-metadata/public.txt"))
                    .await
                    .expect("saved disk text"),
                "UNSAVED ORIGINAL\n"
            );
        }
        ResyncHeldMetadataCase::Reload => {
            host.fs()
                .save(
                    Path::new("/root/resync-metadata/public.txt"),
                    &Rope::from("RELOADED_CONTROL\n"),
                    LineEnding::Windows,
                )
                .await
                .expect("genuine disk replacement");
            executor.run_until_parked();
            project_a
                .update(cx_a, |project, cx| {
                    project.reload_buffers(HashSet::from_iter([buffer_a.clone()]), true, cx)
                })
                .await
                .expect("genuine host reload while serialization result held");
        }
    }
    executor.run_until_parked();
    let newer = buffer_a.read_with(cx_a, metadata);
    assert_ne!(newer.0, before.0, "save/reload must advance saved version");
    assert_ne!(newer.1, before.1, "save/reload must change saved mtime");
    assert_ne!(
        newer.5, before.5,
        "real file metadata must advance before resync release"
    );
    assert!(!newer.3 && !newer.4);
    assert_eq!(
        newer.0,
        buffer_a.read_with(cx_a, |buffer, _| language::proto::serialize_version(
            &buffer.version()
        ))
    );
    if matches!(case, ResyncHeldMetadataCase::Reload) {
        assert_eq!(newer.2, LineEnding::Windows);
        assert_eq!(newer.6, "RELOADED_CONTROL\n");
        assert_ne!(
            newer.0, checkpoint.version.version,
            "reload genuinely changes current version after captured operation snapshot"
        );
    }
    assert_eq!(
        buffer_b.read_with(cx_b, metadata),
        newer,
        "observe actual newer metadata at guest before releasing old serialization result"
    );
    assert!(
        result_receiver
            .try_recv()
            .expect("live result observer")
            .is_none()
    );
    // Arm after the genuine save/reload operations drained, so its ordinary updates cannot consume this gate.
    let (ack, release_ack) = server
        .hold_update_buffer_ack(host_peer, project_id, buffer_id.to_proto())
        .expect("record only resumed resync metadata and hold its first Ack");
    let metadata_observed = server
        .observe_buffer_metadata_pair(host_peer, project_id, buffer_id.to_proto())
        .expect("observe only resumed file and reload handlers after genuine save/reload drained");
    release_prepared
        .send(())
        .expect("owner remains at serialization checkpoint");
    let held = resync_gate_within(&executor, ack)
        .await
        .expect("Ack timeout")
        .expect("Ack observer");
    assert_eq!(
        held.request.operations,
        checkpoint
            .chunks
            .first()
            .expect("final empty chunk exists")
            .clone()
    );
    resync_gate_within(&executor, metadata_observed)
        .await
        .expect("resumed authorized metadata timeout")
        .expect("resumed metadata observer");
    let messages = server.recorded_buffer_metadata();
    assert_eq!(messages.len(), 2);
    assert_resync_metadata_pair(&messages, project_id, buffer_id.to_proto());
    let file = messages
        .iter()
        .find_map(|message| match &message.payload {
            Some(proto::envelope::Payload::UpdateBufferFile(file)) => Some(file),
            _ => None,
        })
        .expect("resumed file metadata");
    let reloaded = messages
        .iter()
        .find_map(|message| match &message.payload {
            Some(proto::envelope::Payload::BufferReloaded(reloaded)) => Some(reloaded),
            _ => None,
        })
        .expect("resumed saved/reloaded metadata");
    assert_eq!(
        file.file, newer.5,
        "file metadata must be recaptured at guarded send"
    );
    assert_eq!(
        reloaded.version, newer.0,
        "saved version must not replay pre-await value"
    );
    assert_eq!(reloaded.mtime, newer.1.map(Into::into));
    assert_eq!(
        reloaded.line_ending,
        language::proto::serialize_line_ending(newer.2) as i32
    );
    release_ack.send(Ok(())).expect("real server Ack gate");
    let response = resync_gate_within(&executor, result_receiver)
        .await
        .expect("Sync timeout")
        .expect("result observer")
        .expect("successful current metadata resync");
    synchronization.await.expect("result transfer");
    assert_eq!(
        response.buffers,
        vec![checkpoint.version],
        "response operations version remains captured even when saved metadata advances"
    );
    executor.run_until_parked();
    assert_eq!(
        buffer_b.read_with(cx_b, metadata),
        newer,
        "resumed resync must not regress already-observed saved/file/line-ending state"
    );
    assert!(!buffer_b.read_with(cx_b, |buffer, _| buffer.has_deferred_ops()));
}

#[gpui::test]
async fn test_resync_held_save_current_metadata(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_held_current_metadata(executor, cx_a, cx_b, ResyncHeldMetadataCase::Save).await;
}
#[gpui::test]
async fn test_resync_held_reload_current_metadata(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_held_current_metadata(executor, cx_a, cx_b, ResyncHeldMetadataCase::Reload).await;
}

async fn queued_flusher_within<T>(
    executor: &BackgroundExecutor,
    future: impl std::future::Future<Output = T>,
) -> anyhow::Result<T> {
    match futures::future::select(
        Box::pin(future),
        Box::pin(executor.timer(Duration::from_secs(5))),
    )
    .await
    {
        futures::future::Either::Left((output, _)) => Ok(output),
        futures::future::Either::Right(_) => Err(anyhow!("queued flusher checkpoint timed out")),
    }
}

fn queued_flusher_set_private(cx: &mut TestAppContext, private: bool) {
    cx.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(
                    if private {
                        vec!["**/secret.txt".to_string()]
                    } else {
                        Vec::new()
                    }
                    .into(),
                );
            });
        })
    });
}

fn queued_flusher_has_marker(operations: &[proto::Operation], marker: &str) -> bool {
    operations.iter().any(|operation| match &operation.variant {
        Some(proto::operation::Variant::Edit(edit)) => {
            edit.new_text.iter().any(|text| text.contains(marker))
        }
        _ => false,
    })
}

async fn check_queued_flusher_current_policy(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    starts_private: bool,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let host = server.create_client(cx_a, "queued_host").await;
    let guest = server.create_client(cx_b, "queued_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree(
            "/root/queued-flusher",
            json!({
                "secret.txt": "SECRET_BASE", "public.txt": "PUBLIC_BASE"
            }),
        )
        .await;
    let (project_a, worktree_id) = host.build_local_project("/root/queued-flusher", cx_a).await;
    let secret = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("host secret base");
    let public = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("host public base");
    let secret_id = secret.read_with(cx_a, |buffer, _| buffer.remote_id());
    let public_id = public.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share genuine project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let guest_secret = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(secret_id, cx))
        .await
        .expect("guest complete secret base before policy changes");
    let guest_public = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(public_id, cx))
        .await
        .expect("guest complete public sibling");
    let initial_secret_version = guest_secret.read_with(cx_b, |buffer, _| buffer.version());
    queued_flusher_set_private(cx_a, starts_private);
    executor.run_until_parked();
    secret.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("local worktree file");
        assert!(
            !file.is_private,
            "scan exclusions retain stale public File metadata"
        );
        assert_eq!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("secret.txt")),
            starts_private
        );
    });
    let (ready, release, settled) = project_a
        .update(cx_a, |project, _| {
            project.hold_queued_buffer_operation_for_test(project_id, secret_id)
        })
        .expect("arm exact project/buffer gate before request construction");
    cx_a.update(|cx| {
        secret.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "QUEUED_PRIVATE_CANARY ")], None, cx)
        });
        public.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "QUEUED_PUBLIC_CANARY ")], None, cx)
        });
    });
    let observed = queued_flusher_within(&executor, ready)
        .await
        .expect("pre-request flusher timeout")
        .expect("pre-request observer");
    assert_eq!(observed.project_id, project_id);
    assert_eq!(observed.buffer_id, secret_id);
    assert!(observed.batch_buffer_ids.contains(&secret_id));
    assert!(
        observed.batch_buffer_ids.contains(&public_id),
        "require actual mixed buffer batch, independent of HashMap iteration order"
    );
    assert!(queued_flusher_has_marker(
        &observed.operations,
        "QUEUED_PRIVATE_CANARY"
    ));
    assert_eq!(
        guest_secret.read_with(cx_b, |buffer, _| buffer.text()),
        "SECRET_BASE",
        "target operation is owned by flusher before Client::request is constructed"
    );

    let private_at_flush = !starts_private;
    queued_flusher_set_private(cx_a, private_at_flush);
    executor.run_until_parked();
    secret.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("current file");
        assert!(!file.is_private);
        assert_eq!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("current local policy owner")
                .is_path_private(rel_path("secret.txt")),
            private_at_flush,
            "observe actual current policy change before releasing the owned operation"
        );
    });
    release
        .send(())
        .expect("flusher still waits before guard/request");
    let outcome = queued_flusher_within(&executor, settled)
        .await
        .expect("flusher settlement timeout")
        .expect("flusher settlement observer");
    assert_eq!(
        outcome.request_constructed, !private_at_flush,
        "only current policy at guarded request creation decides publication"
    );
    assert!(
        !outcome.request_failed,
        "a transport failure is not a private-denial negative"
    );
    let public_version = public.read_with(cx_a, |buffer, _| buffer.version());
    let received_public =
        guest_public.update(cx_b, |buffer, _| buffer.wait_for_version(public_version));
    queued_flusher_within(&executor, received_public)
        .await
        .expect("public continuation timeout")
        .expect("actual guest public operation receipt");
    assert_eq!(
        guest_public.read_with(cx_b, |buffer, _| buffer.text()),
        public.read_with(cx_a, |buffer, _| buffer.text())
    );
    if private_at_flush {
        let received = guest_secret
            .read_with(cx_b, |buffer, cx| buffer.serialize_ops(None, cx))
            .await;
        assert!(
            !queued_flusher_has_marker(&received, "QUEUED_PRIVATE_CANARY"),
            "withheld text must not reach guest history even as a deferred operation"
        );
        assert_eq!(
            guest_secret.read_with(cx_b, |buffer, _| buffer.text()),
            "SECRET_BASE"
        );
        assert_eq!(
            guest_secret.read_with(cx_b, |buffer, _| buffer.version()),
            initial_secret_version
        );
        assert!(
            secret
                .read_with(cx_a, |buffer, _| buffer.text())
                .contains("QUEUED_PRIVATE_CANARY")
        );
    } else {
        let secret_version = secret.read_with(cx_a, |buffer, _| buffer.version());
        let received_secret =
            guest_secret.update(cx_b, |buffer, _| buffer.wait_for_version(secret_version));
        queued_flusher_within(&executor, received_secret)
            .await
            .expect("policy-corrected receipt timeout")
            .expect("actual corrected operation receipt");
        assert_eq!(
            guest_secret.read_with(cx_b, |buffer, _| buffer.text()),
            secret.read_with(cx_a, |buffer, _| buffer.text())
        );
        assert!(
            guest_secret
                .read_with(cx_b, |buffer, _| buffer.text())
                .contains("QUEUED_PRIVATE_CANARY")
        );
        assert!(!guest_secret.read_with(cx_b, |buffer, _| buffer.has_deferred_ops()));
    }
}

#[gpui::test]
async fn test_queued_flusher_private_before_request(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_queued_flusher_current_policy(executor, cx_a, cx_b, false).await;
}
#[gpui::test]
async fn test_queued_flusher_policy_correction_before_request(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_queued_flusher_current_policy(executor, cx_a, cx_b, true).await;
}

async fn check_peer_open_post_publication_scope(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    unshare_before_response: bool,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let host = server.create_client(cx_a, "peer_scope_host").await;
    let guest = server.create_client(cx_b, "peer_scope_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree(
            "/root/peer-scope",
            json!({"public.txt": "PUBLISHED_PUBLIC_CONTROL"}),
        )
        .await;
    let (project_a, worktree_id) = host.build_local_project("/root/peer-scope", cx_a).await;
    let buffer_a = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("host opens public file");
    let buffer_id = buffer_a.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share actual host project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let guest_peer = guest.peer_id().expect("connected guest");
    assert!(!project_a.read_with(cx_a, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .is_buffer_shared_with_peer(guest_peer, buffer_id)
    }));
    let (published, release, settled) = project_a
        .update(cx_a, |project, _| {
            project.hold_peer_open_response_for_test(project_id, guest_peer, buffer_id)
        })
        .expect("arm exact local caller window");
    let client = guest.client().clone();
    let (result_sender, mut result_receiver) = futures::channel::oneshot::channel();
    let request = cx_b.spawn(async move |_| {
        let result = client
            .request(proto::OpenBufferById {
                project_id,
                id: buffer_id.to_proto(),
            })
            .await;
        result_sender
            .send(result)
            .map_err(|_| anyhow!("peer-open result observer dropped"))
    });
    let observed = resync_gate_within(&executor, published)
        .await
        .expect("publication checkpoint timeout")
        .expect("publication checkpoint sender");
    assert_eq!(observed.project_id, project_id);
    assert_eq!(observed.peer_id, guest_peer);
    assert_eq!(observed.buffer_id, buffer_id);
    assert!(project_a.read_with(cx_a, |project, cx| {
        project
            .buffer_store()
            .read(cx)
            .is_buffer_shared_with_peer(guest_peer, buffer_id)
    }));
    assert!(
        result_receiver
            .try_recv()
            .expect("live raw result observer")
            .is_none()
    );
    let guest_buffer = project_b.update(cx_b, |project, cx| {
        project
            .buffer_store()
            .update(cx, |store, cx| store.wait_for_remote_buffer(buffer_id, cx))
    });
    let guest_buffer = resync_gate_within(&executor, guest_buffer)
        .await
        .expect("constructor delivery timeout")
        .expect("actual guest receives constructor/history");
    assert_eq!(
        guest_buffer.read_with(cx_b, |buffer, _| buffer.text()),
        "PUBLISHED_PUBLIC_CONTROL"
    );
    if unshare_before_response {
        project_a
            .update(cx_a, |project, cx| project.unshare(cx))
            .expect("real host unshare after publication and before helper response");
        assert!(!project_a.read_with(cx_a, |project, cx| {
            project
                .buffer_store()
                .read(cx)
                .is_buffer_shared_with_peer(guest_peer, buffer_id)
        }));
    }
    release.send(()).expect("host remains at caller checkpoint");
    let local_outcome = resync_gate_within(&executor, settled)
        .await
        .expect("host outcome timeout")
        .expect("host outcome observer");
    let response = resync_gate_within(&executor, result_receiver)
        .await
        .expect("raw peer-open completion timeout")
        .expect("raw peer-open result sender");
    request.await.expect("raw result transfer");
    let guest_text = guest_buffer.read_with(cx_b, |buffer, _| buffer.text());
    let host_hang_up = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.hang_up(cx));
    let guest_hang_up = cx_b
        .read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx));
    resync_gate_within(&executor, host_hang_up)
        .await
        .expect("host fixture hang-up timeout")
        .expect("host leaves fixture call");
    resync_gate_within(&executor, guest_hang_up)
        .await
        .expect("guest fixture hang-up timeout")
        .expect("guest leaves fixture call");
    cx_a.run_until_parked();
    cx_b.run_until_parked();

    if unshare_before_response {
        assert_eq!(
            local_outcome,
            Err(proto::ErrorCode::Disconnected),
            "local helper itself must reject stale success; server authorization failure alone is insufficient"
        );
    } else {
        assert_eq!(local_outcome, Ok(buffer_id.to_proto()));
    }
    if unshare_before_response {
        assert!(response.is_err(), "unshared request cannot return success");
    } else {
        assert_eq!(
            response.expect("ordinary public open succeeds").buffer_id,
            buffer_id.to_proto()
        );
    }
    assert_eq!(
        guest_text, "PUBLISHED_PUBLIC_CONTROL",
        "scope refusal is not revocation of previously delivered public content"
    );
}

#[gpui::test]
async fn test_peer_open_post_publication_unshare_rejects_local_success(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_open_post_publication_scope(executor, cx_a, cx_b, true).await;
}

#[gpui::test]
async fn test_peer_open_post_publication_public_success_control(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_open_post_publication_scope(executor, cx_a, cx_b, false).await;
}

async fn check_resync_cancellation(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cancel_owner: bool,
) {
    let mut server = TestServer::start_postgres(executor.clone()).await;
    let host = server.create_client(cx_a, "resync_host").await;
    let guest = server.create_client(cx_b, "resync_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree(
            "/root",
            json!({"resync-gate": {
                "target.txt": "TARGET_BASE", "prior.txt": "PRIOR_PUBLIC_BASE"
            }}),
        )
        .await;
    let (project_a, worktree_id) = host.build_local_project("/root/resync-gate", cx_a).await;
    let target = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("target.txt")), cx)
        })
        .await
        .expect("host target");
    let target_id = target.read_with(cx_a, |buffer, _| buffer.remote_id());
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share real project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let guest_target = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(target_id, cx))
        .await
        .expect("guest has a complete target base");
    let guest_peer = guest.peer_id().expect("connected guest");
    let host_peer = host.peer_id().expect("connected host");
    let store = project_a.read_with(cx_a, |project, _| project.buffer_store().clone());
    assert!(store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    store.update(cx_a, |store, _| {
        store.forget_shared_buffers_for(&guest_peer)
    });
    // Actual lifecycle forget models a retained guest buffer with absent constructor bookkeeping.
    let prior = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("prior.txt")), cx)
        })
        .await
        .expect("host public sibling");
    let prior_id = prior.read_with(cx_a, |buffer, _| buffer.remote_id());
    let guest_prior = project_b
        .update(cx_b, |project, cx| project.open_buffer_by_id(prior_id, cx))
        .await
        .expect("guest admits the public sibling after lifecycle forget");
    assert!(!store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    assert!(store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, prior_id)
    }));

    resync_gate_set_private(cx_a, true);
    executor.run_until_parked();
    target.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("worktree target");
        assert!(
            !file.is_private,
            "excluded scan must keep stale public metadata"
        );
        assert!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local owner")
                .is_path_private(rel_path("target.txt"))
        );
    });
    for index in 0..12 {
        let text = format!("WITHHELD_{index} ");
        target.update(cx_a, |buffer, cx| {
            buffer.edit([(0..0, text.as_str())], None, cx)
        });
    }
    target.update(cx_a, |buffer, cx| {
        buffer.edit([(0..0, "RESYNC_BLOCKED_TAIL_CANARY ")], None, cx)
    });
    executor.run_until_parked();
    assert_eq!(
        guest_target.read_with(cx_b, |buffer, _| buffer.text()),
        "TARGET_BASE",
        "ordinary egress must have been denied while preparing withheld history"
    );
    resync_gate_set_private(cx_a, false);
    executor.run_until_parked();

    let (prepared, mut release_prepared) = store
        .update(cx_a, |store, _| {
            store.hold_synchronization_preparation_for_test(
                project_id,
                guest_peer,
                target_id.to_proto(),
            )
        })
        .expect("arm real owner preparation checkpoint");
    let mut committed = store
        .update(cx_a, |store, _| {
            store.observe_synchronization_commit_for_test(project_id, guest_peer)
        })
        .expect("arm exact local commit observation");
    let (mut ack_observed, release_ack) = server
        .hold_update_buffer_ack(host_peer, project_id, target_id.to_proto())
        .expect("activate exact existing transport recorder");
    release_ack
        .send(Ok(()))
        .expect("pre-authorize ordinary Ack, without injecting a failure");
    let metadata_observed = server
        .observe_buffer_metadata_pair(host_peer, project_id, target_id.to_proto())
        .expect("actual metadata handlers have an independent completion barrier");
    let payload = proto::SynchronizeBuffers {
        project_id,
        buffers: vec![proto::BufferVersion {
            id: target_id.to_proto(),
            version: Vec::new(),
        }],
    };
    let (result_sender, mut result_receiver) = futures::channel::oneshot::channel();
    let synchronization = if cancel_owner {
        let owner = project_a.update(cx_a, |project, cx| {
            project.synchronize_buffers_for_test(payload, guest_peer, cx)
        });
        cx_a.spawn(async move |_| {
            result_sender
                .send(owner.await)
                .map_err(|_| anyhow!("local result observer dropped"))
        })
    } else {
        let client = guest.client().clone();
        cx_b.spawn(async move |_| {
            result_sender
                .send(client.request(payload).await)
                .map_err(|_| anyhow!("requester result observer dropped"))
        })
    };
    let checkpoint = resync_gate_within(&executor, prepared)
        .await
        .expect("preparation checkpoint timeout")
        .expect("real preparation observed");
    assert_eq!(checkpoint.version.id, target_id.to_proto());
    assert!(checkpoint.chunks.len() >= 3);
    assert!(checkpoint.chunks.last().expect("final chunk").is_empty());
    assert!(
        checkpoint
            .chunks
            .iter()
            .skip(1)
            .any(|chunk| { resync_gate_has_text(chunk, "RESYNC_BLOCKED_TAIL_CANARY") })
    );
    assert!(store.read_with(cx_a, |store, _| {
        store.has_live_synchronization_for_test(guest_peer)
    }));
    assert!(!store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    }));
    assert!(
        result_receiver
            .try_recv()
            .expect("live pre-cancellation result observer")
            .is_none()
    );
    assert!(server.recorded_update_buffers().is_empty());
    assert!(server.recorded_buffer_metadata().is_empty());

    drop(synchronization);
    let (cancellation_observed, interim_pending, interim_admitted, final_response) = if cancel_owner
    {
        resync_gate_within(&executor, release_prepared.cancellation())
            .await
            .expect("dropping actual owner must drop the held serialization receiver");
        executor.run_until_parked();
        assert!(
            result_receiver.try_recv().is_err(),
            "local wrapper's result sender was dropped"
        );
        let pending = store.read_with(cx_a, |store, _| {
            store.has_live_synchronization_for_test(guest_peer)
        });
        let admitted = store.read_with(cx_a, |store, _| {
            store.is_buffer_shared_with_peer(guest_peer, target_id)
        });
        assert!(
            !pending && !admitted,
            "cancelled preparation cannot retain a live token or create an admission"
        );
        assert!(
            committed
                .try_recv()
                .expect("commit observer remains armed")
                .is_none()
        );
        assert!(server.recorded_update_buffers().is_empty());
        assert!(server.recorded_buffer_metadata().is_empty());
        assert!(
            ack_observed
                .try_recv()
                .expect("unused actual Ack observer remains armed")
                .is_none()
        );
        assert_eq!(
            guest_target.read_with(cx_b, |buffer, _| buffer.text()),
            "TARGET_BASE"
        );
        let response = resync_gate_within(
            &executor,
            guest.client().request(proto::SynchronizeBuffers {
                project_id,
                buffers: vec![proto::BufferVersion {
                    id: target_id.to_proto(),
                    version: Vec::new(),
                }],
            }),
        )
        .await
        .expect("fresh post-cancellation request timeout")
        .expect("fresh actual request succeeds");
        let local_commit = resync_gate_within(&executor, committed)
            .await
            .expect("fresh local commit timeout")
            .expect("fresh local commit observed");
        assert_eq!(response.buffers, local_commit.buffers);
        (true, pending, admitted, local_commit)
    } else {
        executor.run_until_parked();
        assert!(
            result_receiver.try_recv().is_err(),
            "guest requester was locally cancelled"
        );
        let receiver_cancelled = release_prepared.is_canceled();
        let pending = store.read_with(cx_a, |store, _| {
            store.has_live_synchronization_for_test(guest_peer)
        });
        let admitted = store.read_with(cx_a, |store, _| {
            store.is_buffer_shared_with_peer(guest_peer, target_id)
        });
        assert!(
            !receiver_cancelled && pending && !admitted,
            "requester drop is not a remote host cancellation message"
        );
        release_prepared
            .send(())
            .expect("the host still owns its preparation receiver");
        let local_commit = resync_gate_within(&executor, committed)
            .await
            .expect("detached actual host completion timeout")
            .expect("actual host commit observed");
        (receiver_cancelled, pending, admitted, local_commit)
    };
    resync_gate_within(&executor, metadata_observed)
        .await
        .expect("authorized metadata completion timeout")
        .expect("metadata pair observed");
    let first_update = resync_gate_within(&executor, &mut ack_observed)
        .await
        .expect("actual first UpdateBuffer handler timeout")
        .expect("actual UpdateBuffer observed");
    let checkpoint_global = language::proto::deserialize_version(&checkpoint.version.version);
    let guest_received = guest_target.update(cx_b, |buffer, _| {
        buffer.wait_for_version(checkpoint_global.clone())
    });
    resync_gate_within(&executor, guest_received)
        .await
        .expect("actual guest version delivery timeout")
        .expect("actual guest received the captured operation version");
    executor.run_until_parked();
    let final_pending = store.read_with(cx_a, |store, _| {
        store.has_live_synchronization_for_test(guest_peer)
    });
    let final_target_admitted = store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, target_id)
    });
    let final_prior_admitted = store.read_with(cx_a, |store, _| {
        store.is_buffer_shared_with_peer(guest_peer, prior_id)
    });
    let final_target = guest_target.read_with(cx_b, |buffer, _| {
        (buffer.text(), buffer.has_deferred_ops(), buffer.version())
    });
    let host_target = target.read_with(cx_a, |buffer, _| (buffer.text(), buffer.version()));
    let final_prior = guest_prior.read_with(cx_b, |buffer, _| buffer.text());
    let records = server.recorded_update_buffers();
    let all_chunk_identities_match = records.iter().all(|record| {
        record.sender_id == host_peer
            && record.request.project_id == project_id
            && record.request.buffer_id == target_id.to_proto()
    });
    let chunks: Vec<_> = records
        .into_iter()
        .map(|record| record.request.operations)
        .collect();
    let metadata = server.recorded_buffer_metadata();
    let host_hang_up = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.hang_up(cx));
    let guest_hang_up = cx_b
        .read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx));
    resync_gate_within(&executor, host_hang_up)
        .await
        .expect("host fixture hang-up timeout")
        .expect("host leaves fixture call");
    resync_gate_within(&executor, guest_hang_up)
        .await
        .expect("guest fixture hang-up timeout")
        .expect("guest leaves fixture call");
    cx_a.update(|_| drop((target, prior, store, project_a)));
    cx_b.update(|_| drop((guest_target, guest_prior, project_b)));
    cx_a.run_until_parked();
    cx_b.run_until_parked();
    drop(host);
    drop(guest);
    cx_a.run_until_parked();
    cx_b.run_until_parked();
    assert_eq!(cancellation_observed, cancel_owner);
    assert_eq!(interim_pending, !cancel_owner);
    assert!(!interim_admitted);
    assert!(!final_pending);
    assert!(final_target_admitted && final_prior_admitted);
    assert_eq!(final_response.buffers, vec![checkpoint.version]);
    assert_eq!(first_update.sender_id, host_peer);
    assert_eq!(first_update.request.project_id, project_id);
    assert_eq!(first_update.request.buffer_id, target_id.to_proto());
    assert_eq!(
        first_update.request.operations,
        checkpoint
            .chunks
            .first()
            .expect("captured first chunk")
            .clone()
    );
    assert!(
        all_chunk_identities_match,
        "every UpdateBuffer envelope, including the final empty chunk, targets the exact host/project/buffer"
    );
    assert_eq!(
        chunks, checkpoint.chunks,
        "all captured chunks and final empty chunk traverse actual transport"
    );
    assert_resync_metadata_pair(&metadata, project_id, target_id.to_proto());
    assert_eq!(final_target.0, host_target.0);
    assert_eq!(
        final_target.2, checkpoint_global,
        "actual guest Global version equals the captured serialization checkpoint"
    );
    assert_eq!(
        host_target.1, checkpoint_global,
        "host Global version remains the exact captured checkpoint"
    );
    assert!(!final_target.1);
    assert_eq!(final_prior, "PRIOR_PUBLIC_BASE");
}

#[gpui::test]
async fn test_resync_owner_task_drop_releases_preparation_without_admission(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_cancellation(executor, cx_a, cx_b, true).await;
}

#[gpui::test]
async fn test_resync_requester_drop_does_not_cancel_host_owner(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_resync_cancellation(executor, cx_a, cx_b, false).await;
}

#[derive(Clone, Copy)]
enum PeerIngressMutation {
    Edit,
    Save,
    SaveAs(&'static str),
}

async fn check_peer_ingress_retained_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    mutation: PeerIngressMutation,
    private_source: bool,
    private_destination: bool,
    explicit_opt_in: bool,
) {
    use rpc::ErrorExt as _;

    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(cx_a, "ingress_host").await;
    let guest = server.create_client(cx_b, "ingress_guest").await;
    server
        .create_room(&mut [(&host, cx_a), (&guest, cx_b)])
        .await;
    host.fs()
        .insert_tree(
            "/root/peer-ingress",
            json!({
                "secret.txt": "INGRESS_BASE",
                "public.txt": "INGRESS_PUBLIC",
                "existing-destination.txt": "DO_NOT_OVERWRITE"
            }),
        )
        .await;
    let (project_a, worktree_id) = host.build_local_project("/root/peer-ingress", cx_a).await;
    let host_buffer = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("host opens initially public source");
    let project_id = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share actual project");
    let project_b = guest.join_remote_project(project_id, cx_b).await;
    let guest_buffer = project_b
        .update(cx_b, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("guest retains source before policy transition");
    assert_eq!(
        guest_buffer.read_with(cx_b, |buffer, _| buffer.text()),
        "INGRESS_BASE"
    );

    let mut edit_operations = Vec::new();
    if matches!(mutation, PeerIngressMutation::Edit) {
        let (state, replica_id, capability, version, original_text) =
            guest_buffer.read_with(cx_b, |buffer, cx| {
                (
                    buffer.to_proto(cx),
                    buffer.replica_id(),
                    buffer.capability(),
                    buffer.version(),
                    buffer.text(),
                )
            });
        let history = guest_buffer
            .read_with(cx_b, |buffer, cx| buffer.serialize_ops(None, cx))
            .await;
        let detached = cx_b.new(|_| {
            language::Buffer::from_proto(replica_id, capability, state, None)
                .expect("detached genuine guest replica")
        });
        let history = history
            .into_iter()
            .map(language::proto::deserialize_operation)
            .collect::<Result<Vec<_>>>()
            .expect("actual base history deserializes");
        detached.update(cx_b, |buffer, cx| {
            buffer.apply_ops(history, cx);
            assert_eq!(buffer.version(), version);
            assert_eq!(buffer.text(), original_text);
            buffer.edit([(0..0, "GUEST_INGRESS_EDIT ")], None, cx);
        });
        edit_operations = detached
            .read_with(cx_b, |buffer, cx| buffer.serialize_ops(Some(version), cx))
            .await;
        assert!(queued_flusher_has_marker(
            &edit_operations,
            "GUEST_INGRESS_EDIT"
        ));
    } else {
        host_buffer.update(cx_a, |buffer, cx| {
            buffer.edit([(0..0, "HOST_UNSAVED ")], None, cx)
        });
    }
    let destination = match mutation {
        PeerIngressMutation::SaveAs(path) => Some(path),
        _ => None,
    };
    let patterns = [
        private_source.then_some("**/secret.txt"),
        destination
            .filter(|_| private_destination)
            .map(|_| "**/*destination.txt"),
    ]
    .into_iter()
    .flatten()
    .map(str::to_string)
    .collect::<Vec<_>>();
    cx_a.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(patterns.into());
            });
        });
    });
    executor.run_until_parked();
    if explicit_opt_in {
        let worktree = host_buffer.read_with(cx_a, |buffer, _| {
            worktree::File::from_dyn(buffer.file())
                .expect("local file")
                .worktree
                .clone()
        });
        worktree.update(cx_a, |worktree, cx| {
            worktree
                .as_local_mut()
                .expect("local owner")
                .share_private_files(cx);
        });
        executor.run_until_parked();
    }
    host_buffer.read_with(cx_a, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("local source");
        assert!(!file.is_private, "require stale public File flag");
        let local = file
            .worktree
            .read(cx)
            .as_local()
            .expect("local policy owner");
        assert_eq!(
            local.is_path_private(rel_path("secret.txt")),
            private_source && !explicit_opt_in
        );
        if let Some(destination) = destination {
            assert_eq!(
                local.is_path_private(rel_path(destination)),
                private_destination && !explicit_opt_in
            );
        }
    });
    let before = host_buffer.read_with(cx_a, |buffer, cx| {
        (
            buffer.text(),
            buffer.version(),
            buffer.saved_version().clone(),
            buffer.saved_mtime(),
            buffer.is_dirty(),
            buffer.file().map(|file| file.to_proto(cx)),
        )
    });
    let source_path = Path::new("/root/peer-ingress/secret.txt");
    let disk_before = host.fs().load(source_path).await.expect("source on disk");
    let metadata_snapshot = |metadata: fs::Metadata| {
        (
            metadata.inode,
            metadata.mtime,
            metadata.is_symlink,
            metadata.is_dir,
            metadata.len,
            metadata.is_fifo,
            metadata.is_executable,
            metadata.is_writable,
        )
    };
    let source_metadata_before = host
        .fs()
        .metadata(source_path)
        .await
        .expect("source metadata before request")
        .map(metadata_snapshot);
    let destination_path =
        destination.map(|destination| Path::new("/root/peer-ingress").join(destination));
    let destination_metadata_before = if let Some(path) = &destination_path {
        host.fs()
            .metadata(path)
            .await
            .expect("destination metadata before request")
            .map(metadata_snapshot)
    } else {
        None
    };
    let destination_before = if let Some(path) = &destination_path {
        if host
            .fs()
            .metadata(path)
            .await
            .expect("destination metadata")
            .is_some()
        {
            Some(host.fs().load(path).await.expect("existing destination"))
        } else {
            None
        }
    } else {
        None
    };
    let buffer_id = host_buffer.read_with(cx_a, |buffer, _| buffer.remote_id().to_proto());
    let outcome = match mutation {
        PeerIngressMutation::Edit => resync_gate_within(
            &executor,
            guest.request(proto::UpdateBuffer {
                project_id,
                buffer_id,
                operations: edit_operations,
            }),
        )
        .await
        .expect("actual guest edit deadline")
        .map(|_| ()),
        PeerIngressMutation::Save | PeerIngressMutation::SaveAs(_) => resync_gate_within(
            &executor,
            guest.request(proto::SaveBuffer {
                project_id,
                buffer_id,
                version: language::proto::serialize_version(&before.1),
                new_path: destination.map(|path| proto::ProjectPath {
                    worktree_id: worktree_id.to_proto(),
                    path: path.into(),
                }),
            }),
        )
        .await
        .expect("actual guest save deadline")
        .map(|_| ()),
    }
    .map_err(|error| error.error_code());
    executor.run_until_parked();
    let after = host_buffer.read_with(cx_a, |buffer, cx| {
        (
            buffer.text(),
            buffer.version(),
            buffer.saved_version().clone(),
            buffer.saved_mtime(),
            buffer.is_dirty(),
            buffer.file().map(|file| file.to_proto(cx)),
        )
    });
    let disk_after = host
        .fs()
        .load(source_path)
        .await
        .expect("source remains on disk");
    let source_metadata_after = host
        .fs()
        .metadata(source_path)
        .await
        .expect("source metadata after request")
        .map(metadata_snapshot);
    let destination_metadata_after = if let Some(path) = &destination_path {
        host.fs()
            .metadata(path)
            .await
            .expect("destination metadata after request")
            .map(metadata_snapshot)
    } else {
        None
    };
    let destination_after = if let Some(path) = &destination_path {
        if host
            .fs()
            .metadata(path)
            .await
            .expect("result destination metadata")
            .is_some()
        {
            Some(
                host.fs()
                    .load(path)
                    .await
                    .expect("result destination bytes"),
            )
        } else {
            None
        }
    } else {
        None
    };
    let host_hang_up = cx_a
        .read(ActiveCall::global)
        .update(cx_a, |call, cx| call.hang_up(cx));
    let guest_hang_up = cx_b
        .read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx));
    resync_gate_within(&executor, host_hang_up)
        .await
        .expect("host cleanup deadline")
        .expect("host cleanup");
    resync_gate_within(&executor, guest_hang_up)
        .await
        .expect("guest cleanup deadline")
        .expect("guest cleanup");
    cx_a.run_until_parked();
    cx_b.run_until_parked();
    if (private_source || private_destination) && !explicit_opt_in {
        assert_eq!(
            outcome,
            Err(proto::ErrorCode::UnsharedItem),
            "current host privacy must reject a retained guest mutation"
        );
        assert_eq!(
            after, before,
            "denied ingress must preserve host text/version/saved state/file identity"
        );
        assert_eq!(
            disk_after, disk_before,
            "denied ingress must not save host content"
        );
        assert_eq!(
            destination_after, destination_before,
            "denied SaveAs must neither create nor overwrite destination"
        );
        assert_eq!(
            source_metadata_after, source_metadata_before,
            "denied ingress must preserve source filesystem metadata"
        );
        assert_eq!(
            destination_metadata_after, destination_metadata_before,
            "denied SaveAs must preserve destination filesystem metadata"
        );
    } else {
        assert_eq!(outcome, Ok(()), "ordinary public ingress must succeed");
        match mutation {
            PeerIngressMutation::Edit => {
                assert_eq!(after.0, "GUEST_INGRESS_EDIT INGRESS_BASE");
                assert_ne!(after.1, before.1);
                assert_eq!(disk_after, disk_before);
            }
            PeerIngressMutation::Save => {
                assert_eq!(disk_after, "HOST_UNSAVED INGRESS_BASE");
                assert_eq!(after.0, before.0);
                assert_eq!(after.2, before.1);
                assert!(!after.4);
            }
            PeerIngressMutation::SaveAs(_) => {
                assert_eq!(destination_after, Some(before.0));
                assert_eq!(disk_after, disk_before);
                assert_eq!(after.1, before.1);
                assert_eq!(after.2, before.1);
                assert!(!after.4);
                assert_eq!(
                    after.5.expect("saved public destination file").path,
                    destination.expect("SaveAs path")
                );
            }
        }
    }
}

#[gpui::test]
async fn test_peer_edit_current_private_retained_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::Edit,
        true,
        false,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_edit_current_public_control(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::Edit,
        false,
        false,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_current_private_retained_buffer(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::Save,
        true,
        false,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_current_public_control(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::Save,
        false,
        false,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_current_private_destination(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::SaveAs("existing-destination.txt"),
        false,
        true,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_current_private_new_destination(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::SaveAs("new-destination.txt"),
        false,
        true,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_private_source_save_as_public(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::SaveAs("public-destination.txt"),
        true,
        false,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_current_public_control(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        cx_a,
        cx_b,
        PeerIngressMutation::SaveAs("public-destination.txt"),
        false,
        false,
        false,
    )
    .await;
}

#[derive(Clone, Copy, PartialEq)]
enum PeerUpdateRevocation {
    GuestLeavesRoom,
    HostLeavesRoom,
    GuestLeavesProjectBeforeRequest,
    GuestLeavesProjectAfterAcceptance,
}

async fn check_peer_edit_observer_publication(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
    private: bool,
    revocation: Option<PeerUpdateRevocation>,
) {
    use rpc::ErrorExt as _;
    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(host_cx, "acceptance_host").await;
    let guest = server.create_client(guest_cx, "acceptance_guest").await;
    let observer = server
        .create_client(observer_cx, "acceptance_observer")
        .await;
    server
        .create_room(&mut [
            (&host, host_cx),
            (&guest, guest_cx),
            (&observer, observer_cx),
        ])
        .await;
    host.fs()
        .insert_tree("/root/peer-acceptance", json!({"secret.txt":"BASE"}))
        .await;
    let (host_project, worktree_id) = host
        .build_local_project("/root/peer-acceptance", host_cx)
        .await;
    let host_buffer = host_project
        .update(host_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("host source");
    let project_id = host_cx
        .read(ActiveCall::global)
        .update(host_cx, |call, cx| {
            call.share_project(host_project.clone(), cx)
        })
        .await
        .expect("shared project");
    let guest_project = guest.join_remote_project(project_id, guest_cx).await;
    let observer_project = observer.join_remote_project(project_id, observer_cx).await;
    let guest_buffer = guest_project
        .update(guest_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("guest source");
    let observer_buffer = observer_project
        .update(observer_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("secret.txt")), cx)
        })
        .await
        .expect("observer source");
    let (state, replica_id, capability, version) =
        guest_buffer.read_with(guest_cx, |buffer, cx| {
            (
                buffer.to_proto(cx),
                buffer.replica_id(),
                buffer.capability(),
                buffer.version(),
            )
        });
    let history = guest_buffer
        .read_with(guest_cx, |buffer, cx| buffer.serialize_ops(None, cx))
        .await;
    let detached = guest_cx.new(|_| {
        language::Buffer::from_proto(replica_id, capability, state, None).expect("guest replica")
    });
    let history = history
        .into_iter()
        .map(language::proto::deserialize_operation)
        .collect::<Result<Vec<_>>>()
        .expect("base operations");
    detached.update(guest_cx, |buffer, cx| {
        buffer.apply_ops(history, cx);
        assert_eq!(buffer.version(), version);
        buffer.edit([(0..0, "REJECTED_PEER_EDIT ")], None, cx);
    });
    let operations = detached
        .read_with(guest_cx, |buffer, cx| {
            buffer.serialize_ops(Some(version), cx)
        })
        .await;
    assert!(queued_flusher_has_marker(&operations, "REJECTED_PEER_EDIT"));
    host_cx.update(|cx| {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                settings.project.worktree.private_files = Some(
                    if private {
                        vec!["**/secret.txt".to_owned()]
                    } else {
                        Vec::new()
                    }
                    .into(),
                );
            })
        })
    });
    executor.run_until_parked();
    host_buffer.read_with(host_cx, |buffer, cx| {
        let file = worktree::File::from_dyn(buffer.file()).expect("host file");
        assert!(
            !file.is_private,
            "retained public metadata is deliberately stale"
        );
        assert_eq!(
            file.worktree
                .read(cx)
                .as_local()
                .expect("local policy")
                .is_path_private(rel_path("secret.txt")),
            private
        );
    });
    let buffer_id = host_buffer.read_with(host_cx, |buffer, _| buffer.remote_id().to_proto());
    let request = proto::UpdateBuffer {
        project_id,
        buffer_id,
        operations,
    };
    if revocation == Some(PeerUpdateRevocation::GuestLeavesProjectBeforeRequest) {
        guest
            .send(proto::LeaveProject { project_id })
            .expect("leave only the project");
        resync_gate_within(&executor, async {
            while host_project.read_with(host_cx, |project, _| {
                project
                    .collaborators()
                    .contains_key(&guest.peer_id().expect("guest connection"))
            }) {
                executor.timer(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("project departure deadline");
        assert!(guest_cx.read(|cx| ActiveCall::global(cx).read(cx).room().is_some()));
    }
    let outcome = if let Some(revocation) = revocation
        .filter(|revocation| *revocation != PeerUpdateRevocation::GuestLeavesProjectBeforeRequest)
    {
        let (observed, release) = server
            .hold_peer_update_after_acceptance(
                host.peer_id().expect("host connection"),
                project_id,
                buffer_id,
            )
            .expect("server acceptance gate");
        let pending = executor.spawn(guest.request(request));
        let observed = resync_gate_within(&executor, observed)
            .await
            .expect("host acceptance deadline")
            .expect("actual accepted request");
        assert_eq!(observed.request.buffer_id, buffer_id);
        assert_eq!(
            observed.sender_id,
            guest.peer_id().expect("guest connection")
        );
        assert_eq!(
            host_buffer.read_with(host_cx, |buffer, _| buffer.text()),
            "REJECTED_PEER_EDIT BASE"
        );
        match revocation {
            PeerUpdateRevocation::GuestLeavesProjectAfterAcceptance => {
                guest
                    .send(proto::LeaveProject { project_id })
                    .expect("leave only the project");
                resync_gate_within(&executor, async {
                    while host_project.read_with(host_cx, |project, _| {
                        project
                            .collaborators()
                            .contains_key(&guest.peer_id().expect("guest connection"))
                    }) {
                        executor.timer(Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("project departure deadline");
                assert!(guest_cx.read(|cx| ActiveCall::global(cx).read(cx).room().is_some()));
            }
            PeerUpdateRevocation::HostLeavesRoom | PeerUpdateRevocation::GuestLeavesRoom => {
                let leave = if revocation == PeerUpdateRevocation::HostLeavesRoom {
                    host_cx
                        .read(ActiveCall::global)
                        .update(host_cx, |call, cx| call.hang_up(cx))
                } else {
                    guest_cx
                        .read(ActiveCall::global)
                        .update(guest_cx, |call, cx| call.hang_up(cx))
                };
                resync_gate_within(&executor, leave)
                    .await
                    .expect("membership change deadline")
                    .expect("leave before publication");
            }
            PeerUpdateRevocation::GuestLeavesProjectBeforeRequest => {
                unreachable!("handled before request")
            }
        }
        executor.run_until_parked();
        release
            .send(Ok(()))
            .expect("release actual accepted request");
        resync_gate_within(&executor, pending)
            .await
            .expect("update deadline")
    } else {
        resync_gate_within(&executor, guest.request(request))
            .await
            .expect("update deadline")
    }
    .map(|_| ())
    .map_err(|error| error.error_code());
    executor.run_until_parked();
    let host_text = host_buffer.read_with(host_cx, |buffer, _| buffer.text());
    let observer_text = observer_buffer.read_with(observer_cx, |buffer, _| buffer.text());
    let mut cleanup = Vec::new();
    if revocation != Some(PeerUpdateRevocation::HostLeavesRoom) {
        cleanup.push(
            host_cx
                .read(ActiveCall::global)
                .update(host_cx, |call, cx| call.hang_up(cx)),
        );
    }
    if revocation != Some(PeerUpdateRevocation::GuestLeavesRoom) {
        cleanup.push(
            guest_cx
                .read(ActiveCall::global)
                .update(guest_cx, |call, cx| call.hang_up(cx)),
        );
    }
    cleanup.push(
        observer_cx
            .read(ActiveCall::global)
            .update(observer_cx, |call, cx| call.hang_up(cx)),
    );
    for hang_up in cleanup {
        resync_gate_within(&executor, hang_up)
            .await
            .expect("cleanup deadline")
            .expect("leave call");
    }
    executor.run_until_parked();
    if revocation.is_some() {
        assert!(
            outcome.is_err(),
            "lost authority must not produce a successful Ack"
        );
        assert_eq!(
            host_text,
            if revocation == Some(PeerUpdateRevocation::GuestLeavesProjectBeforeRequest) {
                "BASE"
            } else {
                "REJECTED_PEER_EDIT BASE"
            },
            "pre-admission departure rejects mutation; accepted edits are not rolled back"
        );
        assert_eq!(
            observer_text, "BASE",
            "revoked authority must not reach observers"
        );
    } else if private {
        assert_eq!(outcome, Err(proto::ErrorCode::UnsharedItem));
        assert_eq!(host_text, "BASE", "host must reject the mutation");
        assert_eq!(
            observer_text, "BASE",
            "host-rejected operations must never reach another guest"
        );
    } else {
        assert_eq!(outcome, Ok(()));
        assert_eq!(host_text, "REJECTED_PEER_EDIT BASE");
        assert_eq!(
            observer_text, host_text,
            "accepted public edits still converge"
        );
    }
}

#[gpui::test]
async fn test_peer_edit_observer_rejects_host_denied_operation(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(executor, host_cx, guest_cx, observer_cx, true, None)
        .await;
}

#[gpui::test]
async fn test_peer_edit_observer_public_control(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(executor, host_cx, guest_cx, observer_cx, false, None)
        .await;
}

#[gpui::test]
async fn test_peer_edit_observer_membership_lost_after_acceptance(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(
        executor,
        host_cx,
        guest_cx,
        observer_cx,
        false,
        Some(PeerUpdateRevocation::GuestLeavesRoom),
    )
    .await;
}

#[gpui::test]
async fn test_peer_edit_observer_host_lost_after_acceptance(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(
        executor,
        host_cx,
        guest_cx,
        observer_cx,
        false,
        Some(PeerUpdateRevocation::HostLeavesRoom),
    )
    .await;
}

#[gpui::test]
async fn test_peer_edit_observer_project_membership_required_before_request(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(
        executor,
        host_cx,
        guest_cx,
        observer_cx,
        false,
        Some(PeerUpdateRevocation::GuestLeavesProjectBeforeRequest),
    )
    .await;
}

#[gpui::test]
async fn test_peer_edit_observer_project_membership_lost_after_acceptance(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    observer_cx: &mut TestAppContext,
) {
    check_peer_edit_observer_publication(
        executor,
        host_cx,
        guest_cx,
        observer_cx,
        false,
        Some(PeerUpdateRevocation::GuestLeavesProjectAfterAcceptance),
    )
    .await;
}

#[derive(Clone, Copy, PartialEq)]
enum PeerIngressBufferState {
    Unknown,
    Queued,
    Expired,
}

async fn check_peer_ingress_unknown_buffer(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    buffer_state: PeerIngressBufferState,
) {
    use rpc::ErrorExt as _;
    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(host_cx, "unknown_host").await;
    let guest = server.create_client(guest_cx, "unknown_guest").await;
    server
        .create_room(&mut [(&host, host_cx), (&guest, guest_cx)])
        .await;
    host.fs()
        .insert_tree("/root/unknown-ingress", json!({"public.txt":"BASE"}))
        .await;
    let (project, worktree_id) = host
        .build_local_project("/root/unknown-ingress", host_cx)
        .await;
    let host_buffer = project
        .update(host_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("host source");
    let project_id = host_cx
        .read(ActiveCall::global)
        .update(host_cx, |call, cx| call.share_project(project.clone(), cx))
        .await
        .expect("shared project");
    let guest_project = guest.join_remote_project(project_id, guest_cx).await;
    let guest_buffer = guest_project
        .update(guest_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("public.txt")), cx)
        })
        .await
        .expect("guest source");
    guest_buffer.update(guest_cx, |buffer, cx| {
        buffer.edit([(0..0, "PUBLIC_CONTROL ")], None, cx)
    });
    executor.run_until_parked();
    assert_eq!(
        host_buffer.read_with(host_cx, |buffer, _| buffer.text()),
        "PUBLIC_CONTROL BASE"
    );
    let operations = guest_buffer
        .read_with(guest_cx, |buffer, cx| buffer.serialize_ops(None, cx))
        .await;
    assert!(!operations.is_empty(), "genuine peer operations");
    let store = project.read_with(host_cx, |project, _| project.buffer_store().clone());
    let unknown_id = if buffer_state == PeerIngressBufferState::Expired {
        let expired = project.update(host_cx, |project, cx| {
            project.create_local_buffer("EXPIRED_OWNER_BUFFER", None, false, cx)
        });
        let buffer_id = expired.read_with(host_cx, |buffer, _| buffer.remote_id());
        let weak = expired.downgrade();
        drop(expired);
        host_cx.update(|_| {});
        executor.run_until_parked();
        assert!(weak.upgrade().is_none(), "real Complete buffer has expired");
        assert!(
            store.read_with(host_cx, |store, _| {
                store.has_buffer_or_pending_operations(buffer_id)
            }),
            "retain the expired Complete entry"
        );
        buffer_id
    } else {
        let buffer_id = text::BufferId::new(900_000_000).expect("unused fixture ID");
        assert!(!store.read_with(host_cx, |store, _| {
            store.has_buffer_or_pending_operations(buffer_id)
        }));
        buffer_id
    };
    if buffer_state == PeerIngressBufferState::Queued {
        let envelope = rpc::TypedEnvelope {
            sender_id: host.peer_id().expect("host connection"),
            original_sender_id: None,
            message_id: 0,
            received_at: std::time::Instant::now(),
            payload: proto::UpdateBuffer {
                project_id,
                buffer_id: unknown_id.to_proto(),
                operations: Vec::new(),
            },
        };
        resync_gate_within(
            &executor,
            project::buffer_store::BufferStore::handle_update_buffer_from_owner(
                store.clone(),
                envelope,
                host_cx.to_async(),
            ),
        )
        .await
        .expect("owner queue deadline")
        .expect("owner initialization path remains available");
    }
    let before = store.read_with(host_cx, |store, _| {
        store.queued_operation_count_for_test(unknown_id)
    });
    assert_eq!(
        before,
        (buffer_state == PeerIngressBufferState::Queued).then_some(0)
    );
    let outcome = resync_gate_within(
        &executor,
        guest.request(proto::UpdateBuffer {
            project_id,
            buffer_id: unknown_id.to_proto(),
            operations,
        }),
    )
    .await
    .expect("guest update deadline")
    .map(|_| ())
    .map_err(|error| error.error_code());
    executor.run_until_parked();
    let after = store.read_with(host_cx, |store, _| {
        store.queued_operation_count_for_test(unknown_id)
    });
    let source = host_buffer.read_with(host_cx, |buffer, _| buffer.text());
    let host_leave = host_cx
        .read(ActiveCall::global)
        .update(host_cx, |call, cx| call.hang_up(cx));
    let guest_leave = guest_cx
        .read(ActiveCall::global)
        .update(guest_cx, |call, cx| call.hang_up(cx));
    for leave in [host_leave, guest_leave] {
        resync_gate_within(&executor, leave)
            .await
            .expect("cleanup deadline")
            .expect("leave call");
    }
    executor.run_until_parked();
    assert_eq!(outcome, Err(proto::ErrorCode::UnsharedItem));
    assert_eq!(
        after, before,
        "peer ingress must neither create nor extend initialization queues"
    );
    assert_eq!(source, "PUBLIC_CONTROL BASE");
}

#[gpui::test]
async fn test_peer_ingress_unknown_buffer_rejected(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_ingress_unknown_buffer(executor, host_cx, guest_cx, PeerIngressBufferState::Unknown)
        .await;
}

#[gpui::test]
async fn test_peer_ingress_queued_only_buffer_rejected(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_ingress_unknown_buffer(executor, host_cx, guest_cx, PeerIngressBufferState::Queued)
        .await;
}

#[gpui::test]
async fn test_peer_ingress_expired_buffer_rejected(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_ingress_unknown_buffer(executor, host_cx, guest_cx, PeerIngressBufferState::Expired)
        .await;
}

#[derive(Clone, Copy)]
enum PeerSaveWaitChange {
    DestinationPrivate,
    SourcePrivate,
    Unshared,
    Public,
}

async fn check_peer_save_rechecks_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
    change: PeerSaveWaitChange,
    save_as: bool,
) {
    use futures::FutureExt as _;
    use rpc::ErrorExt as _;
    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(host_cx, "save_wait_host").await;
    let guest = server.create_client(guest_cx, "save_wait_guest").await;
    server
        .create_room(&mut [(&host, host_cx), (&guest, guest_cx)])
        .await;
    host.fs()
        .insert_tree(
            "/root/save-wait",
            json!({"source.txt":"BASE","destination.txt":"DO_NOT_OVERWRITE"}),
        )
        .await;
    let (project, worktree_id) = host.build_local_project("/root/save-wait", host_cx).await;
    let host_buffer = project
        .update(host_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("source.txt")), cx)
        })
        .await
        .expect("host source");
    let project_id = host_cx
        .read(ActiveCall::global)
        .update(host_cx, |call, cx| call.share_project(project.clone(), cx))
        .await
        .expect("shared project");
    let guest_project = guest.join_remote_project(project_id, guest_cx).await;
    let guest_buffer = guest_project
        .update(guest_cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("source.txt")), cx)
        })
        .await
        .expect("guest source");
    let (state, replica_id, capability, version) =
        guest_buffer.read_with(guest_cx, |buffer, cx| {
            (
                buffer.to_proto(cx),
                buffer.replica_id(),
                buffer.capability(),
                buffer.version(),
            )
        });
    let history = guest_buffer
        .read_with(guest_cx, |buffer, cx| buffer.serialize_ops(None, cx))
        .await;
    let detached = guest_cx.new(|_| {
        language::Buffer::from_proto(replica_id, capability, state, None).expect("guest replica")
    });
    let history = history
        .into_iter()
        .map(language::proto::deserialize_operation)
        .collect::<Result<Vec<_>>>()
        .expect("base history");
    detached.update(guest_cx, |buffer, cx| {
        buffer.apply_ops(history, cx);
        assert_eq!(buffer.version(), version);
        buffer.edit([(0..0, "PENDING_EDIT ")], None, cx);
    });
    let requested_version = detached.read_with(guest_cx, |buffer, _| buffer.version());
    let operations = detached
        .read_with(guest_cx, |buffer, cx| {
            buffer.serialize_ops(Some(version), cx)
        })
        .await;
    assert!(queued_flusher_has_marker(&operations, "PENDING_EDIT"));
    let buffer_id = host_buffer.read_with(host_cx, |buffer, _| buffer.remote_id());
    let store = project.read_with(host_cx, |project, _| project.buffer_store().clone());
    let observed = store
        .update(host_cx, |store, _| {
            store.observe_peer_save_wait_for_test(project_id, buffer_id)
        })
        .expect("save wait observer");
    let mut save = executor.spawn(guest.request(proto::SaveBuffer {
        project_id,
        buffer_id: buffer_id.to_proto(),
        version: language::proto::serialize_version(&requested_version),
        new_path: save_as.then(|| proto::ProjectPath {
            worktree_id: worktree_id.to_proto(),
            path: "destination.txt".into(),
        }),
    }));
    resync_gate_within(&executor, observed)
        .await
        .expect("save registration deadline")
        .expect("actual save waiter registered");
    assert!(!host_buffer.read_with(host_cx, |buffer, _| {
        buffer.version().observed_all(&requested_version)
    }));
    assert!(
        (&mut save).now_or_never().is_none(),
        "save must actually be waiting for withheld operations"
    );
    match change {
        PeerSaveWaitChange::DestinationPrivate | PeerSaveWaitChange::SourcePrivate => {
            let private_path = match change {
                PeerSaveWaitChange::DestinationPrivate => "**/destination.txt",
                _ => "**/source.txt",
            };
            host_cx.update(|cx| {
                SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |settings| {
                        settings.project.worktree.file_scan_exclusions = Some(vec!["**".into()]);
                        settings.project.worktree.private_files =
                            Some(vec![private_path.to_owned()].into());
                    })
                })
            });
            executor.run_until_parked();
            host_buffer.read_with(host_cx, |buffer, cx| {
                let file = worktree::File::from_dyn(buffer.file()).expect("host file");
                assert!(!file.is_private, "retained metadata must remain stale");
                let local = file.worktree.read(cx).as_local().expect("local policy");
                assert_eq!(
                    local.is_path_private(rel_path("source.txt")),
                    matches!(change, PeerSaveWaitChange::SourcePrivate)
                );
                assert_eq!(
                    local.is_path_private(rel_path("destination.txt")),
                    matches!(change, PeerSaveWaitChange::DestinationPrivate)
                );
            });
        }
        PeerSaveWaitChange::Unshared => {
            host_cx
                .read(ActiveCall::global)
                .update(host_cx, |call, cx| {
                    call.unshare_project(project.clone(), cx)
                })
                .expect("unshare while save awaits its version");
            executor.run_until_parked();
        }
        PeerSaveWaitChange::Public => {}
    }
    if matches!(
        change,
        PeerSaveWaitChange::SourcePrivate | PeerSaveWaitChange::Unshared
    ) {
        // Only the owner releases the clock barrier: peer ingress is no longer authorized.
        let operations = operations
            .into_iter()
            .map(language::proto::deserialize_operation)
            .collect::<Result<Vec<_>>>()
            .expect("withheld operations");
        host_buffer.update(host_cx, |buffer, cx| buffer.apply_ops(operations, cx));
    } else {
        resync_gate_within(
            &executor,
            guest.request(proto::UpdateBuffer {
                project_id,
                buffer_id: buffer_id.to_proto(),
                operations,
            }),
        )
        .await
        .expect("public edit deadline")
        .expect("public source edit releases version waiter");
    }
    let outcome = resync_gate_within(&executor, save)
        .await
        .expect("save response deadline")
        .map(|_| ())
        .map_err(|error| error.error_code());
    executor.run_until_parked();
    let destination = host
        .fs()
        .load(Path::new("/root/save-wait/destination.txt"))
        .await
        .expect("destination remains");
    let source = host
        .fs()
        .load(Path::new("/root/save-wait/source.txt"))
        .await
        .expect("source remains");
    let (text, dirty, path) = host_buffer.read_with(host_cx, |buffer, cx| {
        (
            buffer.text(),
            buffer.is_dirty(),
            buffer.file().map(|file| file.to_proto(cx).path),
        )
    });
    let host_leave = host_cx
        .read(ActiveCall::global)
        .update(host_cx, |call, cx| call.hang_up(cx));
    let guest_leave = guest_cx
        .read(ActiveCall::global)
        .update(guest_cx, |call, cx| call.hang_up(cx));
    for leave in [host_leave, guest_leave] {
        resync_gate_within(&executor, leave)
            .await
            .expect("cleanup deadline")
            .expect("leave call");
    }
    executor.run_until_parked();
    assert_eq!(text, "PENDING_EDIT BASE");
    if matches!(change, PeerSaveWaitChange::Public) {
        assert_eq!(outcome, Ok(()));
        assert!(!dirty);
        if save_as {
            assert_eq!(destination, "PENDING_EDIT BASE");
            assert_eq!(source, "BASE");
            assert_eq!(path.as_deref(), Some("destination.txt"));
        } else {
            assert_eq!(source, "PENDING_EDIT BASE");
            assert_eq!(destination, "DO_NOT_OVERWRITE");
            assert_eq!(path.as_deref(), Some("source.txt"));
        }
    } else {
        assert_eq!(outcome, Err(proto::ErrorCode::UnsharedItem));
        assert_eq!(destination, "DO_NOT_OVERWRITE");
        assert_eq!(source, "BASE");
        assert!(dirty);
        assert_eq!(path.as_deref(), Some("source.txt"));
    }
}

#[gpui::test]
async fn test_peer_save_as_rechecks_destination_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::DestinationPrivate,
        true,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_rechecks_source_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::SourcePrivate,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_rechecks_source_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::SourcePrivate,
        true,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_rechecks_share_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::Unshared,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_rechecks_share_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::Unshared,
        true,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_public_control_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::Public,
        false,
    )
    .await;
}

#[gpui::test]
async fn test_peer_save_as_public_control_after_version_wait(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_save_rechecks_after_version_wait(
        executor,
        host_cx,
        guest_cx,
        PeerSaveWaitChange::Public,
        true,
    )
    .await;
}

#[gpui::test]
async fn test_peer_ingress_explicit_private_edit_opt_in(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        host_cx,
        guest_cx,
        PeerIngressMutation::Edit,
        true,
        false,
        true,
    )
    .await;
}

#[gpui::test]
async fn test_peer_ingress_explicit_private_save_as_opt_in(
    executor: BackgroundExecutor,
    host_cx: &mut TestAppContext,
    guest_cx: &mut TestAppContext,
) {
    check_peer_ingress_retained_buffer(
        executor,
        host_cx,
        guest_cx,
        PeerIngressMutation::SaveAs("existing-destination.txt"),
        true,
        true,
        true,
    )
    .await;
}
