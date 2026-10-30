use agent::collaboration_session::{
    CollaborationExecutorId, CollaborationSessionError, CollaborationSessionIdentity,
    CollaborationSessionRegistry, CollaborationSessionResolution, CollaborationSessionScope,
};
use agent_client_protocol::schema::v1 as acp;
use uuid::Uuid;

fn executor(value: u128) -> CollaborationExecutorId {
    CollaborationExecutorId::new(Uuid::from_u128(value)).expect("executor")
}

fn channel(value: u128) -> CollaborationSessionIdentity {
    CollaborationSessionIdentity::new(
        Uuid::from_u128(1),
        CollaborationSessionScope::Channel {
            channel_id: Uuid::from_u128(value),
        },
    )
    .expect("channel identity")
}

fn thread(channel: u128, thread: u128) -> CollaborationSessionIdentity {
    CollaborationSessionIdentity::new(
        Uuid::from_u128(1),
        CollaborationSessionScope::Thread {
            channel_id: Uuid::from_u128(channel),
            thread_id: Uuid::from_u128(thread),
        },
    )
    .expect("thread identity")
}

fn session(value: &str) -> acp::SessionId {
    acp::SessionId::new(value)
}

#[test]
fn collaboration_session_create_and_resume_are_idempotent() {
    let mut registry = CollaborationSessionRegistry::default();
    let identity = thread(10, 20);
    let executor = executor(30);

    let CollaborationSessionResolution::Create(first_lease) =
        registry.resolve(identity, executor).expect("reserve")
    else {
        panic!("expected creation reservation")
    };
    assert_eq!(
        registry.resolve(identity, executor),
        Ok(CollaborationSessionResolution::Create(first_lease.clone()))
    );

    let session_id = session("native-session-1");
    registry
        .activate(&first_lease, session_id.clone())
        .expect("activate");
    registry
        .activate(&first_lease, session_id.clone())
        .expect("idempotent activate");
    assert_eq!(
        registry.resolve(identity, executor),
        Ok(CollaborationSessionResolution::Resume {
            lease: first_lease,
            session_id: session_id.clone(),
        })
    );
    assert_eq!(registry.active_session(identity), Some(&session_id));
    assert_eq!(registry.len(), 1);
}

#[test]
fn collaboration_session_allows_exactly_one_executor_and_native_session_owner() {
    let mut registry = CollaborationSessionRegistry::default();
    let first_identity = channel(10);
    let second_identity = channel(20);
    let first_executor = executor(30);
    let second_executor = executor(40);

    let first = registry
        .resolve(first_identity, first_executor)
        .expect("first reservation")
        .lease()
        .clone();
    assert_eq!(
        registry.resolve(first_identity, second_executor),
        Err(CollaborationSessionError::ExecutorAlreadyClaimed)
    );
    registry
        .activate(&first, session("native-session-1"))
        .expect("first activation");
    assert_eq!(
        registry.resolve(first_identity, second_executor),
        Err(CollaborationSessionError::ExecutorAlreadyClaimed)
    );

    let second = registry
        .resolve(second_identity, second_executor)
        .expect("second reservation")
        .lease()
        .clone();
    assert_eq!(
        registry.activate(&second, session("native-session-1")),
        Err(CollaborationSessionError::SessionAlreadyBound)
    );
    registry
        .activate(&second, session("native-session-2"))
        .expect("distinct activation");
    assert_eq!(registry.len(), 2);
}

#[test]
fn collaboration_session_cancellation_is_owned_and_generation_fenced() {
    let mut registry = CollaborationSessionRegistry::default();
    let identity = channel(10);
    let first_executor = executor(20);
    let second_executor = executor(30);

    let stale_lease = registry
        .resolve(identity, first_executor)
        .expect("reserve")
        .lease()
        .clone();
    assert_eq!(
        registry.authorize_cancellation(&stale_lease),
        Err(CollaborationSessionError::SessionNotActive)
    );
    registry.abort_creation(&stale_lease).expect("abort");

    let current_lease = registry
        .resolve(identity, second_executor)
        .expect("replace")
        .lease()
        .clone();
    registry
        .activate(&current_lease, session("native-session-2"))
        .expect("activate");
    assert_eq!(
        registry.authorize_cancellation(&stale_lease),
        Err(CollaborationSessionError::LeaseNotCurrent)
    );

    let authorization = registry
        .authorize_cancellation(&current_lease)
        .expect("current owner can cancel");
    assert_eq!(authorization.session_id(), &session("native-session-2"));
    registry
        .complete_cancellation(&authorization)
        .expect("complete cancel");
    assert!(registry.is_empty());
}

#[test]
fn collaboration_session_scopes_and_identifiers_fail_closed() {
    assert_eq!(
        CollaborationSessionIdentity::new(
            Uuid::nil(),
            CollaborationSessionScope::Channel {
                channel_id: Uuid::from_u128(1),
            },
        ),
        Err(CollaborationSessionError::InvalidIdentity)
    );
    assert_eq!(
        CollaborationSessionIdentity::new(
            Uuid::from_u128(1),
            CollaborationSessionScope::Job {
                channel_id: Uuid::from_u128(2),
                thread_id: Some(Uuid::nil()),
                job_id: Uuid::from_u128(3),
            },
        ),
        Err(CollaborationSessionError::InvalidIdentity)
    );

    let mut registry = CollaborationSessionRegistry::default();
    let identity = channel(10);
    let lease = registry
        .resolve(identity, executor(20))
        .expect("reserve")
        .lease()
        .clone();
    assert_eq!(
        registry.activate(&lease, session("   ")),
        Err(CollaborationSessionError::InvalidSessionId)
    );
    assert_ne!(identity, thread(10, 20));
}
