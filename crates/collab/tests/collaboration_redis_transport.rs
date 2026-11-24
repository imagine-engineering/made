use std::time::Duration;

use collab::pubsub::{
    envelope::FanoutEnvelope,
    redis::RedisFanoutTransport,
    subscription_bus::{FanoutTransport as _, SubscriptionBusError},
};
use collaboration_domain::{CommunityId, Provenance, SourceRecordId, SourceSystem};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    net::TcpListener,
    sync::{mpsc, oneshot, watch},
};

fn envelope(community: CommunityId) -> anyhow::Result<Vec<u8>> {
    FanoutEnvelope::new(
        community,
        1,
        "conversation.activity",
        Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new("redis-boundary").ok_or_else(|| anyhow::anyhow!("source ID"))?,
            1_900_000_000_000,
        )
        .with_source_version("1"),
        [3; 32],
    )?
    .encode()
    .map_err(Into::into)
}

async fn publish_with_response(
    response: Vec<u8>,
) -> anyhow::Result<Result<(), SubscriptionBusError>> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("redis://{}/0", listener.local_addr()?);
    let (release, released) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        let mut command = [0; 256];
        anyhow::ensure!(stream.read(&mut command).await? > 0);
        stream.write_all(&response).await?;
        // Keep the connection open so EOF cannot make an unbounded reader pass.
        released.await?;
        anyhow::Ok(())
    });
    let transport = RedisFanoutTransport::new(&endpoint)?;
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        transport.publish(envelope(CommunityId::new())?),
    )
    .await;
    release
        .send(())
        .map_err(|_| anyhow::anyhow!("response server stopped before cleanup"))?;
    server.await??;
    Ok(result?)
}

fn frame(parts: &[&[u8]]) -> Vec<u8> {
    let mut frame = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        frame.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
        frame.extend_from_slice(part);
        frame.extend_from_slice(b"\r\n");
    }
    frame
}

async fn read_exact_command(
    stream: &mut tokio::net::TcpStream,
    expected: &[u8],
) -> anyhow::Result<()> {
    let mut actual = vec![0; expected.len()];
    stream.read_exact(&mut actual).await?;
    anyhow::ensure!(actual == expected, "unexpected Redis command");
    Ok(())
}

fn channel(community: CommunityId) -> String {
    format!("zed.collaboration.message.v1:{}", community.as_uuid())
}

#[tokio::test]
async fn redis_transport_decodes_credentials_before_authentication() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!(
        "redis://user%40name:pass%3Aword%25@{}/2",
        listener.local_addr()?
    ))?;
    let payload = envelope(CommunityId::new())?;
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_exact_command(&mut stream, &frame(&[b"AUTH", b"user@name", b"pass:word%"])).await?;
        stream.write_all(b"+OK\r\n").await?;
        read_exact_command(&mut stream, &frame(&[b"SELECT", b"2"])).await?;
        stream.write_all(b"+OK\r\n").await?;
        let mut command = [0; 512];
        anyhow::ensure!(stream.read(&mut command).await? > 0);
        stream.write_all(b":1\r\n").await?;
        anyhow::Ok(())
    });
    transport.publish(payload).await?;
    server.await??;
    for endpoint in [
        "redis://username@localhost",
        "redis://localhost/0?tls=false",
        "redis://localhost/0#ignored",
        "redis://localhost//2",
        "redis://localhost/2/",
    ] {
        assert!(RedisFanoutTransport::new(endpoint).is_err());
    }
    Ok(())
}

#[tokio::test]
async fn redis_transport_connects_to_an_ipv6_literal() -> anyhow::Result<()> {
    let listener = TcpListener::bind("[::1]:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let community = CommunityId::new();
    let payload = envelope(community)?;
    let command = frame(&[b"PUBLISH", channel(community).as_bytes(), &payload]);
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_exact_command(&mut stream, &command).await?;
        stream.write_all(b":1\r\n").await?;
        anyhow::Ok(())
    });
    transport.publish(payload).await?;
    server.await??;
    Ok(())
}

#[tokio::test]
async fn redis_transport_tls_never_falls_back_to_plaintext() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!(
        "rediss://username:secret@{}",
        listener.local_addr()?
    ))?;
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        let mut header = [0; 3];
        stream.read_exact(&mut header).await?;
        // A TLS handshake precedes any RESP credentials or payload.
        anyhow::ensure!(header[0] == 0x16 && header[1] == 0x03);
        stream.write_all(b"+OK\r\n").await?;
        let mut remainder = Vec::new();
        stream.read_to_end(&mut remainder).await?;
        anyhow::ensure!(!remainder.windows(6).any(|bytes| bytes == b"secret"));
        anyhow::ensure!(!remainder.windows(4).any(|bytes| bytes == b"AUTH"));
        anyhow::Ok(())
    });
    assert_eq!(
        transport.publish(envelope(CommunityId::new())?).await,
        Err(SubscriptionBusError::Unavailable)
    );
    server.await??;
    Ok(())
}

#[tokio::test]
async fn redis_transport_times_out_stalled_publish_and_subscription() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let (release, released) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (first, _) = listener.accept().await?;
        let (second, _) = listener.accept().await?;
        released.await?;
        drop((first, second));
        anyhow::Ok(())
    });
    let (_scope, communities) = watch::channel(vec![CommunityId::new()]);
    let payload = envelope(CommunityId::new())?;
    let result = tokio::time::timeout(Duration::from_secs(6), async {
        tokio::join!(
            transport.publish(payload),
            transport.subscribe(communities, |_| std::future::ready(()))
        )
    })
    .await;
    release
        .send(())
        .map_err(|_| anyhow::anyhow!("server stopped"))?;
    server.await??;
    let (published, subscribed) = result?;
    assert_eq!(published, Err(SubscriptionBusError::Unavailable));
    assert_eq!(subscribed, Err(SubscriptionBusError::Unavailable));
    Ok(())
}

#[tokio::test]
async fn redis_transport_routes_publish_by_validated_community() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let community = CommunityId::new();
    let payload = envelope(community)?;
    let command = frame(&[b"PUBLISH", channel(community).as_bytes(), &payload]);
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_exact_command(&mut stream, &command).await?;
        stream.write_all(b":1\r\n").await?;
        anyhow::Ok(())
    });
    assert!(transport.publish(vec![1]).await.is_err());
    tokio::time::timeout(Duration::from_secs(2), transport.publish(payload)).await??;
    server.await??;
    if let Ok(endpoint) = std::env::var("COLLAB_TEST_REDIS_URL") {
        verify_disposable_redis_scope(&endpoint).await?;
        println!("disposable Redis: tenant isolation, scope replacement and shutdown verified");
    }
    Ok(())
}

async fn wait_for_subscribers(
    endpoint: &str,
    community: CommunityId,
    expected: usize,
) -> anyhow::Result<()> {
    let endpoint = url::Url::parse(endpoint)?;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let mut stream = tokio::net::TcpStream::connect((
                endpoint
                    .host_str()
                    .ok_or_else(|| anyhow::anyhow!("Redis host"))?,
                endpoint.port().unwrap_or(6379),
            ))
            .await?;
            let channel = channel(community);
            stream
                .write_all(&frame(&[b"PUBSUB", b"NUMSUB", channel.as_bytes()]))
                .await?;
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            anyhow::ensure!(line == "*2\r\n", "invalid PUBSUB result");
            line.clear();
            reader.read_line(&mut line).await?;
            anyhow::ensure!(
                line == format!("${}\r\n", channel.len()),
                "invalid channel length"
            );
            line.clear();
            reader.read_line(&mut line).await?;
            anyhow::ensure!(line == format!("{channel}\r\n"), "wrong channel result");
            line.clear();
            reader.read_line(&mut line).await?;
            let count = line
                .trim()
                .strip_prefix(':')
                .ok_or_else(|| anyhow::anyhow!("invalid count"))?
                .parse::<usize>()?;
            if count == expected {
                return anyhow::Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?
}

async fn verify_disposable_redis_scope(endpoint: &str) -> anyhow::Result<()> {
    let transport = RedisFanoutTransport::new(endpoint)?;
    let first = CommunityId::new();
    let second = CommunityId::new();
    let replacement = CommunityId::new();
    let (scope, communities) = watch::channel(vec![first]);
    let (other_scope, other_communities) = watch::channel(vec![second]);
    let (delivered, mut received) = mpsc::unbounded_channel();
    let (other_delivered, mut other_received) = mpsc::unbounded_channel();
    let subscriber_transport = transport.clone();
    let subscriber = tokio::spawn(async move {
        subscriber_transport
            .subscribe(communities, move |payload| {
                delivered
                    .send(payload)
                    .expect("live test receiver remains active");
                std::future::ready(())
            })
            .await
    });
    let other_transport = transport.clone();
    let other_subscriber = tokio::spawn(async move {
        other_transport
            .subscribe(other_communities, move |payload| {
                other_delivered
                    .send(payload)
                    .expect("other live test receiver remains active");
                std::future::ready(())
            })
            .await
    });
    wait_for_subscribers(endpoint, first, 1).await?;
    wait_for_subscribers(endpoint, second, 1).await?;
    transport.publish(envelope(first)?).await?;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), received.recv()).await?,
        Some(envelope(first)?)
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), other_received.recv())
            .await
            .is_err(),
        "foreign community received the notification"
    );
    scope.send_replace(vec![replacement]);
    wait_for_subscribers(endpoint, first, 0).await?;
    wait_for_subscribers(endpoint, replacement, 1).await?;
    transport.publish(envelope(first)?).await?;
    transport.publish(envelope(replacement)?).await?;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), received.recv()).await?,
        Some(envelope(replacement)?)
    );
    drop(scope);
    drop(other_scope);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), subscriber).await??,
        Err(SubscriptionBusError::Shutdown)
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), other_subscriber).await??,
        Err(SubscriptionBusError::Shutdown)
    );
    wait_for_subscribers(endpoint, second, 0).await?;
    wait_for_subscribers(endpoint, replacement, 0).await?;
    Ok(())
}

#[tokio::test]
async fn redis_transport_replaces_scope_and_closes_when_registration_ends() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let first = CommunityId::new();
    let second = CommunityId::new();
    let (scope, communities) = watch::channel(vec![first]);
    let (delivered, mut received) = mpsc::unbounded_channel();
    let subscriber = tokio::spawn(async move {
        transport
            .subscribe(communities, move |payload| {
                delivered
                    .send(payload)
                    .expect("test receiver remains active");
                std::future::ready(())
            })
            .await
    });
    let server = tokio::spawn(async move {
        for community in [first, second] {
            let (mut stream, _) = listener.accept().await?;
            let channel = channel(community);
            read_exact_command(&mut stream, &frame(&[b"SUBSCRIBE", channel.as_bytes()])).await?;
            stream
                .write_all(&frame(&[b"subscribe", channel.as_bytes(), b"1"]))
                .await?;
            stream
                .write_all(&frame(&[
                    b"message",
                    channel.as_bytes(),
                    &envelope(community)?,
                ]))
                .await?;
            let mut byte = [0];
            anyhow::ensure!(
                stream.read(&mut byte).await? == 0,
                "superseded socket was not closed"
            );
        }
        anyhow::Ok(())
    });
    let received_first = tokio::time::timeout(Duration::from_secs(2), received.recv()).await?;
    assert_eq!(received_first, Some(envelope(first)?));
    scope.send_replace(vec![second]);
    let received_second = tokio::time::timeout(Duration::from_secs(2), received.recv()).await?;
    assert_eq!(received_second, Some(envelope(second)?));
    drop(scope);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), subscriber).await??,
        Err(SubscriptionBusError::Shutdown)
    );
    tokio::time::timeout(Duration::from_secs(2), server).await???;
    Ok(())
}

#[tokio::test]
async fn redis_transport_cancels_pending_subscription_acknowledgment() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let community = CommunityId::new();
    let (scope, communities) = watch::channel(vec![community]);
    let (accepted, acceptance) = oneshot::channel();
    let subscriber = tokio::spawn(async move {
        transport
            .subscribe(communities, |_| std::future::ready(()))
            .await
    });
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_exact_command(
            &mut stream,
            &frame(&[b"SUBSCRIBE", channel(community).as_bytes()]),
        )
        .await?;
        accepted
            .send(())
            .map_err(|_| anyhow::anyhow!("test subscriber stopped"))?;
        let mut byte = [0];
        anyhow::ensure!(
            stream.read(&mut byte).await? == 0,
            "pending socket was not closed"
        );
        anyhow::Ok(())
    });
    tokio::time::timeout(Duration::from_secs(2), acceptance).await??;
    drop(scope);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), subscriber).await??,
        Err(SubscriptionBusError::Shutdown)
    );
    tokio::time::timeout(Duration::from_secs(2), server).await???;
    Ok(())
}

#[tokio::test]
async fn redis_transport_rejects_mismatched_wire_community() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let transport = RedisFanoutTransport::new(&format!("redis://{}", listener.local_addr()?))?;
    let community = CommunityId::new();
    let (_scope, communities) = watch::channel(vec![community]);
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        let channel = channel(community);
        read_exact_command(&mut stream, &frame(&[b"SUBSCRIBE", channel.as_bytes()])).await?;
        stream
            .write_all(&frame(&[b"subscribe", channel.as_bytes(), b"1"]))
            .await?;
        stream
            .write_all(&frame(&[
                b"message",
                channel.as_bytes(),
                &envelope(CommunityId::new())?,
            ]))
            .await?;
        anyhow::Ok(())
    });
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        transport.subscribe(communities, |_| async {
            panic!("foreign envelope reached the consumer");
        }),
    )
    .await?;
    assert_eq!(result, Err(SubscriptionBusError::TenantMismatch));
    server.await??;
    Ok(())
}

#[tokio::test]
async fn redis_transport_bounds_scope_before_connecting() -> anyhow::Result<()> {
    let transport = RedisFanoutTransport::new("redis://127.0.0.1:1")?;
    let maximum_subscriptions = collab::pubsub::subscription_bus::MAX_BUS_SUBSCRIPTIONS;
    let (_scope, communities) = watch::channel(vec![CommunityId::new(); maximum_subscriptions + 1]);
    assert_eq!(
        transport
            .subscribe(communities, |_| std::future::ready(()))
            .await,
        Err(SubscriptionBusError::CapacityExceeded)
    );
    let (scope, communities) = watch::channel(Vec::new());
    drop(scope);
    assert_eq!(
        transport
            .subscribe(communities, |_| std::future::ready(()))
            .await,
        Err(SubscriptionBusError::Shutdown)
    );
    Ok(())
}

#[tokio::test]
async fn redis_transport_rejects_oversized_unterminated_response() -> anyhow::Result<()> {
    assert_eq!(
        publish_with_response(vec![b':'; 64 * 1024 + 1]).await?,
        Err(SubscriptionBusError::Unavailable)
    );
    Ok(())
}

#[tokio::test]
async fn redis_transport_accepts_publish_count_and_rejects_malformed_response() -> anyhow::Result<()>
{
    assert_eq!(publish_with_response(b":1\r\n".to_vec()).await?, Ok(()));
    assert_eq!(
        publish_with_response(b"+OK\r\n".to_vec()).await?,
        Err(SubscriptionBusError::Unavailable)
    );
    Ok(())
}
