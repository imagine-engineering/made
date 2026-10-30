use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use collaboration_domain::CommunityId;
use percent_encoding::percent_decode_str;
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    net::TcpStream,
    sync::watch,
};
use url::Url;

use super::{
    envelope::FanoutEnvelope,
    subscription_bus::{FanoutTransport, MAX_BUS_SUBSCRIPTIONS, SubscriptionBusError},
};

const REDIS_FANOUT_CHANNEL_PREFIX: &str = "zed.collaboration.message.v1";
const MAX_REDIS_FRAME_BYTES: usize = 64 * 1024;
const REDIS_IO_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) trait RedisStream:
    tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send
{
}
impl<T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send> RedisStream for T {}

#[derive(Clone)]
pub struct RedisFanoutTransport {
    client: RedisClient,
}

#[derive(Clone)]
pub(crate) struct RedisClient {
    endpoint: Url,
}

impl RedisClient {
    pub(crate) fn new(endpoint: &str) -> Result<Self, SubscriptionBusError> {
        let endpoint = Url::parse(endpoint).map_err(|_| SubscriptionBusError::InvalidRequest)?;
        let database = endpoint.path().strip_prefix('/').unwrap_or(endpoint.path());
        if !matches!(endpoint.scheme(), "redis" | "rediss")
            || endpoint.host_str().is_none()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
            || !endpoint.username().is_empty() && endpoint.password().is_none()
            || database.parse::<u16>().is_err() && !database.is_empty()
        {
            return Err(SubscriptionBusError::InvalidRequest);
        }
        Ok(Self { endpoint })
    }

    pub(crate) async fn connect(&self) -> Result<Box<dyn RedisStream>, SubscriptionBusError> {
        let host = match self
            .endpoint
            .host()
            .ok_or(SubscriptionBusError::InvalidRequest)?
        {
            url::Host::Domain(domain) => domain.to_owned(),
            url::Host::Ipv4(address) => address.to_string(),
            url::Host::Ipv6(address) => address.to_string(),
        };
        let port = self.endpoint.port().unwrap_or(6379);
        let stream = TcpStream::connect((host.as_str(), port))
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        let mut stream: Box<dyn RedisStream> = if self.endpoint.scheme() == "rediss" {
            Box::new(connect_tls(stream, &host, http_client_tls::tls_config()).await?)
        } else {
            Box::new(stream)
        };
        authenticate(&mut stream, &self.endpoint).await?;
        let database = self
            .endpoint
            .path()
            .strip_prefix('/')
            .unwrap_or(self.endpoint.path());
        if !database.is_empty() && database != "0" {
            write_command(&mut stream, &[b"SELECT", database.as_bytes()]).await?;
            read_resp_simple(&mut BufReader::new(&mut stream)).await?;
        }
        Ok(stream)
    }
}

async fn connect_tls(
    stream: TcpStream,
    host: &str,
    configuration: tokio_rustls::rustls::ClientConfig,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, SubscriptionBusError> {
    let server_name = tokio_rustls::rustls::pki_types::ServerName::try_from(host.to_owned())
        .map_err(|_| SubscriptionBusError::InvalidRequest)?;
    tokio_rustls::TlsConnector::from(Arc::new(configuration))
        .connect(server_name, stream)
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)
}

impl RedisFanoutTransport {
    pub fn new(endpoint: &str) -> Result<Self, SubscriptionBusError> {
        Ok(Self {
            client: RedisClient::new(endpoint)?,
        })
    }

    pub async fn subscribe<Callback, CallbackFuture>(
        &self,
        mut communities: watch::Receiver<Vec<CommunityId>>,
        mut on_envelope: Callback,
    ) -> Result<(), SubscriptionBusError>
    where
        Callback: FnMut(Vec<u8>) -> CallbackFuture + Send,
        CallbackFuture: std::future::Future<Output = ()> + Send,
    {
        loop {
            let selected = communities.borrow_and_update().clone();
            if selected.len() > MAX_BUS_SUBSCRIPTIONS {
                return Err(SubscriptionBusError::CapacityExceeded);
            }
            if selected.is_empty() {
                communities
                    .changed()
                    .await
                    .map_err(|_| SubscriptionBusError::Shutdown)?;
                continue;
            }
            let channels = selected
                .into_iter()
                .map(community_channel)
                .collect::<Vec<_>>();
            let mut reader = tokio::select! {
                changed = communities.changed() => {
                    changed.map_err(|_| SubscriptionBusError::Shutdown)?;
                    continue;
                }
                reader = tokio::time::timeout(REDIS_IO_TIMEOUT, async {
                    let mut stream = self.client.connect().await?;
                    let mut command = vec![b"SUBSCRIBE".as_slice()];
                    command.extend(channels.iter().map(|channel| channel.as_bytes()));
                    write_command(&mut stream, &command).await?;
                    let mut reader = BufReader::new(stream);
                    for channel in &channels {
                        let acknowledgment = read_resp_array(&mut reader).await?;
                        if acknowledgment.len() != 3
                            || acknowledgment.first().map(Vec::as_slice) != Some(b"subscribe")
                            || acknowledgment.get(1).map(Vec::as_slice) != Some(channel.as_bytes())
                        {
                            return Err(SubscriptionBusError::Unavailable);
                        }
                    }
                    Ok(reader)
                }) => reader.map_err(|_| SubscriptionBusError::Unavailable)??,
            };
            loop {
                tokio::select! {
                    changed = communities.changed() => {
                        changed.map_err(|_| SubscriptionBusError::Shutdown)?;
                        // A cancelled RESP read may have consumed part of a frame. Reopen
                        // the socket rather than interpreting its remainder as a new frame.
                        break;
                    }
                    frame = read_resp_array(&mut reader) => {
                        let frame = frame?;
                        if frame.len() != 3 || frame.first().map(Vec::as_slice) != Some(b"message") {
                            return Err(SubscriptionBusError::Unavailable);
                        }
                        let envelope = FanoutEnvelope::decode(&frame[2])?;
                        let channel = community_channel(envelope.community_id());
                        if frame[1] != channel.as_bytes() || !channels.contains(&channel) {
                            return Err(SubscriptionBusError::TenantMismatch);
                        }
                        on_envelope(frame[2].clone()).await;
                    }
                }
            }
        }
    }
}

#[async_trait]
impl FanoutTransport for RedisFanoutTransport {
    async fn publish(&self, encoded_envelope: Vec<u8>) -> Result<(), SubscriptionBusError> {
        if encoded_envelope.len() > MAX_REDIS_FRAME_BYTES {
            return Err(SubscriptionBusError::InvalidRequest);
        }
        let envelope = FanoutEnvelope::decode(&encoded_envelope)?;
        let channel = community_channel(envelope.community_id());
        tokio::time::timeout(REDIS_IO_TIMEOUT, async {
            let mut stream = self.client.connect().await?;
            write_command(
                &mut stream,
                &[b"PUBLISH", channel.as_bytes(), &encoded_envelope],
            )
            .await?;
            let mut reader = BufReader::new(stream);
            read_resp_integer(&mut reader).await?;
            Ok(())
        })
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)?
    }
}

fn community_channel(community_id: CommunityId) -> String {
    format!("{REDIS_FANOUT_CHANNEL_PREFIX}:{}", community_id.as_uuid())
}

async fn authenticate(
    stream: &mut (impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin),
    endpoint: &Url,
) -> Result<(), SubscriptionBusError> {
    let password = endpoint.password();
    let username = endpoint.username();
    let Some(password) = password else {
        return Ok(());
    };
    let password = percent_decode_str(password).collect::<Vec<_>>();
    let username = percent_decode_str(username).collect::<Vec<_>>();
    if username.is_empty() {
        write_command(stream, &[b"AUTH", &password]).await?;
    } else {
        write_command(stream, &[b"AUTH", &username, &password]).await?;
    }
    let mut reader = BufReader::new(stream);
    read_resp_simple(&mut reader).await?;
    Ok(())
}

pub(crate) async fn write_command(
    stream: &mut (impl tokio::io::AsyncWrite + Unpin),
    parts: &[&[u8]],
) -> Result<(), SubscriptionBusError> {
    stream
        .write_all(format!("*{}\r\n", parts.len()).as_bytes())
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    for part in parts {
        stream
            .write_all(format!("${}\r\n", part.len()).as_bytes())
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        stream
            .write_all(part)
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        stream
            .write_all(b"\r\n")
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
    }
    stream
        .flush()
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)
}

pub(crate) async fn read_line(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
) -> Result<Vec<u8>, SubscriptionBusError> {
    let mut line = Vec::new();
    reader
        .take(MAX_REDIS_FRAME_BYTES as u64 + 1)
        .read_until(b'\n', &mut line)
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    if line.len() < 2 || !line.ends_with(b"\r\n") || line.len() > MAX_REDIS_FRAME_BYTES {
        return Err(SubscriptionBusError::Unavailable);
    }
    line.truncate(line.len() - 2);
    Ok(line)
}

async fn read_resp_simple(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
) -> Result<(), SubscriptionBusError> {
    let line = read_line(reader).await?;
    if line == b"+OK" {
        Ok(())
    } else {
        Err(SubscriptionBusError::Unavailable)
    }
}

async fn read_resp_integer(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
) -> Result<i64, SubscriptionBusError> {
    let line = read_line(reader).await?;
    if line.first() != Some(&b':') {
        return Err(SubscriptionBusError::Unavailable);
    }
    std::str::from_utf8(&line[1..])
        .ok()
        .and_then(|value| value.parse().ok())
        .ok_or(SubscriptionBusError::Unavailable)
}

async fn read_resp_array(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
) -> Result<Vec<Vec<u8>>, SubscriptionBusError> {
    let header = read_line(reader).await?;
    if header.first() != Some(&b'*') {
        return Err(SubscriptionBusError::Unavailable);
    }
    let length = std::str::from_utf8(&header[1..])
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|length| *length <= 16)
        .ok_or(SubscriptionBusError::Unavailable)?;
    let mut parts = Vec::with_capacity(length);
    for _ in 0..length {
        let bulk_header = read_line(reader).await?;
        match bulk_header.first() {
            Some(b'$') => {
                let bulk_length = std::str::from_utf8(&bulk_header[1..])
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|length| *length <= MAX_REDIS_FRAME_BYTES)
                    .ok_or(SubscriptionBusError::Unavailable)?;
                let mut value = vec![0; bulk_length];
                reader
                    .read_exact(&mut value)
                    .await
                    .map_err(|_| SubscriptionBusError::Unavailable)?;
                let mut terminator = [0; 2];
                reader
                    .read_exact(&mut terminator)
                    .await
                    .map_err(|_| SubscriptionBusError::Unavailable)?;
                if terminator != *b"\r\n" {
                    return Err(SubscriptionBusError::Unavailable);
                }
                parts.push(value);
            }
            Some(b':') | Some(b'+') => parts.push(bulk_header[1..].to_vec()),
            _ => return Err(SubscriptionBusError::Unavailable),
        }
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio_rustls::{TlsAcceptor, rustls};

    // Public test-only localhost certificate/key shared with the native HTTPS
    // conformance fixture; it is never installed in platform trust.
    const CERTIFICATE: &str = "3082017330820118a0030201020209009239e51ef94926b2300a06082a8648ce3d04030230143112301006035504030c096c6f63616c686f73743020170d3236303731343037343231325a180f32313236303632303037343231325a30143112301006035504030c096c6f63616c686f73743059301306072a8648ce3d020106082a8648ce3d03010703420004e8a53ae45a5102a16513b8eca4846b0d7fb839433b9f63455b4f8e3c87a65d917c2cb89b05efe4670c574cc84c8e49501f415535c796f2208b7be37c98a1d276a351304f301a0603551d110413301182096c6f63616c686f737487047f000001300c0603551d130101ff04023000300e0603551d0f0101ff04040302078030130603551d25040c300a06082b06010505070301300a06082a8648ce3d0403020349003046022100d28feed17c2977f293e9710d60f7d2e8f7030fcfbad64a6b811fe59bad22841e02210096c34263e1724068eb1c57d894beef8beda8e95b68400ca908cee911ef72f06c";
    const PRIVATE_KEY: &str = "308187020100301306072a8648ce3d020106082a8648ce3d030107046d306b02010104203fad6b2b66129a76f0de09af738ada166ebe481204d8fc4c0d189d5dd23a7c8aa14403420004e8a53ae45a5102a16513b8eca4846b0d7fb839433b9f63455b4f8e3c87a65d917c2cb89b05efe4670c574cc84c8e49501f415535c796f2208b7be37c98a1d276";

    #[tokio::test]
    async fn redis_tls_checks_certificate_trust_and_host() -> anyhow::Result<()> {
        let certificate = rustls::pki_types::CertificateDer::from(hex::decode(CERTIFICATE)?);
        let private_key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(hex::decode(PRIVATE_KEY)?),
        );
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_no_client_auth()
            .with_single_cert(vec![certificate.clone()], private_key)?;
        let mut roots = rustls::RootCertStore::empty();
        roots.add(certificate)?;
        let trusted = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots)
            .with_no_client_auth();
        for (host, configuration, accepted) in [
            ("localhost", trusted.clone(), true),
            ("wrong.example.test", trusted, false),
            ("localhost", http_client_tls::tls_config(), false),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let acceptor = TlsAcceptor::from(Arc::new(server.clone()));
            let server = tokio::spawn(async move {
                let (stream, _) = listener.accept().await?;
                let result = acceptor.accept(stream).await;
                if accepted {
                    let mut stream = result?;
                    write_command(&mut stream, &[b"PING"]).await?;
                } else {
                    anyhow::ensure!(
                        result.is_err(),
                        "untrusted or wrong-host handshake accepted"
                    );
                }
                anyhow::Ok(())
            });
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                connect_tls(TcpStream::connect(address).await?, host, configuration),
            )
            .await?;
            if accepted {
                let mut reader = BufReader::new(result?);
                assert_eq!(read_resp_array(&mut reader).await?, vec![b"PING".to_vec()]);
            } else {
                assert!(matches!(result, Err(SubscriptionBusError::Unavailable)));
            }
            server.await??;
        }
        Ok(())
    }
}
