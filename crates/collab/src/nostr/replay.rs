use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use collaboration_domain::{CommunityId, TenantContext};
use nostr_compat::EventId;
use tokio::io::BufReader;

use crate::pubsub::{
    redis::{RedisClient, read_line, write_command},
    subscription_bus::SubscriptionBusError,
};

use super::{
    auth::{NostrAuthReplayStore, NostrAuthenticationInfrastructureError, ReplayClaim},
    http::{Nip98ReplayClaim, Nip98ReplayStore, NostrHttpDirectoryError},
};

const MAX_REPLAY_LIFETIME_SECONDS: u64 = 120;
const REPLAY_IO_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub struct RedisAuthenticationReplayStore {
    client: RedisClient,
}

impl RedisAuthenticationReplayStore {
    pub fn new(endpoint: &str) -> Result<Self, SubscriptionBusError> {
        Ok(Self {
            client: RedisClient::new(endpoint)?,
        })
    }

    async fn claim_once(
        &self,
        community: CommunityId,
        event: EventId,
        expires_at_seconds: u64,
    ) -> Result<bool, NostrAuthenticationInfrastructureError> {
        let unavailable = NostrAuthenticationInfrastructureError::Unavailable;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| unavailable)?
            .as_secs();
        let remaining = expires_at_seconds
            .checked_sub(now)
            .filter(|remaining| *remaining > 0 && *remaining <= MAX_REPLAY_LIFETIME_SECONDS)
            .ok_or(unavailable)?;
        // Signed-event freshness accepts the final whole second inclusively.
        // Keep the fence through that second rather than reopening replay then.
        let lifetime = (remaining + 1).to_string();
        let key = format!(
            "zed.collaboration.auth.v1:{}:{}",
            community.as_uuid(),
            hex::encode(event.as_bytes())
        );
        tokio::time::timeout(REPLAY_IO_TIMEOUT, async {
            let mut stream = self.client.connect().await.map_err(|_| unavailable)?;
            write_command(
                &mut stream,
                &[
                    b"SET",
                    key.as_bytes(),
                    b"1",
                    b"NX",
                    b"EX",
                    lifetime.as_bytes(),
                ],
            )
            .await
            .map_err(|_| unavailable)?;
            match read_line(&mut BufReader::new(stream))
                .await
                .map_err(|_| unavailable)?
                .as_slice()
            {
                b"+OK" => Ok(true),
                b"$-1" => Ok(false),
                _ => Err(unavailable),
            }
        })
        .await
        .map_err(|_| unavailable)?
    }
}

#[async_trait]
impl NostrAuthReplayStore for RedisAuthenticationReplayStore {
    async fn claim(
        &self,
        community_id: CommunityId,
        event_id: EventId,
        expires_at_seconds: u64,
    ) -> Result<ReplayClaim, NostrAuthenticationInfrastructureError> {
        self.claim_once(community_id, event_id, expires_at_seconds)
            .await
            .map(|claimed| {
                if claimed {
                    ReplayClaim::Claimed
                } else {
                    ReplayClaim::Replay
                }
            })
    }
}

#[async_trait]
impl Nip98ReplayStore for RedisAuthenticationReplayStore {
    async fn claim(
        &self,
        tenant: &TenantContext,
        event_id: EventId,
        expires_at_seconds: u64,
    ) -> Result<Nip98ReplayClaim, NostrHttpDirectoryError> {
        self.claim_once(tenant.community_id(), event_id, expires_at_seconds)
            .await
            .map(|claimed| {
                if claimed {
                    Nip98ReplayClaim::Claimed
                } else {
                    Nip98ReplayClaim::Replay
                }
            })
            .map_err(|_| NostrHttpDirectoryError::Unavailable)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpQuotaAdmissionError {
    Exceeded { reset_in_seconds: u64 },
    Unavailable,
}

#[derive(Clone)]
pub struct RedisHttpAdmissionQuota {
    client: RedisClient,
    calls_per_minute: u64,
}

impl RedisHttpAdmissionQuota {
    pub fn new(endpoint: &str, calls_per_minute: u64) -> Result<Self, SubscriptionBusError> {
        Ok(Self {
            client: RedisClient::new(endpoint)?,
            calls_per_minute,
        })
    }

    pub async fn check(
        &self,
        tenant: &TenantContext,
        public_key: nostr_compat::PublicKey,
    ) -> Result<(), HttpQuotaAdmissionError> {
        let key = format!(
            "buzz:{}:ratelimit:{}:api",
            tenant.community_id(),
            public_key.to_hex()
        );
        const SCRIPT: &str = "local count = redis.call('INCR', KEYS[1]); if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end; local ttl = redis.call('TTL', KEYS[1]); if ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end; return {count, ttl}";
        let unavailable = HttpQuotaAdmissionError::Unavailable;
        let (count, reset_in_seconds) = tokio::time::timeout(REPLAY_IO_TIMEOUT, async {
            let mut stream = self.client.connect().await.map_err(|_| unavailable)?;
            write_command(
                &mut stream,
                &[b"EVAL", SCRIPT.as_bytes(), b"1", key.as_bytes(), b"60"],
            )
            .await
            .map_err(|_| unavailable)?;
            let mut reader = BufReader::new(stream);
            if read_line(&mut reader).await.map_err(|_| unavailable)? != b"*2" {
                return Err(unavailable);
            }
            let count = read_line(&mut reader).await.map_err(|_| unavailable)?;
            let ttl = read_line(&mut reader).await.map_err(|_| unavailable)?;
            let integer = |value: &[u8]| -> Result<u64, HttpQuotaAdmissionError> {
                std::str::from_utf8(value)
                    .map_err(|_| unavailable)?
                    .strip_prefix(':')
                    .ok_or(unavailable)?
                    .parse()
                    .map_err(|_| unavailable)
            };
            let count = integer(&count)?;
            let ttl = integer(&ttl)?;
            if count == 0 || ttl > 60 {
                return Err(unavailable);
            }
            Ok((count, ttl))
        })
        .await
        .map_err(|_| unavailable)??;
        if count <= self.calls_per_minute {
            Ok(())
        } else {
            Err(HttpQuotaAdmissionError::Exceeded { reset_in_seconds })
        }
    }
}
