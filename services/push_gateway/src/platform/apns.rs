use std::{fmt, sync::Arc, sync::Mutex, time::Duration};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use collaboration_domain::{PushEndpointGeneration, PushWakePayload};
use p256::{
    ecdsa::{Signature, SigningKey, signature::Signer},
    pkcs8::DecodePrivateKey,
};
use reqwest::{StatusCode, header::AUTHORIZATION, header::CONTENT_TYPE};
use serde::Deserialize;
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{ApprovedPushProfile, grant::ApnsDeviceToken};
use crate::executor::{PushDeliveryRequest, PushProvider, PushProviderError, PushProviderOutcome};

pub const APNS_RECONNECT_BODY: &[u8] =
    br#"{"aps":{"alert":{"body":"Reconnect to your relay now"},"mutable-content":1}}"#;
const APNS_PRODUCTION_URL: &str = "https://api.push.apple.com";
const APNS_SANDBOX_URL: &str = "https://api.sandbox.push.apple.com";
const MAX_APNS_ERROR_BODY_BYTES: usize = 4_096;
const MAX_APNS_IDENTIFIER_BYTES: usize = 128;
const MAX_APNS_TOPIC_BYTES: usize = 255;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ApnsDeliveryAttempt {
    request_id: Uuid,
    expires_at_seconds: i64,
}

impl fmt::Debug for ApnsDeliveryAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApnsDeliveryAttempt")
            .field("expires_at_seconds", &self.expires_at_seconds)
            .finish_non_exhaustive()
    }
}

impl ApnsDeliveryAttempt {
    pub fn new(request_id: Uuid, expires_at_seconds: i64) -> Result<Self, ApnsRequestError> {
        if request_id.is_nil() || expires_at_seconds <= 0 {
            return Err(ApnsRequestError);
        }
        Ok(Self {
            request_id,
            expires_at_seconds,
        })
    }

    pub const fn request_id(self) -> Uuid {
        self.request_id
    }

    pub const fn expires_at_seconds(self) -> i64 {
        self.expires_at_seconds
    }

    pub const fn body(self) -> &'static [u8] {
        APNS_RECONNECT_BODY
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid APNs delivery request")]
pub struct ApnsRequestError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApnsDeliveryOutcome {
    Accepted,
    InvalidEndpoint {
        unregistered_at_seconds: Option<i64>,
    },
    Retry {
        retry_after_seconds: Option<u64>,
    },
    RefreshCredential,
    ConfigurationFault,
    PermanentRequestFault,
}

pub fn classify_apns_response(
    status: u16,
    reason: Option<&str>,
    timestamp: Option<i64>,
    retry_after_seconds: Option<u64>,
) -> ApnsDeliveryOutcome {
    match (status, reason) {
        (200, _) => ApnsDeliveryOutcome::Accepted,
        (410, Some("Unregistered")) => ApnsDeliveryOutcome::InvalidEndpoint {
            unregistered_at_seconds: timestamp.filter(|timestamp| *timestamp >= 0),
        },
        (400, Some("BadDeviceToken" | "DeviceTokenNotForTopic")) => {
            ApnsDeliveryOutcome::InvalidEndpoint {
                unregistered_at_seconds: None,
            }
        }
        (403, Some("ExpiredProviderToken")) => ApnsDeliveryOutcome::RefreshCredential,
        (403, _) | (429, Some("TooManyProviderTokenUpdates")) => {
            ApnsDeliveryOutcome::ConfigurationFault
        }
        (429 | 500 | 503, _)
        | (
            _,
            Some(
                "IdleTimeout"
                | "InternalServerError"
                | "ServiceUnavailable"
                | "Shutdown"
                | "TooManyRequests",
            ),
        ) => ApnsDeliveryOutcome::Retry {
            retry_after_seconds: retry_after_seconds.map(|seconds| seconds.clamp(1, 3_600)),
        },
        _ => ApnsDeliveryOutcome::PermanentRequestFault,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ApnsTransportError {
    #[error("APNs transport is unavailable")]
    Unavailable,
    #[error("APNs profile is not configured")]
    NotConfigured,
}

#[async_trait]
pub trait ApnsTransport: Send + Sync {
    async fn send(
        &self,
        profile: ApprovedPushProfile,
        endpoint: &ApnsDeviceToken,
        attempt: ApnsDeliveryAttempt,
    ) -> Result<ApnsDeliveryOutcome, ApnsTransportError>;

    fn refresh_credential(&self, _profile: ApprovedPushProfile) -> Result<(), ApnsTransportError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ApnsEndpointAuthorityError {
    #[error("APNs endpoint authority rejected the request")]
    Rejected,
    #[error("APNs endpoint authority is unavailable")]
    Unavailable,
}

pub struct AuthorizedApnsEndpoint {
    profile: ApprovedPushProfile,
    token: ApnsDeviceToken,
    endpoint_generation: PushEndpointGeneration,
}

impl AuthorizedApnsEndpoint {
    pub(crate) const fn new(
        profile: ApprovedPushProfile,
        token: ApnsDeviceToken,
        endpoint_generation: PushEndpointGeneration,
    ) -> Self {
        Self {
            profile,
            token,
            endpoint_generation,
        }
    }
}

impl fmt::Debug for AuthorizedApnsEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizedApnsEndpoint")
            .field("profile", &self.profile)
            .field("endpoint_generation", &self.endpoint_generation)
            .finish_non_exhaustive()
    }
}

#[async_trait]
pub trait ApnsEndpointAuthority: Send + Sync {
    async fn resolve(
        &self,
        request: &PushDeliveryRequest,
    ) -> Result<AuthorizedApnsEndpoint, ApnsEndpointAuthorityError>;
}

pub struct ApnsProvider<A, T> {
    authority: A,
    transport: T,
}

impl<A, T> ApnsProvider<A, T> {
    pub const fn new(authority: A, transport: T) -> Self {
        Self {
            authority,
            transport,
        }
    }
}

#[async_trait]
impl<A, T> PushProvider for ApnsProvider<A, T>
where
    A: ApnsEndpointAuthority,
    T: ApnsTransport,
{
    async fn deliver(
        &self,
        request: PushDeliveryRequest,
    ) -> Result<PushProviderOutcome, PushProviderError> {
        if request.payload() != PushWakePayload::Reconnect {
            return Ok(PushProviderOutcome::PermanentRequestFault);
        }
        let expiration_seconds = request.expires_at_millis() / 1_000;
        let expiration_seconds = i64::try_from(expiration_seconds)
            .ok()
            .filter(|expiration| *expiration > 0)
            .ok_or(PushProviderError)?;
        let attempt = ApnsDeliveryAttempt::new(request.request_id(), expiration_seconds)
            .map_err(|_| PushProviderError)?;
        let endpoint = match self.authority.resolve(&request).await {
            Ok(endpoint) => endpoint,
            Err(ApnsEndpointAuthorityError::Rejected) => {
                return Ok(PushProviderOutcome::PermanentRequestFault);
            }
            Err(ApnsEndpointAuthorityError::Unavailable) => return Err(PushProviderError),
        };
        let mut outcome = self
            .transport
            .send(endpoint.profile, &endpoint.token, attempt)
            .await
            .map_err(|error| match error {
                ApnsTransportError::Unavailable | ApnsTransportError::NotConfigured => {
                    PushProviderError
                }
            })?;
        if outcome == ApnsDeliveryOutcome::RefreshCredential {
            self.transport
                .refresh_credential(endpoint.profile)
                .map_err(|_| PushProviderError)?;
            outcome = self
                .transport
                .send(endpoint.profile, &endpoint.token, attempt)
                .await
                .map_err(|_| PushProviderError)?;
            if outcome == ApnsDeliveryOutcome::RefreshCredential {
                outcome = ApnsDeliveryOutcome::ConfigurationFault;
            }
        }
        Ok(map_apns_outcome(outcome, endpoint.endpoint_generation))
    }
}

fn map_apns_outcome(
    outcome: ApnsDeliveryOutcome,
    endpoint_generation: PushEndpointGeneration,
) -> PushProviderOutcome {
    match outcome {
        ApnsDeliveryOutcome::Accepted => PushProviderOutcome::Accepted,
        ApnsDeliveryOutcome::InvalidEndpoint {
            unregistered_at_seconds,
        } => PushProviderOutcome::InvalidEndpoint {
            endpoint_generation,
            invalid_at_millis: unregistered_at_seconds
                .and_then(|seconds| u64::try_from(seconds).ok())
                .and_then(|seconds| seconds.checked_mul(1_000)),
        },
        ApnsDeliveryOutcome::Retry {
            retry_after_seconds,
        } => PushProviderOutcome::Retry {
            retry_after_millis: retry_after_seconds
                .map(|seconds| seconds.clamp(1, 3_600))
                .and_then(|seconds| seconds.checked_mul(1_000)),
        },
        ApnsDeliveryOutcome::RefreshCredential | ApnsDeliveryOutcome::ConfigurationFault => {
            PushProviderOutcome::ConfigurationFault
        }
        ApnsDeliveryOutcome::PermanentRequestFault => PushProviderOutcome::PermanentRequestFault,
    }
}

pub trait ApnsProviderCredential: Send + Sync {
    fn bearer_token(&self, now_seconds: i64) -> Result<Zeroizing<String>, ApnsCredentialError>;
    fn refresh(&self) -> Result<(), ApnsCredentialError>;
}

struct CachedProviderToken {
    token: Zeroizing<String>,
    issued_at_seconds: i64,
}

pub struct ApnsTokenCredential {
    signing_key: SigningKey,
    key_id: String,
    team_id: String,
    cached: Mutex<Option<CachedProviderToken>>,
}

impl ApnsTokenCredential {
    pub fn new(
        p8: &[u8],
        key_id: impl Into<String>,
        team_id: impl Into<String>,
    ) -> Result<Self, ApnsCredentialError> {
        let key_id = key_id.into();
        let team_id = team_id.into();
        if !valid_identifier(&key_id) || !valid_identifier(&team_id) {
            return Err(ApnsCredentialError);
        }
        let pem = std::str::from_utf8(p8).map_err(|_| ApnsCredentialError)?;
        let signing_key = SigningKey::from_pkcs8_pem(pem).map_err(|_| ApnsCredentialError)?;
        Ok(Self {
            signing_key,
            key_id,
            team_id,
            cached: Mutex::new(None),
        })
    }
}

impl ApnsProviderCredential for ApnsTokenCredential {
    fn bearer_token(&self, now_seconds: i64) -> Result<Zeroizing<String>, ApnsCredentialError> {
        if now_seconds <= 0 {
            return Err(ApnsCredentialError);
        }
        let mut cached = self.cached.lock().map_err(|_| ApnsCredentialError)?;
        if let Some(cached) = cached
            .as_ref()
            .filter(|cached| now_seconds.saturating_sub(cached.issued_at_seconds) < 50 * 60)
        {
            return Ok(cached.token.clone());
        }
        let header = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({"alg":"ES256","kid":self.key_id}))
                .map_err(|_| ApnsCredentialError)?,
        );
        let claims = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({"iss":self.team_id,"iat":now_seconds}))
                .map_err(|_| ApnsCredentialError)?,
        );
        let signing_input = Zeroizing::new(format!("{header}.{claims}"));
        let signature: Signature = self.signing_key.sign(signing_input.as_bytes());
        let token = Zeroizing::new(format!(
            "{}.{}",
            signing_input.as_str(),
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        ));
        *cached = Some(CachedProviderToken {
            token: token.clone(),
            issued_at_seconds: now_seconds,
        });
        Ok(token)
    }

    fn refresh(&self) -> Result<(), ApnsCredentialError> {
        *self.cached.lock().map_err(|_| ApnsCredentialError)? = None;
        Ok(())
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_APNS_IDENTIFIER_BYTES
        && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid APNs provider credential")]
pub struct ApnsCredentialError;

struct ApnsHttpProfile {
    topic: String,
    credential: Arc<dyn ApnsProviderCredential>,
}

pub struct ApnsHttpTransport {
    client: reqwest::Client,
    production: Option<ApnsHttpProfile>,
    sandbox: Option<ApnsHttpProfile>,
}

impl ApnsHttpTransport {
    pub fn new(
        production: Option<(String, Arc<dyn ApnsProviderCredential>)>,
        sandbox: Option<(String, Arc<dyn ApnsProviderCredential>)>,
    ) -> Result<Self, ApnsTransportConfigurationError> {
        if production.is_none() && sandbox.is_none() {
            return Err(ApnsTransportConfigurationError);
        }
        let production = production
            .map(|(topic, credential)| profile(topic, credential))
            .transpose()?;
        let sandbox = sandbox
            .map(|(topic, credential)| profile(topic, credential))
            .transpose()?;
        if production
            .as_ref()
            .zip(sandbox.as_ref())
            .is_some_and(|(production, sandbox)| {
                production.topic == sandbox.topic
                    || Arc::ptr_eq(&production.credential, &sandbox.credential)
            })
        {
            return Err(ApnsTransportConfigurationError);
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .redirect_policy(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|_| ApnsTransportConfigurationError)?;
        Ok(Self {
            client,
            production,
            sandbox,
        })
    }

    fn profile(
        &self,
        profile: ApprovedPushProfile,
    ) -> Result<(&ApnsHttpProfile, &'static str), ApnsTransportError> {
        match profile {
            ApprovedPushProfile::BuzzIosProduction => self
                .production
                .as_ref()
                .map(|profile| (profile, APNS_PRODUCTION_URL))
                .ok_or(ApnsTransportError::NotConfigured),
            ApprovedPushProfile::BuzzIosSandbox => self
                .sandbox
                .as_ref()
                .map(|profile| (profile, APNS_SANDBOX_URL))
                .ok_or(ApnsTransportError::NotConfigured),
        }
    }
}

fn profile(
    topic: String,
    credential: Arc<dyn ApnsProviderCredential>,
) -> Result<ApnsHttpProfile, ApnsTransportConfigurationError> {
    if topic.is_empty()
        || topic.len() > MAX_APNS_TOPIC_BYTES
        || topic.chars().any(char::is_control)
        || topic.trim() != topic
    {
        return Err(ApnsTransportConfigurationError);
    }
    Ok(ApnsHttpProfile { topic, credential })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid APNs transport configuration")]
pub struct ApnsTransportConfigurationError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApnsErrorBody {
    reason: Option<String>,
    timestamp: Option<i64>,
}

#[async_trait]
impl ApnsTransport for ApnsHttpTransport {
    async fn send(
        &self,
        profile: ApprovedPushProfile,
        endpoint: &ApnsDeviceToken,
        attempt: ApnsDeliveryAttempt,
    ) -> Result<ApnsDeliveryOutcome, ApnsTransportError> {
        let (profile_configuration, base_url) = self.profile(profile)?;
        let now_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|duration| i64::try_from(duration.as_secs()).ok())
            .ok_or(ApnsTransportError::Unavailable)?;
        let provider_token = profile_configuration
            .credential
            .bearer_token(now_seconds)
            .map_err(|_| ApnsTransportError::Unavailable)?;
        let authorization = Zeroizing::new(format!("bearer {}", provider_token.as_str()));
        let endpoint = endpoint.encode_hex();
        let mut response = self
            .client
            .post(format!("{base_url}/3/device/{}", endpoint.as_str()))
            .header(AUTHORIZATION, authorization.as_str())
            .header(CONTENT_TYPE, "application/json")
            .header("apns-id", attempt.request_id().to_string())
            .header("apns-topic", &profile_configuration.topic)
            .header("apns-push-type", "alert")
            .header("apns-priority", "10")
            .header("apns-expiration", attempt.expires_at_seconds().to_string())
            .body(attempt.body())
            .send()
            .await
            .map_err(|_| ApnsTransportError::Unavailable)?;
        if response.status() == StatusCode::OK {
            return Ok(ApnsDeliveryOutcome::Accepted);
        }
        let status = response.status().as_u16();
        let retry_after_seconds = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .map(|seconds| seconds.clamp(1, 3_600));
        if response
            .content_length()
            .is_some_and(|length| length > MAX_APNS_ERROR_BODY_BYTES as u64)
        {
            return Ok(ApnsDeliveryOutcome::PermanentRequestFault);
        }
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| ApnsTransportError::Unavailable)?
        {
            if body.len().saturating_add(chunk.len()) > MAX_APNS_ERROR_BODY_BYTES {
                return Ok(ApnsDeliveryOutcome::PermanentRequestFault);
            }
            body.extend_from_slice(&chunk);
        }
        let detail = serde_json::from_slice::<ApnsErrorBody>(&body).ok();
        Ok(classify_apns_response(
            status,
            detail.as_ref().and_then(|detail| detail.reason.as_deref()),
            detail.as_ref().and_then(|detail| detail.timestamp),
            retry_after_seconds,
        ))
    }

    fn refresh_credential(&self, profile: ApprovedPushProfile) -> Result<(), ApnsTransportError> {
        let (profile, _) = self.profile(profile)?;
        profile
            .credential
            .refresh()
            .map_err(|_| ApnsTransportError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use collaboration_domain::{
        PushCapabilityReference, PushEndpointGeneration, PushLeaseGeneration,
    };

    use super::*;

    struct FakeAuthority {
        endpoint: AuthorizedApnsEndpoint,
    }

    struct FakeCredential;

    impl ApnsProviderCredential for FakeCredential {
        fn bearer_token(
            &self,
            _now_seconds: i64,
        ) -> Result<Zeroizing<String>, ApnsCredentialError> {
            Ok(Zeroizing::new("provider-token".to_owned()))
        }

        fn refresh(&self) -> Result<(), ApnsCredentialError> {
            Ok(())
        }
    }

    #[async_trait]
    impl ApnsEndpointAuthority for FakeAuthority {
        async fn resolve(
            &self,
            _request: &PushDeliveryRequest,
        ) -> Result<AuthorizedApnsEndpoint, ApnsEndpointAuthorityError> {
            Ok(AuthorizedApnsEndpoint::new(
                self.endpoint.profile,
                self.endpoint.token.clone(),
                self.endpoint.endpoint_generation,
            ))
        }
    }

    #[derive(Default)]
    struct FakeTransport {
        outcomes: Mutex<Vec<ApnsDeliveryOutcome>>,
        attempts: Mutex<Vec<(ApprovedPushProfile, ApnsDeliveryAttempt)>>,
        refreshes: Mutex<u32>,
    }

    #[async_trait]
    impl ApnsTransport for FakeTransport {
        async fn send(
            &self,
            profile: ApprovedPushProfile,
            _endpoint: &ApnsDeviceToken,
            attempt: ApnsDeliveryAttempt,
        ) -> Result<ApnsDeliveryOutcome, ApnsTransportError> {
            self.attempts
                .lock()
                .expect("attempt lock")
                .push((profile, attempt));
            let mut outcomes = self.outcomes.lock().expect("outcome lock");
            if outcomes.is_empty() {
                return Ok(ApnsDeliveryOutcome::Accepted);
            }
            Ok(outcomes.remove(0))
        }

        fn refresh_credential(
            &self,
            _profile: ApprovedPushProfile,
        ) -> Result<(), ApnsTransportError> {
            *self.refreshes.lock().expect("refresh lock") += 1;
            Ok(())
        }
    }

    fn request(expires_at_millis: u64) -> PushDeliveryRequest {
        PushDeliveryRequest::for_test(
            Uuid::from_u128(5),
            PushLeaseGeneration::new(2).expect("lease generation"),
            PushEndpointGeneration::new(3).expect("endpoint generation"),
            PushCapabilityReference::from_digest([4; 32]).expect("capability"),
            expires_at_millis,
        )
    }

    fn provider(outcomes: Vec<ApnsDeliveryOutcome>) -> ApnsProvider<FakeAuthority, FakeTransport> {
        ApnsProvider::new(
            FakeAuthority {
                endpoint: AuthorizedApnsEndpoint::new(
                    ApprovedPushProfile::BuzzIosSandbox,
                    ApnsDeviceToken::from_hex(&"ab".repeat(32)).expect("token"),
                    PushEndpointGeneration::new(3).expect("endpoint generation"),
                ),
            },
            FakeTransport {
                outcomes: Mutex::new(outcomes),
                ..FakeTransport::default()
            },
        )
    }

    #[tokio::test]
    async fn provider_uses_fixed_body_profile_and_bounded_expiry() {
        let provider = provider(Vec::new());
        assert_eq!(
            provider.deliver(request(9_999)).await.expect("delivery"),
            PushProviderOutcome::Accepted
        );
        let attempts = provider.transport.attempts.lock().expect("attempt lock");
        assert_eq!(attempts.as_slice().len(), 1);
        assert_eq!(attempts[0].0, ApprovedPushProfile::BuzzIosSandbox);
        assert_eq!(attempts[0].1.expires_at_seconds(), 9);
        assert_eq!(attempts[0].1.body(), APNS_RECONNECT_BODY);
        assert_eq!(
            APNS_RECONNECT_BODY,
            br#"{"aps":{"alert":{"body":"Reconnect to your relay now"},"mutable-content":1}}"#
        );
    }

    #[tokio::test]
    async fn provider_refreshes_once_and_sanitizes_all_outcomes() {
        let provider = provider(vec![
            ApnsDeliveryOutcome::RefreshCredential,
            ApnsDeliveryOutcome::Retry {
                retry_after_seconds: Some(9_999),
            },
        ]);
        assert_eq!(
            provider.deliver(request(20_000)).await.expect("delivery"),
            PushProviderOutcome::Retry {
                retry_after_millis: Some(3_600_000)
            }
        );
        assert_eq!(
            *provider.transport.refreshes.lock().expect("refresh lock"),
            1
        );

        assert_eq!(
            map_apns_outcome(
                ApnsDeliveryOutcome::InvalidEndpoint {
                    unregistered_at_seconds: Some(7),
                },
                PushEndpointGeneration::new(3).expect("endpoint generation")
            ),
            PushProviderOutcome::InvalidEndpoint {
                endpoint_generation: PushEndpointGeneration::new(3).expect("endpoint generation"),
                invalid_at_millis: Some(7_000)
            }
        );
        assert_eq!(
            classify_apns_response(403, Some("InvalidProviderToken"), None, None),
            ApnsDeliveryOutcome::ConfigurationFault
        );
        assert_eq!(
            classify_apns_response(400, Some("BadTopic"), None, None),
            ApnsDeliveryOutcome::PermanentRequestFault
        );
        assert_eq!(
            classify_apns_response(410, Some("Unregistered"), Some(8), None),
            ApnsDeliveryOutcome::InvalidEndpoint {
                unregistered_at_seconds: Some(8)
            }
        );
    }

    #[test]
    fn production_and_sandbox_configuration_cannot_fall_through() {
        let shared: Arc<dyn ApnsProviderCredential> = Arc::new(FakeCredential);
        assert!(
            ApnsHttpTransport::new(
                Some(("production.topic".to_owned(), shared.clone())),
                Some(("sandbox.topic".to_owned(), shared)),
            )
            .is_err()
        );
        assert!(
            ApnsHttpTransport::new(
                Some(("shared.topic".to_owned(), Arc::new(FakeCredential))),
                Some(("shared.topic".to_owned(), Arc::new(FakeCredential))),
            )
            .is_err()
        );
        assert!(
            ApnsHttpTransport::new(
                Some(("production.topic".to_owned(), Arc::new(FakeCredential))),
                Some(("sandbox.topic".to_owned(), Arc::new(FakeCredential))),
            )
            .is_ok()
        );
    }
}
