use std::{collections::BTreeSet, error::Error, fmt};

use nostr::{
    Keys, PublicKey as NostrPublicKey, SecretKey,
    hashes::Hash as _,
    nips::nip44::{self, Version as Nip44Version},
    util::{generate_shared_key, hkdf},
};
use serde::{Deserialize, Serialize};
use url::{Url, form_urlencoded};
use zeroize::{Zeroize, Zeroizing};

pub const NIP_AB_VERSION: u16 = 1;
pub const NIP_AB_SESSION_MILLIS: u64 = 120_000;
pub const MAX_NIP_AB_QR_BYTES: usize = 2_048;
pub const MAX_NIP_AB_RELAYS: usize = 4;
pub const MAX_NIP_AB_RELAY_BYTES: usize = 512;
pub const MAX_NIP_AB_PAYLOAD_BYTES: usize = 64 * 1_024;
pub const MAX_NIP_AB_CIPHERTEXT_BYTES: usize = 88_000;
pub const MAX_NIP_AB_PROCESSED_FRAMES: usize = 16;
const MAX_NIP_AB_CLOCK_SKEW_MILLIS: u64 = 5_000;
const SESSION_ID_INFO: &[u8] = b"nostr-pair-session-id";
const SAS_INFO: &[u8] = b"nostr-pair-sas-v1";
const TRANSCRIPT_INFO: &[u8] = b"nostr-pair-transcript-v1";

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PairingFrameId([u8; 32]);

impl PairingFrameId {
    pub fn new(value: [u8; 32]) -> Result<Self, PairingError> {
        if value == [0; 32] {
            return Err(PairingError::InvalidFrame);
        }
        Ok(Self(value))
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for PairingFrameId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PairingFrameId([redacted])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairingRelayUrl(String);

impl PairingRelayUrl {
    pub fn parse(value: impl Into<String>) -> Result<Self, PairingError> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_NIP_AB_RELAY_BYTES {
            return Err(PairingError::InvalidQr);
        }
        let parsed = Url::parse(&value).map_err(|_| PairingError::InvalidQr)?;
        if !matches!(parsed.scheme(), "ws" | "wss")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.fragment().is_some()
        {
            return Err(PairingError::InvalidQr);
        }
        Ok(Self(parsed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub struct PairingQr {
    source_public_key: NostrPublicKey,
    session_secret: [u8; 32],
    relays: Vec<PairingRelayUrl>,
    version: u16,
}

impl PairingQr {
    pub fn new(
        source_public_key: NostrPublicKey,
        session_secret: [u8; 32],
        relays: Vec<PairingRelayUrl>,
    ) -> Result<Self, PairingError> {
        if session_secret == [0; 32]
            || relays.is_empty()
            || relays.len() > MAX_NIP_AB_RELAYS
            || relays
                .iter()
                .enumerate()
                .any(|(index, relay)| relays[..index].contains(relay))
        {
            return Err(PairingError::InvalidQr);
        }
        Ok(Self {
            source_public_key,
            session_secret,
            relays,
            version: NIP_AB_VERSION,
        })
    }

    pub fn parse(uri: &str) -> Result<Self, PairingError> {
        if uri.is_empty() || uri.len() > MAX_NIP_AB_QR_BYTES {
            return Err(PairingError::InvalidQr);
        }
        let parsed = Url::parse(uri).map_err(|_| PairingError::InvalidQr)?;
        if parsed.scheme() != "nostrpair"
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.port().is_some()
            || !matches!(parsed.path(), "" | "/")
            || parsed.fragment().is_some()
        {
            return Err(PairingError::InvalidQr);
        }
        let public_key_hex = parsed.host_str().ok_or(PairingError::InvalidQr)?;
        if public_key_hex.len() != 64 || !public_key_hex.bytes().all(is_lower_hex) {
            return Err(PairingError::InvalidQr);
        }
        let source_public_key =
            NostrPublicKey::from_hex(public_key_hex).map_err(|_| PairingError::InvalidQr)?;
        let mut session_secret = None;
        let mut relays = Vec::new();
        let mut version = None;
        for (key, value) in parsed.query_pairs() {
            match key.as_ref() {
                "secret" if session_secret.is_none() => {
                    let value = value.as_ref();
                    if value.len() != 64 || !value.bytes().all(is_lower_hex) {
                        return Err(PairingError::InvalidQr);
                    }
                    let decoded = hex::decode(value).map_err(|_| PairingError::InvalidQr)?;
                    session_secret = Some(decoded.try_into().map_err(|_| PairingError::InvalidQr)?);
                }
                "secret" => return Err(PairingError::InvalidQr),
                "relay" => relays.push(PairingRelayUrl::parse(value.into_owned())?),
                "v" if version.is_none() => {
                    version = Some(
                        value
                            .parse::<u16>()
                            .map_err(|_| PairingError::UnsupportedVersion)?,
                    );
                }
                "v" => return Err(PairingError::InvalidQr),
                _ => {}
            }
        }
        let version = version.unwrap_or(NIP_AB_VERSION);
        if version != NIP_AB_VERSION {
            return Err(PairingError::UnsupportedVersion);
        }
        let mut qr = Self::new(
            source_public_key,
            session_secret.ok_or(PairingError::InvalidQr)?,
            relays,
        )?;
        qr.version = version;
        Ok(qr)
    }

    pub fn encode(&self) -> Result<String, PairingError> {
        let mut query = form_urlencoded::Serializer::new(String::new());
        query.append_pair("secret", &hex::encode(self.session_secret));
        for relay in &self.relays {
            query.append_pair("relay", relay.as_str());
        }
        query.append_pair("v", &self.version.to_string());
        let uri = format!(
            "nostrpair://{}?{}",
            self.source_public_key.to_hex(),
            query.finish()
        );
        if uri.len() > MAX_NIP_AB_QR_BYTES {
            return Err(PairingError::InvalidQr);
        }
        Ok(uri)
    }

    pub const fn source_public_key(&self) -> NostrPublicKey {
        self.source_public_key
    }

    pub fn relays(&self) -> &[PairingRelayUrl] {
        &self.relays
    }

    pub const fn version(&self) -> u16 {
        self.version
    }
}

impl Clone for PairingQr {
    fn clone(&self) -> Self {
        Self {
            source_public_key: self.source_public_key,
            session_secret: self.session_secret,
            relays: self.relays.clone(),
            version: self.version,
        }
    }
}

impl fmt::Debug for PairingQr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingQr")
            .field("source_public_key", &self.source_public_key)
            .field("session_secret", &"[redacted]")
            .field("relays", &self.relays)
            .field("version", &self.version)
            .finish()
    }
}

impl Drop for PairingQr {
    fn drop(&mut self) {
        self.session_secret.zeroize();
    }
}

pub struct PairingEncryptedFrame {
    frame_id: PairingFrameId,
    sender_public_key: NostrPublicKey,
    recipient_public_key: NostrPublicKey,
    created_at_millis: u64,
    ciphertext: String,
}

impl PairingEncryptedFrame {
    pub fn from_transport(
        frame_id: PairingFrameId,
        sender_public_key: NostrPublicKey,
        recipient_public_key: NostrPublicKey,
        created_at_millis: u64,
        ciphertext: String,
    ) -> Result<Self, PairingError> {
        if created_at_millis == 0
            || ciphertext.is_empty()
            || ciphertext.len() > MAX_NIP_AB_CIPHERTEXT_BYTES
        {
            return Err(PairingError::InvalidFrame);
        }
        Ok(Self {
            frame_id,
            sender_public_key,
            recipient_public_key,
            created_at_millis,
            ciphertext,
        })
    }

    pub const fn frame_id(&self) -> PairingFrameId {
        self.frame_id
    }

    pub const fn sender_public_key(&self) -> NostrPublicKey {
        self.sender_public_key
    }

    pub const fn recipient_public_key(&self) -> NostrPublicKey {
        self.recipient_public_key
    }

    pub const fn created_at_millis(&self) -> u64 {
        self.created_at_millis
    }

    pub fn ciphertext(&self) -> &str {
        &self.ciphertext
    }
}

impl fmt::Debug for PairingEncryptedFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingEncryptedFrame")
            .field("frame_id", &self.frame_id)
            .field("sender_public_key", &self.sender_public_key)
            .field("recipient_public_key", &self.recipient_public_key)
            .field("created_at_millis", &self.created_at_millis)
            .field("ciphertext_bytes", &self.ciphertext.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingPayloadType {
    Nsec,
    Bunker,
    Connect,
    Custom,
}

pub struct PairingPayload {
    payload_type: PairingPayloadType,
    secret: Zeroizing<String>,
}

impl PairingPayload {
    pub fn new(
        payload_type: PairingPayloadType,
        secret: Zeroizing<String>,
    ) -> Result<Self, PairingError> {
        if secret.is_empty() || secret.len() > MAX_NIP_AB_PAYLOAD_BYTES {
            return Err(PairingError::InvalidPayload);
        }
        Ok(Self {
            payload_type,
            secret,
        })
    }

    pub const fn payload_type(&self) -> PairingPayloadType {
        self.payload_type
    }

    pub fn secret(&self) -> &str {
        self.secret.as_str()
    }

    pub fn into_secret(self) -> Zeroizing<String> {
        self.secret
    }
}

impl fmt::Debug for PairingPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingPayload")
            .field("payload_type", &self.payload_type)
            .field("secret_bytes", &self.secret.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairingRole {
    Source,
    Target,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairingSessionState {
    WaitingOffer,
    AwaitingSourceConfirmation,
    AwaitingSasConfirm,
    AwaitingTargetConfirmation,
    Transferring,
    PayloadSent,
    PayloadReceived,
    Completed,
    Aborted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingAbortReason {
    SasMismatch,
    UserDenied,
    Timeout,
    ProtocolError,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum OutboundMessage<'a> {
    Offer {
        version: u16,
        session_id: String,
    },
    SasConfirm {
        transcript_hash: String,
    },
    Payload {
        payload_type: PairingPayloadType,
        payload: &'a str,
    },
    Complete {
        success: bool,
    },
    Abort {
        reason: PairingAbortReason,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum InboundMessage {
    Offer {
        #[serde(default = "default_nip_ab_version")]
        version: u16,
        session_id: String,
    },
    SasConfirm {
        transcript_hash: String,
    },
    Payload {
        payload_type: PairingPayloadType,
        payload: String,
    },
    Complete {
        success: bool,
    },
    Abort {
        reason: PairingAbortReason,
    },
}

pub struct PairingSession {
    role: PairingRole,
    state: PairingSessionState,
    ephemeral_secret: [u8; 32],
    local_public_key: NostrPublicKey,
    peer_public_key: Option<NostrPublicKey>,
    session_secret: [u8; 32],
    session_id: [u8; 32],
    sas_input: Option<[u8; 32]>,
    sas_code: Option<u32>,
    created_at_millis: u64,
    expires_at_millis: u64,
    processed_frames: BTreeSet<PairingFrameId>,
}

impl PairingSession {
    pub fn new_source(
        ephemeral_secret: [u8; 32],
        session_secret: [u8; 32],
        relays: Vec<PairingRelayUrl>,
        created_at_millis: u64,
    ) -> Result<(Self, PairingQr), PairingError> {
        if session_secret == [0; 32] {
            return Err(PairingError::InvalidSecret);
        }
        let local_public_key = public_key(ephemeral_secret)?;
        let expires_at_millis = created_at_millis
            .checked_add(NIP_AB_SESSION_MILLIS)
            .ok_or(PairingError::InvalidTimestamp)?;
        if created_at_millis == 0 {
            return Err(PairingError::InvalidTimestamp);
        }
        let qr = PairingQr::new(local_public_key, session_secret, relays)?;
        Ok((
            Self {
                role: PairingRole::Source,
                state: PairingSessionState::WaitingOffer,
                ephemeral_secret,
                local_public_key,
                peer_public_key: None,
                session_secret,
                session_id: derive_session_id(&session_secret),
                sas_input: None,
                sas_code: None,
                created_at_millis,
                expires_at_millis,
                processed_frames: BTreeSet::new(),
            },
            qr,
        ))
    }

    pub fn new_target(
        qr: &PairingQr,
        ephemeral_secret: [u8; 32],
        created_at_millis: u64,
    ) -> Result<Self, PairingError> {
        if qr.version != NIP_AB_VERSION {
            return Err(PairingError::UnsupportedVersion);
        }
        if created_at_millis == 0 {
            return Err(PairingError::InvalidTimestamp);
        }
        let local_public_key = public_key(ephemeral_secret)?;
        let expires_at_millis = created_at_millis
            .checked_add(NIP_AB_SESSION_MILLIS)
            .ok_or(PairingError::InvalidTimestamp)?;
        let (sas_code, sas_input) =
            derive_sas(ephemeral_secret, qr.source_public_key, qr.session_secret)?;
        Ok(Self {
            role: PairingRole::Target,
            state: PairingSessionState::AwaitingSasConfirm,
            ephemeral_secret,
            local_public_key,
            peer_public_key: Some(qr.source_public_key),
            session_secret: qr.session_secret,
            session_id: derive_session_id(&qr.session_secret),
            sas_input: Some(sas_input),
            sas_code: Some(sas_code),
            created_at_millis,
            expires_at_millis,
            processed_frames: BTreeSet::new(),
        })
    }

    pub const fn role(&self) -> PairingRole {
        self.role
    }

    pub const fn state(&self) -> PairingSessionState {
        self.state
    }

    pub const fn local_public_key(&self) -> NostrPublicKey {
        self.local_public_key
    }

    pub const fn peer_public_key(&self) -> Option<NostrPublicKey> {
        self.peer_public_key
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    pub fn sas(&self) -> Option<String> {
        self.sas_code.map(|code| format!("{code:06}"))
    }

    pub fn offer(
        &self,
        frame_id: PairingFrameId,
        now_millis: u64,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.require(
            PairingRole::Target,
            PairingSessionState::AwaitingSasConfirm,
            now_millis,
        )?;
        self.encrypt(
            frame_id,
            now_millis,
            &OutboundMessage::Offer {
                version: NIP_AB_VERSION,
                session_id: hex::encode(self.session_id),
            },
        )
    }

    pub fn receive_offer(
        &mut self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
    ) -> Result<String, PairingError> {
        self.check_replay(frame.frame_id)?;
        self.require(
            PairingRole::Source,
            PairingSessionState::WaitingOffer,
            now_millis,
        )?;
        self.validate_frame(frame, now_millis, false)?;
        let message = self.decrypt(frame)?;
        let InboundMessage::Offer {
            version,
            session_id,
        } = message
        else {
            return Err(PairingError::UnexpectedMessage);
        };
        if version != NIP_AB_VERSION {
            return Err(PairingError::UnsupportedVersion);
        }
        let received = decode_32(&session_id).ok_or(PairingError::InvalidSession)?;
        if !constant_time_eq(&received, &self.session_id) {
            return Err(PairingError::InvalidSession);
        }
        let (sas_code, sas_input) = derive_sas(
            self.ephemeral_secret,
            frame.sender_public_key,
            self.session_secret,
        )?;
        self.record(frame.frame_id)?;
        self.peer_public_key = Some(frame.sender_public_key);
        self.sas_code = Some(sas_code);
        self.sas_input = Some(sas_input);
        self.state = PairingSessionState::AwaitingSourceConfirmation;
        Ok(format!("{sas_code:06}"))
    }

    pub fn confirm_source_sas(
        &mut self,
        frame_id: PairingFrameId,
        now_millis: u64,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.require(
            PairingRole::Source,
            PairingSessionState::AwaitingSourceConfirmation,
            now_millis,
        )?;
        let transcript_hash = self.transcript_hash()?;
        let frame = self.encrypt(
            frame_id,
            now_millis,
            &OutboundMessage::SasConfirm {
                transcript_hash: hex::encode(transcript_hash),
            },
        )?;
        self.state = PairingSessionState::Transferring;
        Ok(frame)
    }

    pub fn receive_sas_confirm(
        &mut self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
    ) -> Result<(), PairingError> {
        self.check_replay(frame.frame_id)?;
        self.require(
            PairingRole::Target,
            PairingSessionState::AwaitingSasConfirm,
            now_millis,
        )?;
        self.validate_frame(frame, now_millis, true)?;
        let InboundMessage::SasConfirm { transcript_hash } = self.decrypt(frame)? else {
            return Err(PairingError::UnexpectedMessage);
        };
        let received = decode_32(&transcript_hash).ok_or(PairingError::TranscriptMismatch)?;
        if !constant_time_eq(&received, &self.transcript_hash()?) {
            return Err(PairingError::TranscriptMismatch);
        }
        self.record(frame.frame_id)?;
        self.state = PairingSessionState::AwaitingTargetConfirmation;
        Ok(())
    }

    pub fn confirm_target_sas(&mut self, now_millis: u64) -> Result<(), PairingError> {
        self.require(
            PairingRole::Target,
            PairingSessionState::AwaitingTargetConfirmation,
            now_millis,
        )?;
        self.state = PairingSessionState::Transferring;
        Ok(())
    }

    pub fn send_payload(
        &mut self,
        frame_id: PairingFrameId,
        payload: &PairingPayload,
        now_millis: u64,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.require(
            PairingRole::Source,
            PairingSessionState::Transferring,
            now_millis,
        )?;
        let frame = self.encrypt(
            frame_id,
            now_millis,
            &OutboundMessage::Payload {
                payload_type: payload.payload_type,
                payload: payload.secret(),
            },
        )?;
        self.state = PairingSessionState::PayloadSent;
        Ok(frame)
    }

    pub fn receive_payload(
        &mut self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
    ) -> Result<PairingPayload, PairingError> {
        self.check_replay(frame.frame_id)?;
        self.require(
            PairingRole::Target,
            PairingSessionState::Transferring,
            now_millis,
        )?;
        self.validate_frame(frame, now_millis, true)?;
        let InboundMessage::Payload {
            payload_type,
            mut payload,
        } = self.decrypt(frame)?
        else {
            return Err(PairingError::UnexpectedMessage);
        };
        if payload.is_empty() || payload.len() > MAX_NIP_AB_PAYLOAD_BYTES {
            payload.zeroize();
            return Err(PairingError::InvalidPayload);
        }
        self.record(frame.frame_id)?;
        self.state = PairingSessionState::PayloadReceived;
        Ok(PairingPayload {
            payload_type,
            secret: Zeroizing::new(payload),
        })
    }

    pub fn complete_target(
        &mut self,
        frame_id: PairingFrameId,
        now_millis: u64,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.require(
            PairingRole::Target,
            PairingSessionState::PayloadReceived,
            now_millis,
        )?;
        let frame = self.encrypt(
            frame_id,
            now_millis,
            &OutboundMessage::Complete { success: true },
        )?;
        self.state = PairingSessionState::Completed;
        Ok(frame)
    }

    pub fn receive_complete(
        &mut self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
    ) -> Result<(), PairingError> {
        self.check_replay(frame.frame_id)?;
        self.require(
            PairingRole::Source,
            PairingSessionState::PayloadSent,
            now_millis,
        )?;
        self.validate_frame(frame, now_millis, true)?;
        let InboundMessage::Complete { success } = self.decrypt(frame)? else {
            return Err(PairingError::UnexpectedMessage);
        };
        if !success {
            self.state = PairingSessionState::Aborted;
            return Err(PairingError::PeerAborted);
        }
        self.record(frame.frame_id)?;
        self.state = PairingSessionState::Completed;
        Ok(())
    }

    pub fn abort(
        &mut self,
        frame_id: PairingFrameId,
        reason: PairingAbortReason,
        now_millis: u64,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.check_active(now_millis)?;
        let frame = self.encrypt(frame_id, now_millis, &OutboundMessage::Abort { reason })?;
        self.state = PairingSessionState::Aborted;
        Ok(frame)
    }

    pub fn receive_abort(
        &mut self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
    ) -> Result<PairingAbortReason, PairingError> {
        self.check_replay(frame.frame_id)?;
        self.check_active(now_millis)?;
        self.validate_frame(frame, now_millis, true)?;
        let InboundMessage::Abort { reason } = self.decrypt(frame)? else {
            return Err(PairingError::UnexpectedMessage);
        };
        self.record(frame.frame_id)?;
        self.state = PairingSessionState::Aborted;
        Ok(reason)
    }

    fn require(
        &self,
        role: PairingRole,
        state: PairingSessionState,
        now_millis: u64,
    ) -> Result<(), PairingError> {
        self.check_active(now_millis)?;
        if self.role != role {
            return Err(PairingError::WrongRole);
        }
        if self.state != state {
            return Err(PairingError::UnexpectedMessage);
        }
        Ok(())
    }

    fn check_active(&self, now_millis: u64) -> Result<(), PairingError> {
        if now_millis < self.created_at_millis {
            return Err(PairingError::InvalidTimestamp);
        }
        if now_millis >= self.expires_at_millis {
            return Err(PairingError::Expired);
        }
        if matches!(
            self.state,
            PairingSessionState::Completed | PairingSessionState::Aborted
        ) {
            return Err(PairingError::Terminal);
        }
        Ok(())
    }

    fn validate_frame(
        &self,
        frame: &PairingEncryptedFrame,
        now_millis: u64,
        require_peer: bool,
    ) -> Result<(), PairingError> {
        if frame.recipient_public_key != self.local_public_key
            || frame.created_at_millis < self.created_at_millis
            || frame.created_at_millis >= self.expires_at_millis
            || frame.created_at_millis
                > now_millis
                    .checked_add(MAX_NIP_AB_CLOCK_SKEW_MILLIS)
                    .ok_or(PairingError::InvalidTimestamp)?
            || frame.ciphertext.is_empty()
            || frame.ciphertext.len() > MAX_NIP_AB_CIPHERTEXT_BYTES
        {
            return Err(PairingError::InvalidFrame);
        }
        if require_peer && self.peer_public_key != Some(frame.sender_public_key) {
            return Err(PairingError::WrongPeer);
        }
        Ok(())
    }

    fn encrypt(
        &self,
        frame_id: PairingFrameId,
        now_millis: u64,
        message: &OutboundMessage<'_>,
    ) -> Result<PairingEncryptedFrame, PairingError> {
        self.check_active(now_millis)?;
        let recipient_public_key = self.peer_public_key.ok_or(PairingError::WrongPeer)?;
        let plaintext = serde_json::to_vec(message).map_err(|_| PairingError::Codec)?;
        if plaintext.len() > MAX_NIP_AB_PAYLOAD_BYTES {
            return Err(PairingError::InvalidPayload);
        }
        let secret = nostr_secret(self.ephemeral_secret)?;
        let ciphertext =
            nip44::encrypt(&secret, &recipient_public_key, plaintext, Nip44Version::V2)
                .map_err(|_| PairingError::Crypto)?;
        PairingEncryptedFrame::from_transport(
            frame_id,
            self.local_public_key,
            recipient_public_key,
            now_millis,
            ciphertext,
        )
    }

    fn decrypt(&self, frame: &PairingEncryptedFrame) -> Result<InboundMessage, PairingError> {
        let secret = nostr_secret(self.ephemeral_secret)?;
        let mut plaintext =
            nip44::decrypt_to_bytes(&secret, &frame.sender_public_key, &frame.ciphertext)
                .map_err(|_| PairingError::Crypto)?;
        if plaintext.len() > MAX_NIP_AB_PAYLOAD_BYTES {
            plaintext.zeroize();
            return Err(PairingError::InvalidPayload);
        }
        let decoded = serde_json::from_slice(&plaintext).map_err(|_| PairingError::Codec);
        plaintext.zeroize();
        decoded
    }

    fn record(&mut self, frame_id: PairingFrameId) -> Result<(), PairingError> {
        if self.processed_frames.len() >= MAX_NIP_AB_PROCESSED_FRAMES {
            return Err(PairingError::ReplayCapacity);
        }
        if !self.processed_frames.insert(frame_id) {
            return Err(PairingError::Replay);
        }
        Ok(())
    }

    fn check_replay(&self, frame_id: PairingFrameId) -> Result<(), PairingError> {
        if self.processed_frames.contains(&frame_id) {
            Err(PairingError::Replay)
        } else {
            Ok(())
        }
    }

    fn transcript_hash(&self) -> Result<[u8; 32], PairingError> {
        let peer = self.peer_public_key.ok_or(PairingError::WrongPeer)?;
        let sas_input = self.sas_input.ok_or(PairingError::SasUnavailable)?;
        let (source, target) = match self.role {
            PairingRole::Source => (self.local_public_key, peer),
            PairingRole::Target => (peer, self.local_public_key),
        };
        let mut transcript = [0; 128];
        transcript[..32].copy_from_slice(&self.session_id);
        transcript[32..64].copy_from_slice(&source.to_bytes());
        transcript[64..96].copy_from_slice(&target.to_bytes());
        transcript[96..].copy_from_slice(&sas_input);
        Ok(hkdf32(&self.session_secret, &transcript, TRANSCRIPT_INFO))
    }
}

impl fmt::Debug for PairingSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingSession")
            .field("role", &self.role)
            .field("state", &self.state)
            .field("local_public_key", &self.local_public_key)
            .field("peer_public_key", &self.peer_public_key)
            .field("created_at_millis", &self.created_at_millis)
            .field("expires_at_millis", &self.expires_at_millis)
            .field("processed_frames", &self.processed_frames.len())
            .finish()
    }
}

impl Drop for PairingSession {
    fn drop(&mut self) {
        self.ephemeral_secret.zeroize();
        self.session_secret.zeroize();
        self.session_id.zeroize();
        if let Some(sas_input) = &mut self.sas_input {
            sas_input.zeroize();
        }
        self.sas_code = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairingError {
    InvalidQr,
    UnsupportedVersion,
    InvalidSecret,
    InvalidTimestamp,
    InvalidFrame,
    WrongPeer,
    WrongRole,
    UnexpectedMessage,
    InvalidSession,
    SasUnavailable,
    TranscriptMismatch,
    InvalidPayload,
    Replay,
    ReplayCapacity,
    Expired,
    Terminal,
    PeerAborted,
    Crypto,
    Codec,
}

impl fmt::Display for PairingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidQr => "invalid NIP-AB QR payload",
            Self::UnsupportedVersion => "unsupported NIP-AB version",
            Self::InvalidSecret => "invalid pairing secret",
            Self::InvalidTimestamp => "invalid pairing timestamp",
            Self::InvalidFrame => "invalid pairing frame",
            Self::WrongPeer => "pairing frame came from the wrong peer",
            Self::WrongRole => "pairing operation is unavailable for this role",
            Self::UnexpectedMessage => "pairing message is out of order",
            Self::InvalidSession => "pairing session proof did not match",
            Self::SasUnavailable => "pairing authentication string is unavailable",
            Self::TranscriptMismatch => "pairing transcript did not match",
            Self::InvalidPayload => "invalid pairing payload",
            Self::Replay => "pairing frame was already processed",
            Self::ReplayCapacity => "pairing replay capacity was reached",
            Self::Expired => "pairing session expired",
            Self::Terminal => "pairing session is already terminal",
            Self::PeerAborted => "pairing peer aborted",
            Self::Crypto => "pairing cryptography failed",
            Self::Codec => "pairing message codec failed",
        })
    }
}

impl Error for PairingError {}

fn default_nip_ab_version() -> u16 {
    NIP_AB_VERSION
}

fn is_lower_hex(value: u8) -> bool {
    value.is_ascii_digit() || (b'a'..=b'f').contains(&value)
}

fn public_key(secret: [u8; 32]) -> Result<NostrPublicKey, PairingError> {
    Ok(Keys::new(nostr_secret(secret)?).public_key())
}

fn nostr_secret(secret: [u8; 32]) -> Result<SecretKey, PairingError> {
    SecretKey::from_slice(&secret).map_err(|_| PairingError::InvalidSecret)
}

fn derive_session_id(session_secret: &[u8; 32]) -> [u8; 32] {
    hkdf32(&[], session_secret, SESSION_ID_INFO)
}

fn derive_sas(
    ephemeral_secret: [u8; 32],
    peer_public_key: NostrPublicKey,
    session_secret: [u8; 32],
) -> Result<(u32, [u8; 32]), PairingError> {
    let secret = nostr_secret(ephemeral_secret)?;
    let mut shared =
        generate_shared_key(&secret, &peer_public_key).map_err(|_| PairingError::Crypto)?;
    let sas_input = hkdf32(&session_secret, &shared, SAS_INFO);
    shared.zeroize();
    let code =
        u32::from_be_bytes([sas_input[0], sas_input[1], sas_input[2], sas_input[3]]) % 1_000_000;
    Ok((code, sas_input))
}

fn hkdf32(salt: &[u8], input: &[u8], info: &[u8]) -> [u8; 32] {
    let extracted = hkdf::extract(salt, input);
    let expanded = hkdf::expand(&extracted.to_byte_array(), info, 32);
    let mut result = [0; 32];
    result.copy_from_slice(&expanded[..32]);
    result
}

fn decode_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(is_lower_hex) {
        return None;
    }
    hex::decode(value).ok()?.try_into().ok()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut difference = 0u8;
    for (left, right) in left.iter().zip(right) {
        difference |= left ^ right;
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SECRET: [u8; 32] = [0x11; 32];
    const TARGET_SECRET: [u8; 32] = [0x22; 32];
    const SESSION_SECRET: [u8; 32] = [
        0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5,
        0xd6, 0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0,
        0xa1, 0xb2,
    ];

    fn frame(value: u8) -> PairingFrameId {
        PairingFrameId::new([value; 32]).expect("frame id")
    }

    fn relay() -> PairingRelayUrl {
        PairingRelayUrl::parse("wss://pair.example.test/").expect("relay")
    }

    fn source_session(secret: [u8; 32]) -> (PairingSession, PairingQr) {
        PairingSession::new_source(SOURCE_SECRET, secret, vec![relay()], 1_000)
            .expect("source session")
    }

    #[test]
    fn qr_and_encrypted_session_round_trip_match_nip_ab_vectors() {
        assert_eq!(
            hex::encode(derive_session_id(&SESSION_SECRET)),
            "fb357d0f8e8d5a5ba3b2a91cb18c119e1567b07ffa38cdebb73e68df78f5a380"
        );
        let (mut source, qr) = source_session(SESSION_SECRET);
        let encoded = qr.encode().expect("encode QR");
        let decoded = PairingQr::parse(&encoded).expect("decode QR");
        assert_eq!(decoded.source_public_key(), qr.source_public_key());
        assert_eq!(decoded.relays(), qr.relays());
        let mut target =
            PairingSession::new_target(&decoded, TARGET_SECRET, 1_000).expect("target session");
        let offer = target.offer(frame(1), 1_010).expect("offer");
        assert_eq!(
            source.receive_offer(&offer, 1_010).expect("receive offer"),
            target.sas().expect("target SAS")
        );
        let confirmation = source
            .confirm_source_sas(frame(2), 1_020)
            .expect("source confirms SAS");
        target
            .receive_sas_confirm(&confirmation, 1_020)
            .expect("target verifies transcript");
        target
            .confirm_target_sas(1_021)
            .expect("target confirms SAS");
        let payload = PairingPayload::new(
            PairingPayloadType::Nsec,
            Zeroizing::new("nsec1test-secret".to_string()),
        )
        .expect("payload");
        let encrypted = source
            .send_payload(frame(3), &payload, 1_030)
            .expect("send payload");
        assert!(!format!("{encrypted:?}").contains("nsec1test-secret"));
        let received = target
            .receive_payload(&encrypted, 1_030)
            .expect("receive payload");
        assert_eq!(received.payload_type(), PairingPayloadType::Nsec);
        assert_eq!(received.secret(), "nsec1test-secret");
        let complete = target
            .complete_target(frame(4), 1_040)
            .expect("complete target");
        source
            .receive_complete(&complete, 1_040)
            .expect("complete source");
        assert_eq!(source.state(), PairingSessionState::Completed);
        assert_eq!(target.state(), PairingSessionState::Completed);
    }

    #[test]
    fn wrong_session_secret_is_rejected_without_locking_peer() {
        let (mut source, _) = source_session(SESSION_SECRET);
        let (_, wrong_qr) = source_session([0x44; 32]);
        let target =
            PairingSession::new_target(&wrong_qr, TARGET_SECRET, 1_000).expect("target session");
        let offer = target.offer(frame(1), 1_010).expect("offer");
        assert_eq!(
            source.receive_offer(&offer, 1_010),
            Err(PairingError::InvalidSession)
        );
        assert_eq!(source.state(), PairingSessionState::WaitingOffer);
        assert_eq!(source.peer_public_key(), None);
    }

    #[test]
    fn exact_session_deadline_expires_without_state_advance() {
        let (mut source, qr) = source_session(SESSION_SECRET);
        let target = PairingSession::new_target(&qr, TARGET_SECRET, 1_000).expect("target session");
        assert_eq!(
            target.offer(frame(1), 121_000).err(),
            Some(PairingError::Expired)
        );
        let offer = target.offer(frame(2), 120_999).expect("last valid offer");
        assert_eq!(
            source.receive_offer(&offer, 121_000),
            Err(PairingError::Expired)
        );
        assert_eq!(source.state(), PairingSessionState::WaitingOffer);
    }

    #[test]
    fn processed_frame_replay_is_rejected_without_reprocessing() {
        let (mut source, qr) = source_session(SESSION_SECRET);
        let target = PairingSession::new_target(&qr, TARGET_SECRET, 1_000).expect("target session");
        let offer = target.offer(frame(1), 1_010).expect("offer");
        source.receive_offer(&offer, 1_010).expect("first offer");
        assert_eq!(
            source.receive_offer(&offer, 1_011),
            Err(PairingError::Replay)
        );
        assert_eq!(
            source.state(),
            PairingSessionState::AwaitingSourceConfirmation
        );
    }

    #[test]
    fn corrupted_qr_inputs_fail_closed() {
        let (_, qr) = source_session(SESSION_SECRET);
        let valid = qr.encode().expect("QR");
        let corruptions = [
            "https://example.test/",
            "nostrpair://deadbeef?secret=00&relay=wss%3A%2F%2Frelay.test&v=1",
            &valid.replace("secret=a1", "secret=A1"),
            &valid.replace("wss%3A%2F%2F", "https%3A%2F%2F"),
            &format!("{}{}", valid, "x".repeat(MAX_NIP_AB_QR_BYTES)),
        ];
        for corrupted in corruptions {
            assert!(PairingQr::parse(corrupted).is_err(), "accepted {corrupted}");
        }
    }

    #[test]
    fn qr_and_offer_version_mismatch_are_rejected() {
        let (mut source, qr) = source_session(SESSION_SECRET);
        let encoded = qr.encode().expect("QR").replace("v=1", "v=2");
        assert_eq!(
            PairingQr::parse(&encoded).err(),
            Some(PairingError::UnsupportedVersion)
        );
        let target = PairingSession::new_target(&qr, TARGET_SECRET, 1_000).expect("target session");
        let incompatible = target
            .encrypt(
                frame(1),
                1_010,
                &OutboundMessage::Offer {
                    version: 2,
                    session_id: hex::encode(target.session_id),
                },
            )
            .expect("incompatible offer");
        assert_eq!(
            source.receive_offer(&incompatible, 1_010),
            Err(PairingError::UnsupportedVersion)
        );
        assert_eq!(source.state(), PairingSessionState::WaitingOffer);
    }
}
