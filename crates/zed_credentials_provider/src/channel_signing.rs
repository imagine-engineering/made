use collaboration_domain::CommunityId;
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use nostr_compat::{
    CanonicalEvent, EventCodecError, EventSignature, PublicKey, SignedEvent, VerificationError,
};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use zeroize::Zeroizing;

const MAX_GENERATION_ATTEMPTS: usize = 32;

#[derive(thiserror::Error, Debug)]
pub enum ChannelSigningError {
    #[error("protected collaboration signing storage is unavailable")]
    Storage,
    #[error("stored collaboration signing identity is invalid")]
    CorruptIdentity,
    #[error("operating-system entropy is unavailable")]
    Entropy,
    #[error(transparent)]
    Event(#[from] EventCodecError),
    #[error(transparent)]
    Verification(#[from] VerificationError),
}

pub struct ChannelSigningIdentity {
    secret: Zeroizing<[u8; 32]>,
    public_key: [u8; 32],
}

impl ChannelSigningIdentity {
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public_key
    }

    pub fn sign(
        &self,
        created_at: u64,
        kind: u16,
        tags: Vec<Vec<String>>,
        content: String,
    ) -> Result<SignedEvent, ChannelSigningError> {
        let event = CanonicalEvent::new(
            PublicKey::from_bytes(self.public_key),
            created_at,
            kind,
            tags,
            content,
        );
        let event_id = event.event_id()?;
        let secret = SecretKey::from_slice(self.secret.as_ref())
            .map_err(|_| ChannelSigningError::CorruptIdentity)?;
        let secp = Secp256k1::new();
        let keypair = Keypair::from_secret_key(&secp, &secret);
        let signature =
            secp.sign_schnorr_no_aux_rand(&Message::from_digest(*event_id.as_bytes()), &keypair);
        Ok(SignedEvent {
            claimed_id: event_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string())?,
        })
    }
}

pub async fn load_or_create_channel_signing_identity(
    provider: &dyn CredentialsProvider,
    community_id: CommunityId,
    account_id: u64,
    cx: &AsyncApp,
) -> Result<ChannelSigningIdentity, ChannelSigningError> {
    let identifier = format!("zed-collaboration-message-signing:v1:{community_id}:{account_id}");
    if let Some((encoded_public_key, encoded_secret)) = provider
        .read_credentials(&identifier, cx)
        .await
        .map_err(|_| ChannelSigningError::Storage)?
    {
        return decode_identity(&encoded_public_key, &encoded_secret);
    }
    let identity = generate_identity()?;
    provider
        .write_credentials(
            &identifier,
            &hex::encode(identity.public_key),
            identity.secret.as_ref(),
            cx,
        )
        .await
        .map_err(|_| ChannelSigningError::Storage)?;
    Ok(identity)
}

fn generate_identity() -> Result<ChannelSigningIdentity, ChannelSigningError> {
    for _ in 0..MAX_GENERATION_ATTEMPTS {
        let mut bytes = Zeroizing::new([0; 32]);
        getrandom::fill(bytes.as_mut()).map_err(|_| ChannelSigningError::Entropy)?;
        if SecretKey::from_slice(bytes.as_ref()).is_ok() {
            return identity_from_secret(bytes);
        }
    }
    Err(ChannelSigningError::Entropy)
}

fn decode_identity(
    encoded_public_key: &str,
    encoded_secret: &[u8],
) -> Result<ChannelSigningIdentity, ChannelSigningError> {
    let secret = <[u8; 32]>::try_from(encoded_secret)
        .map(Zeroizing::new)
        .map_err(|_| ChannelSigningError::CorruptIdentity)?;
    let identity = identity_from_secret(secret)?;
    let expected_public_key = hex::decode(encoded_public_key)
        .ok()
        .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
        .ok_or(ChannelSigningError::CorruptIdentity)?;
    if identity.public_key != expected_public_key {
        return Err(ChannelSigningError::CorruptIdentity);
    }
    Ok(identity)
}

fn identity_from_secret(
    secret: Zeroizing<[u8; 32]>,
) -> Result<ChannelSigningIdentity, ChannelSigningError> {
    let secret_key =
        SecretKey::from_slice(secret.as_ref()).map_err(|_| ChannelSigningError::CorruptIdentity)?;
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret_key);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    Ok(ChannelSigningIdentity {
        secret,
        public_key: public_key.serialize(),
    })
}
