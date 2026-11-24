use std::{collections::BTreeSet, error::Error, fmt, num::NonZeroU64};

use crate::{AggregateId, CommunityId, PrincipalId, TenantContext};

pub const MAX_MEDIA_CONTENT_TYPE_BYTES: usize = 127;
pub const MAX_MEDIA_VARIANTS: usize = 2;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MediaContentHash([u8; 32]);

impl MediaContentHash {
    pub const fn from_digest(value: [u8; 32]) -> Self {
        Self(value)
    }

    pub fn from_lower_hex(value: &str) -> Result<Self, MediaMetadataError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(MediaMetadataError::InvalidContentHash);
        }
        let mut digest = [0; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            digest[index] = (decode_hex_nibble(pair[0]) << 4) | decode_hex_nibble(pair[1]);
        }
        Ok(Self(digest))
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_lower_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(64);
        for byte in self.0 {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        encoded
    }
}

impl fmt::Debug for MediaContentHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MediaContentHash([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MediaContentType(String);

impl MediaContentType {
    pub fn new(value: impl Into<String>) -> Result<Self, MediaMetadataError> {
        let value = value.into();
        let mut components = value.split('/');
        let top_level = components.next().unwrap_or_default();
        let subtype = components.next().unwrap_or_default();
        if value.len() > MAX_MEDIA_CONTENT_TYPE_BYTES
            || top_level.is_empty()
            || subtype.is_empty()
            || components.next().is_some()
            || !top_level.bytes().all(valid_media_type_byte)
            || !subtype.bytes().all(valid_media_type_byte)
        {
            return Err(MediaMetadataError::InvalidContentType);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MediaByteSize(NonZeroU64);

impl MediaByteSize {
    pub const fn new(value: u64) -> Result<Self, MediaMetadataError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(MediaMetadataError::InvalidSize),
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MediaIdentity {
    community_id: CommunityId,
    content_hash: MediaContentHash,
}

impl MediaIdentity {
    pub fn new(
        community_id: CommunityId,
        content_hash: MediaContentHash,
    ) -> Result<Self, MediaMetadataError> {
        if community_id.as_uuid().is_nil() {
            return Err(MediaMetadataError::InvalidIdentity);
        }
        Ok(Self {
            community_id,
            content_hash,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn content_hash(self) -> MediaContentHash {
        self.content_hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaDescriptor {
    content_hash: MediaContentHash,
    content_type: MediaContentType,
    byte_size: MediaByteSize,
}

impl MediaDescriptor {
    pub const fn new(
        content_hash: MediaContentHash,
        content_type: MediaContentType,
        byte_size: MediaByteSize,
    ) -> Self {
        Self {
            content_hash,
            content_type,
            byte_size,
        }
    }

    pub const fn content_hash(&self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn content_type(&self) -> &MediaContentType {
        &self.content_type
    }

    pub const fn byte_size(&self) -> MediaByteSize {
        self.byte_size
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MediaVariantKind {
    Thumbnail,
    Poster,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaVariant {
    kind: MediaVariantKind,
    descriptor: MediaDescriptor,
}

impl MediaVariant {
    pub const fn new(kind: MediaVariantKind, descriptor: MediaDescriptor) -> Self {
        Self { kind, descriptor }
    }

    pub const fn kind(&self) -> MediaVariantKind {
        self.kind
    }

    pub const fn descriptor(&self) -> &MediaDescriptor {
        &self.descriptor
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaObjectSelection {
    Original,
    Variant(MediaVariantKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaTenantPath {
    community_id: CommunityId,
    content_hash: MediaContentHash,
    selection: MediaObjectSelection,
}

impl MediaTenantPath {
    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn content_hash(self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn selection(self) -> MediaObjectSelection {
        self.selection
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaMetadataFields {
    pub identity: MediaIdentity,
    pub owner_principal_id: PrincipalId,
    pub content_type: MediaContentType,
    pub byte_size: MediaByteSize,
    pub uploaded_at_millis: u64,
    pub variants: Vec<MediaVariant>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaMetadata {
    fields: MediaMetadataFields,
}

impl MediaMetadata {
    pub fn from_record(fields: MediaMetadataFields) -> Result<Self, MediaMetadataError> {
        if fields.owner_principal_id.as_uuid().is_nil() {
            return Err(MediaMetadataError::InvalidIdentity);
        }
        if fields.uploaded_at_millis == 0 {
            return Err(MediaMetadataError::InvalidTimestamp);
        }
        if fields.variants.len() > MAX_MEDIA_VARIANTS {
            return Err(MediaMetadataError::TooManyVariants);
        }
        let mut kinds = BTreeSet::new();
        for variant in &fields.variants {
            if !kinds.insert(variant.kind) {
                return Err(MediaMetadataError::DuplicateVariant);
            }
        }
        Ok(Self { fields })
    }

    pub fn new(
        identity: MediaIdentity,
        owner_principal_id: PrincipalId,
        content_type: MediaContentType,
        byte_size: MediaByteSize,
        uploaded_at_millis: u64,
    ) -> Result<Self, MediaMetadataError> {
        Self::from_record(MediaMetadataFields {
            identity,
            owner_principal_id,
            content_type,
            byte_size,
            uploaded_at_millis,
            variants: Vec::new(),
        })
    }

    pub const fn fields(&self) -> &MediaMetadataFields {
        &self.fields
    }

    pub fn add_variant(&mut self, variant: MediaVariant) -> Result<(), MediaMetadataError> {
        if self.fields.variants.len() >= MAX_MEDIA_VARIANTS {
            return Err(MediaMetadataError::TooManyVariants);
        }
        if self
            .fields
            .variants
            .iter()
            .any(|existing| existing.kind == variant.kind)
        {
            return Err(MediaMetadataError::DuplicateVariant);
        }
        self.fields.variants.push(variant);
        self.fields
            .variants
            .sort_unstable_by_key(MediaVariant::kind);
        Ok(())
    }

    pub fn tenant_path(
        &self,
        tenant: &TenantContext,
        selection: MediaObjectSelection,
    ) -> Result<MediaTenantPath, MediaMetadataError> {
        if tenant.community_id() != self.fields.identity.community_id {
            return Err(MediaMetadataError::TenantMismatch);
        }
        let content_hash = match selection {
            MediaObjectSelection::Original => self.fields.identity.content_hash,
            MediaObjectSelection::Variant(kind) => self
                .fields
                .variants
                .iter()
                .find(|variant| variant.kind == kind)
                .map(|variant| variant.descriptor.content_hash)
                .ok_or(MediaMetadataError::MissingVariant)?,
        };
        Ok(MediaTenantPath {
            community_id: self.fields.identity.community_id,
            content_hash,
            selection,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaAttachmentLink {
    media_identity: MediaIdentity,
    channel_id: AggregateId,
    message_id: AggregateId,
}

impl MediaAttachmentLink {
    pub fn new(
        tenant: &TenantContext,
        media_identity: MediaIdentity,
        channel_id: AggregateId,
        message_id: AggregateId,
    ) -> Result<Self, MediaMetadataError> {
        if tenant.community_id() != media_identity.community_id
            || channel_id.as_uuid().is_nil()
            || message_id.as_uuid().is_nil()
        {
            return Err(MediaMetadataError::InvalidAttachmentLink);
        }
        Ok(Self {
            media_identity,
            channel_id,
            message_id,
        })
    }

    pub const fn media_identity(self) -> MediaIdentity {
        self.media_identity
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }

    pub const fn message_id(self) -> AggregateId {
        self.message_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaMetadataError {
    InvalidContentHash,
    InvalidContentType,
    InvalidSize,
    InvalidIdentity,
    InvalidTimestamp,
    DuplicateVariant,
    TooManyVariants,
    MissingVariant,
    TenantMismatch,
    InvalidAttachmentLink,
}

impl fmt::Display for MediaMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidContentHash => "media content hash is invalid",
            Self::InvalidContentType => "media content type is invalid",
            Self::InvalidSize => "media byte size is invalid",
            Self::InvalidIdentity => "media identity is invalid",
            Self::InvalidTimestamp => "media upload timestamp is invalid",
            Self::DuplicateVariant => "media variant is duplicated",
            Self::TooManyVariants => "media metadata has too many variants",
            Self::MissingVariant => "media variant is unavailable",
            Self::TenantMismatch => "media tenant path does not match the admitted community",
            Self::InvalidAttachmentLink => "media attachment link is invalid",
        })
    }
}

impl Error for MediaMetadataError {}

fn valid_media_type_byte(byte: u8) -> bool {
    byte.is_ascii_lowercase()
        || byte.is_ascii_digit()
        || matches!(
            byte,
            b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'.' | b'+' | b'-'
        )
}

const fn decode_hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrustedTenantRoute;
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community_id, "media").expect("route")),
            &[],
        )
        .expect("tenant")
    }

    fn metadata(community_id: CommunityId) -> MediaMetadata {
        MediaMetadata::new(
            MediaIdentity::new(community_id, MediaContentHash::from_digest([1; 32]))
                .expect("identity"),
            PrincipalId::from_uuid(Uuid::from_u128(2)),
            MediaContentType::new("image/png").expect("content type"),
            MediaByteSize::new(1_024).expect("size"),
            1_000,
        )
        .expect("metadata")
    }

    #[test]
    fn media_hash_identity_is_exact_and_tenant_scoped() {
        let parsed = MediaContentHash::from_lower_hex(&"01".repeat(32)).expect("hash");
        assert_eq!(parsed.to_lower_hex(), "01".repeat(32));
        assert_eq!(parsed.as_bytes(), [1; 32]);
        assert!(MediaContentHash::from_lower_hex(&"AB".repeat(32)).is_err());
        assert!(MediaContentHash::from_lower_hex("../content").is_err());

        let local = MediaIdentity::new(community(1), parsed).expect("local identity");
        let same = MediaIdentity::new(community(1), parsed).expect("same identity");
        let foreign = MediaIdentity::new(community(2), parsed).expect("foreign identity");
        assert_eq!(local, same);
        assert_ne!(local, foreign);
    }

    #[test]
    fn media_attachment_link_binds_the_admitted_message_tenant() {
        let metadata = metadata(community(1));
        let link = MediaAttachmentLink::new(
            &tenant(community(1)),
            metadata.fields().identity,
            AggregateId::from_uuid(Uuid::from_u128(3)),
            AggregateId::from_uuid(Uuid::from_u128(4)),
        )
        .expect("attachment link");
        assert_eq!(link.media_identity(), metadata.fields().identity);
        assert_eq!(link.channel_id().as_uuid(), Uuid::from_u128(3));
        assert_eq!(link.message_id().as_uuid(), Uuid::from_u128(4));

        assert_eq!(
            MediaAttachmentLink::new(
                &tenant(community(2)),
                metadata.fields().identity,
                link.channel_id(),
                link.message_id(),
            ),
            Err(MediaMetadataError::InvalidAttachmentLink)
        );
    }

    #[test]
    fn media_variants_are_bounded_unique_and_resolve_by_content_identity() {
        let community_id = community(1);
        let mut metadata = metadata(community_id);
        let thumbnail = MediaVariant::new(
            MediaVariantKind::Thumbnail,
            MediaDescriptor::new(
                MediaContentHash::from_digest([2; 32]),
                MediaContentType::new("image/jpeg").expect("thumbnail type"),
                MediaByteSize::new(128).expect("thumbnail size"),
            ),
        );
        metadata.add_variant(thumbnail.clone()).expect("variant");
        assert_eq!(
            metadata.add_variant(thumbnail),
            Err(MediaMetadataError::DuplicateVariant)
        );
        let path = metadata
            .tenant_path(
                &tenant(community_id),
                MediaObjectSelection::Variant(MediaVariantKind::Thumbnail),
            )
            .expect("thumbnail path");
        assert_eq!(path.community_id(), community_id);
        assert_eq!(path.content_hash().as_bytes(), [2; 32]);
        assert_eq!(
            path.selection(),
            MediaObjectSelection::Variant(MediaVariantKind::Thumbnail)
        );
        assert_eq!(
            metadata.tenant_path(
                &tenant(community_id),
                MediaObjectSelection::Variant(MediaVariantKind::Poster),
            ),
            Err(MediaMetadataError::MissingVariant)
        );
    }

    #[test]
    fn media_metadata_rejects_invalid_tenant_paths_and_record_shapes() {
        let metadata = metadata(community(1));
        assert_eq!(
            metadata.tenant_path(&tenant(community(2)), MediaObjectSelection::Original),
            Err(MediaMetadataError::TenantMismatch)
        );
        assert!(MediaContentType::new("Image/PNG").is_err());
        assert!(MediaContentType::new("image/png; charset=utf-8").is_err());
        assert_eq!(MediaByteSize::new(0), Err(MediaMetadataError::InvalidSize));

        let mut invalid_timestamp = metadata.fields().clone();
        invalid_timestamp.uploaded_at_millis = 0;
        assert_eq!(
            MediaMetadata::from_record(invalid_timestamp),
            Err(MediaMetadataError::InvalidTimestamp)
        );

        let fields = MediaMetadataFields {
            identity: metadata.fields().identity,
            owner_principal_id: metadata.fields().owner_principal_id,
            content_type: metadata.fields().content_type.clone(),
            byte_size: metadata.fields().byte_size,
            uploaded_at_millis: metadata.fields().uploaded_at_millis,
            variants: vec![
                MediaVariant::new(
                    MediaVariantKind::Thumbnail,
                    MediaDescriptor::new(
                        MediaContentHash::from_digest([2; 32]),
                        MediaContentType::new("image/jpeg").expect("type"),
                        MediaByteSize::new(128).expect("size"),
                    ),
                ),
                MediaVariant::new(
                    MediaVariantKind::Thumbnail,
                    MediaDescriptor::new(
                        MediaContentHash::from_digest([3; 32]),
                        MediaContentType::new("image/jpeg").expect("type"),
                        MediaByteSize::new(64).expect("size"),
                    ),
                ),
            ],
        };
        assert_eq!(
            MediaMetadata::from_record(fields),
            Err(MediaMetadataError::DuplicateVariant)
        );
    }
}
