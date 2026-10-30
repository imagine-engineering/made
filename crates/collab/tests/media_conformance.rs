use std::{
    collections::BTreeSet,
    io::{Cursor, Read},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose};
use collab::media::{
    object_store::{
        AuthorizedMediaObject, MediaCheckpointCommitOutcome, MediaCleanupCandidate,
        MediaCleanupCheckpoint, MediaListingSafety, MediaObjectAuthority,
        MediaObjectAuthorityError, MediaObjectBackend, MediaObjectBackendError,
        MediaObjectDeleteOutcome, MediaObjectPage, MediaObjectStore, MediaObjectStoreLimits,
        MediaObjectVersion, MediaObjectWriteOutcome, MediaOrphanDeletionLease,
        MediaOrphanFinalization, MediaOrphanReservationOutcome, MediaPublication,
        MediaPublicationOutcome, MediaRangeRequest, MediaResolvedRange, MediaStoreOutcome,
        StoredMediaObject,
    },
    upload_admission::{
        MediaUploadAdmission, MediaUploadAdmissionBackend, MediaUploadAdmissionBackendError,
        MediaUploadAdmissionLimits, MediaUploadAdmissionOutcome, MediaUploadAdmissionService,
        MediaUploadRequest, MediaUploadReservationOutcome,
    },
    validation::{
        MAX_DECODED_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES,
        MediaValidationLimits, MediaValidationSession,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MediaByteSize, MediaContentHash, MediaContentType,
    MediaDescriptor, MediaObjectSelection, MediaTenantPath, MediaVariant, MediaVariantKind,
    MembershipRole, MembershipStatus, OperationId, PrincipalId, PrincipalScopes, ServiceAccountId,
    TenantContext, TrustedTenantRoute,
};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use media::collaboration::{
    NativeAttachmentPresentation, NativeImageFormat, VariantFallback, plan_native_media_attachment,
};
use nostr_compat::blossom::{
    BLOSSOM_AUTH_KIND, BlossomAdapterError, BlossomContentDisposition, BlossomMediaPath,
    BlossomPublicOrigin, BlossomRangeSelection, BlossomUploadRoute, authorize_blossom_download,
    authorize_blossom_upload, project_blossom_descriptor, resolve_blossom_object,
    resolve_blossom_range,
};
use nostr_compat::{CanonicalEvent, PublicKey};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const NOW_SECONDS: u64 = 1_700_000_000;
const NOW_MILLIS: u64 = NOW_SECONDS * 1_000;
const SERVER_HOST: &str = "media.example";
const ACCESSIBILITY_LABEL: &str = "Release architecture diagram";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ObservationKind {
    Upload,
    Download,
    Render,
    AuthorizationFailure,
    ProtocolFailure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MediaObservation {
    Upload {
        route: &'static str,
        hash: String,
        size: u64,
        content_type: String,
        width: u32,
        height: u32,
    },
    Download {
        alias: &'static str,
        selection: MediaObjectSelection,
        content_hash: String,
        content_type: String,
        disposition: BlossomContentDisposition,
        start: u64,
        end_inclusive: u64,
        body_hash: String,
    },
    Render {
        url: String,
        thumbnail_url: String,
        source_selection: MediaObjectSelection,
        source_type: String,
        image_format: NativeImageFormat,
        accessibility_label: String,
        variant_fallback: VariantFallback,
        animated_playback_requires_user_action: bool,
    },
    Failure {
        kind: ObservationKind,
        case: &'static str,
        status: u16,
        public_message: String,
    },
}

impl MediaObservation {
    const fn kind(&self) -> ObservationKind {
        match self {
            Self::Upload { .. } => ObservationKind::Upload,
            Self::Download { .. } => ObservationKind::Download,
            Self::Render { .. } => ObservationKind::Render,
            Self::Failure { kind, .. } => *kind,
        }
    }
}

#[derive(Clone, Debug)]
struct ConformanceScenario {
    trace: Vec<MediaObservation>,
    required: BTreeSet<ObservationKind>,
}

impl ConformanceScenario {
    fn all_media_seams(trace: Vec<MediaObservation>) -> Self {
        Self {
            trace,
            required: BTreeSet::from([
                ObservationKind::Upload,
                ObservationKind::Download,
                ObservationKind::Render,
                ObservationKind::AuthorizationFailure,
                ObservationKind::ProtocolFailure,
            ]),
        }
    }

    fn verify_coverage(&self) -> Result<(), Vec<ObservationKind>> {
        let seen = self
            .trace
            .iter()
            .map(MediaObservation::kind)
            .collect::<BTreeSet<_>>();
        let missing = self.required.difference(&seen).copied().collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }
}

#[derive(Clone, Default)]
struct AdmissionBackend {
    admission: Arc<Mutex<Option<MediaUploadAdmission>>>,
}

#[async_trait]
impl MediaUploadAdmissionBackend for AdmissionBackend {
    async fn reserve(
        &self,
        tenant: &TenantContext,
        admission: MediaUploadAdmission,
    ) -> Result<MediaUploadReservationOutcome, MediaUploadAdmissionBackendError> {
        if tenant.community_id() != admission.community_id() {
            return Err(MediaUploadAdmissionBackendError::Conflict);
        }
        let mut stored = self.admission.lock().expect("admission lock");
        if let Some(existing) = *stored {
            return Ok(MediaUploadReservationOutcome::Existing(existing));
        }
        *stored = Some(admission);
        Ok(MediaUploadReservationOutcome::Reserved)
    }
}

#[derive(Clone)]
struct ObjectRecord {
    descriptor: MediaDescriptor,
    version: MediaObjectVersion,
    bytes: Vec<u8>,
}

#[derive(Clone, Default)]
struct ObjectBackend {
    object: Arc<Mutex<Option<ObjectRecord>>>,
}

#[async_trait]
impl MediaObjectBackend for ObjectBackend {
    async fn put_if_absent(
        &self,
        descriptor: &MediaDescriptor,
        mut reader: std::fs::File,
    ) -> Result<MediaObjectWriteOutcome, MediaObjectBackendError> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|_| MediaObjectBackendError::Unavailable)?;
        if MediaContentHash::from_digest(Sha256::digest(&bytes).into()) != descriptor.content_hash()
            || u64::try_from(bytes.len()).map_err(|_| MediaObjectBackendError::InvalidData)?
                != descriptor.byte_size().get()
        {
            return Err(MediaObjectBackendError::InvalidData);
        }
        let mut object = self.object.lock().expect("object lock");
        if let Some(existing) = object.as_ref() {
            return StoredMediaObject::new(
                existing.descriptor.clone(),
                existing.version.clone(),
                NOW_MILLIS,
            )
            .map(MediaObjectWriteOutcome::Existing)
            .map_err(|_| MediaObjectBackendError::InvalidData);
        }
        let version = MediaObjectVersion::new("media-conformance-v1")
            .map_err(|_| MediaObjectBackendError::InvalidData)?;
        let stored = StoredMediaObject::new(descriptor.clone(), version.clone(), NOW_MILLIS)
            .map_err(|_| MediaObjectBackendError::InvalidData)?;
        *object = Some(ObjectRecord {
            descriptor: descriptor.clone(),
            version,
            bytes,
        });
        Ok(MediaObjectWriteOutcome::Created(stored))
    }

    async fn get_range(
        &self,
        descriptor: &MediaDescriptor,
        object_version: &MediaObjectVersion,
        range: MediaResolvedRange,
    ) -> Result<Option<Vec<u8>>, MediaObjectBackendError> {
        let object = self.object.lock().expect("object lock");
        let Some(object) = object.as_ref() else {
            return Ok(None);
        };
        if object.descriptor != *descriptor || object.version != *object_version {
            return Err(MediaObjectBackendError::InvalidData);
        }
        let start =
            usize::try_from(range.start()).map_err(|_| MediaObjectBackendError::InvalidData)?;
        let end = usize::try_from(range.end_inclusive())
            .map_err(|_| MediaObjectBackendError::InvalidData)?;
        Ok(object.bytes.get(start..=end).map(ToOwned::to_owned))
    }

    async fn list_page(
        &self,
        _after: Option<MediaContentHash>,
        _limit: u32,
    ) -> Result<MediaObjectPage, MediaObjectBackendError> {
        Ok(MediaObjectPage::new(
            Vec::new(),
            None,
            MediaListingSafety::KnownUnversioned,
        ))
    }

    async fn delete_if_match(
        &self,
        _content_hash: MediaContentHash,
        _object_version: &MediaObjectVersion,
    ) -> Result<MediaObjectDeleteOutcome, MediaObjectBackendError> {
        Err(MediaObjectBackendError::Unavailable)
    }
}

#[derive(Clone, Default)]
struct Authority {
    publication: Arc<Mutex<Option<MediaPublication>>>,
}

impl Authority {
    fn metadata(&self) -> collaboration_domain::MediaMetadata {
        self.publication
            .lock()
            .expect("publication lock")
            .as_ref()
            .expect("published metadata")
            .metadata()
            .clone()
    }
}

#[async_trait]
impl MediaObjectAuthority for Authority {
    async fn publish(
        &self,
        tenant: &TenantContext,
        publication: &MediaPublication,
    ) -> Result<MediaPublicationOutcome, MediaObjectAuthorityError> {
        if publication.metadata().fields().identity.community_id() != tenant.community_id() {
            return Err(MediaObjectAuthorityError::Conflict);
        }
        let mut stored = self.publication.lock().expect("publication lock");
        if let Some(existing) = stored.as_ref() {
            return Ok(MediaPublicationOutcome::Existing(existing.clone()));
        }
        *stored = Some(publication.clone());
        Ok(MediaPublicationOutcome::Published)
    }

    async fn resolve_for_read(
        &self,
        tenant: &TenantContext,
        _principal_id: PrincipalId,
        _resource: AuthorizationResource,
        path: MediaTenantPath,
    ) -> Result<Option<AuthorizedMediaObject>, MediaObjectAuthorityError> {
        let stored = self.publication.lock().expect("publication lock");
        let Some(publication) = stored.as_ref() else {
            return Ok(None);
        };
        let fields = publication.metadata().fields();
        if tenant.community_id() != path.community_id()
            || fields.identity.community_id() != path.community_id()
            || fields.identity.content_hash() != path.content_hash()
            || path.selection() != MediaObjectSelection::Original
        {
            return Ok(None);
        }
        AuthorizedMediaObject::new(
            path,
            MediaDescriptor::new(
                fields.identity.content_hash(),
                fields.content_type.clone(),
                fields.byte_size,
            ),
            publication.object_version().clone(),
        )
        .map(Some)
        .map_err(|_| MediaObjectAuthorityError::InvalidData)
    }

    async fn load_cleanup_checkpoint(
        &self,
        _job_id: Uuid,
        _scan_started_at_millis: u64,
    ) -> Result<MediaCleanupCheckpoint, MediaObjectAuthorityError> {
        Err(MediaObjectAuthorityError::Unavailable)
    }

    async fn reserve_orphan_deletion(
        &self,
        _checkpoint: MediaCleanupCheckpoint,
        _candidate: &MediaCleanupCandidate,
        _orphan_grace_millis: u64,
    ) -> Result<MediaOrphanReservationOutcome, MediaObjectAuthorityError> {
        Err(MediaObjectAuthorityError::Unavailable)
    }

    async fn finalize_orphan_deletion(
        &self,
        _lease: &MediaOrphanDeletionLease,
        _outcome: MediaOrphanFinalization,
    ) -> Result<(), MediaObjectAuthorityError> {
        Err(MediaObjectAuthorityError::Unavailable)
    }

    async fn commit_cleanup_checkpoint(
        &self,
        _expected: MediaCleanupCheckpoint,
        _next: MediaCleanupCheckpoint,
    ) -> Result<MediaCheckpointCommitOutcome, MediaObjectAuthorityError> {
        Err(MediaObjectAuthorityError::Unavailable)
    }
}

struct AuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    membership: CommunityMembership,
    write_scope: AuthorizationScope,
    read_scope: AuthorizationScope,
}

impl AuthorizationFixture {
    fn new() -> Self {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
        let principal_id = PrincipalId::from_uuid(Uuid::from_u128(2));
        let write_scope = AuthorizationScope::new("media:write").expect("write scope");
        let read_scope = AuthorizationScope::new("media:read").expect("read scope");
        Self {
            tenant: TenantContext::establish(
                Some(
                    TrustedTenantRoute::from_direct_host(community_id, SERVER_HOST)
                        .expect("trusted route"),
                ),
                &[],
            )
            .expect("tenant"),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(3),
                PrincipalScopes::new([write_scope.clone(), read_scope.clone()]).expect("scopes"),
            ),
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            write_scope,
            read_scope,
        }
    }

    fn upload_request(&self, operation_id: OperationId) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.write_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Media,
                resource_id: AggregateId::from_uuid(operation_id.as_uuid()),
                owner_principal_id: Some(self.principal.principal_id()),
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: NOW_MILLIS,
        }
    }

    fn read_request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.read_scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Media,
                resource_id: AggregateId::from_uuid(Uuid::from_u128(4)),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: NOW_MILLIS + 200,
        }
    }
}

fn png_bytes() -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 2, Rgba([17, 31, 47, 255])))
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("PNG fixture");
    bytes.into_inner()
}

fn signed_authorization_header(verb: &str, hash: &str, server: &str) -> String {
    let secret_key = SecretKey::from_slice(&[7; 32]).expect("secret key");
    let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret_key);
    let (public_key, _) = keypair.x_only_public_key();
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        NOW_SECONDS,
        BLOSSOM_AUTH_KIND,
        vec![
            vec!["t".into(), verb.into()],
            vec!["expiration".into(), (NOW_SECONDS + 300).to_string()],
            vec!["x".into(), hash.into()],
            vec!["server".into(), server.into()],
        ],
        format!("Authorize Blossom {verb}"),
    );
    let event_id = event.event_id().expect("event id");
    let signature = Secp256k1::new()
        .sign_schnorr_no_aux_rand(&Message::from_digest(*event_id.as_bytes()), &keypair);
    let wire = serde_json::json!({
        "id": event_id.to_hex(),
        "pubkey": event.public_key.to_hex(),
        "created_at": event.created_at,
        "kind": event.kind,
        "tags": event.tags,
        "content": event.content,
        "sig": signature.to_string(),
    });
    format!(
        "Nostr {}",
        general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&wire).expect("wire JSON"))
    )
}

fn upload_observation(route: &'static str, hash: &str, size: u64) -> MediaObservation {
    MediaObservation::Upload {
        route,
        hash: hash.to_owned(),
        size,
        content_type: "image/png".into(),
        width: 4,
        height: 2,
    }
}

fn download_observation(
    alias: &'static str,
    selection: MediaObjectSelection,
    content_hash: MediaContentHash,
    content_type: &str,
    start: u64,
    end_inclusive: u64,
    body: &[u8],
) -> MediaObservation {
    MediaObservation::Download {
        alias,
        selection,
        content_hash: content_hash.to_lower_hex(),
        content_type: content_type.to_owned(),
        disposition: BlossomContentDisposition::Inline,
        start,
        end_inclusive,
        body_hash: hex::encode(Sha256::digest(body)),
    }
}

fn failure_observation(
    kind: ObservationKind,
    case: &'static str,
    error: BlossomAdapterError,
) -> MediaObservation {
    MediaObservation::Failure {
        kind,
        case,
        status: error.http_status(),
        public_message: error.public_message().to_owned(),
    }
}

fn buzz_oracle_trace(
    bytes: &[u8],
    hash: &str,
    thumbnail_hash: MediaContentHash,
) -> Vec<MediaObservation> {
    let size = u64::try_from(bytes.len()).expect("fixture size");
    let range_end = 15_u64.min(size - 1);
    let origin = format!("https://{SERVER_HOST}/media");
    vec![
        upload_observation("/upload", hash, size),
        upload_observation("/media/upload", hash, size),
        download_observation(
            "canonical",
            MediaObjectSelection::Original,
            MediaContentHash::from_lower_hex(hash).expect("oracle hash"),
            "image/png",
            0,
            range_end,
            &bytes[..=usize::try_from(range_end).expect("range end")],
        ),
        download_observation(
            "bare",
            MediaObjectSelection::Original,
            MediaContentHash::from_lower_hex(hash).expect("oracle hash"),
            "image/png",
            0,
            range_end,
            &bytes[..=usize::try_from(range_end).expect("range end")],
        ),
        download_observation(
            "thumbnail",
            MediaObjectSelection::Variant(MediaVariantKind::Thumbnail),
            thumbnail_hash,
            "image/jpeg",
            0,
            4,
            b"thumb",
        ),
        MediaObservation::Render {
            url: format!("{origin}/{hash}.png"),
            thumbnail_url: format!("{origin}/{hash}.thumb.jpg"),
            source_selection: MediaObjectSelection::Variant(MediaVariantKind::Thumbnail),
            source_type: "image/jpeg".into(),
            image_format: NativeImageFormat::Jpeg,
            accessibility_label: ACCESSIBILITY_LABEL.into(),
            variant_fallback: VariantFallback::NotNeeded,
            animated_playback_requires_user_action: false,
        },
        MediaObservation::Failure {
            kind: ObservationKind::AuthorizationFailure,
            case: "wrong server scope",
            status: 401,
            public_message: "authentication failed".into(),
        },
        MediaObservation::Failure {
            kind: ObservationKind::ProtocolFailure,
            case: "unknown alias",
            status: 404,
            public_message: "not found".into(),
        },
        MediaObservation::Failure {
            kind: ObservationKind::ProtocolFailure,
            case: "unsatisfied range",
            status: 416,
            public_message: "range not satisfiable".into(),
        },
        MediaObservation::Failure {
            kind: ObservationKind::ProtocolFailure,
            case: "unsupported media",
            status: 415,
            public_message: "unsupported media type".into(),
        },
        MediaObservation::Failure {
            kind: ObservationKind::ProtocolFailure,
            case: "backend unavailable",
            status: 503,
            public_message: "media service unavailable".into(),
        },
    ]
}

async fn consolidated_trace(bytes: &[u8], hash: &str) -> Vec<MediaObservation> {
    let fixture = AuthorizationFixture::new();
    let upload_header = signed_authorization_header("upload", hash, SERVER_HOST);
    let canonical = authorize_blossom_upload(
        "/upload",
        &upload_header,
        Some(hash),
        SERVER_HOST,
        NOW_SECONDS,
    )
    .expect("canonical Blossom upload");
    let legacy = authorize_blossom_upload(
        "/media/upload",
        &upload_header,
        Some(hash),
        SERVER_HOST,
        NOW_SECONDS,
    )
    .expect("legacy Blossom upload");
    assert_eq!(canonical.route(), BlossomUploadRoute::Bud02);
    assert_eq!(legacy.route(), BlossomUploadRoute::LegacyMediaAlias);
    assert_eq!(
        canonical.authorization().operation_id(),
        legacy.authorization().operation_id()
    );

    let byte_size = u64::try_from(bytes.len()).expect("fixture size");
    let request = MediaUploadRequest::new(
        canonical.authorization().operation_id(),
        canonical.content_hash(),
        byte_size,
    )
    .expect("canonical request");
    let admission = MediaUploadAdmissionService::new(
        AdmissionBackend::default(),
        MediaUploadAdmissionLimits::new(MAX_IMAGE_UPLOAD_BYTES, 60_000).expect("admission limits"),
    )
    .admit(&fixture.upload_request(request.operation_id()), request)
    .await
    .expect("canonical admission");
    let MediaUploadAdmissionOutcome::Issued(admission) = admission else {
        panic!("first upload must issue an admission");
    };
    let mut validation = MediaValidationSession::begin(
        admission,
        &fixture.tenant,
        fixture.principal.principal_id(),
        MediaContentType::new("image/png").expect("content type"),
        NOW_MILLIS + 100,
        MediaValidationLimits::new(
            MAX_IMAGE_UPLOAD_BYTES,
            MAX_VIDEO_UPLOAD_BYTES,
            MAX_DECODED_IMAGE_PIXELS,
        )
        .expect("validation limits"),
    )
    .expect("validation session");
    for chunk in bytes.chunks(7) {
        validation.write_chunk(chunk).expect("validation chunk");
    }
    let validated = validation
        .finish(
            &fixture.tenant,
            fixture.principal.principal_id(),
            NOW_MILLIS + 100,
        )
        .expect("validated upload");
    let properties = validated.properties();

    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let store = MediaObjectStore::new(
        object_backend,
        authority.clone(),
        MediaObjectStoreLimits::default(),
    )
    .expect("object store");
    assert_eq!(
        store
            .store_validated(
                &fixture.tenant,
                fixture.principal.principal_id(),
                NOW_MILLIS + 200,
                validated,
            )
            .await,
        Ok(MediaStoreOutcome::Published)
    );
    let mut metadata = authority.metadata();
    let thumbnail_hash = MediaContentHash::from_digest(Sha256::digest(b"thumb").into());
    metadata
        .add_variant(MediaVariant::new(
            MediaVariantKind::Thumbnail,
            MediaDescriptor::new(
                thumbnail_hash,
                MediaContentType::new("image/jpeg").expect("thumbnail content type"),
                MediaByteSize::new(5).expect("thumbnail size"),
            ),
        ))
        .expect("thumbnail metadata");

    let range_end = 15_u64.min(byte_size - 1);
    let download_header = signed_authorization_header("get", hash, SERVER_HOST);
    let canonical_path = BlossomMediaPath::parse(&format!("{hash}.png")).expect("canonical path");
    authorize_blossom_download(&download_header, &canonical_path, SERVER_HOST, NOW_SECONDS)
        .expect("download authorization");
    let canonical_object = resolve_blossom_object(&metadata, &fixture.tenant, &canonical_path)
        .expect("canonical object");
    let BlossomRangeSelection::Partial(range) = resolve_blossom_range(
        Some(&format!("bytes=0-{range_end}")),
        byte_size,
        MediaObjectStoreLimits::default().max_range_bytes(),
    )
    .expect("Blossom range") else {
        panic!("single range must remain partial");
    };
    let response = store
        .read_range(
            &fixture.read_request(),
            canonical_object.tenant_path(),
            MediaRangeRequest::new(range.start(), range.end_inclusive()).expect("canonical range"),
        )
        .await
        .expect("stored range");
    let bare_path = BlossomMediaPath::parse(hash).expect("bare path");
    let bare_object =
        resolve_blossom_object(&metadata, &fixture.tenant, &bare_path).expect("bare object");
    let thumbnail_path =
        BlossomMediaPath::parse(&format!("{hash}.thumb.jpg")).expect("thumbnail path");
    let thumbnail_object = resolve_blossom_object(&metadata, &fixture.tenant, &thumbnail_path)
        .expect("thumbnail object");

    let origin_url = format!("https://{SERVER_HOST}");
    let origin = BlossomPublicOrigin::new(&origin_url).expect("public origin");
    let descriptor =
        project_blossom_descriptor(&metadata, &fixture.tenant, &origin).expect("descriptor");
    let presentation =
        plan_native_media_attachment(&metadata, &fixture.tenant, Some(ACCESSIBILITY_LABEL))
            .expect("native presentation");
    let NativeAttachmentPresentation::Image {
        source,
        format,
        accessibility_label,
        variant_fallback,
        animated_playback_requires_user_action,
    } = presentation
    else {
        panic!("PNG attachment must use native image presentation");
    };

    let wrong_server = authorize_blossom_upload(
        "/upload",
        &upload_header,
        Some(hash),
        "other.example",
        NOW_SECONDS,
    )
    .expect_err("foreign server scope");
    let unknown_alias = BlossomMediaPath::parse("../../etc/passwd").expect_err("unknown alias");
    let invalid_range = resolve_blossom_range(
        Some(&format!("bytes={byte_size}-")),
        byte_size,
        MediaObjectStoreLimits::default().max_range_bytes(),
    )
    .expect_err("unsatisfied range");

    vec![
        MediaObservation::Upload {
            route: "/upload",
            hash: canonical.content_hash().to_lower_hex(),
            size: admission.byte_size().get(),
            content_type: "image/png".into(),
            width: properties.width,
            height: properties.height,
        },
        MediaObservation::Upload {
            route: "/media/upload",
            hash: legacy.content_hash().to_lower_hex(),
            size: admission.byte_size().get(),
            content_type: "image/png".into(),
            width: properties.width,
            height: properties.height,
        },
        download_observation(
            "canonical",
            canonical_object.tenant_path().selection(),
            canonical_object.descriptor().content_hash(),
            canonical_object.descriptor().content_type().as_str(),
            response.range().start(),
            response.range().end_inclusive(),
            response.bytes(),
        ),
        download_observation(
            "bare",
            bare_object.tenant_path().selection(),
            bare_object.descriptor().content_hash(),
            bare_object.descriptor().content_type().as_str(),
            response.range().start(),
            response.range().end_inclusive(),
            response.bytes(),
        ),
        download_observation(
            "thumbnail",
            thumbnail_object.tenant_path().selection(),
            thumbnail_object.descriptor().content_hash(),
            thumbnail_object.descriptor().content_type().as_str(),
            0,
            4,
            b"thumb",
        ),
        MediaObservation::Render {
            url: descriptor.url,
            thumbnail_url: descriptor.thumb.expect("thumbnail URL"),
            source_selection: source.path().selection(),
            source_type: source.descriptor().content_type().as_str().to_owned(),
            image_format: format,
            accessibility_label: accessibility_label.as_str().to_owned(),
            variant_fallback,
            animated_playback_requires_user_action,
        },
        failure_observation(
            ObservationKind::AuthorizationFailure,
            "wrong server scope",
            wrong_server,
        ),
        failure_observation(
            ObservationKind::ProtocolFailure,
            "unknown alias",
            unknown_alias,
        ),
        failure_observation(
            ObservationKind::ProtocolFailure,
            "unsatisfied range",
            invalid_range,
        ),
        failure_observation(
            ObservationKind::ProtocolFailure,
            "unsupported media",
            BlossomAdapterError::UnsupportedMedia,
        ),
        failure_observation(
            ObservationKind::ProtocolFailure,
            "backend unavailable",
            BlossomAdapterError::ServiceUnavailable,
        ),
    ]
}

#[tokio::test]
async fn buzz_and_consolidated_media_paths_emit_the_same_observations() {
    let bytes = png_bytes();
    let content_hash = MediaContentHash::from_digest(Sha256::digest(&bytes).into());
    let hash = content_hash.to_lower_hex();
    let thumbnail_hash = MediaContentHash::from_digest(Sha256::digest(b"thumb").into());
    let buzz =
        ConformanceScenario::all_media_seams(buzz_oracle_trace(&bytes, &hash, thumbnail_hash));
    let consolidated =
        ConformanceScenario::all_media_seams(consolidated_trace(&bytes, &hash).await);

    buzz.verify_coverage().expect("Buzz trace coverage");
    consolidated
        .verify_coverage()
        .expect("consolidated trace coverage");
    assert_eq!(consolidated.trace, buzz.trace);
}

#[test]
fn independent_checker_bites_missing_and_divergent_observations() {
    let complete = ConformanceScenario::all_media_seams(vec![
        upload_observation("/upload", &"00".repeat(32), 1),
        download_observation(
            "bare",
            MediaObjectSelection::Original,
            MediaContentHash::from_digest([0; 32]),
            "image/png",
            0,
            0,
            b"x",
        ),
        MediaObservation::Render {
            url: "https://media.example/media/hash.png".into(),
            thumbnail_url: "https://media.example/media/hash.thumb.jpg".into(),
            source_selection: MediaObjectSelection::Original,
            source_type: "image/png".into(),
            image_format: NativeImageFormat::Png,
            accessibility_label: "Image attachment".into(),
            variant_fallback: VariantFallback::MissingOrUnsupported(MediaVariantKind::Thumbnail),
            animated_playback_requires_user_action: false,
        },
        failure_observation(
            ObservationKind::AuthorizationFailure,
            "denied",
            BlossomAdapterError::InvalidAuthentication,
        ),
        failure_observation(
            ObservationKind::ProtocolFailure,
            "not found",
            BlossomAdapterError::NotFound,
        ),
    ]);
    complete.verify_coverage().expect("complete trace");

    let mut missing = complete.clone();
    missing
        .trace
        .retain(|observation| observation.kind() != ObservationKind::Download);
    assert_eq!(
        missing.verify_coverage(),
        Err(vec![ObservationKind::Download])
    );

    let mut divergent = complete.trace.clone();
    let MediaObservation::Upload { hash, .. } = &mut divergent[0] else {
        panic!("first observation is upload");
    };
    *hash = "ff".repeat(32);
    assert_ne!(divergent, complete.trace);
}
