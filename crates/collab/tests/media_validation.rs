use std::io::{Cursor, Read};

use bytes::Bytes;
use collab::media::{
    upload_admission::{MediaUploadAdmission, MediaUploadRequest},
    validation::{
        MAX_DECODED_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES,
        MediaValidationError, MediaValidationLimits, MediaValidationSession, SupportedMediaKind,
    },
};
use collaboration_domain::{
    CommunityId, MediaContentHash, MediaContentType, OperationId, PrincipalId, TenantContext,
    TrustedTenantRoute,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};
use sha2::{Digest, Sha256};
use uuid::Uuid;

fn community() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn principal() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(2))
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(TrustedTenantRoute::from_listener(community(), "media-validation").expect("route")),
        &[],
    )
    .expect("tenant")
}

fn digest(bytes: &[u8]) -> MediaContentHash {
    MediaContentHash::from_digest(Sha256::digest(bytes).into())
}

fn admission(bytes: &[u8], expected_hash: MediaContentHash) -> MediaUploadAdmission {
    let tenant = tenant();
    MediaUploadAdmission::restore(
        &tenant,
        principal(),
        MediaUploadRequest::new(
            OperationId::from_uuid(Uuid::from_u128(3)),
            expected_hash,
            bytes.len() as u64,
        )
        .expect("request"),
        100,
        1_100,
    )
    .expect("admission")
}

fn limits() -> MediaValidationLimits {
    MediaValidationLimits::new(
        MAX_IMAGE_UPLOAD_BYTES,
        MAX_VIDEO_UPLOAD_BYTES,
        MAX_DECODED_IMAGE_PIXELS,
    )
    .expect("limits")
}

fn encoded_image(kind: SupportedMediaKind) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    match kind {
        SupportedMediaKind::Jpeg => {
            DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([17, 31, 47])))
                .write_to(&mut bytes, ImageFormat::Jpeg)
                .expect("jpeg")
        }
        SupportedMediaKind::Png => {
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([17, 31, 47, 255])))
                .write_to(&mut bytes, ImageFormat::Png)
                .expect("png")
        }
        SupportedMediaKind::Gif => {
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([17, 31, 47, 255])))
                .write_to(&mut bytes, ImageFormat::Gif)
                .expect("gif")
        }
        SupportedMediaKind::Webp => {
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([17, 31, 47, 255])))
                .write_to(&mut bytes, ImageFormat::WebP)
                .expect("webp")
        }
        SupportedMediaKind::Mp4 => panic!("fixture requires an encoded video"),
    }
    bytes.into_inner()
}

fn encoded_mp4() -> Vec<u8> {
    let config = mp4::Mp4Config {
        major_brand: mp4::FourCC::from(*b"isom"),
        minor_version: 0,
        compatible_brands: vec![mp4::FourCC::from(*b"isom"), mp4::FourCC::from(*b"avc1")],
        timescale: 1_000,
    };
    let mut writer = mp4::Mp4Writer::write_start(Cursor::new(Vec::new()), &config).expect("writer");
    writer
        .add_track(&mp4::TrackConfig::from(mp4::AvcConfig {
            width: 2,
            height: 2,
            seq_param_set: vec![0x67, 0x42, 0x00, 0x1e],
            pic_param_set: vec![0x68, 0xce],
        }))
        .expect("track");
    writer
        .write_sample(
            1,
            &mp4::Mp4Sample {
                start_time: 0,
                duration: 1_000,
                rendering_offset: 0,
                is_sync: true,
                bytes: Bytes::from_static(&[0, 0, 0, 1, 0x65]),
            },
        )
        .expect("sample");
    writer.write_end().expect("finish writer");
    let bytes = writer.into_writer().into_inner();

    let mut boxes = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let size_bytes: [u8; 4] = bytes[offset..offset + 4].try_into().expect("box size");
        let size = u32::from_be_bytes(size_bytes) as usize;
        let end = offset.checked_add(size).expect("box end");
        boxes.push((&bytes[offset + 4..offset + 8], &bytes[offset..end]));
        offset = end;
    }
    let mut fast_start = Vec::with_capacity(bytes.len());
    for expected in [b"ftyp", b"moov", b"mdat"] {
        let (_, bytes) = boxes
            .iter()
            .find(|(kind, _)| *kind == expected)
            .expect("expected MP4 box");
        fast_start.extend_from_slice(bytes);
    }
    fast_start
}

fn validate(bytes: &[u8], kind: SupportedMediaKind) -> Result<(), MediaValidationError> {
    let tenant = tenant();
    let mut session = MediaValidationSession::begin(
        admission(bytes, digest(bytes)),
        &tenant,
        principal(),
        MediaContentType::new(kind.content_type()).expect("content type"),
        200,
        limits(),
    )?;
    for chunk in bytes.chunks(3) {
        session.write_chunk(chunk)?;
    }
    let validated = session.finish(&tenant, principal(), 200)?;
    assert_eq!(validated.kind(), kind);
    assert_eq!(validated.properties().width, 2);
    assert_eq!(validated.properties().height, 2);
    let diagnostics = format!("{validated:?}");
    assert!(!diagnostics.contains(&digest(bytes).to_lower_hex()));
    let mut stored = Vec::new();
    validated
        .into_reader()
        .expect("reader")
        .read_to_end(&mut stored)
        .expect("read");
    assert_eq!(stored, bytes);
    Ok(())
}

#[test]
fn supported_media_stream_validate_and_remain_readable() {
    for kind in [
        SupportedMediaKind::Jpeg,
        SupportedMediaKind::Png,
        SupportedMediaKind::Gif,
        SupportedMediaKind::Webp,
    ] {
        let bytes = encoded_image(kind);
        validate(&bytes, kind)
            .unwrap_or_else(|error| panic!("{} validation failed: {error}", kind.content_type()));
    }
    let bytes = encoded_mp4();
    mp4::Mp4Reader::read_header(Cursor::new(&bytes), bytes.len() as u64)
        .unwrap_or_else(|error| panic!("generated MP4 must parse: {error}"));
    validate(&bytes, SupportedMediaKind::Mp4).expect("mp4 validation");
}

#[test]
fn polyglot_and_trailing_payload_are_rejected() {
    let mut bytes = encoded_image(SupportedMediaKind::Png);
    bytes.extend_from_slice(b"PK\x03\x04hidden");
    let tenant = tenant();
    let mut session = MediaValidationSession::begin(
        admission(&bytes, digest(&bytes)),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    session.write_chunk(&bytes).expect("chunk");
    assert!(matches!(
        session.finish(&tenant, principal(), 200),
        Err(MediaValidationError::MetadataForbidden)
    ));
}

#[test]
fn truncated_stream_is_rejected_before_decode() {
    let bytes = encoded_image(SupportedMediaKind::Png);
    let tenant = tenant();
    let mut session = MediaValidationSession::begin(
        admission(&bytes, digest(&bytes)),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    session
        .write_chunk(&bytes[..bytes.len() - 5])
        .expect("partial chunk");
    assert!(matches!(
        session.finish(&tenant, principal(), 200),
        Err(MediaValidationError::Truncated)
    ));
}

#[test]
fn oversized_stream_and_admission_are_rejected() {
    let bytes = encoded_image(SupportedMediaKind::Png);
    let tenant = tenant();
    let strict_limits = MediaValidationLimits::new(
        (bytes.len() - 1) as u64,
        MAX_VIDEO_UPLOAD_BYTES,
        MAX_DECODED_IMAGE_PIXELS,
    )
    .expect("strict limits");
    assert!(matches!(
        MediaValidationSession::begin(
            admission(&bytes, digest(&bytes)),
            &tenant,
            principal(),
            MediaContentType::new("image/png").expect("content type"),
            200,
            strict_limits,
        ),
        Err(MediaValidationError::TooLarge)
    ));

    let mut session = MediaValidationSession::begin(
        admission(&bytes, digest(&bytes)),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    let mut overflow = bytes.clone();
    overflow.push(0);
    assert_eq!(
        session.write_chunk(&overflow),
        Err(MediaValidationError::TooLarge)
    );
    assert!(matches!(
        session.finish(&tenant, principal(), 200),
        Err(MediaValidationError::InvalidState)
    ));

    let pixel_limits =
        MediaValidationLimits::new(MAX_IMAGE_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES, 1)
            .expect("pixel limits");
    let mut pixel_bomb = MediaValidationSession::begin(
        admission(&bytes, digest(&bytes)),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        pixel_limits,
    )
    .expect("session");
    pixel_bomb.write_chunk(&bytes).expect("chunk");
    assert!(matches!(
        pixel_bomb.finish(&tenant, principal(), 200),
        Err(MediaValidationError::ResolutionTooHigh)
    ));
}

#[test]
fn hash_and_declared_type_mismatches_are_rejected() {
    let bytes = encoded_image(SupportedMediaKind::Png);
    let tenant = tenant();
    let mut bad_hash = MediaValidationSession::begin(
        admission(&bytes, MediaContentHash::from_digest([9; 32])),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    bad_hash.write_chunk(&bytes).expect("chunk");
    assert!(matches!(
        bad_hash.finish(&tenant, principal(), 200),
        Err(MediaValidationError::HashMismatch)
    ));

    let mut wrong_type = MediaValidationSession::begin(
        admission(&bytes, digest(&bytes)),
        &tenant,
        principal(),
        MediaContentType::new("image/jpeg").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    wrong_type.write_chunk(&bytes).expect("chunk");
    assert!(matches!(
        wrong_type.finish(&tenant, principal(), 200),
        Err(MediaValidationError::ContentTypeMismatch)
    ));

    let mut corrupt = bytes.clone();
    let image_data = corrupt
        .windows(4)
        .position(|window| window == b"IDAT")
        .expect("image data");
    corrupt[image_data + 4] ^= 0xff;
    let mut invalid_decode = MediaValidationSession::begin(
        admission(&corrupt, digest(&corrupt)),
        &tenant,
        principal(),
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits(),
    )
    .expect("session");
    invalid_decode.write_chunk(&corrupt).expect("chunk");
    assert!(matches!(
        invalid_decode.finish(&tenant, principal(), 200),
        Err(MediaValidationError::DecodeFailed)
    ));
}
