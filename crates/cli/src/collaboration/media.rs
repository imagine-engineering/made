use std::fmt;

use collab::media::{object_store::MediaRangeRequest, upload_admission::MediaUploadRequest};
use collaboration_domain::{
    MediaAttachmentLink, MediaContentHash, MediaContentType, MediaDescriptor, MediaIdentity,
    MediaMetadata, MediaObjectSelection, MediaVariantKind, OperationId,
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

const MAX_PATH_BYTES: usize = 4_096;

#[derive(Clone, Eq, PartialEq)]
pub struct MediaPath(String);

impl MediaPath {
    pub fn new(value: impl Into<String>) -> Result<Self, MediaCliError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.len() > MAX_PATH_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(MediaCliError::InvalidRequest);
        }
        Ok(Self(value))
    }

    pub fn expose_to_executor(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MediaPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MediaPath(<redacted>)")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MediaDownloadTarget {
    Stdout,
    File(MediaPath),
}

#[derive(Clone, Eq, PartialEq)]
pub enum MediaCliCommand {
    Upload {
        request: MediaUploadRequest,
        content_type: MediaContentType,
        source: MediaPath,
    },
    Download {
        identity: MediaIdentity,
        selection: MediaObjectSelection,
        range: Option<MediaRangeRequest>,
        target: MediaDownloadTarget,
    },
    Metadata {
        identity: MediaIdentity,
    },
    Attach {
        link: MediaAttachmentLink,
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MediaCliVerb {
    Upload,
    Download,
    Metadata,
    Attach,
}

impl MediaCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload.file",
            Self::Download => "media.get",
            Self::Metadata => "media.metadata",
            Self::Attach => "media.attach",
        }
    }
}

impl MediaCliCommand {
    const fn verb(&self) -> MediaCliVerb {
        match self {
            Self::Upload { .. } => MediaCliVerb::Upload,
            Self::Download { .. } => MediaCliVerb::Download,
            Self::Metadata { .. } => MediaCliVerb::Metadata,
            Self::Attach { .. } => MediaCliVerb::Attach,
        }
    }
}

impl fmt::Debug for MediaCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaProgress {
    pub transferred_bytes: u64,
    pub total_bytes: u64,
}

impl MediaProgress {
    fn is_valid(self) -> bool {
        self.total_bytes > 0 && self.transferred_bytes <= self.total_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaWriteReceipt {
    pub operation_id: OperationId,
    pub content_hash: MediaContentHash,
}

#[derive(Clone, Eq, PartialEq)]
pub enum MediaCliOutcome {
    Uploaded {
        metadata: MediaMetadata,
        progress: MediaProgress,
    },
    Downloaded {
        descriptor: MediaDescriptor,
        progress: MediaProgress,
        stdout_bytes: Option<Vec<u8>>,
    },
    Metadata(MediaMetadata),
    Attached(MediaWriteReceipt),
}

impl fmt::Debug for MediaCliOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let variant = match self {
            Self::Uploaded { .. } => "Uploaded",
            Self::Downloaded { .. } => "Downloaded",
            Self::Metadata(_) => "Metadata",
            Self::Attached(_) => "Attached",
        };
        formatter
            .debug_struct("MediaCliOutcome")
            .field("variant", &variant)
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaCliError {
    InvalidRequest,
    UnsupportedMedia,
    NotFound,
    Unavailable,
    PermissionDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl MediaCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "media_cli_invalid_request",
            Self::UnsupportedMedia => "media_cli_unsupported_media",
            Self::NotFound => "media_cli_not_found",
            Self::Unavailable => "media_cli_unavailable",
            Self::PermissionDenied => "media_cli_permission_denied",
            Self::PartialFailure => "media_cli_completion_unknown",
            Self::Unexpected => "media_cli_unexpected_response",
            Self::Conflict => "media_cli_conflict",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest | Self::UnsupportedMedia => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::PermissionDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait MediaCliExecutor {
    fn execute(&self, command: MediaCliCommand) -> Result<MediaCliOutcome, MediaCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaCliExecution {
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_media_command(
    executor: &impl MediaCliExecutor,
    command: MediaCliCommand,
) -> MediaCliExecution {
    let verb = command.verb();
    let target = match &command {
        MediaCliCommand::Download { target, .. } => Some(target.clone()),
        _ => None,
    };
    match executor.execute(command) {
        Ok(outcome) => success_output(verb, target.as_ref(), outcome)
            .unwrap_or_else(|| error_output(verb, MediaCliError::Unexpected)),
        Err(error) => error_output(verb, error),
    }
}

fn success_output(
    verb: MediaCliVerb,
    target: Option<&MediaDownloadTarget>,
    outcome: MediaCliOutcome,
) -> Option<MediaCliExecution> {
    let stdout = match (verb, outcome) {
        (MediaCliVerb::Upload, MediaCliOutcome::Uploaded { metadata, progress })
            if progress.is_valid()
                && progress.transferred_bytes == progress.total_bytes
                && progress.total_bytes == metadata.fields().byte_size.get() =>
        {
            json_bytes(json!({
                "command": verb.as_str(),
                "media": metadata_output(&metadata),
                "ok": true,
                "progress": progress_output(progress),
            }))
        }
        (MediaCliVerb::Metadata, MediaCliOutcome::Metadata(metadata)) => json_bytes(json!({
            "command": verb.as_str(), "media": metadata_output(&metadata), "ok": true,
        })),
        (MediaCliVerb::Attach, MediaCliOutcome::Attached(receipt)) => json_bytes(json!({
            "command": verb.as_str(),
            "content_hash": receipt.content_hash.to_lower_hex(),
            "ok": true,
            "operation_id": receipt.operation_id,
        })),
        (
            MediaCliVerb::Download,
            MediaCliOutcome::Downloaded {
                descriptor,
                progress,
                stdout_bytes,
            },
        ) if progress.is_valid()
            && progress.total_bytes == descriptor.byte_size().get()
            && progress.transferred_bytes == descriptor.byte_size().get() =>
        {
            match (target?, stdout_bytes) {
                (MediaDownloadTarget::Stdout, Some(bytes))
                    if u64::try_from(bytes.len()).ok() == Some(descriptor.byte_size().get()) =>
                {
                    bytes
                }
                (MediaDownloadTarget::File(_), None) => Vec::new(),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(MediaCliExecution {
        stdout,
        stderr: String::new(),
        exit_code: 0,
    })
}

fn error_output(verb: MediaCliVerb, error: MediaCliError) -> MediaCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    MediaCliExecution {
        stdout: Vec::new(),
        stderr: format!(
            "{}\n",
            json!({
                "command": verb.as_str(),
                "error": contract.category,
                "error_code": diagnostic,
                "message": diagnostic,
                "ok": false,
                "retryable": contract.retryable,
            })
        ),
        exit_code: contract.exit_class as i32,
    }
}

fn json_bytes(value: Value) -> Vec<u8> {
    format!("{value}\n").into_bytes()
}

fn metadata_output(metadata: &MediaMetadata) -> Value {
    let fields = metadata.fields();
    json!({
        "byte_size": fields.byte_size.get(),
        "community_id": fields.identity.community_id(),
        "content_hash": fields.identity.content_hash().to_lower_hex(),
        "content_type": fields.content_type.as_str(),
        "owner_principal_id": fields.owner_principal_id,
        "uploaded_at_millis": fields.uploaded_at_millis,
        "variants": fields.variants.iter().map(|variant| json!({
            "byte_size": variant.descriptor().byte_size().get(),
            "content_hash": variant.descriptor().content_hash().to_lower_hex(),
            "content_type": variant.descriptor().content_type().as_str(),
            "kind": variant_name(variant.kind()),
        })).collect::<Vec<_>>(),
    })
}

const fn variant_name(kind: MediaVariantKind) -> &'static str {
    match kind {
        MediaVariantKind::Thumbnail => "thumbnail",
        MediaVariantKind::Poster => "poster",
    }
}

fn progress_output(progress: MediaProgress) -> Value {
    json!({
        "total_bytes": progress.total_bytes,
        "transferred_bytes": progress.transferred_bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use collaboration_domain::{CommunityId, MediaByteSize, MediaMetadataFields, PrincipalId};
    use uuid::Uuid;

    use super::*;

    struct TestExecutor(RefCell<Option<Result<MediaCliOutcome, MediaCliError>>>);

    impl TestExecutor {
        fn returning(result: Result<MediaCliOutcome, MediaCliError>) -> Self {
            Self(RefCell::new(Some(result)))
        }
    }

    impl MediaCliExecutor for TestExecutor {
        fn execute(&self, _command: MediaCliCommand) -> Result<MediaCliOutcome, MediaCliError> {
            self.0.borrow_mut().take().expect("called once")
        }
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn content_hash() -> MediaContentHash {
        MediaContentHash::from_digest([2; 32])
    }

    fn identity() -> MediaIdentity {
        MediaIdentity::new(community_id(), content_hash()).expect("identity")
    }

    fn descriptor() -> MediaDescriptor {
        MediaDescriptor::new(
            content_hash(),
            MediaContentType::new("image/png").expect("type"),
            MediaByteSize::new(4).expect("size"),
        )
    }

    fn metadata() -> MediaMetadata {
        MediaMetadata::from_record(MediaMetadataFields {
            identity: identity(),
            owner_principal_id: PrincipalId::from_uuid(Uuid::from_u128(3)),
            content_type: MediaContentType::new("image/png").expect("type"),
            byte_size: MediaByteSize::new(4).expect("size"),
            uploaded_at_millis: 10,
            variants: Vec::new(),
        })
        .expect("metadata")
    }

    #[test]
    fn upload_metadata_and_progress_have_stable_json() {
        let source = MediaPath::new("/private/upload.png").expect("path");
        let command = MediaCliCommand::Upload {
            request: MediaUploadRequest::new(
                OperationId::from_uuid(Uuid::from_u128(4)),
                content_hash(),
                4,
            )
            .expect("request"),
            content_type: MediaContentType::new("image/png").expect("type"),
            source,
        };
        assert!(!format!("{command:?}").contains("/private/upload.png"));
        let output = execute_media_command(
            &TestExecutor::returning(Ok(MediaCliOutcome::Uploaded {
                metadata: metadata(),
                progress: MediaProgress {
                    transferred_bytes: 4,
                    total_bytes: 4,
                },
            })),
            command,
        );
        let stdout = String::from_utf8(output.stdout).expect("JSON");
        assert!(stdout.contains("image/png"));
        assert!(stdout.contains("transferred_bytes"));
    }

    #[test]
    fn download_emits_raw_bytes_only_for_explicit_stdout() {
        let bytes = vec![0, 1, 2, 3];
        let output = execute_media_command(
            &TestExecutor::returning(Ok(MediaCliOutcome::Downloaded {
                descriptor: descriptor(),
                progress: MediaProgress {
                    transferred_bytes: 4,
                    total_bytes: 4,
                },
                stdout_bytes: Some(bytes.clone()),
            })),
            MediaCliCommand::Download {
                identity: identity(),
                selection: MediaObjectSelection::Original,
                range: None,
                target: MediaDownloadTarget::Stdout,
            },
        );
        assert_eq!(output.stdout, bytes);

        let mismatch = execute_media_command(
            &TestExecutor::returning(Ok(MediaCliOutcome::Downloaded {
                descriptor: descriptor(),
                progress: MediaProgress {
                    transferred_bytes: 4,
                    total_bytes: 4,
                },
                stdout_bytes: Some(vec![0; 4]),
            })),
            MediaCliCommand::Download {
                identity: identity(),
                selection: MediaObjectSelection::Original,
                range: None,
                target: MediaDownloadTarget::File(MediaPath::new("out.png").expect("path")),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
        assert!(mismatch.stdout.is_empty());
    }

    #[test]
    fn unsupported_permission_and_complete_error_matrix_are_stable() {
        let cases = [
            (MediaCliError::InvalidRequest, "user_error", 1, false),
            (MediaCliError::UnsupportedMedia, "user_error", 1, false),
            (MediaCliError::NotFound, "not_found", 1, false),
            (MediaCliError::Unavailable, "network_error", 2, true),
            (MediaCliError::PartialFailure, "delivery_unknown", 2, false),
            (MediaCliError::PermissionDenied, "auth_error", 3, false),
            (MediaCliError::Unexpected, "error", 4, false),
            (MediaCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_media_command(
                &TestExecutor::returning(Err(error)),
                MediaCliCommand::Metadata {
                    identity: identity(),
                },
            );
            assert_eq!(output.exit_code, exit_code);
            let value: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(value["error"], category);
            assert_eq!(value["retryable"], retryable);
        }
    }

    #[test]
    fn invalid_progress_and_outcome_shape_fail_closed() {
        let invalid = execute_media_command(
            &TestExecutor::returning(Ok(MediaCliOutcome::Uploaded {
                metadata: metadata(),
                progress: MediaProgress {
                    transferred_bytes: 5,
                    total_bytes: 4,
                },
            })),
            MediaCliCommand::Metadata {
                identity: identity(),
            },
        );
        assert_eq!(invalid.exit_code, 4);
        assert!(invalid.stdout.is_empty());
    }
}
