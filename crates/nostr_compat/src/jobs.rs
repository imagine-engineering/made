use std::{collections::BTreeSet, fmt};

use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, JobCommand, JobCommandKind, JobCommandType,
    JobError, JobIdentity, OperationId, PrincipalId,
};
use uuid::Uuid;

use crate::{
    CanonicalEvent, EventId, PublicKey, SignedEvent, TimestampPolicy, VerificationError,
    generated_kinds::{
        KIND_JOB_ACCEPTED, KIND_JOB_CANCEL, KIND_JOB_ERROR, KIND_JOB_PROGRESS, KIND_JOB_REQUEST,
        KIND_JOB_RESULT,
    },
    verification::MAX_EVENT_CONTENT_BYTES,
    verify_signed_event,
};

pub const MAX_JOB_ANCESTRY_DEPTH: usize = 8;
const JOB_OPERATION_NAMESPACE: Uuid = Uuid::from_u128(0x9bc54d13_5f93_527b_aca7_3072bf16b1ca);

#[derive(Clone, Eq, PartialEq)]
pub struct SignedJobCommand {
    source_event_id: EventId,
    command: JobCommand,
    ancestry: Vec<JobIdentity>,
    content: String,
}

impl SignedJobCommand {
    pub const fn source_event_id(&self) -> EventId {
        self.source_event_id
    }

    pub const fn command(&self) -> &JobCommand {
        &self.command
    }

    pub fn ancestry(&self) -> &[JobIdentity] {
        &self.ancestry
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

impl fmt::Debug for SignedJobCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SignedJobCommand")
            .field("source_event_id", &self.source_event_id)
            .field("command", &self.command)
            .field("ancestry", &self.ancestry)
            .field("content", &"<redacted>")
            .finish()
    }
}

pub fn parse_signed_job_event(
    event: &SignedEvent,
    timestamp_policy: TimestampPolicy,
    expected_community_id: CommunityId,
    mut resolve_principal: impl FnMut(PublicKey) -> Option<PrincipalId>,
) -> Result<SignedJobCommand, JobNostrCodecError> {
    verify_signed_event(event, timestamp_policy)?;
    let command_type = command_type_for_kind(event.event.kind)?;
    let tags = parse_job_tags(
        &event.event.tags,
        expected_community_id,
        command_type == JobCommandType::Request,
    )?;
    let actor_principal_id = resolve_principal(event.event.public_key)
        .ok_or(JobNostrCodecError::UnknownPrincipal(event.event.public_key))?;
    let kind = match command_type {
        JobCommandType::Request => {
            let target_public_key = tags.target_executor.ok_or(JobNostrCodecError::InvalidTags(
                "job request requires one target p tag",
            ))?;
            let target_executor_principal_id = resolve_principal(target_public_key)
                .ok_or(JobNostrCodecError::UnknownPrincipal(target_public_key))?;
            JobCommandKind::Request {
                requester_principal_id: actor_principal_id,
                target_executor_principal_id,
            }
        }
        JobCommandType::Accept => JobCommandKind::Accept {
            executor_principal_id: actor_principal_id,
        },
        JobCommandType::Progress => JobCommandKind::Progress {
            executor_principal_id: actor_principal_id,
        },
        JobCommandType::Result => JobCommandKind::Result {
            executor_principal_id: actor_principal_id,
        },
        JobCommandType::Cancel => JobCommandKind::Cancel { actor_principal_id },
        JobCommandType::Error => JobCommandKind::Error { actor_principal_id },
    };
    let occurred_at_millis = event
        .event
        .created_at
        .checked_mul(1_000)
        .ok_or(JobNostrCodecError::InvalidTimestamp)?;
    let identity = JobIdentity::new(expected_community_id, tags.job_id)?;
    let command = JobCommand::new(
        identity,
        operation_id(expected_community_id, event.claimed_id),
        tags.version,
        occurred_at_millis,
        kind,
    )?;
    Ok(SignedJobCommand {
        source_event_id: event.claimed_id,
        command,
        ancestry: tags.ancestry,
        content: event.event.content.clone(),
    })
}

pub fn canonical_job_event(
    author: PublicKey,
    command: &JobCommand,
    target_executor: Option<PublicKey>,
    ancestry: &[JobIdentity],
    content: impl Into<String>,
) -> Result<CanonicalEvent, JobNostrCodecError> {
    if !command.occurred_at_millis().is_multiple_of(1_000) {
        return Err(JobNostrCodecError::InvalidTimestamp);
    }
    let content = content.into();
    if content.len() > MAX_EVENT_CONTENT_BYTES {
        return Err(JobNostrCodecError::ContentTooLarge {
            actual: content.len(),
            maximum: MAX_EVENT_CONTENT_BYTES,
        });
    }
    let command_type = command.kind().command_type();
    if (command_type == JobCommandType::Request) != target_executor.is_some() {
        return Err(JobNostrCodecError::InvalidTags(
            "only a job request carries one target p tag",
        ));
    }
    validate_ancestry(command.identity(), ancestry)?;

    let mut tags = vec![
        vec![
            "h".to_owned(),
            command.identity().community_id().to_string(),
        ],
        vec!["job".to_owned(), command.identity().job_id().to_string()],
        vec!["version".to_owned(), command.version().get().to_string()],
    ];
    if let Some(target_executor) = target_executor {
        tags.push(vec!["p".to_owned(), target_executor.to_hex()]);
    }
    tags.extend(
        ancestry
            .iter()
            .map(|ancestor| vec!["parent".to_owned(), ancestor.job_id().to_string()]),
    );
    Ok(CanonicalEvent::new(
        author,
        command.occurred_at_millis() / 1_000,
        kind_for_command(command_type)?,
        tags,
        content,
    ))
}

struct JobTags {
    job_id: AggregateId,
    version: AggregateVersion,
    target_executor: Option<PublicKey>,
    ancestry: Vec<JobIdentity>,
}

fn parse_job_tags(
    tags: &[Vec<String>],
    expected_community_id: CommunityId,
    is_request: bool,
) -> Result<JobTags, JobNostrCodecError> {
    if tags.len() > 4 + MAX_JOB_ANCESTRY_DEPTH {
        return Err(JobNostrCodecError::InvalidAncestry);
    }
    let mut community_id = None;
    let mut job_id = None;
    let mut version = None;
    let mut target_executor = None;
    let mut ancestry_ids = Vec::new();
    for tag in tags {
        let [name, value] = tag.as_slice() else {
            return Err(JobNostrCodecError::InvalidTags(
                "job tags must contain exactly two values",
            ));
        };
        match name.as_str() {
            "h" => set_once(
                &mut community_id,
                parse_canonical_uuid(value)
                    .map(CommunityId::from_uuid)
                    .ok_or(JobNostrCodecError::InvalidTags(
                        "job h tag must be one canonical community UUID",
                    ))?,
            )?,
            "job" => set_once(
                &mut job_id,
                parse_canonical_uuid(value)
                    .map(AggregateId::from_uuid)
                    .ok_or(JobNostrCodecError::InvalidTags(
                        "job tag must be one canonical job UUID",
                    ))?,
            )?,
            "version" => set_once(
                &mut version,
                parse_version(value).ok_or(JobNostrCodecError::InvalidTags(
                    "version tag must be one canonical positive integer",
                ))?,
            )?,
            "p" if is_request => set_once(
                &mut target_executor,
                PublicKey::from_hex(value).map_err(|_| {
                    JobNostrCodecError::InvalidTags(
                        "job request p tag must be one canonical public key",
                    )
                })?,
            )?,
            "parent" => ancestry_ids.push(
                parse_canonical_uuid(value)
                    .map(AggregateId::from_uuid)
                    .ok_or(JobNostrCodecError::InvalidAncestry)?,
            ),
            _ => {
                return Err(JobNostrCodecError::InvalidTags(
                    "job event contains an unsupported tag",
                ));
            }
        }
    }
    if community_id != Some(expected_community_id) {
        return Err(JobNostrCodecError::TenantMismatch);
    }
    let job_id = job_id.ok_or(JobNostrCodecError::InvalidTags(
        "job event is missing its job tag",
    ))?;
    let version = version.ok_or(JobNostrCodecError::InvalidTags(
        "job event is missing its version tag",
    ))?;
    if is_request != target_executor.is_some() {
        return Err(JobNostrCodecError::InvalidTags(
            "only a job request carries one target p tag",
        ));
    }
    let identity = JobIdentity::new(expected_community_id, job_id)?;
    let ancestry = ancestry_ids
        .into_iter()
        .map(|ancestor_id| JobIdentity::new(expected_community_id, ancestor_id))
        .collect::<Result<Vec<_>, _>>()?;
    validate_ancestry(identity, &ancestry)?;
    Ok(JobTags {
        job_id,
        version,
        target_executor,
        ancestry,
    })
}

fn validate_ancestry(
    identity: JobIdentity,
    ancestry: &[JobIdentity],
) -> Result<(), JobNostrCodecError> {
    if ancestry.len() > MAX_JOB_ANCESTRY_DEPTH {
        return Err(JobNostrCodecError::InvalidAncestry);
    }
    let mut ancestor_ids = BTreeSet::new();
    for ancestor in ancestry {
        if ancestor.community_id() != identity.community_id()
            || ancestor.job_id() == identity.job_id()
            || !ancestor_ids.insert(ancestor.job_id())
        {
            return Err(JobNostrCodecError::InvalidAncestry);
        }
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T) -> Result<(), JobNostrCodecError> {
    if slot.replace(value).is_some() {
        return Err(JobNostrCodecError::InvalidTags(
            "job event contains a duplicate singleton tag",
        ));
    }
    Ok(())
}

fn parse_canonical_uuid(value: &str) -> Option<Uuid> {
    let parsed = Uuid::parse_str(value).ok()?;
    (parsed.to_string() == value && !parsed.is_nil()).then_some(parsed)
}

fn parse_version(value: &str) -> Option<AggregateVersion> {
    let parsed = value.parse::<u64>().ok()?;
    if parsed.to_string() != value {
        return None;
    }
    AggregateVersion::new(parsed)
}

fn command_type_for_kind(kind: u16) -> Result<JobCommandType, JobNostrCodecError> {
    match u32::from(kind) {
        KIND_JOB_REQUEST => Ok(JobCommandType::Request),
        KIND_JOB_ACCEPTED => Ok(JobCommandType::Accept),
        KIND_JOB_PROGRESS => Ok(JobCommandType::Progress),
        KIND_JOB_RESULT => Ok(JobCommandType::Result),
        KIND_JOB_CANCEL => Ok(JobCommandType::Cancel),
        KIND_JOB_ERROR => Ok(JobCommandType::Error),
        _ => Err(JobNostrCodecError::UnsupportedKind(kind)),
    }
}

fn kind_for_command(command: JobCommandType) -> Result<u16, JobNostrCodecError> {
    let kind = match command {
        JobCommandType::Request => KIND_JOB_REQUEST,
        JobCommandType::Accept => KIND_JOB_ACCEPTED,
        JobCommandType::Progress => KIND_JOB_PROGRESS,
        JobCommandType::Result => KIND_JOB_RESULT,
        JobCommandType::Cancel => KIND_JOB_CANCEL,
        JobCommandType::Error => KIND_JOB_ERROR,
    };
    u16::try_from(kind).map_err(|_| JobNostrCodecError::RegisteredKindOutOfRange(kind))
}

fn operation_id(community_id: CommunityId, event_id: EventId) -> OperationId {
    let mut source = [0_u8; 48];
    source[..16].copy_from_slice(community_id.as_uuid().as_bytes());
    source[16..].copy_from_slice(event_id.as_bytes());
    OperationId::from_uuid(Uuid::new_v5(&JOB_OPERATION_NAMESPACE, &source))
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum JobNostrCodecError {
    #[error(transparent)]
    Verification(#[from] VerificationError),
    #[error("unsupported signed job kind {0}")]
    UnsupportedKind(u16),
    #[error("registered signed job kind {0} exceeds the NIP-01 kind range")]
    RegisteredKindOutOfRange(u32),
    #[error("invalid signed job tags: {0}")]
    InvalidTags(&'static str),
    #[error("signed job event belongs to another community")]
    TenantMismatch,
    #[error("signed job event contains invalid ancestry")]
    InvalidAncestry,
    #[error("signed job event references an unknown principal {0}")]
    UnknownPrincipal(PublicKey),
    #[error("signed job timestamp cannot be represented in milliseconds")]
    InvalidTimestamp,
    #[error("signed job content is {actual} bytes, maximum is {maximum}")]
    ContentTooLarge { actual: usize, maximum: usize },
    #[error(transparent)]
    Job(#[from] JobError),
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{Job, JobCommandOutcome};
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    use super::*;
    use crate::EventSignature;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn public_key(secret: [u8; 32]) -> PublicKey {
        let secret_key = SecretKey::from_slice(&secret).expect("secret key");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret_key);
        PublicKey::from_bytes(keypair.x_only_public_key().0.serialize())
    }

    fn sign(event: CanonicalEvent, secret: [u8; 32]) -> SignedEvent {
        let event_id = event.event_id().expect("event ID");
        let secret_key = SecretKey::from_slice(&secret).expect("secret key");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret_key);
        let signature = Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*event_id.as_bytes()), &keypair);
        SignedEvent {
            claimed_id: event_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    fn command(version: u64, occurred_at_seconds: u64, kind: JobCommandKind) -> JobCommand {
        JobCommand::new(
            JobIdentity::new(community(1), aggregate(10)).expect("identity"),
            OperationId::from_uuid(Uuid::from_u128(500 + u128::from(version))),
            AggregateVersion::new(version).expect("version"),
            occurred_at_seconds * 1_000,
            kind,
        )
        .expect("command")
    }

    fn parse(event: &SignedEvent) -> Result<SignedJobCommand, JobNostrCodecError> {
        let requester_key = public_key([1; 32]);
        let executor_key = public_key([2; 32]);
        parse_signed_job_event(
            event,
            TimestampPolicy::Historical,
            community(1),
            |key| match key {
                key if key == requester_key => Some(principal(1)),
                key if key == executor_key => Some(principal(2)),
                _ => None,
            },
        )
    }

    fn event(
        command: &JobCommand,
        author_secret: [u8; 32],
        target: Option<PublicKey>,
        ancestry: &[JobIdentity],
        content: &str,
    ) -> SignedEvent {
        sign(
            canonical_job_event(
                public_key(author_secret),
                command,
                target,
                ancestry,
                content,
            )
            .expect("canonical job event"),
            author_secret,
        )
    }

    #[test]
    fn golden_job_kinds_translate_to_exact_commands_and_wire_tags() {
        let requester_key = public_key([1; 32]);
        let executor_key = public_key([2; 32]);
        let cases = [
            (
                JobCommandKind::Request {
                    requester_principal_id: principal(1),
                    target_executor_principal_id: principal(2),
                },
                requester_key,
                Some(executor_key),
                KIND_JOB_REQUEST,
                "request",
            ),
            (
                JobCommandKind::Accept {
                    executor_principal_id: principal(2),
                },
                executor_key,
                None,
                KIND_JOB_ACCEPTED,
                "accept",
            ),
            (
                JobCommandKind::Progress {
                    executor_principal_id: principal(2),
                },
                executor_key,
                None,
                KIND_JOB_PROGRESS,
                "progress",
            ),
            (
                JobCommandKind::Result {
                    executor_principal_id: principal(2),
                },
                executor_key,
                None,
                KIND_JOB_RESULT,
                "result",
            ),
            (
                JobCommandKind::Cancel {
                    actor_principal_id: principal(1),
                },
                requester_key,
                None,
                KIND_JOB_CANCEL,
                "cancel",
            ),
            (
                JobCommandKind::Error {
                    actor_principal_id: principal(2),
                },
                executor_key,
                None,
                KIND_JOB_ERROR,
                "error",
            ),
        ];
        for (index, (kind, author, target, expected_kind, content)) in cases.into_iter().enumerate()
        {
            let version = u64::try_from(index + 1).expect("version fits");
            let command = command(version, version + 10, kind);
            let canonical = canonical_job_event(author, &command, target, &[], content)
                .expect("canonical event");
            assert_eq!(u32::from(canonical.kind), expected_kind);
            assert_eq!(canonical.tags[0], ["h", &community(1).to_string()]);
            assert_eq!(canonical.tags[1], ["job", &aggregate(10).to_string()]);
            assert_eq!(canonical.tags[2], ["version", &version.to_string()]);
            assert_eq!(canonical.content, content);

            let signed = sign(
                canonical,
                if author == requester_key {
                    [1; 32]
                } else {
                    [2; 32]
                },
            );
            let parsed = parse(&signed).expect("parsed job event");
            assert_eq!(parsed.command().kind(), kind);
            assert_eq!(parsed.command().version(), command.version());
            assert_eq!(
                parsed.command().occurred_at_millis(),
                command.occurred_at_millis()
            );
            assert_eq!(parsed.content(), content);
            assert_eq!(parsed.source_event_id(), signed.claimed_id);
        }
    }

    #[test]
    fn golden_job_traces_apply_progress_duplicate_cancel_and_error() {
        let requester_key = public_key([1; 32]);
        let executor_key = public_key([2; 32]);
        let request = command(
            1,
            11,
            JobCommandKind::Request {
                requester_principal_id: principal(1),
                target_executor_principal_id: principal(2),
            },
        );
        let request_event = event(&request, [1; 32], Some(executor_key), &[], "build");
        let parsed_request = parse(&request_event).expect("request");
        let mut job = Job::request(parsed_request.command().clone()).expect("requested job");

        for (version, kind, content) in [
            (
                2,
                JobCommandKind::Accept {
                    executor_principal_id: principal(2),
                },
                "accepted",
            ),
            (
                3,
                JobCommandKind::Progress {
                    executor_principal_id: principal(2),
                },
                "halfway",
            ),
            (
                4,
                JobCommandKind::Result {
                    executor_principal_id: principal(2),
                },
                "done",
            ),
        ] {
            let transition = command(version, version + 10, kind);
            let signed = event(&transition, [2; 32], None, &[], content);
            let parsed = parse(&signed).expect("transition");
            assert_eq!(
                job.apply(parsed.command().clone()),
                Ok(JobCommandOutcome::Applied)
            );
            assert_eq!(
                job.apply(parsed.command().clone()),
                Ok(JobCommandOutcome::Unchanged)
            );
        }

        let mut cancelled = Job::request(parsed_request.command().clone()).expect("cancel job");
        let cancel = command(
            2,
            13,
            JobCommandKind::Cancel {
                actor_principal_id: principal(1),
            },
        );
        assert_eq!(
            cancelled.apply(
                parse(&event(&cancel, [1; 32], None, &[], "stop"))
                    .expect("cancel")
                    .command()
                    .clone()
            ),
            Ok(JobCommandOutcome::Applied)
        );

        let mut failed = Job::request(parsed_request.command().clone()).expect("failed job");
        let failure = command(
            2,
            13,
            JobCommandKind::Error {
                actor_principal_id: principal(2),
            },
        );
        assert_eq!(
            failed.apply(
                parse(&event(&failure, [2; 32], None, &[], "failed"))
                    .expect("error")
                    .command()
                    .clone()
            ),
            Ok(JobCommandOutcome::Applied)
        );
        assert_ne!(requester_key, executor_key);
    }

    #[test]
    fn malformed_ancestry_tenant_and_signature_fail_before_commands() {
        let executor_key = public_key([2; 32]);
        let request = command(
            1,
            11,
            JobCommandKind::Request {
                requester_principal_id: principal(1),
                target_executor_principal_id: principal(2),
            },
        );
        let parent = JobIdentity::new(community(1), aggregate(20)).expect("parent");
        let mut duplicate_parent = canonical_job_event(
            public_key([1; 32]),
            &request,
            Some(executor_key),
            &[parent],
            "build",
        )
        .expect("event");
        duplicate_parent
            .tags
            .push(vec!["parent".to_owned(), aggregate(20).to_string()]);
        assert_eq!(
            parse(&sign(duplicate_parent, [1; 32])),
            Err(JobNostrCodecError::InvalidAncestry)
        );

        let mut foreign = canonical_job_event(
            public_key([1; 32]),
            &request,
            Some(executor_key),
            &[],
            "build",
        )
        .expect("event");
        foreign.tags[0][1] = community(2).to_string();
        assert_eq!(
            parse(&sign(foreign, [1; 32])),
            Err(JobNostrCodecError::TenantMismatch)
        );

        let mut tampered = event(&request, [1; 32], Some(executor_key), &[], "build");
        tampered.event.content.push('!');
        assert!(matches!(
            parse(&tampered),
            Err(JobNostrCodecError::Verification(
                VerificationError::InvalidEventId { .. }
            ))
        ));

        assert!(matches!(
            canonical_job_event(
                public_key([1; 32]),
                &request,
                Some(executor_key),
                &[],
                "x".repeat(MAX_EVENT_CONTENT_BYTES + 1),
            ),
            Err(JobNostrCodecError::ContentTooLarge { .. })
        ));
    }
}
