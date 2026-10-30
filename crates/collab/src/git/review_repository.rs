use std::collections::BTreeSet;

use collaboration_domain::{
    AggregateId, AggregateVersion, CiArtifactDigest, CiArtifactLink, CiCheckRun, CiCheckStatus,
    CiCheckSuite, CiCheckSuiteIdentity, CiCheckSuiteRecordFields, CiExternalLink, CiLabel,
    CiOutputText, CiWorkflowLink, GitCommitId, IntegrityAlgorithm, IntegrityReference,
    PatchRevisionNumber, Provenance, Review, ReviewIdentity, ReviewRecordFields, SourceRecordId,
    SourceSystem, TenantContext,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Postgres, Row, Transaction};

const MAX_CI_SUITES_PER_REVIEW: usize = 1_000;
const MAX_SOURCE_VERSION_BYTES: usize = 256;
const MAX_INTEGRITY_VALUE_BYTES: usize = 1_024;
const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true)";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewProjection {
    review: Review,
    ci_suites: Vec<CiCheckSuite>,
    provenance: Provenance,
}

impl ReviewProjection {
    pub fn new(
        review: Review,
        mut ci_suites: Vec<CiCheckSuite>,
        provenance: Provenance,
    ) -> Result<Self, ReviewRepositoryError> {
        validate_provenance(&provenance)?;
        if ci_suites.len() > MAX_CI_SUITES_PER_REVIEW {
            return Err(ReviewRepositoryError::InvalidProjection);
        }
        ci_suites.sort_by_key(|suite| suite.fields().identity.suite_id());
        let review_identity = &review.fields().identity;
        let revisions = review
            .fields()
            .revisions
            .iter()
            .map(|revision| (revision.number, &revision.head_commit))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut suite_ids = BTreeSet::new();
        for suite in &ci_suites {
            let identity = &suite.fields().identity;
            if identity.review() != review_identity
                || revisions.get(&identity.revision()) != Some(&identity.head_commit())
                || !suite_ids.insert(identity.suite_id())
            {
                return Err(ReviewRepositoryError::InvalidProjection);
            }
        }
        Ok(Self {
            review,
            ci_suites,
            provenance,
        })
    }

    pub const fn review(&self) -> &Review {
        &self.review
    }

    pub fn ci_suites(&self) -> &[CiCheckSuite] {
        &self.ci_suites
    }

    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewProjectionWriteOutcome {
    Inserted,
    Replaced,
    Rebuilt,
}

#[derive(Clone)]
pub struct ReviewProjectionRepository {
    pool: PgPool,
}

impl ReviewProjectionRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn replace(
        &self,
        tenant: &TenantContext,
        projection: &ReviewProjection,
    ) -> Result<ReviewProjectionWriteOutcome, ReviewRepositoryError> {
        let community_id = projection.review.fields().identity.branch().community_id();
        if tenant.community_id() != community_id {
            return Err(ReviewRepositoryError::TenantMismatch);
        }
        let encoded = EncodedProjection::new(projection)?;
        let current_revision = projection
            .review
            .current_revision()
            .ok_or(ReviewRepositoryError::InvalidProjection)?;
        let mut transaction = self.pool.begin().await?;
        set_tenant(&mut transaction, community_id.as_uuid()).await?;
        let existing = sqlx::query(
            "SELECT aggregate_version::bigint AS aggregate_version, projection_generation::bigint AS projection_generation, review_hash, projection_hash FROM public.collaboration_git_review_projections WHERE community_id = $1 AND review_id = $2 FOR UPDATE",
        )
        .bind(community_id.as_uuid())
        .bind(projection.review.fields().identity.review_id().as_uuid())
        .fetch_optional(&mut *transaction)
        .await?;

        let (projection_generation, outcome) = if let Some(existing) = existing {
            let stored_version = u64_from_i64(existing.try_get("aggregate_version")?)?;
            let incoming_version = projection.review.fields().version.get();
            if stored_version > incoming_version {
                return Err(ReviewRepositoryError::StaleReviewVersion);
            }
            let stored_review_hash: Vec<u8> = existing.try_get("review_hash")?;
            if stored_version == incoming_version
                && stored_review_hash.as_slice() != encoded.review_hash
            {
                return Err(ReviewRepositoryError::ConflictingReviewVersion);
            }
            let stored_projection_hash: Vec<u8> = existing.try_get("projection_hash")?;
            let generation = u64_from_i64(existing.try_get("projection_generation")?)?
                .checked_add(1)
                .ok_or(ReviewRepositoryError::VersionExhausted)?;
            let outcome = if stored_projection_hash.as_slice() == encoded.projection_hash {
                ReviewProjectionWriteOutcome::Rebuilt
            } else {
                ReviewProjectionWriteOutcome::Replaced
            };
            sqlx::query(
                "DELETE FROM public.collaboration_git_ci_projections WHERE community_id = $1 AND review_id = $2",
            )
            .bind(community_id.as_uuid())
            .bind(projection.review.fields().identity.review_id().as_uuid())
            .execute(&mut *transaction)
            .await?;
            sqlx::query(
                "UPDATE public.collaboration_git_review_projections SET repository_id = $3, current_revision = $4, current_head_commit = $5, aggregate_version = $6, projection_generation = $7, ci_suite_count = $8, review_payload = $9, review_hash = $10, projection_hash = $11, source_system = $12, source_record_id = $13, source_version = $14, source_observed_at = to_timestamp($15::double precision / 1000), integrity_algorithm = $16, integrity_value = $17, updated_at = now() WHERE community_id = $1 AND review_id = $2",
            )
            .bind(community_id.as_uuid())
            .bind(projection.review.fields().identity.review_id().as_uuid())
            .bind(projection.review.fields().identity.repository_id().as_uuid())
            .bind(i64_from_u64(current_revision.number.get())?)
            .bind(current_revision.head_commit.as_str())
            .bind(i64_from_u64(projection.review.fields().version.get())?)
            .bind(i64_from_u64(generation)?)
            .bind(i32::try_from(encoded.suites.len()).map_err(|_| ReviewRepositoryError::InvalidProjection)?)
            .bind(encoded.review_payload.clone())
            .bind(encoded.review_hash.to_vec())
            .bind(encoded.projection_hash.to_vec())
            .bind(source_system_name(projection.provenance.source_system))
            .bind(projection.provenance.source_record_id.as_str())
            .bind(projection.provenance.source_version.as_deref())
            .bind(i64_from_u64(projection.provenance.observed_at_millis)?)
            .bind(projection.provenance.integrity.as_ref().map(|integrity| integrity_algorithm_name(integrity.algorithm)))
            .bind(projection.provenance.integrity.as_ref().map(|integrity| integrity.value.as_str()))
            .execute(&mut *transaction)
            .await?;
            (generation, outcome)
        } else {
            sqlx::query(
                "INSERT INTO public.collaboration_git_review_projections (community_id, review_id, repository_id, current_revision, current_head_commit, aggregate_version, projection_generation, ci_suite_count, review_payload, review_hash, projection_hash, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, 1, $7, $8, $9, $10, $11, $12, $13, to_timestamp($14::double precision / 1000), $15, $16, now(), now())",
            )
            .bind(community_id.as_uuid())
            .bind(projection.review.fields().identity.review_id().as_uuid())
            .bind(projection.review.fields().identity.repository_id().as_uuid())
            .bind(i64_from_u64(current_revision.number.get())?)
            .bind(current_revision.head_commit.as_str())
            .bind(i64_from_u64(projection.review.fields().version.get())?)
            .bind(i32::try_from(encoded.suites.len()).map_err(|_| ReviewRepositoryError::InvalidProjection)?)
            .bind(encoded.review_payload.clone())
            .bind(encoded.review_hash.to_vec())
            .bind(encoded.projection_hash.to_vec())
            .bind(source_system_name(projection.provenance.source_system))
            .bind(projection.provenance.source_record_id.as_str())
            .bind(projection.provenance.source_version.as_deref())
            .bind(i64_from_u64(projection.provenance.observed_at_millis)?)
            .bind(projection.provenance.integrity.as_ref().map(|integrity| integrity_algorithm_name(integrity.algorithm)))
            .bind(projection.provenance.integrity.as_ref().map(|integrity| integrity.value.as_str()))
            .execute(&mut *transaction)
            .await?;
            (1, ReviewProjectionWriteOutcome::Inserted)
        };

        insert_ci_suites(
            &mut transaction,
            projection,
            &encoded.suites,
            projection_generation,
        )
        .await?;
        transaction.commit().await?;
        Ok(outcome)
    }

    pub async fn load(
        &self,
        tenant: &TenantContext,
        review_id: AggregateId,
    ) -> Result<Option<ReviewProjection>, ReviewRepositoryError> {
        if review_id.as_uuid().is_nil() {
            return Err(ReviewRepositoryError::InvalidProjection);
        }
        let mut transaction = self.pool.begin().await?;
        set_tenant(&mut transaction, tenant.community_id().as_uuid()).await?;
        let Some(row) = sqlx::query(
            "SELECT repository_id, current_revision::bigint AS current_revision, current_head_commit, aggregate_version::bigint AS aggregate_version, projection_generation::bigint AS projection_generation, ci_suite_count, review_payload, review_hash, projection_hash, source_system, source_record_id, source_version, floor(extract(epoch FROM source_observed_at) * 1000)::bigint AS source_observed_at_millis, integrity_algorithm, integrity_value FROM public.collaboration_git_review_projections WHERE community_id = $1 AND review_id = $2 FOR SHARE",
        )
        .bind(tenant.community_id().as_uuid())
        .bind(review_id.as_uuid())
        .fetch_optional(&mut *transaction)
        .await?
        else {
            transaction.commit().await?;
            return Ok(None);
        };
        let generation = u64_from_i64(row.try_get("projection_generation")?)?;
        let suite_rows = sqlx::query(
            "SELECT suite_payload, suite_hash FROM public.collaboration_git_ci_projections WHERE community_id = $1 AND review_id = $2 AND projection_generation = $3 ORDER BY suite_id",
        )
        .bind(tenant.community_id().as_uuid())
        .bind(review_id.as_uuid())
        .bind(i64_from_u64(generation)?)
        .fetch_all(&mut *transaction)
        .await?;
        let expected_suite_count: i32 = row.try_get("ci_suite_count")?;
        if usize::try_from(expected_suite_count).ok() != Some(suite_rows.len()) {
            return Err(ReviewRepositoryError::CorruptProjection);
        }

        let review_payload: Value = row.try_get("review_payload")?;
        let review_fields: ReviewRecordFields = serde_json::from_value(review_payload.clone())
            .map_err(|_| ReviewRepositoryError::CorruptProjection)?;
        let review = Review::from_record(review_fields)
            .map_err(|_| ReviewRepositoryError::CorruptProjection)?;
        let mut ci_suites = Vec::with_capacity(suite_rows.len());
        let mut encoded_suites = Vec::with_capacity(suite_rows.len());
        for suite_row in suite_rows {
            let payload: Value = suite_row.try_get("suite_payload")?;
            let stored: StoredCiSuite = serde_json::from_value(payload.clone())
                .map_err(|_| ReviewRepositoryError::CorruptProjection)?;
            let suite = stored
                .to_domain()
                .map_err(|_| ReviewRepositoryError::CorruptProjection)?;
            let suite_hash = hash_json(&payload)?;
            let stored_hash: Vec<u8> = suite_row.try_get("suite_hash")?;
            if stored_hash.as_slice() != suite_hash {
                return Err(ReviewRepositoryError::CorruptProjection);
            }
            ci_suites.push(suite);
            encoded_suites.push(EncodedSuite {
                payload,
                hash: suite_hash,
            });
        }
        let provenance = provenance_from_row(&row)?;
        let projection = ReviewProjection::new(review, ci_suites, provenance)
            .map_err(|_| ReviewRepositoryError::CorruptProjection)?;
        validate_loaded_metadata(&projection, &row)?;
        let review_hash = hash_json(&review_payload)?;
        let stored_review_hash: Vec<u8> = row.try_get("review_hash")?;
        if stored_review_hash.as_slice() != review_hash {
            return Err(ReviewRepositoryError::CorruptProjection);
        }
        let projection_hash = projection_hash(&review_hash, &projection.ci_suites, &encoded_suites);
        let stored_projection_hash: Vec<u8> = row.try_get("projection_hash")?;
        if stored_projection_hash.as_slice() != projection_hash {
            return Err(ReviewRepositoryError::CorruptProjection);
        }
        transaction.commit().await?;
        Ok(Some(projection))
    }
}

async fn insert_ci_suites(
    transaction: &mut Transaction<'_, Postgres>,
    projection: &ReviewProjection,
    suites: &[EncodedSuite],
    projection_generation: u64,
) -> Result<(), ReviewRepositoryError> {
    for (suite, encoded) in projection.ci_suites.iter().zip(suites) {
        let fields = suite.fields();
        sqlx::query(
            "INSERT INTO public.collaboration_git_ci_projections (community_id, review_id, suite_id, repository_id, revision, head_commit, suite_status, aggregate_version, projection_generation, suite_payload, suite_hash, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now(), now())",
        )
        .bind(projection.review.fields().identity.branch().community_id().as_uuid())
        .bind(projection.review.fields().identity.review_id().as_uuid())
        .bind(fields.identity.suite_id().as_uuid())
        .bind(fields.identity.repository_id().as_uuid())
        .bind(i64_from_u64(fields.identity.revision().get())?)
        .bind(fields.identity.head_commit().as_str())
        .bind(ci_status_name(suite.status()))
        .bind(i64_from_u64(fields.version.get())?)
        .bind(i64_from_u64(projection_generation)?)
        .bind(encoded.payload.clone())
        .bind(encoded.hash.to_vec())
        .execute(&mut **transaction)
        .await?;
    }
    Ok(())
}

async fn set_tenant(
    transaction: &mut Transaction<'_, Postgres>,
    community_id: uuid::Uuid,
) -> Result<(), ReviewRepositoryError> {
    sqlx::query(SET_TENANT_SQL)
        .bind(community_id.to_string())
        .execute(&mut **transaction)
        .await?;
    Ok(())
}

struct EncodedProjection {
    review_payload: Value,
    review_hash: [u8; 32],
    suites: Vec<EncodedSuite>,
    projection_hash: [u8; 32],
}

impl EncodedProjection {
    fn new(projection: &ReviewProjection) -> Result<Self, ReviewRepositoryError> {
        let review_payload = serde_json::to_value(projection.review.fields())
            .map_err(|_| ReviewRepositoryError::InvalidProjection)?;
        let review_hash = hash_json(&review_payload)?;
        let suites = projection
            .ci_suites
            .iter()
            .map(|suite| {
                let payload = serde_json::to_value(StoredCiSuite::from_domain(suite))
                    .map_err(|_| ReviewRepositoryError::InvalidProjection)?;
                Ok(EncodedSuite {
                    hash: hash_json(&payload)?,
                    payload,
                })
            })
            .collect::<Result<Vec<_>, ReviewRepositoryError>>()?;
        let projection_hash = projection_hash(&review_hash, &projection.ci_suites, &suites);
        Ok(Self {
            review_payload,
            review_hash,
            suites,
            projection_hash,
        })
    }
}

struct EncodedSuite {
    payload: Value,
    hash: [u8; 32],
}

fn hash_json(value: &Value) -> Result<[u8; 32], ReviewRepositoryError> {
    let mut hasher = Sha256::new();
    hash_json_value(&mut hasher, value)?;
    Ok(hasher.finalize().into())
}

fn hash_json_value(hasher: &mut Sha256, value: &Value) -> Result<(), ReviewRepositoryError> {
    match value {
        Value::Null => hasher.update(b"n"),
        Value::Bool(value) => hasher.update(if *value { b"t" } else { b"f" }),
        Value::Number(value) => {
            hasher.update(b"d");
            hash_bytes(hasher, value.to_string().as_bytes())?;
        }
        Value::String(value) => {
            hasher.update(b"s");
            hash_bytes(hasher, value.as_bytes())?;
        }
        Value::Array(values) => {
            hasher.update(b"a");
            hash_length(hasher, values.len())?;
            for value in values {
                hash_json_value(hasher, value)?;
            }
        }
        Value::Object(values) => {
            hasher.update(b"o");
            hash_length(hasher, values.len())?;
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_unstable_by_key(|(name, _)| *name);
            for (name, value) in fields {
                hash_bytes(hasher, name.as_bytes())?;
                hash_json_value(hasher, value)?;
            }
        }
    }
    Ok(())
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), ReviewRepositoryError> {
    hash_length(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn hash_length(hasher: &mut Sha256, length: usize) -> Result<(), ReviewRepositoryError> {
    let length = u64::try_from(length).map_err(|_| ReviewRepositoryError::InvalidProjection)?;
    hasher.update(length.to_be_bytes());
    Ok(())
}

fn projection_hash(
    review_hash: &[u8; 32],
    suites: &[CiCheckSuite],
    encoded_suites: &[EncodedSuite],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-collaboration-review-projection-v1\0");
    hasher.update(review_hash);
    for (suite, encoded) in suites.iter().zip(encoded_suites) {
        hasher.update(suite.fields().identity.suite_id().as_uuid().as_bytes());
        hasher.update(encoded.hash);
    }
    hasher.finalize().into()
}

#[derive(Deserialize, Serialize)]
struct StoredCiSuite {
    suite_id: AggregateId,
    review: ReviewIdentity,
    revision: PatchRevisionNumber,
    head_commit: GitCommitId,
    workflow_id: AggregateId,
    workflow_run_id: AggregateId,
    workflow_label: StoredText,
    workflow_url: Option<String>,
    runs: Vec<StoredCiRun>,
    created_at_millis: u64,
    version: AggregateVersion,
}

impl StoredCiSuite {
    fn from_domain(suite: &CiCheckSuite) -> Self {
        let fields = suite.fields();
        Self {
            suite_id: fields.identity.suite_id(),
            review: fields.identity.review().clone(),
            revision: fields.identity.revision(),
            head_commit: fields.identity.head_commit().clone(),
            workflow_id: fields.workflow.workflow_id,
            workflow_run_id: fields.workflow.workflow_run_id,
            workflow_label: StoredText::from_label(&fields.workflow.label),
            workflow_url: fields
                .workflow
                .url
                .as_ref()
                .map(|url| url.as_str().to_owned()),
            runs: fields.runs.iter().map(StoredCiRun::from_domain).collect(),
            created_at_millis: fields.created_at_millis,
            version: fields.version,
        }
    }

    fn to_domain(&self) -> Result<CiCheckSuite, ReviewRepositoryError> {
        let identity = CiCheckSuiteIdentity::new(
            self.suite_id,
            self.review.clone(),
            self.revision,
            self.head_commit.clone(),
        )?;
        let workflow = CiWorkflowLink::new(
            self.workflow_id,
            self.workflow_run_id,
            self.workflow_label.to_label()?,
            self.workflow_url
                .as_ref()
                .map(|url| CiExternalLink::parse(url.clone()))
                .transpose()?,
        )?;
        let runs = self
            .runs
            .iter()
            .map(|run| run.to_domain(&identity))
            .collect::<Result<Vec<_>, ReviewRepositoryError>>()?;
        Ok(CiCheckSuite::from_record(CiCheckSuiteRecordFields {
            identity,
            workflow,
            runs,
            created_at_millis: self.created_at_millis,
            version: self.version,
        })?)
    }
}

#[derive(Deserialize, Serialize)]
struct StoredCiRun {
    check_run_id: AggregateId,
    label: StoredText,
    status: String,
    output: Option<StoredText>,
    artifacts: Vec<StoredArtifact>,
    queued_at_millis: u64,
    started_at_millis: Option<u64>,
    completed_at_millis: Option<u64>,
    version: AggregateVersion,
}

impl StoredCiRun {
    fn from_domain(run: &CiCheckRun) -> Self {
        Self {
            check_run_id: run.check_run_id,
            label: StoredText::from_label(&run.label),
            status: ci_status_name(run.status).to_owned(),
            output: run.output.as_ref().map(StoredText::from_output),
            artifacts: run
                .artifacts
                .iter()
                .map(StoredArtifact::from_domain)
                .collect(),
            queued_at_millis: run.queued_at_millis,
            started_at_millis: run.started_at_millis,
            completed_at_millis: run.completed_at_millis,
            version: run.version,
        }
    }

    fn to_domain(&self, suite: &CiCheckSuiteIdentity) -> Result<CiCheckRun, ReviewRepositoryError> {
        Ok(CiCheckRun {
            check_run_id: self.check_run_id,
            suite: suite.clone(),
            label: self.label.to_label()?,
            status: parse_ci_status(&self.status)?,
            output: self
                .output
                .as_ref()
                .map(StoredText::to_output)
                .transpose()?,
            artifacts: self
                .artifacts
                .iter()
                .map(StoredArtifact::to_domain)
                .collect::<Result<Vec<_>, ReviewRepositoryError>>()?,
            queued_at_millis: self.queued_at_millis,
            started_at_millis: self.started_at_millis,
            completed_at_millis: self.completed_at_millis,
            version: self.version,
        })
    }
}

#[derive(Deserialize, Serialize)]
struct StoredText {
    value: String,
    truncated: bool,
    sanitized: bool,
}

impl StoredText {
    fn from_label(value: &CiLabel) -> Self {
        Self {
            value: value.as_str().to_owned(),
            truncated: value.was_truncated(),
            sanitized: value.was_sanitized(),
        }
    }

    fn from_output(value: &CiOutputText) -> Self {
        Self {
            value: value.as_str().to_owned(),
            truncated: value.was_truncated(),
            sanitized: value.was_sanitized(),
        }
    }

    fn to_label(&self) -> Result<CiLabel, ReviewRepositoryError> {
        Ok(CiLabel::from_record(
            self.value.clone(),
            self.truncated,
            self.sanitized,
        )?)
    }

    fn to_output(&self) -> Result<CiOutputText, ReviewRepositoryError> {
        Ok(CiOutputText::from_record(
            self.value.clone(),
            self.truncated,
            self.sanitized,
        )?)
    }
}

#[derive(Deserialize, Serialize)]
struct StoredArtifact {
    artifact_id: AggregateId,
    label: StoredText,
    url: String,
    digest: Option<String>,
}

impl StoredArtifact {
    fn from_domain(artifact: &CiArtifactLink) -> Self {
        Self {
            artifact_id: artifact.artifact_id,
            label: StoredText::from_label(&artifact.label),
            url: artifact.url.as_str().to_owned(),
            digest: artifact
                .digest
                .as_ref()
                .map(|digest| digest.as_str().to_owned()),
        }
    }

    fn to_domain(&self) -> Result<CiArtifactLink, ReviewRepositoryError> {
        Ok(CiArtifactLink::new(
            self.artifact_id,
            self.label.to_label()?,
            CiExternalLink::parse(self.url.clone())?,
            self.digest
                .as_ref()
                .map(|digest| CiArtifactDigest::parse(digest.clone()))
                .transpose()?,
        )?)
    }
}

fn validate_loaded_metadata(
    projection: &ReviewProjection,
    row: &sqlx::postgres::PgRow,
) -> Result<(), ReviewRepositoryError> {
    let current = projection
        .review
        .current_revision()
        .ok_or(ReviewRepositoryError::CorruptProjection)?;
    let repository_id: uuid::Uuid = row.try_get("repository_id")?;
    let current_revision = u64_from_i64(row.try_get("current_revision")?)?;
    let current_head_commit: String = row.try_get("current_head_commit")?;
    let aggregate_version = u64_from_i64(row.try_get("aggregate_version")?)?;
    if repository_id
        != projection
            .review
            .fields()
            .identity
            .repository_id()
            .as_uuid()
        || current_revision != current.number.get()
        || current_head_commit != current.head_commit.as_str()
        || aggregate_version != projection.review.fields().version.get()
    {
        return Err(ReviewRepositoryError::CorruptProjection);
    }
    Ok(())
}

fn provenance_from_row(row: &sqlx::postgres::PgRow) -> Result<Provenance, ReviewRepositoryError> {
    let source_system = parse_source_system(row.try_get("source_system")?)?;
    let source_record_id = SourceRecordId::new(row.try_get::<String, _>("source_record_id")?)
        .ok_or(ReviewRepositoryError::CorruptProjection)?;
    let observed_at = u64_from_i64(row.try_get("source_observed_at_millis")?)?;
    let mut provenance = Provenance::new(source_system, source_record_id, observed_at);
    if let Some(source_version) = row.try_get::<Option<String>, _>("source_version")? {
        provenance = provenance.with_source_version(source_version);
    }
    let algorithm = row.try_get::<Option<String>, _>("integrity_algorithm")?;
    let value = row.try_get::<Option<String>, _>("integrity_value")?;
    match (algorithm, value) {
        (None, None) => {}
        (Some(algorithm), Some(value)) => {
            provenance = provenance.with_integrity(IntegrityReference {
                algorithm: parse_integrity_algorithm(&algorithm)?,
                value,
            });
        }
        _ => return Err(ReviewRepositoryError::CorruptProjection),
    }
    Ok(provenance)
}

fn validate_provenance(provenance: &Provenance) -> Result<(), ReviewRepositoryError> {
    if provenance
        .source_version
        .as_ref()
        .is_some_and(|version| version.is_empty() || version.len() > MAX_SOURCE_VERSION_BYTES)
        || provenance.integrity.as_ref().is_some_and(|integrity| {
            integrity.value.is_empty() || integrity.value.len() > MAX_INTEGRITY_VALUE_BYTES
        })
    {
        return Err(ReviewRepositoryError::InvalidProjection);
    }
    Ok(())
}

fn parse_ci_status(value: &str) -> Result<CiCheckStatus, ReviewRepositoryError> {
    match value {
        "pending" => Ok(CiCheckStatus::Pending),
        "running" => Ok(CiCheckStatus::Running),
        "success" => Ok(CiCheckStatus::Success),
        "failure" => Ok(CiCheckStatus::Failure),
        "cancelled" => Ok(CiCheckStatus::Cancelled),
        _ => Err(ReviewRepositoryError::CorruptProjection),
    }
}

const fn ci_status_name(value: CiCheckStatus) -> &'static str {
    match value {
        CiCheckStatus::Pending => "pending",
        CiCheckStatus::Running => "running",
        CiCheckStatus::Success => "success",
        CiCheckStatus::Failure => "failure",
        CiCheckStatus::Cancelled => "cancelled",
    }
}

const fn source_system_name(value: SourceSystem) -> &'static str {
    match value {
        SourceSystem::Zed => "zed",
        SourceSystem::Buzz => "buzz",
        SourceSystem::Nostr => "nostr",
        SourceSystem::Acp => "acp",
        SourceSystem::ExternalGit => "external_git",
    }
}

fn parse_source_system(value: &str) -> Result<SourceSystem, ReviewRepositoryError> {
    match value {
        "zed" => Ok(SourceSystem::Zed),
        "buzz" => Ok(SourceSystem::Buzz),
        "nostr" => Ok(SourceSystem::Nostr),
        "acp" => Ok(SourceSystem::Acp),
        "external_git" => Ok(SourceSystem::ExternalGit),
        _ => Err(ReviewRepositoryError::CorruptProjection),
    }
}

const fn integrity_algorithm_name(value: IntegrityAlgorithm) -> &'static str {
    match value {
        IntegrityAlgorithm::Sha256 => "sha256",
        IntegrityAlgorithm::NostrEventId => "nostr_event_id",
        IntegrityAlgorithm::GitObjectId => "git_object_id",
    }
}

fn parse_integrity_algorithm(value: &str) -> Result<IntegrityAlgorithm, ReviewRepositoryError> {
    match value {
        "sha256" => Ok(IntegrityAlgorithm::Sha256),
        "nostr_event_id" => Ok(IntegrityAlgorithm::NostrEventId),
        "git_object_id" => Ok(IntegrityAlgorithm::GitObjectId),
        _ => Err(ReviewRepositoryError::CorruptProjection),
    }
}

fn i64_from_u64(value: u64) -> Result<i64, ReviewRepositoryError> {
    i64::try_from(value).map_err(|_| ReviewRepositoryError::VersionExhausted)
}

fn u64_from_i64(value: i64) -> Result<u64, ReviewRepositoryError> {
    u64::try_from(value).map_err(|_| ReviewRepositoryError::CorruptProjection)
}

#[derive(Debug, thiserror::Error)]
pub enum ReviewRepositoryError {
    #[error("review projection does not match the trusted tenant")]
    TenantMismatch,
    #[error("review projection is invalid or exceeds a bound")]
    InvalidProjection,
    #[error("review projection version is stale")]
    StaleReviewVersion,
    #[error("review projection reuses a version with different review state")]
    ConflictingReviewVersion,
    #[error("stored review projection is corrupt or incomplete")]
    CorruptProjection,
    #[error("review projection version is exhausted")]
    VersionExhausted,
    #[error("review projection storage is unavailable")]
    Storage(#[from] sqlx::Error),
}

impl From<collaboration_domain::CiStatusError> for ReviewRepositoryError {
    fn from(_: collaboration_domain::CiStatusError) -> Self {
        Self::InvalidProjection
    }
}
