use std::{collections::BTreeSet, error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    ChannelDescription, ChannelName, ChannelType, ChannelVisibility, CommunityId, PrincipalId,
    authorize,
};

const MAX_METADATA_BYTES: usize = 1_048_576;
const MAX_TEMPLATE_REFERENCES: usize = 128;
const MAX_REFERENCE_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelMetadataText(String);

impl ChannelMetadataText {
    pub fn new(value: impl Into<String>) -> Result<Self, ChannelMetadataError> {
        let value = value.into();
        if value.len() > MAX_METADATA_BYTES {
            return Err(ChannelMetadataError::ContentTooLarge);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ChannelTemplateReferenceKind {
    Persona,
    Team,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChannelTemplateBackend {
    Local,
    Provider(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelTemplateReference {
    pub kind: ChannelTemplateReferenceKind,
    pub id: String,
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub role: Option<String>,
    pub backend: Option<ChannelTemplateBackend>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelTemplate {
    pub template_id: AggregateId,
    pub name: ChannelName,
    pub description: Option<ChannelDescription>,
    pub channel_type: ChannelType,
    pub visibility: ChannelVisibility,
    pub canvas_template: Option<ChannelMetadataText>,
    pub references: Vec<ChannelTemplateReference>,
    pub is_builtin: bool,
    pub version: AggregateVersion,
}

impl ChannelTemplate {
    pub fn new(
        template_id: AggregateId,
        name: ChannelName,
        description: Option<ChannelDescription>,
        channel_type: ChannelType,
        visibility: ChannelVisibility,
        canvas_template: Option<ChannelMetadataText>,
        references: Vec<ChannelTemplateReference>,
        is_builtin: bool,
        version: AggregateVersion,
    ) -> Result<Self, ChannelMetadataError> {
        if !matches!(channel_type, ChannelType::Stream | ChannelType::Forum) {
            return Err(ChannelMetadataError::InvalidTemplateType);
        }
        if references.len() > MAX_TEMPLATE_REFERENCES {
            return Err(ChannelMetadataError::TooManyReferences);
        }
        let mut identities = BTreeSet::new();
        for reference in &references {
            validate_reference(reference)?;
            if !identities.insert((reference.kind, reference.id.as_str())) {
                return Err(ChannelMetadataError::DuplicateReference);
            }
        }
        if let Some(canvas) = &canvas_template {
            validate_placeholders(canvas.as_str())?;
        }
        Ok(Self {
            template_id,
            name,
            description,
            channel_type,
            visibility,
            canvas_template,
            references,
            is_builtin,
            version,
        })
    }

    pub fn render_canvas(&self, channel_name: &ChannelName) -> Option<String> {
        self.canvas_template.as_ref().map(|canvas| {
            canvas
                .as_str()
                .replace("{channel.name}", channel_name.as_str())
                .replace("{template.name}", self.name.as_str())
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelMetadataRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub topic: Option<ChannelMetadataText>,
    pub canvas: Option<ChannelMetadataText>,
    pub version: AggregateVersion,
    pub updated_by_principal_id: PrincipalId,
    pub updated_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelMetadataOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelMetadata {
    fields: ChannelMetadataRecordFields,
}

impl ChannelMetadata {
    pub const fn from_record(fields: ChannelMetadataRecordFields) -> Self {
        Self { fields }
    }

    pub const fn fields(&self) -> &ChannelMetadataRecordFields {
        &self.fields
    }

    pub fn set_topic(
        &mut self,
        expected_version: AggregateVersion,
        topic: Option<ChannelMetadataText>,
        updated_at_millis: u64,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelMetadataOutcome, ChannelMetadataError> {
        self.authorize_write(authorization)?;
        self.require_version(expected_version)?;
        if self.fields.topic == topic {
            return Ok(ChannelMetadataOutcome::Unchanged);
        }
        let next_version = self.prepare_update(updated_at_millis)?;
        self.fields.topic = topic;
        self.finish_update(next_version, updated_at_millis, authorization);
        Ok(ChannelMetadataOutcome::Applied)
    }

    pub fn set_canvas(
        &mut self,
        expected_version: AggregateVersion,
        canvas: Option<ChannelMetadataText>,
        updated_at_millis: u64,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelMetadataOutcome, ChannelMetadataError> {
        self.authorize_write(authorization)?;
        self.require_version(expected_version)?;
        if self.fields.canvas == canvas {
            return Ok(ChannelMetadataOutcome::Unchanged);
        }
        let next_version = self.prepare_update(updated_at_millis)?;
        self.fields.canvas = canvas;
        self.finish_update(next_version, updated_at_millis, authorization);
        Ok(ChannelMetadataOutcome::Applied)
    }

    fn authorize_write(
        &self,
        request: &AuthorizationRequest<'_>,
    ) -> Result<(), ChannelMetadataError> {
        if request.action != AuthorizationAction::Write
            || request.resource.community_id != self.fields.community_id
            || request.resource.kind != AuthorizationResourceKind::Channel
            || request.resource.resource_id != self.fields.channel_id
            || request.resource.channel_id != Some(self.fields.channel_id)
        {
            return Err(ChannelMetadataError::AuthorizationShape);
        }
        match authorize(request) {
            AuthorizationDecision::Allowed => Ok(()),
            AuthorizationDecision::Denied(denial) => {
                Err(ChannelMetadataError::Unauthorized(denial))
            }
        }
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), ChannelMetadataError> {
        if self.fields.version != expected {
            return Err(ChannelMetadataError::StaleVersion);
        }
        Ok(())
    }

    fn prepare_update(
        &self,
        updated_at_millis: u64,
    ) -> Result<AggregateVersion, ChannelMetadataError> {
        if updated_at_millis < self.fields.updated_at_millis {
            return Err(ChannelMetadataError::InvalidTimestamp);
        }
        self.fields
            .version
            .next()
            .ok_or(ChannelMetadataError::VersionExhausted)
    }

    fn finish_update(
        &mut self,
        next_version: AggregateVersion,
        updated_at_millis: u64,
        request: &AuthorizationRequest<'_>,
    ) {
        self.fields.version = next_version;
        self.fields.updated_by_principal_id = authorization_subject(request);
        self.fields.updated_at_millis = updated_at_millis;
    }
}

fn validate_reference(reference: &ChannelTemplateReference) -> Result<(), ChannelMetadataError> {
    for value in [
        Some(reference.id.as_str()),
        reference.runtime.as_deref(),
        reference.model.as_deref(),
        reference.role.as_deref(),
        match &reference.backend {
            Some(ChannelTemplateBackend::Provider(id)) => Some(id.as_str()),
            Some(ChannelTemplateBackend::Local) | None => None,
        },
    ]
    .into_iter()
    .flatten()
    {
        if value.is_empty()
            || value.len() > MAX_REFERENCE_BYTES
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(ChannelMetadataError::InvalidReference);
        }
    }
    Ok(())
}

fn validate_placeholders(value: &str) -> Result<(), ChannelMetadataError> {
    let mut remainder = value;
    while let Some(start) = remainder.find('{') {
        if remainder[..start].contains('}') {
            return Err(ChannelMetadataError::InvalidPlaceholder);
        }
        let after_start = &remainder[start..];
        let Some(end) = after_start.find('}') else {
            return Err(ChannelMetadataError::InvalidPlaceholder);
        };
        let placeholder = &after_start[..=end];
        if !matches!(placeholder, "{channel.name}" | "{template.name}") {
            return Err(ChannelMetadataError::InvalidPlaceholder);
        }
        remainder = &after_start[end + 1..];
    }
    if remainder.contains('}') {
        return Err(ChannelMetadataError::InvalidPlaceholder);
    }
    Ok(())
}

fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelMetadataError {
    ContentTooLarge,
    InvalidTemplateType,
    TooManyReferences,
    InvalidReference,
    DuplicateReference,
    InvalidPlaceholder,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    StaleVersion,
    InvalidTimestamp,
    VersionExhausted,
}

impl fmt::Display for ChannelMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContentTooLarge => formatter.write_str("channel metadata is too large"),
            Self::InvalidTemplateType
            | Self::TooManyReferences
            | Self::InvalidReference
            | Self::DuplicateReference
            | Self::InvalidPlaceholder => formatter.write_str("channel template is invalid"),
            Self::AuthorizationShape | Self::Unauthorized(_) => {
                formatter.write_str("channel metadata write is not authorized")
            }
            Self::StaleVersion => formatter.write_str("channel metadata version is stale"),
            Self::InvalidTimestamp => formatter.write_str("channel metadata timestamp is invalid"),
            Self::VersionExhausted => formatter.write_str("channel metadata version is exhausted"),
        }
    }
}

impl Error for ChannelMetadataError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, ChannelMembership,
        CommunityMembership, MembershipRole, MembershipStatus, PrincipalScopes, ServiceAccountId,
        TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn channel() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(2))
    }

    fn principal() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(3))
    }

    fn template(
        channel_type: ChannelType,
        references: Vec<ChannelTemplateReference>,
        canvas: &str,
    ) -> Result<ChannelTemplate, ChannelMetadataError> {
        ChannelTemplate::new(
            AggregateId::from_uuid(Uuid::from_u128(4)),
            ChannelName::new("Review").expect("name"),
            None,
            channel_type,
            ChannelVisibility::Private,
            Some(ChannelMetadataText::new(canvas).expect("canvas")),
            references,
            false,
            AggregateVersion::FIRST,
        )
    }

    fn reference(id: &str) -> ChannelTemplateReference {
        ChannelTemplateReference {
            kind: ChannelTemplateReferenceKind::Persona,
            id: id.to_owned(),
            runtime: Some("acp".to_owned()),
            model: None,
            role: Some("bot".to_owned()),
            backend: Some(ChannelTemplateBackend::Local),
        }
    }

    fn authorization() -> (TenantContext, AuthenticatedPrincipal, AuthorizationScope) {
        let tenant = TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community(), "metadata-test").expect("route")),
            &[],
        )
        .expect("tenant");
        let scope = AuthorizationScope::new("channels:write").expect("scope");
        let principal = AuthenticatedPrincipal::zed_account(
            principal(),
            community(),
            ServiceAccountId::new(1),
            PrincipalScopes::new([scope.clone()]).expect("scopes"),
        );
        (tenant, principal, scope)
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: community(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: channel(),
                owner_principal_id: None,
                channel_id: Some(channel()),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community(),
                principal_id: self::principal(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community(),
                channel_id: channel(),
                principal_id: self::principal(),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }

    fn metadata() -> ChannelMetadata {
        ChannelMetadata::from_record(ChannelMetadataRecordFields {
            community_id: community(),
            channel_id: channel(),
            topic: None,
            canvas: None,
            version: AggregateVersion::FIRST,
            updated_by_principal_id: principal(),
            updated_at_millis: 1,
        })
    }

    #[test]
    fn template_validation_preserves_references_and_known_placeholders() {
        let valid = template(
            ChannelType::Forum,
            vec![reference("reviewer")],
            "# {template.name}\nChannel: {channel.name}",
        )
        .expect("template");
        assert_eq!(
            valid.render_canvas(&ChannelName::new("patches").expect("channel")),
            Some("# Review\nChannel: patches".to_owned())
        );
        assert_eq!(
            template(ChannelType::DirectMessage, Vec::new(), ""),
            Err(ChannelMetadataError::InvalidTemplateType)
        );
        assert_eq!(
            template(
                ChannelType::Stream,
                vec![reference("same"), reference("same")],
                "",
            ),
            Err(ChannelMetadataError::DuplicateReference)
        );
        assert_eq!(
            template(ChannelType::Stream, Vec::new(), "{unknown}"),
            Err(ChannelMetadataError::InvalidPlaceholder)
        );
        assert_eq!(
            template(ChannelType::Stream, Vec::new(), "} before {channel.name}",),
            Err(ChannelMetadataError::InvalidPlaceholder)
        );
    }

    #[test]
    fn topic_and_canvas_writes_enforce_version_and_channel_policy() {
        let (tenant, principal, scope) = authorization();
        let member = request(&tenant, &principal, &scope, MembershipRole::Member);
        let mut metadata = metadata();
        metadata
            .set_topic(
                AggregateVersion::FIRST,
                Some(ChannelMetadataText::new("Ship it").expect("topic")),
                2,
                &member,
            )
            .expect("topic write");
        let before = metadata.clone();
        assert_eq!(
            metadata.set_canvas(
                AggregateVersion::FIRST,
                Some(ChannelMetadataText::new("canvas").expect("canvas")),
                3,
                &member,
            ),
            Err(ChannelMetadataError::StaleVersion)
        );
        assert_eq!(metadata, before);

        let version = AggregateVersion::FIRST.next().expect("second");
        assert_eq!(
            metadata.set_canvas(
                version,
                Some(ChannelMetadataText::new("canvas").expect("canvas")),
                0,
                &member,
            ),
            Err(ChannelMetadataError::InvalidTimestamp)
        );
        assert_eq!(metadata, before);

        let guest = request(&tenant, &principal, &scope, MembershipRole::Guest);
        assert_eq!(
            metadata.set_canvas(
                version,
                Some(ChannelMetadataText::new("canvas").expect("canvas")),
                3,
                &guest,
            ),
            Err(ChannelMetadataError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );
    }
}
