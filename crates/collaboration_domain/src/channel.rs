use std::{error::Error, fmt, num::NonZeroU32};

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, PrincipalId, authorize,
};

const MAX_CHANNEL_NAME_BYTES: usize = 255;
const MAX_CHANNEL_DESCRIPTION_BYTES: usize = 65_536;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ChannelName(String);

impl ChannelName {
    pub fn new(value: impl Into<String>) -> Result<Self, ChannelError> {
        let value = value.into();
        let value = value
            .trim_start_matches(|character: char| character == '#' || character.is_whitespace())
            .trim_end()
            .to_owned();
        if value.is_empty()
            || value.len() > MAX_CHANNEL_NAME_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(ChannelError::InvalidName);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ChannelName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ChannelDescription(String);

impl ChannelDescription {
    pub fn new(value: impl Into<String>) -> Result<Self, ChannelError> {
        let value = value.into();
        if value.len() > MAX_CHANNEL_DESCRIPTION_BYTES {
            return Err(ChannelError::InvalidDescription);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ChannelDescription {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    Stream,
    Forum,
    DirectMessage,
    Workflow,
    Ephemeral,
    Huddle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelVisibility {
    Open,
    Private,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelLifecycleState {
    Active,
    Archived,
    Deleted,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelExpiration {
    pub ttl_seconds: NonZeroU32,
    pub expires_at_millis: u64,
}

impl ChannelExpiration {
    pub fn starting_at(ttl_seconds: NonZeroU32, now_millis: u64) -> Result<Self, ChannelError> {
        let ttl_millis = u64::from(ttl_seconds.get())
            .checked_mul(1_000)
            .ok_or(ChannelError::InvalidExpiration)?;
        let expires_at_millis = now_millis
            .checked_add(ttl_millis)
            .ok_or(ChannelError::InvalidExpiration)?;
        Ok(Self {
            ttl_seconds,
            expires_at_millis,
        })
    }

    fn renewed_at(self, now_millis: u64) -> Result<Self, ChannelError> {
        Self::starting_at(self.ttl_seconds, now_millis)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub name: ChannelName,
    pub channel_type: ChannelType,
    pub visibility: ChannelVisibility,
    pub lifecycle_state: ChannelLifecycleState,
    pub description: Option<ChannelDescription>,
    pub creator_principal_id: PrincipalId,
    pub expiration: Option<ChannelExpiration>,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelCreateFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub name: ChannelName,
    pub channel_type: ChannelType,
    pub visibility: ChannelVisibility,
    pub description: Option<ChannelDescription>,
    pub creator_principal_id: PrincipalId,
    pub ttl_seconds: Option<NonZeroU32>,
    pub now_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Channel {
    fields: ChannelRecordFields,
}

impl Channel {
    pub fn from_record(fields: ChannelRecordFields) -> Result<Self, ChannelError> {
        validate_shape(
            fields.channel_id,
            fields.channel_type,
            fields.visibility,
            fields.expiration,
        )?;
        Ok(Self { fields })
    }

    pub fn create(
        fields: ChannelCreateFields,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ChannelError> {
        authorize_create(authorization, fields.community_id)?;
        if authorization_subject(authorization) != fields.creator_principal_id {
            return Err(ChannelError::CreatorMismatch);
        }
        let expiration = fields
            .ttl_seconds
            .map(|ttl| ChannelExpiration::starting_at(ttl, fields.now_millis))
            .transpose()?;
        validate_shape(
            fields.channel_id,
            fields.channel_type,
            fields.visibility,
            expiration,
        )?;
        Ok(Self {
            fields: ChannelRecordFields {
                community_id: fields.community_id,
                channel_id: fields.channel_id,
                name: fields.name,
                channel_type: fields.channel_type,
                visibility: fields.visibility,
                lifecycle_state: ChannelLifecycleState::Active,
                description: fields.description,
                creator_principal_id: fields.creator_principal_id,
                expiration,
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn fields(&self) -> &ChannelRecordFields {
        &self.fields
    }

    pub fn archive(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelCommandOutcome, ChannelError> {
        self.authorize_lifecycle(authorization, AuthorizationAction::Manage)?;
        self.require_version(expected_version)?;
        match self.fields.lifecycle_state {
            ChannelLifecycleState::Active => {
                self.fields.lifecycle_state = ChannelLifecycleState::Archived;
                self.advance_version()?;
                Ok(ChannelCommandOutcome::Applied)
            }
            ChannelLifecycleState::Archived => Ok(ChannelCommandOutcome::Unchanged),
            ChannelLifecycleState::Deleted | ChannelLifecycleState::Expired => {
                Err(ChannelError::InvalidTransition)
            }
        }
    }

    pub fn restore(
        &mut self,
        expected_version: AggregateVersion,
        now_millis: u64,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelCommandOutcome, ChannelError> {
        self.authorize_lifecycle(authorization, AuthorizationAction::Manage)?;
        self.require_version(expected_version)?;
        match self.fields.lifecycle_state {
            ChannelLifecycleState::Archived | ChannelLifecycleState::Expired => {
                if let Some(expiration) = self.fields.expiration {
                    self.fields.expiration = Some(expiration.renewed_at(now_millis)?);
                }
                self.fields.lifecycle_state = ChannelLifecycleState::Active;
                self.advance_version()?;
                Ok(ChannelCommandOutcome::Applied)
            }
            ChannelLifecycleState::Active => Ok(ChannelCommandOutcome::Unchanged),
            ChannelLifecycleState::Deleted => Err(ChannelError::InvalidTransition),
        }
    }

    pub fn delete(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelCommandOutcome, ChannelError> {
        self.authorize_lifecycle(authorization, AuthorizationAction::Delete)?;
        self.require_version(expected_version)?;
        if self.fields.lifecycle_state == ChannelLifecycleState::Deleted {
            return Ok(ChannelCommandOutcome::Unchanged);
        }
        self.fields.lifecycle_state = ChannelLifecycleState::Deleted;
        self.advance_version()?;
        Ok(ChannelCommandOutcome::Applied)
    }

    pub fn expire_if_due(
        &mut self,
        expected_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<ChannelCommandOutcome, ChannelError> {
        self.require_version(expected_version)?;
        if self.fields.lifecycle_state != ChannelLifecycleState::Active {
            return Err(ChannelError::InvalidTransition);
        }
        let expiration = self.fields.expiration.ok_or(ChannelError::NotExpirable)?;
        if now_millis < expiration.expires_at_millis {
            return Ok(ChannelCommandOutcome::Unchanged);
        }
        self.fields.lifecycle_state = ChannelLifecycleState::Expired;
        self.advance_version()?;
        Ok(ChannelCommandOutcome::Applied)
    }

    pub fn record_activity(
        &mut self,
        expected_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<ChannelCommandOutcome, ChannelError> {
        self.require_version(expected_version)?;
        if self.fields.lifecycle_state != ChannelLifecycleState::Active {
            return Err(ChannelError::InvalidTransition);
        }
        let Some(expiration) = self.fields.expiration else {
            return Ok(ChannelCommandOutcome::Unchanged);
        };
        let renewed = expiration.renewed_at(now_millis)?;
        if renewed == expiration {
            return Ok(ChannelCommandOutcome::Unchanged);
        }
        self.fields.expiration = Some(renewed);
        self.advance_version()?;
        Ok(ChannelCommandOutcome::Applied)
    }

    fn authorize_lifecycle(
        &self,
        request: &AuthorizationRequest<'_>,
        action: AuthorizationAction,
    ) -> Result<(), ChannelError> {
        if request.action != action
            || request.resource.community_id != self.fields.community_id
            || request.resource.kind != AuthorizationResourceKind::Channel
            || request.resource.resource_id != self.fields.channel_id
            || request.resource.channel_id != Some(self.fields.channel_id)
        {
            return Err(ChannelError::AuthorizationShape);
        }
        match authorize(request) {
            AuthorizationDecision::Allowed => Ok(()),
            AuthorizationDecision::Denied(denial) => Err(ChannelError::Unauthorized(denial)),
        }
    }

    fn require_version(&self, expected_version: AggregateVersion) -> Result<(), ChannelError> {
        if self.fields.version != expected_version {
            return Err(ChannelError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn advance_version(&mut self) -> Result<(), ChannelError> {
        self.fields.version = self
            .fields
            .version
            .next()
            .ok_or(ChannelError::VersionExhausted)?;
        Ok(())
    }
}

fn authorize_create(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
) -> Result<(), ChannelError> {
    if request.action != AuthorizationAction::Manage
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Community
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.channel_id.is_some()
    {
        return Err(ChannelError::AuthorizationShape);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(ChannelError::Unauthorized(denial)),
    }
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

fn validate_shape(
    channel_id: AggregateId,
    channel_type: ChannelType,
    visibility: ChannelVisibility,
    expiration: Option<ChannelExpiration>,
) -> Result<(), ChannelError> {
    if channel_id.as_uuid().is_nil() {
        return Err(ChannelError::InvalidChannelId);
    }
    if matches!(
        channel_type,
        ChannelType::DirectMessage | ChannelType::Workflow | ChannelType::Huddle
    ) && visibility != ChannelVisibility::Private
    {
        return Err(ChannelError::InvalidVisibility);
    }
    if channel_type == ChannelType::Ephemeral && expiration.is_none() {
        return Err(ChannelError::InvalidExpiration);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelError {
    InvalidName,
    InvalidDescription,
    InvalidChannelId,
    InvalidVisibility,
    InvalidExpiration,
    CreatorMismatch,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTransition,
    NotExpirable,
    VersionExhausted,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => formatter.write_str("channel name is invalid"),
            Self::InvalidDescription => formatter.write_str("channel description is invalid"),
            Self::InvalidChannelId => formatter.write_str("channel identifier is invalid"),
            Self::InvalidVisibility => formatter.write_str("channel visibility is invalid"),
            Self::InvalidExpiration => formatter.write_str("channel expiration is invalid"),
            Self::CreatorMismatch | Self::AuthorizationShape | Self::Unauthorized(_) => {
                formatter.write_str("channel command is not authorized")
            }
            Self::StaleVersion { .. } => formatter.write_str("channel version is stale"),
            Self::InvalidTransition => formatter.write_str("channel transition is invalid"),
            Self::NotExpirable => formatter.write_str("channel has no expiration"),
            Self::VersionExhausted => formatter.write_str("channel version is exhausted"),
        }
    }
}

impl Error for ChannelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, ChannelMembership,
        CommunityMembership, MembershipRole, MembershipStatus, PrincipalScopes, ServiceAccountId,
        TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn channel_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(2))
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community_id(), "channel-test").expect("route")),
            &[],
        )
        .expect("tenant")
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("channels:manage").expect("scope")
    }

    fn principal() -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id(),
            community_id(),
            ServiceAccountId::new(1),
            PrincipalScopes::new([scope()]).expect("scopes"),
        )
    }

    fn community_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal_id(),
                role: MembershipRole::Owner,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn channel_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        id: AggregateId,
        action: AuthorizationAction,
    ) -> AuthorizationRequest<'a> {
        let mut request = community_request(tenant, principal, required_scope);
        request.action = action;
        request.resource = AuthorizationResource {
            community_id: community_id(),
            kind: AuthorizationResourceKind::Channel,
            resource_id: id,
            owner_principal_id: None,
            channel_id: Some(id),
        };
        request.current_channel_membership_version = Some(AggregateVersion::FIRST);
        request.channel_membership = Some(ChannelMembership {
            community_id: community_id(),
            channel_id: id,
            principal_id: principal_id(),
            role: MembershipRole::Owner,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        });
        request
    }

    fn create_fields(
        id: AggregateId,
        channel_type: ChannelType,
        visibility: ChannelVisibility,
        ttl_seconds: Option<NonZeroU32>,
    ) -> ChannelCreateFields {
        ChannelCreateFields {
            community_id: community_id(),
            channel_id: id,
            name: ChannelName::new("# builders").expect("name"),
            channel_type,
            visibility,
            description: None,
            creator_principal_id: principal_id(),
            ttl_seconds,
            now_millis: 1_000,
        }
    }

    #[test]
    fn channel_types_enforce_visibility_and_expiration_shape() {
        let tenant = tenant();
        let principal = principal();
        let required_scope = scope();
        let authorization = community_request(&tenant, &principal, &required_scope);
        let cases = [
            (ChannelType::Stream, ChannelVisibility::Open, None),
            (ChannelType::Forum, ChannelVisibility::Open, None),
            (ChannelType::DirectMessage, ChannelVisibility::Private, None),
            (ChannelType::Workflow, ChannelVisibility::Private, None),
            (
                ChannelType::Ephemeral,
                ChannelVisibility::Private,
                NonZeroU32::new(60),
            ),
            (ChannelType::Huddle, ChannelVisibility::Private, None),
        ];
        for (index, (channel_type, visibility, ttl)) in cases.into_iter().enumerate() {
            let id = channel_id(u128::try_from(index + 1).expect("id"));
            assert!(
                Channel::create(
                    create_fields(id, channel_type, visibility, ttl),
                    &authorization
                )
                .is_ok()
            );
        }
        assert_eq!(
            Channel::create(
                create_fields(
                    channel_id(20),
                    ChannelType::DirectMessage,
                    ChannelVisibility::Open,
                    None,
                ),
                &authorization,
            ),
            Err(ChannelError::InvalidVisibility)
        );
        assert_eq!(
            Channel::create(
                create_fields(
                    channel_id(21),
                    ChannelType::Ephemeral,
                    ChannelVisibility::Private,
                    None,
                ),
                &authorization,
            ),
            Err(ChannelError::InvalidExpiration)
        );
    }

    #[test]
    fn every_channel_type_archives_and_restores() {
        let tenant = tenant();
        let principal = principal();
        let required_scope = scope();
        let create_authorization = community_request(&tenant, &principal, &required_scope);
        for (index, channel_type) in [
            ChannelType::Stream,
            ChannelType::Forum,
            ChannelType::DirectMessage,
            ChannelType::Workflow,
            ChannelType::Ephemeral,
            ChannelType::Huddle,
        ]
        .into_iter()
        .enumerate()
        {
            let id = channel_id(u128::try_from(index + 30).expect("id"));
            let visibility = if matches!(channel_type, ChannelType::Stream | ChannelType::Forum) {
                ChannelVisibility::Open
            } else {
                ChannelVisibility::Private
            };
            let ttl =
                (channel_type == ChannelType::Ephemeral).then(|| NonZeroU32::new(60).expect("ttl"));
            let mut channel = Channel::create(
                create_fields(id, channel_type, visibility, ttl),
                &create_authorization,
            )
            .expect("channel");
            let lifecycle = channel_request(
                &tenant,
                &principal,
                &required_scope,
                id,
                AuthorizationAction::Manage,
            );
            channel
                .archive(AggregateVersion::FIRST, &lifecycle)
                .expect("archive");
            let second = AggregateVersion::FIRST.next().expect("second");
            channel.restore(second, 2_000, &lifecycle).expect("restore");
            assert_eq!(
                channel.fields().lifecycle_state,
                ChannelLifecycleState::Active
            );
        }
    }

    #[test]
    fn ephemeral_expiry_and_activity_are_versioned_and_recoverable() {
        let tenant = tenant();
        let principal = principal();
        let required_scope = scope();
        let create_authorization = community_request(&tenant, &principal, &required_scope);
        let id = channel_id(50);
        let mut channel = Channel::create(
            create_fields(
                id,
                ChannelType::Ephemeral,
                ChannelVisibility::Private,
                NonZeroU32::new(10),
            ),
            &create_authorization,
        )
        .expect("ephemeral channel");
        assert_eq!(
            channel
                .expire_if_due(AggregateVersion::FIRST, 10_999)
                .expect("not due"),
            ChannelCommandOutcome::Unchanged
        );
        channel
            .record_activity(AggregateVersion::FIRST, 2_000)
            .expect("activity");
        let second = AggregateVersion::FIRST.next().expect("second");
        channel.expire_if_due(second, 12_000).expect("expire");
        assert_eq!(
            channel.fields().lifecycle_state,
            ChannelLifecycleState::Expired
        );
        let third = second.next().expect("third");
        let lifecycle = channel_request(
            &tenant,
            &principal,
            &required_scope,
            id,
            AuthorizationAction::Manage,
        );
        channel.restore(third, 20_000, &lifecycle).expect("restore");
        assert!(
            channel
                .fields()
                .expiration
                .expect("expiration")
                .expires_at_millis
                > 20_000
        );
    }
}
