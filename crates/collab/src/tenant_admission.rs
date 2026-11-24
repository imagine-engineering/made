use std::future::Future;

use collaboration_domain::{
    AuthenticatedPrincipal, AuthorizationDecision, AuthorizationRequest, TenantContext,
    TrustedTenantRoute, UntrustedTenantClaim, authorize,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RpcAdmissionError {
    Denied,
}

pub fn bind_rpc_tenant(
    trusted_route: Option<TrustedTenantRoute>,
    untrusted_claims: &[UntrustedTenantClaim],
) -> Result<TenantContext, RpcAdmissionError> {
    TenantContext::establish(trusted_route, untrusted_claims).map_err(|_| RpcAdmissionError::Denied)
}

pub struct AuthorizedRpcRequest {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
}

impl AuthorizedRpcRequest {
    pub fn authorize(request: &AuthorizationRequest<'_>) -> Result<Self, RpcAdmissionError> {
        if authorize(request) != AuthorizationDecision::Allowed {
            return Err(RpcAdmissionError::Denied);
        }
        Ok(Self {
            tenant: request.tenant.clone(),
            principal: request.principal.clone(),
        })
    }

    pub const fn tenant(&self) -> &TenantContext {
        &self.tenant
    }

    pub const fn principal(&self) -> &AuthenticatedPrincipal {
        &self.principal
    }

    pub async fn run<T, E, F, Fut>(self, operation: F) -> Result<T, E>
    where
        F: FnOnce(TenantContext, AuthenticatedPrincipal) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        operation(self.tenant, self.principal).await
    }
}
