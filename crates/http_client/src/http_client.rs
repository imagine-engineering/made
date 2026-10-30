mod async_body;
#[cfg(not(target_family = "wasm"))]
pub mod github;
#[cfg(all(not(target_family = "wasm"), feature = "github-download"))]
pub mod github_download;
#[cfg(not(target_family = "wasm"))]
pub mod github_security;

pub use anyhow::{Result, anyhow};
pub use async_body::{AsyncBody, Inner, Json};
use derive_more::Deref;
pub use http::{self, Method, Request, Response, StatusCode, Uri, request::Builder};
use http::{HeaderName, HeaderValue};

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::Serialize;
#[cfg(feature = "test-support")]
use std::{any::type_name, fmt};
use std::{sync::Arc, time::Duration};
pub use url::{Host, Url};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum RedirectPolicy {
    #[default]
    NoFollow,
    FollowLimit(u32),
    FollowAll,
}
pub struct FollowRedirects(pub bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestTimeout(pub Duration);

pub trait HttpRequestExt {
    /// Conditionally modify self with the given closure.
    fn when(self, condition: bool, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        if condition { then(self) } else { self }
    }

    /// Conditionally unwrap and modify self with the given closure, if the given option is Some.
    fn when_some<T>(self, option: Option<T>, then: impl FnOnce(Self, T) -> Self) -> Self
    where
        Self: Sized,
    {
        match option {
            Some(value) => then(self, value),
            None => self,
        }
    }

    /// Whether or not to follow redirects
    fn follow_redirects(self, follow: RedirectPolicy) -> Self;

    /// Sets a deadline for the complete HTTP request, including its response body.
    fn timeout(self, timeout: Duration) -> Self;
}

impl HttpRequestExt for http::request::Builder {
    fn follow_redirects(self, follow: RedirectPolicy) -> Self {
        self.extension(follow)
    }

    fn timeout(self, timeout: Duration) -> Self {
        debug_assert!(!timeout.is_zero(), "timeout must be positive");
        self.extension(RequestTimeout(timeout))
    }
}

/// A set of pre-validated user-supplied HTTP headers.
///
/// Construction (and the per-name validation that goes with it) happens once
/// at settings load time. Cloning is `Arc`-cheap, so providers can hand a copy
/// to each outgoing request without re-parsing or re-allocating.
#[derive(Default, Clone, Debug)]
pub struct CustomHeaders(Arc<[(HeaderName, HeaderValue)]>);

impl CustomHeaders {
    pub fn new(headers: Vec<(HeaderName, HeaderValue)>) -> Self {
        Self(headers.into())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&HeaderName, &HeaderValue)> {
        self.0.iter().map(|(n, v)| (n, v))
    }
}

impl PartialEq for CustomHeaders {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|(a, b)| a.0 == b.0 && a.1 == b.1)
    }
}

pub trait RequestBuilderExt {
    /// Append every header in `headers` to the request being built.
    fn extra_headers(self, headers: &CustomHeaders) -> Self;
}

impl RequestBuilderExt for http::request::Builder {
    fn extra_headers(mut self, headers: &CustomHeaders) -> Self {
        if headers.is_empty() {
            return self;
        }
        if let Some(map) = self.headers_mut() {
            for (name, value) in headers.iter() {
                map.append(name.clone(), value.clone());
            }
        }
        self
    }
}

pub trait HttpClient: 'static + Send + Sync {
    fn user_agent(&self) -> Option<&HeaderValue>;

    fn proxy(&self) -> Option<&Url>;

    fn send(
        &self,
        req: http::Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>>;

    /// Connect only to these caller-vetted addresses, retaining the URL's TLS
    /// identity and refusing redirects. Unsupported transports fail closed.
    fn send_with_pinned_addresses(
        &self,
        _request: Request<AsyncBody>,
        _addresses: Vec<std::net::SocketAddr>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        Box::pin(async {
            anyhow::bail!("This HTTP transport does not support pinned destinations")
        })
    }

    fn with_endpoint_scope(&self, _endpoint: Url) -> Result<Arc<dyn HttpClient>> {
        anyhow::bail!("This HTTP transport cannot enforce a reviewed provider endpoint")
    }

    fn get(
        &self,
        uri: &str,
        body: AsyncBody,
        follow_redirects: bool,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        let request = Builder::new()
            .uri(uri)
            .follow_redirects(if follow_redirects {
                RedirectPolicy::FollowAll
            } else {
                RedirectPolicy::NoFollow
            })
            .body(body);

        match request {
            Ok(request) => self.send(request),
            Err(e) => Box::pin(async move { Err(e.into()) }),
        }
    }

    fn post_json(
        &self,
        uri: &str,
        body: AsyncBody,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        let request = Builder::new()
            .uri(uri)
            .method(Method::POST)
            .header("Content-Type", "application/json")
            .body(body);

        match request {
            Ok(request) => self.send(request),
            Err(e) => Box::pin(async move { Err(e.into()) }),
        }
    }

    #[cfg(feature = "test-support")]
    fn as_fake(&self) -> &FakeHttpClient {
        panic!("called as_fake on {}", type_name::<Self>())
    }
}

pub struct ScopedEndpointClient {
    client: Arc<dyn HttpClient>,
    endpoint: Url,
}

impl ScopedEndpointClient {
    pub fn parse_endpoint(value: &str) -> Result<Url> {
        let endpoint = Url::parse(value)?;
        anyhow::ensure!(
            endpoint.scheme() == "https"
                && endpoint.host_str().is_some()
                && endpoint.username().is_empty()
                && endpoint.password().is_none()
                && endpoint.query().is_none()
                && endpoint.fragment().is_none()
                && !endpoint.path().contains('%'),
            "A scoped provider endpoint requires HTTPS without embedded credentials, query or fragment"
        );
        Ok(endpoint)
    }

    pub fn new(client: Arc<dyn HttpClient>, endpoint: Url) -> Result<Self> {
        Self::parse_endpoint(endpoint.as_str())?;
        anyhow::ensure!(
            client.proxy().is_none(),
            "A configured proxy is outside the reviewed provider-only network scope"
        );
        Ok(Self { client, endpoint })
    }

    fn validate_request(&self, request: &Request<AsyncBody>) -> Result<()> {
        anyhow::ensure!(
            self.client.proxy().is_none(),
            "The provider proxy changed after review"
        );
        let url = Url::parse(&request.uri().to_string())?;
        anyhow::ensure!(
            !url.path().contains('%'),
            "Encoded provider request paths are outside the reviewed endpoint scope"
        );
        let base_path = self.endpoint.path().trim_end_matches('/');
        let path_is_scoped = url.path() == base_path
            || url
                .path()
                .strip_prefix(base_path)
                .is_some_and(|suffix| suffix.starts_with('/'));
        anyhow::ensure!(
            url.origin() == self.endpoint.origin()
                && path_is_scoped
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none(),
            "The provider request is outside the reviewed endpoint"
        );
        if let Some(host) = request.headers().get(http::header::HOST) {
            let host = Url::parse(&format!("https://{}/", host.to_str()?))?;
            anyhow::ensure!(
                host.origin() == self.endpoint.origin()
                    && host.username().is_empty()
                    && host.password().is_none()
                    && host.path() == "/"
                    && host.query().is_none()
                    && host.fragment().is_none(),
                "The provider Host header differs from the reviewed endpoint"
            );
        }
        Ok(())
    }
}

impl HttpClient for ScopedEndpointClient {
    fn user_agent(&self) -> Option<&HeaderValue> {
        self.client.user_agent()
    }
    fn proxy(&self) -> Option<&Url> {
        None
    }

    fn send(
        &self,
        mut request: Request<AsyncBody>,
    ) -> BoxFuture<'static, Result<Response<AsyncBody>>> {
        if let Err(error) = self.validate_request(&request) {
            return Box::pin(async move { Err(error) });
        }
        request.extensions_mut().insert(RedirectPolicy::NoFollow);
        let response = self.client.send(request);
        Box::pin(async move {
            let response = response.await?;
            anyhow::ensure!(
                !response.status().is_redirection(),
                "Provider redirects are outside the reviewed network scope"
            );
            Ok(response)
        })
    }
}

/// An [`HttpClient`] that may have a proxy.
#[derive(Deref)]
pub struct HttpClientWithProxy {
    #[deref]
    client: Arc<dyn HttpClient>,
    proxy: Option<Url>,
}

impl HttpClientWithProxy {
    /// Returns a new [`HttpClientWithProxy`] with the given proxy URL.
    pub fn new(client: Arc<dyn HttpClient>, proxy_url: Option<String>) -> Self {
        let proxy_url = proxy_url
            .and_then(|proxy| proxy.parse().ok())
            .or_else(read_proxy_from_env);

        Self::new_url(client, proxy_url)
    }
    pub fn new_url(client: Arc<dyn HttpClient>, proxy_url: Option<Url>) -> Self {
        Self {
            client,
            proxy: proxy_url,
        }
    }
}

impl HttpClient for HttpClientWithProxy {
    fn with_endpoint_scope(&self, endpoint: Url) -> Result<Arc<dyn HttpClient>> {
        anyhow::ensure!(
            self.proxy.is_none(),
            "A configured proxy is outside the reviewed provider-only network scope"
        );
        self.client.with_endpoint_scope(endpoint)
    }

    fn send_with_pinned_addresses(
        &self,
        request: Request<AsyncBody>,
        addresses: Vec<std::net::SocketAddr>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        if self.proxy.is_some() {
            return Box::pin(async {
                anyhow::bail!("Pinned imports are unavailable through the configured proxy")
            });
        }
        self.client.send_with_pinned_addresses(request, addresses)
    }
    fn send(
        &self,
        req: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        self.client.send(req)
    }

    fn user_agent(&self) -> Option<&HeaderValue> {
        self.client.user_agent()
    }

    fn proxy(&self) -> Option<&Url> {
        self.proxy.as_ref()
    }

    #[cfg(feature = "test-support")]
    fn as_fake(&self) -> &FakeHttpClient {
        self.client.as_fake()
    }
}

/// An [`HttpClient`] that has a base URL.
#[derive(Deref)]
pub struct HttpClientWithUrl {
    base_url: Mutex<String>,
    #[deref]
    client: HttpClientWithProxy,
}

impl HttpClientWithUrl {
    /// Returns a new [`HttpClientWithUrl`] with the given base URL.
    pub fn new(
        client: Arc<dyn HttpClient>,
        base_url: impl Into<String>,
        proxy_url: Option<String>,
    ) -> Self {
        let client = HttpClientWithProxy::new(client, proxy_url);

        Self {
            base_url: Mutex::new(base_url.into()),
            client,
        }
    }

    pub fn new_url(
        client: Arc<dyn HttpClient>,
        base_url: impl Into<String>,
        proxy_url: Option<Url>,
    ) -> Self {
        let client = HttpClientWithProxy::new_url(client, proxy_url);

        Self {
            base_url: Mutex::new(base_url.into()),
            client,
        }
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> String {
        self.base_url.lock().clone()
    }

    /// Sets the base URL.
    pub fn set_base_url(&self, base_url: impl Into<String>) {
        let base_url = base_url.into();
        *self.base_url.lock() = base_url;
    }

    /// Builds a URL using the given path.
    pub fn build_url(&self, path: &str) -> String {
        format!("{}{}", self.base_url(), path)
    }

    /// Builds a Zed API URL using the given path.
    pub fn build_zed_api_url(&self, path: &str, query: &[(&str, &str)]) -> Result<Url> {
        let base_url = self.base_url();
        let base_api_url = match base_url.as_ref() {
            "https://zed.dev" => "https://api.zed.dev",
            "https://staging.zed.dev" => "https://api-staging.zed.dev",
            "http://localhost:3000" => "http://localhost:8080",
            other => other,
        };

        Ok(Url::parse_with_params(
            &format!("{}{}", base_api_url, path),
            query,
        )?)
    }

    /// Builds a Zed Cloud URL using the given path.
    pub fn build_zed_cloud_url(&self, path: &str) -> Result<Url> {
        let base_url = self.base_url();
        let base_api_url = match base_url.as_ref() {
            "https://zed.dev" => "https://cloud.zed.dev",
            "https://staging.zed.dev" => "https://cloud.zed.dev",
            "http://localhost:3000" => "http://localhost:8787",
            other => other,
        };

        Ok(Url::parse(&format!("{}{}", base_api_url, path))?)
    }

    /// Builds a Zed Cloud URL using the given path and query params.
    pub fn build_zed_cloud_url_with_query(&self, path: &str, query: impl Serialize) -> Result<Url> {
        let base_url = self.base_url();
        let base_api_url = match base_url.as_ref() {
            "https://zed.dev" => "https://cloud.zed.dev",
            "https://staging.zed.dev" => "https://cloud.zed.dev",
            "http://localhost:3000" => "http://localhost:8787",
            other => other,
        };
        let query = serde_urlencoded::to_string(&query)?;
        Ok(Url::parse(&format!("{}{}?{}", base_api_url, path, query))?)
    }

    /// Builds a Zed LLM URL using the given path.
    pub fn build_zed_llm_url(&self, path: &str, query: &[(&str, &str)]) -> Result<Url> {
        let base_url = self.base_url();
        let base_api_url = match base_url.as_ref() {
            "https://zed.dev" => "https://cloud.zed.dev",
            "https://staging.zed.dev" => "https://llm-staging.zed.dev",
            "http://localhost:3000" => "http://localhost:8787",
            other => other,
        };

        Ok(Url::parse_with_params(
            &format!("{}{}", base_api_url, path),
            query,
        )?)
    }
}

impl HttpClient for HttpClientWithUrl {
    fn with_endpoint_scope(&self, endpoint: Url) -> Result<Arc<dyn HttpClient>> {
        self.client.with_endpoint_scope(endpoint)
    }

    fn send_with_pinned_addresses(
        &self,
        request: Request<AsyncBody>,
        addresses: Vec<std::net::SocketAddr>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        self.client.send_with_pinned_addresses(request, addresses)
    }
    fn send(
        &self,
        req: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        self.client.send(req)
    }

    fn user_agent(&self) -> Option<&HeaderValue> {
        self.client.user_agent()
    }

    fn proxy(&self) -> Option<&Url> {
        self.client.proxy.as_ref()
    }

    #[cfg(feature = "test-support")]
    fn as_fake(&self) -> &FakeHttpClient {
        self.client.as_fake()
    }
}

pub fn read_proxy_from_env() -> Option<Url> {
    const ENV_VARS: &[&str] = &[
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ];

    ENV_VARS
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .and_then(|env| env.parse().ok())
}

pub fn read_no_proxy_from_env() -> Option<String> {
    const ENV_VARS: &[&str] = &["NO_PROXY", "no_proxy"];

    ENV_VARS.iter().find_map(|var| std::env::var(var).ok())
}

pub struct BlockedHttpClient;

impl BlockedHttpClient {
    pub fn new() -> Self {
        BlockedHttpClient
    }
}

impl HttpClient for BlockedHttpClient {
    fn send(
        &self,
        _req: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        Box::pin(async {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "BlockedHttpClient disallowed request",
            )
            .into())
        })
    }

    fn user_agent(&self) -> Option<&HeaderValue> {
        None
    }

    fn proxy(&self) -> Option<&Url> {
        None
    }

    #[cfg(feature = "test-support")]
    fn as_fake(&self) -> &FakeHttpClient {
        panic!("called as_fake on {}", type_name::<Self>())
    }
}

#[cfg(feature = "test-support")]
type FakeHttpHandler = Arc<
    dyn Fn(Request<AsyncBody>) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>>
        + Send
        + Sync
        + 'static,
>;

#[cfg(feature = "test-support")]
pub struct FakeHttpClient {
    handler: Mutex<Option<FakeHttpHandler>>,
    user_agent: HeaderValue,
}

#[cfg(feature = "test-support")]
impl FakeHttpClient {
    pub fn create<Fut, F>(handler: F) -> Arc<HttpClientWithUrl>
    where
        Fut: futures::Future<Output = anyhow::Result<Response<AsyncBody>>> + Send + 'static,
        F: Fn(Request<AsyncBody>) -> Fut + Send + Sync + 'static,
    {
        Arc::new(HttpClientWithUrl {
            base_url: Mutex::new("http://test.example".into()),
            client: HttpClientWithProxy {
                client: Arc::new(Self {
                    handler: Mutex::new(Some(Arc::new(move |req| Box::pin(handler(req))))),
                    user_agent: HeaderValue::from_static(type_name::<Self>()),
                }),
                proxy: None,
            },
        })
    }

    pub fn with_404_response() -> Arc<HttpClientWithUrl> {
        log::warn!("Using fake HTTP client with 404 response");
        Self::create(|_| async move {
            Ok(Response::builder()
                .status(404)
                .body(Default::default())
                .unwrap())
        })
    }

    pub fn with_200_response() -> Arc<HttpClientWithUrl> {
        log::warn!("Using fake HTTP client with 200 response");
        Self::create(|_| async move {
            Ok(Response::builder()
                .status(200)
                .body(Default::default())
                .unwrap())
        })
    }

    pub fn replace_handler<Fut, F>(&self, new_handler: F)
    where
        Fut: futures::Future<Output = anyhow::Result<Response<AsyncBody>>> + Send + 'static,
        F: Fn(FakeHttpHandler, Request<AsyncBody>) -> Fut + Send + Sync + 'static,
    {
        let mut handler = self.handler.lock();
        let old_handler = handler.take().unwrap();
        *handler = Some(Arc::new(move |req| {
            Box::pin(new_handler(old_handler.clone(), req))
        }));
    }
}

#[cfg(feature = "test-support")]
impl fmt::Debug for FakeHttpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeHttpClient").finish()
    }
}

#[cfg(feature = "test-support")]
impl HttpClient for FakeHttpClient {
    fn with_endpoint_scope(&self, endpoint: Url) -> Result<Arc<dyn HttpClient>> {
        let client = Arc::new(Self {
            handler: Mutex::new(self.handler.lock().clone()),
            user_agent: self.user_agent.clone(),
        });
        Ok(Arc::new(ScopedEndpointClient::new(client, endpoint)?))
    }

    fn send_with_pinned_addresses(
        &self,
        request: Request<AsyncBody>,
        _addresses: Vec<std::net::SocketAddr>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        self.send(request)
    }
    fn send(
        &self,
        req: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        ((self.handler.lock().as_ref().unwrap())(req)) as _
    }

    fn user_agent(&self) -> Option<&HeaderValue> {
        Some(&self.user_agent)
    }

    fn proxy(&self) -> Option<&Url> {
        None
    }

    fn as_fake(&self) -> &FakeHttpClient {
        self
    }
}

#[cfg(test)]
mod scoped_endpoint_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingClient {
        calls: Arc<AtomicUsize>,
        status: http::StatusCode,
    }

    impl HttpClient for RecordingClient {
        fn user_agent(&self) -> Option<&HeaderValue> {
            None
        }
        fn proxy(&self) -> Option<&Url> {
            None
        }
        fn send(
            &self,
            request: Request<AsyncBody>,
        ) -> BoxFuture<'static, Result<Response<AsyncBody>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(matches!(
                request.extensions().get::<RedirectPolicy>(),
                Some(RedirectPolicy::NoFollow)
            ));
            assert_eq!(
                request
                    .headers()
                    .get(http::header::AUTHORIZATION)
                    .expect("authorization"),
                "Bearer test-key"
            );
            let status = self.status;
            Box::pin(async move {
                Ok(Response::builder()
                    .status(status)
                    .body(AsyncBody::empty())?)
            })
        }
    }

    fn request(url: &str) -> Request<AsyncBody> {
        Request::builder()
            .uri(url)
            .header(http::header::AUTHORIZATION, "Bearer test-key")
            .follow_redirects(RedirectPolicy::FollowAll)
            .body(AsyncBody::empty())
            .expect("request")
    }

    #[test]
    fn endpoint_scope_rejects_escape_before_transport() {
        let calls = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(RecordingClient {
            calls: calls.clone(),
            status: http::StatusCode::OK,
        });
        let client = ScopedEndpointClient::new(
            transport,
            ScopedEndpointClient::parse_endpoint("https://provider.example/v1").expect("endpoint"),
        )
        .expect("scope");
        for url in [
            "https://other.example/v1/chat",
            "http://provider.example/v1/chat",
            "https://provider.example:444/v1/chat",
            "https://provider.example/v10/chat",
            "https://provider.example/v1/../secret",
            "https://provider.example/v1/%2fsecret",
        ] {
            assert!(
                futures::executor::block_on(client.send(request(url))).is_err(),
                "{url}"
            );
        }
        for host in [
            "other.example",
            "provider.example/extra",
            "provider.example?query",
        ] {
            let mut request = request("https://provider.example/v1/chat");
            request
                .headers_mut()
                .insert(http::header::HOST, host.parse().expect("header"));
            assert!(futures::executor::block_on(client.send(request)).is_err());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(
            futures::executor::block_on(client.send(request("https://provider.example/v1/chat")))
                .is_ok()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn endpoint_scope_refuses_redirects_and_unsupported_transports() {
        let calls = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(RecordingClient {
            calls: calls.clone(),
            status: http::StatusCode::FOUND,
        });
        let endpoint =
            ScopedEndpointClient::parse_endpoint("https://provider.example/v1").expect("endpoint");
        assert!(transport.with_endpoint_scope(endpoint.clone()).is_err());
        let proxied = Arc::new(HttpClientWithProxy::new_url(
            transport.clone(),
            Some(Url::parse("http://proxy.example:8080").expect("proxy URL")),
        ));
        assert!(proxied.with_endpoint_scope(endpoint.clone()).is_err());
        assert!(ScopedEndpointClient::new(proxied, endpoint.clone()).is_err());
        let client = ScopedEndpointClient::new(transport, endpoint).expect("scope");
        assert!(
            futures::executor::block_on(client.send(request("https://provider.example/v1/chat")))
                .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        for endpoint in [
            "http://provider.example",
            "https://user:secret@provider.example",
            "https://provider.example/v1?key=secret",
            "https://provider.example/#fragment",
            "https://provider.example/%2f",
        ] {
            assert!(
                ScopedEndpointClient::parse_endpoint(endpoint).is_err(),
                "{endpoint}"
            );
        }
    }
}
