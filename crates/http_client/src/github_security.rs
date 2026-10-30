use crate::{AsyncBody, HttpClient, HttpRequestExt, RedirectPolicy, Request};
use futures::AsyncReadExt as _;
use http::HeaderValue;
use serde::Deserialize;
use std::{collections::BTreeSet, fmt, sync::Arc, time::Duration};

pub const GITHUB_SECURITY_API_VERSION: &str = "2026-03-10";
pub const MAX_GITHUB_SECURITY_PAGE_BYTES: usize = 1024 * 1024;
pub const MAX_GITHUB_SECURITY_COLLECTION_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_GITHUB_SECURITY_PAGES: usize = 10;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

type CredentialReader = Arc<dyn Fn() -> Option<String> + Send + Sync>;
type SecurityResult<T> = std::result::Result<T, GithubSecurityError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GithubSecurityError {
    InvalidSelection,
    MissingCredential,
    CredentialChanged,
    AuthorityChanged,
    Cancelled,
    TimedOut,
    AccountMismatch,
    RepositoryMismatch,
    TransportUnavailable,
    AuthenticationUnavailable,
    AccessUnavailable,
    ResourceUnavailable,
    RateLimited,
    ServiceUnavailable,
    RedirectRejected,
    InvalidResponse,
    BoundExceeded,
}

impl fmt::Display for GithubSecurityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidSelection => "Select an exact GitHub.com account and repository",
            Self::MissingCredential => "The explicitly selected existing GITHUB_TOKEN is unavailable or invalid",
            Self::CredentialChanged => "Selected GitHub credentials changed or were revoked; review current authority",
            Self::AuthorityChanged => "Reviewed project/repository authority changed; evidence remains incomplete",
            Self::Cancelled => "GitHub evidence collection was cancelled; coverage is incomplete",
            Self::TimedOut => "GitHub evidence collection timed out; coverage is incomplete",
            Self::AccountMismatch => "The credential's verified account differs from the selected account",
            Self::RepositoryMismatch => "GitHub returned a different repository; no replacement authority was admitted",
            Self::TransportUnavailable => "GitHub retrieval failed; no complete evidence was observed",
            Self::AuthenticationUnavailable => "GitHub authentication is unavailable or revoked; installation-only credentials may not support account verification",
            Self::AccessUnavailable => "GitHub denied access; required read permission or service availability is unknown",
            Self::ResourceUnavailable => "GitHub did not expose this resource; private access or service availability is unknown",
            Self::RateLimited => "GitHub rate limited retrieval; evidence is incomplete",
            Self::ServiceUnavailable => "GitHub returned an API failure; evidence is incomplete",
            Self::RedirectRejected => "GitHub redirected the reviewed request; redirects are not authorized",
            Self::InvalidResponse => "GitHub returned unsupported or malformed bounded metadata",
            Self::BoundExceeded => "GitHub metadata exceeds the reviewed collection bound",
        })
    }
}

impl std::error::Error for GithubSecurityError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubSecurityRepository {
    owner: String,
    name: String,
}

impl GithubSecurityRepository {
    pub fn new(owner: &str, name: &str) -> SecurityResult<Self> {
        if !valid_account(owner)
            || name.is_empty()
            || name.len() > 100
            || matches!(name, "." | "..")
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        {
            return Err(GithubSecurityError::InvalidSelection);
        }
        Ok(Self {
            owner: owner.to_owned(),
            name: name.to_owned(),
        })
    }

    pub fn full_name(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }

    pub fn web_url(&self) -> String {
        format!("https://github.com/{}", self.full_name())
    }

    fn api_url(&self) -> String {
        format!("https://api.github.com/repos/{}", self.full_name())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubSecuritySelection {
    account: String,
    repository: GithubSecurityRepository,
}

impl GithubSecuritySelection {
    pub fn new(account: &str, repository: GithubSecurityRepository) -> SecurityResult<Self> {
        if !valid_account(account) {
            return Err(GithubSecurityError::InvalidSelection);
        }
        Ok(Self {
            account: account.to_owned(),
            repository,
        })
    }

    pub fn reviewed_evidence_requests(
        &self,
        reference: Option<&str>,
        dependabot: bool,
        code_scanning: bool,
    ) -> SecurityResult<Vec<String>> {
        if !dependabot && !code_scanning {
            return Err(GithubSecurityError::InvalidSelection);
        }
        let mut requests = vec![
            "https://api.github.com/user".to_owned(),
            self.repository.api_url(),
        ];
        if dependabot {
            requests.push(EvidenceEndpoint::Dependabot.first_url(&self.repository)?);
            requests.push(EvidenceEndpoint::PullRequests.first_url(&self.repository)?);
        }
        if code_scanning {
            let reference = reference.ok_or(GithubSecurityError::InvalidSelection)?;
            metadata_reference(reference).map_err(|_| GithubSecurityError::InvalidSelection)?;
            requests.push(
                EvidenceEndpoint::CodeAlerts(reference.to_owned()).first_url(&self.repository)?,
            );
            requests.push(
                EvidenceEndpoint::CodeAnalyses(reference.to_owned()).first_url(&self.repository)?,
            );
        }
        Ok(requests)
    }
    pub fn account(&self) -> &str {
        &self.account
    }
    pub fn repository(&self) -> &GithubSecurityRepository {
        &self.repository
    }
}

fn valid_account(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 39
        && !value.starts_with('-')
        && !value.ends_with('-')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

// This snapshot only binds a reviewed operation to the existing credential
// owner. It is intentionally neither Debug nor serializable.
#[derive(Clone)]
pub struct SelectedGithubSecurityAuthority {
    selection: GithubSecuritySelection,
    token: String,
    credential_reader: CredentialReader,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubSecurityVerifiedMetadata {
    pub account_id: u64,
    pub repository_id: u64,
    pub private_repository: bool,
    pub default_branch: Option<String>,
}

#[derive(Clone)]
pub struct VerifiedGithubSecurityAuthority {
    selected: SelectedGithubSecurityAuthority,
    metadata: GithubSecurityVerifiedMetadata,
    verification_bytes: usize,
}

#[derive(Clone, Copy)]
enum AuthorityEndpoint {
    Account,
    Repository,
}

#[derive(Deserialize)]
struct AccountMetadata {
    id: u64,
    login: String,
}
#[derive(Deserialize)]
struct RepositoryMetadata {
    id: u64,
    full_name: String,
    private: bool,
    default_branch: Option<String>,
}

impl SelectedGithubSecurityAuthority {
    #[cfg(feature = "test-support")]
    pub fn capture_fixture_credential(
        selection: GithubSecuritySelection,
        reader: Arc<dyn Fn() -> Option<String> + Send + Sync>,
    ) -> SecurityResult<Self> {
        Self::capture(selection, reader)
    }

    pub fn capture_selected_environment(
        selection: GithubSecuritySelection,
    ) -> SecurityResult<Self> {
        Self::capture(selection, Arc::new(|| std::env::var("GITHUB_TOKEN").ok()))
    }

    fn capture(
        selection: GithubSecuritySelection,
        credential_reader: CredentialReader,
    ) -> SecurityResult<Self> {
        let token = credential_reader().ok_or(GithubSecurityError::MissingCredential)?;
        if token.is_empty()
            || token.len() > 4096
            || !token.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(GithubSecurityError::MissingCredential);
        }
        Ok(Self {
            selection,
            token,
            credential_reader,
        })
    }

    pub fn selection(&self) -> &GithubSecuritySelection {
        &self.selection
    }

    pub fn revalidate_credential(&self) -> SecurityResult<()> {
        if (self.credential_reader)().as_deref() != Some(self.token.as_str()) {
            return Err(GithubSecurityError::CredentialChanged);
        }
        Ok(())
    }

    fn request(&self, endpoint: AuthorityEndpoint) -> SecurityResult<Request<AsyncBody>> {
        self.revalidate_credential()?;
        let url = match endpoint {
            AuthorityEndpoint::Account => "https://api.github.com/user".to_owned(),
            AuthorityEndpoint::Repository => self.selection.repository.api_url(),
        };
        self.request_at_reviewed_url(&url)
    }

    fn request_at_reviewed_url(&self, url: &str) -> SecurityResult<Request<AsyncBody>> {
        self.revalidate_credential()?;
        let mut authorization = HeaderValue::from_str(&format!("Bearer {}", self.token))
            .map_err(|_| GithubSecurityError::MissingCredential)?;
        authorization.set_sensitive(true);
        Request::get(url)
            .header("Accept", "application/vnd.github+json")
            .header("Accept-Encoding", "identity")
            .header("Authorization", authorization)
            .header("User-Agent", "Made-Doctor")
            .header("X-GitHub-Api-Version", GITHUB_SECURITY_API_VERSION)
            .follow_redirects(RedirectPolicy::NoFollow)
            .timeout(REQUEST_TIMEOUT)
            .body(AsyncBody::empty())
            .map_err(|_| GithubSecurityError::InvalidSelection)
    }

    async fn read_authority<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: AuthorityEndpoint,
        client: &Arc<dyn HttpClient>,
    ) -> SecurityResult<MetadataPage<T>> {
        self.read_metadata(
            self.request(endpoint)?,
            MAX_GITHUB_SECURITY_PAGE_BYTES,
            client,
        )
        .await
    }

    async fn read_metadata<T: serde::de::DeserializeOwned>(
        &self,
        request: Request<AsyncBody>,
        byte_limit: usize,
        client: &Arc<dyn HttpClient>,
    ) -> SecurityResult<MetadataPage<T>> {
        if byte_limit == 0 || byte_limit > MAX_GITHUB_SECURITY_PAGE_BYTES {
            return Err(GithubSecurityError::BoundExceeded);
        }
        let mut response = client
            .send(request)
            .await
            .map_err(|_| GithubSecurityError::TransportUnavailable)?;
        self.revalidate_credential()?;
        let status = response.status().as_u16();
        if (300..400).contains(&status) {
            return Err(GithubSecurityError::RedirectRejected);
        }
        match status {
            200 => {}
            401 => return Err(GithubSecurityError::AuthenticationUnavailable),
            403 if response
                .headers()
                .get("x-ratelimit-remaining")
                .is_some_and(|value| value == "0") =>
            {
                return Err(GithubSecurityError::RateLimited);
            }
            403 => return Err(GithubSecurityError::AccessUnavailable),
            404 => return Err(GithubSecurityError::ResourceUnavailable),
            429 => return Err(GithubSecurityError::RateLimited),
            _ => return Err(GithubSecurityError::ServiceUnavailable),
        }
        let headers = response.headers();
        if headers.get_all("content-encoding").iter().count() > 1
            || headers
                .get("content-encoding")
                .is_some_and(|value| value != "identity")
            || headers.get_all("content-length").iter().count() > 1
        {
            return Err(GithubSecurityError::InvalidResponse);
        }
        let content_type = headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(str::trim);
        if !matches!(
            content_type,
            Some("application/json" | "application/vnd.github+json")
        ) {
            return Err(GithubSecurityError::InvalidResponse);
        }
        let expected_length = headers
            .get("content-length")
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .ok_or(GithubSecurityError::InvalidResponse)
            })
            .transpose()?;
        if expected_length.is_some_and(|length| length > byte_limit) {
            return Err(GithubSecurityError::BoundExceeded);
        }
        let pagination = next_link(headers);
        let mut bytes = Vec::new();
        response
            .body_mut()
            .take((byte_limit + 1) as u64)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| GithubSecurityError::TransportUnavailable)?;
        self.revalidate_credential()?;
        if bytes.len() > byte_limit {
            return Err(GithubSecurityError::BoundExceeded);
        }
        if expected_length.is_some_and(|length| bytes.len() != length) {
            return Err(GithubSecurityError::InvalidResponse);
        }
        Ok(MetadataPage {
            value: serde_json::from_slice(&bytes)
                .map_err(|_| GithubSecurityError::InvalidResponse)?,
            bytes: bytes.len(),
            pagination,
        })
    }

    pub async fn verify(
        self,
        client: Arc<dyn HttpClient>,
    ) -> SecurityResult<VerifiedGithubSecurityAuthority> {
        self.verify_with_checkpoint(client, &mut || std::future::ready(Ok(())))
            .await
    }
    pub async fn verify_with_checkpoint<Check: std::future::Future<Output = SecurityResult<()>>>(
        self,
        client: Arc<dyn HttpClient>,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> SecurityResult<VerifiedGithubSecurityAuthority> {
        checkpoint().await?;
        let account: MetadataPage<AccountMetadata> = self
            .read_authority(AuthorityEndpoint::Account, &client)
            .await?;
        if account.value.id == 0
            || !account
                .value
                .login
                .eq_ignore_ascii_case(&self.selection.account)
        {
            return Err(GithubSecurityError::AccountMismatch);
        }
        checkpoint().await?;
        let repository: MetadataPage<RepositoryMetadata> = self
            .read_authority(AuthorityEndpoint::Repository, &client)
            .await?;
        if repository.value.id == 0
            || !repository
                .value
                .full_name
                .eq_ignore_ascii_case(&self.selection.repository.full_name())
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        if repository
            .value
            .default_branch
            .as_ref()
            .is_some_and(|value| {
                value.is_empty() || value.len() > 256 || value.chars().any(char::is_control)
            })
        {
            return Err(GithubSecurityError::InvalidResponse);
        }
        checkpoint().await?;
        self.revalidate_credential()?;
        Ok(VerifiedGithubSecurityAuthority {
            verification_bytes: account.bytes + repository.bytes,
            selected: self,
            metadata: GithubSecurityVerifiedMetadata {
                account_id: account.value.id,
                repository_id: repository.value.id,
                private_repository: repository.value.private,
                default_branch: repository.value.default_branch,
            },
        })
    }
}

impl VerifiedGithubSecurityAuthority {
    pub fn selection(&self) -> &GithubSecuritySelection {
        self.selected.selection()
    }
    pub fn metadata(&self) -> &GithubSecurityVerifiedMetadata {
        &self.metadata
    }
    pub fn verification_bytes(&self) -> usize {
        self.verification_bytes
    }
    pub fn revalidate_credential(&self) -> SecurityResult<()> {
        self.selected.revalidate_credential()
    }
}

struct MetadataPage<T> {
    value: T,
    bytes: usize,
    pagination: SecurityResult<Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GithubSecurityCoverage {
    NotRequested,
    Complete,
    Partial(GithubSecurityError),
    Unavailable(GithubSecurityError),
}

#[derive(Clone, Debug)]
pub struct GithubSecurityCollection<T> {
    pub observations: Vec<T>,
    pub coverage: GithubSecurityCoverage,
    pub pages_completed: usize,
    pub bytes_collected: usize,
}

pub struct GithubSecurityBudget {
    remaining_bytes: usize,
    remaining_pages: usize,
}

impl GithubSecurityBudget {
    pub fn after_verification(authority: &VerifiedGithubSecurityAuthority) -> SecurityResult<Self> {
        Self::with_limits(
            MAX_GITHUB_SECURITY_PAGES,
            MAX_GITHUB_SECURITY_COLLECTION_BYTES.saturating_sub(authority.verification_bytes()),
        )
    }
    pub fn with_limits(pages: usize, bytes: usize) -> SecurityResult<Self> {
        if pages == 0
            || pages > MAX_GITHUB_SECURITY_PAGES
            || bytes < 2
            || bytes > MAX_GITHUB_SECURITY_COLLECTION_BYTES
        {
            return Err(GithubSecurityError::BoundExceeded);
        }
        Ok(Self {
            remaining_bytes: bytes,
            remaining_pages: pages,
        })
    }
    fn reserve_page(&mut self) -> SecurityResult<usize> {
        if self.remaining_pages == 0 || self.remaining_bytes < 2 {
            return Err(GithubSecurityError::BoundExceeded);
        }
        let allowance = MAX_GITHUB_SECURITY_PAGE_BYTES.min(self.remaining_bytes - 1);
        self.remaining_bytes -= allowance + 1;
        self.remaining_pages -= 1;
        Ok(allowance)
    }
    fn admit_bytes(&mut self, allowance: usize, observed: usize) {
        self.remaining_bytes += allowance + 1 - observed;
    }
}

#[derive(Clone)]
enum EvidenceEndpoint {
    Dependabot,
    PullRequests,
    CodeAlerts(String),
    CodeAnalyses(String),
}

impl EvidenceEndpoint {
    fn path(&self) -> &'static str {
        match self {
            Self::Dependabot => "/dependabot/alerts",
            Self::PullRequests => "/pulls",
            Self::CodeAlerts(_) => "/code-scanning/alerts",
            Self::CodeAnalyses(_) => "/code-scanning/analyses",
        }
    }
    fn fixed_query(&self) -> Vec<(&str, &str)> {
        let mut values = vec![("per_page", "100")];
        match self {
            Self::Dependabot => values.extend([
                ("state", "open,fixed,dismissed,auto_dismissed"),
                ("sort", "updated"),
                ("direction", "desc"),
            ]),
            Self::PullRequests => {
                values.extend([("state", "all"), ("sort", "updated"), ("direction", "desc")])
            }
            Self::CodeAlerts(reference) => values.extend([
                ("ref", reference.as_str()),
                ("sort", "updated"),
                ("direction", "desc"),
            ]),
            Self::CodeAnalyses(reference) => values.push(("ref", reference.as_str())),
        }
        values
    }
    fn uses_page(&self) -> bool {
        !matches!(self, Self::Dependabot)
    }
    fn first_url(&self, repository: &GithubSecurityRepository) -> SecurityResult<String> {
        let mut url = crate::Url::parse(&format!("{}{}", repository.api_url(), self.path()))
            .map_err(|_| GithubSecurityError::InvalidSelection)?;
        for (key, value) in self.fixed_query() {
            url.query_pairs_mut().append_pair(key, value);
        }
        if self.uses_page() {
            url.query_pairs_mut().append_pair("page", "1");
        }
        Ok(url.to_string())
    }
    fn validate_next(
        &self,
        value: &str,
        repository: &GithubSecurityRepository,
        current: &str,
    ) -> SecurityResult<String> {
        if value.len() > 4096 {
            return Err(GithubSecurityError::BoundExceeded);
        }
        let url = crate::Url::parse(value).map_err(|_| GithubSecurityError::InvalidResponse)?;
        let expected = crate::Url::parse(&format!("{}{}", repository.api_url(), self.path()))
            .map_err(|_| GithubSecurityError::InvalidSelection)?;
        if url.scheme() != "https"
            || url.host_str() != Some("api.github.com")
            || url.port().is_some()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
            || !url.path().eq_ignore_ascii_case(expected.path())
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        let fixed = self.fixed_query();
        let mut keys = BTreeSet::new();
        let mut pagination = None;
        for (key, value) in url.query_pairs() {
            if !keys.insert(key.to_string()) {
                return Err(GithubSecurityError::InvalidResponse);
            }
            let accepted =
                if let Some((_, expected)) = fixed.iter().find(|(name, _)| *name == key.as_ref()) {
                    value == *expected
                } else {
                    match key.as_ref() {
                        "before" | "after"
                            if matches!(self, Self::Dependabot | Self::CodeAlerts(_)) =>
                        {
                            if pagination.is_some()
                                || value.is_empty()
                                || value.len() > 1024
                                || !value.bytes().all(|byte| byte.is_ascii_graphic())
                            {
                                false
                            } else {
                                pagination = Some(format!("{key}:{value}"));
                                true
                            }
                        }
                        "page" if self.uses_page() => {
                            let previous = crate::Url::parse(current).ok().and_then(|url| {
                                url.query_pairs()
                                    .find(|(key, _)| key == "page")
                                    .and_then(|(_, value)| value.parse::<usize>().ok())
                            });
                            match (previous, value.parse::<usize>().ok()) {
                                (Some(previous), Some(next))
                                    if pagination.is_none()
                                        && previous.checked_add(1) == Some(next)
                                        && next <= MAX_GITHUB_SECURITY_PAGES =>
                                {
                                    pagination = Some(format!("page:{next}"));
                                    true
                                }
                                _ => false,
                            }
                        }
                        _ => false,
                    }
                };
            if !accepted {
                return Err(GithubSecurityError::InvalidResponse);
            }
        }
        if !fixed.iter().all(|(key, _)| keys.contains(*key)) {
            return Err(GithubSecurityError::InvalidResponse);
        }
        pagination.ok_or(GithubSecurityError::InvalidResponse)
    }
}

fn next_link(headers: &http::HeaderMap) -> SecurityResult<Option<String>> {
    let mut text = String::new();
    for value in headers.get_all("link") {
        let value = value
            .to_str()
            .map_err(|_| GithubSecurityError::InvalidResponse)?;
        if text.len() + value.len() + 1 > 8192 {
            return Err(GithubSecurityError::BoundExceeded);
        }
        if !text.is_empty() {
            text.push(',');
        }
        text.push_str(value);
    }
    if text.is_empty() {
        return Ok(None);
    }
    let mut segments = Vec::new();
    let mut start = 0;
    let mut angle = false;
    let mut quoted = false;
    for (index, byte) in text.bytes().enumerate() {
        match byte {
            b'<' if !quoted && !angle => angle = true,
            b'>' if !quoted && angle => angle = false,
            b'"' if !angle => quoted = !quoted,
            b',' if !angle && !quoted => {
                segments.push(&text[start..index]);
                start = index + 1;
            }
            b'\r' | b'\n' => return Err(GithubSecurityError::InvalidResponse),
            _ => {}
        }
    }
    if angle || quoted {
        return Err(GithubSecurityError::InvalidResponse);
    }
    segments.push(&text[start..]);
    if segments.len() > 8 {
        return Err(GithubSecurityError::BoundExceeded);
    }
    let mut next = None;
    for segment in segments {
        let segment = segment.trim();
        let end = segment
            .find('>')
            .ok_or(GithubSecurityError::InvalidResponse)?;
        if !segment.starts_with('<') || end < 2 {
            return Err(GithubSecurityError::InvalidResponse);
        }
        let target = &segment[1..end];
        let mut relation = None;
        for parameter in segment[end + 1..]
            .split(';')
            .filter(|parameter| !parameter.trim().is_empty())
        {
            let (name, value) = parameter
                .trim()
                .split_once('=')
                .ok_or(GithubSecurityError::InvalidResponse)?;
            if name == "rel" {
                if relation.is_some() {
                    return Err(GithubSecurityError::InvalidResponse);
                }
                relation = Some(value.trim().trim_matches('"'));
            }
        }
        let relation = relation.ok_or(GithubSecurityError::InvalidResponse)?;
        if relation
            .split_ascii_whitespace()
            .any(|relation| relation == "next")
        {
            if next.is_some() {
                return Err(GithubSecurityError::InvalidResponse);
            }
            next = Some(target.to_owned());
        }
    }
    Ok(next)
}

impl VerifiedGithubSecurityAuthority {
    async fn collect<
        Raw: serde::de::DeserializeOwned,
        Observation,
        Check: std::future::Future<Output = SecurityResult<()>>,
    >(
        &self,
        endpoint: EvidenceEndpoint,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
        interpret: impl Fn(Raw, &GithubSecurityRepository) -> SecurityResult<Option<(u64, Observation)>>,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> GithubSecurityCollection<Observation> {
        let mut collection = GithubSecurityCollection {
            observations: Vec::new(),
            coverage: GithubSecurityCoverage::Complete,
            pages_completed: 0,
            bytes_collected: 0,
        };
        let mut seen_pages = BTreeSet::new();
        let mut seen_observations = BTreeSet::new();
        let operation: SecurityResult<()> = async {
            let mut url = endpoint.first_url(self.selection().repository())?;
            loop {
                checkpoint().await?;
                self.revalidate_credential()?;
                let allowance = budget.reserve_page()?;
                let request = self.selected.request_at_reviewed_url(&url)?;
                let page: MetadataPage<Vec<Raw>> = self
                    .selected
                    .read_metadata(request, allowance, &client)
                    .await?;
                checkpoint().await?;
                budget.admit_bytes(allowance, page.bytes);
                collection.pages_completed += 1;
                collection.bytes_collected += page.bytes;
                if page.value.len() > 100 {
                    return Err(GithubSecurityError::BoundExceeded);
                }
                for raw in page.value {
                    if let Some((number, observation)) =
                        interpret(raw, self.selection().repository())?
                    {
                        if !seen_observations.insert(number) {
                            return Err(GithubSecurityError::InvalidResponse);
                        }
                        collection.observations.push(observation);
                    }
                }
                let Some(next) = page.pagination? else {
                    break;
                };
                let identity =
                    endpoint.validate_next(&next, self.selection().repository(), &url)?;
                if !seen_pages.insert(identity) {
                    return Err(GithubSecurityError::InvalidResponse);
                }
                url = next;
            }
            checkpoint().await?;
            self.revalidate_credential()
        }
        .await;
        if let Err(error) = operation {
            collection.coverage = if collection.pages_completed == 0 {
                GithubSecurityCoverage::Unavailable(error)
            } else {
                GithubSecurityCoverage::Partial(error)
            };
        }
        collection
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GithubDependabotState {
    Open,
    Fixed,
    Dismissed,
    AutoDismissed,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubDependabotEvidence {
    pub number: u64,
    pub state: GithubDependabotState,
    pub ecosystem: String,
    pub package: String,
    pub manifest_path: Option<String>,
    pub relationship: Option<String>,
    pub scope: Option<String>,
    pub advisory_id: String,
    pub aliases: Vec<String>,
    pub severity: Option<String>,
    pub affected_range: Option<String>,
    pub first_fixed_version: Option<String>,
    pub advisory_published_at: Option<String>,
    pub advisory_updated_at: Option<String>,
    pub withdrawn_at: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub url: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubDependabotPullRequest {
    pub number: u64,
    pub state: String,
    pub head_ref: String,
    pub head_commit: String,
    pub base_ref: String,
    pub base_commit: String,
    pub updated_at: Option<String>,
    pub url: String,
}

#[derive(Deserialize)]
struct PackageMetadata {
    ecosystem: String,
    name: String,
}
#[derive(Deserialize)]
struct DependencyMetadata {
    package: PackageMetadata,
    manifest_path: Option<String>,
    relationship: Option<String>,
    scope: Option<String>,
}
#[derive(Deserialize)]
struct AdvisoryIdentifier {
    value: String,
}
#[derive(Deserialize)]
struct AdvisoryMetadata {
    ghsa_id: String,
    cve_id: Option<String>,
    #[serde(default)]
    identifiers: Vec<AdvisoryIdentifier>,
    severity: Option<String>,
    published_at: Option<String>,
    updated_at: Option<String>,
    withdrawn_at: Option<String>,
}
#[derive(Deserialize)]
struct PatchedVersion {
    identifier: String,
}
#[derive(Deserialize)]
struct VulnerabilityMetadata {
    package: PackageMetadata,
    severity: Option<String>,
    vulnerable_version_range: Option<String>,
    first_patched_version: Option<PatchedVersion>,
}
#[derive(Deserialize)]
struct DependabotMetadata {
    number: u64,
    state: String,
    dependency: DependencyMetadata,
    security_advisory: AdvisoryMetadata,
    security_vulnerability: VulnerabilityMetadata,
    created_at: Option<String>,
    updated_at: Option<String>,
    url: String,
    html_url: String,
}
#[derive(Deserialize)]
struct PullRequestAuthor {
    login: String,
    #[serde(rename = "type")]
    kind: String,
}
#[derive(Deserialize)]
struct PullRequestRepository {
    id: u64,
    full_name: String,
}
#[derive(Deserialize)]
struct PullRequestRef {
    #[serde(rename = "ref")]
    name: String,
    sha: String,
    repo: Option<PullRequestRepository>,
}
#[derive(Deserialize)]
struct PullRequestMetadata {
    number: u64,
    state: String,
    user: Option<PullRequestAuthor>,
    head: PullRequestRef,
    base: PullRequestRef,
    updated_at: Option<String>,
    url: String,
    html_url: String,
}

fn metadata_text(value: &str) -> SecurityResult<()> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return Err(GithubSecurityError::InvalidResponse);
    }
    Ok(())
}
fn optional_metadata(value: &Option<String>) -> SecurityResult<()> {
    if let Some(value) = value {
        metadata_text(value)?;
    }
    Ok(())
}
fn metadata_time(value: &Option<String>) -> SecurityResult<()> {
    if let Some(value) = value {
        if !(20..=35).contains(&value.len())
            || !value.contains('T')
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || b"TZ-+:.".contains(&byte))
        {
            return Err(GithubSecurityError::InvalidResponse);
        }
    }
    Ok(())
}
fn metadata_commit(value: &str) -> SecurityResult<()> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(GithubSecurityError::InvalidResponse);
    }
    Ok(())
}
fn metadata_advisory(value: &str) -> SecurityResult<()> {
    metadata_text(value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        || !["GHSA-", "CVE-", "RUSTSEC-", "OSV-"]
            .iter()
            .any(|prefix| value.starts_with(prefix))
    {
        return Err(GithubSecurityError::InvalidResponse);
    }
    Ok(())
}

impl DependabotMetadata {
    fn evidence(
        self,
        repository: &GithubSecurityRepository,
    ) -> SecurityResult<Option<(u64, GithubDependabotEvidence)>> {
        if self.number == 0
            || !self.url.eq_ignore_ascii_case(&format!(
                "{}/dependabot/alerts/{}",
                repository.api_url(),
                self.number
            ))
            || !self.html_url.eq_ignore_ascii_case(&format!(
                "{}/security/dependabot/{}",
                repository.web_url(),
                self.number
            ))
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        metadata_text(&self.state)?;
        metadata_text(&self.dependency.package.ecosystem)?;
        metadata_text(&self.dependency.package.name)?;
        if self.dependency.package.ecosystem != self.security_vulnerability.package.ecosystem
            || self.dependency.package.name != self.security_vulnerability.package.name
        {
            return Err(GithubSecurityError::InvalidResponse);
        }
        for value in [
            &self.dependency.manifest_path,
            &self.dependency.relationship,
            &self.dependency.scope,
            &self.security_advisory.severity,
            &self.security_vulnerability.severity,
            &self.security_vulnerability.vulnerable_version_range,
        ] {
            optional_metadata(value)?;
        }
        if self.dependency.manifest_path.as_ref().is_some_and(|path| {
            path.starts_with('/')
                || path.contains('\\')
                || path
                    .split('/')
                    .any(|component| component == ".." || component == "." || component.is_empty())
        }) {
            return Err(GithubSecurityError::InvalidResponse);
        }
        metadata_advisory(&self.security_advisory.ghsa_id)?;
        if !self.security_advisory.ghsa_id.starts_with("GHSA-")
            || self.security_advisory.identifiers.len() > 32
        {
            return Err(GithubSecurityError::InvalidResponse);
        }
        let mut aliases = BTreeSet::from([self.security_advisory.ghsa_id.clone()]);
        for value in self.security_advisory.identifiers {
            metadata_advisory(&value.value)?;
            aliases.insert(value.value);
        }
        if let Some(value) = self.security_advisory.cve_id {
            metadata_advisory(&value)?;
            aliases.insert(value);
        }
        let fixed = self
            .security_vulnerability
            .first_patched_version
            .map(|value| value.identifier);
        optional_metadata(&fixed)?;
        for value in [
            &self.security_advisory.published_at,
            &self.security_advisory.updated_at,
            &self.security_advisory.withdrawn_at,
            &self.created_at,
            &self.updated_at,
        ] {
            metadata_time(value)?;
        }
        let state = match self.state.as_str() {
            "open" => GithubDependabotState::Open,
            "fixed" => GithubDependabotState::Fixed,
            "dismissed" => GithubDependabotState::Dismissed,
            "auto_dismissed" => GithubDependabotState::AutoDismissed,
            _ => GithubDependabotState::Unknown,
        };
        Ok(Some((
            self.number,
            GithubDependabotEvidence {
                number: self.number,
                state,
                ecosystem: self.dependency.package.ecosystem,
                package: self.dependency.package.name,
                manifest_path: self.dependency.manifest_path,
                relationship: self.dependency.relationship,
                scope: self.dependency.scope,
                advisory_id: self.security_advisory.ghsa_id,
                aliases: aliases.into_iter().collect(),
                severity: self
                    .security_vulnerability
                    .severity
                    .or(self.security_advisory.severity),
                affected_range: self.security_vulnerability.vulnerable_version_range,
                first_fixed_version: fixed,
                advisory_published_at: self.security_advisory.published_at,
                advisory_updated_at: self.security_advisory.updated_at,
                withdrawn_at: self.security_advisory.withdrawn_at,
                created_at: self.created_at,
                updated_at: self.updated_at,
                url: format!(
                    "{}/security/dependabot/{}",
                    repository.web_url(),
                    self.number
                ),
            },
        )))
    }
}

impl PullRequestMetadata {
    fn evidence(
        self,
        repository: &GithubSecurityRepository,
        repository_id: u64,
    ) -> SecurityResult<Option<(u64, GithubDependabotPullRequest)>> {
        let base = self
            .base
            .repo
            .as_ref()
            .ok_or(GithubSecurityError::RepositoryMismatch)?;
        if self.number == 0
            || base.id != repository_id
            || !base.full_name.eq_ignore_ascii_case(&repository.full_name())
            || !self.url.eq_ignore_ascii_case(&format!(
                "{}/pulls/{}",
                repository.api_url(),
                self.number
            ))
            || !self.html_url.eq_ignore_ascii_case(&format!(
                "{}/pull/{}",
                repository.web_url(),
                self.number
            ))
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        let dependabot = self
            .user
            .as_ref()
            .is_some_and(|author| author.kind == "Bot" && author.login == "dependabot[bot]");
        if !dependabot || !self.head.name.starts_with("dependabot/") {
            return Ok(None);
        }
        for value in [&self.head.name, &self.base.name, &self.state] {
            metadata_text(value)?;
        }
        metadata_commit(&self.head.sha)?;
        metadata_commit(&self.base.sha)?;
        metadata_time(&self.updated_at)?;
        Ok(Some((
            self.number,
            GithubDependabotPullRequest {
                number: self.number,
                state: self.state,
                head_ref: self.head.name,
                head_commit: self.head.sha,
                base_ref: self.base.name,
                base_commit: self.base.sha,
                updated_at: self.updated_at,
                url: format!("{}/pull/{}", repository.web_url(), self.number),
            },
        )))
    }
}

impl VerifiedGithubSecurityAuthority {
    pub async fn read_dependabot(
        &self,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
    ) -> GithubSecurityCollection<GithubDependabotEvidence> {
        self.read_dependabot_with_checkpoint(client, budget, &mut || std::future::ready(Ok(())))
            .await
    }
    pub async fn read_dependabot_with_checkpoint<
        Check: std::future::Future<Output = SecurityResult<()>>,
    >(
        &self,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> GithubSecurityCollection<GithubDependabotEvidence> {
        self.collect(
            EvidenceEndpoint::Dependabot,
            client,
            budget,
            DependabotMetadata::evidence,
            checkpoint,
        )
        .await
    }
    pub async fn read_dependabot_pull_requests(
        &self,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
    ) -> GithubSecurityCollection<GithubDependabotPullRequest> {
        self.read_dependabot_pull_requests_with_checkpoint(client, budget, &mut || {
            std::future::ready(Ok(()))
        })
        .await
    }
    pub async fn read_dependabot_pull_requests_with_checkpoint<
        Check: std::future::Future<Output = SecurityResult<()>>,
    >(
        &self,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> GithubSecurityCollection<GithubDependabotPullRequest> {
        let repository_id = self.metadata.repository_id;
        self.collect(
            EvidenceEndpoint::PullRequests,
            client,
            budget,
            move |raw: PullRequestMetadata, repository| raw.evidence(repository, repository_id),
            checkpoint,
        )
        .await
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubCodeScanningTool {
    pub name: String,
    pub version: Option<String>,
    pub guid: Option<String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubCodeScanningEvidence {
    pub number: u64,
    pub state: String,
    pub rule_id: Option<String>,
    pub severity: Option<String>,
    pub security_severity: Option<String>,
    pub tool: GithubCodeScanningTool,
    pub reference: Option<String>,
    pub commit: Option<String>,
    pub path: Option<String>,
    pub start_line: Option<u64>,
    pub end_line: Option<u64>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub url: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GithubCodeScanningAnalysis {
    pub id: u64,
    pub tool: GithubCodeScanningTool,
    pub reference: String,
    pub commit: String,
    pub created_at: Option<String>,
    pub results_count: Option<u64>,
    pub rules_count: Option<u64>,
    pub category: Option<String>,
    pub observed_error: bool,
    pub observed_warning: bool,
}
#[derive(Deserialize)]
struct CodeToolMetadata {
    name: String,
    version: Option<String>,
    guid: Option<String>,
}
#[derive(Deserialize)]
struct CodeRuleMetadata {
    id: Option<String>,
    severity: Option<String>,
    security_severity_level: Option<String>,
}
#[derive(Deserialize)]
struct CodeLocationMetadata {
    path: Option<String>,
    start_line: Option<u64>,
    end_line: Option<u64>,
}
#[derive(Deserialize)]
struct CodeInstanceMetadata {
    #[serde(rename = "ref")]
    reference: Option<String>,
    commit_sha: Option<String>,
    location: Option<CodeLocationMetadata>,
}
#[derive(Deserialize)]
struct CodeAlertMetadata {
    number: u64,
    state: String,
    rule: CodeRuleMetadata,
    tool: CodeToolMetadata,
    most_recent_instance: Option<CodeInstanceMetadata>,
    created_at: Option<String>,
    updated_at: Option<String>,
    url: String,
    html_url: String,
}
#[derive(Deserialize)]
struct CodeAnalysisMetadata {
    id: u64,
    tool: CodeToolMetadata,
    #[serde(rename = "ref")]
    reference: String,
    commit_sha: String,
    created_at: Option<String>,
    results_count: Option<u64>,
    rules_count: Option<u64>,
    category: Option<String>,
    error: Option<String>,
    warning: Option<String>,
    url: String,
}
fn metadata_reference(value: &str) -> SecurityResult<()> {
    metadata_text(value)?;
    if !value.starts_with("refs/")
        || value.ends_with('/')
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.bytes().any(|byte| b" ~^:?*[\\".contains(&byte))
    {
        return Err(GithubSecurityError::InvalidResponse);
    }
    Ok(())
}
fn metadata_relative_path(value: &str) -> SecurityResult<()> {
    metadata_text(value)?;
    if value.starts_with('/')
        || value.contains('\\')
        || value
            .split('/')
            .any(|component| matches!(component, "" | "." | ".."))
    {
        return Err(GithubSecurityError::InvalidResponse);
    }
    Ok(())
}
impl CodeToolMetadata {
    fn evidence(self) -> SecurityResult<GithubCodeScanningTool> {
        metadata_text(&self.name)?;
        optional_metadata(&self.version)?;
        optional_metadata(&self.guid)?;
        Ok(GithubCodeScanningTool {
            name: self.name,
            version: self.version,
            guid: self.guid,
        })
    }
}
impl CodeAlertMetadata {
    fn evidence(
        self,
        repository: &GithubSecurityRepository,
    ) -> SecurityResult<Option<(u64, GithubCodeScanningEvidence)>> {
        if self.number == 0
            || !self.url.eq_ignore_ascii_case(&format!(
                "{}/code-scanning/alerts/{}",
                repository.api_url(),
                self.number
            ))
            || !self.html_url.eq_ignore_ascii_case(&format!(
                "{}/security/code-scanning/{}",
                repository.web_url(),
                self.number
            ))
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        metadata_text(&self.state)?;
        for value in [
            &self.rule.id,
            &self.rule.severity,
            &self.rule.security_severity_level,
        ] {
            optional_metadata(value)?;
        }
        metadata_time(&self.created_at)?;
        metadata_time(&self.updated_at)?;
        let tool = self.tool.evidence()?;
        let (reference, commit, path, start_line, end_line) = if let Some(instance) =
            self.most_recent_instance
        {
            if let Some(reference) = &instance.reference {
                metadata_reference(reference)?;
            }
            if let Some(commit) = &instance.commit_sha {
                metadata_commit(commit)?;
            }
            let (path, start, end) = if let Some(location) = instance.location {
                if let Some(path) = &location.path {
                    metadata_relative_path(path)?;
                }
                if location.start_line == Some(0)
                    || location.end_line == Some(0)
                    || matches!((location.start_line,location.end_line),(Some(start),Some(end)) if start>end)
                {
                    return Err(GithubSecurityError::InvalidResponse);
                }
                (location.path, location.start_line, location.end_line)
            } else {
                (None, None, None)
            };
            (instance.reference, instance.commit_sha, path, start, end)
        } else {
            (None, None, None, None, None)
        };
        Ok(Some((
            self.number,
            GithubCodeScanningEvidence {
                number: self.number,
                state: self.state,
                rule_id: self.rule.id,
                severity: self.rule.severity,
                security_severity: self.rule.security_severity_level,
                tool,
                reference,
                commit,
                path,
                start_line,
                end_line,
                created_at: self.created_at,
                updated_at: self.updated_at,
                url: format!(
                    "{}/security/code-scanning/{}",
                    repository.web_url(),
                    self.number
                ),
            },
        )))
    }
}
impl CodeAnalysisMetadata {
    fn evidence(
        self,
        repository: &GithubSecurityRepository,
    ) -> SecurityResult<Option<(u64, GithubCodeScanningAnalysis)>> {
        if self.id == 0
            || !self.url.eq_ignore_ascii_case(&format!(
                "{}/code-scanning/analyses/{}",
                repository.api_url(),
                self.id
            ))
        {
            return Err(GithubSecurityError::RepositoryMismatch);
        }
        metadata_reference(&self.reference)?;
        metadata_commit(&self.commit_sha)?;
        metadata_time(&self.created_at)?;
        optional_metadata(&self.category)?;
        let observed_error = self.error.as_ref().is_some_and(|value| !value.is_empty());
        let observed_warning = self.warning.as_ref().is_some_and(|value| !value.is_empty());
        Ok(Some((
            self.id,
            GithubCodeScanningAnalysis {
                id: self.id,
                tool: self.tool.evidence()?,
                reference: self.reference,
                commit: self.commit_sha,
                created_at: self.created_at,
                results_count: self.results_count,
                rules_count: self.rules_count.filter(|count| *count > 0),
                category: self.category,
                observed_error,
                observed_warning,
            },
        )))
    }
}
impl VerifiedGithubSecurityAuthority {
    pub async fn read_code_scanning(
        &self,
        reference: &str,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
    ) -> GithubSecurityCollection<GithubCodeScanningEvidence> {
        self.read_code_scanning_with_checkpoint(reference, client, budget, &mut || {
            std::future::ready(Ok(()))
        })
        .await
    }
    pub async fn read_code_scanning_with_checkpoint<
        Check: std::future::Future<Output = SecurityResult<()>>,
    >(
        &self,
        reference: &str,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> GithubSecurityCollection<GithubCodeScanningEvidence> {
        if metadata_reference(reference).is_err() {
            return GithubSecurityCollection {
                observations: vec![],
                coverage: GithubSecurityCoverage::Unavailable(
                    GithubSecurityError::InvalidSelection,
                ),
                pages_completed: 0,
                bytes_collected: 0,
            };
        }
        self.collect(
            EvidenceEndpoint::CodeAlerts(reference.to_owned()),
            client,
            budget,
            CodeAlertMetadata::evidence,
            checkpoint,
        )
        .await
    }
    pub async fn read_code_scanning_analyses(
        &self,
        reference: &str,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
    ) -> GithubSecurityCollection<GithubCodeScanningAnalysis> {
        self.read_code_scanning_analyses_with_checkpoint(reference, client, budget, &mut || {
            std::future::ready(Ok(()))
        })
        .await
    }
    pub async fn read_code_scanning_analyses_with_checkpoint<
        Check: std::future::Future<Output = SecurityResult<()>>,
    >(
        &self,
        reference: &str,
        client: Arc<dyn HttpClient>,
        budget: &mut GithubSecurityBudget,
        checkpoint: &mut impl FnMut() -> Check,
    ) -> GithubSecurityCollection<GithubCodeScanningAnalysis> {
        if metadata_reference(reference).is_err() {
            return GithubSecurityCollection {
                observations: vec![],
                coverage: GithubSecurityCoverage::Unavailable(
                    GithubSecurityError::InvalidSelection,
                ),
                pages_completed: 0,
                bytes_collected: 0,
            };
        }
        self.collect(
            EvidenceEndpoint::CodeAnalyses(reference.to_owned()),
            client,
            budget,
            CodeAnalysisMetadata::evidence,
            checkpoint,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RequestTimeout, Response};
    use futures::{Future as _, future::BoxFuture};
    use parking_lot::Mutex;
    use std::{
        collections::VecDeque,
        sync::atomic::{AtomicUsize, Ordering},
    };

    struct FixtureClient {
        responses: Mutex<VecDeque<Response<AsyncBody>>>,
        requests: Mutex<Vec<String>>,
        after_send: Option<Arc<dyn Fn() + Send + Sync>>,
    }

    impl HttpClient for FixtureClient {
        fn user_agent(&self) -> Option<&HeaderValue> {
            None
        }
        fn proxy(&self) -> Option<&crate::Url> {
            None
        }
        fn send(
            &self,
            request: Request<AsyncBody>,
        ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
            assert_eq!(request.method(), http::Method::GET);
            assert_eq!(
                request.extensions().get::<RedirectPolicy>(),
                Some(&RedirectPolicy::NoFollow)
            );
            assert_eq!(
                request
                    .extensions()
                    .get::<RequestTimeout>()
                    .map(|timeout| timeout.0),
                Some(REQUEST_TIMEOUT)
            );
            assert_eq!(
                request
                    .headers()
                    .get("X-GitHub-Api-Version")
                    .expect("Pinned version"),
                GITHUB_SECURITY_API_VERSION
            );
            assert_eq!(
                request
                    .headers()
                    .get("Accept-Encoding")
                    .expect("Bounded encoding"),
                "identity"
            );
            assert!(
                request
                    .headers()
                    .get("Authorization")
                    .expect("Selected credential")
                    .is_sensitive()
            );
            assert!(request.headers().get("Cookie").is_none());
            let (parts, mut body) = request.into_parts();
            self.requests.lock().push(parts.uri.to_string());
            let response = self.responses.lock().pop_front();
            if let Some(after_send) = &self.after_send {
                after_send();
            }
            Box::pin(async move {
                let mut bytes = Vec::new();
                body.read_to_end(&mut bytes).await?;
                assert!(bytes.is_empty(), "No source or dependency submission");
                response.ok_or_else(|| anyhow::anyhow!("Unexpected fixture request"))
            })
        }
    }

    fn response(status: u16, body: impl Into<AsyncBody>) -> Response<AsyncBody> {
        Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(body.into())
            .expect("Fixture response")
    }

    fn selection() -> GithubSecuritySelection {
        GithubSecuritySelection::new(
            "selected-user",
            GithubSecurityRepository::new("selected-owner", "selected-repository")
                .expect("Repository"),
        )
        .expect("Selection")
    }

    fn authority(token: Arc<Mutex<Option<String>>>) -> SelectedGithubSecurityAuthority {
        SelectedGithubSecurityAuthority::capture(
            selection(),
            Arc::new(move || token.lock().clone()),
        )
        .expect("Explicit fixture credential owner")
    }

    fn token() -> Arc<Mutex<Option<String>>> {
        Arc::new(Mutex::new(Some("FIXTURE_CREDENTIAL_NOT_EVIDENCE".into())))
    }

    fn client(responses: Vec<Response<AsyncBody>>) -> Arc<FixtureClient> {
        Arc::new(FixtureClient {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
            after_send: None,
        })
    }

    fn account_response() -> Response<AsyncBody> {
        response(
            200,
            r#"{"id":42,"login":"selected-user","ignored_secret":"RAW_BODY_NOT_EVIDENCE"}"#,
        )
    }
    fn repository_response() -> Response<AsyncBody> {
        response(
            200,
            r#"{"id":123,"full_name":"selected-owner/selected-repository","private":true,"default_branch":"dev","ignored_secret":"RAW_BODY_NOT_EVIDENCE"}"#,
        )
    }

    #[test]
    fn doctor_github_authority_exact_account_repository_headers_and_no_preparation_effects() {
        let credential = token();
        let selected = authority(credential.clone());
        let transport = client(vec![account_response(), repository_response()]);
        assert!(transport.requests.lock().is_empty());
        let verified = futures::executor::block_on(selected.verify(transport.clone()))
            .expect("Verified selected identity");
        assert_eq!(
            *transport.requests.lock(),
            [
                "https://api.github.com/user",
                "https://api.github.com/repos/selected-owner/selected-repository"
            ]
        );
        assert_eq!(verified.metadata().account_id, 42);
        assert_eq!(verified.metadata().repository_id, 123);
        assert!(verified.metadata().private_repository);
        assert_eq!(verified.metadata().default_branch.as_deref(), Some("dev"));
        assert!(!format!("{:?}", verified.metadata()).contains("RAW_BODY_NOT_EVIDENCE"));
        *credential.lock() = None;
        assert_eq!(
            verified.revalidate_credential(),
            Err(GithubSecurityError::CredentialChanged)
        );
        assert_eq!(transport.requests.lock().len(), 2);
    }

    #[test]
    fn doctor_github_authority_invalid_selection_missing_credentials_and_wrong_identity() {
        for owner in [
            "",
            "-owner",
            "owner-",
            "owner/name",
            "owner%2fname",
            "owner@example.com",
            "owner\n",
        ] {
            assert!(GithubSecurityRepository::new(owner, "repository").is_err());
        }
        for name in [
            "",
            ".",
            "..",
            "repository/other",
            "repository?token=x",
            "repository#fragment",
            "repository%2fother",
            "repository\\other",
        ] {
            assert!(GithubSecurityRepository::new("owner", name).is_err());
        }
        for value in [
            None,
            Some("".into()),
            Some("token\nother".into()),
            Some("token other".into()),
        ] {
            assert!(
                SelectedGithubSecurityAuthority::capture(
                    selection(),
                    Arc::new(move || value.clone())
                )
                .is_err()
            );
        }
        let wrong_account = client(vec![response(200, r#"{"id":42,"login":"other-user"}"#)]);
        let error = futures::executor::block_on(authority(token()).verify(wrong_account.clone()))
            .err()
            .expect("No verified replacement");
        assert_eq!(error, GithubSecurityError::AccountMismatch);
        assert_eq!(wrong_account.requests.lock().len(), 1);
        let wrong_repository = client(vec![
            account_response(),
            response(
                200,
                r#"{"id":999,"full_name":"other-owner/other-repository","private":false,"default_branch":"main"}"#,
            ),
        ]);
        assert_eq!(
            futures::executor::block_on(authority(token()).verify(wrong_repository.clone())).err(),
            Some(GithubSecurityError::RepositoryMismatch)
        );
        assert_eq!(wrong_repository.requests.lock().len(), 2);
    }

    #[test]
    fn doctor_github_authority_revocation_before_and_during_response_rejects_admission() {
        let credential = token();
        let selected = authority(credential.clone());
        *credential.lock() = Some("REPLACEMENT_CREDENTIAL".into());
        let transport = client(vec![]);
        assert_eq!(
            futures::executor::block_on(selected.verify(transport.clone())).err(),
            Some(GithubSecurityError::CredentialChanged)
        );
        assert!(transport.requests.lock().is_empty());
        let credential = token();
        let selected = authority(credential.clone());
        let transport = Arc::new(FixtureClient {
            responses: Mutex::new(VecDeque::from([account_response()])),
            requests: Mutex::new(Vec::new()),
            after_send: Some(Arc::new(move || *credential.lock() = None)),
        });
        assert_eq!(
            futures::executor::block_on(selected.verify(transport.clone())).err(),
            Some(GithubSecurityError::CredentialChanged)
        );
        assert_eq!(transport.requests.lock().len(), 1);
    }

    #[test]
    fn doctor_github_authority_failed_partial_malformed_and_oversized_data_are_not_verified() {
        for (status, expected) in [
            (301, GithubSecurityError::RedirectRejected),
            (401, GithubSecurityError::AuthenticationUnavailable),
            (403, GithubSecurityError::AccessUnavailable),
            (404, GithubSecurityError::ResourceUnavailable),
            (429, GithubSecurityError::RateLimited),
            (500, GithubSecurityError::ServiceUnavailable),
        ] {
            let transport = client(vec![response(status, "RAW_SERVER_SECRET_NOT_EVIDENCE")]);
            let error = futures::executor::block_on(authority(token()).verify(transport.clone()))
                .err()
                .expect("No identity admission");
            assert_eq!(error, expected);
            assert!(!error.to_string().contains("RAW_SERVER_SECRET"));
            assert_eq!(transport.requests.lock().len(), 1);
        }
        for invalid in [
            response(200, "RAW_SERVER_SECRET_NOT_EVIDENCE"),
            response(200, vec![b' '; MAX_GITHUB_SECURITY_PAGE_BYTES + 1]),
            Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .header("Content-Encoding", "gzip")
                .body(AsyncBody::empty())
                .expect("Invalid encoding"),
            Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .header("Content-Length", MAX_GITHUB_SECURITY_PAGE_BYTES + 1)
                .body(AsyncBody::empty())
                .expect("Oversized header"),
        ] {
            let transport = client(vec![invalid]);
            let error = futures::executor::block_on(authority(token()).verify(transport.clone()))
                .err()
                .expect("No metadata admission");
            assert!(matches!(
                error,
                GithubSecurityError::InvalidResponse | GithubSecurityError::BoundExceeded
            ));
            assert!(!error.to_string().contains("RAW_SERVER_SECRET"));
        }
        let partial = client(vec![
            account_response(),
            response(403, "RAW_SERVER_SECRET_NOT_EVIDENCE"),
        ]);
        assert_eq!(
            futures::executor::block_on(authority(token()).verify(partial)).err(),
            Some(GithubSecurityError::AccessUnavailable)
        );
    }

    #[test]
    fn doctor_github_authority_rate_limit_and_response_framing_remain_incomplete() {
        let limited = Response::builder()
            .status(403)
            .header("X-RateLimit-Remaining", "0")
            .body(AsyncBody::from("RAW_RATE_LIMIT_SECRET"))
            .expect("Rate limit fixture");
        assert_eq!(
            futures::executor::block_on(authority(token()).verify(client(vec![limited]))).err(),
            Some(GithubSecurityError::RateLimited)
        );
        let mismatched = Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .header("Content-Length", "1")
            .body(AsyncBody::from(r#"{"id":42,"login":"selected-user"}"#))
            .expect("Framing fixture");
        assert_eq!(
            futures::executor::block_on(authority(token()).verify(client(vec![mismatched]))).err(),
            Some(GithubSecurityError::InvalidResponse)
        );
        let ambiguous = Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .header("Content-Length", "1")
            .header("Content-Length", "2")
            .body(AsyncBody::empty())
            .expect("Ambiguous framing fixture");
        assert_eq!(
            futures::executor::block_on(authority(token()).verify(client(vec![ambiguous]))).err(),
            Some(GithubSecurityError::InvalidResponse)
        );
    }

    #[test]
    fn doctor_github_authority_cancellation_drops_existing_transport_without_next_request() {
        struct PendingTransport {
            starts: Arc<AtomicUsize>,
            dropped: Arc<AtomicUsize>,
        }
        struct DropReceipt(Arc<AtomicUsize>);
        impl Drop for DropReceipt {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        impl HttpClient for PendingTransport {
            fn user_agent(&self) -> Option<&HeaderValue> {
                None
            }
            fn proxy(&self) -> Option<&crate::Url> {
                None
            }
            fn send(
                &self,
                request: Request<AsyncBody>,
            ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
                assert_eq!(request.uri(), "https://api.github.com/user");
                self.starts.fetch_add(1, Ordering::SeqCst);
                let dropped = self.dropped.clone();
                Box::pin(async move {
                    let _receipt = DropReceipt(dropped);
                    futures::future::pending().await
                })
            }
        }
        let starts = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(PendingTransport {
            starts: starts.clone(),
            dropped: dropped.clone(),
        });
        let mut pending = Box::pin(authority(token()).verify(transport));
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(pending.as_mut().poll(&mut context).is_pending());
        drop(pending);
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
    fn alert(number: u64) -> serde_json::Value {
        serde_json::json!({
            "number": number, "state": "open",
            "dependency": {"package":{"ecosystem":"rust","name":"public-example"},
                "manifest_path":"Cargo.lock", "relationship":"indirect", "scope":"runtime"},
            "security_advisory": {"ghsa_id":"GHSA-2222-3333-4444", "cve_id":"CVE-2026-1234",
                "identifiers":[{"type":"GHSA","value":"GHSA-2222-3333-4444"}],
                "severity":"high", "published_at":"2026-10-01T00:00:00Z",
                "updated_at":"2026-10-02T00:00:00Z", "withdrawn_at":null,
                "summary":"RAW_PRIVATE_SUMMARY_NOT_EVIDENCE", "description":"RAW_SOURCE_NOT_EVIDENCE"},
            "security_vulnerability":{"package":{"ecosystem":"rust","name":"public-example"},
                "severity":"high", "vulnerable_version_range":"< 2.0.0", "first_patched_version":{"identifier":"2.0.0"}},
            "created_at":"2026-10-01T00:00:00Z", "updated_at":"2026-10-02T00:00:00Z",
            "url":format!("https://api.github.com/repos/selected-owner/selected-repository/dependabot/alerts/{number}"),
            "html_url":format!("https://github.com/selected-owner/selected-repository/security/dependabot/{number}"),
            "dismissed_comment":"RAW_CREDENTIAL_NOT_EVIDENCE"
        })
    }
    fn pull_request(number: u64) -> serde_json::Value {
        serde_json::json!({"number":number,"state":"open", "user":{"login":"dependabot[bot]","type":"Bot"},
            "head":{"ref":"dependabot/cargo/public-example-2.0.0","sha":"a".repeat(40),"repo":null},
            "base":{"ref":"dev","sha":"b".repeat(40),"repo":{"id":123,"full_name":"selected-owner/selected-repository"}},
            "updated_at":"2026-10-02T00:00:00Z", "title":"RAW_TITLE_NOT_EVIDENCE","body":"RAW_SOURCE_NOT_EVIDENCE",
            "url":format!("https://api.github.com/repos/selected-owner/selected-repository/pulls/{number}"),
            "html_url":format!("https://github.com/selected-owner/selected-repository/pull/{number}")})
    }
    fn metadata_response(
        values: Vec<serde_json::Value>,
        next: Option<&str>,
    ) -> Response<AsyncBody> {
        let mut result = response(200, serde_json::to_string(&values).expect("Fixture JSON"));
        if let Some(next) = next {
            result.headers_mut().insert(
                "link",
                HeaderValue::from_str(&format!("<{next}>; rel=\"next\"")).expect("Fixture link"),
            );
        }
        result
    }
    fn verified_fixture(
        responses: Vec<Response<AsyncBody>>,
    ) -> (
        VerifiedGithubSecurityAuthority,
        Arc<FixtureClient>,
        Arc<Mutex<Option<String>>>,
    ) {
        let credential = token();
        let mut all = vec![account_response(), repository_response()];
        all.extend(responses);
        let transport = client(all);
        let verified =
            futures::executor::block_on(authority(credential.clone()).verify(transport.clone()))
                .expect("Verified fixture");
        (verified, transport, credential)
    }
    fn alert_next(cursor: &str) -> String {
        format!(
            "https://api.github.com/repos/selected-owner/selected-repository/dependabot/alerts?per_page=100&state=open,fixed,dismissed,auto_dismissed&sort=updated&direction=desc&after={cursor}"
        )
    }

    #[test]
    fn doctor_github_dependabot_complete_pagination_preserves_only_bounded_metadata() {
        let next = alert_next("cursor-one");
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![alert(1)], Some(&next)),
            metadata_response(vec![alert(2)], None),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(evidence.coverage, GithubSecurityCoverage::Complete);
        assert_eq!(evidence.pages_completed, 2);
        assert_eq!(evidence.observations.len(), 2);
        let first = evidence.observations.first().expect("Observed alert");
        assert_eq!(first.package, "public-example");
        assert_eq!(first.first_fixed_version.as_deref(), Some("2.0.0"));
        assert_eq!(first.aliases.len(), 2);
        assert!(!format!("{evidence:?}").contains("RAW_"));
        assert_eq!(transport.requests.lock().last(), Some(&next));
        assert!(evidence.bytes_collected > 0);
    }

    #[test]
    fn doctor_github_dependabot_unavailable_partial_and_empty_are_distinct() {
        for (status, error) in [
            (401, GithubSecurityError::AuthenticationUnavailable),
            (403, GithubSecurityError::AccessUnavailable),
            (404, GithubSecurityError::ResourceUnavailable),
            (429, GithubSecurityError::RateLimited),
            (500, GithubSecurityError::ServiceUnavailable),
        ] {
            let (verified, transport, _) =
                verified_fixture(vec![response(status, "RAW_SECRET_NOT_EVIDENCE")]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let evidence =
                futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
            assert_eq!(
                evidence.coverage,
                GithubSecurityCoverage::Unavailable(error)
            );
            assert_eq!(evidence.pages_completed, 0);
        }
        let next = alert_next("cursor-one");
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![alert(1)], Some(&next)),
            response(403, "RAW_SECRET_NOT_EVIDENCE"),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::AccessUnavailable)
        );
        assert_eq!(evidence.observations.len(), 1);
        let (verified, transport, _) = verified_fixture(vec![metadata_response(vec![], None)]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
        assert_eq!(evidence.coverage, GithubSecurityCoverage::Complete);
        assert!(evidence.observations.is_empty());
        assert_eq!(evidence.pages_completed, 1);
    }

    #[test]
    fn doctor_github_dependabot_links_cannot_change_authority_filters_or_repeat_cursor() {
        let safe = alert_next("cursor-one");
        for next in [
            safe.replace("api.github.com", "evil.example"),
            safe.replace("selected-repository", "other-repository"),
            safe.replace("direction=desc", "direction=asc"),
            format!("{safe}&per_page=1"),
            format!("{safe}#fragment"),
            safe.replace("https://", "http://"),
        ] {
            let (verified, transport, _) =
                verified_fixture(vec![metadata_response(vec![alert(1)], Some(&next))]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let evidence = futures::executor::block_on(
                verified.read_dependabot(transport.clone(), &mut budget),
            );
            assert!(matches!(
                evidence.coverage,
                GithubSecurityCoverage::Partial(_)
            ));
            assert_eq!(
                transport.requests.lock().len(),
                3,
                "No unreviewed destination request"
            );
            assert_eq!(evidence.observations.len(), 1);
        }
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![alert(1)], Some(&safe)),
            metadata_response(vec![alert(2)], Some(&safe)),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::InvalidResponse)
        );
        assert_eq!(evidence.observations.len(), 2);
        assert_eq!(transport.requests.lock().len(), 4);
    }

    #[test]
    fn doctor_github_dependabot_page_byte_and_record_bounds_remain_incomplete() {
        let next = alert_next("cursor-one");
        let (verified, transport, _) =
            verified_fixture(vec![metadata_response(vec![alert(1)], Some(&next))]);
        let mut budget = GithubSecurityBudget::with_limits(1, MAX_GITHUB_SECURITY_COLLECTION_BYTES)
            .expect("Bounded budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::BoundExceeded)
        );
        assert_eq!(transport.requests.lock().len(), 3);
        let (verified, transport, _) =
            verified_fixture(vec![metadata_response(vec![alert(1)], None)]);
        let mut budget = GithubSecurityBudget::with_limits(1, 8).expect("Tiny budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Unavailable(GithubSecurityError::BoundExceeded)
        );
        for values in [vec![alert(1), alert(1)], (1..=101).map(alert).collect()] {
            let (verified, transport, _) = verified_fixture(vec![metadata_response(values, None)]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let evidence =
                futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
            assert!(matches!(
                evidence.coverage,
                GithubSecurityCoverage::Partial(_)
            ));
        }
    }

    #[test]
    fn doctor_github_dependabot_revocation_before_and_during_collection_never_completes() {
        let (verified, transport, credential) =
            verified_fixture(vec![metadata_response(vec![alert(1)], None)]);
        *credential.lock() = None;
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Unavailable(GithubSecurityError::CredentialChanged)
        );
        assert_eq!(transport.requests.lock().len(), 2);
        let next = alert_next("cursor-one");
        let (verified, _, credential) = verified_fixture(vec![]);
        let count = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(FixtureClient {
            responses: Mutex::new(
                vec![
                    metadata_response(vec![alert(1)], Some(&next)),
                    metadata_response(vec![alert(2)], None),
                ]
                .into(),
            ),
            requests: Mutex::new(vec![]),
            after_send: Some(Arc::new(move || {
                if count.fetch_add(1, Ordering::SeqCst) == 1 {
                    *credential.lock() = None;
                }
            })),
        });
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::CredentialChanged)
        );
        assert_eq!(evidence.observations.len(), 1);
        assert_eq!(transport.requests.lock().len(), 2);
    }

    #[test]
    fn doctor_github_dependabot_invalid_metadata_is_not_an_admitted_finding() {
        let mut outside = alert(1);
        outside["dependency"]["manifest_path"] = "../private/Cargo.lock".into();
        let mut swapped = alert(1);
        swapped["security_vulnerability"]["package"]["name"] = "other".into();
        let mut injection = alert(1);
        injection["security_advisory"]["ghsa_id"] = "GHSA-2222\nSECRET".into();
        let mut foreign = alert(1);
        foreign["html_url"] = "https://evil.example/secret".into();
        for value in [outside, swapped, injection, foreign] {
            let (verified, transport, _) =
                verified_fixture(vec![metadata_response(vec![value], None)]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let evidence =
                futures::executor::block_on(verified.read_dependabot(transport, &mut budget));
            assert!(matches!(
                evidence.coverage,
                GithubSecurityCoverage::Partial(_)
            ));
            assert!(evidence.observations.is_empty());
            assert!(!format!("{evidence:?}").contains("SECRET"));
        }
    }

    #[test]
    fn doctor_github_dependabot_pull_requests_are_candidates_without_alert_association() {
        let mut human = pull_request(2);
        human["user"]["type"] = "User".into();
        let mut lookalike = pull_request(3);
        lookalike["user"]["login"] = "dependabot-lookalike".into();
        let (verified, transport, _) = verified_fixture(vec![metadata_response(
            vec![pull_request(1), human, lookalike],
            None,
        )]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence = futures::executor::block_on(
            verified.read_dependabot_pull_requests(transport.clone(), &mut budget),
        );
        assert_eq!(evidence.coverage, GithubSecurityCoverage::Complete);
        assert_eq!(evidence.observations.len(), 1);
        assert_eq!(
            evidence.observations.first().expect("Candidate").base_ref,
            "dev"
        );
        assert!(!format!("{evidence:?}").contains("RAW_"));
        assert!(
            transport
                .requests
                .lock()
                .last()
                .expect("Read")
                .contains("state=all")
        );
        let mut swapped = pull_request(1);
        swapped["base"]["repo"]["id"] = 124.into();
        let (verified, transport, _) =
            verified_fixture(vec![metadata_response(vec![swapped], None)]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let evidence = futures::executor::block_on(
            verified.read_dependabot_pull_requests(transport, &mut budget),
        );
        assert_eq!(
            evidence.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::RepositoryMismatch)
        );
        assert!(evidence.observations.is_empty());
    }

    #[test]
    fn doctor_github_dependabot_pull_request_pagination_shares_the_operation_budget() {
        let next = "https://api.github.com/repos/selected-owner/selected-repository/pulls?per_page=100&state=all&sort=updated&direction=desc&page=2";
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![alert(1)], None),
            metadata_response(vec![pull_request(1)], Some(next)),
        ]);
        let mut budget = GithubSecurityBudget::with_limits(2, MAX_GITHUB_SECURITY_COLLECTION_BYTES)
            .expect("Shared budget");
        let alerts =
            futures::executor::block_on(verified.read_dependabot(transport.clone(), &mut budget));
        assert_eq!(alerts.coverage, GithubSecurityCoverage::Complete);
        let pulls = futures::executor::block_on(
            verified.read_dependabot_pull_requests(transport.clone(), &mut budget),
        );
        assert_eq!(
            pulls.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::BoundExceeded)
        );
        assert_eq!(pulls.observations.len(), 1);
        assert_eq!(transport.requests.lock().len(), 4);
    }
    fn code_alert(number: u64) -> serde_json::Value {
        serde_json::json!({"number":number,"state":"open","rule":{"id":"rust/example","severity":"warning","security_severity_level":"high","description":"RAW_SOURCE_NOT_EVIDENCE"},
            "tool":{"name":"Selected Scanner","version":"1.2.3","guid":null},"created_at":"2026-10-01T00:00:00Z","updated_at":"2026-10-02T00:00:00Z",
            "most_recent_instance":{"ref":"refs/heads/dev","commit_sha":"b".repeat(40),"location":{"path":"src/main.rs","start_line":12,"end_line":14},"message":{"text":"RAW_SECRET_NOT_EVIDENCE"},"classifications":["RAW_SOURCE_NOT_EVIDENCE"]},
            "url":format!("https://api.github.com/repos/selected-owner/selected-repository/code-scanning/alerts/{number}"),
            "html_url":format!("https://github.com/selected-owner/selected-repository/security/code-scanning/{number}")})
    }
    fn analysis(number: u64) -> serde_json::Value {
        serde_json::json!({"id":number,"ref":"refs/heads/dev","commit_sha":"b".repeat(40),"created_at":"2026-10-02T00:00:00Z","results_count":1,"rules_count":49,
            "category":"rust","tool":{"name":"Selected Scanner","version":"1.2.3","guid":null},"environment":"RAW_CREDENTIAL_NOT_EVIDENCE","error":"","warning":"",
            "url":format!("https://api.github.com/repos/selected-owner/selected-repository/code-scanning/analyses/{number}"),"sarif_id":"RAW_SARIF_NOT_EVIDENCE"})
    }
    #[test]
    fn doctor_github_code_scanning_metadata_has_selected_ref_scanner_and_no_raw_evidence() {
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![code_alert(1)], None),
            metadata_response(vec![analysis(2)], None),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let alerts = futures::executor::block_on(verified.read_code_scanning(
            "refs/heads/dev",
            transport.clone(),
            &mut budget,
        ));
        let analyses = futures::executor::block_on(verified.read_code_scanning_analyses(
            "refs/heads/dev",
            transport.clone(),
            &mut budget,
        ));
        assert_eq!(alerts.coverage, GithubSecurityCoverage::Complete);
        assert_eq!(analyses.coverage, GithubSecurityCoverage::Complete);
        let alert = alerts.observations.first().expect("Alert");
        assert_eq!(alert.tool.version.as_deref(), Some("1.2.3"));
        assert_eq!(alert.path.as_deref(), Some("src/main.rs"));
        let analysis = analyses.observations.first().expect("Analysis");
        assert_eq!(analysis.rules_count, Some(49));
        assert_eq!(analysis.results_count, Some(1));
        assert!(!format!("{alerts:?}{analyses:?}").contains("RAW_"));
        for request in transport.requests.lock().iter().skip(2) {
            let url = crate::Url::parse(request).expect("Request URL");
            assert!(
                url.query_pairs()
                    .any(|(key, value)| key == "ref" && value == "refs/heads/dev")
            );
            assert!(!request.contains("sarif"));
            assert!(!request.contains("secret-scanning"));
        }
    }
    #[test]
    fn doctor_github_code_scanning_missing_provenance_failed_analysis_and_mismatch_stay_visible() {
        let mut absent = code_alert(1);
        absent["most_recent_instance"] = serde_json::Value::Null;
        absent["tool"]["version"] = serde_json::Value::Null;
        let mut mismatch = code_alert(2);
        mismatch["most_recent_instance"]["ref"] = "refs/heads/other".into();
        mismatch["most_recent_instance"]["commit_sha"] = "c".repeat(40).into();
        let mut failed = analysis(1);
        failed["rules_count"] = 0.into();
        failed["error"] = "RAW_SECRET_ERROR_NOT_EVIDENCE".into();
        failed["warning"] = "RAW_SECRET_WARNING_NOT_EVIDENCE".into();
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![absent, mismatch], None),
            metadata_response(vec![failed], None),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let alerts = futures::executor::block_on(verified.read_code_scanning(
            "refs/heads/dev",
            transport.clone(),
            &mut budget,
        ));
        assert_eq!(alerts.observations.first().expect("Unknown").commit, None);
        assert_eq!(
            alerts
                .observations
                .get(1)
                .expect("Mismatch")
                .reference
                .as_deref(),
            Some("refs/heads/other")
        );
        let analyses = futures::executor::block_on(verified.read_code_scanning_analyses(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        let failed = analyses.observations.first().expect("Failed analysis");
        assert!(failed.observed_error);
        assert!(failed.observed_warning);
        assert_eq!(failed.rules_count, None);
        assert!(!format!("{analyses:?}").contains("RAW_"));
    }
    #[test]
    fn doctor_github_code_scanning_pagination_cannot_change_ref_or_endpoint() {
        let next = "https://api.github.com/repos/selected-owner/selected-repository/code-scanning/alerts?per_page=100&ref=refs%2Fheads%2Fdev&sort=updated&direction=desc&page=2";
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![code_alert(1)], Some(next)),
            metadata_response(vec![code_alert(2)], None),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let alerts = futures::executor::block_on(verified.read_code_scanning(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        assert_eq!(alerts.coverage, GithubSecurityCoverage::Complete);
        assert_eq!(alerts.observations.len(), 2);
        for next in [
            next.replace("%2Fdev", "%2Fother"),
            next.replace("/alerts?", "/analyses?"),
            next.replace("api.github.com", "evil.example"),
            next.replace("page=2", "page=4"),
        ] {
            let (verified, transport, _) =
                verified_fixture(vec![metadata_response(vec![code_alert(1)], Some(&next))]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let alerts = futures::executor::block_on(verified.read_code_scanning(
                "refs/heads/dev",
                transport.clone(),
                &mut budget,
            ));
            assert!(matches!(
                alerts.coverage,
                GithubSecurityCoverage::Partial(_)
            ));
            assert_eq!(transport.requests.lock().len(), 3);
        }
    }
    #[test]
    fn doctor_github_code_scanning_unavailable_empty_and_partial_coverage_are_not_security_claims()
    {
        for status in [403, 404, 429, 503] {
            let (verified, transport, _) =
                verified_fixture(vec![response(status, "RAW_ERROR_NOT_EVIDENCE")]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let alerts = futures::executor::block_on(verified.read_code_scanning(
                "refs/heads/dev",
                transport,
                &mut budget,
            ));
            assert!(matches!(
                alerts.coverage,
                GithubSecurityCoverage::Unavailable(_)
            ));
        }
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![], None),
            metadata_response(vec![], None),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let alerts = futures::executor::block_on(verified.read_code_scanning(
            "refs/heads/dev",
            transport.clone(),
            &mut budget,
        ));
        let analyses = futures::executor::block_on(verified.read_code_scanning_analyses(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        assert_eq!(alerts.coverage, GithubSecurityCoverage::Complete);
        assert_eq!(analyses.coverage, GithubSecurityCoverage::Complete);
        assert!(
            analyses.observations.is_empty(),
            "No analysis cannot prove scanner coverage"
        );
        let next = "https://api.github.com/repos/selected-owner/selected-repository/code-scanning/analyses?per_page=100&ref=refs%2Fheads%2Fdev&page=2";
        let (verified, transport, _) = verified_fixture(vec![
            metadata_response(vec![analysis(1)], Some(next)),
            response(403, "RAW_ERROR_NOT_EVIDENCE"),
        ]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let analyses = futures::executor::block_on(verified.read_code_scanning_analyses(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        assert_eq!(
            analyses.coverage,
            GithubSecurityCoverage::Partial(GithubSecurityError::AccessUnavailable)
        );
        assert_eq!(analyses.observations.len(), 1);
    }
    #[test]
    fn doctor_github_code_scanning_invalid_selection_location_and_revocation_block_admission() {
        let (verified, transport, credential) = verified_fixture(vec![]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        for reference in [
            "dev",
            "refs/heads/../private",
            "refs/heads/dev?token=x",
            "refs/heads/dev\n",
        ] {
            let result = futures::executor::block_on(verified.read_code_scanning(
                reference,
                transport.clone(),
                &mut budget,
            ));
            assert_eq!(
                result.coverage,
                GithubSecurityCoverage::Unavailable(GithubSecurityError::InvalidSelection)
            );
        }
        assert_eq!(transport.requests.lock().len(), 2);
        *credential.lock() = None;
        let result = futures::executor::block_on(verified.read_code_scanning_analyses(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        assert_eq!(
            result.coverage,
            GithubSecurityCoverage::Unavailable(GithubSecurityError::CredentialChanged)
        );
        let mut bad = code_alert(1);
        bad["most_recent_instance"]["location"]["path"] = "../../private/source.rs".into();
        let (verified, transport, _) = verified_fixture(vec![metadata_response(vec![bad], None)]);
        let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
        let result = futures::executor::block_on(verified.read_code_scanning(
            "refs/heads/dev",
            transport,
            &mut budget,
        ));
        assert!(matches!(
            result.coverage,
            GithubSecurityCoverage::Partial(_)
        ));
        assert!(result.observations.is_empty());
    }
    #[test]
    fn doctor_github_review_authority_checkpoints_precede_requests_and_late_identity_admission() {
        for denied in 1..=3 {
            let selected = authority(token());
            let transport = client(vec![account_response(), repository_response()]);
            let mut calls = 0;
            let mut checkpoint = || {
                calls += 1;
                std::future::ready(if calls == denied {
                    Err(GithubSecurityError::AuthorityChanged)
                } else {
                    Ok(())
                })
            };
            let result = futures::executor::block_on(
                selected.verify_with_checkpoint(transport.clone(), &mut checkpoint),
            );
            assert_eq!(result.err(), Some(GithubSecurityError::AuthorityChanged));
            assert_eq!(transport.requests.lock().len(), denied - 1);
        }
    }
    #[test]
    fn doctor_github_review_each_page_checkpoint_blocks_stale_dispatch_and_late_observations() {
        for denied in [3, 4, 5] {
            let next = alert_next("cursor-one");
            let (verified, transport, _) = verified_fixture(vec![
                metadata_response(vec![alert(1)], Some(&next)),
                metadata_response(vec![alert(2)], None),
            ]);
            let mut budget = GithubSecurityBudget::after_verification(&verified).expect("Budget");
            let mut calls = 0;
            let mut checkpoint = || {
                calls += 1;
                std::future::ready(if calls == denied {
                    Err(GithubSecurityError::AuthorityChanged)
                } else {
                    Ok(())
                })
            };
            let result = futures::executor::block_on(verified.read_dependabot_with_checkpoint(
                transport.clone(),
                &mut budget,
                &mut checkpoint,
            ));
            assert_eq!(
                result.coverage,
                GithubSecurityCoverage::Partial(GithubSecurityError::AuthorityChanged)
            );
            assert_eq!(
                transport.requests.lock().len(),
                if denied == 3 { 3 } else { 4 }
            );
            assert_eq!(result.observations.len(), if denied == 5 { 2 } else { 1 });
        }
    }
    #[test]
    fn doctor_github_review_exact_preview_never_executes_or_substitutes_a_reference() {
        let preview = selection()
            .reviewed_evidence_requests(Some("refs/heads/dev"), true, true)
            .expect("Exact preview");
        assert_eq!(preview.len(), 6);
        assert_eq!(
            preview.first().map(String::as_str),
            Some("https://api.github.com/user")
        );
        for request in preview.iter().skip(2) {
            let parsed = crate::Url::parse(request).expect("Exact request");
            assert_eq!(parsed.host_str(), Some("api.github.com"));
            assert!(
                parsed
                    .path()
                    .starts_with("/repos/selected-owner/selected-repository/")
            );
        }
        assert!(
            selection()
                .reviewed_evidence_requests(None, false, true)
                .is_err()
        );
        assert!(
            selection()
                .reviewed_evidence_requests(None, false, false)
                .is_err()
        );
        assert_eq!(
            selection()
                .reviewed_evidence_requests(None, true, false)
                .expect("Dependabot only")
                .len(),
            4
        );
    }
}
