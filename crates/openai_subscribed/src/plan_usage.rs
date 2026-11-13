use super::{PROVIDER_ID, PROVIDER_NAME};
use anyhow::{Context as _, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use futures::{FutureExt as _, StreamExt as _, future::BoxFuture};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, WeakEntity};
use http_client::{AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest};
use language_model::{
    LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelId,
    LanguageModelName, LanguageModelProviderId, LanguageModelProviderName, LanguageModelRequest,
    LanguageModelToolChoice,
};
use open_ai::{
    completion::{OpenAiResponseEventMapper, into_open_ai_response},
    responses::stream_response,
};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use url::form_urlencoded;
use util::ResultExt as _;

const TOKEN_REFRESH_BUFFER_MS: u64 = 5 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

use http_client::{HttpRequestExt as _, RedirectPolicy, StatusCode};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, jwk::JwkSet};
use open_ai::responses::StreamEvent;
use smol::io::AsyncReadExt as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

const ISSUER: &str = "https://auth.openai.com";
const AUTHORIZE: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN: &str = "https://auth.openai.com/api/accounts/oauth/token";
const JWKS: &str = "https://auth.openai.com/.well-known/jwks.json";
const DISCOVERY: &str = "https://auth.openai.com/.well-known/openid-configuration";
const RESOURCE: &str = "https://api.openai.com/v1";
const MODELS: &str = "https://api.openai.com/v1/models";
const DYNAMIC_CLIENT: &str = "dynamic_agent_client";
const PLAN_SCOPE: &str = "chatgpt.tokens.use.direct";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
const MAX_RESPONSE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
struct Tokens {
    access_token: String,
    refresh_token: String,
    id_token: String,
    expires_at_ms: u64,
    scope: String,
}

impl Tokens {
    fn can_use_plan(&self) -> bool {
        !self.access_token.is_empty()
            && self
                .scope
                .split_ascii_whitespace()
                .any(|scope| scope == PLAN_SCOPE)
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Account {
    client_id: String,
    subject: String,
    email: Option<String>,
    tokens: Option<Tokens>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Accounts {
    host_id: String,
    active: Option<String>,
    pending_client: Option<String>,
    profiles: Vec<Account>,
}

impl Default for Accounts {
    fn default() -> Self {
        Self {
            host_id: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
            active: None,
            pending_client: None,
            profiles: Vec::new(),
        }
    }
}

impl Accounts {
    fn active(&self) -> Option<&Account> {
        self.account(self.active.as_deref()?)
    }

    fn account(&self, client_id: &str) -> Option<&Account> {
        self.profiles
            .iter()
            .find(|account| account.client_id == client_id)
    }
}

fn read_accounts(path: &Path) -> Result<Accounts> {
    match std::fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes).context("Could not read saved ChatGPT accounts")
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Accounts::default()),
        Err(error) => Err(error).context("Could not open saved ChatGPT accounts"),
    }
}

fn save_accounts(path: &Path, accounts: &Accounts) -> Result<()> {
    use std::io::Write as _;

    let directory = path
        .parent()
        .context("Missing ChatGPT credentials directory")?;
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    // NamedTempFile creates an owner-only file on Unix. Persist replaces the old
    // record atomically, including the rotated refresh token and its scope.
    serde_json::to_writer(file.as_file_mut(), accounts)?;
    file.flush()?;
    file.as_file().sync_all()?;
    file.persist(path)
        .context("Could not save ChatGPT accounts")?;
    Ok(())
}

#[derive(Clone, Deserialize)]
struct AccountModel {
    slug: String,
    display_name: String,
    visibility: String,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    input_modalities: Vec<String>,
}

#[derive(Deserialize)]
struct Catalog {
    models: Vec<AccountModel>,
}

async fn read_json<T: serde::de::DeserializeOwned>(
    http: &dyn HttpClient,
    request: HttpRequest<AsyncBody>,
    operation: &str,
) -> Result<T> {
    let response = http
        .send(request)
        .await
        .with_context(|| format!("{operation} could not connect"))?;
    if !response.status().is_success() {
        return Err(anyhow::Error::new(RequestStatus(response.status()))
            .context(format!("{operation} failed (HTTP {})", response.status())));
    }
    let mut bytes = Vec::new();
    response
        .into_body()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut bytes)
        .await?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_RESPONSE_BYTES,
        "{operation} response was too large"
    );
    serde_json::from_slice(&bytes)
        .with_context(|| format!("{operation} returned an invalid response"))
}

#[derive(Debug)]
struct RequestStatus(StatusCode);

impl std::fmt::Display for RequestStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for RequestStatus {}

fn form_request(url: &str, fields: &[(&str, &str)]) -> Result<HttpRequest<AsyncBody>> {
    let body = form_urlencoded::Serializer::new(String::new())
        .extend_pairs(fields.iter().copied())
        .finish();
    Ok(HttpRequest::builder()
        .method(Method::POST)
        .uri(url)
        .follow_redirects(RedirectPolicy::NoFollow)
        .timeout(Duration::from_secs(20))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?)
}

#[derive(Deserialize)]
struct TokenGrant {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: u64,
    scope: Option<String>,
    token_type: String,
}

impl TokenGrant {
    fn into_tokens(self, previous: Option<&Tokens>) -> Result<Tokens> {
        anyhow::ensure!(
            self.token_type.eq_ignore_ascii_case("Bearer") && !self.access_token.is_empty(),
            "ChatGPT returned an invalid bearer token"
        );
        Ok(Tokens {
            access_token: self.access_token,
            refresh_token: self
                .refresh_token
                .or_else(|| previous.map(|tokens| tokens.refresh_token.clone()))
                .filter(|token| !token.is_empty())
                .context("ChatGPT did not provide a refresh token")?,
            id_token: self
                .id_token
                .or_else(|| previous.map(|tokens| tokens.id_token.clone()))
                .context("ChatGPT did not provide an ID token")?,
            expires_at_ms: now_ms().saturating_add(self.expires_in.saturating_mul(1000)),
            scope: self
                .scope
                .or_else(|| previous.map(|tokens| tokens.scope.clone()))
                .unwrap_or_default(),
        })
    }
}

#[derive(Deserialize)]
struct Identity {
    sub: String,
    nonce: Option<String>,
    email: Option<String>,
    azp: Option<String>,
    aud: serde_json::Value,
}

async fn verify_identity(
    http: &dyn HttpClient,
    token: &str,
    client_id: &str,
    nonce: Option<&str>,
) -> Result<Identity> {
    let header = jsonwebtoken::decode_header(token).context("Invalid ChatGPT ID token header")?;
    anyhow::ensure!(
        header.alg == Algorithm::RS256,
        "Unsupported ChatGPT ID token signature algorithm"
    );
    let key_id = header
        .kid
        .context("ChatGPT ID token has no signing key ID")?;
    let keys: JwkSet = read_json(http, get_request(JWKS)?, "ChatGPT signing keys").await?;
    let key = keys
        .find(&key_id)
        .context("ChatGPT signing key was not found")?;
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_issuer(&[ISSUER]);
    validation.set_audience(&[client_id]);
    validation.set_required_spec_claims(&["exp", "iss", "aud", "sub"]);
    let identity =
        jsonwebtoken::decode::<Identity>(token, &DecodingKey::from_jwk(key)?, &validation)
            .context("ChatGPT ID token could not be verified")?
            .claims;
    anyhow::ensure!(!identity.sub.is_empty(), "ChatGPT ID token has no subject");
    if let Some(nonce) = nonce {
        anyhow::ensure!(
            identity.nonce.as_deref() == Some(nonce),
            "ChatGPT ID token nonce mismatch"
        );
    }
    if identity.azp.is_some()
        || identity
            .aud
            .as_array()
            .is_some_and(|audience| audience.len() > 1)
    {
        anyhow::ensure!(
            identity.azp.as_deref() == Some(client_id),
            "ChatGPT ID token authorized party mismatch"
        );
    }
    Ok(identity)
}

fn get_request(url: &str) -> Result<HttpRequest<AsyncBody>> {
    Ok(HttpRequest::builder()
        .uri(url)
        .follow_redirects(RedirectPolicy::NoFollow)
        .timeout(Duration::from_secs(20))
        .body(AsyncBody::empty())?)
}

fn random_oauth_value() -> String {
    let mut bytes = [0; 32];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

struct Authorization {
    state: String,
    nonce: String,
    verifier: String,
}

impl Authorization {
    fn new() -> Self {
        Self {
            state: random_oauth_value(),
            nonce: random_oauth_value(),
            verifier: random_oauth_value(),
        }
    }

    fn url(&self, redirect: &str, accounts: &Accounts, selected: Option<&str>) -> Result<url::Url> {
        let mut url = url::Url::parse(AUTHORIZE)?;
        let mut query = url.query_pairs_mut();
        query.extend_pairs([
            ("client_id", selected.unwrap_or(DYNAMIC_CLIENT)),
            ("ext_agent_host_id", &accounts.host_id),
            ("response_type", "code"),
            ("redirect_uri", redirect),
            ("scope", SCOPE),
            ("resource", RESOURCE),
            ("state", &self.state),
            ("nonce", &self.nonce),
            ("code_challenge_method", "S256"),
        ]);
        query.append_pair(
            "code_challenge",
            &URL_SAFE_NO_PAD.encode(Sha256::digest(&self.verifier)),
        );
        if selected.is_none() {
            query.append_pair("agent_name_hint", paths::APP_NAME);
        }
        if let Some(account) = selected.and_then(|id| accounts.account(id)) {
            if let Some(tokens) = &account.tokens {
                query.append_pair("id_token_hint", &tokens.id_token);
            }
            if let Some(email) = &account.email {
                query.append_pair("login_hint", email);
            }
            if !account.tokens.as_ref().is_some_and(Tokens::can_use_plan) {
                query.append_pair("prompt", "consent");
            }
        }
        drop(query);
        Ok(url)
    }

    async fn exchange(
        &self,
        http: &dyn HttpClient,
        code: &str,
        client_id: &str,
        redirect: &str,
        previous: Option<&Account>,
    ) -> Result<Account> {
        let grant: TokenGrant = read_json(
            http,
            form_request(
                TOKEN,
                &[
                    ("grant_type", "authorization_code"),
                    ("client_id", client_id),
                    ("code", code),
                    ("code_verifier", &self.verifier),
                    ("redirect_uri", redirect),
                    ("resource", RESOURCE),
                ],
            )?,
            "ChatGPT token exchange",
        )
        .await?;
        let tokens = grant.into_tokens(None)?;
        let identity =
            verify_identity(http, &tokens.id_token, client_id, Some(&self.nonce)).await?;
        anyhow::ensure!(
            previous.is_none_or(|account| account.subject == identity.sub),
            "ChatGPT returned a different account. Use Add Account to connect it."
        );
        Ok(Account {
            client_id: client_id.to_owned(),
            subject: identity.sub,
            email: identity.email,
            tokens: Some(tokens),
        })
    }
}

fn issued_client_id<'a>(selected: Option<&'a str>, returned: Option<&'a str>) -> Result<&'a str> {
    if let Some(selected) = selected {
        anyhow::ensure!(
            returned.is_none_or(|returned| returned == selected),
            "ChatGPT returned a different registration"
        );
        return Ok(selected);
    }
    returned
        .filter(|id| id.starts_with("oaiapp_") && id.len() > "oaiapp_".len())
        .context("ChatGPT did not return an issued client ID")
}

async fn fetch_catalog(http: &dyn HttpClient, access_token: &str) -> Result<Vec<AccountModel>> {
    let mut request = get_request(MODELS)?;
    request.headers_mut().insert(
        http_client::http::header::AUTHORIZATION,
        format!("Bearer {access_token}")
            .parse()
            .context("Invalid ChatGPT bearer token")?,
    );
    let catalog: Catalog = read_json(http, request, "ChatGPT model refresh").await?;
    let mut seen = std::collections::HashSet::new();
    Ok(catalog
        .models
        .into_iter()
        .filter(|model| {
            model.visibility == "list"
                && !model.slug.trim().is_empty()
                && !model.display_name.trim().is_empty()
                && seen.insert(model.slug.clone())
        })
        .collect())
}

#[derive(Clone)]
pub struct AccountProfile {
    pub id: String,
    pub label: String,
    pub selected: bool,
    pub connected: bool,
    pub can_use_plan: bool,
}

struct PlanUsage {
    path: PathBuf,
    accounts: Accounts,
    storage_available: bool,
    storage_lock: Option<std::fs::File>,
    operation: Option<Task<()>>,
    catalog_task: Option<Task<()>>,
    token_lock: Arc<smol::lock::Mutex<()>>,
    generation: u64,
    session: (smol::channel::Sender<()>, smol::channel::Receiver<()>),
    models: Vec<AccountModel>,
    error: Option<SharedString>,
    catalog_error: Option<SharedString>,
}

impl PlanUsage {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            accounts: Accounts::default(),
            storage_available: false,
            storage_lock: None,
            operation: None,
            catalog_task: None,
            token_lock: Arc::default(),
            generation: 0,
            session: smol::channel::bounded(1),
            models: Vec::new(),
            error: None,
            catalog_error: None,
        }
    }

    fn load(&mut self) -> Result<()> {
        if self.storage_available {
            return Ok(());
        }
        let directory = self
            .path
            .parent()
            .context("Missing ChatGPT credentials directory")?;
        std::fs::create_dir_all(directory)?;
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.path.with_extension("lock"))?;
        lock.try_lock()
            .context("Saved ChatGPT accounts are in use by another Copper process")?;
        self.accounts = read_accounts(&self.path)?;
        self.storage_lock = Some(lock);
        self.storage_available = true;
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        if self.path.try_exists()? {
            self.load()?;
        }
        Ok(())
    }

    fn persist(&mut self, accounts: Accounts) -> Result<()> {
        anyhow::ensure!(
            self.storage_available,
            "Saved ChatGPT accounts are unavailable"
        );
        save_accounts(&self.path, &accounts)?;
        self.accounts = accounts;
        Ok(())
    }

    fn invalidate_catalog(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if let Some(task) = self.catalog_task.take() {
            // A refresh can already have rotated the token upstream. Let it
            // persist the rotation and discard its catalog by generation.
            task.detach();
        }
        self.session.0.close();
        self.session = smol::channel::bounded(1);
        self.catalog_error = None;
        self.models.clear();
    }
}

pub struct State {
    plan: PlanUsage,
    http_client: Arc<dyn HttpClient>,
    restoring: bool,
}

impl State {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut Context<Self>) -> Self {
        Self::with_path(
            http_client,
            paths::data_dir().join("chatgpt-plan-accounts.json"),
            cx,
        )
    }

    fn with_path(http_client: Arc<dyn HttpClient>, path: PathBuf, cx: &mut Context<Self>) -> Self {
        let state = cx.entity().downgrade();
        cx.defer(move |cx| {
            state
                .update(cx, |state, cx| {
                    if let Err(error) = state.plan.restore() {
                        state.plan.error =
                            Some(format!("Could not restore your ChatGPT sign-in. {error}").into());
                    }
                    state.restoring = false;
                    state.refresh_chatgpt_catalog(cx);
                    cx.notify();
                })
                .log_err();
        });
        Self {
            plan: PlanUsage::new(path),
            http_client,
            restoring: true,
        }
    }

    pub fn account_models(entity: &Entity<Self>, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        entity.read(cx).plan_models_for_state(entity)
    }

    pub fn has_chatgpt_plan_permission(&self) -> bool {
        self.plan
            .accounts
            .active()
            .and_then(|account| account.tokens.as_ref())
            .is_some_and(Tokens::can_use_plan)
    }

    pub fn chatgpt_profiles(&self) -> Vec<AccountProfile> {
        self.plan
            .accounts
            .profiles
            .iter()
            .enumerate()
            .map(|(index, account)| AccountProfile {
                id: account.client_id.clone(),
                label: format!(
                    "{} (Account {})",
                    account.email.as_deref().unwrap_or("ChatGPT"),
                    index + 1
                ),
                selected: self.plan.accounts.active.as_ref() == Some(&account.client_id),
                connected: account.tokens.is_some(),
                can_use_plan: account.tokens.as_ref().is_some_and(Tokens::can_use_plan),
            })
            .collect()
    }

    pub fn chatgpt_plan_busy(&self) -> bool {
        self.plan.operation.is_some() || self.restoring
    }

    pub fn chatgpt_catalog_loading(&self) -> bool {
        self.plan.catalog_task.is_some()
    }

    pub fn chatgpt_plan_error(&self) -> Option<SharedString> {
        self.plan.error.clone()
    }

    pub fn chatgpt_catalog_error(&self) -> Option<SharedString> {
        self.plan.catalog_error.clone()
    }

    pub fn has_chatgpt_catalog(&self) -> bool {
        !self.plan.models.is_empty()
    }

    pub fn select_chatgpt_account(&mut self, client_id: String, cx: &mut Context<Self>) {
        if self.chatgpt_plan_busy() {
            return;
        }
        let Some(account) = self.plan.accounts.account(&client_id) else {
            return;
        };
        if account.tokens.is_none()
            || (self.plan.accounts.active.as_deref() == Some(&client_id)
                && !self.has_chatgpt_plan_permission())
        {
            self.sign_in_chatgpt_plan(Some(client_id), cx);
            return;
        }
        self.activate_chatgpt_account(client_id, cx);
    }

    fn activate_chatgpt_account(&mut self, client_id: String, cx: &mut Context<Self>) {
        let mut accounts = self.plan.accounts.clone();
        accounts.active = Some(client_id);
        match self.plan.persist(accounts) {
            Ok(()) => {
                self.plan.invalidate_catalog();
                self.plan.error = None;
                self.refresh_chatgpt_catalog(cx);
            }
            Err(error) => self.plan.error = Some(error.to_string().into()),
        }
        cx.notify();
    }

    pub fn sign_in_chatgpt_plan(&mut self, selected: Option<String>, cx: &mut Context<Self>) {
        if self.chatgpt_plan_busy() {
            return;
        }
        if let Err(error) = self.plan.load() {
            self.plan.error = Some(error.to_string().into());
            cx.notify();
            return;
        }

        self.plan.error = None;
        let http = self.http_client.clone();
        let lock = self.plan.token_lock.clone();
        self.plan.operation = Some(cx.spawn(async move |state, cx| {
            let result = async {
                let accounts = state.read_with(cx, |state, _| state.plan.accounts.clone())?;
                let selected = selected.as_deref().or(accounts.pending_client.as_deref());
                let previous = selected.and_then(|id| accounts.account(id)).cloned();
                // Persist the host ID before the browser can register it.
                state.update(cx, |state, _| state.plan.persist(accounts.clone()))??;

                let authorization = Authorization::new();
                let (redirect, callback) =
                    oauth_callback_server::start_oauth_callback_server_with_state(
                        oauth_callback_server::OAuthCallbackServerConfig {
                            path: "/auth/callback",
                            ..Default::default()
                        },
                        authorization.state.clone(),
                    )?;
                let url = authorization.url(&redirect, &accounts, selected)?;
                cx.update(|cx| cx.open_url(url.as_str()));
                let callback = callback
                    .await
                    .context("ChatGPT sign-in timed out or was cancelled")??;
                let client_id =
                    issued_client_id(selected, callback.client_id.as_deref())?.to_owned();
                if previous.is_none() {
                    state.update(cx, |state, _| {
                        let mut accounts = state.plan.accounts.clone();
                        accounts.pending_client = Some(client_id.clone());
                        state.plan.persist(accounts)
                    })??;
                }
                let _guard = lock.lock().await;
                let account = authorization
                    .exchange(
                        http.as_ref(),
                        &callback.code,
                        &client_id,
                        &redirect,
                        previous.as_ref(),
                    )
                    .await?;
                state.update(cx, |state, _| {
                    state.install_chatgpt_account(account, previous.is_none())
                })??;
                anyhow::Ok(())
            }
            .await;
            state
                .update(cx, |state, cx| {
                    state.plan.operation = None;
                    state.plan.error = result.err().map(|error| error.to_string().into());
                    state.refresh_chatgpt_catalog(cx);
                    cx.notify();
                })
                .log_err();
        }));
        cx.notify();
    }

    fn install_chatgpt_account(&mut self, account: Account, registered: bool) -> Result<()> {
        let mut accounts = self.plan.accounts.clone();
        let client_id = account.client_id.clone();
        if let Some(existing) = accounts
            .profiles
            .iter_mut()
            .find(|existing| existing.client_id == client_id)
        {
            anyhow::ensure!(
                existing.subject == account.subject,
                "ChatGPT returned a different account"
            );
            *existing = account;
        } else {
            accounts.profiles.push(account);
        }
        if registered {
            accounts.pending_client = None;
        }
        accounts.active = Some(client_id);
        self.plan.persist(accounts)?;
        self.plan.invalidate_catalog();
        Ok(())
    }

    pub fn sign_out_chatgpt_plan(&mut self, cx: &mut Context<Self>) {
        if self.chatgpt_plan_busy() {
            return;
        }
        let Some(client_id) = self.plan.accounts.active.clone() else {
            return;
        };
        self.plan.invalidate_catalog();
        self.plan.error = None;
        let http = self.http_client.clone();
        let lock = self.plan.token_lock.clone();
        self.plan.operation = Some(cx.spawn(async move |state, cx| {
            let result = async {
                let _guard = lock.lock().await;
                let mut accounts = state.read_with(cx, |state, _| state.plan.accounts.clone())?;
                let tokens = accounts
                    .profiles
                    .iter_mut()
                    .find(|account| account.client_id == client_id)
                    .and_then(|account| account.tokens.take());
                accounts.active = None;
                let persisted = state.update(cx, |state, _| {
                    let result = state.plan.persist(accounts.clone());
                    // Stop using this session even if local storage becomes unavailable.
                    state.plan.accounts = accounts;
                    result
                })?;
                let revoked = if let Some(tokens) = tokens {
                    revoke_session(http.as_ref(), &client_id, &tokens.refresh_token).await
                } else {
                    Ok(())
                };
                persisted.context("Could not clear the saved ChatGPT session")?;
                revoked
            }
            .await;
            state
                .update(cx, |state, cx| {
                    state.plan.operation = None;
                    state.plan.error = result.err().map(|error| error.to_string().into());
                    cx.notify();
                })
                .log_err();
        }));
        cx.notify();
    }

    pub fn refresh_chatgpt_catalog(&mut self, cx: &mut Context<Self>) {
        if !self.has_chatgpt_plan_permission()
            || self.plan.operation.is_some()
            || self.plan.catalog_task.is_some()
        {
            return;
        }
        let Some(client_id) = self.plan.accounts.active.clone() else {
            return;
        };
        let generation = self.plan.generation;
        let http = self.http_client.clone();
        let lock = self.plan.token_lock.clone();
        self.plan.catalog_error = None;
        self.plan.catalog_task = Some(cx.spawn(async move |state, cx| {
            let result = async {
                let token =
                    active_access_token(&state, http.as_ref(), &lock, &client_id, generation, cx)
                        .await?;
                fetch_catalog(http.as_ref(), &token).await
            }
            .await;
            state
                .update(cx, |state, cx| {
                    if state.plan.generation != generation {
                        return;
                    }
                    state.plan.catalog_task = None;
                    match result {
                        Ok(models) => state.plan.models = models,
                        Err(error) => {
                            if error.downcast_ref::<RequestStatus>().is_some_and(|error| {
                                matches!(error.0, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
                            }) {
                                state.plan.models.clear();
                            }
                            state.plan.catalog_error = Some(error.to_string().into());
                        }
                    }
                    cx.notify();
                })
                .log_err();
        }));
        cx.notify();
    }

    fn plan_models_for_state(&self, entity: &Entity<Self>) -> Vec<Arc<dyn LanguageModel>> {
        let Some(client_id) = self.plan.accounts.active.as_ref() else {
            return Vec::new();
        };
        if !self.has_chatgpt_plan_permission() {
            return Vec::new();
        }
        self.plan
            .models
            .iter()
            .map(|model| {
                Arc::new(PlanLanguageModel {
                    model: model.clone(),
                    client_id: client_id.clone(),
                    generation: self.plan.generation,
                    session: self.plan.session.1.clone(),
                    state: entity.downgrade(),
                    http: self.http_client.clone(),
                    lock: self.plan.token_lock.clone(),
                }) as Arc<dyn LanguageModel>
            })
            .collect()
    }
}

#[derive(Deserialize)]
struct OpenIdConfiguration {
    revocation_endpoint: url::Url,
}

async fn revoke_session(http: &dyn HttpClient, client_id: &str, token: &str) -> Result<()> {
    let result = async {
        let configuration: OpenIdConfiguration =
            read_json(http, get_request(DISCOVERY)?, "ChatGPT sign-out discovery").await?;
        let endpoint = configuration.revocation_endpoint;
        anyhow::ensure!(
            endpoint.origin() == url::Url::parse(ISSUER)?.origin()
                && endpoint.username().is_empty()
                && endpoint.password().is_none(),
            "Unexpected ChatGPT revocation endpoint"
        );
        for attempt in 0..2 {
            if attempt > 0 {
                smol::Timer::after(Duration::from_millis(350)).await;
            }
            let response = http
                .send(form_request(
                    endpoint.as_str(),
                    &[
                        ("token", token),
                        ("token_type_hint", "refresh_token"),
                        ("client_id", client_id),
                    ],
                )?)
                .await;
            match response {
                Ok(response) if response.status() == StatusCode::OK => return Ok(()),
                Ok(response) if !response.status().is_server_error() => break,
                _ => {}
            }
        }
        anyhow::bail!("ChatGPT did not confirm revocation");
    }
    .await;
    result.context("Signed out on this device. ChatGPT could not confirm revocation. You can disconnect Copper in ChatGPT Settings.")
}

async fn active_access_token(
    state: &WeakEntity<State>,
    http: &dyn HttpClient,
    lock: &smol::lock::Mutex<()>,
    client_id: &str,
    generation: u64,
    cx: &mut AsyncApp,
) -> Result<String> {
    let _guard = lock.lock().await;
    let account = state.read_with(cx, |state, _| {
        anyhow::ensure!(
            state.plan.generation == generation
                && state.plan.accounts.active.as_deref() == Some(client_id),
            "The selected ChatGPT account changed. Select a model for the active account."
        );
        state
            .plan
            .accounts
            .active()
            .cloned()
            .context("No ChatGPT account is active")
    })??;
    let tokens = account
        .tokens
        .context("Sign in to the selected ChatGPT account")?;
    anyhow::ensure!(
        tokens.can_use_plan(),
        "Enable ChatGPT plan usage for the selected account"
    );
    if now_ms().saturating_add(TOKEN_REFRESH_BUFFER_MS) < tokens.expires_at_ms {
        return Ok(tokens.access_token);
    }

    let response = http
        .send(form_request(
            TOKEN,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", client_id),
                ("refresh_token", &tokens.refresh_token),
                ("resource", RESOURCE),
            ],
        )?)
        .await
        .context("ChatGPT token refresh could not connect")?;
    let status = response.status();
    let mut bytes = Vec::new();
    response
        .into_body()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut bytes)
        .await?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_RESPONSE_BYTES,
        "ChatGPT token response was too large"
    );
    if !status.is_success() {
        let error_code = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|body| body.get("error")?.as_str().map(str::to_owned));
        if matches!(
            error_code.as_deref(),
            Some("invalid_grant" | "invalid_token" | "invalid_client")
        ) {
            state.update(cx, |state, cx| {
                let mut accounts = state.plan.accounts.clone();
                if let Some(account) = accounts
                    .profiles
                    .iter_mut()
                    .find(|account| account.client_id == client_id)
                {
                    account.tokens = None;
                }
                let result = state.plan.persist(accounts);
                if state.plan.accounts.active.as_deref() == Some(client_id) {
                    state.plan.models.clear();
                }
                cx.notify();
                result
            })??;
            anyhow::bail!("ChatGPT session expired. Reconnect this account.");
        }
        anyhow::bail!("ChatGPT token refresh failed (HTTP {status})");
    }
    let grant: TokenGrant = serde_json::from_slice(&bytes)
        .context("ChatGPT token refresh returned an invalid response")?;
    if let Some(id_token) = grant.id_token.as_ref() {
        let identity = verify_identity(http, id_token, client_id, None).await?;
        anyhow::ensure!(
            identity.sub == account.subject,
            "ChatGPT refreshed a different account"
        );
    }
    let refreshed = grant.into_tokens(Some(&tokens))?;
    state.update(cx, |state, cx| {
        let mut accounts = state.plan.accounts.clone();
        let account = accounts
            .profiles
            .iter_mut()
            .find(|account| account.client_id == client_id)
            .context("ChatGPT account was removed")?;
        account.tokens = Some(refreshed.clone());
        state.plan.persist(accounts)?;
        if !refreshed.can_use_plan() && state.plan.accounts.active.as_deref() == Some(client_id) {
            state.plan.models.clear();
            cx.notify();
        }
        anyhow::ensure!(
            state.plan.generation == generation,
            "The selected ChatGPT account changed"
        );
        anyhow::ensure!(
            refreshed.can_use_plan(),
            "ChatGPT plan usage is no longer authorized"
        );
        Ok(refreshed.access_token)
    })?
}

struct PlanLanguageModel {
    model: AccountModel,
    client_id: String,
    generation: u64,
    session: smol::channel::Receiver<()>,
    state: WeakEntity<State>,
    http: Arc<dyn HttpClient>,
    lock: Arc<smol::lock::Mutex<()>>,
}

impl LanguageModel for PlanLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.model.slug.clone().into()
    }

    fn name(&self) -> LanguageModelName {
        self.model.display_name.clone().into()
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn telemetry_id(&self) -> String {
        "openai-subscribed/chatgpt-plan".into()
    }

    fn max_token_count(&self) -> u64 {
        self.model
            .context_window
            .filter(|size| *size > 0)
            .unwrap_or(32_768)
    }

    fn supports_images(&self) -> bool {
        self.model
            .input_modalities
            .iter()
            .any(|modality| modality == "image")
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        true
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn stream_completion(
        &self,
        mut request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        request.speed = None;
        request.temperature = None;
        request.compact_at_tokens = None;
        // Compaction items are encrypted for their originating backend. A legacy
        // Codex transcript can have the same provider ID as this connection.
        let mut request = match into_open_ai_response(
            request,
            &self.model.slug,
            false,
            false,
            None,
            None,
            false,
            &LanguageModelProviderId::new("openai-subscribed-plan"),
        ) {
            Ok(request) => request,
            Err(error) => return futures::future::ready(Err(error.into())).boxed(),
        };
        request.store = Some(false);
        request.stream = true;
        if !request.tools.is_empty() {
            request.tools = vec![open_ai::responses::ToolDefinition::Namespace {
                name: "copper".to_owned(),
                description: "Tools run by the Copper agent".to_owned(),
                tools: std::mem::take(&mut request.tools),
            }];
        }
        let state = self.state.clone();
        let http = self.http.clone();
        let lock = self.lock.clone();
        let client_id = self.client_id.clone();
        let generation = self.generation;
        let session = self.session.clone();
        cx.spawn(async move |cx| {
            state.read_with(cx, |state, _| {
                anyhow::ensure!(state.plan.models.iter().any(|model| model.slug == request.model),
                    "This model is no longer available. Refresh the ChatGPT model list and choose a model.");
                anyhow::Ok(())
            })??;
            let access_token = active_access_token(&state, http.as_ref(), &lock, &client_id, generation, cx).await?;
            let headers = CustomHeaders::default();
            let provider_name = PROVIDER_NAME;
            let request = stream_response(http.as_ref(), provider_name.0.as_str(), RESOURCE, &access_token, request, &headers);
            let cancelled = session.recv();
            futures::pin_mut!(request, cancelled);
            let stream = match futures::future::select(cancelled, request).await {
                futures::future::Either::Left(_) => return Err(anyhow!("The selected ChatGPT account changed").into()),
                futures::future::Either::Right((result, _)) => result.map_err(LanguageModelCompletionError::from)?,
            };
            Ok(OpenAiResponseEventMapper::new(PROVIDER_ID).map_stream(require_terminal_event(stream, session.clone())).boxed())
        }).boxed()
    }
}

fn require_terminal_event(
    stream: futures::stream::BoxStream<'static, Result<StreamEvent>>,
    session: smol::channel::Receiver<()>,
) -> futures::stream::BoxStream<'static, Result<StreamEvent>> {
    futures::stream::unfold(
        (stream, session, false),
        |(mut stream, session, terminal)| async move {
            if terminal {
                return None;
            }
            let event = match futures::future::select(Box::pin(session.recv()), stream.next()).await
            {
                futures::future::Either::Left(_) => {
                    Err(anyhow!("The selected ChatGPT account changed"))
                }
                futures::future::Either::Right((event, _)) => event.unwrap_or_else(|| {
                    Err(anyhow!(
                        "ChatGPT response stream ended before a terminal event"
                    ))
                }),
            };
            let terminal = matches!(
                &event,
                Ok(StreamEvent::Completed { .. }
                    | StreamEvent::Incomplete { .. }
                    | StreamEvent::Failed { .. }
                    | StreamEvent::Error { .. }
                    | StreamEvent::GenericError { .. })
                    | Err(_)
            );
            Some((event, (stream, session, terminal)))
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AppContext as _, TestAppContext};
    use http_client::{FakeHttpClient, Response};
    use jsonwebtoken::{EncodingKey, Header};
    use parking_lot::Mutex;
    use rsa::{RsaPrivateKey, pkcs1::EncodeRsaPrivateKey as _, traits::PublicKeyParts as _};
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[gpui::test]
    async fn restores_active_registration_even_when_legacy_was_selected(cx: &mut TestAppContext) {
        let directory = tempfile::tempdir().expect("private credential directory");
        let path = directory.path().join("accounts.json");
        let mut accounts = Accounts::default();
        accounts.active = Some("oaiapp_fixture".into());
        accounts.profiles.push(Account {
            client_id: "oaiapp_fixture".into(),
            subject: "fixture-subject".into(),
            email: None,
            tokens: Some(Tokens {
                access_token: "fixture-token".into(),
                refresh_token: "fixture-refresh".into(),
                id_token: "fixture-identity".into(),
                expires_at_ms: u64::MAX,
                scope: SCOPE.into(),
            }),
        });
        let mut saved = serde_json::to_value(&accounts).expect("saved account");
        saved["use_plan"] = json!(false);
        std::fs::write(&path, saved.to_string()).expect("write existing account file");
        let fixture = FakeHttpClient::create(|request| async move {
            assert_eq!(request.uri(), MODELS);
            assert_eq!(request.headers()["Authorization"], "Bearer fixture-token");
            Ok(Response::new(AsyncBody::from(catalog("restored-model"))))
        });
        let state = cx.new(|cx| State::with_path(fixture, path, cx));
        cx.run_until_parked();
        assert_eq!(model_ids(&state, cx), ["restored-model"]);
    }

    #[gpui::test]
    async fn browser_sign_in_and_account_catalog_lifecycle(cx: &mut TestAppContext) {
        use smol::io::AsyncWriteExt as _;

        cx.executor().allow_parking();
        let directory = tempfile::tempdir().expect("private credential directory");
        let path = directory.path().join("accounts.json");
        let key =
            RsaPrivateKey::new(&mut rsa::rand_core::OsRng, 2048).expect("fixture signing key");
        let jwks = json!({"keys": [{
            "kty": "RSA", "use": "sig", "kid": "fixture", "alg": "RS256",
            "n": URL_SAFE_NO_PAD.encode(key.n().to_bytes_be()),
            "e": URL_SAFE_NO_PAD.encode(key.e().to_bytes_be()),
        }]})
        .to_string();
        let key = EncodingKey::from_rsa_der(key.to_pkcs1_der().expect("DER").as_bytes());
        let authorization = Arc::new(Mutex::new(None::<url::Url>));
        let model_requests = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(AtomicUsize::new(0));
        let revocations = Arc::new(AtomicUsize::new(0));
        let fixture = FakeHttpClient::create({
            let authorization = authorization.clone();
            let model_requests = model_requests.clone();
            let responses = responses.clone();
            let revocations = revocations.clone();
            move |request| {
                let key = key.clone();
                let jwks = jwks.clone();
                let authorization = authorization.clone();
                let model_requests = model_requests.clone();
                let responses = responses.clone();
                let revocations = revocations.clone();
                async move {
                    let uri = request.uri().to_string();
                    let bearer = request.headers().get("Authorization").cloned();
                    let mut body = String::new();
                    request.into_body().read_to_string(&mut body).await?;
                    let fields = form_urlencoded::parse(body.as_bytes())
                        .collect::<std::collections::HashMap<_, _>>();
                    let (status, body) = match uri.as_str() {
                        JWKS => (200, jwks),
                        TOKEN if fields.get("grant_type").map(|value| value.as_ref()) == Some("authorization_code") => {
                            let url = authorization.lock().clone().expect("opened authorization URL");
                            let query = url.query_pairs().collect::<std::collections::HashMap<_, _>>();
                            assert_eq!(fields.get("client_id").map(|value| value.as_ref()), Some("oaiapp_fixture"));
                            assert_eq!(fields.get("resource").map(|value| value.as_ref()), Some(RESOURCE));
                            assert_eq!(fields.get("redirect_uri"), query.get("redirect_uri"));
                            assert_eq!(fields.get("code").map(|value| value.as_ref()), Some("fixture-code"));
                            assert_eq!(URL_SAFE_NO_PAD.encode(Sha256::digest(fields["code_verifier"].as_bytes())), query["code_challenge"]);
                            let claims = json!({
                                "sub": "first-subject", "nonce": query["nonce"], "aud": "oaiapp_fixture",
                                "iss": ISSUER, "exp": now_ms() / 1000 + 3600, "email": "same@example.test",
                            });
                            let mut header = Header::new(Algorithm::RS256);
                            header.kid = Some("fixture".into());
                            let token = jsonwebtoken::encode(&header, &claims, &key).expect("sign fixture");
                            (200, json!({"access_token": "first-token", "refresh_token": "first-refresh",
                                "token_type": "Bearer", "expires_in": 3600, "id_token": token, "scope": SCOPE}).to_string())
                        }
                        TOKEN => {
                            assert_eq!(fields.get("client_id").map(|value| value.as_ref()), Some("oaiapp_fixture"));
                            assert_eq!(fields.get("grant_type").map(|value| value.as_ref()), Some("refresh_token"));
                            assert_eq!(fields.get("refresh_token").map(|value| value.as_ref()), Some("first-refresh"));
                            assert_eq!(fields.get("resource").map(|value| value.as_ref()), Some(RESOURCE));
                            assert!(!fields.contains_key("scope"));
                            (200, json!({"access_token": "first-token", "refresh_token": "rotated-refresh",
                                "token_type": "Bearer", "expires_in": 3600}).to_string())
                        }
                        MODELS => {
                            let count = model_requests.fetch_add(1, Ordering::SeqCst);
                            match bearer.as_ref().and_then(|header| header.to_str().ok()) {
                                Some("Bearer first-token") if count == 1 => (503, String::new()),
                                Some("Bearer first-token") if count == 2 => (200, catalog("newly-eligible")),
                                Some("Bearer first-token") => (200, catalog("first-account-model")),
                                Some("Bearer second-token") => (200, catalog("second-account-model")),
                                _ => panic!("Wrong token for catalog"),
                            }
                        }
                        "https://api.openai.com/v1/responses" => {
                            assert_eq!(bearer.as_ref().and_then(|header| header.to_str().ok()), Some("Bearer first-token"));
                            let request: Value = serde_json::from_str(&body)?;
                            assert_eq!(request["model"], "newly-eligible");
                            assert_eq!(request["stream"], true);
                            assert_eq!(request["store"], false);
                            assert!(request["input"].is_array());
                            assert!(request.get("temperature").is_none());
                            assert_eq!(request["tools"][0]["type"], "namespace");
                            let attempt = responses.fetch_add(1, Ordering::SeqCst);
                            let body = if attempt == 0 {
                                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                            } else {
                                "data: [DONE]\n\n"
                            };
                            (200, body.to_owned())
                        }
                        DISCOVERY => (200, json!({"revocation_endpoint": "https://auth.openai.com/revoke-fixture"}).to_string()),
                        "https://auth.openai.com/revoke-fixture" => {
                            assert_eq!(fields.get("client_id").map(|value| value.as_ref()), Some("oaiapp_second"));
                            assert_eq!(fields.get("token").map(|value| value.as_ref()), Some("second-refresh"));
                            assert_eq!(fields.get("token_type_hint").map(|value| value.as_ref()), Some("refresh_token"));
                            revocations.fetch_add(1, Ordering::SeqCst);
                            (200, String::new())
                        }
                        _ => panic!("Unexpected request endpoint {uri}"),
                    };
                    Ok(Response::builder()
                        .status(status)
                        .body(AsyncBody::from(body))?)
                }
            }
        });
        let state = cx.new(|cx| State::with_path(fixture, path.clone(), cx));
        cx.run_until_parked();
        state.update(cx, |state, cx| state.sign_in_chatgpt_plan(None, cx));
        cx.run_until_parked();
        let opened = cx.opened_url().unwrap_or_else(|| {
            let error = state.read_with(cx, |state, _| state.chatgpt_plan_error());
            panic!("authorization URL was not opened: {error:?}");
        });
        let url = url::Url::parse(&opened).expect("valid authorize URL");
        assert_eq!(url.path(), "/api/accounts/authorize");
        let query = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(query["client_id"], DYNAMIC_CLIENT);
        assert_eq!(query["agent_name_hint"], paths::APP_NAME);
        assert_eq!(query["scope"], SCOPE);
        assert_eq!(query["resource"], RESOURCE);
        assert_eq!(
            query["ext_agent_host_id"],
            read_accounts(&path).expect("host ID saved").host_id
        );
        *authorization.lock() = Some(url.clone());
        let redirect = url::Url::parse(&query["redirect_uri"]).expect("callback URL");
        assert_eq!(redirect.host_str(), Some("127.0.0.1"));
        let addr = (
            std::net::Ipv4Addr::LOCALHOST,
            redirect.port().expect("callback port"),
        );
        let path_and_query = format!(
            "{}?code=fixture-code&state={}&client_id=oaiapp_fixture",
            redirect.path(),
            query["state"]
        );
        cx.background_spawn(async move {
            let mut stream = smol::net::TcpStream::connect(addr)
                .await
                .expect("connect loopback");
            let request = format!(
                "GET {path_and_query} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
                addr.0, addr.1
            );
            stream
                .write_all(request.as_bytes())
                .await
                .expect("callback");
            let mut response = String::new();
            stream
                .read_to_string(&mut response)
                .await
                .expect("callback response");
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        })
        .await;
        cx.condition(&state, |state, _| {
            !state.chatgpt_plan_busy() && !state.chatgpt_catalog_loading()
        })
        .await;
        assert_eq!(model_ids(&state, cx), ["first-account-model"]);

        state.update(cx, |state, cx| state.refresh_chatgpt_catalog(cx));
        cx.run_until_parked();
        assert_eq!(model_ids(&state, cx), ["first-account-model"]);
        assert!(state.read_with(cx, |state, _| state.chatgpt_catalog_error().is_some()));

        state.update(cx, |state, cx| {
            let mut accounts = state.plan.accounts.clone();
            accounts.profiles[0]
                .tokens
                .as_mut()
                .expect("first account token")
                .expires_at_ms = 0;
            state.plan.persist(accounts).expect("expire test token");
            state.refresh_chatgpt_catalog(cx);
        });
        cx.run_until_parked();
        let accounts = read_accounts(&path).expect("saved refreshed token");
        assert_eq!(
            accounts.profiles[0]
                .tokens
                .as_ref()
                .expect("refreshed token")
                .refresh_token,
            "rotated-refresh"
        );
        assert_eq!(model_ids(&state, cx), ["newly-eligible"]);
        let model = cx.update(|cx| State::account_models(&state, cx).remove(0));
        assert_eq!(model.name().0.as_str(), "Available newly-eligible");
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("Say hello".into())],
                cache: false,
                reasoning_details: None,
            }],
            temperature: Some(0.7),
            tools: vec![language_model::LanguageModelRequestTool::function(
                "read_file".into(),
                "Read a local file".into(),
                json!({"type":"object"}),
                false,
            )],
            ..Default::default()
        };
        let events = model
            .stream_completion(request.clone(), &cx.to_async())
            .await
            .expect("Responses request")
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().any(
            |event| matches!(event, Ok(LanguageModelCompletionEvent::Text(text)) if text == "Hello")
        ));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Ok(LanguageModelCompletionEvent::Stop(_))))
        );
        let interrupted = model
            .stream_completion(request.clone(), &cx.to_async())
            .await
            .expect("Responses request")
            .collect::<Vec<_>>()
            .await;
        assert!(interrupted.iter().any(|event| {
            event
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("ended before a terminal event"))
        }));

        state.update(cx, |state, cx| {
            let mut accounts = state.plan.accounts.clone();
            accounts.profiles.push(Account {
                client_id: "oaiapp_second".into(),
                subject: "second-subject".into(),
                email: Some("same@example.test".into()),
                tokens: Some(Tokens {
                    access_token: "second-token".into(),
                    refresh_token: "second-refresh".into(),
                    id_token: "fixture".into(),
                    expires_at_ms: u64::MAX,
                    scope: SCOPE.into(),
                }),
            });
            state.plan.persist(accounts).expect("save second account");
            state.select_chatgpt_account("oaiapp_second".into(), cx);
            assert!(state.plan.models.is_empty());
        });
        cx.run_until_parked();
        assert_eq!(model_ids(&state, cx), ["second-account-model"]);
        assert!(
            model
                .stream_completion(request, &cx.to_async())
                .await
                .is_err()
        );
        assert_eq!(responses.load(Ordering::SeqCst), 2);

        state.update(cx, |state, cx| {
            state.sign_out_chatgpt_plan(cx);
            assert!(state.plan.models.is_empty());
        });
        cx.run_until_parked();
        assert!(model_ids(&state, cx).is_empty());
        let accounts = read_accounts(&path).expect("saved signed-out accounts");
        assert!(accounts.active.is_none());
        assert!(
            accounts
                .account("oaiapp_second")
                .expect("registration retained")
                .tokens
                .is_none()
        );
        assert!(
            accounts
                .account("oaiapp_fixture")
                .expect("other registration retained")
                .tokens
                .is_some()
        );
        assert_eq!(revocations.load(Ordering::SeqCst), 1);
    }

    fn model_ids(state: &Entity<State>, cx: &mut TestAppContext) -> Vec<String> {
        cx.update(|cx| {
            State::account_models(state, cx)
                .into_iter()
                .map(|model| model.id().0.to_string())
                .collect()
        })
    }

    fn catalog(slug: &str) -> String {
        json!({"models": [
            {"slug":"hidden", "display_name":"Hidden", "visibility":"hide"},
            {"slug":slug, "display_name":format!("Available {slug}"), "visibility":"list"},
            {"slug":slug, "display_name":"Duplicate", "visibility":"list"}
        ]})
        .to_string()
    }
}
