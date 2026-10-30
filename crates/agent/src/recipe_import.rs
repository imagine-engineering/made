use anyhow::{Context as _, Result, ensure};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient, HttpRequestExt as _, RedirectPolicy, Request, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{net::SocketAddr, sync::Arc, time::Duration};

use crate::recipes::Recipe;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeImportRequest {
    pub url: String,
    #[serde(default)]
    pub github_revision: Option<String>,
    #[serde(default)]
    pub github_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeProvenance {
    pub request: RecipeImportRequest,
    pub commit: Option<String>,
    pub sha256: String,
}

pub const MAX_RECIPE_LINK_BYTES: usize = 64 * 1024;
pub const MAX_EMBEDDED_RECIPE_BYTES: usize = 32 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecipeLink {
    Embedded { yaml: String },
    Source(RecipeImportRequest),
}

impl RecipeLink {
    pub fn encode(&self) -> Result<String> {
        let mut url = Url::parse(&product_flavor::url("recipe"))?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("v", "1");
            match self {
                Self::Embedded { yaml } => {
                    ensure!(
                        yaml.len() <= MAX_EMBEDDED_RECIPE_BYTES,
                        "Embedded recipe links support at most 32 KiB; use a reviewed HTTPS source for larger recipes"
                    );
                    Recipe::from_yaml(yaml)?;
                    query.append_pair("data", &URL_SAFE_NO_PAD.encode(yaml.as_bytes()));
                }
                Self::Source(request) => {
                    validate_request(request)?;
                    query.append_pair("source", &request.url);
                    if let Some(revision) = &request.github_revision {
                        query.append_pair("revision", revision);
                    }
                    if let Some(path) = &request.github_path {
                        query.append_pair("path", path);
                    }
                }
            }
        }
        let encoded = url.to_string();
        ensure!(
            encoded.len() <= MAX_RECIPE_LINK_BYTES,
            "Recipe link exceeds 64 KiB"
        );
        Ok(encoded)
    }

    pub fn decode(value: &str) -> Result<Self> {
        ensure!(
            value.len() <= MAX_RECIPE_LINK_BYTES,
            "Recipe link exceeds 64 KiB"
        );
        ensure!(
            value.trim() == value && !value.chars().any(char::is_control),
            "Recipe link contains ambiguous characters"
        );
        let normalized = product_flavor::normalize_url(value);
        let url = Url::parse(&normalized).context("Invalid recipe link")?;
        ensure!(
            url.scheme() == "zed"
                && url.host_str() == Some("recipe")
                && url.path().is_empty()
                && url.port().is_none()
                && url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none(),
            "Invalid recipe link route"
        );
        let mut fields = std::collections::BTreeMap::new();
        for (key, value) in url.query_pairs() {
            ensure!(
                matches!(key.as_ref(), "v" | "data" | "source" | "revision" | "path"),
                "Unsupported recipe link field"
            );
            ensure!(
                fields
                    .insert(key.into_owned(), value.into_owned())
                    .is_none(),
                "Duplicate recipe link field"
            );
        }
        ensure!(
            fields.remove("v").as_deref() == Some("1"),
            "Unsupported recipe link version"
        );
        match (fields.remove("data"), fields.remove("source")) {
            (Some(data), None) => {
                ensure!(
                    fields.is_empty(),
                    "Embedded links cannot include source fields"
                );
                ensure!(
                    data.len() <= MAX_EMBEDDED_RECIPE_BYTES.div_ceil(3) * 4,
                    "Embedded recipe exceeds 32 KiB"
                );
                let bytes = URL_SAFE_NO_PAD
                    .decode(data)
                    .context("Invalid embedded recipe encoding")?;
                ensure!(
                    bytes.len() <= MAX_EMBEDDED_RECIPE_BYTES,
                    "Embedded recipe exceeds 32 KiB"
                );
                let yaml = String::from_utf8(bytes).context("Embedded recipe is not UTF-8")?;
                Recipe::from_yaml(&yaml)?;
                Ok(Self::Embedded { yaml })
            }
            (None, Some(url)) => {
                let request = RecipeImportRequest {
                    url,
                    github_revision: fields.remove("revision"),
                    github_path: fields.remove("path"),
                };
                validate_request(&request)?;
                Ok(Self::Source(request))
            }
            _ => anyhow::bail!("Recipe links require exactly one embedded recipe or HTTPS source"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Source {
    Https(Url),
    Github {
        owner: String,
        repository: String,
        revision: String,
        path: Vec<String>,
    },
}

fn safe_url(value: &str) -> Result<Url> {
    ensure!(value.len() <= 4096, "Import URL exceeds 4096 bytes");
    ensure!(
        value.trim() == value && !value.chars().any(char::is_control) && !value.contains('\\'),
        "Import URL contains ambiguous characters"
    );
    ensure!(
        !value.split('/').any(|part| matches!(part, "." | ".."))
            && !value.to_ascii_lowercase().contains("%2e"),
        "Import URLs cannot contain traversal segments"
    );
    let url = Url::parse(value).context("Enter a valid HTTPS recipe URL")?;
    ensure!(
        url.scheme() == "https" && url.port_or_known_default() == Some(443),
        "Recipe imports require HTTPS on port 443"
    );
    ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "Import URLs cannot contain credentials, queries or fragments"
    );
    ensure!(
        matches!(url.host(), Some(url::Host::Domain(_))),
        "Recipe imports require a public hostname"
    );
    let host = url.host_str().context("Missing destination host")?;
    ensure!(
        host.contains('.')
            && host != "localhost"
            && !host.ends_with(".localhost")
            && !host.ends_with(".local"),
        "Local destinations cannot be imported"
    );
    Ok(url)
}

fn component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn source(request: &RecipeImportRequest) -> Result<Source> {
    let url = safe_url(&request.url)?;
    if url.host_str() != Some("github.com") {
        ensure!(
            request.github_revision.is_none() && request.github_path.is_none(),
            "GitHub revision and path require a github.com repository URL"
        );
        ensure!(
            url.host_str() != Some("raw.githubusercontent.com"),
            "Use the GitHub repository or blob URL so Made can pin and verify the revision"
        );
        return Ok(Source::Https(url));
    }
    let parts: Vec<_> = url
        .path_segments()
        .context("Missing repository path")?
        .filter(|part| !part.is_empty())
        .collect();
    let [owner, repository, rest @ ..] = parts.as_slice() else {
        anyhow::bail!("Provide a GitHub owner and repository")
    };
    ensure!(
        component(owner) && component(repository),
        "Invalid GitHub owner or repository"
    );
    ensure!(
        component(repository.trim_end_matches(".git")),
        "Invalid GitHub repository"
    );
    let (revision, path) = match (
        request.github_revision.as_deref(),
        request.github_path.as_deref(),
        rest,
    ) {
        (Some(revision), Some(path), []) => (revision.to_owned(), path.to_owned()),
        (None, None, ["blob", revision, path @ ..]) if !path.is_empty() => {
            (revision.to_string(), path.join("/"))
        }
        _ => anyhow::bail!(
            "For a repository URL, specify both revision and recipe path; for a blob URL, do not override them"
        ),
    };
    ensure!(
        !revision.is_empty() && revision.len() <= 256 && revision.split('/').all(component),
        "Invalid GitHub revision; specify slash-containing branches with a repository URL"
    );
    let path: Vec<String> = path.split('/').map(str::to_owned).collect();
    ensure!(
        !path.is_empty() && path.len() <= 16 && path.iter().all(|part| component(part)),
        "Use a contained recipe path with at most 16 components"
    );
    Ok(Source::Github {
        owner: owner.to_string(),
        repository: repository.trim_end_matches(".git").to_owned(),
        revision,
        path,
    })
}

pub fn validate_request(request: &RecipeImportRequest) -> Result<()> {
    source(request).map(|_| ())
}

impl RecipeProvenance {
    pub fn validate(&self) -> Result<()> {
        validate_request(&self.request)?;
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Invalid imported recipe content identity"
        );
        ensure!(
            self.commit.as_deref().is_none_or(sha),
            "Invalid imported recipe commit"
        );
        Ok(())
    }
}

fn sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub async fn fetch_recipe(
    http: Arc<dyn HttpClient>,
    request: RecipeImportRequest,
    github_token: Option<String>,
) -> Result<Recipe> {
    fetch_recipe_with_resolver(http, request, github_token, |host| {
        Ok(http_proxy::PinnedHost::resolve(host, 443)?
            .socket_addrs()
            .collect())
    })
    .await
}

async fn fetch_recipe_with_resolver(
    http: Arc<dyn HttpClient>,
    request: RecipeImportRequest,
    github_token: Option<String>,
    resolve: impl Fn(&str) -> Result<Vec<SocketAddr>>,
) -> Result<Recipe> {
    let (body, commit) = match source(&request)? {
        Source::Https(url) => (
            fetch_body(&http, url, None, "text/plain", 256 * 1024, &resolve).await?,
            None,
        ),
        Source::Github {
            owner,
            repository,
            revision,
            path,
        } => {
            let mut endpoint = Url::parse("https://api.github.com/")?;
            endpoint
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("Invalid API URL"))?
                .extend(["repos", &owner, &repository, "commits", &revision]);
            let token = github_token
                .as_deref()
                .filter(|token| !token.trim().is_empty());
            let commit = fetch_body(
                &http,
                endpoint,
                token,
                "application/vnd.github.sha",
                128,
                &resolve,
            )
            .await?;
            let commit = String::from_utf8(commit)
                .context("Invalid GitHub commit response")?
                .trim()
                .to_owned();
            ensure!(sha(&commit), "GitHub did not return an immutable commit");
            let mut tree = commit.clone();
            for (index, component) in path.iter().enumerate() {
                let endpoint = Url::parse(&format!(
                    "https://api.github.com/repos/{owner}/{repository}/git/trees/{tree}"
                ))?;
                let body = fetch_body(
                    &http,
                    endpoint,
                    token,
                    "application/vnd.github+json",
                    1024 * 1024,
                    &resolve,
                )
                .await?;
                #[derive(Deserialize)]
                struct Tree {
                    tree: Vec<Entry>,
                    #[serde(default)]
                    truncated: bool,
                }
                #[derive(Deserialize)]
                struct Entry {
                    path: String,
                    mode: String,
                    #[serde(rename = "type")]
                    kind: String,
                    sha: String,
                }
                let entries: Tree =
                    serde_json::from_slice(&body).context("Invalid GitHub tree response")?;
                ensure!(!entries.truncated, "GitHub directory listing is incomplete");
                let entry = entries
                    .tree
                    .into_iter()
                    .find(|entry| &entry.path == component)
                    .context("Recipe path is unavailable at this commit")?;
                let final_component = index + 1 == path.len();
                ensure!(
                    if final_component {
                        entry.kind == "blob" && matches!(entry.mode.as_str(), "100644" | "100755")
                    } else {
                        entry.kind == "tree" && entry.mode == "040000"
                    },
                    "Recipe imports cannot traverse symlinks, submodules or non-file dependencies"
                );
                ensure!(sha(&entry.sha), "Invalid GitHub object identity");
                tree = entry.sha;
            }
            let endpoint = Url::parse(&format!(
                "https://api.github.com/repos/{owner}/{repository}/git/blobs/{tree}"
            ))?;
            (
                fetch_body(
                    &http,
                    endpoint,
                    token,
                    "application/vnd.github.raw+json",
                    256 * 1024,
                    &resolve,
                )
                .await?,
                Some(commit),
            )
        }
    };
    let sha256 = format!("{:x}", Sha256::digest(&body));
    let content = String::from_utf8(body).context("Recipe source is not UTF-8")?;
    let mut recipe = Recipe::from_yaml(&content)?;
    recipe.imported_from = Some(RecipeProvenance {
        request,
        commit,
        sha256,
    });
    recipe.to_yaml()?;
    Ok(recipe)
}

async fn fetch_body(
    http: &Arc<dyn HttpClient>,
    url: Url,
    token: Option<&str>,
    accept: &str,
    limit: usize,
    resolve: &impl Fn(&str) -> Result<Vec<SocketAddr>>,
) -> Result<Vec<u8>> {
    let host = url.host_str().context("Missing source hostname")?;
    ensure!(
        token.is_none() || host == "api.github.com",
        "Credentials cannot be sent to this destination"
    );
    let addresses =
        resolve(host).context("Recipe destination is not a permitted public address")?;
    let request = Request::get(url.as_str())
        .follow_redirects(RedirectPolicy::NoFollow)
        .timeout(Duration::from_secs(30))
        .header("Accept", accept)
        .when(host == "api.github.com", |request| {
            request.header("X-GitHub-Api-Version", "2022-11-28")
        })
        .when_some(token, |request, token| {
            request.header("Authorization", format!("Bearer {token}"))
        })
        .body(AsyncBody::default())?;
    let mut response = http.send_with_pinned_addresses(request, addresses).await
        .map_err(|_| anyhow::anyhow!("Could not fetch recipe source; check network and pinned-destination transport support"))?;
    ensure!(
        !response.status().is_redirection(),
        "The source redirects; review and import the final HTTPS URL explicitly"
    );
    ensure!(
        response.status().is_success(),
        "Recipe source returned HTTP {}; check revision, path and repository access",
        response.status().as_u16()
    );
    let mut body = Vec::new();
    response
        .body_mut()
        .take(limit as u64 + 1)
        .read_to_end(&mut body)
        .await
        .context("Could not read recipe source")?;
    ensure!(
        body.len() <= limit,
        "Recipe source response exceeds its size limit"
    );
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_client::{FakeHttpClient, Response};
    use std::sync::Mutex;

    const YAML: &str =
        "version: 1\ntitle: Imported\ndescription: Reviewed copy\nprompt: Review changes\n";

    #[test]
    fn recipe_import_links_round_trip_without_side_effects() -> Result<()> {
        for link in [
            RecipeLink::Embedded { yaml: YAML.into() },
            RecipeLink::Source(RecipeImportRequest {
                url: "https://github.com/owner/repo".into(),
                github_revision: Some("feature/review".into()),
                github_path: Some("recipes/review.yaml".into()),
            }),
        ] {
            let encoded = link.encode()?;
            assert!(encoded.starts_with(&product_flavor::url("recipe?")));
            assert_eq!(RecipeLink::decode(&encoded)?, link);
        }
        Ok(())
    }

    #[test]
    fn recipe_import_links_reject_ambiguous_malformed_and_executable_fields() -> Result<()> {
        let valid = RecipeLink::Embedded { yaml: YAML.into() }.encode()?;
        for value in [
            format!("{valid}&v=1"),
            format!("{valid}&enable=true"),
            format!("{valid}&run=true"),
            format!("{valid}&source=https%3A%2F%2Fexample.com%2Fr.yaml"),
            valid.replace("v=1", "v=2"),
            format!("{valid}#fragment"),
            product_flavor::url("recipes?v=1&data=bad"),
            product_flavor::url("recipe?v=1&data=!!!"),
            product_flavor::url("recipe?v=1&source=http%3A%2F%2Flocalhost%2Fr.yaml"),
            product_flavor::url("recipe?v=1&data=eA&path=ignored"),
        ] {
            assert!(RecipeLink::decode(&value).is_err(), "Accepted invalid link");
        }
        let large = RecipeLink::Embedded {
            yaml: format!("{YAML}#{}", "x".repeat(MAX_EMBEDDED_RECIPE_BYTES)),
        };
        assert!(large.encode().is_err());
        assert!(RecipeLink::decode(&"x".repeat(MAX_RECIPE_LINK_BYTES + 1)).is_err());
        Ok(())
    }

    fn public_addresses(_: &str) -> Result<Vec<SocketAddr>> {
        Ok(vec!["93.184.216.34:443".parse()?])
    }

    #[test]
    fn recipe_import_rejects_unsafe_and_ambiguous_sources() {
        for url in [
            "http://example.com/r.yaml",
            "https://127.0.0.1/r.yaml",
            "https://user:secret@example.com/r.yaml",
            "https://example.com/r.yaml?token=secret",
            "https://project.local/r.yaml",
            "https://github.com/owner/repo",
            "https://github.com/owner/repo/blob/main/%2e%2e/r.yaml",
        ] {
            assert!(
                validate_request(&RecipeImportRequest {
                    url: url.into(),
                    ..Default::default()
                })
                .is_err(),
                "accepted {url}"
            );
        }
        let request = RecipeImportRequest {
            url: "https://github.com/owner/repo".into(),
            github_revision: Some("feature/review".into()),
            github_path: Some("recipes/review.yaml".into()),
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn recipe_import_generic_never_sends_github_credentials_and_preserves_identity() -> Result<()> {
        let http = FakeHttpClient::create(|request| async move {
            assert!(request.headers().get("Authorization").is_none());
            assert_eq!(
                request.extensions().get::<RedirectPolicy>(),
                Some(&RedirectPolicy::NoFollow)
            );
            Ok(Response::builder().status(200).body(YAML.into())?)
        });
        let request = RecipeImportRequest {
            url: "https://example.com/recipe.yaml".into(),
            ..Default::default()
        };
        let recipe = futures::executor::block_on(fetch_recipe_with_resolver(
            http,
            request.clone(),
            Some("must-not-leak".into()),
            public_addresses,
        ))?;
        let provenance = recipe.imported_from.as_ref().expect("provenance");
        assert_eq!(provenance.request, request);
        assert!(provenance.commit.is_none());
        assert_eq!(
            provenance.sha256,
            format!("{:x}", Sha256::digest(YAML.as_bytes()))
        );
        assert_eq!(Recipe::from_yaml(&recipe.to_yaml()?)?, recipe);
        Ok(())
    }

    #[test]
    fn recipe_import_pins_commit_and_rejects_symlink_dependencies() -> Result<()> {
        for mode in ["100644", "120000", "160000"] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let observed = requests.clone();
            let http = FakeHttpClient::create(move |request| {
                let observed = observed.clone();
                async move {
                    assert_eq!(request.uri().host(), Some("api.github.com"));
                    assert_eq!(
                        request
                            .headers()
                            .get("Authorization")
                            .expect("private token"),
                        "Bearer test-token"
                    );
                    let path = request.uri().path().to_owned();
                    observed.lock().expect("requests").push(path.clone());
                    let body = if path.ends_with("/commits/main") {
                        "a".repeat(40)
                    } else if path.ends_with(&format!("/git/trees/{}", "a".repeat(40))) {
                        serde_json::json!({"tree":[{"path":"review.yaml","mode":mode,"type":"blob","sha":"b".repeat(40)}],"truncated":false}).to_string()
                    } else if path.ends_with(&format!("/git/blobs/{}", "b".repeat(40))) {
                        YAML.to_owned()
                    } else {
                        anyhow::bail!("Unexpected request")
                    };
                    Ok(Response::builder().status(200).body(body.into())?)
                }
            });
            let request = RecipeImportRequest {
                url: "https://github.com/owner/repo/blob/main/review.yaml".into(),
                ..Default::default()
            };
            let result = futures::executor::block_on(fetch_recipe_with_resolver(
                http,
                request,
                Some("test-token".into()),
                public_addresses,
            ));
            if mode == "100644" {
                assert_eq!(
                    result?.imported_from.expect("provenance").commit,
                    Some("a".repeat(40))
                );
                assert_eq!(requests.lock().expect("requests").len(), 3);
            } else {
                assert!(result.is_err());
                assert_eq!(requests.lock().expect("requests").len(), 2);
            }
        }
        Ok(())
    }

    #[test]
    fn recipe_import_errors_never_turn_into_drafts() -> Result<()> {
        for (status, body) in [
            (302, YAML.to_owned()),
            (404, "private token".into()),
            (200, "x".repeat(256 * 1024 + 1)),
            (200, "version: invalid".into()),
        ] {
            let http = FakeHttpClient::create(move |_| {
                let body = body.clone();
                async move { Ok(Response::builder().status(status).body(body.into())?) }
            });
            let request = RecipeImportRequest {
                url: "https://example.com/recipe.yaml".into(),
                ..Default::default()
            };
            let error = futures::executor::block_on(fetch_recipe_with_resolver(
                http,
                request,
                None,
                public_addresses,
            ))
            .expect_err("no reviewable draft");
            assert!(!error.to_string().contains("private token"));
        }
        Ok(())
    }

    #[test]
    fn recipe_import_cancellation_drops_pending_response() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct PendingBody(Arc<AtomicBool>);
        impl futures::AsyncRead for PendingBody {
            fn poll_read(
                self: std::pin::Pin<&mut Self>,
                _: &mut std::task::Context<'_>,
                _: &mut [u8],
            ) -> std::task::Poll<std::io::Result<usize>> {
                std::task::Poll::Pending
            }
        }
        impl Drop for PendingBody {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let observed = dropped.clone();
        let http = FakeHttpClient::create(move |_| {
            let observed = observed.clone();
            async move {
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from_reader(PendingBody(observed)))?)
            }
        });
        let request = RecipeImportRequest {
            url: "https://example.com/recipe.yaml".into(),
            ..Default::default()
        };
        futures::executor::block_on(async {
            let mut fetch = Box::pin(fetch_recipe_with_resolver(
                http,
                request,
                None,
                public_addresses,
            ));
            assert!(futures::poll!(fetch.as_mut()).is_pending());
            drop(fetch);
        });
        assert!(dropped.load(Ordering::SeqCst));
        Ok(())
    }
}
