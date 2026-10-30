use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{error::Error, fmt};

pub const NAMESPACE: &str = "zed.sim.provider.openrouter";
pub const SOURCE_SHA256: &str = "17657a7a611ceb322c729e99f5de4bd6eca264a2ecb2bec6a320e661b59dae85";
pub const NODE_FEATURE_ID: &str = "COMFY-NODE-0466";
pub const NODE_IDENTIFIER: &str = "OpenRouterLLMNode";
pub const ROUTE_FEATURE_ID: &str = "COMFY-API-EXT-0127";
pub const CHAT_PATH: &str = "/proxy/openrouter/api/v1/chat/completions";
pub const MAXIMUM_REQUEST_BYTES: usize = 1_048_576;
pub const MAXIMUM_RESPONSE_BYTES: usize = 16_777_216;
pub const MAXIMUM_COMPLETION_TOKENS: u32 = 4_096;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    Standard,
    Reasoning,
    FrontierReasoning,
    Perplexity,
    PerplexityReasoning,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSpec {
    pub slug: &'static str,
    pub profile: Profile,
    pub price_in: f64,
    pub price_out: f64,
    pub max_images: usize,
    pub max_videos: usize,
}

#[derive(Debug)]
pub enum OpenRouterError {
    InvalidInput(&'static str),
    InvalidResponse(&'static str),
    Provider { code: String, message: String },
    Refused(String),
    Json(serde_json::Error),
}

impl fmt::Display for OpenRouterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::InvalidResponse(message) => {
                formatter.write_str(message)
            }
            Self::Provider { code, message } => {
                write!(formatter, "OpenRouter error ({code}): {message}")
            }
            Self::Refused(message) => write!(formatter, "Model refused to respond: {message}"),
            Self::Json(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for OpenRouterError {}

impl From<serde_json::Error> for OpenRouterError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub struct ChatInput<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub system_prompt: &'a str,
    pub seed: i64,
    pub reasoning_effort: Option<&'a str>,
    pub search_context_size: Option<&'a str>,
}

pub fn model_spec(slug: &str) -> Result<&'static ModelSpec, OpenRouterError> {
    MODELS
        .iter()
        .find(|model| model.slug == slug)
        .ok_or(OpenRouterError::InvalidInput("Unknown OpenRouter model"))
}

pub fn validate_input(
    input: &ChatInput<'_>,
    image_count: usize,
    video_count: usize,
) -> Result<&'static ModelSpec, OpenRouterError> {
    let model = model_spec(input.model)?;
    if input
        .prompt
        .chars()
        .all(|character| character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}'))
        || input.prompt.len() > MAXIMUM_REQUEST_BYTES
        || input.system_prompt.len() > MAXIMUM_REQUEST_BYTES
        || !(0..=2_147_483_647).contains(&input.seed)
    {
        return Err(OpenRouterError::InvalidInput(
            "Invalid OpenRouter prompt or seed",
        ));
    }
    if image_count > model.max_images || video_count > model.max_videos {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter model media limit exceeded",
        ));
    }
    if input
        .reasoning_effort
        .is_some_and(|effort| !matches!(effort, "off" | "low" | "medium" | "high"))
        || input
            .search_context_size
            .is_some_and(|size| !matches!(size, "low" | "medium" | "high"))
    {
        return Err(OpenRouterError::InvalidInput(
            "Invalid OpenRouter model option",
        ));
    }
    Ok(model)
}

pub fn request_body(
    input: &ChatInput<'_>,
    image_urls: &[String],
    video_urls: &[String],
) -> Result<Vec<u8>, OpenRouterError> {
    validate_input(input, image_urls.len(), video_urls.len())?;
    let mut messages = Vec::new();
    if !input.system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": input.system_prompt}));
    }
    let content = if image_urls.is_empty() && video_urls.is_empty() {
        Value::String(input.prompt.to_owned())
    } else {
        let mut blocks = Vec::new();
        for (kind, urls) in [("image_url", image_urls), ("video_url", video_urls)] {
            for url in urls {
                if !url.starts_with("https://")
                    || url.len() > 2_048
                    || url.chars().any(char::is_control)
                {
                    return Err(OpenRouterError::InvalidInput("Invalid uploaded media URL"));
                }
                blocks.push(json!({"type": kind, kind: {"url": url}}));
            }
        }
        blocks.push(json!({"type": "text", "text": input.prompt}));
        Value::Array(blocks)
    };
    messages.push(json!({"role": "user", "content": content}));
    let mut request = json!({"model": input.model, "messages": messages, "stream": false});
    if input.seed > 0 {
        request["seed"] = json!(input.seed);
    }
    if let Some(effort) = input.reasoning_effort.filter(|effort| *effort != "off") {
        request["reasoning"] = json!({"effort": effort, "exclude": true});
    }
    if let Some(size) = input.search_context_size {
        request["web_search_options"] = json!({"search_context_size": size});
    }
    let bytes = serde_json::to_vec(&request)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter request exceeds its bound",
        ));
    }
    Ok(bytes)
}

/// Add the run's explicit completion bound to the source-derived request.
pub fn bounded_request_body(
    input: &ChatInput<'_>,
    image_urls: &[String],
    video_urls: &[String],
    maximum_completion_tokens: u32,
) -> Result<Vec<u8>, OpenRouterError> {
    if maximum_completion_tokens == 0 || maximum_completion_tokens > MAXIMUM_COMPLETION_TOKENS {
        return Err(OpenRouterError::InvalidInput(
            "Invalid OpenRouter completion-token bound",
        ));
    }
    let mut body: Value = serde_json::from_slice(&request_body(input, image_urls, video_urls)?)?;
    body["max_completion_tokens"] = json!(maximum_completion_tokens);
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter request exceeds its bound",
        ));
    }
    Ok(bytes)
}

/// A per-run planning estimate, not a billing limit. Unknown media and search
/// charges fail closed until a complete pricing snapshot can be supplied.
pub fn estimated_cost_microunits(
    input: &ChatInput<'_>,
    request_bytes: &[u8],
    image_count: usize,
    video_count: usize,
    maximum_completion_tokens: u32,
) -> Result<u64, OpenRouterError> {
    let model = validate_input(input, image_count, video_count)?;
    if image_count != 0
        || video_count != 0
        || input.search_context_size.is_some()
        || matches!(
            model.profile,
            Profile::Perplexity | Profile::PerplexityReasoning
        )
    {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter media or search charges cannot be estimated",
        ));
    }
    if maximum_completion_tokens == 0 || maximum_completion_tokens > MAXIMUM_COMPLETION_TOKENS {
        return Err(OpenRouterError::InvalidInput(
            "Invalid OpenRouter completion-token bound",
        ));
    }
    if request_bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter request exceeds its bound",
        ));
    }
    if request_bytes != bounded_request_body(input, &[], &[], maximum_completion_tokens)? {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter estimate does not match request",
        ));
    }
    let estimate = (request_bytes.len() as f64 * model.price_in
        + f64::from(maximum_completion_tokens) * model.price_out)
        * 1_000_000.0;
    if !estimate.is_finite() || estimate <= 0.0 || estimate > u64::MAX as f64 {
        return Err(OpenRouterError::InvalidInput(
            "OpenRouter cost cannot be estimated",
        ));
    }
    Ok(estimate.ceil() as u64)
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "model")]
    _model: Option<String>,
    #[serde(rename = "object")]
    _object: Option<String>,
    #[serde(rename = "provider")]
    _provider: Option<String>,
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
    error: Option<ProviderError>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(rename = "index")]
    _index: Option<i64>,
    message: Option<ResponseMessage>,
    #[serde(rename = "finish_reason")]
    _finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(rename = "role")]
    _role: Option<String>,
    content: Option<String>,
    #[serde(rename = "reasoning")]
    _reasoning: Option<String>,
    refusal: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(rename = "prompt_tokens")]
    _prompt_tokens: Option<i64>,
    #[serde(rename = "completion_tokens")]
    _completion_tokens: Option<i64>,
    #[serde(rename = "total_tokens")]
    _total_tokens: Option<i64>,
    cost: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ProviderError {
    code: Option<ErrorCode>,
    message: Option<String>,
    #[serde(rename = "metadata")]
    _metadata: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ErrorCode {
    Integer(i64),
    String(String),
}

#[derive(Debug, PartialEq)]
pub struct ChatResult {
    pub text: String,
    pub authoritative_cost_usd: Option<f64>,
}

pub fn parse_response(bytes: &[u8]) -> Result<ChatResult, OpenRouterError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(OpenRouterError::InvalidResponse(
            "OpenRouter response exceeds its bound",
        ));
    }
    let response: ChatResponse = serde_json::from_slice(bytes)?;
    if let Some(error) = response.error {
        let code = match error.code {
            Some(ErrorCode::Integer(code)) => code.to_string(),
            Some(ErrorCode::String(code)) => code,
            None => "unknown".into(),
        };
        return Err(OpenRouterError::Provider {
            code,
            message: error
                .message
                .filter(|message| !message.is_empty())
                .unwrap_or_else(|| "no message".into()),
        });
    }
    let choice = response
        .choices
        .and_then(|choices| choices.into_iter().next())
        .ok_or(OpenRouterError::InvalidResponse(
            "Empty response from OpenRouter (no choices).",
        ))?;
    let message = choice.message.ok_or(OpenRouterError::InvalidResponse(
        "Empty response from OpenRouter (no message).",
    ))?;
    if let Some(refusal) = message.refusal.filter(|refusal| !refusal.is_empty()) {
        return Err(OpenRouterError::Refused(refusal));
    }
    let cost = response.usage.and_then(|usage| usage.cost);
    if cost.is_some_and(|cost| !cost.is_finite() || cost < 0.0) {
        return Err(OpenRouterError::InvalidResponse(
            "Invalid authoritative OpenRouter cost",
        ));
    }
    Ok(ChatResult {
        text: message.content.unwrap_or_default(),
        authoritative_cost_usd: cost,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_conformance(fixture: &str) -> Result<(), crate::common::ConformanceError> {
    let fixture: Value = serde_json::from_str(fixture)
        .map_err(|_| crate::common::ConformanceError::InvalidRequest)?;
    let expected = json!({
        "schema_version": 1,
        "namespace": NAMESPACE,
        "node_feature_id": NODE_FEATURE_ID,
        "node_identifier": NODE_IDENTIFIER,
        "route_feature_id": ROUTE_FEATURE_ID,
        "method": "POST",
        "path": CHAT_PATH,
        "source_sha256": SOURCE_SHA256,
        "models": MODELS,
    });
    if fixture != expected {
        return Err(crate::common::ConformanceError::InvalidRequest);
    }
    Ok(())
}
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        slug: "anthropic/claude-opus-4.7",
        profile: Profile::FrontierReasoning,
        price_in: 5e-06,
        price_out: 2.5e-05,
        max_images: 20,
        max_videos: 0,
    },
    ModelSpec {
        slug: "openai/gpt-5.5-pro",
        profile: Profile::FrontierReasoning,
        price_in: 3e-05,
        price_out: 0.00018,
        max_images: 20,
        max_videos: 0,
    },
    ModelSpec {
        slug: "openai/gpt-5.5",
        profile: Profile::FrontierReasoning,
        price_in: 5e-06,
        price_out: 3e-05,
        max_images: 20,
        max_videos: 0,
    },
    ModelSpec {
        slug: "google/gemini-3.5-flash",
        profile: Profile::Reasoning,
        price_in: 1.5e-06,
        price_out: 9e-06,
        max_images: 20,
        max_videos: 4,
    },
    ModelSpec {
        slug: "x-ai/grok-4.20",
        profile: Profile::Reasoning,
        price_in: 1.25e-06,
        price_out: 2.5e-06,
        max_images: 20,
        max_videos: 0,
    },
    ModelSpec {
        slug: "x-ai/grok-4.3",
        profile: Profile::Reasoning,
        price_in: 1.25e-06,
        price_out: 2.5e-06,
        max_images: 20,
        max_videos: 0,
    },
    ModelSpec {
        slug: "deepseek/deepseek-v4-pro",
        profile: Profile::Reasoning,
        price_in: 4.35e-07,
        price_out: 8.7e-07,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "deepseek/deepseek-v4-flash",
        profile: Profile::Reasoning,
        price_in: 1.12e-07,
        price_out: 2.24e-07,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "deepseek/deepseek-v3.2",
        profile: Profile::Reasoning,
        price_in: 2.52e-07,
        price_out: 3.78e-07,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "qwen/qwen3.6-max-preview",
        profile: Profile::Reasoning,
        price_in: 1.04e-06,
        price_out: 6.24e-06,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "qwen/qwen3.6-plus",
        profile: Profile::Reasoning,
        price_in: 3.25e-07,
        price_out: 1.95e-06,
        max_images: 10,
        max_videos: 4,
    },
    ModelSpec {
        slug: "qwen/qwen3.6-flash",
        profile: Profile::Reasoning,
        price_in: 1.875e-07,
        price_out: 1.125e-06,
        max_images: 10,
        max_videos: 4,
    },
    ModelSpec {
        slug: "mistralai/mistral-large-2512",
        profile: Profile::Standard,
        price_in: 5e-07,
        price_out: 1.5e-06,
        max_images: 8,
        max_videos: 0,
    },
    ModelSpec {
        slug: "mistralai/mistral-medium-3-5",
        profile: Profile::Reasoning,
        price_in: 1.5e-06,
        price_out: 7.5e-06,
        max_images: 8,
        max_videos: 0,
    },
    ModelSpec {
        slug: "z-ai/glm-4.6",
        profile: Profile::Reasoning,
        price_in: 4.3e-07,
        price_out: 1.74e-06,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "z-ai/glm-5",
        profile: Profile::Reasoning,
        price_in: 6e-07,
        price_out: 1.92e-06,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "moonshotai/kimi-k2.6",
        profile: Profile::Reasoning,
        price_in: 7.3e-07,
        price_out: 3.49e-06,
        max_images: 10,
        max_videos: 0,
    },
    ModelSpec {
        slug: "moonshotai/kimi-k2-thinking",
        profile: Profile::Reasoning,
        price_in: 6e-07,
        price_out: 2.5e-06,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "perplexity/sonar-pro",
        profile: Profile::Perplexity,
        price_in: 3e-06,
        price_out: 1.5e-05,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "perplexity/sonar-reasoning-pro",
        profile: Profile::PerplexityReasoning,
        price_in: 2e-06,
        price_out: 8e-06,
        max_images: 0,
        max_videos: 0,
    },
    ModelSpec {
        slug: "perplexity/sonar-deep-research",
        profile: Profile::PerplexityReasoning,
        price_in: 2e-06,
        price_out: 8e-06,
        max_images: 0,
        max_videos: 0,
    },
];

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn input(model: &str) -> ChatInput<'_> {
        ChatInput {
            model,
            prompt: "  Hello 🌍  ",
            system_prompt: "",
            seed: 0,
            reasoning_effort: None,
            search_context_size: None,
        }
    }

    #[test]
    fn openrouter_claims_and_all_model_limits_match_pinned_source() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/openrouter/conformance.fixture"
        )))?;
        assert_eq!(MODELS.len(), 21);
        for model in MODELS {
            let input = input(model.slug);
            assert!(validate_input(&input, model.max_images, model.max_videos).is_ok());
            assert!(validate_input(&input, model.max_images + 1, 0).is_err());
            assert!(validate_input(&input, 0, model.max_videos + 1).is_err());
            let body: Value = serde_json::from_slice(&request_body(&input, &[], &[])?)?;
            assert_eq!(
                body,
                json!({
                    "model": model.slug,
                    "messages": [{"role": "user", "content": "  Hello 🌍  "}],
                    "stream": false,
                })
            );
        }
        Ok(())
    }

    #[test]
    fn openrouter_media_order_seed_and_optional_model_fields_match_source()
    -> Result<(), Box<dyn Error>> {
        let mut input = input("google/gemini-3.5-flash");
        input.seed = 2_147_483_647;
        input.system_prompt = "Be concise";
        input.reasoning_effort = Some("high");
        input.search_context_size = Some("medium");
        let body: Value = serde_json::from_slice(&request_body(
            &input,
            &[
                "https://fixture.invalid/image-1.png".into(),
                "https://fixture.invalid/image-2.png".into(),
            ],
            &["https://fixture.invalid/video.mp4".into()],
        )?)?;
        assert_eq!(
            body,
            json!({
                "model": "google/gemini-3.5-flash",
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "https://fixture.invalid/image-1.png"}},
                        {"type": "image_url", "image_url": {"url": "https://fixture.invalid/image-2.png"}},
                        {"type": "video_url", "video_url": {"url": "https://fixture.invalid/video.mp4"}},
                        {"type": "text", "text": "  Hello 🌍  "},
                    ]},
                ],
                "seed": 2_147_483_647,
                "reasoning": {"effort": "high", "exclude": true},
                "web_search_options": {"search_context_size": "medium"},
                "stream": false,
            })
        );
        for effort in [None, Some("off"), Some("low"), Some("medium"), Some("high")] {
            input.reasoning_effort = effort;
            let body: Value = serde_json::from_slice(&request_body(&input, &[], &[])?)?;
            if effort.is_none() || effort == Some("off") {
                assert!(body.get("reasoning").is_none());
            } else {
                assert_eq!(
                    body["reasoning"],
                    json!({"effort": effort, "exclude": true})
                );
            }
        }
        Ok(())
    }

    #[test]
    fn openrouter_invalid_inputs_fail_before_transport() {
        let mut candidate = input("unknown/model");
        assert!(validate_input(&candidate, 0, 0).is_err());
        candidate.model = "perplexity/sonar-pro";
        for seed in [-1, 2_147_483_648] {
            candidate.seed = seed;
            assert!(validate_input(&candidate, 0, 0).is_err());
        }
        candidate.seed = 0;
        candidate.prompt = " \n\t";
        assert!(validate_input(&candidate, 0, 0).is_err());
        candidate.prompt = "\u{001c}\u{001d}\u{001e}\u{001f}";
        assert!(validate_input(&candidate, 0, 0).is_err());
        candidate.prompt = "hello";
        candidate.reasoning_effort = Some("extreme");
        assert!(validate_input(&candidate, 0, 0).is_err());
        candidate.reasoning_effort = None;
        candidate.search_context_size = Some("huge");
        assert!(validate_input(&candidate, 0, 0).is_err());
    }

    #[test]
    fn openrouter_estimate_requires_bounded_text_only_request() -> Result<(), Box<dyn Error>> {
        let chat_input = input("deepseek/deepseek-v3.2");
        let body = bounded_request_body(&chat_input, &[], &[], 1_024)?;
        let value: Value = serde_json::from_slice(&body)?;
        assert_eq!(value["max_completion_tokens"], 1_024);
        let estimate = estimated_cost_microunits(&chat_input, &body, 0, 0, 1_024)?;
        let model = model_spec(chat_input.model)?;
        let expected = ((body.len() as f64 * model.price_in + 1_024.0 * model.price_out)
            * 1_000_000.0)
            .ceil() as u64;
        assert_eq!(estimate, expected);
        assert!(bounded_request_body(&chat_input, &[], &[], 0).is_err());
        assert!(
            bounded_request_body(&chat_input, &[], &[], MAXIMUM_COMPLETION_TOKENS + 1).is_err()
        );
        assert!(estimated_cost_microunits(&chat_input, &body, 1, 0, 1_024).is_err());
        assert!(estimated_cost_microunits(&chat_input, &body, 0, 1, 1_024).is_err());
        let search = ChatInput {
            search_context_size: Some("low"),
            ..chat_input
        };
        assert!(estimated_cost_microunits(&search, &body, 0, 0, 1_024).is_err());
        let search_model = input("perplexity/sonar-pro");
        let search_body = bounded_request_body(&search_model, &[], &[], 1_024)?;
        assert!(estimated_cost_microunits(&search_model, &search_body, 0, 0, 1_024).is_err());
        Ok(())
    }

    #[test]
    fn openrouter_extracts_only_first_choice_and_preserves_authoritative_cost()
    -> Result<(), Box<dyn Error>> {
        let result = parse_response(br#"{"choices":[{"message":{"content":"first","reasoning":"private"}},{"message":{"content":"second"}}],"usage":{"cost":0.0001234,"prompt_tokens":100,"completion_tokens":20},"unknown":"preserved by source tolerance"}"#)?;
        assert_eq!(
            result,
            ChatResult {
                text: "first".into(),
                authoritative_cost_usd: Some(0.0001234)
            }
        );
        for message in [
            r#"{}"#,
            r#"{"content":null}"#,
            r#"{"content":"","refusal":""}"#,
        ] {
            let body = format!("{{\"choices\":[{{\"message\":{message}}}]}}");
            assert_eq!(
                parse_response(body.as_bytes())?,
                ChatResult {
                    text: String::new(),
                    authoritative_cost_usd: None
                }
            );
        }
        Ok(())
    }

    #[test]
    fn openrouter_error_refusal_and_missing_choices_match_source() {
        let cases = [
            (
                r#"{"error":{"code":429,"message":"busy"}}"#,
                "OpenRouter error (429): busy",
            ),
            (
                r#"{"error":{"code":"denied","message":""}}"#,
                "OpenRouter error (denied): no message",
            ),
            (r#"{"error":{}}"#, "OpenRouter error (unknown): no message"),
            (r#"{}"#, "Empty response from OpenRouter (no choices)."),
            (
                r#"{"choices":[]}"#,
                "Empty response from OpenRouter (no choices).",
            ),
            (
                r#"{"choices":[{}]}"#,
                "Empty response from OpenRouter (no message).",
            ),
            (
                r#"{"choices":[{"message":{"content":"ignored","refusal":"declined"}}]}"#,
                "Model refused to respond: declined",
            ),
        ];
        for (body, expected) in cases {
            let error =
                parse_response(body.as_bytes()).expect_err("invalid provider result must fail");
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn openrouter_rejects_malformed_tails_oversize_and_invalid_cost() {
        for body in [
            r#"{"choices":[{"message":{"content":"prefix"}}]}x"#,
            r#"{"choices":[{"message":{"content":"prefix"}}]}{}"#,
            r#"{"choices":[{"message":{"content":42}}]}"#,
            r#"{"choices":[{"message":{}}],"usage":{"cost":-1}}"#,
            r#"{"choices":[{"message":{}}],"usage":{"cost":1e999}}"#,
            r#"{"choices":[{"message":{}}],"usage":{"prompt_tokens":{}}}"#,
            r#"{"choices":[{"message":{"content":"hello","reasoning":[]}}]}"#,
            r#"{"choices":[{"message":{},"finish_reason":{}}]}"#,
            r#"{"choices":[{"message":{}}],"provider":[]}"#,
            r#"{"error":{"metadata":[]}}"#,
        ] {
            assert!(parse_response(body.as_bytes()).is_err());
        }
        assert!(parse_response(&vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]).is_err());
        let prompt = "a".repeat(MAXIMUM_REQUEST_BYTES);
        let mut candidate = input("deepseek/deepseek-v4-pro");
        candidate.prompt = &prompt;
        assert!(request_body(&candidate, &[], &[]).is_err());
    }
}
