use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.elevenlabs";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "e68033612e73bffa5c9a4353f3a2a280a8803a257debd86e32b0c35dbe9590ef";
pub const MAXIMUM_REQUEST_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0141",
        node_identifier: "ElevenLabsAudioIsolation",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0142",
        node_identifier: "ElevenLabsInstantVoiceClone",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0143",
        node_identifier: "ElevenLabsSpeechToSpeech",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0144",
        node_identifier: "ElevenLabsSpeechToText",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0145",
        node_identifier: "ElevenLabsTextToDialogue",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0146",
        node_identifier: "ElevenLabsTextToSoundEffects",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0147",
        node_identifier: "ElevenLabsTextToSpeech",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0148",
        node_identifier: "ElevenLabsVoiceSelector",
    },
];
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0025",
        method: "POST",
        path: "/proxy/elevenlabs/v1/audio-isolation",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0026",
        method: "POST",
        path: "/proxy/elevenlabs/v1/sound-generation",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0027",
        method: "POST",
        path: "/proxy/elevenlabs/v1/speech-to-speech/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0028",
        method: "POST",
        path: "/proxy/elevenlabs/v1/speech-to-speech/{voice}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0029",
        method: "POST",
        path: "/proxy/elevenlabs/v1/speech-to-text",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0030",
        method: "POST",
        path: "/proxy/elevenlabs/v1/text-to-dialogue",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0031",
        method: "POST",
        path: "/proxy/elevenlabs/v1/text-to-speech/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0032",
        method: "POST",
        path: "/proxy/elevenlabs/v1/text-to-speech/{voice}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0033",
        method: "POST",
        path: "/proxy/elevenlabs/v1/voices/add",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum ElevenLabsError {
    Input(&'static str),
    Response(&'static str),
    Http {
        status: u16,
        retry_after: Option<String>,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for ElevenLabsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "ElevenLabs HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for ElevenLabsError {}
impl From<ConformanceError> for ElevenLabsError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for ElevenLabsError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, ElevenLabsError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/elevenlabs/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
#[derive(Clone, Debug, Default)]
pub struct Resolution {
    pub cloud_storage_url: Option<String>,
    pub clone_name: Option<String>,
    pub clone_file_order: Vec<String>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FormField {
    pub name: String,
    pub value: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct AudioPart {
    pub name: String,
    pub file_name: String,
    pub input_port: String,
    pub source: Value,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: String,
    pub query_params: Option<Value>,
    pub body: Option<Value>,
    pub form_fields: Vec<FormField>,
    pub audio_parts: Vec<AudioPart>,
    pub binary: bool,
    pub local_voice: Option<String>,
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, ElevenLabsError> {
    input
        .arguments
        .get(name)
        .ok_or(ElevenLabsError::Input("Missing ElevenLabs source argument"))
}
fn text(value: &Value) -> Result<&str, ElevenLabsError> {
    value
        .as_str()
        .ok_or(ElevenLabsError::Input("Expected ElevenLabs string"))
}
fn boolean(value: &Value) -> Result<bool, ElevenLabsError> {
    value
        .as_bool()
        .ok_or(ElevenLabsError::Input("Expected ElevenLabs boolean"))
}
fn number(value: &Value, minimum: f64, maximum: f64) -> Result<f64, ElevenLabsError> {
    let number = value
        .as_f64()
        .ok_or(ElevenLabsError::Input("Expected ElevenLabs number"))?;
    if !number.is_finite() || number < minimum || number > maximum {
        return Err(ElevenLabsError::Input("ElevenLabs numeric bound exceeded"));
    }
    Ok(number)
}
fn integer(value: &Value, maximum: u64) -> Result<u64, ElevenLabsError> {
    let number = value.as_u64().ok_or(ElevenLabsError::Input(
        "Expected ElevenLabs unsigned integer",
    ))?;
    if number > maximum {
        return Err(ElevenLabsError::Input("ElevenLabs integer bound exceeded"));
    }
    Ok(number)
}
fn choice<'a>(value: &'a Value, options: &[&str]) -> Result<&'a str, ElevenLabsError> {
    let value = text(value)?;
    if !options.contains(&value) {
        return Err(ElevenLabsError::Input("Unknown ElevenLabs source choice"));
    }
    Ok(value)
}
fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
}
fn prompt(value: &Value) -> Result<&str, ElevenLabsError> {
    let value = text(value)?;
    if value.trim_matches(python_whitespace).is_empty() {
        return Err(ElevenLabsError::Input("Empty ElevenLabs prompt"));
    }
    Ok(value)
}
fn voice(value: &Value) -> Result<&str, ElevenLabsError> {
    let value = text(value)?;
    if value.is_empty()
        || value.len() > 256
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(ElevenLabsError::Input(
            "Invalid ElevenLabs voice identifier",
        ));
    }
    Ok(value)
}
fn language(body: &mut Value, value: &Value) -> Result<(), ElevenLabsError> {
    let language = text(value)?;
    if !language.trim_matches(python_whitespace).is_empty() {
        body["language_code"] = json!(language);
    }
    Ok(())
}
fn form(name: &str, value: &Value) -> Result<FormField, ElevenLabsError> {
    let value = match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => number.to_string(),
        _ => {
            return Err(ElevenLabsError::Input(
                "Unsupported ElevenLabs multipart field",
            ));
        }
    };
    Ok(FormField {
        name: name.into(),
        value,
    })
}
fn audio_part(
    name: &str,
    file_name: &str,
    input_port: &str,
    source: &Value,
) -> Result<AudioPart, ElevenLabsError> {
    if source.is_null()
        || file_name
            .chars()
            .any(|character| character.is_control() || matches!(character, '"' | '\\' | '/'))
    {
        return Err(ElevenLabsError::Input("Invalid ElevenLabs audio source"));
    }
    Ok(AudioPart {
        name: name.into(),
        file_name: file_name.into(),
        input_port: input_port.into(),
        source: source.clone(),
    })
}
#[derive(Serialize)]
struct VoiceSettings {
    stability: f64,
    similarity_boost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_speaker_boost: Option<bool>,
    speed: f64,
}
fn voice_settings(
    input: &Input,
    model: &Value,
    optional: bool,
) -> Result<VoiceSettings, ElevenLabsError> {
    Ok(VoiceSettings {
        stability: number(argument(input, "stability")?, 0.0, 1.0)?,
        similarity_boost: number(&model["similarity_boost"], 0.0, 1.0)?,
        style: match model.get("style") {
            Some(value) => Some(number(value, 0.0, 0.2)?),
            None if optional => None,
            None => return Err(ElevenLabsError::Input("Missing ElevenLabs style")),
        },
        use_speaker_boost: match model.get("use_speaker_boost") {
            Some(value) => Some(boolean(value)?),
            None if optional => None,
            None => return Err(ElevenLabsError::Input("Missing ElevenLabs speaker boost")),
        },
        speed: number(&model["speed"], 0.7, 1.3)?,
    })
}
pub fn price_badge_microunits(node: &str) -> Result<Option<u64>, ElevenLabsError> {
    match node {
        "ElevenLabsSpeechToText" => Ok(Some(7300)),
        "ElevenLabsTextToSpeech"
        | "ElevenLabsSpeechToSpeech"
        | "ElevenLabsAudioIsolation"
        | "ElevenLabsTextToDialogue" => Ok(Some(240000)),
        "ElevenLabsTextToSoundEffects" => Ok(Some(140000)),
        "ElevenLabsInstantVoiceClone" => Ok(Some(150000)),
        "ElevenLabsVoiceSelector" => Ok(None),
        _ => Err(ElevenLabsError::Input("Unknown ElevenLabs node")),
    }
}
pub fn request(input: &Input, resolution: &Resolution) -> Result<Request, ElevenLabsError> {
    let oracle = oracle()?;
    let contract = &oracle["contracts"][&input.node];
    let arguments = contract["arguments"]
        .as_array()
        .ok_or(ElevenLabsError::Input("Unknown ElevenLabs node"))?;
    if input.arguments.len() != arguments.len()
        || input
            .arguments
            .keys()
            .any(|name| !arguments.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(ElevenLabsError::Input(
            "ElevenLabs source argument set differs",
        ));
    }
    let mut request = Request {
        path: String::new(),
        query_params: None,
        body: None,
        form_fields: vec![],
        audio_parts: vec![],
        binary: true,
        local_voice: None,
    };
    let mut form_order: Vec<&str> = vec![];
    match input.node.as_str() {
        "ElevenLabsVoiceSelector" => {
            let selected = text(argument(input, "voice")?)?;
            request.local_voice = Some(
                oracle["voice_map"][selected]
                    .as_str()
                    .ok_or(ElevenLabsError::Input("Unknown ElevenLabs voice"))?
                    .into(),
            );
            return Ok(request);
        }
        "ElevenLabsSpeechToText" => {
            let model = argument(input, "model")?;
            choice(&model["model"], &["scribe_v2"])?;
            let diarize = boolean(&model["diarize"])?;
            let speakers = integer(argument(input, "num_speakers")?, 32)?;
            if diarize && speakers != 0 {
                return Err(ElevenLabsError::Input(
                    "Number of speakers cannot be specified with diarization",
                ));
            }
            let cloud = resolution
                .cloud_storage_url
                .as_deref()
                .ok_or(ElevenLabsError::Input("Canonical audio upload is required"))?;
            if !cloud.starts_with("https://")
                || cloud.contains('@')
                || cloud.chars().any(char::is_control)
            {
                return Err(ElevenLabsError::Input("Invalid ElevenLabs uploaded URL"));
            }
            let mut body = json!({"model_id":model["model"],"cloud_storage_url":cloud,"tag_audio_events":boolean(&model["tag_audio_events"])? ,"timestamps_granularity":choice(&model["timestamps_granularity"],&["word","character","none"])? ,"diarize":diarize,"temperature":number(&model["temperature"],0.0,2.0)?,"seed":integer(argument(input,"seed")?,i32::MAX as u64)?});
            language(&mut body, argument(input, "language_code")?)?;
            if speakers > 0 {
                body["num_speakers"] = json!(speakers);
            }
            if diarize {
                body["diarization_threshold"] =
                    json!(number(&model["diarization_threshold"], 0.1, 0.4)?);
            }
            request.path = "speech-to-text".into();
            request.body = Some(body);
            request.binary = false;
            form_order = vec![
                "model_id",
                "cloud_storage_url",
                "language_code",
                "tag_audio_events",
                "num_speakers",
                "timestamps_granularity",
                "diarize",
                "diarization_threshold",
                "temperature",
                "seed",
            ];
        }
        "ElevenLabsTextToSpeech" => {
            let model = argument(input, "model")?;
            let model_id = choice(&model["model"], &["eleven_multilingual_v2", "eleven_v3"])?;
            let settings = voice_settings(input, model, true)?;
            let mut body = json!({"text":prompt(argument(input,"text")?)?,"model_id":model_id,"voice_settings":settings,"seed":integer(argument(input,"seed")?,i32::MAX as u64)?,"apply_text_normalization":choice(argument(input,"apply_text_normalization")?,&["auto","on","off"])?});
            language(&mut body, argument(input, "language_code")?)?;
            request.path = format!("text-to-speech/{}", voice(argument(input, "voice")?)?);
            request.body = Some(body);
        }
        "ElevenLabsAudioIsolation" => {
            request.path = "audio-isolation".into();
            request.audio_parts.push(audio_part(
                "audio",
                "audio.mp4",
                "audio",
                argument(input, "audio")?,
            )?);
        }
        "ElevenLabsTextToSoundEffects" => {
            let model = argument(input, "model")?;
            choice(&model["model"], &["eleven_sfx_v2"])?;
            let mut body = json!({"text":prompt(argument(input,"text")?)?,"duration_seconds":number(&model["duration"],0.5,30.0)?,"prompt_influence":number(&model["prompt_influence"],0.0,1.0)?});
            if let Some(value) = model.get("loop") {
                body["loop"] = json!(boolean(value)?);
            }
            request.path = "sound-generation".into();
            request.body = Some(body);
        }
        "ElevenLabsInstantVoiceClone" => {
            let name = resolution
                .clone_name
                .as_deref()
                .ok_or(ElevenLabsError::Input(
                    "Host-generated clone UUID is required",
                ))?;
            let uuid = name.as_bytes();
            if uuid.len() != 36
                || !uuid.iter().enumerate().all(|(index, byte)| {
                    if [8, 13, 18, 23].contains(&index) {
                        *byte == b'-'
                    } else {
                        byte.is_ascii_hexdigit()
                    }
                })
                || uuid.get(14) != Some(&b'4')
                || !uuid
                    .get(19)
                    .is_some_and(|byte| matches!(byte, b'8' | b'9' | b'a' | b'b' | b'A' | b'B'))
            {
                return Err(ElevenLabsError::Input("Invalid clone UUIDv4"));
            }
            let files = argument(input, "files")?
                .as_object()
                .ok_or(ElevenLabsError::Input("Invalid clone file inputs"))?;
            let ordered = &resolution.clone_file_order;
            if files.is_empty()
                || ordered.len() != files.len()
                || ordered.iter().collect::<BTreeSet<_>>().len() != files.len()
            {
                return Err(ElevenLabsError::Input(
                    "Canonical clone input order is required",
                ));
            }
            for key in ordered {
                let source = files
                    .get(key)
                    .ok_or(ElevenLabsError::Input("Unknown clone input"))?;
                request
                    .audio_parts
                    .push(audio_part("files", &format!("{key}.mp4"), key, source)?);
            }
            request.path = "voices/add".into();
            request.body = Some(
                json!({"name":name,"remove_background_noise":boolean(argument(input,"remove_background_noise")?)?}),
            );
            request.binary = false;
            form_order = vec!["name", "remove_background_noise"];
        }
        "ElevenLabsSpeechToSpeech" => {
            let model = argument(input, "model")?;
            let model_id = choice(
                &model["model"],
                &["eleven_multilingual_sts_v2", "eleven_english_sts_v2"],
            )?;
            let settings = serde_json::to_string(&voice_settings(input, model, false)?)?;
            request.path = format!("speech-to-speech/{}", voice(argument(input, "voice")?)?);
            request.body = Some(
                json!({"model_id":model_id,"voice_settings":settings,"seed":integer(argument(input,"seed")?,u32::MAX as u64)?,"remove_background_noise":boolean(argument(input,"remove_background_noise")?)?}),
            );
            form_order = vec![
                "model_id",
                "voice_settings",
                "seed",
                "remove_background_noise",
            ];
            request.audio_parts.push(audio_part(
                "audio",
                "audio.mp4",
                "audio",
                argument(input, "audio")?,
            )?);
        }
        "ElevenLabsTextToDialogue" => {
            let inputs = argument(input, "inputs")?;
            let count = text(&inputs["inputs"])?
                .parse::<usize>()
                .map_err(|_| ElevenLabsError::Input("Invalid dialogue count"))?;
            if !(1..=10).contains(&count) {
                return Err(ElevenLabsError::Input(
                    "Dialogue count exceeds source range",
                ));
            }
            let mut segments = vec![];
            for index in 1..=count {
                segments.push(json!({"text":prompt(&inputs[format!("text{index}")])?,"voice_id":voice(&inputs[format!("voice{index}")])?}));
            }
            let mut body = json!({"inputs":segments,"model_id":choice(argument(input,"model")?,&["eleven_v3"])? ,"settings":{"stability":number(argument(input,"stability")?,0.0,1.0)?},"seed":integer(argument(input,"seed")?,u32::MAX as u64)?,"apply_text_normalization":choice(argument(input,"apply_text_normalization")?,&["auto","on","off"])?});
            language(&mut body, argument(input, "language_code")?)?;
            request.path = "text-to-dialogue".into();
            request.body = Some(body);
        }
        _ => return Err(ElevenLabsError::Input("Unknown ElevenLabs node")),
    }
    if let Some(value) = input.arguments.get("output_format") {
        request.query_params =
            Some(json!({"output_format":choice(value,&["mp3_44100_192","opus_48000_192"])?}));
    }
    for name in form_order {
        if let Some(value) = request.body.as_ref().and_then(|body| body.get(name)) {
            request.form_fields.push(form(name, value)?);
        }
    }
    request.path = format!("/proxy/elevenlabs/v1/{}", request.path);
    Ok(request)
}
#[derive(Clone, Copy, Debug)]
pub struct CredentialAvailability {
    pub auth_token: bool,
    pub api_key: bool,
}
fn request_head(
    path: &str,
    method: HttpMethod,
    credentials: CredentialAvailability,
    binary: bool,
) -> Result<RequestHead, ElevenLabsError> {
    if !path.starts_with("/proxy/elevenlabs/v1/") || path.chars().any(char::is_control) {
        return Err(ElevenLabsError::Input("Invalid ElevenLabs endpoint"));
    }
    let mut headers = vec![Header {
        name: "Accept".into(),
        value: if binary { "*/*" } else { "application/json" }.into(),
    }];
    let secret_id = {
        let (secret, name, value) = if credentials.auth_token {
            (
                "sim_org.auth_token",
                "Authorization",
                "host-capability:auth_token_sim_org",
            )
        } else if credentials.api_key {
            (
                "sim_org.api_key",
                "X-API-KEY",
                "host-capability:api_key_sim_org",
            )
        } else {
            return Err(ElevenLabsError::Input("Missing Comfy API credentials"));
        };
        headers.push(Header {
            name: name.into(),
            value: value.into(),
        });
        headers.push(Header {
            name: "Comfy-Env".into(),
            value: "host".into(),
        });
        headers.push(Header {
            name: "Comfy-Usage-Source".into(),
            value: "comfyui-api".into(),
        });
        Some(secret.into())
    };
    if method == HttpMethod::Post {
        headers.push(Header {
            name: "Content-Type".into(),
            value: "application/json".into(),
        });
    }
    Ok(RequestHead {
        method,
        path: path.into(),
        secret_id,
        headers,
    })
}

pub trait ElevenLabsTransport: ProviderTransport {
    fn retry_after_delay(
        &mut self,
        header: &str,
        fallback_milliseconds: u64,
    ) -> Result<u64, ConformanceError>;
    fn retry_request(
        &mut self,
        handle: StreamHandle,
        delay_milliseconds: u64,
    ) -> Result<StreamHandle, ConformanceError>;
}
pub trait ElevenLabsAudioPreparation {
    fn prepare_mp4_aac(&mut self, part: &AudioPart) -> Result<Vec<u8>, ConformanceError>;
    fn upload_audio(&mut self, source: &Value) -> Result<String, ConformanceError>;
    fn clone_uuid(&mut self) -> Result<String, ConformanceError>;
    fn clone_file_order(&mut self, input: &Input) -> Result<Vec<String>, ConformanceError>;
}
pub struct MultipartBody {
    pub bytes: Vec<u8>,
    pub content_type: String,
}
pub trait ElevenLabsMultipartEncoder {
    fn content_type(&self) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn encode(&mut self, fields: &[MultipartField]) -> Result<MultipartBody, ConformanceError>;
}
pub struct CanonicalMultipartEncoder {
    pub boundary: String,
}
impl ElevenLabsMultipartEncoder for CanonicalMultipartEncoder {
    fn content_type(&self) -> Result<String, ConformanceError> {
        MultipartField::encode_checked(
            &[],
            &self.boundary,
            crate::common::TransportLimits::conformance(),
        )
        .map(|_| format!("multipart/form-data; boundary={}", self.boundary))
    }
    fn encode(&mut self, fields: &[MultipartField]) -> Result<MultipartBody, ConformanceError> {
        let mut limits = crate::common::TransportLimits::conformance();
        limits.maximum_body_bytes = MAXIMUM_REQUEST_BYTES;
        Ok(MultipartBody {
            bytes: MultipartField::encode_checked(fields, &self.boundary, limits)?,
            content_type: self.content_type()?,
        })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Audio,
    Transcript(Vec<String>),
    Voice(String),
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub node: String,
    pub output: Output,
    pub response_bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
}
#[derive(Deserialize, Serialize)]
struct Word {
    text: String,
    #[serde(default = "word_type")]
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprob: Option<f64>,
}
fn word_type() -> String {
    "word".into()
}
#[derive(Deserialize)]
struct Transcript {
    text: String,
    language_code: String,
    words: Option<Vec<Word>>,
}
fn ascii_json(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        if character.is_ascii() {
            output.push(character);
        } else {
            let mut buffer = [0u16; 2];
            for word in character.encode_utf16(&mut buffer) {
                output.push_str(&format!("\\u{word:04x}"));
            }
        }
    }
    output
}
pub fn result_value(
    node: &str,
    bytes: Vec<u8>,
    receipt: Vec<u8>,
) -> Result<ResultValue, ElevenLabsError> {
    if bytes.len() > MAXIMUM_MEDIA_BYTES
        || bytes.is_empty()
        || receipt.is_empty()
        || receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ElevenLabsError::Response(
            "Invalid ElevenLabs response bounds",
        ));
    }
    let output = match node {
        "ElevenLabsSpeechToText" => {
            let transcript: Transcript = serde_json::from_slice(&bytes)?;
            let words = ascii_json(&serde_json::to_string_pretty(
                &transcript.words.unwrap_or_default(),
            )?);
            Output::Transcript(vec![transcript.text, transcript.language_code, words])
        }
        "ElevenLabsInstantVoiceClone" => {
            let value: Value = serde_json::from_slice(&bytes)?;
            Output::Voice(voice(&value["voice_id"])?.into())
        }
        "ElevenLabsTextToSpeech"
        | "ElevenLabsSpeechToSpeech"
        | "ElevenLabsAudioIsolation"
        | "ElevenLabsTextToSoundEffects"
        | "ElevenLabsTextToDialogue" => Output::Audio,
        _ => return Err(ElevenLabsError::Input("Unknown ElevenLabs response node")),
    };
    Ok(ResultValue {
        node: node.into(),
        output,
        response_bytes: bytes,
        response_receipt: receipt,
    })
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn ElevenLabsTransport,
    preparation: &mut dyn ElevenLabsAudioPreparation,
    encoder: &mut dyn ElevenLabsMultipartEncoder,
) -> Result<ResultValue, ElevenLabsError> {
    if input.node == "ElevenLabsVoiceSelector" {
        let selected = request(input, &Resolution::default())?
            .local_voice
            .ok_or(ElevenLabsError::Input("Missing local voice"))?;
        transport.check_cancelled(None)?;
        return Ok(ResultValue {
            node: input.node.clone(),
            output: Output::Voice(selected),
            response_bytes: vec![],
            response_receipt: vec![],
        });
    }
    // Resolve only after credentials and cost have been approved; native preparation remains host owned.
    let mut head = request_head(
        "/proxy/elevenlabs/v1/preflight",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let mut resolution = Resolution::default();
    // Validate every source argument before upload, using explicitly non-executable resolution values.
    if input.node == "ElevenLabsSpeechToText" {
        resolution.cloud_storage_url = Some("https://preflight.invalid/audio".into());
    }
    if input.node == "ElevenLabsInstantVoiceClone" {
        resolution.clone_name = Some("00000000-0000-4000-8000-000000000000".into());
        resolution.clone_file_order = preparation.clone_file_order(input)?;
    }
    let preflight = request(input, &resolution)?;
    let path = if let Some(query) = &preflight.query_params {
        format!(
            "{}?output_format={}",
            preflight.path,
            text(&query["output_format"])?
        )
    } else {
        preflight.path.clone()
    };
    head.path = path;
    head.headers
        .first_mut()
        .ok_or(ElevenLabsError::Input("Missing Accept header"))?
        .value = if preflight.binary {
        "*/*".into()
    } else {
        "application/json".into()
    };
    transport.check_cancelled(None)?;
    let declared_content_type =
        if !preflight.form_fields.is_empty() || !preflight.audio_parts.is_empty() {
            let content_type = encoder.content_type()?;
            let boundary = content_type
                .strip_prefix("multipart/form-data; boundary=")
                .ok_or(ElevenLabsError::Input(
                    "Invalid canonical multipart Content-Type",
                ))?;
            MultipartField::encode_checked(
                &[],
                boundary,
                crate::common::TransportLimits::conformance(),
            )?;
            head.headers
                .iter_mut()
                .find(|header| header.name == "Content-Type")
                .ok_or(ElevenLabsError::Input("Missing Content-Type header"))?
                .value = content_type.clone();
            Some(content_type)
        } else {
            None
        };
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    approve(handle, maximum_cost_microunits, transport)?;
    transport.check_cancelled(Some(handle))?;
    if input.node == "ElevenLabsSpeechToText" {
        resolution.cloud_storage_url = Some(preparation.upload_audio(argument(input, "audio")?)?);
    }
    if input.node == "ElevenLabsInstantVoiceClone" {
        resolution.clone_name = Some(preparation.clone_uuid()?);
    }
    let request = request(input, &resolution)?;
    let body = if !request.form_fields.is_empty() || !request.audio_parts.is_empty() {
        let mut fields = request
            .form_fields
            .iter()
            .map(|field| MultipartField {
                name: field.name.clone(),
                file_name: None,
                media_type: Some("text/plain; charset=utf-8".into()),
                bytes: field.value.as_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        for part in &request.audio_parts {
            transport.check_cancelled(Some(handle))?;
            let bytes = preparation.prepare_mp4_aac(part)?;
            if bytes.is_empty() || bytes.len() > MAXIMUM_MEDIA_BYTES {
                return Err(ElevenLabsError::Input("Invalid canonical MP4/AAC payload"));
            }
            fields.push(MultipartField {
                name: part.name.clone(),
                file_name: Some(part.file_name.clone()),
                media_type: Some("audio/mp4".into()),
                bytes,
            });
        }
        let encoded = encoder.encode(&fields)?;
        if declared_content_type.as_deref() != Some(encoded.content_type.as_str()) {
            return Err(ElevenLabsError::Input(
                "Canonical multipart Content-Type changed after authorization",
            ));
        }
        encoded.bytes
    } else {
        serde_json::to_vec(
            &request
                .body
                .ok_or(ElevenLabsError::Input("Missing ElevenLabs request body"))?,
        )?
    };
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (bytes, receipt, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_MEDIA_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    result_value(&input.node, bytes, receipt)
}
fn approve(
    handle: StreamHandle,
    maximum_microunits: u64,
    transport: &mut dyn ElevenLabsTransport,
) -> Result<(), ElevenLabsError> {
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "elevenlabs.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !approval.accepted
        || approval.approved_microunits < maximum_microunits
        || approval.receipt.is_empty()
        || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    Ok(())
}
#[derive(Clone, Copy)]
struct RetryPolicy {
    ordinary_limit: u32,
    rate_limit: u32,
    backoff: f64,
    download: bool,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 3,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
};
#[cfg(test)]
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn ElevenLabsTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), ElevenLabsError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    exchange_existing(head, body, cost, maximum_bytes, policy, handle, transport)
}
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    transport: &mut dyn ElevenLabsTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), ElevenLabsError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(ElevenLabsError::Http {
                status,
                retry_after,
            }) => {
                let delay = if status == 429
                    && !policy.download
                    && rate_limit_retries < policy.rate_limit
                {
                    let delay = (1000.0 * policy.backoff.powi(rate_limit_retries as i32))
                        .round()
                        .min(30_000.0) as u64;
                    rate_limit_retries += 1;
                    delay
                } else if (matches!(status, 408 | 500 | 502 | 503 | 504)
                    || policy.download && status == 429)
                    && ordinary_retries < policy.ordinary_limit
                {
                    let delay =
                        (1000.0 * policy.backoff.powi(ordinary_retries as i32)).round() as u64;
                    ordinary_retries += 1;
                    delay
                } else {
                    return Err(ElevenLabsError::Http {
                        status,
                        retry_after,
                    });
                };
                let delay = if !policy.download {
                    match retry_after {
                        Some(value) => transport.retry_after_delay(&value, delay)?.min(150_000),
                        None => delay,
                    }
                } else {
                    delay
                };
                transport.check_cancelled(Some(handle))?;
                let fresh = transport.retry_request(handle, delay)?;
                if fresh == handle || fresh.invocation != handle.invocation {
                    return Err(ConformanceError::InvalidHandle.into());
                }
                handle = fresh;
            }
            result => return result,
        }
    }
}

fn exchange_on_handle(
    handle: StreamHandle,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    transport: &mut dyn ElevenLabsTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), ElevenLabsError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "elevenlabs.generation.maximum".into(),
                currency: "USD".into(),
                maximum_microunits,
            },
        )?;
        if !approval.accepted
            || approval.approved_microunits < maximum_microunits
            || approval.receipt.is_empty()
            || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(ConformanceError::InvalidCost.into());
        }
    }
    if body.is_empty() {
        transport.check_cancelled(Some(handle))?;
        transport.write_request_chunk(handle, 0, &[], true)?;
    } else {
        let count = body.len().div_ceil(MAXIMUM_CHUNK_BYTES);
        for (sequence, chunk) in body.chunks(MAXIMUM_CHUNK_BYTES).enumerate() {
            transport.check_cancelled(Some(handle))?;
            transport.write_request_chunk(handle, sequence as u64, chunk, sequence + 1 == count)?;
        }
    }
    let mut status = None;
    let mut retry_after = None;
    let mut bytes = Vec::new();
    let mut after_sequence = None;
    let mut frame_sequence = 0u64;
    for _ in 0..4096 {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(handle, after_sequence, 1000)? {
            ResponseEvent::TimedOut => {}
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled.into()),
            ResponseEvent::Head {
                status: response_status,
                headers,
            } if status.is_none() => {
                if headers.len() > 16
                    || headers.iter().any(|header| {
                        header.name.len() + header.value.len() > 16 * 1024
                            || header.name.chars().any(char::is_control)
                            || header.value.chars().any(char::is_control)
                    })
                {
                    return Err(ConformanceError::HeaderLimit.into());
                }
                retry_after = headers
                    .iter()
                    .find(|header| header.name.eq_ignore_ascii_case("Retry-After"))
                    .map(|header| header.value.clone());
                status = Some(response_status);
                after_sequence = Some(frame_sequence);
                frame_sequence += 1;
            }
            ResponseEvent::Chunk(chunk) if status.is_some() => {
                if chunk.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(ConformanceError::ChunkLimit.into());
                }
                if bytes
                    .len()
                    .checked_add(chunk.len())
                    .is_none_or(|length| length > maximum_bytes)
                {
                    return Err(ConformanceError::BodyLimit.into());
                }
                bytes.extend(chunk);
                after_sequence = Some(frame_sequence);
                frame_sequence += 1;
            }
            ResponseEvent::Completed(receipt) if status.is_some() => {
                if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                if !status.is_some_and(|status| (200..300).contains(&status)) {
                    return Err(ElevenLabsError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                    });
                }
                transport.check_cancelled(Some(handle))?;
                return Ok((bytes, receipt, handle));
            }
            _ => return Err(ConformanceError::InvalidResponse.into()),
        }
    }
    Err(ConformanceError::PollLimit.into())
}

pub trait ElevenLabsMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct ExecutionAdapters<'a> {
    pub transport: &'a mut dyn ElevenLabsTransport,
    pub preparation: &'a mut dyn ElevenLabsAudioPreparation,
    pub encoder: &'a mut dyn ElevenLabsMultipartEncoder,
    pub materializer: &'a mut dyn ElevenLabsMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    adapters: ExecutionAdapters<'_>,
) -> Result<(), ElevenLabsError> {
    let ExecutionAdapters {
        transport,
        preparation,
        encoder,
        materializer,
        sink,
    } = adapters;
    let result = invoke(
        input,
        credentials,
        maximum_cost_microunits,
        transport,
        preparation,
        encoder,
    )?;
    transport.check_cancelled(None)?;
    let outputs = materializer.materialize(&result)?;
    let registry =
        TypeRegistry::built_in().map_err(|_| ElevenLabsError::Response("Missing type registry"))?;
    let aliases = match result.output {
        Output::Audio => vec!["AUDIO"],
        Output::Transcript(_) => vec!["STRING", "STRING", "STRING"],
        Output::Voice(_) => vec!["ELEVENLABS_VOICE"],
    };
    let expected = aliases
        .into_iter()
        .enumerate()
        .map(|(index, alias)| {
            let type_id = registry
                .resolve(alias)
                .map_err(|_| ElevenLabsError::Response("Missing output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| ElevenLabsError::Response("Missing output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect::<Result<Vec<_>, ElevenLabsError>>()?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn case_resolution(case: &Value) -> Result<Resolution, ElevenLabsError> {
    let mut resolution = Resolution::default();
    if case["input"]["node"] == "ElevenLabsSpeechToText" {
        resolution.cloud_storage_url = Some("https://storage.fixture.invalid/audio".into());
    }
    if case["input"]["node"] == "ElevenLabsInstantVoiceClone" {
        resolution.clone_name = Some("00000000-0000-4000-8000-000000000042".into());
        resolution.clone_file_order = case["requests"][0]["file_fields"]
            .as_array()
            .ok_or(ElevenLabsError::Input("Missing source clone file order"))?
            .iter()
            .map(|field| Ok(text(&field["file_name"])?.trim_end_matches(".mp4").into()))
            .collect::<Result<Vec<_>, ElevenLabsError>>()?;
    }
    Ok(resolution)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({
            "schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,
            "aliases":ALIASES,"nodes":NODES,"routes":ROUTES,
            "scope":"source request and bounded transport; canonical audio, multipart and custom voice integration pending",
            "source_validation":{
                "python_strip_control_separators":[28,29,30,31],
                "zero_width_space_is_whitespace":false,
                "byte_order_mark_is_whitespace":false,
                "nonempty_text_and_language_preserve_original":true,
                "validation_utils_sha256":"77554cc809f651f54943f73e6391b8899b3a52779d949edfc08166d8fdf86c66",
                "text_validation_nodes":["ElevenLabsTextToSpeech","ElevenLabsTextToSoundEffects","ElevenLabsTextToDialogue"],
                "language_omission_nodes":["ElevenLabsSpeechToText","ElevenLabsTextToSpeech","ElevenLabsTextToDialogue"]
            },
            "multipart_header_binding": {
                "declared_before_request_and_cost":true,
                "encoded_header_must_match":true,
                "mutable_transport_header":false,
                "client_sha256":"939696eb63765794ecd8a1a90a076fd33118603045a9d55851fd8fb67db60ecc"
            }
        })
    {
        return Err(ConformanceError::InvalidRequest);
    }
    for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
        let input = json!(separator.to_string());
        let mut body = json!({});
        language(&mut body, &input).map_err(|_| ConformanceError::InvalidRequest)?;
        if prompt(&input).is_ok() || body.get("language_code").is_some() {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    for original in ["\u{001c} Original text \u{001f}", "\u{200b}", "\u{feff}"] {
        let input = json!(original);
        let mut body = json!({});
        language(&mut body, &input).map_err(|_| ConformanceError::InvalidRequest)?;
        if prompt(&input).map_err(|_| ConformanceError::InvalidRequest)? != original
            || body["language_code"] != original
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    let oracle = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_elevenlabs.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let observed = request(
            &input,
            &case_resolution(case).map_err(|_| ConformanceError::InvalidRequest)?,
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if let Some(voice) = observed.local_voice {
            if case["outputs"] != json!([voice]) || case["requests"] != json!([]) {
                return Err(ConformanceError::InvalidRequest);
            }
        } else {
            let expected = &case["requests"][0];
            let files=observed.audio_parts.iter().map(|part|json!({"name":part.name,"file_name":part.file_name,"media_type":"audio/mp4","source_slot":part.source["waveform"]["source_audio"],"boundary":"canonical MP4/AAC owner required"})).collect::<Vec<_>>();
            if expected["path"] != observed.path
                || expected["body"] != observed.body.unwrap_or(Value::Null)
                || expected["query_params"] != observed.query_params.unwrap_or(Value::Null)
                || expected["form_fields"] != json!(observed.form_fields)
                || expected["file_fields"] != json!(files)
                || expected["as_binary"] != observed.binary
            {
                return Err(ConformanceError::InvalidRequest);
            }
        }
        covered.insert(input.node);
    }
    for case in oracle["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let resolution = Resolution {
            cloud_storage_url: Some("https://storage.fixture.invalid/audio".into()),
            ..Resolution::default()
        };
        if request(&input, &resolution).is_ok()
            || case["requests"] != json!([])
            || case["uploads"] != json!([])
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if covered
        != NODES
            .iter()
            .map(|node| node.node_identifier.to_owned())
            .collect()
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        CostResponse, MaterializationError, MaterializedBatch, Progress, SignedPutRequest,
        StorageCreateRequest, StorageCreateResponse, ValueFamily,
    };
    use std::collections::VecDeque;
    #[derive(Default)]
    struct Transport {
        heads: Vec<RequestHead>,
        bodies: Vec<Vec<u8>>,
        costs: Vec<CostRequest>,
        events: VecDeque<Result<ResponseEvent, ConformanceError>>,
        delays: Vec<u64>,
        cancelled: bool,
        deny_cost: bool,
        same_retry_handle: bool,
        expected_content_type_at_cost: Option<String>,
    }
    fn handle(slot: u32) -> StreamHandle {
        StreamHandle {
            invocation: 42,
            slot,
            generation: 1,
        }
    }
    impl ProviderTransport for Transport {
        fn check_cancelled(
            &mut self,
            _handle: Option<StreamHandle>,
        ) -> Result<(), ConformanceError> {
            if self.cancelled {
                Err(ConformanceError::Cancelled)
            } else {
                Ok(())
            }
        }
        fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
            self.heads.push(head.clone());
            self.bodies.push(vec![]);
            Ok(handle(self.heads.len() as u32))
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            _end: bool,
        ) -> Result<(), ConformanceError> {
            let body = self
                .bodies
                .get_mut(handle.slot as usize - 1)
                .ok_or(ConformanceError::InvalidHandle)?;
            if body.len().div_ceil(MAXIMUM_CHUNK_BYTES) != sequence as usize {
                return Err(ConformanceError::InvalidRequest);
            }
            body.extend_from_slice(bytes);
            Ok(())
        }
        fn upload(&mut self, _handle: StreamHandle, _bytes: &[u8]) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn create_storage(
            &mut self,
            _handle: StreamHandle,
            _request: &StorageCreateRequest,
        ) -> Result<StorageCreateResponse, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn put_signed(
            &mut self,
            _handle: StreamHandle,
            _request: &SignedPutRequest,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn request_cost(
            &mut self,
            _handle: StreamHandle,
            request: &CostRequest,
        ) -> Result<CostResponse, ConformanceError> {
            if let Some(expected) = &self.expected_content_type_at_cost {
                assert!(self.heads.last().is_some_and(|head| {
                    head.headers
                        .iter()
                        .any(|header| header.name == "Content-Type" && header.value == *expected)
                }));
            }
            self.costs.push(request.clone());
            Ok(CostResponse {
                accepted: !self.deny_cost,
                approved_microunits: request.maximum_microunits,
                receipt: vec![42],
            })
        }
        fn report_progress(
            &mut self,
            _handle: StreamHandle,
            _progress: &Progress,
        ) -> Result<(), ConformanceError> {
            Ok(())
        }
        fn wait_response(
            &mut self,
            _handle: StreamHandle,
            _after: Option<u64>,
            _timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            self.events
                .pop_front()
                .ok_or(ConformanceError::HostFailure)?
        }
    }
    impl ElevenLabsTransport for Transport {
        fn retry_after_delay(
            &mut self,
            header: &str,
            fallback: u64,
        ) -> Result<u64, ConformanceError> {
            Ok(header
                .parse::<u64>()
                .ok()
                .map(|seconds| seconds.saturating_mul(1000))
                .unwrap_or(fallback))
        }
        fn retry_request(
            &mut self,
            old: StreamHandle,
            delay: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            self.delays.push(delay);
            if self.same_retry_handle {
                return Ok(old);
            }
            let head = self
                .heads
                .last()
                .cloned()
                .ok_or(ConformanceError::InvalidHandle)?;
            self.start_request(&head)
        }
    }
    #[derive(Default)]
    struct NativeBoundary {
        preparations: usize,
        uploads: usize,
        uuid_calls: usize,
        encodes: usize,
    }
    impl ElevenLabsAudioPreparation for NativeBoundary {
        fn prepare_mp4_aac(&mut self, _part: &AudioPart) -> Result<Vec<u8>, ConformanceError> {
            self.preparations += 1;
            Err(ConformanceError::HostFailure)
        }
        fn upload_audio(&mut self, _source: &Value) -> Result<String, ConformanceError> {
            self.uploads += 1;
            Err(ConformanceError::HostFailure)
        }
        fn clone_uuid(&mut self) -> Result<String, ConformanceError> {
            self.uuid_calls += 1;
            Ok("00000000-0000-4000-8000-000000000042".into())
        }
        fn clone_file_order(&mut self, input: &Input) -> Result<Vec<String>, ConformanceError> {
            if input.node == "ElevenLabsInstantVoiceClone" {
                Ok(vec!["sample_2".into(), "sample_1".into()])
            } else {
                Err(ConformanceError::InvalidRequest)
            }
        }
    }
    impl ElevenLabsMultipartEncoder for NativeBoundary {
        fn content_type(&self) -> Result<String, ConformanceError> {
            CanonicalMultipartEncoder {
                boundary: "actual-canonical-boundary".into(),
            }
            .content_type()
        }
        fn encode(
            &mut self,
            _fields: &[MultipartField],
        ) -> Result<MultipartBody, ConformanceError> {
            self.encodes += 1;
            Err(ConformanceError::HostFailure)
        }
    }
    fn fixture_case(node: &str) -> Result<Value, Box<dyn Error>> {
        Ok(oracle()?["cases"]
            .as_array()
            .ok_or("Missing oracle cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .ok_or("Missing source case")?
            .clone())
    }
    fn fixture_input(node: &str) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(
            fixture_case(node)?["input"].clone(),
        )?)
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn audio() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/elevenlabs/canonical-response.wav"
        ))
        .to_vec()
    }
    fn events(
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
    ) -> VecDeque<Result<ResponseEvent, ConformanceError>> {
        vec![
            Ok(ResponseEvent::Head { status, headers }),
            Ok(ResponseEvent::Chunk(bytes)),
            Ok(ResponseEvent::Completed(vec![7, 8, 9])),
        ]
        .into()
    }
    #[test]
    fn elevenlabs_source_oracle_covers_all_nodes_and_route_dispositions()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/elevenlabs/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 8);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            7
        );
        assert_eq!(ROUTES.len(), 9);
        assert_eq!(oracle()?["cases"].as_array().ok_or("cases")?.len(), 10);
        assert_eq!(
            oracle()?["preflight_errors"]
                .as_array()
                .ok_or("errors")?
                .len(),
            5
        );
        Ok(())
    }
    #[test]
    fn elevenlabs_voice_selection_requires_no_credentials_or_paid_transport()
    -> Result<(), Box<dyn Error>> {
        let oracle = oracle()?;
        for (label, id) in oracle["voice_map"].as_object().ok_or("voice map")? {
            let input = Input {
                node: "ElevenLabsVoiceSelector".into(),
                arguments: BTreeMap::from([("voice".into(), json!(label))]),
            };
            let mut transport = Transport::default();
            let result = invoke(
                &input,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                },
                0,
                &mut transport,
                &mut NativeBoundary::default(),
                &mut NativeBoundary::default(),
            )?;
            assert_eq!(result.output, Output::Voice(text(id)?.into()));
            assert!(transport.heads.is_empty());
            assert!(transport.costs.is_empty());
        }
        let registry = TypeRegistry::built_in()?;
        assert_eq!(
            registry.family(registry.resolve("ELEVENLABS_VOICE")?)?,
            ValueFamily::Model
        );
        Ok(())
    }
    #[test]
    fn elevenlabs_form_order_booleans_and_compact_voice_settings_match_source()
    -> Result<(), Box<dyn Error>> {
        let case = fixture_case("ElevenLabsSpeechToSpeech")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let observed = request(&input, &Resolution::default())?;
        assert_eq!(observed.form_fields.last().ok_or("field")?.value, "False");
        assert_eq!(
            observed.form_fields.get(1).ok_or("settings")?.value,
            "{\"stability\":0.5,\"similarity_boost\":0.75,\"style\":0.0,\"use_speaker_boost\":false,\"speed\":1.0}"
        );
        let case = fixture_case("ElevenLabsInstantVoiceClone")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let observed = request(&input, &case_resolution(&case)?)?;
        assert_eq!(
            observed
                .audio_parts
                .iter()
                .map(|part| part.file_name.as_str())
                .collect::<Vec<_>>(),
            vec!["sample_2.mp4", "sample_1.mp4"]
        );
        assert!(observed.audio_parts.iter().all(|part| part.name == "files"));
        assert_eq!(observed.form_fields.get(1).ok_or("boolean")?.value, "True");
        Ok(())
    }
    #[test]
    fn elevenlabs_seed_limits_and_prompt_whitespace_are_source_specific()
    -> Result<(), Box<dyn Error>> {
        let mut tts = fixture_input("ElevenLabsTextToSpeech")?;
        tts.arguments.insert("seed".into(), json!(u32::MAX));
        assert!(request(&tts, &Resolution::default()).is_err());
        tts.arguments.insert("seed".into(), json!(i32::MAX));
        let body = request(&tts, &Resolution::default())?.body.ok_or("body")?;
        assert_eq!(body["text"], "  hello café ☁  ");
        assert_eq!(body["language_code"], " en ");
        for node in ["ElevenLabsSpeechToSpeech", "ElevenLabsTextToDialogue"] {
            let mut input = fixture_input(node)?;
            input.arguments.insert("seed".into(), json!(u32::MAX));
            assert!(request(&input, &Resolution::default()).is_ok());
            input
                .arguments
                .insert("seed".into(), json!(u64::from(u32::MAX) + 1));
            assert!(request(&input, &Resolution::default()).is_err());
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_python_whitespace_text_rejects_before_cost_and_preparation()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "ElevenLabsTextToSpeech",
            "ElevenLabsTextToSoundEffects",
            "ElevenLabsTextToDialogue",
        ] {
            let last_segment = if node == "ElevenLabsTextToDialogue" {
                10
            } else {
                1
            };
            for segment in 1..=last_segment {
                for whitespace in [
                    "\u{001c}",
                    "\u{001d}",
                    "\u{001e}",
                    "\u{001f}",
                    " \t\n\u{001c}\u{001d}\u{001e}\u{001f}\u{0085}\u{00a0}",
                ] {
                    let mut input = fixture_input(node)?;
                    if node == "ElevenLabsTextToDialogue" {
                        let voice = input.arguments["inputs"]["voice1"].clone();
                        let mut inputs = json!({"inputs":"10"});
                        for index in 1..=10 {
                            inputs[format!("voice{index}")] = voice.clone();
                            inputs[format!("text{index}")] = json!("Original segment");
                        }
                        inputs[format!("text{segment}")] = json!(whitespace);
                        input.arguments.insert("inputs".into(), inputs);
                    } else {
                        input.arguments.insert("text".into(), json!(whitespace));
                    }
                    let mut transport = Transport::default();
                    let mut preparation = NativeBoundary::default();
                    let mut encoder = NativeBoundary::default();
                    assert!(matches!(
                        invoke(
                            &input,
                            credentials(),
                            500,
                            &mut transport,
                            &mut preparation,
                            &mut encoder
                        ),
                        Err(ElevenLabsError::Input("Empty ElevenLabs prompt"))
                    ));
                    assert!(transport.heads.is_empty());
                    assert!(transport.bodies.is_empty());
                    assert!(transport.costs.is_empty());
                    assert_eq!(preparation.uploads, 0);
                    assert_eq!(preparation.preparations, 0);
                    assert_eq!(preparation.uuid_calls, 0);
                    assert_eq!(encoder.encodes, 0);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_nonempty_text_preserves_control_wrapping_zero_width_and_bom()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "ElevenLabsTextToSpeech",
            "ElevenLabsTextToSoundEffects",
            "ElevenLabsTextToDialogue",
        ] {
            for original in ["\u{001c} \tOriginal text\n\u{001f}", "\u{200b}", "\u{feff}"] {
                let mut input = fixture_input(node)?;
                if node == "ElevenLabsTextToDialogue" {
                    let voice = input.arguments["inputs"]["voice1"].clone();
                    input.arguments.insert(
                        "inputs".into(),
                        json!({"inputs":"2","voice1":voice,"text1":original,"voice2":voice,"text2":original}),
                    );
                } else {
                    input.arguments.insert("text".into(), json!(original));
                }
                let mut transport = Transport {
                    events: events(200, audio(), vec![]),
                    ..Transport::default()
                };
                let mut preparation = NativeBoundary::default();
                let mut encoder = NativeBoundary::default();
                let result = invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut preparation,
                    &mut encoder,
                )?;
                let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
                if node == "ElevenLabsTextToDialogue" {
                    for segment in body["inputs"].as_array().ok_or("segments")? {
                        assert_eq!(segment["text"], original);
                    }
                } else {
                    assert_eq!(body["text"], original);
                }
                assert_eq!(result.response_bytes, audio());
                assert_eq!(transport.costs.len(), 1);
                assert_eq!(preparation.uploads, 0);
                assert_eq!(preparation.preparations, 0);
                assert_eq!(encoder.encodes, 0);
            }
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_language_omission_preserves_source_upload_and_original_bytes()
    -> Result<(), Box<dyn Error>> {
        #[derive(Default)]
        struct Upload {
            calls: usize,
        }
        impl ElevenLabsAudioPreparation for Upload {
            fn prepare_mp4_aac(&mut self, _part: &AudioPart) -> Result<Vec<u8>, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
            fn upload_audio(&mut self, source: &Value) -> Result<String, ConformanceError> {
                self.calls += 1;
                assert_eq!(source["sample_rate"], 48000);
                Ok("https://storage.fixture.invalid/audio".into())
            }
            fn clone_uuid(&mut self) -> Result<String, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
            fn clone_file_order(
                &mut self,
                _input: &Input,
            ) -> Result<Vec<String>, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
        }
        for node in [
            "ElevenLabsSpeechToText",
            "ElevenLabsTextToSpeech",
            "ElevenLabsTextToDialogue",
        ] {
            for (original, included) in [
                ("\u{001c}", false),
                ("\u{001d}", false),
                ("\u{001e}", false),
                ("\u{001f}", false),
                (" \t\n\u{001c}\u{001d}\u{001e}\u{001f}", false),
                ("\u{200b}", true),
                ("\u{feff}", true),
                ("\u{001c} en \u{001f}", true),
            ] {
                let mut input = fixture_input(node)?;
                input
                    .arguments
                    .insert("language_code".into(), json!(original));
                let response = if node == "ElevenLabsSpeechToText" {
                    serde_json::to_vec(
                        &json!({"language_code":"en","text":"Original transcript","words":[]}),
                    )?
                } else {
                    audio()
                };
                let mut transport = Transport {
                    events: events(200, response, vec![]),
                    ..Transport::default()
                };
                let mut upload = Upload::default();
                let mut encoder = CanonicalMultipartEncoder {
                    boundary: "actual-canonical-boundary".into(),
                };
                invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut upload,
                    &mut encoder,
                )?;
                assert_eq!(transport.costs.len(), 1);
                let body = transport.bodies.first().ok_or("body")?;
                if node == "ElevenLabsSpeechToText" {
                    assert_eq!(upload.calls, 1);
                    let marker = b"name=\"language_code\"";
                    assert_eq!(
                        body.windows(marker.len()).any(|window| window == marker),
                        included
                    );
                    if included {
                        let field = format!("name=\"language_code\"\r\n\r\n{original}\r\n");
                        assert!(
                            body.windows(field.len())
                                .any(|window| window == field.as_bytes())
                        );
                    }
                    let mut rejected = Transport::default();
                    let mut preparation = NativeBoundary::default();
                    let mut failed_encoder = NativeBoundary::default();
                    assert!(matches!(
                        invoke(
                            &input,
                            credentials(),
                            500,
                            &mut rejected,
                            &mut preparation,
                            &mut failed_encoder
                        ),
                        Err(ElevenLabsError::Transport(ConformanceError::HostFailure))
                    ));
                    assert_eq!(rejected.costs.len(), 1);
                    assert_eq!(preparation.uploads, 1);
                    assert_eq!(failed_encoder.encodes, 0);
                    assert!(rejected.bodies.iter().all(Vec::is_empty));
                } else {
                    assert_eq!(upload.calls, 0);
                    let body: Value = serde_json::from_slice(body)?;
                    assert_eq!(body.get("language_code").is_some(), included);
                    if included {
                        assert_eq!(body["language_code"], original);
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_transcript_defaults_omit_null_and_preserve_python_unicode_json()
    -> Result<(), Box<dyn Error>> {
        let result = result_value(
            "ElevenLabsSpeechToText",
            serde_json::to_vec(
                &json!({"language_code":"en","text":"source transcript","words":[{"text":"café ☁","start":0.0,"end":1.5}]}),
            )?,
            vec![42],
        )?;
        assert_eq!(
            result.output,
            Output::Transcript(serde_json::from_value(
                fixture_case("ElevenLabsSpeechToText")?["outputs"].clone()
            )?)
        );
        let result = result_value(
            "ElevenLabsSpeechToText",
            br#"{"language_code":"en","text":"","words":null}"#.to_vec(),
            vec![42],
        )?;
        assert_eq!(
            result.output,
            Output::Transcript(vec!["".into(), "en".into(), "[]".into()])
        );
        assert_eq!(ascii_json("😀"), "\\ud83d\\ude00");
        assert!(
            result_value(
                "ElevenLabsSpeechToText",
                br#"{"text":"missing language"}"#.to_vec(),
                vec![42]
            )
            .is_err()
        );
        assert!(
            result_value(
                "ElevenLabsInstantVoiceClone",
                br#"{"voice_id":""}"#.to_vec(),
                vec![42]
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn elevenlabs_raw_audio_transport_preserves_bytes_cost_and_hidden_auth()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "ElevenLabsTextToSpeech",
            "ElevenLabsTextToSoundEffects",
            "ElevenLabsTextToDialogue",
        ] {
            let input = fixture_input(node)?;
            let mut transport = Transport {
                events: events(200, audio(), vec![]),
                ..Transport::default()
            };
            let result = invoke(
                &input,
                credentials(),
                500_000,
                &mut transport,
                &mut NativeBoundary::default(),
                &mut NativeBoundary::default(),
            )?;
            assert_eq!(result.response_bytes, audio());
            assert_eq!(result.response_receipt, vec![7, 8, 9]);
            assert_eq!(result.output, Output::Audio);
            assert_eq!(transport.costs.len(), 1);
            assert_eq!(
                transport.heads.first().ok_or("head")?.secret_id.as_deref(),
                Some("sim_org.auth_token")
            );
            assert!(
                transport
                    .heads
                    .first()
                    .ok_or("head")?
                    .path
                    .contains("output_format=")
            );
            let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
            assert_eq!(
                body,
                request(&input, &Resolution::default())?
                    .body
                    .ok_or("body")?
            );
        }
        let head = request_head(
            "/proxy/elevenlabs/v1/sound-generation",
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            true,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.api_key"));
        Ok(())
    }
    #[test]
    fn elevenlabs_auth_and_cost_denial_precede_native_preparation() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("ElevenLabsAudioIsolation")?;
        let mut preparation = NativeBoundary::default();
        let mut encoder = NativeBoundary::default();
        let mut transport = Transport::default();
        assert!(
            invoke(
                &input,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                500,
                &mut transport,
                &mut preparation,
                &mut encoder
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        transport.deny_cost = true;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                500,
                &mut transport,
                &mut preparation,
                &mut encoder
            ),
            Err(ElevenLabsError::Transport(ConformanceError::InvalidCost))
        ));
        assert_eq!(preparation.preparations, 0);
        assert_eq!(encoder.encodes, 0);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn elevenlabs_source_errors_precede_audio_upload_and_paid_requests()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut preparation = NativeBoundary::default();
            let mut encoder = NativeBoundary::default();
            let mut transport = Transport::default();
            assert!(
                invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut preparation,
                    &mut encoder
                )
                .is_err()
            );
            assert!(transport.heads.is_empty());
            assert_eq!(preparation.uploads, 0);
            assert_eq!(preparation.preparations, 0);
            assert_eq!(encoder.encodes, 0);
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_native_encoding_and_upload_failures_do_not_send_fabricated_media()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "ElevenLabsAudioIsolation",
            "ElevenLabsSpeechToSpeech",
            "ElevenLabsInstantVoiceClone",
            "ElevenLabsSpeechToText",
        ] {
            let input = fixture_input(node)?;
            let mut preparation = NativeBoundary::default();
            let mut encoder = NativeBoundary::default();
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut preparation,
                    &mut encoder
                ),
                Err(ElevenLabsError::Transport(ConformanceError::HostFailure))
            ));
            assert!(transport.bodies.iter().all(Vec::is_empty));
            assert_eq!(encoder.encodes, 0);
            if node == "ElevenLabsSpeechToText" {
                assert_eq!(preparation.uploads, 1);
            } else {
                assert_eq!(preparation.preparations, 1);
            }
        }
        Ok(())
    }
    #[test]
    fn elevenlabs_retry_replays_body_and_delegates_retry_after_with_fresh_handle()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("ElevenLabsTextToSpeech")?;
        let mut transport = Transport {
            events: events(
                429,
                b"rate limited".to_vec(),
                vec![Header {
                    name: "Retry-After".into(),
                    value: "2".into(),
                }],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(200, audio(), vec![]));
        let result = invoke(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut NativeBoundary::default(),
            &mut NativeBoundary::default(),
        )?;
        assert_eq!(result.response_bytes, audio());
        assert_eq!(transport.delays, vec![2000]);
        assert_eq!(transport.bodies.len(), 2);
        assert_eq!(transport.bodies.first(), transport.bodies.get(1));
        let mut transport = Transport {
            events: events(503, vec![1], vec![]),
            same_retry_handle: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                500,
                &mut transport,
                &mut NativeBoundary::default(),
                &mut NativeBoundary::default()
            ),
            Err(ElevenLabsError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn elevenlabs_cancel_worker_loss_malformed_and_oversized_streams_fail()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("ElevenLabsTextToSpeech")?;
        for events in [
            vec![Err(ConformanceError::WorkerLost)],
            vec![Ok(ResponseEvent::Cancelled)],
            vec![Ok(ResponseEvent::Chunk(vec![1]))],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES + 1])),
            ],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Completed(vec![])),
            ],
        ] {
            let mut transport = Transport {
                events: events.into(),
                ..Transport::default()
            };
            assert!(
                invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut NativeBoundary::default(),
                    &mut NativeBoundary::default()
                )
                .is_err()
            );
        }
        let mut transport = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                500,
                &mut transport,
                &mut NativeBoundary::default(),
                &mut NativeBoundary::default()
            ),
            Err(ElevenLabsError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let head = request_head(
            "/proxy/elevenlabs/v1/sound-generation",
            HttpMethod::Post,
            credentials(),
            true,
        )?;
        let mut transport = Transport {
            events: events(200, vec![0; 10], vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, b"request", None, 5, GENERATION_RETRY, &mut transport),
            Err(ElevenLabsError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn elevenlabs_actual_audio_reaches_decoder_and_failure_publishes_nothing()
    -> Result<(), Box<dyn Error>> {
        struct Decoder;
        impl ElevenLabsMaterializer for Decoder {
            fn materialize(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                assert_eq!(result.response_bytes, audio());
                assert_eq!(result.response_receipt, vec![7, 8, 9]);
                assert!(result.response_bytes.starts_with(b"RIFF"));
                Err(ConformanceError::HostFailure)
            }
        }
        #[derive(Default)]
        struct Sink {
            batches: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
                self.batches += 1;
                Ok(())
            }
        }
        let input = fixture_input("ElevenLabsTextToSpeech")?;
        let mut transport = Transport {
            events: events(200, audio(), vec![]),
            ..Transport::default()
        };
        let mut preparation = NativeBoundary::default();
        let mut encoder = NativeBoundary::default();
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                credentials(),
                500,
                ExecutionAdapters {
                    transport: &mut transport,
                    preparation: &mut preparation,
                    encoder: &mut encoder,
                    materializer: &mut Decoder,
                    sink: &mut sink
                }
            ),
            Err(ElevenLabsError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.batches, 0);
        Ok(())
    }

    #[test]
    fn elevenlabs_canonical_multipart_preserves_real_bytes_repeated_order_and_bounds()
    -> Result<(), Box<dyn Error>> {
        let fields = vec![
            MultipartField {
                name: "remove_background_noise".into(),
                file_name: None,
                media_type: Some("text/plain; charset=utf-8".into()),
                bytes: b"True".to_vec(),
            },
            MultipartField {
                name: "files".into(),
                file_name: Some("sample_2.wav".into()),
                media_type: Some("audio/wav".into()),
                bytes: audio(),
            },
            MultipartField {
                name: "files".into(),
                file_name: Some("sample_1.wav".into()),
                media_type: Some("audio/wav".into()),
                bytes: audio(),
            },
        ];
        let mut encoder = CanonicalMultipartEncoder {
            boundary: "sim-elevenlabs-boundary".into(),
        };
        let declared = encoder.content_type()?;
        let encoded = encoder.encode(&fields)?;
        assert_eq!(declared, encoded.content_type);
        let wav = audio();
        assert_eq!(
            encoded
                .bytes
                .windows(wav.len())
                .filter(|window| *window == wav.as_slice())
                .count(),
            2
        );
        let name2 = b"filename=\"sample_2.wav\"";
        let name1 = b"filename=\"sample_1.wav\"";
        let first = encoded
            .bytes
            .windows(name2.len())
            .position(|window| window == name2)
            .ok_or("first file")?;
        let second = encoded
            .bytes
            .windows(name1.len())
            .position(|window| window == name1)
            .ok_or("second file")?;
        assert!(first < second);
        assert!(
            encoded
                .bytes
                .windows(b"\r\n\r\nTrue\r\n".len())
                .any(|window| window == b"\r\n\r\nTrue\r\n")
        );
        let mut limits = crate::common::TransportLimits::conformance();
        limits.maximum_body_bytes = 10;
        assert_eq!(
            MultipartField::encode_checked(&fields, "boundary", limits),
            Err(ConformanceError::BodyLimit)
        );
        assert_eq!(
            MultipartField::encode_checked(
                &fields,
                "invalid\r\nboundary",
                crate::common::TransportLimits::conformance()
            ),
            Err(ConformanceError::InvalidRequest)
        );
        Ok(())
    }
    #[test]
    fn elevenlabs_transcript_runs_through_canonical_multipart_and_injected_upload()
    -> Result<(), Box<dyn Error>> {
        struct Upload;
        impl ElevenLabsAudioPreparation for Upload {
            fn prepare_mp4_aac(&mut self, _part: &AudioPart) -> Result<Vec<u8>, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
            fn upload_audio(&mut self, source: &Value) -> Result<String, ConformanceError> {
                assert_eq!(source["sample_rate"], 48000);
                Ok("https://storage.fixture.invalid/audio".into())
            }
            fn clone_uuid(&mut self) -> Result<String, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
            fn clone_file_order(
                &mut self,
                _input: &Input,
            ) -> Result<Vec<String>, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
        }
        let input = fixture_input("ElevenLabsSpeechToText")?;
        let response = serde_json::to_vec(
            &json!({"language_code":"en","text":"source transcript","words":[{"text":"café ☁","start":0.0,"end":1.5}]}),
        )?;
        let mut transport = Transport {
            events: events(200, response, vec![]),
            ..Transport::default()
        };
        let mut encoder = CanonicalMultipartEncoder {
            boundary: "actual-canonical-boundary".into(),
        };
        let result = invoke(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut Upload,
            &mut encoder,
        )?;
        assert_eq!(
            result.output,
            Output::Transcript(serde_json::from_value(
                fixture_case("ElevenLabsSpeechToText")?["outputs"].clone()
            )?)
        );
        let head = transport.heads.first().ok_or("head")?;
        assert!(
            head.headers
                .iter()
                .any(|header| header.name == "Content-Type"
                    && header.value == "multipart/form-data; boundary=actual-canonical-boundary")
        );
        let body = transport.bodies.first().ok_or("body")?;
        assert!(
            body.windows(b"\r\n\r\nTrue\r\n".len())
                .any(|window| window == b"\r\n\r\nTrue\r\n")
        );
        Ok(())
    }
    #[test]
    fn elevenlabs_multipart_head_precedes_cost_and_native_preparation() -> Result<(), Box<dyn Error>>
    {
        for node in [
            "ElevenLabsSpeechToText",
            "ElevenLabsAudioIsolation",
            "ElevenLabsInstantVoiceClone",
            "ElevenLabsSpeechToSpeech",
        ] {
            let input = fixture_input(node)?;
            for deny_cost in [true, false] {
                let mut preparation = NativeBoundary::default();
                let mut encoder = CanonicalMultipartEncoder {
                    boundary: "authorized-elevenlabs-boundary".into(),
                };
                let mut transport = Transport {
                    deny_cost,
                    expected_content_type_at_cost: Some(encoder.content_type()?),
                    ..Transport::default()
                };
                let result = invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut preparation,
                    &mut encoder,
                );
                if deny_cost {
                    assert!(matches!(
                        result,
                        Err(ElevenLabsError::Transport(ConformanceError::InvalidCost))
                    ));
                    assert_eq!(preparation.uploads, 0);
                    assert_eq!(preparation.preparations, 0);
                    assert_eq!(preparation.uuid_calls, 0);
                } else {
                    assert!(matches!(
                        result,
                        Err(ElevenLabsError::Transport(ConformanceError::HostFailure))
                    ));
                    assert_eq!(
                        preparation.uploads,
                        usize::from(node == "ElevenLabsSpeechToText")
                    );
                    assert_eq!(
                        preparation.preparations,
                        usize::from(node != "ElevenLabsSpeechToText")
                    );
                }
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.bodies.iter().all(Vec::is_empty));
            }
        }
        Ok(())
    }

    #[test]
    fn elevenlabs_unbound_invalid_or_cancelled_multipart_head_sends_nothing()
    -> Result<(), Box<dyn Error>> {
        struct UnboundEncoder;
        impl ElevenLabsMultipartEncoder for UnboundEncoder {
            fn encode(
                &mut self,
                _fields: &[MultipartField],
            ) -> Result<MultipartBody, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
        }
        for node in [
            "ElevenLabsSpeechToText",
            "ElevenLabsAudioIsolation",
            "ElevenLabsInstantVoiceClone",
            "ElevenLabsSpeechToSpeech",
        ] {
            let input = fixture_input(node)?;
            for cancelled in [false, true] {
                let mut preparation = NativeBoundary::default();
                let mut transport = Transport {
                    cancelled,
                    ..Transport::default()
                };
                let result = invoke(
                    &input,
                    credentials(),
                    500,
                    &mut transport,
                    &mut preparation,
                    &mut UnboundEncoder,
                );
                if cancelled {
                    assert!(matches!(
                        result,
                        Err(ElevenLabsError::Transport(ConformanceError::Cancelled))
                    ));
                } else {
                    assert!(matches!(
                        result,
                        Err(ElevenLabsError::Transport(ConformanceError::HostFailure))
                    ));
                }
                assert!(transport.heads.is_empty());
                assert!(transport.costs.is_empty());
                assert!(transport.bodies.is_empty());
                assert_eq!(
                    preparation.uploads + preparation.preparations + preparation.uuid_calls,
                    0
                );
            }
            for boundary in [String::new(), "invalid\r\nboundary".into(), "a".repeat(71)] {
                let mut transport = Transport::default();
                let mut preparation = NativeBoundary::default();
                assert!(matches!(
                    invoke(
                        &input,
                        credentials(),
                        500,
                        &mut transport,
                        &mut preparation,
                        &mut CanonicalMultipartEncoder { boundary }
                    ),
                    Err(ElevenLabsError::Transport(ConformanceError::InvalidRequest))
                ));
                assert!(transport.heads.is_empty());
                assert!(transport.costs.is_empty());
                assert!(transport.bodies.is_empty());
                assert_eq!(
                    preparation.uploads + preparation.preparations + preparation.uuid_calls,
                    0
                );
            }
        }
        for node in [
            "ElevenLabsTextToSpeech",
            "ElevenLabsTextToSoundEffects",
            "ElevenLabsTextToDialogue",
        ] {
            let mut transport = Transport {
                events: events(200, audio(), vec![]),
                expected_content_type_at_cost: Some("application/json".into()),
                ..Transport::default()
            };
            assert_eq!(
                invoke(
                    &fixture_input(node)?,
                    credentials(),
                    500,
                    &mut transport,
                    &mut NativeBoundary::default(),
                    &mut UnboundEncoder
                )?
                .response_bytes,
                audio()
            );
        }
        Ok(())
    }

    #[derive(Default)]
    struct UploadedAudio {
        uploads: usize,
    }
    impl ElevenLabsAudioPreparation for UploadedAudio {
        fn prepare_mp4_aac(&mut self, _part: &AudioPart) -> Result<Vec<u8>, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn upload_audio(&mut self, source: &Value) -> Result<String, ConformanceError> {
            assert_eq!(source["sample_rate"], 48000);
            self.uploads += 1;
            Ok("https://storage.fixture.invalid/audio".into())
        }
        fn clone_uuid(&mut self) -> Result<String, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn clone_file_order(&mut self, _input: &Input) -> Result<Vec<String>, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
    }

    #[test]
    fn elevenlabs_changed_multipart_header_is_rejected_before_body_submission()
    -> Result<(), Box<dyn Error>> {
        struct ChangedBoundary;
        impl ElevenLabsMultipartEncoder for ChangedBoundary {
            fn content_type(&self) -> Result<String, ConformanceError> {
                CanonicalMultipartEncoder {
                    boundary: "authorized-boundary".into(),
                }
                .content_type()
            }
            fn encode(
                &mut self,
                fields: &[MultipartField],
            ) -> Result<MultipartBody, ConformanceError> {
                CanonicalMultipartEncoder {
                    boundary: "changed-boundary".into(),
                }
                .encode(fields)
            }
        }
        let mut preparation = UploadedAudio::default();
        let mut transport = Transport {
            expected_content_type_at_cost: Some(ChangedBoundary.content_type()?),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &fixture_input("ElevenLabsSpeechToText")?,
                credentials(),
                500,
                &mut transport,
                &mut preparation,
                &mut ChangedBoundary
            ),
            Err(ElevenLabsError::Input(
                "Canonical multipart Content-Type changed after authorization"
            ))
        ));
        assert_eq!(preparation.uploads, 1);
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }

    #[test]
    fn elevenlabs_multipart_retry_preserves_authorized_header_and_canonical_body()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("ElevenLabsSpeechToText")?;
        let mut encoder = CanonicalMultipartEncoder {
            boundary: "authorized-retry-boundary".into(),
        };
        let content_type = encoder.content_type()?;
        let mut preparation = UploadedAudio::default();
        let mut transport = Transport {
            events: events(503, b"unavailable".to_vec(), vec![]),
            expected_content_type_at_cost: Some(content_type.clone()),
            ..Transport::default()
        };
        let response = serde_json::to_vec(
            &json!({"language_code":"en","text":"source transcript","words":[]}),
        )?;
        transport
            .events
            .extend(events(200, response.clone(), vec![]));
        let result = invoke(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut preparation,
            &mut encoder,
        )?;
        assert_eq!(result.response_bytes, response);
        assert_eq!(result.response_receipt, vec![7, 8, 9]);
        assert_eq!(preparation.uploads, 1);
        assert_eq!(transport.heads.len(), 2);
        assert_eq!(transport.costs.len(), 1);
        for head in &transport.heads {
            assert!(
                head.headers
                    .iter()
                    .any(|header| header.name == "Content-Type" && header.value == content_type)
            );
        }
        assert_eq!(transport.bodies.first(), transport.bodies.get(1));
        let source_request = request(
            &input,
            &case_resolution(&fixture_case("ElevenLabsSpeechToText")?)?,
        )?;
        let fields = source_request
            .form_fields
            .iter()
            .map(|field| MultipartField {
                name: field.name.clone(),
                file_name: None,
                media_type: Some("text/plain; charset=utf-8".into()),
                bytes: field.value.as_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            transport.bodies.first(),
            Some(&encoder.encode(&fields)?.bytes)
        );
        Ok(())
    }

    #[test]
    fn elevenlabs_source_unit_badges_do_not_replace_host_maximum_cost() -> Result<(), Box<dyn Error>>
    {
        let oracle = oracle()?;
        for claim in NODES {
            let observed = price_badge_microunits(claim.node_identifier)?;
            let expression = &oracle["contracts"][claim.node_identifier]["price_badge_expr"];
            if expression.is_null() {
                assert_eq!(observed, None);
            } else {
                let badge: Value = serde_json::from_str(text(expression)?)?;
                let usd = badge["usd"].as_f64().ok_or("price")?;
                assert_eq!(observed, Some((usd * 1_000_000.0).round() as u64));
            }
        }
        Ok(())
    }
}
