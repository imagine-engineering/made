use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, Progress, ProviderTransport, RequestHead, ResponseEvent, StreamHandle,
    TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    sync::OnceLock,
};
pub const NAMESPACE: &str = "zed.sim.provider.wan";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "2c6f0c74dde9e2000b1c49635d9abe7144f2bb52841d3a2c00a46616ce8bbacb";
pub const MAXIMUM_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0220",
        node_identifier: "HappyHorseImageToVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0221",
        node_identifier: "HappyHorseReferenceVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0222",
        node_identifier: "HappyHorseTextToVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0223",
        node_identifier: "HappyHorseVideoEditApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0751",
        node_identifier: "Wan2ImageToVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0752",
        node_identifier: "Wan2ReferenceVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0753",
        node_identifier: "Wan2TextToVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0754",
        node_identifier: "Wan2VideoContinuationApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0755",
        node_identifier: "Wan2VideoEditApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0769",
        node_identifier: "WanImageToImageApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0771",
        node_identifier: "WanImageToVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0778",
        node_identifier: "WanReferenceVideoApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0782",
        node_identifier: "WanTextToImageApi",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0783",
        node_identifier: "WanTextToVideoApi",
    },
];
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0201",
        method: "POST",
        path: "/proxy/wan/api/v1/services/aigc/image2image/image-synthesis",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0202",
        method: "POST",
        path: "/proxy/wan/api/v1/services/aigc/text2image/image-synthesis",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0203",
        method: "POST",
        path: "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0204",
        method: "GET",
        path: "/proxy/wan/api/v1/tasks/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0205",
        method: "GET",
        path: "/proxy/wan/api/v1/tasks/{initial_response.output.task_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum WanError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Http {
        status: u16,
        retry_after: Option<String>,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for WanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Wan HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for WanError {}
impl From<ConformanceError> for WanError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for WanError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<&'static Value, WanError> {
    static SOURCE: OnceLock<Result<Value, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| {
            serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/wan/source-contract.json"
            )))
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| WanError::Provider(format!("Invalid pinned Wan source contract: {error}")))
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, WanError> {
    if !NODES.iter().any(|node| node.node_identifier == input.node) {
        return Err(WanError::Input("Unknown Wan node"));
    }
    let source = oracle()?;
    let contract = &source["contracts"][&input.node];
    let names = contract["arguments"]
        .as_array()
        .ok_or(WanError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(WanError::Input("Unknown Wan source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(WanError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            input.arguments.insert(
                name.into(),
                contract["defaults"]
                    .get(name)
                    .ok_or(WanError::Input("Missing Wan source argument"))?
                    .clone(),
            );
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, WanError> {
    input
        .arguments
        .get(name)
        .ok_or(WanError::Input("Missing Wan argument"))
}
fn text(value: &Value) -> Result<&str, WanError> {
    value
        .as_str()
        .ok_or(WanError::Input("Expected source string"))
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
fn receipt(value: &[u8]) -> Result<(), ConformanceError> {
    if value.is_empty() || value.len() > MAXIMUM_RECEIPT_BYTES {
        Err(ConformanceError::InvalidResponse)
    } else {
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiKind {
    Generation,
    ImagePoll,
    VideoPoll,
}
impl ApiKind {
    fn name(self) -> &'static str {
        match self {
            Self::Generation => "Generation",
            Self::ImagePoll => "ImagePoll",
            Self::VideoPoll => "VideoPoll",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeOperation {
    ImageCount,
    ImageBatchItems,
    MappingItems,
    EncodeImage,
    EncodeAudio,
    ValidateString,
    ValidateAudioDuration,
    ValidateVideoDuration,
    ValidateImageDimensions,
    ValidateImageAspectRatio,
    UploadImage,
    UploadAudio,
    UploadVideo,
    ApiConstruct,
}
impl NativeOperation {
    fn name(self) -> &'static str {
        match self {
            Self::ImageCount => "get_number_of_images",
            Self::ImageBatchItems => "image_batch_items",
            Self::MappingItems => "mapping_items",
            Self::EncodeImage => "tensor_to_base64_string",
            Self::EncodeAudio => "audio_to_base64_string",
            Self::ValidateString => "validate_string",
            Self::ValidateAudioDuration => "validate_audio_duration",
            Self::ValidateVideoDuration => "validate_video_duration",
            Self::ValidateImageDimensions => "validate_image_dimensions",
            Self::ValidateImageAspectRatio => "validate_image_aspect_ratio",
            Self::UploadImage => "upload_image_to_simapi",
            Self::UploadAudio => "upload_audio_to_simapi",
            Self::UploadVideo => "upload_video_to_simapi",
            Self::ApiConstruct => "api_construct",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarOperation {
    Size,
    EditDuration,
    DurationEstimate,
    PythonString,
    Status,
}
#[derive(Clone, Debug)]
pub struct PreparedValue {
    pub value: Value,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct PreparedBytes {
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
pub trait WanPreparation {
    fn native(
        &mut self,
        _kind: NativeOperation,
        _value: &Value,
        _options: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn scalar(
        &mut self,
        _kind: ScalarOperation,
        _value: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_response(
        &mut self,
        _kind: ApiKind,
        _value: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn encode_json(&mut self, _value: &Value) -> Result<PreparedBytes, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn failed_poll_message(
        &mut self,
        _raw_bytes: &[u8],
        _value: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputKind {
    Image,
    Video,
}
impl OutputKind {
    fn type_name(self) -> &'static str {
        match self {
            Self::Image => "IMAGE",
            Self::Video => "VIDEO",
        }
    }
}
#[derive(Clone, Debug)]
pub struct Request {
    pub node: String,
    pub generate_path: String,
    pub body: Value,
    pub preparation_receipts: Vec<Vec<u8>>,
    pub output_kind: OutputKind,
    pub poll_interval_milliseconds: u64,
    pub estimated_duration: Option<u64>,
}
fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, WanError> {
    value
        .get(name)
        .ok_or(WanError::Input("Missing source model field"))
}
fn native(
    kind: NativeOperation,
    value: &Value,
    options: Value,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, WanError> {
    let result = owner.native(kind, value, &options)?;
    receipt(&result.receipt)?;
    receipts.push(result.receipt);
    Ok(result.value)
}
fn scalar(
    kind: ScalarOperation,
    value: &Value,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, WanError> {
    let result = owner.scalar(kind, value)?;
    receipt(&result.receipt)?;
    receipts.push(result.receipt);
    Ok(result.value)
}
fn construct(
    name: &str,
    value: Value,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, WanError> {
    native(
        NativeOperation::ApiConstruct,
        &value,
        json!({"model":name}),
        owner,
        receipts,
    )
}
fn array(value: &Value) -> Result<&Vec<Value>, WanError> {
    value
        .as_array()
        .ok_or(WanError::Input("Expected canonical source sequence"))
}
fn mapping(
    value: &Value,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Vec<Value>, WanError> {
    let result = native(
        NativeOperation::MappingItems,
        value,
        json!({}),
        owner,
        receipts,
    )?;
    array(&result)?
        .iter()
        .map(|pair| {
            array(pair)?
                .get(1)
                .cloned()
                .ok_or(WanError::Input("Invalid ordered source mapping"))
        })
        .collect()
}
fn upload_media(
    kind: NativeOperation,
    value: &Value,
    media_type: &str,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, WanError> {
    let url = native(kind, value, json!({}), owner, receipts)?;
    construct(
        "Wan27MediaItem",
        json!({"type":media_type,"url":url}),
        owner,
        receipts,
    )
}
fn validate_prompt(
    value: &Value,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<(), WanError> {
    native(
        NativeOperation::ValidateString,
        value,
        json!({"strip_whitespace":false,"min_length":1}),
        owner,
        receipts,
    )?;
    Ok(())
}
fn validate_video(
    value: &Value,
    minimum: u64,
    maximum: u64,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<(), WanError> {
    native(
        NativeOperation::ValidateVideoDuration,
        value,
        json!({"min_duration":minimum,"max_duration":maximum}),
        owner,
        receipts,
    )?;
    Ok(())
}
fn validate_horse_image(
    value: &Value,
    minimum: u64,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<(), WanError> {
    native(
        NativeOperation::ValidateImageDimensions,
        value,
        json!({"min_width":minimum,"min_height":minimum}),
        owner,
        receipts,
    )?;
    native(
        NativeOperation::ValidateImageAspectRatio,
        value,
        json!({"min_ratio":[1,2.5],"max_ratio":[2.5,1],"strict":false}),
        owner,
        receipts,
    )?;
    Ok(())
}
fn audio_url(
    input: &Input,
    low: f64,
    high: f64,
    uploaded: bool,
    owner: &mut dyn WanPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, WanError> {
    let audio = input
        .arguments
        .get("audio")
        .filter(|audio| !audio.is_null());
    if let Some(audio) = audio {
        native(
            NativeOperation::ValidateAudioDuration,
            audio,
            json!({"min_duration":low,"max_duration":high}),
            owner,
            receipts,
        )?;
        let options = if uploaded {
            json!({"container_format":"mp3","codec_name":"libmp3lame","mime_type":"audio/mpeg"})
        } else {
            json!({"container_format":"mp3","codec_name":"libmp3lame"})
        };
        let result = native(
            if uploaded {
                NativeOperation::UploadAudio
            } else {
                NativeOperation::EncodeAudio
            },
            audio,
            options,
            owner,
            receipts,
        )?;
        Ok(if uploaded {
            result
        } else {
            json!(format!("data:audio/mp3;base64,{}", text(&result)?))
        })
    } else {
        Ok(Value::Null)
    }
}
fn generate_path(node: &str) -> &'static str {
    match node {
        "WanTextToImageApi" => "/proxy/wan/api/v1/services/aigc/text2image/image-synthesis",
        "WanImageToImageApi" => "/proxy/wan/api/v1/services/aigc/image2image/image-synthesis",
        _ => "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
    }
}
pub fn request(input: &Input, owner: &mut dyn WanPreparation) -> Result<Request, WanError> {
    let input = normalized(input)?;
    let node = input.node.as_str();
    let model = argument(&input, "model")?;
    let mut receipts = vec![];
    let mut interval = 7000;
    let mut estimate = None;

    let (request_model, source_input, parameters) = match node {
        "WanTextToImageApi" => {
            interval = 3000;
            estimate = Some(9);
            let source_input = construct(
                "Text2ImageInputField",
                json!({"prompt":argument(&input,"prompt")?,"negative_prompt":argument(&input,"negative_prompt")?}),
                owner,
                &mut receipts,
            )?;
            let size = scalar(
                ScalarOperation::PythonString,
                argument(&input, "width")?,
                owner,
                &mut receipts,
            )?;
            let height = scalar(
                ScalarOperation::PythonString,
                argument(&input, "height")?,
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Txt2ImageParametersField",
                json!({"size":format!("{}*{}",text(&size)?,text(&height)?),"seed":argument(&input,"seed")?,"prompt_extend":argument(&input,"prompt_extend")?,"watermark":argument(&input,"watermark")?}),
                owner,
                &mut receipts,
            )?;
            ("Text2ImageTaskCreationRequest", source_input, parameters)
        }
        "WanImageToImageApi" => {
            interval = 4000;
            estimate = Some(42);
            let image = argument(&input, "image")?;
            let count = native(
                NativeOperation::ImageCount,
                image,
                json!({}),
                owner,
                &mut receipts,
            )?;
            if ![json!(1), json!(2)].contains(&count) {
                return Err(WanError::Input("Expected one or two source images"));
            }
            let batch = native(
                NativeOperation::ImageBatchItems,
                image,
                json!({}),
                owner,
                &mut receipts,
            )?;
            let mut images = vec![];
            for value in array(&batch)? {
                let encoded = native(
                    NativeOperation::EncodeImage,
                    value,
                    json!({"total_pixels":16777216}),
                    owner,
                    &mut receipts,
                )?;
                images.push(json!(format!("data:image/png;base64,{}", text(&encoded)?)));
            }
            let source_input = construct(
                "Image2ImageInputField",
                json!({"prompt":argument(&input,"prompt")?,"negative_prompt":argument(&input,"negative_prompt")?,"images":images}),
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Image2ImageParametersField",
                json!({"seed":argument(&input,"seed")?,"watermark":argument(&input,"watermark")?}),
                owner,
                &mut receipts,
            )?;
            ("Image2ImageTaskCreationRequest", source_input, parameters)
        }
        "WanTextToVideoApi" => {
            interval = 6000;
            let size = argument(&input, "size")?;
            if text(size)?.contains("480p") && model == "wan2.6-t2v" {
                return Err(WanError::Input("The Wan 2.6 model does not support 480p."));
            }
            if argument(&input, "duration")? == 15 && model == "wan2.5-t2v-preview" {
                return Err(WanError::Input(
                    "A 15-second duration is supported only by the Wan 2.6 model.",
                ));
            }
            let dimensions = scalar(ScalarOperation::Size, size, owner, &mut receipts)?;
            let dimensions = array(&dimensions)?;
            let width = dimensions.first().ok_or(WanError::Input("Missing width"))?;
            let height = dimensions.get(1).ok_or(WanError::Input("Missing height"))?;
            let audio = audio_url(&input, 3.0, 29.0, false, owner, &mut receipts)?;
            let source_input = construct(
                "Text2VideoInputField",
                json!({"prompt":argument(&input,"prompt")?,"negative_prompt":argument(&input,"negative_prompt")?,"audio_url":audio}),
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Text2VideoParametersField",
                json!({"size":format!("{}*{}",text(width)?,text(height)?),"duration":argument(&input,"duration")?,"seed":argument(&input,"seed")?,"audio":argument(&input,"generate_audio")?,"prompt_extend":argument(&input,"prompt_extend")?,"watermark":argument(&input,"watermark")?,"shot_type":argument(&input,"shot_type")?}),
                owner,
                &mut receipts,
            )?;
            ("Text2VideoTaskCreationRequest", source_input, parameters)
        }
        "WanImageToVideoApi" => {
            interval = 6000;
            let image = argument(&input, "image")?;
            if native(
                NativeOperation::ImageCount,
                image,
                json!({}),
                owner,
                &mut receipts,
            )? != 1
            {
                return Err(WanError::Input("Exactly one input image is required."));
            }
            if text(argument(&input, "resolution")?)?.contains("480P") && model == "wan2.6-i2v" {
                return Err(WanError::Input("The Wan 2.6 model does not support 480P."));
            }
            if argument(&input, "duration")? == 15 && model == "wan2.5-i2v-preview" {
                return Err(WanError::Input(
                    "A 15-second duration is supported only by the Wan 2.6 model.",
                ));
            }
            let encoded = native(
                NativeOperation::EncodeImage,
                image,
                json!({"total_pixels":4000000}),
                owner,
                &mut receipts,
            )?;
            let audio = audio_url(&input, 3.0, 29.0, false, owner, &mut receipts)?;
            let source_input = construct(
                "Image2VideoInputField",
                json!({"prompt":argument(&input,"prompt")?,"negative_prompt":argument(&input,"negative_prompt")?,"img_url":format!("data:image/png;base64,{}",text(&encoded)?),"audio_url":audio}),
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Image2VideoParametersField",
                json!({"resolution":argument(&input,"resolution")?,"duration":argument(&input,"duration")?,"seed":argument(&input,"seed")?,"audio":argument(&input,"generate_audio")?,"prompt_extend":argument(&input,"prompt_extend")?,"watermark":argument(&input,"watermark")?,"shot_type":argument(&input,"shot_type")?}),
                owner,
                &mut receipts,
            )?;
            ("Image2VideoTaskCreationRequest", source_input, parameters)
        }
        "WanReferenceVideoApi" => {
            interval = 6000;
            let references = argument(&input, "reference_videos")?;
            for value in mapping(references, owner, &mut receipts)? {
                validate_video(&value, 2, 30, owner, &mut receipts)?;
            }
            let mut urls = vec![];
            for value in mapping(references, owner, &mut receipts)? {
                urls.push(native(
                    NativeOperation::UploadVideo,
                    &value,
                    json!({}),
                    owner,
                    &mut receipts,
                )?);
            }
            let dimensions = scalar(
                ScalarOperation::Size,
                argument(&input, "size")?,
                owner,
                &mut receipts,
            )?;
            let dimensions = array(&dimensions)?;
            let width = dimensions.first().ok_or(WanError::Input("Missing width"))?;
            let height = dimensions.get(1).ok_or(WanError::Input("Missing height"))?;
            let source_input = construct(
                "Reference2VideoInputField",
                json!({"prompt":argument(&input,"prompt")?,"negative_prompt":argument(&input,"negative_prompt")?,"reference_video_urls":urls}),
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Reference2VideoParametersField",
                json!({"size":format!("{}*{}",text(width)?,text(height)?),"duration":argument(&input,"duration")?,"shot_type":argument(&input,"shot_type")?,"watermark":argument(&input,"watermark")?,"seed":argument(&input,"seed")?}),
                owner,
                &mut receipts,
            )?;
            (
                "Reference2VideoTaskCreationRequest",
                source_input,
                parameters,
            )
        }
        "Wan2TextToVideoApi" | "HappyHorseTextToVideoApi" => {
            validate_prompt(field(model, "prompt")?, owner, &mut receipts)?;
            let horse = node == "HappyHorseTextToVideoApi";
            let audio = if horse {
                Value::Null
            } else {
                audio_url(&input, 1.5, 60.0, true, owner, &mut receipts)?
            };
            field(model, "model")?;
            let mut raw_input = json!({"prompt":field(model,"prompt")?,"negative_prompt":if horse {Value::Null}else{let value=field(model,"negative_prompt")?;if truthy(value){value.clone()}else{Value::Null}}});
            if !horse {
                raw_input["audio_url"] = audio;
            }
            field(model, "model")?;
            let source_input = construct("Text2VideoInputField", raw_input, owner, &mut receipts)?;
            let mut raw_parameters = json!({"resolution":field(model,"resolution")?,"ratio":field(model,"ratio")?,"duration":field(model,"duration")?,"seed":argument(&input,"seed")?,"watermark":argument(&input,"watermark")?});
            if !horse {
                raw_parameters["prompt_extend"] = argument(&input, "prompt_extend")?.clone();
            }
            let parameters = construct(
                "Wan27Text2VideoParametersField",
                raw_parameters,
                owner,
                &mut receipts,
            )?;
            (
                "Wan27Text2VideoTaskCreationRequest",
                source_input,
                parameters,
            )
        }
        "Wan2ImageToVideoApi" | "Wan2VideoContinuationApi" | "HappyHorseImageToVideoApi" => {
            let horse = node == "HappyHorseImageToVideoApi";
            let continuation = node == "Wan2VideoContinuationApi";
            let first = argument(
                &input,
                if continuation {
                    "first_clip"
                } else {
                    "first_frame"
                },
            )?;
            if continuation {
                validate_video(first, 2, 10, owner, &mut receipts)?;
            } else if horse {
                validate_horse_image(first, 300, owner, &mut receipts)?;
            }
            let mut media = vec![upload_media(
                if continuation {
                    NativeOperation::UploadVideo
                } else {
                    NativeOperation::UploadImage
                },
                first,
                if continuation {
                    "first_clip"
                } else {
                    "first_frame"
                },
                owner,
                &mut receipts,
            )?];
            if !horse {
                if let Some(last) = input
                    .arguments
                    .get("last_frame")
                    .filter(|value| !value.is_null())
                {
                    media.push(upload_media(
                        NativeOperation::UploadImage,
                        last,
                        "last_frame",
                        owner,
                        &mut receipts,
                    )?);
                }
                if !continuation {
                    let audio = audio_url(&input, 2.0, 30.0, true, owner, &mut receipts)?;
                    if !audio.is_null() {
                        media.push(construct(
                            "Wan27MediaItem",
                            json!({"type":"driving_audio","url":audio}),
                            owner,
                            &mut receipts,
                        )?);
                    }
                }
            }
            field(model, "model")?;
            let prompt = field(model, "prompt")?;
            let negative = if horse {
                Value::Null
            } else {
                let value = field(model, "negative_prompt")?;
                if truthy(value) {
                    value.clone()
                } else {
                    Value::Null
                }
            };
            field(model, "model")?;
            let source_input = construct(
                "Wan27ImageToVideoInputField",
                json!({"prompt":if truthy(prompt){prompt.clone()}else{Value::Null},"negative_prompt":negative,"media":media}),
                owner,
                &mut receipts,
            )?;
            let mut raw_parameters = json!({"resolution":field(model,"resolution")?,"duration":field(model,"duration")?,"seed":argument(&input,"seed")?,"watermark":argument(&input,"watermark")?});
            if !horse {
                raw_parameters["prompt_extend"] = argument(&input, "prompt_extend")?.clone();
            }
            let parameters = construct(
                "Wan27ImageToVideoParametersField",
                raw_parameters,
                owner,
                &mut receipts,
            )?;
            (
                "Wan27ImageToVideoTaskCreationRequest",
                source_input,
                parameters,
            )
        }
        "Wan2VideoEditApi" | "HappyHorseVideoEditApi" => {
            let horse = node == "HappyHorseVideoEditApi";
            validate_prompt(field(model, "prompt")?, owner, &mut receipts)?;
            let video = argument(&input, "video")?;
            validate_video(
                video,
                if horse { 3 } else { 2 },
                if horse { 60 } else { 10 },
                owner,
                &mut receipts,
            )?;
            let duration = if horse {
                Value::Null
            } else {
                scalar(
                    ScalarOperation::EditDuration,
                    field(model, "duration")?,
                    owner,
                    &mut receipts,
                )?
            };
            let mut media = vec![upload_media(
                NativeOperation::UploadVideo,
                video,
                "video",
                owner,
                &mut receipts,
            )?];
            let references = model.get("reference_images").cloned().unwrap_or(json!({}));
            for value in mapping(&references, owner, &mut receipts)? {
                media.push(upload_media(
                    NativeOperation::UploadImage,
                    &value,
                    "reference_image",
                    owner,
                    &mut receipts,
                )?);
            }
            field(model, "model")?;
            let source_input = construct(
                "Wan27VideoEditInputField",
                json!({"prompt":field(model,"prompt")?,"media":media}),
                owner,
                &mut receipts,
            )?;
            let mut raw_parameters = json!({"resolution":field(model,"resolution")?,"ratio":field(model,"ratio")?,"duration":duration,"watermark":argument(&input,"watermark")?,"seed":argument(&input,"seed")?});
            if !horse {
                raw_parameters["audio_setting"] = argument(&input, "audio_setting")?.clone();
            }
            let parameters = construct(
                "Wan27VideoEditParametersField",
                raw_parameters,
                owner,
                &mut receipts,
            )?;
            (
                "Wan27VideoEditTaskCreationRequest",
                source_input,
                parameters,
            )
        }
        "Wan2ReferenceVideoApi" | "HappyHorseReferenceVideoApi" => {
            let horse = node == "HappyHorseReferenceVideoApi";
            validate_prompt(field(model, "prompt")?, owner, &mut receipts)?;
            let images = model.get("reference_images").cloned().unwrap_or(json!({}));
            let mut media = vec![];
            if horse {
                for value in mapping(&images, owner, &mut receipts)? {
                    validate_horse_image(&value, 400, owner, &mut receipts)?;
                }
            } else {
                let videos = model.get("reference_videos").cloned().unwrap_or(json!({}));
                for value in mapping(&videos, owner, &mut receipts)? {
                    media.push(upload_media(
                        NativeOperation::UploadVideo,
                        &value,
                        "reference_video",
                        owner,
                        &mut receipts,
                    )?);
                }
            }
            for value in mapping(&images, owner, &mut receipts)? {
                media.push(upload_media(
                    NativeOperation::UploadImage,
                    &value,
                    "reference_image",
                    owner,
                    &mut receipts,
                )?);
            }
            if media.is_empty() {
                return Err(WanError::Input("At least one source reference is required"));
            }
            if !horse && media.len() > 5 {
                return Err(WanError::Input(
                    "Source reference count exceeds five after uploading",
                ));
            }
            field(model, "model")?;
            let negative = if horse {
                Value::Null
            } else {
                let value = field(model, "negative_prompt")?;
                if truthy(value) {
                    value.clone()
                } else {
                    Value::Null
                }
            };
            field(model, "model")?;
            let source_input = construct(
                "Wan27ReferenceVideoInputField",
                json!({"prompt":field(model,"prompt")?,"negative_prompt":negative,"media":media}),
                owner,
                &mut receipts,
            )?;
            let parameters = construct(
                "Wan27ReferenceVideoParametersField",
                json!({"resolution":field(model,"resolution")?,"ratio":field(model,"ratio")?,"duration":field(model,"duration")?,"watermark":argument(&input,"watermark")?,"seed":argument(&input,"seed")?}),
                owner,
                &mut receipts,
            )?;
            (
                "Wan27ReferenceVideoTaskCreationRequest",
                source_input,
                parameters,
            )
        }
        _ => return Err(WanError::Input("Unknown Wan source branch")),
    };
    let model_label = if model.is_object() {
        field(model, "model")?
    } else {
        model
    };
    let body = construct(
        request_model,
        json!({"model":model_label,"input":source_input,"parameters":parameters}),
        owner,
        &mut receipts,
    )?;
    Ok(Request {
        node: input.node.clone(),
        generate_path: generate_path(node).into(),
        body,
        preparation_receipts: receipts,
        output_kind: if ["WanTextToImageApi", "WanImageToImageApi"].contains(&node) {
            OutputKind::Image
        } else {
            OutputKind::Video
        },
        poll_interval_milliseconds: interval,
        estimated_duration: estimate,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct CredentialAvailability {
    pub auth_token: bool,
    pub api_key: bool,
}
fn head(
    path: &str,
    credentials: CredentialAvailability,
    download: bool,
) -> Result<RequestHead, WanError> {
    if download {
        if !path.starts_with("https://")
            || path.len() > 8192
            || path.contains(['\r', '\n', '#', '\\'])
            || path.chars().any(char::is_whitespace)
        {
            return Err(WanError::Response(
                "Canonical HTTPS download authority required",
            ));
        }
        return Ok(RequestHead {
            method: HttpMethod::Get,
            path: path.into(),
            secret_id: None,
            headers: vec![],
        });
    }
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
        return Err(WanError::Input("Missing Comfy API credentials"));
    };
    let mut request = RequestHead {
        method: if path.starts_with("/proxy/wan/api/v1/tasks/") {
            HttpMethod::Get
        } else {
            HttpMethod::Post
        },
        path: path.into(),
        secret_id: Some(secret.into()),
        headers: vec![
            Header {
                name: name.into(),
                value: value.into(),
            },
            Header {
                name: "Comfy-Env".into(),
                value: "host".into(),
            },
            Header {
                name: "Comfy-Usage-Source".into(),
                value: "comfyui-api".into(),
            },
            Header {
                name: "Content-Type".into(),
                value: "application/json".into(),
            },
        ],
    };
    if request.method == HttpMethod::Get {
        request
            .headers
            .retain(|header| header.name != "Content-Type");
    }
    Ok(request)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Generate,
    Poll,
    Download,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ResponsePolicy {
    pub stage: Stage,
    pub timeout_milliseconds: Option<u64>,
    pub ordinary_retries: u32,
    pub rate_limit_retries: u32,
    pub retry_backoff: f64,
    pub monitor_progress: bool,
    pub wait_label: Option<&'static str>,
    pub final_label: Option<&'static str>,
}
impl ResponsePolicy {
    fn source(stage: Stage) -> Self {
        match stage {
            Stage::Generate => Self {
                stage,
                timeout_milliseconds: Some(3_600_000),
                ordinary_retries: 3,
                rate_limit_retries: 16,
                retry_backoff: 2.0,
                monitor_progress: true,
                wait_label: Some("Waiting for server"),
                final_label: Some("Completed"),
            },
            Stage::Poll => Self {
                stage,
                timeout_milliseconds: Some(120_000),
                ordinary_retries: 10,
                rate_limit_retries: 16,
                retry_backoff: 1.4,
                monitor_progress: false,
                wait_label: Some("Checking"),
                final_label: None,
            },
            Stage::Download => Self {
                stage,
                timeout_milliseconds: None,
                ordinary_retries: 5,
                rate_limit_retries: 0,
                retry_backoff: 2.0,
                monitor_progress: false,
                wait_label: None,
                final_label: None,
            },
        }
    }
}
pub trait WanTransport: ProviderTransport {
    fn begin_poll(
        &mut self,
        _interval_milliseconds: u64,
        _estimated_duration_seconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn configure_response(
        &mut self,
        handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError>;
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
    fn poll_delay(&mut self, delay_milliseconds: u64) -> Result<(), ConformanceError>;
    fn poll_status(
        &mut self,
        _status: &str,
        _queued: bool,
        _estimated_duration_seconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn status_extraction_failed(
        &mut self,
        _error: &ConformanceError,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn exchange_on_handle(
    body: &[u8],
    handle: StreamHandle,
    policy: &ResponsePolicy,
    transport: &mut dyn WanTransport,
) -> Result<(Vec<u8>, Vec<u8>), WanError> {
    transport.configure_response(handle, policy)?;
    if body.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    if body.is_empty() {
        transport.write_request_chunk(handle, 0, &[], true)?;
    } else {
        for (sequence, chunk) in body.chunks(MAXIMUM_CHUNK_BYTES).enumerate() {
            transport.check_cancelled(Some(handle))?;
            transport.write_request_chunk(
                handle,
                sequence as u64,
                chunk,
                (sequence + 1) * MAXIMUM_CHUNK_BYTES >= body.len(),
            )?;
        }
    }
    let mut status = None;
    let mut retry_after = None;
    let mut bytes = vec![];
    let mut sequence = None;
    loop {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(
            handle,
            sequence,
            policy.timeout_milliseconds.unwrap_or(300_000),
        )? {
            ResponseEvent::Head {
                status: code,
                headers,
            } => {
                if status.is_some()
                    || !(100..=599).contains(&code)
                    || headers.len() > 16
                    || headers
                        .iter()
                        .map(|header| header.name.len() + header.value.len())
                        .sum::<usize>()
                        > 16384
                    || headers.iter().any(|header| {
                        header.name.contains(['\r', '\n']) || header.value.contains(['\r', '\n'])
                    })
                {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                status = Some(code);
                retry_after = headers
                    .iter()
                    .find(|header| header.name.eq_ignore_ascii_case("retry-after"))
                    .map(|header| header.value.clone());
            }
            ResponseEvent::Chunk(chunk) => {
                if status.is_none() {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                if chunk.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(ConformanceError::ChunkLimit.into());
                }
                if bytes
                    .len()
                    .checked_add(chunk.len())
                    .is_none_or(|size| size > MAXIMUM_BODY_BYTES)
                {
                    return Err(ConformanceError::BodyLimit.into());
                }
                bytes.extend_from_slice(&chunk);
            }
            ResponseEvent::Completed(value) => {
                let status = status.ok_or(ConformanceError::InvalidResponse)?;
                receipt(&value)?;
                if status >= 400 {
                    return Err(WanError::Http {
                        status,
                        retry_after,
                    });
                }
                return Ok((bytes, value));
            }
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled.into()),
            ResponseEvent::TimedOut => return Err(ConformanceError::HostFailure.into()),
        }
        sequence = Some(sequence.map_or(0u64, |sequence| sequence + 1));
    }
}
fn exchange_existing(
    body: &[u8],
    mut handle: StreamHandle,
    policy: &ResponsePolicy,
    transport: &mut dyn WanTransport,
) -> Result<(Vec<u8>, Vec<u8>), WanError> {
    let mut ordinary = 0u32;
    let mut rate = 0u32;
    loop {
        match exchange_on_handle(body, handle, policy, transport) {
            Err(WanError::Http {
                status,
                retry_after,
            }) => {
                let fallback = if status == 429
                    && policy.stage != Stage::Download
                    && rate < policy.rate_limit_retries
                {
                    let delay = (1000.0 * policy.retry_backoff.powi(rate as i32))
                        .round()
                        .min(30_000.0) as u64;
                    rate += 1;
                    delay
                } else if (matches!(status, 408 | 500 | 502 | 503 | 504)
                    || status == 429 && policy.stage == Stage::Download)
                    && ordinary < policy.ordinary_retries
                {
                    let delay =
                        (1000.0 * policy.retry_backoff.powi(ordinary as i32)).round() as u64;
                    ordinary += 1;
                    delay
                } else {
                    return Err(WanError::Http {
                        status,
                        retry_after,
                    });
                };
                let delay = if policy.stage != Stage::Download {
                    if let Some(header) = retry_after {
                        transport.retry_after_delay(&header, fallback)?.min(150_000)
                    } else {
                        fallback
                    }
                } else {
                    fallback
                };
                transport.check_cancelled(Some(handle))?;
                let fresh = transport.retry_request(handle, delay)?;
                if fresh == handle
                    || fresh.invocation != handle.invocation
                    || fresh.slot == 0
                    || fresh.generation == 0
                {
                    return Err(ConformanceError::InvalidHandle.into());
                }
                handle = fresh;
            }
            result => return result,
        }
    }
}
fn exchange(
    head: &RequestHead,
    body: &[u8],
    stage: Stage,
    transport: &mut dyn WanTransport,
) -> Result<(Vec<u8>, Vec<u8>), WanError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    exchange_existing(body, handle, &ResponsePolicy::source(stage), transport)
}
fn prepared_json(value: &Value, owner: &mut dyn WanPreparation) -> Result<PreparedBytes, WanError> {
    let result = owner.encode_json(value)?;
    receipt(&result.receipt)?;
    if result.bytes.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(result)
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub bytes: Vec<u8>,
    pub transport_receipts: Vec<Vec<u8>>,
    pub preparation_receipts: Vec<Vec<u8>>,
    pub approval_receipt: Vec<u8>,
    pub approved_maximum_microunits: u64,
    pub poll_count: u32,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn WanTransport,
    owner: &mut dyn WanPreparation,
) -> Result<ResultValue, WanError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    let generate = head(generate_path(&input.node), credentials, false)?;
    let handle = transport.start_request(&generate)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "wan.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !approval.accepted || approval.approved_microunits < maximum_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    receipt(&approval.receipt)?;
    transport.check_cancelled(Some(handle))?;
    let mut plan = request(&input, owner)?;
    let body = prepared_json(&plan.body, owner)?;
    let (raw, generation_receipt) = exchange_existing(
        &body.bytes,
        handle,
        &ResponsePolicy::source(Stage::Generate),
        transport,
    )?;
    let generation =
        owner.validate_response(ApiKind::Generation, &serde_json::from_slice(&raw)?)?;
    let mut preparation_receipts = vec![body.receipt];
    let output = generation.get("output").filter(|value| !value.is_null());
    let output = if let Some(output) = output {
        output
    } else {
        let code = scalar(
            ScalarOperation::PythonString,
            generation.get("code").unwrap_or(&Value::Null),
            owner,
            &mut preparation_receipts,
        )?;
        let message = scalar(
            ScalarOperation::PythonString,
            generation.get("message").unwrap_or(&Value::Null),
            owner,
            &mut preparation_receipts,
        )?;
        return Err(WanError::Provider(format!(
            "An unknown error occurred: {} - {}",
            text(&code)?,
            text(&message)?
        )));
    };
    let task = text(field(output, "task_id")?)?;
    let poll_head = head(
        &format!("/proxy/wan/api/v1/tasks/{task}"),
        credentials,
        false,
    )?;
    if ["WanTextToVideoApi", "WanImageToVideoApi"].contains(&input.node.as_str()) {
        plan.estimated_duration = scalar(
            ScalarOperation::DurationEstimate,
            argument(&input, "duration")?,
            owner,
            &mut preparation_receipts,
        )?
        .as_u64();
        if plan.estimated_duration.is_none() {
            return Err(WanError::Input(
                "Canonical source duration estimate unavailable",
            ));
        }
    }
    transport.begin_poll(plan.poll_interval_milliseconds, plan.estimated_duration)?;
    let kind = if plan.output_kind == OutputKind::Image {
        ApiKind::ImagePoll
    } else {
        ApiKind::VideoPoll
    };
    let mut transport_receipts = vec![generation_receipt];
    let mut poll_count = 0u32;
    let mut consumed = 0u32;
    let final_response = loop {
        if consumed >= 480 {
            return Err(ConformanceError::PollLimit.into());
        }
        let (bytes, poll_receipt) = exchange(&poll_head, &[], Stage::Poll, transport)?;
        transport_receipts.push(poll_receipt);
        poll_count = poll_count
            .checked_add(1)
            .ok_or(ConformanceError::PollLimit)?;
        let raw: Value = serde_json::from_slice(&bytes)?;
        if !raw.is_object() {
            return Err(WanError::Provider(
                "Polling aborted due to error: Polling endpoint returned non-JSON response.".into(),
            ));
        }
        let validated = match owner.validate_response(kind, &raw) {
            Ok(value) => Some(value),
            Err(error @ ConformanceError::InvalidRequest) => {
                transport.status_extraction_failed(&error)?;
                None
            }
            Err(error) => return Err(error.into()),
        };
        let source_status = validated
            .as_ref()
            .and_then(|value| value.get("output"))
            .and_then(|value| value.get("task_status"));
        if validated.is_some() && source_status.is_none() {
            transport.status_extraction_failed(&ConformanceError::InvalidResponse)?;
        }
        let status = scalar(
            ScalarOperation::Status,
            source_status.unwrap_or(&Value::Null),
            owner,
            &mut preparation_receipts,
        )?;
        let queued = status["queued"] == true;
        let label = status
            .get("output")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("Processing");
        transport.poll_status(label, queued, plan.estimated_duration)?;
        if status["completed"] == true {
            break validated.ok_or(WanError::Response("Missing validated completed response"))?;
        }
        if status["failed"] == true {
            let message = owner.failed_poll_message(&bytes, &raw)?;
            receipt(&message.receipt)?;
            return Err(WanError::Provider(text(&message.value)?.into()));
        }
        transport.check_cancelled(None)?;
        transport.poll_delay(plan.poll_interval_milliseconds)?;
        if !queued {
            consumed += 1;
        }
    };
    let output = field(&final_response, "output")?;
    let url = if plan.output_kind == OutputKind::Image {
        let result = field(output, "results")?
            .as_array()
            .and_then(|results| results.first())
            .ok_or(WanError::Response("Missing source first image result"))?;
        scalar(
            ScalarOperation::PythonString,
            field(result, "url")?,
            owner,
            &mut preparation_receipts,
        )?
    } else {
        field(output, "video_url")?.clone()
    };
    let (bytes, download_receipt) = exchange(
        &head(text(&url)?, credentials, true)?,
        &[],
        Stage::Download,
        transport,
    )?;
    transport_receipts.push(download_receipt);
    Ok(ResultValue {
        plan,
        bytes,
        transport_receipts,
        preparation_receipts,
        approval_receipt: approval.receipt,
        approved_maximum_microunits: maximum_microunits,
        poll_count,
    })
}
pub trait WanMaterializer {
    fn materialize(
        &mut self,
        _result: &ResultValue,
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
pub struct MaterializationOwners<'a> {
    pub preparation: &'a mut dyn WanPreparation,
    pub media: &'a mut dyn WanMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn WanTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), WanError> {
    let result = invoke(
        input,
        credentials,
        maximum_microunits,
        transport,
        owners.preparation,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.media.materialize(&result)?;
    let registry = TypeRegistry::built_in().map_err(|_| ConformanceError::HostFailure)?;
    let type_id = registry
        .resolve(result.plan.output_kind.type_name())
        .map_err(|_| ConformanceError::HostFailure)?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| ConformanceError::HostFailure)?;
    let batch = validate_outputs(
        &[ExpectedOutput {
            port_id: "output_0".into(),
            type_id,
            family,
        }],
        outputs,
        MAXIMUM_BODY_BYTES,
    )
    .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn decode_hex(value: &Value) -> Result<Vec<u8>, ConformanceError> {
    let text = value.as_str().ok_or(ConformanceError::InvalidRequest)?;
    if !text.len().is_multiple_of(2) || text.len() > 2 * MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::InvalidRequest);
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            std::str::from_utf8(pair)
                .map_err(|_| ConformanceError::InvalidRequest)
                .and_then(|pair| {
                    u8::from_str_radix(pair, 16).map_err(|_| ConformanceError::InvalidRequest)
                })
        })
        .collect()
}
fn expand_runs(value: &Value) -> Result<Vec<Value>, ConformanceError> {
    let runs = value.as_array().ok_or(ConformanceError::InvalidRequest)?;
    let mut result = vec![];
    for run in runs {
        let repeat = run["repeat"]
            .as_u64()
            .filter(|repeat| *repeat <= 1024)
            .ok_or(ConformanceError::InvalidRequest)?;
        for _ in 0..repeat {
            result.push(run["value"].clone());
        }
    }
    Ok(result)
}
struct SourceProbePreparation {
    source: &'static Value,
    case: Value,
    operations: Vec<Value>,
    consumed: usize,
    unbound: bool,
    bad_receipt: bool,
}
impl SourceProbePreparation {
    fn new(case: &Value) -> Result<Self, WanError> {
        Ok(Self {
            source: oracle()?,
            case: case.clone(),
            operations: vec![],
            consumed: 0,
            unbound: false,
            bad_receipt: false,
        })
    }
    fn value(&self, value: Value) -> Result<PreparedValue, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        Ok(PreparedValue {
            value,
            receipt: if self.bad_receipt {
                vec![]
            } else {
                b"injected-source-preparation-receipt".to_vec()
            },
        })
    }
    fn lookup(&self, key: &str, input: &Value) -> Result<Value, ConformanceError> {
        self.source[key]
            .as_array()
            .and_then(|probes| probes.iter().find(|probe| probe["input"] == *input))
            .and_then(|probe| probe.get("output"))
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)
    }
}
impl WanPreparation for SourceProbePreparation {
    fn native(
        &mut self,
        kind: NativeOperation,
        value: &Value,
        options: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        let expected = self.case["operations"]
            .as_array()
            .and_then(|records| records.get(self.consumed));
        if kind == NativeOperation::MappingItems
            && value.as_object().is_some_and(|value| value.is_empty())
            && expected
                .is_none_or(|record| record["kind"] != kind.name() || record["input"] != *value)
        {
            return self.value(json!([]));
        }
        let record = expected
            .filter(|record| {
                record["kind"] == kind.name()
                    && record["input"] == *value
                    && record["options"] == *options
            })
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)?;
        self.operations.push(record.clone());
        self.consumed += 1;
        let output = record
            .get("output")
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)?;
        self.value(output)
    }
    fn scalar(
        &mut self,
        kind: ScalarOperation,
        value: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        let output = match kind {
            ScalarOperation::Size => self.lookup("size_probes", value)?,
            ScalarOperation::EditDuration => self.lookup("edit_duration_probes", value)?,
            ScalarOperation::DurationEstimate => self.lookup("duration_probes", value)?,
            ScalarOperation::PythonString => self.lookup("scalar_probes", value)?,
            ScalarOperation::Status => self.source["status_probes"]
                .as_array()
                .and_then(|probes| probes.iter().find(|probe| probe["input"] == *value))
                .cloned()
                .ok_or(ConformanceError::InvalidRequest)?,
        };
        self.value(output)
    }
    fn validate_response(
        &mut self,
        kind: ApiKind,
        value: &Value,
    ) -> Result<Value, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        self.source["response_api_probes"]
            .as_array()
            .and_then(|probes| {
                probes
                    .iter()
                    .find(|probe| probe["kind"] == kind.name() && probe["input"] == *value)
            })
            .and_then(|probe| probe.get("output"))
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)
    }
    fn encode_json(&mut self, value: &Value) -> Result<PreparedBytes, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        let requests = expand_runs(&self.case["request_runs"])?;
        let request = requests
            .iter()
            .find(|request| request["body"] == *value)
            .ok_or(ConformanceError::InvalidRequest)?;
        Ok(PreparedBytes {
            bytes: decode_hex(&request["json_body_hex"])?,
            receipt: self.value(Value::Null)?.receipt,
        })
    }
    fn failed_poll_message(
        &mut self,
        raw_bytes: &[u8],
        value: &Value,
    ) -> Result<PreparedValue, ConformanceError> {
        if serde_json::from_slice::<Value>(raw_bytes)
            .map_err(|_| ConformanceError::InvalidResponse)?
            != *value
            || !expand_runs(&self.case["poll_runs"])?.contains(value)
        {
            return Err(ConformanceError::InvalidRequest);
        }
        self.value(
            self.case
                .get("message")
                .cloned()
                .ok_or(ConformanceError::InvalidRequest)?,
        )
    }
}
struct SourceProbeTransport {
    case: Value,
    heads: Vec<RequestHead>,
    bodies: Vec<Vec<u8>>,
    finished: Vec<bool>,
    poll_index: usize,
    poll_values: Result<Vec<Value>, ConformanceError>,
    begun: Vec<(u64, Option<u64>)>,
    events: VecDeque<Result<ResponseEvent, ConformanceError>>,
    costs: Vec<CostRequest>,
    policies: Vec<ResponsePolicy>,
    delays: Vec<u64>,
    retry_delays: Vec<u64>,
    statuses: Vec<String>,
    diagnostics: usize,
    cancelled: bool,
    deny_cost: bool,
    bad_receipt: bool,
    same_handle: bool,
    unbound_clock: bool,
}
impl SourceProbeTransport {
    fn new(case: &Value) -> Self {
        Self {
            case: case.clone(),
            heads: vec![],
            bodies: vec![],
            finished: vec![],
            poll_index: 0,
            poll_values: expand_runs(&case["poll_runs"]),
            begun: vec![],
            events: VecDeque::new(),
            costs: vec![],
            policies: vec![],
            delays: vec![],
            retry_delays: vec![],
            statuses: vec![],
            diagnostics: 0,
            cancelled: false,
            deny_cost: false,
            bad_receipt: false,
            same_handle: false,
            unbound_clock: false,
        }
    }
}
fn handle(slot: u32) -> StreamHandle {
    StreamHandle {
        invocation: 42,
        slot,
        generation: 1,
    }
}
impl ProviderTransport for SourceProbeTransport {
    fn check_cancelled(&mut self, _handle: Option<StreamHandle>) -> Result<(), ConformanceError> {
        if self.cancelled {
            Err(ConformanceError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
        let index = self.heads.len();
        self.heads.push(head.clone());
        self.bodies.push(vec![]);
        self.finished.push(false);
        if self.events.is_empty() {
            let response = if head.method == HttpMethod::Post {
                Some(
                    self.case["override"].get("generate").cloned().unwrap_or(
                        oracle().map_err(|_| ConformanceError::InvalidRequest)?["initial_response"]
                            .clone(),
                    ),
                )
            } else if head.path.starts_with("/proxy/wan/api/v1/tasks/") {
                let polls = self.poll_values.as_ref().map_err(Clone::clone)?;
                let result = polls
                    .get(self.poll_index)
                    .cloned()
                    .ok_or(ConformanceError::InvalidResponse)?;
                self.poll_index += 1;
                Some(result)
            } else {
                None
            };
            let bytes = if let Some(response) = response {
                serde_json::to_vec(&response).map_err(|_| ConformanceError::InvalidRequest)?
            } else if ["WanTextToImageApi", "WanImageToImageApi"]
                .iter()
                .any(|node| self.case["input"]["node"] == *node)
            {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/wan/source-image.png"
                ))
                .to_vec()
            } else {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/wan/source-response.mp4"
                ))
                .to_vec()
            };
            self.events.push_back(Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![],
            }));
            for chunk in bytes.chunks(MAXIMUM_CHUNK_BYTES) {
                self.events
                    .push_back(Ok(ResponseEvent::Chunk(chunk.to_vec())));
            }
            self.events
                .push_back(Ok(ResponseEvent::Completed(if self.bad_receipt {
                    vec![]
                } else {
                    b"injected-source-response-receipt".to_vec()
                })));
        }
        Ok(handle((index + 1) as u32))
    }
    fn write_request_chunk(
        &mut self,
        handle: StreamHandle,
        sequence: u64,
        bytes: &[u8],
        end: bool,
    ) -> Result<(), ConformanceError> {
        let index = handle
            .slot
            .checked_sub(1)
            .ok_or(ConformanceError::InvalidHandle)? as usize;
        let body = self
            .bodies
            .get_mut(index)
            .ok_or(ConformanceError::InvalidHandle)?;
        if sequence != body.len().div_ceil(MAXIMUM_CHUNK_BYTES) as u64 {
            return Err(ConformanceError::InvalidRequest);
        }
        body.extend_from_slice(bytes);
        if end {
            *self
                .finished
                .get_mut(index)
                .ok_or(ConformanceError::InvalidHandle)? = true;
        }
        Ok(())
    }
    fn upload(&mut self, _handle: StreamHandle, _bytes: &[u8]) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn create_storage(
        &mut self,
        _handle: StreamHandle,
        _request: &crate::common::StorageCreateRequest,
    ) -> Result<crate::common::StorageCreateResponse, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn put_signed(
        &mut self,
        _handle: StreamHandle,
        _request: &crate::common::SignedPutRequest,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn request_cost(
        &mut self,
        _handle: StreamHandle,
        cost: &CostRequest,
    ) -> Result<crate::common::CostResponse, ConformanceError> {
        self.costs.push(cost.clone());
        Ok(crate::common::CostResponse {
            accepted: !self.deny_cost,
            approved_microunits: cost.maximum_microunits,
            receipt: b"source-cost-receipt".to_vec(),
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
        _after_sequence: Option<u64>,
        _timeout_milliseconds: u64,
    ) -> Result<ResponseEvent, ConformanceError> {
        self.events
            .pop_front()
            .ok_or(ConformanceError::InvalidResponse)?
    }
}
impl WanTransport for SourceProbeTransport {
    fn begin_poll(
        &mut self,
        interval: u64,
        estimated: Option<u64>,
    ) -> Result<(), ConformanceError> {
        if self.unbound_clock {
            return Err(ConformanceError::HostFailure);
        }
        let expected = self.case["poll_calls"]
            .as_array()
            .and_then(|calls| calls.first())
            .ok_or(ConformanceError::InvalidProgress)?;
        if expected["interval"]
            .as_f64()
            .is_none_or(|value| value * 1000.0 != interval as f64)
            || expected["estimated_duration"].as_u64() != estimated
        {
            return Err(ConformanceError::InvalidProgress);
        }
        self.begun.push((interval, estimated));
        Ok(())
    }
    fn configure_response(
        &mut self,
        _handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError> {
        self.policies.push(policy.clone());
        Ok(())
    }
    fn retry_after_delay(&mut self, header: &str, _fallback: u64) -> Result<u64, ConformanceError> {
        header
            .parse::<u64>()
            .map(|seconds| seconds.saturating_mul(1000))
            .map_err(|_| ConformanceError::HostFailure)
    }
    fn retry_request(
        &mut self,
        old: StreamHandle,
        delay: u64,
    ) -> Result<StreamHandle, ConformanceError> {
        self.retry_delays.push(delay);
        if self.same_handle {
            return Ok(old);
        }
        let previous = self
            .heads
            .last()
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)?;
        self.heads.push(previous);
        self.bodies.push(vec![]);
        self.finished.push(false);
        Ok(handle(self.heads.len() as u32))
    }
    fn poll_delay(&mut self, delay: u64) -> Result<(), ConformanceError> {
        self.delays.push(delay);
        Ok(())
    }
    fn poll_status(
        &mut self,
        status: &str,
        _queued: bool,
        _estimated: Option<u64>,
    ) -> Result<(), ConformanceError> {
        self.statuses.push(status.into());
        Ok(())
    }
    fn status_extraction_failed(
        &mut self,
        _error: &ConformanceError,
    ) -> Result<(), ConformanceError> {
        self.diagnostics += 1;
        Ok(())
    }
}
fn source_input(case: &Value) -> Result<Input, WanError> {
    Ok(serde_json::from_value(case["input"].clone())?)
}
fn credentials() -> CredentialAvailability {
    CredentialAvailability {
        auth_token: true,
        api_key: true,
    }
}
fn compare_case(case: &Value, failure: bool) -> Result<(), Box<dyn Error>> {
    let mut transport = SourceProbeTransport::new(case);
    let mut owner = SourceProbePreparation::new(case)?;
    let result = invoke(
        &source_input(case)?,
        credentials(),
        1,
        &mut transport,
        &mut owner,
    );
    if failure {
        match result {
            Ok(_) => return Err("Expected source Wan failure".into()),
            Err(WanError::Provider(message)) => {
                if case["message"].as_str() != Some(&message) {
                    return Err(format!(
                        "Source Wan error message differs: {message} vs {}",
                        case["message"]
                    )
                    .into());
                }
            }
            Err(_) => {}
        }
    } else {
        let result = result?;
        let expected = if result.plan.output_kind == OutputKind::Image {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/wan/source-image.png"
            ))
            .as_slice()
        } else {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/wan/source-response.mp4"
            ))
            .as_slice()
        };
        if result.bytes != expected
            || result.poll_count as usize != expand_runs(&case["poll_runs"])?.len()
        {
            return Err("Source Wan bytes or poll count differs".into());
        }
    }
    if json!(owner.operations) != case["operations"] {
        return Err(format!(
            "Wan native validation/upload/API order differs for {}: consumed {} of {}",
            case["input"],
            owner.consumed,
            case["operations"].as_array().ok_or("operations")?.len()
        )
        .into());
    }
    let expected = expand_runs(&case["request_runs"])?;
    let actual: Vec<_> = transport
        .heads
        .iter()
        .zip(&transport.bodies)
        .zip(&transport.finished)
        .filter(|(_, finished)| **finished)
        .map(|((head, body), _)| (head, body))
        .collect();
    let provider: Vec<_> = actual
        .iter()
        .filter(|(head, _)| head.path.starts_with("/proxy/wan/"))
        .collect();
    if provider.len() != expected.len() {
        return Err(format!(
            "Wan source request count {} vs {} for {}",
            provider.len(),
            expected.len(),
            case["input"]
        )
        .into());
    }
    for ((head, body), expected) in provider.iter().zip(&expected) {
        let method = if head.method == HttpMethod::Post {
            "POST"
        } else if head.method == HttpMethod::Get {
            "GET"
        } else {
            "unknown"
        };
        if head.path != expected["path"]
            || method != expected["method"]
            || **body != decode_hex(&expected["json_body_hex"])?
            || json!(
                head.headers
                    .iter()
                    .map(|header| &header.name)
                    .collect::<Vec<_>>()
            ) != expected["header_order"]
        {
            return Err("Wan source path/method/JSON bytes/header order differs".into());
        }
    }
    let expected_delays = expand_runs(&case["delay_runs"])?;
    if transport.delays.len() != expected_delays.len()
        || transport
            .delays
            .iter()
            .zip(&expected_delays)
            .any(|(actual, expected)| expected.as_f64() != Some(*actual as f64 / 1000.0))
    {
        return Err("Wan source poll delay differs".into());
    }
    let expected_downloads: Vec<_> = case["downloads"]
        .as_array()
        .ok_or("downloads")?
        .iter()
        .filter(|record| {
            record["url"]
                .as_str()
                .is_some_and(|url| url.starts_with("https://"))
        })
        .collect();
    let downloads: Vec<_> = actual
        .iter()
        .filter(|(head, _)| head.path.starts_with("https://"))
        .collect();
    if downloads.len() != expected_downloads.len()
        || downloads
            .iter()
            .zip(expected_downloads)
            .any(|((head, _), expected)| {
                head.path != expected["url"] || head.secret_id.is_some() || !head.headers.is_empty()
            })
    {
        return Err("Wan source download selection or secret isolation differs".into());
    }
    if transport.begun.len() != case["poll_calls"].as_array().ok_or("poll_calls")?.len() {
        return Err("Wan source poll/UI phase differs".into());
    }
    if transport
        .costs
        .iter()
        .any(|cost| cost.operation != "wan.generation.maximum" || cost.maximum_microunits != 1)
    {
        return Err("Wan source cost identity differs".into());
    }
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let checked = || -> Result<(), Box<dyn Error>> {
        let claims: Value = serde_json::from_str(fixture)?;
        if claims["namespace"] != NAMESPACE
            || claims["source_sha256"] != SOURCE_SHA256
            || claims["nodes"] != serde_json::to_value(NODES)?
            || claims["routes"] != serde_json::to_value(ROUTES)?
            || claims["aliases"] != json!(ALIASES)
        {
            return Err("Wan source claims drift".into());
        }
        let source = oracle()?;
        if source["source_sha256"]["comfy_api_nodes/nodes_wan.py"] != SOURCE_SHA256 {
            return Err("Wan source fingerprint drift".into());
        }
        let mut covered = BTreeSet::new();
        for case in source["cases"].as_array().ok_or("cases")? {
            compare_case(case, false)?;
            covered.insert(source_input(case)?.node);
        }
        for key in ["ordered_errors", "response_errors"] {
            for case in source[key].as_array().ok_or("failures")? {
                compare_case(case, true)?;
            }
        }
        if covered
            != NODES
                .iter()
                .map(|node| node.node_identifier.to_owned())
                .collect()
        {
            return Err("Wan source coverage drift".into());
        }
        Ok(())
    };
    checked().map_err(|_| ConformanceError::InvalidRequest)
}
#[cfg(test)]
mod tests {
    use super::*;
    type TestResult = Result<(), Box<dyn Error>>;
    fn case(node: &str) -> Result<Value, WanError> {
        oracle()?["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|case| case["input"]["node"] == node))
            .cloned()
            .ok_or(WanError::Input("Missing source case"))
    }
    fn first_case() -> Result<Value, WanError> {
        case("WanTextToImageApi")
    }
    #[test]
    fn wan_exact_claims_all_actual_source_cases_and_error_phases() -> TestResult {
        for (key, failure) in [
            ("cases", false),
            ("ordered_errors", true),
            ("response_errors", true),
        ] {
            for (index, case) in oracle()?[key].as_array().ok_or("cases")?.iter().enumerate() {
                compare_case(case, failure).map_err(|error| {
                    format!("Wan source {key}[{index}] {}: {error}", case["input"])
                })?;
            }
        }
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/wan/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 14);
        assert_eq!(ROUTES.len(), 5);
        assert!(ALIASES.is_empty());
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            4
        );
        assert_eq!(oracle()?["cases"].as_array().ok_or("cases")?.len(), 221);
        assert_eq!(
            oracle()?["ordered_errors"]
                .as_array()
                .ok_or("errors")?
                .len(),
            133
        );
        assert_eq!(
            oracle()?["response_errors"]
                .as_array()
                .ok_or("responses")?
                .len(),
            102
        );
        Ok(())
    }
    #[test]
    fn wan_execute_defaults_preserve_ignored_continuation_fields_and_schema() -> TestResult {
        let source = oracle()?;
        assert_eq!(
            source["contracts"]["Wan2VideoContinuationApi"]["defaults"]["prompt"],
            ""
        );
        assert_eq!(
            source["contracts"]["WanTextToVideoApi"]["defaults"]["size"],
            "720p: 1:1 (960x960)"
        );
        assert_eq!(
            source["contracts"]["WanImageToVideoApi"]["defaults"]["resolution"],
            "720P"
        );
        for node in NODES {
            assert!(
                source["contracts"][node.node_identifier]["hidden"]
                    .as_str()
                    .is_some_and(|hidden| hidden.contains("auth_token_sim_org")
                        && hidden.contains("api_key_sim_org"))
            );
        }
        for record in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                case["input"]["node"] == "Wan2VideoContinuationApi"
                    && case["input"]["arguments"].get("prompt").is_some()
            })
        {
            compare_case(record, false)?;
        }
        Ok(())
    }
    #[test]
    fn wan_legacy_image_count_precedes_conversion_and_empty_reference_is_source_legal() -> TestResult
    {
        let source = oracle()?;
        for record in source["ordered_errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .filter(|case| {
                case["input"]["node"] == "WanImageToImageApi"
                    && case["operations"]
                        .as_array()
                        .is_some_and(|operations| operations.len() == 1)
            })
        {
            compare_case(record, true)?;
            assert_eq!(record["operations"][0]["kind"], "get_number_of_images");
        }
        let record = source["cases"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "WanReferenceVideoApi"
                        && case["input"]["arguments"]["reference_videos"] == json!({})
                })
            })
            .ok_or("empty legacy refs")?;
        compare_case(record, false)?;
        assert!(record["uploads"].as_array().ok_or("uploads")?.is_empty());
        Ok(())
    }
    #[test]
    fn wan_reference_limit_runs_after_all_six_source_uploads() -> TestResult {
        let record = oracle()?["ordered_errors"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "Wan2ReferenceVideoApi"
                        && case["uploads"]
                            .as_array()
                            .is_some_and(|uploads| uploads.len() == 6)
                })
            })
            .ok_or("six refs")?;
        compare_case(record, true)?;
        assert!(
            record["request_runs"]
                .as_array()
                .ok_or("requests")?
                .is_empty()
        );
        Ok(())
    }
    #[test]
    fn wan_happyhorse_validates_every_reference_before_upload() -> TestResult {
        let source = oracle()?;
        let record = source["ordered_errors"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "HappyHorseReferenceVideoApi"
                        && case["message"]
                            .as_str()
                            .is_some_and(|message| message.contains("399px"))
                })
            })
            .ok_or("bad second geometry")?;
        compare_case(record, true)?;
        assert!(record["uploads"].as_array().ok_or("uploads")?.is_empty());
        for record in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                case["input"]["node"] == "HappyHorseImageToVideoApi"
                    && case["input"]["arguments"]["first_frame"]["shape"][2] == 750
            })
        {
            compare_case(record, false)?;
        }
        Ok(())
    }
    #[test]
    fn wan_reference_insertion_order_is_delegated_without_sorting_native_slots() -> TestResult {
        let record = case("Wan2ReferenceVideoApi")?;
        let mut owner = SourceProbePreparation::new(&record)?;
        request(&source_input(&record)?, &mut owner)?;
        let slots: Vec<_> = owner
            .operations
            .iter()
            .filter(|operation| operation["kind"] == "upload_video_to_simapi")
            .map(|operation| operation["input"]["native_slot"].clone())
            .collect();
        assert_eq!(slots, vec![json!("z"), json!("a")]);
        Ok(())
    }
    #[test]
    fn wan_video_metadata_read_failure_retains_source_allow_then_upload_boundary() -> TestResult {
        for record in oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                case["input"]["arguments"]["video"]["duration_error"] == true
                    || case["input"]["arguments"]["first_clip"]["duration_error"] == true
            })
        {
            compare_case(record, false)?;
            assert!(!record["uploads"].as_array().ok_or("uploads")?.is_empty());
        }
        Ok(())
    }
    #[test]
    fn wan_legacy_duration_coercion_can_fail_after_approved_generation_before_poll() -> TestResult {
        let record = oracle()?["ordered_errors"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "WanTextToVideoApi"
                        && case["input"]["arguments"]["duration"] == "5"
                })
            })
            .ok_or("duration source phase")?;
        let mut transport = SourceProbeTransport::new(record);
        let mut owner = SourceProbePreparation::new(record)?;
        assert!(
            invoke(
                &source_input(record)?,
                credentials(),
                1,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert_eq!(transport.finished, vec![true]);
        assert!(transport.begun.is_empty());
        assert_eq!(transport.costs.len(), 1);
        compare_case(record, true)?;
        Ok(())
    }
    #[test]
    fn wan_mp3_wrapper_and_upload_mime_omission_follow_actual_helpers() -> TestResult {
        let source = oracle()?;
        for record in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                case["codec_calls"]
                    .as_array()
                    .is_some_and(|calls| !calls.is_empty())
            })
        {
            compare_case(record, false)?;
            for call in record["codec_calls"].as_array().ok_or("codec calls")? {
                assert_eq!(call["container"], "mp3");
                assert_eq!(call["codec"], "libmp3lame");
            }
            for upload in record["uploads"].as_array().ok_or("uploads")? {
                if upload["file_name"]
                    .as_str()
                    .is_some_and(|name| name.ends_with(".mp3"))
                {
                    assert_eq!(upload["content_type"], "audio/mpeg");
                }
            }
        }
        let record = case("Wan2ImageToVideoApi")?;
        assert_eq!(record["uploads"][0]["content_type"], Value::Null);
        assert_eq!(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/wan/source-audio.mp3"
            ))
            .len(),
            797
        );
        Ok(())
    }
    #[test]
    fn wan_generation_error_precedence_and_failed_status_match_source() -> TestResult {
        for record in oracle()?["response_errors"]
            .as_array()
            .ok_or("response errors")?
            .iter()
            .filter(|case| {
                case["message"].as_str().is_some_and(|message| {
                    message.starts_with("An unknown error occurred:")
                        || message.contains("Task failed:")
                })
            })
        {
            compare_case(record, true)?;
        }
        Ok(())
    }
    #[test]
    fn wan_queued_polls_do_not_consume_active_attempt_budget() -> TestResult {
        let record = oracle()?["cases"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    expand_runs(&case["poll_runs"]).is_ok_and(|polls| polls.len() == 502)
                })
            })
            .ok_or("long queued trace")?;
        let mut transport = SourceProbeTransport::new(record);
        let mut owner = SourceProbePreparation::new(record)?;
        let result = invoke(
            &source_input(record)?,
            credentials(),
            1,
            &mut transport,
            &mut owner,
        )?;
        assert_eq!(result.poll_count, 502);
        assert_eq!(transport.delays, vec![3000; 501]);
        assert_eq!(transport.begun, vec![(3000, Some(9))]);
        compare_case(record, false)?;
        Ok(())
    }
    #[test]
    fn wan_invalid_status_extraction_consumes_480_attempts_with_diagnostics() -> TestResult {
        let record = oracle()?["response_errors"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "WanTextToImageApi"
                        && case["override"]["polls"].as_array().is_some_and(|polls| {
                            polls.first().is_some_and(|poll| poll["output"].is_null())
                        })
                })
            })
            .ok_or("invalid poll model")?;
        let mut transport = SourceProbeTransport::new(record);
        let mut owner = SourceProbePreparation::new(record)?;
        assert!(matches!(
            invoke(
                &source_input(record)?,
                credentials(),
                1,
                &mut transport,
                &mut owner
            ),
            Err(WanError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.diagnostics, 480);
        assert_eq!(transport.delays, vec![3000; 480]);
        compare_case(record, true)?;
        Ok(())
    }
    #[test]
    fn wan_auth_cost_unbound_preparation_and_missing_clock_send_no_later_body() -> TestResult {
        let record = first_case()?;
        for (credentials, denied, unbound) in [
            (
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                },
                false,
                false,
            ),
            (credentials(), true, false),
            (credentials(), false, true),
        ] {
            let mut transport = SourceProbeTransport::new(&record);
            transport.deny_cost = denied;
            let mut owner = SourceProbePreparation::new(&record)?;
            owner.unbound = unbound;
            assert!(
                invoke(
                    &source_input(&record)?,
                    credentials,
                    1,
                    &mut transport,
                    &mut owner
                )
                .is_err()
            );
            assert!(transport.finished.iter().all(|finished| !*finished));
        }
        let mut transport = SourceProbeTransport::new(&record);
        transport.unbound_clock = true;
        assert!(
            invoke(
                &source_input(&record)?,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbePreparation::new(&record)?
            )
            .is_err()
        );
        assert_eq!(transport.finished, vec![true]);
        Ok(())
    }
    #[test]
    fn wan_api_key_get_headers_and_external_download_secret_isolation() -> TestResult {
        let request = head(
            "/proxy/wan/api/v1/tasks/task-1",
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(request.method, HttpMethod::Get);
        assert_eq!(request.secret_id.as_deref(), Some("sim_org.api_key"));
        assert_eq!(
            request
                .headers
                .iter()
                .map(|header| header.name.as_str())
                .collect::<Vec<_>>(),
            vec!["X-API-KEY", "Comfy-Env", "Comfy-Usage-Source"]
        );
        let download = head("https://fixture.invalid/image.png", credentials(), true)?;
        assert!(download.secret_id.is_none());
        assert!(download.headers.is_empty());
        assert!(head("/cloud/image.png", credentials(), true).is_err());
        Ok(())
    }
    #[test]
    fn wan_caller_ceiling_one_needs_approval_without_approximate_badge_policy() -> TestResult {
        let record = first_case()?;
        let mut transport = SourceProbeTransport::new(&record);
        let result = invoke(
            &source_input(&record)?,
            credentials(),
            1,
            &mut transport,
            &mut SourceProbePreparation::new(&record)?,
        )?;
        assert_eq!(result.approved_maximum_microunits, 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().ok_or("missing cost")?.operation,
            "wan.generation.maximum"
        );
        Ok(())
    }
    #[test]
    fn wan_ordinary_retry_budgets_and_replayed_body_stay_supervisor_owned() -> TestResult {
        let record = first_case()?;
        for (stage, budget, path) in [
            (Stage::Generate, 3, generate_path("WanTextToImageApi")),
            (Stage::Poll, 10, "/proxy/wan/api/v1/tasks/task-1"),
            (Stage::Download, 5, "https://fixture.invalid/image.png"),
        ] {
            let mut transport = SourceProbeTransport::new(&record);
            for _ in 0..=budget {
                transport.events.push_back(Ok(ResponseEvent::Head {
                    status: 503,
                    headers: vec![],
                }));
                transport
                    .events
                    .push_back(Ok(ResponseEvent::Completed(b"retry-receipt".to_vec())));
            }
            assert!(
                exchange(
                    &head(path, credentials(), stage == Stage::Download)?,
                    b"exact-body",
                    stage,
                    &mut transport
                )
                .is_err()
            );
            assert_eq!(transport.retry_delays.len(), budget);
            assert!(transport.bodies.iter().all(|body| body == b"exact-body"));
        }
        assert_eq!(oracle()?["helper_defaults"]["sync_op"]["max_retries"], "3");
        assert_eq!(
            oracle()?["helper_defaults"]["poll_op"]["max_retries_per_poll"],
            "10"
        );
        Ok(())
    }
    #[test]
    fn wan_rate_retry_after_limits_and_fresh_handle_identity_are_checked() -> TestResult {
        let record = first_case()?;
        let request = head(generate_path("WanTextToImageApi"), credentials(), false)?;
        let mut transport = SourceProbeTransport::new(&record);
        for _ in 0..17 {
            transport.events.push_back(Ok(ResponseEvent::Head {
                status: 429,
                headers: vec![Header {
                    name: "Retry-After".into(),
                    value: "999".into(),
                }],
            }));
            transport
                .events
                .push_back(Ok(ResponseEvent::Completed(b"receipt".to_vec())));
        }
        assert!(exchange(&request, b"{}", Stage::Generate, &mut transport).is_err());
        assert_eq!(transport.retry_delays, vec![150000; 16]);
        let mut transport = SourceProbeTransport::new(&record);
        transport.same_handle = true;
        transport.events = VecDeque::from([
            Ok(ResponseEvent::Head {
                status: 503,
                headers: vec![],
            }),
            Ok(ResponseEvent::Completed(b"receipt".to_vec())),
        ]);
        assert!(matches!(
            exchange(&request, b"{}", Stage::Generate, &mut transport),
            Err(WanError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn wan_worker_loss_cancel_bad_frames_and_fresh_injection_do_not_replay() -> TestResult {
        let record = first_case()?;
        for event in [
            Err(ConformanceError::WorkerLost),
            Ok(ResponseEvent::Cancelled),
            Ok(ResponseEvent::Chunk(vec![1])),
            Ok(ResponseEvent::TimedOut),
        ] {
            let mut transport = SourceProbeTransport::new(&record);
            transport.events.push_back(event);
            assert!(
                invoke(
                    &source_input(&record)?,
                    credentials(),
                    1,
                    &mut transport,
                    &mut SourceProbePreparation::new(&record)?
                )
                .is_err()
            );
            assert!(transport.retry_delays.is_empty());
        }
        compare_case(&record, false)?;
        compare_case(&record, false)?;
        Ok(())
    }
    #[test]
    fn wan_chunk_receipt_and_aggregate_limits_fail_before_materialization() -> TestResult {
        let record = first_case()?;
        let request = head(generate_path("WanTextToImageApi"), credentials(), false)?;
        for event in [
            ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES + 1]),
            ResponseEvent::Completed(vec![0; MAXIMUM_RECEIPT_BYTES + 1]),
        ] {
            let mut transport = SourceProbeTransport::new(&record);
            transport.events = VecDeque::from([
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(event),
            ]);
            assert!(exchange(&request, b"{}", Stage::Generate, &mut transport).is_err());
        }
        let mut transport = SourceProbeTransport::new(&record);
        transport.events.push_back(Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        }));
        for _ in 0..MAXIMUM_BODY_BYTES / MAXIMUM_CHUNK_BYTES {
            transport
                .events
                .push_back(Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES])));
        }
        transport
            .events
            .push_back(Ok(ResponseEvent::Chunk(vec![0])));
        assert!(matches!(
            exchange(&request, b"{}", Stage::Generate, &mut transport),
            Err(WanError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn wan_unbound_owner_and_empty_preparation_receipt_fail_closed() -> TestResult {
        struct Unbound;
        impl WanPreparation for Unbound {}
        let record = first_case()?;
        assert!(request(&source_input(&record)?, &mut Unbound).is_err());
        let mut owner = SourceProbePreparation::new(&record)?;
        owner.bad_receipt = true;
        assert!(request(&source_input(&record)?, &mut owner).is_err());
        Ok(())
    }
    #[test]
    fn wan_real_png_and_mp4_decoder_rejections_publish_no_native_placeholders() -> TestResult {
        use crate::common::{MaterializationError, MaterializedBatch};
        struct Decoder {
            calls: usize,
        }
        impl WanMaterializer for Decoder {
            fn materialize(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                let expected = if result.plan.output_kind == OutputKind::Image {
                    include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/wan/source-image.png"
                    ))
                    .as_slice()
                } else {
                    include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/wan/source-response.mp4"
                    ))
                    .as_slice()
                };
                if result.bytes != expected {
                    return Err(ConformanceError::InvalidResponse);
                }
                self.calls += 1;
                Err(ConformanceError::HostFailure)
            }
        }
        struct Sink {
            calls: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
                self.calls += 1;
                Ok(())
            }
        }
        for node in ["WanTextToImageApi", "WanTextToVideoApi"] {
            let record = case(node)?;
            let mut transport = SourceProbeTransport::new(&record);
            let mut preparation = SourceProbePreparation::new(&record)?;
            let mut decoder = Decoder { calls: 0 };
            let mut sink = Sink { calls: 0 };
            assert!(
                invoke_and_materialize(
                    &source_input(&record)?,
                    credentials(),
                    1,
                    &mut transport,
                    &mut MaterializationOwners {
                        preparation: &mut preparation,
                        media: &mut decoder,
                        sink: &mut sink
                    }
                )
                .is_err()
            );
            assert_eq!(decoder.calls, 1);
            assert_eq!(sink.calls, 0);
        }
        Ok(())
    }
}
