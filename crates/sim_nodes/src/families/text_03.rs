use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeEffectClass,
    NativeInputDescriptor, NativeInputRequirement, NativeMultimodalTextGenerationRequest,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativePrimitiveType, NativeStructuredValue,
    NativeTextGenerationServiceError, NativeValue, NativeValueType, built_in_source_schema,
    native_value_types_for_input_schema,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "StripWhitespace",
    "TextGenerate",
    "TextGenerateLTX2Prompt",
    "TextToLowercase",
    "TextToUppercase",
    "TruncateText",
];
const FEATURES: [&str; 6] = [
    "COMFY-NODE-0649",
    "COMFY-NODE-0671",
    "COMFY-NODE-0672",
    "COMFY-NODE-0673",
    "COMFY-NODE-0674",
    "COMFY-NODE-0701",
];
const LTX2_T2V_SYSTEM_PROMPT: &str = r###"You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
"###;
const LTX2_I2V_SYSTEM_PROMPT: &str = r###"You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output:
Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"###;
#[derive(Clone, Copy)]
struct TextNode {
    class_type: &'static str,
}
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    NODE_DESCRIPTOR_IDS
        .iter()
        .zip(FEATURES)
        .map(|(class_type, feature_id)| {
            let catalog = built_in_source_schema(class_type)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let output_name = if class_type.starts_with("TextGenerate") {
                "generated_text"
            } else {
                "texts"
            };
            let names = catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect::<Vec<_>>();
            let source_schema = catalog
                .bind_execution_ports(&names, &[], &[output_name.to_owned()])
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let inputs = catalog
                .inputs
                .iter()
                .map(|input| {
                    Ok(NativeInputDescriptor {
                        name: input.schema.name.clone(),
                        accepted_types: native_value_types_for_input_schema(&input.schema)
                            .map_err(|error| {
                                NativeNodeContractError::InvalidSourceSchema(error.to_string())
                            })?,
                        required: input.requirement == NativeInputRequirement::Required,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: true,
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            let node = TextNode { class_type };
            Ok(NativeNodeBinding::Executable {
                feature_id: feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: (*class_type).to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(source_schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
                    outputs: vec![NativeOutputDescriptor {
                        name: output_name.to_owned(),
                        produced_type: NativeValueType::Primitive(NativePrimitiveType::String),
                        is_list: false,
                    }],
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name: catalog
                        .presentation
                        .display_name
                        .clone()
                        .unwrap_or_else(|| (*class_type).to_owned()),
                    category: "text".to_owned(),
                    description: catalog.presentation.description.clone().unwrap_or_default(),
                    output_names: vec![output_name.to_owned()],
                    search_aliases: ALIASES
                        .iter()
                        .find(|(name, _)| name == class_type)
                        .map_or_else(Vec::new, |(_, aliases)| {
                            aliases.iter().map(|alias| (*alias).to_owned()).collect()
                        }),
                    is_deprecated: catalog.presentation.is_deprecated,
                    is_experimental: catalog.presentation.is_experimental,
                },
                node: Arc::new(node),
            })
        })
        .collect()
}
impl TextNode {
    fn validate_mapped(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let mut scalar_inputs = inputs.clone();
        let mut variants = Vec::new();
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            if let NativeValue::List { values } = value
                && name != "sampling_mode"
            {
                let first = values
                    .first()
                    .ok_or_else(|| invalid("Text input lists must not be empty"))?;
                scalar_inputs.insert(name.clone(), first.clone());
                for value in values {
                    variants.push((name.clone(), None, value.clone()));
                }
            } else if name == "sampling_mode" {
                let mut fields = sampling_fields(Some(value))?;
                for (field, value) in fields.clone() {
                    if let NativeValue::List { values } = value {
                        let first = values
                            .first()
                            .ok_or_else(|| invalid("Sampling input lists must not be empty"))?;
                        fields.insert(field.clone(), first.clone());
                        for value in values {
                            variants.push((name.clone(), Some(field.clone()), value));
                        }
                    }
                }
                scalar_inputs.insert(
                    name.clone(),
                    NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", fields)
                        .and_then(NativeStructuredValue::into_runtime_value)
                        .map_err(invalid)?,
                );
            }
        }
        self.validate(&scalar_inputs)?;
        for (name, field, value) in variants {
            let mut inputs = scalar_inputs.clone();
            if let Some(field) = field {
                let mut fields = sampling_fields(inputs.get(&name))?;
                fields.insert(field, value);
                inputs.insert(
                    name,
                    NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", fields)
                        .and_then(NativeStructuredValue::into_runtime_value)
                        .map_err(invalid)?,
                );
            } else {
                inputs.insert(name, value);
            }
            self.validate(&inputs)?;
        }
        Ok(())
    }
    fn validate(&self, inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        if inputs.keys().any(|name| {
            !catalog
                .inputs
                .iter()
                .any(|input| &input.schema.name == name)
        }) || catalog.inputs.iter().any(|input| {
            input.requirement == NativeInputRequirement::Required
                && !inputs.contains_key(&input.schema.name)
        }) {
            return Err(invalid("Missing or unknown text input"));
        }
        for value in inputs.values() {
            value.validate().map_err(invalid)?;
        }
        if self.class_type.starts_with("TextGenerate") {
            generation_request(self.class_type, inputs)?;
        } else {
            required_string(inputs, "texts")?;
            if self.class_type == "TruncateText" {
                integer(inputs, "max_length", 1, 10000)?;
            }
        }
        Ok(())
    }
}
impl NativeNode for TextNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-text-part-three-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.validate_mapped(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.validate_mapped(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            self.validate(&inputs)?;
            let text = match self.class_type {
                "StripWhitespace" => required_string(&inputs, "texts")?
                    .trim_matches(|character: char| {
                        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                    })
                    .to_owned(),
                "TextToLowercase" => required_string(&inputs, "texts")?.to_lowercase(),
                "TextToUppercase" => required_string(&inputs, "texts")?.to_uppercase(),
                "TruncateText" => required_string(&inputs, "texts")?
                    .chars()
                    .take(integer(&inputs, "max_length", 1, 10000)? as usize)
                    .collect(),
                _ => context
                    .execute_multimodal_text_generation(&generation_request(
                        self.class_type,
                        &inputs,
                    )?)
                    .map_err(|error| service_failure(&context, error))?
                    .into_text(),
            };
            check(&context)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Primitive {
                    value: NativePrimitive::String(text),
                }],
                ui: None,
                effects: Vec::new(),
            };
            outcome.validate().map_err(invalid)?;
            Ok(outcome)
        })
    }
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_native_text_input".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn service_failure(
    context: &NativeNodeContext,
    error: NativeTextGenerationServiceError,
) -> NativeNodeFailure {
    let interrupted =
        error == NativeTextGenerationServiceError::Cancelled || context.cancellation.is_cancelled();
    NativeNodeFailure {
        code: if interrupted {
            "interrupted"
        } else {
            "native_text_generation_failure"
        }
        .to_owned(),
        message: error.to_string(),
        kind: if interrupted {
            NativeNodeFailureKind::Interrupted
        } else {
            NativeNodeFailureKind::Failure
        },
        retryable: interrupted,
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| service_failure(context, NativeTextGenerationServiceError::Cancelled))
}
const ALIASES: &[(&str, &[&str])] = &[
    ("StripWhitespace", &[]),
    ("TextGenerate", &["LLM", "gemma"]),
    (
        "TextGenerateLTX2Prompt",
        &["prompt enhance", "LLM", "gemma"],
    ),
    ("TextToLowercase", &["lowercase"]),
    ("TextToUppercase", &["uppercase"]),
    ("TruncateText", &["truncate", "cut", "shorten"]),
];
fn generation_request(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeMultimodalTextGenerationRequest, NativeNodeFailure> {
    let clip = required_handle(inputs, "clip")?;
    let image = optional_handle(inputs, "image")?;
    let video = optional_handle(inputs, "video")?;
    let audio = optional_handle(inputs, "audio")?;
    let prompt = required_string(inputs, "prompt")?;
    let formatted_prompt = if class_type == "TextGenerateLTX2Prompt" {
        let (system_prompt, image_marker) = if image.is_some() {
            (LTX2_I2V_SYSTEM_PROMPT, "\n<image_soft_token>\n\n")
        } else {
            (LTX2_T2V_SYSTEM_PROMPT, "")
        };
        format!(
            "<start_of_turn>system\n{}<end_of_turn>\n<start_of_turn>user\n{image_marker}User Raw Input Prompt: {prompt}.<end_of_turn>\n<start_of_turn>model\n",
            system_prompt.trim()
        )
    } else {
        prompt.to_owned()
    };
    let sampling = sampling_fields(inputs.get("sampling_mode"))?;
    let do_sample = match required_string(&sampling, "sampling_mode")? {
        "on" => true,
        "off" => false,
        _ => return Err(invalid("sampling_mode must be on or off")),
    };
    let allowed = [
        "sampling_mode",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "seed",
    ];
    if sampling
        .keys()
        .any(|name| !allowed.contains(&name.as_str()))
    {
        return Err(invalid("sampling_mode contains an unknown field"));
    }
    NativeMultimodalTextGenerationRequest::checked(
        clip,
        image,
        video,
        audio,
        formatted_prompt,
        integer(inputs, "max_length", 1, 32768)? as usize,
        do_sample,
        optional_number(&sampling, "temperature", 1.0, 0.01, 2.0)?.to_bits(),
        optional_integer(&sampling, "top_k", 50, 0, 1000)? as usize,
        optional_number(&sampling, "top_p", 1.0, 0.0, 1.0)?.to_bits(),
        optional_number(&sampling, "min_p", 0.0, 0.0, 1.0)?.to_bits(),
        optional_number(&sampling, "repetition_penalty", 1.0, 0.0, 5.0)?.to_bits(),
        optional_number(&sampling, "presence_penalty", 0.0, 0.0, 5.0)?.to_bits(),
        optional_integer(&sampling, "seed", 0, 0, u64::MAX)?,
        optional_boolean(inputs, "use_default_template", true)?,
        optional_boolean(inputs, "thinking", false)?,
    )
    .map_err(invalid)
}
fn sampling_fields(
    value: Option<&NativeValue>,
) -> Result<BTreeMap<String, NativeValue>, NativeNodeFailure> {
    let value = value.ok_or_else(|| invalid("sampling_mode is required"))?;
    if let Some(structured) = NativeStructuredValue::from_native_value(value).map_err(invalid)? {
        if structured.type_name() != "COMFY_DYNAMICCOMBO_V3" {
            return Err(invalid("sampling_mode requires COMFY_DYNAMICCOMBO_V3"));
        }
        return Ok(structured.fields().clone());
    }
    let NativeValue::PreservedUnknown { type_name, value } = value else {
        return Err(invalid("sampling_mode requires a structured value"));
    };
    if type_name != "COMFY_DYNAMICCOMBO_V3" {
        return Err(invalid("sampling_mode has the wrong structured type"));
    }
    let fields = value
        .as_object()
        .ok_or_else(|| invalid("sampling_mode must contain an object"))?;
    fields
        .iter()
        .map(|(name, value)| Ok((name.clone(), sampling_value(value)?)))
        .collect()
}
fn sampling_value(value: &serde_json::Value) -> Result<NativeValue, NativeNodeFailure> {
    if let serde_json::Value::Array(values) = value {
        return Ok(NativeValue::List {
            values: values
                .iter()
                .map(sampling_value)
                .collect::<Result<_, _>>()?,
        });
    }
    let primitive = match value {
        serde_json::Value::Bool(value) => NativePrimitive::Boolean(*value),
        serde_json::Value::String(value) => NativePrimitive::String(value.clone()),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                NativePrimitive::Integer(value)
            } else if let Some(value) = value.as_u64() {
                NativePrimitive::UnsignedInteger(value)
            } else {
                NativePrimitive::Number(
                    value
                        .as_f64()
                        .ok_or_else(|| invalid("sampling number cannot be represented"))?,
                )
            }
        }
        _ => {
            return Err(invalid(
                "sampling fields must contain scalar values or mapped lists",
            ));
        }
    };
    Ok(NativeValue::Primitive { value: primitive })
}

fn optional_number(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    default: f32,
    minimum: f64,
    maximum: f64,
) -> Result<f32, NativeNodeFailure> {
    if !inputs.contains_key(name) {
        return Ok(default);
    }
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => *value as f64,
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value as f64,
        _ => return Err(invalid(format!("{name} must be numeric"))),
    };
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(invalid(format!("{name} is outside the source range")));
    }
    Ok(value as f32)
}
fn optional_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    default: u64,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    if inputs.contains_key(name) {
        integer(inputs, name, minimum, maximum)
    } else {
        Ok(default)
    }
}
fn integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).ok(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Some(*value),
        _ => None,
    };
    value
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            invalid(format!(
                "{name} must be an integer from {minimum} through {maximum}"
            ))
        })
}
fn optional_boolean(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    default: bool,
) -> Result<bool, NativeNodeFailure> {
    match inputs.get(name) {
        None => Ok(default),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(value),
        }) => Ok(*value),
        _ => Err(invalid(format!("{name} must be boolean"))),
    }
}
fn required_string<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a str, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid(format!("{name} must be a string"))),
    }
}
fn required_handle(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<NativeOpaqueHandle, NativeNodeFailure> {
    optional_handle(inputs, name)?.ok_or_else(|| invalid(format!("{name} is required")))
}
fn optional_handle(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<Option<NativeOpaqueHandle>, NativeNodeFailure> {
    match inputs.get(name) {
        None => Ok(None),
        Some(NativeValue::Handle { value }) => Ok(Some(value.clone())),
        _ => Err(invalid(format!("{name} must be a native handle"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_source_text_part_three_contracts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../sim_test_support/fixtures/nodes/text-comfy-node-0649/fixture.json"
        ))?;
        let rows = fixture["nodes"]
            .as_array()
            .ok_or("source contracts absent")?;
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), rows.len());
        for (binding, row) in bindings.iter().zip(rows) {
            let NativeNodeBinding::Executable {
                feature_id,
                descriptor,
                node,
                ..
            } = binding
            else {
                return Err("native implementation absent".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(Some(feature_id.as_str()), row["feature_id"].as_str());
            assert_eq!(
                serde_json::to_value(built_in_source_schema(node.class_type())?)?,
                row["schema"]
            );
            assert_eq!(
                *descriptor,
                serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
        }
        Ok(())
    }
}
