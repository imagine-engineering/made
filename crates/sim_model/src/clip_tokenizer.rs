use crate::{
    ArtifactKey, ModelStoreError, ModelTokenizerDescriptor, SentencePieceType,
    VerifiedEmbeddingArchivePayload, VerifiedModelTensorPayload, VerifiedSentencePieceVocabulary,
    clip::{ClipError, NativeTokenizer, Sd1Tokenizer},
    parser_limits::ParserLimits,
};
use fancy_regex::Regex as FancyRegex;
use sha2::{Digest, Sha256};
use sim_tensor::{CancellationToken, TensorError};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

pub const MAX_NATIVE_PROMPT_BYTES: usize = crate::clip::SD1_MAX_PROMPT_BYTES;
const MAX_NATIVE_TOKENIZER_LENGTH: usize = 99_999_999;
pub const MAX_NATIVE_WEIGHT_SEGMENTS: usize = crate::clip::SD1_MAX_WEIGHTED_SEGMENTS;
pub const MAX_NATIVE_TOKEN_SECTIONS: usize = 16_384;
pub const MAX_NATIVE_PROMPT_BATCH: usize = 64;
pub const MAX_NATIVE_EMBEDDING_VALUES: usize = crate::formats::MAX_EMBEDDING_ARCHIVE_VALUES;
const MAX_WEIGHT_NESTING: usize = 128;

fn reserve_tokenizer_values<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), NativeTokenizerError> {
    values
        .try_reserve(additional)
        .map_err(|_| NativeTokenizerError::Allocation(context))
}

pub const CLIP_TOKENIZER_SOURCE_ROWS: [&str; 12] = [
    "gen_empty_tokens",
    "ClipTokenWeightEncoder",
    "parse_parentheses",
    "token_weights",
    "escape_important",
    "unescape_important",
    "safe_load_embed_zip",
    "expand_directory_list",
    "bundled_embed",
    "load_embed",
    "SDTokenizer",
    "SD1Tokenizer",
];

pub fn generate_empty_tokens(
    start_token: Option<u32>,
    end_token: Option<u32>,
    pad_token: u32,
    length: usize,
) -> Result<Vec<u32>, NativeTokenizerError> {
    NativePromptTokenizer::empty_token_ids(start_token, end_token, pad_token, length)
}

pub fn apply_empty_baseline_token_weights(
    encoded_sections: &[Vec<f32>],
    empty_baseline: &[f32],
    section_weights: &[Vec<f32>],
    hidden_width: usize,
) -> Result<Vec<Vec<f32>>, NativeTokenizerError> {
    if hidden_width == 0 || encoded_sections.len() != section_weights.len() {
        return Err(NativeTokenizerError::InvalidWeightProjection);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(encoded_sections.len())
        .map_err(|_| NativeTokenizerError::Allocation("weighted section outputs"))?;
    for (encoded, weights) in encoded_sections.iter().zip(section_weights) {
        let expected = weights.len().checked_mul(hidden_width).ok_or(
            NativeTokenizerError::ArithmeticOverflow("weighted output shape"),
        )?;
        if encoded.len() != expected || empty_baseline.len() != expected {
            return Err(NativeTokenizerError::InvalidWeightProjection);
        }
        let mut weighted = Vec::new();
        weighted
            .try_reserve_exact(expected)
            .map_err(|_| NativeTokenizerError::Allocation("weighted section"))?;
        for ((encoded_token, empty_token), weight) in encoded
            .chunks_exact(hidden_width)
            .zip(empty_baseline.chunks_exact(hidden_width))
            .zip(weights)
        {
            if !weight.is_finite() {
                return Err(NativeTokenizerError::InvalidWeight(*weight));
            }
            for (encoded_value, empty_value) in encoded_token.iter().zip(empty_token) {
                let value = (*encoded_value - *empty_value).mul_add(*weight, *empty_value);
                if !value.is_finite() {
                    return Err(NativeTokenizerError::NonFiniteWeightProjection);
                }
                weighted.push(value);
            }
        }
        output.push(weighted);
    }
    Ok(output)
}

fn decode_f32_rows(
    bytes: &[u8],
    width: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<Arc<[f32]>>, NativeTokenizerError> {
    let row_bytes = width
        .checked_mul(4)
        .ok_or(NativeTokenizerError::ArithmeticOverflow(
            "embedding row bytes",
        ))?;
    if width == 0 || bytes.is_empty() || !bytes.len().is_multiple_of(row_bytes) {
        return Err(NativeTokenizerError::InvalidEmbeddingShape {
            values: bytes.len() / 4,
            width,
        });
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(bytes.len() / row_bytes)
        .map_err(|_| NativeTokenizerError::Allocation("embedding rows"))?;
    for bytes_row in bytes.chunks_exact(row_bytes) {
        cancellation.check()?;
        let mut row = Vec::new();
        row.try_reserve_exact(width)
            .map_err(|_| NativeTokenizerError::Allocation("embedding row"))?;
        for value in bytes_row.chunks_exact(4) {
            let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
            if !value.is_finite() {
                return Err(NativeTokenizerError::NonFiniteEmbedding);
            }
            row.push(value);
        }
        rows.push(Arc::from(row));
    }
    Ok(rows)
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromptWeightSegment {
    text: String,
    weight: f32,
}

impl PromptWeightSegment {
    fn checked(text: String, weight: f32) -> Result<Self, NativeTokenizerError> {
        if !weight.is_finite() {
            return Err(NativeTokenizerError::InvalidWeight(weight));
        }
        Ok(Self { text, weight })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn weight(&self) -> f32 {
        self.weight
    }
}

pub fn parse_prompt_weights(
    text: &str,
    cancellation: &CancellationToken,
) -> Result<Vec<PromptWeightSegment>, NativeTokenizerError> {
    let escaped = escape_important(text, cancellation)?;
    let mut output = token_weights(&escaped, 1.0, cancellation)?;
    for segment in &mut output {
        segment.text = unescape_important(&segment.text, cancellation)?;
    }
    Ok(output)
}

pub fn source_prompt_requires_textual_inversion(
    text: &str,
    cancellation: &CancellationToken,
) -> Result<bool, NativeTokenizerError> {
    cancellation.check()?;
    for segment in parse_prompt_weights(text, cancellation)? {
        cancellation.check()?;
        if split_embedding_references(segment.text(), cancellation)?
            .into_iter()
            .any(|part| matches!(part, PromptPart::Embedding(_)))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn escape_important(
    text: &str,
    cancellation: &CancellationToken,
) -> Result<String, NativeTokenizerError> {
    transform_important(text, true, cancellation)
}

pub fn unescape_important(
    text: &str,
    cancellation: &CancellationToken,
) -> Result<String, NativeTokenizerError> {
    transform_important(text, false, cancellation)
}

fn transform_important(
    text: &str,
    escape: bool,
    cancellation: &CancellationToken,
) -> Result<String, NativeTokenizerError> {
    cancellation.check()?;
    if text.len() > MAX_NATIVE_PROMPT_BYTES {
        return Err(NativeTokenizerError::PromptTooLarge(text.len()));
    }
    let mut transformed = String::new();
    transformed
        .try_reserve_exact(text.len())
        .map_err(|_| NativeTokenizerError::Allocation("important prompt escapes"))?;
    let mut characters = text.chars().peekable();
    let mut index = 0_usize;
    while let Some(character) = characters.next() {
        if index.is_multiple_of(256) {
            cancellation.check()?;
        }
        index = index
            .checked_add(1)
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "important prompt character count",
            ))?;
        if escape && character == '\\' {
            match characters.peek().copied() {
                Some(')') => {
                    characters.next();
                    transformed.push('\0');
                    transformed.push('\u{1}');
                    continue;
                }
                Some('(') => {
                    characters.next();
                    transformed.push('\0');
                    transformed.push('\u{2}');
                    continue;
                }
                _ => {}
            }
        } else if !escape && character == '\0' {
            match characters.peek().copied() {
                Some('\u{1}') => {
                    characters.next();
                    transformed.push(')');
                    continue;
                }
                Some('\u{2}') => {
                    characters.next();
                    transformed.push('(');
                    continue;
                }
                _ => {}
            }
        }
        transformed.push(character);
    }
    Ok(transformed)
}

pub fn token_weights(
    text: &str,
    current_weight: f32,
    cancellation: &CancellationToken,
) -> Result<Vec<PromptWeightSegment>, NativeTokenizerError> {
    cancellation.check()?;
    if text.len() > MAX_NATIVE_PROMPT_BYTES {
        return Err(NativeTokenizerError::PromptTooLarge(text.len()));
    }
    if !current_weight.is_finite() {
        return Err(NativeTokenizerError::InvalidWeight(current_weight));
    }
    let mut output = Vec::new();
    parse_weighted_inner(text, current_weight, 0, cancellation, &mut output)?;
    Ok(output)
}

fn parse_weighted_inner(
    text: &str,
    current_weight: f32,
    depth: usize,
    cancellation: &CancellationToken,
    output: &mut Vec<PromptWeightSegment>,
) -> Result<(), NativeTokenizerError> {
    cancellation.check()?;
    if depth > MAX_WEIGHT_NESTING {
        return Err(NativeTokenizerError::WeightNestingTooDeep(depth));
    }
    for item in parse_parentheses(text, cancellation)? {
        cancellation.check()?;
        if item.starts_with('(') && item.ends_with(')') && item.len() >= 2 {
            let mut inner = &item[1..item.len() - 1];
            let mut weight = current_weight * 1.1;
            if let Some(colon) = inner.rfind(':').filter(|index| *index > 0) {
                if let Ok(explicit) = inner[colon + 1..].parse::<f32>() {
                    if !explicit.is_finite() {
                        return Err(NativeTokenizerError::InvalidWeight(explicit));
                    }
                    weight = explicit;
                    inner = &inner[..colon];
                }
            }
            parse_weighted_inner(inner, weight, depth + 1, cancellation, output)?;
        } else {
            if output.len() == MAX_NATIVE_WEIGHT_SEGMENTS {
                return Err(NativeTokenizerError::TooManyWeightSegments(
                    output.len() + 1,
                ));
            }
            output.push(PromptWeightSegment::checked(item, current_weight)?);
        }
    }
    Ok(())
}

pub fn parse_parentheses(
    text: &str,
    cancellation: &CancellationToken,
) -> Result<Vec<String>, NativeTokenizerError> {
    cancellation.check()?;
    if text.len() > MAX_NATIVE_PROMPT_BYTES {
        return Err(NativeTokenizerError::PromptTooLarge(text.len()));
    }
    let mut result = Vec::new();
    let mut current = String::new();
    let mut nesting = 0_i64;
    for (index, character) in text.chars().enumerate() {
        if index.is_multiple_of(256) {
            cancellation.check()?;
        }
        match character {
            '(' => {
                if nesting == 0 && !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current.push(character);
                nesting =
                    nesting
                        .checked_add(1)
                        .ok_or(NativeTokenizerError::ArithmeticOverflow(
                            "parenthesis nesting",
                        ))?;
            }
            ')' => {
                nesting =
                    nesting
                        .checked_sub(1)
                        .ok_or(NativeTokenizerError::ArithmeticOverflow(
                            "parenthesis nesting",
                        ))?;
                current.push(character);
                if nesting == 0 {
                    result.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(character),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerConfiguration {
    pub maximum_length: usize,
    pub minimum_length: Option<usize>,
    pub minimum_padding: Option<usize>,
    pub pad_to_maximum_length: bool,
    pub pad_left: bool,
    pub start_token: Option<u32>,
    pub end_token: Option<u32>,
    pub pad_token: u32,
    pub maximum_word_length: usize,
    pub disable_weights: bool,
    pub embedding_width: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeTokenizerInvocationOptions {
    pub minimum_length: Option<usize>,
    pub minimum_padding: Option<usize>,
    pub disable_weights: Option<bool>,
}

impl NativeTokenizerInvocationOptions {
    fn validate(self) -> Result<Self, NativeTokenizerError> {
        if self
            .minimum_length
            .is_some_and(|length| length == 0 || length > MAX_NATIVE_PROMPT_BYTES)
            || self
                .minimum_padding
                .is_some_and(|padding| padding > MAX_NATIVE_PROMPT_BYTES)
        {
            return Err(NativeTokenizerError::InvalidConfiguration(
                "tokenizer invocation length or padding override is invalid".to_owned(),
            ));
        }
        Ok(self)
    }
}

fn hash_tokenizer_configuration(
    hasher: &mut Sha256,
    configuration: &TokenizerConfiguration,
) -> Result<(), NativeTokenizerError> {
    for value in [
        configuration.maximum_length,
        configuration.maximum_word_length,
    ] {
        hasher.update(
            u64::try_from(value)
                .map_err(|_| NativeTokenizerError::ArithmeticOverflow("tokenizer digest"))?
                .to_be_bytes(),
        );
    }
    for value in [configuration.minimum_length, configuration.minimum_padding] {
        match value {
            Some(value) => {
                hasher.update([1]);
                hasher.update(
                    u64::try_from(value)
                        .map_err(|_| NativeTokenizerError::ArithmeticOverflow("tokenizer digest"))?
                        .to_be_bytes(),
                );
            }
            None => hasher.update([0]),
        }
    }
    hasher.update([
        u8::from(configuration.pad_to_maximum_length),
        u8::from(configuration.pad_left),
        u8::from(configuration.disable_weights),
    ]);
    for value in [configuration.start_token, configuration.end_token] {
        match value {
            Some(value) => {
                hasher.update([1]);
                hasher.update(value.to_be_bytes());
            }
            None => hasher.update([0]),
        }
    }
    hasher.update(configuration.pad_token.to_be_bytes());
    match configuration.embedding_width {
        Some(value) => {
            hasher.update([1]);
            hasher.update(
                u64::try_from(value)
                    .map_err(|_| NativeTokenizerError::ArithmeticOverflow("tokenizer digest"))?
                    .to_be_bytes(),
            );
        }
        None => hasher.update([0]),
    }
    Ok(())
}

impl TokenizerConfiguration {
    pub fn checked(self) -> Result<Self, NativeTokenizerError> {
        if self.maximum_length == 0
            || self.maximum_length > MAX_NATIVE_TOKENIZER_LENGTH
            || self.maximum_word_length == 0
            || self.maximum_word_length > MAX_NATIVE_PROMPT_BYTES
        {
            return Err(NativeTokenizerError::InvalidConfiguration(
                "maximum lengths must be nonzero".to_owned(),
            ));
        }
        if self
            .embedding_width
            .is_some_and(|width| width == 0 || width > MAX_NATIVE_EMBEDDING_VALUES)
        {
            return Err(NativeTokenizerError::InvalidConfiguration(
                "embedding width must be nonzero".to_owned(),
            ));
        }
        let required_special = usize::from(self.start_token.is_some())
            .checked_add(usize::from(self.end_token.is_some()))
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "special token count",
            ))?;
        if self.maximum_length <= required_special {
            return Err(NativeTokenizerError::InvalidConfiguration(
                "maximum length has no content capacity".to_owned(),
            ));
        }
        if self
            .minimum_length
            .is_some_and(|value| value > MAX_NATIVE_PROMPT_BYTES)
            || self
                .minimum_padding
                .is_some_and(|value| value > MAX_NATIVE_PROMPT_BYTES)
        {
            return Err(NativeTokenizerError::InvalidConfiguration(
                "minimum padding or length exceeds the native bound".to_owned(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextualInversionEmbedding {
    artifact_key: ArtifactKey,
    artifact_sha256: String,
    width: usize,
    rows: Vec<Arc<[f32]>>,
    store_identity: Arc<()>,
}

impl TextualInversionEmbedding {
    pub fn from_verified_tensor_payload(
        payload: VerifiedModelTensorPayload,
        embedding_key: Option<&str>,
        width: usize,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        if width == 0 || width > MAX_NATIVE_EMBEDDING_VALUES {
            return Err(NativeTokenizerError::InvalidEmbeddingShape { values: 0, width });
        }
        let selected = select_embedding_tensors(&payload, embedding_key, width)?;
        let mut rows = Vec::new();
        let mut value_count = 0_usize;
        for tensor in selected {
            cancellation.check()?;
            if tensor.native_dtype() != Some(sim_tensor::DType::F32) {
                return Err(NativeTokenizerError::UnsupportedEmbeddingDType(
                    tensor.data_type().to_owned(),
                ));
            }
            let tensor_values =
                tensor
                    .shape()
                    .iter()
                    .try_fold(1_u64, |product, dimension| {
                        product.checked_mul(*dimension).ok_or(
                            NativeTokenizerError::ArithmeticOverflow("embedding tensor shape"),
                        )
                    })?;
            let tensor_values = usize::try_from(tensor_values)
                .map_err(|_| NativeTokenizerError::ArithmeticOverflow("embedding tensor values"))?;
            let expected_bytes =
                tensor_values
                    .checked_mul(4)
                    .ok_or(NativeTokenizerError::ArithmeticOverflow(
                        "embedding tensor bytes",
                    ))?;
            let width_u64 = u64::try_from(width)
                .map_err(|_| NativeTokenizerError::ArithmeticOverflow("embedding width"))?;
            if tensor.shape().last().copied() != Some(width_u64)
                || tensor_values == 0
                || tensor.bytes().len() != expected_bytes
            {
                return Err(NativeTokenizerError::InvalidEmbeddingShape {
                    values: tensor_values,
                    width,
                });
            }
            value_count = value_count.checked_add(tensor_values).ok_or(
                NativeTokenizerError::ArithmeticOverflow("embedding aggregate values"),
            )?;
            if value_count > MAX_NATIVE_EMBEDDING_VALUES {
                return Err(NativeTokenizerError::InvalidEmbeddingShape {
                    values: value_count,
                    width,
                });
            }
            rows.try_reserve(tensor_values / width)
                .map_err(|_| NativeTokenizerError::Allocation("embedding rows"))?;
            rows.extend(decode_f32_rows(tensor.bytes(), width, cancellation)?);
        }
        Self::from_checked_rows(
            payload.artifact_key().clone(),
            payload.artifact_sha256().to_owned(),
            width,
            rows,
            payload.store_identity(),
            cancellation,
        )
    }

    pub fn from_verified_archive_payload(
        payload: VerifiedEmbeddingArchivePayload,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        let width = payload.width();
        let mut rows = Vec::new();
        rows.try_reserve_exact(payload.rows().len())
            .map_err(|_| NativeTokenizerError::Allocation("embedding archive rows"))?;
        rows.extend(payload.rows().iter().cloned());
        Self::from_checked_rows(
            payload.artifact_key().clone(),
            payload.artifact_sha256().to_owned(),
            width,
            rows,
            payload.store_identity(),
            cancellation,
        )
    }

    fn from_checked_rows(
        artifact_key: ArtifactKey,
        artifact_sha256: String,
        width: usize,
        rows: Vec<Arc<[f32]>>,
        store_identity: Arc<()>,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        let value_count = rows.iter().try_fold(0_usize, |total, row| {
            if row.len() != width {
                return Err(NativeTokenizerError::InvalidEmbeddingShape {
                    values: row.len(),
                    width,
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(NativeTokenizerError::NonFiniteEmbedding);
            }
            total
                .checked_add(row.len())
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "embedding row values",
                ))
        })?;
        if width == 0 || rows.is_empty() || value_count > MAX_NATIVE_EMBEDDING_VALUES {
            return Err(NativeTokenizerError::InvalidEmbeddingShape {
                values: value_count,
                width,
            });
        }
        Ok(Self {
            artifact_key,
            artifact_sha256,
            width,
            rows,
            store_identity,
        })
    }

    pub fn artifact_key(&self) -> &ArtifactKey {
        &self.artifact_key
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    fn store_identity(&self) -> &Arc<()> {
        &self.store_identity
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub fn rows(&self) -> &[Arc<[f32]>] {
        &self.rows
    }
}

fn select_embedding_tensors<'a>(
    payload: &'a VerifiedModelTensorPayload,
    embedding_key: Option<&str>,
    width: usize,
) -> Result<Vec<&'a crate::VerifiedModelTensor>, NativeTokenizerError> {
    if embedding_key.is_some_and(|key| key.trim().is_empty()) {
        return Err(NativeTokenizerError::InvalidEmbeddingSelector);
    }
    let tensors = payload.tensors();
    if payload.has_nested_string_to_param() {
        return tensors
            .first()
            .map(singleton_embedding_tensor)
            .transpose()?
            .ok_or(NativeTokenizerError::MissingEmbeddingTensor);
    }

    if let Some(key) = embedding_key
        && let Some(exact) = tensors.iter().find(|tensor| tensor.name() == key)
    {
        return singleton_embedding_tensor(exact);
    }

    let key_suffix = embedding_key
        .map(|key| {
            let mut suffix = String::new();
            suffix
                .try_reserve_exact(key.len().saturating_add(1))
                .map_err(|_| NativeTokenizerError::Allocation("embedding key suffix"))?;
            suffix.push('.');
            suffix.push_str(key);
            Ok::<String, NativeTokenizerError>(suffix)
        })
        .transpose()?;
    for suffix in std::iter::once(".string_to_param.*").chain(key_suffix.as_deref()) {
        let mut bundled = Vec::new();
        for tensor in tensors.iter().filter(|tensor| {
            tensor.name().starts_with("bundle_emb.") && tensor.name().ends_with(suffix)
        }) {
            if tensor.shape().last().copied()
                != Some(
                    u64::try_from(width)
                        .map_err(|_| NativeTokenizerError::ArithmeticOverflow("embedding width"))?,
                )
            {
                return Err(NativeTokenizerError::EmbeddingWidthMismatch { expected: width });
            }
            bundled
                .try_reserve(1)
                .map_err(|_| NativeTokenizerError::Allocation("bundled embedding tensors"))?;
            bundled.push(tensor);
        }
        if !bundled.is_empty() {
            return Ok(bundled);
        }
    }
    tensors
        .first()
        .map(singleton_embedding_tensor)
        .transpose()?
        .ok_or(NativeTokenizerError::MissingEmbeddingTensor)
}

fn singleton_embedding_tensor(
    tensor: &crate::VerifiedModelTensor,
) -> Result<Vec<&crate::VerifiedModelTensor>, NativeTokenizerError> {
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(1)
        .map_err(|_| NativeTokenizerError::Allocation("selected embedding tensor"))?;
    selected.push(tensor);
    Ok(selected)
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeTokenValue {
    Token(u32),
    Embedding {
        artifact_key: ArtifactKey,
        artifact_sha256: String,
        values: Arc<[f32]>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeWeightedToken {
    value: NativeTokenValue,
    weight: f32,
    word_id: u64,
}

impl NativeWeightedToken {
    fn token(token: u32, weight: f32, word_id: u64) -> Self {
        Self {
            value: NativeTokenValue::Token(token),
            weight,
            word_id,
        }
    }

    pub fn value(&self) -> &NativeTokenValue {
        &self.value
    }

    pub const fn weight(&self) -> f32 {
        self.weight
    }

    pub const fn word_id(&self) -> u64 {
        self.word_id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeTokenSection {
    tokens: Vec<NativeWeightedToken>,
}

impl NativeTokenSection {
    pub fn tokens(&self) -> &[NativeWeightedToken] {
        &self.tokens
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeTokenizedPrompt {
    sections: Vec<NativeTokenSection>,
}

impl NativeTokenizedPrompt {
    pub fn sections(&self) -> &[NativeTokenSection] {
        &self.sections
    }
}

#[derive(Clone, Debug)]
pub struct SentencePieceTokenizer {
    entries: Vec<SentencePieceToken>,
    candidates_by_first: BTreeMap<char, Vec<usize>>,
    control_tokens: Vec<u32>,
    byte_tokens: BTreeMap<u8, u32>,
    unknown_token: u32,
    verified_artifact_key: Option<ArtifactKey>,
    artifact_sha256: String,
    verified_store_identity: Option<Arc<()>>,
}

#[derive(Clone, Debug)]
struct SentencePieceToken {
    piece: String,
    score: f32,
    piece_type: SentencePieceType,
}

#[derive(Debug)]
struct SentencePiecePathStep {
    previous: usize,
    tokens: Vec<u32>,
}

impl SentencePieceTokenizer {
    pub fn from_verified_vocabulary(
        verified: VerifiedSentencePieceVocabulary,
    ) -> Result<Self, NativeTokenizerError> {
        Self::from_vocabulary(
            verified.vocabulary(),
            verified.artifact_sha256().to_owned(),
            Some(verified.artifact_key().clone()),
            Some(verified.store_identity()),
        )
    }

    pub fn from_model_bytes(
        bytes: &[u8],
        artifact_sha256: String,
        limits: &ParserLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        if artifact_sha256.len() != 64
            || !artifact_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || format!("{:x}", Sha256::digest(bytes)) != artifact_sha256
        {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let vocabulary =
            crate::formats::parse_sentencepiece_model_bytes(bytes, limits, cancellation).map_err(
                |error| NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string()),
            )?;
        Self::from_vocabulary(&vocabulary, artifact_sha256, None, None)
    }

    fn from_vocabulary(
        vocabulary: &crate::SentencePieceVocabulary,
        artifact_sha256: String,
        verified_artifact_key: Option<ArtifactKey>,
        verified_store_identity: Option<Arc<()>>,
    ) -> Result<Self, NativeTokenizerError> {
        if vocabulary.entries().is_empty() {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(vocabulary.entries().len())
            .map_err(|_| NativeTokenizerError::Allocation("SentencePiece vocabulary"))?;
        let mut candidates_by_first = BTreeMap::<char, Vec<usize>>::new();
        let mut control_tokens = Vec::new();
        let mut byte_tokens = BTreeMap::new();
        let mut unknown_token = None;
        for (index, entry) in vocabulary.entries().iter().enumerate() {
            let token = u32::try_from(index)
                .map_err(|_| NativeTokenizerError::ArithmeticOverflow("SentencePiece token ID"))?;
            match entry.piece_type() {
                SentencePieceType::Unknown => {
                    if unknown_token.replace(token).is_some() {
                        return Err(NativeTokenizerError::InvalidVocabulary);
                    }
                }
                SentencePieceType::Control => {
                    control_tokens
                        .try_reserve(1)
                        .map_err(|_| NativeTokenizerError::Allocation("control token IDs"))?;
                    control_tokens.push(token);
                }
                SentencePieceType::Byte => {
                    let byte = sentencepiece_byte(entry.piece())?;
                    if byte_tokens.insert(byte, token).is_some() {
                        return Err(NativeTokenizerError::InvalidVocabulary);
                    }
                }
                SentencePieceType::Normal | SentencePieceType::UserDefined => {
                    if entry.piece_type() == SentencePieceType::UserDefined {
                        control_tokens
                            .try_reserve(1)
                            .map_err(|_| NativeTokenizerError::Allocation("special token IDs"))?;
                        control_tokens.push(token);
                    }
                    let first = entry
                        .piece()
                        .chars()
                        .next()
                        .ok_or(NativeTokenizerError::InvalidVocabulary)?;
                    let candidates = candidates_by_first.entry(first).or_default();
                    candidates.try_reserve(1).map_err(|_| {
                        NativeTokenizerError::Allocation("SentencePiece candidates")
                    })?;
                    candidates.push(index);
                }
                SentencePieceType::Unused => {}
            }
            let mut piece = String::new();
            piece
                .try_reserve_exact(entry.piece().len())
                .map_err(|_| NativeTokenizerError::Allocation("SentencePiece piece"))?;
            piece.push_str(entry.piece());
            entries.push(SentencePieceToken {
                piece,
                score: entry.score(),
                piece_type: entry.piece_type(),
            });
        }
        Ok(Self {
            entries,
            candidates_by_first,
            control_tokens,
            byte_tokens,
            unknown_token: unknown_token.ok_or(NativeTokenizerError::InvalidVocabulary)?,
            verified_artifact_key,
            artifact_sha256,
            verified_store_identity,
        })
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn matches_verified_vocabulary(&self, verified: &VerifiedSentencePieceVocabulary) -> bool {
        self.verified_artifact_key.as_ref() == Some(verified.artifact_key())
            && self.artifact_sha256 == verified.artifact_sha256()
            && self
                .verified_store_identity
                .as_ref()
                .is_some_and(|identity| Arc::ptr_eq(identity, &verified.store_identity()))
    }

    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        cancellation.check()?;
        if text.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(text.len()));
        }
        let normalized = normalize_sentencepiece(text)?;
        if normalized.is_empty() {
            return Ok(Vec::new());
        }
        let positions =
            normalized
                .len()
                .checked_add(1)
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "SentencePiece path positions",
                ))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(positions)
            .map_err(|_| NativeTokenizerError::Allocation("SentencePiece path scores"))?;
        scores.resize(positions, f64::NEG_INFINITY);
        scores[0] = 0.0;
        let mut paths = Vec::<Option<SentencePiecePathStep>>::new();
        paths
            .try_reserve_exact(positions)
            .map_err(|_| NativeTokenizerError::Allocation("SentencePiece paths"))?;
        paths.resize_with(positions, || None);

        for offset in 0..normalized.len() {
            if offset.is_multiple_of(256) {
                cancellation.check()?;
            }
            if !scores[offset].is_finite() || !normalized.is_char_boundary(offset) {
                continue;
            }
            let tail = normalized
                .get(offset..)
                .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?;
            let first = tail
                .chars()
                .next()
                .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?;
            let mut matched = false;
            if let Some(candidates) = self.candidates_by_first.get(&first) {
                for index in candidates {
                    let candidate = self
                        .entries
                        .get(*index)
                        .ok_or(NativeTokenizerError::InvalidVocabulary)?;
                    if tail.starts_with(&candidate.piece) {
                        matched = true;
                        let end = offset.checked_add(candidate.piece.len()).ok_or(
                            NativeTokenizerError::ArithmeticOverflow("SentencePiece endpoint"),
                        )?;
                        let token = u32::try_from(*index).map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("SentencePiece token ID")
                        })?;
                        update_sentencepiece_path(
                            &mut scores,
                            &mut paths,
                            offset,
                            end,
                            f64::from(candidate.score),
                            &[token],
                        )?;
                    }
                }
            }
            if matched {
                continue;
            }
            let character_length = first.len_utf8();
            let end = offset.checked_add(character_length).ok_or(
                NativeTokenizerError::ArithmeticOverflow("SentencePiece unknown endpoint"),
            )?;
            let unknown = self
                .entries
                .get(self.unknown_token as usize)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            let mut fallback = Vec::new();
            if self.byte_tokens.len() == 256 {
                fallback
                    .try_reserve_exact(character_length)
                    .map_err(|_| NativeTokenizerError::Allocation("SentencePiece byte fallback"))?;
                let mut encoded = [0_u8; 4];
                for byte in first.encode_utf8(&mut encoded).as_bytes() {
                    fallback.push(
                        *self
                            .byte_tokens
                            .get(byte)
                            .ok_or(NativeTokenizerError::InvalidVocabulary)?,
                    );
                }
            } else {
                fallback
                    .try_reserve_exact(1)
                    .map_err(|_| NativeTokenizerError::Allocation("SentencePiece unknown token"))?;
                fallback.push(self.unknown_token);
            }
            update_sentencepiece_path(
                &mut scores,
                &mut paths,
                offset,
                end,
                f64::from(unknown.score),
                &fallback,
            )?;
        }
        cancellation.check()?;
        let mut reversed = Vec::<u32>::new();
        let mut cursor = normalized.len();
        while cursor != 0 {
            let step = paths
                .get_mut(cursor)
                .and_then(Option::take)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            reversed
                .try_reserve(step.tokens.len())
                .map_err(|_| NativeTokenizerError::Allocation("SentencePiece output"))?;
            reversed.extend(step.tokens.into_iter().rev());
            cursor = step.previous;
        }
        reversed.reverse();
        Ok(reversed)
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        if tokens.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(tokens.len()));
        }
        let mut decoded = String::new();
        let mut byte_run = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            if index.is_multiple_of(256) {
                cancellation.check()?;
            }
            let entry = self
                .entries
                .get(*token as usize)
                .ok_or(NativeTokenizerError::UnknownToken(*token))?;
            if skip_special && self.control_tokens.contains(token) {
                continue;
            }
            if *token == self.unknown_token {
                flush_sentencepiece_bytes(&mut decoded, &mut byte_run)?;
                if decoded.len().saturating_add('�'.len_utf8()) > MAX_NATIVE_PROMPT_BYTES {
                    return Err(NativeTokenizerError::TooManyTokenValues(decoded.len() + 1));
                }
                decoded.push('�');
                continue;
            }
            if entry.piece_type == SentencePieceType::Byte {
                if byte_run.len() == MAX_NATIVE_PROMPT_BYTES {
                    return Err(NativeTokenizerError::TooManyTokenValues(byte_run.len() + 1));
                }
                byte_run
                    .try_reserve(1)
                    .map_err(|_| NativeTokenizerError::Allocation("decoded SentencePiece bytes"))?;
                byte_run.push(sentencepiece_byte(&entry.piece)?);
            } else {
                flush_sentencepiece_bytes(&mut decoded, &mut byte_run)?;
                let decoded_length = decoded.len().checked_add(entry.piece.len()).ok_or(
                    NativeTokenizerError::ArithmeticOverflow("decoded SentencePiece bytes"),
                )?;
                if decoded_length > MAX_NATIVE_PROMPT_BYTES {
                    return Err(NativeTokenizerError::TooManyTokenValues(decoded_length));
                }
                decoded
                    .try_reserve(entry.piece.len())
                    .map_err(|_| NativeTokenizerError::Allocation("decoded SentencePiece text"))?;
                decoded.push_str(&entry.piece);
            }
        }
        flush_sentencepiece_bytes(&mut decoded, &mut byte_run)?;
        normalize_sentencepiece_decoded_text(&decoded)
    }
}

fn normalize_sentencepiece(text: &str) -> Result<String, NativeTokenizerError> {
    let mut normalized = String::new();
    normalized
        .try_reserve(text.len().saturating_add(3))
        .map_err(|_| NativeTokenizerError::Allocation("normalized SentencePiece prompt"))?;
    for word in text.split_whitespace() {
        normalized
            .try_reserve(word.len().saturating_add(3))
            .map_err(|_| NativeTokenizerError::Allocation("normalized SentencePiece prompt"))?;
        normalized.push('▁');
        normalized.push_str(word);
        if normalized.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(normalized.len()));
        }
    }
    Ok(normalized)
}

fn update_sentencepiece_path(
    scores: &mut [f64],
    paths: &mut [Option<SentencePiecePathStep>],
    previous: usize,
    end: usize,
    added_score: f64,
    tokens: &[u32],
) -> Result<(), NativeTokenizerError> {
    let score = scores
        .get(previous)
        .copied()
        .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?
        + added_score;
    let target_score = scores
        .get_mut(end)
        .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?;
    let replace = score > *target_score
        || (score == *target_score
            && paths
                .get(end)
                .and_then(Option::as_ref)
                .is_none_or(|current| tokens < current.tokens.as_slice()));
    if replace {
        let mut owned_tokens = Vec::new();
        owned_tokens
            .try_reserve_exact(tokens.len())
            .map_err(|_| NativeTokenizerError::Allocation("SentencePiece path step"))?;
        owned_tokens.extend_from_slice(tokens);
        *target_score = score;
        *paths
            .get_mut(end)
            .ok_or(NativeTokenizerError::InvalidUtf8Boundary)? = Some(SentencePiecePathStep {
            previous,
            tokens: owned_tokens,
        });
    }
    Ok(())
}

fn sentencepiece_byte(piece: &str) -> Result<u8, NativeTokenizerError> {
    let value = piece
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))
        .ok_or(NativeTokenizerError::InvalidVocabulary)?;
    if value.len() != 2 {
        return Err(NativeTokenizerError::InvalidVocabulary);
    }
    u8::from_str_radix(value, 16).map_err(|_| NativeTokenizerError::InvalidVocabulary)
}

fn flush_sentencepiece_bytes(
    decoded: &mut String,
    byte_run: &mut Vec<u8>,
) -> Result<(), NativeTokenizerError> {
    if byte_run.is_empty() {
        return Ok(());
    }
    let text =
        std::str::from_utf8(byte_run).map_err(|_| NativeTokenizerError::InvalidDecodedUtf8)?;
    let decoded_length =
        decoded
            .len()
            .checked_add(text.len())
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "decoded SentencePiece byte run",
            ))?;
    if decoded_length > MAX_NATIVE_PROMPT_BYTES {
        return Err(NativeTokenizerError::TooManyTokenValues(decoded_length));
    }
    decoded
        .try_reserve(text.len())
        .map_err(|_| NativeTokenizerError::Allocation("decoded SentencePiece byte run"))?;
    decoded.push_str(text);
    byte_run.clear();
    Ok(())
}

fn normalize_sentencepiece_decoded_text(decoded: &str) -> Result<String, NativeTokenizerError> {
    let mut output = String::new();
    output
        .try_reserve(decoded.len())
        .map_err(|_| NativeTokenizerError::Allocation("decoded SentencePiece text"))?;
    let mut leading = true;
    for character in decoded.chars() {
        let character = if character == '▁' { ' ' } else { character };
        if leading && character.is_whitespace() {
            continue;
        }
        leading = false;
        output.push(character);
    }
    Ok(output)
}

#[derive(Clone, Debug)]
pub struct ClipBpeTokenizer {
    tokenizer: Sd1Tokenizer,
}

impl ClipBpeTokenizer {
    pub fn from_json_and_merges(
        descriptor: ModelTokenizerDescriptor,
        vocabulary_json: &str,
        merges: &str,
    ) -> Result<Self, NativeTokenizerError> {
        Ok(Self {
            tokenizer: Sd1Tokenizer::from_json_and_merges(descriptor, vocabulary_json, merges)?,
        })
    }

    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        let encoded = self.tokenizer.encode_content(text, cancellation)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(encoded.len())
            .map_err(|_| NativeTokenizerError::Allocation("CLIP content tokens"))?;
        output.extend(encoded.into_iter().map(|token| token.token()));
        Ok(output)
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        self.tokenizer
            .decode(tokens, skip_special, cancellation)
            .map_err(NativeTokenizerError::from)
    }
}

const QWEN2_PRETOKENIZER_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_PRETOKENIZER_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_PRETOKENIZER_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub const QWEN25_TOKENIZER_ARTIFACT_DIGEST: &str =
    "1fb681b4e905b87b60a5e597adf33f825a9d4820855da3610d380796e7fd6258";
pub const QWEN35_TOKENIZER_ARTIFACT_DIGEST: &str =
    "c101b201cd118a38f16f1cb37368134ed9e4bdd72f551478d4ef53c96c316374";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen2PretokenizerProfile {
    Qwen2,
    Qwen35Declared,
}

impl Qwen2PretokenizerProfile {
    fn pattern(self) -> &'static str {
        match self {
            Self::Qwen2 => QWEN2_PRETOKENIZER_PATTERN,
            Self::Qwen35Declared => QWEN35_PRETOKENIZER_PATTERN,
        }
    }
}

#[derive(Clone, Debug)]
struct Qwen2AddedToken {
    content: String,
    token: u32,
}

#[derive(Clone, Debug)]
pub struct Qwen2BpeTokenizer {
    profile: Qwen2PretokenizerProfile,
    pretokenizer: FancyRegex,
    vocabulary: BTreeMap<String, u32>,
    tokens: Vec<String>,
    merge_ranks: BTreeMap<(String, String), usize>,
    added_tokens: Vec<Qwen2AddedToken>,
    special_tokens: BTreeSet<u32>,
    byte_encoder: [char; 256],
    byte_decoder: BTreeMap<char, u8>,
    artifact_digest: String,
}

impl Qwen2BpeTokenizer {
    pub fn from_artifacts(
        profile: Qwen2PretokenizerProfile,
        tokenizer_configuration_json: &str,
        vocabulary_json: &str,
        merges: &str,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        let vocabulary: BTreeMap<String, u32> = serde_json::from_str(vocabulary_json)
            .map_err(|error| NativeTokenizerError::InvalidVocabularyJson(error.to_string()))?;
        if vocabulary.is_empty() {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let mut tokens = Vec::<String>::new();
        tokens
            .try_reserve_exact(vocabulary.len())
            .map_err(|_| NativeTokenizerError::Allocation("Qwen2 vocabulary"))?;
        tokens.resize(vocabulary.len(), String::new());
        for (piece, token) in &vocabulary {
            cancellation.check()?;
            let index = usize::try_from(*token)
                .map_err(|_| NativeTokenizerError::ArithmeticOverflow("Qwen2 token ID"))?;
            let target = tokens
                .get_mut(index)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            if !target.is_empty() || piece.is_empty() {
                return Err(NativeTokenizerError::InvalidVocabulary);
            }
            target
                .try_reserve_exact(piece.len())
                .map_err(|_| NativeTokenizerError::Allocation("Qwen2 vocabulary token"))?;
            target.push_str(piece);
        }
        if tokens.iter().any(String::is_empty) {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }

        let configuration: serde_json::Value = serde_json::from_str(tokenizer_configuration_json)
            .map_err(|error| {
            NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string())
        })?;
        validate_qwen2_configuration(&configuration, profile)?;
        let added_decoder = configuration
            .get("added_tokens_decoder")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "added_tokens_decoder is missing".to_owned(),
                )
            })?;
        let mut added_tokens = Vec::new();
        added_tokens
            .try_reserve_exact(added_decoder.len())
            .map_err(|_| NativeTokenizerError::Allocation("Qwen2 added tokens"))?;
        let mut special_tokens = BTreeSet::new();
        for (token_text, record) in added_decoder {
            cancellation.check()?;
            let token = token_text.parse::<u32>().map_err(|_| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "added token ID is invalid".to_owned(),
                )
            })?;
            let expected = vocabulary
                .len()
                .checked_add(added_tokens.len())
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "Qwen2 added token ID",
                ))?;
            if token != expected {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "added token IDs must be contiguous after the base vocabulary".to_owned(),
                ));
            }
            let object = record.as_object().ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "added token record is invalid".to_owned(),
                )
            })?;
            let content = object
                .get("content")
                .and_then(serde_json::Value::as_str)
                .filter(|content| !content.is_empty())
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "added token content is invalid".to_owned(),
                    )
                })?;
            for flag in ["lstrip", "normalized", "rstrip", "single_word"] {
                if object.get(flag).and_then(serde_json::Value::as_bool) != Some(false) {
                    return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                        format!("unsupported added token flag {flag}"),
                    ));
                }
            }
            let special = object
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "added token special classification is missing".to_owned(),
                    )
                })?;
            if vocabulary.contains_key(content)
                || added_tokens
                    .iter()
                    .any(|added: &Qwen2AddedToken| added.content == content)
            {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "added token content is duplicated".to_owned(),
                ));
            }
            if special {
                special_tokens.insert(token);
            }
            added_tokens.push(Qwen2AddedToken {
                content: content.to_owned(),
                token,
            });
        }
        added_tokens.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.token.cmp(&right.token))
        });

        let mut merge_ranks = BTreeMap::new();
        for line in merges.lines() {
            cancellation.check()?;
            let line = line.trim_end_matches('\r');
            if line.is_empty() || line == "#version: 0.2" {
                continue;
            }
            let mut pieces = line.split(' ');
            let left = pieces.next().unwrap_or_default();
            let right = pieces.next().unwrap_or_default();
            if left.is_empty() || right.is_empty() || pieces.next().is_some() {
                return Err(NativeTokenizerError::InvalidMerges);
            }
            let merged = format!("{left}{right}");
            if !vocabulary.contains_key(left)
                || !vocabulary.contains_key(right)
                || !vocabulary.contains_key(&merged)
            {
                return Err(NativeTokenizerError::InvalidMerges);
            }
            let rank = merge_ranks.len();
            if merge_ranks
                .insert((left.to_owned(), right.to_owned()), rank)
                .is_some()
            {
                return Err(NativeTokenizerError::InvalidMerges);
            }
        }
        if merge_ranks.is_empty() {
            return Err(NativeTokenizerError::InvalidMerges);
        }
        let pretokenizer = FancyRegex::new(profile.pattern()).map_err(|error| {
            NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string())
        })?;
        let (byte_encoder, byte_decoder) = qwen2_byte_alphabet()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.qwen2-byte-bpe.v1");
        hasher.update(format!("{profile:?}").as_bytes());
        hasher.update(tokenizer_configuration_json.as_bytes());
        hasher.update(vocabulary_json.as_bytes());
        hasher.update(merges.as_bytes());
        Ok(Self {
            profile,
            pretokenizer,
            vocabulary,
            tokens,
            merge_ranks,
            added_tokens,
            special_tokens,
            byte_encoder,
            byte_decoder,
            artifact_digest: format!("{:x}", hasher.finalize()),
        })
    }

    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }

    pub const fn profile(&self) -> Qwen2PretokenizerProfile {
        self.profile
    }

    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        cancellation.check()?;
        if text.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(text.len()));
        }
        let normalized: String = text.nfc().collect();
        if normalized.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(normalized.len()));
        }
        let mut output = Vec::new();
        let mut cursor = 0;
        while cursor < normalized.len() {
            cancellation.check()?;
            let next = self.next_added_token(&normalized, cursor);
            let end = next
                .as_ref()
                .map_or(normalized.len(), |(position, _)| *position);
            self.encode_ordinary(&normalized[cursor..end], cancellation, &mut output)?;
            if let Some((position, added)) = next {
                output
                    .try_reserve(1)
                    .map_err(|_| NativeTokenizerError::Allocation("Qwen2 token output"))?;
                output.push(added.token);
                cursor = position.checked_add(added.content.len()).ok_or(
                    NativeTokenizerError::ArithmeticOverflow("Qwen2 added token"),
                )?;
            } else {
                cursor = end;
            }
        }
        cancellation.check()?;
        Ok(output)
    }

    fn next_added_token(&self, text: &str, cursor: usize) -> Option<(usize, &Qwen2AddedToken)> {
        self.added_tokens
            .iter()
            .filter_map(|added| {
                text[cursor..]
                    .find(&added.content)
                    .map(|offset| (cursor + offset, added))
            })
            .min_by(|(left_position, left), (right_position, right)| {
                left_position
                    .cmp(right_position)
                    .then_with(|| right.content.len().cmp(&left.content.len()))
                    .then_with(|| left.token.cmp(&right.token))
            })
    }

    fn encode_ordinary(
        &self,
        text: &str,
        cancellation: &CancellationToken,
        output: &mut Vec<u32>,
    ) -> Result<(), NativeTokenizerError> {
        let mut covered = 0;
        for matched in self.pretokenizer.find_iter(text) {
            cancellation.check()?;
            let matched = matched
                .map_err(|error| NativeTokenizerError::Pretokenization(error.to_string()))?;
            if matched.start() != covered {
                return Err(NativeTokenizerError::Pretokenization(
                    "pretokenizer did not cover the input contiguously".to_owned(),
                ));
            }
            covered = matched.end();
            let mut symbols = Vec::new();
            symbols
                .try_reserve_exact(matched.as_str().len())
                .map_err(|_| NativeTokenizerError::Allocation("Qwen2 byte symbols"))?;
            for byte in matched.as_str().as_bytes() {
                symbols.push(self.byte_encoder[usize::from(*byte)].to_string());
            }
            self.apply_merges(&mut symbols, cancellation)?;
            output
                .try_reserve(symbols.len())
                .map_err(|_| NativeTokenizerError::Allocation("Qwen2 token output"))?;
            for symbol in symbols {
                output.push(
                    *self
                        .vocabulary
                        .get(&symbol)
                        .ok_or(NativeTokenizerError::InvalidVocabulary)?,
                );
            }
            if output.len() > MAX_NATIVE_PROMPT_BYTES {
                return Err(NativeTokenizerError::TooManyTokenValues(output.len()));
            }
        }
        if covered != text.len() {
            return Err(NativeTokenizerError::Pretokenization(
                "pretokenizer did not consume the complete input".to_owned(),
            ));
        }
        Ok(())
    }

    fn apply_merges(
        &self,
        symbols: &mut Vec<String>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeTokenizerError> {
        loop {
            cancellation.check()?;
            let Some((selected_left, selected_right)) = symbols
                .windows(2)
                .filter_map(|pair| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (*rank, pair[0].as_str(), pair[1].as_str()))
                })
                .min_by_key(|(rank, _, _)| *rank)
                .map(|(_, left, right)| (left.to_owned(), right.to_owned()))
            else {
                break;
            };
            let mut merged = Vec::new();
            merged
                .try_reserve_exact(symbols.len())
                .map_err(|_| NativeTokenizerError::Allocation("Qwen2 BPE merge"))?;
            let mut index = 0;
            while index < symbols.len() {
                if index.is_multiple_of(256) {
                    cancellation.check()?;
                }
                if symbols
                    .get(index)
                    .is_some_and(|value| value == &selected_left)
                    && symbols
                        .get(index + 1)
                        .is_some_and(|value| value == &selected_right)
                {
                    let mut value = selected_left.clone();
                    value.push_str(&selected_right);
                    merged.push(value);
                    index += 2;
                } else {
                    merged.push(symbols[index].clone());
                    index += 1;
                }
            }
            *symbols = merged;
        }
        Ok(())
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        if tokens.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(tokens.len()));
        }
        let mut output = String::new();
        let mut bytes = Vec::new();
        for (index, token) in tokens.iter().copied().enumerate() {
            if index.is_multiple_of(256) {
                cancellation.check()?;
            }
            if skip_special && self.special_tokens.contains(&token) {
                continue;
            }
            if let Some(added) = self.added_tokens.iter().find(|added| added.token == token) {
                flush_qwen2_bytes(&mut output, &mut bytes)?;
                output
                    .try_reserve(added.content.len())
                    .map_err(|_| NativeTokenizerError::Allocation("Qwen2 decoded text"))?;
                output.push_str(&added.content);
                continue;
            }
            let piece = self
                .tokens
                .get(token as usize)
                .ok_or(NativeTokenizerError::UnknownToken(token))?;
            bytes
                .try_reserve(piece.len())
                .map_err(|_| NativeTokenizerError::Allocation("Qwen2 decoded bytes"))?;
            for character in piece.chars() {
                bytes.push(
                    *self
                        .byte_decoder
                        .get(&character)
                        .ok_or(NativeTokenizerError::InvalidQwenByte(character))?,
                );
            }
        }
        flush_qwen2_bytes(&mut output, &mut bytes)?;
        cancellation.check()?;
        Ok(output)
    }

    fn resident_bytes(&self) -> Result<u64, NativeTokenizerError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| NativeTokenizerError::ArithmeticOverflow("Qwen2 residency"))?;
        for value in self
            .vocabulary
            .keys()
            .chain(self.tokens.iter())
            .chain(self.added_tokens.iter().map(|token| &token.content))
        {
            bytes = bytes
                .checked_add(
                    u64::try_from(value.capacity())
                        .map_err(|_| NativeTokenizerError::ArithmeticOverflow("Qwen2 residency"))?,
                )
                .ok_or(NativeTokenizerError::ArithmeticOverflow("Qwen2 residency"))?;
        }
        for (left, right) in self.merge_ranks.keys() {
            bytes =
                bytes
                    .checked_add(u64::try_from(left.capacity() + right.capacity()).map_err(
                        |_| NativeTokenizerError::ArithmeticOverflow("Qwen2 merge residency"),
                    )?)
                    .ok_or(NativeTokenizerError::ArithmeticOverflow("Qwen2 residency"))?;
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
pub struct Llama3BpeTokenizer {
    inner: Qwen2BpeTokenizer,
    artifact_digest: String,
}

impl Llama3BpeTokenizer {
    pub fn from_tokenizer_json(
        tokenizer_json: &[u8],
        expected_sha256: &str,
        limits: &ParserLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        limits.validate().map_err(|error| {
            NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string())
        })?;
        if expected_sha256.len() != 64
            || !expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || format!("{:x}", Sha256::digest(tokenizer_json)) != expected_sha256
        {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        if tokenizer_json.len() > MAX_GEMMA_TOKENIZER_JSON_BYTES
            || u64::try_from(tokenizer_json.len()).unwrap_or(u64::MAX) > limits.manifest_bytes
        {
            return Err(NativeTokenizerError::PromptTooLarge(tokenizer_json.len()));
        }
        let document = crate::formats::parse_strict_json(tokenizer_json, limits)
            .map_err(|error| NativeTokenizerError::InvalidVocabularyJson(error.to_string()))?;
        let model = document
            .get("model")
            .and_then(serde_json::Value::as_object)
            .ok_or(NativeTokenizerError::InvalidVocabulary)?;
        if model.get("type").and_then(serde_json::Value::as_str) != Some("BPE")
            || model
                .get("byte_fallback")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let vocabulary = serde_json::to_string(
            model
                .get("vocab")
                .ok_or(NativeTokenizerError::InvalidVocabulary)?,
        )
        .map_err(|error| NativeTokenizerError::InvalidVocabularyJson(error.to_string()))?;
        let vocabulary_values = model
            .get("vocab")
            .and_then(serde_json::Value::as_object)
            .ok_or(NativeTokenizerError::InvalidVocabulary)?;
        let merge_values = model
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .filter(|merges| !merges.is_empty())
            .ok_or(NativeTokenizerError::InvalidMerges)?;
        let mut merges = String::new();
        for value in merge_values {
            cancellation.check()?;
            let line = if let Some(line) = value.as_str() {
                line.to_owned()
            } else if let Some(pair) = value.as_array() {
                let left = pair
                    .first()
                    .and_then(serde_json::Value::as_str)
                    .ok_or(NativeTokenizerError::InvalidMerges)?;
                let right = pair
                    .get(1)
                    .and_then(serde_json::Value::as_str)
                    .ok_or(NativeTokenizerError::InvalidMerges)?;
                if pair.len() != 2 {
                    return Err(NativeTokenizerError::InvalidMerges);
                }
                format!("{left} {right}")
            } else {
                return Err(NativeTokenizerError::InvalidMerges);
            };
            merges
                .try_reserve(line.len().saturating_add(1))
                .map_err(|_| NativeTokenizerError::Allocation("Llama3 merges"))?;
            merges.push_str(&line);
            merges.push('\n');
        }
        let pretokenizers = document
            .pointer("/pre_tokenizer/pretokenizers")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Llama3 pretokenizer sequence is missing".to_owned(),
                )
            })?;
        let regex = pretokenizers
            .first()
            .and_then(|value| value.pointer("/pattern/Regex"))
            .and_then(serde_json::Value::as_str);
        let byte_level = pretokenizers.get(1).and_then(serde_json::Value::as_object);
        if pretokenizers.len() != 2
            || regex != Some(LLAMA3_PRETOKENIZER_PATTERN)
            || byte_level.and_then(|value| value.get("type"))
                != Some(&serde_json::Value::String("ByteLevel".to_owned()))
            || byte_level
                .and_then(|value| value.get("add_prefix_space"))
                .and_then(serde_json::Value::as_bool)
                != Some(false)
            || byte_level
                .and_then(|value| value.get("use_regex"))
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Llama3 pretokenizer differs from the checked byte-BPE profile".to_owned(),
            ));
        }
        let added = document
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Llama3 added tokens are missing".to_owned(),
                )
            })?;
        let metadata_values = vocabulary_values
            .len()
            .checked_add(merge_values.len())
            .and_then(|count| count.checked_add(added.len()))
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "Llama3 tokenizer metadata",
            ))?;
        if metadata_values > limits.maximum_metadata_values
            || vocabulary_values.keys().any(|token| {
                u64::try_from(token.len()).unwrap_or(u64::MAX) > limits.maximum_name_bytes
            })
        {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let mut added_decoder = serde_json::Map::new();
        for record in added {
            cancellation.check()?;
            let object = record.as_object().ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Llama3 added-token record is invalid".to_owned(),
                )
            })?;
            let token = object
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "Llama3 added-token ID is invalid".to_owned(),
                    )
                })?;
            let mut normalized = object.clone();
            normalized.remove("id");
            if added_decoder
                .insert(token.to_string(), serde_json::Value::Object(normalized))
                .is_some()
            {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Llama3 added-token ID is duplicated".to_owned(),
                ));
            }
        }
        let configuration = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "errors": "replace",
            "clean_up_tokenization_spaces": false,
            "add_bos_token": false,
            "add_prefix_space": false,
            "split_special_tokens": false,
            "bos_token": null,
            "unk_token": null,
            "added_tokens_decoder": added_decoder,
        });
        let configuration = serde_json::to_string(&configuration).map_err(|error| {
            NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string())
        })?;
        let mut inner = Qwen2BpeTokenizer::from_artifacts(
            Qwen2PretokenizerProfile::Qwen2,
            &configuration,
            &vocabulary,
            &merges,
            cancellation,
        )?;
        inner.pretokenizer = FancyRegex::new(LLAMA3_PRETOKENIZER_PATTERN).map_err(|error| {
            NativeTokenizerError::InvalidTokenizerConfiguration(error.to_string())
        })?;
        Ok(Self {
            inner,
            artifact_digest: expected_sha256.to_owned(),
        })
    }

    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }

    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        self.inner.encode(text, cancellation)
    }

    fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        self.inner.decode(tokens, skip_special, cancellation)
    }

    fn resident_bytes(&self) -> Result<u64, NativeTokenizerError> {
        self.inner
            .resident_bytes()?
            .checked_add(u64::try_from(self.artifact_digest.capacity()).map_err(|_| {
                NativeTokenizerError::ArithmeticOverflow("Llama3 tokenizer residency")
            })?)
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "Llama3 tokenizer residency",
            ))
    }
}

fn validate_qwen2_configuration(
    configuration: &serde_json::Value,
    profile: Qwen2PretokenizerProfile,
) -> Result<(), NativeTokenizerError> {
    let string = |name| configuration.get(name).and_then(serde_json::Value::as_str);
    let boolean = |name| configuration.get(name).and_then(serde_json::Value::as_bool);
    if string("tokenizer_class") != Some("Qwen2Tokenizer")
        || string("errors") != Some("replace")
        || boolean("clean_up_tokenization_spaces") != Some(false)
        || boolean("add_bos_token") != Some(false)
        || boolean("add_prefix_space") != Some(false)
        || boolean("split_special_tokens") != Some(false)
        || !configuration
            .get("bos_token")
            .is_some_and(serde_json::Value::is_null)
        || !configuration
            .get("unk_token")
            .is_some_and(serde_json::Value::is_null)
    {
        return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
            "Qwen2 tokenizer flags are unsupported".to_owned(),
        ));
    }
    match profile {
        Qwen2PretokenizerProfile::Qwen2 => {
            if configuration.get("pretokenize_regex").is_some() {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Qwen2 profile unexpectedly declares a pretokenizer regex".to_owned(),
                ));
            }
        }
        Qwen2PretokenizerProfile::Qwen35Declared => {
            if string("pretokenize_regex") != Some(QWEN35_PRETOKENIZER_PATTERN) {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Qwen3.5 pretokenizer regex does not match the checked artifact".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn qwen2_byte_alphabet() -> Result<([char; 256], BTreeMap<char, u8>), NativeTokenizerError> {
    let mut bytes = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect::<Vec<_>>();
    let mut codepoints = bytes
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut extension = 0_u32;
    for byte in 0_u8..=u8::MAX {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extension);
            extension += 1;
        }
    }
    let mut encoder = ['\0'; 256];
    let mut decoder = BTreeMap::new();
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        let character = char::from_u32(codepoint).ok_or(NativeTokenizerError::InvalidVocabulary)?;
        encoder[usize::from(byte)] = character;
        if decoder.insert(character, byte).is_some() {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
    }
    Ok((encoder, decoder))
}

fn flush_qwen2_bytes(output: &mut String, bytes: &mut Vec<u8>) -> Result<(), NativeTokenizerError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let decoded = String::from_utf8_lossy(bytes);
    let length =
        output
            .len()
            .checked_add(decoded.len())
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "Qwen2 decoded text",
            ))?;
    if length > MAX_NATIVE_PROMPT_BYTES {
        return Err(NativeTokenizerError::TooManyTokenValues(length));
    }
    output
        .try_reserve(decoded.len())
        .map_err(|_| NativeTokenizerError::Allocation("Qwen2 decoded text"))?;
    output.push_str(&decoded);
    bytes.clear();
    Ok(())
}

pub const GEMMA3_IMAGE_TOKEN: u32 = 262_144;
pub const GEMMA3_END_OF_TURN_TOKEN: u32 = 106;
pub const GEMMA4_PAD_TOKEN: u32 = 0;
pub const GEMMA4_START_TOKEN: u32 = 2;
pub const GEMMA4_IMAGE_TOKEN: u32 = 258_880;
pub const GEMMA4_AUDIO_TOKEN: u32 = 258_881;
pub const GEMMA4_VIDEO_TOKEN: u32 = 258_884;
pub const MAX_GEMMA_TOKENIZER_JSON_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_GEMMA_TOKENIZER_VOCABULARY: usize = 1_000_000;

const GEMMA3_IMAGE_TOKEN_TEXT: &str = "<image_soft_token>";
const GEMMA3_END_OF_TURN_TOKEN_TEXT: &str = "<end_of_turn>";
const GEMMA4_IMAGE_TOKEN_TEXT: &str = "<|image|>";
const GEMMA4_AUDIO_TOKEN_TEXT: &str = "<|audio|>";
const GEMMA4_VIDEO_TOKEN_TEXT: &str = "<|video|>";
const GEMMA4_THOUGHT_CHANNEL_PREFIX: &str = "<|channel>thought\n";
const GEMMA4_THOUGHT_CHANNEL_SUFFIX: &str = "<channel|>";
const GEMMA4_TURN_TOKEN_TEXT: &str = "<turn|>";
const GEMMA4_END_TOKEN_TEXT: &str = "<eos>";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaTokenizerProfile {
    Gemma3SentencePiece,
    Gemma4TokenizerJson,
}

#[derive(Clone, Debug)]
struct GemmaAddedToken {
    content: String,
    token: u32,
    special: bool,
}

#[derive(Clone, Debug)]
struct GemmaUnigramToken {
    piece: String,
    score: f64,
}

#[derive(Clone, Debug)]
enum GemmaNormalizerOperation {
    Nfc,
    Nfkc,
    Strip { left: bool, right: bool },
    Prepend(String),
    ReplaceSpace(String),
}

#[derive(Clone, Debug)]
struct GemmaUnigramTokenizer {
    tokens: Vec<GemmaUnigramToken>,
    candidates_by_first: BTreeMap<char, Vec<usize>>,
    byte_tokens: BTreeMap<u8, u32>,
    unknown_token: u32,
    byte_fallback: bool,
    normalizer: Vec<GemmaNormalizerOperation>,
    decoder_replacement: Option<String>,
    decoder_prepends: bool,
}

#[derive(Clone, Debug)]
enum GemmaTokenizerImplementation {
    SentencePiece(SentencePieceTokenizer),
    Unigram(GemmaUnigramTokenizer),
}

#[derive(Clone, Debug)]
pub struct GemmaTokenizer {
    profile: GemmaTokenizerProfile,
    implementation: GemmaTokenizerImplementation,
    added_tokens: Vec<GemmaAddedToken>,
    special_tokens: BTreeSet<u32>,
    artifact_digest: String,
}

impl GemmaTokenizer {
    pub fn gemma3(
        tokenizer: SentencePieceTokenizer,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        let end_of_turn = tokenizer
            .entries
            .get(GEMMA3_END_OF_TURN_TOKEN as usize)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma3 SentencePiece vocabulary omits <end_of_turn>".to_owned(),
                )
            })?;
        if end_of_turn.piece != GEMMA3_END_OF_TURN_TOKEN_TEXT {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma3 SentencePiece token 106 is not <end_of_turn>".to_owned(),
            ));
        }
        if tokenizer.entries.len() > GEMMA3_IMAGE_TOKEN as usize {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma3 external image token collides with SentencePiece vocabulary".to_owned(),
            ));
        }
        let added_tokens = vec![
            GemmaAddedToken {
                content: GEMMA3_IMAGE_TOKEN_TEXT.to_owned(),
                token: GEMMA3_IMAGE_TOKEN,
                special: true,
            },
            GemmaAddedToken {
                content: GEMMA3_END_OF_TURN_TOKEN_TEXT.to_owned(),
                token: GEMMA3_END_OF_TURN_TOKEN,
                special: true,
            },
        ];
        let special_tokens = BTreeSet::from([GEMMA3_IMAGE_TOKEN, GEMMA3_END_OF_TURN_TOKEN]);
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.gemma3-sentencepiece-tokenizer.v1");
        hasher.update(tokenizer.artifact_sha256.as_bytes());
        for added in &added_tokens {
            hasher.update(added.token.to_le_bytes());
            hasher.update(added.content.as_bytes());
            hasher.update([u8::from(added.special)]);
        }
        Ok(Self {
            profile: GemmaTokenizerProfile::Gemma3SentencePiece,
            implementation: GemmaTokenizerImplementation::SentencePiece(tokenizer),
            added_tokens,
            special_tokens,
            artifact_digest: format!("{:x}", hasher.finalize()),
        })
    }

    pub fn gemma4_from_tokenizer_json(
        tokenizer_json: &str,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeTokenizerError> {
        cancellation.check()?;
        if tokenizer_json.is_empty() || tokenizer_json.len() > MAX_GEMMA_TOKENIZER_JSON_BYTES {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 tokenizer JSON size is outside the native bound".to_owned(),
            ));
        }
        let document: serde_json::Value = serde_json::from_str(tokenizer_json)
            .map_err(|error| NativeTokenizerError::InvalidVocabularyJson(error.to_string()))?;
        if document.get("version").and_then(serde_json::Value::as_str) != Some("1.0")
            || !document
                .get("truncation")
                .is_none_or(serde_json::Value::is_null)
            || !document
                .get("padding")
                .is_none_or(serde_json::Value::is_null)
            || !document
                .get("pre_tokenizer")
                .is_none_or(serde_json::Value::is_null)
            || !document
                .get("post_processor")
                .is_none_or(serde_json::Value::is_null)
        {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 tokenizer JSON envelope is unsupported".to_owned(),
            ));
        }
        let model = document
            .get("model")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 tokenizer JSON model is missing".to_owned(),
                )
            })?;
        if model.get("type").and_then(serde_json::Value::as_str) != Some("Unigram") {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 tokenizer JSON must use the Unigram model".to_owned(),
            ));
        }
        let unknown_token = model
            .get("unk_id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 tokenizer unknown token is invalid".to_owned(),
                )
            })?;
        let byte_fallback = model
            .get("byte_fallback")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let vocabulary = model
            .get("vocab")
            .and_then(serde_json::Value::as_array)
            .ok_or(NativeTokenizerError::InvalidVocabulary)?;
        if vocabulary.is_empty() || vocabulary.len() > MAX_GEMMA_TOKENIZER_VOCABULARY {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(vocabulary.len())
            .map_err(|_| NativeTokenizerError::Allocation("Gemma4 Unigram vocabulary"))?;
        let mut candidates_by_first = BTreeMap::<char, Vec<usize>>::new();
        let mut byte_tokens = BTreeMap::new();
        let mut seen_pieces = BTreeSet::new();
        for (index, entry) in vocabulary.iter().enumerate() {
            if index.is_multiple_of(256) {
                cancellation.check()?;
            }
            let row = entry
                .as_array()
                .filter(|row| row.len() == 2)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            let piece = row[0]
                .as_str()
                .filter(|piece| !piece.is_empty())
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            let score = row[1]
                .as_f64()
                .filter(|score| score.is_finite())
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            if !seen_pieces.insert(piece.to_owned()) {
                return Err(NativeTokenizerError::InvalidVocabulary);
            }
            if let Ok(byte) = sentencepiece_byte(piece) {
                let token = u32::try_from(index).map_err(|_| {
                    NativeTokenizerError::ArithmeticOverflow("Gemma4 byte token ID")
                })?;
                if byte_tokens.insert(byte, token).is_some() {
                    return Err(NativeTokenizerError::InvalidVocabulary);
                }
            } else if let Some(first) = piece.chars().next() {
                let candidates = candidates_by_first.entry(first).or_default();
                candidates
                    .try_reserve(1)
                    .map_err(|_| NativeTokenizerError::Allocation("Gemma4 candidates"))?;
                candidates.push(index);
            }
            tokens.push(GemmaUnigramToken {
                piece: piece.to_owned(),
                score,
            });
        }
        if usize::try_from(unknown_token)
            .ok()
            .is_none_or(|index| index >= tokens.len())
        {
            return Err(NativeTokenizerError::InvalidVocabulary);
        }
        if byte_fallback && byte_tokens.len() != 256 {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 byte fallback requires all 256 byte tokens".to_owned(),
            ));
        }

        let added_values = document
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 added-token table is missing".to_owned(),
                )
            })?;
        let mut added_tokens = Vec::new();
        added_tokens
            .try_reserve_exact(added_values.len())
            .map_err(|_| NativeTokenizerError::Allocation("Gemma4 added tokens"))?;
        let mut special_tokens = BTreeSet::new();
        let mut added_ids = BTreeSet::new();
        let mut added_contents = BTreeSet::new();
        for value in added_values {
            cancellation.check()?;
            let object = value.as_object().ok_or_else(|| {
                NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 added-token record is invalid".to_owned(),
                )
            })?;
            let token = object
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "Gemma4 added-token ID is invalid".to_owned(),
                    )
                })?;
            let content = object
                .get("content")
                .and_then(serde_json::Value::as_str)
                .filter(|content| !content.is_empty())
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "Gemma4 added-token content is invalid".to_owned(),
                    )
                })?;
            for flag in ["lstrip", "normalized", "rstrip", "single_word"] {
                if object.get(flag).and_then(serde_json::Value::as_bool) != Some(false) {
                    return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                        format!("unsupported Gemma4 added-token flag {flag}"),
                    ));
                }
            }
            let special = object
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    NativeTokenizerError::InvalidTokenizerConfiguration(
                        "Gemma4 added-token classification is missing".to_owned(),
                    )
                })?;
            if !added_ids.insert(token) || !added_contents.insert(content.to_owned()) {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 added-token table contains a duplicate".to_owned(),
                ));
            }
            if let Some(base) = usize::try_from(token)
                .ok()
                .and_then(|index| tokens.get(index))
                && base.piece != content
            {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 added-token ID disagrees with the Unigram vocabulary".to_owned(),
                ));
            }
            if special {
                special_tokens.insert(token);
            }
            added_tokens.push(GemmaAddedToken {
                content: content.to_owned(),
                token,
                special,
            });
        }
        for (token, content) in [
            (GEMMA4_IMAGE_TOKEN, GEMMA4_IMAGE_TOKEN_TEXT),
            (GEMMA4_AUDIO_TOKEN, GEMMA4_AUDIO_TOKEN_TEXT),
            (GEMMA4_VIDEO_TOKEN, GEMMA4_VIDEO_TOKEN_TEXT),
        ] {
            if !added_tokens
                .iter()
                .any(|added| added.token == token && added.content == content)
            {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    format!("Gemma4 token {token} is not {content}"),
                ));
            }
        }
        for content in [
            GEMMA4_THOUGHT_CHANNEL_SUFFIX,
            GEMMA4_TURN_TOKEN_TEXT,
            GEMMA4_END_TOKEN_TEXT,
        ] {
            if !added_tokens.iter().any(|added| added.content == content)
                && !tokens.iter().any(|token| token.piece == content)
            {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    format!("Gemma4 cleanup token {content} is missing"),
                ));
            }
        }
        if !token_exists(GEMMA4_PAD_TOKEN, &tokens, &added_tokens)
            || !token_exists(GEMMA4_START_TOKEN, &tokens, &added_tokens)
        {
            return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 start or pad token is missing".to_owned(),
            ));
        }

        let normalizer = parse_gemma_normalizer(document.get("normalizer"))?;
        let (decoder_replacement, decoder_prepends) = parse_gemma_decoder(document.get("decoder"))?;
        cancellation.check()?;
        Ok(Self {
            profile: GemmaTokenizerProfile::Gemma4TokenizerJson,
            implementation: GemmaTokenizerImplementation::Unigram(GemmaUnigramTokenizer {
                tokens,
                candidates_by_first,
                byte_tokens,
                unknown_token,
                byte_fallback,
                normalizer,
                decoder_replacement,
                decoder_prepends,
            }),
            added_tokens,
            special_tokens,
            artifact_digest: format!("{:x}", Sha256::digest(tokenizer_json.as_bytes())),
        })
    }

    pub const fn profile(&self) -> GemmaTokenizerProfile {
        self.profile
    }

    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }

    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        cancellation.check()?;
        if text.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(text.len()));
        }
        let mut output = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            cancellation.check()?;
            let next = next_gemma_added_token(&self.added_tokens, text, cursor);
            let ordinary_end = next.map_or(text.len(), |(position, _)| position);
            if ordinary_end > cursor {
                match &self.implementation {
                    GemmaTokenizerImplementation::SentencePiece(tokenizer) => {
                        output.extend(tokenizer.encode(&text[cursor..ordinary_end], cancellation)?)
                    }
                    GemmaTokenizerImplementation::Unigram(tokenizer) => {
                        tokenizer.encode(&text[cursor..ordinary_end], cancellation, &mut output)?
                    }
                }
            }
            let Some((position, added)) = next else {
                break;
            };
            if position != ordinary_end {
                return Err(NativeTokenizerError::InvalidUtf8Boundary);
            }
            output
                .try_reserve(1)
                .map_err(|_| NativeTokenizerError::Allocation("Gemma token output"))?;
            output.push(added.token);
            cursor = position.checked_add(added.content.len()).ok_or(
                NativeTokenizerError::ArithmeticOverflow("Gemma added token cursor"),
            )?;
        }
        if text.is_empty() {
            return Ok(output);
        }
        cancellation.check()?;
        Ok(output)
    }

    fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        if tokens.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(tokens.len()));
        }
        match &self.implementation {
            GemmaTokenizerImplementation::SentencePiece(tokenizer) => {
                let mut filtered = Vec::new();
                filtered
                    .try_reserve_exact(tokens.len())
                    .map_err(|_| NativeTokenizerError::Allocation("Gemma3 decode tokens"))?;
                for token in tokens {
                    cancellation.check()?;
                    if self.special_tokens.contains(token) {
                        if !skip_special {
                            return Err(NativeTokenizerError::UnsupportedSpecialTokenDecode(
                                *token,
                            ));
                        }
                    } else {
                        filtered.push(*token);
                    }
                }
                tokenizer.decode(&filtered, skip_special, cancellation)
            }
            GemmaTokenizerImplementation::Unigram(tokenizer) => tokenizer.decode(
                tokens,
                &self.added_tokens,
                &self.special_tokens,
                skip_special,
                cancellation,
            ),
        }
    }

    fn decode_generated(
        &self,
        tokens: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        match self.profile {
            GemmaTokenizerProfile::Gemma3SentencePiece => self.decode(tokens, true, cancellation),
            GemmaTokenizerProfile::Gemma4TokenizerJson => {
                let decoded = self.decode(tokens, false, cancellation)?;
                let translated = decoded
                    .replace(GEMMA4_THOUGHT_CHANNEL_PREFIX, "<think>\n")
                    .replace(GEMMA4_THOUGHT_CHANNEL_SUFFIX, "</think>")
                    .replace(GEMMA4_TURN_TOKEN_TEXT, "")
                    .replace(GEMMA4_END_TOKEN_TEXT, "");
                Ok(translated.trim().to_owned())
            }
        }
    }

    fn resident_bytes(&self) -> Result<u64, NativeTokenizerError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| NativeTokenizerError::ArithmeticOverflow("Gemma residency"))?;
        for added in &self.added_tokens {
            bytes = bytes
                .checked_add(u64::try_from(added.content.capacity()).map_err(|_| {
                    NativeTokenizerError::ArithmeticOverflow("Gemma added-token residency")
                })?)
                .ok_or(NativeTokenizerError::ArithmeticOverflow("Gemma residency"))?;
        }
        match &self.implementation {
            GemmaTokenizerImplementation::SentencePiece(tokenizer) => {
                for entry in &tokenizer.entries {
                    bytes = bytes
                        .checked_add(u64::try_from(entry.piece.capacity()).map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("Gemma3 residency")
                        })?)
                        .ok_or(NativeTokenizerError::ArithmeticOverflow("Gemma residency"))?;
                }
            }
            GemmaTokenizerImplementation::Unigram(tokenizer) => {
                for token in &tokenizer.tokens {
                    bytes = bytes
                        .checked_add(u64::try_from(token.piece.capacity()).map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("Gemma4 residency")
                        })?)
                        .ok_or(NativeTokenizerError::ArithmeticOverflow("Gemma residency"))?;
                }
                for operation in &tokenizer.normalizer {
                    if let GemmaNormalizerOperation::Prepend(value)
                    | GemmaNormalizerOperation::ReplaceSpace(value) = operation
                    {
                        bytes = bytes
                            .checked_add(u64::try_from(value.capacity()).map_err(|_| {
                                NativeTokenizerError::ArithmeticOverflow("Gemma4 residency")
                            })?)
                            .ok_or(NativeTokenizerError::ArithmeticOverflow("Gemma residency"))?;
                    }
                }
            }
        }
        Ok(bytes)
    }
}

impl GemmaUnigramTokenizer {
    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
        output: &mut Vec<u32>,
    ) -> Result<(), NativeTokenizerError> {
        let normalized = apply_gemma_normalizer(text, &self.normalizer)?;
        if normalized.is_empty() {
            return Ok(());
        }
        let positions =
            normalized
                .len()
                .checked_add(1)
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "Gemma4 path positions",
                ))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(positions)
            .map_err(|_| NativeTokenizerError::Allocation("Gemma4 path scores"))?;
        scores.resize(positions, f64::NEG_INFINITY);
        scores[0] = 0.0;
        let mut paths = Vec::<Option<SentencePiecePathStep>>::new();
        paths
            .try_reserve_exact(positions)
            .map_err(|_| NativeTokenizerError::Allocation("Gemma4 paths"))?;
        paths.resize_with(positions, || None);
        for offset in 0..normalized.len() {
            if offset.is_multiple_of(256) {
                cancellation.check()?;
            }
            if !scores[offset].is_finite() || !normalized.is_char_boundary(offset) {
                continue;
            }
            let tail = normalized
                .get(offset..)
                .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?;
            let first = tail
                .chars()
                .next()
                .ok_or(NativeTokenizerError::InvalidUtf8Boundary)?;
            let mut matched = false;
            if let Some(candidates) = self.candidates_by_first.get(&first) {
                for index in candidates {
                    let candidate = self
                        .tokens
                        .get(*index)
                        .ok_or(NativeTokenizerError::InvalidVocabulary)?;
                    if tail.starts_with(&candidate.piece) {
                        matched = true;
                        let end = offset
                            .checked_add(candidate.piece.len())
                            .ok_or(NativeTokenizerError::ArithmeticOverflow("Gemma4 endpoint"))?;
                        let token = u32::try_from(*index).map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("Gemma4 token ID")
                        })?;
                        update_sentencepiece_path(
                            &mut scores,
                            &mut paths,
                            offset,
                            end,
                            candidate.score,
                            &[token],
                        )?;
                    }
                }
            }
            if matched {
                continue;
            }
            let character_length = first.len_utf8();
            let end = offset.checked_add(character_length).ok_or(
                NativeTokenizerError::ArithmeticOverflow("Gemma4 fallback endpoint"),
            )?;
            let mut fallback = Vec::new();
            if self.byte_fallback {
                fallback
                    .try_reserve_exact(character_length)
                    .map_err(|_| NativeTokenizerError::Allocation("Gemma4 byte fallback"))?;
                let mut encoded = [0_u8; 4];
                for byte in first.encode_utf8(&mut encoded).as_bytes() {
                    fallback.push(
                        *self
                            .byte_tokens
                            .get(byte)
                            .ok_or(NativeTokenizerError::InvalidVocabulary)?,
                    );
                }
            } else {
                fallback.push(self.unknown_token);
            }
            let unknown = self
                .tokens
                .get(self.unknown_token as usize)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            update_sentencepiece_path(
                &mut scores,
                &mut paths,
                offset,
                end,
                unknown.score,
                &fallback,
            )?;
        }
        let mut reversed = Vec::new();
        let mut cursor = normalized.len();
        while cursor != 0 {
            let step = paths
                .get_mut(cursor)
                .and_then(Option::take)
                .ok_or(NativeTokenizerError::InvalidVocabulary)?;
            reversed
                .try_reserve(step.tokens.len())
                .map_err(|_| NativeTokenizerError::Allocation("Gemma4 output"))?;
            reversed.extend(step.tokens.into_iter().rev());
            cursor = step.previous;
        }
        reversed.reverse();
        output
            .try_reserve(reversed.len())
            .map_err(|_| NativeTokenizerError::Allocation("Gemma4 token output"))?;
        output.extend(reversed);
        Ok(())
    }

    fn decode(
        &self,
        tokens: &[u32],
        added_tokens: &[GemmaAddedToken],
        special_tokens: &BTreeSet<u32>,
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        let mut pieces = String::new();
        for (index, token) in tokens.iter().copied().enumerate() {
            if index.is_multiple_of(256) {
                cancellation.check()?;
            }
            if skip_special && special_tokens.contains(&token) {
                continue;
            }
            let piece = if let Some(added) = added_tokens.iter().find(|added| added.token == token)
            {
                added.content.as_str()
            } else {
                self.tokens
                    .get(token as usize)
                    .map(|token| token.piece.as_str())
                    .ok_or(NativeTokenizerError::UnknownToken(token))?
            };
            pieces
                .try_reserve(piece.len())
                .map_err(|_| NativeTokenizerError::Allocation("Gemma4 decoded text"))?;
            pieces.push_str(piece);
            if pieces.len() > MAX_NATIVE_PROMPT_BYTES {
                return Err(NativeTokenizerError::TooManyTokenValues(pieces.len()));
            }
        }
        if let Some(replacement) = &self.decoder_replacement {
            let mut decoded = pieces.replace(replacement, " ");
            if self.decoder_prepends && decoded.starts_with(' ') {
                decoded.remove(0);
            }
            Ok(decoded)
        } else {
            Ok(pieces)
        }
    }
}

fn token_exists(token: u32, tokens: &[GemmaUnigramToken], added: &[GemmaAddedToken]) -> bool {
    usize::try_from(token)
        .ok()
        .is_some_and(|index| index < tokens.len())
        || added.iter().any(|added| added.token == token)
}

fn next_gemma_added_token<'a>(
    added_tokens: &'a [GemmaAddedToken],
    text: &str,
    cursor: usize,
) -> Option<(usize, &'a GemmaAddedToken)> {
    added_tokens
        .iter()
        .filter_map(|added| {
            text.get(cursor..)?
                .find(&added.content)
                .map(|relative| (cursor + relative, added))
        })
        .min_by(|(left_position, left), (right_position, right)| {
            left_position
                .cmp(right_position)
                .then_with(|| right.content.len().cmp(&left.content.len()))
                .then_with(|| left.token.cmp(&right.token))
        })
}

fn parse_gemma_normalizer(
    value: Option<&serde_json::Value>,
) -> Result<Vec<GemmaNormalizerOperation>, NativeTokenizerError> {
    fn append(
        value: &serde_json::Value,
        output: &mut Vec<GemmaNormalizerOperation>,
    ) -> Result<(), NativeTokenizerError> {
        let object = value.as_object().ok_or_else(|| {
            NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 normalizer is invalid".to_owned(),
            )
        })?;
        match object.get("type").and_then(serde_json::Value::as_str) {
            Some("Sequence") => {
                let normalizers = object
                    .get("normalizers")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| {
                        NativeTokenizerError::InvalidTokenizerConfiguration(
                            "Gemma4 normalizer sequence is invalid".to_owned(),
                        )
                    })?;
                for normalizer in normalizers {
                    append(normalizer, output)?;
                }
            }
            Some("NFC") => output.push(GemmaNormalizerOperation::Nfc),
            Some("NFKC") => output.push(GemmaNormalizerOperation::Nfkc),
            Some("Strip") => output.push(GemmaNormalizerOperation::Strip {
                left: object
                    .get("strip_left")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                right: object
                    .get("strip_right")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            }),
            Some("Prepend") => output.push(GemmaNormalizerOperation::Prepend(
                object
                    .get("prepend")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        NativeTokenizerError::InvalidTokenizerConfiguration(
                            "Gemma4 prepend normalizer is invalid".to_owned(),
                        )
                    })?
                    .to_owned(),
            )),
            Some("Replace") => {
                let pattern = object
                    .get("pattern")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|pattern| pattern.get("String"))
                    .and_then(serde_json::Value::as_str);
                if pattern != Some(" ") {
                    return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                        "Gemma4 replace normalizer only supports literal spaces".to_owned(),
                    ));
                }
                let content = object
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        NativeTokenizerError::InvalidTokenizerConfiguration(
                            "Gemma4 replacement is invalid".to_owned(),
                        )
                    })?;
                output.push(GemmaNormalizerOperation::ReplaceSpace(content.to_owned()));
            }
            Some(other) => {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    format!("unsupported Gemma4 normalizer {other}"),
                ));
            }
            None => {
                return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
                    "Gemma4 normalizer type is missing".to_owned(),
                ));
            }
        }
        Ok(())
    }

    let mut operations = Vec::new();
    if let Some(value) = value.filter(|value| !value.is_null()) {
        append(value, &mut operations)?;
    }
    Ok(operations)
}

fn parse_gemma_decoder(
    value: Option<&serde_json::Value>,
) -> Result<(Option<String>, bool), NativeTokenizerError> {
    let Some(object) = value
        .filter(|value| !value.is_null())
        .and_then(serde_json::Value::as_object)
    else {
        return Ok((None, false));
    };
    if object.get("type").and_then(serde_json::Value::as_str) != Some("Metaspace") {
        return Err(NativeTokenizerError::InvalidTokenizerConfiguration(
            "Gemma4 tokenizer decoder must use Metaspace".to_owned(),
        ));
    }
    let replacement = object
        .get("replacement")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            NativeTokenizerError::InvalidTokenizerConfiguration(
                "Gemma4 Metaspace replacement is invalid".to_owned(),
            )
        })?;
    let prepends = matches!(
        object
            .get("prepend_scheme")
            .and_then(serde_json::Value::as_str),
        Some("always") | Some("first")
    );
    Ok((Some(replacement.to_owned()), prepends))
}

fn apply_gemma_normalizer(
    text: &str,
    operations: &[GemmaNormalizerOperation],
) -> Result<String, NativeTokenizerError> {
    let mut value = text.to_owned();
    for operation in operations {
        value = match operation {
            GemmaNormalizerOperation::Nfc => value.nfc().collect(),
            GemmaNormalizerOperation::Nfkc => value.nfkc().collect(),
            GemmaNormalizerOperation::Strip { left, right } => match (*left, *right) {
                (true, true) => value.trim().to_owned(),
                (true, false) => value.trim_start().to_owned(),
                (false, true) => value.trim_end().to_owned(),
                (false, false) => value,
            },
            GemmaNormalizerOperation::Prepend(prefix) => {
                let mut result = String::new();
                result
                    .try_reserve(prefix.len().saturating_add(value.len()))
                    .map_err(|_| NativeTokenizerError::Allocation("Gemma4 normalization"))?;
                result.push_str(prefix);
                result.push_str(&value);
                result
            }
            GemmaNormalizerOperation::ReplaceSpace(replacement) => value.replace(' ', replacement),
        };
        if value.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(value.len()));
        }
    }
    Ok(value)
}

#[derive(Clone, Debug)]
pub enum NativeTokenizerFamily {
    ClipBpe(ClipBpeTokenizer),
    Gemma(GemmaTokenizer),
    Llama3ByteBpe(Llama3BpeTokenizer),
    Qwen2ByteBpe(Qwen2BpeTokenizer),
    SentencePiece(SentencePieceTokenizer),
}

impl NativeTokenizerFamily {
    fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        match self {
            Self::ClipBpe(tokenizer) => tokenizer.encode(text, cancellation),
            Self::Gemma(tokenizer) => tokenizer.encode(text, cancellation),
            Self::Llama3ByteBpe(tokenizer) => tokenizer.encode(text, cancellation),
            Self::Qwen2ByteBpe(tokenizer) => tokenizer.encode(text, cancellation),
            Self::SentencePiece(tokenizer) => tokenizer.encode(text, cancellation),
        }
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        match self {
            Self::ClipBpe(tokenizer) => tokenizer.decode(tokens, skip_special, cancellation),
            Self::Gemma(tokenizer) => tokenizer.decode(tokens, skip_special, cancellation),
            Self::Llama3ByteBpe(tokenizer) => tokenizer.decode(tokens, skip_special, cancellation),
            Self::Qwen2ByteBpe(tokenizer) => tokenizer.decode(tokens, skip_special, cancellation),
            Self::SentencePiece(tokenizer) => tokenizer.decode(tokens, skip_special, cancellation),
        }
    }
}

#[derive(Clone)]
pub struct NativePromptTokenizer {
    family: NativeTokenizerFamily,
    configuration: TokenizerConfiguration,
    embeddings: BTreeMap<String, TextualInversionEmbedding>,
}

impl NativePromptTokenizer {
    pub fn configuration(&self) -> &TokenizerConfiguration {
        &self.configuration
    }

    pub fn qwen2_profile(&self) -> Option<Qwen2PretokenizerProfile> {
        match &self.family {
            NativeTokenizerFamily::Qwen2ByteBpe(tokenizer) => Some(tokenizer.profile()),
            NativeTokenizerFamily::ClipBpe(_)
            | NativeTokenizerFamily::Gemma(_)
            | NativeTokenizerFamily::Llama3ByteBpe(_)
            | NativeTokenizerFamily::SentencePiece(_) => None,
        }
    }

    pub fn qwen2_artifact_digest(&self) -> Option<&str> {
        match &self.family {
            NativeTokenizerFamily::Qwen2ByteBpe(tokenizer) => Some(tokenizer.artifact_digest()),
            NativeTokenizerFamily::ClipBpe(_)
            | NativeTokenizerFamily::Gemma(_)
            | NativeTokenizerFamily::Llama3ByteBpe(_)
            | NativeTokenizerFamily::SentencePiece(_) => None,
        }
    }

    pub fn gemma_profile(&self) -> Option<GemmaTokenizerProfile> {
        match &self.family {
            NativeTokenizerFamily::Gemma(tokenizer) => Some(tokenizer.profile()),
            NativeTokenizerFamily::ClipBpe(_)
            | NativeTokenizerFamily::Llama3ByteBpe(_)
            | NativeTokenizerFamily::Qwen2ByteBpe(_)
            | NativeTokenizerFamily::SentencePiece(_) => None,
        }
    }

    pub fn gemma_artifact_digest(&self) -> Option<&str> {
        match &self.family {
            NativeTokenizerFamily::Gemma(tokenizer) => Some(tokenizer.artifact_digest()),
            NativeTokenizerFamily::ClipBpe(_)
            | NativeTokenizerFamily::Llama3ByteBpe(_)
            | NativeTokenizerFamily::Qwen2ByteBpe(_)
            | NativeTokenizerFamily::SentencePiece(_) => None,
        }
    }

    pub fn has_textual_inversion_embeddings(&self) -> bool {
        !self.embeddings.is_empty()
    }

    pub fn semantic_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-prompt-tokenizer.v2");
        hash_tokenizer_configuration(&mut hasher, &self.configuration)?;
        match &self.family {
            NativeTokenizerFamily::ClipBpe(tokenizer) => {
                hasher.update(b"clip-bpe");
                hasher.update(tokenizer.tokenizer.identity().digest().as_bytes());
            }
            NativeTokenizerFamily::Gemma(tokenizer) => {
                hasher.update(b"gemma");
                hasher.update([match tokenizer.profile() {
                    GemmaTokenizerProfile::Gemma3SentencePiece => 0,
                    GemmaTokenizerProfile::Gemma4TokenizerJson => 1,
                }]);
                hasher.update(tokenizer.artifact_digest().as_bytes());
            }
            NativeTokenizerFamily::Llama3ByteBpe(tokenizer) => {
                hasher.update(b"llama3-byte-bpe");
                hasher.update(tokenizer.artifact_digest().as_bytes());
            }
            NativeTokenizerFamily::Qwen2ByteBpe(tokenizer) => {
                hasher.update(b"qwen2-byte-bpe");
                hasher.update(tokenizer.artifact_digest().as_bytes());
            }
            NativeTokenizerFamily::SentencePiece(tokenizer) => {
                hasher.update(b"sentencepiece");
                hasher.update(tokenizer.artifact_sha256.as_bytes());
                for entry in &tokenizer.entries {
                    cancellation.check()?;
                    hasher.update([0]);
                    hasher.update(entry.piece.as_bytes());
                    hasher.update(entry.score.to_bits().to_le_bytes());
                    hasher.update([match entry.piece_type {
                        SentencePieceType::Normal => 0,
                        SentencePieceType::Unknown => 1,
                        SentencePieceType::Control => 2,
                        SentencePieceType::UserDefined => 3,
                        SentencePieceType::Unused => 4,
                        SentencePieceType::Byte => 5,
                    }]);
                }
            }
        }
        for (name, embedding) in &self.embeddings {
            cancellation.check()?;
            hasher.update([0]);
            hasher.update(name.as_bytes());
            hasher.update([0]);
            hasher.update(embedding.artifact_sha256().as_bytes());
        }
        cancellation.check()?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn encode_numeric(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        cancellation.check()?;
        if text.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(text.len()));
        }
        let content = self.family.encode(text, cancellation)?;
        let unpadded_count = content
            .len()
            .checked_add(usize::from(self.configuration.start_token.is_some()))
            .and_then(|count| {
                count.checked_add(usize::from(self.configuration.end_token.is_some()))
            })
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "numeric prompt tokens",
            ))?;
        let token_count = unpadded_count.max(self.configuration.minimum_length.unwrap_or(0));
        if token_count == 0 || token_count > self.configuration.maximum_length {
            return Err(NativeTokenizerError::TooManyTokenValues(token_count));
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(token_count)
            .map_err(|_| NativeTokenizerError::Allocation("numeric prompt tokens"))?;
        let padding = token_count.checked_sub(unpadded_count).ok_or(
            NativeTokenizerError::ArithmeticOverflow("numeric prompt padding"),
        )?;
        if self.configuration.pad_left {
            tokens.extend(std::iter::repeat_n(self.configuration.pad_token, padding));
        }
        tokens.extend(self.configuration.start_token);
        tokens.extend(content);
        tokens.extend(self.configuration.end_token);
        if !self.configuration.pad_left {
            tokens.extend(std::iter::repeat_n(self.configuration.pad_token, padding));
        }
        cancellation.check()?;
        Ok(tokens)
    }

    pub fn decode_numeric(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        if tokens.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(tokens.len()));
        }
        let decoded = self.family.decode(tokens, skip_special, cancellation)?;
        cancellation.check()?;
        Ok(decoded)
    }

    pub fn decode_generated(
        &self,
        tokens: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTokenizerError> {
        cancellation.check()?;
        if tokens.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(tokens.len()));
        }
        let decoded = match &self.family {
            NativeTokenizerFamily::Gemma(tokenizer) => {
                tokenizer.decode_generated(tokens, cancellation)?
            }
            NativeTokenizerFamily::ClipBpe(_)
            | NativeTokenizerFamily::Llama3ByteBpe(_)
            | NativeTokenizerFamily::Qwen2ByteBpe(_)
            | NativeTokenizerFamily::SentencePiece(_) => {
                self.family.decode(tokens, true, cancellation)?
            }
        };
        cancellation.check()?;
        Ok(decoded)
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeTokenizerError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| NativeTokenizerError::ArithmeticOverflow("tokenizer residency"))?;
        let family_bytes = match &self.family {
            NativeTokenizerFamily::ClipBpe(tokenizer) => tokenizer
                .tokenizer
                .resident_bytes()
                .map_err(NativeTokenizerError::from)?,
            NativeTokenizerFamily::Gemma(tokenizer) => tokenizer.resident_bytes()?,
            NativeTokenizerFamily::Qwen2ByteBpe(tokenizer) => tokenizer.resident_bytes()?,
            NativeTokenizerFamily::Llama3ByteBpe(tokenizer) => tokenizer.resident_bytes()?,
            NativeTokenizerFamily::SentencePiece(tokenizer) => {
                let mut family_bytes = u64::try_from(std::mem::size_of::<SentencePieceTokenizer>())
                    .map_err(|_| {
                        NativeTokenizerError::ArithmeticOverflow("SentencePiece residency")
                    })?;
                for entry in &tokenizer.entries {
                    family_bytes = family_bytes
                        .checked_add(u64::try_from(entry.piece.capacity()).map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("SentencePiece text")
                        })?)
                        .ok_or(NativeTokenizerError::ArithmeticOverflow(
                            "SentencePiece residency",
                        ))?;
                }
                family_bytes = family_bytes
                    .checked_add(
                        u64::try_from(
                            tokenizer
                                .control_tokens
                                .capacity()
                                .checked_mul(std::mem::size_of::<u32>())
                                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                                    "SentencePiece controls",
                                ))?,
                        )
                        .map_err(|_| {
                            NativeTokenizerError::ArithmeticOverflow("SentencePiece controls")
                        })?,
                    )
                    .ok_or(NativeTokenizerError::ArithmeticOverflow(
                        "SentencePiece residency",
                    ))?;
                family_bytes
            }
        };
        bytes = bytes
            .checked_add(family_bytes)
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "tokenizer residency",
            ))?;
        for (name, embedding) in &self.embeddings {
            let embedding_bytes = embedding.rows().iter().try_fold(0_u64, |total, row| {
                let row_bytes = row
                    .len()
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or(NativeTokenizerError::ArithmeticOverflow(
                        "embedding residency",
                    ))?;
                total
                    .checked_add(row_bytes)
                    .ok_or(NativeTokenizerError::ArithmeticOverflow(
                        "embedding residency",
                    ))
            })?;
            bytes = bytes
                .checked_add(u64::try_from(name.capacity()).map_err(|_| {
                    NativeTokenizerError::ArithmeticOverflow("embedding name residency")
                })?)
                .and_then(|value| value.checked_add(embedding_bytes))
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "embedding residency",
                ))?;
        }
        Ok(bytes)
    }

    pub fn empty_token_ids(
        start_token: Option<u32>,
        end_token: Option<u32>,
        pad_token: u32,
        length: usize,
    ) -> Result<Vec<u32>, NativeTokenizerError> {
        let special_count = usize::from(start_token.is_some())
            .checked_add(usize::from(end_token.is_some()))
            .ok_or(NativeTokenizerError::ArithmeticOverflow(
                "empty special tokens",
            ))?;
        if length < special_count || length > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::InvalidEmptyTokenLength(length));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| NativeTokenizerError::Allocation("empty tokens"))?;
        output.extend(start_token);
        output.extend(end_token);
        output.resize(length, pad_token);
        Ok(output)
    }

    pub fn checked(
        family: NativeTokenizerFamily,
        configuration: TokenizerConfiguration,
        embeddings: BTreeMap<String, TextualInversionEmbedding>,
    ) -> Result<Self, NativeTokenizerError> {
        let configuration = configuration.checked()?;
        if let NativeTokenizerFamily::Gemma(tokenizer) = &family {
            match tokenizer.profile() {
                GemmaTokenizerProfile::Gemma3SentencePiece => {
                    if configuration.start_token != Some(GEMMA4_START_TOKEN)
                        || configuration.end_token.is_some()
                        || configuration.pad_token != GEMMA4_PAD_TOKEN
                        || !configuration.pad_left
                        || !configuration.disable_weights
                        || configuration.pad_to_maximum_length
                        || configuration
                            .minimum_length
                            .is_none_or(|length| length == 0)
                    {
                        return Err(NativeTokenizerError::InvalidConfiguration(
                            "Gemma3 requires BOS 2, pad 0, no EOS, left minimum padding, and disabled prompt weights"
                                .to_owned(),
                        ));
                    }
                }
                GemmaTokenizerProfile::Gemma4TokenizerJson => {
                    if configuration.start_token != Some(GEMMA4_START_TOKEN)
                        || configuration.end_token.is_some()
                        || configuration.pad_token != GEMMA4_PAD_TOKEN
                        || !configuration.pad_left
                        || !configuration.disable_weights
                        || configuration.pad_to_maximum_length
                        || configuration.minimum_length != Some(1)
                    {
                        return Err(NativeTokenizerError::InvalidConfiguration(
                            "Gemma4 requires BOS 2, pad 0, no EOS, minimum length 1, left padding, and disabled prompt weights"
                                .to_owned(),
                        ));
                    }
                }
            }
        }
        if embeddings.keys().any(|name| name.trim().is_empty()) {
            return Err(NativeTokenizerError::InvalidEmbeddingName);
        }
        if !embeddings.is_empty() {
            let expected_width = configuration.embedding_width.ok_or_else(|| {
                NativeTokenizerError::InvalidConfiguration(
                    "textual inversion requires an explicit embedding width".to_owned(),
                )
            })?;
            if embeddings
                .values()
                .any(|embedding| embedding.width() != expected_width)
            {
                return Err(NativeTokenizerError::EmbeddingWidthMismatch {
                    expected: expected_width,
                });
            }
            let first = embeddings
                .values()
                .next()
                .ok_or(NativeTokenizerError::InvalidEmbeddingName)?;
            if embeddings
                .values()
                .any(|embedding| !Arc::ptr_eq(embedding.store_identity(), first.store_identity()))
            {
                return Err(NativeTokenizerError::ArtifactMismatch(
                    first.artifact_key().clone(),
                ));
            }
        }
        Ok(Self {
            family,
            configuration,
            embeddings,
        })
    }

    pub fn tokenize(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<NativeTokenizedPrompt, NativeTokenizerError> {
        self.tokenize_with_options(
            text,
            NativeTokenizerInvocationOptions::default(),
            cancellation,
        )
    }

    pub fn tokenize_with_options(
        &self,
        text: &str,
        options: NativeTokenizerInvocationOptions,
        cancellation: &CancellationToken,
    ) -> Result<NativeTokenizedPrompt, NativeTokenizerError> {
        let options = options.validate()?;
        let mut tokenizer = self.clone();
        if let Some(minimum_length) = options.minimum_length {
            tokenizer.configuration.minimum_length = Some(minimum_length);
        }
        if let Some(minimum_padding) = options.minimum_padding {
            tokenizer.configuration.minimum_padding = Some(minimum_padding);
        }
        if let Some(disable_weights) = options.disable_weights {
            tokenizer.configuration.disable_weights = disable_weights;
        }
        tokenizer.tokenize_effective(text, cancellation)
    }

    fn tokenize_effective(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<NativeTokenizedPrompt, NativeTokenizerError> {
        cancellation.check()?;
        if text.len() > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::PromptTooLarge(text.len()));
        }
        let segments = if self.configuration.disable_weights {
            let mut owned_text = String::new();
            owned_text
                .try_reserve_exact(text.len())
                .map_err(|_| NativeTokenizerError::Allocation("unweighted prompt"))?;
            owned_text.push_str(text);
            let mut segments = Vec::new();
            segments
                .try_reserve_exact(1)
                .map_err(|_| NativeTokenizerError::Allocation("prompt weight segments"))?;
            segments.push(PromptWeightSegment::checked(owned_text, 1.0)?);
            segments
        } else {
            parse_prompt_weights(text, cancellation)?
        };
        let mut groups = Vec::<Vec<(NativeTokenValue, f32)>>::new();
        for segment in segments {
            for part in split_embedding_references(segment.text(), cancellation)? {
                cancellation.check()?;
                match part {
                    PromptPart::Text(text) if !text.is_empty() => {
                        let encoded = self.family.encode(text, cancellation)?;
                        if !encoded.is_empty() {
                            let mut group = Vec::new();
                            group.try_reserve_exact(encoded.len()).map_err(|_| {
                                NativeTokenizerError::Allocation("numeric token group")
                            })?;
                            group.extend(
                                encoded.into_iter().map(|token| {
                                    (NativeTokenValue::Token(token), segment.weight())
                                }),
                            );
                            groups
                                .try_reserve(1)
                                .map_err(|_| NativeTokenizerError::Allocation("token groups"))?;
                            groups.push(group);
                        }
                    }
                    PromptPart::Embedding(name) => {
                        let (embedding, comma_fallback) = match self.embeddings.get(name) {
                            Some(embedding) => (Some(embedding), None),
                            None => {
                                let stripped = name.trim_matches(',');
                                if stripped.len() < name.len() {
                                    (self.embeddings.get(stripped), Some(&name[stripped.len()..]))
                                } else {
                                    (None, None)
                                }
                            }
                        };
                        if let Some(embedding) = embedding {
                            let mut group = Vec::new();
                            group
                                .try_reserve_exact(embedding.rows().len())
                                .map_err(|_| {
                                    NativeTokenizerError::Allocation("embedding token group")
                                })?;
                            for row in embedding.rows() {
                                group.push((
                                    NativeTokenValue::Embedding {
                                        artifact_key: embedding.artifact_key().clone(),
                                        artifact_sha256: embedding.artifact_sha256().to_owned(),
                                        values: row.clone(),
                                    },
                                    segment.weight(),
                                ));
                            }
                            groups
                                .try_reserve(1)
                                .map_err(|_| NativeTokenizerError::Allocation("token groups"))?;
                            groups.push(group);
                        }
                        if let Some(fallback) = comma_fallback.filter(|text| !text.is_empty()) {
                            let encoded = self.family.encode(fallback, cancellation)?;
                            if !encoded.is_empty() {
                                let mut group = Vec::new();
                                group.try_reserve_exact(encoded.len()).map_err(|_| {
                                    NativeTokenizerError::Allocation("fallback token group")
                                })?;
                                group.extend(encoded.into_iter().map(|token| {
                                    (NativeTokenValue::Token(token), segment.weight())
                                }));
                                groups.try_reserve(1).map_err(|_| {
                                    NativeTokenizerError::Allocation("token groups")
                                })?;
                                groups.push(group);
                            }
                        }
                    }
                    PromptPart::Text(_) => {}
                }
            }
        }
        self.pack(groups, cancellation)
    }

    pub fn tokenize_list(
        &self,
        prompts: &[String],
        cancellation: &CancellationToken,
    ) -> Result<Vec<NativeTokenizedPrompt>, NativeTokenizerError> {
        if prompts.is_empty() || prompts.len() > MAX_NATIVE_PROMPT_BATCH {
            return Err(NativeTokenizerError::InvalidBatchSize(prompts.len()));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(prompts.len())
            .map_err(|_| NativeTokenizerError::Allocation("prompt batch"))?;
        for prompt in prompts {
            output.push(self.tokenize(prompt, cancellation)?);
        }
        Ok(output)
    }

    pub fn tokenize_list_with_options(
        &self,
        prompts: &[String],
        options: NativeTokenizerInvocationOptions,
        cancellation: &CancellationToken,
    ) -> Result<Vec<NativeTokenizedPrompt>, NativeTokenizerError> {
        if prompts.is_empty() || prompts.len() > MAX_NATIVE_PROMPT_BATCH {
            return Err(NativeTokenizerError::InvalidBatchSize(prompts.len()));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(prompts.len())
            .map_err(|_| NativeTokenizerError::Allocation("prompt batch"))?;
        for prompt in prompts {
            output.push(self.tokenize_with_options(prompt, options, cancellation)?);
        }
        Ok(output)
    }

    fn pack(
        &self,
        groups: Vec<Vec<(NativeTokenValue, f32)>>,
        cancellation: &CancellationToken,
    ) -> Result<NativeTokenizedPrompt, NativeTokenizerError> {
        let special_end = usize::from(self.configuration.end_token.is_some());
        let mut sections = Vec::new();
        let mut current = self.new_section()?;
        let mut next_word_id = 1_u64;
        let mut aggregate_values = 0_usize;
        for mut group in groups {
            cancellation.check()?;
            aggregate_values = aggregate_values.checked_add(group.len()).ok_or(
                NativeTokenizerError::ArithmeticOverflow("aggregate prompt tokens"),
            )?;
            if aggregate_values > MAX_NATIVE_PROMPT_BYTES {
                return Err(NativeTokenizerError::TooManyTokenValues(aggregate_values));
            }
            let empty_section_content_capacity = self
                .configuration
                .maximum_length
                .checked_sub(usize::from(self.configuration.start_token.is_some()))
                .and_then(|value| value.checked_sub(special_end))
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "empty section content capacity",
                ))?;
            let large = group.len() >= self.configuration.maximum_word_length
                || group.len() > empty_section_content_capacity;
            while !group.is_empty() {
                let remaining = self
                    .configuration
                    .maximum_length
                    .checked_sub(current.len())
                    .and_then(|value| value.checked_sub(special_end))
                    .ok_or(NativeTokenizerError::ArithmeticOverflow("section capacity"))?;
                if group.len() > remaining {
                    if large && remaining > 0 {
                        for (value, weight) in group.drain(..remaining) {
                            current.push(NativeWeightedToken {
                                value,
                                weight,
                                word_id: next_word_id,
                            });
                        }
                    }
                    self.finish_section(&mut current, false)?;
                    reserve_tokenizer_values(&mut sections, 1, "token sections")?;
                    sections.push(NativeTokenSection { tokens: current });
                    if sections.len() == MAX_NATIVE_TOKEN_SECTIONS {
                        return Err(NativeTokenizerError::TooManySections(sections.len() + 1));
                    }
                    current = self.new_section()?;
                } else {
                    for (value, weight) in group.drain(..) {
                        current.push(NativeWeightedToken {
                            value,
                            weight,
                            word_id: next_word_id,
                        });
                    }
                }
            }
            next_word_id = next_word_id
                .checked_add(1)
                .ok_or(NativeTokenizerError::ArithmeticOverflow("word identity"))?;
        }
        self.finish_section(&mut current, true)?;
        reserve_tokenizer_values(&mut sections, 1, "token sections")?;
        sections.push(NativeTokenSection { tokens: current });
        let published_values =
            sections.iter().try_fold(0_usize, |total, section| {
                total.checked_add(section.tokens.len()).ok_or(
                    NativeTokenizerError::ArithmeticOverflow("published prompt token values"),
                )
            })?;
        if published_values > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(published_values));
        }
        Ok(NativeTokenizedPrompt { sections })
    }

    fn new_section(&self) -> Result<Vec<NativeWeightedToken>, NativeTokenizerError> {
        let mut section = Vec::new();
        if let Some(token) = self.configuration.start_token {
            section
                .try_reserve_exact(1)
                .map_err(|_| NativeTokenizerError::Allocation("section start token"))?;
            section.push(NativeWeightedToken::token(token, 1.0, 0));
        }
        Ok(section)
    }

    fn finish_section(
        &self,
        section: &mut Vec<NativeWeightedToken>,
        final_section: bool,
    ) -> Result<(), NativeTokenizerError> {
        if let Some(token) = self.configuration.end_token {
            section
                .try_reserve(1)
                .map_err(|_| NativeTokenizerError::Allocation("section end token"))?;
            section.push(NativeWeightedToken::token(token, 1.0, 0));
        }
        if final_section {
            if let Some(padding) = self.configuration.minimum_padding {
                self.pad(section, padding)?;
            }
        }
        if self.configuration.pad_to_maximum_length
            && section.len() < self.configuration.maximum_length
        {
            self.pad(section, self.configuration.maximum_length - section.len())?;
        }
        if final_section {
            if let Some(minimum) = self.configuration.minimum_length {
                if section.len() < minimum {
                    self.pad(section, minimum - section.len())?;
                }
            }
        }
        Ok(())
    }

    fn pad(
        &self,
        section: &mut Vec<NativeWeightedToken>,
        count: usize,
    ) -> Result<(), NativeTokenizerError> {
        let padded_length =
            section
                .len()
                .checked_add(count)
                .ok_or(NativeTokenizerError::ArithmeticOverflow(
                    "token padding length",
                ))?;
        if padded_length > MAX_NATIVE_PROMPT_BYTES {
            return Err(NativeTokenizerError::TooManyTokenValues(padded_length));
        }
        section
            .try_reserve(count)
            .map_err(|_| NativeTokenizerError::Allocation("token padding"))?;
        if self.configuration.pad_left {
            section.splice(
                0..0,
                std::iter::repeat_n(
                    NativeWeightedToken::token(self.configuration.pad_token, 1.0, 0),
                    count,
                ),
            );
        } else {
            section.extend(std::iter::repeat_n(
                NativeWeightedToken::token(self.configuration.pad_token, 1.0, 0),
                count,
            ));
        }
        Ok(())
    }
}

enum PromptPart<'a> {
    Text(&'a str),
    Embedding(&'a str),
}

fn split_embedding_references<'a>(
    text: &'a str,
    cancellation: &CancellationToken,
) -> Result<Vec<PromptPart<'a>>, NativeTokenizerError> {
    let mut output = Vec::new();
    let mut cursor = 0;
    while let Some((relative, marker_length)) = if cursor == 0 && text.starts_with("embedding:") {
        Some((0, "embedding:".len()))
    } else {
        [
            text[cursor..]
                .find(" embedding:")
                .map(|index| (index, " embedding:".len())),
            text[cursor..]
                .find("\nembedding:")
                .map(|index| (index, "\nembedding:".len())),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|(index, _)| *index)
    } {
        cancellation.check()?;
        let marker = cursor + relative;
        if marker > cursor {
            output
                .try_reserve(1)
                .map_err(|_| NativeTokenizerError::Allocation("embedding references"))?;
            output.push(PromptPart::Text(&text[cursor..marker]));
        }
        let name_start = marker + marker_length;
        let name_end = text[name_start..]
            .find(char::is_whitespace)
            .map(|index| name_start + index)
            .unwrap_or(text.len());
        output
            .try_reserve(1)
            .map_err(|_| NativeTokenizerError::Allocation("embedding references"))?;
        output.push(PromptPart::Embedding(&text[name_start..name_end]));
        cursor = name_end;
    }
    if cursor < text.len() {
        output
            .try_reserve(1)
            .map_err(|_| NativeTokenizerError::Allocation("embedding references"))?;
        output.push(PromptPart::Text(&text[cursor..]));
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum NativeTokenizerError {
    #[error(transparent)]
    Clip(#[from] ClipError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Cancellation(#[from] sim_types::CancellationError),
    #[error(transparent)]
    ModelStore(#[from] ModelStoreError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("prompt byte length exceeds the native bound: {0}")]
    PromptTooLarge(usize),
    #[error("prompt has too many weighted segments: {0}")]
    TooManyWeightSegments(usize),
    #[error("prompt weighting nesting exceeds the native bound: {0}")]
    WeightNestingTooDeep(usize),
    #[error("token weight must be finite, got {0}")]
    InvalidWeight(f32),
    #[error("native tokenizer configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("native tokenizer vocabulary is invalid")]
    InvalidVocabulary,
    #[error("native tokenizer vocabulary JSON is invalid: {0}")]
    InvalidVocabularyJson(String),
    #[error("native tokenizer configuration is invalid: {0}")]
    InvalidTokenizerConfiguration(String),
    #[error("native tokenizer merge table is invalid")]
    InvalidMerges,
    #[error("native tokenizer pretokenization failed: {0}")]
    Pretokenization(String),
    #[error("unknown token ID {0}")]
    UnknownToken(u32),
    #[error("Gemma3 external special token {0} requires skip-special decoding")]
    UnsupportedSpecialTokenDecode(u32),
    #[error("CLIP vocabulary contains an invalid byte-encoding character {0:?}")]
    InvalidClipByte(char),
    #[error("Qwen2 vocabulary contains an invalid byte-encoding character {0:?}")]
    InvalidQwenByte(char),
    #[error("decoded CLIP bytes are not valid UTF-8")]
    InvalidDecodedUtf8,
    #[error("invalid UTF-8 token boundary")]
    InvalidUtf8Boundary,
    #[error("textual inversion embedding name is invalid")]
    InvalidEmbeddingName,
    #[error("embedding tensor selector is invalid")]
    InvalidEmbeddingSelector,
    #[error("verified embedding artifact contains no selectable tensor")]
    MissingEmbeddingTensor,
    #[error("textual inversion artifact does not match canonical record {0:?}")]
    ArtifactMismatch(ArtifactKey),
    #[error("textual inversion embedding has {values} values for width {width}")]
    InvalidEmbeddingShape { values: usize, width: usize },
    #[error("textual inversion embedding width does not match required width {expected}")]
    EmbeddingWidthMismatch { expected: usize },
    #[error("textual inversion embedding contains a non-finite value")]
    NonFiniteEmbedding,
    #[error("textual inversion embedding dtype {0:?} is unsupported")]
    UnsupportedEmbeddingDType(String),
    #[error("empty token length is invalid: {0}")]
    InvalidEmptyTokenLength(usize),
    #[error("CLIP token-weight projection shape is invalid")]
    InvalidWeightProjection,
    #[error("CLIP token-weight projection produced a non-finite value")]
    NonFiniteWeightProjection,
    #[error("prompt batch size must be in 1..={MAX_NATIVE_PROMPT_BATCH}, got {0}")]
    InvalidBatchSize(usize),
    #[error("prompt requires too many token sections: {0}")]
    TooManySections(usize),
    #[error("prompt contains too many token or embedding values: {0}")]
    TooManyTokenValues(usize),
    #[error("native tokenizer allocation failed for {0}")]
    Allocation(&'static str),
    #[error("native tokenizer arithmetic overflowed while computing {0}")]
    ArithmeticOverflow(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_reserve_failure_is_typed_and_leaves_staging_unchanged() {
        let mut staged = Vec::<NativeTokenSection>::new();
        assert!(matches!(
            reserve_tokenizer_values(&mut staged, usize::MAX, "token sections"),
            Err(NativeTokenizerError::Allocation("token sections"))
        ));
        assert!(staged.is_empty());
    }
    #[test]
    fn source_textual_inversion_predicate_uses_canonical_weight_and_reference_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        for (text, expected) in [
            ("a test", false),
            ("notembedding:name", false),
            ("a\tembedding:name", false),
            ("embedding:name", true),
            ("a embedding:name", true),
            ("a\nembedding:name", true),
            ("(embedding:name:1.25)", true),
            ("a (embedding:name) b", true),
        ] {
            assert_eq!(
                source_prompt_requires_textual_inversion(text, &cancellation)?,
                expected,
                "{text:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn source_textual_inversion_predicate_propagates_bounds_cancellation_and_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            source_prompt_requires_textual_inversion("embedding:name", &cancelled),
            Err(NativeTokenizerError::Cancellation(_))
        ));
        assert!(matches!(
            source_prompt_requires_textual_inversion("(embedding:name:nan)", &cancellation),
            Err(NativeTokenizerError::InvalidWeight(_))
        ));
        assert!(matches!(
            source_prompt_requires_textual_inversion(
                &"x".repeat(MAX_NATIVE_PROMPT_BYTES + 1),
                &cancellation
            ),
            Err(NativeTokenizerError::PromptTooLarge(_))
        ));
        assert!(source_prompt_requires_textual_inversion(
            "embedding:name",
            &cancellation
        )?);
        Ok(())
    }
}
