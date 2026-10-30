use crate::{
    attention::{
        AttentionBackend, AttentionFallbackPolicy, AttentionRequest,
        scaled_dot_product_attention_with_context,
    },
    native_node_payload::{AudioEncoderOutput, NativeModelPayloadError},
    native_ops::{GeluApproximation, NativeModule},
};
use sha2::{Digest, Sha256};
use sim_media::NativeAudioPayload;
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, StorageId, StreamId, Tensor,
    TensorBackend, TensorError,
    generated_external_tensor_kernel_01::{
        NativeMelNormalization, NativeMelScale, NativeMelSpectrogramConfiguration,
        NativeResampleConfiguration, mel_spectrogram_with_context_exact_native,
        resample_with_context_exact_native,
    },
    generated_indexing_masking_01::narrow_function_exact_native,
    generated_reduction_01::{
        tensor_max_with_context_exact_native, tensor_mean_with_context_exact_native,
        tensor_var_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        FunctionalPadMode, functional_pad_with_context_exact_native, tensor_transpose_exact_native,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, tensor_from_f32_with_context_exact_native,
        tensor_to_f32_with_context_exact_native,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};
use thiserror::Error;

pub const AUDIO_ENCODERS_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/sim/audio_encoders/audio_encoders.py";
pub const AUDIO_ENCODERS_SOURCE_SHA256: &str =
    "c8d3260799ea0222b6bf9e1bde8d16f105a46aaf1944213bf66fdaa05433dec8";
pub const NODES_AUDIO_ENCODER_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/comfy_extras/nodes_audio_encoder.py";
pub const NODES_AUDIO_ENCODER_SOURCE_SHA256: &str =
    "fbc6f4d8ca0e099dc2f35f9420dce9aced3763325b9804b031a1641db2bb4a8a";
pub const WAV2VEC2_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/audio_encoders/wav2vec2.py";
pub const WAV2VEC2_SOURCE_SHA256: &str =
    "32494297021e54e42845276255a026ce8ab62be5d54f6d40cecfb042c8b238e7";
pub const WHISPER_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/audio_encoders/whisper.py";
pub const WHISPER_SOURCE_SHA256: &str =
    "f0e214e79fdfa9926fbc863038fe3e0455caa61f486a1b37a3039ca7253dea22";

const WAV2VEC2_PREFIX: &str = "wav2vec2.";
const WHISPER_PREFIX: &str = "model.";
const WAV2VEC2_MARKER: &str = "encoder.layer_norm.bias";
const WHISPER_MARKER: &str = "model.encoder.embed_positions.weight";
const WAV2VEC2_CONV_DIMENSION: usize = 512;
const WHISPER_MEL_BINS: usize = 128;
const WHISPER_CONTEXT: usize = 1_500;
const WHISPER_STATE: usize = 1_280;
const WHISPER_HEADS: usize = 20;
const WHISPER_LAYERS: usize = 32;
const MODEL_SAMPLE_RATE: u32 = 16_000;
const MAX_AUDIO_STATE_TENSORS: usize = 2_048;
const MAX_AUDIO_STATE_KEY_BYTES: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeAudioEncoderArchitecture {
    Wav2Vec2Base,
    Wav2Vec2Large,
    WhisperLargeV3,
}

pub type AudioEncoderProfile = NativeAudioEncoderArchitecture;

#[derive(Clone, Debug, Eq, PartialEq)]
struct AudioEncoderExecutionProfile {
    convolution_width: usize,
    hidden_width: usize,
    attention_heads: usize,
    transformer_layers: usize,
    feed_forward_width: usize,
    positional_kernel: usize,
    positional_groups: usize,
    mel_bins: usize,
    audio_context: usize,
    audio_samples: usize,
    fft_length: usize,
    hop_length: usize,
}

impl AudioEncoderExecutionProfile {
    fn source(architecture: NativeAudioEncoderArchitecture) -> Self {
        match architecture {
            NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large => {
                let hidden_width = architecture.hidden_width();
                Self {
                    convolution_width: WAV2VEC2_CONV_DIMENSION,
                    hidden_width,
                    attention_heads: architecture.attention_heads(),
                    transformer_layers: architecture.transformer_layers(),
                    feed_forward_width: hidden_width * 4,
                    positional_kernel: 128,
                    positional_groups: 16,
                    mel_bins: 0,
                    audio_context: 0,
                    audio_samples: 0,
                    fft_length: 0,
                    hop_length: 0,
                }
            }
            NativeAudioEncoderArchitecture::WhisperLargeV3 => Self {
                convolution_width: 0,
                hidden_width: WHISPER_STATE,
                attention_heads: WHISPER_HEADS,
                transformer_layers: WHISPER_LAYERS,
                feed_forward_width: WHISPER_STATE * 4,
                positional_kernel: 0,
                positional_groups: 0,
                mel_bins: WHISPER_MEL_BINS,
                audio_context: WHISPER_CONTEXT,
                audio_samples: 480_000,
                fft_length: 400,
                hop_length: 160,
            },
        }
    }

    fn output_layers(&self) -> usize {
        self.transformer_layers + 1
    }

    #[cfg(any(test, feature = "test-support"))]
    fn reduced(architecture: NativeAudioEncoderArchitecture) -> Self {
        match architecture {
            NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large => Self {
                convolution_width: 2,
                hidden_width: 2,
                attention_heads: 1,
                transformer_layers: architecture.transformer_layers(),
                feed_forward_width: 4,
                positional_kernel: 4,
                positional_groups: 1,
                mel_bins: 0,
                audio_context: 0,
                audio_samples: 0,
                fft_length: 0,
                hop_length: 0,
            },
            NativeAudioEncoderArchitecture::WhisperLargeV3 => Self {
                convolution_width: 0,
                hidden_width: 2,
                attention_heads: 1,
                transformer_layers: WHISPER_LAYERS,
                feed_forward_width: 4,
                positional_kernel: 0,
                positional_groups: 0,
                mel_bins: 4,
                audio_context: 4,
                audio_samples: 32,
                fft_length: 16,
                hop_length: 4,
            },
        }
    }
}

impl NativeAudioEncoderArchitecture {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Wav2Vec2Base => "wav2vec2-base",
            Self::Wav2Vec2Large => "wav2vec2-large",
            Self::WhisperLargeV3 => "whisper-large-v3",
        }
    }

    pub const fn architecture_identifier(self) -> &'static str {
        self.identifier()
    }

    const fn hidden_width(self) -> usize {
        match self {
            Self::Wav2Vec2Base => 768,
            Self::Wav2Vec2Large => 1_024,
            Self::WhisperLargeV3 => WHISPER_STATE,
        }
    }

    const fn attention_heads(self) -> usize {
        match self {
            Self::Wav2Vec2Base => 12,
            Self::Wav2Vec2Large => 16,
            Self::WhisperLargeV3 => WHISPER_HEADS,
        }
    }

    const fn transformer_layers(self) -> usize {
        match self {
            Self::Wav2Vec2Base => 12,
            Self::Wav2Vec2Large => 24,
            Self::WhisperLargeV3 => WHISPER_LAYERS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeAudioEncoderCheckpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAudioEncoderDiagnostic {
    recognized_nonexecuting_state_keys: Arc<[String]>,
    missing_nonexecuting_state_keys: Arc<[String]>,
    unexpected_state_keys: Arc<[String]>,
}

impl NativeAudioEncoderDiagnostic {
    pub fn recognized_nonexecuting_state_keys(&self) -> &[String] {
        &self.recognized_nonexecuting_state_keys
    }

    pub fn missing_nonexecuting_state_keys(&self) -> &[String] {
        &self.missing_nonexecuting_state_keys
    }

    pub fn unexpected_state_keys(&self) -> &[String] {
        &self.unexpected_state_keys
    }
}

#[derive(Clone, Debug, Error)]
pub enum AudioEncoderError {
    #[error("audio encoder execution was cancelled")]
    Cancelled,
    #[error("audio encoder checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("audio encoder checkpoint has duplicate source key {0}")]
    DuplicateSourceKey(String),
    #[error("audio encoder architecture is unsupported: {0}")]
    UnsupportedArchitecture(String),
    #[error("audio encoder executable state is missing key {0}")]
    MissingState(String),
    #[error("audio encoder state {key} has invalid shape: expected {expected:?}, got {actual:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("audio encoder state {key} must use CPU F32 on stream {expected_stream:?}")]
    StatePlacement {
        key: String,
        expected_stream: sim_tensor::StreamId,
    },
    #[error("audio encoder input is invalid: {0}")]
    InvalidInput(String),
    #[error("audio encoder requires {required} bytes but its budget is {budget} bytes")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("audio encoder memory accounting overflowed")]
    MemoryOverflow,
    #[error("audio encoder allocation failed for {0}")]
    Allocation(&'static str),
    #[error("audio encoder semantic state changed")]
    SemanticStateChanged,
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("audio encoder tensor operation failed: {0}")]
    TensorOperation(String),
    #[error("audio encoder output construction failed: {0}")]
    Output(String),
}

impl From<sim_types::CancellationError> for AudioEncoderError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct NormalizedAudioEncoderCheckpoint {
    architecture: NativeAudioEncoderArchitecture,
    execution_profile: AudioEncoderExecutionProfile,
    artifact_sha256: String,
    executable_state: BTreeMap<String, Tensor>,
    recognized_nonexecuting_state_keys: Vec<String>,
    missing_nonexecuting_state_keys: Vec<String>,
    unexpected_state_keys: Vec<String>,
    memory_budget_bytes: u64,
    stream: StreamId,
}

impl NormalizedAudioEncoderCheckpoint {
    pub const fn architecture(&self) -> NativeAudioEncoderArchitecture {
        self.architecture
    }

    pub fn executable_state(&self) -> &BTreeMap<String, Tensor> {
        &self.executable_state
    }

    pub fn unexpected_state_keys(&self) -> &[String] {
        &self.unexpected_state_keys
    }
}

pub fn normalize_and_select_architecture(
    checkpoint: NativeAudioEncoderCheckpoint,
    context: &ExecutionContext<'_>,
) -> Result<NormalizedAudioEncoderCheckpoint, AudioEncoderError> {
    context.cancellation.check()?;
    validate_sha256(&checkpoint.artifact_sha256)?;
    if checkpoint.ordered_state.is_empty()
        || checkpoint.ordered_state.len() > MAX_AUDIO_STATE_TENSORS
        || checkpoint.memory_budget_bytes == 0
    {
        return Err(AudioEncoderError::InvalidCheckpoint(
            "state cardinality or memory budget is invalid".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut source = BTreeMap::new();
    let mut source_order = Vec::new();
    source_order
        .try_reserve_exact(checkpoint.ordered_state.len())
        .map_err(|_| AudioEncoderError::Allocation("ordered checkpoint keys"))?;
    for (index, (key, tensor)) in checkpoint.ordered_state.into_iter().enumerate() {
        if index.is_multiple_of(64) {
            context.cancellation.check()?;
        }
        validate_state_key(&key)?;
        if !seen.insert(key.clone()) {
            return Err(AudioEncoderError::DuplicateSourceKey(key));
        }
        source_order.push(key.clone());
        source.insert(key, tensor);
    }

    let prefixed_destinations = source_order
        .iter()
        .filter_map(|key| key.strip_prefix(WAV2VEC2_PREFIX))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for destination in &prefixed_destinations {
        if source.contains_key(destination) {
            return Err(AudioEncoderError::InvalidCheckpoint(format!(
                "Wav2Vec2 prefix normalization collides at {destination}"
            )));
        }
    }
    for key in &source_order {
        if let Some(stripped) = key.strip_prefix(WAV2VEC2_PREFIX) {
            let tensor = source.remove(key).ok_or_else(|| {
                AudioEncoderError::InvalidCheckpoint(
                    "ordered state changed during normalization".to_owned(),
                )
            })?;
            source.insert(stripped.to_owned(), tensor);
        }
    }

    let architecture = if let Some(marker) = source.get(WAV2VEC2_MARKER) {
        match marker.descriptor().shape() {
            [768] => NativeAudioEncoderArchitecture::Wav2Vec2Base,
            [1_024] => NativeAudioEncoderArchitecture::Wav2Vec2Large,
            shape => {
                return Err(AudioEncoderError::UnsupportedArchitecture(format!(
                    "Wav2Vec2 marker width {shape:?}"
                )));
            }
        }
    } else if source.contains_key(WHISPER_MARKER) {
        let model_keys = source
            .keys()
            .filter(|key| key.starts_with(WHISPER_PREFIX))
            .cloned()
            .collect::<Vec<_>>();
        for key in &model_keys {
            let stripped = key.strip_prefix(WHISPER_PREFIX).ok_or_else(|| {
                AudioEncoderError::InvalidCheckpoint(
                    "Whisper prefix normalization failed".to_owned(),
                )
            })?;
            if source.contains_key(stripped) {
                return Err(AudioEncoderError::InvalidCheckpoint(format!(
                    "Whisper prefix normalization collides at {stripped}"
                )));
            }
        }
        for key in model_keys {
            context.cancellation.check()?;
            let tensor = source.remove(&key).ok_or_else(|| {
                AudioEncoderError::InvalidCheckpoint(
                    "Whisper state changed during normalization".to_owned(),
                )
            })?;
            let stripped = key.strip_prefix(WHISPER_PREFIX).ok_or_else(|| {
                AudioEncoderError::InvalidCheckpoint(
                    "Whisper prefix normalization failed".to_owned(),
                )
            })?;
            source.insert(stripped.to_owned(), tensor);
        }
        NativeAudioEncoderArchitecture::WhisperLargeV3
    } else {
        return Err(AudioEncoderError::UnsupportedArchitecture(
            "no Wav2Vec2 or Whisper Large V3 marker".to_owned(),
        ));
    };

    let execution_profile = AudioEncoderExecutionProfile::source(architecture);
    let expected = expected_state_shapes(architecture, &execution_profile)?;
    let mut executable_state = BTreeMap::new();
    for (key, expected_shape) in &expected {
        context.cancellation.check()?;
        let tensor = source
            .remove(key)
            .ok_or_else(|| AudioEncoderError::MissingState(key.clone()))?;
        validate_state_tensor(key, &tensor, expected_shape, context)?;
        executable_state.insert(key.clone(), tensor);
    }
    let mut recognized_nonexecuting_state_keys = Vec::new();
    let mut missing_nonexecuting_state_keys = Vec::new();
    if matches!(
        architecture,
        NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large
    ) {
        let key = "masked_spec_embed".to_owned();
        if let Some(tensor) = source.remove(&key) {
            validate_state_tensor(
                &key,
                &tensor,
                &[u64::try_from(execution_profile.hidden_width)
                    .map_err(|_| AudioEncoderError::MemoryOverflow)?],
                context,
            )?;
            recognized_nonexecuting_state_keys.push(key);
        } else {
            missing_nonexecuting_state_keys.push(key);
        }
    }
    let unexpected_state_keys = source.into_keys().collect();
    Ok(NormalizedAudioEncoderCheckpoint {
        architecture,
        execution_profile,
        artifact_sha256: checkpoint.artifact_sha256,
        executable_state,
        recognized_nonexecuting_state_keys,
        missing_nonexecuting_state_keys,
        unexpected_state_keys,
        memory_budget_bytes: checkpoint.memory_budget_bytes,
        stream: context.stream,
    })
}

pub fn normalize_and_select_profile(
    checkpoint: NativeAudioEncoderCheckpoint,
    context: &ExecutionContext<'_>,
) -> Result<NormalizedAudioEncoderCheckpoint, AudioEncoderError> {
    normalize_and_select_architecture(checkpoint, context)
}

#[derive(Clone, Debug)]
pub struct NativeAudioEncoder {
    architecture: NativeAudioEncoderArchitecture,
    execution_profile: AudioEncoderExecutionProfile,
    artifact_sha256: String,
    executable_state: Arc<BTreeMap<String, Tensor>>,
    diagnostic: NativeAudioEncoderDiagnostic,
    memory_budget_bytes: u64,
    stream: StreamId,
    semantic_state_digest_sha256: String,
}

impl NativeAudioEncoder {
    pub fn from_checkpoint(
        checkpoint: NativeAudioEncoderCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, AudioEncoderError> {
        let normalized = normalize_and_select_architecture(checkpoint, context)?;
        Self::from_normalized(normalized, context.cancellation)
    }

    fn from_normalized(
        mut normalized: NormalizedAudioEncoderCheckpoint,
        cancellation: &CancellationToken,
    ) -> Result<Self, AudioEncoderError> {
        cancellation.check()?;
        normalized.executable_state = normalized
            .executable_state
            .into_iter()
            .map(|(mut key, tensor)| {
                key.shrink_to_fit();
                (key, tensor)
            })
            .collect();
        for key in normalized
            .recognized_nonexecuting_state_keys
            .iter_mut()
            .chain(normalized.missing_nonexecuting_state_keys.iter_mut())
            .chain(normalized.unexpected_state_keys.iter_mut())
        {
            key.shrink_to_fit();
        }
        normalized.artifact_sha256.shrink_to_fit();
        let mut resource = Self {
            architecture: normalized.architecture,
            execution_profile: normalized.execution_profile,
            artifact_sha256: normalized.artifact_sha256,
            executable_state: Arc::new(normalized.executable_state),
            diagnostic: NativeAudioEncoderDiagnostic {
                recognized_nonexecuting_state_keys: normalized
                    .recognized_nonexecuting_state_keys
                    .into(),
                missing_nonexecuting_state_keys: normalized.missing_nonexecuting_state_keys.into(),
                unexpected_state_keys: normalized.unexpected_state_keys.into(),
            },
            memory_budget_bytes: normalized.memory_budget_bytes,
            stream: normalized.stream,
            semantic_state_digest_sha256: String::new(),
        };
        resource.semantic_state_digest_sha256 =
            resource.project_semantic_state_digest(cancellation)?;
        resource.semantic_state_digest_sha256.shrink_to_fit();
        resource.validate(cancellation)?;
        let required = resource.resident_bytes()?;
        if required > resource.memory_budget_bytes {
            return Err(AudioEncoderError::OutOfMemory {
                required,
                budget: resource.memory_budget_bytes,
            });
        }
        cancellation.check()?;
        Ok(resource)
    }

    pub const fn architecture(&self) -> NativeAudioEncoderArchitecture {
        self.architecture
    }

    pub const fn identifier(&self) -> &'static str {
        self.architecture.identifier()
    }

    pub const fn architecture_identifier(&self) -> &'static str {
        self.identifier()
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn diagnostic(&self) -> &NativeAudioEncoderDiagnostic {
        &self.diagnostic
    }

    pub const fn memory_budget_bytes(&self) -> u64 {
        self.memory_budget_bytes
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), AudioEncoderError> {
        cancellation.check()?;
        validate_sha256(&self.artifact_sha256)?;
        self.validate_diagnostic()?;
        let expected = expected_state_shapes(self.architecture, &self.execution_profile)?;
        if expected.len() != self.executable_state.len() {
            return Err(AudioEncoderError::SemanticStateChanged);
        }
        for (key, shape) in expected {
            cancellation.check()?;
            let tensor = self
                .executable_state
                .get(&key)
                .ok_or_else(|| AudioEncoderError::MissingState(key.clone()))?;
            if tensor.descriptor().shape() != shape
                || tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(AudioEncoderError::SemanticStateChanged);
            }
        }
        if self.semantic_state_digest_sha256 != self.project_semantic_state_digest(cancellation)? {
            return Err(AudioEncoderError::SemanticStateChanged);
        }
        self.resident_tensor_allocations()?;
        self.resident_owned_bytes()?;
        cancellation.check()?;
        Ok(())
    }

    fn validate_diagnostic(&self) -> Result<(), AudioEncoderError> {
        let recognized = self.diagnostic.recognized_nonexecuting_state_keys();
        let missing = self.diagnostic.missing_nonexecuting_state_keys();
        let optional_state_is_valid = match self.architecture {
            NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large => {
                (matches!(recognized, [key] if key == "masked_spec_embed") && missing.is_empty())
                    || (matches!(missing, [key] if key == "masked_spec_embed")
                        && recognized.is_empty())
            }
            NativeAudioEncoderArchitecture::WhisperLargeV3 => {
                recognized.is_empty() && missing.is_empty()
            }
        };
        if !optional_state_is_valid {
            return Err(AudioEncoderError::SemanticStateChanged);
        }
        let mut previous = None;
        for key in self.diagnostic.unexpected_state_keys() {
            validate_state_key(key).map_err(|_| AudioEncoderError::SemanticStateChanged)?;
            if previous.is_some_and(|previous| previous >= key.as_str())
                || recognized.iter().any(|recognized| recognized == key)
                || missing.iter().any(|missing| missing == key)
            {
                return Err(AudioEncoderError::SemanticStateChanged);
            }
            previous = Some(key.as_str());
        }
        Ok(())
    }

    pub fn reconstruct(&self, cancellation: &CancellationToken) -> Result<Self, AudioEncoderError> {
        self.validate(cancellation)?;
        let normalized = NormalizedAudioEncoderCheckpoint {
            architecture: self.architecture,
            execution_profile: self.execution_profile.clone(),
            artifact_sha256: self.artifact_sha256.clone(),
            executable_state: self.executable_state.as_ref().clone(),
            unexpected_state_keys: self.diagnostic.unexpected_state_keys.to_vec(),
            memory_budget_bytes: self.memory_budget_bytes,
            recognized_nonexecuting_state_keys: self
                .diagnostic
                .recognized_nonexecuting_state_keys
                .to_vec(),
            missing_nonexecuting_state_keys: self
                .diagnostic
                .missing_nonexecuting_state_keys
                .to_vec(),
            stream: self.stream,
        };
        Self::from_normalized(normalized, cancellation)
    }

    pub fn encode(
        &self,
        backend: &CpuBackend,
        audio: &NativeAudioPayload,
        context: &ExecutionContext<'_>,
    ) -> Result<AudioEncoderOutput, AudioEncoderError> {
        self.validate(context.cancellation)?;
        // NativeAudioPayload is sealed and owns a copy-on-write Tensor, so its checked
        // construction invariant cannot drift while this borrowed invocation is running.
        if context.stream != self.stream {
            return Err(AudioEncoderError::InvalidInput(
                "execution stream does not match the retained encoder state".to_owned(),
            ));
        }
        let waveform = audio.waveform();
        let sample_rate = audio.sample_rate();
        validate_audio_input(backend, waveform, sample_rate, context)?;
        let required = self
            .resident_bytes()?
            .checked_add(self.invocation_memory_upper_bound(waveform, sample_rate)?)
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        if required > self.memory_budget_bytes {
            return Err(AudioEncoderError::OutOfMemory {
                required,
                budget: self.memory_budget_bytes,
            });
        }
        context.cancellation.check()?;
        match self.architecture {
            NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large => {
                execute_wav2vec2(self, backend, waveform, sample_rate, context)
            }
            NativeAudioEncoderArchitecture::WhisperLargeV3 => {
                execute_whisper(self, backend, waveform, sample_rate, context)
            }
        }
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, AudioEncoderError> {
        let resource = mem::size_of::<Self>()
            .checked_add(self.artifact_sha256.capacity())
            .and_then(|bytes| bytes.checked_add(self.semantic_state_digest_sha256.capacity()))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let map = conservative_tensor_map_owned_bytes(&self.executable_state)?;
        let mut diagnostic_keys = self
            .diagnostic
            .recognized_nonexecuting_state_keys
            .iter()
            .chain(self.diagnostic.missing_nonexecuting_state_keys.iter())
            .chain(self.diagnostic.unexpected_state_keys.iter());
        let diagnostic_count = diagnostic_keys.clone().count();
        let diagnostics = mem::size_of::<String>()
            .checked_mul(diagnostic_count)
            .and_then(|bytes| {
                diagnostic_keys.try_fold(bytes, |total, key| total.checked_add(key.capacity()))
            })
            .and_then(|bytes| bytes.checked_add(mem::size_of::<usize>() * 6))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        u64::try_from(resource)
            .ok()
            .and_then(|bytes| bytes.checked_add(map))
            .and_then(|bytes| bytes.checked_add(u64::try_from(diagnostics).ok()?))
            .ok_or(AudioEncoderError::MemoryOverflow)
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, AudioEncoderError> {
        let mut allocations = BTreeMap::new();
        for tensor in self.executable_state.values() {
            let storage_id = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some(existing) = allocations.insert(storage_id.get(), (storage_id, bytes))
                && existing.1 != bytes
            {
                return Err(AudioEncoderError::SemanticStateChanged);
            }
        }
        Ok(allocations.into_values().collect())
    }

    pub fn resident_bytes(&self) -> Result<u64, AudioEncoderError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(AudioEncoderError::MemoryOverflow)
            },
        )
    }

    fn project_semantic_state_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, AudioEncoderError> {
        cancellation.check()?;
        let mut digest = Sha256::new();
        digest.update(b"zed.sim.audio-encoder-resource.v1\0");
        hash_field(&mut digest, self.architecture.identifier().as_bytes())?;
        hash_field(&mut digest, self.artifact_sha256.as_bytes())?;
        for dimension in [
            self.execution_profile.convolution_width,
            self.execution_profile.hidden_width,
            self.execution_profile.attention_heads,
            self.execution_profile.transformer_layers,
            self.execution_profile.feed_forward_width,
            self.execution_profile.positional_kernel,
            self.execution_profile.positional_groups,
            self.execution_profile.mel_bins,
            self.execution_profile.audio_context,
            self.execution_profile.audio_samples,
            self.execution_profile.fft_length,
            self.execution_profile.hop_length,
        ] {
            hash_usize(&mut digest, dimension)?;
        }
        hash_usize(&mut digest, self.executable_state.len())?;
        for (index, (key, tensor)) in self.executable_state.iter().enumerate() {
            if index.is_multiple_of(16) {
                cancellation.check()?;
            }
            hash_field(&mut digest, key.as_bytes())?;
            hash_tensor(&mut digest, tensor, cancellation)?;
        }
        hash_usize(
            &mut digest,
            self.diagnostic.recognized_nonexecuting_state_keys.len(),
        )?;
        for key in self.diagnostic.recognized_nonexecuting_state_keys.iter() {
            hash_field(&mut digest, key.as_bytes())?;
        }
        hash_usize(
            &mut digest,
            self.diagnostic.missing_nonexecuting_state_keys.len(),
        )?;
        for key in self.diagnostic.missing_nonexecuting_state_keys.iter() {
            hash_field(&mut digest, key.as_bytes())?;
        }
        hash_usize(&mut digest, self.diagnostic.unexpected_state_keys.len())?;
        for key in self.diagnostic.unexpected_state_keys.iter() {
            hash_field(&mut digest, key.as_bytes())?;
        }
        cancellation.check()?;
        Ok(format!("{:x}", digest.finalize()))
    }

    fn invocation_memory_upper_bound(
        &self,
        waveform: &Tensor,
        sample_rate: u32,
    ) -> Result<u64, AudioEncoderError> {
        let [batch, channels, samples] = waveform.descriptor().shape() else {
            return Err(AudioEncoderError::InvalidInput(
                "canonical AUDIO waveform rank changed".to_owned(),
            ));
        };
        let batch = usize::try_from(*batch).map_err(|_| AudioEncoderError::MemoryOverflow)?;
        let channels = usize::try_from(*channels).map_err(|_| AudioEncoderError::MemoryOverflow)?;
        let samples = usize::try_from(*samples).map_err(|_| AudioEncoderError::MemoryOverflow)?;
        let resampled = samples
            .checked_mul(MODEL_SAMPLE_RATE as usize)
            .and_then(|value| value.checked_add(sample_rate as usize - 1))
            .and_then(|value| value.checked_div(sample_rate as usize))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let original_input_values = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(samples))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let resampled_values = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(resampled))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let (tokens, width, heads, multiplier, preprocessing_values) = match self.architecture {
            NativeAudioEncoderArchitecture::Wav2Vec2Base
            | NativeAudioEncoderArchitecture::Wav2Vec2Large => {
                let mut tokens = resampled;
                for (kernel, stride) in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)] {
                    tokens = tokens
                        .checked_sub(kernel)
                        .map(|value| value / stride + 1)
                        .ok_or_else(|| {
                            AudioEncoderError::InvalidInput(
                                "waveform is too short for Wav2Vec2 convolutions".to_owned(),
                            )
                        })?;
                }
                let first_convolution_length = resampled
                    .checked_sub(10)
                    .map(|value| value / 5 + 1)
                    .ok_or_else(|| {
                        AudioEncoderError::InvalidInput(
                            "waveform is too short for Wav2Vec2 convolutions".to_owned(),
                        )
                    })?;
                let preprocessing_values = batch
                    .checked_mul(self.execution_profile.convolution_width)
                    .and_then(|value| value.checked_mul(first_convolution_length))
                    .and_then(|value| value.checked_mul(2))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                (
                    tokens,
                    self.execution_profile.hidden_width,
                    self.execution_profile.attention_heads,
                    self.execution_profile.output_layers() + 12,
                    preprocessing_values,
                )
            }
            NativeAudioEncoderArchitecture::WhisperLargeV3 => {
                let mel_frames = self
                    .execution_profile
                    .audio_samples
                    .checked_div(self.execution_profile.hop_length)
                    .and_then(|frames| frames.checked_add(1))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let padded_values = batch
                    .checked_mul(self.execution_profile.audio_samples)
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let mel_values = batch
                    .checked_mul(self.execution_profile.mel_bins)
                    .and_then(|value| value.checked_mul(mel_frames))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let encoded_values = batch
                    .checked_mul(self.execution_profile.hidden_width)
                    .and_then(|value| value.checked_mul(self.execution_profile.audio_context))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let stft_values = batch
                    .checked_mul(self.execution_profile.fft_length / 2 + 1)
                    .and_then(|value| value.checked_mul(mel_frames))
                    .and_then(|value| value.checked_mul(4))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let log_mel_values = mel_values
                    .checked_mul(5)
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                let preprocessing_values = padded_values
                    .checked_add(mel_values)
                    .and_then(|value| value.checked_add(encoded_values))
                    .and_then(|value| value.checked_add(stft_values.max(log_mel_values)))
                    .ok_or(AudioEncoderError::MemoryOverflow)?;
                (
                    self.execution_profile.audio_context,
                    self.execution_profile.hidden_width,
                    self.execution_profile.attention_heads,
                    self.execution_profile.output_layers() + 12,
                    preprocessing_values,
                )
            }
        };
        let activation_values = batch
            .checked_mul(tokens)
            .and_then(|value| value.checked_mul(width))
            .and_then(|value| value.checked_mul(multiplier))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let attention_values = batch
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(2))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        original_input_values
            .checked_add(
                resampled_values
                    .checked_mul(2)
                    .ok_or(AudioEncoderError::MemoryOverflow)?,
            )
            .and_then(|value| value.checked_add(preprocessing_values))
            .and_then(|value| value.checked_add(activation_values))
            .and_then(|value| value.checked_add(attention_values))
            .and_then(|value| value.checked_mul(mem::size_of::<f32>()))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(AudioEncoderError::MemoryOverflow)
    }

    fn state(&self, key: &str) -> Result<Tensor, AudioEncoderError> {
        self.executable_state
            .get(key)
            .cloned()
            .ok_or_else(|| AudioEncoderError::MissingState(key.to_owned()))
    }
}

#[cfg(any(test, feature = "test-support"))]
fn build_deterministic_reduced_audio_encoder_fixture(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    architecture: NativeAudioEncoderArchitecture,
    mutation: u32,
) -> Result<NativeAudioEncoder, AudioEncoderError> {
    build_deterministic_audio_encoder_fixture_with_profile(
        backend,
        context,
        architecture,
        AudioEncoderExecutionProfile::reduced(architecture),
        mutation,
    )
}

#[cfg(any(test, feature = "test-support"))]
fn build_deterministic_audio_encoder_fixture_with_profile(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    architecture: NativeAudioEncoderArchitecture,
    execution_profile: AudioEncoderExecutionProfile,
    mutation: u32,
) -> Result<NativeAudioEncoder, AudioEncoderError> {
    context.cancellation.check()?;
    let expected = expected_state_shapes(architecture, &execution_profile)?;
    let mut executable_state = BTreeMap::new();
    for (state_index, (key, shape)) in expected.into_iter().enumerate() {
        if state_index.is_multiple_of(16) {
            context.cancellation.check()?;
        }
        let elements = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let elements = usize::try_from(elements).map_err(|_| AudioEncoderError::MemoryOverflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| AudioEncoderError::Allocation("reduced fixture state"))?;
        let normalization_weight = key.ends_with(".weight") && key.contains("norm");
        let position_scale = key.ends_with("parametrizations.weight.original0");
        let key_seed = key.bytes().fold(0_u32, |seed, byte| {
            seed.wrapping_mul(16_777_619).wrapping_add(u32::from(byte))
        });
        for element in 0..elements {
            if element.is_multiple_of(4_096) {
                context.cancellation.check()?;
            }
            let value = if normalization_weight || position_scale {
                1.0 + f32::from((key_seed.wrapping_add(element as u32) % 5) as u8) * 0.01
            } else if key.ends_with(".bias") {
                (f32::from((key_seed.wrapping_add(element as u32) % 7) as u8) - 3.0) * 0.002
            } else {
                (f32::from((key_seed.wrapping_add((element as u32).wrapping_mul(17)) % 13) as u8)
                    - 6.0)
                    * 0.01
            };
            values.push(value);
        }
        let mutation_target = match mutation {
            1 => key == "encoder.layer_norm.bias",
            2 => {
                key == "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
                    || key == "encoder.conv1.weight"
            }
            3 => {
                key.ends_with("encoder.layers.0.attention.q_proj.weight")
                    || key.ends_with("encoder.layers.0.self_attn.q_proj.weight")
            }
            4 => {
                key.ends_with("encoder.layers.0.feed_forward.intermediate_dense.weight")
                    || key.ends_with("encoder.layers.0.fc1.weight")
            }
            _ => false,
        };
        if mutation_target {
            let value = values
                .first_mut()
                .ok_or(AudioEncoderError::InvalidCheckpoint(
                    "reduced fixture mutation target is empty".to_owned(),
                ))?;
            *value += if mutation == 1 {
                0.125
            } else {
                mutation as f32 * 5.0
            };
        }
        let tensor = tensor_from_f32_with_context_exact_native(
            backend,
            &shape,
            &values,
            DType::F32,
            backend.device(),
            context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        executable_state.insert(key, tensor);
    }
    let artifact_sha256 = format!(
        "{:x}",
        Sha256::digest(format!("{}:{mutation}", architecture.identifier()).as_bytes())
    );
    NativeAudioEncoder::from_normalized(
        NormalizedAudioEncoderCheckpoint {
            architecture,
            execution_profile,
            artifact_sha256,
            executable_state,
            recognized_nonexecuting_state_keys: Vec::new(),
            missing_nonexecuting_state_keys: if matches!(
                architecture,
                NativeAudioEncoderArchitecture::Wav2Vec2Base
                    | NativeAudioEncoderArchitecture::Wav2Vec2Large
            ) {
                vec!["masked_spec_embed".to_owned()]
            } else {
                Vec::new()
            },
            unexpected_state_keys: Vec::new(),
            memory_budget_bytes: 128 * 1024 * 1024,
            stream: context.stream,
        },
        context.cancellation,
    )
}

#[cfg(any(test, feature = "test-support"))]
pub fn deterministic_reduced_audio_encoder_fixture(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    architecture: NativeAudioEncoderArchitecture,
    mutation: u32,
) -> Result<NativeAudioEncoder, AudioEncoderError> {
    build_deterministic_reduced_audio_encoder_fixture(backend, context, architecture, mutation)
}

fn validate_audio_input(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<(), AudioEncoderError> {
    context.cancellation.check()?;
    let [batch, channels, samples] = waveform.descriptor().shape() else {
        return Err(AudioEncoderError::InvalidInput(
            "audio encoder requires a rank-3 AUDIO waveform".to_owned(),
        ));
    };
    if *batch == 0 || *channels == 0 || *samples == 0 || !(8_000..=384_000).contains(&sample_rate) {
        return Err(AudioEncoderError::InvalidInput(
            "audio encoder requires nonempty AUDIO at 8000..=384000 Hz before resampling"
                .to_owned(),
        ));
    }
    if backend.device() != DeviceId::CPU
        || waveform.descriptor().dtype() != DType::F32
        || waveform.descriptor().device() != DeviceId::CPU
        || waveform.descriptor().stream() != context.stream
        || !waveform.descriptor().is_contiguous()?
    {
        return Err(AudioEncoderError::InvalidInput(
            "waveform must be contiguous CPU F32 on the execution stream".to_owned(),
        ));
    }
    context.cancellation.check()?;
    Ok(())
}

fn validate_state_tensor(
    key: &str,
    tensor: &Tensor,
    expected_shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<(), AudioEncoderError> {
    if tensor.descriptor().shape() != expected_shape {
        return Err(AudioEncoderError::StateShape {
            key: key.to_owned(),
            expected: expected_shape.to_vec(),
            actual: tensor.descriptor().shape().to_vec(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32
        || tensor.descriptor().device() != DeviceId::CPU
        || tensor.descriptor().stream() != context.stream
        || !tensor.descriptor().is_contiguous()?
    {
        return Err(AudioEncoderError::StatePlacement {
            key: key.to_owned(),
            expected_stream: context.stream,
        });
    }
    Ok(())
}

fn expected_state_shapes(
    architecture: NativeAudioEncoderArchitecture,
    profile: &AudioEncoderExecutionProfile,
) -> Result<BTreeMap<String, Vec<u64>>, AudioEncoderError> {
    match architecture {
        NativeAudioEncoderArchitecture::Wav2Vec2Base
        | NativeAudioEncoderArchitecture::Wav2Vec2Large => {
            wav2vec2_state_shapes(architecture, profile)
        }
        NativeAudioEncoderArchitecture::WhisperLargeV3 => whisper_state_shapes(profile),
    }
}

fn wav2vec2_state_shapes(
    architecture: NativeAudioEncoderArchitecture,
    profile: &AudioEncoderExecutionProfile,
) -> Result<BTreeMap<String, Vec<u64>>, AudioEncoderError> {
    let width = profile.hidden_width;
    let layers = profile.transformer_layers;
    let convolution_width = profile.convolution_width;
    let large = architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large;
    let mut shapes = BTreeMap::new();
    let convolution_geometry = [
        (1, 10),
        (convolution_width, 3),
        (convolution_width, 3),
        (convolution_width, 3),
        (convolution_width, 3),
        (convolution_width, 2),
        (convolution_width, 2),
    ];
    for (index, (input_channels, kernel)) in convolution_geometry.into_iter().enumerate() {
        insert_shape(
            &mut shapes,
            format!("feature_extractor.conv_layers.{index}.conv.weight"),
            [convolution_width, input_channels, kernel],
        )?;
        if large || index == 0 {
            insert_shape(
                &mut shapes,
                format!("feature_extractor.conv_layers.{index}.layer_norm.weight"),
                [convolution_width],
            )?;
            insert_shape(
                &mut shapes,
                format!("feature_extractor.conv_layers.{index}.layer_norm.bias"),
                [convolution_width],
            )?;
        }
        if large {
            insert_shape(
                &mut shapes,
                format!("feature_extractor.conv_layers.{index}.conv.bias"),
                [convolution_width],
            )?;
        }
    }
    insert_shape(
        &mut shapes,
        "feature_projection.layer_norm.weight",
        [convolution_width],
    )?;
    insert_shape(
        &mut shapes,
        "feature_projection.layer_norm.bias",
        [convolution_width],
    )?;
    insert_shape(
        &mut shapes,
        "feature_projection.projection.weight",
        [width, convolution_width],
    )?;
    insert_shape(&mut shapes, "feature_projection.projection.bias", [width])?;
    insert_shape(
        &mut shapes,
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        [1, 1, profile.positional_kernel],
    )?;
    insert_shape(
        &mut shapes,
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        [
            width,
            width / profile.positional_groups,
            profile.positional_kernel,
        ],
    )?;
    insert_shape(&mut shapes, "encoder.pos_conv_embed.conv.bias", [width])?;
    for layer in 0..layers {
        let prefix = format!("encoder.layers.{layer}");
        for projection in ["k_proj", "v_proj", "q_proj", "out_proj"] {
            insert_shape(
                &mut shapes,
                format!("{prefix}.attention.{projection}.weight"),
                [width, width],
            )?;
            insert_shape(
                &mut shapes,
                format!("{prefix}.attention.{projection}.bias"),
                [width],
            )?;
        }
        for normalization in ["layer_norm", "final_layer_norm"] {
            insert_shape(
                &mut shapes,
                format!("{prefix}.{normalization}.weight"),
                [width],
            )?;
            insert_shape(
                &mut shapes,
                format!("{prefix}.{normalization}.bias"),
                [width],
            )?;
        }
        insert_shape(
            &mut shapes,
            format!("{prefix}.feed_forward.intermediate_dense.weight"),
            [profile.feed_forward_width, width],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.feed_forward.intermediate_dense.bias"),
            [profile.feed_forward_width],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.feed_forward.output_dense.weight"),
            [width, profile.feed_forward_width],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.feed_forward.output_dense.bias"),
            [width],
        )?;
    }
    insert_shape(&mut shapes, "encoder.layer_norm.weight", [width])?;
    insert_shape(&mut shapes, "encoder.layer_norm.bias", [width])?;
    Ok(shapes)
}

fn whisper_state_shapes(
    profile: &AudioEncoderExecutionProfile,
) -> Result<BTreeMap<String, Vec<u64>>, AudioEncoderError> {
    let mut shapes = BTreeMap::new();
    let width = profile.hidden_width;
    insert_shape(
        &mut shapes,
        "encoder.conv1.weight",
        [width, profile.mel_bins, 3],
    )?;
    insert_shape(&mut shapes, "encoder.conv1.bias", [width])?;
    insert_shape(&mut shapes, "encoder.conv2.weight", [width, width, 3])?;
    insert_shape(&mut shapes, "encoder.conv2.bias", [width])?;
    insert_shape(
        &mut shapes,
        "encoder.embed_positions.weight",
        [profile.audio_context, width],
    )?;
    for layer in 0..profile.transformer_layers {
        let prefix = format!("encoder.layers.{layer}");
        for projection in ["q_proj", "v_proj", "out_proj"] {
            insert_shape(
                &mut shapes,
                format!("{prefix}.self_attn.{projection}.weight"),
                [width, width],
            )?;
            insert_shape(
                &mut shapes,
                format!("{prefix}.self_attn.{projection}.bias"),
                [width],
            )?;
        }
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.k_proj.weight"),
            [width, width],
        )?;
        for normalization in ["self_attn_layer_norm", "final_layer_norm"] {
            insert_shape(
                &mut shapes,
                format!("{prefix}.{normalization}.weight"),
                [width],
            )?;
            insert_shape(
                &mut shapes,
                format!("{prefix}.{normalization}.bias"),
                [width],
            )?;
        }
        insert_shape(
            &mut shapes,
            format!("{prefix}.fc1.weight"),
            [profile.feed_forward_width, width],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.fc1.bias"),
            [profile.feed_forward_width],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.fc2.weight"),
            [width, profile.feed_forward_width],
        )?;
        insert_shape(&mut shapes, format!("{prefix}.fc2.bias"), [width])?;
    }
    insert_shape(&mut shapes, "encoder.layer_norm.weight", [width])?;
    insert_shape(&mut shapes, "encoder.layer_norm.bias", [width])?;
    Ok(shapes)
}

fn insert_shape<const N: usize>(
    shapes: &mut BTreeMap<String, Vec<u64>>,
    key: impl Into<String>,
    shape: [usize; N],
) -> Result<(), AudioEncoderError> {
    let shape = shape
        .into_iter()
        .map(|dimension| u64::try_from(dimension).map_err(|_| AudioEncoderError::MemoryOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    if shapes.insert(key.into(), shape).is_some() {
        return Err(AudioEncoderError::InvalidCheckpoint(
            "duplicate expected audio encoder state key".to_owned(),
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), AudioEncoderError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AudioEncoderError::InvalidCheckpoint(
            "artifact SHA-256 must be lowercase hexadecimal".to_owned(),
        ));
    }
    Ok(())
}

fn validate_state_key(value: &str) -> Result<(), AudioEncoderError> {
    if value.is_empty()
        || value.len() > MAX_AUDIO_STATE_KEY_BYTES
        || value.contains('\0')
        || value.starts_with('.')
        || value.ends_with('.')
        || value.split('.').any(str::is_empty)
    {
        return Err(AudioEncoderError::InvalidCheckpoint(format!(
            "invalid state key {value:?}"
        )));
    }
    Ok(())
}

fn conservative_tensor_map_owned_bytes(
    tensors: &Arc<BTreeMap<String, Tensor>>,
) -> Result<u64, AudioEncoderError> {
    const BTREE_NODE_KEY_CAPACITY: usize = 11;
    const BTREE_NODE_EDGE_CAPACITY: usize = BTREE_NODE_KEY_CAPACITY + 1;
    const BTREE_NODE_HEADER_WORDS: usize = 3;
    let arc_and_map = (mem::size_of::<usize>() * 2)
        .checked_add(mem::size_of::<BTreeMap<String, Tensor>>())
        .ok_or(AudioEncoderError::MemoryOverflow)?;
    let node_bytes = (mem::size_of::<String>() + mem::size_of::<Tensor>())
        .checked_mul(BTREE_NODE_KEY_CAPACITY)
        .and_then(|bytes| bytes.checked_add(mem::size_of::<usize>() * BTREE_NODE_EDGE_CAPACITY))
        .and_then(|bytes| bytes.checked_add(mem::size_of::<usize>() * BTREE_NODE_HEADER_WORDS))
        .ok_or(AudioEncoderError::MemoryOverflow)?;
    let nodes = if tensors.is_empty() {
        0
    } else {
        node_bytes
            .checked_mul(tensors.len())
            .ok_or(AudioEncoderError::MemoryOverflow)?
    };
    let keys = tensors.keys().try_fold(0_usize, |total, key| {
        total
            .checked_add(key.capacity())
            .ok_or(AudioEncoderError::MemoryOverflow)
    })?;
    let bytes = arc_and_map
        .checked_add(nodes)
        .and_then(|bytes| bytes.checked_add(keys))
        .ok_or(AudioEncoderError::MemoryOverflow)?;
    u64::try_from(bytes).map_err(|_| AudioEncoderError::MemoryOverflow)
}

fn hash_usize(digest: &mut Sha256, value: usize) -> Result<(), AudioEncoderError> {
    let value = u64::try_from(value).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    digest.update(value.to_le_bytes());
    Ok(())
}

fn hash_field(digest: &mut Sha256, value: &[u8]) -> Result<(), AudioEncoderError> {
    hash_usize(digest, value.len())?;
    digest.update(value);
    Ok(())
}

fn hash_tensor(
    digest: &mut Sha256,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), AudioEncoderError> {
    cancellation.check()?;
    hash_usize(digest, tensor.descriptor().shape().len())?;
    for dimension in tensor.descriptor().shape() {
        digest.update(dimension.to_le_bytes());
    }
    digest.update([tensor.descriptor().dtype() as u8]);
    let bytes = tensor.contiguous_bytes()?;
    hash_usize(digest, bytes.len())?;
    for chunk in bytes.chunks(64 * 1024) {
        cancellation.check()?;
        digest.update(chunk);
    }
    Ok(())
}

fn execute_wav2vec2(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<AudioEncoderOutput, AudioEncoderError> {
    let resampled = resample_audio(backend, waveform, sample_rate, context)?;
    let audio_samples = *resampled.descriptor().shape().get(2).ok_or_else(|| {
        AudioEncoderError::InvalidInput("resampled waveform rank changed".to_owned())
    })?;
    let mut hidden = channel_mean(backend, &resampled, context)?;
    if resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large {
        hidden = normalize_wav2vec2_global(backend, &hidden, context)?;
    }
    hidden = add_channel_axis(backend, &hidden, context)?;

    let profile = &resource.execution_profile;
    for (index, (kernel, stride)) in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
        .into_iter()
        .enumerate()
    {
        context.cancellation.check()?;
        let input_channels = if index == 0 {
            1
        } else {
            profile.convolution_width
        };
        let prefix = format!("feature_extractor.conv_layers.{index}");
        hidden = convolution_1d(
            resource,
            backend,
            &hidden,
            &format!("{prefix}.conv"),
            input_channels,
            profile.convolution_width,
            kernel,
            stride,
            0,
            1,
            resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large,
            context,
        )?;
        if resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large {
            hidden = transpose_ncl_nlc(backend, &hidden, context)?;
            hidden = layer_norm(
                resource,
                backend,
                &hidden,
                &format!("{prefix}.layer_norm"),
                profile.convolution_width,
                1e-5,
                context,
            )?;
            hidden = transpose_ncl_nlc(backend, &hidden, context)?;
        } else if index == 0 {
            hidden = group_norm(
                resource,
                backend,
                &hidden,
                &format!("{prefix}.layer_norm"),
                profile.convolution_width,
                context,
            )?;
        }
        hidden = gelu(backend, &hidden, context)?;
    }

    hidden = transpose_ncl_nlc(backend, &hidden, context)?;
    hidden = layer_norm(
        resource,
        backend,
        &hidden,
        "feature_projection.layer_norm",
        profile.convolution_width,
        1e-5,
        context,
    )?;
    hidden = linear(
        resource,
        backend,
        &hidden,
        "feature_projection.projection",
        profile.convolution_width,
        profile.hidden_width,
        true,
        context,
    )?;

    let position_input = transpose_ncl_nlc(backend, &hidden, context)?;
    let mut position = weight_normalized_convolution_1d(
        resource,
        backend,
        &position_input,
        "encoder.pos_conv_embed.conv",
        profile.hidden_width,
        profile.positional_kernel,
        profile.positional_kernel / 2,
        profile.positional_groups,
        context,
    )?;
    position = crop_last_ncl(backend, &position, context)?;
    position = gelu(backend, &position, context)?;
    position = transpose_ncl_nlc(backend, &position, context)?;
    hidden = add_tensors(backend, &hidden, &position, context)?;

    if resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Base {
        hidden = layer_norm(
            resource,
            backend,
            &hidden,
            "encoder.layer_norm",
            profile.hidden_width,
            1e-5,
            context,
        )?;
    }
    let mut all_layers = Vec::new();
    all_layers
        .try_reserve_exact(profile.output_layers())
        .map_err(|_| AudioEncoderError::Allocation("Wav2Vec2 layer outputs"))?;
    for layer in 0..profile.transformer_layers {
        context.cancellation.check()?;
        all_layers.push(hidden.clone());
        hidden = wav2vec2_layer(resource, backend, hidden, layer, context)?;
    }
    if resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large {
        hidden = layer_norm(
            resource,
            backend,
            &hidden,
            "encoder.layer_norm",
            profile.hidden_width,
            1e-5,
            context,
        )?;
    }
    all_layers.push(hidden.clone());
    context.cancellation.check()?;
    AudioEncoderOutput::layered_with_cancellation(
        hidden,
        all_layers,
        audio_samples,
        context.cancellation,
    )
    .map_err(map_audio_encoder_output_error)
}

fn execute_whisper(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<AudioEncoderOutput, AudioEncoderError> {
    let resampled = resample_audio(backend, waveform, sample_rate, context)?;
    let audio_samples = *resampled.descriptor().shape().get(2).ok_or_else(|| {
        AudioEncoderError::InvalidInput("resampled waveform rank changed".to_owned())
    })?;
    let profile = &resource.execution_profile;
    let mono = channel_mean(backend, &resampled, context)?;
    let padded = trim_or_pad_audio(backend, &mono, profile.audio_samples, context)?;
    let mel = mel_spectrogram_with_context_exact_native(
        backend,
        &padded,
        NativeMelSpectrogramConfiguration {
            sample_rate: MODEL_SAMPLE_RATE,
            n_fft: profile.fft_length,
            win_length: None,
            hop_length: Some(profile.hop_length),
            f_min: 0.0,
            f_max: Some(8_000.0),
            n_mels: profile.mel_bins,
            power: 2.0,
            center: true,
            normalized: false,
            mel_scale: NativeMelScale::Slaney,
            mel_normalization: NativeMelNormalization::Slaney,
        },
        context,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let mel = crop_last_ncl(backend, &mel, context)?;
    let mut hidden = whisper_log_mel(backend, &mel, context)?;
    hidden = convolution_1d(
        resource,
        backend,
        &hidden,
        "encoder.conv1",
        profile.mel_bins,
        profile.hidden_width,
        3,
        1,
        1,
        1,
        true,
        context,
    )?;
    hidden = gelu(backend, &hidden, context)?;
    hidden = convolution_1d(
        resource,
        backend,
        &hidden,
        "encoder.conv2",
        profile.hidden_width,
        profile.hidden_width,
        3,
        2,
        1,
        1,
        true,
        context,
    )?;
    hidden = gelu(backend, &hidden, context)?;
    hidden = transpose_ncl_nlc(backend, &hidden, context)?;
    hidden = add_position_embedding(resource, backend, &hidden, context)?;

    let mut all_layers = Vec::new();
    all_layers
        .try_reserve_exact(profile.output_layers())
        .map_err(|_| AudioEncoderError::Allocation("Whisper layer outputs"))?;
    for layer in 0..profile.transformer_layers {
        context.cancellation.check()?;
        all_layers.push(hidden.clone());
        hidden = whisper_layer(resource, backend, hidden, layer, context)?;
    }
    hidden = layer_norm(
        resource,
        backend,
        &hidden,
        "encoder.layer_norm",
        profile.hidden_width,
        1e-5,
        context,
    )?;
    all_layers.push(hidden.clone());
    context.cancellation.check()?;
    AudioEncoderOutput::layered_with_cancellation(
        hidden,
        all_layers,
        audio_samples,
        context.cancellation,
    )
    .map_err(map_audio_encoder_output_error)
}

fn map_audio_encoder_output_error(error: NativeModelPayloadError) -> AudioEncoderError {
    match error {
        NativeModelPayloadError::Tensor(TensorError::Cancelled) => AudioEncoderError::Cancelled,
        NativeModelPayloadError::Tensor(error) => AudioEncoderError::Tensor(error),
        error => AudioEncoderError::Output(error.to_string()),
    }
}

fn resample_audio(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    resample_with_context_exact_native(
        backend,
        waveform,
        NativeResampleConfiguration::torchaudio_default(sample_rate, MODEL_SAMPLE_RATE),
        context,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn wav2vec2_layer(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    mut hidden: Tensor,
    layer: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let width = resource.execution_profile.hidden_width;
    let stable = resource.architecture == NativeAudioEncoderArchitecture::Wav2Vec2Large;
    let prefix = format!("encoder.layers.{layer}");
    let residual = hidden.clone();
    if stable {
        hidden = layer_norm(
            resource,
            backend,
            &hidden,
            &format!("{prefix}.layer_norm"),
            width,
            1e-5,
            context,
        )?;
    }
    hidden = self_attention(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.attention"),
        resource.execution_profile.attention_heads,
        true,
        context,
    )?;
    hidden = add_tensors(backend, &residual, &hidden, context)?;
    if !stable {
        hidden = layer_norm(
            resource,
            backend,
            &hidden,
            &format!("{prefix}.layer_norm"),
            width,
            1e-5,
            context,
        )?;
    }
    let feed_input = if stable {
        layer_norm(
            resource,
            backend,
            &hidden,
            &format!("{prefix}.final_layer_norm"),
            width,
            1e-5,
            context,
        )?
    } else {
        hidden.clone()
    };
    let feed = feed_forward(
        resource,
        backend,
        &feed_input,
        &format!("{prefix}.feed_forward.intermediate_dense"),
        &format!("{prefix}.feed_forward.output_dense"),
        context,
    )?;
    let hidden = add_tensors(backend, &hidden, &feed, context)?;
    if stable {
        Ok(hidden)
    } else {
        layer_norm(
            resource,
            backend,
            &hidden,
            &format!("{prefix}.final_layer_norm"),
            width,
            1e-5,
            context,
        )
    }
}

fn whisper_layer(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    hidden: Tensor,
    layer: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let width = resource.execution_profile.hidden_width;
    let prefix = format!("encoder.layers.{layer}");
    let normalized = layer_norm(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.self_attn_layer_norm"),
        width,
        1e-5,
        context,
    )?;
    let attention = self_attention(
        resource,
        backend,
        &normalized,
        &format!("{prefix}.self_attn"),
        resource.execution_profile.attention_heads,
        false,
        context,
    )?;
    let hidden = add_tensors(backend, &hidden, &attention, context)?;
    let normalized = layer_norm(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.final_layer_norm"),
        width,
        1e-5,
        context,
    )?;
    let feed = feed_forward(
        resource,
        backend,
        &normalized,
        &format!("{prefix}.fc1"),
        &format!("{prefix}.fc2"),
        context,
    )?;
    add_tensors(backend, &hidden, &feed, context)
}

fn self_attention(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    hidden: &Tensor,
    prefix: &str,
    heads: usize,
    all_biases: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let shape = hidden.descriptor().shape();
    let [batch, tokens, width] = shape else {
        return Err(AudioEncoderError::InvalidInput(
            "attention input must have [batch, tokens, width] shape".to_owned(),
        ));
    };
    let width_usize = usize::try_from(*width).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let batch_usize = usize::try_from(*batch).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let tokens_usize = usize::try_from(*tokens).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let query = linear(
        resource,
        backend,
        hidden,
        &format!("{prefix}.q_proj"),
        width_usize,
        width_usize,
        true,
        context,
    )?;
    let key = linear(
        resource,
        backend,
        hidden,
        &format!("{prefix}.k_proj"),
        width_usize,
        width_usize,
        all_biases,
        context,
    )?;
    let value = linear(
        resource,
        backend,
        hidden,
        &format!("{prefix}.v_proj"),
        width_usize,
        width_usize,
        true,
        context,
    )?;
    let query_values = tensor_values(backend, &query, context)?;
    let key_values = tensor_values(backend, &key, context)?;
    let value_values = tensor_values(backend, &value, context)?;
    let head_dimension = width_usize
        .checked_div(heads)
        .filter(|dimension| *dimension > 0 && width_usize.is_multiple_of(heads))
        .ok_or_else(|| {
            AudioEncoderError::InvalidCheckpoint("attention head width is invalid".to_owned())
        })?;
    let outcome = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: batch_usize,
            query_tokens: tokens_usize,
            key_tokens: tokens_usize,
            heads,
            head_dimension,
            value_dimension: head_dimension,
            scale: None,
            workspace_limit_bytes: tokens_usize
                .checked_mul(mem::size_of::<f32>())
                .ok_or(AudioEncoderError::MemoryOverflow)?,
        },
        &query_values,
        &key_values,
        &value_values,
        None,
        context,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let attention = tensor_from_values(backend, shape, &outcome.values, context)?;
    linear(
        resource,
        backend,
        &attention,
        &format!("{prefix}.out_proj"),
        width_usize,
        width_usize,
        true,
        context,
    )
}

fn feed_forward(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    first: &str,
    second: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let mut hidden = linear(
        resource,
        backend,
        input,
        first,
        resource.execution_profile.hidden_width,
        resource.execution_profile.feed_forward_width,
        true,
        context,
    )?;
    hidden = gelu(backend, &hidden, context)?;
    linear(
        resource,
        backend,
        &hidden,
        second,
        resource.execution_profile.feed_forward_width,
        resource.execution_profile.hidden_width,
        true,
        context,
    )
}

fn linear(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    input_features: usize,
    output_features: usize,
    bias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let mut module = NativeModule::linear(prefix, input_features, output_features, bias, false)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .load_dense_parameters(
            resource.state(&format!("{prefix}.weight"))?,
            bias.then(|| resource.state(&format!("{prefix}.bias")))
                .transpose()?,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn layer_norm(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    width: usize,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let mut module = NativeModule::layer_norm(prefix, vec![width], epsilon, true, true, false)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .load_dense_parameters(
            resource.state(&format!("{prefix}.weight"))?,
            Some(resource.state(&format!("{prefix}.bias"))?),
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn group_norm(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    channels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let mut module = NativeModule::group_norm(prefix, channels, channels, 1e-5, true, false)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .load_dense_parameters(
            resource.state(&format!("{prefix}.weight"))?,
            Some(resource.state(&format!("{prefix}.bias"))?),
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn convolution_1d(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let geometry = ConvolutionGeometry::new(
        1,
        vec![stride],
        vec![padding],
        vec![1],
        groups,
        false,
        vec![0],
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let mut module = NativeModule::convolution(
        prefix,
        input_channels,
        output_channels,
        vec![kernel],
        bias,
        geometry,
        false,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .load_dense_parameters(
            resource.state(&format!("{prefix}.weight"))?,
            bias.then(|| resource.state(&format!("{prefix}.bias")))
                .transpose()?,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn weight_normalized_convolution_1d(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    channels: usize,
    kernel: usize,
    padding: usize,
    groups: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let geometry =
        ConvolutionGeometry::new(1, vec![1], vec![padding], vec![1], groups, false, vec![0])
            .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let mut module = NativeModule::convolution(
        prefix,
        channels,
        channels,
        vec![kernel],
        true,
        geometry,
        false,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .load_weight_norm_parameters_with_context_exact_native(
            backend,
            resource.state(&format!("{prefix}.parametrizations.weight.original0"))?,
            resource.state(&format!("{prefix}.parametrizations.weight.original1"))?,
            Some(resource.state(&format!("{prefix}.bias"))?),
            Some(2),
            context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn gelu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let mut module = NativeModule::gelu("audio_encoder.gelu", GeluApproximation::None)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    module
        .forward_with_context(backend, input, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn tensor_values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AudioEncoderError> {
    tensor_to_f32_with_context_exact_native(backend, tensor, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn tensor_from_values(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn channel_mean(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    tensor_mean_with_context_exact_native(backend, input, Some(&[1]), false, None, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn normalize_wav2vec2_global(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let element_count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| AudioEncoderError::MemoryOverflow)?;
    if element_count < 2 {
        return Err(AudioEncoderError::InvalidInput(
            "Wav2Vec2 normalization requires at least two samples".to_owned(),
        ));
    }
    let mean = tensor_mean_with_context_exact_native(backend, input, None, false, None, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let variance = tensor_var_with_context_exact_native(backend, input, None, 1, false, context)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
    let mean = *tensor_values(backend, &mean, context)?
        .first()
        .ok_or_else(|| AudioEncoderError::TensorOperation("mean result is empty".to_owned()))?;
    let variance = *tensor_values(backend, &variance, context)?
        .first()
        .ok_or_else(|| AudioEncoderError::TensorOperation("variance result is empty".to_owned()))?;
    let denominator = (variance + 1e-7).sqrt();
    let values = tensor_values(backend, input, context)?;
    let mut output = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(256) {
            context.cancellation.check()?;
        }
        output.try_push((*value - mean) / denominator)?;
    }
    tensor_from_values(backend, input.descriptor().shape(), &output, context)
}

fn add_channel_axis(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let [batch, samples] = input.descriptor().shape() else {
        return Err(AudioEncoderError::InvalidInput(
            "audio channel insertion requires rank two input".to_owned(),
        ));
    };
    tensor_from_values(
        backend,
        &[*batch, 1, *samples],
        &tensor_values(backend, input, context)?,
        context,
    )
}

fn transpose_ncl_nlc(
    _backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    tensor_transpose_exact_native(input, 1, 2, context.cancellation)
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn add_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(AudioEncoderError::InvalidInput(
            "audio residual shapes differ".to_owned(),
        ));
    }
    let left_values = tensor_values(backend, left, context)?;
    let right_values = tensor_values(backend, right, context)?;
    let mut output = backend.workspace_vec(context, left_values.len())?;
    for (index, (left, right)) in left_values.iter().zip(&right_values).enumerate() {
        if index.is_multiple_of(256) {
            context.cancellation.check()?;
        }
        output.try_push(*left + *right)?;
    }
    tensor_from_values(backend, left.descriptor().shape(), &output, context)
}

fn crop_last_ncl(
    _backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let [_batch, _channels, samples] = input.descriptor().shape() else {
        return Err(AudioEncoderError::InvalidInput(
            "audio frame crop requires rank three input".to_owned(),
        ));
    };
    let samples = usize::try_from(*samples).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let retained = samples.checked_sub(1).ok_or_else(|| {
        AudioEncoderError::InvalidInput("audio frame crop received no samples".to_owned())
    })?;
    narrow_function_exact_native(
        input,
        2,
        0,
        u64::try_from(retained).map_err(|_| AudioEncoderError::MemoryOverflow)?,
        context.cancellation,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn trim_or_pad_audio(
    backend: &CpuBackend,
    input: &Tensor,
    target: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let [_batch, samples] = input.descriptor().shape() else {
        return Err(AudioEncoderError::InvalidInput(
            "Whisper trim-or-pad requires rank two input".to_owned(),
        ));
    };
    let samples = usize::try_from(*samples).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    if samples > target {
        return narrow_function_exact_native(
            input,
            1,
            0,
            u64::try_from(target).map_err(|_| AudioEncoderError::MemoryOverflow)?,
            context.cancellation,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()));
    }
    if samples == target {
        return Ok(input.clone());
    }
    let padding = i64::try_from(target - samples).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    functional_pad_with_context_exact_native(
        backend,
        input,
        &[0, padding],
        FunctionalPadMode::Constant,
        None,
        context,
    )
    .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))
}

fn whisper_log_mel(
    backend: &CpuBackend,
    mel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let values = tensor_values(backend, mel, context)?;
    let mut logs = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(256) {
            context.cancellation.check()?;
        }
        let value = value.max(1e-10).log10();
        logs.try_push(value)?;
    }
    let logs_tensor = tensor_from_values(backend, mel.descriptor().shape(), &logs, context)?;
    let maximum_tensor =
        tensor_max_with_context_exact_native(backend, &logs_tensor, None, false, context)
            .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?
            .values;
    let maximum = *tensor_values(backend, &maximum_tensor, context)?
        .first()
        .ok_or_else(|| AudioEncoderError::TensorOperation("mel maximum is empty".to_owned()))?;
    let floor = maximum - 8.0;
    for (index, value) in logs.iter_mut().enumerate() {
        if index.is_multiple_of(256) {
            context.cancellation.check()?;
        }
        *value = (value.max(floor) + 4.0) / 4.0;
    }
    tensor_from_values(backend, mel.descriptor().shape(), &logs, context)
}

fn add_position_embedding(
    resource: &NativeAudioEncoder,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AudioEncoderError> {
    let [batch, tokens, width] = input.descriptor().shape() else {
        return Err(AudioEncoderError::InvalidInput(
            "Whisper position input must have rank three".to_owned(),
        ));
    };
    let batch = usize::try_from(*batch).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let tokens = usize::try_from(*tokens).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    let width = usize::try_from(*width).map_err(|_| AudioEncoderError::MemoryOverflow)?;
    if tokens > resource.execution_profile.audio_context {
        return Err(AudioEncoderError::InvalidInput(
            "Whisper encoded token count exceeds position state".to_owned(),
        ));
    }
    let input_values = tensor_values(backend, input, context)?;
    let position_values = tensor_values(
        backend,
        &resource.state("encoder.embed_positions.weight")?,
        context,
    )?;
    let mut output = backend.workspace_vec(context, input_values.len())?;
    for batch_index in 0..batch {
        for token in 0..tokens {
            for channel in 0..width {
                if channel.is_multiple_of(256) {
                    context.cancellation.check()?;
                }
                output.try_push(
                    input_values[(batch_index * tokens + token) * width + channel]
                        + position_values[token * width + channel],
                )?;
            }
        }
    }
    tensor_from_values(backend, input.descriptor().shape(), &output, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sim_tensor::{CpuWorkspaceAuthority, TensorBackend};
    use std::error::Error;

    const TEST_MEMORY_LIMIT: u64 = 256 * 1024 * 1024;
    const ORACLE: &str = include_str!(
        "../../sim_test_support/fixtures/models/audio-encoder-resource-foundation/oracle.json"
    );

    fn test_backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn Error>> {
        Ok(CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT)?)
    }

    fn test_context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
        stream: StreamId,
    ) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
        Ok(ExecutionContext {
            stream,
            scratch: authority.authorize_workspace(TEST_MEMORY_LIMIT)?,
            rng_phase: None,
            cancellation,
        })
    }

    fn test_audio(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        samples: usize,
        sample_rate: u32,
    ) -> Result<NativeAudioPayload, Box<dyn Error>> {
        let count = 2_usize
            .checked_mul(2)
            .and_then(|value| value.checked_mul(samples))
            .ok_or("audio fixture size overflowed")?;
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        for index in 0..count {
            let batch = index / (2 * samples);
            let channel = (index / samples) % 2;
            let sample = index % samples;
            values.push(
                ((sample as f32 * 0.013 + channel as f32 * 0.31).sin()
                    * (1.0 + batch as f32 * 1.7))
                    + batch as f32 * 0.2,
            );
        }
        let waveform = tensor_from_f32_with_context_exact_native(
            backend,
            &[2, 2, u64::try_from(samples)?],
            &values,
            DType::F32,
            backend.device(),
            context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        Ok(NativeAudioPayload::checked(waveform, sample_rate)?)
    }

    fn expected_oracle(
        architecture: NativeAudioEncoderArchitecture,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let document: Value = serde_json::from_str(ORACLE)?;
        let values = document
            .get("results")
            .and_then(|results| results.get(architecture.identifier()))
            .and_then(|result| result.get("values"))
            .and_then(Value::as_array)
            .ok_or("oracle result is missing")?;
        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or("oracle value is invalid".into())
            })
            .collect()
    }

    fn execute_reduced(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        architecture: NativeAudioEncoderArchitecture,
        mutation: u32,
    ) -> Result<(NativeAudioEncoder, AudioEncoderOutput, Vec<f32>), Box<dyn Error>> {
        let encoder =
            deterministic_reduced_audio_encoder_fixture(backend, context, architecture, mutation)?;
        let samples = if architecture == NativeAudioEncoderArchitecture::WhisperLargeV3 {
            29
        } else {
            1_600
        };
        let audio = test_audio(backend, context, samples, MODEL_SAMPLE_RATE)?;
        let output = encoder.encode(backend, &audio, context)?;
        let encoded = output
            .encoded_audio()
            .ok_or("layered audio output is missing")?;
        let values = tensor_to_f32_with_context_exact_native(backend, encoded, context)
            .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        Ok((encoder, output, values))
    }

    #[test]
    fn audio_encoder_reduced_oracles_are_exact_and_transactional() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = test_backend()?;
        let cancellation = CancellationToken::default();
        let context = test_context(&authority, &cancellation, StreamId::DEFAULT)?;
        for architecture in [
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            NativeAudioEncoderArchitecture::Wav2Vec2Large,
            NativeAudioEncoderArchitecture::WhisperLargeV3,
        ] {
            let (_, output, actual) = execute_reduced(&backend, &context, architecture, 0)?;
            let expected = expected_oracle(architecture)?;
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = 0.0002_f32 + 0.0002_f32 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{} oracle[{index}] expected {expected}, got {actual}",
                    architecture.identifier(),
                );
            }
            assert_eq!(
                output
                    .encoded_audio_all_layers()
                    .ok_or("layer outputs are missing")?
                    .len(),
                architecture.transformer_layers() + 1,
            );
            assert_eq!(
                output.audio_samples(),
                Some(
                    if architecture == NativeAudioEncoderArchitecture::WhisperLargeV3 {
                        29
                    } else {
                        1_600
                    }
                ),
            );
        }

        let (_, baseline_output, baseline) = execute_reduced(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            0,
        )?;
        for mutation in [2, 3, 4] {
            let (_, changed_output, changed) = execute_reduced(
                &backend,
                &context,
                NativeAudioEncoderArchitecture::Wav2Vec2Base,
                mutation,
            )?;
            let mut layer_changed = false;
            for (baseline_layer, changed_layer) in baseline_output
                .encoded_audio_all_layers()
                .ok_or("baseline layer outputs are missing")?
                .iter()
                .zip(
                    changed_output
                        .encoded_audio_all_layers()
                        .ok_or("changed layer outputs are missing")?,
                )
            {
                let baseline_layer =
                    tensor_to_f32_with_context_exact_native(&backend, baseline_layer, &context)?;
                let changed_layer =
                    tensor_to_f32_with_context_exact_native(&backend, changed_layer, &context)?;
                layer_changed |= baseline_layer != changed_layer;
            }
            assert!(
                baseline != changed || layer_changed,
                "mutation {mutation} did not affect execution",
            );
        }

        let encoder = deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            0,
        )?;
        let resampled_audio = test_audio(&backend, &context, 1_600, 12_000)?;
        let resampled_output = encoder.encode(&backend, &resampled_audio, &context)?;
        assert_eq!(resampled_output.audio_samples(), Some(2_134));

        let whisper = deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::WhisperLargeV3,
            0,
        )?;
        let long_audio = test_audio(&backend, &context, 480_001, MODEL_SAMPLE_RATE)?;
        let long_output = whisper.encode(&backend, &long_audio, &context)?;
        assert_eq!(long_output.audio_samples(), Some(480_001));
        Ok(())
    }

    #[test]
    fn audio_encoder_identity_residency_reconstruction_and_failures_are_atomic()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = test_backend()?;
        let cancellation = CancellationToken::default();
        let context = test_context(&authority, &cancellation, StreamId::DEFAULT)?;
        let encoder = deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            0,
        )?;
        let changed = deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            1,
        )?;
        assert_ne!(
            encoder.semantic_state_digest_sha256(),
            changed.semantic_state_digest_sha256()
        );
        assert_eq!(encoder.resident_bytes()?, changed.resident_bytes()?);
        let cloned = encoder.clone();
        assert_eq!(
            encoder.resident_tensor_allocations()?,
            cloned.resident_tensor_allocations()?
        );
        let reconstructed = encoder.reconstruct(&cancellation)?;
        assert_eq!(
            encoder.semantic_state_digest_sha256(),
            reconstructed.semantic_state_digest_sha256()
        );
        assert_eq!(encoder.resident_bytes()?, reconstructed.resident_bytes()?);

        let mut aliased = encoder.clone();
        let allocations_before = aliased.resident_tensor_allocations()?;
        let query = aliased
            .executable_state
            .get("encoder.layers.0.attention.q_proj.weight")
            .ok_or("query fixture state is missing")?
            .clone();
        Arc::make_mut(&mut aliased.executable_state).insert(
            "encoder.layers.0.attention.k_proj.weight".to_owned(),
            query.clone(),
        );
        aliased.semantic_state_digest_sha256 =
            aliased.project_semantic_state_digest(&cancellation)?;
        aliased.validate(&cancellation)?;
        let allocations_after = aliased.resident_tensor_allocations()?;
        assert_eq!(allocations_after.len() + 1, allocations_before.len());
        assert_eq!(
            allocations_after
                .iter()
                .filter(|(storage, _)| *storage == query.storage_id())
                .count(),
            1,
        );
        let aliased_reconstruction = aliased.reconstruct(&cancellation)?;
        assert_eq!(
            aliased.resident_tensor_allocations()?,
            aliased_reconstruction.resident_tensor_allocations()?,
        );

        let mut diagnostic_accounting = encoder.clone();
        diagnostic_accounting.diagnostic = NativeAudioEncoderDiagnostic {
            recognized_nonexecuting_state_keys: vec!["recognized".to_owned()].into(),
            missing_nonexecuting_state_keys: vec!["missing".to_owned()].into(),
            unexpected_state_keys: vec!["unexpected".to_owned()].into(),
        };
        assert!(diagnostic_accounting.resident_owned_bytes()? > encoder.resident_owned_bytes()?,);

        let audio = test_audio(&backend, &context, 1_600, MODEL_SAMPLE_RATE)?;
        let wrong_stream = test_context(&authority, &cancellation, StreamId::new(9))?;
        assert!(matches!(
            encoder.encode(&backend, &audio, &wrong_stream),
            Err(AudioEncoderError::InvalidInput(message)) if message.contains("stream")
        ));

        let before = authority.memory_snapshot();
        let mut under_budget = encoder.clone();
        under_budget.memory_budget_bytes = under_budget.resident_bytes()?;
        assert!(matches!(
            under_budget.encode(&backend, &audio, &context),
            Err(AudioEncoderError::OutOfMemory { .. })
        ));
        assert_eq!(authority.memory_snapshot(), before);

        let mut wide_front_end_profile =
            AudioEncoderExecutionProfile::reduced(NativeAudioEncoderArchitecture::Wav2Vec2Base);
        wide_front_end_profile.convolution_width = WAV2VEC2_CONV_DIMENSION;
        let mut wide_front_end = build_deterministic_audio_encoder_fixture_with_profile(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            wide_front_end_profile,
            0,
        )?;
        let waveform_values = (0..400)
            .map(|index| (index as f32 * 0.017).sin())
            .collect::<Vec<_>>();
        let waveform = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 400],
            &waveform_values,
            DType::F32,
            backend.device(),
            &context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        let short_audio = NativeAudioPayload::checked(waveform, MODEL_SAMPLE_RATE)?;
        for architecture in [
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            NativeAudioEncoderArchitecture::Wav2Vec2Large,
        ] {
            let mut source_profile = encoder.clone();
            source_profile.architecture = architecture;
            source_profile.execution_profile = AudioEncoderExecutionProfile::source(architecture);
            let required = source_profile
                .invocation_memory_upper_bound(short_audio.waveform(), short_audio.sample_rate())?;
            let minimum_front_end = (400_u64 + 2 * 400 + 2 * 512 * 79)
                .checked_mul(u64::try_from(mem::size_of::<f32>())?)
                .ok_or(AudioEncoderError::MemoryOverflow)?;
            assert!(required >= minimum_front_end);
        }
        wide_front_end.memory_budget_bytes = wide_front_end
            .resident_bytes()?
            .checked_add(100 * 1024)
            .ok_or(AudioEncoderError::MemoryOverflow)?;
        let before = authority.memory_snapshot();
        assert!(matches!(
            wide_front_end.encode(&backend, &short_audio, &context),
            Err(AudioEncoderError::OutOfMemory { .. })
        ));
        assert_eq!(authority.memory_snapshot(), before);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = test_context(&authority, &cancelled, StreamId::DEFAULT)?;
        let before = authority.memory_snapshot();
        assert!(matches!(
            encoder.encode(&backend, &audio, &cancelled_context),
            Err(AudioEncoderError::Cancelled)
        ));
        assert_eq!(authority.memory_snapshot(), before);
        Ok(())
    }

    #[test]
    fn audio_encoder_admission_and_diagnostics_are_typed() -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            map_audio_encoder_output_error(
                NativeModelPayloadError::Tensor(TensorError::Cancelled,)
            ),
            AudioEncoderError::Cancelled,
        ));
        assert!(matches!(
            map_audio_encoder_output_error(NativeModelPayloadError::Tensor(
                TensorError::ShapeOverflow,
            )),
            AudioEncoderError::Tensor(TensorError::ShapeOverflow),
        ));
        let (backend, authority) = test_backend()?;
        let cancellation = CancellationToken::default();
        let context = test_context(&authority, &cancellation, StreamId::DEFAULT)?;
        let encoder = deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            NativeAudioEncoderArchitecture::Wav2Vec2Base,
            0,
        )?;
        assert!(
            encoder
                .diagnostic()
                .recognized_nonexecuting_state_keys()
                .is_empty()
        );
        assert_eq!(
            encoder.diagnostic().missing_nonexecuting_state_keys(),
            ["masked_spec_embed"],
        );

        let mut recognized = encoder.clone();
        recognized.diagnostic = NativeAudioEncoderDiagnostic {
            recognized_nonexecuting_state_keys: vec!["masked_spec_embed".to_owned()].into(),
            missing_nonexecuting_state_keys: Vec::<String>::new().into(),
            unexpected_state_keys: Vec::<String>::new().into(),
        };
        recognized.semantic_state_digest_sha256 =
            recognized.project_semantic_state_digest(&cancellation)?;
        recognized.validate(&cancellation)?;

        let mut unexpected = encoder.clone();
        unexpected.diagnostic = NativeAudioEncoderDiagnostic {
            recognized_nonexecuting_state_keys: Vec::<String>::new().into(),
            missing_nonexecuting_state_keys: vec!["masked_spec_embed".to_owned()].into(),
            unexpected_state_keys: vec!["unused.source.tensor".to_owned()].into(),
        };
        unexpected.semantic_state_digest_sha256 =
            unexpected.project_semantic_state_digest(&cancellation)?;
        unexpected.validate(&cancellation)?;
        assert_eq!(
            unexpected
                .reconstruct(&cancellation)?
                .diagnostic()
                .unexpected_state_keys(),
            ["unused.source.tensor"],
        );

        let malformed_shape = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1],
            &[0.0],
            DType::F32,
            backend.device(),
            &context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        assert!(matches!(
            validate_state_tensor("masked_spec_embed", &malformed_shape, &[2], &context),
            Err(AudioEncoderError::StateShape { .. })
        ));
        let malformed_dtype = tensor_from_f32_with_context_exact_native(
            &backend,
            &[2],
            &[0.0, 0.0],
            DType::F16,
            backend.device(),
            &context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        assert!(matches!(
            validate_state_tensor("masked_spec_embed", &malformed_dtype, &[2], &context),
            Err(AudioEncoderError::StatePlacement { .. })
        ));

        let other_context = test_context(&authority, &cancellation, StreamId::new(17))?;
        let misplaced_audio = test_audio(&backend, &other_context, 1_600, MODEL_SAMPLE_RATE)?;
        assert!(matches!(
            encoder.encode(&backend, &misplaced_audio, &context),
            Err(AudioEncoderError::InvalidInput(message)) if message.contains("execution stream")
                || message.contains("waveform")
        ));

        let empty_waveform = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 0],
            &[],
            DType::F32,
            backend.device(),
            &context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        let empty_audio = NativeAudioPayload::checked(empty_waveform, MODEL_SAMPLE_RATE)?;
        let before = authority.memory_snapshot();
        assert!(matches!(
            encoder.encode(&backend, &empty_audio, &context),
            Err(AudioEncoderError::InvalidInput(message))
                if message.contains("nonempty AUDIO")
        ));
        assert_eq!(authority.memory_snapshot(), before);

        let low_rate_waveform = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 1],
            &[0.0],
            DType::F32,
            backend.device(),
            &context,
        )
        .map_err(|error| AudioEncoderError::TensorOperation(error.to_string()))?;
        let low_rate_audio = NativeAudioPayload::checked(low_rate_waveform, 1)?;
        let before = authority.memory_snapshot();
        assert!(matches!(
            encoder.encode(&backend, &low_rate_audio, &context),
            Err(AudioEncoderError::InvalidInput(message))
                if message.contains("8000..=384000 Hz")
        ));
        assert_eq!(authority.memory_snapshot(), before);
        Ok(())
    }
}
