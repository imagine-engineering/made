use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt,
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use collaboration_domain::{
    Huddle, HuddleIdentity, HuddleLifecycleState, HuddleParticipantPresence, PrincipalId,
};

pub const NATIVE_TTS_QUEUE_DEPTH: usize = 8;
pub const NATIVE_TTS_MAX_TOKENS_PER_CHUNK: usize = 50;
pub const NATIVE_TTS_MAX_INPUT_CHARACTERS: usize = 4_096;
pub const NATIVE_TTS_SAMPLE_RATE: u32 = 24_000;
pub const NATIVE_TTS_MAX_OUTPUT_SECONDS: usize = 30;
pub const NATIVE_TTS_MAX_OUTPUT_SAMPLES: usize =
    NATIVE_TTS_SAMPLE_RATE as usize * NATIVE_TTS_MAX_OUTPUT_SECONDS;
pub const NATIVE_TTS_MAX_IMPORTED_VOICE_BYTES: u64 = 25 * 1024 * 1024;
pub const NATIVE_TTS_MIN_VOICE_SAMPLE_RATE: u32 = 8_000;
pub const NATIVE_TTS_MAX_VOICE_SAMPLE_RATE: u32 = 96_000;
pub const NATIVE_TTS_MIN_VOICE_DURATION_MILLIS: u64 = 2_000;
pub const NATIVE_TTS_MAX_VOICE_DURATION_MILLIS: u64 = 30_000;

const NATIVE_TTS_MAX_MODEL_ARTIFACT_BYTES: u64 = 100 * 1024 * 1024;
const NATIVE_TTS_MAX_VOICES: usize = 64;
const NATIVE_TTS_MAX_FAILURE_BYTES: usize = 512;
const POCKET_APRIL_MODEL_ID: &str = "KevinAHM/pocket-tts-onnx";
const POCKET_APRIL_REVISION: &str = "58a6d00cf13d239b6748cb0769f35c580a8f606c";
const POCKET_APRIL_BUNDLE_ID: &str = "english_2026-04";

const POCKET_APRIL_ARTIFACTS: [(&str, &str, u64); 8] = [
    (
        "bundle.json",
        "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f",
        24_381,
    ),
    (
        "bos_before_voice.npy",
        "f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1",
        4_224,
    ),
    (
        "tokenizer.model",
        "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
        59_339,
    ),
    (
        "flow_lm_main_int8.onnx",
        "f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116",
        76_341_079,
    ),
    (
        "flow_lm_flow_int8.onnx",
        "3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d",
        9_962_530,
    ),
    (
        "mimi_decoder_int8.onnx",
        "3630450a3297a101792a6ac66619ebc70ab916b265e6220c2afaef8b1673f925",
        22_684_077,
    ),
    (
        "mimi_encoder.onnx",
        "853e2ca623b8782d94c3745ec6133bfdff7ce33d9b11128bd29ea03f28d76e3d",
        39_768_446,
    ),
    (
        "text_conditioner.onnx",
        "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb",
        16_388_344,
    ),
];

const POCKET_BUNDLED_VOICES: [(&str, &str); 12] = [
    (
        "pocket:anna",
        "0a6de25cf12bf1540beb85979f306a92be81fecc051c547c5395e7e5237a3856",
    ),
    (
        "pocket:vera",
        "309cf91a895830f15842b398f69a4962cb1f7e0bfab10e25dd27838e826c204b",
    ),
    (
        "pocket:fantine",
        "5f07d4e2a3f20a15572aae885156b43ef3fc12ef3812996fd135680d9956448b",
    ),
    (
        "pocket:charles",
        "6b681a429198f16e378d53bccb08d06939da7b00144a7696111d4f8f76be7756",
    ),
    (
        "pocket:paul",
        "7aba504fe0b3b16478b69eb27ce6007e3cb42b0c1915b5f1c6a6024ae37d679b",
    ),
    (
        "pocket:eponine",
        "a13c27fb47627b05223691a0ef2974358a18c886e6c2f9d2762ff1d02c20926b",
    ),
    (
        "pocket:azelma",
        "60e3d26cdf2efdec5df712152c839928f4d5522821e6554ae11fd96c57ab1026",
    ),
    (
        "pocket:george",
        "29a41f93bf5236e5b21501091d7774c255d5f3d4e62fa4f9fdf0a92a793c84ae",
    ),
    (
        "pocket:mary",
        "a35b0468382218e9f37a9a7494d1e4b74deaf18d7ced22265b4e325bb55c183f",
    ),
    (
        "pocket:jane",
        "2f12e7f155eb3118f55425394f1b049e5b1b67bdc9b3932c8ba4521420aeb84a",
    ),
    (
        "pocket:michael",
        "b6743e9195e5e3fd34fe9d1633ae93f7ffab787b249e45f6467d7d6f7a6ee6ad",
    ),
    (
        "pocket:eve",
        "396e7cbd066b0f3fb6d67fa26e7904076958239d736d4390f15b5fe88feb14cd",
    ),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTtsModelArtifact {
    file_name: String,
    sha256: String,
    size_bytes: u64,
}

impl NativeTtsModelArtifact {
    pub fn new(
        file_name: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) -> Result<Self, NativeTtsError> {
        let file_name = file_name.into();
        let sha256 = sha256.into();
        if !valid_file_name(&file_name)
            || !valid_sha256(&sha256)
            || size_bytes == 0
            || size_bytes > NATIVE_TTS_MAX_MODEL_ARTIFACT_BYTES
        {
            return Err(NativeTtsError::InvalidModel);
        }
        Ok(Self {
            file_name,
            sha256,
            size_bytes,
        })
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNativeTtsModel {
    artifacts: Vec<NativeTtsModelArtifact>,
}

impl VerifiedNativeTtsModel {
    pub fn verify_pocket_april(
        artifacts: impl IntoIterator<Item = NativeTtsModelArtifact>,
    ) -> Result<Self, NativeTtsError> {
        let artifacts: BTreeMap<_, _> = artifacts
            .into_iter()
            .map(|artifact| (artifact.file_name.clone(), artifact))
            .collect();
        if artifacts.len() != POCKET_APRIL_ARTIFACTS.len() {
            return Err(NativeTtsError::InvalidModel);
        }
        for (file_name, sha256, size_bytes) in POCKET_APRIL_ARTIFACTS {
            let Some(artifact) = artifacts.get(file_name) else {
                return Err(NativeTtsError::InvalidModel);
            };
            if artifact.sha256 != sha256 || artifact.size_bytes != size_bytes {
                return Err(NativeTtsError::InvalidModel);
            }
        }
        Ok(Self {
            artifacts: artifacts.into_values().collect(),
        })
    }

    pub const fn model_id(&self) -> &'static str {
        POCKET_APRIL_MODEL_ID
    }

    pub const fn revision(&self) -> &'static str {
        POCKET_APRIL_REVISION
    }

    pub const fn bundle_id(&self) -> &'static str {
        POCKET_APRIL_BUNDLE_ID
    }

    pub fn artifacts(&self) -> &[NativeTtsModelArtifact] {
        &self.artifacts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTtsVoiceKind {
    Bundled,
    Imported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTtsVoice {
    key: String,
    content_hash: String,
    kind: NativeTtsVoiceKind,
}

impl NativeTtsVoice {
    pub fn bundled(key: impl Into<String>) -> Result<Self, NativeTtsError> {
        let key = key.into();
        let content_hash = POCKET_BUNDLED_VOICES
            .iter()
            .find_map(|(known_key, hash)| (*known_key == key).then_some(*hash))
            .ok_or(NativeTtsError::InvalidVoice)?;
        Ok(Self {
            key,
            content_hash: content_hash.to_string(),
            kind: NativeTtsVoiceKind::Bundled,
        })
    }

    pub fn imported(
        key: impl Into<String>,
        content_hash: impl Into<String>,
        file_name: impl Into<String>,
        size_bytes: u64,
        sample_rate: u32,
        duration_millis: u64,
    ) -> Result<Self, NativeTtsError> {
        let key = key.into();
        let content_hash = content_hash.into();
        let file_name = file_name.into();
        if !valid_sha256(&content_hash)
            || key != format!("pocket:imported:{content_hash}")
            || file_name != format!("{content_hash}.wav")
            || size_bytes == 0
            || size_bytes > NATIVE_TTS_MAX_IMPORTED_VOICE_BYTES
            || !(NATIVE_TTS_MIN_VOICE_SAMPLE_RATE..=NATIVE_TTS_MAX_VOICE_SAMPLE_RATE)
                .contains(&sample_rate)
            || !(NATIVE_TTS_MIN_VOICE_DURATION_MILLIS..=NATIVE_TTS_MAX_VOICE_DURATION_MILLIS)
                .contains(&duration_millis)
        {
            return Err(NativeTtsError::InvalidVoice);
        }
        Ok(Self {
            key,
            content_hash,
            kind: NativeTtsVoiceKind::Imported,
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    pub const fn kind(&self) -> NativeTtsVoiceKind {
        self.kind
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTtsInput {
    text: String,
    token_count: usize,
}

impl NativeTtsInput {
    pub fn new(text: impl Into<String>, token_count: usize) -> Result<Self, NativeTtsError> {
        let text = text.into();
        if text.trim().is_empty()
            || text.chars().count() > NATIVE_TTS_MAX_INPUT_CHARACTERS
            || text.chars().any(|character| character == '\0')
            || token_count == 0
            || token_count > NATIVE_TTS_MAX_TOKENS_PER_CHUNK
        {
            return Err(NativeTtsError::InvalidInput);
        }
        Ok(Self { text, token_count })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn token_count(&self) -> usize {
        self.token_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTtsSpeakerKind {
    Human,
    Agent,
}

#[derive(Clone, Debug)]
pub struct NativeTtsCancellation {
    cancelled: Arc<AtomicBool>,
}

impl NativeTtsCancellation {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct NativeTtsSynthesisRequest {
    request_id: NonZeroU64,
    huddle_identity: HuddleIdentity,
    speaker_principal_id: PrincipalId,
    speaker_kind: NativeTtsSpeakerKind,
    model: VerifiedNativeTtsModel,
    voice: NativeTtsVoice,
    input: NativeTtsInput,
    cancellation: NativeTtsCancellation,
}

impl NativeTtsSynthesisRequest {
    pub const fn request_id(&self) -> NonZeroU64 {
        self.request_id
    }

    pub const fn huddle_identity(&self) -> HuddleIdentity {
        self.huddle_identity
    }

    pub const fn speaker_principal_id(&self) -> PrincipalId {
        self.speaker_principal_id
    }

    pub const fn speaker_kind(&self) -> NativeTtsSpeakerKind {
        self.speaker_kind
    }

    pub fn model(&self) -> &VerifiedNativeTtsModel {
        &self.model
    }

    pub fn voice(&self) -> &NativeTtsVoice {
        &self.voice
    }

    pub fn input(&self) -> &NativeTtsInput {
        &self.input
    }

    pub fn cancellation(&self) -> &NativeTtsCancellation {
        &self.cancellation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTtsBackendFailureCode {
    ModelLoad,
    VoiceLoad,
    Synthesis,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTtsBackendFailure {
    code: NativeTtsBackendFailureCode,
    message: String,
}

impl NativeTtsBackendFailure {
    pub fn new(
        code: NativeTtsBackendFailureCode,
        message: impl Into<String>,
    ) -> Result<Self, NativeTtsError> {
        let message = message.into();
        if message.trim().is_empty()
            || message.len() > NATIVE_TTS_MAX_FAILURE_BYTES
            || message.chars().any(char::is_control)
        {
            return Err(NativeTtsError::InvalidBackendFailure);
        }
        Ok(Self { code, message })
    }

    pub const fn code(&self) -> NativeTtsBackendFailureCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeTtsOutput {
    request_id: NonZeroU64,
    speaker_principal_id: PrincipalId,
    speaker_kind: NativeTtsSpeakerKind,
    voice_key: String,
    samples: Vec<f32>,
}

impl NativeTtsOutput {
    pub const fn request_id(&self) -> NonZeroU64 {
        self.request_id
    }

    pub const fn speaker_principal_id(&self) -> PrincipalId {
        self.speaker_principal_id
    }

    pub const fn speaker_kind(&self) -> NativeTtsSpeakerKind {
        self.speaker_kind
    }

    pub fn voice_key(&self) -> &str {
        &self.voice_key
    }

    pub const fn sample_rate(&self) -> u32 {
        NATIVE_TTS_SAMPLE_RATE
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTtsVisibleFailure {
    request_id: Option<NonZeroU64>,
    error: NativeTtsError,
    backend: Option<NativeTtsBackendFailure>,
    retryable: bool,
}

impl NativeTtsVisibleFailure {
    pub const fn request_id(&self) -> Option<NonZeroU64> {
        self.request_id
    }

    pub const fn error(&self) -> NativeTtsError {
        self.error
    }

    pub const fn retryable(&self) -> bool {
        self.retryable
    }

    pub const fn backend(&self) -> Option<&NativeTtsBackendFailure> {
        self.backend.as_ref()
    }
}

pub struct NativeTtsService {
    huddle_identity: HuddleIdentity,
    model: Option<VerifiedNativeTtsModel>,
    voices: BTreeMap<String, NativeTtsVoice>,
    selected_voice: Option<String>,
    queued: VecDeque<NativeTtsSynthesisRequest>,
    in_flight: Option<NativeTtsSynthesisRequest>,
    retry_request: Option<NativeTtsSynthesisRequest>,
    last_failure: Option<NativeTtsVisibleFailure>,
    next_request_id: u64,
}

impl NativeTtsService {
    pub fn new(huddle_identity: HuddleIdentity) -> Self {
        Self {
            huddle_identity,
            model: None,
            voices: BTreeMap::new(),
            selected_voice: None,
            queued: VecDeque::new(),
            in_flight: None,
            retry_request: None,
            last_failure: None,
            next_request_id: 1,
        }
    }

    pub const fn huddle_identity(&self) -> HuddleIdentity {
        self.huddle_identity
    }

    pub fn install_model(&mut self, model: VerifiedNativeTtsModel) {
        self.model = Some(model);
        if self
            .last_failure
            .as_ref()
            .is_some_and(|failure| failure.error == NativeTtsError::MissingModel)
        {
            self.last_failure = None;
        }
    }

    pub fn remove_model(&mut self) {
        self.model = None;
        self.cancel_all();
    }

    pub fn register_voice(&mut self, voice: NativeTtsVoice) -> Result<(), NativeTtsError> {
        if !self.voices.contains_key(voice.key()) && self.voices.len() >= NATIVE_TTS_MAX_VOICES {
            return self.fail(None, NativeTtsError::VoiceLimitReached, None, false);
        }
        self.voices.insert(voice.key.clone(), voice);
        Ok(())
    }

    pub fn select_voice(&mut self, key: &str) -> Result<(), NativeTtsError> {
        if !self.voices.contains_key(key) {
            return self.fail(None, NativeTtsError::InvalidVoice, None, false);
        }
        if self.selected_voice.as_deref() != Some(key) {
            self.cancel_all();
            self.selected_voice = Some(key.to_string());
        }
        self.last_failure = None;
        Ok(())
    }

    pub fn selected_voice(&self) -> Option<&NativeTtsVoice> {
        self.selected_voice
            .as_ref()
            .and_then(|key| self.voices.get(key))
    }

    pub fn enqueue(
        &mut self,
        huddle: &Huddle,
        speaker_principal_id: PrincipalId,
        speaker_kind: NativeTtsSpeakerKind,
        input: NativeTtsInput,
    ) -> Result<NonZeroU64, NativeTtsError> {
        self.validate_huddle(huddle, speaker_principal_id)?;
        let Some(model) = self.model.clone() else {
            return self.fail(None, NativeTtsError::MissingModel, None, true);
        };
        let Some(voice) = self.selected_voice().cloned() else {
            return self.fail(None, NativeTtsError::InvalidVoice, None, false);
        };
        if self.queued.len() >= NATIVE_TTS_QUEUE_DEPTH {
            return self.fail(None, NativeTtsError::QueueFull, None, true);
        }
        let request_id = self.allocate_request_id()?;
        self.queued.push_back(NativeTtsSynthesisRequest {
            request_id,
            huddle_identity: self.huddle_identity,
            speaker_principal_id,
            speaker_kind,
            model,
            voice,
            input,
            cancellation: NativeTtsCancellation {
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        });
        self.last_failure = None;
        Ok(request_id)
    }

    pub fn start_next(&mut self) -> Result<Option<NativeTtsSynthesisRequest>, NativeTtsError> {
        if self.in_flight.is_some() {
            return self.fail(None, NativeTtsError::SynthesisInProgress, None, true);
        }
        let Some(request) = self.queued.pop_front() else {
            return Ok(None);
        };
        self.in_flight = Some(request.clone());
        Ok(Some(request))
    }

    pub fn complete(
        &mut self,
        request_id: NonZeroU64,
        result: Result<Vec<f32>, NativeTtsBackendFailure>,
    ) -> Result<NativeTtsOutput, NativeTtsError> {
        let Some(request) = self.in_flight.take() else {
            return Err(NativeTtsError::StaleCompletion);
        };
        if request.request_id != request_id {
            self.in_flight = Some(request);
            return Err(NativeTtsError::StaleCompletion);
        }
        if request.cancellation.is_cancelled() {
            return self.fail(Some(request_id), NativeTtsError::Cancelled, None, true);
        }
        let samples = match result {
            Ok(samples) => samples,
            Err(backend) => {
                self.retry_request = Some(request);
                return self.fail(
                    Some(request_id),
                    NativeTtsError::BackendFailed,
                    Some(backend),
                    true,
                );
            }
        };
        if samples.is_empty()
            || samples.len() > NATIVE_TTS_MAX_OUTPUT_SAMPLES
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            self.retry_request = Some(request);
            return self.fail(Some(request_id), NativeTtsError::InvalidOutput, None, true);
        }
        self.retry_request = None;
        self.last_failure = None;
        Ok(NativeTtsOutput {
            request_id,
            speaker_principal_id: request.speaker_principal_id,
            speaker_kind: request.speaker_kind,
            voice_key: request.voice.key,
            samples,
        })
    }

    pub fn cancel(&mut self, request_id: NonZeroU64) -> Result<(), NativeTtsError> {
        if self
            .in_flight
            .as_ref()
            .is_some_and(|request| request.request_id == request_id)
        {
            let request = self
                .in_flight
                .take()
                .ok_or(NativeTtsError::StaleCompletion)?;
            request
                .cancellation
                .cancelled
                .store(true, Ordering::Release);
            self.retry_request = None;
            self.last_failure = Some(NativeTtsVisibleFailure {
                request_id: Some(request_id),
                error: NativeTtsError::Cancelled,
                backend: None,
                retryable: true,
            });
            return Ok(());
        }
        if let Some(position) = self
            .queued
            .iter()
            .position(|request| request.request_id == request_id)
        {
            self.queued.remove(position);
            self.last_failure = Some(NativeTtsVisibleFailure {
                request_id: Some(request_id),
                error: NativeTtsError::Cancelled,
                backend: None,
                retryable: true,
            });
            return Ok(());
        }
        self.fail(
            Some(request_id),
            NativeTtsError::StaleCompletion,
            None,
            false,
        )
    }

    pub fn retry_last(&mut self, huddle: &Huddle) -> Result<NonZeroU64, NativeTtsError> {
        let Some(request) = self.retry_request.as_ref() else {
            return self.fail(None, NativeTtsError::NothingToRetry, None, false);
        };
        self.validate_huddle(huddle, request.speaker_principal_id)?;
        if self.queued.len() >= NATIVE_TTS_QUEUE_DEPTH {
            return self.fail(None, NativeTtsError::QueueFull, None, true);
        }
        let mut request = self
            .retry_request
            .take()
            .ok_or(NativeTtsError::NothingToRetry)?;
        let request_id = self.allocate_request_id()?;
        request.request_id = request_id;
        request.cancellation = NativeTtsCancellation {
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        self.queued.push_back(request);
        self.last_failure = None;
        Ok(request_id)
    }

    pub fn cancel_all(&mut self) {
        for request in self.queued.drain(..) {
            request
                .cancellation
                .cancelled
                .store(true, Ordering::Release);
        }
        if let Some(request) = self.in_flight.take() {
            request
                .cancellation
                .cancelled
                .store(true, Ordering::Release);
        }
        self.retry_request = None;
    }

    pub fn queued_count(&self) -> usize {
        self.queued.len()
    }

    pub fn in_flight_request_id(&self) -> Option<NonZeroU64> {
        self.in_flight.as_ref().map(|request| request.request_id)
    }

    pub const fn last_failure(&self) -> Option<&NativeTtsVisibleFailure> {
        self.last_failure.as_ref()
    }

    fn allocate_request_id(&mut self) -> Result<NonZeroU64, NativeTtsError> {
        let request_id =
            NonZeroU64::new(self.next_request_id).ok_or(NativeTtsError::RequestIdsExhausted)?;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(NativeTtsError::RequestIdsExhausted)?;
        Ok(request_id)
    }

    fn validate_huddle(
        &mut self,
        huddle: &Huddle,
        speaker_principal_id: PrincipalId,
    ) -> Result<(), NativeTtsError> {
        let error = if huddle.identity() != self.huddle_identity {
            Some(NativeTtsError::WrongHuddle)
        } else if huddle.lifecycle() != HuddleLifecycleState::Active {
            Some(NativeTtsError::HuddleEnded)
        } else if huddle
            .participant(speaker_principal_id)
            .is_none_or(|participant| participant.presence() != HuddleParticipantPresence::Present)
        {
            Some(NativeTtsError::SpeakerUnavailable)
        } else {
            None
        };
        match error {
            Some(error) => self.fail(None, error, None, false),
            None => Ok(()),
        }
    }

    fn fail<T>(
        &mut self,
        request_id: Option<NonZeroU64>,
        error: NativeTtsError,
        backend: Option<NativeTtsBackendFailure>,
        retryable: bool,
    ) -> Result<T, NativeTtsError> {
        self.last_failure = Some(NativeTtsVisibleFailure {
            request_id,
            error,
            backend,
            retryable,
        });
        Err(error)
    }
}

impl Drop for NativeTtsService {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTtsError {
    InvalidModel,
    MissingModel,
    InvalidVoice,
    VoiceLimitReached,
    InvalidInput,
    QueueFull,
    SynthesisInProgress,
    InvalidBackendFailure,
    BackendFailed,
    InvalidOutput,
    Cancelled,
    StaleCompletion,
    NothingToRetry,
    WrongHuddle,
    HuddleEnded,
    SpeakerUnavailable,
    RequestIdsExhausted,
}

impl fmt::Display for NativeTtsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidModel => "local TTS model verification failed",
            Self::MissingModel => "local TTS model is not installed",
            Self::InvalidVoice => "local TTS voice is invalid or unavailable",
            Self::VoiceLimitReached => "local TTS voice limit reached",
            Self::InvalidInput => "local TTS input exceeds the model limits",
            Self::QueueFull => "local TTS queue is full",
            Self::SynthesisInProgress => "local TTS synthesis is already in progress",
            Self::InvalidBackendFailure => "local TTS backend failure is invalid",
            Self::BackendFailed => "local TTS synthesis failed",
            Self::InvalidOutput => "local TTS output exceeds the audio limits",
            Self::Cancelled => "local TTS synthesis was cancelled",
            Self::StaleCompletion => "local TTS completion is stale",
            Self::NothingToRetry => "local TTS has no failed synthesis to retry",
            Self::WrongHuddle => "local TTS request belongs to another huddle",
            Self::HuddleEnded => "local TTS huddle has ended",
            Self::SpeakerUnavailable => "local TTS speaker is not present",
            Self::RequestIdsExhausted => "local TTS request identifiers exhausted",
        };
        formatter.write_str(message)
    }
}

impl Error for NativeTtsError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_file_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value != "."
        && value != ".."
        && !value.contains(['/', '\\'])
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{
        AggregateId, CommunityId, HuddleGeneration, HuddleParticipantRole, OperationId,
    };

    use super::*;

    fn huddle() -> (Huddle, PrincipalId) {
        let owner = PrincipalId::new();
        let speaker = PrincipalId::new();
        let identity = HuddleIdentity::new(
            CommunityId::new(),
            AggregateId::new(),
            AggregateId::new(),
            HuddleGeneration::new(3).expect("generation"),
        )
        .expect("identity");
        let mut huddle =
            Huddle::start(identity, owner, OperationId::new(), 1).expect("start huddle");
        huddle
            .join(
                speaker,
                HuddleParticipantRole::Speaker,
                OperationId::new(),
                2,
            )
            .expect("join speaker");
        (huddle, speaker)
    }

    fn model() -> VerifiedNativeTtsModel {
        VerifiedNativeTtsModel::verify_pocket_april(POCKET_APRIL_ARTIFACTS.map(
            |(file_name, sha256, size_bytes)| {
                NativeTtsModelArtifact::new(file_name, sha256, size_bytes).expect("artifact")
            },
        ))
        .expect("verified model")
    }

    fn ready_service(huddle: &Huddle) -> NativeTtsService {
        let mut service = NativeTtsService::new(huddle.identity());
        service.install_model(model());
        service
            .register_voice(NativeTtsVoice::bundled("pocket:mary").expect("voice"))
            .expect("register voice");
        service.select_voice("pocket:mary").expect("select voice");
        service
    }

    fn input() -> NativeTtsInput {
        NativeTtsInput::new("The native service is bounded.", 7).expect("input")
    }

    #[test]
    fn missing_model_is_visible_and_install_recovers_without_ending_huddle() {
        let mut artifacts = POCKET_APRIL_ARTIFACTS
            .map(|(file_name, sha256, size_bytes)| {
                NativeTtsModelArtifact::new(file_name, sha256, size_bytes).expect("artifact")
            })
            .to_vec();
        artifacts[0] =
            NativeTtsModelArtifact::new("bundle.json", "0".repeat(64), POCKET_APRIL_ARTIFACTS[0].2)
                .expect("well-formed mismatched artifact");
        assert_eq!(
            VerifiedNativeTtsModel::verify_pocket_april(artifacts),
            Err(NativeTtsError::InvalidModel)
        );
        assert_eq!(
            NativeTtsModelArtifact::new("../bundle.json", "0".repeat(64), 1),
            Err(NativeTtsError::InvalidModel)
        );

        let (huddle, speaker) = huddle();
        let mut service = NativeTtsService::new(huddle.identity());
        service
            .register_voice(NativeTtsVoice::bundled("pocket:mary").expect("voice"))
            .expect("register voice");
        service.select_voice("pocket:mary").expect("select voice");

        assert_eq!(
            service.enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input()),
            Err(NativeTtsError::MissingModel)
        );
        assert_eq!(
            service.last_failure().map(NativeTtsVisibleFailure::error),
            Some(NativeTtsError::MissingModel)
        );
        assert_eq!(huddle.lifecycle(), HuddleLifecycleState::Active);

        service.install_model(model());
        assert!(
            service
                .enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input())
                .is_ok()
        );
    }

    #[test]
    fn invalid_imported_voice_identity_and_bounds_are_rejected() {
        let (huddle, _) = huddle();
        let mut service = NativeTtsService::new(huddle.identity());
        assert_eq!(
            service.select_voice("pocket:unknown"),
            Err(NativeTtsError::InvalidVoice)
        );
        assert_eq!(
            service.last_failure().map(NativeTtsVisibleFailure::error),
            Some(NativeTtsError::InvalidVoice)
        );

        let hash = "a".repeat(64);
        assert_eq!(
            NativeTtsVoice::imported(
                "pocket:imported:wrong",
                &hash,
                format!("{hash}.wav"),
                1_024,
                32_000,
                3_000,
            ),
            Err(NativeTtsError::InvalidVoice)
        );
        assert_eq!(
            NativeTtsVoice::imported(
                format!("pocket:imported:{hash}"),
                &hash,
                format!("{hash}.wav"),
                NATIVE_TTS_MAX_IMPORTED_VOICE_BYTES + 1,
                32_000,
                3_000,
            ),
            Err(NativeTtsError::InvalidVoice)
        );
    }

    #[test]
    fn cancellation_signals_worker_and_stale_completion_cannot_publish() {
        let (huddle, speaker) = huddle();
        let mut service = ready_service(&huddle);
        let request_id = service
            .enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input())
            .expect("enqueue");
        let request = service.start_next().expect("start").expect("request");

        service.cancel(request_id).expect("cancel");
        assert!(request.cancellation().is_cancelled());
        assert_eq!(
            service.complete(request_id, Ok(vec![0.0; 24_000])),
            Err(NativeTtsError::StaleCompletion)
        );
        assert_eq!(
            service.last_failure().map(NativeTtsVisibleFailure::error),
            Some(NativeTtsError::Cancelled)
        );
        assert_eq!(huddle.lifecycle(), HuddleLifecycleState::Active);
    }

    #[test]
    fn queue_and_pcm_output_are_strictly_bounded() {
        let (huddle, speaker) = huddle();
        let mut service = ready_service(&huddle);
        for _ in 0..NATIVE_TTS_QUEUE_DEPTH {
            service
                .enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input())
                .expect("bounded enqueue");
        }
        assert_eq!(
            service.enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input()),
            Err(NativeTtsError::QueueFull)
        );
        let request = service.start_next().expect("start").expect("request");
        assert_eq!(
            service.complete(
                request.request_id(),
                Ok(vec![0.0; NATIVE_TTS_MAX_OUTPUT_SAMPLES + 1]),
            ),
            Err(NativeTtsError::InvalidOutput)
        );
        assert_eq!(
            service.last_failure().map(NativeTtsVisibleFailure::error),
            Some(NativeTtsError::InvalidOutput)
        );
    }

    #[test]
    fn backend_failure_retries_with_new_fence_and_preserves_attribution() {
        let (huddle, speaker) = huddle();
        let mut service = ready_service(&huddle);
        let first_id = service
            .enqueue(&huddle, speaker, NativeTtsSpeakerKind::Agent, input())
            .expect("enqueue");
        service.start_next().expect("start").expect("request");
        let backend = NativeTtsBackendFailure::new(
            NativeTtsBackendFailureCode::Synthesis,
            "local inference stopped",
        )
        .expect("backend failure");
        assert_eq!(
            service.complete(first_id, Err(backend.clone())),
            Err(NativeTtsError::BackendFailed)
        );
        assert_eq!(
            service.last_failure().and_then(|failure| failure.backend()),
            Some(&backend)
        );

        let retry_id = service.retry_last(&huddle).expect("retry");
        assert_ne!(retry_id, first_id);
        let retry = service.start_next().expect("start retry").expect("request");
        let output = service
            .complete(retry.request_id(), Ok(vec![0.25; 24_000]))
            .expect("complete retry");
        assert_eq!(output.speaker_principal_id(), speaker);
        assert_eq!(output.speaker_kind(), NativeTtsSpeakerKind::Agent);
        assert_eq!(output.voice_key(), "pocket:mary");
        assert_eq!(output.sample_rate(), NATIVE_TTS_SAMPLE_RATE);
        assert!(service.last_failure().is_none());
    }
}
