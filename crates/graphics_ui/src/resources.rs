use graphics_core::resources::{
    AdmittedCpuImage, CpuImageDescriptor, CpuImageOrigin, EncodedPngBudget, ImageResourceIdentity,
    PngAdmissionError, PngNetworkAdmissionError, PreparedEncodedPngSource,
    admit_encoded_png_via_original_network, prepare_encoded_png_source,
};
use sim_runtime::{
    AuthorizedCapabilities,
    assets::{AssetError, AssetIdentity, AssetNamespace, AssetRecord, SharedAssetService},
};
use sim_tensor::CancellationToken;
use std::{fmt, sync::Arc};

#[derive(Clone, Debug)]
pub struct VerifiedGraphiteAsset {
    record: AssetRecord,
    bytes: Arc<[u8]>,
}

impl VerifiedGraphiteAsset {
    pub fn record(&self) -> &AssetRecord {
        &self.record
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug)]
pub enum GraphiteAssetReadError {
    Asset(AssetError),
    Admission(PngAdmissionError),
    Network(Box<PngNetworkAdmissionError>),
    Resource(ImageResourceError),
    ServicePoisoned,
    MissingRecord(AssetIdentity),
}

impl fmt::Display for GraphiteAssetReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asset(error) => fmt::Display::fmt(error, formatter),
            Self::Admission(error) => fmt::Display::fmt(error, formatter),
            Self::Network(error) => fmt::Display::fmt(error, formatter),
            Self::Resource(error) => fmt::Display::fmt(error, formatter),
            Self::ServicePoisoned => formatter.write_str("The asset service lock is poisoned."),
            Self::MissingRecord(identity) => write!(
                formatter,
                "The verified asset has no current indexed metadata: {identity:?}."
            ),
        }
    }
}

impl std::error::Error for GraphiteAssetReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Asset(error) => Some(error),
            Self::Admission(error) => Some(error),
            Self::Network(error) => Some(error.as_ref()),
            Self::Resource(error) => Some(error),
            Self::ServicePoisoned | Self::MissingRecord(_) => None,
        }
    }
}

impl From<AssetError> for GraphiteAssetReadError {
    fn from(error: AssetError) -> Self {
        Self::Asset(error)
    }
}

impl From<PngAdmissionError> for GraphiteAssetReadError {
    fn from(error: PngAdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<PngNetworkAdmissionError> for GraphiteAssetReadError {
    fn from(error: PngNetworkAdmissionError) -> Self {
        match error {
            PngNetworkAdmissionError::Admission(error) => Self::Admission(error),
            error => Self::Network(Box::new(error)),
        }
    }
}

impl From<ImageResourceError> for GraphiteAssetReadError {
    fn from(error: ImageResourceError) -> Self {
        Self::Resource(error)
    }
}

pub fn read_verified_graphite_asset(
    assets: &SharedAssetService,
    identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    maximum_bytes: u64,
) -> Result<VerifiedGraphiteAsset, GraphiteAssetReadError> {
    let (record, bytes) = {
        let service = assets
            .lock()
            .map_err(|_| GraphiteAssetReadError::ServicePoisoned)?;
        let bytes = service.read_verified(identity, authorization, cancellation, maximum_bytes)?;
        // One guard prevents a record refresh from racing the verified bytes-to-metadata
        // association, and metadata is never returned before verification succeeds.
        let record = service
            .record(identity)
            .ok_or_else(|| GraphiteAssetReadError::MissingRecord(identity.clone()))?;
        (record, bytes)
    };
    publish_verified_graphite_asset(record, bytes, cancellation)
}

fn publish_verified_graphite_asset(
    record: AssetRecord,
    bytes: Vec<u8>,
    cancellation: &CancellationToken,
) -> Result<VerifiedGraphiteAsset, GraphiteAssetReadError> {
    let verified = VerifiedGraphiteAsset {
        record,
        bytes: bytes.into(),
    };
    if cancellation.is_cancelled() {
        return Err(GraphiteAssetReadError::Asset(AssetError::Cancelled));
    }
    Ok(verified)
}

pub struct VerifiedGraphiteEncodedPngSource {
    record: AssetRecord,
    source: Arc<PreparedEncodedPngSource>,
}

impl VerifiedGraphiteEncodedPngSource {
    pub fn record(&self) -> &AssetRecord {
        &self.record
    }

    pub fn source(&self) -> &Arc<PreparedEncodedPngSource> {
        &self.source
    }
}

pub fn prepare_graphite_encoded_png_source(
    assets: &SharedAssetService,
    asset_identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    identity: &ImageResourceIdentity,
    budget: EncodedPngBudget,
) -> Result<VerifiedGraphiteEncodedPngSource, GraphiteAssetReadError> {
    if asset_identity.namespace != AssetNamespace::Input {
        return Err(AssetError::UnsupportedNamespace(asset_identity.namespace).into());
    }
    prepare_graphite_authorized_encoded_png_source(
        assets,
        asset_identity,
        authorization,
        cancellation,
        identity,
        budget,
    )
}

pub(crate) fn prepare_graphite_authorized_encoded_png_source(
    assets: &SharedAssetService,
    asset_identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    identity: &ImageResourceIdentity,
    budget: EncodedPngBudget,
) -> Result<VerifiedGraphiteEncodedPngSource, GraphiteAssetReadError> {
    if !matches!(
        asset_identity.namespace,
        AssetNamespace::Input | AssetNamespace::Output
    ) {
        return Err(AssetError::UnsupportedNamespace(asset_identity.namespace).into());
    }
    check_graphite_asset_active(cancellation)?;
    let maximum_bytes = u64::try_from(budget.max_encoded_bytes)
        .map_err(|_| GraphiteAssetReadError::Asset(AssetError::InvalidLimit))?;
    let verified = read_verified_graphite_asset(
        assets,
        asset_identity,
        authorization,
        cancellation,
        maximum_bytes,
    )?;
    if identity.asset_id.as_ref() != verified.record.identity.to_reference()? {
        return Err(ImageResourceError::IdentityMismatch.into());
    }
    check_graphite_asset_active(cancellation)?;
    let source = Arc::new(prepare_encoded_png_source(
        identity.clone(),
        verified.bytes,
        budget,
    )?);
    check_graphite_asset_active(cancellation)?;
    Ok(VerifiedGraphiteEncodedPngSource {
        record: verified.record,
        source,
    })
}

#[derive(Clone, Debug)]
pub enum GraphiteResourceLoadError {
    Read(Arc<GraphiteAssetReadError>),
    RequestedHashMismatch {
        expected: graphite_resource::ResourceHash,
        requested: graphite_resource::ResourceHash,
    },
    ContentHashMismatch {
        expected: graphite_resource::ResourceHash,
        actual: graphite_resource::ResourceHash,
    },
    DiagnosticPoisoned,
}

impl fmt::Display for GraphiteResourceLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => fmt::Display::fmt(error, formatter),
            Self::RequestedHashMismatch {
                expected,
                requested,
            } => write!(
                formatter,
                "The requested original resource hash {requested} does not match {expected}."
            ),
            Self::ContentHashMismatch { expected, actual } => write!(
                formatter,
                "The authenticated original resource hash {actual} does not match {expected}."
            ),
            Self::DiagnosticPoisoned => {
                formatter.write_str("The resource load diagnostic lock is poisoned.")
            }
        }
    }
}

impl std::error::Error for GraphiteResourceLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

pub struct GraphiteAssetResourceProxy {
    assets: SharedAssetService,
    asset_identity: AssetIdentity,
    authorization: AuthorizedCapabilities,
    expected_hash: graphite_resource::ResourceHash,
    cancellation: CancellationToken,
    maximum_encoded_bytes: u64,
    diagnostic: std::sync::Mutex<Option<Arc<GraphiteResourceLoadError>>>,
}

impl GraphiteAssetResourceProxy {
    pub fn new(
        assets: SharedAssetService,
        asset_identity: AssetIdentity,
        authorization: AuthorizedCapabilities,
        expected_hash: graphite_resource::ResourceHash,
        cancellation: CancellationToken,
        maximum_encoded_bytes: u64,
    ) -> Self {
        Self {
            assets,
            asset_identity,
            authorization,
            expected_hash,
            cancellation,
            maximum_encoded_bytes,
            diagnostic: std::sync::Mutex::new(None),
        }
    }

    pub fn last_diagnostic(
        &self,
    ) -> Result<Option<Arc<GraphiteResourceLoadError>>, GraphiteResourceLoadError> {
        self.diagnostic
            .lock()
            .map(|diagnostic| diagnostic.clone())
            .map_err(|_| GraphiteResourceLoadError::DiagnosticPoisoned)
    }

    fn read_resource(
        &self,
        requested: graphite_resource::ResourceHash,
    ) -> Result<graphite_resource::Resource, GraphiteResourceLoadError> {
        if requested != self.expected_hash {
            return Err(GraphiteResourceLoadError::RequestedHashMismatch {
                expected: self.expected_hash,
                requested,
            });
        }
        let verified = read_verified_graphite_asset(
            &self.assets,
            &self.asset_identity,
            &self.authorization,
            &self.cancellation,
            self.maximum_encoded_bytes,
        )
        .map_err(|error| GraphiteResourceLoadError::Read(Arc::new(error)))?;
        let resource = graphite_resource::Resource::new(verified.bytes);
        check_graphite_asset_active(&self.cancellation)
            .map_err(|error| GraphiteResourceLoadError::Read(Arc::new(error)))?;
        if resource.hash() != self.expected_hash {
            return Err(GraphiteResourceLoadError::ContentHashMismatch {
                expected: self.expected_hash,
                actual: resource.hash(),
            });
        }
        Ok(resource)
    }

    fn finish_load(
        &self,
        result: Result<graphite_resource::Resource, GraphiteResourceLoadError>,
    ) -> Option<graphite_resource::Resource> {
        let mut diagnostic = match self.diagnostic.lock() {
            Ok(diagnostic) => diagnostic,
            Err(_) => {
                if let Err(error) = &result {
                    log::error!("Graphite resource load failed: {error}");
                }
                log::error!("{}", GraphiteResourceLoadError::DiagnosticPoisoned);
                return None;
            }
        };
        let result = result.and_then(|resource| {
            check_graphite_asset_active(&self.cancellation)
                .map_err(|error| GraphiteResourceLoadError::Read(Arc::new(error)))?;
            Ok(resource)
        });
        match result {
            Ok(resource) => {
                *diagnostic = None;
                Some(resource)
            }
            Err(error) => {
                log::error!("Graphite resource load failed: {error}");
                *diagnostic = Some(Arc::new(error));
                None
            }
        }
    }
}

impl graphite_resource::LoadResource for GraphiteAssetResourceProxy {
    fn load(&self, hash: graphite_resource::ResourceHash) -> graphite_resource::ResourceFuture<'_> {
        Box::pin(async move { self.finish_load(self.read_resource(hash)) })
    }
}

struct GraphiteResourceProxyHandle(Arc<GraphiteAssetResourceProxy>);

impl graphite_resource::LoadResource for GraphiteResourceProxyHandle {
    fn load(&self, hash: graphite_resource::ResourceHash) -> graphite_resource::ResourceFuture<'_> {
        graphite_resource::LoadResource::load(self.0.as_ref(), hash)
    }
}

pub fn graphite_resource_editor_api(
    proxy: Arc<GraphiteAssetResourceProxy>,
) -> Arc<graphite_graph_craft::application_io::PlatformEditorApi> {
    use graphite_graph_craft::application_io::{PlatformApplicationIo, PlatformEditorApi};

    let mut application_io = PlatformApplicationIo::default();
    application_io.inject_resource_proxy(Box::new(GraphiteResourceProxyHandle(proxy)));
    Arc::new(PlatformEditorApi {
        application_io: Some(Arc::new(application_io)),
        ..PlatformEditorApi::default()
    })
}

pub fn resolve_graphite_png_request(
    assets: &SharedAssetService,
    asset_identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    request: &ImageResourceRequest,
    budget: EncodedPngBudget,
) -> Result<AdmittedCpuImage, GraphiteAssetReadError> {
    let image = resolve_graphite_png_bytes(
        assets,
        asset_identity,
        authorization,
        cancellation,
        request.identity(),
        request.descriptor().transform,
        budget,
    )?;
    finish_graphite_png_request(image, request, cancellation)
}

pub fn resolve_graphite_encoded_png_request(
    assets: &SharedAssetService,
    asset_identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    request: &EncodedPngResourceRequest,
) -> Result<ResolvedEncodedPngImage, GraphiteAssetReadError> {
    let image = resolve_graphite_png_bytes(
        assets,
        asset_identity,
        authorization,
        cancellation,
        request.identity(),
        request.transform(),
        request.budget(),
    )?;
    check_graphite_asset_active(cancellation)?;
    Ok(ResolvedEncodedPngImage {
        image,
        request: request.clone(),
    })
}

fn resolve_graphite_png_bytes(
    assets: &SharedAssetService,
    asset_identity: &AssetIdentity,
    authorization: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
    identity: &ImageResourceIdentity,
    transform: [f64; 6],
    budget: EncodedPngBudget,
) -> Result<AdmittedCpuImage, GraphiteAssetReadError> {
    let maximum_bytes = u64::try_from(budget.max_encoded_bytes)
        .map_err(|_| GraphiteAssetReadError::Asset(AssetError::InvalidLimit))?;
    let verified = read_verified_graphite_asset(
        assets,
        asset_identity,
        authorization,
        cancellation,
        maximum_bytes,
    )?;
    let reference = verified.record().identity.to_reference()?;
    if identity.asset_id.as_ref() != reference {
        return Err(ImageResourceError::IdentityMismatch.into());
    }
    check_graphite_asset_active(cancellation)?;
    // Interpreter futures stay on this background job's thread; callers must not resolve on GPUI's foreground.
    let image = futures::executor::block_on(admit_encoded_png_via_original_network(
        identity.clone(),
        transform,
        verified.bytes,
        budget,
    ))?;
    check_graphite_asset_active(cancellation)?;
    Ok(image)
}

fn check_graphite_asset_active(
    cancellation: &CancellationToken,
) -> Result<(), GraphiteAssetReadError> {
    cancellation
        .check()
        .map_err(|_| GraphiteAssetReadError::Asset(AssetError::Cancelled))
}

fn finish_graphite_png_request(
    image: AdmittedCpuImage,
    request: &ImageResourceRequest,
    cancellation: &CancellationToken,
) -> Result<AdmittedCpuImage, GraphiteAssetReadError> {
    check_graphite_asset_active(cancellation)?;
    if !same_descriptor(image.descriptor(), request.descriptor()) {
        return Err(ImageResourceError::DescriptorMismatch.into());
    }
    check_graphite_asset_active(cancellation)?;
    Ok(image)
}

#[derive(Clone, Debug)]
pub struct ImageResourceRequest<Descriptor = CpuImageDescriptor> {
    owner: Arc<()>,
    generation: u64,
    identity: ImageResourceIdentity,
    descriptor: Descriptor,
}

impl<Descriptor> ImageResourceRequest<Descriptor> {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn identity(&self) -> &ImageResourceIdentity {
        &self.identity
    }
}

impl ImageResourceRequest {
    pub fn descriptor(&self) -> &CpuImageDescriptor {
        &self.descriptor
    }
}

#[derive(Clone, Debug)]
pub enum ImageRequestDecision<Descriptor = CpuImageDescriptor> {
    Reuse(Arc<AdmittedCpuImage>),
    Load(ImageResourceRequest<Descriptor>),
}

#[derive(Clone, Debug)]
pub struct EncodedPngExpectation {
    transform: [f64; 6],
    budget: EncodedPngBudget,
}

pub type EncodedPngResourceRequest = ImageResourceRequest<EncodedPngExpectation>;

impl EncodedPngResourceRequest {
    pub fn transform(&self) -> [f64; 6] {
        self.descriptor.transform
    }

    pub fn budget(&self) -> EncodedPngBudget {
        self.descriptor.budget
    }
}

#[derive(Debug)]
pub struct ResolvedEncodedPngImage {
    image: AdmittedCpuImage,
    request: EncodedPngResourceRequest,
}

impl ResolvedEncodedPngImage {
    pub fn image(&self) -> &AdmittedCpuImage {
        &self.image
    }
}

#[derive(Clone, Debug)]
enum ActiveImageRequest {
    Known(ImageResourceRequest),
    Encoded(EncodedPngResourceRequest),
}

impl ActiveImageRequest {
    fn identity(&self) -> &ImageResourceIdentity {
        match self {
            Self::Known(request) => request.identity(),
            Self::Encoded(request) => request.identity(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageResourceFailure {
    Denied,
    Cancelled,
    Failed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageResourceError {
    ForeignRequest,
    StaleRequest,
    IdentityMismatch,
    DescriptorMismatch,
    NonFiniteTransform,
    Denied,
    Cancelled,
    Failed(String),
    Revoked,
    UnknownResource,
    GenerationExhausted,
    RevisionExhausted,
}

impl fmt::Display for ImageResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignRequest => {
                formatter.write_str("The image request belongs to another session.")
            }
            Self::StaleRequest => formatter.write_str("The image request is no longer active."),
            Self::IdentityMismatch => {
                formatter.write_str("The image does not match the requested asset version.")
            }
            Self::DescriptorMismatch => {
                formatter.write_str("The image does not match the requested pixel description.")
            }
            Self::NonFiniteTransform => {
                formatter.write_str("The image transform must contain finite components.")
            }
            Self::Denied => formatter.write_str("Access to the requested image was denied."),
            Self::Cancelled => formatter.write_str("The image request was cancelled."),
            Self::Failed(message) => write!(formatter, "The image request failed: {message}"),
            Self::Revoked => {
                formatter.write_str("The image is no longer available for this request.")
            }
            Self::UnknownResource => formatter
                .write_str("The image does not belong to this session's current resources."),
            Self::GenerationExhausted => {
                formatter.write_str("The image request generation limit was reached.")
            }
            Self::RevisionExhausted => {
                formatter.write_str("The committed image revision limit was reached.")
            }
        }
    }
}

impl std::error::Error for ImageResourceError {}

#[derive(Debug, Default)]
pub struct ImageResourceState {
    owner: Arc<()>,
    generation: u64,
    active_request: Option<ActiveImageRequest>,
    committed: Option<Arc<AdmittedCpuImage>>,
    revision: u64,
    fresh: bool,
    committed_revoked: bool,
    diagnostic: Option<ImageResourceError>,
}

impl ImageResourceState {
    pub fn request(
        &mut self,
        identity: ImageResourceIdentity,
        descriptor: CpuImageDescriptor,
    ) -> Result<ImageRequestDecision, ImageResourceError> {
        if descriptor
            .transform
            .iter()
            .any(|component| !component.is_finite())
        {
            return self.record_error(ImageResourceError::NonFiniteTransform);
        }
        if self.fresh
            && let Some(image) = self.committed.as_ref()
            && image.identity() == &identity
            && same_descriptor(image.descriptor(), &descriptor)
        {
            self.active_request = None;
            self.diagnostic = None;
            return Ok(ImageRequestDecision::Reuse(image.clone()));
        }

        let Some(generation) = self.generation.checked_add(1) else {
            return self.record_error(ImageResourceError::GenerationExhausted);
        };
        let request = ImageResourceRequest {
            owner: self.owner.clone(),
            generation,
            identity,
            descriptor,
        };
        self.generation = generation;
        self.active_request = Some(ActiveImageRequest::Known(request.clone()));
        self.fresh = false;
        self.diagnostic = None;
        Ok(ImageRequestDecision::Load(request))
    }

    pub fn request_encoded_png(
        &mut self,
        identity: ImageResourceIdentity,
        transform: [f64; 6],
        budget: EncodedPngBudget,
    ) -> Result<ImageRequestDecision<EncodedPngExpectation>, ImageResourceError> {
        if transform.iter().any(|component| !component.is_finite()) {
            return self.record_error(ImageResourceError::NonFiniteTransform);
        }
        let Some(generation) = self.generation.checked_add(1) else {
            return self.record_error(ImageResourceError::GenerationExhausted);
        };
        // Decoded leases do not retain a metadata-budget receipt. Reverify the encoded
        // input rather than claiming that a cached image satisfies a changed budget.
        let request = ImageResourceRequest {
            owner: self.owner.clone(),
            generation,
            identity,
            descriptor: EncodedPngExpectation { transform, budget },
        };
        self.generation = generation;
        self.active_request = Some(ActiveImageRequest::Encoded(request.clone()));
        self.fresh = false;
        self.diagnostic = None;
        Ok(ImageRequestDecision::Load(request))
    }

    pub fn accept_encoded_png(
        &mut self,
        request: &EncodedPngResourceRequest,
        resolved: ResolvedEncodedPngImage,
    ) -> Result<(), ImageResourceError> {
        self.require_active_encoded(request)?;
        // Only the capability-scoped resolver constructs this receipt, after decoding
        // with this exact request's budgets. A decoded lease alone cannot prove that.
        self.require_active_encoded(&resolved.request)?;
        let image = resolved.image;
        if image.identity() != request.identity() {
            return Err(ImageResourceError::IdentityMismatch);
        }
        if !matches!(
            image.origin(),
            CpuImageOrigin::EncodedPng {
                bit_depth: 8 | 16,
                channels: 3 | 4,
                ..
            }
        ) || !same_transform(&image.descriptor().transform, &request.transform())
        {
            return Err(ImageResourceError::DescriptorMismatch);
        }
        // The admitted image's private constructor establishes profile and layout.
        self.commit_image(image)
    }

    pub fn reject_encoded_png(
        &mut self,
        request: &EncodedPngResourceRequest,
        failure: ImageResourceFailure,
    ) -> Result<(), ImageResourceError> {
        self.require_active_encoded(request)?;
        self.fail_request(failure)
    }

    pub fn accept(
        &mut self,
        request: &ImageResourceRequest,
        image: AdmittedCpuImage,
    ) -> Result<(), ImageResourceError> {
        self.require_active(request)?;
        if image.identity() != request.identity() {
            return Err(ImageResourceError::IdentityMismatch);
        }
        if !same_descriptor(image.descriptor(), request.descriptor()) {
            return Err(ImageResourceError::DescriptorMismatch);
        }
        self.commit_image(image)
    }

    fn commit_image(&mut self, image: AdmittedCpuImage) -> Result<(), ImageResourceError> {
        let Some(revision) = self.revision.checked_add(1) else {
            return self.record_error(ImageResourceError::RevisionExhausted);
        };

        self.committed = Some(Arc::new(image));
        self.revision = revision;
        self.fresh = true;
        self.committed_revoked = false;
        self.active_request = None;
        self.diagnostic = None;
        Ok(())
    }

    pub fn reject(
        &mut self,
        request: &ImageResourceRequest,
        failure: ImageResourceFailure,
    ) -> Result<(), ImageResourceError> {
        self.require_active(request)?;
        self.fail_request(failure)
    }

    fn fail_request(&mut self, failure: ImageResourceFailure) -> Result<(), ImageResourceError> {
        self.active_request = None;
        let error = match failure {
            ImageResourceFailure::Denied => ImageResourceError::Denied,
            ImageResourceFailure::Cancelled => ImageResourceError::Cancelled,
            ImageResourceFailure::Failed(message) => ImageResourceError::Failed(message),
        };
        self.record_error(error)
    }

    pub fn revoke(&mut self, identity: &ImageResourceIdentity) -> Result<(), ImageResourceError> {
        let committed_matches = self
            .committed
            .as_ref()
            .is_some_and(|image| image.identity() == identity);
        let active_matches = self
            .active_request
            .as_ref()
            .is_some_and(|request| request.identity() == identity);
        if !committed_matches && !active_matches {
            return Err(ImageResourceError::UnknownResource);
        }
        if committed_matches {
            self.fresh = false;
            self.committed_revoked = true;
        }
        if active_matches {
            self.active_request = None;
        }
        self.diagnostic = Some(ImageResourceError::Revoked);
        Ok(())
    }

    pub fn committed(&self) -> Option<&Arc<AdmittedCpuImage>> {
        self.committed.as_ref()
    }

    pub fn active_request(&self) -> Option<&ImageResourceRequest> {
        match self.active_request.as_ref() {
            Some(ActiveImageRequest::Known(request)) => Some(request),
            _ => None,
        }
    }

    pub fn active_encoded_request(&self) -> Option<&EncodedPngResourceRequest> {
        match self.active_request.as_ref() {
            Some(ActiveImageRequest::Encoded(request)) => Some(request),
            _ => None,
        }
    }

    pub fn has_active_request(&self) -> bool {
        self.active_request.is_some()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn is_fresh(&self) -> bool {
        self.fresh
    }

    pub fn is_revoked(&self) -> bool {
        self.committed_revoked
    }

    pub fn diagnostic(&self) -> Option<&ImageResourceError> {
        self.diagnostic.as_ref()
    }

    fn require_active(&self, request: &ImageResourceRequest) -> Result<(), ImageResourceError> {
        if !Arc::ptr_eq(&self.owner, &request.owner) {
            return Err(ImageResourceError::ForeignRequest);
        }
        if !self.active_request.as_ref().is_some_and(|active| {
            matches!(active, ActiveImageRequest::Known(active)
                if active.generation == request.generation
                && active.identity == request.identity
                && same_descriptor(&active.descriptor, &request.descriptor))
        }) {
            return Err(ImageResourceError::StaleRequest);
        }
        Ok(())
    }

    fn require_active_encoded(
        &self,
        request: &EncodedPngResourceRequest,
    ) -> Result<(), ImageResourceError> {
        if !Arc::ptr_eq(&self.owner, &request.owner) {
            return Err(ImageResourceError::ForeignRequest);
        }
        if !self.active_request.as_ref().is_some_and(|active| {
            matches!(active, ActiveImageRequest::Encoded(active)
                if active.generation == request.generation
                && active.identity == request.identity
                && same_transform(&active.transform(), &request.transform())
                && active.budget().max_encoded_bytes == request.budget().max_encoded_bytes
                && active.budget().max_pixel_bytes == request.budget().max_pixel_bytes
                && active.budget().max_metadata_bytes == request.budget().max_metadata_bytes)
        }) {
            return Err(ImageResourceError::StaleRequest);
        }
        Ok(())
    }

    fn record_error<T>(&mut self, error: ImageResourceError) -> Result<T, ImageResourceError> {
        self.diagnostic = Some(error.clone());
        Err(error)
    }
}

fn same_descriptor(left: &CpuImageDescriptor, right: &CpuImageDescriptor) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.row_stride == right.row_stride
        && left.sample_format == right.sample_format
        && left.color_space == right.color_space
        && left.alpha == right.alpha
        && same_transform(&left.transform, &right.transform)
}

fn same_transform(left: &[f64; 6], right: &[f64; 6]) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use graphics_core::resources::admit_encoded_png;
    use graphics_core::resources::{
        AlphaAssociation, CpuImageBudget, CpuSampleFormat, ImageColorSpace, admit_cpu_image,
    };

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const RED_SHA256: [u8; 32] = [
        52, 170, 167, 70, 194, 90, 15, 16, 92, 67, 22, 187, 177, 240, 9, 170, 53, 159, 73, 88, 38,
        86, 238, 151, 215, 60, 88, 19, 45, 86, 52, 35,
    ];
    const BLUE_SHA256: [u8; 32] = [
        183, 209, 179, 161, 16, 76, 200, 107, 28, 234, 49, 7, 147, 207, 119, 112, 2, 219, 5, 23,
        40, 29, 19, 90, 2, 222, 7, 155, 14, 168, 124, 35,
    ];

    fn descriptor() -> CpuImageDescriptor {
        CpuImageDescriptor {
            width: 1,
            height: 1,
            row_stride: 4,
            sample_format: CpuSampleFormat::Rgba8,
            color_space: ImageColorSpace::Srgb,
            alpha: AlphaAssociation::Straight,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    fn identity(asset: &str, version: &str, digest: [u8; 32]) -> ImageResourceIdentity {
        ImageResourceIdentity {
            asset_id: asset.into(),
            version_id: version.into(),
            content_sha256: digest,
        }
    }

    fn image(
        identity: ImageResourceIdentity,
        descriptor: CpuImageDescriptor,
        pixels: [u8; 4],
    ) -> AdmittedCpuImage {
        admit_cpu_image(
            identity,
            descriptor,
            Arc::<[u8]>::from(pixels),
            CpuImageBudget {
                max_input_bytes: 4,
                max_decoded_bytes: 64,
            },
        )
        .expect("the exact hashed one-pixel fixture is admitted by the original core adapter")
    }

    fn request(
        state: &mut ImageResourceState,
        identity: ImageResourceIdentity,
        descriptor: CpuImageDescriptor,
    ) -> ImageResourceRequest {
        let ImageRequestDecision::Load(request) = state
            .request(identity, descriptor)
            .expect("request generation is available")
        else {
            panic!("this fixture requires a new request");
        };
        request
    }

    fn commit_red(state: &mut ImageResourceState) -> Arc<AdmittedCpuImage> {
        let identity = identity("asset", "red-v1", RED_SHA256);
        let request = request(state, identity.clone(), descriptor());
        state
            .accept(&request, image(identity, descriptor(), RED))
            .expect("the matching red completion is accepted");
        state.committed().expect("red is committed").clone()
    }

    fn assert_retained(
        state: &ImageResourceState,
        committed: &Arc<AdmittedCpuImage>,
        revision: u64,
        fresh: bool,
        diagnostic: Option<&ImageResourceError>,
    ) {
        let current = state
            .committed()
            .expect("the prior image remains available");
        assert!(Arc::ptr_eq(current, committed));
        assert_eq!(current.identity(), committed.identity());
        assert!(same_descriptor(
            current.descriptor(),
            committed.descriptor()
        ));
        assert_eq!(
            current.srgba8_pixels().collect::<Vec<_>>(),
            committed.srgba8_pixels().collect::<Vec<_>>()
        );
        assert_eq!(state.revision(), revision);
        assert_eq!(state.is_fresh(), fresh);
        assert_eq!(state.diagnostic(), diagnostic);
    }

    #[test]
    fn at_graphite_image_request_matching_completion_commits_whole_image() {
        let mut state = ImageResourceState::default();
        assert_eq!(state.revision(), 0);
        assert!(!state.is_fresh());
        assert!(state.committed().is_none());
        let committed = commit_red(&mut state);
        assert_eq!(committed.srgba8_pixels().collect::<Vec<_>>(), vec![RED]);
        assert_eq!(committed.image().width, 1);
        assert_eq!(committed.image().height, 1);
        assert_eq!(state.revision(), 1);
        assert!(state.is_fresh());
        assert!(state.active_request().is_none());
        assert!(state.diagnostic().is_none());
    }

    #[test]
    fn at_graphite_image_request_out_of_order_completions_preserve_active_generation() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        let next_identity = identity("asset", "blue-v2", BLUE_SHA256);
        let old = request(&mut state, next_identity.clone(), descriptor());
        let current = request(&mut state, next_identity.clone(), descriptor());
        assert!(current.generation() > old.generation());
        assert_eq!(
            state.accept(&old, image(next_identity.clone(), descriptor(), BLUE)),
            Err(ImageResourceError::StaleRequest)
        );
        assert_retained(&state, &committed, 1, false, None);
        assert_eq!(
            state.reject(&old, ImageResourceFailure::Denied),
            Err(ImageResourceError::StaleRequest)
        );
        assert_retained(&state, &committed, 1, false, None);
        assert_eq!(
            state
                .active_request()
                .expect("current request survives")
                .generation(),
            current.generation()
        );
        state
            .accept(&current, image(next_identity.clone(), descriptor(), BLUE))
            .expect("current generation recovers after the stale completion");
        assert_eq!(
            state.committed().expect("blue is committed").identity(),
            &next_identity
        );
        assert_eq!(
            state
                .committed()
                .expect("blue is committed")
                .srgba8_pixels()
                .collect::<Vec<_>>(),
            vec![BLUE]
        );
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert!(state.diagnostic().is_none());
        assert_eq!(
            state.accept(&current, image(next_identity, descriptor(), BLUE)),
            Err(ImageResourceError::StaleRequest)
        );
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
    }

    #[test]
    fn at_graphite_image_request_all_identity_components_are_required() {
        for foreign in [
            identity("another-asset", "blue-v2", BLUE_SHA256),
            identity("asset", "another-version", BLUE_SHA256),
            identity("asset", "blue-v2", RED_SHA256),
        ] {
            let mut state = ImageResourceState::default();
            let committed = commit_red(&mut state);
            let expected = identity("asset", "blue-v2", BLUE_SHA256);
            let request = request(&mut state, expected.clone(), descriptor());
            let pixels = if foreign.content_sha256 == RED_SHA256 {
                RED
            } else {
                BLUE
            };
            assert_eq!(
                state.accept(&request, image(foreign, descriptor(), pixels)),
                Err(ImageResourceError::IdentityMismatch)
            );
            assert_retained(&state, &committed, 1, false, None);
            state
                .accept(&request, image(expected, descriptor(), BLUE))
                .expect("foreign completion did not consume the active request");
            assert_eq!(state.revision(), 2);
            assert!(state.diagnostic().is_none());
        }
    }

    #[test]
    fn at_graphite_image_request_descriptor_is_exact_including_transform_bits() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        let identity = identity("asset", "blue-v2", BLUE_SHA256);
        let request = request(&mut state, identity.clone(), descriptor());
        for transform in [
            [1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [1.0, -0.0, 0.0, 1.0, 0.0, 0.0],
        ] {
            let changed = CpuImageDescriptor {
                transform,
                ..descriptor()
            };
            assert_eq!(
                state.accept(&request, image(identity.clone(), changed, BLUE)),
                Err(ImageResourceError::DescriptorMismatch)
            );
            assert_retained(&state, &committed, 1, false, None);
        }
        state
            .accept(&request, image(identity, descriptor(), BLUE))
            .expect("exact descriptor recovers");
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
    }

    #[test]
    fn at_graphite_image_request_denial_cancel_failure_and_recovery_are_atomic() {
        for (failure, expected) in [
            (ImageResourceFailure::Denied, ImageResourceError::Denied),
            (
                ImageResourceFailure::Cancelled,
                ImageResourceError::Cancelled,
            ),
            (
                ImageResourceFailure::Failed("decoder rejected the input".into()),
                ImageResourceError::Failed("decoder rejected the input".into()),
            ),
        ] {
            let mut state = ImageResourceState::default();
            let committed = commit_red(&mut state);
            let identity = identity("asset", "blue-v2", BLUE_SHA256);
            let failed = request(&mut state, identity.clone(), descriptor());
            assert_eq!(state.reject(&failed, failure), Err(expected.clone()));
            assert_retained(&state, &committed, 1, false, Some(&expected));
            assert_eq!(
                state.reject(&failed, ImageResourceFailure::Cancelled),
                Err(ImageResourceError::StaleRequest)
            );
            assert_retained(&state, &committed, 1, false, Some(&expected));
            assert!(state.active_request().is_none());
            let recovery = request(&mut state, identity.clone(), descriptor());
            state
                .accept(&recovery, image(identity, descriptor(), BLUE))
                .expect("the later request commits successfully");
            assert_eq!(state.revision(), 2);
            assert!(state.is_fresh());
            assert!(state.diagnostic().is_none());
        }
    }

    #[test]
    fn at_graphite_image_request_identical_verified_version_reuses_single_image() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        let ImageRequestDecision::Reuse(reused) = state
            .request(committed.identity().clone(), *committed.descriptor())
            .expect("identical version is reusable")
        else {
            panic!("the verified immutable image should be reused")
        };
        assert!(Arc::ptr_eq(&committed, &reused));
        assert!(Arc::ptr_eq(
            state.committed().expect("same committed image"),
            &reused
        ));
        assert_eq!(state.revision(), 1);
        assert!(state.is_fresh());
        assert!(state.active_request().is_none());
        assert!(state.diagnostic().is_none());
        let changed = CpuImageDescriptor {
            transform: [1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            ..descriptor()
        };
        let request = request(&mut state, committed.identity().clone(), changed);
        assert!(!state.is_fresh());
        assert!(Arc::ptr_eq(
            state.committed().expect("prior image is preserved"),
            &committed
        ));
        state
            .accept(&request, image(committed.identity().clone(), changed, RED))
            .expect("changed descriptor is a new commit");
        assert_eq!(state.revision(), 2);
        assert!(same_descriptor(
            state.committed().expect("transformed image").descriptor(),
            &changed
        ));
    }

    #[test]
    fn at_graphite_image_request_sessions_reject_foreign_tokens_with_same_generation() {
        let mut first = ImageResourceState::default();
        let committed = commit_red(&mut first);
        let mut second = ImageResourceState::default();
        commit_red(&mut second);
        let identity = identity("asset", "blue-v2", BLUE_SHA256);
        let first_request = request(&mut first, identity.clone(), descriptor());
        let second_request = request(&mut second, identity.clone(), descriptor());
        assert_eq!(first_request.generation(), second_request.generation());
        assert_eq!(
            first.accept(&second_request, image(identity.clone(), descriptor(), BLUE)),
            Err(ImageResourceError::ForeignRequest)
        );
        assert_retained(&first, &committed, 1, false, None);
        assert_eq!(
            first.reject(
                &second_request,
                ImageResourceFailure::Failed("foreign worker failure".into())
            ),
            Err(ImageResourceError::ForeignRequest)
        );
        assert_retained(&first, &committed, 1, false, None);
        assert_eq!(second.revision(), 1);
        assert!(second.diagnostic().is_none());
        first
            .accept(&first_request, image(identity, descriptor(), BLUE))
            .expect("own request is still active");
        assert_eq!(first.revision(), 2);
        assert_eq!(second.revision(), 1);
        assert!(second.active_request().is_some());
    }

    #[test]
    fn at_graphite_image_request_revocation_preserves_pixels_and_requires_readmission() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        state
            .revoke(committed.identity())
            .expect("committed resource can be revoked");
        assert_retained(
            &state,
            &committed,
            1,
            false,
            Some(&ImageResourceError::Revoked),
        );
        let revoked_request = request(&mut state, committed.identity().clone(), descriptor());
        state
            .revoke(committed.identity())
            .expect("active request can be revoked");
        assert!(state.active_request().is_none());
        assert_eq!(
            state.accept(
                &revoked_request,
                image(committed.identity().clone(), descriptor(), RED)
            ),
            Err(ImageResourceError::StaleRequest)
        );
        assert_retained(
            &state,
            &committed,
            1,
            false,
            Some(&ImageResourceError::Revoked),
        );
        let recovery = request(&mut state, committed.identity().clone(), descriptor());
        state
            .accept(
                &recovery,
                image(committed.identity().clone(), descriptor(), RED),
            )
            .expect("explicit readmission restores freshness");
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert!(state.diagnostic().is_none());
    }

    #[test]
    fn at_graphite_image_request_exhaustion_never_wraps_or_changes_committed_pixels() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        state.generation = u64::MAX;
        assert!(matches!(
            state.request(identity("asset", "blue-v2", BLUE_SHA256), descriptor()),
            Err(ImageResourceError::GenerationExhausted)
        ));
        assert_retained(
            &state,
            &committed,
            1,
            true,
            Some(&ImageResourceError::GenerationExhausted),
        );
        state.generation = 1;
        let identity = identity("asset", "blue-v2", BLUE_SHA256);
        let request = request(&mut state, identity.clone(), descriptor());
        state.revision = u64::MAX;
        assert_eq!(
            state.accept(&request, image(identity, descriptor(), BLUE)),
            Err(ImageResourceError::RevisionExhausted)
        );
        assert_retained(
            &state,
            &committed,
            u64::MAX,
            false,
            Some(&ImageResourceError::RevisionExhausted),
        );
    }
    #[test]
    fn at_graphite_image_request_invalid_expected_transform_never_reuses_or_starts_work() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = CpuImageDescriptor {
                transform: [1.0, 0.0, 0.0, 1.0, value, 0.0],
                ..descriptor()
            };
            assert!(matches!(
                state.request(committed.identity().clone(), invalid),
                Err(ImageResourceError::NonFiniteTransform)
            ));
            assert_retained(
                &state,
                &committed,
                1,
                true,
                Some(&ImageResourceError::NonFiniteTransform),
            );
            assert!(state.active_request().is_none());
            assert_eq!(state.generation, 1);
        }
        let ImageRequestDecision::Reuse(reused) = state
            .request(committed.identity().clone(), descriptor())
            .expect("a finite exact descriptor reuses the prior image")
        else {
            panic!("finite exact descriptor should reuse the prior image")
        };
        assert!(Arc::ptr_eq(&reused, &committed));
        assert!(state.diagnostic().is_none());
    }

    #[test]
    fn at_graphite_image_request_changed_token_metadata_never_matches_active_request() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        let identity = identity("asset", "blue-v2", BLUE_SHA256);
        let current = request(&mut state, identity.clone(), descriptor());
        let mut changed_generation = current.clone();
        changed_generation.generation += 1;
        let mut changed_identity = current.clone();
        changed_identity.identity.version_id = "foreign-version".into();
        let mut changed_descriptor = current.clone();
        changed_descriptor.descriptor.transform = [1.0, -0.0, 0.0, 1.0, 0.0, 0.0];
        for changed in [changed_generation, changed_identity, changed_descriptor] {
            assert_eq!(
                state.accept(&changed, image(identity.clone(), descriptor(), BLUE)),
                Err(ImageResourceError::StaleRequest)
            );
            assert_retained(&state, &committed, 1, false, None);
            assert_eq!(
                state
                    .active_request()
                    .expect("original request is active")
                    .generation(),
                current.generation()
            );
        }
        state
            .accept(&current, image(identity, descriptor(), BLUE))
            .expect("exact original token remains valid");
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
    }

    #[test]
    fn at_graphite_image_revocation_survives_request_and_rejection_until_valid_accept() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        assert!(!state.is_revoked());
        state
            .revoke(committed.identity())
            .expect("committed identity is revoked");
        assert!(state.is_revoked());
        let next_identity = identity("asset", "blue-v2", BLUE_SHA256);
        let rejected = request(&mut state, next_identity.clone(), descriptor());
        assert!(state.diagnostic().is_none());
        assert!(state.is_revoked());
        assert_eq!(
            state.reject(&rejected, ImageResourceFailure::Denied),
            Err(ImageResourceError::Denied)
        );
        assert!(state.is_revoked());
        let active = request(&mut state, next_identity.clone(), descriptor());
        assert_eq!(
            state.accept(&rejected, image(next_identity.clone(), descriptor(), BLUE)),
            Err(ImageResourceError::StaleRequest)
        );
        assert!(state.is_revoked());
        assert_eq!(
            state.accept(
                &active,
                image(committed.identity().clone(), descriptor(), RED)
            ),
            Err(ImageResourceError::IdentityMismatch)
        );
        assert!(state.is_revoked());
        assert_retained(&state, &committed, 1, false, None);
        state
            .accept(&active, image(next_identity, descriptor(), BLUE))
            .expect("only matching current completion clears committed revocation");
        assert!(!state.is_revoked());
        assert!(state.is_fresh());
        assert_eq!(state.revision(), 2);
    }

    #[test]
    fn at_graphite_image_revoking_pending_other_identity_preserves_valid_last_committed() {
        let mut state = ImageResourceState::default();
        let committed = commit_red(&mut state);
        let pending_identity = identity("asset", "blue-v2", BLUE_SHA256);
        let pending = request(&mut state, pending_identity.clone(), descriptor());
        state
            .revoke(&pending_identity)
            .expect("pending identity is revoked");
        assert!(!state.is_revoked());
        assert_retained(
            &state,
            &committed,
            1,
            false,
            Some(&ImageResourceError::Revoked),
        );
        assert_eq!(
            state.accept(&pending, image(pending_identity, descriptor(), BLUE)),
            Err(ImageResourceError::StaleRequest)
        );
        assert!(!state.is_revoked());
        assert_eq!(state.revision(), 1);
    }
    const HOST_PROFILE: &str = "53c3ea98-67d5-4d96-8817-c91236585024";
    const HOST_BYTES: &[u8] = b"graphite-input";
    const HOST_SHA256: &str = "66c57e04b1c79ef15d6684099dfb9eaa468f944069fe05fe48901022abdaff3a";

    pub(crate) struct HostReaderFixture {
        _directory: tempfile::TempDir,
        pub(crate) assets: SharedAssetService,
        input_path: std::path::PathBuf,
        input_identity: AssetIdentity,
        output_identity: AssetIdentity,
        pub(crate) input_authorization: AuthorizedCapabilities,
    }

    fn host_fixture_authorization(
        capabilities: sim_runtime::CapabilitySet,
    ) -> Result<AuthorizedCapabilities, sim_runtime::PermissionError> {
        let grant = sim_runtime::PermissionGrant::new(
            HOST_PROFILE,
            "graphite-host-reader-fixture",
            capabilities.clone(),
            "isolated-local-asset-test",
        )?;
        sim_runtime::PermissionPolicy::new(HOST_PROFILE, [grant])?
            .authorize("graphite-host-reader-fixture", &capabilities)
    }

    pub(crate) fn host_reader_fixture() -> Result<HostReaderFixture, Box<dyn std::error::Error>> {
        use sim_runtime::{
            AssetOperation, Capability, CapabilitySet,
            assets::{AssetNamespace, AssetRoots, AssetService},
        };
        let directory = tempfile::tempdir()?;
        let mut roots = Vec::new();
        for namespace in [
            AssetNamespace::Input,
            AssetNamespace::Output,
            AssetNamespace::Temporary,
        ] {
            let path = directory.path().join(namespace.locator_type());
            std::fs::create_dir(&path)?;
            roots.push((namespace, path));
        }
        let input_path = directory.path().join("input/image.bin");
        std::fs::write(&input_path, HOST_BYTES)?;
        std::fs::write(
            directory.path().join("output/image.bin"),
            b"graphite-output",
        )?;
        let mut service = AssetService::open(AssetRoots::new(HOST_PROFILE, roots)?)?;
        let scanning = host_fixture_authorization(CapabilitySet::new(
            [AssetNamespace::Input, AssetNamespace::Output]
                .into_iter()
                .map(|namespace| Capability::Asset {
                    namespace: namespace.locator_type().to_owned(),
                    action: AssetOperation::Read,
                }),
        ))?;
        service.scan_namespaces(
            &[AssetNamespace::Input, AssetNamespace::Output],
            &scanning,
            &CancellationToken::default(),
        )?;
        let input_identity = AssetIdentity::new(HOST_PROFILE, AssetNamespace::Input, "image.bin")?;
        let output_identity =
            AssetIdentity::new(HOST_PROFILE, AssetNamespace::Output, "image.bin")?;
        Ok(HostReaderFixture {
            _directory: directory,
            assets: Arc::new(std::sync::Mutex::new(service)),
            input_path,
            input_identity,
            output_identity,
            input_authorization: sim_runtime::authorize_native_input_reader(HOST_PROFILE)?,
        })
    }

    #[test]
    fn at_graphite_host_reader_returns_real_verified_input_bytes_and_indexed_record()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let before = fixture
            .assets
            .lock()
            .expect("fixture service")
            .record(&fixture.input_identity)
            .expect("actual scanned input record");
        let verified = read_verified_graphite_asset(
            &fixture.assets,
            &fixture.input_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            HOST_BYTES.len() as u64,
        )?;
        assert_eq!(verified.bytes(), HOST_BYTES);
        assert_eq!(verified.record(), &before);
        assert_eq!(verified.record().sha256, HOST_SHA256);
        assert_eq!(verified.record().byte_size, HOST_BYTES.len() as u64);
        assert_eq!(
            verified.record().identity.namespace,
            sim_runtime::assets::AssetNamespace::Input
        );
        assert_eq!(verified.record().identity.profile_id, HOST_PROFILE);
        let clone = verified.clone();
        assert!(Arc::ptr_eq(&verified.bytes, &clone.bytes));
        {
            let service = fixture
                .assets
                .try_lock()
                .expect("reader released the service before returning");
            assert_eq!(
                service.record(&fixture.input_identity),
                Some(before.clone())
            );
        }
        let repeated = read_verified_graphite_asset(
            &fixture.assets,
            &fixture.input_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            64,
        )?;
        assert_eq!(repeated.bytes(), verified.bytes());
        assert_eq!(repeated.record(), verified.record());
        Ok(())
    }

    #[test]
    fn at_graphite_host_reader_changed_digest_or_length_cannot_publish_old_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let verified = read_verified_graphite_asset(
            &fixture.assets,
            &fixture.input_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            64,
        )?;
        let modified = std::fs::metadata(&fixture.input_path)?.modified()?;
        for changed in [
            b"graphite-wrong".as_slice(),
            b"graphite-wrong-length".as_slice(),
        ] {
            std::fs::write(&fixture.input_path, changed)?;
            std::fs::File::options()
                .write(true)
                .open(&fixture.input_path)?
                .set_times(std::fs::FileTimes::new().set_modified(modified))?;
            assert!(matches!(
                read_verified_graphite_asset(
                    &fixture.assets, &fixture.input_identity, &fixture.input_authorization,
                    &CancellationToken::default(), 64,
                ),
                Err(GraphiteAssetReadError::Asset(AssetError::ChangedDuringRead(ref identity)))
                    if identity == &fixture.input_identity
            ));
            assert_eq!(verified.bytes(), HOST_BYTES);
            assert_eq!(verified.record().sha256, HOST_SHA256);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("service unchanged")
                    .record(&fixture.input_identity),
                Some(verified.record().clone())
            );
        }
        Ok(())
    }

    #[test]
    fn at_graphite_host_reader_respects_denial_and_does_not_relabel_output_as_input()
    -> Result<(), Box<dyn std::error::Error>> {
        use sim_runtime::assets::{AssetAction, AssetNamespace};
        let fixture = host_reader_fixture()?;
        let denied = host_fixture_authorization(sim_runtime::CapabilitySet::default())?;
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets,
                &fixture.input_identity,
                &denied,
                &CancellationToken::default(),
                64,
            ),
            Err(GraphiteAssetReadError::Asset(
                AssetError::PermissionDenied {
                    namespace: AssetNamespace::Input,
                    action: AssetAction::Read,
                }
            ))
        ));
        let output_record = fixture
            .assets
            .lock()
            .expect("fixture service")
            .record(&fixture.output_identity)
            .expect("real indexed Output");
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets,
                &fixture.output_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                64,
            ),
            Err(GraphiteAssetReadError::Asset(
                AssetError::PermissionDenied {
                    namespace: AssetNamespace::Output,
                    action: AssetAction::Read,
                }
            ))
        ));
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("service after denial")
                .record(&fixture.output_identity),
            Some(output_record)
        );
        Ok(())
    }

    #[test]
    fn at_graphite_host_reader_cap_and_cancellation_publish_nothing_and_recover()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let before = fixture
            .assets
            .lock()
            .expect("fixture service")
            .record(&fixture.input_identity);
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets,
                &fixture.input_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                0,
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::InvalidLimit))
        ));
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets, &fixture.input_identity, &fixture.input_authorization,
                &CancellationToken::default(), HOST_BYTES.len() as u64 - 1,
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::TooLarge { actual, limit }))
                if actual == HOST_BYTES.len() as u64 && limit == actual - 1
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets,
                &fixture.input_identity,
                &fixture.input_authorization,
                &cancelled,
                64,
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("service unchanged")
                .record(&fixture.input_identity),
            before
        );
        let recovered = read_verified_graphite_asset(
            &fixture.assets,
            &fixture.input_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            HOST_BYTES.len() as u64,
        )?;
        assert_eq!(recovered.bytes(), HOST_BYTES);
        assert_eq!(Some(recovered.record().clone()), before);
        Ok(())
    }

    #[test]
    fn at_graphite_host_reader_cancellation_after_actual_read_blocks_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let cancellation = CancellationToken::default();
        let (record, bytes) = {
            let service = fixture.assets.lock().expect("actual fixture service");
            let bytes = service.read_verified(
                &fixture.input_identity,
                &fixture.input_authorization,
                &cancellation,
                64,
            )?;
            let record = service
                .record(&fixture.input_identity)
                .expect("verified record under same lock");
            (record, bytes)
        };
        assert!(fixture.assets.try_lock().is_ok());
        assert!(cancellation.cancel());
        assert!(matches!(
            publish_verified_graphite_asset(record, bytes, &cancellation),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn at_graphite_host_reader_reports_unknown_assets_and_real_poisoned_service()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let unknown = AssetIdentity::new(
            HOST_PROFILE,
            sim_runtime::assets::AssetNamespace::Input,
            "unknown.bin",
        )?;
        assert!(matches!(read_verified_graphite_asset(
            &fixture.assets, &unknown, &fixture.input_authorization, &CancellationToken::default(), 64,
        ), Err(GraphiteAssetReadError::Asset(AssetError::UnknownAsset(ref identity))) if identity == &unknown));
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = fixture
                .assets
                .lock()
                .expect("genuine service lock before poisoning");
            panic!("exercise typed failure for the real poisoned service mutex");
        }));
        assert!(poisoned.is_err());
        assert!(matches!(
            read_verified_graphite_asset(
                &fixture.assets,
                &fixture.input_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                64,
            ),
            Err(GraphiteAssetReadError::ServicePoisoned)
        ));
        Ok(())
    }
    fn host_tensor_png(
        channels: u64,
        workflow: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use sim_tensor::{CpuWorkspaceAuthority, ImageTensor, StreamId};
        let (width, values) = match channels {
            3 => (2, vec![1., 0., 0., 0.25, 0.5, 0.75]),
            4 => (
                3,
                vec![1., 0., 0., 1., 0.25, 0.5, 0.75, 0.5, 0.25, 0.5, 0.75, 0.],
            ),
            _ => return Err("this real producer fixture requires three or four channels".into()),
        };
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let image = ImageTensor::from_f32(&backend, &context, 1, 1, width, channels, &values)?;
        let before = image.as_f32_slice()?.to_vec();
        let metadata =
            std::collections::BTreeMap::from([("workflow".to_owned(), workflow.to_owned())]);
        let encoded = sim_media::png::encode_png_frame_with_policy_and_context(
            &backend,
            &context,
            &before,
            1,
            1,
            width,
            channels,
            0,
            &metadata,
            sim_media::MetadataWritePolicy::default(),
            sim_media::png::PngLimits {
                max_input_bytes: 16 * 1024,
                max_dimension: 16,
                max_pixels: 16,
                max_decoder_allocation_bytes: 16 * 1024,
            },
        )?;
        assert_eq!(before, values);
        assert_eq!(image.as_f32_slice()?.to_vec(), before);
        assert_eq!(
            encoded.get(..8),
            Some(&[137, 80, 78, 71, 13, 10, 26, 10][..])
        );
        assert_eq!(encoded.get(24), Some(&8));
        assert_eq!(encoded.get(25), Some(&(if channels == 3 { 2 } else { 6 })));
        Ok(encoded)
    }

    #[test]
    fn at_graphite_host_png_interchange_uses_real_tensor_encoder_reader_edit_preview_and_export()
    -> Result<(), Box<dyn std::error::Error>> {
        use graphics_core::resources::{
            AlphaAssociation, CpuImageEdit, CpuImageOrigin, EncodedPngBudget, admit_encoded_png,
            evaluate_cpu_image_edit,
        };
        use sha2::{Digest, Sha256};
        const WORKFLOW: &str = "{\"nodes\":[],\"fixture\":\"graphite-interchange\"}";
        for channels in [3, 4] {
            let encoded = host_tensor_png(channels, WORKFLOW)?;
            let fixture = host_reader_fixture()?;
            let path = fixture.input_path.with_file_name("producer.png");
            std::fs::write(&path, &encoded)?;
            let identity = AssetIdentity::new(
                HOST_PROFILE,
                sim_runtime::assets::AssetNamespace::Input,
                "producer.png",
            )?;
            fixture
                .assets
                .lock()
                .expect("real fixture index")
                .scan_namespaces(
                    &[sim_runtime::assets::AssetNamespace::Input],
                    &fixture.input_authorization,
                    &CancellationToken::default(),
                )?;
            let verified = read_verified_graphite_asset(
                &fixture.assets,
                &identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                encoded.len() as u64,
            )?;
            assert_eq!(verified.bytes(), encoded);
            assert_eq!(std::fs::read(&path)?, encoded);
            assert_eq!(verified.record().identity, identity);
            assert_eq!(
                verified.record().sha256,
                format!("{:x}", Sha256::digest(&encoded))
            );
            assert_eq!(verified.record().byte_size, encoded.len() as u64);
            assert_eq!(verified.record().content_type, "image/png");
            assert_eq!(
                verified.record().metadata.workflow.as_deref(),
                Some(WORKFLOW)
            );
            assert_eq!(
                verified.record().metadata_carrier,
                sim_media::MetadataCarrier::Png
            );
            let record = verified.record().clone();
            // This fixture label is supplied by the caller; it does not establish a durable host version.
            let input_identity = ImageResourceIdentity {
                asset_id: record.identity.to_reference()?.into(),
                version_id: "test-supplied-png-version-1".into(),
                content_sha256: Sha256::digest(verified.bytes()).into(),
            };
            let input = admit_encoded_png(
                input_identity.clone(),
                [1., 0., 0., 1., 0., 0.],
                Arc::from(verified.bytes()),
                EncodedPngBudget {
                    max_encoded_bytes: encoded.len(),
                    max_pixel_bytes: 4096,
                    max_metadata_bytes: 4096,
                },
            )?;
            assert_eq!(input.identity(), &input_identity);
            assert_eq!(
                input.origin(),
                CpuImageOrigin::EncodedPng {
                    bytes: encoded.len(),
                    bit_depth: 8,
                    channels: channels as u8
                }
            );
            assert_eq!(
                input.origin().digest_representation(),
                "encoded-png-bytes-sha256"
            );
            assert_eq!(
                input.origin().decode_profile(),
                Some("graphite-static-png-rgb-rgba8-rgb-rgba16-srgb-v1")
            );
            assert_eq!(input.descriptor().alpha, AlphaAssociation::Straight);
            let expected = if channels == 3 {
                vec![255, 0, 0, 255, 63, 127, 191, 255]
            } else {
                vec![255, 0, 0, 255, 63, 127, 191, 127, 0, 0, 0, 0]
            };
            let (actual, width, height) = input.image().to_flat_u8();
            assert_eq!(actual, expected);
            assert_eq!((width, height), (if channels == 3 { 2 } else { 3 }, 1));
            if channels == 3 {
                assert!(
                    input
                        .image()
                        .data
                        .iter()
                        .all(|pixel| pixel.a().to_bits() == 1_f32.to_bits())
                );
            } else {
                assert_eq!(
                    input
                        .image()
                        .data
                        .last()
                        .expect("transparent original pixel")
                        .a()
                        .to_bits(),
                    0_f32.to_bits()
                );
                assert!(
                    input
                        .image()
                        .data
                        .last()
                        .expect("transparent original RGB retained before flattening")
                        .r()
                        > 0.
                );
            }
            let input_image = input.image().clone();
            let input_descriptor = *input.descriptor();
            for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
                let result = evaluate_cpu_image_edit(&input, operation, 4096)?;
                let result_image = result.image().clone();
                let attributes = result.attributes().clone();
                let pixels_sha256 = result.pixels_sha256();
                let original_png = result.image().to_png();
                let prepared =
                    graphics_core::export::prepare_cpu_result_png(&result, original_png.len())?;
                assert_eq!(prepared.encoded_bytes(), original_png);
                assert_eq!(
                    prepared.content_sha256(),
                    <[u8; 32]>::from(Sha256::digest(prepared.encoded_bytes()))
                );
                assert_eq!(prepared.input_identity(), &input_identity);
                assert_eq!(prepared.input_origin(), input.origin());
                assert_eq!(
                    prepared.input_digest_representation(),
                    "encoded-png-bytes-sha256"
                );
                assert_eq!(prepared.source_operation(), Some(operation));
                assert_eq!(prepared.source_pixels_sha256(), Some(pixels_sha256));
                assert_eq!(
                    prepared.source_revision(),
                    "28ed22824fae8828756f6f4163db62f3585f21e0"
                );
                assert_eq!((prepared.width(), prepared.height()), (width, height));
                let preview = crate::image_preview::prepare_cpu_result_preview(&result, 4096)?;
                let repeated = crate::image_preview::prepare_cpu_result_preview(&result, 4096)?;
                let (mut original_pixels, _, _) = result.image().to_flat_u8();
                for pixel in original_pixels.as_chunks_mut::<4>().0 {
                    pixel.swap(0, 2);
                }
                assert_eq!(preview.as_bytes(0), Some(original_pixels.as_slice()));
                assert_eq!(repeated.as_bytes(0), preview.as_bytes(0));
                let repeated_png =
                    graphics_core::export::prepare_cpu_result_png(&result, original_png.len())?;
                assert_eq!(repeated_png.encoded_bytes(), prepared.encoded_bytes());
                assert_eq!(result.image(), &result_image);
                assert_eq!(result.attributes(), &attributes);
                assert_eq!(result.pixels_sha256(), pixels_sha256);
                assert_eq!(input.image(), &input_image);
                assert_eq!(input.descriptor(), &input_descriptor);
            }
            assert_eq!(std::fs::read(&path)?, encoded);
            assert_eq!(verified.bytes(), encoded);
            assert_eq!(verified.record(), &record);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("unchanged host index")
                    .record(&identity),
                Some(record)
            );
        }
        Ok(())
    }

    #[test]
    fn at_graphite_host_png_interchange_rejects_changed_dependency_before_explicit_refresh()
    -> Result<(), Box<dyn std::error::Error>> {
        use graphics_core::resources::{
            EncodedPngBudget, ImageAdmissionError, PngAdmissionError, admit_encoded_png,
        };
        use sha2::{Digest, Sha256};
        let fixture = host_reader_fixture()?;
        let path = fixture.input_path.with_file_name("versioned.png");
        let identity = AssetIdentity::new(
            HOST_PROFILE,
            sim_runtime::assets::AssetNamespace::Input,
            "versioned.png",
        )?;
        let original = host_tensor_png(3, "{\"version\":1}")?;
        std::fs::write(&path, &original)?;
        fixture
            .assets
            .lock()
            .expect("real service")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let old = read_verified_graphite_asset(
            &fixture.assets,
            &identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            16 * 1024,
        )?;
        let before_record = old.record().clone();
        let old_identity = ImageResourceIdentity {
            asset_id: identity.to_reference()?.into(),
            version_id: "test-supplied-png-version-1".into(),
            content_sha256: Sha256::digest(old.bytes()).into(),
        };
        let replacement = host_tensor_png(4, "{\"version\":2}")?;
        std::fs::write(&path, &replacement)?;
        assert!(
            matches!(read_verified_graphite_asset(&fixture.assets, &identity, &fixture.input_authorization,
            &CancellationToken::default(), 16 * 1024),
            Err(GraphiteAssetReadError::Asset(AssetError::ChangedDuringRead(ref changed))) if changed == &identity)
        );
        assert!(matches!(
            admit_encoded_png(
                old_identity.clone(),
                [1., 0., 0., 1., 0., 0.],
                Arc::from(replacement.as_slice()),
                EncodedPngBudget {
                    max_encoded_bytes: 16 * 1024,
                    max_pixel_bytes: 4096,
                    max_metadata_bytes: 4096
                }
            ),
            Err(PngAdmissionError::Image(
                ImageAdmissionError::DigestMismatch
            ))
        ));
        assert_eq!(old.bytes(), original);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("old indexed version remains")
                .record(&identity),
            Some(before_record.clone())
        );
        fixture
            .assets
            .lock()
            .expect("explicit index refresh")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let current = read_verified_graphite_asset(
            &fixture.assets,
            &identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            16 * 1024,
        )?;
        assert_eq!(current.bytes(), replacement);
        assert_ne!(current.record().sha256, before_record.sha256);
        assert_eq!(
            current.record().metadata.workflow.as_deref(),
            Some("{\"version\":2}")
        );
        let current_identity = ImageResourceIdentity {
            asset_id: identity.to_reference()?.into(),
            version_id: "test-supplied-png-version-2".into(),
            content_sha256: Sha256::digest(current.bytes()).into(),
        };
        let input = admit_encoded_png(
            current_identity.clone(),
            [1., 0., 0., 1., 0., 0.],
            Arc::from(current.bytes()),
            EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        )?;
        assert_eq!(input.identity(), &current_identity);
        assert_eq!(old.bytes(), original);
        assert_eq!(old.record(), &before_record);
        assert_eq!(std::fs::read(&path)?, replacement);
        Ok(())
    }
    pub(crate) struct IndexedRequestPng {
        pub(crate) asset_identity: AssetIdentity,
        pub(crate) encoded: Vec<u8>,
        pub(crate) image_identity: ImageResourceIdentity,
        pub(crate) descriptor: CpuImageDescriptor,
    }

    fn request_png_budget() -> EncodedPngBudget {
        EncodedPngBudget {
            max_encoded_bytes: 16 * 1024,
            max_pixel_bytes: 4096,
            max_metadata_bytes: 4096,
        }
    }

    pub(crate) fn indexed_request_png(
        fixture: &HostReaderFixture,
        channels: u64,
    ) -> Result<IndexedRequestPng, Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let encoded = host_tensor_png(channels, "{\"fixture\":\"request-resolution\"}")?;
        let path = fixture.input_path.with_file_name("requested.png");
        std::fs::write(&path, &encoded)?;
        let asset_identity = AssetIdentity::new(
            HOST_PROFILE,
            sim_runtime::assets::AssetNamespace::Input,
            "requested.png",
        )?;
        fixture
            .assets
            .lock()
            .expect("actual fixture service")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let image_identity = ImageResourceIdentity {
            asset_id: asset_identity.to_reference()?.into(),
            version_id: "test-supplied-request-version-1".into(),
            content_sha256: Sha256::digest(&encoded).into(),
        };
        let oracle = admit_encoded_png(
            image_identity.clone(),
            [1., 0., 0., 1., 0., 0.],
            Arc::from(encoded.as_slice()),
            request_png_budget(),
        )?;
        Ok(IndexedRequestPng {
            asset_identity,
            encoded,
            image_identity,
            descriptor: *oracle.descriptor(),
        })
    }

    fn pending_png_request(
        state: &mut ImageResourceState,
        identity: ImageResourceIdentity,
        descriptor: CpuImageDescriptor,
    ) -> ImageResourceRequest {
        match state
            .request(identity, descriptor)
            .expect("public request admits this finite expected descriptor")
        {
            ImageRequestDecision::Load(request) => request,
            ImageRequestDecision::Reuse(_) => panic!("this control requires a new pending request"),
        }
    }

    #[test]
    fn at_graphite_png_request_resolution_commits_exact_host_png_then_original_result_preview()
    -> Result<(), Box<dyn std::error::Error>> {
        for channels in [3, 4] {
            let fixture = host_reader_fixture()?;
            let png = indexed_request_png(&fixture, channels)?;
            let mut state = ImageResourceState::default();
            let request =
                pending_png_request(&mut state, png.image_identity.clone(), png.descriptor);
            let decoded = resolve_graphite_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &request,
                request_png_budget(),
            )?;
            assert_eq!(decoded.identity(), request.identity());
            assert!(same_descriptor(decoded.descriptor(), request.descriptor()));
            assert_eq!(
                decoded.origin().digest_representation(),
                "encoded-png-bytes-sha256"
            );
            assert_eq!(state.revision(), 0);
            assert!(state.committed().is_none());
            assert_eq!(
                state
                    .active_request()
                    .expect("resolver has not published")
                    .generation(),
                request.generation()
            );
            state.accept(&request, decoded)?;
            assert_eq!(state.revision(), 1);
            assert!(state.is_fresh());
            assert!(state.active_request().is_none());
            let source = state.committed().expect("authoritative committed image");
            assert_eq!(source.identity(), &png.image_identity);
            let source_image = source.image().clone();
            for operation in [
                graphics_core::resources::CpuImageEdit::Invert,
                graphics_core::resources::CpuImageEdit::MakeOpaque,
            ] {
                let result =
                    graphics_core::resources::evaluate_cpu_image_edit(source, operation, 4096)?;
                let preview = crate::image_preview::prepare_cpu_result_preview(&result, 4096)?;
                let (mut original, _, _) = result.image().to_flat_u8();
                for pixel in original.as_chunks_mut::<4>().0 {
                    pixel.swap(0, 2);
                }
                assert_eq!(preview.as_bytes(0), Some(original.as_slice()));
                assert_eq!(result.input_identity(), &png.image_identity);
                assert_eq!(source.image(), &source_image);
            }
            assert_eq!(
                std::fs::read(fixture.input_path.with_file_name("requested.png"))?,
                png.encoded
            );
            assert_eq!(state.revision(), 1);
            assert_eq!(state.diagnostic(), None);
        }
        Ok(())
    }

    #[test]
    fn at_graphite_png_request_resolution_rejects_foreign_asset_digest_and_decoded_descriptor()
    -> Result<(), Box<dyn std::error::Error>> {
        use graphics_core::resources::ImageAdmissionError;
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut foreign_identity = png.image_identity.clone();
        foreign_identity.asset_id = fixture.input_identity.to_reference()?.into();
        let mut state = ImageResourceState::default();
        let request = pending_png_request(&mut state, foreign_identity, png.descriptor);
        assert!(matches!(
            resolve_graphite_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &request,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Resource(
                ImageResourceError::IdentityMismatch
            ))
        ));
        assert_eq!(state.revision(), 0);
        assert!(state.committed().is_none());
        assert_eq!(state.diagnostic(), None);
        let mut wrong_digest = png.image_identity.clone();
        wrong_digest.content_sha256[0] ^= 1;
        let request = pending_png_request(&mut state, wrong_digest, png.descriptor);
        assert!(matches!(
            resolve_graphite_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &request,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Admission(PngAdmissionError::Image(
                ImageAdmissionError::DigestMismatch
            )))
        ));
        for descriptor in [
            CpuImageDescriptor {
                width: png.descriptor.width + 1,
                ..png.descriptor
            },
            CpuImageDescriptor {
                row_stride: png.descriptor.row_stride + 1,
                ..png.descriptor
            },
            CpuImageDescriptor {
                sample_format: graphics_core::resources::CpuSampleFormat::Rgba8,
                ..png.descriptor
            },
            CpuImageDescriptor {
                color_space: graphics_core::resources::ImageColorSpace::Srgb,
                ..png.descriptor
            },
        ] {
            let request = pending_png_request(&mut state, png.image_identity.clone(), descriptor);
            assert!(matches!(
                resolve_graphite_png_request(
                    &fixture.assets,
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &CancellationToken::default(),
                    &request,
                    request_png_budget(),
                ),
                Err(GraphiteAssetReadError::Resource(
                    ImageResourceError::DescriptorMismatch
                ))
            ));
            assert_eq!(
                state
                    .active_request()
                    .expect("matching request remains active")
                    .generation(),
                request.generation()
            );
            assert_eq!(state.revision(), 0);
            assert!(state.committed().is_none());
            assert!(!state.is_fresh());
            assert_eq!(state.diagnostic(), None);
        }
        Ok(())
    }

    #[test]
    fn at_graphite_png_request_resolution_late_generation_owner_and_cancel_never_publish()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut state = ImageResourceState::default();
        let initial = pending_png_request(&mut state, png.image_identity.clone(), png.descriptor);
        let first = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &initial,
            request_png_budget(),
        )?;
        state.accept(&initial, first)?;
        let committed = state.committed().expect("prior complete tuple").clone();
        let mut next_identity = png.image_identity.clone();
        next_identity.version_id = "test-supplied-request-version-2".into();
        let obsolete = pending_png_request(&mut state, next_identity, png.descriptor);
        let obsolete_result = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &obsolete,
            request_png_budget(),
        )?;
        let mut current_identity = png.image_identity.clone();
        current_identity.version_id = "test-supplied-request-version-3".into();
        let active = pending_png_request(&mut state, current_identity, png.descriptor);
        assert_eq!(
            state.accept(&obsolete, obsolete_result),
            Err(ImageResourceError::StaleRequest)
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert!(matches!(
            resolve_graphite_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &cancellation,
                &active,
                request_png_budget()
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        let decoded = admit_encoded_png(
            active.identity().clone(),
            active.descriptor().transform,
            Arc::from(png.encoded.as_slice()),
            request_png_budget(),
        )?;
        let after_decode = CancellationToken::default();
        assert!(after_decode.cancel());
        assert!(matches!(
            finish_graphite_png_request(decoded, &active, &after_decode),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        assert!(Arc::ptr_eq(
            state.committed().expect("old tuple retained"),
            &committed
        ));
        assert_eq!(state.revision(), 1);
        assert!(!state.is_fresh());
        assert_eq!(
            state
                .active_request()
                .expect("latest active request")
                .generation(),
            active.generation()
        );
        assert_eq!(state.diagnostic(), None);
        let accepted = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &active,
            request_png_budget(),
        )?;
        state.accept(&active, accepted)?;
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        let committed = state.committed().expect("recovered tuple").clone();
        let mut foreign_state = ImageResourceState::default();
        let foreign = pending_png_request(
            &mut foreign_state,
            active.identity().clone(),
            png.descriptor,
        );
        let foreign_result = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &foreign,
            request_png_budget(),
        )?;
        assert_eq!(
            state.accept(&foreign, foreign_result),
            Err(ImageResourceError::ForeignRequest)
        );
        assert!(Arc::ptr_eq(
            state
                .committed()
                .expect("foreign request cannot replace tuple"),
            &committed
        ));
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert_eq!(state.diagnostic(), None);
        Ok(())
    }

    #[test]
    fn at_graphite_png_request_resolution_dependency_refresh_keeps_prior_commit_until_accept()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 3)?;
        let mut state = ImageResourceState::default();
        let initial = pending_png_request(&mut state, png.image_identity.clone(), png.descriptor);
        let first = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &initial,
            request_png_budget(),
        )?;
        state.accept(&initial, first)?;
        let committed = state.committed().expect("initial resolved PNG").clone();
        let replacement = host_tensor_png(4, "{\"dependency\":\"refreshed\"}")?;
        let path = fixture.input_path.with_file_name("requested.png");
        std::fs::write(&path, &replacement)?;
        let refreshed_identity = ImageResourceIdentity {
            asset_id: png.image_identity.asset_id.clone(),
            version_id: "test-supplied-request-refreshed-version".into(),
            content_sha256: Sha256::digest(&replacement).into(),
        };
        let oracle = admit_encoded_png(
            refreshed_identity.clone(),
            png.descriptor.transform,
            Arc::from(replacement.as_slice()),
            request_png_budget(),
        )?;
        let request =
            pending_png_request(&mut state, refreshed_identity.clone(), *oracle.descriptor());
        assert!(
            matches!(resolve_graphite_png_request(&fixture.assets, &png.asset_identity,
            &fixture.input_authorization, &CancellationToken::default(), &request, request_png_budget()),
            Err(GraphiteAssetReadError::Asset(AssetError::ChangedDuringRead(ref changed))) if changed == &png.asset_identity)
        );
        assert!(Arc::ptr_eq(
            state.committed().expect("prior valid image retained"),
            &committed
        ));
        assert_eq!(state.revision(), 1);
        assert!(!state.is_fresh());
        fixture
            .assets
            .lock()
            .expect("actual explicit dependency index refresh")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let current = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &request,
            request_png_budget(),
        )?;
        assert_eq!(current.identity(), &refreshed_identity);
        assert!(Arc::ptr_eq(
            state.committed().expect("decode is not publication"),
            &committed
        ));
        state.accept(&request, current)?;
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert_eq!(
            state.committed().expect("refreshed image").identity(),
            &refreshed_identity
        );
        assert_eq!(
            state
                .committed()
                .expect("RGBA refreshed result")
                .image()
                .width,
            3
        );
        assert_eq!(committed.identity(), &png.image_identity);
        assert_eq!(std::fs::read(&path)?, replacement);
        assert_eq!(state.diagnostic(), None);
        Ok(())
    }
    fn encoded_request(
        state: &mut ImageResourceState,
        png: &IndexedRequestPng,
    ) -> EncodedPngResourceRequest {
        let ImageRequestDecision::Load(request) = state
            .request_encoded_png(
                png.image_identity.clone(),
                png.descriptor.transform,
                request_png_budget(),
            )
            .expect("finite transform")
        else {
            panic!("encoded bytes must be verified")
        };
        request
    }

    fn resolve_encoded(
        fixture: &HostReaderFixture,
        png: &IndexedRequestPng,
        request: &EncodedPngResourceRequest,
    ) -> Result<ResolvedEncodedPngImage, GraphiteAssetReadError> {
        resolve_graphite_encoded_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            request,
        )
    }

    #[test]
    fn at_graphite_discovered_png_publishes_verified_shape_and_reverifies_cached_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        for channels in [3, 4] {
            let fixture = host_reader_fixture()?;
            let png = indexed_request_png(&fixture, channels)?;
            let mut state = ImageResourceState::default();
            let request = encoded_request(&mut state, &png);
            assert!(state.has_active_request());
            assert!(state.active_request().is_none());
            let decoded = resolve_encoded(&fixture, &png, &request)?;
            assert!(same_descriptor(
                decoded.image().descriptor(),
                &png.descriptor
            ));
            assert_eq!(decoded.image().identity(), &png.image_identity);
            state.accept_encoded_png(&request, decoded)?;
            assert_eq!(state.revision(), 1);
            let prior = state.committed().expect("first admission").clone();
            let next = encoded_request(&mut state, &png);
            assert!(next.generation() > request.generation());
            assert!(!state.is_fresh());
            assert!(Arc::ptr_eq(
                &prior,
                state.committed().expect("last valid lease")
            ));
            let decoded = resolve_encoded(&fixture, &png, &next)?;
            state.accept_encoded_png(&next, decoded)?;
            assert_eq!(state.revision(), 2);
            assert!(!state.has_active_request());
            assert!(state.is_fresh());
            assert_eq!(state.diagnostic(), None);
        }
        Ok(())
    }

    #[test]
    fn at_graphite_discovered_png_rejects_raw_rows_identity_transform_and_changed_expectation()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut state = ImageResourceState::default();
        let rows = identity("rows", "version", RED_SHA256);
        let ImageRequestDecision::Load(raw_request) = state.request_encoded_png(
            rows.clone(),
            descriptor().transform,
            request_png_budget(),
        )?
        else {
            panic!("pending raw substitution control")
        };
        assert_eq!(
            state.accept_encoded_png(
                &raw_request,
                ResolvedEncodedPngImage {
                    image: image(rows, descriptor(), RED),
                    request: raw_request.clone()
                }
            ),
            Err(ImageResourceError::DescriptorMismatch)
        );
        let request = encoded_request(&mut state, &png);
        for field in 0..4 {
            let mut changed = request.clone();
            match field {
                0 => changed.descriptor.transform[1] = -0.0,
                1 => changed.descriptor.budget.max_encoded_bytes -= 1,
                2 => changed.descriptor.budget.max_pixel_bytes -= 1,
                _ => changed.descriptor.budget.max_metadata_bytes -= 1,
            }
            assert_eq!(
                state.accept_encoded_png(&changed, resolve_encoded(&fixture, &png, &request)?),
                Err(ImageResourceError::StaleRequest)
            );
        }
        let wrong_transform = admit_encoded_png(
            png.image_identity.clone(),
            [1., -0.0, 0., 1., 0., 0.],
            Arc::from(png.encoded.as_slice()),
            request_png_budget(),
        )?;
        assert_eq!(
            state.accept_encoded_png(
                &request,
                ResolvedEncodedPngImage {
                    image: wrong_transform,
                    request: request.clone()
                }
            ),
            Err(ImageResourceError::DescriptorMismatch)
        );
        let mut wrong_version = png.image_identity.clone();
        wrong_version.version_id = "other-version".into();
        let wrong_identity = admit_encoded_png(
            wrong_version,
            request.transform(),
            Arc::from(png.encoded.as_slice()),
            request_png_budget(),
        )?;
        assert_eq!(
            state.accept_encoded_png(
                &request,
                ResolvedEncodedPngImage {
                    image: wrong_identity,
                    request: request.clone()
                }
            ),
            Err(ImageResourceError::IdentityMismatch)
        );
        assert_eq!(state.revision(), 0);
        assert!(state.committed().is_none());
        assert_eq!(
            state
                .active_encoded_request()
                .expect("valid request retained")
                .generation(),
            request.generation()
        );
        Ok(())
    }

    #[test]
    fn at_graphite_discovered_and_known_requests_share_one_generation_and_reject_foreign_owner()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut state = ImageResourceState::default();
        let known = pending_png_request(&mut state, png.image_identity.clone(), png.descriptor);
        let discovered = encoded_request(&mut state, &png);
        assert!(discovered.generation() > known.generation());
        let decoded = resolve_encoded(&fixture, &png, &discovered)?;
        assert_eq!(
            state.accept(&known, decoded.image),
            Err(ImageResourceError::StaleRequest)
        );
        let current = pending_png_request(&mut state, png.image_identity.clone(), png.descriptor);
        assert!(current.generation() > discovered.generation());
        assert_eq!(
            state.accept_encoded_png(&discovered, resolve_encoded(&fixture, &png, &discovered)?),
            Err(ImageResourceError::StaleRequest)
        );
        let mut foreign = ImageResourceState::default();
        let foreign_request = encoded_request(&mut foreign, &png);
        assert_eq!(
            state.accept_encoded_png(
                &foreign_request,
                resolve_encoded(&fixture, &png, &foreign_request)?
            ),
            Err(ImageResourceError::ForeignRequest)
        );
        assert_eq!(state.revision(), 0);
        assert!(state.active_encoded_request().is_none());
        assert_eq!(
            state
                .active_request()
                .expect("current known request")
                .generation(),
            current.generation()
        );
        let decoded = resolve_graphite_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &current,
            request_png_budget(),
        )?;
        state.accept(&current, decoded)?;
        assert_eq!(state.revision(), 1);
        Ok(())
    }

    #[test]
    fn at_graphite_discovered_png_cancel_revoke_failure_and_recovery_preserve_last_valid_lease()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut state = ImageResourceState::default();
        let first = encoded_request(&mut state, &png);
        state.accept_encoded_png(&first, resolve_encoded(&fixture, &png, &first)?)?;
        let lease = state.committed().expect("first complete source").clone();
        let request = encoded_request(&mut state, &png);
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert!(matches!(
            resolve_graphite_encoded_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &cancellation,
                &request
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        assert_eq!(
            state.reject_encoded_png(&request, ImageResourceFailure::Cancelled),
            Err(ImageResourceError::Cancelled)
        );
        assert_eq!(
            state.accept_encoded_png(&request, resolve_encoded(&fixture, &png, &request)?),
            Err(ImageResourceError::StaleRequest)
        );
        assert!(Arc::ptr_eq(
            &lease,
            state.committed().expect("preserved source")
        ));
        let request = encoded_request(&mut state, &png);
        state.revoke(&png.image_identity)?;
        assert!(state.is_revoked());
        assert!(!state.has_active_request());
        assert_eq!(
            state.accept_encoded_png(&request, resolve_encoded(&fixture, &png, &request)?),
            Err(ImageResourceError::StaleRequest)
        );
        let request = encoded_request(&mut state, &png);
        state.accept_encoded_png(&request, resolve_encoded(&fixture, &png, &request)?)?;
        assert!(state.is_fresh());
        assert!(!state.is_revoked());
        assert_eq!(state.revision(), 2);
        assert_eq!(lease.identity(), &png.image_identity);
        Ok(())
    }

    #[test]
    fn at_graphite_discovered_png_sealed_decode_budgets_and_nonfinite_transform_prevent_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let mut state = ImageResourceState::default();
        for budget in [
            EncodedPngBudget {
                max_encoded_bytes: 1,
                ..request_png_budget()
            },
            EncodedPngBudget {
                max_pixel_bytes: 1,
                ..request_png_budget()
            },
            EncodedPngBudget {
                max_metadata_bytes: 1,
                ..request_png_budget()
            },
        ] {
            let ImageRequestDecision::Load(request) = state.request_encoded_png(
                png.image_identity.clone(),
                png.descriptor.transform,
                budget,
            )?
            else {
                panic!("budget control must read encoded bytes")
            };
            assert!(resolve_encoded(&fixture, &png, &request).is_err());
            assert_eq!(state.revision(), 0);
            assert!(state.committed().is_none());
            assert!(state.has_active_request());
        }
        let prior = state
            .active_encoded_request()
            .expect("last pending budget request")
            .generation();
        let mut transform = png.descriptor.transform;
        transform[0] = f64::NAN;
        assert!(matches!(
            state.request_encoded_png(png.image_identity.clone(), transform, request_png_budget()),
            Err(ImageResourceError::NonFiniteTransform)
        ));
        assert_eq!(
            state
                .active_encoded_request()
                .expect("invalid input does not cancel pending work")
                .generation(),
            prior
        );
        Ok(())
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_loads_original_hashed_authenticated_png_without_source_changes()
     {
        use graphite_resource::{LoadResource, Resource};
        for channels in [3, 4] {
            let fixture = host_reader_fixture().expect("real indexed service");
            let png = indexed_request_png(&fixture, channels).expect("actual tensor PNG");
            let expected = Resource::new(png.encoded.clone()).hash();
            let before = fixture
                .assets
                .lock()
                .expect("fixture service")
                .record(&png.asset_identity)
                .expect("indexed PNG");
            let proxy = GraphiteAssetResourceProxy::new(
                fixture.assets.clone(),
                png.asset_identity.clone(),
                fixture.input_authorization.clone(),
                expected,
                CancellationToken::default(),
                png.encoded.len() as u64,
            );
            assert!(proxy.last_diagnostic().expect("diagnostic lock").is_none());
            let resource = proxy
                .load(expected)
                .await
                .expect("authenticated original resource");
            assert_eq!(resource.hash(), expected);
            assert_eq!(resource.as_ref(), png.encoded.as_slice());
            let repeated = proxy
                .load(expected)
                .await
                .expect("repeat authenticated read");
            assert_eq!(repeated.hash(), resource.hash());
            assert_eq!(repeated.as_ref(), resource.as_ref());
            assert!(proxy.last_diagnostic().expect("diagnostic lock").is_none());
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("fixture service")
                    .record(&png.asset_identity),
                Some(before)
            );
            assert_eq!(
                std::fs::read(fixture.input_path.with_file_name("requested.png"))
                    .expect("unchanged producer bytes"),
                png.encoded
            );
            assert_eq!(
                png.image_identity.version_id.as_ref(),
                "test-supplied-request-version-1"
            );
        }
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_distinguishes_requested_hash_and_authenticated_content_hash()
     {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let other = Resource::new(b"another original resource").hash();
        assert_ne!(expected, other);
        let proxy = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        assert!(proxy.load(other).await.is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::RequestedHashMismatch { expected: actual_expected, requested })
                if *actual_expected == expected && *requested == other)
        );
        let recovered = proxy
            .load(expected)
            .await
            .expect("correct requested hash recovers");
        assert_eq!(recovered.as_ref(), png.encoded.as_slice());
        assert!(
            proxy
                .last_diagnostic()
                .expect("success clears prior diagnostic")
                .is_none()
        );
        let wrong_binding = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            other,
            CancellationToken::default(),
            16 * 1024,
        );
        assert!(wrong_binding.load(other).await.is_none());
        assert!(
            matches!(wrong_binding.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::ContentHashMismatch { expected: actual_expected, actual })
                if *actual_expected == other && *actual == expected)
        );
        assert_eq!(recovered.hash(), expected);
        assert_eq!(recovered.as_ref(), png.encoded.as_slice());
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_keeps_denied_foreign_and_missing_reader_errors_visible() {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 3).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let denied = host_fixture_authorization(sim_runtime::CapabilitySet::default())
            .expect("real denial policy");
        let foreign = sim_runtime::authorize_native_input_reader("foreign-resource-profile")
            .expect("genuine grant for a different profile");
        for (authorization, is_foreign) in [(denied, false), (foreign, true)] {
            let proxy = GraphiteAssetResourceProxy::new(
                fixture.assets.clone(),
                png.asset_identity.clone(),
                authorization,
                expected,
                CancellationToken::default(),
                16 * 1024,
            );
            assert!(proxy.load(expected).await.is_none());
            let diagnostic = proxy
                .last_diagnostic()
                .expect("diagnostic lock")
                .expect("read failure retained");
            match diagnostic.as_ref() {
                GraphiteResourceLoadError::Read(error) if is_foreign => {
                    assert!(matches!(error.as_ref(),
                    GraphiteAssetReadError::Asset(AssetError::ProfileMismatch { actual, .. })
                        if actual == "foreign-resource-profile"))
                }
                GraphiteResourceLoadError::Read(error) => assert!(matches!(
                    error.as_ref(),
                    GraphiteAssetReadError::Asset(AssetError::PermissionDenied {
                        namespace: sim_runtime::assets::AssetNamespace::Input,
                        action: sim_runtime::assets::AssetAction::Read
                    })
                )),
                error => panic!("unexpected resource failure: {error}"),
            }
        }
        let unknown = AssetIdentity::new(
            HOST_PROFILE,
            sim_runtime::assets::AssetNamespace::Input,
            "unknown-resource.png",
        )
        .expect("exact missing identity");
        let missing = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            unknown.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        assert!(missing.load(expected).await.is_none());
        assert!(
            matches!(missing.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::UnknownAsset(identity)) if identity == &unknown))
        );
        let recovered = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        assert_eq!(
            recovered
                .load(expected)
                .await
                .expect("authorized new binding")
                .as_ref(),
            png.encoded.as_slice()
        );
        std::fs::remove_file(fixture.input_path.with_file_name("requested.png"))
            .expect("real indexed dependency removed");
        fixture
            .assets
            .lock()
            .expect("actual service")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )
            .expect("record actual dependency absence");
        assert!(recovered.load(expected).await.is_none());
        assert!(
            matches!(recovered.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::Missing(identity)) if identity == &png.asset_identity))
        );
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_rejects_changed_indexed_bytes_then_requires_new_hash_binding()
     {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 3).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let proxy = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        let old = proxy
            .load(expected)
            .await
            .expect("initial original resource");
        let before = fixture
            .assets
            .lock()
            .expect("service")
            .record(&png.asset_identity);
        let replacement =
            host_tensor_png(4, "{\"resource\":\"replacement\"}").expect("actual new PNG bytes");
        let current_hash = Resource::new(replacement.clone()).hash();
        assert_ne!(current_hash, expected);
        std::fs::write(
            fixture.input_path.with_file_name("requested.png"),
            &replacement,
        )
        .expect("real fixture dependency change");
        assert!(proxy.load(expected).await.is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::ChangedDuringRead(identity)) if identity == &png.asset_identity))
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("service")
                .record(&png.asset_identity),
            before
        );
        fixture
            .assets
            .lock()
            .expect("service")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )
            .expect("explicit actual index refresh");
        assert!(proxy.load(expected).await.is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::ContentHashMismatch { expected: actual_expected, actual })
                if *actual_expected == expected && *actual == current_hash)
        );
        let successor = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            current_hash,
            CancellationToken::default(),
            16 * 1024,
        );
        let current = successor
            .load(current_hash)
            .await
            .expect("new expected original hash binding");
        assert_eq!(current.as_ref(), replacement.as_slice());
        assert_eq!(current.hash(), current_hash);
        assert!(
            successor
                .last_diagnostic()
                .expect("diagnostic lock")
                .is_none()
        );
        assert_eq!(old.as_ref(), png.encoded.as_slice());
        assert_eq!(old.hash(), expected);
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_preserves_cancellation_and_encoded_byte_caps_with_recovery()
    {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let cancellation = CancellationToken::default();
        let cancelled = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            cancellation.clone(),
            16 * 1024,
        );
        assert!(cancellation.cancel());
        assert!(cancelled.load(expected).await.is_none());
        assert!(
            matches!(cancelled.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::Cancelled)))
        );
        let byte_count = png.encoded.len() as u64;
        for cap in [0, byte_count - 1] {
            let proxy = GraphiteAssetResourceProxy::new(
                fixture.assets.clone(),
                png.asset_identity.clone(),
                fixture.input_authorization.clone(),
                expected,
                CancellationToken::default(),
                cap,
            );
            assert!(proxy.load(expected).await.is_none());
            let diagnostic = proxy
                .last_diagnostic()
                .expect("diagnostic lock")
                .expect("cap failure retained");
            match diagnostic.as_ref() {
                GraphiteResourceLoadError::Read(error) if cap == 0 => assert!(matches!(
                    error.as_ref(),
                    GraphiteAssetReadError::Asset(AssetError::InvalidLimit)
                )),
                GraphiteResourceLoadError::Read(error) => assert!(matches!(error.as_ref(),
                    GraphiteAssetReadError::Asset(AssetError::TooLarge { actual, limit }) if *actual == byte_count && *limit == cap)),
                error => panic!("unexpected cap failure: {error}"),
            }
        }
        let recovered = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            byte_count,
        );
        let resource = recovered
            .load(expected)
            .await
            .expect("fresh token and exact byte cap");
        assert_eq!(resource.as_ref(), png.encoded.as_slice());
        assert!(
            recovered
                .last_diagnostic()
                .expect("diagnostic lock")
                .is_none()
        );
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_reports_real_service_and_diagnostic_lock_poisoning() {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let proxy = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = fixture.assets.lock().expect("actual service lock");
            panic!("exercise original reader poison boundary");
        }));
        assert!(poisoned.is_err());
        assert!(proxy.load(expected).await.is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::ServicePoisoned))
        );
        let fixture = host_reader_fixture().expect("independent unpoisoned real service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let proxy = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        );
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = proxy.diagnostic.lock().expect("actual diagnostic lock");
            panic!("exercise diagnostic poison boundary");
        }));
        assert!(poisoned.is_err());
        assert!(proxy.load(expected).await.is_none());
        assert!(matches!(
            proxy.last_diagnostic(),
            Err(GraphiteResourceLoadError::DiagnosticPoisoned)
        ));
        assert_eq!(
            std::fs::read(fixture.input_path.with_file_name("requested.png"))
                .expect("source bytes retained"),
            png.encoded
        );
    }

    #[gpui::test]
    async fn at_graphite_resource_proxy_rechecks_cancellation_at_authenticated_publication() {
        use graphite_resource::{LoadResource, Resource};
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let cancellation = CancellationToken::default();
        let proxy = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            cancellation.clone(),
            png.encoded.len() as u64,
        );
        let before = fixture
            .assets
            .lock()
            .expect("actual service")
            .record(&png.asset_identity);
        let authenticated = proxy
            .read_resource(expected)
            .expect("actual authenticated original resource before cancellation");
        assert_eq!(authenticated.hash(), expected);
        assert_eq!(authenticated.as_ref(), png.encoded.as_slice());
        assert!(cancellation.cancel());
        assert!(proxy.finish_load(Ok(authenticated)).is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("diagnostic lock").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::Cancelled)))
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("unchanged indexed service")
                .record(&png.asset_identity),
            before
        );
        assert_eq!(
            std::fs::read(fixture.input_path.with_file_name("requested.png"))
                .expect("producer file unchanged"),
            png.encoded
        );
        let recovered = GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            png.encoded.len() as u64,
        );
        let current = recovered
            .load(expected)
            .await
            .expect("fresh binding recovers through original trait");
        assert_eq!(current.hash(), expected);
        assert_eq!(current.as_ref(), png.encoded.as_slice());
        assert!(
            recovered
                .last_diagnostic()
                .expect("successful diagnostic clear")
                .is_none()
        );
        assert!(
            proxy
                .last_diagnostic()
                .expect("old cancelled binding retained")
                .is_some()
        );
    }

    #[gpui::test]
    async fn at_graphite_resource_editor_api_loads_through_original_application_io_without_gpu() {
        use graphite_graph_craft::application_io::ApplicationIo;
        use graphite_resource::Resource;
        for channels in [3, 4] {
            let fixture = host_reader_fixture().expect("real indexed service");
            let png = indexed_request_png(&fixture, channels).expect("actual tensor PNG");
            let expected = Resource::new(png.encoded.clone()).hash();
            let before = fixture
                .assets
                .lock()
                .expect("actual service")
                .record(&png.asset_identity);
            let proxy = Arc::new(GraphiteAssetResourceProxy::new(
                fixture.assets.clone(),
                png.asset_identity.clone(),
                fixture.input_authorization.clone(),
                expected,
                CancellationToken::default(),
                png.encoded.len() as u64,
            ));
            let api = graphite_resource_editor_api(proxy.clone());
            let application_io = api
                .application_io
                .as_ref()
                .expect("injected original application IO");
            assert!(application_io.gpu_executor().is_none());
            let resource = application_io
                .load_resource(expected)
                .await
                .expect("original application IO delegates exact authenticated read");
            assert_eq!(resource.hash(), expected);
            assert_eq!(resource.as_ref(), png.encoded.as_slice());
            assert!(application_io.gpu_executor().is_none());
            assert!(
                proxy
                    .last_diagnostic()
                    .expect("same caller-retained proxy")
                    .is_none()
            );
            assert_eq!(Arc::strong_count(&proxy), 2);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("unchanged indexed record")
                    .record(&png.asset_identity),
                before
            );
            assert_eq!(
                std::fs::read(fixture.input_path.with_file_name("requested.png"))
                    .expect("unchanged producer bytes"),
                png.encoded
            );
        }
    }

    #[gpui::test]
    async fn at_graphite_resource_editor_api_optional_fallback_keeps_same_proxy_denial_and_cancel_errors()
     {
        use graphite_graph_craft::application_io::ApplicationIo;
        use graphite_resource::Resource;
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let denied = host_fixture_authorization(sim_runtime::CapabilitySet::default())
            .expect("genuine denied grant");
        let proxy = Arc::new(GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            denied,
            expected,
            CancellationToken::default(),
            16 * 1024,
        ));
        let api = graphite_resource_editor_api(proxy.clone());
        assert!(
            api.application_io
                .as_ref()
                .expect("original IO")
                .load_resource(expected)
                .await
                .is_none()
        );
        let diagnostic = proxy
            .last_diagnostic()
            .expect("same retained proxy accessor")
            .expect("denied read retained");
        assert!(
            matches!(diagnostic.as_ref(), GraphiteResourceLoadError::Read(error) if matches!(error.as_ref(),
            GraphiteAssetReadError::Asset(AssetError::PermissionDenied { namespace: sim_runtime::assets::AssetNamespace::Input, action: sim_runtime::assets::AssetAction::Read })))
        );
        drop(api);
        assert!(Arc::ptr_eq(
            &diagnostic,
            &proxy
                .last_diagnostic()
                .expect("caller diagnostics survive API drop")
                .expect("same denied error")
        ));
        let cancellation = CancellationToken::default();
        let proxy = Arc::new(GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            cancellation.clone(),
            16 * 1024,
        ));
        let api = graphite_resource_editor_api(proxy.clone());
        let application_io = api.application_io.as_ref().expect("original IO");
        let resource = application_io
            .load_resource(expected)
            .await
            .expect("initial authorized read");
        assert_eq!(resource.as_ref(), png.encoded.as_slice());
        assert!(cancellation.cancel());
        assert!(application_io.load_resource(expected).await.is_none());
        assert!(
            matches!(proxy.last_diagnostic().expect("same proxy diagnostic").as_deref(),
            Some(GraphiteResourceLoadError::Read(error)) if matches!(error.as_ref(),
                GraphiteAssetReadError::Asset(AssetError::Cancelled)))
        );
        assert_eq!(resource.as_ref(), png.encoded.as_slice());
        assert!(application_io.gpu_executor().is_none());
    }

    #[gpui::test]
    async fn at_graphite_resource_editor_api_owns_only_supplied_proxy_and_releases_service_after_final_drop()
     {
        use graphite_graph_craft::application_io::ApplicationIo;
        use graphite_resource::Resource;
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 3).expect("actual tensor PNG");
        let expected = Resource::new(png.encoded.clone()).hash();
        let service = Arc::downgrade(&fixture.assets);
        let proxy = Arc::new(GraphiteAssetResourceProxy::new(
            fixture.assets.clone(),
            png.asset_identity.clone(),
            fixture.input_authorization.clone(),
            expected,
            CancellationToken::default(),
            16 * 1024,
        ));
        let weak_proxy = Arc::downgrade(&proxy);
        let api = graphite_resource_editor_api(proxy.clone());
        let weak_api = Arc::downgrade(&api);
        let shared_api = api.clone();
        let resource = api
            .application_io
            .as_ref()
            .expect("original IO")
            .load_resource(expected)
            .await
            .expect("actual authenticated bytes");
        assert_eq!(resource.as_ref(), png.encoded.as_slice());
        drop(fixture);
        drop(proxy);
        assert!(weak_proxy.upgrade().is_some());
        assert!(service.upgrade().is_some());
        drop(api);
        assert!(weak_api.upgrade().is_some());
        assert!(weak_proxy.upgrade().is_some());
        assert!(
            shared_api
                .application_io
                .as_ref()
                .expect("shared original IO")
                .gpu_executor()
                .is_none()
        );
        drop(shared_api);
        assert!(weak_api.upgrade().is_none());
        assert!(weak_proxy.upgrade().is_none());
        assert!(service.upgrade().is_none());
        assert_eq!(resource.as_ref(), png.encoded.as_slice());
        assert_eq!(resource.hash(), expected);
    }

    #[test]
    fn at_graphite_authenticated_png_resolvers_admit_original_network_pixels_and_exact_request_receipts()
    -> Result<(), Box<dyn std::error::Error>> {
        for channels in [3, 4] {
            let fixture = host_reader_fixture()?;
            let png = indexed_request_png(&fixture, channels)?;
            let before_record = fixture
                .assets
                .lock()
                .expect("real indexed service")
                .record(&png.asset_identity);
            let transform = [1., -0., 0., 1., -0., 0.];
            let oracle = admit_encoded_png(
                png.image_identity.clone(),
                transform,
                Arc::from(png.encoded.as_slice()),
                request_png_budget(),
            )?;
            let mut known_state = ImageResourceState::default();
            let known = pending_png_request(
                &mut known_state,
                png.image_identity.clone(),
                *oracle.descriptor(),
            );
            let image = resolve_graphite_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &known,
                request_png_budget(),
            )?;
            assert_eq!(image.image(), oracle.image());
            assert_eq!(image.origin(), oracle.origin());
            assert_eq!(image.identity(), &png.image_identity);
            assert!(same_descriptor(image.descriptor(), oracle.descriptor()));
            known_state.accept(&known, image)?;
            assert_eq!(known_state.revision(), 1);
            assert!(known_state.is_fresh());
            assert!(!known_state.has_active_request());
            assert_eq!(known_state.diagnostic(), None);
            let mut encoded_state = ImageResourceState::default();
            let ImageRequestDecision::Load(encoded) = encoded_state.request_encoded_png(
                png.image_identity.clone(),
                transform,
                request_png_budget(),
            )?
            else {
                panic!("new unknown-dimension owner must load");
            };
            let resolved = resolve_graphite_encoded_png_request(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &encoded,
            )?;
            assert_eq!(resolved.image().image(), oracle.image());
            assert_eq!(
                resolved.image().origin(),
                CpuImageOrigin::EncodedPng {
                    bytes: png.encoded.len(),
                    bit_depth: 8,
                    channels: channels as u8,
                }
            );
            assert_eq!(resolved.image().identity(), &png.image_identity);
            assert!(same_descriptor(
                resolved.image().descriptor(),
                oracle.descriptor()
            ));
            assert!(Arc::ptr_eq(&resolved.request.owner, &encoded.owner));
            assert_eq!(resolved.request.generation(), encoded.generation());
            assert_eq!(resolved.request.identity(), encoded.identity());
            assert!(same_transform(&resolved.request.transform(), &transform));
            assert_eq!(
                resolved.request.budget().max_encoded_bytes,
                encoded.budget().max_encoded_bytes
            );
            assert_eq!(
                resolved.request.budget().max_pixel_bytes,
                encoded.budget().max_pixel_bytes
            );
            assert_eq!(
                resolved.request.budget().max_metadata_bytes,
                encoded.budget().max_metadata_bytes
            );
            encoded_state.accept_encoded_png(&encoded, resolved)?;
            assert_eq!(encoded_state.revision(), 1);
            assert!(encoded_state.is_fresh());
            assert!(!encoded_state.has_active_request());
            assert_eq!(encoded_state.diagnostic(), None);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("unchanged indexed service")
                    .record(&png.asset_identity),
                before_record
            );
            assert_eq!(
                std::fs::read(fixture.input_path.with_file_name("requested.png"))?,
                png.encoded
            );
            assert_eq!(
                oracle.identity().version_id.as_ref(),
                "test-supplied-request-version-1"
            );
        }
        Ok(())
    }

    #[test]
    fn at_graphite_authenticated_png_original_network_decode_failure_preserves_commit_then_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 3)?;
        let mut state = ImageResourceState::default();
        let first = encoded_request(&mut state, &png);
        state.accept_encoded_png(&first, resolve_encoded(&fixture, &png, &first)?)?;
        let committed = state
            .committed()
            .expect("real producer admitted via original network")
            .clone();
        let source_before = committed.image().clone();
        let mut invalid_zlib = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut invalid_zlib, 3, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_chunk(png::chunk::IDAT, b"not zlib pixels")?;
            writer.finish()?;
        }
        let path = fixture.input_path.with_file_name("requested.png");
        std::fs::write(&path, &invalid_zlib)?;
        fixture
            .assets
            .lock()
            .expect("real index update")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let invalid_identity = ImageResourceIdentity {
            asset_id: png.image_identity.asset_id.clone(),
            version_id: "test-supplied-valid-framing-invalid-zlib".into(),
            content_sha256: Sha256::digest(&invalid_zlib).into(),
        };
        let ImageRequestDecision::Load(request) = state.request_encoded_png(
            invalid_identity,
            png.descriptor.transform,
            request_png_budget(),
        )?
        else {
            panic!("changed content requires actual encoded read");
        };
        let error = match resolve_graphite_encoded_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &request,
        ) {
            Ok(_) => panic!("original DecodeImage empty fallback must not be admitted"),
            Err(error) => error,
        };
        assert!(matches!(
            &error,
            GraphiteAssetReadError::Admission(PngAdmissionError::SourceDecodeFailed)
        ));
        assert!(std::error::Error::source(&error).is_some());
        assert!(Arc::ptr_eq(
            state.committed().expect("failed decode did not publish"),
            &committed
        ));
        assert_eq!(state.revision(), 1);
        assert!(!state.is_fresh());
        let message = error.to_string();
        assert_eq!(
            state.reject_encoded_png(&request, ImageResourceFailure::Failed(message.clone())),
            Err(ImageResourceError::Failed(message.clone()))
        );
        assert_eq!(
            state.diagnostic(),
            Some(&ImageResourceError::Failed(message))
        );
        assert!(!state.has_active_request());
        assert_eq!(committed.image(), &source_before);
        assert_eq!(std::fs::read(&path)?, invalid_zlib);
        std::fs::write(&path, &png.encoded)?;
        fixture
            .assets
            .lock()
            .expect("explicit dependency recovery")
            .scan_namespaces(
                &[sim_runtime::assets::AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let recovered_identity = ImageResourceIdentity {
            version_id: "test-supplied-network-recovered-version".into(),
            ..png.image_identity.clone()
        };
        let ImageRequestDecision::Load(recovery) = state.request_encoded_png(
            recovered_identity.clone(),
            png.descriptor.transform,
            request_png_budget(),
        )?
        else {
            panic!("explicit recovered version requires a new request");
        };
        let resolved = resolve_graphite_encoded_png_request(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &recovery,
        )?;
        assert_eq!(resolved.image().image(), &source_before);
        state.accept_encoded_png(&recovery, resolved)?;
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert!(!state.has_active_request());
        assert_eq!(state.diagnostic(), None);
        assert_eq!(
            state.committed().expect("recovered tuple").identity(),
            &recovered_identity
        );
        assert_eq!(committed.identity(), &png.image_identity);
        assert_eq!(committed.image(), &source_before);
        assert_eq!(std::fs::read(&path)?, png.encoded);
        let detail = "original compiler diagnostic";
        let network_error =
            GraphiteAssetReadError::from(PngNetworkAdmissionError::Compilation(detail.into()));
        assert!(
            matches!(&network_error, GraphiteAssetReadError::Network(error)
            if matches!(error.as_ref(), PngNetworkAdmissionError::Compilation(message) if message == detail))
        );
        assert_eq!(
            network_error.to_string(),
            format!("Original PNG graph compilation failed: {detail}")
        );
        assert!(std::error::Error::source(&network_error).is_some());
        Ok(())
    }

    #[test]
    fn at_graphite_encoded_source_seal_keeps_real_index_record_identity_and_header()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        for channels in [3, 4] {
            let fixture = host_reader_fixture()?;
            let png = indexed_request_png(&fixture, channels)?;
            let record = fixture
                .assets
                .lock()
                .expect("indexed producer")
                .record(&png.asset_identity)
                .expect("real scanned PNG record");
            let verified = prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &png.image_identity,
                request_png_budget(),
            )?;
            assert_eq!(verified.record(), &record);
            assert_eq!(verified.record().identity, png.asset_identity);
            assert_eq!(
                verified.record().byte_size,
                u64::try_from(png.encoded.len())?
            );
            assert_eq!(
                verified.record().sha256,
                format!("{:x}", Sha256::digest(&png.encoded))
            );
            assert_eq!(verified.source().identity(), &png.image_identity);
            assert_eq!(
                (verified.source().width(), verified.source().height()),
                (png.descriptor.width, png.descriptor.height)
            );
            let source = verified.source().clone();
            assert!(Arc::ptr_eq(verified.source(), &source));
            let weak = Arc::downgrade(&source);
            drop(verified);
            assert!(weak.upgrade().is_some());
            assert_eq!(source.identity(), &png.image_identity);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("unchanged record")
                    .record(&png.asset_identity),
                Some(record)
            );
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("canonical source read")
                    .read_verified(
                        &png.asset_identity,
                        &fixture.input_authorization,
                        &CancellationToken::default(),
                        16 * 1024,
                    )?,
                png.encoded
            );
            drop(source);
            assert!(weak.upgrade().is_none());
        }
        Ok(())
    }

    #[test]
    fn at_graphite_encoded_source_seal_requires_real_input_grant_namespace_and_index()
    -> Result<(), Box<dyn std::error::Error>> {
        use sim_runtime::assets::AssetAction;
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 3)?;
        let denied = host_fixture_authorization(sim_runtime::CapabilitySet::default())?;
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &denied,
                &CancellationToken::default(),
                &png.image_identity,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Asset(
                AssetError::PermissionDenied {
                    namespace: AssetNamespace::Input,
                    action: AssetAction::Read,
                }
            ))
        ));
        let foreign = sim_runtime::authorize_native_input_reader("foreign-graphite-seal-profile")?;
        assert!(matches!(prepare_graphite_encoded_png_source(
            &fixture.assets, &png.asset_identity, &foreign, &CancellationToken::default(),
            &png.image_identity, request_png_budget(),
        ), Err(GraphiteAssetReadError::Asset(AssetError::ProfileMismatch { expected, actual }))
            if expected == HOST_PROFILE && actual == "foreign-graphite-seal-profile"));
        let output_grant = host_fixture_authorization(sim_runtime::CapabilitySet::new([
            sim_runtime::Capability::Asset {
                namespace: AssetNamespace::Output.locator_type().to_owned(),
                action: sim_runtime::AssetOperation::Read,
            },
        ]))?;
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &fixture.output_identity,
                &output_grant,
                &CancellationToken::default(),
                &png.image_identity,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Asset(
                AssetError::UnsupportedNamespace(AssetNamespace::Output)
            ))
        ));
        let unknown = AssetIdentity::new(HOST_PROFILE, AssetNamespace::Input, "unknown.png")?;
        assert!(matches!(prepare_graphite_encoded_png_source(
            &fixture.assets, &unknown, &fixture.input_authorization, &CancellationToken::default(),
            &png.image_identity, request_png_budget(),
        ), Err(GraphiteAssetReadError::Asset(AssetError::UnknownAsset(identity))) if identity == unknown));
        let path = fixture.input_path.with_file_name("requested.png");
        std::fs::remove_file(&path)?;
        fixture
            .assets
            .lock()
            .expect("index genuine missing source")
            .scan_namespaces(
                &[AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        assert!(matches!(prepare_graphite_encoded_png_source(
            &fixture.assets, &png.asset_identity, &fixture.input_authorization, &CancellationToken::default(),
            &png.image_identity, request_png_budget(),
        ), Err(GraphiteAssetReadError::Asset(AssetError::Missing(identity))) if identity == png.asset_identity));
        std::fs::write(&path, &png.encoded)?;
        fixture
            .assets
            .lock()
            .expect("explicit source recovery")
            .scan_namespaces(
                &[AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let recovered = prepare_graphite_encoded_png_source(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &png.image_identity,
            request_png_budget(),
        )?;
        assert_eq!(recovered.source().identity(), &png.image_identity);
        assert_eq!(recovered.record().identity, png.asset_identity);
        Ok(())
    }

    #[test]
    fn at_graphite_encoded_source_seal_cancel_identity_digest_and_caps_never_return_source()
    -> Result<(), Box<dyn std::error::Error>> {
        use graphics_core::resources::ImageAdmissionError;
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 4)?;
        let before = fixture
            .assets
            .lock()
            .expect("before failures")
            .record(&png.asset_identity);
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &cancelled,
                &png.image_identity,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Asset(AssetError::Cancelled))
        ));
        let mut foreign = png.image_identity.clone();
        foreign.asset_id = fixture.input_identity.to_reference()?.into();
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &foreign,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Resource(
                ImageResourceError::IdentityMismatch
            ))
        ));
        let mut wrong = png.image_identity.clone();
        wrong.content_sha256[0] ^= 1;
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &wrong,
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Admission(PngAdmissionError::Image(
                ImageAdmissionError::DigestMismatch
            )))
        ));
        for maximum in [0, png.encoded.len() - 1] {
            let error = prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &png.image_identity,
                EncodedPngBudget {
                    max_encoded_bytes: maximum,
                    ..request_png_budget()
                },
            )
            .err()
            .expect("host cap rejects before a source is returned");
            if maximum == 0 {
                assert!(matches!(
                    error,
                    GraphiteAssetReadError::Asset(AssetError::InvalidLimit)
                ));
            } else {
                assert!(
                    matches!(error, GraphiteAssetReadError::Asset(AssetError::TooLarge { actual, limit })
                    if actual == png.encoded.len() as u64 && limit == maximum as u64)
                );
            }
        }
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &png.image_identity,
                EncodedPngBudget {
                    max_pixel_bytes: 0,
                    ..request_png_budget()
                },
            ),
            Err(GraphiteAssetReadError::Admission(PngAdmissionError::Image(
                ImageAdmissionError::DecodedBudgetExceeded
            )))
        ));
        assert!(matches!(
            prepare_graphite_encoded_png_source(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &png.image_identity,
                EncodedPngBudget {
                    max_metadata_bytes: 0,
                    ..request_png_budget()
                },
            ),
            Err(GraphiteAssetReadError::Admission(
                PngAdmissionError::MetadataBudgetExceeded
            ))
        ));
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("index unchanged by rejection")
                .record(&png.asset_identity),
            before
        );
        let recovered = prepare_graphite_encoded_png_source(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &png.image_identity,
            EncodedPngBudget {
                max_encoded_bytes: png.encoded.len(),
                ..request_png_budget()
            },
        )?;
        assert_eq!(recovered.source().identity(), &png.image_identity);
        assert_eq!(Some(recovered.record().clone()), before);
        Ok(())
    }

    #[test]
    fn at_graphite_encoded_source_seal_rejects_changed_file_and_preserves_prior_envelope()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 3)?;
        let first = prepare_graphite_encoded_png_source(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &png.image_identity,
            request_png_budget(),
        )?;
        let first_record = first.record().clone();
        let path = fixture.input_path.with_file_name("requested.png");
        let modified = std::fs::metadata(&path)?.modified()?;
        let replacement = host_tensor_png(3, "{\"fixture\":\"explicit-seal-successor\"}")?;
        std::fs::write(&path, &replacement)?;
        std::fs::File::options()
            .write(true)
            .open(&path)?
            .set_times(std::fs::FileTimes::new().set_modified(modified))?;
        assert!(matches!(prepare_graphite_encoded_png_source(
            &fixture.assets, &png.asset_identity, &fixture.input_authorization, &CancellationToken::default(),
            &png.image_identity, request_png_budget(),
        ), Err(GraphiteAssetReadError::Asset(AssetError::ChangedDuringRead(identity))) if identity == png.asset_identity));
        assert_eq!(first.record(), &first_record);
        assert_eq!(first.source().identity(), &png.image_identity);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("old index retained")
                .record(&png.asset_identity),
            Some(first_record.clone())
        );
        fixture
            .assets
            .lock()
            .expect("explicit reindex")
            .scan_namespaces(
                &[AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let successor = ImageResourceIdentity {
            version_id: "caller-supplied-seal-successor-label".into(),
            content_sha256: Sha256::digest(&replacement).into(),
            ..png.image_identity.clone()
        };
        let recovered = prepare_graphite_encoded_png_source(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &successor,
            request_png_budget(),
        )?;
        assert_eq!(recovered.source().identity(), &successor);
        assert_eq!(
            recovered.record().sha256,
            format!("{:x}", Sha256::digest(&replacement))
        );
        assert_ne!(recovered.record().sha256, first_record.sha256);
        assert!(!Arc::ptr_eq(first.source(), recovered.source()));
        assert_eq!(first.source().identity(), &png.image_identity);
        assert_eq!(first.record(), &first_record);
        Ok(())
    }

    #[test]
    fn at_graphite_encoded_source_seal_accepts_real_indexed_invalid_zlib_without_decode_claim()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture = host_reader_fixture()?;
        let png = indexed_request_png(&fixture, 3)?;
        let mut malformed = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut malformed, 3, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_chunk(png::chunk::IDAT, b"not zlib pixels")?;
            writer.finish()?;
        }
        let path = fixture.input_path.with_file_name("requested.png");
        std::fs::write(&path, &malformed)?;
        fixture
            .assets
            .lock()
            .expect("genuine index refresh")
            .scan_namespaces(
                &[AssetNamespace::Input],
                &fixture.input_authorization,
                &CancellationToken::default(),
            )?;
        let identity = ImageResourceIdentity {
            version_id: "caller-supplied-envelope-only-label".into(),
            content_sha256: Sha256::digest(&malformed).into(),
            ..png.image_identity.clone()
        };
        let envelope = prepare_graphite_encoded_png_source(
            &fixture.assets,
            &png.asset_identity,
            &fixture.input_authorization,
            &CancellationToken::default(),
            &identity,
            request_png_budget(),
        )?;
        assert_eq!(envelope.source().identity(), &identity);
        assert_eq!(
            (envelope.source().width(), envelope.source().height()),
            (3, 1)
        );
        assert_eq!(
            envelope.record().sha256,
            format!("{:x}", Sha256::digest(&malformed))
        );
        assert_eq!(envelope.record().byte_size, malformed.len() as u64);
        let before = envelope.record().clone();
        assert!(matches!(
            resolve_graphite_png_bytes(
                &fixture.assets,
                &png.asset_identity,
                &fixture.input_authorization,
                &CancellationToken::default(),
                &identity,
                [1., 0., 0., 1., 0., 0.],
                request_png_budget(),
            ),
            Err(GraphiteAssetReadError::Admission(
                PngAdmissionError::SourceDecodeFailed
            ))
        ));
        assert_eq!(envelope.source().identity(), &identity);
        assert_eq!(envelope.record(), &before);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("malformed encoded input unchanged")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &CancellationToken::default(),
                    16 * 1024,
                )?,
            malformed
        );
        Ok(())
    }
}
