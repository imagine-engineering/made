use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendWorkspaceLease, DType, DeviceId, ExecutionContext, TensorBackend, TensorError,
    decode_float8, encode_float8,
};
use sim_types::CancellationToken;
use std::collections::BTreeMap;
use thiserror::Error;

const MXFP8_GROUP_SIZE: usize = 32;
const NVFP4_GROUP_SIZE: usize = 16;
const MAX_QUANTIZATION_METADATA_BYTES: usize = 1024 * 1024;
const MAX_QUANTIZATION_LAYERS: usize = 4096;
const E4M3_MAX: f32 = 448.0;
const E5M2_MAX: f32 = 57_344.0;
const E2M1_MAX: f32 = 6.0;
const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationKind {
    Int8Tensorwise,
    #[serde(rename = "mxfp8")]
    MxFp8,
    #[serde(rename = "nvfp4")]
    NvFp4,
    MixedPerLayerV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantLinearLayout {
    TensorCoreFp8E4M3,
    TensorCoreFp8E5M2,
    TensorCoreMxFp8,
    TensorCoreNvFp4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantLinearScale {
    Default,
    Explicit(f32),
    Recalculate,
}

impl From<Option<f32>> for QuantLinearScale {
    fn from(scale: Option<f32>) -> Self {
        match scale {
            Some(scale) => Self::Explicit(scale),
            None => Self::Default,
        }
    }
}

impl QuantLinearLayout {
    pub const fn source_name(self) -> &'static str {
        match self {
            Self::TensorCoreFp8E4M3 => "TensorCoreFP8E4M3Layout",
            Self::TensorCoreFp8E5M2 => "TensorCoreFP8E5M2Layout",
            Self::TensorCoreMxFp8 => "TensorCoreMXFP8Layout",
            Self::TensorCoreNvFp4 => "TensorCoreNVFP4Layout",
        }
    }

    pub const fn is_fp8(self) -> bool {
        matches!(self, Self::TensorCoreFp8E4M3 | Self::TensorCoreFp8E5M2)
    }

    pub fn from_source_name(name: &str) -> Option<Self> {
        match name {
            "TensorCoreFP8Layout" | "TensorCoreFP8E4M3Layout" => Some(Self::TensorCoreFp8E4M3),
            "TensorCoreFP8E5M2Layout" => Some(Self::TensorCoreFp8E5M2),
            "TensorCoreMXFP8Layout" => Some(Self::TensorCoreMxFp8),
            "TensorCoreNVFP4Layout" => Some(Self::TensorCoreNvFp4),
            _ => None,
        }
    }
}

impl QuantizationKind {
    pub const fn feature_id(self) -> &'static str {
        match self {
            Self::Int8Tensorwise => "COMFY-MODEL-0155",
            Self::MxFp8 => "COMFY-MODEL-0156",
            Self::NvFp4 => "COMFY-MODEL-0157",
            Self::MixedPerLayerV1 => "COMFY-MODEL-0158",
        }
    }

    pub const fn group_size(self) -> Option<usize> {
        match self {
            Self::Int8Tensorwise | Self::MixedPerLayerV1 => None,
            Self::MxFp8 => Some(MXFP8_GROUP_SIZE),
            Self::NvFp4 => Some(NVFP4_GROUP_SIZE),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedMatrix {
    rows: usize,
    columns: usize,
    original_dtype: DType,
    storage: QuantizedStorage,
    source_identity: QuantizedSourceIdentity,
    content_identity: QuantizedContentIdentity,
}

#[derive(Clone, Debug, PartialEq)]
enum QuantizedStorage {
    Int8Tensorwise {
        values: Vec<i8>,
        scale: f32,
    },
    MxFp8 {
        values: Vec<u8>,
        block_scales: Vec<u8>,
        padded_columns: usize,
    },
    NvFp4 {
        packed_values: Vec<u8>,
        global_scale: f32,
        block_scales: Vec<u8>,
        padded_columns: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedLinearMatrix {
    rows: usize,
    columns: usize,
    original_dtype: DType,
    layout: QuantLinearLayout,
    storage: QuantizedLinearStorage,
    source_identity: QuantizedSourceIdentity,
    content_identity: QuantizedContentIdentity,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QuantizedSourceIdentity([u8; 32]);

impl QuantizedSourceIdentity {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_digest(self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QuantizedContentIdentity([u8; 32]);

impl QuantizedContentIdentity {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_digest(self.0)
    }
}

#[derive(Debug)]
pub struct QuantizedMaterialization {
    values: Vec<f32>,
    content_identity: QuantizedContentIdentity,
    _workspace: BackendWorkspaceLease,
}

impl QuantizedMaterialization {
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub const fn content_identity(&self) -> QuantizedContentIdentity {
        self.content_identity
    }
}

#[derive(Clone, Debug, PartialEq)]
enum QuantizedLinearStorage {
    Fp8Tensorwise {
        values: Vec<u8>,
        scale: f32,
        dtype: DType,
    },
    Catalog(QuantizedMatrix),
}

impl QuantizedLinearMatrix {
    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn original_dtype(&self) -> DType {
        self.original_dtype
    }

    pub const fn layout(&self) -> QuantLinearLayout {
        self.layout
    }

    pub const fn source_identity(&self) -> QuantizedSourceIdentity {
        self.source_identity
    }

    pub const fn content_identity(&self) -> QuantizedContentIdentity {
        self.content_identity
    }

    pub fn storage_bytes(&self) -> usize {
        match &self.storage {
            QuantizedLinearStorage::Fp8Tensorwise { values, scale, .. } => {
                values.len().saturating_add(std::mem::size_of_val(scale))
            }
            QuantizedLinearStorage::Catalog(matrix) => matrix.storage_bytes(),
        }
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, QuantizationError> {
        match &self.storage {
            QuantizedLinearStorage::Fp8Tensorwise { values, .. } => vec_resident_bytes(values),
            QuantizedLinearStorage::Catalog(matrix) => matrix.resident_storage_bytes(),
        }
    }

    pub fn dequantize(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, QuantizationError> {
        cancellation.check()?;
        match &self.storage {
            QuantizedLinearStorage::Fp8Tensorwise {
                values,
                scale,
                dtype,
            } => {
                let mut output = Vec::new();
                output.try_reserve_exact(values.len()).map_err(|_| {
                    QuantizationError::AllocationFailed {
                        requested: values.len(),
                    }
                })?;
                for (index, value) in values.iter().copied().enumerate() {
                    if index.is_multiple_of(1_024) {
                        cancellation.check()?;
                    }
                    output.push(decode_float8(*dtype, value) * *scale);
                }
                cancellation.check()?;
                Ok(output)
            }
            QuantizedLinearStorage::Catalog(matrix) => matrix.dequantize(cancellation),
        }
    }

    pub fn materialize(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantizedMaterialization, QuantizationError> {
        materialize_exact(
            self.rows,
            self.columns,
            self.content_identity,
            backend,
            context,
            |cancellation| self.dequantize(cancellation),
        )
    }
}

impl QuantizedMatrix {
    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn original_dtype(&self) -> DType {
        self.original_dtype
    }

    pub const fn kind(&self) -> QuantizationKind {
        match self.storage {
            QuantizedStorage::Int8Tensorwise { .. } => QuantizationKind::Int8Tensorwise,
            QuantizedStorage::MxFp8 { .. } => QuantizationKind::MxFp8,
            QuantizedStorage::NvFp4 { .. } => QuantizationKind::NvFp4,
        }
    }

    pub const fn source_identity(&self) -> QuantizedSourceIdentity {
        self.source_identity
    }

    pub const fn content_identity(&self) -> QuantizedContentIdentity {
        self.content_identity
    }

    pub fn storage_bytes(&self) -> usize {
        match &self.storage {
            QuantizedStorage::Int8Tensorwise { values, scale } => {
                values.len().saturating_add(std::mem::size_of_val(scale))
            }
            QuantizedStorage::MxFp8 {
                values,
                block_scales,
                ..
            } => values.len().saturating_add(block_scales.len()),
            QuantizedStorage::NvFp4 {
                packed_values,
                global_scale,
                block_scales,
                ..
            } => packed_values
                .len()
                .saturating_add(block_scales.len())
                .saturating_add(std::mem::size_of_val(global_scale)),
        }
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, QuantizationError> {
        match &self.storage {
            QuantizedStorage::Int8Tensorwise { values, .. } => vec_resident_bytes(values),
            QuantizedStorage::MxFp8 {
                values,
                block_scales,
                ..
            } => checked_resident_sum(
                vec_resident_bytes(values)?,
                vec_resident_bytes(block_scales)?,
            ),
            QuantizedStorage::NvFp4 {
                packed_values,
                block_scales,
                ..
            } => checked_resident_sum(
                vec_resident_bytes(packed_values)?,
                vec_resident_bytes(block_scales)?,
            ),
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_storage_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<[u8; 32], sim_types::CancellationError> {
        raw_quantized_storage_digest(&self.storage, cancellation)
    }

    pub fn dequantize(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, QuantizationError> {
        match &self.storage {
            QuantizedStorage::Int8Tensorwise { values, scale } => {
                cancellation.check()?;
                Ok(values
                    .iter()
                    .map(|value| f32::from(*value) * *scale)
                    .collect())
            }
            QuantizedStorage::MxFp8 {
                values,
                block_scales,
                padded_columns,
            } => dequantize_mxfp8(
                self.rows,
                self.columns,
                *padded_columns,
                values,
                block_scales,
                cancellation,
            ),
            QuantizedStorage::NvFp4 {
                packed_values,
                global_scale,
                block_scales,
                padded_columns,
            } => dequantize_nvfp4(
                self.rows,
                self.columns,
                *padded_columns,
                packed_values,
                *global_scale,
                block_scales,
                cancellation,
            ),
        }
    }

    pub fn materialize(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantizedMaterialization, QuantizationError> {
        materialize_exact(
            self.rows,
            self.columns,
            self.content_identity,
            backend,
            context,
            |cancellation| self.dequantize(cancellation),
        )
    }
}

fn raw_quantized_storage_digest(
    storage: &QuantizedStorage,
    cancellation: &CancellationToken,
) -> Result<[u8; 32], sim_types::CancellationError> {
    cancellation.check()?;
    let mut hasher = Sha256::new();
    hash_quantized_part(&mut hasher, b"zed.quantized-matrix.raw.v1");
    match storage {
        QuantizedStorage::Int8Tensorwise { values, scale } => {
            hash_quantized_part(&mut hasher, b"int8_tensorwise");
            hash_quantized_usize(&mut hasher, values.len());
            for (index, value) in values.iter().enumerate() {
                if index.is_multiple_of(1_024) {
                    cancellation.check()?;
                }
                hash_quantized_part(&mut hasher, &value.to_le_bytes());
            }
            hash_quantized_part(&mut hasher, &scale.to_bits().to_le_bytes());
        }
        QuantizedStorage::MxFp8 {
            values,
            block_scales,
            padded_columns,
        } => {
            hash_quantized_part(&mut hasher, b"mxfp8");
            hash_quantized_bytes_cancellable(&mut hasher, values, cancellation)?;
            hash_quantized_bytes_cancellable(&mut hasher, block_scales, cancellation)?;
            hash_quantized_usize(&mut hasher, *padded_columns);
        }
        QuantizedStorage::NvFp4 {
            packed_values,
            global_scale,
            block_scales,
            padded_columns,
        } => {
            hash_quantized_part(&mut hasher, b"nvfp4");
            hash_quantized_bytes_cancellable(&mut hasher, packed_values, cancellation)?;
            hash_quantized_part(&mut hasher, &global_scale.to_bits().to_le_bytes());
            hash_quantized_bytes_cancellable(&mut hasher, block_scales, cancellation)?;
            hash_quantized_usize(&mut hasher, *padded_columns);
        }
    }
    cancellation.check()?;
    Ok(hasher.finalize().into())
}

fn hash_quantized_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

fn hash_quantized_usize(hasher: &mut Sha256, value: usize) {
    hash_quantized_part(
        hasher,
        &u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes(),
    );
}

fn hash_quantized_bytes_cancellable(
    hasher: &mut Sha256,
    bytes: &[u8],
    cancellation: &CancellationToken,
) -> Result<(), sim_types::CancellationError> {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    for chunk in bytes.chunks(64 * 1024) {
        cancellation.check()?;
        hasher.update(chunk);
    }
    cancellation.check()?;
    Ok(())
}

fn source_identity(
    dtype: DType,
    values: &[f32],
    rows: usize,
    columns: usize,
    cancellation: &CancellationToken,
) -> Result<QuantizedSourceIdentity, QuantizationError> {
    cancellation.check()?;
    let mut hasher = Sha256::new();
    hash_quantized_part(&mut hasher, b"zed.quantized-source.v1");
    hash_quantized_part(&mut hasher, dtype_identity(dtype));
    hash_quantized_usize(&mut hasher, rows);
    hash_quantized_usize(&mut hasher, columns);
    hash_quantized_usize(&mut hasher, values.len());
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        hash_quantized_part(&mut hasher, &value.to_bits().to_le_bytes());
    }
    cancellation.check()?;
    Ok(QuantizedSourceIdentity(hasher.finalize().into()))
}

fn quantized_content_identity(
    source_identity: QuantizedSourceIdentity,
    dtype: DType,
    rows: usize,
    columns: usize,
    kind: QuantizationKind,
    storage: &QuantizedStorage,
    cancellation: &CancellationToken,
) -> Result<QuantizedContentIdentity, QuantizationError> {
    let storage_digest = raw_quantized_storage_digest(storage, cancellation)?;
    let mut hasher = Sha256::new();
    hash_quantized_part(&mut hasher, b"zed.quantized-content.v1");
    hash_quantized_part(&mut hasher, source_identity.as_bytes());
    hash_quantized_part(&mut hasher, dtype_identity(dtype));
    hash_quantized_usize(&mut hasher, rows);
    hash_quantized_usize(&mut hasher, columns);
    hash_quantized_part(&mut hasher, quantization_kind_identity(kind));
    hash_quantized_part(&mut hasher, &storage_digest);
    cancellation.check()?;
    Ok(QuantizedContentIdentity(hasher.finalize().into()))
}

fn quantized_linear_content_identity(
    source_identity: QuantizedSourceIdentity,
    dtype: DType,
    rows: usize,
    columns: usize,
    layout: QuantLinearLayout,
    storage: &QuantizedLinearStorage,
    cancellation: &CancellationToken,
) -> Result<QuantizedContentIdentity, QuantizationError> {
    cancellation.check()?;
    let mut hasher = Sha256::new();
    hash_quantized_part(&mut hasher, b"zed.quantized-linear-content.v1");
    hash_quantized_part(&mut hasher, source_identity.as_bytes());
    hash_quantized_part(&mut hasher, dtype_identity(dtype));
    hash_quantized_usize(&mut hasher, rows);
    hash_quantized_usize(&mut hasher, columns);
    hash_quantized_part(&mut hasher, quant_linear_layout_identity(layout));
    match storage {
        QuantizedLinearStorage::Fp8Tensorwise {
            values,
            scale,
            dtype,
        } => {
            hash_quantized_part(&mut hasher, b"fp8_tensorwise");
            hash_quantized_part(&mut hasher, dtype_identity(*dtype));
            hash_quantized_part(&mut hasher, &scale.to_bits().to_le_bytes());
            hash_quantized_bytes_cancellable(&mut hasher, values, cancellation)?;
        }
        QuantizedLinearStorage::Catalog(matrix) => {
            hash_quantized_part(&mut hasher, b"catalog");
            hash_quantized_part(&mut hasher, matrix.content_identity().as_bytes());
        }
    }
    cancellation.check()?;
    Ok(QuantizedContentIdentity(hasher.finalize().into()))
}

fn materialize_exact(
    rows: usize,
    columns: usize,
    content_identity: QuantizedContentIdentity,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
    decode: impl FnOnce(&CancellationToken) -> Result<Vec<f32>, QuantizationError>,
) -> Result<QuantizedMaterialization, QuantizationError> {
    context.cancellation.check()?;
    if backend.device() != DeviceId::CPU {
        return Err(QuantizationError::MaterializationUnsupportedDevice {
            device: backend.device(),
        });
    }
    let element_count = rows
        .checked_mul(columns)
        .ok_or(QuantizationError::ShapeOverflow)?;
    let requested = u64::try_from(element_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or(QuantizationError::ShapeOverflow)?;
    let workspace = backend
        .reserve_workspace(context, requested)
        .map_err(|error| map_materialization_error(error, requested, backend.device()))?;
    let values = decode(context.cancellation)?;
    if values.len() != element_count {
        return Err(QuantizationError::ValueCount {
            expected: element_count,
            actual: values.len(),
        });
    }
    context.cancellation.check()?;
    Ok(QuantizedMaterialization {
        values,
        content_identity,
        _workspace: workspace,
    })
}

fn map_materialization_error(
    error: TensorError,
    requested: u64,
    device: DeviceId,
) -> QuantizationError {
    match error {
        TensorError::AllocationFailed { .. }
        | TensorError::WorkspaceAuthorizationExceeded { .. } => {
            QuantizationError::MaterializationCapacity { requested }
        }
        TensorError::UnsupportedCapability { .. } | TensorError::NonHostDevice { .. } => {
            QuantizationError::MaterializationUnsupportedDevice { device }
        }
        TensorError::Cancelled => QuantizationError::Cancelled,
        error => QuantizationError::MaterializationBackend {
            reason: error.to_string(),
        },
    }
}

fn dtype_identity(dtype: DType) -> &'static [u8] {
    match dtype {
        DType::F64 => b"f64",
        DType::F32 => b"f32",
        DType::F16 => b"f16",
        DType::Bf16 => b"bf16",
        DType::I64 => b"i64",
        DType::I32 => b"i32",
        DType::I16 => b"i16",
        DType::I8 => b"i8",
        DType::U64 => b"u64",
        DType::U32 => b"u32",
        DType::U16 => b"u16",
        DType::U8 => b"u8",
        DType::Bool => b"bool",
        DType::Complex64 => b"complex64",
        DType::Complex128 => b"complex128",
        DType::Float8E4m3Fn => b"float8_e4m3fn",
        DType::Float8E5m2 => b"float8_e5m2",
        DType::Float8E4m3Fnuz => b"float8_e4m3fnuz",
        DType::Float8E5m2Fnuz => b"float8_e5m2fnuz",
        DType::Float8E8m0Fnu => b"float8_e8m0fnu",
    }
}

fn quantization_kind_identity(kind: QuantizationKind) -> &'static [u8] {
    match kind {
        QuantizationKind::Int8Tensorwise => b"int8_tensorwise",
        QuantizationKind::MxFp8 => b"mxfp8",
        QuantizationKind::NvFp4 => b"nvfp4",
        QuantizationKind::MixedPerLayerV1 => b"mixed_per_layer_v1",
    }
}

fn quant_linear_layout_identity(layout: QuantLinearLayout) -> &'static [u8] {
    match layout {
        QuantLinearLayout::TensorCoreFp8E4M3 => b"tensor_core_fp8_e4m3",
        QuantLinearLayout::TensorCoreFp8E5M2 => b"tensor_core_fp8_e5m2",
        QuantLinearLayout::TensorCoreMxFp8 => b"tensor_core_mxfp8",
        QuantLinearLayout::TensorCoreNvFp4 => b"tensor_core_nvfp4",
    }
}

fn hex_digest(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerQuantizationV1 {
    pub algorithm: QuantizationKind,
    pub original_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationMetadataV1 {
    pub version: u16,
    pub layers: BTreeMap<String, LayerQuantizationV1>,
}

impl QuantizationMetadataV1 {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, QuantizationError> {
        if bytes.len() > MAX_QUANTIZATION_METADATA_BYTES {
            return Err(QuantizationError::InvalidMetadata {
                reason: format!(
                    "quantization metadata exceeds the {MAX_QUANTIZATION_METADATA_BYTES}-byte limit"
                ),
            });
        }
        let metadata: Self =
            serde_json::from_slice(bytes).map_err(|error| QuantizationError::InvalidMetadata {
                reason: error.to_string(),
            })?;
        metadata.validate()?;
        Ok(metadata)
    }

    pub fn validate(&self) -> Result<(), QuantizationError> {
        if self.version != 1 {
            return Err(QuantizationError::InvalidMetadata {
                reason: format!("unsupported quantization metadata version {}", self.version),
            });
        }
        if self.layers.is_empty() {
            return Err(QuantizationError::InvalidMetadata {
                reason: "quantization metadata has no layers".to_owned(),
            });
        }
        if self.layers.len() > MAX_QUANTIZATION_LAYERS {
            return Err(QuantizationError::InvalidMetadata {
                reason: format!(
                    "quantization metadata exceeds the {MAX_QUANTIZATION_LAYERS}-layer limit"
                ),
            });
        }
        for (name, layer) in &self.layers {
            if name.is_empty() || name.len() > 1024 {
                return Err(QuantizationError::InvalidMetadata {
                    reason: "layer names must contain 1..=1024 bytes".to_owned(),
                });
            }
            if layer.algorithm == QuantizationKind::MixedPerLayerV1 {
                return Err(QuantizationError::InvalidMetadata {
                    reason: format!("layer {name} recursively selects mixed metadata"),
                });
            }
            if !matches!(layer.original_dtype, DType::F16 | DType::Bf16 | DType::F32) {
                return Err(QuantizationError::UnsupportedDType {
                    dtype: layer.original_dtype,
                });
            }
        }
        Ok(())
    }

    pub fn quantize_layer(
        &self,
        layer_name: &str,
        values: &[f32],
        rows: usize,
        columns: usize,
        cancellation: &CancellationToken,
    ) -> Result<QuantizedMatrix, QuantizationError> {
        self.validate()?;
        let layer = self
            .layers
            .get(layer_name)
            .ok_or_else(|| QuantizationError::MissingLayer {
                layer: layer_name.to_owned(),
            })?;
        quantize_matrix(
            layer.algorithm,
            layer.original_dtype,
            values,
            rows,
            columns,
            cancellation,
        )
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum QuantizationError {
    #[error("quantization matrix shape overflow")]
    ShapeOverflow,
    #[error("quantized resident byte accounting overflow")]
    ResidentBytesOverflow,
    #[error("quantization matrix expected {expected} values, got {actual}")]
    ValueCount { expected: usize, actual: usize },
    #[error("quantization input contains a non-finite value at index {index}")]
    NonFinite { index: usize },
    #[error("dtype {dtype:?} is unsupported for native quantization input")]
    UnsupportedDType { dtype: DType },
    #[error("mixed per-layer quantization requires versioned metadata")]
    MetadataRequired,
    #[error("invalid quantization metadata: {reason}")]
    InvalidMetadata { reason: String },
    #[error("quantization metadata does not contain layer {layer}")]
    MissingLayer { layer: String },
    #[error("a quantization scale is outside the representable native float8 range")]
    ScaleOutOfRange,
    #[error("quantization scale must be finite and greater than zero")]
    InvalidScale,
    #[error("quantization allocation for {requested} values failed")]
    AllocationFailed { requested: usize },
    #[error("quantized materialization is unsupported on {device:?}")]
    MaterializationUnsupportedDevice { device: DeviceId },
    #[error("quantized materialization requires {requested} bytes of caller-authorized capacity")]
    MaterializationCapacity { requested: u64 },
    #[error("quantized materialization backend failed: {reason}")]
    MaterializationBackend { reason: String },
    #[error("quantization was cancelled")]
    Cancelled,
}

fn vec_resident_bytes<T>(values: &Vec<T>) -> Result<u64, QuantizationError> {
    let bytes = values
        .capacity()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(QuantizationError::ResidentBytesOverflow)?;
    u64::try_from(bytes).map_err(|_| QuantizationError::ResidentBytesOverflow)
}

fn checked_resident_sum(left: u64, right: u64) -> Result<u64, QuantizationError> {
    left.checked_add(right)
        .ok_or(QuantizationError::ResidentBytesOverflow)
}

impl From<sim_types::CancellationError> for QuantizationError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn quantize_matrix(
    kind: QuantizationKind,
    original_dtype: DType,
    values: &[f32],
    rows: usize,
    columns: usize,
    cancellation: &CancellationToken,
) -> Result<QuantizedMatrix, QuantizationError> {
    cancellation.check()?;
    if !matches!(original_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(QuantizationError::UnsupportedDType {
            dtype: original_dtype,
        });
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(QuantizationError::ShapeOverflow)?;
    if values.len() != expected {
        return Err(QuantizationError::ValueCount {
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(QuantizationError::NonFinite { index });
    }
    let source_identity = source_identity(original_dtype, values, rows, columns, cancellation)?;
    let storage = match kind {
        QuantizationKind::Int8Tensorwise => quantize_int8(values, cancellation)?,
        QuantizationKind::MxFp8 => quantize_mxfp8(values, rows, columns, cancellation)?,
        QuantizationKind::NvFp4 => quantize_nvfp4(values, rows, columns, None, cancellation)?,
        QuantizationKind::MixedPerLayerV1 => return Err(QuantizationError::MetadataRequired),
    };
    let content_identity = quantized_content_identity(
        source_identity,
        original_dtype,
        rows,
        columns,
        kind,
        &storage,
        cancellation,
    )?;
    Ok(QuantizedMatrix {
        rows,
        columns,
        original_dtype,
        storage,
        source_identity,
        content_identity,
    })
}

pub fn quantize_linear_matrix(
    layout: QuantLinearLayout,
    original_dtype: DType,
    values: &[f32],
    rows: usize,
    columns: usize,
    scale: QuantLinearScale,
    cancellation: &CancellationToken,
) -> Result<QuantizedLinearMatrix, QuantizationError> {
    cancellation.check()?;
    validate_matrix_input(original_dtype, values, rows, columns)?;
    let source_identity = source_identity(original_dtype, values, rows, columns, cancellation)?;
    let storage = match layout {
        QuantLinearLayout::TensorCoreFp8E4M3 => quantize_fp8_linear_storage(
            values,
            original_dtype,
            DType::Float8E4m3Fn,
            E4M3_MAX,
            scale,
            cancellation,
        )?,
        QuantLinearLayout::TensorCoreFp8E5M2 => quantize_fp8_linear_storage(
            values,
            original_dtype,
            DType::Float8E5m2,
            E5M2_MAX,
            scale,
            cancellation,
        )?,
        QuantLinearLayout::TensorCoreMxFp8 => {
            let matrix_storage = quantize_mxfp8(values, rows, columns, cancellation)?;
            let content_identity = quantized_content_identity(
                source_identity,
                original_dtype,
                rows,
                columns,
                QuantizationKind::MxFp8,
                &matrix_storage,
                cancellation,
            )?;
            QuantizedLinearStorage::Catalog(QuantizedMatrix {
                rows,
                columns,
                original_dtype,
                storage: matrix_storage,
                source_identity,
                content_identity,
            })
        }
        QuantLinearLayout::TensorCoreNvFp4 => {
            let matrix_storage = quantize_nvfp4(
                values,
                rows,
                columns,
                match scale {
                    QuantLinearScale::Explicit(scale) => Some(scale),
                    QuantLinearScale::Default | QuantLinearScale::Recalculate => None,
                },
                cancellation,
            )?;
            let content_identity = quantized_content_identity(
                source_identity,
                original_dtype,
                rows,
                columns,
                QuantizationKind::NvFp4,
                &matrix_storage,
                cancellation,
            )?;
            QuantizedLinearStorage::Catalog(QuantizedMatrix {
                rows,
                columns,
                original_dtype,
                storage: matrix_storage,
                source_identity,
                content_identity,
            })
        }
    };
    cancellation.check()?;
    let content_identity = quantized_linear_content_identity(
        source_identity,
        original_dtype,
        rows,
        columns,
        layout,
        &storage,
        cancellation,
    )?;
    Ok(QuantizedLinearMatrix {
        rows,
        columns,
        original_dtype,
        layout,
        storage,
        source_identity,
        content_identity,
    })
}

fn validate_matrix_input(
    original_dtype: DType,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Result<(), QuantizationError> {
    if !matches!(original_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(QuantizationError::UnsupportedDType {
            dtype: original_dtype,
        });
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(QuantizationError::ShapeOverflow)?;
    if values.len() != expected {
        return Err(QuantizationError::ValueCount {
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(QuantizationError::NonFinite { index });
    }
    Ok(())
}

fn checked_scale(scale: f32) -> Result<f32, QuantizationError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(QuantizationError::InvalidScale);
    }
    Ok(scale)
}

fn resolve_fp8_scale(
    values: &[f32],
    original_dtype: DType,
    maximum: f32,
    scale: QuantLinearScale,
) -> Result<f32, QuantizationError> {
    let scale = match scale {
        QuantLinearScale::Default => 1.0,
        QuantLinearScale::Explicit(scale) => scale,
        QuantLinearScale::Recalculate => {
            let input_maximum = values
                .iter()
                .fold(0.0_f32, |current, value| current.max(value.abs()));
            let calculated = input_maximum / maximum;
            if original_dtype == DType::F16 {
                calculated.max(1.0 / 65_504.0)
            } else {
                calculated
            }
        }
    };
    checked_scale(scale)
}

fn quantize_fp8_linear_storage(
    values: &[f32],
    original_dtype: DType,
    dtype: DType,
    maximum: f32,
    requested_scale: QuantLinearScale,
    cancellation: &CancellationToken,
) -> Result<QuantizedLinearStorage, QuantizationError> {
    let scale = resolve_fp8_scale(values, original_dtype, maximum, requested_scale)?;
    Ok(QuantizedLinearStorage::Fp8Tensorwise {
        values: quantize_fp8_tensorwise(values, dtype, maximum, scale, cancellation)?,
        scale,
        dtype,
    })
}

fn quantize_fp8_tensorwise(
    values: &[f32],
    dtype: DType,
    maximum: f32,
    scale: f32,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, QuantizationError> {
    let scale = checked_scale(scale)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| QuantizationError::AllocationFailed {
            requested: values.len(),
        })?;
    for (index, value) in values.iter().copied().enumerate() {
        if index.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        output.push(
            encode_float8(dtype, (value / scale).clamp(-maximum, maximum))
                .ok_or(QuantizationError::ScaleOutOfRange)?,
        );
    }
    cancellation.check()?;
    Ok(output)
}

fn quantize_int8(
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<QuantizedStorage, QuantizationError> {
    let maximum = values
        .iter()
        .fold(0.0_f32, |current, value| current.max(value.abs()));
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
    let mut quantized = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        if index % 1024 == 0 {
            cancellation.check()?;
        }
        let rounded = round_ties_even(*value / scale).clamp(-127.0, 127.0);
        quantized.push(rounded as i8);
    }
    Ok(QuantizedStorage::Int8Tensorwise {
        values: quantized,
        scale,
    })
}

fn quantize_mxfp8(
    values: &[f32],
    rows: usize,
    columns: usize,
    cancellation: &CancellationToken,
) -> Result<QuantizedStorage, QuantizationError> {
    let padded_columns = checked_padded_columns(columns, MXFP8_GROUP_SIZE)?;
    let total = rows
        .checked_mul(padded_columns)
        .ok_or(QuantizationError::ShapeOverflow)?;
    let groups_per_row = padded_columns / MXFP8_GROUP_SIZE;
    let mut quantized = Vec::with_capacity(total);
    let mut block_scales = Vec::with_capacity(
        rows.checked_mul(groups_per_row)
            .ok_or(QuantizationError::ShapeOverflow)?,
    );
    for row in 0..rows {
        cancellation.check()?;
        for group in 0..groups_per_row {
            let start = group * MXFP8_GROUP_SIZE;
            let end = start.saturating_add(MXFP8_GROUP_SIZE).min(columns);
            let maximum = (start..end).fold(0.0_f32, |current, column| {
                current.max(values[row * columns + column].abs())
            });
            let scale = power_of_two_scale(maximum, E4M3_MAX);
            let scale_bits = encode_float8(DType::Float8E8m0Fnu, scale)
                .ok_or(QuantizationError::ScaleOutOfRange)?;
            let effective_scale = decode_float8(DType::Float8E8m0Fnu, scale_bits);
            block_scales.push(scale_bits);
            for column in start..start + MXFP8_GROUP_SIZE {
                let value = if column < columns {
                    values[row * columns + column]
                } else {
                    0.0
                };
                let normalized = (value / effective_scale).clamp(-E4M3_MAX, E4M3_MAX);
                quantized.push(
                    encode_float8(DType::Float8E4m3Fn, normalized)
                        .ok_or(QuantizationError::ScaleOutOfRange)?,
                );
            }
        }
    }
    Ok(QuantizedStorage::MxFp8 {
        values: quantized,
        block_scales,
        padded_columns,
    })
}

fn quantize_nvfp4(
    values: &[f32],
    rows: usize,
    columns: usize,
    requested_scale: Option<f32>,
    cancellation: &CancellationToken,
) -> Result<QuantizedStorage, QuantizationError> {
    let padded_columns = checked_padded_columns(columns, NVFP4_GROUP_SIZE)?;
    let total = rows
        .checked_mul(padded_columns)
        .ok_or(QuantizationError::ShapeOverflow)?;
    let maximum = values
        .iter()
        .fold(0.0_f32, |current, value| current.max(value.abs()));
    let global_scale = match requested_scale {
        Some(scale) => checked_scale(scale)?,
        None if maximum == 0.0 => 1.0,
        None => maximum / (E4M3_MAX * E2M1_MAX),
    };
    let groups_per_row = padded_columns / NVFP4_GROUP_SIZE;
    let mut nibbles = Vec::with_capacity(total);
    let mut block_scales = Vec::with_capacity(
        rows.checked_mul(groups_per_row)
            .ok_or(QuantizationError::ShapeOverflow)?,
    );
    for row in 0..rows {
        cancellation.check()?;
        for group in 0..groups_per_row {
            let start = group * NVFP4_GROUP_SIZE;
            let end = start.saturating_add(NVFP4_GROUP_SIZE).min(columns);
            let maximum = (start..end).fold(0.0_f32, |current, column| {
                current.max(values[row * columns + column].abs())
            });
            let block_scale = if maximum == 0.0 {
                1.0
            } else {
                (maximum / (global_scale * E2M1_MAX)).clamp(0.0, E4M3_MAX)
            };
            let block_scale_bits = encode_float8(DType::Float8E4m3Fn, block_scale)
                .ok_or(QuantizationError::ScaleOutOfRange)?;
            let effective_scale =
                global_scale * decode_float8(DType::Float8E4m3Fn, block_scale_bits);
            block_scales.push(block_scale_bits);
            for column in start..start + NVFP4_GROUP_SIZE {
                let value = if column < columns {
                    values[row * columns + column]
                } else {
                    0.0
                };
                nibbles.push(encode_e2m1(value / effective_scale));
            }
        }
    }
    let packed_values = nibbles
        .chunks(2)
        .map(|pair| pair[0] | pair.get(1).copied().unwrap_or(0) << 4)
        .collect();
    Ok(QuantizedStorage::NvFp4 {
        packed_values,
        global_scale,
        block_scales,
        padded_columns,
    })
}

fn dequantize_mxfp8(
    rows: usize,
    columns: usize,
    padded_columns: usize,
    values: &[u8],
    block_scales: &[u8],
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, QuantizationError> {
    let mut output = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or(QuantizationError::ShapeOverflow)?,
    );
    let groups_per_row = padded_columns / MXFP8_GROUP_SIZE;
    for row in 0..rows {
        cancellation.check()?;
        for column in 0..columns {
            let group = column / MXFP8_GROUP_SIZE;
            let scale_index = row * groups_per_row + group;
            let value_index = row * padded_columns + column;
            let scale = decode_float8(DType::Float8E8m0Fnu, block_scales[scale_index]);
            output.push(decode_float8(DType::Float8E4m3Fn, values[value_index]) * scale);
        }
    }
    Ok(output)
}

fn dequantize_nvfp4(
    rows: usize,
    columns: usize,
    padded_columns: usize,
    packed_values: &[u8],
    global_scale: f32,
    block_scales: &[u8],
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, QuantizationError> {
    let mut output = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or(QuantizationError::ShapeOverflow)?,
    );
    let groups_per_row = padded_columns / NVFP4_GROUP_SIZE;
    for row in 0..rows {
        cancellation.check()?;
        for column in 0..columns {
            let flat = row * padded_columns + column;
            let packed = packed_values[flat / 2];
            let nibble = if flat.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let group = column / NVFP4_GROUP_SIZE;
            let scale = global_scale
                * decode_float8(
                    DType::Float8E4m3Fn,
                    block_scales[row * groups_per_row + group],
                );
            output.push(decode_e2m1(nibble) * scale);
        }
    }
    Ok(output)
}

fn checked_padded_columns(columns: usize, group: usize) -> Result<usize, QuantizationError> {
    let remainder = columns % group;
    if remainder == 0 {
        Ok(columns)
    } else {
        columns
            .checked_add(group - remainder)
            .ok_or(QuantizationError::ShapeOverflow)
    }
}

fn power_of_two_scale(maximum: f32, format_maximum: f32) -> f32 {
    if maximum == 0.0 {
        1.0
    } else {
        let minimum = 2.0_f32.powi(-127);
        (maximum / format_maximum).max(minimum).log2().ceil().exp2()
    }
}

fn encode_e2m1(value: f32) -> u8 {
    let magnitude = value.abs();
    let mut best_index = 0_usize;
    let mut best_distance = f32::INFINITY;
    for (index, candidate) in E2M1_VALUES.iter().enumerate() {
        let distance = (magnitude - candidate).abs();
        if distance < best_distance
            || (distance == best_distance
                && index.is_multiple_of(2)
                && !best_index.is_multiple_of(2))
        {
            best_index = index;
            best_distance = distance;
        }
    }
    u8::try_from(best_index).unwrap_or(0) | if value.is_sign_negative() { 0x08 } else { 0 }
}

fn decode_e2m1(bits: u8) -> f32 {
    let magnitude = E2M1_VALUES[usize::from(bits & 0x07)];
    if bits & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn round_ties_even(value: f32) -> f32 {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 {
        lower
    } else if fraction > 0.5 {
        lower + 1.0
    } else if (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_catalog_contracts_are_explicit_and_checked() {
        let token = CancellationToken::default();
        let values = [-6.0, -1.0, 0.0, 1.0, 6.0, 12.0];
        for kind in [
            QuantizationKind::Int8Tensorwise,
            QuantizationKind::MxFp8,
            QuantizationKind::NvFp4,
        ] {
            let quantized = quantize_matrix(kind, DType::F32, &values, 2, 3, &token)
                .expect("catalog quantization is supported");
            let decoded = quantized
                .dequantize(&token)
                .expect("catalog dequantization is supported");
            assert_eq!(decoded.len(), values.len());
            assert!(decoded.iter().all(|value| value.is_finite()));
        }
        let metadata = QuantizationMetadataV1::parse_json(
            br#"{"version":1,"layers":{"weight":{"algorithm":"int8_tensorwise","original_dtype":"f32"}}}"#,
        )
        .expect("metadata is valid");
        assert_eq!(
            metadata
                .quantize_layer("weight", &values, 2, 3, &token)
                .expect("mixed layer quantizes")
                .kind(),
            QuantizationKind::Int8Tensorwise
        );
    }

    #[test]
    fn mixed_metadata_is_bounded_before_and_after_decode() {
        let oversized = vec![b' '; MAX_QUANTIZATION_METADATA_BYTES + 1];
        assert!(matches!(
            QuantizationMetadataV1::parse_json(&oversized),
            Err(QuantizationError::InvalidMetadata { .. })
        ));

        let layer = LayerQuantizationV1 {
            algorithm: QuantizationKind::Int8Tensorwise,
            original_dtype: DType::F32,
        };
        let layers = (0..=MAX_QUANTIZATION_LAYERS)
            .map(|index| (format!("layer-{index}"), layer.clone()))
            .collect();
        assert!(matches!(
            (QuantizationMetadataV1 { version: 1, layers }).validate(),
            Err(QuantizationError::InvalidMetadata { .. })
        ));
    }

    #[test]
    fn materialization_errors_preserve_unsupported_device_and_capacity_kinds() {
        let cuda = DeviceId::from_source_device("cuda:0").expect("valid test device");
        assert!(matches!(
            map_materialization_error(
                TensorError::UnsupportedCapability {
                    operation: "zed.quantized.materialize".to_owned(),
                    device: cuda,
                    reason: "test backend has no host materializer".to_owned(),
                },
                64,
                cuda,
            ),
            QuantizationError::MaterializationUnsupportedDevice { device } if device == cuda
        ));
        assert!(matches!(
            map_materialization_error(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested: 64,
                    authorized: 32,
                    in_use: 0,
                },
                64,
                DeviceId::CPU,
            ),
            QuantizationError::MaterializationCapacity { requested: 64 }
        ));
    }

    #[test]
    fn resident_storage_bytes_counts_allocated_capacity_not_semantic_length()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let mut matrix = quantize_matrix(
            QuantizationKind::Int8Tensorwise,
            DType::F32,
            &[0.0, 1.0, 2.0, 3.0],
            2,
            2,
            &cancellation,
        )?;
        let semantic_bytes = matrix.storage_bytes();
        let capacity = if let QuantizedStorage::Int8Tensorwise { values, .. } = &mut matrix.storage
        {
            values.reserve_exact(64);
            values.capacity()
        } else {
            return Err("test quantization kind changed".into());
        };
        assert_eq!(matrix.storage_bytes(), semantic_bytes);
        assert_eq!(matrix.resident_storage_bytes()?, u64::try_from(capacity)?);
        assert!(matrix.resident_storage_bytes()? > u64::try_from(semantic_bytes)?);
        Ok(())
    }
}
