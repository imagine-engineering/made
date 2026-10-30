use serde::{Deserialize, Serialize};
use sim_types::DeviceKind;
use std::{
    any::Any,
    fmt,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;

pub mod autograd;
#[cfg(feature = "cpu")]
pub mod cpu_backend;
pub mod dtypes;
#[cfg(feature = "cpu")]
pub mod image_ops;
#[cfg(feature = "cpu")]
pub mod latent_composite;
pub mod native_node_payload;
pub mod operation;
pub mod operation_contracts;
pub mod promotion;
pub mod rng;
pub mod shader;

pub use autograd::*;
#[cfg(feature = "cpu")]
pub use cpu_backend::*;
pub use dtypes::*;
#[cfg(feature = "cpu")]
pub use image_ops::*;
#[cfg(feature = "cpu")]
pub use latent_composite::*;
pub use native_node_payload::*;
pub use operation::*;
pub use operation_contracts::*;
pub use promotion::*;
pub use rng::*;
pub use shader::*;

include!(concat!(env!("OUT_DIR"), "/generated_modules.rs"));

#[cfg(feature = "rocm")]
pub use generated_backend_amd_rocm_sim_model_0014::RocmTensorBackend;
#[cfg(feature = "metal")]
pub use generated_backend_apple_metal_mps_sim_model_0015::MetalTensorBackend;
#[cfg(feature = "mlu")]
pub use generated_backend_cambricon_mlu_sim_model_0017::MluTensorBackend;
#[cfg(feature = "directml")]
pub use generated_backend_directml_sim_model_0018::DirectMlTensorBackend;
#[cfg(feature = "npu")]
pub use generated_backend_huawei_ascend_npu_sim_model_0019::NpuTensorBackend;
#[cfg(feature = "xpu")]
pub use generated_backend_intel_xpu_sim_model_0021::XpuTensorBackend;
#[cfg(feature = "cuda")]
pub use generated_backend_nvidia_cuda_sim_model_0022::CudaTensorBackend;

#[cfg(test)]
pub(crate) mod validation_artifacts {
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        error::Error,
        fs, io,
        path::{Path, PathBuf},
    };

    fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn target_directory(workspace_root: &Path) -> PathBuf {
        match std::env::var_os("CARGO_TARGET_DIR") {
            Some(directory) => {
                let directory = PathBuf::from(directory);
                if directory.is_absolute() {
                    directory
                } else {
                    workspace_root.join(directory)
                }
            }
            None => workspace_root.join("target"),
        }
    }

    fn json_string(value: &str) -> String {
        let mut encoded = String::with_capacity(value.len() + 2);
        encoded.push('"');
        for character in value.chars() {
            match character {
                '"' => encoded.push_str("\\\""),
                '\\' => encoded.push_str("\\\\"),
                '\n' => encoded.push_str("\\n"),
                '\r' => encoded.push_str("\\r"),
                '\t' => encoded.push_str("\\t"),
                character if character.is_control() => {
                    encoded.push_str(&format!("\\u{:04x}", u32::from(character)));
                }
                character => encoded.push(character),
            }
        }
        encoded.push('"');
        encoded
    }

    fn json_object_string_values(values: &BTreeMap<&str, &str>) -> String {
        values
            .iter()
            .map(|(key, value)| format!("{}: {}", json_string(key), json_string(value)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn json_object_boolean_values(values: &BTreeMap<&str, bool>) -> String {
        values
            .iter()
            .map(|(key, value)| format!("{}: {value}", json_string(key)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn json_string_array(values: &[&str]) -> String {
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(crate) fn workspace_fixture_digest(
        relative_path: &str,
        expected: &str,
    ) -> Result<String, Box<dyn Error>> {
        let bytes = fs::read(workspace_root()?.join(relative_path))?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != expected {
            return Err(io::Error::other(format!(
                "fixture digest mismatch for {relative_path}: expected {expected}, got {actual}"
            ))
            .into());
        }
        Ok(actual)
    }

    pub(crate) fn workspace_contract_fixture_digest(
        relative_path: &str,
        expected_digest: &str,
        expected_fixture_id: &str,
        expected_operation_id: &str,
    ) -> Result<String, Box<dyn Error>> {
        let digest = workspace_fixture_digest(relative_path, expected_digest)?;
        let bytes = fs::read(workspace_root()?.join(relative_path))?;
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        for (field, expected) in [
            ("fixture_id", expected_fixture_id),
            ("operation_id", expected_operation_id),
        ] {
            if fixture.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
                return Err(io::Error::other(format!(
                    "fixture identity mismatch for {relative_path}: {field} is not {expected}"
                ))
                .into());
            }
        }
        Ok(digest)
    }

    pub(crate) fn write(
        artifact_name: &str,
        validation_id: &str,
        scope: &str,
        validation_stage: &str,
        fixture_digests: &BTreeMap<&str, &str>,
        cases: &BTreeMap<&str, bool>,
        remaining_release_gates: &[&str],
    ) -> Result<(), Box<dyn Error>> {
        if let Some((case, _)) = cases.iter().find(|(_, passed)| !**passed) {
            return Err(io::Error::other(format!(
                "{validation_id} validation case failed: {case}"
            ))
            .into());
        }
        if let Some((fixture, _)) = fixture_digests.iter().find(|(_, digest)| {
            digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }) {
            return Err(io::Error::other(format!(
                "{validation_id} fixture has an invalid SHA-256 digest: {fixture}"
            ))
            .into());
        }

        let workspace_root = workspace_root()?;
        let artifact_directory = target_directory(&workspace_root).join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        let artifact_path = artifact_directory.join(artifact_name);
        let temporary_path = artifact_directory.join(format!("{artifact_name}.tmp"));
        match fs::remove_file(&temporary_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        match fs::remove_file(&artifact_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let artifact = format!(
            concat!(
                "{{\n",
                "  \"validation_id\": {},\n",
                "  \"validation\": {},\n",
                "  \"scope\": {},\n",
                "  \"environment\": {{\"operating_system\": {}, \"architecture\": {}, \"backend\": \"native-rust-cpu\", \"development_oracle_executed\": false}},\n",
                "  \"fixture_digests\": {{{}}},\n",
                "  \"summary\": {{\"passed\": {}, \"failed\": 0, \"skipped\": 0}},\n",
                "  \"cases\": {{{}}},\n",
                "  \"skipped\": [],\n",
                "  \"validation_closure\": {{\"claimed\": true, \"stage\": {}, \"validated_scope\": {}}},\n",
                "  \"release_closure_claimed\": false,\n",
                "  \"release_closure_required\": true,\n",
                "  \"remaining_release_gates\": [{}]\n",
                "}}\n"
            ),
            json_string(validation_id),
            json_string(validation_id),
            json_string(scope),
            json_string(std::env::consts::OS),
            json_string(std::env::consts::ARCH),
            json_object_string_values(fixture_digests),
            cases.len(),
            json_object_boolean_values(cases),
            json_string(validation_stage),
            json_string(scope),
            json_string_array(remaining_release_gates),
        );
        fs::write(&temporary_path, artifact.as_bytes())?;
        fs::rename(temporary_path, artifact_path)?;
        Ok(())
    }
}

const STORAGE_ALIGNMENT: usize = 16;
static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

fn next_storage_id() -> Result<StorageId, TensorError> {
    NEXT_STORAGE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map(StorageId)
        .map_err(|_| TensorError::IdentifierOverflow)
}

fn next_tensor_id() -> Result<TensorId, TensorError> {
    NEXT_TENSOR_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map(TensorId)
        .map_err(|_| TensorError::IdentifierOverflow)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: u32,
}

impl DeviceId {
    pub const CPU: Self = Self::new(DeviceKind::Cpu, 0);

    pub const fn new(kind: DeviceKind, ordinal: u32) -> Self {
        Self { kind, ordinal }
    }

    pub const fn kind(self) -> DeviceKind {
        self.kind
    }

    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }

    pub fn from_source_device(value: &str) -> Result<Self, TensorError> {
        let (kind, ordinal) = match value.split_once(':') {
            Some((kind, ordinal)) if !ordinal.contains(':') && !ordinal.is_empty() => {
                let ordinal = ordinal.parse::<u32>().map_err(|_| TensorError::Faulted {
                    reason: format!("device ordinal is invalid in {value:?}"),
                })?;
                (kind, ordinal)
            }
            Some(_) => {
                return Err(TensorError::Faulted {
                    reason: format!("device identifier is invalid: {value:?}"),
                });
            }
            None => (value, 0),
        };
        let kind = match kind {
            "cpu" => DeviceKind::Cpu,
            "cuda" => DeviceKind::Cuda,
            "rocm" | "hip" => DeviceKind::Rocm,
            "metal" | "mps" => DeviceKind::Metal,
            "directml" => DeviceKind::DirectMl,
            "xpu" => DeviceKind::Xpu,
            "npu" => DeviceKind::Npu,
            "mlu" => DeviceKind::Mlu,
            "corex" | "ixuca" => DeviceKind::CoreX,
            _ => {
                return Err(TensorError::Faulted {
                    reason: format!("device kind is unsupported: {kind:?}"),
                });
            }
        };
        Ok(Self::new(kind, ordinal))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct StreamId(u64);

impl StreamId {
    pub const DEFAULT: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Contiguous,
    ChannelsLast,
    ChannelsLast3d,
    Strided,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TensorDescriptorWire", into = "TensorDescriptorWire")]
pub struct TensorDescriptor {
    shape: Vec<u64>,
    strides: Vec<i64>,
    offset_elements: u64,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    stream: StreamId,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TensorDescriptorWire {
    shape: Vec<u64>,
    strides: Vec<i64>,
    offset_elements: u64,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    stream: StreamId,
}

impl TryFrom<TensorDescriptorWire> for TensorDescriptor {
    type Error = TensorError;

    fn try_from(value: TensorDescriptorWire) -> Result<Self, Self::Error> {
        Self::new_strided(
            value.shape,
            value.strides,
            value.offset_elements,
            value.dtype,
            value.layout,
            value.device,
            value.stream,
        )
    }
}

impl From<TensorDescriptor> for TensorDescriptorWire {
    fn from(value: TensorDescriptor) -> Self {
        Self {
            shape: value.shape,
            strides: value.strides,
            offset_elements: value.offset_elements,
            dtype: value.dtype,
            layout: value.layout,
            device: value.device,
            stream: value.stream,
        }
    }
}

impl TensorDescriptor {
    pub fn contiguous(
        shape: Vec<u64>,
        dtype: DType,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Self, TensorError> {
        let strides = contiguous_strides(&shape)?;
        Self::new_strided(shape, strides, 0, dtype, Layout::Contiguous, device, stream)
    }

    pub fn channels_last(
        shape: Vec<u64>,
        dtype: DType,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Self, TensorError> {
        let strides = channels_last_strides(&shape, false)?;
        Self::new_strided(
            shape,
            strides,
            0,
            dtype,
            Layout::ChannelsLast,
            device,
            stream,
        )
    }

    pub fn channels_last_3d(
        shape: Vec<u64>,
        dtype: DType,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Self, TensorError> {
        let strides = channels_last_strides(&shape, true)?;
        Self::new_strided(
            shape,
            strides,
            0,
            dtype,
            Layout::ChannelsLast3d,
            device,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_strided(
        shape: Vec<u64>,
        strides: Vec<i64>,
        offset_elements: u64,
        dtype: DType,
        layout: Layout,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Self, TensorError> {
        if shape.len() != strides.len() {
            return Err(TensorError::StrideRankMismatch {
                rank: shape.len(),
                strides: strides.len(),
            });
        }
        let descriptor = Self {
            shape,
            strides,
            offset_elements,
            dtype,
            layout,
            device,
            stream,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn offset_elements(&self) -> u64 {
        self.offset_elements
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn stream(&self) -> StreamId {
        self.stream
    }

    pub fn narrowed_view(
        &self,
        dimension: usize,
        start: i64,
        length: u64,
    ) -> Result<Self, TensorError> {
        let size = self
            .shape
            .get(dimension)
            .copied()
            .ok_or(TensorError::DimensionOutOfBounds {
                dimension,
                rank: self.rank(),
            })?;
        let range = normalize_narrow_range(size, start, length)?;
        let stride = self
            .strides
            .get(dimension)
            .copied()
            .ok_or(TensorError::ShapeOverflow)?;
        let offset_elements = i128::from(self.offset_elements)
            .checked_add(
                i128::from(range.start)
                    .checked_mul(i128::from(stride))
                    .ok_or(TensorError::ShapeOverflow)?,
            )
            .ok_or(TensorError::ShapeOverflow)?;
        let offset_elements =
            u64::try_from(offset_elements).map_err(|_| TensorError::NegativeStorageOffset)?;
        let mut shape = self.shape.clone();
        *shape.get_mut(dimension).ok_or(TensorError::ShapeOverflow)? = length;
        Self::new_strided(
            shape,
            self.strides.clone(),
            offset_elements,
            self.dtype,
            Layout::Strided,
            self.device,
            self.stream,
        )
    }

    pub fn reshaped_view(&self, shape: Vec<u64>) -> Result<Self, TensorError> {
        let target_elements = shape.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(TensorError::ShapeOverflow)
        })?;
        if target_elements != self.element_count()? {
            return Err(TensorError::StorageLength {
                expected: self.byte_len()?,
                actual: target_elements
                    .checked_mul(self.dtype.byte_width())
                    .ok_or(TensorError::ShapeOverflow)?,
            });
        }

        let strides = compatible_reshape_strides(&self.shape, &self.strides, &shape)?
            .ok_or(TensorError::NonContiguousAccess)?;
        Self::new_strided(
            shape,
            strides,
            self.offset_elements,
            self.dtype,
            Layout::Strided,
            self.device,
            self.stream,
        )
    }

    pub fn permuted_view(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        if permutation.len() != self.rank() {
            return Err(TensorError::DimensionOutOfBounds {
                dimension: permutation.len(),
                rank: self.rank(),
            });
        }
        let mut seen = vec![false; self.rank()];
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        shape
            .try_reserve_exact(self.rank())
            .map_err(|_| TensorError::ShapeOverflow)?;
        strides
            .try_reserve_exact(self.rank())
            .map_err(|_| TensorError::ShapeOverflow)?;
        for &axis in permutation {
            let was_seen = seen
                .get_mut(axis)
                .ok_or(TensorError::DimensionOutOfBounds {
                    dimension: axis,
                    rank: self.rank(),
                })?;
            if *was_seen {
                return Err(TensorError::DimensionOutOfBounds {
                    dimension: axis,
                    rank: self.rank(),
                });
            }
            *was_seen = true;
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        }
        Self::new_strided(
            shape,
            strides,
            self.offset_elements,
            self.dtype,
            Layout::Strided,
            self.device,
            self.stream,
        )
    }

    pub fn reinterpreted_dtype_view(&self, dtype: DType) -> Result<Self, TensorError> {
        let last_axis = self
            .rank()
            .checked_sub(1)
            .ok_or(TensorError::NonContiguousAccess)?;
        if self.strides.get(last_axis).copied() != Some(1) {
            return Err(TensorError::NonContiguousAccess);
        }

        let source_width = self.dtype.byte_width();
        let target_width = dtype.byte_width();
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let mut offset_elements = self.offset_elements;

        if source_width > target_width {
            if !source_width.is_multiple_of(target_width) {
                return Err(TensorError::ShapeOverflow);
            }
            let ratio = source_width / target_width;
            shape[last_axis] = shape[last_axis]
                .checked_mul(ratio)
                .ok_or(TensorError::ShapeOverflow)?;
            offset_elements = offset_elements
                .checked_mul(ratio)
                .ok_or(TensorError::ShapeOverflow)?;
            for stride in &mut strides[..last_axis] {
                *stride = i64::try_from(
                    i128::from(*stride)
                        .checked_mul(i128::from(ratio))
                        .ok_or(TensorError::ShapeOverflow)?,
                )
                .map_err(|_| TensorError::ShapeOverflow)?;
            }
        } else if source_width < target_width {
            if !target_width.is_multiple_of(source_width) {
                return Err(TensorError::ShapeOverflow);
            }
            let ratio = target_width / source_width;
            if !shape[last_axis].is_multiple_of(ratio) || !offset_elements.is_multiple_of(ratio) {
                return Err(TensorError::NonContiguousAccess);
            }
            for stride in &mut strides[..last_axis] {
                let ratio = i64::try_from(ratio).map_err(|_| TensorError::ShapeOverflow)?;
                if *stride % ratio != 0 {
                    return Err(TensorError::NonContiguousAccess);
                }
                *stride /= ratio;
            }
            shape[last_axis] /= ratio;
            offset_elements /= ratio;
        }

        Self::new_strided(
            shape,
            strides,
            offset_elements,
            dtype,
            if self.layout == Layout::Contiguous {
                Layout::Contiguous
            } else {
                Layout::Strided
            },
            self.device,
            self.stream,
        )
    }

    pub fn element_count(&self) -> Result<u64, TensorError> {
        self.shape.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(TensorError::ShapeOverflow)
        })
    }

    pub fn byte_len(&self) -> Result<u64, TensorError> {
        self.element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or(TensorError::ShapeOverflow)
    }

    pub fn storage_span_elements(&self) -> Result<Option<Range<u64>>, TensorError> {
        if self.element_count()? == 0 {
            return Ok(Some(self.offset_elements..self.offset_elements));
        }
        let mut minimum = i128::from(self.offset_elements);
        let mut maximum = minimum;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides) {
            if dimension <= 1 {
                continue;
            }
            let extent = i128::from(dimension - 1)
                .checked_mul(i128::from(stride))
                .ok_or(TensorError::ShapeOverflow)?;
            if extent < 0 {
                minimum = minimum
                    .checked_add(extent)
                    .ok_or(TensorError::ShapeOverflow)?;
            } else {
                maximum = maximum
                    .checked_add(extent)
                    .ok_or(TensorError::ShapeOverflow)?;
            }
        }
        if minimum < 0 {
            return Err(TensorError::NegativeStorageOffset);
        }
        let maximum = maximum.checked_add(1).ok_or(TensorError::ShapeOverflow)?;
        let start = u64::try_from(minimum).map_err(|_| TensorError::ShapeOverflow)?;
        let end = u64::try_from(maximum).map_err(|_| TensorError::ShapeOverflow)?;
        Ok(Some(start..end))
    }

    pub fn storage_span_bytes(&self) -> Result<Option<Range<u64>>, TensorError> {
        self.storage_span_elements()?
            .map(|range| {
                let start = range
                    .start
                    .checked_mul(self.dtype.byte_width())
                    .ok_or(TensorError::ShapeOverflow)?;
                let end = range
                    .end
                    .checked_mul(self.dtype.byte_width())
                    .ok_or(TensorError::ShapeOverflow)?;
                Ok(start..end)
            })
            .transpose()
    }

    pub fn minimum_backing_byte_length(&self) -> Result<u64, TensorError> {
        Ok(self.storage_span_bytes()?.map_or(0, |range| range.end))
    }

    pub fn validate_backing_byte_length(&self, byte_length: u64) -> Result<(), TensorError> {
        let required = self.minimum_backing_byte_length()?;
        if required > byte_length {
            return Err(TensorError::StorageBounds {
                required,
                actual: byte_length,
            });
        }
        Ok(())
    }

    pub fn is_contiguous(&self) -> Result<bool, TensorError> {
        if self.shape.contains(&0) {
            return Ok(true);
        }
        let mut expected_stride = 1_i128;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension == 1 {
                continue;
            }
            if i128::from(stride) != expected_stride {
                return Ok(false);
            }
            expected_stride = expected_stride
                .checked_mul(i128::from(dimension))
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok(true)
    }

    pub fn is_non_overlapping(&self) -> Result<bool, TensorError> {
        if self.element_count()? == 0 {
            return Ok(true);
        }
        let mut dimensions = self
            .shape
            .iter()
            .zip(&self.strides)
            .filter_map(|(&dimension, &stride)| (dimension > 1).then_some((dimension, stride)))
            .collect::<Vec<_>>();
        dimensions.sort_by_key(|(_, stride)| stride.unsigned_abs());
        let mut required_span = 1_u128;
        for (dimension, stride) in dimensions {
            let absolute_stride = u128::from(stride.unsigned_abs());
            if absolute_stride < required_span {
                return Ok(false);
            }
            required_span = absolute_stride
                .checked_mul(u128::from(dimension - 1))
                .and_then(|extent| extent.checked_add(required_span))
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok(true)
    }

    pub fn preserving_format_for(
        &self,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self, TensorError> {
        if !self.is_non_overlapping()? {
            return Self::contiguous(self.shape.clone(), dtype, device, self.stream);
        }
        let offset_elements = self.shape.iter().zip(&self.strides).try_fold(
            0_i128,
            |offset, (&dimension, &stride)| {
                if stride >= 0 || dimension <= 1 {
                    Ok(offset)
                } else {
                    let absolute_stride = stride.checked_abs().ok_or(TensorError::ShapeOverflow)?;
                    i128::from(dimension - 1)
                        .checked_mul(i128::from(absolute_stride))
                        .and_then(|extent| offset.checked_add(extent))
                        .ok_or(TensorError::ShapeOverflow)
                }
            },
        )?;
        Self::new_strided(
            self.shape.clone(),
            self.strides.clone(),
            u64::try_from(offset_elements).map_err(|_| TensorError::ShapeOverflow)?,
            dtype,
            self.layout,
            device,
            self.stream,
        )
    }

    fn validate(&self) -> Result<(), TensorError> {
        self.element_count()?;
        self.byte_len()?;
        self.storage_span_bytes()?;
        let expected = match self.layout {
            Layout::Contiguous => Some(contiguous_strides(&self.shape)?),
            Layout::ChannelsLast => Some(channels_last_strides(&self.shape, false)?),
            Layout::ChannelsLast3d => Some(channels_last_strides(&self.shape, true)?),
            Layout::Strided => None,
        };
        if let Some(expected) = expected
            && expected != self.strides
        {
            return Err(TensorError::InvalidLayout {
                layout: self.layout,
            });
        }
        Ok(())
    }
}

fn contiguous_strides(shape: &[u64]) -> Result<Vec<i64>, TensorError> {
    let mut strides = vec![0_i64; shape.len()];
    let mut stride = 1_u64;
    for (index, dimension) in shape.iter().enumerate().rev() {
        let slot = strides.get_mut(index).ok_or(TensorError::ShapeOverflow)?;
        *slot = i64::try_from(stride).map_err(|_| TensorError::ShapeOverflow)?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or(TensorError::ShapeOverflow)?;
    }
    Ok(strides)
}

fn compatible_reshape_strides(
    source_shape: &[u64],
    source_strides: &[i64],
    target_shape: &[u64],
) -> Result<Option<Vec<i64>>, TensorError> {
    if source_shape.contains(&0) {
        return contiguous_strides(target_shape).map(Some);
    }
    if source_shape.is_empty() {
        return contiguous_strides(target_shape).map(Some);
    }

    let mut target_strides = vec![0_i64; target_shape.len()];
    let mut target_dimension = target_shape.len();
    let mut source_dimension = source_shape.len();
    let mut chunk_elements = 1_u128;
    let mut target_chunk_elements = 1_u128;
    let mut chunk_base_stride =
        i128::from(*source_strides.last().ok_or(TensorError::ShapeOverflow)?);

    while source_dimension > 0 {
        source_dimension -= 1;
        chunk_elements = chunk_elements
            .checked_mul(u128::from(source_shape[source_dimension]))
            .ok_or(TensorError::ShapeOverflow)?;
        let starts_chunk = source_dimension == 0
            || (source_shape[source_dimension - 1] != 1
                && i128::from(source_strides[source_dimension - 1])
                    != i128::try_from(chunk_elements)
                        .map_err(|_| TensorError::ShapeOverflow)?
                        .checked_mul(chunk_base_stride)
                        .ok_or(TensorError::ShapeOverflow)?);
        if !starts_chunk {
            continue;
        }

        while target_dimension > 0
            && (target_chunk_elements < chunk_elements || target_shape[target_dimension - 1] == 1)
        {
            target_dimension -= 1;
            let stride = i128::try_from(target_chunk_elements)
                .map_err(|_| TensorError::ShapeOverflow)?
                .checked_mul(chunk_base_stride)
                .ok_or(TensorError::ShapeOverflow)?;
            target_strides[target_dimension] =
                i64::try_from(stride).map_err(|_| TensorError::ShapeOverflow)?;
            target_chunk_elements = target_chunk_elements
                .checked_mul(u128::from(target_shape[target_dimension]))
                .ok_or(TensorError::ShapeOverflow)?;
        }
        if target_chunk_elements != chunk_elements {
            return Ok(None);
        }
        if source_dimension > 0 {
            chunk_base_stride = i128::from(source_strides[source_dimension - 1]);
            chunk_elements = 1;
            target_chunk_elements = 1;
        }
    }

    Ok((target_dimension == 0).then_some(target_strides))
}

fn channels_last_strides(shape: &[u64], three_dimensional: bool) -> Result<Vec<i64>, TensorError> {
    let expected_rank = if three_dimensional { 5 } else { 4 };
    if shape.len() != expected_rank {
        return Err(TensorError::InvalidLayoutRank {
            layout: if three_dimensional {
                Layout::ChannelsLast3d
            } else {
                Layout::ChannelsLast
            },
            expected: expected_rank,
            actual: shape.len(),
        });
    }
    let order: &[usize] = if three_dimensional {
        &[1, 4, 3, 2, 0]
    } else {
        &[1, 3, 2, 0]
    };
    let mut strides = vec![0_i64; shape.len()];
    let mut stride = 1_u64;
    for &index in order {
        let slot = strides.get_mut(index).ok_or(TensorError::ShapeOverflow)?;
        *slot = i64::try_from(stride).map_err(|_| TensorError::ShapeOverflow)?;
        let dimension = shape.get(index).ok_or(TensorError::ShapeOverflow)?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or(TensorError::ShapeOverflow)?;
    }
    Ok(strides)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StorageId(u64);

impl StorageId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TensorId(u64);

impl TensorId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
pub struct MutationLineage {
    epoch: AtomicU64,
}

impl MutationLineage {
    fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    fn advance(&self) -> Result<(), TensorError> {
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(|_| ())
            .map_err(|_| TensorError::IdentifierOverflow)
    }
}

#[derive(Clone, Debug)]
pub struct MutationWitness {
    tensor_id: TensorId,
    lineage: Arc<MutationLineage>,
    expected_epoch: u64,
}

impl MutationWitness {
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }

    pub fn expected_epoch(&self) -> u64 {
        self.expected_epoch
    }

    pub fn actual_epoch(&self) -> u64 {
        self.lineage.epoch()
    }

    pub fn is_current(&self) -> bool {
        self.expected_epoch == self.actual_epoch()
    }
}

pub(crate) trait BackendStorage: Any + fmt::Debug + Send + Sync {
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any;
    fn device(&self) -> DeviceId;
    fn byte_len(&self) -> u64;
    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError>;
    fn host_bytes(&self) -> Option<&[u8]>;
    fn host_bytes_mut(&mut self) -> Option<&mut [u8]>;
}

struct Storage {
    id: StorageId,
    allocation: Box<dyn BackendStorage>,
}

impl fmt::Debug for Storage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Storage")
            .field("id", &self.id)
            .field("allocation", &self.allocation)
            .finish()
    }
}

impl Storage {
    fn new(allocation: Box<dyn BackendStorage>) -> Result<Self, TensorError> {
        Ok(Self {
            id: next_storage_id()?,
            allocation,
        })
    }

    fn clone_for_write(&self) -> Result<Self, TensorError> {
        Ok(Self {
            id: next_storage_id()?,
            allocation: self.allocation.clone_for_write()?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewAccess {
    ReadOnly,
    Writable,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    id: TensorId,
    mutation: Arc<MutationLineage>,
    descriptor: TensorDescriptor,
    storage: Arc<Storage>,
    access: ViewAccess,
}

impl Tensor {
    #[cfg(test)]
    pub(crate) fn from_bytes(
        descriptor: TensorDescriptor,
        bytes: Vec<u8>,
    ) -> Result<Self, TensorError> {
        if descriptor.device.kind() != DeviceKind::Cpu {
            return Err(TensorError::NonHostDevice {
                device: descriptor.device,
            });
        }
        let required = required_storage_bytes(&descriptor)?;
        let actual = u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
        if actual != required {
            return Err(TensorError::StorageLength {
                expected: required,
                actual,
            });
        }
        let allocation = CpuStorage::from_bytes(&bytes)?;
        Self::from_backend_storage(descriptor, Box::new(allocation), ViewAccess::Writable)
    }

    pub(crate) fn from_backend_storage(
        descriptor: TensorDescriptor,
        allocation: Box<dyn BackendStorage>,
        access: ViewAccess,
    ) -> Result<Self, TensorError> {
        if descriptor.device != allocation.device() {
            return Err(TensorError::DeviceMismatch {
                expected: descriptor.device,
                actual: allocation.device(),
            });
        }
        validate_storage_bounds(&descriptor, allocation.byte_len())?;
        if access == ViewAccess::Writable && !descriptor.is_non_overlapping()? {
            return Err(TensorError::OverlappingWrite);
        }
        Ok(Self {
            id: next_tensor_id()?,
            mutation: Arc::new(MutationLineage::new()),
            descriptor,
            storage: Arc::new(Storage::new(allocation)?),
            access,
        })
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn storage_id(&self) -> StorageId {
        self.storage.id
    }

    pub fn tensor_id(&self) -> TensorId {
        self.id
    }

    pub fn mutation_version(&self) -> u64 {
        self.mutation.epoch()
    }

    pub fn mutation_witness(&self) -> MutationWitness {
        MutationWitness {
            tensor_id: self.id,
            lineage: self.mutation.clone(),
            expected_epoch: self.mutation.epoch(),
        }
    }

    pub fn storage_version(&self) -> u64 {
        self.mutation_version()
    }

    pub fn storage_byte_len(&self) -> u64 {
        self.storage.allocation.byte_len()
    }

    #[allow(dead_code)]
    pub(crate) fn backend_storage<StorageType: Any>(&self) -> Option<&StorageType> {
        self.storage.allocation.as_any().downcast_ref()
    }

    pub fn access(&self) -> ViewAccess {
        self.access
    }

    pub fn host_storage_bytes(&self) -> Result<&[u8], TensorError> {
        self.storage
            .allocation
            .host_bytes()
            .ok_or(TensorError::NonHostStorage)
    }

    pub fn contiguous_bytes(&self) -> Result<&[u8], TensorError> {
        if !self.descriptor.is_contiguous()? {
            return Err(TensorError::NonContiguousAccess);
        }
        let bytes = self.host_storage_bytes()?;
        let range = descriptor_byte_range(&self.descriptor)?;
        bytes.get(range).ok_or(TensorError::StorageBounds {
            required: required_storage_bytes(&self.descriptor)?,
            actual: self.storage.allocation.byte_len(),
        })
    }

    pub fn element_bytes(&self, indices: &[u64]) -> Result<&[u8], TensorError> {
        let range = element_byte_range(&self.descriptor, indices)?;
        self.host_storage_bytes()?
            .get(range)
            .ok_or(TensorError::StorageBounds {
                required: required_storage_bytes(&self.descriptor)?,
                actual: self.storage.allocation.byte_len(),
            })
    }

    pub fn linear_element_bytes(&self, linear_index: u64) -> Result<&[u8], TensorError> {
        let element_count = self.descriptor.element_count()?;
        if linear_index >= element_count {
            return Err(TensorError::IndexOutOfBounds {
                dimension: 0,
                index: linear_index,
                size: element_count,
            });
        }
        let mut remainder = linear_index;
        let mut element = i128::from(self.descriptor.offset_elements);
        for (&dimension, &stride) in self
            .descriptor
            .shape
            .iter()
            .zip(&self.descriptor.strides)
            .rev()
        {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            let contribution = i128::from(coordinate)
                .checked_mul(i128::from(stride))
                .ok_or(TensorError::ShapeOverflow)?;
            element = element
                .checked_add(contribution)
                .ok_or(TensorError::ShapeOverflow)?;
        }
        let range = element_storage_byte_range(&self.descriptor, element)?;
        self.host_storage_bytes()?
            .get(range)
            .ok_or(TensorError::StorageBounds {
                required: required_storage_bytes(&self.descriptor)?,
                actual: self.storage.allocation.byte_len(),
            })
    }

    pub fn view(
        &self,
        descriptor: TensorDescriptor,
        access: ViewAccess,
    ) -> Result<Self, TensorError> {
        if self.access == ViewAccess::ReadOnly && access == ViewAccess::Writable {
            return Err(TensorError::ReadOnlyView);
        }
        if descriptor.dtype != self.descriptor.dtype {
            return Err(TensorError::DTypeMismatch {
                expected: self.descriptor.dtype,
                actual: descriptor.dtype,
            });
        }
        if descriptor.device != self.descriptor.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.descriptor.device,
                actual: descriptor.device,
            });
        }
        if descriptor.stream != self.descriptor.stream {
            return Err(TensorError::StreamMismatch {
                expected: self.descriptor.stream,
                actual: descriptor.stream,
            });
        }
        validate_storage_bounds(&descriptor, self.storage.allocation.byte_len())?;
        if access == ViewAccess::Writable && !descriptor.is_non_overlapping()? {
            return Err(TensorError::OverlappingWrite);
        }
        Ok(Self {
            id: next_tensor_id()?,
            mutation: self.mutation.clone(),
            descriptor,
            storage: self.storage.clone(),
            access,
        })
    }

    pub fn narrow_read_only(
        &self,
        dimension: usize,
        start: i64,
        length: u64,
    ) -> Result<Self, TensorError> {
        self.view(
            self.descriptor.narrowed_view(dimension, start, length)?,
            ViewAccess::ReadOnly,
        )
    }

    pub fn reinterpret_contiguous_read_only(
        &self,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        if self.descriptor.layout != Layout::Contiguous
            || descriptor.layout != Layout::Contiguous
            || self.descriptor.offset_elements != 0
            || descriptor.offset_elements != 0
        {
            return Err(TensorError::InvalidLayout {
                layout: descriptor.layout,
            });
        }
        if descriptor.device != self.descriptor.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.descriptor.device,
                actual: descriptor.device,
            });
        }
        if descriptor.stream != self.descriptor.stream {
            return Err(TensorError::StreamMismatch {
                expected: self.descriptor.stream,
                actual: descriptor.stream,
            });
        }
        let source_bytes = self.descriptor.byte_len()?;
        let target_bytes = descriptor.byte_len()?;
        if source_bytes != target_bytes {
            return Err(TensorError::StorageLength {
                expected: source_bytes,
                actual: target_bytes,
            });
        }
        validate_storage_bounds(&descriptor, self.storage.allocation.byte_len())?;
        Ok(Self {
            id: next_tensor_id()?,
            mutation: self.mutation.clone(),
            descriptor,
            storage: self.storage.clone(),
            access: ViewAccess::ReadOnly,
        })
    }

    pub fn reinterpret_read_only(&self, descriptor: TensorDescriptor) -> Result<Self, TensorError> {
        if descriptor.device != self.descriptor.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.descriptor.device,
                actual: descriptor.device,
            });
        }
        if descriptor.stream != self.descriptor.stream {
            return Err(TensorError::StreamMismatch {
                expected: self.descriptor.stream,
                actual: descriptor.stream,
            });
        }
        if self.descriptor.storage_span_bytes()? != descriptor.storage_span_bytes()? {
            return Err(TensorError::StorageLength {
                expected: required_storage_bytes(&self.descriptor)?,
                actual: required_storage_bytes(&descriptor)?,
            });
        }
        validate_storage_bounds(&descriptor, self.storage.allocation.byte_len())?;
        Ok(Self {
            id: next_tensor_id()?,
            mutation: self.mutation.clone(),
            descriptor,
            storage: self.storage.clone(),
            access: ViewAccess::ReadOnly,
        })
    }

    pub fn write(&mut self) -> Result<TensorWrite<'_>, TensorError> {
        if self.access != ViewAccess::Writable {
            return Err(TensorError::ReadOnlyView);
        }
        if !self.descriptor.is_non_overlapping()? {
            return Err(TensorError::OverlappingWrite);
        }
        if self.storage.allocation.host_bytes().is_none() {
            return Err(TensorError::NonHostStorage);
        }
        let contiguous = self.descriptor.is_contiguous()?;
        let range = descriptor_byte_range(&self.descriptor)?;
        let descriptor = self.descriptor.clone();
        if Arc::get_mut(&mut self.storage).is_none() {
            self.storage = Arc::new(self.storage.clone_for_write()?);
        }
        let storage = Arc::get_mut(&mut self.storage).ok_or(TensorError::SharedWriteLease)?;
        Ok(TensorWrite {
            storage,
            mutation: &self.mutation,
            descriptor,
            range,
            contiguous,
            version_bumped: false,
        })
    }

    pub fn detached_alias(&self) -> Result<Self, TensorError> {
        Ok(Self {
            id: next_tensor_id()?,
            mutation: self.mutation.clone(),
            descriptor: self.descriptor.clone(),
            storage: self.storage.clone(),
            access: self.access,
        })
    }

    pub fn data_alias(&self) -> Result<Self, TensorError> {
        self.detached_alias()
    }

    pub fn detached_in_place(&mut self) -> &mut Self {
        self
    }

    pub fn data_in_place(&mut self) -> &mut Self {
        self
    }

    pub fn replace_data(&mut self, staged: Self) -> Result<(), TensorError> {
        self.commit_in_place(staged)
    }

    pub fn commit_in_place(&mut self, staged: Self) -> Result<(), TensorError> {
        if self.access != ViewAccess::Writable {
            return Err(TensorError::ReadOnlyView);
        }
        if self.descriptor.device != staged.descriptor.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.descriptor.device,
                actual: staged.descriptor.device,
            });
        }
        if self.descriptor.stream != staged.descriptor.stream {
            return Err(TensorError::StreamMismatch {
                expected: self.descriptor.stream,
                actual: staged.descriptor.stream,
            });
        }
        validate_storage_bounds(&staged.descriptor, staged.storage.allocation.byte_len())?;
        self.mutation.advance()?;
        self.descriptor = staged.descriptor;
        self.storage = staged.storage;
        Ok(())
    }
}

pub(crate) fn normalize_narrow_range(
    size: u64,
    start: i64,
    length: u64,
) -> Result<Range<u64>, TensorError> {
    let start = if start < 0 {
        i128::from(size)
            .checked_add(i128::from(start))
            .ok_or(TensorError::ShapeOverflow)?
    } else {
        i128::from(start)
    };
    let start = u64::try_from(start).map_err(|_| TensorError::InvalidNarrowRange {
        start,
        length,
        size,
    })?;
    let end = start
        .checked_add(length)
        .ok_or(TensorError::ShapeOverflow)?;
    if end > size {
        return Err(TensorError::InvalidNarrowRange {
            start: i128::from(start),
            length,
            size,
        });
    }
    Ok(start..end)
}

fn required_storage_bytes(descriptor: &TensorDescriptor) -> Result<u64, TensorError> {
    descriptor.minimum_backing_byte_length()
}

fn descriptor_byte_range(descriptor: &TensorDescriptor) -> Result<Range<usize>, TensorError> {
    let start = descriptor
        .offset_elements
        .checked_mul(descriptor.dtype.byte_width())
        .ok_or(TensorError::ShapeOverflow)?;
    let end = start
        .checked_add(descriptor.byte_len()?)
        .ok_or(TensorError::ShapeOverflow)?;
    Ok(
        usize::try_from(start).map_err(|_| TensorError::ShapeOverflow)?
            ..usize::try_from(end).map_err(|_| TensorError::ShapeOverflow)?,
    )
}

fn element_byte_range(
    descriptor: &TensorDescriptor,
    indices: &[u64],
) -> Result<Range<usize>, TensorError> {
    if indices.len() != descriptor.rank() {
        return Err(TensorError::IndexRankMismatch {
            rank: descriptor.rank(),
            indices: indices.len(),
        });
    }
    let mut element = i128::from(descriptor.offset_elements);
    for (dimension_index, ((&index, &dimension), &stride)) in indices
        .iter()
        .zip(&descriptor.shape)
        .zip(&descriptor.strides)
        .enumerate()
    {
        if index >= dimension {
            return Err(TensorError::IndexOutOfBounds {
                dimension: dimension_index,
                index,
                size: dimension,
            });
        }
        let contribution = i128::from(index)
            .checked_mul(i128::from(stride))
            .ok_or(TensorError::ShapeOverflow)?;
        element = element
            .checked_add(contribution)
            .ok_or(TensorError::ShapeOverflow)?;
    }
    element_storage_byte_range(descriptor, element)
}

fn element_storage_byte_range(
    descriptor: &TensorDescriptor,
    element: i128,
) -> Result<Range<usize>, TensorError> {
    if element < 0 {
        return Err(TensorError::NegativeStorageOffset);
    }
    let element = u64::try_from(element).map_err(|_| TensorError::ShapeOverflow)?;
    let start = element
        .checked_mul(descriptor.dtype.byte_width())
        .ok_or(TensorError::ShapeOverflow)?;
    let end = start
        .checked_add(descriptor.dtype.byte_width())
        .ok_or(TensorError::ShapeOverflow)?;
    Ok(
        usize::try_from(start).map_err(|_| TensorError::ShapeOverflow)?
            ..usize::try_from(end).map_err(|_| TensorError::ShapeOverflow)?,
    )
}

fn validate_storage_bounds(
    descriptor: &TensorDescriptor,
    storage_byte_len: u64,
) -> Result<(), TensorError> {
    descriptor.validate_backing_byte_length(storage_byte_len)
}

pub struct TensorWrite<'a> {
    storage: &'a mut Storage,
    mutation: &'a MutationLineage,
    descriptor: TensorDescriptor,
    range: Range<usize>,
    contiguous: bool,
    version_bumped: bool,
}

impl TensorWrite<'_> {
    pub(crate) fn storage_bytes_mut(&mut self) -> Result<&mut [u8], TensorError> {
        self.bump_version()?;
        self.storage
            .allocation
            .host_bytes_mut()
            .ok_or(TensorError::NonHostStorage)
    }

    pub fn bytes_mut(&mut self) -> Result<&mut [u8], TensorError> {
        if !self.contiguous {
            return Err(TensorError::NonContiguousAccess);
        }
        self.bump_version()?;
        self.storage
            .allocation
            .host_bytes_mut()
            .and_then(|bytes| bytes.get_mut(self.range.clone()))
            .ok_or(TensorError::NonHostStorage)
    }

    pub fn element_bytes_mut(&mut self, indices: &[u64]) -> Result<&mut [u8], TensorError> {
        let range = element_byte_range(&self.descriptor, indices)?;
        self.bump_version()?;
        self.storage
            .allocation
            .host_bytes_mut()
            .and_then(|bytes| bytes.get_mut(range))
            .ok_or(TensorError::NonHostStorage)
    }

    fn bump_version(&mut self) -> Result<(), TensorError> {
        if !self.version_bumped {
            self.mutation.advance()?;
            self.version_bumped = true;
        }
        Ok(())
    }
}

struct AlignedBytes {
    words: Vec<u128>,
    len: usize,
}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlignedBytes")
            .field("len", &self.len)
            .field("alignment", &STORAGE_ALIGNMENT)
            .finish()
    }
}

impl AlignedBytes {
    fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        let word_count = bytes
            .len()
            .checked_add(STORAGE_ALIGNMENT - 1)
            .ok_or(TensorError::ShapeOverflow)?
            / STORAGE_ALIGNMENT;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|error| TensorError::AllocationFailed {
                requested: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                reason: error.to_string(),
            })?;
        words.resize(word_count, 0_u128);
        let target = bytemuck::cast_slice_mut::<u128, u8>(&mut words);
        let target = target
            .get_mut(..bytes.len())
            .ok_or(TensorError::ShapeOverflow)?;
        target.copy_from_slice(bytes);
        Ok(Self {
            words,
            len: bytes.len(),
        })
    }

    #[cfg(feature = "cpu")]
    fn zeroed(len: usize) -> Result<Self, TensorError> {
        let word_count = len
            .checked_add(STORAGE_ALIGNMENT - 1)
            .ok_or(TensorError::ShapeOverflow)?
            / STORAGE_ALIGNMENT;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|error| TensorError::AllocationFailed {
                requested: u64::try_from(len).unwrap_or(u64::MAX),
                reason: error.to_string(),
            })?;
        words.resize(word_count, 0_u128);
        Ok(Self { words, len })
    }

    fn try_clone(&self) -> Result<Self, TensorError> {
        let bytes = self.as_bytes().ok_or(TensorError::StorageBounds {
            required: u64::try_from(self.len).unwrap_or(u64::MAX),
            actual: u64::try_from(self.words.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(STORAGE_ALIGNMENT as u64),
        })?;
        Self::from_bytes(bytes)
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        let bytes = bytemuck::cast_slice::<u128, u8>(&self.words);
        bytes.get(..self.len)
    }

    fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        let bytes = bytemuck::cast_slice_mut::<u128, u8>(&mut self.words);
        bytes.get_mut(..self.len)
    }
}

#[derive(Debug)]
struct CpuStorage {
    bytes: AlignedBytes,
}

impl CpuStorage {
    #[cfg(test)]
    fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        Ok(Self {
            bytes: AlignedBytes::from_bytes(bytes)?,
        })
    }

    #[cfg(feature = "cpu")]
    fn zeroed(len: usize) -> Result<Self, TensorError> {
        Ok(Self {
            bytes: AlignedBytes::zeroed(len)?,
        })
    }

    fn try_clone(&self) -> Result<Self, TensorError> {
        Ok(Self {
            bytes: self.bytes.try_clone()?,
        })
    }
}

impl BackendStorage for CpuStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn device(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn byte_len(&self) -> u64 {
        u64::try_from(self.bytes.len).unwrap_or(u64::MAX)
    }

    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError> {
        Ok(Box::new(self.try_clone()?))
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        self.bytes.as_bytes()
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.bytes.as_bytes_mut()
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TensorError {
    #[error("tensor shape, stride, offset, or byte length overflow")]
    ShapeOverflow,
    #[error("tensor rank {rank} does not match stride count {strides}")]
    StrideRankMismatch { rank: usize, strides: usize },
    #[error("layout {layout:?} requires rank {expected}, got {actual}")]
    InvalidLayoutRank {
        layout: Layout,
        expected: usize,
        actual: usize,
    },
    #[error("strides do not match declared layout {layout:?}")]
    InvalidLayout { layout: Layout },
    #[error("tensor strides address storage before element zero")]
    NegativeStorageOffset,
    #[error("tensor storage length mismatch: expected {expected}, got {actual}")]
    StorageLength { expected: u64, actual: u64 },
    #[error("tensor view requires {required} storage bytes, but allocation contains {actual}")]
    StorageBounds { required: u64, actual: u64 },
    #[error("tensor allocation of {requested} bytes failed: {reason}")]
    AllocationFailed { requested: u64, reason: String },
    #[error("backend resource {resource} reached its deterministic limit of {limit}")]
    ResourceLimitExceeded {
        resource: &'static str,
        limit: usize,
    },
    #[error(
        "workspace authorization belongs to backend {actual_backend} authority {actual_authority}, expected backend {expected_backend} authority {expected_authority}"
    )]
    WorkspaceAuthorizationMismatch {
        expected_backend: u64,
        expected_authority: u64,
        actual_backend: u64,
        actual_authority: u64,
    },
    #[error(
        "workspace request of {requested} bytes exceeds the {authorized}-byte authorization with {in_use} bytes already leased"
    )]
    WorkspaceAuthorizationExceeded {
        requested: u64,
        authorized: u64,
        in_use: u64,
    },
    #[error("writable tensor views may not overlap themselves")]
    OverlappingWrite,
    #[error("tensor view is read-only")]
    ReadOnlyView,
    #[error("a unique tensor write lease could not be acquired")]
    SharedWriteLease,
    #[error("tensor storage is not host-addressable")]
    NonHostStorage,
    #[error("host byte construction is unavailable for device {device:?}")]
    NonHostDevice { device: DeviceId },
    #[error("contiguous byte access requires a contiguous tensor")]
    NonContiguousAccess,
    #[error("tensor rank {rank} does not match index count {indices}")]
    IndexRankMismatch { rank: usize, indices: usize },
    #[error("tensor dimension {dimension} is outside rank {rank}")]
    DimensionOutOfBounds { dimension: usize, rank: usize },
    #[error("tensor narrow range starting at {start} with length {length} exceeds size {size}")]
    InvalidNarrowRange { start: i128, length: u64, size: u64 },
    #[error("tensor index {index} is outside dimension {dimension} of size {size}")]
    IndexOutOfBounds {
        dimension: usize,
        index: u64,
        size: u64,
    },
    #[error("tensor device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error("tensor dtype mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("tensor stream mismatch: expected {expected:?}, got {actual:?}")]
    StreamMismatch {
        expected: StreamId,
        actual: StreamId,
    },
    #[error("operation {operation} is unsupported on {device:?}: {reason}")]
    UnsupportedCapability {
        operation: String,
        device: DeviceId,
        reason: String,
    },
    #[error("tensor operation was cancelled")]
    Cancelled,
    #[error("tensor device was lost: {reason}")]
    DeviceLost { reason: String },
    #[error("tensor operation produced an invalid numeric value: {reason}")]
    InvalidNumeric { reason: String },
    #[error("tensor operation faulted: {reason}")]
    Faulted { reason: String },
    #[error("tensor identity or version counter overflowed")]
    IdentifierOverflow,
}

impl From<sim_types::CancellationError> for TensorError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::{BTreeMap, HashSet},
        error::Error,
        io,
    };

    fn descriptor(shape: Vec<u64>) -> TensorDescriptor {
        match TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT) {
            Ok(value) => value,
            Err(error) => panic!("test descriptor failed: {error}"),
        }
    }

    #[test]
    fn source_device_adapter_maps_only_the_closed_canonical_vocabulary() {
        assert_eq!(DeviceId::from_source_device("cpu"), Ok(DeviceId::CPU));
        assert_eq!(
            DeviceId::from_source_device("hip:3"),
            Ok(DeviceId::new(DeviceKind::Rocm, 3))
        );
        assert_eq!(
            DeviceId::from_source_device("mps:2"),
            Ok(DeviceId::new(DeviceKind::Metal, 2))
        );
        assert!(DeviceId::from_source_device("cuda:-1").is_err());
        assert!(DeviceId::from_source_device("cuda:").is_err());
        assert!(DeviceId::from_source_device("cuda:1:2").is_err());
        assert!(DeviceId::from_source_device("cuda:4294967296").is_err());
        assert!(DeviceId::from_source_device("python").is_err());
    }

    #[test]
    fn validates_storage_size_and_alignment() {
        let tensor = match Tensor::from_bytes(descriptor(vec![1, 2]), vec![0; 8]) {
            Ok(value) => value,
            Err(error) => panic!("matching storage failed: {error}"),
        };
        let bytes = match tensor.contiguous_bytes() {
            Ok(value) => value,
            Err(error) => panic!("CPU byte access failed: {error}"),
        };
        assert_eq!(bytes.len(), 8);
        let storage_bytes = match tensor.host_storage_bytes() {
            Ok(value) => value,
            Err(error) => panic!("CPU storage access failed: {error}"),
        };
        assert_eq!(storage_bytes.as_ptr().align_offset(STORAGE_ALIGNMENT), 0);
        assert!(matches!(
            Tensor::from_bytes(descriptor(vec![1, 2]), vec![0; 7]),
            Err(TensorError::StorageLength {
                expected: 8,
                actual: 7,
            })
        ));
    }

    #[test]
    fn descriptor_owns_minimum_backing_length_validation() {
        let descriptor = TensorDescriptor::new_strided(
            vec![2, 3],
            vec![4, 1],
            2,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )
        .expect("checked strided descriptor");
        assert_eq!(descriptor.minimum_backing_byte_length(), Ok(36));
        assert_eq!(descriptor.validate_backing_byte_length(36), Ok(()));
        assert_eq!(
            descriptor.validate_backing_byte_length(35),
            Err(TensorError::StorageBounds {
                required: 36,
                actual: 35,
            })
        );
    }

    #[test]
    fn checked_descriptors_reject_overflow_and_invalid_layouts() {
        assert_eq!(
            TensorDescriptor::contiguous(
                vec![u64::MAX, 2],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            ),
            Err(TensorError::ShapeOverflow)
        );
        assert!(matches!(
            TensorDescriptor::channels_last(
                vec![1, 2, 3],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            ),
            Err(TensorError::InvalidLayoutRank { .. })
        ));
        let empty_with_offset = match TensorDescriptor::new_strided(
            vec![0],
            vec![1],
            2,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("empty descriptor failed: {error}"),
        };
        assert!(matches!(
            Tensor::from_bytes(empty_with_offset.clone(), vec![]),
            Err(TensorError::StorageLength {
                expected: 8,
                actual: 0,
            })
        ));
        assert!(Tensor::from_bytes(empty_with_offset, vec![0; 8]).is_ok());
    }

    #[test]
    fn overlapping_views_are_read_only() {
        let tensor = match Tensor::from_bytes(descriptor(vec![2]), vec![0; 8]) {
            Ok(value) => value,
            Err(error) => panic!("base tensor failed: {error}"),
        };
        let overlapping = match TensorDescriptor::new_strided(
            vec![2, 2],
            vec![1, 0],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("overlapping descriptor failed: {error}"),
        };
        assert!(
            tensor
                .view(overlapping.clone(), ViewAccess::ReadOnly)
                .is_ok()
        );
        assert!(matches!(
            tensor.view(overlapping, ViewAccess::Writable),
            Err(TensorError::OverlappingWrite)
        ));
    }

    #[test]
    fn canonical_narrow_geometry_produces_checked_read_only_views() {
        let bytes = (0_i32..6).flat_map(i32::to_ne_bytes).collect::<Vec<_>>();
        let tensor = match Tensor::from_bytes(descriptor(vec![2, 3]), bytes) {
            Ok(value) => value,
            Err(error) => panic!("base tensor failed: {error}"),
        };
        let narrowed = match tensor.narrow_read_only(1, -2, 1) {
            Ok(value) => value,
            Err(error) => panic!("narrow view failed: {error}"),
        };
        assert_eq!(narrowed.storage_id(), tensor.storage_id());
        assert_eq!(narrowed.access(), ViewAccess::ReadOnly);
        assert_eq!(narrowed.descriptor().shape(), &[2, 1]);
        assert_eq!(narrowed.descriptor().strides(), &[3, 1]);
        assert_eq!(narrowed.descriptor().offset_elements(), 1);
        assert_eq!(
            narrowed.element_bytes(&[0, 0]),
            Ok(1_i32.to_ne_bytes().as_slice())
        );
        assert_eq!(
            narrowed.element_bytes(&[1, 0]),
            Ok(4_i32.to_ne_bytes().as_slice())
        );
        assert!(matches!(
            tensor.narrow_read_only(1, 2, 2),
            Err(TensorError::InvalidNarrowRange { .. })
        ));
        assert!(matches!(
            tensor.narrow_read_only(2, 0, 0),
            Err(TensorError::DimensionOutOfBounds { .. })
        ));
    }

    #[test]
    fn empty_layouts_are_non_overlapping_and_writable() {
        let descriptor = match TensorDescriptor::contiguous(
            vec![3, 0],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("empty descriptor failed: {error}"),
        };
        assert_eq!(descriptor.strides(), &[0, 1]);
        assert_eq!(descriptor.is_non_overlapping(), Ok(true));
        let mut tensor = match Tensor::from_bytes(descriptor, Vec::new()) {
            Ok(value) => value,
            Err(error) => panic!("empty writable tensor failed: {error}"),
        };
        assert!(tensor.write().is_ok());
    }

    #[test]
    fn contiguous_reinterpretation_preserves_storage_and_enforces_read_only_access() {
        let tensor = match Tensor::from_bytes(descriptor(vec![2, 2]), vec![0; 16]) {
            Ok(value) => value,
            Err(error) => panic!("base tensor failed: {error}"),
        };
        let complex_descriptor = match TensorDescriptor::contiguous(
            vec![2],
            DType::Complex64,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("complex descriptor failed: {error}"),
        };
        let mut complex = match tensor.reinterpret_contiguous_read_only(complex_descriptor) {
            Ok(value) => value,
            Err(error) => panic!("checked reinterpretation failed: {error}"),
        };
        assert_eq!(complex.storage_id(), tensor.storage_id());
        assert_eq!(complex.descriptor().dtype(), DType::Complex64);
        assert!(matches!(complex.write(), Err(TensorError::ReadOnlyView)));

        let short_descriptor = match TensorDescriptor::contiguous(
            vec![1],
            DType::Complex64,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("short descriptor failed: {error}"),
        };
        assert!(matches!(
            tensor.reinterpret_contiguous_read_only(short_descriptor),
            Err(TensorError::StorageLength {
                expected: 16,
                actual: 8,
            })
        ));

        let strided_descriptor = match TensorDescriptor::new_strided(
            vec![2, 2],
            vec![1, 2],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("strided descriptor failed: {error}"),
        };
        let strided = match tensor.view(strided_descriptor, ViewAccess::ReadOnly) {
            Ok(value) => value,
            Err(error) => panic!("strided view failed: {error}"),
        };
        let target_descriptor = match TensorDescriptor::contiguous(
            vec![2],
            DType::Complex64,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("target descriptor failed: {error}"),
        };
        assert!(matches!(
            strided.reinterpret_contiguous_read_only(target_descriptor),
            Err(TensorError::InvalidLayout { .. })
        ));
    }

    #[test]
    fn writes_are_unique_and_copy_on_write() {
        let mut first = match Tensor::from_bytes(descriptor(vec![2]), vec![0; 8]) {
            Ok(value) => value,
            Err(error) => panic!("base tensor failed: {error}"),
        };
        let second = first.clone();
        let original_storage = first.storage_id();
        {
            let mut write = match first.write() {
                Ok(value) => value,
                Err(error) => panic!("write lease failed: {error}"),
            };
            let bytes = match write.bytes_mut() {
                Ok(value) => value,
                Err(error) => panic!("mutable CPU bytes failed: {error}"),
            };
            if let Some(first_byte) = bytes.first_mut() {
                *first_byte = 9;
            }
        }
        assert_ne!(first.storage_id(), original_storage);
        assert_eq!(second.storage_id(), original_storage);
        assert_eq!(first.storage_version(), 1);
        assert_eq!(second.storage_version(), 1);
        assert!(matches!(first.contiguous_bytes(), Ok(bytes) if bytes.first() == Some(&9)));
        assert!(matches!(second.contiguous_bytes(), Ok(bytes) if bytes.first() == Some(&0)));
    }

    #[test]
    fn strided_write_leases_validate_indices_and_preserve_shared_storage() {
        let base = match Tensor::from_bytes(descriptor(vec![3]), vec![0; 12]) {
            Ok(value) => value,
            Err(error) => panic!("base tensor failed: {error}"),
        };
        let reversed = match TensorDescriptor::new_strided(
            vec![3],
            vec![-1],
            2,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("reversed descriptor failed: {error}"),
        };
        let mut view = match base.view(reversed, ViewAccess::Writable) {
            Ok(value) => value,
            Err(error) => panic!("reversed view failed: {error}"),
        };
        {
            let mut write = match view.write() {
                Ok(value) => value,
                Err(error) => panic!("strided write lease failed: {error}"),
            };
            assert!(matches!(
                write.element_bytes_mut(&[3]),
                Err(TensorError::IndexOutOfBounds { .. })
            ));
            let bytes = match write.element_bytes_mut(&[0]) {
                Ok(value) => value,
                Err(error) => panic!("strided element write failed: {error}"),
            };
            if let Some(byte) = bytes.first_mut() {
                *byte = 9;
            }
        }
        assert!(matches!(view.element_bytes(&[0]), Ok(bytes) if bytes.first() == Some(&9)));
        assert!(matches!(base.element_bytes(&[2]), Ok(bytes) if bytes.first() == Some(&0)));
        assert_ne!(view.storage_id(), base.storage_id());
        assert_eq!(view.storage_version(), 1);
    }

    #[test]
    fn descriptor_deserialization_revalidates_invariants() {
        let invalid = TensorDescriptorWire {
            shape: vec![2],
            strides: vec![],
            offset_elements: 0,
            dtype: DType::F32,
            layout: Layout::Strided,
            device: DeviceId::CPU,
            stream: StreamId::DEFAULT,
        };
        assert!(TensorDescriptor::try_from(invalid).is_err());
    }

    #[test]
    fn contiguous_semantics_ignore_singleton_strides_and_accept_empty_tensors() {
        let singleton = TensorDescriptor::new_strided(
            vec![2, 1, 3],
            vec![3, 91, 1],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )
        .expect("singleton-stride descriptor is valid");
        assert_eq!(singleton.is_contiguous(), Ok(true));

        let non_contiguous = TensorDescriptor::new_strided(
            vec![2, 1, 3],
            vec![4, 91, 1],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )
        .expect("non-contiguous singleton descriptor is valid");
        assert_eq!(non_contiguous.is_contiguous(), Ok(false));

        let empty = TensorDescriptor::new_strided(
            vec![2, 0, 3],
            vec![-17, 0, 29],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )
        .expect("empty descriptor is valid");
        assert_eq!(empty.is_contiguous(), Ok(true));
    }

    #[test]
    fn generated_manifest_is_sorted() {
        assert!(GENERATED_MODULES.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn val_tensor_001() -> Result<(), Box<dyn Error>> {
        if OPERATION_CONTRACTS.is_empty() {
            return Err(io::Error::other("tensor operation contract table is empty").into());
        }

        let contiguous_descriptor =
            TensorDescriptor::contiguous(vec![2, 3], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let mut writable_tensor = Tensor::from_bytes(contiguous_descriptor, vec![0; 24])?;
        let shared_tensor = writable_tensor.clone();
        let original_storage = writable_tensor.storage_id();
        {
            let mut write = writable_tensor.write()?;
            let Some(first_byte) = write.bytes_mut()?.first_mut() else {
                return Err(io::Error::other("validation tensor storage is empty").into());
            };
            *first_byte = 7;
        }
        let overlapping_descriptor = TensorDescriptor::new_strided(
            vec![2, 2],
            vec![1, 0],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let overlapping_write_rejected = matches!(
            shared_tensor.view(overlapping_descriptor, ViewAccess::Writable),
            Err(TensorError::OverlappingWrite)
        );
        let offset_descriptor = TensorDescriptor::new_strided(
            vec![2, 2],
            vec![2, 1],
            5,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let cpu_capabilities = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
        let compiled_resolution_ids = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .map(|resolution| resolution.operation_id)
            .collect::<HashSet<_>>();

        let mut cases = BTreeMap::new();
        cases.insert(
            "all_contract_records_are_structurally_valid",
            validate_operation_contracts(OPERATION_CONTRACTS).is_ok(),
        );
        cases.insert(
            "callable_candidates_have_owned_release_blockers",
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.inventory_kind == ContractInventoryKind::CallableOperation
                        && contract.resolution_state.is_blocked()
                        && !contract.blocker_reason.is_empty()
                        && !contract.resolution_owner_task_id.is_empty()
                        && contract.release_closure_required
                })
                .count()
                == 511,
        );
        cases.insert(
            "reference_rows_are_typed_contracts",
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.typed_reference().is_some()
                        && !contract.resolution_owner_task_id.is_empty()
                        && !contract.release_closure_required
                })
                .count()
                == 82,
        );
        cases.insert(
            "receiver_candidates_are_explicitly_classified",
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.resolution_state == ContractResolutionState::BlockedReceiverUnverified
                        && !contract.blocker_reason.is_empty()
                })
                .count()
                == 94,
        );
        cases.insert("contract_fixture_identities_are_per_row", {
            let fixture_names = OPERATION_CONTRACTS
                .iter()
                .map(|contract| contract.oracle_fixture)
                .collect::<HashSet<_>>();
            let fixture_digests = OPERATION_CONTRACTS
                .iter()
                .map(|contract| contract.oracle_fixture_sha256)
                .collect::<HashSet<_>>();
            fixture_names.len() == OPERATION_CONTRACTS.len()
                && fixture_digests.len() == OPERATION_CONTRACTS.len()
        });
        cases.insert(
            "production_id_issuance_is_resolution_gated",
            OPERATION_CONTRACTS
                .iter()
                .all(|contract| match contract.inventory_kind {
                    ContractInventoryKind::CallableOperation => {
                        OperationContractId::cataloged(contract.operation_id).is_ok()
                            == compiled_resolution_ids.contains(contract.operation_id)
                            && OperationContractId::cataloged(contract.overload_id).is_err()
                    }
                    ContractInventoryKind::ReclassifiedExternalOperation
                    | ContractInventoryKind::NamespaceValueReference
                    | ContractInventoryKind::TypeReference => {
                        OperationContractId::cataloged(contract.operation_id).is_err()
                            && OperationContractId::cataloged(contract.overload_id).is_err()
                    }
                })
                && GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                    .iter()
                    .flat_map(|slice| slice.iter())
                    .all(|resolution| {
                        OperationContractId::cataloged(resolution.overload_id).is_ok()
                    }),
        );
        cases.insert(
            "native_primitive_dispatch_cannot_forge_catalog_ids",
            OperationContractId::new("zed.native-internal.forged").is_err(),
        );
        cases.insert(
            "copy_on_write_preserves_logical_identity_and_mutation_lineage",
            writable_tensor.storage_id() != original_storage
                && shared_tensor.storage_id() == original_storage
                && writable_tensor.tensor_id() == shared_tensor.tensor_id()
                && writable_tensor.storage_version() == 1
                && shared_tensor.storage_version() == 1
                && matches!(writable_tensor.contiguous_bytes(), Ok(bytes) if bytes.first() == Some(&7))
                && matches!(shared_tensor.contiguous_bytes(), Ok(bytes) if bytes.first() == Some(&0)),
        );
        cases.insert(
            "overlapping_writable_views_are_rejected",
            overlapping_write_rejected,
        );
        cases.insert(
            "tensor_descriptor_owns_backing_span_validation",
            offset_descriptor.minimum_backing_byte_length() == Ok(36)
                && offset_descriptor.validate_backing_byte_length(36).is_ok()
                && matches!(
                    offset_descriptor.validate_backing_byte_length(35),
                    Err(TensorError::StorageBounds {
                        required: 36,
                        actual: 35,
                    })
                ),
        );
        cases.insert(
            "native_cpu_capability_matrix_is_registered",
            cpu_capabilities.device() == DeviceId::CPU,
        );
        cases.insert(
            "native_cpu_bmm_is_capability_owned",
            [Layout::Contiguous, Layout::Strided]
                .into_iter()
                .all(|layout| {
                    cpu_capabilities.supports(OperationSupport::linear_algebra_input(
                        LinearAlgebraOperation::BatchMatrixMultiply,
                        DType::F32,
                        layout,
                    )) && cpu_capabilities.supports(OperationSupport::linear_algebra_output(
                        LinearAlgebraOperation::BatchMatrixMultiply,
                        DType::F32,
                        layout,
                    ))
                }),
        );
        cases.insert(
            "unsupported_cpu_breadth_is_not_advertised",
            [
                PrimitiveOperation::Gather,
                PrimitiveOperation::Scatter,
                PrimitiveOperation::MaskedSelect,
                PrimitiveOperation::CustomKernel,
            ]
            .into_iter()
            .all(|primitive| !cpu_capabilities.supports_primitive(primitive)),
        );
        cases.insert(
            "native_cpu_convolution_dtypes_and_layouts_are_capability_owned",
            [DType::F32, DType::F16, DType::Bf16]
                .into_iter()
                .all(|dtype| {
                    [
                        Layout::Contiguous,
                        Layout::ChannelsLast,
                        Layout::ChannelsLast3d,
                        Layout::Strided,
                    ]
                    .into_iter()
                    .all(|layout| {
                        cpu_capabilities
                            .supports(OperationSupport::convolution_input(dtype, layout))
                            && cpu_capabilities
                                .supports(OperationSupport::convolution_output(dtype, layout))
                    })
                }),
        );

        let fixture_path = "crates/sim_test_support/parity/catalogs/backend-tensor-operations.csv";
        let fixture_digest = validation_artifacts::workspace_fixture_digest(
            fixture_path,
            "e60844bdf259418b8f808f1122408edaeac11bffdf03a65d205a79b251e9911b",
        )?;
        let resize_fixture_path =
            "crates/sim_test_support/fixtures/tensor_operations/image_resize_foundation.json";
        let resize_fixture_digest = validation_artifacts::workspace_fixture_digest(
            resize_fixture_path,
            "869ff9c4dcd537c6fd9df3e8eae08bba11f9ca3a8740aa61aa8cb4f95dd1b8a2",
        )?;
        let resize_fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../sim_test_support/fixtures/tensor_operations/image_resize_foundation.json"
        ))?;
        cases.insert(
            "checked_in_sim_resize_oracle_is_complete",
            resize_fixture
                .get("cases")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|cases| cases.len() == 11)
                && resize_fixture
                    .get("classification")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|classification| {
                        classification.contains("development-time ComfyUI conformance oracle")
                            && classification.contains("production Rust must not import")
                    }),
        );
        cases.extend(crate::image_ops::tests::checked_in_resize_oracle_case_results()?);
        let verified_contract_fixtures = OPERATION_CONTRACTS
            .iter()
            .map(|contract| {
                let fixture_path = format!(
                    "crates/sim_test_support/fixtures/tensor_signatures/contracts/{}.json",
                    contract.operation_id.to_ascii_lowercase()
                );
                let digest = validation_artifacts::workspace_contract_fixture_digest(
                    &fixture_path,
                    contract.oracle_fixture_sha256,
                    contract.oracle_fixture,
                    contract.operation_id,
                )?;
                Ok::<_, Box<dyn Error>>((fixture_path, digest))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let verified_resolution_fixtures = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .map(|resolution| {
                let digest = validation_artifacts::workspace_fixture_digest(
                    resolution.evidence_fixture,
                    resolution.evidence_fixture_sha256,
                )?;
                Ok::<_, Box<dyn Error>>((resolution.evidence_fixture, digest))
            })
            .collect::<Result<Vec<_>, _>>()?;
        cases.insert(
            "per_row_fixture_files_are_runtime_verified",
            verified_contract_fixtures.len() == OPERATION_CONTRACTS.len(),
        );
        cases.insert(
            "compiled_resolution_evidence_files_are_runtime_verified",
            verified_resolution_fixtures.len()
                == GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                    .iter()
                    .map(|slice| slice.len())
                    .sum::<usize>(),
        );
        let mut fixture_digests = BTreeMap::from([
            (fixture_path, fixture_digest.as_str()),
            (resize_fixture_path, resize_fixture_digest.as_str()),
        ]);
        for (fixture_path, digest) in &verified_contract_fixtures {
            fixture_digests.insert(fixture_path.as_str(), digest.as_str());
        }
        for (fixture_path, digest) in &verified_resolution_fixtures {
            fixture_digests.insert(*fixture_path, digest.as_str());
        }
        let mut remaining_release_gates = OPERATION_CONTRACTS
            .iter()
            .filter(|contract| {
                contract.release_closure_required
                    && !compiled_resolution_ids.contains(contract.operation_id)
            })
            .map(|contract| contract.resolution_owner_task_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        remaining_release_gates.extend([
            "comfy-parity-native-autograd-breadth",
            "comfy-parity-native-rng-breadth",
            "comfy-parity-final-validation",
        ]);
        remaining_release_gates.sort_unstable();
        remaining_release_gates.dedup();
        validation_artifacts::write(
            "val-tensor-001.json",
            "VAL-TENSOR-001",
            "Task 7 checked tensor domain and operation-contract classification foundation",
            "task-7-tensor-foundation",
            &fixture_digests,
            &cases,
            &remaining_release_gates,
        )
    }
}
