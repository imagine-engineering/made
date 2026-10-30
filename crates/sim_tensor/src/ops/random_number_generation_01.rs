use crate::{
    BrownianTree, CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, NumericClass,
    RngAlgorithm, RngError, RngProfileVersion, RngStream, RngStreamAddress, RngTransaction, Scalar,
    SobolEngine, Tensor, TensorDescriptor, TensorError,
};
use thiserror::Error;

pub const GENERATOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-CB163DF58130";
pub const GENERATOR_MANUAL_SEED_OPERATION_ID: &str = "COMFY-TENSOR-OP-82073E6E054A";
pub const MANUAL_SEED_OPERATION_ID: &str = "COMFY-TENSOR-OP-EA3150484FE4";
pub const MULTINOMIAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-46213648166E";
pub const NORMAL_INITIALIZER_OPERATION_ID: &str = "COMFY-TENSOR-OP-3754E9847F8D";
pub const UNIFORM_INITIALIZER_OPERATION_ID: &str = "COMFY-TENSOR-OP-4A9703265641";
pub const SOBOL_ENGINE_OPERATION_ID: &str = "COMFY-TENSOR-OP-7745541008DF";
pub const RAND_OPERATION_ID: &str = "COMFY-TENSOR-OP-19CD261729FC";
pub const RANDINT_OPERATION_ID: &str = "COMFY-TENSOR-OP-095B3E192800";
pub const RANDN_LIKE_OPERATION_ID: &str = "COMFY-TENSOR-OP-59B309167618";
pub const RANDPERM_OPERATION_ID: &str = "COMFY-TENSOR-OP-271D62B3F4FF";
pub const BROWNIAN_TREE_OPERATION_ID: &str = "COMFY-TENSOR-OP-E0A9E435AF72";

#[derive(Debug, Error)]
pub enum RandomNumberGenerationPartOneError {
    #[error("random-number-generation part-one operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Rng(RngError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("operation {operation} supports only CPU, not {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} does not support dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("allocation failed while preparing {0}")]
    AllocationFailed(&'static str),
}

impl From<sim_types::CancellationError> for RandomNumberGenerationPartOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<RngError> for RandomNumberGenerationPartOneError {
    fn from(error: RngError) -> Self {
        match error {
            RngError::Cancelled => Self::Cancelled,
            error => Self::Rng(error),
        }
    }
}

pub struct RandomTensorForward {
    pub tensor: Tensor,
    pub transaction: RngTransaction,
}

pub struct MultinomialForward {
    pub indices: Vec<i64>,
    pub shape: Vec<usize>,
    pub transaction: RngTransaction,
}

pub fn generator_exact_native(
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    seed: u64,
    address: RngStreamAddress,
    cancellation: &CancellationToken,
) -> Result<RngStream, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    Ok(RngStream::new(profile, algorithm, seed, address)?)
}

pub fn generator_manual_seed_exact_native(
    generator: &RngStream,
    seed: i128,
    cancellation: &CancellationToken,
) -> Result<RngStream, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    Ok(generator.reseed(normalize_seed(seed, GENERATOR_MANUAL_SEED_OPERATION_ID)?)?)
}

pub fn manual_seed_exact_native(
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    seed: i128,
    address: RngStreamAddress,
    cancellation: &CancellationToken,
) -> Result<RngStream, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    generator_exact_native(
        profile,
        algorithm,
        normalize_seed(seed, MANUAL_SEED_OPERATION_ID)?,
        address,
        cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn multinomial_with_context_exact_native(
    weights: &[f64],
    rows: usize,
    columns: usize,
    sample_count: usize,
    replacement: bool,
    mut transaction: RngTransaction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<MultinomialForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    require_cpu(device, MULTINOMIAL_OPERATION_ID)?;
    transaction.require_device(device)?;
    let expected = rows
        .checked_mul(columns)
        .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
            "multinomial weights",
        ))?;
    if rows == 0 || columns == 0 || weights.len() != expected || sample_count == 0 {
        return invalid(
            MULTINOMIAL_OPERATION_ID,
            "weights must be a nonempty row-major vector or matrix and sample_count must be positive",
        );
    }
    let output_len = rows
        .checked_mul(sample_count)
        .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
            "multinomial output",
        ))?;
    let mut output = allocate(output_len, "multinomial output")?;
    for row_index in 0..rows {
        let row_start = row_index
            .checked_mul(columns)
            .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
                "multinomial row",
            ))?;
        let row_end = row_start
            .checked_add(columns)
            .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
                "multinomial row",
            ))?;
        let row = weights.get(row_start..row_end).ok_or(
            RandomNumberGenerationPartOneError::ShapeOverflow("multinomial row"),
        )?;
        validate_weight_row(row, sample_count, replacement)?;
        let mut available = row.to_vec();
        for sample_index in 0..sample_count {
            check_periodically(sample_index, context)?;
            let sum = available.iter().sum::<f64>();
            let target = transaction.next_unit_f64(context.cancellation)? * sum;
            let selected = select_weighted_index(&available, target)?;
            output.push(i64::try_from(selected).map_err(|_| {
                RandomNumberGenerationPartOneError::ShapeOverflow("multinomial index")
            })?);
            if !replacement {
                let selected_weight = available.get_mut(selected).ok_or(
                    RandomNumberGenerationPartOneError::ShapeOverflow(
                        "multinomial selected index",
                    ),
                )?;
                *selected_weight = 0.0;
            }
        }
    }
    context.cancellation.check()?;
    Ok(MultinomialForward {
        indices: output,
        shape: if rows == 1 {
            vec![sample_count]
        } else {
            vec![rows, sample_count]
        },
        transaction,
    })
}

pub fn normal_in_place_exact_native(
    input: &mut Tensor,
    mean: f64,
    standard_deviation: f64,
    mut transaction: RngTransaction,
    cancellation: &CancellationToken,
) -> Result<RngTransaction, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    require_writable_floating_tensor(input, NORMAL_INITIALIZER_OPERATION_ID)?;
    transaction.require_device(input.descriptor().device())?;
    if !mean.is_finite() || !standard_deviation.is_finite() || standard_deviation < 0.0 {
        return invalid(
            NORMAL_INITIALIZER_OPERATION_ID,
            "mean and nonnegative standard_deviation must be finite",
        );
    }
    let mut candidate = input.clone();
    let shape = candidate.descriptor().shape().to_vec();
    let dtype = candidate.descriptor().dtype();
    let count = element_count(&shape, "normal initializer")?;
    let mut pair = [0.0; 2];
    let mut pair_index = 2;
    {
        let mut write = candidate.write()?;
        for linear_index in 0..count {
            if linear_index % 1_024 == 0 {
                cancellation.check()?;
            }
            if pair_index == 2 {
                pair = transaction.next_standard_normal_pair(cancellation)?;
                pair_index = 0;
            }
            let normal = pair
                .get(pair_index)
                .copied()
                .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
                    "normal pair",
                ))?;
            pair_index += 1;
            let bytes = dtype.encode_scalar(
                Scalar::Float(normal.mul_add(standard_deviation, mean)),
                NORMAL_INITIALIZER_OPERATION_ID,
                DeviceId::CPU,
            )?;
            let indices = unravel_index(linear_index, &shape)?;
            write.element_bytes_mut(&indices)?.copy_from_slice(&bytes);
        }
    }
    cancellation.check()?;
    input.commit_in_place(candidate)?;
    Ok(transaction)
}

pub fn uniform_in_place_exact_native(
    input: &mut Tensor,
    lower: f64,
    upper: f64,
    mut transaction: RngTransaction,
    cancellation: &CancellationToken,
) -> Result<RngTransaction, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    require_writable_floating_tensor(input, UNIFORM_INITIALIZER_OPERATION_ID)?;
    transaction.require_device(input.descriptor().device())?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return invalid(
            UNIFORM_INITIALIZER_OPERATION_ID,
            "finite bounds must satisfy lower <= upper",
        );
    }
    let mut candidate = input.clone();
    let shape = candidate.descriptor().shape().to_vec();
    let dtype = candidate.descriptor().dtype();
    let count = element_count(&shape, "uniform initializer")?;
    {
        let mut write = candidate.write()?;
        for linear_index in 0..count {
            if linear_index % 1_024 == 0 {
                cancellation.check()?;
            }
            let unit = transaction.next_unit_f64(cancellation)?;
            let bytes = dtype.encode_scalar(
                Scalar::Float(unit.mul_add(upper - lower, lower)),
                UNIFORM_INITIALIZER_OPERATION_ID,
                DeviceId::CPU,
            )?;
            let indices = unravel_index(linear_index, &shape)?;
            write.element_bytes_mut(&indices)?.copy_from_slice(&bytes);
        }
    }
    cancellation.check()?;
    input.commit_in_place(candidate)?;
    Ok(transaction)
}

pub fn sobol_engine_exact_native(
    dimension: usize,
    scramble: bool,
    seed: u64,
    cancellation: &CancellationToken,
) -> Result<SobolEngine, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    Ok(SobolEngine::new(dimension, scramble, seed)?)
}

pub fn sobol_draw_with_context_exact_native(
    backend: &CpuBackend,
    mut engine: SobolEngine,
    count: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, SobolEngine), RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    let dimension = engine.dimension();
    let values = engine.draw(count, context.cancellation)?;
    let tensor = upload_f32(
        backend,
        &[
            u64::try_from(count).map_err(|_| {
                RandomNumberGenerationPartOneError::ShapeOverflow("Sobol draw count")
            })?,
            u64::try_from(dimension).map_err(|_| {
                RandomNumberGenerationPartOneError::ShapeOverflow("Sobol dimension")
            })?,
        ],
        &values,
        context,
    )?;
    Ok((tensor, engine))
}

pub fn rand_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    mut transaction: RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    transaction.require_device(DeviceId::CPU)?;
    require_floating_dtype(dtype, RAND_OPERATION_ID)?;
    let count = element_count(shape, "rand output")?;
    let mut values = allocate(count, "rand output")?;
    for index in 0..count {
        check_periodically(index, context)?;
        values.push(transaction.next_unit_f64(context.cancellation)?);
    }
    let tensor = upload_real(backend, shape, dtype, &values, RAND_OPERATION_ID, context)?;
    Ok(RandomTensorForward {
        tensor,
        transaction,
    })
}

pub fn randint_with_context_exact_native(
    backend: &CpuBackend,
    low: i64,
    high: i64,
    shape: &[u64],
    mut transaction: RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    transaction.require_device(DeviceId::CPU)?;
    let width = high.checked_sub(low).ok_or_else(|| {
        RandomNumberGenerationPartOneError::Invalid {
            operation: RANDINT_OPERATION_ID,
            reason: "high - low overflows i64".to_owned(),
        }
    })?;
    let width = u64::try_from(width).map_err(|_| RandomNumberGenerationPartOneError::Invalid {
        operation: RANDINT_OPERATION_ID,
        reason: "low must be strictly less than high".to_owned(),
    })?;
    if width == 0 {
        return invalid(RANDINT_OPERATION_ID, "low must be strictly less than high");
    }
    let count = element_count(shape, "randint output")?;
    let byte_count = count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
            "randint bytes",
        ))?;
    let mut bytes = allocate(byte_count, "randint output")?;
    for index in 0..count {
        check_periodically(index, context)?;
        let offset = transaction.next_bounded_u64(width, context.cancellation)?;
        let offset = i64::try_from(offset).map_err(|_| {
            RandomNumberGenerationPartOneError::ShapeOverflow("randint value")
        })?;
        let value = low.checked_add(offset).ok_or(
            RandomNumberGenerationPartOneError::ShapeOverflow("randint value"),
        )?;
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::I64,
        DeviceId::CPU,
        context.stream,
    )?;
    let tensor = backend.upload_bytes(descriptor, &bytes, context)?.0;
    Ok(RandomTensorForward {
        tensor,
        transaction,
    })
}

pub fn randn_like_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    transaction: RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    require_cpu(input.descriptor().device(), RANDN_LIKE_OPERATION_ID)?;
    transaction.require_device(input.descriptor().device())?;
    let dtype = input.descriptor().dtype();
    require_floating_dtype(dtype, RANDN_LIKE_OPERATION_ID)?;
    if input.descriptor().stream() != context.stream {
        return invalid(
            RANDN_LIKE_OPERATION_ID,
            "input stream must match the execution context",
        );
    }
    standard_normal_tensor_with_context(
        backend,
        input.descriptor().shape(),
        dtype,
        transaction,
        RANDN_LIKE_OPERATION_ID,
        context,
    )
}

pub(crate) fn standard_normal_tensor_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    mut transaction: RngTransaction,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    transaction.require_device(DeviceId::CPU)?;
    require_floating_dtype(dtype, operation)?;
    let count = element_count(shape, "standard-normal output")?;
    let mut values = allocate(count, "standard-normal output")?;
    while values.len() < count {
        context.cancellation.check()?;
        let pair = transaction.next_standard_normal_pair(context.cancellation)?;
        values.push(pair[0]);
        if values.len() < count {
            values.push(pair[1]);
        }
    }
    let tensor = upload_real(
        backend,
        shape,
        dtype,
        &values,
        operation,
        context,
    )?;
    Ok(RandomTensorForward {
        tensor,
        transaction,
    })
}

pub fn randperm_with_context_exact_native(
    backend: &CpuBackend,
    count: u64,
    mut transaction: RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartOneError> {
    context.cancellation.check()?;
    transaction.require_device(DeviceId::CPU)?;
    let count_usize = usize::try_from(count).map_err(|_| {
        RandomNumberGenerationPartOneError::ShapeOverflow("randperm element count")
    })?;
    if count > i64::MAX as u64 {
        return invalid(RANDPERM_OPERATION_ID, "count exceeds the I64 output range");
    }
    let mut values = allocate(count_usize, "randperm output")?;
    for value in 0..count {
        values.push(i64::try_from(value).map_err(|_| {
            RandomNumberGenerationPartOneError::ShapeOverflow("randperm value")
        })?);
    }
    for upper in (1..count_usize).rev() {
        check_periodically(count_usize - upper, context)?;
        let selected = transaction.next_bounded_u64(
            u64::try_from(upper + 1).map_err(|_| {
                RandomNumberGenerationPartOneError::ShapeOverflow("randperm bound")
            })?,
            context.cancellation,
        )?;
        let selected = usize::try_from(selected).map_err(|_| {
            RandomNumberGenerationPartOneError::ShapeOverflow("randperm selected index")
        })?;
        values.swap(upper, selected);
    }
    context.cancellation.check()?;
    let byte_count = count_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
            "randperm bytes",
        ))?;
    let mut bytes = allocate(byte_count, "randperm bytes")?;
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![count],
        DType::I64,
        DeviceId::CPU,
        context.stream,
    )?;
    let tensor = backend.upload_bytes(descriptor, &bytes, context)?.0;
    Ok(RandomTensorForward {
        tensor,
        transaction,
    })
}

pub fn brownian_tree_exact_native(
    start: f64,
    initial: Vec<f64>,
    end: f64,
    entropy: u64,
    cancellation: &CancellationToken,
) -> Result<BrownianTree, RandomNumberGenerationPartOneError> {
    cancellation.check()?;
    Ok(BrownianTree::new(start, initial, end, entropy)?)
}

fn normalize_seed(
    seed: i128,
    operation: &'static str,
) -> Result<u64, RandomNumberGenerationPartOneError> {
    if seed < i128::from(i64::MIN) || seed > i128::from(u64::MAX) {
        return invalid(
            operation,
            "seed must be in PyTorch's inclusive signed-to-unsigned 64-bit range",
        );
    }
    if seed < 0 {
        let remapped = i128::from(u64::MAX)
            .checked_add(seed)
            .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
                "negative seed remapping",
            ))?;
        u64::try_from(remapped).map_err(|_| RandomNumberGenerationPartOneError::Invalid {
            operation,
            reason: "negative seed remapping failed".to_owned(),
        })
    } else {
        u64::try_from(seed).map_err(|_| RandomNumberGenerationPartOneError::Invalid {
            operation,
            reason: "seed exceeds the unsigned 64-bit range".to_owned(),
        })
    }
}

fn validate_weight_row(
    row: &[f64],
    sample_count: usize,
    replacement: bool,
) -> Result<(), RandomNumberGenerationPartOneError> {
    if row
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || row.iter().all(|weight| *weight == 0.0)
    {
        return invalid(
            MULTINOMIAL_OPERATION_ID,
            "each row must contain finite nonnegative weights with a positive sum",
        );
    }
    let nonzero = row.iter().filter(|weight| **weight > 0.0).count();
    if !replacement && sample_count > nonzero {
        return invalid(
            MULTINOMIAL_OPERATION_ID,
            "sampling without replacement cannot exceed the positive-weight count",
        );
    }
    Ok(())
}

fn select_weighted_index(
    weights: &[f64],
    target: f64,
) -> Result<usize, RandomNumberGenerationPartOneError> {
    let mut cumulative = 0.0;
    let mut fallback = None;
    for (index, weight) in weights.iter().copied().enumerate() {
        if weight > 0.0 {
            fallback = Some(index);
            cumulative += weight;
            if target < cumulative {
                return Ok(index);
            }
        }
    }
    fallback.ok_or_else(|| RandomNumberGenerationPartOneError::Invalid {
        operation: MULTINOMIAL_OPERATION_ID,
        reason: "weight row has no selectable value".to_owned(),
    })
}

fn require_writable_floating_tensor(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), RandomNumberGenerationPartOneError> {
    require_cpu(input.descriptor().device(), operation)?;
    require_floating_dtype(input.descriptor().dtype(), operation)
}

fn require_floating_dtype(
    dtype: DType,
    operation: &'static str,
) -> Result<(), RandomNumberGenerationPartOneError> {
    if dtype.class() != NumericClass::FloatingPoint {
        return Err(RandomNumberGenerationPartOneError::UnsupportedDType { operation, dtype });
    }
    Ok(())
}

fn require_cpu(
    device: DeviceId,
    operation: &'static str,
) -> Result<(), RandomNumberGenerationPartOneError> {
    if device != DeviceId::CPU {
        return Err(RandomNumberGenerationPartOneError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn element_count(
    shape: &[u64],
    label: &'static str,
) -> Result<usize, RandomNumberGenerationPartOneError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(label))
    })?;
    usize::try_from(count).map_err(|_| RandomNumberGenerationPartOneError::ShapeOverflow(label))
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, RandomNumberGenerationPartOneError> {
    let mut indices = allocate(shape.len(), "tensor indices")?;
    indices.resize(shape.len(), 0);
    for dimension_index in (0..shape.len()).rev() {
        let dimension = shape.get(dimension_index).copied().ok_or(
            RandomNumberGenerationPartOneError::ShapeOverflow("tensor dimension"),
        )?;
        let dimension = usize::try_from(dimension).map_err(|_| {
            RandomNumberGenerationPartOneError::ShapeOverflow("tensor dimension")
        })?;
        if dimension == 0 {
            return Err(RandomNumberGenerationPartOneError::ShapeOverflow(
                "zero-sized tensor index",
            ));
        }
        let coordinate = linear_index % dimension;
        linear_index /= dimension;
        let destination = indices.get_mut(dimension_index).ok_or(
            RandomNumberGenerationPartOneError::ShapeOverflow("tensor index"),
        )?;
        *destination = u64::try_from(coordinate).map_err(|_| {
            RandomNumberGenerationPartOneError::ShapeOverflow("tensor coordinate")
        })?;
    }
    Ok(indices)
}

fn upload_real(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    values: &[f64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, RandomNumberGenerationPartOneError> {
    require_floating_dtype(dtype, operation)?;
    let byte_count = values
        .len()
        .checked_mul(dtype.byte_width() as usize)
        .ok_or(RandomNumberGenerationPartOneError::ShapeOverflow(
            "random tensor bytes",
        ))?;
    let mut bytes = allocate(byte_count, "random tensor bytes")?;
    for (index, value) in values.iter().copied().enumerate() {
        check_periodically(index, context)?;
        bytes.extend_from_slice(&dtype.encode_scalar(
            Scalar::Float(value),
            operation,
            DeviceId::CPU,
        )?);
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, RandomNumberGenerationPartOneError> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn allocate<T>(
    capacity: usize,
    label: &'static str,
) -> Result<Vec<T>, RandomNumberGenerationPartOneError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| RandomNumberGenerationPartOneError::AllocationFailed(label))?;
    Ok(output)
}

fn check_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), RandomNumberGenerationPartOneError> {
    if index.is_multiple_of(1_024) {
        context.cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, RandomNumberGenerationPartOneError> {
    Err(RandomNumberGenerationPartOneError::Invalid {
        operation,
        reason: reason.into(),
    })
}
